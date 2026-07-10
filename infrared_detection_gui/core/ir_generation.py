"""
可见光 → 红外图像生成算法接口

算法一 (MIIGAN): 基于 Mamba 的 GAN 架构，使用 U-Net 生成器
算法二 (PhysMamba): 基于 PhysMamba 测试流程的可见光到红外生成

GUI 传入单张 RGB numpy 图像，这里负责调用对应算法接口并返回 RGB numpy 图像。
"""

import contextlib
import io
import os
import sys
import time
import cv2
import numpy as np


def _resource_root():
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


WORKSPACE_ROOT = _resource_root()
MIIGAN_ROOT = os.path.join(WORKSPACE_ROOT, "miigan-master")
PHYSMAMBA_ROOT = os.path.join(WORKSPACE_ROOT, "PhysMamba")
DEFAULT_WEIGHT_ROOT = os.path.join(WORKSPACE_ROOT, "weight")
DEFAULT_PHYSMAMBA_WEIGHT = os.path.join(DEFAULT_WEIGHT_ROOT, "PhysMamba", "latest_net_G.pth")
MIIGAN_DEFAULT_INPUT_SIZE = 512
PHYSMAMBA_DEFAULT_INPUT_SIZE = 512
DEFAULT_DEVICE_PREFERENCE = "cpu"

_MIIGAN_MODEL_CACHE = {}
_PHYSMAMBA_MODEL_CACHE = {}
_ALGORITHM_MODULE_PREFIXES = ("options", "models", "util", "data", "ssim", "lpips")


def _normalize_device_preference(device_preference=None):
    device = (device_preference or DEFAULT_DEVICE_PREFERENCE).strip().lower()
    if device in ("cuda", "gpu", "cuda:0"):
        return "cuda"
    return "cpu"


def _resolve_optional_torch_device(torch_module, device_preference=None):
    device = _normalize_device_preference(device_preference)
    if device == "cuda" and torch_module.cuda.is_available():
        return torch_module.device("cuda:0"), [0], "cuda"
    return torch_module.device("cpu"), [], "cpu"


def _find_checkpoint(model_path=None, algorithm_dir=None, extra_roots=None):
    """查找生成器权重文件 *_net_G.pth。"""
    search_roots = []
    if model_path:
        model_path = os.path.abspath(model_path)
        if os.path.isfile(model_path):
            return model_path
        search_roots.append(model_path)

    if algorithm_dir:
        search_roots.append(os.path.join(DEFAULT_WEIGHT_ROOT, algorithm_dir))
    search_roots.extend([
        DEFAULT_WEIGHT_ROOT,
    ])
    if extra_roots:
        search_roots.extend(extra_roots)

    preferred_names = ("latest_net_G.pth", "200_net_G.pth")
    found = []
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        if os.path.abspath(root) == os.path.abspath(DEFAULT_WEIGHT_ROOT):
            walk_items = [(root, [], os.listdir(root))]
        else:
            walk_items = os.walk(root)

        for dirpath, _, filenames in walk_items:
            for preferred in preferred_names:
                if preferred in filenames:
                    return os.path.join(dirpath, preferred)
            for filename in filenames:
                if filename.endswith("_net_G.pth"):
                    found.append(os.path.join(dirpath, filename))

    if found:
        found.sort(key=lambda path: os.path.getmtime(path), reverse=True)
        return found[0]
    return None


def _find_miigan_checkpoint(model_path=None):
    """查找 MIIGAN 的生成器权重文件 *_net_G.pth。"""
    return _find_checkpoint(
        model_path,
        algorithm_dir="miigan",
        extra_roots=[os.path.join(MIIGAN_ROOT, "checkpoints")],
    )


def _find_physmamba_checkpoint(model_path=None):
    """返回 PhysMamba 固定权重路径，或校验调用方显式传入的权重路径。"""
    checkpoint_path = os.path.abspath(model_path or DEFAULT_PHYSMAMBA_WEIGHT)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "未找到 PhysMamba 生成器权重。"
            f"请确认权重文件存在: {checkpoint_path}"
        )
    return checkpoint_path


def _checkpoint_to_option_parts(checkpoint_path):
    """
    将 F:/.../<checkpoints_dir>/<name>/<epoch>_net_G.pth
    解析成算法仓库 BaseModel.load_network 需要的三段参数。
    """
    filename = os.path.basename(checkpoint_path)
    if not filename.endswith("_net_G.pth"):
        raise ValueError(f"权重文件名必须形如 200_net_G.pth: {checkpoint_path}")

    epoch = filename[:-len("_net_G.pth")]
    experiment_dir = os.path.dirname(checkpoint_path)
    checkpoints_dir = os.path.dirname(experiment_dir)
    experiment_name = os.path.basename(experiment_dir)
    return checkpoints_dir, experiment_name, epoch


def _ensure_algorithm_root(root, label):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"未找到 {label} 目录: {root}")


@contextlib.contextmanager
def _algorithm_import_context(root):
    """
    隔离外部算法仓库的同名顶层包。

    算法仓库可能都有 options、models、util、data 等包名；加载某个算法时
    临时清理这些模块并把对应仓库放到 sys.path 首位，避免串包。
    """
    saved_path = sys.path[:]
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name.split(".", 1)[0] in _ALGORITHM_MODULE_PREFIXES
    }
    for name in list(saved_modules):
        sys.modules.pop(name, None)

    sys.path = [root] + [path for path in sys.path if os.path.abspath(path) != root]
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name.split(".", 1)[0] in _ALGORITHM_MODULE_PREFIXES:
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path = saved_path


@contextlib.contextmanager
def _temporary_argv(argv):
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old_argv


@contextlib.contextmanager
def _cpu_safe_torch_load(torch_module, force_cpu=False):
    """让 CPU 环境也能加载保存自 CUDA 的权重。"""
    original_load = torch_module.load

    def safe_load(*args, **kwargs):
        if (force_cpu or not torch_module.cuda.is_available()) and "map_location" not in kwargs:
            kwargs["map_location"] = "cpu"
        return original_load(*args, **kwargs)

    torch_module.load = safe_load
    try:
        yield
    finally:
        torch_module.load = original_load


def _build_physmamba_options(checkpoint_path, torch_module, device_preference=None):
    from options.test_options import TestOptions

    checkpoints_dir, experiment_name, epoch = _checkpoint_to_option_parts(checkpoint_path)
    _, gpu_id_list, device_name = _resolve_optional_torch_device(torch_module, device_preference)
    gpu_ids = "0" if device_name == "cuda" else "-1"
    argv = [
        "physmamba_gui_inference",
        "--model", "physmamba",
        "--dataset_mode", "M3FD",
        "--dataroot", DEFAULT_WEIGHT_ROOT,
        "--checkpoints_dir", checkpoints_dir,
        "--name", experiment_name,
        "--which_epoch", epoch,
        "--batchSize", "1",
        "--nThreads", "1",
        "--serial_batches",
        "--no_flip",
        "--input_nc", "3",
        "--output_nc", "1",
        "--which_model_netG", "unet_512",
        "--loadSize", str(PHYSMAMBA_DEFAULT_INPUT_SIZE),
        "--fineSize", str(PHYSMAMBA_DEFAULT_INPUT_SIZE),
        "--resize_or_crop", "resize_and_crop",
        "--gpu_ids", gpu_ids,
        "--phase", "test",
        "--how_many", "1",
    ]

    with _temporary_argv(argv):
        # TestOptions 会打印配置并写 opt.txt；这里静默掉控制台噪声。
        with contextlib.redirect_stdout(io.StringIO()):
            opt = TestOptions().parse()

    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.gpu_ids = gpu_id_list
    opt.inference_device = device_name
    opt.weight_path = checkpoint_path
    return opt


def _load_physmamba_model(checkpoint_path, device_preference=None):
    checkpoint_path = os.path.abspath(checkpoint_path)
    requested_device = _normalize_device_preference(device_preference)
    cache_key = (checkpoint_path, requested_device)
    if cache_key in _PHYSMAMBA_MODEL_CACHE:
        return _PHYSMAMBA_MODEL_CACHE[cache_key]

    _ensure_algorithm_root(PHYSMAMBA_ROOT, "PhysMamba")

    with _algorithm_import_context(PHYSMAMBA_ROOT):
        import torch
        from models.models import create_model

        opt = _build_physmamba_options(checkpoint_path, torch, requested_device)
        device = torch.device("cuda:0" if opt.inference_device == "cuda" else "cpu")
        with _cpu_safe_torch_load(torch, force_cpu=opt.inference_device == "cpu"):
            with contextlib.redirect_stdout(io.StringIO()):
                model = create_model(opt)

    if hasattr(model, "netG"):
        model.netG.to(device)
        model.netG.eval()

    _PHYSMAMBA_MODEL_CACHE[cache_key] = (model, opt, torch)
    return _PHYSMAMBA_MODEL_CACHE[cache_key]


def _prepare_physmamba_input(input_image, opt, torch_module):
    from PIL import Image, ImageOps
    import torchvision.transforms as transforms

    if input_image is None:
        raise ValueError("输入图像为空，请先选择一张可见光图像。")

    image = np.asarray(input_image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"输入图像格式不正确，期望 RGB 图像，实际形状: {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    pil_rgb = Image.fromarray(image[:, :, :3]).convert("RGB")
    pil_rgb = pil_rgb.resize((opt.loadSize, opt.loadSize), Image.BICUBIC)
    pil_gray = ImageOps.grayscale(pil_rgb)

    a_tensor = transforms.ToTensor()(pil_rgb.copy()).float()
    b_tensor = transforms.ToTensor()(pil_gray.copy()).float()

    w_total = a_tensor.size(2)
    h_total = a_tensor.size(1)
    w_offset = max(0, (w_total - opt.fineSize - 1) // 2)
    h_offset = max(0, (h_total - opt.fineSize - 1) // 2)
    a_tensor = a_tensor[:, h_offset:h_offset + opt.fineSize, w_offset:w_offset + opt.fineSize]
    b_tensor = b_tensor[:, h_offset:h_offset + opt.fineSize, w_offset:w_offset + opt.fineSize]

    a_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a_tensor)
    b_tensor = transforms.Normalize([0.5], [0.5])(b_tensor)
    return {
        "A": a_tensor.unsqueeze(0),
        "B": b_tensor.unsqueeze(0),
        "A_paths": ["gui_input"],
        "B_paths": ["gui_dummy_gray"],
    }


def _physmamba_tensor_to_rgb(image_tensor):
    image_numpy = image_tensor[0].detach().cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))

    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    tmin = float(np.min(image_numpy))
    tmax = float(np.max(image_numpy))
    if tmax - tmin < 1e-8:
        image_numpy = np.zeros_like(image_numpy)
    else:
        image_numpy = (image_numpy - tmin) / (tmax - tmin) * 255.0

    image_numpy = np.nan_to_num(image_numpy, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(image_numpy, 0, 255).astype(np.uint8)


def _build_miigan_options(checkpoint_path, torch_module, device_preference="cuda"):
    if _normalize_device_preference(device_preference) == "cpu":
        raise RuntimeError(
            "算法一 DFSMamba 暂未适配 CPU 推理，请切换为 CUDA，或等待算法二重新接入后再使用。"
        )
    if not torch_module.cuda.is_available():
        raise RuntimeError(
            "当前 DFSMamba 代码依赖 CUDA 和 mamba_ssm，CPU 版 Python 暂不运行算法一。"
            "请切换 CUDA，或后续再单独适配 DFSMamba。"
        )

    from options.test_options import TestOptions

    checkpoints_dir, experiment_name, epoch = _checkpoint_to_option_parts(checkpoint_path)
    argv = [
        "miigan_gui_inference",
        "--model", "miigan",
        "--dataset_mode", "VEDAI",
        "--dataroot", DEFAULT_WEIGHT_ROOT,
        "--checkpoints_dir", checkpoints_dir,
        "--name", experiment_name,
        "--which_epoch", epoch,
        "--batchSize", "1",
        "--nThreads", "1",
        "--serial_batches",
        "--no_flip",
        "--input_nc", "3",
        "--output_nc", "1",
        "--which_model_netG", "MIIGANGenerator",
        "--which_model_netD", "MIIGANDiscriminator",
        "--loadSize", str(MIIGAN_DEFAULT_INPUT_SIZE),
        "--fineSize", str(MIIGAN_DEFAULT_INPUT_SIZE),
        "--resize_or_crop", "resize_and_crop",
        "--gpu_ids", "0",
        "--phase", "test",
        "--how_many", "1",
    ]

    with _temporary_argv(argv):
        with contextlib.redirect_stdout(io.StringIO()):
            opt = TestOptions().parse()

    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True
    return opt


def _load_miigan_model(checkpoint_path, device_preference="cuda"):
    checkpoint_path = os.path.abspath(checkpoint_path)
    cache_key = checkpoint_path
    if cache_key in _MIIGAN_MODEL_CACHE:
        return _MIIGAN_MODEL_CACHE[cache_key]

    _ensure_algorithm_root(MIIGAN_ROOT, "miigan-master")

    with _algorithm_import_context(MIIGAN_ROOT):
        import torch
        from models.models import create_model

        opt = _build_miigan_options(checkpoint_path, torch, device_preference)
        with _cpu_safe_torch_load(torch):
            with contextlib.redirect_stdout(io.StringIO()):
                model = create_model(opt)

    if hasattr(model, "netG"):
        model.netG.eval()

    _MIIGAN_MODEL_CACHE[cache_key] = (model, opt, torch)
    return _MIIGAN_MODEL_CACHE[cache_key]


def _prepare_miigan_input(input_image, opt, torch_module):
    from PIL import Image, ImageOps
    import torchvision.transforms as transforms

    image = np.asarray(input_image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    pil_rgb = Image.fromarray(image).convert("RGB")
    pil_rgb = pil_rgb.resize((opt.loadSize, opt.loadSize), Image.BICUBIC)
    pil_gray = ImageOps.grayscale(pil_rgb)

    a_tensor = transforms.ToTensor()(pil_rgb.copy()).float()
    b_tensor = transforms.ToTensor()(pil_gray.copy()).float()

    if opt.fineSize < opt.loadSize:
        offset = max(0, (opt.loadSize - opt.fineSize) // 2)
        a_tensor = a_tensor[:, offset:offset + opt.fineSize, offset:offset + opt.fineSize]
        b_tensor = b_tensor[:, offset:offset + opt.fineSize, offset:offset + opt.fineSize]

    a_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a_tensor)
    b_tensor = transforms.Normalize([0.5], [0.5])(b_tensor)
    return {
        "A": a_tensor.unsqueeze(0),
        "B": b_tensor.unsqueeze(0),
        "A_paths": ["gui_input"],
        "B_paths": ["gui_dummy_gray"],
    }


def _miigan_tensor_to_rgb(image_tensor):
    image_numpy = image_tensor[0].detach().cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1.0) / 2.0 * 255.0
    image_numpy = np.nan_to_num(image_numpy, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(image_numpy, 0, 255).astype(np.uint8)


# ============================================================
# 算法一：MIIGAN 接口
# ============================================================

def generate_ir_algorithm_1(input_image, model_path=None, device_preference="cuda"):
    """
    算法一：MIIGAN 可见光→红外生成

    MIIGAN 使用 Mamba blocks 作为核心模块的 GAN 架构，
    包含 SCAM (Spatial and Channel Attention Module) 和
    Dual-encoder (Transformer + Mamba) 判别器。

    Args:
        input_image:  numpy 数组 (RGB), 输入可见光图像
        model_path:   str or None, 模型权重路径 (.pth)

    Returns:
        output_image: numpy 数组 (RGB), 生成的红外图像
        elapsed:      float, 推理耗时(秒)

    权重放置规则：
    - 推荐：F:/software/weight/miigan/latest_net_G.pth
    - 也支持：F:/software/weight/latest_net_G.pth
    - 文件名需要保持 MIIGAN 框架格式：<epoch>_net_G.pth
    """
    start_time = time.time()

    checkpoint_path = _find_miigan_checkpoint(model_path)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "未找到 DFSMamba 生成器权重 *_net_G.pth。"
            f"请将 latest_net_G.pth 或 200_net_G.pth 放到 {DEFAULT_WEIGHT_ROOT} "
            "或其子目录（例如 weight/miigan/）下。"
        )

    model, opt, torch_module = _load_miigan_model(checkpoint_path, device_preference)
    data = _prepare_miigan_input(input_image, opt, torch_module)

    with torch_module.no_grad():
        model.set_input(data)
        model.test()
        output_image = _miigan_tensor_to_rgb(model.fake_B.data)

    if output_image.shape[:2] != input_image.shape[:2]:
        original_h, original_w = input_image.shape[:2]
        output_image = cv2.resize(
            output_image, (original_w, original_h), interpolation=cv2.INTER_CUBIC
        )

    elapsed = time.time() - start_time
    return output_image, elapsed


# ============================================================
# 算法二：PhysMamba 接口
# ============================================================

def generate_ir_algorithm_2(input_image, model_path=None, device_preference=None):
    """
    算法二：PhysMamba 可见光→红外生成

    Args:
        input_image:  numpy 数组 (RGB), 输入可见光图像
        model_path:   str or None, 模型权重路径 (.pth)。默认使用固定路径：
                      F:/software/weight/PhysMamba/latest_net_G.pth
        device_preference: str or None, 运行设备偏好，支持 "cpu" / "cuda"。
                           CUDA 不可用时自动回退到 CPU。

    Returns:
        output_image: numpy 数组 (RGB), 生成的红外图像
        elapsed:      float, 推理耗时(秒)
    """
    start_time = time.time()

    checkpoint_path = _find_physmamba_checkpoint(model_path)
    model, opt, torch_module = _load_physmamba_model(checkpoint_path, device_preference)
    data = _prepare_physmamba_input(input_image, opt, torch_module)

    if hasattr(model, "netG"):
        model.netG.eval()

    with torch_module.no_grad():
        model.set_input(data)
        model.test()
        output_image = _physmamba_tensor_to_rgb(model.fake_B.data)

    if output_image.shape[:2] != input_image.shape[:2]:
        original_h, original_w = input_image.shape[:2]
        output_image = cv2.resize(
            output_image, (original_w, original_h), interpolation=cv2.INTER_CUBIC
        )

    elapsed = time.time() - start_time
    return output_image, elapsed
