"""
图像质量评估指标计算模块
PSNR, SSIM, L1, LIPIPS
"""

import os
import sys

import cv2
import numpy as np


def _resource_root():
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


WORKSPACE_ROOT = _resource_root()
PHYSMAMBA_ROOT = os.path.join(WORKSPACE_ROOT, "PhysMamba")
PHYSMAMBA_LPIPS_WEIGHT = os.path.join(PHYSMAMBA_ROOT, "lpips", "weights", "v0.1", "alex.pth")
TORCH_HOME = os.path.join(WORKSPACE_ROOT, "weight", "torch")
TORCH_HUB_DIR = os.path.join(TORCH_HOME, "hub")
TORCH_CHECKPOINT_DIR = os.path.join(TORCH_HUB_DIR, "checkpoints")
ALEXNET_CHECKPOINT = os.path.join(TORCH_CHECKPOINT_DIR, "alexnet-owt-7be5be79.pth")
os.environ.setdefault("TORCH_HOME", TORCH_HOME)

_LPIPS_MODEL_CACHE = {}
_LPIPS_WARNING_SHOWN = False
_LPIPS_DISABLED_REASON = None


def calculate_psnr(img1, img2):
    """计算 PSNR (Peak Signal-to-Noise Ratio)"""
    mse_val = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse_val == 0:
        return float('inf')
    max_val = 255.0
    psnr_val = 20 * np.log10(max_val / np.sqrt(mse_val))
    return psnr_val


def calculate_ssim(img1, img2, L=255.0):
    """
    计算 SSIM (Structural Similarity Index)

    使用滑动窗口方式计算，K1=0.01, K2=0.03
    """
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    if min(img1.shape[:2]) < 11 or min(img2.shape[:2]) < 11:
        return 0.0

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 使用统一的高斯窗口
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean())


def _ensure_rgb(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3]
    raise ValueError(f"Unsupported image shape for metrics: {image.shape}")


def _normalize_rgb_to_minus_one_one(image):
    image = _ensure_rgb(image).astype(np.float32)
    return image / 127.5 - 1.0


def calculate_l1(img1, img2):
    """Calculate RGB L1 loss after normalizing images to [-1, 1]."""
    img1_norm = _normalize_rgb_to_minus_one_one(img1)
    img2_norm = _normalize_rgb_to_minus_one_one(img2)
    return float(np.mean(np.abs(img1_norm - img2_norm)))


def _warn_lpips_once(message):
    global _LPIPS_WARNING_SHOWN
    if not _LPIPS_WARNING_SHOWN:
        print(f"[Metrics] LIPIPS unavailable: {message}")
        _LPIPS_WARNING_SHOWN = True


def _import_lpips_class():
    try:
        import lpips

        if hasattr(lpips, "LPIPS"):
            return lpips.LPIPS
    except Exception:
        pass

    if not os.path.isdir(PHYSMAMBA_ROOT):
        raise ImportError("lpips package is not installed and PhysMamba/lpips was not found")

    added_path = False
    if PHYSMAMBA_ROOT not in sys.path:
        sys.path.insert(0, PHYSMAMBA_ROOT)
        added_path = True
    try:
        from lpips.lpips import LPIPS

        return LPIPS
    finally:
        if added_path:
            try:
                sys.path.remove(PHYSMAMBA_ROOT)
            except ValueError:
                pass


def _load_lpips_model(torch_module):
    device_name = "cuda:0" if torch_module.cuda.is_available() else "cpu"
    if device_name in _LPIPS_MODEL_CACHE:
        return _LPIPS_MODEL_CACHE[device_name]

    os.makedirs(TORCH_CHECKPOINT_DIR, exist_ok=True)
    if os.path.isfile(ALEXNET_CHECKPOINT) and os.path.getsize(ALEXNET_CHECKPOINT) < 50 * 1024 * 1024:
        raise RuntimeError(
            f"AlexNet checkpoint is incomplete: {ALEXNET_CHECKPOINT} "
            f"({os.path.getsize(ALEXNET_CHECKPOINT)} bytes). "
            "Please replace it with the full alexnet-owt-7be5be79.pth file."
        )
    if hasattr(torch_module, "hub"):
        torch_module.hub.set_dir(TORCH_HUB_DIR)

    LPIPS = _import_lpips_class()
    kwargs = {"net": "alex"}
    if os.path.isfile(PHYSMAMBA_LPIPS_WEIGHT):
        kwargs["model_path"] = PHYSMAMBA_LPIPS_WEIGHT
    try:
        model = LPIPS(**kwargs, verbose=False)
    except TypeError:
        model = LPIPS(**kwargs)

    device = torch_module.device(device_name)
    model = model.to(device)
    model.eval()
    _LPIPS_MODEL_CACHE[device_name] = (model, device)
    return _LPIPS_MODEL_CACHE[device_name]


def calculate_lipips(img1, img2):
    """Calculate LPIPS/AlexNet perceptual distance for RGB images."""
    global _LPIPS_DISABLED_REASON
    if _LPIPS_DISABLED_REASON:
        return None
    try:
        import torch

        model, device = _load_lpips_model(torch)
        img1_norm = _normalize_rgb_to_minus_one_one(img1)
        img2_norm = _normalize_rgb_to_minus_one_one(img2)
        tensor1 = torch.from_numpy(img1_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        tensor2 = torch.from_numpy(img2_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            value = model(tensor1, tensor2).mean().item()
        return float(value)
    except Exception as exc:
        _LPIPS_DISABLED_REASON = str(exc)
        _warn_lpips_once(_LPIPS_DISABLED_REASON)
        return None


def get_lipips_error():
    return _LPIPS_DISABLED_REASON


def calculate_metrics(generated_image, gt_image=None):
    """
    计算生成图像的质量评估指标

    Args:
        generated_image: numpy数组 (RGB), 生成的图像
        gt_image: numpy数组 (RGB) 或 None, 真实红外标签图像

    Returns:
        dict: {'psnr', 'ssim', 'l1', 'lipips'}
    """
    metrics = {
        'psnr': None,
        'ssim': None,
        'l1': None,
        'lipips': None,
    }

    if generated_image is None or gt_image is None:
        return metrics

    try:
        generated_image = _ensure_rgb(generated_image)
        gt_image = _ensure_rgb(gt_image)

        if generated_image.shape[:2] != gt_image.shape[:2]:
            gen_h, gen_w = generated_image.shape[:2]
            gt_image = cv2.resize(gt_image, (gen_w, gen_h), interpolation=cv2.INTER_AREA)

        gen_gray = cv2.cvtColor(generated_image, cv2.COLOR_RGB2GRAY)
        gt_gray = cv2.cvtColor(gt_image, cv2.COLOR_RGB2GRAY)

        metrics['psnr'] = round(calculate_psnr(gen_gray, gt_gray), 4)
        metrics['ssim'] = round(calculate_ssim(gen_gray, gt_gray), 4)
        metrics['l1'] = round(calculate_l1(generated_image, gt_image), 4)
        lipips_value = calculate_lipips(generated_image, gt_image)
        metrics['lipips'] = round(lipips_value, 4) if lipips_value is not None else None
    except Exception as e:
        print(f"[Metrics] 计算指标时出错: {e}")

    return metrics
