"""
通用工具函数：图像加载、显示、保存
"""

import os
import time
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_image(image_path):
    """加载图像，返回 numpy 数组 (RGB)"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def load_image_gray(image_path):
    """加载灰度图像"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return img


def numpy_to_qpixmap(img_numpy):
    """将 numpy 数组 (RGB) 转换为 QPixmap"""
    img_numpy = np.ascontiguousarray(img_numpy)
    h, w, c = img_numpy.shape
    bytes_per_line = 3 * w
    qimg = QImage(img_numpy.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def show_image_on_label(label, img_numpy):
    """
    在 QLabel 上显示图像，自适应缩放，保持原始比例
    img_numpy: numpy 数组 (RGB)
    """
    if img_numpy is None:
        label.clear()
        label.setText("无图像")
        return

    pixmap = numpy_to_qpixmap(img_numpy)
    label_w = label.width()
    label_h = label.height()

    if label_w <= 0 or label_h <= 0:
        label.setPixmap(pixmap)
        return

    # 等比例缩放
    scaled_pixmap = pixmap.scaled(
        label_w, label_h,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )
    label.setPixmap(scaled_pixmap)
    label.setAlignment(Qt.AlignCenter)


def save_image(img_numpy, save_dir, prefix="image"):
    """
    保存图像到指定目录，文件名带时间戳
    返回保存路径
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    img_bgr = cv2.cvtColor(ensure_uint8_rgb(img_numpy), cv2.COLOR_RGB2BGR)
    _write_image_unicode_safe(img_bgr, save_path)
    return save_path


def ensure_uint8_rgb(img_numpy):
    """规范化为 RGB uint8 图像，便于保存和拼接可视化。"""
    image = np.asarray(img_numpy)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3:
        raise ValueError(f"图像维度不支持: {image.shape}")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.shape[2] != 3:
        raise ValueError(f"图像通道数不支持: {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def make_unique_path(file_path):
    """如果目标文件已存在，自动追加时间戳和序号，避免覆盖旧结果。"""
    path = Path(file_path)
    if not path.exists():
        return str(path)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for index in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{timestamp}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return str(candidate)
    raise FileExistsError(f"无法生成不重复文件名: {file_path}")


def save_image_to_path(img_numpy, file_path, overwrite=False):
    """保存 RGB 图像到指定文件路径；overwrite=True 时使用固定文件名覆盖旧结果。"""
    save_path = str(Path(file_path)) if overwrite else make_unique_path(file_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_bgr = cv2.cvtColor(ensure_uint8_rgb(img_numpy), cv2.COLOR_RGB2BGR)
    _write_image_unicode_safe(img_bgr, save_path)
    return save_path


def _write_image_unicode_safe(img_bgr, save_path):
    """使用 imencode + tofile 保存，规避 Windows 下 cv2.imwrite 中文路径失败。"""
    suffix = Path(save_path).suffix.lower() or ".png"
    success, buffer = cv2.imencode(suffix, img_bgr)
    if not success:
        raise IOError(f"图像编码失败: {save_path}")
    try:
        buffer.tofile(save_path)
    except Exception as exc:
        raise IOError(f"图像保存失败: {save_path}") from exc
    if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
        raise IOError(f"图像保存失败: {save_path}")


def scan_image_files(folder_path):
    """非递归扫描文件夹中的支持格式图像。"""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"输入文件夹不存在: {folder_path}")
    files = [
        str(path)
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda item: Path(item).name.lower())


def find_matching_label(input_image_path, label_folder):
    """按输入图像 stem 在标签文件夹中匹配同名或可见光/红外成对标签图。"""
    if not label_folder:
        return None

    folder = Path(label_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"标签文件夹不存在: {label_folder}")

    label_index = {
        path.stem.lower(): path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    for stem in _label_stem_candidates(Path(input_image_path).stem):
        indexed = label_index.get(stem.lower())
        if indexed is not None:
            return str(indexed)
        for suffix in IMAGE_EXTENSIONS:
            candidate = folder / f"{stem}{suffix}"
            if candidate.is_file():
                return str(candidate)
    return None


def _label_stem_candidates(input_stem):
    """生成批量标签匹配候选名，例如 000001_vis -> 000001_ir。"""
    candidates = [input_stem]
    lower_stem = input_stem.lower()
    suffix_map = {
        "_vis": "_ir",
        "-vis": "-ir",
        "_visible": "_ir",
        "-visible": "-ir",
        "_rgb": "_ir",
        "-rgb": "-ir",
        "_color": "_ir",
        "-color": "-ir",
        "_co": "_ir",
        "-co": "-ir",
    }
    for old_suffix, new_suffix in suffix_map.items():
        if lower_stem.endswith(old_suffix):
            candidates.append(input_stem[: -len(old_suffix)] + new_suffix)

    replace_pairs = (
        ("_vis_", "_ir_"),
        ("-vis-", "-ir-"),
        ("_visible_", "_ir_"),
        ("-visible-", "-ir-"),
        ("_rgb_", "_ir_"),
        ("-rgb-", "-ir-"),
    )
    for old_token, new_token in replace_pairs:
        if old_token in lower_stem:
            start = lower_stem.index(old_token)
            candidates.append(
                input_stem[:start] + new_token + input_stem[start + len(old_token):]
            )

    deduped = []
    seen = set()
    for candidate in candidates:
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def create_comparison_image(input_img, generated_img, label_img=None):
    """横向拼接输入、生成结果和可选标签图，生成稳定的对比可视化图。"""
    input_rgb = ensure_uint8_rgb(input_img)
    generated_rgb = ensure_uint8_rgb(generated_img)
    images = [input_rgb, generated_rgb]
    labels = ["Input", "Generated IR"]

    if label_img is not None:
        images.append(ensure_uint8_rgb(label_img))
        labels.append("Ground Truth")

    target_h, target_w = generated_rgb.shape[:2]
    header_h = 34
    normalized = []
    for image in images:
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        header = np.full((header_h, target_w, 3), 245, dtype=np.uint8)
        normalized.append((header, resized))

    panels = []
    for (header, image), label in zip(normalized, labels):
        cv2.putText(
            header,
            label,
            (12, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (35, 55, 70),
            1,
            cv2.LINE_AA,
        )
        panels.append(np.vstack([header, image]))

    separator = np.full((target_h + header_h, 8, 3), 225, dtype=np.uint8)
    canvas_parts = []
    for index, panel in enumerate(panels):
        if index > 0:
            canvas_parts.append(separator)
        canvas_parts.append(panel)
    return np.hstack(canvas_parts)


def rgb_to_ir_pseudo(rgb_image):
    """
    将 RGB 图像转为伪红外效果（占位算法用）
    使用灰度 + 热力图颜色映射模拟红外效果
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    # 直方图均衡化增强对比度
    gray_eq = cv2.equalizeHist(gray)
    # 高斯模糊模拟红外热扩散
    gray_blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    # 应用热力图颜色映射
    ir_pseudo = cv2.applyColorMap(gray_blur, cv2.COLORMAP_INFERNO)
    ir_rgb = cv2.cvtColor(ir_pseudo, cv2.COLOR_BGR2RGB)
    return ir_rgb
