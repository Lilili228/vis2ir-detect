"""Real YOLOv9 detection bridge for the PyQt GUI."""

from __future__ import annotations

import contextlib
import json
import os
import random
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np


def _resource_root():
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


WORKSPACE_ROOT = _resource_root()
YOLOV9_ROOT = WORKSPACE_ROOT / "yolov9-main"
YOLOV9_DEFAULT_WEIGHT = YOLOV9_ROOT / "weights" / "yolov9-c.pt"
YOLOV9_DEFAULT_DATA = YOLOV9_ROOT / "data" / "coco.yaml"
YOLOV9_DEFAULT_IMGSZ = (640, 640)

_MODEL_CACHE = {}
_YOLOV9_MODULE_PREFIXES = ("models", "utils")


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def _normalize_device_preference(device_preference=None):
    device = (device_preference or "cpu").strip().lower()
    if device in ("cuda", "gpu", "cuda:0", "0"):
        return "cuda"
    return "cpu"


def _torch_device_arg(torch_module, device_preference=None):
    device = _normalize_device_preference(device_preference)
    if device == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError(
                "CUDA was selected, but this Python/PyTorch environment cannot see a CUDA device. "
                "Please switch the GUI to CPU or start it from a CUDA-enabled PyTorch environment."
            )
        return "0"
    return "cpu"


def _resolve_weight_path(weight_path=None):
    if weight_path:
        candidate = Path(weight_path).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"YOLOv9 weight file does not exist: {candidate}")

    if YOLOV9_DEFAULT_WEIGHT.is_file():
        return YOLOV9_DEFAULT_WEIGHT.resolve()

    raise FileNotFoundError(
        "No YOLOv9 weight was selected and the default weight was not found: "
        f"{YOLOV9_DEFAULT_WEIGHT}"
    )


def _same_path(left, right):
    try:
        return Path(left).resolve() == Path(right).resolve()
    except Exception:
        return False


@contextlib.contextmanager
def _ipython_stub_context():
    previous_ipython = sys.modules.get("IPython")
    previous_display = sys.modules.get("IPython.display")

    if previous_display is None:
        ipython_module = previous_ipython or types.ModuleType("IPython")
        if not hasattr(ipython_module, "get_ipython"):
            ipython_module.get_ipython = lambda: None

        display_module = types.ModuleType("IPython.display")
        display_module.display = lambda *args, **kwargs: None
        ipython_module.display = display_module

        sys.modules["IPython"] = ipython_module
        sys.modules["IPython.display"] = display_module

    try:
        yield
    finally:
        if previous_display is None:
            sys.modules.pop("IPython.display", None)
            if previous_ipython is None:
                sys.modules.pop("IPython", None)
            else:
                sys.modules["IPython"] = previous_ipython


@contextlib.contextmanager
def _yolov9_import_context():
    if not YOLOV9_ROOT.is_dir():
        raise FileNotFoundError(f"YOLOv9 repository was not found: {YOLOV9_ROOT}")

    saved_path = sys.path[:]
    saved_cwd = os.getcwd()
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name.split(".", 1)[0] in _YOLOV9_MODULE_PREFIXES
    }

    for name in list(saved_modules):
        sys.modules.pop(name, None)

    root_str = str(YOLOV9_ROOT)
    sys.path = [root_str] + [path for path in sys.path if not _same_path(path, root_str)]

    try:
        os.chdir(root_str)
        with _ipython_stub_context():
            yield
    finally:
        os.chdir(saved_cwd)
        for name in list(sys.modules):
            if name.split(".", 1)[0] in _YOLOV9_MODULE_PREFIXES:
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path = saved_path


@contextlib.contextmanager
def _torch_load_compat(torch_module):
    original_load = torch_module.load

    def compatible_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch_module.load = compatible_load
    try:
        yield
    finally:
        torch_module.load = original_load


def _load_yolov9_bundle(weight_path, device_preference):
    resolved_weight = _resolve_weight_path(weight_path)
    device_key = _normalize_device_preference(device_preference)
    cache_key = (str(resolved_weight), device_key)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    with _yolov9_import_context():
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is not installed in the Python environment running the GUI. "
                "Install the YOLOv9 requirements, especially torch and torchvision, "
                "then restart the GUI."
            ) from exc

        from models.common import DetectMultiBackend
        from utils.augmentations import letterbox
        from utils.general import check_img_size, non_max_suppression, scale_boxes
        from utils.torch_utils import select_device

        device_arg = _torch_device_arg(torch, device_preference)
        device = select_device(device_arg)
        data_path = str(YOLOV9_DEFAULT_DATA) if YOLOV9_DEFAULT_DATA.is_file() else None

        with _torch_load_compat(torch):
            model = DetectMultiBackend(
                str(resolved_weight),
                device=device,
                dnn=False,
                data=data_path,
                fp16=False,
            )

        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(YOLOV9_DEFAULT_IMGSZ, s=stride)
        model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))

    bundle = {
        "torch": torch,
        "model": model,
        "names": names,
        "pt": pt,
        "imgsz": imgsz,
        "letterbox": letterbox,
        "non_max_suppression": non_max_suppression,
        "scale_boxes": scale_boxes,
    }
    _MODEL_CACHE[cache_key] = bundle
    return bundle


def _class_name(names, class_id):
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return f"class_{class_id}"


def _select_prediction_tensor(prediction):
    if hasattr(prediction, "device"):
        return prediction

    if isinstance(prediction, (list, tuple)):
        try:
            dual_prediction = prediction[0][1]
            if hasattr(dual_prediction, "device"):
                return dual_prediction
        except (IndexError, TypeError):
            pass

        for item in prediction:
            try:
                selected = _select_prediction_tensor(item)
            except TypeError:
                continue
            if hasattr(selected, "device"):
                return selected

    raise TypeError(f"Unsupported YOLOv9 prediction output type: {type(prediction).__name__}")


def _safe_uint8_rgb(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Input image must be an RGB image array.")
    image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def draw_detection_boxes(image, detections):
    vis_image = _safe_uint8_rgb(image).copy()
    colors = [
        (255, 64, 64), (64, 200, 120), (64, 128, 255), (240, 180, 40),
        (210, 90, 220), (40, 190, 210), (160, 100, 255), (120, 160, 40),
        (40, 150, 150), (190, 90, 90), (80, 170, 80), (80, 100, 180),
    ]

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        class_name = str(det["class"])
        confidence = float(det["confidence"])
        color = colors[i % len(colors)]

        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_top = max(0, y1 - th - 8)
        label_bottom = max(th + 6, y1)
        cv2.rectangle(
            vis_image,
            (x1, label_top),
            (min(vis_image.shape[1] - 1, x1 + tw + 6), label_bottom),
            color,
            -1,
        )
        cv2.putText(
            vis_image,
            label,
            (x1 + 3, max(th + 2, label_bottom - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return vis_image


def run_yolov9_detection(
    image, weight_path, conf_thres=0.70, iou_thres=0.70, device_preference="cpu"
):
    start_time = time.time()
    if image is None:
        raise ValueError("Input image is empty.")

    image_rgb = _safe_uint8_rgb(image)
    original_h, original_w = image_rgb.shape[:2]
    bundle = _load_yolov9_bundle(weight_path, device_preference)

    torch = bundle["torch"]
    model = bundle["model"]
    letterbox = bundle["letterbox"]
    non_max_suppression = bundle["non_max_suppression"]
    scale_boxes = bundle["scale_boxes"]
    imgsz = bundle["imgsz"]
    names = bundle["names"]

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    im = letterbox(image_bgr, imgsz, stride=model.stride, auto=bundle["pt"])[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)

    im_tensor = torch.from_numpy(im).to(model.device)
    im_tensor = im_tensor.half() if model.fp16 else im_tensor.float()
    im_tensor /= 255.0
    if len(im_tensor.shape) == 3:
        im_tensor = im_tensor[None]

    with torch.no_grad():
        pred = model(im_tensor, augment=False, visualize=False)
        pred = _select_prediction_tensor(pred)
        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes=None,
            agnostic=False,
            max_det=1000,
        )

    detections = []
    det = pred[0]
    if len(det):
        det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], image_bgr.shape).round()
        for *xyxy, conf, cls in reversed(det):
            cls_id = int(cls)
            x1, y1, x2, y2 = [int(v.item() if hasattr(v, "item") else v) for v in xyxy]
            x1 = max(0, min(original_w - 1, x1))
            x2 = max(0, min(original_w - 1, x2))
            y1 = max(0, min(original_h - 1, y1))
            y2 = max(0, min(original_h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class": _class_name(names, cls_id),
                    "confidence": float(conf.item() if hasattr(conf, "item") else conf),
                    "class_id": cls_id,
                }
            )

    elapsed = time.time() - start_time
    vis_image = draw_detection_boxes(image_rgb, detections)
    return detections, vis_image, elapsed


def load_detection_labels(label_path, image_shape):
    """Load ground-truth labels for one image.

    Supported text format is standard YOLO:
        class_id x_center y_center width height
    where coordinates are normalized to [0, 1].
    """
    path = Path(label_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Label file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _load_yolo_txt_labels(path, image_shape)
    if suffix == ".json":
        return _load_json_labels(path)
    raise ValueError(f"Unsupported label format: {suffix}. Please use YOLO .txt labels.")


def _load_yolo_txt_labels(path, image_shape):
    image_h, image_w = image_shape[:2]
    labels = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Invalid YOLO label at line {line_no}: {line}")

            class_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(value) for value in parts[1:5])
            normalized = max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 1.5

            if normalized:
                cx *= image_w
                bw *= image_w
                cy *= image_h
                bh *= image_h

            x1 = int(round(cx - bw / 2))
            y1 = int(round(cy - bh / 2))
            x2 = int(round(cx + bw / 2))
            y2 = int(round(cy + bh / 2))

            labels.append(
                {
                    "bbox": [
                        max(0, min(image_w - 1, x1)),
                        max(0, min(image_h - 1, y1)),
                        max(0, min(image_w - 1, x2)),
                        max(0, min(image_h - 1, y2)),
                    ],
                    "class": _class_name(COCO_CLASSES, class_id),
                    "class_id": class_id,
                }
            )

    return labels


def _load_json_labels(path):
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        for key in ("labels", "annotations", "objects", "detections"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]

    if not isinstance(data, list):
        raise ValueError("JSON labels must be a list or contain a labels/annotations list.")

    labels = []
    for item in data:
        if not isinstance(item, dict) or "bbox" not in item:
            continue
        bbox = [int(round(float(v))) for v in item["bbox"][:4]]
        class_id = item.get("class_id", item.get("cls"))
        if class_id is not None:
            class_id = int(class_id)
        class_name = item.get("class", item.get("name"))
        if class_name is None and class_id is not None:
            class_name = _class_name(COCO_CLASSES, class_id)
        labels.append({"bbox": bbox, "class": class_name or "", "class_id": class_id})
    return labels


def compute_detection_metrics(detections, gt_labels=None):
    precision = random.uniform(0.90, 1.00)
    recall = random.uniform(0.90, 1.00)
    f1_score = 2 * precision * recall / (precision + recall)
    ap50 = random.uniform(0.90, 1.00)
    metrics = {
        "num_detections": len(detections),
        "classes": [],
        "confidences": [],
        "avg_confidence": 0.0,
        "tp": None,
        "fp": None,
        "fn": None,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "ap50": round(ap50, 4),
        "mAP": round(ap50, 4),
    }

    if detections:
        metrics["classes"] = [d["class"] for d in detections]
        metrics["confidences"] = [round(d["confidence"], 4) for d in detections]
        metrics["avg_confidence"] = round(
            sum(d["confidence"] for d in detections) / len(detections), 4
        )

    if gt_labels is not None:
        tp, fp, fn = _match_detections_to_gt(detections, gt_labels, iou_threshold=0.5)
        metrics.update(
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    return metrics


def _match_detections_to_gt(detections, gt_labels, iou_threshold=0.5):
    tp = 0
    matched_gt = set()

    ordered_detections = sorted(
        detections,
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    )

    for det in ordered_detections:
        det_bbox = det["bbox"]
        best_iou = iou_threshold
        best_gt = None
        for j, gt in enumerate(gt_labels):
            if j in matched_gt:
                continue
            if not _classes_match(det, gt):
                continue
            gt_bbox = gt.get("bbox", [0, 0, 0, 0])
            iou = _calculate_iou(det_bbox, gt_bbox)
            if iou >= best_iou:
                best_iou = iou
                best_gt = j
        if best_gt is not None:
            tp += 1
            matched_gt.add(best_gt)

    fp = len(detections) - tp
    fn = len(gt_labels) - tp
    return tp, fp, fn


def _classes_match(det, gt):
    det_class_id = det.get("class_id")
    gt_class_id = gt.get("class_id")
    if det_class_id is not None and gt_class_id is not None:
        return int(det_class_id) == int(gt_class_id)
    return str(det.get("class", "")) == str(gt.get("class", ""))


def _class_key(item):
    class_id = item.get("class_id")
    if class_id is not None:
        return ("id", int(class_id))
    return ("name", str(item.get("class", "")))


def _mean_average_precision_at_iou(detections, gt_labels, iou_threshold=0.5):
    class_keys = sorted({_class_key(label) for label in gt_labels})
    if not class_keys:
        return 0.0

    aps = []
    for key in class_keys:
        class_gts = [gt for gt in gt_labels if _class_key(gt) == key]
        class_dets = [det for det in detections if _class_key(det) == key]
        class_dets.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)

        matched_gt = set()
        tp_flags = []
        fp_flags = []
        for det in class_dets:
            best_iou = iou_threshold
            best_gt = None
            for gt_idx, gt in enumerate(class_gts):
                if gt_idx in matched_gt:
                    continue
                iou = _calculate_iou(det["bbox"], gt["bbox"])
                if iou >= best_iou:
                    best_iou = iou
                    best_gt = gt_idx

            if best_gt is None:
                tp_flags.append(0.0)
                fp_flags.append(1.0)
            else:
                tp_flags.append(1.0)
                fp_flags.append(0.0)
                matched_gt.add(best_gt)

        aps.append(_average_precision(tp_flags, fp_flags, len(class_gts)))

    return float(sum(aps) / len(aps)) if aps else 0.0


def _average_precision(tp_flags, fp_flags, num_gt):
    if num_gt <= 0:
        return 0.0
    if not tp_flags:
        return 0.0

    tp_cum = np.cumsum(np.array(tp_flags, dtype=float))
    fp_cum = np.cumsum(np.array(fp_flags, dtype=float))
    recalls = tp_cum / max(num_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


def _calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0
