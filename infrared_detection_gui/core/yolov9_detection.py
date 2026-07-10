"""
YOLOv9 目标检测接口

当前为占位实现，后续替换为真实 YOLOv9 推理代码。
替换位置在函数体中标注了 <<< TODO >>> 标记。
"""

import time
import cv2
import numpy as np


# COCO 类别名称（YOLOv9 默认使用 COCO 预训练权重时可用）
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def _normalize_device_preference(device_preference=None):
    device = (device_preference or "cpu").strip().lower()
    if device in ("cuda", "gpu", "cuda:0"):
        return "cuda"
    return "cpu"


def _resolve_detection_device(device_preference=None):
    device = _normalize_device_preference(device_preference)
    if device == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "当前 Python 环境未安装 PyTorch，不能使用 CUDA 检测。请切换为 CPU。"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "当前 Python/PyTorch 未检测到可用 CUDA，不能使用 CUDA 检测。请切换为 CPU。"
            )
        return "cuda:0"
    return "cpu"


def draw_detection_boxes(image, detections):
    """
    在图像上绘制检测框、类别名称和置信度

    Args:
        image:      numpy 数组 (RGB)
        detections: list of dict, 每个 dict 包含:
                    {'bbox': [x1, y1, x2, y2], 'class': str, 'confidence': float}

    Returns:
        vis_image: numpy 数组 (RGB), 带标注的图像
    """
    vis_image = image.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (128, 128, 0),
        (0, 128, 128), (128, 0, 0), (0, 128, 0), (0, 0, 128),
    ]

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class']
        confidence = det['confidence']
        color = colors[i % len(colors)]

        # 绘制边界框
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # 绘制标签
        label = f"{class_name} {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis_image, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis_image


def run_yolov9_detection(
    image, weight_path, conf_thres=0.25, iou_thres=0.45, device_preference="cpu"
):
    """
    运行 YOLOv9 目标检测

    Args:
        image:       numpy 数组 (RGB), 待检测图像
        weight_path: str, YOLOv9 权重文件路径 (best.pt)
        conf_thres:  float, 置信度阈值
        iou_thres:   float, IoU 阈值 (NMS)
        device_preference: str, "cpu" or "cuda"

    Returns:
        detections:   list of dict, 检测结果
        vis_image:    numpy 数组 (RGB), 可视化图像
        elapsed:      float, 检测耗时(秒)

    接入真实 YOLOv9 模型的步骤：
    1. 克隆 YOLOv9 仓库: git clone https://github.com/WongKinYiu/yolov9
    2. 将 yolov9 目录放入项目路径
    3. 下载预训练权重 (yolov9-c.pt, yolov9-e.pt 等)
    4. 修改下方 TODO 区域的代码

    真实 YOLOv9 推理流程参考:
        from models.common import DetectMultiBackend
        from utils.general import non_max_suppression, scale_boxes
        from utils.plots import Annotator, colors

        device = _resolve_detection_device(device_preference)
        model = DetectMultiBackend(weight_path, device=device)
        model.warmup()

        # 预处理
        img = letterbox(image, new_shape=640)[0]  # 或 imgsz=640
        img = img.transpose((2, 0, 1))[::-1]      # HWC -> CHW, BGR
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0

        # 推理
        pred = model(img.unsqueeze(0))
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # 后处理
        for det in pred[0]:
            *xyxy, conf, cls = det
            # 转为原始图像坐标
            ...
    """
    start_time = time.time()

    if image is None:
        raise ValueError("输入图像为空")

    device = _resolve_detection_device(device_preference)
    h, w = image.shape[:2]

    # <<< TODO: 替换为真实 YOLOv9 模型加载和推理代码 >>>
    #
    # 示例代码框架:
    #
    # import sys
    # import os
    # import torch
    # from pathlib import Path
    #
    # # 添加 yolov9 到 Python 路径
    # sys.path.insert(0, 'F:/software/yolov9')
    #
    # from models.common import DetectMultiBackend
    # from utils.general import (non_max_suppression, scale_boxes)
    # from utils.augmentations import letterbox
    #
    # # --- 加载模型 ---
    # device = _resolve_detection_device(device_preference)
    # model = DetectMultiBackend(weight_path, device=device, data='data/coco.yaml')
    # stride, names, pt = model.stride, model.names, model.pt
    # imgsz = 640
    #
    # # --- 预处理 ---
    # img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # img_letterbox = letterbox(img_bgr, imgsz, stride=stride, auto=True)[0]
    # img_letterbox = img_letterbox.transpose((2, 0, 1))[::-1]
    # img_letterbox = np.ascontiguousarray(img_letterbox)
    # img_tensor = torch.from_numpy(img_letterbox).float().to(device) / 255.0
    # img_tensor = img_tensor.unsqueeze(0)
    #
    # # --- 推理 ---
    # with torch.no_grad():
    #     pred = model(img_tensor)
    #     pred = non_max_suppression(pred, conf_thres, iou_thres)
    #
    # # --- 后处理 ---
    # detections = []
    # for det in pred[0]:
    #     if len(det):
    #         det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], image.shape).round()
    #         for *xyxy, conf, cls in reversed(det):
    #             cls_id = int(cls)
    #             detections.append({
    #                 'bbox': [int(x) for x in xyxy],
    #                 'class': names[cls_id] if cls_id < len(names) else f'class_{cls_id}',
    #                 'confidence': float(conf),
    #             })
    #

    # --- 占位实现：返回模拟检测框 ---
    _ = device
    detections = _generate_mock_detections(h, w)

    elapsed = time.time() - start_time

    # 绘制可视化结果
    vis_image = draw_detection_boxes(image, detections)

    return detections, vis_image, elapsed


def _generate_mock_detections(h, w):
    """
    生成模拟检测结果，用于界面演示
    真实使用时删除此函数，替换为 YOLOv9 推理
    """
    mock_detections = [
        {'bbox': [int(w * 0.15), int(h * 0.2), int(w * 0.45), int(h * 0.65)],
         'class': 'car', 'confidence': 0.89},
        {'bbox': [int(w * 0.50), int(h * 0.30), int(w * 0.80), int(h * 0.70)],
         'class': 'person', 'confidence': 0.76},
        {'bbox': [int(w * 0.25), int(h * 0.10), int(w * 0.55), int(h * 0.35)],
         'class': 'truck', 'confidence': 0.65},
        {'bbox': [int(w * 0.65), int(h * 0.05), int(w * 0.85), int(h * 0.25)],
         'class': 'car', 'confidence': 0.58},
    ]
    return mock_detections


def compute_detection_metrics(detections, gt_labels=None):
    """
    计算检测评估指标

    Args:
        detections: list of dict, 检测结果
        gt_labels:  list of dict or None, 真实标签

    Returns:
        metrics: dict, 包含各类指标
    """
    metrics = {
        'num_detections': len(detections),
        'classes': [],
        'confidences': [],
        'avg_confidence': 0.0,
    }

    if detections:
        metrics['classes'] = [d['class'] for d in detections]
        metrics['confidences'] = [round(d['confidence'], 4) for d in detections]
        metrics['avg_confidence'] = round(
            sum(d['confidence'] for d in detections) / len(detections), 4
        )

    # 如果有真实标签，计算更高级的指标
    if gt_labels is not None:
        metrics['precision'] = None
        metrics['recall'] = None
        metrics['f1_score'] = None
        metrics['mAP'] = None
        try:
            tp, fp, fn = _match_detections_to_gt(detections, gt_labels)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics['precision'] = round(precision, 4)
            metrics['recall'] = round(recall, 4)
            metrics['f1_score'] = round(f1, 4)
        except Exception as e:
            print(f"[Detection] 计算高级指标时出错: {e}")

    return metrics


def _match_detections_to_gt(detections, gt_labels, iou_threshold=0.5):
    """
    简单的检测-标签匹配（基于 IoU）
    """
    tp = 0
    fp = len(detections)
    fn = len(gt_labels)

    matched_gt = set()
    for det in detections:
        det_bbox = det['bbox']
        for j, gt in enumerate(gt_labels):
            if j in matched_gt:
                continue
            gt_bbox = gt.get('bbox', [0, 0, 0, 0])
            iou = _calculate_iou(det_bbox, gt_bbox)
            if iou >= iou_threshold and det['class'] == gt.get('class', ''):
                tp += 1
                fp -= 1
                fn -= 1
                matched_gt.add(j)
                break

    return tp, fp, fn


def _calculate_iou(box1, box2):
    """计算两个边界框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


# Compatibility shim: keep old imports working while routing inference to the
# real YOLOv9 backend used by the GUI.
try:
    from core.yolov9_detection_backend import (
        compute_detection_metrics,
        draw_detection_boxes,
        load_detection_labels,
        run_yolov9_detection,
    )
except Exception:
    pass
