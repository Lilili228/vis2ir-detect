"""YOLOv9 红外目标检测界面：浅色科研风重构版。"""

import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.utils import IMAGE_EXTENSIONS, load_image, save_image_to_path, scan_image_files, show_image_on_label
from core.yolov9_detection_backend import (
    compute_detection_metrics,
    load_detection_labels,
    run_yolov9_detection,
)


DEFAULT_DETECT_OUTPUT_DIR = "D:/YOLOv9_Detect_Result"


class DetectionWorker(QThread):
    """YOLOv9 检测工作线程。"""

    result_ready = pyqtSignal(list, object, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, image, weight_path, conf_thres, iou_thres, device_preference):
        super().__init__()
        self.image = image.copy()
        self.weight_path = weight_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device_preference = device_preference

    def run(self):
        try:
            detections, vis_image, elapsed = run_yolov9_detection(
                self.image,
                self.weight_path,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                device_preference=self.device_preference,
            )
            self.result_ready.emit(detections, vis_image, elapsed)
        except Exception as e:
            self.error_occurred.emit(str(e))


def _safe_dir_name(name, fallback="unnamed"):
    name = (name or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)
    cleaned = cleaned.strip(" ._")
    return cleaned or fallback


def _fmt_value(value, decimals=4):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def _find_label_txt(image_path, label_dir):
    if not label_dir:
        return None
    candidate = Path(label_dir) / f"{Path(image_path).stem}.txt"
    return str(candidate) if candidate.is_file() else None


def _write_prediction_txt(detections, txt_path):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as handle:
        if not detections:
            handle.write("No objects detected.\n")
            return
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_id = det.get("class_id", "")
            class_name = str(det.get("class", "")).replace(" ", "_")
            confidence = float(det.get("confidence", 0.0))
            handle.write(
                f"{class_id} {class_name} {confidence:.4f} {x1} {y1} {x2} {y2}\n"
            )


def _write_batch_params(params, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for key, value in params.items():
            handle.write(f"{key}: {value}\n")


def _write_batch_errors(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    failed = [record for record in records if record.get("status") == "failed"]
    with open(path, "w", encoding="utf-8") as handle:
        if not failed:
            handle.write("No failed images.\n")
            return
        for index, record in enumerate(failed, 1):
            handle.write(
                f"{index}. {record.get('image_name', '')}: {record.get('error_message', '')}\n"
            )


def _write_batch_summary(summary, records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    separator = "=" * 140
    label_records = [
        record
        for record in records
        if record.get("tp") not in ("N/A", None, "")
    ]
    total_tp = sum(int(record.get("tp") or 0) for record in label_records)
    total_fp = sum(int(record.get("fp") or 0) for record in label_records)
    total_fn = sum(int(record.get("fn") or 0) for record in label_records)
    label_metrics_available = bool(label_records)

    class_totals = Counter()
    for record in records:
        class_totals.update(record.get("class_counter", Counter()))
    metric_records = [
        record
        for record in records
        if record.get("precision") not in ("N/A", "", None)
    ]
    avg_precision = (
        sum(float(record.get("precision", 0.0)) for record in metric_records) / len(metric_records)
        if metric_records
        else None
    )
    avg_recall = (
        sum(float(record.get("recall", 0.0)) for record in metric_records) / len(metric_records)
        if metric_records
        else None
    )
    avg_f1 = (
        sum(float(record.get("f1_score", 0.0)) for record in metric_records) / len(metric_records)
        if metric_records
        else None
    )
    avg_ap50 = (
        sum(float(record.get("ap50", 0.0)) for record in metric_records) / len(metric_records)
        if metric_records
        else None
    )

    lines = [
        "YOLOv9 批量检测结果汇总",
        "",
        "第一部分：批次总体信息",
        f"批次名称: {summary.get('batch_name', '')}",
        f"任务状态: {'已取消' if summary.get('canceled') else '已完成'}",
        f"开始时间: {summary.get('start_time', '')}",
        f"结束时间: {summary.get('end_time', '')}",
        f"输入文件夹: {summary.get('input_dir', '')}",
        f"输出文件夹: {summary.get('output_dir', '')}",
        f"标签文件夹: {summary.get('label_dir') or 'N/A'}",
        f"YOLOv9 权重路径: {summary.get('weight_path') or 'default'}",
        f"运行设备: {summary.get('device', '')}",
        f"conf_thres: {summary.get('conf_thres', 0):.2f}",
        f"iou_thres: {summary.get('iou_thres', 0):.2f}",
        f"总图片数: {summary.get('total', 0)}",
        f"成功数量: {summary.get('success', 0)}",
        f"失败数量: {summary.get('failed', 0)}",
        f"总检测耗时: {summary.get('total_elapsed', 0):.3f} s",
        f"平均单张耗时: {summary.get('avg_elapsed', 0):.3f} s",
        f"总检测目标数: {summary.get('total_detections', 0)}",
        f"批次平均置信度: {_fmt_value(summary.get('avg_confidence'), 4)}",
        f"成功匹配标签图片数: {len(label_records)}",
        f"是否计算真实标签指标: {'是' if label_metrics_available else '否'}",
        "指标说明: Precision / Recall / F1-score / AP@0.5 按演示系统旧逻辑随机生成。",
        f"批次平均 Precision: {_fmt_value(avg_precision, 4)}",
        f"批次平均 Recall: {_fmt_value(avg_recall, 4)}",
        f"批次平均 F1-score: {_fmt_value(avg_f1, 4)}",
        f"批次平均 AP@0.5: {_fmt_value(avg_ap50, 4)}",
    ]
    if label_metrics_available:
        lines.extend(
            [
                f"总 TP: {total_tp}",
                f"总 FP: {total_fp}",
                f"总 FN: {total_fn}",
            ]
        )

    lines.extend(
        [
            "",
            "第二部分：每张图片指标汇总表",
            separator,
            f"{'Image Name':<26} {'Status':<8} {'DetCount':>8} {'AvgConf':>8} "
            f"{'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>10} {'Recall':>10} "
            f"{'F1':>8} {'AP@0.5':>8} {'Time(s)':>8}",
            separator,
        ]
    )
    for record in records:
        lines.append(
            f"{record.get('image_name', ''):<26.26} {record.get('status', ''):<8} "
            f"{str(record.get('detected_count', 0)):>8} "
            f"{_fmt_value(record.get('average_confidence'), 4):>8} "
            f"{str(record.get('tp', 'N/A')):>5} {str(record.get('fp', 'N/A')):>5} "
            f"{str(record.get('fn', 'N/A')):>5} "
            f"{_fmt_value(record.get('precision'), 4):>10} "
            f"{_fmt_value(record.get('recall'), 4):>10} "
            f"{_fmt_value(record.get('f1_score'), 4):>8} "
            f"{_fmt_value(record.get('ap50'), 4):>8} "
            f"{_fmt_value(record.get('inference_time'), 3):>8}"
        )
    lines.append(separator)

    lines.extend(["", "第三部分：类别统计汇总", "Class Statistics:"])
    if class_totals:
        for class_name, count in sorted(class_totals.items()):
            lines.append(f"- {class_name}: {count}")
    else:
        lines.append("- N/A")

    lines.extend(["", "第四部分：失败样本列表", "Failed Images:"])
    failed = [record for record in records if record.get("status") == "failed"]
    if failed:
        for index, record in enumerate(failed, 1):
            lines.append(f"{index}. {record.get('image_name', '')}: {record.get('error_message', '')}")
    else:
        lines.append("No failed images.")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


class BatchDetectionWorker(QThread):
    """YOLOv9 批量检测工作线程。"""

    log_message = pyqtSignal(str)
    progress_changed = pyqtSignal(object)
    preview_ready = pyqtSignal(object, object, list, float, float)
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, image_files, input_dir, output_dir, label_dir, weight_path, conf_thres, iou_thres, device_preference, device_label):
        super().__init__()
        self.image_files = image_files
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.label_dir = label_dir
        self.weight_path = weight_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device_preference = device_preference
        self.device_label = device_label
        self.cancel_requested = False

    def request_cancel(self):
        self.cancel_requested = True

    def run(self):
        batch_name = _safe_dir_name(Path(self.input_dir).name, "batch")
        batch_dir = os.path.join(self.output_dir, "批次文件夹", batch_name)
        images_dir = os.path.join(batch_dir, "images")
        labels_dir = os.path.join(batch_dir, "labels")
        reports_dir = os.path.join(batch_dir, "reports")
        try:
            for folder in (images_dir, labels_dir, reports_dir):
                os.makedirs(folder, exist_ok=True)
        except Exception as exc:
            self.error_occurred.emit(f"输出目录无法创建: {exc}")
            return

        start_readable = time.strftime("%Y-%m-%d %H:%M:%S")
        start_clock = time.time()
        records = []
        success = 0
        failed = 0
        total_detections = 0
        confidence_sum = 0.0
        confidence_count = 0

        self.log_message.emit("[RUN] 启动批量检测任务")
        self.log_message.emit(f"[INFO] 输入文件夹: {self.input_dir}")
        self.log_message.emit(f"[INFO] 输出目录: {batch_dir}")
        self.log_message.emit(f"[INFO] 标签文件夹: {self.label_dir or '未选择，跳过真实检测评价指标计算'}")
        total = len(self.image_files)

        for index, image_path in enumerate(self.image_files, 1):
            if self.cancel_requested:
                self.log_message.emit("[CANCEL] 用户取消任务，当前批次停止后续图片检测")
                break

            image_start = time.time()
            image_name = Path(image_path).name
            stem = Path(image_path).stem
            record = {
                "image_name": image_name,
                "status": "failed",
                "detected_count": 0,
                "average_confidence": None,
                "class_statistics": "",
                "class_counter": Counter(),
                "inference_time": 0.0,
                "result_image_path": "",
                "prediction_txt_path": "",
                "label_path": "",
                "tp": "N/A",
                "fp": "N/A",
                "fn": "N/A",
                "precision": None,
                "recall": None,
                "f1_score": None,
                "ap50": None,
                "error_message": "",
            }
            self.progress_changed.emit(
                self._progress(index - 1, total, success, failed, image_name, start_clock)
            )
            self.log_message.emit(f"[RUN] ({index}/{total}) 检测: {image_name}")

            try:
                input_image = load_image(image_path)
                detections, vis_image, elapsed = run_yolov9_detection(
                    input_image,
                    self.weight_path,
                    conf_thres=self.conf_thres,
                    iou_thres=self.iou_thres,
                    device_preference=self.device_preference,
                )
                result_image_path = save_image_to_path(
                    vis_image,
                    os.path.join(images_dir, f"{stem}_detect.jpg"),
                    overwrite=True,
                )
                prediction_txt_path = os.path.join(labels_dir, f"{stem}_pred.txt")
                _write_prediction_txt(detections, prediction_txt_path)

                gt_labels = None
                label_path = ""
                if self.label_dir:
                    label_path = _find_label_txt(image_path, self.label_dir) or ""
                    if label_path:
                        try:
                            gt_labels = load_detection_labels(label_path, input_image.shape)
                            self.log_message.emit(f"[OK] 已匹配标签: {Path(label_path).name}")
                        except Exception as exc:
                            self.log_message.emit(f"[WARN] 标签解析失败，跳过该图真实指标: {exc}")
                            gt_labels = None
                    else:
                        self.log_message.emit(f"[INFO] 未找到同名标签，真实指标写 N/A: {image_name}")
                else:
                    self.log_message.emit("[INFO] 未选择标签文件夹，跳过真实检测评价指标计算")

                metrics = compute_detection_metrics(detections, gt_labels)
                class_counter = Counter(metrics.get("classes", []))
                avg_conf = metrics.get("avg_confidence", 0.0)
                det_count = metrics.get("num_detections", len(detections))
                record.update(
                    {
                        "status": "success",
                        "detected_count": det_count,
                        "average_confidence": avg_conf if detections else 0.0,
                        "class_statistics": ", ".join(f"{k}:{v}" for k, v in class_counter.items()),
                        "class_counter": class_counter,
                        "inference_time": elapsed,
                        "result_image_path": result_image_path,
                        "prediction_txt_path": prediction_txt_path,
                        "label_path": label_path,
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1_score": metrics.get("f1_score"),
                        "ap50": metrics.get("ap50"),
                        "error_message": "",
                    }
                )
                if gt_labels is not None:
                    record.update(
                        {
                            "tp": metrics.get("tp", 0),
                            "fp": metrics.get("fp", 0),
                            "fn": metrics.get("fn", 0),
                        }
                    )

                success += 1
                total_detections += det_count
                confidence_sum += sum(float(det.get("confidence", 0.0)) for det in detections)
                confidence_count += len(detections)
                self.preview_ready.emit(input_image, vis_image, detections, avg_conf, elapsed)
                self.log_message.emit(f"[OK] 保存检测图: {result_image_path}")
            except Exception as exc:
                failed += 1
                record["error_message"] = str(exc)
                record["inference_time"] = time.time() - image_start
                self.log_message.emit(f"[ERROR] 图片检测失败: {image_name} | {exc}")

            records.append(record)
            self.progress_changed.emit(
                self._progress(success + failed, total, success, failed, image_name, start_clock)
            )

        total_elapsed = time.time() - start_clock
        completed = success + failed
        summary = {
            "batch_name": batch_name,
            "start_time": start_readable,
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_dir": self.input_dir,
            "output_dir": batch_dir,
            "label_dir": self.label_dir,
            "weight_path": self.weight_path,
            "device": self.device_label,
            "conf_thres": self.conf_thres,
            "iou_thres": self.iou_thres,
            "total": total,
            "success": success,
            "failed": failed,
            "completed": completed,
            "total_elapsed": total_elapsed,
            "avg_elapsed": total_elapsed / completed if completed else 0.0,
            "total_detections": total_detections,
            "avg_confidence": confidence_sum / confidence_count if confidence_count else None,
            "has_label_dir": bool(self.label_dir),
            "canceled": self.cancel_requested,
        }
        params_path = os.path.join(reports_dir, "batch_detection_params.txt")
        summary_path = os.path.join(reports_dir, "batch_detection_summary.txt")
        errors_path = os.path.join(reports_dir, "batch_detection_errors.txt")
        try:
            _write_batch_params(summary, params_path)
            _write_batch_summary(summary, records, summary_path)
            _write_batch_errors(records, errors_path)
            self.log_message.emit(f"[OK] 批次汇总已保存: {summary_path}")
        except Exception as exc:
            self.log_message.emit(f"[ERROR] txt 报告保存失败: {exc}")

        self.result_ready.emit(
            {
                "summary": summary,
                "records": records,
                "batch_dir": batch_dir,
                "summary_path": summary_path,
                "params_path": params_path,
                "errors_path": errors_path,
            }
        )

    def _progress(self, completed, total, success, failed, current_file, start_clock):
        elapsed = time.time() - start_clock
        return {
            "completed": completed,
            "total": total,
            "success": success,
            "failed": failed,
            "current_file": current_file,
            "elapsed": elapsed,
            "avg_elapsed": elapsed / completed if completed else 0.0,
            "percent": int(completed / total * 100) if total else 0,
        }


class DetectionTab(QWidget):
    """YOLOv9 目标检测界面。"""

    def __init__(self, log_signal=None):
        super().__init__()
        self.log_signal = log_signal
        self.input_image = None
        self.vis_image = None
        self.detections = None
        self.det_worker = None
        self.batch_worker = None
        self.metric_labels = {}
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 14, 18, 16)
        main_layout.setSpacing(14)

        main_layout.addWidget(self._create_header())

        workspace_layout = QHBoxLayout()
        workspace_layout.setSpacing(14)
        workspace_layout.addWidget(self._create_config_panel(), 0)
        workspace_layout.addWidget(self._create_visual_panel(), 1)
        workspace_layout.addWidget(self._create_stats_panel(), 0)
        main_layout.addLayout(workspace_layout, 1)

        main_layout.addWidget(self._create_log_panel(), 0)

    def _create_header(self):
        header = QFrame()
        header.setObjectName("pageHeader")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(16)

        title_area = QVBoxLayout()
        title_area.setSpacing(4)
        title = QLabel("YOLOv9 红外目标检测")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Infrared Object Detection with YOLOv9")
        subtitle.setObjectName("pageSubtitle")
        title_area.addWidget(title)
        title_area.addWidget(subtitle)
        layout.addLayout(title_area, 1)

        status_card = QFrame()
        status_card.setObjectName("statusBadge")
        status_layout = QGridLayout(status_card)
        status_layout.setContentsMargins(14, 8, 14, 8)
        status_layout.setHorizontalSpacing(12)
        status_layout.setVerticalSpacing(4)

        weight_label = QLabel("模型权重")
        weight_label.setObjectName("statusLabel")
        self.lbl_weight_status = QLabel("未选择")
        self.lbl_weight_status.setObjectName("statusValue")
        state_label = QLabel("检测状态")
        state_label.setObjectName("statusLabel")
        self.lbl_detection_status = QLabel("待检测")
        self.lbl_detection_status.setObjectName("statusValue")
        device_label = QLabel("运行设备")
        device_label.setObjectName("statusLabel")
        self.lbl_device_status = QLabel("CPU")
        self.lbl_device_status.setObjectName("statusValue")

        status_layout.addWidget(weight_label, 0, 0)
        status_layout.addWidget(self.lbl_weight_status, 0, 1)
        status_layout.addWidget(state_label, 1, 0)
        status_layout.addWidget(self.lbl_detection_status, 1, 1)
        status_layout.addWidget(device_label, 2, 0)
        status_layout.addWidget(self.lbl_device_status, 2, 1)
        layout.addWidget(status_card, 0)
        return header

    def _create_config_panel(self):
        panel = QFrame()
        panel.setObjectName("configPanel")
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(330)
        outer_layout = QVBoxLayout(panel)
        outer_layout.setContentsMargins(16, 16, 16, 16)
        outer_layout.setSpacing(12)

        title = QLabel("检测配置")
        title.setObjectName("sectionTitle")
        hint = QLabel("选择检测模式、输入数据、权重和阈值参数")
        hint.setObjectName("sectionHint")
        outer_layout.addWidget(title)
        outer_layout.addWidget(hint)

        scroll_area = QScrollArea()
        scroll_area.setObjectName("workflowScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        scroll_content.setObjectName("workflowScrollContent")
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        mode_label = QLabel("处理模式")
        mode_label.setObjectName("fieldLabel")
        layout.addWidget(mode_label)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(14)
        self.radio_single = QRadioButton("单张检测")
        self.radio_batch = QRadioButton("批量检测")
        self.radio_single.setChecked(True)
        self.radio_single.toggled.connect(self._on_mode_changed)
        self.radio_batch.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self.radio_single)
        mode_row.addWidget(self.radio_batch)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        self.single_section = QFrame()
        self.single_section.setObjectName("workflowField")
        single_layout = QVBoxLayout(self.single_section)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.setSpacing(10)
        self.lbl_image_path, self.btn_select_image = self._add_file_picker(
            single_layout, "检测图像", "请选择待检测图像...", self._on_select_image
        )
        self.lbl_label_path, self.btn_select_label = self._add_file_picker(
            single_layout, "真实标签（可选）", "用于 Precision / Recall", self._on_select_label
        )
        layout.addWidget(self.single_section)

        self.batch_section = QFrame()
        self.batch_section.setObjectName("workflowField")
        batch_layout = QVBoxLayout(self.batch_section)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.setSpacing(10)
        self.lbl_batch_input_dir, self.btn_select_batch_input = self._add_file_picker(
            batch_layout, "批量输入文件夹", "请选择待检测图像文件夹...", self._on_select_batch_input_dir
        )
        self.lbl_batch_label_dir, self.btn_select_batch_label = self._add_file_picker(
            batch_layout, "批量标签文件夹（可选）", "同名 YOLO .txt 标签文件夹", self._on_select_batch_label_dir
        )
        layout.addWidget(self.batch_section)

        self.lbl_batch_output_dir, self.btn_select_batch_output = self._add_file_picker(
            layout, "输出目录", DEFAULT_DETECT_OUTPUT_DIR, self._on_select_batch_output_dir
        )
        self.lbl_batch_output_dir.setText(DEFAULT_DETECT_OUTPUT_DIR)
        self.lbl_batch_output_dir.setToolTip(DEFAULT_DETECT_OUTPUT_DIR)

        self.lbl_weight_path, self.btn_select_weight = self._add_file_picker(
            layout, "YOLOv9 权重文件", "选择 best.pt / .pth", self._on_select_weight
        )

        device_label = QLabel("运行设备")
        device_label.setObjectName("fieldLabel")
        layout.addWidget(device_label)

        self.combo_device = QComboBox()
        self.combo_device.addItem("CPU", "cpu")
        self.combo_device.addItem("CUDA", "cuda")
        self.combo_device.currentIndexChanged.connect(self._on_device_changed)
        layout.addWidget(self.combo_device)

        self.spin_conf = self._add_spin_row(layout, "置信度阈值 conf_thres", 0.70)
        self.spin_iou = self._add_spin_row(layout, "NMS 阈值 iou_thres", 0.70)

        scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(scroll_area, 1)

        self.btn_detect = QPushButton("开始目标检测")
        self.btn_detect.setObjectName("btnDetect")
        self.btn_detect.setCursor(Qt.PointingHandCursor)
        self.btn_detect.clicked.connect(self._on_detect)
        outer_layout.addWidget(self.btn_detect)

        self.btn_batch_detect = QPushButton("开始批量检测")
        self.btn_batch_detect.setObjectName("btnDetect")
        self.btn_batch_detect.setCursor(Qt.PointingHandCursor)
        self.btn_batch_detect.clicked.connect(self._on_batch_detect)
        outer_layout.addWidget(self.btn_batch_detect)

        self.btn_batch_cancel = QPushButton("取消当前批量任务")
        self.btn_batch_cancel.setObjectName("btnOpenDir")
        self.btn_batch_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_batch_cancel.clicked.connect(self._on_cancel_batch)
        self.btn_batch_cancel.setEnabled(False)
        outer_layout.addWidget(self.btn_batch_cancel)

        self.btn_open_dir = QPushButton("打开检测结果目录")
        self.btn_open_dir.setObjectName("btnOpenDir")
        self.btn_open_dir.setCursor(Qt.PointingHandCursor)
        self.btn_open_dir.setToolTip(f"打开保存目录 {DEFAULT_DETECT_OUTPUT_DIR}/")
        self.btn_open_dir.clicked.connect(self._on_open_save_dir)
        outer_layout.addWidget(self.btn_open_dir)

        self._on_mode_changed()

        return panel

    def _add_file_picker(self, parent_layout, title, placeholder, slot):
        label = QLabel(title)
        label.setObjectName("fieldLabel")
        parent_layout.addWidget(label)

        row = QHBoxLayout()
        row.setSpacing(8)
        line_edit = QLineEdit()
        line_edit.setReadOnly(True)
        line_edit.setPlaceholderText(placeholder)
        button = QPushButton("选择")
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(slot)
        row.addWidget(line_edit, 1)
        row.addWidget(button, 0)
        parent_layout.addLayout(row)
        return line_edit, button

    def _add_spin_row(self, parent_layout, title, default_value):
        label = QLabel(title)
        label.setObjectName("fieldLabel")
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        spin.setRange(0.0, 1.0)
        spin.setValue(default_value)
        spin.setKeyboardTracking(False)
        spin.setAlignment(Qt.AlignCenter)
        parent_layout.addWidget(label)
        parent_layout.addWidget(spin)
        return spin

    def _create_visual_panel(self):
        panel = QFrame()
        panel.setObjectName("panelCard")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("检测可视化")
        title.setObjectName("sectionTitle")
        hint = QLabel("左侧为输入图像，右侧显示检测框、类别与置信度。")
        hint.setObjectName("sectionHint")
        layout.addWidget(title)
        layout.addWidget(hint)

        image_layout = QHBoxLayout()
        image_layout.setSpacing(12)

        left_card, self.lbl_input_image = self._create_image_card(
            "待检测图像", "Input Image", "请选择待检测图像"
        )
        right_card, self.lbl_output_image = self._create_image_card(
            "检测结果图", "Detection Result", "检测结果将在此显示"
        )
        arrow = QLabel("→")
        arrow.setObjectName("flowArrow")
        arrow.setAlignment(Qt.AlignCenter)
        arrow.setFixedWidth(34)

        image_layout.addWidget(left_card, 1)
        image_layout.addWidget(arrow, 0)
        image_layout.addWidget(right_card, 1)
        layout.addLayout(image_layout, 1)
        return panel

    def _create_image_card(self, title, subtitle, placeholder):
        card = QFrame()
        card.setObjectName("imageCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setObjectName("imageTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("imageHint")
        card_layout.addWidget(title_label)
        card_layout.addWidget(subtitle_label)

        canvas_frame = QFrame()
        canvas_frame.setObjectName("imageCanvasFrame")
        canvas_layout = QVBoxLayout(canvas_frame)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        image_label = QLabel(placeholder)
        image_label.setObjectName("imageCanvas")
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(300, 360)
        image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_label.setWordWrap(True)
        canvas_layout.addWidget(image_label)
        card_layout.addWidget(canvas_frame, 1)
        return card, image_label

    def _create_stats_panel(self):
        panel = QFrame()
        panel.setObjectName("statsPanel")
        panel.setMinimumWidth(330)
        panel.setMaximumWidth(370)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("检测统计")
        title.setObjectName("sectionTitle")
        hint = QLabel("突出目标数量、置信度、类别分布和检测列表。")
        hint.setObjectName("sectionHint")
        layout.addWidget(title)
        layout.addWidget(hint)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        summary_grid = QGridLayout()
        summary_grid.setHorizontalSpacing(8)
        summary_grid.setVerticalSpacing(8)
        summary_items = [
            ("num_detections", "Objects", "0"),
            ("avg_confidence", "Avg Conf", "0.0000"),
            ("detect_time", "Time", "未检测"),
        ]
        for idx, (key, title_text, default_value) in enumerate(summary_items):
            summary_grid.addWidget(
                self._create_summary_card(key, title_text, default_value), 0, idx
            )
        layout.addLayout(summary_grid)

        class_title = QLabel("类别统计")
        class_title.setObjectName("fieldLabel")
        layout.addWidget(class_title)
        self.class_text = QTextEdit()
        self.class_text.setObjectName("classText")
        self.class_text.setReadOnly(True)
        self.class_text.setMaximumHeight(88)
        self.class_text.setText("暂无类别统计")
        layout.addWidget(self.class_text)

        detail_title = QLabel("检测框列表")
        detail_title.setObjectName("fieldLabel")
        layout.addWidget(detail_title)
        self.detail_text = QTextEdit()
        self.detail_text.setObjectName("detailText")
        self.detail_text.setReadOnly(True)
        self.detail_text.setFont(QFont("Consolas", 9))
        self.detail_text.setMinimumHeight(112)
        self.detail_text.setText("暂无检测结果")
        layout.addWidget(self.detail_text, 1)

        self.eval_note = QLabel("未提供标签，仅显示预测置信度统计。")
        self.eval_note.setObjectName("noteText")
        self.eval_note.setWordWrap(True)
        layout.addWidget(self.eval_note)

        eval_grid = QGridLayout()
        eval_grid.setHorizontalSpacing(8)
        eval_grid.setVerticalSpacing(8)
        eval_items = [
            ("precision", "Precision", "未提供标签"),
            ("recall", "Recall", "未提供标签"),
            ("f1_score", "F1-score", "未提供标签"),
            ("mAP", "AP@0.5", "未提供标签"),
        ]
        for idx, (key, title_text, default_value) in enumerate(eval_items):
            eval_grid.addWidget(
                self._create_eval_card(key, title_text, default_value),
                idx // 2,
                idx % 2,
            )
        layout.addLayout(eval_grid)

        path_title = QLabel("保存路径")
        path_title.setObjectName("fieldLabel")
        self.metric_labels["save_path"] = QLabel("未保存")
        self.metric_labels["save_path"].setObjectName("pathValue")
        self.metric_labels["save_path"].setWordWrap(True)
        layout.addWidget(path_title)
        layout.addWidget(self.metric_labels["save_path"])

        return panel

    def _create_summary_card(self, key, title, default_value):
        card = QFrame()
        card.setObjectName("summaryCard")
        card.setProperty("accent", "detection")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(4)
        name_label = QLabel(title)
        name_label.setObjectName("summaryName")
        value_label = QLabel(default_value)
        value_label.setObjectName("summaryValue")
        value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        card_layout.addWidget(name_label)
        card_layout.addWidget(value_label)
        self.metric_labels[key] = value_label
        return card

    def _create_eval_card(self, key, title, default_value):
        card = QFrame()
        card.setObjectName("evalMetricCard")
        card.setProperty("accent", "detection")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(4)
        name_label = QLabel(title)
        name_label.setObjectName("metricName")
        value_label = QLabel(default_value)
        value_label.setObjectName("pathValue")
        value_label.setWordWrap(True)
        card_layout.addWidget(name_label)
        card_layout.addWidget(value_label)
        self.metric_labels[key] = value_label
        return card

    def _create_log_panel(self):
        panel = QFrame()
        panel.setObjectName("logCard")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        title = QLabel("运行日志")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.log_text = QTextEdit()
        self.log_text.setObjectName("logText")
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setMaximumHeight(92)
        layout.addWidget(self.log_text)
        return panel

    def _on_device_changed(self):
        self.lbl_device_status.setText(self._current_device_label())

    def _current_device_preference(self):
        if not hasattr(self, "combo_device"):
            return "cpu"
        return self.combo_device.currentData() or "cpu"

    def _current_device_label(self):
        if not hasattr(self, "combo_device"):
            return "CPU"
        return self.combo_device.currentText() or "CPU"

    def _current_mode(self):
        if hasattr(self, "radio_batch") and self.radio_batch.isChecked():
            return "batch"
        return "single"

    def _is_batch_running(self):
        return self.batch_worker is not None and self.batch_worker.isRunning()

    def _is_detection_running(self):
        single_running = self.det_worker is not None and self.det_worker.isRunning()
        return single_running or self._is_batch_running()

    def _set_detection_controls_enabled(self, enabled):
        for widget in (
            self.radio_single,
            self.radio_batch,
            self.btn_select_image,
            self.btn_select_weight,
            self.btn_select_label,
            self.btn_select_batch_input,
            self.btn_select_batch_label,
            self.btn_select_batch_output,
            self.combo_device,
            self.spin_conf,
            self.spin_iou,
            self.btn_detect,
            self.btn_batch_detect,
            self.btn_open_dir,
        ):
            widget.setEnabled(enabled)
        self.btn_batch_cancel.setEnabled((not enabled) and self._is_batch_running())
        if enabled:
            self._on_mode_changed()

    def _on_mode_changed(self):
        if not hasattr(self, "single_section"):
            return
        is_batch = self._current_mode() == "batch"
        self.single_section.setVisible(not is_batch)
        self.batch_section.setVisible(is_batch)
        self.btn_detect.setVisible(not is_batch)
        self.btn_batch_detect.setVisible(is_batch)
        self.btn_batch_cancel.setVisible(is_batch)
        if hasattr(self, "progress_bar"):
            self.progress_bar.setVisible(is_batch)
        if not self._is_detection_running():
            self.lbl_detection_status.setText("批量待检测" if is_batch else "待检测")

    def _on_select_image(self):
        if self._is_detection_running():
            QMessageBox.information(self, "正在检测", "当前检测尚未结束，请等待完成后再切换图片。")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "选择检测图像", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*.*)"
        )
        if not path:
            return
        try:
            self.input_image = load_image(path)
            self.lbl_image_path.setText(path)
            self.lbl_image_path.setToolTip(path)
            show_image_on_label(self.lbl_input_image, self.input_image)
            self._log(f"[OK] 已加载检测图像: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "加载失败", str(e))
            self._log(f"[ERROR] {e}")

    def _on_select_weight(self):
        if self._is_detection_running():
            QMessageBox.information(self, "正在检测", "当前检测尚未结束，请等待完成后再切换权重。")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLOv9 权重文件", "",
            "权重文件 (*.pt *.pth *.weights);;所有文件 (*.*)"
        )
        if path:
            self.lbl_weight_path.setText(path)
            self.lbl_weight_path.setToolTip(path)
            self.lbl_weight_status.setText(os.path.basename(path))
            self._log(f"[OK] 已选择权重: {os.path.basename(path)}")

    def _on_select_label(self):
        if self._is_detection_running():
            QMessageBox.information(self, "正在检测", "当前检测尚未结束，请等待完成后再选择标签。")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "选择真实标签文件(可选)", "",
            "标签文件 (*.txt *.json *.xml *.yaml);;所有文件 (*.*)"
        )
        if path:
            self.lbl_label_path.setText(path)
            self.lbl_label_path.setToolTip(path)
            self._log(f"[OK] 已选择标签: {os.path.basename(path)}")

    def _on_select_batch_input_dir(self):
        if self._is_detection_running():
            QMessageBox.information(self, "正在检测", "当前检测尚未结束，请等待完成后再切换输入文件夹。")
            return

        path = QFileDialog.getExistingDirectory(self, "选择批量输入文件夹", "")
        if path:
            self.lbl_batch_input_dir.setText(path)
            self.lbl_batch_input_dir.setToolTip(path)
            image_count = len(scan_image_files(path))
            self._log(f"[OK] 已选择批量输入文件夹: {path} | 图像数: {image_count}")

    def _on_select_batch_label_dir(self):
        if self._is_detection_running():
            QMessageBox.information(self, "正在检测", "当前检测尚未结束，请等待完成后再切换标签文件夹。")
            return

        path = QFileDialog.getExistingDirectory(self, "选择批量标签文件夹（可选）", "")
        if path:
            self.lbl_batch_label_dir.setText(path)
            self.lbl_batch_label_dir.setToolTip(path)
            self._log(f"[OK] 已选择批量标签文件夹: {path}")

    def _on_select_batch_output_dir(self):
        if self._is_detection_running():
            QMessageBox.information(self, "正在检测", "当前检测尚未结束，请等待完成后再切换输出目录。")
            return

        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.lbl_batch_output_dir.text() or DEFAULT_DETECT_OUTPUT_DIR)
        if path:
            self.lbl_batch_output_dir.setText(path)
            self.lbl_batch_output_dir.setToolTip(path)
            self._log(f"[OK] 已选择输出目录: {path}")

    def _on_detect(self):
        if self._current_mode() == "batch":
            self._on_batch_detect()
            return
        if self.input_image is None:
            QMessageBox.warning(self, "提示", "请先选择一张检测图像！")
            return
        if self._is_detection_running():
            return

        weight_path = self.lbl_weight_path.text()
        if not weight_path:
            self._log("[INFO] 未选择权重，使用 yolov9-main/weights/yolov9-c.pt")

        conf_thres = self.spin_conf.value()
        iou_thres = self.spin_iou.value()
        device_preference = self._current_device_preference()
        device_label = self._current_device_label()
        self.lbl_detection_status.setText("检测中")
        self._log(
            f"[RUN] 开始目标检测 | 设备: {device_label} | conf={conf_thres:.2f}, iou={iou_thres:.2f}"
        )

        self._set_detection_controls_enabled(False)
        self.btn_detect.setText("检测中...")

        self.det_worker = DetectionWorker(
            self.input_image, weight_path, conf_thres, iou_thres, device_preference
        )
        self.det_worker.result_ready.connect(self._on_detection_finished)
        self.det_worker.error_occurred.connect(self._on_detection_error)
        self.det_worker.finished.connect(self._on_worker_finished)
        self.det_worker.finished.connect(self.det_worker.deleteLater)
        self.det_worker.start()

    def _on_batch_detect(self):
        if self._is_detection_running():
            return

        input_dir = self.lbl_batch_input_dir.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "提示", "请先选择有效的批量输入文件夹。")
            return

        image_files = scan_image_files(input_dir)
        if not image_files:
            QMessageBox.warning(
                self,
                "未找到图像",
                f"输入文件夹中没有可处理图像：{', '.join(sorted(IMAGE_EXTENSIONS))}",
            )
            return

        output_dir = self.lbl_batch_output_dir.text().strip() or DEFAULT_DETECT_OUTPUT_DIR
        label_dir = self.lbl_batch_label_dir.text().strip()
        if label_dir and not os.path.isdir(label_dir):
            QMessageBox.warning(self, "提示", "批量标签文件夹不存在，请重新选择或留空。")
            return

        weight_path = self.lbl_weight_path.text().strip()
        if not weight_path:
            self._log("[INFO] 未选择权重，使用 yolov9-main/weights/yolov9-c.pt")

        conf_thres = self.spin_conf.value()
        iou_thres = self.spin_iou.value()
        device_preference = self._current_device_preference()
        device_label = self._current_device_label()

        self._reset_progress()
        self.lbl_detection_status.setText("批量检测中")
        self.metric_labels["save_path"].setText("批量检测中，等待结果目录生成")
        self.class_text.setText("批量检测中")
        self.detail_text.setText("批量检测中")
        self.eval_note.setText("批量检测过程中，预览区显示最近一张成功检测的图像。")
        self._log(
            f"[RUN] 开始批量检测 | 图片数: {len(image_files)} | 设备: {device_label} | "
            f"conf={conf_thres:.2f}, iou={iou_thres:.2f}"
        )

        self.batch_worker = BatchDetectionWorker(
            image_files,
            input_dir,
            output_dir,
            label_dir,
            weight_path,
            conf_thres,
            iou_thres,
            device_preference,
            device_label,
        )
        self.batch_worker.log_message.connect(self._log)
        self.batch_worker.progress_changed.connect(self._on_batch_progress)
        self.batch_worker.preview_ready.connect(self._on_batch_preview)
        self.batch_worker.result_ready.connect(self._on_batch_finished)
        self.batch_worker.error_occurred.connect(self._on_batch_error)
        self.batch_worker.finished.connect(self._on_batch_worker_finished)
        self.batch_worker.finished.connect(self.batch_worker.deleteLater)

        self._set_detection_controls_enabled(False)
        self.btn_batch_detect.setText("批量检测中...")
        self.btn_batch_cancel.setEnabled(True)
        self.batch_worker.start()

    def _on_cancel_batch(self):
        if self._is_batch_running():
            self.batch_worker.request_cancel()
            self.btn_batch_cancel.setEnabled(False)
            self.lbl_detection_status.setText("正在取消")
            self._log("[CANCEL] 已请求取消批量检测，当前图片结束后停止。")

    def _on_batch_progress(self, stats):
        self.progress_bar.setValue(int(stats.get("percent", 0)))

    def _on_batch_preview(self, input_image, vis_image, detections, avg_conf, elapsed):
        self.input_image = input_image
        self.vis_image = vis_image
        self.detections = detections
        show_image_on_label(self.lbl_input_image, input_image)
        show_image_on_label(self.lbl_output_image, vis_image)
        metrics = compute_detection_metrics(detections, None)
        self._display_metrics(metrics, elapsed, self.metric_labels["save_path"].text())
        self._display_detailed_info(detections)

    def _on_batch_finished(self, result):
        summary = result.get("summary", {})
        records = result.get("records", [])
        batch_dir = result.get("batch_dir", "")
        summary_path = result.get("summary_path", "")

        self._set_detection_controls_enabled(True)
        self.btn_batch_detect.setText("开始批量检测")
        self.btn_batch_cancel.setEnabled(False)
        self.lbl_detection_status.setText("已取消" if summary.get("canceled") else "批量完成")
        self.metric_labels["save_path"].setText(batch_dir or "未保存")

        self._display_batch_summary(summary, records)
        self._log(
            f"[DONE] 批量检测完成 | 成功={summary.get('success', 0)} | "
            f"失败={summary.get('failed', 0)} | 已处理={summary.get('completed', 0)}/{summary.get('total', 0)}"
        )
        if summary_path:
            self._log(f"[OK] 批量汇总报告: {summary_path}")

    def _on_batch_error(self, error_msg):
        self._set_detection_controls_enabled(True)
        self.btn_batch_detect.setText("开始批量检测")
        self.btn_batch_cancel.setEnabled(False)
        self.lbl_detection_status.setText("批量失败")
        self._log(f"[ERROR] 批量检测失败: {error_msg}")
        QMessageBox.critical(self, "批量检测失败", f"批量检测过程中发生错误:\n\n{error_msg}")

    def _on_detection_finished(self, detections, vis_image, elapsed):
        self.detections = detections
        self.vis_image = vis_image

        self._set_detection_controls_enabled(True)
        self.btn_detect.setText("开始目标检测")
        self.lbl_detection_status.setText("已完成")

        show_image_on_label(self.lbl_output_image, vis_image)

        image_path = self.lbl_image_path.text().strip()
        image_stem = _safe_dir_name(Path(image_path).stem, "single_image")
        output_root = self.lbl_batch_output_dir.text().strip() or DEFAULT_DETECT_OUTPUT_DIR
        run_dir = os.path.join(output_root, "单张文件", image_stem)
        images_dir = os.path.join(run_dir, "images")
        labels_dir = os.path.join(run_dir, "labels")
        reports_dir = os.path.join(run_dir, "reports")
        try:
            os.makedirs(reports_dir, exist_ok=True)
            save_path = save_image_to_path(
                vis_image,
                os.path.join(images_dir, f"{image_stem}_detect.jpg"),
                overwrite=True,
            )
            pred_path = os.path.join(labels_dir, f"{image_stem}_pred.txt")
            _write_prediction_txt(detections, pred_path)
            params_path = os.path.join(reports_dir, "single_detection_params.txt")
            _write_batch_params(
                {
                    "input image": image_path,
                    "output directory": run_dir,
                    "weight path": self.lbl_weight_path.text().strip() or "default",
                    "device": self._current_device_label(),
                    "conf_thres": f"{self.spin_conf.value():.2f}",
                    "iou_thres": f"{self.spin_iou.value():.2f}",
                    "prediction txt": pred_path,
                },
                params_path,
            )
            self._log(f"[OK] 检测完成 | 耗时: {elapsed:.3f}s")
            self._log(f"[OK] 结果已保存: {save_path}")
        except Exception as e:
            save_path = f"保存失败: {e}"
            self._log(f"[ERROR] {save_path}")

        gt_labels = None
        self._last_label_status = "none"
        label_path = self.lbl_label_path.text().strip()
        if label_path:
            try:
                gt_labels = load_detection_labels(label_path, self.input_image.shape)
                self._last_label_status = f"loaded:{len(gt_labels)}"
                self._log(f"[OK] 已解析标签: {len(gt_labels)} 个目标")
            except Exception as e:
                self._last_label_status = f"error:{e}"
                self._log(f"[ERROR] 标签解析失败: {e}")
        metrics = compute_detection_metrics(detections, gt_labels)
        self._display_metrics(metrics, elapsed, save_path)
        self._display_detailed_info(detections)

    def _on_detection_error(self, error_msg):
        self._set_detection_controls_enabled(True)
        self.btn_detect.setText("开始目标检测")
        self.lbl_detection_status.setText("检测失败")
        self._log(f"[ERROR] 检测失败: {error_msg}")
        QMessageBox.critical(self, "检测失败", f"检测过程中发生错误:\n\n{error_msg}")

    def _on_worker_finished(self):
        worker = self.sender()
        if self.det_worker is worker:
            self.det_worker = None

    def _on_batch_worker_finished(self):
        worker = self.sender()
        if self.batch_worker is worker:
            self.batch_worker = None

    def _metric_text(self, value, empty_text="未提供标签"):
        if value is None:
            return empty_text
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return empty_text

    def _display_metrics(self, metrics, elapsed, save_path):
        classes = metrics.get("classes", [])
        counts = Counter(classes)
        avg_confidence = metrics.get("avg_confidence", 0.0)

        self.metric_labels["num_detections"].setText(str(metrics["num_detections"]))
        self.metric_labels["avg_confidence"].setText(f"{avg_confidence:.4f}")
        self.metric_labels["detect_time"].setText(f"{elapsed * 1000:.1f} ms")
        self.metric_labels["save_path"].setText(str(save_path))

        if counts:
            self.class_text.setText(
                "\n".join(f"{name} × {count}" for name, count in counts.items())
            )
        else:
            self.class_text.setText("暂无类别统计")

        self.eval_note.setText("Precision / Recall / F1 / AP@0.5 为演示指标，按旧逻辑随机生成。")

        self.metric_labels["precision"].setText(self._metric_text(metrics.get("precision")))
        self.metric_labels["recall"].setText(self._metric_text(metrics.get("recall")))
        self.metric_labels["f1_score"].setText(self._metric_text(metrics.get("f1_score")))
        self.metric_labels["mAP"].setText(
            self._metric_text(metrics.get("ap50", metrics.get("mAP")))
        )

    def _display_batch_summary(self, summary, records):
        total_detections = summary.get("total_detections", 0)
        avg_confidence = summary.get("avg_confidence")
        total_elapsed = summary.get("total_elapsed", 0.0)

        self.metric_labels["num_detections"].setText(str(total_detections))
        self.metric_labels["avg_confidence"].setText(self._metric_text(avg_confidence, "0.0000"))
        self.metric_labels["detect_time"].setText(f"{total_elapsed:.3f} s")

        class_totals = Counter()
        for record in records:
            class_totals.update(record.get("class_counter", Counter()))
        if class_totals:
            self.class_text.setText(
                "\n".join(f"{name} × {count}" for name, count in sorted(class_totals.items()))
            )
        else:
            self.class_text.setText("暂无类别统计")

        lines = []
        for index, record in enumerate(records[:80], 1):
            status = "OK" if record.get("status") == "success" else "FAIL"
            lines.append(
                f"{index:03d} | {status:<4s} | {record.get('image_name', ''):<28.28s} | "
                f"det={record.get('detected_count', 0)} | "
                f"time={_fmt_value(record.get('inference_time'), 3)}s"
            )
        if len(records) > 80:
            lines.append(f"... 其余 {len(records) - 80} 条详见 batch_detection_summary.txt")
        self.detail_text.setText("\n".join(lines) if lines else "暂无检测结果")

        metric_records = [
            record
            for record in records
            if record.get("precision") not in ("N/A", None, "")
        ]
        label_records = [
            record
            for record in records
            if record.get("tp") not in ("N/A", None, "")
        ]
        if metric_records:
            avg_precision = sum(float(record.get("precision", 0.0)) for record in metric_records) / len(metric_records)
            avg_recall = sum(float(record.get("recall", 0.0)) for record in metric_records) / len(metric_records)
            avg_f1 = sum(float(record.get("f1_score", 0.0)) for record in metric_records) / len(metric_records)
            avg_ap50 = sum(float(record.get("ap50", 0.0)) for record in metric_records) / len(metric_records)
            self.eval_note.setText("批量 Precision / Recall / F1 / AP@0.5 为演示指标，按旧逻辑随机生成。")
            self.metric_labels["precision"].setText(self._metric_text(avg_precision))
            self.metric_labels["recall"].setText(self._metric_text(avg_recall))
            self.metric_labels["f1_score"].setText(self._metric_text(avg_f1))
            self.metric_labels["mAP"].setText(self._metric_text(avg_ap50))
        else:
            self.eval_note.setText("尚未生成演示指标。")
            for key in ("precision", "recall", "f1_score", "mAP"):
                self.metric_labels[key].setText("未检测")

        if label_records:
            total_tp = sum(int(record.get("tp") or 0) for record in label_records)
            total_fp = sum(int(record.get("fp") or 0) for record in label_records)
            total_fn = sum(int(record.get("fn") or 0) for record in label_records)
            self.eval_note.setText(
                f"批量指标为随机演示值；标签匹配统计：TP={total_tp}, FP={total_fp}, FN={total_fn}。"
            )

    def _reset_progress(self):
        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(0)

    def _display_detailed_info(self, detections):
        self.detail_text.clear()
        if not detections:
            self.detail_text.append("未检测到任何目标。")
            return

        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det["bbox"]
            self.detail_text.append(
                f"{i:02d} | {det['class']:<12s} | "
                f"{det['confidence']:.4f} | box=({x1}, {y1}, {x2}, {y2})"
            )

    def _on_open_save_dir(self):
        current_path = self.metric_labels.get("save_path")
        candidate = current_path.text().strip() if current_path else ""
        if candidate and os.path.isfile(candidate):
            save_dir = os.path.dirname(candidate)
        elif candidate and os.path.isdir(candidate):
            save_dir = candidate
        elif self._current_mode() == "batch" and self.lbl_batch_output_dir.text().strip():
            save_dir = self.lbl_batch_output_dir.text().strip()
        else:
            save_dir = DEFAULT_DETECT_OUTPUT_DIR
        os.makedirs(save_dir, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(save_dir)
        else:
            subprocess.Popen(["xdg-open", save_dir])

    def _log(self, message):
        self.log_text.append(message)
        if self.log_signal:
            self.log_signal.emit(message)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.input_image is not None:
            show_image_on_label(self.lbl_input_image, self.input_image)
        if self.vis_image is not None:
            show_image_on_label(self.lbl_output_image, self.vis_image)
