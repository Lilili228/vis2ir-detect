"""
Tab 2: YOLOv9 目标检测界面
"""

import os
import subprocess
import sys

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QGroupBox, QTextEdit, QFileDialog, QMessageBox,
    QGridLayout, QLineEdit, QFrame, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from core.utils import load_image, show_image_on_label, save_image
from core.yolov9_detection_backend import (
    compute_detection_metrics,
    load_detection_labels,
    run_yolov9_detection,
)


LOCAL_STYLE = """
QPushButton#btnSelectImage, QPushButton#btnSelectWeight, QPushButton#btnSelectLabel {
    padding: 8px 16px;
    border: 1px solid #cfd8e3;
    border-radius: 6px;
    background-color: #f8f9fb;
    color: #34495e;
    font-weight: 500;
}
QPushButton#btnSelectImage:hover, QPushButton#btnSelectWeight:hover, QPushButton#btnSelectLabel:hover {
    background-color: #e3edfd;
    border-color: #1a73e8;
    color: #1a73e8;
}

QPushButton#btnDetect {
    padding: 10px 28px;
    border: none;
    border-radius: 8px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #10b981, stop:1 #059669);
    color: #ffffff;
    font-size: 14px;
    font-weight: 700;
    min-height: 24px;
    letter-spacing: 1px;
}
QPushButton#btnDetect:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #34d399, stop:1 #10b981);
}
QPushButton#btnDetect:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #059669, stop:1 #047857);
}
QPushButton#btnDetect:disabled {
    background: #b0bec5;
    color: #eceff1;
}

QPushButton#btnOpenDir {
    padding: 8px 16px;
    border: 1px solid #b0bec5;
    border-radius: 6px;
    background-color: transparent;
    color: #546e7a;
    font-weight: 500;
}
QPushButton#btnOpenDir:hover {
    background-color: #eceff1;
    border-color: #78909c;
    color: #37474f;
}

QFrame#imageCardLeft, QFrame#imageCardRight {
    background-color: #1e272e;
    border: 2px solid #2d3a42;
    border-radius: 12px;
}
QFrame#imageCardLeft:hover, QFrame#imageCardRight:hover {
    border-color: #4a6274;
}

QLabel#lblInputImage, QLabel#lblOutputImage {
    color: #8899a6;
    font-size: 13px;
    background-color: transparent;
    border: none;
}

QFrame#metricCard {
    background-color: #f8fafc;
    border: 1px solid #e8ecf1;
    border-radius: 8px;
}

QLabel#metricValue {
    color: #065f46;
    font-size: 13px;
    font-weight: 700;
    font-family: "Consolas", "JetBrains Mono", monospace;
}

QLabel#metricLabel {
    color: #64748b;
    font-size: 11px;
}

QFrame#divider {
    background-color: #e2e8f0;
    max-width: 1px;
}
"""


class DetectionWorker(QThread):
    """YOLOv9 检测工作线程"""
    finished = pyqtSignal(list, object, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, image, weight_path, conf_thres, iou_thres):
        super().__init__()
        self.image = image
        self.weight_path = weight_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def run(self):
        try:
            detections, vis_image, elapsed = run_yolov9_detection(
                self.image, self.weight_path,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
            )
            self.finished.emit(detections, vis_image, elapsed)
        except Exception as e:
            self.error_occurred.emit(str(e))


class DetectionTab(QWidget):
    """YOLOv9 目标检测 Tab"""

    def __init__(self, log_signal=None):
        super().__init__()
        self.log_signal = log_signal
        self.input_image = None
        self.vis_image = None
        self.detections = None
        self.det_worker = None
        self.setStyleSheet(LOCAL_STYLE)
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(12)

        # ========== 顶部操作栏 ==========
        toolbar = QFrame()
        toolbar.setObjectName("toolbar")
        toolbar.setStyleSheet("""
            QFrame#toolbar {
                background-color: #ffffff;
                border: 1px solid #e2e6ed;
                border-radius: 12px;
                padding: 4px;
            }
        """)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(16, 12, 16, 12)
        toolbar_layout.setSpacing(12)

        # 图像选择
        img_area = QHBoxLayout()
        img_area.setSpacing(8)
        lbl_img = QLabel("📷 检测图像")
        lbl_img.setStyleSheet("font-weight:600; color:#334155; font-size:12px;")
        self.lbl_image_path = QLineEdit()
        self.lbl_image_path.setReadOnly(True)
        self.lbl_image_path.setPlaceholderText("请选择图像...")
        self.lbl_image_path.setMinimumWidth(180)
        self.btn_select_image = QPushButton("选择")
        self.btn_select_image.setObjectName("btnSelectImage")
        self.btn_select_image.setCursor(Qt.PointingHandCursor)
        self.btn_select_image.clicked.connect(self._on_select_image)

        img_area.addWidget(lbl_img)
        img_area.addWidget(self.lbl_image_path, 1)
        img_area.addWidget(self.btn_select_image)

        toolbar_layout.addLayout(img_area, 3)

        sep1 = QFrame()
        sep1.setObjectName("divider")
        sep1.setFrameShape(QFrame.VLine)
        toolbar_layout.addWidget(sep1)

        # 权重选择
        weight_area = QHBoxLayout()
        weight_area.setSpacing(8)
        lbl_weight = QLabel("🧠 权重文件")
        lbl_weight.setStyleSheet("font-weight:600; color:#334155; font-size:12px;")
        self.lbl_weight_path = QLineEdit()
        self.lbl_weight_path.setReadOnly(True)
        self.lbl_weight_path.setPlaceholderText("选择 best.pt ...")
        self.lbl_weight_path.setMinimumWidth(160)
        self.btn_select_weight = QPushButton("选择")
        self.btn_select_weight.setObjectName("btnSelectWeight")
        self.btn_select_weight.setCursor(Qt.PointingHandCursor)
        self.btn_select_weight.clicked.connect(self._on_select_weight)

        weight_area.addWidget(lbl_weight)
        weight_area.addWidget(self.lbl_weight_path, 1)
        weight_area.addWidget(self.btn_select_weight)

        toolbar_layout.addLayout(weight_area, 2)

        sep2 = QFrame()
        sep2.setObjectName("divider")
        sep2.setFrameShape(QFrame.VLine)
        toolbar_layout.addWidget(sep2)

        # 标签 + 阈值
        label_area = QHBoxLayout()
        label_area.setSpacing(8)
        lbl_label = QLabel("🏷️ 标签")
        lbl_label.setStyleSheet("font-weight:600; color:#334155; font-size:12px;")
        self.lbl_label_path = QLineEdit()
        self.lbl_label_path.setReadOnly(True)
        self.lbl_label_path.setPlaceholderText("可选标签...")
        self.lbl_label_path.setMinimumWidth(120)
        self.btn_select_label = QPushButton("选择")
        self.btn_select_label.setObjectName("btnSelectLabel")
        self.btn_select_label.setCursor(Qt.PointingHandCursor)
        self.btn_select_label.clicked.connect(self._on_select_label)

        label_area.addWidget(lbl_label)
        label_area.addWidget(self.lbl_label_path, 1)
        label_area.addWidget(self.btn_select_label)

        toolbar_layout.addLayout(label_area, 2)

        sep3 = QFrame()
        sep3.setObjectName("divider")
        sep3.setFrameShape(QFrame.VLine)
        toolbar_layout.addWidget(sep3)

        # 检测按钮
        detect_area = QHBoxLayout()
        detect_area.setSpacing(10)

        conf_label = QLabel("置信度")
        conf_label.setStyleSheet("font-weight:600; color:#334155; font-size:12px;")
        self.lbl_conf = QLineEdit("0.70")
        self.lbl_conf.setFixedWidth(60)
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        self.lbl_conf.setToolTip("检测置信度阈值 (0.0 ~ 1.0)")

        self.btn_detect = QPushButton("▶  开始目标检测")
        self.btn_detect.setObjectName("btnDetect")
        self.btn_detect.setCursor(Qt.PointingHandCursor)
        self.btn_detect.clicked.connect(self._on_detect)

        self.btn_open_dir = QPushButton("📂")
        self.btn_open_dir.setObjectName("btnOpenDir")
        self.btn_open_dir.setToolTip("打开保存目录 D:/YOLOv9_Detect_Result/")
        self.btn_open_dir.setCursor(Qt.PointingHandCursor)
        self.btn_open_dir.setFixedWidth(40)
        self.btn_open_dir.clicked.connect(self._on_open_save_dir)

        detect_area.addWidget(conf_label)
        detect_area.addWidget(self.lbl_conf)
        detect_area.addWidget(self.btn_detect)
        detect_area.addWidget(self.btn_open_dir)

        toolbar_layout.addLayout(detect_area, 3)

        main_layout.addWidget(toolbar)

        # ========== 图像显示区域 ==========
        image_layout = QHBoxLayout()
        image_layout.setSpacing(14)

        # 左侧：待检测图像
        left_card = QFrame()
        left_card.setObjectName("imageCardLeft")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        left_header = QLabel("  待检测图像")
        left_header.setStyleSheet(
            "color: #94a3b8; font-size: 12px; font-weight: 600; "
            "background-color: rgba(30,39,46,0.95); padding: 10px 0px; "
            "border-top-left-radius: 12px; border-top-right-radius: 12px;"
        )
        left_header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.lbl_input_image = QLabel()
        self.lbl_input_image.setObjectName("lblInputImage")
        self.lbl_input_image.setAlignment(Qt.AlignCenter)
        self.lbl_input_image.setMinimumSize(420, 380)
        self.lbl_input_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_input_image.setText("暂无图像\n\n点击上方「选择」按钮\n加载待检测图像")

        left_layout.addWidget(left_header)
        left_layout.addWidget(self.lbl_input_image, 1)

        image_layout.addWidget(left_card, 1)

        # 中间箭头
        arrow_label = QLabel()
        arrow_label.setAlignment(Qt.AlignCenter)
        arrow_label.setFixedWidth(40)
        arrow_label.setText("→")
        arrow_label.setStyleSheet(
            "font-size: 32px; color: #10b981; font-weight: bold;"
        )
        image_layout.addWidget(arrow_label)

        # 右侧：检测结果
        right_card = QFrame()
        right_card.setObjectName("imageCardRight")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        right_header = QLabel("  检测结果可视化")
        right_header.setStyleSheet(
            "color: #94a3b8; font-size: 12px; font-weight: 600; "
            "background-color: rgba(30,39,46,0.95); padding: 10px 0px; "
            "border-top-left-radius: 12px; border-top-right-radius: 12px;"
        )
        right_header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.lbl_output_image = QLabel()
        self.lbl_output_image.setObjectName("lblOutputImage")
        self.lbl_output_image.setAlignment(Qt.AlignCenter)
        self.lbl_output_image.setMinimumSize(420, 380)
        self.lbl_output_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_output_image.setText("等待检测\n\n选择权重后点击\n「开始目标检测」按钮")

        right_layout.addWidget(right_header)
        right_layout.addWidget(self.lbl_output_image, 1)

        image_layout.addWidget(right_card, 1)

        main_layout.addLayout(image_layout, 1)

        # ========== 底部：指标 + 日志 ==========
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(14)

        # 指标面板
        metrics_panel = QFrame()
        metrics_panel.setObjectName("metricCard")
        metrics_panel.setStyleSheet("""
            QFrame#metricCard {
                background-color: #ffffff;
                border: 1px solid #e2e6ed;
                border-radius: 10px;
            }
        """)
        metrics_layout = QVBoxLayout(metrics_panel)
        metrics_layout.setContentsMargins(16, 10, 16, 10)
        metrics_layout.setSpacing(0)

        metrics_title = QLabel("🎯 检测结果信息")
        metrics_title.setStyleSheet(
            "color: #1e293b; font-size: 13px; font-weight: 700; padding-bottom: 8px;"
        )
        metrics_layout.addWidget(metrics_title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(6)

        self.metric_labels = {}
        items = [
            ('num_detections', '目标数量', '#3b82f6'),
            ('classes', '类别列表', '#8b5cf6'),
            ('avg_confidence', '平均置信度', '#f59e0b'),
            ('precision', 'Precision', '#10b981'),
            ('recall', 'Recall', '#ef4444'),
            ('f1_score', 'F1-Score', '#6366f1'),
            ('mAP', 'mAP', '#ec4899'),
            ('detect_time', '检测耗时', '#14b8a6'),
            ('save_path', '保存路径', '#6366f1'),
        ]
        for idx, (key, name, color) in enumerate(items):
            row = idx // 2
            col_base = (idx % 2) * 3

            name_lbl = QLabel(f"  {name}")
            name_lbl.setObjectName("metricLabel")
            name_lbl.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 600;")

            val_lbl = QLabel("—")
            val_lbl.setObjectName("metricValue")
            if key == 'save_path':
                val_lbl.setStyleSheet(
                    "color: #6366f1; font-size: 10px; font-weight: 600; "
                    "font-family: Consolas, monospace;"
                )
                val_lbl.setMaximumWidth(260)
            elif key == 'classes':
                val_lbl.setStyleSheet(
                    f"color: {color}; font-size: 11px; font-weight: 600;"
                )
                val_lbl.setMaximumWidth(200)
            else:
                val_lbl.setStyleSheet(
                    f"color: {color}; font-size: 14px; font-weight: 700; "
                    "font-family: Consolas, JetBrains Mono, monospace;"
                )

            self.metric_labels[key] = val_lbl

            grid.addWidget(name_lbl, row, col_base)
            grid.addWidget(val_lbl, row, col_base + 1)

        for col in range(6):
            grid.setColumnStretch(col, 1)

        metrics_layout.addLayout(grid)
        metrics_layout.addStretch()

        bottom_layout.addWidget(metrics_panel, 5)

        # 详细信息 + 日志 合并
        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)

        # 详细检测信息
        detail_frame = QFrame()
        detail_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f9ff;
                border: 1px solid #bae6fd;
                border-radius: 8px;
            }
        """)
        detail_inner = QVBoxLayout(detail_frame)
        detail_inner.setContentsMargins(12, 8, 12, 8)

        detail_header = QLabel("📋 详细检测信息")
        detail_header.setStyleSheet(
            "color: #0369a1; font-size: 12px; font-weight: 700; padding-bottom: 2px;"
        )
        detail_inner.addWidget(detail_header)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setFont(QFont("Consolas", 9))
        self.detail_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8fafc;
                color: #0f172a;
                border: 1px solid #e0f2fe;
                border-radius: 4px;
                font-size: 11px;
            }
        """)
        self.detail_text.setMaximumHeight(80)
        detail_inner.addWidget(self.detail_text)

        right_panel.addWidget(detail_frame, 1)

        # 日志
        log_frame = QFrame()
        log_frame.setStyleSheet("""
            QFrame {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
            }
        """)
        log_inner = QVBoxLayout(log_frame)
        log_inner.setContentsMargins(12, 8, 12, 8)

        log_header = QLabel("📜 运行日志")
        log_header.setStyleSheet(
            "color: #94a3b8; font-size: 12px; font-weight: 700; padding-bottom: 2px;"
        )
        log_inner.addWidget(log_header)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #0f172a;
                color: #cbd5e1;
                border: 1px solid #334155;
                border-radius: 4px;
                font-size: 11px;
            }
            QScrollBar:vertical { background: #0f172a; width: 6px; }
            QScrollBar::handle:vertical { background: #475569; border-radius: 3px; }
        """)
        self.log_text.setMaximumHeight(80)
        log_inner.addWidget(self.log_text)

        right_panel.addWidget(log_frame, 1)

        # 包含右侧面板
        right_container = QFrame()
        right_container.setLayout(right_panel)
        bottom_layout.addWidget(right_container, 3)

        main_layout.addLayout(bottom_layout, 0)

    # ========== 槽函数 ==========

    def _on_select_image(self):
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
            self.lbl_input_image.setStyleSheet("border: none; background: transparent;")
            self._log(f"[✓] 已加载检测图像: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "加载失败", str(e))
            self._log(f"[✗] {e}")

    def _on_select_weight(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLOv9 权重文件", "",
            "权重文件 (*.pt *.pth *.weights);;所有文件 (*.*)"
        )
        if path:
            self.lbl_weight_path.setText(path)
            self.lbl_weight_path.setToolTip(path)
            self._log(f"[✓] 已选择权重: {os.path.basename(path)}")

    def _on_select_label(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择真实标签文件(可选)", "",
            "标签文件 (*.txt *.json *.xml *.yaml);;所有文件 (*.*)"
        )
        if path:
            self.lbl_label_path.setText(path)
            self.lbl_label_path.setToolTip(path)
            self._log(f"[✓] 已选择标签: {os.path.basename(path)}")

    def _on_detect(self):
        if self.input_image is None:
            QMessageBox.warning(self, "提示", "请先选择一张检测图像！")
            return

        weight_path = self.lbl_weight_path.text()
        if not weight_path:
            self._log("[!] 未选择权重，使用 yolov9-main/weights/yolov9-c.pt")

        try:
            conf_thres = float(self.lbl_conf.text())
        except ValueError:
            conf_thres = 0.70
            self.lbl_conf.setText("0.70")

        self._log(f"[▶] 开始目标检测 | conf={conf_thres}")

        self.btn_detect.setEnabled(False)
        self.btn_detect.setText("⏳ 检测中...")

        self.det_worker = DetectionWorker(
            self.input_image, weight_path, conf_thres, 0.70
        )
        self.det_worker.finished.connect(self._on_detection_finished)
        self.det_worker.error_occurred.connect(self._on_detection_error)
        self.det_worker.start()

    def _on_detection_finished(self, detections, vis_image, elapsed):
        self.detections = detections
        self.vis_image = vis_image

        self.btn_detect.setEnabled(True)
        self.btn_detect.setText("▶  开始目标检测")

        show_image_on_label(self.lbl_output_image, vis_image)
        self.lbl_output_image.setStyleSheet("border: none; background: transparent;")

        save_dir = "D:/YOLOv9_Detect_Result"
        try:
            save_path = save_image(vis_image, save_dir, prefix="detect_result")
            self._log(f"[✓] 检测完成 | 耗时: {elapsed:.3f}s")
            self._log(f"[✓] 结果已保存: {save_path}")
        except Exception as e:
            save_path = f"保存失败: {e}"
            self._log(f"[✗] {save_path}")

        gt_labels = None
        label_path = self.lbl_label_path.text().strip()
        if label_path:
            try:
                gt_labels = load_detection_labels(label_path, self.input_image.shape)
                self._log(f"[OK] 已解析标签: {len(gt_labels)} 个目标")
            except Exception as e:
                self._log(f"[ERROR] 标签解析失败: {e}")
        metrics = compute_detection_metrics(detections, gt_labels)
        self._display_metrics(metrics, elapsed, save_path)
        self._display_detailed_info(detections)

    def _on_detection_error(self, error_msg):
        self.btn_detect.setEnabled(True)
        self.btn_detect.setText("▶  开始目标检测")
        self._log(f"[✗] 检测失败: {error_msg}")
        QMessageBox.critical(self, "检测失败", f"检测过程中发生错误:\n\n{error_msg}")

    def _display_metrics(self, metrics, elapsed, save_path):
        self.metric_labels['num_detections'].setText(
            f"{metrics['num_detections']} 个"
        )
        self.metric_labels['classes'].setText(
            ', '.join(metrics['classes']) if metrics['classes'] else '无'
        )
        self.metric_labels['avg_confidence'].setText(
            f"{metrics['avg_confidence']:.4f}" if metrics['avg_confidence'] else "—"
        )
        self.metric_labels['detect_time'].setText(f"{elapsed:.3f} s")
        self.metric_labels['save_path'].setText(str(save_path))

        for key in ['precision', 'recall', 'f1_score', 'mAP']:
            val = metrics.get(key)
            if val is not None:
                self.metric_labels[key].setText(f"{val:.4f}")
            else:
                self.metric_labels[key].setText("需标签")
                self.metric_labels[key].setStyleSheet(
                    "color: #94a3b8; font-size: 11px; font-weight: 500;"
                )

    def _display_detailed_info(self, detections):
        self.detail_text.clear()
        if not detections:
            self.detail_text.append("未检测到任何目标。")
            return

        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det['bbox']
            self.detail_text.append(
                f"  #{i:<2}  {det['class']:<14s}  "
                f"conf = {det['confidence']:.4f}  "
                f"box = ({x1:>4}, {y1:>4}, {x2:>4}, {y2:>4})"
            )

    def _on_open_save_dir(self):
        save_dir = "D:/YOLOv9_Detect_Result"
        os.makedirs(save_dir, exist_ok=True)
        if sys.platform == 'win32':
            os.startfile(save_dir)
        else:
            subprocess.Popen(['xdg-open', save_dir])

    def _log(self, message):
        self.log_text.append(message)
        if self.log_signal:
            self.log_signal.emit(message)
