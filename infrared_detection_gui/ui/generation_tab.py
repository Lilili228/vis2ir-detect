"""
Tab 1: 可见光图像 → 红外图像生成界面
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
from PyQt5.QtGui import QFont, QPixmap, QIcon

from core.utils import load_image, show_image_on_label, save_image
from core.metrics import calculate_metrics
from core.ir_generation import generate_ir_algorithm_1, generate_ir_algorithm_2


# ── 局部样式 ─────────────────────────────────────────────────────
LOCAL_STYLE = """
/* 操作按钮 - 蓝色 */
QPushButton#btnSelectInput, QPushButton#btnSelectGT {
    padding: 8px 16px;
    border: 1px solid #cfd8e3;
    border-radius: 6px;
    background-color: #f8f9fb;
    color: #34495e;
    font-weight: 500;
}
QPushButton#btnSelectInput:hover, QPushButton#btnSelectGT:hover {
    background-color: #e3edfd;
    border-color: #1a73e8;
    color: #1a73e8;
}

/* 生成按钮 - 橙色高亮 */
QPushButton#btnGenerate {
    padding: 10px 28px;
    border: none;
    border-radius: 8px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #ff7b42, stop:1 #e8612d);
    color: #ffffff;
    font-size: 14px;
    font-weight: 700;
    min-height: 24px;
    letter-spacing: 1px;
}
QPushButton#btnGenerate:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #ff8f5e, stop:1 #f06b38);
}
QPushButton#btnGenerate:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #e8612d, stop:1 #d4511f);
}
QPushButton#btnGenerate:disabled {
    background: #b0bec5;
    color: #eceff1;
}

/* 打开目录按钮 */
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

/* 图像显示区域 - 深色卡片 */
QFrame#imageCardLeft, QFrame#imageCardRight {
    background-color: #1e272e;
    border: 2px solid #2d3a42;
    border-radius: 12px;
}
QFrame#imageCardLeft:hover, QFrame#imageCardRight:hover {
    border-color: #4a6274;
}

/* 图像标签 */
QLabel#lblInputImage, QLabel#lblOutputImage {
    color: #8899a6;
    font-size: 13px;
    background-color: transparent;
    border: none;
}

/* 指标卡片 */
QFrame#metricCard {
    background-color: #f8fafc;
    border: 1px solid #e8ecf1;
    border-radius: 8px;
    padding: 2px;
}

QLabel#metricValue {
    color: #1e40af;
    font-size: 13px;
    font-weight: 700;
    font-family: "Consolas", "JetBrains Mono", monospace;
}

QLabel#metricLabel {
    color: #64748b;
    font-size: 11px;
}

/* 分割线 */
QFrame#divider {
    background-color: #e2e8f0;
    max-height: 1px;
}
"""


class GenerationWorker(QThread):
    """红外图像生成工作线程"""
    finished = pyqtSignal(object, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, algorithm_id, input_image):
        super().__init__()
        self.algorithm_id = algorithm_id
        self.input_image = input_image

    def run(self):
        try:
            if self.algorithm_id == 1:
                output, elapsed = generate_ir_algorithm_1(self.input_image)
            else:
                output, elapsed = generate_ir_algorithm_2(self.input_image)
            self.finished.emit(output, elapsed)
        except Exception as e:
            self.error_occurred.emit(str(e))


class GenerationTab(QWidget):
    """可见光→红外生成 Tab"""

    def __init__(self, log_signal=None):
        super().__init__()
        self.log_signal = log_signal
        self.input_image = None
        self.generated_image = None
        self.gt_image = None
        self.gen_worker = None
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

        # 左侧：文件选择
        file_area = QHBoxLayout()
        file_area.setSpacing(8)

        lbl_input = QLabel("📷 输入图像")
        lbl_input.setStyleSheet("font-weight:600; color:#334155; font-size:12px;")
        self.lbl_input_path = QLineEdit()
        self.lbl_input_path.setReadOnly(True)
        self.lbl_input_path.setPlaceholderText("请选择可见光图像...")
        self.lbl_input_path.setMinimumWidth(200)
        self.btn_select_input = QPushButton("选择")
        self.btn_select_input.setObjectName("btnSelectInput")
        self.btn_select_input.setCursor(Qt.PointingHandCursor)
        self.btn_select_input.clicked.connect(self._on_select_input)

        file_area.addWidget(lbl_input)
        file_area.addWidget(self.lbl_input_path, 1)
        file_area.addWidget(self.btn_select_input)

        toolbar_layout.addLayout(file_area, 2)

        # 分隔
        sep1 = QFrame()
        sep1.setObjectName("divider")
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("background-color:#e2e8f0; max-width:1px;")
        toolbar_layout.addWidget(sep1)

        # 中间：标签图像
        gt_area = QHBoxLayout()
        gt_area.setSpacing(8)

        lbl_gt = QLabel("🏷️ 标签图像")
        lbl_gt.setStyleSheet("font-weight:600; color:#334155; font-size:12px;")
        self.lbl_gt_path = QLineEdit()
        self.lbl_gt_path.setReadOnly(True)
        self.lbl_gt_path.setPlaceholderText("可选，用于计算 PSNR/SSIM")
        self.lbl_gt_path.setMinimumWidth(180)
        self.btn_select_gt = QPushButton("选择")
        self.btn_select_gt.setObjectName("btnSelectGT")
        self.btn_select_gt.setCursor(Qt.PointingHandCursor)
        self.btn_select_gt.clicked.connect(self._on_select_gt)

        gt_area.addWidget(lbl_gt)
        gt_area.addWidget(self.lbl_gt_path, 1)
        gt_area.addWidget(self.btn_select_gt)

        toolbar_layout.addLayout(gt_area, 2)

        toolbar_layout.addWidget(sep1)

        # 右侧：算法+生成
        action_area = QHBoxLayout()
        action_area.setSpacing(10)

        self.combo_algorithm = QComboBox()
        self.combo_algorithm.addItem("🔬 算法一：DFSMamba", 1)
        self.combo_algorithm.addItem("⚛️ 算法二：PhysMamba", 2)
        self.combo_algorithm.setMinimumWidth(220)
        self.combo_algorithm.setCursor(Qt.PointingHandCursor)

        self.btn_generate = QPushButton("▶  生成红外图像")
        self.btn_generate.setObjectName("btnGenerate")
        self.btn_generate.setCursor(Qt.PointingHandCursor)
        self.btn_generate.clicked.connect(self._on_generate)

        self.btn_open_dir = QPushButton("📂")
        self.btn_open_dir.setObjectName("btnOpenDir")
        self.btn_open_dir.setToolTip("打开保存目录 D:/Generated_IR/")
        self.btn_open_dir.setCursor(Qt.PointingHandCursor)
        self.btn_open_dir.setFixedWidth(40)
        self.btn_open_dir.clicked.connect(self._on_open_save_dir)

        action_area.addWidget(self.combo_algorithm)
        action_area.addWidget(self.btn_generate)
        action_area.addWidget(self.btn_open_dir)

        toolbar_layout.addLayout(action_area, 3)

        main_layout.addWidget(toolbar)

        # ========== 图像显示区域 ==========
        image_layout = QHBoxLayout()
        image_layout.setSpacing(14)

        # 左侧卡片
        left_card = QFrame()
        left_card.setObjectName("imageCardLeft")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        left_header = QLabel("  原始可见光图像")
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
        self.lbl_input_image.setText("暂无图像\n\n点击上方「选择」按钮\n加载可见光图像")

        left_layout.addWidget(left_header)
        left_layout.addWidget(self.lbl_input_image, 1)

        image_layout.addWidget(left_card, 1)

        # 中间箭头
        arrow_label = QLabel()
        arrow_label.setAlignment(Qt.AlignCenter)
        arrow_label.setFixedWidth(40)
        arrow_label.setText("→")
        arrow_label.setStyleSheet(
            "font-size: 32px; color: #1a73e8; font-weight: bold;"
        )
        image_layout.addWidget(arrow_label)

        # 右侧卡片
        right_card = QFrame()
        right_card.setObjectName("imageCardRight")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        right_header = QLabel("  生成的红外图像")
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
        self.lbl_output_image.setText("等待生成\n\n选择算法后点击\n「生成红外图像」按钮")

        right_layout.addWidget(right_header)
        right_layout.addWidget(self.lbl_output_image, 1)

        image_layout.addWidget(right_card, 1)

        main_layout.addLayout(image_layout, 1)

        # ========== 底部：指标 + 日志 ==========
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(14)

        # 指标卡片面板
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

        metrics_title = QLabel("📊 图像质量评估")
        metrics_title.setStyleSheet(
            "color: #1e293b; font-size: 13px; font-weight: 700; padding-bottom: 8px;"
        )
        metrics_layout.addWidget(metrics_title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(6)

        self.metric_labels = {}
        items = [
            ('psnr', 'PSNR (dB)', '#3b82f6'),
            ('ssim', 'SSIM', '#8b5cf6'),
            ('l1', 'L1', '#f59e0b'),
            ('lipips', 'LIPIPS', '#ef4444'),
            ('infer_time', '推理耗时', '#10b981'),
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
                val_lbl.setMaximumWidth(280)
            else:
                val_lbl.setStyleSheet(
                    f"color: {color}; font-size: 14px; font-weight: 700; "
                    "font-family: Consolas, JetBrains Mono, monospace;"
                )

            self.metric_labels[key] = val_lbl

            grid.addWidget(name_lbl, row, col_base)
            grid.addWidget(val_lbl, row, col_base + 1)

        # 需要让各列宽度均衡
        for col in range(6):
            grid.setColumnStretch(col, 1)

        metrics_layout.addLayout(grid)
        metrics_layout.addStretch()

        bottom_layout.addWidget(metrics_panel, 3)

        # 日志面板
        log_panel = QFrame()
        log_panel.setStyleSheet("""
            QFrame {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 10px;
            }
        """)
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(12, 10, 12, 10)

        log_title = QLabel("📋 运行日志")
        log_title.setStyleSheet(
            "color: #94a3b8; font-size: 13px; font-weight: 700; "
            "padding-bottom: 4px; background: transparent;"
        )
        log_layout.addWidget(log_title)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #0f172a;
                color: #cbd5e1;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px;
                font-size: 11px;
            }
            QTextEdit:focus {
                border-color: #475569;
            }
            QScrollBar:vertical {
                background: #0f172a;
                width: 6px;
            }
            QScrollBar::handle:vertical {
                background: #475569;
                border-radius: 3px;
            }
        """)
        log_layout.addWidget(self.log_text)

        bottom_layout.addWidget(log_panel, 2)

        main_layout.addLayout(bottom_layout, 0)

    # ========== 槽函数 ==========

    def _on_select_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择可见光图像", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*.*)"
        )
        if not path:
            return
        try:
            self.input_image = load_image(path)
            self.lbl_input_path.setText(path)
            self.lbl_input_path.setToolTip(path)
            show_image_on_label(self.lbl_input_image, self.input_image)
            self.lbl_input_image.setStyleSheet("border: none; background: transparent;")
            self._log(f"[✓] 已加载输入图像: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "加载失败", str(e))
            self._log(f"[✗] 加载图像失败: {e}")

    def _on_select_gt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择真实红外标签图像(可选)", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*.*)"
        )
        if not path:
            return
        try:
            self.gt_image = load_image(path)
            self.lbl_gt_path.setText(path)
            self.lbl_gt_path.setToolTip(path)
            self._log(f"[✓] 已加载标签图像: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "加载失败", str(e))
            self._log(f"[✗] 加载标签图像失败: {e}")

    def _on_generate(self):
        if self.input_image is None:
            QMessageBox.warning(self, "提示", "请先选择一张可见光图像！")
            return

        algo_id = self.combo_algorithm.currentData()
        algo_name = self.combo_algorithm.currentText().split("：")[1] if "：" in self.combo_algorithm.currentText() else self.combo_algorithm.currentText()
        self._log(f"[▶] 开始生成 | 算法: {algo_name}")

        self.btn_generate.setEnabled(False)
        self.btn_generate.setText("⏳ 生成中...")

        self.gen_worker = GenerationWorker(algo_id, self.input_image)
        self.gen_worker.finished.connect(self._on_generation_finished)
        self.gen_worker.error_occurred.connect(self._on_generation_error)
        self.gen_worker.start()

    def _on_generation_finished(self, output_image, elapsed):
        self.generated_image = output_image
        self.btn_generate.setEnabled(True)
        self.btn_generate.setText("▶  生成红外图像")

        show_image_on_label(self.lbl_output_image, output_image)
        self.lbl_output_image.setStyleSheet("border: none; background: transparent;")

        # 保存
        save_dir = "D:/Generated_IR"
        try:
            save_path = save_image(output_image, save_dir, prefix="generated_ir")
            self._log(f"[✓] 生成完成 | 耗时: {elapsed:.3f}s")
            self._log(f"[✓] 已保存至: {save_path}")
        except Exception as e:
            save_path = f"保存失败: {e}"
            self._log(f"[✗] {save_path}")
            QMessageBox.warning(self, "保存失败", str(e))

        # 指标
        self.metric_labels['infer_time'].setText(f"{elapsed:.3f} s")
        self.metric_labels['save_path'].setText(save_path)

        metrics = calculate_metrics(output_image, self.gt_image)
        self._display_metrics(metrics)

    def _on_generation_error(self, error_msg):
        self.btn_generate.setEnabled(True)
        self.btn_generate.setText("▶  生成红外图像")
        self._log(f"[✗] 生成失败: {error_msg}")
        QMessageBox.critical(self, "生成失败", f"推理过程中发生错误:\n\n{error_msg}")

    def _display_metrics(self, metrics):
        for key in ['psnr', 'ssim', 'l1', 'lipips']:
            val = metrics.get(key)
            if val is not None:
                self.metric_labels[key].setText(f"{val:.4f}" if key in ('ssim', 'l1', 'lipips') else f"{val:.2f}")
            elif key == 'lipips' and any(metrics.get(metric_key) is not None for metric_key in ('psnr', 'ssim', 'l1')):
                self.metric_labels[key].setText("无法计算")
            else:
                self.metric_labels[key].setText("需标签")
                self.metric_labels[key].setStyleSheet(
                    "color: #94a3b8; font-size: 11px; font-weight: 500;"
                )

    def _on_open_save_dir(self):
        save_dir = "D:/Generated_IR"
        os.makedirs(save_dir, exist_ok=True)
        if sys.platform == 'win32':
            os.startfile(save_dir)
        else:
            subprocess.Popen(['xdg-open', save_dir])

    def _log(self, message):
        self.log_text.append(message)
        if self.log_signal:
            self.log_signal.emit(message)
