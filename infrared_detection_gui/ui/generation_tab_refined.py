"""可见光到红外图像生成界面：浅色科研风重构版。"""

import os
import subprocess
import sys

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.ir_generation import generate_ir_algorithm_1, generate_ir_algorithm_2
from core.metrics import calculate_metrics
from core.utils import load_image, save_image, show_image_on_label


class GenerationWorker(QThread):
    """红外图像生成工作线程。"""

    result_ready = pyqtSignal(object, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, algorithm_id, input_image, device_preference):
        super().__init__()
        self.algorithm_id = algorithm_id
        self.input_image = input_image.copy()
        self.device_preference = device_preference

    def run(self):
        try:
            if self.algorithm_id == 1:
                output, elapsed = generate_ir_algorithm_1(
                    self.input_image, device_preference=self.device_preference
                )
            else:
                output, elapsed = generate_ir_algorithm_2(
                    self.input_image, device_preference=self.device_preference
                )
            self.result_ready.emit(output, elapsed)
        except Exception as e:
            self.error_occurred.emit(str(e))


class GenerationTab(QWidget):
    """可见光到红外生成界面。"""

    ALGORITHM_HINTS = {
        1: "DFSMamba 为默认算法，支持 CPU/CUDA 设备选择，CUDA 不可用时后端可回退 CPU。",
        2: "使用 PhysMamba 权重进行可见光到红外生成，支持 CPU/CUDA 自动适配。",
    }

    def __init__(self, log_signal=None):
        super().__init__()
        self.log_signal = log_signal
        self.input_image = None
        self.generated_image = None
        self.gt_image = None
        self.gen_worker = None
        self.metric_labels = {}
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 14, 18, 16)
        main_layout.setSpacing(14)

        main_layout.addWidget(self._create_header())

        content_layout = QHBoxLayout()
        content_layout.setSpacing(14)
        content_layout.addWidget(self._create_config_panel(), 0)
        content_layout.addWidget(self._create_image_flow_panel(), 1)
        main_layout.addLayout(content_layout, 1)

        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(14)
        bottom_layout.addWidget(self._create_metric_panel(), 3)
        bottom_layout.addWidget(self._create_log_panel(), 2)
        main_layout.addLayout(bottom_layout, 0)

    def _create_header(self):
        header = QFrame()
        header.setObjectName("pageHeader")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(16)

        title_area = QVBoxLayout()
        title_area.setSpacing(4)
        title = QLabel("可见光到红外图像生成")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Visible-to-Infrared Image Translation")
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

        algo_label = QLabel("当前算法")
        algo_label.setObjectName("statusLabel")
        self.lbl_algorithm_status = QLabel("DFSMamba")
        self.lbl_algorithm_status.setObjectName("statusValue")
        state_label = QLabel("运行状态")
        state_label.setObjectName("statusLabel")
        self.lbl_generation_status = QLabel("待生成")
        self.lbl_generation_status.setObjectName("statusValue")
        device_label = QLabel("运行设备")
        device_label.setObjectName("statusLabel")
        self.lbl_device_status = QLabel("CPU")
        self.lbl_device_status.setObjectName("statusValue")

        status_layout.addWidget(algo_label, 0, 0)
        status_layout.addWidget(self.lbl_algorithm_status, 0, 1)
        status_layout.addWidget(state_label, 1, 0)
        status_layout.addWidget(self.lbl_generation_status, 1, 1)
        status_layout.addWidget(device_label, 2, 0)
        status_layout.addWidget(self.lbl_device_status, 2, 1)
        layout.addWidget(status_card, 0)

        return header

    def _create_config_panel(self):
        panel = QFrame()
        panel.setObjectName("configPanel")
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(330)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        title = QLabel("生成配置")
        title.setObjectName("sectionTitle")
        hint = QLabel("选择输入、可选标签与转换算法")
        hint.setObjectName("sectionHint")
        layout.addWidget(title)
        layout.addWidget(hint)

        self.lbl_input_path, self.btn_select_input = self._add_file_picker(
            layout, "可见光图像", "请选择可见光图像...", self._on_select_input
        )
        self.lbl_gt_path, self.btn_select_gt = self._add_file_picker(
            layout, "真实红外标签（可选）", "用于计算 PSNR / SSIM", self._on_select_gt
        )

        choice_row = QHBoxLayout()
        choice_row.setSpacing(8)

        algo_col = QVBoxLayout()
        algo_col.setSpacing(5)
        algo_label = QLabel("算法选择")
        algo_label.setObjectName("fieldLabel")
        algo_col.addWidget(algo_label)

        self.combo_algorithm = QComboBox()
        self.combo_algorithm.addItem("算法一：DFSMamba", 1)
        self.combo_algorithm.addItem("算法二：PhysMamba", 2)
        self.combo_algorithm.currentIndexChanged.connect(self._on_algorithm_changed)
        algo_col.addWidget(self.combo_algorithm)

        device_col = QVBoxLayout()
        device_col.setSpacing(5)
        device_label = QLabel("运行设备")
        device_label.setObjectName("fieldLabel")
        device_col.addWidget(device_label)

        self.combo_device = QComboBox()
        self.combo_device.addItem("CPU", "cpu")
        self.combo_device.addItem("CUDA", "cuda")
        self.combo_device.currentIndexChanged.connect(self._on_device_changed)
        device_col.addWidget(self.combo_device)

        choice_row.addLayout(algo_col, 2)
        choice_row.addLayout(device_col, 1)
        layout.addLayout(choice_row)

        algorithm_card = QFrame()
        algorithm_card.setObjectName("algorithmCard")
        algorithm_card.setMinimumHeight(54)
        algorithm_card.setMaximumHeight(64)
        algo_layout = QVBoxLayout(algorithm_card)
        algo_layout.setContentsMargins(10, 8, 10, 8)
        self.algorithm_hint = QLabel(self.ALGORITHM_HINTS[1])
        self.algorithm_hint.setObjectName("sectionHint")
        self.algorithm_hint.setWordWrap(True)
        self.algorithm_hint.setMaximumHeight(42)
        algo_layout.addWidget(self.algorithm_hint)
        layout.addWidget(algorithm_card)
        self.combo_algorithm.setCurrentIndex(1)
        self._on_algorithm_changed()
        self._on_device_changed()

        layout.addStretch(1)

        self.btn_generate = QPushButton("开始生成红外图像")
        self.btn_generate.setObjectName("btnGenerate")
        self.btn_generate.setCursor(Qt.PointingHandCursor)
        self.btn_generate.clicked.connect(self._on_generate)
        layout.addWidget(self.btn_generate)

        self.btn_open_dir = QPushButton("打开保存目录")
        self.btn_open_dir.setObjectName("btnOpenDir")
        self.btn_open_dir.setCursor(Qt.PointingHandCursor)
        self.btn_open_dir.setToolTip("打开保存目录 D:/Generated_IR/")
        self.btn_open_dir.clicked.connect(self._on_open_save_dir)
        layout.addWidget(self.btn_open_dir)

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

    def _create_image_flow_panel(self):
        panel = QFrame()
        panel.setObjectName("panelCard")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("模态转换流程")
        title.setObjectName("sectionTitle")
        hint = QLabel("输入可见光图像，输出红外风格图像，保持比例缩放显示。")
        hint.setObjectName("sectionHint")
        layout.addWidget(title)
        layout.addWidget(hint)

        image_layout = QHBoxLayout()
        image_layout.setSpacing(12)

        left_card, self.lbl_input_image = self._create_image_card(
            "输入可见光图像", "Input Visible Image", "请选择可见光图像"
        )
        right_card, self.lbl_output_image = self._create_image_card(
            "生成红外图像", "Generated Infrared Image", "等待生成红外图像"
        )

        arrow = QLabel("→")
        arrow.setObjectName("flowArrow")
        arrow.setAlignment(Qt.AlignCenter)
        arrow.setFixedWidth(36)

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
        image_label.setMinimumSize(300, 250)
        image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_label.setWordWrap(True)
        canvas_layout.addWidget(image_label)
        card_layout.addWidget(canvas_frame, 1)

        return card, image_label

    def _create_metric_panel(self):
        panel = QFrame()
        panel.setObjectName("metricPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        title = QLabel("图像质量评估")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        metric_items = [
            ("psnr", "PSNR", "未提供标签"),
            ("ssim", "SSIM", "未提供标签"),
            ("l1", "L1", "未提供标签"),
            ("lipips", "LIPIPS", "未提供标签"),
            ("infer_time", "推理时间", "未生成"),
            ("save_path", "保存路径", "未保存"),
        ]
        for idx, (key, title_text, default_value) in enumerate(metric_items):
            card = self._create_metric_card(key, title_text, default_value)
            grid.addWidget(card, idx // 3, idx % 3)

        layout.addLayout(grid)
        return panel

    def _create_metric_card(self, key, title, default_value):
        card = QFrame()
        card.setObjectName("metricCard")
        card.setProperty("accent", "generation")
        card.setMinimumHeight(64)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 7, 12, 7)
        card_layout.setSpacing(4)

        name_label = QLabel(title)
        name_label.setObjectName("metricName")
        value_label = QLabel(default_value)
        value_label.setObjectName("pathValue" if key == "save_path" else "metricValue")
        value_label.setWordWrap(True)
        card_layout.addWidget(name_label)
        card_layout.addWidget(value_label, 1)

        self.metric_labels[key] = value_label
        return card

    def _create_log_panel(self):
        panel = QFrame()
        panel.setObjectName("logCard")
        panel.setMinimumWidth(330)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        title = QLabel("运行信息")
        title.setObjectName("sectionTitle")
        hint = QLabel("仅显示关键流程信息，辅助排查输入、推理和保存状态。")
        hint.setObjectName("sectionHint")
        layout.addWidget(title)
        layout.addWidget(hint)

        self.log_text = QTextEdit()
        self.log_text.setObjectName("logText")
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setMinimumHeight(92)
        self.log_text.setMaximumHeight(116)
        layout.addWidget(self.log_text)
        return panel

    def _on_algorithm_changed(self):
        algo_id = self.combo_algorithm.currentData()
        algo_name = self._current_algorithm_name()
        hint = self.ALGORITHM_HINTS.get(algo_id, "")
        if algo_id == 1 and self._current_device_preference() == "cpu":
            hint = "DFSMamba 为默认算法，支持 CPU/CUDA 设备选择，CUDA 不可用时后端可回退 CPU。"
        self.lbl_algorithm_status.setText(algo_name)
        self.algorithm_hint.setText(hint)

    def _on_device_changed(self):
        self.lbl_device_status.setText(self._current_device_label())
        if hasattr(self, "algorithm_hint"):
            self._on_algorithm_changed()

    def _current_algorithm_name(self):
        text = self.combo_algorithm.currentText()
        return text.split("：", 1)[1] if "：" in text else text

    def _current_device_preference(self):
        if not hasattr(self, "combo_device"):
            return "cpu"
        return self.combo_device.currentData() or "cpu"

    def _current_device_label(self):
        if not hasattr(self, "combo_device"):
            return "CPU"
        return self.combo_device.currentText() or "CPU"

    def _on_select_input(self):
        if self._is_generation_running():
            QMessageBox.information(self, "正在生成", "当前推理尚未结束，请等待完成后再切换图片。")
            return

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
            self._reset_state_for_new_input()
            self._log(f"[OK] 已加载输入图像: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "加载失败", str(e))
            self._log(f"[ERROR] 加载图像失败: {e}")

    def _on_select_gt(self):
        if self._is_generation_running():
            QMessageBox.information(self, "正在生成", "当前推理尚未结束，请等待完成后再选择标签。")
            return

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
            self._log(f"[OK] 已加载标签图像: {os.path.basename(path)}")
            self._refresh_quality_metrics_state()
        except Exception as e:
            QMessageBox.warning(self, "加载失败", str(e))
            self._log(f"[ERROR] 加载标签图像失败: {e}")

    def _is_generation_running(self):
        return self.gen_worker is not None and self.gen_worker.isRunning()

    def _set_generation_controls_enabled(self, enabled):
        for widget in (
            self.btn_select_input,
            self.btn_select_gt,
            self.combo_algorithm,
            self.combo_device,
            self.btn_generate,
        ):
            widget.setEnabled(enabled)

    def _reset_state_for_new_input(self):
        self.generated_image = None
        self.gt_image = None
        self.lbl_gt_path.clear()
        self.lbl_gt_path.setToolTip("")
        self.lbl_output_image.clear()
        self.lbl_output_image.setText("等待生成红外图像")
        self.lbl_generation_status.setText("待生成")
        self.metric_labels["infer_time"].setText("未生成")
        self.metric_labels["save_path"].setText("未保存")
        self._refresh_quality_metrics_state()

    def _prepare_generation_state(self):
        self.generated_image = None
        self.lbl_output_image.clear()
        self.lbl_output_image.setText("生成中...")
        self.metric_labels["infer_time"].setText("生成中")
        self.metric_labels["save_path"].setText("等待保存")
        if self.gt_image is not None:
            self._set_quality_metrics_text("等待生成")

    def _on_generate(self):
        if self.input_image is None:
            QMessageBox.warning(self, "提示", "请先选择一张可见光图像！")
            return
        if self._is_generation_running():
            return

        algo_id = self.combo_algorithm.currentData()
        algo_name = self._current_algorithm_name()
        device_preference = self._current_device_preference()
        device_label = self._current_device_label()
        if algo_id == 1 and device_preference == "cpu":
            QMessageBox.warning(
                self,
                "设备不支持",
                "算法一 DFSMamba 当前暂未适配 CPU 推理，请切换 CUDA 或选择算法二 PhysMamba。",
            )
            return

        self.lbl_generation_status.setText("生成中")
        self._log(f"[RUN] 开始生成 | 算法: {algo_name} | 设备: {device_label}")
        self._prepare_generation_state()

        self._set_generation_controls_enabled(False)
        self.btn_generate.setText("生成中...")

        self.gen_worker = GenerationWorker(algo_id, self.input_image, device_preference)
        self.gen_worker.result_ready.connect(self._on_generation_finished)
        self.gen_worker.error_occurred.connect(self._on_generation_error)
        self.gen_worker.finished.connect(self._on_worker_finished)
        self.gen_worker.finished.connect(self.gen_worker.deleteLater)
        self.gen_worker.start()

    def _on_generation_finished(self, output_image, elapsed):
        self.generated_image = output_image
        self._set_generation_controls_enabled(True)
        self.btn_generate.setText("开始生成红外图像")
        self.lbl_generation_status.setText("已完成")

        show_image_on_label(self.lbl_output_image, output_image)

        save_dir = "D:/Generated_IR"
        try:
            save_path = save_image(output_image, save_dir, prefix="generated_ir")
            self._log(f"[OK] 生成完成 | 耗时: {elapsed:.3f}s")
            self._log(f"[OK] 已保存至: {save_path}")
        except Exception as e:
            save_path = f"保存失败: {e}"
            self._log(f"[ERROR] {save_path}")
            QMessageBox.warning(self, "保存失败", str(e))

        self.metric_labels["infer_time"].setText(f"{elapsed:.3f} s")
        self.metric_labels["save_path"].setText(save_path)
        self._update_quality_metrics_after_generation()

    def _on_generation_error(self, error_msg):
        self._set_generation_controls_enabled(True)
        self.btn_generate.setText("开始生成红外图像")
        self.lbl_generation_status.setText("生成失败")
        self.lbl_output_image.clear()
        self.lbl_output_image.setText("生成失败，请查看运行信息")
        self.metric_labels["infer_time"].setText("生成失败")
        self.metric_labels["save_path"].setText("未保存")
        self._log(f"[ERROR] 生成失败: {error_msg}")
        QMessageBox.critical(self, "生成失败", f"推理过程中发生错误:\n\n{error_msg}")

    def _on_worker_finished(self):
        worker = self.sender()
        if self.gen_worker is worker:
            self.gen_worker = None

    def _display_metrics(self, metrics):
        for key in ["psnr", "ssim", "l1", "lipips"]:
            val = metrics.get(key)
            if val is None:
                self.metric_labels[key].setText(self._quality_metric_empty_text())
            elif key == "psnr":
                self.metric_labels[key].setText(f"{val:.2f} dB")
            elif key in ("ssim", "l1", "lipips"):
                self.metric_labels[key].setText(f"{val:.4f}")
            else:
                self.metric_labels[key].setText(f"{val:.2f}")

    def _update_quality_metrics_after_generation(self):
        """真实标签可选；没有标签时跳过质量指标计算，不影响生成流程。"""
        if self.gt_image is None:
            self._set_quality_metrics_text("未提供标签")
            self._log("[INFO] 未选择真实红外标签，已跳过 PSNR/SSIM/L1/LIPIPS 计算")
            return

        try:
            metrics = calculate_metrics(self.generated_image, self.gt_image)
            self._display_metrics(metrics)
        except Exception as e:
            self._set_quality_metrics_text("无法计算")
            self._log(f"[WARN] 质量指标计算失败，已跳过: {e}")

    def _refresh_quality_metrics_state(self):
        """根据标签和生成结果状态刷新 PSNR/SSIM/L1/LIPIPS 的占位文案。"""
        if self.gt_image is not None and self.generated_image is not None:
            self._update_quality_metrics_after_generation()
            return

        self._set_quality_metrics_text(self._quality_metric_empty_text())

    def _set_quality_metrics_text(self, text):
        for key in ["psnr", "ssim", "l1", "lipips"]:
            self.metric_labels[key].setText(text)

    def _quality_metric_empty_text(self):
        if self.gt_image is None:
            return "未提供标签"
        if self.generated_image is None:
            return "等待生成"
        return "无法计算"

    def _on_open_save_dir(self):
        save_dir = "D:/Generated_IR"
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
        if self.generated_image is not None:
            show_image_on_label(self.lbl_output_image, self.generated_image)


# Keep the existing import path stable for MainWindow while routing the refined
# generation page to the full workflow implementation.
from .generation_tab_workflow import GenerationTab as GenerationTab
