"""主窗口：承载两个科研演示界面。"""

from PyQt5.QtWidgets import QMainWindow, QTabWidget, QStatusBar, QMessageBox, QLabel
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont

from .generation_tab_refined import GenerationTab
from .detection_tab_refined import DetectionTab
from .styles import APP_STYLE


class MainWindow(QMainWindow):
    """红外图像生成与目标检测演示系统"""

    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("红外图像生成与目标检测演示系统  |  Infrared Detection Demo")
        self.resize(1360, 900)
        self.setMinimumSize(1180, 760)

        # 全局字体
        font = QFont("Microsoft YaHei", 9)
        self.setFont(font)

        # 应用统一科研风浅色主题
        self.setStyleSheet(APP_STYLE)

        # 中央 Tab
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self.generation_tab = GenerationTab(log_signal=self.log_signal)
        self.detection_tab = DetectionTab(log_signal=self.log_signal)

        self.tab_widget.addTab(self.generation_tab, "  可见光 → 红外生成  ")
        self.tab_widget.addTab(self.detection_tab, "  YOLOv9 红外目标检测  ")

        # 状态栏
        self.status_label = QLabel("● 就绪")
        self.status_label.setObjectName("statusText")
        self.status_bar = QStatusBar()
        self.status_bar.addPermanentWidget(self.status_label)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        self.log_signal.connect(self._on_log_message)

    def _on_log_message(self, message):
        self.status_label.setText(f"● {message[:80]}")
        self.status_bar.showMessage(message, 5000)

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "确认退出",
            "确定要退出红外图像生成与目标检测演示系统吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
