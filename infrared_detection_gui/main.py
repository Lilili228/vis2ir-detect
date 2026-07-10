"""
红外图像生成与目标检测演示系统 — 入口文件

使用方法:
    python main.py
"""

import sys
import os

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from ui.main_window import MainWindow


def main():
    # 高 DPI 适配
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("红外图像生成与目标检测系统")

    # 创建输出目录
    os.makedirs("D:/Generated_IR", exist_ok=True)
    os.makedirs("D:/YOLOv9_Detect_Result", exist_ok=True)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
