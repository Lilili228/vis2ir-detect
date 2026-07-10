# 红外图像生成与目标检测演示系统

基于 PyQt5 的桌面应用，集成可见光→红外图像生成和 YOLOv9 目标检测两大功能。

## 项目结构

```
infrared_detection_gui/
├── main.py                     # 程序入口
├── README.md
├── ui/
│   ├── __init__.py
│   ├── main_window.py          # 主窗口 (QTabWidget)
│   ├── generation_tab.py       # Tab1: 可见光→红外生成界面
│   └── detection_tab.py        # Tab2: YOLOv9目标检测界面
└── core/
    ├── __init__.py
    ├── ir_generation.py        # 红外生成算法接口 (MIIGAN / PhysMamba)
    ├── yolov9_detection.py     # YOLOv9检测接口
    ├── metrics.py              # 图像质量评估指标 (PSNR/SSIM/L1/LIPIPS)
    └── utils.py                # 通用工具函数
```

## 功能说明

### Tab 1: 可见光 → 红外图像生成

- 选择可见光图像，左侧显示原始图像
- 两套算法可切换：MIIGAN (Mamba GAN) 和 PhysMamba
- 点击"生成红外图像"，右侧显示生成结果
- 自动保存到 `D:/Generated_IR/`
- 可选：加载真实红外标签图像，计算 PSNR/SSIM/L1/LIPIPS

### Tab 2: YOLOv9 目标检测

- 选择待检测图像，左侧显示原图
- 选择 YOLOv9 权重文件 (.pt)
- 点击"开始目标检测"，右侧显示带检测框的结果
- 自动保存到 `D:/YOLOv9_Detect_Result/`
- 显示检测数量、类别、置信度等指标

## 安装与运行

### 环境要求

- Windows 10/11 (D盘用于保存结果)
- Python 3.8+

### 安装依赖

```bash
pip install PyQt5 opencv-python numpy scikit-image
```

### 运行程序

```bash
cd infrared_detection_gui
python main.py
```

## 后续接入真实模型

### 接入 MIIGAN / PhysMamba

编辑 `core/ir_generation.py`：
1. 取消 TODO 区域的注释
2. MIIGAN 路径指向 `miigan-master`
3. PhysMamba 默认加载 `F:/software/weight/PhysMamba/latest_net_G.pth`

### 接入 YOLOv9

编辑 `core/yolov9_detection.py`：
1. 取消 TODO 区域的注释
2. 安装 YOLOv9 依赖并下载权重文件
3. 设置正确的 YOLOv9 路径

## 当前状态

- 红外生成：使用伪彩色占位算法（INFERNO/HOT 颜色映射模拟红外效果）
- YOLOv9检测：使用模拟检测框进行界面演示
- 接入真实模型后即可用于实际推理
