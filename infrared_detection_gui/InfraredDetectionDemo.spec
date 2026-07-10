# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

PROJECT_DIR = Path(SPECPATH).resolve()
WORKSPACE_ROOT = PROJECT_DIR.parent

datas = []
binaries = []
hiddenimports = []


def collect_package(package_name):
    try:
        package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
        datas.extend(package_datas)
        binaries.extend(package_binaries)
        hiddenimports.extend(package_hiddenimports)
    except Exception as exc:
        print(f"[WARN] Skip package {package_name}: {exc}")


def add_file_tree(src, dest, suffixes=None, exclude_parts=None):
    src = Path(src)
    if not src.exists():
        print(f"[WARN] Missing data path: {src}")
        return
    suffixes = {suffix.lower() for suffix in suffixes} if suffixes else None
    exclude_parts = set(exclude_parts or [])
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        if any(part in exclude_parts for part in path.parts):
            continue
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        target_dir = Path(dest) / path.relative_to(src).parent
        datas.append((str(path), str(target_dir)))


def add_one_file(src, dest):
    src = Path(src)
    if src.is_file():
        datas.append((str(src), dest))
    else:
        print(f"[WARN] Missing data file: {src}")
tmp_ret = collect_all('PyQt5')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torchvision')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('PIL')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('yaml')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

for package in [
    # Network/download stack used by YOLOv9 helpers and algorithm utilities.
    "requests",
    "charset_normalizer",
    "urllib3",
    "certifi",
    "idna",
    # GUI reports and image-generation quality metrics.
    "lpips",
    "docx",
    "lxml",
    "imageio",
    # Scientific/image stack used by PhysMamba, MIIGAN/DFSMamba and YOLOv9.
    "matplotlib",
    "pandas",
    "scipy",
    "seaborn",
    "skimage",
    "pytorch_msssim",
    # DFSMamba/MIIGAN model dependencies.
    "timm",
    "einops",
    "ml_collections",
    "mamba_ssm",
    "causal_conv1d",
    # YOLOv9 runtime and optional plotting/evaluation helpers.
    "ultralytics",
    "psutil",
    "tqdm",
    "thop",
    "albumentations",
    "pycocotools",
    # Legacy algorithm utility modules. They are not on the main inference path,
    # but are cheap enough to collect when installed.
    "dominate",
    "bs4",
]:
    collect_package(package)

hiddenimports += [
    "IPython",
    "IPython.display",
    "mamba_ssm.ops.selective_scan_interface",
    "selective_scan_cuda",
    "causal_conv1d_cuda",
]

# External generation algorithm code. Keep source/config/weight files only;
# do not include generated previews, IDE metadata, caches, datasets, or runs.
add_file_tree(
    WORKSPACE_ROOT / "PhysMamba",
    "PhysMamba",
    suffixes={".py", ".pth", ".pt", ".txt", ".yaml", ".yml"},
    exclude_parts={"__pycache__", ".idea", "fig", "ours-fake"},
)
add_file_tree(
    WORKSPACE_ROOT / "miigan-master",
    "miigan-master",
    suffixes={".py", ".pth", ".pt", ".txt", ".yaml", ".yml"},
    exclude_parts={"__pycache__", ".idea", "figs"},
)

# YOLOv9 runtime: code + default model/data only. Do not include datasets/runs.
add_file_tree(
    WORKSPACE_ROOT / "yolov9-main" / "models",
    "yolov9-main/models",
    suffixes={".py", ".yaml", ".yml", ".txt"},
    exclude_parts={"__pycache__"},
)
add_file_tree(
    WORKSPACE_ROOT / "yolov9-main" / "utils",
    "yolov9-main/utils",
    suffixes={".py", ".yaml", ".yml", ".txt", ".ttf"},
    exclude_parts={"__pycache__"},
)
add_one_file(WORKSPACE_ROOT / "yolov9-main" / "export.py", "yolov9-main")
add_one_file(WORKSPACE_ROOT / "yolov9-main" / "data" / "coco.yaml", "yolov9-main/data")
add_one_file(WORKSPACE_ROOT / "yolov9-main" / "weights" / "yolov9-c.pt", "yolov9-main/weights")

# Generation weights and torch hub checkpoints used by LPIPS.
add_file_tree(
    WORKSPACE_ROOT / "weight",
    "weight",
    suffixes={".pth", ".pt", ".txt", ".yaml", ".yml"},
    exclude_parts={"__pycache__"},
)


a = Analysis(
    ['main.py'],
    pathex=[str(PROJECT_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='InfraredDetectionDemo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='InfraredDetectionDemo',
)
