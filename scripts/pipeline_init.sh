#!/bin/bash
# ============================================================
# pipeline_init.sh — 远程服务器首次环境配置
# 在 ~/pipeline_tool/ 目录下执行一次
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$SCRIPT_DIR"

echo "========================================"
echo "[INIT] 初始化 pipeline 环境"
echo "[INIT] PIPELINE_ROOT=$PIPELINE_ROOT"
echo "========================================"

# ---- 1. Conda 环境 ----
echo "[INIT] 1/6 创建 Conda 环境 ascenddevtool..."
if conda env list | grep -q ascenddevtool; then
    echo "[INIT] Conda 环境 ascenddevtool 已存在，跳过"
else
    conda create -n ascenddevtool python=3.13 -y
fi

# ---- 2. 克隆 AscendDevTool ----
echo "[INIT] 2/6 克隆 AscendDevTool..."
if [ -d "$PIPELINE_ROOT/AscendDevTool/.git" ]; then
    echo "[INIT] AscendDevTool 已存在，执行 git pull"
    cd "$PIPELINE_ROOT/AscendDevTool"
    git pull
else
    git clone https://github.com/dsg-lzt/ascendevtool.git "$PIPELINE_ROOT/AscendDevTool"
fi

# ---- 3. 创建虚拟环境 ----
echo "[INIT] 3/6 创建项目虚拟环境..."
cd "$PIPELINE_ROOT/AscendDevTool"
if [ ! -d "ascenddevtool" ]; then
    python -m venv ascenddevtool
fi
source ascenddevtool/bin/activate

PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "[INIT] 安装项目依赖..."
pip install -r gui/requirements.txt -i "$PIP_MIRROR"
pip install pandas prettytable -i "$PIP_MIRROR"

# ---- 4. SAM-6D 依赖（来自 environment.txt） ----
echo "[INIT] 4/6 安装 SAM-6D 依赖..."

# 4.1 基础 Python 库
pip install attrs cython 'numpy>=1.19.2,<=1.24.0' decorator sympy cffi pyyaml \
    pathlib2 psutil protobuf==3.20.0 scipy requests absl-py -i "$PIP_MIRROR"

# 4.2 timm
pip install timm -i "$PIP_MIRROR"

# 4.3 torch + torch_npu（优先 wheel 文件，否则 pip 安装）
if [ -f "$PIPELINE_ROOT/torch-2.6.0+cpu-cp310-cp310-linux_x86_64.whl" ]; then
    pip install "$PIPELINE_ROOT/torch-2.6.0+cpu-cp310-cp310-linux_x86_64.whl"
elif [ -f "$PIPELINE_ROOT/AscendDevTool/torch-2.6.0+cpu-cp310-cp310-linux_x86_64.whl" ]; then
    pip install "$PIPELINE_ROOT/AscendDevTool/torch-2.6.0+cpu-cp310-cp310-linux_x86_64.whl"
else
    echo "[WARN] 未找到 torch wheel 文件，尝试 pip 安装"
    pip install torch==2.6.0 -i "$PIP_MIRROR"
fi

if [ -f "$PIPELINE_ROOT/torch_npu-2.6.0.post3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" ]; then
    pip install "$PIPELINE_ROOT/torch_npu-2.6.0.post3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
elif [ -f "$PIPELINE_ROOT/AscendDevTool/torch_npu-2.6.0.post3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" ]; then
    pip install "$PIPELINE_ROOT/AscendDevTool/torch_npu-2.6.0.post3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
else
    echo "[WARN] 未找到 torch_npu wheel 文件，尝试 pip 安装"
    pip install torch_npu -i "$PIP_MIRROR"
fi

# 4.4 torchvision
pip install torchvision==0.21.0 -i "$PIP_MIRROR"

# 4.5 triton-ascend
pip uninstall triton -y 2>/dev/null || true
pip install triton-ascend==3.2.0rc4 -i "$PIP_MIRROR"

# 4.6 其他库
pip install opencv-python addict imageio pycocotools trimesh scipy einops -i "$PIP_MIRROR"

# ---- 5. 目录准备 ----
echo "[INIT] 5/6 准备输出目录..."
mkdir -p "$PIPELINE_ROOT/ascenddev_output"
mkdir -p "$PIPELINE_ROOT/logs"

# ---- 6. Git 配置（避免首次 commit 报错） ----
echo "[INIT] 6/6 配置 Git..."
cd "$PIPELINE_ROOT/AscendDevTool"
git config user.email "pipeline@ascend-dev.local" || true
git config user.name "Pipeline Bot" || true

echo ""
echo "========================================"
echo "[INIT] 环境配置完成！"
echo ""
echo "  接下来执行："
echo "    cd $PIPELINE_ROOT"
echo "    nohup bash pipeline_loop.sh > pipeline_loop.log 2>&1 &"
echo "========================================"
