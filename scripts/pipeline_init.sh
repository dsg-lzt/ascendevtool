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

echo "[INIT] 安装项目依赖..."
pip install -r gui/requirements.txt
pip install pandas prettytable -i https://pypi.tuna.tsinghua.edu.cn/simple

# ---- 4. SAM-6D 依赖 ----
echo "[INIT] 4/6 安装 SAM-6D 依赖..."
pip install torch torchvision torchaudio
pip install timm gorilla-core==0.2.7.8 trimesh==3.22.1
pip install imgaug opencv-python gpustat==1.0.0 einops
pip install torch_npu 2>/dev/null || echo "[WARN] torch_npu 安装失败（可能不需要手动装）"

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
