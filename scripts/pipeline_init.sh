#!/bin/bash
# ============================================================
# pipeline_init.sh — 远程服务器环境配置（可反复执行）
# 每次执行都会检查已有步骤，跳过已完成的，重试失败的
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"
WHEEL_DIR="$(dirname "$PIPELINE_ROOT")"
LOG_DIR="$PIPELINE_ROOT/logs/init"
mkdir -p "$LOG_DIR"

INIT_LOG="$LOG_DIR/init_$(date +%Y%m%d_%H%M%S).log"
STATUS_FILE="$LOG_DIR/init_status.txt"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

PASSED=0
FAILED=0
> "$INIT_LOG"

log() {
    echo "[INIT] $(date '+%H:%M:%S') $*" | tee -a "$INIT_LOG"
}

step_pass() {
    PASSED=$((PASSED + 1))
    log "  ✅ $1"
}

step_fail() {
    FAILED=$((FAILED + 1))
    log "  ❌ $1 失败（继续执行）"
}

# ============================================================
log "========================================"
log "初始化 pipeline 环境"
log "PIPELINE_ROOT=$PIPELINE_ROOT"
log "日志文件: $INIT_LOG"
log "========================================"

# ---- 1. Conda 环境 ----
log "1/7 创建 Conda 环境 ascenddevtool (python=3.10)..."
if conda env list 2>/dev/null | grep -q ascenddevtool; then
    log "  Conda 环境已存在，跳过"
    step_pass "conda env"
else
    if conda create -n ascenddevtool python=3.10 -y >> "$INIT_LOG" 2>&1; then
        step_pass "conda env"
    else
        step_fail "conda env"
    fi
fi

# ---- 2. Git 拉取最新代码 ----
log "2/7 Git 拉取 AscendDevTool..."
if [ -d "$PIPELINE_ROOT/AscendDevTool/.git" ]; then
    cd "$PIPELINE_ROOT/AscendDevTool"
    if git pull >> "$INIT_LOG" 2>&1; then
        step_pass "git pull"
    else
        step_fail "git pull"
    fi
else
    if git clone https://github.com/dsg-lzt/ascendevtool.git "$PIPELINE_ROOT/AscendDevTool" >> "$INIT_LOG" 2>&1; then
        step_pass "git clone"
    else
        step_fail "git clone"
    fi
fi

# ---- 3. 虚拟环境 ----
log "3/7 创建项目虚拟环境..."
cd "$PIPELINE_ROOT/AscendDevTool"
if [ ! -d "ascenddevtool" ]; then
    python -m venv ascenddevtool >> "$INIT_LOG" 2>&1
fi
if source ascenddevtool/bin/activate 2>> "$INIT_LOG"; then
    step_pass "venv"
else
    step_fail "venv"
fi

# ---- 4. 项目依赖 ----
log "4/7 安装项目依赖..."
if pip install -r gui/requirements.txt -i "$PIP_MIRROR" >> "$INIT_LOG" 2>&1; then
    step_pass "gui deps (PySide6, libcst)"
else
    step_fail "gui deps"
fi

if pip install pandas prettytable -i "$PIP_MIRROR" >> "$INIT_LOG" 2>&1; then
    step_pass "CANN deps (pandas, prettytable)"
else
    step_fail "CANN deps"
fi

# ---- 5. SAM-6D 基础依赖 ----
log "5/7 安装 SAM-6D 基础依赖..."
SAM6D_DEPS=(
    "attrs cython decorator sympy cffi pyyaml"
    "pathlib2 psutil protobuf==3.20.0 scipy requests absl-py"
    "timm"
    "opencv-python addict imageio pycocotools trimesh scipy einops"
)

for deps in "${SAM6D_DEPS[@]}"; do
    name=$(echo "$deps" | awk '{print $1}')
    if pip install $deps -i "$PIP_MIRROR" >> "$INIT_LOG" 2>&1; then
        step_pass "sam6d: $name"
    else
        step_fail "sam6d: $name"
    fi
done

# numpy 特殊处理（版本范围）
if pip install 'numpy>=1.19.2,<=1.24.0' -i "$PIP_MIRROR" >> "$INIT_LOG" 2>&1; then
    step_pass "sam6d: numpy"
else
    step_fail "sam6d: numpy"
fi

# ---- 6. torch + torch_npu ----
log "6/7 安装 torch + torch_npu..."
TORCH_WHEEL=""
for d in "$WHEEL_DIR" "$PIPELINE_ROOT"; do
    if [ -f "$d/torch-2.6.0+cpu-cp310-cp310-linux_x86_64.whl" ]; then
        TORCH_WHEEL="$d/torch-2.6.0+cpu-cp310-cp310-linux_x86_64.whl"
        break
    fi
done
if [ -n "$TORCH_WHEEL" ]; then
    if pip install "$TORCH_WHEEL" >> "$INIT_LOG" 2>&1; then
        step_pass "torch (wheel)"
    else
        step_fail "torch (wheel)"
    fi
else
    if pip install torch==2.6.0 -i "$PIP_MIRROR" >> "$INIT_LOG" 2>&1; then
        step_pass "torch (pip)"
    else
        step_fail "torch (pip)"
    fi
fi

NPU_WHEEL=""
for d in "$WHEEL_DIR" "$PIPELINE_ROOT"; do
    if [ -f "$d/torch_npu-2.6.0.post3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" ]; then
        NPU_WHEEL="$d/torch_npu-2.6.0.post3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        break
    fi
done
if [ -n "$NPU_WHEEL" ]; then
    if pip install "$NPU_WHEEL" >> "$INIT_LOG" 2>&1; then
        step_pass "torch_npu (wheel)"
    else
        step_fail "torch_npu (wheel)"
    fi
else
    if pip install torch_npu -i "$PIP_MIRROR" >> "$INIT_LOG" 2>&1; then
        step_pass "torch_npu (pip)"
    else
        step_fail "torch_npu (pip)"
    fi
fi

if pip install torchvision==0.21.0 -i "$PIP_MIRROR" >> "$INIT_LOG" 2>&1; then
    step_pass "torchvision"
else
    step_fail "torchvision"
fi

# ---- 7. triton-ascend + 目录 ----
log "7/7 triton-ascend + 目录准备..."
pip uninstall triton -y >> "$INIT_LOG" 2>&1 || true
if pip install triton-ascend==3.2.0rc4 -i "$PIP_MIRROR" >> "$INIT_LOG" 2>&1; then
    step_pass "triton-ascend"
else
    step_fail "triton-ascend"
fi

mkdir -p "$PIPELINE_ROOT/ascenddev_output"
mkdir -p "$PIPELINE_ROOT/logs"

# ── Git 配置 ──
cd "$PIPELINE_ROOT/AscendDevTool"
git config user.email "pipeline@ascend-dev.local" 2>/dev/null || true
git config user.name "Pipeline Bot" 2>/dev/null || true

# ============================================================
log "========================================"
log "初始化完成: 成功 $PASSED / 失败 $FAILED"
echo "PASSED=$PASSED FAILED=$FAILED" > "$STATUS_FILE"
echo "LOG=$INIT_LOG" >> "$STATUS_FILE"

# ── 上传日志 ──
log "上传初始化日志到 git..."
cd "$PIPELINE_ROOT/AscendDevTool"
git add "$LOG_DIR/" 2>/dev/null || true
git commit -m "logs: pipeline init [auto]" 2>/dev/null || true
if git push origin master >> "$INIT_LOG" 2>&1; then
    log "  ✅ 日志已上传"
else
    log "  ❌ 日志上传失败（可在 AscendDevTool 目录手动 git push）"
fi

if [ $FAILED -gt 0 ]; then
    log "⚠️  有 $FAILED 个步骤失败，查看日志: $INIT_LOG"
    log "  修改脚本后重新执行 bash AscendDevTool/scripts/pipeline_init.sh 即可重试"
    log "========================================"
    exit 1
else
    log "全部成功！接下来启动循环："
    log "  cd $PIPELINE_ROOT"
    log "  nohup bash AscendDevTool/scripts/pipeline_loop.sh > pipeline_loop.log 2>&1 &"
    log "========================================"
fi
