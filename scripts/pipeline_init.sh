#!/bin/bash
# ============================================================
# pipeline_init.sh — 远程服务器环境配置（可反复执行）
# 每次执行都会检查已有步骤，跳过已完成的，重试失败的
# 在 ~/pipeline_tool/ 下执行: bash AscendDevTool/scripts/pipeline_init.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 自动检测 PIPELINE_ROOT：如果脚本在 AscendDevTool/scripts/ 下，则 PIPELINE_ROOT 为其父目录的父目录
if echo "$SCRIPT_DIR" | grep -q "/AscendDevTool/"; then
    PIPELINE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"
fi

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
log "TOOL_DIR=$PIPELINE_ROOT/AscendDevTool"
log "日志文件: $INIT_LOG"
log "========================================"

# ---- 1. Conda 环境 ----
log "1/7 创建 Conda 环境 ascenddevtool (python=3.10)..."
CONDA_RECREATE=""
if conda env list 2>/dev/null | grep -q ascenddevtool; then
    PY_VER=$(conda run -n ascenddevtool python --version 2>&1 || echo "unknown")
    if echo "$PY_VER" | grep -q "Python 3.10"; then
        log "  Conda 环境已存在 ($PY_VER)，跳过"
        step_pass "conda env"
    else
        log "  Conda 环境 Python 版本不对 ($PY_VER)，需要 3.10，重建..."
        conda env remove -n ascenddevtool -y >> "$INIT_LOG" 2>&1 || true
        CONDA_RECREATE="yes"
    fi
else
    CONDA_RECREATE="yes"
fi
if [ "$CONDA_RECREATE" = "yes" ]; then
    if conda create -n ascenddevtool python=3.10 -y >> "$INIT_LOG" 2>&1; then
        step_pass "conda env"
    else
        step_fail "conda env"
    fi
fi

# ---- 2. Git 拉取最新代码 ----
log "2/7 Git 拉取 AscendDevTool..."
TOOL_DIR="$PIPELINE_ROOT/AscendDevTool"
if [ -d "$TOOL_DIR/.git" ]; then
    cd "$TOOL_DIR"
    if git pull >> "$INIT_LOG" 2>&1; then
        step_pass "git pull"
    else
        step_fail "git pull"
    fi
else
    if git clone https://github.com/dsg-lzt/ascendevtool.git "$TOOL_DIR" >> "$INIT_LOG" 2>&1; then
        step_pass "git clone"
    else
        step_fail "git clone"
    fi
fi

# ---- 3. 虚拟环境 ----
log "3/7 创建项目虚拟环境（使用 conda python 3.10）..."
cd "$TOOL_DIR"

# 确保用 conda 的 python 3.10 来创建 venv
CONDA_PYTHON=$(conda run -n ascenddevtool which python 2>/dev/null || echo "")
if [ -z "$CONDA_PYTHON" ]; then
    CONDA_PYTHON=$(which python3 2>/dev/null || which python 2>/dev/null || echo "python3")
fi

if [ -d "ascenddevtool" ]; then
    # 检查 venv 的 python 版本
    VENV_PY_VER=$("$TOOL_DIR/ascenddevtool/bin/python" --version 2>&1 || echo "unknown")
    if echo "$VENV_PY_VER" | grep -q "Python 3.10"; then
        log "  venv 已存在 ($VENV_PY_VER)，跳过"
    else
        log "  venv Python 版本不对 ($VENV_PY_VER)，重建..."
        rm -rf ascenddevtool
        "$CONDA_PYTHON" -m venv ascenddevtool >> "$INIT_LOG" 2>&1
    fi
else
    "$CONDA_PYTHON" -m venv ascenddevtool >> "$INIT_LOG" 2>&1
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

for deps in \
    "attrs cython decorator sympy cffi pyyaml" \
    "pathlib2 psutil protobuf==3.20.0 scipy requests absl-py" \
    "timm" \
    "opencv-python addict imageio pycocotools trimesh scipy einops"
do
    name=$(echo "$deps" | awk '{print $1}')
    if pip install $deps -i "$PIP_MIRROR" >> "$INIT_LOG" 2>&1; then
        step_pass "sam6d: $name"
    else
        step_fail "sam6d: $name"
    fi
done

# numpy: 不要限定上限版本，python 3.10 以上用 >=1.19.2 即可
if pip install 'numpy>=1.19.2' -i "$PIP_MIRROR" >> "$INIT_LOG" 2>&1; then
    step_pass "sam6d: numpy"
else
    step_fail "sam6d: numpy"
fi

# ---- 6. torch + torch_npu ----
log "6/7 安装 torch + torch_npu..."

# 检查 whl 文件：优先在 PIPELINE_ROOT（~/pipeline_tool/）下找
TORCH_WHEEL="$PIPELINE_ROOT/torch-2.6.0+cpu-cp310-cp310-linux_x86_64.whl"
if [ -f "$TORCH_WHEEL" ]; then
    if pip install "$TORCH_WHEEL" >> "$INIT_LOG" 2>&1; then
        step_pass "torch (wheel)"
    else
        log "  WARN: cp310 whl 不兼容当前 Python，尝试在线安装"
        pip install torch==2.6.0 -i "$PIP_MIRROR" >> "$INIT_LOG" 2>&1 && step_pass "torch (pip)" || step_fail "torch (pip)"
    fi
else
    pip install torch==2.6.0 -i "$PIP_MIRROR" >> "$INIT_LOG" 2>&1 && step_pass "torch (pip)" || step_fail "torch (pip)"
fi

NPU_WHEEL="$PIPELINE_ROOT/torch_npu-2.6.0.post3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
if [ -f "$NPU_WHEEL" ]; then
    if pip install "$NPU_WHEEL" >> "$INIT_LOG" 2>&1; then
        step_pass "torch_npu (wheel)"
    else
        log "  WARN: cp310 whl 不兼容当前 Python，尝试在线安装"
        pip install torch_npu -i "$PIP_MIRROR" >> "$INIT_LOG" 2>&1 && step_pass "torch_npu (pip)" || step_fail "torch_npu (pip)"
    fi
else
    pip install torch_npu -i "$PIP_MIRROR" >> "$INIT_LOG" 2>&1 && step_pass "torch_npu (pip)" || step_fail "torch_npu (pip)"
fi

if pip install torchvision==0.21.0 -i "$PIP_MIRROR" >> "$INIT_LOG" 2>&1; then
    step_pass "torchvision"
else
    step_fail "torchvision"
fi

# ---- 7. triton-ascend + 目录 ----
log "7/7 triton-ascend + 目录准备..."
pip uninstall triton -y >> "$INIT_LOG" 2>&1 || true
# triton-ascend 不在清华镜像，用默认源
if pip install triton-ascend==3.2.0rc4 >> "$INIT_LOG" 2>&1; then
    step_pass "triton-ascend"
else
    step_fail "triton-ascend"
fi

mkdir -p "$PIPELINE_ROOT/ascenddev_output"
mkdir -p "$PIPELINE_ROOT/logs"

# ── Git 配置 ──
# 需要在远程服务器设置 GIT_TOKEN 环境变量
cd "$TOOL_DIR"
git config user.email "pipeline@ascend-dev.local" 2>/dev/null || true
git config user.name "Pipeline Bot" 2>/dev/null || true
if [ -n "$GIT_TOKEN" ]; then
    REPO_URL="https://${GIT_TOKEN}@github.com/dsg-lzt/ascendevtool.git"
    git remote set-url origin "$REPO_URL" 2>/dev/null
fi

# ============================================================
# ── 上传日志 ──
log "上传初始化日志到 git..."
cd "$TOOL_DIR"
LOG_DIR_ABS="$PIPELINE_ROOT/logs/init"

# 复制日志到 AscendDevTool 的 logs 目录（才能在 git 中提交）
mkdir -p logs
cp -r "$LOG_DIR_ABS" logs/ 2>/dev/null || true

git add logs/ 2>/dev/null || true
git commit -m "logs: pipeline init [auto]" 2>/dev/null || true
if git push origin master >> "$INIT_LOG" 2>&1; then
    log "  ✅ 日志已上传"
else
    log "  ❌ 日志上传失败"
    log "  请设置 GIT_TOKEN: export GIT_TOKEN=ghp_xxxxxxxx"
    log "  或手动 cd $TOOL_DIR && git push"
fi

log "========================================"
log "初始化完成: 成功 $PASSED / 失败 $FAILED"
echo "PASSED=$PASSED FAILED=$FAILED" > "$STATUS_FILE"
echo "LOG=$INIT_LOG" >> "$STATUS_FILE"

if [ $FAILED -gt 0 ]; then
    log "⚠️  有 $FAILED 个步骤失败"
    log "  日志: $INIT_LOG"
    log "  重新执行即可重试: bash AscendDevTool/scripts/pipeline_init.sh"
else
    log "全部成功！启动循环："
    log "  cd $PIPELINE_ROOT"
    log "  nohup bash AscendDevTool/scripts/pipeline_loop.sh > pipeline_loop.log 2>&1 &"
fi
log "========================================"
