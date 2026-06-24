#!/bin/bash
# ============================================================
# pipeline_init.sh — 远程服务器环境配置（仅 AscendDevTool 自身依赖）
# SAM-6D 使用服务器已有的 torch_npu 环境运行
# 在 ~/pipeline_tool/ 下执行: bash AscendDevTool/scripts/pipeline_init.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if echo "$SCRIPT_DIR" | grep -q "/AscendDevTool/"; then
    PIPELINE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"
fi

LOG_DIR="$PIPELINE_ROOT/AscendDevTool/logs/init"
mkdir -p "$LOG_DIR"

INIT_LOG="$LOG_DIR/init_$(date +%Y%m%d_%H%M%S).log"
STATUS_FILE="$LOG_DIR/init_status.txt"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

PASSED=0
FAILED=0
> "$INIT_LOG"

log() { echo "[INIT] $(date '+%H:%M:%S') $*" | tee -a "$INIT_LOG"; }
step_pass() { PASSED=$((PASSED + 1)); log "  ✅ $1"; }
step_fail() { FAILED=$((FAILED + 1)); log "  ❌ $1 失败（继续执行）"; }

# ============================================================
log "========================================"
log "AscendDevTool 环境初始化（不安装 SAM-6D 依赖）"
log "PIPELINE_ROOT=$PIPELINE_ROOT"
log "TOOL_DIR=$PIPELINE_ROOT/AscendDevTool"
log "========================================"

# ---- 1. Git 拉取最新代码 ----
log "1/3 Git 拉取 AscendDevTool..."
TOOL_DIR="$PIPELINE_ROOT/AscendDevTool"
if [ -d "$TOOL_DIR/.git" ]; then
    cd "$TOOL_DIR"
    git pull >> "$INIT_LOG" 2>&1 && step_pass "git pull" || step_fail "git pull"
else
    git clone https://github.com/dsg-lzt/ascendevtool.git "$TOOL_DIR" >> "$INIT_LOG" 2>&1 \
        && step_pass "git clone" || step_fail "git clone"
fi

# ---- 2. Python 虚拟环境（只装工具自身依赖）----
log "2/3 创建虚拟环境 + 安装工具依赖..."
cd "$TOOL_DIR"

if [ ! -d "ascenddevtool" ]; then
    python3 -m venv ascenddevtool >> "$INIT_LOG" 2>&1
fi
source ascenddevtool/bin/activate 2>> "$INIT_LOG" || { step_fail "venv"; }

# PySide6 + libcst（GUI）
pip install PySide6>=6.6 libcst>=1.1.0 -i "$PIP_MIRROR" >> "$INIT_LOG" 2>&1 \
    && step_pass "PySide6 + libcst" || step_fail "PySide6 + libcst"

# pandas + prettytable（CANN 扫描工具依赖）
pip install pandas prettytable -i "$PIP_MIRROR" >> "$INIT_LOG" 2>&1 \
    && step_pass "pandas + prettytable" || step_fail "pandas + prettytable"

# ---- 3. 目录 + Git 配置 ----
log "3/3 目录准备 + Git 配置..."
mkdir -p "$PIPELINE_ROOT/ascenddev_output"
mkdir -p "$PIPELINE_ROOT/logs"

cd "$TOOL_DIR"
git config user.email "pipeline@ascend-dev.local" 2>/dev/null || true
git config user.name "Pipeline Bot" 2>/dev/null || true
if [ -n "$GIT_TOKEN" ]; then
    git remote set-url origin "https://${GIT_TOKEN}@github.com/dsg-lzt/ascendevtool.git" 2>/dev/null
fi
step_pass "git config"

# ============================================================
# ── 上传日志 ──
cd "$TOOL_DIR"
git add logs/ 2>/dev/null || true
git commit -m "logs: pipeline init [auto]" 2>/dev/null || true
git push origin master >> "$INIT_LOG" 2>&1 \
    && log "  ✅ 日志已上传" \
    || log "  ❌ 日志上传失败（请设置 GIT_TOKEN 或手动 git push）"

log "========================================"
log "初始化完成: 成功 $PASSED / 失败 $FAILED"
echo "PASSED=$PASSED FAILED=$FAILED" > "$STATUS_FILE"

if [ $FAILED -gt 0 ]; then
    log "⚠️  有 $FAILED 个步骤失败，日志: $INIT_LOG"
else
    log "全部成功！启动循环："
    log "  nohup bash AscendDevTool/scripts/pipeline_loop.sh > pipeline_loop.log 2>&1 &"
fi
log "========================================"
