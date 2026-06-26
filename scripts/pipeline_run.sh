#!/bin/bash
# ============================================================
# pipeline_run.sh — 单次流水线执行（泛化版，支持任意模型）
# 用法: bash pipeline_run.sh <round_number> <model_name> [inference_cmd]
# ============================================================

ROUND="${1:-01}"
ROUND=$(printf '%02d' "$ROUND")
MODEL_NAME="${2:-SAM-6D}"
_INF_CMD="${3:-}"                   # 内部变量，避免 env 污染
_INF_PYTHON="${ASCEND_INF_PYTHON:-/home/orange/miniconda3/envs/torch_npu/bin/python}"
# 外部可覆盖推理脚本路径
_INF_SCRIPT="${ASCEND_INF_SCRIPT:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if echo "$SCRIPT_DIR" | grep -q "/AscendDevTool/"; then
    PIPELINE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"
fi
TOOL_DIR="$PIPELINE_ROOT/AscendDevTool"
MODEL_SRC="$PIPELINE_ROOT/$MODEL_NAME"
SCAN_OUT="$PIPELINE_ROOT/ascenddev_output/${MODEL_NAME}_scan"
MODEL_OUT="$PIPELINE_ROOT/ascenddev_output/${MODEL_NAME}_NPU"
LOG_DIR="$TOOL_DIR/logs/run_$ROUND"

mkdir -p "$LOG_DIR" "$SCAN_OUT" "$MODEL_OUT"
# 清空本轮状态文件，避免累积
: > "$LOG_DIR/status.txt"

log() {
    echo "[ROUND $ROUND] $(date '+%H:%M:%S') $*"
}

fail() {
    log "ERROR: $*"
    echo "PIPELINE_ERROR: $*" >> "$LOG_DIR/status.txt"
    exit 1
}

# ---- 0. 激活环境 ----
log "激活虚拟环境..."
source "$TOOL_DIR/ascenddevtool/bin/activate"

cd "$TOOL_DIR"

# ---- 1. CANN 扫描 ----
log "1/4 CANN 扫描 $MODEL_NAME..."

# CANN 环境变量（仅扫描步骤需要）
for d in "$HOME/Ascend/ascend-toolkit/latest" "/usr/local/Ascend/ascend-toolkit/latest" "$ASCEND_TOOLKIT_HOME"; do
    if [ -f "$d/tools/ms_fmk_transplt/analysis/pytorch_analyse.py" ]; then
        ASCEND_TOOLKIT_HOME="$d"
        break
    fi
done
for d in "$HOME/Ascend/cann/latest" "/usr/local/Ascend/cann/latest"; do
    if [ -f "$d/tools/ms_fmk_transplt/analysis/pytorch_analyse.py" ]; then
        ASCEND_TOOLKIT_HOME="$d"
        break
    fi
done
export ASCEND_TOOLKIT_HOME
export ASCEND_HOME_PATH="$ASCEND_TOOLKIT_HOME"
export ASCEND_OPP_PATH="$ASCEND_TOOLKIT_HOME/opp"
export PATH="$ASCEND_TOOLKIT_HOME/bin:$ASCEND_TOOLKIT_HOME/compiler/ccec_compiler/bin:$PATH"
export PYTHONPATH="$ASCEND_TOOLKIT_HOME/tools/ms_fmk_transplt:$ASCEND_TOOLKIT_HOME/python/site-packages:$PYTHONPATH"
SCAN_TS=$(date +%Y%m%d_%H%M%S)
SCAN_OUT_DIR="$SCAN_OUT/scan_$ROUND_$SCAN_TS"
mkdir -p "$SCAN_OUT_DIR"
SCAN_TOOL="$ASCEND_TOOLKIT_HOME/tools/ms_fmk_transplt/analysis/pytorch_analyse.py"
if [ -f "$SCAN_TOOL" ]; then
    export PYTHONPATH="$ASCEND_TOOLKIT_HOME/tools/ms_fmk_transplt:$PYTHONPATH"
    timeout 600 python "$SCAN_TOOL" -i "$MODEL_SRC" -o "$SCAN_OUT_DIR" -v 2.6.0 -m torch_apis \
        > "$LOG_DIR/scan.log" 2>&1 || log "WARN: 扫描失败（继续执行）"
    UNSUPPORTED_CSV=$(find "$SCAN_OUT_DIR" -name "unsupported_api.csv" 2>/dev/null | head -1)
else
    log "WARN: 未找到 pytorch_analyse.py，跳过扫描"
    echo "SCAN_TOOL_NOT_FOUND" > "$LOG_DIR/scan.log"
fi

# ---- 2. 代码迁移 + 算子替换 ----
log "2/4 算子替换 + CUDA→NPU..."
if [ -n "$UNSUPPORTED_CSV" ]; then
    # 2.1 先算子替换（全量复制 + 替换 _ext 调用 + 生成 gorilla/ascend）
    log "2.1 算子替换..."
    python -c "
import sys
sys.path.insert(0, '$TOOL_DIR')
from pathlib import Path
from rewriter.rewriter_core import rewrite_unsupported_ops
result = rewrite_unsupported_ops(
    unsupported_csv=Path('$UNSUPPORTED_CSV'),
    local_ops_csv=Path('$TOOL_DIR/patcher/local_op_lib/local_ops.csv'),
    output_dir=Path('$MODEL_OUT'),
    source_dir=Path('$MODEL_SRC'),
)
print(result.status)
" > "$LOG_DIR/rewrite.log" 2>&1 || log "WARN: 算子替换失败（继续执行）"

    # 2.2 再 CUDA→NPU 迁移输出（跳过已生成的 gorilla/ascend）
    log "2.2 CUDA→NPU 迁移..."
    python -c "
import sys
sys.path.insert(0, '$TOOL_DIR')
from pathlib import Path
from migrator.torch_to_npu import transform_source
py_files = list(Path('$MODEL_OUT').rglob('*.py'))
changes = 0
skipped = set()
for f in py_files:
    if 'ascend_pointnet2' in f.name or 'gorilla' in f.name:
        skipped.add(f.name)
        continue
    src = f.read_text(encoding='utf-8')
    new_src, c = transform_source(src)
    if c > 0:
        f.write_text(new_src, encoding='utf-8')
        changes += c
print(f'迁移完成: {changes} 处变更, 跳过 {len(skipped)} 个文件 ({skipped})')
" >> "$LOG_DIR/rewrite.log" 2>&1 || log "WARN: NPU迁移失败（继续执行）"
else
    log "WARN: 未找到 unsupported_api.csv，跳过算子替换"
    echo "NO_UNSUPPORTED_CSV" > "$LOG_DIR/rewrite.log"
fi

# ---- 3. 推理测试 ----
log "3/4 推理测试..."

_RUN_INF=0

if [ -n "$_INF_CMD" ]; then
    # 用户指定了完整推理命令（第3参数）
    log "使用自定义推理命令..."
    cd "$MODEL_OUT" 2>/dev/null || cd "$TOOL_DIR"
    bash -c "$_INF_CMD" > "$LOG_DIR/inference.log" 2>&1 &
    _RUN_INF=1
elif [ -n "$_INF_SCRIPT" ]; then
    # 外部通过 ASCEND_INF_SCRIPT 指定
    _INF_DIR=$(dirname "$_INF_SCRIPT")
    log "使用推理脚本: $_INF_SCRIPT"
    cd "$_INF_DIR" 2>/dev/null || cd "$MODEL_OUT" 2>/dev/null || cd "$TOOL_DIR"
    $_INF_PYTHON "$_INF_SCRIPT" > "$LOG_DIR/inference.log" 2>&1 &
    _RUN_INF=1
else
    # 自动查找推理脚本（仅限当前模型输出目录内）
    _AUTO_SCRIPT=$(find "$MODEL_OUT" -maxdepth 5 -name "*.py" -path "*run_*" 2>/dev/null | head -1)
    if [ -z "$_AUTO_SCRIPT" ]; then
        _AUTO_SCRIPT=$(find "$MODEL_OUT" -maxdepth 5 -name "*.py" -path "*infer*" -o -name "*.py" -path "*demo*" 2>/dev/null | head -1)
    fi
    if [ -n "$_AUTO_SCRIPT" ]; then
        _INF_DIR=$(dirname "$_AUTO_SCRIPT")
        log "自动找到推理脚本: $_AUTO_SCRIPT"
        cd "$_INF_DIR" 2>/dev/null || cd "$MODEL_OUT" 2>/dev/null
        $_INF_PYTHON "$_AUTO_SCRIPT" > "$LOG_DIR/inference.log" 2>&1 &
        _RUN_INF=1
    else
        log "WARN: 未找到推理脚本，跳过推理"
        echo "INFERENCE_SKIPPED" >> "$LOG_DIR/status.txt"
    fi
fi

if [ $_RUN_INF -eq 1 ]; then
    INF_PID=$!
    (sleep 1800; kill $INF_PID 2>/dev/null) &
    TIMEOUT_PID=$!
    wait $INF_PID 2>/dev/null
    INF_EXIT=$?
    kill $TIMEOUT_PID 2>/dev/null
    if grep -qE "Traceback \(most recent call\)|Error|AssertionError|RuntimeError" "$LOG_DIR/inference.log" 2>/dev/null; then
        log "  ❌ 推理测试报错"
        echo "INFERENCE_ERROR" >> "$LOG_DIR/status.txt"
    else
        log "  ✅ 推理测试完成"
        echo "SUCCESS" >> "$LOG_DIR/status.txt"
    fi
else
    log "推理已跳过"
fi

cd "$TOOL_DIR"

# ---- 4. 生成轮次摘要 ----
log "4/4 生成摘要..."
{
    echo "=== Pipeline Round $ROUND Summary ==="
    echo "Model: $MODEL_NAME"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo ""
    echo "--- scan.log (tail 30) ---"
    tail -30 "$LOG_DIR/scan.log" 2>/dev/null || echo "N/A"
    echo ""
    echo "--- rewrite.log (tail 30) ---"
    tail -30 "$LOG_DIR/rewrite.log" 2>/dev/null || echo "N/A"
    echo ""
    echo "--- inference.log (tail 50) ---"
    tail -50 "$LOG_DIR/inference.log" 2>/dev/null || echo "N/A"
    echo ""
    echo "--- status ---"
    cat "$LOG_DIR/status.txt" 2>/dev/null || echo "NO_STATUS"
} > "$LOG_DIR/summary.txt"

log "完成"
echo "DONE" >> "$LOG_DIR/status.txt"
