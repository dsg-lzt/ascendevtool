#!/bin/bash
# ============================================================
# pipeline_run.sh — 单次流水线执行
# 被 pipeline_loop.sh 调用
# 用法: bash pipeline_run.sh <round_number>
# ============================================================

ROUND="${1:-01}"
ROUND=$(printf '%02d' "$ROUND")
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if echo "$SCRIPT_DIR" | grep -q "/AscendDevTool/"; then
    PIPELINE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"
fi
TOOL_DIR="$PIPELINE_ROOT/AscendDevTool"
SAM6D_SRC="$PIPELINE_ROOT/SAM-6D"
SAM6D_OUT="$PIPELINE_ROOT/ascenddev_output/SAM-6D_NPU"
SCAN_OUT="$PIPELINE_ROOT/ascenddev_output/scan"
LOG_DIR="$TOOL_DIR/logs/run_$ROUND"

mkdir -p "$LOG_DIR"

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

# 自动检测 CANN 安装路径
for d in "$HOME/Ascend/ascend-toolkit/latest" "/usr/local/Ascend/ascend-toolkit/latest" "$ASCEND_TOOLKIT_HOME"; do
    if [ -f "$d/tools/ms_fmk_transplt/analysis/pytorch_analyse.py" ]; then
        ASCEND_TOOLKIT_HOME="$d"
        break
    fi
done
# 也检查 cann 路径
if [ ! -f "$ASCEND_TOOLKIT_HOME/tools/ms_fmk_transplt/analysis/pytorch_analyse.py" ]; then
    for d in "$HOME/Ascend/cann/latest" "/usr/local/Ascend/cann/latest"; do
        if [ -f "$d/tools/ms_fmk_transplt/analysis/pytorch_analyse.py" ]; then
            ASCEND_TOOLKIT_HOME="$d"
            break
        fi
    done
fi

export ASCEND_TOOLKIT_HOME
export ASCEND_HOME_PATH="$ASCEND_TOOLKIT_HOME"
export ASCEND_OPP_PATH="$ASCEND_TOOLKIT_HOME/opp"
export PATH="$ASCEND_TOOLKIT_HOME/bin:$ASCEND_TOOLKIT_HOME/compiler/ccec_compiler/bin:$PATH"
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_HOME/x86_64-linux/devlib:$ASCEND_TOOLKIT_HOME/runtime/lib64/stub:$LD_LIBRARY_PATH"
export PYTHONPATH="$ASCEND_TOOLKIT_HOME/tools/ms_fmk_transplt:$ASCEND_TOOLKIT_HOME/python/site-packages:$PYTHONPATH"

cd "$TOOL_DIR"

# ---- 1. CANN 扫描 ----
log "1/4 CANN 扫描 SAM-6D..."
mkdir -p "$SCAN_OUT"
SCAN_TOOL="$ASCEND_TOOLKIT_HOME/tools/ms_fmk_transplt/analysis/pytorch_analyse.py"
if [ -f "$SCAN_TOOL" ]; then
    # 修复 CANN resource 文件属主（root 安装问题，只需执行一次）
    SCAN_RES="$ASCEND_TOOLKIT_HOME/tools/ms_fmk_transplt/resource"
    if [ -d "$SCAN_RES" ]; then
        CURRENT_OWNER=$(stat -c %U "$SCAN_RES/op_list_2_6.json" 2>/dev/null || echo "")
        if [ "$CURRENT_OWNER" != "$USER" ] && [ "$CURRENT_OWNER" = "root" ]; then
            sudo chown -R "$USER:$USER" "$SCAN_RES" 2>/dev/null || true
        fi
    fi
    export PYTHONPATH="$ASCEND_TOOLKIT_HOME/tools/ms_fmk_transplt:$PYTHONPATH"
    python "$SCAN_TOOL" -i "$SAM6D_SRC" -o "$SCAN_OUT" -v 2.6.0 -m torch_apis \
        > "$LOG_DIR/scan.log" 2>&1 || log "WARN: 扫描失败（继续执行）"
else
    log "WARN: 未找到 pytorch_analyse.py，跳过扫描"
    echo "SCAN_TOOL_NOT_FOUND" > "$LOG_DIR/scan.log"
fi

# ---- 2. 代码迁移 + 算子替换 ----
log "2/4 代码迁移(NPU) + 算子替换..."
UNSUPPORTED_CSV=$(find "$SCAN_OUT" -name "unsupported_api.csv" 2>/dev/null | head -1)
if [ -n "$UNSUPPORTED_CSV" ]; then
    python -c "
import sys
sys.path.insert(0, '$TOOL_DIR')
from pathlib import Path
from rewriter.rewriter_core import rewrite_unsupported_ops
result = rewrite_unsupported_ops(
    unsupported_csv=Path('$UNSUPPORTED_CSV'),
    local_ops_csv=Path('$TOOL_DIR/patcher/local_op_lib/local_ops.csv'),
    output_dir=Path('$SAM6D_OUT'),
    source_dir=Path('$SAM6D_SRC'),
)
print(result.status)
" > "$LOG_DIR/rewrite.log" 2>&1 || log "WARN: 算子替换失败（继续执行）"
else
    log "WARN: 未找到 unsupported_api.csv，跳过算子替换"
    echo "NO_UNSUPPORTED_CSV" > "$LOG_DIR/rewrite.log"
fi

# ---- 3. SAM-6D 推理测试（使用服务器已有的 torch_npu 环境）----
log "3/4 SAM-6D 推理测试..."
INFERENCE_DIR="$SAM6D_OUT/Pose_Estimation_Model"
if [ -f "$INFERENCE_DIR/run_inference_custom.py" ]; then
    cd "$INFERENCE_DIR"

    DATA_DIR="$SAM6D_SRC/Data/Example"
    OUTPUT_DIR="$DATA_DIR/outputs"
    CAD_PATH="$DATA_DIR/obj_000005.ply"
    RGB_PATH="$DATA_DIR/rgb.png"
    DEPTH_PATH="$DATA_DIR/depth.png"
    CAM_PATH="$DATA_DIR/camera.json"
    SEG_PATH="$OUTPUT_DIR/sam6d_results"

    mkdir -p "$OUTPUT_DIR" "$SEG_PATH"

    # 用服务器已有的 torch_npu python 运行（非 ascenddevtool venv）
    SAM6D_PYTHON="${SAM6D_PYTHON:-python3}"
    log "使用 $SAM6D_PYTHON 运行推理..."

    $SAM6D_PYTHON run_inference_custom.py \
        --output_dir "$OUTPUT_DIR" \
        --cad_path "$CAD_PATH" \
        --rgb_path "$RGB_PATH" \
        --depth_path "$DEPTH_PATH" \
        --cam_path "$CAM_PATH" \
        --seg_path "$SEG_PATH" \
        > "$LOG_DIR/inference.log" 2>&1 &
    INF_PID=$!
    wait $INF_PID 2>/dev/null || true
    INF_EXIT=$?
    if [ $INF_EXIT -ne 0 ]; then
        log "WARN: 推理测试退出码 $INF_EXIT"
        echo "EXIT_CODE=$INF_EXIT" >> "$LOG_DIR/status.txt"
    else
        log "SUCCESS: 推理测试完成"
        echo "SUCCESS" >> "$LOG_DIR/status.txt"
    fi
else
    log "WARN: 未找到 run_inference_custom.py，跳过推理"
fi

cd "$TOOL_DIR"

# ---- 4. 生成轮次摘要 ----
log "4/4 生成摘要..."
{
    echo "=== Pipeline Round $ROUND Summary ==="
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
