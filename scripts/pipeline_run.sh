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

cd "$TOOL_DIR"

# ---- 1. CANN 扫描 ----
log "1/4 CANN 扫描 SAM-6D..."

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
    timeout 600 python "$SCAN_TOOL" -i "$SAM6D_SRC" -o "$SCAN_OUT_DIR" -v 2.6.0 -m torch_apis \
        > "$LOG_DIR/scan.log" 2>&1 || log "WARN: 扫描失败（继续执行）"
    # 更新 unsupported_api.csv 路径为最新扫描结果
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
    output_dir=Path('$SAM6D_OUT'),
    source_dir=Path('$SAM6D_SRC'),
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
py_files = list(Path('$SAM6D_OUT').rglob('*.py'))
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

# ---- 3. SAM-6D 推理测试（使用服务器已有的 torch_npu 环境）----
log "3/4 SAM-6D 推理测试..."
INFERENCE_DIR=$(find "$SAM6D_OUT" -name "run_inference_custom.py" -path "*/Pose_Estimation_Model/*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
if [ -n "$INFERENCE_DIR" ] && [ -f "$INFERENCE_DIR/run_inference_custom.py" ]; then
    cd "$INFERENCE_DIR"
    # 确保 CWD 有效（避免 rewrite 重建输出目录后 CWD 失效）
    if [ ! -d "$PWD" ]; then
        cd "$(dirname "$INFERENCE_DIR")" || cd "$HOME"
        INFERENCE_DIR="$PWD"
    fi

    DATA_DIR="$SAM6D_SRC/SAM-6D/Data/Example"
    OUTPUT_DIR="$DATA_DIR/outputs"
    CAD_PATH="$DATA_DIR/obj_000005.ply"
    RGB_PATH="$DATA_DIR/rgb.png"
    DEPTH_PATH="$DATA_DIR/depth.png"
    CAM_PATH="$DATA_DIR/camera.json"
    SEG_PATH="$OUTPUT_DIR/sam6d_results/detection_ism.json"

    mkdir -p "$OUTPUT_DIR" "$(dirname "$SEG_PATH")"

    SAM6D_PYTHON="${SAM6D_PYTHON:-/home/orange/miniconda3/envs/torch_npu/bin/python}"
    export PYTHONHTTPSVERIFY=0
    export CURL_CA_BUNDLE=""
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
    # 推理超时 30 分钟
    (sleep 1800; kill $INF_PID 2>/dev/null) &
    TIMEOUT_PID=$!
    wait $INF_PID 2>/dev/null
    INF_EXIT=$?
    kill $TIMEOUT_PID 2>/dev/null
    # 检查推理日志是否有错误，而非只看退出码
    if grep -qE "Error|Traceback|AssertionError|RuntimeError" "$LOG_DIR/inference.log" 2>/dev/null; then
        log "  ❌ 推理测试报错"
        echo "INFERENCE_ERROR" >> "$LOG_DIR/status.txt"
    fi
else
    log "WARN: 未找到 run_inference_custom.py，跳过推理"
    echo "INFERENCE_SKIPPED" >> "$LOG_DIR/status.txt"
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
