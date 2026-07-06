#!/bin/bash
# ============================================================
# op_pipeline_run.sh — 算子开发流水线（编译 → 验证 → 安装 → 测试）
# 用法: bash scripts/op_pipeline_run.sh <round> <op_name>
# 测试脚本: export ASCEND_OP_TEST_SCRIPT=<脚本.py>
# ============================================================

ROUND="${1:-01}"
ROUND=$(printf '%02d' "$ROUND")
OP_NAME="${2}"
if [ -z "$OP_NAME" ]; then
    echo "用法: bash op_pipeline_run.sh <round> <op_name>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if echo "$SCRIPT_DIR" | grep -q "/AscendDevTool/"; then
    PIPELINE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"
fi
TOOL_DIR="$PIPELINE_ROOT/AscendDevTool"
LOG_DIR="$TOOL_DIR/logs/op_${OP_NAME}/run_$ROUND"

mkdir -p "$LOG_DIR"
: > "$LOG_DIR/status.txt"

log() {
    echo "[OP $OP_NAME R$ROUND] $(date '+%H:%M:%S') $*"
}

fail() {
    log "ERROR: $*"
    echo "BUILD_ERROR: $*" >> "$LOG_DIR/status.txt"
}

# ---- 0. 检查算子工程 ----
OP_SRC_DIR="$TOOL_DIR/op_builder/ops_src/${OP_NAME}Sample/FrameworkLaunch/${OP_NAME}"
if [ ! -d "$OP_SRC_DIR" ]; then
    # 名字可能不完全匹配，尝试模糊查找
    MATCH=$(find "$TOOL_DIR/op_builder/ops_src" -maxdepth 1 -type d -name "*${OP_NAME}*Sample" 2>/dev/null | head -1)
    if [ -n "$MATCH" ]; then
        REAL_NAME=$(basename "$MATCH" | sed 's/Sample$//')
        OP_SRC_DIR="$MATCH/FrameworkLaunch/$REAL_NAME"
    fi
    if [ ! -d "$OP_SRC_DIR" ]; then
        fail "未找到算子工程: op_builder/ops_src/${OP_NAME}Sample"
        exit 1
    fi
fi

# ---- 1. 编译 ----
log "1/4 编译算子..."
(
    cd "$TOOL_DIR"
    source ascenddevtool/bin/activate
    python op_builder/op_manager.py build "$OP_NAME" > "$LOG_DIR/build.log" 2>&1
    BUILD_EXIT=$?
    if [ $BUILD_EXIT -ne 0 ]; then
        echo "BUILD_FAILED" >> "$LOG_DIR/status.txt"
        fail "编译失败，详见 build.log"
    else
        echo "BUILD_OK" >> "$LOG_DIR/status.txt"
        log "编译成功"
    fi
)
if grep -q "BUILD_FAILED" "$LOG_DIR/status.txt" 2>/dev/null; then
    _gen_summary
    exit 1
fi

# ---- 2. 验证 ----
log "2/4 精度验证..."
(
    cd "$TOOL_DIR"
    source ascenddevtool/bin/activate
    python op_builder/op_manager.py verify "$OP_NAME" > "$LOG_DIR/verify.log" 2>&1
    VERIFY_EXIT=$?
    if [ $VERIFY_EXIT -ne 0 ]; then
        echo "VERIFY_FAILED" >> "$LOG_DIR/status.txt"
        log "WARN: 精度验证失败（继续安装）"
    else
        echo "VERIFY_OK" >> "$LOG_DIR/status.txt"
        log "精度验证通过"
    fi
)

# ---- 3. 安装 ----
log "3/4 安装算子..."
(
    cd "$TOOL_DIR"
    source ascenddevtool/bin/activate
    python op_builder/op_manager.py install "$OP_NAME" >> "$LOG_DIR/build.log" 2>&1
    INSTALL_EXIT=$?
    if [ $INSTALL_EXIT -ne 0 ]; then
        echo "INSTALL_FAILED" >> "$LOG_DIR/status.txt"
        fail "安装失败"
    else
        echo "INSTALL_OK" >> "$LOG_DIR/status.txt"
        log "安装成功"
    fi
)

# ---- 4. 运行测试 ----
log "4/4 运行测试..."
# 测试脚本通过环境变量传入（也可让管道自动从算子目录查找）
#   ASCEND_OP_TEST_SCRIPT — 测试脚本路径（相对 op_builder/ops_src/）
#   ASCEND_OP_TEST_PYTHON — Python 解释器
TEST_SCRIPT="${ASCEND_OP_TEST_SCRIPT:-}"
if [ -z "$TEST_SCRIPT" ]; then
    # 自动查找算子目录下的测试脚本
    TEST_SCRIPT=$(find "$TOOL_DIR/op_builder/ops_src" -maxdepth 2 -name "test_*.py" -path "*${OP_NAME}*" 2>/dev/null | head -1)
    if [ -z "$TEST_SCRIPT" ]; then
        TEST_SCRIPT=$(find "$TOOL_DIR/op_builder/ops_src" -maxdepth 2 \( -name "test_*.py" -o -name "*_test.py" \) 2>/dev/null | head -1)
    fi
fi
if [ -n "$TEST_SCRIPT" ]; then
    TEST_PYTHON="${ASCEND_OP_TEST_PYTHON:-/home/orange/miniconda3/envs/torch_npu/bin/python}"
    if [ -f "$TEST_SCRIPT" ]; then
        log "运行测试: $TEST_SCRIPT"
        cd "$(dirname "$TEST_SCRIPT")"
        $TEST_PYTHON "$TEST_SCRIPT" > "$LOG_DIR/test.log" 2>&1
        TEST_EXIT=$?
        if grep -qE "Traceback \(most recent call\)|Error|AssertionError|RuntimeError" "$LOG_DIR/test.log" 2>/dev/null; then
            echo "TEST_ERROR" >> "$LOG_DIR/status.txt"
            log "❌ 测试失败"
        elif [ $TEST_EXIT -eq 0 ]; then
            echo "TEST_OK" >> "$LOG_DIR/status.txt"
            log "✅ 测试通过"
        else
            echo "TEST_ERROR" >> "$LOG_DIR/status.txt"
            log "❌ 测试退出码: $TEST_EXIT"
        fi
    else
        log "WARN: 测试脚本不存在: $TEST_SCRIPT，跳过"
        echo "TEST_SKIPPED" >> "$LOG_DIR/status.txt"
    fi
else
    log "WARN: 未找到测试脚本，跳过测试"
    echo "TEST_SKIPPED" >> "$LOG_DIR/status.txt"
fi

# ---- 5. 生成摘要 ----
_gen_summary() {
    {
        echo "=== Op Pipeline Round $ROUND Summary ==="
        echo "Operator: $OP_NAME"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        echo "--- build.log (tail 20) ---"
        tail -20 "$LOG_DIR/build.log" 2>/dev/null || echo "N/A"
        echo ""
        echo "--- verify.log (tail 20) ---"
        tail -20 "$LOG_DIR/verify.log" 2>/dev/null || echo "N/A"
        echo ""
        echo "--- test.log (tail 30) ---"
        tail -30 "$LOG_DIR/test.log" 2>/dev/null || echo "N/A"
        echo ""
        echo "--- status ---"
        cat "$LOG_DIR/status.txt" 2>/dev/null || echo "NO_STATUS"
    } > "$LOG_DIR/summary.txt"
}
_gen_summary

log "完成"
echo "DONE" >> "$LOG_DIR/status.txt"
