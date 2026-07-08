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

rm -rf "$LOG_DIR"
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
    # 名字可能不完全匹配，尝试大小写不敏感的模糊查找
    MATCH=$(find "$TOOL_DIR/op_builder/ops_src" -maxdepth 1 -type d -iname "*${OP_NAME}*" 2>/dev/null | head -1)
    if [ -n "$MATCH" ]; then
        REAL_NAME=$(basename "$MATCH" | sed 's/Sample$//')
        OP_SRC_DIR="$MATCH/FrameworkLaunch/$REAL_NAME"
    fi
    if [ ! -d "$OP_SRC_DIR" ]; then
        echo "可用的算子工程:"
        find "$TOOL_DIR/op_builder/ops_src" -maxdepth 1 -type d -name "*Sample" 2>/dev/null | while read d; do
            base=$(basename "$d" | sed 's/Sample$//')
            echo "  $base"
        done
        fail "未找到算子工程: op_builder/ops_src/${OP_NAME}Sample"
        exit 1
    fi
fi

# ---- 1. 编译 ----
log "1/4 编译算子..."
(
    cd "$TOOL_DIR"
    export PATH=/home/orange/miniconda3/envs/torch_npu/bin:$PATH
    export PYTHONPATH=/home/orange/miniconda3/envs/torch_npu/lib/python3.9/site-packages:$PYTHONPATH
    export ASCEND_PYTHON_EXECUTABLE=/home/orange/miniconda3/envs/torch_npu/bin/python3
    python3 -m pip install decorator -q 2>/dev/null || true
    cd "$OP_SRC_DIR"
    rm -rf build_out
    sed -i 's|--preset=default|--preset=default -DASCEND_PYTHON_EXECUTABLE=/home/orange/miniconda3/envs/torch_npu/bin/python3|g' build.sh

    bash build.sh > "$LOG_DIR/build.log" 2>&1 || true
    git checkout -- build.sh   # 撤销 sed 修改，避免 git push 冲突

    RUN_FILE=$(find "$OP_SRC_DIR/build_out" -maxdepth 1 -name "*.run" -type f 2>/dev/null | head -1)
    if [ -n "$RUN_FILE" ]; then
        echo "BUILD_OK" > "$LOG_DIR/status.txt"
        log "编译成功"
    else
        echo "BUILD_FAILED" > "$LOG_DIR/status.txt"
        fail ".run 包未生成"
    fi
)
if tail -1 "$LOG_DIR/status.txt" 2>/dev/null | grep -q "BUILD_FAILED"; then
    _gen_summary
    exit 1
fi

# ---- 2. 验证 ----
log "2/4 精度验证..."
(
    cd "$TOOL_DIR"
    source /home/orange/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate torch_npu 2>/dev/null || true
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
    source /home/orange/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate torch_npu 2>/dev/null || true
    # 直接查找并安装 .run 包
    RUN_FILE=$(find "$OP_SRC_DIR/build_out" -maxdepth 1 -name "*.run" -type f 2>/dev/null | head -1)
    if [ -n "$RUN_FILE" ]; then
        CUSTOM_PATH="$TOOL_DIR/oplib/custom_opp_packages"
        rm -rf "$CUSTOM_PATH" && mkdir -p "$CUSTOM_PATH"
        bash "$RUN_FILE" --extract="$CUSTOM_PATH" --quiet >> "$LOG_DIR/install.log" 2>&1
        if [ -f "$CUSTOM_PATH/install.sh" ]; then
            cd "$CUSTOM_PATH" && bash install.sh --quiet --install-path="$CUSTOM_PATH" >> "$LOG_DIR/install.log" 2>&1 && log "安装成功" && echo "INSTALL_OK" >> "$LOG_DIR/status.txt" && cd "$TOOL_DIR"
        else
            cd "$TOOL_DIR" && echo "INSTALL_FAILED" >> "$LOG_DIR/status.txt"
        fi
    else
        echo "INSTALL_FAILED" >> "$LOG_DIR/status.txt"
        log "WARN: 安装失败，未找到 .run 包"
    fi
)

# ---- 4. 运行测试 ----
log "4/4 运行测试..."
export ASCEND_SLOG_PRINT_TO_STDOUT=1
TEST_SCRIPT="${ASCEND_OP_TEST_SCRIPT:-}"
CUSTOM_OPP="$TOOL_DIR/oplib/custom_opp_packages"
SET_ENV="$CUSTOM_OPP/vendors/customize/bin/set_env.bash"
if [ -f "$SET_ENV" ]; then
    source "$SET_ENV" 2>/dev/null || true
    log "已设置自定义算子路径: $CUSTOM_OPP"
fi
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
