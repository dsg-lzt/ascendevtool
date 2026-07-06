#!/bin/bash
# ============================================================
# op_pipeline_loop.sh — 算子开发 git 检测循环（最多10轮）
# 用法: bash scripts/op_pipeline_loop.sh <op_name> [max_rounds]
# 后台: nohup bash scripts/op_pipeline_loop.sh <op_name> > op_loop.log 2>&1 &
# 注: 测试脚本由本地生成并推送到 op_builder/ops_src/ 下，管道自动查找
# ============================================================

OP_NAME="${1}"
if [ -z "$OP_NAME" ]; then
    echo "用法: bash op_pipeline_loop.sh <op_name> [max_rounds]"
    echo "示例: bash op_pipeline_loop.sh BallQuery 10"
    exit 1
fi
MAX_ROUNDS="${2:-20}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if echo "$SCRIPT_DIR" | grep -q "/AscendDevTool/"; then
    PIPELINE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"
fi
TOOL_DIR="$PIPELINE_ROOT/AscendDevTool"
LOG_ROOT="$TOOL_DIR/logs/op_${OP_NAME}"
LAST_COMMIT_FILE="$PIPELINE_ROOT/.op_pipeline_last_commit_${OP_NAME}"

round=0
mkdir -p "$LOG_ROOT"
rm -f "$LOG_ROOT/loop_status.txt" 2>/dev/null

# GIT_TOKEN 配置
if [ -n "$GIT_TOKEN" ]; then
    REPO_URL="https://${GIT_TOKEN}@github.com/dsg-lzt/ascendevtool.git"
    git remote set-url origin "$REPO_URL" 2>/dev/null
fi

cd "$TOOL_DIR"
git config pull.rebase false 2>/dev/null || true
git config user.email "op-pipeline@ascend-dev.local" 2>/dev/null || true
git config user.name "Op Pipeline Bot" 2>/dev/null || true

log() {
    echo "[OP-LOOP] $(date '+%H:%M:%S') $*"
}

# 拉取初始状态
cd "$TOOL_DIR"
git pull origin master 2>/dev/null || log "WARN: git pull 失败"
LAST_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "")
echo "$LAST_COMMIT" > "$LAST_COMMIT_FILE"
log "算子: $OP_NAME | 初始 commit: $LAST_COMMIT"

while [ $round -lt $MAX_ROUNDS ]; do
    cd "$TOOL_DIR"

    if [ $round -eq 0 ]; then
        round=1
        log "========================================"
        log "首次启动，立即执行第 1/$MAX_ROUNDS 轮 ($OP_NAME)"
    else
        git fetch origin master 2>/dev/null
        CURRENT_REMOTE=$(git rev-parse origin/master 2>/dev/null || echo "")
        LAST_COMMIT=$(cat "$LAST_COMMIT_FILE" 2>/dev/null || echo "")

        if [ "$CURRENT_REMOTE" = "$LAST_COMMIT" ]; then
            sleep 30
            continue
        fi

        round=$((round + 1))
        log "========================================"
        log "检测到新提交，开始第 $round/$MAX_ROUNDS 轮 ($OP_NAME)"
        log "  $LAST_COMMIT -> $CURRENT_REMOTE"

        git pull origin master 2>/dev/null || log "WARN: git pull 失败"
        echo "$CURRENT_REMOTE" > "$LAST_COMMIT_FILE"
    fi

    log "执行 op_pipeline_run.sh $OP_NAME..."
    bash "$TOOL_DIR/scripts/op_pipeline_run.sh" "$round" "$OP_NAME"
    PIPELINE_EXIT=$?

    log "上传日志..."
    git add "$LOG_ROOT/" 2>/dev/null || true
    git commit -m "logs: op pipeline round $round ($OP_NAME) [auto]" 2>/dev/null || log "WARN: 无日志变更"
    git push origin master 2>/dev/null || log "WARN: git push 失败"

    git fetch origin master 2>/dev/null
    CURRENT_REMOTE=$(git rev-parse origin/master 2>/dev/null || echo "")
    echo "$CURRENT_REMOTE" > "$LAST_COMMIT_FILE"

    STATUS_FILE="$LOG_ROOT/run_$(printf '%02d' $round)/status.txt"
    if tail -1 "$STATUS_FILE" 2>/dev/null | grep -q "^TEST_OK$"; then
        log "========================================"
        log "✅ 第 $round 轮算子测试通过！退出循环"
        echo "OP_PIPELINE_SUCCESS_ROUND=$round" >> "$LOG_ROOT/loop_status.txt"
        git add "$LOG_ROOT/" 2>/dev/null || true
        git commit -m "logs: OP PIPELINE SUCCESS at round $round ($OP_NAME)" 2>/dev/null || true
        git push origin master 2>/dev/null || true
        exit 0
    fi

    if [ $PIPELINE_EXIT -ne 0 ]; then continue; fi
    sleep 5
done

log "========================================"
log "达到最大轮次 $MAX_ROUNDS，退出循环"
echo "OP_PIPELINE_MAX_ROUNDS_REACHED" >> "$LOG_ROOT/loop_status.txt"
git add "$LOG_ROOT/" 2>/dev/null || true
git commit -m "logs: OP MAX ROUNDS reached ($OP_NAME) [auto]" 2>/dev/null || true
git push origin master 2>/dev/null || true
