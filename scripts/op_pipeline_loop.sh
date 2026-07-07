#!/bin/bash
# op_pipeline_loop.sh — 算子开发 git 检测循环
# 后台: nohup bash -i scripts/op_pipeline_loop.sh <op_name> > op_loop.log 2>&1 &

OP_NAME="${1}"
if [ -z "$OP_NAME" ]; then
    echo "用法: bash op_pipeline_loop.sh <op_name> [max_rounds]"
    exit 1
fi
MAX_ROUNDS="${2:-9999}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOOL_DIR="$PIPELINE_ROOT/AscendDevTool"
LOG_ROOT="$TOOL_DIR/logs/op_${OP_NAME}"
LOOP_LOG="$LOG_ROOT/loop.log"
LAST_COMMIT_FILE="$PIPELINE_ROOT/.op_pipeline_last_commit_${OP_NAME}"

round=0
mkdir -p "$LOG_ROOT"
: > "$LOOP_LOG"
rm -rf "$LOG_ROOT"/run_*
rm -f "$LOG_ROOT/loop_status.txt" 2>/dev/null

cd "$TOOL_DIR"
git config user.email "op-pipeline@ascend-dev.local" 2>/dev/null || true
git config user.name "Op Pipeline Bot" 2>/dev/null || true

log() {
    local msg="[OP-LOOP] $(date '+%H:%M:%S') $*"
    echo "$msg"
    echo "$msg" >> "$LOOP_LOG"
}

git checkout -- .
git pull origin master || log "WARN: 初始 git pull 失败"
LAST_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "")
echo "$LAST_COMMIT" > "$LAST_COMMIT_FILE"
log "算子: $OP_NAME | 初始 commit: $LAST_COMMIT"

while [ $round -lt $MAX_ROUNDS ]; do
    cd "$TOOL_DIR"

    if [ $round -eq 0 ]; then
        round=1
        log "========================================"
        log "首次启动，第 1/$MAX_ROUNDS 轮 ($OP_NAME)"
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
        log "新提交，第 $round/$MAX_ROUNDS 轮 ($OP_NAME)"

        git checkout -- .
        git pull origin master || log "WARN: git pull 失败"
        echo "$CURRENT_REMOTE" > "$LAST_COMMIT_FILE"
    fi

    log "执行 op_pipeline_run.sh $OP_NAME..."
    bash "$TOOL_DIR/scripts/op_pipeline_run.sh" "$round" "$OP_NAME"

    cd "$TOOL_DIR"
    git checkout -- .

    log "上传日志..."
    git add "$LOG_ROOT/"
    git commit -m "logs: op pipeline round $round ($OP_NAME)" || true
    git pull --rebase origin master || true
    git push origin master || log "WARN: git push 失败"

    git fetch origin master 2>/dev/null
    CURRENT_REMOTE=$(git rev-parse origin/master 2>/dev/null || echo "")
    echo "$CURRENT_REMOTE" > "$LAST_COMMIT_FILE"

    STATUS_FILE="$LOG_ROOT/run_$(printf '%02d' $round)/status.txt"
    if tail -1 "$STATUS_FILE" 2>/dev/null | grep -q "^TEST_OK$"; then
        log "✅ 测试通过，退出循环"
        echo "OP_PIPELINE_SUCCESS_ROUND=$round" >> "$LOG_ROOT/loop_status.txt"
        git add "$LOG_ROOT/"
        git commit -m "logs: OP SUCCESS round $round ($OP_NAME)" || true
        git pull --rebase origin master || true
        git push origin master || true
        exit 0
    fi

    sleep 5
done

log "达到最大轮次 $MAX_ROUNDS，退出"
echo "OP_PIPELINE_MAX_ROUNDS_REACHED" >> "$LOG_ROOT/loop_status.txt"
git add "$LOG_ROOT/"
git commit -m "logs: OP MAX ROUNDS reached ($OP_NAME)" || true
git pull --rebase origin master || true
git push origin master || true
