#!/bin/bash
# op_pipeline_loop.sh

OP_NAME="${1}"
if [ -z "$OP_NAME" ]; then
    echo "用法: bash op_pipeline_loop.sh <op_name>"
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

cd "$TOOL_DIR"
export GIT_MERGE_AUTOEDIT=no
git config user.email "op-pipeline@ascend-dev.local" 2>/dev/null || true
git config user.name "Op Pipeline Bot" 2>/dev/null || true

log() {
    echo "[OP-LOOP] $(date '+%H:%M:%S') $*"
    echo "[OP-LOOP] $(date '+%H:%M:%S') $*" >> "$LOOP_LOG"
}

# 重试最多3次，单次超时30秒
git_pull()  { for i in 1 2 3; do timeout 30 git pull  origin master 2>/dev/null && return 0; sleep 5; done; return 1; }
git_fetch() { for i in 1 2 3; do timeout 30 git fetch origin master 2>/dev/null && return 0; sleep 5; done; return 1; }
git_push()  { for i in 1 2 3; do timeout 30 git push  origin master 2>/dev/null && return 0; sleep 5; done; return 1; }
git_rebase_push() {
    for i in 1 2 3; do
        timeout 30 git pull --rebase origin master 2>/dev/null
        timeout 30 git push origin master 2>/dev/null && return 0
        sleep 5
    done
    return 1
}

git checkout -- .
git_pull || log "WARN: 初始 git pull 失败"
LAST_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "")
echo "$LAST_COMMIT" > "$LAST_COMMIT_FILE"
log "算子: $OP_NAME | 初始 commit: $LAST_COMMIT"

while [ $round -lt $MAX_ROUNDS ]; do
    cd "$TOOL_DIR"

    if [ $round -eq 0 ]; then
        round=1
        log "=== 第 1/$MAX_ROUNDS 轮 ($OP_NAME) ==="
    else
        git_fetch
        CURRENT_REMOTE=$(git rev-parse origin/master 2>/dev/null || echo "")
        LAST_COMMIT=$(cat "$LAST_COMMIT_FILE" 2>/dev/null || echo "")

        if [ "$CURRENT_REMOTE" = "$LAST_COMMIT" ]; then
            sleep 30
            continue
        fi

        round=$((round + 1))
        log "=== 第 $round/$MAX_ROUNDS 轮 ($OP_NAME) ==="

        git checkout -- .
        git_pull || log "WARN: git pull 失败"
        echo "$CURRENT_REMOTE" > "$LAST_COMMIT_FILE"
    fi

    log "执行 op_pipeline_run.sh $OP_NAME..."
    bash "$TOOL_DIR/scripts/op_pipeline_run.sh" "$round" "$OP_NAME"

    cd "$TOOL_DIR"
    git checkout -- .

    log "上传日志..."
    git add "$LOG_ROOT/" 2>/dev/null
    git commit -m "logs: op pipeline round $round ($OP_NAME)" 2>/dev/null || true
    git_rebase_push || log "WARN: git push 失败"

    git_fetch
    CURRENT_REMOTE=$(git rev-parse origin/master 2>/dev/null || echo "")
    echo "$CURRENT_REMOTE" > "$LAST_COMMIT_FILE"

    STATUS_FILE="$LOG_ROOT/run_$(printf '%02d' $round)/status.txt"
    if tail -1 "$STATUS_FILE" 2>/dev/null | grep -q "^TEST_OK$"; then
        log "✅ 测试通过！"
        echo "OP_PIPELINE_SUCCESS_ROUND=$round" >> "$LOG_ROOT/loop_status.txt"
        git add "$LOG_ROOT/" 2>/dev/null
        git commit -m "logs: OP SUCCESS round $round ($OP_NAME)" 2>/dev/null || true
        git_rebase_push || true
        exit 0
    fi

    sleep 5
done

log "达到最大轮次 $MAX_ROUNDS"
echo "OP_PIPELINE_MAX_ROUNDS_REACHED" >> "$LOG_ROOT/loop_status.txt"
git add "$LOG_ROOT/" 2>/dev/null
git commit -m "logs: OP MAX ROUNDS ($OP_NAME)" 2>/dev/null || true
git_rebase_push || true
