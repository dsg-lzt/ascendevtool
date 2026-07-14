#!/bin/bash
# op_pipeline_loop.sh — 算子开发循环（git poll → run → push logs → repeat）
# 用法: bash scripts/op_pipeline_loop.sh <op_name> [max_rounds]

OP_NAME="${1}"
if [ -z "$OP_NAME" ]; then
    echo "用法: bash op_pipeline_loop.sh <op_name>"
    exit 1
fi
MAX_ROUNDS="${2:-9999}"

TOOL_DIR="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$TOOL_DIR" ]; then
    echo "[FATAL] 不在 git 仓库内，无法确定项目根目录"
    exit 1
fi

LOG_ROOT="$TOOL_DIR/logs/op_${OP_NAME}"
LOOP_LOG="$LOG_ROOT/loop.log"
LAST_COMMIT_FILE="/tmp/.op_pipeline_last_commit_${OP_NAME}"

round=0

cd "$TOOL_DIR" || exit 1
export GIT_MERGE_AUTOEDIT=no
git config user.email "op-pipeline@ascend-dev.local" 2>/dev/null || true
git config user.name "Op Pipeline Bot" 2>/dev/null || true

log() {
    echo "[OP-LOOP] $(date '+%H:%M:%S') $*"
}

# ==================== DIAG ====================
{
    log "=== DIAG ==="
    log "TOOL_DIR=$TOOL_DIR"
    log "LOG_ROOT=$LOG_ROOT"
    log "GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo 'NOT_A_GIT_REPO')"
    log "PWD=$(pwd)"

    if [ -n "$GIT_TOKEN" ]; then
        log "GIT_TOKEN 已设置 (${#GIT_TOKEN} chars)"
        if git ls-remote origin HEAD > /dev/null 2>&1; then
            log "GIT_TOKEN 认证通过 ✓"
        else
            log "⚠ GIT_TOKEN 认证失败 — git push 将无法上传日志"
        fi
    else
        log "⚠ GIT_TOKEN 未设置 — git push 将无法上传日志"
    fi
} >> "$LOOP_LOG" 2>&1
# ==================== DIAG END ====================

# 首次 pull
git checkout -- . >> "$LOOP_LOG" 2>&1
for i in 1 2 3; do
    timeout 30 git pull origin master >> "$LOOP_LOG" 2>&1 && break
    sleep 5
done
LAST_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "")
echo "$LAST_COMMIT" > "$LAST_COMMIT_FILE"
log "算子: $OP_NAME | 初始 commit: $LAST_COMMIT" >> "$LOOP_LOG" 2>&1

# 清理旧日志（在 git pull 之后，确保旧日志不会从 git 恢复回来）
mkdir -p "$LOG_ROOT"
rm -f "$LOOP_LOG"
rm -rf "$LOG_ROOT"/run_*

while [ $round -lt $MAX_ROUNDS ]; do
    cd "$TOOL_DIR" || exit 1

    if [ $round -eq 0 ]; then
        round=1
        log "=== 第 1/$MAX_ROUNDS 轮 ($OP_NAME) ===" >> "$LOOP_LOG" 2>&1
    else
        # poll: fetch 并检查是否有新 commit
        for i in 1 2 3; do
            timeout 30 git fetch origin master >> "$LOOP_LOG" 2>&1 && break
            sleep 5
        done

        CURRENT_REMOTE=$(git rev-parse origin/master 2>/dev/null || echo "")
        SAVED_COMMIT=$(cat "$LAST_COMMIT_FILE" 2>/dev/null || echo "")

        if [ "$CURRENT_REMOTE" = "$SAVED_COMMIT" ]; then
            sleep 30
            continue
        fi

        round=$((round + 1))
        log "=== 第 $round/$MAX_ROUNDS 轮 ($OP_NAME) ===" >> "$LOOP_LOG" 2>&1

        # 清理工作区 + 拉最新代码
        git checkout -- . >> "$LOOP_LOG" 2>&1
        for i in 1 2 3; do
            timeout 30 git pull origin master >> "$LOOP_LOG" 2>&1 && break
            sleep 5
        done
        echo "$CURRENT_REMOTE" > "$LAST_COMMIT_FILE"
    fi

    log "执行 op_pipeline_run.sh $OP_NAME..." >> "$LOOP_LOG" 2>&1
    bash "$TOOL_DIR/scripts/op_pipeline_run.sh" "$round" "$OP_NAME" >> "$LOOP_LOG" 2>&1
    RUN_EXIT=$?

    log "op_pipeline_run.sh 退出码: $RUN_EXIT" >> "$LOOP_LOG" 2>&1

    # 提交并推送日志
    cd "$TOOL_DIR" || exit 1
    git add "${LOG_ROOT}/" >> "$LOOP_LOG" 2>&1
    git commit -m "logs: op pipeline round $round ($OP_NAME)" >> "$LOOP_LOG" 2>&1 || true

    for i in 1 2 3; do
        timeout 30 git pull --rebase origin master >> "$LOOP_LOG" 2>&1 || { sleep 5; continue; }
        timeout 30 git push origin master >> "$LOOP_LOG" 2>&1 && break
        sleep 5
    done

    # 获取最新 HEAD 写入记录
    for i in 1 2 3; do
        timeout 30 git fetch origin master >> "$LOOP_LOG" 2>&1 && break
        sleep 5
    done
    CURRENT_REMOTE=$(git rev-parse origin/master 2>/dev/null || echo "")
    echo "$CURRENT_REMOTE" > "$LAST_COMMIT_FILE"

    # 检查测试是否通过
    STATUS_FILE="$LOG_ROOT/run_$(printf '%02d' $round)/status.txt"
    if grep -q "^TEST_OK$" "$STATUS_FILE" 2>/dev/null; then
        log "✅ 测试通过！" >> "$LOOP_LOG" 2>&1
        echo "OP_PIPELINE_SUCCESS_ROUND=$round" >> "$LOG_ROOT/loop_status.txt"
        git add "${LOG_ROOT}/" >> "$LOOP_LOG" 2>&1
        git commit -m "logs: OP SUCCESS round $round ($OP_NAME)" >> "$LOOP_LOG" 2>&1 || true
        for i in 1 2 3; do
            timeout 30 git pull --rebase origin master >> "$LOOP_LOG" 2>&1 || { sleep 5; continue; }
            timeout 30 git push origin master >> "$LOOP_LOG" 2>&1 && break
            sleep 5
        done
        exit 0
    fi

    sleep 5
done

log "达到最大轮次 $MAX_ROUNDS" >> "$LOOP_LOG" 2>&1
echo "OP_PIPELINE_MAX_ROUNDS_REACHED" >> "$LOG_ROOT/loop_status.txt"
git add "${LOG_ROOT}/" >> "$LOOP_LOG" 2>&1
git commit -m "logs: OP MAX ROUNDS ($OP_NAME)" >> "$LOOP_LOG" 2>&1 || true
for i in 1 2 3; do
    timeout 30 git pull --rebase origin master >> "$LOOP_LOG" 2>&1 || { sleep 5; continue; }
    timeout 30 git push origin master >> "$LOOP_LOG" 2>&1 && break
    sleep 5
done
