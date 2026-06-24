#!/bin/bash
# ============================================================
# pipeline_loop.sh — git 变更检测 + 循环执行（最多10轮）
# 用法: nohup bash AscendDevTool/scripts/pipeline_loop.sh > pipeline_loop.log 2>&1 &
# ============================================================

MAX_ROUNDS=10
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 自动检测 PIPELINE_ROOT
if echo "$SCRIPT_DIR" | grep -q "/AscendDevTool/"; then
    PIPELINE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"
fi
TOOL_DIR="$PIPELINE_ROOT/AscendDevTool"
LOG_ROOT="$TOOL_DIR/logs"
LAST_COMMIT_FILE="$PIPELINE_ROOT/.pipeline_last_commit"

round=0
mkdir -p "$LOG_ROOT"

# GIT_TOKEN 配置
if [ -n "$GIT_TOKEN" ]; then
    REPO_URL="https://${GIT_TOKEN}@github.com/dsg-lzt/ascendevtool.git"
    git remote set-url origin "$REPO_URL" 2>/dev/null
fi

# Git 配置
cd "$TOOL_DIR"
git config user.email "pipeline@ascend-dev.local" 2>/dev/null || true
git config user.name "Pipeline Bot" 2>/dev/null || true

log() {
    echo "[LOOP] $(date '+%H:%M:%S') $*"
}

# 拉取初始状态
cd "$TOOL_DIR"
git pull origin master 2>/dev/null || log "WARN: git pull 失败"
LAST_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "")
echo "$LAST_COMMIT" > "$LAST_COMMIT_FILE"
log "初始 commit: $LAST_COMMIT"

while [ $round -lt $MAX_ROUNDS ]; do
    cd "$TOOL_DIR"

    # 第一轮立即执行，不需要等新提交
    if [ $round -eq 0 ]; then
        round=1
        log "========================================"
        log "首次启动，立即执行第 1/$MAX_ROUNDS 轮"
    else
        # 检测新提交
        git fetch origin master 2>/dev/null
        CURRENT_REMOTE=$(git rev-parse origin/master 2>/dev/null || echo "")
        LAST_COMMIT=$(cat "$LAST_COMMIT_FILE" 2>/dev/null || echo "")

        if [ "$CURRENT_REMOTE" = "$LAST_COMMIT" ]; then
            sleep 30
            continue
        fi

        round=$((round + 1))
        log "========================================"
        log "检测到新提交，开始第 $round/$MAX_ROUNDS 轮"
        log "  $LAST_COMMIT -> $CURRENT_REMOTE"

        git pull origin master 2>/dev/null || log "WARN: git pull 失败"
        echo "$CURRENT_REMOTE" > "$LAST_COMMIT_FILE"
    fi

    # 执行流水线
    log "执行 pipeline_run.sh..."
    bash "$TOOL_DIR/scripts/pipeline_run.sh" "$round"
    PIPELINE_EXIT=$?

    # 上传日志
    log "上传日志..."
    git add "$LOG_ROOT/" 2>/dev/null || true
    git commit -m "logs: pipeline round $round [auto]" 2>/dev/null || log "WARN: 无日志变更"
    git push origin master 2>/dev/null || log "WARN: git push 失败"

    # 更新 Last commit，避免把日志提交误判为"新变更"
    git fetch origin master 2>/dev/null
    CURRENT_REMOTE=$(git rev-parse origin/master 2>/dev/null || echo "")
    echo "$CURRENT_REMOTE" > "$LAST_COMMIT_FILE"

    # 检查是否成功
    STATUS_FILE="$LOG_ROOT/run_$(printf '%02d' $round)/status.txt"
    if grep -q "SUCCESS" "$STATUS_FILE" 2>/dev/null; then
        log "========================================"
        log "✅ 第 $round 轮推理成功！退出循环"
        echo "PIPELINE_SUCCESS_ROUND=$round" >> "$LOG_ROOT/loop_status.txt"
        git add "$LOG_ROOT/" 2>/dev/null || true
        git commit -m "logs: PIPELINE SUCCESS at round $round" 2>/dev/null || true
        git push origin master 2>/dev/null || true
        exit 0
    fi

    if [ $PIPELINE_EXIT -ne 0 ]; then
        log "WARN: 流水线退出码 $PIPELINE_EXIT"
    fi

    sleep 5
done

log "========================================"
log "达到最大轮次 $MAX_ROUNDS，退出循环"
echo "PIPELINE_MAX_ROUNDS_REACHED" >> "$LOG_ROOT/loop_status.txt"
git add "$LOG_ROOT/" 2>/dev/null || true
git commit -m "logs: MAX ROUNDS reached [auto]" 2>/dev/null || true
git push origin master 2>/dev/null || true
