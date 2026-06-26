#!/bin/bash
# ============================================================
# pipeline_loop.sh — git 变更检测 + 循环执行（泛化版）
# 用法:
#   仅扫描迁移:  bash scripts/pipeline_loop.sh <model_name>
#   指定推理脚本: ASCEND_INF_SCRIPT=/path/to/infer.py bash scripts/pipeline_loop.sh sam2
#   指定推理命令: ASCEND_INF_CMD="python infer.py --arg val" bash scripts/pipeline_loop.sh sam2
#   后台运行:      nohup bash scripts/pipeline_loop.sh sam2 > pipeline_loop.log 2>&1 &
# ============================================================

MODEL_NAME="${1:-SAM-6D}"
MAX_ROUNDS="${2:-10}"
# 推理配置通过环境变量传入（nohup 会继承）：
#   ASCEND_INF_SCRIPT — 推理脚本路径
#   ASCEND_INF_CMD    — 完整推理命令
#   ASCEND_INF_PYTHON — Python 解释器 (默认 /home/orange/.../torch_npu/bin/python)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
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
rm -f "$LOG_ROOT/loop_status.txt" 2>/dev/null

# GIT_TOKEN 配置
if [ -n "$GIT_TOKEN" ]; then
    REPO_URL="https://${GIT_TOKEN}@github.com/dsg-lzt/ascendevtool.git"
    git remote set-url origin "$REPO_URL" 2>/dev/null
fi

# Git 配置
cd "$TOOL_DIR"
git config pull.rebase false 2>/dev/null || true
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
log "模型: $MODEL_NAME | 初始 commit: $LAST_COMMIT"

while [ $round -lt $MAX_ROUNDS ]; do
    cd "$TOOL_DIR"

    if [ $round -eq 0 ]; then
        round=1
        log "========================================"
        log "首次启动，立即执行第 1/$MAX_ROUNDS 轮 ($MODEL_NAME)"
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
        log "检测到新提交，开始第 $round/$MAX_ROUNDS 轮 ($MODEL_NAME)"
        log "  $LAST_COMMIT -> $CURRENT_REMOTE"

        git pull origin master 2>/dev/null || log "WARN: git pull 失败"
        echo "$CURRENT_REMOTE" > "$LAST_COMMIT_FILE"
    fi

    log "执行 pipeline_run.sh $MODEL_NAME..."
    bash "$TOOL_DIR/scripts/pipeline_run.sh" "$round" "$MODEL_NAME"
    PIPELINE_EXIT=$?

    log "上传日志..."
    git add "$LOG_ROOT/" 2>/dev/null || true
    git commit -m "logs: pipeline round $round ($MODEL_NAME) [auto]" 2>/dev/null || log "WARN: 无日志变更"
    git push origin master 2>/dev/null || log "WARN: git push 失败"

    git fetch origin master 2>/dev/null
    CURRENT_REMOTE=$(git rev-parse origin/master 2>/dev/null || echo "")
    echo "$CURRENT_REMOTE" > "$LAST_COMMIT_FILE"

    STATUS_FILE="$LOG_ROOT/run_$(printf '%02d' $round)/status.txt"
    if tail -1 "$STATUS_FILE" 2>/dev/null | grep -q "^SUCCESS$"; then
        log "========================================"
        log "✅ 第 $round 轮推理成功！退出循环"
        echo "PIPELINE_SUCCESS_ROUND=$round" >> "$LOG_ROOT/loop_status.txt"
        git add "$LOG_ROOT/" 2>/dev/null || true
        git commit -m "logs: PIPELINE SUCCESS at round $round ($MODEL_NAME)" 2>/dev/null || true
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
git commit -m "logs: MAX ROUNDS reached ($MODEL_NAME) [auto]" 2>/dev/null || true
git push origin master 2>/dev/null || true
