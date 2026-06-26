# Pipeline 脚本（泛化版）

远程 NPU 服务器自动测试流水线，支持任意模型。

## 远程部署结构

```
~/pipeline_tool/
├── AscendDevTool/          # 本仓库 (git clone)
├── <model_name>/           # 待转换模型 (sam2, SAM-6D 等)
├── ascenddev_output/       # 输出目录
│   ├── <model>_scan/       # 扫描结果
│   └── <model>_NPU/        # 迁移后代码
└── pipeline_loop.log       # 循环日志
```

## 使用流程

### 远程服务器（首次）

```bash
mkdir ~/pipeline_tool && cd ~/pipeline_tool
git clone https://github.com/dsg-lzt/ascendevtool.git AscendDevTool

# 放置模型源码
cp -r /path/to/sam2 ~/pipeline_tool/

# 配置环境
bash AscendDevTool/scripts/pipeline_init.sh

# 后台启动循环
nohup bash AscendDevTool/scripts/pipeline_loop.sh sam2 > pipeline_loop.log 2>&1 &
```

### 本地开发机

1. 修改代码 → `git push`
2. 等待远程自动执行（约 30 秒检测一次）
3. `git pull` 拉取 `logs/run_NN/` 日志
4. 根据报错修改 → 回到步骤 1

## 命令行手动执行

```bash
# 单次执行（不循环）
bash scripts/pipeline_run.sh 1 <model_name>

# 自定义推理脚本
INFERENCE_SCRIPT=/path/to/my_infer.py bash scripts/pipeline_run.sh 1 sam2

# 自定义推理命令
INFERENCE_CMD="python run.py --arg1 val" bash scripts/pipeline_run.sh 1 sam2

# 指定推理用的 Python
INFERENCE_PYTHON=/home/orange/miniconda3/envs/torch_npu/bin/python \
  bash scripts/pipeline_run.sh 1 sam2
```

## 脚本说明

| 脚本 | 参数 | 功能 |
|------|------|------|
| `pipeline_run.sh` | `<round> <model_name> [inference_cmd]` | 扫描→迁移→替换→推理 |
| `pipeline_loop.sh` | `<model_name> [max_rounds]` | git 变更检测 + 循环（默认10轮） |

## 循环终止条件

- 推理成功（`inference.log` 无 Traceback/Error）→ 自动停止 ✅
- 达到最大轮次 → 推 `MAX_ROUNDS` 标记
- 手动 `kill` 进程

## 日志结构

```
logs/run_01/
├── scan.log          # CANN 扫描输出
├── rewrite.log       # 算子替换输出
├── inference.log     # 推理输出
├── summary.txt       # 轮次摘要（含模型名）
└── status.txt        # SUCCESS / INFERENCE_ERROR / INFERENCE_SKIPPED
```
