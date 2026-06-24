# Pipeline 脚本

远程 NPU 服务器自动测试流水线，与本地开发迭代配合使用。

## 远程部署结构

```
~/pipeline_tool/
├── AscendDevTool/          # 本仓库 (git clone)
├── SAM-6D/                 # 待转换模型 (手动放置)
├── Data/                   # 测试数据 → 软链到 SAM-6D/Data/
├── ascenddev_output/       # 扫描+重写输出
├── logs/                   # 每轮日志
├── pipeline_init.sh        # 首次环境配置
├── pipeline_loop.sh        # 主循环（检测 git → 执行 → 推日志）
└── pipeline_run.sh         # 单次流水线
```

## 使用流程

### 远程服务器（首次）

```bash
mkdir ~/pipeline_tool && cd ~/pipeline_tool

# 把脚本传上去（或直接用 git）
git clone https://github.com/dsg-lzt/ascendevtool.git AscendDevTool

# 放置 SAM-6D 和测试数据
cp -r /path/to/SAM-6D ~/pipeline_tool/
ln -sfn ~/pipeline_tool/SAM-6D/SAM-6D/Data ~/pipeline_tool/Data

# 配置环境
bash AscendDevTool/scripts/pipeline_init.sh

# 启动后台循环
nohup bash AscendDevTool/scripts/pipeline_loop.sh > pipeline_loop.log 2>&1 &
```

### 本地开发机

1. 修改代码 → `git push`
2. 等待远程自动执行（约 30 秒检测一次）
3. `git pull` 拉取 `logs/run_NN/` 日志
4. 根据报错修改 → 回到步骤 1

## 脚本说明

| 脚本 | 执行方式 | 功能 |
|------|---------|------|
| `pipeline_init.sh` | 手动执行一次 | conda、venv、依赖安装 |
| `pipeline_loop.sh` | `nohup bash ... &` 后台常驻 | git 变更检测 + 循环（最多10轮） |
| `pipeline_run.sh` | 被 loop 调用 | 扫描→迁移→替换→推理→日志 |

## 循环终止条件

- 推理成功（`inference.log` 无报错）→ 自动停止
- 达到 10 轮 → 停止并推 `MAX_ROUNDS` 标记
- 手动 `kill` 进程

## 日志结构

```
logs/run_01/
├── scan.log          # CANN 扫描输出
├── rewrite.log       # 算子替换输出
├── inference.log     # SAM-6D 推理输出
├── summary.txt       # 轮次摘要
└── status.txt        # SUCCESS / EXIT_CODE=X / PIPELINE_ERROR
```
