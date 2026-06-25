# AscendDevTool 项目上下文

## 项目概述

PyTorch 模型全流程迁移至 Ascend NPU 部署的工具链。目标硬件：Ascend 310P（香橙派 AI Studio Pro）。

## 目录结构

```
AscendDevTool/
├── gui/                    # PySide6+QML 界面
│   ├── main.py             # 入口
│   ├── backend.py          # 业务逻辑 (ScanWorker/MigrateWorker/RewriteWorker/DevWorker)
│   ├── scanner_core.py     # CANN 扫描封装
│   └── qml/Main.qml        # UI 布局
├── scanner/                # 扫描模块
│   └── reports/            # 扫描输出 (gitignore)
├── migrator/               # CUDA→NPU 代码转换
│   ├── torch_to_npu.py     # libcst AST 转换器
│   └── migrator_core.py    # 批量入口
├── rewriter/               # 算子分析/替换/拆分/开发
│   ├── op_database.py      # 知识库 (CANN支持列表 + 分解规则)
│   ├── op_analyzer.py      # 分析引擎 (4策略分级)
│   ├── code_patcher.py     # 源码级替换 + gorilla/config/mmcv 生成
│   ├── config_stub.py      # config.config 模板 (addict.Dict + NPU monkey-patch)
│   ├── gorilla_stub.py     # gorilla 桩 (310P 不支持时用)
│   ├── op_dev_agent.py     # msopgen 脚手架 + cannbot-skills 开发提示
│   └── rewriter_core.py    # 主协调器
├── op_builder/             # Ascend C 算子开发车间
│   ├── op_manager.py       # msopgen 生命周期管理
│   ├── ops_src/            # 算子源码工程
│   │   ├── BallQuerySample/
│   │   ├── ThreeNNSample/
│   │   ├── pointnet2__ext_furthest_point_samplingSample/  ← 已实现 kernel
│   │   └── ...
│   └── skill_docs/         # Ascend C API 参考
├── patcher/                # Monkey Patch 运行时替换
│   └── local_op_lib/
│       ├── local_ops.csv   # OP→Local_Module→Local_Func 映射表
│       └── custom_ops.py   # Python 层替身实现
├── scripts/                # CI/CD 流水线
│   ├── pipeline_init.sh    # 远程首次环境配置
│   ├── pipeline_loop.sh    # git 检测 + 循环执行 (最多10轮)
│   ├── pipeline_run.sh     # 单次流水线
│   └── README.md           # 远程部署指南
├── logs/                   # 流水线日志 (git 追踪)
├── env.md                  # 环境配置指南
├── environment.txt         # 远程交互日志 (用户用此文件传信息)
└── context.md              # 本文件
```

## 外部依赖

| 路径 | 用途 |
|------|------|
| `/home/lzt/ascend_project/cannbot-skills/` | Ascend C 开发知识库 (SKILL.md) |
| `/home/lzt/ascend_project/SAM-6D/` | 测试模型 |
| `/home/lzt/ascend_project/AscendDev_output/` | 扫描+重写输出 |
| Conda 环境 `torch_npu` (`/home/orange/miniconda3/envs/torch_npu/`) | 远程 NPU 推理环境 |
| `ascenddevtool/` venv | 工具自身运行环境 |

## GitHub 工作流

**仓库**：`https://github.com/dsg-lzt/ascendevtool.git`

```
本地 (无 NPU)                        远程服务器 (有 NPU, 用户 orange)
──────────                          ──────────────────────
1. 修改代码                           
2. git commit && git push ──────→    pipeline_loop.sh 检测到新提交
                                     ├─ git pull
                                     ├─ pipeline_run.sh (扫描→迁移→替换→推理)
                                     ├─ 日志写入 AscendDevTool/logs/
                                     └─ git push logs ──────→
3. git pull 拉取日志 ←─────────────
4. 分析报错，修改代码 → 回到步骤1

循环最多10轮 或 推理SUCCESS后停止
```

**远程操作**：
```bash
# 首次
git clone https://github.com/dsg-lzt/ascendevtool.git AscendDevTool
cd AscendDevTool && bash scripts/pipeline_init.sh

# 启动循环
export GIT_TOKEN=ghp_xxx
cd ~/pipeline_tool
nohup bash AscendDevTool/scripts/pipeline_loop.sh > pipeline_loop.log 2>&1 &

# 重启 (代码更新后)
pkill -f pipeline_loop
cd ~/pipeline_tool/AscendDevTool && git pull
cd ~/pipeline_tool && nohup bash AscendDevTool/scripts/pipeline_loop.sh > pipeline_loop.log 2>&1 &
```

## 当前状态

### SAM-6D 推理：已通过 ✅

```
✅ CANN 扫描 (14 unsupported APIs)
✅ CUDA→NPU 迁移 (88处 .cuda→.npu 变更)
✅ gorilla 替换 (config.config + 直接 torch.load)
✅ PointNet2 算子替换:
   - gather_points → mmcv.ops.gather_points
   - group_points → mmcv.ops.grouping_operation
   - ball_query → mmcv.ops.ball_query / 纯PyTorch fallback
   - furthest_point_sampling → 纯PyTorch (待换 gather_ops_lib)
   - three_nn / three_interpolate → 纯PyTorch
✅ ViT attention → 手动 matmul+softmax (310P 不支持 fused attention)
✅ torch.cross → 纯PyTorch monkey-patch
✅ 模型推理: saving results + visualizating 完成
```

### 待完成

1. **精度调优** — 手动版用 `gather_ops_lib` (编译好的 Ascend C 算子)，当前工具用纯PyTorch/部分mmcv，结果有偏差
   - FPS 替换为 `gather_ops_lib.furthest_point_sampling`
   - 确认 mmcv ball_query 参数 (手动版用 `ball_query(0, radius, nsample, xyz, new_xyz)`)
2. **Ascend C 算子开发** — 可映射/可拆分的算子已覆盖，但编译+入库流程还需完善
3. **泛用性测试** — 当前只测了 SAM-6D，需验证其他模型
4. **Pipeline 稳定性** — 循环偶尔因日志提交自触发 / git 分叉 / loop_status.txt 残留退出
5. **GUI 完善** — 命令行功能基本 OK，GUI 部分功能待打磨

## 关键代码入口

| 功能 | 入口 |
|------|------|
| 启动 GUI | `python gui/main.py` |
| 命令行扫描 | `python $ASCEND_TOOLKIT_HOME/tools/ms_fmk_transplt/analysis/pytorch_analyse.py -i <src> -o <out> -v 2.6.0 -m torch_apis` |
| 命令行迁移 | `python -c "from rewriter.rewriter_core import rewrite_unsupported_ops; ..."` |
| 算子替换规则 | `rewriter/op_database.py` → `_DECOMPOSE_RULES` 和 `_OP_TO_REPLACEMENT_FUNC` |
| 替换策略 | `rewriter/op_analyzer.py` → 已入库→可映射→可拆分→需开发 |
| NPU monkey-patch | `rewriter/config_stub.py` (attention + cross) |
| 生成替换模块 | `rewriter/code_patcher.py` → `_generate_replacement_module()` |
| Gorilla 桩 | `rewriter/code_patcher.py` → `_generate_gorilla_stub()` + `rewriter/gorilla_stub.py` |
| 流水线脚本 | `scripts/pipeline_run.sh` (扫描→迁移→替换→推理) |

## 重要注意事项

1. **gorilla 替换**：源码中 `import gorilla` 会被自动替换为 `from config.config import Config`
2. **模型加载顺序**：必须先 `load_state_dict` 再 `.npu()`，代码里已通过行交换修复
3. **mmcv 算子**：310P 上 gather_points 和 grouping_operation 可用，ball_query 需传 `min_radius=0`
4. **310P 限制**：不支持 `F.scaled_dot_product_attention` (用手动 matmul+softmax)、不支持 `torch.cross` (用手动实现)、不支持 `npu_fusion_attention`
5. **环境**：推理用 conda `torch_npu` 环境，工具用 `ascenddevtool` venv，不可混淆
6. **远程交互**：用户通过 `environment.txt` 传日志/命令输出，agent 通过 git 接收
7. **GIT_TOKEN**：远程必须 export 才能 push 日志
