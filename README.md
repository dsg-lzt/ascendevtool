# AscendDevTool — PyTorch 模型迁移至 Ascend NPU 工具链

目标硬件：Ascend 310P（香橙派 AI Studio Pro）

---

## 快速开始（一条命令）

```bash
cd AscendDevTool
source ascenddevtool/bin/activate

# 单次执行
bash scripts/pipeline_run.sh 1 <模型名>     # 如: bash scripts/pipeline_run.sh 1 sam2

# 或后台循环（自动检测 git 变更，最多10轮迭代调试）
nohup bash scripts/pipeline_loop.sh <模型名> > ../pipeline_loop.log 2>&1 &
```

自动完成 **扫描 → 算子替换 → CUDA迁移 → 推理**，日志在 `logs/run_01/`。

模型源码放在 `~/pipeline_tool/<模型名>/`，输出在 `~/pipeline_tool/ascenddev_output/<模型名>_NPU/`。

---

## 命令行分步操作

适用于需要自定义路径、换模型等场景。以下 `< >` 标记为用户需替换的路径。

### 0. 激活环境

```bash
cd AscendDevTool
source ascenddevtool/bin/activate
```

### 1. 扫描模型

```bash
# 设置 CANN 环境（根据实际路径选一个）
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
# 或: export ASCEND_TOOLKIT_HOME=$HOME/Ascend/ascend-toolkit/latest

export PYTHONPATH="$ASCEND_TOOLKIT_HOME/tools/ms_fmk_transplt:$PYTHONPATH"

python $ASCEND_TOOLKIT_HOME/tools/ms_fmk_transplt/analysis/pytorch_analyse.py \
  -i <待迁移模型目录> \
  -o <扫描输出目录> \
  -v 2.6.0 \
  -m torch_apis
```

扫描完成后，`unsupported_api.csv` 在 `<扫描输出目录>/*_analysis/` 下。

### 2. 算子替换 + CUDA→NPU 迁移

```bash
python migrate.py <待迁移模型目录> <unsupported_api.csv 路径> -o <迁移输出目录>
```

示例：
```bash
python migrate.py ../sam2 ../ascenddev_output/sam2_scan/scan_01_xxx/sam2_analysis/unsupported_api.csv -o ../ascenddev_output/sam2_NPU
```

一步完成算子替换分析、代码改写、CUDA→NPU 迁移（`.cuda()`→`.npu()`、`"gpu"`→`"npu"`、compile_mode 设置等）。

### 3. 推理测试

```bash
cd <迁移输出目录>
# 使用 torch_npu 环境运行推理脚本
/home/orange/miniconda3/envs/torch_npu/bin/python <推理脚本.py>
```

如需指定参数：

```bash
/home/orange/miniconda3/envs/torch_npu/bin/python run_inference_custom.py \
  --output_dir <输出目录> \
  --cad_path <CAD模型路径> \
  --rgb_path <RGB图片路径> \
  --depth_path <深度图路径> \
  --cam_path <相机参数> \
  --seg_path <分割结果>
```

---

## 完整示例（以 SAM-6D 为例）

```bash
# 目录结构
# ~/pipeline_tool/
#   AscendDevTool/      ← 本工具
#   SAM-6D/             ← 待迁移模型
#   ascenddev_output/   ← 输出目录（自动创建）

cd ~/pipeline_tool/AscendDevTool
source ascenddevtool/bin/activate

# 扫描
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export PYTHONPATH="$ASCEND_TOOLKIT_HOME/tools/ms_fmk_transplt:$PYTHONPATH"
python $ASCEND_TOOLKIT_HOME/tools/ms_fmk_transplt/analysis/pytorch_analyse.py \
  -i ../SAM-6D -o ../ascenddev_output/scan -v 2.6.0 -m torch_apis

# 找到 csv
CSV=$(find ../ascenddev_output/scan -name unsupported_api.csv | head -1)

# 替换 + 迁移（一行搞定）
python migrate.py ../SAM-6D "$CSV" -o ../ascenddev_output/SAM-6D_NPU

# 推理
cd ../ascenddev_output/SAM-6D_NPU/SAM-6D/Pose_Estimation_Model

/home/orange/miniconda3/envs/torch_npu/bin/python run_inference_custom.py \
  --output_dir ../Data/Example/outputs \
  --cad_path ../Data/Example/obj_000005.ply \
  --rgb_path ../Data/Example/rgb.png \
  --depth_path ../Data/Example/depth.png \
  --cam_path ../Data/Example/camera.json \
  --seg_path ../Data/Example/outputs/sam6d_results/detection_ism.json
```

---

## 算子开发（进阶）

对于扫描后标记为"需开发"的算子，可生成 Ascend C 工程骨架：

```bash
python -c "
from pathlib import Path
from rewriter.rewriter_core import run_operator_development
result = run_operator_development(
    Path('<unsupported_api.csv 路径>'),
    Path('patcher/local_op_lib/local_ops.csv'),
    Path('op_builder/ops_src'),
)
print(result.status)
"
```

工程生成在 `op_builder/ops_src/<算子名>Sample/` 下，编译：

```bash
cd op_builder/ops_src/<算子名>Sample/FrameworkLaunch/<op_name>
bash build.sh
```

---

## 环境配置

详见 [env.md](env.md)，关键点：

| 项目 | 说明 |
|------|------|
| 工具 venv | `AscendDevTool/ascenddevtool/`（PySide6、libcst） |
| 推理环境 | conda `torch_npu`（`/home/orange/miniconda3/envs/torch_npu/`） |
| CANN 路径 | `/usr/local/Ascend/ascend-toolkit/latest` 或 `$HOME/Ascend/ascend-toolkit/latest` |
| 环境变量 | `ASCEND_TOOLKIT_HOME`、`PYTHONPATH` 需包含 `ms_fmk_transplt` |

---

## 目录结构

```
AscendDevTool/
├── gui/                 # PySide6+QML 界面（可选，命令行版不需要）
├── scanner/             # CANN 扫描封装
├── migrator/            # CUDA→NPU 代码转换（libcst AST）
├── rewriter/            # 算子分析/替换/拆分/开发
│   ├── op_database.py   # 算子知识库（拆解规则）
│   ├── op_analyzer.py   # 分析引擎（可映射→可拆分→需开发）
│   ├── code_patcher.py  # 源码级替换
│   └── rewriter_core.py # 主协调器
├── op_builder/          # Ascend C 算子开发车间
│   └── ops_src/         # 算子源码工程
├── patcher/             # Monkey Patch 运行时替换
│   └── local_op_lib/    # 算子映射表
├── scripts/             # CI/CD 流水线脚本
│   ├── pipeline_run.sh  # 单次流水线（扫描→迁移→替换→推理）
│   └── pipeline_init.sh # 远程首次环境配置
├── logs/                # 流水线日志
├── env.md               # 环境配置指南
└── context.md           # 项目上下文（开发用）
```
