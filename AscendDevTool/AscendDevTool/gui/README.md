# GUI 设计需求

## 1. 功能需求

- **选择待迁移模型目录**：用户可以通过文件浏览器选择包含待迁移模型的目录。
- **选择迁移目标目录**：用户可以通过文件浏览器选择迁移后的模型存储目录。
- **选择 PyTorch 版本**：支持 `默认`、`1.11.0`、`2.1.0`、`2.2.0`、`2.3.1`、`2.4.0`、`2.5.1`、`2.6.0`。
  - 选择 `默认` 时：优先使用当前 Python 环境已安装的 `torch.__version__`；若未安装 torch，则回退到 `2.6.0`。
- **扫描模型**：用户点击“扫描模型”按钮后，系统会扫描待迁移模型目录中的所有模型文件，并统计不支持的算子，显示在界面上。
- **加载已扫描结果**：用户可手动选择历史扫描结果路径（目录），点击“加载已扫描结果”后直接读取并展示该路径下最新的 `unsupported_api.csv`。

## 2. 与总体方案的衔接

根据项目根目录 README 的规划，算子扫描阶段优先使用 CANN 自带的迁移分析工具 `ms_fmk_transplt/pytorch_analyse.sh` 生成报告。

当前 GUI 的“扫描模型”按钮采用如下策略：

1. **若系统已安装 CANN 且存在 `pytorch_analyse.sh`**：GUI 会尝试调用该脚本，对你选择的“待迁移模型目录”执行扫描，并将输出写到本项目的 `scanner/reports/`。扫描时会携带界面中选择的 PyTorch 版本参数。
2. **若未发现官方脚本或执行失败**：GUI 会退化为“汇总模式”，直接在 `scanner/reports/**/unsupported_api.csv` 中选择最新的一份报告做统计展示。

> 说明：为避免官方工具在“输出目录已存在”时触发交互式覆盖询问（GUI 子进程 stdin 关闭会导致 `EOF when reading a line`），GUI 会把每次扫描输出写到 `scanner/reports/run_<扫描目录名>_torch<版本号>_<时间戳>/` 这样的子目录中，并保留历史结果。

> 说明：GUI 展示的数据来源于扫描工具输出的 `unsupported_api.csv`（包含 `OP` 列），本期主要完成“选择路径 + 扫描/汇总 + 统计展示”的闭环。

## 3. 展示内容

- 不支持算子（OP）按出现次数降序列表
- 不支持算子种类数（Unique OP 数）
- 不支持算子出现总次数（CSV 行数汇总）
- 支持手动加载历史扫描路径并展示结果

## 4. 运行方式

### 4.1 安装依赖

在你的 Python 环境中安装 Qt 绑定：

```bash
pip install -r gui/requirements.txt
```

> 说明：GUI 会优先用当前 Python 解释器执行 CANN 官方扫描入口（避免系统 python 缺依赖）。
> 官方扫描工具依赖 `libcst` 做静态分析，因此 `gui/requirements.txt` 里已包含 `libcst`。

### 4.2 启动 GUI

在项目根目录执行：

```bash
python gui/main.py
```

### 4.3 先用官方工具生成报告（可选，但推荐）

如果你的环境已安装 CANN，可先按 [scanner/README.md](../scanner/README.md) 的指引手动执行：

```bash
cd /usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/
./pytorch_analyse.sh -i <YOUR_WORKSPACE_PATH>/target_models \
  -o <YOUR_WORKSPACE_PATH>/scanner/reports \
  -v 2.1.0 \
  -m torch_apis
```

然后再打开 GUI 点击“扫描模型”（会自动读取最新报告并展示统计）。
  