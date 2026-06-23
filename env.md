# 环境配置指南

## 1. Conda 环境与虚拟环境

### 1.1 创建 Conda 环境

```bash
conda create -n ascend-dev python=3.13 -y
conda activate ascend-dev
```

### 1.2 在项目内创建 Python 虚拟环境

```bash
cd /home/lzt/ascend_project/AscendDevTool
python -m venv ascenddevtool
source ascenddevtool/bin/activate
```

### 1.3 安装项目 GUI 依赖

```bash
pip install -r gui/requirements.txt
```

> `gui/requirements.txt` 内容：
> - `PySide6>=6.6` — Qt for Python 图形界面
> - `libcst>=1.1.0` — CANN 官方扫描工具做静态分析所需

### 1.4 安装 CANN 扫描工具所需的 Python 依赖

CANN 的 `ms_fmk_transplt` 扫描工具内部依赖了一些额外的 Python 包，需要手动安装到虚拟环境中：

```bash
pip install pandas prettytable
```

> 如果网络受限，可使用清华镜像：
> ```bash
> pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas prettytable
> ```

---

## 2. CANN / Ascend Toolkit 环境变量

CANN（Ascend Computing and Network）安装路径为 `$HOME/Ascend/ascend-toolkit/latest`。
将以下环境变量写入 `~/.bashrc` 或每次使用前执行：

```bash
export ASCEND_TOOLKIT_HOME=$HOME/Ascend/ascend-toolkit/latest
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
export ASCEND_OPP_PATH=$HOME/Ascend/ascend-toolkit/latest/opp
export TOOLCHAIN_HOME=$HOME/Ascend/ascend-toolkit/latest/toolkit

export PATH=$ASCEND_TOOLKIT_HOME/bin:$ASCEND_TOOLKIT_HOME/compiler/ccec_compiler/bin:$ASCEND_TOOLKIT_HOME/tools/ccec_compiler/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_TOOLKIT_HOME/x86_64-linux/devlib:$ASCEND_TOOLKIT_HOME/runtime/lib64/stub:$ASCEND_TOOLKIT_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$ASCEND_TOOLKIT_HOME/python/site-packages:$ASCEND_TOOLKIT_HOME/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
```

> **说明**：项目扫描器 (`gui/scanner_core.py`) 会优先读取 `ASCEND_TOOLKIT_HOME` 或 `ASCEND_HOME_PATH` 来定位 `pytorch_analyse.sh`。若未设置，则依次回退到 `$HOME/Ascend/ascend-toolkit/latest/` 和 `/usr/local/Ascend/`。

---

## 3. 启动 GUI

在项目根目录执行：

```bash
cd /home/lzt/ascend_project/AscendDevTool
python gui/main.py
```

GUI 功能：
- **选择待迁移模型目录** — 通过文件浏览器选择
- **选择 PyTorch 版本** — 支持 `默认`、`1.11.0`、`2.1.0`、`2.2.0`、`2.3.1`、`2.4.0`、`2.5.1`、`2.6.0`
- **扫描模型** — 调用 CANN `pytorch_analyse.sh` 对模型代码做 API 兼容性分析
- **加载已扫描结果** — 手动选择历史 `unsupported_api.csv` 路径直接展示

> 扫描策略：先尝试执行 CANN 官方扫描；若官方脚本不存在或执行失败，则退化为读取 `scanner/reports/` 下已有的最新报告。

---

## 4. 命令行扫描

如果不使用 GUI，也可以直接在命令行中运行 CANN 扫描：

```bash
export PYTHONPATH="$ASCEND_TOOLKIT_HOME/tools/ms_fmk_transplt:$PYTHONPATH"
python $ASCEND_TOOLKIT_HOME/tools/ms_fmk_transplt/analysis/pytorch_analyse.py \
  -i <待扫描模型目录> \
  -o <输出目录> \
  -v 2.6.0 \
  -m torch_apis
```

参数说明：
| 参数 | 含义 |
|------|------|
| `-i` | 待分析模型代码的目录 |
| `-o` | 扫描报告输出目录 |
| `-v` | PyTorch 框架版本（如 2.1.0, 2.6.0） |
| `-m` | 分析模式，`torch_apis` 表示仅分析 PyTorch API 兼容性 |

扫描超时可通过环境变量调整（默认 1800 秒）：
```bash
export ASCEND_SCAN_TIMEOUT_SEC=3600
```

---

## 5. 输出结果文件说明

扫描完成后，`-o` 指定的输出目录下会生成 `*_analysis/` 子目录，包含：

| 文件 | 说明 |
|------|------|
| `unsupported_api.csv` | 不受支持的 PyTorch API 列表（算子补全目标） |
| `cuda_op_list.csv` | CUDA 算子列表 |
| `unknown_api.csv` | 未能识别的 API 列表 |
| `api_performance_advice.csv` | 性能建议 |
| `api_precision_advice.csv` | 精度建议 |

---

## 6. 常见问题

### Q: 报错 "未发现 CANN 扫描脚本 pytorch_analyse.sh"

确认 `ASCEND_TOOLKIT_HOME` 已设置，且以下文件存在：
```
$ASCEND_TOOLKIT_HOME/tools/ms_fmk_transplt/pytorch_analyse.sh
```

### Q: ModuleNotFoundError: pandas / prettytable

按 1.4 节安装缺失的 Python 包。

### Q: 官方扫描执行失败（退出码 1）

查看日志文件 `scanner/reports/gui_scan_*.log` 的尾部排查具体原因。

### Q: 扫描超时

增大超时值：`export ASCEND_SCAN_TIMEOUT_SEC=3600`，或设为 0 表示无限等待。
