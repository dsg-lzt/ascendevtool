# Pytorch VLA模型310P端侧部署迁移方案

## 硬件设备

极摩客K11电脑、香橙派ai studio pro（Ascend 310P）。

---

## 快速开始

```bash
cd /home/lzt/ascend_project/AscendDevTool
python gui/main.py
```

GUI 界面提供四个功能面板：

| 面板 | 功能 |
|------|------|
| 路径选择 | 选择模型目录、输出目录、PyTorch版本 → 点击"扫描模型" |
| 迁移到 NPU | 将 CUDA 代码转换为 NPU 代码（`.cuda()` → `.npu()` 等） |
| 算子替换分析 | 分析不适配算子 → 拆分/改写/替换 → 生成算子开发工程 |
| 不支持算子统计 | 按出现次数展示不支持算子 |

---

## 完整迁移工作流

```
① 扫描模型 → ② 代码迁移(NPU) → ③ 算子替换分析 → ④ 逐个开发算子 → ⑤ 编译入库
```

### ① 扫描模型
使用 CANN 内置 `ms_fmk_transplt` 扫描 PyTorch 代码，输出 `unsupported_api.csv`。

### ② 代码迁移（NPU）
将原始代码完整复制到输出目录，并自动替换：
- `import torch` 后追加 `import torch_npu`
- `.cuda()` → `.npu()`
- `.to("cuda")` → `.to("npu")`
- `torch.cuda.*` → `torch.npu.*`
- `from torch.cuda.amp import ...` → `from torch.npu.amp import ...`

### ③ 算子替换分析
对 `unsupported_api.csv` 中的每个不适配算子，按优先级尝试：
1. **可映射** — 已有 Ascend C 实现，写入 `local_ops.csv` 映射表
2. **可拆分** — 拆解为原生 PyTorch 算子组合，生成 `decomposed_ops.py`
3. **数学等价改写** — 替换为数学等价表达式
4. **需开发** — 生成 Ascend C 工程骨架 + Agent 开发提示

替换模块 `ascend_pointnet2_ops.py` 随输出目录一起生成。

### ④ 逐个开发算子
在"算子替换分析"面板中，每个"需开发"算子都有独立的"开发"按钮，点击后：
- 调用 `msopgen` 生成完整 Ascend C 工程（`op_builder/ops_src/` 下）
- 生成 `AGENT_PROMPT_*.md`，内含 kernel stub、原始 CUDA 源码、cannbot-skills 参考文档

### ⑤ 编译入库
```bash
cd op_builder/ops_src/<OpName>Sample/FrameworkLaunch/<op_name>
bash build.sh              # 编译 Ascend C
# 然后写回 patcher/local_op_lib/local_ops.csv
```

---

## 目录结构

```text
.
├── target_models/              # 待迁移的原始 PyTorch 模型
├── scanner/                    # 算子扫描模块（依赖 CANN ms_fmk_transplt）
│   ├── README.md
│   └── reports/                # 扫描输出（unsupported_api.csv 等）
├── patcher/                    # Monkey Patch 运行时替换模块
│   ├── patch_core.py           # 运行时动态劫持 torch API
│   ├── ops_config.yaml         # API → 本地库映射配置
│   └── local_op_lib/           # 本地算子库
│       ├── local_ops.csv       # 算子 → 本地实现映射表
│       ├── custom_ops.py       # Python 层替身实现
│       └── loader.py           # .so 动态库加载器
├── op_builder/                 # Ascend C 算子开发车间
│   ├── op_manager.py           # 算子生命周期管理（generate/build/install）
│   ├── create_op.py            # C++ 绑定脚手架生成
│   ├── ops_src/                # 所有算子源码工程
│   ├── skill_docs/             # Ascend C 开发参考文档（LLM 用）
│   └── templates/              # 代码模板
├── oplib/                      # 编译产物（.so / .run 安装包）
│   └── custom_opp_packages/
├── gui/                        # PySide6 + QML 可视化界面
├── migrator/                   # CUDA → NPU 代码转换模块
│   ├── torch_to_npu.py         # libcst AST 转换器
│   └── migrator_core.py        # 批量迁移入口
├── rewriter/                   # 算子替换/拆分/重写/开发模块
│   ├── op_database.py          # 算子知识库（CANN 支持列表 + 拆解规则）
│   ├── op_analyzer.py          # 算子分析引擎
│   ├── op_decomposer.py        # 算子拆分引擎
│   ├── op_rewriter.py          # 数学等价改写
│   ├── op_dev_generator.py     # Ascend C 脚手架生成
│   ├── op_dev_agent.py         # Agent 开发任务生成（msopgen + cannbot-skills）
│   ├── code_patcher.py         # 源码级算子调用替换
│   └── rewriter_core.py        # 主协调器
├── tests/                      # 测试用例
├── scripts/                    # 全局自动化脚本
├── env.md                      # 环境配置指南
└── README.md
```

---

## 环境配置

详见 [env.md](env.md)，主要内容：
- Conda 环境创建
- 虚拟环境 + `requirements.txt` 依赖安装
- CANN / Ascend Toolkit 环境变量配置
- `pandas` / `prettytable` 等 CANN 扫描工具依赖

