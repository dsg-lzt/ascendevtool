# Pytorch VLA模型310P端侧部署迁移方案

## 硬件设备

极摩客K11电脑、香橙派ai studio pro（Ascend 310P)。

## 功能需求

该方案计划设计一套 Pytorch 模型全流程迁移至昇腾设备部署的工具链。

1. 扫描模型内不支持的Pytorch算子，导出算子名到Excel表内用于后续补齐。
2. 维护一个本地算子库，该算子库用于补齐不支持的算子到模型中。需要实习自动化补齐的功能，通过脚本文件匹配不支持算子名与库内算子名，并生成胶水代码替换模型内的算子函数。
3. 本地算子库没有可替换算子的时候，需要进行算子开发替代，提供Ascend C算子开发模板，用于快速算子实现，并可以入库用于匹配和补全。

## 迁移部署方案规划

本方案采取分阶段迁移策略。第一阶段聚焦于基于 PyTorch 框架的原生端侧部署，主要分为以下核心模块：

### 一、 算子扫描与导出模块 (Scanner)
* **技术路线**：使用 CANN 内置的昇腾官方模型迁移分析工具（`ms_fmk_transplt`），直接对 PyTorch 模型代码进行深度扫描，本项目无需自行开发扫描脚本，仅提供官方工具的操作指南及环境挂载。
* **功能实现**：
  * **API扫描与报表生成**：用户依据指引，直接运行昇腾官方 `pytorch_analyse.sh` 脚本分析待迁移模型所在的文件夹，官方工具会自动识别不支持的 `torch` API 并输出详尽的分析报告（含 Excel 表格清单），作为后续猴子补丁（Patcher）开发的依据。
  * **使用示例**：
    ```bash
    # 1. 切换至分析工具所在路径 (以 CANN 默认安装路径为例)
    cd /usr/local/Ascend/cann/tools/ms_fmk_transplt/
    # 2. 启动分析任务
    ./pytorch_analyse.sh -i /your/project/target_models -o /your/project/scanner/reports -v 2.1.0 -m torch_apis
    ```

### 二、 本地算子库与自动化补齐模块 (Patcher)
* **技术路线**：基于 Monkey Patch（猴子补丁）机制，在 Python 层进行运行时的动态 API 替换。
* **功能实现**：
  * **本地算子库维护**：构建并维护一套基于 Ascend C 开发的本地计算算子库。底层的 Ascend C 算子经过编译后，通过 PyTorch 自定义算子框架绑定封装为 Python 层可调用的函数实现。
  * **代码挂载与替换**：采用 Monkey Patch 方案，实现“非侵入式”替换。只需在模型主程序入口处优先导入补丁脚本，自动劫持 `torch` 命名空间下不支持的算子入口，将其动态重定向至本地算子库中对应的 Ascend C 算子实现，无需修改模型原始代码。

### 三、 Ascend C 自定义算子开发模块 (OpBuilder)
* **技术路线**：作为本地算子库的核心算力来源，直接基于 Ascend C 编写原生底层算子，追求极致端侧硬件加速。
* **功能实现**：
  * **标准化模板引擎**：提供包含 Device 侧 Kernel 实现和 Host 侧调用注册的 Ascend C 算子快速开发模板。
  * **编译与绑定入库**：开发者基于模板填充 C++ 层逻辑并完成编译后，由工具链自动通过 PyTorch C++ API 完成注册，生成可被 Python 直接导入的算子包，持续扩充上述阶段的“本地算子库”。

### 四、 可视化交互界面模块 (GUI) (后期规划)
* **技术路线**：采用 Python Qt (PyQt/PySide) 结合 QML 技术栈构建现代化、流畅的图形交互界面。
* **功能实现**：
  * **一站式工具链整合**：通过 UI 界面将“算子扫描导出”、“动态 Patch 注入补齐”、“Ascend C 开发模板生成”等底层命令操作进行图形化封装。
  * **开发策略说明**：作为工程的最终包装呈现，该部分被安排在项目最后一期进行。前期专注将 scanner、patcher、op_builder 核心逻辑跑通，UI 仅做预留不开发。

### 五、 第一阶段总结工作流
1. **官方工具扫描**：使用昇腾官方工具扫描 PyTorch VLA 模型代码，并导出缺失、不支持算子及其 API 的 Excel 清单。
2. **底层算子开发**：针对清单中缺少的算子，利用 OpBuilder 快速生成 Ascend C 工程进行开发不仅补齐算力，验证通过后自动编译归入本地算子库。
3. **Monkey Patch 动态补齐**：一键运行 Patcher，利用 Monkey Patch 机制，在原模型代码运行前拦截掉所有不受支持的 PyTorch 算子，动态映射给本地算子库中的自有实现。
4. **PyTorch 端侧直通部署**：模型算子全部适配后，直接在硬件设备（香橙派等）上基于 PyTorch 环境（配合 `torch_npu` 插件）进行 Native 模型推理与验证。

## 目录结构设计

为了实现核心模块的高内聚低耦合，我们对算子源码开发（op_builder）与编译产物（oplib）进行了严格的职责分离。项目目录结构设计如下：

```text
.
├── target_models/          # 待迁移部署的原始 PyTorch VLA 模型存放目录
├── scanner/                # 算子扫描与导出模块 (直接依赖昇腾官方迁移工具)
│   ├── README.md           # 官方扫描工具使用指南与环境配置说明
│   └── reports/            # 存放扫描输出的不支持算子清单 (Excel表格)
├── patcher/                # 猴子补丁与动态映射调用模块
│   ├── patch_core.py       # Monkey Patch 核心脚本，在模型运行前导入以动态劫持 torch API
│   ├── ops_config.yaml     # 配置表：维护不支持的 torch API 到 oplib 动态库的映射关系
│   └── custom_ops_loader.py# 负责利用 torch.ops.load_library() 动态加载 .so 库
├── op_builder/             # 算子构建与开发车间 (仅包含源码与生成/编译脚本，保持纯净)
│   ├── create_op.py        # 脚本工具：根据模板一键生成算子源码工程目录
│   ├── compile_op.py       # 脚本工具：一键编译指定的算子，并自动将产物分发至 oplib 对应目录
│   ├── templates/          # Ascend C / PyTorch C++ 绑定代码及 CMake 构建的模板库
│   └── ops_src/            # 所有本地自定义算子的源码集合 (每个算子为一个独立的 CMake 工程)
├── oplib/                  # 本地算子库与编译产物 (仅存放动态库与安装包)
│   ├── custom_opp_packages/# 存放编译生成的 Ascend C .run 安装包，用于底层算子信息注册
│   ├── python_bindings/    # 存放通过 PyTorch C++ API 编译生成的 .so 动态拓展库，供 Python 加载
│   └── build_cache/        # (可选) 存放 CMake 中间构建缓存，统一隔离避免污染 ops_src 源码区
├── gui/                    # 可视化交互界面模块 (后期基于 Python Qt + QML 开发)
│   ├── qml/                # QML 界面布局与交互设计文件
│   └── main.py             # UI 应用程序启动入口 (前期暂留空)
├── tests/                  # 本地算子精度对齐验证及端侧推理测试用例
├── scripts/                # 全局自动化流程脚本 (如一键环境配置、数据预处理等)
└── README.md               # 本项目总体方案说明与维护指南
```
