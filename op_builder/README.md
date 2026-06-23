# Ascend C 自定义算子开发与集成规范

## 1. 概述

Ascend C 是昇腾 NPU 面向算子开发的高级编程语言。本模块 (`op_builder`) 旨在规范因算子缺失或性能瓶颈而引发的底层自定义算子开发流程。
通过标准化的工程目录与代码范式，保障异构计算资源的充分利用，并实现从 Device 侧内核开发到 PyTorch 前端映射的全链路系统化构建。

---

## 2. 算子开发生命周期管理工具 (op_manager.py)

为了彻底打通 **"骨架生成 -> 交叉编译 -> ACLNN精度验证 -> 归入本地算子库"** 的闭环工作流，我们提供了统一的自动化管理工程脚本 `op_manager.py`。该脚本依托底层 `msopgen` 工具，为您屏蔽了复杂的 CMake 编译和 CANN 环境挂载细节。

### 2.1 适用场景
* 快速创建一个全新的 Ascend C 算子工程模板。
* 交叉编译写好的算子，并在 NPU 上开展底层的数值精度与性能验证。
* 一键将验证通过的算子打包为 `.run` 镜像，并归档至 `oplib/custom_opp_packages/` 供上层 PyTorch 框架中 `patcher` 动态加载。

### 2.2 使用说明与命令参考

首先进入 `op_builder` 目录，确保有执行权限：
```bash
cd op_builder
chmod +x op_manager.py
```

**命令帮助：**

```bash
./op_manager.py --help
```

**核心子命令清单：**

1. **生成多算子工程骨架 (`generate`)**
   准备一份包含单个或多个算子定义的 json 协议文件（格式为 JSON 数组），由工具自动解析并生成统一的 Ascend C C++ 桩代码工程：

   ```bash
   ./op_manager.py generate EmbodiedAiOps --json path/to/multi_ops.json
   ```

   > **高阶特性提示**：我们已全面支持**“单包多算子”**架构！工具会自动解析 JSON 数组中的全部算子，通过迭代追加模式合并生成统一工程框架，同时将底层易冲突的默认 `vendor_name`（`customize`）智能替换为您指定的工程小写名称（如 `embodiedaiops`）。这彻底解决了多次归库时引发的相互覆盖与交叉干扰问题。
   > 生成的内容位于 `ops_src/EmbodiedAiOpsSample/FrameworkLaunch/`，随后只需前往修改对应算子的 `op_kernel` 与 `op_host` 即可。

2. **交叉编译算子 (`build`)**
   将算子代码编译为能被 CANN 系统加载的 `.run` 安装包：

   ```bash
   ./op_manager.py build BallQuery
   ```

3. **精度与运行验证 (`verify`)**
   （可选）在脱离 PyTorch 的纯 C++ 基础环境下，调用 AclNN 接口验证算子执行结果是否正确：

   ```bash
   ./op_manager.py verify BallQuery
   ```

4. **部署与归库 (`install`)**
   将生成的算子 `.run` 镜像静默安装至本机的 `$ASCEND_OPP_PATH`，同时备份到工程的本地算子库 (`oplib/custom_opp_packages/`) 供后续分发使用：

   ```bash
   ./op_manager.py install BallQuery
   ```

5. **全自动流水线 (`pipeline`)**
   一键串联 `build` -> `verify` -> `install`，这是日常调试中最常用的指令：

   ```bash
   ./op_manager.py pipeline BallQuery
   ```

### 2.3 PyTorch 适配器与集成工具 (`create_op.py`)

在底层 Ascend C 算子打包完成后，需要使用 `create_op.py` 辅助生成 PyTorch 的 C++ API 适配层绑定代码（Stub）：

```bash
python3 create_op.py --op_name ball_query --inputs "Tensor xyz, Tensor center_xyz" --outputs "Tensor"
```

这个脚本会生成 PyTorch Wrapper 代码。接下来在 `patcher/local_op_lib/local_ops.csv` 注册该算子的映射策略即可完成全链路的算子平替闭环。

---

## 3. 标准开发工作流与异构并发模型

在 Ascend C 编程模型中，系统被抽象为双侧协同架构：

### 3.1 Host 与 Device 协同机制

* **Host（主机侧，通常即 CPU 端）**：承担控制面职责。主要负责模型张量 Shape 推导、内存分配、计算算子的 Tiling（数据切分策略）生成以及任务异步下发。
* **Device（设备侧，即 NPU AI Core）**：承担数据面职责。接收 Host 侧调度的任务，执行核心张量计算（如 Vector 矢量计算与 Cube 矩阵计算）。

### 3.2 内存层级与多阶流水线并行

AI Core 计算单元无法直接高效访问全局大内存（Global Memory）。必须通过异步机制将数据搬移至高速局部内存（Local Memory）。
为掩盖数据 I/O 延迟，Ascend C 引入了 `TPipe` 及双缓冲（Double Buffer）机制，形成标准的三阶段流水线编程范式：

1. `CopyIn`：将数据从 Global Memory 异步搬入 Local Memory。
2. `Compute`：调用 AI Core 的多核计算指令处理 Local Memory 的数据。
3. `CopyOut`：将计算结果从 Local Memory 异步搬出至 Global Memory。

---

## 4. CUDA 算子迁移至 Ascend C 实践指南

在遇到不适配的 CUDA 算子时，将其重写为 Ascend C 算子需要编程思维与执行模型的转变。CUDA 基于 SIMT（单指令多线程）模型，而 Ascend C 基于 SPMD（单程序多数据）的高流水线模型。以下是迁移的核心步骤与注意事项：

### 4.1 迁移实施步骤

1. **计算逻辑抽离与语义映射**
   - **分析 CUDA Kernel**：理清 CUDA 算子的网格（Grid）、线程块（Block）和单个线程（Thread）内执行的细粒度逻辑。
   - **核级映射（Core 映射）**：CUDA 的 Block 级并发或者 Grid 并发任务通常被映射为 Ascend C 的跨核并发（依赖 `block_idx` 分发任务给多个 AI Core）。
2. **数据切分（Tiling）策略重构**
   - CUDA 通过 `threadIdx.x` 定位线程，处理单一元素计算。
   - Ascend C 必须在 CPU Host 侧进行 Tiling 计算，由于片上 Local Memory 有限（常为 256KB），必须将 Global Memory 分割成一块块能放进片上内存的 Tile（如每次 128  个 Byte），再逐块处理。
3. **基于队列的异步流水线**
   - 弃用 CUDA 中的全局内存直接访存操作。
   - 将 CUDA 内共享内存（Shared Memory）的使用转换为 Local Memory 申请（`AllocTensor` / `FreeTensor`）。
   - 利用硬件事件级队列 `TPipe` 和 `TQue`，严格构建 `CopyIn` -> `Compute` -> `CopyOut` 异步三阶流水。
4. **计算逻辑矢量化（Vectorization）**
   - CUDA 线程专注于标量（Scalar）计算。
   - Ascend C 须在 `Compute` 阶段中采用矢量 API（如 `Add`, `Mul`, `Maxs`, `Exp` ），一次性对整个 Tile Tensor 的大块数据进行并行推演计算。

### 4.2 核心注意事项（踩坑指南）

* **数据地址强制对齐（32-Byte Alignment）**：
  昇腾架构对 Global Memory 与 Local Memory 数据搬运极为严格。必须按 **32 Bytes (即 16 个 float16 或 8 个 float32)** 边界对齐。当末尾数据（Tail）不对齐时，一定要单独做条件分支判定与 Padding 操作，否则运行时将抛出内存异常。
* **警惕 OOM（片上内存短缺）**：
  不同型号 NPU 的 Unified Buffer 规格不一。如果一个阶段申请过多个局部张量参与复杂运算，极易引起编译期分配超限错。设计复杂算子时务必精打细算张量复用机制。
* **活用双缓冲隐藏读写延迟（Double Buffer）**：
  CUDA 通过多 Warp 切换掩蔽访存停顿，而 Ascend C 没有这个机制。必须将各类队列的深度设为至少为 `2`。这样当算力核在处理 `缓冲 A` 的计算时，总线资源可独立向 `缓冲 B` 搬入/搬出数据。
* **禁用并行锁与同步变量思维**：
  不同于使用类似 CUDA `__syncthreads()` 的同步机制，Ascend 的计算流由 `EnQue`/`DeQue` 接管。流水线内部的数据依赖是由底层逻辑控制阻塞来实现的非显式时序同步。不要尝试手动管理线程锁。
