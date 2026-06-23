# 高可靠异构算子库 (op_builder) 架构方案

为了满足具身智能模型迁移中大量自定义算子（Custom Ops）的动态补齐与适配需求，本系统采用 **“分散开发，集中编译，统一加载”** 的系统性设计，解决单独编译算子导致的文件碎片化、命名冲突及 PyTorch Dispatch 残留问题。

## 1. 系统核心设计目标

1.  **消除“动库灾难”（.so Hell）**：避免每个算子独立编译成 `.so`，减少 Python 启动时的加载开销。
2.  **避免符号冲突与冗余分发**：禁止不同开发者使用相同的命名空间或变量名。通过隔离命名空间和底层宏，收敛到统一的 PyTorch 设备注册入口。
3.  **Tiling 隔离保障**：严控 Ascend C Host/Device 侧由于同名数据结构引起的内存越界与错构。

## 2. 架构目录规范

```text
op_builder/
├── CMakeLists.txt              # 全局单一编译入口，扫描所有 ops 并生成统一的 libembodied_custom_ops.so
├── src/
│   ├── common/                 # 公共依赖：包含统一的错误捕获、宏定义和辅助组件 (如 NPU Stream 封装)
│   ├── ops/                    # 业务实现：按算子独立子目录，代码由脚手架统一生成
│   │   ├── custom_relu/
│   │   │   ├── kernel_impl.cpp # Device 端计算逻辑 (核函数)
│   │   │   ├── host_tiling.cpp # Host 端数据切分逻辑 (CPU)
│   │   │   └── adapter.cpp     # C++ 拓展与 C10/ATen 绑定
│   └── op_registry.cpp         # Toplevel 全局注册中心
├── templates/                  # 代码模板存放区
└── create_op.py                # 算子创建脚手架 CLI
```

## 3. C++ 侧分布式隔离与映射机制

利用 PyTorch 的 `TORCH_LIBRARY` 和 `TORCH_LIBRARY_IMPL` 将定义与实现解耦。

*   **算子定义（Schema）**：在统一文件 `op_registry.cpp` 中执行 `TORCH_LIBRARY(embodied_ops, m)` 进行所有的 `m.def` 声明。
*   **算子实现（Impl）**：在各自独立的 `adapter.cpp` 中通过强制生成的 `namespace embodied_ops::[op_name]` 包裹逻辑，并在模块内利用 `TORCH_LIBRARY_IMPL(embodied_ops, PrivateUse1, m)` 注册至具体的硬件 Dispatch（如 NPU）。

## 4. Python 侧动态集成中枢 (`loader.py`)

Python 侧在拦截阶段启动懒加载（Lazy Load），暴露单例代理工厂，如 `patcher.local_op_lib.loader.get_op('op_name')`。该函数将确保底层的 `.so` 只在首次请求时被装载一次，且抛出友好的缺失提示。

## 5. 防御性脚手架 (`create_op.py`) 职责

- 依据算子名自动生成完全隔离的 C++ Namespace 与 Tiling 结构体（例如 `SoftmaxTilingData` 而非通用名 `TilingData`）。
- 生成统一的异常拦截和连续性（`.contiguous()`）张量检查包装器，极大地减少手动编写 Adapter 代码的几率，保证研发安全。