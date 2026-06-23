# Ascend C 算子开发 Skill 文档

> 供大语言模型使用的昇腾算子开发指南

---

## 概述

本 Skill 是LLM用于帮助用户开发昇腾（Ascend NPU）算子的核心文档。

**主入口**: [SKILL.md](SKILL.md) - 包含完整 Role 定义、核心知识、开发流程、红线约束

---

## 快速开始

当用户请求开发Ascend C算子时，直接参考 [SKILL.md](SKILL.md)：

---

## 快速开始

### 核心流程

```cpp
// 1. Tiling函数
void GetOpTiling(OpTiling& tiling, uint32_t param) { ... }
REGISTER_TILING_DEFAULT(GetOpTiling);

// 2. Kernel函数
__CCEATTR__ void KernelOp(KernelContext* ctx, void* param) {
    GET_TILING_DATA(tiling);
    TPipe pipe; pipe.Init();
    TQue<4, TPosition::POS_LOCAL> que; que.InitBuffer();
    // 计算逻辑
    pipe.Destroy();
}
```

---

## 常用接口

| 场景 | 接口 | 位置 |
|------|------|------|
| GM→UB | `DataCopy(local, global, size)` | [04](04_datacopy.md) |
| UB→GM | `DataCopy(global, local, size)` | [04](04_datacopy.md) |
| 计算 | `Add/Mul/Exp/Relu(...)` | [02](02_scalar_vector.md) |
| 分配 | `que.AllocTensor<DT>()` | [03](03_memory_sync.md) |
| 类型转换 | `Cast<DT>(dst, src, repeat)` | [02](02_scalar_vector.md) |
| Matmul | `Matmul(dst, A, B, config)` | [11](11_highlevel_matmul.md) |
| SoftMax | `SoftMax(dst, src, tmp, len)` | [12](12_highlevel_activation.md) |
| LayerNorm | `LayerNorm(...)` | [13](13_highlevel_norm.md) |

---

## 模板代码

### 模板1: 简单矢量算子

```cpp
__CCEATTR__ void KernelSimple(KernelContext* ctx, void* param) {
    GET_TILING_DATA(tiling);
    TPipe pipe; pipe.Init();
    TQue<4, TPosition::POS_LOCAL> que; que.InitBuffer();
    auto in = que.AllocTensor<float>();
    auto out = que.AllocTensor<float>();
    DataCopy(in, ctx->input, size);
    Exp(out, in, tiling.repeatTimes);  // 或其他计算
    DataCopy(ctx->output, out, size);
    pipe.Destroy();
}
```

### 模板2: 需要临时缓冲

```cpp
__CCEATTR__ void KernelWithTmp(KernelContext* ctx, void* param) {
    GET_TILING_DATA(tiling);
    TPipe pipe; pipe.Init();
    TQue<4, TPosition::POS_LOCAL> que; que.InitBuffer();
    auto in = que.AllocTensor<float>();
    auto out = que.AllocTensor<float>();
    auto tmp = que.AllocTensor<float>();  // 临时缓冲
    DataCopy(in, ctx->input, size);
    Gelu(out, in, tmp, tiling.length);  // 使用tmp
    DataCopy(ctx->output, out, size);
    pipe.Destroy();
}
```

### 模板3: 多核算子

```cpp
__CCEATTR__ void KernelMultiCore(KernelContext* ctx, void* param) {
    GET_TILING_DATA(tiling);
    TPipe pipe; pipe.Init();
    TQue<4, TPosition::POS_LOCAL> que; que.InitBuffer();
    uint32_t bidx = GetBlockIdx();
    uint32_t bnum = GetBlockNum();
    uint32_t per = total / bnum;
    uint32_t off = bidx * per;
    // 计算...
    SyncAll();
    pipe.Destroy();
}
```

---

## 约束规则

1. **必须**通过TQue分配张量：`que.AllocTensor<DT>()`
2. **必须**释放张量：`que.FreeTensor(tensor)`
3. **必须**数据传输用DataCopy
4. **必须**结束调用：`pipe.Destroy()`
5. 高阶API需要临时缓冲区要正确计算大小
6. repeatTimes = elements / 32 (float32) 或 elements / 64 (float16)

---

## 详细文档

| 文档 | 内容 |
|------|------|
| [01](01_datatypes.md) | LocalTensor, GlobalTensor, Layout |
| [02](02_scalar_vector.md) | 矢量计算API完整列表 |
| [03](03_memory_sync.md) | TPipe, TQue, SyncAll, IBSet/IBWait |
| [04](04_datacopy.md) | DataCopy变体 |
| [05](05_cache_system.md) | 系统变量 |
| [06_host_api.md](06_host_api.md) | Host侧API (原型注册、Tiling) |
| [06_atomic_matrix.md](06_atomic_matrix.md) | Mmad, Conv2D, Gemm |
| [07](07_kernel_tiling.md) | Tiling宏定义 |
| [08](08_quick_ref.md) | 常用代码片段 |
| [10](10_highlevel_math.md) | Tanh, Sin, Log... |
| [11](11_highlevel_matmul.md) | Matmul |
| [12](12_highlevel_activation.md) | SoftMax, Gelu, SwiGLU |
| [13](13_highlevel_norm.md) | LayerNorm, RmsNorm, GroupNorm |
| [14](14_highlevel_quant.md) | Quant, Dequant |
| [17](17_highlevel_hccl.md) | AllReduce, AllGather |

---

## 输出格式

帮助用户时，输出：

```markdown
## 算子: OpName

### 代码
```cpp
// 代码
```

### 说明
- Tiling策略
- 计算逻辑
- 内存使用
```
