# 快速参考卡片

> Ascend C 算子开发常用代码片段

## 目录

- [基本算子结构](#基本算子结构)
- [数据类型转换](#数据类型转换)
- [常用计算](#常用计算)
- [Tiling数据定义](#tiling数据定义)
- [数据搬运模式](#数据搬运模式)
- [队列使用模式](#队列使用模式)
- [同步模式](#同步模式)

---

## 基本算子结构

### 完整算子框架

```cpp
// ==================== Tiling结构体 ====================
struct OpTiling {
    uint32_t blockDim;
    uint32_t repeatTimes;
    uint32_t srcStride;
};

// ==================== Tiling函数 (Host侧) ====================
void GetOpTiling(OpTiling& tiling, uint32_t length) {
    tiling.blockDim = 1;
    tiling.repeatTimes = length / 32;
    tiling.srcStride = tiling.repeatTimes;
}
REGISTER_TILING_DEFAULT(GetOpTiling);

// ==================== Kernel函数 ====================
__CCEATTR__ void KernelOp(KernelContext* ctx, void* param) {
    GET_TILING_DATA(tiling);
    
    TPipe pipe;
    pipe.Init();
    
    TQue<4, TPosition::POS_LOCAL> que;
    que.InitBuffer();
    
    LocalTensor<float> input = que.AllocTensor<float>();
    LocalTensor<float> output = que.AllocTensor<float>();
    
    DataCopy(input, ctx->input, size);
    
    // 计算
    Exp(output, input, tiling.repeatTimes);
    
    DataCopy(ctx->output, output, size);
    
    pipe.Destroy();
}
```

---

## 数据类型转换

### Cast 类型转换

```cpp
// float32 -> float16
Cast<half>(dstHalf, srcFloat, repeat);

// float16 -> float32  
Cast<float>(dstFloat, srcHalf, repeat);

// int32 -> float32
Cast<float>(dstFloat, srcInt32, repeat);

// int8 -> float32 (需要量化参数)
CastDeq(dstFloat, srcInt8, deqScale, repeat);
```

### 数据类型对应

| C++类型 | Ascend C类型 | 说明 |
|---------|--------------|------|
| `float` | `DT_FLOAT` | 32位浮点 |
| `half` | `DT_HALF` | 16位浮点 |
| `int32_t` | `DT_INT32` | 32位整数 |
| `int8_t` | `DT_INT8` | 8位整数 |
| `uint8_t` | `DT_UINT8` | 无符号8位 |

---

## 常用计算

### 矢量计算

```cpp
// 基本运算
Exp(dst, src, repeat);           // 指数
Ln(dst, src, repeat);            // 对数
Abs(dst, src, repeat);           // 绝对值
Sqrt(dst, src, repeat);          // 平方根

// 算术运算
Add(dst, src0, src1, repeat);   // 加法
Sub(dst, src0, src1, repeat);   // 减法
Mul(dst, src0, src1, repeat);   // 乘法
Div(dst, src0, src1, repeat);   // 除法

// 带标量
Adds(dst, src, scalar, repeat);  // 加标量
Muls(dst, src, scalar, repeat);  // 乘标量
Maxs(dst, src, scalar, repeat); // 最大值
Mins(dst, src, scalar, repeat); // 最小值

// 激活函数
Relu(dst, src, repeat);          // ReLU
LeakyRelu(dst, src, alpha, repeat); // Leaky ReLU
```

### 归约计算

```cpp
ReduceSum(dst, src, repeat, stride);  // 求和
ReduceMax(dst, src, repeat, stride);  // 最大值
ReduceMin(dst, src, repeat, stride);  // 最小值
```

### 高阶API

```cpp
// SoftMax
uint32_t tmpSize = GetSoftMaxMaxMinTmpSize();
LocalTensor<float> tmp = que.AllocTensor<float>();
SoftMax(output, input, tmp, length);

// LayerNorm
LayerNorm(output, input, mean, var, gamma, beta, tmp, length, eps);

// Gelu
Gelu(output, input, tmp, length);

// Matmul
using Config = MatmulConfig<half, half, half>;
Config config;
config.Init(m, n, k);
config.SetTensorA(tensorA);
config.SetTensorB(tensorB);
Matmul(tensorC, tensorA, tensorB, config);
```

---

## Tiling数据定义

### 常用Tiling结构

```cpp
// 简单矢量运算
struct SimpleTiling {
    uint32_t repeatTimes;
    uint32_t srcStride;
};

// 多核运算
struct MultiCoreTiling {
    uint32_t blockDim;
    uint32_t blockSize;
    uint32_t totalLength;
};

// Matmul
struct MatmulTiling {
    uint32_t blockDim;
    uint32_t mBlock;
    uint32_t nBlock;
    uint32_t kBlock;
    uint32_t mTail;
    uint32_t nTail;
};
```

### repeatTimes 计算

```cpp
// float32: 32元素/次
uint32_t repeat = length / 32;

// float16: 64元素/次
uint32_t repeat = length / 64;

// int8: 128元素/次
uint32_t repeat = length / 128;
```

---

## 数据搬运模式

### GM -> UB

```cpp
// 基础复制
DataCopy(localDst, globalSrc, size);

// 带偏移
DataCopy(localDst, 0, globalSrc, offset, size);
```

### UB -> GM

```cpp
DataCopy(globalDst, localSrc, size);
```

### 带步长复制

```cpp
// srcStride: 源步长
// repeatTimes: 重复次数
DataCopy(localDst, globalSrc, repeatSize, repeatTimes, srcStride);
```

### 广播

```cpp
Duplicate(dst, src, repeatTimes);  // 复制N次
```

---

## 队列使用模式

### 简单队列

```cpp
TQue<4, TPosition::POS_LOCAL> que;
que.InitBuffer();

LocalTensor<float> tensor = que.AllocTensor<float>();
// 使用tensor
que.FreeTensor(tensor);
```

### Ping-Pong双缓冲

```cpp
TQueBind<2, TPosition::POS_LOCAL> que;
que.Init();

void* buf0 = malloc(1024);
void* buf1 = malloc(1024);
que.InitBufHandle(buf0, 1024, 512);

// Ping位置
LocalTensor<float> input = que.AllocTensor(TPosition::POS_PING);

// 计算
Exp(output, input, repeat);
que.EnQue(pipe, TPosition::POS_PING);

// Pong位置
LocalTensor<float> result = que.DeQue(pipe, TPosition::POS_PONG);
```

---

## 同步模式

### 队列同步

```cpp
que.EnQue(pipe);           // 入队
LocalTensor result = que.DeQue(pipe);  // 出队等待
```

### 核间同步

```cpp
uint32_t blockId = GetBlockIdx();
uint32_t blockNum = GetBlockNum();

if (blockId == 0) {
    IBSet(1);  // 主Block设置标志
} else {
    IBWait(1);  // 等待
}
SyncAll();  // 全部同步
```

### 屏障同步

```cpp
PipeBarrier();      // 管道屏障
DataSyncBarrier();  // 数据同步屏障
```

---

## 常用计算公式

### 数据大小计算

```cpp
// 字节数
uint32_t bytes = elements * sizeof(DT);

// repeatTimes (float32)
uint32_t repeat = (elements + 31) / 32;

// repeatTimes (float16)
uint32_t repeat = (elements + 63) / 64;
```

### Tiling计算

```cpp
// 简单Tiling
tiling.blockDim = (length + 255) / 256;
tiling.repeatTimes = length / 32;

// 多核Tiling
uint32_t perCore = length / blockNum;
uint32_t remain = length % blockNum;
```

### 临时缓冲区大小

```cpp
// 高阶API临时缓冲
uint32_t tmpSize = GetFuncMaxMinTmpSize();

// 或带因子
uint32_t factor = GetFuncTmpBufferFactorSize();
uint32_t totalSize = length * sizeof(DT) * factor;
```

---

## 快速查找表

### 数据位置枚举

| 值 | 说明 |
|----|------|
| `POS_LOCAL` | 本地内存 (UB) |
| `POS_PING` | Ping缓冲区 |
| `POS_PONG` | Pong缓冲区 |
| `POS_GLOBAL` | 全局内存 (GM) |

### 比较模式

| 模式 | 说明 |
|------|------|
| `LT` | 小于 |
| `LE` | 小于等于 |
| `GT` | 大于 |
| `GE` | 大于等于 |
| `EQ` | 等于 |
| `NE` | 不等于 |

### 归约操作

| 操作 | 说明 |
|------|------|
| `SUM` | 求和 |
| `MAX` | 最大值 |
| `MIN` | 最小值 |
| `PROD` | 乘积 |
| `AND` | 逻辑与 |
| `OR` | 逻辑或 |
