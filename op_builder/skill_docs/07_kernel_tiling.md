# Kernel Tiling

> Ascend C 算子Tiling数据获取与注册

## 目录

- [Tiling数据获取宏](#tiling数据获取宏)
- [Tiling数据注册](#tiling数据注册)
- [Tiling键管理](#tiling键管理)

---

## Tiling数据获取宏

### GET_TILING_DATA

获取Tiling数据。

```cpp
GET_TILING_DATA(tilingVarName);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| tilingVarName | identifier | Tiling变量名 |

### GET_TILING_DATA_WITH_STRUCT

使用结构体获取Tiling数据。

```cpp
GET_TILING_DATA_WITH_STRUCT(tilingStructType, tilingVarName);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| tilingStructType | type | Tiling结构体类型 |
| tilingVarName | identifier | Tiling变量名 |

### GET_TILING_DATA_MEMBER

获取Tiling数据成员。

```cpp
GET_TILING_DATA_MEMBER(tilingVarName, memberName);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| tilingVarName | identifier | Tiling变量名 |
| memberName | identifier | 成员名 |

### GET_TILING_DATA_PTR_WITH_STRUCT

使用结构体获取Tiling数据指针。

```cpp
GET_TILING_DATA_PTR_WITH_STRUCT(tilingStructType, tilingVarName);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| tilingStructType | type | Tiling结构体类型 |
| tilingVarName | identifier | Tiling变量名 |

---

## Tiling数据复制

### COPY_TILING_WITH_STRUCT

使用结构体复制Tiling数据。

```cpp
COPY_TILING_WITH_STRUCT(tilingStructType, srcVar, dstVar);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| tilingStructType | type | Tiling结构体类型 |
| srcVar | identifier | 源变量 |
| dstVar | identifier | 目标变量 |

### COPY_TILING_WITH_ARRAY

使用数组复制Tiling数据。

```cpp
COPY_TILING_WITH_ARRAY(arrayType, srcArray, dstArray, arraySize);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| arrayType | type | 数组类型 |
| srcArray | identifier | 源数组 |
| dstArray | identifier | 目标数组 |
| arraySize | uint32_t | 数组大小 |

---

## Tiling键管理

### TILING_KEY_IS

判断Tiling键是否匹配。

```cpp
TILING_KEY_IS(tilingKey);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| tilingKey | identifier | Tiling键 |

### REGISTER_TILING_DEFAULT

注册默认Tiling数据。

```cpp
REGISTER_TILING_DEFAULT(tilingFunc);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| tilingFunc | identifier | Tiling函数名 |

### REGISTER_TILING_FOR_TILINGKEY

为特定Tiling键注册Tiling数据。

```cpp
REGISTER_TILING_FOR_TILINGKEY(tilingKey, tilingFunc);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| tilingKey | identifier | Tiling键 |
| tilingFunc | identifier | Tiling函数名 |

---

## 设置Kernel类型

### 内核类型枚举

```cpp
enum class KernelType {
    AiKernel,      // AI Core核
    AicpuKernel,   // AI CPU核
};
```

### 设置Kernel类型

```cpp
void SetKernelType(KernelType type);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| type | KernelType | 内核类型 |

---

## Tiling结构体定义示例

```cpp
struct GemmTilingData {
    uint32_t blockDim;    // Block数量
    uint32_t mTail;       // M尾部
    uint32_t nTail;       // N尾部
    uint32_t mLength;     // M长度
    uint32_t nLength;     // N长度
    uint32_t kLength;     // K长度
    uint32_t mBlock;      // M块数
    uint32_t nBlock;      // N块数
};

struct OpTilingData {
    uint32_t repeatTimes;   // 重复次数
    uint32_t srcStride;    // 源步长
    uint32_t dstStride;    // 目标步长
    uint32_t length;       // 数据长度
};
```

---

## 使用示例

### 基本Tiling使用

```cpp
// 1. 定义Tiling结构体
struct MyOpTiling {
    uint32_t blockDim;
    uint32_t repeat;
    uint32_t srcStride;
};

// 2. Tiling函数 (Host侧)
void GetMyOpTiling(MyOpTiling& tiling, uint32_t length) {
    tiling.blockDim = 1;
    tiling.repeat = length / 32;  // 假设32元素为一次
    tiling.srcStride = tiling.repeat;
}

// 3. 注册Tiling
REGISTER_TILING_DEFAULT(GetMyOpTiling);

// 4. Kernel侧使用
__CCEATTR__ void KernelMyOp(KernelContext* ctx, void* param) {
    GET_TILING_DATA(tiling);
    
    // 使用tiling数据进行计算
    Exp(output, input, tiling.repeat);
}
```

### 完整示例

```cpp
// ==================== Tiling结构体 ====================
struct MatmulTiling {
    uint32_t blockDim;
    uint32_t mBlock;
    uint32_t nBlock;
    uint32_t kBlock;
    uint32_t mTail;
    uint32_t nTail;
    uint32_t kTail;
};

// ==================== Tiling函数 ====================
void MatmulTilingFunc(MatmulTiling& tiling, uint32_t m, uint32_t n, uint32_t k) {
    tiling.blockDim = (n + 255) / 256;
    tiling.mBlock = m / 16;
    tiling.nBlock = n / 256;
    tiling.kBlock = k / 16;
    tiling.mTail = m % 16;
    tiling.nTail = n % 256;
    tiling.kTail = k % 16;
}

// 注册
REGISTER_TILING_DEFAULT(MatmulTilingFunc);

// ==================== Kernel函数 ====================
__CCEATTR__ void KernelMatmul(KernelContext* ctx, void* param) {
    GET_TILING_DATA(tiling);
    
    TPipe pipe;
    pipe.Init();
    TQue<4, TPosition::POS_LOCAL> que;
    que.InitBuffer();
    
    // 分配张量
    auto inputA = que.AllocTensor<half>();
    auto inputB = que.AllocTensor<half>();
    auto outputC = que.AllocTensor<half>();
    
    // 数据搬运
    DataCopy(inputA, ctx->inputA, inputSize);
    DataCopy(inputB, ctx->inputB, inputSize);
    
    // Matmul计算
    // ... 使用tiling数据进行计算
    
    // 结果写回
    DataCopy(ctx->output, outputC, outputSize);
}
```

### 多Tiling键示例

```cpp
// 为不同数据形状注册不同Tiling
void Tiling128(MatmulTiling& tiling) { /* 128x128 */ }
void Tiling256(MatmulTiling& tiling) { /* 256x256 */ }
void Tiling512(MatmulTiling& tiling) { /* 512x512 */ }

REGISTER_TILING_FOR_TILINGKEY(128, Tiling128);
REGISTER_TILING_FOR_TILINGKEY(256, Tiling256);
REGISTER_TILING_FOR_TILINGKEY(512, Tiling512);

// Kernel中判断
__CCEATTR__ void KernelMatmul(KernelContext* ctx, void* param) {
    if (TILING_KEY_IS(128)) {
        GET_TILING_DATA(tiling);
        // 处理128场景
    } else if (TILING_KEY_IS(256)) {
        GET_TILING_DATA(tiling);
        // 处理256场景
    }
}
```

### COPY_TILING_WITH_STRUCT示例

```cpp
__CCEATTR__ void KernelFusion(KernelContext* ctx, void* param) {
    // 复制Tiling数据
    MatmulTiling mmTiling;
    MatmulTiling newMmTiling;
    COPY_TILING_WITH_STRUCT(MatmulTiling, mmTiling, newMmTiling);
    
    // 修改后使用
    newMmTiling.blockDim = mmTiling.blockDim * 2;
}
```

---

## 总结

Kernel Tiling是Ascend C算子开发中的关键步骤：

1. **定义Tiling结构体** - 根据算子需求定义
2. **实现Tiling函数** - Host侧计算tiling参数
3. **注册Tiling** - 使用REGISTER_TILING宏
4. **Kernel中使用** - 使用GET_TILING_DATA获取

Tiling的主要作用是：
- 决定AI Core的使用数量
- 划分数据块
- 优化内存使用
- 提高计算效率
