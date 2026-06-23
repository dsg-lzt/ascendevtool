# 高阶API - Matmul

> Ascend C 矩阵乘法库

## 目录

- [Kernel侧接口](#kernel侧接口)
- [Tiling侧接口](#tiling侧接口)
- [使用示例](#使用示例)

---

## Kernel侧接口

### Matmul

矩阵乘法核心函数。

```cpp
template<typename SrcAType, typename SrcBType, typename DstType>
void Matmul(LocalTensor<DstType> dst, 
           LocalTensor<SrcAType> tensorA,
           LocalTensor<SrcBType> tensorB,
           MatmulConfig<SrcAType, SrcBType, DstType> config);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| tensorA | LocalTensor | 输入矩阵A |
| tensorB | LocalTensor | 输入矩阵B |
| config | MatmulConfig | 矩阵乘法配置 |

### MatmulConfig

矩阵乘法配置类。

```cpp
template<typename SrcAType, typename SrcBType, typename DstType>
class MatmulConfig {
public:
    // 初始化
    void Init(uint32_t m, uint32_t n, uint32_t k);
    
    // 设置张量
    void SetTensorA(const LocalTensor<SrcAType>& tensorA);
    void SetTensorB(const LocalTensor<SrcBType>& tensorB);
    void SetBias(const LocalTensor<float>& bias);
    void DisableBias();
    
    // 设置参数
    void SetSingleShape(uint32_t m, uint32_t n, uint32_t k);
    void SetBatchNum(uint32_t batchNum);
    
    // 迭代控制
    void Iterate();
    void IterateAll();
    void WaitIterateAll();
    void IterateBatch(uint32_t batchId);
    void WaitIterateBatch();
    void IterateNBatch(uint32_t batchId, uint32_t n);
    void End();
    
    // 获取结果
    LocalTensor<DstType> GetTensorC();
    void WaitGetTensorC();
    uint32_t GetOffsetC();
    void AsyncGetTensorC();
    
    // 获取张量C
    LocalTensor<DstType> GetBatchC();
};
```

#### Init

初始化配置。

```cpp
void Init(uint32_t m, uint32_t n, uint32_t k);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| m | uint32_t | 矩阵A的行数 |
| n | uint32_t | 矩阵B的列数 |
| k | uint32_t | 公共维度 |

#### SetTensorA

设置张量A。

```cpp
void SetTensorA(const LocalTensor<SrcAType>& tensorA);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| tensorA | LocalTensor | 输入矩阵A |

#### SetTensorB

设置张量B。

```cpp
void SetTensorB(const LocalTensor<SrcBType>& tensorB);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| tensorB | LocalTensor | 输入矩阵B |

#### SetBias

设置偏置。

```cpp
void SetBias(const LocalTensor<float>& bias);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| bias | LocalTensor | 偏置张量 |

#### Iterate

单次迭代。

```cpp
void Iterate();
```

#### IterateAll

迭代全部。

```cpp
void IterateAll();
```

#### WaitIterateAll

等待全部迭代完成。

```cpp
void WaitIterateAll();
```

#### IterateBatch

迭代批处理。

```cpp
void IterateBatch(uint32_t batchId);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| batchId | uint32_t | 批处理ID |

#### GetTensorC

获取结果张量。

```cpp
LocalTensor<DstType> GetTensorC();
```

- **返回值**: 输出张量 |

#### WaitGetTensorC

等待获取结果。

```cpp
void WaitGetTensorC();
```

#### AsyncGetTensorC

异步获取结果。

```cpp
void AsyncGetTensorC();
```

#### End

结束计算。

```cpp
void End();
```

### MatmulPolicy

矩阵乘法策略。

```cpp
class MatmulPolicy {
public:
    // 获取配置
    static MatmulConfig GetNormalConfig();
    static MatmulConfig GetMDLConfig();
    static MatmulConfig GetSpecialMDLConfig();
    static MatmulConfig GetIBShareNormConfig();
    static MatmulConfig GetBasicConfig();
    static MatmulConfig GetSpecialBasicConfig();
    static MatmulConfig GetMMConfig();
};
```

### 使用示例

```cpp
void MatmulExample() {
    using Config = MatmulConfig<half, half, half>;
    Config config;
    
    // 初始化
    config.Init(m, n, k);
    
    // 设置张量
    config.SetTensorA(tensorA);
    config.SetTensorB(tensorB);
    
    // 执行计算
    Matmul(tensorC, tensorA, tensorB, config);
    
    // 或者使用迭代
    config.IterateAll();
    config.WaitIterateAll();
    
    // 获取结果
    auto result = config.GetTensorC();
}
```

---

## Tiling侧接口

### Matmul Tiling 类

矩阵乘法Tiling计算类。

```cpp
class MatmulTiling {
public:
    MatmulTiling();
    ~MatmulTiling();
    
    // 设置类型
    void SetAType(DataType type);
    void SetBType(DataType type);
    void SetCType(DataType type);
    void SetBiasType(DataType type);
    
    // 设置形状
    void SetShape(uint32_t m, uint32_t n, uint32_t k);
    void SetOrgShape(uint32_t m, uint32_t n, uint32_t k);
    void SetSingleShape(uint32_t m, uint32_t n, uint32_t k);
    void SetBatchNum(uint32_t batchNum);
    
    // 获取形状
    void GetBaseM(uint32_t& m);
    void GetBaseN(uint32_t& n);
    void GetBaseK(uint32_t& k);
    void GetTiling(TilingData& tiling);
    void GetSingleShape(uint32_t& m, uint32_t& n, uint32_t& k);
    void GetCoreNum(uint32_t& coreNum);
    
    // 配置选项
    void SetDim(uint32_t dim);
    void SetSplitRange(uint32_t splitM, uint32_t splitN);
    void SetDoubleBuffer(bool enable);
    void EnableMultiCoreSplitK(bool enable);
    void EnableBias(bool enable);
    void SetBufferSpace(void* buffer, uint32_t size);
    void SetTraverse(TraverseType type);
    void SetAlignSplit(bool enable);
    void SetDequantType(DequantType type);
    
    // 布局设置
    void SetALayout(LayoutType layout);
    void SetBLayout(LayoutType layout);
    void SetCLayout(LayoutType layout);
    
    // 分割K设置
    void SetSplitK(uint32_t splitK);
    void SetSparse(bool enable);
};
```

#### 构造函数

```cpp
MatmulTiling();
```

#### SetShape

设置矩阵形状。

```cpp
void SetShape(uint32_t m, uint32_t n, uint32_t k);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| m | uint32_t | M维度 |
| n | uint32_t | N维度 |
| k | uint32_t | K维度 |

#### SetAType/SetBType/SetCType

设置数据类型。

```cpp
void SetAType(DataType type);
void SetBType(DataType type);
void SetCType(DataType type);
```

#### SetBatchNum

设置批次数。

```cpp
void SetBatchNum(uint32_t batchNum);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| batchNum | uint32_t | 批次数 |

#### EnableMultiCoreSplitK

启用多核分割K。

```cpp
void EnableMultiCoreSplitK(bool enable);
```

#### SetDoubleBuffer

启用双缓冲。

```cpp
void SetDoubleBuffer(bool enable);
```

#### GetTiling

获取Tiling数据。

```cpp
void GetTiling(TilingData& tiling);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| tiling | TilingData& | Tiling数据输出 |

#### GetCoreNum

获取所需的AI Core数量。

```cpp
void GetCoreNum(uint32_t& coreNum);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| coreNum | uint32_t& | AI Core数量输出 |

### TCubeTiling 结构体

Cube Tiling结构。

```cpp
struct TCubeTiling {
    uint32_t blockDim;      // Block维度
    uint32_t mTail;        // M尾部
    uint32_t nTail;        // N尾部
    uint32_t kFix;         // K固定值
    uint32_t kRem;         // K余数
    uint32_t splitKMode;   // 分割K模式
    uint32_t splitKBatch;  // 分割K批次数
    ...
};
```

### 获���临��缓冲区大小

```cpp
// 单核
uint32_t MatmulGetTmpBufSize();
uint32_t MatmulGetTmpBufSizeV2();

// 多核
uint32_t MultiCoreMatmulGetTmpBufSize();
uint32_t MultiCoreMatmulGetTmpBufSizeV2();

// 批处理
uint32_t BatchMatmulGetTmpBufSize();
uint32_t BatchMatmulGetTmpBufSizeV2();
```

---

## 使用示例

### 完整Matmul调用

```cpp
// Tiling侧
void GetMatmulTiling(MatmulTilings& tilingData) {
    MatmulTiling tiling;
    
    // 设置类型
    tiling.SetAType(DataType::FP16);
    tiling.SetBType(DataType::FP16);
    tiling.SetCType(DataType::FP16);
    
    // 设置形状
    tiling.SetShape(m, n, k);
    
    // 获取Tiling
    tiling.GetTiling(tilingData.tiling);
    tiling.GetCoreNum(tilingData.coreNum);
    
    // 获取临时缓冲区大小
    tilingData.tmpSize = MultiCoreMatmulGetTmpBufSizeV2();
}

// Kernel侧
__CCEATTR__ void KernelMatmul(KernelContext* ctx, void* param) {
    GET_TILING_DATA(tiling);
    
    TPipe pipe;
    pipe.Init();
    
    // 初始化配置
    using Config = MatmulConfig<half, half, half>;
    Config config;
    config.Init(tiling.m, tiling.n, tiling.k);
    config.SetTensorA(tensorA);
    config.SetTensorB(tensorB);
    
    // 执行乘加
    Matmul(tensorC, tensorA, tensorB, config);
    
    pipe.Destroy();
}
```

### 批处理Matmul

```cpp
void BatchMatmulExample() {
    Config config;
    config.SetBatchNum(batchNum);
    config.SetSingleShape(m, n, k);
    
    // 迭代所有批次
    for (uint32_t b = 0; b < batchNum; ++b) {
        config.IterateBatch(b);
    }
    
    config.WaitIterateAll();
    
    // 获取所有批次结果
    for (uint32_t b = 0; b < batchNum; ++b) {
        auto tensorC = config.GetBatchC();
        // 处理结果...
    }
}
```

### 带偏置的Matmul

```cpp
void MatmulWithBias() {
    Config config;
    config.Init(m, n, k);
    config.SetTensorA(tensorA);
    config.SetTensorB(tensorB);
    config.SetBias(bias);  // 设置偏置
    
    // 或者禁用偏置
    config.DisableBias();
    
    Matmul(tensorC, tensorA, tensorB, config);
}
```

### 使用策略获取配置

```cpp
void UsePolicy() {
    // 使用标准配置
    auto config = MatmulPolicy::GetNormalConfig();
    
    // 使用MDL配置
    auto mdlConfig = MatmulPolicy::GetMDLConfig();
    
    // 使用基础配置
    auto basicConfig = MatmulPolicy::GetBasicConfig();
}
```