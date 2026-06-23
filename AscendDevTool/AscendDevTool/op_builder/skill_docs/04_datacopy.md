# 数据搬运

> Ascend C 数据传输API - DataCopy及其变体

## 目录

- [DataCopy 简介](#datacopy-简介)
- [基础数据搬运](#基础数据搬运)
- [增强数据搬运](#增强数据搬运)
- [切片数据搬运](#切片数据搬运)
- [随路转换搬运](#随路转换搬运)
- [随路量化搬运](#随路量化搬运)
- [其他数据操作](#其他数据操作)

---

## DataCopy 简介

DataCopy是Ascend C中最核心的数据搬运API，用于在不同存储位置之间复制数据。

### 基本原型

```cpp
void DataCopy(LocalTensor<DT> dst, GlobalTensor<DT> src, uint32_t length);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | GlobalTensor | 源张量 |
| length | uint32_t | 数据长度 |

---

## 基础数据搬运

### Global到Local

从全局内存复制到本地内存。

```cpp
void DataCopy(LocalTensor<DT> dst, GlobalTensor<DT> src, uint32_t length);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 本地目标张量 |
| src | GlobalTensor | 全局源张量 |
| length | uint32_t | 数据长度(字节) |

### Local到Global

从本地内存复制到全局内存。

```cpp
void DataCopy(GlobalTensor<DT> dst, LocalTensor<DT> src, uint32_t length);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | GlobalTensor | 全局目标张量 |
| src | LocalTensor | 本地源张量 |
| length | uint32_t | 数据长度(字节) |

### Local到Local

本地内存到本地内存的复制。

```cpp
void DataCopy(LocalTensor<DT> dst, LocalTensor<DT> src, uint32_t length);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标本地张量 |
| src | LocalTensor | 源本地张量 |
| length | uint32_t | 数据长度(字节) |

### 带重复的数据搬运

使用RepeatParams进行重复数据搬运。

```cpp
void DataCopy(LocalTensor<DT> dst, GlobalTensor<DT> src, 
             UnaryRepeatParams param);

void DataCopy(GlobalTensor<DT> dst, LocalTensor<DT> src, 
             UnaryRepeatParams param);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor/GlobalTensor | 目标张量 |
| src | GlobalTensor/LocalTensor | 源张量 |
| param | UnaryRepeatParams | 重复参数 |

### 使用示例

```cpp
// 基础复制
LocalTensor<float> input = que.AllocTensor();
DataCopy(input, ctx->input, 1024);

// 带重复复制
UnaryRepeatParams params;
params.repeatSize = 256;
params.repeatTimes = 4;
params.src0Stride = 256;

DataCopy(output, input, params);
```

---

## 增强数据搬运

### 带偏移的数据搬运

带源和目标偏移的复制。

```cpp
void DataCopy(LocalTensor<DT> dst, uint32_t dstOffset,
             GlobalTensor<DT> src, uint32_t srcOffset,
             uint32_t length);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| dstOffset | uint32_t | 目标偏移量 |
| src | GlobalTensor | 源张量 |
| srcOffset | uint32_t | 源偏移量 |
| length | uint32_t | 数据长度 |

### 固定步长的数据搬运

指定源和目标的步长。

```cpp
void DataCopy(LocalTensor<DT> dst, GlobalTensor<DT> src,
             uint32_t dstStride, uint32_t srcStride,
             uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | GlobalTensor | 源张量 |
| dstStride | uint32_t | 目标步长 |
| srcStride | uint32_t | 源步长 |
| repeatTimes | uint32_t | 重复次数 |

### 使用BinaryRepeatParams

```cpp
void DataCopy(LocalTensor<DT> dst, GlobalTensor<DT> src,
             BinaryRepeatParams param);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | GlobalTensor | 源张量 |
| param | BinaryRepeatParams | 二元重复参数 |

### 完整参数版本

```cpp
void DataCopy(LocalTensor<DT> dst, uint32_t dstOffset,
             GlobalTensor<DT> src, uint32_t srcOffset,
             uint32_t dstStride, uint32_t srcStride,
             uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| dstOffset | uint32_t | 目标偏移 |
| src | GlobalTensor | 源张量 |
| srcOffset | uint32_t | 源偏移 |
| dstStride | uint32_t | 目标步长 |
| srcStride | uint32_t | 源步长 |
| repeatTimes | uint32_t | 重复次数 |

---

## 切片数据搬运

### 切片复制

从源张量复制指定范围到目标张量。

```cpp
void DataCopy(LocalTensor<DT> dst, GlobalTensor<DT> src,
             uint32_t srcOffset, uint32_t copySize,
             uint32_t repeatTimes, uint32_t srcStride);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | GlobalTensor | 源张量 |
| srcOffset | uint32_t | 源偏移 |
| copySize | uint32_t | 每次复制大小 |
| repeatTimes | uint32_t | 重复次数 |
| srcStride | uint32_t | 源步长 |

### 示例

```cpp
// 复制多个非连续片段
// 每次从源复制256字节，重复4次，每次间隔512字节
DataCopy(dst, src, 0, 256, 4, 512);
```

---

## 随路转换搬运

### ND2NZ 转换搬运

从N-D格式转换为NZ格式。

```cpp
void DataCopy(LocalTensor<DT> dst, GlobalTensor<DT> src,
             TensorDesc tensorDesc, DataTypeConvertMode mode);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | GlobalTensor | 源张量 |
| tensorDesc | TensorDesc | 张量描述符 |
| mode | DataTypeConvertMode | 转换模式 |

### NZ2ND 转换搬运

从NZ格式转换为N-D格式。

```cpp
void DataCopy(GlobalTensor<DT> dst, LocalTensor<DT> src,
             TensorDesc tensorDesc, DataTypeConvertMode mode);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | GlobalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| tensorDesc | TensorDesc | 张量描述符 |
| mode | DataTypeConvertMode | 转换模式 |

### DataTypeConvertMode 枚举

```cpp
enum class DataTypeConvertMode {
    NONE,      // 无转换
    ND2NZ,     // ND转NZ
    NZ2ND,     // NZ转ND
    ND2NZ_ND,  // ND到NZ到ND
    ND2NZ_5D,  // ND到5D
};
```

---

## 随路量化搬运

### 量化激活搬运

带量化的数据搬运。

```cpp
void DataCopy(LocalTensor<DT> dst, GlobalTensor<DT> src,
             TensorDesc tensorDesc, QuantizeParam quantParam);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | GlobalTensor | 源张量 |
| tensorDesc | TensorDesc | 张量描述符 |
| quantParam | QuantizeParam | 量化参数 |

### 反量化激活搬运

带反量化的数据搬运。

```cpp
void DataCopy(LocalTensor<DT> dst, GlobalTensor<DT> src,
             TensorDesc tensorDesc, DequantizeParam deqParam);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | GlobalTensor | 源张量 |
| tensorDesc | TensorDesc | 张量描述符 |
| deqParam | DequantizeParam | 反量化参数 |

### QuantizeParam 结构

```cpp
struct QuantizeParam {
    float scale;         // 量化尺度
    float offset;        // 量化偏移
    int32_t roundMode;  // 舍入模式
};
```

### DequantizeParam 结构

```cpp
struct DequantizeParam {
    float scale;        // 反量化尺度
    int32_t roundMode; // 舍入模式
};
```

---

## 其他数据操作

### Copy

通用数据复制 (Local到Local)。

```cpp
void Copy(LocalTensor<DT> dst, LocalTensor<DT> src, uint32_t length);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| length | uint32_t | 数据长度 |

### DataCopyPad (ISASI)

带填充的数据复制。

```cpp
void DataCopyPad(LocalTensor<DT> dst, GlobalTensor<DT> src,
                PadAttr padAttr);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | GlobalTensor | 源张量 |
| padAttr | PadAttr | 填充属性 |

### PadAttr 结构 (ISASI)

```cpp
struct PadAttr {
    uint32_t padValue;     // 填充值
    uint32_t leftPad;      // 左侧填充
    uint32_t rightPad;     // 右侧填充
    uint32_t topPad;      // 顶部填充
    uint32_t bottomPad;   // 底部填充
};
```

### SetPadValue (ISASI)

设置填充值。

```cpp
void SetPadValue(DT value);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| value | DT | 填充值 |

---

## ISASI 特定API

以下API为ISASI (Inner Software Adaptive Specific Instruction) 特定指令：

### SetFixPipeConfig

设置定标管道配置。

```cpp
void SetFixPipeConfig(FixPipeConfig config);
```

### SetFixpipeNz2ndFlag

设置NZ到ND转换标志。

```cpp
void SetFixpipeNz2ndFlag(bool flag);
```

### SetFixpipePreQuantFlag

设置预量化标志。

```cpp
void SetFixpipePreQuantFlag(bool flag);
```

### SetFixPipeClipRelu

设置定标管道Clip ReLU。

```cpp
void SetFixPipeClipRelu(bool enable);
```

### SetFixPipeAddr

设置定标管道地址。

```cpp
void SetFixPipeAddr(void* addr);
```

### InitConstValue

初始化常量值。

```cpp
void InitConstValue(DT value, uint32_t count);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| value | DT | 常量值 |
| count | uint32_t | 数量 |

### LoadData

加载数据。

```cpp
void LoadData(LocalTensor<DT> dst, GlobalTensor<DT> src, 
             LoadConfig config);
```

### LoadDataWithTranspose

带转置加载数据。

```cpp
void LoadDataWithTranspose(LocalTensor<DT> dst, GlobalTensor<DT> src,
                          LoadConfig config, TransposeParam param);
```

### LoadDataWithSparse

带稀疏加载数据。

```cpp
void LoadDataWithSparse(LocalTensor<DT> dst, GlobalTensor<DT> src,
                       LoadConfig config, SparseParam param);
```

### Fixpipe

定标管道操作。

```cpp
void Fixpipe(LocalTensor<DT> dst, LocalTensor<DT> src,
            FixPipeAttr attr);
```

---

## 完整使用示例

### 基本数据搬运流程

```cpp
__CCEATTR__ void KernelExample(KernelContext* ctx, void* param) {
    // 1. 初始化
    TPipe pipe;
    pipe.Init();
    
    TQue<4, TPosition::POS_LOCAL> que;
    que.InitBuffer();
    
    // 2. 分配张量
    LocalTensor<float> input = que.AllocTensor();
    LocalTensor<float> output = que.AllocTensor();
    
    // 3. GM -> UB 复制
    DataCopy(input, ctx->input, 4096);
    
    // 4. 计算
    Exp(output, input, 128);
    
    // 5. UB -> GM 复制
    DataCopy(ctx->output, output, 4096);
    
    pipe.Destroy();
}
```

### 带偏移和步长的复制

```cpp
// 从输入张量的第1024字节位置开始，复制2048字节
DataCopy(localBuf, 0, globalSrc, 1024, 2048);

// 复制多个片段，每次256字节，重复8次，间隔512字节
DataCopy(localBuf, globalSrc, 256, 8, 512);
```

### 带转换的复制

```cpp
// 设置张量描述符
TensorDesc desc;
desc.SetShape(shape, 4);

// ND到NZ转换复制
DataCopy(localDst, globalSrc, desc, DataTypeConvertMode::ND2NZ);

// NZ到ND转换复制  
DataCopy(globalDst, localSrc, desc, DataTypeConvertMode::NZ2ND);
```

### 带量化的复制

```cpp
// 量化参数
DequantizeParam deqParam;
deqParam.scale = 0.01f;
deqParam.roundMode = 0;

// 带反量化的数据复制
DataCopy(localDst, globalSrc, desc, deqParam);
```
