# 高阶API - 排序与数据填充

> Ascend C 排序、数据填充相关函数

## 目录

- [排序](#排序)
- [数据填充](#数据填充)
- [索引操作](#索引操作)
- [比较选择](#比较选择)
- [数据过滤](#数据过滤)
- [变形](#变形)

---

## 排序

### TopK

Top-K 排序: 找出前K个最大/最小值。

```cpp
void TopK(LocalTensor<DT> dst, LocalTensor<DT> src,
         LocalTensor<DT> indices, uint32_t k, bool largest,
         LocalTensor<DT> tmp, uint32_t length);

uint32_t GetTopKTmpBufferSize(uint32_t length, uint32_t k);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 (排序后的值) |
| src | LocalTensor | 输入张量 |
| indices | LocalTensor | 输出索引 |
| k | uint32_t | K值 |
| largest | bool | true=找最大值, false=找最小值 |
| tmp | LocalTensor | 临时缓冲区 |
| length | uint32_t | 输入长度 |

### Sort

排序: 对数据进行排序。

```cpp
void Sort(LocalTensor<DT> dst, LocalTensor<DT> src,
         LocalTensor<DT> tmp, uint32_t length, bool descending);

uint32_t GetSortTmpBufferSize(uint32_t length);
uint32_t GetSortOffset();
uint32_t GetSortLen();
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src | LocalTensor | 输入张量 |
| tmp | LocalTensor | 临时缓冲区 |
| length | uint32_t | 数据长度 |
| descending | bool | true=降序, false=升序 |

### MrgSort

归并排序。

```cpp
void MrgSort(LocalTensor<DT> dst, LocalTensor<DT> src,
            LocalTensor<DT> tmp, uint32_t length);

uint32_t GetMrgSortTmpBufferSize(uint32_t length);
```

### Concat

连接: 将多个张量连接成一个。

```cpp
void Concat(LocalTensor<DT> dst, LocalTensor<DT>* srcs,
           uint32_t num, uint32_t axis, LocalTensor<DT> tmp);

uint32_t GetConcatTmpBufferSize(uint32_t totalLength);
```

### Extract

提取: 从张量中提取指定元素。

```cpp
void Extract(LocalTensor<DT> dst, LocalTensor<DT> src,
            uint32_t* indices, uint32_t num, LocalTensor<DT> tmp);
```

---

## 数据填充

### Pad

填充: 在张量边缘填充值。

```cpp
void Pad(LocalTensor<DT> dst, LocalTensor<DT> src,
        PadAttr padAttr, LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetPadTmpBufferSize(uint32_t totalLength);
```

#### PadAttr 结构

```cpp
struct PadAttr {
    DT padValue;          // 填充值
    PadMode mode;        // 填充模式
    uint32_t padDimNum; // 填充维度数
    uint32_t* padLeft;  // 左侧填充
    uint32_t* padRight; // 右侧填充
};
```

#### PadMode 枚举

```cpp
enum class PadMode {
    PAD,          // 常量填充
    REFLECT,     // 反射填充
    REPLICATE,   // 复制填充
    CIRCULAR,    // 循环填充
};
```

### UnPad

取消填充: 移除填充区域。

```cpp
void UnPad(LocalTensor<DT> dst, LocalTensor<DT> src,
           UnPadAttr unpadAttr, LocalTensor<DT> tmp, uint32_t srcLength);

uint32_t GetUnPadTmpBufferSize(uint32_t srcLength);
```

### Broadcast

广播: 将数据广播到更大形状。

```cpp
void Broadcast(LocalTensor<DT> dst, LocalTensor<DT> src,
              LocalTensor<DT> tmp, uint32_t srcLength,
              uint32_t* targetShape, uint32_t dimNum);

uint32_t GetBroadCastMaxMinTmpSize();
```

---

## 索引操作

### ArithProgression

等差数列: 生成等差数列。

```cpp
void ArithProgression(LocalTensor<DT> dst, DT start, DT step,
                     uint32_t length);

uint32_t GetArithProgressionMaxMinTmpSize();
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| start | DT | 起始值 |
| step | DT | 步长 |
| length | uint32_t | 长度 |

---

## 比较选择

### SelectWithBytesMask

使用字节掩码进行条件选择。

```cpp
void SelectWithBytesMask(LocalTensor<DT> dst, LocalTensor<DT> src0,
                        LocalTensor<DT> src1, LocalTensor<uint8_t> mask,
                        uint32_t length);

uint32_t GetSelectWithBytesMaskMaxMinTmpSize();
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src0 | LocalTensor | 条件为false时的值 |
| src1 | LocalTensor | 条件为true时的值 |
| mask | LocalTensor | 字节掩码 |
| length | uint32_t | 数据长度 |

---

## 数据过滤

### DropOut

Dropout: 随机置零。

```cpp
void DropOut(LocalTensor<DT> dst, LocalTensor<DT> src,
             LocalTensor<uint8_t> mask, DT keepProb,
             LocalTensor<DT> tmp, uint32_t length);

uint32_t GetDropOutMaxMinTmpSize();
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src | LocalTensor | 输入张量 |
| mask | LocalTensor | 掩码张量 |
| keepProb | DT | 保留概率 |
| tmp | LocalTensor | 临时缓冲区 |
| length | uint32_t | 数据长度 |

---

## 变形

### TransData

数据格式转换。

```cpp
void TransData(LocalTensor<DT> dst, LocalTensor<DT> src,
               TransParam param, LocalTensor<DT> tmp,
               uint32_t length);

uint32_t GetTransDataMaxMinTmpSize();
```

### ConfusionTranspose

混淆转置。

```cpp
void ConfusionTranspose(LocalTensor<DT> dst, LocalTensor<DT> src,
                       TransParam param, LocalTensor<DT> tmp,
                       uint32_t length);

uint32_t GetConfusionTransposeTmpBufferSize();
```

#### TransParam 结构

```cpp
struct TransParam {
    TransType type;       // 转换类型
    uint32_t* srcShape;   // 源形状
    uint32_t* dstShape;   // 目标形状
    uint32_t dimNum;      // 维度数
};
```

#### TransType 枚举

```cpp
enum class TransType {
    NONE,
    NCHW_TO_NHWC,
    NHWC_TO_NCHW,
    NCHW_TO_NC1HWC0,
    NC1HWC0_TO_NCHW,
    // ... 其他类型
};
```

---

## 使用示例

```cpp
// TopK 示例
void TopKExample() {
    uint32_t k = 10;
    uint32_t tmpSize = GetTopKTmpBufferSize(1024, k);
    LocalTensor<float> tmp = que.AllocTensor();
    LocalTensor<float> indices = que.AllocTensor();
    
    TopK(output, input, indices, k, true, tmp, 1024);
    
    que.FreeTensor(tmp);
    que.FreeTensor(indices);
}

// Pad 示例
void PadExample() {
    PadAttr attr;
    attr.padValue = 0.0f;
    attr.mode = PadMode::PAD;
    
    uint32_t tmpSize = GetPadTmpBufferSize(1024);
    LocalTensor<float> tmp = que.AllocTensor();
    
    Pad(output, input, attr, tmp, 1024);
    
    que.FreeTensor(tmp);
}

// Broadcast 示例
void BroadcastExample() {
    uint32_t targetShape[] = {1, 32, 224, 224};
    uint32_t tmpSize = GetBroadCastMaxMinTmpSize();
    LocalTensor<float> tmp = que.AllocTensor();
    
    Broadcast(output, input, tmp, 32, targetShape, 4);
    
    que.FreeTensor(tmp);
}
```