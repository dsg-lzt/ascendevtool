# 高阶API - 激活函数

> Ascend C 激活函数库

## 目录

- [SoftMax](#softmax)
- [LogSoftMax](#logsoftmax)
- [Gelu](#gelu)
- [SwiGLU](#swiglu)
- [Silu](#silu)
- [Swish](#swish)
- [GeGLU](#geglu)
- [ReGlu](#reglu)
- [Sigmoid](#sigmoid)

---

## SoftMax

SoftMax函数: `dst[i] = exp(src[i]) / sum(exp(src[j]))`

### 主函数

```cpp
void SoftMax(LocalTensor<DT> dst, LocalTensor<DT> src,
            LocalTensor<DT> tmp, uint32_t totalLength);

void SimpleSoftMax(LocalTensor<DT> dst, LocalTensor<DT> src,
                  LocalTensor<DT> tmp, uint32_t totalLength);

void SoftmaxFlash(LocalTensor<DT> dst, LocalTensor<DT> src,
                 LocalTensor<DT> tmp, uint32_t totalLength);

void SoftmaxGrad(LocalTensor<DT> grad, LocalTensor<DT> src,
                LocalTensor<DT> output, LocalTensor<DT> tmp,
                uint32_t totalLength);

void SoftmaxFlashV2(LocalTensor<DT> dst, LocalTensor<DT> src,
                    LocalTensor<DT> tmp, uint32_t totalLength);

void SoftmaxFlashV3(LocalTensor<DT> dst, LocalTensor<DT> src,
                    LocalTensor<DT> tmp, uint32_t totalLength);

void SoftmaxGradFront(LocalTensor<DT> grad, LocalTensor<DT> src,
                     LocalTensor<DT> output, LocalTensor<DT> tmp,
                     uint32_t totalLength);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src | LocalTensor | 输入张量 |
| grad | LocalTensor | 梯度输入 (用于梯度函数) |
| output | LocalTensor | SoftMax输出 (用于梯度函数) |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 总数据长度 |

### 获取临时缓冲区大小

```cpp
uint32_t GetSoftMaxMaxMinTmpSize();
uint32_t GetSimpleSoftMaxMaxMinTmpSize();
uint32_t GetSoftmaxFlashMaxMinTmpSize();
uint32_t GetSoftmaxGradMaxMinTmpSize();
uint32_t GetSoftmaxFlashV2MaxMinTmpSize();
uint32_t GetSoftmaxFlashV3MaxMinTmpSize();
```

### Tiling接口

```cpp
// SoftMax Tiling
uint32_t SoftMaxGetTmpBufferSize(uint32_t length, bool isForward);
uint32_t SimpleSoftMaxGetTmpBufferSize(uint32_t length);

// Flash版本
uint32_t SoftmaxFlashGetTmpBufferSize(uint32_t length);
uint32_t SoftmaxFlashV2GetTmpBufferSize(uint32_t length);
uint32_t SoftmaxFlashV3GetTmpBufferSize(uint32_t length);

// 梯度版本
uint32_t SoftmaxGradGetTmpBufferSize(uint32_t length);
```

### AdjustSoftMaxRes

调整SoftMax结果。

```cpp
void AdjustSoftMaxRes(LocalTensor<DT> dst, LocalTensor<DT> src,
                    uint32_t length);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| length | uint32_t | 长度 |

### IsBasicBlockInSoftMax

判断是否在SoftMax基本块中。

```cpp
bool IsBasicBlockInSoftMax();
```

- **返回值**: 是否在SoftMax块中 |

---

## LogSoftMax

Log SoftMax: `dst[i] = log(exp(src[i]) / sum(exp(src[j])))`

### 主函数

```cpp
void LogSoftMax(LocalTensor<DT> dst, LocalTensor<DT> src,
               LocalTensor<DT> tmp, uint32_t totalLength);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src | LocalTensor | 输入张量 |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 总数据长度 |

### Tiling接口

```cpp
uint32_t LogSoftMaxGetTmpBufferSize(uint32_t length);
```

---

## Gelu

Gaussian Error Linear Unit: `dst[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`

### 主函数

```cpp
void Gelu(LocalTensor<DT> dst, LocalTensor<DT> src,
         LocalTensor<DT> tmp, uint32_t totalLength);

void FasterGelu(LocalTensor<DT> dst, LocalTensor<DT> src,
               LocalTensor<DT> tmp, uint32_t totalLength);

void FasterGeluV2(LocalTensor<DT> dst, LocalTensor<DT> src,
                 LocalTensor<DT> tmp, uint32_t totalLength);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src | LocalTensor | 输入张量 |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 总数据长度 |

### 获取临时缓冲区大小

```cpp
uint32_t GetGeluMaxMinTmpSize();
```

---

## SwiGLU

Swish门控线性单元: `dst[i] = swish(x1[i]) * x2[i]`

### 主函数

```cpp
void SwiGLU(LocalTensor<DT> dst, LocalTensor<DT> src0,
           LocalTensor<DT> src1, LocalTensor<DT> tmp, uint32_t totalLength);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src0 | LocalTensor | 输入张量0 |
| src1 | LocalTensor | 输入张量1 |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 总数据长度 |

### 获取临时缓冲区大小

```cpp
uint32_t GetSwiGLUMaxMinTmpSize();
uint32_t GetSwiGLUTmpBufferFactorSize();
```

---

## Silu

Sigmoid Linear Unit (SiLU): `dst[i] = x * sigmoid(x)`

### 主函数

```cpp
void Silu(LocalTensor<DT> dst, LocalTensor<DT> src,
         LocalTensor<DT> tmp, uint32_t totalLength);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src | LocalTensor | 输入张量 |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 总数据长度 |

### 获取临时缓冲区大小

```cpp
uint32_t GetSiluTmpSize();
```

---

## Swish

Swish: `dst[i] = x * sigmoid(beta * x)`

### 主函数

```cpp
void Swish(LocalTensor<DT> dst, LocalTensor<DT> src,
          LocalTensor<DT> tmp, uint32_t totalLength);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src | LocalTensor | 输入张量 |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 总数据长度 |

### 获取临时缓冲区大小

```cpp
uint32_t GetSwishTmpSize();
```

---

## GeGLU

Gated GELU: `dst[i] = gelu(x1[i]) * x2[i]`

### 主函数

```cpp
void GeGLU(LocalTensor<DT> dst, LocalTensor<DT> src0,
          LocalTensor<DT> src1, LocalTensor<DT> tmp, uint32_t totalLength);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src0 | LocalTensor | 输入张量0 |
| src1 | LocalTensor | 输入张量1 |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 总数据长度 |

### 获取临时缓冲区大小

```cpp
uint32_t GetGeGLUMaxMinTmpSize();
uint32_t GetGeGLUTmpBufferFactorSize();
```

---

## ReGlu

ReLU门控: `dst[i] = max(0, x1[i]) * x2[i]`

### 主函数

```cpp
void ReGlu(LocalTensor<DT> dst, LocalTensor<DT> src0,
          LocalTensor<DT> src1, LocalTensor<DT> tmp, uint32_t totalLength);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src0 | LocalTensor | 输入张量0 |
| src1 | LocalTensor | 输入张量1 |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 总数据长度 |

### 获取临时缓冲区大小

```cpp
uint32_t GetReGluMaxMinTmpSize();
```

---

## Sigmoid

Sigmoid: `dst[i] = 1 / (1 + exp(-x[i]))`

### 主函数

```cpp
void Sigmoid(LocalTensor<DT> dst, LocalTensor<DT> src,
           LocalTensor<DT> tmp, uint32_t totalLength);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src | LocalTensor | 输入张量 |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 总数据长度 |

### 获取临时缓冲区大小

```cpp
uint32_t GetSigmoidMaxMinTmpSize();
```

---

## 使用示例

```cpp
// SoftMax示例
void SoftmaxExample() {
    uint32_t tmpSize = GetSoftMaxMaxMinTmpSize();
    LocalTensor<float> tmp = que.AllocTensor();
    
    SoftMax(output, input, tmp, 1024);
    
    que.FreeTensor(tmp);
}

// Gelu示例
void GeluExample() {
    uint32_t tmpSize = GetGeluMaxMinTmpSize();
    LocalTensor<float> tmp = que.AllocTensor();
    
    Gelu(output, input, tmp, 1024);
    
    que.FreeTensor(tmp);
}

// SwiGLU示例 (需要两个输入)
void SwiGLUExample() {
    uint32_t tmpSize = GetSwiGLUMaxMinTmpSize();
    LocalTensor<float> tmp = que.AllocTensor();
    
    SwiGLU(output, input0, input1, tmp, 1024);
    
    que.FreeTensor(tmp);
}

// SoftMax梯度
void SoftmaxGradExample() {
    uint32_t tmpSize = GetSoftmaxGradMaxMinTmpSize();
    LocalTensor<float> tmp = que.AllocTensor();
    
    SoftmaxGrad(grad, input, output, tmp, 1024);
    
    que.FreeTensor(tmp);
}
```