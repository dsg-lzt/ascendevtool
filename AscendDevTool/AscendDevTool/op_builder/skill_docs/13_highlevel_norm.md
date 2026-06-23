# 高阶API - 数据归一化

> Ascend C 归一化函数库

## 目录

- [LayerNorm](#layernorm)
- [RmsNorm](#rmsnorm)
- [BatchNorm](#batchnorm)
- [DeepNorm](#deepnorm)
- [Normalize](#normalize)
- [Welford](#welford)
- [GroupNorm](#groupnorm)

---

## LayerNorm

Layer Normalization: 标准化 `dst[i] = (src[i] - mean) / sqrt(var + eps) * gamma + beta`

### 主函数

```cpp
void LayerNorm(LocalTensor<DT> dst, LocalTensor<DT> src,
             LocalTensor<DT> mean, LocalTensor<DT> var,
             LocalTensor<DT> gamma, LocalTensor<DT> beta,
             LocalTensor<DT> tmp, uint32_t totalLength, DT eps);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src | LocalTensor | 输入张量 |
| mean | LocalTensor | 均值输出/输入 |
| var | LocalTensor | 方差输出/输入 |
| gamma | LocalTensor | 缩放参数 |
| beta | LocalTensor | 偏移参数 |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 总数据长度 |
| eps | DT | 防止除零的常数 |

### 梯度函数

```cpp
void LayerNormGrad(LocalTensor<DT> grad, LocalTensor<DT> src,
                  LocalTensor<DT> mean, LocalTensor<DT> var,
                  LocalTensor<DT> output, LocalTensor<DT> gamma,
                  LocalTensor<DT> tmp, uint32_t totalLength, DT eps);

void LayerNormGradBeta(LocalTensor<DT> grad, LocalTensor<DT> src,
                     LocalTensor<DT> mean, LocalTensor<DT> var,
                     LocalTensor<DT> output, LocalTensor<DT> tmp,
                     uint32_t totalLength, DT eps);
```

### Tiling接口

```cpp
uint32_t LayerNormGetTmpBufferSize(uint32_t length);
uint32_t LayerNormGradGetTmpBufferSize(uint32_t length);
uint32_t LayerNormGradBetaGetTmpBufferSize(uint32_t length);
```

---

## RmsNorm

RMS Normalization: `dst[i] = src[i] / sqrt(mean(square(src)) * gamma`

### 主函数

```cpp
void RmsNorm(LocalTensor<DT> dst, LocalTensor<DT> src,
           LocalTensor<DT> gamma, LocalTensor<DT> tmp,
           uint32_t totalLength, DT eps);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src | LocalTensor | 输入张量 |
| gamma | LocalTensor | 缩放参数 |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 总数据长度 |
| eps | DT | 防止除零的常数 |

### Tiling接口

```cpp
uint32_t RmsNormGetTmpBufferSize(uint32_t length);
```

---

## BatchNorm

Batch Normalization

### 主函数

```cpp
void BatchNorm(LocalTensor<DT> dst, LocalTensor<DT> src,
              LocalTensor<DT> mean, LocalTensor<DT> var,
              LocalTensor<DT> gamma, LocalTensor<DT> beta,
              LocalTensor<DT> tmp, uint32_t batch, uint32_t channel,
              uint32_t height, uint32_t width, DT eps);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src | LocalTensor | 输入张量 |
| mean | LocalTensor | 均值 |
| var | LocalTensor | 方差 |
| gamma | LocalTensor | 缩放参数 |
| beta | LocalTensor | 偏移参数 |
| tmp | LocalTensor | 临时缓冲区 |
| batch | uint32_t | batch维度 |
| channel | uint32_t | channel维度 |
| height | uint32_t | 高度 |
| width | uint32_t | 宽度 |
| eps | DT | 防止除零的常数 |

### Tiling接口

```cpp
uint32_t BatchNormGetTmpBufferSize(uint32_t batch, uint32_t channel,
                              uint32_t height, uint32_t width);
```

---

## DeepNorm

Deep Normalization: 用于深度网络的归一化

### 主函数

```cpp
void DeepNorm(LocalTensor<DT> dst, LocalTensor<DT> src,
             LocalTensor<DT> mean, LocalTensor<DT> var,
             LocalTensor<DT> gamma, LocalTensor<DT> beta,
             LocalTensor<DT> tmp, uint32_t totalLength, DT eps, DT alpha);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src | LocalTensor | 输入张量 |
| mean | LocalTensor | 均值 |
| var | LocalTensor | 方差 |
| gamma | LocalTensor | 缩放参数 |
| beta | LocalTensor | 偏移参数 |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 总数据长度 |
| eps | DT | 防止除零的常数 |
| alpha | DT | DeepNorm缩放因子 |

### Tiling接口

```cpp
uint32_t DeepNormGetTmpBufferSize(uint32_t length);
```

---

## Normalize

通用归一化

### 主函数

```cpp
void Normalize(LocalTensor<DT> dst, LocalTensor<DT> src,
               LocalTensor<DT> norm, LocalTensor<DT> tmp,
               uint32_t length, uint32_t axisNum, DT eps);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src | LocalTensor | 输入张量 |
| norm | LocalTensor | 归一化参数 |
| tmp | LocalTensor | 临时缓冲区 |
| length | uint32_t | 数据长度 |
| axisNum | uint32_t | 归一化的轴数 |
| eps | DT | 防止除零的常数 |

### Tiling接口

```cpp
uint32_t NormalizeGetTmpBufferSize(uint32_t length, uint32_t axisNum);
```

---

## Welford

Welford算法 - 用于在线计算均值和方差

### 主函数

```cpp
void WelfordUpdate(LocalTensor<DT> src, LocalTensor<DT> mean,
                  LocalTensor<DT> var, LocalTensor<DT> count,
                  uint32_t batch, uint32_t length);

void WelfordFinalize(LocalTensor<DT> mean, LocalTensor<DT> var,
                    LocalTensor<DT> count, DT eps);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| src | LocalTensor | 输入数据 |
| mean | LocalTensor | 均值 |
| var | LocalTensor | 方差 |
| count | LocalTensor | 计数 |
| batch | uint32_t | batch大小 |
| length | uint32_t | 数据长度 |
| eps | DT | 防止除零的常数 |

### Tiling接口

```cpp
uint32_t WelfordUpdateGetTmpBufferSize(uint32_t batch, uint32_t length);
uint32_t WelfordFinalizeGetTmpBufferSize(uint32_t batch);
```

---

## GroupNorm

Group Normalization: 在channel维度上分组进行归一化

### 主函数

```cpp
void GroupNorm(LocalTensor<DT> dst, LocalTensor<DT> src,
              LocalTensor<DT> mean, LocalTensor<DT> var,
              LocalTensor<DT> gamma, LocalTensor<DT> beta,
              LocalTensor<DT> tmp, uint32_t batch, uint32_t group,
              uint32_t channel, uint32_t height, uint32_t width, DT eps);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src | LocalTensor | 输入张量 |
| mean | LocalTensor | 均值 |
| var | LocalTensor | 方差 |
| gamma | LocalTensor | 缩放参数 |
| beta | LocalTensor | 偏移参数 |
| tmp | LocalTensor | 临时缓冲区 |
| batch | uint32_t | batch大小 |
| group | uint32_t | 分组数 |
| channel | uint32_t | channel数量 |
| height | uint32_t | 高度 |
| width | uint32_t | 宽度 |
| eps | DT | 防止除零的常数 |

### Tiling接口

```cpp
uint32_t GroupNormGetTmpBufferSize(uint32_t batch, uint32_t group,
                                uint32_t channel, uint32_t height, uint32_t width);
```

---

## 使用示例

### LayerNorm

```cpp
void LayerNormExample() {
    uint32_t tmpSize = GetLayerNormTmpBufferSize(length);
    LocalTensor<float> tmp = que.AllocTensor();
    LocalTensor<float> mean = que.AllocTensor();
    LocalTensor<float> var = que.AllocTensor();
    
    float eps = 1e-5f;
    LayerNorm(output, input, mean, var, gamma, beta, tmp, length, eps);
    
    que.FreeTensor(tmp);
    que.FreeTensor(mean);
    que.FreeTensor(var);
}
```

### LayerNormGrad

```cpp
void LayerNormGradExample() {
    uint32_t tmpSize = GetLayerNormGradTmpBufferSize(length);
    LocalTensor<float> tmp = que.AllocTensor();
    
    float eps = 1e-5f;
    LayerNormGrad(gradInput, input, mean, var, output, gamma, tmp, length, eps);
    
    que.FreeTensor(tmp);
}
```

### RmsNorm

```cpp
void RmsNormExample() {
    uint32_t tmpSize = GetRmsNormTmpBufferSize(length);
    LocalTensor<float> tmp = que.AllocTensor();
    
    float eps = 1e-5f;
    RmsNorm(output, input, gamma, tmp, length, eps);
    
    que.FreeTensor(tmp);
}
```

### GroupNorm

```cpp
void GroupNormExample() {
    uint32_t tmpSize = GetGroupNormTmpBufferSize(batch, groupNum, channels, height, width);
    LocalTensor<float> tmp = que.AllocTensor();
    LocalTensor<float> mean = que.AllocTensor();
    LocalTensor<float> var = que.AllocTensor();
    
    float eps = 1e-5f;
    GroupNorm(output, input, mean, var, gamma, beta, tmp, 
            batch, groupNum, channels, height, width, eps);
    
    que.FreeTensor(tmp);
    que.FreeTensor(mean);
    que.FreeTensor(var);
}
```