# 高阶API - 归约操作

> Ascend C 归约函数库

## 目录

- [求和与均值](#求和与均值)
- [异或归约](#异或归约)
- [其他归约](#其他归约)

---

## 求和与均值

### Sum

求和: `dst = sum(src)`

```cpp
void Sum(LocalTensor<DT> dst, LocalTensor<DT> src,
        LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetSumMaxMinTmpSize();
```

### Mean

均值: `dst = mean(src)`

```cpp
void Mean(LocalTensor<DT> dst, LocalTensor<DT> src,
         LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetMeanMaxMinTmpSize();
uint32_t GetMeanTmpBufferFactorSize();
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 (标量) |
| src | LocalTensor | 输入张量 |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 数据长度 |

---

## 异或归约

### ReduceXorSum

异或归约求和。

```cpp
void ReduceXorSum(LocalTensor<DT> dst, LocalTensor<DT> src,
                  LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetReduceXorSumMaxMinTmpSize();
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出 |
| src | LocalTensor | 输入 |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 长度 |

---

## 其他归约

### ReduceSum

求和归约。

```cpp
void ReduceSum(LocalTensor<DT> dst, LocalTensor<DT> src,
              LocalTensor<DT> tmp, uint32_t srcRepStride,
              uint32_t repeatTimes);

uint32_t GetReduceSumMaxMinTmpSize();
```

### ReduceMean

均值归约。

```cpp
void ReduceMean(LocalTensor<DT> dst, LocalTensor<DT> src,
               LocalTensor<DT> tmp, uint32_t srcRepStride,
               uint32_t repeatTimes);

uint32_t GetReduceMeanMaxMinTmpSize();
```

### ReduceMax

最大值归约。

```cpp
void ReduceMax(LocalTensor<DT> dst, LocalTensor<DT> src,
              LocalTensor<DT> tmp, uint32_t srcRepStride,
              uint32_t repeatTimes);

uint32_t GetReduceMaxMaxMinTmpSize();
```

### ReduceMin

最小值归约。

```cpp
void ReduceMin(LocalTensor<DT> dst, LocalTensor<DT> src,
              LocalTensor<DT> tmp, uint32_t srcRepStride,
              uint32_t repeatTimes);

uint32_t GetReduceMinMaxMinTmpSize();
```

### ReduceAny

任意归约 (逻辑或)。

```cpp
void ReduceAny(LocalTensor<DT> dst, LocalTensor<DT> src,
               LocalTensor<DT> tmp, uint32_t srcRepStride,
               uint32_t repeatTimes);

uint32_t GetReduceAnyMaxMinTmpSize();
```

### ReduceAll

全部归约 (逻辑与)。

```cpp
void ReduceAll(LocalTensor<DT> dst, LocalTensor<DT> src,
               LocalTensor<DT> tmp, uint32_t srcRepStride,
               uint32_t repeatTimes);

uint32_t GetReduceAllMaxMinTmpSize();
```

### ReduceProd

乘积归约。

```cpp
void ReduceProd(LocalTensor<DT> dst, LocalTensor<DT> src,
               LocalTensor<DT> tmp, uint32_t srcRepStride,
               uint32_t repeatTimes);

uint32_t GetReduceProdMaxMinTmpSize();
```

---

## 使用示例

```cpp
void ReduceExample() {
    uint32_t tmpSize = GetSumMaxMinTmpSize();
    LocalTensor<float> tmp = que.AllocTensor();
    LocalTensor<float> result = que.AllocTensor();
    
    Sum(result, input, tmp, 1024);
    
    que.FreeTensor(tmp);
    que.FreeTensor(result);
}

void ReduceMaxExample() {
    uint32_t tmpSize = GetReduceMaxMaxMinTmpSize();
    LocalTensor<float> tmp = que.AllocTensor();
    LocalTensor<float> result = que.AllocTensor();
    
    ReduceMax(result, input, tmp, 256, 4);
    
    que.FreeTensor(tmp);
    que.FreeTensor(result);
}
```