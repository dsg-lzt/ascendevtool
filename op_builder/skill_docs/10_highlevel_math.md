# 高阶API - 数学库

> Ascend C 高级数学函数库

## 目录

- [通用模式](#通用模式)
- [三角函数](#三角函数)
- [双曲函数](#双曲函数)
- [指数对数](#指数对数)
- [幂函数](#幂函数)
- [其他数学函数](#其他数学函数)

---

## 通用模式

每个数学库接口包含三个函数：

### 主函数

```cpp
void FuncName(LocalTensor<DT> dst, LocalTensor<DT> src,
             LocalTensor<DT> tmp, uint32_t totalLength);
```

### 获取临时缓冲区大小

```cpp
uint32_t GetFuncNameMaxMinTmpSize();
```

### 获取临时缓冲区因子大小

```cpp
uint32_t GetFuncNameTmpBufferFactorSize();
```

---

## 三角函数

### Tanh

双曲正切: `dst[i] = tanh(src[i])`

```cpp
void Tanh(LocalTensor<DT> dst, LocalTensor<DT> src,
          LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetTanhMaxMinTmpSize();
uint32_t GetTanhTmpBufferFactorSize();
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 总数据长度 |

### Sin

正弦: `dst[i] = sin(src[i])`

```cpp
void Sin(LocalTensor<DT> dst, LocalTensor<DT> src,
        LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetSinMaxMinTmpSize();
uint32_t GetSinTmpBufferFactorSize();
```

### Cos

余弦: `dst[i] = cos(src[i])`

```cpp
void Cos(LocalTensor<DT> dst, LocalTensor<DT> src,
        LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetCosMaxMinTmpSize();
uint32_t GetCosTmpBufferFactorSize();
```

### Asin

反正弦: `dst[i] = asin(src[i])`

```cpp
void Asin(LocalTensor<DT> dst, LocalTensor<DT> src,
         LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetAsinMaxMinTmpSize();
uint32_t GetAsinTmpBufferFactorSize();
```

### Acos

反余弦: `dst[i] = acos(src[i])`

```cpp
void Acos(LocalTensor<DT> dst, LocalTensor<DT> src,
         LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetAcosMaxMinTmpSize();
uint32_t GetAcosTmpBufferFactorSize();
```

### Atan

反正切: `dst[i] = atan(src[i])`

```cpp
void Atan(LocalTensor<DT> dst, LocalTensor<DT> src,
         LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetAtanMaxMinTmpSize();
uint32_t GetAtanTmpBufferFactorSize();
```

### Tan

正切: `dst[i] = tan(src[i])`

```cpp
void Tan(LocalTensor<DT> dst, LocalTensor<DT> src,
        LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetTanMaxMinTmpSize();
uint32_t GetTanTmpBufferFactorSize();
```

---

## 双曲函数

### Sinh

双曲正弦: `dst[i] = sinh(src[i])`

```cpp
void Sinh(LocalTensor<DT> dst, LocalTensor<DT> src,
         LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetSinhMaxMinTmpSize();
uint32_t GetSinhTmpBufferFactorSize();
```

### Cosh

双曲余弦: `dst[i] = cosh(src[i])`

```cpp
void Cosh(LocalTensor<DT> dst, LocalTensor<DT> src,
         LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetCoshMaxMinTmpSize();
uint32_t GetCoshTmpBufferFactorSize();
```

### Atanh

反双曲正切: `dst[i] = atanh(src[i])`

```cpp
void Atanh(LocalTensor<DT> dst, LocalTensor<DT> src,
          LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetAtanhMaxMinTmpSize();
uint32_t GetAtanhTmpBufferFactorSize();
```

### Asinh

反双曲正弦: `dst[i] = asinh(src[i])`

```cpp
void Asinh(LocalTensor<DT> dst, LocalTensor<DT> src,
          LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetAsinhMaxMinTmpSize();
uint32_t GetAsinhTmpBufferFactorSize();
```

### Acosh

反双曲余弦: `dst[i] = acosh(src[i])`

```cpp
void Acosh(LocalTensor<DT> dst, LocalTensor<DT> src,
          LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetAcoshMaxMinTmpSize();
uint32_t GetAcoshTmpBufferFactorSize();
```

---

## 指数对数

### Exp

指数: `dst[i] = exp(src[i])`

```cpp
void Exp(LocalTensor<DT> dst, LocalTensor<DT> src,
        LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetExpMaxMinTmpSize();
uint32_t GetExpTmpBufferFactorSize();
```

### Ln

自然对数: `dst[i] = ln(src[i])`

```cpp
void Ln(LocalTensor<DT> dst, LocalTensor<DT> src,
       LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetLogMaxMinTmpSize();
uint32_t GetLogTmpBufferFactorSize();
```

### Lgamma

对数伽马函数: `dst[i] = lgamma(src[i])`

```cpp
void Lgamma(LocalTensor<DT> dst, LocalTensor<DT> src,
           LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetLgammaMaxMinTmpSize();
uint32_t GetLgammaTmpBufferFactorSize();
```

### Digamma

Digamma函数: `dst[i] = digamma(src[i])`

```cpp
void Digamma(LocalTensor<DT> dst, LocalTensor<DT> src,
            LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetDigammaMaxMinTmpSize();
uint32_t GetDigammaTmpBufferFactorSize();
```

---

## 幂函数

### Power

幂函数: `dst[i] = pow(src[i], power)`

```cpp
void Power(LocalTensor<DT> dst, LocalTensor<DT> src,
          LocalTensor<DT> tmp, DT power, uint32_t totalLength);

uint32_t GetPowerMaxMinTmpSize();
uint32_t GetPowerTmpBufferFactorSize();
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| tmp | LocalTensor | 临时缓冲区 |
| power | DT | 幂次 |
| totalLength | uint32_t | 总数据长度 |

---

## 其他数学函数

### Trunc

截断: `dst[i] = trunc(src[i])`

```cpp
void Trunc(LocalTensor<DT> dst, LocalTensor<DT> src,
          LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetTruncMaxMinTmpSize();
uint32_t GetTruncTmpBufferFactorSize();
```

### Frac

小数部分: `dst[i] = src[i] - floor(src[i])`

```cpp
void Frac(LocalTensor<DT> dst, LocalTensor<DT> src,
         LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetFracMaxMinTmpSize();
uint32_t GetFracTmpBufferFactorSize();
```

### Erf

误差函数: `dst[i] = erf(src[i])`

```cpp
void Erf(LocalTensor<DT> dst, LocalTensor<DT> src,
        LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetErfMaxMinTmpSize();
uint32_t GetErfTmpBufferFactorSize();
```

### Erfc

互补误差函数: `dst[i] = erfc(src[i])`

```cpp
void Erfc(LocalTensor<DT> dst, LocalTensor<DT> src,
         LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetErfcMaxMinTmpSize();
uint32_t GetErfcTmpBufferFactorSize();
```

### Sign

符号函数: `dst[i] = sign(src[i])`

```cpp
void Sign(LocalTensor<DT> dst, LocalTensor<DT> src,
         LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetSignMaxMinTmpSize();
uint32_t GetSignTmpBufferFactorSize();
```

### Floor

向下取整: `dst[i] = floor(src[i])`

```cpp
void Floor(LocalTensor<DT> dst, LocalTensor<DT> src,
          LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetFloorMaxMinTmpSize();
uint32_t GetFloorTmpBufferFactorSize();
```

### Ceil

向上取整: `dst[i] = ceil(src[i])`

```cpp
void Ceil(LocalTensor<DT> dst, LocalTensor<DT> src,
         LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetCeilMaxMinTmpSize();
uint32_t GetCeilTmpBufferFactorSize();
```

### Round

四舍五入: `dst[i] = round(src[i])`

```cpp
void Round(LocalTensor<DT> dst, LocalTensor<DT> src,
           LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetRoundMaxMinTmpSize();
uint32_t GetRoundTmpBufferFactorSize();
```

### Clamp

钳位: `dst[i] = clamp(src[i], min, max)`

```cpp
void ClampMax(LocalTensor<DT> dst, LocalTensor<DT> src,
              LocalTensor<DT> tmp, DT maxValue, uint32_t totalLength);

void ClampMin(LocalTensor<DT> dst, LocalTensor<DT> src,
              LocalTensor<DT> tmp, DT minValue, uint32_t totalLength);

uint32_t GetClampMaxMinTmpSize();
uint32_t GetClampTmpBufferFactorSize();
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| tmp | LocalTensor | 临时缓冲区 |
| maxValue/minValue | DT | 边界值 |
| totalLength | uint32_t | 总数据长度 |

### Axpy

矢量乘加: `dst[i] = src[i] * scalar + dst[i]`

```cpp
void Axpy(LocalTensor<DT> dst, LocalTensor<DT> src,
         LocalTensor<DT> tmp, DT scalar, uint32_t totalLength);

uint32_t GetAxpyMaxMinTmpSize();
uint32_t GetAxpyTmpBufferFactorSize();
```

### Xor

异或: `dst[i] = src0[i] ^ src1[i]`

```cpp
void Xor(LocalTensor<DT> dst, LocalTensor<DT> src0,
         LocalTensor<DT> src1, LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetXorMaxMinTmpSize();
uint32_t GetXorTmpBufferFactorSize();
```

### CumSum

累加和: `dst[i] = sum(src[0:i+1])`

```cpp
void CumSum(LocalTensor<DT> dst, LocalTensor<DT> src,
           LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetCumSumMaxMinTmpSize();
```

### Fmod

取余: `dst[i] = fmod(src0[i], src1[i])`

```cpp
void Fmod(LocalTensor<DT> dst, LocalTensor<DT> src0,
          LocalTensor<DT> src1, LocalTensor<DT> tmp, uint32_t totalLength);

uint32_t GetFmodMaxMinTmpSize();
uint32_t GetFmodTmpBufferFactorSize();
```

---

## 使用示例

```cpp
void MathLibExample() {
    // 1. 获取临时缓冲区大小
    uint32_t tmpSize = GetTanhMaxMinTmpSize();
    
    // 2. 分配临时缓冲区
    LocalTensor<float> tmp = que.AllocTensor();
    
    // 3. 执行Tanh计算
    Tanh(output, input, tmp, 1024);
    
    // 4. 释放临时缓冲区
    que.FreeTensor(tmp);
}
```

### Tiling侧计算临时缓冲区大小

```cpp
// Tiling侧
uint32_t GetTmpBufferSize() {
    uint32_t factor = GetTanhTmpBufferFactorSize();
    uint32_t size = tiling.totalLength * sizeof(float) * factor;
    return size;
}
```
