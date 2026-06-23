# 标量与矢量计算

> Ascend C 基础计算API

## 目录

- [标量计算](#标量计算)
- [矢量计算 - 基础算术](#基础算术)
- [矢量计算 - 逻辑运算](#逻辑运算)
- [矢量计算 - 复合计算](#复合计算)
- [矢量计算 - 比较指令](#比较指令)
- [矢量计算 - 选择指令](#选择指令)
- [矢量计算 - 精度转换](#精度转换)
- [矢量计算 - 归约指令](#归约指令)
- [矢量计算 - 数据转换](#数据转换)
- [矢量计算 - 数据填充](#数据填充)

---

## 标量计算

### ScalarGetCountOfValue

统计指定值的个数。

```cpp
uint32_t ScalarGetCountOfValue(DT value);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| value | DT | 要统计的值 |
| **返回值** | uint32_t | 值的个数 |

### ScalarCountLeadingZero

计算前导零的个数。

```cpp
uint32_t ScalarCountLeadingZero(DT value);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| value | DT | 输入值 |
| **返回值** | uint32_t | 前导零个数 |

### ScalarCast

标量类型转换。

```cpp
template<typename DST_T, typename SRC_T>
DST_T ScalarCast(SRC_T value);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| value | SRC_T | 源类型值 |
| **返回值** | DST_T | 目标类型值 |

### CountBitsCntSameAsSignBit

统计与符号位相同的位数。

```cpp
uint32_t CountBitsCntSameAsSignBit(DT value);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| value | DT | 输入值 |
| **返回值** | uint32_t | 与符号位相同的位数 |

### ScalarGetSFFValue

获取SFF (Special Function) 值。

```cpp
float ScalarGetSFFValue(int index);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| index | int | SFF表索引 |
| **返回值** | float | SFF值 |

### ToBfloat16

转换为 bfloat16 格式。

```cpp
half ToBfloat16(float value);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| value | float | 输入浮点数 |
| **返回值** | half | bfloat16值 |

### ToFloat

从 bfloat16 转换为 float。

```cpp
float ToFloat(half value);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| value | half | bfloat16值 |
| **返回值** | float | 浮点数 |

---

## 基础算术

### Exp

指数运算: `dst[i] = exp(src[i])`

```cpp
void Exp(LocalTensor<DT> dst, LocalTensor<DT> src, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量，存储结果 |
| src | LocalTensor | 源张量，输入数据 |
| repeatTimes | uint32_t | 重复次数 |

### Ln

自然对数: `dst[i] = ln(src[i])`

```cpp
void Ln(LocalTensor<DT> dst, LocalTensor<DT> src, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| repeatTimes | uint32_t | 重复次数 |

### Abs

绝对值: `dst[i] = abs(src[i])`

```cpp
void Abs(LocalTensor<DT> dst, LocalTensor<DT> src, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| repeatTimes | uint32_t | 重复次数 |

### Reciprocal

倒数: `dst[i] = 1.0 / src[i]`

```cpp
void Reciprocal(LocalTensor<DT> dst, LocalTensor<DT> src, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| repeatTimes | uint32_t | 重复次数 |

### Sqrt

平方根: `dst[i] = sqrt(src[i])`

```cpp
void Sqrt(LocalTensor<DT> dst, LocalTensor<DT> src, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| repeatTimes | uint32_t | 重复次数 |

### Rsqrt

平方根倒数: `dst[i] = 1.0 / sqrt(src[i])`

```cpp
void Rsqrt(LocalTensor<DT> dst, LocalTensor<DT> src, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| repeatTimes | uint32_t | 重复次数 |

### Relu

ReLU激活: `dst[i] = max(0, src[i])`

```cpp
void Relu(LocalTensor<DT> dst, LocalTensor<DT> src, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| repeatTimes | uint32_t | 重复次数 |

### VectorPadding (ISASI)

矢量填充。

```cpp
void VectorPadding(LocalTensor<DT> dst, LocalTensor<DT> src, 
                  uint32_t srcSize, uint32_t padValue, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| srcSize | uint32_t | 源数据大小 |
| padValue | uint32_t | 填充值 |
| repeatTimes | uint32_t | 重复次数 |

### Add

加法: `dst[i] = src0[i] + src1[i]`

```cpp
void Add(LocalTensor<DT> dst, LocalTensor<DT> src0, 
        LocalTensor<DT> src1, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src0 | LocalTensor | 源张量0 |
| src1 | LocalTensor | 源张量1 |
| repeatTimes | uint32_t | 重复次数 |

### Sub

减法: `dst[i] = src0[i] - src1[i]`

```cpp
void Sub(LocalTensor<DT> dst, LocalTensor<DT> src0, 
        LocalTensor<DT> src1, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src0 | LocalTensor | 被减数 |
| src1 | LocalTensor | 减数 |
| repeatTimes | uint32_t | 重复次数 |

### Mul

乘法: `dst[i] = src0[i] * src1[i]`

```cpp
void Mul(LocalTensor<DT> dst, LocalTensor<DT> src0, 
        LocalTensor<DT> src1, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src0 | LocalTensor | 源张量0 |
| src1 | LocalTensor | 源张量1 |
| repeatTimes | uint32_t | 重复次数 |

### Div

除法: `dst[i] = src0[i] / src1[i]`

```cpp
void Div(LocalTensor<DT> dst, LocalTensor<DT> src0, 
        LocalTensor<DT> src1, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src0 | LocalTensor | 被除数 |
| src1 | LocalTensor | 除数 |
| repeatTimes | uint32_t | 重复次数 |

### Max

逐元素取最大值: `dst[i] = max(src0[i], src1[i])`

```cpp
void Max(LocalTensor<DT> dst, LocalTensor<DT> src0, 
        LocalTensor<DT> src1, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src0 | LocalTensor | 源张量0 |
| src1 | LocalTensor | 源张量1 |
| repeatTimes | uint32_t | 重复次数 |

### Min

逐元素取最小值: `dst[i] = min(src0[i], src1[i])`

```cpp
void Min(LocalTensor<DT> dst, LocalTensor<DT> src0, 
        LocalTensor<DT> src1, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src0 | LocalTensor | 源张量0 |
| src1 | LocalTensor | 源张量1 |
| repeatTimes | uint32_t | 重复次数 |

### Adds

标量加法: `dst[i] = src[i] + scalar`

```cpp
void Adds(LocalTensor<DT> dst, LocalTensor<DT> src, 
         DT scalar, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| scalar | DT | 标量值 |
| repeatTimes | uint32_t | 重复次数 |

### Muls

标量乘法: `dst[i] = src[i] * scalar`

```cpp
void Muls(LocalTensor<DT> dst, LocalTensor<DT> src, 
         DT scalar, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| scalar | DT | 标量值 |
| repeatTimes | uint32_t | 重复次数 |

### Maxs

标量最大值: `dst[i] = max(src[i], scalar)`

```cpp
void Maxs(LocalTensor<DT> dst, LocalTensor<DT> src, 
         DT scalar, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| scalar | DT | 标量值 |
| repeatTimes | uint32_t | 重复次数 |

### Mins

标量最小值: `dst[i] = min(src[i], scalar)`

```cpp
void Mins(LocalTensor<DT> dst, LocalTensor<DT> src, 
         DT scalar, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| scalar | DT | 标量值 |
| repeatTimes | uint32_t | 重复次数 |

### LeakyRelu

Leaky ReLU: `dst[i] = src[i] > 0 ? src[i] : src[i] * alpha`

```cpp
void LeakyRelu(LocalTensor<DT> dst, LocalTensor<DT> src, 
              DT alpha, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| alpha | DT | 负斜率 |
| repeatTimes | uint32_t | 重复次数 |

### Axpy

矢量乘加: `dst[i] = src0[i] * scalar + src1[i]`

```cpp
void Axpy(LocalTensor<DT> dst, LocalTensor<DT> src0, 
         LocalTensor<DT> src1, DT scalar, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src0 | LocalTensor | 源张量0 |
| src1 | LocalTensor | 源张量1 |
| scalar | DT | 标量系数 |
| repeatTimes | uint32_t | 重复次数 |

---

## 逻辑运算

### Not

逻辑非: `dst[i] = !src[i]`

```cpp
void Not(LocalTensor<DT> dst, LocalTensor<DT> src, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| repeatTimes | uint32_t | 重复次数 |

### And

逻辑与: `dst[i] = src0[i] & src1[i]`

```cpp
void And(LocalTensor<DT> dst, LocalTensor<DT> src0, 
        LocalTensor<DT> src1, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src0 | LocalTensor | 源张量0 |
| src1 | LocalTensor | 源张量1 |
| repeatTimes | uint32_t | 重复次数 |

### Or

逻辑或: `dst[i] = src0[i] | src1[i]`

```cpp
void Or(LocalTensor<DT> dst, LocalTensor<DT> src0, 
       LocalTensor<DT> src1, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src0 | LocalTensor | 源张量0 |
| src1 | LocalTensor | 源张量1 |
| repeatTimes | uint32_t | 重复次数 |

### ShiftLeft

左移: `dst[i] = src[i] << shift`

```cpp
void ShiftLeft(LocalTensor<DT> dst, LocalTensor<DT> src, 
              uint32_t shift, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| shift | uint32_t | 移位量 |
| repeatTimes | uint32_t | 重复次数 |

### ShiftRight

右移: `dst[i] = src[i] >> shift`

```cpp
void ShiftRight(LocalTensor<DT> dst, LocalTensor<DT> src, 
               uint32_t shift, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| shift | uint32_t | 移位量 |
| repeatTimes | uint32_t | 重复次数 |

---

## 复合计算

### AddRelu

加法 + ReLU: `dst[i] = max(0, src0[i] + src1[i])`

```cpp
void AddRelu(LocalTensor<DT> dst, LocalTensor<DT> src0, 
            LocalTensor<DT> src1, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src0 | LocalTensor | 源张量0 |
| src1 | LocalTensor | 源张量1 |
| repeatTimes | uint32_t | 重复次数 |

### AddReluCast

加法 + ReLU + 类型转换

```cpp
template<typename DST_T>
void AddReluCast(LocalTensor<DST_T> dst, LocalTensor<DT> src0, 
                LocalTensor<DT> src1, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src0 | LocalTensor | 源张量0 |
| src1 | LocalTensor | 源张量1 |
| repeatTimes | uint32_t | 重复次数 |

### AddDeqRelu

加法 + 反量化 + ReLU

```cpp
void AddDeqRelu(LocalTensor<DT> dst, LocalTensor<DT> src0, 
               LocalTensor<DT> src1, DT deqScale, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src0 | LocalTensor | 源张量0 |
| src1 | LocalTensor | 源张量1 |
| deqScale | DT | 反量化尺度 |
| repeatTimes | uint32_t | 重复次数 |

### SubRelu

减法 + ReLU: `dst[i] = max(0, src0[i] - src1[i])`

```cpp
void SubRelu(LocalTensor<DT> dst, LocalTensor<DT> src0, 
            LocalTensor<DT> src1, uint32_t repeatTimes);
```

### SubReluCast

减法 + ReLU + 类型转换

```cpp
template<typename DST_T>
void SubReluCast(LocalTensor<DST_T> dst, LocalTensor<DT> src0, 
                LocalTensor<DT> src1, uint32_t repeatTimes);
```

### MulAddDst

乘加: `dst[i] = src0[i] * src1[i] + src2[i]`

```cpp
void MulAddDst(LocalTensor<DT> dst, LocalTensor<DT> src0, 
              LocalTensor<DT> src1, LocalTensor<DT> src2, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src0 | LocalTensor | 乘数0 |
| src1 | LocalTensor | 乘数1 |
| src2 | LocalTensor | 加数 |
| repeatTimes | uint32_t | 重复次数 |

### MulCast

乘法 + 类型转换

```cpp
template<typename DST_T>
void MulCast(LocalTensor<DST_T> dst, LocalTensor<DT> src0, 
            LocalTensor<DT> src1, uint32_t repeatTimes);
```

### FusedMulAdd

融合乘加: `dst[i] = src0[i] * src1[i] + src2[i]` (融合实现)

```cpp
void FusedMulAdd(LocalTensor<DT> dst, LocalTensor<DT> src0, 
                LocalTensor<DT> src1, LocalTensor<DT> src2, uint32_t repeatTimes);
```

### FusedMulAddRelu

融合乘加 + ReLU: `dst[i] = max(0, src0[i] * src1[i] + src2[i])`

```cpp
void FusedMulAddRelu(LocalTensor<DT> dst, LocalTensor<DT> src0, 
                     LocalTensor<DT> src1, LocalTensor<DT> src2, uint32_t repeatTimes);
```

---

## 比较指令

### Compare

逐元素比较: `dst[i] = (src0[i] cmp src1[i]) ? 1 : 0`

```cpp
enum class CompareSelectMode {LT, LE, GT, GE, EQ, NE};

void Compare(LocalTensor<DT> dst, LocalTensor<DT> src0, 
           LocalTensor<DT> src1, CompareSelectMode mode, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 (uint8_t) |
| src0 | LocalTensor | 源张量0 |
| src1 | LocalTensor | 源张量1 |
| mode | CompareSelectMode | 比较模式 |
| repeatTimes | uint32_t | 重复次数 |

### CompareScalar

与标量比较

```cpp
void CompareScalar(LocalTensor<DT> dst, LocalTensor<DT> src, 
                  DT scalar, CompareSelectMode mode, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| scalar | DT | 标量值 |
| mode | CompareSelectMode | 比较模式 |
| repeatTimes | uint32_t | 重复次数 |

### GetCmpMask (ISASI)

获取比较掩码

```cpp
uint64_t GetCmpMask();
```

- **返回值**: 比较掩码

### SetCmpMask (ISASI)

设置比较掩码

```cpp
void SetCmpMask(uint64_t mask);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| mask | uint64_t | 掩码值 |

---

## 选择指令

### Select

条件选择: `dst[i] = mask[i] ? src1[i] : src0[i]`

```cpp
void Select(LocalTensor<DT> dst, LocalTensor<DT> src0, 
           LocalTensor<DT> src1, LocalTensor<uint8_t> mask, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src0 | LocalTensor | 条件为false时的值 |
| src1 | LocalTensor | 条件为true时的值 |
| mask | LocalTensor | 条件掩码 |
| repeatTimes | uint32_t | 重复次数 |

### GatherMask

聚集掩码操作

```cpp
void GatherMask(LocalTensor<DT> dst, LocalTensor<DT> src, 
               LocalTensor<uint8_t> mask, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| mask | LocalTensor | 掩码张量 |
| repeatTimes | uint32_t | 重复次数 |

---

## 精度转换

### Cast

类型转换: `dst[i] = (DST_T)src[i]`

```cpp
template<typename DST_T>
void Cast(LocalTensor<DST_T> dst, LocalTensor<DT> src, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| repeatTimes | uint32_t | 重复次数 |

### CastDeq

类型转换 + 反量化

```cpp
void CastDeq(LocalTensor<DT> dst, LocalTensor<DT> src, 
            DT deqScale, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| deqScale | DT | 反量化尺度 |
| repeatTimes | uint32_t | 重复次数 |

---

## 归约指令

### ReduceMax

最大值归约

```cpp
void ReduceMax(LocalTensor<DT> dst, LocalTensor<DT> src, 
              uint32_t repeatTimes, uint32_t srcStride);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 (存储归约结果) |
| src | LocalTensor | 源张量 |
| repeatTimes | uint32_t | 重复次数 |
| srcStride | uint32_t | 源张量步长 |

### ReduceMin

最小值归约

```cpp
void ReduceMin(LocalTensor<DT> dst, LocalTensor<DT> src, 
              uint32_t repeatTimes, uint32_t srcStride);
```

### ReduceSum

求和归约

```cpp
void ReduceSum(LocalTensor<DT> dst, LocalTensor<DT> src, 
              uint32_t repeatTimes, uint32_t srcStride);
```

### WholeReduceMax

全局最大值归约 (跨所有block)

```cpp
void WholeReduceMax(LocalTensor<DT> dst, LocalTensor<DT> src, 
                   uint32_t repeatTimes);
```

### WholeReduceMin

全局最小值归约

```cpp
void WholeReduceMin(LocalTensor<DT> dst, LocalTensor<DT> src, 
                   uint32_t repeatTimes);
```

### WholeReduceSum

全局求和归约

```cpp
void WholeReduceSum(LocalTensor<DT> dst, LocalTensor<DT> src, 
                   uint32_t repeatTimes);
```

### BlockReduceMax

分块最大值归约

```cpp
void BlockReduceMax(LocalTensor<DT> dst, LocalTensor<DT> src, 
                   uint32_t repeatTimes);
```

### BlockReduceMin

分块最小值归约

```cpp
void BlockReduceMin(LocalTensor<DT> dst, LocalTensor<DT> src, 
                   uint32_t repeatTimes);
```

### BlockReduceSum

分块求和归约

```cpp
void BlockReduceSum(LocalTensor<DT> dst, LocalTensor<DT> src, 
                   uint32_t repeatTimes);
```

---

## 数据转换

### Transpose

转置: `dst[j][i] = src[i][j]`

```cpp
void Transpose(LocalTensor<DT> dst, LocalTensor<DT> src, 
              uint32_t rowNum, uint32_t colNum);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| rowNum | uint32_t | 行数 |
| colNum | uint32_t | 列数 |

### TransDataTo5HD

转换为5HD格式

```cpp
void TransDataTo5HD(LocalTensor<DT> dst, LocalTensor<DT> src, 
                   TensorDesc desc);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| desc | TensorDesc | 张量描述符 |

---

## 数据填充

### Duplicate

数据复制/广播: `dst[i] = src[i % N]`

```cpp
void Duplicate(LocalTensor<DT> dst, LocalTensor<DT> src, 
              uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src | LocalTensor | 源张量 |
| repeatTimes | uint32_t | 重复次数 |

### Brcb

广播: 将标量或小张量广播到大张量

```cpp
void Brcb(LocalTensor<DT> dst, LocalTensor<DT> src, 
         uint32_t repeatTimes);
```

### CreateVecIndex

创建矢量索引

```cpp
void CreateVecIndex(LocalTensor<DT> dst, uint32_t start, 
                   uint32_t step, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| start | uint32_t | 起始值 |
| step | uint32_t | 步长 |
| repeatTimes | uint32_t | 重复次数 |

---

## repeatTimes 说明

`repeatTimes` 控制矢量计算重复执行的次数。每次重复处理的数据量取决于：
- 数据类型 (float32: 32 elements, float16: 64 elements, int8: 128 elements)
- 具体指令

典型计算: `total_elements = repeatTimes * elements_per_repeat`

### 示例

```cpp
// 处理256个float32元素
// float32每repeat处理32个元素
// repeatTimes = 256 / 32 = 8
Exp(dst, src, 8);
```
