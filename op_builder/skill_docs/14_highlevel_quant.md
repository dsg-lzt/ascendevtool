# 高阶API - 量化反量化

> Ascend C 量化与反量化函数

## 目录

- [AscendQuant](#ascendquant)
- [AscendDequant](#ascenddequant)
- [AscendAntiQuant](#ascendantiquant)

---

## AscendQuant

Ascend量化: 将浮点数转换为定点数。

### 主函数

```cpp
void AscendQuant(LocalTensor<DT> dst, LocalTensor<DT> src,
                LocalTensor<DT> tmp, uint32_t totalLength,
                QuantParam quantParam);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 量化后的输出 |
| src | LocalTensor | 浮点输入 |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 数据长度 |
| quantParam | QuantParam | 量化参数 |

### QuantParam 结构

```cpp
struct QuantParam {
    float scale;           // 量化尺度
    float offset;          // 量化偏移
    int32_t roundMode;    // 舍入模式
    int32_t method;       // 量化方法
};
```

### 获取临时缓冲区大小

```cpp
uint32_t GetAscendQuantMaxMinTmpSize();
```

---

## AscendDequant

Ascend反量化: 将定点数转换为浮点数。

### 主函数

```cpp
void AscendDequant(LocalTensor<DT> dst, LocalTensor<DT> src,
                  LocalTensor<DT> tmp, uint32_t totalLength,
                  DequantParam dequantParam);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 反量化后的输出 |
| src | LocalTensor | 定点输入 |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 数据长度 |
| dequantParam | DequantParam | 反量化参数 |

### DequantParam 结构

```cpp
struct DequantParam {
    float scale;         // 反量化尺度
    int32_t roundMode;   // 舍入模式
    int32_t method;     // 反量化方法
};
```

### 获取临时缓冲区大小

```cpp
uint32_t GetAscendDequantMaxMinTmpSize();
```

---

## AscendAntiQuant

Ascend反量化 (与AscendDequant相同)。

### 主函数

```cpp
void AscendAntiQuant(LocalTensor<DT> dst, LocalTensor<DT> src,
                    LocalTensor<DT> tmp, uint32_t totalLength,
                    AntiQuantParam antiQuantParam);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出 |
| src | LocalTensor | 输入 |
| tmp | LocalTensor | 临时缓冲区 |
| totalLength | uint32_t | 数据长度 |
| antiQuantParam | AntiQuantParam | 反量化参数 |

### AntiQuantParam 结构

```cpp
struct AntiQuantParam {
    float scale;         // 尺度
    int32_t roundMode;  // 舍入模式
};
```

### 获取临时缓冲区大小

```cpp
uint32_t GetAscendAntiQuantMaxMinTmpSize();
```

---

## 使用示例

```cpp
// 量化
void QuantExample() {
    QuantParam param;
    param.scale = 0.01f;
    param.offset = 0.0f;
    param.roundMode = 0;
    
    uint32_t tmpSize = GetAscendQuantMaxMinTmpSize();
    LocalTensor<float> tmp = que.AllocTensor();
    
    AscendQuant(quantOutput, floatInput, tmp, length, param);
    
    que.FreeTensor(tmp);
}

// 反量化
void DequantExample() {
    DequantParam param;
    param.scale = 100.0f;
    param.roundMode = 0;
    
    uint32_t tmpSize = GetAscendDequantMaxMinTmpSize();
    LocalTensor<float> tmp = que.AllocTensor();
    
    AscendDequant(floatOutput, quantInput, tmp, length, param);
    
    que.FreeTensor(tmp);
}
```