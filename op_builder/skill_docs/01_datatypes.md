# 数据类型定义

> Ascend C 算子开发中的核心数据类型

## 目录

- [LocalTensor](#localtensor)
- [GlobalTensor](#globaltensor)
- [ShapeInfo](#shapeinfo)
- [ListTensorDesc](#listtensordesc)
- [TensorDesc](#tensordesc)
- [Coordinate](#coordinate)
- [Layout](#layout)
- [TensorTrait](#tensortrait)
- [LocalMemAllocator](#localmemallocator)
- [RepeatParams](#repeatparams)

---

## LocalTensor

本地张量，用于核内数据存储和处理。

### 构造函数

```cpp
LocalTensor<DT>();
LocalTensor<DT>(const LocalTensor<DT>& other);
```

### 方法列表

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| SetValue | void | 设置张量值 |
| GetValue | DT | 获取张量值 |
| operator[] | LocalTensor | 下标访问 |
| operator() | LocalTensor | 函数调用访问 |
| SetSize | void | 设置张量大小 |
| GetSize | uint32_t | 获取张量大小 |
| SetUserTag | void | 设置用户标签 |
| GetUserTag | uint32_t | 获取用户标签 |
| ReinterpretCast | LocalTensor<T> | 类型重解释转换 |
| GetPhyAddr | uint64_t | 获取物理地址 |
| GetPosition | TPosition | 获取数据位置 |
| GetLength | uint32_t | 获取数据长度 |
| SetShapeInfo | void | 设置形状信息 |
| GetShapeInfo | ShapeInfo | 获取形状信息 |
| SetAddrWithOffset | void | 设置带偏移的地址 |
| SetBufferLen | void | 设置缓冲区长度 |
| ToFile | void | 写入文件 |
| Print | void | 打印张量 |

### 详细方法

#### SetValue

```cpp
void SetValue(uint32_t index, DT value);
```

- **index**: 元素索引
- **value**: 要设置的值

#### GetValue

```cpp
DT GetValue(uint32_t index) const;
```

- **index**: 元素索引
- **返回值**: 指定位置的元素值

#### ReinterpretCast

```cpp
template<typename T>
LocalTensor<T> ReinterpretCast();
```

- **返回值**: 重新解释类型后的LocalTensor

#### GetPhyAddr

```cpp
uint64_t GetPhyAddr() const;
```

- **返回值**: 张量的物理地址

#### SetShapeInfo

```cpp
void SetShapeInfo(const ShapeInfo& shapeInfo);
```

- **shapeInfo**: 形状信息结构

---

## GlobalTensor

全局张量，用于核间数据通信和存储。

### 构造函数

```cpp
GlobalTensor<DT>();
GlobalTensor<DT>(const GlobalTensor<DT>& other);
```

### 方法列表

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| SetGlobalBuffer | void | 设置全局缓冲区 |
| GetPhyAddr | uint64_t | 获取物理地址 |
| GetValue | DT | 获取张量值 |
| operator() | GlobalTensor | 函数调用访问 |
| SetValue | void | 设置张量值 |
| GetSize | uint32_t | 获取张量大小 |
| operator[] | GlobalTensor | 下标访问 |
| SetShapeInfo | void | 设置形状信息 |
| GetShapeInfo | ShapeInfo | 获取形状信息 |
| SetL2CacheHint | void | 设置L2缓存提示 |

### 详细方法

#### SetGlobalBuffer

```cpp
void SetGlobalBuffer(const void* buffer, uint32_t size);
```

- **buffer**: 全局缓冲区指针
- **size**: 缓冲区大小(字节)

#### GetPhyAddr

```cpp
uint64_t GetPhyAddr() const;
```

- **返回值**: 物理地址

#### GetValue

```cpp
DT GetValue(uint32_t index) const;
```

- **index**: 索引
- **返回值**: 元素值

#### SetValue

```cpp
void SetValue(uint32_t index, DT value);
```

- **index**: 索引
- **value**: 要设置的值

#### SetL2CacheHint

```cpp
void SetL2CacheHint(uint32_t hint);
```

- **hint**: 缓存提示值

---

## ShapeInfo

形状信息数据结构，用于描述张量的维度信息。

### 定义

```cpp
struct ShapeInfo {
    uint32_t dimNum;      // 维度数量
    uint32_t shape[8];     // 各维度大小
};
```

### 成员

| 成员 | 类型 | 描述 |
|------|------|------|
| dimNum | uint32_t | 维度数量 |
| shape | uint32_t[8] | 各维度大小 |

---

## ListTensorDesc

张量描述符列表，用于批量管理张量描述信息。

### 主要方法

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| GetDim | uint32_t | 获取维度 |
| GetIndex | uint32_t | 获取索引 |
| GetShape | uint32_t* | 获取形状 |
| GetDataPtr | void* | 获取数据指针 |

---

## TensorDesc

张量描述符，用于描述张量的属性。

### 构造函数

```cpp
TensorDesc();
TensorDesc(const TensorDesc& other);
~TensorDesc();
```

### 方法列表

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| SetShapeAddr | void | 设置形状地址 |
| GetDim | uint32_t | 获取维度 |
| GetIndex | uint32_t | 获取索引 |
| GetShape | uint32_t* | 获取形状 |
| GetDataPtr | void* | 获取数据指针 |
| GetDataObj | void* | 获取数据对象 |

### 详细方法

#### SetShapeAddr

```cpp
void SetShapeAddr(const void* shapeAddr);
```

- **shapeAddr**: 形状数据地址

#### GetShape

```cpp
uint32_t* GetShape() const;
```

- **返回值**: 形状数组指针

#### GetDataPtr

```cpp
void* GetDataPtr(TPosition position) const;
```

- **position**: 数据位置
- **返回值**: 数据指针

---

## Coordinate

坐标操作，用于多维索引计算。

### 静态方法

#### MakeCoord

```cpp
static Coordinate MakeCoord(uint32_t c0);
static Coordinate MakeCoord(uint32_t c0, uint32_t c1);
static Coordinate MakeCoord(uint32_t c0, uint32_t c1, uint32_t c2);
// ... 支持更多维度
```

- **c0, c1, c2, ...**: 各维度坐标值
- **返回值**: Coordinate对象

#### Crd2Idx

```cpp
static uint32_t Crd2Idx(const Coordinate& coord, const uint32_t* stride);
```

- **coord**: 坐标
- **stride**: 步长数组
- **返回值**: 线性索引

### 使用示例

```cpp
// 创建3D坐标
auto coord = Coordinate::MakeCoord(1, 2, 3);

// 计算索引
uint32_t stride[] = {100, 10, 1};
uint32_t idx = Coordinate::Crd2Idx(coord, stride);  // idx = 123
```

---

## Layout

布局描述，定义张量的内存布局。

### 构造函数

```cpp
Layout();
Layout(const Layout& other);
```

### 静态方法

#### MakeLayout

```cpp
static Layout MakeLayout(const uint32_t* shape, uint32_t dimNum);
```

- **shape**: 形状数组
- **dimNum**: 维度数量
- **返回值**: Layout对象

#### MakeShape

```cpp
static Shape MakeShape(const uint32_t* shape, uint32_t dimNum);
```

- **shape**: 形状数组
- **dimNum**: 维度数量
- **返回值**: Shape对象

#### MakeStride

```cpp
static Stride MakeStride(const uint32_t* shape, uint32_t dimNum);
```

- **shape**: 形状数组
- **dimNum**: 维度数量
- **返回值**: Stride对象

### 方法

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| GetShape | Shape | 获取形状 |
| GetStride | Stride | 获取步长 |
| is_layout | bool | 判断是否为有效布局 |

### 使用示例

```cpp
// 创建NC1HWC0布局
uint32_t shape[] = {1, 32, 224, 224, 16};
auto layout = Layout::MakeLayout(shape, 5);
auto stride = layout.GetStride();
```

---

## TensorTrait

张量特征，描述张量的类型特征。

### 模板参数

```cpp
template<typename T, TPosition POS, LayoutType LAYOUT>
struct TensorTrait { ... };
```

- **T**: 数据类型
- **POS**: 位置枚举
- **LAYOUT**: 布局类型

### 静态方法

#### MakeTensorTrait

```cpp
template<typename T, TPosition POS, LayoutType LAYOUT>
static TensorTrait<T, POS, LAYOUT> MakeTensorTrait();
```

- **返回值**: TensorTrait对象

#### is_tensorTrait

```cpp
template<typename T>
static bool is_tensorTrait(const T& obj);
```

- **obj**: 待检查对象
- **返回值**: 是否为TensorTrait

### 方法

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| GetLayout | Layout | 获取布局 |
| SetLayout | void | 设置布局 |

---

## LocalMemAllocator

本地内存分配器，用于管理核内本地内存。

### 构造函数

```cpp
LocalMemAllocator();
LocalMemAllocator(TPosition position);
```

- **position**: 内存位置

### 方法

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| GetCurAddr | void* | 获取当前地址 |
| Alloc | void* | 分配内存 |

#### GetCurAddr

```cpp
void* GetCurAddr() const;
```

- **返回值**: 当前可用地址

#### Alloc

```cpp
void* Alloc(uint32_t size);
```

- **size**: 分配大小(字节)
- **返回值**: 分配的内存地址

### 使用示例

```cpp
LocalMemAllocator allocator(Position::POS_LOCAL);
void* ptr = allocator.Alloc(1024);  // 分配1KB
void* curAddr = allocator.GetCurAddr();  // 获取当前地址
```

---

## RepeatParams

重复参数结构，用于矢量计算的重复控制。

### UnaryRepeatParams

```cpp
struct UnaryRepeatParams {
    uint32_t repeatSize;    // 每次重复处理的数据大小
    uint32_t repeatTimes;   // 重复次数
    uint32_t src0Stride;   // 源张量0步长
};
```

### BinaryRepeatParams

```cpp
struct BinaryRepeatParams {
    uint32_t repeatSize;    // 每次重复处理的数据大小
    uint32_t repeatTimes;   // 重复次数
    uint32_t src0Stride;   // 源张量0步长
    uint32_t src1Stride;   // 源张量1步长
};
```

### 成员说明

| 成员 | 类型 | 描述 |
|------|------|------|
| repeatSize | uint32_t | 单次处理的数据量 |
| repeatTimes | uint32_t | 重复执行次数 |
| src0Stride | uint32_t | 源张量0的步长 |
| src1Stride | uint32_t | 源张量1的步长 |

### 使用示例

```cpp
// 配置: 每次处理256个float, 重复4次
UnaryRepeatParams params;
params.repeatSize = 256;
params.repeatTimes = 4;
params.src0Stride = 256;
```

---

## TPosition 枚举

数据存储位置枚举。

```cpp
enum class TPosition {
    POS_LOCAL,    // 本地内存
    POS_PING,     // Ping缓冲区
    POS_PONG,     // Pong缓冲区
    POS_GLOBAL,   // 全局内存
};
```

---

## 附录：模板参数说明

### 数据类型 (DT)

| 类型 | 描述 |
|------|------|
| half | 半精度浮点 (16-bit) |
| float | 单精度浮点 (32-bit) |
| int8_t | 8位整数 |
| int16_t | 16位整数 |
| int32_t | 32位整数 |
| int64_t | 64位整数 |
| uint8_t | 无符号8位整数 |
| uint16_t | 无符号16位整数 |
| uint32_t | 无符号32位整数 |
| uint64_t | 无符号64位整数 |

### 布局类型 (LayoutType)

| 类型 | 描述 |
|------|------|
| ROW | 行优先 |
| COL | 列优先 |
| ND | N维 |
| NCHW | NCHW格式 |
| NHWC | NHWC格式 |
| NC1HWC0 | 5D格式 |
| ND_NCHW | NCHW转ND |
| NCHW_ND | ND转NCHW |
