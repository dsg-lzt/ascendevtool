# Host 侧 API

> Ascend C 算子开发 - Host 侧接口文档

## 目录

- [6 Host API 概述](#6-host-api-概述)
- [6.1 原型注册与管理](#61-原型注册与管理)
- [6.2 Tiling 数据结构注册](#62-tiling-数据结构注册)
- [6.3 平台信息获取](#63-平台信息获取)
- [6.4 单算子 API 执行相关接口](#64-单算子-api-执行相关接口)

---

## 6 Host API 概述

Host 侧 API 主要包括：
- **原型注册与管理 API**：用于 Ascend C 算子原型定义和注册
- **Tiling 数据结构注册 API**：用于 Ascend C 算子 TilingData 数据结构定义和注册
- **平台信息获取 API**：获取硬件平台信息（核数、内存大小等）
- **单算子 API 执行相关接口**：框架能力接口

---

## 6.1 原型注册与管理

### 6.1.1 原型注册接口 (OP_ADD)

注册算子的原型定义。

```cpp
OP_ADD(OP_NAME)
    .Input(x)
    .Output(y)
    .Attr(...)
    .SetInferShape(InferShape);
```

### 6.1.2 OpAICoreConfig 注册接口

```cpp
REGISTER_OP_AICORE_CONFIG(OpName)
    .Input(...)
    .Output(...)
    .SetOpSelectFormat(...)
    .AddConfig(...)
```

### 6.1.3 OpDef - 算子定义

#### Input

定义输入张量。

```cpp
.Input(name, type, format)
.Input(name, type, format, shape)
```

| 参数 | 类型 | 描述 |
|------|------|------|
| name | const char* | 输入名称 |
| type | DataType | 数据类型 |
| format | Format | 数据格式 |
| shape | Shape | 形状 |

#### Output

定义输出张量。

```cpp
.Output(name, type, format)
.Output(name, type, format, shape)
```

#### Attr

定义属性。

```cpp
.Attr<AttrType>(name)
.Attr<AttrType>(name, default_value)
```

### 6.1.4 OpParamDef - 参数定义

#### DataType

```cpp
DataType::DT_FLOAT    // 32位浮点
DataType::DT_FLOAT16 // 16位浮点
DataType::DT_INT8    // 8位整数
DataType::DT_INT32   // 32位整数
DataType::DT_BOOL    // 布尔
DataType::DT_UINT8   // 无符号8位
```

#### Format

```cpp
Format::FORMAT_NCHW   // NCHW格式
Format::FORMAT_NHWC   // NHWC格式
Format::FORMAT_ND     // ND格式
Format::FORMAT_NC1HWC0 // 5D格式
Format::FORMAT_FRACTAL_Z // 分形格式
```

### 6.1.5 OpAttrDef - 属性定义

```cpp
// 标量属性
.Attr<int>("kernel_size")
.Attr<float>("alpha")
.Attr<bool>("normalize")

// 列表属性
.Attr<std::vector<int>>("axis")
```

### 6.1.6 OpAICoreDef - AICore算子定义

#### SetTiling

```cpp
.SetTiling("tiling_name")
```

#### SetCheckSupport

```cpp
.SetCheckSupport(CHECK_MODE mode)
```

#### SetOpSelectFormat

```cpp
.SetOpSelectFormat("{0}")
```

#### AddConfig

```cpp
.AddConfig("runascurrenttik")
```

### 6.1.7 OpAICoreConfig - AICore配置

```cpp
OpAICoreConfig config;
config.Input(0, "x")
      .Output(0, "y")
      .SetOpSelectFormat("{0}")
      .AddConfig("runascurrenttik");
```

### 6.1.8 OpMC2Def - MC2算子定义

用于多卡通信场景的算子定义。

```cpp
OpMC2Def()
    .HcclGroup(groupName)
    .HcclServerType(serverType)
```

---

## 6.2 Tiling 数据结构注册

### 6.2.1 TilingData 结构定义

```cpp
BEGIN_TILING_DATA_DEF(TilingData)
    TILING_DATA_MEMBER(uint32_t, blockDim)
    TILING_DATA_MEMBER(uint32_t, repeatTimes)
    TILING_DATA_MEMBER(uint32_t, srcStride)
END_TILING_DATA_DEF(TilingData)
```

### 6.2.2 TilingData 结构注册

```cpp
REGISTER_TILING_DATA_CLASS(TilingData)
```

### 6.2.3 ContextBuilder - 上下文构建器

#### 构造函数

```cpp
ContextBuilder();
```

#### 基础方法

```cpp
ContextBuilder& Inputs(uint32_t num);
ContextBuilder& Outputs(uint32_t num);
ContextBuilder& NodeIoNum(uint32_t num);
ContextBuilder& SetOpNameType(const std::string& opName, const std::string& opType);
ContextBuilder& IrInstanceNum(uint32_t num);
```

#### 添加输入

```cpp
ContextBuilder& AddInputTd(const std::string& name, 
                          DataType type, 
                          const std::vector<int64_t>& shape);
```

#### 添加输出

```cpp
ContextBuilder& AddOutputTd(const std::string& name,
                            DataType type,
                            const std::vector<int64_t>& shape);
```

#### 添加属性

```cpp
ContextBuilder& AddAttr(const std::string& name, 
                       const std::string& attrType, 
                       const std::string& value);
```

#### 构建上下文

```cpp
KernelRunContext BuildKernelRunContext();
```

### 6.2.4 OpTilingRegistry - Tiling注册器

#### 构造函数

```cpp
OpTilingRegistry(const std::string& opType);
```

#### 获取Tiling函数

```cpp
std::shared_ptr<TilingFunc> GetTilingFunc(const std::string& tilingKey);
```

#### 加载Tiling库

```cpp
void LoadTilingLibrary(const std::string& libPath);
```

### 6.2.5 模板参数定义

```cpp
GET_TPL_TILING_KEY(key)

ASCENDC_TPL_SEL_PARAM(param)
```

### 6.2.6 设备端实现

```cpp
DEVICE_IMPL_OP_OPTILING(func_name)
```

---

## 6.3 平台信息获取

### 6.3.1 PlatformAscendC - 平台信息类

#### 构造函数

```cpp
PlatformAscendC();
PlatformAscendC(uint32_t deviceId);
```

#### 获取AI Core数量

```cpp
uint32_t GetCoreNum();       // 总核心数
uint32_t GetCoreNumAic();   // AIC核心数
uint32_t GetCoreNumAiv();    // AIV核心数
uint32_t GetCoreNumVector(); // Vector核心数
```

#### 获取SoC版本

```cpp
std::string GetSocVersion();
```

#### 计算Tiling Block维度

```cpp
uint32_t CalcTschBlockDim(uint32_t dataSize);
```

#### 获取内存大小

```cpp
uint32_t GetCoreMemSize();     // 核心内存大小
uint32_t GetCoreMemBw();        // 核心内存带宽
uint64_t GetLibApiWorkSpaceSize(); // 库API工作空间大小
```

#### 获取Barrier工作空间

```cpp
uint32_t GetResGroupBarrierWorkSpaceSize(uint32_t groupNum);
```

#### 获取Cube组工作空间

```cpp
uint32_t GetResCubeGroupWorkSpaceSize(uint32_t groupNum);
```

#### 预留本地内存

```cpp
void ReserveLocalMemory(uint32_t size);
```

### 6.3.2 PlatformAscendCManager - 平台管理器

```cpp
// 获取单例
static PlatformAscendC& GetInstance();

// 获取平台信息
PlatformAscendC& platform = PlatformAscendCManager::GetInstance();
uint32_t coreNum = platform.GetCoreNum();
std::string socVersion = platform.GetSocVersion();
```

---

## 6.4 单算子 API 执行相关接口

### 6.4.1 常用宏和类

#### ADD_TO_LAUNCHER_LIST_AICORE

```cpp
ADD_TO_LAUNCHER_LIST_AICORE(kernel_func)
```

#### CREATE_EXECUTOR

```cpp
auto executor = CREATE_EXECUTOR(op_type, tiling_data);
```

#### INFER_SHAPE

```cpp
INFER_SHAPE(op, input_shape, output_shape);
```

#### OP_ATTR

```cpp
OP_ATTR(op, attr_name)
OP_ATTR_NAMES(op, attr1, attr2, ...)
```

#### OP_INPUT / OP_OUTPUT

```cpp
OP_INPUT(op, index)
OP_OUTPUT(op, index)
```

#### OP_OUTSHAPE

```cpp
OP_OUTSHAPE(op, index, shape)
```

#### OP_WORKSPACE

```cpp
OP_WORKSPACE(op, size)
```

#### OpImplMode

```cpp
enum class OpImplMode {
    AI_CPU,    // AI CPU实现
    AICORE,    // AI Core实现
};
```

#### OpExecMode

```cpp
enum class OpExecMode {
    KERNEL,    // Kernel实现
    TBE,       // TBE实现
};
```

### 6.4.2 bfloat16 支持

```cpp
// bfloat16 类型转换
bfloat16_t value = static_cast<bfloat16_t>(float_value);
float value = static_cast<float>(bfloat_value);
```

---

## 使用示例

### 完整算子注册示例

```cpp
// 1. 定义Tiling结构体
BEGIN_TILING_DATA_DEF(MyOpTiling)
    TILING_DATA_MEMBER(uint32_t, blockDim)
    TILING_DATA_MEMBER(uint32_t, repeatTimes)
    TILING_DATA_MEMBER(uint32_t, totalLength)
END_TILING_DATA_DEF(MyOpTiling)

// 2. 注册Tiling结构体
REGISTER_TILING_DATA_CLASS(MyOpTiling)

// 3. 注册算子原型
OP_ADD("MyOp")
    .Input("x", DT_FLOAT, FORMAT_NCHW)
    .Output("y", DT_FLOAT, FORMAT_NCHW)
    .Attr<int>("kernel_size")
    .SetInferShape(InferShape);

// 4. InferShape函数
static bool InferShape(const std::vector<std::vector<int64_t>>& inputShapes,
                       std::vector<std::vector<int64_t>>& outputShapes,
                       const std::map<std::string, attr_val>& attrs) {
    outputShapes[0] = inputShapes[0];
    return true;
}

// 5. 注册AICore配置
REGISTER_OP_AICORE_CONFIG(MyOp)
    .Input(0, "x")
    .Output(0, "y")
    .SetOpSelectFormat("{0}")
    .AddConfig("runascurrenttik");
```

### 平台信息获取示例

```cpp
void GetPlatformInfo() {
    PlatformAscendC& platform = PlatformAscendCManager::GetInstance();
    
    uint32_t coreNum = platform.GetCoreNum();
    std::string socVersion = platform.GetSocVersion();
    uint32_t memSize = platform.GetCoreMemSize();
    
    printf("Core Num: %d\n", coreNum);
    printf("SoC Version: %s\n", socVersion.c_str());
    printf("Core Memory: %d KB\n", memSize / 1024);
}
```

### Tiling计算示例

```cpp
void ComputeTiling(MyOpTiling& tiling,
                  const std::vector<std::vector<int64_t>>& shapes) {
    PlatformAscendC& platform = PlatformAscendCManager::GetInstance();
    uint32_t coreNum = platform.GetCoreNum();
    
    uint64_t totalLength = shapes[0][0] * shapes[0][1] * shapes[0][2] * shapes[0][3];
    
    tiling.blockDim = std::min(coreNum, static_cast<uint32_t>(totalLength / 32));
    tiling.repeatTimes = totalLength / 32;
    tiling.totalLength = totalLength;
}
```
