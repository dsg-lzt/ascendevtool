# 原子操作与矩阵计算

> Ascend C ISASI 特定指令 - 矩阵运算

## 目录

- [矩阵计算 (ISASI)](#矩阵计算-isasi)
  - [Mmad](#mmad)
  - [MmadWithSparse](#mmadwithsparse)
  - [Conv2D](#conv2d)
  - [Gemm](#gemm)
- [资源管理 (ISASI)](#资源管理-isasi)
  - [CubeResGroupHandle](#cuberesgrouphandle)
  - [GroupBarrier](#groupbarrier)
  - [KfcWorkspace](#kfcworkspace)

---

## 矩阵计算 (ISASI)

以下为ISASI (Inner Software Adaptive Specific Instruction) 特定指令。

### Mmad

矩阵乘加运算。

```cpp
void Mmad(LocalTensor<DT> dst, LocalTensor<DT> src0, 
         LocalTensor<DT> src1, MmadParams params);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 (结果矩阵) |
| src0 | LocalTensor | 源张量0 (矩阵A) |
| src1 | LocalTensor | 源张量1 (矩阵B) |
| params | MmadParams | 矩阵运算参数 |

#### MmadParams 结构

```cpp
struct MmadParams {
    uint32_t m;        // 矩阵A的行数
    uint32_t n;        // 矩阵B的列数
    uint32_t k;        // 矩阵A的列数/矩阵B的行数
    uint32_t lda;      // 矩阵A的leading dimension
    uint32_t ldb;      // 矩阵B的leading dimension
    uint32_t ldd;      // 结果矩阵的leading dimension
    bool transA;      // 是否转置矩阵A
    bool transB;      // 是否转置矩阵B
};
```

### MmadWithSparse

稀疏矩阵乘加。

```cpp
void MmadWithSparse(LocalTensor<DT> dst, LocalTensor<DT> src0,
                   LocalTensor<DT> src1, SparseTensorDesc sparseDesc,
                   MmadParams params);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| src0 | LocalTensor | 源张量0 |
| src1 | LocalTensor | 源张量1 |
| sparseDesc | SparseTensorDesc | 稀疏张量描述 |
| params | MmadParams | 矩阵运算参数 |

#### 设置函数

##### SetMMLayoutTransform

设置MM布局转换。

```cpp
void SetMMLayoutTransform(bool enable);
```

##### SetHF32Mode

设置HF32模式。

```cpp
void SetHF32Mode(bool enable);
```

##### SetHF32TransMode

设置HF32转换模式。

```cpp
void SetHF32TransMode(HF32TransMode mode);
```

---

### Conv2D

二维卷积运算。

```cpp
void Conv2D(LocalTensor<DT> dst, LocalTensor<DT> src,
           LocalTensor<DT> weight, Conv2DParams params);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出张量 |
| src | LocalTensor | 输入张量 |
| weight | LocalTensor | 权重张量 |
| params | Conv2DParams | 卷积参数 |

#### Conv2DParams 结构

```cpp
struct Conv2DParams {
    uint32_t inputW;      // 输入宽度
    uint32_t inputH;     // 输入高度
    uint32_t inputC;     // 输入通道数
    uint32_t outputW;    // 输出宽度
    uint32_t outputH;   // 输出高度
    uint32_t outputC;   // 输出通道数
    uint32_t kernelW;    // 卷积核宽度
    uint32_t kernelH;    // 卷积核高度
    uint32_t strideW;    // 步长宽度
    uint32_t strideH;   // 步长高度
    uint32_t padW;       // 填充宽度
    uint32_t padH;      // 填充高度
    uint32_t dilationW; // 膨胀宽度
    uint32_t dilationH; // 膨胀高度
};
```

---

### Gemm

通用矩阵乘法。

```cpp
void Gemm(LocalTensor<DT> dst, LocalTensor<DT> srcA,
         LocalTensor<DT> srcB, GemmParams params);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 输出矩阵 |
| srcA | LocalTensor | 输入矩阵A |
| srcB | LocalTensor | 输入矩阵B |
| params | GemmParams | GEMM参数 |

#### GemmParams 结构

```cpp
struct GemmParams {
    uint32_t m;        // 矩阵A的行数
    uint32_t n;        // 矩阵B的列数
    uint32_t k;        // 公共维度
    uint32_t lda;      // 矩阵A的leading dimension
    uint32_t ldb;      // 矩阵B的leading dimension
    uint32_t ldc;      // 结果矩阵的leading dimension
    float alpha;       // 缩放因子
    float beta;        // 偏置因子
    bool transA;       // 是否转置A
    bool transB;       // 是否转置B
};
```

---

## 资源管理 (ISASI)

### CubeResGroupHandle

Cube资源组句柄，用于管理Cube计算资源。

#### 构造函数

```cpp
CubeResGroupHandle();
CubeResGroupHandle(uint32_t size);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| size | uint32_t | 资源组大小 |

#### 方法列表

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| CreateCubeResGroup | void | 创建Cube资源组 |
| AssignQueue | void | 分配队列 |
| AllocMessage | MessageHandle | 分配消息 |
| PostMessage | void | 发送消息 |
| PostFakeMsg | void | 发送假消息 |
| SetQuit | void | 设置退出 |
| Wait | void | 等待 |
| FreeMessage | void | 释放消息 |
| SetSkipMsg | void | 设置跳过消息 |

#### CreateCubeResGroup

创建Cube资源组。

```cpp
void CreateCubeResGroup(uint32_t size);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| size | uint32_t | 资源组大小 |

#### AssignQueue

分配队列。

```cpp
void AssignQueue(TQue<QueSize, POS> que);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| que | TQue | 队列对象 |

#### AllocMessage

分配消息。

```cpp
MessageHandle AllocMessage();
```

- **返回值**: 消息句柄 |

#### PostMessage

发送消息。

```cpp
void PostMessage(MessageHandle msg);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| msg | MessageHandle | 消息句柄 |

#### PostFakeMsg

发送假消息。

```cpp
void PostFakeMsg();
```

#### SetQuit

设置退出标志。

```cpp
void SetQuit();
```

#### Wait

等待资源组完成。

```cpp
void Wait();
```

#### FreeMessage

释放消息。

```cpp
void FreeMessage(MessageHandle msg);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| msg | MessageHandle | 消息句柄 |

#### 使用示例

```cpp
CubeResGroupHandle group;
group.CreateCubeResGroup(4);
group.AssignQueue(que);

MessageHandle msg = group.AllocMessage();
// 填充消息数据
group.PostMessage(msg);
group.Wait();
group.FreeMessage(msg);
```

---

### GroupBarrier

组屏障，用于同步组内成员。

#### 构造函数

```cpp
GroupBarrier();
GroupBarrier(uint32_t groupSize);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| groupSize | uint32_t | 组大小 |

#### 方法列表

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| Arrive | void | 成员到达 |
| Wait | void | 等待所有成员 |
| GetWorkspaceLen | uint32_t | 获取工作空间长度 |

#### Arrive

成员到达屏障点。

```cpp
void Arrive();
```

#### Wait

等待所有组成员到达。

```cpp
void Wait();
```

#### GetWorkspaceLen

获取所需工作空间长度。

```cpp
uint32_t GetWorkspaceLen() const;
```

- **返回值**: 工作空间长度 |

#### 使用示例

```cpp
GroupBarrier barrier(4);  // 4个成员
barrier.Arrive();  // 到达屏障
barrier.Wait();   // 等待所有成员
```

---

### KfcWorkspace

KFC (Kernel Fusion Controller) 工作空间。

#### 构造函数与析构函数

```cpp
KfcWorkspace();
~KfcWorkspace();
```

#### 方法

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| UpdateKfcWorkspace | void | 更新KFC工作空间 |
| GetKfcWorkspace | void* | 获取KFC工作空间 |

#### UpdateKfcWorkspace

更新KFC工作空间。

```cpp
void UpdateKfcWorkspace(void* workspace, uint32_t size);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| workspace | void* | 工作空间指针 |
| size | uint32_t | 大小 |

#### GetKfcWorkspace

获取KFC工作空间。

```cpp
void* GetKfcWorkspace();
```

- **返回值**: 工作空间指针 |

#### 使用示例

```cpp
KfcWorkspace kfc;
kfc.UpdateKfcWorkspace(workspace, size);
void* ptr = kfc.GetKfcWorkspace();
```

---

## 矩阵计算示例

### 基本矩阵乘法 (使用Mmad)

```cpp
void MmadExample() {
    MmadParams params;
    params.m = 16;
    params.n = 16;
    params.k = 16;
    params.lda = 16;
    params.ldb = 16;
    params.ldc = 16;
    params.transA = false;
    params.transB = false;
    
    Mmad(dst, srcA, srcB, params);
}
```

### 基本卷积 (使用Conv2D)

```cpp
void Conv2DExample() {
    Conv2DParams params;
    params.inputW = 224;
    params.inputH = 224;
    params.inputC = 3;
    params.outputW = 224;
    params.outputH = 224;
    params.outputC = 64;
    params.kernelW = 3;
    params.kernelH = 3;
    params.strideW = 1;
    params.strideH = 1;
    params.padW = 1;
    params.padH = 1;
    
    Conv2D(output, input, weight, params);
}
```
