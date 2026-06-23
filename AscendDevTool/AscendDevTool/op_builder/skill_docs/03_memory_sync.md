# 内存管理与同步控制

> Ascend C 核心API - TPipe, TQue, 队列管理与同步机制

## 目录

- [TPipe](#tpipe)
- [TBufPool](#tbufpool)
- [TBufPool 自定义](#tbufpool自定义)
- [TQue](#tque)
- [TQueBind](#tquebind)
- [TBuf](#tbuf)
- [workspace](#workspace)
- [核内同步](#核内同步)
- [核间同步](#核间同步)
- [任务间同步](#任务间同步)
- [TPosition](#tposition)

---

## TPipe

管道管理器，用于核内内存管理和任务调度。

### 构造函数

```cpp
TPipe();
TPipe(TPosition position);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| position | TPosition | 数据存储位置 (默认: POS_LOCAL) |

### 方法列表

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| Init | void | 初始化 |
| Destroy | void | 销毁 |
| InitBuffer | void | 初始化缓冲区 |
| Reset | void | 重置 |
| AllocEventID | uint32_t | 分配事件ID |
| ReleaseEventID | void | 释放事件ID |
| FetchEventID | uint32_t | 获取事件ID |
| GetBaseAddr | uint64_t | 获取基地址 |
| InitBufPool | void | 初始化缓冲池 |
| InitSpmBuffer | void | 初始化SPM缓冲区 |
| WriteSpmBuffer | void | 写SPM缓冲区 |
| ReadSpmBuffer | void | 读SPM缓冲区 |
| GetTPipePtr | TPipe* | 获取TPipe指针 |

### Init

初始化TPipe实例。

```cpp
void Init();
```

### Destroy

销毁TPipe实例，释放资源。

```cpp
void Destroy();
```

### InitBuffer

初始化缓冲区。

```cpp
void InitBuffer(void* buffer, uint32_t size);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| buffer | void* | 缓冲区指针 |
| size | uint32_t | 缓冲区大小(字节) |

### Reset

重置TPipe。

```cpp
void Reset();
```

### AllocEventID

分配一个新的事件ID。

```cpp
uint32_t AllocEventID();
```

- **返回值**: 分配的事件ID

### ReleaseEventID

释放指定的事件ID。

```cpp
void ReleaseEventID(uint32_t eventId);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| eventId | uint32_t | 要释放的事件ID |

### FetchEventID

获取下一个可用的临时事件ID (临时使用，不需要释放)。

```cpp
uint32_t FetchEventID();
```

- **返回值**: 事件ID

### GetBaseAddr

获取缓冲区的基地址。

```cpp
uint64_t GetBaseAddr() const;
```

- **返回值**: 物理地址

### InitBufPool

初始化缓冲池。

```cpp
void InitBufPool(uint32_t blockNum, uint32_t blockSize);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| blockNum | uint32_t | 块数量 |
| blockSize | uint32_t | 每块大小(字节) |

### InitSpmBuffer

初始化SPM (Scratch Pad Memory) 缓冲区。

```cpp
void InitSpmBuffer(uint32_t size);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| size | uint32_t | SPM缓冲区大小(字节) |

### WriteSpmBuffer

写入SPM缓冲区。

```cpp
void WriteSpmBuffer(const void* src, uint32_t size, uint32_t offset = 0);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| src | const void* | 源数据指针 |
| size | uint32_t | 数据大小 |
| offset | uint32_t | 偏移量 (默认: 0) |

### ReadSpmBuffer

读取SPM缓冲区。

```cpp
void ReadSpmBuffer(void* dst, uint32_t size, uint32_t offset = 0);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | void* | 目标缓冲区 |
| size | uint32_t | 数据大小 |
| offset | uint32_t | 偏移量 (默认: 0) |

### GetTPipePtr

获取TPipe指针 (全局函数)。

```cpp
TPipe* GetTPipePtr();
```

- **返回值**: TPipe实例指针

### 使用示例

```cpp
// 创建和��始化TPipe
TPipe pipe;
pipe.Init();

// 初始化缓冲区
char buffer[1024];
pipe.InitBuffer(buffer, sizeof(buffer));

// 使用队列
TQue<4, TPosition::POS_LOCAL> que;
que.InitBuffer();

// 计算核心
Exp(dst, src, repeat);

pipe.Destroy();
```

---

## TBufPool

缓冲区池，用于高效管理多个固定大小的缓冲区。

### 构造函数

```cpp
TBufPool();
TBufPool(TPosition position);
```

### 方法列表

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| InitBufPool | void | 初始化缓冲池 |
| InitBuffer | void | 初始化缓冲区 |
| Reset | void | 重置 |

### InitBufPool

初始化缓冲池。

```cpp
void InitBufPool(uint32_t blockNum, uint32_t blockSize);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| blockNum | uint32_t | 块数量 |
| blockSize | uint32_t | 每块大小(字节) |

### InitBuffer

初始化缓冲区。

```cpp
void InitBuffer(void* buffer, uint32_t bufferSize, uint32_t blockSize);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| buffer | void* | 缓冲区指针 |
| bufferSize | uint32_t | 缓冲区总大小 |
| blockSize | uint32_t | 单块大小 |

### 使用示例

```cpp
TBufPool pool;
pool.InitBufPool(4, 256);  // 4个块，每块256字节

char buf[1024];
pool.InitBuffer(buf, sizeof(buf), 256);

void* block = pool.GetBufHandle();
```

---

## TBufPool自定义

外部实现缓冲池，用于自定义内存管理。

### EXTERN_IMPL_BUFPOOL 宏

```cpp
EXTERN_IMPL_BUFPOOL(poolName, blockNum, blockSize);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| poolName | identifier | 缓冲池名称 |
| blockNum | uint32_t | 块数量 |
| blockSize | uint32_t | 每块大小 |

### 方法

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| Reset | void | 重置 |
| Init | void | 初始化 |
| GetBufHandle | void* | 获取缓冲区句柄 |
| SetCurAddr | void | 设置当前地址 |
| GetCurAddr | void* | 获取当前地址 |
| SetCurBufSize | void | 设置当前缓冲区大小 |
| GetCurBufSize | uint32_t | 获取当前缓冲区大小 |

### Reset

重置缓冲池。

```cpp
void Reset();
```

### Init

初始化。

```cpp
void Init();
```

### GetBufHandle

获取空闲缓冲区句柄。

```cpp
void* GetBufHandle();
```

- **返回值**: 缓冲区句柄

### SetCurAddr

设置当前地址。

```cpp
void SetCurAddr(void* addr);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| addr | void* | 地址 |

### GetCurAddr

获取当前地址。

```cpp
void* GetCurAddr() const;
```

- **返回值**: 当前地址

### GetCurBufSize

获取当前缓冲区大小。

```cpp
uint32_t GetCurBufSize() const;
```

- **返回值**: 当前大小

---

## TQue

队列管理器，用于任务排队和张量管理。

### 模板参数

```cpp
template<uint32_t QueSize, TPosition POS>
class TQue { ... };
```

| 参数 | 描述 |
|------|------|
| QueSize | 队列大小 |
| POS | 数据位置 |

### 构造函数

```cpp
TQue();
TQue(const TQue& other);
```

### 方法列表

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| Init | void | 初始化 |
| AllocTensor | LocalTensor | 分配张量 |
| FreeTensor | void | 释放张量 |
| EnQue | void | 入队 |
| DeQue | LocalTensor | 出队 |
| VacantInQue | uint32_t | 队列空位数 |
| HasTensorInQue | bool | 队列是否有张量 |
| GetTensorCountInQue | uint32_t | 队列中张量数 |
| HasIdleBuffer | bool | 是否有空闲缓冲区 |

### Init

初始化队列。

```cpp
void Init();
```

### InitBuffer

初始化队列缓冲区。

```cpp
void InitBuffer();
```

### AllocTensor

从队列分配张量。

```cpp
LocalTensor<DT> AllocTensor();
```

- **返回值**: 分配的LocalTensor

### FreeTensor

释放张量。

```cpp
void FreeTensor(LocalTensor<DT>& tensor);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| tensor | LocalTensor& | 要释放的张量 |

### EnQue

入队。

```cpp
void EnQue(const TPipe& pipe);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| pipe | const TPipe& | TPipe引用 |

### DeQue

出队。

```cpp
LocalTensor<DT> DeQue(const TPipe& pipe);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| pipe | const TPipe& | TPipe引用 |
| **返回值** | LocalTensor | 出队的张量 |

### VacantInQue

获取队列空位数。

```cpp
uint32_t VacantInQue() const;
```

- **返回值**: 可用空位数

### HasTensorInQue

检查队列是否有张量。

```cpp
bool HasTensorInQue() const;
```

- **返回值**: 是否有张量

### GetTensorCountInQue

获取队列中张量数。

```cpp
uint32_t GetTensorCountInQue() const;
```

- **返回值**: 张量数量

### HasIdleBuffer

是否有空闲缓冲区。

```cpp
bool HasIdleBuffer() const;
```

- **返回值**: 是否有空闲缓冲区

### 使用示例

```cpp
// 定义队列
TQue<4, TPosition::POS_LOCAL> que;
TPipe pipe;

pipe.Init();
que.InitBuffer();

// 分配输入张量
LocalTensor<float> input = que.AllocTensor();
// 使用输入张量
Exp(output, input, repeat);

// 入队 (标记计算完成)
que.EnQue(pipe);

// 出队 (等待计算完成)
LocalTensor<float> result = que.DeQue(pipe);
```

---

## TQueBind

绑定的队列，支持多位置 (Ping/Pong)。

### 模板参数

```cpp
template<uint32_t QueSize, TPosition POS>
class TQueBind { ... };
```

### 构造函数

```cpp
TQueBind();
TQueBind(const TQueBind& other);
```

### 方法列表

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| Init | void | 初始化 |
| AllocTensor | LocalTensor | 分配张量 |
| FreeTensor | void | 释放张量 |
| EnQue | void | 入队 |
| DeQue | LocalTensor | 出队 |
| VacantInQue | uint32_t | 队列空位数 |
| HasTensorInQue | bool | 队列是否有张量 |
| GetTensorCountInQue | uint32_t | 队列中张量数 |
| HasIdleBuffer | bool | 是否有空闲缓冲区 |
| FreeAllEvent | void | 释放所有事件 |
| InitBufHandle | void | 初始化缓冲区句柄 |
| InitStartBufHandle | void | 初始化起始句柄 |

### Init

初始化。

```cpp
void Init();
```

### AllocTensor

分配张量。

```cpp
LocalTensor<DT> AllocTensor(TPosition pos);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| pos | TPosition | 位置 |
| **返回值** | LocalTensor | 分配的张量 |

### EnQue

入队。

```cpp
void EnQue(const TPipe& pipe, TPosition pos = TPosition::POS_LOCAL);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| pipe | const TPipe& | TPipe引用 |
| pos | TPosition | 位置 (默认: POS_LOCAL) |

### DeQue

出队。

```cpp
LocalTensor<DT> DeQue(const TPipe& pipe, TPosition pos = TPosition::POS_LOCAL);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| pipe | const TPipe& | TPipe引用 |
| pos | TPosition | 位置 (默认: POS_LOCAL) |
| **返回值** | LocalTensor | 出队的张量 |

### FreeAllEvent

释放所有事件。

```cpp
void FreeAllEvent();
```

### InitBufHandle

初始化缓冲区句柄。

```cpp
void InitBufHandle(void* buffer, uint32_t size, uint32_t blockSize);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| buffer | void* | 缓冲区指针 |
| size | uint32_t | 大小 |
| blockSize | uint32_t | 块大小 |

### 使用示例

```cpp
// Ping-Pong双缓冲
TQueBind<2, TPosition::POS_LOCAL> que;
TPipe pipe;

pipe.Init();
que.Init();

void* buf0 = malloc(1024);
void* buf1 = malloc(1024);
que.InitBufHandle(buf0, 1024, 512);

// Ping位置分配
LocalTensor<float> input = que.AllocTensor(TPosition::POS_PING);

// 计算
Exp(output, input, repeat);

que.EnQue(pipe, TPosition::POS_PING);
auto result = que.DeQue(pipe, TPosition::POS_PING);
```

---

## TBuf

张量缓冲区，用于简单的内存管理。

### 构造函数

```cpp
TBuf();
TBuf(const TBuf& other);
```

### 方法

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| Get | template<T> T | 获取数据 |
| GetWithOffset | template<T> T | 带偏移获取数据 |

### Get

获取缓冲区数据。

```cpp
template<typename T>
T Get() const;
```

- **返回值**: 类型T的数据

### GetWithOffset

带偏移获取数据。

```cpp
template<typename T>
T Get(uint32_t offset) const;
```

| 参数 | 类型 | 描述 |
|------|------|------|
| offset | uint32_t | 偏移量 |
| **返回值** | T | 偏移后的数据 |

---

## workspace

工作空间管理。

### GetSysWorkSpacePtr

获取系统工作空间指针。

```cpp
void* GetSysWorkSpacePtr();
```

- **返回值**: 系统工作空间指针

### SetSysWorkSpace

设置系统工作空间。

```cpp
void SetSysWorkSpace(const void* workspace, uint32_t size);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| workspace | const void* | 工作空间指针 |
| size | uint32_t | 大小 |

### GetUserWorkspace

获取用户工作空间。

```cpp
void* GetUserWorkspace();
```

- **返回值**: 用户工作空间指针

### 使用示例

```cpp
// 获取用户工作空间
void* workspace = GetUserWorkspace();
void* inputBuf = workspace;
void* outputBuf = (char*)workspace + 512;
```

---

## 核内同步

核内的同步控制机制。

### TQueSync

基于队列的同步。

```cpp
void SetFlag(const TPipe& pipe, uint32_t flag);
uint32_t WaitFlag(const TPipe& pipe, uint32_t flag);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| pipe | const TPipe& | TPipe引用 |
| flag | uint32_t | 标志位 |

### SetFlag/WaitFlag (ISASI)

设置标志 / 等待标志。

```cpp
void SetFlag(uint32_t flag);
uint32_t WaitFlag(uint32_t flag);
```

### PipeBarrier (ISASI)

管道屏障同步。

```cpp
void PipeBarrier();
```

- 确保同一管道内所有操作完成

### DataSyncBarrier (ISASI)

数据同步屏障。

```cpp
void DataSyncBarrier();
```

- 确保数据操作完成

### 使用示例

```cpp
// 使用SetFlag/WaitFlag进行同步
SetFlag(1);  // Block 0 设置标志
WaitFlag(1); // 等待标志

// 使用PipeBarrier
Exp(output, input, repeat);
PipeBarrier();  // 等待Exp完成
```

---

## 核间同步

跨核/跨AI Core的同步机制。

### IBSet

设置核间标志。

```cpp
void IBSet(uint32_t flag);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| flag | uint32_t | 标志位 |

### IBWait

等待核间标志。

```cpp
void IBWait(uint32_t flag);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| flag | uint32_t | 标志位 |

### SyncAll

所有核同步。

```cpp
void SyncAll();
```

- 等待所有核完成当前阶段

### CrossCoreSetFlag (ISASI)

跨核设置标志。

```cpp
void CrossCoreSetFlag(uint32_t flag);
```

### CrossCoreWaitFlag (ISASI)

跨核等待标志。

```cpp
void CrossCoreWaitFlag(uint32_t flag);
```

### InitDetermineComputeWorkspace

初始化计算工作空间。

```cpp
void InitDetermineComputeWorkspace(uint32_t blockNum, uint32_t blockSize);
```

### NotifyNextBlock

通知下一块。

```cpp
void NotifyNextBlock();
```

### WaitPreBlock

等待上一块。

```cpp
void WaitPreBlock();
```

### 使用示例

```cpp
// 核间同步
uint32_t blockId = GetBlockIdx();
uint32_t blockNum = GetBlockNum();

if (blockId == 0) {
    // Block 0 设置标志表示数据已准备完成
    IBSet(1);
} else {
    // 其他Block等待
    IBWait(1);
}

// 所有Block同步
SyncAll();
```

---

## 任务间同步

任务级同步控制。

### SetNextTaskStart

设置下一任务开始。

```cpp
void SetNextTaskStart(uint32_t taskId);
```

### WaitPreTaskEnd

等待上一任务结束。

```cpp
void WaitPreTaskEnd();
```

---

## TPosition

数据存储位置枚举。

```cpp
enum class TPosition {
    POS_LOCAL,      // 本地内存 (UB)
    POS_PING,       // Ping缓冲区
    POS_PONG,      // Pong缓冲区
    POS_GLOBAL,     // 全局内存 (GM)
};
```

| 值 | 描述 |
|------|------|
| POS_LOCAL | 本地内存 (Unified Buffer) |
| POS_PING | Ping缓冲区 (双缓冲) |
| POS_PONG | Pong缓冲区 (双缓冲) |
| POS_GLOBAL | 全局内存 (Global Memory) |

---

## 完整使用示例

```cpp
// 完整算子框架
__CCEATTR__ void KernelName(KernelContext* ctx, void* param) {
    // 1. 获取Tiling数据
    GET_TILING_DATA(tiling);
    
    // 2. 初始化
    TPipe pipe;
    pipe.Init();
    
    TQue<4, TPosition::POS_LOCAL> que;
    que.InitBuffer();
    
    // 3. 分配张量
    LocalTensor<float> input = que.AllocTensor();
    LocalTensor<float> output = que.AllocTensor();
    
    // 4. 数据搬运
    DataCopy(input, ctx->input, size);
    
    // 5. 计算
    Exp(output, input, repeat);
    
    // 6. 同步
    que.EnQue(pipe);
    auto result = que.DeQue(pipe);
    
    // 7. 结果写回
    DataCopy(ctx->output, result, size);
    
    pipe.Destroy();
}
```