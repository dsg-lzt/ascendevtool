# 高阶API - Hccl通信

> Ascend C 集合通信库 (Huawei Collective Communication Library)

## 目录

- [Kernel侧接口](#kernel侧接口)
- [Tiling侧接口](#tiling侧接口)
- [使用示例](#使用示例)

---

## Kernel侧接口

### Hccl 初始化

#### InitV2

初始化Hccl。

```cpp
void InitV2(uint64_t* ringId);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| ringId | uint64_t* | 环ID |

#### SetCcTilingV2

设置集合通信Tiling。

```cpp
void SetCcTilingV2(const void* tilingData);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| tilingData | const void* | Tiling数据 |

### 集合通信操作

#### AllReduce

全归约: 所有节点数据归约后分发到所有节点。

```cpp
void AllReduce(LocalTensor<DT> sendBuf, LocalTensor<DT> recvBuf,
              uint32_t dataCount,HcclReduceOp op, void* comm);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| sendBuf | LocalTensor | 发送缓冲区 |
| recvBuf | LocalTensor | 接收缓冲区 |
| dataCount | uint32_t | 数据数量 |
| op | HcclReduceOp | 归约操作类型 |
| comm | void* | 通信器 |

#### HcclReduceOp 枚举

```cpp
enum class HcclReduceOp {
    SUM,      // 求和
    MAX,      // 最大值
    MIN,      // 最小值
    PROD,     // 乘积
    AND,      // 逻辑与
    OR,       // 逻辑或
};
```

#### AllGather

全收集: 所有节点数据收集后分发到所有节点。

```cpp
void AllGather(LocalTensor<DT> sendBuf, LocalTensor<DT> recvBuf,
              uint32_t dataCount, void* comm);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| sendBuf | LocalTensor | 发送缓冲区 |
| recvBuf | LocalTensor | 接收缓冲区 |
| dataCount | uint32_t | 数据数量 |
| comm | void* | 通信器 |

#### ReduceScatter

归约分散: 先归约再分散到各节点。

```cpp
void ReduceScatter(LocalTensor<DT> sendBuf, LocalTensor<DT> recvBuf,
                  uint32_t dataCount, HcclReduceOp op, void* comm);
```

#### AlltoAll

全交换: 每个节点向所有其他节点发送数据。

```cpp
void AlltoAll(LocalTensor<DT> sendBuf, LocalTensor<DT> recvBuf,
              uint32_t dataCount, void* comm);
```

#### AlltoAllV

变长全交换: 支持不同长度的数据交换。

```cpp
void AlltoAllV(LocalTensor<DT> sendBuf, uint32_t* sendCounts,
              uint32_t* sdispls, LocalTensor<DT> recvBuf,
              uint32_t* recvCounts, uint32_t* rdispls, void* comm);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| sendBuf | LocalTensor | 发送缓冲区 |
| sendCounts | uint32_t* | 各节点发送数量 |
| sdispls | uint32_t* | 发送偏移 |
| recvBuf | LocalTensor | 接收缓冲区 |
| recvCounts | uint32_t* | 各节点接收数量 |
| rdispls | uint32_t* | 接收偏移 |
| comm | void* | 通信器 |

#### BatchWrite

批量写入。

```cpp
void BatchWrite(LocalTensor<DT> buf, uint64_t addr, 
               uint32_t count, void* comm);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| buf | LocalTensor | 数据缓冲区 |
| addr | uint64_t | 地址 |
| count | uint32_t | 数据数量 |
| comm | void* | 通信器 |

#### BatchRead

批量读取。

```cpp
void BatchRead(uint64_t addr, LocalTensor<DT> buf,
              uint32_t count, void* comm);
```

#### Reduce

归约: 数据归约到指定节点。

```cpp
void Reduce(LocalTensor<DT> sendBuf, LocalTensor<DT> recvBuf,
           uint32_t dataCount, HcclReduceOp op, uint32_t root, void* comm);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| sendBuf | LocalTensor | 发送缓冲区 |
| recvBuf | LocalTensor | 接收缓冲区 |
| dataCount | uint32_t | 数据数量 |
| op | HcclReduceOp | 归约操作 |
| root | uint32_t | 根节点ID |
| comm | void* | 通信器 |

#### Broadcast

广播: 从根节点发送数据到所有节点。

```cpp
void Broadcast(LocalTensor<DT> buf, uint32_t dataCount,
              uint32_t root, void* comm);
```

#### Scatter

分散: 将数据分散到各节点。

```cpp
void Scatter(LocalTensor<DT> sendBuf, LocalTensor<DT> recvBuf,
            uint32_t dataCount, uint32_t root, void* comm);
```

#### Gather

聚集: 从各节点收集数据。

```cpp
void Gather(LocalTensor<DT> sendBuf, LocalTensor<DT> recvBuf,
           uint32_t dataCount, uint32_t root, void* comm);
```

#### AlltoAllVC

全交换变体。

```cpp
void AlltoAllVC(LocalTensor<DT> sendBuf, uint32_t* sendCounts,
               uint32_t* sendOffsets, LocalTensor<DT> recvBuf,
               uint32_t* recvCounts, uint32_t* recvOffsets, void* comm);
```

### 结束

#### HcclFinalize

结束Hccl通信。

```cpp
void HcclFinalize(void* comm);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| comm | void* | 通信器 |

---

## Tiling侧接口

### Hccl Tiling 类

```cpp
class HcclTiling {
public:
    // 构造函数
    HcclTiling();
    
    // 设置数据
    void SetTilingData(const HcclTilingData& data);
    void SetRankTable(const void* rankTable);
    void SetRankId(uint32_t rankId);
    
    // 获取Tiling数据
    void GetTilingData(HcclTilingData& data);
    
    // 获取缓冲区大小
    uint32_t GetTilingBufSize();
};
```

### HcclTilingData 结构

```cpp
struct HcclTilingData {
    uint32_t dataCount;      // 数据数量
    uint32_t rankSize;       // 节点数量
    uint32_t rankId;         // 当前节点ID
    HcclOpType opType;      // 操作类型
    // ... 其他字段
};
```

### HcclOpType 枚举

```cpp
enum class HcclOpType {
    ALLREDUCE,
    ALLGATHER,
    REDUCESCATTER,
    ALLTOALL,
    ALLTOALLV,
    BROADCAST,
    REDUCE,
    SCATTER,
    GATHER,
};
```

---

## 使用示例

### AllReduce

```cpp
void AllReduceExample() {
    uint64_t ringId;
    InitV2(&ringId);
    
    void* comm = /* 获取通信器 */;
    
    AllReduce(sendBuf, recvBuf, 1024, HcclReduceOp::SUM, comm);
    
    HcclFinalize(comm);
}
```

### AllGather

```cpp
void AllGatherExample() {
    void* comm = /* 获取通信器 */;
    
    AllGather(sendBuf, recvBuf, 256, comm);
    
    // recvBuf大小应为 256 * rankSize
}
```

### ReduceScatter

```cpp
void ReduceScatterExample() {
    void* comm = /* 获取通信器 */;
    
    ReduceScatter(sendBuf, recvBuf, 1024, HcclReduceOp::SUM, comm);
}
```

### 完整Tiling使用

```cpp
// Tiling侧
void GetHcclTiling(HcclTilingData& tilingData) {
    HcclTiling tiling;
    
    tiling.SetRankTable(rankTable);
    tiling.SetRankId(rankId);
    
    tilingData.opType = HcclOpType::ALLREDUCE;
    tilingData.dataCount = 1024;
    
    tiling.SetTilingData(tilingData);
}

// Kernel侧
__CCEATTR__ void KernelHccl(KernelContext* ctx, void* param) {
    GET_TILING_DATA(tiling);
    
    uint64_t ringId;
    InitV2(&ringId);
    
    void* comm = GetHcclComm();  // 获取通信器
    
    AllReduce(input, output, tiling.dataCount, HcclReduceOp::SUM, comm);
    
    HcclFinalize(comm);
}
```
