# 缓存处理与系统变量

> Ascend C 系统API - 缓存控制、系统变量访问

## 目录

- [缓存处理](#缓存处理)
- [系统变量访问](#系统变量访问)
- [原子操作](#原子操作)

---

## 缓存处理

### DataCachePreload

数据缓存预加载。

```cpp
void DataCachePreload(const void* addr, uint32_t size);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| addr | const void* | 地址 |
| size | uint32_t | 大小 |

### DataCacheCleanAndInvalid

清理并失效数据缓存。

```cpp
void DataCacheCleanAndInvalid(const void* addr, uint32_t size);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| addr | const void* | 地址 |
| size | uint32_t | 大小 |

### ICachePreLoad (ISASI)

指令缓存预加载。

```cpp
void ICachePreLoad(const void* addr);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| addr | const void* | 地址 |

### GetICachePreloadStatus (ISASI)

获取指令缓存预加载状态。

```cpp
uint32_t GetICachePreloadStatus();
```

- **返回值**: 预加载状态 |

---

## 系统变量访问

### GetBlockNum

获取AI Core的数量。

```cpp
uint32_t GetBlockNum();
```

- **返回值**: AI Core数量 |

### GetBlockIdx

获取当前AI Core的索引。

```cpp
uint32_t GetBlockIdx();
```

- **返回值**: AI Core索引 (从0开始) |

### GetDataBlockSizeInBytes

获取数据块大小 (字节)。

```cpp
uint32_t GetDataBlockSizeInBytes();
```

- **返回值**: 数据块大小 |

### GetArchVersion

获取架构版本。

```cpp
uint32_t GetArchVersion();
```

- **返回值**: 架构版本号 |

### GetTaskRatio

获取任务比例/数量。

```cpp
uint32_t GetTaskRatio();
```

- **返回值**: 任务比例 |

### InitSocState

初始化SoC状态。

```cpp
void InitSocState();
```

### GetProgramCounter (ISASI)

获取程序计数器。

```cpp
uint64_t GetProgramCounter();
```

- **返回值**: 程序计数器值 |

### GetSubBlockNum (ISASI)

获取子块数量。

```cpp
uint32_t GetSubBlockNum();
```

- **返回值**: 子块数量 |

### GetSubBlockIdx (ISASI)

获取子块索引。

```cpp
uint32_t GetSubBlockIdx();
```

- **返回值**: 子块索引 |

### GetSystemCycle (ISASI)

获取系统周期数。

```cpp
uint64_t GetSystemCycle();
```

- **返回值**: 系统周期数 |

### CheckLocalMemoryIA (ISASI)

检查本地内存IA (Integrity check)。

```cpp
bool CheckLocalMemoryIA();
```

- **返回值**: 检查结果 |

---

## 原子操作

### SetAtomicAdd

设置原子加操作。

```cpp
void SetAtomicAdd(LocalTensor<DT> dst, const DT value, uint32_t repeatTimes);
```

| 参数 | 类型 | 描述 |
|------|------|------|
| dst | LocalTensor | 目标张量 |
| value | DT | 累加值 |
| repeatTimes | uint32_t | 重复次数 |

### SetAtomicType

设置原子操作类型。

```cpp
template<typename OP>
void SetAtomicType();
```

| 参数 | 类型 | 描述 |
|------|------|------|
| OP | template | 原子操作类型 (Add/Max/Min) |

### SetAtomicNone

禁用原子操作。

```cpp
void SetAtomicNone();
```

### SetAtomicMax (ISASI)

设置原子最大值操作。

```cpp
void SetAtomicMax(LocalTensor<DT> dst, const DT value, uint32_t repeatTimes);
```

### SetAtomicMin (ISASI)

设置原子最小值操作。

```cpp
void SetAtomicMin(LocalTensor<DT> dst, const DT value, uint32_t repeatTimes);
```

### SetStoreAtomicConfig (ISASI)

设置存储原子配置。

```cpp
void SetStoreAtomicConfig(AtomicConfig config);
```

### GetStoreAtomicConfig (ISASI)

获取存储原子配置。

```cpp
AtomicConfig GetStoreAtomicConfig();
```

- **返回值**: 原子配置 |

### 使用示例

```cpp
// 原子加操作
SetAtomicType<AtomicAdd>();
SetAtomicAdd(output, scalar, repeat);

// 原子最大值
SetAtomicType<AtomicMax>();
SetAtomicMax(output, maxValue, repeat);

// 禁用原子操作
SetAtomicNone();
```