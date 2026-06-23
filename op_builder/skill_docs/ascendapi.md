# Ascend C 算子开发接口

> CANN 社区版 8.3.RC1 | 文档版本 01 | 发布日期 2025-12-15

---

## 目录

- [数据类型定义](#数据类型定义)
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
- [基础 API](#基础-api)
  - [标量计算](#标量计算)
  - [矢量计算](#矢量计算)
  - [数据搬运](#数据搬运)
  - [内存管理与同步控制](#内存管理与同步控制)
  - [缓存处理](#缓存处理)
  - [系统变量访问](#系统变量访问)
  - [原子操作](#原子操作)
  - [矩阵计算(ISASI)](#矩阵计算isasi)
  - [资源管理(ISASI)](#资源管理isasi)
  - [Kernel Tiling](#kernel-tiling)
- [高阶 API](#高阶-api)
  - [数学库](#数学库)
  - [Matmul](#matmul)
  - [激活函数](#激活函数)
  - [数据归一化](#数据归一化)
  - [量化反量化](#量化反量化)
  - [归约操作](#归约操作)
  - [排序](#排序)
  - [数据填充](#数据填充)
  - [索引操作](#索引操作)
  - [比较选择](#比较选择)
  - [数据过滤](#数据过滤)
  - [变形](#变形)
  - [Hccl](#hccl)

---

## 数据类型定义

### LocalTensor

本地张量，用于核内数据存储和处理。

| 接口 | 描述 |
|------|------|
| LocalTensor 构造函数 | 创建 LocalTensor 对象 |
| SetValue | 设置张量值 |
| GetValue | 获取张量值 |
| operator[] | 下标访问 |
| operator() | 函数调用访问 |
| SetSize | 设置张量大小 |
| GetSize | 获取张量大小 |
| SetUserTag | 设置用户标签 |
| GetUserTag | 获取用户标签 |
| ReinterpretCast | 类型重解释转换 |
| GetPhyAddr | 获取物理地址 |
| GetPosition | 获取数据位置 |
| GetLength | 获取数据长度 |
| SetShapeInfo | 设置形状信息 |
| GetShapeInfo | 获取形状信息 |
| SetAddrWithOffset | 设置带偏移的地址 |
| SetBufferLen | 设置缓冲区长度 |
| ToFile | 写入文件 |
| Print | 打印张量 |

### GlobalTensor

全局张量，用于核间数据通信和存储。

| 接口 | 描述 |
|------|------|
| GlobalTensor 构造函数 | 创建 GlobalTensor 对象 |
| SetGlobalBuffer | 设置全局缓冲区 |
| GetPhyAddr | 获取物理地址 |
| GetValue | 获取张量值 |
| operator() | 函数调用访问 |
| SetValue | 设置张量值 |
| GetSize | 获取张量大小 |
| operator[] | 下标访问 |
| SetShapeInfo | 设置形状信息 |
| GetShapeInfo | 获取形状信息 |
| SetL2CacheHint | 设置L2缓存提示 |

### ShapeInfo

形状信息数据结构。

### ListTensorDesc

张量描述符列表。

### TensorDesc

张量描述符，用于描述张量的属性。

| 接口 | 描述 |
|------|------|
| 构造和析构函数 | 创建和销毁 TensorDesc |
| SetShapeAddr | 设置形状地址 |
| GetDim | 获取维度 |
| GetIndex | 获取索引 |
| GetShape | 获取形状 |
| GetDataPtr | 获取数据指针 |
| GetDataObj | 获取数据对象 |

### Coordinate

坐标操作，用于多维索引计算。

| 接口 | 描述 |
|------|------|
| MakeCoord | 创建坐标 |
| Crd2Idx | 坐标转索引 |

### Layout

布局描述，定义张量的内存布局。

| 接口 | 描述 |
|------|------|
| Layout 构造函数 | 创建 Layout 对象 |
| layout | 布局构造 |
| 运算符重载 | 布局运算 |
| GetShape | 获取形状 |
| GetStride | 获取步长 |
| MakeShape | 创建形状 |
| MakeStride | 创建步长 |
| MakeLayout | 创建布局 |
| is_layout | 布局判断 |

### TensorTrait

张量特征，描述张量的类型特征。

| 接口 | 描述 |
|------|------|
| TensorTrait 构造函数 | 创建 TensorTrait 对象 |
| GetLayout | 获取布局 |
| SetLayout | 设置布局 |
| MakeTensorTrait | 创建张量特征 |
| is_tensorTrait | 类型判断 |

### LocalMemAllocator

本地内存分配器。

| 接口 | 描述 |
|------|------|
| LocalMemAllocator 构造函数 | 创建分配器 |
| GetCurAddr | 获取当前地址 |
| Alloc | 分配内存 |

### RepeatParams

| 接口 | 描述 |
|------|------|
| UnaryRepeatParams | 一元重复参数 |
| BinaryRepeatParams | 二元重复参数 |

---

## 基础 API

### 标量计算

| 接口 | 描述 |
|------|------|
| ScalarGetCountOfValue | 获取值的个数 |
| ScalarCountLeadingZero | 计数前导零 |
| ScalarCast | 标量类型转换 |
| CountBitsCntSameAsSignBit | 统计与符号位相同的位数 |
| ScalarGetSFFValue | 获取SFF值 |
| ToBfloat16 | 转换为bfloat16 |
| ToFloat | 转换为float |

### 矢量计算

#### 基础算术

| 接口 | 描述 |
|------|------|
| Exp | 指数运算 |
| Ln | 自然对数 |
| Abs | 绝对值 |
| Reciprocal | 倒数 |
| Sqrt | 平方根 |
| Rsqrt | 平方根倒数 |
| Relu | ReLU激活 |
| VectorPadding(ISASI) | 矢量填充 |
| Add | 加法 |
| Sub | 减法 |
| Mul | 乘法 |
| Div | 除法 |
| Max | 最大值 |
| Min | 最小值 |
| BilinearInterpolation(ISASI) | 双线性插值 |
| Adds | 标量加法 |
| Muls | 标量乘法 |
| Maxs | 标量最大值 |
| Mins | 标量最小值 |
| LeakyRelu | Leaky ReLU |
| Axpy | 矢量乘加 |

#### 逻辑计算

| 接口 | 描述 |
|------|------|
| Not | 逻辑非 |
| And | 逻辑与 |
| Or | 逻辑或 |
| ShiftLeft | 左移 |
| ShiftRight | 右移 |

#### 复合计算

| 接口 | 描述 |
|------|------|
| AddRelu | 加法+ReLU |
| AddReluCast | 加法+ReLU+类型转换 |
| AddDeqRelu | 加法+反量化+ReLU |
| SubRelu | 减法+ReLU |
| SubReluCast | 减法+ReLU+类型转换 |
| MulAddDst | 乘加 |
| MulCast | 乘法+类型转换 |
| FusedMulAdd | 融合乘加 |
| FusedMulAddRelu | 融合乘加+ReLU |

#### 比较指令

| 接口 | 描述 |
|------|------|
| Compare | 比较运算 |
| Compare(结果存入寄存器) | 比较结果存入寄存器 |
| CompareScalar | 标量比较 |
| GetCmpMask(ISASI) | 获取比较掩码 |
| SetCmpMask(ISASI) | 设置比较掩码 |

#### 选择指令

| 接口 | 描述 |
|------|------|
| Select | 条件选择 |
| GatherMask | 聚集掩码 |

#### 精度转换指令

| 接口 | 描述 |
|------|------|
| Cast | 类型转换 |
| CastDeq | 类型转换+反量化 |

#### 归约指令

| 接口 | 描述 |
|------|------|
| ReduceMax | 最大值归约 |
| ReduceMin | 最小值归约 |
| ReduceSum | 求和归约 |
| WholeReduceMax | 全局最大值归约 |
| WholeReduceMin | 全局最小值归约 |
| WholeReduceSum | 全局求和归约 |
| BlockReduceMax | 分块最大值归约 |
| BlockReduceMin | 分块最小值归约 |
| BlockReduceSum | 分块求和归约 |
| PairReduceSum | 成对求和归约 |
| RepeatReduceSum | 重复求和归约 |
| GetAccVal(ISASI) | 获取累加值 |
| GetReduceMaxMinCount(ISASI) | 获取最大最小值计数 |

#### 数据转换

| 接口 | 描述 |
|------|------|
| Transpose | 转置 |
| TransDataTo5HD | 转换为5HD格式 |

#### 数据填充

| 接口 | 描述 |
|------|------|
| Duplicate | 数据复制 |
| Brcb | 广播 |
| CreateVecIndex | 创建矢量索引 |

#### 排序组合(ISASI)

| 接口 | 描述 |
|------|------|
| ProposalConcat | 提议连接 |
| ProposalExtract | 提议提取 |
| RpSort16 | 16元素排序 |
| MrgSort4 | 4元素归并排序 |
| Sort32 | 32元素排序 |
| MrgSort | 归并排序 |
| GetMrgSortResult | 获取归并排序结果 |

#### 数据分散/数据收集

| 接口 | 描述 |
|------|------|
| Gather | 聚集操作 |
| Gatherb(ISASI) | 批量聚集 |
| Scatter(ISASI) | 分散操作 |

#### 掩码操作

| 接口 | 描述 |
|------|------|
| SetMaskCount | 设置掩码计数 |
| SetMaskNorm | 设置掩码归一化 |
| SetVectorMask | 设置矢量掩码 |
| ResetMask | 重置掩码 |

#### 量化设置

| 接口 | 描述 |
|------|------|
| SetDeqScale | 设置反量化尺度 |

### 数据搬运

| 接口 | 描述 |
|------|------|
| DataCopy | 数据拷贝 |
| DataCopyPad(ISASI) | 带填充的数据拷贝 |
| SetPadValue(ISASI) | 设置填充值 |
| Copy | 数据复制 |
| SetFixPipeConfig(ISASI) | 设置定标管道配置 |
| SetFixpipeNz2ndFlag(ISASI) | 设置NZ2ND标志 |
| SetFixpipePreQuantFlag(ISASI) | 设置预量化标志 |
| SetFixPipeClipRelu(ISASI) | 设置定标管道ClipReLU |
| SetFixPipeAddr(ISASI) | 设置定标管道地址 |
| InitConstValue(ISASI) | 初始化常量值 |
| LoadData(ISASI) | 加载数据 |
| LoadDataWithTranspose(ISASI) | 带转置加载数据 |
| LoadDataWithSparse(ISASI) | 带稀疏加载数据 |
| LoadUnzipIndex(ISASI) | 加载解压索引 |
| LoadDataUnzip(ISASI) | 解压加载数据 |
| SetFmatrix(ISASI) | 设置F矩阵 |
| SetLoadDataBoundary(ISASI) | 设置加载边界 |
| SetLoadDataRepeat(ISASI) | 设置加载重复 |
| SetLoadDataPaddingValue(ISASI) | 设置填充值 |
| SetAippFunctions(ISASI) | 设置AIPP函数 |
| LoadImageToLocal(ISASI) | 加载图像到本地 |
| Fixpipe(ISASI) | 定标管道 |

### 内存管理与同步控制

#### TPipe

| 接口 | 描述 |
|------|------|
| TPipe 构造函数 | 创建TPipe |
| Init | 初始化 |
| Destroy | 销毁 |
| InitBuffer | 初始化缓冲区 |
| Reset | 重置 |
| AllocEventID | 分配事件ID |
| ReleaseEventID | 释放事件ID |
| FetchEventID | 获取事件ID |
| GetBaseAddr | 获取基地址 |
| InitBufPool | 初始化缓冲池 |
| InitSpmBuffer | 初始化SPM缓冲区 |
| WriteSpmBuffer | 写SPM缓冲区 |
| ReadSpmBuffer | 读SPM缓冲区 |

| 接口 | 描述 |
|------|------|
| GetTPipePtr | 获取TPipe指针 |

#### TBufPool

| 接口 | 描述 |
|------|------|
| TBufPool 构造函数 | 创建TBufPool |
| InitBufPool | 初始化缓冲池 |
| InitBuffer | 初始化缓冲区 |
| Reset | 重置 |

#### 自定义TBufPool

| 接口 | 描述 |
|------|------|
| EXTERN_IMPL_BUFPOOL 宏 | 外部实现缓冲池 |
| Reset | 重置 |
| Init | 初始化 |
| GetBufHandle | 获取缓冲区句柄 |
| SetCurAddr | 设置当前地址 |
| GetCurAddr | 获取当前地址 |
| SetCurBufSize | 设置当前缓冲区大小 |
| GetCurBufSize | 获取当前缓冲区大小 |

#### TQue

| 接口 | 描述 |
|------|------|
| AllocTensor | 分配张量 |
| FreeTensor | 释放张量 |
| EnQue | 入队 |
| DeQue | 出队 |
| VacantInQue | 队列空位检查 |
| HasTensorInQue | 队列是否有张量 |
| GetTensorCountInQue | 获取队列中张量数 |
| HasIdleBuffer | 是否有空闲缓冲区 |

#### TQueBind

| 接口 | 描述 |
|------|------|
| TQueBind 构造函数 | 创建TQueBind |
| AllocTensor | 分配张量 |
| FreeTensor | 释放张量 |
| EnQue | 入队 |
| DeQue | 出队 |
| VacantInQue | 队列空位检查 |
| HasTensorInQue | 队列是否有张量 |
| GetTensorCountInQue | 获取队列中张量数 |
| HasIdleBuffer | 是否有空闲缓冲区 |
| FreeAllEvent | 释放所有事件 |
| InitBufHandle | 初始化缓冲区句柄 |
| InitStartBufHandle | 初始化起始缓冲区句柄 |

#### TBuf

| 接口 | 描述 |
|------|------|
| TBuf 构造函数 | 创建TBuf |
| Get | 获取数据 |
| GetWithOffset | 带偏移获取数据 |

#### workspace

| 接口 | 描述 |
|------|------|
| GetSysWorkSpacePtr | 获取系统工作空间指针 |
| SetSysWorkSpace | 设置系统工作空间 |
| GetUserWorkspace | 获取用户工作空间 |

#### 核内同步

| 接口 | 描述 |
|------|------|
| TQueSync | 队列同步 |
| SetFlag/WaitFlag(ISASI) | 设置/等待标志 |
| PipeBarrier(ISASI) | 管道屏障 |
| DataSyncBarrier(ISASI) | 数据同步屏障 |

#### 核间同步

| 接口 | 描述 |
|------|------|
| IBSet | 核间设置 |
| IBWait | 核间等待 |
| SyncAll | 全部同步 |
| CrossCoreSetFlag(ISASI) | 跨核设置标志 |
| CrossCoreWaitFlag(ISASI) | 跨核等待标志 |
| InitDetermineComputeWorkspace | 初始化计算工作空间 |
| NotifyNextBlock | 通知下一块 |
| WaitPreBlock | 等待上一块 |

#### 任务间同步

| 接口 | 描述 |
|------|------|
| SetNextTaskStart | 设置下一任务开始 |
| WaitPreTaskEnd | 等待上一任务结束 |

#### TPosition

数据位置枚举。

### 缓存处理

| 接口 | 描述 |
|------|------|
| DataCachePreload | 数据缓存预加载 |
| DataCacheCleanAndInvalid | 数据缓存清理和失效 |
| ICachePreLoad(ISASI) | 指令缓存预加载 |
| GetICachePreloadStatus(ISASI) | 获取指令缓存预加载状态 |

### 系统变量访问

| 接口 | 描述 |
|------|------|
| GetBlockNum | 获取块数 |
| GetBlockIdx | 获取块索引 |
| GetDataBlockSizeInBytes | 获取数据块大小(字节) |
| GetArchVersion | 获取架构版本 |
| GetTaskRatio | 获取任务比例 |
| InitSocState | 初始化SoC状态 |
| GetProgramCounter(ISASI) | 获取程序计数器 |
| GetSubBlockNum(ISASI) | 获取子块数 |
| GetSubBlockIdx(ISASI) | 获取子块索引 |
| GetSystemCycle(ISASI) | 获取系统周期 |
| CheckLocalMemoryIA(ISASI) | 检查本地内存IA |

### 原子操作

| 接口 | 描述 |
|------|------|
| SetAtomicAdd | 设置原子加 |
| SetAtomicType | 设置原子类型 |
| SetAtomicNone | 设置无原子操作 |
| SetAtomicMax(ISASI) | 设置原子最大值 |
| SetAtomicMin(ISASI) | 设置原子最小值 |
| SetStoreAtomicConfig(ISASI) | 设置存储原子配置 |
| GetStoreAtomicConfig(ISASI) | 获取存储原子配置 |

### 矩阵计算(ISASI)

| 接口 | 描述 |
|------|------|
| Mmad | 矩阵乘加 |
| MmadWithSparse | 稀疏矩阵乘加 |
| SetMMLayoutTransform | 设置MM布局转换 |
| SetHF32Mode | 设置HF32模式 |
| SetHF32TransMode | 设置HF32转换模式 |
| Conv2D | 二维卷积 |
| Gemm | 通用矩阵乘法 |

### 资源管理(ISASI)

#### CubeResGroupHandle

| 接口 | 描述 |
|------|------|
| CubeResGroupHandle 构造函数 | 创建Cube资源组句柄 |
| CreateCubeResGroup | 创建Cube资源组 |
| AssignQueue | 分配队列 |
| AllocMessage | 分配消息 |
| PostMessage | 发送消息 |
| PostFakeMsg | 发送假消息 |
| SetQuit | 设置退出 |
| Wait | 等待 |
| FreeMessage | 释放消息 |
| SetSkipMsg | 设置跳过消息 |

#### GroupBarrier

| 接口 | 描述 |
|------|------|
| GroupBarrier 构造函数 | 创建组屏障 |
| Arrive | 到达 |
| Wait | 等待 |
| GetWorkspaceLen | 获取工作空间长度 |

#### KfcWorkspace

| 接口 | 描述 |
|------|------|
| 构造函数与析构函数 | 创建和销毁KfcWorkspace |
| UpdateKfcWorkspace | 更新Kfc工作空间 |
| GetKfcWorkspace | 获取Kfc工作空间 |

### Kernel Tiling

| 接口 | 描述 |
|------|------|
| GET_TILING_DATA | 获取Tiling数据 |
| GET_TILING_DATA_WITH_STRUCT | 获取结构体Tiling数据 |
| GET_TILING_DATA_MEMBER | 获取Tiling数据成员 |
| GET_TILING_DATA_PTR_WITH_STRUCT | 获取结构体Tiling数据指针 |
| COPY_TILING_WITH_STRUCT | 复制结构体Tiling |
| COPY_TILING_WITH_ARRAY | 复制数组Tiling |
| TILING_KEY_IS | Tiling键判断 |
| REGISTER_TILING_DEFAULT | 注册默认Tiling |
| REGISTER_TILING_FOR_TILINGKEY | 为Tiling键注册Tiling |
| 设置Kernel类型 | 设置Kernel类型 |

---

## 高阶 API

### 数学库

每个数学库接口包含以下函数：
- 主函数
- GetXxxMaxMinTmpSize: 获取临时缓冲区大小
- GetXxxTmpBufferFactorSize: 获取临时缓冲区因子大小

| 接口 | 主函数 | 临时缓冲区函数 |
|------|--------|---------------|
| Tanh | Tanh | GetTanhMaxMinTmpSize, GetTanhTmpBufferFactorSize |
| Asin | Asin | GetAsinMaxMinTmpSize, GetAsinTmpBufferFactorSize |
| Sin | Sin | GetSinMaxMinTmpSize, GetSinTmpBufferFactorSize |
| Acos | Acos | GetAcosMaxMinTmpSize, GetAcosTmpBufferFactorSize |
| Cos | Cos | GetCosMaxMinTmpSize, GetCosTmpBufferFactorSize |
| Log | Log | GetLogMaxMinTmpSize, GetLogTmpBufferFactorSize |
| Atan | Atan | GetAtanMaxMinTmpSize, GetAtanTmpBufferFactorSize |
| Power | Power | GetPowerMaxMinTmpSize, GetPowerTmpBufferFactorSize |
| Sinh | Sinh | GetSinhMaxMinTmpSize, GetSinhTmpBufferFactorSize |
| Cosh | Cosh | GetCoshMaxMinTmpSize, GetCoshTmpBufferFactorSize |
| Tan | Tan | GetTanMaxMinTmpSize, GetTanTmpBufferFactorSize |
| Trunc | Trunc | GetTruncMaxMinTmpSize, GetTruncTmpBufferFactorSize |
| Frac | Frac | GetFracMaxMinTmpSize, GetFracTmpBufferFactorSize |
| Erf | Erf | GetErfMaxMinTmpSize, GetErfTmpBufferFactorSize |
| Erfc | Erfc | GetErfcMaxMinTmpSize, GetErfcTmpBufferFactorSize |
| Sign | Sign | GetSignMaxMinTmpSize, GetSignTmpBufferFactorSize |
| Atanh | Atanh | GetAtanhMaxMinTmpSize, GetAtanhTmpBufferFactorSize |
| Asinh | Asinh | GetAsinhMaxMinTmpSize, GetAsinhTmpBufferFactorSize |
| Acosh | Acosh | GetAcoshMaxMinTmpSize, GetAcoshTmpBufferFactorSize |
| Floor | Floor | GetFloorMaxMinTmpSize, GetFloorTmpBufferFactorSize |
| Ceil | Ceil | GetCeilMaxMinTmpSize, GetCeilTmpBufferFactorSize |
| Clamp | ClampMax, ClampMin | GetClampMaxMinTmpSize, GetClampTmpBufferFactorSize |
| Round | Round | GetRoundMaxMinTmpSize, GetRoundTmpBufferFactorSize |
| Axpy | Axpy | GetAxpyMaxMinTmpSize, GetAxpyTmpBufferFactorSize |
| Exp | Exp | GetExpMaxMinTmpSize, GetExpTmpBufferFactorSize |
| Lgamma | Lgamma | GetLgammaMaxMinTmpSize, GetLgammaTmpBufferFactorSize |
| Digamma | Digamma | GetDigammaMaxMinTmpSize, GetDigammaTmpBufferFactorSize |
| Xor | Xor | GetXorMaxMinTmpSize, GetXorTmpBufferFactorSize |
| CumSum | CumSum | GetCumSumMaxMinTmpSize |
| Fmod | Fmod | GetFmodMaxMinTmpSize, GetFmodTmpBufferFactorSize |

### Matmul

#### Kernel侧接口

| 接口 | 描述 |
|------|------|
| Matmul | 矩阵乘法 |
| MatmulConfig | 矩阵乘法配置 |
| MatmulCallBackFunc | 矩阵乘法回调函数 |
| MatmulPolicy | 矩阵乘法策略 |
| GetNormalConfig | 获取普通配置 |
| GetMDLConfig | 获取MDL配置 |
| GetSpecialMDLConfig | 获取特殊MDL配置 |
| GetIBShareNormConfig | 获取IB共享普通配置 |
| GetBasicConfig | 获取基础配置 |
| GetSpecialBasicConfig | 获取特殊基础配置 |
| GetMMConfig | 获取MM配置 |
| REGIST_MATMUL_OBJ | 注册矩阵乘法对象 |
| Init | 初始化 |
| SetTensorA | 设置张量A |
| SetTensorB | 设置张量B |
| SetBias | 设置偏置 |
| DisableBias | 禁用偏置 |
| GetBatchTensorC | 获取批处理张量C |
| Iterate | 迭代 |
| GetTensorC | 获取张量C |
| IterateAll | 迭代全部 |
| WaitIterateAll | 等待迭代全部 |
| IterateBatch | 迭代批处理 |
| WaitIterateBatch | 等待迭代批处理 |
| IterateNBatch | 迭代N批处理 |
| End | 结束 |
| SetHF32 | 设置HF32 |
| SetTail | 设置尾数据 |
| SetBatchNum | 设置批次数 |
| SetQuantScalar | 设置量化标量 |
| SetQuantVector | 设置量化矢量 |
| SetOrgShape | 设置原始形状 |
| SetSingleShape | 设置单一形状 |
| SetLocalWorkspace | 设置本地工作空间 |
| SetWorkspace | 设置工作空间 |
| SetAntiQuantScalar | 设置反量化标量 |
| SetAntiQuantVector | 设置反量化矢量 |
| WaitGetTensorC | 等待获取张量C |
| GetOffsetC | 获取C偏移 |
| AsyncGetTensorC | 异步获取张量C |
| SetUserDefInfo | 设置用户定义信息 |
| SetSelfDefineData | 设置自定义数据 |
| GetMatmulApiTiling | 获取矩阵乘法API Tiling |
| ClearBias | 清除偏置 |
| GetBatchC | 获取批处理C |
| SetSparseIndex | 设置稀疏索引 |

#### Tiling侧接口

| 接口 | 描述 |
|------|------|
| Matmul Tiling 类 | Matmul Tiling类 |
| TCubeTiling 结构体 | TCubeTiling结构体 |
| SetAType | 设置A类型 |
| SetBType | 设置B类型 |
| SetCType | 设置C类型 |
| SetBiasType | 设置偏置类型 |
| SetSingleShape | 设置单一形状 |
| SetShape | 设置形状 |
| SetOrgShape | 设置原始形状 |
| SetFixSplit | 设置固定分割 |
| EnableMultiCoreSplitK | 启用多核分割K |
| EnableBias | 启用偏置 |
| SetBufferSpace | 设置缓冲区空间 |
| SetTraverse | 设置遍历 |
| SetMadType | 设置Mad类型 |
| SetSplitRange | 设置分割范围 |
| SetDoubleBuffer | 设置双缓冲 |
| GetBaseM | 获取基M |
| GetBaseN | 获取基N |
| GetBaseK | 获取基K |
| GetTiling | 获取Tiling |
| EnableL1BankConflictOptimise | 启用L1 bank冲突优化 |
| SetDim | 设置维度 |
| SetSingleRange | 设置单一范围 |
| GetSingleShape | 获取单一形状 |
| GetCoreNum | 获取核数 |
| SetAlignSplit | 设置对齐分割 |
| SetDequantType | 设置反量化类型 |
| SetALayout | 设置A布局 |
| SetBLayout | 设置B布局 |
| SetCLayout | 设置C布局 |
| SetBatchNum | 设置批次数 |
| SetBatchInfoForNormal | 设置普通批处理信息 |
| SetMatmulConfigParams | 设置矩阵乘法配置参数 |
| SetBias | 设置偏置 |
| SetSplitK | 设置分割K |
| SetSparse | 设置稀疏 |
| MultiCoreMatmulGetTmpBufSize | 获取多核矩阵乘法临时缓冲区大小 |
| MultiCoreMatmulGetTmpBufSizeV2 | 获取多核矩阵乘法临时缓冲区大小V2 |
| BatchMatmulGetTmpBufSize | 获取批处理矩阵乘法临时缓冲区大小 |
| BatchMatmulGetTmpBufSizeV2 | 获取批处理矩阵乘法临时缓冲区大小V2 |
| MatmulGetTmpBufSize | 获取矩阵乘法临时缓冲区大小 |
| MatmulGetTmpBufSizeV2 | 获取矩阵乘法临时缓冲区大小V2 |

### 激活函数

#### SoftMax

| 接口 | 描述 |
|------|------|
| SoftMax | SoftMax函数 |
| SimpleSoftMax | 简化SoftMax |
| SoftmaxFlash | Flash SoftMax |
| SoftmaxGrad | SoftMax梯度 |
| SoftmaxFlashV2 | Flash SoftMax V2 |
| SoftmaxFlashV3 | Flash SoftMax V3 |
| SoftmaxGradFront | 前置SoftMax梯度 |
| AdjustSoftMaxRes | 调整SoftMax结果 |
| SoftMax Tiling | SoftMax Tiling |
| SoftMax/SimpleSoftMax Tiling | SoftMax/简化SoftMax Tiling |
| SoftmaxFlash Tiling | Flash SoftMax Tiling |
| SoftmaxGrad Tiling | SoftMax梯度Tiling |
| SoftmaxFlashV2 Tiling | Flash SoftMax V2 Tiling |
| SoftmaxFlashV3 Tiling | Flash SoftMax V3 Tiling |
| IsBasicBlockInSoftMax | 判断是否在SoftMax基本块中 |

#### LogSoftMax

| 接口 | 描述 |
|------|------|
| LogSoftMax | LogSoftMax函数 |
| LogSoftMax Tiling | LogSoftMax Tiling |

#### Gelu

| 接口 | 描述 |
|------|------|
| Gelu | Gelu激活函数 |
| FasterGelu | 快速Gelu |
| FasterGeluV2 | 快速Gelu V2 |
| GetGeluMaxMinTmpSize | 获取Gelu临时缓冲区大小 |

#### SwiGLU

| 接口 | 描述 |
|------|------|
| SwiGLU | SwiGLU激活函数 |
| GetSwiGLUMaxMinTmpSize | 获取SwiGLU临时缓冲区大小 |
| GetSwiGLUTmpBufferFactorSize | 获取SwiGLU临时缓冲区因子大小 |

#### Silu

| 接口 | 描述 |
|------|------|
| Silu | SiLU激活函数 |
| GetSiluTmpSize | 获取SiLU临时缓冲区大小 |

#### Swish

| 接口 | 描述 |
|------|------|
| Swish | Swish激活函数 |
| GetSwishTmpSize | 获取Swish临时缓冲区大小 |

#### GeGLU

| 接口 | 描述 |
|------|------|
| GeGLU | GeGLU激活函数 |
| GetGeGLUMaxMinTmpSize | 获取GeGLU临时缓冲区大小 |
| GetGeGLUTmpBufferFactorSize | 获取GeGLU临时缓冲区因子大小 |

#### ReGlu

| 接口 | 描述 |
|------|------|
| ReGlu | ReGLU激活函数 |
| GetReGluMaxMinTmpSize | 获取ReGLU临时缓冲区大小 |

#### Sigmoid

| 接口 | 描述 |
|------|------|
| Sigmoid | Sigmoid激活函数 |
| GetSigmoidMaxMinTmpSize | 获取Sigmoid临时缓冲区大小 |

### 数据归一化

| 接口 | 描述 |
|------|------|
| LayerNorm | LayerNorm归一化 |
| LayerNorm Tiling | LayerNorm Tiling |
| LayerNormGrad | LayerNorm梯度 |
| LayerNormGrad Tiling | LayerNorm梯度Tiling |
| LayerNormGradBeta | LayerNorm梯度Beta |
| LayerNormGradBeta Tiling | LayerNorm梯度Beta Tiling |
| RmsNorm | RmsNorm归一化 |
| RmsNorm Tiling | RmsNorm Tiling |
| BatchNorm | BatchNorm归一化 |
| BatchNorm Tiling | BatchNorm Tiling |
| DeepNorm | DeepNorm归一化 |
| DeepNorm Tiling | DeepNorm Tiling |
| Normalize | 归一化 |
| Normalize Tiling | 归一化Tiling |
| WelfordUpdate | Welford更新 |
| WelfordUpdate Tiling | Welford更新Tiling |
| WelfordFinalize | Welford完成 |
| WelfordFinalize Tiling | Welford完成Tiling |
| GroupNorm | GroupNorm归一化 |
| GroupNorm Tiling | GroupNorm Tiling |

### 量化反量化

| 接口 | 描述 |
|------|------|
| AscendQuant | Ascend量化 |
| GetAscendQuantMaxMinTmpSize | 获取量化临时缓冲区大小 |
| AscendDequant | Ascend反量化 |
| GetAscendDequantMaxMinTmpSize | 获取反量化临时缓冲区大小 |
| AscendAntiQuant | Ascend反量化 |
| GetAscendAntiQuantMaxMinTmpSize | 获取反量化临时缓冲区大小 |

### 归约操作

| 接口 | 描述 |
|------|------|
| Sum | 求和 |
| GetSumMaxMinTmpSize | 获取Sum临时缓冲区大小 |
| Mean | 均值 |
| GetMeanMaxMinTmpSize | 获取Mean临时缓冲区大小 |
| GetMeanTmpBufferFactorSize | 获取Mean临时缓冲区因子大小 |
| ReduceXorSum | 异或求和 |
| GetReduceXorSumMaxMinTmpSize | 获取异或求和临时缓冲区大小 |
| ReduceSum | 求和归约 |
| GetReduceSumMaxMinTmpSize | 获取求和归约临时缓冲区大小 |
| ReduceMean | 均值归约 |
| GetReduceMeanMaxMinTmpSize | 获取均值归约临时缓冲区大小 |
| ReduceMax | 最大值归约 |
| GetReduceMaxMaxMinTmpSize | 获取最大值归约临时缓冲区大小 |
| ReduceMin | 最小值归约 |
| GetReduceMinMaxMinTmpSize | 获取最小值归约临时缓冲区大小 |
| ReduceAny | 任意归约 |
| GetReduceAnyMaxMinTmpSize | 获取任意归约临时缓冲区大小 |
| ReduceAll | 全部归约 |
| GetReduceAllMaxMinTmpSize | 获取全部归约临时缓冲区大小 |
| ReduceProd | 乘积归约 |
| GetReduceProdMaxMinTmpSize | 获取乘积归约临时缓冲区大小 |

### 排序

| 接口 | 描述 |
|------|------|
| TopK | TopK排序 |
| TopK Tiling | TopK Tiling |
| Concat | 连接 |
| GetConcatTmpSize | 获取Concat临时缓冲区大小 |
| Extract | 提取 |
| Sort | 排序 |
| GetSortTmpSize | 获取Sort临时缓冲区大小 |
| GetSortOffset | 获取排序偏移 |
| GetSortLen | 获取排序长度 |
| MrgSort | 归并排序 |

### 数据填充

| 接口 | 描述 |
|------|------|
| Pad | 填充 |
| Pad Tiling | 填充Tiling |
| UnPad | 取消填充 |
| UnPad Tiling | 取消填充Tiling |
| Broadcast | 广播 |
| GetBroadCastMaxMinTmpSize | 获取Broadcast临时缓冲区大小 |

### 索引操作

| 接口 | 描述 |
|------|------|
| ArithProgression | 等差数列 |
| GetArithProgressionMaxMinTmpSize | 获取等差数列临时缓冲区大小 |

### 比较选择

| 接口 | 描述 |
|------|------|
| SelectWithBytesMask | 字节掩码选择 |
| GetSelectWithBytesMaskMaxMinTmpSize | 获取选择临时缓冲区大小 |

### 数据过滤

| 接口 | 描述 |
|------|------|
| DropOut | Dropout |
| GetDropOutMaxMinTmpSize | 获取Dropout临时缓冲区大小 |

### 变形

| 接口 | 描述 |
|------|------|
| ConfusionTranspose | 混淆转置 |
| ConfusionTranspose Tiling | 混淆转置Tiling |
| TransData | 数据转换 |
| GetTransDataMaxMinTmpSize | 获取数据转换临时缓冲区大小 |

### Hccl

#### Kernel侧接口

| 接口 | 描述 |
|------|------|
| InitV2 | 初始化V2 |
| SetCcTilingV2 | 设置CC Tiling V2 |
| AllReduce | 全归约 |
| AllGather | 全收集 |
| ReduceScatter | 归约分散 |
| AlltoAll | 全交换 |
| AlltoAllV | 全交换V |
| BatchWrite | 批量写入 |
| BatchRead | 批量读取 |
| Reduce | 归约 |
| Broadcast | 广播 |
| Scatter | 分散 |
| Gather | 聚集 |
| AlltoAllVC | 全交换VC |
| HcclFinalize | 结束 |

#### Tiling侧接口

| 接口 | 描述 |
|------|------|
| Hccl Tiling 类 | Hccl Tiling类 |
| GetTilingData | 获取Tiling数据 |
| SetRankTable | 设置等级表 |
| SetRankId | 设置等级ID |
| GetTilingBufSize | 获取Tiling缓冲区大小 |

---

## 附录：接口标记说明

- **(ISASI)**: 表示该接口为ISASI (Innner Software Adaptive Specific Instruction) 特定指令，仅在特定场景下使用
