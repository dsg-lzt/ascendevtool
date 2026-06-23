# 开发最佳实践

> Ascend C 算子开发经验总结

## 目录

- [开发流程](#开发流程)
- [性能优化](#性能优化)
- [常见问题与解决方案](#常见问题与解决方案)
- [调试技巧](#调试技巧)
- [代码规范](#代码规范)

---

## 开发流程

### 1. 需求分析

- 确定输入输出张量形状和数据类型
- 分析计算复杂度，确定是否需要多核
- 评估内存需求

### 2. Tiling设计

```cpp
// 设计思路
struct OpTiling {
    uint32_t blockDim;      // 使用多少个AI Core
    uint32_t repeatTimes;   // 每个Core处理多少数据
    uint32_t srcStride;     // 数据步长
};

// 原则：
// - blockDim * repeatTimes >= totalLength
// - 尽量让blockDim整除totalLength
// - 考虑UB大小限制
```

### 3. 内存规划

```cpp
// UB内存使用估算
// 1. 输入张量
uint32_t inputSize = size * sizeof(DT);
// 2. 输出张量
uint32_t outputSize = size * sizeof(DT);
// 3. 临时缓冲区 (高阶API需要)
uint32_t tmpSize = GetFuncTmpSize();
// 4. 队列缓冲
uint32_t queueSize = queSize * size * sizeof(DT);

// 总计应小于UB大小 (通常约512KB)
```

### 4. 代码实现

```cpp
// 标准结构
__CCEATTR__ void KernelOp(KernelContext* ctx, void* param) {
    GET_TILING_DATA(tiling);
    
    // 1. 初始化
    TPipe pipe;
    pipe.Init();
    
    TQue<QueSize, TPosition::POS_LOCAL> que;
    que.InitBuffer();
    
    // 2. 分配张量
    auto in = que.AllocTensor<DT>();
    auto out = que.AllocTensor<DT>();
    auto tmp = que.AllocTensor<DT>();  // 如果需要
    
    // 3. 数据搬运 GM->UB
    DataCopy(in, ctx->input, size);
    
    // 4. 计算
    Compute(out, in, tmp, params);
    
    // 5. 数据搬运 UB->GM
    DataCopy(ctx->output, out, size);
    
    // 6. 释放资源
    que.FreeTensor(in);
    que.FreeTensor(out);
    if (tmp) que.FreeTensor(tmp);
    
    pipe.Destroy();
}
```

### 5. 测试验证

- 功能测试：验证计算正确性
- 性能测试：对比CPU/GPU性能
- 边界测试：处理各种输入边界情况

---

## 性能优化

### 1. 减少数据搬运

```cpp
// ❌ 不好：多次搬运
DataCopy(tmp1, src, size);
Exp(tmp2, tmp1, repeat);
DataCopy(tmp3, tmp2, size);
Relu(dst, tmp3, repeat);

// ✅ 好：尽量减少中间张量
Exp(tmp1, src, repeat);
Relu(dst, tmp1, repeat);
```

### 2. 合理使用双缓冲

```cpp
// 计算和数据搬运并行
TQueBind<2, TPosition::POS_LOCAL> que;

// 第一批
auto in0 = que.AllocTensor(TPosition::POS_PING);
DataCopy(in0, globalSrc, size);
Exp(out0, in0, repeat);
que.EnQue(pipe, TPosition::POS_PING);

// 第二批 (与第一批并行)
auto in1 = que.AllocTensor(TPosition::POS_PONG);
DataCopy(in1, globalSrc + offset, size);  // 可以在第一批计算时进行
Exp(out1, in1, repeat);

// 等待第一批完成
auto result0 = que.DeQue(pipe, TPosition::POS_PING);
DataCopy(globalDst, result0, size);
```

### 3. 使用向量化

```cpp
// ❌ 不好：逐元素处理
for (int i = 0; i < length; i++) {
    dst[i] = src[i] * 2.0f;
}

// ✅ 好：矢量指令
Muls(dst, src, 2.0f, repeat);  // 自动向量化
```

### 4. 减少分支

```cpp
// ❌ 不好：分支预测失败
if (condition) {
    Exp(dst, src, repeat);
} else {
    Ln(dst, src, repeat);
}

// ✅ 好：使用掩码
SetVectorMask(condition ? 0xFFFFFFFFFFFFFFFF : 0);
Exp(dst, src, repeat);
ResetMask();
```

### 5. 内存对齐

```cpp
// 确保数据对齐
alignas(64) float buffer[1024];  // 64字节对齐

// 使用对齐的地址
uint64_t addr = (uint64_t)buffer;
addr = (addr + 63) & ~63;  // 64字节对齐
```

### 6. 合并归约

```cpp
// ❌ 不好：多次归约
ReduceSum(tmp, src, repeat, stride);
BlockReduceSum(result, tmp, 1);

// ✅ 好：一次归约
ReduceSum(result, src, repeat, stride);  // 使用WholeReduce
```

---

## 常见问题与解决方案

### 1. UB内存不足

**问题**：运行时提示UB内存不足

**解决**：
- 减少队列大小
- 分批处理数据
- 减少临时缓冲区
- 优化Tiling参数

```cpp
// 分批处理
uint32_t batchSize = 256 * 32;  // 根据UB大小调整
for (uint32_t offset = 0; offset < totalSize; offset += batchSize) {
    DataCopy(localBuf, globalSrc + offset, batchSize);
    Compute(localDst, localBuf, repeat);
    DataCopy(globalDst + offset, localDst, batchSize);
}
```

### 2. repeatTimes计算错误

**问题**：数据处理不完整

**解决**：
```cpp
// 计算repeatTimes
uint32_t repeat = (length + 31) / 32;  // float32
uint32_t tail = length % 32;
if (tail > 0) {
    // 处理尾部数据
}
```

### 3. 数据越界

**问题**：访问越界

**解决**：
- 检查张量大小
- 确保偏移量正确
- 使用边界检查宏（调试时）

### 4. 多核同步问题

**问题**：多核结果不正确

**解决**：
```cpp
// 确保使用正确的同步机制
if (GetBlockIdx() == 0) {
    // 主核
    IBSet(1);  // 通知其他核
} else {
    IBWait(1);  // 等待主核
}

// 最后同步
SyncAll();
```

### 5. 类型不匹配

**问题**：编译错误或结果异常

**解决**：
- 确保源和目标类型一致
- 使用Cast进行类型转换
- 注意量化/反量化

---

## 调试技巧

### 1. 打印调试信息

```cpp
// 打印张量
tensor.Print();

// 打印值
printf("blockIdx=%d, blockNum=%d\n", GetBlockIdx(), GetBlockNum());

// 写入文件
tensor.ToFile("/home/debug.bin");
```

### 2. 使用单核调试

```cpp
// 在Tiling中强制使用单核
void GetOpTiling(OpTiling& tiling, uint32_t length) {
    tiling.blockDim = 1;  // 强制单核
    tiling.repeatTimes = length / 32;
}
```

### 3. 分步验证

```cpp
// 1. 先验证数据搬运
DataCopy(localBuf, globalSrc, size);
DataCopy(verifyBuf, localBuf, size);  // 验证是否正确

// 2. 再验证计算
Exp(out, in, repeat);
// 与CPU结果对比

// 3. 最后验证多核
// 分阶段启用多核
```

### 4. 内存检查

```cpp
// 检查UB使用
void* start = GetTPipePtr()->GetBaseAddr();
void* end = start + UB_SIZE;
assert((void*)tensor.GetPhyAddr() >= start);
assert((void*)tensor.GetPhyAddr() < end);
```

---

## 代码规范

### 1. 命名规范

```cpp
// 类/结构体：大驼峰
class TensorDescriptor {};
struct MatmulTiling {};

// 函数/变量：小驼峰
void GetTilingData();
uint32_t blockDim;

// 常量：全大写下划线
constexpr uint32_t MAX_REPEAT = 1024;
```

### 2. 注释规范

```cpp
/**
 * @brief 执行SoftMax计算
 * @param dst 输出张量
 * @param src 输入张量
 * @param tmp 临时缓冲区
 * @param length 数据长度
 */
void SoftMax(LocalTensor<DT> dst, LocalTensor<DT> src, ...);
```

### 3. 错误处理

```cpp
// 检查返回值
if (result == nullptr) {
    return ERR_NOMEM;
}

// 检查状态
if (!que.HasTensorInQue()) {
    return ERR_QUEUE_EMPTY;
}
```

### 4. 资源管理

```cpp
// 原则：谁分配谁释放
// 1. 分配
auto tensor = que.AllocTensor<DT>();
// 2. 使用
Exp(dst, src, repeat);
// 3. 释放
que.FreeTensor(tensor);

// 使用RAII模式
struct TensorGuard {
    TQue& que;
    LocalTensor<DT> tensor;
    ~TensorGuard() { que.FreeTensor(tensor); }
};
```

---

## 性能分析工具

### 1. Profiling

```bash
# 使用msprof进行性能分析
msprof --output=/home/profile \
       --kernel=YourKernelName \
       ./your_app
```

### 2. 关键指标

| 指标 | 说明 | 优化方向 |
|------|------|----------|
| 计算时间 | Kernel执行时间 | 减少计算量 |
| 搬运时间 | DataCopy时间 | 合并搬运 |
| 等待时间 | 同步等待时间 | 优化并行 |
| UB使用率 | 内存利用率 | 调整Tiling |

### 3. 优化检查清单

- [ ] 是否使用向量化指令
- [ ] 是否减少数据搬运
- [ ] 是否使用双缓冲
- [ ] Tiling是否合理
- [ ] 是否有不必要的同步
- [ ] 内存是否对齐
