# ThreeNN 自定义算子样例

## 简介

ThreeNN（Three Nearest Neighbors）是点云处理中的常见算子，用于在参考点集(xyz1)中为每个查询点(xyz2)找到距离最近的3个邻居点及其索引。

本样例基于Ascend C开发，提供框架式调用（aclnn）的完整实现。

### 算子描述

```
输入:
  xyz1: (B, N, 3)  float16/float32  参考点云坐标
  xyz2: (B, M, 3)  float16/float32  查询点云坐标

输出:
  dist: (B, M, 3)  float16/float32  最近3邻距离（欧氏距离平方）
  idx:  (B, M, 3)  int32            最近3邻索引
```

## 目录结构

```
ThreeNNSample/
├── FrameworkLaunch/
│   ├── ThreeNN.json              # 算子原型定义
│   ├── ThreeNN/                  # 算子项目
│   │   ├── op_host/              # Host侧: 注册 + Tiling
│   │   ├── op_kernel/            # Device侧: Kernel实现
│   │   ├── framework/            # 框架插件 (TensorFlow)
│   │   ├── cmake/                # 构建脚本
│   │   └── CMakeLists.txt
│   └── AclNNInvocation/          # ACLNN调用示例
│       ├── src/                  # C++ 调用源码
│       ├── inc/                  # 头文件
│       ├── scripts/              # 数据生成与验证
│       └── run.sh               # 运行脚本
```

## 支持平台

- Atlas A2 训练系列（Ascend 910B）
- Atlas 200/500 A2（Ascend 310B）

## 编译与运行

详细步骤请参考FrameworkLaunch内各目录的README。
