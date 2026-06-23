## 概述
本目录用于通过 FrameworkLaunch 方式调用 BallQuery 自定义算子。

## 目录结构介绍
```
├── FrameworkLaunch
│   ├── AclNNInvocation       // 通过 aclnn 调用 BallQuery
│   ├── FastGelu              // 算子工程目录（当前目录名沿用模板）
│   ├── BallQuery.json        // BallQuery 算子原型定义（推荐）
│   └── FastGelu.json         // 兼容文件，内容已切换为 BallQuery 定义
```

## 算子工程说明
当前算子工程目录名为 FastGelu，但 FrameworkLaunch 调用侧已经按 BallQuery 的输入输出与属性进行适配。

BallQuery 规格（ball_query 分支）：
- 输入：xyz(float16/float32, ND)、center_xyz(float16/float32, ND)
- 属性：min_radius(float)、max_radius(float)、sample_num(int)
- 输出：idx(int32, ND)

## 编译与运行
### 1. 编译算子工程
在算子工程目录中完成编译与安装（与工程内脚本保持一致）。

### 2. 运行 AclNNInvocation
```bash
cd ${your_repo}/op_builder/ops_src/BallQuerySample/FrameworkLaunch/AclNNInvocation
bash run.sh
```

执行流程：
1. 生成 `input/input_xyz.bin`、`input/input_center_xyz.bin` 与 `output/golden.bin`
2. 编译并执行 `execute_op`
3. 将 `output/output.bin` 与 `output/golden.bin` 做精度比对
