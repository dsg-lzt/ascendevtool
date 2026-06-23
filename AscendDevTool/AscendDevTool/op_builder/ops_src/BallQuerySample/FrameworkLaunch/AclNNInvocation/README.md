## 目录结构
```
├── AclNNInvocation
│   ├── inc
│   ├── input
│   ├── output
│   ├── scripts
│   │   ├── acl.json
│   │   ├── gen_data.py
│   │   └── verify_result.py
│   ├── src
│   │   ├── CMakeLists.txt
│   │   ├── common.cpp
│   │   ├── main.cpp
│   │   ├── op_runner.cpp
│   │   └── operator_desc.cpp
│   └── run.sh
```

## BallQuery 调用说明
本样例通过 aclnn 两段式接口调用 BallQuery。

调用形式：
```cpp
aclnnStatus aclnnBallQueryGetWorkspaceSize(
    const aclTensor *xyz,
    const aclTensor *center_xyz,
    float minRadius,
    float maxRadius,
    int64_t sampleNum,
    aclTensor *idx,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnBallQuery(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 样例输入输出
- 输入文件：
  - `input/input_xyz.bin`，shape=`[64, 3]`，dtype=`float16`
  - `input/input_center_xyz.bin`，shape=`[16, 3]`，dtype=`float16`
- 属性：
  - `min_radius=0.15`
  - `max_radius=0.45`
  - `sample_num=8`
- 输出文件：
  - `output/output.bin`，shape=`[16, 8]`，dtype=`int32`

## 运行
```bash
cd ${your_repo}/op_builder/ops_src/BallQuerySample/FrameworkLaunch/AclNNInvocation
bash run.sh
```
