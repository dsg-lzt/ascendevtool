## Directory
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

## BallQuery Invocation
This sample uses the aclnn two-phase API to run BallQuery.

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

## Sample Input/Output Contract
- Input files:
  - `input/input_xyz.bin`, shape=`[64, 3]`, dtype=`float16`
  - `input/input_center_xyz.bin`, shape=`[16, 3]`, dtype=`float16`
- Attributes:
  - `min_radius=0.15`
  - `max_radius=0.45`
  - `sample_num=8`
- Output file:
  - `output/output.bin`, shape=`[16, 8]`, dtype=`int32`

## Run
```bash
cd ${your_repo}/op_builder/ops_src/BallQuerySample/FrameworkLaunch/AclNNInvocation
bash run.sh
```
