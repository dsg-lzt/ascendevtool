## Overview
This directory demonstrates BallQuery custom operator invocation through FrameworkLaunch.

## Directory Structure
```
├── FrameworkLaunch
│   ├── AclNNInvocation       // Invoke BallQuery via aclnn
│   ├── FastGelu              // Operator project directory (template folder name kept)
│   ├── BallQuery.json        // Recommended BallQuery prototype definition
│   └── FastGelu.json         // Compatibility file, content switched to BallQuery
```

## Operator Contract
The invocation side is adapted to BallQuery with the following contract:
- Inputs: `xyz` (float16/float32, ND), `center_xyz` (float16/float32, ND)
- Attributes: `min_radius` (float), `max_radius` (float), `sample_num` (int)
- Output: `idx` (int32, ND)

## Build and Run
### 1. Build operator project
Build and install the operator package using scripts in the operator project directory.

### 2. Run AclNNInvocation
```bash
cd ${your_repo}/op_builder/ops_src/BallQuerySample/FrameworkLaunch/AclNNInvocation
bash run.sh
```

Execution flow:
1. Generate `input/input_xyz.bin`, `input/input_center_xyz.bin`, and `output/golden.bin`
2. Build and run `execute_op`
3. Compare `output/output.bin` with `output/golden.bin`
