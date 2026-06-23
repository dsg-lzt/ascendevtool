# ThreeNN Custom Operator Sample

## Introduction

ThreeNN (Three Nearest Neighbors) is a common operator for point cloud processing. For each query point (xyz2), it finds the 3 nearest neighbors in the reference point set (xyz1), returning their distances and indices.

### Operator Description

```
Input:
  xyz1: (B, N, 3)  float16/float32  Reference point cloud
  xyz2: (B, M, 3)  float16/float32  Query point cloud

Output:
  dist: (B, M, 3)  float16/float32  Top-3 nearest distances (squared Euclidean)
  idx:  (B, M, 3)  int32            Top-3 nearest indices
```

## Supported Platforms

- Atlas A2 Training Series (Ascend 910B)
- Atlas 200/500 A2 (Ascend 310B)

## Build & Run

Please refer to README files in each subdirectory under FrameworkLaunch.
