#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os

def gen_golden_data():
    """
    ThreeNN: 对每个query点(xyz2)，在参考点集(xyz1)中找3个最近邻
    """
    B, N, M = 1, 128, 64

    xyz1 = np.random.uniform(-1, 1, [B, N, 3]).astype(np.float16)
    xyz2 = np.random.uniform(-1, 1, [B, M, 3]).astype(np.float16)

    dist_out = np.zeros([B, M, 3], dtype=np.float16)
    idx_out = np.zeros([B, M, 3], dtype=np.int32)

    for b in range(B):
        for m in range(M):
            query = xyz2[b, m, :]
            dists = np.sum((xyz1[b] - query) ** 2, axis=1)

            top3_idx = np.argsort(dists)[:3]
            idx_out[b, m, :] = top3_idx.astype(np.int32)
            dist_out[b, m, :] = dists[top3_idx].astype(np.float16)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    xyz1.tofile("./input/input_xyz1.bin")
    xyz2.tofile("./input/input_xyz2.bin")
    dist_out.tofile("./output/golden_dist.bin")
    idx_out.tofile("./output/golden_idx.bin")

if __name__ == "__main__":
    gen_golden_data()
