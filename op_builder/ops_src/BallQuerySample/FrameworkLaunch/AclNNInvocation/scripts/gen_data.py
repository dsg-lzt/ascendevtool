#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os


def ball_query_reference(xyz, center_xyz, min_radius, max_radius, sample_num):
    min_r2 = min_radius * min_radius
    max_r2 = max_radius * max_radius
    result = np.full((center_xyz.shape[0], sample_num), -1, dtype=np.int32)

    for i in range(center_xyz.shape[0]):
        diff = xyz - center_xyz[i]
        dist2 = np.sum(diff * diff, axis=1)
        candidates = np.where((dist2 >= min_r2) & (dist2 <= max_r2))[0].astype(np.int32)
        if candidates.size == 0:
            continue

        if candidates.size >= sample_num:
            result[i, :] = candidates[:sample_num]
        else:
            result[i, :candidates.size] = candidates
            result[i, candidates.size:] = candidates[0]

    return result


def gen_golden_data_simple():
    np.random.seed(2026)

    xyz = np.random.uniform(-1.0, 1.0, [64, 3]).astype(np.float16)
    center_xyz = np.random.uniform(-0.5, 0.5, [16, 3]).astype(np.float16)

    min_radius = 0.15
    max_radius = 0.45
    sample_num = 8

    golden = ball_query_reference(
        xyz.astype(np.float32),
        center_xyz.astype(np.float32),
        min_radius,
        max_radius,
        sample_num,
    )

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    xyz.tofile("./input/input_xyz.bin")
    center_xyz.tofile("./input/input_center_xyz.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
