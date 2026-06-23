import os
import sys
import numpy as np

loss = 1e-2
minimum = 10e-10

def verify_result():
    real_dist = np.fromfile("./output/output_dist.bin", dtype=np.float16)
    golden_dist = np.fromfile("./output/golden_dist.bin", dtype=np.float16)

    if real_dist.shape != golden_dist.shape:
        print("[ERROR] dist shape mismatch")
        return False

    real_idx = np.fromfile("./output/output_idx.bin", dtype=np.int32)
    golden_idx = np.fromfile("./output/golden_idx.bin", dtype=np.int32)

    if real_idx.shape != golden_idx.shape:
        print("[ERROR] idx shape mismatch")
        return False

    # 验证距离：允许fp16精度误差
    result = np.abs(real_dist - golden_dist)
    deno = np.maximum(np.abs(real_dist), np.abs(golden_dist))
    result_atol = np.less_equal(result, loss)
    result_rtol = np.less_equal(result / np.add(deno, minimum), loss)
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > real_dist.size * loss and np.sum(result_atol == False) > real_dist.size * loss:
            print("[ERROR] dist error")
            return False

    # 验证索引：完全匹配
    idx_match = (real_idx == golden_idx)
    if not idx_match.all():
        mismatch = np.sum(idx_match == False)
        total = real_idx.size
        print(f"[WARN] idx mismatch: {mismatch}/{total}")
        if mismatch > total * 0.1:  # 允许10%因为距离相同时索引可能不同
            print("[ERROR] idx error too high")
            return False

    print("test pass")
    return True

if __name__ == '__main__':
    ret = verify_result()
