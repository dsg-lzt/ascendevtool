import os
import sys
import numpy as np

def verify_result(real_result, golden):
    real_result = np.fromfile(real_result, dtype=np.int32)
    golden = np.fromfile(golden, dtype=np.int32)

    if real_result.size != golden.size:
        print("[ERROR] result size mismatch, real: {}, golden: {}".format(real_result.size, golden.size))
        return False

    if not np.array_equal(real_result, golden):
        mismatch = np.where(real_result != golden)[0]
        print("[ERROR] result mismatch count: {}".format(mismatch.size))
        if mismatch.size > 0:
            first = mismatch[0]
            print("[ERROR] first mismatch at {}, real: {}, golden: {}".format(first, real_result[first], golden[first]))
        return False

    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1],sys.argv[2])
