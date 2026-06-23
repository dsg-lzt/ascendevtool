#ifndef _THREE_NN_TILING_H_
#define _THREE_NN_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling
{
  enum class ThreeNNTilingKey : int64_t {
    FP16_NORM = 10000001,
    FP32_NORM = 10000002,
  };

  BEGIN_TILING_DATA_DEF(ThreeNNTilingData)
  TILING_DATA_FIELD_DEF(int64_t, tilingKey);
  TILING_DATA_FIELD_DEF(uint32_t, B);            // 批次数
  TILING_DATA_FIELD_DEF(uint32_t, N);            // 参考点数量
  TILING_DATA_FIELD_DEF(uint32_t, M);            // 查询点数量
  TILING_DATA_FIELD_DEF(uint32_t, N_block);      // 每次处理的参考点块大小
  TILING_DATA_FIELD_DEF(uint32_t, alignNum);     // 32Byte对齐下的元素数
  TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);  // 使用的AI核数
  TILING_DATA_FIELD_DEF(uint64_t, UBSize);       // UB内存大小
  TILING_DATA_FIELD_DEF(uint32_t, totalQuery);   // 总查询点数 B*M
  TILING_DATA_FIELD_DEF(uint32_t, coreQuerySize);// 每核处理查询点数
  TILING_DATA_FIELD_DEF(uint32_t, coreRemain);   // 尾块查询点
  TILING_DATA_FIELD_DEF(uint32_t, formatB3N);    // 0: (B,N,3), 1: (B,3,N)
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(ThreeNN, ThreeNNTilingData)
}

#endif  // _THREE_NN_TILING_H_
