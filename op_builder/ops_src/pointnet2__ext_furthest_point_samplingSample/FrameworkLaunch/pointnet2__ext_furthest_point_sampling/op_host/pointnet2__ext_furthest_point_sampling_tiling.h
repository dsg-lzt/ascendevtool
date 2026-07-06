
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FpsCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, B);
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, M);
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);
  TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FurthestPointSampling, FpsCustomTilingData)
}
