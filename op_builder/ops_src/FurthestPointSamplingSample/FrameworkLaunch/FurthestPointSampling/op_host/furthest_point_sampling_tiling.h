
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FurthestPointSamplingTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, B);
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, M);
  TILING_DATA_FIELD_DEF(uint32_t, dataTypeLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileN);
  TILING_DATA_FIELD_DEF(uint32_t, tileLoopNum);
  TILING_DATA_FIELD_DEF(uint32_t, batchPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, coreRem);
  TILING_DATA_FIELD_DEF(float,    initVal);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FurthestPointSampling, FurthestPointSamplingTilingData)
}
