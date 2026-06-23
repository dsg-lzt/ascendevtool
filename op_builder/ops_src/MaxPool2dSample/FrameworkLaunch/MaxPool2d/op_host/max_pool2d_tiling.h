#include "register/tilingdata_base.h"

namespace optiling
{
  BEGIN_TILING_DATA_DEF(MaxPool2dTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, c1);
  TILING_DATA_FIELD_DEF(uint32_t, h);
  TILING_DATA_FIELD_DEF(uint32_t, w);
  TILING_DATA_FIELD_DEF(uint32_t, c0);
  TILING_DATA_FIELD_DEF(uint32_t, out_h);
  TILING_DATA_FIELD_DEF(uint32_t, out_w);
  TILING_DATA_FIELD_DEF(uint32_t, row_total);
  TILING_DATA_FIELD_DEF(uint32_t, out_total);
  TILING_DATA_FIELD_DEF(uint32_t, tile_ow);
  TILING_DATA_FIELD_DEF(uint32_t, core_row_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_row_remain);
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(MaxPool2d, MaxPool2dTilingData)
}
