
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(pointnet2_3090_pointnet2_utils_three_nnTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(pointnet2_3090_pointnet2_utils_three_nn, pointnet2_3090_pointnet2_utils_three_nnTilingData)
}
