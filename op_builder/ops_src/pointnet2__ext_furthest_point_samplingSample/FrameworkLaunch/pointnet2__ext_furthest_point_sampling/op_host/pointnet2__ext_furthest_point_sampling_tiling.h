
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(pointnet2__ext_furthest_point_samplingTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(pointnet2__ext_furthest_point_sampling, pointnet2__ext_furthest_point_samplingTilingData)
}
