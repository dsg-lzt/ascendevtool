/*
 * AscendDevTool: manually written ACLNN wrapper for FPS operator
 */

#ifndef ACLNN_POINTNET2__EXT_FURTHEST_POINT_SAMPLING_H_
#define ACLNN_POINTNET2__EXT_FURTHEST_POINT_SAMPLING_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnpointnet2__ext_furthest_point_samplingGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *out,
    int64_t npoint,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnpointnet2__ext_furthest_point_sampling(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
