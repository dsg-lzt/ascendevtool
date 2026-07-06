#include <string.h>
#include "graph/types.h"
#include "aclnn_pointnet2__ext_furthest_point_sampling.h"

namespace {
typedef struct { uint32_t id; const char *funcName; bool hasReg; } NnopbaseDfxId;
typedef struct { ge::DataType dtype; ge::Format format; } TensorDesc;
typedef struct { TensorDesc *inputsDesc; size_t inputsNum; TensorDesc *outputsDesc; size_t outputsNum; } SupportInfo;
typedef struct { SupportInfo *supportInfo; size_t num; } OpSocSupportInfo;
typedef struct { OpSocSupportInfo *socSupportInfo; size_t num; } OpSupportList;
enum SocType {
    SOC_VERSION_ASCEND910A = 1, SOC_VERSION_ASCEND910B, SOC_VERSION_ASCEND910_93,
    SOC_VERSION_ASCEND910_95, SOC_VERSION_ASCEND310P, SOC_VERSION_ASCEND310B,
    SOC_VERSION_BS9SX1A, SOC_VERSION_ASCEND610Lite, SOC_VERSION_ASCEND910_55,
    SOC_VERSION_MC61AM21A, SOC_VERSION_MC62CM12A, SOC_VERSION_BS9SX2A, SOC_VERSION_ASCEND910_96
};
uint32_t socSupportList[] = {SOC_VERSION_ASCEND310P, SOC_VERSION_ASCEND910B};
uint32_t socSupportListLen = 2;

TensorDesc inputDesc0_0[1] = {{ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc0_1[1] = {{ge::DT_FLOAT16, ge::FORMAT_ND}};
TensorDesc outputDesc0_0[1] = {{ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_1[1] = {{ge::DT_FLOAT16, ge::FORMAT_ND}};
SupportInfo list0_0 = {inputDesc0_0, 1, outputDesc0_0, 1};
SupportInfo list0_1 = {inputDesc0_1, 1, outputDesc0_1, 1};
SupportInfo supportInfo0[2] = {list0_0, list0_1};
OpSocSupportInfo socSupportInfo0 = {supportInfo0, 2};

TensorDesc inputDesc1_0[1] = {{ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc1_1[1] = {{ge::DT_FLOAT16, ge::FORMAT_ND}};
TensorDesc outputDesc1_0[1] = {{ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1_1[1] = {{ge::DT_FLOAT16, ge::FORMAT_ND}};
SupportInfo list1_0 = {inputDesc1_0, 1, outputDesc1_0, 1};
SupportInfo list1_1 = {inputDesc1_1, 1, outputDesc1_1, 1};
SupportInfo supportInfo1[2] = {list1_0, list1_1};
OpSocSupportInfo socSupportInfo1 = {supportInfo1, 2};

OpSocSupportInfo opSocSupportList[2] = {socSupportInfo0, socSupportInfo1};
OpSupportList supportList = {opSocSupportList, 2};
[[maybe_unused]] uint32_t NNOPBASE_pointnet2__ext_furthest_point_sampling = 0U;
} // namespace

extern void NnopbaseOpLogE(const aclnnStatus code, const char *const expr);

#define NNOPBASE_ASSERT_OK(expr)                             \
    do {                                                     \
        aclnnStatus code = (expr);                           \
        NnopbaseOpLogE(code, #expr);                         \
    } while (0)
#define NNOPBASE_ASSERT_OK_RETVAL(expr)                      \
    do {                                                     \
        aclnnStatus code = (expr);                           \
        if (code != ACLNN_SUCCESS) {                         \
            NnopbaseOpLogE(code, #expr);                     \
            return code;                                     \
        }                                                    \
    } while (0)

enum NnopbaseAttrDtype {
    kNnopbaseBool = 0U, kNnopbaseFloat, kNnopbaseInt,
    kNnopbaseString, kNnopbaseAttrEnd
};
typedef struct TagNnopbaseAttr {                   \
    int32_t dtype;                                 \
    const void *value;                             \
    int64_t valueSize;                             \
    const char *name;                              \
} NnopbaseAttr;

extern void *NnopbaseInit(int32_t deviceId);
extern void *NnopbaseGetExecutor(void *executorSpace, const char *opType, char *inputDescs, size_t inputDescsNum,
                                 char *outputDescs, size_t outputDescsNum, NnopbaseAttr *attrDescs, size_t attrDescsNum);
extern aclnnStatus NnopbaseSetWorkspaceSize(void *executor, uint64_t *workspaceSize);
extern aclnnStatus NnopbaseLaunch(void *executor, void *workspace, uint64_t workspaceSize, aclrtStream stream);
extern aclnnStatus NnopbaseSetInputsAndOutputs(void *executor, struct aclTensor *const *inputs, size_t inputNum,
                                                struct aclTensor *const *output, size_t outputNum);
extern aclnnStatus NnopbaseExecute(void *executor, struct aclTensor *const *inputs, size_t inputNum,
                                    struct aclTensor *const *output, size_t outputNum, void *workspace,
                                    uint64_t workspaceSize, aclrtStream stream);
extern aclnnStatus NnopbaseAddSupportList(void *executor, OpSupportList *list, uint32_t *socSupportList, size_t socSupportListLen);

aclnnStatus aclnnpointnet2__ext_furthest_point_samplingGetWorkspaceSize(
    const aclTensor *x, const aclTensor *out, int64_t npoint,
    uint64_t *workspaceSize, aclOpExecutor **executor) {
    void *executorSpace = NnopbaseInit(0);
    char inputDesc[] = {1};
    char outputDesc[] = {1};
    char attrValue[16] = {0};
    memcpy(attrValue, &npoint, sizeof(int64_t));
    NnopbaseAttr attrArray[] = {{kNnopbaseInt, attrValue, sizeof(int64_t), "npoint"}};
    void *nnopExecutor = NnopbaseGetExecutor(executorSpace,
        "pointnet2__ext_furthest_point_sampling",
        inputDesc, sizeof(inputDesc) / sizeof(char),
        outputDesc, sizeof(outputDesc) / sizeof(char),
        attrArray, sizeof(attrArray) / sizeof(NnopbaseAttr));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddSupportList(nnopExecutor, &supportList, socSupportList, socSupportListLen));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseSetInputsAndOutputs(nnopExecutor, &x, 1, &out, 1));
    NNOPBASE_ASSERT_OK(NnopbaseSetWorkspaceSize(nnopExecutor, workspaceSize));
    *executor = (aclOpExecutor *)nnopExecutor;
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnpointnet2__ext_furthest_point_sampling(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    return NnopbaseLaunch(executor, workspace, workspaceSize, stream);
}
