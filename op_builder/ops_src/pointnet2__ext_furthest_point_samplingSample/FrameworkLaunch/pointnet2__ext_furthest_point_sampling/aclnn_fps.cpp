/* Hand-written ACLNN wrapper for pointnet2__ext_furthest_point_sampling */
#include <string.h>
#include "graph/types.h"
#include "aclnn_pointnet2__ext_furthest_point_sampling.h"

namespace {
typedef struct { uint32_t id; const char *funcName; bool hasReg; } NnopbaseDfxId;
typedef struct { ge::DataType dtype; ge::Format format; } TensorDesc;
typedef struct { TensorDesc *iDesc; size_t iNum; TensorDesc *oDesc; size_t oNum; } SupportInfo;
typedef struct { SupportInfo *sInfo; size_t num; } OpSocSupportInfo;
typedef struct { OpSocSupportInfo *ssInfo; size_t num; } OpSupportList;
enum SocType { SOC_VERSION_ASCEND910A=1,SOC_VERSION_ASCEND910B,SOC_VERSION_ASCEND310P=5 };
uint32_t socSupportList[]={SOC_VERSION_ASCEND310P};
uint32_t socSupportListLen=1;

TensorDesc i0_0[1]={{ge::DT_FLOAT,ge::FORMAT_ND}};
TensorDesc i0_1[1]={{ge::DT_FLOAT16,ge::FORMAT_ND}};
TensorDesc o0_0[1]={{ge::DT_INT32,ge::FORMAT_ND}};
TensorDesc o0_1[1]={{ge::DT_INT32,ge::FORMAT_ND}};
SupportInfo l0_0={i0_0,1,o0_0,1};
SupportInfo l0_1={i0_1,1,o0_1,1};
SupportInfo s0[2]={l0_0,l0_1};
OpSocSupportInfo socSupportInfo0={s0,2};
OpSocSupportInfo opSocSupportList[1]={socSupportInfo0};
OpSupportList supportList={opSocSupportList,1};
[[maybe_unused]] uint32_t N_ops=0U;

extern void NnopbaseOpLogE(const aclnnStatus code, const char *const expr);
#define NNOPBASE_ASSERT_OK_RETVAL(v) do{const aclnnStatus _s=(v);if(_s!=ACLNN_SUCCESS){NnopbaseOpLogE(_s,#v);return _s;}}while(0)
#define NNOPBASE_ASSERT_OK(v) do{aclnnStatus _s=(v);NnopbaseOpLogE(_s,#v);}while(0)

extern void *NnopbaseInit(int32_t deviceId);
extern void *NnopbaseGetExecutor(void*,const char*,char*,size_t,char*,size_t,void*,size_t);
extern aclnnStatus NnopbaseSetWorkspaceSize(void*,uint64_t*);
extern aclnnStatus NnopbaseLaunch(void*,void*,uint64_t,aclrtStream);
extern aclnnStatus NnopbaseSetInputsAndOutputs(void*,struct aclTensor*const*,size_t,struct aclTensor*const*,size_t);
extern aclnnStatus NnopbaseAddSupportList(void*,OpSupportList*,uint32_t*,size_t);

enum { kInt=2 };
typedef struct { int32_t dtype; const void *value; int64_t valueSize; const char *name; } NAttr;
} // namespace

aclnnStatus aclnnpointnet2__ext_furthest_point_samplingGetWorkspaceSize(
    const aclTensor *x, const aclTensor *out, int64_t npoint,
    uint64_t *workspaceSize, aclOpExecutor **executor) {
    void *es = NnopbaseInit(0);
    char iD[]={1}, oD[]={1};
    char aV[16]={0}; memcpy(aV,&npoint,sizeof(int64_t));
    NAttr aA[]={{kInt,aV,sizeof(int64_t),"npoint"}};
    void *ne = NnopbaseGetExecutor(es,"pointnet2__ext_furthest_point_sampling",iD,1,oD,1,aA,1);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddSupportList(ne,&supportList,socSupportList,socSupportListLen));
    const aclTensor *ins[1]={x}, *outs[1]={out};
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseSetInputsAndOutputs(ne,(aclTensor**)ins,1,(aclTensor**)outs,1));
    NNOPBASE_ASSERT_OK(NnopbaseSetWorkspaceSize(ne,workspaceSize));
    *executor = (aclOpExecutor*)ne;
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnpointnet2__ext_furthest_point_sampling(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    return NnopbaseLaunch(executor, workspace, workspaceSize, stream);
}
