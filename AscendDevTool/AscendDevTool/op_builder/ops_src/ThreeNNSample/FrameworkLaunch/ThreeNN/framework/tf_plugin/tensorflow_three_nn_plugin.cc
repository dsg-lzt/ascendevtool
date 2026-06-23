#include "register/register.h"
namespace domi {
REGISTER_CUSTOM_OP("ThreeNN")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ThreeNN")
    .ParseParamsByOperatorFn(AutoMappingByOpFn);
}
