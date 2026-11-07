#ifndef HEADER_ACLRTLAUNCH_KR_PUBLISH_AIC_H
#define HEADER_ACLRTLAUNCH_KR_PUBLISH_AIC_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_kr_publish_aic(uint32_t numBlocks, aclrtStream stream, void* ctrGM, uint32_t index);
#endif
