#ifndef HEADER_ACLRTLAUNCH_KR_WAIT_AIV_H
#define HEADER_ACLRTLAUNCH_KR_WAIT_AIV_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_kr_wait_aiv(uint32_t numBlocks, aclrtStream stream, void* ctrGM, uint32_t index, uint32_t maxSpins);
#endif
