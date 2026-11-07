#ifndef HEADER_ACLRTLAUNCH_NHP_AIV_DELAY_PUBLISH_H
#define HEADER_ACLRTLAUNCH_NHP_AIV_DELAY_PUBLISH_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_nhp_aiv_delay_publish(uint32_t numBlocks, aclrtStream stream, void* flagGM, uint32_t index, uint32_t delayIters);
#endif
