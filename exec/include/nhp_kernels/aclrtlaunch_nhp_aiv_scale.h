#ifndef HEADER_ACLRTLAUNCH_NHP_AIV_SCALE_H
#define HEADER_ACLRTLAUNCH_NHP_AIV_SCALE_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_nhp_aiv_scale(uint32_t numBlocks, aclrtStream stream, void* inGM, void* outGM, uint32_t elems, uint32_t iters);
#endif
