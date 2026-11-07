#ifndef HEADER_ACLRTLAUNCH_NHP_AIC_SPIN_H
#define HEADER_ACLRTLAUNCH_NHP_AIC_SPIN_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_nhp_aic_spin(uint32_t numBlocks, aclrtStream stream, void* flagGM, uint32_t index, void* statusGM, uint32_t noWait);
#endif
