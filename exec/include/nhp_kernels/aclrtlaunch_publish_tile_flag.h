#ifndef HEADER_ACLRTLAUNCH_PUBLISH_TILE_FLAG_H
#define HEADER_ACLRTLAUNCH_PUBLISH_TILE_FLAG_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_publish_tile_flag(uint32_t numBlocks, aclrtStream stream, void* flagsGM, uint32_t tile);
#endif
