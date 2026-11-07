#pragma once
#include <stdint.h>

struct GateArgs {
    uint64_t flagAddress;
    uint64_t statusAddress;
    uint64_t maxSpins;
    uint64_t scratchAddress; // 自旋期间周期性遍历以驱逐 flag 缓存行（可为 0）
    uint64_t scratchBytes;
};

