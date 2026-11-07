#pragma once
#include <stdint.h>

// 自定义纯算子/自旋探针的公共参数。
// Ascend C 内核直接按值接收；AICPU 内核接收指向该结构体的设备指针。
struct CustomScaleArgs {
    uint64_t in;        // 输入 GM 地址（fp16）
    uint64_t out;       // 输出 GM 地址（fp16）
    uint32_t elems;     // 元素数
    uint32_t iters;     // 重复计算次数（控制时长）
};

// 跨流通知探针：等待 flag[idx*FLAG_STRIDE] 非 0
struct SpinArgs {
    uint64_t flagAddress;
    uint64_t statusAddress;
    uint32_t index;
    uint32_t maxSpins;
    uint32_t noWait;    // 1 = 只读一次不进自旋（基线）
    uint32_t pad;
};
