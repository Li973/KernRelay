// publish_tile_flag：producer 侧逐 tile 就绪标记发布（Ascend C，AI Core）。
// 与 aicpu_operator_other4_demo/producer.cpp 中的 DCCI 协议一致：
// 写 flag -> DataCacheCleanAndInvalid(DCCI) -> DataSyncBarrier<DDR>，
// 消费侧 AICPU tile_gate 自旋读取该 flag（DDR 一致性保证可见）。
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t FLAG_STRIDE = 16;
}

extern "C" __global__ __aicore__ void publish_tile_flag(GM_ADDR flagsGM, uint32_t tile)
{
    GlobalTensor<int32_t> flags;
    flags.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flagsGM), (tile + 1U) * FLAG_STRIDE);
    auto flag = flags[tile * FLAG_STRIDE];
    flag.SetValue(0, 1);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(flag);
    DataSyncBarrier<MemDsbT::DDR>();
}
