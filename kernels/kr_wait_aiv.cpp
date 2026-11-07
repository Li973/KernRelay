#include "kr_sync_common.h"

using namespace AscendC;

// consumer 侧 AIV 空 kernel：等待 published >= consumed+1，然后推进 consumed。
// 每个边只有一个 consumer，consumed 无写竞争；published 只由 producer 递增。
extern "C" __global__ __aicore__ void kr_wait_aiv(GM_ADDR ctrGM, uint32_t index,
                                                  uint32_t maxSpins)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<int32_t> ctr;
    ctr.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ctrGM) + index * kr::kWords,
                        kr::kWords);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, 32);
    LocalTensor<int32_t> local = buf.Get<int32_t>();
    kr::DmaRead(local, ctr, kr::kWords);
    const int32_t target = local.GetValue(kr::kOffConsumed) + 1;
    int32_t value = local.GetValue(kr::kOffPublished);
    uint32_t spins = 0;
    while (value < target && spins < maxSpins) {
        ++spins;
        kr::DmaRead(local, ctr, kr::kWords);
        value = local.GetValue(kr::kOffPublished);
    }
    auto con = ctr[kr::kOffConsumed];
    con.SetValue(0, target);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(con);
    auto st = ctr[kr::kOffStatus];
    st.SetValue(0, value >= target ? 1 : 2);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(st);
    DataSyncBarrier<MemDsbT::DDR>();
}
