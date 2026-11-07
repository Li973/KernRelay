#include "kr_sync_common.h"

using namespace AscendC;

// producer 侧 AIV 空 kernel：published += 1（DMA 读回 + 标量写 + DCCI）。
extern "C" __global__ __aicore__ void kr_publish_aiv(GM_ADDR ctrGM, uint32_t index)
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
    const int32_t next = local.GetValue(kr::kOffPublished) + 1;
    auto pub = ctr[kr::kOffPublished];
    pub.SetValue(0, next);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(pub);
    DataSyncBarrier<MemDsbT::DDR>();
}
