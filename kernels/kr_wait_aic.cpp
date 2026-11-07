#include "kr_sync_common.h"

using namespace AscendC;

// consumer 侧 AIC 空 kernel（单 kernel 独占文件）。volatile 读 + DCCI ALL
// 规避 AIC 标量缓存读到 stale published（父项目 probe_aic_spin 同款处理）。
extern "C" __global__ __aicore__ void kr_wait_aic(GM_ADDR ctrGM, uint32_t index,
                                                  uint32_t maxSpins)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GlobalTensor<int32_t> ctr;
    ctr.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ctrGM) + index * kr::kWords,
                        kr::kWords);
    auto pub = ctr[kr::kOffPublished];
    auto con = ctr[kr::kOffConsumed];
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_ALL>(pub);
    volatile __gm__ int32_t *slot =
        reinterpret_cast<volatile __gm__ int32_t *>(ctrGM) + index * kr::kWords;
    const int32_t target = slot[kr::kOffConsumed] + 1;
    int32_t value = slot[kr::kOffPublished];
    uint32_t spins = 0;
    while (value < target && spins < maxSpins) {
        ++spins;
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                 DcciDst::CACHELINE_ALL>(pub);
        value = slot[kr::kOffPublished];
    }
    con.SetValue(0, target);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(con);
    auto st = ctr[kr::kOffStatus];
    st.SetValue(0, value >= target ? 1 : 2);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(st);
    DataSyncBarrier<MemDsbT::DDR>();
}
