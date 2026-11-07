#include "kr_sync_common.h"

using namespace AscendC;

// producer 侧 AIC 空 kernel（单 kernel 独占文件，见父项目 AIC meta 约束）。
extern "C" __global__ __aicore__ void kr_publish_aic(GM_ADDR ctrGM, uint32_t index)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GlobalTensor<int32_t> ctr;
    ctr.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ctrGM) + index * kr::kWords,
                        kr::kWords);
    auto pub = ctr[kr::kOffPublished];
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_ALL>(pub);
    volatile __gm__ int32_t *slot =
        reinterpret_cast<volatile __gm__ int32_t *>(ctrGM) + index * kr::kWords;
    const int32_t next = *slot + 1;
    pub.SetValue(0, next);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(pub);
    DataSyncBarrier<MemDsbT::DDR>();
}
