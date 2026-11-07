// 纯 AIC 自定义算子：显式 AIC_ONLY。
// 计算走标量循环（AIC 标量单元），时长由 elems*iters 控制；用于验证
// 「纯 AIC 任务」在多流并发下的物理占位与稳定性。
//
// 注意：本文件只放一个 AIC kernel。CANN ascendc 生成器给同文件所有 kernel
// 的 .ascend.meta 用同一个 base_key 后缀，第二个 kernel 起 meta 名与符号名
// 不匹配，runtime 注册 AIC 二进制会以 107000 失败（见 probe_aic_delay.cpp）。
#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void nhp_aic_mac(GM_ADDR inGM, GM_ADDR outGM,
                                                  uint32_t elems, uint32_t iters)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const uint32_t blockNum = GetBlockNum();
    const uint32_t blockIdx = GetBlockIdx();
    if (blockNum == 0 || blockIdx >= blockNum) return;
    const uint32_t per = (elems + blockNum - 1) / blockNum;
    const uint32_t begin = blockIdx * per;
    if (begin >= elems) return;
    const uint32_t count = (per < elems - begin) ? per : (elems - begin);

    GlobalTensor<half> inGm;
    GlobalTensor<half> outGm;
    inGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(inGM) + begin, count);
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(outGM) + begin, count);

    float acc = 0.03125f;
    // 乘子取 <1 保证累加器收敛；>1 会指数发散，fp16 输出溢出为 inf（checksum 写出非法 JSON）。
    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t i = 0; i < count; ++i) {
            acc = acc * 0.9990234375f + (float)inGm.GetValue(i);
        }
    }
    // 单点写回：保持计算不被优化，又不与消费者读取的主体数据竞争。
    outGm.SetValue(0, (half)acc);
    DataCacheCleanAndInvalid<half, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(outGm);
    DataSyncBarrier<MemDsbT::DDR>();
}
