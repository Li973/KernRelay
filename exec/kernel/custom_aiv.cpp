// 纯 AIV 自定义算子与 AICore 侧自旋探针。
// 显式 KERNEL_TASK_TYPE_DEFAULT(AIV_ONLY) 由编译期元数据固定物理引擎，
// 不依赖编译器对指令通道的推断（见 registry/engine_map.json）。
#include "kernel_operator.h"

#include "custom_args.h"

using namespace AscendC;

namespace {
constexpr uint32_t VEC_TILE = 8192;      // half 元素/次，16KB
constexpr uint32_t FLAG_STRIDE = 16;
} // namespace

// out = (in * 1.0009765625) 重复 iters 次；纯向量（AIV）执行。
extern "C" __global__ __aicore__ void nhp_aiv_scale(GM_ADDR inGM, GM_ADDR outGM,
                                                    uint32_t elems, uint32_t iters)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t blockNum = GetBlockNum();
    const uint32_t blockIdx = GetBlockIdx();
    if (blockNum == 0 || blockIdx >= blockNum) return;
    const uint32_t per = (elems + blockNum - 1) / blockNum;
    const uint32_t begin = blockIdx * per;
    if (begin >= elems) return;
    const uint32_t count = (per < elems - begin) ? per : (elems - begin);

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    pipe.InitBuffer(inQue, 1, VEC_TILE * sizeof(half));
    pipe.InitBuffer(outQue, 1, VEC_TILE * sizeof(half));

    GlobalTensor<half> inGm;
    GlobalTensor<half> outGm;
    inGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(inGM) + begin, count);
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(outGM) + begin, count);

    for (uint32_t off = 0; off < count; off += VEC_TILE) {
        const uint32_t n = (VEC_TILE < count - off) ? VEC_TILE : (count - off);
        LocalTensor<half> in = inQue.AllocTensor<half>();
        DataCopy(in, inGm[off], n);
        inQue.EnQue(in);
        in = inQue.DeQue<half>();
        LocalTensor<half> out = outQue.AllocTensor<half>();
        const half k = (half)1.0009765625;
        for (uint32_t it = 0; it < iters; ++it) {
            Muls(out, in, k, n);
        }
        outQue.EnQue(out);
        out = outQue.DeQue<half>();
        DataCopy(outGm[off], out, n);
        outQue.FreeTensor(out);
        inQue.FreeTensor(in);
    }
}

// AIV 侧延迟发布（跨流通知探针的 producer）：忙等 delayIters 次后发布 flag。
// 用于与 AICPU/AIC 消费者配对——同单元自旋会阻塞发布 kernel 被派发。
extern "C" __global__ __aicore__ void nhp_aiv_delay_publish(GM_ADDR flagGM, uint32_t index,
                                                            uint32_t delayIters)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    volatile uint32_t sink = 0;
    for (uint32_t i = 0; i < delayIters; ++i) {
        sink += i;
    }
    GlobalTensor<int32_t> flags;
    flags.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flagGM),
                          static_cast<uint64_t>(index + 1U) * FLAG_STRIDE);
    auto flag = flags[index * FLAG_STRIDE];
    flag.SetValue(0, 1);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(flag);
    DataSyncBarrier<MemDsbT::DDR>();
    (void)sink;
}

// AICore(AIV) 侧跨流通知探针：自旋等待 producer 发布的 DDR flag。
// 关键：标量 GetValue 会命中 AIV 标量缓存，DCCI(CACHELINE_OUT) 实测无法让
// 跨单元写入可见（AIC/AICPU 发布后仍读 stale 0）。这里用 DMA（DataCopy）把
// flag 拉进 UB 再读，保证每次轮询都走 DDR。
// noWait=1 时只采样一次（基线：同样的启动/访存，不等待）。
extern "C" __global__ __aicore__ void nhp_aiv_spin(GM_ADDR flagGM, uint32_t index,
                                                   GM_ADDR statusGM, uint32_t noWait)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t SPIN_DEAD_DMA = 200000U;   // DMA 轮询 ~us/次
    GlobalTensor<int32_t> flags;
    flags.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flagGM),
                          static_cast<uint64_t>(index + 1U) * FLAG_STRIDE);
    GlobalTensor<int32_t> flag = flags[index * FLAG_STRIDE];
    GlobalTensor<uint32_t> status;
    status.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(statusGM),
                           static_cast<uint64_t>(index + 1U) * FLAG_STRIDE);

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, 32);                     // 8 x int32 = 32B（DMA 最小粒度）
    LocalTensor<int32_t> local = buf.Get<int32_t>();

    DataCopy(local, flag, 8);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    uint32_t value = static_cast<uint32_t>(local.GetValue(0));
    if (noWait == 0) {
        uint32_t spins = 0;
        while (value == 0 && spins < SPIN_DEAD_DMA) {
            ++spins;
            DataCopy(local, flag, 8);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            value = static_cast<uint32_t>(local.GetValue(0));
        }
    }
    status.SetValue(index * FLAG_STRIDE, value != 0 ? 1U : 2U);
    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(status);
    DataSyncBarrier<MemDsbT::DDR>();
}
