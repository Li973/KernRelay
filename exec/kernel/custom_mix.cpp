// MIX 自定义算子：显式 MIX_AIC_1_2（1 个 AIC 配 2 个 AIV）。
// AIC 分支做标量累加，AIV 分支做向量缩放；两侧互不同步、无数据交换，
// 用于验证「MIX 任务的物理占位」以及 MIX 与纯 AIC/AIV 的互斥执行。
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t VEC_TILE = 8192;

__aicore__ inline void MixVectorBody(GM_ADDR inGM, GM_ADDR outGM, uint32_t elems, uint32_t iters)
{
    const uint32_t blockNum = GetBlockNum();   // AIV 侧为 2 × AIC 块数
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
        const half k = (half)1.001953125;
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

__aicore__ inline float MixScalarBody(GM_ADDR inGM, uint32_t elems, uint32_t iters)
{
    const uint32_t blockNum = GetBlockNum();   // AIC 块数
    const uint32_t blockIdx = GetBlockIdx();
    if (blockNum == 0 || blockIdx >= blockNum) return 0.0f;
    const uint32_t per = (elems + blockNum - 1) / blockNum;
    const uint32_t begin = blockIdx * per;
    if (begin >= elems) return 0.0f;
    const uint32_t count = (per < elems - begin) ? per : (elems - begin);

    GlobalTensor<half> inGm;
    inGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(inGM) + begin, count);
    float acc = 0.0625f;
    // 乘子取 <1 保证累加器收敛；>1 会指数发散，fp16 输出溢出为 inf（checksum 写出非法 JSON）。
    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t i = 0; i < count; ++i) {
            acc = acc * 0.9990234375f + (float)inGm.GetValue(i);
        }
    }
    return acc;
}
} // namespace

extern "C" __global__ __aicore__ void nhp_mix_scale(GM_ADDR inGM, GM_ADDR outGM,
                                                    uint32_t elems, uint32_t iters)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC {
        const float acc = MixScalarBody(inGM, elems, iters);
        if (acc == 1234567.0f) {   // 不可达：保持标量循环不被消除
            GlobalTensor<half> outGm;
            outGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(outGM), 1);
            outGm.SetValue(0, (half)acc);
        }
    }
    if ASCEND_IS_AIV {
        MixVectorBody(inGM, outGM, elems, iters);
    }
}
