// 跨 stream 空 kernel 同步协议（KernRelay）。
// 每个依赖边一个 32B 控制块：[published, consumed, status, pad...] int32。
// producer 在产生数据的 kernel 之后启动 publish：对 published 原子 +1；
// consumer 在消费数据的 kernel 之前启动 wait：读 consumed 得到 target=consumed+1，
// DMA 轮询 published 直到 >= target，再把 consumed 写回 target。
// 计数器随 replay 单调递增，无需 host 每次清零；所有轮询走 DMA，规避标量缓存。
#pragma once
#include "kernel_operator.h"

namespace kr {
constexpr uint32_t kWords = 8;       // 32B
constexpr uint32_t kOffPublished = 0;
constexpr uint32_t kOffConsumed = 1;
constexpr uint32_t kOffStatus = 2;

template <typename T>
__aicore__ inline void DmaRead(AscendC::LocalTensor<T> &local,
                               AscendC::GlobalTensor<T> &src, uint32_t words)
{
    AscendC::DataCopy(local, src, words);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
}
} // namespace kr
