// runtime/hccl_session.hpp —— HCCL 通信域生命周期契约（RAII）
//
// 契约（CANN 正常用法，对照稳定参考实现 main.cpp:133-135/549、
// aclnn_matmul_overlap.cpp:184/519）：
//   1. 通信域在设备上下文就绪后创建（HcclCommInitAll，每 rank 一个）；
//   2. 销毁前先 aclrtSetDevice + aclrtSynchronizeDevice，排干在飞工作；
//   3. HcclCommDestroy 在【运行期】、所有 rank 对称执行，且先于
//      aclrtDestroyDevice / aclFinalize / 任何静态析构；
//   4. 成功与失败路径一视同仁（析构函数兜底异常展开）。
//
// 为什么以契约为前提而不是"泄漏到进程退出"：泄漏模式把 ~HcclCommunicator
// 推进 __cxa_finalize 期，依赖 libhcomm/libprofimpl 之间【未定义】的静态析构
// 顺序，实测触发跨库 UAF（hrtSetDevice → ProfAclMgr 已析构 map 插入，
// MALLOC_PERTURB_ 毒化下 100% 必现），见 doc/design.md §7.1。
#pragma once

#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_comm.h"

class HcclSession {
public:
    HcclSession() = default;
    ~HcclSession() { Destroy(); }  // 异常展开/提前返回也走契约
    HcclSession(const HcclSession &) = delete;
    HcclSession &operator=(const HcclSession &) = delete;

    // 设备上下文就绪后调用；随后把 Comms() 交给 HcclCommInitAll 写入。
    void Init(const std::vector<int32_t> &devices)
    {
        devices_ = devices;
        comms_.assign(devices.size(), nullptr);
    }

    HcclComm *Comms() { return comms_.data(); }

    // 运行期对称销毁（幂等）：结果落盘后、return 前调用。
    void Destroy()
    {
        for (size_t r = 0; r < comms_.size(); ++r) {
            if (comms_[r] == nullptr || r >= devices_.size()) continue;
            (void)aclrtSetDevice(devices_[r]);
            (void)aclrtSynchronizeDevice();
            (void)HcclCommDestroy(comms_[r]);
            comms_[r] = nullptr;
        }
    }

private:
    std::vector<int32_t> devices_;
    std::vector<HcclComm> comms_;
};
