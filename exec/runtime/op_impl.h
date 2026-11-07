#pragma once
// NeutronHP 算子真实实现层：调用高级算子（aclnn level2 / HCCL C API）。
// 每个 OpTask = 一次真实调用（全量或单 tile），prepare 阶段创建 aclTensor
// 与 workspace 并绑定设备指针，Enqueue 阶段仅向指定 stream 入队。
#include <cstdint>
#include <string>
#include <utility>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "hccl/hccl.h"
#include "hccl/hccl_comm.h"

namespace nhp {

enum class OpKind {
    MatMul, Add, Gelu, Softmax, LayerNorm, Cast, TopK, Sort, Embedding,
    AllReduce, AllGather, ReduceScatter,
    // 自定义算子（kernel/ 下，编译期固定物理引擎）
    CustomAivScale, CustomAicMac, CustomMixScale, CustomAicpuScale
};

OpKind ParseOpKind(const std::string &name);
const char *OpKindName(OpKind kind);
bool IsCommKind(OpKind kind);

// 一次调用的逻辑形状（已按 tile 切好）：
//   matmul      : rows=M, k=K, n=N；in=A, in2=B, out=C
//   add         : rows, cols；in/in2=输入, out
//   gelu/softmax/cast : rows, cols；in, out   (cast: in fp16 -> out fp32)
//   layernorm   : rows, cols；in, out；weight/bias[cols], mean/rstd[rows]
//   topk        : rows, cols, k=K；in, out=values, out2=indices(int64)
//   sort        : rows, cols；in, out=values, out2=indices(int64)
//   embedding   : rows(=indices 数), cols(=dim)；in=weight[V,D], in2=indices(int64), out
//   allreduce   : rows, cols；in=send, out=recv（元素数 rows*cols，fp16）
//   allgather   : rows, cols；in=send, out=recv 归一后最终缓冲, aux=临时 recv
//   reducescatter: rows, cols；in=send, out=recv 最终缓冲
struct OpShape {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t k = 0;   // matmul K / topk K
    int64_t n = 0;   // matmul N (本 tile 的列数)
    int64_t dim0 = 0; // embedding 词表大小 V
    // matmul 列切（N 切）时的存储步长与列偏移；0 表示连续整幅。
    int64_t ldb = 0;
    int64_t ldc = 0;
    int64_t b_col_off = 0;
    int64_t c_col_off = 0;
};

class OpTask {
public:
    OpTask() = default;
    ~OpTask();
    OpTask(const OpTask &) = delete;
    OpTask &operator=(const OpTask &) = delete;

    // compute 类：绑定输入/输出指针并准备执行器。aux1/aux2 视算子而定（见上）。
    void PrepareCompute(OpKind kind, const OpShape &shape,
                        void *in, void *in2, void *out, void *out2,
                        void *aux1, void *aux2);
    // comm 类：绑定 send/recv 与通信域。
    void PrepareComm(OpKind kind, const OpShape &shape,
                     void *sendBuf, void *recvBuf, void *auxRecv,
                     HcclComm comm, uint32_t rank, uint32_t nranks);
    // 自定义类：in/out 为 fp16 [rows, cols]，iters 控制重复计算时长，
    // blocks 为 AIC/AIV 启动块数（AICPU 固定单块）。
    void PrepareCustom(OpKind kind, const OpShape &shape,
                       void *in, void *out, uint32_t iters, uint32_t blocks);
    // allgather 逐 tile 归一：把临时 recv（[r tile] 布局）按 rank 块拷入最终缓冲。
    // allgather 布局归一：把 [nranks][rowsPerChunk] 的临时结果按 rank 块拷进最终缓冲。
    // outRowOff = 本块在最终输出行空间中的起点（块连续布局：
    // [outRowOff + r*rowsPerChunk, +rowsPerChunk)）。
    void SetAllGatherNorm(int64_t rowsTotal, int64_t rowsPerChunk, int64_t outRowOff);

    void Enqueue(aclrtStream stream);
    uint64_t WorkspaceBytes() const { return workspaceSize_; }

    // workspace 处理（进程内泄漏规避）：CANN 9.0.0 下 executor/句柄/workspace
    // 销毁均实测触发堆损坏或设备侧异常，plan 重建时 workspace 自持全新分配、
    // 随析构只置空（短进程泄漏，进程退出回收）。曾尝试 wsPool 池化跨轮复用
    // 同一 buffer——实测为 507015 主凶，已移除（见 doc/design.md 第 6 节）。
    void BindWorkspace(void *buf, uint64_t capacity);
    std::pair<void *, uint64_t> ShareWorkspace() const;
    void MarkWorkspaceExternal();
    // 显式释放（仅允许在全部入队任务同步完成后调用）：回收自持 workspace。
    // 注：exec_main/bench_main 当前均不调用（plan 随轮次重建，短进程泄漏）。
    void Release();

private:
    void Destroy();
    void MakeTensor(void *data, aclDataType dtype,
                    const int64_t *dims, int64_t dimCount);
    void MakeTensorView(void *data, aclDataType dtype,
                        const int64_t *dims, int64_t dimCount,
                        const int64_t *strides, int64_t storageOffset,
                        const int64_t *storageDims);

    OpKind kind_ = OpKind::MatMul;
    OpShape shape_{};
    bool isComm_ = false;
    bool isCustom_ = false;
    uint32_t customIters_ = 1;
    uint32_t customBlocks_ = 1;
    bool allGatherNorm_ = false;
    int64_t rowsTotal_ = 0, rowsPerTile_ = 0, outRowOff_ = 0;
    HcclComm comm_ = nullptr;
    uint32_t rank_ = 0, nranks_ = 1;
    void *sendBuf_ = nullptr;
    void *recvBuf_ = nullptr;
    void *auxRecv_ = nullptr;

    aclTensor *tensors_[8] = {nullptr};
    int tensorCount_ = 0;
    aclIntArray *intArray_ = nullptr;
    aclScalar *scalar_ = nullptr;
    aclOpExecutor *executor_ = nullptr;
    void *workspace_ = nullptr;
    uint64_t workspaceSize_ = 0;
    uint64_t workspaceCapacity_ = 0;
    bool externalWs_ = false;
};

} // namespace nhp
