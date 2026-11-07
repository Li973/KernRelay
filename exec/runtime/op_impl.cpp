// NeutronHP 算子真实实现：aclnn level2 / HCCL C API 封装。
#include "runtime/op_impl.h"

#include <cstring>
#include <stdexcept>
#include <vector>

#include "aclnnop/level2/aclnn_add.h"
#include "aclnnop/level2/aclnn_cast.h"
#include "aclnnop/level2/aclnn_embedding.h"
#include "aclnnop/level2/aclnn_gelu.h"
#include "aclnnop/level2/aclnn_layer_norm.h"
#include "aclnnop/level2/aclnn_matmul.h"
#include "aclnnop/level2/aclnn_softmax.h"
#include "aclnnop/level2/aclnn_sort.h"
#include "aclnnop/level2/aclnn_topk.h"

#include "aclrtlaunch_nhp_aic_mac.h"
#include "aclrtlaunch_nhp_aiv_scale.h"
#include "aclrtlaunch_nhp_mix_scale.h"
#include "kernel/custom_args.h"

// AICPU 自定义内核 host stub（由 AICPU 编译器生成，签名与 tile_gate 一致）
extern "C" void nhp_aicpu_scale(uint32_t, void *, aclrtStream, void *, size_t);

namespace nhp {

namespace {
constexpr uint64_t WORKSPACE_CAP = 4ULL * 1024 * 1024 * 1024;

void CheckAcl(aclError ret, const char *what)
{
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed, acl=" + std::to_string(static_cast<int>(ret)));
    }
}

void CheckOp(aclnnStatus ret, const char *what)
{
    if (ret != 0) {
        throw std::runtime_error(std::string(what) + " failed, aclnn=" + std::to_string(static_cast<int>(ret)));
    }
}

void CheckHccl(HcclResult ret, const char *what)
{
    if (ret != HCCL_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed, hccl=" + std::to_string(static_cast<int>(ret)));
    }
}

void *MallocDevice(size_t bytes)
{
    void *ptr = nullptr;
    CheckAcl(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    return ptr;
}
} // namespace

OpKind ParseOpKind(const std::string &name)
{
    if (name == "matmul") return OpKind::MatMul;
    if (name == "add") return OpKind::Add;
    if (name == "gelu") return OpKind::Gelu;
    if (name == "softmax") return OpKind::Softmax;
    if (name == "layernorm") return OpKind::LayerNorm;
    if (name == "cast") return OpKind::Cast;
    if (name == "topk") return OpKind::TopK;
    if (name == "sort") return OpKind::Sort;
    if (name == "embedding") return OpKind::Embedding;
    if (name == "allreduce") return OpKind::AllReduce;
    if (name == "allgather") return OpKind::AllGather;
    if (name == "reducescatter") return OpKind::ReduceScatter;
    if (name == "custom_aiv_scale") return OpKind::CustomAivScale;
    if (name == "custom_aic_mac") return OpKind::CustomAicMac;
    if (name == "custom_mix_scale") return OpKind::CustomMixScale;
    if (name == "custom_aicpu_scale") return OpKind::CustomAicpuScale;
    throw std::runtime_error("unknown op: " + name);
}

const char *OpKindName(OpKind kind)
{
    switch (kind) {
        case OpKind::MatMul: return "matmul";
        case OpKind::Add: return "add";
        case OpKind::Gelu: return "gelu";
        case OpKind::Softmax: return "softmax";
        case OpKind::LayerNorm: return "layernorm";
        case OpKind::Cast: return "cast";
        case OpKind::TopK: return "topk";
        case OpKind::Sort: return "sort";
        case OpKind::Embedding: return "embedding";
        case OpKind::AllReduce: return "allreduce";
        case OpKind::AllGather: return "allgather";
        case OpKind::ReduceScatter: return "reducescatter";
        case OpKind::CustomAivScale: return "custom_aiv_scale";
        case OpKind::CustomAicMac: return "custom_aic_mac";
        case OpKind::CustomMixScale: return "custom_mix_scale";
        case OpKind::CustomAicpuScale: return "custom_aicpu_scale";
    }
    return "unknown";
}

bool IsCommKind(OpKind kind)
{
    return kind == OpKind::AllReduce || kind == OpKind::AllGather || kind == OpKind::ReduceScatter;
}


void OpTask::MakeTensor(void *data, aclDataType dtype,
                        const int64_t *dims, int64_t dimCount)
{
    std::vector<int64_t> strides(dimCount, 1);
    for (int64_t i = dimCount - 2; i >= 0; --i) strides[i] = strides[i + 1] * dims[i + 1];
    aclTensor *tensor = aclCreateTensor(dims, dimCount, dtype, strides.data(), 0, ACL_FORMAT_ND,
                                        dims, dimCount, data);
    if (tensor == nullptr) throw std::runtime_error("aclCreateTensor returned nullptr");
    if (tensorCount_ >= 8) throw std::runtime_error("too many tensors");
    tensors_[tensorCount_++] = tensor;
}

void OpTask::MakeTensorView(void *data, aclDataType dtype,
                            const int64_t *dims, int64_t dimCount,
                            const int64_t *strides, int64_t storageOffset,
                            const int64_t *storageDims)
{
    aclTensor *tensor = aclCreateTensor(dims, dimCount, dtype, strides, storageOffset,
                                        ACL_FORMAT_ND, storageDims, dimCount, data);
    if (tensor == nullptr) throw std::runtime_error("aclCreateTensor(view) returned nullptr");
    if (tensorCount_ >= 8) throw std::runtime_error("too many tensors");
    tensors_[tensorCount_++] = tensor;
}

OpTask::~OpTask()
{
    Destroy();
}

void OpTask::Destroy()
{
    // 实测 CANN 9.0.0（Ascend 910B3）资源回收边界（两类实测缺陷 -> 只置空泄漏）：
    //   1) aclDestroyAclOpExecutor 析构 executor 触发 glibc 堆损坏（同步后亦然）；
    //   2) aclDestroyTensor/IntArray/Scalar 在 executor 泄漏存活期间调用，退出
    //      析构期必现 glibc 堆损坏（corrupted double-linked list, SIGABRT）；
    //   均改为置空随进程泄漏。nhp_bench/nhp_exec 均为短进程，泄漏可接受。
    //   （去池化后运行期已稳定；此改动消除退出期 abort。）
    executor_ = nullptr;
    tensorCount_ = 0;
    for (int i = 0; i < 8; ++i) tensors_[i] = nullptr;
    intArray_ = nullptr;
    scalar_ = nullptr;
    isCustom_ = false;
    sendBuf_ = nullptr;
    recvBuf_ = nullptr;
    if (!externalWs_) {
        workspace_ = nullptr;
        workspaceCapacity_ = 0;
    }
    workspaceSize_ = 0;
}

void OpTask::PrepareCompute(OpKind kind, const OpShape &shape,
                            void *in, void *in2, void *out, void *out2,
                            void *aux1, void *aux2)
{
    Destroy();
    kind_ = kind;
    shape_ = shape;
    isComm_ = false;
    isCustom_ = false;
    const int64_t rows = shape.rows;
    const int64_t cols = shape.cols;

    switch (kind) {
        case OpKind::MatMul: {
            const int64_t aDims[2] = {rows, shape.k};
            const int64_t bDims[2] = {shape.k, shape.n};
            const int64_t cDims[2] = {rows, shape.n};
            MakeTensor(in, ACL_FLOAT16, aDims, 2);
            if (shape.ldb > 0) {
                const int64_t bStrides[2] = {shape.ldb, 1};
                const int64_t bStorage[2] = {shape.k, shape.ldb};
                MakeTensorView(in2, ACL_FLOAT16, bDims, 2, bStrides,
                               shape.b_col_off, bStorage);
            } else {
                MakeTensor(in2, ACL_FLOAT16, bDims, 2);
            }
            if (shape.ldc > 0) {
                const int64_t cStrides[2] = {shape.ldc, 1};
                const int64_t cStorage[2] = {rows, shape.ldc};
                MakeTensorView(out, ACL_FLOAT16, cDims, 2, cStrides,
                               shape.c_col_off, cStorage);
            } else {
                MakeTensor(out, ACL_FLOAT16, cDims, 2);
            }
            const int8_t cubeMathType = 0;
            CheckOp(aclnnMatmulGetWorkspaceSize(tensors_[0], tensors_[1], tensors_[2], cubeMathType,
                                                &workspaceSize_, &executor_),
                    "aclnnMatmulGetWorkspaceSize");
            break;
        }
        case OpKind::Add: {
            const int64_t dims[2] = {rows, cols};
            MakeTensor(in, ACL_FLOAT16, dims, 2);
            MakeTensor(in2, ACL_FLOAT16, dims, 2);
            MakeTensor(out, ACL_FLOAT16, dims, 2);
            static float alphaValue = 1.0f;
            scalar_ = aclCreateScalar(&alphaValue, ACL_FLOAT);
            CheckOp(aclnnAddGetWorkspaceSize(tensors_[0], tensors_[1], scalar_, tensors_[2],
                                             &workspaceSize_, &executor_),
                    "aclnnAddGetWorkspaceSize");
            break;
        }
        case OpKind::Gelu:
        case OpKind::Softmax:
        case OpKind::Cast: {
            const int64_t dims[2] = {rows, cols};
            MakeTensor(in, ACL_FLOAT16, dims, 2);
            if (kind == OpKind::Cast) {
                MakeTensor(out, ACL_FLOAT, dims, 2);
                CheckOp(aclnnCastGetWorkspaceSize(tensors_[0], ACL_FLOAT, tensors_[1],
                                                  &workspaceSize_, &executor_),
                        "aclnnCastGetWorkspaceSize");
            } else if (kind == OpKind::Gelu) {
                MakeTensor(out, ACL_FLOAT16, dims, 2);
                CheckOp(aclnnGeluGetWorkspaceSize(tensors_[0], tensors_[1],
                                                  &workspaceSize_, &executor_),
                        "aclnnGeluGetWorkspaceSize");
            } else {
                MakeTensor(out, ACL_FLOAT16, dims, 2);
                CheckOp(aclnnSoftmaxGetWorkspaceSize(tensors_[0], 1, tensors_[1],
                                                     &workspaceSize_, &executor_),
                        "aclnnSoftmaxGetWorkspaceSize");
            }
            break;
        }
        case OpKind::LayerNorm: {
            const int64_t dims[2] = {rows, cols};
            const int64_t meanDims[1] = {rows};
            MakeTensor(in, ACL_FLOAT16, dims, 2);       // 0 input
            MakeTensor(out, ACL_FLOAT16, dims, 2);      // 1 out
            MakeTensor(aux1, ACL_FLOAT16, dims + 1, 1); // 2 weight[cols]
            MakeTensor(aux2, ACL_FLOAT16, dims + 1, 1); // 3 bias[cols]
            MakeTensor(out2, ACL_FLOAT, meanDims, 1);   // 4 mean[rows]
            MakeTensor(in2, ACL_FLOAT, meanDims, 1);    // 5 rstd[rows]
            const int64_t normDims[1] = {cols};
            intArray_ = aclCreateIntArray(normDims, 1);
            // 实测 CANN 9.0.0：mean/rstd 可选输出传非空会触发 561103，按 nullptr 传入
            (void)out2;
            (void)in2;
            CheckOp(aclnnLayerNormGetWorkspaceSize(tensors_[0], intArray_, tensors_[2], tensors_[3],
                                                   1e-5, tensors_[1], nullptr, nullptr,
                                                   &workspaceSize_, &executor_),
                    "aclnnLayerNormGetWorkspaceSize");
            break;
        }
        case OpKind::TopK: {
            const int64_t dims[2] = {rows, cols};
            const int64_t outDims[2] = {rows, shape.k};
            MakeTensor(in, ACL_FLOAT16, dims, 2);
            MakeTensor(out, ACL_FLOAT16, outDims, 2);   // values
            MakeTensor(out2, ACL_INT64, outDims, 2);    // indices
            CheckOp(aclnnTopkGetWorkspaceSize(tensors_[0], shape.k, 1, true, true,
                                              tensors_[1], tensors_[2],
                                              &workspaceSize_, &executor_),
                    "aclnnTopkGetWorkspaceSize");
            break;
        }
        case OpKind::Sort: {
            const int64_t dims[2] = {rows, cols};
            MakeTensor(in, ACL_FLOAT16, dims, 2);
            MakeTensor(out, ACL_FLOAT16, dims, 2);      // values
            MakeTensor(out2, ACL_INT64, dims, 2);       // indices
            CheckOp(aclnnSortGetWorkspaceSize(tensors_[0], false, 1, true,
                                              tensors_[1], tensors_[2],
                                              &workspaceSize_, &executor_),
                    "aclnnSortGetWorkspaceSize");
            break;
        }
        case OpKind::Embedding: {
            const int64_t wDims[2] = {shape.dim0, cols};
            const int64_t iDims[1] = {rows};
            const int64_t oDims[2] = {rows, cols};
            MakeTensor(in, ACL_FLOAT16, wDims, 2);      // weight [V, D]
            MakeTensor(in2, ACL_INT64, iDims, 1);       // indices [rows]
            MakeTensor(out, ACL_FLOAT16, oDims, 2);     // out [rows, D]
            CheckOp(aclnnEmbeddingGetWorkspaceSize(tensors_[0], tensors_[1], tensors_[2],
                                                   &workspaceSize_, &executor_),
                    "aclnnEmbeddingGetWorkspaceSize");
            break;
        }
        default:
            throw std::runtime_error("PrepareCompute on comm op");
    }

    if (workspaceSize_ > WORKSPACE_CAP) {
        throw std::runtime_error("workspace too large: " + std::to_string(workspaceSize_));
    }
    if (workspaceSize_ != 0) {
        if (externalWs_) {
            if (workspace_ == nullptr || workspaceCapacity_ < workspaceSize_) {
                throw std::runtime_error("external workspace too small");
            }
        } else {
            workspace_ = MallocDevice(workspaceSize_);
        }
    }
}

void OpTask::BindWorkspace(void *buf, uint64_t capacity)
{
    workspace_ = buf;
    workspaceCapacity_ = capacity;
    externalWs_ = true;
}

std::pair<void *, uint64_t> OpTask::ShareWorkspace() const
{
    return std::make_pair(workspace_, workspaceSize_);
}

void OpTask::MarkWorkspaceExternal()
{
    externalWs_ = true;
}

void OpTask::Release()
{
    // 仅在同步完成后调用（exec_main 在每轮 RunOnce 尾部统一释放）。
    // 注：aclDestroyAclOpExecutor 即使同步后仍触发 CANN 9.0.0 堆损坏（实测
    // core dump），executor 只能进程内泄漏；此处仅回收自持 workspace。
    if (!externalWs_ && workspace_ != nullptr) {
        (void)aclrtFree(workspace_);
    }
    workspace_ = nullptr;
    workspaceSize_ = 0;
    workspaceCapacity_ = 0;
    externalWs_ = false;
}

void OpTask::PrepareCustom(OpKind kind, const OpShape &shape,
                           void *in, void *out, uint32_t iters, uint32_t blocks)
{
    Destroy();
    kind_ = kind;
    shape_ = shape;
    isComm_ = false;
    isCustom_ = true;
    sendBuf_ = in;
    recvBuf_ = out;
    customIters_ = iters == 0 ? 1 : iters;
    customBlocks_ = blocks == 0 ? 1 : blocks;
}

void OpTask::PrepareComm(OpKind kind, const OpShape &shape,
                         void *sendBuf, void *recvBuf, void *auxRecv,
                         HcclComm comm, uint32_t rank, uint32_t nranks)
{
    Destroy();
    kind_ = kind;
    shape_ = shape;
    isComm_ = true;
    isCustom_ = false;
    sendBuf_ = sendBuf;
    recvBuf_ = recvBuf;
    auxRecv_ = auxRecv;
    comm_ = comm;
    rank_ = rank;
    nranks_ = nranks;
    allGatherNorm_ = false;
}

void OpTask::SetAllGatherNorm(int64_t rowsTotal, int64_t rowsPerTile, int64_t outRowOff)
{
    allGatherNorm_ = true;
    rowsTotal_ = rowsTotal;
    rowsPerTile_ = rowsPerTile;
    outRowOff_ = outRowOff;
}

void OpTask::Enqueue(aclrtStream stream)
{
    if (isCustom_) {
        const uint64_t elems64 = static_cast<uint64_t>(shape_.rows) * static_cast<uint64_t>(shape_.cols);
        if (elems64 == 0 || elems64 > 0xFFFFFFFFULL) {
            throw std::runtime_error("custom op elems out of range");
        }
        const uint32_t elems = static_cast<uint32_t>(elems64);
        uint32_t rc = 0;
        switch (kind_) {
            case OpKind::CustomAivScale:
                rc = ACLRT_LAUNCH_KERNEL(nhp_aiv_scale)(customBlocks_, stream, sendBuf_, recvBuf_,
                                                        elems, customIters_);
                break;
            case OpKind::CustomAicMac:
                rc = ACLRT_LAUNCH_KERNEL(nhp_aic_mac)(customBlocks_, stream, sendBuf_, recvBuf_,
                                                      elems, customIters_);
                break;
            case OpKind::CustomMixScale:
                rc = ACLRT_LAUNCH_KERNEL(nhp_mix_scale)(customBlocks_, stream, sendBuf_, recvBuf_,
                                                        elems, customIters_);
                break;
            case OpKind::CustomAicpuScale: {
                CustomScaleArgs args;
                args.in = reinterpret_cast<uint64_t>(sendBuf_);
                args.out = reinterpret_cast<uint64_t>(recvBuf_);
                args.elems = elems;
                args.iters = customIters_;
                nhp_aicpu_scale(1, nullptr, stream, &args, sizeof(args));
                return;
            }
            default:
                throw std::runtime_error("Enqueue on unknown custom kind");
        }
        if (rc != 0) {
            throw std::runtime_error("custom kernel launch failed, rc=" + std::to_string(rc));
        }
        return;
    }
    if (!isComm_) {
        switch (kind_) {
            case OpKind::MatMul:
                CheckOp(aclnnMatmul(workspace_, workspaceSize_, executor_, stream), "aclnnMatmul");
                break;
            case OpKind::Add:
                CheckOp(aclnnAdd(workspace_, workspaceSize_, executor_, stream), "aclnnAdd");
                break;
            case OpKind::Gelu:
                CheckOp(aclnnGelu(workspace_, workspaceSize_, executor_, stream), "aclnnGelu");
                break;
            case OpKind::Softmax:
                CheckOp(aclnnSoftmax(workspace_, workspaceSize_, executor_, stream), "aclnnSoftmax");
                break;
            case OpKind::LayerNorm:
                CheckOp(aclnnLayerNorm(workspace_, workspaceSize_, executor_, stream), "aclnnLayerNorm");
                break;
            case OpKind::Cast:
                CheckOp(aclnnCast(workspace_, workspaceSize_, executor_, stream), "aclnnCast");
                break;
            case OpKind::TopK:
                CheckOp(aclnnTopk(workspace_, workspaceSize_, executor_, stream), "aclnnTopk");
                break;
            case OpKind::Sort:
                CheckOp(aclnnSort(workspace_, workspaceSize_, executor_, stream), "aclnnSort");
                break;
            case OpKind::Embedding:
                CheckOp(aclnnEmbedding(workspace_, workspaceSize_, executor_, stream), "aclnnEmbedding");
                break;
            default:
                throw std::runtime_error("Enqueue on unprepared comm op");
        }
        return;
    }

    const uint64_t elems = static_cast<uint64_t>(shape_.rows) * static_cast<uint64_t>(shape_.cols);
    switch (kind_) {
        case OpKind::AllReduce:
            CheckHccl(HcclAllReduce(sendBuf_, recvBuf_, elems, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM,
                                    comm_, stream),
                      "HcclAllReduce");
            break;
        case OpKind::AllGather: {
            void *rawRecv = allGatherNorm_ ? auxRecv_ : recvBuf_;
            CheckHccl(HcclAllGather(sendBuf_, rawRecv, elems, HCCL_DATA_TYPE_FP16, comm_, stream),
                      "HcclAllGather");
            if (allGatherNorm_) {
                // 布局归一：临时 recv 为 [r tile]，拷入最终缓冲 [r all] 内 tile 行块。
                const size_t tileBytes =
                    static_cast<size_t>(rowsPerTile_) * static_cast<size_t>(shape_.cols) * sizeof(uint16_t);
                auto *src = static_cast<uint8_t *>(auxRecv_);
                auto *dst = static_cast<uint8_t *>(recvBuf_);
                for (uint32_t r = 0; r < nranks_; ++r) {
                    const size_t dstOffset =
                        (static_cast<size_t>(outRowOff_) +
                         static_cast<size_t>(r) * static_cast<size_t>(rowsPerTile_)) *
                        static_cast<size_t>(shape_.cols) * sizeof(uint16_t);
                    CheckAcl(aclrtMemcpyAsync(dst + dstOffset, tileBytes, src + r * tileBytes,
                                              tileBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                             "allgather norm copy");
                }
            }
            break;
        }
        case OpKind::ReduceScatter:
            CheckHccl(HcclReduceScatter(sendBuf_, recvBuf_, elems / nranks_, HCCL_DATA_TYPE_FP16,
                                        HCCL_REDUCE_SUM, comm_, stream),
                      "HcclReduceScatter");
            break;
        default:
            throw std::runtime_error("PrepareComm on compute op");
    }
}

} // namespace nhp

