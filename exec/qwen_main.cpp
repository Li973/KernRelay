// KernRelay C++ Qwen2.5 prefill 运行时。
//
// 同一份 ACLNN 算子序列支持：
//   serial / serial_fused   : 单流逐算子 C++ 派发（不融合 / add+rmsnorm 融合）
//   serial_graph            : 单流未融合整流 ACL Graph
//   graph_whole             : 单流融合流 ACL Graph（NHP_GRAPH 单流策略）
//   relay_event             : 双流 + ACL event + C++ 派发
//   relay_flag              : 双流 + GM flag 空 kernel + C++ 派发
//   relay_flag_graph        : 双流 + GM flag 空 kernel + 逐 stream ACL Graph
//
// 权重由 scripts/export_qwen.py 导出（matmul 权重已转置 [K,N]）。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <condition_variable>
#include <mutex>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_rms_norm.h"
#include "aclnnop/aclnn_apply_rotary_pos_emb.h"
#include "aclnnop/aclnn_embedding.h"
#include "aclnnop/aclnn_flash_attention_score.h"
#include "aclnnop/aclnn_matmul.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_rms_norm.h"
#include "aclnnop/aclnn_silu.h"
#include "aclrtlaunch_kr_publish_aic.h"
#include "aclrtlaunch_kr_publish_aiv.h"
#include "aclrtlaunch_kr_wait_aic.h"
#include "aclrtlaunch_kr_wait_aiv.h"

#include "hccl/hccl.h"
#include "hccl/hccl_comm.h"
#include "runtime/json_min.hpp"

using Clock = std::chrono::steady_clock;
using nhp::json::Value;

namespace {

void Check(aclError r, const char *w)
{
    if (r != ACL_SUCCESS) {
        throw std::runtime_error(std::string(w) + " acl=" + std::to_string(int(r)));
    }
}

void CheckOp(aclnnStatus r, const char *w)
{
    if (r != 0) {
        throw std::runtime_error(std::string(w) + " aclnn=" + std::to_string(int(r)));
    }
}

void CheckHccl(HcclResult r, const char *w)
{
    if (r != HCCL_SUCCESS) {
        throw std::runtime_error(std::string(w) + " hccl=" + std::to_string(int(r)));
    }
}

double Us(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::micro>(b - a).count();
}


struct Call {
    // ACLNN 的 executor 在 CANN 9.0.0 下不能长期持有：后续创建其他算子会
    // 覆盖 op-exec cache 的共享 buffer，已在 rope/embed 组合上复现
    // “check graph != nullptr” 与段错误。因此每次执行时重建 executor，
    // workspace 则按算子实例预分配并复用。
    uint64_t ws_cap = 0;
    void *ws_dev = nullptr;
    std::function<void(uint64_t &, aclOpExecutor **)> make;
    std::function<void(void *, uint64_t, aclOpExecutor *, aclrtStream)> exec;
    std::function<void(aclrtStream)> run;
};

struct DevTensor {
    void *ptr = nullptr;
    size_t bytes = 0;
    aclTensor *t = nullptr;
};

void FinalizeCall(Call *c)
{
    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    c->make(ws, &ex);
    c->ws_cap = ws;
    if (ws) Check(aclrtMalloc(&c->ws_dev, ws, ACL_MEM_MALLOC_HUGE_FIRST), "op ws");
    c->run = [c](aclrtStream s) {
        uint64_t w = 0;
        aclOpExecutor *e = nullptr;
        c->make(w, &e);
        if (w > c->ws_cap) throw std::runtime_error("op workspace grew");
        c->exec(w ? c->ws_dev : nullptr, w, e, s);
    };
}

aclTensor *MakeAclTensor(void *ptr, const std::vector<int64_t> &dims, aclDataType dt)
{
    return aclCreateTensor(dims.data(), int64_t(dims.size()), dt, nullptr, 0,
                           ACL_FORMAT_ND, dims.data(), int64_t(dims.size()), ptr);
}

aclTensor *MakeAclTensorStrided(void *ptr, const std::vector<int64_t> &dims,
                                const std::vector<int64_t> &strides,
                                int64_t offset,
                                const std::vector<int64_t> &storage, aclDataType dt)
{
    return aclCreateTensor(dims.data(), int64_t(dims.size()), dt, strides.data(),
                           offset, ACL_FORMAT_ND, storage.data(),
                           int64_t(storage.size()), ptr);
}

DevTensor AllocTensor(const std::vector<int64_t> &dims, aclDataType dt, size_t bytes)
{
    DevTensor d;
    d.bytes = bytes;
    Check(aclrtMalloc(&d.ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc tensor");
    d.t = MakeAclTensor(d.ptr, dims, dt);
    return d;
}

// 连续行切片（2D），用于 matmul/add/rmsnorm/silu/mul/embedding 的行切。
DevTensor SliceRowsEx(const DevTensor &src, int64_t row_off, int64_t rows,
                      int64_t cols, aclDataType dt, size_t elem_bytes)
{
    DevTensor d;
    d.ptr = static_cast<char *>(src.ptr) + row_off * cols * elem_bytes;
    d.bytes = size_t(rows) * size_t(cols) * elem_bytes;
    d.t = MakeAclTensor(d.ptr, {rows, cols}, dt);
    return d;
}

DevTensor SliceRows(const DevTensor &src, int64_t row_off, int64_t rows,
                    int64_t cols, aclDataType dt)
{
    return SliceRowsEx(src, row_off, rows, cols, dt, 2);
}

// 4D [1,rows,heads,dim] 行切片；存储布局为 [1,S,heads,dim]。
DevTensor SliceRows4D(const DevTensor &src, int64_t row_off, int64_t rows,
                      int64_t heads, int64_t dim, aclDataType dt)
{
    DevTensor d;
    d.ptr = static_cast<char *>(src.ptr) + row_off * heads * dim * 2;
    d.bytes = size_t(rows) * size_t(heads) * size_t(dim) * 2;
    d.t = MakeAclTensor(d.ptr, {1, rows, heads, dim}, dt);
    return d;
}

// 4D 头切片：dims [1,S,h,D]，strides 指向原 [1,S,H,D] 存储。
DevTensor SliceHeads4D(const DevTensor &src, int64_t head_off, int64_t heads,
                       int64_t total_heads, int64_t seq, int64_t dim,
                       aclDataType dt)
{
    DevTensor d;
    d.ptr = src.ptr; // offset 由 aclTensor storageOffset 表达
    d.bytes = size_t(seq) * size_t(heads) * size_t(dim) * 2;
    const std::vector<int64_t> dims = {1, seq, heads, dim};
    const std::vector<int64_t> strides = {seq * total_heads * dim,
                                          total_heads * dim, dim, 1};
    const std::vector<int64_t> storage = {1, seq, total_heads, dim};
    d.t = MakeAclTensorStrided(d.ptr, dims, strides, head_off * dim, storage, dt);
    return d;
}

DevTensor SliceHeadsSmax(const DevTensor &src, int64_t head_off, int64_t heads,
                         int64_t total_heads, int64_t seq)
{
    DevTensor d;
    d.ptr = src.ptr;
    d.bytes = size_t(heads) * size_t(seq) * 8 * 4;
    const std::vector<int64_t> dims = {1, heads, seq, 8};
    const std::vector<int64_t> strides = {total_heads * seq * 8, seq * 8, 8, 1};
    const std::vector<int64_t> storage = {1, total_heads, seq, 8};
    d.t = MakeAclTensorStrided(d.ptr, dims, strides, head_off * seq * 8,
                               storage, ACL_FLOAT);
    return d;
}

struct Item {
    int stream; // 0 = aic, 1 = aiv
    std::function<void(aclrtStream)> fn;
};

class Qwen {
public:
    Qwen(std::string dir, int seq, int device, std::string mode_name)
        : dir_(std::move(dir)), seq_(seq), device_(device), mode_name_(std::move(mode_name)) {}

    void Build();
    void RunOnce();
    void CaptureAll();
    void Bench(int warmup, int iters);
    void Verify();
    void ProfileMany(const std::string &spec_json, std::string &result);
    void ConfigurePP(int stage, int stages, int requests, HcclComm comm,
                     const std::string &mode = "pp");
    void ConfigureTP(int rank, int size, int split, HcclComm comm,
                     const std::string &mode);
    void EnqueueTP();
    void EnqueuePP(int request);
    void VerifyPP();
    void CommitTiming(const std::vector<double> &samples);
    int device() const { return device_; }
    std::string ResultJson(const std::string &mode, int warmup, int iters) const;

private:
    struct Mode {
        bool serial = true;
        bool graph = false;
        bool sync = false;
        bool flags = false;
        bool fused = false;
    };

    Call *Matmul(const DevTensor &a, const DevTensor &b, const DevTensor &c,
                 const char *tag);
    Call *Add(const DevTensor &a, const DevTensor &b, const DevTensor &c,
              const char *tag);
    Call *RmsNorm(const DevTensor &x, const DevTensor &g, const DevTensor &y,
                  const DevTensor &rstd, const char *tag);
    Call *AddRmsNorm(const DevTensor &a, const DevTensor &b, const DevTensor &g,
                     const DevTensor &y, const DevTensor &rstd,
                     const DevTensor &sum, const char *tag);
    Call *Rope(const DevTensor &q, const DevTensor &k, const DevTensor &cos,
               const DevTensor &sin, const char *tag);
    Call *Flash(const DevTensor &q, const DevTensor &k, const DevTensor &v,
                const DevTensor &mask, const DevTensor &out,
                const DevTensor &smax, const DevTensor &ssum, const char *tag);
    Call *FlashN(const DevTensor &q, const DevTensor &k, const DevTensor &v,
                 const DevTensor &mask, const DevTensor &out,
                 const DevTensor &smax, const DevTensor &ssum, int heads,
                 const char *tag);
    Call *Silu(const DevTensor &x, const DevTensor &y, const char *tag);
    Call *Mul(const DevTensor &a, const DevTensor &b, const DevTensor &c,
              const char *tag);
    Call *Embed(const DevTensor &w, const DevTensor &ids, const DevTensor &out,
                const char *tag);

    void Emit(int stream, Call *c) { prog_.push_back({stream, [c](aclrtStream s) { c->run(s); }}); }
    void Sync(int stream, std::function<void(aclrtStream)> fn)
    {
        prog_.push_back({stream, std::move(fn)});
    }
    void Wait(int edge) { WaitOn(edge, true); }
    void WaitOn(int edge, bool producer_aiv);
    void Publish(int edge, bool producer_aiv);
    void PPSync(std::vector<Item> &prog, bool producer_aiv);
    int NewEdge();

    void BuildDual();
    void BuildSerial();
    void BuildPP();
    void BuildTP();
    void LoadWeightsTP();
    using EmitFn = std::function<void(int, Call *)>;
    void EmitQkv(const EmitFn &emit, int stream, const DevTensor &x,
                 const std::string &w, const std::string &b,
                 const DevTensor &raw, const DevTensor &out, const char *tag);
    void EmitQkNorm(const EmitFn &emit, int stream, const std::string &p,
                    const DevTensor &q3t, const DevTensor &k3t,
                    const DevTensor &q3n, const DevTensor &k3n);
    void EmitTo(std::vector<Item> &prog, int stream, Call *c)
    {
        if (pp_one_stream_) stream = 0;
        prog.push_back({stream, [c](aclrtStream s) { c->run(s); }});
    }
    void EmitFnTo(std::vector<Item> &prog, int stream, std::function<void(aclrtStream)> fn)
    {
        if (pp_one_stream_) stream = 0;
        prog.push_back({stream, std::move(fn)});
    }
    bool KeepWeight(const std::string &name) const;
    void SplitProgram();
    void MakeMask();
    void LoadManifest();
    void LoadWeights();
    void UploadIds();
    DevTensor &W(const std::string &name) { return weights_.at(name); }
    std::string LPrefix(int l) const
    {
        return "L" + std::string(l < 10 ? "0" : "") + std::to_string(l) + ".";
    }
    void RunProgramDirect();
    void Replay();

    std::string dir_;
    int seq_ = 2048;
    int device_ = 0;
    std::string mode_name_;
    Mode mode_;
    int H_ = 0, I_ = 0, L_ = 0, HQ_ = 0, HKV_ = 0, D_ = 0, V_ = 0;
    double eps_ = 1e-6;
    int max_pos_ = 4096;
    bool qk_norm_ = false;
    bool has_qkv_bias_ = true;

    aclrtStream aic_ = nullptr;
    aclrtStream aiv_ = nullptr;
    aclrtStream serial_ = nullptr;
    aclrtStream hccl_ = nullptr;
    bool pp_ = false;
    bool tp_ = false;
    int tp_rank_ = 0, tp_size_ = 2, tp_split_ = 1;
    bool pp_one_stream_ = false;
    int pp_stage_ = 0, pp_stages_ = 1, pp_requests_ = 1;
    int l0_ = 0, l1_ = 0;
    HcclComm comm_ = nullptr;
    std::vector<std::vector<Item>> pp_prog_;
    std::vector<aclrtEvent> pp_events_;
    DevTensor hreq_[8][2];
    DevTensor pp_out_[8];
    DevTensor down_red_;
    std::vector<Item> tp_prog_;
    aclmdlRI gAic_ = nullptr;
    aclmdlRI gAiv_ = nullptr;
    aclmdlRI gSerial_ = nullptr;

    std::vector<std::unique_ptr<Call>> calls_;
    std::vector<Item> prog_;
    std::vector<Item> aicActs_, aivActs_;
    std::vector<aclrtEvent> events_;
    void *ctr_ = nullptr;
    size_t ctr_bytes_ = 0;
    int edge_count_ = 0;

    std::map<std::string, DevTensor> weights_;
    std::map<std::string, Value> manifest_;
    void *weight_dev_ = nullptr;
    size_t weight_bytes_ = 0;
    void *weight_map_ = nullptr;
    int weight_fd_ = -1;

    DevTensor h_[2], x_, q_, k_, v_, qraw_, kraw_, vraw_, attn_, o_;
    DevTensor qn_, kn_, rstdq_, rstdk_;
    DevTensor gate_, up_, act_, down_, logits_, hf_, xf_;
    DevTensor rstd_, rstd2_, rstd3_, smax_, ssum_, mask_, ids_;

    int verify_argmax_ = -1;
    int ref_argmax_ = -1;
    double verify_maxdiff_ = 0.0;
    double verify_absmean_ = 0.0;
    int flag_timeouts_ = 0;
    int flag_edges_ = 0;
    double best_us_ = 0.0, typical_us_ = 0.0;
    double host_best_us_ = 0.0, host_typical_us_ = 0.0;
    std::vector<double> samples_;
    std::vector<double> host_samples_;
};

void Qwen::LoadManifest()
{
    Value doc = nhp::json::ParseFile(dir_ + "/manifest.json");
    for (const auto &kv : doc.At("tensors").object) manifest_[kv.first] = kv.second;
    const Value &cfg = doc.At("config");
    H_ = int(cfg.At("hidden_size").number);
    I_ = int(cfg.At("intermediate_size").number);
    L_ = int(cfg.At("num_hidden_layers").number);
    HQ_ = int(cfg.At("num_attention_heads").number);
    HKV_ = int(cfg.At("num_key_value_heads").number);
    V_ = int(cfg.At("vocab_size").number);
    D_ = cfg.Has("head_dim") ? int(cfg.At("head_dim").number) : H_ / HQ_;
    qk_norm_ = cfg.Has("qk_norm") ? cfg.At("qk_norm").boolean : false;
    bool attn_bias = cfg.Has("attention_bias") ? cfg.At("attention_bias").boolean : true;
    has_qkv_bias_ = attn_bias && manifest_.count("L00.q_proj.bias") != 0;
    eps_ = cfg.At("rms_norm_eps").number;
    max_pos_ = int(doc.At("max_pos").number);
    if (int(doc.At("seq").number) != seq_) {
        throw std::runtime_error("manifest seq mismatch");
    }
}

void Qwen::LoadWeights()
{
    int fd = ::open((dir_ + "/weights.bin").c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("open weights.bin");
    struct stat st {};
    ::fstat(fd, &st);
    weight_fd_ = fd;
    weight_bytes_ = size_t(st.st_size);
    weight_map_ = ::mmap(nullptr, weight_bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
    if (weight_map_ == MAP_FAILED) throw std::runtime_error("mmap weights");
    if (!pp_) {
        // 单卡路径保持整块布局：图重放对权重地址/页对齐敏感，逐 tensor
        // 分配会改变 L2 命中（实测 graph_whole 慢约 10%）。
        Check(aclrtMalloc(&weight_dev_, weight_bytes_, ACL_MEM_MALLOC_HUGE_FIRST),
              "malloc weights");
        Check(aclrtMemcpy(weight_dev_, weight_bytes_, weight_map_, weight_bytes_,
                          ACL_MEMCPY_HOST_TO_DEVICE),
              "h2d weights");
        for (const auto &kv : manifest_) {
            const Value &v = kv.second;
            std::vector<int64_t> dims;
            for (const auto &d : v.At("shape").array) dims.push_back(int64_t(d.number));
            DevTensor t;
            t.ptr = static_cast<char *>(weight_dev_) +
                    size_t(v.At("offset").number);
            t.bytes = size_t(v.At("bytes").number);
            t.t = MakeAclTensor(t.ptr, dims, ACL_BF16);
            weights_[kv.first] = t;
        }
        return;
    }
    size_t used = 0;
    for (const auto &kv : manifest_) {
        if (!KeepWeight(kv.first)) continue;
        const Value &v = kv.second;
        std::vector<int64_t> dims;
        for (const auto &d : v.At("shape").array) dims.push_back(int64_t(d.number));
        const size_t bytes = size_t(v.At("bytes").number);
        DevTensor t;
        Check(aclrtMalloc(&t.ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc weight");
        Check(aclrtMemcpy(t.ptr, bytes,
                          static_cast<char *>(weight_map_) +
                              size_t(v.At("offset").number),
                          bytes, ACL_MEMCPY_HOST_TO_DEVICE),
              "h2d weight");
        t.bytes = bytes;
        t.t = MakeAclTensor(t.ptr, dims, ACL_BF16);
        weights_[kv.first] = t;
        used += bytes;
    }
    weight_bytes_ = used;
}

bool Qwen::KeepWeight(const std::string &name) const
{
    if (!pp_) return true;
    if (name == "embed.weight") return pp_stage_ == 0;
    if (name == "lm_head.weight_t" || name == "final_norm.weight") {
        return pp_stage_ == pp_stages_ - 1;
    }
    if (name == "rope.cos" || name == "rope.sin") return true;
    if (name.size() > 2 && name[0] == 'L' && std::isdigit(uint8_t(name[1]))) {
        int layer = std::stoi(name.substr(1, 2));
        if (layer >= l0_ && layer < l1_) return true;
        // stage 0 的最后一层 addnorm 需要下一层（stage 1 首层）的 ln1 权重。
        if (pp_stage_ == 0 && layer == l1_ &&
            name.find("input_layernorm.weight") != std::string::npos) {
            return true;
        }
        return false;
    }
    return true;
}

void Qwen::UploadIds()
{
    std::ifstream f(dir_ + "/ids.bin", std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("ids.bin");
    size_t bytes = size_t(f.tellg());
    f.seekg(0);
    std::vector<int64_t> ids(bytes / 8);
    f.read(reinterpret_cast<char *>(ids.data()), std::streamsize(bytes));
    if (int(ids.size()) != seq_) throw std::runtime_error("ids size != seq");
    ids_ = AllocTensor({1, seq_}, ACL_INT64, bytes);
    Check(aclrtMemcpy(ids_.ptr, bytes, ids.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
          "ids h2d");
}

void Qwen::MakeMask()
{
    int m = std::max(2048, seq_);
    mask_ = AllocTensor({m, m}, ACL_BOOL, size_t(m) * size_t(m));
    std::vector<uint8_t> host(size_t(m) * size_t(m), 0);
    for (int i = 0; i < m; ++i)
        for (int j = i + 1; j < m; ++j) host[size_t(i) * size_t(m) + size_t(j)] = 1;
    Check(aclrtMemcpy(mask_.ptr, host.size(), host.data(), host.size(),
                      ACL_MEMCPY_HOST_TO_DEVICE),
          "mask h2d");
}

Call *Qwen::Matmul(const DevTensor &a, const DevTensor &b, const DevTensor &c, const char *tag)
{
    auto *call = new Call();
    aclTensor *at = a.t, *bt = b.t, *ct = c.t;
    call->make = [at, bt, ct, tag](uint64_t &ws, aclOpExecutor **ex) {
        CheckOp(aclnnMatmulGetWorkspaceSize(at, bt, ct, 0, &ws, ex), tag);
    };
    call->exec = [tag](void *w, uint64_t ws, aclOpExecutor *ex, aclrtStream s) {
        CheckOp(aclnnMatmul(w, ws, ex, s), tag);
    };
    FinalizeCall(call);
    calls_.emplace_back(call);
    return call;
}

Call *Qwen::Add(const DevTensor &a, const DevTensor &b, const DevTensor &c, const char *tag)
{
    static float one = 1.0f;
    static aclScalar *alpha = aclCreateScalar(&one, ACL_FLOAT);
    auto *call = new Call();
    aclTensor *at = a.t, *bt = b.t, *ct = c.t;
    call->make = [at, bt, ct, tag](uint64_t &ws, aclOpExecutor **ex) {
        CheckOp(aclnnAddGetWorkspaceSize(at, bt, alpha, ct, &ws, ex), tag);
    };
    call->exec = [tag](void *w, uint64_t ws, aclOpExecutor *ex, aclrtStream s) {
        CheckOp(aclnnAdd(w, ws, ex, s), tag);
    };
    FinalizeCall(call);
    calls_.emplace_back(call);
    return call;
}

Call *Qwen::RmsNorm(const DevTensor &x, const DevTensor &g, const DevTensor &y,
                    const DevTensor &rstd, const char *tag)
{
    auto *call = new Call();
    aclTensor *xt = x.t, *gt = g.t, *yt = y.t, *rt = rstd.t;
    const double eps = eps_;
    call->make = [xt, gt, yt, rt, eps, tag](uint64_t &ws, aclOpExecutor **ex) {
        CheckOp(aclnnRmsNormGetWorkspaceSize(xt, gt, eps, yt, rt, &ws, ex), tag);
    };
    call->exec = [tag](void *w, uint64_t ws, aclOpExecutor *ex, aclrtStream s) {
        CheckOp(aclnnRmsNorm(w, ws, ex, s), tag);
    };
    FinalizeCall(call);
    calls_.emplace_back(call);
    return call;
}

Call *Qwen::AddRmsNorm(const DevTensor &a, const DevTensor &b, const DevTensor &g,
                       const DevTensor &y, const DevTensor &rstd,
                       const DevTensor &sum, const char *tag)
{
    auto *call = new Call();
    aclTensor *at = a.t, *bt = b.t, *gt = g.t, *yt = y.t, *rt = rstd.t, *st = sum.t;
    const double eps = eps_;
    call->make = [at, bt, gt, yt, rt, st, eps, tag](uint64_t &ws, aclOpExecutor **ex) {
        CheckOp(aclnnAddRmsNormGetWorkspaceSize(at, bt, gt, eps, yt, rt, st, &ws, ex),
                tag);
    };
    call->exec = [tag](void *w, uint64_t ws, aclOpExecutor *ex, aclrtStream s) {
        CheckOp(aclnnAddRmsNorm(w, ws, ex, s), tag);
    };
    FinalizeCall(call);
    calls_.emplace_back(call);
    return call;
}

Call *Qwen::Rope(const DevTensor &q, const DevTensor &k, const DevTensor &cos,
                 const DevTensor &sin, const char *tag)
{
    auto *call = new Call();
    aclTensor *qt = q.t, *kt = k.t, *ct = cos.t, *st = sin.t;
    call->make = [qt, kt, ct, st, tag](uint64_t &ws, aclOpExecutor **ex) {
        CheckOp(aclnnApplyRotaryPosEmbGetWorkspaceSize(qt, kt, ct, st, 1, &ws, ex),
                tag);
    };
    call->exec = [tag](void *w, uint64_t ws, aclOpExecutor *ex, aclrtStream s) {
        CheckOp(aclnnApplyRotaryPosEmb(w, ws, ex, s), tag);
    };
    FinalizeCall(call);
    calls_.emplace_back(call);
    return call;
}

Call *Qwen::Flash(const DevTensor &q, const DevTensor &k, const DevTensor &v,
                  const DevTensor &mask, const DevTensor &out,
                  const DevTensor &smax, const DevTensor &ssum, const char *tag)
{
    return FlashN(q, k, v, mask, out, smax, ssum, HQ_, tag);
}

Call *Qwen::FlashN(const DevTensor &q, const DevTensor &k, const DevTensor &v,
                   const DevTensor &mask, const DevTensor &out,
                   const DevTensor &smax, const DevTensor &ssum, int heads,
                   const char *tag)
{
    auto *call = new Call();
    aclTensor *qt = q.t, *kt = k.t, *vt = v.t, *mt = mask.t, *ot = out.t;
    aclTensor *smt = smax.t, *sst = ssum.t;
    const double scale = 1.0 / std::sqrt(double(D_));
    call->make = [qt, kt, vt, mt, ot, smt, sst, scale, heads, tag](
                     uint64_t &ws, aclOpExecutor **ex) {
        CheckOp(aclnnFlashAttentionScoreGetWorkspaceSize(
                    qt, kt, vt, nullptr, nullptr, nullptr, mt, nullptr, scale, 1.0,
                    2147483647, 2147483647, heads, const_cast<char *>("BSND"), 0, 3,
                    smt, sst, nullptr, ot, &ws, ex),
                tag);
    };
    call->exec = [tag](void *w, uint64_t ws, aclOpExecutor *ex, aclrtStream s) {
        CheckOp(aclnnFlashAttentionScore(w, ws, ex, s), tag);
    };
    FinalizeCall(call);
    calls_.emplace_back(call);
    return call;
}

Call *Qwen::Silu(const DevTensor &x, const DevTensor &y, const char *tag)
{
    auto *call = new Call();
    aclTensor *xt = x.t, *yt = y.t;
    call->make = [xt, yt, tag](uint64_t &ws, aclOpExecutor **ex) {
        CheckOp(aclnnSiluGetWorkspaceSize(xt, yt, &ws, ex), tag);
    };
    call->exec = [tag](void *w, uint64_t ws, aclOpExecutor *ex, aclrtStream s) {
        CheckOp(aclnnSilu(w, ws, ex, s), tag);
    };
    FinalizeCall(call);
    calls_.emplace_back(call);
    return call;
}

Call *Qwen::Mul(const DevTensor &a, const DevTensor &b, const DevTensor &c, const char *tag)
{
    auto *call = new Call();
    aclTensor *at = a.t, *bt = b.t, *ct = c.t;
    call->make = [at, bt, ct, tag](uint64_t &ws, aclOpExecutor **ex) {
        CheckOp(aclnnMulGetWorkspaceSize(at, bt, ct, &ws, ex), tag);
    };
    call->exec = [tag](void *w, uint64_t ws, aclOpExecutor *ex, aclrtStream s) {
        CheckOp(aclnnMul(w, ws, ex, s), tag);
    };
    FinalizeCall(call);
    calls_.emplace_back(call);
    return call;
}

Call *Qwen::Embed(const DevTensor &w, const DevTensor &ids, const DevTensor &out, const char *tag)
{
    auto *call = new Call();
    aclTensor *wt = w.t, *it = ids.t, *ot = out.t;
    call->make = [wt, it, ot, tag](uint64_t &ws, aclOpExecutor **ex) {
        CheckOp(aclnnEmbeddingGetWorkspaceSize(wt, it, ot, &ws, ex), tag);
    };
    call->exec = [tag](void *p, uint64_t ws, aclOpExecutor *ex, aclrtStream s) {
        CheckOp(aclnnEmbedding(p, ws, ex, s), tag);
    };
    FinalizeCall(call);
    calls_.emplace_back(call);
    return call;
}

int Qwen::NewEdge()
{
    int e = edge_count_++;
    if (mode_.sync && !mode_.flags) {
        aclrtEvent ev = nullptr;
        Check(aclrtCreateEvent(&ev), "create event");
        events_.push_back(ev);
    }
    return e;
}

void Qwen::WaitOn(int edge, bool producer_aiv)
{
    if (!mode_.sync) return;
    int consumer = producer_aiv ? 0 : 1; // AIV 生产 -> AIC 消费
    if (mode_.flags) {
        Sync(consumer, [this, edge, producer_aiv](aclrtStream s) {
            uint32_t rc = producer_aiv
                              ? aclrtlaunch_kr_wait_aic(1, s, ctr_, uint32_t(edge), 20000000U)
                              : aclrtlaunch_kr_wait_aiv(1, s, ctr_, uint32_t(edge), 20000000U);
            if (rc != 0) throw std::runtime_error("launch wait failed");
        });
    } else {
        Sync(consumer, [this, edge](aclrtStream s) {
            Check(aclrtStreamWaitEvent(s, events_[edge]), "wait event");
        });
    }
}

void Qwen::Publish(int edge, bool producer_aiv)
{
    if (!mode_.sync) return;
    int producer = producer_aiv ? 1 : 0;
    if (mode_.flags) {
        Sync(producer, [this, edge, producer_aiv](aclrtStream s) {
            uint32_t rc = producer_aiv
                              ? aclrtlaunch_kr_publish_aiv(1, s, ctr_, uint32_t(edge))
                              : aclrtlaunch_kr_publish_aic(1, s, ctr_, uint32_t(edge));
            if (rc != 0) throw std::runtime_error("launch publish failed");
        });
    } else {
        Sync(producer, [this, edge](aclrtStream s) {
            Check(aclrtRecordEvent(events_[edge], s), "record event");
        });
    }
}

// PP 直接派发不建图，跨流依赖用 AIC/AIV 空 kernel 的 GM flag 对：
// producer 侧 publish 后，consumer 侧 wait 自旋；每条边只消费一次发布，
// 因此同一 prog 反复入队不会累积计数。
void Qwen::PPSync(std::vector<Item> &prog, bool producer_aiv)
{
    if (pp_one_stream_) return;
    const int edge = edge_count_++;
    const int producer = producer_aiv ? 1 : 0;
    const int consumer = producer_aiv ? 0 : 1;
    prog.push_back({producer, [this, edge, producer_aiv](aclrtStream s) {
        uint32_t rc = producer_aiv
                          ? aclrtlaunch_kr_publish_aiv(1, s, ctr_, uint32_t(edge))
                          : aclrtlaunch_kr_publish_aic(1, s, ctr_, uint32_t(edge));
        if (rc != 0) throw std::runtime_error("pp publish failed");
    }});
    prog.push_back({consumer, [this, edge, producer_aiv](aclrtStream s) {
        uint32_t rc = producer_aiv
                          ? aclrtlaunch_kr_wait_aic(1, s, ctr_, uint32_t(edge), 20000000U)
                          : aclrtlaunch_kr_wait_aiv(1, s, ctr_, uint32_t(edge), 20000000U);
        if (rc != 0) throw std::runtime_error("pp wait failed");
    }});
}

void Qwen::Build()
{
    mode_.serial = mode_name_ == "serial" || mode_name_ == "serial_fused" ||
                   mode_name_ == "serial_graph" || mode_name_ == "graph_whole";
    mode_.graph = mode_name_ == "serial_graph" || mode_name_ == "graph_whole" ||
                  mode_name_ == "relay_flag_graph";
    mode_.flags = mode_name_ == "relay_flag" || mode_name_ == "relay_flag_graph";
    mode_.sync = mode_name_ == "relay_event" || mode_.flags;
    pp_ = mode_name_.rfind("pp", 0) == 0;
    tp_ = mode_name_.rfind("tp", 0) == 0;
    pp_one_stream_ = mode_name_ == "pp1" || mode_name_ == "pp1_serial";
    mode_.fused = mode_name_ == "serial_fused" || mode_name_ == "graph_whole" ||
                  mode_name_.rfind("relay", 0) == 0 || pp_ || tp_;

    LoadManifest();
    l0_ = pp_stage_ * L_ / std::max(1, pp_stages_);
    l1_ = (pp_stage_ + 1) * L_ / std::max(1, pp_stages_);
    if (tp_) {
        LoadWeightsTP();
    } else {
        LoadWeights();
    }
    UploadIds();
    MakeMask();

    const int S = seq_;
    h_[0] = AllocTensor({S, H_}, ACL_BF16, size_t(S) * H_ * 2);
    h_[1] = AllocTensor({S, H_}, ACL_BF16, size_t(S) * H_ * 2);
    x_ = AllocTensor({S, H_}, ACL_BF16, size_t(S) * H_ * 2);
    qraw_ = AllocTensor({S, HQ_ * D_}, ACL_BF16, size_t(S) * HQ_ * D_ * 2);
    kraw_ = AllocTensor({S, HKV_ * D_}, ACL_BF16, size_t(S) * HKV_ * D_ * 2);
    vraw_ = AllocTensor({S, HKV_ * D_}, ACL_BF16, size_t(S) * HKV_ * D_ * 2);
    q_ = AllocTensor({S, HQ_ * D_}, ACL_BF16, size_t(S) * HQ_ * D_ * 2);
    k_ = AllocTensor({S, HKV_ * D_}, ACL_BF16, size_t(S) * HKV_ * D_ * 2);
    v_ = AllocTensor({S, HKV_ * D_}, ACL_BF16, size_t(S) * HKV_ * D_ * 2);
    attn_ = AllocTensor({S, HQ_ * D_}, ACL_BF16, size_t(S) * HQ_ * D_ * 2);
    o_ = AllocTensor({S, H_}, ACL_BF16, size_t(S) * H_ * 2);
    if (qk_norm_) {
        qn_ = AllocTensor({S, HQ_ * D_}, ACL_BF16, size_t(S) * HQ_ * D_ * 2);
        kn_ = AllocTensor({S, HKV_ * D_}, ACL_BF16, size_t(S) * HKV_ * D_ * 2);
        rstdq_ = AllocTensor({S * HQ_, 1}, ACL_FLOAT, size_t(S) * HQ_ * 4);
        rstdk_ = AllocTensor({S * HKV_, 1}, ACL_FLOAT, size_t(S) * HKV_ * 4);
    }
    const int Iloc = tp_ ? (I_ / tp_size_) : I_;
    gate_ = AllocTensor({S, Iloc}, ACL_BF16, size_t(S) * Iloc * 2);
    up_ = AllocTensor({S, Iloc}, ACL_BF16, size_t(S) * Iloc * 2);
    act_ = AllocTensor({S, Iloc}, ACL_BF16, size_t(S) * Iloc * 2);
    down_ = AllocTensor({S, H_}, ACL_BF16, size_t(S) * H_ * 2);
    if (tp_) down_red_ = AllocTensor({S, H_}, ACL_BF16, size_t(S) * H_ * 2);
    hf_ = AllocTensor({S, H_}, ACL_BF16, size_t(S) * H_ * 2);
    xf_ = AllocTensor({S, H_}, ACL_BF16, size_t(S) * H_ * 2);
    logits_ = AllocTensor({S, V_}, ACL_BF16, size_t(S) * V_ * 2);
    rstd_ = AllocTensor({S, 1}, ACL_FLOAT, size_t(S) * 4);
    rstd2_ = AllocTensor({S, 1}, ACL_FLOAT, size_t(S) * 4);
    rstd3_ = AllocTensor({S, 1}, ACL_FLOAT, size_t(S) * 4);
    smax_ = AllocTensor({1, HQ_, S, 8}, ACL_FLOAT, size_t(HQ_) * S * 8 * 4);
    ssum_ = AllocTensor({1, HQ_, S, 8}, ACL_FLOAT, size_t(HQ_) * S * 8 * 4);
    ctr_bytes_ = 4096 * 32;
    Check(aclrtMalloc(&ctr_, ctr_bytes_, ACL_MEM_MALLOC_HUGE_FIRST), "malloc ctr");
    Check(aclrtMemset(ctr_, ctr_bytes_, 0, ctr_bytes_), "zero ctr");

    if (tp_) {
        aclrtCreateStream(&serial_);
        aclrtCreateStream(&hccl_);
        BuildTP();
    } else if (pp_) {
        aclrtCreateStream(&serial_);
        aclrtCreateStream(&hccl_);
        BuildPP();
    } else if (mode_.serial) {
        aclrtCreateStream(&serial_);
        BuildSerial();
    } else {
        aclrtCreateStream(&aic_);
        aclrtCreateStream(&aiv_);
        BuildDual();
        SplitProgram();
    }
}

void Qwen::BuildDual()
{
    const int S = seq_;
    const int Dh = D_;
    const int H = H_;
    const int I = I_;
    DevTensor *hcur = &h_[0];
    DevTensor *hnxt = &h_[1];
    auto q3 = [&](DevTensor &d) { return MakeAclTensor(d.ptr, {1, S, HQ_, Dh}, ACL_BF16); };
    auto kv3 = [&](DevTensor &d) { return MakeAclTensor(d.ptr, {1, S, HKV_, Dh}, ACL_BF16); };
    auto x3 = [&](DevTensor &d) { return MakeAclTensor(d.ptr, {1, S, H}, ACL_BF16); };
    DevTensor q3t{q_.ptr, 0, q3(q_)};
    DevTensor k3t{k_.ptr, 0, kv3(k_)};
    DevTensor v3t{v_.ptr, 0, kv3(v_)};
    DevTensor a3t{attn_.ptr, 0, q3(attn_)};
    DevTensor h03d{h_[0].ptr, 0, x3(h_[0])};
    DevTensor cosv{W("rope.cos").ptr, 0,
                   MakeAclTensor(W("rope.cos").ptr, {1, S, 1, Dh}, ACL_BF16)};
    DevTensor sinv{W("rope.sin").ptr, 0,
                   MakeAclTensor(W("rope.sin").ptr, {1, S, 1, Dh}, ACL_BF16)};
    DevTensor q3n{qn_.ptr, 0, qk_norm_ ? MakeAclTensor(qn_.ptr, {1, S, HQ_, Dh}, ACL_BF16)
                                       : q3t.t};
    DevTensor k3n{kn_.ptr, 0, qk_norm_ ? MakeAclTensor(kn_.ptr, {1, S, HKV_, Dh}, ACL_BF16)
                                       : k3t.t};
    auto emit = [&](int st, Call *c) { Emit(st, c); };

    int next_entry = -1;
    for (int l = 0; l < L_; ++l) {
        const std::string p = LPrefix(l);
        DevTensor &g1 = W(p + "input_layernorm.weight");
        DevTensor &g2 = W(p + "post_attention_layernorm.weight");
        int entry = next_entry;
        if (l == 0) {
            entry = NewEdge();
            Emit(1, Embed(W("embed.weight"), ids_, h03d, "embed"));
            Emit(1, RmsNorm(*hcur, g1, x_, rstd_, "ln1_0"));
            Publish(entry, true);
        }
        WaitOn(entry, true); // AIC 等待本层入口（上一层末尾已发布或 embed+ln1）

        EmitQkv(emit, 0, x_, p + "q_proj.weight_t", p + "q_proj.bias", qraw_, q_, "q_proj");
        EmitQkv(emit, 0, x_, p + "k_proj.weight_t", p + "k_proj.bias", kraw_, k_, "k_proj");
        EmitQkv(emit, 0, x_, p + "v_proj.weight_t", p + "v_proj.bias", vraw_, v_, "v_proj");
        int e_qkv = NewEdge();
        Publish(e_qkv, false);
        WaitOn(e_qkv, false);

        EmitQkNorm(emit, 1, p, q3t, k3t, q3n, k3n);
        DevTensor q3r = qk_norm_ ? q3n : q3t;
        DevTensor k3r = qk_norm_ ? k3n : k3t;
        emit(1, Rope(q3r, k3r, cosv, sinv, "rope"));
        emit(1, Flash(q3r, k3r, v3t, mask_, a3t, smax_, ssum_, "flash"));
        int e_attn = NewEdge();
        Publish(e_attn, true);
        WaitOn(e_attn, true);

        Emit(0, Matmul(attn_, W(p + "o_proj.weight_t"), o_, "o_proj"));
        int e_o = NewEdge();
        Publish(e_o, false);
        WaitOn(e_o, false);

        if (mode_.fused) {
            Emit(1, AddRmsNorm(*hcur, o_, g2, x_, rstd2_, *hnxt, "addnorm_ln2"));
        } else {
            Emit(1, Add(*hcur, o_, *hnxt, "add1"));
            Emit(1, RmsNorm(*hnxt, g2, x_, rstd2_, "ln2"));
        }
        int e_ln2 = NewEdge();
        Publish(e_ln2, true);
        WaitOn(e_ln2, true);

        Emit(0, Matmul(x_, W(p + "gate_proj.weight_t"), gate_, "gate"));
        Emit(0, Matmul(x_, W(p + "up_proj.weight_t"), up_, "up"));
        int e_gu = NewEdge();
        Publish(e_gu, false);
        WaitOn(e_gu, false);

        Emit(1, Silu(gate_, act_, "silu"));
        Emit(1, Mul(act_, up_, act_, "mul"));
        int e_act = NewEdge();
        Publish(e_act, true);
        WaitOn(e_act, true);

        Emit(0, Matmul(act_, W(p + "down_proj.weight_t"), down_, "down"));
        int e_down = NewEdge();
        Publish(e_down, false);
        WaitOn(e_down, false);

        if (l + 1 < L_) {
            DevTensor &gn = W(LPrefix(l + 1) + "input_layernorm.weight");
            if (mode_.fused) {
                Emit(1, AddRmsNorm(*hnxt, down_, gn, x_, rstd3_, *hcur, "addnorm_ln1"));
            } else {
                Emit(1, Add(*hnxt, down_, *hcur, "add2"));
                Emit(1, RmsNorm(*hcur, gn, x_, rstd3_, "ln1_next"));
            }
            next_entry = NewEdge();
            Publish(next_entry, true);
        } else {
            DevTensor &gf = W("final_norm.weight");
            if (mode_.fused) {
                Emit(1, AddRmsNorm(*hnxt, down_, gf, xf_, rstd3_, hf_, "addnorm_final"));
            } else {
                Emit(1, Add(*hnxt, down_, hf_, "add_final"));
                Emit(1, RmsNorm(hf_, gf, xf_, rstd3_, "ln_final"));
            }
            int e_final = NewEdge();
            Publish(e_final, true);
            WaitOn(e_final, true);
            Emit(0, Matmul(xf_, W("lm_head.weight_t"), logits_, "lm_head"));
        }
    }
}

void Qwen::BuildSerial()
{
    const int S = seq_;
    const int Dh = D_;
    const int H = H_;
    const int I = I_;
    DevTensor *hcur = &h_[0];
    DevTensor *hnxt = &h_[1];
    DevTensor q3t{q_.ptr, 0, MakeAclTensor(q_.ptr, {1, S, HQ_, Dh}, ACL_BF16)};
    DevTensor k3t{k_.ptr, 0, MakeAclTensor(k_.ptr, {1, S, HKV_, Dh}, ACL_BF16)};
    DevTensor v3t{v_.ptr, 0, MakeAclTensor(v_.ptr, {1, S, HKV_, Dh}, ACL_BF16)};
    DevTensor a3t{attn_.ptr, 0, MakeAclTensor(attn_.ptr, {1, S, HQ_, Dh}, ACL_BF16)};
    DevTensor h03d{h_[0].ptr, 0, MakeAclTensor(h_[0].ptr, {1, S, H}, ACL_BF16)};
    DevTensor cosv{W("rope.cos").ptr, 0,
                   MakeAclTensor(W("rope.cos").ptr, {1, S, 1, Dh}, ACL_BF16)};
    DevTensor sinv{W("rope.sin").ptr, 0,
                   MakeAclTensor(W("rope.sin").ptr, {1, S, 1, Dh}, ACL_BF16)};
    DevTensor q3n{qn_.ptr, 0, qk_norm_ ? MakeAclTensor(qn_.ptr, {1, S, HQ_, Dh}, ACL_BF16)
                                       : q3t.t};
    DevTensor k3n{kn_.ptr, 0, qk_norm_ ? MakeAclTensor(kn_.ptr, {1, S, HKV_, Dh}, ACL_BF16)
                                       : k3t.t};
    auto emit = [&](int st, Call *c) { Emit(st, c); };
    Emit(0, Embed(W("embed.weight"), ids_, h03d, "embed"));
    for (int l = 0; l < L_; ++l) {
        const std::string p = LPrefix(l);
        DevTensor &g1 = W(p + "input_layernorm.weight");
        DevTensor &g2 = W(p + "post_attention_layernorm.weight");
        if (l == 0) Emit(0, RmsNorm(*hcur, g1, x_, rstd_, "ln1_0"));
        EmitQkv(emit, 0, x_, p + "q_proj.weight_t", p + "q_proj.bias", qraw_, q_, "q_proj");
        EmitQkv(emit, 0, x_, p + "k_proj.weight_t", p + "k_proj.bias", kraw_, k_, "k_proj");
        EmitQkv(emit, 0, x_, p + "v_proj.weight_t", p + "v_proj.bias", vraw_, v_, "v_proj");
        EmitQkNorm(emit, 0, p, q3t, k3t, q3n, k3n);
        DevTensor q3r = qk_norm_ ? q3n : q3t;
        DevTensor k3r = qk_norm_ ? k3n : k3t;
        emit(0, Rope(q3r, k3r, cosv, sinv, "rope"));
        emit(0, Flash(q3r, k3r, v3t, mask_, a3t, smax_, ssum_, "flash"));
        Emit(0, Matmul(attn_, W(p + "o_proj.weight_t"), o_, "o_proj"));
        if (mode_.fused) {
            Emit(0, AddRmsNorm(*hcur, o_, g2, x_, rstd2_, *hnxt, "addnorm_ln2"));
        } else {
            Emit(0, Add(*hcur, o_, *hnxt, "add1"));
            Emit(0, RmsNorm(*hnxt, g2, x_, rstd2_, "ln2"));
        }
        Emit(0, Matmul(x_, W(p + "gate_proj.weight_t"), gate_, "gate"));
        Emit(0, Matmul(x_, W(p + "up_proj.weight_t"), up_, "up"));
        Emit(0, Silu(gate_, act_, "silu"));
        Emit(0, Mul(act_, up_, act_, "mul"));
        Emit(0, Matmul(act_, W(p + "down_proj.weight_t"), down_, "down"));
        if (l + 1 < L_) {
            DevTensor &gn = W(LPrefix(l + 1) + "input_layernorm.weight");
            if (mode_.fused) {
                Emit(0, AddRmsNorm(*hnxt, down_, gn, x_, rstd3_, *hcur, "addnorm_ln1"));
            } else {
                Emit(0, Add(*hnxt, down_, *hcur, "add2"));
                Emit(0, RmsNorm(*hcur, gn, x_, rstd3_, "ln1_next"));
            }
        } else {
            DevTensor &gf = W("final_norm.weight");
            if (mode_.fused) {
                Emit(0, AddRmsNorm(*hnxt, down_, gf, xf_, rstd3_, hf_, "addnorm_final"));
            } else {
                Emit(0, Add(*hnxt, down_, hf_, "add_final"));
                Emit(0, RmsNorm(hf_, gf, xf_, rstd3_, "ln_final"));
            }
            Emit(0, Matmul(xf_, W("lm_head.weight_t"), logits_, "lm_head"));
        }
    }
}

void Qwen::SplitProgram()
{
    for (auto &it : prog_) {
        if (it.stream == 0) aicActs_.push_back(it);
        else aivActs_.push_back(it);
    }
}

void Qwen::RunProgramDirect()
{
    if (mode_.serial) {
        for (auto &it : prog_) it.fn(serial_);
    } else {
        for (auto &it : prog_) it.fn(it.stream == 0 ? aic_ : aiv_);
    }
}

void Qwen::RunOnce()
{
    if (mode_.graph && (gSerial_ || gAic_)) {
        if (mode_.serial) {
            Check(aclmdlRIExecuteAsync(gSerial_, serial_), "replay serial");
        } else {
            Check(aclmdlRIExecuteAsync(gAic_, aic_), "replay aic");
            Check(aclmdlRIExecuteAsync(gAiv_, aiv_), "replay aiv");
        }
        return;
    }
    RunProgramDirect();
}

void Qwen::CaptureAll()
{
    Check(aclrtSynchronizeDevice(), "pre capture sync");
    if (mode_.serial) {
        Check(aclmdlRICaptureBegin(serial_, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), "cap begin");
        for (auto &it : prog_) it.fn(serial_);
        Check(aclmdlRICaptureEnd(serial_, &gSerial_), "cap end");
        Check(aclrtSynchronizeStream(serial_), "cap sync");
        return;
    }
    Check(aclmdlRICaptureBegin(aic_, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), "cap aic");
    for (auto &it : aicActs_) it.fn(aic_);
    Check(aclmdlRICaptureEnd(aic_, &gAic_), "cap aic end");
    Check(aclrtSynchronizeStream(aic_), "cap aic sync");
    Check(aclmdlRICaptureBegin(aiv_, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), "cap aiv");
    for (auto &it : aivActs_) it.fn(aiv_);
    Check(aclmdlRICaptureEnd(aiv_, &gAiv_), "cap aiv end");
    Check(aclrtSynchronizeStream(aiv_), "cap aiv sync");
}

void Qwen::Bench(int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i) {
        RunOnce();
        Check(aclrtSynchronizeDevice(), "warmup sync");
    }
    samples_.clear();
    host_samples_.clear();
    for (int i = 0; i < iters; ++i) {
        Check(aclrtSynchronizeDevice(), "pre sync");
        auto t0 = Clock::now();
        RunOnce();
        auto th = Clock::now();
        Check(aclrtSynchronizeDevice(), "post sync");
        auto t1 = Clock::now();
        samples_.push_back(Us(t0, t1));
        host_samples_.push_back(Us(t0, th));
    }
    std::vector<double> sorted = samples_;
    std::sort(sorted.begin(), sorted.end());
    best_us_ = sorted.front();
    typical_us_ = sorted[sorted.size() / 2];
    std::vector<double> hs = host_samples_;
    std::sort(hs.begin(), hs.end());
    host_best_us_ = hs.front();
    host_typical_us_ = hs[hs.size() / 2];
}

void Qwen::CommitTiming(const std::vector<double> &samples)
{
    samples_ = samples;
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    if (!sorted.empty()) {
        best_us_ = sorted.front();
        typical_us_ = sorted[sorted.size() / 2];
    }
    host_best_us_ = 0.0;
    host_typical_us_ = 0.0;
}

void Qwen::Verify()
{
    std::vector<uint16_t> host(static_cast<size_t>(V_));
    void *last = static_cast<char *>(logits_.ptr) + size_t(seq_ - 1) * size_t(V_) * 2;
    Check(aclrtMemcpy(host.data(), size_t(V_) * 2, last, size_t(V_) * 2,
                      ACL_MEMCPY_DEVICE_TO_HOST),
          "logits d2h");
    auto bf = [](uint16_t v) {
        uint32_t u = uint32_t(v) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    };
    std::ifstream rf(dir_ + "/ref_last_logits.bin", std::ios::binary);
    std::vector<float> ref(static_cast<size_t>(V_));
    rf.read(reinterpret_cast<char *>(ref.data()), std::streamsize(ref.size() * 4));
    double mx = 0, sum = 0;
    float bestv = -1e30f;
    int arg = 0;
    for (int i = 0; i < V_; ++i) {
        float v = bf(host[i]);
        mx = std::max(mx, std::fabs(double(v) - double(ref[i])));
        sum += std::fabs(double(v));
        if (v > bestv) {
            bestv = v;
            arg = i;
        }
    }
    verify_argmax_ = arg;
    verify_maxdiff_ = mx;
    verify_absmean_ = sum / V_;
    Value rj = nhp::json::ParseFile(dir_ + "/ref.json");
    ref_argmax_ = int(rj.At("argmax_last").number);
    flag_timeouts_ = 0;
    flag_edges_ = edge_count_;
    if (mode_.flags) {
        std::vector<int32_t> ctr(size_t(edge_count_) * 8);
        Check(aclrtMemcpy(ctr.data(), ctr.size() * 4, ctr_, ctr.size() * 4,
                          ACL_MEMCPY_DEVICE_TO_HOST),
              "ctr d2h");
        for (int e = 0; e < edge_count_; ++e) {
            if (ctr[size_t(e) * 8 + 2] != 1) ++flag_timeouts_;
        }
    }
}


void Qwen::ConfigureTP(int rank, int size, int split, HcclComm comm,
                       const std::string &mode)
{
    tp_ = true;
    tp_rank_ = rank;
    tp_size_ = std::max(1, size);
    tp_split_ = std::max(1, split);
    comm_ = comm;
    mode_name_ = mode;
}

void Qwen::EnqueueTP()
{
    for (auto &it : tp_prog_) {
        it.fn(it.stream == 0 ? serial_ : hccl_);
    }
}

void Qwen::LoadWeightsTP()
{
    int fd = ::open((dir_ + "/weights.bin").c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("open weights.bin");
    struct stat st {};
    ::fstat(fd, &st);
    weight_fd_ = fd;
    weight_bytes_ = size_t(st.st_size);
    weight_map_ = ::mmap(nullptr, weight_bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
    if (weight_map_ == MAP_FAILED) throw std::runtime_error("mmap weights");
    const auto *base_bytes = static_cast<const char *>(weight_map_);

    for (const auto &kv : manifest_) {
        const std::string &name = kv.first;
        const Value &v = kv.second;
        std::vector<int64_t> dims;
        for (const auto &d : v.At("shape").array) dims.push_back(int64_t(d.number));
        const size_t off = size_t(v.At("offset").number);
        const bool is_gate = name.find(".gate_proj.weight_t") != std::string::npos;
        const bool is_up = name.find(".up_proj.weight_t") != std::string::npos;
        const bool is_down = name.find(".down_proj.weight_t") != std::string::npos;
        if ((is_gate || is_up || is_down) && dims.size() == 2) {
            DevTensor t;
            const int64_t src_rows = dims[0], src_cols = dims[1];
            if (is_down) {
                const int64_t rows = src_rows / tp_size_;
                const size_t bytes = size_t(rows) * size_t(src_cols) * 2;
                Check(aclrtMalloc(&t.ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc down");
                Check(aclrtMemcpy(t.ptr, bytes,
                                  base_bytes + off +
                                      size_t(tp_rank_) * size_t(rows) * size_t(src_cols) * 2,
                                  bytes, ACL_MEMCPY_HOST_TO_DEVICE), "h2d down");
                t.bytes = bytes;
                t.t = MakeAclTensor(t.ptr, {rows, src_cols}, ACL_BF16);
            } else {
                const int64_t cols = src_cols / tp_size_;
                const int64_t c0 = int64_t(tp_rank_) * cols;
                std::vector<uint16_t> stage(size_t(src_rows) * size_t(cols));
                const auto *src16 = reinterpret_cast<const uint16_t *>(base_bytes + off);
                for (int64_t r = 0; r < src_rows; ++r) {
                    std::memcpy(stage.data() + size_t(r) * size_t(cols),
                                src16 + size_t(r) * size_t(src_cols) + size_t(c0),
                                size_t(cols) * 2);
                }
                const size_t bytes = size_t(src_rows) * size_t(cols) * 2;
                Check(aclrtMalloc(&t.ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc gate/up");
                Check(aclrtMemcpy(t.ptr, bytes, stage.data(), bytes,
                                  ACL_MEMCPY_HOST_TO_DEVICE), "h2d gate/up");
                t.bytes = bytes;
                t.t = MakeAclTensor(t.ptr, {src_rows, cols}, ACL_BF16);
            }
            weights_[name] = t;
            continue;
        }
        // rank1 不产出 logits，省掉 final norm 与 lm_head。
        if (tp_rank_ != 0 && (name == "lm_head.weight_t" || name == "final_norm.weight")) {
            continue;
        }
        const size_t bytes = size_t(v.At("bytes").number);
        DevTensor t;
        Check(aclrtMalloc(&t.ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc weight");
        Check(aclrtMemcpy(t.ptr, bytes, base_bytes + off, bytes, ACL_MEMCPY_HOST_TO_DEVICE),
              "h2d weight");
        t.bytes = bytes;
        t.t = MakeAclTensor(t.ptr, dims, ACL_BF16);
        weights_[name] = t;
    }
}

void Qwen::BuildTP()
{
    const int S = seq_;
    const int Dh = D_;
    const int H = H_;
    auto q3 = [&](DevTensor &d) { return MakeAclTensor(d.ptr, {1, S, HQ_, Dh}, ACL_BF16); };
    auto kv3 = [&](DevTensor &d) { return MakeAclTensor(d.ptr, {1, S, HKV_, Dh}, ACL_BF16); };
    DevTensor q3t{q_.ptr, 0, q3(q_)};
    DevTensor k3t{k_.ptr, 0, kv3(k_)};
    DevTensor v3t{v_.ptr, 0, kv3(v_)};
    DevTensor a3t{attn_.ptr, 0, q3(attn_)};
    DevTensor cosv{W("rope.cos").ptr, 0,
                   MakeAclTensor(W("rope.cos").ptr, {1, S, 1, Dh}, ACL_BF16)};
    DevTensor sinv{W("rope.sin").ptr, 0,
                   MakeAclTensor(W("rope.sin").ptr, {1, S, 1, Dh}, ACL_BF16)};
    DevTensor q3n{qn_.ptr, 0, qk_norm_ ? MakeAclTensor(qn_.ptr, {1, S, HQ_, Dh}, ACL_BF16)
                                       : q3t.t};
    DevTensor k3n{kn_.ptr, 0, qk_norm_ ? MakeAclTensor(kn_.ptr, {1, S, HKV_, Dh}, ACL_BF16)
                                       : k3t.t};
    const int tiles = tp_split_;

    std::vector<Item> prog;
    auto emit = [&](int st, Call *c) { EmitTo(prog, st, c); };
    DevTensor eout{h_[0].ptr, 0, MakeAclTensor(h_[0].ptr, {1, S, H}, ACL_BF16)};
    EmitTo(prog, 0, Embed(W("embed.weight"), ids_, eout, "tp_embed"));
    PPSync(prog, false);
    EmitTo(prog, 1, RmsNorm(h_[0], W(LPrefix(0) + "input_layernorm.weight"),
                            x_, rstd_, "tp_ln1_0"));
    PPSync(prog, true);

    for (int l = 0; l < L_; ++l) {
        const std::string p = LPrefix(l);
        EmitQkv(emit, 0, x_, p + "q_proj.weight_t", p + "q_proj.bias", qraw_, q_, "tp_q");
        EmitQkv(emit, 0, x_, p + "k_proj.weight_t", p + "k_proj.bias", kraw_, k_, "tp_k");
        EmitQkv(emit, 0, x_, p + "v_proj.weight_t", p + "v_proj.bias", vraw_, v_, "tp_v");
        PPSync(prog, false);
        EmitQkNorm(emit, 1, p, q3t, k3t, q3n, k3n);
        DevTensor q3r = qk_norm_ ? q3n : q3t;
        DevTensor k3r = qk_norm_ ? k3n : k3t;
        emit(1, Rope(q3r, k3r, cosv, sinv, "tp_rope"));
        emit(1, Flash(q3r, k3r, v3t, mask_, a3t, smax_, ssum_, "tp_flash"));
        PPSync(prog, true);
        EmitTo(prog, 0, Matmul(attn_, W(p + "o_proj.weight_t"), o_, "tp_o"));
        PPSync(prog, false);
        EmitTo(prog, 1, AddRmsNorm(h_[0], o_, W(p + "post_attention_layernorm.weight"),
                                   x_, rstd2_, h_[1], "tp_addnorm_ln2"));
        PPSync(prog, true);
        EmitTo(prog, 0, Matmul(x_, W(p + "gate_proj.weight_t"), gate_, "tp_gate"));
        EmitTo(prog, 0, Matmul(x_, W(p + "up_proj.weight_t"), up_, "tp_up"));
        PPSync(prog, false);
        EmitTo(prog, 1, Silu(gate_, act_, "tp_silu"));
        EmitTo(prog, 1, Mul(act_, up_, act_, "tp_mul"));
        PPSync(prog, true);

        for (int t = 0; t < tiles; ++t) {
            const int rows = S / tiles;
            const int off = t * rows;
            DevTensor act_t = SliceRowsEx(act_, off, rows, I_ / tp_size_, ACL_BF16, 2);
            DevTensor down_t = SliceRowsEx(down_, off, rows, H, ACL_BF16, 2);
            DevTensor red_t = SliceRowsEx(down_red_, off, rows, H, ACL_BF16, 2);
            EmitTo(prog, 0, Matmul(act_t, W(p + "down_proj.weight_t"), down_t, "tp_down"));
            PPSync(prog, false); // down AIC -> allreduce HCCL
            const uint64_t elems = uint64_t(rows) * uint64_t(H);
            EmitFnTo(prog, 1, [this, down_t, red_t, elems](aclrtStream s) {
                CheckHccl(HcclAllReduce(down_t.ptr, red_t.ptr, elems,
                                        HCCL_DATA_TYPE_BFP16, HCCL_REDUCE_SUM,
                                        comm_, s), "tp allreduce");
            });
            DevTensor h0t = SliceRowsEx(h_[0], off, rows, H, ACL_BF16, 2);
            DevTensor h1t = SliceRowsEx(h_[1], off, rows, H, ACL_BF16, 2);
            DevTensor xt = SliceRowsEx(x_, off, rows, H, ACL_BF16, 2);
            DevTensor rt = SliceRowsEx(rstd3_, off, rows, 1, ACL_FLOAT, 4);
            if (l + 1 < L_) {
                EmitTo(prog, 1, AddRmsNorm(h1t, red_t,
                                           W(LPrefix(l + 1) + "input_layernorm.weight"),
                                           xt, rt, h0t, "tp_addnorm_ln1"));
            } else if (tp_rank_ == 0) {
                DevTensor hft = SliceRowsEx(hf_, off, rows, H, ACL_BF16, 2);
                DevTensor xft = SliceRowsEx(xf_, off, rows, H, ACL_BF16, 2);
                EmitTo(prog, 1, AddRmsNorm(h1t, red_t, W("final_norm.weight"),
                                           xft, rt, hft, "tp_final_norm"));
            }
        }
        PPSync(prog, true); // addnorm AIV -> 下一层 q AIC
    }
    if (tp_rank_ == 0) {
        EmitTo(prog, 0, Matmul(xf_, W("lm_head.weight_t"), logits_, "tp_lm"));
    }
    tp_prog_ = std::move(prog);
}

void Qwen::ConfigurePP(int stage, int stages, int requests, HcclComm comm,
                       const std::string &mode)
{
    pp_ = true;
    pp_stage_ = stage;
    pp_stages_ = std::max(1, stages);
    pp_requests_ = std::max(1, requests);
    comm_ = comm;
    mode_name_ = mode;
}

void Qwen::EnqueuePP(int request)
{
    if (request < 0 || request >= int(pp_prog_.size())) {
        throw std::runtime_error("pp request out of range");
    }
    for (auto &it : pp_prog_[request]) {
        if (it.stream == 0) {
            it.fn(serial_);
        } else {
            it.fn(hccl_);
        }
    }
}

void Qwen::EmitQkv(const EmitFn &emit, int stream, const DevTensor &x,
                  const std::string &w, const std::string &b,
                  const DevTensor &raw, const DevTensor &out, const char *tag)
{
    emit(stream, Matmul(x, W(w), has_qkv_bias_ ? raw : out, tag));
    if (has_qkv_bias_) {
        emit(stream, Add(raw, W(b), out, (std::string(tag) + "_b").c_str()));
    }
}

void Qwen::EmitQkNorm(const EmitFn &emit, int stream, const std::string &p,
                      const DevTensor &q3t, const DevTensor &k3t,
                      const DevTensor &q3n, const DevTensor &k3n)
{
    if (!qk_norm_) return;
    DevTensor q2{q3t.ptr, 0, MakeAclTensor(q3t.ptr, {seq_ * HQ_, D_}, ACL_BF16)};
    DevTensor k2{k3t.ptr, 0, MakeAclTensor(k3t.ptr, {seq_ * HKV_, D_}, ACL_BF16)};
    DevTensor qo{q3n.ptr, 0, MakeAclTensor(q3n.ptr, {seq_ * HQ_, D_}, ACL_BF16)};
    DevTensor ko{k3n.ptr, 0, MakeAclTensor(k3n.ptr, {seq_ * HKV_, D_}, ACL_BF16)};
    emit(stream, RmsNorm(q2, W(p + "q_norm.weight"), qo, rstdq_, "q_norm"));
    emit(stream, RmsNorm(k2, W(p + "k_norm.weight"), ko, rstdk_, "k_norm"));
}

void Qwen::BuildPP()
{
    const int S = seq_;
    const int H = H_;
    for (int r = 0; r < pp_requests_; ++r) {
        hreq_[r][0] = AllocTensor({S, H}, ACL_BF16, size_t(S) * H * 2);
        hreq_[r][1] = AllocTensor({S, H}, ACL_BF16, size_t(S) * H * 2);
    }

    auto q3 = [&](DevTensor &d) {
        return MakeAclTensor(d.ptr, {1, S, HQ_, D_}, ACL_BF16);
    };
    auto kv3 = [&](DevTensor &d) {
        return MakeAclTensor(d.ptr, {1, S, HKV_, D_}, ACL_BF16);
    };
    DevTensor q3t{q_.ptr, 0, q3(q_)};
    DevTensor k3t{k_.ptr, 0, kv3(k_)};
    DevTensor v3t{v_.ptr, 0, kv3(v_)};
    DevTensor a3t{attn_.ptr, 0, q3(attn_)};
    DevTensor cosv{W("rope.cos").ptr, 0,
                   MakeAclTensor(W("rope.cos").ptr, {1, S, 1, D_}, ACL_BF16)};
    DevTensor sinv{W("rope.sin").ptr, 0,
                   MakeAclTensor(W("rope.sin").ptr, {1, S, 1, D_}, ACL_BF16)};
    DevTensor q3n{qn_.ptr, 0, qk_norm_ ? MakeAclTensor(qn_.ptr, {1, S, HQ_, D_}, ACL_BF16)
                                       : q3t.t};
    DevTensor k3n{kn_.ptr, 0, qk_norm_ ? MakeAclTensor(kn_.ptr, {1, S, HKV_, D_}, ACL_BF16)
                                       : k3t.t};
    const uint64_t sendElems = uint64_t(S) * uint64_t(H);

    // 层内算子与请求无关（同一 stage 的请求在流上串行执行），只建一次，
    // 避免 R 个请求重复分配 workspace。
    struct LayerCalls {
        Call *q = nullptr;
        Call *qb = nullptr;
        Call *k = nullptr;
        Call *kb = nullptr;
        Call *v = nullptr;
        Call *vb = nullptr;
        Call *qn = nullptr;
        Call *kn = nullptr;
        Call *rope = nullptr;
        Call *flash = nullptr;
        Call *o = nullptr;
        Call *gate = nullptr;
        Call *up = nullptr;
        Call *silu = nullptr;
        Call *mul = nullptr;
        Call *down = nullptr;
    };
    std::vector<LayerCalls> lc;
    lc.reserve(size_t(l1_ - l0_));
    for (int l = l0_; l < l1_; ++l) {
        const std::string p = LPrefix(l);
        LayerCalls c;
        c.q = Matmul(x_, W(p + "q_proj.weight_t"), has_qkv_bias_ ? qraw_ : q_, "pp_q");
        c.k = Matmul(x_, W(p + "k_proj.weight_t"), has_qkv_bias_ ? kraw_ : k_, "pp_k");
        c.v = Matmul(x_, W(p + "v_proj.weight_t"), has_qkv_bias_ ? vraw_ : v_, "pp_v");
        if (has_qkv_bias_) {
            c.qb = Add(qraw_, W(p + "q_proj.bias"), q_, "pp_qb");
            c.kb = Add(kraw_, W(p + "k_proj.bias"), k_, "pp_kb");
            c.vb = Add(vraw_, W(p + "v_proj.bias"), v_, "pp_vb");
        }
        if (qk_norm_) {
            DevTensor q2{q3t.ptr, 0, MakeAclTensor(q3t.ptr, {S * HQ_, D_}, ACL_BF16)};
            DevTensor k2{k3t.ptr, 0, MakeAclTensor(k3t.ptr, {S * HKV_, D_}, ACL_BF16)};
            DevTensor qo{q3n.ptr, 0, MakeAclTensor(q3n.ptr, {S * HQ_, D_}, ACL_BF16)};
            DevTensor ko{k3n.ptr, 0, MakeAclTensor(k3n.ptr, {S * HKV_, D_}, ACL_BF16)};
            c.qn = RmsNorm(q2, W(p + "q_norm.weight"), qo, rstdq_, "pp_qn");
            c.kn = RmsNorm(k2, W(p + "k_norm.weight"), ko, rstdk_, "pp_kn");
        }
        DevTensor q3r = qk_norm_ ? q3n : q3t;
        DevTensor k3r = qk_norm_ ? k3n : k3t;
        c.rope = Rope(q3r, k3r, cosv, sinv, "pp_rope");
        c.flash = Flash(q3r, k3r, v3t, mask_, a3t, smax_, ssum_, "pp_flash");
        c.o = Matmul(attn_, W(p + "o_proj.weight_t"), o_, "pp_o");
        c.gate = Matmul(x_, W(p + "gate_proj.weight_t"), gate_, "pp_gate");
        c.up = Matmul(x_, W(p + "up_proj.weight_t"), up_, "pp_up");
        c.silu = Silu(gate_, act_, "pp_silu");
        c.mul = Mul(act_, up_, act_, "pp_mul");
        c.down = Matmul(act_, W(p + "down_proj.weight_t"), down_, "pp_down");
        lc.push_back(c);
    }
    Call *lm = nullptr;
    if (pp_stage_ == pp_stages_ - 1) {
        lm = Matmul(xf_, W("lm_head.weight_t"), logits_, "pp_lm");
    }

    for (int r = 0; r < pp_requests_; ++r) {
        std::vector<Item> prog;
        DevTensor &h0 = hreq_[r][0]; // 残差主缓冲
        DevTensor &h1 = hreq_[r][1]; // attention 残差临时缓冲
        const int start_l = l0_, end_l = l1_;
        if (pp_stage_ == 0) {
            DevTensor eout{h0.ptr, 0, MakeAclTensor(h0.ptr, {1, S, H}, ACL_BF16)};
            EmitTo(prog, 0, Embed(W("embed.weight"), ids_, eout, "pp_embed"));
            PPSync(prog, false); // embed AIC -> ln1 AIV
            EmitTo(prog, 1,
                   RmsNorm(h0, W(LPrefix(0) + "input_layernorm.weight"),
                           x_, rstd_, "pp_ln1_0"));
            PPSync(prog, true); // ln1 AIV -> q AIC
        } else {
            // recv 与 ln1 同在 hccl/AIV stream，直接保序；
            // AIC 上的 q/k/v 由 ln1 后的 GM flag 边约束。
            EmitFnTo(prog, 1, [this, h0, sendElems](aclrtStream s) {
                CheckHccl(HcclRecv(h0.ptr, sendElems, HCCL_DATA_TYPE_BFP16, 0,
                                   comm_, s), "pp recv");
            });
            EmitTo(prog, 1,
                   RmsNorm(h0, W(LPrefix(start_l) + "input_layernorm.weight"),
                           x_, rstd_, "pp_ln1_first"));
            PPSync(prog, true);
        }

        for (int l = start_l; l < end_l; ++l) {
            const std::string p = LPrefix(l);
            const LayerCalls &c = lc[size_t(l - start_l)];
            EmitTo(prog, 0, c.q);
            if (c.qb) EmitTo(prog, 0, c.qb);
            EmitTo(prog, 0, c.k);
            if (c.kb) EmitTo(prog, 0, c.kb);
            EmitTo(prog, 0, c.v);
            if (c.vb) EmitTo(prog, 0, c.vb);
            PPSync(prog, false);
            if (c.qn) { EmitTo(prog, 1, c.qn); EmitTo(prog, 1, c.kn); }
            EmitTo(prog, 1, c.rope);
            EmitTo(prog, 1, c.flash);
            PPSync(prog, true);
            EmitTo(prog, 0, c.o);
            PPSync(prog, false);
            EmitTo(prog, 1, AddRmsNorm(h0, o_, W(p + "post_attention_layernorm.weight"),
                                       x_, rstd2_, h1, "pp_addnorm_ln2"));
            PPSync(prog, true);
            EmitTo(prog, 0, c.gate);
            EmitTo(prog, 0, c.up);
            PPSync(prog, false);
            EmitTo(prog, 1, c.silu);
            EmitTo(prog, 1, c.mul);
            PPSync(prog, true);
            EmitTo(prog, 0, c.down);
            PPSync(prog, false);

            const bool last_global = (l == L_ - 1);
            const bool last_stage = (l == end_l - 1);
            if (last_global) {
                EmitTo(prog, 1, AddRmsNorm(h1, down_, W("final_norm.weight"),
                                           xf_, rstd3_, hf_, "pp_final_norm"));
                PPSync(prog, true); // final norm AIV -> lm_head AIC
                EmitTo(prog, 0, lm);
            } else if (last_stage && pp_stage_ == 0) {
                // 只做残差，下一 stage 自己算 ln1；只传一个 hidden。
                EmitTo(prog, 1, Add(h1, down_, h0, "pp_residual_out"));
            } else {
                EmitTo(prog, 1,
                       AddRmsNorm(h1, down_, W(LPrefix(l + 1) + "input_layernorm.weight"),
                                  x_, rstd3_, h0, "pp_addnorm_ln1"));
                PPSync(prog, true); // ln1 AIV -> 下一层 q AIC
            }
        }

        if (pp_stage_ == 0) {
            pp_out_[r] = h0;
            // 残差 add 与 send 都在 hccl/AIV stream，保序即可。
            EmitFnTo(prog, 1, [this, h0, sendElems](aclrtStream s) {
                CheckHccl(HcclSend(h0.ptr, sendElems, HCCL_DATA_TYPE_BFP16, 1,
                                   comm_, s), "pp send");
            });
        }
        pp_prog_.push_back(std::move(prog));
    }
}

void Qwen::VerifyPP()
{
    Verify();
}

void Qwen::ProfileMany(const std::string &spec_json, std::string &result)
{
    Value spec = nhp::json::ParseFile(spec_json);
    const int64_t rows = seq_;
    std::vector<std::string> cases;
    for (const auto &c : spec.At("cases").array) {
        const std::string op = c.Str("op");
        const int k = int(c.Num("k", 1));
        if (k < 1 || rows % k != 0) continue;
        const int64_t per = rows / k;
        std::vector<Call *> calls;
        std::string shape;
        try {
            if (op.rfind("matmul_", 0) == 0) {
                DevTensor *A = &x_;
                DevTensor *C = &qraw_;
                DevTensor *B = nullptr;
                int64_t K = H_, N = H_;
                const char *tag = "profile_mm";
                if (op == "matmul_q") {
                    B = &W("L00.q_proj.weight_t"); C = &qraw_;
                    N = int64_t(HQ_) * D_;
                } else if (op == "matmul_o") {
                    A = &attn_; B = &W("L00.o_proj.weight_t"); C = &o_;
                    K = int64_t(HQ_) * D_;
                } else if (op == "matmul_k") {
                    B = &W("L00.k_proj.weight_t"); C = &kraw_;
                    N = int64_t(HKV_) * D_;
                } else if (op == "matmul_v") {
                    B = &W("L00.v_proj.weight_t"); C = &vraw_;
                    N = int64_t(HKV_) * D_;
                } else if (op == "matmul_gate") {
                    B = &W("L00.gate_proj.weight_t"); C = &gate_; N = I_;
                } else if (op == "matmul_up") {
                    B = &W("L00.up_proj.weight_t"); C = &up_; N = I_;
                } else if (op == "matmul_down") {
                    A = &act_; B = &W("L00.down_proj.weight_t"); C = &down_;
                    K = I_; N = H_;
                } else if (op == "matmul_lm") {
                    A = &xf_; B = &W("lm_head.weight_t"); C = &logits_;
                    N = V_;
                } else {
                    continue;
                }
                shape = "matmul|" + std::to_string(rows) + "x" +
                        std::to_string(K) + "x" + std::to_string(N);
                for (int i = 0; i < k; ++i) {
                    calls.push_back(Matmul(
                        SliceRows(*A, i * per, per, K, ACL_BF16),
                        *B,
                        SliceRows(*C, i * per, per, N, ACL_BF16), tag));
                }
            } else if (op == "add") {
                shape = "add|" + std::to_string(rows) + "x" + std::to_string(H_);
                for (int i = 0; i < k; ++i) {
                    calls.push_back(Add(
                        SliceRows(h_[0], i * per, per, H_, ACL_BF16),
                        SliceRows(o_, i * per, per, H_, ACL_BF16),
                        SliceRows(h_[1], i * per, per, H_, ACL_BF16),
                        "profile_add"));
                }
            } else if (op == "rmsnorm") {
                shape = "rmsnorm|" + std::to_string(rows) + "x" + std::to_string(H_);
                for (int i = 0; i < k; ++i) {
                    calls.push_back(RmsNorm(
                        SliceRows(x_, i * per, per, H_, ACL_BF16),
                        W("L00.input_layernorm.weight"),
                        SliceRows(qraw_, i * per, per, H_, ACL_BF16),
                        SliceRowsEx(rstd_, i * per, per, 1, ACL_FLOAT, 4),
                        "profile_rmsnorm"));
                }
            } else if (op == "add_rmsnorm") {
                shape = "add_rmsnorm|" + std::to_string(rows) + "x" + std::to_string(H_);
                for (int i = 0; i < k; ++i) {
                    calls.push_back(AddRmsNorm(
                        SliceRows(h_[0], i * per, per, H_, ACL_BF16),
                        SliceRows(o_, i * per, per, H_, ACL_BF16),
                        W("L00.post_attention_layernorm.weight"),
                        SliceRows(x_, i * per, per, H_, ACL_BF16),
                        SliceRowsEx(rstd2_, i * per, per, 1, ACL_FLOAT, 4),
                        SliceRows(h_[1], i * per, per, H_, ACL_BF16),
                        "profile_addnorm"));
                }
            } else if (op == "rope") {
                shape = "rope|" + std::to_string(rows) + "x" +
                        std::to_string(HQ_ * D_ + HKV_ * D_);
                for (int i = 0; i < k; ++i) {
                    calls.push_back(Rope(
                        SliceRows4D(q_, i * per, per, HQ_, D_, ACL_BF16),
                        SliceRows4D(k_, i * per, per, HKV_, D_, ACL_BF16),
                        SliceRows4D(W("rope.cos"), i * per, per, 1, D_, ACL_BF16),
                        SliceRows4D(W("rope.sin"), i * per, per, 1, D_, ACL_BF16),
                        "profile_rope"));
                }
            } else if (op == "silu") {
                shape = "silu|" + std::to_string(rows) + "x" + std::to_string(I_);
                for (int i = 0; i < k; ++i) {
                    calls.push_back(Silu(
                        SliceRows(gate_, i * per, per, I_, ACL_BF16),
                        SliceRows(act_, i * per, per, I_, ACL_BF16),
                        "profile_silu"));
                }
            } else if (op == "mul") {
                shape = "mul|" + std::to_string(rows) + "x" + std::to_string(I_);
                for (int i = 0; i < k; ++i) {
                    calls.push_back(Mul(
                        SliceRows(act_, i * per, per, I_, ACL_BF16),
                        SliceRows(up_, i * per, per, I_, ACL_BF16),
                        SliceRows(act_, i * per, per, I_, ACL_BF16),
                        "profile_mul"));
                }
            } else if (op == "embedding") {
                shape = "embedding|" + std::to_string(rows) + "x" +
                        std::to_string(H_);
                for (int i = 0; i < k; ++i) {
                    DevTensor idt;
                    idt.ptr = static_cast<char *>(ids_.ptr) + i * per * 8;
                    idt.bytes = size_t(per) * 8;
                    idt.t = MakeAclTensor(idt.ptr, {1, per}, ACL_INT64);
                    DevTensor ot;
                    ot.ptr = static_cast<char *>(h_[0].ptr) +
                             i * per * H_ * 2;
                    ot.bytes = size_t(per) * size_t(H_) * 2;
                    ot.t = MakeAclTensor(ot.ptr, {1, per, H_}, ACL_BF16);
                    calls.push_back(Embed(W("embed.weight"), idt, ot,
                                          "profile_embed"));
                }
            } else if (op == "attention") {
                if (int64_t(HQ_) % k != 0) continue;
                const int64_t hq = HQ_ / k;
                const bool kv_div = int64_t(HKV_) % k == 0;
                const int64_t hkv = kv_div ? HKV_ / k : 1;
                shape = "attention|" + std::to_string(HQ_) + "x" +
                        std::to_string(rows) + "x" + std::to_string(D_);
                if (k == 1) {
                    calls.push_back(FlashN(
                        SliceRows4D(q_, 0, rows, HQ_, D_, ACL_BF16),
                        SliceRows4D(k_, 0, rows, HKV_, D_, ACL_BF16),
                        SliceRows4D(v_, 0, rows, HKV_, D_, ACL_BF16),
                        mask_,
                        SliceRows4D(attn_, 0, rows, HQ_, D_, ACL_BF16),
                        smax_, ssum_, HQ_, "profile_attn"));
                } else {
                    for (int g = 0; g < k; ++g) {
                        // k > HKV 时按 E1 的 GQA 复制策略：每组 1 个 KV 头，
                        // 在 HKV 个 KV 头间循环复制。
                        const int64_t kv_off = kv_div ? int64_t(g) * hkv
                                                      : int64_t(g % HKV_);
                        calls.push_back(FlashN(
                            SliceHeads4D(q_, int64_t(g) * hq, hq, HQ_, rows, D_,
                                         ACL_BF16),
                            SliceHeads4D(k_, kv_off, hkv, HKV_, rows, D_, ACL_BF16),
                            SliceHeads4D(v_, kv_off, hkv, HKV_, rows, D_, ACL_BF16),
                            mask_,
                            SliceHeads4D(attn_, int64_t(g) * hq, hq, HQ_, rows, D_,
                                         ACL_BF16),
                            SliceHeadsSmax(smax_, int64_t(g) * hq, hq, HQ_, rows),
                            SliceHeadsSmax(ssum_, int64_t(g) * hq, hq, HQ_, rows),
                            int(hq), "profile_attn"));
                    }
                }
            } else {
                continue;
            }
        } catch (const std::exception &e) {
            cases.push_back("{\"op\":\"" + op + "\",\"k\":" + std::to_string(k) +
                            ",\"error\":\"" + e.what() + "\"}");
            continue;
        }

        // 先捕获得 k 个子 kernel 的图；图重放可以去掉逐 op 的 host 空泡，
        // 得到纯设备时间。捕获失败的算子回退直接入队并标记 graph=false。
        aclmdlRI graph = nullptr;
        bool captured = false;
        try {
            Check(aclmdlRICaptureBegin(serial_, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),
                  "profile capture begin");
            for (auto *c : calls) c->run(serial_);
            Check(aclmdlRICaptureEnd(serial_, &graph), "profile capture end");
            captured = true;
        } catch (const std::exception &) {
            captured = false;
        }
        aclrtStream runStream = serial_;
        if (!captured) Check(aclrtCreateStream(&runStream), "profile fallback stream");
        const int repeats = captured ? 20 : 3;
        auto dispatch = [&]() {
            if (captured) {
                Check(aclmdlRIExecuteAsync(graph, runStream), "profile replay");
            } else {
                for (auto *c : calls) c->run(runStream);
            }
        };
        for (int w = 0; w < 2; ++w) {
            dispatch();
            Check(aclrtSynchronizeStream(runStream), "profile warmup");
        }
        aclrtEvent ev0 = nullptr, ev1 = nullptr;
        Check(aclrtCreateEvent(&ev0), "profile event0");
        Check(aclrtCreateEvent(&ev1), "profile event1");
        std::vector<double> wall, dev, host;
        for (int it = 0; it < 3; ++it) {
            Check(aclrtRecordEvent(ev0, runStream), "record ev0");
            const auto t0 = Clock::now();
            for (int r = 0; r < repeats; ++r) dispatch();
            const auto t_enq = Clock::now();
            Check(aclrtRecordEvent(ev1, runStream), "record ev1");
            Check(aclrtSynchronizeStream(runStream), "profile sync");
            const auto t1 = Clock::now();
            float ms = 0.0f;
            Check(aclrtEventElapsedTime(&ms, ev0, ev1), "event elapsed");
            host.push_back(Us(t0, t_enq) / repeats);
            dev.push_back(double(ms) * 1000.0 / repeats);
            wall.push_back(Us(t0, t1) / repeats);
        }
        Check(aclrtDestroyEvent(ev0), "destroy ev0");
        Check(aclrtDestroyEvent(ev1), "destroy ev1");
        auto best = [](const std::vector<double> &v) {
            return *std::min_element(v.begin(), v.end());
        };
        auto median = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        std::ostringstream os;
        os << "{\"op\":\"" << op << "\",\"k\":" << k << ",\"rows\":" << rows
           << ",\"shape\":\"" << shape << "\",\"best_us\":" << best(dev)
           << ",\"median_us\":" << median(dev)
           << ",\"per_tile_us\":" << best(dev) / k
           << ",\"wall_best_us\":" << best(wall)
           << ",\"host_enqueue_us\":" << best(host)
           << ",\"graph\":" << (captured ? "true" : "false")
           << ",\"samples_us\":[";
        for (size_t i = 0; i < dev.size(); ++i) {
            if (i) os << ",";
            os << dev[i];
        }
        os << "]}";
        cases.push_back(os.str());
    }
    std::ostringstream out;
    out << "{\"model\":\"" << dir_ << "\",\"cases\":[";
    for (size_t i = 0; i < cases.size(); ++i) {
        if (i) out << ",";
        out << cases[i];
    }
    out << "]}";
    result = out.str();
}

std::string Qwen::ResultJson(const std::string &mode, int warmup, int iters) const
{
    std::ostringstream os;
    os << "{\"mode\":\"" << mode << "\",\"seq\":" << seq_
       << ",\"device\":" << device_ << ",\"warmup\":" << warmup
       << ",\"iters\":" << iters << ",\"best_us\":" << best_us_
       << ",\"typical_us\":" << typical_us_ << ",\"samples_us\":[";
    for (size_t i = 0; i < samples_.size(); ++i) {
        if (i) os << ",";
        os << samples_[i];
    }
    os << "],\"argmax\":" << verify_argmax_ << ",\"ref_argmax\":" << ref_argmax_
       << ",\"maxdiff\":" << verify_maxdiff_ << ",\"abs_mean\":" << verify_absmean_
       << ",\"flag_edges\":" << flag_edges_
       << ",\"flag_timeouts\":" << flag_timeouts_
       << ",\"host_best_us\":" << host_best_us_
       << ",\"host_typical_us\":" << host_typical_us_ << "}";
    return os.str();
}

// ---------------- main ----------------
struct Args {
    std::string model;
    std::string mode = "serial";
    int seq = 2048;
    int device = 0;
    int device2 = -1;
    int requests = 2;
    int tiles = 2;
    int warmup = 3;
    int iters = 10;
    std::string out;
};

Args ParseArgs(int argc, char **argv)
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if (k == "--model") a.model = next();
        else if (k == "--mode") a.mode = next();
        else if (k == "--seq") a.seq = std::stoi(next());
        else if (k == "--device") a.device = std::stoi(next());
        else if (k == "--device2") a.device2 = std::stoi(next());
        else if (k == "--requests") a.requests = std::stoi(next());
        else if (k == "--tiles") a.tiles = std::stoi(next());
        else if (k == "--warmup") a.warmup = std::stoi(next());
        else if (k == "--iters") a.iters = std::stoi(next());
        else if (k == "--out") a.out = next();
        else throw std::runtime_error("unknown arg " + k);
    }
    if (a.model.empty()) throw std::runtime_error("--model required");
    return a;
}

class Barrier {
public:
    explicit Barrier(int n) : n_(n) {}
    void Wait()
    {
        std::unique_lock<std::mutex> lk(m_);
        const int gen = gen_;
        if (++count_ == n_) {
            count_ = 0;
            ++gen_;
            cv_.notify_all();
        } else {
            cv_.wait(lk, [&] { return gen_ != gen; });
        }
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    int n_;
    int count_ = 0;
    int gen_ = 0;
};

// 单进程双卡 TP-MLP：attention 复制，MLP 按 gate/up 列、down 行各切一半，
// 每层 down 的部分和做一次 HCCL AllReduce。tp2=整幅 allreduce；tp2_split=
// down 按行切 tiles，allreduce/AddNorm 逐 tile 提前放行（KernRelay 调度）。
int RunQwenTP(const Args &args, bool split, std::string &result)
{
    if (args.device2 < 0 || args.device2 == args.device) {
        throw std::runtime_error("--device2 must be a different device");
    }
    if (args.tiles < 1 || args.tiles > 8) {
        throw std::runtime_error("--tiles must be in 1..8");
    }
    Check(aclInit(nullptr), "aclInit");
    int devices[2] = {args.device, args.device2};
    HcclComm comms[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i) {
        Check(aclrtSetDevice(devices[i]), "warmup set device");
        aclrtStream warm = nullptr;
        Check(aclrtCreateStream(&warm), "warmup stream");
        Check(aclrtSynchronizeDevice(), "warmup sync");
        (void)aclrtDestroyStream(warm);
    }
    CheckHccl(HcclCommInitAll(2, devices, comms), "hccl init all");

    Barrier bar(3);
    std::mutex mu;
    std::string err;
    std::string stage_json[2];
    std::vector<double> wall_samples;
    auto fail = [&](const std::string &what) {
        std::lock_guard<std::mutex> lk(mu);
        if (err.empty()) err = what;
    };
    const std::string mode = split ? "tp2_split" : "tp2";
    const int tiles = split ? args.tiles : 1;

    auto worker = [&](int idx) {
        Qwen q(args.model, args.seq, devices[idx], mode);
        bool ok = true;
        std::vector<double> samples;
        try {
            Check(aclrtSetDevice(devices[idx]), "set device");
            q.ConfigureTP(idx, 2, tiles, comms[idx], mode);
            q.Build();
            q.EnqueueTP();
            Check(aclrtSynchronizeDevice(), "tp first sync");
            if (idx == 0) q.Verify();
        } catch (const std::exception &e) {
            ok = false;
            fail(std::string("rank") + std::to_string(idx) + ": " + e.what());
        }
        auto one = [&]() {
            q.EnqueueTP();
            Check(aclrtSynchronizeDevice(), "tp sync");
        };
        for (int w = 0; w < args.warmup; ++w) {
            bar.Wait();
            if (ok) {
                try { one(); } catch (const std::exception &e) {
                    ok = false;
                    fail(std::string("rank") + std::to_string(idx) + ": " + e.what());
                }
            }
            bar.Wait();
        }
        for (int i = 0; i < args.iters; ++i) {
            bar.Wait();
            const auto t0 = Clock::now();
            if (ok) {
                try { one(); } catch (const std::exception &e) {
                    ok = false;
                    fail(std::string("rank") + std::to_string(idx) + ": " + e.what());
                }
            }
            samples.push_back(Us(t0, Clock::now()));
            bar.Wait();
        }
        if (ok) {
            try {
                q.CommitTiming(samples);
                if (idx == 0) q.Verify();
                stage_json[idx] = q.ResultJson(
                    idx == 0 ? "tp_rank0" : "tp_rank1", args.warmup, args.iters);
            } catch (const std::exception &e) {
                fail(std::string("rank") + std::to_string(idx) + ": " + e.what());
            }
        }
        bar.Wait();
        if (comms[idx]) HcclCommDestroy(comms[idx]);
        aclrtResetDevice(devices[idx]);
    };

    std::thread t0(worker, 0);
    std::thread t1(worker, 1);
    for (int w = 0; w < args.warmup * 2; ++w) bar.Wait();
    for (int i = 0; i < args.iters; ++i) {
        bar.Wait();
        const auto t0 = Clock::now();
        bar.Wait();
        wall_samples.push_back(Us(t0, Clock::now()));
    }
    bar.Wait();
    t0.join();
    t1.join();
    if (!err.empty()) throw std::runtime_error(err);
    std::vector<double> ws = wall_samples;
    std::sort(ws.begin(), ws.end());
    std::ostringstream os;
    os << "{\"mode\":\"" << (split ? "tp2_split" : "tp2")
       << "\",\"seq\":" << args.seq << ",\"tiles\":" << tiles
       << ",\"devices\":[" << devices[0] << "," << devices[1] << "]"
       << ",\"wall_best_us\":" << ws.front()
       << ",\"wall_typical_us\":" << ws[ws.size() / 2]
       << ",\"wall_samples_us\":[";
    for (size_t i = 0; i < wall_samples.size(); ++i) {
        if (i) os << ",";
        os << wall_samples[i];
    }
    os << "],\"stages\":[" << stage_json[0] << "," << stage_json[1] << "]}";
    result = os.str();
    Check(aclFinalize(), "finalize");
    return 0;
}

// 单进程双卡 PP：stage0/stage1 各自在线程内 set device、建模、入队；
// 主线程只做 barrier 与 wall 计时，不碰设备。
int RunQwenPP(const Args &args, bool pipeline, std::string &result)
{
    if (args.device2 < 0 || args.device2 == args.device) {
        throw std::runtime_error("--device2 must be a different device");
    }
    if (args.requests < 1 || args.requests > 8) {
        throw std::runtime_error("--requests must be in 1..8");
    }
    Check(aclInit(nullptr), "aclInit");
    int devices[2] = {args.device, args.device2};
    HcclComm comms[2] = {nullptr, nullptr};
    // HcclCommInitAll 前必须先为每个 device 建立 runtime 上下文
    for (int i = 0; i < 2; ++i) {
        Check(aclrtSetDevice(devices[i]), "warmup set device");
        aclrtStream warm = nullptr;
        Check(aclrtCreateStream(&warm), "warmup stream");
        Check(aclrtSynchronizeDevice(), "warmup sync");
        (void)aclrtDestroyStream(warm);
    }
    CheckHccl(HcclCommInitAll(2, devices, comms), "hccl init all");

    Barrier bar(3);
    std::mutex mu;
    std::string err;
    std::string stage_json[2];
    std::vector<double> wall;
    // 真串行基线：stage0 的第 r 个请求要等 stage1 完成第 r-1 个请求。
    std::mutex seq_mu;
    std::condition_variable seq_cv;
    int stage1_done = 0;
    auto fail = [&](const std::string &what) {
        std::lock_guard<std::mutex> lk(mu);
        if (err.empty()) err = what;
    };

    auto worker = [&](int idx) {
        Qwen q(args.model, args.seq, devices[idx], "pp");
        bool ok = true;
        std::vector<double> samples;
        try {
            Check(aclrtSetDevice(devices[idx]), "set device");
            q.ConfigurePP(idx, 2, args.requests, comms[idx], args.mode);
            q.Build();
            for (int r = 0; r < args.requests; ++r) q.EnqueuePP(r);
            Check(aclrtSynchronizeDevice(), "pp first sync");
            if (idx == 1 && !pipeline) {
                std::lock_guard<std::mutex> lk(seq_mu);
                stage1_done = args.requests; // 首轮正确性运行独立于计时
            }
            if (idx == 1) q.VerifyPP();
        } catch (const std::exception &e) {
            ok = false;
            fail(std::string("stage") + std::to_string(idx) + ": " + e.what());
        }
        auto enqueue = [&](int iter) {
            const int base = (iter + 1) * args.requests;
            if (pipeline) {
                for (int r = 0; r < args.requests; ++r) q.EnqueuePP(r);
                Check(aclrtSynchronizeDevice(), "pp sync");
            } else if (idx == 0) {
                for (int r = 0; r < args.requests; ++r) {
                    {
                        std::unique_lock<std::mutex> lk(seq_mu);
                        seq_cv.wait(lk, [&] { return stage1_done >= base + r; });
                    }
                    q.EnqueuePP(r);
                    Check(aclrtSynchronizeDevice(), "pp sync");
                }
            } else {
                for (int r = 0; r < args.requests; ++r) {
                    q.EnqueuePP(r);
                    Check(aclrtSynchronizeDevice(), "pp sync");
                    {
                        std::lock_guard<std::mutex> lk(seq_mu);
                        stage1_done = base + r + 1;
                    }
                    seq_cv.notify_all();
                }
            }
        };
        for (int w = 0; w < args.warmup; ++w) {
            bar.Wait();
            if (ok) {
                try {
                    enqueue(w);
                } catch (const std::exception &e) {
                    ok = false;
                    fail(std::string("stage") + std::to_string(idx) + ": " + e.what());
                }
            }
            bar.Wait();
        }
        for (int i = 0; i < args.iters; ++i) {
            bar.Wait();
            const auto t0 = Clock::now();
            if (ok) {
                try {
                    enqueue(args.warmup + i);
                } catch (const std::exception &e) {
                    ok = false;
                    fail(std::string("stage") + std::to_string(idx) + ": " + e.what());
                }
            }
            samples.push_back(Us(t0, Clock::now()));
            bar.Wait();
        }
        if (ok) {
            try {
                q.CommitTiming(samples);
                if (idx == 1) q.VerifyPP();
                stage_json[idx] = q.ResultJson(
                    idx == 0 ? "pp_stage0" : "pp_stage1", args.warmup, args.iters);
            } catch (const std::exception &e) {
                fail(std::string("stage") + std::to_string(idx) + ": " + e.what());
            }
        }
        bar.Wait();
        if (comms[idx]) HcclCommDestroy(comms[idx]);
        aclrtResetDevice(devices[idx]);
    };

    std::thread t0(worker, 0);
    std::thread t1(worker, 1);
    for (int w = 0; w < args.warmup * 2; ++w) bar.Wait();
    std::vector<double> wall_samples;
    for (int i = 0; i < args.iters; ++i) {
        bar.Wait();
        const auto t0 = Clock::now();
        bar.Wait();
        wall_samples.push_back(Us(t0, Clock::now()));
    }
    bar.Wait();
    t0.join();
    t1.join();

    if (!err.empty()) throw std::runtime_error(err);
    std::vector<double> ws = wall_samples;
    std::sort(ws.begin(), ws.end());
    std::ostringstream os;
    os << "{\"mode\":\"" << (pipeline ? "pp_pipeline" : "pp_serial")
       << "\",\"seq\":" << args.seq << ",\"requests\":" << args.requests
       << ",\"devices\":[" << devices[0] << "," << devices[1] << "]"
       << ",\"wall_best_us\":" << ws.front()
       << ",\"wall_typical_us\":" << ws[ws.size() / 2]
       << ",\"wall_samples_us\":[";
    for (size_t i = 0; i < wall_samples.size(); ++i) {
        if (i) os << ",";
        os << wall_samples[i];
    }
    os << "],\"stages\":[" << stage_json[0] << "," << stage_json[1] << "]}";
    result = os.str();
    Check(aclFinalize(), "finalize");
    return 0;
}

int RunQwenCli(const Args &args, std::string &result)
{
    if (args.mode.rfind("pp", 0) == 0) {
        return RunQwenPP(args, args.mode == "pp" || args.mode == "pp1", result);
    }
    if (args.mode.rfind("tp", 0) == 0) {
        return RunQwenTP(args, args.mode == "tp2_split", result);
    }
    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(args.device), "set device");
    Qwen q(args.model, args.seq, args.device, args.mode);
    q.Build();
    q.RunOnce();
    Check(aclrtSynchronizeDevice(), "first sync");
    q.Verify();
    if (args.mode == "serial_graph" || args.mode == "graph_whole" ||
        args.mode == "relay_flag_graph") {
        q.CaptureAll();
    }
    q.Bench(args.warmup, args.iters);
    q.Verify(); // 图重放后用最后一次输出复核数值与 flag 状态
    result = q.ResultJson(args.mode, args.warmup, args.iters);
    Check(aclrtResetDevice(args.device), "reset");
    Check(aclFinalize(), "finalize");
    return 0;
}

} // namespace

#ifndef KR_LIBRARY
int main(int argc, char **argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        Args args = ParseArgs(argc, argv);
        std::string res;
        RunQwenCli(args, res);
        printf("%s\n", res.c_str());
        if (!args.out.empty()) {
            std::ofstream f(args.out);
            f << res << "\n";
        }
        return 0;
    } catch (const std::exception &e) {
        printf("QWEN_FAIL: %s\n", e.what());
        return 1;
    }
}
#else
// 稳定 C ABI：外部项目通过 libkernrelay.so 调用，不依赖 argv/子进程。
extern "C" int kr_qwen_profile_many(const char *model, const char *spec_json,
                                    int seq, int device, char *out,
                                    size_t out_cap)
{
    try {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device), "set device");
        Qwen q(model, seq, device, "serial");
        q.Build();
        std::string res;
        q.ProfileMany(spec_json, res);
        Check(aclrtResetDevice(device), "reset");
        Check(aclFinalize(), "finalize");
        if (!out || out_cap < res.size() + 1) {
            throw std::runtime_error("output buffer too small");
        }
        std::memcpy(out, res.c_str(), res.size() + 1);
        return 0;
    } catch (const std::exception &e) {
        if (out && out_cap) {
            const std::string msg = std::string("{\"error\":\"") + e.what() + "\"}";
            std::snprintf(out, out_cap, "%s", msg.c_str());
        }
        return 1;
    }
}

extern "C" int kr_qwen_tp_run(const char *model, const char *mode, int seq,
                              int dev0, int dev1, int tiles, int warmup,
                              int iters, char *out, size_t out_cap)
{
    try {
        Args args;
        args.model = model ? model : "";
        args.mode = mode ? mode : "tp2";
        args.seq = seq;
        args.device = dev0;
        args.device2 = dev1;
        args.tiles = tiles;
        args.warmup = warmup;
        args.iters = iters;
        if (args.model.empty()) throw std::runtime_error("model path required");
        std::string res;
        RunQwenTP(args, args.mode == "tp2_split", res);
        if (!out || out_cap < res.size() + 1) {
            throw std::runtime_error("output buffer too small");
        }
        std::memcpy(out, res.c_str(), res.size() + 1);
        return 0;
    } catch (const std::exception &e) {
        if (out && out_cap) {
            const std::string msg = std::string("{\"error\":\"") + e.what() + "\"}";
            std::snprintf(out, out_cap, "%s", msg.c_str());
        }
        return 1;
    }
}

extern "C" int kr_qwen_pp_run(const char *model, const char *mode, int seq,
                              int dev0, int dev1, int requests, int warmup,
                              int iters, char *out, size_t out_cap)
{
    try {
        Args args;
        args.model = model ? model : "";
        args.mode = mode ? mode : "pp";
        args.seq = seq;
        args.device = dev0;
        args.device2 = dev1;
        args.requests = requests;
        args.warmup = warmup;
        args.iters = iters;
        if (args.model.empty()) throw std::runtime_error("model path required");
        std::string res;
        RunQwenPP(args, args.mode == "pp" || args.mode == "pp1", res);
        if (!out || out_cap < res.size() + 1) {
            throw std::runtime_error("output buffer too small");
        }
        std::memcpy(out, res.c_str(), res.size() + 1);
        return 0;
    } catch (const std::exception &e) {
        if (out && out_cap) {
            const std::string msg = std::string("{\"error\":\"") + e.what() + "\"}";
            std::snprintf(out, out_cap, "%s", msg.c_str());
        }
        return 1;
    }
}

extern "C" int kr_qwen_run(const char *model, const char *mode, int seq,
                           int device, int warmup, int iters, char *out,
                           size_t out_cap)
{
    try {
        Args args;
        args.model = model ? model : "";
        args.mode = mode ? mode : "serial_graph";
        args.seq = seq;
        args.device = device;
        args.warmup = warmup;
        args.iters = iters;
        if (args.model.empty()) throw std::runtime_error("model path required");
        std::string res;
        RunQwenCli(args, res);
        if (!out || out_cap < res.size() + 1) {
            throw std::runtime_error("output buffer too small");
        }
        std::memcpy(out, res.c_str(), res.size() + 1);
        return 0;
    } catch (const std::exception &e) {
        if (out && out_cap) {
            const std::string msg = std::string("{\"error\":\"") + e.what() + "\"}";
            std::snprintf(out, out_cap, "%s", msg.c_str());
        }
        return 1;
    }
}
#endif
