// NeutronHP 真实执行器：读取调度产物 schedule.json（含内嵌 DAG），
// 将其物化为「每 stream 的任务序列」（算子 tile + tile_gate + publish_tile_flag），
// 在 nranks 张卡上按调度顺序真实入队并测量 makespan。
//
// 用法：nhp_exec <schedule.json> <iters> <warmup> <out.json>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_nhp_aic_delay_publish.h"
#include "aclrtlaunch_nhp_aiv_spin.h"
#include "aclrtlaunch_publish_tile_flag.h"
#include "hccl/hccl.h"
#include "hccl/hccl_comm.h"

#include "kernel/gate_args.h"
#include "runtime/hccl_session.hpp"
#include "runtime/json_min.hpp"
#include "runtime/op_impl.h"

extern "C" void tile_gate(uint32_t, void *, aclrtStream, void *, size_t);

using Clock = std::chrono::steady_clock;
using nhp::json::Value;

namespace {
constexpr uint32_t FLAG_STRIDE = 16; // 与 kernel/gate_args.h 协议一致（int32 单位）
constexpr uint64_t MAX_GATE_SPINS = 4000000000ULL;
constexpr size_t GATE_SCRATCH_BYTES = 2 * 1024 * 1024; // gate 自旋缓存行驱逐 scratch

void CheckAcl(aclError ret, const char *what)
{
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed, acl=" + std::to_string(static_cast<int>(ret)));
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

uint16_t ToHalf(float value)
{
    const __fp16 half = static_cast<__fp16>(value);
    uint16_t bits;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

// 确定性输入：同一 (seed, index) 生成同一数值，保证 A/B 两次运行输入一致。
float DetValue(uint32_t seed, size_t index)
{
    const uint32_t h = static_cast<uint32_t>(index) * 2654435761U + seed * 40503U + 7U;
    const int v = static_cast<int>((h >> 8) % 23U) - 11;
    return static_cast<float>(v) * 0.00390625f;
}

void FillHostHalf(std::vector<uint16_t> &dst, uint32_t seed)
{
    for (size_t i = 0; i < dst.size(); ++i) dst[i] = ToHalf(DetValue(seed, i));
}

// 五条物理执行流（与 kernel/engine_map.json 的 physical 分类一致）：
//   aicore = MIX 任务（保守视为占用全部 AICore）
//   aic / aiv = 纯 AIC / 纯 AIV 任务；aicpu = 纯 AICPU 任务；hccl = 通信
constexpr int kMaxStreams = 5;

int EngineIndex(const std::string &engine)
{
    if (engine == "aicore" || engine == "compute") return 0;
    if (engine == "aic") return 1;
    if (engine == "aiv") return 2;
    if (engine == "aicpu") return 3;
    if (engine == "hccl") return 4;
    throw std::runtime_error("unknown engine " + engine);
}

struct NodeSpec {
    std::string id;
    nhp::OpKind kind = nhp::OpKind::Add;
    std::string engine = "aiv";
    int64_t rows = 0, cols = 0, k = 0, n = 0, dim0 = 0;
    int tiles = 1;
    bool split = false;
    uint32_t iters = 1;     // 自定义算子重复计算次数
    uint32_t blocks = 1;    // 自定义算子 AIC/AIV 启动块数
    std::vector<std::string> preds;
};

struct TaskSpec {
    std::string node;
    int tile = 0;
    std::vector<std::pair<std::string, int>> waits;
    bool publish = false;
    int flagSlot = -1;
};

// 每块的显式行/列区间（scheduler 计算，缺省时回退旧 stride）
struct TaskGeom {
    int64_t inRowOff = 0;    // 输入(发送)行起点，producer 输出行空间
    int64_t inRows = 0;      // 本块输入(发送)行数
    int64_t outRowOff = 0;   // 本块输出行起点，本节点输出行空间
    int64_t nOff = 0;        // matmul 本块输出列起点（N 切）
    int64_t nTile = 0;       // matmul 本块列数；0 表示整幅 N
    int64_t outColOff = 0;   // 输出列起点（N 切/2D）
    int64_t outCols = 0;     // 输出列数；0 表示整幅
};

struct EngineSeq {
    std::string engine;
    std::vector<TaskSpec> tasks;
};

struct SchedDoc {
    std::string dag;
    std::string group;
    uint32_t nranks = 1;
    std::vector<NodeSpec> nodes;
    std::map<std::string, EngineSeq> streams;
    std::map<std::string, size_t> nodeIndex;
    std::map<std::pair<std::string, int>, TaskGeom> geom;

    const NodeSpec &Node(const std::string &id) const
    {
        auto it = nodeIndex.find(id);
        if (it == nodeIndex.end()) throw std::runtime_error("unknown node " + id);
        return nodes[it->second];
    }

    const TaskGeom *Geom(const std::string &id, int tile) const
    {
        auto it = geom.find(std::make_pair(id, tile));
        return it == geom.end() ? nullptr : &it->second;
    }
};

SchedDoc ParseSchedule(const Value &root)
{
    SchedDoc doc;
    doc.dag = root.Str("dag");
    doc.group = root.Str("group");
    doc.nranks = static_cast<uint32_t>(root.Num("nranks", 1));
    for (const auto &n : root.At("nodes").array) {
        NodeSpec spec;
        spec.id = n.Str("id");
        spec.kind = nhp::ParseOpKind(n.Str("op"));
        spec.engine = n.Str("engine", "aiv");
        spec.rows = static_cast<int64_t>(n.Num("rows"));
        spec.cols = static_cast<int64_t>(n.Num("cols"));
        spec.k = static_cast<int64_t>(n.Num("k"));
        spec.n = static_cast<int64_t>(n.Num("n"));
        spec.dim0 = static_cast<int64_t>(n.Num("dim0"));
        spec.tiles = static_cast<int>(n.Num("tiles", 1));
        spec.split = n.Has("split") ? n.At("split").boolean : false;
        spec.iters = static_cast<uint32_t>(n.Num("iters", 1));
        spec.blocks = static_cast<uint32_t>(n.Num("blocks", 1));
        if (n.Has("preds")) {
            for (const auto &p : n.At("preds").array) spec.preds.push_back(p.string);
        }
        doc.nodeIndex[spec.id] = doc.nodes.size();
        doc.nodes.push_back(spec);
    }
    const Value &streams = root.At("streams");
    for (const auto &kv : streams.object) {
        EngineSeq seq;
        seq.engine = kv.first;
        for (const auto &t : kv.second.array) {
            TaskSpec task;
            task.node = t.Str("node");
            task.tile = static_cast<int>(t.Num("tile", 0));
            task.publish = t.Has("publish") ? t.At("publish").boolean : false;
            if (t.Has("wait")) {
                for (const auto &w : t.At("wait").array) {
                    task.waits.emplace_back(w.At(0).string, static_cast<int>(w.At(1).number));
                }
            }
            if (t.Has("in_row_off") && t.Has("in_rows")) {
                TaskGeom g;
                g.inRowOff = static_cast<int64_t>(t.Num("in_row_off"));
                g.inRows = static_cast<int64_t>(t.Num("in_rows"));
                g.outRowOff = t.Has("out_row_off")
                                  ? static_cast<int64_t>(t.Num("out_row_off"))
                                  : g.inRowOff;
                g.nOff = t.Has("n_off") ? static_cast<int64_t>(t.Num("n_off")) : 0;
                g.nTile = t.Has("n_tiles") ? static_cast<int64_t>(t.Num("n_tiles")) : 0;
                g.outColOff = t.Has("out_col_off")
                                  ? static_cast<int64_t>(t.Num("out_col_off"))
                                  : g.nOff;
                g.outCols = t.Has("out_cols")
                                ? static_cast<int64_t>(t.Num("out_cols"))
                                : g.nTile;
                if (doc.geom.find(std::make_pair(task.node, task.tile)) != doc.geom.end()) {
                    throw std::runtime_error("duplicate task for node " + task.node +
                                             " tile " + std::to_string(task.tile));
                }
                doc.geom[std::make_pair(task.node, task.tile)] = g;
            }
            seq.tasks.push_back(task);
        }
        doc.streams[kv.first] = seq;
    }
    return doc;
}

// ---------------- 拓扑规范化：五条物理流 ----------------
//
// schedule 的 engine 字段是物理执行流名（scheduler/fixed.py 与 engine_map.json
// 共同决定）：aicore（MIX）、aic（纯 AIC）、aiv（纯 AIV）、aicpu、hccl。
//
// 约束（用户模型）：MIX 暂时视为占用全部 AICore，不允许与纯 AIC/AIV 重叠。
// 执行器的落地方式是：当 schedule 同时含 aicore 与 aic/aiv 时，把这三条流按
// 【全局拓扑序】投影合并进 aicore 一条流（串行化），aicpu/hccl 保持独立。
// 不涉及 MIX 时，纯 AIC 与纯 AIV 可以真正并发（异构重叠）。
//
// 合并序必须是全局拓扑序（节点 = 全部任务；硬边 = 同流 FIFO + 全部 wait 边），
// 否则"合并流 FIFO × 跨流 wait"会组合出死锁环（doc/design.md §8）。
// NHP_LAYOUT=merged 把 aicpu 也并入 aicore（旧"单一计算 lane"消融）；
// NHP_LAYOUT=split/4stream 保持五流不合并（跳过 MIX 互斥的静态合并，
// 仅用于排查；含 MIX 的 DAG 默认仍走互斥合并）。
void NormalizeTopology(SchedDoc &doc)
{
    std::map<std::string, EngineSeq> renamed;
    for (auto &kv : doc.streams) {
        const std::string name = kv.first == "compute" ? "aicore" : kv.first;
        (void)EngineIndex(name);
        if (renamed.count(name) != 0) {
            throw std::runtime_error("duplicate engine stream " + name);
        }
        renamed[name] = std::move(kv.second);
    }
    doc.streams = std::move(renamed);

    const char *layoutEnv = std::getenv("NHP_LAYOUT");
    const std::string layout = layoutEnv == nullptr ? "split" : std::string(layoutEnv);
    const bool forceMerged = layout == "merged";
    const bool hasAicore = doc.streams.count("aicore") != 0 && !doc.streams["aicore"].tasks.empty();
    const bool hasAic = doc.streams.count("aic") != 0 && !doc.streams["aic"].tasks.empty();
    const bool hasAiv = doc.streams.count("aiv") != 0 && !doc.streams["aiv"].tasks.empty();

    std::set<std::string> mergeSet;
    if (forceMerged) {
        mergeSet = {"aicore", "aic", "aiv", "aicpu"};
    } else if (hasAicore && (hasAic || hasAiv)) {
        mergeSet = {"aicore", "aic", "aiv"};   // MIX 与纯 AIC/AIV 互斥
    }
    if (mergeSet.empty()) return;

    std::vector<std::pair<std::string, size_t>> keys;
    std::map<std::pair<std::string, int>, std::pair<std::string, size_t>> pos;
    for (auto &kv : doc.streams) {
        for (size_t i = 0; i < kv.second.tasks.size(); ++i) {
            const auto &t = kv.second.tasks[i];
            keys.emplace_back(kv.first, i);
            pos[{t.node, t.tile}] = keys.back();
        }
    }
    if (keys.empty()) return;

    const auto rank = [](const std::pair<std::string, size_t> &k) {
        return std::make_pair(EngineIndex(k.first), k.second);
    };

    std::map<std::pair<std::string, size_t>, std::set<std::pair<std::string, size_t>>> deps;
    for (const auto &key : keys) {
        std::set<std::pair<std::string, size_t>> d;
        if (key.second > 0) d.insert({key.first, key.second - 1});
        for (const auto &w : doc.streams[key.first].tasks[key.second].waits) {
            auto it = pos.find(w);
            if (it != pos.end() && it->second != key) d.insert(it->second);
        }
        deps[key] = std::move(d);
    }

    std::vector<std::pair<std::string, size_t>> order;
    std::set<std::pair<std::string, size_t>> done;
    std::vector<std::pair<std::string, size_t>> ready;
    for (const auto &key : keys) {
        if (deps[key].empty()) ready.push_back(key);
    }
    const auto byRank = [&rank](const std::pair<std::string, size_t> &a,
                                const std::pair<std::string, size_t> &b) {
        return rank(a) < rank(b);
    };
    std::sort(ready.begin(), ready.end(), byRank);
    while (!ready.empty()) {
        const auto cur = ready.front();
        ready.erase(ready.begin());
        order.push_back(cur);
        done.insert(cur);
        for (auto &kv : deps) {
            if (done.count(kv.first)) continue;
            auto it = kv.second.find(cur);
            if (it == kv.second.end()) continue;
            kv.second.erase(it);
            if (kv.second.empty()) {
                ready.push_back(kv.first);
                std::sort(ready.begin(), ready.end(), byRank);
            }
        }
    }
    if (order.size() != keys.size()) {
        throw std::runtime_error("lane normalize failed: wait cycle in schedule");
    }

    EngineSeq mergedLane;
    mergedLane.engine = "aicore";
    for (const auto &key : order) {
        if (mergeSet.count(key.first) == 0) continue;
        mergedLane.tasks.push_back(doc.streams[key.first].tasks[key.second]);
    }
    std::map<std::string, EngineSeq> out;
    if (!mergedLane.tasks.empty()) out["aicore"] = std::move(mergedLane);
    for (auto &kv : doc.streams) {
        if (mergeSet.count(kv.first) != 0 || kv.second.tasks.empty()) continue;
        out[kv.first] = std::move(kv.second);
    }
    doc.streams = std::move(out);
}

// ---------------- 每 rank 运行时上下文 ----------------

struct NodeBuf {
    void *out = nullptr;
    void *out2 = nullptr;
    void *aux1 = nullptr;   // matmul B / layernorm weight / embedding weight
    void *aux2 = nullptr;   // 第二输入 / embedding indices / layernorm bias
    void *aux3 = nullptr;   // layernorm mean
    void *ownedIn = nullptr;
    void *auxRecv = nullptr;
    size_t outBytes = 0;
    int outElem = 2;        // 主输出元素字节
};

struct SegmentGraph {
    int engine = 0;
    std::string engineName;
    int start = 0;
    int end = 0;
    std::vector<std::pair<std::string, int>> waits;
    std::vector<std::pair<std::string, int>> publishes;
    aclmdlRI graph = nullptr;
    bool launched = false;
};

struct RankCtx {
    uint32_t rank = 0;
    uint32_t nranks = 1;
    int device = 0;
    aclrtStream streams[kMaxStreams] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    HcclComm comm = nullptr;
    std::map<std::string, NodeBuf> bufs;
    std::map<std::pair<std::string, int>, std::unique_ptr<nhp::OpTask>> plans;
    // 同 rank 跨 stream 依赖事件：每个被等待的 (node,tile) 一个 ACL 事件
    std::map<std::pair<std::string, int>, aclrtEvent> events;
    aclmdlRI graph = nullptr;   // NHP_GRAPH=1：单流 schedule 的 ACL Graph
    int graphStream = -1;
    aclmdlRI graphs[kMaxStreams] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    std::vector<SegmentGraph> segments;  // NHP_GRAPH_SEGMENT=1
};

int64_t RowsPerTile(const NodeSpec &node)
{
    return node.split ? node.rows / node.tiles : node.rows;
}

int TileCount(const NodeSpec &node)
{
    return node.split ? node.tiles : 1;
}

// 节点输出行数（与 scheduler/chunks.out_rows 一致）：声明的 rows 是输入(send)行数
int64_t OutRowsOf(const NodeSpec &node, uint32_t nranks)
{
    switch (node.kind) {
        case nhp::OpKind::AllGather:
            return static_cast<int64_t>(nranks) * node.rows;
        case nhp::OpKind::ReduceScatter:
            return node.rows / static_cast<int64_t>(nranks);
        default:
            return node.rows;
    }
}


size_t OutElemBytes(const NodeSpec &node, uint32_t nranks)
{
    switch (node.kind) {
        case nhp::OpKind::Cast: return 4;
        case nhp::OpKind::TopK: return 2;
        case nhp::OpKind::Sort: return 2;
        case nhp::OpKind::ReduceScatter: return 2;
        case nhp::OpKind::AllGather: return 2;
        default: (void)nranks; return 2;
    }
}

size_t OutBytes(const NodeSpec &node, uint32_t nranks)
{
    const size_t rows = static_cast<size_t>(node.rows);
    const size_t cols = static_cast<size_t>(node.cols);
    switch (node.kind) {
        case nhp::OpKind::Cast: return rows * cols * 4;
        case nhp::OpKind::TopK: return rows * static_cast<size_t>(node.k) * 2 + rows * static_cast<size_t>(node.k) * 8;
        case nhp::OpKind::Sort: return rows * cols * 2 + rows * cols * 8;
        case nhp::OpKind::ReduceScatter: return rows * cols * 2;
        case nhp::OpKind::AllGather: return static_cast<size_t>(nranks) * rows * cols * 2;
        default: return rows * cols * 2;
    }
}

void AllocateNode(RankCtx &ctx, const NodeSpec &node, uint32_t nranks)
{
    NodeBuf buf;
    buf.outBytes = OutBytes(node, nranks);
    buf.outElem = static_cast<int>(OutElemBytes(node, nranks));
    buf.out = MallocDevice(buf.outBytes);
    // 自定义标量 kernel（custom_aic_mac/custom_aicpu_scale）只写 out[0]，
    // 其余输出若保留 aclrtMalloc 垃圾值，checksum 会读到 inf/NaN（JSON 非法）
    // 且 A/B 不可比；输出缓冲统一清零保证 checksum 确定性。
    CheckAcl(aclrtMemset(buf.out, buf.outBytes, 0, buf.outBytes), "memset out");
    const size_t rows = static_cast<size_t>(node.rows);
    const size_t cols = static_cast<size_t>(node.cols);

    switch (node.kind) {
        case nhp::OpKind::MatMul:
            buf.aux1 = MallocDevice(static_cast<size_t>(node.k) * static_cast<size_t>(node.n) * 2); // B
            break;
        case nhp::OpKind::Add:
            if (node.preds.size() < 2) buf.aux2 = MallocDevice(rows * cols * 2);
            break;
        case nhp::OpKind::LayerNorm:
            buf.aux1 = MallocDevice(cols * 2);  // weight
            buf.aux2 = MallocDevice(cols * 2);  // bias
            buf.aux3 = MallocDevice(rows * 4);  // mean
            buf.out2 = MallocDevice(rows * 4);  // rstd
            break;
        case nhp::OpKind::TopK:
            buf.out2 = MallocDevice(rows * static_cast<size_t>(node.k) * 8); // indices
            break;
        case nhp::OpKind::Sort:
            buf.out2 = MallocDevice(rows * cols * 8);
            break;
        case nhp::OpKind::Embedding:
            buf.aux1 = MallocDevice(static_cast<size_t>(node.dim0) * cols * 2); // weight
            buf.aux2 = MallocDevice(rows * 8);                                  // indices(int64)
            break;
        case nhp::OpKind::AllGather:
            buf.auxRecv = MallocDevice(static_cast<size_t>(nranks) * rows * cols * 2);
            break;
        default:
            break;
    }

    // 无前驱节点（源/通信源）：节点自备输入
    bool needOwnedIn = node.preds.empty();
    if (needOwnedIn) {
        size_t inCols = cols;
        if (node.kind == nhp::OpKind::MatMul) inCols = static_cast<size_t>(node.k);
        if (node.kind == nhp::OpKind::Embedding) {
            // indices 由节点自备，见 PreparePlan
        } else {
            buf.ownedIn = MallocDevice(rows * inCols * 2);
        }
    }
    ctx.bufs[node.id] = buf;
}

uint32_t NodeSeed(const std::string &id, uint32_t rank)
{
    uint32_t h = 2166136261U;
    for (char c : id) h = (h ^ static_cast<uint8_t>(c)) * 16777619U;
    return h + rank * 7919U;
}

void UploadInputs(RankCtx &ctx, const SchedDoc &doc)
{
    for (const auto &node : doc.nodes) {
        NodeBuf &buf = ctx.bufs[node.id];
        const uint32_t seed = NodeSeed(node.id, ctx.rank);
        const size_t rows = static_cast<size_t>(node.rows);
        const size_t cols = static_cast<size_t>(node.cols);
        auto uploadHalf = [&](void *dst, size_t elems, uint32_t s) {
            std::vector<uint16_t> host(elems);
            FillHostHalf(host, s);
            CheckAcl(aclrtMemcpy(dst, elems * 2, host.data(), elems * 2, ACL_MEMCPY_HOST_TO_DEVICE),
                     "upload input");
        };
        if (buf.ownedIn != nullptr) {
            size_t inCols = cols;
            if (node.kind == nhp::OpKind::MatMul) inCols = static_cast<size_t>(node.k);
            uploadHalf(buf.ownedIn, rows * inCols, seed + 1);
        }
        if (buf.aux1 != nullptr) {
            if (node.kind == nhp::OpKind::MatMul) {
                uploadHalf(buf.aux1, static_cast<size_t>(node.k) * static_cast<size_t>(node.n), seed + 2);
            } else if (node.kind == nhp::OpKind::Embedding) {
                uploadHalf(buf.aux1, static_cast<size_t>(node.dim0) * cols, seed + 2);
            } else {
                uploadHalf(buf.aux1, cols, seed + 2);
            }
        }
        if (buf.aux2 != nullptr) {
            if (node.kind == nhp::OpKind::LayerNorm) {
                uploadHalf(buf.aux2, cols, seed + 3);
            } else if (node.kind == nhp::OpKind::Embedding) {
                std::vector<int64_t> idx(rows);
                for (size_t i = 0; i < rows; ++i) {
                    idx[i] = static_cast<int64_t>((i * 131 + 7 + seed) % static_cast<size_t>(node.dim0));
                }
                CheckAcl(aclrtMemcpy(buf.aux2, rows * 8, idx.data(), rows * 8, ACL_MEMCPY_HOST_TO_DEVICE),
                         "upload embedding indices");
            } else if (node.kind == nhp::OpKind::Add) {
                uploadHalf(buf.aux2, rows * cols, seed + 3);
            }
        }
    }
}

void PreparePlans(RankCtx &ctx, const SchedDoc &doc)
{
    for (const auto &node : doc.nodes) {
        NodeBuf &buf = ctx.bufs[node.id];
        const int64_t rt = RowsPerTile(node);
        const int tiles = TileCount(node);
        const size_t cols = static_cast<size_t>(node.cols);
        void *inBase = nullptr;
        size_t inCols = cols;
        if (node.kind == nhp::OpKind::MatMul) inCols = static_cast<size_t>(node.k);
        if (!node.preds.empty()) {
            inBase = ctx.bufs[node.preds[0]].out;
        } else if (node.kind != nhp::OpKind::Embedding) {
            inBase = buf.ownedIn;
        }

        for (int t = 0; t < tiles; ++t) {
            auto task = std::unique_ptr<nhp::OpTask>(new nhp::OpTask());
            // workspace 每 plan 自持全新分配（同 nhp_bench 行为）：实测跨 RunOnce
            // 复用同一 workspace buffer（wsPool 池化）会让新 executor 使用旧 buffer，
            // 与 nhp_exec 高概率 507015/507035/堆损坏强相关（nhp_bench 无池化机制
            // 12/12 稳定、exec 带池化 ~50% 崩，去池化后恢复稳定）。短进程泄漏可接受。
            //
            // 分块几何：优先取 schedule.json 的显式行区间（scheduler/export.py）。
            // 旧版一律 rowOff = t * RowsPerTile(consumer) 去索引 producer 缓冲：
            // 1:1 且两侧 tiles 相同时碰巧正确，m:n / 粒度不同 / reducescatter /
            // allgather 时越界或错位（507015/507035 的一类触发面）。
            const TaskGeom *g = doc.Geom(node.id, t);
            const int64_t inRowOff = g ? g->inRowOff : static_cast<int64_t>(t) * rt;
            const int64_t inRows = g ? g->inRows : rt;
            const int64_t outRowOff = g ? g->outRowOff : static_cast<int64_t>(t) * rt;
            const int64_t inLimit = node.preds.empty()
                                        ? node.rows
                                        : OutRowsOf(doc.Node(node.preds[0]), ctx.nranks);
            if (inRowOff < 0 || inRows <= 0 || inRowOff + inRows > inLimit) {
                throw std::runtime_error("task " + node.id + "#" + std::to_string(t) +
                                         " input rows [" + std::to_string(inRowOff) + "," +
                                         std::to_string(inRowOff + inRows) + ") exceed " +
                                         std::to_string(inLimit));
            }
            if (outRowOff < 0 || outRowOff > OutRowsOf(node, ctx.nranks)) {
                throw std::runtime_error("task " + node.id + "#" + std::to_string(t) +
                                         " output row offset " + std::to_string(outRowOff) +
                                         " exceed " + std::to_string(OutRowsOf(node, ctx.nranks)));
            }
            nhp::OpShape shape;
            shape.rows = inRows;
            shape.cols = static_cast<int64_t>(cols);
            shape.k = node.k;
            shape.n = node.n;
            shape.dim0 = node.dim0;
            if (node.kind == nhp::OpKind::MatMul && g && g->nTile > 0) {
                if (g->nOff < 0 || g->nTile <= 0 || g->nOff + g->nTile > node.n) {
                    throw std::runtime_error("task " + node.id + "#" + std::to_string(t) +
                                             " N tile out of range");
                }
                shape.n = g->nTile;
                shape.ldb = node.n;
                shape.ldc = node.n;
                shape.b_col_off = g->nOff;
                shape.c_col_off = g->outColOff;
            }
            auto *inTile = static_cast<uint8_t *>(inBase) + inRowOff * inCols * 2;
            auto *outTile = static_cast<uint8_t *>(buf.out) + outRowOff * cols * 2;

            if (node.kind == nhp::OpKind::CustomAivScale ||
                node.kind == nhp::OpKind::CustomAicMac ||
                node.kind == nhp::OpKind::CustomMixScale ||
                node.kind == nhp::OpKind::CustomAicpuScale) {
                task->PrepareCustom(node.kind, shape, inTile, outTile, node.iters, node.blocks);
                ctx.plans[{node.id, t}] = std::move(task);
                continue;
            }

            if (nhp::IsCommKind(node.kind)) {
                void *send = inTile;
                void *recv = nullptr;
                void *auxRecv = nullptr;
                if (node.kind == nhp::OpKind::AllReduce) {
                    recv = outTile;
                } else if (node.kind == nhp::OpKind::ReduceScatter) {
                    recv = static_cast<uint8_t *>(buf.out) +
                           (static_cast<size_t>(ctx.rank) * static_cast<size_t>(node.rows) / ctx.nranks +
                            static_cast<size_t>(outRowOff)) * cols * 2;
                } else {
                    recv = buf.out;   // allgather 布局归一后的最终缓冲
                    auxRecv = static_cast<uint8_t *>(buf.auxRecv) +
                              static_cast<size_t>(inRowOff) * ctx.nranks * cols * 2;
                }
                task->PrepareComm(node.kind, shape, send, recv, auxRecv, ctx.comm, ctx.rank, ctx.nranks);
                if (node.kind == nhp::OpKind::AllGather) {
                    // 归一目标：块 c 的输出占据 [c*nranks*R_c, +nranks*R_c)（块连续布局，
                    // 与消费者按连续行区间读取一致；chunks=1 时与原 [rank][rows] 布局等价）
                    task->SetAllGatherNorm(node.rows, inRows, outRowOff);
                }
                ctx.plans[{node.id, t}] = std::move(task);
                continue;
            }

            switch (node.kind) {
                case nhp::OpKind::MatMul:
                    task->PrepareCompute(node.kind, shape, inTile, buf.aux1, outTile,
                                         nullptr, nullptr, nullptr);
                    break;
                case nhp::OpKind::Add: {
                    void *in2 = nullptr;
                    if (node.preds.size() > 1) {
                        in2 = static_cast<uint8_t *>(ctx.bufs[node.preds[1]].out) + inRowOff * cols * 2;
                    } else {
                        in2 = static_cast<uint8_t *>(buf.aux2) + inRowOff * cols * 2;
                    }
                    task->PrepareCompute(node.kind, shape, inTile, in2, outTile,
                                         nullptr, nullptr, nullptr);
                    break;
                }
                case nhp::OpKind::Gelu:
                case nhp::OpKind::Softmax:
                    task->PrepareCompute(node.kind, shape, inTile, nullptr, outTile,
                                         nullptr, nullptr, nullptr);
                    break;
                case nhp::OpKind::Cast:
                    task->PrepareCompute(node.kind, shape, inTile, nullptr,
                                         static_cast<uint8_t *>(buf.out) + outRowOff * cols * 4,
                                         nullptr, nullptr, nullptr);
                    break;
                case nhp::OpKind::LayerNorm:
                    task->PrepareCompute(node.kind, shape, inTile,
                                         static_cast<uint8_t *>(buf.out2) + outRowOff * 4,
                                         outTile,
                                         static_cast<uint8_t *>(buf.aux3) + outRowOff * 4,
                                         buf.aux1, buf.aux2);
                    break;
                case nhp::OpKind::TopK:
                    task->PrepareCompute(node.kind, shape, inTile, nullptr,
                                         static_cast<uint8_t *>(buf.out) +
                                             outRowOff * static_cast<size_t>(node.k) * 2,
                                         static_cast<uint8_t *>(buf.out2) +
                                             outRowOff * static_cast<size_t>(node.k) * 8,
                                         nullptr, nullptr);
                    break;
                case nhp::OpKind::Sort:
                    task->PrepareCompute(node.kind, shape, inTile, nullptr, outTile,
                                         static_cast<uint8_t *>(buf.out2) + outRowOff * cols * 8,
                                         nullptr, nullptr);
                    break;
                case nhp::OpKind::Embedding:
                    task->PrepareCompute(node.kind, shape, buf.aux1,
                                         static_cast<uint8_t *>(buf.aux2) + inRowOff * 8,
                                         outTile, nullptr, nullptr, nullptr);
                    break;
                default:
                    break;
            }
            ctx.plans[{node.id, t}] = std::move(task);
        }
    }
}

// ---------------- 调度物化：stream 任务序列 -> 真实入队 ----------------
// 同 rank 跨 stream 依赖用 ACL 事件表达（aclrtRecordEvent / aclrtStreamWaitEvent），
// 由 stream 调度器处理依赖：零 AICPU 任务、零自旋、零派发开销。
// （tile_gate 自旋机制实测孤立 gate ~1.5ms/次 —— AICPU 任务派发延迟，
//   事件机制实测 ~us 级；wait/publish 物化语义不变。）

void BuildEvents(RankCtx &ctx, const SchedDoc &doc)
{
    for (const auto &kv : doc.streams) {
        for (const auto &task : kv.second.tasks) {
            if (!task.publish) continue;
            const auto key = std::make_pair(task.node, task.tile);
            if (ctx.events.find(key) != ctx.events.end()) continue;
            aclrtEvent ev = nullptr;
            if (std::getenv("NHP_GRAPH_PER_STREAM") != nullptr) {
                CheckAcl(aclrtCreateEventWithFlag(&ev, ACL_EVENT_CAPTURE_STREAM_PROGRESS),
                         "create capture event");
            } else {
                CheckAcl(aclrtCreateEvent(&ev), "create event");
            }
            ctx.events[key] = ev;
        }
    }
}

// NHP_SERIALIZE=1：稳定性兜底——每个任务入队后立即同步排干，完全关闭
// compute×comm 重叠，规避 CANN 9.0.0 collective×多流并发缺陷（507015/
// 507035，见 doc/design.md §7）。代价是重叠收益归零（B 组加速比→~1x），
// 仅建议在"跑完比测准更重要"的场景使用；默认关闭。
bool SerializeMode()
{
    return std::getenv("NHP_SERIALIZE") != nullptr;
}

bool GraphMode()
{
    return std::getenv("NHP_GRAPH") != nullptr;
}

bool GraphPerStreamMode()
{
    return std::getenv("NHP_GRAPH_PER_STREAM") != nullptr;
}

double SyncMedian(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// NHP_SYNC_BENCH=<out.json>：跨流同步微基准。三种模式的 kernel 启动完全相同
// （AIC 空发布 + AIV 空自旋），只改变同步方式；event/flag 减去 no-sync 即为
// 同步增量，launch 开销已包含在三种模式内。
double RunSyncMode(const std::string &mode, int n, int rounds, int warmup,
                   aclrtStream prod, aclrtStream cons, void *flags, void *status,
                   std::vector<aclrtEvent> &events, uint32_t &timeouts)
{
    const size_t bytes = static_cast<size_t>(n) * FLAG_STRIDE * 4;
    std::vector<double> samples;
    for (int r = 0; r < warmup + rounds; ++r) {
        CheckAcl(aclrtMemset(flags, bytes, 0, bytes), "sync clear flags");
        CheckAcl(aclrtMemset(status, bytes, 0, bytes), "sync clear status");
        CheckAcl(aclrtSynchronizeDevice(), "sync reset");
        auto t0 = Clock::now();
        for (int i = 0; i < n; ++i) {
            CheckAcl(static_cast<aclError>(aclrtlaunch_nhp_aic_delay_publish(
                         1, prod, flags, static_cast<uint32_t>(i), 0)),
                     "sync publish");
            if (mode == "event") {
                CheckAcl(aclrtRecordEvent(events[i], prod), "sync record");
                CheckAcl(aclrtStreamWaitEvent(cons, events[i]), "sync wait");
            }
            const uint32_t noWait = mode == "flag" ? 0U : 1U;
            CheckAcl(static_cast<aclError>(aclrtlaunch_nhp_aiv_spin(
                         1, cons, flags, static_cast<uint32_t>(i), status, noWait)),
                     "sync spin");
        }
        CheckAcl(aclrtSynchronizeStream(prod), "sync prod");
        CheckAcl(aclrtSynchronizeStream(cons), "sync cons");
        const auto t1 = Clock::now();
        if (r >= warmup) {
            samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / n);
        }
    }
    std::vector<uint32_t> host(static_cast<size_t>(n) * FLAG_STRIDE);
    CheckAcl(aclrtMemcpy(host.data(), host.size() * 4, status, host.size() * 4,
                         ACL_MEMCPY_DEVICE_TO_HOST), "sync status copy");
    for (int i = 0; i < n; ++i) {
        if (host[static_cast<size_t>(i) * FLAG_STRIDE] != 1U) ++timeouts;
    }
    return SyncMedian(samples);
}

void RunSyncBench(const std::string &outPath)
{
    const int n = 256;
    const int rounds = 10;
    CheckAcl(aclrtSetDevice(0), "sync set device");
    aclrtStream prod = nullptr, cons = nullptr;
    CheckAcl(aclrtCreateStream(&prod), "sync create prod");
    CheckAcl(aclrtCreateStream(&cons), "sync create cons");
    const size_t bytes = static_cast<size_t>(n) * FLAG_STRIDE * 4;
    void *flags = MallocDevice(bytes);
    void *status = MallocDevice(bytes);
    std::vector<aclrtEvent> events(n, nullptr);
    for (auto &ev : events) CheckAcl(aclrtCreateEvent(&ev), "sync create event");
    uint32_t toNo = 0, toEv = 0, toFl = 0;
    const double nosync = RunSyncMode("nosync", n, rounds, 2, prod, cons,
                                      flags, status, events, toNo);
    const double event = RunSyncMode("event", n, rounds, 2, prod, cons,
                                     flags, status, events, toEv);
    const double flag = RunSyncMode("flag", n, rounds, 2, prod, cons,
                                    flags, status, events, toFl);
    std::ofstream out(outPath);
    out << "{\n  \"n\": " << n << ",\n  \"rounds\": " << rounds
        << ",\n  \"us_per_pair_nosync\": " << nosync
        << ",\n  \"us_per_pair_event\": " << event
        << ",\n  \"us_per_pair_flag\": " << flag
        << ",\n  \"event_minus_nosync_us\": " << (event - nosync)
        << ",\n  \"flag_minus_nosync_us\": " << (flag - nosync)
        << ",\n  \"timeouts_nosync\": " << toNo
        << ",\n  \"timeouts_event\": " << toEv
        << ",\n  \"timeouts_flag\": " << toFl << "\n}\n";
    std::printf("sync bench: nosync %.3f us/pair (to %u), event %.3f (to %u), "
                "flag %.3f (to %u), event-base %.3f, flag-base %.3f\n",
                nosync, toNo, event, toEv, flag, toFl,
                event - nosync, flag - nosync);
    for (auto ev : events) CheckAcl(aclrtDestroyEvent(ev), "sync destroy event");
    CheckAcl(aclrtFree(flags), "sync free flags");
    CheckAcl(aclrtFree(status), "sync free status");
    CheckAcl(aclrtDestroyStream(prod), "sync destroy prod");
    CheckAcl(aclrtDestroyStream(cons), "sync destroy cons");
}

// 按跨 stream 拓扑序入队：保证每个事件的 record 先于所有 wait 入队
// （aclrtStreamWaitEvent 对未 record 事件的语义未定义，规避之）。
void EnqueueAll(RankCtx &ctx, const SchedDoc &doc)
{
    std::vector<std::pair<aclrtStream, const EngineSeq *>> seqs;
    for (int e = 0; e < kMaxStreams; ++e) {
        for (const auto &kv : doc.streams) {
            if (EngineIndex(kv.first) == e) {
                seqs.emplace_back(ctx.streams[e], &kv.second);
            }
        }
    }
    std::vector<size_t> idx(seqs.size(), 0);
    std::set<std::pair<std::string, int>> recorded;
    size_t remaining = 0;
    for (const auto &s : seqs) remaining += s.second->tasks.size();
    while (remaining > 0) {
        bool progress = false;
        for (size_t s = 0; s < seqs.size(); ++s) {
            if (idx[s] >= seqs[s].second->tasks.size()) continue;
            const auto &task = seqs[s].second->tasks[idx[s]];
            bool ready = true;
            for (const auto &w : task.waits) {
                if (recorded.find(w) == recorded.end()) { ready = false; break; }
            }
            if (!ready) continue;
            aclrtStream stream = seqs[s].first;
            for (const auto &w : task.waits) {
                auto it = ctx.events.find(w);
                if (it == ctx.events.end()) throw std::runtime_error("wait on unpublished flag");
                CheckAcl(aclrtStreamWaitEvent(stream, it->second), "stream wait event");
            }
            auto plan = ctx.plans.find({task.node, task.tile});
            if (plan == ctx.plans.end()) throw std::runtime_error("missing plan for task");
            plan->second->Enqueue(stream);
            if (SerializeMode()) {
                CheckAcl(aclrtSynchronizeStream(stream), "serialize sync");
            }
            if (task.publish) {
                const auto key = std::make_pair(task.node, task.tile);
                auto it = ctx.events.find(key);
                if (it == ctx.events.end()) throw std::runtime_error("missing event");
                CheckAcl(aclrtRecordEvent(it->second, stream), "record event");
                recorded.insert(key);
            }
            idx[s]++;
            progress = true;
            remaining--;
        }
        if (!progress) throw std::runtime_error("schedule contains wait cycle");
    }
}

double RunOnce(std::vector<RankCtx> &ctxs, const SchedDoc &doc, uint32_t &gateTimeouts)
{
    // 每次运行前重建全部 plan（aclnn executor 一次性语义：复用 Enqueue 实测
    // core dump；准备耗时不计入 makespan）
    for (auto &ctx : ctxs) {
        CheckAcl(aclrtSetDevice(ctx.device), "prep set device");
        ctx.plans.clear();
        for (auto &kv : ctx.events) (void)aclrtDestroyEvent(kv.second);
        ctx.events.clear();
        BuildEvents(ctx, doc);
        PreparePlans(ctx, doc);
        CheckAcl(aclrtSynchronizeDevice(), "sync prep");
    }

    std::vector<double> rankEnd(ctxs.size(), 0.0);
    std::vector<std::string> errors(ctxs.size());
    const auto begin = Clock::now();
    std::vector<std::thread> workers;
    workers.reserve(ctxs.size());
    for (uint32_t r = 0; r < ctxs.size(); ++r) {
        workers.emplace_back([&ctxs, &doc, &rankEnd, &errors, r, begin]() {
            try {
                RankCtx &ctx = ctxs[r];
                CheckAcl(aclrtSetDevice(ctx.device), "worker set device");
                EnqueueAll(ctx, doc);
                for (int e = 0; e < kMaxStreams; ++e) {
                    const aclError rc = aclrtSynchronizeStreamWithTimeout(ctx.streams[e], 120000);
                    if (rc != ACL_SUCCESS) {
                        throw std::runtime_error("sync engine stream " + std::to_string(e) +
                                                 " failed, acl=" + std::to_string(static_cast<int>(rc)));
                    }
                }
                rankEnd[r] = std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
            } catch (const std::exception &ex) {
                errors[r] = ex.what();
            }
        });
    }
    for (auto &w : workers) w.join();
    for (const auto &err : errors) {
        if (!err.empty()) throw std::runtime_error("rank worker failed: " + err);
    }
    // NHP_FREE_WS=1：每轮排干后回收 workspace（Release 契约：仅同步后调用）。
    // 判别实验：c08 类多轮失败是否源于"每轮泄漏 workspace × HCCL"累积
    // （doc/design.md §8.6）。
    if (std::getenv("NHP_FREE_WS") != nullptr) {
        for (auto &ctx : ctxs) {
            for (auto &kv : ctx.plans) kv.second->Release();
        }
    }

    double makespan = 0.0;
    for (double v : rankEnd) makespan = std::max(makespan, v);
    gateTimeouts = 0; // 事件等待无超时语义（stream 调度器保证依赖）
    return makespan;
}

// NHP_GRAPH=1：单流 schedule 用 ACL Graph 录制一次，后续进行图重放。
// 用于回答“kernel 数固定时如何降低 launch 开销”：图重放把 k 次 host launch
// 折叠成一次 aclmdlRIExecuteAsync。多流/带跨流 event 的 schedule 暂不支持。
void CaptureGraphs(std::vector<RankCtx> &ctxs, const SchedDoc &doc)
{
    for (auto &ctx : ctxs) {
        CheckAcl(aclrtSetDevice(ctx.device), "graph set device");
        ctx.plans.clear();
        for (auto &kv : ctx.events) (void)aclrtDestroyEvent(kv.second);
        ctx.events.clear();
        BuildEvents(ctx, doc);
        PreparePlans(ctx, doc);
        int active = -1;
        for (int e = 0; e < kMaxStreams; ++e) {
            bool nonempty = false;
            for (const auto &kv : doc.streams) {
                if (EngineIndex(kv.first) == e && !kv.second.tasks.empty()) nonempty = true;
            }
            if (nonempty) {
                if (active >= 0) {
                    throw std::runtime_error("NHP_GRAPH 仅支持单流 schedule");
                }
                active = e;
            }
        }
        if (active < 0) throw std::runtime_error("NHP_GRAPH: 没有任务流");
        ctx.graphStream = active;
        CheckAcl(aclrtSynchronizeDevice(), "graph sync prep");
        CheckAcl(aclmdlRICaptureBegin(ctx.streams[active],
                                      ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),
                 "graph capture begin");
        EnqueueAll(ctx, doc);
        CheckAcl(aclmdlRICaptureEnd(ctx.streams[active], &ctx.graph),
                 "graph capture end");
        CheckAcl(aclrtSynchronizeStream(ctx.streams[active]), "graph sync after capture");
    }
}

double RunGraphOnce(std::vector<RankCtx> &ctxs)
{
    std::vector<double> rankEnd(ctxs.size(), 0.0);
    std::vector<std::string> errors(ctxs.size());
    const auto begin = Clock::now();
    std::vector<std::thread> workers;
    workers.reserve(ctxs.size());
    for (uint32_t r = 0; r < ctxs.size(); ++r) {
        workers.emplace_back([&ctxs, &rankEnd, &errors, r, begin]() {
            try {
                RankCtx &ctx = ctxs[r];
                CheckAcl(aclrtSetDevice(ctx.device), "graph worker set device");
                if (ctx.graph == nullptr || ctx.graphStream < 0) {
                    throw std::runtime_error("graph not captured");
                }
                CheckAcl(aclmdlRIExecuteAsync(ctx.graph, ctx.streams[ctx.graphStream]),
                         "graph execute");
                CheckAcl(aclrtSynchronizeStream(ctx.streams[ctx.graphStream]),
                         "graph sync");
                rankEnd[r] = std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
            } catch (const std::exception &ex) {
                errors[r] = ex.what();
            }
        });
    }
    for (auto &w : workers) w.join();
    for (const auto &err : errors) {
        if (!err.empty()) throw std::runtime_error("graph rank worker failed: " + err);
    }
    double makespan = 0.0;
    for (double v : rankEnd) makespan = std::max(makespan, v);
    return makespan;
}

void EnqueueStream(RankCtx &ctx, const SchedDoc &doc, int e)
{
    aclrtStream stream = ctx.streams[e];
    for (const auto &kv : doc.streams) {
        if (EngineIndex(kv.first) != e) continue;
        for (const auto &task : kv.second.tasks) {
            for (const auto &w : task.waits) {
                auto it = ctx.events.find(w);
                if (it == ctx.events.end()) throw std::runtime_error("wait on unpublished flag");
                CheckAcl(aclrtStreamWaitEvent(stream, it->second), "stream wait event");
            }
            auto plan = ctx.plans.find({task.node, task.tile});
            if (plan == ctx.plans.end()) throw std::runtime_error("missing plan for task");
            plan->second->Enqueue(stream);
            if (task.publish) {
                const auto key = std::make_pair(task.node, task.tile);
                auto it = ctx.events.find(key);
                if (it == ctx.events.end()) throw std::runtime_error("missing event");
                CheckAcl(aclrtRecordEvent(it->second, stream), "record event");
            }
        }
    }
}

std::vector<int> StreamTopoOrder(const SchedDoc &doc)
{
    std::map<std::string, int> nodeEngine;
    for (const auto &n : doc.nodes) nodeEngine[n.id] = EngineIndex(n.engine);
    std::map<int, std::set<int>> succ;
    std::map<int, int> indeg;
    for (int e = 0; e < kMaxStreams; ++e) indeg[e] = 0;
    for (const auto &kv : doc.streams) {
        const int dst = EngineIndex(kv.first);
        if (kv.second.tasks.empty()) continue;
        if (indeg.find(dst) == indeg.end()) indeg[dst] = 0;
        for (const auto &task : kv.second.tasks) {
            for (const auto &w : task.waits) {
                auto it = nodeEngine.find(w.first);
                if (it == nodeEngine.end()) continue;
                const int src = it->second;
                if (src != dst && succ[src].insert(dst).second) ++indeg[dst];
            }
        }
    }
    std::vector<int> order;
    std::set<int> done;
    while (order.size() < indeg.size()) {
        bool progress = false;
        for (const auto &kv : indeg) {
            const int e = kv.first;
            if (done.count(e) || kv.second != 0) continue;
            order.push_back(e);
            done.insert(e);
            for (int v : succ[e]) --indeg[v];
            progress = true;
        }
        if (!progress) break;  // 环：退回引擎号顺序
    }
    if (order.size() < indeg.size()) {
        order.clear();
        for (const auto &kv : indeg) order.push_back(kv.first);
    }
    return order;
}

// NHP_GRAPH_PER_STREAM=1：planner 已确定每条 stream 的任务序列，逐 stream
// 单独录制 ACL Graph，再并发重放。跨 stream 依赖仍是 ACL event；event 的
// record 在 producer graph 内，wait 在 consumer graph 内，按 stream 拓扑序
// 录制。
void CaptureGraphsPerStream(std::vector<RankCtx> &ctxs, const SchedDoc &doc)
{
    for (auto &ctx : ctxs) {
        CheckAcl(aclrtSetDevice(ctx.device), "pstream graph set device");
        ctx.plans.clear();
        for (auto &kv : ctx.events) (void)aclrtDestroyEvent(kv.second);
        ctx.events.clear();
        BuildEvents(ctx, doc);
        PreparePlans(ctx, doc);
        CheckAcl(aclrtSynchronizeDevice(), "pstream graph sync prep");
        for (int e : StreamTopoOrder(doc)) {
            bool nonempty = false;
            for (const auto &kv : doc.streams) {
                if (EngineIndex(kv.first) == e && !kv.second.tasks.empty()) nonempty = true;
            }
            if (!nonempty) continue;
            CheckAcl(aclmdlRICaptureBegin(ctx.streams[e],
                                          ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),
                     "pstream capture begin");
            EnqueueStream(ctx, doc, e);
            CheckAcl(aclmdlRICaptureEnd(ctx.streams[e], &ctx.graphs[e]),
                     "pstream capture end");
            CheckAcl(aclrtSynchronizeStream(ctx.streams[e]), "pstream sync after capture");
        }
    }
}

double RunGraphsPerStreamOnce(std::vector<RankCtx> &ctxs)
{
    std::vector<double> rankEnd(ctxs.size(), 0.0);
    std::vector<std::string> errors(ctxs.size());
    const auto begin = Clock::now();
    std::vector<std::thread> workers;
    workers.reserve(ctxs.size());
    for (uint32_t r = 0; r < ctxs.size(); ++r) {
        workers.emplace_back([&ctxs, &rankEnd, &errors, r, begin]() {
            try {
                RankCtx &ctx = ctxs[r];
                CheckAcl(aclrtSetDevice(ctx.device), "pstream worker set device");
                for (int e = 0; e < kMaxStreams; ++e) {
                    if (ctx.graphs[e] == nullptr) continue;
                    CheckAcl(aclmdlRIExecuteAsync(ctx.graphs[e], ctx.streams[e]),
                             "pstream graph execute");
                }
                for (int e = 0; e < kMaxStreams; ++e) {
                    if (ctx.graphs[e] == nullptr) continue;
                    CheckAcl(aclrtSynchronizeStream(ctx.streams[e]),
                             "pstream graph sync");
                }
                rankEnd[r] = std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
            } catch (const std::exception &ex) {
                errors[r] = ex.what();
            }
        });
    }
    for (auto &w : workers) w.join();
    for (const auto &err : errors) {
        if (!err.empty()) throw std::runtime_error("pstream rank worker failed: " + err);
    }
    double makespan = 0.0;
    for (double v : rankEnd) makespan = std::max(makespan, v);
    return makespan;
}

std::vector<SegmentGraph> BuildSegments(const SchedDoc &doc)
{
    std::vector<SegmentGraph> out;
    for (int e = 0; e < kMaxStreams; ++e) {
        for (const auto &kv : doc.streams) {
            if (EngineIndex(kv.first) != e) continue;
            const auto &tasks = kv.second.tasks;
            int start = 0;
            auto close = [&](int end) {
                if (end <= start) return;
                SegmentGraph seg;
                seg.engine = e;
                seg.engineName = kv.first;
                seg.start = start;
                seg.end = end;
                if (!tasks[start].waits.empty()) seg.waits = tasks[start].waits;
                for (int i = start; i < end; ++i) {
                    if (tasks[i].publish) {
                        seg.publishes.emplace_back(tasks[i].node, tasks[i].tile);
                    }
                }
                out.push_back(std::move(seg));
            };
            for (int i = 0; i < static_cast<int>(tasks.size()); ++i) {
                if (i > start && !tasks[i].waits.empty()) {
                    close(i);
                    start = i;
                }
                if (tasks[i].publish) {
                    close(i + 1);
                    start = i + 1;
                }
            }
            close(static_cast<int>(tasks.size()));
        }
    }
    return out;
}

void CaptureSegments(std::vector<RankCtx> &ctxs, const SchedDoc &doc)
{
    for (auto &ctx : ctxs) {
        CheckAcl(aclrtSetDevice(ctx.device), "segment graph set device");
        ctx.plans.clear();
        for (auto &kv : ctx.events) (void)aclrtDestroyEvent(kv.second);
        ctx.events.clear();
        BuildEvents(ctx, doc);
        PreparePlans(ctx, doc);
        CheckAcl(aclrtSynchronizeDevice(), "segment graph sync prep");
        ctx.segments = BuildSegments(doc);
        for (auto &seg : ctx.segments) {
            aclrtStream stream = ctx.streams[seg.engine];
            CheckAcl(aclmdlRICaptureBegin(stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),
                     "segment capture begin");
            for (int i = seg.start; i < seg.end; ++i) {
                const auto &task = doc.streams.at(seg.engineName).tasks[i];
                auto plan = ctx.plans.find({task.node, task.tile});
                if (plan == ctx.plans.end()) throw std::runtime_error("missing plan for segment");
                plan->second->Enqueue(stream);
            }
            CheckAcl(aclmdlRICaptureEnd(stream, &seg.graph), "segment capture end");
            CheckAcl(aclrtSynchronizeStream(stream), "segment sync after capture");
        }
    }
}

double RunSegmentsOnce(std::vector<RankCtx> &ctxs)
{
    std::vector<double> rankEnd(ctxs.size(), 0.0);
    std::vector<std::string> errors(ctxs.size());
    const auto begin = Clock::now();
    std::vector<std::thread> workers;
    workers.reserve(ctxs.size());
    for (uint32_t r = 0; r < ctxs.size(); ++r) {
        workers.emplace_back([&ctxs, &rankEnd, &errors, r, begin]() {
            try {
                RankCtx &ctx = ctxs[r];
                CheckAcl(aclrtSetDevice(ctx.device), "segment worker set device");
                std::set<std::pair<std::string, int>> recorded;
                for (auto &seg : ctx.segments) seg.launched = false;
                size_t remaining = ctx.segments.size();
                while (remaining > 0) {
                    bool progress = false;
                    for (auto &seg : ctx.segments) {
                        if (seg.launched) continue;
                        bool ready = true;
                        for (const auto &w : seg.waits) {
                            if (recorded.find(w) == recorded.end()) { ready = false; break; }
                        }
                        if (!ready) continue;
                        aclrtStream stream = ctx.streams[seg.engine];
                        for (const auto &w : seg.waits) {
                            auto it = ctx.events.find(w);
                            if (it == ctx.events.end()) throw std::runtime_error("segment wait on unpublished flag");
                            CheckAcl(aclrtStreamWaitEvent(stream, it->second), "segment wait");
                        }
                        CheckAcl(aclmdlRIExecuteAsync(seg.graph, stream), "segment execute");
                        for (const auto &p : seg.publishes) {
                            auto it = ctx.events.find(p);
                            if (it == ctx.events.end()) throw std::runtime_error("segment missing event");
                            CheckAcl(aclrtRecordEvent(it->second, stream), "segment record");
                            recorded.insert(p);
                        }
                        seg.launched = true;
                        --remaining;
                        progress = true;
                    }
                    if (!progress) throw std::runtime_error("segment wait cycle");
                }
                for (int e = 0; e < kMaxStreams; ++e) {
                    CheckAcl(aclrtSynchronizeStream(ctx.streams[e]), "segment sync");
                }
                rankEnd[r] = std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
            } catch (const std::exception &ex) {
                errors[r] = ex.what();
            }
        });
    }
    for (auto &w : workers) w.join();
    for (const auto &err : errors) {
        if (!err.empty()) throw std::runtime_error("segment rank worker failed: " + err);
    }
    double makespan = 0.0;
    for (double v : rankEnd) makespan = std::max(makespan, v);
    return makespan;
}

// 每轮重建通信域（NHP_COMM_PER_ROUND=1，实验开关）：c08 类"HCCL collective ×
// 多轮"形态实测失败率 40–50% 且与并发无关（W5/W6 同窗对照），去 HCCL 仅 ~4%
// （W8）——指向 HCCL 每 collective 内部累积态腐蚀后续计算内核（doc/design.md
// §8.6）。在整轮排干后销毁并重建通信域可重置该累积态；重建开销在计时之外。
void RecreateComms(HcclSession &hccl, std::vector<RankCtx> &ctxs,
                   const std::vector<int32_t> &devices)
{
    hccl.Destroy();
    hccl.Init(devices);
    std::vector<int32_t> devs = devices;  // HcclCommInitAll 收非 const 指针
    CheckHccl(HcclCommInitAll(static_cast<uint32_t>(devs.size()), devs.data(), hccl.Comms()),
              "reinit HcclCommInitAll");
    for (size_t r = 0; r < ctxs.size(); ++r) ctxs[r].comm = hccl.Comms()[r];
}

double ChecksumAll(std::vector<RankCtx> &ctxs, const SchedDoc &doc)
{
    double sum = 0.0;
    for (auto &ctx : ctxs) {
        CheckAcl(aclrtSetDevice(ctx.device), "set device for checksum");
        for (const auto &node : doc.nodes) {
            NodeBuf &buf = ctx.bufs[node.id];
            const size_t elems = std::min<size_t>(buf.outBytes / 2, 8192);
            std::vector<uint16_t> host(elems);
            CheckAcl(aclrtMemcpy(host.data(), elems * 2, buf.out, elems * 2, ACL_MEMCPY_DEVICE_TO_HOST),
                     "checksum copy");
            for (size_t i = 0; i < elems; ++i) {
                __fp16 h = 0;
                std::memcpy(&h, &host[i], sizeof(h));
                sum += static_cast<float>(h);
            }
        }
    }
    return sum;
}

double Median(std::vector<double> values)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 == 1 ? values[mid] : (values[mid - 1] + values[mid]) / 2.0;
}

} // namespace

// 外部库/可执行文件共用入口。NHP_DEVICES=4,5 可显式指定物理卡；
// 未设置时保持旧行为（0..nranks-1）。
int RunPlanCli(int argc, char **argv)
{
    std::vector<int32_t> devices;
    // HCCL 通信域按契约收尾（runtime/hccl_session.hpp）：运行期显式、逐 rank
    // 对称销毁，成功/失败路径都由 RAII 兜底。泄漏到进程退出会把析构推进
    // __cxa_finalize 期触发跨库 UAF（doc/design.md §7.1）。tensor/executor
    // 维持泄漏不动（§6：析构次序不当同样堆损坏，泄漏是当前安全前提下最稳态）。
    HcclSession hccl;
    try {
        CheckAcl(aclInit(nullptr), "aclInit");
        if (const char *syncOut = std::getenv("NHP_SYNC_BENCH")) {
            RunSyncBench(syncOut);
            return 0;
        }
        if (argc < 5) {
            std::fprintf(stderr, "usage: nhp_exec <schedule.json> <iters> <warmup> <out.json>\n");
            return 2;
        }
        const std::string schedPath = argv[1];
        const int iters = std::atoi(argv[2]);
        const int warmup = std::atoi(argv[3]);
        const std::string outPath = argv[4];

        const Value root = nhp::json::ParseFile(schedPath);
        SchedDoc doc = ParseSchedule(root);
        // 拓扑规范化：默认计算 lane 合并（安全并发面），NHP_LAYOUT=4stream 回退
        NormalizeTopology(doc);
        {
            const char *layout = std::getenv("NHP_LAYOUT");
            const bool legacy = layout != nullptr && std::string(layout) == "4stream";
            std::fprintf(stderr, "nhp_exec layout: %s (streams: aicore/aic/aiv/aicpu/hccl)\n",
                         legacy ? "4stream" : "split + mix-exclusive");
        }
        const uint32_t nranks = std::max<uint32_t>(1, doc.nranks);

        devices.resize(nranks);
        const char *devEnv = std::getenv("NHP_DEVICES");
        if (devEnv != nullptr && *devEnv != '\0') {
            std::string s(devEnv);
            size_t pos = 0;
            uint32_t r = 0;
            while (pos <= s.size() && r < nranks) {
                const size_t comma = s.find(',', pos);
                const std::string tok = s.substr(pos, comma == std::string::npos
                                                          ? std::string::npos
                                                          : comma - pos);
                if (tok.empty()) throw std::runtime_error("NHP_DEVICES has empty entry");
                devices[r++] = std::stoi(tok);
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
            if (r != nranks) {
                throw std::runtime_error("NHP_DEVICES needs exactly " +
                                         std::to_string(nranks) + " devices");
            }
        } else {
            for (uint32_t r = 0; r < nranks; ++r) devices[r] = static_cast<int32_t>(r);
        }
        // HCCL 初始化前先建立每个设备的 runtime 上下文
        for (uint32_t r = 0; r < nranks; ++r) {
            CheckAcl(aclrtSetDevice(devices[r]), "warmup set device");
            aclrtStream warm = nullptr;
            CheckAcl(aclrtCreateStream(&warm), "warmup stream");
            CheckAcl(aclrtSynchronizeDevice(), "warmup sync");
            (void)aclrtDestroyStream(warm);
        }
        hccl.Init(devices);
        CheckHccl(HcclCommInitAll(nranks, devices.data(), hccl.Comms()), "HcclCommInitAll");

        std::vector<RankCtx> ctxs(nranks);
        for (uint32_t r = 0; r < nranks; ++r) {
            RankCtx &ctx = ctxs[r];
            ctx.rank = r;
            ctx.nranks = nranks;
            ctx.device = devices[r];
            ctx.comm = hccl.Comms()[r];
            CheckAcl(aclrtSetDevice(ctx.device), "init set device");
            for (int e = 0; e < kMaxStreams; ++e) {
                CheckAcl(aclrtCreateStream(&ctx.streams[e]), "create engine stream");
            }
            for (const auto &node : doc.nodes) AllocateNode(ctx, node, nranks);
            BuildEvents(ctx, doc);
            UploadInputs(ctx, doc);
            CheckAcl(aclrtSynchronizeDevice(), "sync prepare");
        }

        const bool commPerRound = std::getenv("NHP_COMM_PER_ROUND") != nullptr;
        const bool graphMode = GraphMode();
        const bool graphPerStream = GraphPerStreamMode();
        const bool graphSegment = std::getenv("NHP_GRAPH_SEGMENT") != nullptr;
        const int graphKinds = (graphMode ? 1 : 0) + (graphPerStream ? 1 : 0) +
                               (graphSegment ? 1 : 0);
        if (graphKinds > 1) {
            throw std::runtime_error("NHP_GRAPH / NHP_GRAPH_PER_STREAM / "
                                     "NHP_GRAPH_SEGMENT 只能三选一");
        }
        if (graphMode) {
            if (commPerRound) throw std::runtime_error("NHP_GRAPH 与 NHP_COMM_PER_ROUND 不兼容");
            CaptureGraphs(ctxs, doc);
        } else if (graphPerStream) {
            if (commPerRound) {
                throw std::runtime_error("NHP_GRAPH_PER_STREAM 与 NHP_COMM_PER_ROUND 不兼容");
            }
            CaptureGraphsPerStream(ctxs, doc);
        } else if (graphSegment) {
            if (commPerRound) {
                throw std::runtime_error("NHP_GRAPH_SEGMENT 与 NHP_COMM_PER_ROUND 不兼容");
            }
            CaptureSegments(ctxs, doc);
        }
        for (int i = 0; i < warmup; ++i) {
            uint32_t timeouts = 0;
            if (graphMode) {
                (void)RunGraphOnce(ctxs);
            } else if (graphPerStream) {
                (void)RunGraphsPerStreamOnce(ctxs);
            } else if (graphSegment) {
                (void)RunSegmentsOnce(ctxs);
            } else {
                (void)RunOnce(ctxs, doc, timeouts);
            }
            if (commPerRound) RecreateComms(hccl, ctxs, devices);
        }

        std::vector<double> times;
        uint32_t gateTimeouts = 0;
        for (int i = 0; i < iters; ++i) {
            uint32_t timeout = 0;
            const double us = graphMode ? RunGraphOnce(ctxs)
                                        : (graphPerStream
                                               ? RunGraphsPerStreamOnce(ctxs)
                                               : (graphSegment
                                                      ? RunSegmentsOnce(ctxs)
                                                      : RunOnce(ctxs, doc, timeout)));
            gateTimeouts += timeout;
            times.push_back(us);
            if (commPerRound) RecreateComms(hccl, ctxs, devices);
        }
        const double checksum = ChecksumAll(ctxs, doc);

        Value result = Value::MakeObject();
        result.Set("dag", Value::MakeString(doc.dag));
        result.Set("group", Value::MakeString(doc.group));
        result.Set("nranks", Value::MakeNumber(nranks));
        result.Set("iters", Value::MakeNumber(iters));
        result.Set("warmup", Value::MakeNumber(warmup));
        result.Set("p50_us", Value::MakeNumber(Median(times)));
        result.Set("gate_timeouts", Value::MakeNumber(gateTimeouts));
        result.Set("checksum", Value::MakeNumber(checksum));
        result.Set("ok", Value::MakeBool(gateTimeouts == 0));
        Value timesArr = Value::MakeArray();
        for (double t : times) timesArr.Push(Value::MakeNumber(t));
        result.Set("times_us", timesArr);

        nhp::json::WriteFile(outPath, result);
        std::printf("%s\n", result.Dump().c_str());
        std::fflush(stdout);
        // 运行期契约收尾：显式销毁 HCCL 通信域（HCCL_SESSION 契约见
        // runtime/hccl_session.hpp）。不要改用 std::_Exit 跳过析构——实测会令
        // HCCL/流资源残留、污染后续进程（逐 tile 并发 507015/507035 从 ~20%
        // 升至必现）。
        hccl.Destroy();
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "nhp_exec error: %s\n", ex.what());
        std::fflush(stderr);
        hccl.Destroy();  // 失败路径同样在运行期销毁，避免退出期 UAF 掩盖真实错误码
        return 1;
    }
}

#ifndef KR_LIBRARY
int main(int argc, char **argv)
{
    return RunPlanCli(argc, argv);
}
#endif


