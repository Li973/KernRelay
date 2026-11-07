# KernRelay 0.3 设计

## 1. 目标与边界

KernRelay 解决的是“**依赖算子在不同引擎上的切分/提前放行**”问题：当一个 producer（例如 AIC 上的 MatMul）按 tile 完成时，consumer（例如 AIV 上的 SiLU、Add+RMSNorm，或 HCCL 通信）不必等整个 producer 结束，可以在 tile 0 就绪后启动。这样把原本由依赖边串行化的异构算子重排成流水。

0.3 不试图做通用图编译器，也不做所有模型的万能运行时。边界：

- DAG 调度与执行器面向 Ascend 910B3 / CANN 9.0；引擎为 `aicore/aic/aiv/aicpu/hccl`。
- LLM 端到端支持 Qwen2.5 dense 与 Qwen3-VL 的**文本语言模型**（dense）。MoE 只做 DAG/算子统计，不执行。
- 不实现 vision/audio tower；VL 模型只跑 `model.language_model.* + lm_head` 的文本 prefill。
- 不做量化；全程 bf16/fp16。

## 2. 总体结构

```
外部项目/模型目录
   │  export_model.py（HF -> manifest + weights.bin，权重在外部 cache）
   ▼
DAG / LLM adapter
   │  layer DAG: embed -> [ln/qkv/qk_norm/rope/attn/o/addnorm/gate/up/silu/mul/down/addnorm]×L -> lm
   ▼
planner-2
   │  结构校验 -> 正常调度 A -> 气泡 -> 1:N 切分枚举 -> 离散事件模拟 -> 计划
   ▼
执行器
   ├─ nhp_exec：通用 schedule（E3/E4/E5/E9）
   └─ libkernrelay.so：Qwen 单卡 / PP=2 / TP-MLP
```

### 2.1 两种项目形态

1. **LLM 项目**：输入 HF 模型目录（或已导出的 KernRelay 目录）。`kernrelay.optimize` 自动导出、建层 DAG、对唯一 shape 预跑 k 粒度、应用 planner、执行单卡或 TP。
2. **DAG 项目**：输入 nodes/edges JSON（算子的 `op`、`engine`、`shape`、`tiles`）。用 `nhp_exec` 执行 schedule；用 planner-2 做切分选择。

## 3. DAG 与 schedule

节点字段：`id/name`、`op`、`engine`、`shape`、`dtype`、`tiles`、`iters`、`blocks`。边为数据依赖（`kind=data`）。

schedule 把每个节点展开成 tile 列表，每个 tile 带：

- `stream`：`aicore/aic/aiv/aicpu/hccl` 之一；
- `wait`：该 tile 开始前要等待的 `(producer, tile)`；
- `publish`：完成后是否发布通知；
- 行偏移/行数；对通信还有发送/接收 rank。

执行器按 stream 建流，按全局拓扑序入队；跨流 tile 默认用 ACL event 同步，Qwen `relay_flag`/PP 等模式用 GM flag 空 kernel 自旋。

## 4. 引擎、流与设备侧同步

### 4.1 跨流同步：ACL event 与 GM flag 空 kernel

每条跨流边使用 32 B 控制块：

```
[published, consumed, status, pad]  // int32
```

- producer 后插入 `kr_publish_aic/aiv` 空 kernel：DMA 读计数、加一写回、DCCI、DSB。
- consumer 前插入 `kr_wait_aic/aiv` 空 kernel：volatile 轮询 `published >= consumed+1`，写 `consumed` 与 `status`（1 成功 / 2 超时）。
- 计数器只增不清零，因此同一 program 可反复入队、可进 ACL Graph 任意重放；超时上限 20,000,000 spins。

优点：不经过 host、不依赖 ACL event；消费者可以是任意 ACLNN 算子，不需要 kernel 内自旋。代价：每条边两个空 kernel。

通用 `nhp_exec` 默认用 ACL event（零额外 kernel）；Qwen `relay_flag`/PP 使用 GM flag 空 kernel。两条路径都由设备/流调度器保证依赖，host 不参与运行期等待。

### 4.2 流布局

- 单卡双流：AIC 计算流 + AIV/HCCL 流；层内 8 对 flag 边。
- 单卡单流：所有算子一条流（serial / ACL Graph 最优路径）。
- PP/TP：每 stage 一条 AIC 计算流 + 一条 AIV/HCCL 流；HCCL Send/Recv/AllReduce 在 AIV/HCCL 流上，与 AIV 同流保序。

### 4.3 ACL Graph

- 整图多流暂不支持；每流单独 Graph 支持（`NHP_GRAPH_PER_STREAM`，跨流用 capture-progress event）；GM flag 空 kernel 也可逐流捕获。
- C++ 直接调用 `aclmdlRICaptureBegin` + `aclnnFlashAttentionScore`（V1/BSND）可捕获；Python v1 包装报 107025，Python v3/BNSD 可捕获。
- 图重放的收益主要是把 host 从关键路径移除；Qwen 每层算子少，逐 stream 图并不总是比单流图快。

## 5. 切分与提前放行

### 5.1 1:1 与 1:N

- **1:1**：producer 切 k 个 tile，consumer 也切 k 个 tile，第 i 个 consumer 只依赖第 i 个 producer。用于 MatMul→GELU/SiLU/AddNorm 等。
- **1:N**：consumer 在 producer 的每个 tile 完成后都启动一次（或 fanout 到多个 consumer）。用于 attention 输出→多个逐行算子、MatMul→多个 AIV 后继。
- **partial consumer**：allreduce 只读 producer 前若干行时，只等覆盖该行范围的最少 producer tile（E9 mixed）。

### 5.2 收益判据

planner 的决策单位是“气泡”：consumer 在引擎上等待 producer 的时间 `dep_gap = ready - engine_free`。对每个气泡尝试 producer/consumer 的 k 组合，模拟 makespan；只有 `gain >= 1 µs` 才接受。

切分税来自：

- 每个子 kernel 的固定 launch/tiling 开销；
- N 切/大 N 的 TransData、Slice/ViewCopy；
- K 维切分需要累加，几乎必负；
- 跨流 event/flag 的固定成本（每个切分节点每 tile 计 event 成本）。

## 6. planner-2

流程：

1. 计算入度轮次 A（同引擎流按拓扑序串行，跨流可并发）。
2. 在 A 上找气泡：数据依赖导致的引擎空闲，且关键边跨引擎。
3. 对每个气泡，枚举 producer/consumer 的 k（1..16，按 shape 可用），可选切 consumer 链。
4. 用 `simulate` 重算 makespan；接受正收益最大的组合；多轮直到无正收益。
5. 输出 splits、计划 makespan、每轮决策（接受/拒绝原因）。

结构校验在规划前执行：空节点、重名、未知算子、非法引擎、MatMul/attention/embedding/通信坏 shape、边端点缺失、自环、成环、GQA head 不整除。校验失败直接拒绝，不进入模拟。

## 7. LLM 运行时

### 7.1 权重格式

`scripts/export_model.py` 把 HF 权重导出为：

- `manifest.json`：`config`（arch/H/I/L/HQ/HKV/head_dim/V/eps/qk_norm/attention_bias/tie/rope_theta）+ 每个 tensor 的 `offset/bytes/shape`；
- `weights.bin`：单文件连续 bf16；matmul 权重转置为 `[K,N]`；
- `ids.bin`、`ref_last_logits.bin`、`ref.json`：确定性输入与参考 argmax/maxdiff。

支持：

- Qwen2.5：`model.*`，有 q/k/v bias，head_dim=H/HQ，无 QK norm；
- Qwen3-VL dense text：`model.language_model.*`，无 attention bias，`head_dim` 显式（128），QK RMSNorm（对每个 head 的 D 维归一化）。

### 7.2 单卡

`kr_qwen_run`：

- `serial`：单流逐算子；
- `serial_fused`：Add+RMSNorm 融合；Qwen3 还包含 QK RMSNorm；
- `serial_graph` / `graph_whole`：单流 ACL Graph；
- `relay_event` / `relay_flag` / `relay_flag_graph`：双流 + event/flag/逐 stream 图。

Qwen3 的 QK RMSNorm 以 2D `[S*HQ, D]` 调用 ACLNN RmsNorm（4D 输入报 561002），结果再以 4D 视图供 RoPE/Flash 使用。

### 7.3 PP=2

stage0：embedding + 前 L/2 层；stage1：后 L/2 层 + final norm + lm_head。stage0 最后一层只做残差，一次 HCCL Send 把 hidden 发给 stage1；stage1 收到后自己算 ln1。R 个请求一次入队，stage0 算 r+1 时 stage1 算 r。

**归因**：PP 的 wall = “stage0 的 R 个请求总时间 + stage1 的一个请求时间”，同 DAG 单流对照更快；因此 PP 是设备层划分 + 请求流水，不是 KernRelay 的切分收益。PP 只作为 device-parallel baseline。

### 7.4 TP-MLP（KernRelay 归因路径）

- attention 复制；gate/up 按输出列、down 按 K 行各切一半；
- 每个 rank 的 down 是部分和，每层一次 `HcclAllReduce(SUM, BFP16)` 得到完整 down；
- `tp2`：down 整幅 → 一次 AllReduce → 整幅 AddNorm；
- `tp2_split`：down 按行切 k 块；每块 down → 该块 AllReduce → 该块 AddNorm；下一块 down 与上一块 AllReduce 并发，AddNorm 逐块放行。

两者设备数、权重切分、通信字节数完全相同，只差调度；`tp2_split/tp2` 才是 KernRelay 的归因比值。

AllReduce 固定开销约 250–300 µs（2048×896；7B 的 14.7 MB 约 1.05 ms），因此：

- 0.5B/4B：通信相对层计算太大，TP 与切分都负收益；
- 7B/8B：层计算足以隐藏部分 AllReduce，k=2/4 有正收益，k=8 因固定开销回落。

### 7.5 为什么不做 30B-A3B

Qwen3-Omni-30B-A3B 是 128 experts、top-8 的 MoE（48 层、hidden 2048、head_dim 128）。完整执行需要 routing/top-k、token permutation、grouped GEMM、scatter，以及 60 GB 权重；0.3 只解析 config、输出 DAG/算子统计并明确 SKIP，不用 dense 近似冒充。

## 8. 库与进程模型

`include/kernrelay.h` 提供：

| ABI | 作用 |
|---|---|
| `kr_version` | 版本 |
| `kr_plan_run` | 执行 schedule JSON（通用 DAG） |
| `kr_qwen_run` | 单卡 Qwen（7 种模式） |
| `kr_qwen_profile_many` | 一次加载权重，批量测算子 k 粒度 |
| `kr_qwen_pp_run` | PP=2 |
| `kr_qwen_tp_run` | TP-MLP |

**进程级限制**：CANN 9.0 的 ACLNN executor 缓存与 HCCL 生命周期不支持同进程反复初始化；一次调用跑完一份计划，第二次调用请用新进程。`libkernrelay.so` 不依赖 argv，外部项目直接链接即可。

## 9. 正确性与验证

- 所有 LLM 路径用 `ref_last_logits.bin` 校验末 token argmax；bf16 maxdiff 通常 < 0.2。
- GM flag 模式统计每条边的 `status`；超时数为 0 才算通过。
- E3/E4/E5/E9 用 checksum 对比 serial/B0/split；不一致即判失败。
- planner 对问题 DAG 只判拒绝，不记录耗时。

## 10. 限制与下一步

- TP 只切 MLP；attention 复制。后续可切 attention（每层 2 次 AllReduce）或做 block-wise K/V 序列并行，把每层通信降到 0.26–1 MB。
- 30B MoE、vision/audio tower 未实现。
- 细粒度 p2p Send/Recv 的双流对称交换探针未跑通；TP 当前用 AllReduce。
- 逐 stream ACL Graph 对 Qwen 的收益受每层算子数限制；单流图仍是单卡最优。
