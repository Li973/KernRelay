# KernRelay 0.3 最终实验结果

> 日期：2026-11-06。所有 LLM/TP/planner/profile 数据由最终 `libkernrelay.so` 与 `kr_qwen` 重跑；E3/E4/E5/E9 执行器 `nhp_exec` 源自 2026-11-02 未改，2026-11-06 用最终二进制重跑关键 case 冒烟（方向一致），全量表见 §4。

## 1. 协议

- 硬件：Ascend 910B3；CANN 9.0.0；bf16；seq=2048，batch=1。
- LLM 单卡：3 进程 ×（0.5B/4B 5 轮，7B/8B 3 轮），warmup 2；best 为全局最小，typical 为各进程最小值的中位数。
- TP-MLP：单进程双卡，3 个进程；3 轮，warmup 1；`tiles` 为 down 行切块数。
- 1:1/1:N/自设 DAG：3 进程 × 5 轮（`nhp_exec`，D12 口径）。
- 正确性：argmax 与 `ref_last_logits.bin` 一致；checksum 一致；flag 超时为 0。
- 表内单位：LLM 为 ms；E 系列为 µs。

## 2. 端到端：单卡基线

`serial`：单流逐算子（精确 FlashAttention）；`fused`：Add+RMSNorm 融合（Qwen3 含 QK RMSNorm）；`graph`：单流 ACL Graph。

| 模型 | serial | fused | graph | graph/serial |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 28.0 | 26.1 | **22.1** | 1.266 |
| Qwen2.5-7B | 146.8 | 146.9 | **146.2** | 1.004 |
| Qwen3-VL-4B（文本） | 99.5 | 99.8 | **99.3** | 1.002 |
| Qwen3-VL-8B（文本） | 170.4 | 170.2 | **170.1** | 1.002 |

- 0.5B 的融合与图收益明显（计算/启动占比高）；7B/8B 是计算受限，图与融合的收益在噪声内。
- Qwen3 的 `fused` 在 4B 上略慢（99.8 vs 99.5），说明融合收益是 shape 相关的，不能外推。

## 3. 端到端：TP-MLP 与 KernRelay 归因

TP-MLP：attention 复制，gate/up 按输出列、down 按 K 行各切一半；每层一次 HCCL AllReduce。

- `TP k=1`：down 整幅 → 一次 AllReduce → 整幅 AddNorm。
- `TP split`：down 按行切 k 块，AllReduce/AddNorm 逐块放行。

| 模型 | TP k=1 (ms) | TP split (ms) | split/k=1 | TP split vs 单卡 graph |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B（k=2） | 40.6 | 49.4 | 0.823 | 0.448 |
| Qwen2.5-7B（k=4） | 125.9 | 118.2 | **1.065** | **1.237** |
| Qwen3-VL-4B（k=2） | 108.3 | 105.0 | **1.031** | 0.945 |
| Qwen3-VL-8B（k=4） | 160.5 | 153.4 | **1.046** | **1.109** |

**归因**：TP split 相对 TP k=1 的 1.065（7B）/1.046（8B）才是 KernRelay 的切分/提前放行收益；TP 相对单卡的 1.237/1.109 中还包含 2 卡设备并行。0.5B 的 AllReduce（434 µs/层）相对层计算太大，TP 与切分都为负；4B 的 TP 仍慢于单卡，但切分本身已转正（1.031×）。

所有配置的末 token argmax 与参考一致：0.5B 33、7B 82、VL-4B/8B 151645；maxdiff ≤ 0.19。

PP=2 仅作吞吐基线（R=1，非 KernRelay 归因）：Qwen2.5-0.5B 29.1 ms、Qwen2.5-7B 154.3 ms、Qwen3-VL-4B 132.9 ms、Qwen3-VL-8B 186.0 ms，argmax 均与参考一致。

### 3.1 planner 对 TP 层气泡的判断

层 DAG：`down(AIC) → AllReduce(HCCL) → RMSNorm(AIV) → 下一层 QKV(AIC)`；allreduce 曲线用实测固定开销（约 283 µs/tile）。

| 模型 | 基线层 (µs) | 计划层 (µs) | planner 选择 | 预测总收益 | 实测最优 |
|---|---:|---:|---|---:|---|
| Qwen2.5-0.5B | 587.5 | 587.5 | 0 切分 | 0 | k=1（拒绝切分，正确） |
| Qwen2.5-7B | 2433.9 | 2278.2 | down/ar/an/qkv k=2 | 4.36 ms / 28 层 | k=4，7.69 ms |

planner 在 0.5B 上拒绝、7B 上接受，方向与实测一致；数值保守（线性 allreduce 曲线未计 k=4 的并发收益）。端到端 `tp2_split` 的 k 目前由运行时参数固定，planner 只在层 DAG 上复算方向。

## 4. 1:1、1:N、自设 DAG、计算/通信

执行器 `nhp_exec`；serial=未切分单流，B0=未切分多流，split=KernRelay 切分。校验全部通过（除已知的 AICPU 行偏移负例）。

### 4.1 E3/E4（1:1 与 1:N）

| case | serial (µs) | B0 (µs) | split (µs) | serial/split | B0/split |
|---|---:|---:|---:|---:|---:|
| e3_mm_gelu_k2 | 971.7 | 981.1 | 890.4 | 1.091 | 1.102 |
| e3_mm_gelu_k4 | 978.8 | 974.0 | 889.2 | 1.101 | 1.095 |
| e3_mm_gelu_k8 | 1008.4 | 1014.9 | 911.3 | 1.107 | 1.114 |
| e3_mm_gelu_k16 | 1005.6 | 1022.0 | 959.2 | 1.048 | 1.066 |
| e3_mm_gelu_q2 | 1012.2 | 998.2 | 920.7 | 1.099 | 1.084 |
| e3_mm_gelu_q4 | 1003.2 | 998.4 | 1048.0 | 0.957 | 0.953 |
| e3_ln_mm_k4 | 1890.3 | 1885.5 | 1456.8 | 1.298 | 1.294 |
| e3_mm_softmax_k4 | 1133.0 | 1132.3 | 888.4 | 1.275 | 1.274 |
| e3_mm_cast_k4 | 1018.6 | 966.3 | 944.2 | 1.079 | 1.023 |
| e4_same_aiv_k4 | 1406.1 | 1424.5 | 1318.3 | 1.067 | 1.081 |
| e4_cross_hccl_k4 | 8233.8 | 7987.8 | 7704.3 | 1.069 | 1.037 |
| e4_cross_aicpu_k1 | 145190.7 | 145291.1 | 146380.2 | 0.992 | 0.993 |
| e4_cross_aicpu_k4 | 144988.4 | 145088.4 | 146845.1 | 0.987 | 0.988 |

- 行切 1:1（MatMul→GELU）在 k=2/4/8 稳定 1.09–1.11×；k=16 回落；q4 为负（0.957×），说明切分收益依赖形状/队列。
- `ln→mm`、`mm→softmax` 的跨引擎 1:1 为 1.27–1.30×。
- 1:N（same-AIV、跨 HCCL）为 1.04–1.08×。
- AICPU 作为 producer/consumer 为负；`e4_cross_aicpu_k4` 的 checksum 不一致（AICPU 行偏移不支持），只保留 k=1 的负结果。

### 4.2 E5（计算/通信）

| case | serial (µs) | B0 (µs) | split (µs) | serial/split | B0/split |
|---|---:|---:|---:|---:|---:|
| e5_mm_ar_part_k2 | 1151.7 | 1175.9 | 961.7 | 1.198 | 1.223 |
| e5_mm_ar_part_k4 | 1194.8 | 1201.7 | 896.7 | **1.332** | **1.340** |
| e5_mm_ar_part_k8 | 1194.1 | 1191.9 | 959.3 | 1.245 | 1.242 |
| e5_mm_ar_part_k16 | 1152.7 | 1158.5 | 1171.5 | 0.984 | 0.989 |
| e5_mm_ar_full_k4 | 7933.5 | 7922.1 | 7676.2 | 1.034 | 1.032 |
| e5_mm_ag_part_k4 | 1170.8 | 1164.4 | 980.0 | 1.195 | 1.188 |
| e5_mm_rs_part_k4 | 932.9 | 954.4 | 752.8 | 1.239 | 1.268 |

partial AllReduce/AllGather/ReduceScatter 只读前若干行，k=4 为 1.19–1.33×；整幅 allreduce 仅 1.034×；k=16 因切分税回落。

### 4.3 E9（自设有利 DAG）

| case | serial (µs) | B0 (µs) | split (µs) | serial/split | B0/split |
|---|---:|---:|---:|---:|---:|
| e9_two_waves_k4 | 1791.2 | 1617.6 | 1532.0 | 1.169 | 1.056 |
| e9_two_waves_k8 | 1840.9 | 1648.0 | 1571.4 | 1.171 | 1.049 |
| e9_mixed_k4 | 1997.8 | 1810.9 | 1518.5 | 1.316 | 1.193 |
| e9_chain2_k4 | 2270.1 | 2276.0 | 2063.6 | 1.100 | 1.103 |
| e9_chain2_k8 | 2275.8 | 2290.0 | 2105.0 | 1.081 | 1.088 |

两波 DAG 的收益来自尾部提前；混合 DAG（partial allreduce + mm2 并发）在 k=4 为 1.32×；依赖链仍为正但更小。

### 4.4 最终二进制冒烟

2026-11-06 用最终 `nhp_exec` 重跑：`e3_mm_gelu_k4` split 903.8 µs（serial 1048.4，1.134×）；`e5_mm_ar_part_k4` split 997.0 µs（1.181×）；`e9_mixed_k4` 1.232×。与上表方向一致，差异在设备噪声/轮次内。

## 5. planner 鲁棒性（只判拒绝）

| DAG | 期望 | 结果 | 原因 |
|---|---|---|---|
| e0_tiny_mm_gelu | accept | ACCEPT | - |
| e3_ln_mm | accept | ACCEPT | - |
| e9_mixed | accept | ACCEPT | - |
| bad_cycle | reject | REJECT | 依赖成环: graph cycle |
| bad_unknown_op | reject | REJECT | 不支持算子 'frobnicate' |
| bad_shape | reject | REJECT | matmul M<=0 |
| bad_edge | reject | REJECT | edge dst 不存在: 'missing' |
| bad_dup_name | reject | REJECT | 节点名重复 |
| bad_attention | reject | REJECT | q_heads=14 不是 kv_heads=3 的整数倍 |
| aicpu_producer | accept(0-split) | ACCEPT | 结构合法，无正收益切分 |
| tiny_matmul | accept(0-split) | ACCEPT | 512 行太小，无正收益切分 |

6 个结构/语义非法的 DAG 全部拒绝；2 个合法但无收益的 DAG 被接受并返回 0 切分，未误报加速。

## 6. 算子清单与预跑

`optimize` 对每个模型建层 DAG、按唯一 shape 预跑 k=1/2/4/8（同类节点只测一次），再交给 planner。

| 模型 | DAG 节点 | 唯一 shape | attention k=1 (µs) | lm_head k=1 (µs) | down k=1 (µs) | lm_head k=8 (µs) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 339 | 12 | 196.6 | 4152.8 | 96.5 | 7970.8 |
| Qwen2.5-7B | 395 | 12 | 332.7 | 10305.7 | 1217.4 | 29820.5 |
| Qwen3-VL-4B | 543 | 14 | 367.8 | 8725.5 | 376.0 | 19638.6 |
| Qwen3-VL-8B | 543 | 13 | 364.8 | 13331.3 | 777.8 | 30752.1 |

全量 k 曲线在 `results/final/opt/<model>/optimize.json` 的 `profile_table`。行切税随 k 单调上升；`lm_head` 的 k=8 约 1.9–2.9× 于 k=1，因此 planner 在完整 LLM DAG 上对四个模型都返回 **0 切分**（计划 makespan 21.1/138.8/88.8/147.2 ms，与实测 22.1/146.2/99.3/170.1 ms 的差来自 host/固定开销）。

## 7. 30B-A3B（明确 SKIP）

| 项 | 值 |
|---|---|
| 架构 | Qwen3-Omni MoE thinker |
| hidden / layers / heads / kv / head_dim | 2048 / 48 / 32 / 4 / 128 |
| experts / top-k / moe_intermediate | 128 / 8 / 768 |
| vocab | 152064 |
| 结果 | **SKIP**：0.3 未实现 routing/top-k、permutation、grouped matmul、scatter；不用 dense 近似冒充 |

DAG/算子统计与 config 解析已完成，可作为 1.1 的 MoE 扩展输入。

## 8. 结论

1. 行切 1:1 是稳定正收益（1.09–1.11×）；`ln→mm`、`mm→softmax` 为 1.27–1.30×；1:N 为 1.04–1.08×；AICPU 为负。
2. partial 通信是计算/通信重叠的关键：k=4 partial allreduce 1.33×，整幅 allreduce 仅 1.03×。
3. planner 能拒绝 6 类问题 DAG；对合法但无收益的 DAG 返回 0 切分。
4. 真实 LLM：0.5B/7B/VL-4B/VL-8B 单卡 graph 为 22.1/146.2/99.3/170.1 ms；TP split 相对 TP k=1 为 0.823/1.065/1.031/1.046，7B/8B 的总加速 1.237/1.109 中含设备并行。
5. 30B-A3B MoE 在 0.3 明确 SKIP；不用近似结果冒充。
