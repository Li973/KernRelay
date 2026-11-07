# KernRelay 0.3

KernRelay 是面向 **Ascend 910B3 / CANN 9.0** 的多算子 DAG 调度与运行时。它把依赖算子拆到 AIC / AIV / HCCL 等引擎与流上，按 tile 切分 producer，用 ACL event 或设备侧 GM flag 空 kernel 提前放行 consumer，从而重叠原本被依赖关系串行化的异构算子；同时提供 planner、真实 LLM 运行时（单卡 / PP=2 / TP-MLP）和稳定 C ABI，可作为库被外部昇腾项目调用。

0.3 的目标不是“又一个算子库”，而是回答三个问题：

1. 依赖算子能否在 AIC/AIV/HCCL 之间安全地切分并行，收益由什么决定；
2. planner 能否依据实测的算子粒度代价，识别气泡并拒绝无收益切分；
3. 真实 LLM（含通信）能否通过这套运行时获得可归因、可复现的加速。

## 0.3 能力

- **1:1 与 1:N 重叠**：AIC producer 按行切分，AIV/HCCL consumer 逐 tile 放行；跨流依赖支持两种实现：通用 `nhp_exec` 默认 ACL event，Qwen `relay_flag`/PP 用 GM flag 空 kernel 自旋（publish/wait 计数器单调递增，可进 ACL Graph）。
- **planner-2**：DAG 结构/语义校验、正常调度 A、气泡检测、1:N 切分枚举、离散事件模拟；可拒绝成环、未知算子、坏 shape、缺边、重名、非法 GQA 等 DAG。
- **真实 LLM prefill**：Qwen2.5-0.5B / Qwen2.5-7B；Qwen3-VL-4B / Qwen3-VL-8B 的**文本语言模型**（支持 `head_dim=128` 与 QK RMSNorm）。
- **执行路径**：单卡 serial / serial+融合 / ACL Graph；PP=2 分层流水；TP-MLP（attention 复制、MLP 各半、每层一次 HCCL AllReduce），支持 down 按行切分、AllReduce/AddNorm 逐块放行。
- **库化**：`bin/libkernrelay.so` + `include/kernrelay.h`，6 个 C ABI；Python 侧仅为 ctypes 薄封装。
- **一条命令优化外部项目**：`project -> DAG -> 同类节点只测一次 -> planner -> 执行最优调度`。
- **可复现实验**：`repro/` 下每个实验一个 `.sh`，模型位置由命令行 `--qwen-dir` / `--project` 指定。

## 支持矩阵

| 模型 | 架构 | 单卡 | PP=2 | TP-MLP | 说明 |
|---|---|:---:|:---:|:---:|---|
| Qwen2.5-0.5B | Qwen2.5 dense | ✅ | ✅ | ✅ | 基准小模型 |
| Qwen2.5-7B | Qwen2.5 dense | ✅ | ✅ | ✅ | TP 有正收益 |
| Qwen3-VL-4B-Instruct | Qwen3 dense text | ✅ | ✅ | ✅ | 仅文本 prefill，不含视觉塔 |
| Qwen3-VL-8B-Instruct | Qwen3 dense text | ✅ | ✅ | ✅ | 仅文本 prefill，不含视觉塔 |
| Qwen3-Omni-30B-A3B-Instruct | MoE（128 experts, top-8） | ⛔ | ⛔ | ⛔ | 0.3 仅完成 DAG/算子统计；MoE grouped matmul 未实现，明确 SKIP |

“文本语言模型”指只运行 `model.language_model.*` + `lm_head`，不包含 vision tower / audio tower；输入为纯文本 token。

## 构建

环境：Ascend 910B3，CANN 9.0.0，Python 3.11（推荐仓库内 `.venv`），g++，`torch` + `torch_npu` + `transformers`（仅导出/参考基线需要）。

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
repro/build.sh          # 构建 bin/nhp_exec、bin/libkernrelay.so、bin/kr_qwen
```

`bin/lib/libnhp_kernels.so`、`libnhp_aicpu_gate.so`、`libnhp_aicpu_custom.so` 是外部预编译依赖（源码不在本仓库）；本仓库的 `kernels/CMakeLists.txt` 只重建 `libkr_sync.so`。修改 `kernels/` 后重新编译该库即可。

## 快速开始：优化一个外部模型

模型权重**不进入仓库**。`--cache` 指定导出缓存目录（默认 `$KERNRELAY_CACHE`，未设置时为 `~/.cache/kernrelay`），`--project` 可以是 HF 模型目录、已导出的 KernRelay 目录或 DAG JSON。

```bash
# 一条命令：自动导出 -> 建 DAG -> 同类节点只测一次 -> planner -> 执行
python -m kernrelay optimize \
  --project /path/to/Qwen3-VL-8B-Instruct \
  --seq 2048 --device 1 --devices 1,2 --mode auto \
  --cache /data/kr_cache --out results/final/opt/qwen3-vl-8b
```

输出 `optimize.json`，包含：

- `summary`：DAG 节点/边、唯一 shape、层数；
- `profile_table`：每个唯一算子/ shape 在 k=1/2/4/8 的设备时间；
- `plan`：planner 的正常调度 makespan、计划 makespan、切分选择；
- `runs`：单卡与 TP 的 wall、argmax、参考 argmax、maxdiff。

### C ABI

```c
#include "kernrelay.h"

char out[1 << 20];
kr_qwen_run("/data/kr_cache/qwen2.5-0.5b", "graph_whole", 2048,
            /*device=*/1, /*warmup=*/1, /*iters=*/3, out, sizeof(out));

kr_qwen_tp_run("/data/kr_cache/qwen2.5-7b", "tp2_split", 2048,
               /*dev0=*/1, /*dev1=*/2, /*tiles=*/4,
               /*warmup=*/1, /*iters=*/3, out, sizeof(out));
```

编译：`-I include -L bin -lkernrelay -Wl,-rpath,'$ORIGIN/lib'`，运行时 `LD_LIBRARY_PATH=bin/lib`。
当前 ABI 是**进程级**的：一次调用跑完一份计划；不要在同一进程连续第二次初始化 ACL/HCCL。

### 自定义 DAG 项目

外部项目把算子图写成 JSON（`nodes`/`edges`，含 `op`、`engine`、`shape`），即可复用执行器和 planner：

```bash
python -m kernrelay validate --dag my_dag.json
python -m kernrelay e34 --config configs/e34_overlap.json --devices 1,2
```

`kr_plan_run(schedule_json, iters, warmup, devices, mode, out_json)` 直接执行 planner/export 生成的 schedule。

## 目录

| 路径 | 说明 |
|---|---|
| `include/kernrelay.h` | 稳定 C ABI |
| `exec/` | `nhp_exec` 通用执行器、Qwen 运行时、C ABI 实现 |
| `kernels/` | GM flag publish/wait Ascend C kernel（relay/PP 与同步微基准使用）与构建 |
| `python/kernrelay/` | planner、DAG/调度、优化流水线、模型适配器、libapi |
| `scripts/` | 构建、导出、矩阵/TP/planner 脚本 |
| `repro/` | 每个实验的可复现 `.sh` |
| `dags/` | 1:1、1:N、自设 DAG、planner 问题 DAG |
| `configs/` | 代价表、曲线、实验配置 |
| `results/final/` | 最终实验的汇总 JSON/Markdown（原始日志不入库） |
| `DESIGN.md` | 最终设计 |
| `RESULTS.md` | 最终实验结果 |

## 复现实验

```bash
# 单卡：serial / 融合 / ACL Graph
repro/qwen_single.sh --qwen-dir /data/kr_cache/qwen2.5-7b --device 1

# 双卡 TP-MLP 的 k=1 vs 切分 A/B
repro/qwen_tp.sh --qwen-dir /data/kr_cache/qwen2.5-7b --devices 1,2 --tiles 4

# 1:1 / 1:N / 计算-通信 / 自设 DAG
repro/e34_overlap.sh --devices 4,5
repro/e5_comm.sh --devices 4,5
repro/e9_dag.sh --devices 4,5

# planner 鲁棒性（只判拒绝，不记耗时）
python scripts/planner_robustness.py --out results/final/planner_robustness

# 全部
repro/all.sh --qwen-dir /data/kr_cache/qwen2.5-7b --device 1 --devices 1,2
```

## 边界

- **PP=2 不是 KernRelay 的切分收益**：同 DAG 单流对照更快；PP 只作为 device-parallel baseline。KernRelay 的归因用 `tp2` vs `tp2_split`（同设备数、同权重切分、同通信字节数，只改调度）。
- **TP-MLP 只切 MLP**：attention 复制；0.5B/4B 上 AllReduce 固定开销相对层计算过大，TP 与切分都是负收益；7B/8B 有正收益。
- **30B-A3B 是 MoE**：0.3 不实现 grouped matmul / routing，只做 DAG/算子统计并明确 SKIP，不用 dense 近似冒充。
- 整图多流暂不支持；每流单独 Graph 支持（`NHP_GRAPH_PER_STREAM`，跨流依赖用 capture-progress event），GM flag 空 kernel 也可逐流捕获。C++ 可直接捕获 ACLNN FlashAttentionScore V1/BSND；Python v1 包装报 107025。
- SuperKernel（同 DAG、TBE 不融合）与 autofuse（`ERR03005`）在目标环境不可用，记 SKIP；HF eager 仅作参考，不作为强基线。

## 结果

最终数字、协议与归因见 [RESULTS.md](RESULTS.md)。设计细节与边界见 [DESIGN.md](DESIGN.md)。
