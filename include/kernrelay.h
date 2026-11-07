#ifndef KERNRELAY_H
#define KERNRELAY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 版本字符串；调用方不需要释放。 */
const char *kr_version(void);

/*
 * 运行一份 KernRelay schedule.json。
 *   schedule_json : planner/export 生成的调度文件
 *   iters/warmup  : 计时轮数与预热轮数
 *   devices       : "4,5" 或 NULL（默认 0..nranks-1）
 *   mode          : "direct" | "graph" | "graph_per_stream" | "graph_segment"
 *   out_json      : 结果 JSON 写入路径
 * 返回 0 成功；非 0 失败（错误打印到 stderr）。
 *
 * 说明：这是进程级入口，每次调用会重新初始化 ACL/HCCL；适合一个外部
 * 进程跑一份 plan。多次调用请使用独立进程，避免 CANN 9.0 的 executor
 * 缓存与 HCCL 生命周期问题。
 */
int kr_plan_run(const char *schedule_json, int iters, int warmup,
                const char *devices, const char *mode, const char *out_json);

/*
 * 运行真实 Qwen2.5 prefill（单卡）。mode:
 *   serial | serial_fused | serial_graph | graph_whole |
 *   relay_event | relay_flag | relay_flag_graph
 * out/out_cap 接收结果 JSON（含 best/typical/argmax/maxdiff/host 等）。
 * 返回 0 成功；失败时 out 内为 {"error":...}。
 */
int kr_qwen_run(const char *model_dir, const char *mode, int seq, int device,
                int warmup, int iters, char *out, size_t out_cap);

/*
 * 运行双卡 PP=2 的真实 Qwen2.5 prefill（HCCL Send/Recv 传 hidden）。
 * mode: "pp"（两个请求流水重叠）| "pp_serial"（逐请求串行，无重叠基线）。
 * dev0/dev1 是 stage0/stage1 设备号；requests 目前支持 1..2。
 * 返回 JSON 含 wall_best/wall_typical、每 stage 的 samples 与末尾 argmax。
 */
int kr_qwen_pp_run(const char *model_dir, const char *mode, int seq, int dev0,
                   int dev1, int requests, int warmup, int iters, char *out,
                   size_t out_cap);

/*
 * 运行双卡 TP-MLP 的真实 Qwen2.5 prefill。
 * attention 复制，gate/up 按输出列、down 按 K 行各切一半，每层对 down
 * 部分和做一次 HCCL AllReduce。mode:
 *   "tp2"       整幅 down -> 一次 AllReduce -> AddNorm（无重叠基线）
 *   "tp2_split" down 按 tiles 行切，AllReduce/AddNorm 逐 tile 放行
 * tiles 建议 2；两台设备同进程，一次调用跑完。
 */
int kr_qwen_tp_run(const char *model_dir, const char *mode, int seq, int dev0,
                   int dev1, int tiles, int warmup, int iters, char *out,
                   size_t out_cap);

/*
 * 用最终 Qwen 运行时对任务算子做 k 粒度预跑；spec_json 形如
 * {"cases":[{"op":"matmul_gate","k":1},{"op":"matmul_gate","k":2}]}。
 * op: matmul_q/o/k/v/gate/up/down/lm, add, rmsnorm, add_rmsnorm, rope,
 *     silu, mul, embedding, attention。
 * 一次加载权重，批量测所有 case；结果 JSON 的 cases 数组含 best/median。
 */
int kr_qwen_profile_many(const char *model_dir, const char *spec_json, int seq,
                         int device, char *out, size_t out_cap);

#ifdef __cplusplus
}
#endif

#endif /* KERNRELAY_H */
