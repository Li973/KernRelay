# planner 鲁棒性（只判拒绝，不记耗时）

| DAG | 期望 | 结果 | 原因 |
|---|---|---|---|
| e0_tiny_mm_gelu | accept | ACCEPT |  |
| e3_ln_mm | accept | ACCEPT |  |
| e9_mixed | accept | ACCEPT |  |
| bad_cycle | reject | REJECT | 依赖成环: graph cycle |
| bad_unknown_op | reject | REJECT | mm: 不支持算子 'frobnicate' |
| bad_shape | reject | REJECT | mm: matmul M<=0 |
| bad_edge | reject | REJECT | edge dst 不存在: 'missing' |
| bad_dup_name | reject | REJECT | 节点名重复 |
| bad_attention | reject | REJECT | attn: attention q_heads=14 不是 kv_heads=3 的整数倍 |
| aicpu_producer | accept(0-split) | ACCEPT |  |
| tiny_matmul | accept(0-split) | ACCEPT |  |
