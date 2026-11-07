"""调度结果物化（NeutronHP scheduler 层 -> 真实执行）。

导出只以 ``schedule.plan`` 为准（单一事实源），不再"模型一套、导出另一套"。

## 流结构

执行器把 aic/aiv/aicpu 合并为一条计算 lane（exec_main.cpp NormalizeTopology），
所以导出直接把计算块写进 ``aic`` 流、通信块写进 ``hccl`` 流：**导出顺序 = 模型
顺序**，不会因为执行器的全序投影而改变（以前四流导出 + C++ 重排会让模型时间线
和执行时间线脱节）。

## 每块几何（修复旧版 stride 缺陷）

旧版用 ``rowOff = t * RowsPerTile(consumer)`` 去索引 **producer** 的缓冲区：

* 1:1 且两侧 tiles 相同时"碰巧"正确（p02 的 checksum 因此没暴露问题）；
* m:n / 粒度不同 / reducescatter / allgather 时越界或错位（507015/507035 的
  一类触发面）。

v2 每个任务显式给出三段行区间：

    in_row_off, in_rows   # 输入(发送)行区间，位于 producer 输出行空间
    out_row_off           # 本块输出行区间起点（输出行数由 op 语义决定）

执行器按这三个字段取指针（缺省时回退旧 stride，兼容历史 schedule.json）。
"""

from kernel.ops import get_op
from scheduler.chunks import node_rows, out_rows
from scheduler.lanes import lane_of


def node_dims(node) -> dict:
    """DAG shape -> 执行器字段（rows/cols/k/n/dim0）。"""
    s = node.shape
    if node.op == "matmul":
        return {"rows": int(s["M"]), "cols": int(s["N"]),
                "k": int(s["K"]), "n": int(s["N"])}
    if node.op == "topk":
        return {"rows": int(s["rows"]), "cols": int(s["cols"]), "k": int(s["k"])}
    if node.op == "embedding":
        return {"rows": int(s["rows"]), "cols": int(s["cols"]), "dim0": int(s["vocab"])}
    return {"rows": int(s["rows"]), "cols": int(s["cols"])}


# 计算三引擎合并成一条 lane 后，导出流名固定为 aic / hccl
STREAM_OF_LANE = {"compute": "aic", "hccl": "hccl"}


def validate_executable(spec, chunking: dict, nranks: int = 2) -> None:
    """真实执行可行性：行整除、通信整除、cast 仅作 sink。"""
    nm = spec.node_map
    for e in spec.edges:
        if nm[e.src].op == "cast":
            raise ValueError("%s: cast 输出为 fp32，只能作 sink，不能被 %s 消费"
                             % (spec.name, e.dst))
    for n in spec.nodes:
        d = node_dims(n)
        chunks = int(chunking.get(n.name, 1))
        if d["rows"] % chunks != 0:
            raise ValueError("%s: 节点 %s rows=%d 不能整除 chunks=%d"
                             % (spec.name, n.name, d["rows"], chunks))
        if n.op == "reducescatter" and (d["rows"] // chunks) % nranks != 0:
            raise ValueError("%s: reducescatter %s 每块行数需整除 nranks=%d"
                             % (spec.name, n.name, nranks))
        get_op(n.op)


def validate_plan(spec, plan, nranks: int = 2) -> None:
    """几何与顺序自洽性检查（导出前硬门禁，替代真机上才会暴露的越界/错位）。"""
    nm = spec.node_map
    index = plan.by_key
    for t in plan.tasks:
        node = nm[t.node]
        own = out_rows(node, nranks)
        if t.out_row_off < 0 or t.out_row_off + t.out_rows > own:
            raise ValueError("%s: %s#%d 输出区间 [%d,%d) 越界（输出 %d 行）"
                             % (spec.name, t.node, t.idx, t.out_row_off,
                                t.out_row_off + t.out_rows, own))
        if t.in_row_off < 0 or t.in_row_off + t.in_rows > node_rows(node):
            raise ValueError("%s: %s#%d 输入区间 [%d,%d) 越界（输入 %d 行）"
                             % (spec.name, t.node, t.idx, t.in_row_off,
                                t.in_row_off + t.in_rows, node_rows(node)))
        for d in t.deps:
            p = index[d]
            if p.lane == t.lane and p.start > t.start + 1e-9:
                raise ValueError("%s: 同 lane 依赖 %s#%d 晚于消费者 %s#%d 开始"
                                 % (spec.name, p.node, p.idx, t.node, t.idx))
    for lane, seq in plan.lane_tasks.items():
        prev_finish = -1.0
        for t in seq:
            if t.start + 1e-9 < prev_finish:
                raise ValueError("%s: lane %s 上任务重叠（%s#%d）"
                                 % (spec.name, lane, t.node, t.idx))
            prev_finish = t.finish


def export_schedule(spec, schedule, nranks: int = 2) -> dict:
    """调度结果 -> schedule.json 内容（dict）。"""
    plan = getattr(schedule, "plan", None)
    if plan is None:
        raise ValueError("%s: schedule 缺少 plan（v2 调度结果才能导出）" % spec.name)
    if int(plan.nranks) != int(nranks):
        raise ValueError("%s: plan 用 nranks=%d 求解，导出却要 nranks=%d"
                         % (spec.name, plan.nranks, nranks))
    chunking = dict(plan.chunking)
    validate_executable(spec, chunking, nranks)
    validate_plan(spec, plan, nranks)

    streams = {}
    for lane, seq in plan.lane_tasks.items():
        name = STREAM_OF_LANE.get(lane, lane)
        tasks = []
        for t in seq:
            tasks.append({
                "node": t.node, "tile": t.idx,
                "wait": [[d[0], d[1]] for d in t.cross_deps],
                "publish": bool(t.publish),
                "in_row_off": int(t.in_row_off), "in_rows": int(t.in_rows),
                "out_row_off": int(t.out_row_off),
                "est_us": round(float(t.start), 1), "end_us": round(float(t.finish), 1),
            })
        streams[name] = tasks

    nodes = []
    for n in spec.nodes:
        entry = {"id": n.name, "op": n.op, "engine": n.engine, "dtype": n.dtype,
                 "tiles": int(chunking.get(n.name, 1)),
                 "split": int(chunking.get(n.name, 1)) > 1,
                 "preds": [e.src for e in spec.edges if e.dst == n.name]}
        entry.update(node_dims(n))
        nodes.append(entry)

    return {"dag": spec.name, "group": schedule.method, "nranks": int(nranks),
            "nodes": nodes, "streams": streams,
            "chunking": {k: v for k, v in chunking.items() if v > 1},
            "fallback": getattr(schedule, "fallback", "")}
