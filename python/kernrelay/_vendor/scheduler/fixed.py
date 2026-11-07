"""固定切分模式（绕过自动调度搜索）。

用户要求悬置 A/B 自动调度：在 example/fixed/ 下用**固定**的 1:1、1:N 切分
模式直接验证五流执行器（物理 engine 感知）的重叠能力。本模块只做机械物化：

* 节点物理流 = ``kernel/engines.stream_of(op)``（registry/engine_map.json，
  msprof 实测或自研算子编译期声明）：aicore(MIX) / aic / aiv / aicpu / hccl；
* 切分 = 输入行空间均分（``rows % chunks == 0``），输出行空间按 comm 语义
  缩放后均分；不搜索、不换序；
* 边依赖 = 消费者块与生产者输出块**行区间相交**（``in_row_off/in_rows``），
  同流依赖由流内 FIFO（节点拓扑序入流）保证，跨流依赖物化为 wait/publish
  （执行器用 ACL 事件实现）；
* ``registry/engine_map.json`` 标记的不稳定算子（未来外部写入）拒绝进入
  固定计划，避免同时间不稳定 kernel 共存。

DAG JSON 里可选顶层 ``"fixed": {"chunking": {"节点": 块数}}`` 声明该 DAG 的
固定切分方案；A 组（不切）一律 chunking=1，B 组按该方案。
"""

import json
from pathlib import Path

from kernel.engines import is_unstable, stream_of
from kernel.ops import get_op
from scheduler.chunks import node_rows, out_rows
from scheduler.export import node_dims
from scheduler.model import load_dag


def load_fixed_dag(path):
    """返回 (DAGSpec, 固定切分方案)。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    spec = load_dag(path)
    chunking = dict((raw.get("fixed") or {}).get("chunking") or {})
    unknown = set(chunking) - {n.name for n in spec.nodes}
    if unknown:
        raise ValueError("%s: fixed.chunking 引用未知节点 %s"
                         % (spec.name, sorted(unknown)))
    return spec, chunking


def _topo_order(spec):
    names = [n.name for n in spec.nodes]
    indeg = {n: 0 for n in names}
    succ = {n: [] for n in names}
    for e in spec.edges:
        indeg[e.dst] += 1
        succ[e.src].append(e.dst)
    ready = [n for n in names if indeg[n] == 0]
    order = []
    while ready:
        ready.sort(key=names.index)   # 稳定：按 DAG 声明顺序
        cur = ready.pop(0)
        order.append(cur)
        for v in succ[cur]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    if len(order) != len(names):
        raise ValueError("%s: 依赖成环" % spec.name)
    return order


def build_fixed_schedule(spec, chunking=None, nranks: int = 2,
                         method: str = "fixed") -> dict:
    """DAG + 固定切分方案 -> schedule.json 内容（与 export.py 同形态）。"""
    chunking = {k: int(v) for k, v in dict(chunking or {}).items() if int(v) > 0}
    nm = spec.node_map
    for n in spec.nodes:
        get_op(n.op)
        if is_unstable(n.op):
            raise ValueError("%s: 节点 %s 的算子 %s 在 engine_map unstable 列表中，"
                             "禁止参与固定计划" % (spec.name, n.name, n.op))
    for e in spec.edges:
        if nm[e.src].op == "cast":
            raise ValueError("%s: cast 输出为 fp32，只能作 sink，不能被 %s 消费"
                             % (spec.name, e.dst))

    # 行空间几何：输入(send)行均分；输出行按 out_scale 缩放后均分
    geom = {}
    for n in spec.nodes:
        rows = node_rows(n)
        chunks = chunking.get(n.name, 1)
        if rows % chunks != 0:
            raise ValueError("%s: 节点 %s rows=%d 不能整除 chunks=%d"
                             % (spec.name, n.name, rows, chunks))
        per = rows // chunks
        own_out = out_rows(n, nranks)
        if own_out % chunks != 0:
            raise ValueError("%s: 节点 %s 输出行 %d 不能整除 chunks=%d"
                             % (spec.name, n.name, own_out, chunks))
        if n.op == "reducescatter" and per % nranks != 0:
            raise ValueError("%s: reducescatter %s 每块行数需整除 nranks=%d"
                             % (spec.name, n.name, nranks))
        geom[n.name] = (chunks, per, own_out // chunks)

    tasks = {}
    for n in spec.nodes:
        chunks, per, out_per = geom[n.name]
        for i in range(chunks):
            tasks[(n.name, i)] = {
                "node": n.name, "tile": i,
                "wait": [], "publish": False,
                "in_row_off": i * per, "in_rows": per,
                "out_row_off": i * out_per,
            }

    published = set()
    for e in spec.edges:
        src, dst = nm[e.src], nm[e.dst]
        src_chunks, _, src_out_per = geom[src.name]
        dst_chunks, dst_per, _ = geom[dst.name]
        if node_rows(dst) > out_rows(src, nranks):
            raise ValueError("%s: 边 %s->%s 消费者输入行 %d 超过生产者输出行 %d"
                             % (spec.name, e.src, e.dst, node_rows(dst),
                                out_rows(src, nranks)))
        cross_stream = stream_of(src.op) != stream_of(dst.op)
        src_out_rows = out_rows(src, nranks)
        for j in range(dst_chunks):
            lo, hi = j * dst_per, j * dst_per + dst_per
            hits = [k for k in range(src_chunks)
                    if k * src_out_per < hi and (k + 1) * src_out_per > lo]
            if not hits:   # 行区间不重叠（消费者只读部分生产者输出）：保守取末块
                hits = [src_chunks - 1]
            for k in hits:
                if k * src_out_per >= src_out_rows:
                    continue
                if cross_stream:
                    tasks[(dst.name, j)]["wait"].append([src.name, k])
                    published.add((src.name, k))
    for key in published:
        tasks[key]["publish"] = True

    streams = {}
    for name in _topo_order(spec):
        node = nm[name]
        s = stream_of(node.op)
        streams.setdefault(s, [])
        for i in range(geom[name][0]):
            streams[s].append(tasks[(name, i)])

    nodes = []
    for n in spec.nodes:
        entry = {"id": n.name, "op": n.op, "engine": stream_of(n.op),
                 "dtype": n.dtype, "tiles": geom[n.name][0],
                 "split": geom[n.name][0] > 1,
                 "preds": [e.src for e in spec.edges if e.dst == n.name],
                 "iters": int(getattr(n, "iters", 1)),
                 "blocks": int(getattr(n, "blocks", 1))}
        entry.update(node_dims(n))
        nodes.append(entry)

    return {"dag": spec.name, "group": method, "nranks": int(nranks),
            "nodes": nodes, "streams": {k: v for k, v in streams.items() if v},
            "chunking": {k: v for k, v in chunking.items() if v > 1},
            "fallback": ""}
