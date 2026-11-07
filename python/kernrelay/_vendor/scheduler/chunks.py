"""分块流水图与 lane 级列表调度（NeutronHP scheduler 层，策略 v2）。

## 为什么重写

旧 S1 模型（``model.build_constraints``）有三个结构性缺陷，直接导致"模型预测
1.6x、真机 ≤1.0x"：

1. **强制 1:1 链式对齐**：只有 ``len(src_tiles) == len(dst_tiles)`` 且 tile i 对 i
   才算 pipe。producer 粒度粗、consumer 粒度细（或反之）时无法表达——而这恰恰是
   填流水气泡的主要手段；
2. **kernel 不交错**：同引擎相邻 kernel 之间硬加 ``last_tile(a) -> first_tile(b)``，
   一条 lane 上任何时刻只有一个 kernel 在跑，气泡无法用"下一个 kernel 的 tile"填；
3. **引擎 = 资源**：把 aic / aiv / aicpu 当三条独立资源，而执行器把三者合并成一条
   计算 lane（doc/design.md §8），于是模型白拿"异构重叠"收益。

## v2 模型

* 每个节点可选 **执行分块数** ``chunking[n] ∈ {1, ..., n.tiles}``（整数分块）；
* 每条数据边可选 ``serial``（consumer 等 producer 全部完成）或 ``pipe``
  （consumer 第 j 块只等**它真正读取的行区间**所覆盖的 producer 块）；
  行区间由 ``in_row_off/in_rows`` 显式给出，因此天然支持 1:1 / 1:k / k:1 / m:n；
* 依赖图 + **lane 上 FIFO**（compute 一条、hccl 一条）做关键路径列表调度，
  允许同 lane 上不同 kernel 的块交错执行（气泡填充）；
* 只有**跨 lane** 的依赖才付 ``o``（event record + wait + 额外入队）——同 lane
  由 FIFO 顺序保证，不付钱；
* 分块时长取保守值（``CostModel.tile_duration``），波动算子禁止分块。

分块几何（行空间约定，与 runtime/op_impl.cpp 一致）：
节点声明的 ``rows`` 是**输入(send)行数**；输出行数 = ``rows * out_scale``，
allgather 的 out_scale = nranks、reducescatter = 1/nranks、其余 = 1。
``in_row_off/in_rows`` 位于 producer 输出行空间（也就是本节点的输入行空间），
``out_row_off/out_rows`` 位于本节点自己的输出行空间。
"""

import itertools
from dataclasses import dataclass, field

from scheduler.lanes import LANES, lane_of
from scheduler.model import PIPE, SERIAL


# ---------------- 行空间几何 ----------------

def node_rows(node) -> int:
    """节点声明的输入(send)行数。"""
    s = node.shape
    return int(s["M"]) if node.op == "matmul" else int(s["rows"])


def out_scale(node, nranks: int) -> float:
    if node.op == "allgather":
        return float(nranks)
    if node.op == "reducescatter":
        return 1.0 / float(nranks)
    return 1.0


def out_rows(node, nranks: int) -> int:
    return int(round(node_rows(node) * out_scale(node, nranks)))


@dataclass
class ChunkTask:
    node: str
    idx: int
    engine: str
    lane: str
    dur: float
    in_row_off: int
    in_rows: int
    out_row_off: int
    out_rows: int
    deps: list = field(default_factory=list)        # [(node, idx), ...]
    cross_deps: list = field(default_factory=list)  # deps 中跨 lane 者
    start: float = 0.0
    finish: float = 0.0
    publish: bool = False

    @property
    def key(self):
        return (self.node, self.idx)


@dataclass
class ChunkPlan:
    spec: object
    tasks: list
    chunking: dict
    modes: dict
    nranks: int
    makespan: float
    lane_tasks: dict                 # lane -> [ChunkTask]（按开始时间）
    lane_busy: dict                  # lane -> 累计忙碌时长
    lane_bubbles: dict               # lane -> 累计空闲
    feasible: bool = True
    note: str = ""

    @property
    def by_key(self):
        return {t.key: t for t in self.tasks}

    @property
    def utilization(self) -> dict:
        return {k: (self.lane_busy[k] / self.makespan if self.makespan > 0 else 0.0)
                for k in self.lane_tasks}

    def engine_orders(self):
        """模型摘要：每引擎按其任务开始时间列出 kernel（同 kernel 只出现一次）。"""
        out = {}
        for t in sorted(self.tasks, key=lambda x: (x.start, x.node, x.idx)):
            seq = out.setdefault(t.engine, [])
            if t.node not in seq:
                seq.append(t.node)
        return out


# ---------------- 构图 ----------------

def _chunk_geom(spec, chunking, nranks):
    """node -> (chunks, 每块输入行数, 输出缩放)。"""
    nm = spec.node_map
    geom = {}
    for name, node in nm.items():
        rows = node_rows(node)
        chunks = int(chunking.get(name, 1))
        if chunks < 1:
            raise ValueError("%s: %s chunking=%d 非法" % (spec.name, name, chunks))
        if rows % chunks != 0:
            raise ValueError("%s: %s rows=%d 不能整除 chunking=%d"
                             % (spec.name, name, rows, chunks))
        per = rows // chunks
        scale = out_scale(node, nranks)
        if scale < 1.0 and per % nranks != 0:
            raise ValueError("%s: %s 每块输入行 %d 不能整除 nranks=%d（reducescatter 输出）"
                             % (spec.name, name, per, nranks))
        geom[name] = (chunks, per, scale)
    return geom


def build_chunk_tasks(spec, costs, chunking: dict, modes: dict, nranks: int = 2):
    """(spec, 分块数, 边模式) -> (任务列表, (node, idx) -> 任务)。"""
    nm = spec.node_map
    geom = _chunk_geom(spec, chunking, nranks)
    tasks, index = [], {}
    for node in spec.nodes:
        chunks, per, scale = geom[node.name]
        dur_base = (costs.duration(node) if chunks == 1
                    else costs.chunk_duration(node, per))
        for i in range(chunks):
            in_off = i * per
            t = ChunkTask(node=node.name, idx=i, engine=node.engine,
                          lane=lane_of(node.engine), dur=dur_base,
                          in_row_off=in_off, in_rows=per,
                          out_row_off=int(round(in_off * scale)),
                          out_rows=int(round(per * scale)))
            index[(node.name, i)] = t
            tasks.append(t)

    for e in spec.edges:
        u_chunks, u_per, u_scale = geom[e.src]
        u_out_rows = int(round(u_per * u_scale))
        mode = modes.get(e.key, SERIAL)
        for j in range(geom[e.dst][0]):
            t = index[(e.dst, j)]
            if u_chunks == 1:
                dep = [(e.src, 0)]
            elif mode == PIPE:
                lo, hi = t.in_row_off, t.in_row_off + t.in_rows
                dep = [(e.src, k) for k in range(u_chunks)
                       if k * u_out_rows < hi and (k + 1) * u_out_rows > lo]
                if not dep:      # 行区间不重叠：保守退化为等 producer 全部完成
                    dep = [(e.src, u_chunks - 1)]
            else:
                dep = [(e.src, u_chunks - 1)]
            for d in dep:
                if d not in t.deps:
                    t.deps.append(d)

    # 跨 lane 依赖付 gate/event 开销，并把 producer 标记为需要 publish
    for t in tasks:
        cross = []
        for d in t.deps:
            p = index[d]
            if p.lane != t.lane:
                p.publish = True
                cross.append(d)
        t.cross_deps = cross
        if cross:
            t.dur += costs.tile_overhead(t.engine) * len(cross)
    return tasks, index


# ---------------- 调度 ----------------

def _bottom_level(tasks):
    """bottom-level：自身时长 + 到 sink 的最长路径（关键路径优先级）。"""
    succ = {t.key: [] for t in tasks}
    for t in tasks:
        for d in t.deps:
            succ[d].append(t.key)
    bl = {t.key: t.dur for t in tasks}
    changed = True
    while changed:
        changed = False
        for t in tasks:
            v = t.dur + max((bl[s] for s in succ[t.key]), default=0.0)
            if v > bl[t.key] + 1e-9:
                bl[t.key] = v
                changed = True
    return bl


def list_schedule(tasks, index=None):
    """lane 级非延迟列表调度：每次在全图就绪任务里选最早可开始者（关键路径破平）。"""
    index = index or {t.key: t for t in tasks}
    bl = _bottom_level(tasks)
    pending = {t.key: set(t.deps) for t in tasks}
    lane_free = {lane: 0.0 for lane in LANES}
    ready = [t for t in tasks if not pending[t.key]]
    done = 0
    while ready:
        best, best_key = None, None
        for t in ready:
            est = max(lane_free[t.lane],
                      max((index[d].finish for d in t.deps), default=0.0))
            key = (round(est, 6), -bl[t.key], t.node, t.idx)
            if best_key is None or key < best_key:
                best, best_key = t, key
        t = best
        t.start = best_key[0]
        t.finish = t.start + t.dur
        lane_free[t.lane] = t.finish
        ready.remove(t)
        done += 1
        for other in tasks:
            if t.key in pending[other.key]:
                pending[other.key].discard(t.key)
                if not pending[other.key]:
                    ready.append(other)
    if done != len(tasks):
        raise RuntimeError("分块依赖图成环（%d/%d）" % (done, len(tasks)))
    return tasks


def _assemble(spec, tasks, chunking, modes, nranks):
    lane_tasks = {lane: [] for lane in LANES}
    for t in sorted(tasks, key=lambda x: (x.start, x.lane, x.node, x.idx)):
        lane_tasks.setdefault(t.lane, []).append(t)
    lane_busy, lane_bubbles = {}, {}
    for lane, seq in lane_tasks.items():
        busy = sum(t.dur for t in seq)
        span = max((t.finish for t in seq), default=0.0)
        lane_busy[lane] = busy
        lane_bubbles[lane] = max(0.0, span - busy)
    makespan = max((t.finish for t in tasks), default=0.0)
    return ChunkPlan(spec=spec, tasks=tasks, chunking=dict(chunking),
                     modes=dict(modes), nranks=nranks, makespan=makespan,
                     lane_tasks={k: v for k, v in lane_tasks.items() if v},
                     lane_busy=lane_busy, lane_bubbles=lane_bubbles)


def plan_chunks(spec, costs, chunking: dict, modes: dict, nranks: int = 2) -> ChunkPlan:
    """完整流程：构图 -> 列表调度 -> 结果。"""
    tasks, index = build_chunk_tasks(spec, costs, chunking, modes, nranks)
    list_schedule(tasks, index)
    return _assemble(spec, tasks, chunking, modes, nranks)


# ---------------- 精确枚举（A 用：整 kernel，无分块） ----------------

def lane_order_cap(spec) -> int:
    by_lane = {}
    for n in spec.nodes:
        by_lane.setdefault(lane_of(n.engine), []).append(n.name)
    total = 1
    for names in by_lane.values():
        f = 1
        for i in range(2, len(names) + 1):
            f *= i
        total *= f
    return total


def iter_lane_orders(spec):
    """枚举 (lane -> 节点全序)，过滤违反同 lane 数据优先的排列。"""
    by_lane = {}
    for n in spec.nodes:
        by_lane.setdefault(lane_of(n.engine), []).append(n.name)
    prec = [(e.src, e.dst) for e in spec.edges
            if lane_of(spec.node_map[e.src].engine) == lane_of(spec.node_map[e.dst].engine)]

    def ok(order):
        pos = {name: i for i, name in enumerate(order)}
        return all(pos[u] < pos[v] for u, v in prec if u in pos and v in pos)

    per_lane = []
    for lane in LANES:
        names = by_lane.get(lane, [])
        perms = ([list(p) for p in itertools.permutations(names)]
                 if len(names) > 1 else [list(names)])
        per_lane.append((lane, [p for p in perms if ok(p)]))
    for combo in itertools.product(*[p for _, p in per_lane]):
        yield {lane: order for (lane, _), order in zip(per_lane, combo)}


def plan_from_orders(spec, costs, orders: dict, nranks: int = 2):
    """给定 lane 全序，按 ASAP 求值（整 kernel，无分块）。

    两条 lane 的 FIFO 顺序固定，按依赖就绪逐轮推进（可能交错）。若某条 lane 的
    首任务依赖另一条 lane 上排在后面的任务（跨 lane 顺序环），该全序不可行 ——
    返回 ``None``（与旧模型"约束成环 = 死锁，不可行"语义一致）。
    """
    chunking = {n.name: 1 for n in spec.nodes}
    modes = {e.key: SERIAL for e in spec.edges}
    tasks, index = build_chunk_tasks(spec, costs, chunking, modes, nranks)
    by_node = {t.node: t for t in tasks}
    lane_free = {lane: 0.0 for lane in LANES}
    cursor = {lane: 0 for lane in LANES}
    placed = set()
    total = len(tasks)
    while len(placed) < total:
        progressed = False
        for lane in LANES:
            seq = orders.get(lane, [])
            if cursor[lane] >= len(seq):
                continue
            t = by_node[seq[cursor[lane]]]
            if t.key in placed or any(d not in placed for d in t.deps):
                continue
            t.start = max(lane_free[lane],
                          max((index[d].finish for d in t.deps), default=0.0))
            t.finish = t.start + t.dur
            lane_free[lane] = t.finish
            placed.add(t.key)
            cursor[lane] += 1
            progressed = True
        if not progressed:
            return None
    return _assemble(spec, tasks, chunking, modes, nranks)




