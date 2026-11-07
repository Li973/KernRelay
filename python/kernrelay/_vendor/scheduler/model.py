"""DAG 任务模型（NeutronHP scheduler 层）。

DAG JSON 格式（example/**/*.json）::

    {
      "name": "p01_gemm_allreduce",
      "description": "...",
      "nodes": [
        {"name": "gemm", "op": "matmul", "engine": "aic",
         "shape": {"M": 2048, "K": 4096, "N": 2048}, "dtype": "fp16", "tiles": 4}
      ],
      "edges": [ {"src": "gemm", "dst": "ar", "kind": "data"} ]
    }

时间输入一律经 registry 耗时表（含波动度）；tile 数为 kernel 固有切分。

**注意**：本模块的 ``build_tasks`` / ``build_constraints`` 是 S1 语义（1:1 链式
pipe + 同引擎 kernel 不交错 + 引擎=资源）的历史实现，仅供 ``build/`` 下的历史
分析脚本使用。当前调度策略 v2 见 ``scheduler/chunks.py`` / ``cost_model.py``
与 doc/scheduler_v2.md。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from kernel.engines import ALL_ENGINES
from kernel.ops import get_op

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = ROOT / "example"

SERIAL = "serial"
PIPE = "pipe"


@dataclass
class NodeSpec:
    name: str
    op: str
    engine: str
    shape: dict
    dtype: str
    tiles: int
    iters: int = 1      # 自研算子重复计算次数（控制时长）
    blocks: int = 1     # 自研算子 AIC/AIV 启动块数


@dataclass
class EdgeSpec:
    src: str
    dst: str
    kind: str = "data"

    @property
    def key(self):
        return (self.src, self.dst)


@dataclass
class DAGSpec:
    name: str
    description: str
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    source: str = ""

    @property
    def node_map(self):
        return {n.name: n for n in self.nodes}

    def engines(self):
        out = []
        for n in self.nodes:
            if n.engine not in out:
                out.append(n.engine)
        return out

    def validate(self):
        names = [n.name for n in self.nodes]
        if len(names) != len(set(names)):
            raise ValueError("%s: 节点名重复" % self.name)
        nm = self.node_map
        for n in self.nodes:
            if n.engine not in ALL_ENGINES:
                raise ValueError("%s: 节点 %s 引擎非法 %s" % (self.name, n.name, n.engine))
            if n.tiles < 1:
                raise ValueError("%s: 节点 %s tiles 必须 >= 1" % (self.name, n.name))
            get_op(n.op).validate_shape(n.shape)   # 算子登记与 shape 键校验
        if len(self.edges) != len({e.key for e in self.edges}):
            raise ValueError("%s: 依赖边重复" % self.name)
        for e in self.edges:
            if e.src not in nm or e.dst not in nm:
                raise ValueError("%s: 边 %s->%s 引用未知节点" % (self.name, e.src, e.dst))
            if e.src == e.dst:
                raise ValueError("%s: 自环边 %s" % (self.name, e.src))
        return self


# ---------------- 装载 ----------------

def load_dag(path) -> DAGSpec:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    nodes = [NodeSpec(n["name"], n["op"], n["engine"], dict(n["shape"]),
                      n["dtype"], int(n.get("tiles", get_op(n["op"]).default_tiles)),
                      int(n.get("iters", 1)), int(n.get("blocks", 1)))
             for n in raw["nodes"]]
    edges = [EdgeSpec(e["src"], e["dst"], e.get("kind", "data")) for e in raw["edges"]]
    return DAGSpec(raw["name"], raw.get("description", ""), nodes, edges,
                   source=str(path)).validate()


def iter_example_paths(suite: str = "all", example_dir=None) -> list:
    base = Path(example_dir) if example_dir else EXAMPLE_DIR
    dirs = {"pairwise": [base / "pairwise"], "complex": [base / "complex"],
            "all": [base / "pairwise", base / "complex"]}[suite]
    return [p for d in dirs for p in sorted(d.glob("*.json"))]


def load_suite(suite: str = "all", example_dir=None) -> list:
    return [load_dag(p) for p in iter_example_paths(suite, example_dir)]


# ---------------- 任务展开 ----------------

@dataclass
class Task:
    id: int
    node: str
    tile: int
    engine: str
    dur: float            # = tile_kernel_time + 消费侧 o（无入边则无 o）


@dataclass
class TaskGraph:
    spec: DAGSpec
    tasks: list
    node_task_ids: dict      # node name -> [task id, ...]（tile 顺序）
    node_total_us: dict      # node name -> 整 kernel 耗时（查表值，不含 o）

    @property
    def durations(self):
        return [t.dur for t in self.tasks]

    def first_tile(self, node):
        return self.node_task_ids[node][0]

    def last_tile(self, node):
        return self.node_task_ids[node][-1]


def build_tasks(spec: DAGSpec, cost_table) -> TaskGraph:
    """tile 展开：每个 (node, tile) 一个任务。

    任务时长 = 实测单 tile 调用耗时 + 消费侧 o（无入边则无 o）；
    node_total_us = 实测全量单次调用耗时（serial 口径）。
    """
    nm = spec.node_map
    has_incoming = {e.dst for e in spec.edges}
    tasks, node_ids, totals = [], {}, {}
    tid = 0
    for node in spec.nodes:
        entry = cost_table.lookup(node.op, node.shape, node.dtype, node.tiles)
        totals[node.name] = entry["full_us"]
        o = cost_table.tile_overhead(node.engine) if node.name in has_incoming else 0.0
        tile_dur = entry["tile_us"]
        node_ids[node.name] = []
        for i in range(node.tiles):
            tasks.append(Task(tid, node.name, i, node.engine, tile_dur + o))
            node_ids[node.name].append(tid)
            tid += 1
    return TaskGraph(spec, tasks, node_ids, totals)


# ---------------- 约束构造 ----------------

def build_constraints(task_graph: TaskGraph, decisions: dict, engine_orders: dict) -> list:
    """把 (切分决策, 引擎全序) 翻译为任务级优先约束 (u, v)：v 必须等 u 完成。"""
    tg = task_graph
    cons = []
    # kernel 内 tile 顺序（同引擎串行执行 tile）
    for ids in tg.node_task_ids.values():
        cons.extend(zip(ids[:-1], ids[1:]))
    # 数据依赖边
    for e in tg.spec.edges:
        src_ids, dst_ids = tg.node_task_ids[e.src], tg.node_task_ids[e.dst]
        mode = decisions.get(e.key, SERIAL)
        if mode == PIPE:
            if len(src_ids) != len(dst_ids):
                raise ValueError("pipe 边 %s->%s 要求 tile 数一致" % e.key)
            cons.extend(zip(src_ids, dst_ids))
        else:
            cons.append((src_ids[-1], dst_ids[0]))
    # 引擎全序：相邻 kernel 首尾相接（tile 不跨 kernel 交错，S1 语义）
    for engine, order in engine_orders.items():
        for a, b in zip(order[:-1], order[1:]):
            cons.append((tg.last_tile(a), tg.first_tile(b)))
    return cons
