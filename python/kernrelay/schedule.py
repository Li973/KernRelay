"""DAG loading, fixed schedule materialization, and a structural validator."""

import copy
import json
from pathlib import Path

from .runtime import (KERNRELAY_ROOT, FIXED_DAG_DIR, DAGSpec, NodeSpec,
                     build_fixed_schedule, load_dag, load_fixed_dag, stream_of)

DAG_DIR = KERNRELAY_ROOT / "dags"
ENGINES = ("aicore", "aic", "aiv", "aicpu", "hccl")


def resolve_dag(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p.resolve()
    for base in (DAG_DIR, FIXED_DAG_DIR):
        cand = base / (name_or_path if name_or_path.endswith(".json")
                       else name_or_path + ".json")
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "找不到 DAG %r（已查 %s 与 %s）" % (name_or_path, DAG_DIR, FIXED_DAG_DIR))


def load_fixed(path) -> tuple:
    return load_fixed_dag(resolve_dag(str(path)))


def load_spec(name) -> DAGSpec:
    return load_dag(resolve_dag(str(name)))


def effective_chunking(chunking: dict) -> dict:
    return {k: int(v) for k, v in chunking.items() if int(v) > 1}


def build_ab(spec, chunking: dict, mode: str = "ab", nranks: int = 2) -> dict:
    known = {n.name for n in spec.nodes}
    unknown = set(chunking) - known
    if unknown:
        raise ValueError("%s: chunking 引用未知节点 %s"
                         % (spec.name, sorted(unknown)))
    bad = {k: v for k, v in chunking.items() if int(v) < 1}
    if bad:
        raise ValueError("%s: chunking 必须 >=1，非法项 %s" % (spec.name, bad))
    eff = effective_chunking(chunking)
    out = {}
    if mode in ("a", "ab"):
        out["a"] = build_fixed_schedule(spec, {}, nranks=nranks, method="A-nosplit")
    if mode in ("b", "ab"):
        out["b"] = build_fixed_schedule(spec, eff, nranks=nranks, method="B-split")
    return out


def standalone_schedule(node, chunk: int = 1, nranks: int = 1) -> dict:
    copy = NodeSpec(node.name, node.op, node.engine, dict(node.shape),
                    node.dtype, node.tiles, node.iters, node.blocks)
    spec = DAGSpec(name="%s_c%d" % (node.name, chunk),
                   description="单节点切片表征", nodes=[copy], edges=[])
    chunking = {node.name: int(chunk)} if int(chunk) > 1 else {}
    return build_fixed_schedule(spec, chunking, nranks=nranks,
                                method="e1-standalone")


def serial_schedule(spec, nranks: int = 2) -> dict:
    """串行基线：全部 tile 按全局拓扑序放进一条 aicore 流（同流 FIFO 串行）。"""
    base = build_fixed_schedule(spec, {}, nranks=nranks, method="serial")
    order = {"aicore": 0, "compute": 0, "aic": 1, "aiv": 2, "aicpu": 3,
             "hccl": 4}
    tasks, deps, rank = {}, {}, {}
    for engine, seq in base["streams"].items():
        for i, t in enumerate(seq):
            key = (t["node"], int(t.get("tile", 0)))
            tasks[key] = t
            rank[key] = (order.get(engine, 9), i)
    for engine, seq in base["streams"].items():
        for i, t in enumerate(seq):
            key = (t["node"], int(t.get("tile", 0)))
            d = {(w[0], int(w[1])) for w in (t.get("wait") or [])}
            if i:
                p = seq[i - 1]
                d.add((p["node"], int(p.get("tile", 0))))
            deps[key] = {x for x in d if x in tasks}
    done = set()
    ordered = []
    while len(ordered) < len(tasks):
        ready = [k for k in tasks if k not in done and deps[k] <= done]
        if not ready:
            raise RuntimeError("%s: 串行基线拓扑排序成环" % spec.name)
        key = min(ready, key=lambda k: rank[k])
        ordered.append(key)
        done.add(key)
    merged = []
    for key in ordered:
        t = dict(tasks[key])
        t["wait"] = []
        t["publish"] = False
        merged.append(t)
    out = dict(base)
    out["streams"] = {"aicore": merged}
    out["group"] = "serial"
    out["chunking"] = {}
    return out


def validate_schedule(sched: dict) -> list:
    """Lightweight structural checks; executor remains the authority."""
    errors = []
    nodes = sched.get("nodes") or []
    ids = [n.get("id") for n in nodes]
    if len(ids) != len(set(ids)):
        errors.append("节点 id 重复")
    known = set(ids)
    streams = sched.get("streams") or {}
    if not streams:
        errors.append("没有 stream")
    tasks = []
    seen = {}
    for engine, seq in streams.items():
        if engine not in ENGINES:
            errors.append("非法 stream %r" % engine)
        if not isinstance(seq, list):
            errors.append("stream %r 不是任务数组" % engine)
            continue
        for i, t in enumerate(seq):
            node = t.get("node")
            tile = int(t.get("tile", 0))
            key = (node, tile)
            if node not in known:
                errors.append("stream %s 任务引用未知节点 %r" % (engine, node))
            if key in seen:
                errors.append("节点 %s tile %d 在 stream %s/%s 重复"
                              % (node, tile, seen[key], engine))
            seen[key] = engine
            if int(t.get("in_rows", 1)) <= 0:
                errors.append("节点 %s tile %d 的 in_rows<=0" % (node, tile))
            tasks.append({"key": key, "engine": engine, "idx": i, "task": t})
    published = set()
    for seq in streams.values():
        for t in seq:
            if t.get("publish"):
                published.add((t.get("node"), int(t.get("tile", 0))))

    pos = {item["key"]: i for i, item in enumerate(tasks)}
    adj = {i: set() for i in range(len(tasks))}
    indeg = {i: 0 for i in range(len(tasks))}
    first_in_stream = {}
    for i, item in enumerate(tasks):
        engine = item["engine"]
        if engine in first_in_stream:
            prev = first_in_stream[engine]
            adj[prev].add(i)
            indeg[i] += 1
        first_in_stream[engine] = i
    for i, item in enumerate(tasks):
        for w in item["task"].get("wait") or []:
            wkey = (w[0], int(w[1]))
            if wkey not in pos:
                errors.append("节点 %s tile %d 等待不存在任务 %s#%s"
                              % (item["key"][0], item["key"][1], wkey[0], wkey[1]))
                continue
            if wkey not in published:
                errors.append("节点 %s tile %d 等待的任务 %s#%s 不 publish"
                              % (item["key"][0], item["key"][1], wkey[0], wkey[1]))
            j = pos[wkey]
            if i not in adj[j]:
                adj[j].add(i)
                indeg[i] += 1
    ready = [i for i, d in indeg.items() if d == 0]
    done = 0
    while ready:
        cur = ready.pop()
        done += 1
        for nxt in adj[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
    if done != len(tasks):
        errors.append("schedule 存在等待环")
    return errors


def invalid_variants(sched: dict) -> dict:
    """Deterministic mutations used by E0's rejection gate."""
    out = {}

    dup = copy.deepcopy(sched)
    dup["streams"]["compute"] = dup["streams"].pop(next(
        k for k in dup["streams"] if dup["streams"][k]))
    dup["streams"]["aicore"] = []
    out["duplicate_engine"] = (dup, "duplicate engine stream")

    ghost = copy.deepcopy(sched)
    first = next(k for k, v in ghost["streams"].items() if v)
    ghost["streams"][first].append({"node": "__ghost__", "tile": 0,
                                    "wait": [], "publish": False,
                                    "in_row_off": 0, "in_rows": 1})
    out["ghost_task"] = (ghost, "missing plan for task")

    cycle = copy.deepcopy(sched)
    streams = [k for k, v in cycle["streams"].items() if len(v) >= 1]
    if len(streams) < 2:
        return out
    s0, s1 = streams[0], streams[1]
    t0 = cycle["streams"][s0][0]
    t1 = cycle["streams"][s1][0]
    t0["publish"] = True
    t1["publish"] = True
    t0.setdefault("wait", []).append([t1["node"], int(t1.get("tile", 0))])
    t1.setdefault("wait", []).append([t0["node"], int(t0.get("tile", 0))])
    out["wait_cycle"] = (cycle, "wait cycle")

    over = copy.deepcopy(sched)
    for seq in over["streams"].values():
        for t in seq:
            if t.get("in_row_off", 0) > 0:
                t["in_rows"] = int(t["in_rows"]) + 10 ** 9
                out["row_overflow"] = (over, "exceed")
                return out
    return out


def dump_schedule(sched: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sched, indent=1, ensure_ascii=False),
                    encoding="utf-8")
