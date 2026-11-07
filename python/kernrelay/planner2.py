"""Planner-2: normal schedule A, bubble attribution, and split search.

The planner follows the rules requested for the new phase:

1. Schedule A: repeatedly take every node whose indegree is zero, enqueue it
   on its engine stream, evict it and update the remaining indegrees.  The
   simulator does not add a round barrier: a node starts as soon as its data
   dependencies are met and its engine is free.  Same-engine tasks are FIFO.
2. Record every engine-idle bubble and the critical predecessor edge.  Only
   cross-engine edges are split candidates.
3. Support 1:N splitting: producer and a chain of same-engine consumers are
   split with a common k, tile i of each stage depends on tile i of the
   previous stage.  Independent indegree-zero nodes may be interleaved between
   sub-consumers by the list scheduler.  When the consumer is much shorter
   than the producer, the planner also tries splitting the consumer only.
4. The split cost model is built from the measured E1/E8 tables; a split is
   accepted only when the simulated makespan improves.
"""

import argparse
import copy
import json
import re
from collections import defaultdict
from pathlib import Path

from .runtime import KERNRELAY_ROOT

ELEMENTWISE = {"gelu", "add", "silu", "mul", "cast", "softmax", "layernorm",
               "rmsnorm", "rope", "custom_aicpu_scale"}
COMM = {"allreduce", "allgather", "reducescatter"}


def shape_key(node):
    s = node["shape"]
    op = node["op"]
    if op == "matmul":
        return "matmul|%dx%dx%d" % (s["M"], s["K"], s["N"])
    if op == "attention":
        return "attention|%dx%dx%dx%d" % (s["batch"], s["q_heads"],
                                          s["seq"], s["head_dim"])
    return "%s|%dx%d" % (op, s.get("rows", 0), s.get("cols", 0))


def _shape_bytes(node):
    s = node["shape"]
    if node["op"] == "matmul":
        return 2 * (s["M"] * s["K"] + s["K"] * s["N"] + s["M"] * s["N"])
    if node["op"] == "attention":
        return 2 * (s["batch"] * s["q_heads"] * s["seq"] * s["head_dim"] * 3)
    return 4 * s.get("rows", 0) * s.get("cols", 0)


class Costs:
    """Base time and split tax.

    ``full`` is measured on the real executor (E1/E8).  ``curves`` is the E1
    fit table: for each operator/shape it stores the measured slope (extra us
    per extra tile) and the k=1 total, which is exactly the split tax the
    planner must pay.  Unseen shapes fall back to a roofline estimate.
    """

    def __init__(self, path, curves_path=None):
        self.data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.full = self.data.get("full", {})
        self.launch = float(self.data.get("launch_us", 8.0))
        self.bw = float(self.data.get("traffic_gbps", 600.0)) * 1e9
        self.l2 = float(self.data.get("l2_mb", 32.0)) * 1024 * 1024
        self.event = float(self.data.get("event_us", 20.0))
        self.curves = {}
        self._base_cache = {}
        self._tax_cache = {}
        if curves_path and Path(curves_path).exists():
            raw = json.loads(Path(curves_path).read_text(encoding="utf-8"))
            for case in raw.get("cases", []):
                key = self._curve_key(case)
                if key:
                    self.curves[key] = case

    @staticmethod
    def _curve_key(case):
        op = case.get("op")
        s = case.get("shape") or {}
        if op == "matmul":
            return "matmul|%dx%dx%d" % (s["M"], s["K"], s["N"])
        if op == "attention":
            return "attention|%dx%dx%dx%d" % (
                s["batch"], s["q_heads"], s["seq"], s["head_dim"])
        if s.get("rows") and s.get("cols"):
            return "%s|%dx%d" % (op, s["rows"], s["cols"])
        return None

    def base(self, node):
        key = shape_key(node)
        if key in self._base_cache:
            return self._base_cache[key]
        if key in self.full:
            val = float(self.full[key])
            self._base_cache[key] = val
            return val
        s = node["shape"]
        if node["op"] == "custom_aicpu_scale":
            # Measured scalar AICPU kernel: ~2.2 ns per element (E4/E2).
            return max(1.0, s.get("rows", 0) * s.get("cols", 0) * 2.2e-3)
        if node["op"] == "matmul":
            for k, v in self.full.items():
                if not k.startswith("matmul|"):
                    continue
                mm = re.match(r"matmul\|(\d+)x(\d+)x(\d+)$", k)
                if mm and int(mm.group(2)) == s["K"] and int(mm.group(3)) == s["N"]:
                    val = float(v) * s["M"] / int(mm.group(1))
                    self._base_cache[key] = val
                    return val
        else:
            for k, v in self.full.items():
                if k.startswith(node["op"] + "|"):
                    mm = re.match(re.escape(node["op"]) + r"\|(\d+)x(\d+)$", k)
                    if mm and int(mm.group(2)) == s.get("cols", 0):
                        val = float(v) * s.get("rows", 0) / max(int(mm.group(1)), 1)
                        self._base_cache[key] = val
                        return val
        if node["op"] == "matmul":
            flops = 2.0 * s["M"] * s["K"] * s["N"]
            val = max(flops / 250e12 * 1e6, _shape_bytes(node) / self.bw * 1e6)
        else:
            val = max(_shape_bytes(node) / self.bw * 1e6, 1.0)
        self._base_cache[key] = val
        return val

    def split_tax(self, node, strategy, k):
        """Extra time of k consecutive tiles over one unsplit kernel.

        E1 curves are measured with back-to-back launches, so they do not
        include the cross-stream record/wait of a DAG edge; ``event_us`` is
        added per tile for every split node that participates in a DAG.
        """
        if k <= 1:
            return 0.0
        ck = (shape_key(node), strategy, k)
        if ck in self._tax_cache:
            return self._tax_cache[ck]
        ev = k * self.event
        case = self.curves.get(shape_key(node))
        if case and case.get("fit"):
            slope = float(case["fit"]["slope_us_per_tile"])
            base = float(case["base_best_us"])
            val = max(0.0, slope * (k - 1) +
                      (float(case["fit"]["intercept_us"]) - base)) + ev
        elif strategy == "head":
            val = k * self.launch + ev
        elif strategy in ("col", "m2d"):
            s = node["shape"]
            c_mb = s["M"] * s["N"] * 2 / (1024.0 * 1024.0)
            val = k * self.launch + c_mb * 1.6 + ev
        else:
            b_bytes = 0.0
            if node["op"] == "matmul":
                s = node["shape"]
                b_bytes = s["K"] * s["N"] * 2
            traffic = 0.0
            if b_bytes > self.l2:
                traffic = (k - 1) * b_bytes / self.bw * 1e6
            val = k * self.launch + traffic + ev
        self._tax_cache[ck] = val
        return val

    def split_total(self, node, strategy, k):
        return self.base(node) + self.split_tax(node, strategy, k)

    def tax_ratio(self, node, strategy, k):
        base = self.base(node)
        return self.split_tax(node, strategy, k) / base if base else 1.0


def load_graph(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _topo_rounds(nodes, edges):
    names = [n["name"] for n in nodes]
    indeg = {n: 0 for n in names}
    succ = defaultdict(list)
    for e in edges:
        indeg[e["dst"]] += 1
        succ[e["src"]].append(e["dst"])
    rounds, node_round = [], {}
    done = set()
    while len(done) < len(names):
        ready = [n for n in names if n not in done and indeg[n] == 0]
        if not ready:
            raise ValueError("graph cycle")
        for n in ready:
            node_round[n] = len(rounds)
        rounds.append(ready)
        for n in ready:
            done.add(n)
        for n in ready:
            for v in succ[n]:
                indeg[v] -= 1
    return rounds, node_round


def _engine_res(engine):
    return {"aic", "aiv"} if engine == "aicore" else {engine}


def _split_parts(node, strategy, k, costs):
    total = costs.split_total(node, strategy, k)
    return [{"name": "%s#%d" % (node["name"], i), "tile": i, "parts": k,
             "dur": total / k, "strategy": strategy}
            for i in range(k)]


def _row_ranges(total_rows, k):
    edges = [round(i * total_rows / k) for i in range(k + 1)]
    return [(edges[i], edges[i + 1]) for i in range(k)]


def _tile_overlap(src_node, dst_node, sp_src, sp_dst):
    """Return dst tile -> list of source tiles it depends on.

    Row/head splits use proportional ranges; everything else is conservative
    all-to-all.  This is the 1:N mapping: a short consumer can start as soon
    as the source tiles covering its range are done.
    """
    ks, kd = sp_src["k"], sp_dst["k"]
    # Partial consumers (allreduce/allgather/... read only the first rows)
    # map their ranges onto the head of the producer output.  This must run
    # before the equal-k shortcut: every AR tile may still depend on producer
    # tile 0 when the consumed span is smaller than one producer tile.
    dst_rows = dst_node["shape"].get("rows") or dst_node["shape"].get("M") or 0
    src_rows = (src_node["shape"].get("M") or src_node["shape"].get("rows") or 0)
    if dst_rows and src_rows and dst_rows < src_rows \
            and sp_dst["strategy"] == "row":
        if sp_src["strategy"] == "row" and src_node["op"] == "matmul":
            total = src_node["shape"]["M"]
        elif sp_src["strategy"] == "head":
            total = src_node["shape"]["q_heads"]
        else:
            total = src_rows
        span = min(total, dst_rows)
        sr = _row_ranges(total, ks)
        dr = _row_ranges(span, kd)
        out = {}
        for j, (a, b) in enumerate(dr):
            out[j] = [i for i, (c, d) in enumerate(sr) if max(a, c) < min(b, d)]
        return out
    if ks == kd:
        return {j: [j] for j in range(kd)}
    if sp_src["strategy"] != sp_dst["strategy"]:
        return {j: list(range(ks)) for j in range(kd)}
    if sp_src["strategy"] == "row" and src_node["op"] == "matmul":
        total = src_node["shape"]["M"]
    elif sp_src["strategy"] == "head":
        total = src_node["shape"]["q_heads"]
    else:
        rows = src_node["shape"].get("rows") or src_node["shape"].get("M")
        if not rows:
            return {j: list(range(ks)) for j in range(kd)}
        total = rows
    sr = _row_ranges(total, ks)
    dr = _row_ranges(total, kd)
    out = {}
    for j, (a, b) in enumerate(dr):
        out[j] = [i for i, (c, d) in enumerate(sr) if max(a, c) < min(b, d)]
    return out


def simulate(graph, costs, splits=None, topo=None):
    """Level list schedule with per-engine FIFO and tile-wise dependencies.

    All split tiles of a node are consecutive in the node's engine queue.  The
    scheduler may start any task whose data deps are done and whose engine is
    free; thus an independent indegree-zero node can be placed between two
    sub-consumers when it becomes ready earlier.
    """
    splits = splits or {}
    nodes = graph["nodes"]
    nm = {n["name"]: n for n in nodes}
    if topo is None:
        rounds, node_round = _topo_rounds(nodes, graph["edges"])
        order = {n["name"]: i for i, n in enumerate(nodes)}
        topo = (rounds, node_round, order)
    else:
        rounds, node_round, order = topo

    tasks, by_node = [], defaultdict(list)
    for n in nodes:
        name = n["name"]
        sp = splits.get(name)
        k = int(sp.get("k", 1)) if sp else 1
        if k <= 1:
            parts = [{"name": name, "tile": 0, "parts": 1,
                      "dur": costs.base(n), "strategy": "none"}]
        else:
            parts = _split_parts(n, sp.get("strategy", "row"), k, costs)
        for p in parts:
            p["node"] = n
            p["engine"] = n["engine"]
            p["round"] = node_round[name]
            p["deps"] = set()
            p["data_preds"] = []
            by_node[name].append(p)
            tasks.append(p)
        for i in range(1, len(parts)):
            parts[i]["deps"].add(parts[i - 1]["name"])

    for e in graph["edges"]:
        src, dst = e["src"], e["dst"]
        sp_s, sp_d = splits.get(src), splits.get(dst)
        ks = int(sp_s.get("k", 1)) if sp_s else 1
        kd = int(sp_d.get("k", 1)) if sp_d else 1
        s_parts, d_parts = by_node[src], by_node[dst]
        if ks <= 1 and kd <= 1:
            for dp in d_parts:
                dp["deps"].add(s_parts[0]["name"])
                dp["data_preds"].append(s_parts[0]["name"])
        elif kd <= 1:
            # A partial consumer (allreduce reading only the first rows) only
            # needs the producer tiles covering that row range, even when the
            # consumer itself stays unsplit.
            sp_d = {"strategy": sp_s.get("strategy", "row"), "k": 1}
            mapping = _tile_overlap(nm[src], nm[dst], sp_s, sp_d)
            for i in mapping.get(0, range(ks)):
                s_part = s_parts[i]
                d_parts[0]["deps"].add(s_part["name"])
                d_parts[0]["data_preds"].append(s_part["name"])
        elif ks <= 1:
            for dp in d_parts:
                dp["deps"].add(s_parts[0]["name"])
                dp["data_preds"].append(s_parts[0]["name"])
        else:
            mapping = _tile_overlap(nm[src], nm[dst], sp_s, sp_d)
            for j, dp in enumerate(d_parts):
                for i in mapping.get(j, range(ks)):
                    dp["deps"].add(s_parts[i]["name"])
                    dp["data_preds"].append(s_parts[i]["name"])

    task_by_name = {t["name"]: t for t in tasks}
    finish = {}
    engine_free = defaultdict(float)
    schedule = []
    pending = list(task_by_name.values())
    done = set()
    while len(done) < len(pending):
        # Any task with satisfied deps is a candidate.  Choose the one that
        # can start earliest; ties by (round, node order, tile).
        best = None
        for t in pending:
            if t["name"] in done:
                continue
            if any(d not in finish for d in t["deps"]):
                continue
            res = _engine_res(t["engine"])
            efree = max(engine_free[r] for r in res)
            dfin = max([finish[d] for d in t["deps"]] or [0.0])
            st = max(efree, dfin)
            key = (st, t["round"], t["tile"], order[t["node"]["name"]])
            if best is None or key < best[0]:
                best = (key, t, st, dfin, efree)
        if best is None:
            missing = [t["name"] for t in pending if t["name"] not in done]
            raise ValueError("task graph cycle at %s" % missing[:5])
        _, t, st, dfin, efree = best
        en = st + t["dur"]
        finish[t["name"]] = en
        for r in _engine_res(t["engine"]):
            engine_free[r] = en
        critical = None
        if t["data_preds"]:
            critical = max(((d, finish[d]) for d in set(t["data_preds"])),
                           key=lambda x: x[1])
        schedule.append({
            "name": t["name"], "node": t["node"]["name"], "tile": t["tile"],
            "parts": t["parts"], "engine": t["engine"], "round": t["round"],
            "start": st, "end": en, "dur": t["dur"],
            "ready": dfin, "engine_free": efree,
            "critical": critical[0] if critical else None,
            "critical_end": critical[1] if critical else 0.0,
            "deps": sorted(t["deps"]),
            "data_preds": sorted(set(t["data_preds"])),
        })
        done.add(t["name"])

    schedule.sort(key=lambda x: (x["start"], x["engine"], x["name"]))
    makespan = max((s["end"] for s in schedule), default=0.0)
    return {"makespan_us": makespan, "tasks": schedule,
            "splits": copy.deepcopy(splits), "rounds": rounds}


def bubbles(sched):
    """Engine-idle gap caused by a data dependency (not by engine FIFO)."""
    out = []
    by_engine = defaultdict(list)
    for t in sched["tasks"]:
        by_engine[t["engine"]].append(t)
    for engine, ts in by_engine.items():
        ts = sorted(ts, key=lambda x: x["start"])
        prev_end = 0.0
        for t in ts:
            dep_gap = t["ready"] - t["engine_free"]
            if dep_gap > 1e-6 and t["critical"]:
                out.append({"engine": engine, "task": t["name"],
                            "node": t["node"], "tile": t["tile"],
                            "round": t["round"],
                            "dep_gap_us": dep_gap,
                            "engine_gap_us": t["start"] - prev_end,
                            "critical": t["critical"],
                            "critical_end": t["critical_end"],
                            "start": t["start"]})
            prev_end = max(prev_end, t["end"])
    out.sort(key=lambda x: -x["dep_gap_us"])
    return out


def edge_cross_stream(graph, edge):
    nm = {n["name"]: n for n in graph["nodes"]}
    src = edge["src"].split("#")[0]
    dst = edge["dst"].split("#")[0]
    return nm[src]["engine"] != nm[dst]["engine"]


def _chain_successors(graph, start, limit=3):
    """Same-engine elementwise chain following ``start`` (1:N insertion)."""
    nm = {n["name"]: n for n in graph["nodes"]}
    succ = defaultdict(list)
    for e in graph["edges"]:
        succ[e["src"]].append(e["dst"])
    chain, cur = [], start
    while len(chain) < limit:
        nxt = [v for v in succ.get(cur, []) if nm[v]["engine"] == nm[start]["engine"]]
        if len(nxt) != 1:
            break
        v = nxt[0]
        if nm[v]["op"] not in ELEMENTWISE and nm[v]["op"] != "matmul":
            break
        chain.append(v)
        cur = v
    return chain


def _data_predecessors(graph, dst):
    return [e["src"] for e in graph["edges"] if e["dst"] == dst]


def _consumers_of(graph, src):
    return [e["dst"] for e in graph["edges"] if e["src"] == src]


def _candidate_ks(node):
    s = node["shape"]
    if node["op"] == "attention":
        h = int(s.get("q_heads", 0))
        return [k for k in (2, 4, 7, 8, 14) if h and h % k == 0]
    rows = int(s.get("M") or s.get("rows") or 0)
    if not rows:
        return []
    ks = [k for k in (2, 4, 8, 16) if rows % k == 0]
    if node["op"] == "matmul":
        # Keep at least 1024 rows per tile: below that Block Num drops.
        ks = [k for k in ks if rows // k >= 1024]
    return ks


def _try_split_set(graph, costs, base_splits, src, dst, kp, kc, split_chain):
    nm = {n["name"]: n for n in graph["nodes"]}
    trial = copy.deepcopy(base_splits)
    # A join consumer (e.g. silu(gate, up)) only pipelines when every producer
    # is split; the producer side uses kp, the consumer/fanout chain uses kc.
    # kp != kc is the true 1:N case (one producer tile feeds several
    # sub-consumers, or several producer tiles feed one sub-consumer).
    producers = _data_predecessors(graph, dst) or [src]
    producer_set = set(producers)
    fanout = []
    for prod in producers:
        fanout.extend(_consumers_of(graph, prod))
    names = list(dict.fromkeys(producers + [dst] + fanout +
                               (split_chain if split_chain else [])))
    for name in names:
        n = nm[name]
        if n["op"] == "custom_aicpu_scale":
            # AICPU dispatch loses at k>1 (E4 probe).  It may stay whole as a
            # consumer, but cannot serve as a split producer.
            if name in producer_set:
                return None
            continue
        k = kp if name in producer_set else kc
        if n["op"] == "attention":
            strategy = "head"
        elif n["op"] == "matmul":
            strategy = "row"
        elif n["op"] in ELEMENTWISE or n["op"] in COMM:
            strategy = "row"
        else:
            return None
        if k not in _candidate_ks(n):
            return None
        trial[name] = {"strategy": strategy, "k": k}
    return trial


def _bubble_options(graph, costs, cur, b, min_gain_us, topo=None):
    """All positive split options for one bubble, best first."""
    src = b["critical"].split("#")[0]
    dst = b["node"]
    nm = {n["name"]: n for n in graph["nodes"]}
    if nm[src]["engine"] == nm[dst]["engine"]:
        return []
    chain = _chain_successors(graph, dst)
    # 局部上界剪枝：把 producer/consumer/fanout 都切到最小 k 时的额外
    # 工作量已经超过气泡 gap，则本气泡不可能有正收益。
    producers = _data_predecessors(graph, dst) or [src]
    fanout = []
    for prod in producers:
        fanout.extend(_consumers_of(graph, prod))
    base_names = list(dict.fromkeys(producers + [dst] + fanout))
    min_tax = 0.0
    for name in base_names:
        n = nm[name]
        if n["op"] == "custom_aicpu_scale" and name in set(producers):
            return []
        ks = _candidate_ks(n)
        if not ks:
            return []
        st = "head" if n["op"] == "attention" else "row"
        min_tax += costs.split_tax(n, st, min(ks))
    if min_tax >= b["dep_gap_us"]:
        return []
    options = []
    ks_p = _candidate_ks(nm[src])
    ks_c = _candidate_ks(nm[dst])
    pairs = [(kp, kc) for kp in ks_p for kc in ks_c]
    pairs = sorted(set(pairs), key=lambda x: (x[0] > x[1], x[0], x[1]))[:6]
    for kp, kc in pairs:
        for split_chain in (False, True):
            trial = _try_split_set(graph, costs, cur["splits"], src, dst,
                                   kp, kc, chain if split_chain else None)
            if not trial:
                continue
            try:
                s2 = simulate(graph, costs, trial, topo=topo)
            except Exception:
                continue
            gain = cur["makespan_us"] - s2["makespan_us"]
            if gain >= min_gain_us:
                options.append((gain, (kp, kc), split_chain, trial, s2))
    options.sort(key=lambda x: -x[0])
    return options


def _decide_bubble(graph, costs, cur, b, min_gain_us, topo=None):
    opts = _bubble_options(graph, costs, cur, b, min_gain_us, topo=topo)
    return opts[0] if opts else None


def _union_trial(t1, t2):
    """Merge two split sets; None when they disagree on a shared node."""
    trial = copy.deepcopy(t1)
    for name, spec in t2.items():
        if name in trial and trial[name] != spec:
            return None
        trial[name] = dict(spec)
    return trial


def _merge_decision(decisions, rnd, b, options, accepted):
    got = options[0] if options else None
    decisions.append({
        "round": rnd, "consumer": b["node"], "engine": b["engine"],
        "critical_edge": "%s -> %s" % (b["critical"], b["node"]),
        "dep_gap_us": b["dep_gap_us"],
        "k": ("%d/%d" % got[1]) if got else None,
        "split_consumer_chain": got[2] if got else False,
        "predicted_gain_us": got[0] if got else 0.0,
        "alternatives": [{"k": "%d/%d" % o[1], "gain_us": o[0],
                          "split_chain": o[2]} for o in (options or [])],
        "accepted": accepted,
    })


def greedy_plan(graph, costs, max_rounds=40, min_gain_us=1.0,
                top_options=3):
    free_splits = {}
    nodes = graph["nodes"]
    nm = {n["name"]: n for n in nodes}
    rounds, node_round = _topo_rounds(nodes, graph["edges"])
    topo = (rounds, node_round, {n["name"]: i for i, n in enumerate(nodes)})
    cur = simulate(graph, costs, free_splits, topo=topo)
    base = cur["makespan_us"]
    history, decisions = [], []
    seen = set()
    for rnd in range(max_rounds):
        cands = [b for b in bubbles(cur)
                 if edge_cross_stream(graph, {"src": b["critical"],
                                              "dst": b["node"]})
                 and b["dep_gap_us"] >= min_gain_us
                 and (b["node"], b["critical"]) not in seen]
        # 同构层去重：Qwen 24 层有相同的跨引擎边，逐层模拟是重复工作。
        uniq, sigs = [], set()
        for b in cands:
            src = nm[b["critical"].split("#")[0]]
            dstn = nm[b["node"]]
            sig = (src["op"], dstn["op"],
                   json.dumps(src.get("shape", {}), sort_keys=True),
                   json.dumps(dstn.get("shape", {}), sort_keys=True),
                   round(b["dep_gap_us"], 1))
            if sig in sigs:
                continue
            sigs.add(sig)
            uniq.append(b)
        cands = uniq
        if not cands:
            break
        per_bubble = []
        for b in cands[:8]:
            opts = _bubble_options(graph, costs, cur, b, min_gain_us, topo=topo)
            if opts:
                per_bubble.append((b, opts))
        if not per_bubble:
            break
        # Option 1: the single best (bubble, k) split.
        best = None
        for b, opts in per_bubble:
            for o in opts[:top_options]:
                if best is None or o[0] > best[1]:
                    best = ([(b, o)], o[0], o[3], o[4])
        # Option 2: unions of top options from two different bubbles.  A
        # single split may starve a second wave on the shared engine; the
        # union is re-simulated because the k choices interact.
        flat = sorted(((b, o) for b, opts in per_bubble
                       for o in opts[:top_options]),
                      key=lambda x: -x[1][0])[:8]
        for i in range(len(flat)):
            b1, o1 = flat[i]
            for j in range(i + 1, len(flat)):
                b2, o2 = flat[j]
                if b1["node"] == b2["node"] \
                        and b1["critical"] == b2["critical"]:
                    continue
                trial = _union_trial(o1[3], o2[3])
                if trial is None:
                    continue
                try:
                    s2 = simulate(graph, costs, trial)
                except Exception:
                    continue
                gain = cur["makespan_us"] - s2["makespan_us"]
                if gain > best[1] + 1e-9:
                    best = ([(b1, o1), (b2, o2)], gain, trial, s2)
        chosen, gain, trial, s2 = best
        chosen_bubbles = {id(b) for b, _ in chosen}
        for b, opts in per_bubble:
            _merge_decision(decisions, rnd, b, opts,
                            id(b) in chosen_bubbles)
        parts = []
        for b, o in chosen:
            parts.append("k=%d/%d" % o[1] + ("+chain" if o[2] else ""))
            for name in o[3]:
                seen.add((name, b["critical"]))
        history.append({"round": rnd, "gain_us": gain,
                        "k": ", ".join(parts),
                        "split_chain": any(o[2] for _, o in chosen),
                        "splits": copy.deepcopy(trial),
                        "bubble": chosen[0][0],
                        "bubbles": [{"consumer": b["node"],
                                     "critical": b["critical"]}
                                    for b, _ in chosen],
                        "makespan_us": s2["makespan_us"]})
        free_splits = trial
        cur = s2
    return free_splits, cur, history, base, decisions


def graph_from_dag(path):
    """Convert an executor DAG (tiles, edges) to the planner graph format."""
    dag = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes, edges = [], []
    for n in dag["nodes"]:
        s = dict(n.get("shape") or {})
        if n.get("op") == "matmul":
            s.setdefault("M", n.get("rows", 0))
        nodes.append({"name": n["name"], "op": n["op"],
                      "engine": n["engine"], "shape": s})
    for e in dag.get("edges", []):
        edges.append({"src": e["src"], "dst": e["dst"],
                      "kind": e.get("kind", "data")})
    return {"nodes": nodes, "edges": edges, "name": dag.get("name", "")}


def summary_markdown(graph, costs, base, planned, history, decisions):
    nm = {n["name"]: n for n in graph["nodes"]}
    lines = ["# planner-2: schedule A + bubble decisions", "",
             "- graph: %s (%d nodes, %d edges)"
             % (graph.get("name", "?"), len(graph["nodes"]), len(graph["edges"])),
             "- schedule A makespan: %.1f us" % base["makespan_us"],
             "- planned makespan: %.1f us (%.2f%% vs A)"
             % (planned["makespan_us"],
                100.0 * (planned["makespan_us"] / base["makespan_us"] - 1.0)),
             "- accepted splits: %d" % len(history), ""]
    lines += ["## Bubbles and decisions", "",
              "| engine | consumer | critical edge | cross stream | dep gap us | best k | consumer chain | predicted gain us | decision |",
              "|---|---|---|---|---:|---:|---|---:|---|"]
    bub = bubbles(base)
    for b in bub[:40]:
        src = b["critical"].split("#")[0]
        cross = nm[src]["engine"] != nm[b["node"]]["engine"]
        d = next((x for x in decisions
                  if x["consumer"] == b["node"]
                  and x["critical_edge"].startswith(b["critical"])), None)
        lines.append("| %s | %s | %s -> %s | %s | %.1f | %s | %s | %.1f | %s |"
                     % (b["engine"], b["node"], src, b["node"],
                        "yes" if cross else "no", b["dep_gap_us"],
                        d["k"] if d and d["k"] else "-",
                        "yes" if d and d["split_consumer_chain"] else "no",
                        d["predicted_gain_us"] if d else 0.0,
                        ("accept" if d and d["accepted"] else
                         ("reject (no positive gain)" if (d or cross)
                          else "reject (same engine)"))))
    lines += ["", "## Accepted split rounds", "",
              "| round | split nodes | k | gain us | makespan us |",
              "|---|---|---:|---:|---:|"]
    for h in history:
        nodes = ", ".join("%s:%s/%d" % (k, v["strategy"], v["k"])
                          for k, v in sorted(h["splits"].items()))
        lines.append("| %d | %s | %s | %.1f | %.1f |"
                     % (h["round"], nodes, h["k"], h["gain_us"], h["makespan_us"]))
    lines += ["", "Final splits: `%s`" % json.dumps(planned["splits"],
                                                    ensure_ascii=False)]
    return "\n".join(lines) + "\n"
