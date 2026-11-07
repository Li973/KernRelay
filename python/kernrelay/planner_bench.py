"""Planner 独立实验：不同 DAG 的规划耗时 + 问题 DAG 拒绝。"""
import argparse
import copy
import json
import time
from pathlib import Path

from .runtime import KERNRELAY_ROOT
from .planner2 import (COMM, ELEMENTWISE, Costs, _topo_rounds,
                       greedy_plan, simulate)

SUPPORTED = {"matmul", "attention", "embedding", "custom_aicpu_scale"} | \
            ELEMENTWISE | COMM


def validate_graph(graph):
    """结构/语义校验；返回错误列表，非空表示 planner 必须拒绝。"""
    errs = []
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not nodes:
        return ["nodes 缺失或为空"]
    if not isinstance(edges, list):
        return ["edges 缺失"]
    names = [n.get("name") for n in nodes]
    if any(not n for n in names):
        errs.append("存在空节点名")
    if len(names) != len(set(names)):
        errs.append("节点名重复")
    nm = {n.get("name"): n for n in nodes}
    for n in nodes:
        name = n.get("name", "?")
        op = n.get("op")
        if op not in SUPPORTED:
            errs.append("%s: 不支持算子 %r" % (name, op))
            continue
        eng = n.get("engine")
        if eng not in ("aic", "aiv", "aicore", "aicpu", "hccl", "compute"):
            errs.append("%s: 非法引擎 %r" % (name, eng))
        s = n.get("shape") or {}
        try:
            if op == "matmul":
                for key in ("M", "K", "N"):
                    if int(s.get(key, 0)) <= 0:
                        errs.append("%s: matmul %s<=0" % (name, key))
            elif op == "attention":
                for key in ("batch", "seq", "q_heads", "kv_heads", "head_dim"):
                    if int(s.get(key, 0)) <= 0:
                        errs.append("%s: attention %s<=0" % (name, key))
                q, kv = int(s.get("q_heads", 0)), int(s.get("kv_heads", 0))
                if q and kv and (q % kv != 0 or kv > q):
                    errs.append("%s: attention q_heads=%d 不是 kv_heads=%d 的整数倍"
                                % (name, q, kv))
            elif op in ("embedding", "allreduce", "allgather", "reducescatter"):
                for key in ("rows", "cols"):
                    if int(s.get(key, 0)) <= 0:
                        errs.append("%s: %s %s<=0" % (name, op, key))
            elif op != "custom_aicpu_scale":
                for key in ("rows", "cols"):
                    if int(s.get(key, 0)) <= 0:
                        errs.append("%s: %s %s<=0" % (name, op, key))
        except Exception as e:
            errs.append("%s: shape 非法 (%s)" % (name, e))
    for e in edges:
        src, dst = e.get("src"), e.get("dst")
        if src not in nm:
            errs.append("edge src 不存在: %r" % src)
        if dst not in nm:
            errs.append("edge dst 不存在: %r" % dst)
        if src == dst:
            errs.append("自环: %s" % src)
    if not errs:
        try:
            _topo_rounds(nodes, edges)
        except ValueError as ex:
            errs.append("依赖成环: %s" % ex)
    return errs


def _node(name, op, engine, shape, deps=()):
    return {"name": name, "op": op, "engine": engine, "shape": shape,
            "deps": list(deps)}


def problem_graphs():
    base = {"nodes": [
        _node("ln", "rmsnorm", "aiv", {"rows": 2048, "cols": 896}),
        _node("mm", "matmul", "aic", {"M": 2048, "K": 896, "N": 896}, ["ln"]),
    ], "edges": [{"src": "ln", "dst": "mm"}]}
    out = []

    g = copy.deepcopy(base); g["name"] = "bad_cycle"
    g["nodes"].append(_node("x", "add", "aiv", {"rows": 2048, "cols": 896}, ["mm"]))
    g["edges"].append({"src": "mm", "dst": "ln"})
    out.append(("bad_cycle", g, "cycle"))

    g = copy.deepcopy(base); g["name"] = "bad_unknown_op"
    g["nodes"][1]["op"] = "frobnicate"
    out.append(("bad_unknown_op", g, "unknown op"))

    g = copy.deepcopy(base); g["name"] = "bad_shape"
    g["nodes"][1]["shape"]["M"] = 0
    out.append(("bad_shape", g, "M=0"))

    g = copy.deepcopy(base); g["name"] = "bad_edge"
    g["edges"].append({"src": "mm", "dst": "missing"})
    out.append(("bad_edge", g, "missing node"))

    g = copy.deepcopy(base); g["name"] = "bad_dup_name"
    g["nodes"].append(copy.deepcopy(g["nodes"][0]))
    out.append(("bad_dup_name", g, "duplicate"))

    g = {"name": "bad_attention", "nodes": [
        _node("attn", "attention", "aicore",
              {"batch": 1, "seq": 2048, "q_heads": 14, "kv_heads": 3,
               "head_dim": 64})], "edges": []}
    out.append(("bad_attention", g, "kv_heads not divisor"))

    # 合法但不该切的 DAG：AICPU producer / 小 tile。
    g = {"name": "aicpu_producer", "nodes": [
        _node("cpu", "custom_aicpu_scale", "aicpu", {"rows": 2048, "cols": 896}),
        _node("mm", "matmul", "aic", {"M": 2048, "K": 896, "N": 896}, ["cpu"])],
        "edges": [{"src": "cpu", "dst": "mm"}]}
    out.append(("aicpu_producer", g, "valid_but_no_split"))
    g = {"name": "tiny_matmul", "nodes": [
        _node("mm", "matmul", "aic", {"M": 512, "K": 1024, "N": 1024})],
        "edges": []}
    out.append(("tiny_matmul", g, "valid_but_no_split"))
    return out
