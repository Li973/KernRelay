"""One-command optimize pipeline for an external Ascend project.

    project -> DAG -> unique-shape operator profile -> planner -> execute

Supported projects in 0.3:
  * an exported KernRelay LLM directory (manifest.json + weights.bin)
  * a HuggingFace Qwen2.5 / Qwen3-VL dense model directory (auto-export)
  * a DAG JSON (nodes/edges) for the generic executor

The pipeline never writes model weights into the repository; the cache
directory is chosen by the caller (``--cache`` / cache_root).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .libapi import KERNRELAY_ROOT
from .planner2 import Costs, greedy_plan


def detect_project(path):
    p = Path(path)
    if (p / "manifest.json").exists():
        return "kernrelay", p
    if (p / "kr" / "manifest.json").exists():
        return "kernrelay", p / "kr"
    if (p / "config.json").exists():
        cfg = json.loads((p / "config.json").read_text())
        mt = cfg.get("model_type", "")
        if mt in ("qwen3_omni_moe",) or "thinker_config" in cfg:
            return "inventory", p
        if "text_config" in cfg:
            return "hf", p
        return "hf", p
    if p.suffix == ".json":
        return "dag", p
    raise FileNotFoundError("无法识别的项目: %s（需要 manifest/kr、HF 模型或 DAG JSON）" % path)


def ensure_export(model_dir, cache_root, seq, device):
    model_dir = Path(model_dir)
    out = Path(cache_root) / model_dir.name
    if (out / "manifest.json").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(KERNRELAY_ROOT / "scripts/export_model.py"),
           "--model", str(model_dir), "--out", str(out), "--seq", str(seq),
           "--device", str(device)]
    subprocess.run(cmd, check=True)
    return out


def manifest_cfg(kr_dir):
    return json.loads((Path(kr_dir) / "manifest.json").read_text())["config"]


def layer_dag(cfg, seq):
    """Compact per-layer DAG used for inventory + planner."""
    H = cfg["hidden_size"]; I = cfg["intermediate_size"]
    L = cfg["num_hidden_layers"]; HQ = cfg["num_attention_heads"]
    HKV = cfg["num_key_value_heads"]
    D = cfg.get("head_dim", H // HQ)
    V = cfg["vocab_size"]; qk = bool(cfg.get("qk_norm", False))
    nodes, edges = [], []

    def add(name, op, engine, shape, **kw):
        n = {"name": name, "op": op, "engine": engine, "shape": shape,
             "dtype": "bf16"}
        n.update(kw)
        nodes.append(n)

    def edge(a, b):
        edges.append({"src": a, "dst": b, "kind": "data"})

    add("embed", "embedding", "aiv", {"rows": seq, "cols": H})
    prev = "embed"
    for l in range(L):
        p = "l%d" % l
        add(p + ".ln1", "rmsnorm", "aiv", {"rows": seq, "cols": H})
        edge(prev, p + ".ln1")
        add(p + ".q", "matmul", "aic",
            {"M": seq, "K": H, "N": HQ * D})
        add(p + ".k", "matmul", "aic",
            {"M": seq, "K": H, "N": HKV * D})
        add(p + ".v", "matmul", "aic",
            {"M": seq, "K": H, "N": HKV * D})
        for x in ("q", "k", "v"):
            edge(p + ".ln1", p + "." + x)
        if qk:
            add(p + ".qk_norm", "rmsnorm", "aiv",
                {"rows": seq * HQ, "cols": D})
            edge(p + ".q", p + ".qk_norm")
            edge(p + ".k", p + ".qk_norm")
            qsrc = p + ".qk_norm"
        else:
            qsrc = p + ".q"
        add(p + ".rope", "rope", "aiv", {"rows": seq, "cols": HQ * D})
        edge(qsrc, p + ".rope")
        add(p + ".attn", "attention", "aicore",
            {"batch": 1, "q_heads": HQ, "kv_heads": HKV,
             "seq": seq, "head_dim": D})
        edge(p + ".rope", p + ".attn")
        edge(p + ".v", p + ".attn")
        add(p + ".o", "matmul", "aic",
            {"M": seq, "K": HQ * D, "N": H})
        edge(p + ".attn", p + ".o")
        add(p + ".addnorm1", "add_rmsnorm", "aiv",
            {"rows": seq, "cols": H})
        edge(prev, p + ".addnorm1")
        edge(p + ".o", p + ".addnorm1")
        add(p + ".gate", "matmul", "aic",
            {"M": seq, "K": H, "N": I})
        add(p + ".up", "matmul", "aic",
            {"M": seq, "K": H, "N": I})
        edge(p + ".addnorm1", p + ".gate")
        edge(p + ".addnorm1", p + ".up")
        add(p + ".silu", "silu", "aiv", {"rows": seq, "cols": I})
        edge(p + ".gate", p + ".silu")
        add(p + ".mul", "mul", "aiv", {"rows": seq, "cols": I})
        edge(p + ".silu", p + ".mul")
        edge(p + ".up", p + ".mul")
        add(p + ".down", "matmul", "aic",
            {"M": seq, "K": I, "N": H})
        edge(p + ".mul", p + ".down")
        add(p + ".addnorm2", "add_rmsnorm", "aiv",
            {"rows": seq, "cols": H})
        edge(p + ".addnorm1", p + ".addnorm2")
        edge(p + ".down", p + ".addnorm2")
        prev = p + ".addnorm2"
    add("final_norm", "rmsnorm", "aiv", {"rows": seq, "cols": H})
    edge(prev, "final_norm")
    add("lm_head", "matmul", "aic", {"M": seq, "K": H, "N": V})
    edge("final_norm", "lm_head")
    return {"name": "llm_" + cfg.get("arch", "qwen"), "nodes": nodes,
            "edges": edges}


def unique_ops(dag):
    seen = {}
    for n in dag["nodes"]:
        s = n["shape"]
        if n["op"] == "matmul":
            key = "matmul|%dx%dx%d" % (s["M"], s["K"], s["N"])
        elif n["op"] == "attention":
            key = "attention|%dx%dx%dx%d" % (
                s["batch"], s["q_heads"], s["seq"], s["head_dim"])
        else:
            key = "%s|%dx%d" % (n["op"], s["rows"], s["cols"])
        e = seen.setdefault(key, {"key": key, "op": n["op"],
                                  "count": 0, "engines": set()})
        e["count"] += 1
        e["engines"].add(n["engine"])
    out = list(seen.values())
    for e in out:
        e["engines"] = sorted(e["engines"])
    out.sort(key=lambda x: (-x["count"], x["key"]))
    return out


def profile_project(kr_dir, cfg, seq, device):
    # C ABI 是进程级：profile 必须在独立进程中跑，随后主进程再执行。
    fd, tmp = tempfile.mkstemp(prefix="kr_profile_", suffix=".json")
    os.close(fd)
    cmd = [sys.executable, str(KERNRELAY_ROOT / "scripts/profile_model.py"),
           "--model", str(kr_dir), "--seq", str(seq), "--device", str(device),
           "--out", tmp]
    subprocess.run(cmd, check=True, cwd=str(KERNRELAY_ROOT))
    try:
        return json.loads(Path(tmp).read_text(encoding="utf-8"))
    finally:
        os.unlink(tmp)


def plan_dag(dag, profile, layers):
    base = json.loads((KERNRELAY_ROOT / "configs/planner2_costs.json").read_text())
    full = dict(base.get("full", {}))
    curves = {"cases": []}
    key_of = {}
    for n in dag["nodes"]:
        s = n["shape"]
        if n["op"] == "matmul":
            key = "matmul|%dx%dx%d" % (s["M"], s["K"], s["N"])
        elif n["op"] == "attention":
            key = "attention|%dx%dx%dx%d" % (
                s["batch"], s["q_heads"], s["seq"], s["head_dim"])
        else:
            key = "%s|%dx%d" % (n["op"], s["rows"], s["cols"])
        key_of[n["name"]] = key
    def prof_op(n):
        if n["op"] == "matmul":
            nm = n["name"]
            tail = nm.rsplit(".", 1)[-1]
            if tail in ("q", "k", "v", "o", "gate", "up", "down"):
                return "matmul_" + tail
            if tail == "lm_head":
                return "matmul_lm"
            return "matmul"
        if n["op"] == "embedding":
            return "embedding"
        if n["op"] == "attention":
            return "attention"
        return n["op"]

    for n in dag["nodes"]:
        op = prof_op(n)
        if op in profile["table"] and profile["table"][op]:
            v = profile["table"][op]
            k1 = v.get(1)
            if k1:
                full[key_of[n["name"]]] = k1
    costs = Costs.__new__(Costs)
    costs.data = base; costs.full = full
    costs.launch = base.get("launch_us", 8.0)
    costs.bw = base.get("traffic_gbps", 600.0) * 1e9
    costs.l2 = base.get("l2_mb", 32.0) * 1024 * 1024
    costs.event = base.get("event_us", 20.0)
    costs.curves = {}; costs._base_cache = {}; costs._tax_cache = {}
    for n in dag["nodes"]:
        op = prof_op(n)
        v = profile["table"].get(op) or {}
        if not v.get(1) or not v.get(8):
            continue
        slope = max(0.0, (v[8] - v[1]) / 7.0)
        key = key_of[n["name"]]
        costs.curves[key] = {"op": n["op"], "strategy": "row",
                             "shape": n["shape"], "base_best_us": v[1],
                             "fit": {"slope_us_per_tile": slope,
                                     "intercept_us": v[1]}}
    splits, cur, hist, base_m, decisions = greedy_plan(dag, costs, max_rounds=8)
    return {"base_makespan_us": base_m, "planned_makespan_us": cur["makespan_us"],
            "splits": splits, "history": hist,
            "decisions": decisions}


def project_summary(dag, cfg, seq):
    inv = unique_ops(dag)
    return {"arch": cfg.get("arch"), "seq": seq,
            "hidden": cfg.get("hidden_size"), "layers": cfg.get("num_hidden_layers"),
            "nodes": len(dag["nodes"]), "edges": len(dag["edges"]),
            "unique_shapes": len(inv), "inventory": inv}


def _exec_once(kind, kr_dir, seq, device, pair, mode, tiles, warmup, iters):
    fd, tmp = tempfile.mkstemp(prefix="kr_run_", suffix=".json")
    os.close(fd)
    cmd = [sys.executable, str(KERNRELAY_ROOT / "scripts/run_once.py"),
           "--kind", kind, "--model", str(kr_dir), "--seq", str(seq),
           "--warmup", str(warmup), "--iters", str(iters), "--out", tmp]
    if kind == "single":
        cmd += ["--device", str(device), "--mode", mode]
    else:
        cmd += ["--dev0", str(pair[0]), "--dev1", str(pair[1]),
                "--tiles", str(tiles), "--mode", mode]
    subprocess.run(cmd, check=True, cwd=str(KERNRELAY_ROOT))
    try:
        return json.loads(Path(tmp).read_text(encoding="utf-8"))
    finally:
        os.unlink(tmp)


def optimize(project, seq=2048, device=1, dev_pair=None, mode="auto",
             cache_root=None, out_dir=None, tiles=None):
    kind, path = detect_project(project)
    if cache_root is None:
        cache_root = Path(os.environ.get(
            "KERNRELAY_CACHE", Path.home() / ".cache/kernrelay"))
    if kind == "inventory":
        cfg = json.loads((Path(path) / "config.json").read_text())
        from .models import inventory_config
        inv = inventory_config(cfg)
        return {"project": str(path), "kind": "inventory",
                "inventory": inv, "executed": False,
                "reason": inv.get("reason")}
    if kind == "dag":
        # generic schedule JSON: profiler/planner are the caller's job.
        return {"project": str(path), "kind": "dag",
                "executed": False,
                "reason": "use `python -m kernrelay e34/e5/e9` or kr_plan_run"}
    kr_dir = ensure_export(path, cache_root, seq, device) if kind == "hf" else Path(path)
    cfg = manifest_cfg(kr_dir)
    dag = layer_dag(cfg, seq)
    summary = project_summary(dag, cfg, seq)
    prof = profile_project(kr_dir, cfg, seq, device)
    plan = plan_dag(dag, prof, cfg["num_hidden_layers"])
    result = {"project": str(path), "cache": str(kr_dir), "kind": kind,
              "summary": summary, "plan": {
                  "base_makespan_us": plan["base_makespan_us"],
                  "planned_makespan_us": plan["planned_makespan_us"],
                  "splits": plan["splits"]},
              "profile_table": prof["table"], "runs": {}}
    if mode in ("auto", "single"):
        result["runs"]["single"] = _exec_once(
            "single", kr_dir, seq, device, dev_pair, "graph_whole",
            1, 1, 3)
    if mode in ("auto", "tp") and dev_pair:
        k = tiles
        if k is None:
            h = cfg["hidden_size"]
            k = 4 if h >= 3000 else (2 if h >= 2000 else 1)
        variants = (False,) if k == 1 else (False, True)
        for split in variants:
            payload = _exec_once(
                "tp", kr_dir, seq, device, dev_pair,
                "tp2_split" if split else "tp2", k, 1, 3)
            stages = payload.get("stages") or []
            if stages:
                payload["argmax"] = stages[0].get("argmax")
                payload["maxdiff"] = stages[0].get("maxdiff")
            result["runs"]["tp_split" if split else "tp"] = payload
    if out_dir:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "optimize.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True,
                    help="HF 模型目录、KernRelay 导出目录或 DAG JSON")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--devices", default=None, help="TP 双卡，例如 1,2")
    ap.add_argument("--mode", default="auto", choices=["auto", "single", "tp"])
    ap.add_argument("--tiles", type=int, default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    pair = tuple(int(x) for x in args.devices.split(",")) if args.devices else None
    r = optimize(args.project, args.seq, args.device, pair, args.mode,
                 args.cache, args.out, args.tiles)
    print(json.dumps(r.get("summary", r), ensure_ascii=False, indent=1)[:4000])
    for k, v in r.get("runs", {}).items():
        print(k, "argmax", v.get("argmax"), "wall/best_us",
              v.get("wall_best_us", v.get("best_us")), "rc", v.get("_rc"))
    if args.out:
        print("wrote", Path(args.out) / "optimize.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
