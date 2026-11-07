"""E5: 计算/通信重叠（matmul -> allreduce/allgather/reducescatter）。"""

import json
import time
from pathlib import Path

from .e34 import _build_variants, _row
from .runtime import KERNRELAY_ROOT, resolve_devices
from .runner import RunConfig, run_schedule
from .schedule import load_spec, serial_schedule, validate_schedule
from .stats import checksum_consistent, speedup

DEFAULT_CONFIG = KERNRELAY_ROOT / "configs" / "e5_comm.json"


def _metrics(row, calib):
    serial = row["variants"]["serial"]["best_us"]
    split = row["variants"]["split"]["best_us"]
    comm_us = (calib or {}).get("best_us")
    if not comm_us:
        return {"comm_us": comm_us, "mm_est_us": None, "exposed_us": None,
                "hidden_ratio": None, "ideal_us": None, "overlap_eff": None}
    mm_est = serial - comm_us
    exposed = max(0.0, split - mm_est)
    ideal = max(mm_est, comm_us)
    denom = serial - ideal
    return {"comm_us": comm_us, "mm_est_us": mm_est, "exposed_us": exposed,
            "hidden_ratio": max(0.0, min(1.0, 1.0 - exposed / comm_us)),
            "ideal_us": ideal,
            "overlap_eff": ((serial - split) / denom) if denom > 1e-9 else None}


def _summary(raw):
    m = raw["meta"]
    lines = ["# E5 计算/通信重叠", "",
             "- nranks=%d，每变体至少 %d 进程 × %d 计时轮；best=全局最小，"
             "typical=干净进程最小值的中位数"
             % (m["nranks"], m["min_processes"], m["iters_per_process"]),
             "- serial=单流拓扑序；B0=不切分多流；split=matmul 行切 + 逐块 collective",
             "- 通信标定：`e5_ar_only`（2048 行）与 `e5_ar_full_only`（32768 行）"
             "单算子 allreduce；allgather/reducescatter 复用同规模 allreduce 作近似分母",
             ""]
    lines.extend(["## 通信单算子标定", "",
                  "| 标定 | DAG | best (us) | typical | raw p50 | retry |",
                  "|---|---|---:|---:|---:|---:|"])
    for name, c in raw["calibration"].items():
        lines.append("| %s | %s | %.1f | %.1f | %.1f | %d |"
                     % (name, c["dag"], c["best_us"], c["typical_us"],
                        c["p50_us"], c["retries"]))
    lines.extend(["", "## 重叠结果", "",
                  "| 实验 | DAG | chunking | serial best/typ | B0 best/typ | "
                  "split best/typ | 加速比 serial/split | 加速比 B0/split | 隐藏通信比 | "
                  "重叠效率 | exposed (us) | checksum | retry |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"])
    for r in raw["experiments"]:
        s, b0, sp = (r["variants"]["serial"], r["variants"]["b0"],
                     r["variants"]["split"])
        mt = r["metrics"]
        lines.append("| %s | %s | %s | %.1f / %.1f | %.1f / %.1f | %.1f / %.1f | "
                     "%.3f× | %.3f× | %s | %s | %s | %s | %d/%d/%d |"
                     % (r["name"], r["dag"], r["chunking"],
                        s["best_us"], s["typical_us"],
                        b0["best_us"], b0["typical_us"],
                        sp["best_us"], sp["typical_us"],
                        r["speedup_min_vs_serial"], r["speedup_min_vs_b0"],
                        _pct(mt["hidden_ratio"]), _pct(mt["overlap_eff"]),
                        _us(mt["exposed_us"]),
                        "一致" if r["checksum_ok"] else "差异",
                        s["retries"], b0["retries"], sp["retries"]))
    lines.append("")
    lines.extend(["## 逐变体明细", "",
                  "| 实验 | 变体 | best | typical(clean) | raw p50 | raw p90 | "
                  "samples | procs(clean/dirty) | retry | gate |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for r in raw["experiments"]:
        for name in ("serial", "b0", "split"):
            v = r["variants"][name]
            lines.append("| %s | %s | %.1f | %.1f | %.1f | %.1f | %d | %d (%s/%s) | %d | %d |"
                         % (r["name"], name, v["best_us"], v["typical_us"],
                            v["p50_us"], v["p90_us"], v["n_samples"],
                            v["n_processes"], v.get("clean_processes", "-"),
                            v.get("contaminated_processes", "-"),
                            v["retries"], v["gate_timeouts"]))
    lines.extend(["", "## 判定", ""])
    for item in raw["findings"]:
        lines.append("- %s" % item)
    return "\n".join(lines) + "\n"


def _pct(v):
    return "-" if v is None else "%.1f%%" % (100.0 * v)


def _us(v):
    return "-" if v is None else "%.1f" % v


def _findings(raw):
    out = []
    rows = raw["experiments"]
    win = [r for r in rows if r["speedup_min_vs_serial"] > 1.0
           and r["speedup_min_vs_b0"] > 1.0]
    out.append("E5：%d/%d 个 split 同时快于 serial 与 B0。" % (len(win), len(rows)))
    best = max(rows, key=lambda r: r["speedup_min_vs_b0"]) if rows else None
    if best:
        out.append("对 B0 的最大收益来自 %s：%.3f×（隐藏通信比 %s）。"
                   % (best["name"], best["speedup_min_vs_b0"],
                      _pct(best["metrics"]["hidden_ratio"])))
    if not all(r["checksum_ok"] for r in rows):
        out.append("存在 checksum 不一致，性能数据不可用。")
    else:
        out.append("全部实验 serial/B0/split checksum 一致。")
    if any(r["variants"]["split"]["gate_timeouts"] for r in rows):
        out.append("存在 gate 超时，需复测。")
    dirty = sum((v.get("contaminated_processes") or 0)
                for r in rows for v in r["variants"].values())
    if dirty:
        out.append("共 %d 个进程踩到设备停顿（proc min >3×全局 best），"
                   "已从 typical 剔除；详见逐变体明细。" % dirty)
    return out


def run_e5(config_path=None, devices="1,2", out_dir=None, samples=3,
           nranks=None, only=None, iters_per_process=5, warmup=2):
    config_path = Path(config_path or DEFAULT_CONFIG)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    nranks = int(nranks if nranks is not None else config.get("nranks", 2))
    out_dir = Path(out_dir) if out_dir else (
        KERNRELAY_ROOT / "results" / "e5" / time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    logical = resolve_devices(devices)
    cfg = RunConfig(devices=logical, nranks=nranks, samples=samples,
                    process_isolation=False, fail_fast=True,
                    warmup=warmup, iters_per_process=iters_per_process)
    raw = {"meta": {"config": str(config_path), "devices_arg": devices,
                    "devices_logical": logical, "nranks": nranks,
                    "samples_target": samples, "warmup": cfg.warmup,
                    "iters_per_process": cfg.iters_per_process,
                    "min_processes": cfg.min_processes,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
           "calibration": {}, "experiments": []}
    for cal in config.get("calibration", []):
        spec = load_spec(cal["dag"])
        sched = serial_schedule(spec, nranks=nranks)
        errors = validate_schedule(sched)
        if errors:
            raise ValueError("%s 结构非法: %s" % (cal["name"], errors))
        res = run_schedule(sched, cfg, cal["name"], out_dir / "calib" / cal["name"])
        raw["calibration"][cal["name"]] = dict(
            _row(None, res), dag=cal["dag"], rows=cal["rows"])
        c = raw["calibration"][cal["name"]]
        print("%-22s calib  best %9.1f  typical %9.1f  retry %d"
              % (cal["name"], c["best_us"], c["typical_us"], c["retries"]))
    wanted = [s.strip() for s in only.split(",")] if only else None
    exps = [e for e in config["experiments"]
            if not wanted or e["name"] in wanted]
    for exp in exps:
        spec = load_spec(exp["dag"])
        variants = _build_variants(spec, exp["chunking"], nranks)
        row = {"name": exp["name"], "group": exp["group"], "dag": exp["dag"],
               "chunking": exp["chunking"], "note": exp.get("note", ""),
               "variants": {}}
        for name in ("serial", "b0", "split"):
            sched = variants[name]
            errors = validate_schedule(sched)
            if errors:
                raise ValueError("%s/%s 结构非法: %s" % (exp["name"], name, errors))
            res = run_schedule(sched, cfg, name,
                               out_dir / exp["name"] / name)
            row["variants"][name] = _row(exp, res)
            v = row["variants"][name]
            print("%-22s %-6s best %9.1f  typical %9.1f  p50 %9.1f  retry %d"
                  % (exp["name"], name, v["best_us"], v["typical_us"],
                     v["p50_us"], v["retries"]))
        base = row["variants"]["serial"]["checksum"]
        row["checksum_ok"] = all(
            checksum_consistent(base, row["variants"][k]["checksum"])
            for k in ("b0", "split"))
        row["speedup_min_vs_serial"] = speedup(
            row["variants"]["serial"]["best_us"],
            row["variants"]["split"]["best_us"])
        row["speedup_min_vs_b0"] = speedup(
            row["variants"]["b0"]["best_us"],
            row["variants"]["split"]["best_us"])
        row["metrics"] = _metrics(
            row, raw["calibration"].get(exp.get("comm_calib")))
        raw["experiments"].append(row)
    raw["findings"] = _findings(raw)
    (out_dir / "raw.json").write_text(
        json.dumps(raw, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.md").write_text(_summary(raw), encoding="utf-8")
    return raw
