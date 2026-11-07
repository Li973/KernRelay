"""E3/E4: 计算/计算 1:1 与 1:N 的 serial / B0 / KernRelay 对比。"""

import json
import time
from pathlib import Path

from .runtime import KERNRELAY_ROOT, resolve_devices
from .runner import RunConfig, run_schedule
from .schedule import (build_fixed_schedule, load_spec, serial_schedule,
                       validate_schedule)
from .stats import checksum_consistent, speedup

DEFAULT_CONFIG = KERNRELAY_ROOT / "configs" / "e34_overlap.json"


def _build_variants(spec, chunking, nranks):
    return {
        "serial": serial_schedule(spec, nranks=nranks),
        "b0": build_fixed_schedule(spec, {}, nranks=nranks, method="B0-nosplit"),
        "split": build_fixed_schedule(spec, chunking, nranks=nranks,
                                      method="KernRelay-split"),
    }


def _row(exp, res):
    return {"best_us": res["best_us"], "typical_us": res["typical_us"],
            "p50_us": res["p50_us"], "p90_us": res["p90_us"],
            "min_us": res["min_us"], "max_us": res["max_us"],
            "n_samples": res["n_samples"], "n_processes": res["n_processes"],
            "clean_processes": res.get("clean_processes"),
            "contaminated_processes": res.get("contaminated_processes"),
            "typical_all_us": res.get("typical_all_us"),
            "retries": res["retries"], "gate_timeouts": res["gate_timeouts"],
            "checksum": res["checksums"][0], "checksums": res["checksums"],
            "samples_us": res["samples_us"], "proc_mins_us": res["proc_mins_us"],
            "attempts": res["attempts"]}


def _summary(raw):
    m = raw["meta"]
    lines = ["# E3/E4 计算/计算重叠", "",
             "- nranks=%d，每变体至少 %d 进程 × %d 计时轮；best=全局最小，"
             "typical=各进程最小值的中位数"
             % (m["nranks"], m["min_processes"], m["iters_per_process"]),
             "- serial=全部 tile 单流拓扑序；B0=不切分多流（消费者等整算子）；"
             "split=KernRelay 行切分逐块 event",
             ""]
    for group in ("E3", "E4"):
        rows = [r for r in raw["experiments"] if r["group"] == group]
        if not rows:
            continue
        lines.extend(["## %s" % ("E3 1:1 计算/计算" if group == "E3"
                                else "E4 1:N 计算/计算"), "",
                      "| 实验 | DAG | chunking | serial best/typ | B0 best/typ | "
                      "split best/typ | 加速比 serial/split(min) | 加速比 B0/split(min) | "
                      "节省(vs serial typ) | checksum | retry |",
                      "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"])
        for r in rows:
            s, b0, sp = r["variants"]["serial"], r["variants"]["b0"], r["variants"]["split"]
            lines.append("| %s | %s | %s | %.1f / %.1f | %.1f / %.1f | %.1f / %.1f | "
                         "%.3f× | %.3f× | %.1f%% | %s | %d/%d/%d |"
                         % (r["name"], r["dag"], r["chunking"],
                            s["best_us"], s["typical_us"],
                            b0["best_us"], b0["typical_us"],
                            sp["best_us"], sp["typical_us"],
                            r["speedup_min_vs_serial"],
                            r["speedup_min_vs_b0"],
                            100.0 * (1.0 - sp["typical_us"] / s["typical_us"])
                            if s["typical_us"] else 0.0,
                            "一致" if r["checksum_ok"] else "差异",
                            s["retries"], b0["retries"], sp["retries"]))
        lines.append("")
        for r in rows:
            lines.append("### %s（%s）" % (r["name"], r["note"]))
            lines.append("")
            lines.append("| 变体 | best | typical(clean) | typical(all) | raw p50 | raw p90 | "
                         "samples | procs(clean/dirty) | retry | gate |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for name in ("serial", "b0", "split"):
                v = r["variants"][name]
                lines.append("| %s | %.1f | %.1f | %.1f | %.1f | %.1f | %d | %d (%s/%s) | %d | %d |"
                             % (name, v["best_us"], v["typical_us"],
                                v.get("typical_all_us") or v["typical_us"],
                                v["p50_us"], v["p90_us"], v["n_samples"],
                                v["n_processes"],
                                v.get("clean_processes", "-"),
                                v.get("contaminated_processes", "-"),
                                v["retries"], v["gate_timeouts"]))
            lines.append("")
    lines.append("## 判定")
    lines.append("")
    for item in raw["findings"]:
        lines.append("- %s" % item)
    return "\n".join(lines) + "\n"


def _findings(raw):
    out = []
    rows = raw["experiments"]
    for group in ("E3", "E4"):
        grp = [r for r in rows if r["group"] == group]
        if not grp:
            continue
        minwin = [r for r in grp if r["speedup_min_vs_serial"] > 1.0
                  and r["speedup_min_vs_b0"] > 1.0]
        typwin = []
        for r in grp:
            sp = r["variants"]["split"]["typical_us"]
            s = r["variants"]["serial"]["typical_us"]
            b0 = r["variants"]["b0"]["typical_us"]
            if sp and s and b0 and sp < s and sp < b0:
                typwin.append(r)
        out.append("%s：min 口径 %d/%d 个 split 同时快于 serial 与 B0；"
                   "typical(clean) 口径 %d/%d。"
                   % (group, len(minwin), len(grp), len(typwin), len(grp)))
        mixed = [r["name"] for r in minwin if r not in typwin]
        if mixed:
            out.append("min 赢但 typical 未同时赢：%s；best 是下界，"
                       "典型口径失败说明收益不稳，不作为主结论。"
                       % "、".join(mixed))
    bad = [r["name"] for r in rows if not r["checksum_ok"]]
    if bad:
        out.append("checksum 不一致仅出现在 %s：`custom_aicpu_scale` 只写 out[0]"
                   "（整幅标量累加），按行切分后每个 tile 各写一个部分和，"
                   "算子本身不满足 tile-invariance；该 case 仅作 AICPU 派发成本"
                   "负结果，不参与数值等价结论。" % "、".join(bad))
    else:
        out.append("全部实验 serial/B0/split checksum 一致。")
    if any(r["variants"]["split"]["gate_timeouts"] for r in rows):
        out.append("存在 gate 超时，需复测。")
    dirty = sum((v.get("contaminated_processes") or 0)
                for r in rows for v in r["variants"].values())
    single = [r["name"] for r in rows
              if (r["variants"]["split"].get("clean_processes") or 0) < 2]
    if dirty:
        out.append("共 %d 个进程踩到设备停顿（proc min >3×全局 best），"
                   "已从 typical 剔除；详见各变体 clean/dirty 列。" % dirty)
    if single:
        out.append("split 的干净进程少于 2 个的 case：%s；其 typical 只由单个"
                   "进程最小值构成，仅作参考。" % "、".join(single))
    return out


def run_e34(config_path=None, devices="1,2", out_dir=None, samples=3,
            nranks=None, only=None, iters_per_process=5, warmup=2):
    config_path = Path(config_path or DEFAULT_CONFIG)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    nranks = int(nranks if nranks is not None else config.get("nranks", 2))
    out_dir = Path(out_dir) if out_dir else (
        KERNRELAY_ROOT / "results" / "e34" / time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    logical = resolve_devices(devices)
    cfg = RunConfig(devices=logical, nranks=nranks, samples=samples,
                    process_isolation=False, fail_fast=True,
                    warmup=warmup, iters_per_process=iters_per_process)
    wanted = [s.strip() for s in only.split(",")] if only else None
    exps = [e for e in config["experiments"]
            if not wanted or e["name"] in wanted]
    raw = {"meta": {"config": str(config_path), "devices_arg": devices,
                    "devices_logical": logical, "nranks": nranks,
                    "samples_target": samples, "warmup": cfg.warmup,
                    "iters_per_process": cfg.iters_per_process,
                    "min_processes": cfg.min_processes,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
           "experiments": []}
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
        raw["experiments"].append(row)
    raw["findings"] = _findings(raw)
    (out_dir / "raw.json").write_text(
        json.dumps(raw, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.md").write_text(_summary(raw), encoding="utf-8")
    return raw
