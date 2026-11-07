"""E0: correctness, repeatability, and invalid-schedule rejection."""

import json
import time
from pathlib import Path

from .runtime import resolve_devices
from .runner import RunConfig, run_expect_failure, run_schedule
from .schedule import (build_ab, effective_chunking, invalid_variants,
                       load_fixed, validate_schedule)
from .stats import checksum_consistent, checksum_spread, speedup


def _dag_row(name, path, spec, chunking, ra, rb, runs):
    all_checksums = ra["checksums"] + rb["checksums"]
    all_spread = checksum_spread(all_checksums)
    ab_ok = checksum_consistent(ra["checksums"][0], rb["checksums"][0])
    repeat_ok = (checksum_spread(ra["checksums"]) < 1e-3
                 and checksum_spread(rb["checksums"]) < 1e-3)
    split_ok = bool(effective_chunking(chunking))
    checks = {
        "split_declared": split_ok,
        "checksum_ab": ab_ok,
        "checksum_repeat_a": checksum_spread(ra["checksums"]) < 1e-3,
        "checksum_repeat_b": checksum_spread(rb["checksums"]) < 1e-3,
        "checksum_all_spread_ok": all_spread < 1e-3,
        "samples_a": ra["n_samples"] >= runs,
        "samples_b": rb["n_samples"] >= runs,
        "a_no_gate_timeout": ra["gate_timeouts"] == 0,
        "b_no_gate_timeout": rb["gate_timeouts"] == 0,
    }
    return {
        "name": name, "file": str(path), "chunking": chunking,
        "a": {k: ra[k] for k in ("p50_us", "p90_us", "min_us", "max_us",
                                 "best_us", "typical_us", "n_samples",
                                 "n_processes", "proc_mins_us",
                                 "proc_medians_us", "checksums",
                                 "gate_timeouts", "retries",
                                 "teardown_warns", "stalls", "attempts")},
        "b": {k: rb[k] for k in ("p50_us", "p90_us", "min_us", "max_us",
                                 "best_us", "typical_us", "n_samples",
                                 "n_processes", "proc_mins_us",
                                 "proc_medians_us", "checksums",
                                 "gate_timeouts", "retries",
                                 "teardown_warns", "stalls", "attempts")},
        "intrinsic_speedup_min": speedup(ra["best_us"], rb["best_us"]),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _builder_invalid(spec, chunking):
    cases = []
    for label, bad in (("chunk_not_divisible", {"mm": 3}),
                       ("unknown_node", {"__ghost__": 2})):
        try:
            build_ab(spec, dict(chunking, **bad), mode="b", nranks=2)
            cases.append({"case": label, "rejected": False, "error": ""})
        except Exception as ex:  # builder is expected to reject
            cases.append({"case": label, "rejected": True, "error": str(ex)})
    return cases


def run_e0(dag_names, devices, runs=3, nranks=2, out_dir=None):
    out_dir = Path(out_dir or ("results/e0/%s" % time.strftime("%Y%m%d_%H%M%S")))
    out_dir.mkdir(parents=True, exist_ok=True)
    logical = resolve_devices(devices)
    cfg = RunConfig(devices=logical, nranks=nranks, samples=runs,
                    process_isolation=False, fail_fast=True)
    raw = {"meta": {"devices_arg": devices, "devices_logical": logical,
                    "nranks": nranks, "runs": runs,
                    "process_isolation": cfg.process_isolation,
                    "warmup": cfg.warmup,
                    "iters_per_process": cfg.iters_per_process,
                    "min_processes": cfg.min_processes,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
           "dags": [], "invalid_schedules": [], "builder_rejections": []}
    failures = []
    for name in dag_names:
        path = name
        spec, chunking = load_fixed(name)
        scheds = build_ab(spec, chunking, mode="ab", nranks=nranks)
        for key, sched in scheds.items():
            errors = validate_schedule(sched)
            if errors:
                raise ValueError("%s/%s 结构非法: %s" % (name, key, errors))
        run_dir = out_dir / "runs" / Path(name).stem
        ra = run_schedule(scheds["a"], cfg, "A-nosplit", run_dir / "A")
        rb = run_schedule(scheds["b"], cfg, "B-split", run_dir / "B")
        row = _dag_row(Path(name).stem, path, spec, chunking, ra, rb, runs)
        raw["dags"].append(row)
        if not row["pass"]:
            failures.append("%s: %s" % (name, [k for k, v in row["checks"].items()
                                               if not v]))

    tiny, tiny_chunk = load_fixed("e0_tiny_mm_gelu")
    tiny_sched = build_ab(tiny, tiny_chunk, mode="b", nranks=nranks)["b"]
    invalid_dir = out_dir / "invalid"
    for label, (bad_sched, marker) in invalid_variants(tiny_sched).items():
        rec = run_expect_failure(bad_sched, cfg, label, invalid_dir)
        rejected = (rec["exit_code"] != 0 and not rec["has_result"]
                    and marker in rec["stderr_tail"])
        rec.update({"case": label, "expected_marker": marker,
                    "rejected": rejected})
        raw["invalid_schedules"].append(rec)
        if not rejected:
            failures.append("invalid/%s 未被拒绝" % label)

    raw["builder_rejections"] = _builder_invalid(tiny, tiny_chunk)
    failures.extend("%s 未被 builder 拒绝" % c["case"]
                    for c in raw["builder_rejections"] if not c["rejected"])
    raw["failures"] = failures
    raw["pass"] = not failures
    (out_dir / "raw.json").write_text(
        json.dumps(raw, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.md").write_text(_summary(raw), encoding="utf-8")
    return raw


def _summary(raw):
    lines = ["# E0 正确性与门禁", ""]
    lines.append("- 设备: `%s` → 逻辑 `%s`；nranks=%d；目标 %d 个计时样本，"
                 "至少 %d 个进程（每进程 warmup=%d, iters=%d）"
                 % (raw["meta"]["devices_arg"], raw["meta"]["devices_logical"],
                    raw["meta"]["nranks"], raw["meta"]["runs"],
                    raw["meta"]["min_processes"], raw["meta"]["warmup"],
                    raw["meta"]["iters_per_process"]))
    lines.append("- 结论: **%s**" % ("通过" if raw["pass"] else "失败"))
    lines.append("- 注：E0 只验证正确性/可重复性/拒绝非法调度；计时受共享设备停顿影响，"
                 "不作为性能结论。")
    lines.append("")
    lines.append("| DAG | chunking | A best (us) | B best (us) | A typical | B typical | A raw p50 | B raw p50 | best 比 | checksum | 重复 | retry |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|")
    for r in raw["dags"]:
        lines.append("| %s | %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.3fx | %s | %s | %d/%d |"
                     % (r["name"], r["chunking"], r["a"]["best_us"],
                        r["b"]["best_us"], r["a"]["typical_us"],
                        r["b"]["typical_us"], r["a"]["p50_us"],
                        r["b"]["p50_us"], r["intrinsic_speedup_min"],
                        "一致" if r["checks"]["checksum_ab"] else "差异",
                        "是" if (r["checks"]["checksum_repeat_a"]
                                and r["checks"]["checksum_repeat_b"]) else "否",
                        r["a"]["retries"], r["b"]["retries"]))
    lines.append("")
    lines.append("| 非法 schedule | 期望错误 | exit | 拒绝 |")
    lines.append("|---|---|---:|---|")
    for r in raw["invalid_schedules"]:
        lines.append("| %s | `%s` | %s | %s |"
                     % (r["case"], r["expected_marker"], r["exit_code"],
                        "是" if r["rejected"] else "否"))
    lines.append("")
    if raw["failures"]:
        lines.append("失败项：")
        lines.extend("- " + f for f in raw["failures"])
    return "\n".join(lines) + "\n"
