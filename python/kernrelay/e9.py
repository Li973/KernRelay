"""E9: favorable self-designed DAGs built from the E8 planning table.

serial = one topological stream; B0 = unsplit multi-stream; split = KernRelay
row chunking with per-tile ACL events.  These DAGs are mechanism upper bounds,
not a real model end-to-end (see E7).
"""

import json
import time
from pathlib import Path

from .e34 import _build_variants, _row
from .runtime import KERNRELAY_ROOT, resolve_devices
from .runner import RunConfig, run_schedule
from .schedule import load_spec, validate_schedule
from .stats import checksum_consistent, speedup

DEFAULT_CONFIG = KERNRELAY_ROOT / "configs" / "e9_favorable.json"


def _checksums_ok(row):
    base = row["variants"]["serial"]["checksums"]
    for key in ("b0", "split"):
        other = row["variants"][key]["checksums"]
        if len(other) != len(base):
            return False
        if not all(checksum_consistent(a, b) for a, b in zip(base, other)):
            return False
    return True


def _findings(raw):
    out = []
    rows = raw["experiments"]
    if not rows:
        return ["无有效实验。"]
    both = [r for r in rows
            if r["speedup_min_vs_serial"] > 1.0 and r["speedup_min_vs_b0"] > 1.0]
    typ = []
    for r in rows:
        s = r["variants"]["serial"]["typical_us"]
        b = r["variants"]["b0"]["typical_us"]
        sp = r["variants"]["split"]["typical_us"]
        if s and b and sp and sp < s and sp < b:
            typ.append(r)
    out.append("min 口径：%d/%d 个 DAG 的 split 同时快于 serial 与 B0；"
               "typical(clean) 口径 %d/%d。"
               % (len(both), len(rows), len(typ), len(rows)))

    waves = {r["name"]: r for r in rows if r["dag"] == "e9_two_waves"}
    mixed = {r["name"]: r for r in rows if r["dag"] == "e9_mixed"}
    chain = {r["name"]: r for r in rows if r["dag"] == "e9_chain2"}

    if waves:
        w = min(waves.values(), key=lambda r: r["variants"]["split"]["best_us"])
        out.append("独立两波 %s：B0/split 仅 %.3f×，但 serial/split %.3f×。"
                   "B0 已让第一波 gelu 与第二波 GEMM 跨 AIC/AIV 重叠；"
                   "切分只把最后一个 gelu tail 提前，边际收益来自尾部而非新增并行。"
                   % (w["name"], w["speedup_min_vs_b0"],
                      w["speedup_min_vs_serial"]))
    if mixed:
        m = next(iter(mixed.values()))
        out.append("混合 DAG %s：B0/split %.3f×、serial/split %.3f×，是本组对 B0 "
                   "收益最大的形态。allreduce 只消费 mm2 的前 2048 行，"
                   "因此只需等 mm2 tile 0 完成即可启动，剩余 GEMM tile 与 HCCL "
                   "真正并行；通信隐藏不额外占用 AIC/AIV。"
                   % (m["name"], m["speedup_min_vs_b0"],
                      m["speedup_min_vs_serial"]))
    if chain:
        c = min(chain.values(), key=lambda r: r["variants"]["split"]["best_us"])
        b0_over_serial = (c["variants"]["b0"]["typical_us"]
                          / c["variants"]["serial"]["typical_us"])
        out.append("依赖链 %s：B0/serial=%.3f（整节点跨流依赖使 B0 退化为串行），"
                   "split 同时赢两者 %.3f×/%.3f×。收益被 AIC 流内 FIFO 限制："
                   "mm2 仍排在整个 mm1 之后，只能与 AIV 的 gelu1/gelu2 波前重叠；"
                   "k=8 略差于 k=4。"
                   % (c["name"], b0_over_serial,
                      c["speedup_min_vs_serial"], c["speedup_min_vs_b0"]))
    if waves and mixed:
        w = min(waves.values(), key=lambda r: r["variants"]["split"]["best_us"])
        m = next(iter(mixed.values()))
        if m["variants"]["split"]["best_us"] < w["variants"]["split"]["best_us"]:
            out.append("注意 %s 与 %s 的节点集合不同（前者为 3 计算节点+通信，"
                       "后者为 4 计算节点），绝对耗时不可直接横比；"
                       "混合 DAG 的结论只用于说明通信可被 GEMM 隐藏。"
                       % (m["name"], w["name"]))
    bad = [r["name"] for r in rows if not r["checksum_ok"]]
    out.append("checksum：%s。"
               % ("全部一致" if not bad else "不一致：" + "、".join(bad)))
    dirty = sum((v.get("contaminated_processes") or 0)
                for r in rows for v in r["variants"].values())
    retry = sum(v["retries"] for r in rows for v in r["variants"].values())
    if dirty or retry:
        out.append("设备停顿/重试：dirty 进程 %d 个、retry %d 次；"
                   "typical 已剔除脏进程，明细见各变体表。" % (dirty, retry))
    return out


def _summary(raw):
    m = raw["meta"]
    lines = ["# E9 有利自设 DAG（机制上限，非真实模型）", "",
             "- nranks=%d，每变体至少 %d 进程 × %d 计时轮；口径同 E3/E4。"
             % (m["nranks"], m["min_processes"], m["iters_per_process"]),
             "- E8 表给出的预期：mm→gelu k=4 的 serial/split≈1.10；"
             "两波/混合 DAG 把该收益重复叠加，目标是验证“多波持续隐藏”。",
             "",
             "| 实验 | DAG | chunking | serial best/typ | B0 best/typ | "
             "split best/typ | 加速比 serial/split | 加速比 B0/split | 节省(vs serial typ) | checksum | retry |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in raw["experiments"]:
        s, b0, sp = (r["variants"]["serial"], r["variants"]["b0"],
                     r["variants"]["split"])
        lines.append(
            "| %s | %s | %s | %.1f / %.1f | %.1f / %.1f | %.1f / %.1f | "
            "%.3f× | %.3f× | %.1f%% | %s | %d/%d/%d |"
            % (r["name"], r["dag"], r["chunking"],
               s["best_us"], s["typical_us"], b0["best_us"], b0["typical_us"],
               sp["best_us"], sp["typical_us"],
               r["speedup_min_vs_serial"], r["speedup_min_vs_b0"],
               100.0 * (1.0 - sp["typical_us"] / s["typical_us"])
               if s["typical_us"] else 0.0,
               "一致" if r["checksum_ok"] else "差异",
               s["retries"], b0["retries"], sp["retries"]))
    lines.append("")
    for r in raw["experiments"]:
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
    lines += ["- " + s for s in raw["findings"]]
    lines.append("- 本实验为自设有利 DAG 的机制上限，不代表真实模型端到端（E7 单独报告）。")
    return "\n".join(lines) + "\n"


def run_e9(config_path=None, devices="1,2", out_dir=None, samples=3,
           nranks=None, only=None, iters_per_process=5, warmup=2):
    config_path = Path(config_path or DEFAULT_CONFIG)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    nranks = int(nranks if nranks is not None else config.get("nranks", 2))
    out_dir = Path(out_dir) if out_dir else (
        KERNRELAY_ROOT / "results" / "e9" / time.strftime("%Y%m%d_%H%M%S"))
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
                     v["p50_us"], v["retries"]), flush=True)
        row["checksum_ok"] = _checksums_ok(row)
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
    print("写入: %s" % out_dir)
    return raw
