"""Command line entry: ``python -m kernrelay {run,e0,e1,validate}``."""

import argparse
import json
import sys
import time
from pathlib import Path

from .e0 import run_e0
from .e34 import run_e34
from .e5 import run_e5
from .e9 import run_e9
from .runtime import KERNRELAY_ROOT, resolve_devices
from .runner import RunConfig, run_schedule
from .schedule import build_ab, load_fixed, validate_schedule
from .stats import checksum_consistent, checksum_spread, speedup


def _env_pairs(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("--env 需要 KEY=VALUE: %r" % item)
        k, v = item.split("=", 1)
        out[k] = v
    return out


def _out_dir(kind, out):
    if out:
        return Path(out)
    return KERNRELAY_ROOT / "results" / kind / time.strftime("%Y%m%d_%H%M%S")


def _cmd_validate(args):
    bad = 0
    for name in args.dag:
        spec, chunking = load_fixed(name)
        for key, sched in build_ab(spec, chunking, args.mode,
                                   nranks=args.nranks).items():
            errors = validate_schedule(sched)
            print("%s/%s: %s" % (name, key, "ok" if not errors else errors))
            bad += bool(errors)
    return 0 if not bad else 1


def _cmd_e0(args):
    raw = run_e0(args.dag, args.devices, runs=args.runs,
                 nranks=args.nranks, out_dir=_out_dir("e0", args.out))
    print("E0 %s" % ("PASS" if raw["pass"] else "FAIL"))
    for r in raw["dags"]:
        print("  %-24s min A %8.1f  B %8.1f  %.3fx  p50 A %8.1f  B %8.1f  "
              "checksum=%s  retry=%d/%d"
              % (r["name"], r["a"]["best_us"], r["b"]["best_us"],
                 r["intrinsic_speedup_min"], r["a"]["p50_us"],
                 r["b"]["p50_us"],
                 "ok" if r["checks"]["checksum_ab"] else "DIFF",
                 r["a"]["retries"], r["b"]["retries"]))
    for f in raw["failures"]:
        print("  FAIL: %s" % f)
    print("写入: %s" % _out_dir("e0", args.out))
    return 0 if raw["pass"] else 1


def _cmd_e34(args):
    out_dir = Path(args.out) if args.out else None
    raw = run_e34(args.config, args.devices, out_dir=out_dir,
                  samples=args.samples, nranks=args.nranks, only=args.only,
                  iters_per_process=args.iters_per_process,
                  warmup=args.warmup)
    print("E3/E4 完成: %d 个实验" % len(raw["experiments"]))
    for r in raw["experiments"]:
        print("  %-22s serial/split %.3fx  B0/split %.3fx  checksum=%s"
              % (r["name"], r["speedup_min_vs_serial"],
                 r["speedup_min_vs_b0"],
                 "ok" if r["checksum_ok"] else "DIFF"))
    return 0


def _cmd_e5(args):
    out_dir = Path(args.out) if args.out else None
    raw = run_e5(args.config, args.devices, out_dir=out_dir,
                 samples=args.samples, nranks=args.nranks, only=args.only,
                 iters_per_process=args.iters_per_process,
                 warmup=args.warmup)
    print("E5 完成: %d 个实验" % len(raw["experiments"]))
    for r in raw["experiments"]:
        print("  %-24s serial/split %.3fx  B0/split %.3fx  hidden %s  checksum=%s"
              % (r["name"], r["speedup_min_vs_serial"],
                 r["speedup_min_vs_b0"],
                 _fmt_pct(r["metrics"].get("hidden_ratio")),
                 "ok" if r["checksum_ok"] else "DIFF"))
    return 0


def _fmt_pct(v):
    return "-" if v is None else "%.1f%%" % (100.0 * v)


def _cmd_e9(args):
    out_dir = Path(args.out) if args.out else None
    raw = run_e9(args.config, args.devices, out_dir=out_dir,
                 samples=args.samples, nranks=args.nranks, only=args.only,
                 iters_per_process=args.iters_per_process,
                 warmup=args.warmup)
    print("E9 完成: %d 个实验" % len(raw["experiments"]))
    for r in raw["experiments"]:
        print("  %-22s serial/split %.3fx  B0/split %.3fx  checksum=%s"
              % (r["name"], r["speedup_min_vs_serial"],
                 r["speedup_min_vs_b0"],
                 "ok" if r["checksum_ok"] else "DIFF"))
    return 0


def _cmd_optimize(args):
    from .optimize import optimize
    pair = tuple(int(x) for x in args.devices.split(",")) if args.devices else None
    r = optimize(args.project, args.seq, args.device, pair, args.mode,
                 args.cache, args.out, args.tiles)
    print(json.dumps(r.get("summary", r), ensure_ascii=False, indent=1)[:4000])
    for k, v in r.get("runs", {}).items():
        print("%-9s argmax=%s best=%s rc=%s"
              % (k, v.get("argmax"), v.get("wall_best_us", v.get("best_us")),
                 v.get("_rc")))
    return 0


def _cmd_run(args):
    out_dir = _out_dir("run", args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logical = resolve_devices(args.devices)
    cfg = RunConfig(devices=logical, nranks=args.nranks, samples=args.runs,
                    process_isolation=args.isolation,
                    warmup=args.warmup, iters_per_process=args.iters_per_process,
                    fail_fast=not args.allow_fail,
                    gap_s=args.gap_s, retry_sleep_s=args.retry_sleep,
                    env=_env_pairs(args.env))
    rows = []
    failures = []
    for name in args.dag:
        spec, chunking = load_fixed(name)
        scheds = build_ab(spec, chunking, args.mode, nranks=args.nranks)
        row = {"name": Path(name).stem, "chunking": chunking}
        for key, sched in scheds.items():
            errors = validate_schedule(sched)
            if errors:
                raise ValueError("%s/%s 结构非法: %s" % (name, key, errors))
            res = run_schedule(sched, cfg, key, out_dir / Path(name).stem / key)
            row[key] = res
            print("%-24s %-2s min %9.1f us  p50 %9.1f  p90 %9.1f  "
                  "typical %9.1f  retry %d  gate %d  proc %d"
                  % (row["name"], key.upper(), res["best_us"],
                     res["p50_us"], res["p90_us"], res["typical_us"],
                     res["retries"], res["gate_timeouts"], res["n_processes"]))
        if "a" in row and "b" in row:
            row["checksum_ab"] = checksum_consistent(
                row["a"]["checksums"][0], row["b"]["checksums"][0])
            row["checksum_spread_ab"] = checksum_spread(
                row["a"]["checksums"] + row["b"]["checksums"])
            row["speedup_min_a_over_b"] = speedup(row["a"]["best_us"],
                                                  row["b"]["best_us"])
            row["speedup_p50_a_over_b"] = speedup(row["a"]["p50_us"],
                                                  row["b"]["p50_us"])
            if not row["checksum_ab"]:
                failures.append("%s: A/B checksum 不一致" % row["name"])
            print("%-24s min 加速 %.3fx  p50 加速 %.3fx  checksum %s"
                  % (row["name"], row["speedup_min_a_over_b"],
                     row["speedup_p50_a_over_b"],
                     "ok" if row["checksum_ab"] else "DIFF"))
        rows.append(row)
    payload = {"meta": {"devices_arg": args.devices, "devices_logical": logical,
                        "nranks": args.nranks, "runs": args.runs,
                        "mode": args.mode, "isolation": args.isolation,
                        "warmup": args.warmup,
                        "iters_per_process": args.iters_per_process,
                        "env": _env_pairs(args.env),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
               "rows": rows, "failures": failures}
    (out_dir / "raw.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print("写入: %s" % out_dir)
    return 0 if not failures else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kernrelay")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--dag", required=True, action="append",
                       help="DAG 名称或路径，可重复")
        p.add_argument("--devices", default="0,1")
        p.add_argument("--nranks", type=int, default=2)
        p.add_argument("--mode", default="ab", choices=["a", "b", "ab"])
        p.add_argument("--out", default=None)

    p_run = sub.add_parser("run", help="运行 A/B 或单侧 schedule")
    common(p_run)
    p_run.add_argument("--runs", type=int, default=5, help="目标计时样本数")
    p_run.add_argument("--warmup", type=int, default=2,
                       help="每进程预热轮数（不计入样本）")
    p_run.add_argument("--iters-per-process", type=int, default=3,
                       help="每进程计时轮数")
    p_run.add_argument("--isolation", action="store_true",
                       help="进程内只跑 1 轮（cold 口径；默认 warm 口径）")
    p_run.add_argument("--gap-s", type=float, default=2.0,
                       help="有效进程之间的间隔秒数")
    p_run.add_argument("--retry-sleep", type=float, default=6.0,
                       help="失败重试前的冷却秒数")
    p_run.add_argument("--allow-fail", action="store_true",
                       help="环境/确定性失败也重试（默认 fail-fast）")
    p_run.add_argument("--env", action="append", default=[],
                       help="附加环境变量 KEY=VALUE，可重复")

    p_e0 = sub.add_parser("e0", help="E0 正确性/重复性/非法 schedule 门禁")
    common(p_e0)
    p_e0.add_argument("--runs", type=int, default=3, help="每侧进程隔离样本数")

    p_e34 = sub.add_parser("e34", help="E3/E4 计算/计算 1:1 与 1:N")
    p_e34.add_argument("--config", default=None)
    p_e34.add_argument("--devices", default="1,2")
    p_e34.add_argument("--nranks", type=int, default=None)
    p_e34.add_argument("--samples", type=int, default=3)
    p_e34.add_argument("--iters-per-process", type=int, default=5)
    p_e34.add_argument("--warmup", type=int, default=2)
    p_e34.add_argument("--only", default=None, help="逗号分隔的实验名")
    p_e34.add_argument("--out", default=None)

    p_e5 = sub.add_parser("e5", help="E5 计算/通信重叠")
    p_e5.add_argument("--config", default=None)
    p_e5.add_argument("--devices", default="1,2")
    p_e5.add_argument("--nranks", type=int, default=None)
    p_e5.add_argument("--samples", type=int, default=3)
    p_e5.add_argument("--iters-per-process", type=int, default=5)
    p_e5.add_argument("--warmup", type=int, default=2)
    p_e5.add_argument("--only", default=None, help="逗号分隔的实验名")
    p_e5.add_argument("--out", default=None)

    p_e9 = sub.add_parser("e9", help="E9 有利自设 DAG")
    p_e9.add_argument("--config", default=None)
    p_e9.add_argument("--devices", default="1,2")
    p_e9.add_argument("--nranks", type=int, default=None)
    p_e9.add_argument("--samples", type=int, default=3)
    p_e9.add_argument("--iters-per-process", type=int, default=5)
    p_e9.add_argument("--warmup", type=int, default=2)
    p_e9.add_argument("--only", default=None, help="逗号分隔的实验名")
    p_e9.add_argument("--out", default=None)

    p_opt = sub.add_parser("optimize",
                           help="外部项目：DAG -> 算子预跑 -> planner -> 执行")
    p_opt.add_argument("--project", required=True,
                       help="HF 模型目录、KernRelay 导出目录或 DAG JSON")
    p_opt.add_argument("--seq", type=int, default=2048)
    p_opt.add_argument("--device", type=int, default=1)
    p_opt.add_argument("--devices", default=None, help="TP 双卡，例如 1,2")
    p_opt.add_argument("--mode", default="auto",
                       choices=["auto", "single", "tp"])
    p_opt.add_argument("--tiles", type=int, default=None)
    p_opt.add_argument("--cache", default=None)
    p_opt.add_argument("--out", default=None)

    p_val = sub.add_parser("validate", help="只做结构校验")
    common(p_val)

    args = ap.parse_args(argv)
    if args.cmd == "validate":
        return _cmd_validate(args)
    if args.cmd == "e0":
        return _cmd_e0(args)
    if args.cmd == "e34":
        return _cmd_e34(args)
    if args.cmd == "e5":
        return _cmd_e5(args)
    if args.cmd == "e9":
        return _cmd_e9(args)
    if args.cmd == "optimize":
        return _cmd_optimize(args)
    return _cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
