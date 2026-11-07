"""Qwen TP-MLP 双卡 A/B：同一 DAG/通信量，tp2（整幅 AllReduce 串行等待）
对比 tp2_split（down 按行切 tiles，AllReduce/AddNorm 逐 tile 放行）。

每个进程只调用一次 libkernrelay（CANN 9.0 生命周期限制）。
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def model_dir(p):
    p = Path(p).resolve()
    return p / "kr" if (p / "kr" / "manifest.json").exists() else p


def run_one(args):
    from kernrelay import libapi
    tiles = args.tiles if args.tp_mode == "tp2_split" else 1
    r = libapi.qwen_tp_run(str(model_dir(args.model)), mode=args.tp_mode,
                           seq=args.seq, dev0=args.dev0, dev1=args.dev1,
                           tiles=tiles, warmup=args.warmup, iters=args.iters)
    print(json.dumps(r, ensure_ascii=False))
    return 0 if r.get("_rc") == 0 else 1


def worker_cmd(args, mode, tiles):
    return [sys.executable, str(Path(__file__).resolve()), "--worker",
            "--model", args.model, "--tp-mode", mode, "--seq", str(args.seq),
            "--dev0", str(args.dev0), "--dev1", str(args.dev1),
            "--tiles", str(tiles), "--warmup", str(args.warmup),
            "--iters", str(args.iters)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--dev0", type=int, default=1)
    ap.add_argument("--dev1", type=int, default=2)
    ap.add_argument("--modes", default="tp2,tp2_split")
    ap.add_argument("--tiles", type=int, default=2)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--tp-mode", default="tp2", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        return run_one(args)

    tag = Path(args.model).resolve().name
    if tag == "kr":
        tag = Path(args.model).resolve().parent.name
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "results" / "qwen_tp" / ("%s_t%d" % (tag, args.tiles)))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"model": str(model_dir(args.model)), "seq": args.seq,
               "devices": [args.dev0, args.dev1], "tiles": args.tiles,
               "procs": args.procs, "iters": args.iters,
               "warmup": args.warmup, "modes": {}}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        runs = []
        for p in range(args.procs):
            out = out_dir / ("run_p%d_%s.json" % (p, mode))
            cmd = worker_cmd(args, mode, args.tiles)
            payload = None
            for attempt in range(3):
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      cwd=str(ROOT),
                                      env={**os.environ,
                                           "PYTHONPATH": str(ROOT / "python")})
                payload = None
                for line in reversed(proc.stdout.strip().splitlines()):
                    try:
                        payload = json.loads(line)
                        break
                    except Exception:
                        continue
                if payload is not None and payload.get("_rc") == 0:
                    break
                msg = (payload or {}).get("error") or proc.stderr[-500:]
                payload = {"_rc": proc.returncode, "error": msg}
                if "hccl" not in msg.lower() and "aclInit" not in msg:
                    break
                time.sleep(5)
            payload["attempts"] = attempt + 1
            payload["proc"] = p
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            runs.append(payload)
        good = [r for r in runs if r.get("_rc") == 0]
        bests = [r["wall_best_us"] for r in good]
        mins = [min(r["wall_samples_us"]) for r in good]
        entry = {
            "runs": runs,
            "best_us": min(bests) if bests else None,
            "typical_us": statistics.median(mins) if mins else None,
            "argmax": [s.get("argmax") for r in good for s in r.get("stages", [])][:1],
            "maxdiff": [s.get("maxdiff") for r in good for s in r.get("stages", [])][:1],
        }
        summary["modes"][mode] = entry
        print("%-10s best %9.1f us  typical %9.1f us  argmax=%s"
              % (mode, entry["best_us"] or -1, entry["typical_us"] or -1,
                 entry["argmax"]))
    a = summary["modes"].get("tp2", {}).get("best_us")
    b = summary["modes"].get("tp2_split", {}).get("best_us")
    if a and b:
        summary["split_speedup"] = a / b
        print("tp2_split / tp2 = %.3fx" % (a / b))
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("写入:", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
