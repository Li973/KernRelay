"""Run kr_qwen modes across several processes and aggregate D12-style stats."""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin" / "kr_qwen"

MODES = ["serial", "serial_fused", "serial_graph", "graph_whole",
         "relay_event", "relay_flag", "relay_flag_graph"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "results" / "qwen_cpp" / Path(args.model).resolve().parent.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"model": args.model, "seq": args.seq, "procs": args.procs,
               "iters": args.iters, "warmup": args.warmup, "modes": {}}
    for mode in args.modes.split(","):
        runs = []
        for p in range(args.procs):
            out = out_dir / ("%s_p%d.json" % (mode, p))
            cmd = [str(BIN), "--model", args.model, "--mode", mode,
                   "--seq", str(args.seq), "--device", str(args.device),
                   "--warmup", str(args.warmup), "--iters", str(args.iters),
                   "--out", str(out)]
            env = dict(os.environ)
            r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                               text=True, env=env)
            line = [l for l in r.stdout.splitlines() if l.startswith("{")]
            if r.returncode != 0 or not line:
                print("FAIL", mode, p, r.returncode, r.stdout[-400:],
                      r.stderr[-400:])
                runs.append({"error": r.stdout[-400:] + r.stderr[-400:]})
                continue
            data = json.loads(line[-1])
            data["proc"] = p
            runs.append(data)
            print("ok", mode, p, "best", round(data["best_us"], 1),
                  "typ", round(data["typical_us"], 1),
                  "argmax", data["argmax"], "timeouts", data["flag_timeouts"])
        clean = [r for r in runs if "best_us" in r]
        entry = {"runs": runs}
        if clean:
            entry["best_us"] = min(r["best_us"] for r in clean)
            entry["typical_us"] = statistics.median(
                r["typical_us"] for r in clean)
            entry["all_samples"] = [s for r in clean for s in r["samples_us"]]
            if "host_best_us" in clean[0]:
                entry["host_best_us"] = min(r["host_best_us"] for r in clean)
                entry["host_typical_us"] = statistics.median(
                    r["host_typical_us"] for r in clean)
            entry["argmax_ok"] = all(r["argmax"] == r["ref_argmax"] for r in clean)
            entry["timeouts"] = sum(r["flag_timeouts"] for r in clean)
        summary["modes"][mode] = entry
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
