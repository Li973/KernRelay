"""Run one libkernrelay entry in a fresh process (C ABI is process-level)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from kernrelay import libapi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["single", "tp"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--dev0", type=int, default=1)
    ap.add_argument("--dev1", type=int, default=2)
    ap.add_argument("--tiles", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.kind == "single":
        r = libapi.qwen_run(args.model, args.mode or "graph_whole",
                            seq=args.seq, device=args.device,
                            warmup=args.warmup, iters=args.iters)
    else:
        r = libapi.qwen_tp_run(args.model, args.mode or "tp2_split",
                               seq=args.seq, dev0=args.dev0, dev1=args.dev1,
                               tiles=args.tiles, warmup=args.warmup,
                               iters=args.iters)
    Path(args.out).write_text(json.dumps(r, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False))
    return 0 if r.get("_rc") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
