"""Profile unique LLM operators in an isolated process (C ABI is process-level)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from kernrelay.libapi import qwen_profile_many  # noqa: E402

OPS = ("embedding", "rmsnorm", "add", "add_rmsnorm", "rope", "matmul_q",
       "matmul_k", "matmul_v", "attention", "matmul_o", "matmul_gate",
       "matmul_up", "silu", "mul", "matmul_down", "matmul_lm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    cases = [{"op": op, "k": k} for op in OPS for k in ks]
    raw = qwen_profile_many(args.model, cases, seq=args.seq, device=args.device)
    table = {}
    for c in raw.get("cases", []):
        if "best_us" not in c or "k" not in c:
            continue
        table.setdefault(c["op"], {})[int(c["k"])] = float(c["best_us"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"table": table, "raw": raw},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("profile", args.model, "ops", len(table), "->", out)


if __name__ == "__main__":
    main()
