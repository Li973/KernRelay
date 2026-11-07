"""Planner 鲁棒性：只记录问题 DAG 是否被拒绝，不记录耗时。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from kernrelay.planner_bench import problem_graphs, validate_graph  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/planner_robustness"))
    args = ap.parse_args()
    rows = []
    # 合法 DAG：必须接受（结构层面）；问题 DAG：必须拒绝。
    for name in ("e0_tiny_mm_gelu", "e3_ln_mm", "e9_mixed"):
        p = ROOT / "dags" / (name + ".json")
        if not p.exists():
            continue
        g = json.loads(p.read_text(encoding="utf-8"))
        errs = validate_graph(g)
        rows.append({"name": name, "expected": "accept",
                     "accepted": not errs, "reason": "; ".join(errs)})
    for name, g, _desc in problem_graphs():
        errs = validate_graph(g)
        # aicpu_producer / tiny_matmul 结构合法但无正收益切分：应接受并 0 切分。
        expect = "accept(0-split)" if name in ("aicpu_producer", "tiny_matmul") else "reject"
        rows.append({"name": name, "expected": expect,
                     "accepted": not errs, "reason": "; ".join(errs)})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    lines = ["# planner 鲁棒性（只判拒绝，不记耗时）", "",
             "| DAG | 期望 | 结果 | 原因 |", "|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %s | %s |" % (
            r["name"], r["expected"],
            "ACCEPT" if r["accepted"] else "REJECT", r["reason"]))
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for r in rows:
        print("%-20s expected=%-6s %s %s" % (
            r["name"], r["expected"],
            "ACCEPT" if r["accepted"] else "REJECT", r["reason"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
