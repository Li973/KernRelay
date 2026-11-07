"""TP 层的 planner 实验：down(AIC) -> allreduce(HCCL) -> rmsnorm(AIV) 链。

用实测的 allreduce 固定开销曲线喂 planner，检查它是否能在 7B 上选择切分、
在 0.5B 上拒绝切分（与 tp2 / tp2_split 的实测 A/B 对照）。
"""

import json
from pathlib import Path

from kernrelay.planner2 import Costs, greedy_plan

ROOT = Path(__file__).resolve().parents[1]

# tag, H, K(down), N(down), down_us, allreduce_us, rmsnorm_us, layers
CASES = [
    ("qwen2.5-0.5b", 896, 4864, 896, 95.0, 433.9, 30.1, 24),
    ("qwen2.5-7b", 3584, 18944, 3584, 1122.9, 1051.5, 49.0, 28),
]


def make_costs(base, H, K, N, down_us, ar_us, rms_us):
    cfg = dict(base)
    cfg["full"] = dict(base["full"])
    cfg["full"]["matmul|2048x%dx%d" % (K, N)] = down_us
    cfg["full"]["allreduce|2048x%d" % H] = ar_us
    cfg["full"]["rmsnorm|2048x%d" % H] = rms_us
    c = Costs.__new__(Costs)
    c.data = cfg
    c.full = cfg["full"]
    c.launch = cfg.get("launch_us", 8.0)
    c.bw = cfg.get("traffic_gbps", 600.0) * 1e9
    c.l2 = cfg.get("l2_mb", 32.0) * 1024 * 1024
    c.event = cfg.get("event_us", 20.0)
    c.curves = {}
    c._base_cache = {}
    c._tax_cache = {}
    # 实测：2048xH allreduce 固定开销约 280 us/tile，base 为整幅时间。
    c.curves["allreduce|2048x%d" % H] = {
        "op": "allreduce", "strategy": "row",
        "shape": {"rows": 2048, "cols": H}, "base_best_us": ar_us,
        "fit": {"slope_us_per_tile": 283.0, "intercept_us": ar_us},
    }
    return c


def layer_graph(tag, H, K, N):
    return {
        "name": "tp_layer_" + tag,
        "nodes": [
            {"name": "down", "op": "matmul", "engine": "aic",
             "shape": {"M": 2048, "K": K, "N": N}, "dtype": "bf16", "tiles": 1},
            {"name": "ar", "op": "allreduce", "engine": "hccl",
             "shape": {"rows": 2048, "cols": H}, "dtype": "bf16", "tiles": 1},
            {"name": "an", "op": "rmsnorm", "engine": "aiv",
             "shape": {"rows": 2048, "cols": H}, "dtype": "bf16", "tiles": 1},
            {"name": "qkv", "op": "matmul", "engine": "aic",
             "shape": {"M": 2048, "K": H, "N": H}, "dtype": "bf16", "tiles": 1},
        ],
        "edges": [{"src": "down", "dst": "ar"}, {"src": "ar", "dst": "an"},
                  {"src": "an", "dst": "qkv"}],
    }


def main():
    base = json.loads((ROOT / "configs/planner2_costs.json").read_text())
    out = {"cases": []}
    for tag, H, K, N, down_us, ar_us, rms_us, layers in CASES:
        costs = make_costs(base, H, K, N, down_us, ar_us, rms_us)
        g = layer_graph(tag, H, K, N)
        splits, cur, hist, base_m, decisions = greedy_plan(g, costs,
                                                           max_rounds=10)
        out["cases"].append({
            "tag": tag, "layers": layers,
            "base_layer_us": base_m,
            "planned_layer_us": cur["makespan_us"],
            "gain_layer_us": base_m - cur["makespan_us"],
            "gain_total_us": (base_m - cur["makespan_us"]) * layers,
            "splits": splits,
            "history": [{"round": h["round"], "k": h["k"],
                         "gain_us": h["gain_us"], "splits": h["splits"]}
                        for h in hist],
            "decisions": [{"consumer": d["consumer"],
                           "critical_edge": d["critical_edge"],
                           "k": d["k"], "dep_gap_us": d["dep_gap_us"],
                           "predicted_gain_us": d["predicted_gain_us"],
                           "accepted": d["accepted"]} for d in decisions],
        })
        print("%-16s splits=%s  %.1f -> %.1f us/layer  gain %.1f us total"
              % (tag, splits, base_m, cur["makespan_us"],
                 (base_m - cur["makespan_us"]) * layers))
    p = ROOT / "results" / "final" / "tp_planner" / "summary.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("写入:", p)


if __name__ == "__main__":
    main()
