#!/usr/bin/env bash
# planner 独立实验：不同 DAG 的规划耗时、计划 makespan、问题 DAG 拒绝。
set -euo pipefail
source "$(dirname "$0")/common.sh"
OUT="$KR_ROOT/results/final/planner"
REPEATS=3
MAX_ROUNDS=40
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
        -h|--help) echo "用法: $0 [--out DIR] [--repeats 3] [--max-rounds 40]"; exit 0 ;;
        *) die "未知参数 $1" ;;
    esac
done
"$PY" "$KR_ROOT/scripts/planner_robustness.py" --out "$OUT/robustness"
"$PY" "$KR_ROOT/scripts/tp_planner.py"
