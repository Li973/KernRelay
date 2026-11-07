#!/usr/bin/env bash
# E3/E4：计算/计算 1:1 与 1:N 重叠（最终 nhp_exec + GM flag / 逐 stream 图）。
set -euo pipefail
source "$(dirname "$0")/common.sh"
DEVICES="1,2"
SAMPLES=3
ITERS=5
WARMUP=2
OUT="$KR_ROOT/results/e34_final"
ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --devices) DEVICES="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        -h|--help) echo "用法: $0 [--devices 1,2] [--samples 3] [--only name] [--out DIR]"; exit 0 ;;
        *) die "未知参数 $1" ;;
    esac
done
ARGS=(--config "$KR_ROOT/configs/e34_overlap.json" --devices "$DEVICES"
      --samples "$SAMPLES" --iters-per-process "$ITERS" --warmup "$WARMUP"
      --out "$OUT")
[ -n "$ONLY" ] && ARGS+=(--only "$ONLY")
"$PY" -m kernrelay e34 "${ARGS[@]}"
