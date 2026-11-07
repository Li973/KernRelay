#!/usr/bin/env bash
# E5：计算/通信重叠（partial allreduce 与 GEMM 并发）。
set -euo pipefail
source "$(dirname "$0")/common.sh"
DEVICES="1,2"
SAMPLES=3
ITERS=5
WARMUP=2
OUT="$KR_ROOT/results/e5_final"
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
ARGS=(--config "$KR_ROOT/configs/e5_comm.json" --devices "$DEVICES"
      --samples "$SAMPLES" --iters-per-process "$ITERS" --warmup "$WARMUP"
      --out "$OUT")
[ -n "$ONLY" ] && ARGS+=(--only "$ONLY")
"$PY" -m kernrelay e5 "${ARGS[@]}"
