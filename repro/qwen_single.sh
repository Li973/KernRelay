#!/usr/bin/env bash
# 单卡 Qwen2.5 端到端：串行 / 融合 / 单流图 / 双流 event / 双流 GM flag。
set -euo pipefail
source "$(dirname "$0")/common.sh"
QWEN_DIR="$KERNRELAY_CACHE/qwen2.5-0.5b"
DEVICE=1
SEQ=2048
PROCS=3
ITERS=5
WARMUP=2
MODES="serial,serial_fused,serial_graph,graph_whole,relay_event,relay_flag,relay_flag_graph"
OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --qwen-dir) QWEN_DIR="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --seq) SEQ="$2"; shift 2 ;;
        --procs) PROCS="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --modes) MODES="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        -h|--help) echo "用法: $0 [--qwen-dir DIR] [--device N] [--modes m1,m2] [--procs 3] [--out DIR]"; usage_common; exit 0 ;;
        *) die "未知参数 $1" ;;
    esac
done
KR="$(require_model "$QWEN_DIR")"
TAG="$(model_tag "$QWEN_DIR")"
OUT=${OUT:-$KR_ROOT/results/qwen_cpp/$TAG}
"$PY" "$KR_ROOT/scripts/run_qwen_matrix.py" --model "$KR" --seq "$SEQ" \
    --device "$DEVICE" --procs "$PROCS" --iters "$ITERS" --warmup "$WARMUP" \
    --modes "$MODES" --out-dir "$OUT"
