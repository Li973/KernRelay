#!/usr/bin/env bash
# 双卡 PP=2 + HCCL Send/Recv 的 Qwen2.5 端到端（通过 libkernrelay.so 外部调用）。
# pp_serial 是真串行基线（第 r 个请求等第 r-1 个端到端完成）；pp 是流水重叠。
set -euo pipefail
source "$(dirname "$0")/common.sh"
QWEN_DIR="$KERNRELAY_CACHE/qwen2.5-0.5b"
DEV0=1
DEV1=2
REQUESTS=8
PROCS=3
ITERS=5
WARMUP=2
MODES="pp_serial,pp"
SEQ=2048
OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --qwen-dir) QWEN_DIR="$2"; shift 2 ;;
        --dev0) DEV0="$2"; shift 2 ;;
        --dev1) DEV1="$2"; shift 2 ;;
        --devices) DEV0="${2%%,*}"; DEV1="${2##*,}"; shift 2 ;;
        --requests) REQUESTS="$2"; shift 2 ;;
        --procs) PROCS="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --modes) MODES="$2"; shift 2 ;;
        --seq) SEQ="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        -h|--help) echo "用法: $0 [--qwen-dir DIR] [--dev0 1 --dev1 2] [--requests 8] [--procs 3] [--out DIR]"; usage_common; exit 0 ;;
        *) die "未知参数 $1" ;;
    esac
done
KR="$(require_model "$QWEN_DIR")"
TAG="$(model_tag "$QWEN_DIR")"
OUT=${OUT:-$KR_ROOT/results/qwen_pp/${TAG}_r${REQUESTS}}
"$PY" "$KR_ROOT/scripts/run_qwen_pp.py" --model "$KR" --seq "$SEQ" \
    --dev0 "$DEV0" --dev1 "$DEV1" --requests "$REQUESTS" --modes "$MODES" \
    --procs "$PROCS" --iters "$ITERS" --warmup "$WARMUP" --out-dir "$OUT"
