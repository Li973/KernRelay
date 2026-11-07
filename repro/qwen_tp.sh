#!/usr/bin/env bash
# 双卡 TP-MLP 的 KernRelay 归因 A/B：
#   tp2        attention 复制、MLP 各半；每层 down 后整幅 HCCL AllReduce
#   tp2_split  down 按行切 tiles；AllReduce/AddNorm 逐 tile 放行
# 两者设备数、权重切分、通信总量相同，只差调度。
set -euo pipefail
source "$(dirname "$0")/common.sh"
QWEN_DIR="$KERNRELAY_CACHE/qwen2.5-7b"
DEV0=1
DEV1=2
TILES=2
PROCS=3
ITERS=3
WARMUP=1
MODES="tp2,tp2_split"
SEQ=2048
OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --qwen-dir) QWEN_DIR="$2"; shift 2 ;;
        --dev0) DEV0="$2"; shift 2 ;;
        --dev1) DEV1="$2"; shift 2 ;;
        --devices) DEV0="${2%%,*}"; DEV1="${2##*,}"; shift 2 ;;
        --tiles) TILES="$2"; shift 2 ;;
        --procs) PROCS="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --modes) MODES="$2"; shift 2 ;;
        --seq) SEQ="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        -h|--help) echo "用法: $0 [--qwen-dir DIR] [--dev0 1 --dev1 2] [--tiles 2] [--procs 3] [--out DIR]"; usage_common; exit 0 ;;
        *) die "未知参数 $1" ;;
    esac
done
KR="$(require_model "$QWEN_DIR")"
TAG="$(model_tag "$QWEN_DIR")"
OUT=${OUT:-$KR_ROOT/results/qwen_tp/${TAG}_t${TILES}}
"$PY" "$KR_ROOT/scripts/run_qwen_tp.py" --model "$KR" --seq "$SEQ" \
    --dev0 "$DEV0" --dev1 "$DEV1" --tiles "$TILES" --modes "$MODES" \
    --procs "$PROCS" --iters "$ITERS" --warmup "$WARMUP" --out-dir "$OUT"
