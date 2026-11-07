#!/usr/bin/env bash
# 把 HF Qwen2.5 目录导出成 KernRelay 运行时格式（weights.bin/manifest/ids/ref）。
# 例: repro/export_qwen.sh --model /data/qwen2.5-7b --seq 2048
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODEL=""
SEQ=2048
MAX_POS=4096
OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --seq) SEQ="$2"; shift 2 ;;
        --max-pos) MAX_POS="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        -h|--help) echo "用法: $0 --model <HF目录> [--seq 2048] [--max-pos 4096] [--out DIR]"; exit 0 ;;
        *) die "未知参数 $1" ;;
    esac
done
[ -n "$MODEL" ] || die "--model 必填"
[ -f "$MODEL/config.json" ] || die "$MODEL/config.json 不存在"
OUT=${OUT:-$KERNRELAY_CACHE/$(basename "$MODEL")}
"$PY" "$KR_ROOT/scripts/export_model.py" --model "$MODEL" --seq "$SEQ" \
    --max-pos "$MAX_POS" --out "$OUT"
echo "导出完成: $OUT"
