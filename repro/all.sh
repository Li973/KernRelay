#!/usr/bin/env bash
# 复现全部实验。依赖位置/设备均可从命令行指定。
# 例:
#   repro/all.sh --qwen-dir /data/qwen2.5-0.5b --device 1 --devices 1,2
#   repro/all.sh --qwen-dir /data/qwen2.5-7b --device 1 --devices 1,2 --skip-split
set -euo pipefail
source "$(dirname "$0")/common.sh"
QWEN_DIR="$KERNRELAY_CACHE/qwen2.5-0.5b"
DEVICE=1
DEVICES="1,2"
REQUESTS=8
PROCS=3
SKIP_SPLIT=0
while [ $# -gt 0 ]; do
    case "$1" in
        --qwen-dir) QWEN_DIR="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --devices) DEVICES="$2"; shift 2 ;;
        --requests) REQUESTS="$2"; shift 2 ;;
        --procs) PROCS="$2"; shift 2 ;;
        --skip-split) SKIP_SPLIT=1; shift ;;
        -h|--help) echo "用法: $0 [--qwen-dir DIR] [--device 1] [--devices 1,2] [--requests 8] [--procs 3] [--skip-split]"; usage_common; exit 0 ;;
        *) die "未知参数 $1" ;;
    esac
done
bash "$REPRO_DIR/build.sh"
if [ "$SKIP_SPLIT" = 0 ]; then
    bash "$REPRO_DIR/e34_overlap.sh" --devices "$DEVICES"
    bash "$REPRO_DIR/e5_comm.sh" --devices "$DEVICES"
    bash "$REPRO_DIR/e9_dag.sh" --devices "$DEVICES"
fi
bash "$REPRO_DIR/planner.sh"
bash "$REPRO_DIR/qwen_single.sh" --qwen-dir "$QWEN_DIR" --device "$DEVICE" \
    --procs "$PROCS"
bash "$REPRO_DIR/qwen_pp.sh" --qwen-dir "$QWEN_DIR" --devices "$DEVICES" \
    --requests "$REQUESTS" --procs "$PROCS"
bash "$REPRO_DIR/qwen_tp.sh" --qwen-dir "$QWEN_DIR" --devices "$DEVICES" \
    --tiles 4 --procs "$PROCS"
echo "全部实验完成。"
