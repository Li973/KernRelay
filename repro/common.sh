#!/usr/bin/env bash
# 被 repro/*.sh source：CANN 环境、Python、模型目录解析。
set -euo pipefail
REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KR_ROOT="$(cd "$REPRO_DIR/.." && pwd)"
CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}"
if [ -f "$CANN_ROOT/set_env.sh" ]; then
    source "$CANN_ROOT/set_env.sh" >/dev/null 2>&1
fi
export PYTHONPATH="$KR_ROOT/python:${PYTHONPATH:-}"
export KERNRELAY_ROOT="$KR_ROOT"
export KERNRELAY_CACHE="${KERNRELAY_CACHE:-$HOME/.cache/kernrelay}"
export LD_LIBRARY_PATH="$KR_ROOT/bin/lib:${LD_LIBRARY_PATH:-}"
if [ -x "$KR_ROOT/.venv/bin/python" ]; then
    PY="${PYTHON:-$KR_ROOT/.venv/bin/python}"
else
    PY="${PYTHON:-python3}"
fi
export PY

die() { echo "错误: $*" >&2; exit 1; }

# 接受 HF 模型目录（其下有 kr/）或已导出的 kr 目录。
model_kr() {
    local d="$1"
    if [ -f "$d/kr/manifest.json" ]; then (cd "$d/kr" && pwd); else (cd "$d" && pwd); fi
}

model_root() {
    local d
    d="$(model_kr "$1")"
    if [ "$(basename "$d")" = "kr" ]; then dirname "$d"; else echo "$d"; fi
}

model_tag() { basename "$(model_root "$1")"; }

require_model() {
    local d
    d="$(model_kr "$1")"
    [ -f "$d/manifest.json" ] || die "$d 缺少 manifest.json（先运行 repro/export_qwen.sh --model <HF目录>）"
    [ -f "$d/weights.bin" ] || die "$d 缺少 weights.bin"
    echo "$d"
}

usage_common() {
    echo "  --qwen-dir DIR   HF 模型目录或其 kr 导出目录（默认 $KERNRELAY_CACHE/qwen2.5-0.5b）"
    echo "  --out DIR        输出目录"
}
