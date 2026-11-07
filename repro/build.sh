#!/usr/bin/env bash
# 重建 KernRelay 执行器、稳定 C ABI 库和 Qwen 单卡可执行文件。
set -euo pipefail
source "$(dirname "$0")/common.sh"
bash "$KR_ROOT/scripts/build_exec.sh"
bash "$KR_ROOT/scripts/build_lib.sh"
bash "$KR_ROOT/scripts/build_qwen.sh"
echo "构建完成: bin/nhp_exec  bin/libkernrelay.so  bin/kr_qwen"
