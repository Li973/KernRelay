#!/bin/bash
# Build the KernRelay C++ Qwen runtime against CANN 9.0.0 + bin/lib.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KR="$(cd "$HERE/.." && pwd)"
CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
source "$CANN/set_env.sh" >/dev/null 2>&1 || true
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I"$KR/exec" -I"$KR/exec/include/kr_kernels" \
    -I"$CANN/aarch64-linux/include" -I"$CANN/include" \
    "$KR/exec/qwen_main.cpp" -o "$KR/bin/kr_qwen" \
    -L"$KR/bin/lib" -L"$CANN/lib64" \
    -lkr_sync -lnhp_kernels -lacl_rt -lopapi -lnnopbase -lascendcl -lruntime \
    -ltiling_api -lregister -lplatform -lerror_manager -lprofapi \
    -lge_common_base -lunified_dlog -lmmpa -ldl -lascend_dump -lc_sec \
    -L"$CANN/aarch64-linux/lib64" -lhccl -lpthread \
    -Wl,-rpath,'$ORIGIN/lib' -Wl,-rpath,"$CANN/lib64"
echo "built $KR/bin/kr_qwen"
