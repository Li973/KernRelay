#!/bin/bash
# Build libkernrelay.so: stable C ABI for external Qwen/multi-op callers.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KR="$(cd "$HERE/.." && pwd)"
CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
if [ ! -d "$CANN" ]; then CANN=/usr/local/Ascend/cann-9.0.0; fi
source "$CANN/set_env.sh" >/dev/null 2>&1 || true
mkdir -p "$KR/bin"
g++ -O2 -std=c++17 -fPIC -shared -DKR_LIBRARY -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I"$KR/include" -I"$KR/exec" -I"$KR/exec/include/nhp_kernels" \
    -I"$KR/exec/include/kr_kernels" \
    -I"$CANN/aarch64-linux/include" -I"$CANN/include" \
    "$KR/exec/exec_main.cpp" "$KR/exec/qwen_main.cpp" "$KR/exec/kr_api.cpp" \
    "$KR/exec/runtime/op_impl.cpp" \
    -o "$KR/bin/libkernrelay.so" \
    -L"$KR/bin/lib" -L"$CANN/lib64" \
    -lkr_sync -lnhp_kernels -lnhp_aicpu_gate -lnhp_aicpu_custom \
    -lacl_rt -lopapi -lnnopbase -lascendcl -lruntime -ltiling_api \
    -lregister -lplatform -lerror_manager -lprofapi -lge_common_base \
    -lunified_dlog -lmmpa -ldl -lascend_dump -lc_sec \
    -L"$CANN/aarch64-linux/lib64" -lhccl -lpthread \
    -Wl,-rpath,'$ORIGIN/lib' -Wl,-rpath,"$CANN/lib64"
echo "built $KR/bin/libkernrelay.so"
