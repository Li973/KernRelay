"""KernRelay-owned runtime paths, vendor imports, and executor.

The project is self-contained: scheduler/model/ops/devices and the ACLNN
executor are vendored under ``python/kernrelay/_vendor`` and ``KernRelay/bin``.
No external project tree is imported at runtime.
"""

import sys
from pathlib import Path

KERNRELAY_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = Path(__file__).resolve().parent / "_vendor"
VENDOR_LIB = KERNRELAY_ROOT / "bin" / "lib"
EXEC_BIN = KERNRELAY_ROOT / "bin" / "nhp_exec"
FIXED_DAG_DIR = KERNRELAY_ROOT / "dags"

if not VENDOR_ROOT.is_dir():
    raise RuntimeError("缺少 vendor 目录：%s" % VENDOR_ROOT)
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from kernel.engines import stream_of  # noqa: E402
from registry.devices import resolve_devices  # noqa: E402
from scheduler.fixed import build_fixed_schedule, load_fixed_dag  # noqa: E402
from scheduler.model import DAGSpec, EdgeSpec, NodeSpec, load_dag  # noqa: E402

__all__ = [
    "KERNRELAY_ROOT",
    "VENDOR_ROOT",
    "VENDOR_LIB",
    "EXEC_BIN",
    "FIXED_DAG_DIR",
    "resolve_devices",
    "stream_of",
    "build_fixed_schedule",
    "load_fixed_dag",
    "load_dag",
    "DAGSpec",
    "EdgeSpec",
    "NodeSpec",
]
