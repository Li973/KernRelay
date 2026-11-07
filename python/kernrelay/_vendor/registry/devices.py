"""npu-smi 物理设备号 ↔ ACL 逻辑设备号 的解析与映射。

背景：``npu-smi info`` 显示的是**物理设备号**（本机容器：0,5,6,7），而 ACL 的
``ASCEND_RT_VISIBLE_DEVICES`` / ``aclrtSetDevice`` 使用**容器可见设备列表中的
索引**（逻辑号 0-3）。两者在多容器/多机环境下通常不一致，直接传物理号会得到
``ACL_ERROR_RT_NO_DEVICE (207004)``。

``--devices`` 因此两种号都接受：

* 传入的号全部出现在可见物理列表中 → 按**物理号**映射为逻辑索引
  （如物理 ``6,7`` → 逻辑 ``2,3``）；
* 否则 → 视为**逻辑号**直接使用（如 ``0,1`` / ``2,3``）。
  当两种解释结果相同时（如单卡容器）映射无副作用。

可见物理列表优先取**驱动**枚举（``libascend_hal.so`` 的 ``drvGetDevNum`` +
``drvDeviceGetPhyIdByIndex``，其顺序即 ACL 逻辑号顺序）；驱动不可用时退回
npu-smi。宿主热插拔会让两者短暂不一致（实测 NPU 4/5 反复上下线导致中期
``aclrtSetDevice`` 107001），驱动是权威。

无法解析（未安装/无权限）时退化为按逻辑号直传。
"""

from __future__ import annotations

import ctypes
import re
import subprocess


def parse_npu_smi(text: str) -> list:
    """从 npu-smi info 文本解析可见物理设备号（升序去重）。

    识别信息表格中「设备行」：``| <id> <型号> | <Health> |``。排除含 PCI
    Bus-Id（形如 0000:C1:00.0）的 chip 行与 Process 区表头。
    """
    ids = []
    for line in text.splitlines():
        if ":" in line:            # chip 行含 Bus-Id（0000:xx:xx.x），排除
            continue
        m = re.match(r"^\|\s+(\d+)\s+\S+", line)
        if m:
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def driver_physical_devices() -> list:
    """驱动当前枚举的物理设备号（按 ACL 逻辑号顺序）；失败返回 []。"""
    try:
        hal = ctypes.CDLL("libascend_hal.so")
        num = ctypes.c_uint32(0)
        if hal.drvGetDevNum(ctypes.byref(num)) != 0 or num.value == 0:
            return []
        phys = []
        for i in range(num.value):
            pid = ctypes.c_uint32(0)
            if hal.drvDeviceGetPhyIdByIndex(ctypes.c_uint32(i),
                                            ctypes.byref(pid)) != 0:
                return []
            phys.append(int(pid.value))
        return phys
    except (OSError, AttributeError):
        return []


def visible_physical_devices() -> list:
    """容器可见物理设备号（驱动优先，npu-smi 兜底）；均失败返回 []。"""
    phys = driver_physical_devices()
    if phys:
        return sorted(phys)
    try:
        proc = subprocess.run(["npu-smi", "info"], capture_output=True,
                              text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return parse_npu_smi(proc.stdout)


def resolve_devices(spec: str, phys: list = None) -> str:
    """把 --devices 参数解析为 ASCEND_RT_VISIBLE_DEVICES 逻辑号串（升序）。

    spec: 逗号分隔的设备号，物理号或逻辑号均可（见模块 docstring）。
    phys: 可见物理号列表；None 时自动调 npu-smi。仅供单测注入。
    """
    ids = [int(x) for x in str(spec).replace(" ", "").split(",") if x != ""]
    if not ids:
        raise ValueError("--devices 为空")
    if phys is None:
        phys = visible_physical_devices()
    if phys and set(ids) <= set(phys):
        # 全部是可见物理号 -> 映射为逻辑索引（升序，规避非升序 NO_DEVICE）
        logical = sorted(phys.index(d) for d in set(ids))
    else:
        # 视为逻辑号；同时也防呆：逻辑号必须落在 [0, len(phys)) 内
        if phys and any(d >= len(phys) for d in ids):
            raise ValueError(
                "--devices %s 无效：可见物理号 %s（传物理号），"
                "或逻辑号 0-%d（传逻辑号）" % (spec, phys, len(phys) - 1))
        logical = sorted(set(ids))
    return ",".join(str(d) for d in logical)
