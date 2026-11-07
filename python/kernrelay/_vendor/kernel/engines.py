"""物理引擎清单（NeutronHP kernel 层）。

逻辑 engine 标签（kernel/ops.py 的 ``OpSpec.engine``）只是历史成本模型口径，
**不是**真实物理占位。真实占位以 ``registry/engine_map.json`` 为准：
由 msprof 的 Task Type 实测 + 自定义 kernel 的编译期 ``KERNEL_TASK_TYPE_DEFAULT``
元数据共同确定。

物理类别（physical）与执行流（stream）的映射::

    aicore_mix -> aicore   MIX 任务（保守视为占用全部 AICore）
    aic        -> aic      纯 AIC
    aiv        -> aiv      纯 AIV
    aicpu      -> aicpu    纯 AICPU
    hccl       -> hccl     通信

未登记 / 未测量的算子按 ``defaults.unknown_op``（默认 aicore_mix）保守处理，
即占用全部 AICore、不与纯 AIC/AIV 重叠，直到真机测量后再放开。

``unstable`` 名单（manifest 字段，或 ``NHP_UNSTABLE_OPS`` 环境变量追加）内的
算子物理占位在并发下可能漂移：调度/固定计划必须让它们独占执行（不参与任何
并发）。当前先假设全部算子物理引擎固定，名单为空。
"""

import json
import os
from pathlib import Path

ALL_ENGINES = ["aicore", "aic", "aiv", "aicpu", "hccl"]

PHYSICAL_ENGINES = ("aicore_mix", "aic", "aiv", "aicpu", "hccl")

STREAM_OF_PHYSICAL = {
    "aicore_mix": "aicore",
    "aic": "aic",
    "aiv": "aiv",
    "aicpu": "aicpu",
    "hccl": "hccl",
}

TASK_TYPE_TO_PHYSICAL = {
    "AI_CORE": "aic",
    "AI_VECTOR_CORE": "aiv",
    "AICPU": "aicpu",
    "AI_CPU": "aicpu",
    "MIX_AIC": "aicore_mix",
    "MIX_AIV": "aicore_mix",
    "MIX_AICORE": "aicore_mix",
    "MIX_VECTOR_CORE": "aicore_mix",
    "COMMUNICATION": "hccl",
    "HCCL": "hccl",
}

ENGINE_MAP_PATH = Path(__file__).resolve().parent.parent / "registry" / "engine_map.json"

_DEFAULT_PHYSICAL = "aicore_mix"
_CACHE = {}


def load_engine_map(path=None, reload: bool = False) -> dict:
    p = Path(path) if path else ENGINE_MAP_PATH
    key = str(p)
    if not reload and key in _CACHE:
        return _CACHE[key]
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    for name, entry in data.get("ops", {}).items():
        phys = entry.get("physical", _DEFAULT_PHYSICAL)
        if phys not in PHYSICAL_ENGINES:
            raise ValueError("engine_map: %s 物理类别非法 %s" % (name, phys))
    _CACHE[key] = data
    return data


def default_physical() -> str:
    return load_engine_map().get("defaults", {}).get("unknown_op", _DEFAULT_PHYSICAL)


def engine_entry(op: str, path=None) -> dict:
    data = load_engine_map(path)
    entry = data.get("ops", {}).get(op)
    if entry is None:
        return {"physical": default_physical(), "source": "default-unknown",
                "stable": True, "task_types": []}
    return dict(entry)


def physical_of(op: str, path=None) -> str:
    phys = engine_entry(op, path).get("physical", default_physical())
    return phys if phys in PHYSICAL_ENGINES else default_physical()


def stream_of(op: str, path=None) -> str:
    return STREAM_OF_PHYSICAL[physical_of(op, path)]


def task_type_to_physical(task_type: str) -> str:
    return TASK_TYPE_TO_PHYSICAL.get(str(task_type).strip().upper(), _DEFAULT_PHYSICAL)


def unstable_ops(path=None) -> set:
    names = set(load_engine_map(path).get("unstable", []))
    env = os.environ.get("NHP_UNSTABLE_OPS", "")
    names.update(x.strip() for x in env.split(",") if x.strip())
    return names


def is_unstable(op: str, path=None) -> bool:
    return op in unstable_ops(path)
