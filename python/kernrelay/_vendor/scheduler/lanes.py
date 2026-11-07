"""执行 lane 归属（NeutronHP scheduler 层）。

模型必须与**执行器真实的并发面**一致，否则求解出来的"重叠收益"只是纸面收益。

``executor/exec_main.cpp`` 在物化阶段把 aic / aiv / aicpu 三条计算流按依赖拓扑
序合并为一条计算 lane（同单元多流并发在 CANN 9.0.0 上触发 507015/507035，
doc/design.md §8），只有 hccl 保持独立 lane。因此：

* 默认（``COALESCE_COMPUTE = True``）模型只有 2 个资源：``compute`` 与 ``hccl``
  —— aic×aiv 的"异构重叠"在本执行器上并不存在，不能再计入收益；
* 置 ``False`` 只用于消融对照（复现旧 S1 模型的乐观口径）。
"""

COALESCE_COMPUTE = True

LANE_COMPUTE = "compute"
LANE_HCCL = "hccl"
LANES = (LANE_COMPUTE, LANE_HCCL)

_LANE_OF_ENGINE = {"aicore": LANE_COMPUTE, "aic": LANE_COMPUTE,
                   "aiv": LANE_COMPUTE, "aicpu": LANE_COMPUTE, "hccl": LANE_HCCL}


def lane_of(engine: str) -> str:
    """引擎 -> 资源 lane（默认合并计算引擎）。"""
    if engine not in _LANE_OF_ENGINE:
        raise KeyError("未知引擎: %s" % engine)
    if COALESCE_COMPUTE:
        return _LANE_OF_ENGINE[engine]
    return engine


def is_cross_lane(engine_a: str, engine_b: str) -> bool:
    return lane_of(engine_a) != lane_of(engine_b)
