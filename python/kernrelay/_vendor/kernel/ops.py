"""算子登记（NeutronHP kernel 层）：真实实现的算子元数据。

真实实现位于 runtime/op_impl.cpp（调用 aclnn level2 高级算子 / HCCL C API）
与 kernel/（Ascend C publish_tile_flag、AICPU tile_gate）。
此处只登记 schema：引擎归属、shape 键、是否支持逐 tile 切分，
供 DAG 校验与调度器使用。耗时一律查 registry 真机测量表。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OpSpec:
    name: str
    engine: str
    shape_keys: tuple
    supports_split: bool = True
    default_tiles: int = 4
    note: str = ""

    def validate_shape(self, shape: dict) -> None:
        missing = [k for k in self.shape_keys if k not in shape]
        if missing:
            raise ValueError("算子 %s shape 缺键 %s（需要 %s）" % (self.name, missing, list(self.shape_keys)))
        for k, v in shape.items():
            if int(v) <= 0:
                raise ValueError("算子 %s shape[%s] 必须 > 0" % (self.name, k))


OP_SPECS = {
    # 调用高级算子 aclnn
    "matmul": OpSpec("matmul", "aic", ("M", "K", "N"), note="aclnnMatmul，按 M 行切"),
    "add": OpSpec("add", "aiv", ("rows", "cols"), note="aclnnAdd，按行切"),
    "gelu": OpSpec("gelu", "aiv", ("rows", "cols"), note="aclnnGelu，按行切"),
    "softmax": OpSpec("softmax", "aiv", ("rows", "cols"), note="aclnnSoftmax(dim=1)，按行切"),
    "layernorm": OpSpec("layernorm", "aiv", ("rows", "cols"), note="aclnnLayerNorm，按行切"),
    "cast": OpSpec("cast", "aiv", ("rows", "cols"), note="aclnnCast fp16->fp32，按行切；仅作 sink"),
    "topk": OpSpec("topk", "aicpu", ("rows", "cols", "k"), note="aclnnTopk，按行切"),
    "sort": OpSpec("sort", "aicpu", ("rows", "cols"), note="aclnnSort，按行切"),
    "embedding": OpSpec("embedding", "aicpu", ("rows", "cols", "vocab"),
                        note="aclnnEmbedding，按 indices 行切；indices 为节点本地张量"),
    # HCCL 真实集合通信
    "allreduce": OpSpec("allreduce", "hccl", ("rows", "cols"), note="HcclAllReduce，按行切"),
    "allgather": OpSpec("allgather", "hccl", ("rows", "cols"), note="HcclAllGather + 布局归一拷贝"),
    "reducescatter": OpSpec("reducescatter", "hccl", ("rows", "cols"),
                            note="HcclReduceScatter，rows*cols 需整除 nranks"),
    # 自定义 Ascend C 算子（kernel/custom_*.cpp），物理引擎由编译期
    # KERNEL_TASK_TYPE_DEFAULT 固定，msprof 实测写入 registry/engine_map.json。
    # 高级算子不一定含纯 AIC/AIV/AICPU 实现，这几个算子用于构造确定性的
    # 异构重叠实验（耗时由 iters/blocks 控制）。
    "custom_aiv_scale": OpSpec("custom_aiv_scale", "aiv", ("rows", "cols"),
                               note="自研纯 AIV（AIV_ONLY）向量缩放；iters 控制时长"),
    "custom_aic_mac": OpSpec("custom_aic_mac", "aic", ("rows", "cols"),
                             note="自研纯 AIC（AIC_ONLY）标量乘累加；iters 控制时长"),
    "custom_mix_scale": OpSpec("custom_mix_scale", "aicore", ("rows", "cols"),
                               note="自研 MIX（MIX_AIC_1_2）缩放；保守视为占满 AICore"),
    "custom_aicpu_scale": OpSpec("custom_aicpu_scale", "aicpu", ("rows", "cols"),
                                 note="自研纯 AICPU 标量缩放；iters 控制时长"),
}


def get_op(name: str) -> OpSpec:
    if name not in OP_SPECS:
        raise KeyError("未登记算子: %s（已删除 conv2d/transpose/alltoall，见 doc/op_analysis.md）" % name)
    return OP_SPECS[name]


def all_ops():
    return list(OP_SPECS.values())


def validate_shape(op_name: str, shape: dict) -> None:
    get_op(op_name).validate_shape(shape)
