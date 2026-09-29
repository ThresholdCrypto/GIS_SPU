"""plain backend：明文参考实现。

作用不是"跑得快"，而是提供**语义基准**：
    - JAX 实现与 SPU 实现的结果都必须与它对齐（在给定 tolerance 内）；
    - 它是用户在编译前自测业务逻辑的依据。

所有实现都是纯 Python，不导入 numpy / jax / spu。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "PLAIN_IMPLEMENTATIONS",
    "PlainResult",
    "plain_contains",
    "plain_cellset_intersect",
    "plain_distance_le",
    "plain_intersects",
    "plain_temporal_overlap",
    "plain_weighted_sum",
    "run_plain",
]


@dataclass
class PlainResult:
    """明文执行结果。"""

    op: str
    value: Any
    dtype: str
    elements: int = 0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "value": self.value,
            "dtype": self.dtype,
            "elements": self.elements,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# 逐算子明文实现
# --------------------------------------------------------------------------


def plain_intersects(left: Sequence[int], right: Sequence[int]) -> PlainResult:
    """集合交是否存在 → 布尔。"""

    a, b = set(int(x) for x in left), set(int(x) for x in right)
    return PlainResult(
        op="Intersects",
        value=bool(a & b),
        dtype="bool",
        elements=len(a) + len(b),
        notes=("集合交：密态下由 PSI 完成，明文侧仅作语义基准",),
    )


def plain_contains(outer: Sequence[int], inner: Sequence[int]) -> PlainResult:
    """子集关系 → 布尔。"""

    a, b = set(int(x) for x in outer), set(int(x) for x in inner)
    return PlainResult(
        op="Contains",
        value=bool(b <= a),
        dtype="bool",
        elements=len(a) + len(b),
        notes=("包含 = 子集关系；空集包含判定需业务侧显式约定",),
    )


def plain_cellset_intersect(left: Sequence[int], right: Sequence[int]) -> PlainResult:
    """集合交本体 → 有序 tuple。"""

    a, b = set(int(x) for x in left), set(int(x) for x in right)
    return PlainResult(
        op="CellSetIntersect",
        value=tuple(sorted(a & b)),
        dtype="u64[]",
        elements=len(a & b),
        notes=("输出交集本体，供后续链式计算使用",),
    )


def plain_distance_le(
    left: Sequence[int], right: Sequence[int], threshold: int
) -> PlainResult:
    """距离平方与阈值平方比较 → 布尔。

    刻意不做 sqrt：d 从 2 降到 1，且避开 MPC 的除法电路。
    """

    if len(left) != len(right):
        raise ValueError(f"维度不一致：{len(left)} vs {len(right)}")
    total = sum((int(a) - int(b)) ** 2 for a, b in zip(left, right))
    return PlainResult(
        op="DistanceLE",
        value=bool(total <= int(threshold) ** 2),
        dtype="bool",
        elements=len(left),
        notes=(
            f"距离平方={total}，阈值平方={int(threshold) ** 2}",
            "避免 sqrt：乘法深度 d=1",
        ),
    )


def plain_weighted_sum(
    values: Sequence[int], weights: Sequence[int], scale: int = 1
) -> PlainResult:
    """定点加权和 → 整数。

    scale 与 JAX 实现保持同一签名：三份实现（明文 / JAX / SPU）必须可直接对拍。
    """

    if len(values) != len(weights):
        raise ValueError(f"权重维度不一致：{len(weights)} vs {len(values)}")
    if int(scale) <= 0:
        raise ValueError("scale 必须为正整数（定点缩放因子）")
    total = sum(int(w) * int(v) for w, v in zip(weights, values))
    return PlainResult(
        op="WeightedSum",
        value=int(total) // int(scale),
        dtype="i64",
        elements=len(values),
        notes=(f"K={len(values)} 个属性通道", f"scale={scale}（定点右移）"),
    )


def plain_temporal_overlap(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]
) -> PlainResult:
    """段式区间是否存在**真实重叠**（含部分重叠）→ 布尔。

    重叠判据：存在一对节点满足  l.start < r.end  且  r.start < l.end。
    注意不能用区间元组的集合交来做——那只识别"完全相同"的区间。
    """

    def spans(nodes: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
        out = []
        for node in nodes:
            toff, lt = int(node[0]), int(node[1])
            out.append((toff, toff + (1 << lt)))
        return out

    a, b = spans(left), spans(right)
    overlaps = any(
        l_start < r_end and r_start < l_end
        for l_start, l_end in a
        for r_start, r_end in b
    )
    return PlainResult(
        op="TemporalOverlap",
        value=bool(overlaps),
        dtype="bool",
        elements=len(left) + len(right),
        notes=(
            f"左侧 {len(left)} 节点 / 右侧 {len(right)} 节点",
            f"逐对比较次数={len(a) * len(b)}（节点上界 2·Lt_max=28）",
        ),
    )


# --------------------------------------------------------------------------
# 分派
# --------------------------------------------------------------------------

PLAIN_IMPLEMENTATIONS: Mapping[str, Any] = {
    "Intersects": plain_intersects,
    "Contains": plain_contains,
    "CellSetIntersect": plain_cellset_intersect,
    "DistanceLE": plain_distance_le,
    "WeightedSum": plain_weighted_sum,
    "TemporalOverlap": plain_temporal_overlap,
}


def run_plain(op: str, *args: Any, **kwargs: Any) -> PlainResult:
    """按算子名调用明文实现。"""

    impl = PLAIN_IMPLEMENTATIONS.get(op)
    if impl is None:
        raise NotImplementedError(
            f"明文后端没有 {op} 的实现；可用：{sorted(PLAIN_IMPLEMENTATIONS)}"
        )
    return impl(*args, **kwargs)