"""geo_privacy 的 `geo` 门面：地理业务开发者唯一需要接触的 API。

用法（业务侧不需要知道 jax / spu / mpc 的存在）：

    from geo_privacy import geo

    def check_conflict(route, no_fly_zone):
        return geo.intersects(route, no_fly_zone)

    def check_distance(p1, p2, threshold):
        return geo.distance_le(p1, p2, threshold)

这些函数有两重身份：
    1. 直接调用时：执行明文参考语义（用于业务自测与交叉验证）；
    2. 被编译器分析源码时：作为算子标记被识别（frontend 按 AST 匹配，不执行）。

因此实现里"返回什么"不重要，重要的是"调用形态"。
"""

from __future__ import annotations

from typing import Any

from .core import CellSet, QuantVector, TimeInterval, quantize

__all__ = [
    "geo",
    "CellSet",
    "QuantVector",
    "TimeInterval",
    "quantize",
    "GEO_OPERATIONS",
]


#: AST 识别器与语义层共享的算子名表。改这里等于改方言。
GEO_OPERATIONS: dict[str, dict[str, Any]] = {
    "intersects": {
        "op": "Intersects",
        "arity": 2,
        "geotypes": ("EntitySet", "EntitySet"),
        "returns": "Relation",
        "output_geo_type": "Relation",
        "predicate": "Intersects",
    },
    "contains": {
        "op": "Contains",
        "arity": 2,
        "geotypes": ("EntitySet", "EntitySet"),
        "returns": "Relation",
        "output_geo_type": "Relation",
        "predicate": "Contains",
    },
    "distance_le": {
        "op": "DistanceLE",
        "arity": 3,
        "geotypes": ("Vector", "Vector", "Scalar"),
        "returns": "Relation",
        "output_geo_type": "Relation",
        "predicate": "DistanceLE",
    },
    "cellset_intersect": {
        "op": "CellSetIntersect",
        "arity": 2,
        "geotypes": ("CellSet", "CellSet"),
        # returns 是该算子**供应**给业务侧的东西（关系），
        # 但它在 Geo-IR 里产出的是格网集合本体——下游算子按
        # CellSet 消费它。两者不能混为一谈：旧版一律按 returns
        # 当成 Relation，于是"格网集合喂给定点向量"这类链式类型
        # 错误一路无诊断地通过。
        "returns": "Relation",
        "output_geo_type": "CellSet",
        "predicate": "CellSetIntersect",
    },
    "weighted_sum": {
        "op": "WeightedSum",
        "arity": 2,
        "geotypes": ("Vector", "Vector"),
        "returns": "Scalar",
        "output_geo_type": "Scalar",
        "predicate": "WeightedSum",
    },
    "temporal_overlap": {
        "op": "TemporalOverlap",
        "arity": 2,
        "geotypes": ("TimeInterval", "TimeInterval"),
        "returns": "Relation",
        "output_geo_type": "Relation",
        "predicate": "TemporalOverlap",
    },
    # ---- 物化算子：把 (XY 单元, 高度带) 展开成 3D 层集合 ----
    #
    # 它不是隐私算子——层集合的编码在本方明文完成，密态边界落在后续的 PSI 交集上。
    # 之所以仍然登记：未登记时前端会把整条调用**静默丢弃**（README 7.1.1），
    # 编译照样报成功，高度带却根本没进 Geo-IR。那是"假验证"，
    # 比报一条"算子不支持"危险得多。
    "height_band": {
        "op": "HeightBand",
        "arity": 5,
        "materializes": True,
        #: 进 Geo-IR inputs：被编码的数据（XY 单元索引）
        "value_params": ("x", "y"),
        #: 进 Geo-IR params：高度带与层级。它们是被物化的**配置**，
        #: 不是被计算的数据；混进 inputs 会让敏感度判定拿配置去算密态需求。
        "config_params": ("height_min", "height_max", "level", "toff", "lt"),
        #: 缺一不可的实参（toff / lt / ver 有默认值）
        "required_params": ("x", "y", "height_min", "height_max", "level"),
        #: 与 _GeoFacade.height_band 的默认值逐字一致（tests 里有对拍，防止两处漂移）
        "defaults": {"toff": 0, "lt": 0},
        "geotypes": ("Scalar", "Scalar", "Scalar", "Scalar", "Scalar"),
        "returns": "CellSet",
        "output_geo_type": "CellSet",
    },
}


class _GeoFacade:
    """`geo` 单例。所有方法都是算子标记 + 明文参考实现。"""

    # ---------------- 空间 ----------------

    def intersects(self, left: Any, right: Any) -> bool:
        """两个格网集合是否存在公共格网 → PSI。"""

        if isinstance(left, CellSet) and isinstance(right, CellSet):
            return left.intersects(right)
        return _plain_intersects(left, right)

    def contains(self, outer: Any, inner: Any) -> bool:
        """outer 是否包含 inner 的全部格网 → PSI/MPC。"""

        if isinstance(outer, CellSet) and isinstance(inner, CellSet):
            return outer.contains(inner)
        return _plain_contains(outer, inner)

    def cellset_intersect(self, left: Any, right: Any) -> CellSet:
        """格网集合求交，返回交集本身（不只是布尔）→ PSI。"""

        if isinstance(left, CellSet) and isinstance(right, CellSet):
            return left.intersection(right)
        raise TypeError("cellset_intersect 的输入必须是 CellSet")

    # ---------------- 属性 ----------------

    def distance_le(self, left: Any, right: Any, threshold: Any) -> bool:
        """量化向量距离是否不超过阈值 → MPC/SPU。"""

        if isinstance(left, QuantVector) and isinstance(right, QuantVector):
            return left.distance_le(right, int(threshold))
        return _plain_distance_le(left, right, threshold)

    def weighted_sum(self, values: Any, weights: Any) -> int:
        """定点加权和 → SPU/MPC。"""

        if isinstance(values, QuantVector) and isinstance(weights, QuantVector):
            return values.weighted_sum(weights)
        return _plain_weighted_sum(values, weights)

    # ---------------- 时间 ----------------

    def temporal_overlap(self, left: Any, right: Any) -> bool:
        """两个段式时间区间是否重叠 → MPC/SPU。"""

        if isinstance(left, TimeInterval) and isinstance(right, TimeInterval):
            return left.overlap(right)
        return _plain_temporal_overlap(left, right)

    def quantize(self, value: float, vmin: float, vmax: float, bits: int = 8) -> int:
        """量化工具（明文，不进密态）。"""

        return quantize(value, vmin, vmax, bits)

    # ---------------- 高度带（GB/T 40087 附录 B） ----------------

    def height_band(
        self,
        x: int,
        y: int,
        height_min: float,
        height_max: float,
        level: int,
        toff: int = 0,
        lt: int = 0,
    ) -> CellSet:
        """把 (XY 单元, 高度带) 物化成 3D 格网集合。

        明文侧工具：层号由 GB/T 40087 式(B.7) 算出，业务侧看不到也不需要
        层号。物化完成后，高度带重叠 = 集合交，交给 geo.intersects 即可。
        """

        return CellSet.from_height_band(
            x,
            y,
            level=level,
            height_min=height_min,
            height_max=height_max,
            toff=toff,
            lt=lt,
        )


# --------------------------------------------------------------------------
# 明文参考实现：允许用户传入 numpy 数组、列表等（便于业务自测）
# --------------------------------------------------------------------------


def _as_codes(obj: Any) -> set[int]:
    if isinstance(obj, CellSet):
        return set(obj.codes)
    if hasattr(obj, "codes"):
        return set(int(c) for c in obj.codes)
    if hasattr(obj, "__iter__"):
        return set(int(c) for c in obj)
    raise TypeError(f"无法把 {type(obj).__name__} 解释为格网集合")


def _as_vector(obj: Any) -> list[int]:
    if isinstance(obj, QuantVector):
        return list(obj.values)
    if hasattr(obj, "tolist"):
        return [int(v) for v in obj.tolist()]
    if hasattr(obj, "__iter__"):
        return [int(v) for v in obj]
    raise TypeError(f"无法把 {type(obj).__name__} 解释为量化向量")


def _plain_intersects(left: Any, right: Any) -> bool:
    return not _as_codes(left).isdisjoint(_as_codes(right))


def _plain_contains(outer: Any, inner: Any) -> bool:
    return _as_codes(inner) <= _as_codes(outer)


def _plain_distance_le(left: Any, right: Any, threshold: Any) -> bool:
    left_values, right_values = _as_vector(left), _as_vector(right)
    if len(left_values) != len(right_values):
        raise ValueError(f"维度不一致：{len(left_values)} vs {len(right_values)}")
    total = sum((a - b) ** 2 for a, b in zip(left_values, right_values))
    return total <= int(threshold) ** 2


def _plain_weighted_sum(values: Any, weights: Any) -> int:
    value_list, weight_list = _as_vector(values), _as_vector(weights)
    if len(value_list) != len(weight_list):
        raise ValueError("权重维度不一致")
    return sum(w * v for w, v in zip(weight_list, value_list))


def _plain_temporal_overlap(left: Any, right: Any) -> bool:
    """真实重叠（含部分重叠）判据，与后端 Plain/JAX 实现保持一致。"""

    def spans(obj: Any) -> list[tuple[int, int]]:
        if isinstance(obj, TimeInterval):
            return obj.spans()
        result = []
        for node in obj:
            toff, lt = int(node[0]), int(node[1])
            result.append((toff, toff + (1 << lt)))
        return result

    return any(
        l_start < r_end and r_start < l_end
        for l_start, l_end in spans(left)
        for r_start, r_end in spans(right)
    )


#: 单例
geo = _GeoFacade()
