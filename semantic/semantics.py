"""semantic：地理语义层。

这一层回答三个问题，且只回答这三个问题：
    1. 这个调用表达的是什么地理关系？（谓词）
    2. 该关系的作用域（空间 / 时间）是什么？
    3. 该关系涉及的数据有多敏感？

它不做类型检查（frontend 做），不做密码学选择（planner 做）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ir import GeoType, Sensitivity, max_sensitivity

# --------------------------------------------------------------------------
# 关系谓词表
# --------------------------------------------------------------------------

#: 算子 → 标准谓词名。谓词是 Geo-IR 关系的公共语汇。
PREDICATE_TABLE: Mapping[str, str] = {
    "Intersects": "Intersects",
    "Contains": "Contains",
    "DistanceLE": "DistanceLE",
    "CellSetIntersect": "Intersects",
    "TemporalOverlap": "TemporalOverlap",
    "WeightedSum": "WeightedSum",
}

#: 关系方向性：True 表示 subject/object 不可交换
ASYMMETRIC_PREDICATES = frozenset({"Contains", "DistanceLE", "WeightedSum"})

#: 关系的语义族，便于按族做批量代价核算
PREDICATE_FAMILY: Mapping[str, str] = {
    "Intersects": "spatial-set",
    "Contains": "spatial-set",
    "DistanceLE": "spatial-metric",
    "TemporalOverlap": "temporal-interval",
    "WeightedSum": "attribute-aggregate",
}

# --------------------------------------------------------------------------
# 作用域推断
# --------------------------------------------------------------------------

#: 常见命名约定：route_A / zone_A / A_route → 主体 A
_SUBJECT_HINTS = ("route", "path", "traj", "track", "flight", "route_a")
_OBJECT_HINTS = ("zone", "no_fly", "nfz", "area", "region", "restrict", "buffer")


def split_subject_object(inputs: Sequence[str]) -> tuple[str, str]:
    """把算子输入切分为关系的主语与宾语。

    命名约定优先于位置：`no_fly_zone` 无论出现在第几个参数，都应是宾语。
    """

    if not inputs:
        return "<none>", "<none>"
    if len(inputs) == 1:
        return inputs[0], "<literal>"

    subject, obj = inputs[0], inputs[1]
    for name in inputs:
        lowered = name.lower()
        if any(hint in lowered for hint in _OBJECT_HINTS):
            obj = name
            others = [n for n in inputs if n != name]
            if others:
                subject = others[0]
            break
    return subject, obj


def infer_spatial_scope(inputs: Sequence[str]) -> str | None:
    """从命名约定推断空间作用域。

    只在命名有明确分区语义时给出结论，否则返回 None —— 不猜。
    """

    for name in inputs:
        match = re.match(r"^(?:route|path|traj|track|flight|zone|area|region)_([A-Za-z0-9]+)$", name)
        if match:
            return match.group(1)
    return None


def infer_temporal_scope(operation_or_params: Any) -> str | None:
    """从算子参数中取显式时间作用域。"""

    params: Mapping[str, Any]
    if isinstance(operation_or_params, Mapping):
        params = operation_or_params
    else:
        params = getattr(operation_or_params, "params", {}) or {}
    for key in ("time", "time_scope", "temporal_scope", "window"):
        if key in params and params[key] is not None:
            return str(params[key])
    return None


def build_scope_string(
    spatial: str | None, temporal: str | None
) -> str | None:
    """拼出可读作用域串，例如 `A @ [2160,2164)`。"""

    if spatial is None and temporal is None:
        return None
    if spatial is None:
        return temporal
    if temporal is None:
        return spatial
    return f"{spatial} @ {temporal}"


# --------------------------------------------------------------------------
# 敏感度规则
# --------------------------------------------------------------------------


def default_sensitivity_for(geo_type: GeoType) -> Sensitivity:
    """值类型的默认敏感级别。

    标量（阈值、计数）默认公开；其余地理数据默认敏感。
    这是保守默认：宁可多算一次密态，也不要漏掉一次脱敏。
    """

    if geo_type in (GeoType.SCALAR, GeoType.BOOL):
        return Sensitivity.PUBLIC
    if geo_type is GeoType.RELATION:
        return Sensitivity.INTERNAL
    return Sensitivity.SENSITIVE


def combine_sensitivity(levels: Iterable[Sensitivity | None]) -> Sensitivity:
    """关系级敏感度 = 参与方中最敏感者。"""

    return max_sensitivity(*levels)


def policy_for_operation(op: str, input_levels: Sequence[Sensitivity]) -> dict[str, Any]:
    """给出该算子的语义策略摘要，便于报告与审计。"""

    predicate = PREDICATE_TABLE.get(op, op)
    return {
        "predicate": predicate,
        "family": PREDICATE_FAMILY.get(predicate, "unknown"),
        "asymmetric": predicate in ASYMMETRIC_PREDICATES,
        "sensitivity": str(combine_sensitivity(input_levels)),
    }


# --------------------------------------------------------------------------
# 语义解析结果
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RelationSemantics:
    """一个算子在语义层上的完整解释。"""

    op: str
    predicate: str
    subject: str
    object: str
    spatial_scope: str | None
    temporal_scope: str | None
    sensitivity: Sensitivity
    family: str
    asymmetric: bool

    @property
    def scope(self) -> str | None:
        return build_scope_string(self.spatial_scope, self.temporal_scope)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "predicate": self.predicate,
            "subject": self.subject,
            "object": self.object,
            "spatial_scope": self.spatial_scope,
            "temporal_scope": self.temporal_scope,
            "scope": self.scope,
            "sensitivity": str(self.sensitivity),
            "family": self.family,
            "asymmetric": self.asymmetric,
        }


def resolve_relation_semantics(
    op: str,
    inputs: Sequence[str],
    *,
    params: Mapping[str, Any] | None = None,
    input_levels: Sequence[Sensitivity] = (),
) -> RelationSemantics:
    """从算子与输入推导关系的完整语义。"""

    predicate = PREDICATE_TABLE.get(op, op)
    subject, obj = split_subject_object(inputs)
    spatial = infer_spatial_scope(inputs)
    if params:
        spatial = str(params.get("spatial_scope")) if params.get("spatial_scope") else spatial
    temporal = infer_temporal_scope(params or {})
    return RelationSemantics(
        op=op,
        predicate=predicate,
        subject=subject,
        object=obj,
        spatial_scope=spatial,
        temporal_scope=temporal,
        sensitivity=combine_sensitivity(input_levels),
        family=PREDICATE_FAMILY.get(predicate, "unknown"),
        asymmetric=predicate in ASYMMETRIC_PREDICATES,
    )


# --------------------------------------------------------------------------
# 算子名纠错建议
# --------------------------------------------------------------------------

#: 业务语汇 → 受支持算子的别名表。
#:
#: 这是"错误报告里给替代算子"的唯一真源（frontend 不再自带一份）。
#: 收录原则：地理业务里常见、但与已登记算子语义可对齐的说法。
ALIAS_TABLE: Mapping[str, str] = {
    # ---- 集合交族 ----
    "intersect": "Intersects",
    "intersection": "Intersects",
    "crosses": "Intersects",
    "touches": "Intersects",
    "overlaps": "Intersects",
    "overlap": "Intersects",
    "conflict": "Intersects",
    "collide": "Intersects",
    # ---- 包含族 ----
    "within": "Contains",
    "covers": "Contains",
    "inside": "Contains",
    "buffer": "Contains",
    "contains": "Contains",
    # ---- 距离族 ----
    "distance": "DistanceLE",
    "near": "DistanceLE",
    "close_to": "DistanceLE",
    "distance_lt": "DistanceLE",
    "within_distance": "DistanceLE",
    "proximity": "DistanceLE",
    "nearest": "DistanceLE",
    # ---- 格网集合族 ----
    "cell_intersect": "CellSetIntersect",
    "cells_intersect": "CellSetIntersect",
    "grid_intersect": "CellSetIntersect",
    # ---- 属性和族 ----
    "sum": "WeightedSum",
    "weighted_total": "WeightedSum",
    "score": "WeightedSum",
    "risk_score": "WeightedSum",
    "aggregate": "WeightedSum",
    "density": "WeightedSum",
    "kde": "WeightedSum",
    "estimate": "WeightedSum",
    "cluster": "WeightedSum",
    "centroid": "WeightedSum",
    # ---- 时间族 ----
    "time_overlap": "TemporalOverlap",
    "temporal_intersect": "TemporalOverlap",
    "overlaps_time": "TemporalOverlap",
    "时间重叠": "TemporalOverlap",
    "schedule": "TemporalOverlap",
    "window": "TemporalOverlap",
    # ---- 高度带物化族（三维接入层） ----
    #: 业务侧常把高度带写成 altitude / elevation / 层；这些说法物化后
    #: 由集合族算子承担判定。此前它们不在表里，用户写错名字会拿到
    #: 一条"算子不支持"却没有任何替代算子。
    "height": "HeightBand",
    "height_band": "HeightBand",
    "heightband": "HeightBand",
    "altitude_band": "HeightBand",
    "elevation_band": "HeightBand",
    "vertical_band": "HeightBand",
    "layer": "HeightBand",
    "height_layer": "HeightBand",
    "高度带": "HeightBand",
    "高度层": "HeightBand",
    "海拔带": "HeightBand",
}


def suggest_ops(name: str) -> list[str]:
    """给未识别算子名推荐替代算子（按匹配强度排序，去重）。"""

    lowered = name.lower()
    hits: list[str] = []
    for alias, target in ALIAS_TABLE.items():
        if alias in lowered or lowered in alias:
            if target not in hits:
                hits.append(target)
    return hits


def op_aliases() -> Mapping[str, str]:
    return dict(ALIAS_TABLE)
