"""frontend 方言表：`geo.<method>` 到 Geo-IR 算子的映射。

方言是"低门槛"的落点——只允许业务语汇，不允许密码学语汇。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ir import GeoType

from geo_privacy.geo import GEO_OPERATIONS


@dataclass(frozen=True)
class DialectEntry:
    """一个方言方法到 Geo-IR 算子的确定性映射。"""

    method: str
    op: str
    arity: int
    geotypes: tuple[str, ...]
    returns: GeoType
    #: 该算子在 Geo-IR 里**产出**的类型（供下游算子消费）。
    #: 与 returns（供应给业务侧的东西）可能不同，例如 CellSetIntersect。
    output_geo_type: GeoType | None = None
    predicate: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    #: 是否为"物化算子"：在本方明文算出格网码集合，不参与密态运算。
    #: 它的实参是命名参数（x= / height_min= ...），取值方式与位置实参族不同。
    materializes: bool = False
    #: 物化算子：哪些命名实参是"被编码的数据"（进 Geo-IR inputs）。
    value_params: tuple[str, ...] = ()
    #: 物化算子：哪些命名实参是"物化配置"（进 Geo-IR params，不参与敏感度判定）。
    config_params: tuple[str, ...] = ()
    #: 物化算子：必须给出的命名参数（无默认值的那些）。
    required_params: tuple[str, ...] = ()
    #: 物化算子：命名参数的默认值（与 facade 签名逐字一致）。
    defaults: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.returns, str):
            object.__setattr__(self, "returns", GeoType(self.returns))
        if isinstance(self.output_geo_type, str):
            object.__setattr__(self, "output_geo_type", GeoType(self.output_geo_type))


def _build_dialect() -> dict[str, DialectEntry]:
    table: dict[str, DialectEntry] = {}
    for method, spec in GEO_OPERATIONS.items():
        table[method] = DialectEntry(
            method=method,
            op=spec["op"],
            arity=spec["arity"],
            geotypes=tuple(spec["geotypes"]),
            returns=GeoType(spec["returns"]),
            output_geo_type=GeoType(spec.get("output_geo_type", spec["returns"])),
            predicate=spec.get("predicate"),
            params=dict(spec.get("params") or {}),
            materializes=bool(spec.get("materializes", False)),
            value_params=tuple(spec.get("value_params", ())),
            config_params=tuple(spec.get("config_params", ())),
            required_params=tuple(spec.get("required_params", ())),
            defaults=dict(spec.get("defaults") or {}),
        )
    return table


#: 唯一真源：与 geo_privacy.geo.GEO_OPERATIONS 同源，避免两处漂移
GEO_DIALECT: dict[str, DialectEntry] = _build_dialect()

#: 同样方法名的别名（便于业务侧写法更自然）
_ALIASES: dict[str, str] = {
    "intersect": "intersects",
    "overlap_time": "temporal_overlap",
    "time_overlap": "temporal_overlap",
    "cell_intersect": "cellset_intersect",
    "score": "weighted_sum",
    "altitude_band": "height_band",
    "elevation_band": "height_band",
    "高度带": "height_band",
}


def resolve_dialect(method: str) -> DialectEntry | None:
    """解析方法名（含别名）到方言条目。"""

    if method in GEO_DIALECT:
        return GEO_DIALECT[method]
    target = _ALIASES.get(method)
    if target:
        return GEO_DIALECT.get(target)
    return None


#: `geo` 门面上**刻意不登记**为 Geo-IR 算子的方法。
#:
#: 判据：这些方法返回的不是地理实体，而是**普通标量**（阈值、箱号），
#: 在业务代码里当场算完就用，不参与任何密态计算，因此没有算子语义。
#: 前后端必须对它们给出同一类结论——是"明文工具"，不是"算子不支持"：
#: 报 GEO_OP_UNSUPPORTED 会让合法的业务写法收到一个假错误。
#: 这条名单与下面的覆盖测试一起，保证门面上不会再有第三种状态
#: （既不是算子、又不被说明），那正是 geo.height_band 出问题时的状态。
PLAINTEXT_UTILITIES: frozenset[str] = frozenset({"quantize"})


def is_plaintext_utility(method: str) -> bool:
    return method in PLAINTEXT_UTILITIES


def dialect_ops() -> tuple[str, ...]:
    return tuple(entry.op for entry in GEO_DIALECT.values())
