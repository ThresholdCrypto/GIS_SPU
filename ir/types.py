"""Geo-IR 类型系统。

本模块只描述"接入层能表达什么"，不含任何密码学实现细节。

设计约束（与课题口径一致）：
- Geo-IR 不是通用 GIS 几何库，只覆盖已登记算子所需的数据形态。
- 不引入浮点坐标表示（密态下不可靠），空间一律走格网编码。
- 时间一律走 (Toff, Lt) 段式编码，不逐分钟物化。
"""

from __future__ import annotations

from enum import Enum


class GeoType(str, Enum):
    """Geo-IR 值类型。"""

    ENTITY_SET = "EntitySet"      # 格网/实体集合，如 route、no_fly_zone
    CELL_SET = "CellSet"          # 显式 64 位格网编码集合
    VECTOR = "Vector"             # 定长数值向量（已量化）
    POINT = "Point"               # 单点
    TIME_INTERVAL = "TimeInterval"  # 段式时间区间集合
    SCALAR = "Scalar"             # 标量：阈值、计数
    RELATION = "Relation"         # 关系三元组（GeoRelation）
    BOOL = "Bool"                 # 判定结果
    UNKNOWN = "Unknown"           # 静态无法判定

    def __str__(self) -> str:  # pragma: no cover - 仅便于打印
        return self.value


class Sensitivity(str, Enum):
    """数据敏感级别。决定 planner 的下限：SECRET 必须进密态，PUBLIC 可留明文。"""

    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    SECRET = "secret"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


SENSITIVITY_ORDER = {
    Sensitivity.PUBLIC: 0,
    Sensitivity.INTERNAL: 1,
    Sensitivity.SENSITIVE: 2,
    Sensitivity.SECRET: 3,
}

#: 判定为"必须进密态"的门槛
CRYPTO_REQUIRED_LEVEL = Sensitivity.SENSITIVE

#: 几何类值类型（可参与空间算子）
GEOMETRIC_TYPES = frozenset(
    {GeoType.ENTITY_SET, GeoType.CELL_SET, GeoType.POINT, GeoType.VECTOR}
)

#: 时间类值类型
TEMPORAL_TYPES = frozenset({GeoType.TIME_INTERVAL})


def sensitivity_rank(level: Sensitivity) -> int:
    """返回敏感度序（越大越敏感）。"""

    return SENSITIVITY_ORDER[level]


def max_sensitivity(*levels: Sensitivity | None) -> Sensitivity:
    """取最敏感的级别；全部为空时按 INTERNAL 处理（保守默认）。"""

    present = [lvl for lvl in levels if lvl is not None]
    if not present:
        return Sensitivity.INTERNAL
    return max(present, key=sensitivity_rank)


def requires_crypto(level: Sensitivity) -> bool:
    """该敏感级别是否必须走隐私计算后端。"""

    return sensitivity_rank(level) >= sensitivity_rank(CRYPTO_REQUIRED_LEVEL)