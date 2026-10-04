"""Geo-IR 值对象：GeoEntity / GeoValue / GeoRelation。

格网编码口径（与本课题既有规范一致，勿改）：
    grid_code = X'17 | Y'17 | Z7 | L5 | Toff14 | Lt4  = 64 bit
    时间树原点 = 2026-09-06T22:00:00+08:00，dt_code = 1 min，Lt_max = 14

L 位域的扩展（本版：4 -> 5 位，见 README 7.1.1）：
    原本的 ver 位并入 L 高位（ver 恒为 0），于是层级上限从 15 抬到 31。
    4 位把层级封在 15，而 L<=15 上 0-1000 m 低空带恒为 1 层，
    三维分辨力会退化成'同 XY 交即交'。
    **布局必须与对端一致**：PSI 走 CSV 交换这些 64 位码，
    布局不同不会报错，只会静默算错。用 GRID_CODE_LAYOUT 核对。

Z 位域的语义（本版补齐，见 README 7.1.1）：
    Z 是 **GB/T 40087-2021 附录 B 的高度层号** height_index(height, level)，
    不是"整数相等比较里的一个无关分量"。把同一 XY 上不同高度层展开为不同的
    64 位码，高度参与判定就退化为集合交——PSI 可直接承担，无需额外的 MPC
    区间比较。层号 -> 物理高度区间由 ir.geosot 的等比公式给出。
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from .geosot import (
    HEIGHT_LAYER_BITS,
    HEIGHT_LAYER_MAX,
    height_index,
    height_layer_bounds,
    height_layers,
    max_supported_height,
    validate_height_layer,
)
from .types import GeoType, Sensitivity

# --------------------------------------------------------------------------
# 格网编码常量
# --------------------------------------------------------------------------

GRID_CODE_BITS: Mapping[str, int] = {
    "X": 17,
    "Y": 17,
    "Z": 7,
    "L": 5,
    "Toff": 14,
    "Lt": 4,
}
GRID_CODE_WIDTH = sum(GRID_CODE_BITS.values())  # 64
GRID_CODE_BYTES = GRID_CODE_WIDTH // 8  # 8

#: 布局的规范写法。不只是给人看的注释：PSI 走 CSV
#: 交换这些 64 位码，布局不同**不报错、只静默算错**，
#: 所以它必须是一个能被打印、能被对拍的字符串。
GRID_CODE_LAYOUT = "|".join(f"{name}{width}" for name, width in GRID_CODE_BITS.items())

#: 该布局可编码的最高层级（L 位域全 1 对应的层级）
MAX_ENCODABLE_LEVEL = (1 << GRID_CODE_BITS["L"]) - 1

#: 布局身份：跨方核对用的稳定标识。PSI 走 CSV 交换这些 64 位码，
#: 布局不同不会报错、只会静默算错——对齐必须靠显式握手，不能靠人看 README
#: （见 backends.psi_backend.input_adapter.check_layout_agreement）。
GRID_CODE_LAYOUT_ID = "geosot3d-v1-x17-y17-z7-l5-toff14-lt4"
GRID_CODE_LAYOUT_VERSION = 1


def grid_code_layout_manifest() -> dict[str, Any]:
    """当前布局的机器可读清单（跨方核对手柄）。

    每个字段都取自 GRID_CODE_BITS 本身，不是第二份手写常量：
    布局漂移会在 tests/test_geosot_layout.py 的对拍里被抓住。
    """

    return {
        "layout_id": GRID_CODE_LAYOUT_ID,
        "version": GRID_CODE_LAYOUT_VERSION,
        "bits": GRID_CODE_WIDTH,
        "x_bits": GRID_CODE_BITS["X"],
        "y_bits": GRID_CODE_BITS["Y"],
        "z_bits": GRID_CODE_BITS["Z"],
        "level_bits": GRID_CODE_BITS["L"],
        "toff_bits": GRID_CODE_BITS["Toff"],
        "lt_bits": GRID_CODE_BITS["Lt"],
    }

#: 时间树原点：Toff = 0 的锚点
TIME_TREE_ORIGIN = _dt.datetime(
    2026, 9, 6, 22, 0, 0, tzinfo=_dt.timezone(_dt.timedelta(hours=8))
)
DT_CODE_MIN = 1  # 编码时间粒度，分钟
LT_MAX = 14  # 时间维度最大层级

def _derive_shifts(bits: Mapping[str, int]) -> dict[str, int]:
    """从布局推导每段位移（最低位在右）。

    位移是布局的函数，不是第二份手写常量。这正是
    布局漂移的来源：改了 GRID_CODE_BITS 却忘了改位移，
    编码会**静默错位**而不是报错。
    """

    shift, out = 0, {}
    for name, width in reversed(list(bits.items())):
        out[name] = shift
        shift += width
    return out


# 位段位移（从最高位起），由布局推导
_SHIFT = _derive_shifts(GRID_CODE_BITS)


def encode_grid_code(x: int, y: int, z: int, level: int, toff: int, lt: int) -> int:
    """按课题口径把格网分量打包为 64 位整数。"""

    parts = {"X": x, "Y": y, "Z": z, "L": level, "Toff": toff, "Lt": lt}
    code = 0
    for name, width in GRID_CODE_BITS.items():
        value = parts[name]
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"grid_code 分量 {name} 必须是非负整数，实得 {value!r}")
        if value >= (1 << width):
            raise ValueError(
                f"grid_code 分量 {name}={value} 超出 {width} 位上限 {(1 << width) - 1}"
            )
        code |= value << _SHIFT[name]
    return code


def decode_grid_code(code: int) -> dict[str, int]:
    """拆解 64 位格网编码为各分量。"""

    if not 0 <= code < (1 << GRID_CODE_WIDTH):
        raise ValueError(f"grid_code 必须在 [0, 2^64) 内，实得 {code!r}")
    return {
        "X": (code >> _SHIFT["X"]) & ((1 << GRID_CODE_BITS["X"]) - 1),
        "Y": (code >> _SHIFT["Y"]) & ((1 << GRID_CODE_BITS["Y"]) - 1),
        "Z": (code >> _SHIFT["Z"]) & ((1 << GRID_CODE_BITS["Z"]) - 1),
        "L": (code >> _SHIFT["L"]) & ((1 << GRID_CODE_BITS["L"]) - 1),
        "Toff": (code >> _SHIFT["Toff"]) & ((1 << GRID_CODE_BITS["Toff"]) - 1),
        "Lt": (code >> _SHIFT["Lt"]) & ((1 << GRID_CODE_BITS["Lt"]) - 1),
    }


def toff_of(moment: _dt.datetime) -> int:
    """把绝对时刻换算为 Toff（分钟偏移，相对时间树原点）。"""

    if moment.tzinfo is None:
        raise ValueError("时刻必须带时区信息，否则无法与时间树原点对齐")
    delta = moment - TIME_TREE_ORIGIN
    minutes = int(delta.total_seconds() // 60)
    if minutes < 0:
        raise ValueError(f"时刻 {moment.isoformat()} 早于时间树原点，无法编码")
    return minutes


# --------------------------------------------------------------------------
# 实体与值
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GeoEntity:
    """一个地理实体：**不承载坐标**，只承载格网编码与量化的属性。

    这是刻意设计：密态下坐标与浮点属性不可靠，接入层只承认离散格网。
    """

    name: str
    entity_type: GeoType = GeoType.ENTITY_SET
    sensitivity: Sensitivity = Sensitivity.SENSITIVE
    cell_codes: tuple[int, ...] = ()
    attrs: Mapping[str, int] = field(default_factory=dict)
    time_segments: tuple[tuple[int, int, int], ...] = ()  # (Toff, Lt, mask)
    spatial_scope: str | None = None
    temporal_scope: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for code in self.cell_codes:
            decode_grid_code(code)  # 提前失败优于下游静默错误
        object.__setattr__(self, "attrs", dict(self.attrs))

    @property
    def is_geometric(self) -> bool:
        return self.entity_type in (
            GeoType.ENTITY_SET,
            GeoType.CELL_SET,
            GeoType.POINT,
            GeoType.VECTOR,
        )

    @property
    def cell_count(self) -> int:
        return len(self.cell_codes)

    def with_scope(self, spatial: str | None = None, temporal: str | None = None) -> "GeoEntity":
        return replace(
            self,
            spatial_scope=spatial if spatial is not None else self.spatial_scope,
            temporal_scope=temporal if temporal is not None else self.temporal_scope,
        )


@dataclass(frozen=True)
class GeoValue:
    """Geo-IR 中的一个值：常量、实体引用或中间结果。"""

    name: str
    geo_type: GeoType = GeoType.UNKNOWN
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    const: Any = None
    entity: GeoEntity | None = None
    source: str | None = None  # 形参名 / 上一条算子的输出名
    is_constant: bool = False

    @classmethod
    def constant(
        cls, name: str, value: Any, geo_type: GeoType = GeoType.SCALAR, sensitivity: Sensitivity = Sensitivity.PUBLIC
    ) -> "GeoValue":
        return cls(
            name=name,
            geo_type=geo_type,
            sensitivity=sensitivity,
            const=value,
            is_constant=True,
            source="<const>",
        )

    @property
    def sensitivity_ranked(self) -> Sensitivity:
        """常量一律视为 PUBLIC，变量继承其实体敏感度。"""

        if self.is_constant:
            return Sensitivity.PUBLIC
        return self.sensitivity

    @property
    def effective_sensitivity(self) -> Sensitivity:
        if self.is_constant:
            return Sensitivity.PUBLIC
        if self.entity is not None:
            return self.entity.sensitivity
        return self.sensitivity


@dataclass(frozen=True)
class GeoRelation:
    """关系对象：Geo-IR 的标准输出形态。

    Route_A | Intersects | NoFlyZone_B
    """

    subject: str
    predicate: str
    object: str
    time: str | None = None
    spatial_scope: str | None = None
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    confidence: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "time": self.time,
            "spatial_scope": self.spatial_scope,
            "sensitivity": str(self.sensitivity),
        }

    def __str__(self) -> str:
        return f"{self.subject} | {self.predicate} | {self.object}"

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


def relation_from_operation(
    subject: str,
    predicate: str,
    obj: str,
    *,
    time: str | None = None,
    spatial_scope: str | None = None,
    sensitivity: Sensitivity = Sensitivity.INTERNAL,
    confidence: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> GeoRelation:
    """构造关系的唯一入口，保证字段齐全。"""

    return GeoRelation(
        subject=subject,
        predicate=predicate,
        object=obj,
        time=time,
        spatial_scope=spatial_scope,
        sensitivity=sensitivity,
        confidence=confidence,
        meta=dict(meta or {}),
    )


def codes_to_hex(codes: Iterable[int]) -> list[str]:
    """便于打印与测试比对的定长十六进制表示。"""

    return [f"0x{c:016X}" for c in codes]


def parse_hex_codes(values: Sequence[str]) -> tuple[int, ...]:
    return tuple(int(v, 16) for v in values)

# --------------------------------------------------------------------------
# 高度层语义（GB/T 40087-2021 附录 B）
# --------------------------------------------------------------------------


def layer_codes_for_height(
    x: int, y: int, level: int, height: float, toff: int, lt: int
) -> int:
    """把一个 (X', Y', 大地高) 三元组编成 64 位格网码。

    与 ``encode_grid_code`` 的区别只有一个：Z 由大地高按 GB 式(B.7) 算出，
    而不是由调用方自行给一个整数。层号超出 Z 位域时立刻失败。
    """

    layer = height_index(height, level)
    validate_height_layer(layer, level)
    return encode_grid_code(x=x, y=y, z=layer, level=level, toff=toff, lt=lt)


def height_band_codes(
    x: int,
    y: int,
    level: int,
    height_min: float,
    height_max: float,
    toff: int = 0,
    lt: int = 0,
) -> tuple[int, ...]:
    """高度带 -> 该 XY 单元上的 3D 码集合（每层一个码）。

    这是"高度带参与集合语义"的唯一入口。低空 0-1000 m 在 L=19 上得到 9 个码
    （9 层 × 1 个 XY 单元），因此同一 XY 上高度带重叠的两条航线**天然相交**——
    不需要把高度带表示成区间参数，也不需要额外的密态区间比较。

    返回顺序按层号升序，保证同一输入给出同一元组（可复算、可对拍）。
    """

    codes = []
    for layer in height_layers(height_min, height_max, level):
        validate_height_layer(layer, level)
        codes.append(encode_grid_code(x=x, y=y, z=layer, level=level, toff=toff, lt=lt))
    return tuple(codes)


def grid_code_height_interval(code: int) -> tuple[int, int, float, float]:
    """64 位码 -> (Z 层号, 层级, 下底面大地高 m, 上底面大地高 m)。

    此前 Z 只是一个位域，取出来也无法回答"这个格网在多少米"；本函数让每个
    格网码的物理高度区间可复算，是接线到低空导航场景的最小前提。
    """

    parts = decode_grid_code(code)
    layer, level = parts["Z"], parts["L"]
    lower, upper = height_layer_bounds(layer, level)
    return layer, level, lower, upper
