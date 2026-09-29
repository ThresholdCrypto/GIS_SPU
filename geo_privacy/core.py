"""geo_privacy 的用户侧数据类型（明文语义）。

这些类型是"地理业务开发者真正要写的东西"：格网集合、量化整数向量、段式时间区间。
它们不依赖 jax / spu / mpc，可以脱离编译器直接跑，用于业务自测。

三维语义（低空导航场景）
------------------------
CellSet 承载的是 **64 位 3D 格网码**。同一 XY 上的不同高度是**不同的码**
（Z 位域 = GB/T 40087-2021 附录 B 的高度层号）。因此"高度带重叠"在集合层
就退化为集合交，`intersects` 不需要任何高度参数：

    Route_A = geo.height_band(x=x, y=y, height_min=0, height_max=20000,
                              level=15)                       # 11 层
    NFZ_B   = geo.height_band(x=x, y=y, height_min=5000, height_max=9000,
                              level=15)                       #  3 层
    geo.intersects(Route_A, NFZ_B)                  ->  True（公共层 3）
    Route_A.intersection(NFZ_B).cardinality()       ->  3

两种写法等价：`geo.height_band(...)` 在明文侧就是
`CellSet.from_height_band(...)`，而在编译侧它会被登记为 Geo-IR 的
`HeightBand` 物化算子（不在本方之外暴露层号，也不进密态）。

反之，若把高度带写成"区间的两端各一个码"，非整层边界的中间层会被漏掉，
得到系统性假阴性。所以高度带必须展开为**层集合**，由本模块负责。

层级受 64 位键的 L 位域限制（本版已由 4 位扩到 5 位，可编码 L0-L31）。
低空带因此可以分层：0-1000 m 在 L22 上展开为 66 层，
同一 XY 上不同高度段因此可被区分。见 README 7.1.1。
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Iterable, Sequence

# --------------------------------------------------------------------------
# 量化
# --------------------------------------------------------------------------

#: 取整规则写死为四舍六入五成双中的"半格向上"——b 取整歧义会直接改变业务判据
QUANT_ROUNDING = "half_up"


def quantize(value: float, vmin: float, vmax: float, bits: int = 8) -> int:
    """把物理量按课题口径量化为整数箱号。

        q = floor((v - min) / (max - min) * (2^b - 1) + 0.5)
    """

    if bits <= 0:
        raise ValueError("bits 必须为正整数")
    if vmax <= vmin:
        raise ValueError(f"值域非法：min={vmin} max={vmax}")
    if not vmin <= value <= vmax:
        raise ValueError(f"值 {value} 超出值域 [{vmin}, {vmax}]")
    span = vmax - vmin
    scaled = (value - vmin) / span * ((1 << bits) - 1)
    return int(scaled + 0.5)


# --------------------------------------------------------------------------
# 空间：格网集合
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSet:
    """64 位格网编码集合。

    定长键是本课题的硬约束：变长会让 PSI 出现假阴性与长度信道。
    """

    codes: frozenset[int]
    label: str | None = None

    def __init__(self, codes: Iterable[int], label: str | None = None) -> None:
        frozen = frozenset(int(c) for c in codes)
        for code in frozen:
            if not 0 <= code < (1 << 64):
                raise ValueError(f"格网编码 {code} 不是 64 位无符号整数")
        object.__setattr__(self, "codes", frozen)
        object.__setattr__(self, "label", label)

    def intersects(self, other: "CellSet") -> bool:
        """是否有公共格网。集合交 → 对应后端 PSI。"""

        return not self.codes.isdisjoint(other.codes)

    def contains(self, other: "CellSet") -> bool:
        """是否包含 another 的全部格网。"""

        return other.codes <= self.codes

    def intersection(self, other: "CellSet") -> "CellSet":
        return CellSet(self.codes & other.codes, label=f"{self.label or '?'}∩{other.label or '?'}")

    def cardinality(self) -> int:
        return len(self.codes)

    @property
    def is_empty(self) -> bool:
        return not self.codes

    @classmethod
    def from_height_band(
        cls,
        x: int,
        y: int,
        *,
        level: int,
        height_min: float,
        height_max: float,
        toff: int = 0,
        lt: int = 0,
        label: str | None = None,
    ) -> "CellSet":
        """把一个 (XY 单元, 高度带) 展开为该高度带覆盖的**全部层**的 3D 格网码。

        这是低空业务能写出三维冲突判定的关键：业务只提供"这个空域占哪个
        XY、占多少米"，不需要知道 GB 的层号、也不需要自己枚举层。
        """

        from ir import height_band_codes

        return cls(
            height_band_codes(
                x, y, level, height_min, height_max, toff=toff, lt=lt
            ),
            label=label,
        )


# --------------------------------------------------------------------------
# 属性：量化向量
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantVector:
    """量化整数向量。密态下唯一被接受的数值载体。

    明确禁止浮点与向量嵌入：浮点不可靠，嵌入可反演恢复坐标。
    """

    values: tuple[int, ...]
    label: str | None = None
    scale: int = 1

    def __init__(self, values: Sequence[int], label: str | None = None, scale: int = 1) -> None:
        normalized = tuple(int(v) for v in values)
        for value in normalized:
            if not -(1 << 63) <= value < (1 << 63):
                raise ValueError(f"量化值 {value} 超出 64 位有符号整数范围")
        object.__setattr__(self, "values", normalized)
        object.__setattr__(self, "label", label)
        if scale <= 0:
            raise ValueError("scale 必须为正整数（定点缩放因子）")
        object.__setattr__(self, "scale", int(scale))

    def distance_le(self, other: "QuantVector", threshold: int) -> bool:
        """L2 距离平方与阈值比较——避免开方，密态下不引入 sqrt 电路。"""

        if len(self.values) != len(other.values):
            raise ValueError(f"维度不一致：{len(self.values)} vs {len(other.values)}")
        if self.scale != other.scale:
            raise ValueError(f"定点 scale 不一致：{self.scale} vs {other.scale}")
        total = 0
        for left, right in zip(self.values, other.values):
            delta = left - right
            total += delta * delta
        # 阈值本身是距离，比较对象是距离平方 → 阈值平方对齐量纲
        return total <= threshold * threshold

    def weighted_sum(self, weights: "QuantVector") -> int:
        """定点加权和，保留 scale 因子后右移。"""

        if len(self.values) != len(weights.values):
            raise ValueError(f"权重维度不一致：{len(weights.values)} vs {len(self.values)}")
        acc = sum(w * v for w, v in zip(weights.values, self.values))
        return acc // weights.scale

    def l2_squared(self, other: "QuantVector") -> int:
        if len(self.values) != len(other.values):
            raise ValueError("维度不一致")
        return sum((a - b) ** 2 for a, b in zip(self.values, other.values))


# --------------------------------------------------------------------------
# 时间：段式区间
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeInterval:
    """段式时间区间：节点列表 [(Toff, Lt), ...]。

    单格网时间节点数上界 2 * Lt_max = 28，因此不逐分钟物化。
    """

    nodes: tuple[tuple[int, int], ...]
    label: str | None = None

    def __init__(self, nodes: Iterable[Sequence[int]], label: str | None = None) -> None:
        normalized: list[tuple[int, int]] = []
        for node in nodes:
            toff, lt = int(node[0]), int(node[1])
            if toff < 0:
                raise ValueError(f"Toff 必须非负，实得 {toff}")
            if lt < 0:
                raise ValueError(f"Lt 必须非负，实得 {lt}")
            if toff % (1 << lt) != 0:
                # 课题 F3 号审计发现：非对齐节点破坏唯一分解
                raise ValueError(
                    f"节点 (Toff={toff}, Lt={lt}) 未按 2^Lt 对齐，"
                    "规范分解不成立（参见课题审计发现 F3）"
                )
            normalized.append((toff, lt))
        object.__setattr__(self, "nodes", tuple(sorted(set(normalized))))
        object.__setattr__(self, "label", label)

    @classmethod
    def from_window(cls, start_min: int, end_min: int, lt_max: int = 14) -> "TimeInterval":
        """把 [start, end) 规范分解为二叉对齐节点（节点完全包含于区间）。"""

        if end_min <= start_min:
            raise ValueError("区间长度必须为正")
        nodes: list[tuple[int, int]] = []
        pos = start_min
        while pos < end_min:
            # 取满足对齐且不越界的最大层级
            level = 0
            while True:
                next_level = level + 1
                if next_level > lt_max:
                    break
                span = 1 << next_level
                if pos % span != 0 or pos + span > end_min:
                    break
                level = next_level
            nodes.append((pos, level))
            pos += 1 << level
        return cls(nodes)

    def covered_minutes(self) -> int:
        return sum(1 << lt for _, lt in self.nodes)

    def overlap(self, other: "TimeInterval") -> bool:
        """是否存在**真实重叠**（含部分重叠）。

        判据：存在一对节点满足 l.start < r.end 且 r.start < l.end。
        不能退化为"区间元组相同"——那只识别完全一致的区间。
        """

        return any(
            l_start < r_end and r_start < l_end
            for l_start, l_end in self.spans()
            for r_start, r_end in other.spans()
        )

    def spans(self) -> list[tuple[int, int]]:
        """把段式节点展开为 [start, end) 列表。"""

        return [(toff, toff + (1 << lt)) for toff, lt in self.nodes]

    def _minute_spans(self) -> frozenset[tuple[int, int]]:
        return frozenset(
            (toff, toff + (1 << lt)) for toff, lt in self.nodes
        )

    def normalize(self, lt_max: int = 14) -> "TimeInterval":
        """归并为互不相交的规范分解。"""

        if not self.nodes:
            return self
        spans = sorted(self.spans())
        merged: list[list[int]] = []
        for start, end in spans:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        nodes: list[tuple[int, int]] = []
        for start, end in merged:
            nodes.extend(TimeInterval.from_window(start, end, lt_max).nodes)
        return TimeInterval(nodes, label=self.label)


def toff_from_datetime(moment: _dt.datetime) -> int:
    """绝对时刻 → Toff（分钟）。时间树原点 = 2026-09-06T22:00:00+08:00。"""

    origin = _dt.datetime(2026, 9, 6, 22, 0, tzinfo=_dt.timezone(_dt.timedelta(hours=8)))
    if moment.tzinfo is None:
        raise ValueError("时刻必须带时区")
    return int((moment - origin).total_seconds() // 60)
