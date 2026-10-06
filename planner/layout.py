# -*- coding: utf-8 -*-
"""位平面布局（D3）：按算子的**归约轴**在 L1 / L2 / 逐点之间选，并给出条数下降。

课题口径（不是本项目自创）
--------------------------
课题的 D3 决策是"位平面双布局：L1（同格网跨属性）与 L2（跨格网位平面）按算子
归约方向选择"（README §8.4）。口径与数字出处：

- `outputs/格网数据样例_明文与密态映射_v5.json` → `cost_prediction`
- `outputs/三维格网接入层口径勘误与修正说明.docx` → 表 3-1（课题组自己的复算）

该场景（1000 候选 × 49 缓冲格网 × 4 属性 × b=8）载明四个数：

    values            = 196000          （= 候选 × 缓冲格网 × 属性）
    naive_ciphertexts = 196000          （逐点：一个值一条密文）
    L1                = 49000           （= values / 属性数，4.0×）
    L2                = 192             （= ceil(values / 8192) × 8）
    reduction_L2      = 1020.8          （= values / L2）

本模块把这三条公式参数化为 `LayoutShape`，并**逐项复算**上表的四个数
（`tests/test_layout.py` 直接读上述 JSON 核对，不另抄一份数字）。

三个必须说清楚的边界
--------------------
1. **这是结构代价（密文条数），不是实测通信量**。`docs/MPC_BENCHMARK_PROTOCOL.md`
   §8.4 结论 3 实测到"通信量对 K 线性（ABY3 × DistanceLE ~16 B/元素）"，
   支持"条数主导通信量"这个**前提**；但打包电路尚未实现，因此本模块**不对
   打包后的通信量或墙钟作任何断言**。
2. **不实现打包电路**。JAX 生成器与 SPU 执行路径一行未改：这里只回答
   "按归约轴该用哪种布局、条数会降到多少"，落地（槽位打包 + 槽内归约）属下一阶段。
3. **不给形状就不给数字**。没有 `LayoutShape` 时只做轴向决策（"该走 L1/L2"），
   `selected_ciphertexts` / `reduction` 一律为 `None`——宁可不报，也不编。

归约轴如何定
------------
| 算子 | 归约轴 | 布局 |
|---|---|---|
| `WeightedSum` | 属性轴（Σ_a w_a·x_a，同一格网内跨属性归约） | L1（同格网跨属性） |
| `DistanceLE` | 候选轴（逐候选判定后跨候选聚合） | L2（跨格网位平面） |
| `TemporalOverlap` | 候选/节点轴（事件排序归并 + 前缀扫描） | L2（跨格网位平面） |
| `Intersects` / `Contains` / `CellSetIntersect` | 无（集合交，不经 MPC 值布局） | 不适用 |
| `HeightBand` | 无（明文物化，不进密态） | 不适用 |
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .registry import OPERATOR_REGISTRY

#: 布局名（对外可见的三个取值）
LAYOUT_L1 = "L1"
LAYOUT_L2 = "L2"
LAYOUT_NAIVE = "naive"

#: 归约轴名
AXIS_ATTRIBUTE = "attribute"
AXIS_CANDIDATE = "candidate"

#: 算子 → 归约轴。None = 该算子不经 MPC 值布局（PSI 集合运算 / 明文物化）。
#: 与 `OPERATOR_REGISTRY` 的一致性由 `tests/test_layout.py` 覆盖（每个登记算子
#: 都要在这张表里出现，新增算子时这里会红，逼着做一次显式决定）。
REDUCTION_AXIS: Mapping[str, str | None] = {
    "WeightedSum": AXIS_ATTRIBUTE,
    "DistanceLE": AXIS_CANDIDATE,
    "TemporalOverlap": AXIS_CANDIDATE,
    "Intersects": None,
    "Contains": None,
    "CellSetIntersect": None,
    "HeightBand": None,
}


def next_power_of_two(n: int) -> int:
    """≥ n 的最小 2 的幂（位平面槽位按 2 的幂补齐）。"""

    if n < 1:
        raise ValueError("n 必须为正整数")
    return 1 << (n - 1).bit_length()


@dataclass(frozen=True)
class LayoutShape:
    """布局的规模形状。默认值是"单值单属性"，即退化为逐点。"""

    #: 候选数（课题场景 1000）
    candidates: int
    #: 同格网 / 缓冲格网单元数（课题场景 49）
    cells: int = 1
    #: 属性通道数（课题场景 4）
    attributes: int = 1
    #: 量化位宽 b（课题场景 8）
    bits: int = 8

    def __post_init__(self) -> None:
        for name in ("candidates", "cells", "attributes", "bits"):
            if getattr(self, name) < 1:
                raise ValueError(f"LayoutShape.{name} 必须为正整数")

    @property
    def values(self) -> int:
        """逐点布局下的值个数（= 课题口径的 `values` / `naive_ciphertexts`）。"""

        return self.candidates * self.cells * self.attributes

    @property
    def slots_per_ciphertext(self) -> int:
        """一条密文能容纳的位平面槽位数。

        课题场景载明 8192，且 `8192 = 1024 × 8`：候选轴补齐到 2 的幂（1000 → 1024）
        再乘位宽 b。本模块按这个**可复算的关系**算，不硬编 8192。
        """

        return next_power_of_two(self.candidates) * self.bits

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": self.candidates,
            "cells": self.cells,
            "attributes": self.attributes,
            "bits": self.bits,
            "values": self.values,
            "slots_per_ciphertext": self.slots_per_ciphertext,
        }


@dataclass(frozen=True)
class LayoutPlan:
    """一次布局决策：轴向、选中的布局、以及（给了形状时的）条数账。"""

    op: str
    #: 该算子的归约轴；None = 不经 MPC 值布局
    axis: str | None
    shape: LayoutShape | None
    #: 对外可见的布局选择：L1 / L2 / naive
    selected: str
    #: 条数账：没给形状时为 None（不编数字）
    naive_ciphertexts: int | None = None
    selected_ciphertexts: int | None = None
    reduction: float | None = None
    l1_ciphertexts: int | None = None
    l2_ciphertexts: int | None = None
    basis: str = ""
    caveats: tuple[str, ...] = ()

    @property
    def applies(self) -> bool:
        """该算子是否需要布局（False = 不进 MPC 值布局，`estimated_cost` 不加键）。"""

        return self.axis is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "axis": self.axis,
            "selected": self.selected,
            "naive_ciphertexts": self.naive_ciphertexts,
            "selected_ciphertexts": self.selected_ciphertexts,
            "reduction": self.reduction,
            "l1_ciphertexts": self.l1_ciphertexts,
            "l2_ciphertexts": self.l2_ciphertexts,
            "shape": self.shape.to_dict() if self.shape is not None else None,
            "basis": self.basis,
            "caveats": list(self.caveats),
        }

    def to_cost_keys(self) -> dict[str, Any]:
        """并入 `PlannedStep.estimated_cost` 的键。

        命名刻意与四量（`N_ct` / `b` / `d` / `R`）区分：`N_ct` 是**每元素条数口径**
        的描述串，这里给的是**总条数**，两者不能互相覆盖。
        """

        keys: dict[str, Any] = {
            "layout": self.selected,
            "layout_axis": self.axis,
            "layout_basis": self.basis,
        }
        if self.selected_ciphertexts is not None:
            keys["N_ct_naive"] = self.naive_ciphertexts
            keys["N_ct_layout"] = self.selected_ciphertexts
            keys["layout_reduction"] = self.reduction
        return keys


#: 逐点布局的口径说明（三种选择共用同一句，避免各写各的）
_CAVEAT_LIMITS = (
    "本项是结构代价（密文条数）预测，不是实测通信量：打包电路尚未实现，"
    "JAX 生成器与 SPU 执行路径未改动",
    "通信量 ∝ 条数 的前提有实测支持（docs/MPC_BENCHMARK_PROTOCOL.md §8.4 结论 3："
    "ABY3 × DistanceLE 约 16 B/元素），但据此外推打包后的通信量仍属未验证",
)


def _l1_ciphertexts(shape: LayoutShape) -> int:
    """L1（同格网跨属性）：把每个格网的 A 个属性并进一条密文。"""

    return math.ceil(shape.values / shape.attributes)


def _l2_ciphertexts(shape: LayoutShape) -> int:
    """L2（跨格网位平面）：`ceil(values / 槽位数) × b`（课题表 3-1 的算法）。"""

    return math.ceil(shape.values / shape.slots_per_ciphertext) * shape.bits


def plan_layout(op: str, shape: LayoutShape | None = None) -> LayoutPlan:
    """按归约轴为该算子选布局；给了形状就一并给出条数下降。

    选择规则（`reduction axis → layout`）：

    - 归约轴 = 属性轴 → 先试 **L1**；属性数 = 1 时 L1 无收益，退回逐点；
    - 归约轴 = 候选轴 → 先试 **L2**；条数不下降（小规模）时退回逐点；
    - 无归约轴（PSI 集合运算 / 明文物化）→ 不适用，`applies=False`；
    - 算子不在 `REDUCTION_AXIS` 表里 → 记 `naive` 并说明"未登记归约轴"，
      **不抛异常也不替它猜**（新增算子时靠 `tests/test_layout.py` 提醒登记）。
    """

    axis = REDUCTION_AXIS.get(op)
    caveats = _CAVEAT_LIMITS if axis is not None else ()

    if axis is None:
        known = op in REDUCTION_AXIS
        return LayoutPlan(
            op=op,
            axis=None,
            shape=shape,
            selected=LAYOUT_NAIVE,
            basis=(
                "该算子不经 MPC 值布局（集合运算由 PSI 承担 / 明文物化）"
                if known
                else f"算子 {op} 未登记归约轴，不替它猜布局（按逐点计）"
            ),
        )

    if shape is None:
        policy = LAYOUT_L1 if axis == AXIS_ATTRIBUTE else LAYOUT_L2
        return LayoutPlan(
            op=op,
            axis=axis,
            shape=None,
            selected=policy,
            basis=(
                f"归约轴为{'属性' if axis == AXIS_ATTRIBUTE else '候选'}轴 → 应走 {policy}"
                "；未给布局形状（--layout-shape / layout_shape=…），故不预测条数"
            ),
            caveats=caveats,
        )

    l1 = _l1_ciphertexts(shape)
    l2 = _l2_ciphertexts(shape)
    naive = shape.values

    policy = LAYOUT_L1 if axis == AXIS_ATTRIBUTE else LAYOUT_L2
    packed = l1 if axis == AXIS_ATTRIBUTE else l2

    if packed >= naive:
        return LayoutPlan(
            op=op,
            axis=axis,
            shape=shape,
            selected=LAYOUT_NAIVE,
            naive_ciphertexts=naive,
            selected_ciphertexts=naive,
            reduction=1.0,
            l1_ciphertexts=l1,
            l2_ciphertexts=l2,
            basis=(
                f"{policy} 在本形状下没有收益（条数 {packed} ≥ 逐点 {naive}）"
                "，退回逐点布局"
            ),
            caveats=caveats,
        )

    return LayoutPlan(
        op=op,
        axis=axis,
        shape=shape,
        selected=policy,
        naive_ciphertexts=naive,
        selected_ciphertexts=packed,
        reduction=naive / packed,
        l1_ciphertexts=l1,
        l2_ciphertexts=l2,
        basis=(
            f"归约轴为{'属性' if axis == AXIS_ATTRIBUTE else '候选'}轴 → {policy}："
            f"逐点 {naive} 条 → {packed} 条（{naive / packed:.1f}×）"
        ),
        caveats=caveats,
    )


def layout_shape_from_mapping(values: Mapping[str, Any]) -> LayoutShape:
    """从 `{candidates, cells, attributes, bits}` 构造形状（CLI / JSON 入口用）。

    未知键直接报错：静默忽略等于"设了但不生效"。
    """

    allowed = ("candidates", "cells", "attributes", "bits")
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(
            f"布局形状不认识的键 {unknown}；可用键 {list(allowed)}"
        )
    try:
        parsed = {key: int(values[key]) for key in values}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"布局形状的取值必须是整数：{exc}") from exc
    if "candidates" not in parsed:
        raise ValueError("布局形状必须给 candidates（候选数）")
    return LayoutShape(**parsed)
