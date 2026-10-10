# -*- coding: utf-8 -*-
"""结果策略（任务书 §15）：把"协议内部泄漏"与"编译器向业务层暴露"分开登记。

统一入口（PSI / MPC 共用；Phase 10 从 ``backends.psi_backend.result_policy``
归并而来，旧模块保留为再导出兼容层）。核心纪律（不得违背）：

1. ``protocol_leak``   —— 密码协议内部把什么交给了谁（任何策略都不改变）；
2. ``business_value``  —— 编译器最终向业务层暴露什么（由策略控制）。

两类执行族共享同一张策略表，但**泄漏面登记不同**：

- PSI（libpsi 求交）：接收方在协议内部获得交集本体——即使策略是
  ``REVEAL_BOOLEAN``，也不得对外表述为"PSI 只返回布尔值"；
- MPC（SPU 模拟）：协议输出面是"计算结果在指定输出方揭示"，
  输入与中间值不进入输出面（半诚实模型）。这是对**本项目配置**的登记，
  不是对上游协议安全性证明的转述；**不得默认把原始计算结果广播给所有
  参与方**（任务书 §十）。

策略
====
- ``REVEAL_INTERSECTION``：业务层获得交集本体（CellSetIntersect 的默认）；
- ``REVEAL_BOOLEAN``     ：业务层只获得布尔判定（Intersects / Contains /
  DistanceLE / TemporalOverlap 的默认）；
- ``REVEAL_COUNT``       ：业务层只获得交集基数（CellSetIntersect 可选）；
- ``REVEAL_VALUE``       ：业务层获得数值聚合本身（MPC 数值档，WeightedSum 默认）；
- ``REVEAL_TO_REGULATOR``：结果只形成给监管方的最小化报告——属**部署期**模式，
  当前进程内两方模拟不提供该执行语义（显式拒绝，不伪造实现）。

策略也可用于审计演练：用 ``REVEAL_COUNT`` 代替 ``REVEAL_INTERSECTION`` 时，
业务层看到的从集合本体缩小为一个数；但 ``protocol_leak`` 一栏不会因此变化，
两栏要在同一份输出里并排出现（见 ``docs/PSI_RESULT_POLICY.md``）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

REVEAL_INTERSECTION = "REVEAL_INTERSECTION"
REVEAL_BOOLEAN = "REVEAL_BOOLEAN"
REVEAL_COUNT = "REVEAL_COUNT"
REVEAL_VALUE = "REVEAL_VALUE"
REVEAL_TO_REGULATOR = "REVEAL_TO_REGULATOR"

#: 全部已登记策略名（含 MPC 数值档与部署期档）
RESULT_POLICIES: tuple[str, ...] = (
    REVEAL_INTERSECTION,
    REVEAL_BOOLEAN,
    REVEAL_COUNT,
    REVEAL_VALUE,
    REVEAL_TO_REGULATOR,
)

#: 各族策略名清单（测试与文档按族取用；REVEAL_VALUE 只属 MPC 数值档）
PSI_POLICIES: tuple[str, ...] = (
    REVEAL_INTERSECTION,
    REVEAL_BOOLEAN,
    REVEAL_COUNT,
    REVEAL_TO_REGULATOR,
)
MPC_POLICIES: tuple[str, ...] = (
    REVEAL_BOOLEAN,
    REVEAL_VALUE,
    REVEAL_TO_REGULATOR,
)

#: 协议内部泄漏面的登记码（同一策略在不同族下码不同——族决定泄漏面）
PROTOCOL_LEAK_INTERSECTION_BODY = "intersection-body"
PROTOCOL_LEAK_OUTPUT_ONLY = "output-only"
PROTOCOL_LEAK_NONE = "none"

#: 执行族名（与 planner 的后端口径一致：PSI / MPC）
RESULT_FAMILY_PSI = "PSI"
RESULT_FAMILY_MPC = "MPC"

_PSI_LEAK_SENTENCE = (
    "协议内部泄漏面（不随策略改变）：接收方在 PSI 内部获得交集本体——"
    "PSI 的标准语义，不得对外表述为“只返回布尔值”。"
)

_MPC_LEAK_SENTENCE = (
    "协议内部输出面（本仓库按配置登记）：计算结果在指定输出方之间揭示，"
    "输入与中间值在半诚实模型下不进入输出面——这是对**本项目配置**的登记，"
    "不是对上游 SPU 协议安全性证明的转述；当前进程内模拟不建模按方隔离"
    "（输出直接返回给调用方/编译报告），真实部署必须显式指定输出方，"
    "不得默认广播给所有参与方。"
)

#: 执行族 → (泄漏码, 泄漏披露句)
_FAMILY_LEAK: Mapping[str, tuple[str, str]] = {
    RESULT_FAMILY_PSI: (PROTOCOL_LEAK_INTERSECTION_BODY, _PSI_LEAK_SENTENCE),
    RESULT_FAMILY_MPC: (PROTOCOL_LEAK_OUTPUT_ONLY, _MPC_LEAK_SENTENCE),
}


@dataclass(frozen=True)
class ResultPolicy:
    """一种结果策略（某个执行族下）：业务层暴露什么 + 协议内部泄漏什么。"""

    name: str
    business_value: str  # boolean / intersection / count / value / report
    protocol_leak: str
    description: str
    leak_sentence: str
    executable: bool = True

    @property
    def disclosure(self) -> str:
        return (
            f"结果策略 {self.name}：业务层暴露 {self.business_value}；"
            + self.leak_sentence
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "business_value": self.business_value,
            "protocol_leak": self.protocol_leak,
            "executable": self.executable,
            "disclosure": self.disclosure,
        }


@dataclass(frozen=True)
class _PolicyBase:
    """策略本体（与执行族无关的部分；泄漏面在解析时按族补上）。"""

    business_value: str
    description: str
    executable: bool = True


_POLICY_BASES: Mapping[str, _PolicyBase] = {
    REVEAL_INTERSECTION: _PolicyBase(
        business_value="intersection",
        description="业务层获得交集本体（接收方）；披露面与协议一致",
    ),
    REVEAL_BOOLEAN: _PolicyBase(
        business_value="boolean",
        description="业务层只获得布尔判定；协议内部泄漏面不随策略改变",
    ),
    REVEAL_COUNT: _PolicyBase(
        business_value="count",
        description="业务层只获得交集基数；协议内部泄漏面不随策略改变",
    ),
    REVEAL_VALUE: _PolicyBase(
        business_value="value",
        description="业务层获得数值聚合本身（MPC 数值档）；输出面登记见 reveals",
    ),
    REVEAL_TO_REGULATOR: _PolicyBase(
        business_value="report",
        description="结果只形成给监管方的最小化报告——属部署期模式",
        executable=False,
    ),
}

#: 各算子的执行族（结果策略按族决定泄漏面；算子名全局唯一）
OP_RESULT_FAMILY: Mapping[str, str] = {
    "Intersects": RESULT_FAMILY_PSI,
    "Contains": RESULT_FAMILY_PSI,
    "CellSetIntersect": RESULT_FAMILY_PSI,
    "DistanceLE": RESULT_FAMILY_MPC,
    "WeightedSum": RESULT_FAMILY_MPC,
    "TemporalOverlap": RESULT_FAMILY_MPC,
}

#: 各算子的默认策略（与当前执行语义一一对应，保持行为不变）
DEFAULT_POLICY_BY_OP: Mapping[str, str] = {
    "Intersects": REVEAL_BOOLEAN,
    "Contains": REVEAL_BOOLEAN,
    "CellSetIntersect": REVEAL_INTERSECTION,
    "DistanceLE": REVEAL_BOOLEAN,
    "WeightedSum": REVEAL_VALUE,
    "TemporalOverlap": REVEAL_BOOLEAN,
}

#: 各算子**允许**显式选择的策略
APPLICABLE_POLICIES_BY_OP: Mapping[str, tuple[str, ...]] = {
    "Intersects": (REVEAL_BOOLEAN,),
    "Contains": (REVEAL_BOOLEAN,),
    "CellSetIntersect": (REVEAL_INTERSECTION, REVEAL_COUNT),
    "DistanceLE": (REVEAL_BOOLEAN,),
    "WeightedSum": (REVEAL_VALUE,),
    "TemporalOverlap": (REVEAL_BOOLEAN,),
}

#: MPC 算子的输出面登记（`SpuRunResult.reveals` 文本）。
#: 与策略表分开：同一句"输出面"要能脱离具体策略名被引用（CLI / notes）。
MPC_OP_REVEALS: Mapping[str, str] = {
    "DistanceLE": (
        "输出面：距离判定布尔值在指定输出方揭示；输入坐标与中间差值不进输出面"
        "（半诚实模型）"
    ),
    "WeightedSum": (
        "输出面：加权和数值在指定输出方揭示；输入属性与权重不进输出面"
        "（半诚实模型）"
    ),
    "TemporalOverlap": (
        "输出面：区间重叠布尔值在指定输出方揭示；输入时段不进输出面"
        "（半诚实模型）"
    ),
}


def get_result_policy(name: str, family: str = RESULT_FAMILY_PSI) -> ResultPolicy:
    """按名字取策略登记（不做算子适用性检查；适用性在 resolve 里做）。

    ``family`` 决定泄漏面登记（PSI = 交集本体；MPC = 输出面），
    缺省 PSI 保持历史调用口径不变。
    """

    key = str(name).upper().strip()
    base = _POLICY_BASES.get(key)
    if base is None:
        raise ValueError(f"未知结果策略 {name!r}；可用：{RESULT_POLICIES}")
    leak = _FAMILY_LEAK.get(str(family).upper().strip())
    if leak is None:
        raise ValueError(f"未知执行族 {family!r}；可用：{tuple(_FAMILY_LEAK)}")
    code, sentence = leak
    return ResultPolicy(
        name=key,
        business_value=base.business_value,
        protocol_leak=code,
        description=base.description,
        leak_sentence=sentence,
        executable=base.executable,
    )


def resolve_result_policy(op: str, policy: str | None = None) -> ResultPolicy:
    """解析并校验结果策略；非法/未实现的组合在这里显式失败。

    - ``policy=None`` → 该算子的默认策略；
    - ``REVEAL_TO_REGULATOR`` → 显式拒绝（部署期模式，进程内两方模拟不提供）；
    - 策略不适用于该算子 → 拒绝并列出可用清单。
    """

    if op not in DEFAULT_POLICY_BY_OP:
        raise ValueError(
            f"算子 {op!r} 不在结果策略表中；已登记：{tuple(DEFAULT_POLICY_BY_OP)}"
        )
    family = OP_RESULT_FAMILY[op]
    if policy is None:
        name = DEFAULT_POLICY_BY_OP[op]
    else:
        name = str(policy).upper().strip()
    spec = get_result_policy(name, family)
    if not spec.executable:
        raise ValueError(
            f"{name} 属部署期模式：当前 MVP 的进程内两方模拟不提供该执行语义，"
            "已显式拒绝（不伪造实现）；设计说明见 docs/PSI_RESULT_POLICY.md"
        )
    applicable = APPLICABLE_POLICIES_BY_OP[op]
    if name not in applicable:
        raise ValueError(
            f"策略 {name} 不适用于算子 {op}；可用：{applicable}（§15）"
        )
    return spec
