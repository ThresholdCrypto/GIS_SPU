# -*- coding: utf-8 -*-
"""PSI 结果策略（任务书 §15）：区分"协议内部泄漏"与"编译器向业务层暴露什么"。

核心纪律（不得违背）
==================
PSI 的标准语义是"接收方获得**交集本体**"。编译器不可能通过改一个开关、
或把结果包装成布尔值，就消除协议内部的这个泄漏面。因此本模块把两件事
**分开登记**：

1. ``protocol_leak``   —— 密码协议内部把什么交给了谁（任何策略都不改变）；
2. ``business_value``  —— 编译器最终向业务层暴露什么（由策略控制）。

任何输出都不得宣称"PSI 只返回布尔值"：即使策略是 ``REVEAL_BOOLEAN``，
接收方在协议内部仍然拿到了交集本体——两句话必须同时成立、同时可读。

四种策略
========
- ``REVEAL_INTERSECTION``：业务层获得交集本体（CellSetIntersect 的默认）；
- ``REVEAL_BOOLEAN``     ：业务层只获得布尔判定（Intersects / Contains 的默认）；
- ``REVEAL_COUNT``       ：业务层只获得交集基数（CellSetIntersect 可选）；
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
REVEAL_TO_REGULATOR = "REVEAL_TO_REGULATOR"

RESULT_POLICIES: tuple[str, ...] = (
    REVEAL_INTERSECTION,
    REVEAL_BOOLEAN,
    REVEAL_COUNT,
    REVEAL_TO_REGULATOR,
)

#: 协议内部泄漏面的登记码（所有策略下一致；唯一现实值是交集本体）
PROTOCOL_LEAK_INTERSECTION_BODY = "intersection-body"
PROTOCOL_LEAK_NONE = "none"

_INTERSECTION_BODY_LEAK = (
    "协议内部泄漏面（不随策略改变）：接收方在 PSI 内部获得交集本体——"
    "PSI 的标准语义，不得对外表述为“只返回布尔值”。"
)


@dataclass(frozen=True)
class ResultPolicy:
    """一种结果策略：业务层暴露什么 + 协议内部泄漏什么（分开登记）。"""

    name: str
    business_value: str  # boolean / intersection / count / report
    protocol_leak: str
    description: str
    executable: bool = True

    @property
    def disclosure(self) -> str:
        return (
            f"结果策略 {self.name}：业务层暴露 {self.business_value}；"
            + _INTERSECTION_BODY_LEAK
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "business_value": self.business_value,
            "protocol_leak": self.protocol_leak,
            "executable": self.executable,
            "disclosure": self.disclosure,
        }


_POLICY_TABLE: Mapping[str, ResultPolicy] = {
    REVEAL_INTERSECTION: ResultPolicy(
        name=REVEAL_INTERSECTION,
        business_value="intersection",
        protocol_leak=PROTOCOL_LEAK_INTERSECTION_BODY,
        description="业务层获得交集本体（接收方）；披露面与协议一致",
    ),
    REVEAL_BOOLEAN: ResultPolicy(
        name=REVEAL_BOOLEAN,
        business_value="boolean",
        protocol_leak=PROTOCOL_LEAK_INTERSECTION_BODY,
        description="业务层只获得布尔判定；协议内部仍把交集本体交给接收方",
    ),
    REVEAL_COUNT: ResultPolicy(
        name=REVEAL_COUNT,
        business_value="count",
        protocol_leak=PROTOCOL_LEAK_INTERSECTION_BODY,
        description="业务层只获得交集基数；协议内部仍把交集本体交给接收方",
    ),
    REVEAL_TO_REGULATOR: ResultPolicy(
        name=REVEAL_TO_REGULATOR,
        business_value="report",
        protocol_leak=PROTOCOL_LEAK_INTERSECTION_BODY,
        description="结果只形成给监管方的最小化报告——属部署期模式",
        executable=False,
    ),
}

#: 各算子的默认策略（与当前执行语义一一对应，保持行为不变）
DEFAULT_POLICY_BY_OP: Mapping[str, str] = {
    "Intersects": REVEAL_BOOLEAN,
    "Contains": REVEAL_BOOLEAN,
    "CellSetIntersect": REVEAL_INTERSECTION,
}

#: 各算子**允许**显式选择的策略
APPLICABLE_POLICIES_BY_OP: Mapping[str, tuple[str, ...]] = {
    "Intersects": (REVEAL_BOOLEAN,),
    "Contains": (REVEAL_BOOLEAN,),
    "CellSetIntersect": (REVEAL_INTERSECTION, REVEAL_COUNT),
}


def get_result_policy(name: str) -> ResultPolicy:
    """按名字取策略登记（不做算子适用性检查；适用性在 resolve 里做）。"""

    key = str(name).upper().strip()
    spec = _POLICY_TABLE.get(key)
    if spec is None:
        raise ValueError(f"未知结果策略 {name!r}；可用：{RESULT_POLICIES}")
    return spec


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
    if policy is None:
        name = DEFAULT_POLICY_BY_OP[op]
    else:
        name = str(policy).upper().strip()
    spec = get_result_policy(name)
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
