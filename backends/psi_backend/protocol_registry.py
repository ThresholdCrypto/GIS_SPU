# -*- coding: utf-8 -*-
"""统一的 PSI 协议元数据（任务书 §12）：单一登记处。

背景
====
协议事实（参与方数量 / 结果语义 / 曲线关系 / 参数模式 / 候选算子）此前分散在：

    planner/registry.py
    backends/psi_backend/capability.py
    geosecure/compiler.py（经 capability 的派生函数读取）
    backends/psi_backend/runtime.py（经 capability 的派生函数读取）

本模块成为**唯一登记处**：capability 的公开常量与函数都从这里派生，
compiler / runtime / planner 只读 capability 的公开接口。

为什么 Planner 的 `OperatorRule.protocol_candidates` 仍是字面量？
----------------------------------------------------------------
`planner` 不能在**导入期**依赖 `backends`：`backends/__init__.py` 会导入
`backends.jax_backend.codegen`，而后者在导入期导入 `planner.planner`——
模块级互相导入会形成环。因此 Planner 侧的协议候选保持字面量登记，
与本文档 `candidate_for` 的一致性由 `tests/test_protocol_registry.py`
交叉断言锁定；运行期校验路径（validate_protocol_for_operation）仍按
函数内延迟导入读取本模块。

结果语义（§11）
--------------
EXACT / APPROXIMATE / NOISY 三档。当前注册协议只有两类实例：
exact（ECDH / KKRT / RR22 / 3PC / NPC 族）与 noisy（DP）。
APPROXIMATE 已定义但暂无协议实例——将来新增近似协议时在 spec 上显式声明
`semantics`，不要再写协议专用的 if/else。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

#: 结果语义三档（§11；JSON 里以字符串带出，如 "result_semantics": "exact"）
RESULT_SEMANTICS_EXACT = "exact"
RESULT_SEMANTICS_APPROXIMATE = "approximate"
RESULT_SEMANTICS_NOISY = "noisy"
RESULT_SEMANTICS: tuple[str, ...] = (
    RESULT_SEMANTICS_EXACT,
    RESULT_SEMANTICS_APPROXIMATE,
    RESULT_SEMANTICS_NOISY,
)

#: 本项目注册表里走 PSI 的算子（与 planner PSI 族一致；交叉断言见测试）
PSI_CANDIDATE_OPS: tuple[str, ...] = ("Intersects", "Contains", "CellSetIntersect")

#: 曲线关系三分类（含义见 capability.PSI_CURVE_RELATION 的说明）
_CURVE_RELATIONS: tuple[str, ...] = ("required", "ignored", "implicit")

_COMMON_PARAMS: Mapping[str, Any] = {
    "receiver_rank": {"type": "int", "values": (0, 1), "doc": "谁拿交集（0 或 1）"},
    "broadcast_result": {"type": "bool", "doc": "是否把结果广播给双方"},
}
_RR22_PARAMS: Mapping[str, Any] = {
    **_COMMON_PARAMS,
    "low_comm_mode": {"type": "bool", "doc": "Rr22Rarams.low_comm_mode（仅 RR22 会注入）"},
}


@dataclass(frozen=True)
class PsiProtocolSpec:
    """一种 PSI 协议的完整元数据（§12 建议的形状，逐字段落地）。"""

    name: str
    world_size: int
    exact: bool
    curve_relation: str
    params_schema: Mapping[str, Any]
    candidate_for: tuple[str, ...]
    #: 显式语义档；缺省由 exact 推导（True→exact，False→noisy）。
    #: 将来出现"近似但非噪声"的协议时显式给 SEMANTICS_APPROXIMATE。
    semantics: str | None = None
    #: 协议族名（§四.1）：本类固定 "PSI"；MPC 族见
    #: backends/spu_backend/protocol_registry.MpcProtocolSpec。两族分开登记，
    #: 统一校验入口在 backends/protocol_validation.py。
    family: str = "PSI"

    def __post_init__(self) -> None:
        if self.semantics is not None and self.semantics not in RESULT_SEMANTICS:
            raise ValueError(
                f"协议 {self.name} 的 semantics={self.semantics!r} 不在 {RESULT_SEMANTICS}"
            )
        if self.curve_relation not in _CURVE_RELATIONS:
            raise ValueError(
                f"协议 {self.name} 的 curve_relation={self.curve_relation!r} 不在 {_CURVE_RELATIONS}"
            )
        if self.family != "PSI":
            raise ValueError(
                f"协议 {self.name} 的 family={self.family!r} 不是 'PSI'；"
                "PSI 注册表只登记 PSI 族协议"
            )

    @property
    def result_semantics(self) -> str:
        if self.semantics is not None:
            return self.semantics
        return RESULT_SEMANTICS_EXACT if self.exact else RESULT_SEMANTICS_NOISY

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "world_size": self.world_size,
            "exact": self.exact,
            "result_semantics": self.result_semantics,
            "curve_relation": self.curve_relation,
            "params_schema": {key: dict(value) for key, value in self.params_schema.items()},
            "candidate_for": list(self.candidate_for),
            "family": self.family,
        }


#: 已注册协议（值来自 spu 0.9.5 实测 / 上游源码核对，说明见 capability 模块注释）
PROTOCOL_SPECS: Mapping[str, PsiProtocolSpec] = {
    "PROTOCOL_ECDH": PsiProtocolSpec(
        name="PROTOCOL_ECDH",
        world_size=2,
        exact=True,
        curve_relation="required",
        params_schema=_COMMON_PARAMS,
        candidate_for=PSI_CANDIDATE_OPS,
    ),
    "PROTOCOL_KKRT": PsiProtocolSpec(
        name="PROTOCOL_KKRT",
        world_size=2,
        exact=True,
        curve_relation="ignored",
        params_schema=_COMMON_PARAMS,
        candidate_for=PSI_CANDIDATE_OPS,
    ),
    "PROTOCOL_RR22": PsiProtocolSpec(
        name="PROTOCOL_RR22",
        world_size=2,
        exact=True,
        curve_relation="ignored",
        params_schema=_RR22_PARAMS,
        candidate_for=PSI_CANDIDATE_OPS,
    ),
    "PROTOCOL_ECDH_3PC": PsiProtocolSpec(
        name="PROTOCOL_ECDH_3PC",
        world_size=3,
        exact=True,
        curve_relation="required",
        params_schema=_COMMON_PARAMS,
        candidate_for=(),
    ),
    "PROTOCOL_ECDH_NPC": PsiProtocolSpec(
        name="PROTOCOL_ECDH_NPC",
        world_size=2,
        exact=True,
        curve_relation="required",
        params_schema=_COMMON_PARAMS,
        candidate_for=(),
    ),
    "PROTOCOL_KKRT_NPC": PsiProtocolSpec(
        name="PROTOCOL_KKRT_NPC",
        world_size=2,
        exact=True,
        curve_relation="ignored",
        params_schema=_COMMON_PARAMS,
        candidate_for=(),
    ),
    "PROTOCOL_DP": PsiProtocolSpec(
        name="PROTOCOL_DP",
        world_size=2,
        exact=False,
        curve_relation="implicit",
        params_schema=_COMMON_PARAMS,
        candidate_for=(),
    ),
}

#: 协议名清单（官方枚举成员，顺序与 libpsi.pyi 一致）
PSI_PROTOCOL_NAMES: tuple[str, ...] = tuple(PROTOCOL_SPECS)

#: 派生视图：参与方数量 / 曲线关系 / 带噪协议（capability 从这里读取）
PSI_PROTOCOL_WORLD_SIZE: Mapping[str, int] = {
    name: spec.world_size for name, spec in PROTOCOL_SPECS.items()
}
PSI_CURVE_RELATION: Mapping[str, str] = {
    name: spec.curve_relation for name, spec in PROTOCOL_SPECS.items()
}
PSI_PROTOCOLS_WITH_NOISE: tuple[str, ...] = tuple(
    name
    for name, spec in PROTOCOL_SPECS.items()
    if spec.result_semantics == RESULT_SEMANTICS_NOISY
)


def get_protocol_spec(name: str) -> PsiProtocolSpec:
    """按**官方枚举名**取协议元数据（严格匹配；归一化由 capability 负责）。"""

    spec = PROTOCOL_SPECS.get(name)
    if spec is None:
        raise ValueError(f"未注册的 PSI 协议 {name!r}；已注册：{PSI_PROTOCOL_NAMES}")
    return spec


def candidate_protocols_for(op: str) -> tuple[str, ...]:
    """登记为算子 op 候选的协议清单（DP 等带噪协议不入候选，只允许显式选择）。"""

    return tuple(name for name, spec in PROTOCOL_SPECS.items() if op in spec.candidate_for)
