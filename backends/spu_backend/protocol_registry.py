# -*- coding: utf-8 -*-
"""MPC（SPU）协议元数据（Phase 2）：MpcProtocolSpec。

与 PSI 注册表（`backends/psi_backend/protocol_registry.py`）的关系
----------------------------------------------------------------
两族**分开登记、互不合并**：协议名不重叠（PROTOCOL_* vs REF2K/...），执行
模型也不同（CSV 文件接口 vs jax.jit → SPU 虚拟机）。统一的是元数据形状与
校验入口（`backends/protocol_validation.py`），不是执行接口。

每条登记的出处（不许凭空填；升级 SPU 后先核对再改）
----------------------------------------------------
- world_size（参与方数量下限）按协议定义；项目既有口径见 docs/SPU_CAPABILITY.md。
- security_model：
    * ABY3 / SEMI2K / CHEETAH —— 上游 secretflow/spu `docs/reference/mpc_status.rst`
      （0.9.5rc2 与 main 同文，2026-10-08 核对）：
        ABY3：honest-majority 3PC，SPU 提供 semi-honest 实现（并提示合谋风险）；
        Semi2k-SPDZ：semi-honest NPC；默认用可信首方生成离线随机数，上游原文
        标注 "should be used for debugging purposes only"（建议仅用于调试）；
        Cheetah：fast 2pc semi-honest（HE 算术 + Ferret 布尔）。
    * REF2K —— "none"：本仓库实测三个 MPC 算子 send+recv 恒为 0 B
      （docs/mpc_comm_baseline.json，--comm --repeat 5）；只允许显式指定。
    * SECURENN —— "unverified"：上游当前文档未逐条声明其安全模型，
      本仓库不转述论文安全目标（与 docs/ADVERSARY_MODEL.md 同一纪律）。
- supported_fields：只登记**本仓库真机扫描过**的协议×环宽组合
  （tests/test_spu_backend.py::TestProtocolFieldSweep，2026-10-04 基线）：
    全协议 × FM64；ABY3 × FM32/FM128；SEMI2K / CHEETAH 的 FM32 由
    subset MPC 用例（tests/test_subset_mpc.py）覆盖。
  未实测组合在统一校验层被拒绝——放宽前先补真机扫描，再改这里。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

#: 协议族名（§四.1）；本注册表只登记 MPC 族。
MPC_FAMILY = "MPC"

#: 允许登记的环宽域（与 spu.libspu.FieldType 一致；
#: capability.SPU_FIELDS 是同一事实的实测出处，测试锁定两者相等）。
SUPPORTABLE_FIELDS: tuple[str, ...] = ("FM32", "FM64", "FM128")

#: 结果语义档（§11 三档词汇）。与 PSI 注册表同集，
#: 由 tests/test_protocol_validation.py 跨表锁定——新增档位时两边一起决定。
RESULT_SEMANTICS_EXACT = "exact"
RESULT_SEMANTICS_APPROXIMATE = "approximate"
RESULT_SEMANTICS_NOISY = "noisy"
RESULT_SEMANTICS: tuple[str, ...] = (
    RESULT_SEMANTICS_EXACT,
    RESULT_SEMANTICS_APPROXIMATE,
    RESULT_SEMANTICS_NOISY,
)

#: 安全模型档：semi-honest（上游明确声明）/ none（无密码学保护）/
#: unverified（上游未声明，本仓库不替它下结论）。
MPC_SECURITY_MODELS: tuple[str, ...] = ("semi-honest", "none", "unverified")

#: 本项目注册表里走 MPC 的算子（与 planner MPC 族一致；交叉断言见测试）。
MPC_CANDIDATE_OPS: tuple[str, ...] = ("DistanceLE", "WeightedSum", "TemporalOverlap")

_NOTE_ABY3 = (
    "上游 mpc_status.rst：honest-majority 3PC，SPU 提供 semi-honest 实现；"
    "并提示多于两个计算节点部署在一起时难以抵抗合谋"
)
_NOTE_SEMI2K = (
    "上游 mpc_status.rst：semi-honest NPC 协议；默认用可信首方生成离线随机数，"
    "上游原文标注 should be used for debugging purposes only（建议仅用于调试）"
)
_NOTE_CHEETAH = "上游 mpc_status.rst：2 方 semi-honest（HE 算术 + Ferret 布尔）"
_NOTE_REF2K = (
    "无密码学保护：本仓库实测三个 MPC 算子 send+recv 恒为 0 B"
    "（docs/mpc_comm_baseline.json，--comm --repeat 5）；只允许显式指定"
)
_NOTE_SECURENN = (
    "上游当前文档未逐条声明安全模型（mpc_status.rst 未含 SecureNN，2026-10-08 核对）；"
    "本仓库不转述论文安全目标，待上游确认后更新"
)


@dataclass(frozen=True)
class MpcProtocolSpec:
    """一种 MPC（SPU）协议的完整元数据（与 PsiProtocolSpec 平行、形状对齐）。"""

    name: str
    world_size: int
    security_model: str
    security_note: str
    supported_fields: tuple[str, ...]
    candidate_for: tuple[str, ...]
    params_schema: Mapping[str, Any]
    #: 结果是否精确（MPC 电路是确定性定点语义；将来出现带噪协议显式声明 semantics）
    exact: bool = True
    semantics: str | None = None
    family: str = MPC_FAMILY

    def __post_init__(self) -> None:
        if self.family != MPC_FAMILY:
            raise ValueError(
                f"协议 {self.name} 的 family={self.family!r} 不是 {MPC_FAMILY!r}；"
                "MPC 注册表只登记 MPC 族协议"
            )
        if self.security_model not in MPC_SECURITY_MODELS:
            raise ValueError(
                f"协议 {self.name} 的 security_model={self.security_model!r} "
                f"不在 {MPC_SECURITY_MODELS}"
            )
        if self.world_size < 2:
            raise ValueError(f"协议 {self.name} 的 world_size={self.world_size} < 2")
        if not self.supported_fields:
            raise ValueError(f"协议 {self.name} 未登记任何环宽（supported_fields 为空）")
        bad_fields = [f for f in self.supported_fields if f not in SUPPORTABLE_FIELDS]
        if bad_fields:
            raise ValueError(
                f"协议 {self.name} 的 supported_fields 含未登记环宽 {bad_fields}；"
                f"可选：{SUPPORTABLE_FIELDS}"
            )
        if self.semantics is not None and self.semantics not in RESULT_SEMANTICS:
            raise ValueError(
                f"协议 {self.name} 的 semantics={self.semantics!r} 不在 {RESULT_SEMANTICS}"
            )

    @property
    def result_semantics(self) -> str:
        if self.semantics is not None:
            return self.semantics
        return RESULT_SEMANTICS_EXACT if self.exact else RESULT_SEMANTICS_NOISY

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "world_size": self.world_size,
            "security_model": self.security_model,
            "security_note": self.security_note,
            "result_semantics": self.result_semantics,
            "exact": self.exact,
            "supported_fields": list(self.supported_fields),
            "candidate_for": list(self.candidate_for),
            "params_schema": {
                key: dict(value) for key, value in self.params_schema.items()
            },
        }


#: 已注册协议（world_size 与 docs/SPU_CAPABILITY.md 的「按协议定义」一致）
MPC_PROTOCOL_SPECS: Mapping[str, MpcProtocolSpec] = {
    "REF2K": MpcProtocolSpec(
        name="REF2K",
        world_size=2,
        security_model="none",
        security_note=_NOTE_REF2K,
        supported_fields=("FM64",),
        candidate_for=MPC_CANDIDATE_OPS,
        params_schema={},
    ),
    "SEMI2K": MpcProtocolSpec(
        name="SEMI2K",
        world_size=2,
        security_model="semi-honest",
        security_note=_NOTE_SEMI2K,
        supported_fields=("FM32", "FM64"),
        candidate_for=MPC_CANDIDATE_OPS,
        params_schema={},
    ),
    "ABY3": MpcProtocolSpec(
        name="ABY3",
        world_size=3,
        security_model="semi-honest",
        security_note=_NOTE_ABY3,
        supported_fields=("FM32", "FM64", "FM128"),
        candidate_for=MPC_CANDIDATE_OPS,
        params_schema={},
    ),
    "CHEETAH": MpcProtocolSpec(
        name="CHEETAH",
        world_size=2,
        security_model="semi-honest",
        security_note=_NOTE_CHEETAH,
        supported_fields=("FM32", "FM64"),
        candidate_for=MPC_CANDIDATE_OPS,
        params_schema={},
    ),
    "SECURENN": MpcProtocolSpec(
        name="SECURENN",
        world_size=3,
        security_model="unverified",
        security_note=_NOTE_SECURENN,
        supported_fields=("FM64",),
        candidate_for=MPC_CANDIDATE_OPS,
        params_schema={},
    ),
}

#: 协议名清单（顺序与 spu.libspu.ProtocolKind 枚举一致，capability 从这里派生）
MPC_PROTOCOL_NAMES: tuple[str, ...] = tuple(MPC_PROTOCOL_SPECS)

#: 派生视图：参与方数量下限 / 无密码学保护清单（capability 从这里读取）
MPC_PROTOCOL_WORLD_SIZE: Mapping[str, int] = {
    name: spec.world_size for name, spec in MPC_PROTOCOL_SPECS.items()
}
MPC_PROTOCOLS_WITHOUT_CRYPTO: tuple[str, ...] = tuple(
    name for name, spec in MPC_PROTOCOL_SPECS.items() if spec.security_model == "none"
)


def get_mpc_protocol_spec(name: str) -> MpcProtocolSpec:
    """按**官方枚举名**取协议元数据（严格匹配；归一化由 capability 负责）。"""

    spec = MPC_PROTOCOL_SPECS.get(name)
    if spec is None:
        raise ValueError(f"未注册的 MPC 协议 {name!r}；已注册：{MPC_PROTOCOL_NAMES}")
    return spec


def mpc_candidate_protocols_for(op: str) -> tuple[str, ...]:
    """登记为算子 op 候选的协议清单（与 planner 的 mpc_protocol_candidates 对齐）。"""

    return tuple(
        name for name, spec in MPC_PROTOCOL_SPECS.items() if op in spec.candidate_for
    )