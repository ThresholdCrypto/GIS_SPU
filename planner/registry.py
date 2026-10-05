"""planner 算子注册表：算子 → 计算表征 → 隐私后端。

默认规则（课题给定，勿随意改）：

| 算子              | 表征             | 后端      | 安全级别 |
|-------------------|------------------|-----------|----------|
| Intersects        | CompactCellSet   | PSI       | high     |
| Contains          | CompactCellSet   | PSI/MPC   | high     |
| DistanceLE        | QuantizedVector  | MPC/SPU   | high     |
| WeightedSum       | FixedPointVector | SPU/MPC   | medium   |
| TemporalOverlap   | TimeInterval     | MPC/SPU   | high     |
| CellSetIntersect  | CompactCellSet   | PSI       | high     |

高度维（Z）如何落地
------------------
Z 位域是 GB/T 40087-2021 附录 B 的**高度层号**，不是无关的整数分量。
集合族算子（Intersects / Contains / CellSetIntersect）因此天然带高度语义：

    同一 XY + 不同层号 = 不同的 64 位码 = 判为不相交；
    高度带重叠 = 两个层集合有交 = PSI 直接给出结论。

这一点值得写进规则表：**PSI 只做集合交，而高度带重叠恰好就是集合交**，
所以三维冲突判定不需要额外的 MPC 区间比较，也不增加泄漏面。
反过来说，若把高度带表示成"[z_lo, z_hi] 区间参数"，则必须引入密态区间
比较——那是把一个可被 PSI 承担的集合问题，换成一个更贵的算术问题。

容量约束（见 validator 的第 6 类失败）：层号上界随层级上升，Z 只有 7 位，
0-1000 m 低空带在 L>=23 上即装不下。层级选择必须与高度带一起校验。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class OperatorRule:
    """一个算子的隐私计算落地规则。"""

    op: str
    representation: str
    backend: str
    security_level: str
    cost_profile: Mapping[str, Any] = field(default_factory=dict)
    #: 是否具备可验证的后端实现（决定 planner 是否给出 proceed 结论）
    has_jax_impl: bool = False
    has_spu_protocol_support: bool = False
    #: PSI 族算子的默认协议；None = 该算子不经过协议化的 PSI（数值/物化算子）
    default_protocol: str | None = None
    #: 该算子预期可用的协议清单（不含带噪的 DP）；只作登记与提示，不做强制
    protocol_candidates: tuple[str, ...] = ()
    #: 协议级默认参数（receiver_rank / broadcast_result 等）；曲线不在此列
    protocol_params: Mapping[str, Any] = field(default_factory=dict)
    #: MPC（SPU）族算子的默认协议；None = 该算子不走 SPU 协议化后端。
    #: 与 PSI 的 default_protocol 分开登记：两者命名空间不同（PROTOCOL_* vs
    #: REF2K/SEMI2K/...），合用一个字段会让两种协议互相被当成"未知协议"。
    default_mpc_protocol: str | None = None
    #: 该算子被允许显式选择的 MPC 协议清单（编译期强制，见 validate_mpc_protocol_for_operation）
    mpc_protocol_candidates: tuple[str, ...] = ()
    notes: str = ""
    #: 位宽随元素数 K 变化的算子在此登记公式；None 表示位宽与 K 无关
    bit_width_formula: Callable[[int], int] | None = None

    @property
    def backends(self) -> tuple[str, ...]:
        """`PSI/MPC` → ('PSI', 'MPC')，含首选后端在前。"""

        return tuple(part.strip() for part in self.backend.split("/") if part.strip())

    @property
    def primary_backend(self) -> str:
        return self.backends[0]

    @property
    def fallback_backends(self) -> tuple[str, ...]:
        return self.backends[1:]

    @property
    def is_plaintext_local(self) -> bool:
        """该规则是否只在本方明文执行（不经过任何密态后端）。

        HeightBand 这类**物化**算子属于此列：格网码的编码在本方完成，
        密态边界落在消费它的集合族算子上。能力核查/模拟阶段必须据此跳过，
        否则会给出"该算子可被 SPU 执行"这种不成立的结论。
        """

        return self.primary_backend == "Plaintext"

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "representation": self.representation,
            "backend": self.backend,
            "security_level": self.security_level,
            "estimated_cost": resolve_cost(self),
            "has_jax_impl": self.has_jax_impl,
            "has_spu_protocol_support": self.has_spu_protocol_support,
            "default_protocol": self.default_protocol,
            "protocol_candidates": list(self.protocol_candidates),
            "protocol_params": dict(self.protocol_params),
            "default_mpc_protocol": self.default_mpc_protocol,
            "mpc_protocol_candidates": list(self.mpc_protocol_candidates),
            "notes": self.notes,
        }


#: 四量模型：N_ct 密文条数 / b 位宽 / d 乘法深度 / R 通信轮次
def _cost(n_ct: str, bits: str, depth: str, rounds: str, basis: str) -> dict[str, Any]:
    return {"N_ct": n_ct, "b": bits, "d": depth, "R": rounds, "basis": basis}


#: 规范化使用的元素数上界（服务端只做规范化，实务上按授权单元数截断）
_DEFAULT_K = 3


def weighted_sum_bit_width(k: int) -> int:
    """定点加权和所需位宽：b = 8（量化）+ 8（权重）+ ceil(log2 K)（累加余量）。

    这是 sqrt-free、无除法电路下的上界；K 为参与的属性通道数。
    """

    if k < 1:
        raise ValueError("K 必须为正整数")
    return 16 + (k - 1).bit_length()


def _k_from_nct(n_ct: str) -> int | None:
    """从 N_ct 描述里解析元素数（"1 条/属性通道（K=24）" → 24）。"""

    match = re.search(r"[（(]\s*K\s*=\s*(\d+)", n_ct)
    if match:
        return int(match.group(1))
    return None


def resolve_cost(rule: "OperatorRule", k: int | None = None) -> dict[str, Any]:
    """把静态代价档案实例化为该算子在本方案里的实际代价。

    位宽按 b(K) 实算而不是写死——写死的 "16" 与它自己的公式
    （16 + ceil(log2 K)）在 K>1 时矛盾，报价会低估累加余量。
    """

    profile = dict(rule.cost_profile)
    n_ct = str(profile.get("N_ct", "-"))
    profile["N_ct"] = n_ct

    if "{K}" in n_ct:
        # 描述里带占位符的，按 K 填充
        effective_k = k if k is not None else _DEFAULT_K
        profile["N_ct"] = n_ct.replace("{K}", str(effective_k))
    elif k is not None:
        # 描述里没有元素数信息，但调用方知道 → 在 N_ct 上补齐
        profile["N_ct"] = f"{n_ct}（K={k}）"

    formula = rule.bit_width_formula
    if formula is not None:
        effective_k = k if k is not None else _k_from_nct(n_ct) or _DEFAULT_K
        profile["b"] = f"{formula(effective_k)}（8+8+ceil(log2 K)，K={effective_k}）"
        profile["bit_width_rule"] = "b(K) = 8 + 8 + ceil(log2 K)"
        profile["K"] = effective_k

    return profile


#: PSI 族算子共享的协议候选。`PROTOCOL_DP` **不在**候选内：它是差分隐私
#: 协议，结果带噪（见 backends.psi_backend.capability.PSI_PROTOCOLS_WITH_NOISE），
#: 用于冲突判定会漏报/误报；CLI 显式选择仍可用，但候选清单不给它位置。
PSI_PROTOCOL_CANDIDATES: tuple[str, ...] = (
    "PROTOCOL_ECDH",
    "PROTOCOL_KKRT",
    "PROTOCOL_RR22",
)

#: PSI 族算子的默认协议。与 backends.psi_backend.capability.PSI_DEFAULT_PROTOCOL
#: 是同一个值的两处独立成文（tests/test_planner.py 有交叉断言防漂移）。
#: 真实 RR22 跑通并完成性能/正确性验证前**不把默认改成 RR22**（课题要求）。
PSI_RULE_DEFAULT_PROTOCOL = "PROTOCOL_ECDH"

#: MPC（SPU）族算子共享的协议候选。理由与 PSI 侧相同：planner 不能在导入期
#: 依赖 backends（模块级互导成环），故此处仍是字面量登记；与
#: backends.spu_backend.capability.SPU_PROTOCOLS 的一致性由交叉断言锁定
#: （tests/test_planner.py 与 tests/test_protocol_registry.py）。
MPC_PROTOCOL_CANDIDATES: tuple[str, ...] = (
    "REF2K",
    "SEMI2K",
    "ABY3",
    "CHEETAH",
    "SECURENN",
)

#: MPC 族算子的默认协议。与 backends.spu_backend.run_spu_simulation 的缺省值
#: （ABY3）是同一个值的两处独立成文（tests 有交叉断言防漂移）。
#: SECURENN 需要 3 方、CHEETAH 属半诚实 2PC，均不作默认。
MPC_RULE_DEFAULT_PROTOCOL = "ABY3"

OPERATOR_REGISTRY: dict[str, OperatorRule] = {
    "Intersects": OperatorRule(
        op="Intersects",
        representation="CompactCellSet",
        backend="PSI",
        security_level="high",
        cost_profile=_cost(
            n_ct="1 条/格网（定长 8 B 键）",
            bits="64",
            depth="0",
            rounds="1（PSI 求交轮次）",
            basis="集合交不引入乘法；代价由 N_ct 与 R 主导，b 固定 64 位",
        ),
        has_jax_impl=False,
        has_spu_protocol_support=True,
        default_protocol=PSI_RULE_DEFAULT_PROTOCOL,
        protocol_candidates=PSI_PROTOCOL_CANDIDATES,
        protocol_params={"receiver_rank": 0, "broadcast_result": False},
        notes=(
            "集合交语义，走 PSI；JAX 侧无对应原语（交集性是组合问题，非逐元素算子）",
            "Z 位域是 GB/T 40087 附录 B 的高度层号，故该算子天然是三维判定："
            "同一 XY 上高度带重叠即两集合有交，无需额外 MPC 区间比较",
        ),
    ),
    "Contains": OperatorRule(
        op="Contains",
        representation="CompactCellSet",
        backend="PSI/MPC",
        security_level="high",
        cost_profile=_cost(
            n_ct="1 条/格网",
            bits="64",
            depth="1（包含性判定）",
            rounds="1（PSI）+ 1（包含性确认）",
            basis="包含是交集的子集关系，PSI 出交集后需一次 MPC 比较基数",
        ),
        has_jax_impl=False,
        has_spu_protocol_support=True,
        default_protocol=PSI_RULE_DEFAULT_PROTOCOL,
        protocol_candidates=PSI_PROTOCOL_CANDIDATES,
        protocol_params={"receiver_rank": 0, "broadcast_result": False},
        notes=(
            "交集本体仍由 PSI 求出（CompactCellSet 求交），子集判定**默认**走 MPC 基数等值"
            "（k=|outer∩inner| == n=|inner|），不再做明文比较；MPC 不可用时按 plaintext-fallback 退回明文并披露"
            "（见 backends.psi_backend.subset_mpc 与 README 7.6）",
            "Z 位域是 GB/T 40087 附录 B 的高度层号，高度带包含同样按层集合处理："
            "被包含方的每一层都须落在包含方的层集合内",
        ),
    ),
    "DistanceLE": OperatorRule(
        op="DistanceLE",
        representation="QuantizedVector",
        backend="MPC/SPU",
        security_level="high",
        cost_profile=_cost(
            n_ct="1 条/候选点",
            bits="8（量化箱号）",
            depth="1（平方后求和，无开方）",
            rounds="1（归约）",
            basis="避 sqrt：比较距离平方与阈值平方，d=1 即可",
        ),
        has_jax_impl=True,
        has_spu_protocol_support=True,
        default_mpc_protocol=MPC_RULE_DEFAULT_PROTOCOL,
        mpc_protocol_candidates=MPC_PROTOCOL_CANDIDATES,
        notes="量化整数向量；d 由元素乘法 + 求和决定",
    ),
    "WeightedSum": OperatorRule(
        op="WeightedSum",
        representation="FixedPointVector",
        backend="SPU/MPC",
        security_level="medium",
        cost_profile=_cost(
            n_ct="1 条/属性通道",
            bits="按 b(K) 实算（见 bit_width_rule）",
            depth="1（乘加）",
            rounds="1（归约）",
            basis="定点乘加；位宽按 log2(K) 预留累加余量",
        ),
        has_jax_impl=True,
        has_spu_protocol_support=True,
        default_mpc_protocol=MPC_RULE_DEFAULT_PROTOCOL,
        mpc_protocol_candidates=MPC_PROTOCOL_CANDIDATES,
        bit_width_formula=weighted_sum_bit_width,
        notes=(
            "位宽随属性数 K 上升：b(K) = 8 + 8 + ceil(log2 K)；"
            "另需环宽 ≥ FM64——定点除法路径内部要求 64 位环，"
            "FM32 下该算子起不来（预测位宽 b(K) 只算数据位宽，覆盖不了这个下限，"
            "见 docs/MPC_BENCHMARK_PROTOCOL.md §4.3）"
        ),
    ),
    "TemporalOverlap": OperatorRule(
        op="TemporalOverlap",
        representation="TimeInterval",
        backend="MPC/SPU",
        security_level="high",
        cost_profile=_cost(
            n_ct="≤ 2·Lt_max = 28 条/格网",
            bits="18（Toff 14 + Lt 4）",
            depth="1（区间比较）",
            rounds="1（归约）",
            basis="段式编码使节点数上界为 2·Lt_max，避免 720× 逐分钟物化",
        ),
        has_jax_impl=True,
        has_spu_protocol_support=True,
        default_mpc_protocol=MPC_RULE_DEFAULT_PROTOCOL,
        mpc_protocol_candidates=MPC_PROTOCOL_CANDIDATES,
        notes="区间重叠判定；节点必须按 2^Lt 对齐（见课题审计发现 F3）",
    ),
    "CellSetIntersect": OperatorRule(
        op="CellSetIntersect",
        representation="CompactCellSet",
        backend="PSI",
        security_level="high",
        cost_profile=_cost(
            n_ct="1 条/格网",
            bits="64",
            depth="0",
            rounds="1（PSI）",
            basis="与 Intersects 同族，但输出交集本身而非布尔",
        ),
        has_jax_impl=False,
        has_spu_protocol_support=True,
        default_protocol=PSI_RULE_DEFAULT_PROTOCOL,
        protocol_candidates=PSI_PROTOCOL_CANDIDATES,
        protocol_params={"receiver_rank": 0, "broadcast_result": False},
        notes=(
            "输出为格网集合本体，用于后续链式计算",
            "Z 位域是 GB/T 40087 附录 B 的高度层号，故交集本体会同时带出高度层"
            "信息（码里含 Z），可直接喂给三维下游算子",
            "高度带重叠即层集合有交，由 PSI 承担，不需要额外的密态区间比较",
        ),
    ),
    # ----------------------------------------------------------------------
    # 物化算子：不进密态，但在 IR 与代价账上必须可见
    # ----------------------------------------------------------------------
    "HeightBand": OperatorRule(
        op="HeightBand",
        representation="CompactCellSet",
        backend="Plaintext",
        security_level="low",
        cost_profile=_cost(
            n_ct="0（不进密态）",
            bits="64",
            depth="0",
            rounds="0",
            basis="层集合的编码在本方明文完成；密态代价落在消费它的 PSI 交集上",
        ),
        has_jax_impl=False,
        has_spu_protocol_support=False,
        notes=(
            "把 (XY 单元, 高度带) 展开成 3D 层集合（每层一个 64 位码）",
            "登记它的目的不是隐私计算，而是让“高度带参与判定”在 Geo-IR 里可见："
            "未登记时前端会静默丢弃整条调用，编译报成功而高度带根本没进 IR",
            "层号容量约束（Z7 位域）由第 6 类失败检查从 params 里读取 level/height_max 判定",
        ),
    ),
}


def get_rule(op: str) -> OperatorRule | None:
    return OPERATOR_REGISTRY.get(op)


def registered_ops() -> tuple[str, ...]:
    return tuple(OPERATOR_REGISTRY)


def jax_capable_ops() -> tuple[str, ...]:
    return tuple(op for op, rule in OPERATOR_REGISTRY.items() if rule.has_jax_impl)


def backend_capable_ops(backend: str) -> tuple[str, ...]:
    """在某后端上具备规则的算子。"""

    out = []
    for op, rule in OPERATOR_REGISTRY.items():
        if backend.upper() in [b.upper() for b in rule.backends]:
            out.append(op)
    return tuple(out)


# --------------------------------------------------------------------------
# 算子 × 协议候选校验（编译期）
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtocolCheck:
    """`validate_protocol_for_operation` 的结论。

    `ok=False` 表示**编译期拒绝**：该组合不该等到 Runtime 才报错。
    `notes` 是放行但必须披露的事实（如 DP 带噪）。
    """

    ok: bool
    op: str
    protocol: str | None
    problems: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "op": self.op,
            "protocol": self.protocol,
            "problems": list(self.problems),
            "notes": list(self.notes),
        }


def validate_protocol_for_operation(op: str, protocol: str | None) -> ProtocolCheck:
    """校验"算子在所选协议下是否可以规划/执行"。

    规则（与注册表同源，不另抄名单；协议清单/参与方数量取自 PSI 能力模块）：
    - 未显式选协议（None）→ 放行（各算子用登记默认值）；
    - 非 PSI 算子 + 协议 → 拒绝：该协议不会被这个算子使用，静默接受
      等于容忍"设了但没生效"；
    - PSI 族 + 候选清单内协议 → 放行；
    - PSI 族 + DP → 放行但标记带噪；
    - PSI 族 + 候选清单外协议 → 拒绝，并区分两种原因：
      本链路参与方数量不满足（如 ECDH_3PC），或未登记为候选。

    后端依赖用函数内导入：planner 不在导入期依赖 backends，保持分层无环。
    """

    from backends.psi_backend.capability import (
        PSI_RUNTIME_WORLD_SIZE,
        normalize_psi_protocol,
        protocol_world_size,
        runnable_protocols_hint,
    )

    if protocol is None:
        return ProtocolCheck(ok=True, op=op, protocol=None)

    rule = OPERATOR_REGISTRY.get(op)
    if rule is None:
        return ProtocolCheck(
            ok=False,
            op=op,
            protocol=str(protocol),
            problems=(f"算子 {op} 不在算子注册表中，无法校验协议 {protocol}",),
        )

    try:
        name = normalize_psi_protocol(protocol)
    except ValueError as exc:
        return ProtocolCheck(
            ok=False, op=op, protocol=str(protocol), problems=(str(exc),)
        )

    if rule.default_protocol is None:
        return ProtocolCheck(
            ok=False,
            op=op,
            protocol=name,
            problems=(
                f"算子 {op} 不经过 PSI 协议后端（表征 {rule.representation}，"
                f"后端 {rule.backend}）；协议 {name} 不会被该算子使用",
            ),
        )

    if name == "PROTOCOL_DP":
        return ProtocolCheck(
            ok=True,
            op=op,
            protocol=name,
            notes=(
                "PROTOCOL_DP 为差分隐私协议：允许显式选择，但结果带噪，"
                "不能作为与明文一致的一致性验证依据",
            ),
        )

    if name in rule.protocol_candidates:
        return ProtocolCheck(ok=True, op=op, protocol=name)

    required = protocol_world_size(name)
    if required > PSI_RUNTIME_WORLD_SIZE:
        return ProtocolCheck(
            ok=False,
            op=op,
            protocol=name,
            problems=(
                f"协议 {name} 需要 {required} 个参与方，"
                f"本链路固定 {PSI_RUNTIME_WORLD_SIZE} 方，无法执行该协议；"
                f"请改用：{runnable_protocols_hint()}",
            ),
        )
    return ProtocolCheck(
        ok=False,
        op=op,
        protocol=name,
        problems=(
            f"协议 {name} 不在算子 {op} 的候选协议清单 "
            f"{list(rule.protocol_candidates)}；未登记的协议组合不予放行"
            "（若确需支持，请先登记候选并补测试）",
        ),
    )


def validate_mpc_protocol_for_operation(op: str, protocol: str | None) -> ProtocolCheck:
    """校验"算子在所选 **SPU/MPC** 协议下是否可以规划/执行"。

    与 `validate_protocol_for_operation` 的分工：
    - 那个走 **PSI 命名空间**（`PROTOCOL_ECDH` / `PROTOCOL_RR22` / ...）；
    - 这个走 **SPU 命名空间**（`REF2K` / `SEMI2K` / `ABY3` / `CHEETAH` / `SECURENN`）。

    两者协议名不重叠，因此**不能合并成一个入口**：此前 MPC 协议只能走 PSI
    入口，结果 `CHEETAH` / `SEMI2K` 会被报成"未知 PSI 协议"——即"MPC 协议根本
    进不了规划层"。本函数就是补上这条入口。

    规则：
    - 未显式选协议（None）→ 放行（各算子用登记默认值）；
    - 非 MPC 算子 + 协议 → 拒绝（设了却不生效，等于容忍静默失效）；
    - MPC 族 + 候选清单内协议 → 放行，并披露该协议的最少参与方数量；
    - MPC 族 + 候选清单外协议 → 拒绝。

    后端依赖用函数内导入：planner 不在导入期依赖 backends，保持分层无环。
    """

    from backends.spu_backend.capability import (
        normalize_protocol,
        protocol_min_world_size,
    )

    if protocol is None:
        return ProtocolCheck(ok=True, op=op, protocol=None)

    rule = OPERATOR_REGISTRY.get(op)
    if rule is None:
        return ProtocolCheck(
            ok=False,
            op=op,
            protocol=str(protocol),
            problems=(f"算子 {op} 不在算子注册表中，无法校验协议 {protocol}",),
        )

    try:
        name = normalize_protocol(protocol)
    except ValueError as exc:
        return ProtocolCheck(
            ok=False, op=op, protocol=str(protocol), problems=(str(exc),)
        )

    if rule.default_mpc_protocol is None:
        return ProtocolCheck(
            ok=False,
            op=op,
            protocol=name,
            problems=(
                f"算子 {op} 不经由 SPU/MPC 协议后端（表征 {rule.representation}，"
                f"后端 {rule.backend}）；协议 {name} 不会被该算子使用"
                "（PSI 族协议请走 validate_protocol_for_operation）",
            ),
        )

    if name in rule.mpc_protocol_candidates:
        return ProtocolCheck(
            ok=True,
            op=op,
            protocol=name,
            notes=(
                f"MPC 协议 {name}（最少 {protocol_min_world_size(name)} 方）",
            ),
        )

    return ProtocolCheck(
        ok=False,
        op=op,
        protocol=name,
        problems=(
            f"协议 {name} 不在算子 {op} 的 MPC 候选协议清单 "
            f"{list(rule.mpc_protocol_candidates)}；未登记的协议组合不予放行"
            "（若确需支持，请先登记候选并补测试）",
        ),
    )
