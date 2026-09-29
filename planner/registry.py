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
        bit_width_formula=weighted_sum_bit_width,
        notes="位宽随属性数 K 上升：b(K) = 8 + 8 + ceil(log2 K)",
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
