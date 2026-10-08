"""planner：Geo-IR → 逐算子隐私计算方案。

输出五元组（课题硬性要求）：
    operation / representation / backend / estimated_cost / security_level
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from frontend.analyzer import (
    DIAG_BACKEND_MISSING,
    DIAG_PROTOCOL_UNSUPPORTED,
    Diagnostic,
)
from ir import GeoOperation, GeoProgram, Sensitivity, requires_crypto

from .layout import LayoutShape, plan_layout
from .registry import (
    OPERATOR_REGISTRY,
    SELECTION_BASIS_EXPLICIT,
    OperatorRule,
    get_rule,
    mpc_protocol_ranking_hint,
    resolve_cost,
    select_mpc_protocol,
    validate_mpc_protocol_for_operation,
    validate_protocol_for_operation,
)


@dataclass
class PlannedStep:
    """一个算子对应的隐私计算落地步骤。"""

    operation: str
    representation: str
    backend: str
    estimated_cost: Mapping[str, Any]
    security_level: str
    #: PSI 族算子选用的协议（来自算子规则 default_protocol，或编译器显式指定）。
    #: None = 该算子不经过协议化的后端（数值算子 / 明文物化算子）。
    protocol: str | None = None
    #: 协议级参数（如 RR22 的 low_comm_mode）。协议属于规划/后端层，
    #: 不进入 Geo-IR 的任何数据模型。
    protocol_params: Mapping[str, Any] = field(default_factory=dict)
    #: MPC（SPU）族算子选用的协议。三种来源：编译器显式指定 > 按实测代价自动
    #: 选择（`select_mpc_protocol`）> 算子规则 default_mpc_protocol（无实测依据时）。
    #: 命名空间与上面的 PSI `protocol` 不同，故单列一个字段。
    #: None = 该算子不走 SPU 协议化后端。选择依据写在 `reasons` 里（可留痕可复核）。
    mpc_protocol: str | None = None
    #: 上面这个协议是怎么来的（SELECTION_BASIS_* 之一）：
    #: 按实测代价自动选 / 无依据时退回登记默认值 / 编译器显式指定。
    #: None = 该算子不走 SPU 协议化后端。"选了什么"与"凭什么"要一起留痕。
    mpc_protocol_basis: str | None = None
    inputs: tuple[str, ...] = ()
    output_name: str | None = None
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    status: str = "planned"
    notes: str = ""
    reasons: tuple[str, ...] = ()
    location: Mapping[str, Any] | None = None

    @property
    def needs_crypto(self) -> bool:
        return requires_crypto(self.sensitivity)

    @property
    def effective_backend(self) -> str:
        """不需要密态时的实际执行后端。"""

        return self.backend if self.needs_crypto else "Plaintext"

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "representation": self.representation,
            "backend": self.backend,
            "estimated_cost": dict(self.estimated_cost),
            "security_level": self.security_level,
            "protocol": self.protocol,
            "protocol_params": dict(self.protocol_params),
            "mpc_protocol": self.mpc_protocol,
            "mpc_protocol_basis": self.mpc_protocol_basis,
            "inputs": list(self.inputs),
            "output_name": self.output_name,
            "sensitivity": str(self.sensitivity),
            "status": self.status,
            "notes": self.notes,
        }


@dataclass
class PrivacyPlan:
    """整个程序的隐私计算方案。"""

    steps: list[PlannedStep] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    program_name: str = "<module>"

    def step_for(self, output_name: str) -> PlannedStep | None:
        for step in self.steps:
            if step.output_name == output_name:
                return step
        return None

    def steps_for_op(self, op: str) -> list[PlannedStep]:
        return [s for s in self.steps if s.operation == op]

    @property
    def crypto_steps(self) -> list[PlannedStep]:
        return [s for s in self.steps if s.needs_crypto]

    @property
    def has_errors(self) -> bool:
        return any(d.severity == "error" for d in self.diagnostics)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    def summary(self) -> dict[str, Any]:
        """汇总四量预算（可在报告里直接引用）。"""

        backends = sorted({s.effective_backend for s in self.steps})
        reps = sorted({s.representation for s in self.steps})
        return {
            "program": self.program_name,
            "step_count": len(self.steps),
            "crypto_step_count": len(self.crypto_steps),
            "backends": backends,
            "representations": reps,
            "security_levels": sorted({s.security_level for s in self.steps}),
        }


class Planner:
    """把 Geo-IR 程序规划为隐私计算方案。"""

    def __init__(
        self,
        registry: Mapping[str, OperatorRule] | None = None,
        *,
        psi_protocol: str | None = None,
        psi_protocol_params: Mapping[str, Any] | None = None,
        mpc_protocol: str | None = None,
        mpc_field: str | int | None = None,
        mpc_world_size: int | None = None,
        layout_shape: LayoutShape | None = None,
    ) -> None:
        self.registry = dict(registry or OPERATOR_REGISTRY)
        #: 编译器显式选择的 PSI 协议；None = 各算子用规则里的默认协议
        self.psi_protocol = psi_protocol
        #: 随协议变化的协议级参数（如 RR22 的 low_comm_mode）
        self.psi_protocol_params = dict(psi_protocol_params or {})
        #: 编译器显式选择的 MPC（SPU）协议；None = 该算子按实测代价自动选择
        #: （`select_mpc_protocol`），没有实测依据时退回登记默认值。
        self.mpc_protocol = mpc_protocol
        #: 统一能力校验（Phase 3）用：编译器给出的环宽 / 参与方数量。
        #: None = 未显式给出，该项不校验（world_size 缺省由协议下限决定）。
        self.mpc_field = mpc_field
        self.mpc_world_size = mpc_world_size
        #: 位平面布局的规模形状（D3）。None = 只做轴向决策、不预测条数：
        #: 没有形状就没有条数，硬凑一个"默认规模"等于编数字。
        self.layout_shape = layout_shape

    def plan(self, program: GeoProgram) -> PrivacyPlan:
        plan = PrivacyPlan(program_name=program.name)

        for operation in program.operations:
            step, diagnostic = self._plan_operation(program, operation)
            if diagnostic is not None:
                plan.diagnostics.append(diagnostic)
            if step is not None:
                plan.steps.append(step)

        return plan

    # ---------------- 规模口径 ----------------

    def _effective_k(self, operation) -> tuple[int | None, str | None]:
        """本方案的属性通道数 K 从哪来（P1：不再只有"显式传入"一条路）。

        优先级：

        1. `params.K`——调用方在算子参数里显式声明（最高优先，永远不被覆盖）；
        2. `layout_shape.attributes`——调用方给了 `--layout-shape` 时，
           **同一个 K 既进布局预测也进位宽**。此前两条路各算各的，
           报告里会同时出现 "attributes=4" 与 "K=3"；
        3. 都没有 → 返回 `None`，由 `resolve_cost` 用登记默认值兜底，
           并在 `K_basis` 里写明"未给规模"——**不替调用方猜**。
        """

        declared = operation.params.get("K")
        if declared is not None:
            return declared, "算子的 params.K（调用方显式声明）"
        if self.layout_shape is not None:
            return (
                int(self.layout_shape.attributes),
                "--layout-shape 的 attributes（与位平面布局同一口径）",
            )
        return None, None

    # ---------------- 单算子规划 ----------------

    def _plan_operation(
        self, program: GeoProgram, operation: GeoOperation
    ) -> tuple[PlannedStep | None, Diagnostic | None]:
        rule = self.registry.get(operation.op)

        if rule is None:
            return None, self._missing_backend_diagnostic(program, operation)

        sensitivity = operation.sensitivity or program.input_sensitivity(operation)
        # 代价按本方案实际的元素数实例化：位宽 b(K) 必须算出来，
        # 不能沿用档案里那个与自身公式矛盾的静态值。
        effective_k, k_basis = self._effective_k(operation)
        cost = resolve_cost(rule, k=effective_k, k_basis=k_basis)
        # 位平面布局（D3）：按归约轴选 L1 / L2，并把条数下降写回 estimated_cost。
        # 只对"经 MPC 值布局"的算子加键（PSI 集合运算 / 明文物化不加，避免噪声）。
        layout = plan_layout(operation.op, self.layout_shape)
        if layout.applies:
            cost = {**cost, **layout.to_cost_keys()}

        # 协议配置只对登记了默认协议的算子（PSI 族）生效。协议是规划层面的
        # 选择，不是地理语义——Geo-IR 的算子模型不因 RR22 增加任何字段。
        protocol: str | None = None
        protocol_params: dict[str, Any] = {}
        if rule.default_protocol is not None:
            protocol = self.psi_protocol or rule.default_protocol
            protocol_params = dict(rule.protocol_params)
            protocol_params.update(self.psi_protocol_params)

        # MPC（SPU）协议与 PSI 协议是两套命名空间，分开登记与校验。
        # 自动路径按**实测代价**挑（`select_mpc_protocol`），不再直接取登记默认值：
        # 登记默认值只是"没有实测依据"时的退化，不是"最省"的结论。
        mpc_protocol: str | None = None
        mpc_protocol_basis: str | None = None
        mpc_reason: str | None = None
        if rule.default_mpc_protocol is not None:
            if self.mpc_protocol:
                mpc_protocol = self.mpc_protocol
                mpc_protocol_basis = SELECTION_BASIS_EXPLICIT
                mpc_reason = (
                    f"MPC 协议 {mpc_protocol} 由编译器显式指定（覆盖实测代价排序）"
                )
            else:
                selection = select_mpc_protocol(operation.op)
                mpc_protocol = (
                    selection.protocol
                    if selection is not None
                    else rule.default_mpc_protocol
                )
                if selection is not None:
                    mpc_protocol_basis = selection.basis
                    mpc_reason = selection.reason

        reasons = [
            f"算子 {operation.op} 的默认表征为 {rule.representation}",
            f"首选后端 {rule.primary_backend}，安全级别 {rule.security_level}",
        ]
        if mpc_reason:
            reasons.append(mpc_reason)
        status = "planned"

        if not requires_crypto(sensitivity):
            status = "plaintext-ok"
            reasons.append("输入均为公开常量，可不进密态（计划保留密态路径备选）")

        if not rule.has_jax_impl and rule.primary_backend == "PSI":
            reasons.append("PSI 族算子无 JAX 逐元素原语，JAX 代码生成阶段将标注为不具备实现")

        # 算子 × 协议候选校验：不在候选清单的组合编译期拒绝，不等到 Runtime。
        protocol_check = validate_protocol_for_operation(operation.op, protocol)
        protocol_diagnostic: Diagnostic | None = None
        if protocol_check.ok:
            reasons.extend(protocol_check.notes)
        else:
            protocol_diagnostic = Diagnostic(
                code=DIAG_PROTOCOL_UNSUPPORTED,
                severity="error",
                message="；".join(protocol_check.problems),
                location=dict(operation.location) if operation.location else {},
                cause=(
                    f"所选协议 {protocol!r} 与算子 {operation.op} 的登记规则不匹配；"
                    "若放到运行时再拒绝，错误会晚到，且可能与执行默认值混淆"
                ),
                suggestion=(
                    "改用该算子的候选协议："
                    f"{list(rule.protocol_candidates) or '（无）'}"
                    "（DP 属带噪协议，显式选择可用但不能作为一致性验证依据）"
                ),
                estimated_cost=dict(cost),
            )

        # 算子 × MPC 协议候选校验：同一道编译期闸门，走 SPU 命名空间。
        mpc_check = validate_mpc_protocol_for_operation(
            operation.op,
            mpc_protocol,
            field=self.mpc_field,
            world_size=self.mpc_world_size,
        )
        mpc_diagnostic: Diagnostic | None = None
        if mpc_check.ok:
            reasons.extend(mpc_check.notes)
        else:
            mpc_diagnostic = Diagnostic(
                code=DIAG_PROTOCOL_UNSUPPORTED,
                severity="error",
                message="；".join(mpc_check.problems),
                location=dict(operation.location) if operation.location else {},
                cause=(
                    f"所选 MPC 协议 {mpc_protocol!r} 与算子 {operation.op} 的登记规则不匹配；"
                    "若放到运行时再拒绝，错误会晚到，且可能与执行默认值混淆"
                ),
                suggestion=(
                    "改用该算子的候选 MPC 协议："
                    f"{list(rule.mpc_protocol_candidates) or '（无）'}"
                    f"；{mpc_protocol_ranking_hint(operation.op)}"
                ),
                estimated_cost=dict(cost),
            )

        step = PlannedStep(
            operation=operation.op,
            representation=rule.representation,
            backend=rule.backend,
            estimated_cost=cost,
            security_level=rule.security_level,
            protocol=protocol,
            protocol_params=protocol_params,
            mpc_protocol=mpc_protocol,
            mpc_protocol_basis=mpc_protocol_basis,
            inputs=operation.inputs,
            output_name=operation.output_name,
            sensitivity=sensitivity,
            status=status,
            notes=rule.notes,
            reasons=tuple(reasons),
            location=operation.location,
        )
        return step, protocol_diagnostic or mpc_diagnostic

    def _missing_backend_diagnostic(
        self, program: GeoProgram, operation: GeoOperation
    ) -> Diagnostic:
        """第 5 类失败：后端没有对应隐私算子。"""

        from frontend.analyzer import _nearest_ops, _cost_hint_for

        alternatives = _nearest_ops(operation.op) or _nearest_ops(operation.op.lower())
        suggested = alternatives[0] if alternatives else None

        return Diagnostic(
            code=DIAG_BACKEND_MISSING,
            severity="error",
            message=f"算子 {operation.op} 在算子注册表中没有隐私计算落地规则",
            location=dict(operation.location) if operation.location else {},
            cause=(
                f"后端无法把 {operation.op} 映射到任何受支持的计算表征与协议；"
                f"当前注册表覆盖 {sorted(self.registry)}"
            ),
            suggestion=(
                "改用语义等价的已登记算子（见替代算子），"
                "或为该算子补登记规则并同时提供明文/JAX/SPU 三份实现与测试"
            ),
            suggested_op=suggested,
            estimated_cost=_cost_hint_for(suggested) if suggested else None,
        )


def plan_program(
    program: GeoProgram,
    registry: Mapping[str, OperatorRule] | None = None,
    *,
    psi_protocol: str | None = None,
    psi_protocol_params: Mapping[str, Any] | None = None,
    mpc_protocol: str | None = None,
    mpc_field: str | int | None = None,
    mpc_world_size: int | None = None,
    layout_shape: LayoutShape | None = None,
) -> PrivacyPlan:
    """便捷入口。

    `psi_protocol` / `psi_protocol_params` 是编译器对 PSI 族算子的协议选择；
    给了就覆盖算子规则的默认协议，并把参数并进每个 PSI 步骤的 protocol_params。
    `mpc_protocol` 是编译器对 MPC（SPU）族算子的协议选择；给了就覆盖这些算子
    的 default_mpc_protocol（命名空间是 REF2K/SEMI2K/...，与 PSI 不通用）。
    `mpc_field` / `mpc_world_size` 是编译器显式给出的执行配置；给了就参与
    算子×协议的统一能力校验（Phase 3：不匹配的组合在编译期拒绝）。
    `layout_shape` 是位平面布局（D3）的规模形状；不给就只做轴向决策、不预测条数。
    """

    return Planner(
        registry,
        psi_protocol=psi_protocol,
        psi_protocol_params=psi_protocol_params,
        mpc_protocol=mpc_protocol,
        mpc_field=mpc_field,
        mpc_world_size=mpc_world_size,
        layout_shape=layout_shape,
    ).plan(program)


def plan_table_rows(plan: PrivacyPlan) -> list[list[str]]:
    """给 CLI 输出用的表格行。"""

    rows = []
    for step in plan.steps:
        cost = step.estimated_cost
        rows.append(
            [
                step.operation,
                step.representation,
                step.backend,
                f"b={cost.get('b', '?')} d={cost.get('d', '?')} R={cost.get('R', '?')}",
                step.security_level,
                step.status,
            ]
        )
    return rows


PLAN_TABLE_HEADERS = ["Operation", "Representation", "Backend", "EstimatedCost", "Security", "Status"]
