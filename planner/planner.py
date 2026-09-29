"""planner：Geo-IR → 逐算子隐私计算方案。

输出五元组（课题硬性要求）：
    operation / representation / backend / estimated_cost / security_level
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from frontend.analyzer import (
    DIAG_BACKEND_MISSING,
    Diagnostic,
)
from ir import GeoOperation, GeoProgram, Sensitivity, requires_crypto

from .registry import OPERATOR_REGISTRY, OperatorRule, get_rule, resolve_cost


@dataclass
class PlannedStep:
    """一个算子对应的隐私计算落地步骤。"""

    operation: str
    representation: str
    backend: str
    estimated_cost: Mapping[str, Any]
    security_level: str
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

    def __init__(self, registry: Mapping[str, OperatorRule] | None = None) -> None:
        self.registry = dict(registry or OPERATOR_REGISTRY)

    def plan(self, program: GeoProgram) -> PrivacyPlan:
        plan = PrivacyPlan(program_name=program.name)

        for operation in program.operations:
            step, diagnostic = self._plan_operation(program, operation)
            if diagnostic is not None:
                plan.diagnostics.append(diagnostic)
            if step is not None:
                plan.steps.append(step)

        return plan

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
        cost = resolve_cost(rule, k=operation.params.get("K"))

        reasons = [
            f"算子 {operation.op} 的默认表征为 {rule.representation}",
            f"首选后端 {rule.primary_backend}，安全级别 {rule.security_level}",
        ]
        status = "planned"

        if not requires_crypto(sensitivity):
            status = "plaintext-ok"
            reasons.append("输入均为公开常量，可不进密态（计划保留密态路径备选）")

        if not rule.has_jax_impl and rule.primary_backend == "PSI":
            reasons.append("PSI 族算子无 JAX 逐元素原语，JAX 代码生成阶段将标注为不具备实现")

        step = PlannedStep(
            operation=operation.op,
            representation=rule.representation,
            backend=rule.backend,
            estimated_cost=cost,
            security_level=rule.security_level,
            inputs=operation.inputs,
            output_name=operation.output_name,
            sensitivity=sensitivity,
            status=status,
            notes=rule.notes,
            reasons=tuple(reasons),
            location=operation.location,
        )
        return step, None

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


def plan_program(program: GeoProgram, registry: Mapping[str, OperatorRule] | None = None) -> PrivacyPlan:
    """便捷入口。"""

    return Planner(registry).plan(program)


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