"""validator：编译前验证与错误报告。

六类失败模式（课题指定前五类 + 本项目补充的高度层容量类），每一类都要给出：
    错误位置 / 问题原因 / 建议替代算子 / 预计隐私计算代价

**绝不自动修改用户代码**——只产生诊断与建议。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from frontend.analyzer import (
    DIAG_ARITY,
    DIAG_BACKEND_MISSING,
    DIAG_DYNAMIC_CONTROL_FLOW,
    DIAG_HEIGHT_LAYER,
    DIAG_NOT_TRACEABLE,
    DIAG_SPU_UNSUPPORTED,
    DIAG_STATIC_ASSERT,
    DIAG_UNKNOWN_CALL,
    DIAG_UNSUPPORTED_GEO_OP,
    Diagnostic,
)
from ir import GeoOperation, GeoProgram
from ir.geosot import (
    HEIGHT_LAYER_BITS,
    HEIGHT_LAYER_MAX,
    height_index,
    height_layer_bits,
    height_layer_fits,
)
from planner.planner import PrivacyPlan
from planner.registry import OPERATOR_REGISTRY, get_rule

# --------------------------------------------------------------------------
# 失败类型定义
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureClass:
    """一类失败模式的定义。"""

    code: str
    name: str
    description: str
    remedy: str


FAILURE_CLASSES: tuple[FailureClass, ...] = (
    FailureClass(
        code=DIAG_UNSUPPORTED_GEO_OP,
        name="地理算子不支持",
        description="用户调用了一个未登记的地理算子（含拼写变体）。",
        remedy="改用下方建议的等价算子；或登记新算子并补齐三份实现与测试。",
    ),
    FailureClass(
        code=DIAG_NOT_TRACEABLE,
        name="JAX 算子无法追踪",
        description="生成的或用户提供的 JAX 代码无法被 jax.jit 追踪（含张量形状依赖、"
        "Python 标量运算、numpy 函数混用等）。",
        remedy="改为纯 jnp 表达式；把形状相关的量改成输入张量；避免 numpy 与 python 分支。",
    ),
    FailureClass(
        code=DIAG_SPU_UNSUPPORTED,
        name="SPU 当前版本不支持",
        description="所需原语、协议或环宽不在当前安装的 SPU 版本支持范围内。",
        remedy="核对 docs/SPU_CAPABILITY.md；更换协议/环宽，或改用等价的可支持算子。",
    ),
    FailureClass(
        code=DIAG_DYNAMIC_CONTROL_FLOW,
        name="动态 Python 控制流无法编译",
        description="密态分支/循环在编译期无法展开为静态计算图。",
        remedy="用 jnp.where / jnp.maximum 等无分支原语表达；或把分支拆成独立任务。",
    ),
    FailureClass(
        code=DIAG_BACKEND_MISSING,
        name="后端没有对应隐私算子",
        description="算子注册表中没有该算子的隐私计算落地规则。",
        remedy="改用语义等价的已登记算子，或补登记算子规则。",
    ),
    FailureClass(
        code=DIAG_HEIGHT_LAYER,
        name="高度层号超出 Z 位域",
        description="GB/T 40087-2021 附录 B 的高度层号随层级上升而变多；"
        "64 位定长键里 Z 只有 7 位，在 0-1000 m 低空带上 L>=23 即装不下。",
        remedy="把层级降到 Z 位域容量以内（0-1000 m 带建议 L<=22），"
        "或把高度带收窄到管理方的实际管辖区间；不要放宽 Z 位宽——"
        "那会挤占 X'/Y' 位段并破坏跨方定长键口径。",
    ),
)

FAILURE_BY_CODE = {fc.code: fc for fc in FAILURE_CLASSES}


# --------------------------------------------------------------------------
# 代价估算（用于错误报告中的"预计隐私计算代价"）
# --------------------------------------------------------------------------


@dataclass
class CostEstimate:
    """隐私计算代价估算（四量模型）。"""

    representation: str = "unknown"
    backend: str = "unknown"
    security_level: str = "unknown"
    n_ct: str = "-"
    bits: str = "-"
    depth: str = "-"
    rounds: str = "-"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "backend": self.backend,
            "security_level": self.security_level,
            "N_ct": self.n_ct,
            "b": self.bits,
            "d": self.depth,
            "R": self.rounds,
            "note": self.note,
        }

    def describe(self) -> str:
        return (
            f"{self.backend} / {self.representation}"
            f"（b={self.bits}, d={self.depth}, R={self.rounds}）"
        )


def estimate_cost(op: str) -> CostEstimate:
    """给算子估算隐私计算代价；未登记算子返回带说明的保守估值。"""

    rule = get_rule(op)
    if rule is None:
        return CostEstimate(
            note=f"算子 {op} 未登记，无法给出可信代价；需先补登记规则",
        )
    from planner.registry import resolve_cost

    profile = resolve_cost(rule)
    return CostEstimate(
        representation=rule.representation,
        backend=rule.backend,
        security_level=rule.security_level,
        n_ct=str(profile.get("N_ct", "-")),
        bits=str(profile.get("b", "-")),
        depth=str(profile.get("d", "-")),
        rounds=str(profile.get("R", "-")),
        note=str(profile.get("basis", "")),
    )


def estimate_cost_for_suggestion(op: str | None) -> CostEstimate | None:
    if not op:
        return None
    return estimate_cost(op)


# --------------------------------------------------------------------------
# 验证结果
# --------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """一次编译前验证的总报告。"""

    diagnostics: list[Diagnostic] = field(default_factory=list)
    checked_ops: tuple[str, ...] = ()
    trace_checks: Mapping[str, Any] = field(default_factory=dict)
    capability: Mapping[str, Any] = field(default_factory=dict)
    stage_results: Mapping[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity != "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def by_class(self) -> dict[str, list[Diagnostic]]:
        """按失败类型归组，便于报告分节。"""

        grouped: dict[str, list[Diagnostic]] = {fc.code: [] for fc in FAILURE_CLASSES}
        for diagnostic in self.diagnostics:
            grouped.setdefault(diagnostic.code, []).append(diagnostic)
        return {code: items for code, items in grouped.items() if items}

    def render(self) -> str:
        """渲染为可读报告（CLI 直接用）。"""

        lines: list[str] = []
        if not self.diagnostics:
            lines.append("  未发现编译前问题。")
            return "\n".join(lines)

        for code, items in self.by_class().items():
            failure = FAILURE_BY_CODE.get(code)
            title = f"{failure.name}（{code}）" if failure else code
            lines.append(f"  ▸ {title}")
            for diagnostic in items:
                lines.append(f"      {diagnostic}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "by_class": {code: [d.to_dict() for d in items] for code, items in self.by_class().items()},
            "checked_ops": list(self.checked_ops),
            "trace_checks": {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in self.trace_checks.items()},
            "capability": dict(self.capability),
            "stage_results": dict(self.stage_results),
        }


# --------------------------------------------------------------------------
# 各类检查
# --------------------------------------------------------------------------


def validate_program(program: GeoProgram, plan: PrivacyPlan | None = None) -> list[Diagnostic]:
    """第 1、5 类：算子是否受支持、后端是否有对应隐私算子。"""

    diagnostics: list[Diagnostic] = []

    for operation in program.operations:
        rule = OPERATOR_REGISTRY.get(operation.op)
        if rule is None:
            from semantic import suggest_ops

            suggestions = suggest_ops(operation.op)
            cost = estimate_cost_for_suggestion(suggestions[0] if suggestions else None)
            diagnostics.append(
                Diagnostic(
                    code=DIAG_BACKEND_MISSING,
                    severity="error",
                    message=f"算子 {operation.op} 没有登记隐私计算规则",
                    location=dict(operation.location) if operation.location else {},
                    cause="算子注册表中找不到该算子的 表征/后端/安全级别 三元组。",
                    suggestion=(
                        "改用语义等价的已登记算子；"
                        f"当前已登记：{sorted(OPERATOR_REGISTRY)}"
                    ),
                    suggested_op=suggestions[0] if suggestions else None,
                    estimated_cost=cost.to_dict() if cost else None,
                )
            )

    if plan is not None:
        diagnostics.extend(plan.diagnostics)

    return diagnostics


def validate_height_layer_capacity(program: GeoProgram) -> list[Diagnostic]:
    """第 6 类：高度带 / 层级组合是否装得进 7 位 Z 位域。

    Z 位域是**层号**位域，不是米位域。层号上界 = height_index(hmax, L)，
    随层级上升而变大，因此同一个高度带在高层级上可能溢出。这是三维格网
    接入层独有的容量约束，必须在编译前拦下——不是运行期错误。
    """

    diagnostics: list[Diagnostic] = []
    for operation in program.operations:
        params = operation.params or {}
        level = params.get("height_level", params.get("level"))
        hmax = params.get("height_max")
        if level is None or hmax is None:
            continue
        try:
            fits = height_layer_fits(int(level), float(hmax))
        except ValueError as exc:
            diagnostics.append(
                Diagnostic(
                    code=DIAG_HEIGHT_LAYER,
                    severity="error",
                    message=f"算子 {operation.op} 的高度/层级参数非法：{exc}",
                    location=dict(operation.location) if operation.location else {},
                    cause=str(exc),
                    suggestion="请核对层级落在 [0, 32] 且高度带上界为正数。",
                )
            )
            continue
        if fits:
            continue

        bits = height_layer_bits(int(level), float(hmax))
        limit_level = max(
            L for L in range(0, 33) if height_layer_fits(L, float(hmax))
        )
        layer = height_index(float(hmax), int(level))
        diagnostics.append(
            Diagnostic(
                code=DIAG_HEIGHT_LAYER,
                severity="error",
                message=(
                    f"算子 {operation.op} 的高度带 [0, {hmax:g}] m 在层级 L={level} 上"
                    f"需要 {bits} 位层号，超出 Z{HEIGHT_LAYER_BITS} 位域"
                ),
                location=dict(operation.location) if operation.location else {},
                cause=(
                    f"GB/T 40087-2021 附录 B 的层号上界随层级上升：L={level} 时"
                    f" hmax={hmax:g} m 落在第 {layer} 层，需要 {bits} 位；"
                    f"64 位定长键的 Z 位段只有 {HEIGHT_LAYER_BITS} 位（上限 {HEIGHT_LAYER_MAX}）。"
                ),
                suggestion=(
                    f"把层级降到 L<={limit_level}（该高度带在 L={limit_level} 上只需"
                    f" {height_layer_bits(limit_level, float(hmax))} 位），"
                    "或把高度带收窄到实际管辖区间。"
                ),
                suggested_op=operation.op,
                estimated_cost=estimate_cost(operation.op).to_dict(),
            )
        )
    return diagnostics


def validate_control_flow(program: GeoProgram) -> list[Diagnostic]:
    """第 4 类：算子是否被不可编译的控制流包裹。"""

    diagnostics: list[Diagnostic] = []
    for operation in program.operations:
        if operation.params.get("_inside_control_flow"):
            diagnostics.append(
                Diagnostic(
                    code=DIAG_DYNAMIC_CONTROL_FLOW,
                    severity="error",
                    message=(
                        f"算子 {operation.op}（{operation.output_name}）位于动态控制流内部"
                    ),
                    location=dict(operation.location) if operation.location else {},
                    cause=(
                        "该算子位于 if/for/while/try 体内，且条件依赖运行期数据；"
                        "编译器只能把静态的算子序列降级为密态电路。"
                    ),
                    suggestion=(
                        "把该算子提到分支外，改为数据并行写法（jnp.where / jnp.maximum）；"
                        "或把它拆成独立的 geo 任务由业务侧调度。"
                    ),
                    suggested_op="WeightedSum（若分支本质是加权选择）",
                    estimated_cost=estimate_cost("WeightedSum").to_dict(),
                )
            )
    return diagnostics


def validate_jax_traceability(
    plan: PrivacyPlan,
    *,
    example_inputs: Mapping[str, tuple[Any, ...]] | None = None,
    try_lower: bool = True,
) -> tuple[list[Diagnostic], dict[str, Any]]:
    """第 2 类：生成代码能否被 jax.jit 追踪。"""

    from backends.jax_backend import (
        check_traceable,
        generate_for_plan,
        load_generated_function,
        static_check_source,
    )

    diagnostics: list[Diagnostic] = []
    checks: dict[str, Any] = {}

    generation = generate_for_plan(plan)
    for function in generation.functions:
        # 静态自检：不依赖 jax 是否可用
        problems = static_check_source(function.source)
        if problems:
            diagnostics.append(
                Diagnostic(
                    code=DIAG_NOT_TRACEABLE,
                    severity="error",
                    message=f"生成的 {function.op} 代码未通过静态自检",
                    location={"file": f"<generated:{function.name}>", "line": None},
                    cause="; ".join(problems),
                    suggestion="检查代码生成器：禁止 numpy 依赖与 Python 级控制流。",
                    suggested_op=function.op,
                    estimated_cost=estimate_cost(function.op).to_dict(),
                )
            )
            continue

        examples = (example_inputs or {}).get(function.op)
        if examples is None:
            checks[function.op] = {"traceable": None, "skipped": "未提供样例输入"}
            diagnostics.append(
                Diagnostic(
                    code=DIAG_NOT_TRACEABLE,
                    severity="warning",
                    message=f"{function.op} 未提供样例输入，跳过 jax.jit 追踪验证",
                    location={"file": f"<generated:{function.name}>", "line": None},
                    cause="追踪验证需要具体形状的样例张量。",
                    suggestion="在 validate_jax_traceability(example_inputs=...) 中补上样例。",
                    suggested_op=function.op,
                )
            )
            continue

        fn = load_generated_function(function.source, function.name)
        check = check_traceable(fn, examples, try_lower=try_lower)
        checks[function.op] = check
        if not check.traceable:
            diagnostics.append(
                Diagnostic(
                    code=DIAG_NOT_TRACEABLE,
                    severity="error",
                    message=f"生成的 {function.op} 代码无法被 jax.jit 追踪",
                    location={"file": f"<generated:{function.name}>", "line": None},
                    cause=f"{check.error_type}: {check.error}",
                    suggestion=(
                        "生成代码必须只用 jnp 且无 Python 级分支；"
                        "形状相关量应作为张量参数传入。"
                    ),
                    suggested_op=function.op,
                    estimated_cost=estimate_cost(function.op).to_dict(),
                )
            )
        elif check.error:
            diagnostics.append(
                Diagnostic(
                    code=DIAG_SPU_UNSUPPORTED,
                    severity="warning",
                    message=f"{function.op} 可被 jit 追踪，但无法降级为 SPU 前端的 HLO",
                    location={"file": f"<generated:{function.name}>", "line": None},
                    cause=check.error,
                    suggestion="核对 docs/SPU_CAPABILITY.md 中的平台与版本约束。",
                    suggested_op=function.op,
                    estimated_cost=estimate_cost(function.op).to_dict(),
                )
            )

    return diagnostics, checks


def validate_spu_capability(
    plan: PrivacyPlan, *, report: Any = None
) -> tuple[list[Diagnostic], dict[str, Any]]:
    """第 3 类：SPU 版本/平台是否支持所需算子。

    严重级别刻意区分：
        环境阻断（平台/版本/未安装）→ warning：不是用户代码的问题，
            换到受支持环境即可，不应阻止用户继续编译取证。
        原语阻断（生成代码用到未适配原语）→ error：必须改代码或改算子。
    """

    from backends.spu_backend import check_capabilities, check_operation_capability

    report = report or check_capabilities()
    diagnostics: list[Diagnostic] = []
    detail: dict[str, Any] = {"environment": report.to_dict(), "operations": {}}

    for step in plan.steps:
        capability = check_operation_capability(step.operation, report)
        detail["operations"][step.operation] = capability.to_dict()

        if capability.supported:
            continue

        # 只保留该算子自身的问题（环境阻断在报告里统一陈述，避免每算子重复刷屏）
        own_blockers = [b for b in capability.blockers if b not in report.blockers]
        severity = "error" if own_blockers else "warning"

        if own_blockers:
            message = f"算子 {step.operation} 用到未在 SPU 适配清单中的原语"
            cause = "原语阻断：" + "; ".join(own_blockers)
            suggestion = (
                "改写生成代码，只使用 SPU 已适配的 jnp 原语；"
                "或改用语义等价且原语在清单内的算子。"
            )
        else:
            message = f"算子 {step.operation} 在当前环境不可执行（环境阻断）"
            cause = "环境阻断：" + "; ".join(report.blockers)
            suggestion = (
                "在受支持环境重跑：Linux/WSL2 + Python 3.10 或 3.11 + jax 0.4.34 + spu；"
                "解密态结果需在机群上复算。"
            )

        diagnostics.append(
            Diagnostic(
                code=DIAG_SPU_UNSUPPORTED,
                severity=severity,
                message=message,
                location=dict(step.location) if step.location else {},
                cause=cause,
                suggestion=suggestion,
                suggested_op=step.operation,
                estimated_cost=estimate_cost(step.operation).to_dict(),
            )
        )

    return diagnostics, detail


# --------------------------------------------------------------------------
# 汇总入口
# --------------------------------------------------------------------------


def validate_all(
    program: GeoProgram,
    plan: PrivacyPlan,
    *,
    example_inputs: Mapping[str, tuple[Any, ...]] | None = None,
    capability_report: Any = None,
    upstream_diagnostics: Sequence[Diagnostic] = (),
) -> ValidationReport:
    """执行全部编译前验证。

    upstream_diagnostics 用于把解析/规划阶段已产生的诊断并入同一份报告，
    使 ValidationReport 成为唯一的"编译前问题汇总面"（CLI 与测试都只看它）。
    """

    report = ValidationReport()

    # 上游（解析/规划）的诊断只并入一次。调用方可能已经把 plan.diagnostics
    # 放进 upstream_diagnostics，此处按对象身份去重，避免同一问题在报告里翻倍。
    upstream: list[Diagnostic] = list(upstream_diagnostics)
    if plan is not None:
        for diagnostic in plan.diagnostics:
            if not any(diagnostic is existing for existing in upstream):
                upstream.append(diagnostic)
    report.diagnostics.extend(upstream)

    # 1 + 5：算子支持与后端规则（plan 诊断已在上方并入，此处不再重复）
    report.diagnostics.extend(validate_program(program))
    # 4：控制流
    report.diagnostics.extend(validate_control_flow(program))
    # 6：高度层号是否装得进 Z 位域（三维格网接入层独有）
    report.diagnostics.extend(validate_height_layer_capacity(program))
    # 2：JAX 追踪
    trace_diags, trace_checks = validate_jax_traceability(
        plan, example_inputs=example_inputs
    )
    report.diagnostics.extend(trace_diags)
    report.trace_checks = trace_checks
    # 3：SPU 能力
    spu_diags, spu_detail = validate_spu_capability(plan, report=capability_report)
    report.diagnostics.extend(spu_diags)
    report.capability = spu_detail

    report.checked_ops = tuple(step.operation for step in plan.steps)
    # 同一问题可能被 planner 与 validate_program 各自报出（措辞不同、位置相同）。
    # 报告里只留一条：首次出现的那条来自更接近源码的阶段，位置与描述更精确。
    report.diagnostics = _dedupe(report.diagnostics)
    return report


#: 只按错误码去重：同一算子缺规则，planner 与 validate_program 会各报一次
_DEDUPE_BY_CODE_ONLY = frozenset({DIAG_BACKEND_MISSING})

#: 按"错误码 + 位置"去重：同一处控制流的"语句级"与"算子级"两条合并成一条
_DEDUPE_BY_LOCATION = frozenset({DIAG_DYNAMIC_CONTROL_FLOW})


def _dedupe(diagnostics: Sequence[Diagnostic]) -> list[Diagnostic]:
    """按等价键去重，保留首次出现的诊断（顺序稳定，可读性不变）。

    去重严格限制在已知会重复的类别上：其余诊断按"错误码+位置+消息"
    三重键去重，因此措辞不同的多条提示不会被误删。
    """

    seen: set[str] = set()
    out: list[Diagnostic] = []
    for diagnostic in diagnostics:
        location = diagnostic.location or {}
        location_key = "%s|%s|%s|%s" % (
            diagnostic.code,
            location.get("file"),
            location.get("line"),
            location.get("col"),
        )
        if diagnostic.code in _DEDUPE_BY_CODE_ONLY:
            key = diagnostic.code
        elif diagnostic.code in _DEDUPE_BY_LOCATION:
            key = location_key
        else:
            key = location_key + "|" + diagnostic.message
        if key in seen:
            continue
        seen.add(key)
        out.append(diagnostic)
    return out
