"""geosecure：地理隐私计算编译器主流程。

    Python Source → AST → Geo-IR → Privacy Plan → JAX → SPU simulation

六阶段与 CLI 输出标题一一对应：
    Parsing / IR generation / Privacy planning / JAX generation /
    SPU capability check / SPU simulation
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from backends.jax_backend import (
    check_generated_source,
    generate_for_plan,
    render_module,
    run_jax_jit,
    static_check_source,
)
from backends.plain import run_plain
from backends.psi_backend import (
    PSI_OPS,
    PSI_RUNTIME_WORLD_SIZE,
    PsiCapabilityReport,
    PsiRunResult,
    PsiRuntimeConfig,
    PartyManager,
    ResolvedCellInput,
    check_psi_capabilities,
    check_psi_protocol_capability,
    normalize_inputs,
    normalize_psi_protocol,
    protocol_is_exact,
    protocol_world_size,
    psi_curve_relation,
    resolve_subset_mode,
    resolve_psi_protocol,
    runnable_protocols_hint,
    run_psi_operation,
    same_party_both_sides,
    validate_psi_protocol_params,
)
from backends.psi_backend.subset_mpc import SUBSET_MODE_MPC
from backends.spu_backend import (
    CapabilityReport,
    SpuRunResult,
    check_capabilities,
    check_operation_capability,
    run_spu_simulation,
)
from frontend import ParseResult, parse_source
from frontend.analyzer import _coerce_geotype, _coerce_sensitivity
from ir import GeoProgram, encode_grid_code, render_table
from planner import (
    MPC_RULE_DEFAULT_PROTOCOL,
    PLAN_TABLE_HEADERS,
    PlannedStep,
    PrivacyPlan,
    plan_program,
    plan_table_rows,
)
from validator import ValidationReport, validate_all

# --------------------------------------------------------------------------
# 阶段定义
# --------------------------------------------------------------------------

STAGE_TITLES = {
    "parsing": "Parsing",
    "ir": "IR generation",
    "planning": "Privacy planning",
    "jax": "JAX generation",
    "spu_capability": "SPU capability check",
    "spu_simulation": "SPU simulation",
    "psi_capability": "PSI capability check",
    "psi_simulation": "PSI simulation",
}

STATUS_ICON = {"ok": "+", "warning": "!", "error": "x", "skipped": "-"}

#: PSI 族算子的两方样例格网码（64 位真值，走 ir.values.encode_grid_code 生成）
#: 刻意构造为"有交集但互不包含"，使 Intersects=True / Contains=False，
#: 三个算子的判定方向差异都能被观察到。
PSI_ROUTE_CODES: tuple[int, ...] = (
    encode_grid_code(x=21861, y=27702, z=15, level=9, toff=0, lt=4),
    encode_grid_code(x=21862, y=27702, z=15, level=9, toff=0, lt=4),
    encode_grid_code(x=21863, y=27702, z=15, level=9, toff=0, lt=4),
)
PSI_ZONE_CODES: tuple[int, ...] = (
    encode_grid_code(x=21862, y=27702, z=15, level=9, toff=0, lt=4),
    encode_grid_code(x=21863, y=27702, z=15, level=9, toff=0, lt=4),
    encode_grid_code(x=22999, y=27702, z=15, level=9, toff=0, lt=4),
)

#: 每个算子用于追踪验证与模拟的样例输入
DEFAULT_EXAMPLE_INPUTS: Mapping[str, tuple[Any, ...]] = {
    "DistanceLE": ((1, 2, 3), (1, 3, 3), 2),
    "WeightedSum": ((10, 20, 30), (1, 2, 1)),
    "TemporalOverlap": ((0, 4), (3, 2), (4,), (3,)),
    "Intersects": (PSI_ROUTE_CODES, PSI_ZONE_CODES),
    "Contains": (PSI_ZONE_CODES, PSI_ROUTE_CODES),
    "CellSetIntersect": (PSI_ROUTE_CODES, PSI_ZONE_CODES),
}

#: 明文参考调用的参数拆分位置：TemporalOverlap 需要把节点数组还原为节点列表
PLAIN_ARG_BUILDERS: Mapping[str, Any] = {
    "DistanceLE": lambda a: (list(a[0]), list(a[1]), int(a[2])),
    "WeightedSum": lambda a: (list(a[0]), list(a[1])),
    "TemporalOverlap": lambda a: (
        list(zip(a[0], a[1])),
        list(zip(a[2], a[3])),
    ),
    "Intersects": lambda a: (list(a[0]), list(a[1])),
    "Contains": lambda a: (list(a[0]), list(a[1])),
    "CellSetIntersect": lambda a: (list(a[0]), list(a[1])),
}


@dataclass
class StageResult:
    name: str
    status: str = "ok"
    detail: Any = None
    message: str = ""

    @property
    def title(self) -> str:
        return STAGE_TITLES.get(self.name, self.name)

    @property
    def icon(self) -> str:
        return STATUS_ICON.get(self.status, "?")


@dataclass
class CompileResult:
    """一次完整编译的结果。"""

    source_file: str | None = None
    program: GeoProgram | None = None
    parse_result: ParseResult | None = None
    plan: PrivacyPlan | None = None
    jax_generation: Any = None
    jax_functions: dict[str, Any] = field(default_factory=dict)
    jax_outputs: dict[str, Any] = field(default_factory=dict)
    jax_module: str = ""
    trace_checks: dict[str, Any] = field(default_factory=dict)
    capability: CapabilityReport | None = None
    validation: ValidationReport | None = None
    stages: list[StageResult] = field(default_factory=list)
    spu_runs: dict[str, SpuRunResult] = field(default_factory=dict)
    #: 本次编译实际使用的 PSI 协议与曲线（来自构造参数，不是环境探测所得）
    psi_protocol: str | None = None
    psi_curve: str | None = None
    #: 本次编译实际使用的协议级参数（如 RR22 的 low_comm_mode）
    psi_protocol_params: Mapping[str, Any] = field(default_factory=dict)
    psi_capability: PsiCapabilityReport | None = None
    #: 协议级三层核查结论（PsiProtocolCapability.to_dict()）
    psi_protocol_capability: dict[str, Any] | None = None
    psi_runs: dict[str, PsiRunResult] = field(default_factory=dict)
    #: 本次 PSI 执行的运行时配置快照（PsiRuntimeConfig.to_dict()）。
    #: 各 PSI 步骤配置一致时给出；不一致时为 None，逐 run 的 runtime_config 为准。
    psi_runtime_config: Mapping[str, Any] | None = None
    #: 已规范化的真实输入（名字 → ResolvedCellInput），未绑定输入时为空
    resolved_inputs: dict[str, ResolvedCellInput] = field(default_factory=dict)
    #: 参与方绑定快照（PartyManager.to_dict()）；没有绑定 party_id 时为 None
    parties: dict[str, Any] | None = None
    operator_status: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(stage.status != "error" for stage in self.stages)

    def stage(self, name: str) -> StageResult | None:
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    @property
    def diagnostics(self) -> list[Any]:
        """全部诊断。

        validation 已经是聚合了上游（解析 + 规划）的**唯一汇总面**
        （见 validator.validate_all 的说明）。本属性以前又并了一遍
        parse_result / plan 的诊断，导致同一问题在 result.errors 里翻倍，
        CLI 的 "N 个错误" 也随之虚高。这里只在没有 validation 时回退。
        """

        if self.validation is not None:
            return list(self.validation.diagnostics)
        out: list[Any] = []
        if self.parse_result is not None:
            out.extend(self.parse_result.diagnostics)
        if self.plan is not None:
            out.extend(self.plan.diagnostics)
        return out

    @property
    def errors(self) -> list[Any]:
        return [d for d in self.diagnostics if getattr(d, "severity", "") == "error"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "ok": self.ok,
            "stages": [
                {"name": s.name, "title": s.title, "status": s.status, "message": s.message}
                for s in self.stages
            ],
            "ir": self.program.to_dict() if self.program else None,
            "plan": [
                {
                    "operation": s.operation,
                    "representation": s.representation,
                    "backend": s.backend,
                    "estimated_cost": dict(s.estimated_cost),
                    "security_level": s.security_level,
                    "protocol": s.protocol,
                    "protocol_params": dict(s.protocol_params),
                    "status": s.status,
                }
                for s in (self.plan.steps if self.plan else [])
            ],
            "jax_generated": [
                f.to_dict() for f in (self.jax_generation.functions if self.jax_generation else [])
            ],
            "jax_skipped": list(self.jax_generation.skipped) if self.jax_generation else [],
            "capability": self.capability.to_dict() if self.capability else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "spu_runs": {op: run.to_dict() for op, run in self.spu_runs.items()},
            "psi_protocol": self.psi_protocol,
            "psi_curve": self.psi_curve,
            "psi_protocol_params": dict(self.psi_protocol_params),
            "psi_capability": self.psi_capability.to_dict() if self.psi_capability else None,
            "psi_protocol_capability": self.psi_protocol_capability,
            "psi_runs": {op: run.to_dict() for op, run in self.psi_runs.items()},
            "psi_runtime_config": (
                dict(self.psi_runtime_config) if self.psi_runtime_config else None
            ),
            "inputs": {
                name: value.to_dict() for name, value in self.resolved_inputs.items()
            },
            "parties": dict(self.parties) if self.parties else None,
            "operator_status": list(self.operator_status),
        }


class Compiler:
    """地理隐私计算编译器。"""

    def __init__(
        self,
        *,
        #: MPC（SPU）协议；**None = 按实测代价自动选择**（见 planner.select_mpc_protocol，
        #: 依据是 docs/mpc_comm_baseline.json）。给了具体协议则是显式覆盖。
        protocol: str | None = None,
        field: str | int = 64,
        world_size: int | None = None,
        tolerance: float | None = None,
        run_simulation: bool = True,
        psi_protocol: str | None = None,
        psi_curve: str | None = None,
        psi_subset: str = SUBSET_MODE_MPC,
        psi_rr22_low_comm_mode: bool = False,
        psi_protocol_params: Mapping[str, Any] | None = None,
        inputs: Mapping[str, Any] | None = None,
        input_layouts: Mapping[str, Any] | None = None,
        capability_report: CapabilityReport | None = None,
        psi_capability_report: PsiCapabilityReport | None = None,
        sensitivities: Mapping[str, Any] | None = None,
        type_hints: Mapping[str, Any] | None = None,
    ) -> None:
        #: 编译器**显式**指定的 MPC 协议；None = 让规划器按实测代价选。
        #: 执行期真正使用的具体值见 `protocol_in_use`（规划完一次性定型）。
        self.protocol = protocol
        self._protocol_in_use: str | None = None
        self.field = field
        self.world_size = world_size
        self.tolerance = tolerance
        self.run_simulation = run_simulation
        # PSI 协议/曲线走同一套解析：非法名在此即报错（消息里带可用清单），
        # 不读曲线的协议把曲线归一为 None——避免"设了但不生效"的错觉。
        self.psi_protocol, self.psi_curve = resolve_psi_protocol(psi_protocol, psi_curve)
        # RR22 专用参数（对应 spu.psi.Rr22Rarams.low_comm_mode）。随协议选择
        # 构造进 protocol_params，一路进 Planner 的步骤与 PSI 后端。
        self.psi_rr22_low_comm_mode = bool(psi_rr22_low_comm_mode)
        self.psi_protocol_params: dict[str, Any] = {}
        if self.psi_protocol == "PROTOCOL_RR22":
            self.psi_protocol_params["low_comm_mode"] = self.psi_rr22_low_comm_mode
        # 显式协议参数通道（如审计演练 receiver_rank=1）。与 RR22 开关合并后
        # 统一静态校验：未登记键/非法值在编译前拒绝，不静默忽略、不等到 Runtime。
        self.psi_protocol_params.update(dict(psi_protocol_params or {}))
        param_check = validate_psi_protocol_params(
            self.psi_protocol, self.psi_protocol_params
        )
        if not param_check.ok:
            raise ValueError("PSI 协议参数非法：" + "；".join(param_check.problems))
        self._psi_param_notes = tuple(param_check.notes)
        # 真实格网输入绑定（Phase 2）：名字 → CSV/JSON/CellSet/码序列。
        # 构造期即解析：文件缺失/损坏立即失败（错误带具体位置），
        # 不把坏数据拖到 PSI 执行阶段才炸。
        self.inputs = dict(inputs or {})
        self.input_layouts = {
            str(name): _load_layout_manifest(value, name=str(name))
            for name, value in (input_layouts or {}).items()
        }
        self._resolved_inputs: dict[str, ResolvedCellInput] = (
            normalize_inputs(self.inputs) if self.inputs else {}
        )
        if self.input_layouts:
            unknown = sorted(set(self.input_layouts) - set(self._resolved_inputs))
            if unknown:
                raise ValueError(
                    f"input_layouts 指定的名字未绑定输入：{unknown}；"
                    f"已绑定输入：{sorted(self._resolved_inputs)}"
                )
            # 显式布局清单覆盖输入自带清单：这是跨方握手用的声明面。
            self._resolved_inputs = {
                name: (
                    replace(value, layout=self.input_layouts[name])
                    if name in self.input_layouts
                    else value
                )
                for name, value in self._resolved_inputs.items()
            }
        # 参与方绑定（§14 / §16）：把 PartyInput(party_id=...) 的标签登记成
        # 两方抽象；超过两方在这里显式失败（不静默截断）。未标注 party_id 的
        # 输入不登记；left / right 定位仍由算子输入顺序决定，不被标签改写。
        self.parties = PartyManager.from_inputs(self._resolved_inputs)
        # `Contains` 的子集判定走哪条路。默认 MPC；显式选明文时也照做，
        # 但状态词会变成 subset-plaintext，不会被读成密态子集比较。
        self.psi_subset = resolve_subset_mode(psi_subset)
        self._capability = capability_report
        self._psi_capability = psi_capability_report
        # 这两项以前被 *kwargs 吃掉且静默丢弃：调用方以为给了密态级别，
        # 实际拿到的是默认级别，安全判定因此失真。现在显式接住并校验。
        self.sensitivities = {
            k: _coerce_sensitivity(v) for k, v in (sensitivities or {}).items()
        }
        self.type_hints = {k: _coerce_geotype(v) for k, v in (type_hints or {}).items()}

    # ---------------- 主流程 ----------------

    @property
    def protocol_in_use(self) -> str:
        """执行期实际使用的 MPC 协议（具体值，不会为 None）。

        `self.protocol` 为 None 时取**方案里按实测代价选出**的协议；
        方案也没有（纯 PSI 程序，没有 MPC 步骤）→ 回落到登记默认值。
        """

        return (
            self._protocol_in_use
            or self.protocol
            or MPC_RULE_DEFAULT_PROTOCOL
        )

    def compile_source(
        self, source: str, filename: str = "<source>", *, entry: str | None = None
    ) -> CompileResult:
        result = CompileResult(source_file=None if filename == "<source>" else filename)
        result.psi_protocol = self.psi_protocol
        result.psi_curve = self.psi_curve
        result.psi_protocol_params = dict(self.psi_protocol_params)
        result.resolved_inputs = dict(self._resolved_inputs)
        result.parties = (
            self.parties.to_dict() if self.parties.descriptors() else None
        )

        # ---- 阶段 1：Parsing ----
        parse_result = parse_source(
            source,
            filename,
            entry=entry,
            type_hints=self.type_hints,
            sensitivities=self.sensitivities,
        )
        result.parse_result = parse_result
        result.program = parse_result.program
        errors = parse_result.errors
        result.stages.append(
            StageResult(
                name="parsing",
                status="error" if errors else ("warning" if parse_result.warnings else "ok"),
                detail=parse_result,
                message=(
                    f"{len(parse_result.functions)} 个函数，"
                    f"{len(parse_result.program.operations)} 个算子调用"
                    + (f"，{len(errors)} 个错误" if errors else "")
                ),
            )
        )

        if errors:
            # 不静默继续：后续阶段依赖 Geo-IR，缺失时只会产生噪音
            for name in (
                "ir",
                "planning",
                "jax",
                "spu_capability",
                "spu_simulation",
                "psi_capability",
                "psi_simulation",
            ):
                result.stages.append(
                    StageResult(name, "skipped", message="解析未通过，本阶段已跳过")
                )
            result.validation = validate_all(
                parse_result.program,
                PrivacyPlan(),
                upstream_diagnostics=parse_result.diagnostics,
            )
            return result

        # ---- 阶段 2：IR generation ----
        result.stages.append(
            StageResult(
                "ir",
                "ok",
                parse_result.program,
                f"GeoProgram {parse_result.program.name}："
                f"{len(parse_result.program.entity_inputs)} 输入 / "
                f"{len(parse_result.program.operations)} 算子 / "
                f"{len(parse_result.program.relations)} 关系",
            )
        )

        # ---- 阶段 3：Privacy planning ----
        plan = plan_program(
            parse_result.program,
            psi_protocol=self.psi_protocol,
            psi_protocol_params=self.psi_protocol_params,
            # MPC（SPU）协议同样进规划层：让"算子 × 协议"在编译期就被校验，
            # 而不是等到 SPU 模拟阶段才以运行期错误的形式晚到。
            mpc_protocol=self.protocol,
        )
        result.plan = plan
        result.stages.append(
            StageResult(
                "planning",
                "error" if plan.has_errors else "ok",
                plan,
                f"{len(plan.steps)} 步方案，{len(plan.crypto_steps)} 步需密态",
            )
        )

        # 执行期协议一次性定型：显式指定 > 方案里的实测选择 > 登记默认值。
        # "自动选择"的依据来自方案本身（planner 已按实测通信量选过），这里只把
        # 方案的结论取出来交给执行层，不另做一次判断——两道闸门只留一道判断源。
        self._protocol_in_use = self.protocol or next(
            (step.mpc_protocol for step in plan.steps if step.mpc_protocol), None
        )

        # ---- 阶段 4：JAX generation ----
        self._stage_jax(result)

        # ---- 阶段 5：SPU capability check ----
        self._stage_capability(result)

        # ---- 阶段 6：SPU simulation ----
        self._stage_simulation(result)

        # ---- 阶段 7/8：PSI capability check + PSI simulation ----
        # PSI 与 SPU-MPC 是两条独立的执行路径：PSI 走 spu.psi.psi_execute
        # （文件接口、两方求交），不是 jax.jit → SPU 虚拟机那条路。
        self._stage_psi_capability(result)
        self._stage_psi_simulation(result)

        # ---- 跨阶段验证汇总 ----
        upstream = list(parse_result.diagnostics) + list(plan.diagnostics)
        result.validation = validate_all(
            parse_result.program,
            plan,
            example_inputs=self._example_inputs(parse_result.program),
            capability_report=result.capability,
            upstream_diagnostics=upstream,
        )
        self._build_operator_status(result)
        return result

    def compile_file(self, path: str, *, entry: str | None = None) -> CompileResult:
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        return self.compile_source(source, path, entry=entry)

    # ---------------- 阶段 4 ----------------

    def _stage_jax(self, result: CompileResult) -> None:
        if result.plan is None:
            result.stages.append(StageResult("jax", "skipped", message="无隐私方案"))
            return

        generation = generate_for_plan(result.plan)
        result.jax_generation = generation
        result.jax_module = render_module(generation, source_plan=result.plan)

        problems: list[str] = []
        for function in generation.functions:
            problems.extend(static_check_source(function.source))
            examples = self._example_inputs_for(function.op)
            fn, check = check_generated_source(function.source, function.name, examples)
            result.trace_checks[function.op] = check
            if fn is None:
                if check.error:
                    problems.append(f"{function.op}: {check.error}")
                continue
            result.jax_functions[function.op] = fn
            if check.traceable:
                try:
                    result.jax_outputs[function.op] = run_jax_jit(fn, examples)
                except Exception as exc:
                    problems.append(f"{function.op} jit 执行失败：{exc}")

        status = "error" if problems else ("warning" if generation.skipped else "ok")
        message = f"生成 {len(generation.functions)} 个函数"
        if generation.skipped:
            ops = ", ".join(item["operation"] for item in generation.skipped)
            message += f"；{len(generation.skipped)} 个算子不生成（{ops}）"
        if problems:
            message += f"；{len(problems)} 个问题"
        result.stages.append(StageResult("jax", status, generation, message))

    # ---------------- 阶段 5 ----------------

    def _stage_capability(self, result: CompileResult) -> None:
        if self._capability is None:
            self._capability = check_capabilities()
        result.capability = self._capability

        detail = {
            "report": self._capability.to_dict(),
            "operations": {
                step.operation: check_operation_capability(step.operation, self._capability).to_dict()
                for step in (result.plan.steps if result.plan else [])
            },
        }
        blockers = self._capability.blockers
        result.stages.append(
            StageResult(
                "spu_capability",
                "ok" if self._capability.runnable else "warning",
                detail,
                self._capability.status
                + (f"（{len(blockers)} 项阻断）" if blockers else ""),
            )
        )

    # ---------------- 阶段 6 ----------------

    def _stage_simulation(self, result: CompileResult) -> None:
        if result.plan is None or result.plan.has_errors:
            result.stages.append(
                StageResult("spu_simulation", "skipped", message="隐私方案有错误，跳过模拟")
            )
            return
        if not self.run_simulation:
            result.stages.append(StageResult("spu_simulation", "skipped", message="按要求跳过模拟"))
            return

        planned_but_not_jax: list[str] = []
        not_routed: list[str] = []
        for step in result.plan.steps:
            if _is_plaintext_local(step.operation):
                # 物化算子不进密态：既没有 JAX 实现，也不该算作"缺 JAX 实现"，
                # 否则它会被并进下面的告警里，读起来像是方案有缺口。
                not_routed.append(step.operation)
                continue
            fn = result.jax_functions.get(step.operation)
            examples = self._example_inputs_for(step.operation)
            if fn is None or not examples:
                planned_but_not_jax.append(step.operation)
                continue

            import numpy as np

            plain_inputs = self._plain_args(step.operation, examples)
            reference = _plain_reference(step.operation, plain_inputs)

            def reference_fn(*args: Any, _op: str = step.operation) -> Any:
                # 必须先走与明文参考同一套参数拆解（PLAIN_ARG_BUILDERS）。
                # 否则多参数算子（如 TemporalOverlap 需把 4 个节点数组
                # 还原为 2 个节点列表）会抛异常，被 _plain_reference 吞掉
                # 返回 None，进而误报为 SPU 执行失败。
                return _plain_reference(_op, self._plain_args(_op, args))

            try:
                run = run_spu_simulation(
                    fn,
                    [np.asarray(x) for x in examples],
                    protocol=self.protocol_in_use,
                    field=self.field,
                    world_size=self.world_size,
                    reference_fn=reference_fn if reference is not None else None,
                    tolerance=self.tolerance,
                    report=self._capability,
                )
            except ValueError as exc:
                # run_spu_simulation 对非法协议/环宽是 fail-fast 抛 ValueError
                # （有测试守着这条契约）。编译器这一层接住它并转成可读错误，
                # 否则整份编译输出会被 traceback 冲掉——实测
                # `geo-secure build x.py --protocol SPDZ2K` 就是这样崩的。
                run = SpuRunResult(
                    status="error",
                    protocol=self.protocol_in_use,
                    field=str(self.field),
                    world_size=self.world_size or 2,
                    error=str(exc),
                )
            result.spu_runs[step.operation] = run

        if not result.spu_runs:
            # 区分两种"没有 SPU 模拟"：
            #   PSI 族算子本来就不走 JAX/SPU 这条路，另有 PSI 后端承接；
            #   其余算子才是真的既不生成 JAX 也无后端，需要显式点出来。
            psi_ops = [op for op in planned_but_not_jax if op in PSI_OPS]
            other_ops = [op for op in planned_but_not_jax if op not in PSI_OPS]

            message = "没有可模拟的 JAX 实现"
            if psi_ops:
                message += f"；{', '.join(psi_ops)} 属 PSI 族，由 PSI 阶段执行"
            if other_ops:
                message += f"；{', '.join(other_ops)} 无 JAX 实现，未进入 JAX/SPU 路径"
            if not_routed:
                message += f"；{', '.join(not_routed)} 为明文物化算子，不走密态路径"

            result.stages.append(
                StageResult("spu_simulation", "skipped", result.spu_runs, message)
            )
            return

        if any(run.status == "unavailable" for run in result.spu_runs.values()):
            status, message = "warning", "当前环境不可运行 SPU，结果栏位留空（见能力核查）"
        elif all(run.ok for run in result.spu_runs.values()):
            status = "ok"
            message = f"{len(result.spu_runs)} 个算子在 SPU 模拟器上验证通过"
        else:
            status = "error"
            message = f"{len(result.spu_runs)} 个算子执行失败"

        if planned_but_not_jax:
            message += f"；{', '.join(planned_but_not_jax)} 无 JAX 实现，未进入 JAX/SPU 路径"
        if not_routed:
            message += f"；{', '.join(not_routed)} 为明文物化算子，不走密态路径"
        result.stages.append(StageResult("spu_simulation", status, result.spu_runs, message))

    # ---------------- 阶段 7 / 8：PSI ----------------

    def _stage_psi_capability(self, result: CompileResult) -> None:
        """核查 PSI 执行能力；只对方案中真正走 PSI 的算子报结论。"""

        if result.plan is None:
            result.stages.append(StageResult("psi_capability", "skipped", message="无隐私方案"))
            return

        psi_steps = [step for step in result.plan.steps if step.operation in PSI_OPS]
        if not psi_steps:
            result.stages.append(
                StageResult(
                    "psi_capability",
                    "skipped",
                    message="方案中没有走 PSI 的算子",
                )
            )
            return

        if self._psi_capability is None:
            self._psi_capability = check_psi_capabilities()
        report = self._psi_capability
        result.psi_capability = report

        # 三层能力核查（后端 / 协议 / 参数），一层一结论——不再用一个总
        # runnable 表示所有层次。协议参数取**计划步骤**的合并结果
        # （规则默认值 + 编译器参数），与 Runtime 真正注入的同源。
        params = dict(psi_steps[0].protocol_params)
        protocol_cap = check_psi_protocol_capability(
            self.psi_protocol, params, report, curve=self.psi_curve
        )
        result.psi_protocol_capability = protocol_cap.to_dict()
        detail = {
            "report": report.to_dict(),
            "protocol_capability": protocol_cap.to_dict(),
        }
        layers = (
            f"分层核查：backend={'是' if protocol_cap.backend_runnable else '否'} / "
            f"protocol={'是' if protocol_cap.protocol_runnable else '否'} / "
            f"params={'是' if protocol_cap.params_runnable else '否'}"
        )
        notes = [*self._psi_param_notes, *protocol_cap.notes]
        notes_suffix = ("；" + "；".join(notes)) if notes else ""

        # 选用的协议是否在这条链路上可执行。`ECDH_3PC` 这类三方协议
        # 属"枚举里有、这里跑不了"，必须在方案阶段就说清楚，
        # 而不是等运行时抛 libpsi 的 C++ 栈。
        required = protocol_world_size(self.psi_protocol)
        if required > PSI_RUNTIME_WORLD_SIZE:
            result.stages.append(
                StageResult(
                    "psi_capability",
                    "error",
                    detail,
                    f"选用的 PSI 协议 {self.psi_protocol} 需要 {required} 个参与方，"
                    f"本链路固定 {PSI_RUNTIME_WORLD_SIZE} 方；"
                    f"可改用 {runnable_protocols_hint()}",
                )
            )
            return

        ops = ", ".join(step.operation for step in psi_steps)
        selection = f"选用 {self.psi_protocol}" + curve_suffix(
            self.psi_protocol, self.psi_curve
        )
        if self.psi_protocol_params:
            selection += "（" + ", ".join(
                f"{key}={value}" for key, value in self.psi_protocol_params.items()
            ) + "）"
        if not protocol_is_exact(self.psi_protocol):
            # 带噪协议与三方协议同属"名字有、语义不一样"：
            # 一个跑不起来，一个跑起来但不准。都要在方案阶段就说清楚。
            result.stages.append(
                StageResult(
                    "psi_capability",
                    "warning",
                    detail,
                    f"{len(psi_steps)} 个算子需 PSI（{ops}）；{selection}；"
                    f"{self.psi_protocol} 为差分隐私协议，结果带噪，"
                    f"不能作为与明文一致的验证依据；{layers}{notes_suffix}",
                )
            )
            return
        result.stages.append(
            StageResult(
                "psi_capability",
                "ok" if protocol_cap.runnable else "warning",
                detail,
                f"{len(psi_steps)} 个算子需 PSI（{ops}）；{selection}；{layers}；"
                + (
                    f"当前环境可执行（spu {report.version}）"
                    if protocol_cap.runnable
                    else f"{len(protocol_cap.blockers)} 项阻断，PSI 结果将留空"
                )
                + notes_suffix,
            )
        )

    def _stage_psi_simulation(self, result: CompileResult) -> None:
        """对走 PSI 的算子做真实两方求交验证（真实输入 + 链式数据流）。

        Phase 2/3：实参按**输入名**装配——用户绑定的真实数据（inputs=）
        与上一步 PSI 输出优先；都取不到才回退样例默认值，并在 run.notes
        如实披露来源。不再固定吃 DEFAULT_EXAMPLE_INPUTS。
        """

        if result.plan is None or result.plan.has_errors:
            result.stages.append(
                StageResult("psi_simulation", "skipped", message="隐私方案有错误，跳过模拟")
            )
            return
        if not self.run_simulation:
            result.stages.append(StageResult("psi_simulation", "skipped", message="按要求跳过模拟"))
            return

        required = protocol_world_size(self.psi_protocol)
        if required > PSI_RUNTIME_WORLD_SIZE:
            result.stages.append(
                StageResult(
                    "psi_simulation",
                    "skipped",
                    message=(
                        f"选用的 PSI 协议 {self.psi_protocol} 需 {required} 个参与方，"
                        f"本链路固定 {PSI_RUNTIME_WORLD_SIZE} 方，未执行"
                    ),
                )
            )
            return

        steps = [step for step in result.plan.steps if step.operation in PSI_OPS]
        if not steps:
            result.stages.append(
                StageResult("psi_simulation", "skipped", message="方案中没有走 PSI 的算子")
            )
            return

        # 运行时值表：输入名 → 值。真实输入与链式输出的唯一装配点。
        runtime_values: dict[str, _RuntimeValue] = {
            name: _RuntimeValue(
                codes=value.codes,
                origin=f"输入绑定（{value.source or '用户数据'}）",
                layout=value.layout,
                party_id=value.party_id,
            )
            for name, value in self._resolved_inputs.items()
        }

        run_keys = _psi_run_keys(result.plan)
        configs: dict[str, PsiRuntimeConfig] = {}
        config_problems: list[str] = []
        produced_names: set[str] = set()

        for index, step in enumerate(result.plan.steps):
            if step.operation not in PSI_OPS:
                continue
            op = step.operation
            key = run_keys[index]
            if step.output_name:
                produced_names.add(step.output_name)
            values = self._resolve_psi_inputs(step, runtime_values, produced_names)
            if values is None:
                result.psi_runs[key] = PsiRunResult(
                    status="error",
                    op=op,
                    protocol=normalize_psi_protocol(step.protocol or self.psi_protocol),
                    curve=self.psi_curve,
                    world_size=PSI_RUNTIME_WORLD_SIZE,
                    receiver_rank=0,
                    error=(
                        f"输入无法装配：{list(step.inputs)} 既不在绑定输入 "
                        f"{sorted(self._resolved_inputs)}，也不是上一步输出；"
                        "请用 inputs={...} 绑定数据或确认样例兜底存在"
                    ),
                    notes=("未执行 PSI：输入装配失败在进入运行时之前拦下",),
                )
                continue

            config = PsiRuntimeConfig.from_params(
                normalize_psi_protocol(step.protocol or self.psi_protocol),
                self.psi_curve,
                step.protocol_params,
            )
            configs[key] = config

            left, right = values[0], values[1]
            plain_inputs = (list(left.codes), list(right.codes))
            reference = _plain_reference(op, plain_inputs)

            def reference_fn(left_codes: Any, right_codes: Any, _op: str = op) -> Any:
                return _plain_reference(_op, (left_codes, right_codes))

            if op == "Contains":
                args = {"outer": plain_inputs[0], "inner": plain_inputs[1]}
            else:
                args = {"left": plain_inputs[0], "right": plain_inputs[1]}

            binding = tuple(
                {
                    "role": role,
                    "name": name,
                    "party_id": item.party_id,
                }
                for role, name, item in (
                    (
                        "left",
                        step.inputs[0] if len(step.inputs) > 0 else None,
                        left,
                    ),
                    (
                        "right",
                        step.inputs[1] if len(step.inputs) > 1 else None,
                        right,
                    ),
                )
            )
            run = run_psi_operation(
                op,
                args,
                config=config,
                left_layout=left.layout,
                right_layout=right.layout,
                party_binding=binding,
                reference_fn=reference_fn if reference is not None else None,
                subset_via=self.psi_subset,
                # 子集判定这条 MPC 电路沿用编译器的协议/环宽，与其它 MPC 算子
                # 共用同一套开关，不另开一个配置面。
                mpc_protocol=self.protocol_in_use,
                mpc_field=self.field,
                mpc_world_size=self.world_size,
                mpc_report=self._capability,
            )

            # 输入来源与参与方逐条披露：真实绑定 / 上一步 PSI 输出 / 样例兜底，
            # 以及这份输入属于哪个参与方（§14：party_id 逐输入带出，不改装配）。
            for name, value in zip(step.inputs, values):
                note = f"输入 {name} 来源：{value.origin}"
                if value.party_id:
                    note += f"；参与方：{value.party_id}"
                run.notes = run.notes + (note,)
            if same_party_both_sides(left.party_id, right.party_id):
                run.notes = run.notes + (
                    f"两侧输入的 party_id 相同（{left.party_id}）：本步骤在语义上"
                    "不是跨方求交，请核对 PartyInput 绑定",
                )

            # 配置闭环逐字段核对：Planner 的 protocol_params → Runtime 实际配置。
            config_problems.extend(_check_config_closure(step, config, run))

            result.psi_runs[key] = run

            # 链式数据流：交集本体（CellSetIntersect 的输出）喂给下游算子，
            # 而不是下游重新读样例。
            if op == "CellSetIntersect" and step.output_name and run.value is not None:
                runtime_values[step.output_name] = _RuntimeValue(
                    codes=tuple(int(code) for code in run.value),
                    origin=f"上一步 PSI 输出（{key}）",
                    layout=None,
                )

        if configs:
            first = next(iter(configs.values()))
            if all(item.to_dict() == first.to_dict() for item in configs.values()):
                result.psi_runtime_config = first.to_dict()

        # ---------------- 阶段结论 ----------------
        if config_problems:
            result.stages.append(
                StageResult(
                    "psi_simulation",
                    "error",
                    result.psi_runs,
                    "配置闭环核对失败：" + "；".join(config_problems),
                )
            )
            return

        if any(run.status == "unavailable" for run in result.psi_runs.values()):
            status = "warning"
            message = "当前环境不可运行 PSI，结果栏位留空（见能力核查）"
        elif all(run.ok or run.status == "empty-input" for run in result.psi_runs.values()):
            if protocol_is_exact(self.psi_protocol):
                status = "ok"
                message = f"{len(result.psi_runs)} 个算子经真实 PSI 求交验证"
            else:
                # 跑了，但"验证"二字用不得：带噪协议与明文不一致是设计行为。
                status = "warning"
                message = (
                    f"{len(result.psi_runs)} 个算子执行完成；"
                    f"所选协议 {self.psi_protocol} 结果带噪，"
                    "不作为与明文一致的验证依据"
                )
        else:
            status = "error"
            message = f"{len(result.psi_runs)} 个算子 PSI 执行失败"

        chained_keys = sorted(
            key
            for key, run in result.psi_runs.items()
            if any("上一步 PSI 输出" in note for note in run.notes)
        )
        if chained_keys:
            message += f"；链式输入：{', '.join(chained_keys)} 使用上一步 PSI 输出"
        fallback_keys = sorted(
            key
            for key, run in result.psi_runs.items()
            if any("样例兜底" in note for note in run.notes)
        )
        if fallback_keys:
            message += (
                f"；{', '.join(fallback_keys)} 的部分输入未绑定真实数据，"
                "回退到样例默认值（见各 run 的 notes）"
            )

        # 有算子的子集判定没走 MPC 时，阶段行不能说成单纯"求交验证"就完事。
        plaintext_subsets = sorted(
            run.op
            for run in result.psi_runs.values()
            if run.subset is not None and not run.subset.is_mpc
        )
        if plaintext_subsets:
            message += (
                f"；{', '.join(plaintext_subsets)} 的子集判定未走 MPC，"
                "具体模式见各 run 的 subset 字段"
            )

        result.stages.append(StageResult("psi_simulation", status, result.psi_runs, message))

    def _resolve_psi_inputs(
        self,
        step: PlannedStep,
        runtime_values: Mapping[str, _RuntimeValue],
        produced_names: set[str],
    ) -> list[_RuntimeValue] | None:
        """按**输入名**装配一步 PSI 的实参：绑定输入/链式输出优先，样例最后。

        返回 None = 无法装配：调用方据此构造错误 run，绝不以空输入进入协议。
        样例兜底只适用于**程序实体输入**（用户没绑定）；如果这个名字是上一步
        算子的输出，用样例顶替等于把链路失败替换成样例结果——直接拒绝。
        """

        defaults = self._example_inputs_for(step.operation)
        values: list[_RuntimeValue] = []
        for index, name in enumerate(step.inputs):
            if name in runtime_values:
                values.append(runtime_values[name])
            elif name in produced_names:
                # 上一步输出未能产出可用值（执行失败/未产出集合）：不兜底。
                return None
            elif defaults and index < len(defaults):
                suffix = (
                    "——不得据此宣称真实数据验证" if self._resolved_inputs else ""
                )
                values.append(
                    _RuntimeValue(
                        codes=tuple(int(code) for code in defaults[index]),
                        origin=f"样例兜底（{name} 未绑定真实输入）{suffix}",
                        layout=None,
                    )
                )
            else:
                return None
        if len(values) < 2:
            return None
        return values

    def _plain_args(self, op: str, examples: Sequence[Any]) -> tuple[Any, ...]:
        builder = PLAIN_ARG_BUILDERS.get(op)
        if builder is None:
            return tuple(examples)
        return builder(examples)

    def _example_inputs(self, program: GeoProgram) -> dict[str, tuple[Any, ...]]:
        out: dict[str, tuple[Any, ...]] = {}
        for operation in program.operations:
            if operation.op in DEFAULT_EXAMPLE_INPUTS and operation.op not in out:
                out[operation.op] = DEFAULT_EXAMPLE_INPUTS[operation.op]
        return out

    def _example_inputs_for(self, op: str) -> tuple[Any, ...]:
        return DEFAULT_EXAMPLE_INPUTS.get(op, ())

    # ---------------- 最终状态表 ----------------

    def _build_operator_status(self, result: CompileResult) -> None:
        """构造课题要求的最终输出：Operation / Representation / Backend / Status。"""

        rows: list[dict[str, Any]] = []
        plan = result.plan
        capability = result.capability
        if plan is None:
            result.operator_status = rows
            return

        skipped_ops = {
            item["operation"] for item in (result.jax_generation.skipped if result.jax_generation else [])
        }

        run_keys = _psi_run_keys(plan)
        for index, step in enumerate(plan.steps):
            jax_ready = step.operation in result.jax_functions
            trace = result.trace_checks.get(step.operation)
            trace_ok = bool(trace.traceable) if trace is not None else None
            spu_cap = (
                check_operation_capability(step.operation, capability).to_dict()
                if capability is not None
                else {}
            )
            spu_run = result.spu_runs.get(step.operation)
            key = run_keys.get(index)
            psi_run = result.psi_runs.get(key) if key else None

            status = _status_word(
                jax_ready=jax_ready,
                trace_ok=trace_ok,
                spu_supported=spu_cap.get("supported"),
                spu_run=spu_run,
                psi_run=psi_run,
                backend_direct=step.operation in skipped_ops,
                plaintext_local=_is_plaintext_local(step.operation),
            )

            rows.append(
                {
                    "operation": step.operation,
                    "representation": step.representation,
                    "backend": step.backend,
                    "protocol": step.protocol,
                    "protocol_params": dict(step.protocol_params),
                    "status": status,
                    "security_level": step.security_level,
                    "jax_ready": jax_ready,
                    "traceable": trace_ok,
                    "spu_status": spu_cap.get("status", "unknown"),
                    "psi_status": psi_run.status if psi_run is not None else None,
                    "psi_run_key": key,
                    "result_semantics": (
                        psi_run.result_semantics if psi_run is not None else None
                    ),
                    "result_policy": (
                        psi_run.result_policy.get("policy")
                        if psi_run is not None and psi_run.result_policy
                        else None
                    ),
                    "runtime_params": (
                        dict(psi_run.protocol_params) if psi_run is not None else None
                    ),
                    "reveals": psi_run.reveals if psi_run is not None else None,
                    "subset": (
                        psi_run.subset.to_dict()
                        if psi_run is not None and psi_run.subset is not None
                        else None
                    ),
                    "estimated_cost": dict(step.estimated_cost),
                    "notes": step.notes,
                }
            )

        result.operator_status = rows


def _load_layout_manifest(value: Any, *, name: str) -> Mapping[str, Any]:
    """布局清单：直接给 Mapping，或给 JSON 文件路径。非法形态直接报错。"""

    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (str, os.PathLike)):
        path = os.fspath(value)
        with open(path, encoding="utf-8-sig") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"布局清单 JSON 解析失败：{path}: {exc}"
                ) from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"布局清单必须是 JSON 对象：{path}")
        return dict(payload)
    raise TypeError(
        f"不支持的布局清单形态 {type(value).__name__}（名字 {name}）："
        "期望 Mapping 或 JSON 文件路径"
    )


def _is_plaintext_local(op: str) -> bool:
    """该算子是否只在本方明文执行（判据取自 planner 注册表，不另抄名单）。"""

    from planner.registry import get_rule

    rule = get_rule(op)
    return rule is not None and rule.primary_backend == "Plaintext"


@dataclass(frozen=True)
class _RuntimeValue:
    """PSI 执行装配用的一个实参值：码集合 + 来源标注 + 参与方标签（审计用）。"""

    codes: tuple[int, ...]
    origin: str
    layout: Mapping[str, Any] | None = None
    #: 该输入绑定的参与方（PartyInput.party_id）；未标注时为 None
    party_id: str | None = None


def _psi_run_keys(plan: PrivacyPlan) -> dict[int, str]:
    """给每个 PSI 步骤分配结果键（步索引 → 键）。

    首次出现用算子名（历史行为，测试与文档都依赖）；同一算子的后续步骤用
    output_name 区分——键冲突会静默覆盖前一次执行结果，宁可长一点。
    """

    seen: dict[str, int] = {}
    keys: dict[int, str] = {}
    for index, step in enumerate(plan.steps):
        if step.operation not in PSI_OPS:
            continue
        count = seen.get(step.operation, 0)
        seen[step.operation] = count + 1
        if count == 0:
            keys[index] = step.operation
        else:
            label = step.output_name or f"#{count + 1}"
            keys[index] = f"{step.operation}:{label}"
    return keys


def _check_config_closure(
    step: PlannedStep, config: PsiRuntimeConfig, run: PsiRunResult
) -> list[str]:
    """核对"计划里的协议参数"与"运行实际注入的"是否逐字段一致（验收 A）。

    单一配置源链：PlannedStep.protocol_params → PsiRuntimeConfig → Runtime。
    这里做逐字段比对，任何一环漂移都必须显式失败而不是静默容忍。
    """

    problems: list[str] = []
    actual = dict(run.runtime_config or {})
    if actual != config.to_dict():
        problems.append(
            f"算子 {step.operation}：Runtime 记录配置 {actual!r} 与传入配置 "
            f"{config.to_dict()!r} 不一致"
        )
    split_keys = ("receiver_rank", "broadcast_result")
    planned_params = {
        key: value
        for key, value in step.protocol_params.items()
        if key not in split_keys
    }
    if dict(config.protocol_params) != planned_params:
        problems.append(
            f"算子 {step.operation}：计划参数 {planned_params!r} 与配置参数 "
            f"{dict(config.protocol_params)!r} 不一致"
        )
    if run.protocol != config.protocol:
        problems.append(
            f"算子 {step.operation}：运行协议 {run.protocol!r} 与配置协议 "
            f"{config.protocol!r} 不一致"
        )
    return problems


def curve_suffix(protocol: str, curve: str | None) -> str:
    """把"协议 + 曲线"渲染成一句可读后缀。

    三类关系说三种话（见 `backends.psi_backend.PSI_CURVE_RELATION`）：
    注入曲线 / 不读曲线 / 协议自带默认曲线但本项目未传入。
    最后一种刻意**不写**"不生效"——那只说明本项目没传，不说明协议内部没读。
    """

    if curve:
        return f" / {curve}"
    if psi_curve_relation(protocol) == "implicit":
        return "（协议自带默认曲线，本项目未传入）"
    return "（不读曲线）"


def _status_word(
    *,
    jax_ready: bool,
    trace_ok: bool | None,
    spu_supported: Any,
    spu_run: SpuRunResult | None,
    psi_run: PsiRunResult | None = None,
    backend_direct: bool,
    plaintext_local: bool = False,
) -> str:
    """给一个算子判定最终状态词。

    优先报告"实际发生了什么"，而不是"计划要做什么"。

    PSI 与 SPU 是两条独立执行路径，二者的"已验证"不可互相顶替：
    - `verified`            : 该算子声明的主后端确实跑出过结果并与明文一致
                              （只对**精确**协议成立）；
    - `executed-noisy`      : 真跑了，但所用协议结果带噪（差分隐私 PSI），
                              按定义就不能拿"与明文一致"当验收标准；
    - `subset-plaintext`    : PSI 段真实执行过，但 `Contains` 的**子集判定**
                              落在明文上（显式选择或 MPC 不可用的退路）——
                              这一步没有密态保护，不能用 `verified` 概括；
    - `backend-direct`      : 有直连后端（无 JAX 路径），但本次未真正执行；
    - `plaintext-local`     : 该算子在本方明文完成（物化），本就不该有密态执行；
    - `planned`             : 仅规划，未执行。
    """

    if plaintext_local:
        # "没有密态实现"在这里不是缺口，是这个算子的定义：
        # 层集合在明文展开，密态边界落在消费它的 PSI 交集上。
        # 用 backend-direct 会让人以为"有个密态后端没跑"，用 planned 则像未完成。
        return "plaintext-local"

    if psi_run is not None:
        if psi_run.status == "ok":
            # 带噪协议连"与明文一致"都不成立，这一档优先于下面那档。
            if not protocol_is_exact(psi_run.protocol):
                return "executed-noisy"
            # PSI 段真跑了，但子集判定落在明文/退路上：`verified` 会被读成
            # "这一步也是密态完成的"，所以换一个词。
            subset = psi_run.subset
            if subset is not None and not subset.is_mpc:
                return "subset-plaintext"
            return "verified"
        if psi_run.status == "error":
            return "error"
        # unavailable → 环境不具备；empty-input → 未启动协议（按集合论直接给值）。
        # 两者都不满足"真实执行"，因此不进入 verified。
        return "backend-direct" if backend_direct else "planned"

    if spu_run is not None:
        if spu_run.ok:
            if spu_run.within_tolerance is False:
                return "tolerance-exceeded"
            return "verified"
        if spu_run.status == "unavailable":
            return "jax-verified" if jax_ready and trace_ok else "planned"
        return "error"
    if jax_ready:
        if trace_ok is False:
            return "not-traceable"
        return "jax-verified" if trace_ok else "jax-generated"
    if backend_direct:
        return "backend-direct"
    if spu_supported is True:
        return "backend-direct"
    return "planned"


def _plain_reference(op: str, args: Sequence[Any]) -> Any:
    """构造明文参考值；参数不符时返回 None（不阻塞模拟）。"""

    try:
        return run_plain(op, *args).value
    except Exception:
        return None


# --------------------------------------------------------------------------
# 便捷入口
# --------------------------------------------------------------------------

_COMPILER_KEYS = {
    "protocol",
    "field",
    "world_size",
    "tolerance",
    "run_simulation",
    "capability_report",
    "psi_subset",
    "psi_protocol",
    "psi_curve",
    "psi_rr22_low_comm_mode",
    "psi_protocol_params",
    "inputs",
    "input_layouts",
    "psi_capability_report",
    "sensitivities",
    "type_hints",
}

#: 不属于编译器、而是转发给解析阶段的参数
_FORWARDED_KEYS = {"entry"}


def _split_kwargs(kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    """拆分关键字参数；未知参数直接报错，绝不静默丢弃。

    静默丢弃的代价是真实的：调用方写 sensitivities={...} 时以为声明生效，
    实际被忽略，密态判定退回默认值——错误不会浮出来，只会算错。
    """

    unknown = sorted(set(kwargs) - _COMPILER_KEYS - _FORWARDED_KEYS)
    if unknown:
        raise TypeError(
            f"compile_source()/compile_file() 收到未知参数 {unknown}；"
            f"可用：{sorted(_COMPILER_KEYS | _FORWARDED_KEYS)}"
        )
    compiler_kwargs = {k: v for k, v in kwargs.items() if k in _COMPILER_KEYS}
    return compiler_kwargs, kwargs.get("entry")


def compile_source(source: str, filename: str = "<source>", **kwargs: Any) -> CompileResult:
    compiler_kwargs, entry = _split_kwargs(kwargs)
    return Compiler(**compiler_kwargs).compile_source(source, filename, entry=entry)


def compile_file(path: str, **kwargs: Any) -> CompileResult:
    compiler_kwargs, entry = _split_kwargs(kwargs)
    return Compiler(**compiler_kwargs).compile_file(path, entry=entry)


def operator_status_table(result: CompileResult) -> str:
    """渲染最终状态表：Operation / Representation / Backend / Status。"""

    rows = [
        [row["operation"], row["representation"], row["backend"], row["status"]]
        for row in result.operator_status
    ]
    return render_table(rows, ["Operation", "Representation", "Backend", "Status"])
