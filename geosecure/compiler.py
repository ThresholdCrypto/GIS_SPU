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
from backends.psi_backend.result_policy import REVEAL_COUNT
from backends.psi_backend.subset_mpc import SUBSET_MODE_MPC
from backends.psi_ca_backend import (
    PSI_CA_BACKEND,
    PSI_CA_PROTOCOL,
    PSI_CA_STRUCTURE,
    PSI_CA_SUPPORTED_OPS,
    PSI_CA_SUPPORTED_POLICIES,
    PsiCaCapabilityReport,
    check_psi_ca_capabilities,
    run_psi_cardinality,
)
from backends.psi_sum_backend import (
    PSI_SUM_BACKEND,
    PSI_SUM_PROTOCOL,
    PSI_SUM_RESULT_POLICY,
    PSI_SUM_SUPPORTED_OPS,
    PSI_SUM_SUPPORTED_POLICIES,
    PSI_SUM_UPSTREAM,
    PSI_SUM_UPSTREAM_COMMIT,
    PsiSumCapabilityReport,
    check_psi_sum_capabilities,
    run_psi_intersection_sum,
)
from backends.protocol_validation import validate_protocol_request
from backends.spu_backend import (
    CapabilityReport,
    SpuRunResult,
    check_capabilities,
    check_operation_capability,
    normalize_field,
    normalize_protocol,
    run_spu_simulation,
)
from frontend import ParseResult, parse_source
from frontend.analyzer import _coerce_geotype, _coerce_sensitivity
from ir import GeoProgram, encode_grid_code, render_table
from planner import (
    MPC_RULE_DEFAULT_PROTOCOL,
    PLAN_TABLE_HEADERS,
    LayoutShape,
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
    #: 本次编译的 PSI 执行档：None = libpsi 求交；"psi-ca" = 只出计数的 PSI-CA
    psi_count: str | None = None
    #: PSI-CA 环境核查结论（PsiCaCapabilityReport.to_dict()）；未走该档时为 None
    psi_ca_capability: dict[str, Any] | None = None
    #: 本次编译的 PSI 求和档：None = 不走 PI-Sum；"pjc" = 交集内求和（PI-Sum）
    psi_sum: str | None = None
    #: PI-Sum 环境核查结论（PsiSumCapabilityReport.to_dict()）；未走该档时为 None
    psi_sum_capability: dict[str, Any] | None = None
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
                    "mpc_protocol": s.mpc_protocol,
                    "mpc_protocol_basis": s.mpc_protocol_basis,
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
            "psi_count": self.psi_count,
            "psi_ca_capability": self.psi_ca_capability,
            "psi_sum": self.psi_sum,
            "psi_sum_capability": self.psi_sum_capability,
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
        #: PSI 执行档：None = libpsi 两方求交（默认）；"psi-ca" = PSI-Cardinality
        #: 只出交集基数（OpenMined PSI，只承接 CellSetIntersect）
        psi_count: str | None = None,
        #: PSI 求和档：None = 不走 PI-Sum（默认）；"pjc" = private-join-and-compute
        #: 的交集内求和（同时出交集基数与关联值之和，只承接 CellSetIntersect）
        psi_sum: str | None = None,
        #: PI-Sum 的关联值：输入名 → {格网码: 非负整数}。PI-Sum 的 client 侧
        #: 必须带值；缺值输入在方案阶段显式拒绝，不按 0 补齐
        psi_sum_weights: Mapping[str, Any] | None = None,
        inputs: Mapping[str, Any] | None = None,
        input_layouts: Mapping[str, Any] | None = None,
        capability_report: CapabilityReport | None = None,
        psi_capability_report: PsiCapabilityReport | None = None,
        sensitivities: Mapping[str, Any] | None = None,
        type_hints: Mapping[str, Any] | None = None,
        #: 位平面布局（D3）的规模形状；None = 只做轴向决策、不预测条数
        layout_shape: LayoutShape | None = None,
    ) -> None:
        #: 编译器**显式**指定的 MPC 协议；None = 让规划器按实测代价选。
        #: 执行期真正使用的具体值见 `protocol_in_use`（规划完一次性定型）。
        self.protocol = protocol
        self._protocol_in_use: str | None = None
        self.layout_shape = layout_shape
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

        # MPC（SPU）协议请求的前置校验（Phase 3 统一 capability validation）：
        # 环宽必须合法；显式协议下 world_size 必须满足协议下限、环宽必须落在
        # 该协议已实测支持的档内。协议名本身不存在时**不在构造期抛错**——
        # 留给规划层输出带位置/建议/代价的诊断（CLI 契约：错误码 1 + 完整报告）。
        normalize_field(self.field)
        if self.world_size is not None and (
            isinstance(self.world_size, bool)
            or not isinstance(self.world_size, int)
            or self.world_size < 2
        ):
            raise ValueError(f"world_size 必须是 ≥2 的整数，实得 {self.world_size!r}")
        if self.protocol is not None:
            try:
                mpc_protocol_name = normalize_protocol(self.protocol)
            except ValueError:
                mpc_protocol_name = None
            if mpc_protocol_name is not None:
                mpc_check = validate_protocol_request(
                    family="MPC",
                    protocol=mpc_protocol_name,
                    field=self.field,
                    world_size=self.world_size,
                )
                if not mpc_check.ok:
                    raise ValueError(
                        "MPC 协议请求非法：" + "；".join(mpc_check.problems)
                    )
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
        # PSI 执行档：非法值在构造期显式失败（带可用清单），不静默回到默认档。
        self.psi_count = _normalize_psi_count(psi_count)
        self._psi_ca_capability: PsiCaCapabilityReport | None = None
        self.psi_sum = _normalize_psi_sum(psi_sum)
        self._psi_sum_capability: PsiSumCapabilityReport | None = None
        # 两条 PSI 路径不能同时启用：计数档只出基数、求和档出基数 + 和，
        # 混用会让"这一档到底泄漏了什么"说不清。
        if self.psi_count is not None and self.psi_sum is not None:
            raise ValueError(
                f"--psi-count {PSI_CA_BACKEND} 与 --psi-sum {PSI_SUM_BACKEND} "
                "是两条不同的 PSI 路径，不能同时启用："
                "计数档只出基数，求和档出基数 + 交集内关联值之和"
            )
        # PI-Sum 的关联值：形态非法在这里显式失败，不静默丢成"没有值"——
        # 那会把"配置写错"变成"和为 0"，看起来像跑通了。
        self.psi_sum_weights: dict[str, dict[int, int]] = {}
        for weights_name, weights in (psi_sum_weights or {}).items():
            if not isinstance(weights, Mapping):
                raise ValueError(
                    f"psi_sum_weights[{weights_name!r}] 必须是 Mapping"
                    f"（格网码 → 非负整数），收到 {type(weights).__name__}"
                )
            self.psi_sum_weights[str(weights_name)] = {
                int(code): int(value) for code, value in weights.items()
            }
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
        """方案的**代表** MPC 协议（具体值，不会为 None）。

        取值链：显式指定 > **方案里第一个** MPC 步骤选出的协议 > 登记默认值。
        仅用于“整个方案只用一个 MPC 协议”的摘要/展示场景；执行期逐步解析
        （见 `_step_protocol`），不把这个单值套到所有步骤上。
        """

        return (
            self._protocol_in_use
            or self.protocol
            or MPC_RULE_DEFAULT_PROTOCOL
        )

    def _step_protocol(self, step: PlannedStep) -> str:
        """单个 MPC 步骤的执行期协议：显式指定 > 方案实测选择 > 登记默认值。

        按**步骤**解析，而不是取方案里第一个 MPC 步骤的协议：同一个程序里两个
        MPC 步骤各自选中不同协议时，Runtime 必须各跑各的——把单值套到所有步骤
        上等于让后一步静默换协议（验收口径：Runtime 不允许静默更换协议）。

        归一化“能认就认”：认不出的名字原样返回，交给下游 fail-fast 抛 ValueError，
        不在这里改变既有的报错契约。
        """

        raw = self.protocol or step.mpc_protocol or MPC_RULE_DEFAULT_PROTOCOL
        try:
            return normalize_protocol(raw)
        except ValueError:
            return str(raw)

    def compile_source(
        self, source: str, filename: str = "<source>", *, entry: str | None = None
    ) -> CompileResult:
        result = CompileResult(source_file=None if filename == "<source>" else filename)
        result.psi_protocol = self.psi_protocol
        result.psi_curve = self.psi_curve
        result.psi_protocol_params = dict(self.psi_protocol_params)
        result.psi_count = self.psi_count
        result.psi_sum = self.psi_sum
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
            # Phase 3：环宽 / 参与方数量也进规划层——自动选择出来的协议同样要过
            # world_size 校验（选择发生在规划层内部，构造期无法替它判断）。
            mpc_field=self.field,
            mpc_world_size=self.world_size,
            layout_shape=self.layout_shape,
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

        # 方案级“代表协议”：显式指定 > 方案里第一个 MPC 步骤的选择 > 登记默认值。
        # 仅供 `protocol_in_use` 这个摘要属性使用——执行期**不**按这个单值跑：同一
        # 程序里多个 MPC 步骤可以各选各的协议，逐步解析见 `_step_protocol`（否则
        # 第 2 步会被静默换成第 1 步的协议，违反“Runtime 不允许静默更换协议”）。
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

        # 编译期规模提示：调用方给了 --layout-shape 就顺带用它选电路，
        # 避免"布局按候选数 N 算、电路因为拿不到规模退回 pairwise"（P1）。
        generation = generate_for_plan(
            result.plan,
            size_hint=(
                self.layout_shape.size_hint if self.layout_shape is not None else None
            ),
        )
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
        mpc_closure_problems: list[str] = []
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

            step_protocol = self._step_protocol(step)
            try:
                run = run_spu_simulation(
                    fn,
                    [np.asarray(x) for x in examples],
                    protocol=step_protocol,
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
                    protocol=step_protocol,
                    field=str(self.field),
                    world_size=self.world_size or 2,
                    error=str(exc),
                )
            mpc_closure_problems.extend(_check_mpc_closure(step, step_protocol, run))
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

        if mpc_closure_problems:
            result.stages.append(
                StageResult(
                    "spu_simulation",
                    "error",
                    result.spu_runs,
                    "MPC 执行配置闭环核对失败：" + "；".join(mpc_closure_problems),
                )
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
            message = "方案中没有走 PSI 的算子"
            if self.psi_count == PSI_CA_BACKEND:
                message += "（--psi-count psi-ca 未生效：本档只承接 CellSetIntersect）"
            elif self.psi_sum == PSI_SUM_BACKEND:
                message += "（--psi-sum pjc 未生效：本档只承接 CellSetIntersect）"
            result.stages.append(
                StageResult(
                    "psi_capability",
                    "skipped",
                    message=message,
                )
            )
            return

        # PSI-CA 计数档是与 libpsi 求交并列的第二条 PSI 路径：能力核查对象、
        # 可承接的算子与泄漏承诺都不同，这里整段改道，不混用两套结论。
        if self.psi_count == PSI_CA_BACKEND:
            self._stage_psi_ca_capability(result, psi_steps)
            return
        # PI-Sum 求和档同属整段改道：能力核查对象（上游两个可执行文件）、
        # 可承接的算子与泄漏承诺都不同，不混用两套结论。
        if self.psi_sum == PSI_SUM_BACKEND:
            self._stage_psi_sum_capability(result, psi_steps)
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

    def _psi_ca_rejections(self, psi_steps: Sequence[PlannedStep]) -> list[dict[str, Any]]:
        """PSI-CA 计数档的编译期拒绝清单（空 = 全部可承接）。

        两条契约与运行环境无关，离线也拦得住：
        - 只承接 ``CellSetIntersect``（不做布尔判定、不产出集合）；
        - 计数不是集合：上一步的计数输出不能作为下游 PSI 步骤的集合输入。
        每条拒绝都带：错误位置（算子）、原因、建议替代、预计隐私计算代价。
        """

        rejections: list[dict[str, Any]] = []
        produced = {step.output_name for step in psi_steps if step.output_name}
        for index, step in enumerate(psi_steps, start=1):
            cost = _cost_hint(step.estimated_cost)
            if step.operation not in PSI_CA_SUPPORTED_OPS:
                rejections.append(
                    {
                        "operation": step.operation,
                        "step_index": index,
                        "kind": "unsupported-op",
                        "reason": (
                            f"算子 {step.operation} 不在 PSI-CA 承接清单 "
                            f"{tuple(PSI_CA_SUPPORTED_OPS)}；本档只计算并返回交集基数，"
                            "不做布尔判定、不产出集合本体"
                        ),
                        "suggestion": (
                            "改用 CellSetIntersect 取得基数"
                            f"（配套策略 {', '.join(PSI_CA_SUPPORTED_POLICIES)}），"
                            "或去掉 --psi-count psi-ca 走 libpsi 求交路径"
                            "（Intersects / Contains 由 libpsi 求交后解释）"
                        ),
                        "estimated_cost": cost,
                    }
                )
            for name in step.inputs:
                if name in produced and name not in self._resolved_inputs:
                    rejections.append(
                        {
                            "operation": step.operation,
                            "step_index": index,
                            "kind": "count-not-set",
                            "reason": (
                                f"输入 {name} 是上一步 PSI 步骤的输出；PSI-CA 计数档"
                                "不产出交集本体，计数无法作为下游集合输入"
                            ),
                            "suggestion": (
                                "拆分为各自独立的 CellSetIntersect（两两求交），"
                                "或去掉 --psi-count psi-ca 走 libpsi 求交路径"
                                "（求交链式数据流由它承载）"
                            ),
                            "estimated_cost": cost,
                        }
                    )
        return rejections

    def _stage_psi_ca_capability(
        self, result: CompileResult, psi_steps: Sequence[PlannedStep]
    ) -> None:
        """PSI-CA 计数档的能力核查：编译契约（能不能接）与环境（这里能不能跑）分开。"""

        rejections = self._psi_ca_rejections(psi_steps)
        if rejections:
            # 契约违例与运行环境无关：这里必须是 error，不能降级成 warning，
            # 否则"计数档跑不了这个程序"会被读成"环境问题、装好就能跑"。
            message = "；".join(
                f"{item['operation']}（第 {item['step_index']} 步）：{item['reason']}；"
                f"建议：{item['suggestion']}；预计隐私计算代价：{item['estimated_cost']}"
                for item in rejections
            )
            result.stages.append(
                StageResult(
                    "psi_capability",
                    "error",
                    {"rejections": rejections},
                    f"PSI-CA（--psi-count psi-ca）无法编译本程序（{len(rejections)} 处）：{message}",
                )
            )
            return

        if self._psi_ca_capability is None:
            self._psi_ca_capability = check_psi_ca_capabilities()
        report = self._psi_ca_capability
        report_dict = report.to_dict()
        result.psi_ca_capability = report_dict

        ops = ", ".join(step.operation for step in psi_steps)
        detail = {
            "report": report_dict,
            "ops": [step.operation for step in psi_steps],
            "structure": PSI_CA_STRUCTURE,
            "role_mapping": "client=左侧输入（获得计数），server=右侧输入",
        }
        selection = (
            f"选用 PSI-CA（{report_dict['distribution']}/{PSI_CA_STRUCTURE}，"
            "只出交集基数）"
        )
        tail = (
            f"当前环境可执行（openmined-psi {report.version}）"
            if report.runnable
            else f"{len(report.blockers)} 项阻断，计数结果将留空"
        )
        result.stages.append(
            StageResult(
                "psi_capability",
                "ok" if report.runnable else "warning",
                detail,
                f"{len(psi_steps)} 个算子需 PSI（{ops}）；{selection}；{tail}；"
                "协议只出计数，交集本体不交给任一方",
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

        # PSI-CA 计数档整段改道：libpsi 的协议/配置/子集判定都不参与。
        if self.psi_count == PSI_CA_BACKEND:
            self._stage_psi_ca_simulation(result)
            return
        # PI-Sum 求和档同属整段改道：libpsi 的协议/配置/子集判定都不参与。
        if self.psi_sum == PSI_SUM_BACKEND:
            self._stage_psi_sum_simulation(result)
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
                # 共用同一套开关，不另开一个配置面。协议同样按**本步骤**解析：
                # Contains 步骤的协议未必等于方案里第一个 MPC 步骤的协议。
                mpc_protocol=self._step_protocol(step),
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

    def _stage_psi_ca_simulation(self, result: CompileResult) -> None:
        """PSI-CA 计数档的真实执行：只跑 CellSetIntersect，只出交集基数。

        与 libpsi 路径（``_stage_psi_simulation``）刻意不合并：那边装配
        ``PsiRuntimeConfig``、做配置闭环、解释子集判定；这边的运行时是
        OpenMined PSI 的进程内句柄，没有可对齐的配置对象。输入装配、
        样例兜底与参与方披露沿用同一套 ``_resolve_psi_inputs`` 口径。
        """

        plan = result.plan
        if plan is None or plan.has_errors:
            result.stages.append(
                StageResult("psi_simulation", "skipped", message="隐私方案有错误，跳过模拟")
            )
            return

        steps = [step for step in plan.steps if step.operation in PSI_OPS]
        if not steps:
            result.stages.append(
                StageResult(
                    "psi_simulation",
                    "skipped",
                    message="方案中没有走 PSI 的算子（--psi-count psi-ca 未生效）",
                )
            )
            return

        if self._psi_ca_rejections(steps):
            result.stages.append(
                StageResult(
                    "psi_simulation",
                    "skipped",
                    message="PSI-CA 能力核查未通过（见上一阶段结论），未执行",
                )
            )
            return

        if self._psi_ca_capability is None:
            self._psi_ca_capability = check_psi_ca_capabilities()
        report = self._psi_ca_capability

        runtime_values: dict[str, _RuntimeValue] = {
            name: _RuntimeValue(
                codes=value.codes,
                origin=f"输入绑定（{value.source or '用户数据'}）",
                layout=value.layout,
                party_id=value.party_id,
            )
            for name, value in self._resolved_inputs.items()
        }

        run_keys = _psi_run_keys(plan)
        produced_names: set[str] = set()

        for index, step in enumerate(plan.steps):
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
                    protocol=PSI_CA_PROTOCOL,
                    curve=None,
                    world_size=PSI_RUNTIME_WORLD_SIZE,
                    receiver_rank=0,
                    error=(
                        f"输入无法装配：{list(step.inputs)} 既不在绑定输入 "
                        f"{sorted(self._resolved_inputs)}，也不是上一步输出；"
                        "请用 inputs={...} 绑定数据或确认样例兜底存在"
                    ),
                    notes=("未执行 PSI-CA：输入装配失败在进入运行时之前拦下",),
                )
                continue

            left, right = values[0], values[1]
            plain_inputs = (list(left.codes), list(right.codes))
            reference = _plain_reference(op, plain_inputs)

            def reference_fn(left_codes: Any, right_codes: Any, _op: str = op) -> Any:
                return _plain_reference(_op, (left_codes, right_codes))

            binding = tuple(
                {
                    "role": role,
                    "name": name,
                    "party_id": item.party_id,
                }
                for role, name, item in (
                    ("left", step.inputs[0] if len(step.inputs) > 0 else None, left),
                    ("right", step.inputs[1] if len(step.inputs) > 1 else None, right),
                )
            )
            run = run_psi_cardinality(
                left.codes,
                right.codes,
                op=op,
                result_policy=REVEAL_COUNT,
                left_layout=left.layout,
                right_layout=right.layout,
                party_binding=binding,
                reference_fn=reference_fn if reference is not None else None,
                report=report,
            )
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
            result.psi_runs[key] = run

        # ---------------- 阶段结论 ----------------
        if any(run.status == "unavailable" for run in result.psi_runs.values()):
            status = "warning"
            message = "当前环境不可运行 PSI-CA，计数结果栏位留空（见能力核查）"
        elif all(run.ok or run.status == "empty-input" for run in result.psi_runs.values()):
            status = "ok"
            message = (
                f"{len(result.psi_runs)} 个算子经 PSI-CA 真实执行"
                "（只出交集基数，与明文计数一致）"
            )
        else:
            status = "error"
            message = f"{len(result.psi_runs)} 个算子 PSI-CA 执行失败"

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

        result.stages.append(StageResult("psi_simulation", status, result.psi_runs, message))

    def _psi_sum_rejections(
        self, psi_steps: Sequence[PlannedStep]
    ) -> list[dict[str, Any]]:
        """PI-Sum 求和档的编译期拒绝清单（空 = 全部可承接）。

        三条契约与运行环境无关，离线也拦得住：
        - 只承接 ``CellSetIntersect``（不做布尔判定、不产出集合）；
        - 产出是**两个数**（基数 + 和），不是集合：它不能作为下游 PSI 的集合输入；
        - 左侧输入（client = 值持有方）必须有**覆盖完整**的关联值——上游
          client CSV 每行是「标识符, 非负整数」两列，没有值就跑不了这一档。
        每条拒绝都带：错误位置（算子）、原因、建议替代、预计隐私计算代价。
        """

        rejections: list[dict[str, Any]] = []
        produced = {step.output_name for step in psi_steps if step.output_name}
        for index, step in enumerate(psi_steps, start=1):
            cost = _cost_hint(step.estimated_cost)
            if step.operation not in PSI_SUM_SUPPORTED_OPS:
                rejections.append(
                    {
                        "operation": step.operation,
                        "step_index": index,
                        "kind": "unsupported-op",
                        "reason": (
                            f"算子 {step.operation} 不在 PI-Sum 承接清单 "
                            f"{tuple(PSI_SUM_SUPPORTED_OPS)}；本档只计算并返回"
                            "交集基数与交集内关联值之和，不做布尔判定、"
                            "不产出集合本体"
                        ),
                        "suggestion": (
                            "改用 CellSetIntersect 取「基数 + 和」"
                            f"（配套策略 {', '.join(PSI_SUM_SUPPORTED_POLICIES)}）；"
                            "只需基数请走 --psi-count psi-ca，"
                            "需要交集本体请走 libpsi 求交路径"
                        ),
                        "estimated_cost": cost,
                    }
                )
            for name in step.inputs:
                if name in produced and name not in self._resolved_inputs:
                    rejections.append(
                        {
                            "operation": step.operation,
                            "step_index": index,
                            "kind": "sum-not-set",
                            "reason": (
                                f"输入 {name} 是上一步 PSI 步骤的输出；"
                                "PI-Sum 只产出两个数（基数 + 和），"
                                "两个数都不是集合，无法作为下游集合输入"
                            ),
                            "suggestion": (
                                "拆分为各自独立的 CellSetIntersect（两两求交），"
                                "或去掉 --psi-sum pjc 走 libpsi 求交路径"
                                "（链式数据流由它承载）"
                            ),
                            "estimated_cost": cost,
                        }
                    )
            if step.operation not in PSI_SUM_SUPPORTED_OPS or not step.inputs:
                continue
            left_name = str(step.inputs[0])
            weights = self.psi_sum_weights.get(left_name)
            if not weights:
                rejections.append(
                    {
                        "operation": step.operation,
                        "step_index": index,
                        "kind": "missing-associated-values",
                        "reason": (
                            f"左侧输入 {left_name} 没有关联值：PI-Sum 的 client "
                            "侧 CSV 必须是「格网码, 非负整数」两列，"
                            "没有值就只能求交不能求和"
                        ),
                        "suggestion": (
                            "用 psi_sum_weights={'输入名': {格网码: 权重}} "
                            "声明关联值（CLI：--psi-sum-weights 输入名=PATH，"
                            f"本例输入名是 {left_name!r}）："
                            "本项目不按 0 补齐——补齐会把「数据没接上」"
                            "伪装成「和为 0」；只需基数请改用 --psi-count psi-ca"
                        ),
                        "estimated_cost": cost,
                    }
                )
                continue
            resolved = self._resolved_inputs.get(left_name)
            if resolved is not None:
                missing = [
                    int(code) for code in resolved.codes if int(code) not in weights
                ]
                if missing:
                    rejections.append(
                        {
                            "operation": step.operation,
                            "step_index": index,
                            "kind": "incomplete-associated-values",
                            "reason": (
                                f"输入 {left_name} 的关联值覆盖不全："
                                f"{len(missing)} 个格网码没有对应值"
                                f"（前几个：{missing[:5]}）"
                            ),
                            "suggestion": (
                                "补齐 psi_sum_weights 的覆盖（绑定的每个格网码"
                                "都要有值），或改用只出基数的 --psi-count psi-ca"
                            ),
                            "estimated_cost": cost,
                        }
                    )
        return rejections

    def _stage_psi_sum_capability(
        self, result: CompileResult, psi_steps: Sequence[PlannedStep]
    ) -> None:
        """PI-Sum 求和档的能力核查：编译契约（能不能接）与环境（这里能不能跑）分开。"""

        rejections = self._psi_sum_rejections(psi_steps)
        if rejections:
            # 契约违例与运行环境无关：这里必须是 error，不能降级成 warning，
            # 否则"求和档跑不了这个程序"会被读成"环境问题、装好就能跑"。
            message = "；".join(
                f"{item['operation']}（第 {item['step_index']} 步）：{item['reason']}；"
                f"建议：{item['suggestion']}；预计隐私计算代价：{item['estimated_cost']}"
                for item in rejections
            )
            result.stages.append(
                StageResult(
                    "psi_capability",
                    "error",
                    {"rejections": rejections},
                    f"PI-Sum（--psi-sum pjc）无法编译本程序"
                    f"（{len(rejections)} 处）：{message}",
                )
            )
            return

        if self._psi_sum_capability is None:
            self._psi_sum_capability = check_psi_sum_capabilities()
        report = self._psi_sum_capability
        report_dict = report.to_dict()
        result.psi_sum_capability = report_dict

        ops = ", ".join(step.operation for step in psi_steps)
        detail = {
            "report": report_dict,
            "ops": [step.operation for step in psi_steps],
            "role_mapping": "client=左侧输入（带关联值、获得结果），server=右侧输入",
            "weights_inputs": sorted(
                str(step.inputs[0]) for step in psi_steps if step.inputs
            ),
        }
        selection = (
            f"选用 PI-Sum（{PSI_SUM_UPSTREAM.rsplit('/', 1)[-1]}"
            f"@{PSI_SUM_UPSTREAM_COMMIT[:7]}，只出基数与交集内和）"
        )
        tail = (
            f"当前环境可执行（构建产物 {report.version}）"
            if report.runnable
            else f"{len(report.blockers)} 项阻断，求和结果将留空"
        )
        result.stages.append(
            StageResult(
                "psi_capability",
                "ok" if report.runnable else "warning",
                detail,
                f"{len(psi_steps)} 个算子需 PSI（{ops}）；{selection}；{tail}；"
                "协议只出两个数（基数 + 交集内关联值之和），交集本体不交给任一方",
            )
        )

    def _stage_psi_sum_simulation(self, result: CompileResult) -> None:
        """PI-Sum 求和档的真实执行：只跑 CellSetIntersect，只出「基数 + 和」。

        与 libpsi 路径（``_stage_psi_simulation``）刻意不合并：那边装配
        ``PsiRuntimeConfig``、做配置闭环、解释子集判定；这边的运行时是上游
        两个可执行文件 + 本机回环 gRPC，没有可对齐的配置对象。输入装配、
        样例兜底与参与方披露沿用同一套 ``_resolve_psi_inputs`` 口径。
        """

        plan = result.plan
        if plan is None or plan.has_errors:
            result.stages.append(
                StageResult("psi_simulation", "skipped", message="隐私方案有错误，跳过模拟")
            )
            return

        steps = [step for step in plan.steps if step.operation in PSI_OPS]
        if not steps:
            result.stages.append(
                StageResult(
                    "psi_simulation",
                    "skipped",
                    message="方案中没有走 PSI 的算子（--psi-sum pjc 未生效）",
                )
            )
            return

        if self._psi_sum_rejections(steps):
            result.stages.append(
                StageResult(
                    "psi_simulation",
                    "skipped",
                    message="PI-Sum 能力核查未通过（见上一阶段结论），未执行",
                )
            )
            return

        if self._psi_sum_capability is None:
            self._psi_sum_capability = check_psi_sum_capabilities()
        report = self._psi_sum_capability

        runtime_values: dict[str, _RuntimeValue] = {
            name: _RuntimeValue(
                codes=value.codes,
                origin=f"输入绑定（{value.source or '用户数据'}）",
                layout=value.layout,
                party_id=value.party_id,
            )
            for name, value in self._resolved_inputs.items()
        }

        run_keys = _psi_run_keys(plan)
        produced_names: set[str] = set()

        for index, step in enumerate(plan.steps):
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
                    protocol=PSI_SUM_PROTOCOL,
                    curve=None,
                    world_size=PSI_RUNTIME_WORLD_SIZE,
                    receiver_rank=0,
                    error=(
                        f"输入无法装配：{list(step.inputs)} 既不在绑定输入 "
                        f"{sorted(self._resolved_inputs)}，也不是上一步输出；"
                        "请用 inputs={...} 绑定数据或确认样例兜底存在"
                    ),
                    notes=("未执行 PI-Sum：输入装配失败在进入运行时之前拦下",),
                )
                continue

            left, right = values[0], values[1]
            weights = self.psi_sum_weights.get(str(step.inputs[0]))
            plain_inputs = (list(left.codes), list(right.codes))
            reference = _plain_reference(op, plain_inputs)

            def reference_fn(left_codes: Any, right_codes: Any, _op: str = op) -> Any:
                return _plain_reference(_op, (left_codes, right_codes))

            binding = tuple(
                {
                    "role": role,
                    "name": name,
                    "party_id": item.party_id,
                }
                for role, name, item in (
                    ("left", step.inputs[0] if len(step.inputs) > 0 else None, left),
                    ("right", step.inputs[1] if len(step.inputs) > 1 else None, right),
                )
            )
            run = run_psi_intersection_sum(
                left.codes,
                right.codes,
                left_values=weights,
                op=op,
                result_policy=PSI_SUM_RESULT_POLICY,
                left_layout=left.layout,
                right_layout=right.layout,
                party_binding=binding,
                reference_fn=reference_fn if reference is not None else None,
                report=report,
            )
            for name, value in zip(step.inputs, values):
                note = f"输入 {name} 来源：{value.origin}"
                if value.party_id:
                    note += f"；参与方：{value.party_id}"
                run.notes = run.notes + (note,)
            if weights:
                run.notes = run.notes + (
                    f"关联值：{len(weights)} 个码来自 psi_sum_weights"
                    f"[{step.inputs[0]!r}]（client 侧）",
                )
            if same_party_both_sides(left.party_id, right.party_id):
                run.notes = run.notes + (
                    f"两侧输入的 party_id 相同（{left.party_id}）：本步骤在语义上"
                    "不是跨方求交，请核对 PartyInput 绑定",
                )
            result.psi_runs[key] = run

        # ---------------- 阶段结论 ----------------
        if any(run.status == "unavailable" for run in result.psi_runs.values()):
            status = "warning"
            message = "当前环境不可运行 PI-Sum，求和结果栏位留空（见能力核查）"
        elif all(
            run.ok or run.status == "empty-input" for run in result.psi_runs.values()
        ):
            status = "ok"
            message = (
                f"{len(result.psi_runs)} 个算子经 PI-Sum 真实执行"
                "（只出交集基数与交集内关联值之和，与明文一致）"
            )
        else:
            status = "error"
            message = f"{len(result.psi_runs)} 个算子 PI-Sum 执行失败"

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


def _check_mpc_closure(
    step: PlannedStep, protocol: str, run: SpuRunResult
) -> list[str]:
    """核对“方案里的 MPC 协议”与“运行实际使用的协议”是否一致（Phase 4 验收）。

    PSI 侧的逐字段对拍在 `_check_config_closure`；MPC 侧唯一由**方案逐步决定**的
    量就是协议本身（环宽 / 参与方数量同源于编译器配置，不由某个步骤选择），因此
    这里只比协议：方案写 A、实际跑 B 必须显式失败——Runtime 不允许静默更换协议。
    """

    if run.protocol != protocol:
        return [
            f"算子 {step.operation}：Runtime 实际使用协议 {run.protocol!r} 与方案协议 "
            f"{protocol!r} 不一致（Runtime 不允许静默更换协议）"
        ]
    return []


def _normalize_psi_count(value: str | None) -> str | None:
    """解析 PSI 执行档：None = libpsi 求交（默认），只接受 "psi-ca"。"""

    if value is None:
        return None
    text = str(value).strip().lower()
    if text != PSI_CA_BACKEND:
        raise ValueError(
            f"未知 PSI 执行档 psi_count={value!r}；可用：None（libpsi 两方求交）、"
            f"{PSI_CA_BACKEND!r}（PSI-Cardinality，只出交集基数）"
        )
    return PSI_CA_BACKEND


def _normalize_psi_sum(value: str | None) -> str | None:
    """解析 PSI 求和档：None = 不走 PI-Sum（默认），只接受 "pjc"。"""

    if value is None:
        return None
    text = str(value).strip().lower()
    if text != PSI_SUM_BACKEND:
        raise ValueError(
            f"未知 PSI 求和档 psi_sum={value!r}；可用：None（不走 PI-Sum）、"
            f"{PSI_SUM_BACKEND!r}（private-join-and-compute，交集内求和）"
        )
    return PSI_SUM_BACKEND


def _cost_hint(cost: Mapping[str, Any]) -> str:
    """方案代价的可读摘要（b/d/R）；缺档时如实说没有，不给臆造值。"""

    parts = [
        f"{key}={cost[key]}" for key in ("b", "d", "R") if cost.get(key) is not None
    ]
    return "，".join(parts) if parts else "（方案未登记 b/d/R 代价档）"


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
    - `count-only`          : PSI-CA 计数档真实执行过且与明文计数一致——
                              产出只有基数、没有交集本体，不与 `verified` 混用；
    - `count-and-sum`       : PI-Sum 求和档真实执行过且与明文一致——产出是
                              「基数 + 交集内关联值之和」两个数，同样不与
                              `verified` 混用；
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
            # PSI-CA 计数档：真实执行过且与明文计数一致，但产出只有基数，
            # 与"求交验证"不是一回事；协议名也不在 libpsi 协议表里
            # （protocol_is_exact 对未知名会抛错），必须先分道。
            # PI-Sum 求和档：真实执行过且与明文一致，但产出是「基数 + 交集内
            # 关联值之和」两个数，与"求交验证"不是一回事；协议名同样不在
            # libpsi 协议表里（protocol_is_exact 对未知名会抛错），必须先分道。
            if psi_run.protocol == PSI_SUM_PROTOCOL:
                return "count-and-sum"
            if psi_run.protocol == PSI_CA_PROTOCOL:
                return "count-only"
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
    "psi_count",
    "psi_sum",
    "psi_sum_weights",
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
