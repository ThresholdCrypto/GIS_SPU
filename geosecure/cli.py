"""geosecure CLI：`geo-secure build <source.py>`

输出六个阶段的进展，以及最终状态表（Operation / Representation / Backend / Status）。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Mapping, Sequence

from backends.psi_backend import (
    PSI_DEFAULT_CURVE,
    PSI_DEFAULT_PROTOCOL,
    SUBSET_MODE_MPC,
    SUBSET_MODE_PLAINTEXT,
    protocol_is_exact,
    psi_curve_relation,
    resolve_psi_protocol,
    runnable_protocols_hint,
)
from backends.psi_ca_backend import PSI_CA_BACKEND
from backends.psi_sum_backend import PSI_SUM_BACKEND
from backends.spu_backend import SPU_PROTOCOLS
from geosecure.compiler import Compiler, CompileResult, curve_suffix, operator_status_table
from ir import (
    GRID_CODE_LAYOUT,
    MAX_ENCODABLE_LEVEL,
    render_operations_table,
    render_program,
    render_relations,
    render_table,
)
from planner import (
    PLAN_TABLE_HEADERS,
    SELECTION_BASIS_DECLARED_DEFAULT,
    SELECTION_BASIS_EXPLICIT,
    SELECTION_BASIS_MEASURED,
    layout_shape_from_mapping,
    plan_table_rows,
)

BANNER = "geo-secure — 地理信息行业低门槛隐私计算编译器"

DIVIDER = "-" * 72


def _emit(text: str = "") -> None:
    """输出一行。

    Windows 终端默认 GBK（cp936），实测 '▸'(U+25B8) 与 '⊆'(U+2286) 无法编码——
    直接 print 会抛 UnicodeEncodeError 并中断整条编译输出。这里按 stdout 的
    实际编码先做一次可编码性降级，只影响显示，不影响任何编译语义。
    """

    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except UnicodeEncodeError:
        text = text.encode(encoding, "replace").decode(encoding, "replace")
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(encoding, "replace").decode(encoding, "replace"))


def _stage_line(index: int, total: int, result: CompileResult, name: str) -> None:
    stage = result.stage(name)
    if stage is None:
        return
    _emit(f"[{index}/{total}] {stage.title:<22} {stage.icon} {stage.status.upper():<8} {stage.message}")


def _layout_note(cost: Mapping[str, Any]) -> str:
    """位平面布局的可读后缀（D3）；没预测条数就如实说"不预测"。"""

    if cost.get("N_ct_layout") is None:
        return "（未给布局形状，不预测条数）"
    return (
        f"（逐点 {cost.get('N_ct_naive')} 条 → {cost.get('N_ct_layout')} 条，"
        f"{float(cost.get('layout_reduction') or 1):.1f}×）"
    )


def _mpc_basis_note(basis: str | None) -> str:
    """MPC 协议来源的可读后缀：选了什么都不说清楚，等于没有留痕。"""

    return {
        SELECTION_BASIS_MEASURED: "（按实测代价自动选择）",
        SELECTION_BASIS_DECLARED_DEFAULT: "（无实测依据，用登记默认值）",
        SELECTION_BASIS_EXPLICIT: "（编译器显式指定）",
    }.get(basis or "", "")


def print_result(result: CompileResult, *, verbose: bool = False, as_json: bool = False) -> None:
    if as_json:
        _emit(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    _emit(BANNER)
    _emit(DIVIDER)

    # ---------------- 流水线阶段 ----------------
    stages = (
        "parsing",
        "ir",
        "planning",
        "jax",
        "spu_capability",
        "spu_simulation",
        "psi_capability",
        "psi_simulation",
    )
    _emit("Pipeline")
    total = len(stages)
    for index, name in enumerate(stages, start=1):
        _stage_line(index, total, result, name)
    _emit()

    # ---------------- IR ----------------
    _emit(DIVIDER)
    _emit("IR generation")
    if result.program is not None and result.program.operations:
        _emit(render_operations_table(result.program.operations))
        if result.program.relations:
            _emit()
            _emit("关系（GeoRelation）：")
            _emit(render_relations(result.program.relations))
            for relation in result.program.relations:
                _emit(f"  {relation.subject} | {relation.predicate} | {relation.object}")
    else:
        _emit("  （无算子）")
    _emit()

    # ---------------- 隐私方案 ----------------
    _emit(DIVIDER)
    _emit("Privacy planning")
    if result.plan is not None and result.plan.steps:
        _emit(render_table(plan_table_rows(result.plan), PLAN_TABLE_HEADERS))
        summary = result.plan.summary()
        _emit()
        _emit(f"  后端: {summary['backends']}   表征: {summary['representations']}   安全级别: {summary['security_levels']}")
        picks = [
            f"{step.operation} → {step.mpc_protocol}{_mpc_basis_note(step.mpc_protocol_basis)}"
            for step in result.plan.steps
            if step.mpc_protocol
        ]
        if picks:
            _emit(f"  MPC 协议: {'; '.join(picks)}")
        layouts = [
            f"{step.operation} → {step.estimated_cost['layout']}"
            f"{_layout_note(step.estimated_cost)}"
            for step in result.plan.steps
            if step.estimated_cost.get("layout")
        ]
        if layouts:
            _emit(f"  位平面布局: {'; '.join(layouts)}")
    else:
        _emit("  （无方案）")
    _emit()

    # ---------------- JAX 代码 ----------------
    _emit(DIVIDER)
    _emit("JAX generation")
    if result.jax_generation is not None:
        for function in result.jax_generation.functions:
            check = result.trace_checks.get(function.op)
            trace_note = ""
            if check is not None:
                if check.traceable:
                    trace_note = f" [jax.jit 可追踪, HLO {check.hlo_bytes or '-'} B]"
                else:
                    trace_note = f" [不可追踪: {check.error_type}]"
            _emit(f"  + {function.name}({', '.join(function.inputs)}){trace_note}")
        for item in result.jax_generation.skipped:
            _emit(f"  - {item['operation']} [{item['backend']}]：不生成 JAX（{item['reason']}）")
        if verbose:
            _emit()
            for function in result.jax_generation.functions:
                _emit(f"--- {function.name} ---")
                _emit(function.source)
                _emit()
    else:
        _emit("  （未生成）")
    _emit()

    # ---------------- SPU 能力 ----------------
    _emit(DIVIDER)
    _emit("SPU capability check")
    if result.capability is not None:
        report = result.capability
        _emit(f"  状态      : {report.status}")
        _emit(f"  平台      : {report.platform} / {report.machine} / Python {report.python_version}")
        _emit(f"  jax       : {report.jax.version or '未安装'}"
              + (f"（私有接口缺失: {', '.join(report.jax.missing_deps)}）" if report.jax.missing_deps else ""))
        _emit(f"  spu       : {report.spu.version or '未安装'}")
        if report.blockers:
            _emit("  阻断项    :")
            for blocker in report.blockers:
                _emit(f"    - {blocker}")
        _emit(f"  可运行    : {'是' if report.runnable else '否'}")
    else:
        _emit("  （未执行）")
    _emit()

    # ---------------- PSI 能力 ----------------
    if result.psi_capability is not None or result.stage("psi_capability") is not None:
        stage = result.stage("psi_capability")
        if stage is not None and stage.status != "skipped":
            _emit(DIVIDER)
            _emit("PSI capability check")
            report = result.psi_capability
            if result.psi_count == PSI_CA_BACKEND:
                ca_report = result.psi_ca_capability
                if ca_report is None:
                    _emit("  （未执行 PSI-CA 能力核查：见上方阶段结论）")
                else:
                    _emit(
                        "  实现      : "
                        f"{ca_report.get('distribution')} "
                        f"{ca_report.get('version') or ca_report.get('pinned_version')}"
                        f"（{ca_report.get('structure')} 精确计数；只承接 "
                        + ", ".join(ca_report.get("supported_ops") or [])
                        + " / "
                        + ", ".join(ca_report.get("supported_policies") or [])
                        + "）"
                    )
                    _emit(
                        "  选用      : PSI-CA（OpenMined PSI-Cardinality，"
                        "只出交集基数；不经过 libpsi）"
                    )
                    _emit("  角色映射  : client=左侧输入（获得计数），server=右侧输入")
                    _emit(
                        "  泄漏面    : 只出计数——交集本体不交给任一方"
                        "（与 libpsi 求交的泄漏承诺不同）"
                    )
                    _emit(
                        f"  键布局    : {GRID_CODE_LAYOUT}（层级上限 L{MAX_ENCODABLE_LEVEL}；"
                        "布局须与对端一致）"
                    )
                    _emit(f"  可运行    : {'是' if ca_report.get('runnable') else '否'}")
                    for blocker in ca_report.get("blockers") or []:
                        _emit(f"    - {blocker}")
            elif result.psi_sum == PSI_SUM_BACKEND:
                sum_report = result.psi_sum_capability
                if sum_report is None:
                    _emit("  （未执行 PI-Sum 能力核查：见上方阶段结论）")
                else:
                    _emit(
                        "  实现      : "
                        f"{sum_report.get('upstream')}"
                        f"@{str(sum_report.get('upstream_commit') or '')[:7]}"
                        f"（{sum_report.get('license')}；上游无 PyPI 包，"
                        "接入对象是 Bazel 构建产物）"
                    )
                    _emit(
                        "  选用      : PI-Sum（private-join-and-compute，"
                        "只出交集基数与交集内关联值之和；不经过 libpsi）"
                    )
                    _emit(
                        "  角色映射  : client=左侧输入（带关联值、获得结果），"
                        "server=右侧输入"
                    )
                    _emit(
                        "  泄漏面    : 只出两个数（基数 + 交集内和）——"
                        "交集本体不交给任一方（与 libpsi 求交的泄漏承诺不同）"
                    )
                    _emit(
                        f"  键布局    : {GRID_CODE_LAYOUT}（层级上限 L{MAX_ENCODABLE_LEVEL}；"
                        "布局须与对端一致）"
                    )
                    _emit(f"  产物目录  : {sum_report.get('bin_dir') or '未指定'}")
                    _emit(f"  可运行    : {'是' if sum_report.get('runnable') else '否'}")
                    for blocker in sum_report.get("blockers") or []:
                        _emit(f"    - {blocker}")
            elif report is not None:
                _emit(f"  协议      : {', '.join(report.protocols) or '（未探测到）'}")
                if result.psi_protocol:
                    _emit(
                        f"  选用      : {result.psi_protocol}"
                        + curve_suffix(result.psi_protocol, result.psi_curve)
                    )
                if result.psi_protocol_params:
                    _emit(
                        "  协议参数  : "
                        + ", ".join(
                            f"{key}={value}"
                            for key, value in result.psi_protocol_params.items()
                        )
                    )
                _emit(f"  曲线      : {', '.join(report.curves) or '（未探测到）'}")
                noisy = [name for name in report.protocols if not protocol_is_exact(name)]
                if noisy:
                    _emit(f"  带噪      : {', '.join(noisy)}（差分隐私：结果不与明文保证一致）")
                _emit(f"  输入形态  : {'CSV 文件（无内存张量接口）' if report.file_io_only else '见报告'}")
                if result.resolved_inputs:
                    _emit("  输入绑定  :")
                    for name, value in result.resolved_inputs.items():
                        layout_id = (value.layout or {}).get("layout_id") or "未声明"
                        party = f"；party={value.party_id}" if value.party_id else ""
                        _emit(
                            f"    - {name} ← {value.source or '用户数据'}"
                            f"（{value.count} 个码；layout={layout_id}{party}）"
                        )
                if result.psi_protocol_capability:
                    cap = result.psi_protocol_capability
                    _emit(
                        "  分层核查  : "
                        f"backend={'是' if cap['backend_runnable'] else '否'} / "
                        f"protocol={'是' if cap['protocol_runnable'] else '否'} / "
                        f"params={'是' if cap['params_runnable'] else '否'}"
                    )
                # 密态交换的就是这些 64 位码：两方布局不一致不会报错，
                # 只会静默算错。所以把布局打在跨方对象的能力块里。
                _emit(f"  键布局    : {GRID_CODE_LAYOUT}（层级上限 L{MAX_ENCODABLE_LEVEL}；布局须与对端一致）")
                _emit(f"  可运行    : {'是' if report.runnable else '否'}")
                for blocker in report.blockers:
                    _emit(f"    - {blocker}")
            else:
                _emit("  （未执行）")
            _emit()

    # ---------------- SPU 模拟 ----------------
    _emit(DIVIDER)
    _emit("SPU simulation")
    if result.spu_runs:
        for op, run in result.spu_runs.items():
            _emit(f"  {op}: {run.status}")
            if run.ok:
                _emit(f"      outputs   : {run.outputs}")
                _emit(f"      reference : {run.reference}")
                if run.within_tolerance is not None:
                    _emit(f"      tolerance : {run.tolerance}  err={run.max_abs_error}  ok={run.within_tolerance}")
            else:
                for blocker in run.blockers:
                    _emit(f"      blocker   : {blocker}")
            if run.result_policy:
                _emit(
                    f"      policy    : {run.result_policy.get('policy')}"
                    f"（业务层暴露 {run.result_policy.get('business_value')}；"
                    f"{run.result_policy.get('disclosure')}）"
                )
            if run.reveals:
                _emit(f"      reveals   : {run.reveals}")
    else:
        stage = result.stage("spu_simulation")
        _emit(f"  （未执行：{stage.message if stage else 'n/a'}）")
    _emit()

    # ---------------- PSI 模拟 ----------------
    if result.stage("psi_simulation") is not None and result.stage("psi_simulation").status != "skipped":
        _emit(DIVIDER)
        _emit("PSI simulation")
        if result.psi_runs:
            for op, run in result.psi_runs.items():
                _emit(f"  {op}: {run.status}  [{run.protocol}"
                      + (f" / {run.curve}" if run.curve else "") + "]")
                if run.protocol_params:
                    _emit(
                        "      params    : "
                        + ", ".join(
                            f"{key}={value}" for key, value in run.protocol_params.items()
                        )
                    )
                if run.optimizer.get("applied"):
                    left_meta = dict(run.optimizer.get("left") or {})
                    right_meta = dict(run.optimizer.get("right") or {})
                    _emit(
                        "      preproc   : Geo-RR22 "
                        f"v{run.optimizer.get('version')} 排序+去重"
                        f"（dup left={left_meta.get('duplicate_count')},"
                        f" right={right_meta.get('duplicate_count')}）"
                    )
                if run.status in ("ok", "empty-input"):
                    _emit(f"      |A|={run.original_count}  |A∩B|={run.intersection_count}")
                    _emit(f"      result    : {run.value}")
                    _emit(f"      reference : {run.reference}   agree={run.agreement}")
                if run.result_semantics:
                    _emit(f"      result-sems: {run.result_semantics}")
                if run.result_policy:
                    disclosure = run.result_policy.get("leak_disclosure")
                    if disclosure:
                        # 计数档（PSI-CA）与求和档（PI-Sum）的协议泄漏面
                        # 都与 libpsi 不同：旧句"接收方仍获得交集本体"
                        # 在这两档是错的，必须换成本档的原文。
                        _emit(
                            f"      policy    : {run.result_policy.get('policy')}"
                            f"（业务层暴露 {run.result_policy.get('business_value')}；"
                            f"{disclosure}）"
                        )
                    else:
                        _emit(
                            f"      policy    : {run.result_policy.get('policy')}"
                            f"（业务层暴露 {run.result_policy.get('business_value')}；"
                            "协议内部泄漏面不随策略改变：接收方仍获得交集本体）"
                        )
                if run.party_binding:
                    roles = ", ".join(
                        f"{item.get('role')}={item.get('party_id') or '未标注'}"
                        + (f"[{item.get('name')}]" if item.get("name") else "")
                        for item in run.party_binding
                    )
                    _emit(f"      parties   : {roles}")
                if run.layout_agreement is not None:
                    verdict = {
                        True: "一致",
                        False: "不一致（已拒绝执行）",
                        None: "无法核对（仅单方声明）",
                    }.get(run.layout_agreement.get("agreement"), "未知")
                    _emit(f"      layout    : {verdict}")
                if run.subset is not None:
                    where = (
                        f"{run.subset.protocol}/{run.subset.field}"
                        if run.subset.protocol
                        else run.subset.status
                    )
                    _emit(
                        f"      subset    : {run.subset.value}  "
                        f"(mode={run.subset.mode}, {where})"
                    )
                for blocker in run.blockers:
                    _emit(f"      blocker   : {blocker}")
                if run.error:
                    _emit(f"      error     : {run.error}")
                _emit(f"      leaks     : {run.reveals}")
                for note in run.notes:
                    _emit(f"      note      : {note}")
        else:
            stage = result.stage("psi_simulation")
            _emit(f"  （未执行：{stage.message if stage else 'n/a'}）")
        _emit()

    # ---------------- 最终状态表 ----------------
    _emit(DIVIDER)
    _emit("Result")
    if result.operator_status:
        _emit(operator_status_table(result))
    else:
        _emit("  （无算子）")
    _emit()

    # ---------------- 诊断 ----------------
    if result.validation is not None and result.validation.diagnostics:
        _emit(DIVIDER)
        _emit("Diagnostics")
        _emit(result.validation.render())
        _emit()
    elif result.parse_result is not None and result.parse_result.diagnostics:
        _emit(DIVIDER)
        _emit("Diagnostics")
        for diagnostic in result.parse_result.diagnostics:
            _emit(str(diagnostic))
        _emit()

    # ---------------- 总结 ----------------
    _emit(DIVIDER)
    if result.ok:
        _emit("编译完成。")
    else:
        # 失败不一定表现为 diagnostics：能力核查/模拟阶段也会直接把阶段置为
        # error（如选用了三方 PSI 协议、或 SPU 协议名非法）。只数 result.errors
        # 会输出"编译未通过：0 个错误"这种自相矛盾的话。
        failed_stages = [s for s in result.stages if s.status == "error"]
        detail = f"{len(result.errors)} 个错误"
        if failed_stages and not result.errors:
            detail = f"阶段失败：{'、'.join(s.title for s in failed_stages)}"
        _emit(f"编译未通过：{detail}（未修改任何用户代码，请按上方建议调整）。")


def _parse_layout_shape(text: str):
    """解析 `--layout-shape candidates=1000,cells=49,attributes=4,bits=8`（D3）。

    形状缺省不传：不给形状就不预测条数，而不是拿一个"默认规模"顶替。
    """

    values: dict[str, str] = {}
    for part in text.replace(";", ",").split(","):
        item = part.strip()
        if not item:
            continue
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--layout-shape 需要 KEY=VALUE 形式，实得 {item!r}")
        if key in values:
            raise ValueError(f"--layout-shape 重复指定键 {key!r}")
        values[key] = raw.strip()
    return layout_shape_from_mapping(values)


def _parse_name_path_pairs(
    pairs: Sequence[str] | None, *, flag: str
) -> dict[str, str]:
    """`NAME=PATH` 列表 → 字典；缺等号/空名/重复名给可读错误。"""

    out: dict[str, str] = {}
    for item in pairs or ():
        name, sep, path = item.partition("=")
        name = name.strip()
        if not sep or not name or not path.strip():
            raise ValueError(f"{flag} 需要 NAME=PATH 形式，实得 {item!r}")
        if name in out:
            raise ValueError(
                f"{flag} 重复指定名字 {name!r}（前值 {out[name]!r}）"
            )
        out[name] = path.strip()
    return out


def build_command(args: argparse.Namespace) -> int:
    if args.protocol is not None and args.protocol not in SPU_PROTOCOLS:
        _emit(f"警告：协议 {args.protocol} 不在 SPU 支持清单 {SPU_PROTOCOLS} 中，仍将尝试。")

    # PSI 协议/曲线先解析。放在构造 Compiler 之前，非法名就以可读错误退出，
    # 不会以 ValueError traceback 的形式冲掉整份编译输出。
    try:
        psi_protocol, psi_curve = resolve_psi_protocol(args.psi_protocol, args.psi_curve)
    except ValueError as exc:
        _emit(f"错误：{exc}")
        return 2

    # 显式给了曲线但不会被传进协议时，如实说出来，不静默丢弃。
    # 两种情况的**原因不同**，不能共用一句话：
    #   ignored  → 协议不基于椭圆曲线，读不了；
    #   implicit → 协议自带默认曲线（DP 源码默认 25519），本项目不覆盖它。
    # 后者刻意不说"不生效"：那只说明本后端没传，不说明协议内部没读。
    if args.psi_curve:
        relation = psi_curve_relation(psi_protocol)
        if relation == "ignored":
            _emit(f"提示：协议 {psi_protocol} 不读椭圆曲线，--psi-curve 不生效。")
        elif relation == "implicit":
            _emit(
                f"提示：协议 {psi_protocol} 自带内置默认曲线（上游源码默认 25519），"
                f"本项目不覆盖，--psi-curve 未传入。"
            )

    # RR22 专用参数给了、协议却不是 RR22：如实提示，不静默忽略。
    if args.psi_rr22_low_comm_mode and psi_protocol != "PROTOCOL_RR22":
        _emit(
            f"提示：--psi-rr22-low-comm-mode 是 RR22 专用参数，"
            f"协议 {psi_protocol} 不使用它。"
        )

    # PSI-CA 档不经过 libpsi：协议/曲线参数在这个档里不生效，如实提示。
    if args.psi_count == PSI_CA_BACKEND and (args.psi_protocol or args.psi_curve):
        _emit(
            "提示：--psi-count psi-ca 走 PSI-Cardinality（OpenMined PSI，"
            "不经过 libpsi），--psi-protocol / --psi-curve 在本档不生效；"
            "本档只承接 CellSetIntersect 的计数。"
        )

    # PI-Sum 档同样不经过 libpsi：协议/曲线参数在这个档里不生效，如实提示。
    if args.psi_sum == PSI_SUM_BACKEND and (args.psi_protocol or args.psi_curve):
        _emit(
            "提示：--psi-sum pjc 走 PI-Sum（private-join-and-compute，"
            "不经过 libpsi），--psi-protocol / --psi-curve 在本档不生效；"
            "本档只承接 CellSetIntersect 的交集内求和。"
        )

    # 真实输入绑定（--input NAME=PATH / --input-layout NAME=PATH）。
    # 解析/校验失败在构造期就退出：坏输入不拖到执行阶段。
    try:
        bound_inputs = _parse_name_path_pairs(args.input, flag="--input")
        bound_layouts = _parse_name_path_pairs(args.input_layout, flag="--input-layout")
        bound_weights = _parse_name_path_pairs(
            args.psi_sum_weights, flag="--psi-sum-weights"
        )
        layout_shape = (
            _parse_layout_shape(args.layout_shape) if args.layout_shape else None
        )
        # 关联值在这里就载入：坏文件/坏形态在构造期退出，不拖到执行阶段。
        sum_weights = {
            name: _load_weight_map(path) for name, path in bound_weights.items()
        }
    except ValueError as exc:
        _emit(f"错误：{exc}")
        return 2

    try:
        compiler = Compiler(
            protocol=args.protocol,
            field=args.field,
            world_size=args.world_size,
            tolerance=args.tolerance,
            run_simulation=not args.no_simulation,
            psi_protocol=psi_protocol,
            psi_curve=psi_curve,
            psi_subset=args.psi_subset,
            psi_rr22_low_comm_mode=args.psi_rr22_low_comm_mode,
            psi_count=args.psi_count,
            psi_sum=args.psi_sum,
            psi_sum_weights=sum_weights or None,
            inputs=bound_inputs or None,
            input_layouts=bound_layouts or None,
            layout_shape=layout_shape,
        )
    except (ValueError, TypeError, FileNotFoundError) as exc:
        # 参数非法 / 输入文件缺失或损坏：可读错误，不落 traceback。
        _emit(f"错误：{exc}")
        return 2
    try:
        result = compiler.compile_file(args.source, entry=args.entry)
    except FileNotFoundError:
        _emit(f"错误：找不到源文件 {args.source}")
        return 2
    except SyntaxError as exc:
        _emit(f"错误：源码语法错误 {exc.filename}:{exc.lineno}: {exc.msg}")
        return 2

    print_result(result, verbose=args.verbose, as_json=args.json)

    if args.emit_jax:
        with open(args.emit_jax, "w", encoding="utf-8") as handle:
            handle.write(result.jax_module)
        _emit(f"生成的 JAX 模块已写入 {args.emit_jax}")

    return 0 if result.ok else 1


def ops_command(args: argparse.Namespace) -> int:
    """列出算子、表征、后端与代价。"""

    from planner import OPERATOR_REGISTRY

    _emit("已登记算子")
    _emit(DIVIDER)
    rows = []
    from planner.registry import resolve_cost

    for op, rule in OPERATOR_REGISTRY.items():
        profile = resolve_cost(rule)
        rows.append(
            [
                op,
                rule.representation,
                rule.backend,
                rule.security_level,
                "是" if rule.has_jax_impl else "否",
                f"b={profile.get('b', '-')} d={profile.get('d', '-')} R={profile.get('R', '-')}",
            ]
        )
    _emit(
        render_table(
            rows,
            ["Operation", "Representation", "Backend", "Security", "JAX", "Cost"],
        )
    )
    return 0


def check_command(args: argparse.Namespace) -> int:
    """只做环境能力核查。"""

    from backends.spu_backend import check_capabilities

    report = check_capabilities()
    _emit("SPU 能力核查")
    _emit(DIVIDER)
    _emit(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.runnable else 1


def psi_check_command(args: argparse.Namespace) -> int:
    """只做 PSI 环境能力核查。"""

    from backends.psi_backend import check_psi_capabilities

    report = check_psi_capabilities()
    _emit("PSI 能力核查")
    _emit(DIVIDER)
    _emit(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.runnable else 1


def psi_ca_check_command(args: argparse.Namespace) -> int:
    """只做 PSI-Cardinality 环境能力核查。"""

    from backends.psi_ca_backend import check_psi_ca_capabilities

    report = check_psi_ca_capabilities()
    _emit("PSI-CA 能力核查")
    _emit(DIVIDER)
    _emit(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.runnable else 1


def psi_sum_check_command(args: argparse.Namespace) -> int:
    """只做 PI-Sum 环境能力核查（上游构建产物 + flag 形态）。"""

    from backends.psi_sum_backend import check_psi_sum_capabilities

    report = check_psi_sum_capabilities(bin_dir=getattr(args, "bin_dir", None))
    _emit("PI-Sum 能力核查")
    _emit(DIVIDER)
    _emit(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.runnable else 1


def _is_int(text: str) -> bool:
    """整数字面量判定（关联值只有整数一种形态，不做浮点猜测）。"""

    try:
        int(text)
    except ValueError:
        return False
    return True


def _load_weight_map(path: str) -> dict[int, int]:
    """载入 PI-Sum 关联值：CSV（grid_code,value 两列）或 JSON（码 → 值）。

    首行若是表头（第一列不是整数）按表头跳过——只跳这一行，不猜别的；
    非整数单元一律报错，不静默丢值（丢值会让"和为 0"看起来像跑通了）。
    """

    lowered = path.lower()
    if lowered.endswith(".json"):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, Mapping):
            raise ValueError(
                f"--psi-sum-weights 的 JSON 必须是「码 → 值」对象：{path}"
            )
        return {int(code): int(value) for code, value in data.items()}
    if lowered.endswith((".csv", ".txt")):
        weights: dict[int, int] = {}
        with open(path, encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                columns = [item.strip() for item in line.split(",")]
                if line_no == 1 and not _is_int(columns[0]):
                    continue
                if len(columns) != 2:
                    raise ValueError(
                        f"--psi-sum-weights 需要两列 grid_code,value："
                        f"{path}:{line_no} 实得 {len(columns)} 列"
                    )
                if not (_is_int(columns[0]) and _is_int(columns[1])):
                    raise ValueError(
                        f"--psi-sum-weights 第 {line_no} 行不是整数对：{line!r}"
                    )
                weights[int(columns[0])] = int(columns[1])
        return weights
    raise ValueError(
        "--psi-sum-weights 只支持 .csv（grid_code,value）或 .json（码 → 值）："
        f"{path}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geo-secure",
        description=BANNER,
    )
    sub = parser.add_subparsers(dest="command")

    build = sub.add_parser("build", help="编译一个地理业务源码文件")
    build.add_argument("source", help="待编译的 .py 文件路径")
    build.add_argument("--entry", default=None, help="入口函数名（缺省分析全部函数）")
    build.add_argument(
        "--protocol",
        default=None,
        help=(
            f"SPU 协议，可选 {SPU_PROTOCOLS}；"
            "缺省 = 按实测通信代价自动选择（依据 docs/mpc_comm_baseline.json，"
            "显式指定可覆盖；REF2K 无密码学保护，不会被自动选中）"
        ),
    )
    build.add_argument("--field", default="64", help="环宽：32/64/128 或 FM32/FM64/FM128")
    build.add_argument(
        "--layout-shape",
        default=None,
        help=(
            "位平面布局（D3）的规模形状，形式 "
            "candidates=1000,cells=49,attributes=4,bits=8；缺省不预测条数"
            "（只做轴向决策），不给形状时绝不臆造规模"
        ),
    )
    build.add_argument(
        "--psi-protocol",
        default=None,
        help=(
            f"PSI 协议，两方可选 {runnable_protocols_hint()}"
            f"（缺省 {PSI_DEFAULT_PROTOCOL}）"
        ),
    )
    build.add_argument(
        "--psi-curve",
        default=None,
        help=f"PSI 椭圆曲线，ECDH 族必需（缺省 {PSI_DEFAULT_CURVE}）；DP 自带默认曲线，本项目不覆盖",
    )
    build.add_argument(
        "--psi-subset",
        choices=[SUBSET_MODE_MPC, SUBSET_MODE_PLAINTEXT],
        default=SUBSET_MODE_MPC,
        help=(
            "Contains 的子集判定走哪条路："
            f"{SUBSET_MODE_MPC}（默认，MPC 基数等值）/ "
            f"{SUBSET_MODE_PLAINTEXT}（显式明文）。"
            "MPC 不可用时会自动退回明文并把模式标为 plaintext-fallback"
        ),
    )
    build.add_argument(
        "--psi-rr22-low-comm-mode",
        action="store_true",
        help=(
            "RR22 专用参数：设置 Rr22Rarams.low_comm_mode=True（低通信模式）。"
            "只对 --psi-protocol RR22 生效，不是通用 curve 参数；"
            "缺省 false，使用 RR22 默认通信模式"
        ),
    )
    build.add_argument(
        "--psi-count",
        choices=[PSI_CA_BACKEND],
        default=None,
        help=(
            "PSI 执行档：缺省 = libpsi 两方求交（接收方获得交集本体）；"
            "psi-ca = PSI-Cardinality（OpenMined PSI，只出交集基数，"
            "只承接 CellSetIntersect，其余算子显式拒绝）"
        ),
    )
    build.add_argument(
        "--psi-sum",
        choices=[PSI_SUM_BACKEND],
        default=None,
        help=(
            "PSI 求和档：缺省 = 不走 PI-Sum；pjc = private-join-and-compute 的"
            "交集内求和（同时出交集基数与交集内关联值之和，只承接 "
            "CellSetIntersect，其余算子显式拒绝；需用 --psi-sum-weights "
            "提供左侧输入的关联值）"
        ),
    )
    build.add_argument(
        "--psi-sum-weights",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help=(
            "PI-Sum 的关联值：NAME 为左侧（client）输入名，PATH 为 CSV"
            "（grid_code,value 两列）或 JSON（码 → 值）。可重复。"
            "缺值不按 0 补齐——直接拒绝执行"
        ),
    )
    build.add_argument(
        "--input",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help=(
            "绑定真实格网输入：NAME 为源码中的输入名，PATH 为 CSV（grid_code 列）"
            "或 JSON（码数组 / {grid_codes:[...], layout:{...}}）。可重复。"
            "未绑定的 PSI 输入会回退样例默认值并在结果中如实披露"
        ),
    )
    build.add_argument(
        "--input-layout",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help=(
            "声明某方输入的格网布局清单（JSON manifest，如 {layout_id, version, "
            "bits, x_bits, ...}）。两方都声明且不一致时在进入 PSI 前拒绝执行"
        ),
    )
    build.add_argument("--world-size", type=int, default=None, help="参与方数量")
    build.add_argument("--tolerance", type=float, default=None, help="浮点/定点容差")
    build.add_argument("--no-simulation", action="store_true", help="跳过 SPU 模拟")
    build.add_argument("--emit-jax", default=None, help="把生成的 JAX 模块写入该路径")
    build.add_argument("--verbose", action="store_true", help="打印生成的 JAX 源码")
    build.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    build.set_defaults(func=build_command)

    ops = sub.add_parser("ops", help="列出算子注册表")
    ops.set_defaults(func=ops_command)

    check = sub.add_parser("check", help="核查 SPU / JAX 环境能力")
    check.set_defaults(func=check_command)

    psi_check = sub.add_parser("psi-check", help="核查 PSI 环境能力（协议/曲线/IO 形态）")
    psi_check.set_defaults(func=psi_check_command)

    psi_ca_check = sub.add_parser(
        "psi-ca-check",
        help="核查 PSI-Cardinality 环境能力（OpenMined PSI，只出交集基数）",
    )
    psi_ca_check.set_defaults(func=psi_ca_check_command)

    psi_sum_check = sub.add_parser(
        "psi-sum-check",
        help="核查 PI-Sum 环境能力（private-join-and-compute 构建产物 + flag 形态）",
    )
    psi_sum_check.add_argument(
        "--bin-dir",
        default=None,
        help="上游构建产物目录（缺省读环境变量 GIS_SPU_PJC_BIN_DIR）",
    )
    psi_sum_check.set_defaults(func=psi_sum_check_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
