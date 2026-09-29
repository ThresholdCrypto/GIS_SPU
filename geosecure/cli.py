"""geosecure CLI：`geo-secure build <source.py>`

输出六个阶段的进展，以及最终状态表（Operation / Representation / Backend / Status）。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

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
from planner import PLAN_TABLE_HEADERS, plan_table_rows

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
            if report is not None:
                _emit(f"  协议      : {', '.join(report.protocols) or '（未探测到）'}")
                if result.psi_protocol:
                    _emit(
                        f"  选用      : {result.psi_protocol}"
                        + curve_suffix(result.psi_protocol, result.psi_curve)
                    )
                _emit(f"  曲线      : {', '.join(report.curves) or '（未探测到）'}")
                noisy = [name for name in report.protocols if not protocol_is_exact(name)]
                if noisy:
                    _emit(f"  带噪      : {', '.join(noisy)}（差分隐私：结果不与明文保证一致）")
                _emit(f"  输入形态  : {'CSV 文件（无内存张量接口）' if report.file_io_only else '见报告'}")
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
                if run.status in ("ok", "empty-input"):
                    _emit(f"      |A|={run.original_count}  |A∩B|={run.intersection_count}")
                    _emit(f"      result    : {run.value}")
                    _emit(f"      reference : {run.reference}   agree={run.agreement}")
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


def build_command(args: argparse.Namespace) -> int:
    if args.protocol not in SPU_PROTOCOLS:
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

    compiler = Compiler(
        protocol=args.protocol,
        field=args.field,
        world_size=args.world_size,
        tolerance=args.tolerance,
        run_simulation=not args.no_simulation,
        psi_protocol=psi_protocol,
        psi_curve=psi_curve,
        psi_subset=args.psi_subset,
    )
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geo-secure",
        description=BANNER,
    )
    sub = parser.add_subparsers(dest="command")

    build = sub.add_parser("build", help="编译一个地理业务源码文件")
    build.add_argument("source", help="待编译的 .py 文件路径")
    build.add_argument("--entry", default=None, help="入口函数名（缺省分析全部函数）")
    build.add_argument("--protocol", default="ABY3", help=f"SPU 协议，可选 {SPU_PROTOCOLS}")
    build.add_argument("--field", default="64", help="环宽：32/64/128 或 FM32/FM64/FM128")
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
