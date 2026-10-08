"""端到端测试：Python → Geo-IR → Privacy Plan → JAX → SPU simulation。

同时覆盖课题指定的一次失败模式（错误报告四要素）与 CLI 契约。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from backends.plain import run_plain
from geosecure import Compiler, compile_file, compile_source, operator_status_table
from frontend import parse_source
from geosecure.cli import main as cli_main
from ir import GeoProgram, GeoOperation, GeoRelation, Sensitivity

from tests._helpers import EXAMPLES_DIR, PROJECT_ROOT, example


# --------------------------------------------------------------------------
# 完整流水线
# --------------------------------------------------------------------------


class TestFullPipeline:
    def test_route_conflict_end_to_end(self):
        """课题首要验证目标必须走完整六阶段。"""

        result = compile_file(example("route_conflict.py"))

        assert result.stage("parsing").status == "ok"
        assert result.stage("ir").status == "ok"
        assert result.stage("planning").status == "ok"
        assert result.stage("jax").status in ("ok", "warning")
        assert result.stage("spu_capability").status in ("ok", "warning")
        # SPU 阶段在无 SPU 环境下应明确标注，而不是静默通过
        assert result.stage("spu_simulation").status in ("ok", "warning", "skipped")

    def test_route_conflict_relation_is_standard_geo_ir(self):
        result = compile_file(example("route_conflict.py"))
        relations = result.program.relations
        assert len(relations) == 1
        relation = relations[0]
        assert (relation.subject, relation.predicate, relation.object) == (
            "route_A",
            "Intersects",
            "NoFlyZone_B",
        )
        assert str(relation) == "route_A | Intersects | NoFlyZone_B"
        assert relation.spatial_scope == "A"
        assert relation.sensitivity is Sensitivity.SENSITIVE

    def test_route_conflict_plan_is_psi_compactcellset(self):
        result = compile_file(example("route_conflict.py"))
        step = result.plan.steps[0]
        assert step.operation == "Intersects"
        assert step.representation == "CompactCellSet"
        assert step.backend == "PSI"

    def test_distance_le_end_to_end(self):
        result = compile_file(example("distance_check.py"))

        # 规划
        step = result.plan.steps[0]
        assert step.operation == "DistanceLE"
        assert step.representation == "QuantizedVector"
        assert step.backend == "MPC/SPU"

        # JAX 生成并追踪
        assert "DistanceLE" in result.jax_functions
        check = result.trace_checks["DistanceLE"]
        assert check.traceable, check.error

        # 类型由方言推断（用户没写类型注解）
        assert result.program.entity_inputs["p1"].geo_type.value == "Vector"
        assert result.program.entity_inputs["threshold"].geo_type.value == "Scalar"

    def test_weighted_sum_end_to_end(self):
        result = compile_file(example("risk_score.py"))
        ops = [step.operation for step in result.plan.steps]
        assert ops == ["WeightedSum", "TemporalOverlap"]

        for op in ("WeightedSum", "TemporalOverlap"):
            assert op in result.jax_functions, f"{op} 未生成 JAX 实现"
            assert result.trace_checks[op].traceable

        representations = {step.representation for step in result.plan.steps}
        assert representations == {"FixedPointVector", "TimeInterval"}

    def test_jax_output_matches_plain_for_every_implementation(self):
        """三份实现必须一致：明文 == JAX（整数路径精确）。"""

        cases = {
            "DistanceLE": (
                example("distance_check.py"),
                ((1, 2, 3), (1, 3, 3), 2),
            ),
            "WeightedSum": (
                example("risk_score.py"),
                ((10, 20, 30), (1, 2, 1), 1),
            ),
        }
        for op, (path, args) in cases.items():
            result = compile_file(path)
            jax_value = result.jax_outputs[op]

            if op == "DistanceLE":
                plain_value = run_plain(op, list(args[0]), list(args[1]), int(args[2])).value
            else:
                plain_value = run_plain(op, list(args[0]), list(args[1]), int(args[2])).value

            assert bool(jax_value) == plain_value or int(jax_value) == plain_value

    def test_all_examples_compile_without_errors(self):
        for name in os.listdir(EXAMPLES_DIR):
            if not name.endswith(".py") or name.startswith("_"):
                continue
            result = compile_file(example(name))
            assert not result.errors, [str(d) for d in result.errors]

    def test_simulation_reference_is_not_none_for_every_operator(self):
        """回归：SPU 参考值构造必须先过 PLAIN_ARG_BUILDERS。

        早期 reference_fn 直接把**原始样例数组**交给 run_plain。
        TemporalOverlap 需把 4 个节点数组还原为 2 个节点列表，
        参数不符 -> 抛异常 -> 被 _plain_reference 吞掉成 None
        -> SPU 实际跑对了但被误报为 error。
        """

        from geosecure.compiler import (
            DEFAULT_EXAMPLE_INPUTS,
            Compiler,
            _plain_reference,
        )

        compiler = Compiler()
        checked = 0
        for op, examples in DEFAULT_EXAMPLE_INPUTS.items():
            reference = _plain_reference(op, compiler._plain_args(op, examples))
            assert reference is not None, f"{op} 的明文参考值不应为 None"
            checked += 1
        assert checked >= 3, "至少覆盖 3 个可生成 JAX 的算子"

    def test_temporal_overlap_never_reported_as_error(self):
        """回归：risk_score 的 TemporalOverlap 不应被判为执行失败。

        该断言在所有环境下成立：无 SPU 时状态为 unavailable，
        有 SPU 时应为 ok，两者都不是 error。
        """

        result = compile_file(example("risk_score.py"))
        run = result.spu_runs.get("TemporalOverlap")
        if run is None:
            return
        assert run.status != "error", run.describe()


# --------------------------------------------------------------------------
# 编译入口的参数传递：不允许静默丢弃
# --------------------------------------------------------------------------


class TestDiagnosticAggregation:
    """同一问题在 result.diagnostics 里只允许出现一次。

    validation 是聚合了上游（解析 + 规划）的唯一汇总面；
    CompileResult.diagnostics 若再并一次 parse/plan，报告里会翻倍，
    CLI 的 "N 个错误" 也会虚高——对外交付时是可信度问题。
    """

    def test_unsupported_op_reported_exactly_once(self):
        result = compile_source(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    return geo.touches(a, b)\n"
        )
        codes = [d.code for d in result.diagnostics]
        assert codes.count("GEO_OP_UNSUPPORTED") == 1

    def test_error_count_matches_reported_errors(self):
        result = compile_source(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    return geo.touches(a, b)\n"
        )
        assert len(result.errors) == len(
            [d for d in result.diagnostics if getattr(d, "severity", "") == "error"]
        )
        assert len([d for d in result.diagnostics if d.code == "GEO_OP_UNSUPPORTED"]) == 1

    def test_unresolved_input_reported_exactly_once(self):
        result = compile_source(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    return geo.intersects(a, b[:2])\n"
        )
        codes = [d.code for d in result.diagnostics]
        assert codes.count("GEO_INPUT_UNRESOLVED") == 1

    def test_validation_carries_parse_diagnostics(self):
        """去重不能把上游诊断弄丢。"""
        result = compile_source(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    return geo.touches(a, b)\n"
        )
        validation_codes = {d.code for d in result.validation.diagnostics}
        parse_codes = {d.code for d in result.parse_result.diagnostics}
        assert parse_codes <= validation_codes
        assert set(d.code for d in result.diagnostics) == validation_codes

    def test_plan_diagnostics_appear_once(self):
        result = compile_source(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    return geo.intersects(geo.cellset_intersect(a, b), b)\n"
        )
        codes = [d.code for d in result.diagnostics]
        assert len(codes) == len(set(codes)) or "STATIC_CHECK_FAILED" not in codes

    def test_backend_missing_collapsed_to_one_message_per_code(self):
        """planner 与 validator 会各报一次"算子未登记"，报告里只留一条。"""
        from planner import plan_program
        from validator import validate_all

        program = GeoProgram(name="injected")
        program.declare_input("a")
        program.declare_input("b")
        program.add_operation(
            GeoOperation(
                op="KernelDensityEstimate",
                inputs=["a", "b"],
                output_type="Relation",
                location={"file": "injected.py", "line": 12, "col": 3},
            )
        )
        plan = plan_program(program)
        assert plan.has_errors  # planner 自己就报了
        upstream = list(program.diagnostics) + list(plan.diagnostics)
        report = validate_all(program, plan, upstream_diagnostics=upstream)
        assert [d.code for d in report.diagnostics].count("BACKEND_OP_MISSING") == 1

    def test_control_flow_statement_and_operator_messages_collapse(self):
        """同一处控制流的"语句级"与"算子级"提示合并成一条。"""
        from planner import plan_program
        from validator import validate_control_flow

        parsed = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b, flag):\n"
            "    if flag:\n"
            "        return geo.intersects(a, b)\n"
            "    return None\n"
        )
        assert [d.code for d in parsed.diagnostics].count("DYNAMIC_CONTROL_FLOW") == 1
        assert (
            len(validate_control_flow(parsed.program)) == 1
        ), "同一处控制流只应产生一条算子级提示"

    def test_distinct_messages_at_same_location_are_kept(self):
        """位置相同但措辞不同的诊断不得被误删。"""
        from validator.checks import _dedupe
        from frontend.analyzer import Diagnostic

        location = {"file": "x.py", "line": 1, "col": 0}
        items = [
            Diagnostic(code="STATIC_CHECK_FAILED", message="问题 A", location=location),
            Diagnostic(code="STATIC_CHECK_FAILED", message="问题 B", location=location),
        ]
        assert len(_dedupe(items)) == 2

class TestCompilerArguments:
    """compile_source / compile_file 必须把声明传给解析阶段，且拒绝未知参数。

    旧版用 `*kwargs` 接住一切再按白名单过滤，sensitivities / type_hints
    被吃掉后无声无息——调用方以为声明生效，实际拿到默认级别，安全判定失真。
    """

    SOURCE = (
        "from geo_privacy import geo\n"
        "def f(a, b):\n"
        "    return geo.intersects(a, b)\n"
    )

    def test_sensitivities_reach_the_ir(self):
        from ir import Sensitivity

        result = compile_source(
            self.SOURCE, sensitivities={"a": Sensitivity.SECRET, "b": Sensitivity.PUBLIC}
        )
        assert result.program.entity_inputs["a"].effective_sensitivity is Sensitivity.SECRET
        assert result.program.entity_inputs["b"].effective_sensitivity is Sensitivity.PUBLIC

    def test_sensitivities_change_the_crypto_verdict(self):
        from ir import Sensitivity

        public = compile_source(
            self.SOURCE, sensitivities={"a": Sensitivity.PUBLIC, "b": Sensitivity.PUBLIC}
        )
        secret = compile_source(
            self.SOURCE, sensitivities={"a": Sensitivity.SECRET, "b": Sensitivity.PUBLIC}
        )
        assert public.plan.steps[0].needs_crypto is False
        assert public.plan.steps[0].status == "plaintext-ok"
        assert secret.plan.steps[0].needs_crypto is True
        assert secret.plan.steps[0].status == "planned"

    def test_type_hints_reach_the_ir(self):
        result = compile_source(self.SOURCE, type_hints={"a": "CellSet"})
        assert result.program.entity_inputs["a"].geo_type.value == "CellSet"

    def test_unknown_kwarg_raises_instead_of_being_dropped(self):
        with pytest.raises(TypeError) as excinfo:
            compile_source(self.SOURCE, sentitivities={"a": "secret"})  # 故意拼错
        assert "sentitivities" in str(excinfo.value)

    def test_compile_file_forwards_sensitivities(self, tmp_path):
        from ir import Sensitivity

        path = tmp_path / "s.py"
        path.write_text(self.SOURCE, encoding="utf-8")
        result = compile_file(
            str(path), sensitivities={"a": Sensitivity.SECRET, "b": Sensitivity.PUBLIC}
        )
        assert result.program.entity_inputs["a"].effective_sensitivity is Sensitivity.SECRET

    def test_nested_call_never_yields_plaintext_verdict(self):
        """端到端回归：嵌套写法与显式写法必须给出同一套安全结论。"""
        from ir import Sensitivity

        sensitivities = {"a": Sensitivity.SECRET, "b": Sensitivity.SECRET, "c": Sensitivity.PUBLIC}
        nested = (
            "from geo_privacy import geo\n"
            "def f(a, b, c):\n"
            "    return geo.intersects(geo.cellset_intersect(a, b), c)\n"
        )
        explicit = (
            "from geo_privacy import geo\n"
            "def f(a, b, c):\n"
            "    hit = geo.cellset_intersect(a, b)\n"
            "    return geo.intersects(hit, c)\n"
        )
        nested_result = compile_source(nested, sensitivities=sensitivities)
        explicit_result = compile_source(explicit, sensitivities=sensitivities)

        nested_verdicts = [
            (step.operation, step.needs_crypto) for step in nested_result.plan.steps
        ]
        explicit_verdicts = [
            (step.operation, step.needs_crypto) for step in explicit_result.plan.steps
        ]
        assert nested_verdicts == explicit_verdicts
        assert all(needs for _, needs in nested_verdicts)
        assert all(needs for _, needs in explicit_verdicts)


# --------------------------------------------------------------------------
# 最终状态表
# --------------------------------------------------------------------------


class TestOperatorStatusTable:
    def test_status_table_has_required_columns(self):
        result = compile_file(example("risk_score.py"))
        table = operator_status_table(result)
        for header in ("Operation", "Representation", "Backend", "Status"):
            assert header in table

    def test_status_rows_cover_every_planned_step(self):
        result = compile_file(example("risk_score.py"))
        planned = {step.operation for step in result.plan.steps}
        reported = {row["operation"] for row in result.operator_status}
        assert reported == planned

    def test_status_never_claims_verified_without_spu(self):
        """无 SPU 环境下不允许出现 verified。"""

        result = compile_file(example("distance_check.py"))
        if not (result.capability and result.capability.runnable):
            statuses = {row["status"] for row in result.operator_status}
            assert "verified" not in statuses

    def test_psi_operator_status_reflects_real_execution(self):
        """PSI 族算子的状态词必须反映"实际发生了什么"。

        环境具备真实 PSI 时必须到 `verified；不具备时才允许停在
        `backend-direct`（有直连后端但本次未执行）。
        """

        from tests._helpers import has_psi

        result = compile_file(example("route_conflict.py"))
        status = result.operator_status[0]["status"]
        if has_psi():
            assert status == "verified", result.operator_status[0]
            assert result.psi_runs["Intersects"].status == "ok"
        else:
            assert status == "backend-direct", result.operator_status[0]

    def test_pipeline_reports_psi_stages(self):
        """PSI 走独立阶段；即使方案里没有 PSI 算子，阶段也必须存在且说明原因。"""

        route = compile_file(example("route_conflict.py"))
        assert route.stage("psi_capability") is not None
        assert route.stage("psi_simulation") is not None

        other = compile_file(example("distance_check.py"))
        assert other.stage("psi_simulation").status == "skipped"
        assert "PSI" in other.stage("psi_simulation").message

    def test_psi_operators_are_not_reported_as_missing_jax(self):
        """回归：接入 PSI 后，Intersects 不应再落进 `planned_but_not_jax` 那条路。"""

        result = compile_file(example("route_conflict.py"))
        message = result.stage("spu_simulation").message
        assert "方案中算子" not in message, message

    def test_psi_run_is_serialised_into_json(self):
        result = compile_file(example("route_conflict.py"))
        data = result.to_dict()
        assert "psi_capability" in data and "psi_runs" in data
        json.dumps(data, ensure_ascii=False)  # 必须可 JSON 化

        run = data["psi_runs"].get("Intersects")
        if run is not None:
            assert run["op"] == "Intersects"
            assert run["reveals"], "泄漏面必须随结果带出"
            assert run["semantics"]

    def test_psi_reference_value_is_not_none(self):
        """回归：PSI 参考值构造必须经 PLAIN_ARG_BUILDERS，否则会被当成空输入。"""

        result = compile_file(example("route_conflict.py"))
        run = result.psi_runs.get("Intersects")
        if run is not None and run.status == "ok":
            assert run.reference is not None
            assert run.agreement is True

    def test_psi_status_never_verified_when_environment_unavailable(self):
        """环境不具备 PSI 时，不得有算子被标成 verified。"""

        result = compile_file(example("route_conflict.py"))
        if not (result.psi_capability and result.psi_capability.runnable):
            statuses = {row["status"] for row in result.operator_status}
            assert "verified" not in statuses

    def test_psi_never_reports_verified_without_a_real_run(self):
        """不允许"有 PSI 实现"就当已验证：必须有真实求交结果支撑。"""

        result = compile_file(example("route_conflict.py"))
        row = result.operator_status[0]
        if row["status"] == "verified":
            run = result.psi_runs.get(row["operation"])
            assert run is not None and run.status == "ok"
            assert run.agreement is not False
            assert run.intersection_count is not None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


class TestCli:
    def test_build_prints_all_six_stages(self, capsys):
        exit_code = cli_main(["build", example("route_conflict.py")])
        output = capsys.readouterr().out

        for title in (
            "Parsing",
            "IR generation",
            "Privacy planning",
            "JAX generation",
            "SPU capability check",
            "SPU simulation",
        ):
            assert title in output, f"缺少阶段标题 {title}"

        assert exit_code in (0, 1)

    def test_build_prints_final_result_table(self, capsys):
        cli_main(["build", example("route_conflict.py")])
        output = capsys.readouterr().out

        assert "Result" in output
        assert "Operation" in output
        assert "Representation" in output
        assert "Backend" in output
        assert "Status" in output
        # 首要验证目标的关系三元组应出现在输出里
        assert "route_A | Intersects | NoFlyZone_B" in output

    def test_json_output_is_valid(self, capsys):
        cli_main(["build", example("distance_check.py"), "--json"])
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["ok"] is True
        assert len(data["plan"]) == 1
        assert data["plan"][0]["operation"] == "DistanceLE"

    def test_psi_check_subcommand_reports_capabilities(self, capsys):
        exit_code = cli_main(["psi-check"])
        output = capsys.readouterr().out
        assert "PSI 能力核查" in output
        data = json.loads(output.split("\n", 2)[2])
        assert "protocols" in data and "curves" in data
        assert exit_code in (0, 1)

    def test_build_prints_psi_stages(self, capsys):
        cli_main(["build", example("route_conflict.py")])
        output = capsys.readouterr().out
        assert "PSI capability check" in output
        assert "PSI simulation" in output
        # 泄漏面必须出现在业务可见输出里
        assert "leaks" in output

    def test_ops_subcommand_lists_all_operators(self, capsys):
        cli_main(["ops"])
        output = capsys.readouterr().out
        for op in (
            "Intersects",
            "Contains",
            "DistanceLE",
            "CellSetIntersect",
            "WeightedSum",
            "TemporalOverlap",
        ):
            assert op in output

    def test_check_subcommand_reports_environment(self, capsys):
        cli_main(["check"])
        output = capsys.readouterr().out
        data = json.loads(output[output.index("{") :])
        assert "blockers" in data
        assert "jax" in data

    def test_missing_file_returns_error_code(self, capsys):
        exit_code = cli_main(["build", os.path.join(EXAMPLES_DIR, "does_not_exist.py")])
        assert exit_code == 2

    def test_no_simulation_flag_skips_stage(self, capsys):
        cli_main(["build", example("distance_check.py"), "--no-simulation"])
        output = capsys.readouterr().out
        assert "SKIPPED" in output

    # ---- PSI 协议开关（层面 1：换用 SPU 已有的另一个协议） ----

    def test_psi_protocol_reaches_the_real_run(self, capsys):
        """`--psi-protocol` 必须真的改变执行协议，而不是只改显示。"""

        exit_code = cli_main(
            ["build", example("route_conflict.py"), "--psi-protocol", "KKRT"]
        )
        output = capsys.readouterr().out
        assert exit_code == 0
        assert "PROTOCOL_KKRT" in output
        assert "选用      : PROTOCOL_KKRT" in output
        # 关键：必须体现在**真实执行**那一行（`Intersects: ok  [PROTOCOL_KKRT]`），
        # 只断言"选用"那一行是不够的——那行来自能力阶段，与是否真的传下去无关。
        assert "[PROTOCOL_KKRT]" in output

    def test_psi_protocol_reaches_the_result_object(self):
        from geosecure import Compiler

        compiler = Compiler(psi_protocol="RR22")
        result = compiler.compile_file(example("route_conflict.py"))
        assert result.psi_protocol == "PROTOCOL_RR22"
        # 协议不读曲线时，曲线必须归零，不能留一个不生效的值
        assert result.psi_curve is None
        for run in result.psi_runs.values():
            assert run.protocol == "PROTOCOL_RR22"

    def test_rr22_low_comm_mode_reaches_the_real_run(self, capsys):
        """`--psi-rr22-low-comm-mode` 必须体现在真实执行的参数档里。"""

        exit_code = cli_main(
            [
                "build", example("route_conflict.py"),
                "--psi-protocol", "RR22",
                "--psi-rr22-low-comm-mode",
            ]
        )
        output = capsys.readouterr().out
        assert exit_code == 0
        assert "[PROTOCOL_RR22]" in output
        assert "low_comm_mode=True" in output
        from tests._helpers import has_psi

        if has_psi():
            assert "verified" in output

    def test_rr22_flag_with_another_protocol_is_disclosed(self, capsys):
        """RR22 专用参数配了别的协议：必须提示它不生效，而不是静默忽略。"""

        cli_main(
            [
                "build", example("route_conflict.py"),
                "--psi-protocol", "KKRT",
                "--psi-rr22-low-comm-mode",
            ]
        )
        output = capsys.readouterr().out
        assert "RR22 专用参数" in output

    def test_unknown_psi_protocol_exits_2_without_traceback(self, capsys):
        exit_code = cli_main(
            ["build", example("route_conflict.py"), "--psi-protocol", "NOPE"]
        )
        output = capsys.readouterr().out
        assert exit_code == 2
        assert "未知 PSI 协议" in output
        assert "PROTOCOL_ECDH" in output          # 必须给出可用清单
        assert "Traceback" not in output

    def test_three_party_psi_protocol_is_refused_readably(self, capsys):
        """`ECDH_3PC` 需 3 方：必须是可读错误，而不是 libpsi 的 C++ 栈。"""

        exit_code = cli_main(
            ["build", example("route_conflict.py"), "--psi-protocol", "ECDH_3PC"]
        )
        output = capsys.readouterr().out
        assert exit_code == 1
        assert "需要 3 个参与方" in output
        assert "PROTOCOL_ECDH_3PC" in output
        assert "Enforce fail" not in output
        assert "Stacktrace" not in output
        # 不得报"verified"：协议根本没跑
        assert "verified" not in output

    def test_curve_is_validated_even_when_protocol_ignores_it(self, capsys):
        """拼错的曲线名不能被静默忽略，哪怕当前协议不读它。"""

        exit_code = cli_main(
            ["build", example("route_conflict.py"),
             "--psi-protocol", "KKRT", "--psi-curve", "CURVE_NOPE"]
        )
        output = capsys.readouterr().out
        assert exit_code == 2
        assert "未知椭圆曲线" in output

    def test_ignored_curve_is_disclosed_not_silently_dropped(self, capsys):
        exit_code = cli_main(
            ["build", example("route_conflict.py"),
             "--psi-protocol", "KKRT", "--psi-curve", "CURVE_25519"]
        )
        output = capsys.readouterr().out
        assert exit_code == 0
        assert "不读椭圆曲线" in output

    def test_dp_curve_is_not_injected_and_not_called_ineffective(self, capsys):
        """DP 自带内置默认曲线（上游源码默认 25519）。

        本项目**不覆盖**它——覆盖是否在协议内部生效没核对过。因此：
        - 不许说"不生效"（那是没核对的断言）；
        - 也不许把曲线留在结果里（留着会让人以为它被传下去了）。
        """

        exit_code = cli_main(
            ["build", example("route_conflict.py"),
             "--psi-protocol", "DP", "--psi-curve", "CURVE_SM2"]
        )
        output = capsys.readouterr().out
        assert exit_code == 0, output
        assert "本项目不覆盖" in output
        assert "不生效" not in output

        from geosecure import Compiler

        result = Compiler(psi_protocol="DP", psi_curve="CURVE_SM2").compile_file(
            example("route_conflict.py")
        )
        assert result.psi_protocol == "PROTOCOL_DP"
        assert result.psi_curve is None
        for run in result.psi_runs.values():
            assert run.curve is None

    def test_noisy_psi_protocol_is_not_reported_as_verified(self, capsys):
        """`--psi-protocol DP` 跑得起来，但结果带噪。

        不得报 `verified`（本项目里 verified 专指"与明文一致"），也不得报
        "经真实 PSI 求交验证"；但**也不该**是失败——它确实执行了。
        """

        from tests._helpers import has_psi

        if not has_psi():
            pytest.skip("当前环境不具备真实 PSI 执行能力，DP 的带噪披露无从呈现")

        exit_code = cli_main(
            ["build", example("route_conflict.py"), "--psi-protocol", "DP"]
        )
        output = capsys.readouterr().out
        assert exit_code == 0, output
        assert "executed-noisy" in output
        assert "带噪" in output
        assert "经真实 PSI 求交验证" not in output

    def test_dp_stages_are_warnings_not_errors(self):
        """DP 的两个 PSI 阶段都必须是 warning：执行了，但不构成一致性验证。

        回归：带噪协议与明文不一致曾被升级成 `error`，使编译**随机**失败。
        阶段状态必须与"是否恰好对上"无关，因此这里断言的是机制而不是运气。
        """

        from tests._helpers import has_psi

        if not has_psi():
            pytest.skip("当前环境不具备真实 PSI 执行能力，DP 阶段状态无从呈现")

        from geosecure import Compiler

        result = Compiler(psi_protocol="DP").compile_file(example("route_conflict.py"))
        assert result.ok, [s.message for s in result.stages if s.status == "error"]
        assert "error" not in {s.status for s in result.stages}

        capability = result.stage("psi_capability")
        assert capability.status == "warning", capability.message
        assert "带噪" in capability.message

        simulation = result.stage("psi_simulation")
        assert simulation.status == "warning", simulation.message
        assert "带噪" in simulation.message

        run = result.psi_runs["Intersects"]
        assert run.status == "ok", run.error
        assert result.operator_status[0]["status"] == "executed-noisy"

    def test_three_party_alternatives_mark_the_noisy_protocol(self, capsys):
        """替换建议里不能把 DP 摆成 ECDH 的等价替代。"""

        exit_code = cli_main(
            ["build", example("route_conflict.py"), "--psi-protocol", "ECDH_3PC"]
        )
        output = capsys.readouterr().out
        assert exit_code == 1
        assert "PROTOCOL_DP*" in output

    def test_build_help_marks_the_noisy_protocol(self, capsys):
        """`--help` 的协议清单同样不能把 DP 摆成等价替代——那是使用者的第一入口。"""

        with pytest.raises(SystemExit) as excinfo:
            cli_main(["build", "--help"])
        assert excinfo.value.code == 0
        output = capsys.readouterr().out
        assert "PROTOCOL_DP*" in output
        assert "带 * 者" in output

    def test_invalid_spu_protocol_does_not_crash_the_pipeline(self, capsys):
        """回归：非法 SPU 协议曾让整条流水线带 traceback 崩掉。

        `run_spu_simulation` 对非法协议名是 fail-fast 抛 ValueError（有测试
        守着该契约），编译器这一层必须接住它并转成可读错误。
        """

        exit_code = cli_main(
            ["build", example("distance_check.py"), "--protocol", "SPDZ2K"]
        )
        output = capsys.readouterr().out
        assert exit_code == 1
        assert "Traceback" not in output
        assert "SPDZ2K" in output
        # 总结行不得自相矛盾地报"0 个错误"
        assert "0 个错误" not in output

    def test_success_summary_still_reports_completion(self, capsys):
        cli_main(["build", example("route_conflict.py")])
        assert "编译完成。" in capsys.readouterr().out

    def test_mpc_protocol_choice_reaches_execution(self):
        """`--protocol` 进规划层后仍要一路贯到执行：换成 CHEETAH 必须真跑通。

        本用例同时锁住"MPC 协议不再只在运行时才被认得"——编译期校验放行的
        协议，执行期必须真的能执行（否则就是两道闸门口径不一致）。
        """

        from geosecure import Compiler
        from tests._helpers import has_spu

        result = Compiler(protocol="CHEETAH").compile_file(
            example("distance_check.py")
        )
        assert result.ok, [s.message for s in result.stages if s.status == "error"]

        step = result.plan.steps[0]
        assert step.mpc_protocol == "CHEETAH"

        if has_spu():
            run = result.spu_runs["DistanceLE"]
            assert run.protocol == "CHEETAH"
            assert run.status == "ok", run.error

    def test_contains_subset_defaults_to_the_mpc_round(self, capsys):
        """默认走 MPC：状态词仍是 verified，且子集判定的模式与设置在输出里可见。"""

        exit_code = cli_main(["build", example("vertical_conflict.py")])
        output = capsys.readouterr().out
        assert exit_code == 0, output

        from tests._helpers import has_psi

        if has_psi():
            # 子集判定那几行只在 PSI 真跑得起来时才输出
            assert "subset    : " in output
            assert "mode=mpc" in output
            assert "ABY3/FM64" in output

        result = Compiler().compile_file(example("vertical_conflict.py"))
        rows = {row["operation"]: row for row in result.operator_status}
        if has_psi():
            assert rows["Contains"]["status"] == "verified", rows["Contains"]
            assert rows["Contains"]["subset"]["mode"] == "mpc"
        else:
            assert rows["Contains"]["status"] == "backend-direct", rows["Contains"]

    def test_psi_subset_plaintext_reaches_the_status_table(self, capsys):
        """显式选明文后不许再报 verified，也不许把这步说成密态。"""

        from tests._helpers import has_psi

        exit_code = cli_main(
            ["build", example("vertical_conflict.py"), "--psi-subset", "plaintext"]
        )
        output = capsys.readouterr().out
        assert exit_code == 0, output

        if not has_psi():
            return
        assert "mode=plaintext" in output
        assert "subset-plaintext" in output
        result = Compiler(psi_subset="plaintext").compile_file(
            example("vertical_conflict.py")
        )
        rows = {row["operation"]: row for row in result.operator_status}
        assert rows["Contains"]["status"] == "subset-plaintext"
        assert "明文比较" in rows["Contains"]["reveals"]
        # PSI 段本身仍是真跑的，不能因为子集判定降级就把它说成没执行
        assert rows["Contains"]["psi_status"] == "ok"

    def test_psi_subset_rejects_unknown_mode(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli_main(["build", example("vertical_conflict.py"), "--psi-subset", "nope"])
        assert excinfo.value.code == 2
        assert "invalid choice" in capsys.readouterr().err

    def test_build_help_documents_the_subset_switch(self, capsys):
        with pytest.raises(SystemExit):
            cli_main(["build", "--help"])
        output = capsys.readouterr().out
        assert "--psi-subset" in output
        assert "mpc" in output and "plaintext" in output
        # argparse 会在终端宽度处折行，长词可能被断成 "plaintext-\nfallback"，
        # 故按词断言而不是按整串断言。
        assert "fallback" in output

    def test_cli_runs_as_subprocess_module(self):
        """按模块方式调用也应可用（模拟 console_scripts 入口）。"""

        env = dict(os.environ)
        env["PYTHONPATH"] = PROJECT_ROOT
        env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            [sys.executable, "-m", "geosecure.cli", "ops"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        assert "Intersects" in completed.stdout


# --------------------------------------------------------------------------
# 六类失败模式的错误报告（位置 / 原因 / 替代算子 / 预计代价）
# --------------------------------------------------------------------------


class TestFailureReporting:
    def _assert_actionable(self, diagnostics):
        assert diagnostics, "应产生诊断"
        diagnostic = diagnostics[0]
        assert diagnostic.location, "缺少错误位置"
        assert diagnostic.cause, "缺少问题原因"
        assert diagnostic.suggested_op, "缺少建议替代算子"
        assert diagnostic.estimated_cost is not None, "缺少预计隐私计算代价"
        return diagnostic

    def test_class1_unsupported_geo_operator(self):
        """第 1 类：地理算子不支持。"""

        result = compile_source(
            "from geo_privacy import geo\n"
            "def f(route, zone):\n"
            "    return geo.buffer_zone(route, zone)\n"
        )
        assert result.errors
        codes = {d.code for d in result.errors}
        assert "GEO_OP_UNSUPPORTED" in codes
        diagnostic = next(d for d in result.errors if d.code == "GEO_OP_UNSUPPORTED")
        assert diagnostic.location_str.endswith(":3:11")
        assert diagnostic.suggested_op

    def test_class1_reports_location_and_cost(self):
        result = compile_source(
            "from geo_privacy import geo\n"
            "def f(route, zone):\n"
            "    return geo.intersection_of(route, zone)\n"
        )
        diagnostic = self._assert_actionable(result.errors)
        assert diagnostic.suggested_op == "Intersects"
        assert diagnostic.estimated_cost["backend"] == "PSI"

    def test_class2_jax_not_traceable_is_detected(self):
        """第 2 类：JAX 算子无法追踪（这里用生成的坏代码验证检测能力）。"""

        from backends.jax_backend import check_traceable

        def bad(x):
            import jax.numpy as jnp

            if jnp.sum(x) > 0:  # 依赖数据取值
                return x
            return -x

        import numpy as np

        check = check_traceable(bad, (np.array([1.0, 2.0]),))
        assert not check.traceable
        assert check.error_type

    def test_class3_spu_unsupported_is_reported(self):
        """第 3 类：SPU 当前版本不支持（本环境下必然触发）。"""

        result = compile_file(example("distance_check.py"))
        assert result.capability is not None
        if not result.capability.runnable:
            codes = {d.code for d in result.validation.diagnostics}
            assert "SPU_UNSUPPORTED" in codes
            diagnostic = next(
                d for d in result.validation.diagnostics if d.code == "SPU_UNSUPPORTED"
            )
            assert diagnostic.cause

    def test_class4_dynamic_control_flow_is_reported(self):
        """第 4 类：动态 Python 控制流无法编译。"""

        result = compile_source(
            "from geo_privacy import geo\n"
            "def f(route, zone, flag):\n"
            "    if flag:\n"
            "        return geo.intersects(route, zone)\n"
            "    return None\n"
        )
        codes = {d.code for d in result.diagnostics}
        assert "DYNAMIC_CONTROL_FLOW" in codes
        diagnostic = next(d for d in result.diagnostics if d.code == "DYNAMIC_CONTROL_FLOW")
        assert diagnostic.location_str.endswith(":3:4")
        assert diagnostic.cause
        assert diagnostic.suggested_op
        assert diagnostic.estimated_cost is not None

    def test_class5_backend_missing_is_reported(self):
        """第 5 类：后端没有对应隐私算子（IR 层直接注入未登记算子）。"""

        program = GeoProgram(name="injected")
        program.declare_input("a")
        program.declare_input("b")
        program.add_operation(
            GeoOperation(
                op="KernelDensityEstimate",
                inputs=["a", "b"],
                output_type="Relation",
                location={"file": "injected.py", "line": 12, "col": 3},
            )
        )
        from planner import plan_program

        plan = plan_program(program)
        assert plan.has_errors
        diagnostic = plan.errors[0]
        assert diagnostic.code == "BACKEND_OP_MISSING"
        assert diagnostic.location_str == "injected.py:12:3"
        assert diagnostic.suggested_op in ("DistanceLE", "Intersects", "WeightedSum")
        assert diagnostic.estimated_cost is not None

    def test_all_six_failure_classes_are_registered(self):
        # 课题指定五类；本版补第 6 类"高度层号超出 Z 位域"——
        # 三维格网接入层独有的容量约束，不属于原五类中的任何一类。
        from validator import FAILURE_CLASSES

        codes = {fc.code for fc in FAILURE_CLASSES}
        assert codes == {
            "GEO_OP_UNSUPPORTED",
            "JAX_NOT_TRACEABLE",
            "SPU_UNSUPPORTED",
            "DYNAMIC_CONTROL_FLOW",
            "BACKEND_OP_MISSING",
            "HEIGHT_LAYER_UNSUPPORTED",
        }
        for failure in FAILURE_CLASSES:
            assert failure.name and failure.description and failure.remedy

    def test_compiler_never_modifies_user_source(self, tmp_path):
        """编译器绝不允许改写用户代码。"""

        source = (
            "from geo_privacy import geo\n"
            "def f(route, zone):\n"
            "    return geo.intersection_of(route, zone)\n"
        )
        path = tmp_path / "user_code.py"
        path.write_text(source, encoding="utf-8")

        compile_file(str(path))

        assert path.read_text(encoding="utf-8") == source

    def test_error_report_renders_all_four_elements(self):
        result = compile_source(
            "from geo_privacy import geo\n"
            "def f(route, zone):\n"
            "    return geo.touches(route, zone)\n"
        )
        rendered = result.validation.render()
        assert "GEO_OP_UNSUPPORTED" in rendered
        assert "原因" in rendered
        assert "建议" in rendered
        assert "替代算子" in rendered


# --------------------------------------------------------------------------
# 确定性
# --------------------------------------------------------------------------


class TestDeterminism:
    def test_same_source_gives_same_ir_and_plan(self):
        first = compile_file(example("risk_score.py"))
        second = compile_file(example("risk_score.py"))
        assert first.program.to_dict() == second.program.to_dict()
        assert [s.to_dict() for s in first.plan.steps] == [s.to_dict() for s in second.plan.steps]

    def test_generated_code_is_byte_identical_across_runs(self):
        first = compile_file(example("risk_score.py"))
        second = compile_file(example("risk_score.py"))
        assert first.jax_module == second.jax_module

    def test_emit_jax_module_is_valid_python(self, tmp_path):
        target = tmp_path / "generated.py"
        result = compile_file(example("risk_score.py"))
        target.write_text(result.jax_module, encoding="utf-8")
        compile(target.read_text(encoding="utf-8"), str(target), "exec")
