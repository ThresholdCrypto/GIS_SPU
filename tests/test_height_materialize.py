"""高度带物化算子（`geo.height_band`）的回归测试。

本文件锁定一个**曾被静默接受**的缺陷：README 7.1.1 / examples /
geo_privacy docstring 都把 `geo.height_band(...)` 写成用户可写的业务代码，
但方言表里没有它，而模块顶层语句根本不被遍历。三种写法的实际结果分别是：

    route = geo.height_band(...)        # 模块顶层 → exit=0，PSI verified，
                                        #   但高度带**根本没进 Geo-IR**（假验证）
    band = geo.height_band(...)         # 函数体内 → exit=1，算子不支持
    geo.intersects(geo.height_band(..))  # 内联 → 参数按 SECRET 计入

第一种最危险：编译报成功、状态显示 verified，而"高度带参与判定"这件事
从未发生。因此这里逐条钉死：物化算子必须进 IR、必须可校验、必须不被
当成密态算子，也不得让它的输入敏感度被写死。
"""

from __future__ import annotations

import pytest

from frontend import parse_source
from frontend.dialect import GEO_DIALECT, PLAINTEXT_UTILITIES, resolve_dialect
from geo_privacy import geo
from geo_privacy.geo import GEO_OPERATIONS
from ir import Sensitivity
from planner import get_rule, plan_program
from planner.registry import registered_ops
from semantic import suggest_ops
from validator import validate_all, validate_height_layer_capacity
from tests._helpers import example


#: 模块顶层写法 —— README 7.1.1 原文形态
MODULE_LEVEL_SOURCE = (
    "from geo_privacy import geo\n"
    "\n"
    "route = geo.height_band(x=21861, y=27702, height_min=0, height_max=20000, level=15)\n"
    "zone  = geo.height_band(x=21861, y=27702, height_min=5000, height_max=9000, level=15)\n"
    "\n"
    "verdict = geo.intersects(route, zone)\n"
)


# --------------------------------------------------------------------------
# 方言与注册表：两处必须同源
# --------------------------------------------------------------------------


class TestDialectRegistration:
    def test_facade_methods_are_all_in_the_dialect(self):
        """facade 上的每个 geo 方法都必须有确定归属：算子 / 物化 / 明文工具。

        第三种状态（既不是算子、也没被说明）正是 geo.height_band 出问题时的
        状态——用户照着文档写，得到的却是静默丢弃。这里把"不留灰色地带"
        钉成断言：门面上出现新方法而未归档时，本测试会失败。
        """

        facade = {n for n in dir(geo) if not n.startswith("_")}
        accounted = set(GEO_DIALECT) | set(PLAINTEXT_UTILITIES)
        missing = sorted(facade - accounted)
        assert not missing, f"这些 geo 方法没有归属，用户写出来会被静默忽略：{missing}"

    def test_plaintext_utilities_are_not_declared_as_operators(self):
        """明文工具不得混进方言表：它们没有算子语义，进去会被规划成密态算子。"""

        assert not (set(PLAINTEXT_UTILITIES) & set(GEO_DIALECT))

    def test_plaintext_utility_is_explained_not_rejected(self):
        """回归：模块遍历不得把 geo.quantize 这种明文工具报成"算子不支持"。"""

        result = parse_source(
            "from geo_privacy import geo\n"
            "THRESHOLD = geo.quantize(120.0, 0.0, 300.0, 8)\n"
            "def f(p1, p2):\n"
            "    return geo.distance_le(p1, p2, THRESHOLD)\n",
            filename="util.py",
        )
        codes = {d.code for d in result.diagnostics}
        assert "GEO_OP_UNSUPPORTED" not in codes
        assert "GEO_PLAINTEXT_UTILITY" in codes
        assert result.ok, [str(d) for d in result.diagnostics]
        # 也不得进 Geo-IR
        assert [op.op for op in result.program.operations] == ["DistanceLE"]

    def test_height_band_is_registered_as_materialize(self):
        entry = GEO_DIALECT["height_band"]
        assert entry.op == "HeightBand"
        assert entry.materializes is True
        # 产出类型是格网集合——下游按 CellSet 消费，才能喂给集合族算子
        assert str(entry.output_geo_type) == "CellSet"

    def test_config_defaults_match_the_facade_signature(self):
        """方言里的默认值必须与 _GeoFacade.height_band 的签名逐字一致。

        两处漂移会让"编译期算出的层集合"与"明文参考实现算出的"不一致，
        而两者本该是同一套码。
        """

        import inspect

        signature = inspect.signature(geo.height_band)
        entry = GEO_DIALECT["height_band"]
        for name in entry.config_params:
            if name in entry.required_params:
                continue
            assert name in signature.parameters, name
            assert entry.defaults[name] == signature.parameters[name].default, name

    def test_required_params_have_no_default(self):
        import inspect

        signature = inspect.signature(geo.height_band)
        for name in GEO_DIALECT["height_band"].required_params:
            assert signature.parameters[name].default is inspect.Parameter.empty, name

    def test_height_band_is_registered_in_the_operator_registry(self):
        assert "HeightBand" in registered_ops()
        rule = get_rule("HeightBand")
        assert rule.is_plaintext_local is True
        assert rule.backend == "Plaintext"

    def test_materialize_ops_are_a_subset_of_the_dialect(self):
        """表里声明的物化算子必须都能被解析出来（别名也算）。"""

        for method, entry in GEO_DIALECT.items():
            if entry.materializes:
                assert resolve_dialect(method) is entry

    def test_geo_operations_table_declares_the_same_keys(self):
        """GEO_OPERATIONS 是唯一真源；方言表的字段必须能在它里面找到出处。"""

        spec = GEO_OPERATIONS["height_band"]
        assert spec["materializes"] is True
        assert set(spec["value_params"]) | set(spec["config_params"]) == set(
            spec["required_params"]
        ) | set(spec.get("defaults", {}))


class TestAliases:
    @pytest.mark.parametrize(
        "alias",
        ["altitude_band", "elevation_band", "高度带"],
    )
    def test_dialect_aliases_resolve(self, alias):
        assert resolve_dialect(alias).op == "HeightBand"

    @pytest.mark.parametrize(
        "written", ["height_band", "altitude_band", "height_bandd", "高度带"]
    )
    def test_suggest_ops_offers_an_alternative(self, written):
        """写错名字时必须给得出替代算子，不能只报"不支持"。"""

        assert "HeightBand" in suggest_ops(written)


# --------------------------------------------------------------------------
# 前端：三种写法都必须进 Geo-IR
# --------------------------------------------------------------------------


class TestModuleLevelTraversal:
    def test_documented_module_level_form_reaches_the_ir(self):
        """回归：模块顶层调用以前被整条丢弃，编译却报成功。"""

        result = parse_source(MODULE_LEVEL_SOURCE, filename="vertical.py")
        assert result.ok, [str(d) for d in result.diagnostics]
        assert [op.op for op in result.program.operations] == [
            "HeightBand",
            "HeightBand",
            "Intersects",
        ]
        assert result.program.operations[0].output_name == "route"
        assert result.program.operations[1].output_name == "zone"

    def test_module_level_band_output_feeds_the_set_operator(self):
        """链式：Intersects 必须消费物化出来的两个集合名。"""

        result = parse_source(MODULE_LEVEL_SOURCE, filename="vertical.py")
        operation = result.program.operations[-1]
        assert list(operation.inputs) == ["route", "zone"]

    def test_module_level_calls_are_never_silently_accepted(self):
        """模块顶层的可疑写法必须留下诊断，而不是无声通过。"""

        result = parse_source(
            "from geo_privacy import geo\n"
            "verdict = geo.intersects(route_A, NoFlyZone_B)\n",
            filename="loose.py",
        )
        codes = {d.code for d in result.diagnostics}
        assert "GEO_INPUT_UNRESOLVED" in codes

    def test_main_guard_is_not_reported_as_dynamic_control_flow(self):
        """回归：模块遍历不得把自家 __main__ 脚手架误报成密态分支。"""

        result = parse_source(
            "from geo_privacy import geo\n"
            "def check(route, zone):\n"
            "    return geo.intersects(route, zone)\n"
            'if __name__ == "__main__":\n'
            '    print("hi")\n',
            filename="scaffold.py",
        )
        assert result.ok, [str(d) for d in result.diagnostics]
        assert not [d for d in result.diagnostics if d.code == "DYNAMIC_CONTROL_FLOW"]

    def test_try_import_fallback_is_not_dynamic_control_flow(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "try:\n"
            "    import numpy as np\n"
            "except ImportError:\n"
            "    np = None\n"
            "def f(route, zone):\n"
            "    return geo.intersects(route, zone)\n",
            filename="optional.py",
        )
        assert result.ok, [str(d) for d in result.diagnostics]

    def test_geo_call_inside_main_guard_is_not_dynamic_control_flow(self):
        """关键反例：算子确实落在 `if __name__` 块里，但仍不得被误报。

        上一条用例的 __main__ 块里没有算子，因此它对"是否把字面量比较
        当动态分支"这个判据不敏感。这里把算子放进块内——若把 Compare
        一律当动态分支，本用例会报 DYNAMIC_CONTROL_FLOW。
        """

        result = parse_source(
            "from geo_privacy import geo\n"
            "band = geo.height_band(x=1, y=2, height_min=0, height_max=300, level=15)\n"
            'if __name__ == "__main__":\n'
            "    verdict = geo.intersects(band, band)\n",
            filename="guard_body.py",
        )
        assert result.ok, [str(d) for d in result.diagnostics]
        report = validate_all(result.program, plan_program(result.program))
        assert "DYNAMIC_CONTROL_FLOW" not in {d.code for d in report.diagnostics}
        # 算子必须真的进了 IR，而不是被"跳过"了
        assert [op.op for op in result.program.operations] == ["HeightBand", "Intersects"]

    def test_literal_true_guard_is_not_dynamic_control_flow(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "band = geo.height_band(x=1, y=2, height_min=0, height_max=300, level=15)\n"
            "if True:\n"
            "    verdict = geo.intersects(band, band)\n",
            filename="true_guard.py",
        )
        assert result.ok, [str(d) for d in result.diagnostics]

    def test_dynamic_module_branch_is_still_caught(self):
        """真正的运行期分支在模块层同样不可编译，不能被放行。"""

        result = parse_source(
            "from geo_privacy import geo\n"
            "band = geo.height_band(x=1, y=2, height_min=0, height_max=300, level=15)\n"
            "if compute_flag():\n"
            "    verdict = geo.intersects(band, band)\n",
            filename="branched.py",
        )
        plan = plan_program(result.program)
        report = validate_all(result.program, plan)
        assert "DYNAMIC_CONTROL_FLOW" in {d.code for d in report.diagnostics}


class TestArgumentForms:
    def test_positional_form_matches_the_facade_signature(self):
        """位置写法在本方明文里本来就能跑通，编译器不得误报。"""

        result = parse_source(
            "from geo_privacy import geo\n"
            "band = geo.height_band(1, 2, 0, 300, 15)\n"
            "verdict = geo.intersects(band, band)\n",
            filename="positional.py",
        )
        assert result.ok, [str(d) for d in result.diagnostics]
        params = result.program.operations[0].params
        assert params["height_max"] == 300
        assert params["level"] == 15

    def test_positional_and_keyword_forms_give_the_same_ir(self):
        positional = parse_source(
            "from geo_privacy import geo\n"
            "band = geo.height_band(1, 2, 0, 300, 15)\n"
            "verdict = geo.intersects(band, band)\n",
            filename="a.py",
        )
        keyword = parse_source(
            "from geo_privacy import geo\n"
            "band = geo.height_band(x=1, y=2, height_min=0, height_max=300, level=15)\n"
            "verdict = geo.intersects(band, band)\n",
            filename="a.py",
        )
        left = positional.program.operations[0]
        right = keyword.program.operations[0]
        assert left.op == right.op
        assert dict(left.params) == dict(right.params)

    def test_data_params_go_to_inputs_and_config_to_params(self):
        result = parse_source(MODULE_LEVEL_SOURCE, filename="v.py")
        operation = result.program.operations[0]
        # 被编码的数据进 inputs（敏感度判定看它）
        assert list(operation.inputs) == ["21861", "27702"]
        # 物化配置进 params，且带上 height_level 供第 6 类失败检查使用
        for key in ("height_min", "height_max", "level", "height_level"):
            assert key in operation.params

    def test_keyword_form_inside_a_function_infers_scalar_params(self):
        """回归：命名实参没被类型推断覆盖时，x/y 会停在默认 EntitySet 并误报。"""

        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(x, y, zone):\n"
            "    band = geo.height_band(x=x, y=y, height_min=0, height_max=300, level=20)\n"
            "    return geo.intersects(band, zone)\n",
            filename="infer.py",
        )
        assert result.ok, [str(d) for d in result.diagnostics]
        assert str(result.program.entity_inputs["x"].geo_type) == "Scalar"
        assert str(result.program.entity_inputs["y"].geo_type) == "Scalar"

    def test_inline_nested_form_is_emitted_innermost_first(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(zone, x, y):\n"
            "    return geo.intersects(geo.height_band(x, y, 0, 300, 20), zone)\n",
            filename="inline.py",
        )
        assert result.ok, [str(d) for d in result.diagnostics]
        assert [op.op for op in result.program.operations] == ["HeightBand", "Intersects"]


class TestMaterializeArgumentDiagnostics:
    def test_missing_required_parameter_is_reported(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "band = geo.height_band(x=1, y=2, level=15)\n",
            filename="short.py",
        )
        diagnostic = next(
            d for d in result.diagnostics if d.code == "GEO_ARITY_MISMATCH"
        )
        assert "height_min" in diagnostic.message
        assert diagnostic.suggestion

    def test_unknown_keyword_is_reported_not_ignored(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "band = geo.height_band(x=1, y=2, height_min=0, height_max=300, "
            "level=15, heigth_max=9)\n",
            filename="typo.py",
        )
        assert any("heigth_max" in d.message for d in result.diagnostics)

    def test_non_literal_configuration_is_rejected(self):
        """层号必须编译期可定：运行期层号无法静态展开层集合。"""

        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(level):\n"
            "    band = geo.height_band(x=1, y=2, height_min=0, height_max=300, level=level)\n"
            "    return geo.intersects(band, band)\n",
            filename="dynamic_level.py",
        )
        assert result.ok is False
        diagnostic = next(
            d for d in result.diagnostics if d.code == "STATIC_CHECK_FAILED"
        )
        assert "level" in diagnostic.message
        assert "字面量" in diagnostic.suggestion

    def test_non_scalar_data_argument_is_rejected(self):
        """物化算子的要求类型是 Scalar；拿格网集合去编码必须报错。"""

        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    cells = geo.cellset_intersect(a, b)\n"
            "    return geo.height_band(x=cells, y=cells, height_min=0, "
            "height_max=300, level=20)\n",
            filename="badarg.py",
        )
        assert result.ok is False

    def test_too_many_positional_arguments_is_reported(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "band = geo.height_band(1, 2, 0, 300, 15, 0, 0, 0, 99)\n",
            filename="long.py",
        )
        assert any(d.code == "GEO_ARITY_MISMATCH" for d in result.diagnostics)


# --------------------------------------------------------------------------
# 物化算子不得被当成密态算子
# --------------------------------------------------------------------------


class TestPlannerTreatsMaterializeAsPlaintext:
    def _plan(self, source: str):
        result = parse_source(source, filename="v.py")
        assert result.ok, [str(d) for d in result.diagnostics]
        return plan_program(result.program), result

    def test_band_is_planned_as_plaintext(self):
        plan, _ = self._plan(MODULE_LEVEL_SOURCE)
        band_steps = plan.steps_for_op("HeightBand")
        assert len(band_steps) == 2
        for step in band_steps:
            assert step.backend == "Plaintext"
            assert step.effective_backend == "Plaintext"
            assert step.needs_crypto is False
            assert step.representation == "CompactCellSet"

    def test_set_operator_still_goes_to_psi(self):
        plan, _ = self._plan(MODULE_LEVEL_SOURCE)
        step = plan.step_for("verdict")
        assert step.operation == "Intersects"
        assert step.backend == "PSI"

    def test_secret_coordinates_keep_the_downstream_operator_in_crypto(self):
        """回归：物化算子的产出敏感度必须由输入推导，不能写死 PUBLIC。

        写死 PUBLIC 会让消费它的 PSI 算子被判成"可走明文"，那是一条真实泄漏。
        因此**必须把 x/y 本身标成 secret**——只标 zone 的话，"读 inputs 推导"
        与"写死 PUBLIC"两条路径会得出同一结论，用例就失去区分力。
        """

        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(x, y, zone):\n"
            "    band = geo.height_band(x=x, y=y, height_min=0, height_max=300, level=15)\n"
            "    return geo.intersects(band, zone)\n",
            filename="secret.py",
            sensitivities={"x": Sensitivity.SECRET, "y": Sensitivity.SECRET},
        )
        plan = plan_program(result.program)

        band_step = plan.steps_for_op("HeightBand")[0]
        assert band_step.sensitivity is Sensitivity.SECRET, (
            "物化算子的敏感度必须继承自输入坐标；写死 PUBLIC 会让下游误判可走明文"
        )

        downstream = plan.step_for("geo_intersects_1")
        assert downstream.needs_crypto is True
        assert downstream.effective_backend != "Plaintext"

    def test_secret_threshold_literal_is_still_public(self):
        """反向：字面量坐标不因物化算子而升级敏感度（避免把纯明文任务拖进密态）。"""

        result = parse_source(MODULE_LEVEL_SOURCE, filename="v.py")
        plan = plan_program(result.program)
        band_step = plan.steps_for_op("HeightBand")[0]
        assert band_step.sensitivity is not Sensitivity.SECRET

    def test_plaintext_band_does_not_make_the_program_look_public(self):
        """物化算子的 PUBLIC 常量不得把整条链路拖成明文。"""

        plan, _ = self._plan(MODULE_LEVEL_SOURCE)
        # 常量坐标 + 常量阈值 → 这一步确实可留明文；但结论必须来自
        # 输入敏感度推导，而不是来自物化算子写死的级别。
        band_step = plan.steps_for_op("HeightBand")[0]
        assert band_step.sensitivity is not Sensitivity.SECRET

    def test_every_registered_rule_survives_the_new_entry(self):
        """新增登记不得破坏既有规则的完整性。"""

        for op in registered_ops():
            rule = get_rule(op)
            assert rule.security_level in ("low", "medium", "high"), op
            for key in ("N_ct", "b", "d", "R"):
                assert key in rule.cost_profile, f"{op} 缺少 {key}"


# --------------------------------------------------------------------------
# 第 6 类失败：从真实用户源码触发
# --------------------------------------------------------------------------


class TestHeightLayerFailureFromSource:
    def test_overflowing_level_is_reported_with_four_elements(self):
        """此前这条检查读不到 params，从用户源码无法触发；现在必须能触发。"""

        result = parse_source(
            "from geo_privacy import geo\n"
            "band = geo.height_band(x=1, y=2, height_min=0, height_max=1000, level=23)\n"
            "verdict = geo.intersects(band, band)\n",
            filename="overflow.py",
        )
        assert result.ok, "解析层不该拦；容量约束由 validator 负责"
        diagnostics = validate_height_layer_capacity(result.program)
        assert len(diagnostics) == 1
        diagnostic = diagnostics[0]
        assert diagnostic.code == "HEIGHT_LAYER_UNSUPPORTED"
        assert diagnostic.severity == "error"
        assert diagnostic.location_str.startswith("overflow.py:2")
        # 四要素：位置 / 原因 / 替代 / 代价
        assert "超出 Z" in diagnostic.message
        assert "附录 B" in diagnostic.cause
        assert "L<=22" in diagnostic.suggestion
        assert diagnostic.estimated_cost is not None

    def test_fitting_level_produces_no_diagnostic(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "band = geo.height_band(x=1, y=2, height_min=0, height_max=1000, level=22)\n"
            "verdict = geo.intersects(band, band)\n",
            filename="fits.py",
        )
        assert validate_height_layer_capacity(result.program) == []

    def test_overflow_surfaces_through_validate_all(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "band = geo.height_band(x=1, y=2, height_min=0, height_max=2000, level=25)\n"
            "verdict = geo.intersects(band, band)\n",
            filename="overflow2.py",
        )
        report = validate_all(result.program, plan_program(result.program))
        assert "HEIGHT_LAYER_UNSUPPORTED" in {d.code for d in report.diagnostics}
        assert report.ok is False


# --------------------------------------------------------------------------
# 明文语义与物化结果一致（业务侧可脱离编译器自测）
# --------------------------------------------------------------------------


class TestPlaintextAgreement:
    def test_materialized_codes_equal_the_plaintext_reference(self):
        """编译器写进 IR 的层集合，必须与业务侧明文算出的完全一致。

        CellSet 内部存 frozenset（去重、无序），所以按集合比较；
        逐元素的顺序不是契约，集合相等才是。
        """

        from ir import height_band_codes

        expected = height_band_codes(21861, 27702, 15, 0, 20000)
        actual = geo.height_band(21861, 27702, 0, 20000, 15).codes
        assert set(actual) == set(expected)
        assert len(actual) == len(expected)  # 不因去重而少层

    def test_documented_example_still_answers_true(self):
        route = geo.height_band(
            x=21861, y=27702, height_min=0, height_max=20000, level=15
        )
        zone = geo.height_band(
            x=21861, y=27702, height_min=5000, height_max=9000, level=15
        )
        assert geo.intersects(route, zone) is True
        assert route.intersection(zone).cardinality() == 3

    def test_defaults_are_applied_when_omitted(self):
        """省略 toff/lt 时，方言默认值必须与 facade 默认值同效。"""

        explicit = geo.height_band(
            x=1, y=2, height_min=0, height_max=300, level=15, toff=0, lt=0
        )
        implied = geo.height_band(x=1, y=2, height_min=0, height_max=300, level=15)
        assert explicit.codes == implied.codes


# --------------------------------------------------------------------------
# 示例文件：端到端
# --------------------------------------------------------------------------


class TestAltitudeBandExample:
    """examples/altitude_band.py 是文档写法的可执行版本，必须一直跑得通。"""

    def test_example_compiles_without_errors(self):
        from geosecure import compile_file

        result = compile_file(example("altitude_band.py"))
        assert result.errors == []
        assert [op.op for op in result.program.operations] == [
            "HeightBand",
            "HeightBand",
            "Intersects",
            "Contains",
        ]

    def test_example_uses_module_level_materialization(self):
        """示例必须真的走模块顶层物化这条路（否则它测不到那个回归）。"""

        from geosecure import compile_file

        result = compile_file(example("altitude_band.py"))
        bands = [op for op in result.program.operations if op.op == "HeightBand"]
        assert len(bands) == 2
        # 模块顶层的物化结果名就是源码里的变量名
        assert {op.output_name for op in bands} == {"route", "control_zone"}

    def test_example_final_status_table_matches_the_documented_one(self):
        from geosecure import compile_file, operator_status_table

        result = compile_file(example("altitude_band.py"))
        table = operator_status_table(result)
        rows = {row["operation"]: row["status"] for row in result.operator_status}
        assert rows["HeightBand"] == "plaintext-local"
        # 有 PSI 环境时应为 verified；没有则允许停在 backend-direct
        from tests._helpers import has_psi

        assert rows["Intersects"] == ("verified" if has_psi() else "backend-direct")
        assert "plaintext-local" in table

    def test_example_plaintext_semantics_agree(self):
        """示例里写的两组高度带，明文判定必须是 冲突=True / 覆盖=False。"""

        route = geo.height_band(
            x=21861, y=27702, height_min=0, height_max=20000, level=15
        )
        zone = geo.height_band(
            x=21861, y=27702, height_min=5000, height_max=9000, level=15
        )
        assert geo.intersects(route, zone) is True
        assert geo.contains(zone, route) is False
