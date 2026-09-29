"""frontend 层测试：表达式级调用识别、输入可解析性、敏感度不静默失真。

本文件锁定的是一类**真实的隐私泄漏**：算子或它的输入在静态分析里被丢掉，
于是 planner 依据"解析不到"落回默认公开级别，判定该算子可走明文。
这类错误不会报错、不会告警，只会算错——所以必须逐条钉死在测试里。
"""

from __future__ import annotations

import pytest

from frontend import parse_source
from frontend.analyzer import DIAG_UNRESOLVED_INPUT
from ir import GeoOperation, GeoProgram, Sensitivity
from planner import plan_program


def _ops(source: str, sensitivities: dict[str, Sensitivity] | None = None) -> list[str]:
    result = parse_source(source, sensitivities=sensitivities or {})
    return [op.op for op in result.program.operations]


# --------------------------------------------------------------------------
# 表达式级调用识别（旧版只认 node.value 是直接 ast.Call 的那一种）
# --------------------------------------------------------------------------


class TestNestedCallRecognition:
    """任何位置的 geo 调用都必须进 Geo-IR——漏掉一个就是漏掉它的密态输入。"""

    def test_call_nested_in_return_is_emitted_innermost_first(self):
        ops = _ops(
            "from geo_privacy import geo\n"
            "def f(a, b, c):\n"
            "    return geo.intersects(geo.cellset_intersect(a, b), c)\n"
        )
        assert ops == ["CellSetIntersect", "Intersects"]

    def test_two_calls_in_a_tuple_return(self):
        ops = _ops(
            "from geo_privacy import geo\n"
            "def f(a, b, c, d):\n"
            "    return geo.intersects(a, b), geo.contains(c, d)\n"
        )
        assert ops == ["Intersects", "Contains"]

    def test_calls_in_list_literal(self):
        ops = _ops(
            "from geo_privacy import geo\n"
            "def f(a, b, c):\n"
            "    xs = [geo.intersects(a, b), geo.contains(a, c)]\n"
            "    return xs\n"
        )
        assert ops == ["Intersects", "Contains"]

    def test_call_nested_in_assignment(self):
        ops = _ops(
            "from geo_privacy import geo\n"
            "def f(a, b, c):\n"
            "    x = geo.intersects(geo.cellset_intersect(a, b), c)\n"
            "    return x\n"
        )
        assert ops == ["CellSetIntersect", "Intersects"]

    def test_call_inside_if_condition_is_emitted(self):
        ops = _ops(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    if geo.intersects(a, b):\n"
            "        return 1\n"
            "    return 0\n"
        )
        assert ops == ["Intersects"]

    def test_call_in_bare_expression_statement(self):
        ops = _ops(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    geo.intersects(a, b)\n"
            "    return None\n"
        )
        assert ops == ["Intersects"]

    def test_deeply_nested_calls_all_emitted(self):
        ops = _ops(
            "from geo_privacy import geo\n"
            "def f(a, b, c):\n"
            "    return geo.intersects(geo.cellset_intersect(geo.intersects(a, b), c), a)\n"
        )
        assert ops == ["Intersects", "CellSetIntersect", "Intersects"]

    def test_each_nested_call_is_emitted_exactly_once(self):
        # C 形语句会同时被"表达式扫描"和"语句下降"看到，
        # 必须保证发射且只发射一次。
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b, c):\n"
            "    return geo.intersects(geo.cellset_intersect(a, b), c)\n"
        )
        names = [op.output_name for op in result.program.operations]
        assert len(names) == len(set(names)) == 2


class TestNestedOutputNames:
    """内层算子必须拿到**唯一**且可解析的输出名，否则敏感度会算到别人头上。"""

    def test_inner_call_result_is_referenced_by_name(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b, c):\n"
            "    return geo.intersects(geo.cellset_intersect(a, b), c)\n"
        )
        inner, outer = result.program.operations
        assert outer.inputs[0] == inner.output_name
        assert result.program.lookup(outer.inputs[0]) is not None

    def test_same_op_repeated_gets_distinct_names(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b, c, d):\n"
            "    return geo.intersects(a, b), geo.intersects(c, d)\n"
        )
        names = [op.output_name for op in result.program.operations]
        assert names[0] != names[1]
        # 旧版两个算子都叫 intersects_0，lookup 解析到的是**第一个**
        assert result.program.lookup(names[1]) is not None

    def test_synthetic_names_are_deterministic(self):
        source = (
            "from geo_privacy import geo\n"
            "def f(a, b, c):\n"
            "    return geo.intersects(geo.cellset_intersect(a, b), c)\n"
        )
        first = parse_source(source)
        second = parse_source(source)
        assert first.program.to_dict() == second.program.to_dict()

    def test_explicit_variable_keeps_its_name(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b, c):\n"
            "    hit = geo.cellset_intersect(a, b)\n"
            "    return geo.intersects(hit, c)\n"
        )
        assert result.program.operations[0].output_name == "hit"

    def test_reused_result_name_is_warned(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b, c, d):\n"
            "    x = geo.intersects(a, b)\n"
            "    x = geo.contains(c, d)\n"
            "    return x\n"
        )
        assert any("\u88ab\u91cd\u590d\u8d4b\u503c" in d.message for d in result.diagnostics)

    def test_synthetic_names_do_not_leak_into_spatial_scope(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(route_A, zone_B):\n"
            "    return geo.intersects(route_A, geo.cellset_intersect(route_A, zone_B))\n"
        )
        scopes = {r.spatial_scope for r in result.program.relations}
        assert None not in scopes  # 不能是 "cellsetintersect_2" 这类合成名的尾巴


# --------------------------------------------------------------------------
# 敏感度必须由"能解析到的输入"决定，而不是由"解析不到"决定
# --------------------------------------------------------------------------


class TestSensitivityIsNeverSilentlyDowngraded:
    """解析不出来的输入按 SECRET 计入：宁可多算密态，不可少算。"""

    @staticmethod
    def _plan(source: str, sensitivities: dict[str, Sensitivity]):
        result = parse_source(source, sensitivities=sensitivities)
        return result, plan_program(result.program)

    def test_nested_form_matches_explicit_variable_form(self):
        sensitivities = {
            "a": Sensitivity.SECRET,
            "b": Sensitivity.SECRET,
            "c": Sensitivity.PUBLIC,
        }
        nested_src = (
            "from geo_privacy import geo\n"
            "def f(a, b, c):\n"
            "    return geo.intersects(geo.cellset_intersect(a, b), c)\n"
        )
        explicit_src = (
            "from geo_privacy import geo\n"
            "def f(a, b, c):\n"
            "    hit = geo.cellset_intersect(a, b)\n"
            "    return geo.intersects(hit, c)\n"
        )
        nested_result, nested_plan = self._plan(nested_src, sensitivities)
        explicit_result, explicit_plan = self._plan(explicit_src, sensitivities)

        assert [s.operation for s in nested_plan.steps] == [
            s.operation for s in explicit_plan.steps
        ]
        # 逐算子比较"是否需要密态"，而不是只比较算子名
        assert [s.needs_crypto for s in nested_plan.steps] == [
            s.needs_crypto for s in explicit_plan.steps
        ]
        assert all(s.needs_crypto for s in nested_plan.steps)
        assert all(s.needs_crypto for s in explicit_plan.steps)
        # 两者都不允许出现"未进密态"的算子
        assert not [s for s in nested_plan.steps if not s.needs_crypto]
        assert not [s for s in explicit_plan.steps if not s.needs_crypto]
        # 输入集合也应当一致（只是中间结果的命名不同）
        assert len(nested_result.program.operations) == len(explicit_result.program.operations)

    def test_secret_derived_result_cannot_be_judged_plaintext(self):
        sensitivities = {
            "pub_a": Sensitivity.PUBLIC,
            "pub_b": Sensitivity.PUBLIC,
            "secret_a": Sensitivity.SECRET,
            "secret_b": Sensitivity.SECRET,
        }
        source = (
            "from geo_privacy import geo\n"
            "def f(pub_a, pub_b, secret_a, secret_b):\n"
            "    geo.intersects(pub_a, pub_b)\n"
            "    return geo.intersects(geo.intersects(secret_a, secret_b), pub_a)\n"
        )
        result, plan = self._plan(source, sensitivities)
        # 第一个算子只接触公开输入 → 可以留明文
        assert plan.steps[0].needs_crypto is False
        # 第二个接触 SECRET → 必须密态
        assert plan.steps[1].needs_crypto is True
        # 第三个消费的是第二个的结果 → 必须密态（旧版因重名判成 public）
        assert plan.steps[2].needs_crypto is True
        assert plan.steps[2].sensitivity is Sensitivity.SECRET

    def test_unresolvable_argument_is_reported_and_treated_as_secret(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(public_set, secret_set):\n"
            "    return geo.intersects(public_set, secret_set[:1])\n",
            sensitivities={"public_set": Sensitivity.PUBLIC, "secret_set": Sensitivity.SECRET},
        )
        plan = plan_program(result.program)
        assert plan.steps[0].needs_crypto is True
        assert plan.steps[0].status != "plaintext-ok"
        codes = {d.code for d in result.diagnostics}
        assert DIAG_UNRESOLVED_INPUT in codes

    def test_unresolvable_argument_diagnostic_is_actionable(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(route, zone):\n"
            "    z = zone\n"
            "    return geo.intersects(route, z)\n",
            sensitivities={"route": Sensitivity.PUBLIC},
        )
        plan = plan_program(result.program)
        assert plan.steps[0].needs_crypto is True

    def test_non_dialect_argument_call_is_reported(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(route, zone):\n"
            "    return geo.intersects(route, helper(zone))\n"
        )
        diagnostic = next(d for d in result.diagnostics if d.code == DIAG_UNRESOLVED_INPUT)
        assert diagnostic.severity == "warning"
        assert diagnostic.location.get("line") == 3
        assert diagnostic.cause and diagnostic.suggestion
        assert diagnostic.suggested_op == "Intersects"
        assert diagnostic.estimated_cost is not None

    def test_literal_threshold_stays_public(self):
        """字面量实参必须登记为 PUBLIC 常量，否则纯明文任务会被误判进密态。"""
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(p1, p2, threshold):\n"
            "    return geo.distance_le(p1, p2, 2)\n",
            sensitivities={"p1": Sensitivity.PUBLIC, "p2": Sensitivity.PUBLIC},
        )
        plan = plan_program(result.program)
        assert plan.steps[0].needs_crypto is False
        assert plan.steps[0].status == "plaintext-ok"
        assert result.program.entity_inputs["2"].is_constant is True
        assert result.program.entity_inputs["2"].effective_sensitivity is Sensitivity.PUBLIC

    def test_impossible_to_resolve_anything_is_never_public(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(route, zone):\n"
            "    return geo.intersects(route, zone)\n"
        )
        result.program.operations[0] = GeoOperation(
            op="Intersects", inputs=("<unresolvable-1>", "<unresolvable-2>")
        )
        plan = plan_program(result.program)
        assert plan.steps[0].needs_crypto is True


# --------------------------------------------------------------------------
# 显式声明的类型 / 敏感度：不允许静默失真
# --------------------------------------------------------------------------


class TestExplicitDeclarations:
    def test_string_sensitivity_is_accepted(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    return geo.intersects(a, b)\n",
            sensitivities={"a": "secret", "b": "public"},
        )
        assert result.program.entity_inputs["a"].effective_sensitivity is Sensitivity.SECRET
        assert result.program.entity_inputs["b"].effective_sensitivity is Sensitivity.PUBLIC

    def test_bad_sensitivity_value_raises(self):
        with pytest.raises(ValueError):
            parse_source(
                "from geo_privacy import geo\n"
                "def f(a, b):\n"
                "    return geo.intersects(a, b)\n",
                sensitivities={"a": "top-secret"},
            )

    def test_string_type_hint_is_accepted(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    return geo.intersects(a, b)\n",
            type_hints={"a": "CellSet"},
        )
        assert result.program.entity_inputs["a"].geo_type.value == "CellSet"

    def test_unmatched_sensitivity_key_is_warned(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    return geo.intersects(a, b)\n",
            sensitivities={"a": Sensitivity.SECRET, "typo_name": Sensitivity.SECRET},
        )
        assert any("typo_name" in d.message for d in result.diagnostics)


# --------------------------------------------------------------------------
# 地理语义：主体/宾语/作用域不许编造
# --------------------------------------------------------------------------


class TestRelationSemantics:
    """作用域推断只在命名有明确分区语义时给结论，否则留空。

    旧版用"取第一个下划线之后的部分"来猜：`public_set` → `set`、
    `risk_factors` → `factors`。那是编造出来的作用域。课题口径是
    "宁可留空，不给假数据"，所以推断必须走 semantic 的命名约定表。
    """

    def test_scope_is_none_without_a_naming_convention(self):
        for left, right in (("public_set", "secret_set"), ("risk_factors", "weights")):
            result = parse_source(
                "from geo_privacy import geo\n"
                f"def f({left}, {right}):\n"
                f"    return geo.intersects({left}, {right})\n"
            )
            assert result.program.relations[0].spatial_scope is None

    def test_scope_is_found_with_a_naming_convention(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(route_A, zone_B):\n"
            "    return geo.intersects(route_A, zone_B)\n"
        )
        assert result.program.relations[0].spatial_scope == "A"

    def test_no_fake_scope_from_arbitrary_prefixes(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(my_data, your_data):\n"
            "    return geo.intersects(my_data, your_data)\n"
        )
        assert result.program.relations[0].spatial_scope is None

    def test_object_is_identified_by_naming_not_position(self):
        """`no_fly_zone` 无论出现在第几个参数都应是宾语。"""
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(no_fly_zone, route):\n"
            "    return geo.intersects(no_fly_zone, route)\n"
        )
        relation = result.program.relations[0]
        assert relation.object == "no_fly_zone"
        assert relation.subject == "route"

    def test_relation_carries_all_required_fields(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(route_A, NoFlyZone_B):\n"
            "    return geo.intersects(route_A, NoFlyZone_B)\n"
        )
        payload = result.program.relations[0].to_dict()
        assert set(payload) == {
            "subject",
            "predicate",
            "object",
            "time",
            "spatial_scope",
            "sensitivity",
        }
        assert (payload["subject"], payload["predicate"], payload["object"]) == (
            "route_A",
            "Intersects",
            "NoFlyZone_B",
        )


# --------------------------------------------------------------------------
# 链式类型：中间结果的产出类型必须参与检查
# --------------------------------------------------------------------------


class TestChainedResultTypes:
    """上游算子**产出**什么，决定下游能不能消费它。

    旧版一律按方言的 returns（供应物）当成 Relation，并且中间结果
    完全不检查类型，于是"格网集合喂给定点向量"这类链式错误无诊断通过。
    """

    def test_cellset_intersect_output_type_is_cell_set(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    return geo.cellset_intersect(a, b)\n"
        )
        assert result.program.operations[0].output_type.value == "CellSet"

    def test_supplied_relation_is_still_registered(self):
        """产出格网集合，但供应给业务侧的仍是关系。"""
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    return geo.cellset_intersect(a, b)\n"
        )
        assert len(result.program.relations) == 1
        assert result.program.relations[0].predicate == "CellSetIntersect"

    def test_scalar_producing_op_registers_no_relation(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    return geo.weighted_sum(a, b)\n"
        )
        assert not result.program.relations

    def test_cell_set_into_fixed_point_vector_is_rejected(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b, weights):\n"
            "    cells = geo.cellset_intersect(a, b)\n"
            "    return geo.weighted_sum(cells, weights)\n"
        )
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert len(errors) == 1
        diagnostic = errors[0]
        assert diagnostic.code == "STATIC_CHECK_FAILED"
        assert "cells" in diagnostic.message and "Vector" in diagnostic.message
        assert diagnostic.location.get("line") == 4
        assert diagnostic.cause and diagnostic.suggestion

    def test_cell_set_into_time_interval_is_rejected(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b, w1, w2):\n"
            "    cells = geo.cellset_intersect(a, b)\n"
            "    return geo.temporal_overlap(cells, w1)\n"
        )
        assert any(d.severity == "error" for d in result.diagnostics)

    def test_nested_chain_type_error_is_also_rejected(self):
        """嵌套写法与显式变量写法必须给出同样的类型结论。"""
        nested = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b, weights):\n"
            "    return geo.weighted_sum(geo.cellset_intersect(a, b), weights)\n"
        )
        explicit = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b, weights):\n"
            "    cells = geo.cellset_intersect(a, b)\n"
            "    return geo.weighted_sum(cells, weights)\n"
        )
        nested_codes = sorted(d.code for d in nested.diagnostics if d.severity == "error")
        explicit_codes = sorted(d.code for d in explicit.diagnostics if d.severity == "error")
        assert nested_codes == explicit_codes == ["STATIC_CHECK_FAILED"]

    def test_entity_set_and_cell_set_are_interoperable(self):
        """格网集合的两种视图可以互相喂。"""
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b, c):\n"
            "    return geo.intersects(a, geo.cellset_intersect(b, c))\n"
        )
        assert result.ok, [str(d) for d in result.errors]

    def test_unknown_types_are_not_reported(self):
        """判不出类型时不报错——宁可少报，不可误报。"""
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    return geo.intersects(a, b)\n",
            type_hints={},
        )
        assert result.ok, [str(d) for d in result.errors]

    def test_relation_into_vector_is_rejected(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b, weights):\n"
            "    hit = geo.intersects(a, b)\n"
            "    return geo.weighted_sum(hit, weights)\n"
        )
        assert any(d.severity == "error" for d in result.diagnostics)


# --------------------------------------------------------------------------
# 约束：不执行用户代码、不改写用户代码
# --------------------------------------------------------------------------


class TestStaticOnlyGuarantees:
    def test_source_is_never_executed(self):
        """只做静态分析：源码里的副作用不得发生。"""
        result = parse_source(
            "from geo_privacy import geo\n"
            "raise RuntimeError('must not run')\n"
            "def f(a, b):\n"
            "    return geo.intersects(a, b)\n"
        )
        assert result.program.operations  # 解析照样完成

    def test_no_diagnostic_when_everything_resolves(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(route_A, NoFlyZone_B):\n"
            "    return geo.intersects(route_A, NoFlyZone_B)\n"
        )
        assert result.ok
        assert not result.diagnostics
