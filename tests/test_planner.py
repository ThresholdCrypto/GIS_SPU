"""planner 层测试：算子注册表、规划器输出五元组、敏感度对后端的影响。"""

from __future__ import annotations

import pytest

from frontend import parse_source
from ir import GeoOperation, GeoProgram, GeoType, Sensitivity
from planner import (
    OPERATOR_REGISTRY,
    Planner,
    PrivacyPlan,
    backend_capable_ops,
    get_rule,
    jax_capable_ops,
    plan_program,
    plan_table_rows,
    registered_ops,
)


# --------------------------------------------------------------------------
# 算子注册表：课题给定的默认规则
# --------------------------------------------------------------------------

EXPECTED_RULES = {
    "Intersects": ("CompactCellSet", "PSI"),
    "Contains": ("CompactCellSet", "PSI/MPC"),
    "DistanceLE": ("QuantizedVector", "MPC/SPU"),
    "WeightedSum": ("FixedPointVector", "SPU/MPC"),
    "TemporalOverlap": ("TimeInterval", "MPC/SPU"),
    "CellSetIntersect": ("CompactCellSet", "PSI"),
}


class TestOperatorRegistry:
    @pytest.mark.parametrize("op,expected", sorted(EXPECTED_RULES.items()))
    def test_default_rules(self, op, expected):
        rule = get_rule(op)
        assert rule is not None, f"算子 {op} 未登记"
        assert (rule.representation, rule.backend) == expected

    def test_all_six_operators_registered(self):
        assert set(registered_ops()) >= set(EXPECTED_RULES)

    def test_every_rule_has_security_level(self):
        for op, rule in OPERATOR_REGISTRY.items():
            assert rule.security_level in ("low", "medium", "high"), op

    def test_every_rule_has_four_quantity_cost(self):
        """每条规则都要给出四量（N_ct / b / d / R），否则代价无法对账。"""

        for op, rule in OPERATOR_REGISTRY.items():
            for key in ("N_ct", "b", "d", "R"):
                assert key in rule.cost_profile, f"{op} 缺少 {key}"

    def test_backends_parsing(self):
        assert get_rule("Contains").backends == ("PSI", "MPC")
        assert get_rule("Contains").primary_backend == "PSI"
        assert get_rule("Contains").fallback_backends == ("MPC",)
        assert get_rule("WeightedSum").primary_backend == "SPU"

    def test_jax_capable_ops(self):
        # 只有三个算子具备 JAX 代码生成能力
        assert set(jax_capable_ops()) == {"DistanceLE", "WeightedSum", "TemporalOverlap"}

    def test_psi_ops_have_no_jax_impl(self):
        for op in ("Intersects", "CellSetIntersect"):
            assert get_rule(op).has_jax_impl is False
        assert get_rule("Contains").has_jax_impl is False

    def test_backend_capable_ops(self):
        assert "Intersects" in backend_capable_ops("PSI")
        assert "DistanceLE" in backend_capable_ops("SPU")
        assert "DistanceLE" in backend_capable_ops("MPC")


# --------------------------------------------------------------------------
# 规划器输出
# --------------------------------------------------------------------------


class TestPlannerOutput:
    def _plan_for(self, source: str) -> PrivacyPlan:
        result = parse_source(source)
        assert result.ok, [str(d) for d in result.errors]
        return plan_program(result.program)

    def test_required_five_fields_present(self):
        """planner 输出必须含 operation / representation / backend / estimated_cost / security_level。"""

        plan = self._plan_for(
            "from geo_privacy import geo\n"
            "def f(route_A, NoFlyZone_B):\n"
            "    return geo.intersects(route_A, NoFlyZone_B)\n"
        )
        assert len(plan.steps) == 1
        step = plan.steps[0]
        assert step.operation == "Intersects"
        assert step.representation == "CompactCellSet"
        assert step.backend == "PSI"
        assert step.security_level == "high"
        assert "R" in step.estimated_cost

    def test_to_dict_has_exactly_the_five_keys(self):
        plan = self._plan_for(
            "from geo_privacy import geo\n"
            "def f(p1, p2, threshold):\n"
            "    return geo.distance_le(p1, p2, threshold)\n"
        )
        data = plan.steps[0].to_dict()
        for key in ("operation", "representation", "backend", "estimated_cost", "security_level"):
            assert key in data

    def test_minimal_constructor_signature(self):
        """用课题给的写法直接构造算子，也要能被规划。"""

        program = GeoProgram(name="sig")
        program.declare_input("route", sensitivity=Sensitivity.SENSITIVE)
        program.declare_input("no_fly_zone", sensitivity=Sensitivity.SENSITIVE)
        program.add_operation(
            GeoOperation(
                op="Intersects",
                inputs=["route", "no_fly_zone"],
                output_type="Relation",
            )
        )
        plan = plan_program(program)
        assert plan.steps[0].representation == "CompactCellSet"
        assert plan.steps[0].backend == "PSI"

    def test_summary_counts(self):
        # 注意用不同的变量：同一个变量不能既是向量又是格网集合，
        # 类型冲突会被 frontend 正确拦截（见 TestSensitivityPolicy 之外的类型检查用例）
        plan = self._plan_for(
            "from geo_privacy import geo\n"
            "def f(route_A, NoFlyZone_B, factors, weights):\n"
            "    s = geo.weighted_sum(factors, weights)\n"
            "    return geo.intersects(route_A, NoFlyZone_B)\n"
        )
        summary = plan.summary()
        assert summary["step_count"] == 2
        assert summary["crypto_step_count"] == 2
        assert "PSI" in summary["backends"]
        assert "SPU/MPC" in summary["backends"]

    def test_steps_for_op(self):
        plan = self._plan_for(
            "from geo_privacy import geo\n"
            "def f(a, b, c, d):\n"
            "    x = geo.intersects(a, b)\n"
            "    return geo.intersects(c, d)\n"
        )
        assert len(plan.steps_for_op("Intersects")) == 2


class TestSensitivityPolicy:
    def test_default_geometric_input_is_sensitive(self):
        program = GeoProgram()
        program.declare_input("a")
        assert program.entity_inputs["a"].effective_sensitivity is Sensitivity.SENSITIVE

    def test_scalar_default_is_public(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(p1, p2, threshold):\n"
            "    return geo.distance_le(p1, p2, threshold)\n"
        )
        assert result.program.entity_inputs["threshold"].effective_sensitivity is Sensitivity.PUBLIC
        assert result.program.entity_inputs["p1"].effective_sensitivity is Sensitivity.SENSITIVE

    def test_public_inputs_can_stay_plaintext(self):
        """输入全公开时状态标为 plaintext-ok，但方案仍保留密态路径备选。"""

        program = GeoProgram()
        program.declare_input("a", sensitivity=Sensitivity.PUBLIC)
        program.declare_input("b", sensitivity=Sensitivity.PUBLIC)
        program.add_operation(
            GeoOperation(op="Intersects", inputs=["a", "b"], output_type="Relation")
        )
        plan = plan_program(program)
        assert plan.steps[0].status == "plaintext-ok"
        assert plan.steps[0].effective_backend == "Plaintext"
        assert not plan.steps[0].needs_crypto

    def test_secret_input_raises_operation_sensitivity(self):
        program = GeoProgram()
        program.declare_input("a", sensitivity=Sensitivity.PUBLIC)
        program.declare_input("b", sensitivity=Sensitivity.SECRET)
        program.add_operation(
            GeoOperation(op="Intersects", inputs=["a", "b"], output_type="Relation")
        )
        plan = plan_program(program)
        assert plan.steps[0].sensitivity is Sensitivity.SECRET
        assert plan.steps[0].needs_crypto


class TestPlannerErrors:
    def test_unregistered_op_produces_backend_missing_error(self):
        """第 5 类失败：后端没有对应隐私算子。"""

        program = GeoProgram()
        program.declare_input("a")
        program.declare_input("b")
        program.add_operation(
            GeoOperation(op="NotARealOp", inputs=["a", "b"], output_type="Relation")
        )
        plan = plan_program(program)
        assert plan.has_errors
        codes = [d.code for d in plan.errors]
        assert "BACKEND_OP_MISSING" in codes

    def test_error_is_actionable(self):
        """错误必须带位置、原因、替代算子、预计代价。"""

        program = GeoProgram()
        program.declare_input("route")
        program.declare_input("zone")
        program.add_operation(
            GeoOperation(
                op="intersection",
                inputs=["route", "zone"],
                output_type="Relation",
                location={"file": "x.py", "line": 7, "col": 4},
            )
        )
        plan = plan_program(program)
        diagnostic = plan.errors[0]
        assert diagnostic.location_str == "x.py:7:4"
        assert diagnostic.cause
        assert diagnostic.suggested_op == "Intersects"
        assert diagnostic.estimated_cost is not None
        assert diagnostic.estimated_cost["backend"] == "PSI"

    def test_plan_does_not_mutate_program(self):
        """规划必须无副作用：不做任何用户代码/IR 改写。"""

        program = GeoProgram()
        program.declare_input("a")
        program.declare_input("b")
        operation = program.add_operation(
            GeoOperation(op="Intersects", inputs=["a", "b"], output_type="Relation")
        )
        before = program.to_dict()
        plan_program(program)
        assert program.to_dict() == before
        assert program.operations[0] is operation


# --------------------------------------------------------------------------
# 代价模型：位宽必须按元素数 K 实算
# --------------------------------------------------------------------------


class TestCostModel:
    """代价档案里的数值必须能被复算；不能与它自己的公式矛盾。

    旧版 WeightedSum 写死 b="16"，而 notes 里的公式是 8+8+ceil(log2 K)——
    K>1 时两者不等。报价低估累加余量，是向合作方交付时的实质风险。
    """

    def test_bit_width_formula_matches_its_own_string(self):
        import math
        import re

        from planner.registry import OPERATOR_REGISTRY, resolve_cost

        rule = OPERATOR_REGISTRY["WeightedSum"]
        for k in (1, 2, 3, 4, 5, 8, 24, 64, 256):
            profile = resolve_cost(rule, k=k)
            actual = int(re.match(r"(\d+)", str(profile["b"])).group(1))
            expected = 16 + math.ceil(math.log2(k))
            assert actual == expected, f"K={k}: b={actual} 与公式值 {expected} 不符"

    def test_bit_width_grows_with_k(self):
        from planner.registry import weighted_sum_bit_width

        widths = [weighted_sum_bit_width(k) for k in (1, 2, 4, 16, 64)]
        assert widths == sorted(widths)
        assert widths[0] != widths[-1]

    def test_bit_width_rejects_bad_k(self):
        from planner.registry import weighted_sum_bit_width

        with pytest.raises(ValueError):
            weighted_sum_bit_width(0)

    def test_planner_records_k_when_given(self):
        from planner.registry import resolve_cost

        profile = resolve_cost(OPERATOR_REGISTRY["WeightedSum"], k=24)
        assert profile["K"] == 24
        assert "24" in str(profile["N_ct"])

    def test_every_rule_reports_a_reproducible_bit_width(self):
        """每条规则的 b 都必须是非空字符串，并且 K 可解析时不与公式矛盾。"""
        from planner.registry import resolve_cost

        for op, rule in OPERATOR_REGISTRY.items():
            profile = resolve_cost(rule)
            assert str(profile.get("b", "")).strip() not in ("", "-"), op
            if rule.bit_width_formula is not None:
                assert profile.get("bit_width_rule"), op
                assert profile.get("K") is not None, op


class TestPlanTable:
    def test_table_rows_shape(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    return geo.intersects(a, b)\n"
        )
        plan = plan_program(result.program)
        rows = plan_table_rows(plan)
        assert len(rows) == 1
        assert rows[0][0] == "Intersects"
        assert rows[0][1] == "CompactCellSet"
        assert rows[0][2] == "PSI"