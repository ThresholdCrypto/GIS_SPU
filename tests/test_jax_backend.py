"""JAX 后端测试：代码生成、可追踪性、原语纯净度、与明文一致性。"""

from __future__ import annotations

import numpy as np
import pytest

from backends.jax_backend import (
    GENERATORS,
    TOLERANCES,
    check_traceable,
    extract_hlo_ops,
    generate_distance_le,
    generate_temporal_overlap,
    generate_weighted_sum,
    load_generated_function,
    lower_to_hlo,
    lower_to_hlo_text,
    render_module,
    run_jax_jit,
    static_check_source,
)
from backends.plain import run_plain
from frontend import parse_source
from planner import plan_program
from backends.jax_backend import generate_for_plan

pytest.importorskip("jax", reason="JAX 后端测试需要 jax")


# --------------------------------------------------------------------------
# 生成器
# --------------------------------------------------------------------------


class TestGenerators:
    def test_three_required_operators_have_generators(self):
        assert set(GENERATORS) == {"DistanceLE", "WeightedSum", "TemporalOverlap"}

    def test_every_generator_declares_tolerance(self):
        for op in GENERATORS:
            assert op in TOLERANCES, f"{op} 未声明容差"

    @pytest.mark.parametrize("op", sorted(GENERATORS))
    def test_generated_source_is_clean(self, op):
        function = GENERATORS[op](f"fn_{op}")
        problems = static_check_source(function.source)
        assert not problems, problems

    @pytest.mark.parametrize("op", sorted(GENERATORS))
    def test_generated_source_only_uses_jax_numpy(self, op):
        source = GENERATORS[op](f"fn_{op}").source
        assert "import jax.numpy as jnp" in source
        for banned in ("import numpy", "import spu", "import mpc", "from spu", "from secretflow"):
            assert banned not in source

    @pytest.mark.parametrize("op", sorted(GENERATORS))
    def test_generated_source_has_no_python_control_flow(self, op):
        """动态控制流无法编译——生成代码里不能出现 Python 级分支。"""

        import ast

        source = GENERATORS[op](f"fn_{op}").source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for statement in node.body:
                    assert not isinstance(statement, (ast.If, ast.For, ast.While)), (
                        f"{op} 生成了 Python 级控制流"
                    )

    def test_distance_le_avoids_sqrt(self):
        """DistanceLE 必须用平方比较，避免 sqrt 电路。

        注意：docstring/注释里可以提到 sqrt（说明为什么不用），
        这里检查的是**是否真的调用了** sqrt 原语。
        """

        import ast

        function = generate_distance_le("f")
        source = function.source
        assert "jnp.square" in source

        # 只检查调用表达式，不受注释影响
        tree = ast.parse(source)
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "sqrt" not in called
        assert "rsqrt" not in called

    def test_weighted_sum_scale_is_a_parameter_not_closure(self):
        """scale 必须是参数：闭包捕获会破坏 jax.jit 追踪。"""

        function = generate_weighted_sum("f")
        assert "scale" in function.inputs

    def test_temporal_overlap_uses_broadcast_not_loops(self):
        source = generate_temporal_overlap("f").source
        assert "[None, :]" in source or "[None, :]" in source
        assert "for " not in source


# --------------------------------------------------------------------------
# 可追踪性：硬门槛
# --------------------------------------------------------------------------


class TestTraceability:
    def test_distance_le_traceable(self):
        function = generate_distance_le("f")
        fn = load_generated_function(function.source, function.name)
        check = check_traceable(fn, (np.array([1, 2, 3]), np.array([1, 3, 3]), np.int32(2)))
        assert check.traceable, check.error

    def test_weighted_sum_traceable(self):
        function = generate_weighted_sum("f")
        fn = load_generated_function(function.source, function.name)
        check = check_traceable(
            fn, (np.array([1, 2, 3]), np.array([1, 1, 1]), np.int32(1))
        )
        assert check.traceable, check.error

    def test_temporal_overlap_traceable(self):
        function = generate_temporal_overlap("f")
        fn = load_generated_function(function.source, function.name)
        check = check_traceable(
            fn,
            (np.array([0, 4]), np.array([3, 2]), np.array([4]), np.array([3])),
        )
        assert check.traceable, check.error

    def test_data_dependent_control_flow_is_rejected(self):
        """依赖**密态数据**的 Python 分支必须被判为不可追踪。

        这是第 4 类失败的判定依据：分支条件要读张量的具体值，
        而密态下没有具体值可读，编译期无法展开为静态计算图。
        """

        def bad_fn(x):
            import jax.numpy as jnp

            if jnp.sum(x) > 2:  # 条件依赖数据取值
                return x
            return -x

        check = check_traceable(bad_fn, (np.array([1.0, 2.0, 3.0]),))
        assert not check.traceable
        assert check.error_type == "TracerBoolConversionError"

    def test_shape_dependent_branch_is_statically_traceable(self):
        """依赖**静态形状**的分支是可追踪的——形状在追踪期是已知常量。

        编译器不应把这类写法误报为错误。
        """

        def ok_fn(x):
            if x.shape[0] > 2:  # 形状是静态信息
                return x.sum()
            return x[0]

        check = check_traceable(ok_fn, (np.array([1.0, 2.0, 3.0]),))
        assert check.traceable

    def test_concretization_is_rejected(self):
        """把追踪值当具体值取用（.item()）必须被判为不可追踪。"""

        def bad_fn(x):
            return x.item()

        check = check_traceable(bad_fn, (np.array([1.0, 2.0]),))
        assert not check.traceable
        assert check.error_type in ("ConcretizationTypeError", "TracerBoolConversionError")

    def test_numpy_mixed_with_jax_is_rejected(self):
        """把追踪值交给 numpy 数组构造会打断追踪，应被判为不可追踪。"""

        def bad_fn(x):
            import numpy as onp

            return onp.asarray(x) + 1  # 追踪值无法转成具体 numpy 数组

        check = check_traceable(bad_fn, (np.array([1.0, 2.0]),))
        assert not check.traceable
        assert check.error_type == "TracerArrayConversionError"


# --------------------------------------------------------------------------
# HLO 与原语核对
# --------------------------------------------------------------------------


class TestHloAndPrimitives:
    def test_hlo_lowering_works_for_all_generated_ops(self):
        cases = {
            "DistanceLE": ((np.array([1, 2, 3]), np.array([1, 3, 3]), np.int32(2)), generate_distance_le),
            "WeightedSum": ((np.array([1, 2, 3]), np.array([1, 1, 1]), np.int32(1)), generate_weighted_sum),
            "TemporalOverlap": (
                (np.array([0, 4]), np.array([3, 2]), np.array([4]), np.array([3])),
                generate_temporal_overlap,
            ),
        }
        for op, (args, generator) in cases.items():
            function = generator(f"f_{op}")
            fn = load_generated_function(function.source, function.name)
            hlo, error = lower_to_hlo(fn, args)
            assert error is None, f"{op}: {error}"
            assert hlo and len(hlo) > 0

    def test_measured_primitives_match_registry(self):
        """实测 HLO 原语必须与 capability 模块登记的清单一致。

        这条测试的作用是防止"登记的原语是猜的"——一旦生成代码变化，
        它会在测试里立刻暴露，而不是等到 SPU 编译才失败。
        """

        from backends.spu_backend import OP_HLO_PRIMITIVES

        cases = {
            "DistanceLE": ((np.array([1, 2, 3]), np.array([1, 3, 3]), np.int32(2)), generate_distance_le),
            "WeightedSum": ((np.array([1, 2, 3]), np.array([1, 1, 1]), np.int32(1)), generate_weighted_sum),
            "TemporalOverlap": (
                (np.array([0, 4]), np.array([3, 2]), np.array([4]), np.array([3])),
                generate_temporal_overlap,
            ),
        }
        for op, (args, generator) in cases.items():
            function = generator(f"f_{op}")
            fn = load_generated_function(function.source, function.name)
            text, error = lower_to_hlo_text(fn, args)
            assert error is None, f"{op}: {error}"
            measured = set(extract_hlo_ops(text))
            registered = set(OP_HLO_PRIMITIVES[op])
            assert measured, f"{op} 未提取到任何 HLO 原语（提取逻辑可能失效）"
            # 登记清单应是实测的超集（登记可以更保守，但不能漏掉实际用到的）
            missing = measured - registered
            assert not missing, f"{op} 实际用到但未登记的原语: {sorted(missing)}"

    def test_all_registered_primitives_are_in_whitelist(self):
        from backends.spu_backend import OP_HLO_PRIMITIVES, SPU_ADAPTED_HLO_PRIMITIVES

        whitelist = set(SPU_ADAPTED_HLO_PRIMITIVES)
        for op, primitives in OP_HLO_PRIMITIVES.items():
            unknown = set(primitives) - whitelist
            assert not unknown, f"{op} 用到未在 SPU 适配白名单中的原语: {sorted(unknown)}"


# --------------------------------------------------------------------------
# 与明文对拍
# --------------------------------------------------------------------------


class TestJaxMatchesPlain:
    def test_distance_le_matches_plain_true_case(self):
        function = generate_distance_le("f")
        fn = load_generated_function(function.source, function.name)
        left, right, threshold = np.array([1, 2, 3]), np.array([1, 3, 3]), 2
        jax_value = bool(run_jax_jit(fn, (left, right, np.int32(threshold))))
        plain_value = run_plain(
            "DistanceLE", list(left), list(right), int(threshold)
        ).value
        assert jax_value == plain_value == True  # noqa: E712

    def test_distance_le_matches_plain_false_case(self):
        function = generate_distance_le("f")
        fn = load_generated_function(function.source, function.name)
        left, right, threshold = np.array([0, 0]), np.array([10, 10]), 2
        jax_value = bool(run_jax_jit(fn, (left, right, np.int32(threshold))))
        plain_value = run_plain("DistanceLE", list(left), list(right), int(threshold)).value
        assert jax_value == plain_value == False  # noqa: E712

    def test_weighted_sum_matches_plain(self):
        function = generate_weighted_sum("f")
        fn = load_generated_function(function.source, function.name)
        values, weights, scale = np.array([10, 20, 30]), np.array([1, 2, 1]), 1
        jax_value = int(run_jax_jit(fn, (values, weights, np.int32(scale))))
        plain_value = run_plain("WeightedSum", list(values), list(weights), int(scale)).value
        assert jax_value == plain_value == 80

    def test_weighted_sum_respects_scale(self):
        function = generate_weighted_sum("f")
        fn = load_generated_function(function.source, function.name)
        values, weights = np.array([100, 200]), np.array([1, 1])
        assert int(run_jax_jit(fn, (values, weights, np.int32(10)))) == 30

    @pytest.mark.parametrize(
        "left_nodes,right_nodes,expected",
        [
            (((0, 3),), ((4, 3),), True),      # 部分重叠
            (((0, 3),), ((8, 3),), False),     # 相邻不重叠
            (((0, 3),), ((2, 1),), True),      # 嵌套
            (((0, 3),), ((10, 3),), False),    # 远离
        ],
    )
    def test_temporal_overlap_matches_plain(self, left_nodes, right_nodes, expected):
        function = generate_temporal_overlap("f")
        fn = load_generated_function(function.source, function.name)
        arrays = (
            np.array([n[0] for n in left_nodes], np.int32),
            np.array([n[1] for n in left_nodes], np.int32),
            np.array([n[0] for n in right_nodes], np.int32),
            np.array([n[1] for n in right_nodes], np.int32),
        )
        jax_value = bool(run_jax_jit(fn, arrays))
        plain_value = run_plain("TemporalOverlap", left_nodes, right_nodes).value
        assert jax_value == plain_value == expected


# --------------------------------------------------------------------------
# 容差
# --------------------------------------------------------------------------


class TestTolerance:
    def test_integer_paths_are_exact(self):
        """整数路径下三份实现应完全一致，不需要容差。"""

        for op in ("DistanceLE", "WeightedSum", "TemporalOverlap"):
            assert TOLERANCES[op] == 0.0

    def test_float_inputs_introduce_only_representation_error(self):
        """若上游把输入浮点化，误差应只来自表示精度。"""

        function = generate_distance_le("f")
        fn = load_generated_function(function.source, function.name)
        left, right = np.array([1.0, 2.0, 3.0], np.float32), np.array([1.0, 3.0, 3.0], np.float32)
        value = run_jax_jit(fn, (left, right, np.float32(2.0)))
        assert bool(value) is True
        # 距离平方 = 1.0，阈值平方 = 4.0，边界明确，浮点下无歧义
        assert abs(float(np.sum((left - right) ** 2)) - 1.0) < 1e-4


# --------------------------------------------------------------------------
# 模块渲染
# --------------------------------------------------------------------------


class TestModuleRendering:
    def test_render_module_contains_all_functions(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b, factors, weights):\n"
            "    s = geo.weighted_sum(factors, weights)\n"
            "    return geo.intersects(a, b)\n"
        )
        plan = plan_program(result.program)
        generation = generate_for_plan(plan)
        module = render_module(generation, source_plan=plan)

        assert "import jax.numpy as jnp" in module
        assert "geo_weightedsum_0" in module
        # PSI 算子不生成代码，但要说明原因
        assert "Intersects" in module
        assert "不生成 JAX 代码" in module

    def test_rendered_module_is_importable_python(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(p1, p2, t):\n"
            "    return geo.distance_le(p1, p2, t)\n"
        )
        plan = plan_program(result.program)
        module = render_module(generate_for_plan(plan), source_plan=plan)
        compile(module, "<generated>", "exec")  # 语法必须合法

    def test_skipped_ops_are_recorded_with_reason(self):
        result = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    return geo.intersects(a, b)\n"
        )
        plan = plan_program(result.program)
        generation = generate_for_plan(plan)
        assert not generation.functions
        assert len(generation.skipped) == 1
        assert generation.skipped[0]["operation"] == "Intersects"
        assert "jax.numpy" in generation.skipped[0]["reason"]