# -*- coding: utf-8 -*-
"""结果策略（§15）：业务层暴露 vs 协议内部泄漏，分开登记、拒绝伪造。"""

from __future__ import annotations

import pytest

from backends.psi_backend import run_psi_intersection
from backends.psi_backend.benchmark import generate_benchmark_sets
from backends.psi_backend.result_policy import (
    APPLICABLE_POLICIES_BY_OP,
    DEFAULT_POLICY_BY_OP,
    MPC_OP_REVEALS,
    MPC_POLICIES,
    OP_RESULT_FAMILY,
    PROTOCOL_LEAK_INTERSECTION_BODY,
    PROTOCOL_LEAK_OUTPUT_ONLY,
    RESULT_POLICIES,
    REVEAL_BOOLEAN,
    REVEAL_COUNT,
    REVEAL_INTERSECTION,
    REVEAL_TO_REGULATOR,
    REVEAL_VALUE,
    get_result_policy,
    resolve_result_policy,
)

from tests._helpers import has_psi

needs_psi = pytest.mark.skipif(
    not has_psi(), reason="当前环境不具备真实 PSI 执行能力(python3.11+spu+libgomp1)"
)


class TestPolicyTable:
    def test_defaults_by_op(self):
        assert DEFAULT_POLICY_BY_OP == {
            "Intersects": REVEAL_BOOLEAN,
            "Contains": REVEAL_BOOLEAN,
            "CellSetIntersect": REVEAL_INTERSECTION,
            "DistanceLE": REVEAL_BOOLEAN,
            "WeightedSum": REVEAL_VALUE,
            "TemporalOverlap": REVEAL_BOOLEAN,
        }

    def test_unknown_policy_rejected(self):
        with pytest.raises(ValueError, match="未知结果策略"):
            resolve_result_policy("Intersects", "REVEAL_NOPE")

    def test_regulator_mode_is_rejected_not_faked(self):
        with pytest.raises(ValueError, match="部署期"):
            resolve_result_policy("CellSetIntersect", REVEAL_TO_REGULATOR)

    def test_policy_not_applicable_to_op_is_rejected(self):
        with pytest.raises(ValueError, match="不适用于算子"):
            resolve_result_policy("Intersects", REVEAL_COUNT)
        with pytest.raises(ValueError, match="不适用于算子"):
            resolve_result_policy("Intersects", REVEAL_INTERSECTION)

    def test_every_policy_disclosure_separates_leak_from_exposure(self):
        """披露句必须同时说清两件事，且不得出现"只返回布尔值"这类断言。"""

        for name in RESULT_POLICIES:
            policy = get_result_policy(name)
            assert policy.protocol_leak == PROTOCOL_LEAK_INTERSECTION_BODY
            assert "交集本体" in policy.disclosure
            assert "不随策略改变" in policy.disclosure
            assert "只返回布尔值" in policy.disclosure  # 以“不得表述为…”的否定形式出现
        boolean = get_result_policy(REVEAL_BOOLEAN)
        assert boolean.business_value == "boolean"
        assert "业务层暴露 boolean" in boolean.disclosure

    def test_count_policy_metadata(self):
        policy = get_result_policy(REVEAL_COUNT)
        assert policy.business_value == "count"
        assert policy.executable is True
        assert REVEAL_COUNT in APPLICABLE_POLICIES_BY_OP["CellSetIntersect"]

    def test_unknown_op_rejected(self):
        with pytest.raises(ValueError, match="不在结果策略表"):
            resolve_result_policy("HeightBand")


class TestPolicyRuntime:
    @needs_psi
    def test_count_policy_exposes_cardinality_and_agrees(self):
        bset = generate_benchmark_sets(2 ** 10)
        run = run_psi_intersection(
            bset.left,
            bset.right,
            op="CellSetIntersect",
            protocol="PROTOCOL_ECDH",
            result_policy="REVEAL_COUNT",
            reference_fn=lambda left, right: tuple(sorted(set(left) & set(right))),
        )
        assert run.status == "ok", run.error
        assert run.value == 512
        assert run.agreement is True
        assert run.result_policy["policy"] == REVEAL_COUNT
        assert run.result_policy["protocol_leak"] == PROTOCOL_LEAK_INTERSECTION_BODY

    @needs_psi
    def test_regulator_policy_fails_before_protocol(self):
        run = run_psi_intersection(
            [1],
            [1],
            op="CellSetIntersect",
            protocol="PROTOCOL_ECDH",
            result_policy=REVEAL_TO_REGULATOR,
        )
        assert run.status == "error"
        assert "部署期" in run.error

    @needs_psi
    def test_default_policies_preserve_existing_semantics(self):
        intersects = run_psi_intersection(
            [1, 2], [2], op="Intersects", protocol="PROTOCOL_ECDH"
        )
        assert intersects.result_policy["policy"] == REVEAL_BOOLEAN
        assert intersects.value is True

        intersection = run_psi_intersection(
            [1, 2], [2], op="CellSetIntersect", protocol="PROTOCOL_ECDH"
        )
        assert intersection.result_policy["policy"] == REVEAL_INTERSECTION
        assert intersection.value == (2,)


class TestMpcPolicyTable:
    """Phase 10：结果策略补齐 MPC 侧（任务书 §十——不得默认广播原始结果）。"""

    def test_op_families_cover_every_registered_op(self):
        assert OP_RESULT_FAMILY == {
            "Intersects": "PSI",
            "Contains": "PSI",
            "CellSetIntersect": "PSI",
            "DistanceLE": "MPC",
            "WeightedSum": "MPC",
            "TemporalOverlap": "MPC",
        }
        assert set(OP_RESULT_FAMILY) == set(DEFAULT_POLICY_BY_OP)
        assert set(OP_RESULT_FAMILY) == set(APPLICABLE_POLICIES_BY_OP)

    def test_mpc_leak_sentence_is_output_only_and_never_claims_zero_leak(self):
        for name in MPC_POLICIES:
            policy = get_result_policy(name, "MPC")
            assert policy.protocol_leak == PROTOCOL_LEAK_OUTPUT_ONLY
            assert "输出面" in policy.disclosure
            assert "不得默认广播" in policy.disclosure
            # 不允许把 MPC 说成“零泄漏 / 无泄漏”
            assert "零泄漏" not in policy.disclosure
            assert "无泄漏" not in policy.disclosure

    def test_psi_and_mpc_leak_codes_differ(self):
        assert (
            get_result_policy(REVEAL_BOOLEAN, "PSI").protocol_leak
            == PROTOCOL_LEAK_INTERSECTION_BODY
        )
        assert (
            get_result_policy(REVEAL_BOOLEAN, "MPC").protocol_leak
            == PROTOCOL_LEAK_OUTPUT_ONLY
        )

    def test_value_policy_is_not_applicable_to_psi_ops(self):
        with pytest.raises(ValueError, match="不适用于算子"):
            resolve_result_policy("Intersects", REVEAL_VALUE)
        with pytest.raises(ValueError, match="不适用于算子"):
            resolve_result_policy("CellSetIntersect", REVEAL_VALUE)

    def test_boolean_policy_is_not_applicable_to_weighted_sum(self):
        with pytest.raises(ValueError, match="不适用于算子"):
            resolve_result_policy("WeightedSum", REVEAL_BOOLEAN)

    def test_regulator_mode_rejected_for_mpc_ops_too(self):
        for op in ("DistanceLE", "WeightedSum", "TemporalOverlap"):
            with pytest.raises(ValueError, match="部署期"):
                resolve_result_policy(op, REVEAL_TO_REGULATOR)

    def test_every_mpc_op_resolves_to_its_default(self):
        for op, name in (
            ("DistanceLE", REVEAL_BOOLEAN),
            ("WeightedSum", REVEAL_VALUE),
            ("TemporalOverlap", REVEAL_BOOLEAN),
        ):
            policy = resolve_result_policy(op)
            assert policy.name == name

    def test_mpc_reveals_registered_for_every_mpc_op(self):
        for op in ("DistanceLE", "WeightedSum", "TemporalOverlap"):
            text = MPC_OP_REVEALS[op]
            assert "输出方" in text
            assert "不进输出面" in text

    def test_unified_module_is_the_single_source(self):
        """旧路径只是再导出，不是第二份实现（结果策略独立于密码协议）。"""

        import backends.psi_backend.result_policy as shim
        import backends.result_policy as unified

        assert shim.ResultPolicy is unified.ResultPolicy
        assert shim.resolve_result_policy is unified.resolve_result_policy
        assert shim.DEFAULT_POLICY_BY_OP is unified.DEFAULT_POLICY_BY_OP


def _sum_fn(x):
    import jax.numpy as jnp

    return jnp.sum(x)


class TestMpcRuntimePolicy:
    """MPC 运行时：策略在**任何执行之前**解析并随结果登记。"""

    def test_policy_attached_regardless_of_execution_outcome(self):
        import numpy as np

        from backends.spu_backend import run_spu_simulation

        run = run_spu_simulation(
            _sum_fn,
            [np.asarray([1, 2, 3])],
            protocol="ABY3",
            field=64,
            op="WeightedSum",
        )
        # 环境不可用时 status=unavailable，但策略是编译期属性，必须已在
        assert run.status in ("ok", "unavailable"), run.describe()
        assert run.result_policy is not None
        assert run.result_policy["policy"] == REVEAL_VALUE
        assert run.result_policy["protocol_leak"] == PROTOCOL_LEAK_OUTPUT_ONLY
        assert "输出方" in run.reveals

    def test_op_none_keeps_results_policy_free(self):
        import numpy as np

        from backends.spu_backend import run_spu_simulation

        run = run_spu_simulation(_sum_fn, [np.asarray([1, 2, 3])])
        assert run.result_policy is None
        assert run.reveals == ""

    def test_unknown_op_fails_fast(self):
        import numpy as np

        from backends.spu_backend import run_spu_simulation

        with pytest.raises(ValueError, match="不在结果策略表"):
            run_spu_simulation(_sum_fn, [np.asarray([1])], op="NoSuchOp")

    def test_deployment_mode_fails_fast_before_execution(self):
        import numpy as np

        from backends.spu_backend import run_spu_simulation

        with pytest.raises(ValueError, match="部署期"):
            run_spu_simulation(
                _sum_fn,
                [np.asarray([1])],
                op="DistanceLE",
                result_policy=REVEAL_TO_REGULATOR,
            )
