# -*- coding: utf-8 -*-
"""结果策略（§15）：业务层暴露 vs 协议内部泄漏，分开登记、拒绝伪造。"""

from __future__ import annotations

import pytest

from backends.psi_backend import run_psi_intersection
from backends.psi_backend.benchmark import generate_benchmark_sets
from backends.psi_backend.result_policy import (
    APPLICABLE_POLICIES_BY_OP,
    DEFAULT_POLICY_BY_OP,
    PROTOCOL_LEAK_INTERSECTION_BODY,
    RESULT_POLICIES,
    REVEAL_BOOLEAN,
    REVEAL_COUNT,
    REVEAL_INTERSECTION,
    REVEAL_TO_REGULATOR,
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
