# -*- coding: utf-8 -*-
"""统一协议元数据（§12）与结果语义（§11）：单一来源 + Planner 交叉一致。"""

from __future__ import annotations

import pytest

from backends.psi_backend import (
    PROTOCOL_SPECS,
    PSI_CANDIDATE_OPS,
    PSI_CURVE_RELATION,
    PSI_DEFAULT_PROTOCOL,
    PSI_PROTOCOL_NAMES,
    PSI_PROTOCOL_WORLD_SIZE,
    PSI_PROTOCOLS,
    PSI_PROTOCOLS_WITH_NOISE,
    RESULT_SEMANTICS,
    RESULT_SEMANTICS_APPROXIMATE,
    RESULT_SEMANTICS_EXACT,
    RESULT_SEMANTICS_NOISY,
    candidate_protocols_for,
    protocol_is_exact,
    protocol_result_semantics,
    psi_protocol_spec,
)
from backends.psi_backend.protocol_registry import PsiProtocolSpec
from planner.registry import (
    MPC_PROTOCOL_CANDIDATES,
    MPC_RULE_DEFAULT_PROTOCOL,
    OPERATOR_REGISTRY,
    PSI_PROTOCOL_CANDIDATES,
    PSI_RULE_DEFAULT_PROTOCOL,
)
from backends.spu_backend import SPU_PROTOCOLS


class TestRegistryIsSingleSource:
    def test_capability_tables_are_derived_from_registry(self):
        assert PSI_PROTOCOLS == PSI_PROTOCOL_NAMES == tuple(PROTOCOL_SPECS)
        assert dict(PSI_PROTOCOL_WORLD_SIZE) == {
            name: spec.world_size for name, spec in PROTOCOL_SPECS.items()
        }
        assert dict(PSI_CURVE_RELATION) == {
            name: spec.curve_relation for name, spec in PROTOCOL_SPECS.items()
        }
        assert PSI_PROTOCOLS_WITH_NOISE == ("PROTOCOL_DP",)

    def test_spec_shape_matches_task_12(self):
        spec = psi_protocol_spec("RR22")
        assert spec.name == "PROTOCOL_RR22"
        assert spec.world_size == 2
        assert spec.exact is True
        assert spec.curve_relation == "ignored"
        assert {"receiver_rank", "broadcast_result", "low_comm_mode"} <= set(
            spec.params_schema
        )
        assert spec.candidate_for == PSI_CANDIDATE_OPS
        data = spec.to_dict()
        assert data["result_semantics"] == "exact"

    def test_three_party_protocol_is_registered_but_not_a_candidate(self):
        spec = psi_protocol_spec("ECDH_3PC")
        assert spec.world_size == 3
        assert spec.candidate_for == ()

    def test_unknown_protocol_is_rejected(self):
        with pytest.raises(ValueError, match="未知 PSI 协议"):
            psi_protocol_spec("PROTOCOL_NOPE")

    def test_invalid_spec_fields_are_rejected_at_construction(self):
        with pytest.raises(ValueError, match="curve_relation"):
            PsiProtocolSpec(
                name="X",
                world_size=2,
                exact=True,
                curve_relation="nope",
                params_schema={},
                candidate_for=(),
            )
        with pytest.raises(ValueError, match="semantics"):
            PsiProtocolSpec(
                name="X",
                world_size=2,
                exact=True,
                curve_relation="required",
                params_schema={},
                candidate_for=(),
                semantics="fuzzy",
            )

    def test_explicit_semantics_field_wins(self):
        spec = PsiProtocolSpec(
            name="X",
            world_size=2,
            exact=True,
            curve_relation="required",
            params_schema={},
            candidate_for=(),
            semantics=RESULT_SEMANTICS_APPROXIMATE,
        )
        assert spec.result_semantics == RESULT_SEMANTICS_APPROXIMATE


class TestResultSemantics:
    def test_three_levels_defined(self):
        assert set(RESULT_SEMANTICS) == {
            RESULT_SEMANTICS_EXACT,
            RESULT_SEMANTICS_APPROXIMATE,
            RESULT_SEMANTICS_NOISY,
        }

    def test_exact_and_noisy_instances(self):
        for name in ("PROTOCOL_ECDH", "PROTOCOL_KKRT", "PROTOCOL_RR22"):
            assert protocol_result_semantics(name) == RESULT_SEMANTICS_EXACT
            assert protocol_is_exact(name) is True
        assert protocol_result_semantics("PROTOCOL_DP") == RESULT_SEMANTICS_NOISY
        assert protocol_is_exact("PROTOCOL_DP") is False

    def test_normalized_names_are_accepted(self):
        assert protocol_result_semantics("RR22") == RESULT_SEMANTICS_EXACT


class TestPlannerCrossCheck:
    """Planner 字面量 vs 注册表：防漂移。

    planner 不能在导入期依赖 backends（backends/__init__ → jax_backend.codegen
    → planner.planner 会成环），所以候选清单在 planner 侧仍是字面量登记——
    一致性必须由这里逐项锁定。
    """

    def test_candidate_lists_match_registry(self):
        for op in PSI_CANDIDATE_OPS:
            rule = OPERATOR_REGISTRY[op]
            assert rule.protocol_candidates == candidate_protocols_for(op)
            assert rule.protocol_candidates == PSI_PROTOCOL_CANDIDATES

    def test_default_protocol_matches_registry_and_is_a_candidate(self):
        assert PSI_RULE_DEFAULT_PROTOCOL == PSI_DEFAULT_PROTOCOL
        assert PSI_RULE_DEFAULT_PROTOCOL in PSI_PROTOCOL_CANDIDATES

    def test_noisy_protocol_is_not_a_candidate(self):
        assert "PROTOCOL_DP" not in PSI_PROTOCOL_CANDIDATES
        assert candidate_protocols_for("Intersects") == (
            "PROTOCOL_ECDH",
            "PROTOCOL_KKRT",
            "PROTOCOL_RR22",
        )
        assert candidate_protocols_for("HeightBand") == ()


class TestMpcProtocolCrossCheck:
    """MPC（SPU）协议：planner 字面量 vs 后端真源，防漂移。

    与 PSI 侧同理——planner 不能在导入期依赖 backends，所以候选清单在 planner
    侧是字面量；一致性必须由这里逐项锁定，否则"登记了却没接线"会静默通过。
    """

    def test_mpc_candidates_match_the_spu_registry(self):
        assert MPC_PROTOCOL_CANDIDATES == SPU_PROTOCOLS

    def test_default_mpc_protocol_is_a_candidate(self):
        assert MPC_RULE_DEFAULT_PROTOCOL in MPC_PROTOCOL_CANDIDATES

    def test_default_mpc_protocol_matches_the_runtime_default(self):
        import inspect

        from backends.spu_backend import run_spu_simulation

        default = inspect.signature(run_spu_simulation).parameters["protocol"].default
        assert default == MPC_RULE_DEFAULT_PROTOCOL

    def test_mpc_and_psi_namespaces_do_not_overlap(self):
        """两套命名空间必须互斥，否则同一个名字会有两种含义。"""

        assert set(MPC_PROTOCOL_CANDIDATES) & set(PSI_PROTOCOLS) == set()

    def test_every_mpc_operator_registers_the_shared_candidates(self):
        for op in ("DistanceLE", "WeightedSum", "TemporalOverlap"):
            rule = OPERATOR_REGISTRY[op]
            assert rule.mpc_protocol_candidates == MPC_PROTOCOL_CANDIDATES, op
            assert rule.default_mpc_protocol == MPC_RULE_DEFAULT_PROTOCOL, op

    def test_psi_family_operators_declare_no_mpc_protocol(self):
        for op in PSI_CANDIDATE_OPS:
            rule = OPERATOR_REGISTRY[op]
            assert rule.default_mpc_protocol is None, op
            assert rule.mpc_protocol_candidates == (), op
