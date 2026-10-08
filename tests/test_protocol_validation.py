# -*- coding: utf-8 -*-
"""Phase 2 / Phase 3：MPC 协议元数据 + 统一 capability validation（TDD）。

任务书（本阶段指导文档 §十七）：
    Phase 2：通用 ProtocolSpec 抽象——PsiProtocolSpec 继续可用；
             MpcProtocolSpec 独立扩展，不破坏已有接口。
    Phase 3：统一 capability validation——
             protocol → family → world_size → operation → field
             → security/result semantics → parameters 的完整前置检查。

本文件的纪律（§十四 / §十五）：
- 只校验静态事实（注册表 / 校验器 / 编译期闸门），不制造 fake 协议、不 mock 成功；
- Runtime 的 fail-fast 契约单独锁定：新校验只能让错误**更早**，不许更晚；
- 消息断言只取子串（协议名 / 数字 / 关键词），不锁整句。

覆盖映射（§六检查清单 → 用例）：
    protocol 是否存在        TestUnifiedValidatorFamily::test_unknown_protocol_*
    family 是否正确          TestUnifiedValidatorFamily::test_family_*
    world_size 是否匹配      TestUnifiedValidator*::test_*_world_size_*
    field 是否支持           TestUnifiedValidatorMpc::test_field_*
    operation 是否支持       TestUnifiedValidator*::test_*_operation_*
    exact/noisy 语义         TestUnifiedValidator*::test_require_exact_*
    parameter 是否合法       TestUnifiedValidator*::test_*_params_*
    curve 是否满足要求       TestUnifiedValidatorPsi::test_curve_*
"""

from __future__ import annotations

import json

import pytest

from backends.psi_backend import (
    PROTOCOL_SPECS as PSI_PROTOCOL_SPECS,
    RESULT_SEMANTICS as PSI_RESULT_SEMANTICS,
)
from backends.psi_backend.protocol_registry import PsiProtocolSpec
from backends.protocol_validation import (
    PROTOCOL_FAMILIES,
    ProtocolValidationResult,
    validate_protocol_request,
)
from backends.spu_backend import (
    PROTOCOL_MIN_WORLD_SIZE,
    SPU_FIELDS,
    SPU_PROTOCOLS,
    SPU_PROTOCOLS_WITHOUT_CRYPTO,
    mpc_protocol_spec,
    normalize_field,
    protocol_min_world_size,
    run_spu_simulation,
)
from backends.spu_backend.protocol_registry import (
    MPC_CANDIDATE_OPS,
    MPC_PROTOCOL_SPECS,
    MPC_SECURITY_MODELS,
    RESULT_SEMANTICS as MPC_RESULT_SEMANTICS,
    MpcProtocolSpec,
)
from frontend import parse_source
from geosecure.cli import main as cli_main
from geosecure.compiler import Compiler
from planner import (
    OPERATOR_REGISTRY,
    plan_program,
    select_mpc_protocol,
    validate_mpc_protocol_for_operation,
)

from tests._helpers import example, has_spu

MPC_SOURCE = (
    "from geo_privacy import geo\n"
    "def f(p1, p2, threshold):\n"
    "    return geo.distance_le(p1, p2, threshold)\n"
)

PSI_SOURCE = (
    "from geo_privacy import geo\n"
    "def f(route, no_fly_zone):\n"
    "    return geo.intersects(route, no_fly_zone)\n"
)


def _joined(result: ProtocolValidationResult) -> str:
    return "；".join(result.problems)


# ==========================================================================
# 1) Phase 2：PsiProtocolSpec 的 family 扩展
# ==========================================================================


class TestPsiProtocolSpecFamily:
    def test_every_psi_spec_declares_its_family(self):
        for name, spec in PSI_PROTOCOL_SPECS.items():
            assert spec.family == "PSI", name
            assert spec.to_dict()["family"] == "PSI", name

    def test_wrong_family_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="family"):
            PsiProtocolSpec(
                name="X",
                world_size=2,
                exact=True,
                curve_relation="required",
                params_schema={},
                candidate_for=(),
                family="MPC",
            )


# ==========================================================================
# 2) Phase 2：MpcProtocolSpec 注册表
# ==========================================================================


class TestMpcProtocolRegistry:
    def test_registry_is_the_source_for_the_protocol_enum(self):
        assert tuple(MPC_PROTOCOL_SPECS) == SPU_PROTOCOLS
        assert {spec.name for spec in MPC_PROTOCOL_SPECS.values()} == set(SPU_PROTOCOLS)

    def test_world_size_table_is_derived_from_specs(self):
        assert dict(PROTOCOL_MIN_WORLD_SIZE) == {
            name: spec.world_size for name, spec in MPC_PROTOCOL_SPECS.items()
        }
        assert protocol_min_world_size("ABY3") == 3
        assert protocol_min_world_size("SEMI2K") == 2

    def test_without_crypto_is_derived_from_the_security_model(self):
        assert SPU_PROTOCOLS_WITHOUT_CRYPTO == ("REF2K",)
        assert mpc_protocol_spec("REF2K").security_model == "none"

    def test_security_models_match_the_recorded_evidence(self):
        """安全模型登记必须有出处（上游文档或本仓库实测），不许凭空填。"""

        for name, spec in MPC_PROTOCOL_SPECS.items():
            assert spec.security_model in MPC_SECURITY_MODELS, name
            assert spec.security_note.strip(), f"{name} 缺 security_note（登记依据）"
        for name in ("ABY3", "SEMI2K", "CHEETAH"):
            assert mpc_protocol_spec(name).security_model == "semi-honest", name
        assert mpc_protocol_spec("SECURENN").security_model == "unverified"
        # REF2K 的判据是本仓库实测（0 B 通信），不是对上游文档的推断
        assert "0 B" in mpc_protocol_spec("REF2K").security_note

    def test_supported_fields_only_contain_verified_combinations(self):
        """只登记真机扫描过的协议×环宽组合；放宽前必须先补测试。"""

        assert mpc_protocol_spec("ABY3").supported_fields == ("FM32", "FM64", "FM128")
        assert mpc_protocol_spec("SEMI2K").supported_fields == ("FM32", "FM64")
        assert mpc_protocol_spec("CHEETAH").supported_fields == ("FM32", "FM64")
        assert mpc_protocol_spec("REF2K").supported_fields == ("FM64",)
        assert mpc_protocol_spec("SECURENN").supported_fields == ("FM64",)
        for name, spec in MPC_PROTOCOL_SPECS.items():
            assert set(spec.supported_fields) <= set(SPU_FIELDS), name
            assert "FM64" in spec.supported_fields, name

    def test_semantics_vocabulary_matches_the_psi_registry(self):
        """§11 的三档语义词汇只有一个来源；MPC 侧副本必须与 PSI 侧同集。"""

        assert set(MPC_RESULT_SEMANTICS) == set(PSI_RESULT_SEMANTICS)

    def test_all_current_specs_are_exact(self):
        for name, spec in MPC_PROTOCOL_SPECS.items():
            assert spec.exact is True, name
            assert spec.result_semantics == "exact", name

    def test_candidate_ops_cross_check_with_the_planner(self):
        for op in MPC_CANDIDATE_OPS:
            assert OPERATOR_REGISTRY[op].mpc_protocol_candidates == tuple(
                MPC_PROTOCOL_SPECS
            ), op
        for name, spec in MPC_PROTOCOL_SPECS.items():
            assert spec.candidate_for == MPC_CANDIDATE_OPS, name

    def test_invalid_spec_inputs_are_rejected_at_construction(self):
        base = dict(
            name="X",
            world_size=2,
            security_model="semi-honest",
            security_note="测试用",
            supported_fields=("FM64",),
            candidate_for=(),
            params_schema={},
        )
        with pytest.raises(ValueError, match="security_model"):
            MpcProtocolSpec(**{**base, "security_model": "maybe"})
        with pytest.raises(ValueError, match="FM99|环宽"):
            MpcProtocolSpec(**{**base, "supported_fields": ("FM99",)})
        with pytest.raises(ValueError, match="world_size"):
            MpcProtocolSpec(**{**base, "world_size": 1})
        with pytest.raises(ValueError, match="family"):
            MpcProtocolSpec(**{**base, "family": "PSI"})

    def test_to_dict_shape(self):
        data = mpc_protocol_spec("ABY3").to_dict()
        for key in (
            "name",
            "family",
            "world_size",
            "security_model",
            "result_semantics",
            "exact",
            "supported_fields",
            "candidate_for",
        ):
            assert key in data, key
        assert isinstance(data["supported_fields"], list)
        assert data["family"] == "MPC"

    def test_spec_lookup_is_normalized_and_strict(self):
        assert mpc_protocol_spec("aby3").name == "ABY3"
        with pytest.raises(ValueError, match="REF2K"):
            mpc_protocol_spec("SPDZ2K")


# ==========================================================================
# 3) Phase 3：统一校验——family 判定与协议存在性
# ==========================================================================


class TestUnifiedValidatorFamily:
    def test_family_vocabulary(self):
        assert PROTOCOL_FAMILIES == ("PSI", "MPC")

    def test_invalid_family_argument_is_rejected(self):
        result = validate_protocol_request(family="TEE", protocol="ABY3")
        assert not result.ok
        assert "TEE" in _joined(result)

    def test_protocol_none_is_allowed_with_a_disclosure_note(self):
        result = validate_protocol_request(protocol=None, family="MPC")
        assert result.ok
        assert result.protocol is None
        assert result.notes
        assert "自动" in result.notes[0] or "默认" in result.notes[0]

    def test_family_is_inferred_from_the_name(self):
        psi = validate_protocol_request(protocol="RR22")
        assert psi.ok and psi.family == "PSI" and psi.protocol == "PROTOCOL_RR22"
        mpc = validate_protocol_request(protocol="aby3")
        assert mpc.ok and mpc.family == "MPC" and mpc.protocol == "ABY3"

    def test_family_mismatch_mentions_the_other_namespace(self):
        wrong = validate_protocol_request(family="MPC", protocol="PROTOCOL_RR22")
        assert not wrong.ok
        assert "PSI" in _joined(wrong)
        assert "PROTOCOL_RR22" in _joined(wrong)

        wrong_psi = validate_protocol_request(family="PSI", protocol="ABY3")
        assert not wrong_psi.ok
        assert "MPC" in _joined(wrong_psi)
        assert "ABY3" in _joined(wrong_psi)

    def test_unknown_protocol_lists_both_namespaces(self):
        result = validate_protocol_request(protocol="NOPE_PROTOCOL")
        assert not result.ok
        joined = _joined(result)
        assert "PROTOCOL_ECDH" in joined
        assert "REF2K" in joined

# ==========================================================================
# 4) Phase 3：PSI 族检查
# ==========================================================================


class TestUnifiedValidatorPsi:
    def test_three_party_protocol_is_rejected_in_the_two_party_chain(self):
        result = validate_protocol_request(family="PSI", protocol="PROTOCOL_ECDH_3PC")
        assert not result.ok
        joined = _joined(result)
        assert "3" in joined and "2" in joined

    def test_field_is_not_applicable_for_psi(self):
        result = validate_protocol_request(family="PSI", protocol="ECDH", field=64)
        assert result.ok
        assert any("环宽" in note for note in result.notes)
        assert result.field is None

    def test_curve_validation(self):
        bad = validate_protocol_request(family="PSI", protocol="ECDH", curve="CURVE_NOPE")
        assert not bad.ok
        assert "CURVE_NOPE" in _joined(bad)

        # 不读曲线的协议：曲线是合法名字 → 放行 + 如实披露不生效
        ignored = validate_protocol_request(family="PSI", protocol="RR22", curve="SM2")
        assert ignored.ok
        assert any("曲线" in note for note in ignored.notes)

        # 需要曲线的协议缺省 → 放行（有默认值），不报错
        defaulted = validate_protocol_request(family="PSI", protocol="ECDH")
        assert defaulted.ok

    def test_parameter_validation_delegates_to_the_psi_capability(self):
        bad_rank = validate_protocol_request(
            family="PSI", protocol="RR22", protocol_params={"receiver_rank": 2}
        )
        assert not bad_rank.ok
        assert "receiver_rank" in _joined(bad_rank)

        bool_rank = validate_protocol_request(
            family="PSI", protocol="RR22", protocol_params={"receiver_rank": True}
        )
        assert not bool_rank.ok

        unknown = validate_protocol_request(
            family="PSI", protocol="RR22", protocol_params={"foo": 1}
        )
        assert not unknown.ok
        assert "foo" in _joined(unknown)

        good = validate_protocol_request(
            family="PSI", protocol="RR22", protocol_params={"receiver_rank": 1}
        )
        assert good.ok

    def test_require_exact_semantics(self):
        rejected = validate_protocol_request(
            family="PSI", protocol="PROTOCOL_DP", require_exact=True
        )
        assert not rejected.ok
        joined = _joined(rejected)
        assert "exact" in joined or "精确" in joined

        noisy = validate_protocol_request(family="PSI", protocol="PROTOCOL_DP")
        assert noisy.ok
        assert any("带噪" in note or "noisy" in note for note in noisy.notes)

        exact = validate_protocol_request(
            family="PSI", protocol="ECDH", require_exact=True
        )
        assert exact.ok

    def test_operation_check(self):
        wrong = validate_protocol_request(
            family="PSI", protocol="ECDH", operation="DistanceLE"
        )
        assert not wrong.ok
        assert "DistanceLE" in _joined(wrong)

        right = validate_protocol_request(
            family="PSI", protocol="ECDH", operation="Intersects"
        )
        assert right.ok


# ==========================================================================
# 5) Phase 3：MPC 族检查
# ==========================================================================


class TestUnifiedValidatorMpc:
    def test_world_size_check(self):
        conflict = validate_protocol_request(
            family="MPC", protocol="ABY3", world_size=2
        )
        assert not conflict.ok
        joined = _joined(conflict)
        assert "ABY3" in joined and "3" in joined and "world_size=2" in joined
        # 拒绝信息必须给出可操作替代
        assert "SEMI2K" in joined

        assert validate_protocol_request(
            family="MPC", protocol="ABY3", world_size=3
        ).ok
        assert validate_protocol_request(
            family="MPC", protocol="SEMI2K", world_size=2
        ).ok
        assert validate_protocol_request(family="MPC", protocol="ABY3").ok

        for bad in (1, True):
            result = validate_protocol_request(
                family="MPC", protocol="ABY3", world_size=bad
            )
            assert not result.ok, bad
            assert "world_size" in _joined(result)

    def test_field_check(self):
        assert validate_protocol_request(
            family="MPC", protocol="ABY3", field="FM128"
        ).ok
        assert validate_protocol_request(
            family="MPC", protocol="ABY3", field=128
        ).ok

        unverified = validate_protocol_request(
            family="MPC", protocol="SEMI2K", field="FM128"
        )
        assert not unverified.ok
        joined = _joined(unverified)
        assert "FM128" in joined and "SEMI2K" in joined
        # 未实测组合的拒绝信息要指出放宽路径
        assert "真机" in joined or "实测" in joined

        unknown = validate_protocol_request(
            family="MPC", protocol="ABY3", field="FM99"
        )
        assert not unknown.ok
        assert "FM99" in _joined(unknown)

        assert validate_protocol_request(
            family="MPC", protocol="SEMI2K", field=32
        ).ok
        assert validate_protocol_request(family="MPC", protocol="ABY3").ok

    def test_result_reports_the_normalized_field(self):
        result = validate_protocol_request(family="MPC", protocol="ABY3", field=64)
        assert result.ok and result.field == "FM64"

    def test_operation_check(self):
        assert validate_protocol_request(
            family="MPC", protocol="ABY3", operation="WeightedSum"
        ).ok
        wrong = validate_protocol_request(
            family="MPC", protocol="ABY3", operation="Intersects"
        )
        assert not wrong.ok
        assert "Intersects" in _joined(wrong)

    def test_unregistered_params_are_rejected(self):
        result = validate_protocol_request(
            family="MPC", protocol="ABY3", protocol_params={"foo": 1}
        )
        assert not result.ok
        assert "foo" in _joined(result)
        assert validate_protocol_request(
            family="MPC", protocol="ABY3", protocol_params={}
        ).ok

    def test_curve_is_not_applicable_for_mpc(self):
        result = validate_protocol_request(
            family="MPC", protocol="ABY3", curve="CURVE_SM2"
        )
        assert result.ok
        assert any("曲线" in note for note in result.notes)

    def test_result_to_dict_is_json_ready(self):
        result = validate_protocol_request(family="MPC", protocol="ABY3", field=64)
        data = result.to_dict()
        for key in ("ok", "family", "protocol", "field", "problems", "notes"):
            assert key in data, key
        json.dumps(data)


# ==========================================================================
# 6) Phase 3：Planner 接线（field / world_size 进入算子×协议校验）
# ==========================================================================


class TestPlannerMpcValidation:
    def test_existing_signature_is_backward_compatible(self):
        check = validate_mpc_protocol_for_operation("DistanceLE", "ABY3")
        assert check.ok
        assert any("3 方" in note for note in check.notes)

    def test_world_size_conflict_is_rejected(self):
        check = validate_mpc_protocol_for_operation(
            "DistanceLE", "ABY3", world_size=2
        )
        assert not check.ok
        joined = "；".join(check.problems)
        assert "ABY3" in joined and "world_size=2" in joined
        assert "SEMI2K" in joined
        assert validate_mpc_protocol_for_operation(
            "DistanceLE", "SEMI2K", world_size=2
        ).ok

    def test_field_conflict_is_rejected(self):
        check = validate_mpc_protocol_for_operation(
            "DistanceLE", "SEMI2K", field="FM128"
        )
        assert not check.ok
        assert "FM128" in "；".join(check.problems)
        assert validate_mpc_protocol_for_operation(
            "DistanceLE", "ABY3", field="FM128"
        ).ok

    def test_plan_program_threads_world_size(self):
        plan = plan_program(
            parse_source(MPC_SOURCE).program,
            mpc_protocol="ABY3",
            mpc_world_size=2,
        )
        assert plan.has_errors
        diagnostic = next(
            d for d in plan.diagnostics if d.code == "PROTOCOL_UNSUPPORTED"
        )
        assert "3" in diagnostic.message and "world_size=2" in diagnostic.message

        clean = plan_program(
            parse_source(MPC_SOURCE).program,
            mpc_protocol="ABY3",
            mpc_world_size=3,
        )
        assert not clean.has_errors

    def test_plan_program_threads_field(self):
        dirty = plan_program(
            parse_source(MPC_SOURCE).program,
            mpc_protocol="SEMI2K",
            mpc_field="FM128",
        )
        assert dirty.has_errors
        assert any("FM128" in d.message for d in dirty.diagnostics)

        clean = plan_program(
            parse_source(MPC_SOURCE).program,
            mpc_protocol="ABY3",
            mpc_field="FM128",
        )
        assert not clean.has_errors

    def test_auto_selection_is_validated_against_world_size(self):
        """自动选择路径同样受 world_size 校验（当前策略：拒绝，而非过滤）。

        期望值从实际选择结果推导（选择依据是实测产物），
        避免把"ABY3 恰为最省"这一会随产物变化的巧合锁进断言。
        """

        selection = select_mpc_protocol("DistanceLE")
        assert selection is not None
        required = protocol_min_world_size(selection.protocol)

        plan = plan_program(parse_source(MPC_SOURCE).program, mpc_world_size=2)
        if required > 2:
            assert plan.has_errors
            diagnostic = next(
                d for d in plan.diagnostics if d.code == "PROTOCOL_UNSUPPORTED"
            )
            assert selection.protocol in diagnostic.message
            assert "world_size=2" in diagnostic.message
        else:
            assert not plan.has_errors

    def test_psi_steps_are_not_blocked_by_mpc_world_size(self):
        plan = plan_program(
            parse_source(PSI_SOURCE).program,
            mpc_protocol="ABY3",
            mpc_world_size=2,
        )
        assert not plan.has_errors


# ==========================================================================
# 7) Phase 3：Compiler 编译期前置拒绝（field / world_size）
# ==========================================================================


class TestCompilerGate:
    def test_world_size_conflict_is_rejected_at_construction(self):
        with pytest.raises(ValueError) as excinfo:
            Compiler(protocol="ABY3", world_size=2)
        message = str(excinfo.value)
        assert "ABY3" in message and "3" in message and "world_size=2" in message

    def test_invalid_field_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="FM99"):
            Compiler(field="FM99")

    def test_unverified_field_combination_is_rejected(self):
        with pytest.raises(ValueError, match="FM128"):
            Compiler(protocol="SEMI2K", field=128)
        # ABY3 × FM128 在实测矩阵内：不阻塞
        Compiler(protocol="ABY3", field=128)

    def test_invalid_world_size_value_is_rejected(self):
        with pytest.raises(ValueError, match="world_size"):
            Compiler(world_size=1)
        with pytest.raises(ValueError, match="world_size"):
            Compiler(world_size=True)

    def test_unknown_protocol_is_left_to_the_planner_diagnostic(self):
        """协议名不存在时构造期**不抛错**：保留带位置/建议/代价的规划层诊断
        （CLI 契约：错误码 1 + 完整编译报告，见 test_end_to_end）。"""

        Compiler(protocol="SPDZ2K")

    def test_cli_reports_invalid_field_readably(self, capsys):
        exit_code = cli_main(
            ["build", example("distance_check.py"), "--field", "FM99"]
        )
        output = capsys.readouterr().out
        assert exit_code == 2
        assert "FM99" in output
        assert "Traceback" not in output


# ==========================================================================
# 8) Runtime fail-fast 契约：新校验只能让错误更早，不许把错误挪晚
# ==========================================================================


class TestRuntimeFailFastContract:
    def test_bad_protocol_or_field_still_raises_before_any_execution(self):
        def identity(x):
            return x

        with pytest.raises(ValueError, match="SPDZ2K"):
            run_spu_simulation(identity, [], protocol="SPDZ2K")
        with pytest.raises(ValueError, match="FM99"):
            run_spu_simulation(identity, [], field="FM99")

    @pytest.mark.skipif(not has_spu(), reason="当前环境不可运行 SPU")
    def test_world_size_below_minimum_reports_a_readable_error(self):
        import numpy as np

        def identity(x):
            return x

        run = run_spu_simulation(
            identity,
            [np.array([1, 2, 3], dtype=np.int32)],
            protocol="ABY3",
            world_size=2,
        )
        assert run.status == "error"
        assert "至少需要" in (run.error or "")
