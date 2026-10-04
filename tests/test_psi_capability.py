# -*- coding: utf-8 -*-
"""PSI 协议级能力：三层核查（后端 / 协议 / 参数）与参数静态校验（Phase 1）。"""

from __future__ import annotations

import pytest

from backends.psi_backend import (
    PSI_PARAM_KEYS,
    PSI_RUNTIME_WORLD_SIZE,
    check_psi_protocol_capability,
    validate_psi_protocol_params,
)


class TestParamStaticValidation:
    """`validate_psi_protocol_params`：编译期拦下非法参数组合。"""

    def test_valid_params_pass(self):
        check = validate_psi_protocol_params(
            "PROTOCOL_RR22",
            {"receiver_rank": 0, "broadcast_result": False, "low_comm_mode": True},
        )
        assert check.ok
        assert not check.problems

    def test_unknown_key_is_a_problem(self):
        check = validate_psi_protocol_params("PROTOCOL_ECDH", {"lowcom": True})
        assert not check.ok
        assert "未登记" in check.problems[0]
        assert "lowcom" in check.problems[0]

    def test_receiver_rank_value_table(self):
        for value, expected in (
            (0, True),
            (1, True),
            (True, False),   # bool 不是 int（Python 里 isinstance(True, int) 为真，须显式排除）
            (2, False),
            ("0", False),
            (0.0, False),
            (None, False),
        ):
            check = validate_psi_protocol_params(
                "PROTOCOL_ECDH", {"receiver_rank": value}
            )
            assert check.ok is expected, (value, check.problems)

    def test_bool_keys_must_be_bool(self):
        for value in ("yes", 1, None):
            check = validate_psi_protocol_params(
                "PROTOCOL_RR22", {"low_comm_mode": value}
            )
            assert not check.ok, value
            assert "布尔" in check.problems[0]

    def test_low_comm_on_non_rr22_is_a_note_not_a_problem(self):
        """协议族外参数：不算非法，但要如实披露"不会注入"。"""

        check = validate_psi_protocol_params(
            "PROTOCOL_ECDH", {"low_comm_mode": True}
        )
        assert check.ok
        assert check.notes
        assert "不会注入" in check.notes[0]

    def test_unknown_protocol_is_rejected(self):
        check = validate_psi_protocol_params("PROTOCOL_NOPE", {})
        assert not check.ok
        assert "未知 PSI 协议" in check.problems[0]

    def test_registered_keys_are_exactly_three(self):
        """参数白名单与 Runtime 真正读取的键一致；漂移会在这里暴露。"""

        assert set(PSI_PARAM_KEYS) == {
            "receiver_rank",
            "broadcast_result",
            "low_comm_mode",
        }


class TestProtocolCapabilityLayers:
    """`check_psi_protocol_capability`：三层结论分开报告，不合成一个 runnable。"""

    def test_layers_are_separate_conclusions(self):
        cap = check_psi_protocol_capability("PROTOCOL_ECDH", {}, curve="CURVE_SM2")
        data = cap.to_dict()
        assert {"backend_runnable", "protocol_runnable", "params_runnable"} <= set(data)
        assert data["runnable"] is (
            data["backend_runnable"]
            and data["protocol_runnable"]
            and data["params_runnable"]
        )

    def test_ecdh_without_curve_fails_params_layer(self):
        cap = check_psi_protocol_capability("PROTOCOL_ECDH", {}, curve=None)
        assert not cap.params_runnable
        assert any("曲线" in blocker for blocker in cap.blockers)

    def test_unknown_curve_fails_params_layer(self):
        cap = check_psi_protocol_capability(
            "PROTOCOL_ECDH", {}, curve="CURVE_NOPE"
        )
        assert not cap.params_runnable
        assert any("未知椭圆曲线" in blocker for blocker in cap.blockers)

    def test_three_party_protocol_fails_protocol_layer(self):
        cap = check_psi_protocol_capability(
            "PROTOCOL_ECDH_3PC", {}, curve="CURVE_SM2"
        )
        assert not cap.protocol_runnable
        assert not cap.runnable
        assert any(
            f"{PSI_RUNTIME_WORLD_SIZE} 方" in blocker for blocker in cap.blockers
        )

    def test_curve_ignored_by_protocol_is_a_note(self):
        cap = check_psi_protocol_capability(
            "PROTOCOL_KKRT", {}, curve="CURVE_SM2"
        )
        assert any("不读曲线" in note for note in cap.notes)

    def test_unknown_protocol_is_blocked(self):
        cap = check_psi_protocol_capability("PROTOCOL_NOPE", {})
        assert not cap.runnable
        assert cap.blockers

    def test_rr22_low_comm_request_checked_against_env_probe(self):
        """RR22 + low_comm_mode=True 的可用性取决于环境探测结果。

        环境缺少 Rr22Rarams 时必须 params 层失败并给出原因；
        具备时 params 层通过。两种情况都不允许"假装通过"。
        """

        from backends.psi_backend import check_psi_capabilities

        report = check_psi_capabilities()
        probe_ok = bool((report.rr22_params or {}).get("runnable"))
        cap = check_psi_protocol_capability(
            "PROTOCOL_RR22", {"low_comm_mode": True}, report
        )
        if probe_ok:
            assert cap.params_runnable
        else:
            assert not cap.params_runnable
            assert any("Rr22Rarams" in blocker for blocker in cap.blockers)