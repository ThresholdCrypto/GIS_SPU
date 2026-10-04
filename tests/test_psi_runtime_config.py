# -*- coding: utf-8 -*-
"""PsiRuntimeConfig：单一配置源的行为契约（Phase 1 / 验收 A）。

验收目标（课题 §27-A）：
    Planner protocol_params == Runtime 实际参数 == PsiRunResult 记录参数
Runtime 不再用函数默认值兜底各自的协议参数；配置只有一条链。
"""

from __future__ import annotations

import pytest

from backends.psi_backend import (
    PSI_DEFAULT_CURVE,
    PsiRuntimeConfig,
    run_psi_intersection,
)
from ir import encode_grid_code

from tests._helpers import has_psi


def _code(x: int, y: int = 27702, z: int = 15, level: int = 9) -> int:
    return encode_grid_code(x=x, y=y, z=z, level=level, toff=0, lt=4)


ROUTE = (_code(21861), _code(21862), _code(21863))
ZONE = (_code(21862), _code(21863), _code(22999))
EXPECTED_INTERSECTION = (ZONE[0], ZONE[1])

needs_psi = pytest.mark.skipif(
    not has_psi(), reason="当前环境不具备真实 PSI 执行能力(python3.11+spu+libgomp1)"
)


class TestFromParams:
    """Planner 步骤参数 → 运行时配置：拆分规则要一一对应。"""

    def test_splits_receiver_rank_and_broadcast_out_of_params(self):
        config = PsiRuntimeConfig.from_params(
            "PROTOCOL_RR22",
            None,
            {"receiver_rank": 1, "broadcast_result": True, "low_comm_mode": True},
        )
        assert config.receiver_rank == 1
        assert config.broadcast_result is True
        assert dict(config.protocol_params) == {"low_comm_mode": True}
        data = config.to_dict()
        assert data["protocol"] == "PROTOCOL_RR22"
        assert data["receiver_rank"] == 1
        assert data["broadcast_result"] is True
        assert data["protocol_params"] == {"low_comm_mode": True}

    def test_defaults_when_params_absent(self):
        config = PsiRuntimeConfig.from_params("PROTOCOL_ECDH", "CURVE_SM2", None)
        assert config.receiver_rank == 0
        assert config.broadcast_result is False
        assert dict(config.protocol_params) == {}
        assert config.curve == "CURVE_SM2"

    def test_does_not_mutate_caller_mapping(self):
        params = {"receiver_rank": 1, "low_comm_mode": False}
        PsiRuntimeConfig.from_params("PROTOCOL_RR22", None, params)
        assert params == {"receiver_rank": 1, "low_comm_mode": False}


class TestRuntimeReadsConfigOnly:
    """Runtime 只读配置对象；旧关键字路径产出同一种配置。"""

    @needs_psi
    def test_receiver_rank_from_config_is_actually_executed(self):
        run = run_psi_intersection(
            ROUTE,
            ZONE,
            op="Intersects",
            config=PsiRuntimeConfig(
                protocol="PROTOCOL_ECDH",
                curve=PSI_DEFAULT_CURVE,
                receiver_rank=1,
            ),
        )
        assert run.status == "ok", run.error
        assert run.receiver_rank == 1
        assert run.runtime_config["receiver_rank"] == 1
        assert run.intersection_count == len(EXPECTED_INTERSECTION)
        assert run.value is True

    @needs_psi
    def test_legacy_kwargs_and_explicit_config_agree(self):
        legacy = run_psi_intersection(
            ROUTE, ZONE, op="Intersects",
            protocol="PROTOCOL_ECDH", curve=PSI_DEFAULT_CURVE,
        )
        explicit = run_psi_intersection(
            ROUTE, ZONE, op="Intersects",
            config=PsiRuntimeConfig(
                protocol="PROTOCOL_ECDH", curve=PSI_DEFAULT_CURVE
            ),
        )
        assert legacy.status == explicit.status == "ok"
        assert dict(legacy.runtime_config) == dict(explicit.runtime_config)

    @needs_psi
    def test_broadcast_result_is_injected_from_config(self):
        run = run_psi_intersection(
            ROUTE, ZONE, op="Intersects",
            config=PsiRuntimeConfig(
                protocol="PROTOCOL_ECDH",
                curve=PSI_DEFAULT_CURVE,
                broadcast_result=True,
            ),
        )
        assert run.status == "ok", run.error
        assert run.broadcast_result is True
        assert run.runtime_config["broadcast_result"] is True

    def test_config_precedes_conflicting_kwargs(self):
        """配置是单一来源：config 给出后，旧关键字不再参与决策。"""

        run = run_psi_intersection(
            ROUTE, ZONE, op="Intersects",
            protocol="PROTOCOL_RR22",
            rr22_low_comm_mode=False,
            config=PsiRuntimeConfig(
                protocol="PROTOCOL_RR22",
                curve=None,
                protocol_params={"low_comm_mode": True},
            ),
        )
        # 无论环境是否具备 RR22 执行能力，配置快照都必须来自 config
        assert run.protocol == "PROTOCOL_RR22"
        assert run.runtime_config["protocol_params"]["low_comm_mode"] is True
        if run.status == "ok":
            assert run.protocol_params["low_comm_mode"] is True


class TestConfigValidation:
    """非法配置在进入 PSI 之前失败，错误信息必须指出具体字段与取值。"""

    def test_invalid_receiver_rank_is_rejected_before_psi(self):
        run = run_psi_intersection(
            ROUTE, ZONE, op="Intersects",
            config=PsiRuntimeConfig(
                protocol="PROTOCOL_ECDH",
                curve=PSI_DEFAULT_CURVE,
                receiver_rank=7,
            ),
        )
        assert run.status == "error"
        assert "receiver_rank" in run.error
        assert "0 或 1" in run.error
        assert run.intersection_count is None

    def test_non_int_receiver_rank_is_rejected(self):
        run = run_psi_intersection(
            ROUTE, ZONE, op="Intersects",
            config=PsiRuntimeConfig(
                protocol="PROTOCOL_ECDH",
                curve=PSI_DEFAULT_CURVE,
                receiver_rank="zero",
            ),
        )
        assert run.status == "error"
        assert "receiver_rank" in run.error

    def test_ecdh_without_curve_fails_before_psi(self):
        run = run_psi_intersection(
            ROUTE, ZONE, op="Intersects",
            config=PsiRuntimeConfig(protocol="PROTOCOL_ECDH", curve=None),
        )
        assert run.status == "error"
        assert "曲线" in run.error
        assert run.intersection_count is None

    def test_missing_protocol_is_rejected(self):
        run = run_psi_intersection(
            ROUTE, ZONE, op="Intersects",
            config=PsiRuntimeConfig(protocol="PROTOCOL_NOPE"),
        )
        assert run.status == "error"
        assert "未知 PSI 协议" in run.error