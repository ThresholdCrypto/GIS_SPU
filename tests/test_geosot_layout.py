# -*- coding: utf-8 -*-
"""GridCode 布局身份：manifest 字段与跨方握手（Phase 4 / 验收 D）。

布局不一致的两个 64 位码流本身完全"合法"：PSI 不会报错，只会得到一个
形式上正确的错误交集。这里把核对做成机器行为——声明、比对、拒绝。
"""

from __future__ import annotations

import pytest

from backends.psi_backend import (
    PSI_DEFAULT_CURVE,
    PsiRuntimeConfig,
    check_layout_agreement,
    run_psi_intersection,
)
from ir import (
    GRID_CODE_BITS,
    GRID_CODE_LAYOUT_ID,
    GRID_CODE_LAYOUT_VERSION,
    GRID_CODE_WIDTH,
    encode_grid_code,
    grid_code_layout_manifest,
)


def _code(x: int) -> int:
    return encode_grid_code(x=x, y=27702, z=15, level=9, toff=0, lt=4)


ROUTE = (_code(21861), _code(21862))
ZONE = (_code(21862), _code(22999))


class TestManifest:
    def test_manifest_fields_match_the_bit_constants(self):
        manifest = grid_code_layout_manifest()
        assert manifest["bits"] == GRID_CODE_WIDTH == 64
        assert manifest["x_bits"] == GRID_CODE_BITS["X"]
        assert manifest["y_bits"] == GRID_CODE_BITS["Y"]
        assert manifest["z_bits"] == GRID_CODE_BITS["Z"]
        assert manifest["level_bits"] == GRID_CODE_BITS["L"]
        assert manifest["toff_bits"] == GRID_CODE_BITS["Toff"]
        assert manifest["lt_bits"] == GRID_CODE_BITS["Lt"]

    def test_layout_id_encodes_the_actual_widths(self):
        """layout_id 的每个数字都必须与位域常量一致；两处手写会漂移，这里锁死。"""

        manifest = grid_code_layout_manifest()
        expected = (
            f"geosot3d-v{GRID_CODE_LAYOUT_VERSION}"
            f"-x{GRID_CODE_BITS['X']}"
            f"-y{GRID_CODE_BITS['Y']}"
            f"-z{GRID_CODE_BITS['Z']}"
            f"-l{GRID_CODE_BITS['L']}"
            f"-toff{GRID_CODE_BITS['Toff']}"
            f"-lt{GRID_CODE_BITS['Lt']}"
        )
        assert manifest["layout_id"] == expected == GRID_CODE_LAYOUT_ID

    def test_manifest_is_deterministic(self):
        assert grid_code_layout_manifest() == grid_code_layout_manifest()
        assert isinstance(GRID_CODE_LAYOUT_VERSION, int)


class TestLayoutAgreement:
    def test_matching_manifests_pass(self):
        manifest = grid_code_layout_manifest()
        agreement = check_layout_agreement(manifest, dict(manifest))
        assert agreement.agreement is True
        assert not agreement.mismatches

    def test_mismatch_lists_the_differing_fields(self):
        left = grid_code_layout_manifest()
        right = dict(left)
        right["x_bits"] = 18
        right["layout_id"] = "geosot3d-v0-x18-y16-z7-l5-toff14-lt4"
        agreement = check_layout_agreement(left, right)
        assert agreement.agreement is False
        assert any("x_bits" in item for item in agreement.mismatches)
        assert any("layout_id" in item for item in agreement.mismatches)

    def test_single_sided_declaration_is_unknown_not_pass(self):
        agreement = check_layout_agreement(grid_code_layout_manifest(), None)
        assert agreement.agreement is None
        assert agreement.notes

    def test_both_absent_is_unknown(self):
        assert check_layout_agreement(None, None).agreement is None


class TestRuntimeLayoutGate:
    """布局核对必须发生在进入 PSI 之前（与运行环境无关，离线也拦得住）。"""

    def test_mismatch_is_rejected_before_psi_runs(self):
        left = grid_code_layout_manifest()
        right = dict(left)
        right["version"] = 999
        run = run_psi_intersection(
            ROUTE, ZONE, op="Intersects",
            config=PsiRuntimeConfig(protocol="PROTOCOL_ECDH", curve=PSI_DEFAULT_CURVE),
            left_layout=left,
            right_layout=right,
        )
        assert run.status == "error"
        assert run.error.startswith("LAYOUT_MISMATCH")
        assert "version" in run.error
        assert run.intersection_count is None
        assert run.intersection == ()
        assert run.layout_agreement["agreement"] is False

    def test_matching_layouts_do_not_block_execution(self):
        manifest = grid_code_layout_manifest()
        run = run_psi_intersection(
            ROUTE, ZONE, op="Intersects",
            config=PsiRuntimeConfig(protocol="PROTOCOL_ECDH", curve=PSI_DEFAULT_CURVE),
            left_layout=manifest,
            right_layout=dict(manifest),
        )
        # 环境不可用 → unavailable；可用 → ok。共同点是：布局核对没有拦下它。
        assert run.status in ("ok", "unavailable")
        assert run.layout_agreement["agreement"] is True