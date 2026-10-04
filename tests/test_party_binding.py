# -*- coding: utf-8 -*-
"""Party Binding（§14）端到端：PartyInput → notes / JSON；两方上限（§16）。"""

from __future__ import annotations

import pytest

from backends.psi_backend import PartyInput
from geosecure import Compiler

from tests._helpers import example, has_psi

needs_psi = pytest.mark.skipif(
    not has_psi(), reason="当前环境不具备真实 PSI 执行能力(python3.11+spu+libgomp1)"
)

CHAIN = example("route_zone_chain.py")


def _two_party_inputs():
    return {
        "route": PartyInput("uav_operator", "route", example("route_cells.csv")),
        "no_fly_zone": PartyInput(
            "airspace_authority", "no_fly_zone", example("nofly_cells.csv")
        ),
    }


class TestPartyBinding:
    @needs_psi
    def test_notes_and_json_carry_party_ids(self):
        result = Compiler(inputs=_two_party_inputs()).compile_file(CHAIN)
        run = result.psi_runs["CellSetIntersect"]
        assert run.status == "ok", run.error
        notes = "\n".join(run.notes)
        assert "参与方：uav_operator" in notes
        assert "参与方：airspace_authority" in notes

        roles = {item["role"]: item for item in run.to_dict()["party_binding"]}
        assert roles["left"]["party_id"] == "uav_operator"
        assert roles["left"]["name"] == "route"
        assert roles["right"]["party_id"] == "airspace_authority"

        snapshot = result.to_dict()["parties"]
        assert snapshot["world_size"] == 2
        assert snapshot["parties"][0]["party_id"] == "uav_operator"

    @needs_psi
    def test_operator_status_carries_semantics_and_policy(self):
        result = Compiler(inputs=_two_party_inputs()).compile_file(CHAIN)
        row = next(
            item
            for item in result.operator_status
            if item["operation"] == "CellSetIntersect"
        )
        assert row["result_semantics"] == "exact"
        assert row["result_policy"] == "REVEAL_INTERSECTION"

    @needs_psi
    def test_same_party_both_sides_is_disclosed(self):
        inputs = {
            "route": PartyInput("uav_operator", "route", example("route_cells.csv")),
            "no_fly_zone": PartyInput(
                "uav_operator", "no_fly_zone", example("nofly_cells.csv")
            ),
        }
        result = Compiler(inputs=inputs).compile_file(CHAIN)
        run = result.psi_runs["CellSetIntersect"]
        assert run.status == "ok", run.error
        assert any("不是跨方求交" in note for note in run.notes)

    def test_three_parties_are_rejected_at_compile_time(self):
        inputs = {
            "route": PartyInput("uav_operator", "route", [1]),
            "no_fly_zone": PartyInput("airspace_authority", "no_fly_zone", [2]),
            "sensitive_area": PartyInput("survey_bureau", "sensitive_area", [3]),
        }
        with pytest.raises(ValueError, match="仅支持 2 方"):
            Compiler(inputs=inputs)

    def test_unlabelled_inputs_leave_parties_empty(self):
        compiler = Compiler()
        assert compiler.parties.descriptors() == ()
