# -*- coding: utf-8 -*-
"""两方抽象（§16）：WorldConfig / PartyManager —— 确定性 + 显式拒绝多方。"""

from __future__ import annotations

import pytest

from backends.psi_backend import (
    PartyInput,
    PartyManager,
    WorldConfig,
    same_party_both_sides,
)
from backends.psi_backend.input_adapter import load_cellset


class TestWorldConfig:
    def test_two_party_accepted(self):
        WorldConfig(2).validate()

    def test_three_party_rejected_with_reason(self):
        with pytest.raises(ValueError) as excinfo:
            WorldConfig(3).validate()
        message = str(excinfo.value)
        assert "仅支持 2 方" in message
        assert "不得据此宣称" in message


class TestRanks:
    def test_rank_is_first_seen_and_deterministic(self):
        manager = PartyManager()
        assert manager.bind("route", "uav_operator") == 0
        assert manager.bind("zone", "airspace_authority") == 1
        assert manager.bind("backup", "uav_operator") == 0
        assert manager.rank_of("airspace_authority") == 1
        assert manager.rank_of("nobody") is None

    def test_descriptors_snapshot(self):
        manager = PartyManager()
        manager.bind("route", "uav_operator")
        manager.bind("zone", "airspace_authority")
        manager.bind("alt", "uav_operator")
        descriptors = manager.descriptors()
        assert [(item.party_id, item.rank) for item in descriptors] == [
            ("uav_operator", 0),
            ("airspace_authority", 1),
        ]
        assert descriptors[0].inputs == ("route", "alt")
        assert manager.to_dict()["world_size"] == 2

    def test_third_party_is_rejected(self):
        manager = PartyManager()
        manager.bind("a", "p1")
        manager.bind("b", "p2")
        with pytest.raises(ValueError, match="仅支持 2 方"):
            manager.bind("c", "p3")

    def test_empty_party_id_rejected(self):
        with pytest.raises(ValueError, match="不能为空"):
            PartyManager().bind("route", "  ")


class TestFromInputs:
    def test_from_inputs_reads_party_id_only(self):
        resolved = {
            "route": load_cellset(PartyInput("uav_operator", "route", [1, 2])),
            "bare": load_cellset([3]),
        }
        manager = PartyManager.from_inputs(resolved)
        assert [item.party_id for item in manager.descriptors()] == ["uav_operator"]
        assert manager.descriptors()[0].inputs == ("route",)

    def test_from_inputs_with_three_parties_rejected(self):
        resolved = {
            "route": load_cellset(PartyInput("uav_operator", "route", [1])),
            "zone": load_cellset(PartyInput("airspace_authority", "zone", [2])),
            "sensitive": load_cellset(PartyInput("survey_bureau", "sensitive", [3])),
        }
        with pytest.raises(ValueError, match="仅支持 2 方"):
            PartyManager.from_inputs(resolved)

    def test_same_party_helper(self):
        assert same_party_both_sides("a", "a") is True
        assert same_party_both_sides("a", "b") is False
        assert same_party_both_sides("a", None) is False
        assert same_party_both_sides(None, None) is False
