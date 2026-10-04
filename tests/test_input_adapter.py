# -*- coding: utf-8 -*-
"""InputAdapter：CSV / JSON / CellSet / 码列表 → ResolvedCellInput（Phase 2）。

重点不只"能读"，还有"读错时准确失败"：错误必须带文件位置，
空集/重复/缺列这些边界必须被如实登记，而不是静默吞掉。
"""

from __future__ import annotations

import json

import pytest

from backends.psi_backend import (
    PartyInput,
    ResolvedCellInput,
    check_layout_agreement,
    load_cellset,
    normalize_inputs,
)
from geo_privacy.core import CellSet
from ir import encode_grid_code, grid_code_layout_manifest


def _code(x: int) -> int:
    return encode_grid_code(x=x, y=27702, z=15, level=9, toff=0, lt=4)


CODES = (_code(21861), _code(21862), _code(21863))


class TestLoadCellset:
    def test_csv_roundtrip(self, tmp_path):
        path = tmp_path / "route.csv"
        path.write_text(
            "grid_code,label\n" + "\n".join(f"{c},route" for c in CODES) + "\n",
            encoding="utf-8",
        )
        resolved = load_cellset(str(path), name="route")
        assert isinstance(resolved, ResolvedCellInput)
        assert resolved.codes == CODES
        assert resolved.source.startswith("csv:")
        assert resolved.count == 3

    def test_csv_without_key_column_is_disclosed(self, tmp_path):
        path = tmp_path / "route.csv"
        path.write_text(
            "code,label\n" + "\n".join(f"{c},route" for c in CODES) + "\n",
            encoding="utf-8",
        )
        resolved = load_cellset(str(path), name="route")
        assert resolved.codes == CODES
        assert any("grid_code" in note for note in resolved.notes)

    def test_csv_bad_code_reports_line_number(self, tmp_path):
        path = tmp_path / "route.csv"
        path.write_text("grid_code\n111\nxx\n222\n", encoding="utf-8")
        with pytest.raises(ValueError) as excinfo:
            load_cellset(str(path), name="route")
        message = str(excinfo.value)
        assert "xx" in message
        assert ":3" in message  # 数据行行号（含表头）

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "route.csv"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="无数据"):
            load_cellset(str(path), name="route")

    def test_header_only_is_empty_set_with_note(self, tmp_path):
        path = tmp_path / "route.csv"
        path.write_text("grid_code\n", encoding="utf-8")
        resolved = load_cellset(str(path), name="route")
        assert resolved.count == 0
        assert any("空集合" in note for note in resolved.notes)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_cellset(str(tmp_path / "nope.csv"), name="route")

    def test_out_of_range_code_rejected(self):
        with pytest.raises(ValueError, match="2\\^64"):
            load_cellset([2 ** 64], name="route")

    def test_hex_string_codes_allowed(self):
        resolved = load_cellset(["0x10", "16"], name="route")
        assert resolved.codes == (16, 16)  # 允许重复：本层原样保留

    def test_duplicates_counted_but_kept(self):
        resolved = load_cellset([CODES[0], CODES[0], CODES[1]], name="route")
        assert resolved.count == 3
        assert resolved.duplicates == 1
        assert any("重复" in note for note in resolved.notes)

    def test_cellset_object(self):
        resolved = load_cellset(CellSet(CODES, label="航线段"), name="route")
        # CellSet.codes 是 frozenset：顺序不保证，按集合比较
        assert set(resolved.codes) == set(CODES)
        assert resolved.label == "航线段"

    def test_party_input_carries_party_id(self):
        resolved = load_cellset(
            PartyInput(party_id="uav_operator", name="route", data=list(CODES)),
            name="",
        )
        assert resolved.party_id == "uav_operator"
        assert resolved.name == "route"
        assert resolved.codes == CODES

    def test_json_codes_list(self, tmp_path):
        path = tmp_path / "zone.json"
        path.write_text(json.dumps(list(CODES)), encoding="utf-8")
        resolved = load_cellset(str(path), name="zone")
        assert resolved.codes == CODES
        assert resolved.source.startswith("json:")

    def test_json_object_with_layout(self, tmp_path):
        manifest = grid_code_layout_manifest()
        payload = {
            "grid_codes": list(CODES),
            "layout": manifest,
            "label": "禁飞区",
            "party_id": "airspace_authority",
        }
        path = tmp_path / "zone.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        resolved = load_cellset(str(path), name="zone")
        assert resolved.codes == CODES
        assert resolved.layout["layout_id"] == manifest["layout_id"]
        assert resolved.party_id == "airspace_authority"

    def test_corrupt_json_raises_with_path(self, tmp_path):
        path = tmp_path / "zone.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON 解析失败"):
            load_cellset(str(path), name="zone")

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            load_cellset(object(), name="route")

    def test_normalize_inputs_mapping(self, tmp_path):
        path = tmp_path / "zone.csv"
        path.write_text("grid_code\n" + "\n".join(str(c) for c in CODES) + "\n", encoding="utf-8")
        resolved = normalize_inputs({"route": list(CODES), "zone": str(path)})
        assert set(resolved) == {"route", "zone"}
        assert resolved["route"].codes == CODES
        assert resolved["zone"].source.startswith("csv:")


class TestLayoutAgreement:
    def test_json_layout_flows_into_agreement_check(self):
        manifest = grid_code_layout_manifest()
        left = load_cellset(
            {"grid_codes": list(CODES), "layout": manifest}, name="left"
        )
        right = load_cellset(
            {"grid_codes": list(CODES), "layout": dict(manifest)}, name="right"
        )
        agreement = check_layout_agreement(left.layout, right.layout)
        assert agreement.agreement is True

    def test_partial_manifests_compare_field_by_field(self):
        left = {"layout_id": "a", "x_bits": 17}
        right = {"layout_id": "a", "x_bits": 18}
        agreement = check_layout_agreement(left, right)
        assert agreement.agreement is False
        assert any("x_bits" in item for item in agreement.mismatches)
        # 两边都缺的字段不算不一致（None == None）
        assert not any("toff_bits" in item for item in agreement.mismatches)