"""ir 层测试：类型系统、格网编码口径、算子与程序、关系对象。"""

from __future__ import annotations

import datetime as dt

import pytest

from ir import (
    GRID_CODE_BITS,
    GRID_CODE_BYTES,
    GRID_CODE_LAYOUT,
    GRID_CODE_WIDTH,
    MAX_ENCODABLE_LEVEL,
    LT_MAX,
    TIME_TREE_ORIGIN,
    GeoEntity,
    GeoOperation,
    GeoProgram,
    GeoRelation,
    GeoType,
    GeoValue,
    Sensitivity,
    decode_grid_code,
    encode_grid_code,
    max_sensitivity,
    relation_from_operation,
    requires_crypto,
    toff_of,
)


# --------------------------------------------------------------------------
# 类型系统
# --------------------------------------------------------------------------


class TestGeoType:
    def test_required_members_exist(self):
        for name in (
            "ENTITY_SET",
            "CELL_SET",
            "VECTOR",
            "POINT",
            "TIME_INTERVAL",
            "SCALAR",
            "RELATION",
            "BOOL",
            "UNKNOWN",
        ):
            assert hasattr(GeoType, name)

    def test_string_values_are_stable(self):
        # 字符串值会进入 JSON 与报告，不能用默认枚举数字
        assert str(GeoType.RELATION) == "Relation"
        assert str(GeoType.ENTITY_SET) == "EntitySet"


class TestSensitivity:
    def test_ordering(self):
        assert requires_crypto(Sensitivity.SECRET)
        assert requires_crypto(Sensitivity.SENSITIVE)
        assert not requires_crypto(Sensitivity.INTERNAL)
        assert not requires_crypto(Sensitivity.PUBLIC)

    def test_max_sensitivity_takes_most_sensitive(self):
        assert max_sensitivity(Sensitivity.PUBLIC, Sensitivity.SECRET) is Sensitivity.SECRET
        assert max_sensitivity(Sensitivity.INTERNAL, Sensitivity.PUBLIC) is Sensitivity.INTERNAL

    def test_max_sensitivity_default_is_conservative(self):
        # 全部为空的保守默认必须是 INTERNAL 而不是 PUBLIC
        assert max_sensitivity() is Sensitivity.INTERNAL


# --------------------------------------------------------------------------
# 格网编码口径：与课题规范逐位对齐
# --------------------------------------------------------------------------


class TestGridCodeLayout:
    def test_bit_layout_sums_to_64(self):
        assert GRID_CODE_WIDTH == 64
        assert GRID_CODE_BYTES == 8
        assert sum(GRID_CODE_BITS.values()) == 64

    def test_bit_layout_matches_spec(self):
        # X'17 | Y'17 | Z7 | L5 | Toff14 | Lt4
        # （本版把 L 由 4 位扩到 5 位：原 ver 位并入 L 高位）
        assert GRID_CODE_BITS["X"] == 17
        assert GRID_CODE_BITS["Y"] == 17
        assert GRID_CODE_BITS["Z"] == 7
        assert GRID_CODE_BITS["L"] == 5
        assert GRID_CODE_BITS["Toff"] == 14
        assert GRID_CODE_BITS["Lt"] == 4
        # ver 恒为 0，不再是独立位段
        assert "ver" not in GRID_CODE_BITS
        assert GRID_CODE_LAYOUT == "X17|Y17|Z7|L5|Toff14|Lt4"
        assert MAX_ENCODABLE_LEVEL == 31

    def test_field_bitmasks_partition_the_key(self):
        """每段"单独置满"的掩码必须两两不重叠、并集恰好 64 位。

        位移现在由布局推导（ir.values._derive_shifts），这条是那个
        推导的直接检验：位移走偏会让两个位段重叠，而编码只会
        **静默错位**，不会报错。
        """

        arg_of = {
            "X": "x",
            "Y": "y",
            "Z": "z",
            "L": "level",
            "Toff": "toff",
            "Lt": "lt",
        }
        assert set(arg_of) == set(GRID_CODE_BITS)

        union = 0
        for name, width in GRID_CODE_BITS.items():
            kwargs = dict(x=0, y=0, z=0, level=0, toff=0, lt=0)
            kwargs[arg_of[name]] = (1 << width) - 1
            mask = encode_grid_code(**kwargs)
            assert mask & union == 0, f"位段 {name} 与已占位重叠"
            union |= mask
        assert union == (1 << 64) - 1


class TestGridCodeEncoding:
    def test_matches_project_sample_v5(self):
        """课题 v5 样例在**本版布局**下的取值。

        旧值 0x2AB29B0D87C90E08 是按 L4 + ver1 布局算出的。本版 L 由 4 位
        扩到 5 位，码位重排，因此**所有已生成的码都变了**（旧值
        不可再用）。这里锁的是新布局下的值，供跨方对拍。
        """

        code = encode_grid_code(x=21861, y=27702, z=15, level=9, toff=2160, lt=4)
        assert f"0x{code:016X}" == "0x2AB29B0D87A48704"

    def test_matches_handoff_example(self):
        """同上，Toff=2040（旧值 0x2AB29B0D87C8FF08 同样已作废）。"""

        code = encode_grid_code(x=21861, y=27702, z=15, level=9, toff=2040, lt=4)
        assert f"0x{code:016X}" == "0x2AB29B0D87A47F84"

    def test_roundtrip(self):
        original = dict(x=21861, y=27702, z=15, level=9, toff=2160, lt=4)
        decoded = decode_grid_code(encode_grid_code(**original))
        assert decoded == {
            "X": 21861,
            "Y": 27702,
            "Z": 15,
            "L": 9,
            "Toff": 2160,
            "Lt": 4,
        }

    def test_overflow_is_rejected_early(self):
        # 17 位字段超过上限必须立刻失败，而不是静默截断
        with pytest.raises(ValueError, match="超出"):
            encode_grid_code(x=1 << 17, y=0, z=0, level=0, toff=0, lt=0)

    def test_negative_component_rejected(self):
        with pytest.raises(ValueError, match="非负"):
            encode_grid_code(x=-1, y=0, z=0, level=0, toff=0, lt=0)

    def test_decode_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            decode_grid_code(1 << 64)


class TestTimeAxis:
    def test_origin_matches_spec(self):
        assert TIME_TREE_ORIGIN.isoformat() == "2026-09-06T22:00:00+08:00"

    def test_lt_max(self):
        assert LT_MAX == 14

    def test_toff_from_moment(self):
        moment = dt.datetime(2026, 9, 7, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=8)))
        assert toff_of(moment) == 120  # 22:00 → 次日 00:00 = 120 分钟

    def test_toff_requires_timezone(self):
        with pytest.raises(ValueError, match="时区"):
            toff_of(dt.datetime(2026, 9, 7, 0, 0))

    def test_toff_before_origin_rejected(self):
        moment = dt.datetime(2026, 9, 6, 21, 0, tzinfo=dt.timezone(dt.timedelta(hours=8)))
        with pytest.raises(ValueError, match="早于时间树原点"):
            toff_of(moment)


# --------------------------------------------------------------------------
# GeoValue / GeoEntity
# --------------------------------------------------------------------------


class TestGeoValue:
    def test_constant_is_public(self):
        value = GeoValue.constant("threshold", 2)
        assert value.is_constant
        assert value.effective_sensitivity is Sensitivity.PUBLIC
        assert not requires_crypto(value.effective_sensitivity)

    def test_variable_inherits_entity_sensitivity(self):
        entity = GeoEntity(name="route_A", sensitivity=Sensitivity.SECRET)
        value = GeoValue(
            name="route_A",
            geo_type=GeoType.ENTITY_SET,
            sensitivity=Sensitivity.PUBLIC,  # 故意写错，应被实体覆盖
            entity=entity,
        )
        assert value.effective_sensitivity is Sensitivity.SECRET


class TestGeoEntity:
    def test_rejects_invalid_cell_code(self):
        with pytest.raises(ValueError):
            GeoEntity(name="bad", cell_codes=(1 << 64,))

    def test_metadata_fields(self):
        entity = GeoEntity(
            name="route_A",
            spatial_scope="A",
            temporal_scope="[2160,2164)",
            attrs={"A01": 37},
        )
        assert entity.spatial_scope == "A"
        assert entity.temporal_scope == "[2160,2164)"
        assert entity.attrs == {"A01": 37}


# --------------------------------------------------------------------------
# GeoOperation / GeoProgram
# --------------------------------------------------------------------------


class TestGeoOperation:
    def test_required_constructor_signature(self):
        """课题指定的构造签名必须原样可用。"""

        op = GeoOperation(
            op="Intersects", inputs=["route", "no_fly_zone"], output_type="Relation"
        )
        assert op.op == "Intersects"
        assert op.inputs == ("route", "no_fly_zone")
        assert op.output_type is GeoType.RELATION

    def test_string_output_type_is_coerced(self):
        op = GeoOperation(op="DistanceLE", inputs=["a", "b", "t"], output_type="Relation")
        assert isinstance(op.output_type, GeoType)

    def test_unknown_output_type_rejected(self):
        with pytest.raises(ValueError):
            GeoOperation(op="X", inputs=[], output_type="NotAType")

    def test_default_output_name_is_deterministic(self):
        op = GeoOperation(op="TemporalOverlap", inputs=["a", "b"], output_type="Relation")
        assert op.output_name == "temporaloverlap_0"

    def test_location_str_without_location(self):
        op = GeoOperation(op="Intersects", inputs=["a", "b"], output_type="Relation")
        assert op.location_str == "<unknown>"


class TestGeoProgram:
    def test_declare_input_and_lookup(self):
        program = GeoProgram(name="t")
        program.declare_input("route_A", sensitivity=Sensitivity.SENSITIVE)
        value = program.lookup("route_A")
        assert value is not None
        assert value.effective_sensitivity is Sensitivity.SENSITIVE

    def test_lookup_unknown_returns_none(self):
        assert GeoProgram().lookup("nope") is None

    def test_input_sensitivity_is_max_of_inputs(self):
        program = GeoProgram()
        program.declare_input("a", sensitivity=Sensitivity.PUBLIC)
        program.declare_input("b", sensitivity=Sensitivity.SECRET)
        op = GeoOperation(op="Intersects", inputs=["a", "b"], output_type="Relation")
        assert program.input_sensitivity(op) is Sensitivity.SECRET
        assert program.requires_crypto(op)

    def test_rotation_of_operations_preserves_order(self):
        program = GeoProgram()
        program.declare_input("a")
        program.declare_input("b")
        first = program.add_operation(
            GeoOperation(op="Intersects", inputs=["a", "b"], output_type="Relation", output_name="r1")
        )
        second = program.add_operation(
            GeoOperation(op="Intersects", inputs=["r1", "b"], output_type="Relation", output_name="r2")
        )
        assert program.operations == [first, second]
        assert program.get_operation("r2") is second

    def test_to_dict_shape(self):
        program = GeoProgram(name="t", source_file="t.py")
        program.declare_input("a")
        program.add_operation(
            GeoOperation(op="Intersects", inputs=["a", "b"], output_type="Relation")
        )
        data = program.to_dict()
        assert data["name"] == "t"
        assert data["source_file"] == "t.py"
        assert data["operations"][0]["op"] == "Intersects"
        assert "inputs" in data and "returns" in data


# --------------------------------------------------------------------------
# GeoRelation
# --------------------------------------------------------------------------


class TestGeoRelation:
    def test_all_required_fields(self):
        relation = GeoRelation(
            subject="Route_A",
            predicate="Intersects",
            object="NoFlyZone_B",
            time="[2160,2164)",
            spatial_scope="A",
            sensitivity=Sensitivity.SECRET,
        )
        for field_name in ("subject", "predicate", "object", "time", "spatial_scope", "sensitivity"):
            assert hasattr(relation, field_name)

    def test_triple_and_str(self):
        relation = GeoRelation(subject="Route_A", predicate="Intersects", object="NoFlyZone_B")
        assert relation.triple == ("Route_A", "Intersects", "NoFlyZone_B")
        assert str(relation) == "Route_A | Intersects | NoFlyZone_B"

    def test_to_dict_has_required_keys(self):
        relation = GeoRelation(subject="a", predicate="p", object="b")
        data = relation.to_dict()
        assert set(data) == {"subject", "predicate", "object", "time", "spatial_scope", "sensitivity"}

    def test_factory_fills_all_fields(self):
        relation = relation_from_operation(
            "Route_A",
            "Intersects",
            "NoFlyZone_B",
            time="[2160,2164)",
            spatial_scope="A",
            sensitivity=Sensitivity.SENSITIVE,
        )
        assert relation.to_dict() == {
            "subject": "Route_A",
            "predicate": "Intersects",
            "object": "NoFlyZone_B",
            "time": "[2160,2164)",
            "spatial_scope": "A",
            "sensitivity": "sensitive",
        }


# --------------------------------------------------------------------------
# 课题首要验证目标
# --------------------------------------------------------------------------


class TestPrimaryValidationTarget:
    def test_route_a_intersects_noflyzone_b_is_standard_geo_ir(self):
        """Route_A | Intersects | NoFlyZone_B 必须能表示为标准 Geo-IR。"""

        program = GeoProgram(name="primary")
        program.declare_input("Route_A", sensitivity=Sensitivity.SENSITIVE)
        program.declare_input("NoFlyZone_B", sensitivity=Sensitivity.SENSITIVE)
        operation = program.add_operation(
            GeoOperation(
                op="Intersects",
                inputs=["Route_A", "NoFlyZone_B"],
                output_type="Relation",
            )
        )
        relation = program.add_relation(
            relation_from_operation("Route_A", "Intersects", "NoFlyZone_B")
        )

        assert operation.op == "Intersects"
        assert relation.triple == ("Route_A", "Intersects", "NoFlyZone_B")
        assert str(relation) == "Route_A | Intersects | NoFlyZone_B"

        # 序列化后仍可还原
        data = program.to_dict()
        assert data["relations"][0]["subject"] == "Route_A"
        assert data["relations"][0]["predicate"] == "Intersects"
        assert data["relations"][0]["object"] == "NoFlyZone_B"
