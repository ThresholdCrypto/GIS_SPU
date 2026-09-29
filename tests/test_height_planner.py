"""高度层（Z）在编译链路上的测试：第 6 类失败模式 + 三维端到端。"""

from __future__ import annotations

import pytest

from ir import (
    GeoOperation,
    GeoProgram,
    decode_grid_code,
    encode_grid_code,
    height_index,
    height_layer_lower,
)
from planner import plan_program
from tests._helpers import example
from validator import FAILURE_BY_CODE, validate_height_layer_capacity, validate_all


# --------------------------------------------------------------------------
# 第 6 类失败模式：高度层号超出 Z 位域
# --------------------------------------------------------------------------


def _program_with_height(level, hmax, hmin=0.0):
    program = GeoProgram(name="height_capacity")
    program.declare_input("route")
    program.declare_input("zone")
    program.add_operation(
        GeoOperation(
            op="Intersects",
            inputs=["route", "zone"],
            output_type="Relation",
            params={"height_level": level, "height_min": hmin, "height_max": hmax},
            location={"file": "height_case.py", "line": 7, "col": 11},
        )
    )
    return program


class TestHeightFailureClass:
    def test_registered_as_its_own_class(self):
        failure = FAILURE_BY_CODE["HEIGHT_LAYER_UNSUPPORTED"]
        assert failure.name == "高度层号超出 Z 位域"
        assert failure.remedy and failure.description

    def test_overflow_is_reported_with_location_cause_and_remedy(self):
        program = _program_with_height(level=23, hmax=1000.0)
        diagnostics = validate_height_layer_capacity(program)
        assert len(diagnostics) == 1
        d = diagnostics[0]
        assert d.severity == "error"
        assert d.location_str == "height_case.py:7:11"
        # 原因必须给出可核对的数字，而不是"参数有问题"
        assert "第 8 层" in d.cause or "8 位" in d.cause
        # 建议必须给出可用层级
        assert "L<=22" in d.suggestion
        # 代价提示不可缺（五类报告口径的第 4 项）
        assert d.estimated_cost is not None
        assert d.estimated_cost["backend"] == "PSI"

    def test_fitting_height_band_produces_no_diagnostic(self):
        assert validate_height_layer_capacity(_program_with_height(22, 1000.0)) == []
        # 收窄高度带即可在更高层级工作
        assert validate_height_layer_capacity(_program_with_height(23, 300.0)) == []

    def test_operation_without_height_params_is_skipped(self):
        """没写高度参数的算子不应被这一检查波及。"""

        program = GeoProgram(name="no_height")
        program.declare_input("a")
        program.declare_input("b")
        program.add_operation(GeoOperation(op="Intersects", inputs=["a", "b"]))
        assert validate_height_layer_capacity(program) == []

    def test_invalid_level_is_reported_not_crashed(self):
        program = _program_with_height(level=40, hmax=1000.0)
        diagnostics = validate_height_layer_capacity(program)
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == "error"
        assert "非法" in diagnostics[0].message

    def test_surfaces_through_validate_all(self):
        """必须真接进 validate_all，否则 CLI 拿不到这条诊断。"""

        program = _program_with_height(level=25, hmax=2000.0)
        report = validate_all(program, plan_program(program))
        codes = {d.code for d in report.diagnostics}
        assert "HEIGHT_LAYER_UNSUPPORTED" in codes
        grouped = report.by_class()
        assert "HEIGHT_LAYER_UNSUPPORTED" in grouped


# --------------------------------------------------------------------------
# 三维端到端：高度带 -> 层集合 -> 集合交
# --------------------------------------------------------------------------


class TestVerticalWorkflow:
    LEVEL = 15

    def test_vertical_conflict_example_compiles(self):
        from geosecure import compile_file

        result = compile_file(example("vertical_conflict.py"))
        assert result.program is not None
        ops = [op.op for op in result.program.operations]
        assert ops == ["Intersects", "Contains"]
        assert result.errors == []

    def test_height_band_cells_respect_z_field(self):
        from geo_privacy import geo

        band = geo.height_band(1, 2, 0, 20000, self.LEVEL)
        assert band.cardinality() == height_index(20000.0, self.LEVEL) + 1
        for code in band.codes:
            assert 0 <= decode_grid_code(code)["Z"] <= 127

    def test_three_dimensional_verdict_differs_from_projection(self):
        """三维判定的核心价值：投影重合但高度不重叠 → 不冲突。

        若按旧口径（Z 不参与语义）或按"只比 XY"，这两种情形都会被判为冲突。
        """

        from geo_privacy import CellSet

        xy = dict(x=5, y=6, level=self.LEVEL, toff=0, lt=0)
        low = CellSet([encode_grid_code(z=0, **xy)])
        high = CellSet([encode_grid_code(z=30, **xy)])
        # XY 完全相同
        assert decode_grid_code(next(iter(low.codes)))["X"] == decode_grid_code(
            next(iter(high.codes))
        )["X"]
        # 高度不同 → 不冲突
        assert not low.intersects(high)

    def test_layer_boundaries_are_exact_not_linear(self):
        """层边界按 B.4 精确口径；用线性近似会在高层号上偏小。"""

        layer = 20
        exact = height_layer_lower(layer, self.LEVEL)
        linear = layer * (height_layer_lower(1, self.LEVEL) - height_layer_lower(0, self.LEVEL))
        assert exact > linear
        # 精确值仍是该层的下底面：height_index 必须落回同一层
        assert height_index(exact + 1e-6, self.LEVEL) == layer


# --------------------------------------------------------------------------
# 规划的诚实性：高度语义进了规则表
# --------------------------------------------------------------------------


class TestPlannerHeightNotes:
    def test_set_family_rules_document_height_semantics(self):
        from planner import get_rule

        for op in ("Intersects", "Contains", "CellSetIntersect"):
            notes = get_rule(op).notes
            text = " ".join(notes) if isinstance(notes, tuple) else str(notes)
            assert "高度层号" in text, op
            assert "PSI" in text or "区间比较" in text, op

    def test_plan_reports_height_backend_as_psi(self):
        """高度带冲突仍走 PSI——不能被规划成 MPC 区间比较。"""

        program = GeoProgram(name="vertical")
        program.declare_input("route_band")
        program.declare_input("zone_band")
        program.add_operation(
            GeoOperation(op="Intersects", inputs=["route_band", "zone_band"])
        )
        plan = plan_program(program)
        step = plan.steps[0]
        assert step.backend == "PSI"
        assert step.representation == "CompactCellSet"
