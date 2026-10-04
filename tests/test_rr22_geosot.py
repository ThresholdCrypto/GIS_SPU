"""GeoSOT-3D → 64 位 grid_code → CellSet → CompactCellSet → RR22 链路测试。

课题要求验证的完整链：

    GeoSOT coordinate → grid_code → CellSet
        → CompactCellSet（Planner 表征）→ RR22 PSI → intersection

边界（必须保持，不能混）：
    CellSet         = 业务层格网集合对象（geo_privacy.core，明文可用）
    CompactCellSet  = Planner 给集合族算子登记的计算表征（planner/registry.py）

本文件把这条链**跑真**：RR22 求交结果与 CellSet 明文交集逐一对拍。
环境不具备真实 PSI 时按原因 skip，不放宽断言。
"""

from __future__ import annotations

import pytest

from backends.psi_backend import run_psi_intersection
from frontend import parse_source
from geo_privacy.core import CellSet
from ir import encode_grid_code
from planner import plan_program

from tests._helpers import has_psi

needs_psi = pytest.mark.skipif(
    not has_psi(), reason="当前环境不具备真实 PSI 执行能力(python3.11+spu+libgomp1)"
)


def _code(x: int, y: int = 27702, z: int = 15, level: int = 9) -> int:
    """GeoSOT-3D 坐标/层号 → 课题口径的 64 位码（X17|Y17|Z7|L5|Toff14|Lt4）。"""

    return encode_grid_code(x=x, y=y, z=z, level=level, toff=0, lt=4)


def _rr22_reference(left, right):
    return tuple(sorted(set(left) & set(right)))


@needs_psi
class TestGeosotToRr22:
    """一条链测到底：编码 → CellSet → CompactCellSet → RR22 → 交集。"""

    @staticmethod
    def _run(left: CellSet, right: CellSet):
        run = run_psi_intersection(
            sorted(left.codes),
            sorted(right.codes),
            op="CellSetIntersect",
            protocol="PROTOCOL_RR22",
            reference_fn=_rr22_reference,
        )
        assert run.status == "ok", run.error
        return run

    def test_intersecting_grids(self):
        route = CellSet([_code(21861), _code(21862), _code(21863)], label="route")
        zone = CellSet([_code(21862), _code(21863), _code(22999)], label="zone")
        assert route.intersects(zone)
        run = self._run(route, zone)
        expected = tuple(sorted(route.intersection(zone).codes))
        assert run.value == expected
        assert run.value == tuple(sorted({_code(21862), _code(21863)}))
        assert run.agreement is True

    def test_disjoint_grids_same_xy_different_height_layer(self):
        """三维语义：同一 XY、不同高度层 = 不同码 = 不相交。"""

        low = CellSet([_code(21862, z=15)], label="low")
        high = CellSet([_code(21862, z=16)], label="high")
        assert low.codes != high.codes
        assert not low.intersects(high)
        run = self._run(low, high)
        assert run.value == ()

    def test_identical_sets(self):
        cells = [_code(21861), _code(21862), _code(21863)]
        run = self._run(CellSet(cells), CellSet(cells))
        assert run.value == tuple(sorted(cells))

    def test_one_side_empty_is_resolved_without_starting_the_protocol(self):
        run = run_psi_intersection(
            sorted(CellSet([_code(21861)]).codes),
            [],
            op="CellSetIntersect",
            protocol="PROTOCOL_RR22",
        )
        assert run.status == "empty-input"
        assert run.value == ()
        assert any("未启动 PSI 协议" in note for note in run.notes)

    def test_high_bit_codes_are_not_truncated(self):
        high = _code(x=131071)
        assert high >= 2 ** 63
        run = self._run(CellSet([high]), CellSet([high]))
        assert run.value == (high,)

    def test_duplicate_codes_are_collapsed_before_psi(self):
        duplicated = CellSet([_code(21861), _code(21861), _code(21862)])
        assert duplicated.cardinality() == 2
        run = self._run(duplicated, CellSet([_code(21861), _code(21861)]))
        assert run.value == (_code(21861),)

    def test_input_order_does_not_change_the_result(self):
        route_codes = [_code(21861), _code(21862), _code(21863)]
        zone = CellSet([_code(21862), _code(21863), _code(22999)])
        forward = self._run(CellSet(sorted(route_codes)), zone)
        backward = self._run(CellSet(sorted(route_codes, reverse=True)), zone)
        assert forward.value == backward.value
        assert forward.value == tuple(sorted(forward.value))  # 输出已排序

    def test_planner_keeps_cellset_as_compactcellset_and_rr22_is_planning_layer(self):
        """表征边界：CellSet 是业务对象，CompactCellSet 是 Planner 表征。"""

        parse = parse_source(
            "from geo_privacy import geo\n"
            "def f(a, b):\n"
            "    return geo.cellset_intersect(a, b)\n"
        )
        plan = plan_program(
            parse.program,
            psi_protocol="PROTOCOL_RR22",
            psi_protocol_params={"low_comm_mode": False},
        )
        step = plan.steps[0]
        assert step.operation == "CellSetIntersect"
        assert step.representation == "CompactCellSet"
        assert step.backend == "PSI"
        assert step.protocol == "PROTOCOL_RR22"
        assert step.protocol_params["low_comm_mode"] is False
        # Geo-IR 本身不知道协议（约束 §27-不要 4）：算子模型的字段里没有协议
        import dataclasses

        from ir import GeoOperation

        names = {f.name for f in dataclasses.fields(GeoOperation)}
        assert "protocol" not in names
        assert "protocol_params" not in names
