# -*- coding: utf-8 -*-
"""逐原语真机核验探针（P0）的守卫测试。

守什么
======
1. **用例表本身**：打包要用到的原语（D3 §7 点名）必须在表里，否则"逐原语核查"
   会漏掉点名的那个；
2. **登记一致性**：表里每条原语声明的"jax/HLO 登记名"必须真在 capability 白名单里
   （或明确登记为 `None` = 不在表里），不许两份表各说各话；
3. **不许把"低于秘密乘法下限"的读数当好消息**：`x * 2` 实测 0 B 的教训——
   低于下限必须进 `below_multiply_floor`，而不是被读成"这个原语免费"；
4. **无 SPU 时不给数字**：环境不可执行必须 `unavailable`，不留推测值。
"""

from __future__ import annotations

import pytest

from backends.spu_backend import capability
from backends.spu_backend import primitive_probe as pp
from backends.spu_backend.capability import CapabilityReport


class TestCaseTable:
    def test_d3_named_packing_primitives_are_covered(self):
        covered = {case.name for case in pp.PROBE_PRIMITIVES}
        # docs/BITPLANE_LAYOUT.md §7 第 2 步点名的原语（+ P7 已测的两条，作对拍）
        for required in ("shift_left", "and", "or", "dot", "shift_right", "mul"):
            assert required in covered, f"逐原语核查漏了 {required}"

    def test_names_are_unique_and_non_empty(self):
        names = [case.name for case in pp.PROBE_PRIMITIVES]
        assert len(names) == len(set(names)) and all(names)

    def test_needed_by_is_from_the_known_set(self):
        assert {case.needed_by for case in pp.PROBE_PRIMITIVES} <= {
            "packing", "layout", "general", "fragile"
        }

    def test_every_case_has_a_reference(self):
        for case in pp.PROBE_PRIMITIVES:
            assert callable(case.fn) and callable(case.reference_fn)

    def test_cases_are_instantiable_with_size_and_repeats(self):
        cases = pp.primitive_probe_cases(elements=8, repeats=0)
        assert {case.elements for case in cases} == {8}
        assert {case.repeats for case in cases} == {1}

    def test_only_filter_keeps_order_and_subset(self):
        cases = pp.primitive_probe_cases(only=["mul", "and"])
        assert [case.name for case in cases] == ["mul", "and"]

    def test_label_mentions_op_and_name(self):
        case = pp.PROBE_PRIMITIVES[0]
        assert pp.PRIMITIVE_PROBE_OP in case.label() and case.name in case.label()


class TestRegistrationMirror:
    """"登记了"与"表里写着"必须一致——两份清单不许各说各话。"""

    def test_declared_jax_names_are_actually_registered(self):
        for case in pp.PROBE_PRIMITIVES:
            if case.jax_name is None:
                continue
            assert case.jax_name in capability.SPU_ADAPTED_PRIMITIVES, (
                f"{case.name} 声明 jax 登记名 {case.jax_name}，但 capability 白名单里没有"
            )

    def test_declared_hlo_names_are_actually_registered(self):
        for case in pp.PROBE_PRIMITIVES:
            if case.hlo_name is None:
                continue
            assert case.hlo_name in capability.SPU_ADAPTED_HLO_PRIMITIVES, (
                f"{case.name} 声明 HLO 登记名 {case.hlo_name}，但 capability 白名单里没有"
            )

    def test_unregistered_primitives_are_declared_as_such(self):
        # xor 与 top_k 是"实测能跑、但登记表里没有"的两条：必须显式声明，
        # 否则会变成"看起来登记过"
        by_name = {case.name: case for case in pp.PROBE_PRIMITIVES}
        assert by_name["xor"].jax_name is None
        assert by_name["xor"].hlo_name is None
        assert by_name["top_k"].hlo_name is None

    def test_bitwise_primitive_names_are_the_hlo_ones(self):
        by_name = {case.name: case for case in pp.PROBE_PRIMITIVES}
        assert by_name["and"].hlo_name == "and"
        assert by_name["shift_right"].hlo_name == "shift_right_arithmetic"


class TestRecordsAndSummaries:
    def test_blank_record_has_the_contract_keys(self):
        record = pp._blank_record(pp.PROBE_PRIMITIVES[0])
        for key in (
            "case", "name", "op", "jax_primitive", "hlo_primitive", "needed_by",
            "registered_jax", "registered_hlo", "status", "comm_total_bytes",
            "comm_per_element", "comm_by_primitive", "within_tolerance",
        ):
            assert key in record
        assert record["status"] == "unavailable"
        assert record["comm_total_bytes"] is None

    def test_blocked_environment_gives_no_numbers(self):
        blocked = CapabilityReport(runnable=False, blockers=("测试：环境不可执行",))
        record = pp.run_primitive_case(pp.PROBE_PRIMITIVES[0], report=blocked)
        assert record["status"] == "unavailable"
        assert record["comm_total_bytes"] is None
        assert record["comm_per_element"] is None
        assert "不给任何实测数字" in record["note"]

    @staticmethod
    def _record(name, *, status="ok", comm=None, elements=64, tolerance=True):
        return {
            "name": name, "status": status, "comm_total_bytes": comm,
            "comm_per_element": None if comm is None else comm / elements,
            "elements": elements, "within_tolerance": tolerance,
            "needed_by": "packing",
        }

    def test_dot_below_the_multiply_floor_is_flagged(self):
        # N=64 的秘密收缩至少要 64 次秘密乘法 = 1024 B；16 B 是物理上不可能的读数
        records = [
            self._record("mul", comm=1024.0),
            self._record("dot", comm=16.0),
        ]
        summary = pp.summarize_primitives(records)
        assert summary["below_multiply_floor"] == ("dot",)
        assert any("dot" in note for note in summary["notes"])

    def test_readings_at_or_above_the_floor_are_not_flagged(self):
        records = [
            self._record("mul", comm=1024.0),
            self._record("dot", comm=1024.0),
        ]
        assert pp.summarize_primitives(records)["below_multiply_floor"] == ()

    def test_registered_but_failed_is_reported_as_a_contradiction(self):
        records = [dict(self._record("shift_left", status="error", tolerance=None),
                        registered_jax=True, registered_hlo=True)]
        summary = pp.summarize_primitives(records)
        assert summary["registered_but_not_executed"] == ("shift_left",)
        assert any("登记表与实测不一致" in note for note in summary["notes"])

    def test_empty_records_do_not_invent_a_verdict(self):
        summary = pp.summarize_primitives([])
        assert summary["executed"] == ()
        assert summary["notes"] == ["没有记录，无法判定"]

    def test_packing_primitives_are_listed(self):
        records = [self._record("and", comm=100.0)]
        assert pp.summarize_primitives(records)["packing_primitives"] == ("and",)

    def test_formatter_survives_an_empty_table(self):
        assert "真机跑通：0 项" in pp.format_primitive_summary([])

    def test_csv_columns_cover_every_blank_record_key(self):
        record = pp._blank_record(pp.PROBE_PRIMITIVES[0])
        for column in pp.PRIMITIVE_CSV_COLUMNS:
            assert column in record, f"CSV 列 {column} 不在记录里"
