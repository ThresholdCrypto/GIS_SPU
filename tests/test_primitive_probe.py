# -*- coding: utf-8 -*-
"""逐原语真机核验探针（P0）的守卫测试。

守什么
======
1. **用例表本身**：打包要用到的原语（D3 §7 点名）必须在表里，否则"逐原语核查"
   会漏掉点名的那个；
2. **登记一致性**：表里每条原语声明的"jax/HLO 登记名"必须真在 capability 白名单里
   （或明确登记为 `None` = 不在表里），不许两份表各说各话；
3. **尺子必须与计费维度对齐**：通信量按**计费维度**对账——逐元素电路按输入元素数，
   收缩/矩阵乘按**输出个数**。第一版把 `dot` 实测的 16 B 除以输入元素数（N=64）得到
   0.25 B/元素，误判成"低于秘密乘法下限"；真机 `pphlo` 追踪证明 `dot` 按**输出个数**
   计费、**收缩长度免费**。所以 `below_multiply_floor` 只在**同一维度**上拦真·不可能读数
   （`x * 2` 实测 0 B 仍属这类坑），并新增 `contraction_is_free` 记录"收缩长度免费"；
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
        assert {case.elements for case in cases if not case.fixed_elements} == {8}
        assert {case.repeats for case in cases} == {1}

    def test_contraction_cases_declare_output_billing(self):
        by_name = {case.name: case for case in pp.PROBE_PRIMITIVES}
        for name in ("dot", "dot_long", "matmul", "sum", "sum_long"):
            assert name in by_name, f"收缩/矩阵乘用例缺 {name}"
            assert by_name[name].cost_driver == "outputs", f"{name} 的计费维度不是输出"
            assert by_name[name].outputs, f"{name} 没登记输出个数"
        # 逐元素用例必须仍是 inputs——别被"统一改口径"误伤
        assert by_name["mul"].cost_driver == "inputs"
        assert by_name["mul"].outputs is None

    def test_dot_and_dot_long_share_the_same_output_count(self):
        by_name = {case.name: case for case in pp.PROBE_PRIMITIVES}
        assert by_name["dot"].outputs == by_name["dot_long"].outputs == 1

    def test_contraction_pairs_are_declared_case_names(self):
        names = {case.name for case in pp.PROBE_PRIMITIVES}
        assert {pair[0] for pair in pp.CONTRACTION_PAIRS} <= names
        assert {pair[1] for pair in pp.CONTRACTION_PAIRS} <= names

    def test_scale_cases_are_pinned_against_the_global_size_override(self):
        by_name = {case.name: case for case in pp.primitive_probe_cases(elements=8)}
        assert by_name["mul"].elements == 8
        assert by_name["dot_long"].elements == pp._DOT_LONG_ELEMENTS
        assert by_name["sum_long"].elements == pp._DOT_LONG_ELEMENTS
        assert by_name["matmul"].elements == pp._MATMUL_ELEMENTS

    def test_matmul_case_builds_two_dimensional_inputs(self):
        case = {c.name: c for c in pp.PROBE_PRIMITIVES}["matmul"]
        x, y = case.inputs_fn(case.elements)
        assert x.shape == (pp._MATMUL_M, pp._MATMUL_K)
        assert y.shape == (pp._MATMUL_K, pp._MATMUL_P)

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
            "comm_per_element", "comm_per_output", "cost_per_unit", "cost_driver",
            "outputs", "comm_by_primitive", "within_tolerance",
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
        assert record["comm_per_output"] is None
        assert record["cost_per_unit"] is None
        assert "不给任何实测数字" in record["note"]

    @staticmethod
    def _record(name, *, status="ok", comm=None, elements=64, tolerance=True,
                outputs=None, cost_driver="inputs"):
        billed = outputs if cost_driver == "outputs" else elements
        return {
            "name": name, "status": status, "comm_total_bytes": comm,
            "comm_per_element": None if comm is None else comm / elements,
            "comm_per_output": None if (comm is None or not outputs) else comm / outputs,
            "cost_per_unit": None if (comm is None or not billed) else comm / billed,
            "cost_driver": cost_driver, "outputs": outputs,
            "elements": elements, "within_tolerance": tolerance,
            "needed_by": "packing",
        }

    def test_measured_dot_reading_is_not_flagged_as_impossible(self):
        # 真机读数：dot（收缩长度 64、输出 1 个）实测 16 B。第一版除以输入元素数得到
        # 0.25 B/元素，误判成"低于秘密乘法下限"。按**计费维度**（输出个数）对账：
        # 16 B ≥ 16 B/输出 × 1，不该标可疑。
        records = [
            self._record("mul", comm=1024.0),
            self._record("dot", comm=16.0, elements=64, outputs=1,
                         cost_driver="outputs"),
        ]
        summary = pp.summarize_primitives(records)
        assert summary["below_multiply_floor"] == ()
        assert summary["priced_by_outputs"] == ("dot",)
        assert summary["priced_by_inputs"] == ("mul",)

    def test_a_reading_below_the_billing_axis_floor_is_still_flagged(self):
        # 同维度对账仍要拦真·不可能读数：1 个输出至少 16 B。
        records = [self._record("dot", comm=8.0, elements=64, outputs=1,
                                cost_driver="outputs")]
        summary = pp.summarize_primitives(records)
        assert summary["below_multiply_floor"] == ("dot",)
        assert any("低于同一计费维度上的秘密乘法下限" in note
                   for note in summary["notes"])

    def test_contraction_length_is_reported_as_free(self):
        records = [
            self._record("dot", comm=16.0, elements=64, outputs=1, cost_driver="outputs"),
            self._record("dot_long", comm=16.0, elements=512, outputs=1,
                         cost_driver="outputs"),
            self._record("sum", comm=16.0, elements=64, outputs=1, cost_driver="outputs"),
            self._record("sum_long", comm=16.0, elements=512, outputs=1,
                         cost_driver="outputs"),
        ]
        summary = pp.summarize_primitives(records)
        assert summary["contraction_is_free"] == ("dot", "sum")
        assert sum("收缩长度不计费" in note for note in summary["notes"]) == 2

    def test_contraction_free_is_not_claimed_when_readings_differ(self):
        records = [
            self._record("dot", comm=16.0, elements=64, outputs=1, cost_driver="outputs"),
            self._record("dot_long", comm=1040.0, elements=512, outputs=1,
                         cost_driver="outputs"),
        ]
        summary = pp.summarize_primitives(records)
        assert summary["contraction_is_free"] == ()
        assert any("收缩长度是否免费存疑" in note for note in summary["notes"])

    def test_registered_but_failed_is_reported_as_a_contradiction(self):
        records = [dict(self._record("shift_left", status="error", tolerance=None),
                        registered_jax=True, registered_hlo=True)]
        summary = pp.summarize_primitives(records)
        assert summary["registered_but_not_executed"] == ("shift_left",)
        assert any("登记表与实测不一致" in note for note in summary["notes"])

    def test_empty_records_do_not_invent_a_verdict(self):
        summary = pp.summarize_primitives([])
        assert summary["executed"] == ()
        assert summary["below_multiply_floor"] == ()
        assert summary["contraction_is_free"] == ()
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
