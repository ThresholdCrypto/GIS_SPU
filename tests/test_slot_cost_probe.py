# -*- coding: utf-8 -*-
"""打包电路的取槽代价（P7-P0）：秘密的按位操作到底多贵。

为什么单独一个文件
==================
P6 测到"秘密 × 秘密"乘法 16 B/元素，于是"元素数 ÷8 ⇒ 通信量 ÷8"作为**上界**站得住。
但打包电路要把 8 位值从环元素里取出来（右移 + 掩码），而 MPC 的按位操作不免费。
本文件锁住这件事的三个层次：

  1. 三个变体（mask_only / shift_only / shift_mask）与"无位运算"变体的单价对比；
  2. 取槽步相对纯乘法的倍率——这是"打包值不值"的**前置判据**，不是收益结论；
  3. 口径纪律：`mul_only`（秘密 × 公开常数）实测为 0 B，**不能**当比价基准
     （它根本不通信）；缺数一律留空 + 说明原因。
"""

from __future__ import annotations

import csv
import json
import types

import numpy as np
import pytest

from backends.spu_backend import slot_cost_probe as sp
from tests._helpers import has_spu

requires_spu = pytest.mark.skipif(
    not has_spu(), reason="需要真实 SPU 执行能力（Linux/WSL2 + spu + jax 0.4.34）"
)


# --------------------------------------------------------------------------
# 用例集与变体表
# --------------------------------------------------------------------------


def test_variant_table_is_fixed_and_covers_the_three_steps():
    names = [variant.name for variant in sp.SLOT_COST_VARIANTS]
    assert names == ["mul_only", "mask_only", "shift_only", "shift_mask"]
    table = {variant.name: (variant.mask, variant.shift) for variant in sp.SLOT_COST_VARIANTS}
    assert table["mul_only"] == (False, False)
    assert table["mask_only"] == (True, False)
    assert table["shift_only"] == (False, True)
    assert table["shift_mask"] == (True, True)


def test_case_set_has_one_case_per_variant():
    cases = sp.slot_cost_cases()
    assert [case.variant for case in cases] == [v.name for v in sp.SLOT_COST_VARIANTS]
    assert {case.elements for case in cases} == {sp.SLOT_COST_ELEMENTS}
    assert {case.capture_comm for case in cases} == {True}


def test_elements_and_repeats_are_configurable():
    cases = sp.slot_cost_cases(elements=32, repeats=2)
    assert {case.elements for case in cases} == {32}
    assert {case.repeats for case in cases} == {2}
    assert {case.repeats for case in sp.slot_cost_cases(repeats=0)} == {1}


def test_labels_mention_op_and_variant():
    assert sp.slot_cost_cases()[0].label() == f"{sp.SLOT_COST_OP} mul_only N={sp.SLOT_COST_ELEMENTS}"


def test_unknown_variant_raises_on_lookup():
    with pytest.raises(KeyError):
        sp._variant("no_such_variant")


# --------------------------------------------------------------------------
# 输入与参考（打包的位运算语义）
# --------------------------------------------------------------------------


def test_inputs_pack_every_slot_and_round_trip():
    packed, values = sp._slot_inputs(64, sp.SLOT_COST_SLOTS)
    assert packed.shape == (64,) and values.shape == (sp.SLOT_COST_SLOTS, 64)
    for slot in range(sp.SLOT_COST_SLOTS):
        extracted = (packed >> np.int64(sp.SLOT_WIDTH_BITS * slot)) & np.int64(0xFF)
        assert np.array_equal(extracted, values[slot])
    assert values.max() <= 0xFF  # 不串槽


def test_inputs_are_deterministic():
    first = sp._slot_inputs(32, sp.SLOT_COST_SLOTS)
    second = sp._slot_inputs(32, sp.SLOT_COST_SLOTS)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])


def test_shift_mask_reference_equals_the_plaintext_slot_weighted_sum():
    """取槽步的参考实现必须等于"把每个槽的值取出来加权求和"这个业务语义。"""

    packed, values = sp._slot_inputs(64, sp.SLOT_COST_SLOTS)
    expected = sum(values[slot] * (slot + 1) for slot in range(sp.SLOT_COST_SLOTS))
    assert np.array_equal(sp._variant("shift_mask").reference(packed), expected)


def test_each_variant_reference_matches_its_own_expression():
    packed, _ = sp._slot_inputs(16, sp.SLOT_COST_SLOTS)
    for variant in sp.SLOT_COST_VARIANTS:
        assert np.array_equal(
            variant.reference(packed),
            sp._slot_reference(packed, mask=variant.mask, shift=variant.shift),
        )


def test_variant_circuits_produce_the_same_numbers_as_their_references():
    """电路与参考在明文下必须一致（真机核对在下面的 requires_spu 部分）。"""

    import jax.numpy as jnp

    # 用 jax 的默认整型（int32）：4 槽 × 8 位在 int32 内不溢出，
    # 且显式 int64 在未开 x64 的 jax 上会被静默截断（并告警），那样测的就不是本意
    packed, _ = sp._slot_inputs(16, sp.SLOT_COST_SLOTS)
    assert int(packed.max()) < 2**31
    for variant in sp.SLOT_COST_VARIANTS:
        got = np.asarray(variant.fn(jnp.asarray(packed)))
        assert np.array_equal(got, variant.reference(packed)), variant.name


# --------------------------------------------------------------------------
# 诚实占位
# --------------------------------------------------------------------------


def test_unrunnable_environment_records_unavailable_without_numbers():
    blocked = types.SimpleNamespace(runnable=False, blockers=("stub",))
    record = sp.run_slot_cost_case(sp.slot_cost_cases()[0], report=blocked)
    assert record["status"] == "unavailable"
    assert "无法真实执行" in record["note"]
    for key in (
        "wall_ms",
        "comm_total_bytes",
        "comm_per_element",
        "max_abs_error",
        "within_tolerance",
    ):
        assert record[key] is None


def test_unknown_variant_is_an_error_not_a_silent_fallback():
    record = sp.run_slot_cost_case(
        sp.SlotCostCase(variant="nope"),
        report=types.SimpleNamespace(runnable=True, blockers=()),
    )
    assert record["status"] == "error"
    assert "nope" in record["error"]


def test_blank_record_leaves_every_number_empty():
    record = sp._blank_record(sp.slot_cost_cases()[0])
    assert record["status"] == "unavailable"
    assert record["tolerance"] == 0.0
    for key in ("wall_ms", "comm_total_bytes", "comm_per_element", "within_tolerance"):
        assert record[key] is None


# --------------------------------------------------------------------------
# 结论算术
# --------------------------------------------------------------------------


def _records(per_element: dict[str, float | None]) -> list[dict]:
    records = []
    for case in sp.slot_cost_cases():
        record = sp._blank_record(case)
        value = per_element.get(case.variant)
        record["status"] = "ok" if value is not None else "unavailable"
        record["comm_total_bytes"] = value * case.elements if value is not None else None
        record["comm_per_element"] = value
        records.append(record)
    return records


def test_summary_reports_all_four_prices_and_the_ratio():
    summary = sp.summarize_slot_cost(
        _records({"mul_only": 0.0, "mask_only": 624.0, "shift_only": 1888.0, "shift_mask": 1424.0})
    )
    assert summary["per_element_bytes"] == {
        "mul_only": 0.0,
        "mask_only": 624.0,
        "shift_only": 1888.0,
        "shift_mask": 1424.0,
    }
    extraction = summary["extraction"]
    assert extraction["bytes_per_element"] == 1424.0
    assert extraction["pure_multiply_bytes_per_element"] == sp.PURE_MUL_BYTES_PER_ELEMENT == 16.0
    assert extraction["times_pure_multiply"] == 89.0
    assert "89 倍" in extraction["reading"]
    assert "不含跨元素归约" in extraction["caveat"]


def test_summary_refuses_to_use_the_free_variant_as_a_baseline():
    """mul_only 实测 0 B：拿它当基准会把倍率算成无穷大，口径必须是 P6 的 16 B/元素。"""

    summary = sp.summarize_slot_cost(_records({"mul_only": 0.0, "shift_mask": 1424.0}))
    assert summary["mul_only_is_free"] is True
    assert summary["extraction"]["pure_multiply_bytes_per_element"] == 16.0
    assert "不能当基准" in summary["extraction"]["baseline_source"]


def test_summary_flags_a_non_free_mul_only():
    summary = sp.summarize_slot_cost(_records({"mul_only": 8.0, "shift_mask": 1424.0}))
    assert summary["mul_only_is_free"] is False
    assert any("需要复核" in note for note in summary["notes"])


def test_summary_returns_none_and_explains_when_extraction_missing():
    summary = sp.summarize_slot_cost(_records({"mul_only": 0.0}))
    assert summary["extraction"] is None
    assert summary["notes"] == ["没有取槽步（shift_mask）的通信量，无法与纯乘法比价"]


def test_summary_notes_incomplete_ab_decomposition():
    summary = sp.summarize_slot_cost(_records({"shift_mask": 1424.0}))
    assert len(summary["notes"]) == 2  # mask_only / shift_only 都缺
    assert all("A/B 拆解不完整" in note for note in summary["notes"])


def test_ratio_is_derived_not_hardcoded():
    summary = sp.summarize_slot_cost(_records({"shift_mask": 32.0}))
    assert summary["extraction"]["times_pure_multiply"] == 2.0
    assert "2 倍" in summary["extraction"]["reading"]


# --------------------------------------------------------------------------
# 产物
# --------------------------------------------------------------------------


def test_json_writer_round_trips(tmp_path):
    records = _records({"mul_only": 0.0, "mask_only": 624.0, "shift_only": 1888.0, "shift_mask": 1424.0})
    path = sp.write_slot_cost_json(records, str(tmp_path / "slot.json"))
    assert json.loads(open(path, encoding="utf-8").read()) == records


def test_csv_writer_uses_fixed_columns(tmp_path):
    records = _records({"mul_only": 0.0, "shift_mask": 1424.0})
    records[0]["comm_by_primitive"] = {"and": {"send_bytes": 79872}}
    path = sp.write_slot_cost_csv(records, str(tmp_path / "slot.csv"))
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == sp.SLOT_COST_CSV_COLUMNS
    assert len(rows) == len(records) + 1
    row = dict(zip(rows[0], rows[1]))
    assert row["name"] == "mul_only"
    assert json.loads(row["comm_by_primitive"])["and"]["send_bytes"] == 79872


def test_format_summary_prints_prices_and_ratio():
    text = sp.format_slot_cost_summary(
        _records({"mul_only": 0.0, "mask_only": 624.0, "shift_only": 1888.0, "shift_mask": 1424.0})
    )
    assert "B/元素" in text
    assert "取槽步 1424 B/元素" in text and "89 倍" in text
    assert "shift_right_arithmetic" not in text  # 没有产物字段时不编原语名


def test_format_summary_still_explains_missing_data():
    assert "说明：" in sp.format_slot_cost_summary([])


# --------------------------------------------------------------------------
# 真机（有 SPU 才跑）
# --------------------------------------------------------------------------


@requires_spu
def test_real_bit_ops_cost_far_more_than_a_multiply():
    """核心结论：取槽步的单价远高于纯乘法，且三个变体都能算对。"""

    records = sp.run_slot_cost_cases(sp.slot_cost_cases(elements=64))
    by_name = {record["name"]: record for record in records}
    assert all(record["status"] == "ok" for record in records), by_name
    assert all(record["within_tolerance"] is True for record in records), by_name

    assert by_name["mul_only"]["comm_total_bytes"] == 0  # 秘密 × 公开常数：本地线性运算
    assert by_name["mask_only"]["comm_total_bytes"] > 0
    assert by_name["shift_mask"]["comm_total_bytes"] > by_name["mask_only"]["comm_total_bytes"]
    assert "shift_right_arithmetic" in by_name["shift_only"]["comm_by_primitive"]

    summary = sp.summarize_slot_cost(records)
    assert summary["mul_only_is_free"] is True
    assert summary["extraction"]["times_pure_multiply"] > 1  # 取槽比纯乘法贵
    assert summary["notes"] == []


@requires_spu
def test_real_shift_mask_is_the_correct_packing_extraction():
    record = sp.run_slot_cost_case(sp.SlotCostCase(variant="shift_mask", elements=64))
    assert record["status"] == "ok", record["error"]
    assert record["within_tolerance"] is True
    assert record["max_abs_error"] == 0.0
    assert record["comm_per_element"] == record["comm_total_bytes"] / 64
