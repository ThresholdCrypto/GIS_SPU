# -*- coding: utf-8 -*-
"""打包收益上界探针（P6）：SPU 按**环元素**还是按**输入位**计费。

为什么单独一个文件
==================
位平面布局（D3，`planner/layout.py`）的条数模型有一个前提：**通信量由密文条数
（环元素个数）主导，输入位宽不影响单价**。P5 只能把这条前提标成"未验证"。
本文件锁住把前提变成实测的那几步，防止它悄悄回退成口号：

  实验 A（位宽）：int8 / int32 / int64 同一元素数 → 通信量之比应恒为 1；
  实验 B（规模）：512 / 1024 / 4096 元素 → B/元素 应恒定（通信量随元素数线性）；
  上界：由实测 B/元素 换算出"8 值/元素"的收益，并**同时**记录它不含槽内归约。

三条纪律（与仓库既有口径一致）：
  1. 探针算子必须是**秘密 × 秘密**——第一版用 `x * 2`（乘公开常数）实测恒为
     0 字节，那样的探针测不出任何东西，必须在测试里钉住；
  2. 空就是空：环境跑不了、算术缺输入时一律 `None` + 原因，不填推测值；
  3. 上界必须写明是上界（真实打包电路还要付槽内归约的通信量）。
"""

from __future__ import annotations

import csv
import json
import types

import numpy as np
import pytest

from backends.spu_backend import packing_probe as pp
from tests._helpers import has_spu

requires_spu = pytest.mark.skipif(
    not has_spu(), reason="需要真实 SPU 执行能力（Linux/WSL2 + spu + jax 0.4.34）"
)


# --------------------------------------------------------------------------
# 用例集
# --------------------------------------------------------------------------


def test_case_set_is_five_rows_and_names_unique():
    cases = pp.packing_probe_cases()
    assert len(cases) == 5
    names = [case.name for case in cases]
    assert len(set(names)) == len(names)


def test_case_set_covers_width_and_scale_sweeps():
    cases = {case.name: case for case in pp.packing_probe_cases()}
    widths = [cases[f"width-{dtype}"] for dtype in ("int8", "int32", "int64")]
    assert {case.elements for case in widths} == {pp._WIDTH_SWEEP_ELEMENTS}
    assert [case.dtype for case in widths] == ["int8", "int32", "int64"]

    scales = [cases[f"scale-{n}"] for n in pp._SCALE_SWEEP_ELEMENTS]
    assert [case.elements for case in scales] == list(pp._SCALE_SWEEP_ELEMENTS)
    assert {case.dtype for case in scales} == {"int64"}
    # int64 @ 4096 是位宽扫描与规模扫描的公共点，不该再单开一条 "scale-4096"
    assert "scale-4096" not in cases


def test_repeats_propagate_to_every_case():
    cases = pp.packing_probe_cases(repeats=3)
    assert {case.repeats for case in cases} == {3}
    assert {case.repeats for case in pp.packing_probe_cases(repeats=0)} == {1}


def test_labels_are_stable_and_mention_op():
    cases = pp.packing_probe_cases()
    assert cases[0].label() == f"{pp.PROBE_OP} int8 N={pp._WIDTH_SWEEP_ELEMENTS}"
    assert all(pp.PROBE_OP in case.label() for case in cases)


def test_every_case_uses_capture_comm_by_default():
    """不采通信量就回答不了"按元素还是按位计费"，默认必须开着。"""

    assert {case.capture_comm for case in pp.packing_probe_cases()} == {True}


# --------------------------------------------------------------------------
# 探针算子与输入
# --------------------------------------------------------------------------


def test_probe_op_is_secret_times_secret_not_public_constant():
    """第一版用了秘密 × 公开常数，实测恒为 0 B——这里钉住不许回退。"""

    import jax.numpy as jnp

    left = jnp.asarray(np.array([2, 3], dtype=np.int64))
    right = jnp.asarray(np.array([5, 7], dtype=np.int64))
    assert [int(v) for v in np.asarray(pp._probe_fn(left, right))] == [10, 21]

    case = pp.PackingProbeCase(name="t", dtype="int64", elements=2)
    with pytest.raises(TypeError):
        pp._probe_fn(pp._build_inputs(case)[0])  # 单参数调用不存在


def test_inputs_are_deterministic_and_signed_safe():
    small = pp.PackingProbeCase(name="t", dtype="int8", elements=512)
    left, right = pp._build_inputs(small)
    again_left, again_right = pp._build_inputs(small)
    assert np.array_equal(left, again_left) and np.array_equal(right, again_right)
    assert left.dtype == np.int8 and right.dtype == np.int8
    assert left.min() >= 1 and left.max() <= pp._PROBE_VALUE_MAX
    assert right.min() >= 1 and right.max() <= pp._PROBE_VALUE_MAX
    # 两路输入不能相同（否则退化成乘自己，失去"两方各持一路"的语义）
    assert not np.array_equal(left, right)


def test_int8_products_do_not_overflow():
    left, right = pp._build_inputs(
        pp.PackingProbeCase(name="t", dtype="int8", elements=pp._WIDTH_SWEEP_ELEMENTS)
    )
    products = left.astype(np.int64) * right.astype(np.int64)
    assert products.max() < np.iinfo(np.int8).max


def test_input_values_are_independent_of_element_count():
    """规模扫描只允许变元素数：前 n 个值与位宽扫描行应一致（可比性前提）。"""

    head = pp._build_inputs(
        pp.PackingProbeCase(name="a", dtype="int64", elements=pp._WIDTH_SWEEP_ELEMENTS)
    )[0]
    tail = pp._build_inputs(pp.PackingProbeCase(name="b", dtype="int64", elements=512))[0]
    assert np.array_equal(head[: len(tail)], tail)


def test_reference_matches_probe_semantics():
    left, right = pp._build_inputs(pp.PackingProbeCase(name="t", dtype="int8", elements=64))
    reference = pp._reference(left, right)
    assert reference.dtype == np.int64
    assert np.array_equal(reference, left.astype(np.int64) * right.astype(np.int64))


# --------------------------------------------------------------------------
# 环境不具备时的如实占位
# --------------------------------------------------------------------------


def test_unrunnable_environment_records_unavailable_without_numbers():
    blocked = types.SimpleNamespace(runnable=False, blockers=("stub",))
    record = pp._run_probe_once(pp.packing_probe_cases()[0], report=blocked)
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


def test_unknown_dtype_is_an_error_not_a_silent_fallback():
    record = pp._run_probe_once(
        pp.PackingProbeCase(name="t", dtype="int24", elements=8),
        report=types.SimpleNamespace(runnable=True, blockers=()),
    )
    assert record["status"] == "error"
    assert "int24" in record["error"]


def test_blank_record_leaves_every_number_empty():
    record = pp._blank_record(pp.packing_probe_cases()[0])
    assert record["status"] == "unavailable"
    for key in pp.PACKING_CSV_COLUMNS:
        assert record[key] in ("", None, False, {}, 0.0) or key in (
            "case",
            "name",
            "op",
            "protocol",
            "field",
            "dtype",
            "elements",
            "repeat",
            "status",
            "note",
            "error",
        )
    assert record["tolerance"] == 0.0


# --------------------------------------------------------------------------
# 结论算术
# --------------------------------------------------------------------------


def _records(**comm_by_name: float) -> list[dict]:
    """按标准用例集的 name 造一批"只有通信量"的假记录。"""

    records = []
    for case in pp.packing_probe_cases():
        record = pp._blank_record(case)
        record["status"] = "ok"
        record["comm_total_bytes"] = comm_by_name.get(case.name)
        record["comm_per_element"] = (
            record["comm_total_bytes"] / case.elements
            if record["comm_total_bytes"] is not None
            else None
        )
        records.append(record)
    return records


def _linear_records(per_element: float = 16.0) -> list[dict]:
    return _records(
        **{
            case.name: per_element * case.elements
            for case in pp.packing_probe_cases()
        }
    )


def test_summary_detects_per_element_pricing():
    summary = pp.summarize_packing(_linear_records())
    assert summary["width_ratios"] == {"int8": 1.0, "int32": 1.0, "int64": 1.0}
    assert summary["pricing_per_element"] is True
    assert "按**环元素**计费" in summary["width_effect"]
    assert summary["notes"] == []


def test_summary_flags_width_dependent_pricing():
    records = _linear_records()
    for record in records:
        if record["name"] == "width-int8":
            record["comm_total_bytes"] *= 0.5  # 位宽越窄越便宜 → 打包无收益
    summary = pp.summarize_packing(records)
    assert summary["pricing_per_element"] is False
    assert "与输入位宽有关" in summary["width_effect"]


def test_summary_reports_linearity_by_scale():
    summary = pp.summarize_packing(_linear_records(per_element=16.0))
    linearity = summary["linearity"]
    assert linearity["bytes_per_element_by_scale"] == {"512": 16.0, "1024": 16.0, "4096": 16.0}
    assert linearity["relative_spread"] == 0.0
    assert linearity["linear"] is True


def test_summary_marks_non_linear_scaling():
    records = _linear_records()
    for record in records:
        if record["name"] == "scale-512":
            record["comm_total_bytes"] *= 4.0  # 小规模单价高 → 非线性
    assert pp.summarize_packing(records)["linearity"]["linear"] is False


def test_packing_upper_bound_math_and_caveat():
    summary = pp.summarize_packing(_linear_records(per_element=16.0))
    bound = summary["packing_upper_bound"]
    assert bound["values_per_element_packed"] == pp.PACKED_VALUES_PER_ELEMENT == 8
    assert bound["element_ratio"] == 8.0
    assert bound["bytes_per_element"] == 16.0
    assert bound["comm_pointwise_bytes"] == 16.0 * pp._WIDTH_SWEEP_ELEMENTS
    assert bound["comm_packed_predicted_bytes"] == 16.0 * (pp._WIDTH_SWEEP_ELEMENTS // 8)
    assert bound["comm_packed_measured_bytes"] == 16.0 * 512
    assert bound["prediction_matches_measurement"] is True
    assert bound["comm_ratio_upper"] == 8.0
    assert "上界" in bound["caveat"] and "槽内归约" in bound["caveat"]
    assert "不含槽内归约" in bound["derivation"]


def test_packing_upper_bound_compares_same_payload_not_eight_times_it():
    """逐点行承载 4096 个值，打包行也是 4096 个值——不是 32768。"""

    bound = pp.summarize_packing(_linear_records())["packing_upper_bound"]
    assert bound["values_compared"].startswith(f"{pp._WIDTH_SWEEP_ELEMENTS} 个 8 位值")
    assert "32768" not in bound["values_compared"]


def test_summary_returns_none_and_explains_when_comm_missing():
    summary = pp.summarize_packing([])
    assert summary["width_effect"] is None
    assert summary["pricing_per_element"] is None
    assert summary["linearity"] is None
    assert summary["packing_upper_bound"] is None
    assert len(summary["notes"]) == 3


def test_zero_comm_yields_no_ratios():
    """REF2K 那类"不通信"的配置：不能除以 0 编出一个比值，也不能判成"按元素"。"""

    summary = pp.summarize_packing(
        _records(**{case.name: 0.0 for case in pp.packing_probe_cases()})
    )
    assert summary["width_ratios"] == {}
    assert summary["pricing_per_element"] is None
    assert summary["linearity"] is None  # 0 B/元素 不是"线性"，是"没数"
    assert summary["packing_upper_bound"] is None
    assert summary["notes"]


def test_measured_packed_row_disagreeing_with_prediction_is_visible():
    records = _linear_records()
    for record in records:
        if record["name"] == "scale-512":
            record["comm_total_bytes"] *= 2.0
    bound = pp.summarize_packing(records)["packing_upper_bound"]
    assert bound["prediction_matches_measurement"] is False


# --------------------------------------------------------------------------
# 产物
# --------------------------------------------------------------------------


def test_json_writer_round_trips(tmp_path):
    records = _linear_records()
    path = pp.write_packing_probe_json(records, str(tmp_path / "probe.json"))
    assert json.loads(open(path, encoding="utf-8").read()) == records


def test_csv_writer_uses_fixed_columns_and_serializes_dicts(tmp_path):
    records = _linear_records()
    records[0]["comm_by_primitive"] = {"multiply": {"send_bytes": 32768}}
    path = pp.write_packing_probe_csv(records, str(tmp_path / "probe.csv"))
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == pp.PACKING_CSV_COLUMNS
    assert len(rows) == len(records) + 1
    row = dict(zip(rows[0], rows[1]))
    assert row["name"] == "width-int8"
    assert json.loads(row["comm_by_primitive"])["multiply"]["send_bytes"] == 32768
    assert row["within_tolerance"] == ""


def test_csv_value_handles_none_bool_and_dict():
    assert pp._csv_value(None) == ""
    assert pp._csv_value(True) == "true"
    assert pp._csv_value(False) == "false"
    assert json.loads(pp._csv_value({"a": 1})) == {"a": 1}
    assert pp._csv_value(3) == 3


def test_format_summary_prints_table_and_conclusions():
    text = pp.format_packing_summary(_linear_records())
    assert "B/元素" in text
    assert "mul_ss int64 N=512" in text
    assert "位宽结论" in text and "规模结论" in text and "打包上界" in text


def test_format_summary_still_works_without_any_number():
    text = pp.format_packing_summary([])
    assert "B/元素" in text
    assert "说明：" in text  # 缺数字要给原因，而不是留一句结论


# --------------------------------------------------------------------------
# 真机（有 SPU 才跑）
# --------------------------------------------------------------------------


@requires_spu
def test_real_probe_prices_by_ring_element_not_by_input_width():
    """核心前提：同一元素数下，int8 与 int64 的通信量必须相等且非 0。"""

    cases = [
        pp.PackingProbeCase(name="width-int8", dtype="int8", elements=64),
        pp.PackingProbeCase(name="width-int64", dtype="int64", elements=64),
    ]
    records = pp.run_packing_probe_cases(cases)
    int8_record, int64_record = records
    assert int8_record["status"] == "ok", int8_record["error"]
    assert int64_record["status"] == "ok", int64_record["error"]
    assert int8_record["within_tolerance"] is True
    assert int64_record["comm_total_bytes"] > 0
    assert int8_record["comm_total_bytes"] == int64_record["comm_total_bytes"]
    assert int8_record["comm_per_element"] == int64_record["comm_total_bytes"] / 64


@requires_spu
def test_real_probe_repeats_report_median_and_stats():
    case = pp.PackingProbeCase(name="scale-512", dtype="int64", elements=512, repeats=3)
    record = pp.run_packing_probe_case(case)
    assert record["status"] == "ok", record["error"]
    assert record["repeat"] == 3
    stats = record["comm_total_bytes_stats"]
    assert stats["min"] <= stats["median"] <= stats["max"]
    assert record["comm_total_bytes"] == stats["median"]


@requires_spu
def test_real_probe_summary_lands_on_the_measured_premise():
    records = pp.run_packing_probe_cases(
        [
            case
            for case in pp.packing_probe_cases()
            if case.name in {"width-int8", "width-int64", "scale-512"}
        ]
    )
    summary = pp.summarize_packing(records)
    assert summary["pricing_per_element"] is True
    bound = summary["packing_upper_bound"]
    assert bound["comm_ratio_upper"] == pytest.approx(bound["element_ratio"], rel=0.05)
