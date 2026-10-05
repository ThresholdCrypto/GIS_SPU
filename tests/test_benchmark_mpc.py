# -*- coding: utf-8 -*-
"""MPC 性能基线：生成器 / 记录诚实性 / 预测↔实测对账 / 产物。

与 PSI 基线（tests/test_benchmark.py）同构；差别在**对账轴**：
planner 预测的是结构代价（四量 + 位宽 b(K)），所以这里只对"位宽 ↔ 环宽"
下结论，不给出"预测时间 ↔ 实测时间"的误差（planner 不预测时间）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from backends.spu_backend import (
    EXPECT_DEVIATION,
    FIELD_BITS,
    MIXED_STATUS,
    MPC_BENCHMARK_OPS,
    SPU_FIELDS,
    SPU_PROTOCOLS,
    MpcBenchmarkCase,
    format_summary,
    predicted_bit_width,
    predicted_cost,
    reconcile_bit_width,
    run_benchmark_case,
    summarize_repeats,
    standard_cases,
    unexpected_records,
    write_benchmark_csv,
    write_benchmark_json,
)
from tests._helpers import has_spu

requires_spu = pytest.mark.skipif(
    not has_spu(), reason="需要真实 SPU 执行能力（Linux/WSL2 + spu + jax 0.4.34）"
)


# --------------------------------------------------------------------------
# 输入构造：确定性（无随机数）
# --------------------------------------------------------------------------


class TestInputBuilders:
    @pytest.mark.parametrize("op", MPC_BENCHMARK_OPS)
    def test_builder_is_deterministic(self, op):
        from backends.spu_backend.benchmark import _INPUT_BUILDERS

        builder = _INPUT_BUILDERS[op]
        first, _ = builder(64)
        second, _ = builder(64)
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert np.array_equal(left, right), op

    @pytest.mark.parametrize("op", MPC_BENCHMARK_OPS)
    def test_builder_returns_ints_and_a_callable(self, op):
        from backends.spu_backend.benchmark import _INPUT_BUILDERS

        args, reference = _INPUT_BUILDERS[op](16)
        assert callable(reference)
        assert all(np.issubdtype(np.asarray(a).dtype, np.integer) for a in args), op

    def test_temporal_overlap_nodes_stay_inside_the_14_bit_field(self):
        """Toff + 2^Lt 必须留在 14 位域内，否则输入本身就非法。"""

        from backends.spu_backend.benchmark import _temporal_overlap_inputs

        args, _ = _temporal_overlap_inputs(64)
        toff, lt = np.asarray(args[0]), np.asarray(args[1])
        assert np.all(toff + (1 << lt) <= (1 << 14)), "节点超出 Toff 14 位域"


# --------------------------------------------------------------------------
# planner 侧：位宽预测与对账
# --------------------------------------------------------------------------


class TestPredictedCost:
    def test_predicted_cost_carries_the_four_quantities(self):
        cost = predicted_cost("DistanceLE", 256)
        assert set(cost) >= {"N_ct", "b", "d", "R"}

    def test_weighted_sum_bit_width_follows_its_formula(self):
        # b(K) = 8 + 8 + ceil(log2 K)
        for k, expected in ((1, 16), (8, 19), (256, 24), (1024, 26)):
            assert predicted_bit_width("WeightedSum", k) == expected, k

    def test_fixed_width_ops_report_a_constant(self):
        assert predicted_bit_width("DistanceLE", 4096) == 8
        assert predicted_bit_width("TemporalOverlap", 4096) == 18

    def test_reconcile_marks_covered_and_uncovered(self):
        covered = reconcile_bit_width("WeightedSum", 256, "FM32")
        assert covered == {"predicted_bits": 24, "field_bits": 32, "covered": True}

        uncovered = reconcile_bit_width("WeightedSum", 1 << 20, "FM32")
        assert uncovered["predicted_bits"] == 36
        assert uncovered["covered"] is False

    def test_unknown_op_is_loud(self):
        with pytest.raises(Exception):
            predicted_cost("NotAnOp", 16)


# --------------------------------------------------------------------------
# 用例集策略
# --------------------------------------------------------------------------


class TestCaseSet:
    def test_case_label_is_readable(self):
        assert MpcBenchmarkCase(op="DistanceLE", protocol="ABY3").label() == (
            "DistanceLE ABY3 FM64 K=1024"
        )

    def test_standard_cases_cover_every_protocol_for_every_op(self):
        cases = standard_cases()
        pairs = {(c.op, c.protocol) for c in cases}
        for op in MPC_BENCHMARK_OPS:
            for protocol in SPU_PROTOCOLS:
                assert (op, protocol) in pairs, (op, protocol)

    def test_standard_cases_cover_every_field(self):
        fields = {c.field for c in standard_cases()}
        assert fields == set(SPU_FIELDS)

    def test_every_case_has_a_known_op_and_protocol(self):
        for case in standard_cases():
            assert case.op in MPC_BENCHMARK_OPS, case
            assert case.protocol in SPU_PROTOCOLS, case
            assert case.field in SPU_FIELDS, case

    def test_temporal_overlap_scale_is_capped_for_its_quadratic_circuit(self):
        """TemporalOverlap 是 N×M 两两比较；节点数上界必须显著低于其他算子。"""

        largest = max(c.k for c in standard_cases() if c.op == "TemporalOverlap")
        assert largest <= 32, f"TemporalOverlap 节点数上限过大：{largest}"

    def test_bit_width_probe_row_is_not_executed_and_says_why(self):
        probes = [c for c in standard_cases() if not c.execute]
        assert probes, "应有一条越界组合的占位行"
        for case in probes:
            assert case.unavailable_note.strip(), case.label()
            predicted = predicted_bit_width(case.op, case.k)
            assert predicted > FIELD_BITS[case.field], case.label()

    def test_quick_is_a_subset_of_standard(self):
        quick = {(c.op, c.protocol, c.field, c.k) for c in standard_cases(quick=True)}
        full = {(c.op, c.protocol, c.field, c.k) for c in standard_cases()}
        assert quick <= full
        assert quick, "quick 模式不能是空集"

    def test_known_deviation_rows_are_declared(self):
        """两处已知偏差都登记：FM32 环宽下限（硬失败）+ 大 K 的近似除法（非确定）。"""

        hard = [c for c in standard_cases() if c.expect_status == "error"]
        soft = [c for c in standard_cases() if c.expect_status == EXPECT_DEVIATION]
        assert hard and soft, "两类偏差都要有登记行"
        for case in hard + soft:
            assert case.op == "WeightedSum", case.label()

        assert [c.field for c in hard] == ["FM32"], "硬失败只该是 FM32 那条环宽下限"
        assert all(c.field == "FM64" and c.k >= 1024 for c in soft), "非确定偏差是 FM64 大 K"

    def test_expected_rows_only_cover_ops_we_have_evidence_for(self):
        """不允许给没实测过的算子随手加预期失败——那是掩盖而非登记。"""

        declared = [c for c in standard_cases() if c.expect_status]
        assert {c.op for c in declared} == {"WeightedSum"}
        for case in declared:
            assert case.field in SPU_FIELDS, case.label()

    def test_deviation_status_is_neither_ok_nor_error(self):
        """`deviation` 是个独立标记：它既不能写成 ok，也不能写成 error。

        硬钉 `error` 会把基线做成抽奖（同一份代码两次运行退出码不同），
        写成 `ok` 又等于宣布一个没验证过的结论。
        """

        assert EXPECT_DEVIATION not in ("ok", "error", "")


# --------------------------------------------------------------------------
# 记录：诚实性
# --------------------------------------------------------------------------


class TestRecordHonesty:
    def test_schema_is_stable(self):
        record = run_benchmark_case(
            MpcBenchmarkCase(op="DistanceLE", protocol="ABY3", k=16)
        )
        assert set(record) >= {
            "case", "op", "protocol", "field", "k",
            "predicted_cost", "predicted_bits", "field_bits", "bit_width_covered",
            "status", "note", "error", "wall_ms", "memory_mb", "peak_rss_mb",
            "pphlo_bytes", "agreement", "max_abs_error", "tolerance",
        }

    def test_record_is_jsonable(self):
        record = run_benchmark_case(
            MpcBenchmarkCase(op="WeightedSum", protocol="ABY3", k=16)
        )
        json.dumps(record, ensure_ascii=False)

    def test_non_executed_case_fills_no_numbers(self):
        record = run_benchmark_case(
            MpcBenchmarkCase(
                op="WeightedSum", protocol="ABY3", field="FM32", k=1 << 20,
                execute=False, unavailable_note="越界",
            )
        )
        assert record["status"] == "unavailable"
        assert record["wall_ms"] is None
        assert record["agreement"] is None
        assert record["memory_mb"] is None
        # 预测侧仍然有值——对账本来就是"预测 vs 实测"，实测空缺要显眼
        assert record["predicted_bits"] == 36
        assert record["bit_width_covered"] is False

    def test_unregistered_op_is_an_error_not_a_crash(self):
        record = run_benchmark_case(MpcBenchmarkCase(op="Nope", protocol="ABY3"))
        assert record["status"] == "error"
        assert "未登记" in record["error"]


# --------------------------------------------------------------------------
# unexpected_records
# --------------------------------------------------------------------------


class TestUnexpectedRecords:
    def test_ok_and_unavailable_are_expected(self):
        rows = [{"status": "ok", "agreement": True}, {"status": "unavailable"}]
        assert unexpected_records(rows) == []

    def test_error_without_expectation_is_unexpected(self):
        rows = [{"status": "error", "case": "x"}]
        assert [r["case"] for r in unexpected_records(rows)] == ["x"]

    def test_expectation_is_honored(self):
        rows = [{"status": "error", "expect_status": "error", "case": "x"}]
        assert unexpected_records(rows) == []

    def test_expected_failure_that_succeeded_is_unexpected(self):
        """上游修好除法后，这里必须报出来提示更新文档。"""

        rows = [{"status": "ok", "expect_status": "error", "case": "x"}]
        assert [r["case"] for r in unexpected_records(rows)] == ["x"]

    def test_deviation_rows_accept_both_outcomes(self):
        """非确定偏差行：偏了、恰好对上，都算符合预期；没跑起来仍算预期外。"""

        ok = [{"status": "ok", "expect_status": EXPECT_DEVIATION, "case": "a"}]
        err = [{"status": "error", "expect_status": EXPECT_DEVIATION, "case": "b"}]
        no = [{"status": "unavailable", "expect_status": EXPECT_DEVIATION, "case": "c"}]
        assert unexpected_records(ok) == []
        assert unexpected_records(err) == []
        assert [r["case"] for r in unexpected_records(no)] == ["c"]

    def test_ok_but_disagreeing_with_plain_is_unexpected(self):
        rows = [{"status": "ok", "agreement": False, "case": "x"}]
        assert [r["case"] for r in unexpected_records(rows)] == ["x"]


# --------------------------------------------------------------------------
# 重复实验：统计口径
# --------------------------------------------------------------------------


def _row(**overrides):
    row = {
        "case": "X", "op": "WeightedSum", "protocol": "ABY3", "field": "FM64", "k": 8,
        "status": "ok", "note": "", "error": "", "wall_ms": 100.0,
        "agreement": True, "max_abs_error": 0.0, "repeat": 1,
    }
    row.update(overrides)
    return row


def test_summarize_repeats_needs_at_least_one_record():
    with pytest.raises(ValueError):
        summarize_repeats([])


class TestRepeatSummary:
    def test_all_ok_is_still_ok_with_quantiles(self):
        rows = [_row(wall_ms=w) for w in (90.0, 100.0, 110.0, 120.0, 130.0)]
        summary = summarize_repeats(rows)
        assert summary["status"] == "ok"
        assert summary["repeat"] == 5
        assert summary["status_counts"] == {"ok": 5}
        assert summary["agreement"] is True
        assert summary["deviation_rate"] == 0.0
        assert summary["wall_ms_stats"]["median"] == 110.0
        assert summary["wall_ms_stats"]["min"] == 90.0
        assert summary["wall_ms_stats"]["max"] == 130.0
        assert summary["wall_ms"] == 110.0  # 平面列取中位数
        assert len(summary["wall_ms_samples"]) == 5

    def test_disagreeing_runs_are_mixed_not_a_majority_vote(self):
        """2 成功 3 失败也是 mixed——多数票会把"不稳"说成"稳"。"""

        rows = [
            _row(status="ok"),
            _row(status="ok"),
            _row(status="error", error="超容差", agreement=None, max_abs_error=1.0),
            _row(status="error", error="超容差", agreement=None, max_abs_error=1.0),
            _row(status="error", error="超容差", agreement=None, max_abs_error=4.0),
        ]
        summary = summarize_repeats(rows)
        assert summary["status"] == MIXED_STATUS
        assert summary["status_counts"] == {"ok": 2, "error": 3}
        assert summary["agreement"] is None
        assert "不一致" in summary["note"]

    def test_deviation_rate_ignores_runs_that_never_produced_a_number(self):
        """崩掉的那些次不进分母——它们是"没测出来"，不是"没有偏差"。"""

        rows = [
            _row(max_abs_error=0.0),
            _row(status="error", error="超容差", agreement=None, max_abs_error=4.0),
            _row(status="error", error="崩溃", wall_ms=None, max_abs_error=None),
        ]
        summary = summarize_repeats(rows)
        assert summary["value_runs"] == 2
        assert summary["deviation_rate"] == 0.5
        assert summary["max_abs_error_max"] == 4.0
        assert summary["max_abs_error"] == 4.0

    def test_all_unavailable_stays_unavailable(self):
        rows = [
            _row(status="unavailable", wall_ms=None, agreement=None, max_abs_error=None)
            for _ in range(3)
        ]
        summary = summarize_repeats(rows)
        assert summary["status"] == "unavailable"
        assert summary["wall_ms"] is None
        assert summary["wall_ms_stats"] == {}
        assert summary["deviation_rate"] is None

    def test_mixed_is_only_expected_for_declared_deviation_rows(self):
        mixed = [{"status": MIXED_STATUS, "case": "a"}]
        assert [r["case"] for r in unexpected_records(mixed)] == ["a"]
        declared = [{"status": MIXED_STATUS, "expect_status": EXPECT_DEVIATION, "case": "b"}]
        assert unexpected_records(declared) == []

    def test_standard_cases_carry_repeats(self):
        assert all(case.repeats == 3 for case in standard_cases(repeats=3))
        assert all(case.repeats == 1 for case in standard_cases())
        # 非法值不许静默变成 0 次（那次记录会变成空统计）
        assert all(case.repeats == 1 for case in standard_cases(repeats=0))


# --------------------------------------------------------------------------
# 产物
# --------------------------------------------------------------------------


class TestWriters:
    def test_json_roundtrip(self, tmp_path):
        record = run_benchmark_case(
            MpcBenchmarkCase(op="DistanceLE", protocol="ABY3", k=16)
        )
        path = write_benchmark_json([record], str(tmp_path / "in" / "b.json"))
        with open(path, encoding="utf-8") as handle:
            assert json.load(handle) == [record]

    def test_csv_has_header_and_row(self, tmp_path):
        record = run_benchmark_case(
            MpcBenchmarkCase(op="DistanceLE", protocol="ABY3", k=16)
        )
        path = write_benchmark_csv([record], str(tmp_path / "b.csv"))
        lines = open(path, encoding="utf-8").read().strip().splitlines()
        assert lines[0].startswith("case,op,protocol,field,k")
        assert len(lines) == 2

    def test_summary_mentions_the_label_and_status(self):
        record = run_benchmark_case(
            MpcBenchmarkCase(op="DistanceLE", protocol="ABY3", k=16)
        )
        summary = format_summary([record])
        assert "DistanceLE ABY3 FM64 K=16" in summary
        assert record["status"] in summary  # 环境缺 SPU 时这里就是 unavailable


# --------------------------------------------------------------------------
# 真实执行（受支持环境）
# --------------------------------------------------------------------------


@requires_spu
class TestRealMpcBenchmark:
    def test_small_case_matches_plain_on_every_protocol(self):
        for protocol in SPU_PROTOCOLS:
            record = run_benchmark_case(
                MpcBenchmarkCase(op="DistanceLE", protocol=protocol, field="FM64", k=64)
            )
            assert record["status"] == "ok", (protocol, record["error"])
            assert record["agreement"] is True, protocol
            assert record["max_abs_error"] == 0.0, protocol
            assert record["wall_ms"] is not None

    def test_small_case_repeats_are_stable_and_deviation_free(self):
        """K=64 在噪声范围内：5 次都应与明文逐位一致。"""

        record = run_benchmark_case(
            MpcBenchmarkCase(
                op="WeightedSum", protocol="ABY3", field="FM64", k=64, repeats=5
            )
        )
        assert record["status"] == "ok", record["status_counts"]
        assert record["deviation_rate"] == 0.0
        assert record["wall_ms_stats"]["p25"] <= record["wall_ms"] <= record["wall_ms_stats"]["p75"]

    def test_weighted_sum_is_exact_at_small_accumulation(self):
        """K=256 累加量级下整数路径逐位一致——除法在该量级尚未显形。"""

        record = run_benchmark_case(
            MpcBenchmarkCase(op="WeightedSum", protocol="ABY3", field="FM64", k=256)
        )
        assert record["status"] == "ok", record["error"]
        assert record["agreement"] is True

    def test_fm32_rejects_division_based_weighted_sum(self):
        """对账结论：planner 按位宽预测 b(K=256)=24 ≤ 32"够用"，
        但生成代码里的 `//` 在 SPU 内部需要 64 位环——FM32 实测不可用。

        这条锁住"预测 ≠ 实测"的具体形态，防止有人只看位宽就下结论。
        """

        record = run_benchmark_case(
            MpcBenchmarkCase(op="WeightedSum", protocol="ABY3", field="FM32", k=256)
        )
        assert record["bit_width_covered"] is True, "预测层面本应判定为够用"
        assert record["status"] == "error", record
        assert "FM32" in record["error"] or "ring" in record["error"]

    def test_division_deviation_is_reproducible_in_shape(self):
        """K=4096：重复 5 次都真的跑出了数值，且偏差率被如实记下来。

        **不**断言 `status == "error"`：这条路径是非确定的（同一组合重跑可能
        恰好逐位一致，已实测到），硬钉一种结果会把测试做成抽奖。
        长期形态看 `docs/mpc_deviation_repeat.json` 的偏差率。
        """

        record = run_benchmark_case(
            MpcBenchmarkCase(
                op="WeightedSum", protocol="ABY3", field="FM64", k=4096,
                expect_status=EXPECT_DEVIATION, repeats=5,
            )
        )
        assert record["repeat"] == 5
        assert sum(record["status_counts"].values()) == 5
        assert record["value_runs"] == 5, "5 次都应当得出数值（没有崩）"
        assert record["status"] in ("ok", "error", MIXED_STATUS)
        assert 0.0 <= record["deviation_rate"] <= 1.0
