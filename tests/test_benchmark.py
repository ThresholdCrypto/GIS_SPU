# -*- coding: utf-8 -*-
"""benchmark 模块测试：生成器确定性/合法性、记录 schema、unavailable 诚实规则。

真实 PSI 只跑两条小规模用例（RR22, N=256），其余围绕纯函数与写入器。
"""

from __future__ import annotations

import json

import pytest

from backends.psi_backend.benchmark import (
    BenchmarkCase,
    analyze_benchmark_set,
    compare_with_baseline,
    format_n,
    format_summary,
    generate_benchmark_sets,
    plain_intersects_reference,
    run_benchmark_case,
    run_benchmark_cases,
    standard_cases,
    unexpected_records,
    write_benchmark_csv,
    write_benchmark_json,
)
from ir import decode_grid_code

from tests._helpers import has_psi

needs_psi = pytest.mark.skipif(
    not has_psi(), reason="当前环境不具备真实 PSI 执行能力(python3.11+spu+libgomp1)"
)

#: 任务文档 §7.3 要求的记录字段（benchmark 记录必须是它的超集）
SCHEMA_KEYS = (
    "protocol",
    "low_comm_mode",
    "n_left",
    "n_right",
    "intersection_ratio",
    "psi_execute_ms",
    "total_ms",
    "memory_mb",
    "agreement",
)


class TestGenerator:
    def test_deterministic(self):
        first = generate_benchmark_sets(512, intersection_ratio=0.25)
        second = generate_benchmark_sets(512, intersection_ratio=0.25)
        assert first.left == second.left
        assert first.right == second.right
        assert dict(first.meta) == dict(second.meta)

    def test_basic_shape(self):
        bset = generate_benchmark_sets(1024, intersection_ratio=0.5)
        analysis = analyze_benchmark_set(bset)
        assert analysis["ok"], analysis
        assert analysis["intersection_count"] == 512
        assert bset.n_left == 1024
        assert bset.n_right == 1024
        assert bset.meta["layout_id"].startswith("geosot3d-")

    def test_intersection_ratio_endpoints(self):
        disjoint = generate_benchmark_sets(256, intersection_ratio=0.0)
        assert analyze_benchmark_set(disjoint)["intersection_count"] == 0
        identical = generate_benchmark_sets(256, intersection_ratio=1.0)
        analysis = analyze_benchmark_set(identical)
        assert analysis["ok"]
        assert analysis["intersection_count"] == 256

    def test_duplicates_are_inserted_not_deduped(self):
        bset = generate_benchmark_sets(512, duplicate_ratio=0.25)
        analysis = analyze_benchmark_set(bset)
        assert analysis["ok"]
        assert analysis["duplicate_count_left"] == 128
        assert analysis["duplicate_count_right"] == 128
        assert analysis["duplicate_count_declared"] == 128
        assert bset.n_left == 512 + 128
        assert bset.n_right == 512 + 128

    def test_high_bits_are_legal_and_above_2_pow_63(self):
        bset = generate_benchmark_sets(64, high_bits=True)
        assert all(code >= 2**63 for code in bset.left)
        assert all(code >= 2**63 for code in bset.right)
        for code in bset.left[:4] + bset.right[:4]:
            fields = decode_grid_code(code)
            assert fields["X"] >= 2**16
            assert fields["L"] == 9
            assert fields["Z"] == 15

    def test_level_and_z_passthrough(self):
        bset = generate_benchmark_sets(8, level=6, z=5)
        fields = decode_grid_code(bset.left[0])
        assert fields["L"] == 6
        assert fields["Z"] == 5

    def test_non_power_size(self):
        bset = generate_benchmark_sets(1000, intersection_ratio=0.1)
        analysis = analyze_benchmark_set(bset)
        assert analysis["ok"]
        assert analysis["intersection_count"] == 100

    @pytest.mark.parametrize(
        "kwargs",
        (
            {"n": 0},
            {"n": -3},
            {"n": 16, "intersection_ratio": 1.5},
            {"n": 16, "intersection_ratio": -0.1},
            {"n": 16, "duplicate_ratio": 1.0},
            {"n": 16, "duplicate_ratio": -0.1},
        ),
    )
    def test_invalid_params_rejected(self, kwargs):
        with pytest.raises(ValueError):
            generate_benchmark_sets(**kwargs)


class TestCaseAndRecords:
    def test_case_label(self):
        assert BenchmarkCase("PROTOCOL_RR22", True, 1024).label() == "RR22+low_comm N=2^10"
        label = BenchmarkCase("PROTOCOL_KKRT", False, 4096, receiver_rank=1).label()
        assert "KKRT" in label and "recv=1" in label
        assert BenchmarkCase("PROTOCOL_ECDH", False, 1024, high_bits=True).label().endswith("high_bits")

    def test_format_n(self):
        assert format_n(1024) == "2^10"
        assert format_n(2**24) == "2^24"
        assert format_n(1000) == "1000"

    def test_unavailable_record_is_honest(self):
        case = BenchmarkCase("PROTOCOL_ECDH", False, 2**22, execute=False, unavailable_note="预算之外")
        record = run_benchmark_case(case)
        assert record["status"] == "unavailable"
        assert record["note"] == "预算之外"
        assert record["psi_execute_ms"] is None
        assert record["total_ms"] is None
        assert record["memory_mb"] is None
        assert record["agreement"] is None

    def test_record_schema_keys(self):
        record = run_benchmark_case(BenchmarkCase("PROTOCOL_ECDH", n=2**10, execute=False))
        for key in SCHEMA_KEYS:
            assert key in record, key
        assert record["note"], "unavailable 记录必须说明原因，不能空着"

    def test_standard_quick_cases_are_all_executable(self):
        cases = standard_cases(quick=True)
        assert len(cases) == 8
        assert all(case.execute for case in cases)
        assert {case.n for case in cases} == {2**10, 2**12}

    def test_standard_cases_policy(self):
        cases = standard_cases()
        labels = [case.label() for case in cases]
        assert len(labels) == len(set(labels)), "用例标签必须唯一"

        unavailable = [case for case in cases if not case.execute]
        assert {case.n for case in unavailable} == {2**22, 2**24}
        assert all(case.protocol == "PROTOCOL_ECDH" for case in unavailable)
        assert all(case.unavailable_note for case in unavailable)

        measured = {(case.protocol, case.low_comm_mode, case.n) for case in cases if case.execute}
        assert ("PROTOCOL_RR22", False, 2**24) in measured
        assert ("PROTOCOL_RR22", True, 2**24) in measured
        assert ("PROTOCOL_KKRT", False, 2**24) in measured
        assert ("PROTOCOL_ECDH", False, 2**20) in measured
        # ECDH 的 2^22/2^24 不允许出现在"实测"集合里
        assert not any(
            protocol == "PROTOCOL_ECDH" and n > 2**20 for protocol, _, n in measured
        )

        variant_text = "\n".join(labels)
        for marker in ("intr=0.0", "intr=1.0", "dup=0.25", "high_bits", "L=6", "Z=5", "recv=1"):
            assert marker in variant_text, marker

    def test_duplicate_variant_rows_have_expectations(self):
        cases = standard_cases()
        dup_rows = [case for case in cases if case.duplicate_ratio]
        assert {case.protocol for case in dup_rows} == {
            "PROTOCOL_RR22",
            "PROTOCOL_KKRT",
            "PROTOCOL_ECDH",
        }
        by_protocol = {case.protocol: case for case in dup_rows}
        assert by_protocol["PROTOCOL_RR22"].expect_status == "error"
        assert by_protocol["PROTOCOL_KKRT"].expect_status == "error"
        assert by_protocol["PROTOCOL_ECDH"].expect_status == "ok"
        assert all(case.case_note for case in dup_rows)
        # 顺序纪律：可完成的行排在会失败的行之前（失败用例会污染同进程后续运行）
        labels = [case.label() for case in cases]
        assert labels.index("ECDH N=2^12 dup=0.25") < labels.index("RR22 N=2^12 dup=0.25")
        assert labels.index("RR22 N=2^12 dup=0.25") < labels.index("KKRT N=2^12 dup=0.25")

    def test_unexpected_records_semantics(self):
        def record(status, expected=""):
            return {"status": status, "expect_status": expected}

        assert unexpected_records(
            [record("ok"), record("unavailable"), record("error", "error"), record("ok", "ok")]
        ) == []
        flagged = unexpected_records([record("error"), record("ok", "error")])
        assert [item["status"] for item in flagged] == ["error", "ok"]

    def test_run_benchmark_cases_checkpoints_and_stops_nothing(self, tmp_path):
        cases = [
            BenchmarkCase("PROTOCOL_ECDH", n=2**10, execute=False),
            BenchmarkCase("PROTOCOL_KKRT", n=2**10, execute=False),
        ]
        out = tmp_path / "records.json"
        records = run_benchmark_cases(cases, out_json=str(out))
        assert len(records) == 2
        assert all(record["status"] == "unavailable" for record in records)
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert len(saved) == 2


class TestRegressionCompare:
    """§19：性能回归门禁骨架——只报告，不阻断，阈值可调。"""

    @staticmethod
    def _record(case, status="ok", wall_ms=100.0, protocol="PROTOCOL_RR22"):
        return {
            "case": case,
            "protocol": protocol,
            "status": status,
            "wall_ms": wall_ms,
        }

    def test_ratio_and_flag(self):
        rows = compare_with_baseline(
            [self._record("RR22 N=2^12", wall_ms=150.0)],
            [self._record("RR22 N=2^12", wall_ms=100.0)],
            warn_ratio=1.5,
        )
        assert len(rows) == 1
        assert rows[0]["ratio"] == 1.5
        assert rows[0]["flagged"] is True
        assert "仅提示" in rows[0]["reason"]

    def test_below_threshold_not_flagged(self):
        rows = compare_with_baseline(
            [self._record("RR22 N=2^12", wall_ms=110.0)],
            [self._record("RR22 N=2^12", wall_ms=100.0)],
            warn_ratio=1.5,
        )
        assert rows[0]["ratio"] == 1.1
        assert rows[0]["flagged"] is False

    def test_error_rows_are_not_compared(self):
        rows = compare_with_baseline(
            [self._record("RR22 N=2^12", status="error", wall_ms=None)],
            [self._record("RR22 N=2^12", wall_ms=100.0)],
        )
        assert rows[0]["ratio"] is None
        assert "不作性能对比" in rows[0]["reason"]

    def test_coverage_differences_are_reported(self):
        rows = compare_with_baseline(
            [self._record("KKRT N=2^12")],
            [self._record("RR22 N=2^12")],
        )
        reasons = {row["case"]: row["reason"] for row in rows}
        assert "仅当前有" in reasons["KKRT N=2^12"]
        assert "仅基线有" in reasons["RR22 N=2^12"]

    def test_compare_never_raises_on_empty_inputs(self):
        assert compare_with_baseline([], []) == []


class TestPlainReference:
    def test_intersects_reference(self):
        assert plain_intersects_reference([1, 2], [2, 3]) is True
        assert plain_intersects_reference([1], [2]) is False
        assert plain_intersects_reference([], [1]) is False


class TestWriters:
    def test_json_roundtrip(self, tmp_path):
        record = run_benchmark_case(BenchmarkCase("PROTOCOL_ECDH", n=2**10, execute=False))
        path = tmp_path / "sub" / "records.json"
        write_benchmark_json([record], str(path))
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == [record]

    def test_csv_columns_and_row(self, tmp_path):
        record = run_benchmark_case(BenchmarkCase("PROTOCOL_ECDH", n=2**10, execute=False))
        path = tmp_path / "records.csv"
        write_benchmark_csv([record], str(path))
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0].split(",")[0] == "case"
        assert "PROTOCOL_ECDH" in lines[1]
        assert "unavailable" in lines[1]

    def test_summary_contains_label_and_status(self):
        record = run_benchmark_case(BenchmarkCase("PROTOCOL_ECDH", n=2**10, execute=False))
        text = format_summary([record])
        assert "ECDH" in text
        assert "unavailable" in text


class TestRealBenchmarkRun:
    @needs_psi
    def test_rr22_case_records_real_numbers(self):
        record = run_benchmark_case(BenchmarkCase("PROTOCOL_RR22", False, 256))
        assert record["status"] == "ok", record["error"]
        assert record["agreement"] is True
        assert record["intersection_count"] == 128
        assert record["intersection_count_expected"] == 128
        assert record["n_left"] == 256 and record["n_right"] == 256
        assert record["psi_execute_ms"] > 0
        assert record["total_ms"] > 0
        assert record["memory_mb"] >= 0
        assert record["curve"] is None
        for key in SCHEMA_KEYS:
            assert key in record

    @needs_psi
    def test_rr22_low_comm_case_also_runs(self):
        record = run_benchmark_case(BenchmarkCase("PROTOCOL_RR22", True, 256))
        assert record["status"] == "ok", record["error"]
        assert record["low_comm_mode"] is True
        assert record["agreement"] is True