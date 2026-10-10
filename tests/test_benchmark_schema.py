# -*- coding: utf-8 -*-
"""统一 benchmark metadata（Phase 7 / 任务文档 §九）的测试。

对拍用**真实产物**（`docs/` 下五份基线 JSON），不造假记录：
投影必须与产物逐字段对得上，声明为「缺口」的指标必须真的不在产物里。
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest

from backends import benchmark_schema as bs
from backends.spu_backend.protocol_registry import (
    MPC_PROTOCOL_SPECS,
    SUPPORTABLE_FIELDS,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PSI_BASELINE = os.path.join(PROJECT_ROOT, "docs", "psi_benchmark_baseline.json")
MPC_BASELINE = os.path.join(PROJECT_ROOT, "docs", "mpc_benchmark_baseline.json")
MPC_COMM_BASELINE = os.path.join(PROJECT_ROOT, "docs", "mpc_comm_baseline.json")
PSI_CA_BASELINE = os.path.join(PROJECT_ROOT, "docs", "psi_ca_benchmark_baseline.json")
PSI_SUM_BASELINE = os.path.join(PROJECT_ROOT, "docs", "psi_sum_benchmark_baseline.json")

#: 任务文档 §九 的 `BenchmarkRecord` 字段（逐字对照，测试独立抄一份）
TASK_DOC_COMMON_FIELDS = (
    "family",
    "protocol",
    "operation",
    "world_size",
    "field",
    "input_size",
    "unique_size",
    "result_size",
    "compute_time",
    "communication_bytes",
    "total_time",
    "memory_bytes",
    "status",
)

#: 任务文档 §九「建议至少记录」的指标清单（逐字对照）
TASK_DOC_METRICS = (
    "family",
    "protocol",
    "operation",
    "world_size",
    "field",
    "N",
    "unique_N",
    "intersection_ratio",
    "encode_time",
    "dedup_time",
    "input_io_time",
    "protocol_time",
    "semantic_processing_time",
    "total_time",
    "send_bytes",
    "recv_bytes",
    "total_bytes",
    "peak_memory",
    "status",
    "layout_agreement",
    "result_semantics",
)


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _sum(values):
    """产物里缺一项就整体记 None（与投影层的「全有才算」同口径）。"""

    return None if any(value is None for value in values) else sum(values)


@pytest.fixture(scope="module")
def psi_raw():
    return _read(PSI_BASELINE)


@pytest.fixture(scope="module")
def psi_ca_raw():
    return _read(PSI_CA_BASELINE)


@pytest.fixture(scope="module")
def psi_sum_raw():
    return _read(PSI_SUM_BASELINE)


@pytest.fixture(scope="module")
def mpc_raw():
    return _read(MPC_BASELINE)


@pytest.fixture(scope="module")
def mpc_comm_raw():
    return _read(MPC_COMM_BASELINE)


@pytest.fixture(scope="module")
def psi_records():
    return bs.load_benchmark_file(PSI_BASELINE)


@pytest.fixture(scope="module")
def psi_ca_records():
    return bs.load_benchmark_file(PSI_CA_BASELINE)


@pytest.fixture(scope="module")
def psi_sum_records():
    return bs.load_benchmark_file(PSI_SUM_BASELINE)


@pytest.fixture(scope="module")
def mpc_records():
    return bs.load_benchmark_file(MPC_BASELINE)


@pytest.fixture(scope="module")
def mpc_comm_records():
    return bs.load_benchmark_file(MPC_COMM_BASELINE)


class TestSchemaTables:
    def test_common_fields_are_the_task_doc_record(self):
        assert bs.COMMON_FIELDS == TASK_DOC_COMMON_FIELDS

    def test_every_common_field_is_documented(self):
        assert set(bs.FIELD_SEMANTICS) == set(bs.COMMON_FIELDS)
        assert all(bs.FIELD_SEMANTICS[name] for name in bs.COMMON_FIELDS)

    def test_task_doc_metric_list_is_covered_in_order(self):
        assert tuple(bs.TASK_DOC_METRICS) == TASK_DOC_METRICS

    def test_task_doc_targets_resolve(self):
        psi_fields = set(bs.PSIBenchmarkMetadata.__dataclass_fields__)
        mpc_fields = set(bs.MPCBenchmarkMetadata.__dataclass_fields__)
        declared_gaps = set()
        for names in bs.MISSING_METRICS.values():
            declared_gaps |= set(names)
        for name, metric in bs.TASK_DOC_METRICS.items():
            kind, _, field = metric.target.partition(":")
            if kind == "common":
                assert field in bs.COMMON_FIELDS, name
            elif kind == "metadata":
                assert field in psi_fields or field in mpc_fields, name
            elif kind == "missing":
                assert name in declared_gaps, name
            else:
                pytest.fail(f"未知去向 {metric.target!r}（指标 {name}）")

    def test_declared_gaps_say_why(self):
        for name, metric in bs.TASK_DOC_METRICS.items():
            if metric.target == "missing":
                assert metric.note, f"{name} 记为缺口，但没写原因"

    def test_missing_raw_keys_cover_declared_gaps(self):
        declared = set()
        for names in bs.MISSING_METRICS.values():
            declared |= set(names)
        assert set(bs.MISSING_RAW_KEYS) == declared

    def test_families_are_declared(self):
        assert bs.BENCHMARK_FAMILIES == ("PSI", "PSI-CA", "PI-SUM", "MPC")
        assert set(bs.MISSING_METRICS) == set(bs.BENCHMARK_FAMILIES)

    def test_curve_relation_vocabulary_matches_registry(self):
        assert bs.psi_curve_relation("PROTOCOL_ECDH") == bs.CURVE_RELATION_REQUIRED
        assert bs.psi_curve_relation("PROTOCOL_KKRT") != bs.CURVE_RELATION_REQUIRED


class TestPsiProjection:
    def test_every_row_projects_in_order(self, psi_records, psi_raw):
        assert len(psi_records) == len(psi_raw) == 41
        assert [record.metadata["case"] for record in psi_records] == [
            record["case"] for record in psi_raw
        ]

    def test_common_fields_come_from_the_record(self, psi_records, psi_raw):
        assert len(psi_records) == len(psi_raw)
        for record, raw in zip(psi_records, psi_raw):
            assert record.family == "PSI"
            assert record.protocol == raw["protocol"]
            assert record.operation == "CellSetIntersect"
            assert record.world_size == 2
            assert record.input_size == _sum([raw["n_left"], raw["n_right"]])
            assert record.unique_size == _sum(
                [raw["n_left_unique"], raw["n_right_unique"]]
            )
            assert record.result_size == raw["intersection_count"]
            assert record.compute_time == raw["psi_execute_ms"]
            assert record.total_time == raw["total_ms"]
            assert record.status == raw["status"]
            assert record.communication_bytes is None

    def test_memory_is_peak_rss_converted_from_mb(self, psi_records, psi_raw):
        first_record, first_raw = psi_records[0], psi_raw[0]
        assert first_raw["peak_rss_mb"] == 209.3
        assert first_record.memory_bytes == 219466957
        for record, raw in zip(psi_records, psi_raw):
            assert record.metadata["peak_memory_mb"] == raw["peak_rss_mb"]

    def test_field_is_curve_only_when_the_protocol_reads_it(self, psi_raw):
        seen_required = seen_ignored = 0
        for raw in psi_raw:
            record = bs.psi_record(raw)
            relation = record.metadata["curve_relation"]
            if relation == bs.CURVE_RELATION_REQUIRED:
                seen_required += 1
                assert record.field == raw["curve"]
            elif relation == "ignored":
                seen_ignored += 1
                assert record.field is None
                assert record.metadata["curve"] == raw["curve"]
        assert seen_required and seen_ignored, "两组都必须在真实产物里出现"

    def test_unexecuted_rows_carry_no_numbers(self, psi_records):
        unavailable = [r for r in psi_records if r.status == "unavailable"]
        assert len(unavailable) == 2
        for record in unavailable:
            assert record.input_size is None
            assert record.unique_size is None
            assert record.result_size is None
            assert record.compute_time is None
            assert record.total_time is None
            assert record.memory_bytes is None
            assert record.metadata["agreement"] is None

    def test_timing_breakdown_matches_the_artifact(self, psi_records, psi_raw):
        for record, raw in zip(psi_records, psi_raw):
            metadata = record.metadata
            if raw["io_write_ms"] is None or raw["io_read_ms"] is None:
                assert metadata["input_io_time"] is None
                continue
            assert metadata["input_io_time"] == pytest.approx(
                raw["io_write_ms"] + raw["io_read_ms"]
            )
            assert metadata["protocol_time"] == raw["psi_execute_ms"]
            assert metadata["semantic_processing_time"] == raw["semantic_ms"]

    def test_protocol_time_is_the_measured_psi_execute_everywhere(
        self, psi_records, psi_raw
    ):
        for record, raw in zip(psi_records, psi_raw):
            assert record.metadata["protocol_time"] == raw["psi_execute_ms"]
            assert record.compute_time == raw["psi_execute_ms"]

    def test_result_semantics_is_derived_from_the_registry(self, psi_records):
        for record in psi_records:
            assert record.metadata["result_semantics"] == "exact"

    def test_metadata_keeps_the_whole_raw_record(self, psi_raw):
        for raw in psi_raw:
            metadata = bs.psi_record(raw).metadata
            assert metadata["raw"] == raw


class TestPsiCaProjection:
    def test_every_row_projects_in_order(self, psi_ca_records, psi_ca_raw):
        assert len(psi_ca_records) == len(psi_ca_raw) == 3
        assert [record.metadata["case"] for record in psi_ca_records] == [
            record["case"] for record in psi_ca_raw
        ]

    def test_common_fields_come_from_the_record(self, psi_ca_records, psi_ca_raw):
        for record, raw in zip(psi_ca_records, psi_ca_raw):
            assert record.family == "PSI-CA"
            assert record.protocol == raw["protocol"] == "PSI-CA"
            assert record.operation == "CellSetIntersect"
            assert record.world_size == raw["world_size"] == 2
            assert record.input_size == _sum([raw["n_left"], raw["n_right"]])
            assert record.unique_size == _sum(
                [raw["n_left_unique"], raw["n_right_unique"]]
            )
            assert record.result_size == raw["intersection_count"]
            assert record.compute_time == raw["psi_ca_execute_ms"]
            assert record.total_time == raw["total_ms"]
            assert record.status == raw["status"]
            # Phase 11 起通信量有值：协议消息载荷合计（口径见 §7.1）
            assert record.communication_bytes == raw["total_bytes"]

    def test_protocol_payload_is_split_by_direction(
        self, psi_ca_records, psi_ca_raw
    ):
        for record, raw in zip(psi_ca_records, psi_ca_raw):
            assert raw["comm_meter"] == "protobuf-payload"
            assert raw["send_bytes"] > 0 and raw["recv_bytes"] > 0
            assert raw["send_bytes"] + raw["recv_bytes"] == raw["total_bytes"]
            assert record.metadata["send_bytes"] == raw["send_bytes"]
            assert record.metadata["recv_bytes"] == raw["recv_bytes"]
            assert record.metadata["total_bytes"] == raw["total_bytes"]
            assert record.communication_bytes == raw["total_bytes"]
            # 口径随数走：方向标签与计量名必须一起带出来
            assert "Request" in raw["comm_direction"]
            assert "ServerSetup" in raw["comm_direction"]

    def test_memory_is_the_runner_process_peak_rss(
        self, psi_ca_records, psi_ca_raw
    ):
        for record, raw in zip(psi_ca_records, psi_ca_raw):
            assert record.metadata["peak_memory_mb"] == raw["peak_rss_mb"]
            assert record.memory_bytes == int(
                round(raw["peak_rss_mb"] * bs.BYTES_PER_MB)
            )

    def test_count_matches_the_constructed_expectation(
        self, psi_ca_records, psi_ca_raw
    ):
        for record, raw in zip(psi_ca_records, psi_ca_raw):
            assert raw["intersection_count"] == raw["intersection_count_expected"]
            assert record.metadata["agreement"] is True

    def test_leak_and_semantics_come_from_the_record(self, psi_ca_records):
        for record in psi_ca_records:
            assert record.metadata["structure"] == "RAW"
            assert record.metadata["protocol_leak"] == "count-only"
            assert record.metadata["result_semantics"] == "exact"

    def test_metadata_keeps_the_whole_raw_record(self, psi_ca_raw):
        for raw in psi_ca_raw:
            metadata = bs.psi_ca_record(raw).metadata
            assert metadata["raw"] == raw


class TestPsiSumProjection:
    def test_every_row_projects_in_order(self, psi_sum_records, psi_sum_raw):
        assert len(psi_sum_records) == len(psi_sum_raw) == 3
        assert [record.metadata["case"] for record in psi_sum_records] == [
            record["case"] for record in psi_sum_raw
        ]

    def test_common_fields_come_from_the_record(self, psi_sum_records, psi_sum_raw):
        for record, raw in zip(psi_sum_records, psi_sum_raw):
            assert record.family == "PI-SUM"
            assert record.protocol == raw["protocol"] == "PJC-PI-SUM"
            assert record.operation == "CellSetIntersect"
            assert record.world_size == raw["world_size"] == 2
            assert record.input_size == _sum([raw["n_left"], raw["n_right"]])
            assert record.unique_size == _sum(
                [raw["n_left_unique"], raw["n_right_unique"]]
            )
            assert record.result_size == raw["intersection_count"]
            assert record.compute_time == raw["pi_sum_execute_ms"]
            assert record.total_time == raw["total_ms"]
            assert record.status == raw["status"]
            # Phase 10 计量：中继总字节 + 子进程 VmHWM 峰值（记录里就有）
            assert record.communication_bytes == raw["total_bytes"]
            assert record.memory_bytes == int(round(raw["peak_rss_mb"] * 1024 * 1024))

    def test_phase10_measurements_are_projected(self, psi_sum_records, psi_sum_raw):
        for record, raw in zip(psi_sum_records, psi_sum_raw):
            assert raw["comm_meter"] == "loopback-relay"
            assert raw["memory_probe"] == "procfs-VmHWM"
            assert raw["send_bytes"] + raw["recv_bytes"] == raw["total_bytes"]
            assert record.metadata["send_bytes"] == raw["send_bytes"]
            assert record.metadata["recv_bytes"] == raw["recv_bytes"]
            assert record.metadata["total_bytes"] == raw["total_bytes"]
            assert record.metadata["peak_memory_mb"] == raw["peak_rss_mb"]
            assert record.communication_bytes == raw["total_bytes"]
            assert record.memory_bytes is not None

    def test_input_io_time_comes_from_the_write_leg(
        self, psi_sum_records, psi_sum_raw
    ):
        # Phase 11：PI-Sum 的输入 I/O 只计「输入 CSV 落盘」一段；
        # 结果走 stdout，没有输出文件读取段，故产物里不该有 io_read_ms。
        for record, raw in zip(psi_sum_records, psi_sum_raw):
            assert raw["io_write_ms"] is not None
            assert "io_read_ms" not in raw
            assert record.metadata["input_io_time"] == raw["io_write_ms"]

    def test_sum_matches_the_constructed_expectation(
        self, psi_sum_records, psi_sum_raw
    ):
        for record, raw in zip(psi_sum_records, psi_sum_raw):
            assert raw["intersection_sum"] == raw["intersection_sum_expected"]
            assert record.metadata["intersection_sum"] == raw["intersection_sum"]
            assert record.metadata["intersection_sum_expected"] == raw[
                "intersection_sum_expected"
            ]
            assert record.metadata["agreement"] is True

    def test_upstream_provenance_is_carried(self, psi_sum_records):
        for record in psi_sum_records:
            assert record.metadata["upstream"] == "google/private-join-and-compute"
            assert record.metadata["upstream_commit"] == "950c5e4"
            assert record.metadata["paillier_modulus_size"] == 1536
            assert record.metadata["value_function"] == "1+(code%997)"

    def test_leak_and_semantics_come_from_the_record(self, psi_sum_records):
        for record in psi_sum_records:
            assert record.metadata["protocol_leak"] == "count+sum"
            assert record.metadata["result_semantics"] == "exact"

    def test_metadata_keeps_the_whole_raw_record(self, psi_sum_raw):
        for raw in psi_sum_raw:
            metadata = bs.psi_sum_record(raw).metadata
            assert metadata["raw"] == raw


class TestMpcProjection:
    def test_every_row_projects_in_order(self, mpc_records, mpc_raw):
        assert len(mpc_records) == len(mpc_raw) == 39
        assert [record.metadata["case"] for record in mpc_records] == [
            record["case"] for record in mpc_raw
        ]

    def test_common_fields_come_from_the_record(self, mpc_records, mpc_raw):
        for record, raw in zip(mpc_records, mpc_raw):
            assert record.family == "MPC"
            assert record.protocol == raw["protocol"]
            assert record.operation == raw["op"]
            assert record.field == raw["field"]
            assert record.input_size == raw["k"]
            assert record.unique_size is None
            assert record.result_size is None
            assert record.total_time == raw["wall_ms"]
            assert record.communication_bytes == raw["comm_total_bytes"]
            assert record.status == raw["status"]

    def test_compute_time_excludes_the_setup(self, mpc_records, mpc_raw):
        for record, raw in zip(mpc_records, mpc_raw):
            if raw["wall_ms"] is None or raw["setup_ms"] is None:
                assert record.compute_time is None
                continue
            assert record.compute_time == pytest.approx(
                raw["wall_ms"] - raw["setup_ms"]
            )
            assert record.compute_time < record.total_time

    def test_world_size_and_field_are_registered(self, mpc_records):
        for record in mpc_records:
            assert record.field in SUPPORTABLE_FIELDS
            assert record.world_size == MPC_PROTOCOL_SPECS[record.protocol].world_size

    def test_result_semantics_is_derived_from_the_registry(self, mpc_records):
        for record in mpc_records:
            assert (
                record.metadata["result_semantics"]
                == MPC_PROTOCOL_SPECS[record.protocol].result_semantics
            )

    def test_metadata_keeps_the_whole_raw_record(self, mpc_raw):
        for raw in mpc_raw:
            metadata = bs.mpc_record(raw).metadata
            assert metadata["raw"] == raw

    def test_communication_column_reflects_whether_it_was_measured(
        self, mpc_records, mpc_comm_records
    ):
        plain_ok = [
            record
            for record in mpc_records
            if record.status == bs.STATUS_OK
        ]
        assert plain_ok and all(
            record.communication_bytes is None for record in plain_ok
        )
        comm_ok = [
            record
            for record in mpc_comm_records
            if record.status == bs.STATUS_OK
        ]
        assert comm_ok and all(
            record.communication_bytes is not None for record in comm_ok
        )


class TestLosslessness:
    def test_to_dict_is_json_serialisable_and_stable(self, psi_records, mpc_records):
        for record in list(psi_records[:3]) + list(mpc_records[:3]):
            row = record.to_dict()
            assert list(row) == list(bs.COMMON_FIELDS) + ["metadata"]
            assert json.dumps(row, ensure_ascii=False) == json.dumps(
                record.to_dict(), ensure_ascii=False
            )

    def test_adapters_are_the_only_projection_path(self, psi_raw, mpc_raw):
        assert bs.psi_record(psi_raw[0]).to_dict() == bs.unify_records(
            "PSI", psi_raw[:1]
        )[0].to_dict()
        assert bs.mpc_record(mpc_raw[0]).to_dict() == bs.unify_records(
            "MPC", mpc_raw[:1]
        )[0].to_dict()

    def test_status_counts_are_ordered(self, psi_records):
        counts = bs.status_counts(psi_records)
        assert list(counts) == ["ok", "unavailable", "error"]
        assert counts == {"ok": 37, "unavailable": 2, "error": 2}


class TestFamilyInference:
    def test_infers_every_declared_family(
        self, psi_raw, psi_ca_raw, psi_sum_raw, mpc_raw
    ):
        assert bs.infer_family(psi_raw[0]) == "PSI"
        assert bs.infer_family(psi_ca_raw[0]) == "PSI-CA"
        assert bs.infer_family(psi_sum_raw[0]) == "PI-SUM"
        assert bs.infer_family(mpc_raw[0]) == "MPC"

    def test_refuses_to_guess(self):
        with pytest.raises(ValueError, match="无法判断"):
            bs.infer_family({"case": "something"})

    def test_rejects_a_mixed_file(self, psi_raw, mpc_raw, tmp_path):
        path = tmp_path / "mixed.json"
        path.write_text(
            json.dumps([psi_raw[0], mpc_raw[0]], ensure_ascii=False),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="族不一致"):
            bs.load_benchmark_file(str(path))

    def test_rejects_a_file_that_is_not_a_record_list(self, tmp_path):
        path = tmp_path / "object.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="期望非空记录列表"):
            bs.load_benchmark_file(str(path))

    def test_rejects_unknown_family(self, psi_raw):
        with pytest.raises(ValueError, match="未知 benchmark 族"):
            bs.unify_records("TEE", psi_raw[:1])


class TestMissingMetricsAreReal:
    def test_declared_gaps_are_absent_from_the_psi_artifact(self, psi_raw):
        for name in bs.MISSING_METRICS["PSI"]:
            keys = bs.MISSING_RAW_KEYS[name]
            for raw in psi_raw:
                assert not any(key in raw for key in keys), (name, keys)

    def test_declared_gaps_are_absent_from_the_mpc_artifacts(
        self, mpc_raw, mpc_comm_raw
    ):
        for name in bs.MISSING_METRICS["MPC"]:
            keys = bs.MISSING_RAW_KEYS[name]
            for raw in list(mpc_raw) + list(mpc_comm_raw):
                assert not any(key in raw for key in keys), (name, keys)

    def test_declared_gaps_are_absent_from_the_psi_ca_artifact(self, psi_ca_raw):
        for name in bs.MISSING_METRICS["PSI-CA"]:
            keys = bs.MISSING_RAW_KEYS[name]
            for raw in psi_ca_raw:
                assert not any(key in raw for key in keys), (name, keys)

    def test_declared_gaps_are_absent_from_the_psi_sum_artifact(self, psi_sum_raw):
        for name in bs.MISSING_METRICS["PI-SUM"]:
            keys = bs.MISSING_RAW_KEYS[name]
            for raw in psi_sum_raw:
                assert not any(key in raw for key in keys), (name, keys)

    def test_psi_io_time_is_not_a_gap(self, psi_records):
        for record in psi_records:
            if record.status != bs.STATUS_OK:
                continue
            assert record.metadata["input_io_time"] is not None

    def test_psi_sum_io_time_is_not_a_gap(self, psi_sum_records):
        for record in psi_sum_records:
            if record.status != bs.STATUS_OK:
                continue
            assert record.metadata["input_io_time"] is not None

    def test_psi_ca_communication_is_not_a_gap(self, psi_ca_records):
        for record in psi_ca_records:
            if record.status != bs.STATUS_OK:
                continue
            assert record.communication_bytes is not None



def _load_projection_script():
    """把投影器当模块加载（`scripts/` 不是包）。"""

    path = os.path.join(PROJECT_ROOT, "scripts", "unify_benchmark.py")
    spec = importlib.util.spec_from_file_location("unify_benchmark_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestUnifyBenchmarkScript:
    def test_writes_unified_json_and_csv(self, tmp_path, capsys):
        script = _load_projection_script()
        out_json = tmp_path / "unified.json"
        out_csv = tmp_path / "unified.csv"
        code = script.main(
            [PSI_BASELINE, "--json", str(out_json), "--csv", str(out_csv)]
        )
        assert code == 0
        rows = json.loads(out_json.read_text(encoding="utf-8"))
        assert len(rows) == 41
        assert list(rows[0]) == list(bs.COMMON_FIELDS) + ["metadata"]
        header = out_csv.read_text(encoding="utf-8").splitlines()[0]
        assert header == ",".join(bs.COMMON_FIELDS)
        printed = capsys.readouterr().out
        assert "族 PSI" in printed
        assert "ok=37" in printed

    def test_compare_mode_prints_the_ranking(self, capsys):
        script = _load_projection_script()
        code = script.main(
            [PSI_BASELINE, "--compare", "total_time", "--input-size", "8192"]
        )
        assert code == 0
        printed = capsys.readouterr().out
        assert "比较指标：total_time" in printed
        assert "ECDH N=2^12" in printed

    def test_a_mixed_file_is_reported_as_an_error(self, tmp_path):
        script = _load_projection_script()
        mixed = tmp_path / "mixed.json"
        mixed.write_text(
            json.dumps([_read(PSI_BASELINE)[0], _read(MPC_BASELINE)[0]]),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="族不一致"):
            script.main([str(mixed)])

class TestCompareProtocols:
    def test_psi_ranking_matches_the_documented_finding(self, psi_records):
        compared = bs.compare_protocols(
            psi_records, "total_time", family="PSI", input_size=8192
        )
        assert compared.input_size == 8192
        values = [row.value for row in compared.rows]
        assert values == sorted(values)
        by_key = {row.key: row.value for row in compared.rows}
        # 文档 docs/BENCHMARK_PROTOCOL.md §6：ECDH 在 2^12 上远慢于 KKRT / RR22
        assert by_key["ECDH N=2^12"] > by_key["KKRT N=2^12"]
        assert by_key["ECDH N=2^12"] > by_key["RR22 N=2^12"]

    def test_psi_low_comm_variant_keeps_its_own_row(self, psi_records):
        compared = bs.compare_protocols(
            psi_records, "total_time", family="PSI", input_size=8192
        )
        keys = {row.key for row in compared.rows}
        assert "RR22 N=2^12" in keys
        assert "RR22+low_comm N=2^12" in keys

    def test_duplicate_matrix_rows_carry_their_own_size(self, psi_records):
        by_case = {record.metadata["case"]: record for record in psi_records}
        assert by_case["RR22 N=2^12"].input_size == 8192
        assert by_case["RR22 N=2^12 dup=0.25"].input_size == 10240

    def test_psi_error_rows_are_skipped_not_ranked(self, psi_records):
        compared = bs.compare_protocols(
            psi_records, "total_time", family="PSI", input_size=10240
        )
        assert [row.key for row in compared.rows] == ["ECDH N=2^12 dup=0.25"]
        skipped = dict(compared.skipped)
        assert skipped["RR22 N=2^12 dup=0.25"] == "status=error（未执行的行不给数字）"
        assert skipped["KKRT N=2^12 dup=0.25"] == "status=error（未执行的行不给数字）"

    def test_within_psi_ca_family_is_comparable(self, psi_ca_records):
        compared = bs.compare_protocols(
            psi_ca_records, "compute_time", input_size=2048
        )
        assert [row.key for row in compared.rows] == ["PSI-CA N=2^10"]
        assert compared.rows[0].samples == 1

    def test_cross_family_comparison_is_refused(self, psi_records, psi_ca_records):
        with pytest.raises(ValueError, match="多个族"):
            bs.compare_protocols(
                list(psi_records) + list(psi_ca_records), "total_time"
            )

    def test_mpc_repeats_are_aggregated_by_median(self, mpc_records):
        compared = bs.compare_protocols(
            mpc_records, "compute_time", operation="WeightedSum", input_size=256
        )
        row = next(r for r in compared.rows if r.key == "WeightedSum ABY3 FM64 K=256")
        assert row.samples >= 2
        assert row.spread is not None and row.spread > 0
        projected = sorted(
            r.compute_time
            for r in mpc_records
            if r.metadata["case"] == "WeightedSum ABY3 FM64 K=256"
            and r.status == bs.STATUS_OK
        )
        middle = len(projected) // 2
        expected = (
            projected[middle]
            if len(projected) % 2
            else (projected[middle - 1] + projected[middle]) / 2
        )
        assert row.value == pytest.approx(expected)

    def test_mpc_missing_metric_goes_to_skipped(self, mpc_records):
        compared = bs.compare_protocols(
            mpc_records,
            "communication_bytes",
            operation="DistanceLE",
            input_size=256,
        )
        assert compared.rows == ()
        assert compared.skipped
        for _, reason in compared.skipped:
            assert reason == "记录里没有 communication_bytes"

    def test_refuses_to_rank_across_operations(self, mpc_records):
        with pytest.raises(ValueError, match="多个算子"):
            bs.compare_protocols(mpc_records, "total_time", input_size=256)

    def test_refuses_to_rank_across_sizes(self, mpc_records):
        with pytest.raises(ValueError, match="多个规模"):
            bs.compare_protocols(mpc_records, "total_time", operation="WeightedSum")

    def test_refuses_an_unknown_metric(self, psi_records):
        with pytest.raises(ValueError, match="不可比较的字段"):
            bs.compare_protocols(psi_records, "status")

    def test_comparison_serialises(self, psi_records):
        compared = bs.compare_protocols(
            psi_records, "total_time", family="PSI", input_size=8192
        )
        row = compared.to_dict()
        assert row["metric"] == "total_time"
        assert row["rows"] and set(row["rows"][0]) == {
            "key",
            "value",
            "samples",
            "spread",
        }
