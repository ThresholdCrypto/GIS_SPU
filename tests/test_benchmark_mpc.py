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

from tests.benchmarks.benchmark_mpc import (  # noqa: E402
    build_cases,
    parse_args,
    resolve_output_paths,
)
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
    temporal_ab_cases,
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
        """**两两比较电路**的节点数上界必须显著低于其他算子（O(N·M) 会拖死进程）。

        口径（P3 之后）：`TemporalOverlap` 有两套电路，这里的"上限 32"只约束
        **逐对版**；sweep 版可以跑到 1024，见下一条用例。
        """

        largest = max(
            c.k
            for c in standard_cases()
            if c.op == "TemporalOverlap" and not c.strategy
        )
        assert largest <= 32, f"TemporalOverlap 逐对电路节点数上限过大：{largest}"

    def test_temporal_overlap_sweep_rows_prove_the_ceiling_moved(self):
        """sweep 电路必须真的在基线里跑到远超 32 的规模——否则"上限提高"只是口号。"""

        sweep = [c for c in standard_cases() if c.strategy]
        assert sweep, "基线里应有 sweep 电路的用例"
        assert all(c.op == "TemporalOverlap" for c in sweep)
        assert all(c.strategy == "sweep" for c in sweep)
        assert max(c.k for c in sweep) >= 1024
        pairwise_k = {
            c.k for c in standard_cases() if c.op == "TemporalOverlap" and not c.strategy
        }
        assert pairwise_k & {c.k for c in sweep}, "缺少同规模的 A/B 对照点"

    def test_strategy_rows_are_labelled_so_they_cannot_be_confused(self):
        """换电路的行必须在标签上看得出来，否则汇总表会把两套电路混成一条。"""

        cases = standard_cases()
        labels = [c.label() for c in cases]
        assert "TemporalOverlap ABY3 FM64 K=256 [sweep]" in labels
        assert "TemporalOverlap ABY3 FM64 K=32 [sweep]" in labels
        assert "TemporalOverlap ABY3 FM64 K=32" in labels
        # 同规模（K=32）上两套电路是两个不同的标签，汇总表不会把它们混成一条
        assert "TemporalOverlap ABY3 FM64 K=32 [sweep]" != "TemporalOverlap ABY3 FM64 K=32"
        # 只有 sweep 行带方括号标记，逐对行不带
        assert all("[sweep]" in c.label() for c in cases if c.strategy)
        assert all("[sweep]" not in c.label() for c in cases if not c.strategy)

    def test_strategy_is_rejected_for_ops_without_a_second_circuit(self):
        """给没有第二套电路的算子塞 strategy 必须报错，而不是被静默忽略。"""

        record = run_benchmark_case(
            MpcBenchmarkCase(op="DistanceLE", protocol="ABY3", k=16, strategy="sweep")
        )
        assert record["status"] == "error"
        assert "TemporalOverlap" in record["error"]

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

    def test_no_case_is_declared_as_a_known_failure(self):
        """P2-1 之后标准用例集里**不该**再有"预期失败/预期偏差"登记。

        修除法之前这里登记了两条（FM32 环宽下限、大 K 近似除法）；现在它们
        都不复现。若有人把运行时除法加回生成代码，`unexpected_records` 会
        直接把失败报出来——这正是我们要的回归信号，而不是提前登记好让它闭嘴。
        """

        declared = [c for c in standard_cases() if c.expect_status]
        assert declared == [], [(c.label(), c.expect_status) for c in declared]

    def test_deviation_machinery_is_still_available_and_testable(self):
        """登记机制本身留着：允许 `error` / `deviation` 两种标记，语义有测试守。"""

        hard = [c for c in standard_cases() if c.expect_status == "error"]
        soft = [c for c in standard_cases() if c.expect_status == EXPECT_DEVIATION]
        assert hard == [] and soft == []

    def test_deviation_status_is_neither_ok_nor_error(self):
        """`deviation` 是个独立标记：它既不能写成 ok，也不能写成 error。

        硬钉 `error` 会把基线做成抽奖（同一份代码两次运行退出码不同），
        写成 `ok` 又等于宣布一个没验证过的结论。当前没有用例用到它，
        但机制保留——下次遇到非确定路径时不必重新发明。
        """

        assert EXPECT_DEVIATION not in ("ok", "error", "")


class TestTemporalAbMode:
    """两套电路的 A/B 对照模式（§4.1 的交叉点表要能由入库产物复现）。"""

    def test_ab_pairs_both_circuits_at_every_size(self):
        by_k: dict[int, set[str]] = {}
        for case in temporal_ab_cases():
            by_k.setdefault(case.k, set()).add(case.strategy)
        assert by_k, "A/B 用例集不能是空集"
        for k, strategies in by_k.items():
            assert strategies == {"pairwise", "sweep"}, (k, strategies)

    def test_ab_pins_both_strategies_explicitly(self):
        """两侧的 strategy 都要写死：靠生成器默认值等于把对照物交给别人改。"""

        cases = temporal_ab_cases()
        assert all(case.strategy in ("pairwise", "sweep") for case in cases)
        assert all(case.op == "TemporalOverlap" for case in cases)

    def test_ab_carries_repeats_and_capture_comm(self):
        cases = temporal_ab_cases(repeats=5, capture_comm=True)
        assert all(case.repeats == 5 and case.capture_comm for case in cases)
        assert all(case.repeats == 1 for case in temporal_ab_cases(repeats=0))

    def test_ab_pairwise_arm_is_the_same_circuit_as_the_default(self):
        """A/B 的 pairwise 侧必须就是主扫描里的那套电路，否则对照不成立。"""

        from backends.jax_backend.codegen import generate_temporal_overlap

        default = generate_temporal_overlap("f")
        labelled = generate_temporal_overlap("f", strategy="pairwise")
        assert labelled.source == default.source

    def test_ab_mode_cli_selects_the_cases_and_the_default_paths(self):
        args = parse_args(["--temporal-ab"])
        cases = build_cases(args)
        assert {c.strategy for c in cases} == {"pairwise", "sweep"}
        assert all("[pairwise]" in c.label() or "[sweep]" in c.label() for c in cases)
        json_path, csv_path = resolve_output_paths(args)
        assert json_path.endswith("docs/mpc_temporal_ab.json"), json_path
        assert csv_path.endswith("docs/mpc_temporal_ab.csv"), csv_path

    def test_standard_mode_keeps_the_baseline_default_paths(self):
        json_path, csv_path = resolve_output_paths(parse_args([]))
        assert json_path.endswith("docs/mpc_benchmark_baseline.json"), json_path
        assert csv_path.endswith("docs/mpc_benchmark_baseline.csv"), csv_path

    def test_ab_mode_refuses_to_be_combined_with_filters(self):
        """与 --ops 组合会得到"只有一侧电路"的半张对照表，必须报错而不是照跑。"""

        for argv in (
            ["--temporal-ab", "--ops", "WeightedSum"],
            ["--temporal-ab", "--protocols", "CHEETAH"],
        ):
            with pytest.raises(SystemExit) as excinfo:
                build_cases(parse_args(argv))
            assert "temporal-ab" in str(excinfo.value)


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
            # P2-2：通信量字段（未开 profiling 时全为空，但字段必须在）
            "profiled", "comm_send_bytes", "comm_recv_bytes", "comm_total_bytes",
            "comm_send_actions", "comm_recv_actions", "comm_by_primitive",
        }

    def test_comm_fields_are_empty_without_profiling(self):
        """没开 profiling 就不许有通信量——空着比填个猜的数更诚实。"""

        record = run_benchmark_case(
            MpcBenchmarkCase(op="DistanceLE", protocol="ABY3", k=16)
        )
        assert record["profiled"] is False
        assert record["comm_total_bytes"] is None
        assert record["comm_send_bytes"] is None
        assert record["comm_recv_bytes"] is None
        assert record["comm_by_primitive"] == {}

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

    def test_strategy_is_recorded_not_only_in_the_label(self):
        """电路形态必须落进记录字段：只有标签不同的话，汇总/比价会把两套电路混掉。"""

        record = run_benchmark_case(
            MpcBenchmarkCase(
                op="TemporalOverlap", protocol="ABY3", field="FM64", k=32,
                strategy="sweep",
            )
        )
        assert record["strategy"] == "sweep"
        plain = run_benchmark_case(
            MpcBenchmarkCase(op="TemporalOverlap", protocol="ABY3", field="FM64", k=32)
        )
        assert plain["strategy"] == ""
        # 字段进了 CSV 列，否则导出后这份信息就丢了
        from backends.spu_backend.benchmark import _CSV_COLUMNS

        assert "strategy" in _CSV_COLUMNS

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

    def test_comm_is_aggregated_as_a_median_with_a_band(self):
        """通信量有 ~1–2% 批间抖动：报中位 + 区间，不报单次值。"""

        rows = [
            _row(profiled=True, comm_total_bytes=t, comm_send_bytes=t // 2,
                 comm_recv_bytes=t - t // 2, comm_send_actions=10, comm_recv_actions=10)
            for t in (9286, 9330, 9366)
        ]
        summary = summarize_repeats(rows)
        assert summary["comm_total_bytes_stats"]["min"] == 9286.0
        assert summary["comm_total_bytes_stats"]["max"] == 9366.0
        assert summary["comm_total_bytes"] == 9330.0  # 中位
        assert summary["profiled"] is True
        assert "抖动" in summary["note"]

    def test_runs_without_comm_leave_the_columns_empty(self):
        """一次都没采到就不填数——`profiled` 也要跟着说真话。"""

        summary = summarize_repeats([_row(), _row()])
        assert summary["comm_total_bytes"] is None
        assert summary["comm_total_bytes_stats"] == {}  # 没采到就没有统计块
        assert summary["comm_by_primitive"] == {}
        assert summary["profiled"] is False

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

    def test_standard_cases_carry_capture_comm(self):
        assert all(case.capture_comm for case in standard_cases(capture_comm=True))
        assert not any(case.capture_comm for case in standard_cases())
        # 越界占位行同样带着标记（它不执行，但口径要一致）
        probes = [c for c in standard_cases(capture_comm=True) if not c.execute]
        assert probes and all(case.capture_comm for case in probes)


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
        # P2-2：通信量列必须在表头里（列顺序对表格工具是契约）
        header = lines[0].split(",")
        assert header.index("comm_total_bytes") > header.index("pphlo_bytes")
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
    def test_capture_comm_reports_real_bytes(self):
        """真跑一遍带 profiling 的 ABY3：必须真采到通信量，且与动作数自洽。

        这条测试的作用不是"验证协议对不对"，而是守住"这一栏有没有真的取到数"：
        取不到就该是 None，绝不许被填成一个看起来合理的值。
        """

        record = run_benchmark_case(
            MpcBenchmarkCase(
                op="DistanceLE", protocol="ABY3", field="FM64", k=64,
                capture_comm=True,
            )
        )
        assert record["status"] == "ok", record["error"]
        assert record["profiled"] is True
        assert record["comm_total_bytes"] is not None, record["note"]
        assert record["comm_total_bytes"] > 0
        assert record["comm_send_actions"] > 0
        assert record["comm_recv_actions"] > 0
        # 逐原语明细要么为空（解析不到），要么与总量同量级——不许出现负值/超大值
        for name, entry in record["comm_by_primitive"].items():
            assert entry["send_bytes"] >= 0, name
            assert entry["executions"] >= 1, name

    def test_ref2k_reports_zero_communication(self):
        """REF2K 是明文参照协议（不做密码学保护），通信量为 0 才是对的。

        这正是 P2-2 想要的对照：**墙钟最快 ≠ 隐私代价最低**，
        而 REF2K 的"0 字节"直接说明它根本不该被当成隐私协议候选。
        """

        record = run_benchmark_case(
            MpcBenchmarkCase(
                op="DistanceLE", protocol="REF2K", field="FM64", k=64,
                capture_comm=True,
            )
        )
        assert record["status"] == "ok", record["error"]
        assert record["comm_total_bytes"] == 0

    def test_comm_capture_survives_a_prior_native_log_shutdown(self):
        """PSI 路径会**进程级**关掉 SPU 原生日志（`_set_native_log(quiet=True)`）。

        实测坑：只要被关过一次，同一进程里后续 SPU 运行的 pphlo profile 行
        就再也不会出现，fd 重定向拿到的是空文本——通信量会静默变成"取不到"。
        所以 `capture_comm=True` 必须自己先把 console logger 打开。
        这里先按 PSI 的做法关一次，再断言 profiling 仍然取得到数。
        """

        from backends.psi_backend.runtime import _set_native_log

        _set_native_log(quiet=True)
        try:
            record = run_benchmark_case(
                MpcBenchmarkCase(
                    op="DistanceLE", protocol="ABY3", field="FM64", k=64,
                    capture_comm=True,
                )
            )
        finally:
            # 复原：别把后面用例 / 用户终端弄哑（原生日志开关是进程级的）
            _set_native_log(quiet=False)
        assert record["profiled"] is True
        assert record["comm_total_bytes"] is not None, record["note"]
        assert record["comm_total_bytes"] > 0

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

    def test_fm32_works_after_division_removal(self):
        """P2-1 回归：`WeightedSum × FM32` 现在真的能跑，且与明文逐位一致。

        修复前这条组合必然报
        `ring=FM32 could not represent PT_I64`（除法路径要 64 位环），
        而 planner 的位宽预测 b(K=256)=24 ≤ 32 却判"够用"——
        "预测 ≠ 实测"的实例。现在预测与实测都指向"可用"。
        """

        record = run_benchmark_case(
            MpcBenchmarkCase(op="WeightedSum", protocol="ABY3", field="FM32", k=256)
        )
        assert record["status"] == "ok", record["error"]
        assert record["agreement"] is True
        assert record["max_abs_error"] == 0.0
        assert record["bit_width_covered"] is True

    def test_large_k_accumulation_is_exact_after_division_removal(self):
        """P2-1 回归：K=4096 重复 5 次全部与明文逐位一致。

        修复前这条是 43%–93% 的偏差率（近似除法），现在偏差率必须为 0。
        """

        record = run_benchmark_case(
            MpcBenchmarkCase(
                op="WeightedSum", protocol="ABY3", field="FM64", k=4096, repeats=5
            )
        )
        assert record["repeat"] == 5
        assert record["status_counts"] == {"ok": 5}, record["status_counts"]
        assert record["deviation_rate"] == 0.0
        assert record["max_abs_error_max"] == 0.0
