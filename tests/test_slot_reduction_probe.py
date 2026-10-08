# -*- coding: utf-8 -*-
"""免逐槽提取路线筛选探针（P0）的守卫测试。

守什么
======
1. **路线 (a)/(c) 必须继续对拍失败**——这是本轮的核心结论。若哪天有人把电路
   "修"成能过对拍（例如偷偷加了掩码），结论就被推翻了，这里必须先红。
   本测试用**纯 jax**（不需要 SPU）复算电路，所以它在任何环境都能守。
2. **基准变体必须对拍通过**，否则探针自己写错了；
3. **批间差要参与判定**：同一电路复测（`extract_seq_dup`）必须真的与 `extract_seq`
   同电路，且"小于批间差"的差异不许被写成结论；
4. **SWAR 树不许被当成免费的午餐**：它必须与顺序提取做同语义（sum）比较。
"""

from __future__ import annotations

import numpy as np
import pytest

from backends.spu_backend import slot_reduction_probe as sr
from backends.spu_backend.capability import CapabilityReport

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402


class TestVariantTable:
    def test_routes_are_labelled(self):
        routes = {variant.name: variant.route for variant in sr.SLOT_REDUCTION_VARIANTS}
        assert routes["extract_seq"] == "baseline"
        assert routes["fullmul"] == "route_a"
        assert routes["maskless_shift"] == "route_c"
        assert routes["extract_seq_secret_weight"] == "route_b"
        assert routes["extract_seq_dup"] == "noise"

    def test_targets_match_the_intended_semantics(self):
        targets = {variant.name: variant.target for variant in sr.SLOT_REDUCTION_VARIANTS}
        assert targets["extract_seq"] == "weighted"
        assert targets["extract_seq_secret_weight"] == "weighted"
        assert targets["maskless_shift"] == "weighted"
        assert targets["fullmul"] == "sum"
        assert targets["swar_tree"] == "sum"

    def test_dup_variant_is_the_same_circuit_as_the_baseline(self):
        by_name = {variant.name: variant for variant in sr.SLOT_REDUCTION_VARIANTS}
        assert by_name["extract_seq_dup"].fn is by_name["extract_seq"].fn, (
            "批间差对照必须与基准**同一电路**，否则量的是电路差异不是噪声"
        )

    def test_secret_weight_variant_takes_a_second_input(self):
        by_name = {variant.name: variant for variant in sr.SLOT_REDUCTION_VARIANTS}
        assert by_name["extract_seq_secret_weight"].secret_weights is True
        assert by_name["extract_seq"].secret_weights is False

    def test_only_is_supported(self):
        cases = sr.slot_reduction_cases(elements=8, repeats=0)
        assert {case.repeats for case in cases} == {1}
        assert len(cases) == len(sr.SLOT_REDUCTION_VARIANTS)


class TestCircuitSemanticsWithoutSpu:
    """纯 jax 复算：结论性的对拍结果在这里先钉住。"""

    @staticmethod
    def _inputs(elements: int = 64):
        packed, values = sr._slot_inputs(elements, sr.SLOT_REDUCTION_SLOTS)
        return jnp.asarray(packed), values

    def test_baseline_matches_the_weighted_target(self):
        packed, values = self._inputs()
        out = np.asarray(sr._extract_seq_fn(packed))
        assert np.array_equal(out, sr._weighted_target(values))

    def test_secret_weight_variant_matches_the_weighted_target(self):
        packed, values = self._inputs()
        weights = jnp.asarray(sr._secret_weights())
        out = np.asarray(sr._extract_seq_secret_weight_fn(packed, weights))
        assert np.array_equal(out, sr._weighted_target(values))

    def test_swar_tree_matches_the_unweighted_target(self):
        packed, values = self._inputs()
        out = np.asarray(sr._swar_tree_fn(packed))
        assert np.array_equal(out, sr._sum_target(values))

    def test_route_a_full_multiply_cannot_reach_the_target(self):
        """路线 a：一次整元素乘法拿不到 Σv_s——拿得到就说明结论要重写。"""

        packed, values = self._inputs()
        out = np.asarray(sr._fullmul_fn(packed))
        assert not np.array_equal(out, sr._sum_target(values))

    def test_route_c_maskless_shift_cannot_reach_the_target(self):
        """路线 c：省掉掩码后高位槽会串进来。"""

        packed, values = self._inputs()
        out = np.asarray(sr._maskless_shift_fn(packed))
        assert not np.array_equal(out, sr._weighted_target(values))

    def test_unweighted_variant_matches_the_sum_target(self):
        packed, values = self._inputs()
        out = np.asarray(sr._extract_unweighted_fn(packed))
        assert np.array_equal(out, sr._sum_target(values))


class TestSummaries:
    @staticmethod
    def _record(name, per, *, match=True, route="x"):
        return {
            "name": name, "comm_per_element": per, "within_tolerance": match,
            "route": route, "max_abs_error": 0.0,
        }

    def _full_set(self, **overrides):
        base = {
            "extract_seq": 1424.0,
            "extract_seq_dup": 1424.0,
            "extract_unweighted": 1496.0,
            "swar_tree": 1888.0,
            "fullmul": 224.0,
            "maskless_shift": 1888.0,
            "extract_seq_secret_weight": 1432.0,
        }
        base.update(overrides)
        records = [
            self._record("extract_seq", base["extract_seq"], route="baseline"),
            self._record("extract_seq_dup", base["extract_seq_dup"], route="noise"),
            self._record("extract_unweighted", base["extract_unweighted"]),
            self._record("swar_tree", base["swar_tree"]),
            self._record("fullmul", base["fullmul"], match=False, route="route_a"),
            self._record("maskless_shift", base["maskless_shift"], match=False,
                         route="route_c"),
            self._record("extract_seq_secret_weight",
                         base["extract_seq_secret_weight"], route="route_b"),
        ]
        return records

    def test_routes_a_and_c_are_infeasible_when_the_circuits_mismatch(self):
        summary = sr.summarize_slot_reduction(self._full_set())
        assert summary["routes"]["route_a"]["matches_target"] is False
        assert summary["routes"]["route_c"]["matches_target"] is False
        assert set(summary["conclusion"]["infeasible_routes"]) == {"route_a", "route_c"}

    def test_route_b_saving_smaller_than_the_noise_is_not_a_conclusion(self):
        # 8/1432 = 0.6% 的"节省"，而批间差是 0%……先构造一个明确的噪声更大的例子
        records = self._full_set(extract_seq_dup=1600.0)
        summary = sr.summarize_slot_reduction(records)
        assert summary["noise"]["relative_spread"] > 0.1
        assert "批间差" in summary["routes"]["route_b"]["verdict"]

    def test_route_b_with_identical_public_and_secret_weight_is_a_no_saving(self):
        records = self._full_set(extract_seq_secret_weight=1424.0)
        summary = sr.summarize_slot_reduction(records)
        assert "无收益" in summary["routes"]["route_b"]["verdict"]

    def test_swar_tree_is_compared_on_the_same_semantics(self):
        summary = sr.summarize_slot_reduction(self._full_set())
        item = summary["routes"]["swar_tree"]
        assert item["sequential_bytes_per_element"] == 1496.0
        assert item["swar_bytes_per_element"] == 1888.0
        assert "不省" in item["verdict"]

    def test_missing_baseline_gives_no_verdict(self):
        summary = sr.summarize_slot_reduction([self._record("fullmul", 1.0, match=False)])
        assert summary["routes"] == {}
        assert any("基准" in note for note in summary["notes"])

    def test_blocked_environment_gives_no_numbers(self):
        blocked = CapabilityReport(runnable=False, blockers=("测试",))
        record = sr.run_slot_reduction_case(sr.slot_reduction_cases()[0], report=blocked)
        assert record["status"] == "unavailable"
        assert record["comm_total_bytes"] is None
        assert "不给任何实测数字" in record["note"]

    def test_csv_columns_cover_every_blank_record_key(self):
        record = sr._blank_record(sr.slot_reduction_cases()[0])
        for column in sr.SLOT_REDUCTION_CSV_COLUMNS:
            assert column in record, f"CSV 列 {column} 不在记录里"

    def test_formatter_survives_an_empty_table(self):
        assert "结论" not in sr.format_slot_reduction_summary([])
