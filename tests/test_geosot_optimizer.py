# -*- coding: utf-8 -*-
"""Geo-RR22 预处理（§8 / §9）：排序 / 去重 / 前缀压缩 / 分桶 + 真实 PSI 对拍。

纪律（与模块头一致，测试逐条锁定）：
- 去重是**正确性前提**：RR22/KKRT 遇重复键直接失败，ECDH 计数按重复乘数膨胀；
- 压缩 / 分桶必须**无损且确定性**；候选剪枝未实现，不在此丢弃任何码。
"""

from __future__ import annotations

import pytest

from backends.psi_backend import run_psi_intersection
from backends.psi_backend.benchmark import generate_benchmark_sets
from backends.psi_backend.geosot_optimizer import (
    GEOSOT_OPTIMIZER_VERSION,
    compress_grid_codes,
    deduplicate_grid_codes,
    fingerprint_grid_codes,
    partition_grid_codes,
    prepare_grid_codes,
    sort_grid_codes,
)

from tests._helpers import has_psi

needs_psi = pytest.mark.skipif(
    not has_psi(), reason="当前环境不具备真实 PSI 执行能力(python3.11+spu+libgomp1)"
)


class TestGridCodeBasics:
    def test_sort_is_ascending_and_keeps_duplicates(self):
        assert sort_grid_codes([3, 1, 2, 1]) == (1, 1, 2, 3)

    def test_dedup_keeps_first_occurrence_order(self):
        assert deduplicate_grid_codes([3, 1, 3, 2, 1]) == (3, 1, 2)

    def test_prepare_is_sorted_unique_with_stats(self):
        prep = prepare_grid_codes([3, 1, 3, 2])
        assert prep.codes == (1, 2, 3)
        assert prep.input_count == 4
        assert prep.duplicate_count == 1
        assert prep.unique_count == 3
        data = prep.to_dict()
        assert data["version"] == GEOSOT_OPTIMIZER_VERSION
        assert data["fingerprint"]

    def test_prepare_is_deterministic_regardless_of_input_order(self):
        first = prepare_grid_codes([3, 1, 2])
        second = prepare_grid_codes([2, 1, 3])
        assert first.codes == second.codes
        assert first.to_dict() == second.to_dict()

    def test_invalid_codes_rejected_with_position(self):
        with pytest.raises(ValueError, match="codes\\[1\\]"):
            prepare_grid_codes([1, (1 << 64)])
        with pytest.raises(ValueError, match="布尔值"):
            prepare_grid_codes([True])
        # 字符串会被尝试解析为十进制 / 0x 十六进制（与解析层同一口径），
        # 解析不出来才是 ValueError；浮点这类没有 __index__ 的类型是 TypeError。
        with pytest.raises(ValueError, match="无法解析"):
            prepare_grid_codes(["not-an-int"])
        with pytest.raises(TypeError):
            prepare_grid_codes([1.5])

    def test_fingerprint_is_order_and_duplicate_invariant(self):
        assert fingerprint_grid_codes([3, 1, 2]) == fingerprint_grid_codes([2, 3, 1, 1])
        assert len(fingerprint_grid_codes([])) == 16

    def test_fingerprint_differs_for_different_sets(self):
        assert fingerprint_grid_codes([1, 2]) != fingerprint_grid_codes([1, 3])


class TestCompression:
    def test_expand_restores_sorted_unique(self):
        codes = [5, 1, 3, 2, 1 << 63]
        comp = compress_grid_codes(codes)
        assert comp.expand() == (1, 2, 3, 5, 1 << 63)
        assert comp.code_count == 5
        assert comp.raw_bytes == 40

    def test_sibling_codes_share_a_prefix_group(self):
        base = (0xAB << 56) | 0xCDEF
        comp = compress_grid_codes([base, base + 1, base + 2])
        assert len(comp.groups) == 1
        group = comp.groups[0]
        assert group.shared_bytes == 7
        assert group.expand() == (base, base + 1, base + 2)
        assert comp.compressed_bytes < comp.raw_bytes

    def test_codes_with_different_top_bytes_form_multiple_groups(self):
        comp = compress_grid_codes([1, 1 << 63])
        assert len(comp.groups) == 2
        assert all(group.shared_bytes == 0 for group in comp.groups)

    def test_empty_and_singleton_are_lossless(self):
        empty = compress_grid_codes([])
        assert empty.groups == () and empty.expand() == ()
        single = compress_grid_codes([7])
        assert single.expand() == (7,)


class TestPartition:
    def test_partition_union_is_the_prepared_input(self):
        codes = list(range(0, 512, 7)) + [1 << 63, (1 << 64) - 1]
        buckets = partition_grid_codes(codes, num_buckets=16)
        assert len(buckets) == 16
        flat = tuple(sorted(code for bucket in buckets for code in bucket))
        assert flat == prepare_grid_codes(codes).codes

    def test_bucket_assignment_uses_top_bits(self):
        buckets = partition_grid_codes([5, 1 << 63], num_buckets=2)
        assert 5 in buckets[0]
        assert (1 << 63) in buckets[1]

    def test_num_buckets_must_be_power_of_two(self):
        for value in (0, 3, 12):
            with pytest.raises(ValueError, match="2 的幂"):
                partition_grid_codes([1], num_buckets=value)

    def test_single_bucket_contains_everything_sorted(self):
        assert partition_grid_codes([2, 1, 2], num_buckets=1) == ((1, 2),)


class TestPsiIntegration:
    """去重进入真实 PSI 输入路径（需要 spu 环境）。"""

    @needs_psi
    def test_ecdh_duplicate_input_is_deduped_before_protocol(self):
        bset = generate_benchmark_sets(2 ** 12, duplicate_ratio=0.25)
        run = run_psi_intersection(
            bset.left, bset.right, op="Intersects", protocol="PROTOCOL_ECDH"
        )
        assert run.status == "ok", run.error
        assert run.optimizer["applied"] is True
        assert run.optimizer["version"] == GEOSOT_OPTIMIZER_VERSION
        assert run.optimizer["left"]["duplicate_count"] == 1024
        assert run.optimizer["right"]["duplicate_count"] == 1024
        # 唯一口径：基线实测未去重时是 3072（重复乘数膨胀）
        assert run.intersection_count == 2048
        assert any("Geo-RR22 预处理" in note for note in run.notes)

    @needs_psi
    def test_optimized_result_matches_unique_baseline_element_wise(self):
        bset = generate_benchmark_sets(2 ** 12, duplicate_ratio=0.25)
        optimized = run_psi_intersection(
            bset.left, bset.right, op="CellSetIntersect", protocol="PROTOCOL_ECDH"
        )
        baseline = run_psi_intersection(
            sorted(set(bset.left)),
            sorted(set(bset.right)),
            op="CellSetIntersect",
            protocol="PROTOCOL_ECDH",
        )
        assert optimized.status == "ok" and baseline.status == "ok"
        assert optimized.value == baseline.value
        assert optimized.intersection_unique_count == baseline.intersection_unique_count

    @needs_psi
    def test_raw_mode_preserves_protocol_behaviour(self):
        """optimize_input=False：重复键按协议原始行为处理（基准 / 诊断口径）。"""

        bset = generate_benchmark_sets(2 ** 12, duplicate_ratio=0.25)
        run = run_psi_intersection(
            bset.left,
            bset.right,
            op="Intersects",
            protocol="PROTOCOL_ECDH",
            optimize_input=False,
        )
        assert run.status == "ok"
        assert run.optimizer == {"applied": False}
        assert run.intersection_count == 3072
        assert any("显式关闭" in note for note in run.notes)

    @needs_psi
    def test_rr22_duplicate_input_succeeds_after_dedup(self):
        """RR22 对重复键直接失败（Paxos）；预处理后同一输入可以正常完成。"""

        bset = generate_benchmark_sets(2 ** 10, duplicate_ratio=0.25)
        run = run_psi_intersection(
            bset.left,
            bset.right,
            op="CellSetIntersect",
            protocol="PROTOCOL_RR22",
            reference_fn=lambda left, right: tuple(sorted(set(left) & set(right))),
        )
        assert run.status == "ok", run.error
        assert run.optimizer["left"]["duplicate_count"] == 256
        assert run.agreement is True
        assert run.intersection_count == 512
