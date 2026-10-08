# -*- coding: utf-8 -*-
"""Geo-RR22「覆盖关系判定」的实证锁定（`docs/GEO_RR22_COVERAGE.md`）。

为什么需要这一个文件
==================
"候选集剪枝"是 Geo-RR22 预处理层唯一未落地项，而它的前置是**覆盖关系判定**。
报告里的结论是**否定的**（当前口径下不可实现），否定结论最容易被后来的改动
悄悄推翻，所以三个事实各留一条测试：

1. 国标码**有**前缀覆盖规则（构造性 + 真实产物逐条）；
2. 本项目 64 位码**不是**国标莫顿码（前缀规则不适用）；
3. 3D 码是 88 位，与二维 64 位码不同构。

第 2、3 条用的是**课题样例产物的真实码**，不是自造的：如果哪天有人改了
`GRID_CODE_BITS` 而让样例码"恰好能解出 L=21"，这里会红——那正说明布局变过，
必须重新出报告，而不是让文档继续声称一件已经变了的事。

课题样例产物在仓库外（`../geosot_work-master/out/codes.json`）。文件在就逐条跑，
不在就 `skip`——**不假装跑过**。
"""

from __future__ import annotations

import json
import os

import pytest

from ir.values import GRID_CODE_BITS, decode_grid_code

#: 课题样例里第一条二维码（level 21，closed_area.geojson），逐字取自产物
SAMPLE_CODE_L21 = 506519275675058176

#: 课题样例里第一条三维点码（dim=3，uav_track3d.csv），逐字取自产物
SAMPLE_CODE_3D = 215312933551597201232232484

#: 二维样例的层级（产物字段 `level` / `code_level`）
SAMPLE_LEVEL = 21

_ARTIFACT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "geosot_work-master",
    "out",
    "codes.json",
)


def _morton(row: int, col: int, level: int) -> int:
    """国标参考实现的莫顿交错（`geosot_core.py::morton` 的等价形式）。"""

    value = 0
    for index in range(level):
        value |= ((row >> index) & 1) << (2 * index + 1)
        value |= ((col >> index) & 1) << (2 * index)
    return value


def _route_b_code(row: int, col: int, level: int) -> int:
    """路线 B：2·L 位莫顿码左对齐到 64 位，低位补零。"""

    return _morton(row, col, level) << (64 - 2 * level)


def _parent_of(code: int, parent_level: int) -> int:
    """参考实现的父码公式 `parent_geo_num(code, level, parent_level)`。"""

    return code >> (64 - 2 * parent_level) << (64 - 2 * parent_level)


class TestPrefixPropertyHoldsOnTheNationalStandardCode:
    """第 1 条：国标码的前缀覆盖规则成立（构造性 + 真实产物的非构造性证据）。"""

    def test_parent_is_a_prefix_truncation(self):
        code = _route_b_code(0b101101, 0b010011, level=6)
        assert _parent_of(code, 5) == _route_b_code(0b10110, 0b01001, level=5)
        assert _parent_of(code, 3) == _route_b_code(0b101, 0b010, level=3)

    def test_low_bits_are_zero_at_every_level(self):
        for level in range(1, 32):
            code = _route_b_code(12345, 54321, level)
            assert code & ((1 << (64 - 2 * level)) - 1) == 0

    def test_covers_is_decidable_by_integer_shift(self):
        parent = _route_b_code(0b101, 0b010, level=3)
        for row, col in ((0b101000, 0b010000), (0b101111, 0b010111)):
            child = _route_b_code(row, col, level=6)
            assert _parent_of(child, 3) == parent

    def test_sample_code_carries_a_22_bit_zero_tail(self):
        assert SAMPLE_CODE_L21 & ((1 << (64 - 2 * SAMPLE_LEVEL)) - 1) == 0


@pytest.mark.skipif(not os.path.exists(_ARTIFACT), reason="课题样例产物不在本机")
class TestRealArtifactBacksThePrefixProperty:
    """真实产物逐条复算：21973 条二维码的低 22 位必须全零。"""

    @staticmethod
    def _entries():
        with open(_ARTIFACT, encoding="utf-8") as handle:
            return json.load(handle)

    def test_every_2d_code_has_zero_tail(self):
        data = self._entries()
        level = int(data["level"])
        assert level == SAMPLE_LEVEL
        tail = (1 << (64 - 2 * level)) - 1
        two_d = [
            int(entry["code"].split("-")[0])
            for entry in data["codes"]
            if int(entry["code"].split("-")[0]) < 2**64
        ]
        assert len(two_d) > 20000
        assert all(code & tail == 0 for code in two_d)

    def test_bytes16_high_half_is_the_code(self):
        # bytes16 = [64 位码][64 位扩展]：高 8 字节必须等于 code
        for entry in self._entries()["codes"][:500]:
            code = int(entry["code"].split("-")[0])
            if code >= 2**64:
                continue
            high = int.from_bytes(bytes.fromhex(entry["bytes16_hex"])[:8], "big")
            assert high == code

    def test_3d_codes_are_not_64_bit(self):
        wide = [
            int(entry["code"].split("-")[0])
            for entry in self._entries()["codes"]
            if int(entry["code"].split("-")[0]) >= 2**64
        ]
        assert wide, "样例里应当有 dim=3 的高位码（本测试就是为它存在的）"


class TestProjectLayoutIsNotTheMortonCode:
    """第 2 条：本项目的字段切分布局不能解出课题样例的真实层级。"""

    def test_layout_is_field_split_not_morton(self):
        assert GRID_CODE_BITS == {"X": 17, "Y": 17, "Z": 7, "L": 5, "Toff": 14, "Lt": 4}

    def test_sample_code_decodes_to_a_wrong_level(self):
        decoded = decode_grid_code(SAMPLE_CODE_L21)
        assert decoded["L"] != SAMPLE_LEVEL, (
            "样例码在本项目布局下解出的 L 不等于真实层级 21——"
            "国标前缀规则因此不能套用到本项目的码上；"
            "若这条断言变红，说明布局改过，docs/GEO_RR22_COVERAGE.md 必须重出"
        )

    def test_3d_sample_code_is_out_of_64_bit_range(self):
        assert SAMPLE_CODE_3D >= 2**64
        with pytest.raises(ValueError):
            decode_grid_code(SAMPLE_CODE_3D)


class TestPruningStaysUnimplemented:
    """第 3 条：剪枝不得实现（任务文档 §10：近似不得进入精确路径）。"""

    def test_optimizer_has_no_pruning_entry_point(self):
        from backends.psi_backend import geosot_optimizer as opt

        for forbidden in ("prune", "prune_grid_codes", "covers", "covers_grid_code"):
            assert not hasattr(opt, forbidden), (
                f"{forbidden} 出现了：剪枝的前置（覆盖关系判定）尚未定口径，"
                "实现它等于把近似引进精确路径；见 docs/GEO_RR22_COVERAGE.md §5"
            )

    def test_partition_discards_nothing(self):
        from backends.psi_backend.geosot_optimizer import partition_grid_codes

        codes = tuple(range(1, 40))
        buckets = partition_grid_codes(codes, num_buckets=4)
        assert sorted(code for bucket in buckets for code in bucket) == sorted(codes)
