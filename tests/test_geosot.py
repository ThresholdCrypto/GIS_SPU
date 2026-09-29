"""GB/T 40087-2021 符合性 + 高度带（Z）语义测试。

三层证据：
    1. 纯公式层：与国标附录 A / 附录 B 的可核验特征值逐项比对；
    2. 位域容量层：7 位 Z 位域与 5 位 L 位域共同决定的可用层级与高度带；
    3. 集合语义层：高度带重叠在 64 位定长键上退化为集合交，且不产生假阴性。

特征值取自 GB/T 40087-2021：
    附录 A 表 A.1  各层级单元跨度与赤道尺度
    附录 B 式(B.4) H_n = (1+theta0)^n * r0 - r0，标准给出
                   H_255 = 519501834.1582395 m、r_255 = 525879971.1582395 m
                   H_-256 = -6302106.722602182 m、H_256 = 528680171.1252437 m
    附录 B 式(B.7) n = ln(1+H/r0)/ln(1+theta0)，对 H_255 应回到 255
这些值同时出现在 geosot_work-master/tests/test_gbt40087.py 的参考实现里，
两边独立复算得到同一组数字。

重要前提（见 TestEncodableLevels）：
上版的 64 位定长键里 L 只占 4 位，**只能编码 L<=15**，而 L<=15 时
0-1000 m 低空带恒为 1 层。**本版把 L 扩到 5 位**，可编码 L<=31，
低空带因此在 L22 上展开为 66 层。凡是要真正编成 64 位码的测试，
层级必须 <= 31；纯公式测试不受此限（它们只调用公式，不进位域）。
"""

from __future__ import annotations

import math

import pytest

from ir import (
    HEIGHT_LAYER_BITS,
    HEIGHT_LAYER_MAX,
    MAX_LEVEL,
    R0,
    THETA0,
    cell_deg,
    cells_per_deg,
    decode_grid_code,
    encode_grid_code,
    grid_code_height_interval,
    height_band_codes,
    height_cell,
    height_index,
    height_layer_bits,
    height_layer_bounds,
    height_layer_count,
    height_layer_fits,
    height_layer_lower,
    height_layers,
    layer_codes_for_height,
    max_supported_height,
    validate_height_layer,
)
from ir.geosot import equator_scale


# --------------------------------------------------------------------------
# 附录 A 表 A.1：水平剖分
# --------------------------------------------------------------------------


class TestAppendixA:
    def test_cell_deg_matches_standard(self):
        assert cell_deg(0) == pytest.approx(512.0)
        assert cell_deg(9) == pytest.approx(1.0)
        assert cell_deg(10) == pytest.approx(32.0 / 60.0)
        assert cell_deg(15) == pytest.approx(1.0 / 60.0)
        assert cell_deg(16) == pytest.approx(32.0 / 3600.0)
        assert cell_deg(21) == pytest.approx(1.0 / 3600.0)
        assert cell_deg(22) == pytest.approx(0.5 / 3600.0)
        assert cell_deg(32) == pytest.approx(1.0 / 2048.0 / 3600.0)

    def test_cells_per_deg_matches_standard(self):
        assert cells_per_deg(9) == 1
        assert cells_per_deg(10) == 2
        assert cells_per_deg(15) == 64
        assert cells_per_deg(16) == pytest.approx(3600.0 / 32)
        assert cells_per_deg(17) == 225

    def test_equator_scale_within_standard_tolerance(self):
        # 国标给的赤道尺度（米），容差 5%
        for level, expected_m in [
            (9, 111.3e3),
            (15, 1.8e3),
            (16, 989.5),
            (21, 30.9),
            (25, 1.9),
            (32, 0.015),
        ]:
            scale = equator_scale(level)
            assert abs(scale - expected_m) / expected_m < 0.05

    def test_level_range_is_enforced(self):
        for bad in (-1, MAX_LEVEL + 1):
            with pytest.raises(ValueError, match="层级"):
                cell_deg(bad)


# --------------------------------------------------------------------------
# 附录 B：高度域等比剖分
# --------------------------------------------------------------------------


class TestAppendixB:
    def test_height_cell_level9_is_first_layer_thickness(self):
        # 1 度网格的第 0 层厚度 = r0*((1+theta0)-1) = r0*theta0 ≈ 111.319 km
        assert height_cell(9) == pytest.approx(R0 * THETA0, abs=1e-6)

    def test_standard_characteristic_values(self):
        # 国标给出的四个特征值（式 B.4）。L=9 时 cell_deg=1，层号 == 自然索引 n
        assert height_layer_lower(255, 9) == pytest.approx(519501834.1582395, abs=0.01)
        assert R0 + height_layer_lower(255, 9) == pytest.approx(525879971.1582395, abs=0.01)
        assert height_layer_lower(-256, 9) == pytest.approx(-6302106.722602182, abs=0.01)
        assert height_layer_lower(256, 9) == pytest.approx(528680171.1252437, abs=0.01)

    def test_formula_b7_round_trip(self):
        # n(H_255) 必须回到 255（式 B.7 是式 B.4 的逆）
        n = math.log1p(519501834.1582395 / R0) / math.log1p(THETA0)
        assert abs(n - 255.0) < 1e-6
        assert height_index(519501834.1582395, 9) == 255

    def test_height_index_is_strict_inverse_of_layer_bounds(self):
        """核心不变量：层号唯一确定层，且该层包含原高度。

        覆盖全部 33 个层级（这是纯公式，不受 L 位域限制）。
        """

        for level in range(0, MAX_LEVEL + 1):
            for height in (0.0, 1.0, 10.0, 120.0, 300.0, 1000.0, 3000.0, 10000.0, 1e6):
                layer = height_index(height, level)
                lower, upper = height_layer_bounds(layer, level)
                assert lower - 1e-9 <= height < upper, (level, height, layer, lower, upper)

    def test_lower_than_minus_r0_is_rejected(self):
        with pytest.raises(ValueError, match="定义域"):
            height_index(-R0, 9)

    def test_layer_thickness_grows_geometrically(self):
        """层厚不是常数——第 8 层厚于第 0 层（L=9 时约 1.148 倍）。"""

        d0 = height_layer_lower(1, 9) - height_layer_lower(0, 9)
        d8 = height_layer_lower(9, 9) - height_layer_lower(8, 9)
        assert d8 > d0
        assert d8 / d0 == pytest.approx(1.14846, abs=1e-4)
        # 低空可编码层级（L=15）增长极小，这正是 height_cell 可作近似的原因
        e0 = height_layer_lower(1, 15) - height_layer_lower(0, 15)
        e8 = height_layer_lower(9, 15) - height_layer_lower(8, 15)
        assert e8 / e0 == pytest.approx(1.00231, abs=1e-5)


# --------------------------------------------------------------------------
# 高度带 -> 层集合（纯公式）
# --------------------------------------------------------------------------


class TestHeightLayers:
    def test_low_altitude_band_expands_to_layers(self):
        # 0-1000 m 在 L=19 上覆盖 9 层（纯公式；L=19 本身不可编码，见下节）
        assert height_layers(0, 1000, 19) == tuple(range(9))
        assert height_layer_count(0, 1000, 19) == 9
        assert height_layers(0, 1000, 18) == (0, 1, 2, 3, 4)
        assert height_layers(0, 1000, 17) == (0, 1, 2)
        assert height_layer_count(0, 1000, 21) == 33

    def test_band_layers_are_contiguous_and_ascending(self):
        for level in (15, 17, 18, 19, 20, 21, 22):
            layers = height_layers(0, 1000, level)
            assert list(layers) == sorted(layers)
            assert layers == tuple(range(layers[0], layers[-1] + 1))

    def test_endpoint_sharing_layer_is_included(self):
        """首尾相接的带必须共享端点所在层，否则会系统性漏报。"""

        assert set(height_layers(0, 300, 19)) & set(height_layers(300, 600, 19))
        assert set(height_layers(0, 1000, 19)) & set(height_layers(1000, 2000, 19))

    def test_reversed_bounds_are_normalized(self):
        assert height_layers(1000, 0, 19) == height_layers(0, 1000, 19)

    def test_layer_count_matches_band_layers(self):
        for level in (15, 17, 19, 21, 22):
            for hmax in (0.0, 100.0, 1000.0, 5000.0):
                assert height_layer_count(0, hmax, level) == len(height_layers(0, hmax, level))


# --------------------------------------------------------------------------
# Z 位域容量：7 位的适用边界
# --------------------------------------------------------------------------


class TestZFieldCapacity:
    def test_low_altitude_fits_until_level_22(self):
        # 0-1000 m 带上 Z7 的公式容量边界：L<=22 够，L>=23 溢出
        for level in range(0, 23):
            assert height_layer_fits(level, 1000.0), level
        for level in range(23, 33):
            assert not height_layer_fits(level, 1000.0), level

    def test_bits_required_matches_layers(self):
        for level, expected in [(17, 2), (18, 3), (19, 4), (20, 5), (21, 6), (22, 7), (23, 8)]:
            assert height_layer_bits(level, 1000.0) == expected, level

    def test_tighter_band_fits_at_higher_level(self):
        # 收窄高度带就能在更高层级上工作——给用户的替代方案，不是放宽 Z 位宽
        assert not height_layer_fits(23, 1000.0)
        assert height_layer_fits(23, 300.0)

    def test_max_supported_height_is_not_linear(self):
        """7 位 Z 覆盖的最大高度不是 127*height_cell——层厚随层号增长。"""

        for level in (9, 15, 19, 21):
            assert max_supported_height(level) >= HEIGHT_LAYER_MAX * height_cell(level)
        assert max_supported_height(9) == pytest.approx(height_layer_lower(127, 9))

    def test_validate_height_layer_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="上限"):
            validate_height_layer(HEIGHT_LAYER_MAX + 1, 19)
        with pytest.raises(ValueError, match="负"):
            validate_height_layer(-1, 19)
        validate_height_layer(0, 19)
        validate_height_layer(HEIGHT_LAYER_MAX, 19)

    def test_z_field_width_is_seven_bits(self):
        assert HEIGHT_LAYER_BITS == 7
        assert HEIGHT_LAYER_MAX == 127


# --------------------------------------------------------------------------
# 可编码层级：5 位 L 位域把层级上限抬到 31
# --------------------------------------------------------------------------


class TestEncodableLevels:
    """64 位键的 L 位域在本版由 4 位扩到 5 位（原 ver 位并入 L 高位，ver 恒为 0）。

    上版用 4 位时层级封在 15，而 L<=15 上 0-1000 m 低空带恒为 1 层，三维分辨力
    会退化成"同 XY 交即交"。扩到 5 位后上限是 31，但**真正的绑缚者换成了 Z：**
    0-1000 m 在 L23 上需 8 位层号，Z7 装不下，所以该带的可用上限是 L22。
    """

    def test_level_above_31_cannot_be_encoded(self):
        for level in (32, 33, 40):
            with pytest.raises(ValueError, match="L="):
                encode_grid_code(x=0, y=0, z=0, level=level, toff=0, lt=0)

    def test_level_31_is_the_maximum_encodable(self):
        encode_grid_code(x=0, y=0, z=0, level=31, toff=0, lt=0)

    def test_levels_16_to_31_are_newly_encodable(self):
        """这一段在 4 位 L 位域下全部 ValueError；本版必须全部可编码且可回读。"""

        for level in range(16, 32):
            code = encode_grid_code(x=1, y=2, z=0, level=level, toff=0, lt=0)
            assert decode_grid_code(code)["L"] == level

    def test_low_altitude_band_is_single_layer_at_L15(self):
        """旧位域封顶的 L15 上，整条低空带连一层都不满。"""

        for level in range(9, 16):
            assert height_layer_count(0, 1000, level) == 1, level
            assert height_index(1000.0, level) == 0, level

    def test_low_altitude_band_becomes_resolvable(self):
        """扩位的直接目的：低空带终于能看到分层。"""

        assert height_layer_count(0, 1000, 16) == 2
        assert height_layer_count(0, 1000, 19) == 9
        # Z7 位域内的最高可用层级是 L22
        assert height_layer_bits(22, 1000.0) == 7
        assert height_layer_fits(22, 1000.0)
        assert height_layer_count(0, 1000, 22) == 66
        # 再上一层 L23 需 8 位层号 -> Z7 装不下。上限仍在，只是从 L 挪到了 Z
        assert height_layer_bits(23, 1000.0) == 8
        assert not height_layer_fits(23, 1000.0)

    def test_wide_band_does_produce_multiple_encodable_layers(self):
        # L=15 上 0-20000 m 覆盖 11 层
        assert height_layer_count(0, 20000, 15) == 11
        assert height_layer_bits(15, 20000.0) == 4


class TestLowAltitudeResolvability:
    """扩位带来的**语义**收益：低空不再"同 XY 即相交"。

    这是 L 4->5 的真正目的，不只是“能编码更高层级”。旧上限 L15 上，
    0-100 m 与 800-1000 m 都塔缩成第 0 层，得到**同一个码**，于是判为相交——
    一个真实的**假阳性**：竖直上离开的两段空域会被报成冲突。
    L22 上两者分别是 7 层与 14 层，层号区间不相交，判定正确。
    """

    def _codes(self, band, level):
        return [
            encode_grid_code(x=21861, y=27702, z=layer, level=level, toff=0, lt=0)
            for layer in height_layers(*band, level)
        ]

    def test_L15_collapses_two_vertically_separated_bands(self):
        """旧上限下的假阳性：竖直升离的两段空域被编成同一个码。"""

        route = self._codes((0.0, 100.0), 15)
        zone = self._codes((800.0, 1000.0), 15)
        assert len(route) == 1 and len(zone) == 1
        assert route == zone, "两段应该被塔缩到同一层"

    def test_L22_separates_two_vertically_separated_bands(self):
        """扩位后判定正确：7 层 vs 14 层，层号区间不相交。"""

        route = self._codes((0.0, 100.0), 22)
        zone = self._codes((800.0, 1000.0), 22)
        assert len(route) == 7
        assert len(zone) == 14
        assert (height_index(0.0, 22), height_index(100.0, 22)) == (0, 6)
        assert (height_index(800.0, 22), height_index(1000.0, 22)) == (52, 65)
        assert not set(route) & set(zone), "两段不应有公共层"


# --------------------------------------------------------------------------
# 编码与高度互推（层级 <= 31）
# --------------------------------------------------------------------------


class TestHeightEncoding:
    def test_layer_codes_for_height_sets_z_from_height(self):
        code = layer_codes_for_height(x=21861, y=27702, level=15, height=10000.0, toff=0, lt=0)
        parts = decode_grid_code(code)
        assert parts["Z"] == height_index(10000.0, 15)
        assert parts["L"] == 15

    def test_layer_codes_for_height_rejects_overflow(self):
        # L=14 上 1000000 m 需要 8 位层号 —— 必须报错，绝不静默截断
        with pytest.raises(ValueError, match="超出"):
            layer_codes_for_height(x=0, y=0, level=14, height=1000000.0, toff=0, lt=0)

    def test_height_band_codes_one_per_layer(self):
        codes = height_band_codes(21861, 27702, 15, 0, 20000)
        assert len(codes) == 11
        assert [decode_grid_code(c)["Z"] for c in codes] == list(range(11))

    def test_height_band_codes_are_deterministic(self):
        assert height_band_codes(1, 2, 15, 0, 20000) == height_band_codes(1, 2, 15, 0, 20000)

    def test_height_band_codes_deduplicate_nothing(self):
        """同一层只出一个码，不会因带边界重复计数。"""

        codes = height_band_codes(1, 2, 15, 0, 20000)
        assert len(set(codes)) == len(codes)

    def test_grid_code_height_interval_decodes_physical_height(self):
        upper_h = height_layer_lower(11, 15) - 1e-6
        code = layer_codes_for_height(x=1, y=2, level=15, height=upper_h, toff=0, lt=0)
        layer, level, lower, upper = grid_code_height_interval(code)
        assert layer == 10 and level == 15
        assert lower <= upper_h < upper

    def test_physical_height_is_answerable_from_code(self):
        """旧版 Z 只是位域，"这个格网在多少米"无解；本版可直接回答。"""

        code = encode_grid_code(x=21861, y=27702, z=15, level=9, toff=2160, lt=4)
        layer, level, lower, upper = grid_code_height_interval(code)
        assert (layer, level) == (15, 9)
        # GB 口径下 Z=15 @ L=9 是 1.67 km 高度段
        assert (lower, upper) == pytest.approx(
            (height_layer_lower(15, 9), height_layer_lower(16, 9))
        )
        # 精确等比口径：15 层的下底面是 1.890 km，而不是线性近似 15*height_cell(9)=1.670 km。
        # 这个差值本身说明"层厚不是常数"，是本模块坚持精确口径的理由。
        assert lower == pytest.approx(1890064.770, abs=0.01)
        assert lower != pytest.approx(15 * height_cell(9), rel=1e-6)


# --------------------------------------------------------------------------
# 集合语义：高度带重叠 = 集合交（不需要额外 MPC 区间比较）
# --------------------------------------------------------------------------


class TestHeightBandSetSemantics:
    """全部用 L=15（可编码的最大层级）构造，保证测试检验的是真实码路径。"""

    LEVEL = 15

    def _band(self, hmin, hmax, label=None):
        from geo_privacy import geo

        return geo.height_band(1, 2, hmin, hmax, self.LEVEL)

    def test_overlapping_bands_intersect(self):
        from geo_privacy import geo

        lower = self._band(0, height_layer_lower(6, self.LEVEL) - 1)
        upper = self._band(height_layer_lower(4, self.LEVEL), height_layer_lower(11, self.LEVEL) - 1)
        assert geo.intersects(lower, upper) is True
        # 公共层为 4 与 5
        assert lower.intersection(upper).cardinality() == 2

    def test_disjoint_bands_do_not_intersect(self):
        from geo_privacy import geo

        low = self._band(0, height_layer_lower(3, self.LEVEL) - 1)
        high = self._band(height_layer_lower(20, self.LEVEL), height_layer_lower(30, self.LEVEL) - 1)
        assert geo.intersects(low, high) is False

    def test_same_xy_different_layer_is_not_a_conflict(self):
        """三维语义的实质：投影重合但高度层不同 → 不冲突。"""

        from geo_privacy import CellSet

        a = CellSet([encode_grid_code(x=1, y=2, z=0, level=self.LEVEL, toff=0, lt=0)])
        b = CellSet([encode_grid_code(x=1, y=2, z=8, level=self.LEVEL, toff=0, lt=0)])
        assert not a.intersects(b)

    def test_band_endpoint_contact_counts_as_overlap(self):
        from geo_privacy import geo

        # 以真实层边界切分，确保共享端点层
        cut = height_layer_lower(5, self.LEVEL)
        a = self._band(0, cut)
        b = self._band(cut, height_layer_lower(9, self.LEVEL) - 1)
        assert geo.intersects(a, b) is True
        assert a.intersection(b).cardinality() == 1

    def test_endpoint_only_representation_is_a_false_negative(self):
        """反面对照：只把带的两端各编一个码 → 漏掉中间层，假阴性。

        这是"必须把带展开成层集合"的直接证据，也是本版不做
        "height_band: [z_lo, z_hi] 区间参数"的原因。
        """

        from geo_privacy import CellSet, geo

        span = self._band(0, height_layer_lower(11, self.LEVEL) - 1)   # 层 0..10
        middle = self._band(height_layer_lower(4, self.LEVEL), height_layer_lower(7, self.LEVEL) - 1)
        assert middle.cardinality() == 3

        # 正确做法：整带展开，命中中间层
        assert geo.intersects(span, middle) is True

        # 错误做法：只取两端层（0 与 10）
        endpoints = CellSet(
            [
                encode_grid_code(x=1, y=2, z=0, level=self.LEVEL, toff=0, lt=0),
                encode_grid_code(x=1, y=2, z=10, level=self.LEVEL, toff=0, lt=0),
            ]
        )
        assert geo.intersects(endpoints, middle) is False  # 假阴性

    def test_band_codes_lie_in_z_field_range(self):
        band = self._band(0, height_layer_lower(20, self.LEVEL) - 1)
        for code in band.codes:
            z = decode_grid_code(code)["Z"]
            assert 0 <= z <= HEIGHT_LAYER_MAX
            assert decode_grid_code(code)["L"] == self.LEVEL

    def test_band_expansion_matches_formula_layer_count(self):
        for hmax in (1000.0, 20000.0, 50000.0):
            band = self._band(0, hmax)
            assert band.cardinality() == height_layer_count(0, hmax, self.LEVEL)
