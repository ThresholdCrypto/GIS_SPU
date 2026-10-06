# -*- coding: utf-8 -*-
"""位平面布局（D3）：按归约轴选 L1 / L2，条数账必须与课题口径逐项对得上。

为什么单独一个文件
==================
D3 的"条数下降多少"是**课题口径**，不是本项目自算的：课题组在
`outputs/三维格网接入层口径勘误与修正说明.docx` 表 3-1 里给出了自己的复算，
数字落在 `outputs/格网数据样例_明文与密态映射_v5.json` 的 `cost_prediction`。

本文件做三件事，缺一不可：
  1. **复算对账**：拿上面那份产物逐项核对本模块的公式（不另抄一份数字——
     抄出来的第二份没人再验）；
  2. **钉住边界**：不给形状就不给数字；小规模没有收益就退回逐点；未登记归约轴
     的算子不替它猜。这三条都是"不许编"的具体化；
  3. **前提与实测挂钩**：模型的前提是"通信量 ∝ 密文条数"，这条前提有
     `docs/mpc_comm_baseline.json` 的实测支持（本文件直接读产物核验），
     但**打包后的通信量仍是未验证的**，所以 `caveats` 里必须留着这句话。
"""

from __future__ import annotations

import json
import math
import os

import pytest

from planner import OPERATOR_REGISTRY, plan_program
from planner.layout import (
    LAYOUT_L1,
    LAYOUT_L2,
    LAYOUT_NAIVE,
    REDUCTION_AXIS,
    LayoutShape,
    layout_shape_from_mapping,
    plan_layout,
)
from frontend import parse_source

from tests._helpers import PROJECT_ROOT

SAMPLE_JSON = os.path.join(
    PROJECT_ROOT, "outputs", "格网数据样例_明文与密态映射_v5.json"
)
SAMPLE_JSON_V6 = os.path.join(
    PROJECT_ROOT, "outputs", "格网数据样例_明文与密态映射_v6.json"
)
COMM_ARTIFACT = os.path.join(PROJECT_ROOT, "docs", "mpc_comm_baseline.json")

MPC_VALUE_OPS = ("DistanceLE", "WeightedSum", "TemporalOverlap")
PSI_OPS = ("Intersects", "Contains", "CellSetIntersect")

DISTANCE_SOURCE = (
    "from geo_privacy import geo\n"
    "def f(p1, p2, threshold):\n"
    "    return geo.distance_le(p1, p2, threshold)\n"
)
INTERSECTS_SOURCE = (
    "from geo_privacy import geo\n"
    "def f(route, no_fly_zone):\n"
    "    return geo.intersects(route, no_fly_zone)\n"
)


def _sample_cost(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["cost_prediction"]


def _scenario_shape(cost: dict) -> LayoutShape:
    """课题场景的形状：候选 × 缓冲格网 × 属性 × b（b 从量化规则里读为 8）。"""

    return LayoutShape(
        candidates=cost["candidates"],
        cells=cost["buffer_cells"],
        attributes=cost["attrs"],
        bits=8,
    )


def _comm_rows() -> list[dict]:
    with open(COMM_ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# 1) 复算对账：课题产物的四个数必须能由本模块的公式算出
# --------------------------------------------------------------------------


class TestReconcilesWithTheCaseStudyArtifact:
    def test_shape_reproduces_the_published_value_count(self):
        cost = _sample_cost(SAMPLE_JSON)
        shape = _scenario_shape(cost)
        assert shape.values == cost["values"]
        assert shape.values == cost["naive_ciphertexts"]
        assert shape.values == (
            cost["candidates"] * cost["buffer_cells"] * cost["attrs"]
        )

    def test_l1_formula_reproduces_the_published_count(self):
        cost = _sample_cost(SAMPLE_JSON)
        plan = plan_layout("WeightedSum", _scenario_shape(cost))
        assert plan.l1_ciphertexts == cost["L1_same_cell_cross_attr"]
        assert plan.l1_ciphertexts == math.ceil(cost["values"] / cost["attrs"])

    def test_l2_formula_reproduces_the_published_count(self):
        cost = _sample_cost(SAMPLE_JSON)
        shape = _scenario_shape(cost)
        plan = plan_layout("DistanceLE", shape)
        assert plan.l2_ciphertexts == cost["L2_bitplane"]
        # 课题表 3-1 的算法原话："196000 除以 8192 向上取整再乘 8"
        assert plan.l2_ciphertexts == math.ceil(
            cost["values"] / cost["L2_bitplane_slots"]
        ) * 8

    def test_slots_are_the_padded_candidate_axis_times_bits(self):
        """课题载明 8192；本模块不硬编它，而是复算出同一条关系。"""

        cost = _sample_cost(SAMPLE_JSON)
        shape = _scenario_shape(cost)
        assert shape.slots_per_ciphertext == cost["L2_bitplane_slots"]
        assert shape.slots_per_ciphertext == 1024 * 8
        assert shape.slots_per_ciphertext == 2 ** math.ceil(
            math.log2(cost["candidates"])
        ) * 8

    def test_published_reduction_matches_the_ratio(self):
        cost = _sample_cost(SAMPLE_JSON)
        plan = plan_layout("DistanceLE", _scenario_shape(cost))
        assert plan.naive_ciphertexts == cost["naive_ciphertexts"]
        assert round(plan.reduction, 1) == cost["reduction_L2_vs_naive"]
        assert round(cost["values"] / cost["L2_bitplane"], 1) == (
            cost["reduction_L2_vs_naive"]
        )

    def test_the_two_shipped_versions_agree_on_the_cost_block(self):
        assert _sample_cost(SAMPLE_JSON) == _sample_cost(SAMPLE_JSON_V6)


# --------------------------------------------------------------------------
# 2) 轴向决策：按算子归约方向选布局
# --------------------------------------------------------------------------


class TestReductionAxis:
    def test_every_registered_operator_declares_an_axis(self):
        """新增算子时这里会红——必须显式回答"它的归约轴是什么"。"""

        assert set(REDUCTION_AXIS) == set(OPERATOR_REGISTRY)

    def test_axes_are_from_the_documented_set(self):
        assert set(REDUCTION_AXIS.values()) <= {None, "attribute", "candidate"}

    def test_weighted_sum_goes_l1_and_distance_goes_l2(self):
        shape = LayoutShape(candidates=1000, cells=49, attributes=4, bits=8)
        assert plan_layout("WeightedSum", shape).selected == LAYOUT_L1
        assert plan_layout("WeightedSum", shape).axis == "attribute"
        for op in ("DistanceLE", "TemporalOverlap"):
            assert plan_layout(op, shape).selected == LAYOUT_L2
            assert plan_layout(op, shape).axis == "candidate"

    def test_set_operators_are_out_of_scope(self):
        for op in PSI_OPS:
            plan = plan_layout(op, LayoutShape(candidates=1000))
            assert plan.applies is False
            assert plan.axis is None
            assert plan.selected == LAYOUT_NAIVE
            assert plan.reduction is None

    def test_height_band_is_out_of_scope(self):
        assert plan_layout("HeightBand").applies is False


# --------------------------------------------------------------------------
# 3) 边界：不给形状不给数字、没收益就退回、未登记就不猜
# --------------------------------------------------------------------------


class TestNoFabrication:
    def test_without_a_shape_no_numbers_are_reported(self):
        for op in MPC_VALUE_OPS:
            plan = plan_layout(op)
            assert plan.selected in (LAYOUT_L1, LAYOUT_L2)
            assert plan.naive_ciphertexts is None
            assert plan.selected_ciphertexts is None
            assert plan.reduction is None
            assert "不预测条数" in plan.basis

    def test_cost_keys_omit_counts_without_a_shape(self):
        keys = plan_layout("WeightedSum").to_cost_keys()
        assert keys["layout"] == LAYOUT_L1
        assert "N_ct_layout" not in keys
        assert "layout_reduction" not in keys

    def test_l2_without_gain_falls_back_to_pointwise(self):
        """L2 在小规模上反而更贵（条数 ≥ 逐点）时必须如实退回，不许报负数收益。"""

        tiny = LayoutShape(candidates=1, cells=1, attributes=1, bits=8)
        plan = plan_layout("DistanceLE", tiny)
        assert plan.selected == LAYOUT_NAIVE
        assert plan.selected_ciphertexts == plan.naive_ciphertexts
        assert plan.reduction == 1.0
        assert "没有收益" in plan.basis

    def test_l1_with_a_single_attribute_falls_back_to_pointwise(self):
        one_attr = LayoutShape(candidates=1000, cells=49, attributes=1, bits=8)
        plan = plan_layout("WeightedSum", one_attr)
        assert plan.selected == LAYOUT_NAIVE
        assert plan.selected_ciphertexts == plan.naive_ciphertexts

    def test_unknown_operator_is_not_guessed(self):
        plan = plan_layout("NoSuchOp", LayoutShape(candidates=1000, cells=49))
        assert plan.applies is False
        assert plan.selected == LAYOUT_NAIVE
        assert plan.reduction is None
        assert "未登记归约轴" in plan.basis

    def test_shape_rejects_non_positive_and_unknown_input(self):
        with pytest.raises(ValueError):
            LayoutShape(candidates=0)
        with pytest.raises(ValueError):
            LayoutShape(candidates=10, bits=0)
        with pytest.raises(ValueError):
            layout_shape_from_mapping({"candidates": 10, "attrs": 4})
        with pytest.raises(ValueError):
            layout_shape_from_mapping({"cells": 4})
        with pytest.raises(ValueError):
            layout_shape_from_mapping({"candidates": "many"})

    def test_caveats_disclaim_the_unbuilt_packed_circuit(self):
        plan = plan_layout("DistanceLE", LayoutShape(candidates=1000))
        joined = " ".join(plan.caveats)
        assert "打包电路尚未实现" in joined
        assert "不是实测通信量" in joined
        # 不得出现"实测提速/通信量下降 N 倍"这类没做过的断言
        assert "实测提速" not in joined


# --------------------------------------------------------------------------
# 4) 与实测挂钩：模型前提是"通信量 ∝ 条数"，这条要能从产物里看出来
# --------------------------------------------------------------------------


class TestMeasuredPremise:
    def _rows(self, op: str, field: str = "FM64") -> list[dict]:
        return [
            row
            for row in _comm_rows()
            if row.get("op") == op
            and row.get("protocol") == "ABY3"
            and row.get("field") == field
            and (row.get("strategy") or "") == ""
            and isinstance(row.get("comm_total_bytes"), (int, float))
            and row["comm_total_bytes"] > 0
        ]

    def test_communication_is_proportional_to_the_element_count(self):
        """ABY3 通信量 / 元素数 近似常数 —— 条数确实是通信量的主导项。

        这只支持模型的**前提**（条数少 → 通信少），不构成"打包后按同比例
        下降"的证据：打包电路还没写。
        """

        for op in ("DistanceLE", "WeightedSum"):
            rows = sorted(self._rows(op), key=lambda row: row["k"])
            assert len(rows) >= 3, f"{op} 的可比行不足"
            per_element = [row["comm_total_bytes"] / row["k"] for row in rows]
            assert all(14.0 <= value <= 20.0 for value in per_element), (
                op,
                list(zip([row["k"] for row in rows], per_element)),
            )
            # 线性：最大 K 与最小 K 的通信量之比 ≈ 规模比（容差 10%）
            ratio = rows[-1]["comm_total_bytes"] / rows[0]["comm_total_bytes"]
            scale = rows[-1]["k"] / rows[0]["k"]
            assert abs(ratio / scale - 1.0) <= 0.10, (op, ratio, scale)


# --------------------------------------------------------------------------
# 5) 接进方案层：estimated_cost 带上布局，但不覆盖四量
# --------------------------------------------------------------------------


class TestCostIntegration:
    def test_planner_records_the_layout_decision(self):
        plan = plan_program(parse_source(DISTANCE_SOURCE).program)
        cost = plan.steps[0].estimated_cost
        assert cost["layout"] == LAYOUT_L2
        assert cost["layout_axis"] == "candidate"
        assert "N_ct_layout" not in cost  # 没给形状
        # 四量口径没有被布局键覆盖或改名
        assert {"N_ct", "b", "d", "R"} <= set(cost)

    def test_planner_writes_the_reduction_back_when_given_a_shape(self):
        shape = LayoutShape(candidates=1000, cells=49, attributes=4, bits=8)
        plan = plan_program(
            parse_source(DISTANCE_SOURCE).program, layout_shape=shape
        )
        cost = plan.steps[0].estimated_cost
        assert cost["N_ct_naive"] == 196000
        assert cost["N_ct_layout"] == 192
        assert round(cost["layout_reduction"], 1) == 1020.8

    def test_psi_steps_carry_no_layout_keys(self):
        plan = plan_program(parse_source(INTERSECTS_SOURCE).program)
        cost = plan.steps[0].estimated_cost
        assert not any(key.startswith(("layout", "N_ct_")) for key in cost)

    def test_json_round_trip_keeps_the_layout_fields(self):
        """布局键必须能进编译 JSON（否则"写回 estimated_cost"只活在内存里）。"""

        shape = LayoutShape(candidates=1000, cells=49, attributes=4, bits=8)
        step = plan_program(
            parse_source(DISTANCE_SOURCE).program, layout_shape=shape
        ).steps[0]
        payload = json.loads(json.dumps(step.to_dict(), ensure_ascii=False))
        cost = payload["estimated_cost"]
        assert cost["layout"] == LAYOUT_L2
        assert cost["N_ct_naive"] == 196000
        assert cost["N_ct_layout"] == 192
