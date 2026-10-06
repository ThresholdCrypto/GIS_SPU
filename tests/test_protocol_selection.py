# -*- coding: utf-8 -*-
"""MPC 协议选择（P4）：候选集按**实测代价**排序、REF2K 不自动选中、报错可操作。

为什么单独一个文件
==================
在 P4 之前，"协议选择"在编译期其实没有选择依据：`mpc_protocol_candidates`
只是"允许显式指定"的清单，自动路径一律取登记默认值 `MPC_RULE_DEFAULT_PROTOCOL`。
这里补上的是一条**可复核的链路**：

    实测产物（docs/mpc_comm_baseline.json）
      → 候选集带代价（planner.registry.mpc_protocol_candidates_for）
      → 自动选择取实测最省（planner.registry.select_mpc_protocol）
      → 选择依据写进 plan.steps[].reasons（可留痕）
      → 拒绝时把"能用什么、多贵"写进 Diagnostic.suggestion

三条纪律（与仓库既有口径一致）：
  1. 排序只能来自实测产物——本文件直接读 JSON 核对，不让数字变成口号；
  2. 没有实测就**不编**：退化到登记默认值，并明说"不是实测结论"；
  3. "允许显式指定"与"允许自动选中"是两个层次——REF2K 属前者（对拍要能用），
     不属后者（无密码学保护，实测 send+recv 恒为 0 B）。
"""

from __future__ import annotations

import dataclasses
import json
import statistics

import pytest

from backends.spu_backend import SPU_PROTOCOLS_WITHOUT_CRYPTO
from frontend import parse_source
from planner import (
    MPC_PROTOCOL_CANDIDATES,
    MPC_RULE_DEFAULT_PROTOCOL,
    OPERATOR_REGISTRY,
    mpc_protocol_candidates_for,
    mpc_protocol_ranking_hint,
    plan_program,
    select_mpc_protocol,
    validate_mpc_protocol_for_operation,
)

import planner.registry as registry_module

from tests._helpers import has_spu

MPC_OPS = ("DistanceLE", "WeightedSum", "TemporalOverlap")

SOURCE = (
    "from geo_privacy import geo\n"
    "def f(p1, p2, threshold):\n"
    "    return geo.distance_le(p1, p2, threshold)\n"
)


def _rows():
    from backends.spu_backend.cost_baseline import artifact_path

    with open(artifact_path(), encoding="utf-8") as fh:
        return json.load(fh)


def _comparable_rows(rows, op: str, protocol: str, k: int):
    """与 cost_baseline 同一口径的可比行（FM64、默认电路、status=ok）。"""

    return [
        row
        for row in rows
        if row.get("op") == op
        and row.get("protocol") == protocol
        and row.get("k") == k
        and row.get("field") == "FM64"
        and (row.get("strategy") or "") == ""
        and row.get("status") == "ok"
        and isinstance(row.get("comm_total_bytes"), (int, float))
    ]


# --------------------------------------------------------------------------
# 1) 实测产物：判据来自测量，不来自推断
# --------------------------------------------------------------------------


class TestMeasuredCostBaseline:
    def test_ref2k_has_zero_communication_on_every_mpc_op(self):
        """`SPU_PROTOCOLS_WITHOUT_CRYPTO` 的依据必须能在产物里查到。

        这条把"REF2K 无密码学保护"钉在实测上：三个 MPC 算子的
        send+recv 都是 0 B。上游文档有没有这句话不重要，我们自己量过。
        """

        assert SPU_PROTOCOLS_WITHOUT_CRYPTO == ("REF2K",)
        rows = _rows()
        for op in MPC_OPS:
            candidates = mpc_protocol_candidates_for(op)
            ref2k = next(c for c in candidates if c.protocol == "REF2K")
            comparable = _comparable_rows(rows, op, "REF2K", ref2k.measured_k)
            assert comparable, f"{op} 没有 REF2K 的实测行"
            for row in comparable:
                assert row["comm_send_bytes"] == 0
                assert row["comm_recv_bytes"] == 0
                assert row["comm_total_bytes"] == 0

    def test_candidate_costs_come_from_the_artifact_medians(self):
        """候选上的每个数字都必须等于产物里同口径行的中位数（不是另抄一份）。"""

        rows = _rows()
        for op in MPC_OPS:
            for candidate in mpc_protocol_candidates_for(op):
                if candidate.measured_comm_total_bytes is None:
                    continue
                comparable = _comparable_rows(
                    rows, op, candidate.protocol, candidate.measured_k
                )
                assert comparable, (op, candidate.protocol)
                assert candidate.measured_comm_total_bytes == statistics.median(
                    float(row["comm_total_bytes"]) for row in comparable
                )
                assert candidate.measured_samples == len(comparable)

    def test_ranking_is_ascending_by_measured_communication(self):
        for op in MPC_OPS:
            measured = [
                c.measured_comm_total_bytes
                for c in mpc_protocol_candidates_for(op)
                if c.auto_selectable and c.measured_comm_total_bytes is not None
            ]
            assert measured == sorted(measured), (op, measured)


# --------------------------------------------------------------------------
# 2) 候选集：两个层次（允许显式指定 / 允许自动选中）
# --------------------------------------------------------------------------


class TestCandidateSet:
    def test_explicit_list_is_unchanged_and_still_contains_ref2k(self):
        """REF2K 留着：对拍/排查要能显式选。改动候选清单会连带改这条。"""

        for op in MPC_OPS:
            names = [c.protocol for c in mpc_protocol_candidates_for(op)]
            assert sorted(names) == sorted(MPC_PROTOCOL_CANDIDATES)
            assert "REF2K" in names

    def test_ref2k_is_never_auto_selectable_and_says_why(self):
        for op in MPC_OPS:
            ref2k = next(
                c for c in mpc_protocol_candidates_for(op) if c.protocol == "REF2K"
            )
            assert ref2k.auto_selectable is False
            assert ref2k.disqualification
            assert "无密码学保护" in ref2k.disqualification

    def test_other_protocols_stay_auto_selectable(self):
        for op in MPC_OPS:
            for candidate in mpc_protocol_candidates_for(op):
                if candidate.protocol == "REF2K":
                    continue
                assert candidate.auto_selectable is True
                assert candidate.disqualification is None

    def test_non_mpc_operators_yield_no_candidates(self):
        for op in ("Intersects", "Contains", "CellSetIntersect"):
            assert mpc_protocol_candidates_for(op) == ()
            assert select_mpc_protocol(op) is None
            assert mpc_protocol_ranking_hint(op) == ""

    def test_hint_excludes_ref2k_and_names_the_measurement_scale(self):
        hint = mpc_protocol_ranking_hint("DistanceLE")
        assert "已排除 REF2K" in hint
        assert "K=256" in hint
        assert "FM64" in hint


# --------------------------------------------------------------------------
# 3) 自动选择：按实测最省，且依据可见
# --------------------------------------------------------------------------


class TestAutoSelection:
    def test_select_picks_the_measured_cheapest(self):
        for op in MPC_OPS:
            selection = select_mpc_protocol(op)
            assert selection is not None
            assert selection.basis == "measured-comm"
            cheapest = min(
                (
                    c
                    for c in mpc_protocol_candidates_for(op)
                    if c.auto_selectable and c.measured_comm_total_bytes is not None
                ),
                key=lambda c: c.measured_comm_total_bytes,
            )
            assert selection.protocol == cheapest.protocol

    def test_current_measurements_agree_with_the_declared_default(self):
        """现有实测下 ABY3 恰好是最省的，所以自动选择与登记默认值一致。

        这不是"因为它是默认值"——`select_mpc_protocol` 只在实测里挑；
        一旦实测翻转，选择会跟着翻转，理由里会写明与默认值不同。
        """

        for op in MPC_OPS:
            assert select_mpc_protocol(op).protocol == MPC_RULE_DEFAULT_PROTOCOL

    def test_planner_records_the_selection_basis_in_reasons(self):
        plan = plan_program(parse_source(SOURCE).program)
        assert not plan.has_errors
        step = plan.steps[0]
        assert step.mpc_protocol == MPC_RULE_DEFAULT_PROTOCOL
        assert any("按实测代价选择" in reason for reason in step.reasons)
        assert any("从小到大" in reason for reason in step.reasons)

    def test_explicit_choice_overrides_and_is_disclosed(self):
        plan = plan_program(parse_source(SOURCE).program, mpc_protocol="CHEETAH")
        assert not plan.has_errors
        assert plan.steps[0].mpc_protocol == "CHEETAH"
        assert any("显式指定" in reason for reason in plan.steps[0].reasons)
        # 显式指定不走实测排序：不能同时出现"按实测代价选择"
        assert not any("按实测代价选择" in reason for reason in plan.steps[0].reasons)

    def test_explicit_ref2k_is_allowed_but_disclosed(self):
        """显式选 REF2K 要能过（对拍用），但规划里必须披露它不提供保护。"""

        plan = plan_program(parse_source(SOURCE).program, mpc_protocol="REF2K")
        assert not plan.has_errors
        step = plan.steps[0]
        assert step.mpc_protocol == "REF2K"
        assert any("无密码学保护" in reason for reason in step.reasons)

    def test_declared_default_is_only_a_fallback_without_measurements(
        self, monkeypatch
    ):
        """产物不可用时不得编数字：退回登记默认值，并明说不是实测结论。"""

        monkeypatch.setattr(
            "backends.spu_backend.cost_baseline.measured_comm_table",
            lambda op, **kwargs: {},
        )
        selection = select_mpc_protocol("DistanceLE")
        assert selection.protocol == MPC_RULE_DEFAULT_PROTOCOL
        assert selection.basis == "declared-default"
        assert "不是实测结论" in selection.reason
        assert all(
            c.measured_comm_total_bytes is None
            for c in selection.candidates
            if c.auto_selectable
        )

    def test_missing_artifact_reads_as_no_rows_not_as_empty_rows(self, tmp_path):
        from backends.spu_backend.cost_baseline import load_rows, measured_comm_table

        missing = str(tmp_path / "nope.json")
        assert load_rows(missing) is None
        assert measured_comm_table("DistanceLE", path=missing) == {}

    def test_artifact_without_the_operator_reads_as_empty_table(self, tmp_path):
        from backends.spu_backend.cost_baseline import measured_comm_table

        path = tmp_path / "partial.json"
        path.write_text(
            json.dumps([{"op": "OtherOp", "protocol": "ABY3", "status": "ok"}]),
            encoding="utf-8",
        )
        assert measured_comm_table("DistanceLE", path=str(path)) == {}


# --------------------------------------------------------------------------
# 4) 报错可操作：告诉用户"能用什么、多贵、为什么不行"
# --------------------------------------------------------------------------


class TestActionableErrors:
    def test_unknown_protocol_names_the_candidates_and_their_costs(self):
        check = validate_mpc_protocol_for_operation("DistanceLE", "SPDZ2K")
        assert not check.ok
        joined = " ".join(check.problems)
        assert "SPDZ2K" in joined
        assert "可用候选" in joined
        assert "ABY3" in joined and "SEMI2K" in joined
        assert "已排除 REF2K" in joined

    def test_diagnostic_suggestion_repeats_the_ranking(self):
        plan = plan_program(parse_source(SOURCE).program, mpc_protocol="SPDZ2K")
        assert plan.has_errors
        diagnostic = next(
            d for d in plan.diagnostics if d.code == "PROTOCOL_UNSUPPORTED"
        )
        assert diagnostic.location
        assert "按实测通信量从小到大" in diagnostic.suggestion
        assert "已排除 REF2K" in diagnostic.suggestion

    def test_registered_but_non_candidate_protocol_is_refused_with_alternatives(
        self, monkeypatch
    ):
        """候选清单外的**已登记**协议也必须被拒，并列出可用替代。

        正常注册表里 5 个协议都是候选，这条分支只在收窄候选时才可达——
        收窄一次来证明拒绝逻辑真的走了"清单外"这条路，而不是碰巧放行。
        """

        narrowed = dict(OPERATOR_REGISTRY)
        narrowed["DistanceLE"] = dataclasses.replace(
            OPERATOR_REGISTRY["DistanceLE"], mpc_protocol_candidates=("ABY3",)
        )
        monkeypatch.setattr(registry_module, "OPERATOR_REGISTRY", narrowed)

        check = validate_mpc_protocol_for_operation("DistanceLE", "SEMI2K")
        assert not check.ok
        joined = " ".join(check.problems)
        assert "不在算子 DistanceLE 的 MPC 候选协议清单" in joined
        assert "可用候选" in joined and "ABY3" in joined


# --------------------------------------------------------------------------
# 5) 一路贯到执行：编译期选择与执行期协议必须一致
# --------------------------------------------------------------------------


class TestSelectionReachesExecution:
    def test_auto_selection_reaches_execution(self):
        from geosecure import Compiler

        from tests._helpers import example

        result = Compiler().compile_file(example("distance_check.py"))
        assert result.ok, [s.message for s in result.stages if s.status == "error"]

        step = result.plan.steps[0]
        expected = select_mpc_protocol(step.operation).protocol
        assert step.mpc_protocol == expected

        if has_spu():
            run = result.spu_runs[step.operation]
            assert run.protocol == expected
            assert run.status == "ok", run.error

    def test_ranked_runner_up_also_reaches_execution(self):
        """排序第二贵的协议也要能真跑：证明候选集不是只有默认值能用。"""

        from geosecure import Compiler

        from tests._helpers import example

        candidates = [
            c
            for c in mpc_protocol_candidates_for("DistanceLE")
            if c.auto_selectable and c.measured_comm_total_bytes is not None
        ]
        runner_up = sorted(
            candidates, key=lambda c: c.measured_comm_total_bytes
        )[1].protocol

        result = Compiler(protocol=runner_up).compile_file(
            example("distance_check.py")
        )
        assert result.ok, [s.message for s in result.stages if s.status == "error"]
        assert result.plan.steps[0].mpc_protocol == runner_up

        if has_spu():
            run = result.spu_runs["DistanceLE"]
            assert run.protocol == runner_up
            assert run.status == "ok", run.error
