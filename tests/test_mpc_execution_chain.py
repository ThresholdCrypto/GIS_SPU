# -*- coding: utf-8 -*-
"""Planner → Runtime 的 MPC（SPU）协议参数一致性 —— Phase 4 对拍测试。

与 PSI 侧的 `test_execution_chain.py::TestConfigClosureAtCompileLevel` 对称：
那边对拍 `PsiRuntimeConfig` 的逐字段闭环，这边对拍 MPC 的协议闭环。

对拍口径（roadmap §五「严格打通 Planner → Runtime 参数」）：

    planned.mpc_protocol == runtime.protocol
    planned field        == runtime.field（同一配置源，规范化后）
    planned world_size   == runtime.world_size

回归重点（验收 §十八.5「Runtime 不允许静默更换协议」）：同一个程序里两个 MPC
步骤各自选中不同协议时，Runtime 必须**各跑各的**——此前执行层只取方案里第一个
MPC 步骤的协议套到所有步骤上，第 2 步会被静默换掉。
"""

from __future__ import annotations

import dataclasses

import pytest

from backends.spu_backend import (
    SpuRunResult,
    normalize_field,
    protocol_min_world_size,
)
from geosecure import Compiler
from geosecure.compiler import _check_mpc_closure
from planner import OPERATOR_REGISTRY, SELECTION_BASIS_MEASURED, select_mpc_protocol

import planner.registry as registry_module

from tests._helpers import example, has_spu

#: 覆盖全部三个 MPC 算子的真实样例（DistanceLE；WeightedSum + TemporalOverlap）
MPC_EXAMPLES = ("distance_check.py", "risk_score.py")


class TestPlannedProtocolReachesRuntime:
    @pytest.mark.parametrize("name", MPC_EXAMPLES)
    def test_every_mpc_step_runs_the_protocol_the_planner_chose(self, name):
        compiler = Compiler()
        result = compiler.compile_file(example(name))
        assert result.ok, [s.message for s in result.stages if s.status == "error"]

        mpc_steps = [s for s in result.plan.steps if s.mpc_protocol]
        assert mpc_steps, "样例里应当有 MPC 步骤"

        expected_field = normalize_field(compiler.field)
        for step in mpc_steps:
            run = result.spu_runs[step.operation]
            # 计划选了什么，执行就必须用什么
            assert run.protocol == step.mpc_protocol
            # 环宽 / 参与方数量同源于编译器配置：计划与执行不得漂移
            assert run.field == expected_field
            assert run.world_size == (
                compiler.world_size or protocol_min_world_size(step.mpc_protocol)
            )
            if has_spu():
                assert run.status == "ok", run.error

    def test_selected_protocol_is_the_measured_cheapest(self):
        """自动选择不是随手取的：等于 `select_mpc_protocol` 的实测结论。"""

        result = Compiler().compile_file(example("distance_check.py"))
        step = result.plan.steps[0]
        assert step.mpc_protocol == select_mpc_protocol(step.operation).protocol
        assert result.spu_runs[step.operation].protocol == step.mpc_protocol

    def test_json_plan_carries_the_mpc_protocol_for_cross_check(self):
        """编译 JSON 里"计划协议"与"运行协议"都要在，才谈得上逐层对拍。"""

        result = Compiler().compile_file(example("distance_check.py"))
        step = result.plan.steps[0]
        payload = result.to_dict()

        row = payload["plan"][0]
        assert row["mpc_protocol"] == step.mpc_protocol
        assert row["mpc_protocol_basis"] == step.mpc_protocol_basis
        assert payload["spu_runs"][step.operation]["protocol"] == step.mpc_protocol


class TestTwoMpcStepsKeepTheirOwnProtocols:
    """两个 MPC 步骤各选各的协议：回归「静默更换协议」。

    当前实测下三个 MPC 算子的自动选择都是同一个协议（ABY3 最省），所以"两步
    不同协议"要靠**收窄候选清单**构造——这是既有的测试手法（见
    `test_protocol_selection.py::test_registered_but_non_candidate_protocol_is_refused_with_alternatives`），
    只改"允许选什么"，不改"怎么选"。
    """

    @staticmethod
    def _narrow(monkeypatch, per_op):
        narrowed = dict(OPERATOR_REGISTRY)
        for op, candidates in per_op.items():
            narrowed[op] = dataclasses.replace(
                OPERATOR_REGISTRY[op], mpc_protocol_candidates=candidates
            )
        monkeypatch.setattr(registry_module, "OPERATOR_REGISTRY", narrowed)

    def test_each_step_runs_its_own_protocol(self, monkeypatch):
        self._narrow(
            monkeypatch,
            {"WeightedSum": ("SEMI2K",), "TemporalOverlap": ("ABY3",)},
        )
        result = Compiler().compile_file(example("risk_score.py"))
        assert result.ok, [s.message for s in result.stages if s.status == "error"]

        planned = {s.operation: s.mpc_protocol for s in result.plan.steps}
        assert planned == {"WeightedSum": "SEMI2K", "TemporalOverlap": "ABY3"}
        for step in result.plan.steps:
            run = result.spu_runs[step.operation]
            assert run.protocol == step.mpc_protocol
            assert step.mpc_protocol_basis == SELECTION_BASIS_MEASURED
            if has_spu():
                assert run.status == "ok", run.error
        # 两步协议确实不同，且执行层没有把它们统一成某一个
        assert len({r.protocol for r in result.spu_runs.values()}) == 2


class TestExplicitProtocolOverridesEveryStep:
    def test_single_explicit_protocol_applies_to_all_mpc_steps(self):
        result = Compiler(protocol="SEMI2K").compile_file(example("risk_score.py"))
        assert result.ok, [s.message for s in result.stages if s.status == "error"]

        steps = [s for s in result.plan.steps if s.mpc_protocol]
        assert steps
        for step in steps:
            assert step.mpc_protocol == "SEMI2K"
            run = result.spu_runs[step.operation]
            assert run.protocol == "SEMI2K"
            if has_spu():
                assert run.status == "ok", run.error

    def test_lowercase_explicit_protocol_is_normalized_at_the_runtime_boundary(self):
        """显式协议大小写宽松：执行层归一化后与计划认得出是同一个协议。"""

        result = Compiler(protocol="aby3").compile_file(example("distance_check.py"))
        assert result.ok, [s.message for s in result.stages if s.status == "error"]
        run = result.spu_runs["DistanceLE"]
        assert run.protocol == "ABY3"
        assert run.field == normalize_field(Compiler().field)


class TestClosureCheckerItself:
    """兜底检查本身也要有反面用例——否则"没报错"可能只是它从不报错。"""

    def test_a_swapped_protocol_is_flagged(self):
        result = Compiler().compile_file(example("distance_check.py"))
        step = result.plan.steps[0]
        swapped = SpuRunResult(
            status="ok", protocol="CHEETAH", field="FM64", world_size=2
        )
        problems = _check_mpc_closure(step, step.mpc_protocol, swapped)
        assert problems and "静默更换协议" in problems[0]

    def test_a_matching_protocol_is_silent(self):
        result = Compiler().compile_file(example("distance_check.py"))
        step = result.plan.steps[0]
        run = result.spu_runs[step.operation]
        assert _check_mpc_closure(step, step.mpc_protocol, run) == []
