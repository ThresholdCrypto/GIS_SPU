# -*- coding: utf-8 -*-
"""链式执行 + 真实输入绑定 + CLI --input（Phase 2/3/4 / 验收 B、C、D）。

验收场景：CellSetIntersect 的真实 PSI 输出作为 Intersects 的输入，
而不是下游重新读取 DEFAULT_EXAMPLE_INPUTS。
"""

from __future__ import annotations

import json

import pytest

from backends.psi_backend import PsiRuntimeConfig
from geosecure import Compiler
from geosecure.cli import main as cli_main
from ir import encode_grid_code

from tests._helpers import example, has_psi


def _code(x: int) -> int:
    return encode_grid_code(x=x, y=27702, z=15, level=9, toff=0, lt=4)


#: 与 examples/*_cells.csv 同一批真值码（x 不同、其余同参）
ROUTE = (_code(21861), _code(21862), _code(21863))
NOFLY = (_code(21862), _code(21863), _code(22999))
SENSITIVE = (_code(21863), _code(22000))
EXPECTED_CONFLICT = (_code(21862), _code(21863))  # ROUTE ∩ NOFLY

needs_psi = pytest.mark.skipif(
    not has_psi(), reason="当前环境不具备真实 PSI 执行能力(python3.11+spu+libgomp1)"
)


def _chain_inputs() -> dict:
    return {
        "route": example("route_cells.csv"),
        "no_fly_zone": example("nofly_cells.csv"),
        "sensitive_area": example("sensitive_cells.csv"),
    }


class TestChainedExecution:
    @needs_psi
    def test_psi_output_feeds_the_next_operator(self):
        result = Compiler(inputs=_chain_inputs()).compile_file(
            example("route_zone_chain.py")
        )
        assert result.ok, [s.message for s in result.stages if s.status == "error"]

        upstream = result.psi_runs["CellSetIntersect"]
        assert upstream.status == "ok", upstream.error
        assert tuple(upstream.value) == EXPECTED_CONFLICT
        assert upstream.agreement is True

        downstream = result.psi_runs["Intersects"]
        assert downstream.status == "ok", downstream.error
        assert downstream.value is True
        # 链式证据：下游输入来自上一步 PSI 输出，而不是样例默认值
        assert any("上一步 PSI 输出" in note for note in downstream.notes)
        assert not any("样例兜底" in note for note in downstream.notes)

    @needs_psi
    def test_downstream_reference_is_computed_from_chained_value(self):
        """下游明文参考值必须基于链式实参——否则"对拍通过"没有意义。"""

        result = Compiler(inputs=_chain_inputs()).compile_file(
            example("route_zone_chain.py")
        )
        downstream = result.psi_runs["Intersects"]
        expected = bool(set(EXPECTED_CONFLICT) & set(SENSITIVE))
        assert downstream.reference is expected
        assert downstream.agreement is True

    @needs_psi
    def test_unbound_inputs_fall_back_with_disclosure(self):
        """一个都没绑定：按样例演示，但每条来源都要如实披露。"""

        result = Compiler().compile_file(example("route_zone_chain.py"))
        upstream = result.psi_runs["CellSetIntersect"]
        assert upstream.status == "ok", upstream.error
        assert any("样例兜底" in note for note in upstream.notes)
        message = result.stage("psi_simulation").message
        assert "回退到样例默认值" in message

    @needs_psi
    def test_partial_binding_marks_the_sample_side(self):
        """只绑了一部分：未绑定侧必须带"不得据此宣称真实数据验证"。"""

        result = Compiler(
            inputs={"route": example("route_cells.csv")}
        ).compile_file(example("route_zone_chain.py"))
        upstream = result.psi_runs["CellSetIntersect"]
        assert upstream.status == "ok", upstream.error
        notes = "\n".join(upstream.notes)
        assert "输入 route 来源：输入绑定" in notes
        assert "样例兜底（no_fly_zone 未绑定真实输入）" in notes
        assert "不得据此宣称真实数据验证" in notes

    def test_layout_mismatch_blocks_psi_in_chain(self):
        manifest = __import__("ir").grid_code_layout_manifest()
        bad = dict(manifest)
        bad["x_bits"] = 18
        bad["layout_id"] = "geosot3d-v0-x18-y16-z7-l5-toff14-lt4"
        result = Compiler(
            inputs=_chain_inputs(),
            input_layouts={"route": manifest, "no_fly_zone": bad},
        ).compile_file(example("route_zone_chain.py"))

        upstream = result.psi_runs["CellSetIntersect"]
        assert upstream.status == "error"
        assert upstream.error.startswith("LAYOUT_MISMATCH")
        assert not result.ok
        # 下游没有拿"默认样例"顶上去执行：链式输出缺值 = 输入无法装配 = error
        downstream = result.psi_runs["Intersects"]
        assert downstream.status == "error"
        assert "输入无法装配" in downstream.error
        assert not any("样例兜底" in note for note in downstream.notes)


class TestConfigClosureAtCompileLevel:
    @needs_psi
    def test_planner_params_equal_runtime_config_equal_result_record(self):
        """验收 A：Planner 写什么 = Runtime 注入什么 = 结果记录什么。"""

        result = Compiler(
            psi_protocol_params={"receiver_rank": 1},
            inputs={
                "route_A": example("route_cells.csv"),
                "NoFlyZone_B": example("nofly_cells.csv"),
            },
        ).compile_file(example("route_conflict.py"))

        step = result.plan.steps[0]
        run = result.psi_runs["Intersects"]
        assert step.protocol_params["receiver_rank"] == 1
        assert run.receiver_rank == 1
        assert run.runtime_config["receiver_rank"] == 1
        assert result.psi_runtime_config["receiver_rank"] == 1
        # 逐字段一致：计划参数（去掉被拆分进专属字段的键）== 运行记录参数
        assert run.runtime_config["protocol_params"] == {
            key: value
            for key, value in step.protocol_params.items()
            if key not in ("receiver_rank", "broadcast_result")
        }

    def test_invalid_protocol_params_rejected_at_construction(self):
        with pytest.raises(ValueError, match="receiver_rank"):
            Compiler(psi_protocol_params={"receiver_rank": 7})

    def test_unknown_protocol_params_rejected_at_construction(self):
        with pytest.raises(ValueError, match="未登记"):
            Compiler(psi_protocol_params={"lowcomm": True})

    def test_input_layouts_require_bound_inputs(self):
        manifest = __import__("ir").grid_code_layout_manifest()
        with pytest.raises(ValueError, match="未绑定输入"):
            Compiler(input_layouts={"route": manifest})

    def test_resolved_inputs_summary_is_json_serialisable(self):
        result = Compiler(inputs=_chain_inputs()).compile_file(
            example("route_zone_chain.py")
        )
        data = result.to_dict()
        json.dumps(data, ensure_ascii=False)
        assert set(data["inputs"]) == {"route", "no_fly_zone", "sensitive_area"}
        assert data["inputs"]["route"]["count"] == 3


class TestCliInputFlags:
    def test_cli_binds_real_inputs(self, capsys):
        exit_code = cli_main(
            [
                "build", example("route_zone_chain.py"),
                "--input", f"route={example('route_cells.csv')}",
                "--input", f"no_fly_zone={example('nofly_cells.csv')}",
                "--input", f"sensitive_area={example('sensitive_cells.csv')}",
            ]
        )
        output = capsys.readouterr().out
        assert exit_code == 0, output
        assert "输入绑定" in output
        if has_psi():
            assert "链式输入" in output
            assert "verified" in output

    def test_cli_rejects_malformed_input_flag(self, capsys):
        exit_code = cli_main(
            ["build", example("route_conflict.py"), "--input", "route_A"]
        )
        output = capsys.readouterr().out
        assert exit_code == 2
        assert "NAME=PATH" in output

    def test_cli_missing_input_file_exits_2(self, capsys):
        exit_code = cli_main(
            [
                "build", example("route_conflict.py"),
                "--input", "route_A=/definitely/not/here.csv",
            ]
        )
        output = capsys.readouterr().out
        assert exit_code == 2
        assert "输入文件不存在" in output

    def test_cli_layout_mismatch_exits_1_with_reason(self, capsys, tmp_path):
        manifest = __import__("ir").grid_code_layout_manifest()
        bad = dict(manifest)
        bad["y_bits"] = 16
        bad["layout_id"] = "geosot3d-v0-x17-y16-z7-l5-toff14-lt4"
        left_path = tmp_path / "left.json"
        right_path = tmp_path / "right.json"
        left_path.write_text(json.dumps(manifest), encoding="utf-8")
        right_path.write_text(json.dumps(bad), encoding="utf-8")

        exit_code = cli_main(
            [
                "build", example("route_conflict.py"),
                "--input", f"route_A={example('route_cells.csv')}",
                "--input", f"NoFlyZone_B={example('nofly_cells.csv')}",
                "--input-layout", f"route_A={left_path}",
                "--input-layout", f"NoFlyZone_B={right_path}",
            ]
        )
        output = capsys.readouterr().out
        assert exit_code == 1
        assert "LAYOUT_MISMATCH" in output
        assert "不一致（已拒绝执行）" in output
        assert "verified" not in output

    def test_cli_layout_without_input_exits_2(self, capsys, tmp_path):
        manifest_path = tmp_path / "layout.json"
        manifest_path.write_text(
            json.dumps(__import__("ir").grid_code_layout_manifest()),
            encoding="utf-8",
        )
        exit_code = cli_main(
            [
                "build", example("route_conflict.py"),
                "--input-layout", f"route_A={manifest_path}",
            ]
        )
        output = capsys.readouterr().out
        assert exit_code == 2
        assert "未绑定输入" in output