# -*- coding: utf-8 -*-
"""PI-Sum（--psi-sum pjc）的 CLI 层测试：契约、执行、权重载入。

桩见 ``tests/_psi_sum_stub.py``：它替换"唯一进程启动点" ``spawn_pjc`` 与
flag 核查 ``probe_binary_flags``，并按真实输入 CSV 做明文复算。
"""

from __future__ import annotations

import json

import pytest

from backends.psi_sum_backend import capability as su_capability
from geosecure import Compiler
from geosecure.cli import main as cli_main
from tests._helpers import example
from tests._psi_sum_stub import install, make_bin_dir

#: 示例航线（route_cells.csv）的三个格网码
ROUTE_A = 3076691977860022276
ROUTE_B = 3076832715348377604
ROUTE_C = 3076973452836732932
#: 与禁飞区相交的两个码（B、C）对应的权重之和 = 4 + 9 = 13
ROUTE_WEIGHTS = {ROUTE_A: 1, ROUTE_B: 4, ROUTE_C: 9}

WEIGHTS_CSV = (
    "grid_code,value\n"
    f"{ROUTE_A},1\n"
    f"{ROUTE_B},4\n"
    f"{ROUTE_C},9\n"
)

#: 只走 CellSetIntersect 的程序：求和档唯一可承接的形态。
SUM_PROGRAM = """\
from geo_privacy import geo


def conflict_weight_sum(route, no_fly_zone):
    # 冲突格网基数 + 冲突格网的风险权重和：一次算出两个数。
    return geo.cellset_intersect(route, no_fly_zone)
"""

#: 求和输出（两个数）喂给下一步 PSI 输入：不是集合，必须拒绝。
CHAINED_SUM_PROGRAM = """\
from geo_privacy import geo


def chained_sum(route, no_fly_zone, sensitive_area):
    first = geo.cellset_intersect(route, no_fly_zone)
    return geo.cellset_intersect(first, sensitive_area)
"""


class TestDefaultPathUnchanged:
    def test_default_build_keeps_libpsi_leak_sentence(self, capsys):
        exit_code = cli_main(["build", example("route_conflict.py")])
        output = capsys.readouterr().out
        assert exit_code == 0
        assert "接收方仍获得交集本体" in output
        assert "PI-Sum" not in output

    def test_default_compiler_has_no_sum_mode(self):
        result = Compiler().compile_file(example("route_conflict.py"))
        assert result.psi_sum is None
        assert result.psi_sum_capability is None
        for run in result.psi_runs.values():
            assert run.protocol != "PJC-PI-SUM"


class TestSumModeContract:
    def test_intersects_program_is_rejected_with_guidance(self, capsys):
        exit_code = cli_main(
            ["build", example("route_conflict.py"), "--psi-sum", "pjc"]
        )
        output = capsys.readouterr().out
        assert exit_code == 1
        assert "PI-Sum（--psi-sum pjc）无法编译本程序" in output
        assert "Intersects" in output
        assert "建议" in output
        assert "预计隐私计算代价" in output
        assert "PI-Sum 能力核查未通过（见上一阶段结论），未执行" in output

    def test_missing_weights_are_rejected_with_guidance(self, capsys):
        exit_code = cli_main(
            ["build", example("intersection_sum.py"), "--psi-sum", "pjc"]
        )
        output = capsys.readouterr().out
        assert exit_code == 1
        assert "没有关联值" in output
        assert "--psi-sum-weights" in output
        assert "不存在或不是目录" not in output

    def test_chained_sum_output_is_rejected(self):
        result = Compiler(
            psi_sum="pjc", psi_sum_weights={"route": ROUTE_WEIGHTS}
        ).compile_source(CHAINED_SUM_PROGRAM)
        stage = result.stage("psi_capability")
        assert stage.status == "error"
        assert "无法作为下游集合输入" in stage.message
        assert result.stage("psi_simulation").status == "skipped"
        assert result.psi_runs == {}

    def test_incomplete_weights_against_bound_input_are_rejected(self, tmp_path):
        path = tmp_path / "sum_program.py"
        path.write_text(SUM_PROGRAM, encoding="utf-8")
        result = Compiler(
            psi_sum="pjc",
            inputs={"route": example("route_cells.csv")},
            psi_sum_weights={"route": {ROUTE_B: 4}},
        ).compile_file(str(path))
        stage = result.stage("psi_capability")
        assert stage.status == "error"
        assert "覆盖不全" in stage.message

    def test_psi_sum_value_is_validated(self):
        with pytest.raises(ValueError, match="pjc"):
            Compiler(psi_sum="nope")

    def test_weights_shape_is_validated(self):
        with pytest.raises(ValueError, match="psi_sum_weights"):
            Compiler(psi_sum="pjc", psi_sum_weights={"route": [1, 2, 3]})

    def test_both_psi_paths_cannot_be_enabled_together(self):
        with pytest.raises(ValueError, match="不能同时启用"):
            Compiler(psi_count="psi-ca", psi_sum="pjc")

    def test_non_psi_program_discloses_the_inert_flag(self, capsys):
        exit_code = cli_main(
            ["build", example("distance_check.py"), "--psi-sum", "pjc"]
        )
        output = capsys.readouterr().out
        assert "--psi-sum pjc 未生效" in output

    def test_libpsi_switches_are_disclosed_as_unused(self, capsys):
        exit_code = cli_main(
            [
                "build",
                example("intersection_sum.py"),
                "--psi-sum",
                "pjc",
                "--psi-protocol",
                "PROTOCOL_KKRT",
            ]
        )
        output = capsys.readouterr().out
        assert "--psi-sum pjc 走 PI-Sum" in output
        assert "--psi-protocol / --psi-curve 在本档不生效" in output


class TestSumModeExecution:
    def test_environment_absence_is_a_warning_not_an_error(self, monkeypatch):
        monkeypatch.delenv("GIS_SPU_PJC_BIN_DIR", raising=False)
        result = Compiler(
            psi_sum="pjc", psi_sum_weights={"route": ROUTE_WEIGHTS}
        ).compile_source(SUM_PROGRAM)
        run = result.psi_runs["CellSetIntersect"]
        assert result.psi_sum == "pjc"
        assert run.protocol == "PJC-PI-SUM"
        assert run.result_policy["policy"] == "REVEAL_INTERSECTION_SUM"
        # 本机没有上游构建产物：绝不拿推测值填求和栏位
        assert run.status == "unavailable"
        assert run.value is None
        assert run.blockers
        assert result.stage("psi_simulation").status == "warning"
        assert result.operator_status[0]["status"] == "backend-direct"

    def test_stub_environment_full_green_through_cli(
        self, monkeypatch, capsys, tmp_path
    ):
        install(monkeypatch)
        monkeypatch.setenv("GIS_SPU_PJC_BIN_DIR", make_bin_dir(tmp_path))
        weights = tmp_path / "route_risk_weights.csv"
        weights.write_text(WEIGHTS_CSV, encoding="utf-8")
        exit_code = cli_main(
            [
                "build",
                example("intersection_sum.py"),
                "--psi-sum",
                "pjc",
                "--psi-sum-weights",
                f"route={weights}",
            ]
        )
        output = capsys.readouterr().out
        assert exit_code == 0
        assert "选用 PI-Sum" in output
        assert "1 个算子经 PI-Sum 真实执行" in output
        assert "(2, 13)" in output
        assert "count-and-sum" in output
        # 求和档的泄漏句必须与 libpsi 那句不同，且不能出现旧句
        assert "交集本体不交给任一方" in output
        assert "接收方仍获得交集本体" not in output

    def test_status_word_is_count_and_sum(self, monkeypatch, tmp_path):
        install(monkeypatch)
        monkeypatch.setenv("GIS_SPU_PJC_BIN_DIR", make_bin_dir(tmp_path))
        result = Compiler(
            psi_sum="pjc", psi_sum_weights={"route": ROUTE_WEIGHTS}
        ).compile_source(SUM_PROGRAM)
        run = result.psi_runs["CellSetIntersect"]
        assert run.status == "ok", run.error
        assert run.value == (2, 13)
        assert run.agreement is True
        assert result.stage("psi_simulation").status == "ok"
        assert result.operator_status[0]["status"] == "count-and-sum"
        assert result.operator_status[0]["representation"] == "CompactCellSet"

    def test_psi_sum_check_subcommand_reports_capabilities(self, capsys):
        exit_code = cli_main(["psi-sum-check"])
        output = capsys.readouterr().out
        assert "PI-Sum 能力核查" in output
        data = json.loads(output.split("\n", 2)[2])
        assert data["backend"] == "pjc"
        assert data["supported_ops"] == ["CellSetIntersect"]
        assert data["upstream"] == "google/private-join-and-compute"
        assert "blockers" in data
        assert exit_code in (0, 1)

    def test_psi_sum_check_honours_bin_dir(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(su_capability, "probe_binary_flags", lambda *a, **k: ())
        exit_code = cli_main(["psi-sum-check", "--bin-dir", make_bin_dir(tmp_path)])
        output = capsys.readouterr().out
        assert exit_code == 0
        assert '"runnable": true' in output

    def test_build_help_documents_the_sum_switches(self, capsys):
        with pytest.raises(SystemExit):
            cli_main(["build", "--help"])
        output = capsys.readouterr().out
        assert "--psi-sum" in output
        assert "--psi-sum-weights" in output
        assert "pjc" in output


class TestWeightsLoader:
    def test_csv_with_and_without_header(self, tmp_path):
        from geosecure.cli import _load_weight_map

        with_header = tmp_path / "a.csv"
        with_header.write_text(WEIGHTS_CSV, encoding="utf-8")
        without = tmp_path / "b.csv"
        without.write_text("5,7\n6,8\n", encoding="utf-8")
        assert _load_weight_map(str(with_header)) == ROUTE_WEIGHTS
        assert _load_weight_map(str(without)) == {5: 7, 6: 8}

    def test_json_form(self, tmp_path):
        from geosecure.cli import _load_weight_map

        path = tmp_path / "w.json"
        path.write_text('{"5": 7}', encoding="utf-8")
        assert _load_weight_map(str(path)) == {5: 7}

    def test_bad_shapes_raise_readable_errors(self, tmp_path):
        from geosecure.cli import _load_weight_map

        three = tmp_path / "c.csv"
        three.write_text("1,2,3\n", encoding="utf-8")
        not_int = tmp_path / "d.csv"
        not_int.write_text("1,x\n", encoding="utf-8")
        unsupported = tmp_path / "e.dat"
        unsupported.write_text("1,2\n", encoding="utf-8")

        with pytest.raises(ValueError, match="两列"):
            _load_weight_map(str(three))
        with pytest.raises(ValueError, match="整数对"):
            _load_weight_map(str(not_int))
        with pytest.raises(ValueError, match="只支持"):
            _load_weight_map(str(unsupported))
