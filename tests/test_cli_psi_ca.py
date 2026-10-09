# -*- coding: utf-8 -*-
"""`--psi-count psi-ca`：编译期契约、CLI 输出与计数档执行。"""

from __future__ import annotations

import json

import pytest

from geosecure import Compiler
from geosecure.cli import main as cli_main
from tests._helpers import example
from tests._psi_ca_stub import install_stub_psi

#: 只走 CellSetIntersect 的程序：计数档唯一可承接的形态。
COUNT_PROGRAM = """\
from geo_privacy import geo


def conflict_count(route, no_fly_zone):
    # 冲突格网计数：只关心有多少个格网同时落在两边。
    return geo.cellset_intersect(route, no_fly_zone)
"""

#: 计数输出喂给下一步 PSI 输入：计数不是集合，必须拒绝。
CHAINED_COUNT_PROGRAM = """\
from geo_privacy import geo


def chained_count(route, no_fly_zone, sensitive_area):
    conflict = geo.cellset_intersect(route, no_fly_zone)
    return geo.cellset_intersect(conflict, sensitive_area)
"""


class TestDefaultPathUnchanged:
    def test_default_build_keeps_libpsi_leak_sentence(self, capsys):
        exit_code = cli_main(["build", example("route_conflict.py")])
        output = capsys.readouterr().out
        assert exit_code == 0
        assert "接收方仍获得交集本体" in output
        assert "PSI-CA" not in output

    def test_default_compiler_has_no_count_mode(self):
        result = Compiler().compile_file(example("route_conflict.py"))
        assert result.psi_count is None
        assert result.psi_ca_capability is None
        for run in result.psi_runs.values():
            assert run.protocol != "PSI-CA"


class TestCountModeContract:
    def test_intersects_program_is_rejected_with_guidance(self, capsys):
        exit_code = cli_main(
            ["build", example("route_conflict.py"), "--psi-count", "psi-ca"]
        )
        output = capsys.readouterr().out
        assert exit_code == 1
        assert "PSI-CA（--psi-count psi-ca）无法编译本程序" in output
        assert "Intersects" in output
        assert "建议" in output
        assert "预计隐私计算代价" in output
        assert "PSI-CA 能力核查未通过（见上一阶段结论），未执行" in output

    def test_mixed_program_lists_every_rejection(self, capsys):
        exit_code = cli_main(
            ["build", example("route_zone_chain.py"), "--psi-count", "psi-ca"]
        )
        output = capsys.readouterr().out
        assert exit_code == 1
        assert "Intersects" in output
        assert "计数无法作为下游集合输入" in output

    def test_chained_count_output_is_rejected(self):
        result = Compiler(psi_count="psi-ca").compile_source(CHAINED_COUNT_PROGRAM)
        stage = result.stage("psi_capability")
        assert stage.status == "error"
        assert "计数无法作为下游集合输入" in stage.message
        assert result.stage("psi_simulation").status == "skipped"
        assert result.psi_runs == {}

    def test_psi_count_value_is_validated(self):
        with pytest.raises(ValueError, match="psi-ca"):
            Compiler(psi_count="nope")

    def test_non_psi_program_discloses_the_inert_flag(self, capsys):
        exit_code = cli_main(
            ["build", example("distance_check.py"), "--psi-count", "psi-ca"]
        )
        output = capsys.readouterr().out
        assert exit_code == 0
        assert "--psi-count psi-ca 未生效" in output

    def test_libpsi_switches_are_disclosed_as_unused(self, capsys):
        cli_main(
            [
                "build", example("route_conflict.py"),
                "--psi-count", "psi-ca",
                "--psi-protocol", "KKRT",
            ]
        )
        output = capsys.readouterr().out
        assert "在本档不生效" in output


class TestCountModeExecution:
    def test_environment_absence_is_a_warning_not_an_error(self):
        from backends.psi_ca_backend import check_psi_ca_capabilities

        result = Compiler(psi_count="psi-ca").compile_source(COUNT_PROGRAM)
        run = result.psi_runs["CellSetIntersect"]
        assert run.protocol == "PSI-CA"
        assert run.result_policy["policy"] == "REVEAL_COUNT"
        assert result.psi_count == "psi-ca"
        if check_psi_ca_capabilities().runnable:
            assert run.status == "ok"
            assert run.value == 2
            assert run.agreement is True
            assert result.operator_status[0]["status"] == "count-only"
        else:
            # 本机没有 openmined-psi：绝不拿推测值填计数栏位
            assert run.status == "unavailable"
            assert run.value is None
            assert result.stage("psi_simulation").status == "warning"
            assert result.operator_status[0]["status"] == "backend-direct"

    def test_stub_environment_full_green_through_cli(self, monkeypatch, capsys, tmp_path):
        install_stub_psi(monkeypatch)
        path = tmp_path / "conflict_count.py"
        path.write_text(COUNT_PROGRAM, encoding="utf-8")
        exit_code = cli_main(["build", str(path), "--psi-count", "psi-ca"])
        output = capsys.readouterr().out
        assert exit_code == 0
        assert "选用 PSI-CA" in output
        assert "1 个算子经 PSI-CA 真实执行（只出交集基数，与明文计数一致）" in output
        assert "|A|=3  |A∩B|=2" in output
        assert "count-only" in output
        # 计数档的泄漏句必须与 libpsi 那句不同，且不能出现旧句
        assert "交集本体不交给任一方" in output
        assert "接收方仍获得交集本体" not in output

    def test_status_word_is_count_only(self, monkeypatch):
        install_stub_psi(monkeypatch)
        result = Compiler(psi_count="psi-ca").compile_source(COUNT_PROGRAM)
        run = result.psi_runs["CellSetIntersect"]
        assert run.status == "ok"
        assert run.value == 2
        assert run.agreement is True
        assert result.stage("psi_simulation").status == "ok"
        assert result.operator_status[0]["status"] == "count-only"
        assert result.operator_status[0]["representation"] == "CompactCellSet"

    def test_psi_ca_check_subcommand_reports_capabilities(self, capsys):
        exit_code = cli_main(["psi-ca-check"])
        output = capsys.readouterr().out
        assert "PSI-CA 能力核查" in output
        data = json.loads(output.split("\n", 2)[2])
        assert data["backend"] == "psi-ca"
        assert data["installed"] in (True, False)
        assert "blockers" in data
        assert exit_code in (0, 1)

    def test_build_help_documents_the_count_switch(self, capsys):
        with pytest.raises(SystemExit):
            cli_main(["build", "--help"])
        output = capsys.readouterr().out
        assert "--psi-count" in output
        assert "psi-ca" in output
