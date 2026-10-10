# -*- coding: utf-8 -*-
"""外部 PSI 执行档接入**方案层**（Phase 8）：后端族标注与登记一致性。

此前 `--psi-count psi-ca` / `--psi-sum pjc` 只改执行阶段：计划步骤的
``backend`` 仍写登记规则里的 "PSI"，与执行完成后给出的状态词
``count-only`` / ``count-and-sum`` 自相矛盾——方案层说 PSI，执行层跑的是
PSI-CA / PI-Sum。本模块锁定修复后的口径：

1. 执行档承接的步骤：``execution_backend`` = 该族的方案层后端名，
   ``backend`` 保持登记值（"PSI"）；对外展示走 ``reported_backend``；
2. 档与结果策略、协议泄漏面一一绑定，随步骤的 ``reasons`` 留痕；
3. 未选档时一切不变（``execution_backend is None``，展示仍是 "PSI"）；
4. 规划层的字面量登记与 backends 的能力登记逐项一致（交叉断言防漂移）；
5. 最终状态表的 Backend 列跟着执行档走，不只有状态词跟着走。
"""

from __future__ import annotations

import pytest

from backends.psi_ca_backend import (
    PSI_CA_BACKEND,
    PSI_CA_PROTOCOL,
    PSI_CA_PROTOCOL_LEAK,
    PSI_CA_SUPPORTED_OPS,
)
from backends.psi_ca_backend import capability as ca_capability
from backends.psi_sum_backend import (
    PSI_SUM_BACKEND,
    PSI_SUM_PROTOCOL_LEAK,
    PSI_SUM_RESULT_POLICY,
    PSI_SUM_SUPPORTED_OPS,
)
from frontend import parse_source
from geosecure import Compiler
from geosecure.cli import main as cli_main
from geosecure.compiler import operator_status_table
from planner import (
    EXTERNAL_PSI_FAMILY_BACKENDS,
    EXTERNAL_PSI_FAMILY_OPS,
    EXTERNAL_PSI_FAMILY_PROTOCOL_LEAK,
    EXTERNAL_PSI_FAMILY_PSI_CA,
    EXTERNAL_PSI_FAMILY_PSI_SUM,
    EXTERNAL_PSI_FAMILY_RESULT_POLICY,
    plan_program,
    resolve_external_psi_family,
)
from tests._helpers import example
from tests._psi_ca_stub import install_stub_psi
from tests._psi_sum_stub import install, make_bin_dir

#: 计数档唯一可承接的形态：只走 CellSetIntersect。
COUNT_PROGRAM = """\
from geo_privacy import geo


def conflict_count(route, no_fly_zone):
    return geo.cellset_intersect(route, no_fly_zone)
"""

#: 求和档可承接的形态（关联值经 psi_sum_weights 提供，与 CLI 档一致）。
SUM_PROGRAM = """\
from geo_privacy import geo


def conflict_weight_sum(route, no_fly_zone):
    return geo.cellset_intersect(route, no_fly_zone)
"""

#: Intersects 不在两个外部档的承接清单内（只有 CellSetIntersect 在）。
INTERSECTS_PROGRAM = """\
from geo_privacy import geo


def check_conflict(route, no_fly_zone):
    return geo.intersects(route, no_fly_zone)
"""

ROUTE_WEIGHTS = {3076691977860022276: 1, 3076832715348377604: 4, 3076973452836732932: 9}


def _plan(source, **kwargs):
    return plan_program(parse_source(source).program, **kwargs)


class TestRegistryCrossAssertions:
    """规划层字面量登记 vs backends 能力登记：逐项一致，防两处漂移。"""

    def test_family_names_match_backend_switches(self):
        assert set(EXTERNAL_PSI_FAMILY_BACKENDS) == {PSI_CA_BACKEND, PSI_SUM_BACKEND}
        assert EXTERNAL_PSI_FAMILY_PSI_CA == PSI_CA_BACKEND
        assert EXTERNAL_PSI_FAMILY_PSI_SUM == PSI_SUM_BACKEND

    def test_supported_ops_match_capability_registry(self):
        assert EXTERNAL_PSI_FAMILY_OPS[PSI_CA_BACKEND] == tuple(PSI_CA_SUPPORTED_OPS)
        assert EXTERNAL_PSI_FAMILY_OPS[PSI_SUM_BACKEND] == tuple(PSI_SUM_SUPPORTED_OPS)

    def test_protocol_leaks_match_and_stay_distinct(self):
        assert EXTERNAL_PSI_FAMILY_PROTOCOL_LEAK[PSI_CA_BACKEND] == PSI_CA_PROTOCOL_LEAK
        assert EXTERNAL_PSI_FAMILY_PROTOCOL_LEAK[PSI_SUM_BACKEND] == PSI_SUM_PROTOCOL_LEAK
        # 三族泄漏面互不相同：libpsi 是交集本体，外部两族各自更小
        assert len(set(EXTERNAL_PSI_FAMILY_PROTOCOL_LEAK.values())) == 2

    def test_result_policies_match_the_backends(self):
        assert EXTERNAL_PSI_FAMILY_RESULT_POLICY[PSI_CA_BACKEND] == "REVEAL_COUNT"
        assert (
            EXTERNAL_PSI_FAMILY_RESULT_POLICY[PSI_SUM_BACKEND] == PSI_SUM_RESULT_POLICY
        )

    def test_backend_labels_are_the_names_used_in_docs_and_cli(self):
        # 这两个名字是 README / docs / CLI 输出的既有口径，不能被悄悄改掉
        assert EXTERNAL_PSI_FAMILY_BACKENDS[PSI_CA_BACKEND] == PSI_CA_PROTOCOL
        assert EXTERNAL_PSI_FAMILY_BACKENDS[PSI_SUM_BACKEND] == "PI-Sum"

    def test_resolve_rejects_unknown_family(self):
        assert resolve_external_psi_family(None) is None
        assert resolve_external_psi_family("") is None
        assert resolve_external_psi_family("psi-ca") == PSI_CA_BACKEND
        with pytest.raises(ValueError, match="未知外部 PSI 执行档"):
            resolve_external_psi_family("nope")


class TestDefaultUnchanged:
    def test_planner_default_has_no_execution_family(self):
        step = _plan(COUNT_PROGRAM).steps[0]
        assert step.backend == "PSI"
        assert step.execution_backend is None
        assert step.reported_backend == "PSI"
        assert step.to_dict()["execution_backend"] is None

    def test_default_final_table_keeps_backend_psi(self):
        result = Compiler().compile_file(example("route_conflict.py"))
        assert result.operator_status[0]["backend"] == "PSI"


class TestCountModePlanning:
    def test_plan_marks_the_psi_ca_family(self):
        step = _plan(COUNT_PROGRAM, psi_count="psi-ca").steps[0]
        # 登记值不被改写：规则说 PSI，实际执行族是 PSI-CA，两个都要留下
        assert step.backend == "PSI"
        assert step.execution_backend == "PSI-CA"
        assert step.reported_backend == "PSI-CA"
        assert step.to_dict()["execution_backend"] == "PSI-CA"

    def test_plan_reason_records_policy_and_leak_of_the_family(self):
        step = _plan(COUNT_PROGRAM, psi_count="psi-ca").steps[0]
        joined = " ".join(step.reasons)
        assert "psi-ca" in joined
        assert "REVEAL_COUNT" in joined
        assert EXTERNAL_PSI_FAMILY_PROTOCOL_LEAK[PSI_CA_BACKEND] in joined

    def test_summary_reports_the_family_backend(self):
        plan = _plan(COUNT_PROGRAM, psi_count="psi-ca")
        assert plan.summary()["backends"] == ["PSI-CA"]

    def test_op_outside_the_family_keeps_its_registered_backend(self):
        step = _plan(INTERSECTS_PROGRAM, psi_count="psi-ca").steps[0]
        assert step.execution_backend is None
        assert step.reported_backend == "PSI"

    def test_planning_table_shows_the_family(self):
        from planner import plan_table_rows

        rows = plan_table_rows(_plan(COUNT_PROGRAM, psi_count="psi-ca"))
        assert rows[0][2] == "PSI-CA"

    def test_both_families_at_once_is_rejected(self):
        with pytest.raises(ValueError, match="不能同时启用"):
            _plan(COUNT_PROGRAM, psi_count="psi-ca", psi_sum="pjc")

    def test_bad_family_name_is_rejected(self):
        with pytest.raises(ValueError, match="未知外部 PSI 执行档"):
            _plan(COUNT_PROGRAM, psi_count="nope")


class TestSumModePlanning:
    def test_plan_marks_the_pi_sum_family(self):
        step = _plan(SUM_PROGRAM, psi_sum="pjc").steps[0]
        assert step.backend == "PSI"
        assert step.execution_backend == "PI-Sum"
        assert step.reported_backend == "PI-Sum"


class TestOperatorStatusBackendColumn:
    """最终状态表的 Backend 列必须与执行档一致（本模块要修的就是这一处）。"""

    def test_count_mode_row_backend_is_psi_ca(self, monkeypatch):
        install_stub_psi(monkeypatch)
        result = Compiler(psi_count="psi-ca").compile_source(COUNT_PROGRAM)
        row = result.operator_status[0]
        assert row["status"] == "count-only"
        assert row["backend"] == "PSI-CA"
        assert "PSI-CA" in operator_status_table(result)

    def test_count_mode_without_environment_still_names_the_family(self, monkeypatch):
        """环境缺 openmined-psi 也不能退回写 "PSI"：方案层已定档。"""

        def _raise_import_error():
            raise ImportError("No module named 'private_set_intersection'")

        monkeypatch.setattr(ca_capability, "import_openmined_psi", _raise_import_error)
        result = Compiler(psi_count="psi-ca").compile_source(COUNT_PROGRAM)
        row = result.operator_status[0]
        assert row["status"] == "backend-direct"
        assert row["backend"] == "PSI-CA"

    def test_sum_mode_row_backend_is_pi_sum(self, monkeypatch, tmp_path):
        install(monkeypatch)
        monkeypatch.setenv("GIS_SPU_PJC_BIN_DIR", make_bin_dir(tmp_path))
        result = Compiler(
            psi_sum="pjc", psi_sum_weights={"route": ROUTE_WEIGHTS}
        ).compile_source(SUM_PROGRAM)
        row = result.operator_status[0]
        assert row["status"] == "count-and-sum"
        assert row["backend"] == "PI-Sum"
        assert "PI-Sum" in operator_status_table(result)

    def test_unsupported_op_is_not_relabelled(self, monkeypatch):
        result = Compiler(psi_count="psi-ca").compile_source(INTERSECTS_PROGRAM)
        assert result.operator_status[0]["backend"] == "PSI"


class TestCliEndToEnd:
    def test_count_mode_cli_shows_the_family_in_both_tables(
        self, monkeypatch, capsys, tmp_path
    ):
        install_stub_psi(monkeypatch)
        path = tmp_path / "conflict_count.py"
        path.write_text(COUNT_PROGRAM, encoding="utf-8")
        exit_code = cli_main(["build", str(path), "--psi-count", "psi-ca"])
        output = capsys.readouterr().out
        assert exit_code == 0
        # 方案表与最终状态表都要写 PSI-CA（此前只有状态词 count-only 换了口径）
        assert "PSI-CA" in output
        assert "count-only" in output
