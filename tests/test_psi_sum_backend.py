# -*- coding: utf-8 -*-
"""PI-Sum 后端（private-join-and-compute）测试：能力核查 + 执行路径 + 泄漏登记。

桩见 ``tests/_psi_sum_stub.py``：它替换"唯一进程启动点" ``spawn_pjc`` 与
flag 核查 ``probe_binary_flags``，并按真实输入 CSV 做明文复算后打印上游
格式的结果行。桩只保证"形态正确 + 结果按明文计算"，不替代真机验证。
"""

from __future__ import annotations

import pytest

from backends.psi_backend.protocol_registry import RESULT_SEMANTICS_EXACT
from backends.psi_backend.result_policy import PROTOCOL_LEAK_INTERSECTION_BODY
from backends.psi_ca_backend import PSI_CA_PROTOCOL_LEAK
from backends.psi_sum_backend import (
    PSI_SUM_BACKEND,
    PSI_SUM_BIN_ENV,
    PSI_SUM_OP_LEAKS,
    PSI_SUM_PROTOCOL,
    PSI_SUM_PROTOCOL_LEAK,
    PSI_SUM_RESULT_POLICY,
    PSI_SUM_SUPPORTED_OPS,
    PSI_SUM_SUPPORTED_POLICIES,
    PSI_SUM_UPSTREAM,
    PSI_SUM_UPSTREAM_COMMIT,
    PsiSumCapabilityReport,
    check_psi_sum_capabilities,
    run_psi_intersection_sum,
)
from backends.psi_sum_backend import capability as su_capability
from tests._psi_sum_stub import FakePjc, install, make_bin_dir

LEFT = (101, 102, 103)
RIGHT = (102, 103, 999)
VALUES = {101: 10, 102: 20, 103: 30}


def _plain_intersection(left, right):
    return tuple(sorted(set(int(code) for code in left) & set(int(code) for code in right)))


def _env(monkeypatch, tmp_path, fake=None):
    """桩环境：装好假产物，返回 (假产物, 可运行的能力报告)。"""

    fake = install(monkeypatch, fake)
    return fake, check_psi_sum_capabilities(make_bin_dir(tmp_path))


def _args_for(fake, prefix: str) -> dict[str, str]:
    return next(argmap for name, argmap in fake.calls if name.startswith(prefix))


class TestCapabilityProbe:
    def test_missing_bin_dir_is_blocked_with_build_hint(self, monkeypatch):
        monkeypatch.delenv(PSI_SUM_BIN_ENV, raising=False)
        report = check_psi_sum_capabilities()
        assert report.installed is False
        assert report.runnable is False
        assert report.bin_dir is None
        blockers = " ".join(report.blockers)
        assert "GIS_SPU_PJC_BIN_DIR" in blockers
        assert "bazel build //private_join_and_compute:all" in blockers
        data = report.to_dict()
        assert data["backend"] == PSI_SUM_BACKEND
        assert data["supported_ops"] == ["CellSetIntersect"]
        assert data["supported_policies"] == [PSI_SUM_RESULT_POLICY]

    def test_homonymous_pypi_packages_are_named_as_unrelated(self, monkeypatch):
        # 上游没有官方 PyPI 包；同名包是第三方，必须点名防止顶替。
        monkeypatch.delenv(PSI_SUM_BIN_ENV, raising=False)
        notes = " ".join(check_psi_sum_capabilities().notes)
        assert "private-join-and-compute" in notes and "pjc" in notes
        assert "无关" in notes

    def test_env_var_is_honoured(self, monkeypatch, tmp_path):
        bin_dir = make_bin_dir(tmp_path)
        monkeypatch.setattr(su_capability, "probe_binary_flags", lambda *a, **k: ())
        monkeypatch.setenv(PSI_SUM_BIN_ENV, bin_dir)
        report = check_psi_sum_capabilities()
        assert report.bin_dir == bin_dir
        assert report.runnable is True

    def test_explicit_bin_dir_wins_over_env(self, monkeypatch, tmp_path):
        bin_dir = make_bin_dir(tmp_path)
        monkeypatch.setattr(su_capability, "probe_binary_flags", lambda *a, **k: ())
        monkeypatch.setenv(PSI_SUM_BIN_ENV, str(tmp_path / "does-not-exist"))
        assert check_psi_sum_capabilities(bin_dir).bin_dir == bin_dir

    def test_nonexistent_dir_is_blocked(self, tmp_path):
        report = check_psi_sum_capabilities(str(tmp_path / "nope"))
        assert report.runnable is False
        assert any("不存在" in blocker for blocker in report.blockers)

    def test_empty_dir_lists_the_missing_artifacts(self, tmp_path):
        report = check_psi_sum_capabilities(str(tmp_path))
        assert report.runnable is False
        blockers = " ".join(report.blockers)
        assert "client" in blockers and "server" in blockers

    def test_flag_drift_blocks_execution(self, monkeypatch, tmp_path):
        # 形态变了就不执行：能力核查核对的是 flag 清单，不是"文件在不在"。
        monkeypatch.setattr(
            su_capability,
            "probe_binary_flags",
            lambda name, path, **kwargs: ("--paillier_modulus_size",),
        )
        report = check_psi_sum_capabilities(make_bin_dir(tmp_path))
        assert report.installed is True
        assert report.runnable is False
        assert any("--paillier_modulus_size" in item for item in report.api_problems)

    def test_binary_probe_failure_is_reported(self, monkeypatch, tmp_path):
        def _boom(name, path, **kwargs):
            raise OSError("Exec format error")

        monkeypatch.setattr(su_capability, "probe_binary_flags", _boom)
        report = check_psi_sum_capabilities(make_bin_dir(tmp_path))
        assert report.runnable is False
        assert any("Exec format error" in item for item in report.api_problems)

    def test_complete_artifacts_are_runnable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(su_capability, "probe_binary_flags", lambda *a, **k: ())
        report = check_psi_sum_capabilities(make_bin_dir(tmp_path))
        assert report.installed is True
        assert report.runnable is True
        assert report.version == PSI_SUM_UPSTREAM_COMMIT[:7]
        assert not report.api_problems
        assert set(report.binaries) == {"client", "server"}

    def test_notes_register_the_local_only_transport(self, monkeypatch, tmp_path):
        monkeypatch.setattr(su_capability, "probe_binary_flags", lambda *a, **k: ())
        notes = " ".join(check_psi_sum_capabilities(make_bin_dir(tmp_path)).notes)
        assert "LOCAL_TCP" in notes
        # 上游自陈"缓解措施未实现"必须如实带上，不能被抹掉
        assert "未实现" in notes
        assert "int64" in notes


class TestRuntimeExecution:
    def test_sum_and_count_agree_with_plain_reference(self, monkeypatch, tmp_path):
        fake, report = _env(monkeypatch, tmp_path)
        run = run_psi_intersection_sum(
            LEFT, RIGHT, left_values=VALUES, report=report, reference_fn=_plain_intersection
        )
        assert run.status == "ok", run.error
        assert run.value == (2, 50)
        assert run.intersection_count == 2
        assert run.intersection_unique_count == 2
        assert run.original_count == len(LEFT)
        assert run.agreement is True
        assert run.protocol == PSI_SUM_PROTOCOL
        assert run.curve is None
        assert run.receiver_rank == 0
        assert run.runtime_config == {}
        assert run.result_semantics == RESULT_SEMANTICS_EXACT
        assert run.protocol_params["impl"] == PSI_SUM_UPSTREAM
        assert run.protocol_params["commit"] == PSI_SUM_UPSTREAM_COMMIT[:7]
        assert fake.results == [(2, 50)]

    def test_client_is_the_left_side_carrying_the_values(self, monkeypatch, tmp_path):
        # 角色映射：client = 左侧（带关联值）。左侧码 101/102/103 的值
        # 10/20/30 里只有命中 102（20）与 103（30），结果为 50 即证明左侧带值。
        fake, report = _env(monkeypatch, tmp_path)
        run_psi_intersection_sum(LEFT, RIGHT, left_values=VALUES, report=report)
        assert fake.results == [(2, 50)]

    def test_upstream_flags_match_the_checked_contract(self, monkeypatch, tmp_path):
        fake, report = _env(monkeypatch, tmp_path)
        run_psi_intersection_sum(LEFT, RIGHT, left_values=VALUES, report=report)
        client_args = _args_for(fake, "client")
        server_args = _args_for(fake, "server")
        assert set(client_args) == {
            "client_data_file",
            "port",
            "paillier_modulus_size",
        }
        assert set(server_args) == {"server_data_file", "port"}
        assert client_args["paillier_modulus_size"] == "1536"
        # 上游默认监听 0.0.0.0；本项目收紧到回环（本路径只在本机跑）。
        assert client_args["port"] == "127.0.0.1:10501"
        assert server_args["port"] == "127.0.0.1:10501"

    def test_inputs_are_sorted_and_deduplicated_before_protocol(
        self, monkeypatch, tmp_path
    ):
        fake, report = _env(monkeypatch, tmp_path)
        run = run_psi_intersection_sum(
            (103, 101, 101, 102),
            (103, 103),
            left_values=VALUES,
            report=report,
            reference_fn=_plain_intersection,
        )
        assert run.status == "ok", run.error
        assert run.value == (1, 30)
        assert run.agreement is True
        assert any("去重" in note for note in run.notes)

    def test_empty_input_never_starts_the_subprocesses(self, monkeypatch, tmp_path):
        fake, report = _env(monkeypatch, tmp_path)
        run = run_psi_intersection_sum((), RIGHT, left_values={}, report=report)
        assert run.status == "empty-input"
        assert run.value == (0, 0)
        assert fake.calls == []

    def test_missing_associated_values_are_refused_not_zero_filled(
        self, monkeypatch, tmp_path
    ):
        fake, report = _env(monkeypatch, tmp_path)
        run = run_psi_intersection_sum(LEFT, RIGHT, left_values={101: 1}, report=report)
        assert run.status == "error"
        assert "不按 0 补齐" in run.error
        assert fake.calls == []

    def test_absent_values_mapping_is_refused(self, monkeypatch, tmp_path):
        fake, report = _env(monkeypatch, tmp_path)
        run = run_psi_intersection_sum(LEFT, RIGHT, report=report)
        assert run.status == "error"
        assert "必须显式提供关联值" in run.error
        assert fake.calls == []

    @pytest.mark.parametrize("bad", [-1, 2**63, 1.5, True])
    def test_out_of_range_values_are_refused(self, monkeypatch, tmp_path, bad):
        fake, report = _env(monkeypatch, tmp_path)
        run = run_psi_intersection_sum(
            LEFT, RIGHT, left_values={101: 1, 102: 2, 103: bad}, report=report
        )
        assert run.status == "error"
        assert "越界" in run.error
        assert fake.calls == []

    def test_unsupported_operator_is_rejected(self, monkeypatch, tmp_path):
        fake, report = _env(monkeypatch, tmp_path)
        run = run_psi_intersection_sum(
            LEFT, RIGHT, left_values=VALUES, op="Intersects", report=report
        )
        assert run.status == "error"
        assert "CellSetIntersect" in run.error
        assert fake.calls == []

    def test_non_sum_policies_are_rejected(self, monkeypatch, tmp_path):
        fake, report = _env(monkeypatch, tmp_path)
        for policy in ("REVEAL_INTERSECTION", "REVEAL_COUNT", "REVEAL_BOOLEAN"):
            run = run_psi_intersection_sum(
                LEFT, RIGHT, left_values=VALUES, result_policy=policy, report=report
            )
            assert run.status == "error", policy
            assert PSI_SUM_RESULT_POLICY in run.error
        assert fake.calls == []

    def test_layout_mismatch_is_rejected_before_the_protocol(self, monkeypatch, tmp_path):
        fake, report = _env(monkeypatch, tmp_path)
        run = run_psi_intersection_sum(
            LEFT,
            RIGHT,
            left_values=VALUES,
            left_layout={"layout_id": "layout-A", "version": 1},
            right_layout={"layout_id": "layout-B", "version": 1},
            report=report,
        )
        assert run.status == "error"
        assert "LAYOUT_MISMATCH" in run.error
        assert fake.calls == []

    def test_unavailable_environment_leaves_the_result_empty(self, monkeypatch, tmp_path):
        report = PsiSumCapabilityReport(
            installed=False, bin_dir=None, runnable=False, blockers=("没有构建产物",)
        )
        run = run_psi_intersection_sum(
            LEFT, RIGHT, left_values=VALUES, report=report, reference_fn=_plain_intersection
        )
        assert run.status == "unavailable"
        assert run.value is None
        assert run.blockers == ("没有构建产物",)
        assert any("保持空缺" in note for note in run.notes)

    def test_result_mismatch_is_an_error_not_a_pass(self, monkeypatch, tmp_path):
        fake, report = _env(monkeypatch, tmp_path, FakePjc(count_offset=1))
        run = run_psi_intersection_sum(
            LEFT, RIGHT, left_values=VALUES, report=report, reference_fn=_plain_intersection
        )
        assert run.agreement is False
        assert run.status == "error"
        assert "不一致" in run.error

    def test_missing_result_line_is_an_error(self, monkeypatch, tmp_path):
        fake, report = _env(monkeypatch, tmp_path, FakePjc(with_result_line=False))
        run = run_psi_intersection_sum(LEFT, RIGHT, left_values=VALUES, report=report)
        assert run.status == "error"
        assert "未产出可解析的结果行" in run.error

    def test_client_failure_with_zero_result_is_an_error(self, monkeypatch, tmp_path):
        # 退出码非 0 且报出 0/0：不能当成功——可能是半途失败后打印了默认值。
        # 空交集才有 0/0 的结果行，才谈得上"退出码非 0 却报 0/0"。
        fake, report = _env(monkeypatch, tmp_path, FakePjc(client_returncode=1))
        run = run_psi_intersection_sum(
            (101,), (999,), left_values={101: 5}, report=report
        )
        assert run.status == "error"
        assert "退出码" in run.error

    def test_green_run_registers_the_sum_visibility(self, monkeypatch, tmp_path):
        fake, report = _env(monkeypatch, tmp_path)
        run = run_psi_intersection_sum(LEFT, RIGHT, left_values=VALUES, report=report)
        assert any("交集内关联值之和=50" in note for note in run.notes)
        policy = run.result_policy
        assert policy["policy"] == PSI_SUM_RESULT_POLICY
        assert policy["business_value"] == "count+sum"
        assert policy["protocol_leak"] == PSI_SUM_PROTOCOL_LEAK
        assert "交集本体不离开任一方" in policy["leak_disclosure"]


class TestLeakRegistry:
    def test_leak_code_differs_from_the_other_two_paths(self):
        # 三条 PSI 路径的协议泄漏面互不相同，登记码必须互不相同——同名会让
        # 「PI-Sum 只出两个数」被误读成 libpsi / PSI-CA 的另一种写法。
        assert PSI_SUM_PROTOCOL_LEAK != PROTOCOL_LEAK_INTERSECTION_BODY
        assert PSI_SUM_PROTOCOL_LEAK != PSI_CA_PROTOCOL_LEAK

    def test_registry_covers_exactly_the_supported_ops(self):
        assert tuple(PSI_SUM_OP_LEAKS) == PSI_SUM_SUPPORTED_OPS
        assert PSI_SUM_SUPPORTED_POLICIES == (PSI_SUM_RESULT_POLICY,)

    def test_unknown_op_has_no_registered_leak(self):
        assert "WeightedSum" not in PSI_SUM_OP_LEAKS
