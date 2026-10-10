# -*- coding: utf-8 -*-
"""PSI-Cardinality 后端：能力探测、API 核对、只出计数的执行与拒绝路径。"""

from __future__ import annotations

import types

import pytest

from backends.psi_backend.result_policy import (
    PROTOCOL_LEAK_INTERSECTION_BODY,
    REVEAL_COUNT,
)
from backends.psi_ca_backend import (
    PSI_CA_BACKEND,
    PSI_CA_COMMUNICATION_METER,
    PSI_CA_LEARNING_RANK,
    PSI_CA_OP_LEAKS,
    PSI_CA_PINNED_VERSION,
    PSI_CA_PROTOCOL,
    PSI_CA_PROTOCOL_LEAK,
    PSI_CA_STRUCTURE,
    PSI_CA_SUPPORTED_OPS,
    PSI_CA_SUPPORTED_POLICIES,
    check_psi_ca_capabilities,
    missing_api,
    run_psi_cardinality,
)
from backends.psi_ca_backend import capability as ca_capability
from tests._psi_ca_stub import install_stub_psi, make_stub, without_api

LEFT = (101, 102, 103)
RIGHT = (102, 103, 999)


def _plain_intersection(left, right):
    return tuple(sorted(set(int(code) for code in left) & set(int(code) for code in right)))


def _raise_import_error():
    raise ImportError("No module named 'private_set_intersection'")


class TestCapabilityProbe:
    def test_missing_package_is_blocked_with_install_hint(self, monkeypatch):
        monkeypatch.setattr(ca_capability, "import_openmined_psi", _raise_import_error)
        report = check_psi_ca_capabilities()
        assert report.installed is False
        assert report.runnable is False
        assert any("pip install" in blocker for blocker in report.blockers)
        data = report.to_dict()
        assert data["backend"] == PSI_CA_BACKEND
        assert data["supported_ops"] == ["CellSetIntersect"]
        assert data["supported_policies"] == [REVEAL_COUNT]
        assert data["structure"] == PSI_CA_STRUCTURE

    def test_api_drift_blocks_execution(self, monkeypatch):
        stub = without_api(make_stub(), methods=(("client", "GetIntersectionSize"),))
        monkeypatch.setattr(ca_capability, "import_openmined_psi", lambda: stub)
        report = check_psi_ca_capabilities()
        assert report.installed is True
        assert report.runnable is False
        assert any("GetIntersectionSize" in item for item in report.api_problems)

    def test_missing_module_attr_is_reported(self):
        stub = without_api(make_stub(), module_attrs=("ServerSetup",))
        problems = missing_api(stub)
        assert any("ServerSetup" in item for item in problems)

    def test_matching_api_is_runnable(self, monkeypatch):
        install_stub_psi(monkeypatch)
        report = check_psi_ca_capabilities()
        assert report.installed is True
        assert report.runnable is True
        assert report.version == PSI_CA_PINNED_VERSION
        assert not report.api_problems

    def test_version_mismatch_is_a_note_not_a_block(self, monkeypatch):
        install_stub_psi(monkeypatch, version="9.9.9")
        report = check_psi_ca_capabilities()
        assert report.runnable is True
        assert any("不一致" in note for note in report.notes)


class TestRuntimeExecution:
    def test_count_only_run_agrees_with_plain_reference(self, monkeypatch):
        stub = install_stub_psi(monkeypatch)
        run = run_psi_cardinality(LEFT, RIGHT, reference_fn=_plain_intersection)
        assert run.status == "ok", run.error
        assert run.value == 2
        assert run.intersection_count == 2
        assert run.intersection_unique_count == 2
        assert run.original_count == len(LEFT)
        assert run.agreement is True
        assert run.protocol == PSI_CA_PROTOCOL
        assert run.curve is None
        assert run.result_semantics == "exact"
        assert run.runtime_config == {}
        assert run.protocol_params["structure"] == PSI_CA_STRUCTURE
        assert run.protocol_params["impl"] == "openmined-psi"

        policy = run.result_policy
        assert policy["policy"] == REVEAL_COUNT
        assert policy["business_value"] == "count"
        assert policy["protocol_leak"] == PSI_CA_PROTOCOL_LEAK
        assert "交集本体不离开任一方" in policy["leak_disclosure"]

        # 上游调用秩序（与 psi_server.cpp 的约束对齐）：两侧都关交集本体、
        # fpr=0.0（RAW 档忽略）、数据结构 RAW。
        assert ("client.CreateWithNewKey", False) in stub.calls
        assert ("server.CreateWithNewKey", False) in stub.calls
        assert ("server.CreateSetupMessage", 0.0, len(LEFT), PSI_CA_STRUCTURE) in stub.calls

    def test_protocol_payload_is_measured_and_split_by_direction(self, monkeypatch):
        # Phase 11：量「协议消息的 protobuf 载荷」本身，不是网络观测
        # （本档是进程内链路、无 socket）。方向：send = client 的 Request；
        # recv = server 的 ServerSetup + Response。
        install_stub_psi(monkeypatch)
        run = run_psi_cardinality(LEFT, RIGHT, reference_fn=_plain_intersection)
        assert run.status == "ok", run.error
        comm = run.communication
        assert comm["meter"] == PSI_CA_COMMUNICATION_METER
        assert comm["send_bytes"] == comm["by_message"]["Request"]
        assert comm["recv_bytes"] == (
            comm["by_message"]["ServerSetup"] + comm["by_message"]["Response"]
        )
        assert comm["total_bytes"] == comm["send_bytes"] + comm["recv_bytes"]
        assert comm["client_to_server_bytes"] == comm["send_bytes"]
        assert comm["server_to_client_bytes"] == comm["recv_bytes"]
        assert comm["total_bytes"] > 0

    def test_payload_drift_leaves_communication_empty_with_a_note(self, monkeypatch):
        # 上游 API 漂移（消息对象没有 SerializeToString）时：留空 + 注明，
        # 不以 0 或推测值填充——「没测到」不能读成「测得 0」。
        stub = install_stub_psi(monkeypatch)
        monkeypatch.setattr(
            stub.client,
            "CreateRequest",
            lambda data: types.SimpleNamespace(items=[str(item) for item in data]),
        )
        run = run_psi_cardinality(LEFT, RIGHT, reference_fn=_plain_intersection)
        assert run.status == "ok", run.error
        assert run.communication is None
        assert any("载荷" in note for note in run.notes)

    def test_inputs_are_sorted_and_deduplicated_before_protocol(self, monkeypatch):
        stub = install_stub_psi(monkeypatch)
        run = run_psi_cardinality((103, 101, 101, 102), RIGHT, reference_fn=_plain_intersection)
        sent = next(call for call in stub.calls if call[0] == "client.CreateRequest")
        assert sent[1] == ("101", "102", "103")
        assert run.status == "ok"
        assert run.agreement is True
        assert any("去重" in note for note in run.notes)

    def test_empty_input_never_starts_the_protocol(self, monkeypatch):
        stub = install_stub_psi(monkeypatch)
        run = run_psi_cardinality((), RIGHT)
        assert run.status == "empty-input"
        assert run.value == 0
        assert stub.calls == []

    def test_unsupported_operator_is_rejected(self, monkeypatch):
        stub = install_stub_psi(monkeypatch)
        run = run_psi_cardinality(LEFT, RIGHT, op="Intersects")
        assert run.status == "error"
        assert "CellSetIntersect" in run.error
        assert stub.calls == []

    def test_non_count_policies_are_rejected(self, monkeypatch):
        stub = install_stub_psi(monkeypatch)
        for policy in ("REVEAL_INTERSECTION", "REVEAL_BOOLEAN"):
            run = run_psi_cardinality(LEFT, RIGHT, result_policy=policy)
            assert run.status == "error", policy
            assert REVEAL_COUNT in run.error
        assert stub.calls == []

    def test_layout_mismatch_is_rejected_before_the_protocol(self, monkeypatch):
        stub = install_stub_psi(monkeypatch)
        run = run_psi_cardinality(
            LEFT,
            RIGHT,
            left_layout={"layout_id": "layout-A", "version": 1},
            right_layout={"layout_id": "layout-B", "version": 1},
        )
        assert run.status == "error"
        assert "LAYOUT_MISMATCH" in run.error
        assert stub.calls == []

    def test_unavailable_environment_leaves_count_empty(self, monkeypatch):
        monkeypatch.setattr(ca_capability, "import_openmined_psi", _raise_import_error)
        report = check_psi_ca_capabilities()
        run = run_psi_cardinality(LEFT, RIGHT, reference_fn=_plain_intersection, report=report)
        assert run.status == "unavailable"
        assert run.value is None
        assert run.blockers
        assert any("保持空缺" in note for note in run.notes)

    def test_count_mismatch_is_an_error_not_a_pass(self, monkeypatch):
        install_stub_psi(monkeypatch)
        run = run_psi_cardinality(LEFT, RIGHT, reference_fn=lambda left, right: (102,))
        assert run.agreement is False
        assert run.status == "error"
        assert "不一致" in run.error


class TestLeakRegistry:
    def test_leak_code_differs_from_libpsi(self):
        # 两条 PSI 路径的协议泄漏面不同，登记码必须不同——同名会让
        # "PSI 只出计数"被误读成 libpsi 策略开关的另一种写法。
        assert PSI_CA_PROTOCOL_LEAK != PROTOCOL_LEAK_INTERSECTION_BODY

    def test_registry_covers_exactly_the_supported_ops(self):
        assert tuple(PSI_CA_OP_LEAKS) == PSI_CA_SUPPORTED_OPS
        assert set(PSI_CA_SUPPORTED_POLICIES) == {REVEAL_COUNT}
        assert PSI_CA_STRUCTURE == "RAW"
        assert PSI_CA_LEARNING_RANK == 0

    def test_unknown_op_has_no_registered_leak(self):
        assert "WeightedSum" not in PSI_CA_OP_LEAKS
