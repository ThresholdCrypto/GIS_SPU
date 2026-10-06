"""SPU 后端测试。

分两类：
    A. 不依赖 SPU 安装的测试（能力探测逻辑、门控行为、协议/环宽规范化）
       —— 任何环境都必须通过；
    B. 依赖真实 SPU 的测试（模拟执行、容差）
       —— 仅在受支持环境运行，否则 skip 并给出明确原因（不做假通过）。
"""

from __future__ import annotations

import numpy as np
import pytest

from backends.jax_backend import (
    generate_distance_le,
    generate_temporal_overlap,
    generate_weighted_sum,
    load_generated_function,
)
from backends.spu_backend import (
    OP_HLO_PRIMITIVES,
    PROTOCOL_MIN_WORLD_SIZE,
    SPU_FIELDS,
    SPU_PROTOCOLS,
    SPU_ADAPTED_HLO_PRIMITIVES,
    check_capabilities,
    check_operation_capability,
    normalize_field,
    normalize_protocol,
    platform_can_run_spu,
    protocol_min_world_size,
    run_spu_simulation,
)


# --------------------------------------------------------------------------
# A 类：不依赖 SPU 安装
# --------------------------------------------------------------------------


class TestProtocolRegistry:
    def test_protocol_list_matches_spu_0_9_5(self):
        """SPU 0.9.5 的 ProtocolKind 成员（无 SPDZ2K）。"""

        assert set(SPU_PROTOCOLS) == {"REF2K", "SEMI2K", "ABY3", "CHEETAH", "SECURENN"}
        assert "SPDZ2K" not in SPU_PROTOCOLS

    def test_field_list(self):
        assert set(SPU_FIELDS) == {"FM32", "FM64", "FM128"}

    def test_world_size_requirements(self):
        assert protocol_min_world_size("ABY3") == 3
        assert protocol_min_world_size("SECURENN") == 3
        assert protocol_min_world_size("SEMI2K") == 2
        assert protocol_min_world_size("CHEETAH") == 2

    def test_normalize_protocol_accepts_case(self):
        assert normalize_protocol("aby3") == "ABY3"
        assert normalize_protocol(" semi2k ") == "SEMI2K"

    def test_normalize_protocol_rejects_unknown(self):
        with pytest.raises(ValueError, match="SPDZ2K|未知协议"):
            normalize_protocol("SPDZ2K")

    def test_normalize_field_accepts_short_forms(self):
        assert normalize_field("64") == "FM64"
        assert normalize_field("fm128") == "FM128"
        assert normalize_field(32) == "FM32"

    def test_normalize_field_rejects_unknown(self):
        with pytest.raises(ValueError):
            normalize_field("FM256")


class TestCapabilityProbe:
    def test_report_is_jsonable(self):
        report = check_capabilities()
        data = report.to_dict()
        import json

        json.dumps(data)  # 必须可序列化
        assert "blockers" in data
        assert "jax" in data and "spu" in data

    def test_status_is_one_of_expected(self):
        report = check_capabilities()
        assert report.status in ("available", "installed-unrunnable", "unavailable")

    def test_runnable_implies_no_blockers(self):
        report = check_capabilities()
        assert report.runnable == (len(report.blockers) == 0)

    def test_jax_probe_detects_this_environment_correctly(self):
        report = check_capabilities()
        probe = report.jax
        assert probe.installed is True, "本仓库的 JAX 测试环境应有 jax"
        # jit 必须可用
        assert probe.jit_ok

    def test_probe_does_not_raise_without_spu(self):
        report = check_capabilities()
        if not report.spu.installed:
            assert "spu" in (report.spu.error or "")

    def test_platform_can_run_spu_reports_reason(self):
        ok, note = platform_can_run_spu()
        assert isinstance(ok, bool)
        assert note

    def test_private_dep_check_handles_attribute_form(self):
        """回归：SPU 的依赖里有“祖先模块的属性”而非子模块。

        如 `from jax._src.lib import xla_extension_version`：它不是可 import 的子模块。
        早期实现直接 import_module，把它误判为缺失，导致
        真实可运行的环境被错误标记为 runnable=False。
        """

        from backends.spu_backend.capability import (
            SPU_JAX_PRIVATE_DEPS,
            _private_dep_available,
        )

        if not check_capabilities().jax.installed:
            pytest.skip("本环境无 jax，无法验证私有接口拆解")

        # 清单里每一项在当前 jax 下都应可解析
        for dep in SPU_JAX_PRIVATE_DEPS:
            assert _private_dep_available(dep), f"{dep} 应被判定为可用"

    def test_private_dep_check_rejects_missing_names(self):
        """阴性回归：真正缺失的接口必须被判为不可用。"""

        from backends.spu_backend.capability import _private_dep_available

        if not check_capabilities().jax.installed:
            pytest.skip("本环境无 jax")

        for bogus in (
            "jax._src.lib.no_such_attribute",
            "jax.no_such_module",
            "",
        ):
            assert not _private_dep_available(bogus), f"{bogus!r} 不应被判为可用"

    def test_libspu_present_agrees_with_libspu_loadable(self):
        """回归：不能用 shutil.which 判断共享库是否存在。

        `libspu.so` 是动态库，不在 PATH 上也不可执行，
        `shutil.which("libspu.so")` 永远返回 None，会让
        `libspu_present` 永为 false，与 `libspu_loadable` 自相矛盾。
        """

        from backends.spu_backend.capability import probe_platform

        detail = probe_platform()
        assert isinstance(detail["libspu_present"], bool)

        report = check_capabilities()
        if not report.spu.installed:
            return
        # 已装 spu 的环境：两者必须一致
        assert detail["libspu_present"] == report.spu.libspu_loadable, (
            "libspu_present 与 libspu_loadable 不一致："
            f"{detail['libspu_present']} vs {report.spu.libspu_loadable}"
        )

    def test_module_spec_available_handles_shared_library(self):
        """`find_spec` 能识别 `.so` 扩展模块，且不对不存在的名字报错。"""

        from backends.spu_backend.capability import _module_spec_available

        assert _module_spec_available("sys") is True
        assert _module_spec_available("no.such.module.anywhere") is False
        if check_capabilities().spu.installed:
            assert _module_spec_available("spu.libspu") is True


class TestOperationCapability:
    def test_all_required_ops_have_primitives_registered(self):
        for op in ("DistanceLE", "WeightedSum", "TemporalOverlap"):
            assert op in OP_HLO_PRIMITIVES

    def test_registered_primitives_are_whitelisted(self):
        whitelist = set(SPU_ADAPTED_HLO_PRIMITIVES)
        for op, primitives in OP_HLO_PRIMITIVES.items():
            assert set(primitives) <= whitelist, op

    def test_capability_is_jsonable(self):
        import json

        capability = check_operation_capability("DistanceLE")
        json.dumps(capability.to_dict())

    def test_division_removal_kills_the_expensive_primitive_warning(self):
        """P2-1 之后 WeightedSum 不再有 divide/remainder，也就没有除法告警。

        反向回归：一旦有人把运行时 `scale` 加回生成代码，这条会立刻失败。
        """

        capability = check_operation_capability("WeightedSum")
        assert not any(
            "除法" in w or "divide" in w or "余" in w for w in capability.warnings
        )
        from backends.spu_backend import OP_HLO_PRIMITIVES

        assert "divide" not in OP_HLO_PRIMITIVES["WeightedSum"]

    def test_64bit_grid_code_triggers_overflow_warning(self):
        """grid_code 是 64 位定长键，在 FM64 下有溢出风险，必须告警。"""

        capability = check_operation_capability("Intersects")
        assert any("溢出" in w or "FM128" in w for w in capability.warnings)


class TestRunSpuSimulationGating:
    """核心门控行为：能力不足时必须诚实报告，不得返回伪造数值。"""

    def test_unavailable_environment_returns_unavailable_not_fake_value(self):
        function = generate_distance_le("f")
        fn = load_generated_function(function.source, function.name)
        report = check_capabilities()

        run = run_spu_simulation(
            fn,
            [np.array([1, 2, 3]), np.array([1, 3, 3]), np.array(2)],
            protocol="ABY3",
            field=64,
            report=report,
        )

        if not report.runnable:
            assert run.status == "unavailable"
            assert run.outputs is None
            assert run.blockers
            assert not run.ok

    def test_rejects_invalid_protocol_before_anything_else(self):
        with pytest.raises(ValueError):
            run_spu_simulation(lambda x: x, [np.array([1])], protocol="NOPE", field=64)

    def test_rejects_invalid_field(self):
        with pytest.raises(ValueError):
            run_spu_simulation(lambda x: x, [np.array([1])], protocol="ABY3", field="FM256")

    def test_result_is_jsonable(self):
        import json

        function = generate_distance_le("f")
        fn = load_generated_function(function.source, function.name)
        run = run_spu_simulation(fn, [np.array([1, 2, 3]), np.array([1, 3, 3]), np.array(2)])
        json.dumps(run.to_dict())

    def test_result_describe_is_readable(self):
        function = generate_distance_le("f")
        fn = load_generated_function(function.source, function.name)
        run = run_spu_simulation(fn, [np.array([1, 2, 3]), np.array([1, 3, 3]), np.array(2)])
        text = run.describe()
        assert "SPU simulation" in text
        assert run.status.upper() in text

    def test_never_claims_ok_without_running(self):
        """没有真实运行时，绝不允许出现 status='ok'。"""

        function = generate_distance_le("f")
        fn = load_generated_function(function.source, function.name)
        report = check_capabilities()
        run = run_spu_simulation(
            fn, [np.array([1, 2, 3]), np.array([1, 3, 3]), np.array(2)], report=report
        )
        if not report.runnable:
            assert run.status != "ok"
        assert run.max_abs_error is None


# --------------------------------------------------------------------------
# B 类：需要真实 SPU
# --------------------------------------------------------------------------


requires_spu = pytest.mark.skipif(
    not check_capabilities().runnable,
    reason=(
        "当前环境无法运行 SPU（"
        + "; ".join(check_capabilities().blockers)
        + "）。B 类测试需要 Linux/WSL2 + Python 3.10/3.11 + jax 0.4.34 + spu。"
    ),
)


@requires_spu
class TestRealSpuSimulation:
    """在受支持环境上真实执行；否则整体 skip（并说明缺失项）。"""

    def test_distance_le_on_spu(self):
        from backends.plain import run_plain

        function = generate_distance_le("f")
        fn = load_generated_function(function.source, function.name)
        left, right, threshold = np.array([1, 2, 3]), np.array([1, 3, 3]), np.array(2)

        run = run_spu_simulation(
            fn,
            [left, right, threshold],
            protocol="ABY3",
            field=64,
            reference_fn=lambda a, b, t: run_plain(
                "DistanceLE", list(a), list(b), int(t)
            ).value,
            tolerance=0.0,
        )
        assert run.ok, run.describe()
        assert run.within_tolerance is True
        assert run.max_abs_error == 0.0

    def test_weighted_sum_on_spu(self):
        from backends.plain import run_plain

        function = generate_weighted_sum("f")
        fn = load_generated_function(function.source, function.name)
        values, weights = np.array([10, 20, 30]), np.array([1, 2, 1])

        run = run_spu_simulation(
            fn,
            [values, weights],
            protocol="ABY3",
            field=64,
            reference_fn=lambda v, w: run_plain("WeightedSum", list(v), list(w)).value,
            tolerance=0.0,
        )
        assert run.ok, run.describe()
        assert run.within_tolerance is True

    def test_weighted_sum_runs_on_narrow_ring_after_division_removal(self):
        """P2-1 的回归：去掉除法后 `WeightedSum × FM32` 不再崩。

        修复前该组合是 `ring=FM32 could not represent PT_I64`
        （见 docs/MPC_BENCHMARK_PROTOCOL.md §4.3）。
        """

        from backends.plain import run_plain

        function = generate_weighted_sum("f")
        fn = load_generated_function(function.source, function.name)
        values, weights = np.array([10, 20, 30]), np.array([1, 2, 1])

        run = run_spu_simulation(
            fn,
            [values, weights],
            protocol="ABY3",
            field=32,
            reference_fn=lambda v, w: run_plain("WeightedSum", list(v), list(w)).value,
            tolerance=0.0,
        )
        assert run.ok, run.describe()
        assert run.within_tolerance is True

    def test_spu_result_matches_plain_within_tolerance(self):
        """SPU 结果必须与明文一致（整数路径容差 0）。"""

        from backends.plain import run_plain

        function = generate_distance_le("f")
        fn = load_generated_function(function.source, function.name)
        left, right, threshold = np.array([0, 0]), np.array([3, 4]), np.array(5)
        expected = run_plain("DistanceLE", list(left), list(right), 5).value

        run = run_spu_simulation(
            fn, [left, right, threshold], protocol="ABY3", field=64, tolerance=0.0
        )
        assert run.ok, run.describe()
        assert bool(run.outputs) == expected


def _sweep_case(op_name: str):
    """按算子名构造 (jax_fn, inputs, plain_reference)，供协议 × 环宽扫描复用。"""

    from backends.plain import run_plain

    if op_name == "DistanceLE":
        function = generate_distance_le("f")
        fn = load_generated_function(function.source, function.name)
        args = [np.array([1, 2, 3]), np.array([1, 3, 3]), np.array(2)]
        reference = lambda a, b, t: run_plain(  # noqa: E731
            "DistanceLE", list(a), list(b), int(t)
        ).value
    elif op_name == "WeightedSum":
        function = generate_weighted_sum("f")
        fn = load_generated_function(function.source, function.name)
        args = [np.array([10, 20, 30]), np.array([1, 2, 1])]
        reference = lambda v, w: run_plain(  # noqa: E731
            "WeightedSum", list(v), list(w)
        ).value
    elif op_name == "TemporalOverlap":
        # 段式节点 (Toff, Lt) → 4 个数组；判据见 backends/plain.plain_temporal_overlap
        function = generate_temporal_overlap("f")
        fn = load_generated_function(function.source, function.name)
        left_nodes, right_nodes = ((0, 3),), ((4, 3),)
        args = [
            np.array([n[0] for n in left_nodes], np.int32),
            np.array([n[1] for n in left_nodes], np.int32),
            np.array([n[0] for n in right_nodes], np.int32),
            np.array([n[1] for n in right_nodes], np.int32),
        ]
        reference = lambda lt, llt, rt, rlt: run_plain(  # noqa: E731
            "TemporalOverlap",
            [(int(a), int(b)) for a, b in zip(lt, llt)],
            [(int(a), int(b)) for a, b in zip(rt, rlt)],
        ).value
    else:
        raise AssertionError(f"未登记的扫描算子 {op_name!r}")
    return fn, args, reference


@requires_spu
class TestProtocolFieldSweep:
    """协议 × 算子 × 环宽的参数化对拍（矩阵扫描）。

    这是一条"接入新协议"的固定入口：SPU 升级后新增协议、或本项目放宽某个
    协议的支持范围时，只在 `SPU_PROTOCOLS` / `SPU_FIELDS` 里加一项，
    这里就会自动多跑一例真实对拍——不必为每个协议手写一个用例函数，
    也不存在"登记了却没接线"的静默通过。

    基线（2026-10-04，WSL2 + SPU 0.9.5）：`SPU_PROTOCOLS` 全 5 个协议在
    `DistanceLE` / `WeightedSum` / `TemporalOverlap` 上与明文逐位一致
    （整数路径 `max_abs_error == 0`），`FM32` / `FM64` / `FM128` 三个环宽同样一致。
    """

    @pytest.mark.parametrize(
        "op_name", ("DistanceLE", "WeightedSum", "TemporalOverlap")
    )
    @pytest.mark.parametrize("protocol", SPU_PROTOCOLS)
    def test_each_protocol_matches_plain(self, op_name, protocol):
        fn, args, reference = _sweep_case(op_name)

        run = run_spu_simulation(
            fn, args, protocol=protocol, field=64, reference_fn=reference, tolerance=0.0
        )
        assert run.ok, f"{protocol} / {op_name}: {run.describe()}"
        assert run.within_tolerance is True, f"{protocol} / {op_name}"
        assert run.max_abs_error == 0.0, f"{protocol} / {op_name}"

    @pytest.mark.parametrize("field", SPU_FIELDS)
    def test_each_field_matches_plain(self, field):
        fn, args, reference = _sweep_case("DistanceLE")

        run = run_spu_simulation(
            fn, args, protocol="ABY3", field=field, reference_fn=reference, tolerance=0.0
        )
        assert run.ok, f"{field}: {run.describe()}"
        assert run.within_tolerance is True, field


@requires_spu
class TestTemporalOverlapSweepOnSpu:
    """TemporalOverlap 的 sweep 电路（排序归并 + 前缀扫描）真机对拍。

    这一节守的是 P3 的核心主张：**K 上限不再被 [N, M] 矩阵锁死**。
    逐对版在 K=256 会物化 65 536 个比较、K=4096 是 1 670 万元素（实测拖死进程），
    扫描版把代价降到 O((N+M)·log(N+M)) 并在真机上跑通。

    规模取 256 而不是更大，是为了让测试跑得起：排序在 MPC 里代价高，
    K=1024 单条实测约 1.8 s / 43.7 MB 通信量，不适合放进常规测试套件
    （完整规模曲线见 docs/mpc_comm_baseline.json）。
    """

    @staticmethod
    def _case(k):
        from backends.plain import run_plain

        function = generate_temporal_overlap("f", strategy="sweep")
        fn = load_generated_function(function.source, function.name)
        left_toff = np.array([(i * 8) % 16000 for i in range(k)], np.int32)
        left_lt = np.array([2 + (i % 3) for i in range(k)], np.int32)
        right_toff = np.array([(i * 8 + 4) % 16000 for i in range(k)], np.int32)
        right_lt = np.array([2 + (i % 3) for i in range(k)], np.int32)
        args = [left_toff, left_lt, right_toff, right_lt]
        reference = lambda lt, llt, rt, rlt: run_plain(  # noqa: E731
            "TemporalOverlap",
            [(int(a), int(b)) for a, b in zip(lt, llt)],
            [(int(a), int(b)) for a, b in zip(rt, rlt)],
        ).value
        return fn, args, reference

    def test_sweep_runs_at_k_256_and_matches_plain(self):
        """K=256：逐对版在这一规模上已经开始吃力，扫描版必须能跑且逐位一致。"""

        fn, args, reference = self._case(256)
        run = run_spu_simulation(
            fn, args, protocol="ABY3", field="FM64",
            reference_fn=reference, tolerance=0.0,
        )
        assert run.ok, run.describe()
        assert run.within_tolerance is True, run.describe()
        assert run.max_abs_error == 0.0

    def test_sweep_agrees_with_pairwise_on_the_same_inputs(self):
        """同一批输入上两套电路必须给出同一个布尔——这是等价性的真机证据。"""

        fn, args, reference = self._case(64)
        pairwise_fn = load_generated_function(
            generate_temporal_overlap("g", strategy="pairwise").source, "g"
        )
        sweep = run_spu_simulation(
            fn, args, protocol="ABY3", field="FM64",
            reference_fn=reference, tolerance=0.0,
        )
        pairwise = run_spu_simulation(
            pairwise_fn, args, protocol="ABY3", field="FM64",
            reference_fn=reference, tolerance=0.0,
        )
        assert sweep.ok and pairwise.ok, (sweep.describe(), pairwise.describe())
        assert bool(np.asarray(sweep.outputs)) == bool(np.asarray(pairwise.outputs))
