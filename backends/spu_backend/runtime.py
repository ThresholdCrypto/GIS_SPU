"""SPU 后端：JAX 函数在 SPU 模拟器上的执行（`run_spu_simulation`）。

调用路径严格依据 SPU 0.9.5 官方源码（`spu/utils/simulation.py`）：

    sim = spu.utils.simulation.Simulator.simple(wsize, ProtocolKind.X, FieldType.FMy)
    spu_fn = spu.utils.simulation.sim_jax(sim, jax_fn)
    result = spu_fn(*inputs)

内部再走 `spu.utils.frontend.compile(Kind.JAX, ...)`：
    注册 interpreter 后端 → jax.jit().trace().lower(('interpreter',))
    → compiler_ir('hlo') → spu_api.compile(...) → 各参与方线程执行 → io.reconstruct

**本实现不做任何 API 猜测**：
    - 能力不足时返回 status="unavailable" 与明确的 blockers，绝不返回伪造数值；
    - 只有 `spu.utils.simulation` 真实存在且可导入时才会实际执行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .profile import (
    capture_native_logs,
    enable_native_console_log,
    parse_comm_profile,
)
from ..result_policy import MPC_OP_REVEALS, resolve_result_policy
from .capability import (
    CapabilityReport,
    check_capabilities,
    normalize_field,
    normalize_protocol,
    platform_can_run_spu,
    protocol_min_world_size,
)

# --------------------------------------------------------------------------
# 结果结构
# --------------------------------------------------------------------------


@dataclass
class SpuRunResult:
    """一次 SPU 模拟执行的结果。"""

    status: str  # ok / unavailable / error
    protocol: str
    field: str
    world_size: int
    outputs: Any = None
    outputs_list: list[Any] = field(default_factory=list)
    reference: Any = None
    max_abs_error: float | None = None
    within_tolerance: bool | None = None
    tolerance: float | None = None
    pphlo_bytes: int | None = None
    #: 通信量（仅在 `capture_comm=True` 时填；单位：字节，模拟链路计数）
    comm_send_bytes: int | None = None
    comm_recv_bytes: int | None = None
    comm_total_bytes: int | None = None
    comm_send_actions: int | None = None
    comm_recv_actions: int | None = None
    #: 逐原语通信量 {op: {send_bytes, recv_bytes, executions, duration_s}}
    comm_by_primitive: Mapping[str, Any] = field(default_factory=dict)
    #: 本次是否真的开了 profiling（开了的墙钟不可与没开的直接比）
    profiled: bool = False
    blockers: tuple[str, ...] = ()
    error: str | None = None
    notes: tuple[str, ...] = ()
    skipped_steps: tuple[str, ...] = ()
    #: 结果策略（§15 / Phase 10）：业务层暴露什么 + 协议内部输出面什么
    #: （分开登记；只有显式传入 op 的编译期调用才会填）
    result_policy: Mapping[str, Any] | None = None
    #: MPC 输出面登记（`MPC_OP_REVEALS` 原文）；未登记算子为提示句
    reveals: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "protocol": self.protocol,
            "field": self.field,
            "world_size": self.world_size,
            "outputs": _jsonable(self.outputs),
            "reference": _jsonable(self.reference),
            "max_abs_error": self.max_abs_error,
            "within_tolerance": self.within_tolerance,
            "tolerance": self.tolerance,
            "pphlo_bytes": self.pphlo_bytes,
            "comm_send_bytes": self.comm_send_bytes,
            "comm_recv_bytes": self.comm_recv_bytes,
            "comm_total_bytes": self.comm_total_bytes,
            "comm_send_actions": self.comm_send_actions,
            "comm_recv_actions": self.comm_recv_actions,
            "comm_by_primitive": _jsonable(self.comm_by_primitive),
            "profiled": self.profiled,
            "blockers": list(self.blockers),
            "error": self.error,
            "notes": list(self.notes),
            "skipped_steps": list(self.skipped_steps),
            "result_policy": (
                dict(self.result_policy) if self.result_policy is not None else None
            ),
            "reveals": self.reveals,
        }

    def describe(self) -> str:
        if self.ok:
            lines = [
                f"SPU simulation: OK  (protocol={self.protocol}, field={self.field}, wsize={self.world_size})",
                f"  outputs     : {_jsonable(self.outputs)}",
            ]
            if self.reference is not None:
                lines.append(f"  reference   : {_jsonable(self.reference)}")
            if self.within_tolerance is not None:
                lines.append(
                    f"  tolerance   : {self.tolerance}  max_abs_error={self.max_abs_error}  "
                    f"within={self.within_tolerance}"
                )
            if self.pphlo_bytes is not None:
                lines.append(f"  pphlo bytes : {self.pphlo_bytes}")
            if self.comm_total_bytes is not None:
                lines.append(
                    f"  comm bytes  : send={self.comm_send_bytes} "
                    f"recv={self.comm_recv_bytes} total={self.comm_total_bytes}"
                )
            if self.result_policy is not None:
                lines.append(
                    f"  policy      : {self.result_policy.get('policy')}"
                    f"（业务层暴露 {self.result_policy.get('business_value')}）"
                )
            if self.reveals:
                lines.append(f"  reveals     : {self.reveals}")
            for note in self.notes:
                lines.append(f"  note        : {note}")
            return "\n".join(lines)

        lines = [f"SPU simulation: {self.status.upper()}  (protocol={self.protocol}, field={self.field})"]
        if self.error:
            lines.append(f"  error       : {self.error}")
        if self.result_policy is not None:
            lines.append(
                f"  policy      : {self.result_policy.get('policy')}"
                f"（业务层暴露 {self.result_policy.get('business_value')}）"
            )
        if self.reveals:
            lines.append(f"  reveals     : {self.reveals}")
        for blocker in self.blockers:
            lines.append(f"  blocker     : {blocker}")
        for note in self.notes:
            lines.append(f"  note        : {note}")
        return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------


def run_spu_simulation(
    jax_fn: Callable[..., Any],
    inputs: Sequence[Any],
    protocol: str = "ABY3",
    field: str | int = 64,
    *,
    world_size: int | None = None,
    reference_fn: Callable[..., Any] | None = None,
    tolerance: float | None = None,
    report: CapabilityReport | None = None,
    copts: Any = None,
    static_argnums: Sequence[int] = (),
    op: str | None = None,
    result_policy: str | None = None,
    capture_comm: bool = False,
) -> SpuRunResult:
    """在 SPU 模拟器上执行一个 jax 函数。

    Args:
        jax_fn:   待执行的函数，必须用 jax.numpy 编写且可被 jax.jit 追踪。
        inputs:   输入张量序列（各参与方共享的秘密输入）。
        protocol: SPU 协议名，取值见 SPU_PROTOCOLS（REF2K/SEMI2K/ABY3/CHEETAH/SECURENN）。
        field:    环宽，可为 32/64/128 或 FM32/FM64/FM128。
        world_size: 参与方数量；缺省按协议要求下限（ABY3→3，SEMI2K→2）。
        reference_fn: 明文参考实现；给出后会计算误差并做容差判定。
        tolerance: 容差；缺省 0.0（整数路径应为精确）。
        report:   已探测的能力报告，避免重复探测。
        op: 算子名；给出后按统一结果策略表（`backends.result_policy`）解析
            该算子的结果策略并登记到结果里（编译期调用必须给出；
            未登记算子在解析期 fail-fast）。None = 不登记（探针/基准路径）。
        result_policy: 显式策略名；None = 该算子的默认策略。
            "部署期模式"（REVEAL_TO_REGULATOR）显式拒绝。
        capture_comm: 打开 SPU 的 pphlo profiling 以采集**通信量**
                   （`comm_send_bytes` / `comm_recv_bytes` / 逐原语明细）。
                   注意三点：(a) profiling 有开销，**开了的墙钟不能与没开的比**；
                   (b) 通信量并非每条都确定：多数组合五次一致，但
                   `TemporalOverlap` 的 ABY3、CHEETAH 的线性算子会抖
                   （实测 0.03%–10.1%），报数要给中位/区间；
                   (c) 原生日志是**进程级**开关，本函数会先把它打开
                   （PSI 路径会关掉它，关过就再也取不到通信量）。
                   实现依据见 `backends/spu_backend/profile.py` 的模块说明。

    Returns:
        SpuRunResult。能力不足时 status="unavailable"，绝不返回伪造数值。
    """

    # 结果策略在**任何执行之前**解析（含能力门控）：策略是编译期属性，
    # 不随环境是否可跑变化；非法策略 / 部署期模式在这里 fail-fast。
    policy = None
    if op is not None:
        policy = resolve_result_policy(op, result_policy)

    protocol_name = normalize_protocol(protocol)
    field_name = normalize_field(field)
    size = world_size if world_size is not None else protocol_min_world_size(protocol_name)

    report = report or check_capabilities()

    result = SpuRunResult(
        status="unavailable",
        protocol=protocol_name,
        field=field_name,
        world_size=size,
    )
    if policy is not None:
        result.result_policy = policy.to_dict()
        result.reveals = MPC_OP_REVEALS.get(op, "（未登记输出面，请补充）")

    # ---------------- 步骤 0：平台与能力门控 ----------------
    platform_ok, platform_note = platform_can_run_spu()
    if not platform_ok:
        result.notes = (platform_note,)

    if not report.runnable:
        result.blockers = report.blockers
        result.notes = result.notes + (
            "当前环境无法真实执行 SPU 模拟；明文与 JAX 结果仍然可用，"
            "SPU 结果栏位保持空缺而不以推测值填充。",
        )
        return result

    if size < protocol_min_world_size(protocol_name):
        result.status = "error"
        result.error = (
            f"协议 {protocol_name} 至少需要 {protocol_min_world_size(protocol_name)} 个参与方，"
            f"当前 world_size={size}"
        )
        return result

    # ---------------- 步骤 1：导入官方 API（失败即诚实报告） ----------------
    try:
        import numpy as np

        import spu.libspu as libspu
        from spu.utils import simulation as spu_simulation
    except Exception as exc:
        result.status = "error"
        result.error = f"导入 SPU 运行期 API 失败：{type(exc).__name__}: {exc}"
        return result

    if not hasattr(spu_simulation, "Simulator") or not hasattr(spu_simulation, "sim_jax"):
        result.status = "error"
        result.error = (
            "spu.utils.simulation 中缺少 Simulator / sim_jax；"
            "当前 SPU 版本与预期 API 不一致，请核对 docs/SPU_CAPABILITY.md"
        )
        return result

    # ---------------- 步骤 2：构造模拟器 ----------------
    try:
        protocol_kind = getattr(libspu.ProtocolKind, protocol_name)
        field_type = getattr(libspu.FieldType, field_name)
        if capture_comm:
            # 通信量只在 pphlo profiling 打开时才会被 SPU 统计。
            # 这里故意不复用 Simulator.simple：它内部构造的 RuntimeConfig
            # 不给外部改 profile 开关的机会。
            config = libspu.RuntimeConfig(protocol_kind, field_type)
            config.enable_pphlo_profile = True
            sim = spu_simulation.Simulator(size, config)
        else:
            sim = spu_simulation.Simulator.simple(size, protocol_kind, field_type)
    except Exception as exc:
        result.status = "error"
        result.error = (
            f"构造 SPU Simulator 失败：{type(exc).__name__}: {exc}；"
            f"请核对协议 {protocol_name} / 环宽 {field_name} 是否为该版本支持"
        )
        return result

    # ---------------- 步骤 3：编译并执行 ----------------
    try:
        kwargs: dict[str, Any] = {"static_argnums": tuple(static_argnums)}
        if copts is not None:
            kwargs["copts"] = copts
        spu_fn = spu_simulation.sim_jax(sim, jax_fn, **kwargs)
        flat = [np.asarray(x) for x in inputs]
        if capture_comm:
            # 原生日志是进程级开关，PSI 路径会把它关掉；不先打开就必然取不到数。
            console_on = enable_native_console_log()
            # profile 行走 C 层 spdlog，只有 fd 级重定向拿得到。
            # 解析必须在 with 块内做：离开块时临时 fd 就被关掉了。
            with capture_native_logs() as logs:
                outputs = spu_fn(*flat)
                measured = parse_comm_profile(logs.read())
            result.profiled = True
            for key in (
                "comm_send_bytes", "comm_recv_bytes", "comm_total_bytes",
                "comm_send_actions", "comm_recv_actions",
            ):
                setattr(result, key, measured.get(key))
            result.comm_by_primitive = measured.get("comm_by_primitive", {})
            if result.comm_total_bytes is None:
                reason = (
                    "SPU 原生 console logger 打不开（C++ 侧接口不可用）"
                    if not console_on
                    else "已开 profiling 但日志里没有通信量行"
                )
                result.notes = result.notes + (
                    f"未能取到通信量（{reason}）——该栏位留空，不以推测值填充"
                    "（见 docs/MPC_BENCHMARK_PROTOCOL.md §8.4）",
                )
        else:
            outputs = spu_fn(*flat)
    except Exception as exc:
        result.status = "error"
        result.error = f"SPU 模拟执行失败：{type(exc).__name__}: {exc}"
        return result

    result.status = "ok"
    result.outputs = outputs
    result.outputs_list = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]

    pphlo = getattr(spu_fn, "pphlo", None)
    if isinstance(pphlo, str):
        result.pphlo_bytes = len(pphlo)

    execution_path = (
        f"spu.utils.simulation.Simulator.simple({size}, {protocol_name}, {field_name})"
        if not capture_comm
        else (
            f"spu.utils.simulation.Simulator({size}, RuntimeConfig("
            f"{protocol_name}, {field_name}, enable_pphlo_profile=True))"
        )
    )
    result.notes = result.notes + (
        f"执行路径：{execution_path} → sim_jax → frontend.compile(Kind.JAX)",
    )

    # ---------------- 步骤 4：误差与容差 ----------------
    if reference_fn is not None:
        reference = reference_fn(*inputs)
        result.reference = reference
        tol = 0.0 if tolerance is None else float(tolerance)
        result.tolerance = tol
        result.max_abs_error = _max_abs_error(outputs, reference)
        result.within_tolerance = bool(result.max_abs_error <= tol)
        if not result.within_tolerance:
            result.status = "error"
            result.error = (
                f"结果超出容差：max_abs_error={result.max_abs_error} > tolerance={tol}"
            )

    return result


def _max_abs_error(outputs: Any, reference: Any) -> float:
    """计算最大绝对误差；布尔/整数路径下应为 0。"""

    import numpy as np

    out = np.asarray(outputs)
    ref = np.asarray(reference)
    if out.shape != ref.shape:
        return float("inf")
    if out.dtype == bool or ref.dtype == bool:
        return 0.0 if bool(np.all(out == ref)) else float("inf")
    diff = np.abs(out.astype(np.float64) - ref.astype(np.float64))
    return float(np.max(diff)) if diff.size else 0.0


# --------------------------------------------------------------------------
# 算子级便捷入口（明文 / JAX / SPU 三方对照）
# --------------------------------------------------------------------------


@dataclass
class TripleResult:
    """一个算子的三份实现对照结果。"""

    op: str
    plain: Any = None
    jax: Any = None
    spu_run: SpuRunResult | None = None
    plain_error: str | None = None
    jax_error: str | None = None
    agreement: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "plain": _jsonable(self.plain),
            "jax": _jsonable(self.jax),
            "spu": self.spu_run.to_dict() if self.spu_run else None,
            "plain_error": self.plain_error,
            "jax_error": self.jax_error,
            "agreement": dict(self.agreement),
        }
