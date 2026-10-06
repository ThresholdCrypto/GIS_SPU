# -*- coding: utf-8 -*-
"""MPC 性能基线：SPU 协议 × 算子 × 环宽的实测代价，与 planner 预测代价对账。

依据：任务文档 §8（接入协议）/ §9（代价体系）/ §23；README §8.3
"补不同协议的代价实测（semi2k / aby3 / cheetah），把 planner 的预测代价与
平台实测代价对账——这正是课题里'接入层预测、平台回填实测'的落点"。

职责边界
--------
- 只测量"已生成的 JAX 函数 → SPU 模拟执行"的代价。**不改** JAX 生成器、
  **不改** planner 规则、**不改** SPU 源码。
- planner 预测的是**结构代价**（`N_ct` / `b` / `d` / `R`），**不是墙钟时间**。
  因此本基线记录两侧，但只对**可对账的那一项**下结论：
    * 位宽 `b(K)` ↔ 平台环宽 `FIELD_BITS[field]` 是否覆盖（溢出预测对不对）；
  其余测量（墙钟 / 峰值内存 / PPHLO 字节数）是**平台侧回填**，供后续回归参考；
  **不**给出"预测时间 ↔ 实测时间"的误差——planner 不预测时间，硬凑口径等于伪造。
- 没有真实执行过的组合**不填数字**：`status="unavailable"` + 原因；
  执行失败如实记 `status="error"` + 错误原文；`agreement` 只来自真实明文对拍。
- 输入是**确定性构造**（无随机数）：同一参数在任何机器上给出同一批输入。

字段口径
--------
沿用 PSI 基线（`docs/BENCHMARK_PROTOCOL.md` §3）的名字与诚实规则：
`status` / `note` / `error` / `wall_ms` / `peak_rss_mb` / `memory_mb` / `agreement`。
MPC 侧新增：`predicted_*`（planner 预测，含位宽对账三件套）与 `pphlo_bytes`
（平台侧回填的真实编译产物规模）。

通信量（P2-2 起）：`comm_send_bytes` / `comm_recv_bytes` / `comm_total_bytes` /
`comm_send_actions` / `comm_recv_actions` / `comm_by_primitive` / `profiled`。
它们只在 `capture_comm=True` 的用例上出现（SPU 的 pphlo profiling 开关），
**空就是空**：没开 profiling 或解析不出，一律留 `None`，不以推测值填充。
两条硬约束（见 `docs/MPC_BENCHMARK_PROTOCOL.md` §8.4）：
  1. profiling 有开销——**开了 profiling 的墙钟不能与没开的比**；
  2. 通信量有 ~1–2% 批间抖动，**不是解析确定量**，报数用重复实验的中位/区间。
"""

from __future__ import annotations

import csv
import json
import os
import re
import resource
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from backends.jax_backend import (
    generate_distance_le,
    generate_temporal_overlap,
    generate_weighted_sum,
    load_generated_function,
)
from backends.plain import run_plain

from .capability import FIELD_BITS, SPU_FIELDS, SPU_PROTOCOLS
from .runtime import run_spu_simulation

# --------------------------------------------------------------------------
# 默认扫描策略
# --------------------------------------------------------------------------

#: 参与"协议 × 算子"主扫描的 MPC 算子
MPC_BENCHMARK_OPS: tuple[str, ...] = ("DistanceLE", "WeightedSum", "TemporalOverlap")

#: 主扫描规模：**按算子给**，不强行统一量纲。
#: `TemporalOverlap` 是 N×M 两两比较电路（见 generate_temporal_overlap 的
#: `[:, None] < [None, :]`），代价对节点数**二次**：K=256 在 CHEETAH 上单条
#: 超过 90 s 不返回，K=4096 会展开 1670 万元素直接拖死进程。
_MAIN_SCAN_K: Mapping[str, int] = {
    "DistanceLE": 256,
    "WeightedSum": 256,
    "TemporalOverlap": 32,
}

#: 规模扫描：**按算子各取各的量纲**，不强行统一。
#: `TemporalOverlap` 的电路是 N×M 两两比较（见 generate_temporal_overlap 的
#: `[:, None] < [None, :]`），对节点数是**二次**的——K=4096 会展开 1670 万元素，
#: 实测直接把进程拖死。故它的规模上限单独压低。
_SIZE_SWEEP: Mapping[str, tuple[int, ...]] = {
    "DistanceLE": (64, 256, 1024, 4096),
    "WeightedSum": (64, 256, 1024, 4096),
    "TemporalOverlap": (8, 16, 32),
}

#: 位宽对账用的额外 K（超出实测预算，只做预测不实测）
_BIT_WIDTH_PROBE_K: tuple[int, ...] = (1, 8, 1 << 20)

#: 【历史，P2-1 已闭合：除法已从生成代码里移除】曾观察到两条偏差，
#: 现都不复现，故**不再登记**为预期失败——一旦复现，`unexpected_records`
#: 会直接报出来。留档在 `docs/MPC_BENCHMARK_PROTOCOL.md` §4.2 / §4.3：
#:   1. `WeightedSum` 生成代码的 `acc // scale` 在 SPU 上不精确且非确定
#:      （`div_goldschmidt`）：K=1024 偏差率 43%、K=4096 为 93%，K≤256 为 0；
#:   2. `WeightedSum × FM32` 起不来（除法路径内部要 64 位环：
#:      `ring=FM32 could not represent PT_I64`），而 planner 的位宽预测
#:      b(K=256)=24 ≤ 32 判"够用"——"预测 ≠ 实测"的实例。

#: 非确定偏差行的预期状态标记：`error`（超出容差）与 `ok`（本轮恰好对上）
#: **都算符合预期**，只有第三种状态（如 `unavailable`）才算预期外。
#: 用它而不是 `expect_status="error"`，是因为硬钉一种结果会把基线做成
#: 抽奖：同一份代码两次运行给出不同的退出码（本基线实测到过）。
EXPECT_DEVIATION = "deviation"

#: 重复执行的汇总状态：多次运行里既有成功又有失败。它是**发现**而不是噪声——
#: 一个登记为"应当稳定"的用例出现 mixed，说明这段路径本身不稳定。
MIXED_STATUS = "mixed"

@dataclass(frozen=True)
class MpcBenchmarkCase:
    """一条 MPC 基准用例。"""

    op: str
    protocol: str
    field: str = "FM64"
    k: int = 1024
    execute: bool = True
    unavailable_note: str = ""
    expect_status: str = ""
    #: 重复次数（>1 时记录变成"多次运行的统计"，见 summarize_repeats）
    repeats: int = 1
    #: 是否打开 SPU 的 pphlo profiling 采集通信量（开了的墙钟不可与没开的比）
    capture_comm: bool = False

    def label(self) -> str:
        return f"{self.op} {self.protocol} {self.field} K={self.k}"


def standard_cases(
    *, quick: bool = False, repeats: int = 1, capture_comm: bool = False
) -> list[MpcBenchmarkCase]:
    """标准扫描用例集（见模块 docstring 的默认扫描策略）。

    `repeats > 1` 时每条用例跑多次，记录变成统计口径（`summarize_repeats`）——
    这是"拿方差再谈协议排序"的落地方式，见 `docs/MPC_BENCHMARK_PROTOCOL.md` §8。
    `capture_comm=True` 时每条用例打开 SPU profiling 采集通信量；注意此时
    **墙钟含 profiling 开销**，只能和同样开了 profiling 的用例比。
    """

    cases: list[MpcBenchmarkCase] = []

    def add(op: str, protocol: str, field: str, k: int) -> None:
        cases.append(
            MpcBenchmarkCase(
                op=op,
                protocol=protocol,
                field=field,
                k=k,
                expect_status="",
                repeats=max(1, int(repeats)),
                capture_comm=capture_comm,
            )
        )

    # 1) 协议 × 算子：主扫描（这本就是"接协议要跑哪些组合"的落地口径）
    for op in MPC_BENCHMARK_OPS:
        for protocol in SPU_PROTOCOLS:
            add(op, protocol, "FM64", _MAIN_SCAN_K[op])

    # 2) 环宽矩阵：协议 × 算子固定 ABY3，走三个字段
    if not quick:
        for op in MPC_BENCHMARK_OPS:
            for field in SPU_FIELDS:
                add(op, "ABY3", field, _MAIN_SCAN_K[op])

    # 3) 规模扫描：ABY3 / FM64
    for op, sizes in _SIZE_SWEEP.items():
        for k in sizes[:1] if quick else sizes:
            add(op, "ABY3", "FM64", k)

    # 4) 位宽对账：预测 b(K) 超过环宽的组合**不实测**（避免把模回绕当有效数据）
    for k in _BIT_WIDTH_PROBE_K:
        predicted = predicted_bit_width("WeightedSum", k)
        if predicted is None or predicted <= FIELD_BITS["FM32"]:
            continue
        cases.append(
            MpcBenchmarkCase(
                op="WeightedSum",
                protocol="ABY3",
                field="FM32",
                k=k,
                execute=False,
                repeats=max(1, int(repeats)),
                capture_comm=capture_comm,
                unavailable_note=(
                    f"未实测：planner 预测 b(K={k})={predicted} > FM32={FIELD_BITS['FM32']}，"
                    "属越界组合；模 2^32 回绕的结果不能当有效数据（不伪造数字）"
                ),
            )
        )
    return cases


# --------------------------------------------------------------------------
# planner 侧：预测代价与位宽
# --------------------------------------------------------------------------


def predicted_cost(op: str, k: int | None = None) -> dict[str, Any]:
    """planner 对该算子的结构化预测代价（四量：N_ct / b / d / R）。"""

    from planner.registry import get_rule, resolve_cost

    return resolve_cost(get_rule(op), k=k)


def predicted_bit_width(op: str, k: int) -> int | None:
    """planner 预测的位宽 b(K)；解析不出整数时返回 None（不猜）。"""

    from planner.registry import get_rule

    rule = get_rule(op)
    if rule.bit_width_formula is not None:
        return int(rule.bit_width_formula(k))
    match = re.match(r"\s*(\d+)", str(predicted_cost(op, k).get("b", "")))
    return int(match.group(1)) if match else None


def reconcile_bit_width(op: str, k: int, field: str) -> dict[str, Any]:
    """位宽对账：planner 预测 b(K) 是否被平台环宽覆盖。

    `covered=False` 表示**预测层面就会溢出**——这类组合不该拿去实测，
    也不该被当作"跑通了"。`predicted_bits=None` 表示预测不到具体位数。
    """

    predicted = predicted_bit_width(op, k)
    bits = FIELD_BITS[field]
    return {
        "predicted_bits": predicted,
        "field_bits": bits,
        "covered": None if predicted is None else predicted <= bits,
    }


# --------------------------------------------------------------------------
# 确定性输入构造
# --------------------------------------------------------------------------


def _distance_le_inputs(k: int) -> tuple[list[np.ndarray], Callable[..., Any]]:
    """坐标对 + 阈值。取值域窄（<32），阈值远离溢出边界。"""

    left = np.array([i % 32 for i in range(k)], np.int32)
    right = np.array([(i * 3) % 32 for i in range(k)], np.int32)
    threshold = np.array(64, np.int32)
    reference = lambda a, b, t: run_plain(  # noqa: E731
        "DistanceLE", list(a), list(b), int(t)
    ).value
    return [left, right, threshold], reference


def _weighted_sum_inputs(k: int) -> tuple[list[np.ndarray], Callable[..., Any]]:
    """属性值 + 权重。累加上界远小于 FM32。

    定点 scale 自 P2-1 起是**编译期常量**（生成代码里 `scale=1` → 无除法），
    所以这里不再构造第三个输入。
    """

    values = np.array([(i % 100) + 1 for i in range(k)], np.int32)
    weights = np.array([(i % 7) + 1 for i in range(k)], np.int32)
    reference = lambda v, w: run_plain(  # noqa: E731
        "WeightedSum", list(v), list(w)
    ).value
    return [values, weights], reference


def _temporal_overlap_inputs(k: int) -> tuple[list[np.ndarray], Callable[..., Any]]:
    """段式节点 (Toff, Lt)。Toff + 2^Lt 必须留在 14 位域内。"""

    left_lt = np.array([2 + (i % 3) for i in range(k)], np.int32)
    right_lt = np.array([2 + (i % 3) for i in range(k)], np.int32)
    left_toff = np.array([(i * 8) % 16000 for i in range(k)], np.int32)
    right_toff = np.array([(i * 8 + 4) % 16000 for i in range(k)], np.int32)
    reference = lambda lt, llt, rt, rlt: run_plain(  # noqa: E731
        "TemporalOverlap",
        [(int(a), int(b)) for a, b in zip(lt, llt)],
        [(int(a), int(b)) for a, b in zip(rt, rlt)],
    ).value
    return [left_toff, left_lt, right_toff, right_lt], reference


_INPUT_BUILDERS: Mapping[str, Callable[[int], tuple[list[np.ndarray], Callable[..., Any]]]] = {
    "DistanceLE": _distance_le_inputs,
    "WeightedSum": _weighted_sum_inputs,
    "TemporalOverlap": _temporal_overlap_inputs,
}

_GENERATORS: Mapping[str, Callable[[str], Any]] = {
    "DistanceLE": generate_distance_le,
    "WeightedSum": generate_weighted_sum,
    "TemporalOverlap": generate_temporal_overlap,
}


# --------------------------------------------------------------------------
# 测量
# --------------------------------------------------------------------------


def _peak_rss_mb() -> float:
    """进程峰值 RSS（MB）。

    Linux: ru_maxrss 单位是 KB；macOS: 字节，需要换算。基线在 WSL/Linux 上跑；
    其它平台原样返回（相对增量仍可用，绝对值别引用）。
    """

    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        raw /= 1024.0
    return raw / 1024.0


def _blank_record(case: MpcBenchmarkCase) -> dict[str, Any]:
    # 未登记的算子也要能出"错误记录"而不是抛异常：预测侧降级为 None。
    try:
        reconciliation = reconcile_bit_width(case.op, case.k, case.field)
        cost: dict[str, Any] = predicted_cost(case.op, case.k)
    except Exception:
        reconciliation = {
            "predicted_bits": None,
            "field_bits": FIELD_BITS[case.field],
            "covered": None,
        }
        cost = {}
    return {
        "case": case.label(),
        "op": case.op,
        "protocol": case.protocol,
        "field": case.field,
        "k": case.k,
        "expect_status": case.expect_status,
        "repeat": 1,
        "status_counts": {},
        "deviation_rate": None,
        "wall_ms_stats": {},
        "wall_ms_samples": [],
        "max_abs_error_max": None,
        # planner 侧（接入层预测）
        "predicted_cost": cost,
        "predicted_bits": reconciliation["predicted_bits"],
        "field_bits": reconciliation["field_bits"],
        "bit_width_covered": reconciliation["covered"],
        # 平台侧（实测回填）
        "status": "unavailable",
        "note": "",
        "error": "",
        "setup_ms": None,
        "wall_ms": None,
        "memory_mb": None,
        "peak_rss_mb": None,
        "pphlo_bytes": None,
        # 通信量（P2-2；仅在 capture_comm=True 时填，否则留空不伪造）
        "profiled": False,
        "comm_send_bytes": None,
        "comm_recv_bytes": None,
        "comm_total_bytes": None,
        "comm_send_actions": None,
        "comm_recv_actions": None,
        "comm_by_primitive": {},
        # 重复实验（repeats > 1）才填：通信量的 min/p25/median/p75/max
        "comm_total_bytes_stats": {},
        "agreement": None,
        "output_bits": None,
        "max_abs_error": None,
        "tolerance": None,
    }


def _wall_stats(samples: Sequence[float]) -> dict[str, float]:
    """墙钟样本的 min / p25 / median / p75 / max（线性插值）。

    只描述**这一台机器这一次会话**的抖动；不要拿它当跨机器结论。
    """

    values = sorted(float(v) for v in samples)
    if not values:
        return {}
    return {
        "min": values[0],
        "max": values[-1],
        "median": float(np.percentile(values, 50)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def _numbers(values: Sequence[Any]) -> list[float]:
    """只留真数值（`bool` 不算、`None` 不算）——缺测不能当 0。"""

    keep: list[float] = []
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            keep.append(float(value))
    return keep


def summarize_repeats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """把同一用例的多次运行压成一条**统计记录**。

    口径（与"不伪造数字"一致）：
    - `status`：多次一致就取该状态；**不一致记 `mixed`**，不取多数票；
    - `deviation_rate` = 与明文不一致的次数 / **真正得出数值**的次数
      （崩掉的那些次不计入分母，单独看 `status_counts`）；
    - `agreement`：全部为 ok 才是 `True`；其余一律 `None`，
      细节交给 `deviation_rate` / `max_abs_error_max`，不用一个布尔值含糊过去。
    """

    if not records:
        raise ValueError("summarize_repeats 至少需要一条记录")

    base = dict(records[0])
    base["repeat"] = len(records)

    counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status"))
        counts[status] = counts.get(status, 0) + 1
    base["status_counts"] = counts
    base["status"] = records[0]["status"] if len(counts) == 1 else MIXED_STATUS

    walls = _numbers([record.get("wall_ms") for record in records])
    base["wall_ms_samples"] = walls
    base["wall_ms_stats"] = _wall_stats(walls)
    base["wall_ms"] = base["wall_ms_stats"]["median"] if walls else None

    # 通信量：抖动是真实的（~1–2% 批间），所以按统计口径报，
    # 取中位并在 note 里给出区间；逐原语明细取"总通信量最接近中位那次"的快照。
    base["profiled"] = all(bool(record.get("profiled")) for record in records)
    fill: list[Mapping[str, Any]] = [
        record for record in records if record.get("comm_total_bytes") is not None
    ]
    totals = _numbers([record.get("comm_total_bytes") for record in records])
    if totals:
        median = float(np.percentile(totals, 50))
        nearest = min(fill, key=lambda rec: abs(float(rec["comm_total_bytes"]) - median))
        for key in (
            "comm_send_bytes",
            "comm_recv_bytes",
            "comm_send_actions",
            "comm_recv_actions",
        ):
            values = _numbers([record.get(key) for record in records])
            base[key] = float(np.percentile(values, 50)) if values else None
        base["comm_total_bytes"] = median
        base["comm_total_bytes_stats"] = _wall_stats(totals)
        base["comm_by_primitive"] = dict(nearest.get("comm_by_primitive") or {})
    else:
        # 一次都没采到就不填（不伪造）；字段自体仍在，值为 None。
        base["comm_send_bytes"] = None
        base["comm_recv_bytes"] = None
        base["comm_total_bytes"] = None
        base["comm_send_actions"] = None
        base["comm_recv_actions"] = None
        base["comm_by_primitive"] = {}
        base["comm_total_bytes_stats"] = {}

    errors = _numbers([record.get("max_abs_error") for record in records])
    base["value_runs"] = len(errors)
    base["max_abs_error_max"] = max(errors) if errors else None
    base["max_abs_error"] = base["max_abs_error_max"]
    base["deviation_rate"] = (
        sum(1 for value in errors if value > 0) / len(errors) if errors else None
    )

    if counts.get("ok", 0) == len(records):
        base["agreement"] = True
    else:
        base["agreement"] = None

    error_text = next(
        (str(record.get("error")) for record in records if record.get("error")), ""
    )
    base["error"] = error_text

    notes: list[str] = []
    first_note = str(records[0].get("note") or "")
    if first_note:
        notes.append(first_note)
    notes.append(f"重复 {len(records)} 次：{counts}")
    if base["status"] == MIXED_STATUS:
        notes.append("多次运行结果不一致（非确定路径），不能当稳定通过")
    if base["deviation_rate"]:
        notes.append(
            f"偏差率 {base['deviation_rate']:.0%}（{base['value_runs']} 次得出数值）"
        )
    if totals:
        stats = base["comm_total_bytes_stats"]
        if stats["min"] == stats["max"]:
            notes.append(
                "通信量 {:.0f} B（{} 次逐字节一致）".format(stats["median"], len(totals))
            )
        else:
            notes.append(
                "通信量中位 {:.0f} B（min {:.0f} / max {:.0f}；抖动是实测现象，"
                "非解析确定量）".format(stats["median"], stats["min"], stats["max"])
            )
    base["note"] = "；".join(notes)
    return base


def run_benchmark_case(
    case: MpcBenchmarkCase, *, report: Any = None
) -> dict[str, Any]:
    """执行一条用例并返回记录（失败也不抛：如实记进 `status` / `error`）。

    `case.repeats > 1` 时执行多次并返回**统计记录**（`summarize_repeats`），
    而不是最后一次的结果——重复实验的目的就是看"结果稳不稳"。
    """

    repeats = max(1, int(case.repeats))
    if repeats == 1:
        return _run_once(case, report=report)
    samples = [_run_once(case, report=report) for _ in range(repeats)]
    return summarize_repeats(samples)


def _run_once(case: MpcBenchmarkCase, *, report: Any = None) -> dict[str, Any]:
    """单次执行（内部用；对外请走 `run_benchmark_case`）。"""

    from .capability import check_capabilities

    record = _blank_record(case)

    if not case.execute:
        record["note"] = case.unavailable_note or "未实测（用例声明 execute=False）"
        return record

    builder = _INPUT_BUILDERS.get(case.op)
    generator = _GENERATORS.get(case.op)
    if builder is None or generator is None:
        record["status"] = "error"
        record["error"] = f"未登记的 MPC 基准算子 {case.op!r}；已知：{sorted(_GENERATORS)}"
        return record

    report = report or check_capabilities()
    if not report.runnable:
        record["note"] = (
            "当前环境无法真实执行 SPU 模拟；不给任何实测数字"
            "（见 docs/SPU_CAPABILITY.md）"
        )
        return record

    setup_start = time.perf_counter()
    args, reference = builder(case.k)
    generated = generator("f")
    fn = load_generated_function(generated.source, generated.name)
    record["setup_ms"] = (time.perf_counter() - setup_start) * 1000.0

    before_peak = _peak_rss_mb()
    start = time.perf_counter()
    run = run_spu_simulation(
        fn,
        args,
        protocol=case.protocol,
        field=case.field,
        reference_fn=reference,
        tolerance=0.0,
        report=report,
        capture_comm=case.capture_comm,
    )
    record["wall_ms"] = (time.perf_counter() - start) * 1000.0
    record["memory_mb"] = max(0.0, _peak_rss_mb() - before_peak)
    record["peak_rss_mb"] = _peak_rss_mb()

    # 通信量无论成败都如实回填：失败行也有"跑了多少通信才失败"的参考价值，
    # 但只在 profiling 真的开出统计时才有值（run_spu_simulation 不做推测）。
    record["profiled"] = bool(getattr(run, "profiled", False))
    for key in (
        "comm_send_bytes",
        "comm_recv_bytes",
        "comm_total_bytes",
        "comm_send_actions",
        "comm_recv_actions",
    ):
        record[key] = getattr(run, key, None)
    record["comm_by_primitive"] = dict(getattr(run, "comm_by_primitive", {}) or {})
    # 运行期说明（如"没能取到通信量"的原因）必须能落到基线里，
    # 否则空栏位看起来就像"这项没测"，而不是"这项测了但取不到"。
    run_notes = [str(note) for note in getattr(run, "notes", ()) or ()]
    if run_notes:
        record["note"] = "；".join(part for part in [record["note"], *run_notes] if part)

    if run.ok:
        record["status"] = "ok"
        record["agreement"] = run.within_tolerance
        record["pphlo_bytes"] = run.pphlo_bytes
        record["output_bits"] = _int_bits(run.outputs)
        record["max_abs_error"] = run.max_abs_error
        record["tolerance"] = run.tolerance
        if run.within_tolerance is not True:
            record["note"] = (
                f"实测与明文不一致：max_abs_error={run.max_abs_error}"
                f"（tolerance={run.tolerance}）"
            )
    else:
        record["status"] = run.status
        record["error"] = (run.error or "; ".join(run.blockers) or "").strip()
        record["note"] = (run.error or run.blockers and "环境不具备" or "").strip()
        # 失败行也留实测值：容差不过时，"偏了多少"本身就是结论
        record["max_abs_error"] = run.max_abs_error
        record["tolerance"] = run.tolerance
        record["output_bits"] = _int_bits(run.outputs)
        record["pphlo_bytes"] = run.pphlo_bytes
        if run.status == "unavailable":
            record["wall_ms"] = None
            record["memory_mb"] = None
    return record


def _int_bits(outputs: Any) -> int | None:
    """实测输出的位宽（该值所需的最小位），对账"预测 b(K)"的量级是否合理。"""

    try:
        value = int(np.asarray(outputs).reshape(-1)[0])
    except Exception:
        return None
    return int(value).bit_length()


def run_benchmark_cases(
    cases: Sequence[MpcBenchmarkCase],
    *,
    report: Any = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """逐条执行（串行；SPU 模拟不保证进程内并发安全）。

    `progress` 每完成一条回调一次——长扫描必须能看见进度，否则一条卡住
    会让人误以为整体没动（本基线曾因此被误判为"环境坏了"）。
    """

    records: list[dict[str, Any]] = []
    for case in cases:
        record = run_benchmark_case(case, report=report)
        records.append(record)
        if progress is not None:
            progress(record)
    return records


# --------------------------------------------------------------------------
# 产物与汇总
# --------------------------------------------------------------------------


_CSV_COLUMNS: tuple[str, ...] = (
    "case",
    "op",
    "protocol",
    "field",
    "k",
    "predicted_bits",
    "field_bits",
    "bit_width_covered",
    "status",
    "wall_ms",
    "memory_mb",
    "peak_rss_mb",
    "pphlo_bytes",
    "profiled",
    "comm_send_bytes",
    "comm_recv_bytes",
    "comm_total_bytes",
    "comm_send_actions",
    "comm_recv_actions",
    "output_bits",
    "max_abs_error",
    "tolerance",
    "agreement",
    "note",
    "error",
    # 重复实验（repeats > 1）才填；单次运行留空
    "repeat",
    "status_counts",
    "value_runs",
    "deviation_rate",
    "wall_ms_p25",
    "wall_ms_median",
    "wall_ms_p75",
)


def write_benchmark_json(records: Sequence[Mapping[str, Any]], path: str) -> str:
    """把记录写成 JSON（主格式；父目录自动创建）。"""

    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(list(records), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_benchmark_csv(records: Sequence[Mapping[str, Any]], path: str) -> str:
    """把记录写成 CSV（表格工具友好；列见 `_CSV_COLUMNS`）。"""

    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CSV_COLUMNS)
        for record in records:
            writer.writerow([_csv_value(record.get(column)) for column in _CSV_COLUMNS])
    return path


def unexpected_records(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """挑出"与预期不符"的记录（口径与 PSI 基线一致）。

    - 带 `expect_status` 的用例：状态不等于预期都算预期外；
    - `expect_status="deviation"`（已知非确定偏差）：`ok` / `error` **都算符合**，
      其余状态（如 `unavailable`）仍算预期外；
    - `mixed`（同一次基线里既有成功又有失败）：只有非确定偏差行能接受它，
      其它行出现 `mixed` 一律算预期外——那是真正的抖动；
    - 其余用例：非 ok / unavailable 即为意外失败；
    - `ok` 但**与明文不一致**也算预期外——整数路径容差为 0。
    """

    unexpected: list[Mapping[str, Any]] = []
    for record in records:
        status = record.get("status")
        expected = record.get("expect_status") or ""
        if expected == EXPECT_DEVIATION:
            # 重复实验下 `mixed` 正是这条偏差该有的样子，不算预期外
            if status not in ("ok", "error", MIXED_STATUS):
                unexpected.append(record)
            continue
        if expected:
            if status != expected:
                unexpected.append(record)
            continue
        if status not in ("ok", "unavailable"):
            unexpected.append(record)
        elif status == "ok" and record.get("agreement") is not True:
            unexpected.append(record)
    return unexpected


def _fmt_num(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return str(value)


def _fmt_bool(value: Any) -> str:
    if value is None:
        return "-"
    return "True" if value else "False"


def format_summary(records: Sequence[Mapping[str, Any]]) -> str:
    """把记录渲染成可读表格（供文档粘贴）。"""

    # 有通信量记录时才多出一列（没开 profiling 的基线不硬塞空列）
    show_comm = any(record.get("comm_total_bytes") is not None for record in records)
    if show_comm:
        lines = [
            "| 用例 | 状态 | 预测 b | 环宽 | 覆盖 | wall_ms | mem_MB | comm_B | agreement |",
            "|---|---|---:|---:|---|---:|---:|---:|---|",
        ]
    else:
        lines = [
            "| 用例 | 状态 | 预测 b | 环宽 | 覆盖 | wall_ms | mem_MB | agreement |",
            "|---|---|---:|---:|---|---:|---:|---|",
        ]
    for record in records:
        cells = {
            "case": record.get("case"),
            "status": record.get("status"),
            "pred": record.get("predicted_bits") if record.get("predicted_bits") is not None else "-",
            "bits": record.get("field_bits"),
            "covered": _fmt_bool(record.get("bit_width_covered")),
            "wall": _fmt_num(record.get("wall_ms")),
            "mem": _fmt_num(record.get("memory_mb")),
            "agr": _fmt_bool(record.get("agreement")),
        }
        if show_comm:
            lines.append(
                "| {case} | {status} | {pred} | {bits} | {covered} | {wall} | {mem} | {comm} | {agr} |".format(
                    comm=_fmt_num(record.get("comm_total_bytes")), **cells
                )
            )
            continue
        lines.append(
            "| {case} | {status} | {pred} | {bits} | {covered} | {wall} | {mem} | {agr} |".format(
                **cells
            )
        )
    return "\n".join(lines)
