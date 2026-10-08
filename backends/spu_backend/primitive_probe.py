# -*- coding: utf-8 -*-
"""逐原语真机核验（P0）：capability 表"登记为已适配"的原语，真机到底跑不跑得动。

为什么需要这一个探针
--------------------
`backends/spu_backend/capability.py` 里有两张白名单
（`SPU_ADAPTED_PRIMITIVES` 的 jax 层、`SPU_ADAPTED_HLO_PRIMITIVES` 的 StableHLO 层），
它们的**证据是"官方 CHANGELOG / 测试目录的算子命名"**——也就是"登记"，不是"真机跑过"。

这个区别在 D3 打包方案上正好是卡点（`docs/BITPLANE_LAYOUT.md` §7 第 2 步）：
打包电路要用 `shift_left` / `bitwise_and` / `bitwise_or` / `dot`，
而本轮之前**只真机测过** `shift_right_arithmetic` 与 `and`（P7-P0 探针）。
"表里有"不等于"跑得动"——**登记必须有真机证据**。

本探针把每个原语写成一条**最小电路**（秘密输入 → 该原语 → 输出），在真机上执行，
并记录三件事：`status`（跑不跑得动）、`comm_total_bytes`（要不要通信、通信多少）、
以及 `comm_by_primitive`（真机日志里到底执行了哪些原语）。

严格说清楚本探针**不是**什么
----------------------------
- 它**不给算子可用性结论**：一个原语在该测试形态下跑通，不等于它在任意电路形态下
  都跑通（形态依赖真实存在，见 `docs/MPC_BENCHMARK_PROTOCOL.md` §4.1 的
  `TemporalOverlap` 两套电路）；
- 它**不是性能测试**：元素数取小（只回答"能不能跑"），通信量只作为
  "这个原语要不要钱"的定性证据，不做跨原语排序（形态与规模都会影响它）；
- 它**不覆盖全部登记原语**：只覆盖与打包/值布局相关、以及历史上易碎的那几类
  （见 `PROBE_PRIMITIVES` 的 `needed_by` 字段），浮点超越函数（`exp`/`log`/`sqrt`）
  不在本探针范围——它们不参与位平面打包；
- 参考实现给的是**该电路的明文语义**，所以 `within_tolerance=False` 表示
  "真机结果与明文语义不符"，是有意义的一条记录，不是探针出错。
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .runtime import run_spu_simulation

#: 探针算子名（进产物，便于表格工具筛选）
PRIMITIVE_PROBE_OP = "primitive_check"

#: 默认元素数（只回答"能不能跑"，不需要大数组）
PRIMITIVE_ELEMENTS = 64

#: 输入取值范围（小整数：避免溢出干扰"能不能跑"的判定）
_VALUE_MAX = 10

#: P6 实测的"秘密 × 秘密"逐元素乘法单价（B/元素），用于"低于下限"对账
MUL_BYTES_PER_ELEMENT = 16.0


def _probe_inputs(elements: int) -> tuple[np.ndarray, np.ndarray]:
    """确定性构造两路秘密输入（取值 1.._VALUE_MAX）。"""

    index = np.arange(1, elements + 1, dtype=np.int64)
    x = (index % _VALUE_MAX) + 1
    y = ((index * 3) % _VALUE_MAX) + 1
    return x, y


# --------------------------------------------------------------------------
# 最小电路：每个原语一条
# --------------------------------------------------------------------------


def _fn_add(x, y):
    import jax.numpy as jnp

    return x + y


def _fn_sub(x, y):
    return x - y


def _fn_mul(x, y):
    return x * y


def _fn_neg(x, y):
    return -x


def _fn_max(x, y):
    import jax.numpy as jnp

    return jnp.maximum(x, y)


def _fn_min(x, y):
    import jax.numpy as jnp

    return jnp.minimum(x, y)


def _fn_and(x, y):
    return x & y


def _fn_or(x, y):
    return x | y


def _fn_xor(x, y):
    return x ^ y


def _fn_not(x, y):
    return ~x


def _fn_shift_left(x, y):
    import jax.numpy as jnp

    return x << jnp.asarray(3, dtype=x.dtype)


def _fn_shift_right(x, y):
    import jax.numpy as jnp

    return x >> jnp.asarray(3, dtype=x.dtype)


def _fn_dot(x, y):
    import jax.numpy as jnp

    return jnp.dot(x, y)


def _fn_sum(x, y):
    import jax.numpy as jnp

    return jnp.sum(x)


def _fn_compare(x, y):
    import jax.numpy as jnp

    return jnp.sum((x < y).astype(x.dtype))


def _fn_select(x, y):
    import jax.numpy as jnp

    return jnp.where(x > y, x, y)


def _fn_sort(x, y):
    import jax.numpy as jnp

    return jnp.sort(x)


def _fn_top_k(x, y):
    import jax.lax as lax

    return lax.top_k(x, 3)[0]


def _fn_div(x, y):
    return x // (y + 1)


# --------------------------------------------------------------------------
# 用例表
# --------------------------------------------------------------------------

#: 原语 → 该原语在 jax 层的登记名（`capability.SPU_ADAPTED_PRIMITIVES` 的成员）
_JAX_NAME: Mapping[str, str] = {
    "add": "add", "sub": "sub", "mul": "mul", "neg": "neg",
    "max": "max", "min": "min",
    "and": "and", "or": "or", "xor": None, "not": "not",
    "shift_left": "shift_left", "shift_right": "shift_right",
    "dot": "dot", "sum": "sum", "compare": "comparisons", "select": "select",
    "sort": "sort", "top_k": "top_k", "div": "div",
}

#: 原语 → 该原语在 StableHLO 层的登记名（`capability.SPU_ADAPTED_HLO_PRIMITIVES`）
_HLO_NAME: Mapping[str, str] = {
    "add": "add", "sub": "subtract", "mul": "multiply", "neg": "negate",
    "max": "max", "min": "min",
    "and": "and", "or": "or", "xor": None, "not": "not",
    "shift_left": "shift_left", "shift_right": "shift_right_arithmetic",
    "dot": "dot", "sum": "reduce", "compare": "compare", "select": "select",
    "sort": "sort", "top_k": None, "div": "divide",
}


@dataclass(frozen=True)
class PrimitiveCase:
    """一条用例 = 一个原语的最小电路。"""

    name: str
    fn: Callable[..., Any]
    reference_fn: Callable[..., Any]
    needed_by: str  # packing / layout / general / fragile
    note: str = ""
    elements: int = PRIMITIVE_ELEMENTS
    protocol: str = "ABY3"
    field: str = "FM64"
    repeats: int = 1
    capture_comm: bool = True

    @property
    def jax_name(self) -> str | None:
        return _JAX_NAME.get(self.name)

    @property
    def hlo_name(self) -> str | None:
        return _HLO_NAME.get(self.name)

    def label(self) -> str:
        return f"{PRIMITIVE_PROBE_OP} {self.name} N={self.elements}"


#: 参考实现（numpy 复算同一个电路）
def _ref(fn: Callable[[Any, Any], Any]) -> Callable[..., Any]:
    def reference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(fn(x, y))

    return reference


def _case(name: str, fn, needed_by: str, note: str = "") -> PrimitiveCase:
    return PrimitiveCase(name=name, fn=fn, reference_fn=_ref(fn), needed_by=needed_by, note=note)


#: 探针覆盖的原语（顺序固定，产物表格按它排）
PROBE_PRIMITIVES: tuple[PrimitiveCase, ...] = (
    _case("mul", _fn_mul, "general", "基线：秘密 × 秘密（P6 实测 16 B/元素）"),
    _case("add", _fn_add, "general", "秘密 + 秘密"),
    _case("sub", _fn_sub, "general", ""),
    _case("neg", _fn_neg, "general", "取负"),
    _case("max", _fn_max, "layout", "上界"),
    _case("min", _fn_min, "layout", "下界"),
    _case("and", _fn_and, "packing", "打包取槽要用（P7 只测过“秘密 & 公开常数”）"),
    _case("or", _fn_or, "packing", "打包/位平面合并要用"),
    _case("xor", _fn_xor, "packing", "位平面异或用（登记表里**没有** xor，见 notes）"),
    _case("not", _fn_not, "packing", "位取反"),
    _case("shift_left", _fn_shift_left, "packing", "打包装槽要用（D3 §7 点名，此前未真机测）"),
    _case(
        "shift_right", _fn_shift_right, "packing",
        "算术右移（P7 测过“秘密 >> 常数”，这里测“秘密 >> 秘密输入的同型张量”）",
    ),
    _case("dot", _fn_dot, "packing", "打包归约候选（D3 §7 点名，此前未真机测）"),
    _case("sum", _fn_sum, "layout", "跨元素求和（归约轴聚合的基元）"),
    _case("compare", _fn_compare, "layout", "逐候选判定（L2 归约轴的非线性核心）"),
    _case("select", _fn_select, "layout", "jnp.where（逐候选判定）"),
    _case("sort", _fn_sort, "fragile", "依赖 frontend 的 float→int 补丁，跨版本易碎"),
    _case("top_k", _fn_top_k, "fragile", "登记为已适配但历史未真机核验"),
    _case("div", _fn_div, "fragile", "登记表标为高代价原语，核验其真机可用性"),
)


def primitive_probe_cases(*, elements: int = PRIMITIVE_ELEMENTS, repeats: int = 1,
                          capture_comm: bool = True,
                          only: Sequence[str] | None = None) -> list[PrimitiveCase]:
    wanted = set(only) if only else None
    out: list[PrimitiveCase] = []
    for case in PROBE_PRIMITIVES:
        if wanted is not None and case.name not in wanted:
            continue
        out.append(
            PrimitiveCase(
                name=case.name,
                fn=case.fn,
                reference_fn=case.reference_fn,
                needed_by=case.needed_by,
                note=case.note,
                elements=int(elements),
                protocol=case.protocol,
                field=case.field,
                repeats=max(1, int(repeats)),
                capture_comm=capture_comm,
            )
        )
    return out


def _blank_record(case: PrimitiveCase) -> dict[str, Any]:
    return {
        "case": case.label(),
        "name": case.name,
        "op": PRIMITIVE_PROBE_OP,
        "jax_primitive": case.jax_name,
        "hlo_primitive": case.hlo_name,
        "needed_by": case.needed_by,
        "registered_jax": case.jax_name is not None,
        "registered_hlo": case.hlo_name is not None,
        "protocol": case.protocol,
        "field": case.field,
        "elements": case.elements,
        "repeat": max(1, int(case.repeats)),
        "status": "unavailable",
        "error": "",
        "note": "",
        "wall_ms": None,
        "pphlo_bytes": None,
        "profiled": False,
        "comm_send_bytes": None,
        "comm_recv_bytes": None,
        "comm_total_bytes": None,
        "comm_per_element": None,
        "comm_by_primitive": {},
        "within_tolerance": None,
        "max_abs_error": None,
        "tolerance": 0.0,
    }


def _run_primitive_once(case: PrimitiveCase, *, report: Any = None) -> dict[str, Any]:
    """跑一条用例；失败也如实记录，不抛异常、不填推测值。"""

    import time

    from .capability import check_capabilities

    record = _blank_record(case)
    report = report or check_capabilities()
    if not report.runnable:
        record["note"] = (
            "当前环境无法真实执行 SPU 模拟；不给任何实测数字"
            "（见 docs/SPU_CAPABILITY.md）"
        )
        return record

    x, y = _probe_inputs(case.elements)
    start = time.perf_counter()
    run = run_spu_simulation(
        case.fn,
        [x, y],
        protocol=case.protocol,
        field=case.field,
        reference_fn=case.reference_fn,
        tolerance=0.0,
        report=report,
        capture_comm=case.capture_comm,
    )
    record["wall_ms"] = (time.perf_counter() - start) * 1000.0
    for key in (
        "comm_send_bytes",
        "comm_recv_bytes",
        "comm_total_bytes",
        "pphlo_bytes",
        "within_tolerance",
        "max_abs_error",
        "tolerance",
    ):
        record[key] = getattr(run, key, None)
    record["profiled"] = bool(getattr(run, "profiled", False))
    record["comm_by_primitive"] = dict(getattr(run, "comm_by_primitive", None) or ())
    notes = [str(note) for note in (getattr(run, "notes", None) or ())]
    if run.status != "ok":
        notes.append(f"执行未成功：{run.status}（{run.error or '未给原因'}）")
    record["status"] = run.status
    record["error"] = run.error or ""
    record["note"] = "；".join([case.note, *notes]).strip("；")

    comm = record["comm_total_bytes"]
    if isinstance(comm, (int, float)) and case.elements > 0:
        record["comm_per_element"] = comm / case.elements
    return record


def run_primitive_case(case: PrimitiveCase, *, report: Any = None) -> dict[str, Any]:
    """跑一条用例；`repeats > 1` 时返回多次运行的统计记录（报中位数）。

    P7-P0 的教训：通信量有批间抖动（实测 0.03%–10.1%），单次读数不足以支撑
    "哪条路线更贵"的判断，所以重复次数必须真的重复执行、报中位数与区间。
    """

    repeats = max(1, int(case.repeats))
    if repeats == 1:
        return _run_primitive_once(case, report=report)

    import statistics

    samples = [_run_primitive_once(case, report=report) for _ in range(repeats)]
    record = dict(samples[-1])
    record["repeat"] = repeats
    record["status_counts"] = {}
    for sample in samples:
        key = str(sample["status"])
        record["status_counts"][key] = record["status_counts"].get(key, 0) + 1
    for key in (
        "wall_ms",
        "pphlo_bytes",
        "comm_total_bytes",
        "comm_send_bytes",
        "comm_recv_bytes",
        "comm_per_element",
    ):
        values = [
            float(sample[key])
            for sample in samples
            if isinstance(sample.get(key), (int, float))
        ]
        record[key] = statistics.median(values) if values else None
        record[key + "_stats"] = (
            {"min": min(values), "max": max(values), "median": statistics.median(values)}
            if values
            else None
        )
    return record


def run_primitive_cases(
    cases: Sequence[PrimitiveCase], *, report: Any = None, progress: Any = None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in cases:
        record = run_primitive_case(case, report=report)
        records.append(record)
        if progress is not None:
            progress(record)
    return records


# --------------------------------------------------------------------------
# 结论
# --------------------------------------------------------------------------


#: 每条电路从语义上**必须**含多少个"秘密 × 秘密"乘法（按输入元素数计）。
#: 用它跟实测通信量对账：低于下限的读数物理上不可能，必须标可疑，
#: 不能当成"这个原语免费"——`x * 2` 实测 0 B 就是这个坑（P6 的教训）。
MULTIPLY_DEMAND: Mapping[str, float] = {
    "mul": 1.0,  # 逐元素秘密乘法
    "dot": 1.0,  # 长度为 N 的秘密收缩 = N 次秘密乘法
}


def summarize_primitives(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """把"登记"与"实测"摊开：哪些原语真机跑通、哪些没跑通、有没有登记错。"""

    executed: list[str] = []
    failed: list[str] = []
    silent: list[str] = []
    unverified_claims: list[str] = []
    notes: list[str] = []

    for record in records:
        name = str(record.get("name"))
        status = record.get("status")
        if status == "ok" and record.get("within_tolerance") is not False:
            executed.append(name)
        elif status == "unavailable":
            continue
        elif status == "error" and record.get("within_tolerance") is False:
            failed.append(name)
        elif status == "error":
            failed.append(name)
        else:
            silent.append(name)

        if record.get("registered_jax") and status != "ok":
            unverified_claims.append(name)

    summary: dict[str, Any] = {
        "executed": tuple(executed),
        "not_executed": tuple(failed + silent),
        "failed": tuple(failed),
        "registered_but_not_executed": tuple(unverified_claims),
        "notes": notes,
    }

    if not records:
        notes.append("没有记录，无法判定")
        return summary

    if unverified_claims:
        notes.append(
            "以下原语在 capability 表里登记为已适配，但本探针真机没跑通："
            f"{sorted(unverified_claims)}——登记表与实测不一致，必须复核"
        )
    # 与"秘密乘法下限"对账（P6 实测 16 B/元素/次乘法）
    floor_by_name = {str(r.get("name")): r for r in records}
    below: list[str] = []
    for name, demand in MULTIPLY_DEMAND.items():
        record = floor_by_name.get(name)
        if not record:
            continue
        comm = record.get("comm_total_bytes")
        elements = record.get("elements")
        if not isinstance(comm, (int, float)) or not isinstance(elements, (int, float)):
            continue
        floor = MUL_BYTES_PER_ELEMENT * float(elements) * float(demand)
        if comm < floor:
            below.append(name)
            notes.append(
                f"{name} 实测通信量 {comm:.0f} B 低于秘密乘法下限 {floor:.0f} B "
                f"（{MUL_BYTES_PER_ELEMENT:.0f} B/元素 × {int(elements)}）："
                "要么该原语有更省的专用协议，要么这条路径没在真做保密乘法——"
                "本探针不给结论，标为可疑，需另行复核"
            )
    summary["below_multiply_floor"] = tuple(below)

    packing = [r.get("name") for r in records if r.get("needed_by") == "packing"]
    summary["packing_primitives"] = tuple(str(name) for name in packing)
    return summary


# --------------------------------------------------------------------------
# 产物
# --------------------------------------------------------------------------

PRIMITIVE_CSV_COLUMNS: tuple[str, ...] = (
    "case",
    "name",
    "op",
    "jax_primitive",
    "hlo_primitive",
    "needed_by",
    "registered_jax",
    "registered_hlo",
    "protocol",
    "field",
    "elements",
    "repeat",
    "status",
    "wall_ms",
    "pphlo_bytes",
    "profiled",
    "comm_send_bytes",
    "comm_recv_bytes",
    "comm_total_bytes",
    "comm_per_element",
    "comm_by_primitive",
    "within_tolerance",
    "max_abs_error",
    "tolerance",
    "note",
    "error",
)


def write_primitive_json(records: Sequence[Mapping[str, Any]], path: str) -> str:
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


def write_primitive_csv(records: Sequence[Mapping[str, Any]], path: str) -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PRIMITIVE_CSV_COLUMNS)
        for record in records:
            writer.writerow(
                [_csv_value(record.get(column)) for column in PRIMITIVE_CSV_COLUMNS]
            )
    return path


def format_primitive_summary(records: Sequence[Mapping[str, Any]]) -> str:
    header = f"{'primitive':12s} {'needed_by':10s} {'status':12s} {'对拍':>5s} {'comm_B':>10s} {'B/元素':>9s}  真机原语"
    lines = [header]
    for record in records:
        comm = record.get("comm_total_bytes")
        per = record.get("comm_per_element")
        match = record.get("within_tolerance")
        mark = "-" if match is None else ("是" if match else "否")
        primitives = ",".join(sorted((record.get("comm_by_primitive") or {}).keys()))
        lines.append(
            f"{str(record.get('name', ''))[:12]:12s} "
            f"{str(record.get('needed_by', ''))[:10]:10s} "
            f"{str(record.get('status', '')):12s} "
            f"{mark:>5s} "
            f"{(f'{comm:.0f}' if isinstance(comm, (int, float)) else '-'):>10s} "
            f"{(f'{per:.2f}' if isinstance(per, (int, float)) else '-'):>9s}  "
            f"{primitives}"
        )
    summary = summarize_primitives(records)
    lines.append("")
    lines.append(f"真机跑通：{len(summary['executed'])} 项 {list(summary['executed'])}")
    if summary["not_executed"]:
        lines.append(f"未跑通：{list(summary['not_executed'])}")
    for note in summary.get("notes") or ():
        lines.append(f"说明：{note}")
    return "\n".join(lines)


__all__ = [
    "PRIMITIVE_CSV_COLUMNS",
    "PRIMITIVE_ELEMENTS",
    "PRIMITIVE_PROBE_OP",
    "PROBE_PRIMITIVES",
    "PrimitiveCase",
    "format_primitive_summary",
    "primitive_probe_cases",
    "run_primitive_case",
    "run_primitive_cases",
    "summarize_primitives",
    "write_primitive_csv",
    "write_primitive_json",
]
