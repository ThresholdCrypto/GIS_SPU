# -*- coding: utf-8 -*-
"""打包电路的**前置代价**（P7-P0）：槽内提取到底多贵。

为什么必须先量这一步
--------------------
P6（`packing_probe.py`）测到"秘密 × 秘密"逐元素乘法是 **16 B/元素**，
且单价与输入位宽无关——于是"8 个 8 位值挤进 1 个环元素 ⇒ 通信量 ÷8"成立为上界。

但打包电路**不止**是一次乘法：它要把一个个 8 位值从环元素里**取出来**
（右移 + 掩码）。MPC 里按位操作不是免费的（截断 / 比特分解要走协议），
所以"元素数 ÷8"能不能兑现，取决于这一步的单价。

本探针就量这一件事，并把同一个电路拆成四个变体做 A/B，回答"钱花在哪个原语上"：

| 变体 | 电路 | 期望 |
|---|---|---|
| `mul_only` | `Σ x × c_s`（秘密 × 公开常数） | 本地线性运算，应为 0 B |
| `mask_only` | `Σ (x & 0xFF) × c_s` | 秘密的按位与 |
| `shift_only` | `Σ (x >> 8s) × c_s` | 秘密的算术右移 |
| `shift_mask` | `Σ ((x >> 8s) & 0xFF) × c_s` | 打包电路真正需要的提取 |

严格说清楚本探针**不是**什么
----------------------------
- 它**不是**打包电路：没有跨元素归约、没有槽内进位处理，只量"取槽"一步；
- 它**不给收益结论**：给的是单价对比（B/元素），收益要等打包电路写出来再实测；
- 每个变体都带**自己**的明文参考实现（不是拿申请的目标值去比），
  所以 `within_tolerance` 说的是"这一步算得对不对"，与代价数字分开看。
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
SLOT_COST_OP = "slot_extract"

#: 每元素承载的槽数（课题量化位宽 b = 8）
SLOT_COST_SLOTS = 4

#: 默认元素数（小规模：这一步只比单价，不需要大数组）
SLOT_COST_ELEMENTS = 256

#: 槽宽（8 位）
SLOT_WIDTH_BITS = 8

#: 槽内权重（公开常数）
SLOT_WEIGHTS: tuple[int, ...] = (1, 2, 3, 4)

#: P6 实测的"纯秘密乘法"单价（B/元素），用作比价的基准线
PURE_MUL_BYTES_PER_ELEMENT = 16.0

#: 输入取值上限（保证乘积不溢出 8 位槽）
_VALUE_MAX = 10


def _slot_inputs(elements: int, slots: int) -> tuple[np.ndarray, np.ndarray]:
    """确定性构造：一个打包后的元素向量 + 逐槽明文值。"""

    index = np.arange(1, slots * elements + 1, dtype=np.int64)
    values = ((index % _VALUE_MAX) + 1).reshape(slots, elements)
    packed = np.zeros(elements, dtype=np.int64)
    for slot in range(slots):
        packed |= values[slot] << (SLOT_WIDTH_BITS * slot)
    return packed, values


# --------------------------------------------------------------------------
# 四个变体的电路与各自参考
# --------------------------------------------------------------------------


def _weight(index: int) -> int:
    return SLOT_WEIGHTS[index % len(SLOT_WEIGHTS)]


def _mul_only_fn(x):
    """秘密 × 公开常数的加权和（无位运算）。"""

    import jax.numpy as jnp

    total = jnp.zeros_like(x)
    for slot in range(SLOT_COST_SLOTS):
        total = total + x * jnp.asarray(_weight(slot), dtype=x.dtype)
    return total


def _mask_only_fn(x):
    import jax.numpy as jnp

    total = jnp.zeros_like(x)
    for slot in range(SLOT_COST_SLOTS):
        total = total + (x & jnp.asarray(0xFF, dtype=x.dtype)) * jnp.asarray(
            _weight(slot), dtype=x.dtype
        )
    return total


def _shift_only_fn(x):
    import jax.numpy as jnp

    total = jnp.zeros_like(x)
    for slot in range(SLOT_COST_SLOTS):
        shifted = x >> jnp.asarray(SLOT_WIDTH_BITS * slot, dtype=x.dtype)
        total = total + shifted * jnp.asarray(_weight(slot), dtype=x.dtype)
    return total


def _shift_mask_fn(x):
    """打包电路真正需要的那一步：逐槽提取（算术右移 + 掩码）。"""

    import jax.numpy as jnp

    total = jnp.zeros_like(x)
    for slot in range(SLOT_COST_SLOTS):
        slot_value = (x >> jnp.asarray(SLOT_WIDTH_BITS * slot, dtype=x.dtype)) & jnp.asarray(
            0xFF, dtype=x.dtype
        )
        total = total + slot_value * jnp.asarray(_weight(slot), dtype=x.dtype)
    return total


def _slot_reference(packed: np.ndarray, *, mask: bool, shift: bool) -> np.ndarray:
    """明文参考：用 numpy 复算同一个表达式（位运算在两者间语义一致）。"""

    total = np.zeros_like(packed)
    for slot in range(SLOT_COST_SLOTS):
        piece = packed >> np.int64(SLOT_WIDTH_BITS * slot) if shift else packed
        if mask:
            piece = piece & np.int64(0xFF)
        total = total + piece * np.int64(_weight(slot))
    return total


@dataclass(frozen=True)
class SlotCostVariant:
    """一个变体：名字 + 电路 + 它对应的明文参考。"""

    name: str
    fn: Callable[[Any], Any]
    mask: bool
    shift: bool
    note: str

    def reference(self, packed: np.ndarray) -> np.ndarray:
        return _slot_reference(packed, mask=self.mask, shift=self.shift)


#: 四个变体（顺序固定，产物表格按它排）
SLOT_COST_VARIANTS: tuple[SlotCostVariant, ...] = (
    SlotCostVariant("mul_only", _mul_only_fn, False, False, "无位运算：秘密 × 公开常数"),
    SlotCostVariant("mask_only", _mask_only_fn, True, False, "只掩码：秘密 & 0xFF"),
    SlotCostVariant("shift_only", _shift_only_fn, False, True, "只右移：秘密 >> 8s"),
    SlotCostVariant("shift_mask", _shift_mask_fn, True, True, "全量提取（打包电路的取槽步）"),
)


@dataclass(frozen=True)
class SlotCostCase:
    """一条用例 = 一个变体在一个规模上的运行。"""

    variant: str
    elements: int = SLOT_COST_ELEMENTS
    protocol: str = "ABY3"
    field: str = "FM64"
    repeats: int = 1
    capture_comm: bool = True
    note: str = ""

    def label(self) -> str:
        return f"{SLOT_COST_OP} {self.variant} N={self.elements}"


def slot_cost_cases(*, elements: int = SLOT_COST_ELEMENTS, repeats: int = 1,
                    capture_comm: bool = True) -> list[SlotCostCase]:
    return [
        SlotCostCase(
            variant=variant.name,
            elements=int(elements),
            repeats=max(1, int(repeats)),
            capture_comm=capture_comm,
            note=variant.note,
        )
        for variant in SLOT_COST_VARIANTS
    ]


def _variant(name: str) -> SlotCostVariant:
    for variant in SLOT_COST_VARIANTS:
        if variant.name == name:
            return variant
    raise KeyError(name)


def _blank_record(case: SlotCostCase) -> dict[str, Any]:
    return {
        "case": case.label(),
        "name": case.variant,
        "op": SLOT_COST_OP,
        "protocol": case.protocol,
        "field": case.field,
        "slots": SLOT_COST_SLOTS,
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


def _run_slot_cost_once(case: SlotCostCase, *, report: Any = None) -> dict[str, Any]:
    """跑一条用例；失败也如实记录，不抛异常、不填推测值。"""

    import time

    record = _blank_record(case)

    from .capability import check_capabilities

    report = report or check_capabilities()
    if not report.runnable:
        record["note"] = (
            "当前环境无法真实执行 SPU 模拟；不给任何实测数字"
            "（见 docs/SPU_CAPABILITY.md）"
        )
        return record

    try:
        variant = _variant(case.variant)
    except KeyError:
        record["status"] = "error"
        record["error"] = (
            f"未登记的变体 {case.variant!r}；已知 "
            f"{[item.name for item in SLOT_COST_VARIANTS]}"
        )
        return record

    packed, _ = _slot_inputs(case.elements, SLOT_COST_SLOTS)
    expected = variant.reference(packed)
    start = time.perf_counter()
    run = run_spu_simulation(
        variant.fn,
        [packed],
        protocol=case.protocol,
        field=case.field,
        reference_fn=lambda x, e=expected: e,
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
    record["comm_by_primitive"] = dict(getattr(run, "comm_by_primitive", None) or {})
    notes = [str(note) for note in (getattr(run, "notes", None) or ())]
    if run.status != "ok":
        notes.append(f"执行未成功：{run.status}（{run.error or '未给原因'}）")
    record["status"] = run.status
    record["error"] = run.error or ""
    record["note"] = "；".join([case.note, *notes]).strip("；")

    comm = record["comm_total_bytes"]
    if isinstance(comm, (int, float)) and case.elements > 0:
        record["comm_per_element"] = comm / case.elements
    else:
        record["note"] = (record["note"] + "；通信量为空").strip("；")
    return record


def run_slot_cost_case(case: SlotCostCase, *, report: Any = None) -> dict[str, Any]:
    """跑一条用例；`repeats > 1` 时返回多次运行的统计记录（报中位数）。

    P7-P0 的教训：通信量有批间抖动（实测 0.03%–10.1%），单次读数不足以支撑
    "哪条路线更贵"的判断，所以重复次数必须真的重复执行、报中位数与区间。
    """

    repeats = max(1, int(case.repeats))
    if repeats == 1:
        return _run_slot_cost_once(case, report=report)

    import statistics

    samples = [_run_slot_cost_once(case, report=report) for _ in range(repeats)]
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


def run_slot_cost_cases(
    cases: Sequence[SlotCostCase], *, report: Any = None, progress: Any = None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in cases:
        record = run_slot_cost_case(case, report=report)
        records.append(record)
        if progress is not None:
            progress(record)
    return records


# --------------------------------------------------------------------------
# 结论
# --------------------------------------------------------------------------


def summarize_slot_cost(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """把四个变体的单价摊开；缺数就留空并说明原因。"""

    by_name = {str(record.get("name")): record for record in records}

    def per_element(name: str) -> float | None:
        value = (by_name.get(name) or {}).get("comm_per_element")
        return float(value) if isinstance(value, (int, float)) else None

    summary: dict[str, Any] = {
        "per_element_bytes": {},
        "extraction": None,
        "notes": [],
    }
    for variant in SLOT_COST_VARIANTS:
        value = per_element(variant.name)
        if value is not None:
            summary["per_element_bytes"][variant.name] = value

    extraction = per_element("shift_mask")
    if extraction is None:
        summary["notes"].append("没有取槽步（shift_mask）的通信量，无法与纯乘法比价")
        return summary

    summary["extraction"] = {
        "bytes_per_element": extraction,
        "pure_multiply_bytes_per_element": PURE_MUL_BYTES_PER_ELEMENT,
        "times_pure_multiply": extraction / PURE_MUL_BYTES_PER_ELEMENT,
        "baseline_source": (
            "基准线取自 P6 探针实测（docs/mpc_packing_probe.json：ABY3 × 秘密乘法 "
            f"{PURE_MUL_BYTES_PER_ELEMENT:.0f} B/元素）；本探针里的 mul_only 变体是"
            "秘密 × 公开常数，实测为 0 B，不能当基准（它根本不通信）"
        ),
        "reading": (
            "取槽步的单价是纯乘法的 "
            f"{extraction / PURE_MUL_BYTES_PER_ELEMENT:.0f} 倍——"
            "打包电路要省通信量，必须先让元素数下降的倍数**超过**这个倍率，"
            "否则越打包越贵"
        ),
        "caveat": (
            "这只是“取槽”一步，不含跨元素归约；也不是打包电路的实测收益。"
            "本探针回答“钱花在哪”，不回答“打包值不值”"
        ),
    }

    mul_only = per_element("mul_only")
    if mul_only is not None:
        summary["mul_only_is_free"] = mul_only == 0.0
        if mul_only != 0.0:
            summary["notes"].append(
                "mul_only（秘密 × 公开常数）实测不为 0，与“本地线性运算免费”的"
                "预期不符：需要复核"
            )
    for name in ("mask_only", "shift_only"):
        if per_element(name) is None:
            summary["notes"].append(f"{name} 没有通信量，A/B 拆解不完整")
    return summary


# --------------------------------------------------------------------------
# 产物
# --------------------------------------------------------------------------

SLOT_COST_CSV_COLUMNS: tuple[str, ...] = (
    "case",
    "name",
    "op",
    "protocol",
    "field",
    "slots",
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


def write_slot_cost_json(records: Sequence[Mapping[str, Any]], path: str) -> str:
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


def write_slot_cost_csv(records: Sequence[Mapping[str, Any]], path: str) -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SLOT_COST_CSV_COLUMNS)
        for record in records:
            writer.writerow(
                [_csv_value(record.get(column)) for column in SLOT_COST_CSV_COLUMNS]
            )
    return path


def format_slot_cost_summary(records: Sequence[Mapping[str, Any]]) -> str:
    header = f"{'variant':12s} {'status':12s} {'comm_B':>10s} {'B/元素':>9s}  通信原语"
    lines = [header]
    for record in records:
        comm = record.get("comm_total_bytes")
        per = record.get("comm_per_element")
        primitives = ",".join(sorted((record.get("comm_by_primitive") or {}).keys()))
        lines.append(
            f"{str(record.get('name', ''))[:12]:12s} "
            f"{str(record.get('status', '')):12s} "
            f"{(f'{comm:.0f}' if isinstance(comm, (int, float)) else '-'):>10s} "
            f"{(f'{per:.2f}' if isinstance(per, (int, float)) else '-'):>9s}  "
            f"{primitives}"
        )
    summary = summarize_slot_cost(records)
    extraction = summary.get("extraction")
    if extraction:
        lines.append("")
        lines.append(
            f"取槽步 {extraction['bytes_per_element']:.0f} B/元素 = 纯乘法"
            f"（{extraction['pure_multiply_bytes_per_element']:.0f} B/元素）的 "
            f"{extraction['times_pure_multiply']:.0f} 倍"
        )
    for note in summary.get("notes") or ():
        lines.append(f"说明：{note}")
    return "\n".join(lines)


__all__ = [
    "PURE_MUL_BYTES_PER_ELEMENT",
    "SLOT_COST_CSV_COLUMNS",
    "SLOT_COST_ELEMENTS",
    "SLOT_COST_OP",
    "SLOT_COST_SLOTS",
    "SLOT_COST_VARIANTS",
    "SLOT_WIDTH_BITS",
    "SlotCostCase",
    "SlotCostVariant",
    "format_slot_cost_summary",
    "run_slot_cost_case",
    "run_slot_cost_cases",
    "slot_cost_cases",
    "summarize_slot_cost",
    "write_slot_cost_csv",
    "write_slot_cost_json",
]
