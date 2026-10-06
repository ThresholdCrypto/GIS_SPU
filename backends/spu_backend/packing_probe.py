# -*- coding: utf-8 -*-
"""打包收益的**上界**实测探针（P6）：SPU 到底按"元素"还是按"位"计费。

为什么需要这一个探针
--------------------
位平面布局（D3，`planner/layout.py`）的条数模型有一个前提：
**通信量由密文条数（环元素个数）主导，输入位宽不影响单价**。
没有这条前提，"把 8 个 8 位值挤进 1 个元素"就没有收益可言——如果 SPU 对
8 位输入本来就收 1/8 的钱，打包纯属白干。

P5 只能把这条前提标成"未验证"（见 `docs/BITPLANE_LAYOUT.md` §4）。本探针把它
变成**实测**，并且只回答两件事：

  实验 A（位宽）：同一算子、同一元素数，输入取 int8 / int32 / int64
                  → 三者通信量若相同，说明单价按**环元素**收，与输入位宽无关；
  实验 B（规模）：同一位宽，元素数 512 / 1024 / 4096
                  → 给出"每元素多少字节"，并检查通信量是否随元素数线性增长；
                  有了 B/元素 就能算出打包 8 值/元素 的**上界**收益。

> 本探针第一版用 `x * 2`（乘公开常数），实测通信量恒为 **0 字节**——SPU 把
> 秘密 × 公开常数编译成本地线性运算，不通信。所以现版本改用**秘密 × 秘密**
> 的逐元素乘法（`mul`，最小的、确定会通信的 MPC 原语）作为探针算子。

严格说清楚本探针**不是**什么
----------------------------
- 它**不是**打包电路。跑的是逐元素秘密乘法，不是"槽内归约"。打包电路要在
  槽内完成归约，那部分的通信量必须另行实测，不能拿本文的比例外推；
- 它**只给上界**：真实打包电路的收益只可能低于"元素数按 8 倍减少"这个界；
- 输入值刻意取 1..10，使乘积在 int8 下也不溢出——三档位宽比的是同一个语义，
  不是三个不同的回绕结果。
"""

from __future__ import annotations

import csv
import json
import os
import resource
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .runtime import run_spu_simulation

#: 探针使用的算子名（进产物，便于表格工具筛选）
PROBE_OP = "mul_ss"

#: 输入取值范围：1..10（乘积 ≤ 100，int8 不溢出；且保证秘密×秘密真在通信）
_PROBE_VALUE_MAX = 10

#: 位宽扫描的元素数（三个位宽都用它）
_WIDTH_SWEEP_ELEMENTS = 4096

#: 规模扫描的元素数（固定 int64；4096 复用位宽扫描的 int64 行）
_SCALE_SWEEP_ELEMENTS: tuple[int, ...] = (512, 1024)

#: 打包口径：每个元素承载多少个 8 位值（课题量化位宽 b = 8）
PACKED_VALUES_PER_ELEMENT = 8

_DTYPES: Mapping[str, Any] = {
    "int8": np.int8,
    "int32": np.int32,
    "int64": np.int64,
}


# --------------------------------------------------------------------------
# 用例
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PackingProbeCase:
    """一条探针用例：固定算子（秘密×秘密逐元素乘），只变位宽 / 规模。"""

    name: str
    dtype: str
    elements: int
    protocol: str = "ABY3"
    field: str = "FM64"
    repeats: int = 1
    capture_comm: bool = True
    notes: str = ""

    @property
    def values(self) -> int:
        """该用例承载的 8 位值个数（逐点口径下等于元素数）。"""

        return self.elements

    def label(self) -> str:
        return f"{PROBE_OP} {self.dtype} N={self.elements}"


def packing_probe_cases(
    *, repeats: int = 1, capture_comm: bool = True
) -> list[PackingProbeCase]:
    """标准探针用例集：位宽扫描（3 条）+ 规模扫描（2 条），共 5 条。

    `int64 @ N=4096` 同时属于两条扫描，只跑一次。
    """

    cases: list[PackingProbeCase] = []
    for dtype in ("int8", "int32", "int64"):
        cases.append(
            PackingProbeCase(
                name=f"width-{dtype}",
                dtype=dtype,
                elements=_WIDTH_SWEEP_ELEMENTS,
                protocol="ABY3",
                field="FM64",
                repeats=max(1, int(repeats)),
                capture_comm=capture_comm,
                notes="位宽扫描：同一算子、同一元素数，只变输入位宽",
            )
        )
    for elements in _SCALE_SWEEP_ELEMENTS:
        cases.append(
            PackingProbeCase(
                name=f"scale-{elements}",
                dtype="int64",
                elements=elements,
                protocol="ABY3",
                field="FM64",
                repeats=max(1, int(repeats)),
                capture_comm=capture_comm,
                notes="规模扫描：同一位宽，只变元素数",
            )
        )
    return cases


# --------------------------------------------------------------------------
# 执行
# --------------------------------------------------------------------------


def _build_inputs(case: PackingProbeCase) -> tuple[np.ndarray, np.ndarray]:
    """确定性构造一对输入（无随机数：同一参数在任何机器上给出同一批输入）。"""

    index = np.arange(1, case.elements + 1, dtype=np.int64)
    left = (index % _PROBE_VALUE_MAX) + 1
    right = ((index * 3) % _PROBE_VALUE_MAX) + 1
    dtype = _DTYPES[case.dtype]
    return left.astype(dtype), right.astype(dtype)


def _probe_fn(x, y):
    """探针算子：两个**秘密**张量的逐元素乘法（真会通信的 MPC 原语）。"""

    return x * y


def _reference(x, y):
    return np.asarray(x, dtype=np.int64) * np.asarray(y, dtype=np.int64)


def _blank_record(case: PackingProbeCase) -> dict[str, Any]:
    return {
        "case": case.label(),
        "name": case.name,
        "op": PROBE_OP,
        "protocol": case.protocol,
        "field": case.field,
        "dtype": case.dtype,
        "elements": case.elements,
        "repeat": max(1, int(case.repeats)),
        "status": "unavailable",
        "error": "",
        "note": "",
        "wall_ms": None,
        "peak_rss_mb": None,
        "pphlo_bytes": None,
        "profiled": False,
        "comm_by_primitive": {},
        "comm_send_bytes": None,
        "comm_recv_bytes": None,
        "comm_total_bytes": None,
        "comm_per_element": None,
        "within_tolerance": None,
        "max_abs_error": None,
        "tolerance": 0.0,
    }


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024.0


def _run_probe_once(case: PackingProbeCase, *, report: Any = None) -> dict[str, Any]:
    """跑一次；失败也如实记录，不抛异常、不填推测值。"""

    record = _blank_record(case)

    from .capability import check_capabilities

    report = report or check_capabilities()
    if not report.runnable:
        record["note"] = (
            "当前环境无法真实执行 SPU 模拟；不给任何实测数字"
            "（见 docs/SPU_CAPABILITY.md）"
        )
        return record

    if case.dtype not in _DTYPES:
        record["status"] = "error"
        record["error"] = f"未登记的输入位宽 {case.dtype!r}；已知 {sorted(_DTYPES)}"
        return record

    left, right = _build_inputs(case)
    before_peak = _peak_rss_mb()
    start = time.perf_counter()
    run = run_spu_simulation(
        _probe_fn,
        [left, right],
        protocol=case.protocol,
        field=case.field,
        reference_fn=_reference,
        tolerance=0.0,
        report=report,
        capture_comm=case.capture_comm,
    )
    record["wall_ms"] = (time.perf_counter() - start) * 1000.0
    record["peak_rss_mb"] = max(before_peak, _peak_rss_mb())
    record["pphlo_bytes"] = getattr(run, "pphlo_bytes", None)
    record["profiled"] = bool(getattr(run, "profiled", False))
    for key in ("comm_send_bytes", "comm_recv_bytes", "comm_total_bytes"):
        record[key] = getattr(run, key, None)
    record["within_tolerance"] = getattr(run, "within_tolerance", None)
    record["comm_by_primitive"] = dict(getattr(run, "comm_by_primitive", None) or {})
    record["max_abs_error"] = getattr(run, "max_abs_error", None)
    record["tolerance"] = getattr(run, "tolerance", 0.0)
    notes = [str(note) for note in (getattr(run, "notes", None) or ())]
    if run.status != "ok":
        notes.append(f"执行未成功：{run.status}（{run.error or '未给原因'}）")
    record["status"] = run.status
    record["error"] = run.error or ""
    record["note"] = "；".join([case.notes, *notes]).strip("；")

    comm = record["comm_total_bytes"]
    if isinstance(comm, (int, float)) and case.elements > 0:
        record["comm_per_element"] = comm / case.elements
    else:
        record["note"] = (record["note"] + "；通信量为空").strip("；")
    return record


def run_packing_probe_case(case: PackingProbeCase, *, report: Any = None) -> dict[str, Any]:
    """跑一条用例；`repeats > 1` 时返回多次运行的统计记录（报中位数）。"""

    repeats = max(1, int(case.repeats))
    if repeats == 1:
        return _run_probe_once(case, report=report)

    import statistics

    samples = [_run_probe_once(case, report=report) for _ in range(repeats)]
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
        record[f"{key}_stats"] = (
            {
                "min": min(values),
                "max": max(values),
                "median": statistics.median(values),
            }
            if values
            else None
        )
    return record


def run_packing_probe_cases(
    cases: Sequence[PackingProbeCase],
    *,
    report: Any = None,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """逐条执行（串行；SPU 模拟不保证进程内并发安全）。"""

    records: list[dict[str, Any]] = []
    for case in cases:
        record = run_packing_probe_case(case, report=report)
        records.append(record)
        if progress is not None:
            progress(record)
    return records


# --------------------------------------------------------------------------
# 结论（由实测数字算出，不由人写）
# --------------------------------------------------------------------------

#: 判"接近 1"的相对容差（通信量是整数计数，正常应完全相等）
_RATIO_TOLERANCE = 1e-9


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    top, bottom = _number(numerator), _number(denominator)
    if top is None or bottom is None or bottom == 0:
        return None
    return top / bottom


def summarize_packing(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """从记录里算出结论；任一输入缺失就返回 `None` 并说明原因。"""

    by_name = {str(record.get("name")): record for record in records}

    def comm(name: str) -> float | None:
        return _number((by_name.get(name) or {}).get("comm_total_bytes"))

    def elements(name: str) -> int | None:
        value = (by_name.get(name) or {}).get("elements")
        return int(value) if isinstance(value, int) else None

    summary: dict[str, Any] = {
        "width_ratios": {},
        "pricing_per_element": None,
        "width_effect": None,
        "linearity": None,
        "packing_upper_bound": None,
        "notes": [],
    }

    # ---- 实验 A：位宽 ----------------------------------------------------
    base = comm("width-int64")
    widths = {dtype: comm(f"width-{dtype}") for dtype in ("int8", "int32", "int64")}
    if base:
        for dtype, value in widths.items():
            ratio = _ratio(value, base)
            if ratio is not None:
                summary["width_ratios"][dtype] = ratio
        deviations = [abs(ratio - 1.0) for ratio in summary["width_ratios"].values()]
        per_element = bool(deviations) and max(deviations) <= _RATIO_TOLERANCE
        summary["pricing_per_element"] = per_element
        summary["width_effect"] = (
            "同一元素数下 int8/int32/int64 的通信量之比全部为 1："
            "SPU 按**环元素**计费，与输入位宽无关。打包的收益空间由此而来"
            if per_element
            else "三个位宽的通信量不相等：单价与输入位宽有关，打包口径需要重新论证"
        )
    elif base == 0:
        summary["notes"].append(
            "width-int64 行通信量为 0（该配置下算子根本不通信），无法比较位宽"
        )
    else:
        summary["notes"].append("width-int64 行没有通信量，无法算位宽比")

    # ---- 实验 B：规模（线性性 + 每元素字节） ------------------------------
    per_element_bytes: dict[str, float] = {}
    for name, count in (
        ("scale-512", elements("scale-512")),
        ("scale-1024", elements("scale-1024")),
        ("width-int64", elements("width-int64")),
    ):
        value = comm(name)
        if value and count:  # 0 字节 = 该配置不通信，不是"单价 0"
            per_element_bytes[str(count)] = value / count
    if per_element_bytes:
        values = list(per_element_bytes.values())
        spread = (max(values) - min(values)) / max(values) if max(values) else None
        summary["linearity"] = {
            "bytes_per_element_by_scale": per_element_bytes,
            "relative_spread": spread,
            "linear": spread is not None and spread <= _RATIO_TOLERANCE,
        }
    elif any(comm(name) == 0 for name in ("scale-512", "scale-1024", "width-int64")):
        summary["notes"].append(
            "规模扫描的通信量为 0（该配置下算子根本不通信），算不出每元素字节"
        )
    else:
        summary["notes"].append("规模扫描没有通信量，无法算每元素字节")

    # ---- 打包上界（由实测 B/元素 换算，不是另跑一遍） ---------------------
    pointwise = base
    per_element = per_element_bytes.get(str(_WIDTH_SWEEP_ELEMENTS))
    if pointwise and per_element:
        packed_elements = _WIDTH_SWEEP_ELEMENTS // PACKED_VALUES_PER_ELEMENT
        predicted = per_element * packed_elements
        measured_packed = comm("scale-512")
        summary["packing_upper_bound"] = {
            "values_per_element_packed": PACKED_VALUES_PER_ELEMENT,
            "element_ratio": float(PACKED_VALUES_PER_ELEMENT),
            "bytes_per_element": per_element,
            "comm_pointwise_bytes": pointwise,
            "comm_packed_predicted_bytes": predicted,
            "comm_packed_measured_bytes": measured_packed,
            "prediction_matches_measurement": (
                (abs(measured_packed - predicted) <= max(1.0, predicted * _RATIO_TOLERANCE))
                if measured_packed is not None
                else None
            ),
            "comm_ratio_upper": _ratio(pointwise, measured_packed if measured_packed else predicted),
            "values_compared": (
                f"{_WIDTH_SWEEP_ELEMENTS} 个 8 位值："
                f"逐点 {_WIDTH_SWEEP_ELEMENTS} 元素 vs 打包 8 值/元素 {packed_elements} 元素"
            ),
            "derivation": "上界 = 实测 B/元素 × 打包后元素数；不含槽内归约的通信量",
            "caveat": (
                "这是**上界**：跑的是逐元素秘密乘法，不是打包电路；"
                "槽内归约本身的通信量未测，真实收益只可能更低"
            ),
        }
    else:
        summary["notes"].append("缺少逐点行的通信量，无法算打包上界")
    return summary


# --------------------------------------------------------------------------
# 产物
# --------------------------------------------------------------------------

PACKING_CSV_COLUMNS: tuple[str, ...] = (
    "case",
    "name",
    "op",
    "protocol",
    "field",
    "dtype",
    "elements",
    "repeat",
    "status",
    "wall_ms",
    "peak_rss_mb",
    "pphlo_bytes",
    "profiled",
    "comm_send_bytes",
    "comm_recv_bytes",
    "comm_total_bytes",
    "comm_per_element",
    "within_tolerance",
    "comm_by_primitive",
    "max_abs_error",
    "tolerance",
    "note",
    "error",
)


def write_packing_probe_json(records: Sequence[Mapping[str, Any]], path: str) -> str:
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


def write_packing_probe_csv(records: Sequence[Mapping[str, Any]], path: str) -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PACKING_CSV_COLUMNS)
        for record in records:
            writer.writerow(
                [_csv_value(record.get(column)) for column in PACKING_CSV_COLUMNS]
            )
    return path


def format_packing_summary(records: Sequence[Mapping[str, Any]]) -> str:
    """终端可读摘要表 + 结论。"""

    header = (
        f"{'case':26s} {'status':12s} {'N':>6s} {'comm_B':>10s} "
        f"{'B/元素':>8s} {'wall_ms':>9s}"
    )
    lines = [header]
    for record in records:
        comm = record.get("comm_total_bytes")
        per = record.get("comm_per_element")
        wall = record.get("wall_ms")
        lines.append(
            f"{str(record.get('case', ''))[:26]:26s} "
            f"{str(record.get('status', '')):12s} "
            f"{str(record.get('elements', '-')):>6s} "
            f"{(f'{comm:.0f}' if isinstance(comm, (int, float)) else '-'):>10s} "
            f"{(f'{per:.2f}' if isinstance(per, (int, float)) else '-'):>8s} "
            f"{(f'{wall:.1f}' if isinstance(wall, (int, float)) else '-'):>9s}"
        )

    summary = summarize_packing(records)
    if summary["width_effect"]:
        lines.append("")
        lines.append(f"位宽结论：{summary['width_effect']}")
    linearity = summary.get("linearity")
    if linearity:
        lines.append(
            "规模结论：B/元素 = "
            + "、".join(
                f"N={key} 时 {value:.2f}" for key, value in linearity["bytes_per_element_by_scale"].items()
            )
            + f"（线性：{linearity['linear']}）"
        )
    bound = summary.get("packing_upper_bound")
    if bound:
        lines.append(
            f"打包上界：8 值/元素 → 元素数 ÷{bound['element_ratio']:.0f}，"
            f"通信量 {bound['comm_pointwise_bytes']:.0f} → "
            f"{bound['comm_packed_measured_bytes'] if bound['comm_packed_measured_bytes'] is not None else bound['comm_packed_predicted_bytes']:.0f} B"
            f"（上界，未含槽内归约）"
        )
    for note in summary.get("notes") or ():
        lines.append(f"说明：{note}")
    return "\n".join(lines)


__all__ = [
    "PACKING_CSV_COLUMNS",
    "PACKED_VALUES_PER_ELEMENT",
    "PROBE_OP",
    "PackingProbeCase",
    "format_packing_summary",
    "packing_probe_cases",
    "run_packing_probe_case",
    "run_packing_probe_cases",
    "summarize_packing",
    "write_packing_probe_csv",
    "write_packing_probe_json",
]
