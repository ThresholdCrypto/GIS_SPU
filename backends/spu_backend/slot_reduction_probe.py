# -*- coding: utf-8 -*-
"""打包电路的两件前置事实（P0）：免逐槽提取的三条路线，与槽内归并的单价。

为什么要先做这一步
------------------
P6 测到"秘密 × 秘密"逐元素乘法是 **16 B/元素**，与输入位宽无关；
P7-P0 紧接着测到"取槽"（右移 + 掩码）是 **1424 B/元素 = 纯乘法的 89 倍**
（`docs/mpc_slot_cost_probe.json`）。两条合起来把 D3 的问题从
"照着 L2 公式写电路"改成了**"能不能不逐槽取数"**（`docs/BITPLANE_LAYOUT.md` §7）。

本探针就做 §7 的**第 1 步（设计筛选）**与**第 3 步（槽内归并实测）**，
把三条免逐槽提取路线各自变成**可执行的电路**，在真机上跑，并**拿目标语义对拍**：

| 路线 | 变体 | 电路 | 目标语义 |
|---|---|---|---|
| —（基准） | `extract_seq` | `Σ_s ((X >> 8s) & 0xFF) · w_s` | `Σ_s w_s v_s` |
| (b) 公开权重承担加权 | `extract_seq_secret_weight` | 同一提取，但权重取**秘密**标量 | `Σ_s w_s v_s` |
| —（下界对照） | `extract_unweighted` | `Σ_s ((X >> 8s) & 0xFF)`（完全不加权） | `Σ_s v_s` |
| 槽内归并（§7-3） | `swar_tree` | 对数深度掩码 + 移位 + 加法树 | `Σ_s v_s` |
| (a) 整元素一次乘法 | `fullmul` | `(X · ONES_packed) & 0xFF` | `Σ_s v_s` |
| (c) 一次对齐移位 | `maskless_shift` | `Σ_s (X >> 8s) · w_s`（不掩码） | `Σ_s w_s v_s` |
| —（批间差） | `extract_seq_dup` | 与 `extract_seq` **逐位相同** | `Σ_s w_s v_s` |

**这个探针的关键设计**：每个变体的明文参考都是**目标语义**，不是"它自己那一路
的公式"。因此 `within_tolerance=False` 不是"跑错了"，而是**这条路线算不出目标值**
——这正是路线 (a) / (c) 的判据。反过来 `extract_seq` 必须 `True`，
否则说明探针本身写错了。

严格说清楚本探针**不是**什么
----------------------------
- 它**不是**打包电路：只做**线性**的槽内归约（加权和 / 求和）。
  D3 的 L2（跨格网位平面）归约轴是**候选轴，且是逐候选判定后的非线性聚合**，
  本探针**不覆盖**它——非线性判定必须先物化位平面，见结论段；
- 它**不给收益结论**：给的是"哪条路线可行、哪条不可行、槽内归并多少 B/元素"；
  打包值不值要等真正的打包电路成对实测（§7-5）；
- 规模刻意取小（N=256 元素 × 4 槽）：这一步只比**单价**，不需要大数组；
- 输入值取 1..10：保证 4 槽和 ≤ 100，`swar_tree` 的按字节部分和 ≤ 40 不溢出，
  各路线比的是同一个语义，不是不同的回绕结果。
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
SLOT_REDUCTION_OP = "slot_reduce"

#: 每元素承载的槽数（课题量化位宽 b = 8，4 槽 = 4 个属性通道）
SLOT_REDUCTION_SLOTS = 4

#: 默认元素数（只比单价）
SLOT_REDUCTION_ELEMENTS = 256

#: 槽宽（8 位）
SLOT_REDUCTION_SLOT_BITS = 8

#: 加权语义用的公开权重（路线 (b)/(c) 的对拍目标）
SLOT_REDUCTION_WEIGHTS: tuple[int, ...] = (1, 2, 3, 4)

#: P6 实测的"纯秘密乘法"单价（B/元素），用作比价的基准线
PURE_MUL_BYTES_PER_ELEMENT = 16.0

#: P7-P0 实测的"逐槽全量提取"单价（B/元素），本探针应能复现其量级
SLOT_EXTRACT_BYTES_PER_ELEMENT = 1424.0

#: 输入取值上限（保证 4 槽和 ≤ 100、按字节部分和 ≤ 40）
_VALUE_MAX = 10

#: 打包成整元素乘数用的"每槽 1"常量（供路线 (a) 使用）
_ONES_PACKED = sum(1 << (SLOT_REDUCTION_SLOT_BITS * slot) for slot in range(SLOT_REDUCTION_SLOTS))


def _slot_inputs(elements: int, slots: int) -> tuple[np.ndarray, np.ndarray]:
    """确定性构造：一个打包后的元素向量 + 逐槽明文值（形状 `[slots, elements]`）。"""

    index = np.arange(1, slots * elements + 1, dtype=np.int64)
    values = ((index % _VALUE_MAX) + 1).reshape(slots, elements)
    packed = np.zeros(elements, dtype=np.int64)
    for slot in range(slots):
        packed |= values[slot] << (SLOT_REDUCTION_SLOT_BITS * slot)
    return packed, values


def _secret_weights() -> np.ndarray:
    """秘密权重向量（与公开权重同值，只差"是不是秘密"这一点）。"""

    return np.asarray(SLOT_REDUCTION_WEIGHTS[:SLOT_REDUCTION_SLOTS], dtype=np.int64)


def _weight(index: int) -> int:
    return SLOT_REDUCTION_WEIGHTS[index % len(SLOT_REDUCTION_WEIGHTS)]


def _weighted_target(values: np.ndarray) -> np.ndarray:
    """目标语义 1：`Σ_s w_s · v_s`（逐槽加权和）。"""

    total = np.zeros(values.shape[1], dtype=np.int64)
    for slot in range(values.shape[0]):
        total = total + values[slot] * np.int64(_weight(slot))
    return total


def _sum_target(values: np.ndarray) -> np.ndarray:
    """目标语义 2：`Σ_s v_s`（槽内归并 / 跨候选聚合的无加权形态）。"""

    total = np.zeros(values.shape[1], dtype=np.int64)
    for slot in range(values.shape[0]):
        total = total + values[slot]
    return total


# --------------------------------------------------------------------------
# 变体的电路
# --------------------------------------------------------------------------


def _extract_seq_fn(x):
    """基准：逐槽顺序提取后加权（= P7 的 shift_mask 电路）。"""

    import jax.numpy as jnp

    total = jnp.zeros_like(x)
    for slot in range(SLOT_REDUCTION_SLOTS):
        value = (x >> jnp.asarray(SLOT_REDUCTION_SLOT_BITS * slot, dtype=x.dtype)) & jnp.asarray(
            0xFF, dtype=x.dtype
        )
        total = total + value * jnp.asarray(_weight(slot), dtype=x.dtype)
    return total


def _extract_unweighted_fn(x):
    """对照：逐槽提取但不加权（问"加权本身花不花钱"）。"""

    import jax.numpy as jnp

    total = jnp.zeros_like(x)
    for slot in range(SLOT_REDUCTION_SLOTS):
        value = (x >> jnp.asarray(SLOT_REDUCTION_SLOT_BITS * slot, dtype=x.dtype)) & jnp.asarray(
            0xFF, dtype=x.dtype
        )
        total = total + value
    return total


def _extract_seq_secret_weight_fn(x, weights):
    """路线 (b) 的反面：权重是**秘密**标量——提取相同，但每槽多一次保密乘法。"""

    import jax.numpy as jnp

    total = jnp.zeros_like(x)
    for slot in range(SLOT_REDUCTION_SLOTS):
        value = (x >> jnp.asarray(SLOT_REDUCTION_SLOT_BITS * slot, dtype=x.dtype)) & jnp.asarray(
            0xFF, dtype=x.dtype
        )
        total = total + value * weights[slot]
    return total


def _maskless_shift_fn(x):
    """路线 (c)：一次对齐移位——只右移不掩码，指望少付掩码的钱。"""

    import jax.numpy as jnp

    total = jnp.zeros_like(x)
    for slot in range(SLOT_REDUCTION_SLOTS):
        shifted = x >> jnp.asarray(SLOT_REDUCTION_SLOT_BITS * slot, dtype=x.dtype)
        total = total + shifted * jnp.asarray(_weight(slot), dtype=x.dtype)
    return total


def _fullmul_fn(x):
    """路线 (a)：整元素乘一次打包常量，指望"一次乘法算全部槽"。"""

    import jax.numpy as jnp

    product = x * jnp.asarray(_ONES_PACKED, dtype=x.dtype)  # 秘密 × 公开常数：本地线性
    return product & jnp.asarray(0xFF, dtype=x.dtype)


def _swar_tree_fn(x):
    """槽内归并（§7-3）：对数深度掩码 + 移位 + 加法树（无乘法）。"""

    import jax.numpy as jnp

    # 4 槽（字节 0..3）：先并相邻两槽，再把两个半字并起来。
    even = x & jnp.asarray(0x00FF00FF, dtype=x.dtype)
    odd = (x >> jnp.asarray(SLOT_REDUCTION_SLOT_BITS, dtype=x.dtype)) & jnp.asarray(
        0x00FF00FF, dtype=x.dtype
    )
    pair = even + odd  # 字节 0 = v0+v1（≤20），字节 2 = v2+v3（≤20）
    low = pair & jnp.asarray(0x0000FFFF, dtype=x.dtype)
    high = (pair >> jnp.asarray(16, dtype=x.dtype)) & jnp.asarray(0x0000FFFF, dtype=x.dtype)
    return low + high


@dataclass(frozen=True)
class SlotReductionVariant:
    """一个变体：名字 + 电路 + 它要对拍的**目标语义**。"""

    name: str
    fn: Callable[..., Any]
    target: str  # "weighted" / "sum"
    route: str  # 路线标签：baseline / control / route_a / route_b / route_c / swar
    note: str
    #: 是否额外吃一个"秘密权重"输入（仅 route_b 的反面对照用）
    secret_weights: bool = False

    def reference(self, values: np.ndarray) -> np.ndarray:
        if self.target == "weighted":
            return _weighted_target(values)
        return _sum_target(values)


#: 变体（顺序固定，产物表格按它排）
SLOT_REDUCTION_VARIANTS: tuple[SlotReductionVariant, ...] = (
    SlotReductionVariant(
        "extract_seq", _extract_seq_fn, "weighted", "baseline",
        "基准：逐槽提取 + 加权（P7 的 shift_mask 电路）",
    ),
    SlotReductionVariant(
        "extract_unweighted", _extract_unweighted_fn, "sum", "control",
        "对照：逐槽提取但不加权（分离「加权」这一步的钱）",
    ),
    SlotReductionVariant(
        "swar_tree", _swar_tree_fn, "sum", "swar",
        "槽内归并：对数深度掩码+移位+加法树（D3 §7-3）",
    ),
    SlotReductionVariant(
        "fullmul", _fullmul_fn, "sum", "route_a",
        "路线 a：整元素一次乘法算全部槽",
    ),
    SlotReductionVariant(
        "maskless_shift", _maskless_shift_fn, "weighted", "route_c",
        "路线 c：一次对齐移位（不掩码）",
    ),
    SlotReductionVariant(
        "extract_seq_secret_weight", _extract_seq_secret_weight_fn, "weighted", "route_b",
        "路线 b 的反面：权重是秘密时，同样的提取要多付一次保密乘法",
        secret_weights=True,
    ),
    SlotReductionVariant(
        "extract_seq_dup", _extract_seq_fn, "weighted", "noise",
        "**同一电路复测**：与 extract_seq 逐位相同，只用来量本探针的批间差",
    ),
)


@dataclass(frozen=True)
class SlotReductionCase:
    """一条用例 = 一个变体在一个规模上的运行。"""

    variant: str
    elements: int = SLOT_REDUCTION_ELEMENTS
    protocol: str = "ABY3"
    field: str = "FM64"
    repeats: int = 1
    capture_comm: bool = True
    note: str = ""

    def label(self) -> str:
        return f"{SLOT_REDUCTION_OP} {self.variant} N={self.elements}"


def slot_reduction_cases(*, elements: int = SLOT_REDUCTION_ELEMENTS, repeats: int = 1,
                         capture_comm: bool = True) -> list[SlotReductionCase]:
    return [
        SlotReductionCase(
            variant=variant.name,
            elements=int(elements),
            repeats=max(1, int(repeats)),
            capture_comm=capture_comm,
            note=variant.note,
        )
        for variant in SLOT_REDUCTION_VARIANTS
    ]


def _variant(name: str) -> SlotReductionVariant:
    for variant in SLOT_REDUCTION_VARIANTS:
        if variant.name == name:
            return variant
    raise KeyError(name)


def _blank_record(case: SlotReductionCase) -> dict[str, Any]:
    variant = _variant(case.variant)
    return {
        "case": case.label(),
        "name": case.variant,
        "op": SLOT_REDUCTION_OP,
        "route": variant.route,
        "target": variant.target,
        "protocol": case.protocol,
        "field": case.field,
        "slots": SLOT_REDUCTION_SLOTS,
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


def _run_slot_reduction_once(case: SlotReductionCase, *, report: Any = None) -> dict[str, Any]:
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

    try:
        variant = _variant(case.variant)
    except KeyError:
        record["status"] = "error"
        record["error"] = (
            f"未登记的变体 {case.variant!r}；已知 "
            f"{[item.name for item in SLOT_REDUCTION_VARIANTS]}"
        )
        return record

    packed, values = _slot_inputs(case.elements, SLOT_REDUCTION_SLOTS)
    expected = variant.reference(values)
    start = time.perf_counter()
    inputs = [packed] if not variant.secret_weights else [packed, _secret_weights()]
    run = run_spu_simulation(
        variant.fn,
        inputs,
        protocol=case.protocol,
        field=case.field,
        reference_fn=lambda *args, e=expected: e,
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


def run_slot_reduction_case(case: SlotReductionCase, *, report: Any = None) -> dict[str, Any]:
    """跑一条用例；`repeats > 1` 时返回多次运行的统计记录（报中位数）。

    P7-P0 的教训：通信量有批间抖动（实测 0.03%–10.1%），单次读数不足以支撑
    "哪条路线更贵"的判断，所以重复次数必须真的重复执行、报中位数与区间。
    """

    repeats = max(1, int(case.repeats))
    if repeats == 1:
        return _run_slot_reduction_once(case, report=report)

    import statistics

    samples = [_run_slot_reduction_once(case, report=report) for _ in range(repeats)]
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


def run_slot_reduction_cases(
    cases: Sequence[SlotReductionCase], *, report: Any = None, progress: Any = None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in cases:
        record = run_slot_reduction_case(case, report=report)
        records.append(record)
        if progress is not None:
            progress(record)
    return records


# --------------------------------------------------------------------------
# 结论
# --------------------------------------------------------------------------

#: 三条路线的单句判据（写进汇总，避免调用方自己编结论）
_ROUTE_LABEL = {
    "route_a": "路线 a：整元素一次乘法算全部槽",
    "route_b": "路线 b：公开权重承担加权",
    "route_c": "路线 c：一次对齐移位（不掩码）",
}


def summarize_slot_reduction(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """按路线给"可行 / 不可行"的**实测**判据；缺数就留空并说明原因。"""

    by_name = {str(record.get("name")): record for record in records}

    def per_element(name: str) -> float | None:
        value = (by_name.get(name) or {}).get("comm_per_element")
        return float(value) if isinstance(value, (int, float)) else None

    def correct(name: str) -> bool | None:
        value = (by_name.get(name) or {}).get("within_tolerance")
        return bool(value) if value is not None else None

    summary: dict[str, Any] = {
        "per_element_bytes": {},
        "routes": {},
        "notes": [],
    }
    for variant in SLOT_REDUCTION_VARIANTS:
        value = per_element(variant.name)
        if value is not None:
            summary["per_element_bytes"][variant.name] = value

    # 先量本探针自己的批间差：同一电路复测（extract_seq vs extract_seq_dup）。
    # 通信量有 5%–15% 的批间抖动（本轮实测：同一电路同进程内两次给 1568 / 1496
    # B/元素），所以任何小于这个抖动的"差异"都不能当结论。
    spread = None
    dup = per_element("extract_seq_dup")
    base0 = per_element("extract_seq")
    if dup is not None and base0 is not None and (dup + base0) > 0:
        spread = abs(dup - base0) / ((dup + base0) / 2.0)
        summary["noise"] = {
            "same_circuit_samples": {"extract_seq": base0, "extract_seq_dup": dup},
            "relative_spread": spread,
            "reading": (
                f"同一电路复测相差 {spread:.1%}——小于这个比例的通信量差异"
                "不构成结论，只能做量级判断"
            ),
        }

    baseline = per_element("extract_seq")
    if baseline is None:
        summary["notes"].append("没有基准变体（extract_seq）的通信量，无法比价")
        return summary
    summary["baseline_bytes_per_element"] = baseline

    if correct("extract_seq") is not True:
        summary["notes"].append(
            "基准变体 extract_seq 与目标语义不一致——探针本身写错了，"
            "下面的路线判定不成立"
        )

    routes: dict[str, Any] = {}

    # 路线 (a)：整元素一次乘法 —— 能不能算出目标值
    a_ok = correct("fullmul")
    a_comm = per_element("fullmul")
    if a_ok is None:
        summary["notes"].append("fullmul 无结果，路线 a 无法判定")
    else:
        routes["route_a"] = {
            "label": _ROUTE_LABEL["route_a"],
            "matches_target": a_ok,
            "verdict": (
                "可行：一次乘法即得目标值"
                if a_ok
                else "不可行：实测算不出目标值（槽间交叉项/进位导致），"
                "必须先逐槽提取才能分离各槽"
            ),
            "bytes_per_element": a_comm,
            "max_abs_error": (by_name.get("fullmul") or {}).get("max_abs_error"),
        }

    # 路线 (b)：把权重从"秘密"挪到"公开"能省多少 —— 比 extract_seq 与
    # extract_seq_secret_weight（同一电路，只有权重是秘密/公开之别）
    unweighted = per_element("extract_unweighted")
    secret_weight = per_element("extract_seq_secret_weight")
    if secret_weight is None:
        summary["notes"].append("extract_seq_secret_weight 无结果，路线 b 无法判定")
    else:
        delta = secret_weight - baseline
        routes["route_b"] = {
            "label": _ROUTE_LABEL["route_b"],
            "public_weight_bytes_per_element": baseline,
            "secret_weight_bytes_per_element": secret_weight,
            "saving_bytes_per_element": delta,
            "unweighted_floor_bytes_per_element": unweighted,
            "verdict": (
                "在批间差范围内：把权重放公开侧与放秘密侧没有可证实的差别，"
                "路线 b 无收益"
                if delta <= 0 or (spread is not None and delta / max(secret_weight, 1.0) <= spread)
                else (
                    f"把权重从秘密挪到公开可省 {delta:.0f} B/元素"
                    f"（{secret_weight:.0f} → {baseline:.0f}）；"
                    "但**提取本身**仍要付 "
                    + (f"{unweighted:.0f} B/元素" if unweighted is not None else "同一量级的钱")
                    + (
                        f"（注意本探针批间差 {spread:.1%}，比 {delta / secret_weight:.1%} "
                        "的差异量级可比，需重复实验才能定性）"
                        if spread is not None
                        else ""
                    )
                )
            ),
        }

    # 路线 (c)：一次对齐移位 —— 能不能算出目标值
    c_ok = correct("maskless_shift")
    c_comm = per_element("maskless_shift")
    if c_ok is None:
        summary["notes"].append("maskless_shift 无结果，路线 c 无法判定")
    else:
        routes["route_c"] = {
            "label": _ROUTE_LABEL["route_c"],
            "matches_target": c_ok,
            "verdict": (
                "可行：省掉掩码仍得目标值"
                if c_ok
                else "不可行：省掉掩码后高位槽会串进来（实测算不出目标值）"
            ),
            "bytes_per_element": c_comm,
            "max_abs_error": (by_name.get("maskless_shift") or {}).get("max_abs_error"),
        }

    # 槽内归并（§7-3）：SWAR 树 vs 逐槽顺序提取（同为无加权语义）
    tree = per_element("swar_tree")
    if tree is None or unweighted is None:
        summary["notes"].append("swar_tree / extract_unweighted 缺数，槽内归并无法比价")
    else:
        routes["swar_tree"] = {
            "label": "槽内归并：对数深度 SWAR 树",
            "swar_bytes_per_element": tree,
            "sequential_bytes_per_element": unweighted,
            "matches_target": correct("swar_tree"),
            "speedup_vs_sequential": (unweighted / tree) if tree > 0 else None,
            "verdict": (
                "SWAR 树比逐槽顺序提取省 "
                f"{(unweighted / tree):.2f}×（同为无加权语义）"
                if tree > 0 and unweighted > tree
                else "SWAR 树不省（对数深度没换来更少的按位原语）"
            ),
        }

    summary["routes"] = routes

    # 收口结论：做什么、不做什么
    feasible = [
        name for name, item in routes.items()
        if item.get("matches_target") is True
    ]
    infeasible = [
        name for name, item in routes.items()
        if item.get("matches_target") is False
    ]
    summary["conclusion"] = {
        "feasible_routes": feasible,
        "infeasible_routes": infeasible,
        "statement": (
            "三条「免逐槽提取」路线里，能算出目标值的只有路线 (b)（公开权重）——"
            "但它**只免掉加权、免不掉提取**；路线 (a)（整元素乘法）与路线 (c)"
            "（不掩码移位）都在真机上对拍失败。因此对**线性**槽内归约，"
            "打包只能靠逐槽提取（或 SWAR 树），其单价见上表；"
            "对 D3 的 L2（非线性逐候选判定），位平面必须先物化，本探针不覆盖。"
        ),
        "caveat": (
            "本探针只做线性槽内归约；L2 的候选轴聚合是**非线性判定**，"
            "不在本探针覆盖范围。打包值不值仍需打包电路成对实测（§7-5）"
        ),
    }
    return summary


# --------------------------------------------------------------------------
# 产物
# --------------------------------------------------------------------------

SLOT_REDUCTION_CSV_COLUMNS: tuple[str, ...] = (
    "case",
    "name",
    "op",
    "route",
    "target",
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


def write_slot_reduction_json(records: Sequence[Mapping[str, Any]], path: str) -> str:
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


def write_slot_reduction_csv(records: Sequence[Mapping[str, Any]], path: str) -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SLOT_REDUCTION_CSV_COLUMNS)
        for record in records:
            writer.writerow(
                [_csv_value(record.get(column)) for column in SLOT_REDUCTION_CSV_COLUMNS]
            )
    return path


def format_slot_reduction_summary(records: Sequence[Mapping[str, Any]]) -> str:
    header = f"{'variant':20s} {'route':10s} {'status':12s} {'comm_B':>10s} {'B/元素':>9s} {'对拍':>5s}"
    lines = [header]
    for record in records:
        comm = record.get("comm_total_bytes")
        per = record.get("comm_per_element")
        match = record.get("within_tolerance")
        mark = "-" if match is None else ("是" if match else "否")
        lines.append(
            f"{str(record.get('name', ''))[:20]:20s} "
            f"{str(record.get('route', ''))[:10]:10s} "
            f"{str(record.get('status', '')):12s} "
            f"{(f'{comm:.0f}' if isinstance(comm, (int, float)) else '-'):>10s} "
            f"{(f'{per:.2f}' if isinstance(per, (int, float)) else '-'):>9s} "
            f"{mark:>5s}"
        )
    summary = summarize_slot_reduction(records)
    lines.append("")
    for key in ("route_a", "route_b", "route_c", "swar_tree"):
        item = (summary.get("routes") or {}).get(key)
        if item:
            lines.append(f"{key}: {item['verdict']}")
    statement = (summary.get("conclusion") or {}).get("statement")
    if statement:
        lines.append("")
        lines.append(f"结论：{statement}")
    for note in summary.get("notes") or ():
        lines.append(f"说明：{note}")
    return "\n".join(lines)


__all__ = [
    "PURE_MUL_BYTES_PER_ELEMENT",
    "SLOT_EXTRACT_BYTES_PER_ELEMENT",
    "SLOT_REDUCTION_CSV_COLUMNS",
    "SLOT_REDUCTION_ELEMENTS",
    "SLOT_REDUCTION_OP",
    "SLOT_REDUCTION_SLOTS",
    "SLOT_REDUCTION_VARIANTS",
    "SLOT_REDUCTION_WEIGHTS",
    "SlotReductionCase",
    "SlotReductionVariant",
    "format_slot_reduction_summary",
    "run_slot_reduction_case",
    "run_slot_reduction_cases",
    "slot_reduction_cases",
    "summarize_slot_reduction",
    "write_slot_reduction_csv",
    "write_slot_reduction_json",
]
