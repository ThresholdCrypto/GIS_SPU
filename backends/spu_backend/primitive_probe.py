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

为什么 `B/元素` 这一个数不够用（本轮踩过的坑）
------------------------------------------
第一版把每条电路的通信量都除以**输入元素数**，于是 `dot`（长度 N 的收缩，输出 1 个标量）
得到 `16 / N` —— N=64 时是 **0.25 B/元素**，看着像"dot 比乘法便宜 64 倍"，
甚至像"这条路径没在真做保密乘法"。

后续把 `pphlo` 追踪打开后，真相是**归一化选错了维度**：

```
%2 = pphlo.dot %0, %1 : (tensor<1x64x!pphlo.secret<i32>>, tensor<64x1x!pphlo.secret<i32>>)
                        -> tensor<1x1x!pphlo.secret<i32>>           # 操作数/结果都是 secret
pphlo.dot, executed 1 times, send bytes 8 recv bytes 8        # 1 轮、1 个环元素
```

真机实测（ABY3 / FM64，`outputs` 是输出元素数、`K` 是收缩长度）：

| 形式 | K | outputs | 通信量 | B/输出 |
|---|---|---|---|---|
| 1-D `dot` | 64 / 256 / 1024 / 4096 | 1 | 16 B（恒定） | 16.0 |
| matmul | 64 | 16 | 256 B | 16.0 |
| matmul | 64 | 256 | 4096 B | 16.0 |
| matmul | 64 | 1024 | 16384 B | 16.0 |
| matmul | 256 / 1024 | 256 | 4096 B（恒定） | 16.0 |
| 归约 `sum` | 64 / 512 | 1 | 16 B（恒定） | 16.0 |
| 逐元素 `mul`（对照） | — | 64 | 1024 B | 16.0 |

所以：**收缩类算子与逐元素乘法的单价相同（16 B/计费元素），差别只在"哪个维度计费"**——
`dot` / `sum` 按**输出个数**计费，**收缩长度不计费**。单价 16 B 与 P6 的环元素单价一致。

本探针把这些形态固化成五条用例（`dot` / `dot_long` / `matmul` / `sum` / `sum_long`，
2026-10-08 `--repeat 3` 中位数）：收缩长度 64 与 512 的两组读数**逐字节相同**（各 16 B），
`matmul` 4096 B = 256 输出 × 16 B。`summarize_primitives` 用 `CONTRACTION_PAIRS` +
`contraction_is_free` 把这条结论固定住：**两条读数不一致就报"存疑"**，不许默认成立。

因此本探针给每条用例标一个 `cost_driver`：

- `inputs`：通信量随**输入元素数**走（逐元素运算），单位价 = `comm / 输入元素数`；
- `outputs`：通信量随**输出个数**走（收缩/矩阵乘），单位价 = `comm / 输出个数`。

一律再除以输入元素数会把"收缩长度免费"误读成"这个原语便宜"——
这正是第一版把 `dot` 标成可疑的原因，也是**本探针留下来的教训**
（与 P6 那条 `x * 2` 实测 0 B 的教训同族：读数本身没错，**口径错了**）。

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


def _fn_matmul(x, y):
    import jax.numpy as jnp

    return jnp.matmul(x, y)


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
    #: 输出元素数（恒等/逐元素电路 = elements；收缩电路 = 输出个数）
    outputs: int | None = None
    #: 通信量按哪个维度计费："inputs"（逐元素）或 "outputs"（收缩/矩阵乘）
    cost_driver: str = "inputs"
    #: 规模扫描类用例自带规模，不跟随全局 `elements` 覆盖
    fixed_elements: bool = False
    #: 自定义输入构造（矩阵用例需要二维输入）
    inputs_fn: Callable[[int], tuple[np.ndarray, np.ndarray]] | None = None

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


def _case(name: str, fn, needed_by: str, note: str = "", **kwargs) -> PrimitiveCase:
    return PrimitiveCase(
        name=name, fn=fn, reference_fn=_ref(fn), needed_by=needed_by, note=note, **kwargs
    )


#: 矩阵用例的输入构造（M×K 与 K×P）
_MATMUL_M = 16
_MATMUL_P = 16
_MATMUL_K = 64
#: 矩阵乘的秘密**输入**元素数（A: M×K + B: K×P），与输出个数区分开
_MATMUL_ELEMENTS = _MATMUL_M * _MATMUL_K + _MATMUL_K * _MATMUL_P
_DOT_LONG_ELEMENTS = 512


def _matmul_inputs(_elements: int) -> tuple[np.ndarray, np.ndarray]:
    shape_a = (_MATMUL_M, _MATMUL_K)
    shape_b = (_MATMUL_K, _MATMUL_P)
    a = (((np.arange(1, _MATMUL_M * _MATMUL_K + 1, dtype=np.int64) % 9) + 1).reshape(shape_a))
    b = ((((np.arange(1, _MATMUL_K * _MATMUL_P + 1, dtype=np.int64) * 3) % 9) + 1).reshape(shape_b))
    return a, b


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
    _case("dot", _fn_dot, "packing", "打包归约候选（D3 §7 点名）；**收缩长度不计费**，按输出计",
          outputs=1, cost_driver="outputs"),
    _case("dot_long", _fn_dot, "packing",
          f"同 dot，收缩长度 {_DOT_LONG_ELEMENTS}（与 dot 同输出=1）：通信量应**不变**",
          elements=_DOT_LONG_ELEMENTS, fixed_elements=True, outputs=1, cost_driver="outputs"),
    _case("matmul", _fn_matmul, "packing",
          f"矩阵乘 {_MATMUL_M}x{_MATMUL_K} · {_MATMUL_K}x{_MATMUL_P}：通信量应随**输出个数**走",
          elements=_MATMUL_ELEMENTS, fixed_elements=True,
          outputs=_MATMUL_M * _MATMUL_P, cost_driver="outputs", inputs_fn=_matmul_inputs),
    _case("sum", _fn_sum, "layout", "跨元素求和（归约轴聚合的基元）；与 dot 同族：按**输出个数**计",
          outputs=1, cost_driver="outputs"),
    _case("sum_long", _fn_sum, "layout",
          f"同 sum，收缩长度 {_DOT_LONG_ELEMENTS}（与 sum 同输出=1）：通信量应**不变**",
          elements=_DOT_LONG_ELEMENTS, fixed_elements=True, outputs=1, cost_driver="outputs"),
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
                elements=case.elements if case.fixed_elements else int(elements),
                protocol=case.protocol,
                field=case.field,
                repeats=max(1, int(repeats)),
                capture_comm=capture_comm,
                outputs=case.outputs,
                cost_driver=case.cost_driver,
                fixed_elements=case.fixed_elements,
                inputs_fn=case.inputs_fn,
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
        "outputs": case.outputs,
        "cost_driver": case.cost_driver,
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
        "comm_per_output": None,
        "cost_per_unit": None,
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

    inputs_fn = case.inputs_fn or _probe_inputs
    x, y = inputs_fn(case.elements)
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
    record["comm_per_element"] = None
    record["comm_per_output"] = None
    record["cost_per_unit"] = None
    if isinstance(comm, (int, float)):
        if case.elements > 0:
            record["comm_per_element"] = comm / case.elements
        if case.outputs:
            record["comm_per_output"] = comm / case.outputs
        # 计费维度必须与电路形态对齐：收缩/矩阵乘按**输出个数**计，
        # 逐元素按**输入元素数**计。一律除以输入元素数会把"收缩长度免费"
        # 误读成"这个原语便宜"——`dot` 的 0.25 B/元素 就是这么来的。
        divisor = case.outputs if case.cost_driver == "outputs" else case.elements
        if divisor:
            record["cost_per_unit"] = comm / divisor
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
        "comm_per_output",
        "cost_per_unit",
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


#: 每条电路在每个**计费单位**上至少要付的秘密乘法次数，用于跟实测通信量对账
#: （P6 实测 16 B/环元素/次乘法）。低于下限的读数物理上不可能，必须标可疑，
#: 不能当成"这个原语免费"——P6 的 `x * 2` 实测 0 B 就是这个坑。
#:
#: **计费单位随电路形态走**：逐元素电路 = 输入元素数；收缩/矩阵乘 = 输出个数
#: （`dot` 的收缩长度免费，见模块 docstring）。记录里的 `cost_driver` 决定用哪个，
#: 一律除以输入元素数会把 `dot` 误判成"低于下限"——读数没错，尺子错了。
MULTIPLY_DEMAND: Mapping[str, float] = {
    "mul": 1.0,  # 逐元素：1 次秘密乘法 / 输入元素
    "dot": 1.0,  # 收缩：至少 1 次秘密乘法 / 输出
    "dot_long": 1.0,
    "matmul": 1.0,
}


#: "收缩长度应免费"的成对用例：同一原语、同一输出个数、**仅收缩长度不同**。
#: 两条读数一致才能说明通信量按输出个数走（见模块 docstring 的实测表）；
#: 不一致就说明"收缩长度免费"这条结论在该原语上不成立，必须如实报出来。
CONTRACTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("dot", "dot_long"),
    ("sum", "sum_long"),
)


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
        "below_multiply_floor": (),
        "contraction_is_free": (),
        "priced_by_outputs": (),
        "priced_by_inputs": (),
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
    # 与"秘密乘法下限"对账（P6 实测 16 B/元素/次乘法）。
    # 纪律：下限必须**与被测电路的计费维度对齐**——逐元素电路按输入元素数对账，
    # 收缩/矩阵乘按**输出个数**对账（收缩长度免费，见模块 docstring）。
    # 第一版不分维度、一律除以输入元素数，把 `dot` 实测的 16 B 误判成
    # "低于下限"（N=64 时 16 B < 64×16 B=1024 B）——读数没错，**尺子错了**，
    # 与 P6 `x * 2` 实测 0 B 是同族教训。
    by_name = {str(r.get("name")): r for r in records}
    below: list[str] = []
    for name, demand in MULTIPLY_DEMAND.items():
        record = by_name.get(name)
        if not record:
            continue
        comm = record.get("comm_total_bytes")
        if not isinstance(comm, (int, float)):
            continue
        by_output = record.get("cost_driver") == "outputs"
        billed = record.get("outputs") if by_output else record.get("elements")
        if not isinstance(billed, (int, float)) or billed <= 0:
            continue
        floor = MUL_BYTES_PER_ELEMENT * float(billed) * float(demand)
        if comm < floor:
            below.append(name)
            notes.append(
                f"{name} 实测通信量 {comm:.0f} B 低于同一计费维度上的秘密乘法下限 "
                f"{floor:.0f} B（{MUL_BYTES_PER_ELEMENT:.0f} B/"
                f"{'输出' if by_output else '元素'} × {int(billed)}）："
                "要么该原语有更省的专用协议，要么这条路径没在真做保密乘法——"
                "本探针不给结论，标为可疑，需另行复核"
            )
    summary["below_multiply_floor"] = tuple(below)

    # 计费维度登记：收缩类用例必须显式标 outputs，否则又会退回"除以输入元素数"的旧口径。
    summary["priced_by_outputs"] = tuple(
        str(r.get("name")) for r in records if r.get("cost_driver") == "outputs"
    )
    summary["priced_by_inputs"] = tuple(
        str(r.get("name")) for r in records if r.get("cost_driver") != "outputs"
    )

    # 收缩长度免费：同一原语、同一输出个数、仅收缩长度不同 → 通信量应一致。
    contraction_free: list[str] = []
    for short_name, long_name in CONTRACTION_PAIRS:
        short_case, long_case = by_name.get(short_name), by_name.get(long_name)
        if not short_case or not long_case:
            continue
        short_comm = short_case.get("comm_total_bytes")
        long_comm = long_case.get("comm_total_bytes")
        if not (isinstance(short_comm, (int, float)) and isinstance(long_comm, (int, float))):
            continue
        if float(short_comm) <= 0:
            continue
        if float(short_comm) == float(long_comm):
            contraction_free.append(short_name)
            notes.append(
                f"收缩长度不计费：{short_name}（收缩长度 {short_case.get('elements')}）与 "
                f"{long_name}（收缩长度 {long_case.get('elements')}）输出个数同为 "
                f"{short_case.get('outputs')}，通信量同为 {float(short_comm):.0f} B——"
                "通信量随**输出个数**走，收缩长度是免费维度"
            )
        else:
            notes.append(
                f"{short_name} 与 {long_name} 通信量不一致"
                f"（{short_comm} vs {long_comm}）：收缩长度是否免费存疑，需复核"
            )
    summary["contraction_is_free"] = tuple(contraction_free)

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
    "outputs",
    "cost_driver",
    "repeat",
    "status",
    "wall_ms",
    "pphlo_bytes",
    "profiled",
    "comm_send_bytes",
    "comm_recv_bytes",
    "comm_total_bytes",
    "comm_per_element",
    "comm_per_output",
    "cost_per_unit",
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
    header = (
        f"{'primitive':12s} {'needed_by':10s} {'billing':8s} {'status':12s} "
        f"{'对拍':>5s} {'comm_B':>10s} {'B/单位':>9s}  真机原语"
    )
    lines = [header]
    for record in records:
        comm = record.get("comm_total_bytes")
        unit = record.get("cost_per_unit")
        if unit is None:
            unit = record.get("comm_per_element")
        billing = "outputs" if record.get("cost_driver") == "outputs" else "inputs"
        match = record.get("within_tolerance")
        mark = "-" if match is None else ("是" if match else "否")
        primitives = ",".join(sorted((record.get("comm_by_primitive") or {}).keys()))
        lines.append(
            f"{str(record.get('name', ''))[:12]:12s} "
            f"{str(record.get('needed_by', ''))[:10]:10s} "
            f"{billing:8s} "
            f"{str(record.get('status', '')):12s} "
            f"{mark:>5s} "
            f"{(f'{comm:.0f}' if isinstance(comm, (int, float)) else '-'):>10s} "
            f"{(f'{unit:.2f}' if isinstance(unit, (int, float)) else '-'):>9s}  "
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
