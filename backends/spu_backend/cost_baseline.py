# -*- coding: utf-8 -*-
"""已提交的 MPC 实测代价产物 → 只读查表（供 planner 做代价排序）。

职责边界
--------
只**读** `docs/mpc_comm_baseline.json`（`tests/benchmarks/benchmark_mpc.py
--comm` 的产物），不重新测量、不推测、不补数字：

- 产物不存在 / 解析失败 / 该算子没有可比行 → 返回**空表**，调用方必须自行退化。
  不得用"看起来合理"的常量顶替实测值（这与 `benchmark.py` 的"空就是空"一致）。
- 同一配置被重复测量时（主扫描 / 环宽矩阵 / 规模扫描会重复覆盖同一格），
  报**中位数**并记录样本数，与 `docs/MPC_BENCHMARK_PROTOCOL.md` §6
  "报中位/区间，不报单次"的口径一致。

为什么锁定 (FM64, 默认电路) 并挑同一个 K
----------------------------------------
跨协议比价必须在**同一规模**上做。本模块选"能覆盖最多协议的那个 K"
（覆盖数相同则取更小的 K），于是：

    DistanceLE / WeightedSum → K=256（五个协议都在该 K 上有可比行）
    TemporalOverlap          → K=32 （pairwise 电路只扫到这一档，五个协议同档）

**不**取各协议各自的最大 K：那样 ABY3 会取到 K=4096、其余协议只有 K=256，
比较的就不是同一个量了。环宽固定 FM64、电路固定默认策略（`strategy` 为空的行）
同样是为了让比较对象唯一。

TemporalOverlap 的 sweep 电路只测过 ABY3（`docs/mpc_temporal_ab.json`），
**不**并入本表：单协议数据无法参与跨协议排序。
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from typing import Any, Mapping

#: 实测产物（相对项目根）
ARTIFACT: str = "docs/mpc_comm_baseline.json"

#: 比价时固定的环宽与电路
COMPARE_FIELD: str = "FM64"
COMPARE_STRATEGY: str = ""

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def artifact_path() -> str:
    """实测产物的绝对路径（按本文件位置解析，与 CWD 无关）。"""

    return os.path.join(_ROOT, ARTIFACT)


def load_rows(path: str | None = None) -> list[dict[str, Any]] | None:
    """读实测产物。

    返回 `None` = **没读到**（文件缺失或不是 JSON 列表）；返回 `[]` = 读到了
    但没有行。两者必须区分：前者是"没有测量依据"，后者是"有依据但为空"。
    """

    try:
        with open(path or artifact_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, list):
        return None
    return [row for row in data if isinstance(row, dict)]


@dataclass(frozen=True)
class MeasuredComm:
    """一个 (算子, 协议) 在比价规模上的实测通信量。"""

    protocol: str
    k: int
    field: str
    comm_total_bytes: float
    wall_ms_median: float | None
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "k": self.k,
            "field": self.field,
            "comm_total_bytes": self.comm_total_bytes,
            "wall_ms_median": self.wall_ms_median,
            "samples": self.samples,
        }


def _comparable(row: Mapping[str, Any]) -> bool:
    """该行能否进入跨协议比价（口径固定，缺一项就不入选、不猜）。"""

    return (
        row.get("status") == "ok"
        and row.get("field") == COMPARE_FIELD
        and (row.get("strategy") or "") == COMPARE_STRATEGY
        and isinstance(row.get("protocol"), str)
        and bool(row.get("protocol"))
        and isinstance(row.get("k"), int)
        and isinstance(row.get("comm_total_bytes"), (int, float))
    )


def measured_comm_table(
    op: str, *, path: str | None = None
) -> dict[str, MeasuredComm]:
    """该算子在**同一比价规模**上、各协议的实测通信量。

    返回 `{}` 表示本算子没有可用实测（产物缺失或无可比行）——调用方据此退化。
    """

    rows = load_rows(path)
    if not rows:
        return {}

    usable = [row for row in rows if row.get("op") == op and _comparable(row)]
    if not usable:
        return {}

    by_protocol: dict[str, dict[int, list[Mapping[str, Any]]]] = {}
    for row in usable:
        by_protocol.setdefault(str(row["protocol"]), {}).setdefault(
            int(row["k"]), []
        ).append(row)

    coverage: dict[int, int] = {}
    for by_k in by_protocol.values():
        for k in by_k:
            coverage[k] = coverage.get(k, 0) + 1
    # 覆盖协议数最多者胜；并列时取更小的 K（同量比较，规模越小越省）
    best_k = max(coverage, key=lambda k: (coverage[k], -k))

    table: dict[str, MeasuredComm] = {}
    for protocol, by_k in by_protocol.items():
        group = by_k.get(best_k)
        if not group:
            continue
        comm = statistics.median(
            float(row["comm_total_bytes"]) for row in group
        )
        walls = [
            float(row["wall_ms"])
            for row in group
            if isinstance(row.get("wall_ms"), (int, float))
        ]
        table[protocol] = MeasuredComm(
            protocol=protocol,
            k=best_k,
            field=COMPARE_FIELD,
            comm_total_bytes=comm,
            wall_ms_median=statistics.median(walls) if walls else None,
            samples=len(group),
        )
    return table
