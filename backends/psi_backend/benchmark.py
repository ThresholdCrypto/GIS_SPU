# -*- coding: utf-8 -*-
"""PSI 性能基线：生成可控测试集，真机测量 ECDH / KKRT / RR22，诚实记录。

依据：下一阶段任务文档 §6（性能统计体系）/ §7（真实性能基线）/ §23 Phase 5。

职责边界
--------
- 只测量"原始集合 → PSI"的基线。**不做**去重 / 排序 / 压缩——那是
  `geosot_optimizer`（Geo-RR22 预处理）下一阶段的工作；重复键按原样传入，
  官方 PSI 对重复键的实际行为如实记录，不做粉饰。
- 没有真实执行过的组合**不填数字**：`status="unavailable"` + 说明原因；
  执行失败如实记 `status="error"` + 错误原文；`agreement` 只来自真实对拍。
- 生成器是**确定性构造**（无随机数）：同一参数在任何机器上给出同一批集合，
  便于跨机对拍。它不模拟真实业务分布，只提供可控变量（规模 / 交集比例 /
  重复比例 / 高位码 / 层级 / 高度层）。

默认扫描策略（standard_cases）
------------------------------
- 四个协议变体（ECDH / KKRT / RR22 / RR22+low_comm）× N=2^10…2^18；
- KKRT / RR22 两族继续到 2^20 / 2^22 / 2^24；
- ECDH 只实测到 2^20；2^22 / 2^24 记 `unavailable`（键数二次增长，
  2^18 实测已约 24 s；不伪造数字，可另行补测）。
- 变量矩阵（N=2^12，RR22）：交集比例 0 / 1、重复 25%、高位码（≥2^63）、
  L=6、Z=5、receiver_rank=1。

字段口径（与任务文档 §7.3 对齐，"补充"是本模块多给的字段）
--------------------------------------------------------
§7.3 要求：`protocol` / `low_comm_mode` / `n_left` / `n_right` /
`intersection_ratio` / `psi_execute_ms` / `total_ms` / `memory_mb` /
`agreement`。补充：`setup_ms`（造数）、`io_write_ms` / `io_read_ms` /
`semantic_ms`（来自运行时 `timings_ms`）、`peak_rss_mb`、
`intersection_count`、`status` / `note` / `error`（诚实规则）。

`memory_mb` = 本次运行前后进程峰值 RSS 的增量（ru_maxrss 高水位差，≥0），
是"这次运行把内存高水位抬了多少"，会低估绝对值——绝对值看 `peak_rss_mb`。
"""

from __future__ import annotations

import csv
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ir.values import encode_grid_code, grid_code_layout_manifest

from .capability import (
    PSI_DEFAULT_CURVE,
    PsiCapabilityReport,
    check_psi_capabilities,
    protocol_needs_curve,
)
from .runtime import PsiRuntimeConfig, run_psi_intersection

# --------------------------------------------------------------------------
# 常量：默认扫描策略
# --------------------------------------------------------------------------

_MASK17 = (1 << 17) - 1
_HIGH_BIT = 1 << 63

#: 默认对比的四个协议变体（RR22 各测普通 / 低通信两档）
PROTOCOL_VARIANTS: tuple[tuple[str, bool], ...] = (
    ("PROTOCOL_ECDH", False),
    ("PROTOCOL_KKRT", False),
    ("PROTOCOL_RR22", False),
    ("PROTOCOL_RR22", True),
)

_STANDARD_SIZES = (2**10, 2**12, 2**14, 2**16, 2**18)
_STANDARD_LARGE_SIZES = (2**20, 2**22, 2**24)
#: ECDH 实测到 2^20（键数二次增长）；2^22 / 2^24 记 unavailable，不伪造
_ECDH_MEASURED_SIZES = _STANDARD_SIZES + (2**20,)
_ECDH_UNAVAILABLE_SIZES = (2**22, 2**24)

#: 变量矩阵的基准规模（小规模、每次只变一项）
_VARIANT_N = 2**12

#: CSV 输出列（JSON 是主格式，CSV 只是便于表格工具）
_CSV_COLUMNS: tuple[str, ...] = (
    "case",
    "protocol",
    "low_comm_mode",
    "curve",
    "n_left",
    "n_right",
    "n_left_unique",
    "n_right_unique",
    "intersection_ratio",
    "duplicate_ratio",
    "high_bits",
    "level",
    "z",
    "receiver_rank",
    "status",
    "psi_execute_ms",
    "io_write_ms",
    "io_read_ms",
    "semantic_ms",
    "total_ms",
    "memory_mb",
    "peak_rss_mb",
    "agreement",
    "intersection_count",
    "note",
    "error",
)


def format_n(n: int) -> str:
    """把集合规模写成 `2^10` 这类可读标签（非 2 幂时原样输出）。"""

    if n > 0 and (n & (n - 1)) == 0:
        return f"2^{n.bit_length() - 1}"
    return str(n)


# --------------------------------------------------------------------------
# 测试集生成（确定性）
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkSet:
    """一对确定性构造的格网集合。

    `left` / `right` 按只读约定使用（不复制、不修改，避免大 N 下翻倍吃内存）。
    """

    left: list[int]
    right: list[int]
    meta: Mapping[str, Any]

    @property
    def n_left(self) -> int:
        return len(self.left)

    @property
    def n_right(self) -> int:
        return len(self.right)

    def to_dict(self) -> dict[str, Any]:
        return {"n_left": self.n_left, "n_right": self.n_right, **dict(self.meta)}


def _codes(
    count: int,
    base: int,
    *,
    z: int,
    level: int,
    toff: int,
    lt: int,
    high_bits: bool,
) -> list[int]:
    """按"线性地址 k → (X, Y) 双射"生成 count 个互不相同的 64 位格网码。

    k 的低 17 位进 X、高 17 位进 Y，保证在 k < 2^34 内不重复；
    不同 base 的两段只要落在不同的 k 区间就不会撞码——这是
    "交集/非交集"能按构造精确控制（而不是碰运气）的基础。
    """

    out: list[int] = []
    append = out.append
    for i in range(count):
        k = base + i
        code = encode_grid_code(
            x=k & _MASK17, y=(k >> 17) & _MASK17, z=z, level=level, toff=toff, lt=lt
        )
        append((code | _HIGH_BIT) if high_bits else code)
    return out


def generate_benchmark_sets(
    n: int,
    *,
    intersection_ratio: float = 0.5,
    duplicate_ratio: float = 0.0,
    high_bits: bool = False,
    level: int = 9,
    z: int = 15,
    toff: int = 0,
    lt: int = 4,
) -> BenchmarkSet:
    """构造一对规模为 n（唯一码）/ 交集比例可控的格网集合。

    - 两边各 n 个**唯一**码；交集 = 左边前 round(n * intersection_ratio) 个；
    - `duplicate_ratio`：两边各自把开头 round(n * ratio) 个元素重复追加一遍
      （重复是**原样传给 PSI**，不是先洗掉；去重是下一阶段的事）；
    - `high_bits=True`：把所有码的最高位（X 位域 bit16）置 1，
      即交集判定必须走 `≥2^63` 无符号路径；
    - 布局/层级/高度层参数原样进码，使"不同 L / 不同高度层"矩阵可控。
    """

    n = int(n)
    if n <= 0:
        raise ValueError(f"n 必须为正整数，实得 {n!r}")
    intersection_ratio = float(intersection_ratio)
    duplicate_ratio = float(duplicate_ratio)
    if not 0.0 <= intersection_ratio <= 1.0:
        raise ValueError(
            f"intersection_ratio 必须在 [0, 1]，实得 {intersection_ratio!r}"
        )
    if not 0.0 <= duplicate_ratio < 1.0:
        raise ValueError(f"duplicate_ratio 必须在 [0, 1)，实得 {duplicate_ratio!r}")

    isect = int(round(n * intersection_ratio))
    dup = int(round(n * duplicate_ratio))

    kw = dict(z=z, level=level, toff=toff, lt=lt, high_bits=high_bits)
    left_unique = _codes(n, 0, **kw)
    # 右侧新增段整体挪到左边 k 区间之后（对齐到下一段 Y），保证除前缀外不相交
    extras_base = ((n >> 17) + 1) << 17
    right_unique = left_unique[:isect] + _codes(n - isect, extras_base, **kw)
    left = left_unique + left_unique[:dup]
    right = right_unique + right_unique[:dup]

    meta: dict[str, Any] = {
        "n": n,
        "intersection_count": isect,
        "duplicate_count": dup,
        "intersection_ratio": intersection_ratio,
        "duplicate_ratio": duplicate_ratio,
        "high_bits": bool(high_bits),
        "level": level,
        "z": z,
        "toff": toff,
        "lt": lt,
        "layout_id": grid_code_layout_manifest()["layout_id"],
        "generator": "deterministic-index-addressing-v1",
    }
    return BenchmarkSet(left=left, right=right, meta=meta)


def analyze_benchmark_set(bset: BenchmarkSet) -> dict[str, Any]:
    """重算集合关系做校验（小规模测试用；大 N 会额外吃内存，别在 2^20+ 用）。"""

    left_set = set(bset.left)
    right_set = set(bset.right)
    inter = left_set & right_set
    expected = int(bset.meta["intersection_count"])
    return {
        "n_left_unique": len(left_set),
        "n_right_unique": len(right_set),
        "intersection_count": len(inter),
        "intersection_count_expected": expected,
        "duplicate_count_left": len(bset.left) - len(left_set),
        "duplicate_count_right": len(bset.right) - len(right_set),
        "duplicate_count_declared": int(bset.meta["duplicate_count"]),
        "ok": (
            len(left_set) == int(bset.meta["n"])
            and len(right_set) == int(bset.meta["n"])
            and len(inter) == expected
        ),
    }


# --------------------------------------------------------------------------
# 用例与执行
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkCase:
    """一条基线用例（协议 × 规模 × 单项变量）。

    `execute=False` 的用例**不执行**：结果行记 `unavailable` + `unavailable_note`，
    用于把"本机预算不覆盖"的组合也留在基线文件里（占位但不填数字）。
    """

    protocol: str
    low_comm_mode: bool = False
    n: int = 2**10
    intersection_ratio: float = 0.5
    duplicate_ratio: float = 0.0
    high_bits: bool = False
    level: int = 9
    z: int = 15
    receiver_rank: int = 0
    execute: bool = True
    unavailable_note: str = ""
    #: 期望的终态（"" = 期望 ok）：把"已知会失败的重复键用例"与意外错误分开，
    #: 运行器据此判定退出码；期望失败但实际成功同样算"预期外"（提示该更新文档）。
    expect_status: str = ""
    #: 用例级说明（拼进记录 note，解释该用例为什么有这种预期）
    case_note: str = ""

    def label(self) -> str:
        parts = [
            self.protocol.replace("PROTOCOL_", "")
            + ("+low_comm" if self.low_comm_mode else "")
        ]
        parts.append(f"N={format_n(self.n)}")
        if self.intersection_ratio != 0.5:
            parts.append(f"intr={self.intersection_ratio}")
        if self.duplicate_ratio:
            parts.append(f"dup={self.duplicate_ratio}")
        if self.high_bits:
            parts.append("high_bits")
        if self.level != 9:
            parts.append(f"L={self.level}")
        if self.z != 15:
            parts.append(f"Z={self.z}")
        if self.receiver_rank:
            parts.append(f"recv={self.receiver_rank}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "low_comm_mode": self.low_comm_mode,
            "n": self.n,
            "intersection_ratio": self.intersection_ratio,
            "duplicate_ratio": self.duplicate_ratio,
            "high_bits": self.high_bits,
            "level": self.level,
            "z": self.z,
            "receiver_rank": self.receiver_rank,
            "execute": self.execute,
            "expect_status": self.expect_status,
        }


def plain_intersects_reference(left: Sequence[int], right: Sequence[int]) -> bool:
    """`Intersects` 的明文参考：两边是否有公共元素。

    用 `isdisjoint` 而不是 `set(left) & set(right)`：大 N 下后者会多造一个
    交集集合；`isdisjoint` 只物化一侧的 set，另一侧流式扫描。
    """

    return not set(right).isdisjoint(left)


def _peak_rss_mb() -> float:
    """进程峰值 RSS（MB）。

    Linux: ru_maxrss 单位是 KB；macOS: 字节，需要换算。基线在 WSL/Linux 上跑；
    其它平台原样返回（相对增量仍可用，绝对值别引用）。
    """

    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        raw /= 1024.0
    return raw / 1024.0


def _blank_record(case: BenchmarkCase) -> dict[str, Any]:
    return {
        "case": case.label(),
        "protocol": case.protocol,
        "low_comm_mode": case.low_comm_mode,
        "curve": None,
        "n": case.n,
        "n_left": None,
        "n_right": None,
        "n_left_unique": None,
        "n_right_unique": None,
        "intersection_ratio": case.intersection_ratio,
        "intersection_ratio_actual": None,
        "duplicate_ratio": case.duplicate_ratio,
        "duplicate_count": 0,
        "high_bits": case.high_bits,
        "level": case.level,
        "z": case.z,
        "receiver_rank": case.receiver_rank,
        "expect_status": case.expect_status,
        "layout_id": grid_code_layout_manifest()["layout_id"],
        "status": "unavailable",
        "note": "",
        "setup_ms": None,
        "psi_execute_ms": None,
        "io_write_ms": None,
        "io_read_ms": None,
        "semantic_ms": None,
        "total_ms": None,
        "wall_ms": None,
        "timings_ms": {},
        "memory_mb": None,
        "peak_rss_mb": None,
        "agreement": None,
        "intersection_count": None,
        "intersection_count_expected": None,
        "error": None,
    }


def run_benchmark_case(
    case: BenchmarkCase,
    *,
    report: PsiCapabilityReport | None = None,
    workdir: str | None = None,
) -> dict[str, Any]:
    """执行一条用例并返回记录（失败也不会抛：如实记进 `status` / `error`）。"""

    record = _blank_record(case)
    if not case.execute:
        record["note"] = case.unavailable_note or "未执行（默认预算策略不含该组合）"
        return record

    t_case = time.perf_counter()
    t0 = time.perf_counter()
    bset = generate_benchmark_sets(
        case.n,
        intersection_ratio=case.intersection_ratio,
        duplicate_ratio=case.duplicate_ratio,
        high_bits=case.high_bits,
        level=case.level,
        z=case.z,
    )
    record["setup_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    record.update(
        n_left=bset.n_left,
        n_right=bset.n_right,
        n_left_unique=case.n,
        n_right_unique=case.n,
        duplicate_count=int(bset.meta["duplicate_count"]),
        intersection_count_expected=int(bset.meta["intersection_count"]),
    )

    curve = PSI_DEFAULT_CURVE if protocol_needs_curve(case.protocol) else None
    params: dict[str, Any] = {}
    if case.protocol == "PROTOCOL_RR22":
        params["low_comm_mode"] = case.low_comm_mode
    config = PsiRuntimeConfig(
        protocol=case.protocol,
        curve=curve,
        receiver_rank=case.receiver_rank,
        protocol_params=params,
    )
    report = report or check_psi_capabilities()

    rss_before = _peak_rss_mb()
    try:
        # 基准测的是**协议本体**行为：显式关闭 Geo-RR22 预处理
        # （optimize_input=False），既保留重复键用例的原始失败结论
        # （RR22/Paxos、KKRT/cuckoo 报错；ECDH 计数膨胀），
        # 也与已归档基线（docs/psi_benchmark_baseline.json）口径一致。
        # 生产路径默认开启排序+去重，见 geosot_optimizer 与 GEO_RR22_DESIGN。
        run = run_psi_intersection(
            bset.left,
            bset.right,
            op="Intersects",
            config=config,
            optimize_input=False,
            reference_fn=plain_intersects_reference,
            report=report,
            workdir=workdir,
        )
    except Exception as exc:  # 基线脚本不吞异常：如实记错误，不伪造结果
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        prefix = f"{case.case_note}；" if case.case_note else ""
        record["note"] = prefix + "用例执行抛出异常（未走到 PsiRunResult 路径）"
        record["peak_rss_mb"] = round(_peak_rss_mb(), 1)
        record["wall_ms"] = round((time.perf_counter() - t_case) * 1000.0, 3)
        return record

    record["wall_ms"] = round((time.perf_counter() - t_case) * 1000.0, 3)
    record["peak_rss_mb"] = round(_peak_rss_mb(), 1)
    record["memory_mb"] = round(max(record["peak_rss_mb"] - rss_before, 0.0), 1)
    record["curve"] = run.curve
    record["status"] = run.status
    timings = dict(run.timings_ms)
    record["timings_ms"] = timings
    for key in ("psi_execute_ms", "io_write_ms", "io_read_ms", "semantic_ms", "total_ms"):
        record[key] = timings.get(key)
    record["agreement"] = run.agreement
    record["intersection_count"] = run.intersection_count
    if run.intersection_count is not None and case.n:
        record["intersection_ratio_actual"] = round(run.intersection_count / case.n, 4)
    record["error"] = run.error
    note_parts = [case.case_note] if case.case_note else []
    note_parts.extend(str(note) for note in run.notes)
    record["note"] = "；".join(note_parts)
    return record


def run_benchmark_cases(
    cases: Sequence[BenchmarkCase],
    *,
    report: PsiCapabilityReport | None = None,
    out_json: str | None = None,
    out_csv: str | None = None,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """顺序执行一批用例；`out_json` 给出时**每条一存**（长扫描不怕中断丢结果）。"""

    report = report or check_psi_capabilities()
    records: list[dict[str, Any]] = []
    total = len(cases)
    for idx, case in enumerate(cases, 1):
        if progress is not None:
            progress(f"[{idx}/{total}] {case.label()}")
        try:
            record = run_benchmark_case(case, report=report)
        except Exception as exc:  # 用例级兜底：单条异常不中断整批
            record = _blank_record(case)
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["note"] = "用例级异常（run_benchmark_case 之外）"
        records.append(record)
        if progress is not None:
            progress(
                "    status={status} psi_execute_ms={psi} total_ms={total} agreement={agr}".format(
                    status=record.get("status"),
                    psi=record.get("psi_execute_ms"),
                    total=record.get("total_ms"),
                    agr=record.get("agreement"),
                )
            )
        if out_json is not None:
            write_benchmark_json(records, out_json)
    if out_csv is not None:
        write_benchmark_csv(records, out_csv)
    return records


# --------------------------------------------------------------------------
# 默认用例集
# --------------------------------------------------------------------------


def standard_cases(*, quick: bool = False) -> list[BenchmarkCase]:
    """标准扫描用例集（见模块 docstring 的"默认扫描策略"）。"""

    if quick:
        return [
            BenchmarkCase(protocol=protocol, low_comm_mode=low_comm, n=n)
            for protocol, low_comm in PROTOCOL_VARIANTS
            for n in (2**10, 2**12)
        ]

    cases: list[BenchmarkCase] = []
    for protocol, low_comm in PROTOCOL_VARIANTS:
        ecdh = protocol == "PROTOCOL_ECDH"
        sizes = (
            _ECDH_MEASURED_SIZES
            if ecdh
            else _STANDARD_SIZES + _STANDARD_LARGE_SIZES
        )
        for n in sizes:
            cases.append(BenchmarkCase(protocol=protocol, low_comm_mode=low_comm, n=n))

    # ECDH 的超大尺寸：不实测，如实占位（不伪造数字）
    for n in _ECDH_UNAVAILABLE_SIZES:
        cases.append(
            BenchmarkCase(
                protocol="PROTOCOL_ECDH",
                n=n,
                execute=False,
                unavailable_note=(
                    "未实测：默认预算不含 ECDH 2^22/2^24（键数二次增长，"
                    "N=2^20 已到分钟级）；需要时用 "
                    "tests/benchmarks/benchmark_psi.py --sizes 显式补测"
                ),
            )
        )

    # 变量矩阵（RR22、小规模、单项变量）
    matrix = (
        {"intersection_ratio": 0.0},
        {"intersection_ratio": 1.0},
        {"high_bits": True},
        {"level": 6},
        {"z": 5},
        {"receiver_rank": 1},
    )
    for overrides in matrix:
        cases.append(BenchmarkCase(protocol="PROTOCOL_RR22", n=_VARIANT_N, **overrides))

    # 重复键矩阵：三个协议的真实行为各不同（已探明），分开记录；
    # 不是"意外失败"——RR22/KKRT 的报错就是本阶段要拿到的结论。
    # 顺序纪律：**可完成的行排在会失败的行之前**。官方 PSI 在一次失败后
    # 可能留下影响同进程后续运行的全局状态（曾观察到失败行之后紧邻的
    # ECDH 行出现一次 AllGather 超时，单独复测通过）；把失败行放最后，
    # 让可完成行的结果稳定且可复现。
    cases.append(
        BenchmarkCase(
            protocol="PROTOCOL_ECDH",
            n=_VARIANT_N,
            duplicate_ratio=0.25,
            expect_status="ok",
            case_note=(
                "重复键用例：ECDH 可完成，但 intersection_count 含重复乘数"
                "（N=2^12 实测 3072 vs 唯一 2048），不得按唯一集合预期解读；"
                "曾观察到紧邻失败用例后的偶发 AllGather 超时（复测通过）"
            ),
        )
    )
    cases.append(
        BenchmarkCase(
            protocol="PROTOCOL_RR22",
            n=_VARIANT_N,
            duplicate_ratio=0.25,
            expect_status="error",
            case_note=(
                "重复键用例：预期 error（Paxos duplicate keys）——"
                "去重预处理因此是正确性前提，见 docs/GEO_RR22_DESIGN.md"
            ),
        )
    )
    cases.append(
        BenchmarkCase(
            protocol="PROTOCOL_KKRT",
            n=_VARIANT_N,
            duplicate_ratio=0.25,
            expect_status="error",
            case_note="重复键用例：预期 error（cuckoo stash 无空 bin）",
        )
    )
    return cases


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------


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
    """挑出"与预期不符"的记录。

    - 带 `expect_status` 的用例：状态不等于预期都算预期外（含"期望失败却成功"——
      提示上游行为已变化，该更新文档而不是悄悄放过）；
    - 其余用例：非 ok / unavailable 即为意外失败。
    """

    unexpected: list[Mapping[str, Any]] = []
    for record in records:
        status = record.get("status")
        expected = record.get("expect_status") or ""
        if expected:
            if status != expected:
                unexpected.append(record)
        elif status not in ("ok", "unavailable"):
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


#: 性能对比的默认提示倍率（仅提示、不阻断；§19：阈值不要过早硬化）
DEFAULT_REGRESSION_WARN_RATIO = 1.5


def compare_with_baseline(
    current: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    warn_ratio: float = DEFAULT_REGRESSION_WARN_RATIO,
) -> list[dict[str, Any]]:
    """性能回归门禁骨架（§19）：逐用例对比当前记录与基线（**只报告，不阻断**）。

    - 对位键：记录的 `case` 字段（BenchmarkCase.label()：协议 / 规模 / 变体标记）；
    - 只有两边都 `status == "ok"` 且都记录了 `wall_ms` 时才计算倍率；
      error / unavailable 行参与"覆盖差异"报告，但不做耗时对比；
    - `warn_ratio` 只决定 `flagged` 提示标记；本函数不设全局阈值、不抛异常——
      是否让性能波动阻塞提交，由调用方（CI 任务）显式决定。
    """

    current_by_case = {str(record.get("case")): record for record in current}
    baseline_by_case = {str(record.get("case")): record for record in baseline}
    rows: list[dict[str, Any]] = []
    for case in sorted(set(current_by_case) | set(baseline_by_case)):
        now = current_by_case.get(case)
        before = baseline_by_case.get(case)
        row: dict[str, Any] = {
            "case": case,
            "protocol": (now or before or {}).get("protocol"),
            "wall_ms": None,
            "baseline_wall_ms": None,
            "ratio": None,
            "flagged": False,
            "reason": "",
        }
        if now is None:
            row["reason"] = "仅基线有：当前扫描未覆盖该用例"
        elif before is None:
            row["reason"] = "仅当前有：基线未收录该用例"
        elif (
            now.get("status") == "ok"
            and before.get("status") == "ok"
            and now.get("wall_ms") is not None
            and before.get("wall_ms") is not None
            and before.get("wall_ms")
        ):
            wall = float(now["wall_ms"])
            base = float(before["wall_ms"])
            row["wall_ms"] = now["wall_ms"]
            row["baseline_wall_ms"] = before["wall_ms"]
            row["ratio"] = round(wall / base, 3)
            if row["ratio"] >= warn_ratio:
                row["flagged"] = True
                row["reason"] = (
                    f"耗时 {row['ratio']:.2f}× 基线（≥ {warn_ratio}×，仅提示）"
                )
        else:
            row["reason"] = (
                f"不比较：current={now.get('status')} baseline={before.get('status')}"
                "（error / unavailable 行不作性能对比）"
            )
        rows.append(row)
    return rows


def format_summary(records: Sequence[Mapping[str, Any]]) -> str:
    """把记录渲染成终端可读的摘要表。"""

    header = (
        f"{'case':54s}  {'status':12s} {'psi_ms':>9s} {'total_ms':>9s} "
        f"{'rss_MB':>7s} {'agr':>5s}"
    )
    lines = [header]
    for record in records:
        lines.append(
            f"{str(record.get('case', ''))[:54]:54s}  "
            f"{str(record.get('status', '')):12s} "
            f"{_fmt_num(record.get('psi_execute_ms')):>9s} "
            f"{_fmt_num(record.get('total_ms')):>9s} "
            f"{_fmt_num(record.get('memory_mb')):>7s} "
            f"{_fmt_bool(record.get('agreement')):>5s}"
        )
    return "\n".join(lines)