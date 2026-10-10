# -*- coding: utf-8 -*-
"""PSI-CA 性能基线运行器（Phase 9）：openmined-psi 计数档，真机测量，诚实记录。

在 WSL / Linux（spu311 环境）中运行：

    /opt/miniconda3/envs/spu311/bin/python tests/benchmarks/benchmark_psi_ca.py \
        --json docs/psi_ca_benchmark_baseline.json \
        --csv docs/psi_ca_benchmark_baseline.csv \
        > /tmp/psi_ca_bench_run.log 2>&1

职责边界
--------
- 复用 libpsi 基线的**确定性用例构造**（`backends.psi_backend.benchmark`）：
  三条 PSI 路径测的是同一批输入（同规模、同交集比例、同层级布局）；
- 只采集**真实测得到**的量：运行时的 `timings_ms`（`psi_execute_ms` /
  `total_ms`）与**运行器进程**的 ru_maxrss（openmined-psi 是进程内库，
  该读数覆盖协议内存；与 libpsi 基线同一测量口径）；
- 通信量本档**没有采集**：不填 0、不推测——缺口登记在
  `backends/benchmark_schema.py` 的 `MISSING_METRICS`（族 `PSI-CA`）；
- 执行失败如实记 `status="error"` + 错误原文；没有真正跑过的用例不填数字；
- 能力不可用时**直接失败且不写产物**（退出码 1）——不能用 unavailable 行
  覆盖仓库里的真实基线。

退出码：`0` = 没有预期外记录（全部 ok）；`1` = 存在预期外记录 / 能力不可用。
"""

from __future__ import annotations

import argparse
import csv
import os
import resource
import sys
import time
from typing import Any, Mapping, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backends.psi_backend.benchmark import (  # noqa: E402
    format_n,
    generate_benchmark_sets,
    write_benchmark_json,
)
from backends.psi_ca_backend.capability import check_psi_ca_capabilities  # noqa: E402
from backends.psi_ca_backend.runtime import (  # noqa: E402
    PSI_CA_PROTOCOL,
    run_psi_cardinality,
)
from ir.values import grid_code_layout_manifest  # noqa: E402

#: 默认规模扫描（2 的幂指数；可用 --sizes 覆盖）
_DEFAULT_SIZES = "10,12,14"

#: CSV 输出列（JSON 是主格式；dict 列如 timings_ms / protocol_params 不进 CSV）
_CSV_COLUMNS: tuple[str, ...] = (
    "case",
    "protocol",
    "protocol_leak",
    "result_semantics",
    "structure",
    "n",
    "n_left",
    "n_right",
    "n_left_unique",
    "n_right_unique",
    "intersection_ratio",
    "intersection_ratio_actual",
    "duplicate_ratio",
    "duplicate_count",
    "high_bits",
    "level",
    "z",
    "receiver_rank",
    "layout_id",
    "status",
    "setup_ms",
    "psi_ca_execute_ms",
    "total_ms",
    "wall_ms",
    "memory_mb",
    "peak_rss_mb",
    "agreement",
    "intersection_count",
    "intersection_count_expected",
    "error",
    "note",
)


def _resolve(path: str) -> str:
    """相对路径按项目根解析（脚本从任意 CWD 运行结果一致）。"""

    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


def _peak_rss_mb() -> float:
    """运行器进程峰值 RSS（MB）；与 `backends/psi_backend/benchmark.py` 同口径。

    Linux 的 ru_maxrss 单位是 KB；openmined-psi 进程内加载，该读数覆盖协议内存。
    """

    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _plain_intersection_codes(left: Sequence[int], right: Sequence[int]) -> list[int]:
    """明文参考：交集码（去重排序）；运行时把它折算到计数口径再对拍。"""

    return sorted(set(left) & set(right))


def _blank_record(n: int) -> dict[str, Any]:
    return {
        "case": f"{PSI_CA_PROTOCOL} N={format_n(n)}",
        "protocol": PSI_CA_PROTOCOL,
        "protocol_leak": None,
        "result_semantics": None,
        "structure": None,
        "operation": "CellSetIntersect",
        "world_size": None,
        "n": n,
        "n_left": None,
        "n_right": None,
        "n_left_unique": None,
        "n_right_unique": None,
        "intersection_ratio": 0.5,
        "intersection_ratio_actual": None,
        "duplicate_ratio": 0.0,
        "duplicate_count": 0,
        "high_bits": False,
        "level": 9,
        "z": 15,
        "receiver_rank": None,
        "layout_id": grid_code_layout_manifest()["layout_id"],
        "status": "unavailable",
        "note": "",
        "error": None,
        "blockers": [],
        "protocol_params": {},
        "setup_ms": None,
        "psi_ca_execute_ms": None,
        "total_ms": None,
        "wall_ms": None,
        "timings_ms": {},
        "memory_mb": None,
        "peak_rss_mb": None,
        "agreement": None,
        "intersection_count": None,
        "intersection_count_expected": None,
    }


def run_case(n: int, *, report: Any = None) -> dict[str, Any]:
    """执行一条用例并返回记录（失败不抛：如实记进 `status` / `error`）。"""

    record = _blank_record(n)
    t_case = time.perf_counter()
    t0 = time.perf_counter()
    bset = generate_benchmark_sets(n)
    record["setup_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    record.update(
        n_left=bset.n_left,
        n_right=bset.n_right,
        n_left_unique=n,
        n_right_unique=n,
        duplicate_count=int(bset.meta["duplicate_count"]),
        intersection_count_expected=int(bset.meta["intersection_count"]),
    )

    rss_before = _peak_rss_mb()
    try:
        run = run_psi_cardinality(
            bset.left,
            bset.right,
            reference_fn=_plain_intersection_codes,
            report=report,
        )
    except Exception as exc:  # 基线脚本不吞异常：如实记错误，不伪造结果
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["note"] = "用例执行抛出异常（未走到 PsiRunResult 路径）"
        record["peak_rss_mb"] = round(_peak_rss_mb(), 1)
        record["wall_ms"] = round((time.perf_counter() - t_case) * 1000.0, 3)
        return record

    record["wall_ms"] = round((time.perf_counter() - t_case) * 1000.0, 3)
    record["peak_rss_mb"] = round(_peak_rss_mb(), 1)
    record["memory_mb"] = round(max(record["peak_rss_mb"] - rss_before, 0.0), 1)
    record["status"] = run.status
    timings = dict(run.timings_ms)
    record["timings_ms"] = timings
    record["psi_ca_execute_ms"] = timings.get("psi_execute_ms")
    record["total_ms"] = timings.get("total_ms")
    record["agreement"] = run.agreement
    record["intersection_count"] = run.intersection_count
    if run.intersection_count is not None and n:
        record["intersection_ratio_actual"] = round(run.intersection_count / n, 4)
    record["error"] = run.error
    record["blockers"] = list(run.blockers)
    record["receiver_rank"] = run.receiver_rank
    record["world_size"] = run.world_size
    record["protocol_params"] = dict(run.protocol_params)
    record["structure"] = (run.protocol_params or {}).get("structure")
    record["protocol_leak"] = (run.result_policy or {}).get("protocol_leak")
    record["result_semantics"] = run.result_semantics
    record["note"] = "；".join(str(note) for note in run.notes)
    return record


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _write_csv(records: Sequence[Mapping[str, Any]], path: str) -> str:
    """把记录写成 CSV（列见 `_CSV_COLUMNS`；父目录自动创建）。"""

    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CSV_COLUMNS)
        for record in records:
            writer.writerow([_csv_value(record.get(column)) for column in _CSV_COLUMNS])
    return path


def _fmt_num(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return str(value)


def _fmt_bool(value: Any) -> str:
    if value is None:
        return "-"
    return "true" if value else "false"


def _summary(records: Sequence[Mapping[str, Any]]) -> str:
    header = (
        f"{'case':22s}  {'status':12s} {'psi_ms':>9s} {'total_ms':>9s} "
        f"{'rss_MB':>7s} {'agr':>5s}"
    )
    lines = [header]
    for record in records:
        lines.append(
            f"{str(record.get('case', ''))[:22]:22s}  "
            f"{str(record.get('status', '')):12s} "
            f"{_fmt_num(record.get('psi_ca_execute_ms')):>9s} "
            f"{_fmt_num(record.get('total_ms')):>9s} "
            f"{_fmt_num(record.get('memory_mb')):>7s} "
            f"{_fmt_bool(record.get('agreement')):>5s}"
        )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GIS_SPU PSI-CA 性能基线运行器")
    parser.add_argument(
        "--sizes", default=_DEFAULT_SIZES, help="显式规模列表（2 的幂指数，如 10,12,14）"
    )
    parser.add_argument(
        "--json", default="docs/psi_ca_benchmark_baseline.json", help="JSON 输出路径（相对项目根）"
    )
    parser.add_argument(
        "--csv", default="docs/psi_ca_benchmark_baseline.csv", help="CSV 输出路径（相对项目根）"
    )
    parser.add_argument("--no-json", action="store_true", help="不写 JSON")
    parser.add_argument("--no-csv", action="store_true", help="不写 CSV")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    sizes = [int(part) for part in args.sizes.replace(",", " ").split()]
    if not sizes:
        raise SystemExit("--sizes 不能为空（示例：10,12,14）")
    out_json = None if args.no_json else _resolve(args.json)
    out_csv = None if args.no_csv else _resolve(args.csv)

    report = check_psi_ca_capabilities()
    print(
        f"PSI-CA 能力：runnable={report.runnable} version={report.version or '未知'}",
        flush=True,
    )
    if not report.runnable:
        for blocker in report.blockers:
            print(f"  blocker: {blocker}", flush=True)
        print(
            "能力不可用：不写产物、不产出基线文件——避免用 unavailable 行覆盖"
            "仓库里的真实基线（先按 docs/PSI_CA_CAPABILITY.md §8 备好环境再重跑）",
            flush=True,
        )
        return 1

    print(f"PSI-CA 基线：{len(sizes)} 条用例；json={out_json}；csv={out_csv}", flush=True)

    records: list[dict[str, Any]] = []
    total = len(sizes)
    for idx, size in enumerate(sizes, 1):
        n = 2**size
        print(f"[{idx}/{total}] {PSI_CA_PROTOCOL} N={format_n(n)}", flush=True)
        record = run_case(n, report=report)
        records.append(record)
        print(
            f"    status={record['status']} psi_execute_ms={record.get('psi_ca_execute_ms')} "
            f"total_ms={record.get('total_ms')} agreement={record.get('agreement')}",
            flush=True,
        )
        if out_json is not None:
            write_benchmark_json(records, out_json)
    if out_csv is not None:
        _write_csv(records, out_csv)

    ok = sum(1 for record in records if record["status"] == "ok")
    print()
    print(_summary(records))
    print(f"\n汇总：{len(records)} 条；ok={ok}；其余={len(records) - ok}")
    unexpected = [record for record in records if record["status"] != "ok"]
    for record in unexpected:
        print(
            f"  预期外: {record['case']} status={record['status']} "
            f"error={record.get('error')} blockers={record.get('blockers')}"
        )
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
