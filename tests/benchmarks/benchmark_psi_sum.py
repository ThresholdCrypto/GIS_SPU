# -*- coding: utf-8 -*-
"""PI-Sum 性能基线运行器（Phase 9）：private-join-and-compute 交集内求和，真机测量，诚实记录。

在 WSL / Linux（spu311 环境）中运行（先按 docs/PSI_SUM_CAPABILITY.md §8 构建上游）：

    GIS_SPU_PJC_BIN_DIR=/tmp/pjc/bazel-bin/private_join_and_compute \
    /opt/miniconda3/envs/spu311/bin/python tests/benchmarks/benchmark_psi_sum.py \
        --json docs/psi_sum_benchmark_baseline.json \
        --csv docs/psi_sum_benchmark_baseline.csv \
        > /tmp/psi_sum_bench_run.log 2>&1

职责边界
--------
- 复用 libpsi 基线的**确定性用例构造**（`backends.psi_backend.benchmark`）：
  三条 PSI 路径测的是同一批输入（同规模、同交集比例、同层级布局）；
- 关联值规则固定为 `1 + (code % 997)`（确定性；覆盖去重后全部左侧码），
  期望和由运行器按构造重算后写进记录，便于对拍；
- 计量（Phase 10，默认开启，可用 `--no-measure-comm` / `--no-measure-memory`
  关闭）：通信量 = client↔server 回环 TCP 中继逐字节计数（`send_bytes` /
  `recv_bytes` / `total_bytes`；应用层字节，含 gRPC/HTTP2 封装，不含 TCP/IP 头；
  中继多一跳，耗时含该开销）；峰值内存 = 两个子进程各挂 procfs VmHWM 采样探针
  （20 ms 间隔；`client_peak_rss_mb` / `server_peak_rss_mb`，`peak_rss_mb` 取
  较大者）。探针口径随记录登记（`comm_meter` / `memory_probe` / `note`），
  测不到就留空并注明，绝不填 0；
- 执行失败如实记 `status="error"` + 错误原文；没有真正跑过的用例不填数字；
- 能力不可用时**直接失败且不写产物**（退出码 1）——不能用 unavailable 行
  覆盖仓库里的真实基线。

退出码：`0` = 没有预期外记录（全部 ok）；`1` = 存在预期外记录 / 能力不可用。
"""

from __future__ import annotations

import argparse
import csv
import os
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
from backends.psi_sum_backend.capability import check_psi_sum_capabilities  # noqa: E402
from backends.psi_sum_backend.runtime import (  # noqa: E402
    PSI_SUM_PROTOCOL,
    run_psi_intersection_sum,
)
from ir.values import grid_code_layout_manifest  # noqa: E402

#: 默认规模扫描（2 的幂指数；可用 --sizes 覆盖）。
#: 每个用例要拉起 server/client 两个子进程（Paillier 加密），规模不宜大步长。
_DEFAULT_SIZES = "8,10,12"

#: 关联值规则（写进记录，便于对拍时按同一规则重算）
_VALUE_RULE = "1+(code%997)"

#: 默认监听端口基址（逐用例递增，避免前后用例互相干扰）
_PORT_BASE = 10501

#: CSV 输出列（JSON 是主格式；dict 列如 timings_ms / protocol_params 不进 CSV）
_CSV_COLUMNS: tuple[str, ...] = (
    "case",
    "protocol",
    "protocol_leak",
    "result_semantics",
    "upstream",
    "upstream_commit",
    "paillier_modulus_size",
    "value_function",
    "n",
    "n_left",
    "n_right",
    "n_left_unique",
    "n_right_unique",
    "intersection_ratio",
    "duplicate_ratio",
    "duplicate_count",
    "high_bits",
    "level",
    "z",
    "receiver_rank",
    "layout_id",
    "status",
    "setup_ms",
    "pi_sum_execute_ms",
    "total_ms",
    "wall_ms",
    "send_bytes",
    "recv_bytes",
    "total_bytes",
    "peak_rss_mb",
    "client_peak_rss_mb",
    "server_peak_rss_mb",
    "agreement",
    "intersection_count",
    "intersection_count_expected",
    "intersection_sum",
    "intersection_sum_expected",
    "error",
    "note",
)


def _resolve(path: str) -> str:
    """相对路径按项目根解析（脚本从任意 CWD 运行结果一致）。"""

    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


def _value(code: int) -> int:
    """确定性关联值：`1 + (code % 997)`（非负、远小于 int64 上限）。"""

    return 1 + (int(code) % 997)


def _plain_intersection_codes(left: Sequence[int], right: Sequence[int]) -> list[int]:
    """明文参考：交集码（去重排序）；运行时把它折算到「基数 + 和」口径再对拍。"""

    return sorted(set(left) & set(right))


def _blank_record(n: int) -> dict[str, Any]:
    return {
        "case": f"{PSI_SUM_PROTOCOL} N={format_n(n)}",
        "protocol": PSI_SUM_PROTOCOL,
        "protocol_leak": None,
        "result_semantics": None,
        "upstream": None,
        "upstream_commit": None,
        "paillier_modulus_size": None,
        "value_function": _VALUE_RULE,
        "operation": "CellSetIntersect",
        "world_size": None,
        "n": n,
        "n_left": None,
        "n_right": None,
        "n_left_unique": None,
        "n_right_unique": None,
        "intersection_ratio": 0.5,
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
        "pi_sum_execute_ms": None,
        "total_ms": None,
        "wall_ms": None,
        "timings_ms": {},
        "agreement": None,
        "intersection_count": None,
        "intersection_count_expected": None,
        "intersection_sum": None,
        "intersection_sum_expected": None,
        "send_bytes": None,
        "recv_bytes": None,
        "total_bytes": None,
        "comm_meter": None,
        "comm_direction": None,
        "peak_rss_mb": None,
        "client_peak_rss_mb": None,
        "server_peak_rss_mb": None,
        "memory_probe": None,
    }


def run_case(
    n: int,
    *,
    index: int = 0,
    bin_dir: str | None = None,
    report: Any = None,
    measure_communication: bool = True,
    measure_memory: bool = True,
) -> dict[str, Any]:
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
        intersection_sum_expected=sum(
            _value(int(code))
            for code in bset.left[: int(bset.meta["intersection_count"])]
        ),
    )
    left_values = {int(code): _value(int(code)) for code in bset.left}

    try:
        run = run_psi_intersection_sum(
            bset.left,
            bset.right,
            left_values=left_values,
            reference_fn=_plain_intersection_codes,
            report=report,
            bin_dir=bin_dir,
            port=f"127.0.0.1:{_PORT_BASE + index}",
            measure_communication=measure_communication,
            measure_memory=measure_memory,
        )
    except Exception as exc:  # 基线脚本不吞异常：如实记错误，不伪造结果
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["note"] = "用例执行抛出异常（未走到 PsiRunResult 路径）"
        record["wall_ms"] = round((time.perf_counter() - t_case) * 1000.0, 3)
        return record

    record["wall_ms"] = round((time.perf_counter() - t_case) * 1000.0, 3)
    record["status"] = run.status
    timings = dict(run.timings_ms)
    record["timings_ms"] = timings
    record["pi_sum_execute_ms"] = timings.get("pi_sum_execute_ms")
    record["total_ms"] = timings.get("total_ms")
    record["agreement"] = run.agreement
    record["intersection_count"] = run.intersection_count
    if isinstance(run.value, tuple) and len(run.value) == 2:
        record["intersection_sum"] = int(run.value[1])
    record["error"] = run.error
    record["blockers"] = list(run.blockers)
    record["receiver_rank"] = run.receiver_rank
    record["world_size"] = run.world_size
    record["protocol_params"] = dict(run.protocol_params)
    record["upstream"] = (run.protocol_params or {}).get("impl")
    record["upstream_commit"] = (run.protocol_params or {}).get("commit")
    record["paillier_modulus_size"] = (run.protocol_params or {}).get(
        "paillier_modulus_size"
    )
    record["protocol_leak"] = (run.result_policy or {}).get("protocol_leak")
    record["result_semantics"] = run.result_semantics
    record["note"] = "；".join(str(note) for note in run.notes)
    communication = run.communication or {}
    record["send_bytes"] = communication.get("send_bytes")
    record["recv_bytes"] = communication.get("recv_bytes")
    record["total_bytes"] = communication.get("total_bytes")
    record["comm_meter"] = communication.get("meter")
    record["comm_direction"] = communication.get("direction")
    memory = run.memory or {}
    record["peak_rss_mb"] = memory.get("peak_rss_mb")
    record["client_peak_rss_mb"] = memory.get("client_peak_rss_mb")
    record["server_peak_rss_mb"] = memory.get("server_peak_rss_mb")
    record["memory_probe"] = memory.get("probe")
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
        f"{'case':24s}  {'status':12s} {'proto_ms':>9s} {'total_ms':>9s} "
        f"{'count':>6s} {'sum':>7s} {'comm_B':>9s} {'peak_MB':>8s} {'agr':>5s}"
    )
    lines = [header]
    for record in records:
        lines.append(
            f"{str(record.get('case', ''))[:24]:24s}  "
            f"{str(record.get('status', '')):12s} "
            f"{_fmt_num(record.get('pi_sum_execute_ms')):>9s} "
            f"{_fmt_num(record.get('total_ms')):>9s} "
            f"{_fmt_num(record.get('intersection_count')):>6s} "
            f"{_fmt_num(record.get('intersection_sum')):>7s} "
            f"{_fmt_num(record.get('total_bytes')):>9s} "
            f"{_fmt_num(record.get('peak_rss_mb')):>8s} "
            f"{_fmt_bool(record.get('agreement')):>5s}"
        )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GIS_SPU PI-Sum 性能基线运行器")
    parser.add_argument(
        "--sizes", default=_DEFAULT_SIZES, help="显式规模列表（2 的幂指数，如 8,10,12）"
    )
    parser.add_argument(
        "--bin-dir",
        default=None,
        help="上游构建产物目录（缺省读环境变量 GIS_SPU_PJC_BIN_DIR）",
    )
    parser.add_argument(
        "--json", default="docs/psi_sum_benchmark_baseline.json", help="JSON 输出路径（相对项目根）"
    )
    parser.add_argument(
        "--csv", default="docs/psi_sum_benchmark_baseline.csv", help="CSV 输出路径（相对项目根）"
    )
    parser.add_argument(
        "--no-measure-comm",
        dest="measure_comm",
        action="store_false",
        help="关闭通信量计量（默认开启：回环 TCP 中继逐字节计数）",
    )
    parser.add_argument(
        "--no-measure-memory",
        dest="measure_memory",
        action="store_false",
        help="关闭峰值内存计量（默认开启：procfs VmHWM 采样探针）",
    )
    parser.add_argument("--no-json", action="store_true", help="不写 JSON")
    parser.add_argument("--no-csv", action="store_true", help="不写 CSV")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    sizes = [int(part) for part in args.sizes.replace(",", " ").split()]
    if not sizes:
        raise SystemExit("--sizes 不能为空（示例：8,10,12）")
    out_json = None if args.no_json else _resolve(args.json)
    out_csv = None if args.no_csv else _resolve(args.csv)

    report = check_psi_sum_capabilities(args.bin_dir)
    print(
        f"PI-Sum 能力：runnable={report.runnable} bin_dir={report.bin_dir} "
        f"version={report.version or '未知'}",
        flush=True,
    )
    if not report.runnable:
        for blocker in report.blockers:
            print(f"  blocker: {blocker}", flush=True)
        print(
            "能力不可用：不写产物、不产出基线文件——避免用 unavailable 行覆盖"
            "仓库里的真实基线（先按 docs/PSI_SUM_CAPABILITY.md §8 备好上游产物再重跑）",
            flush=True,
        )
        return 1

    print(f"PI-Sum 基线：{len(sizes)} 条用例；json={out_json}；csv={out_csv}", flush=True)

    records: list[dict[str, Any]] = []
    total = len(sizes)
    for idx, size in enumerate(sizes, 1):
        n = 2**size
        print(f"[{idx}/{total}] {PSI_SUM_PROTOCOL} N={format_n(n)}", flush=True)
        record = run_case(
            n,
            index=idx,
            bin_dir=args.bin_dir,
            report=report,
            measure_communication=args.measure_comm,
            measure_memory=args.measure_memory,
        )
        records.append(record)
        print(
            f"    status={record['status']} pi_sum_execute_ms={record.get('pi_sum_execute_ms')} "
            f"total_ms={record.get('total_ms')} agreement={record.get('agreement')}",
            flush=True,
        )
        print(
            f"    comm={record.get('total_bytes')} bytes (send={record.get('send_bytes')},"
            f" recv={record.get('recv_bytes')}) peak_rss={record.get('peak_rss_mb')} MB",
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
