# -*- coding: utf-8 -*-
"""MPC 性能基线运行器（P1：预测代价 ↔ 实测代价对账）。

在 WSL / Linux（spu311 环境）中运行；libspu 的原生日志会大量写入
stdout/stderr，不影响本脚本产出的 JSON/CSV。建议重定向日志：

    ~/.spuenv/bin/python tests/benchmarks/benchmark_mpc.py \
        --json docs/mpc_benchmark_baseline.json \
        --csv docs/mpc_benchmark_baseline.csv \
        > ~/bench_mpc.log 2>&1

模式：
- 默认：标准扫描（协议 × 算子 × 环宽 × 规模，见 backends/spu_backend/benchmark.py）；
- `--quick`：只跑小规模主扫描（CI / 冒烟）；
- `--ops` / `--protocols`：只测指定组合；
- `--repeat N`：每条用例重复 N 次，输出统计记录（中位数 / p25 / p75 / 偏差率）。
  拿方差才谈协议排序，见 `docs/MPC_BENCHMARK_PROTOCOL.md` §8。

退出码：`0` = 没有"预期外"记录；`1` = 存在预期外记录（含"ok 却与明文不一致"
这类不该发生的行）。位宽越界（预测 b(K) > 环宽）的用例**不实测**，
以 `unavailable` 如实占位，不算预期外。

三类特殊记录：

- `expect_status="error"`：稳定失败（如 `WeightedSum` × FM32 的环宽下限）；
- `expect_status="deviation"`：**非确定**偏差（`WeightedSum` 的近似除法）——
  `ok` 与 `error` 都算符合预期，只有第三种状态才算预期外；
- `execute=False`：越界占位，`unavailable` + 原因，不填任何数字。
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backends.spu_backend.benchmark import (  # noqa: E402
    MPC_BENCHMARK_OPS,
    format_summary,
    run_benchmark_cases,
    standard_cases,
    unexpected_records,
    write_benchmark_csv,
    write_benchmark_json,
)
from backends.spu_backend import SPU_PROTOCOLS, check_capabilities  # noqa: E402


def _resolve(path: str) -> str:
    """相对路径按项目根解析（脚本从任意 CWD 运行结果一致）。"""

    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GIS_SPU MPC 性能基线运行器")
    parser.add_argument("--quick", action="store_true", help="小规模主扫描（快）")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="每条用例重复次数（>1 时输出统计记录：中位数/四分位/偏差率）",
    )
    parser.add_argument("--ops", default="", help=f"只测算子（逗号分隔）：{MPC_BENCHMARK_OPS}")
    parser.add_argument("--protocols", default="", help=f"只测协议（逗号分隔）：{SPU_PROTOCOLS}")
    parser.add_argument(
        "--json",
        default="docs/mpc_benchmark_baseline.json",
        help="JSON 输出路径（相对项目根）",
    )
    parser.add_argument(
        "--csv",
        default="docs/mpc_benchmark_baseline.csv",
        help="CSV 输出路径（相对项目根）",
    )
    parser.add_argument("--no-json", action="store_true", help="不写 JSON")
    parser.add_argument("--no-csv", action="store_true", help="不写 CSV")
    return parser.parse_args(argv)


def _split(value: str) -> set[str]:
    return {part for part in value.replace(",", " ").split() if part}


def build_cases(args):
    repeats = max(1, int(args.repeat))
    cases = standard_cases(quick=args.quick, repeats=repeats)
    ops = _split(args.ops)
    protocols = _split(args.protocols)
    if ops:
        unknown = ops - set(MPC_BENCHMARK_OPS)
        if unknown:
            raise SystemExit(f"未知算子 {sorted(unknown)}；可用：{MPC_BENCHMARK_OPS}")
        cases = [case for case in cases if case.op in ops]
    if protocols:
        unknown = protocols - set(SPU_PROTOCOLS)
        if unknown:
            raise SystemExit(f"未知协议 {sorted(unknown)}；可用：{SPU_PROTOCOLS}")
        cases = [case for case in cases if case.protocol in protocols]
    return cases


def main(argv=None) -> int:
    args = parse_args(argv)
    cases = build_cases(args)
    if not cases:
        raise SystemExit("筛选后没有用例（检查 --ops / --protocols）")
    out_json = None if args.no_json else _resolve(args.json)
    out_csv = None if args.no_csv else _resolve(args.csv)

    report = check_capabilities()
    repeats = max(1, int(args.repeat))
    print(
        f"MPC 基线：{len(cases)} 条用例 × {repeats} 次；json={out_json}；csv={out_csv}",
        flush=True,
    )
    if not report.runnable:
        print(f"  环境不具备真实 SPU 执行能力：{'; '.join(report.blockers)}", flush=True)

    def progress(record) -> None:
        print(
            f"  {record['case']}: {record['status']}"
            f"  wall={record['wall_ms'] and round(record['wall_ms'], 1)} ms"
            f"  agreement={record['agreement']}",
            flush=True,
        )

    records = run_benchmark_cases(cases, report=report, progress=progress)

    if out_json:
        write_benchmark_json(records, out_json)
    if out_csv:
        write_benchmark_csv(records, out_csv)

    ok = sum(1 for r in records if r["status"] == "ok")
    unavailable = sum(1 for r in records if r["status"] == "unavailable")
    errors = sum(1 for r in records if r["status"] == "error")
    unexpected = unexpected_records(records)

    print()
    print(format_summary(records))
    print(
        f"\n汇总：{len(records)} 条；ok={ok}；unavailable={unavailable}；"
        f"error={errors}；预期外={len(unexpected)}"
    )
    for record in unexpected:
        print(
            f"  预期外: {record['case']} status={record['status']} "
            f"agreement={record.get('agreement')} error={record.get('error')}"
        )
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
