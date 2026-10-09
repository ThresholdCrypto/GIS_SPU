# -*- coding: utf-8 -*-
"""PSI 性能基线运行器（下一阶段任务 Step 7 / 验收 F）。

在 WSL / Linux（spu311 环境）中运行；libspu 的原生日志会大量写入
stdout/stderr，不影响本脚本产出的 JSON/CSV。建议重定向日志：

    /opt/miniconda3/envs/spu311/bin/python tests/benchmarks/benchmark_psi.py \
        --json docs/psi_benchmark_baseline.json \
        --csv docs/psi_benchmark_baseline.csv \
        > /tmp/bench_run.log 2>&1

模式：
- 默认：标准扫描（四个协议变体 2^10…2^18；KKRT/RR22 到 2^24；ECDH 到 2^20，
  2^22/2^24 记 unavailable）+ 变量矩阵（N=2^12）+ 重复键矩阵；
- `--quick`：2^10 / 2^12 快扫（CI / 冒烟）；
- `--sizes` / `--protocols`：显式指定扫描（只测指定组合，不做 unavailable 占位）；
  NPC 族（`ecdh-npc` / `kkrt-npc`）只能经此路径跑，不进默认标准扫描。

退出码：`0` = 没有"预期外"记录；`1` = 存在预期外记录（含"预期失败的重复键
用例没有失败"这种上游行为变化）。标准扫描里重复键用例**预期**含 error 行
（RR22/KKRT 报 duplicate keys），它们不算预期外。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backends.psi_backend.benchmark import (  # noqa: E402
    BenchmarkCase,
    compare_with_baseline,
    format_summary,
    run_benchmark_cases,
    standard_cases,
    unexpected_records,
)

_PROTOCOL_CHOICES = {
    "ecdh": (("PROTOCOL_ECDH", False),),
    "kkrt": (("PROTOCOL_KKRT", False),),
    "rr22": (("PROTOCOL_RR22", False),),
    "rr22-low": (("PROTOCOL_RR22", True),),
    # NPC 族（显式放行）：不进默认标准扫描，用 `--protocols` 显式跑
    "ecdh-npc": (("PROTOCOL_ECDH_NPC", False),),
    "kkrt-npc": (("PROTOCOL_KKRT_NPC", False),),
}


def _resolve(path: str) -> str:
    """相对路径按项目根解析（脚本从任意 CWD 运行结果一致）。"""

    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GIS_SPU PSI 性能基线运行器")
    parser.add_argument("--quick", action="store_true", help="只跑 2^10/2^12 快扫")
    parser.add_argument(
        "--sizes", default="", help="显式规模列表（2 的幂指数，如 10,12,20）"
    )
    parser.add_argument(
        "--protocols",
        default="",
        help="显式协议列表（ecdh,kkrt,rr22,rr22-low,ecdh-npc,kkrt-npc）",
    )
    parser.add_argument(
        "--json",
        default="docs/psi_benchmark_baseline.json",
        help="JSON 输出路径（相对项目根）",
    )
    parser.add_argument(
        "--csv",
        default="docs/psi_benchmark_baseline.csv",
        help="CSV 输出路径（相对项目根）",
    )
    parser.add_argument("--no-json", action="store_true", help="不写 JSON")
    parser.add_argument("--no-csv", action="store_true", help="不写 CSV")
    parser.add_argument(
        "--compare",
        default="",
        help="与基线 JSON 逐用例对比耗时倍率（非阻断，仅打印；如 docs/psi_benchmark_baseline.json）",
    )
    return parser.parse_args(argv)


def _explicit_cases(args) -> list[BenchmarkCase]:
    sizes = [int(part) for part in args.sizes.replace(",", " ").split()]
    names = [part for part in args.protocols.replace(",", " ").split()]
    names = names or list(_PROTOCOL_CHOICES)
    variants = []
    for name in names:
        if name not in _PROTOCOL_CHOICES:
            raise SystemExit(f"未知协议 {name!r}；可用：{', '.join(_PROTOCOL_CHOICES)}")
        variants.extend(_PROTOCOL_CHOICES[name])
    return [
        BenchmarkCase(protocol=protocol, low_comm_mode=low_comm, n=2**size)
        for size in sizes
        for protocol, low_comm in variants
    ]


def build_cases(args) -> list[BenchmarkCase]:
    if args.sizes or args.protocols:
        if not args.sizes:
            raise SystemExit("--protocols 需要与 --sizes 一起使用（显式扫描模式）")
        return _explicit_cases(args)
    return standard_cases(quick=args.quick)


def main(argv=None) -> int:
    args = parse_args(argv)
    cases = build_cases(args)
    out_json = None if args.no_json else _resolve(args.json)
    out_csv = None if args.no_csv else _resolve(args.csv)

    print(f"PSI 基线：{len(cases)} 条用例；json={out_json}；csv={out_csv}", flush=True)

    def progress(line: str) -> None:
        print(line, flush=True)

    records = run_benchmark_cases(
        cases, out_json=out_json, out_csv=out_csv, progress=progress
    )

    ok = sum(1 for record in records if record["status"] == "ok")
    unavailable = sum(1 for record in records if record["status"] == "unavailable")
    errors = sum(1 for record in records if record["status"] == "error")
    unexpected = unexpected_records(records)

    print()
    print(format_summary(records))
    print(
        f"\n汇总：{len(records)} 条；ok={ok}；unavailable={unavailable}；"
        f"error={errors}；预期外={len(unexpected)}"
    )
    for record in unexpected:
        print(f"  预期外: {record['case']} status={record['status']} error={record.get('error')}")

    if args.compare:
        baseline_path = _resolve(args.compare)
        try:
            with open(baseline_path, encoding="utf-8") as handle:
                baseline_records = json.load(handle)
        except (OSError, ValueError) as exc:
            print(f"\n对比基线失败（不阻断）：{baseline_path}: {exc}")
        else:
            print(f"\n与基线对比：{baseline_path}（仅提示，不阻断；§19）")
            for row in compare_with_baseline(records, baseline_records):
                if row["ratio"] is not None:
                    mark = " [超提示倍率]" if row["flagged"] else ""
                    print(
                        f"  {row['case']}: {row['ratio']:.3f}×"
                        f"（{row['wall_ms']} / {row['baseline_wall_ms']} ms）"
                        f"{mark}{row['reason']}"
                    )
                elif row["reason"]:
                    print(f"  {row['case']}: {row['reason']}")
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
