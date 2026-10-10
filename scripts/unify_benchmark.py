# -*- coding: utf-8 -*-
"""把两族的 benchmark 基线投影成统一表（Phase 7 / 任务文档 §九）。

    cd GIS_SPU
    python scripts/unify_benchmark.py docs/psi_benchmark_baseline.json \
        --json docs/psi_benchmark_unified.json
    python scripts/unify_benchmark.py docs/mpc_benchmark_baseline.json \
        docs/mpc_comm_baseline.json --json docs/mpc_benchmark_unified.json
    python scripts/unify_benchmark.py docs/psi_benchmark_baseline.json \
        --compare total_time --input-size 8192

本脚本**只读**既有产物：不执行协议、不重跑基线、不改写原文件。
投影口径见 `backends/benchmark_schema.py`（字段来路写在 `FIELD_SEMANTICS`）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backends.benchmark_schema import (  # noqa: E402
    COMMON_FIELDS,
    COMPARABLE_METRICS,
    compare_protocols,
    load_benchmark_file,
    status_counts,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="统一 benchmark metadata 投影器")
    parser.add_argument("sources", nargs="+", help="基线 JSON 路径（可给多份）")
    parser.add_argument("--json", default="", help="统一记录的 JSON 输出路径")
    parser.add_argument("--csv", default="", help="共有字段的 CSV 输出路径（不含 metadata）")
    parser.add_argument(
        "--compare",
        default="",
        help=f"按该共有字段比较协议，可选：{', '.join(COMPARABLE_METRICS)}",
    )
    parser.add_argument("--operation", default=None, help="比较时只取该算子")
    parser.add_argument("--input-size", type=int, default=None, help="比较时只取该规模")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    records = []
    for source in args.sources:
        loaded = load_benchmark_file(source)
        records.extend(loaded)
        counts = status_counts(loaded)
        readable = "，".join(f"{name}={count}" for name, count in counts.items())
        print(f"{source}：{len(loaded)} 条，族 {loaded[0].family}，{readable}", flush=True)

    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="") as handle:
            json.dump([record.to_dict() for record in records], handle, ensure_ascii=False, indent=1)
            handle.write("\n")
        print(f"已写 JSON：{args.json}（{len(records)} 条）", flush=True)

    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(COMMON_FIELDS)
            for record in records:
                writer.writerow([getattr(record, name) for name in COMMON_FIELDS])
        print(f"已写 CSV：{args.csv}（仅共有字段）", flush=True)

    if args.compare:
        compared = compare_protocols(
            records,
            args.compare,
            operation=args.operation,
            input_size=args.input_size,
        )
        print(f"\n比较指标：{compared.metric}；规模：{compared.input_size}", flush=True)
        for row in compared.rows:
            spread = "" if row.spread is None else f"（{row.samples} 次，极差 {row.spread:.3f}）"
            print(f"  {row.value:>12.3f}  {row.key}{spread}", flush=True)
        for key, reason in compared.skipped:
            print(f"  跳过：{key} —— {reason}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
