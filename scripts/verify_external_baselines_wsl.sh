#!/usr/bin/env bash
# 外部 PSI 执行档（PSI-CA / PI-Sum）基准基线一键复跑（Phase 9；PI-Sum 档 Phase 10 起默认带通信量 / 内存计量）。
#
#   用法：  bash scripts/verify_external_baselines_wsl.sh
#
# 前置：  WSL / Linux；PSI-CA 需 `openmined-psi==2.0.6`（见 docs/PSI_CA_CAPABILITY.md §8）；
#         PI-Sum 需上游构建产物（见 docs/PSI_SUM_CAPABILITY.md §8，缺省读
#         GIS_SPU_PJC_BIN_DIR，也可 `PJC_BIN_DIR=... 本脚本` 覆盖）。
#
# 产出：  docs/psi_ca_benchmark_baseline.json / .csv、
#         docs/psi_sum_benchmark_baseline.json / .csv（真机读数，不伪造）；
#         末尾把外部两档投影成统一表（写到 /tmp，派生产物不进仓库）。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [ -n "${PY:-}" ]; then
  :
elif [ -x /opt/miniconda3/envs/spu311/bin/python ]; then
  PY=/opt/miniconda3/envs/spu311/bin/python
else
  PY="$(command -v python3 || true)"
fi
[ -n "${PY:-}" ] || { echo "[失败] 找不到 python3；用 PY=... 指定解释器。" >&2; exit 1; }

if [ -n "${PJC_BIN_DIR:-}" ]; then
  export GIS_SPU_PJC_BIN_DIR="${PJC_BIN_DIR}"
fi

log() { printf '\n=== %s ===\n' "$*"; }

log "0/4 环境自检"
"$PY" -c 'import sys; print("python", sys.version.split()[0], "->", sys.executable)'
"$PY" -c 'import geosecure' 2>/dev/null \
  || { echo "[失败] 该解释器 import 不到 geosecure：先 pip install -e \$PROJECT_DIR，或用 PY=... 指定。" >&2; exit 1; }

log "1/4 PSI-CA 基线（openmined-psi 计数档）"
"$PY" tests/benchmarks/benchmark_psi_ca.py

log "2/4 PI-Sum 基线（private-join-and-compute 交集内求和档）"
"$PY" tests/benchmarks/benchmark_psi_sum.py

log "3/4 统一层测试（真实产物上锁死投影与缺口）"
"$PY" -m pytest tests/test_benchmark_schema.py -q

log "4/4 投影成统一表（派生产物写 /tmp，不进仓库）"
"$PY" scripts/unify_benchmark.py \
    docs/psi_ca_benchmark_baseline.json \
    docs/psi_sum_benchmark_baseline.json \
    --json /tmp/psi_external_unified.json

log "完成"
echo "两份基线已更新：docs/psi_ca_benchmark_baseline.json / docs/psi_sum_benchmark_baseline.json"
