#!/usr/bin/env bash
# Phase 8 复核：外部 PSI 执行档（PSI-CA / PI-Sum）在**方案层**的标注。
#
#   bash scripts/verify_external_family_plan.sh
#
# 看什么：三种档位各自的「最终状态表」——Backend 列必须是 默认 PSI / PSI-CA / PI-Sum，
# 与同表 Status 列的 verified / count-only / count-and-sum 一一对应；以及新增的 21 项
# 规划层测试与全量回归。档位不生效时（程序里没有被承接的算子）Backend 列不得改名。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

# 优先用带 spu 的解释器；找不到就退回当前 python（规划层测试不依赖 spu）。
if [ -x /opt/miniconda3/envs/spu311/bin/python ]; then
  PY=/opt/miniconda3/envs/spu311/bin/python
else
  PY="$(command -v python3 || command -v python)"
fi

# SPU / PSI / WSL 的运行时噪音过滤掉，只留可读结论。
flt() { grep -vE 'wsl:|localhost|resource_manager|perfetto|\[info\]|^\[[0-9]{4}-[0-9]{2}-[0-9]{2}|CUDA|^An NVIDIA' || true; }

# 只保留最后一次出现的表头及其后的内容 = 最终状态表（前面还有规划表）。
last_table() { awk '/^Operation/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}'; }

echo "=== 0  解释器与依赖 ==="
"$PY" - <<'PYEOF'
import sys
print("python", sys.version.split()[0])
for name in ("spu", "jax"):
    try:
        mod = __import__(name)
        print("%-4s %s" % (name, getattr(mod, "__version__", "?")))
    except Exception as exc:
        print("%-4s 未安装 (%s)" % (name, type(exc).__name__))
import importlib.util as u
print("openmined-psi 在位:", u.find_spec("private_set_intersection") is not None)
PYEOF

echo
echo "=== 1  Phase 8 规划层测试 ==="
"$PY" -m pytest tests/test_external_psi_planning.py -q -p no:cacheprovider 2>&1 | flt | tail -2

echo
echo "=== 2  三种档位的最终状态表（看 Backend 列） ==="
show() {
  echo "--- geo-secure build $*"
  "$PY" -m geosecure.cli build "$@" 2>&1 | flt | last_table | grep -vE '^$|^---'
}
show examples/route_conflict.py
show examples/conflict_count.py --psi-count psi-ca
show examples/intersection_sum.py --psi-sum pjc --psi-sum-weights route=examples/route_risk_weights.csv

echo
echo "=== 3  全量回归 ==="
"$PY" -m pytest tests/ -q -p no:cacheprovider 2>&1 | flt | tail -2
echo "=== done ==="
