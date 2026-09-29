#!/bin/bash
# 本地调试：按顺序走完 8 个阶段，每步都打印"这一步在看什么"。
#
#   WSL2/Linux:  bash scripts/debug_local.sh
#   Windows:     双击 scripts\debug_local.cmd（它会转到本脚本）
#
# 设计：任一诊断性步骤失败不中断整体，最后统一汇总；只有编译失败会给非零退出码。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

# 优先用带 spu 的解释器；找不到就退回当前 python。
if [ -x /opt/miniconda3/envs/spu311/bin/python ]; then
  PY=/opt/miniconda3/envs/spu311/bin/python
else
  PY="$(command -v python3 || command -v python)"
fi

# SPU 的日志很吵（perfetto / protobuf / resource_manager），过滤掉便于读。
F() { grep -vE "perfetto|resource_manager|\[info\]|^\[[0-9]{4}-[0-9]|CUDA|^An NVIDIA|^wsl:"; }

step() { printf '\n\n########## %s\n' "$*"; }

# 过滤 WSL / SPU / PSI 的运行时噪音，只留可读结论。
flt() { grep -vE 'wsl:|localhost|resource_manager|perfetto|\[info\]|\[[0-9]{4}-[0-9]{2}-[0-9]{2}' || true; }

step "0/7  解释器与依赖版本（先确认环境，再谈结论）"
"$PY" - <<'PYEOF' 2>&1 | flt
import sys
print("python", sys.version.split()[0])
for name in ("spu", "jax", "numpy"):
    try:
        mod = __import__(name)
        print("%-6s %s" % (name, getattr(mod, "__version__", "?")))
    except Exception as exc:
        print("%-6s 未安装 (%s)" % (name, type(exc).__name__))
PYEOF

step "1/7  算子注册表 —— geo-secure ops"
echo "看什么：六个算子各自被规划成什么表征/后端。这是 Planner 的全部规则来源。"
"$PY" -m geosecure.cli ops 2>&1 | flt

step "2/7  SPU/JAX 能力核查 —— geo-secure check"
echo "看什么：runnable 是否为 true；blockers 是否为空。缺 libgomp1 / 非 Linux 会在这里暴露。"
"$PY" -m geosecure.cli check 2>&1 | flt

step "3/7  PSI 能力核查 —— geo-secure psi-check"
echo "看什么：协议与曲线清单、是否 file_io_only。PSI 走的是独立执行路径，不经过 jax.jit。"
"$PY" -m geosecure.cli psi-check 2>&1 | flt

step "4/7  全量测试 —— pytest tests/ -q"
echo "看什么：passed 数应为 456、skipped 应为 0。出现 skip 说明环境缺件被静默放过。"
"$PY" -m pytest tests/ -q --no-header -p no:cacheprovider 2>&1 | flt | tail -4

step "5/7  编译 PSI 族示例（Intersects → PSI）"
echo "看什么：第 8 阶段做真实 PSI 求交；Result 表 Status=verified。"
"$PY" -m geosecure.cli build examples/route_conflict.py 2>&1 | flt

step "6/7  编译 MPC 族示例（DistanceLE → SPU），并打印生成的 JAX 源码"
echo "看什么：第 6 阶段在 SPU 模拟器上真跑，tolerance 0.0 且 err=0.0；--verbose 给出生成的 jnp 代码。"
"$PY" -m geosecure.cli build examples/distance_check.py --verbose 2>&1 | flt

step "7/7  编译 3D 与组合示例（高度带 / 加权打分）"
echo "看什么：vertical_conflict 走 PSI；risk_score 一次覆盖两个算子（WeightedSum + TemporalOverlap）。"
"$PY" -m geosecure.cli build examples/vertical_conflict.py 2>&1 | flt
"$PY" -m geosecure.cli build examples/risk_score.py 2>&1 | flt

printf '\n\n########## 完成\n'
printf '想看单文件的失败报告，自己造一个错误文件再编译，例如：\n'
printf '  printf "from geo_privacy import geo\\n\\ndef f(a,b):\\n    return geo.buffer_zone(a,b)\\n" > /tmp/bad.py\n'
printf '  %s -m geosecure.cli build /tmp/bad.py\n' "$PY"
