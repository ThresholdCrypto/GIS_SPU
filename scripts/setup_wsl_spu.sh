#!/usr/bin/env bash
# 在 WSL2 / Linux 上准备真实 SPU 运行环境，并跑通验证。
#
#   用法：  bash scripts/setup_wsl_spu.sh
#
# 前置： Ubuntu（或同类）WSL2 发行版，具备 sudo / apt。
# 说明：  SPU 0.9.5 只发布 cp310 / cp311 的 manylinux wheel，且要求
#        requires-python >=3.10,<3.12。若系统自带 Python 不在该区间，
#        本脚本用 conda 单独装一个 3.11 环境，不动系统 Python。
set -euo pipefail

ENV_NAME="${SPU_ENV_NAME:-spu311}"
CONDA_HOME="${CONDA_HOME:-/opt/miniconda3}"
CONDA_ENV="$CONDA_HOME/envs/$ENV_NAME"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_INDEX="${PY_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
PY_HOST="${PY_HOST:-mirrors.aliyun.com}"
CONDA_MIRROR="${CONDA_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/anaconda}"
WORK="${WORK:-/tmp/spu_setup}"

log() { printf '\n=== %s ===\n' "$*"; }

mkdir -p "$WORK"

# ---------------------------------------------------------------- 系统依赖
log "1/5 系统依赖（libgomp1 是 libspu/libpsi 的 OpenMP 运行时，缺失会 ImportError）"
if ! ldconfig -p 2>/dev/null | grep -q 'libgomp\.so\.1'; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq libgomp1
fi
ldconfig -p | grep -q 'libgomp\.so\.1' && echo "libgomp1 OK"

# ------------------------------------------------------- Python 3.10/3.11
log "2/5 准备 Python 3.10 / 3.11 环境"
if [ -x "$CONDA_ENV/bin/python" ]; then
  echo "复用已有环境 $CONDA_ENV"
else
  if [ ! -x "$CONDA_HOME/bin/conda" ]; then
    echo "安装 Miniconda 到 $CONDA_HOME"
    if [ ! -s "$WORK/miniconda.sh" ]; then
      curl -fsSL --retry 5 -o "$WORK/miniconda.sh" \
        "$CONDA_MIRROR/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    fi
    bash "$WORK/miniconda.sh" -b -p "$CONDA_HOME"
  fi
  cat > "$CONDA_HOME/.condarc" <<EOF
channels:
  - defaults
show_channel_urls: true
default_channels:
  - $CONDA_MIRROR/pkgs/main
  - $CONDA_MIRROR/pkgs/r
custom_channels:
  conda-forge: $CONDA_MIRROR/cloud
EOF
  "$CONDA_HOME/bin/conda" create -y -n "$ENV_NAME" python=3.11
fi
PY="$CONDA_ENV/bin/python"
"$PY" -c 'import sys; v=sys.version_info; assert (3,10)<=v<(3,12), f"需要 Python 3.10/3.11，实为 {sys.version}"; print("python", sys.version.split()[0])'

# ------------------------------------------------------------------ 装依赖
log "3/5 安装 SPU 运行依赖与本体"
mkdir -p "$HOME/.pip"
cat > "$HOME/.pip/pip.conf" <<EOF
[global]
index-url = $PY_INDEX
trusted-host = $PY_HOST
timeout = 120
retries = 5
EOF

"$PY" -m pip install --no-cache-dir -r "$PROJECT_DIR/requirements-spu.txt"
"$PY" -m pip install --no-cache-dir pytest

log "4/5 版本核对"
"$PY" - <<'PYEOF'
import numpy, jax, spu
import spu.libspu as libspu
from spu.utils.simulation import Simulator, sim_jax

print("numpy :", numpy.__version__)
print("jax   :", jax.__version__)
print("spu   :", spu.__version__)
assert hasattr(Simulator, "simple") and callable(sim_jax), "SPU 模拟 API 与预期不符"
print("protocols:", sorted(a for a in dir(libspu.ProtocolKind) if a.isupper() and a.startswith(("ABY", "SEMI", "REF", "CHEETAH", "SECURE"))))
print("fields   :", sorted(a for a in dir(libspu.FieldType) if a.isupper() and a.startswith("FM")))
PYEOF

log "5/5 跑通验证"
cd "$PROJECT_DIR"
"$PY" -m geosecure.cli check
"$PY" -m pytest tests/ -q
for f in examples/route_conflict.py examples/distance_check.py examples/risk_score.py; do
  "$PY" -m geosecure.cli build "$f" | sed -n '/^Result/,/^$/p'
done

printf '\n全部完成。\n'
printf '激活环境： source %s/bin/activate\n' "$CONDA_ENV"