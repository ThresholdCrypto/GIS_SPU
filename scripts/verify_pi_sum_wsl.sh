#!/usr/bin/env bash
# 真机复跑 PI-Sum（交集内求和）档：Bazel 构建 Google private-join-and-compute，
# 再跑 geo-secure psi-sum-check / build，期望结果 (2, 13)。
#
#   用法：  bash scripts/verify_pi_sum_wsl.sh
#
# 前置：  WSL2 / Linux，可联网（clone 上游 + Bazel 拉第三方依赖）。
#
# 先看清一件事（避免误判）：
#   (2, 13) 这个期望来自 **PI-Sum**（Google private-join-and-compute）。
#   它需要 Bazel 构建，但 **不依赖 SPU / libpsi** —— 本档是独立执行路径。
#   所以本脚本里的“Bazel 构建”指的是构建 **PJC**，不是构建 SPU 源码。
#   SPU 官方发 manylinux wheel，本脚本只顺带核查它在不在位（不装、不编译），
#   缺 SPU 时给出 scripts/setup_wsl_spu.sh 的提示。
#
# 可调环境变量：
#   PY           指定解释器（默认优先 /opt/miniconda3/envs/spu311/bin/python）
#   PJC_SRC      上游源码目录（默认 /tmp/pjc）
#   WORK         临时工作目录（默认 /tmp/pjc_setup，存日志）
#   PJC_GIT_URL  上游 clone 地址（HTTPS，失败自动回退 SSH）
#   SKIP_BUILD=1 复用已有构建产物，跳过 bazel build
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PJC_SRC="${PJC_SRC:-/tmp/pjc}"
WORK="${WORK:-/tmp/pjc_setup}"
PJC_GIT_URL="${PJC_GIT_URL:-https://github.com/google/private-join-and-compute.git}"
PJC_GIT_URL_SSH="${PJC_GIT_URL_SSH:-git@github.com:google/private-join-and-compute.git}"
PJC_BIN_DIR="${PJC_BIN_DIR:-$PJC_SRC/bazel-bin/private_join_and_compute}"
CHECK_LOG="$WORK/psi_sum_check.txt"
BUILD_LOG="$WORK/build_intersection_sum.txt"

log()  { printf '\n=== %s ===\n' "$*"; }
warn() { printf '[注意] %s\n' "$*" >&2; }
die()  { printf '\n[失败] %s\n' "$*" >&2; exit 1; }

cd "$PROJECT_DIR"
mkdir -p "$WORK"

# ------------------------------------------------------------------ 解释器
if [ -n "${PY:-}" ]; then
  :
elif [ -x /opt/miniconda3/envs/spu311/bin/python ]; then
  PY=/opt/miniconda3/envs/spu311/bin/python
else
  PY="$(command -v python3 || command -v python || true)"
fi
[ -n "${PY:-}" ] || die "找不到 python3：先跑 scripts/setup_wsl_spu.sh，或用 PY=... 指定解释器。"

log "0/7 环境自检"
"$PY" -c 'import sys; print("python", sys.version.split()[0], "->", sys.executable)'
"$PY" -c 'import geosecure' 2>/dev/null \
  || die "该解释器 import 不到 geosecure：请先在该环境里 pip install -e \$PROJECT_DIR（实际路径 $PROJECT_DIR），或用 PY=... 指定装好本项目的解释器。"
for tool in git curl g++; do command -v "$tool" >/dev/null 2>&1 || warn "缺 $tool"; done

log "1/7 系统依赖（Bazel 构建 C++ 需要 g++）"
if ! command -v g++ >/dev/null 2>&1; then
  warn "缺 g++，尝试 sudo apt-get 安装 build-essential（可能要求输入密码）"
  sudo apt-get update -qq
  sudo apt-get install -y -qq build-essential
fi
g++ --version | head -n1

log "2/7 Bazel（bazelisk 会按上游 .bazelversion 自动取版本）"
# 本脚本的三件事（clone 上游 / 下载 bazelisk / Bazel 拉依赖）全走 HTTPS，
# 很多内网或经代理出网的机器会直接卡在这里，所以先判一次，失败就给出可操作的提示。
if ! curl -sSf -m 12 -o /dev/null https://github.com 2>/dev/null; then
  warn "HTTPS 出网失败：github.com 不可达。"
  warn "若本机需经代理出网，先设置代理再重跑，例如："
  warn "    export http_proxy=http://127.0.0.1:7892 https_proxy=http://127.0.0.1:7892"
  warn "  先跑 env | grep -i proxy 看现有值；WSL 镜像网络模式下 127.0.0.1 即 Windows 回环。"
  die "HTTPS 不可达：先解决出网再重跑（Bazel 同样读 http_proxy/https_proxy）。"
fi
BAZEL="$(command -v bazel || command -v bazelisk || true)"
if [ -z "$BAZEL" ]; then
  mkdir -p "$WORK/bin"
  if [ ! -x "$WORK/bin/bazelisk" ]; then
    curl -fsSL --retry 5 -o "$WORK/bin/bazelisk" \
      https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
    chmod +x "$WORK/bin/bazelisk"
  fi
  export PATH="$WORK/bin:$PATH"
  BAZEL="$WORK/bin/bazelisk"
fi
echo "bazel: $BAZEL"

log "3/7 获取上游源码（google/private-join-and-compute，Apache-2.0）"
if [ -d "$PJC_SRC/.git" ]; then
  echo "复用已有仓库 $PJC_SRC"
else
  git clone --depth 1 "$PJC_GIT_URL" "$PJC_SRC" \
    || { warn "HTTPS clone 失败，回退 SSH"; git clone --depth 1 "$PJC_GIT_URL_SSH" "$PJC_SRC"; }
fi
git -C "$PJC_SRC" log -1 --format='上游 commit %h（%ad）' --date=short
echo "上游 .bazelversion = $(cat "$PJC_SRC/.bazelversion" 2>/dev/null || echo '（无）')"

log "4/7 Bazel 构建（首次要拉 absl/grpc/protobuf，耗时较长，属正常）"
if [ "${SKIP_BUILD:-0}" = "1" ] && [ -x "$PJC_BIN_DIR/client" ] && [ -x "$PJC_BIN_DIR/server" ]; then
  echo "SKIP_BUILD=1 且产物已存在，跳过构建"
else
  cd "$PJC_SRC"
  "$BAZEL" build //private_join_and_compute:all \
    || { warn "//…:all 失败，回退逐目标构建"; "$BAZEL" build //private_join_and_compute:client //private_join_and_compute:server; }
  cd "$PROJECT_DIR"
fi
for bin in client server; do
  [ -x "$PJC_BIN_DIR/$bin" ] || die "构建产物缺失：$PJC_BIN_DIR/$bin（看上面 bazel 的报错）"
done
ls -l "$PJC_BIN_DIR/client" "$PJC_BIN_DIR/server"

log "5/7 PI-Sum 能力核查 —— psi-sum-check（二进制在位 + flag 形态）"
export GIS_SPU_PJC_BIN_DIR="$PJC_BIN_DIR"
set +e
"$PY" -m geosecure.cli psi-sum-check | tee "$CHECK_LOG"
check_rc=$?
set -e
[ "$check_rc" -eq 0 ] || die "psi-sum-check 未通过（runnable=false），看上面的 blocker / api 行或 $CHECK_LOG"
grep -q '"runnable": true' "$CHECK_LOG" || die "psi-sum-check 输出里没有 \"runnable\": true，请人工核对 $CHECK_LOG"

log "6/7 编译 + 真机执行 PI-Sum —— 期望 result = (2, 13)"
set +e
"$PY" -m geosecure.cli build examples/intersection_sum.py \
  --psi-sum pjc \
  --psi-sum-weights route=examples/route_risk_weights.csv \
  | tee "$BUILD_LOG"
build_rc=$?
set -e
[ "$build_rc" -eq 0 ] || die "编译/执行失败（rc=$build_rc）：看上面哪一阶段标了 error"

printf '\n--- 关键行 ---\n'
grep -E -e 'result *:' -e 'policy *:' -e 'count-and-sum' "$BUILD_LOG" || true
grep -q '(2, 13)' "$BUILD_LOG" \
  || die "结果不是 (2, 13)：完整输出见 $BUILD_LOG（真机与登记期望不一致，需要校准）"
echo "结果 (2, 13) 与 docs/PSI_SUM_CAPABILITY.md §8 登记一致。"

log "7/7 顺带核查其余两条 PSI 路径与 SPU（不阻塞主结论）"
if "$PY" -c 'import spu' >/dev/null 2>&1; then
  "$PY" -m geosecure.cli psi-check    || warn "psi-check 未通过"
  "$PY" -m geosecure.cli psi-ca-check || warn "psi-ca-check 未通过"
  "$PY" -m geosecure.cli check        || warn "check 未通过"
else
  warn "该解释器没有 spu：本档（PI-Sum）不需要它，主结论不受影响。"
  warn "要真机跑 SPU 模拟路径，先执行 scripts/setup_wsl_spu.sh（装 wheel，不需要编译 SPU 源码）。"
fi

printf '\n=== 完成 ===\n'
printf '日志：%s\n      %s\n' "$CHECK_LOG" "$BUILD_LOG"
printf '把这两份输出贴回来，即可据此校准真机假设并解除文档里的「未验证项」。\n'
