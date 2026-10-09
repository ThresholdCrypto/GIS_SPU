# -*- coding: utf-8 -*-
"""Private Intersection-Sum 后端能力核查（Google private-join-and-compute）。

设计边界（刻意的）
------------------
- 本后端只承接 ``CellSetIntersect`` 的**交集内求和**档
  （``REVEAL_INTERSECTION_SUM``）：一次计算同时给出「交集基数」与
  「交集内关联值之和」；
- **不实现协议本体**：调用 Google 官方实现（Apache-2.0，
  ``google/private-join-and-compute``）；本项目只做能力核查、封装与对拍；
- **不改 SPU、不依赖 libpsi**：与 ``backends/psi_backend``（libpsi 求交，
  协议内部把交集本体交给接收方）、``backends/psi_ca_backend``
  （OpenMined PSI-CA，只交给 client 一个基数）并列的**第三条 PSI 路径**，
  三条路径的泄漏承诺互不相同；
- 上游**没有官方 PyPI 发行包**：能力核查的对象不是"能否 import 一个模块"，
  而是 **Bazel 构建产物的两个可执行文件**在不在、flag 形态对不对。
  PyPI 上的同名包 ``private-join-and-compute`` / ``pjc`` 与本项目无关
  （前者无作者无主页、17 KB 纯 Python 轮子，后者是无关的脚手架工具），
  **不得用来顶替**上游实现。

上游依据（2026-10-09 对 master@950c5e4 逐字核对，非推测）
--------------------------------------------------------
- ``README.md``：功能定义（server 持标识符；client 持标识符 + 非负整数值；
  **client** 得知交集大小与交集内值之和）、honest-but-curious 安全模型、
  泄漏告警（唯一值或过小交集可由 intersection-sum 反推成员）、
  并注明这些缓解措施"本开源库目前未实现"；
- ``private_join_and_compute/client.cc``：flag ``--client_data_file`` /
  ``--port``（默认 ``0.0.0.0:10501``）/ ``--paillier_modulus_size``
  （默认 1536）；失败返回码 1；结果由 ``PrintOutput()`` 打印；
- ``private_join_and_compute/server.cc``：flag ``--server_data_file`` /
  ``--port``；监听语句 ``Server: listening on <port>``；
- ``private_join_and_compute/client_impl.cc``：结果行原文
  ``Client: The intersection size is <N> and the intersection-sum is <S>``；
  和值经 ``ToIntValue()`` 取出（必须落在 int64 内）；
- ``private_join_and_compute/data_util.cc``：server CSV 每行**恰好 1 列**；
  client CSV 每行**恰好 2 列**，第 2 列须为**非负 int64**（负数直接报错）；
  无表头；写侧用 ``"`` 包裹字段、内部引号翻倍；
- ``server.cc`` / ``client.cc``：两侧都用 gRPC
  ``LocalCredentials(LOCAL_TCP)``——**仅本机回环，无 TLS、无身份认证**。

本文件只**探测**环境与二进制形态；"能不能跑"与"跑了对不对"分别在
``runtime`` 与测试里回答。没有可运行环境时如实报 ``runnable=False``，
不填推测值。
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping

#: 后端名（CLI ``--psi-sum`` 的取值之一）
PSI_SUM_BACKEND = "pjc"
#: 上游仓库与**核对过**的提交（本后端所有形态结论都以此为准）
PSI_SUM_UPSTREAM = "google/private-join-and-compute"
PSI_SUM_UPSTREAM_COMMIT = "950c5e4c88d7effe85147beb7856152f7c53394b"
PSI_SUM_UPSTREAM_COMMIT_DATE = "2026-03-09"
PSI_SUM_LICENSE = "Apache-2.0"
#: 上游 ``.bazelversion``（构建工具链版本，非本项目依赖）
PSI_SUM_BAZEL_VERSION = "8.0.1"
#: 上游构建产物的两个可执行文件（bazel-bin/private_join_and_compute/）
PSI_SUM_CLIENT_BINARY = "client"
PSI_SUM_SERVER_BINARY = "server"
PSI_SUM_BINARIES: tuple[str, ...] = (PSI_SUM_CLIENT_BINARY, PSI_SUM_SERVER_BINARY)
#: 二进制目录的指定方式：显式参数优先，其次该环境变量
PSI_SUM_BIN_ENV = "GIS_SPU_PJC_BIN_DIR"
#: 本后端只接的算子与结果策略（单一来源；runtime 与测试都读这里）
PSI_SUM_SUPPORTED_OPS: tuple[str, ...] = ("CellSetIntersect",)
PSI_SUM_SUPPORTED_POLICIES: tuple[str, ...] = ("REVEAL_INTERSECTION_SUM",)
#: 核对过的 flag 形态：缺失即视为"上游形态变了"，不按未核对的形态执行
PSI_SUM_REQUIRED_FLAGS: Mapping[str, tuple[str, ...]] = {
    PSI_SUM_CLIENT_BINARY: (
        "--client_data_file",
        "--port",
        "--paillier_modulus_size",
    ),
    PSI_SUM_SERVER_BINARY: ("--server_data_file", "--port"),
}
#: ``--help`` 探测超时（秒）：只读 flag 清单，跑不出结果也不该拖时间
PSI_SUM_PROBE_TIMEOUT_S = 20.0


def resolve_bin_dir(explicit: str | os.PathLike[str] | None = None) -> str | None:
    """解析上游二进制目录：显式参数 → 环境变量 → None（未指定）。"""

    if explicit is not None:
        return os.fspath(explicit)
    env = os.environ.get(PSI_SUM_BIN_ENV)
    return env or None


def binary_path(bin_dir: str, name: str) -> str:
    """构建产物路径；Windows 上顺带认 ``.exe``（否则原样返回）。"""

    path = os.path.join(bin_dir, name)
    if os.name == "nt" and not os.path.exists(path):
        exe = path + ".exe"
        if os.path.exists(exe):
            return exe
    return path


def probe_binary_flags(
    name: str,
    path: str,
    *,
    timeout: float = PSI_SUM_PROBE_TIMEOUT_S,
) -> tuple[str, ...]:
    """跑 ``<binary> --help`` 核对 flag 形态；返回**未出现**的 flag 清单。

    返回空元组 = flag 形态与本文件登记的核对结论一致。执行失败
    （不存在 / 不可执行 / 超时）时把原始异常抛给调用方，由
    ``check_psi_sum_capabilities`` 转成可读阻断项——不吞掉环境问题。

    注意：absl 的 ``--help`` 正常路径**以退出码 1 结束**（打印后即退），
    因此这里只解析输出文本，不以返回码判成败。
    """

    proc = subprocess.run(
        [path, "--help"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    return tuple(flag for flag in PSI_SUM_REQUIRED_FLAGS.get(name, ()) if flag not in text)


@dataclass(frozen=True)
class PsiSumCapabilityReport:
    """PI-Sum 环境核查结论（只写事实，不写推测）。"""

    installed: bool
    bin_dir: str | None
    runnable: bool
    version: str | None = None
    binaries: Mapping[str, str] = field(default_factory=dict)
    blockers: tuple[str, ...] = ()
    api_problems: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    python_version: str = ""
    platform: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": PSI_SUM_BACKEND,
            "upstream": PSI_SUM_UPSTREAM,
            "upstream_commit": PSI_SUM_UPSTREAM_COMMIT,
            "license": PSI_SUM_LICENSE,
            "bin_env": PSI_SUM_BIN_ENV,
            "bin_dir": self.bin_dir,
            "installed": self.installed,
            "version": self.version,
            "runnable": self.runnable,
            "binaries": dict(self.binaries),
            "supported_ops": list(PSI_SUM_SUPPORTED_OPS),
            "supported_policies": list(PSI_SUM_SUPPORTED_POLICIES),
            "blockers": list(self.blockers),
            "api_problems": list(self.api_problems),
            "notes": list(self.notes),
            "python_version": self.python_version,
            "platform": self.platform,
        }

    def describe(self) -> str:
        lines = [
            f"PI-Sum capability: {'runnable' if self.runnable else 'blocked'}"
            f"  (backend={PSI_SUM_BACKEND}, upstream={PSI_SUM_UPSTREAM}"
            f"@{PSI_SUM_UPSTREAM_COMMIT[:7]}, installed={self.installed})"
        ]
        lines.append(
            "  bin_dir      : " + (self.bin_dir or f"未指定（{PSI_SUM_BIN_ENV} 为空）")
        )
        lines.append(
            "  scope        : ops="
            + ", ".join(PSI_SUM_SUPPORTED_OPS)
            + "；policies="
            + ", ".join(PSI_SUM_SUPPORTED_POLICIES)
        )
        for blocker in self.blockers:
            lines.append(f"  blocker      : {blocker}")
        for problem in self.api_problems:
            lines.append(f"  api          : {problem}")
        for note in self.notes:
            lines.append(f"  note         : {note}")
        return "\n".join(lines)


def _notes() -> tuple[str, ...]:
    """本后端每次核查都要带上的固定披露（事实清单，非评价）。"""

    return (
        f"实现：{PSI_SUM_UPSTREAM}（{PSI_SUM_LICENSE}），核对提交 "
        f"{PSI_SUM_UPSTREAM_COMMIT[:7]}（{PSI_SUM_UPSTREAM_COMMIT_DATE}）",
        "上游无官方 PyPI 包：需 Bazel 构建（" f".bazelversion = {PSI_SUM_BAZEL_VERSION}）；"
        "PyPI 上的同名包 private-join-and-compute / pjc 与本项目无关，不得顶替",
        "结果语义 exact：Paillier 同态求和无噪声，交集大小与和值都是精确值；"
        "但和值须落在 int64 内（上游 ToIntValue），关联值须为非负 int64",
        "只出「基数 + 和」：交集本体不交给任一方——与 libpsi 求交、"
        "PSI-CA 计数档都不同，见 docs/PSI_SUM_CAPABILITY.md",
        "传输面：两侧都用 gRPC LocalCredentials(LOCAL_TCP)，"
        "仅本机回环、无 TLS、无身份认证——跨机部署须自行加通道保护",
        "上游自查泄漏告警：唯一值或过小交集可由 intersection-sum 反推成员；"
        "上游 README 明言其缓解措施（加噪、剪除离群值、过小交集中止）本库未实现",
        "不依赖 SPU / libpsi；与 psi_backend / psi_ca_backend 是三条独立执行路径",
    )


def check_psi_sum_capabilities(
    bin_dir: str | os.PathLike[str] | None = None,
) -> PsiSumCapabilityReport:
    """探测当前环境的 PI-Sum 能力：二进制是否在位 + flag 形态是否一致。"""

    environment = {
        "python_version": platform.python_version(),
        "platform": sys.platform,
    }
    notes = _notes()
    resolved = resolve_bin_dir(bin_dir)
    if resolved is None:
        return PsiSumCapabilityReport(
            installed=False,
            bin_dir=None,
            runnable=False,
            blockers=(
                f"未指定上游二进制目录：请设置环境变量 {PSI_SUM_BIN_ENV}"
                "（或向能力核查显式传入 bin_dir）",
                "获取方式：git clone https://github.com/"
                f"{PSI_SUM_UPSTREAM} && cd {PSI_SUM_UPSTREAM.rsplit('/', 1)[-1]}"
                " && bazel build //private_join_and_compute:all；"
                "产物在 bazel-bin/private_join_and_compute/{client,server}",
                "本环境未能构建或未指定产物目录时，PI-Sum 结果栏位留空，"
                "不以推测值填充",
            ),
            notes=notes,
            **environment,
        )

    if not os.path.isdir(resolved):
        return PsiSumCapabilityReport(
            installed=False,
            bin_dir=resolved,
            runnable=False,
            blockers=(
                f"指定的二进制目录不存在或不是目录：{resolved}",
                f"请把 {PSI_SUM_BIN_ENV} 指向 bazel-bin/private_join_and_compute",
            ),
            notes=notes,
            **environment,
        )

    paths = {name: binary_path(resolved, name) for name in PSI_SUM_BINARIES}
    missing = [name for name in PSI_SUM_BINARIES if not os.path.exists(paths[name])]
    if missing:
        return PsiSumCapabilityReport(
            installed=False,
            bin_dir=resolved,
            runnable=False,
            binaries=paths,
            blockers=(
                f"目录 {resolved} 中缺少构建产物：{missing}；"
                f"期望文件：{sorted(paths.values())}",
                "先完成 bazel build //private_join_and_compute:all 再重试",
            ),
            notes=notes,
            **environment,
        )

    # flag 形态核对：真实执行 --help，逐项比对核对过的 flag（不是猜 API）
    api_problems: list[str] = []
    for name in PSI_SUM_BINARIES:
        try:
            absent = probe_binary_flags(name, paths[name])
        except Exception as exc:
            api_problems.append(
                f"{name} --help 执行失败：{type(exc).__name__}: {exc}"
            )
            continue
        if absent:
            api_problems.append(f"{name} 的 --help 中缺少已核对的 flag {list(absent)}")

    version = PSI_SUM_UPSTREAM_COMMIT[:7]
    if api_problems:
        return PsiSumCapabilityReport(
            installed=True,
            bin_dir=resolved,
            runnable=False,
            version=version,
            binaries=paths,
            api_problems=tuple(api_problems),
            blockers=(
                "已找到构建产物，但形态与核对过的 "
                f"{PSI_SUM_UPSTREAM}@{version} 不符：" + "；".join(api_problems),
                "不按未核对的形态执行；请核对上游源码后再更新本后端",
            ),
            notes=notes,
            **environment,
        )

    extra_notes = notes + (
        f"构建产物的具体提交无法从二进制反查：本后端按 {version} 的 flag / "
        "输出形态核对；若你构建的是别的提交，形态核对仍会逐项跑，"
        "但语义假设未经核对",
    )
    return PsiSumCapabilityReport(
        installed=True,
        bin_dir=resolved,
        runnable=True,
        version=version,
        binaries=paths,
        notes=extra_notes,
        **environment,
    )
