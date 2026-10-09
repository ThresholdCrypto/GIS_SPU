# -*- coding: utf-8 -*-
"""PSI-Cardinality 后端能力核查（openmined-psi）。

设计边界（刻意的）
------------------
- 本后端只承接 ``CellSetIntersect`` 的计数档（``REVEAL_COUNT``，只出基数）；
- **不实现协议本体**：调用 OpenMined 官方实现（Apache-2.0，PyPI 包
  ``openmined-psi``），本项目只做能力核查与封装；
- 不修改 SPU、不依赖 libpsi：与 ``backends/psi_backend`` 是两条**独立执行
  路径**，泄漏承诺不同（本路径只出计数，libpsi 求交把交集本体交给接收方）；
- 只用 ``RAW`` 数据结构（精确计数）。GCS / BloomFilter 属**近似**档，
  未接入也未实测（接之前不登记）。

API 依据（2026-10-09 从上游源码逐项核对，非推测）
------------------------------------------------
- ``private_set_intersection/python/__init__.py``：``client`` / ``server``
  的类方法与签名（``CreateWithNewKey(reveal_intersection)``、
  ``CreateSetupMessage(fpr, num_client_inputs, inputs, ds)``、
  ``CreateRequest`` / ``ProcessRequest`` / ``GetIntersectionSize``）；
- ``private_set_intersection/cpp/psi_server.cpp``：``fpr`` 对 ``Raw`` 档
  被忽略（注释原文 “This is ignored for the `Raw` datastructure”）；
  ``ProcessRequest`` 拒绝两侧 ``reveal_intersection`` 不一致的请求；
- 发行版：``openmined-psi==2.0.6``（PyPI 仅发布 macOS 与 manylinux 轮子，
  cp39–cp313；本项目在 Linux（WSL2）验证）。

本文件只**探测**环境与 API 形态；"能不能跑"与"跑了对不对"分别在
``runtime`` 与测试里回答。没有可运行环境时如实报 ``runnable=False``，
不填推测值。
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any

#: 后端名（CLI ``--psi-count`` 的取值之一）
PSI_CA_BACKEND = "psi-ca"
#: 上游发行包名与核对过的版本（PyPI 发布版本，非猜测）
PSI_CA_DISTRIBUTION = "openmined-psi"
PSI_CA_PINNED_VERSION = "2.0.6"
PSI_CA_IMPORT_PATH = "private_set_intersection.python"
#: 本后端只接的算子与策略（单一来源；runtime 与测试都读这里）
PSI_CA_SUPPORTED_OPS: tuple[str, ...] = ("CellSetIntersect",)
PSI_CA_SUPPORTED_POLICIES: tuple[str, ...] = ("REVEAL_COUNT",)
#: 数据结构：MVP 只接 RAW（精确）。近似档（GCS / BloomFilter）未接、未实测。
PSI_CA_STRUCTURE = "RAW"

#: 运行时必须存在的模块级属性
_REQUIRED_MODULE_ATTRS: tuple[str, ...] = (
    "client",
    "server",
    "DataStructure",
    "ServerSetup",
    "Request",
    "Response",
)
#: 运行时必须存在的方法（owner, method）
_REQUIRED_METHODS: tuple[tuple[str, str], ...] = (
    ("client", "CreateWithNewKey"),
    ("client", "CreateRequest"),
    ("client", "GetIntersectionSize"),
    ("server", "CreateWithNewKey"),
    ("server", "CreateSetupMessage"),
    ("server", "ProcessRequest"),
)


def import_openmined_psi() -> Any:
    """导入 openmined-psi 的 Python 入口（**唯一导入点**，测试可替换）。

    导入即加载原生扩展（``_openmined_psi`` 共享库），因此"导入成功"本身
    就是一次真实的环境探测；导入失败时把原始异常抛给调用方，由
    ``check_psi_ca_capabilities`` 转成可读阻断项。
    """

    import private_set_intersection.python as psi  # noqa: PLC0415

    return psi


def missing_api(psi: Any) -> tuple[str, ...]:
    """核对上游 API 形态；返回**缺失项**清单（空 = 全部就位）。"""

    problems: list[str] = []
    absent = [name for name in _REQUIRED_MODULE_ATTRS if not hasattr(psi, name)]
    if absent:
        problems.append(f"模块缺少属性 {absent}")
    for owner, method in _REQUIRED_METHODS:
        target = getattr(psi, owner, None)
        if target is not None and not hasattr(target, method):
            problems.append(f"{owner} 缺少方法 {method}")
    data_structure = getattr(psi, "DataStructure", None)
    if data_structure is not None and not hasattr(data_structure, PSI_CA_STRUCTURE):
        problems.append(f"DataStructure 缺少 {PSI_CA_STRUCTURE}")
    return tuple(problems)


@dataclass(frozen=True)
class PsiCaCapabilityReport:
    """PSI-Cardinality 环境核查结论（与 PsiCapabilityReport 一样，只写事实）。"""

    installed: bool
    version: str | None
    runnable: bool
    blockers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    python_version: str = ""
    platform: str = ""
    api_problems: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": PSI_CA_BACKEND,
            "distribution": PSI_CA_DISTRIBUTION,
            "pinned_version": PSI_CA_PINNED_VERSION,
            "import_path": PSI_CA_IMPORT_PATH,
            "installed": self.installed,
            "version": self.version,
            "runnable": self.runnable,
            "structure": PSI_CA_STRUCTURE,
            "supported_ops": list(PSI_CA_SUPPORTED_OPS),
            "supported_policies": list(PSI_CA_SUPPORTED_POLICIES),
            "blockers": list(self.blockers),
            "api_problems": list(self.api_problems),
            "notes": list(self.notes),
            "python_version": self.python_version,
            "platform": self.platform,
        }

    def describe(self) -> str:
        lines = [
            f"PSI-CA capability: {'runnable' if self.runnable else 'blocked'}"
            f"  (backend={PSI_CA_BACKEND}, impl={PSI_CA_DISTRIBUTION},"
            f" installed={self.installed}, version={self.version or '未知'})"
        ]
        lines.append(f"  structure    : {PSI_CA_STRUCTURE}（只接精确计数档）")
        lines.append(
            "  scope        : ops="
            + ", ".join(PSI_CA_SUPPORTED_OPS)
            + "；policies="
            + ", ".join(PSI_CA_SUPPORTED_POLICIES)
        )
        for blocker in self.blockers:
            lines.append(f"  blocker      : {blocker}")
        for note in self.notes:
            lines.append(f"  note         : {note}")
        return "\n".join(lines)


def check_psi_ca_capabilities() -> PsiCaCapabilityReport:
    """探测当前环境的 PSI-Cardinality 能力（可运行性 + API 形态）。"""

    environment = {
        "python_version": platform.python_version(),
        "platform": sys.platform,
    }
    notes = (
        f"实现：{PSI_CA_DISTRIBUTION} {PSI_CA_PINNED_VERSION}（Apache-2.0；"
        "PyPI 只发布 macOS 与 manylinux 轮子，Windows 无轮子）",
        "只用 RAW 数据结构：精确计数（上游源码注明 RAW 忽略 fpr）；"
        "GCS / BloomFilter 近似档未接、未实测",
        "只出基数：交集本体不交给任一方——与 libpsi 求交（把交集本体交给"
        "接收方）是两种不同的泄漏承诺，见 docs/PSI_CA_CAPABILITY.md",
        "不落盘：全程内存 protobuf 消息，不产生临时 CSV",
        "不依赖 SPU / libpsi；与 backends/psi_backend 是两条独立执行路径",
    )
    try:
        psi = import_openmined_psi()
    except Exception as exc:  # ImportError / OSError（缺 so）等一律如实报
        return PsiCaCapabilityReport(
            installed=False,
            version=None,
            runnable=False,
            blockers=(
                f"无法导入 {PSI_CA_IMPORT_PATH}：{type(exc).__name__}: {exc}",
                f"安装：pip install {PSI_CA_DISTRIBUTION}=={PSI_CA_PINNED_VERSION}"
                "（仅 Linux / macOS；Windows 无轮子，需 WSL2 或 Linux 容器）",
            ),
            notes=notes,
            **environment,
        )

    version = getattr(psi, "__version__", None)
    problems = missing_api(psi)
    if problems:
        return PsiCaCapabilityReport(
            installed=True,
            version=version,
            runnable=False,
            blockers=(
                "已安装 openmined-psi，但 API 形态与核对过的 2.0.6 不符："
                + "；".join(problems),
                "不按未核对的 API 执行；请核对上游源码后再更新本后端",
            ),
            notes=notes,
            api_problems=problems,
            **environment,
        )

    extra_notes = notes
    if version is not None and version != PSI_CA_PINNED_VERSION:
        extra_notes = notes + (
            f"注意：已安装版本 {version} 与核对过的 {PSI_CA_PINNED_VERSION} 不一致；"
            "API 形态已逐项探到，但行为差异未经核对",
        )
    return PsiCaCapabilityReport(
        installed=True,
        version=version,
        runnable=True,
        notes=extra_notes,
        **environment,
    )
