# -*- coding: utf-8 -*-
"""PSI 能力探测：核对当前安装的 SPU 到底提供哪些 PSI 能力。

**本模块不预设任何 SPU API**，全部在运行期探测：
协议清单、曲线清单、入口函数、是否支持内存 link，
以及"本机是否真能跑"。核对结论记录在 `docs/SPI_CAPABILITY.md`。

spu 0.9.5 实测（2026-09 核对）
-----------------------------
- 入口：`spu.psi.psi_execute(config, lctx) -> PsiExecuteReport`
      另有 `spu.psi.ub_psi_execute`（UB/离线缓存模式，本项目未用）
- 枚举中的协议：`PROTOCOL_ECDH / KKRT / RR22 / ECDH_3PC / ECDH_NPC / KKRT_NPC / DP`
- 本链路（两方）可执行协议：同上**去掉 `ECDH_3PC`**——它是三方协议，
  官方实现强制 `WorldSize() == 3`，在本后端必然失败，故预先拦截
- 曲线：`CURVE_25519 / CURVE_FOURQ / CURVE_SM2 / CURVE_SECP256K1 /
        CURVE_25519_ELLIGATOR2`
- 输入输出：**仅 CSV 文件**（`SOURCE_TYPE_FILE_CSV`），无内存张量接口
- link：`spu.libspu.link.create_mem(desc, rank)` 可建进程内多方链路（模拟用）

已知失败模式（实测）
-------------------
- `PROTOCOL_ECDH` 未指定 curve → `RuntimeError: Curve type is not specified.`
  因 `EcdhParams.curve` 默认值是 `CURVE_INVALID_TYPE`。
- `PROTOCOL_KKRT` / `RR22` 不读 curve，给了也不报错。
"""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping

from . import protocol_registry as _registry
from .protocol_registry import (
    RESULT_SEMANTICS,
    RESULT_SEMANTICS_APPROXIMATE,
    RESULT_SEMANTICS_EXACT,
    RESULT_SEMANTICS_NOISY,
    PsiProtocolSpec,
    get_protocol_spec,
)

# --------------------------------------------------------------------------
# 从官方 libpsi.pyi / psi.py 核对的常量（不来自猜测）
# --------------------------------------------------------------------------

#: `spu.psi.PsiProtocol` 的成员（0.9.5 libpsi.pyi 实测）。
#: §12 起登记在 `protocol_registry.PROTOCOL_SPECS`（单一来源），这里只做派生——
#: 两处各自维护名单的漂移风险就此消除。
PSI_PROTOCOLS: tuple[str, ...] = _registry.PSI_PROTOCOL_NAMES

#: 协议与椭圆曲线的关系（三分类；依据 = 上游 psi 仓库源码 + spu 0.9.5 实测）。
#:
#: - "ignored" : 协议不基于椭圆曲线；给了也不读、也不报错（KKRT 族走 OT 扩展）。
#: - "implicit": 协议基于椭圆曲线，但**自带内置默认值**，不指定也能跑。
#:               `PROTOCOL_DP` 属此类：上游 `psi/legacy/dp_psi/dp_psi.h` 的
#:               `RunDpEcdhPsiAlice/Bob(..., CurveType curve = CurveType::CURVE_25519)`
#:               默认 25519。**显式传入的曲线是否在协议内部生效，本版未核对**，
#:               因此既不说"不生效"、也不做覆盖——本项目不把它传进协议。
#: - "required": 必须由调用方显式指定，否则 `Curve type is not specified.`
#:               （`EcdhParams.curve` 的默认值是不可用的哨位 `CURVE_INVALID_TYPE`）。
#:
#: 早期实现把 `PROTOCOL_DP` 与 KKRT/RR22 并列进 `PSI_PROTOCOLS_WITHOUT_CURVE`，
#: CLI 也随之告诉使用者"--psi-curve 不生效"。源码核对后这是**不成立的断言**：
#: DP 是 ECDH 系协议，曲线是它的形参。宁可少说，不可说错。
PSI_CURVE_RELATION: Mapping[str, str] = dict(_registry.PSI_CURVE_RELATION)

#: 不基于椭圆曲线的协议（由 `PSI_CURVE_RELATION` 派生，避免两张表各自漂移）
PSI_PROTOCOLS_WITHOUT_CURVE: tuple[str, ...] = tuple(
    name for name, relation in PSI_CURVE_RELATION.items() if relation == "ignored"
)

#: 必须由调用方显式指定曲线的协议。本项目注入的默认曲线只对这一类生效。
PSI_PROTOCOLS_CURVE_REQUIRED: tuple[str, ...] = tuple(
    name for name, relation in PSI_CURVE_RELATION.items() if relation == "required"
)

#: 各协议要求的参与方数量（实测结论，非推测）。
#:
#: 枚举里存在**不等于**当前运行期可用：`PROTOCOL_ECDH_3PC` 是三方协议，
#: 官方实现强制 `lctx_->WorldSize() == 3`，而本后端的进程内链路固定两方
#: （见 `runtime.run_psi_intersection`），因此它必然失败：
#:
#:     [Enforce fail at external/psi~/psi/legacy/memory_psi.cc:44]
#:     lctx_->WorldSize() == 3. psi_type:4, only three parties supported, got 2
#:
#: 登记这张表的目的，是让它在**进入协议之前**被拦下并给出可读原因，
#: 而不是抛一段 C++ 栈回溯。这与 `ENUM_SENTINELS` 是同一条纪律：
#: "看起来有这个选项"不等于"能用"。
PSI_PROTOCOL_WORLD_SIZE: Mapping[str, int] = dict(_registry.PSI_PROTOCOL_WORLD_SIZE)

#: 本后端的进程内链路固定两方（`CellSetIntersect` 等算子均为两方求交）
PSI_RUNTIME_WORLD_SIZE = 2

#: 结果**故意带噪**的协议：能跑，但不保证与明文一致。
#:
#: 这是与世界大小**并列而不同**的第二类披露：
#:   世界大小回答"能不能跑"，这张表回答"跑出来准不准"。
#: 少任何一边，使用者都会拿一个"跑得起来但结果不可信"的协议去做冲突判定。
#:
#: `PROTOCOL_DP` 是差分隐私 PSI（DP-PSI）。上游 `psi/legacy/dp_psi/dp_psi.h` 的
#: `DpPsiOptions(bob_p=0.9, epsilon=3.0)` 由 ε 推出
#:     p2 = e^ε / (1 + e^ε) ≈ 0.953（Alice 子采样）
#:     q  = 1 - p2           ≈ 0.047（Alice 上采样：往交集中注入假元素）
#: 叠加 Bob 侧 0.9 的子采样。噪声是协议的**设计目标**，不是缺陷。
#:
#: 原生库自己打出了这些值（`quiet=False`，2026-09-28 实测）：
#:     LEGACY PSI config: {"psi_type":"DP_PSI_2PC", ...,
#:                         "dppsi_params":{"bob_sub_sampling":0.9,"epsilon":3}}
#:     [dp_psi.h:37] DpPsiOptions p1:0.9 epsilon:3 p2:0.9525741268224333,
#:                                       q:0.047425873177566746
#: 下面两行解释了为什么注入是**间歇**的——这一次 q 抽样抽到 0 个：
#:     [dp_psi.cc:49] sample bernoulli_distribution: 0.047425873177566746
#:     [dp_psi.cc:60] bernoulli_items_idx:0 ratio:0
#:
#: 实测（本机 spu 0.9.5，2026-09），两个方向都观测到，且**都是间歇性的**：
#:   1) 交集本体被**注入非成员**：A=[11,22,33]、B=[22,33,44]（真交集 {22,33}），
#:      `CellSetIntersect` 跑 10 次里有 1 次返回 intersection=(11, 22, 33)——
#:      `11` 根本不在 B 里；其余 9 次返回正确的 (22, 33)；
#:   2) `Intersects` **漏报真实冲突**，且漏报率随真交集大小急剧变化：
#:      真交集 2 个元素 → 30 次里 3~4 次给出 False（约 10%）；
#:      真交集 15 个元素 → 20 次里 0 次（交集一大就几乎不会漏空）。
#:      同批对照 PROTOCOL_ECDH 5/5、PROTOCOL_KKRT 12/12 均为 True。
#: 另注：DP 下官方报告的 `intersection_unique_count` 恒为 0（ECDH 同数据为 2），
#: 读 JSON 里的计数时不能把它当作交集基数。
#: 对禁飞区判定这是**安全事故级别**的语义变化——既可能把不冲突判成冲突，
#: 也可能把冲突判成不冲突；而且**小交集（恰恰是偶发冲突、最需要警惕的那一类）
#: 正是更容易被漏报的一类**。必须在选用时就说明，而不是等结果对不上再解释。
PSI_PROTOCOLS_WITH_NOISE: tuple[str, ...] = _registry.PSI_PROTOCOLS_WITH_NOISE

#: 显式放行协议（不进候选/建议清单；显式选择可通过编译期）。
#: 单一来源 `protocol_registry.PSI_PROTOCOLS_EXPLICIT_ONLY`，与候选清单互斥。
PSI_PROTOCOLS_EXPLICIT_ONLY: tuple[str, ...] = _registry.PSI_PROTOCOLS_EXPLICIT_ONLY

#: `spu.psi.EllipticCurveType` 的成员（0.9.5 libpsi.pyi 实测）
PSI_CURVES: tuple[str, ...] = (
    "CURVE_25519",
    "CURVE_FOURQ",
    "CURVE_SM2",
    "CURVE_SECP256K1",
    "CURVE_25519_ELLIGATOR2",
)

#: 默认协议：ECDH 是两方、无需预处理、最适合低频地理围栏判定
PSI_DEFAULT_PROTOCOL = "PROTOCOL_ECDH"

#: 默认曲线。选 SM2 而非 25519：地理数据属涉密测绘场景，
#: 国产商用密码算法（SM2/SM3/SM4）是合规首选。
PSI_DEFAULT_CURVE = "CURVE_SM2"

#: 本项目交给 PSI 后端执行的算子（与 planner 注册表 PSI 族一致）
PSI_OPS: tuple[str, ...] = ("Intersects", "Contains", "CellSetIntersect")

#: PSI 只支持两方（本项目使用的 ECDH / KKRT / RR22 均为 2PC）

#: 哨位成员（存在但不可用）：探测时排除。
#: `PROTOCOL_UNSPECIFIED` / `CURVE_INVALID_TYPE` 是 pybind11 枚举的默认占位值，
#: 好像"有这个选项"，但传进去只会报 `Curve type is not specified.`。
#: 把它们当成可用协议/曲线会误导使用者。
ENUM_SENTINELS: tuple[str, ...] = (
    "PROTOCOL_UNSPECIFIED",
    "CURVE_INVALID_TYPE",
    "SOURCE_TYPE_UNSPECIFIED",
    "JOIN_TYPE_UNSPECIFIED",
)
PSI_MIN_WORLD_SIZE = 2


@dataclass
class PsiCapabilityReport:
    """PSI 能力探测结果。"""

    installed: bool = False
    version: str | None = None
    prefix: str | None = None
    libpsi_loadable: bool = False
    psi_execute_available: bool = False
    create_mem_available: bool = False
    protocols: tuple[str, ...] = ()
    curves: tuple[str, ...] = ()
    source_types: tuple[str, ...] = ()
    join_types: tuple[str, ...] = ()
    file_io_only: bool = True
    runnable: bool = False
    #: RR22 参数能力探测（PROTOCOL_RR22 / Rr22Rarams / rr22_params 是否齐备）
    rr22_params: Mapping[str, Any] = field(default_factory=dict)
    blockers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    error: str | None = None

    @property
    def supported_ops(self) -> tuple[str, ...]:
        return PSI_OPS if self.runnable else ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "version": self.version,
            "prefix": self.prefix,
            "libpsi_loadable": self.libpsi_loadable,
            "psi_execute_available": self.psi_execute_available,
            "create_mem_available": self.create_mem_available,
            "protocols": list(self.protocols),
            "curves": list(self.curves),
            "source_types": list(self.source_types),
            "join_types": list(self.join_types),
            "file_io_only": self.file_io_only,
            "runnable": self.runnable,
            "rr22_params": dict(self.rr22_params),
            "blockers": list(self.blockers),
            "notes": list(self.notes),
            "error": self.error,
            "supported_ops": list(self.supported_ops),
        }


def _enum_members(obj: Any, *, skip: tuple[str, ...] = ()) -> tuple[str, ...]:
    """列出某个枚举类的成员名。

    注意：官方 `PsiProtocol` / `EllipticCurveType` 是 **pybind11 枚举**，
    其成员不是 `int` 实例（`isinstance(v, int) is False`），但带 `.value`。
    早期实现只判 `isinstance(value, int)`，导致协议/曲线清单**静默为空**——
    进而使"默认协议是否可用"这类判据失去依据。这里两种形态都接受。
    """

    out = []
    for name in dir(obj):
        if name.startswith("_") or name in skip:
            continue
        value = getattr(obj, name)
        if callable(value):
            continue
        if isinstance(value, int):
            out.append(name)
            continue
        inner = getattr(value, "value", None)
        if isinstance(inner, int):
            out.append(name)
    return tuple(out)


def probe_rr22_params() -> dict[str, Any]:
    """探测 RR22 的参数能力（不许把"没有参数类"当成"参数可用"）。

    三个结论都来自**实际构造**，不是名字比对：

    1. `PsiProtocol.PROTOCOL_RR22` 是否存在；
    2. `psi.Rr22Rarams` 是否存在，且 `low_comm_mode=True/False` 都能构造；
    3. `PsiProtocolConfig(...)` 是否接受 `rr22_params=` 关键字。

    `runnable` 表示"RR22 的参数路径完整可用"（三者全真）。缺任一环时
    如实置 False 并在 `notes` 里说明——RR22 本体或许仍能按内部默认配置运行，
    但"参数可以不完整"这件事必须让使用者看见。
    """

    probe: dict[str, Any] = {
        "protocol": "PROTOCOL_RR22",
        "protocol_present": False,
        "params": {"rr22_params": False, "low_comm_mode": False},
        "runnable": False,
        "notes": [],
    }

    try:
        from spu import psi
    except Exception as exc:
        probe["notes"].append(f"spu.psi 不可导入：{type(exc).__name__}: {exc}")
        return probe

    probe["protocol_present"] = hasattr(psi.PsiProtocol, "PROTOCOL_RR22")

    rr22_cls = getattr(psi, "Rr22Rarams", None)
    if rr22_cls is None:
        probe["notes"].append("spu.psi 无 Rr22Rarams：RR22 参数能力不完整")
        return probe

    low_comm_mode_obj = None
    try:
        rr22_cls(low_comm_mode=False)
        low_comm_mode_obj = rr22_cls(low_comm_mode=True)
        probe["params"]["low_comm_mode"] = True
    except Exception as exc:
        probe["notes"].append(
            f"Rr22Rarams(low_comm_mode=...) 构造失败：{type(exc).__name__}: {exc}"
        )

    if probe["protocol_present"] and low_comm_mode_obj is not None:
        try:
            psi.PsiProtocolConfig(
                protocol=psi.PsiProtocol.PROTOCOL_RR22,
                receiver_rank=0,
                broadcast_result=False,
                rr22_params=low_comm_mode_obj,
            )
            probe["params"]["rr22_params"] = True
        except TypeError as exc:
            probe["notes"].append(f"PsiProtocolConfig 不接受 rr22_params 关键字：{exc}")
        except Exception as exc:
            probe["notes"].append(
                "PsiProtocolConfig(..., rr22_params=...) 构造失败："
                f"{type(exc).__name__}: {exc}"
            )

    probe["runnable"] = bool(
        probe["protocol_present"]
        and probe["params"]["rr22_params"]
        and probe["params"]["low_comm_mode"]
    )
    return probe


def check_psi_capabilities() -> PsiCapabilityReport:
    """探测当前环境是否具备真实 PSI 执行能力。"""

    report = PsiCapabilityReport()

    # ---------------- 1. spu 本体 ----------------
    try:
        spu = importlib.import_module("spu")
    except Exception as exc:
        report.error = f"spu 未安装：{type(exc).__name__}: {exc}"
        report.blockers = report.blockers + ("spu 未安装",)
        return report

    report.installed = True
    report.version = getattr(spu, "__version__", None)
    report.prefix = getattr(spu, "__file__", None)

    # ---------------- 2. libpsi 原生库 ----------------
    try:
        importlib.import_module("spu.libpsi")
        report.libpsi_loadable = True
    except Exception as exc:
        report.error = f"spu.libpsi 无法加载：{type(exc).__name__}: {exc}"
        report.blockers = report.blockers + (f"libpsi 无法加载：{exc}",)

    # ---------------- 3. 官方入口 ----------------
    try:
        psi = importlib.import_module("spu.psi")
    except Exception as exc:
        report.blockers = report.blockers + (f"spu.psi 无法导入：{exc}",)
        return report

    report.psi_execute_available = hasattr(psi, "psi_execute")
    if not report.psi_execute_available:
        report.blockers = report.blockers + (
            "spu.psi 缺少 psi_execute；当前 SPU 版本与预期 API 不一致",
        )

    for field_name, enum_attr in (
        ("protocols", "PsiProtocol"),
        ("curves", "EllipticCurveType"),
        ("source_types", "SourceType"),
        ("join_types", "ResultJoinType"),
    ):
        enum_obj = getattr(psi, enum_attr, None)
        if enum_obj is not None:
            setattr(
                report,
                field_name,
                _enum_members(enum_obj, skip=("name", "value") + ENUM_SENTINELS),
            )

    # ---------------- 3.5 RR22 参数能力 ----------------
    # "协议在枚举里"不等于"参数类存在"。结论来自实际构造（probe_rr22_params）；
    # 不完整时如实披露：参数注入会失败，而不是被静默忽略。
    report.rr22_params = probe_rr22_params()
    if report.rr22_params and not report.rr22_params.get("runnable"):
        report.notes = report.notes + (
            "RR22 参数能力不完整："
            + "；".join(report.rr22_params.get("notes") or ("原因未记录",))
            + "（RR22 仍可按协议内部默认配置运行，但 low_comm_mode 无法注入）",
        )

    # ---------------- 4. 内存 link（模拟必需） ----------------
    try:
        libspu = importlib.import_module("spu.libspu")
        link = getattr(libspu, "link", None)
        report.create_mem_available = link is not None and hasattr(link, "create_mem")
    except Exception:
        report.create_mem_available = False

    if not report.create_mem_available:
        report.blockers = report.blockers + (
            "spu.libspu.link.create_mem 不可用，无法在进程内建立多方链路",
        )

    # ---------------- 5. 平台与 Python ----------------
    major, minor = sys.version_info[:2]
    if platform.system() not in ("Linux", "Darwin"):
        report.blockers = report.blockers + (
            f"{platform.system()} 平台无 libpsi 原生库；需 WSL2 / Linux",
        )
    if not (major == 3 and minor in (10, 11)):
        report.blockers = report.blockers + (
            f"Python {platform.python_version()} 不在 spu 支持区间（>=3.10,<3.12）",
        )

    # ---------------- 6. IO 形态 ----------------
    report.file_io_only = report.source_types == ("SOURCE_TYPE_FILE_CSV",) or (
        "SOURCE_TYPE_FILE_CSV" in report.source_types
        and "SOURCE_TYPE_MEMORY" not in report.source_types
    )
    if report.file_io_only:
        report.notes = report.notes + (
            "PSI 当前只提供 CSV 文件接口，无内存张量接口；"
            "本后端用临时工作目录承载输入输出",
        )

    report.runnable = not report.blockers

    # ---------------- 7. 两方链路下的可执行性 ----------------
    # 枚举清单是"有什么"，不是"能跑什么"。三方协议在本后端的两方链路上
    # 必然失败（实测 C++ Enforce fail），必须在能力报告里就说清楚，
    # 否则使用者会照着一个"存在但跑不起来"的名字去配。
    two_party = protocols_runnable_here()
    if len(two_party) != len(report.protocols):
        excluded = tuple(n for n in report.protocols if n not in two_party)
        report.notes = report.notes + (
            f"以下协议在当前 {PSI_RUNTIME_WORLD_SIZE} 方链路上不可执行（参与方数量不符）："
            + ", ".join(excluded)
            + f"；可执行协议：{', '.join(two_party) or '（无）'}",
        )

    if PSI_DEFAULT_PROTOCOL not in report.protocols:
        report.notes = report.notes + (
            f"默认协议 {PSI_DEFAULT_PROTOCOL} 不在本版本协议清单中，"
            f"可用：{', '.join(report.protocols) or '（未探测到）'}",
        )
    if PSI_DEFAULT_CURVE not in report.curves:
        report.notes = report.notes + (
            f"默认曲线 {PSI_DEFAULT_CURVE} 不在本版本曲线清单中，"
            f"可用：{', '.join(report.curves) or '（未探测到）'}",
        )

    # ---------------- 8. 结果是否精确 ----------------
    # 与"能不能跑"分开披露：DP-PSI 跑得起来，但结果带噪、会漏报真实冲突。
    noisy = tuple(n for n in report.protocols if n in PSI_PROTOCOLS_WITH_NOISE)
    if noisy:
        report.notes = report.notes + (
            f"以下协议结果**带噪**，不保证与明文一致、不可用于一致性验证："
            + ", ".join(noisy)
            + "（差分隐私 PSI：交集中会注入假元素/丢弃真元素）",
        )

    return report


def normalize_psi_protocol(protocol: str) -> str:
    """把用户写法规范化到官方协议枚举名。"""

    text = str(protocol).upper().strip()
    if not text.startswith("PROTOCOL_"):
        text = f"PROTOCOL_{text}"
    if text in PSI_PROTOCOLS:
        return text
    raise ValueError(
        f"未知 PSI 协议 {protocol!r}；spu 0.9.5 支持 {PSI_PROTOCOLS}"
    )


def normalize_curve(curve: str) -> str:
    """把用户写法规范化到官方曲线枚举名。"""

    text = str(curve).upper().strip()
    if not text.startswith("CURVE_"):
        text = f"CURVE_{text}"
    if text in PSI_CURVES:
        return text
    raise ValueError(
        f"未知椭圆曲线 {curve!r}；spu 0.9.5 支持 {PSI_CURVES}"
    )


def protocol_needs_curve(protocol: str) -> bool:
    """该协议是否**必须**由调用方显式指定曲线。

    只对 `required` 一类返回 True（ECDH 族不指定必有
    `Curve type is not specified.`）。`implicit` 一类（`PROTOCOL_DP`，
    自带默认曲线）返回 False——本项目因此不把曲线传进协议。
    这**不等于**"曲线不生效"；两者的区别见 `PSI_CURVE_RELATION` 的说明。
    """

    return normalize_psi_protocol(protocol) in PSI_PROTOCOLS_CURVE_REQUIRED


def psi_curve_relation(protocol: str) -> str:
    """该协议与椭圆曲线的关系：`required` / `ignored` / `implicit`。

    未登记的协议按 `required` 处理：宁可多要一条曲线，也不要静默漏配。
    调用方据此决定**怎么说**（例如 CLI 的提示语），而不是据此猜协议行为。
    """

    return PSI_CURVE_RELATION.get(normalize_psi_protocol(protocol), "required")


def protocol_result_semantics(protocol: str) -> str:
    """协议结果语义档：exact / approximate / noisy（任务书 §11）。

    单一来源是 `protocol_registry.PROTOCOL_SPECS`；调用方按字符串判档，
    将来新增近似 / 带噪协议时不需要再写协议专用的 if/else。
    """

    return get_protocol_spec(normalize_psi_protocol(protocol)).result_semantics


def psi_protocol_spec(protocol: str) -> PsiProtocolSpec:
    """取协议的完整元数据（名字归一化后从 protocol_registry 读取）。"""

    return get_protocol_spec(normalize_psi_protocol(protocol))


def protocol_is_exact(protocol: str) -> bool:
    """该协议是否给出**精确**交集（不向结果注入噪声）。

    返回 False 的协议仍可执行，但其结果**不能**拿来做与明文比对的一致性验证，
    因此调用方不得据此报"已验证"。实现已改为读 `result_semantics`
    （exact / approximate / noisy，单一来源 protocol_registry）。
    """

    return protocol_result_semantics(protocol) == RESULT_SEMANTICS_EXACT


def psi_protocol_is_explicit_only(protocol: str) -> bool:
    """该协议是否属"显式放行"类：不进候选/建议清单，但显式选择可通过编译期。

    单一来源是 `protocol_registry.PSI_PROTOCOLS_EXPLICIT_ONLY`；放行判定与
    披露文案由 `planner.registry.validate_protocol_for_operation` 使用。
    """

    return normalize_psi_protocol(protocol) in PSI_PROTOCOLS_EXPLICIT_ONLY


def protocol_world_size(protocol: str) -> int:
    """该协议要求的参与方数量（未登记项按本后端的两方链路处理）。"""

    return PSI_PROTOCOL_WORLD_SIZE.get(
        normalize_psi_protocol(protocol), PSI_RUNTIME_WORLD_SIZE
    )


def protocols_runnable_here() -> tuple[str, ...]:
    """在当前两方链路下**真正可执行**的协议清单。

    `PSI_PROTOCOLS` 回答的是"官方枚举里有什么"；
    本函数回答的是"在当前运行期能跑什么"。二者不是一回事：
    前者包含 `PROTOCOL_ECDH_3PC`，后者不含。
    """

    return tuple(
        name
        for name in PSI_PROTOCOLS
        if protocol_world_size(name) <= PSI_RUNTIME_WORLD_SIZE
    )


def runnable_protocols_hint() -> str:
    """把可执行协议渲染成一行提示，给带噪的那些打 `*` 并加注。

    这句话出现在"你选的协议跑不了，换一个"的位置上。若里面混着一个结果带噪的
    协议而不加标记，就等于把 DP 当成 ECDH 的等价替代推荐给使用者——
    "能跑"和"能替"是两件事。
    """

    names = protocols_runnable_here()
    rendered = ", ".join(
        f"{name}*" if name in PSI_PROTOCOLS_WITH_NOISE else name for name in names
    )
    if any(name in PSI_PROTOCOLS_WITH_NOISE for name in names):
        rendered = f"{rendered}（带 * 者为差分隐私协议，结果带噪）"
    return rendered


def resolve_psi_protocol(
    protocol: str | None = None, curve: str | None = None
) -> tuple[str, str | None]:
    """把可选的用户输入解析成 (协议名, 曲线名)。

    - 两者都缺省时落到 `PSI_DEFAULT_PROTOCOL` / `PSI_DEFAULT_CURVE`；
    - 只有 `required` 一类（ECDH 族）会带上曲线；`ignored`（不基于椭圆曲线）
      与 `implicit`（`PROTOCOL_DP`，自带默认曲线）都返回 `None`。
      刻意不保留一个本后端并未传下去的曲线值——留着会让调用方以为它生效了；
    - 名称非法时抛 `ValueError`，消息里带可用清单。**曲线名哪怕用不上也照样校验**，
      否则拼错的曲线会被静默忽略。
    """

    name = normalize_psi_protocol(protocol or PSI_DEFAULT_PROTOCOL)
    chosen = normalize_curve(curve) if curve else None
    if not protocol_needs_curve(name):
        return name, None
    return name, chosen or PSI_DEFAULT_CURVE


# --------------------------------------------------------------------------
# 协议级能力：把"PSI 能不能跑"拆成三层（后端 / 协议 / 参数）
#
# 任务书 §2 的原话：不要再用一个总的 runnable 表示所有层次。
# "PSI 总体 runnable = True" 不等于 "RR22 + low_comm_mode" 这条具体路径可用。
# --------------------------------------------------------------------------

#: 已登记的协议参数键。未登记键视为拼写错误直接报错——
#: 静默忽略的代价是"设了但没生效"，错误不会浮出来，只会算错。
PSI_PARAM_KEYS: tuple[str, ...] = (
    "receiver_rank",
    "broadcast_result",
    "low_comm_mode",
)


@dataclass(frozen=True)
class PsiParamValidation:
    """协议参数的静态（不依赖环境的）校验结果。"""

    ok: bool
    protocol: str
    problems: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "protocol": self.protocol,
            "problems": list(self.problems),
            "notes": list(self.notes),
        }


def validate_psi_protocol_params(
    protocol: str, protocol_params: Mapping[str, Any] | None
) -> PsiParamValidation:
    """参数级静态校验：非法值在**编译期**失败，而不是等到 Runtime。

    规则（两方链路）：
    - `receiver_rank` ∈ {0, 1}，且必须是真 int（bool 不算）；
    - `broadcast_result` / `low_comm_mode` 必须是 bool；
    - 未登记的键直接报错；
    - `low_comm_mode=True` 但协议不是 RR22：不算非法（协议族外的参数），
      记为 note，由调用方决定如何披露（CLI / Runtime 都会如实说明）。
    """

    try:
        name = normalize_psi_protocol(protocol)
    except ValueError as exc:
        return PsiParamValidation(
            ok=False, protocol=str(protocol), problems=(str(exc),)
        )

    params = dict(protocol_params or {})
    problems: list[str] = []
    notes: list[str] = []

    unknown = sorted(set(params) - set(PSI_PARAM_KEYS))
    if unknown:
        problems.append(
            f"未登记的协议参数 {unknown}；已登记：{list(PSI_PARAM_KEYS)}"
            "（若确需新参数，请先登记并补测试）"
        )

    if "receiver_rank" in params:
        rank = params["receiver_rank"]
        if isinstance(rank, bool) or not isinstance(rank, int):
            problems.append(f"receiver_rank 必须是整数 0 或 1，实得 {rank!r}")
        elif rank not in (0, 1):
            problems.append(f"receiver_rank 必须是 0 或 1（两方链路），实得 {rank}")

    for key in ("broadcast_result", "low_comm_mode"):
        if key in params and not isinstance(params[key], bool):
            problems.append(f"{key} 必须是布尔值，实得 {params[key]!r}")

    if params.get("low_comm_mode") is True and name != "PROTOCOL_RR22":
        notes.append(
            f"low_comm_mode=True 但协议是 {name}：该参数不会注入（协议族外），"
            "Runtime 会如实加注"
        )

    return PsiParamValidation(
        ok=not problems, protocol=name, problems=tuple(problems), notes=tuple(notes)
    )


@dataclass
class PsiProtocolCapability:
    """协议级能力核查：三层结论**分开**报告。

    - `backend_runnable` ：PSI 后端整体（libpsi / psi_execute / 两方链路 / 平台）；
    - `protocol_runnable`：所选协议能否在这条链路上执行（协议清单 + 参与方数量）；
    - `params_runnable`  ：所选参数组合能否真正注入（RR22 low_comm_mode、ECDH 曲线）。
    """

    protocol: str
    backend_runnable: bool = False
    protocol_runnable: bool = False
    params_runnable: bool = False
    curve: str | None = None
    protocol_params: Mapping[str, Any] = field(default_factory=dict)
    blockers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def runnable(self) -> bool:
        return self.backend_runnable and self.protocol_runnable and self.params_runnable

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "backend_runnable": self.backend_runnable,
            "protocol_runnable": self.protocol_runnable,
            "params_runnable": self.params_runnable,
            "runnable": self.runnable,
            "curve": self.curve,
            "protocol_params": dict(self.protocol_params),
            "blockers": list(self.blockers),
            "notes": list(self.notes),
        }


def check_psi_protocol_capability(
    protocol: str,
    protocol_params: Mapping[str, Any] | None = None,
    report: PsiCapabilityReport | None = None,
    *,
    curve: str | None = None,
) -> PsiProtocolCapability:
    """核查"该协议 + 该参数组合"在当前环境能否真正执行。

    与 `check_psi_capabilities()`（后端整体）的关系：
    - 后端级回答"这台机器有没有 PSI"；
    - 本函数回答"**这条具体执行路径**能不能走"——包括 RR22 的
      `Rr22Rarams` / `rr22_params` 是否齐备、ECDH 族的曲线是否可用。
    错误在进入 PSI 之前暴露，不必等到 Runtime 线程里再炸。
    """

    try:
        name = normalize_psi_protocol(protocol)
    except ValueError as exc:
        return PsiProtocolCapability(protocol=str(protocol), blockers=(str(exc),))

    report = report or check_psi_capabilities()
    params = dict(protocol_params or {})
    blockers: list[str] = []
    notes: list[str] = []

    backend_runnable = bool(report.runnable)
    if not backend_runnable:
        blockers.append("PSI 后端不可运行：见能力报告的 blockers（此处不重复展开）")

    protocol_runnable = backend_runnable and name in report.protocols
    if backend_runnable and name not in report.protocols:
        blockers.append(
            f"协议 {name} 不在当前 SPU 的协议清单：{list(report.protocols)}"
        )

    required = protocol_world_size(name)
    if required > PSI_RUNTIME_WORLD_SIZE:
        protocol_runnable = False
        blockers.append(
            f"协议 {name} 需要 {required} 个参与方，"
            f"本链路固定 {PSI_RUNTIME_WORLD_SIZE} 方"
        )

    # ---- 参数层 ----
    curve_name: str | None = None
    params_runnable = True
    relation = psi_curve_relation(name)
    if relation == "required":
        if curve is None:
            params_runnable = False
            blockers.append(
                f"协议 {name} 必须显式指定曲线（缺省可用 {PSI_DEFAULT_CURVE}）"
            )
        else:
            try:
                curve_name = normalize_curve(curve)
            except ValueError as exc:
                params_runnable = False
                blockers.append(str(exc))
            else:
                if curve_name not in report.curves:
                    params_runnable = False
                    blockers.append(
                        f"曲线 {curve_name} 不在当前 SPU 的曲线清单："
                        f"{list(report.curves)}"
                    )
    elif curve is not None:
        if relation == "ignored":
            notes.append(f"协议 {name} 不读曲线：curve 参数不会注入")
        else:
            notes.append(f"协议 {name} 自带默认曲线：本项目不覆盖调用方曲线")

    if name == "PROTOCOL_RR22":
        probe = report.rr22_params or probe_rr22_params()
        requested_low_comm = bool(params.get("low_comm_mode", False))
        probe_ok = bool(probe.get("runnable"))
        if requested_low_comm and not probe_ok:
            params_runnable = False
            blockers.append(
                "RR22 参数路径不完整（Rr22Rarams / rr22_params 缺失）："
                "无法设置 low_comm_mode=True；"
                + "；".join(probe.get("notes") or ("原因未记录",))
            )
        elif not probe_ok:
            notes.append(
                "RR22 参数类不完整：未请求 low_comm_mode，可按协议内部默认配置执行"
            )

    return PsiProtocolCapability(
        protocol=name,
        backend_runnable=backend_runnable,
        protocol_runnable=protocol_runnable,
        params_runnable=params_runnable,
        curve=curve_name,
        protocol_params=params,
        blockers=tuple(blockers),
        notes=tuple(notes),
    )
