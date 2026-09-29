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
from dataclasses import dataclass
from typing import Any, Mapping

# --------------------------------------------------------------------------
# 从官方 libpsi.pyi / psi.py 核对的常量（不来自猜测）
# --------------------------------------------------------------------------

#: `spu.psi.PsiProtocol` 的成员（0.9.5 libpsi.pyi 实测）
PSI_PROTOCOLS: tuple[str, ...] = (
    "PROTOCOL_ECDH",
    "PROTOCOL_KKRT",
    "PROTOCOL_RR22",
    "PROTOCOL_ECDH_3PC",
    "PROTOCOL_ECDH_NPC",
    "PROTOCOL_KKRT_NPC",
    "PROTOCOL_DP",
)

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
PSI_CURVE_RELATION: Mapping[str, str] = {
    "PROTOCOL_ECDH": "required",
    "PROTOCOL_KKRT": "ignored",
    "PROTOCOL_RR22": "ignored",
    "PROTOCOL_ECDH_3PC": "required",
    "PROTOCOL_ECDH_NPC": "required",
    "PROTOCOL_KKRT_NPC": "ignored",
    "PROTOCOL_DP": "implicit",
}

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
PSI_PROTOCOL_WORLD_SIZE: Mapping[str, int] = {
    "PROTOCOL_ECDH": 2,
    "PROTOCOL_KKRT": 2,
    "PROTOCOL_RR22": 2,
    "PROTOCOL_ECDH_NPC": 2,
    "PROTOCOL_KKRT_NPC": 2,
    "PROTOCOL_DP": 2,
    "PROTOCOL_ECDH_3PC": 3,
}

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
PSI_PROTOCOLS_WITH_NOISE: tuple[str, ...] = ("PROTOCOL_DP",)

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


def protocol_is_exact(protocol: str) -> bool:
    """该协议是否给出**精确**交集（不向结果注入噪声）。

    返回 False 的协议仍可执行，但其结果**不能**拿来做与明文比对的一致性验证，
    因此调用方不得据此报"已验证"。
    """

    return normalize_psi_protocol(protocol) not in PSI_PROTOCOLS_WITH_NOISE


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
