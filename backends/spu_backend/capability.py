"""SPU 后端：能力探测。

本模块**不假设任何 SPU API**。所有判断都来自对当前安装环境与官方源码的实际核对，
核对结论记录在 `docs/SPU_CAPABILITY.md`，并带证据出处。

已核对事实（SPU 0.9.5，2026-09 发布）：
  - 模块名 `spu`，无 `run_spu_simulation`；相关入口是 `spu.utils.simulation.sim_jax`
    与 `spu.utils.simulation.Simulator.simple(wsize, ProtocolKind, FieldType)`
  - 编译桥 `spu.utils.frontend.compile(Kind.JAX, ...)`：
      注册 `interpreter` 后端 → `jax.jit(...).trace(...).lower(lowering_platforms=('interpreter',))`
      → `compiler_ir('hlo').as_serialized_hlo_module_proto()` → `spu_api.compile(...)`
  - 硬依赖 `jax._src.lib` 下的**属性** `xla_client` / `xla_extension_version`，
      即 `from jax._src.lib import xla_client, xla_extension_version`；
      注意：这是属性而非子模块，判定时不能按模块 import
  - 硬依赖 `jax._src.lax.lax._canonicalize_float_for_sort` 补丁点（0.4.34 / 0.11.2 均存在）
  - `ProtocolKind` = REF2K / SEMI2K / ABY3 / CHEETAH / SECURENN（无 SPDZ2K）
  - `FieldType` = FM32 / FM64 / FM128
  - wheel 仅发布 cp310/cp311 的 macOS 与 manylinux；requires-python >=3.10,<3.12
  - 原生 Windows x64 不受支持（libspu 为 Linux ELF）
"""

from __future__ import annotations

import importlib
import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# --------------------------------------------------------------------------
# 从官方源码核对的常量（不来自猜测）
# --------------------------------------------------------------------------

#: `spu.libspu.ProtocolKind` 的成员（0.9.5 实测）
SPU_PROTOCOLS: tuple[str, ...] = ("REF2K", "SEMI2K", "ABY3", "CHEETAH", "SECURENN")

#: `spu.libspu.FieldType` 的成员（0.9.5 实测）
SPU_FIELDS: tuple[str, ...] = ("FM32", "FM64", "FM128")

#: 各协议要求的参与方数量下限（来自协议定义）
PROTOCOL_MIN_WORLD_SIZE: Mapping[str, int] = {
    "REF2K": 2,
    "SEMI2K": 2,
    "ABY3": 3,
    "CHEETAH": 2,
    "SECURENN": 3,
}

#: 环宽与位宽的对应（决定整数溢出边界）
FIELD_BITS: Mapping[str, int] = {
    "FM32": 32,
    "FM64": 64,
    "FM128": 128,
}

#: SPU 官方绑定的 jax 版本区间（取自 spu 0.9.5 的 METADATA）
SPU_JAX_PIN = ">=0.4.16,<=0.4.34"

#: SPU 的 jax 私有接口依赖清单：缺失即视为不兼容。
#:
#: 解析规则（对应 SPU 源码的两种写法）：
#:   1. 子模块，如 `import jax._src.api_util` -> 直接 import_module
#:   2. 祖先模块的属性，如 `from jax._src.lib import xla_extension_version`
#:      -> 拆出父模块再 getattr。**不能**按子模块 import，
#:      否则会把已存在的属性误判为缺失。
SPU_JAX_PRIVATE_DEPS: tuple[str, ...] = (
    "jax.extend.linear_util",
    "jax._src.api_util",
    "jax._src.lib.xla_client",
    "jax._src.lib.xla_extension_version",
    "jax._src.xla_bridge",
    "jax._src.lax.lax",
)

#: libspu 已适配的 jax 原语（用于 capability check 的白名单）
#: 来源：SPU 官方 CHANGELOG / 测试目录 jnp_*_test.py 的算子命名
SPU_ADAPTED_PRIMITIVES: tuple[str, ...] = (
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "abs",
    "max",
    "min",
    "sum",
    "dot",
    "concatenate",
    "slice",
    "reshape",
    "transpose",
    "broadcast",
    "select",       # jnp.where
    "comparisons",  # < <= > >= == !=
    "and",
    "or",
    "not",
    "shift_left",
    "shift_right",
    "bitwise_and",
    "bitwise_or",
    "bitwise_count",
    "top_k",
    "sort",         # 经 frontend 补丁路径
    "argsort",
    "reduce_max",
    "reduce_min",
    "reduce_any",
    "reduce_all",
    "dynamic_slice",
    "gather",
    "scatter",
    "conv",
    "exp",
    "log",
    "sqrt",         # 存在但代价高：d 与 R 上升，应尽量避免
    "rsqrt",
    "tanh",
    "erf",
)

#: libspu 已适配的 **StableHLO 原语** 白名单（能力核查用）。
#: 判定依据：这些原语在 SPU 的 pphlo 方言中都有对应 lower 规则；
#: 未列入者需先核实，不视为可用。
SPU_ADAPTED_HLO_PRIMITIVES: tuple[str, ...] = (
    "constant", "add", "subtract", "multiply", "divide", "remainder",
    "negate", "abs", "max", "min", "compare", "select", "sign",
    "and", "or", "not", "shift_left", "shift_right_logical", "shift_right_arithmetic",
    "broadcast_in_dim", "reshape", "transpose", "slice", "dynamic_slice",
    "concatenate", "reduce", "dot", "convert", "clamp", "iota",
    "gather", "scatter", "sort", "while", "exp", "log", "sqrt", "rsqrt",
)

#: StableHLO 层的高代价原语
SPU_EXPENSIVE_HLO_PRIMITIVES: Mapping[str, str] = {
    "divide": "定点除法的 MPC 电路代价高（d 与 R 上升）；能改乘倒数或平方比较就改",
    "remainder": "取余随除法一同展开，进一步抬升 d",
    "sqrt": "开方电路代价高；距离比较应改为平方比较",
    "rsqrt": "同上",
    "sort": "依赖 frontend 的 float→int 补丁，跨 jax 版本易碎",
    "while": "动态循环在 MPC 中需定长展开，轮次随迭代数线性增长",
    "exp": "超越函数走多项式逼近，d 随精度要求上升",
    "log": "同上",
}

#: 已知代价高或语义受限、需要显式告警的原语
SPU_EXPENSIVE_PRIMITIVES: Mapping[str, str] = {
    "sqrt": "引入除法/开方电路，乘法深度与通信轮次上升；距离比较应改用平方比较",
    "div": "定点除法在 MPC 中代价高，d 上升；建议改为乘倒数或平方比较",
    "sort": "依赖 frontend 的 float→int 补丁；跨版本易碎",
    "top_k": "已适配但通信量随 k 上升",
}


# --------------------------------------------------------------------------
# 环境探测
# --------------------------------------------------------------------------


@dataclass
class JaxProbe:
    """当前 JAX 环境的探测结果。"""

    installed: bool = False
    version: str | None = None
    prefix: str | None = None
    private_deps_ok: bool = False
    missing_deps: tuple[str, ...] = ()
    jit_ok: bool = False
    hlo_lowering_ok: bool = False
    hlo_bytes: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "version": self.version,
            "prefix": self.prefix,
            "private_deps_ok": self.private_deps_ok,
            "missing_deps": list(self.missing_deps),
            "jit_ok": self.jit_ok,
            "hlo_lowering_ok": self.hlo_lowering_ok,
            "hlo_bytes": self.hlo_bytes,
            "error": self.error,
        }


@dataclass
class SpuProbe:
    """当前 SPU 环境的探测结果。"""

    installed: bool = False
    version: str | None = None
    prefix: str | None = None
    libspu_loadable: bool = False
    simulation_api: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "version": self.version,
            "prefix": self.prefix,
            "libspu_loadable": self.libspu_loadable,
            "simulation_api": list(self.simulation_api),
            "error": self.error,
        }


@dataclass
class CapabilityReport:
    """SPU 能力核查总报告。"""

    platform: str = ""
    machine: str = ""
    python_version: str = ""
    python_supported: bool = False
    jax: JaxProbe = field(default_factory=JaxProbe)
    spu: SpuProbe = field(default_factory=SpuProbe)
    blockers: tuple[str, ...] = ()
    runnable: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.runnable:
            return "available"
        if self.spu.installed:
            return "installed-unrunnable"
        return "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "runnable": self.runnable,
            "platform": self.platform,
            "machine": self.machine,
            "python_version": self.python_version,
            "python_supported": self.python_supported,
            "jax": self.jax.to_dict(),
            "spu": self.spu.to_dict(),
            "blockers": list(self.blockers),
            "details": dict(self.details),
        }


def _private_dep_available(dep_name: str) -> bool:
    """判定 SPU 依赖的 jax 私有接口是否可用。

    支持两种形态：子模块（如 `jax._src.api_util`）与祖先模块的属性（如
    `jax._src.lib.xla_extension_version`）。后者必须先尝试模块导入，失败后才拆出
    父模块取属性；不可直接对属性名调用 import_module。
    """

    try:
        importlib.import_module(dep_name)
        return True
    except Exception:
        pass

    parent, _, attr = dep_name.rpartition(".")
    if not parent or not attr:
        return False
    try:
        return hasattr(importlib.import_module(parent), attr)
    except Exception:
        return False


def probe_jax() -> JaxProbe:
    """探测 JAX：是否可导入、私有接口是否齐全、jit 与 HLO 提取是否可用。"""

    probe = JaxProbe()
    try:
        jax = importlib.import_module("jax")
    except ImportError as exc:
        probe.error = f"jax 未安装：{exc}"
        return probe

    probe.installed = True
    probe.version = getattr(jax, "__version__", None)
    probe.prefix = getattr(jax, "__file__", None)

    missing: list[str] = []
    for dep_name in SPU_JAX_PRIVATE_DEPS:
        if not _private_dep_available(dep_name):
            missing.append(dep_name)
    probe.missing_deps = tuple(missing)
    probe.private_deps_ok = not missing

    # jit 追踪
    try:
        import jax.numpy as jnp

        compiled = jax.jit(lambda x: jnp.sum(jnp.square(x)))(jnp.array([1.0, 2.0, 3.0]))
        probe.jit_ok = float(compiled) == 14.0
    except Exception as exc:
        probe.error = f"jax.jit 追踪失败：{exc}"

    # HLO 提取（SPU 编译桥的第一步，可在无 SPU 的情况下独立验证）
    probe.hlo_lowering_ok, probe.hlo_bytes, hlo_error = _probe_hlo_lowering()
    if not probe.hlo_lowering_ok and probe.error is None:
        probe.error = hlo_error

    return probe


def _probe_hlo_lowering() -> tuple[bool, int, str | None]:
    """验证 SPU 编译桥第一步：interpreter 后端注册 + HLO 序列化。"""

    try:
        import jax
        import jax.numpy as jnp
        from jax._src.xla_bridge import _backend_lock, _backends, register_backend_factory
    except Exception as exc:
        return False, 0, f"导入失败：{exc}"

    try:
        with _backend_lock:
            has_interpreter = "interpreter" in _backends
        if not has_interpreter:
            from jax.interpreters.xla import Backend as xla_backend

            register_backend_factory("interpreter", xla_backend, priority=-100)

        fn = lambda x, y: jnp.sum(jnp.abs(x - y))
        args = (jnp.array([1.0, 2.0, 3.0]), jnp.array([1.0, 3.0, 3.0]))
        lowered = (
            jax.jit(fn, keep_unused=True)
            .trace(*args)
            .lower(lowering_platforms=("interpreter",))
        )
        hlo = lowered.compiler_ir("hlo").as_serialized_hlo_module_proto()
        return True, len(hlo), None
    except Exception as exc:
        return False, 0, f"HLO 提取失败：{type(exc).__name__}: {exc}"


def probe_spu() -> SpuProbe:
    """探测 SPU：模块是否可导入、libspu 是否可加载、可用 API 有哪些。"""

    probe = SpuProbe()
    try:
        spu = importlib.import_module("spu")
    except ImportError as exc:
        probe.error = f"spu 未安装：{exc}"
        return probe
    except Exception as exc:  # 例如 libspu.so 存在但无法加载
        probe.error = f"spu 导入失败（可能是平台不支持）：{type(exc).__name__}: {exc}"
        return probe

    probe.installed = True
    probe.version = getattr(spu, "__version__", None) or _version_from_module(spu)
    probe.prefix = getattr(spu, "__file__", None)

    try:
        libspu = importlib.import_module("spu.libspu")
        probe.libspu_loadable = hasattr(libspu, "ProtocolKind")
    except Exception as exc:
        probe.error = f"spu.libspu 无法加载：{type(exc).__name__}: {exc}"

    api: list[str] = []
    try:
        simulation = importlib.import_module("spu.utils.simulation")
        for name in ("Simulator", "sim_jax"):
            if hasattr(simulation, name):
                api.append(f"spu.utils.simulation.{name}")
    except Exception:
        pass
    try:
        frontend = importlib.import_module("spu.utils.frontend")
        if hasattr(frontend, "compile"):
            api.append("spu.utils.frontend.compile")
        if hasattr(frontend, "Kind"):
            api.append("spu.utils.frontend.Kind")
    except Exception:
        pass
    probe.simulation_api = tuple(api)
    return probe


def _module_spec_available(module_name: str) -> bool:
    """判定某个 Python 模块（含扩展模块 `.so`）是否可找到。

    不能用 `shutil.which`：它只扫 PATH 上的**可执行**文件，
    而 `libspu.so` 是动态库，永远不会被命中。
    """

    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _version_from_module(module: Any) -> str | None:
    for attr in ("version", "__version__"):
        value = getattr(module, attr, None)
        if isinstance(value, str):
            return value
    return None


def probe_platform() -> dict[str, Any]:
    """探测平台是否可能承载 SPU 原生库。"""

    system = platform.system()
    machine = platform.machine()
    python_version = platform.python_version()
    major, minor = sys.version_info[:2]

    detail: dict[str, Any] = {
        "system": system,
        "machine": machine,
        "python_version": python_version,
        # libspu 是共享库，不在 PATH 上、也不可执行，
        # shutil.which("libspu.so") 永远返回 None。
        # 正确做法是问 Python 自己能不能找到这个扩展模块。
        "libspu_present": _module_spec_available("spu.libspu"),
        "notes": [],
    }

    python_supported = (major == 3 and minor in (10, 11))
    detail["python_supported"] = python_supported
    if not python_supported:
        detail["notes"].append(
            f"spu 0.9.5 要求 Python >=3.10,<3.12；当前 {python_version} 不在支持区间"
        )

    if system == "Windows" and machine.lower() in ("amd64", "x86_64"):
        detail["notes"].append(
            "spu 的 libspu 以 manylinux/macOS wheel 发布，原生 Windows x64 不受支持；"
            "官方要求 WSL2 或 Linux 容器"
        )
    return detail


def check_capabilities(*, verbose: bool = False) -> CapabilityReport:
    """执行完整的 SPU 能力核查。"""

    platform_detail = probe_platform()
    report = CapabilityReport(
        platform=platform_detail["system"],
        machine=platform_detail["machine"],
        python_version=platform_detail["python_version"],
        python_supported=platform_detail["python_supported"],
        details={"platform": platform_detail},
    )
    report.jax = probe_jax()
    report.spu = probe_spu()

    blockers: list[str] = []
    if not report.python_supported:
        blockers.append(
            f"Python {report.python_version} 不在 spu 支持区间（>=3.10,<3.12）"
        )
    if report.platform == "Windows" and report.machine.lower() in ("amd64", "x86_64"):
        blockers.append("原生 Windows 无 libspu 原生库（需 WSL2 / Linux）")
    if not report.spu.installed:
        blockers.append("spu 未安装")
    elif not report.spu.libspu_loadable:
        blockers.append("spu.libspu 无法加载")
    if not report.jax.installed:
        blockers.append("jax 未安装")
    else:
        if not report.jax.private_deps_ok:
            blockers.append(
                "SPU 依赖的 jax 私有接口缺失：" + ", ".join(report.jax.missing_deps)
            )
        jax_version = report.jax.version or ""
        if not _version_in_range(jax_version, 0, 4, 16, 0, 4, 34):
            blockers.append(
                f"jax {jax_version} 超出 SPU 绑定区间 {SPU_JAX_PIN}"
            )

    report.blockers = tuple(blockers)
    report.runnable = not blockers
    return report


def _version_in_range(
    version: str, lo_major: int, lo_minor: int, lo_patch: int, hi_major: int, hi_minor: int, hi_patch: int
) -> bool:
    parsed = _parse_version(version)
    if parsed is None:
        return False
    return (lo_major, lo_minor, lo_patch) <= parsed <= (hi_major, hi_minor, hi_patch)


def _parse_version(version: str) -> tuple[int, int, int] | None:
    import re

    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version.strip())
    if not match:
        return None
    return tuple(int(g) for g in match.groups())  # type: ignore[return-value]


# --------------------------------------------------------------------------
# 生成代码的能力核查
# --------------------------------------------------------------------------


@dataclass
class OpCapability:
    """单个算子在当前 SPU 上的可用性结论。"""

    op: str
    supported: bool
    protocol_hint: str = ""
    field_hint: str = ""
    primitives: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    status: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "supported": self.supported,
            "status": self.status,
            "protocol_hint": self.protocol_hint,
            "field_hint": self.field_hint,
            "primitives": list(self.primitives),
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
        }


#: 算子 → 生成代码实测发射的 StableHLO 原语。
#:
#: 来源：对生成代码取 lowered.as_text() 后提取 stablehlo.<op>（见
#: backends/jax_backend/spu_bridge.py 与 tests/test_spu_backend.py 的核对用例），
#: **不是**按源码猜的。
OP_HLO_PRIMITIVES: Mapping[str, tuple[str, ...]] = {
    "DistanceLE": (
        "subtract", "multiply", "reduce", "add", "convert", "compare", "constant",
    ),
    "WeightedSum": (
        # P2-1：定点 scale 改为编译期常量后，管线里的生成代码（scale=1）
        # 不再有除法。历史记录：旧的 `acc // scale` 在 HLO 里展开为
        # divide + remainder + select + sign（源看起来一步，密态代价多步），
        # 实测占该算子 PPHLO 字节数的 61%、通信量的 56%，
        # 且结果与明文不再逐位一致、FM32 直接崩。
        # 见 docs/MPC_BENCHMARK_PROTOCOL.md §4.2 / §4.3。
        "multiply", "reduce", "add", "constant",
    ),
    "TemporalOverlap": (
        "shift_left", "add", "broadcast_in_dim", "compare", "and", "or",
        "reduce", "constant",
    ),
    # Contains 的 JAX 侧仍然不生成代码（集合交不是逐元素算子），但它的
    # **子集判定**那一段是真正的 MPC 电路（见 backends.psi_backend.subset_mpc）。
    # 原语取自 `jax.jit(...).lower().as_text()` 的真实产物：
    #   %0 = stablehlo.compare EQ, %arg0, %arg1, SIGNED : (i32, i32) -> i1
    #   %1 = stablehlo.convert %0 : (i1) -> i32
    # 由 tests/test_psi_backend.py 的用例钉住，不靠人肉推测。
    "Contains": ("compare", "convert"),
}

#: 算子推荐的协议与环宽
OP_PROTOCOL_HINT: Mapping[str, tuple[str, int]] = {
    "DistanceLE": ("ABY3", 64),
    "WeightedSum": ("ABY3", 64),
    "TemporalOverlap": ("ABY3", 64),
}


#: 兼容旧名：语义层原语（供文档与人工阅读），能力核查用 OP_HLO_PRIMITIVES
OP_PRIMITIVES: Mapping[str, tuple[str, ...]] = {
    "DistanceLE": ("sub", "mul", "sum", "comparisons"),
    "WeightedSum": ("mul", "sum"),
    "TemporalOverlap": ("add", "shift_left", "comparisons", "and", "or", "reduce_any"),
}


def is_plaintext_local_op(op: str) -> bool:
    """该算子是否只在本方明文执行（不进 SPU/PSI 任何密态路径）。

    判据取自 planner 注册表的 backend 字段，避免在能力层再抄一份算子名单。
    """

    from planner.registry import get_rule

    rule = get_rule(op)
    return rule is not None and rule.primary_backend == "Plaintext"


def check_operation_capability(op: str, report: CapabilityReport | None = None) -> OpCapability:
    """核查单个算子的生成代码能否在当前 SPU 上运行。"""

    report = report or check_capabilities()

    # 物化算子（如 HeightBand）在本方明文执行，根本不会进入 SPU。
    # 若按默认路径返回 supported=True，等于宣称"该算子可在 SPU 上执行"；
    # 按 supported=False 返回，又会产生一条与 SPU 无关的 SPU_UNSUPPORTED 诊断。
    # 因此单列一个 not-routed 状态：无阻断，但也明确它不走这条路径。
    if is_plaintext_local_op(op):
        return OpCapability(
            op=op,
            supported=True,
            protocol_hint="",
            field_hint="",
            primitives=(),
            warnings=(
                "该算子在本方明文把高度带物化为格网码集合，不进 SPU；"
                "密态边界落在消费它的集合族算子（PSI）上",
            ),
            status="not-routed",
        )

    primitives = OP_HLO_PRIMITIVES.get(op, ())
    protocol, bits = OP_PROTOCOL_HINT.get(op, ("ABY3", 64))

    warnings: list[str] = []
    blockers: list[str] = []
    unsupported_primitives = [p for p in primitives if p not in SPU_ADAPTED_HLO_PRIMITIVES]
    for name in primitives:
        if name in SPU_EXPENSIVE_HLO_PRIMITIVES:
            warnings.append(f"{name}: {SPU_EXPENSIVE_HLO_PRIMITIVES[name]}")
    if unsupported_primitives:
        blockers.append(
            "以下原语未在 SPU 已适配清单中（需先核实再使用）："
            + ", ".join(unsupported_primitives)
        )

    # 环宽与溢出：本项目 grid_code 为 64 位，FM64 下无符号大整数会碰到边界。
    # `Contains` 要单独说：它的 grid_code 全程不出 PSI（等值求交没有大小语义），
    # 进 MPC 的只有基数 k/n 两个小整数，故对它照抄上面那条告警是**错的**。
    if op in ("Intersects", "CellSetIntersect"):
        warnings.append(
            "grid_code 为 64 位定长键；FM64 下作为无符号整数参与运算时"
            "存在溢出风险，需要时改用 FM128"
        )
    elif op == "Contains":
        warnings.append(
            "grid_code 不进 MPC：集合交由 PSI 做等值求交（无大小语义，不存在 FM64 "
            "溢出）；进 MPC 的只有基数 k=|outer∩inner| 与 n=|inner|，均为小整数"
        )

    blockers.extend(report.blockers)

    status = "available"
    if blockers and report.runnable:
        status = "partial"
    elif blockers:
        status = "unavailable"

    return OpCapability(
        op=op,
        supported=not blockers,
        protocol_hint=protocol,
        field_hint=f"FM{bits}" if bits in (32, 64, 128) else "FM64",
        primitives=primitives,
        warnings=tuple(warnings),
        blockers=tuple(blockers),
        status=status,
    )


def protocol_min_world_size(protocol: str) -> int:
    return PROTOCOL_MIN_WORLD_SIZE.get(protocol.upper(), 2)


def normalize_protocol(protocol: str) -> str:
    """宽松接受协议名，返回 SPU 官方枚举名。"""

    upper = protocol.upper().strip()
    if upper in SPU_PROTOCOLS:
        return upper
    raise ValueError(
        f"未知协议 {protocol!r}；SPU 0.9.5 支持 {SPU_PROTOCOLS}（注意：无 SPDZ2K）"
    )


def normalize_field(field: str | int) -> str:
    """宽松接受环宽写法：64 / "64" / FM64 / fm64 均可（签名允许 int）。"""

    text = str(field).upper().strip()
    if text in SPU_FIELDS:
        return text
    if text.isdigit():
        candidate = f"FM{text}"
        if candidate in SPU_FIELDS:
            return candidate
    raise ValueError(f"未知环宽 {field!r}；SPU 0.9.5 支持 {SPU_FIELDS}")


def platform_can_run_spu() -> tuple[bool, str]:
    """快速判断当前平台能否承载 SPU 原生库。"""

    system = platform.system()
    if system in ("Linux", "Darwin"):
        return True, f"{system} 平台具备 libspu 原生库"
    return False, (
        f"{system} 平台不具备 libspu 原生库；"
        "SPU 仅发布 manylinux/macOS wheel，Windows 需 WSL2"
    )
