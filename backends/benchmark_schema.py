# -*- coding: utf-8 -*-
"""统一 benchmark metadata（Phase 7 / 任务文档 §九）：把四个协议族（PSI / PSI-CA / PI-SUM / MPC）的性能记录投影到同一张表。

任务文档 §九 要求「未来 PSI / MPC 都应该能够输出统一的 benchmark metadata」，
并给出三层形状：

    CommonBenchmarkRecord     各族共有的 13 个字段（同名同口径）
    PSIBenchmarkMetadata      libpsi 求交档特有指标（交集比例 / 协议耗时拆段 / 曲线关系 …）
    PsiCaBenchmarkMetadata    PSI-CA 计数档特有指标（结构 / 计数 / 进程内存测量口径 …）
    PsiSumBenchmarkMetadata   PI-Sum 交集内求和档特有指标（上游提交 / 关联值规则 / 和值 …）
    MPCBenchmarkMetadata      MPC 特有指标（环宽 / 重复实验统计 / 通信量拆原语 …）

本模块是**只读投影层**：

- 输入是各族运行器已经落盘的产物（`docs/psi_benchmark_baseline.json`、
  `docs/psi_ca_benchmark_baseline.json`、`docs/psi_sum_benchmark_baseline.json`、
  `docs/mpc_benchmark_baseline.json`、`docs/mpc_comm_baseline.json`）；
  不执行协议、不重跑基线、不改写产物；
- **不丢字段**：原始记录原样放进 `metadata.raw`；统一层只做投影，
  不改写、不补齐、不推断——原记录里没有的数记 `None`；
- **缺口看得见**：`MISSING_METRICS` 写明 §九 建议、但某族基线**确实没采集**
  的指标，并由测试锁死在真实产物上（这些键必须真的不在记录里）。

与 `backends/protocol_validation.py` 的分工
==========================================
后者回答「这个协议请求成不成立」，本模块回答「跑完的结果怎么摆到一张表上」。
两者都只读注册表事实（协议族 / world_size / 结果语义 / 曲线关系 / 环宽），
不各自维护第二份名单。外部两条 PSI 路径（PSI-CA / PI-Sum）没有注册表：
它们的 world_size / result_semantics 等事实从**记录列**读取（由运行器写入
运行时结论），读不到就记 None——不为它们另立一份名单。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Callable, Iterable, Mapping, Sequence

from backends.psi_backend.capability import (
    normalize_psi_protocol,
    protocol_result_semantics,
    protocol_world_size,
    psi_curve_relation,
)
from backends.spu_backend.capability import normalize_protocol as normalize_mpc_protocol
from backends.spu_backend.protocol_registry import MPC_PROTOCOL_SPECS

#: 协议族（顺序稳定：报告与测试按此渲染）。
#: PSI = libpsi 求交；PSI-CA = openmined-psi 计数；PI-SUM = private-join-and-compute
#: 交集内求和——三条 PSI 路径独立执行、泄漏承诺互不相同，各自成族。
BENCHMARK_FAMILY_PSI = "PSI"
BENCHMARK_FAMILY_PSI_CA = "PSI-CA"
BENCHMARK_FAMILY_PSI_SUM = "PI-SUM"
BENCHMARK_FAMILY_MPC = "MPC"
BENCHMARK_FAMILIES: tuple[str, ...] = (
    BENCHMARK_FAMILY_PSI,
    BENCHMARK_FAMILY_PSI_CA,
    BENCHMARK_FAMILY_PSI_SUM,
    BENCHMARK_FAMILY_MPC,
)

#: 状态词（原样透传；未执行的行不给任何数字）
STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ERROR = "error"
STATUS_UNKNOWN = "unknown"

#: PSI 三档基线（libpsi 求交 / PSI-CA 计数 / PI-Sum 交集内求和）都只跑一条执行路径
#: （CellSetIntersect）；记录里没有 operation 列时按此填
PSI_BENCHMARK_OPERATION = "CellSetIntersect"

#: MB → 字节（记录里的内存列以 MB 计；换算后 MB 原值仍在 metadata.raw 里）
BYTES_PER_MB = 1024 * 1024

#: 曲线关系「必须显式给曲线」——与 `psi_backend.protocol_registry` 的
#: `_CURVE_RELATIONS` 同一词汇（不在本模块另立一套分类）。
CURVE_RELATION_REQUIRED = "required"

#: §九 的 `BenchmarkRecord` 字段（两族共有；逐字段口径见 `FIELD_SEMANTICS`）
COMMON_FIELDS: tuple[str, ...] = (
    "family",
    "protocol",
    "operation",
    "world_size",
    "field",
    "input_size",
    "unique_size",
    "result_size",
    "compute_time",
    "communication_bytes",
    "total_time",
    "memory_bytes",
    "status",
)

#: 共有字段的口径与单位（毫秒 / 字节）。各族来路不同的地方逐条写明。
FIELD_SEMANTICS: Mapping[str, str] = {
    "family": (
        "协议族：PSI = libpsi 求交；PSI-CA = openmined-psi 计数；"
        "PI-SUM = private-join-and-compute 交集内求和；MPC = 密态算子运行时"
    ),
    "protocol": "协议名（PSI 三档用各自路径协议名 / MPC 用官方枚举名；各族名字不重叠）",
    "operation": "算子名；PSI 三档基线只跑 CellSetIntersect，MPC 取记录的 op",
    "world_size": (
        "参与方数量下限：PSI / MPC 取自各自协议注册表；PSI-CA / PI-Sum "
        "从记录列读取（运行器写入运行时结论）"
    ),
    "field": (
        "密码域：MPC = 环宽 FM32/FM64/FM128；PSI = 椭圆曲线"
        "（曲线关系为 ignored 的协议记 None，因为给了也不读）；"
        "PSI-CA / PI-Sum 无曲线 / 环宽概念（结构 / Paillier 等参数在 metadata）"
    ),
    "input_size": "N：输入规模。PSI 三档 = n_left + n_right；MPC = 可归约元素数 k",
    "unique_size": "unique_N：去重后规模。PSI 三档 = 两方唯一键数之和；MPC 无去重概念（None）",
    "result_size": (
        "结果规模。PSI = 交集基数（ECDH 含重复乘数）；PSI-CA / PI-Sum = 交集基数"
        "（PI-Sum 的和值在 metadata）；MPC 基线未记录（None）"
    ),
    "compute_time": (
        "协议计算耗时（毫秒）= §九 的 protocol_time。PSI = psi_execute_ms；"
        "PSI-CA = psi_ca_execute_ms；PI-Sum = pi_sum_execute_ms（含两个子进程往返）；"
        "MPC = wall_ms - setup_ms（setup 是造数，wall 含它在内）"
    ),
    "communication_bytes": (
        "发送 + 接收字节 = §九 的 send_bytes + recv_bytes。"
        "MPC = comm_total_bytes；PI-Sum = 回环中继总字节（Phase 10）；"
        "PSI-CA = 协议消息的 protobuf 载荷（Phase 11，进程内链路、非网络观测，"
        "见 PSI_CA_CAPABILITY.md §7.1）；PSI（libpsi）基线未采集（None）"
    ),
    "total_time": "端到端耗时（毫秒）。PSI 三档 = total_ms；MPC = wall_ms（repeat>1 时是中位数）",
    "memory_bytes": (
        "峰值常驻内存（字节）= §九 的 peak_memory。由 peak_rss_mb 换算；"
        "注意同名的 memory_mb 是「本次运行抬升高水位多少」，不是绝对值。"
        "PSI-CA 的运行器进程读数覆盖协议（进程内库）；PI-Sum = 两个子进程的 "
        "procfs VmHWM 探针值取大者（Phase 10，采样口径见记录 note）"
    ),
    "status": "ok / unavailable / error（原样透传）",
}


@dataclass(frozen=True)
class TaskDocMetric:
    """§九 指标清单里的一项：它最终落在哪。"""

    target: str
    note: str = ""


#: 任务文档 §九「建议至少记录」的指标清单，逐项登记去向：
#: `common:<字段>` / `metadata:<本族字段>` / `missing`（各族基线当前都没采集）。
TASK_DOC_METRICS: Mapping[str, TaskDocMetric] = {
    "family": TaskDocMetric("common:family"),
    "protocol": TaskDocMetric("common:protocol"),
    "operation": TaskDocMetric("common:operation"),
    "world_size": TaskDocMetric("common:world_size"),
    "field": TaskDocMetric("common:field"),
    "N": TaskDocMetric("common:input_size", "§九 的 N"),
    "unique_N": TaskDocMetric("common:unique_size", "§九 的 unique_N"),
    "intersection_ratio": TaskDocMetric("metadata:intersection_ratio"),
    "encode_time": TaskDocMetric("missing", "各族产物都没有这一列（编码耗时在编译器侧，不在基线里）"),
    "dedup_time": TaskDocMetric(
        "missing", "各族产物都没有这一列（去重在运行时内部执行，未单独计时）"
    ),
    "input_io_time": TaskDocMetric(
        "metadata:input_io_time",
        "PSI = io_write_ms + io_read_ms；PI-Sum = io_write_ms（只计输入 CSV 落盘段，"
        "结果走 stdout 无读取段）；MPC 无 IO；PSI-CA 全程内存（无 IO 步骤，"
        "按族登记在 MISSING_METRICS）",
    ),
    "protocol_time": TaskDocMetric("common:compute_time"),
    "semantic_processing_time": TaskDocMetric(
        "metadata:semantic_processing_time", "PSI = semantic_ms；其余族未记录"
    ),
    "total_time": TaskDocMetric("common:total_time"),
    "send_bytes": TaskDocMetric(
        "metadata:send_bytes",
        "MPC = comm_send_bytes；PI-Sum = 中继 client→server 字节（Phase 10）；"
        "PSI-CA = client 的 Request 载荷（Phase 11）；PSI（libpsi）未采集",
    ),
    "recv_bytes": TaskDocMetric(
        "metadata:recv_bytes",
        "MPC = comm_recv_bytes；PI-Sum = 中继 server→client 字节（Phase 10）；"
        "PSI-CA = server 的 ServerSetup + Response 载荷（Phase 11）；"
        "PSI（libpsi）未采集",
    ),
    "total_bytes": TaskDocMetric(
        "metadata:total_bytes",
        "MPC = comm_total_bytes；PI-Sum = send + recv（Phase 10）；"
        "PSI-CA = send + recv（Phase 11）；PSI（libpsi）未采集",
    ),
    "peak_memory": TaskDocMetric("common:memory_bytes", "§九 的 peak_memory"),
    "status": TaskDocMetric("common:status"),
    "layout_agreement": TaskDocMetric(
        "missing", "PSI 三档记录 layout_id 但不记握手结论；MPC 无布局概念"
    ),
    "result_semantics": TaskDocMetric(
        "metadata:result_semantics",
        "PSI / MPC 由协议注册表派生；PSI-CA / PI-Sum 取记录列（运行时的声明）",
    ),
}

#: §九 建议、但某族基线**确实没采集**的指标（测试逐条在真实产物上锁死）
MISSING_METRICS: Mapping[str, tuple[str, ...]] = {
    BENCHMARK_FAMILY_PSI: (
        "encode_time",
        "dedup_time",
        "layout_agreement",
        "send_bytes",
        "recv_bytes",
        "total_bytes",
    ),
    BENCHMARK_FAMILY_PSI_CA: (
        "encode_time",
        "dedup_time",
        "input_io_time",
        "semantic_processing_time",
        "layout_agreement",
    ),
    BENCHMARK_FAMILY_PSI_SUM: (
        "encode_time",
        "dedup_time",
        "semantic_processing_time",
        "layout_agreement",
    ),
    BENCHMARK_FAMILY_MPC: (
        "encode_time",
        "dedup_time",
        "input_io_time",
        "semantic_processing_time",
    ),
}

#: 缺口核对用：§九 指标名 → 原记录里可能的键名。
#: 只登记出现在 `MISSING_METRICS` 里的名字（测试锁定两者键集合相等）。
MISSING_RAW_KEYS: Mapping[str, tuple[str, ...]] = {
    "encode_time": ("encode_ms", "encode_time"),
    "dedup_time": ("dedup_ms", "dedup_time"),
    "input_io_time": ("input_io_ms", "io_write_ms", "io_read_ms"),
    "semantic_processing_time": ("semantic_ms", "semantic_processing_ms"),
    "layout_agreement": ("layout_agreement",),
    "send_bytes": ("send_bytes", "comm_send_bytes"),
    "recv_bytes": ("recv_bytes", "comm_recv_bytes"),
    "total_bytes": ("total_bytes", "comm_total_bytes"),
}

#: 可比较的共有字段（数值型；其余字段不是「大小」）
COMPARABLE_METRICS: tuple[str, ...] = (
    "compute_time",
    "total_time",
    "communication_bytes",
    "memory_bytes",
    "input_size",
    "result_size",
)


def _number(value: Any) -> float | None:
    """数值原样取（bool 不算数：状态位不能当读数）。"""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _integer(value: Any) -> int | None:
    """整数原样取；非整数**不截断**（截断等于造假数据）。"""

    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _flag(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sum_integers(*values: Any) -> int | None:
    """全部有值才求和（缺一个就整体记 None，不做「有几个算几个」）。"""

    numbers = [_integer(value) for value in values]
    if any(number is None for number in numbers):
        return None
    return sum(number for number in numbers if number is not None)


def _sum_numbers(*values: Any) -> float | None:
    numbers = [_number(value) for value in values]
    if any(number is None for number in numbers):
        return None
    return sum(number for number in numbers if number is not None)


def _difference(minuend: Any, subtrahend: Any) -> float | None:
    top = _number(minuend)
    bottom = _number(subtrahend)
    if top is None or bottom is None:
        return None
    return top - bottom


def _mb_to_bytes(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(round(number * BYTES_PER_MB))


def _psi_curve_relation(protocol: str | None) -> str | None:
    if not protocol:
        return None
    try:
        return psi_curve_relation(protocol)
    except ValueError:
        return None


def _psi_result_semantics(protocol: str | None) -> str | None:
    if not protocol:
        return None
    try:
        return protocol_result_semantics(protocol)
    except ValueError:
        return None


def _psi_world_size(protocol: str | None) -> int | None:
    if not protocol:
        return None
    try:
        normalize_psi_protocol(protocol)
    except ValueError:
        return None
    return protocol_world_size(protocol)


def _mpc_protocol_spec(protocol: str | None):
    if not protocol:
        return None
    try:
        name = normalize_mpc_protocol(protocol)
    except ValueError:
        return None
    return MPC_PROTOCOL_SPECS.get(name)


def _mpc_world_size(protocol: str | None) -> int | None:
    spec = _mpc_protocol_spec(protocol)
    return None if spec is None else spec.world_size


def _mpc_result_semantics(protocol: str | None) -> str | None:
    spec = _mpc_protocol_spec(protocol)
    return None if spec is None else spec.result_semantics


@dataclass(frozen=True)
class PSIBenchmarkMetadata:
    """PSI 特有指标（§九 的 PSI 档）；来路见 `metadata_from_psi_record`。"""

    case: str | None
    curve: str | None
    curve_relation: str | None
    low_comm_mode: bool | None
    n_left: int | None
    n_right: int | None
    n_left_unique: int | None
    n_right_unique: int | None
    intersection_ratio: float | None
    intersection_ratio_actual: float | None
    duplicate_ratio: float | None
    duplicate_count: int | None
    high_bits: bool | None
    level: int | None
    z: int | None
    layout_id: str | None
    agreement: bool | None
    result_semantics: str | None
    input_io_time: float | None
    protocol_time: float | None
    semantic_processing_time: float | None
    peak_memory_mb: float | None
    raw: Mapping[str, Any] = dataclass_field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_dict(self)


@dataclass(frozen=True)
class PsiCaBenchmarkMetadata:
    """PSI-CA 特有指标（openmined-psi 计数档）；来路见
    `metadata_from_psi_ca_record`。"""

    case: str | None
    structure: str | None
    protocol_leak: str | None
    result_semantics: str | None
    n_left: int | None
    n_right: int | None
    n_left_unique: int | None
    n_right_unique: int | None
    intersection_ratio: float | None
    intersection_ratio_actual: float | None
    duplicate_ratio: float | None
    duplicate_count: int | None
    high_bits: bool | None
    level: int | None
    z: int | None
    layout_id: str | None
    agreement: bool | None
    protocol_time: float | None
    #: Phase 11 计量：协议消息的 protobuf 载荷（未计量时为 None，不填 0）
    send_bytes: float | None
    recv_bytes: float | None
    total_bytes: float | None
    peak_memory_mb: float | None
    raw: Mapping[str, Any] = dataclass_field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_dict(self)


@dataclass(frozen=True)
class PsiSumBenchmarkMetadata:
    """PI-Sum 特有指标（private-join-and-compute 交集内求和档）；来路见
    `metadata_from_psi_sum_record`。"""

    case: str | None
    upstream: str | None
    upstream_commit: str | None
    paillier_modulus_size: int | None
    protocol_leak: str | None
    result_semantics: str | None
    value_function: str | None
    n_left: int | None
    n_right: int | None
    n_left_unique: int | None
    n_right_unique: int | None
    intersection_ratio: float | None
    duplicate_ratio: float | None
    duplicate_count: int | None
    high_bits: bool | None
    level: int | None
    z: int | None
    layout_id: str | None
    agreement: bool | None
    intersection_sum: int | None
    intersection_sum_expected: int | None
    protocol_time: float | None
    #: Phase 11 计量：输入 CSV 落盘段耗时（ms；结果走 stdout，无读取段）
    input_io_time: float | None
    #: Phase 10 计量：回环中继两方向字节数（未计量时为 None，不填 0）
    send_bytes: float | None
    recv_bytes: float | None
    total_bytes: float | None
    #: Phase 10 计量：两个子进程 procfs VmHWM 探针值取大者（MiB）
    peak_memory_mb: float | None
    raw: Mapping[str, Any] = dataclass_field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_dict(self)


@dataclass(frozen=True)
class MPCBenchmarkMetadata:
    """MPC 特有指标（§九 的 MPC 档）；来路见 `metadata_from_mpc_record`。"""

    case: str | None
    strategy: str | None
    field_bits: int | None
    k: int | None
    repeat: int | None
    output_bits: int | None
    max_abs_error: float | None
    tolerance: float | None
    deviation_rate: float | None
    agreement: bool | None
    result_semantics: str | None
    protocol_time: float | None
    send_bytes: float | None
    recv_bytes: float | None
    total_bytes: float | None
    peak_memory_mb: float | None
    profiled: bool | None
    pphlo_bytes: int | None
    by_primitive: Mapping[str, Any] = dataclass_field(repr=False)
    raw: Mapping[str, Any] = dataclass_field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_dict(self)


def _dataclass_dict(instance: Any) -> dict[str, Any]:
    """dataclass → dict（字段顺序即声明顺序；`raw` 原样带出，不做深拷贝）。"""

    return {
        name: getattr(instance, name) for name in type(instance).__dataclass_fields__
    }


def metadata_from_psi_record(raw: Mapping[str, Any]) -> PSIBenchmarkMetadata:
    """按 `docs/BENCHMARK_PROTOCOL.md` §3 的字段口径取 PSI 指标。"""

    protocol = _text(raw.get("protocol"))
    return PSIBenchmarkMetadata(
        case=_text(raw.get("case")),
        curve=_text(raw.get("curve")),
        curve_relation=_psi_curve_relation(protocol),
        low_comm_mode=_flag(raw.get("low_comm_mode")),
        n_left=_integer(raw.get("n_left")),
        n_right=_integer(raw.get("n_right")),
        n_left_unique=_integer(raw.get("n_left_unique")),
        n_right_unique=_integer(raw.get("n_right_unique")),
        intersection_ratio=_number(raw.get("intersection_ratio")),
        intersection_ratio_actual=_number(raw.get("intersection_ratio_actual")),
        duplicate_ratio=_number(raw.get("duplicate_ratio")),
        duplicate_count=_integer(raw.get("duplicate_count")),
        high_bits=_flag(raw.get("high_bits")),
        level=_integer(raw.get("level")),
        z=_integer(raw.get("z")),
        layout_id=_text(raw.get("layout_id")),
        agreement=_flag(raw.get("agreement")),
        result_semantics=_psi_result_semantics(protocol),
        input_io_time=_sum_numbers(raw.get("io_write_ms"), raw.get("io_read_ms")),
        protocol_time=_number(raw.get("psi_execute_ms")),
        semantic_processing_time=_number(raw.get("semantic_ms")),
        peak_memory_mb=_number(raw.get("peak_rss_mb")),
        raw=raw,
    )


def metadata_from_psi_ca_record(raw: Mapping[str, Any]) -> PsiCaBenchmarkMetadata:
    """按 PSI-CA 基线运行器（`tests/benchmarks/benchmark_psi_ca.py`）的字段口径取指标。"""

    return PsiCaBenchmarkMetadata(
        case=_text(raw.get("case")),
        structure=_text(raw.get("structure")),
        protocol_leak=_text(raw.get("protocol_leak")),
        result_semantics=_text(raw.get("result_semantics")),
        n_left=_integer(raw.get("n_left")),
        n_right=_integer(raw.get("n_right")),
        n_left_unique=_integer(raw.get("n_left_unique")),
        n_right_unique=_integer(raw.get("n_right_unique")),
        intersection_ratio=_number(raw.get("intersection_ratio")),
        intersection_ratio_actual=_number(raw.get("intersection_ratio_actual")),
        duplicate_ratio=_number(raw.get("duplicate_ratio")),
        duplicate_count=_integer(raw.get("duplicate_count")),
        high_bits=_flag(raw.get("high_bits")),
        level=_integer(raw.get("level")),
        z=_integer(raw.get("z")),
        layout_id=_text(raw.get("layout_id")),
        agreement=_flag(raw.get("agreement")),
        protocol_time=_number(raw.get("psi_ca_execute_ms")),
        send_bytes=_number(raw.get("send_bytes")),
        recv_bytes=_number(raw.get("recv_bytes")),
        total_bytes=_number(raw.get("total_bytes")),
        peak_memory_mb=_number(raw.get("peak_rss_mb")),
        raw=raw,
    )


def metadata_from_psi_sum_record(raw: Mapping[str, Any]) -> PsiSumBenchmarkMetadata:
    """按 PI-Sum 基线运行器（`tests/benchmarks/benchmark_psi_sum.py`）的字段口径取指标。"""

    return PsiSumBenchmarkMetadata(
        case=_text(raw.get("case")),
        upstream=_text(raw.get("upstream")),
        upstream_commit=_text(raw.get("upstream_commit")),
        paillier_modulus_size=_integer(raw.get("paillier_modulus_size")),
        protocol_leak=_text(raw.get("protocol_leak")),
        result_semantics=_text(raw.get("result_semantics")),
        value_function=_text(raw.get("value_function")),
        n_left=_integer(raw.get("n_left")),
        n_right=_integer(raw.get("n_right")),
        n_left_unique=_integer(raw.get("n_left_unique")),
        n_right_unique=_integer(raw.get("n_right_unique")),
        intersection_ratio=_number(raw.get("intersection_ratio")),
        duplicate_ratio=_number(raw.get("duplicate_ratio")),
        duplicate_count=_integer(raw.get("duplicate_count")),
        high_bits=_flag(raw.get("high_bits")),
        level=_integer(raw.get("level")),
        z=_integer(raw.get("z")),
        layout_id=_text(raw.get("layout_id")),
        agreement=_flag(raw.get("agreement")),
        intersection_sum=_integer(raw.get("intersection_sum")),
        intersection_sum_expected=_integer(raw.get("intersection_sum_expected")),
        protocol_time=_number(raw.get("pi_sum_execute_ms")),
        input_io_time=_number(raw.get("io_write_ms")),
        send_bytes=_number(raw.get("send_bytes")),
        recv_bytes=_number(raw.get("recv_bytes")),
        total_bytes=_number(raw.get("total_bytes")),
        peak_memory_mb=_number(raw.get("peak_rss_mb")),
        raw=raw,
    )


def metadata_from_mpc_record(raw: Mapping[str, Any]) -> MPCBenchmarkMetadata:
    """按 `docs/MPC_BENCHMARK_PROTOCOL.md` 的字段口径取 MPC 指标。"""

    protocol = _text(raw.get("protocol"))
    return MPCBenchmarkMetadata(
        case=_text(raw.get("case")),
        strategy=_text(raw.get("strategy")),
        field_bits=_integer(raw.get("field_bits")),
        k=_integer(raw.get("k")),
        repeat=_integer(raw.get("repeat")),
        output_bits=_integer(raw.get("output_bits")),
        max_abs_error=_number(raw.get("max_abs_error")),
        tolerance=_number(raw.get("tolerance")),
        deviation_rate=_number(raw.get("deviation_rate")),
        agreement=_flag(raw.get("agreement")),
        result_semantics=_mpc_result_semantics(protocol),
        protocol_time=_difference(raw.get("wall_ms"), raw.get("setup_ms")),
        send_bytes=_number(raw.get("comm_send_bytes")),
        recv_bytes=_number(raw.get("comm_recv_bytes")),
        total_bytes=_number(raw.get("comm_total_bytes")),
        peak_memory_mb=_number(raw.get("peak_rss_mb")),
        profiled=_flag(raw.get("profiled")),
        pphlo_bytes=_integer(raw.get("pphlo_bytes")),
        by_primitive=raw.get("comm_by_primitive") or {},
        raw=raw,
    )


@dataclass(frozen=True)
class CommonBenchmarkRecord:
    """各族共有的 13 个字段 + 本族指标（`metadata`）。"""

    family: str
    protocol: str
    operation: str | None
    world_size: int | None
    field: str | None
    input_size: int | None
    unique_size: int | None
    result_size: int | None
    compute_time: float | None
    communication_bytes: float | None
    total_time: float | None
    memory_bytes: int | None
    status: str
    metadata: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = {name: getattr(self, name) for name in COMMON_FIELDS}
        row["metadata"] = dict(self.metadata)
        return row


def psi_record(raw: Mapping[str, Any]) -> CommonBenchmarkRecord:
    """把一条 PSI 基线记录投影成共有字段（口径见 `FIELD_SEMANTICS`）。"""

    metadata = metadata_from_psi_record(raw)
    protocol = _text(raw.get("protocol"))
    return CommonBenchmarkRecord(
        family=BENCHMARK_FAMILY_PSI,
        protocol=protocol or "",
        operation=_text(raw.get("operation")) or PSI_BENCHMARK_OPERATION,
        world_size=_psi_world_size(protocol),
        field=(
            metadata.curve
            if metadata.curve_relation == CURVE_RELATION_REQUIRED
            else None
        ),
        input_size=_sum_integers(raw.get("n_left"), raw.get("n_right")),
        unique_size=_sum_integers(
            raw.get("n_left_unique"), raw.get("n_right_unique")
        ),
        result_size=_integer(raw.get("intersection_count")),
        compute_time=metadata.protocol_time,
        communication_bytes=_number(raw.get("comm_total_bytes")),
        total_time=_number(raw.get("total_ms")),
        memory_bytes=_mb_to_bytes(raw.get("peak_rss_mb")),
        status=_text(raw.get("status")) or STATUS_UNKNOWN,
        metadata=metadata.to_dict(),
    )


def psi_ca_record(raw: Mapping[str, Any]) -> CommonBenchmarkRecord:
    """把一条 PSI-CA 基线记录投影成共有字段（口径见 `FIELD_SEMANTICS`）。"""

    metadata = metadata_from_psi_ca_record(raw)
    protocol = _text(raw.get("protocol"))
    return CommonBenchmarkRecord(
        family=BENCHMARK_FAMILY_PSI_CA,
        protocol=protocol or "",
        operation=_text(raw.get("operation")) or PSI_BENCHMARK_OPERATION,
        world_size=_integer(raw.get("world_size")),
        field=None,
        input_size=_sum_integers(raw.get("n_left"), raw.get("n_right")),
        unique_size=_sum_integers(
            raw.get("n_left_unique"), raw.get("n_right_unique")
        ),
        result_size=_integer(raw.get("intersection_count")),
        compute_time=metadata.protocol_time,
        # Phase 11：协议消息载荷合计（进程内链路；口径见 PSI_CA_CAPABILITY.md §7.1）
        communication_bytes=metadata.total_bytes,
        total_time=_number(raw.get("total_ms")),
        memory_bytes=_mb_to_bytes(raw.get("peak_rss_mb")),
        status=_text(raw.get("status")) or STATUS_UNKNOWN,
        metadata=metadata.to_dict(),
    )


def psi_sum_record(raw: Mapping[str, Any]) -> CommonBenchmarkRecord:
    """把一条 PI-Sum 基线记录投影成共有字段（口径见 `FIELD_SEMANTICS`）。"""

    metadata = metadata_from_psi_sum_record(raw)
    protocol = _text(raw.get("protocol"))
    return CommonBenchmarkRecord(
        family=BENCHMARK_FAMILY_PSI_SUM,
        protocol=protocol or "",
        operation=_text(raw.get("operation")) or PSI_BENCHMARK_OPERATION,
        world_size=_integer(raw.get("world_size")),
        field=None,
        input_size=_sum_integers(raw.get("n_left"), raw.get("n_right")),
        unique_size=_sum_integers(
            raw.get("n_left_unique"), raw.get("n_right_unique")
        ),
        result_size=_integer(raw.get("intersection_count")),
        compute_time=metadata.protocol_time,
        # Phase 10：回环中继计量（未计量时为 None，不填 0）
        communication_bytes=metadata.total_bytes,
        total_time=_number(raw.get("total_ms")),
        memory_bytes=_mb_to_bytes(raw.get("peak_rss_mb")),
        status=_text(raw.get("status")) or STATUS_UNKNOWN,
        metadata=metadata.to_dict(),
    )


def mpc_record(raw: Mapping[str, Any]) -> CommonBenchmarkRecord:
    """把一条 MPC 基线记录投影成共有字段（口径见 `FIELD_SEMANTICS`）。"""

    metadata = metadata_from_mpc_record(raw)
    protocol = _text(raw.get("protocol"))
    return CommonBenchmarkRecord(
        family=BENCHMARK_FAMILY_MPC,
        protocol=protocol or "",
        operation=_text(raw.get("op")),
        world_size=_mpc_world_size(protocol),
        field=_text(raw.get("field")),
        input_size=_integer(raw.get("k")),
        unique_size=None,
        result_size=None,
        compute_time=metadata.protocol_time,
        communication_bytes=metadata.total_bytes,
        total_time=_number(raw.get("wall_ms")),
        memory_bytes=_mb_to_bytes(raw.get("peak_rss_mb")),
        status=_text(raw.get("status")) or STATUS_UNKNOWN,
        metadata=metadata.to_dict(),
    )


#: 族 → 适配器
_RECORD_ADAPTERS: Mapping[
    str, Callable[[Mapping[str, Any]], CommonBenchmarkRecord]
] = {
    BENCHMARK_FAMILY_PSI: psi_record,
    BENCHMARK_FAMILY_PSI_CA: psi_ca_record,
    BENCHMARK_FAMILY_PSI_SUM: psi_sum_record,
    BENCHMARK_FAMILY_MPC: mpc_record,
}


def unify_records(
    family: str, records: Iterable[Mapping[str, Any]]
) -> tuple[CommonBenchmarkRecord, ...]:
    """把一族的一批原始记录投影成共有记录（顺序保持原样）。"""

    adapter = _RECORD_ADAPTERS.get(family)
    if adapter is None:
        raise ValueError(f"未知 benchmark 族 {family!r}；可用：{BENCHMARK_FAMILIES}")
    return tuple(adapter(record) for record in records)


def status_counts(
    records: Iterable[CommonBenchmarkRecord],
) -> dict[str, int]:
    """按状态计数；顺序固定为 ok → unavailable → error → 其余按名字排序。"""

    counts: dict[str, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    preferred = (STATUS_OK, STATUS_UNAVAILABLE, STATUS_ERROR)
    ordered = {name: counts[name] for name in preferred if name in counts}
    for name in sorted(set(counts) - set(preferred)):
        ordered[name] = counts[name]
    return ordered


def infer_family(record: Mapping[str, Any]) -> str:
    """按记录里**实际存在的列**判断族；判不出就报错，不猜。

    三条 PSI 路径靠各自的执行耗时列区分（`psi_ca_execute_ms` /
    `pi_sum_execute_ms` / `psi_execute_ms`）；这两条检查必须在通用 PSI
    规则之前——新路径的记录同样带 `n_left`。
    """

    if "psi_ca_execute_ms" in record:
        return BENCHMARK_FAMILY_PSI_CA
    if "pi_sum_execute_ms" in record:
        return BENCHMARK_FAMILY_PSI_SUM
    if "op" in record and "k" in record:
        return BENCHMARK_FAMILY_MPC
    if "psi_execute_ms" in record or "n_left" in record:
        return BENCHMARK_FAMILY_PSI
    raise ValueError(
        f"无法判断记录的协议族（不像已登记的四个族）：{sorted(record)}"
    )


def load_benchmark_file(
    path: str, family: str | None = None
) -> tuple[CommonBenchmarkRecord, ...]:
    """读一份基线 JSON（记录列表）并投影。

    不给族就按第一条推断，随后**逐条核对同族**——混装的文件直接报错，
    不按第一条的族硬套整份文件。
    """

    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: 期望非空记录列表，实际是 {type(raw).__name__}")
    resolved = family or infer_family(raw[0])
    if resolved not in BENCHMARK_FAMILIES:
        raise ValueError(f"未知 benchmark 族 {resolved!r}；可用：{BENCHMARK_FAMILIES}")
    for record in raw:
        found = infer_family(record)
        if found != resolved:
            raise ValueError(
                f"{path}: 族不一致——按 {resolved} 读取，却遇到 {found} 记录"
            )
    return unify_records(resolved, raw)


@dataclass(frozen=True)
class ComparedRow:
    """一个显示名在某一指标上的一行读数。"""

    key: str
    value: float
    samples: int
    spread: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "samples": self.samples,
            "spread": self.spread,
        }


@dataclass(frozen=True)
class ProtocolComparison:
    """一次「同族、同算子、同规模」下比较协议的结果。"""

    metric: str
    input_size: int | None
    rows: tuple[ComparedRow, ...]
    skipped: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "input_size": self.input_size,
            "rows": [row.to_dict() for row in self.rows],
            "skipped": [
                {"key": key, "reason": reason} for key, reason in self.skipped
            ],
        }


def _median(values: Sequence[float]) -> float:
    """中位数（偶数个取中间两个的平均）。"""

    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _require_single(
    records: Sequence[CommonBenchmarkRecord],
    what: str,
    getter: Callable[[CommonBenchmarkRecord], object],
    hint: str,
) -> None:
    """筛完还剩多个「不可比的维度」时直接报错（不把不可比的读数排成一张榜）。"""

    values = {getter(record) for record in records}
    if len(values) > 1:
        readable = sorted(str(value) for value in values)
        raise ValueError(
            f"筛选后存在多个{what} {readable}；请用 {hint} 收窄到单个再比较"
        )


def protocol_label(record: CommonBenchmarkRecord) -> str:
    """比较用的显示名：优先用原始用例名（`case` 本身已含协议名与规模），
    没有 `case` 时退回协议名。
    """

    case = record.metadata.get("case")
    if case:
        return str(case)
    return record.protocol


def compare_protocols(
    records: Iterable[CommonBenchmarkRecord],
    metric: str = "total_time",
    *,
    family: str | None = None,
    operation: str | None = None,
    input_size: int | None = None,
) -> ProtocolComparison:
    """按共有字段比较协议（验收 §9「Benchmark 能比较不同协议」）。

    纪律：
    - **只比较 `status == ok` 的行**；未执行 / 报错的行进 `skipped`，不给数字；
    - 记录里没有该指标的行也进 `skipped`（不拿别的数凑）；
    - **不同族 / 不同算子 / 不同规模不比**：筛完还剩多个就直接报错，
      宁可让调用方显式收窄，也不把不可比的读数排成一张榜；
    - 同一显示名出现多条 `ok` 记录（基线里确有同用例重测的多行）时取
      **中位数**——与 MPC 运行器对重复实验的聚合口径一致
      （`backends/spu_backend/benchmark.py`：`wall_ms` 取各次中位数），
      并把样本数与极差一并带出：读数稳不稳要看得见，不是抹平。
    """

    if metric not in COMPARABLE_METRICS:
        raise ValueError(f"不可比较的字段 {metric!r}；可用：{COMPARABLE_METRICS}")
    selected = [
        record
        for record in records
        if (family is None or record.family == family)
        and (operation is None or record.operation == operation)
        and (input_size is None or record.input_size == input_size)
    ]
    _require_single(selected, "族", lambda record: record.family, "family=")
    _require_single(selected, "算子", lambda record: record.operation, "operation=")
    _require_single(selected, "规模", lambda record: record.input_size, "input_size=")

    samples: dict[str, list[float]] = {}
    skipped: list[tuple[str, str]] = []
    for record in selected:
        key = protocol_label(record)
        if record.status != STATUS_OK:
            skipped.append((key, f"status={record.status}（未执行的行不给数字）"))
            continue
        value = getattr(record, metric)
        if value is None:
            skipped.append((key, f"记录里没有 {metric}"))
            continue
        samples.setdefault(key, []).append(float(value))

    rows = [
        ComparedRow(
            key=key,
            value=_median(values),
            samples=len(values),
            spread=(max(values) - min(values)) if len(values) > 1 else None,
        )
        for key, values in samples.items()
    ]
    rows.sort(key=lambda row: row.value)
    sizes = {record.input_size for record in selected}
    return ProtocolComparison(
        metric=metric,
        input_size=sizes.pop() if len(sizes) == 1 else None,
        rows=tuple(rows),
        skipped=tuple(skipped),
    )
