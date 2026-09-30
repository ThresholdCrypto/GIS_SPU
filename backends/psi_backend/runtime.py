# -*- coding: utf-8 -*-
"""PSI 运行时：把格网集合交给官方 PSI 求交，并给出语义判定。

执行路径严格对齐 spu 0.9.5 官方 PSI：

    desc = libspu.link.Desc(); add_party(...)
    lctx = libspu.link.create_mem(desc, rank)      # 每个参与方一个线程
    cfg  = psi.PsiExecuteConfig(...)
    rep  = psi.psi_execute(cfg, lctx)

**不实现任何协议**；`run_psi_intersection` 只做"文件搬运 + 配置装配 + 结果解读"。

关于泄漏面的诚实说明
-------------------
PSI 的标准语义是"接收方得到**交集本体**"。也就是说，即使业务只需要一个布尔值，
PSI 也会把交集元素交给接收方。这**比布尔结果泄露更多**：

- `Intersects`：本可用基数是否 >0 表达，但 PSI 会把交集元素给接收方；
  本后端在此之上做"取交集→算是否非空"，**交集本身已在接收方侧出现**。
- `Contains`：需要子集关系判定。PSI 出交集后还需一次比较：`|A∩B| == |A|`
  （A 为被包含方）。这一步**默认在 MPC 里完成**（见 `subset_mpc`），只把两个
  基数当作两方各自的私有输入，输出一个布尔；不再做明文比较。仅在 MPC
  不可用时会**已披露地**退回明文，`subset.mode` 与状态词都会随之变化。
- `CellSetIntersect`：交集本体正是业务要的结果，不额外泄露。

每个算子都在 `PSI_OP_LEAKS` 里如实登记，并由 `PsiRunResult.reveals` 带出。

关于结果精度的诚实说明
--------------------
协议清单里混着两类东西：**精确**协议与**带噪**协议。`PROTOCOL_DP` 是差分隐私
PSI，结果会被子采样/上采样扰动，实测会漏报真实交集（见
`capability.PSI_PROTOCOLS_WITH_NOISE`）。因此本模块：

- 每个带噪协议的结果都带上 `_NOISE_NOTE`，不靠调用方转述；
- `agreement=False` 对带噪协议**不**升级为 `error`——那是协议语义而非执行缺陷，
  升级会让一次正常执行被随机报成失败；
- 调用方（`geosecure.compiler`）据此拒绝把带噪结果称作"已验证"。
"""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .capability import (
    PSI_DEFAULT_CURVE,
    PSI_DEFAULT_PROTOCOL,
    PSI_PROTOCOLS_WITH_NOISE,
    PSI_RUNTIME_WORLD_SIZE,
    PsiCapabilityReport,
    check_psi_capabilities,
    normalize_curve,
    normalize_psi_protocol,
    protocol_is_exact,
    protocol_needs_curve,
    protocol_world_size,
    runnable_protocols_hint,
)
from .subset_mpc import (
    SUBSET_DEFAULT_FIELD,
    SUBSET_DEFAULT_PROTOCOL,
    SUBSET_MODE_MPC,
    SUBSET_MODE_PLAINTEXT,
    SUBSET_MODE_PLAINTEXT_FALLBACK,
    SubsetComparison,
    mpc_subset_comparison,
    plaintext_subset_comparison,
    resolve_subset_mode,
)

#: 每个算子在 PSI 之上会暴露给接收方的信息（诚实登记，不淡化）
PSI_OP_LEAKS: Mapping[str, str] = {
    "Intersects": (
        "接收方获得交集本体（不只是布尔值）；交集基数由 recipient 可见"
    ),
    "Contains": (
        "接收方获得交集本体（PSI 标准语义）；子集判定只输出一个布尔，"
        "其执行形态（MPC / 明文）另行披露，见 subset 字段"
    ),
    "CellSetIntersect": "接收方获得交集本体（即业务所需结果）",
}

#: 各算子从交集推导业务结果的语义
_OP_SEMANTICS: Mapping[str, str] = {
    "Intersects": "非空判定：|A∩B| > 0",
    "Contains": (
        "子集判定：inner ⊆ (outer∩inner)，等价于 |outer∩inner| == |inner|"
        "（后者即 MPC 电路的判据：只比较两个基数，不搬集合）"
    ),
    "CellSetIntersect": "交集本体",
}

#: 带噪协议的固定说明（每个 run 都带出，不靠调用方转述）
_NOISE_NOTE = (
    "所选协议在上述带噪清单中（差分隐私 PSI）：交集里会注入假元素/丢弃真元素，"
    "结果与明文不一致属**预期行为**，不可用于一致性验证"
)

_KEY_COLUMN = "grid_code"


@dataclass
class PsiRunResult:
    """一次真实 PSI 求交的结果。"""

    status: str  # ok / unavailable / error
    op: str
    protocol: str
    curve: str | None
    world_size: int
    receiver_rank: int
    #: 实际注入的协议级参数（如 RR22 的 low_comm_mode）；空 = 无协议级参数。
    #: 只登记真实注入过的值——"传了但协议不读"不写成"已生效"。
    protocol_params: Mapping[str, Any] = field(default_factory=dict)
    # 原始 PSI 计数（来自官方 PsiExecuteReport）
    original_count: int | None = None
    intersection_count: int | None = None
    intersection_unique_count: int | None = None
    # 业务语义结果
    value: Any = None
    intersection: tuple[int, ...] = ()
    reference: Any = None
    agreement: bool | None = None
    reveals: str = ""
    semantics: str = ""
    #: `Contains` 的子集判定结果（MPC / 明文 / 已披露退路）；其余算子为 None
    subset: SubsetComparison | None = None
    blockers: tuple[str, ...] = ()
    error: str | None = None
    notes: tuple[str, ...] = ()
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "op": self.op,
            "protocol": self.protocol,
            "curve": self.curve,
            "protocol_params": dict(self.protocol_params),
            "world_size": self.world_size,
            "receiver_rank": self.receiver_rank,
            "original_count": self.original_count,
            "intersection_count": self.intersection_count,
            "intersection_unique_count": self.intersection_unique_count,
            "value": _jsonable(self.value),
            "intersection": list(self.intersection),
            "reference": _jsonable(self.reference),
            "agreement": self.agreement,
            "reveals": self.reveals,
            "semantics": self.semantics,
            "subset": self.subset.to_dict() if self.subset is not None else None,
            "blockers": list(self.blockers),
            "error": self.error,
            "notes": list(self.notes),
            "timings_ms": dict(self.timings_ms),
        }

    def describe(self) -> str:
        if self.ok:
            lines = [
                f"PSI simulation: OK  (op={self.op}, protocol={self.protocol}"
                + (f", curve={self.curve}" if self.curve else "")
                + f", wsize={self.world_size})",
                f"  semantics    : {self.semantics}",
                f"  |A|          : {self.original_count}   |A∩B| = {self.intersection_count}",
            ]
            if self.protocol_params:
                lines.append(f"  protocol_params: {dict(self.protocol_params)}")
            if self.value is not None:
                lines.append(f"  result       : {_jsonable(self.value)}")
            if self.reference is not None:
                lines.append(
                    f"  plaintext    : {_jsonable(self.reference)}   "
                    f"agreement={self.agreement}"
                )
            if self.intersection:
                shown = list(self.intersection[:4])
                more = "" if len(self.intersection) <= 4 else f" …(+{len(self.intersection)-4})"
                lines.append(f"  intersection : {shown}{more}")
            if self.subset is not None:
                lines.append(
                    f"  subset       : {_jsonable(self.subset.value)}  "
                    f"(mode={self.subset.mode}, status={self.subset.status}"
                    + (f", protocol={self.subset.protocol}" if self.subset.protocol else "")
                    + ")"
                )
            lines.append(f"  reveals      : {self.reveals}")
            for note in self.notes:
                lines.append(f"  note         : {note}")
            return "\n".join(lines)

        lines = [
            f"PSI simulation: {self.status.upper()}  (op={self.op}, protocol={self.protocol})"
        ]
        if self.error:
            lines.append(f"  error        : {self.error}")
        for blocker in self.blockers:
            lines.append(f"  blocker      : {blocker}")
        for note in self.notes:
            lines.append(f"  note         : {note}")
        return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


# --------------------------------------------------------------------------
# 输入输出：PSI 只吃 CSV，用临时目录承载
# --------------------------------------------------------------------------


def _write_keys(path: str, codes: Sequence[int]) -> None:
    """把一个参与方的格网键集合写成 PSI 可读的 CSV。"""

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([_KEY_COLUMN])
        for code in codes:
            code = int(code)
            if not 0 <= code < 2**64:
                raise ValueError(f"grid_code 必须落在 [0, 2^64)：{code}")
            writer.writerow([code])


def _read_intersection(path: str) -> tuple[int, ...]:
    """读回接收方侧的交集 CSV。"""

    if not os.path.exists(path):
        return ()
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return ()
    header = [c.strip().strip('"') for c in rows[0]]
    try:
        idx = header.index(_KEY_COLUMN)
    except ValueError:
        idx = 0
    out: list[int] = []
    for row in rows[1:]:
        if not row or idx >= len(row):
            continue
        cell = row[idx].strip().strip('"')
        if not cell or cell == "NULL":
            continue
        out.append(int(cell))
    return tuple(sorted(out))


def _reveals_with_subset(op: str, subset: SubsetComparison | None) -> str:
    """泄漏面 = 静态核心 + 子集判定形态的动态披露。

    静态表说不出"这一次用的是 MPC 还是明文"，所以两者不能合并成一句话：
    合并后总有人只读到一半。
    """

    base = PSI_OP_LEAKS.get(op, "（未登记泄漏面，请补充）")
    if subset is None:
        return base
    return f"{base}；{subset.disclosure}"


def _subset_notes(subset: SubsetComparison) -> tuple[str, ...]:
    notes = [f"子集判定模式：{subset.mode}（status={subset.status}）"]
    if subset.note:
        notes.append(subset.note)
    if subset.mode == SUBSET_MODE_PLAINTEXT_FALLBACK:
        notes.append(
            "本次子集判定是明文比较的退路：两方部署下它需要把 inner 交给接收方，"
            "密态属性不完整，不能当作密态子集比较的验证依据"
        )
    return tuple(notes)


def _resolve_subset(
    *,
    intersection: Sequence[int],
    inner: Any,
    subset_via: str,
    protocol: str,
    field: str | int,
    world_size: int | None,
    report: Any,
) -> SubsetComparison:
    """`Contains` 的子集判定：默认 MPC；MPC 不可用时**披露式**退回明文。

    退回明文是刻意选择而非偷懒：旧行为就是明文，若直接留空，`Contains`
    会在没有 SPU 的环境上从"能跑"变成"不能跑"。所以退回，但把 mode 标成
    `plaintext-fallback`——状态词、reveals、notes 三处都会跟着变，
    调用方不可能把它误读成密态子集比较。
    """

    # 只算两个基数，不搬集合；这正是"密态子集比较"能便宜地做出来的原因。
    intersection_card = len(set(int(v) for v in intersection))
    inner_card = len(set(int(v) for v in inner))

    if subset_via == SUBSET_MODE_MPC:
        comparison = mpc_subset_comparison(
            intersection_card,
            inner_card,
            protocol=protocol,
            field=field,
            world_size=world_size,
            report=report,
        )
        if comparison.value is not None:
            return comparison
        return plaintext_subset_comparison(
            intersection,
            inner,
            mode=SUBSET_MODE_PLAINTEXT_FALLBACK,
            note=f"MPC 子集比较未能执行：{comparison.note}",
        )

    return plaintext_subset_comparison(
        intersection, inner, mode=SUBSET_MODE_PLAINTEXT
    )

# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------


def run_psi_intersection(
    left: Sequence[int],
    right: Sequence[int],
    *,
    op: str = "Intersects",
    protocol: str = PSI_DEFAULT_PROTOCOL,
    curve: str | None = PSI_DEFAULT_CURVE,
    receiver_rank: int = 0,
    rr22_low_comm_mode: bool = False,
    reference_fn: Any = None,
    report: PsiCapabilityReport | None = None,
    workdir: str | None = None,
    quiet: bool = True,
    subset_via: str = SUBSET_MODE_MPC,
    mpc_protocol: str = SUBSET_DEFAULT_PROTOCOL,
    mpc_field: str | int = SUBSET_DEFAULT_FIELD,
    mpc_world_size: int | None = None,
    mpc_report: Any = None,
) -> PsiRunResult:
    """两方格网集合求交，并给出该算子的业务语义结果。

    Args:
        left / right: 两方的 64 位格网码集合（`CellSetIntersect` 时 left=outer）。
        op:   受支持算子之一（见 PSI_OPS）。
        protocol / curve: PSI 协议与椭圆曲线；ECDH 族必须给曲线。
        receiver_rank: 谁拿交集（0 或 1）；默认 0。
        rr22_low_comm_mode: RR22 专用参数（`Rr22Rarams.low_comm_mode`，低通信
            模式）。仅当 protocol=RR22 时注入并登记进结果；其它协议下不生效，
            且会在 notes 里如实说明——不把"传了但没注入"伪装成"已生效"。
        reference_fn: 明文参考实现；给出后计算一致性。
        workdir: 临时工作目录；缺省自建自删。
        subset_via: `Contains` 的子集判定走哪条路——"mpc"（默认，密态基数
            等值）或 "plaintext"（调用方显式要求明文）。MPC 不可用时自动退回
            明文，但 mode 会标成 "plaintext-fallback" 并整体披露，绝不静默改语义。
        mpc_protocol / mpc_field / mpc_world_size / mpc_report: 子集比较这条
            MPC 电路所用的设置；与上面的 PSI 协议/曲线是两回事。
    """

    report = report or check_psi_capabilities()
    protocol_name = normalize_psi_protocol(protocol)
    curve_name = normalize_curve(curve) if (curve and protocol_needs_curve(protocol_name)) else None

    # RR22 的协议级参数在这里定型：只登记**会实际注入**的参数。
    # 其它协议带出空字典——不把"用户传了"写成"协议读了"。
    protocol_params: dict[str, Any] = {}
    if protocol_name == "PROTOCOL_RR22":
        protocol_params["low_comm_mode"] = bool(rr22_low_comm_mode)

    result = PsiRunResult(
        status="unavailable",
        op=op,
        protocol=protocol_name,
        curve=curve_name,
        protocol_params=protocol_params,
        world_size=2,
        receiver_rank=int(receiver_rank),
        reveals=PSI_OP_LEAKS.get(op, "（未登记泄漏面，请补充）"),
        semantics=_OP_SEMANTICS.get(op, ""),
    )

    # 带噪协议从第一行起就把话说清楚，而不是等结果对不上再解释。
    if protocol_name in PSI_PROTOCOLS_WITH_NOISE:
        result.notes = result.notes + (_NOISE_NOTE,)

    # RR22 专用参数给了、协议却不是 RR22：说了，不静默带过。
    if protocol_name != "PROTOCOL_RR22" and rr22_low_comm_mode:
        result.notes = result.notes + (
            "rr22_low_comm_mode=True 但所选协议不是 RR22："
            "该参数未注入、未生效（结果 protocol_params 为空）",
        )

    if not report.runnable:
        result.blockers = report.blockers
        result.notes = result.notes + (
            "当前环境无法真实执行 PSI；明文结果仍然可用，"
            "PSI 结果栏位保持空缺而不以推测值填充。",
        )
        return result

    if op not in PSI_OP_LEAKS:
        result.status = "error"
        result.error = (
            f"算子在注册表中，但未登记语义/泄漏面。已支持：{tuple(PSI_OP_LEAKS)}"
        )
        return result

    if int(receiver_rank) not in (0, 1):
        result.status = "error"
        result.error = f"receiver_rank 必须是 0 或 1，实得 {receiver_rank}"
        return result

    # 模式归一化与协议名同一条纪律：非法值给出可用清单，不让它变成
    # 走到一半才炸的运行时错误。
    try:
        subset_via = resolve_subset_mode(subset_via)
    except ValueError as exc:
        result.status = "error"
        result.error = str(exc)
        return result

    # 参与方数量前置检查：三方协议在两方链路上必然失败，且官方实现给出的是
    # C++ 栈回溯（Enforce fail at memory_psi.cc:44），不是可读原因。
    # 在这里拦下，错误位置/原因/替代方案三样都给全。
    required = protocol_world_size(protocol_name)
    if required > PSI_RUNTIME_WORLD_SIZE:
        result.status = "error"
        result.error = (
                    f"协议 {protocol_name} 需要 {required} 个参与方，"
                    f"本后端的进程内链路固定 {PSI_RUNTIME_WORLD_SIZE} 方，无法执行该协议；"
                    f"请改用两方可执行协议：{runnable_protocols_hint()}"
                )
        return result

    left_codes = [int(c) for c in left]
    right_codes = [int(c) for c in right]

    # ---------------- 明文参考（先算，失败不影响 PSI） ----------------
    if reference_fn is not None:
        try:
            result.reference = reference_fn(left_codes, right_codes)
        except Exception:
            result.reference = None

    # ---------------- 空输入前置检查 ----------------
    # 实测：官方 PSI 的 CSV 读取走 arrow_csv_batch_provider，要求 CSV 至少有
    # 表头 + 1 行数据；空集合（只有表头）会抛
    #   Enforce fail at arrow_helper.cc:87 ... read csv file second line failed
    # 这不是"隐私计算失败"，而是输入枚举口径问题，故在进入协议前就说清楚。
    empty_side = None
    if not left_codes:
        empty_side = "left"
    elif not right_codes:
        empty_side = "right"
    if empty_side is not None:
        result.status = "empty-input"
        left_set, right_set = set(left_codes), set(right_codes)
        if op == "Intersects":
            result.value = bool(left_set & right_set)
            result.semantics = "非空判定：|A∩B| > 0"
        elif op == "Contains":
            # 严格沿用 `backends.plain.plain_contains` 的口径：inner ⊆ outer。
            # 刻意不设"outer 为空即 False"的特例——明文实现没有该特例，
            # ∅ ⊆ ∅ 在明文侧为 True，密态侧必须一致，否则对拍会假性失败。
            result.value = right_set <= left_set
            result.semantics = "子集判定：inner ⊆ outer"
        else:
            result.value = tuple(sorted(left_set & right_set))
            result.semantics = "交集本体"
        result.notes = result.notes + (
            f"{empty_side} 侧输入为空集合：官方 PSI 的 CSV 通道不接受只有表头的文件，"
            "已按集合论直接给出结果，未启动 PSI 协议",
        )
        if result.reference is not None:
            result.agreement = bool(_norm(result.value) == _norm(result.reference))
        return result

    own_dir = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="geo_psi_")
    try:
        in_left = os.path.join(workdir, "party0.csv")
        in_right = os.path.join(workdir, "party1.csv")
        out_recv = os.path.join(workdir, f"out_rank{int(receiver_rank)}.csv")
        _write_keys(in_left, left_codes)
        _write_keys(in_right, right_codes)

        # ---------------- 导入官方 API ----------------
        try:
            import spu.libspu as libspu
            from spu import psi
        except Exception as exc:
            result.status = "error"
            result.error = f"导入 SPU PSI API 失败：{type(exc).__name__}: {exc}"
            return result

        # 官方 PSI 会把完整协商日志写到 stderr（每轮求交几十行）。
        # 那对排障有用，但会淹没编译器自身的输出，故默认静默、可显式打开。
        _set_native_log(quiet)

        if not hasattr(psi, "psi_execute"):
            result.status = "error"
            result.error = (
                "spu.psi 缺少 psi_execute；当前 SPU 版本与预期 API 不一致，"
                "请核对 docs/PSI_CAPABILITY.md"
            )
            return result

        # RR22 参数类必须真实存在，注入才算数。缺失时：
        # 显式请求 low_comm_mode=True → 直接失败（不能假装设置成功）；
        # 未请求 → 按协议内部默认配置执行，清空参数档并如实加注。
        if protocol_name == "PROTOCOL_RR22" and not hasattr(psi, "Rr22Rarams"):
            if rr22_low_comm_mode:
                result.status = "error"
                result.error = (
                    "当前 spu.psi 没有 Rr22Rarams，无法设置 low_comm_mode=True；"
                    "该 SPU 版本的 RR22 参数能力不完整"
                    "（见 PSI capability 报告的 rr22_params）"
                )
                return result
            result.protocol_params = {}
            result.notes = result.notes + (
                "当前 spu.psi 没有 Rr22Rarams：未注入任何 RR22 参数，"
                "按协议内部默认配置执行",
            )

        # ---------------- 建进程内两方链路 ----------------
        desc = libspu.link.Desc()
        for rank in range(2):
            desc.add_party(f"id_{rank}", f"thread_{rank}")

        inputs = [in_left, in_right]
        outputs = {}
        errors: dict[int, str] = {}

        def worker(rank: int) -> None:
            try:
                lctx = libspu.link.create_mem(desc, rank)
                protocol_conf = psi.PsiProtocolConfig(
                    protocol=getattr(psi.PsiProtocol, protocol_name),
                    receiver_rank=int(receiver_rank),
                    broadcast_result=False,
                )
                if curve_name is not None:
                    protocol_conf.ecdh_params = psi.EcdhParams(
                        curve=getattr(psi.EllipticCurveType, curve_name)
                    )
                if protocol_name == "PROTOCOL_RR22":
                    # RR22 不读 curve，参数只走 Rr22Rarams；可用性已在进入
                    # 线程前核对过。low_comm_mode 用调用方给的实值，不用默认值兜底。
                    protocol_conf.rr22_params = psi.Rr22Rarams(
                        low_comm_mode=bool(rr22_low_comm_mode)
                    )
                cfg = psi.PsiExecuteConfig(
                    protocol_conf=protocol_conf,
                    input_params=psi.InputParams(
                        type=psi.SourceType.SOURCE_TYPE_FILE_CSV,
                        path=inputs[rank],
                        selected_keys=[_KEY_COLUMN],
                        keys_unique=True,
                    ),
                    output_params=psi.OutputParams(
                        type=psi.SourceType.SOURCE_TYPE_FILE_CSV,
                        path=out_recv,
                    ),
                    join_conf=psi.ResultJoinConfig(
                        type=psi.ResultJoinType.JOIN_TYPE_INNER_JOIN,
                        left_side_rank=0,
                    ),
                )
                rep = psi.psi_execute(cfg, lctx)
                outputs[rank] = {
                    "original_count": int(rep.original_count),
                    "intersection_count": int(rep.intersection_count),
                    "intersection_unique_count": int(rep.intersection_unique_count),
                }
            except Exception as exc:  # 线程内异常不能丢，要带回主线程
                errors[rank] = f"{type(exc).__name__}: {exc}"

        threads = [threading.Thread(target=worker, args=(r,)) for r in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            result.status = "error"
            first = sorted(errors)[0]
            result.error = f"PSI 执行失败（rank {first}）：{errors[first]}"
            return result

        recv = outputs.get(int(receiver_rank), {})
        result.original_count = (
            len(set(left_codes)) if int(receiver_rank) == 0 else len(set(right_codes))
        )
        result.intersection_count = recv.get("intersection_count")
        result.intersection_unique_count = recv.get("intersection_unique_count")

        result.intersection = _read_intersection(out_recv)

        # ---------------- 从交集推导业务语义 ----------------
        if op == "Intersects":
            result.value = bool(result.intersection)
        elif op == "Contains":
            # 业务语义（与 backends.plain.plain_contains 对齐）：
            #   Contains(outer, inner) ≜ inner ⊆ outer
            # PSI 给的是 outer∩inner（left=outer, right=inner），故判据是
            #   |outer∩inner| == |inner|   （等价于 inner ⊆ (outer∩inner)）
            # 旧实现是在交集之上做本地集合运算——那是明文比较，且被比较的 inner
            # 在真实两方部署里属于参与方 1，这一步无法装配成两方协议。
            # 现在只把两个**基数**交给 MPC 电路（见 subset_mpc）：
            # 任一方都不必把自己的集合交给对方。
            # 早期实现还写成 `outer <= intersection`，方向与集合都取反了：
            # 它实际在问 "inner ⊇ outer"，会把 True 判成 False。
            inner = set(right_codes)
            subset = _resolve_subset(
                intersection=result.intersection,
                inner=inner,
                subset_via=subset_via,
                protocol=mpc_protocol,
                field=mpc_field,
                world_size=mpc_world_size,
                report=mpc_report,
            )
            result.subset = subset
            result.value = subset.value
            result.reveals = _reveals_with_subset(op, subset)
            result.notes = result.notes + _subset_notes(subset)
            if not inner:
                result.notes = result.notes + (
                    "inner 为空集：∅ ⊆ 任意集合，结果为 True",
                )
        else:  # CellSetIntersect
            result.value = result.intersection

        result.status = "ok"
        result.notes = result.notes + (
            f"执行路径：spu.libspu.link.create_mem + spu.psi.psi_execute（{protocol_name}）",
            "PSI 为文件接口，输入输出经临时目录中转（已自动清理）",
        )
        if result.reference is not None:
            result.agreement = bool(
                _norm(result.value) == _norm(result.reference)
            )
            # 带噪协议（差分隐私 PSI）"与明文不一致"是协议语义，不是执行缺陷：
            # 把它升级成 error 会让编译器把一次正常执行报成失败，而且是**随机**的
            # （实测 40 次里 1 次不一致）。这里如实登记 agreement，但不改 status。
            # 是否允许这种协议参与编译，由 planner/编译器一层决定并显式披露。
            if not result.agreement and protocol_is_exact(protocol_name):
                result.status = "error"
                result.error = (
                    f"PSI 结果与明文不一致：PSI={_jsonable(result.value)}，"
                    f"明文={_jsonable(result.reference)}"
                )
        return result
    finally:
        if own_dir:
            shutil.rmtree(workdir, ignore_errors=True)


def _set_native_log(quiet: bool) -> None:
    """开关官方原生库的日志输出。

    `libspu.logging.setup_logging` 是 C++ 侧接口，签名不可内省；
    这里只按实测可用的字段设置，任何失败都静默忽略——
    日志开关不该影响隐私计算的正确性。
    """

    try:
        import spu.libspu as libspu

        logging = getattr(libspu, "logging", None)
        if logging is None or not hasattr(logging, "setup_logging"):
            return
        options = logging.LogOptions()
        options.enable_console_logger = not quiet
        if quiet:
            options.log_level = logging.LogLevel.ERROR
        # `system_log_path` 默认是相对路径 'spu.log'：无论静默与否，原生库都会在
        # **当前工作目录**落一个文件。编译器不该往用户的项目目录里丢日志，
        # 故一律改指 /dev/null；需要日志时走 console（quiet=False）。
        options.system_log_path = os.devnull
        logging.setup_logging(options)
    except Exception:
        return


def _norm(value: Any) -> Any:
    """把布尔/元组/列表归一化，便于比较。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted(int(v) for v in value))
    if value is None:
        return None
    return value


def run_psi_operation(
    op: str,
    args: Mapping[str, Sequence[int]],
    **kwargs: Any,
) -> PsiRunResult:
    """按算子名分派：从命名参数里取该算子需要的两方集合。"""

    if op in ("Intersects", "CellSetIntersect"):
        left = args.get("left") or args.get("route") or args.get("route_A") or ()
        right = args.get("right") or args.get("no_fly_zone") or args.get("NoFlyZone_B") or ()
        return run_psi_intersection(left, right, op=op, **kwargs)
    if op == "Contains":
        left = args.get("outer") or next(iter(args.values()), ())
        right = args.get("inner") or ()
        return run_psi_intersection(left, right, op=op, **kwargs)
    raise ValueError(f"PSI 后端不处理算子 {op!r}；已支持 {tuple(PSI_OP_LEAKS)}")
