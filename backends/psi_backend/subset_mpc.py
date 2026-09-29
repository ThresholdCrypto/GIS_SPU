# -*- coding: utf-8 -*-
"""`Contains` 的密态子集比较：把"子集判定"从明文比较换成 MPC 基数等值。

问题
----
PSI 只回答"交集是什么"。`Contains` 在交集之上还差一步判定：

    Contains(outer, inner) ≜ inner ⊆ outer  ⟺  |outer ∩ inner| == |inner|

（右式成立是因为 `outer ∩ inner ⊆ inner` 恒成立，故两者相等即互为子集。）

本模块把这一步做成密态：

    参与方 0 私有输入  k = |outer ∩ inner|   （它本来就从 PSI 拿到交集，不新增泄露）
    参与方 1 私有输入  n = |inner|
    MPC 电路输出一个比特  k == n

与旧实现的区别（这是本模块存在的理由）
------------------------------------
旧实现在交集之上做本地集合运算 `inner <= set(intersection)` —— 那是**明文比较**，
而且被比较的 `inner` 在真实两方部署里属于参与方 1，因此这一步**无法装配成两方协议**：
要么把 `inner` 交给参与方 0（等于把整个 inner 交出去），要么把交集交给参与方 1。
换成 MPC 之后，两侧各只提供**一个整数**，输出只有布尔，两侧都拿不到对方的基数。

没有改变的事（不淡化）
----------------------
PSI 的标准语义没变：接收方**仍然**拿到交集本体。本模块只消除"第二次明文比较"，
不消除交集本体的暴露。两者的泄漏面各自登记，不合并。

API 依据（spu 0.9.5 实测，非猜测）
--------------------------------
    sim = spu.utils.simulation.Simulator.simple(wsize, ProtocolKind, FieldType)
    spu_fn = spu.utils.simulation.sim_jax(sim, jax_fn)
    out = spu_fn(*inputs)

`Simulator.__call__` 对每个输入做 `io.make_shares(x, Visibility.VIS_SECRET)`，
再把第 rank 份交给第 rank 个参与方——**没有任何一方看到明文输入**。
这正是本模块需要的语义：两侧各自持有一个私有基数。

电路只用到整数等值比较，实测在 SEMI2K / ABY3 / CHEETAH × FM32 / FM64 上均可执行
（原语清单登记在 `backends.spu_backend.capability.OP_HLO_PRIMITIVES["Contains"]`，
取自真实编译产物，不是抄来的）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..spu_backend import SpuRunResult, run_spu_simulation

#: 子集判定的三种模式，措辞必须能被业务方分辨
SUBSET_MODE_MPC = "mpc"
SUBSET_MODE_PLAINTEXT = "plaintext"
SUBSET_MODE_PLAINTEXT_FALLBACK = "plaintext-fallback"

SUBSET_MODES: tuple[str, ...] = (
    SUBSET_MODE_MPC,
    SUBSET_MODE_PLAINTEXT,
    SUBSET_MODE_PLAINTEXT_FALLBACK,
)

#: 独立调用（未经编译器）时的默认 MPC 设置；编译器会把自身的协议/环宽传进来。
SUBSET_DEFAULT_PROTOCOL = "ABY3"
SUBSET_DEFAULT_FIELD = 64

#: 泄漏面随模式变化，故不能只写一张静态表
SUBSET_MODE_DISCLOSURE: Mapping[str, str] = {
    SUBSET_MODE_MPC: (
        "子集判定由 MPC 基数等值完成：k=|outer∩inner| 与 n=|inner| 各为一方私有输入，"
        "输出只有一个布尔；明文比较已移除"
    ),
    SUBSET_MODE_PLAINTEXT: (
        "子集判定**按调用方要求**退回明文比较：inner 与交集在同一进程内做本地集合运算，"
        "两方部署下无法装配（需把 inner 交给接收方）"
    ),
    SUBSET_MODE_PLAINTEXT_FALLBACK: (
        "子集判定**回退为明文比较**：MPC 电路未能执行（原因见 notes），"
        "本次的密态属性不完整，不得当作密态子集比较的验证依据"
    ),
}

def subset_disclosure(mode: str) -> str:
    """模式 → 一句可读的泄漏面说明。未知模式不静默兜底。"""

    try:
        return SUBSET_MODE_DISCLOSURE[mode]
    except KeyError as exc:
        raise ValueError(
            f"未知的子集判定模式 {mode!r}；可选 {SUBSET_MODES}"
        ) from exc


def resolve_subset_mode(value: str) -> str:
    """归一化子集判定模式；非法值带可用清单抛 ValueError（fail-fast）。

    只接受调用方**可以显式选择**的两种模式。`plaintext-fallback` 是运行时
    自动产生的退路，不允许被显式指定——否则"我选了明文"和"我本想要密态
    但退路了"会在日志里长得一样。
    """

    mode = str(value).strip().lower()
    if mode not in (SUBSET_MODE_MPC, SUBSET_MODE_PLAINTEXT):
        raise ValueError(
            f"未知的子集判定模式 {value!r}；"
            f"可选 {SUBSET_MODE_MPC!r} 或 {SUBSET_MODE_PLAINTEXT!r}"
        )
    return mode


def subset_equality_jax(intersection_card: Any, inner_card: Any) -> Any:
    """MPC 电路：两个私有基数的等值比较，输出 0/1（int32）。

    只用 jax.numpy 的逐元素原语；没有 Python 级 if/for/while，
    因此 `jax.jit` 可追踪，SPU 前端也能编译。
    """

    try:
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "密态子集比较需要 jax；请先安装（参考 requirements-jax.txt）"
        ) from exc

    return jnp.equal(intersection_card, inner_card).astype(jnp.int32)


@dataclass
class SubsetComparison:
    """一次子集判定（密态或明文）的结果。"""

    mode: str
    status: str  # ok / forced / unavailable / error
    value: bool | None = None
    protocol: str = ""
    field: str = ""
    agreement: bool | None = None
    max_abs_error: float | None = None
    pphlo_bytes: int | None = None
    note: str = ""
    run: SpuRunResult | None = None

    @property
    def is_mpc(self) -> bool:
        return self.mode == SUBSET_MODE_MPC

    @property
    def disclosure(self) -> str:
        return subset_disclosure(self.mode)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "status": self.status,
            "value": self.value,
            "protocol": self.protocol,
            "field": self.field,
            "agreement": self.agreement,
            "max_abs_error": self.max_abs_error,
            "pphlo_bytes": self.pphlo_bytes,
            "note": self.note,
            "disclosure": self.disclosure,
        }

    def describe(self) -> str:
        head = f"subset comparison: {self.status.upper()}  (mode={self.mode}"
        if self.protocol:
            head += f", protocol={self.protocol}, field={self.field}"
        head += ")"
        lines = [head, f"  result      : {self.value}"]
        if self.agreement is not None:
            lines.append(
                f"  plaintext   : {self.value if self.agreement else '不一致'}   "
                f"agreement={self.agreement}"
            )
        lines.append(f"  disclosure  : {self.disclosure}")
        if self.note:
            lines.append(f"  note        : {self.note}")
        return "\n".join(lines)


def mpc_subset_comparison(
    intersection_card: int,
    inner_card: int,
    *,
    protocol: str = SUBSET_DEFAULT_PROTOCOL,
    field: str | int = SUBSET_DEFAULT_FIELD,
    world_size: int | None = None,
    report: Any = None,
    tolerance: float = 0.0,
) -> SubsetComparison:
    """在 SPU 上密态判定 `|outer∩inner| == |inner|`。

    两个基数各自作为一方的私有输入；返回值只含一个布尔。
    绝不返回推测值：电路没跑成功时 `value` 留空，由调用方决定是否退路。
    """

    def reference_fn(left: Any, right: Any) -> Any:
        return left == right

    run = run_spu_simulation(
        subset_equality_jax,
        [int(intersection_card), int(inner_card)],
        protocol=protocol,
        field=field,
        world_size=world_size,
        reference_fn=reference_fn,
        tolerance=tolerance,
        report=report,
    )

    comparison = SubsetComparison(
        mode=SUBSET_MODE_MPC,
        status=run.status,
        protocol=run.protocol,
        field=run.field,
        max_abs_error=run.max_abs_error,
        pphlo_bytes=run.pphlo_bytes,
        run=run,
    )

    if run.ok:
        comparison.value = bool(run.outputs)
        comparison.agreement = bool(run.within_tolerance)
        comparison.note = (
            f"执行路径：{run.protocol}/{run.field}，"
            "两方各持一个私有基数（k=|outer∩inner| 与 n=|inner|），输出单个布尔"
        )
    else:
        comparison.note = run.error or "；".join(run.blockers) or "MPC 电路未执行"

    return comparison


def plaintext_subset_comparison(
    intersection: Any,
    inner: Any,
    *,
    mode: str = SUBSET_MODE_PLAINTEXT,
    note: str = "",
) -> SubsetComparison:
    """明文子集判定：`inner ⊆ (outer ∩ inner)`。

    刻意保留这条路径的两个用途：
      1. 调用方显式要求时（`subset_via="plaintext"`）；
      2. MPC 不可执行时的**已披露**退路。
    两种情形都必须能被状态词区分开，故 mode 由调用方指定。
    """

    if mode not in (SUBSET_MODE_PLAINTEXT, SUBSET_MODE_PLAINTEXT_FALLBACK):
        raise ValueError(
            f"明文子集判定的 mode 只能是 {SUBSET_MODE_PLAINTEXT} / "
            f"{SUBSET_MODE_PLAINTEXT_FALLBACK}，实得 {mode!r}"
        )

    return SubsetComparison(
        mode=mode,
        status="forced" if mode == SUBSET_MODE_PLAINTEXT else "fallback",
        value=bool(set(int(v) for v in inner) <= set(int(v) for v in intersection)),
        note=note,
    )
