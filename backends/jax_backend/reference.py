"""JAX 参考实现：与生成代码语义一致的 jax.numpy 实现。

python 实现与 codegen 输出必须逐一对应；本模块是"可执行的规范"，
测试用它作为生成代码的期望值来源。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "JAX_IMPLEMENTATIONS",
    "jax_contains",
    "jax_distance_le",
    "jax_intersects",
    "jax_temporal_overlap",
    "jax_temporal_overlap_sweep",
    "jax_temporal_overlap_pairwise",
    "jax_weighted_sum",
    "run_jax",
]


def _require_jax() -> Any:
    try:
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "JAX 后端需要 jax；请先安装（参考 requirements-jax.txt）"
        ) from exc
    return jnp


# --------------------------------------------------------------------------
# 逐算子实现（与 codegen 生成代码逐行对应）
# --------------------------------------------------------------------------


def jax_distance_le(left: Any, right: Any, threshold: Any) -> Any:
    """距离平方与阈值平方比较。"""

    jnp = _require_jax()
    delta = left - right
    dist_sq = jnp.sum(jnp.square(delta))
    return dist_sq <= jnp.square(threshold)


def jax_weighted_sum(values: Any, weights: Any, scale: int = 1) -> Any:
    """定点加权和（与 codegen 生成代码逐行对应）。

    `scale` 是**静态**参数（Python int），因此这里的分支在追踪期就消解，
    `scale == 1` 时不会留下除法电路——见 `generate_weighted_sum` 的说明。
    """

    jnp = _require_jax()
    acc = jnp.sum(values * weights)
    scale = int(scale)
    if scale == 1:
        return acc
    if scale & (scale - 1) == 0:
        return jnp.right_shift(acc, scale.bit_length() - 1)
    return acc // scale


def jax_temporal_overlap(
    left_toff: Any, left_lt: Any, right_toff: Any, right_lt: Any
) -> Any:
    """段式区间重叠：广播成 [N, M] 逐对比较（O(N·M)）。

    这是**语义规范**：逐对直接判定，显然正确，因此拿来当生成代码的期望值来源。
    代价 O(N·M)——K=256 展开 65 536 个比较、K=4096 是 1 670 万元素。
    大 K 请用 `jax_temporal_overlap_sweep`（等价，O((N+M)·log(N+M))）。
    """

    jnp = _require_jax()
    one = jnp.asarray(1, jnp.int32)
    left_end = left_toff + jnp.left_shift(one, left_lt)
    right_end = right_toff + jnp.left_shift(one, right_lt)
    overlapping = (left_toff[:, None] < right_end[None, :]) & (
        right_toff[None, :] < left_end[:, None]
    )
    return jnp.any(overlapping)


def jax_temporal_overlap_sweep(
    left_toff: Any, left_lt: Any, right_toff: Any, right_lt: Any
) -> Any:
    """段式区间重叠：**事件排序归并 + 前缀扫描**（与 codegen 的 sweep 逐行对应）。

    O((N+M)·log(N+M))，不物化 [N, M] 矩阵。与 `jax_temporal_overlap` 等价，
    但成本常数更大（MPC 里排序本身很贵），故只在规模足够大时才划算：
    实测（ABY3/FM64）K=128 时两两比较 3.81 MB < 扫描 5.01 MB，
    K=256 时反过来 12.59 MB > 9.93 MB——交叉点在 128–256 之间，
    见 docs/MPC_BENCHMARK_PROTOCOL.md §4.1。

    步骤：
        1. 每个段式节点 (Toff, Lt) 展开成半开区间 [Toff, Toff + 2^Lt)，
           拆成"起点/终点"两个事件；
        2. 事件打包成一个整数一起排序：`(pos << 2) | (is_start << 1) | side`，
           起点/终点占**高位** ⇒ 同一位置上所有终点排在所有起点之前
           （半开区间：`[0,2)` 与 `[2,4)` 不重叠）；
        3. 按序做前缀和 ⇒ 两侧各自的"当前活跃区间数"；
        4. **两侧活跃数同时 > 0** 即存在一对区间重叠。

    等价性理由：两侧区间并集相交 ⟺ 存在一对区间相交；扫描在每个事件位置上
    都给出两侧的活跃集合，故不漏判也不误判（632 组输入与逐对版/明文逐位一致）。
    """

    jnp = _require_jax()
    one = jnp.asarray(1, jnp.int32)
    zero = jnp.asarray(0, jnp.int32)
    left_end = left_toff + jnp.left_shift(one, left_lt)
    right_end = right_toff + jnp.left_shift(one, right_lt)
    left_events = jnp.concatenate(
        [
            jnp.left_shift(left_toff, 2) + 3,
            jnp.left_shift(left_end, 2) + 1,
        ]
    )
    right_events = jnp.concatenate(
        [
            jnp.left_shift(right_toff, 2) + 2,
            jnp.left_shift(right_end, 2),
        ]
    )
    keys = jnp.sort(jnp.concatenate([left_events, right_events]))
    is_start = jnp.bitwise_and(jnp.right_shift(keys, 1), one)
    side = jnp.bitwise_and(keys, one)
    delta = jnp.where(is_start == 1, one, -one)
    active_left = jnp.cumsum(jnp.where(side == 1, delta, zero))
    active_right = jnp.cumsum(jnp.where(side == 1, zero, delta))
    return jnp.any((active_left > 0) & (active_right > 0))


def jax_intersects(left_codes: Any, right_codes: Any) -> Any:
    """集合交存在性。

    这不是逐元素算子：jax.numpy 没有集合原语。此处用"广播比较 + 任意归约"
    实现 O(N·M) 的等价判定，仅用于**语义对照**，
    真实密态路径应走 PSI（见 planner 注册表）。
    """

    jnp = _require_jax()
    equality = left_codes[:, None] == right_codes[None, :]
    return jnp.any(equality)


def jax_contains(outer: Any, inner: Any) -> Any:
    """子集关系：inner 的每个元素都出现在 outer 中。"""

    jnp = _require_jax()
    present = inner[:, None] == outer[None, :]
    return jnp.all(jnp.any(present, axis=1))


JAX_IMPLEMENTATIONS: Mapping[str, Callable[..., Any]] = {
    "DistanceLE": jax_distance_le,
    "WeightedSum": jax_weighted_sum,
    "TemporalOverlap": jax_temporal_overlap,
    "Intersects": jax_intersects,
    "Contains": jax_contains,
}


def run_jax(op: str, *args: Any, **kwargs: Any) -> Any:
    impl = JAX_IMPLEMENTATIONS.get(op)
    if impl is None:
        raise NotImplementedError(
            f"JAX 后端没有 {op} 的实现；可用：{sorted(JAX_IMPLEMENTATIONS)}"
        )
    return impl(*args, **kwargs)
