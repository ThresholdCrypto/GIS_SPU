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


def jax_weighted_sum(values: Any, weights: Any, scale: Any = 1) -> Any:
    """定点加权和。"""

    jnp = _require_jax()
    acc = jnp.sum(values * weights)
    return acc // scale


def jax_temporal_overlap(
    left_toff: Any, left_lt: Any, right_toff: Any, right_lt: Any
) -> Any:
    """段式区间重叠（广播比较，无 Python 分支）。"""

    jnp = _require_jax()
    one = jnp.asarray(1, jnp.int32)
    left_end = left_toff + jnp.left_shift(one, left_lt)
    right_end = right_toff + jnp.left_shift(one, right_lt)
    overlapping = (left_toff[:, None] < right_end[None, :]) & (
        right_toff[None, :] < left_end[:, None]
    )
    return jnp.any(overlapping)


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