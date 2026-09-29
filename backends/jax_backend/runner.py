"""JAX 后端：追踪验证与执行。

两个职责：
    1. `check_traceable(fn, arg_specs)` —— 编译前验证 jax.jit 可追踪（不依赖 SPU）；
    2. `run_jax_jit(fn, inputs)`      —— 用 jax.jit 实际执行。

"可追踪"是硬门槛：不可追踪的 Python 控制流必须在编译前被拦下，
而不是等 SPU 编译报错。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


@dataclass
class TraceCheck:
    """jax.jit 可追踪性检查结果。"""

    traceable: bool
    error: str | None = None
    error_type: str | None = None
    strategy: str = ""
    shape_info: Mapping[str, Any] = field(default_factory=dict)
    hlo_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "traceable": self.traceable,
            "error": self.error,
            "error_type": self.error_type,
            "strategy": self.strategy,
            "shape_info": dict(self.shape_info),
            "hlo_bytes": self.hlo_bytes,
        }


def check_traceable(
    fn: Callable[..., Any],
    example_inputs: Sequence[Any],
    *,
    static_argnums: Sequence[int] = (),
    try_lower: bool = True,
) -> TraceCheck:
    """验证函数在给定输入形状下可被 jax.jit 追踪。

    Args:
        fn: 待验证函数（应只用 jax.numpy）。
        example_inputs: 与目标输入同形状/同 dtype 的样例张量。
        static_argnums: 静态参数位置。
        try_lower: 是否进一步验证可降级为 HLO（SPU 编译桥的第一步）。
    """

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        return TraceCheck(
            traceable=False,
            error=f"jax 未安装：{exc}",
            error_type="ImportError",
            strategy="import",
        )

    # 统一成 jax 数组，避免 numpy/python 标量导致的非追踪路径
    try:
        args = [_to_jax_array(x, jnp) for x in example_inputs]
    except Exception as exc:
        return TraceCheck(
            traceable=False,
            error=f"样例输入无法转为 jax 数组：{exc}",
            error_type=type(exc).__name__,
            strategy="convert-inputs",
        )

    shape_info = {
        f"arg{i}": {"shape": tuple(getattr(a, "shape", ())), "dtype": str(getattr(a, "dtype", type(a).__name__))}
        for i, a in enumerate(args)
    }

    # 策略 1：jit 追踪（覆盖动态控制流问题）
    try:
        jitted = jax.jit(fn, static_argnums=tuple(static_argnums))
        traced = jitted.trace(*args)
    except Exception as exc:
        return TraceCheck(
            traceable=False,
            error=str(exc),
            error_type=type(exc).__name__,
            strategy="jit-trace",
            shape_info=shape_info,
        )

    if not try_lower:
        return TraceCheck(traceable=True, strategy="jit-trace", shape_info=shape_info)

    # 策略 2：降级为 HLO（与 SPU frontend 用同一条路径）
    try:
        from .spu_bridge import lower_to_hlo

        hlo, error = lower_to_hlo(fn, args, static_argnums=static_argnums)
        if error:
            return TraceCheck(
                traceable=True,
                error=error,
                error_type="HloLoweringError",
                strategy="jit-trace (lowering failed)",
                shape_info=shape_info,
            )
        return TraceCheck(
            traceable=True,
            strategy="jit-trace + hlo-lower",
            shape_info=shape_info,
            hlo_bytes=len(hlo) if hlo else None,
        )
    except Exception as exc:
        return TraceCheck(
            traceable=True,
            error=f"降级检查异常：{exc}",
            error_type=type(exc).__name__,
            strategy="jit-trace",
            shape_info=shape_info,
        )


def _to_jax_array(value: Any, jnp: Any) -> Any:
    if hasattr(value, "dtype") and type(value).__module__.startswith("jax"):
        return value
    return jnp.asarray(value)


def run_jax_jit(
    fn: Callable[..., Any],
    inputs: Sequence[Any],
    *,
    static_argnums: Sequence[int] = (),
) -> Any:
    """用 jax.jit 执行函数并取回具体值。"""

    import jax
    import jax.numpy as jnp

    jitted = jax.jit(fn, static_argnums=tuple(static_argnums))
    args = [jnp.asarray(x) for x in inputs]
    return jitted(*args)


# --------------------------------------------------------------------------
# 生成源码 → 可调用对象
# --------------------------------------------------------------------------


def load_generated_function(source: str, fn_name: str) -> Callable[..., Any]:
    """把生成的源码文本编译为可调用对象。

    刻意使用受限命名空间执行：生成代码只允许依赖 jax.numpy。
    """

    namespace: dict[str, Any] = {}
    exec(compile(source, f"<generated:{fn_name}>", "exec"), namespace)  # noqa: S102
    fn = namespace.get(fn_name)
    if fn is None:
        raise NameError(f"生成的源码中找不到函数 {fn_name}")
    return fn


def check_generated_source(source: str, fn_name: str, example_inputs: Sequence[Any]) -> tuple[Callable[..., Any] | None, TraceCheck]:
    """对生成的源码做"加载 + 可追踪"两步检查。"""

    try:
        fn = load_generated_function(source, fn_name)
    except Exception as exc:
        return None, TraceCheck(
            traceable=False,
            error=f"生成代码加载失败：{exc}",
            error_type=type(exc).__name__,
            strategy="load-source",
        )
    return fn, check_traceable(fn, example_inputs)