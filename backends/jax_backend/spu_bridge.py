"""SPU 编译桥的第一步：把 jax 函数降级为序列化 HLO。

为什么单列一个模块：
    这是 SPU 的 `frontend.compile(Kind.JAX, ...)` 在真正编译前必走的一步——
    注册 interpreter 后端，再 `jax.jit(...).trace(...).lower(('interpreter',))`，
    最后取 `compiler_ir('hlo').as_serialized_hlo_module_proto()`。

    它**不需要 libspu**，因此可以在没有 SPU 的环境里独立验证：
    "生成的 JAX 代码能否被 SPU 的编译前端吃下"。
    这比等到 SPU 装好才发现代码形态不对要早得多。

实现严格对齐 SPU 0.9.5 的 `spu/utils/frontend.py::_jax_compilation`。
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

__all__ = [
    "ensure_interpreter_backend",
    "extract_hlo_ops",
    "hlo_summary",
    "lower_to_hlo",
    "lower_to_hlo_text",
]

_INTERPRETER_REGISTERED = False


def ensure_interpreter_backend() -> str | None:
    """按 SPU 的做法注册 interpreter 后端。

    返回 xla_extension_version 的字符串形式（若不存在则返回 None）。
    SPU 依赖该值判断走新旧 API 分支。
    """

    global _INTERPRETER_REGISTERED

    version: str | None = None
    try:
        from jax._src.lib import xla_extension_version

        version = str(xla_extension_version)
    except Exception:
        # jax >= 0.5 移除了该符号；SPU 0.9.5 会在此处 ImportError
        version = None

    if _INTERPRETER_REGISTERED:
        return version

    try:
        from jax._src.xla_bridge import _backend_lock, _backends, register_backend_factory

        with _backend_lock:
            has_interpreter = "interpreter" in _backends
        if not has_interpreter:
            from jax.interpreters.xla import Backend as xla_backend

            register_backend_factory("interpreter", xla_backend, priority=-100)
        _INTERPRETER_REGISTERED = True
    except Exception:
        pass

    return version


def lower_to_hlo(
    fn: Callable[..., Any],
    args: Sequence[Any],
    *,
    static_argnums: Sequence[int] = (),
) -> tuple[bytes | None, str | None]:
    """把函数降级为序列化 HLO。

    Returns:
        (hlo_bytes, error)。成功时 error 为 None。
    """

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        return None, f"jax 未安装：{exc}"

    ensure_interpreter_backend()

    jax_args = [jnp.asarray(a) for a in args]
    try:
        lowered = (
            jax.jit(
                fn,
                static_argnums=tuple(static_argnums),
                keep_unused=True,
            )
            .trace(*jax_args)
            .lower(lowering_platforms=("interpreter",))
        )
        hlo = lowered.compiler_ir("hlo").as_serialized_hlo_module_proto()
        return hlo, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def lower_to_hlo_text(
    fn: Callable[..., Any],
    args: Sequence[Any],
    *,
    static_argnums: Sequence[int] = (),
) -> tuple[str | None, str | None]:
    """降级为**文本** HLO。

    比序列化 protobuf 更适合能力核查：能直接读出实际发射的算子，
    从而核对"生成代码是否只用了 SPU 已适配的原语"。
    """

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        return None, f"jax 未安装：{exc}"

    ensure_interpreter_backend()
    jax_args = [jnp.asarray(a) for a in args]
    try:
        lowered = (
            jax.jit(fn, static_argnums=tuple(static_argnums), keep_unused=True)
            .trace(*jax_args)
            .lower(lowering_platforms=("interpreter",))
        )
        return lowered.as_text(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def extract_hlo_ops(hlo_text: str | None) -> tuple[str, ...]:
    """从文本 HLO 中提取实际发射的 StableHLO 算子名（去重有序）。"""

    if not hlo_text:
        return ()
    import re

    found = re.findall(r"stablehlo\.([a-z_]+)", hlo_text)
    seen: list[str] = []
    for name in found:
        if name not in seen:
            seen.append(name)
    return tuple(seen)


def hlo_summary(hlo: bytes | None) -> dict[str, Any]:
    """序列化 HLO 的字节摘要（拿不到算子名时用；算子名请用 lower_to_hlo_text）。"""

    if not hlo:
        return {"bytes": 0, "ops": []}
    return {"bytes": len(hlo), "ops": []}
