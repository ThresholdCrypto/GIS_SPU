"""JAX 后端：代码生成 + 可追踪性验证 + 执行。"""

from .codegen import (
    FLOAT_TOLERANCE,
    GENERATED_HEADER,
    GENERATORS,
    TOLERANCES,
    GeneratedFunction,
    JaxGenerationResult,
    generate_distance_le,
    generate_for_plan,
    generate_temporal_overlap,
    generate_weighted_sum,
    render_module,
    static_check_source,
)
from .reference import JAX_IMPLEMENTATIONS, run_jax
from .runner import (
    TraceCheck,
    check_generated_source,
    check_traceable,
    load_generated_function,
    run_jax_jit,
)
from .spu_bridge import (
    ensure_interpreter_backend,
    extract_hlo_ops,
    hlo_summary,
    lower_to_hlo,
    lower_to_hlo_text,
)

__all__ = [
    "FLOAT_TOLERANCE",
    "GENERATED_HEADER",
    "GENERATORS",
    "GeneratedFunction",
    "JAX_IMPLEMENTATIONS",
    "JaxGenerationResult",
    "TOLERANCES",
    "TraceCheck",
    "check_generated_source",
    "check_traceable",
    "ensure_interpreter_backend",
    "extract_hlo_ops",
    "generate_distance_le",
    "generate_for_plan",
    "generate_temporal_overlap",
    "generate_weighted_sum",
    "hlo_summary",
    "load_generated_function",
    "lower_to_hlo",
    "lower_to_hlo_text",
    "render_module",
    "run_jax",
    "run_jax_jit",
    "static_check_source",
]