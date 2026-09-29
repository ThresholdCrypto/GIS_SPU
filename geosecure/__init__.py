"""geosecure：面向地理信息行业的低门槛隐私计算编译器。

    Python Source → AST → Geo-IR → Privacy Plan → JAX → SPU simulation
"""

from .compiler import (
    DEFAULT_EXAMPLE_INPUTS,
    STAGE_TITLES,
    CompileResult,
    Compiler,
    StageResult,
    compile_file,
    compile_source,
    operator_status_table,
)

__all__ = [
    "CompileResult",
    "Compiler",
    "DEFAULT_EXAMPLE_INPUTS",
    "STAGE_TITLES",
    "StageResult",
    "compile_file",
    "compile_source",
    "operator_status_table",
]
__version__ = "0.1.0"