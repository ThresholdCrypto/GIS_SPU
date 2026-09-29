"""路径与依赖辅助：让 tests/ 能以 GIS_SPU 为根导入各模块。"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")


def example(name: str) -> str:
    """取 examples/ 下某个示例的绝对路径。"""

    return os.path.join(EXAMPLES_DIR, name)


def has_jax() -> bool:
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


def has_spu() -> bool:
    from backends.spu_backend import check_capabilities

    return check_capabilities().runnable


def has_psi() -> bool:
    """当前环境是否具备真实 PSI 执行能力（spu 已装 + libpsi 可载 + Linux）。"""

    from backends.psi_backend import check_psi_capabilities

    return check_psi_capabilities().runnable
