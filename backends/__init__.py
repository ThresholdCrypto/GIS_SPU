"""backends：Plain（明文基准）/ JAX（代码生成与执行）/ SPU（能力探测与 MPC 模拟）/ PSI（两方求交）。

四条后端的职责边界：
    plain   —— 语义基准，三份实现必须向它对齐
    jax     —— 代码生成 + jax.jit 追踪验证
    spu     —— jax.jit → SPU 虚拟机（MPC）模拟
    psi     —— 官方 spu.psi.psi_execute（文件 CSV 接口）两方求交
"""

from . import jax_backend, plain, psi_backend, spu_backend

__all__ = ["plain", "jax_backend", "spu_backend", "psi_backend"]
