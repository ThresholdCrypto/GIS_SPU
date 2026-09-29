# -*- coding: utf-8 -*-
"""PSI 后端：真实多方隐私求交（调用蚂蚁隐语 SPU 官方 PSI 实现）。

设计边界（刻意的）
------------------
- **不做通用 PSI 引擎**。只覆盖本项目注册表里 PSI 族的三个算子：
  `Intersects` / `Contains` / `CellSetIntersect`。
- **不重新实现 PSI 协议**，也不修改 SPU 源码；只封装官方入口
  `spu.psi.psi_execute(config, lctx)`。
- **不隐藏泄漏面**。PSI 的语义是"接收方得到交集本体"，比布尔结果**泄露更多**。
  每个算子都会在 `reveals` 字段如实标注它实际暴露了什么。

API 依据（spu 0.9.5 实测，非猜测）
--------------------------------
    import spu.libspu as libspu
    from spu import psi

    desc = libspu.link.Desc()
    for r in range(2):
        desc.add_party(f"id_{r}", f"thread_{r}")
    lctx = libspu.link.create_mem(desc, rank)          # 每个参与方一个线程

    cfg = psi.PsiExecuteConfig(
        protocol_conf=psi.PsiProtocolConfig(
            protocol=psi.PsiProtocol.PROTOCOL_ECDH,
            receiver_rank=0,
            ecdh_params=psi.EcdhParams(curve=psi.EllipticCurveType.CURVE_SM2),
        ),
        input_params=psi.InputParams(
            type=psi.SourceType.SOURCE_TYPE_FILE_CSV,
            path="<csv>", selected_keys=["key"], keys_unique=True,
        ),
        output_params=psi.OutputParams(
            type=psi.SourceType.SOURCE_TYPE_FILE_CSV, path="<out.csv>",
        ),
        join_conf=psi.ResultJoinConfig(
            type=psi.ResultJoinType.JOIN_TYPE_INNER_JOIN, left_side_rank=0,
        ),
    )
    report = psi.psi_execute(cfg, lctx)
    report.original_count / intersection_count
    report.original_unique_count / intersection_unique_count

要点（踩过的坑，勿重复）
------------------------
1. PSI 目前是 **文件（CSV）接口**，没有内存张量接口。故需临时工作目录。
2. `PROTOCOL_ECDH` **必须显式给 curve**，否则
   `RuntimeError: Curve type is not specified.`
   （`EcdhParams` 默认值是 `CURVE_INVALID_TYPE`，不是可用默认值。）
3. `KKRT` / `RR22` 不需要 curve。
4. 只有 `receiver_rank`（默认 0）会拿到交集并报告计数；另一方
   `intersection_count` 为 -1（未计算），这是协议设计，不是错误。
5. **不需要** `libgomp1` 之外的额外系统库；PSI 与 SPU 共用 libpsi.so。
"""

from .capability import (
    PSI_CURVE_RELATION,
    PSI_CURVES,
    PSI_DEFAULT_CURVE,
    PSI_DEFAULT_PROTOCOL,
    PSI_OPS,
    PSI_PROTOCOL_WORLD_SIZE,
    PSI_PROTOCOLS,
    PSI_PROTOCOLS_CURVE_REQUIRED,
    PSI_PROTOCOLS_WITH_NOISE,
    PSI_PROTOCOLS_WITHOUT_CURVE,
    PSI_RUNTIME_WORLD_SIZE,
    PsiCapabilityReport,
    check_psi_capabilities,
    normalize_curve,
    normalize_psi_protocol,
    protocol_is_exact,
    protocol_needs_curve,
    protocol_world_size,
    protocols_runnable_here,
    psi_curve_relation,
    resolve_psi_protocol,
    runnable_protocols_hint,
)
from .runtime import (
    PSI_OP_LEAKS,
    PsiRunResult,
    run_psi_intersection,
    run_psi_operation,
)
from .subset_mpc import (
    SUBSET_DEFAULT_FIELD,
    SUBSET_DEFAULT_PROTOCOL,
    SUBSET_MODE_DISCLOSURE,
    SUBSET_MODE_MPC,
    SUBSET_MODE_PLAINTEXT,
    SUBSET_MODE_PLAINTEXT_FALLBACK,
    SUBSET_MODES,
    SubsetComparison,
    mpc_subset_comparison,
    plaintext_subset_comparison,
    resolve_subset_mode,
    subset_disclosure,
    subset_equality_jax,
)

__all__ = [
    "PSI_CURVE_RELATION",
    "PSI_CURVES",
    "PSI_DEFAULT_CURVE",
    "PSI_DEFAULT_PROTOCOL",
    "PSI_OPS",
    "PSI_OP_LEAKS",
    "PSI_PROTOCOL_WORLD_SIZE",
    "PSI_PROTOCOLS",
    "PSI_PROTOCOLS_CURVE_REQUIRED",
    "PSI_PROTOCOLS_WITH_NOISE",
    "PSI_PROTOCOLS_WITHOUT_CURVE",
    "PSI_RUNTIME_WORLD_SIZE",
    "PsiCapabilityReport",
    "PsiRunResult",
    "SUBSET_DEFAULT_FIELD",
    "SUBSET_DEFAULT_PROTOCOL",
    "SUBSET_MODES",
    "SUBSET_MODE_DISCLOSURE",
    "SUBSET_MODE_MPC",
    "SUBSET_MODE_PLAINTEXT",
    "SUBSET_MODE_PLAINTEXT_FALLBACK",
    "SubsetComparison",
    "check_psi_capabilities",
    "mpc_subset_comparison",
    "normalize_curve",
    "normalize_psi_protocol",
    "plaintext_subset_comparison",
    "protocol_is_exact",
    "protocol_needs_curve",
    "protocol_world_size",
    "protocols_runnable_here",
    "psi_curve_relation",
    "resolve_subset_mode",
    "resolve_psi_protocol",
    "runnable_protocols_hint",
    "run_psi_intersection",
    "run_psi_operation",
    "subset_disclosure",
    "subset_equality_jax",
]
