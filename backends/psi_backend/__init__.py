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
    PSI_PROTOCOLS_EXPLICIT_ONLY,
    PSI_PROTOCOLS_WITH_NOISE,
    PSI_PROTOCOLS_WITHOUT_CURVE,
    PSI_PARAM_KEYS,
    PSI_RUNTIME_WORLD_SIZE,
    PsiCapabilityReport,
    PsiParamValidation,
    PsiProtocolCapability,
    check_psi_capabilities,
    check_psi_protocol_capability,
    probe_rr22_params,
    normalize_curve,
    normalize_psi_protocol,
    protocol_is_exact,
    protocol_needs_curve,
    protocol_result_semantics,
    protocol_world_size,
    psi_protocol_is_explicit_only,
    psi_protocol_spec,
    protocols_runnable_here,
    psi_curve_relation,
    resolve_psi_protocol,
    runnable_protocols_hint,
    validate_psi_protocol_params,
)
from .runtime import (
    PSI_OP_LEAKS,
    PsiRunResult,
    PsiRuntimeConfig,
    run_psi_intersection,
    run_psi_operation,
)
from .input_adapter import (
    LAYOUT_COMPARE_KEYS,
    LayoutAgreement,
    PartyInput,
    ResolvedCellInput,
    check_layout_agreement,
    load_cellset,
    normalize_inputs,
    validate_grid_code,
)
from .protocol_registry import (
    PROTOCOL_SPECS,
    PSI_CANDIDATE_OPS,
    PSI_PROTOCOL_NAMES,
    RESULT_SEMANTICS,
    RESULT_SEMANTICS_APPROXIMATE,
    RESULT_SEMANTICS_EXACT,
    RESULT_SEMANTICS_NOISY,
    PsiProtocolSpec,
    candidate_protocols_for,
    get_protocol_spec,
)
from .geosot_optimizer import (
    FINGERPRINT_HEX_CHARS,
    GEOSOT_OPTIMIZER_VERSION,
    GridCodeCompression,
    GridCodeGroup,
    GridCodePreparation,
    compress_grid_codes,
    deduplicate_grid_codes,
    fingerprint_grid_codes,
    partition_grid_codes,
    prepare_grid_codes,
    sort_grid_codes,
)
from .result_policy import (
    APPLICABLE_POLICIES_BY_OP,
    DEFAULT_POLICY_BY_OP,
    PROTOCOL_LEAK_INTERSECTION_BODY,
    RESULT_POLICIES,
    REVEAL_BOOLEAN,
    REVEAL_COUNT,
    REVEAL_INTERSECTION,
    REVEAL_TO_REGULATOR,
    ResultPolicy,
    get_result_policy,
    resolve_result_policy,
)
from .party import (
    SUPPORTED_WORLD_SIZE,
    PartyDescriptor,
    PartyManager,
    WorldConfig,
    same_party_both_sides,
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
from .benchmark import (
    PROTOCOL_VARIANTS,
    BenchmarkCase,
    BenchmarkSet,
    analyze_benchmark_set,
    compare_with_baseline,
    format_n,
    format_summary,
    generate_benchmark_sets,
    plain_intersects_reference,
    run_benchmark_case,
    run_benchmark_cases,
    standard_cases,
    unexpected_records,
    write_benchmark_csv,
    write_benchmark_json,
)

__all__ = [
    "BenchmarkCase",
    "BenchmarkSet",
    "PROTOCOL_VARIANTS",
    "PSI_CURVE_RELATION",
    "PSI_CURVES",
    "PSI_DEFAULT_CURVE",
    "PSI_DEFAULT_PROTOCOL",
    "PSI_OPS",
    "PSI_OP_LEAKS",
    "PSI_PARAM_KEYS",
    "PSI_PROTOCOL_WORLD_SIZE",
    "PSI_PROTOCOLS",
    "PSI_PROTOCOLS_CURVE_REQUIRED",
    "PSI_PROTOCOLS_EXPLICIT_ONLY",
    "PSI_PROTOCOLS_WITH_NOISE",
    "PSI_PROTOCOLS_WITHOUT_CURVE",
    "PSI_RUNTIME_WORLD_SIZE",
    "PsiCapabilityReport",
    "PsiParamValidation",
    "PsiProtocolCapability",
    "PsiProtocolSpec",
    "PsiRunResult",
    "PsiRuntimeConfig",
    "PROTOCOL_SPECS",
    "PSI_CANDIDATE_OPS",
    "PSI_PROTOCOL_NAMES",
    "RESULT_SEMANTICS",
    "RESULT_SEMANTICS_APPROXIMATE",
    "RESULT_SEMANTICS_EXACT",
    "RESULT_SEMANTICS_NOISY",
    "APPLICABLE_POLICIES_BY_OP",
    "DEFAULT_POLICY_BY_OP",
    "PROTOCOL_LEAK_INTERSECTION_BODY",
    "RESULT_POLICIES",
    "REVEAL_BOOLEAN",
    "REVEAL_COUNT",
    "REVEAL_INTERSECTION",
    "REVEAL_TO_REGULATOR",
    "ResultPolicy",
    "get_result_policy",
    "resolve_result_policy",
    "FINGERPRINT_HEX_CHARS",
    "GEOSOT_OPTIMIZER_VERSION",
    "GridCodeCompression",
    "GridCodeGroup",
    "GridCodePreparation",
    "compress_grid_codes",
    "deduplicate_grid_codes",
    "fingerprint_grid_codes",
    "partition_grid_codes",
    "prepare_grid_codes",
    "sort_grid_codes",
    "SUPPORTED_WORLD_SIZE",
    "PartyDescriptor",
    "PartyManager",
    "WorldConfig",
    "same_party_both_sides",
    "candidate_protocols_for",
    "get_protocol_spec",
    "LAYOUT_COMPARE_KEYS",
    "LayoutAgreement",
    "PartyInput",
    "ResolvedCellInput",
    "SUBSET_DEFAULT_FIELD",
    "SUBSET_DEFAULT_PROTOCOL",
    "SUBSET_MODES",
    "SUBSET_MODE_DISCLOSURE",
    "SUBSET_MODE_MPC",
    "SUBSET_MODE_PLAINTEXT",
    "SUBSET_MODE_PLAINTEXT_FALLBACK",
    "SubsetComparison",
    "analyze_benchmark_set",
    "check_layout_agreement",
    "compare_with_baseline",
    "check_psi_capabilities",
    "check_psi_protocol_capability",
    "format_n",
    "format_summary",
    "generate_benchmark_sets",
    "load_cellset",
    "mpc_subset_comparison",
    "normalize_curve",
    "normalize_inputs",
    "normalize_psi_protocol",
    "plain_intersects_reference",
    "plaintext_subset_comparison",
    "probe_rr22_params",
    "protocol_is_exact",
    "protocol_needs_curve",
    "protocol_result_semantics",
    "protocol_world_size",
    "psi_protocol_is_explicit_only",
    "psi_protocol_spec",
    "protocols_runnable_here",
    "psi_curve_relation",
    "resolve_subset_mode",
    "resolve_psi_protocol",
    "validate_grid_code",
    "runnable_protocols_hint",
    "run_benchmark_case",
    "run_benchmark_cases",
    "run_psi_intersection",
    "run_psi_operation",
    "subset_disclosure",
    "subset_equality_jax",
    "standard_cases",
    "unexpected_records",
    "validate_psi_protocol_params",
    "write_benchmark_csv",
    "write_benchmark_json",
]
