# -*- coding: utf-8 -*-
"""兼容入口：结果策略已归并到 ``backends/result_policy.py``（PSI / MPC 共用）。

Phase 10 把原模块整体搬到顶层（结果策略必须独立于密码协议——任务书 §十），
本文件只做再导出，保持既有 import 路径（``backends.psi_backend.result_policy``）
与 ``backends.psi_backend`` 包的公开名单不变；新增代码请直接从
``backends.result_policy`` 导入。
"""

from __future__ import annotations

from backends.result_policy import (
    APPLICABLE_POLICIES_BY_OP,
    DEFAULT_POLICY_BY_OP,
    MPC_OP_REVEALS,
    MPC_POLICIES,
    OP_RESULT_FAMILY,
    PROTOCOL_LEAK_INTERSECTION_BODY,
    PROTOCOL_LEAK_NONE,
    PROTOCOL_LEAK_OUTPUT_ONLY,
    PSI_POLICIES,
    RESULT_FAMILY_MPC,
    RESULT_FAMILY_PSI,
    RESULT_POLICIES,
    REVEAL_BOOLEAN,
    REVEAL_COUNT,
    REVEAL_INTERSECTION,
    REVEAL_TO_REGULATOR,
    REVEAL_VALUE,
    ResultPolicy,
    get_result_policy,
    resolve_result_policy,
)

__all__ = [
    "APPLICABLE_POLICIES_BY_OP",
    "DEFAULT_POLICY_BY_OP",
    "MPC_OP_REVEALS",
    "MPC_POLICIES",
    "OP_RESULT_FAMILY",
    "PROTOCOL_LEAK_INTERSECTION_BODY",
    "PROTOCOL_LEAK_NONE",
    "PROTOCOL_LEAK_OUTPUT_ONLY",
    "PSI_POLICIES",
    "RESULT_FAMILY_MPC",
    "RESULT_FAMILY_PSI",
    "RESULT_POLICIES",
    "REVEAL_BOOLEAN",
    "REVEAL_COUNT",
    "REVEAL_INTERSECTION",
    "REVEAL_TO_REGULATOR",
    "REVEAL_VALUE",
    "ResultPolicy",
    "get_result_policy",
    "resolve_result_policy",
]
