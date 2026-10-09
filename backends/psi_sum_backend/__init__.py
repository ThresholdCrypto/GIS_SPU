# -*- coding: utf-8 -*-
"""Private Intersection-Sum 后端：交集内求和的两方计算（private-join-and-compute）。

设计边界（刻意的）
------------------
- **不实现协议本体**：调用 Google 官方实现（Apache-2.0，
  ``google/private-join-and-compute``）；本项目只做能力核查、封装与对拍；
- **不改 SPU、不依赖 libpsi**：这是与 ``backends/psi_backend``（libpsi 求交）、
  ``backends/psi_ca_backend``（PSI-Cardinality）并列的**第三条 PSI 路径**
  （第二个"外部协议接入"实践）。三条路径的泄漏承诺互不相同：
  - ``psi_backend``：协议内部把**交集本体**交给接收方；
  - ``psi_ca_backend``：协议只算**基数**；
  - 本后端：协议只算**基数 + 交集内关联值之和**。
- **只接一个口子**：``CellSetIntersect`` 的交集内求和档
  （``REVEAL_INTERSECTION_SUM``，由 ``geo-secure build --psi-sum pjc`` 触发）；
  其余算子 / 策略显式拒绝。
- **上游没有官方 PyPI 包**：接入对象是 Bazel 构建出来的两个可执行文件，
  能力核查核对的是"二进制在不在 + flag 形态对不对"，不是模块导入。

API 依据（2026-10-09 对 master@950c5e4c 从上游源码逐字核对）
------------------------------------------------------------
- ``README.md``（功能定义、honest-but-curious 模型、泄漏告警）；
- ``private_join_and_compute/client.cc`` / ``server.cc``（flag 与默认端口）；
- ``private_join_and_compute/client_impl.cc``（结果行原文）；
- ``private_join_and_compute/data_util.cc``（CSV 列数与取值范围）。

说明文档：``docs/PSI_SUM_CAPABILITY.md``；选型依据：``outputs/新协议选型报告.md``。
"""

from .capability import (
    PSI_SUM_BACKEND,
    PSI_SUM_BAZEL_VERSION,
    PSI_SUM_BINARIES,
    PSI_SUM_BIN_ENV,
    PSI_SUM_CLIENT_BINARY,
    PSI_SUM_LICENSE,
    PSI_SUM_REQUIRED_FLAGS,
    PSI_SUM_SERVER_BINARY,
    PSI_SUM_SUPPORTED_OPS,
    PSI_SUM_SUPPORTED_POLICIES,
    PSI_SUM_UPSTREAM,
    PSI_SUM_UPSTREAM_COMMIT,
    PsiSumCapabilityReport,
    binary_path,
    check_psi_sum_capabilities,
    probe_binary_flags,
    resolve_bin_dir,
)
from .runtime import (
    PSI_SUM_LEARNING_RANK,
    PSI_SUM_OP_LEAKS,
    PSI_SUM_PROTOCOL,
    PSI_SUM_PROTOCOL_LEAK,
    PSI_SUM_RESULT_POLICY,
    run_psi_intersection_sum,
    spawn_pjc,
)

__all__ = [
    "PSI_SUM_BACKEND",
    "PSI_SUM_BAZEL_VERSION",
    "PSI_SUM_BINARIES",
    "PSI_SUM_BIN_ENV",
    "PSI_SUM_CLIENT_BINARY",
    "PSI_SUM_LEARNING_RANK",
    "PSI_SUM_LICENSE",
    "PSI_SUM_OP_LEAKS",
    "PSI_SUM_PROTOCOL",
    "PSI_SUM_PROTOCOL_LEAK",
    "PSI_SUM_REQUIRED_FLAGS",
    "PSI_SUM_RESULT_POLICY",
    "PSI_SUM_SERVER_BINARY",
    "PSI_SUM_SUPPORTED_OPS",
    "PSI_SUM_SUPPORTED_POLICIES",
    "PSI_SUM_UPSTREAM",
    "PSI_SUM_UPSTREAM_COMMIT",
    "PsiSumCapabilityReport",
    "binary_path",
    "check_psi_sum_capabilities",
    "probe_binary_flags",
    "resolve_bin_dir",
    "run_psi_intersection_sum",
    "spawn_pjc",
]
