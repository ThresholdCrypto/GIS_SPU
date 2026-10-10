# -*- coding: utf-8 -*-
"""PSI-Cardinality 后端：只出交集基数的两方计算（openmined-psi）。

设计边界（刻意的）
------------------
- **不实现协议本体**：调用 OpenMined 官方实现（Apache-2.0，
  ``pip install openmined-psi==2.0.6``）；本项目只做能力核查、封装与对拍；
- **不改 SPU、不依赖 libpsi**：这是与 ``backends/psi_backend`` 并列的
  第四个后端族（首个"外部协议接入"实践），两条路径的泄漏承诺不同：
  - ``psi_backend``（libpsi 求交）：接收方在协议内部获得**交集本体**；
  - 本后端（PSI-CA）：协议只计算**计数**，交集本体不交给任一方。
- **只接一个口子**：``CellSetIntersect`` 的计数档（``REVEAL_COUNT``，由
  ``geo-secure build --psi-count psi-ca`` 触发）；其余算子 / 策略显式拒绝。
- 只用 ``RAW`` 数据结构（精确计数）；GCS / BloomFilter 近似档未接。

API 依据（2026-10-09 从上游源码核对）
------------------------------------
- ``private_set_intersection/python/__init__.py``（方法签名）；
- ``private_set_intersection/cpp/psi_server.cpp``（RAW 忽略 fpr；
  reveal_intersection 两侧必须一致）；
- 发行版 ``openmined-psi==2.0.6``（PyPI 仅 macOS / manylinux 轮子）。

说明文档：``docs/PSI_CA_CAPABILITY.md``；选型依据：``outputs/新协议选型报告.md``。
"""

from .capability import (
    PSI_CA_BACKEND,
    PSI_CA_DISTRIBUTION,
    PSI_CA_IMPORT_PATH,
    PSI_CA_PINNED_VERSION,
    PSI_CA_STRUCTURE,
    PSI_CA_SUPPORTED_OPS,
    PSI_CA_SUPPORTED_POLICIES,
    PsiCaCapabilityReport,
    check_psi_ca_capabilities,
    import_openmined_psi,
    missing_api,
)
from .runtime import (
    PSI_CA_COMMUNICATION_METER,
    PSI_CA_LEARNING_RANK,
    PSI_CA_OP_LEAKS,
    PSI_CA_PROTOCOL,
    PSI_CA_PROTOCOL_LEAK,
    run_psi_cardinality,
)

__all__ = [
    "PSI_CA_BACKEND",
    "PSI_CA_COMMUNICATION_METER",
    "PSI_CA_DISTRIBUTION",
    "PSI_CA_IMPORT_PATH",
    "PSI_CA_LEARNING_RANK",
    "PSI_CA_OP_LEAKS",
    "PSI_CA_PINNED_VERSION",
    "PSI_CA_PROTOCOL",
    "PSI_CA_PROTOCOL_LEAK",
    "PSI_CA_STRUCTURE",
    "PSI_CA_SUPPORTED_OPS",
    "PSI_CA_SUPPORTED_POLICIES",
    "PsiCaCapabilityReport",
    "check_psi_ca_capabilities",
    "import_openmined_psi",
    "missing_api",
    "run_psi_cardinality",
]
