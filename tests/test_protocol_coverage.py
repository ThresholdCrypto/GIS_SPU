# -*- coding: utf-8 -*-
"""协议覆盖镜像：登记的协议必须被**显式归类**，不许静默滑过。

为什么需要这一个文件
==================
协议事实分散在三处：SPU 枚举（`backends/spu_backend/capability.SPU_PROTOCOLS`）、
PSI 注册表（`backends/psi_backend/protocol_registry.PROTOCOL_SPECS`）、
以及真实执行。**"登记了"不等于"接线了"，"接线了"不等于"测过了"** ——
本项目吃过这个亏（`PROTOCOL_ECDH_3PC` 曾与两方协议并列在"可用"清单里）。

本文件把三者钉在一起：每个登记协议必须恰好属于下面三类之一，且不可交叉。
升级 SPU 后新增/移除协议时，这里会红——强制做一次**显式决定**，
而不是让新协议默认按"看起来支持"滑过去。

这三张表是**手工维护的契约**；它们的证据是
`tests/test_spu_backend.py::TestProtocolFieldSweep` 的矩阵扫描。
"""

from __future__ import annotations

from backends.psi_backend import (
    PSI_PROTOCOL_WORLD_SIZE,
    PSI_PROTOCOLS,
    protocols_runnable_here,
)
from backends.spu_backend import (
    SPU_FIELDS,
    SPU_PROTOCOLS,
    SPU_PROTOCOLS_WITHOUT_CRYPTO,
    protocol_min_world_size,
)

# --------------------------------------------------------------------------
# SPU（MPC）侧
# --------------------------------------------------------------------------

#: 本仓库用真实执行验证过的协议（2026-10-04，WSL2 + SPU 0.9.5）。
#: 证据：TestProtocolFieldSweep 的 protocol × op 扫描全部 within_tolerance。
SPU_PROTOCOLS_VERIFIED: frozenset[str] = frozenset(SPU_PROTOCOLS)

#: 登记但当前不可执行的协议 → 可读原因（必须非空字符串）。
SPU_PROTOCOLS_UNAVAILABLE: dict[str, str] = {}

#: **不提供密码学保护**的协议（P4）——与 capability 层单源的镜像。
#: 判据是通信量实测：`REF2K` 在三个 MPC 算子上 send+recv 恒为 0 B
#: （`docs/mpc_comm_baseline.json`，`--comm --repeat 5`），
#: 见 `docs/MPC_BENCHMARK_PROTOCOL.md` §8.4 / §8.5。
#: 这是一个**必须显式决定**的类别：升级 SPU 带进新协议时，这里会红，
#: 逼着确认新协议受不受密码学保护，而不是让它默认被自动选中。
SPU_PROTOCOLS_WITHOUT_CRYPTO_MIRROR: frozenset[str] = frozenset({"REF2K"})

# --------------------------------------------------------------------------
# PSI 侧
# --------------------------------------------------------------------------

#: 本仓库真机执行过的协议（2026-10-04）。证据：tests/test_psi_backend.py 相关用例。
#: NPC 族由 TestRealPsiIntersection::test_npc_protocols_are_really_executed 覆盖。
PSI_PROTOCOLS_EXECUTED: frozenset[str] = frozenset(
    {
        "PROTOCOL_ECDH",
        "PROTOCOL_KKRT",
        "PROTOCOL_RR22",
        "PROTOCOL_DP",
        "PROTOCOL_ECDH_NPC",
        "PROTOCOL_KKRT_NPC",
    }
)

#: 运行期**允许**跑、但尚未真机执行——不得计入"已验证"。
#: 当前为空：NPC 族已于 2026-10-04 真机验证。新增协议若只登记未执行，落在这里。
PSI_PROTOCOLS_RUNTIME_ELIGIBLE: frozenset[str] = frozenset()

#: 登记但本链路不可执行 → 可读原因。
PSI_PROTOCOLS_UNAVAILABLE: dict[str, str] = {
    "PROTOCOL_ECDH_3PC": "需要 3 个参与方；本编译器进程内链路固定两方（§16）",
}


class TestSpuProtocolCoverage:
    def test_classification_covers_every_registered_protocol(self):
        classified = SPU_PROTOCOLS_VERIFIED | set(SPU_PROTOCOLS_UNAVAILABLE)
        assert classified == set(SPU_PROTOCOLS), (
            f"未归类：{set(SPU_PROTOCOLS) - classified}；"
            f"已注销却仍列在表里：{classified - set(SPU_PROTOCOLS)}"
        )

    def test_classification_is_disjoint(self):
        both = SPU_PROTOCOLS_VERIFIED & set(SPU_PROTOCOLS_UNAVAILABLE)
        assert both == set(), f"同一协议不能既已验证又不可执行：{both}"

    def test_unavailable_entries_state_a_reason(self):
        for name, reason in SPU_PROTOCOLS_UNAVAILABLE.items():
            assert reason.strip(), f"{name} 未给原因"

    def test_verified_protocols_have_a_world_size(self):
        for name in SPU_PROTOCOLS_VERIFIED:
            assert protocol_min_world_size(name) >= 2, name

    def test_verified_set_is_within_the_sweep_input(self):
        """扫描（TestProtocolFieldSweep）的 parametrize 源就是 SPU_PROTOCOLS；
        这里防止手工表与扫描范围脱节（表里出现没被扫描过的协议）。"""

        assert SPU_PROTOCOLS_VERIFIED <= set(SPU_PROTOCOLS)

    def test_fields_are_the_three_documented_widths(self):
        assert set(SPU_FIELDS) == {"FM32", "FM64", "FM128"}

    def test_crypto_protection_classification_is_explicit(self):
        """每个登记协议都要能被显式判断"受不受密码学保护"。

        表为空 → "无保护"这个类别会静默消失（于是所有协议都被当成受保护、
        都可自动选中）；表等于全集 → 自动选择直接没有候选。两头都要拦住，
        中间的新增协议则在此被迫做一次**显式决定**。
        """

        assert frozenset(SPU_PROTOCOLS_WITHOUT_CRYPTO) == SPU_PROTOCOLS_WITHOUT_CRYPTO_MIRROR
        assert SPU_PROTOCOLS_WITHOUT_CRYPTO_MIRROR <= set(SPU_PROTOCOLS)
        assert SPU_PROTOCOLS_WITHOUT_CRYPTO_MIRROR, "无保护类别不许为空集合"
        assert SPU_PROTOCOLS_WITHOUT_CRYPTO_MIRROR < set(SPU_PROTOCOLS), (
            "全是无保护协议：自动选择会没有候选，须显式确认"
        )


class TestPsiProtocolCoverage:
    def test_classification_covers_every_registered_protocol(self):
        classified = (
            PSI_PROTOCOLS_EXECUTED
            | PSI_PROTOCOLS_RUNTIME_ELIGIBLE
            | set(PSI_PROTOCOLS_UNAVAILABLE)
        )
        assert classified == set(PSI_PROTOCOLS), (
            f"未归类：{set(PSI_PROTOCOLS) - classified}；"
            f"已注销却仍列在表里：{classified - set(PSI_PROTOCOLS)}"
        )

    def test_three_classes_are_pairwise_disjoint(self):
        groups = {
            "executed": PSI_PROTOCOLS_EXECUTED,
            "eligible-not-executed": PSI_PROTOCOLS_RUNTIME_ELIGIBLE,
            "unavailable": set(PSI_PROTOCOLS_UNAVAILABLE),
        }
        for left, right in (("executed", "eligible-not-executed"),
                            ("executed", "unavailable"),
                            ("eligible-not-executed", "unavailable")):
            overlap = groups[left] & groups[right]
            assert overlap == set(), f"{left} ∩ {right} = {overlap}"

    def test_tables_agree_with_the_live_runnable_detection(self):
        """手工表必须与运行期判定**逐项**一致——防两处漂移。"""

        runnable = set(protocols_runnable_here())
        assert runnable == (PSI_PROTOCOLS_EXECUTED | PSI_PROTOCOLS_RUNTIME_ELIGIBLE)
        assert set(PSI_PROTOCOLS) - runnable == set(PSI_PROTOCOLS_UNAVAILABLE)

    def test_unavailable_entries_state_a_reason(self):
        for name, reason in PSI_PROTOCOLS_UNAVAILABLE.items():
            assert reason.strip(), f"{name} 未给原因"

    def test_unavailable_protocols_require_more_than_two_parties(self):
        """本链路只有不可执行的一种成因：需要多于两方。出现别的成因时
        这条会红——提示去补一条真正的原因说明，而不是含糊带过。"""

        for name in PSI_PROTOCOLS_UNAVAILABLE:
            assert PSI_PROTOCOL_WORLD_SIZE[name] > 2, name
