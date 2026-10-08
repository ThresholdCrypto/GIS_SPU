# -*- coding: utf-8 -*-
"""`Contains` 密态子集比较的测试。

纪律与其它后端一致：**先跑真实实现，再断言**；跑不了就按原因 skip，
绝不放宽断言、也绝不返回推测值。

覆盖四件事：
  1. 电路本身：原语清单与注册表一致、只有两个 i32 入参、jax.jit 可追踪；
  2. 模式口径：闭集、披露文案互不相同、非法值 fail-fast；
  3. 真实执行：MPC 结果与明文逐点一致（格点扫描），误差 0.0；
  4. 隐私到底保住了什么：进 MPC 的只有两个**基数**，格网码一个都没进去。
"""

from __future__ import annotations

import re

import pytest

from backends import plain
from backends.psi_backend import (
    SUBSET_MODE_DISCLOSURE,
    SUBSET_MODE_MPC,
    SUBSET_MODE_PLAINTEXT,
    SUBSET_MODE_PLAINTEXT_FALLBACK,
    SUBSET_MODES,
    SubsetComparison,
    mpc_subset_comparison,
    plaintext_subset_comparison,
    resolve_subset_mode,
    run_psi_intersection,
    subset_disclosure,
    subset_equality_jax,
)
from backends.spu_backend import (
    OP_HLO_PRIMITIVES,
    SpuRunResult,
    check_operation_capability,
)
from ir import encode_grid_code

from tests._helpers import has_jax, has_psi, has_spu

needs_jax = pytest.mark.skipif(not has_jax(), reason="当前环境没有 jax")
needs_spu = pytest.mark.skipif(
    not has_spu(), reason="当前环境无法真实执行 SPU 模拟（见 backends.spu_backend.capability）"
)
needs_psi = pytest.mark.skipif(
    not has_psi(), reason="当前环境不具备真实 PSI 执行能力(python3.11+spu+libgomp1)"
)

_T = 0


def _code(x: int, y: int = 27702, z: int = 15, level: int = 9) -> int:
    return encode_grid_code(x=x, y=y, z=z, level=level, toff=_T, lt=4)


ROUTE = (_code(21861), _code(21862), _code(21863))
ZONE = (_code(21862), _code(21863), _code(22999))
EXPECTED_INTERSECTION = (ZONE[0], ZONE[1])

#: Contains 的四个方向/规模组合：真/假、相等、互不包含
CONTAINS_CASES = [
    (ZONE, EXPECTED_INTERSECTION),
    (EXPECTED_INTERSECTION, ZONE),
    (ROUTE, ROUTE),
    (ROUTE, ZONE),
]


def _plain_reference(left, right):
    return plain.plain_contains(left, right).value


def _hlo_ops_and_args(source_fn, *args):
    """取真实编译产物里的原语名与 %arg 个数——不看源码注释，看产物。"""

    import jax

    text = jax.jit(source_fn).lower(*args).as_text()
    ops = re.findall(r"=\s*(?:stablehlo|mhlo)\.(\w+)", text)
    seen: list[str] = []
    for name in ops:
        if name not in seen:
            seen.append(name)
    return seen, re.findall(r"%arg\d+:", text), text


# --------------------------------------------------------------------------
# 1. 电路本身
# --------------------------------------------------------------------------


@needs_jax
class TestCircuit:
    def test_primitives_match_the_registered_capability_list(self):
        """注册表里写的原语必须就是电路真正用到的，不能靠推测。"""

        import jax.numpy as jnp

        ops, _, _ = _hlo_ops_and_args(subset_equality_jax, jnp.int32(2), jnp.int32(2))
        assert ops == list(OP_HLO_PRIMITIVES["Contains"]), ops

    def test_registered_primitives_are_all_spu_adapted(self):
        from backends.spu_backend import SPU_ADAPTED_HLO_PRIMITIVES

        for name in OP_HLO_PRIMITIVES["Contains"]:
            assert name in SPU_ADAPTED_HLO_PRIMITIVES, name

    def test_takes_exactly_two_int32_scalars(self):
        """入参就是两个基数：不是集合、不是张量、没有第三个。"""

        import jax.numpy as jnp

        _, args, text = _hlo_ops_and_args(subset_equality_jax, jnp.int32(2), jnp.int32(3))
        assert len(args) == 2, args
        assert text.count("tensor<i32>") >= 2
        assert "i64" not in text and "ui64" not in text

    def test_is_jit_traceable(self):
        import jax
        import jax.numpy as jnp

        fn = jax.jit(subset_equality_jax)
        assert int(fn(jnp.int32(3), jnp.int32(3))) == 1
        assert int(fn(jnp.int32(3), jnp.int32(4))) == 0


# --------------------------------------------------------------------------
# 2. 模式口径
# --------------------------------------------------------------------------


class TestModes:
    def test_modes_are_a_closed_set(self):
        assert set(SUBSET_MODES) == {
            SUBSET_MODE_MPC,
            SUBSET_MODE_PLAINTEXT,
            SUBSET_MODE_PLAINTEXT_FALLBACK,
        }
        assert set(SUBSET_MODE_DISCLOSURE) == set(SUBSET_MODES)

    def test_disclosures_are_specific_not_boilerplate(self):
        texts = [SUBSET_MODE_DISCLOSURE[m] for m in SUBSET_MODES]
        assert len(set(texts)) == len(texts), "三种模式的披露文案不允许复制粘贴"

    def test_only_the_fallback_mentions_being_a_fallback(self):
        for mode in SUBSET_MODES:
            text = subset_disclosure(mode)
            if mode == SUBSET_MODE_PLAINTEXT_FALLBACK:
                assert "回退" in text
            else:
                assert "回退" not in text

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError) as err:
            subset_disclosure("mpc-ish")
        assert "mpc-ish" in str(err.value)

    def test_resolve_accepts_the_two_selectable_modes(self):
        assert resolve_subset_mode("MPC") == SUBSET_MODE_MPC
        assert resolve_subset_mode(" plaintext ") == SUBSET_MODE_PLAINTEXT

    def test_resolve_rejects_the_fallback(self):
        """退路是运行时自动产生的，不许被显式指定——否则日志里分不清两者。"""

        with pytest.raises(ValueError) as err:
            resolve_subset_mode(SUBSET_MODE_PLAINTEXT_FALLBACK)
        assert SUBSET_MODE_PLAINTEXT_FALLBACK in str(err.value)

    def test_plaintext_helper_refuses_the_mpc_mode(self):
        with pytest.raises(ValueError):
            plaintext_subset_comparison(ROUTE, ZONE, mode=SUBSET_MODE_MPC)

    def test_plaintext_helper_matches_backends_plain(self):
        for outer, inner in CONTAINS_CASES:
            got = plaintext_subset_comparison(outer, inner).value
            assert got == _plain_reference(outer, inner), (outer, inner)

    def test_comparison_reports_its_own_mode(self):
        mpc_like = SubsetComparison(mode=SUBSET_MODE_MPC, status="ok", value=True)
        plain_like = SubsetComparison(mode=SUBSET_MODE_PLAINTEXT, status="forced", value=True)
        assert mpc_like.is_mpc is True and plain_like.is_mpc is False
        assert mpc_like.disclosure != plain_like.disclosure


# --------------------------------------------------------------------------
# 3. 真实执行：与明文逐点一致
# --------------------------------------------------------------------------


@needs_spu
class TestMpcExactness:
    def test_equality_grid_matches_plaintext(self):
        """k==n 的结论必须逐点正确，且误差为 0（整数电路没有理由有误差）。"""

        for k in range(0, 6):
            for n in range(0, 6):
                run = mpc_subset_comparison(k, n)
                assert run.status == "ok", run.note
                assert run.value is (k == n), (k, n, run.value)
                assert run.agreement is True, (k, n, run.max_abs_error)
                assert run.max_abs_error == 0.0
                assert run.pphlo_bytes and run.pphlo_bytes > 0

    def test_uses_the_requested_protocol_and_field(self):
        run = mpc_subset_comparison(2, 2, protocol="SEMI2K", field=32)
        assert run.status == "ok", run.note
        assert run.protocol == "SEMI2K" and run.field == "FM32"

    def test_never_fabricates_when_spu_is_absent(self, monkeypatch):
        import backends.psi_backend.subset_mpc as mod

        def fake(*args, **kwargs):
            return SpuRunResult(
                status="unavailable",
                protocol=str(kwargs.get("protocol", "ABY3")),
                field="FM64",
                world_size=3,
                blockers=("模拟：当前环境不可运行 SPU",),
            )

        monkeypatch.setattr(mod, "run_spu_simulation", fake)
        run = mpc_subset_comparison(2, 2)
        assert run.value is None
        assert run.agreement is None
        assert "不可运行" in run.note


# --------------------------------------------------------------------------
# 4. 隐私到底保住了什么
# --------------------------------------------------------------------------


@needs_psi
class TestInputPrivacy:
    def test_only_the_two_cardinalities_enter_the_mpc_circuit(self, monkeypatch):
        """这条断言就是本功能的收益：进 MPC 的只有两个**基数**。

        格网码是 64 位大整数，若它们进过电路，捕获到的入参里必然出现它们。
        """

        import backends.psi_backend.runtime as rt

        captured: list[tuple] = []
        real = rt.mpc_subset_comparison

        def spy(intersection_card, inner_card, **kwargs):
            captured.append((intersection_card, inner_card))
            return real(intersection_card, inner_card, **kwargs)

        monkeypatch.setattr(rt, "mpc_subset_comparison", spy)
        run = run_psi_intersection(ZONE, EXPECTED_INTERSECTION, op="Contains")

        assert run.subset.mode == SUBSET_MODE_MPC
        assert captured == [(2, 2)], captured
        for code in ZONE + EXPECTED_INTERSECTION:
            for k, n in captured:
                assert code not in (k, n), "格网码不得进入 MPC 电路"

    def test_the_receiver_still_gets_the_intersection_body(self):
        """不淡化：密态子集比较消掉的是第二次明文比较，不是交集本体。"""

        run = run_psi_intersection(ZONE, EXPECTED_INTERSECTION, op="Contains")
        assert run.intersection == EXPECTED_INTERSECTION
        assert "交集本体" in run.reveals


# --------------------------------------------------------------------------
# 5. 集成：run_psi_intersection 的 Contains 分支
# --------------------------------------------------------------------------


@needs_psi
class TestContainsIntegration:
    def test_default_mode_is_mpc(self):
        run = run_psi_intersection(ZONE, EXPECTED_INTERSECTION, op="Contains")
        assert run.status == "ok", run.error
        assert run.subset is not None
        assert run.subset.mode == SUBSET_MODE_MPC
        assert run.subset.status == "ok"
        assert run.subset.is_mpc is True

    def test_both_modes_give_the_same_verdict(self):
        for outer, inner in CONTAINS_CASES:
            expected = _plain_reference(outer, inner)
            mpc = run_psi_intersection(outer, inner, op="Contains", subset_via="mpc")
            plain_run = run_psi_intersection(
                outer, inner, op="Contains", subset_via="plaintext"
            )
            assert mpc.value == expected, (outer, inner, mpc.value)
            assert plain_run.value == expected, (outer, inner, plain_run.value)
            assert mpc.subset.mode == SUBSET_MODE_MPC
            assert plain_run.subset.mode == SUBSET_MODE_PLAINTEXT

    def test_reveals_follow_the_mode(self):
        mpc = run_psi_intersection(ZONE, EXPECTED_INTERSECTION, op="Contains")
        forced = run_psi_intersection(
            ZONE, EXPECTED_INTERSECTION, op="Contains", subset_via="plaintext"
        )
        assert "MPC 基数等值" in mpc.reveals
        assert "明文比较" in forced.reveals
        assert mpc.reveals != forced.reveals

    def test_notes_record_the_mode(self):
        run = run_psi_intersection(ZONE, EXPECTED_INTERSECTION, op="Contains")
        assert any(SUBSET_MODE_MPC in note for note in run.notes), run.notes

    def test_subset_travels_through_to_dict(self):
        run = run_psi_intersection(ZONE, EXPECTED_INTERSECTION, op="Contains")
        payload = run.to_dict()
        assert payload["subset"]["mode"] == SUBSET_MODE_MPC
        assert payload["subset"]["value"] is True
        assert payload["subset"]["disclosure"]

    def test_other_ops_have_no_subset(self):
        run = run_psi_intersection(ROUTE, ZONE, op="Intersects")
        assert run.subset is None
        assert run.reveals == __import__(
            "backends.psi_backend", fromlist=["PSI_OP_LEAKS"]
        ).PSI_OP_LEAKS["Intersects"]

    def test_empty_input_leaves_the_subset_unset(self):
        """空集合是前置判定，协议都没启动，谈不上子集比较。"""

        run = run_psi_intersection(ZONE, [], op="Contains")
        assert run.status == "empty-input"
        assert run.subset is None
        assert "未启动 PSI" in " ".join(run.notes)

    def test_invalid_subset_via_is_rejected_readably(self):
        run = run_psi_intersection(ZONE, ZONE, op="Contains", subset_via="nope")
        assert run.status == "error"
        assert "nope" in run.error and SUBSET_MODE_MPC in run.error

    def test_fallback_is_disclosed_when_mpc_is_unavailable(self, monkeypatch):
        """MPC 不可用时退路是明文——但必须三处都写明，不能静默。"""

        import backends.psi_backend.subset_mpc as mod

        def fake(*args, **kwargs):
            return SpuRunResult(
                status="unavailable",
                protocol=str(kwargs.get("protocol", "ABY3")),
                field="FM64",
                world_size=3,
                blockers=("模拟：当前环境不可运行 SPU",),
            )

        monkeypatch.setattr(mod, "run_spu_simulation", fake)
        run = run_psi_intersection(ZONE, EXPECTED_INTERSECTION, op="Contains")

        assert run.status == "ok", run.error
        assert run.value is True
        assert run.subset.mode == SUBSET_MODE_PLAINTEXT_FALLBACK
        assert run.subset.is_mpc is False
        assert "回退" in run.reveals
        assert any("退路" in note for note in run.notes), run.notes


# --------------------------------------------------------------------------
# 6. 能力核查不再对 Contains 说错话
# --------------------------------------------------------------------------


class TestCapabilityRegistration:
    @needs_spu
    def test_contains_declares_the_mpc_primitives(self):
        capability = check_operation_capability("Contains")
        assert capability.supported is True
        assert set(capability.primitives) == {"compare", "convert"}

    def test_contains_no_longer_claims_grid_code_overflows(self):
        """grid_code 全程不出 PSI，进 MPC 的只有基数——照抄旧告警就是误导。"""

        capability = check_operation_capability("Contains")
        joined = " ".join(capability.warnings)
        assert "grid_code 不进 MPC" in joined
        assert "存在溢出风险" not in joined

    def test_psi_only_ops_keep_the_overflow_warning(self):
        for op in ("Intersects", "CellSetIntersect"):
            joined = " ".join(check_operation_capability(op).warnings)
            assert "存在溢出风险" in joined, op
