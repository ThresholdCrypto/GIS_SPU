"""PSI 后端测试：能力探测、协议归一化、真实求交、泄漏面、诚实留空。

与本项目其它后端的同一条纪律：**先跑真实实现，再断言**。
环境不具备真实 PSI 时，相关用例按原因 skip，而不是让断言放宽。
"""

from __future__ import annotations

import pytest

from backends import plain
from backends.psi_backend import (
    PSI_CURVE_RELATION,
    PSI_CURVES,
    PSI_DEFAULT_CURVE,
    PSI_DEFAULT_PROTOCOL,
    PSI_OPS,
    PSI_OP_LEAKS,
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
    run_psi_intersection,
    run_psi_operation,
    runnable_protocols_hint,
)
from ir import encode_grid_code

from tests._helpers import has_psi

# --------------------------------------------------------------------------
# 样例格网码：真实 64 位编码，且有交集而互不包含
# --------------------------------------------------------------------------

_T = 0


def _code(x: int, y: int = 27702, z: int = 15, level: int = 9) -> int:
    return encode_grid_code(x=x, y=y, z=z, level=level, toff=_T, lt=4)


ROUTE = (_code(21861), _code(21862), _code(21863))
ZONE = (_code(21862), _code(21863), _code(22999))
#: 真值断言：交集恰好 2 个，且两侧都不包含对方
EXPECTED_INTERSECTION = (ZONE[0], ZONE[1])

needs_psi = pytest.mark.skipif(not has_psi(), reason="当前环境不具备真实 PSI 执行能力(python3.11+spu+libgomp1)")


# --------------------------------------------------------------------------
# 1. 能力探测
# --------------------------------------------------------------------------


class TestCapability:
    def test_probe_reports_installed_and_version(self):
        report = check_psi_capabilities()
        if not report.installed:
            pytest.skip(f"未安装 spu：{report.error}")
        assert report.version

    def test_enum_lists_are_not_silently_empty(self):
        """回归：官方枚举是 pybind11 类型，成员不是 int 实例。

        早期 `_enum_members` 只判 `isinstance(value, int)`，导致协议/曲线
        清单**静默为空**——那样"默认协议是否可用"的判据就失去依据，
        却不会报错。这里直接把清单非空作为契约。
        """

        report = check_psi_capabilities()
        if not report.installed:
            pytest.skip("未安装 spu")
        assert report.protocols, "协议清单为空：枚举探测失效"
        assert report.curves, "曲线清单为空：枚举探测失效"
        assert report.source_types, "输入形态清单为空"

    def test_probe_agrees_with_module_constants(self):
        report = check_psi_capabilities()
        if not report.installed:
            pytest.skip("未安装 spu")
        assert set(report.protocols) == set(PSI_PROTOCOLS)
        assert set(report.curves) == set(PSI_CURVES)

    def test_file_io_only_is_declared(self):
        """PSI 当前只有 CSV 文件接口；这一事实必须显式暴露，不能默认有内存接口。"""

        report = check_psi_capabilities()
        if not report.installed:
            pytest.skip("未安装 spu")
        assert report.file_io_only is True
        assert "SOURCE_TYPE_FILE_CSV" in report.source_types

    def test_supported_ops_lists_the_psi_family(self):
        report = check_psi_capabilities()
        if not report.runnable:
            assert report.supported_ops == ()
            return
        assert set(report.supported_ops) == set(PSI_OPS)


# --------------------------------------------------------------------------
# 2. 协议 / 曲线归一化
# --------------------------------------------------------------------------


class TestProtocolNormalization:
    def test_protocol_accepts_short_and_full_forms(self):
        assert normalize_psi_protocol("ecdh") == "PROTOCOL_ECDH"
        assert normalize_psi_protocol("PROTOCOL_ECDH") == "PROTOCOL_ECDH"
        assert normalize_psi_protocol(" Kkrt ") == "PROTOCOL_KKRT"

    def test_unknown_protocol_raises_with_available_list(self):
        with pytest.raises(ValueError) as excinfo:
            normalize_psi_protocol("PROTOCOL_NOPE")
        assert "PROTOCOL_ECDH" in str(excinfo.value)

    def test_curve_accepts_short_and_full_forms(self):
        assert normalize_curve("sm2") == "CURVE_SM2"
        assert normalize_curve("CURVE_25519") == "CURVE_25519"

    def test_unknown_curve_raises(self):
        with pytest.raises(ValueError):
            normalize_curve("CURVE_NOT_A_CURVE")

    def test_ecdh_family_requires_curve(self):
        """实测：ECDH 未给 curve 会 RuntimeError: Curve type is not specified."""

        assert protocol_needs_curve(PSI_DEFAULT_PROTOCOL) is True
        assert protocol_needs_curve("PROTOCOL_KKRT") is False
        assert protocol_needs_curve("PROTOCOL_RR22") is False

    def test_defaults_are_available_in_this_environment(self):
        """默认协议/曲线必须真的在当前 spu 里存在，否则默认值无意义。"""

        report = check_psi_capabilities()
        if not report.installed:
            pytest.skip("未安装 spu")
        assert PSI_DEFAULT_PROTOCOL in report.protocols
        assert PSI_DEFAULT_CURVE in report.curves

    def test_default_curve_is_a_chinese_standard_curve(self):
        """涉密测绘场景优先国密：默认走 SM2 而不是 25519。"""

        assert PSI_DEFAULT_CURVE == "CURVE_SM2"


# --------------------------------------------------------------------------
# 2.1 枚举里有 ≠ 当前可执行（参与方数量约束）
# --------------------------------------------------------------------------


class TestProtocolWorldSize:
    """回归：`PROTOCOL_ECDH_3PC` 在官方枚举中存在，但要求 3 个参与方。

    本后端的进程内链路固定两方，因此它**必然失败**。实测失败形态是
    libpsi 的 C++ 栈回溯，而不是可读原因：

        [Enforce fail at external/psi~/psi/legacy/memory_psi.cc:44]
        lctx_->WorldSize() == 3. psi_type:4, only three parties supported, got 2

    早期实现把它与两方协议并列在"可用协议"清单里，使用者照单配置就会
    撞上这段栈。这里把"枚举里有"与"这里能跑"分开，并把两者都作为契约。
    """

    def test_every_advertised_protocol_has_a_declared_world_size(self):
        """不许有漏登记：漏了就会默认按两方放行，缺陷会静默复现。"""

        missing = [p for p in PSI_PROTOCOLS if p not in PSI_PROTOCOL_WORLD_SIZE]
        assert missing == [], f"以下协议未登记参与方数量：{missing}"

    def test_three_party_protocol_is_declared_as_three(self):
        assert protocol_world_size("PROTOCOL_ECDH_3PC") == 3
        assert protocol_world_size("ecdh_3pc") == 3

    def test_other_protocols_are_two_party(self):
        for name in PSI_PROTOCOLS:
            if name == "PROTOCOL_ECDH_3PC":
                continue
            assert protocol_world_size(name) == 2, name

    def test_runnable_list_excludes_the_three_party_protocol(self):
        runnable = protocols_runnable_here()
        assert "PROTOCOL_ECDH_3PC" not in runnable
        assert set(runnable) == set(PSI_PROTOCOLS) - {"PROTOCOL_ECDH_3PC"}

    def test_runtime_world_size_is_two(self):
        assert PSI_RUNTIME_WORLD_SIZE == 2

    def test_capability_report_discloses_the_unrunnable_protocol(self):
        """能力报告必须自己说清楚哪个协议在这条链路上跑不了。"""

        report = check_psi_capabilities()
        if not report.installed:
            pytest.skip("未安装 spu")
        joined = " ".join(report.notes)
        assert "PROTOCOL_ECDH_3PC" in joined, "未披露三方协议不可执行"

    def test_three_party_protocol_is_refused_with_a_readable_reason(self):
        """拒绝必须发生在进入协议之前：给可读原因，而不是 C++ 栈。"""

        run = run_psi_intersection(
            [11, 22, 33], [22, 33, 44],
            op="Intersects", protocol="PROTOCOL_ECDH_3PC", curve="CURVE_SM2",
        )
        assert run.status == "error"
        assert "3" in run.error and "参与方" in run.error
        # 不得把 C++ 栈当作原因抛给使用者
        assert "Enforce fail" not in run.error
        assert "Stacktrace" not in run.error
        # 且必须给出替代方案
        assert "PROTOCOL_ECDH" in run.error

    @needs_psi
    def test_every_runnable_protocol_actually_runs(self):
        """把"可执行清单"变成可验证契约：清单里的每个协议都要真能跑。

        没有这条，`protocols_runnable_here()` 会随实现漂移而失去意义。

        **"能跑"与"结果精确"是两件事**：`PROTOCOL_DP` 是差分隐私 PSI，交集里会
        注入假元素/丢弃真元素，故它只参与"状态为 ok"的断言，不参与"结果必须与
        明文一致"的断言。实测该协议在真交集非空时也会给出 False（见
        `PSI_PROTOCOLS_WITH_NOISE`）；把它一起断言，得到的是一条**随机失败**的
        用例，而它掩盖的是语义差异，不是缺陷。
        """

        for name in protocols_runnable_here():
            curve = "CURVE_SM2" if protocol_needs_curve(name) else None
            run = run_psi_intersection(
                ROUTE, ZONE, op="Intersects", protocol=name, curve=curve,
            )
            assert run.status == "ok", f"{name}: {run.status} {run.error}"
            assert isinstance(run.value, bool), f"{name}: {run.value!r}"
            if protocol_is_exact(name):
                assert run.value is True, name


# --------------------------------------------------------------------------
# 2.2 能跑 ≠ 结果精确：差分隐私协议
# --------------------------------------------------------------------------


class TestProtocolNoise:
    """回归：`PROTOCOL_DP` 是差分隐私 PSI，结果**带噪**。

    它曾在"可执行协议逐个跑一遍并断言与明文一致"的用例里造成约 2/12 的随机失败。
    根因不是执行缺陷，而是把"能跑"和"结果精确"混成了一件事——与
    `PROTOCOL_ECDH_3PC` 那类"枚举里有、这里跑不了"同属披露缺口，
    只不过这次漏掉的是"跑得起来但结果不可信"。

    依据（上游 `psi/legacy/dp_psi/dp_psi.h`）：
        DpPsiOptions(bob_p=0.9, epsilon=3.0)
        p2 = e^ε / (1 + e^ε) ≈ 0.953   Alice 子采样
        q  = 1 - p2           ≈ 0.047   Alice 上采样（往交集中注入假元素）
    """

    #: 一个**不可能**等于任何业务结果的参考值。带噪协议的结果只会是布尔或元组，
    #: 都不可能等于这个字符串，因此 "agreement=False" 这条路径可以被**确定性**地
    #: 走到。若改用布尔参考值，DP 的噪声偶尔会恰好与它相等——实测那样写，
    #: 14 轮整套用例里有 5 轮假失败。概率性的用例等于没有覆盖。
    _NEVER_MATCHES = "（故意构造的不匹配参考值）"

    def test_noise_registry_names_dp(self):
        assert PSI_PROTOCOLS_WITH_NOISE == ("PROTOCOL_DP",)

    def test_noise_registry_only_lists_real_protocols(self):
        unknown = [n for n in PSI_PROTOCOLS_WITH_NOISE if n not in PSI_PROTOCOLS]
        assert unknown == [], f"带噪清单里出现未登记协议：{unknown}"

    def test_is_exact_answers_the_precision_question(self):
        assert protocol_is_exact("PROTOCOL_DP") is False
        for name in (
            "PROTOCOL_ECDH",
            "PROTOCOL_KKRT",
            "PROTOCOL_RR22",
            "PROTOCOL_ECDH_NPC",
            "PROTOCOL_KKRT_NPC",
        ):
            assert protocol_is_exact(name) is True, name

    def test_is_exact_accepts_the_short_form(self):
        assert protocol_is_exact("dp") is False

    def test_capability_report_discloses_the_noisy_protocol(self):
        """`psi-check` 必须自己说出哪个协议结果带噪，不能只等人踩坑。"""

        report = check_psi_capabilities()
        if not report.installed:
            pytest.skip("未安装 spu")
        joined = " ".join(report.notes)
        assert "PROTOCOL_DP" in joined, "未披露带噪协议"
        assert "带噪" in joined

    def test_runnable_hint_marks_the_noisy_protocol(self):
        """替代清单里不能把 DP 摆成 ECDH 的等价替代。"""

        hint = runnable_protocols_hint()
        assert "PROTOCOL_DP*" in hint
        assert "带 * 者" in hint

    @needs_psi
    def test_noisy_protocol_runs_and_carries_the_disclosure(self):
        run = run_psi_intersection(
            ROUTE, ZONE, op="Intersects", protocol="PROTOCOL_DP",
        )
        assert run.status == "ok", run.error
        assert isinstance(run.value, bool)
        assert any("带噪" in note for note in run.notes), run.notes

    @needs_psi
    def test_noisy_protocol_mismatch_is_not_an_execution_error(self):
        """带噪协议与明文不一致是**协议语义**，不得升级成 error。

        升级会让一次正常执行被随机报成失败（实测 40 次里 1 次），而且错误消息会
        把"协议本就带噪"说成"PSI 结果与明文不一致"——把设计行为报成缺陷。

        这里用"故意给错的参考值"把不一致**确定性地**造出来：若靠等真实噪声来
        复现，用例本身就成了概率性的——缺陷跑不掉的时候它也就抓不住。
        """

        run = run_psi_intersection(
            ROUTE, ZONE, op="Intersects", protocol="PROTOCOL_DP",
            reference_fn=lambda left, right: self._NEVER_MATCHES,
        )
        assert run.status == "ok", run.error
        assert run.agreement is False
        assert run.reference == self._NEVER_MATCHES
        assert run.error is None

    @needs_psi
    def test_exact_protocol_mismatch_is_still_an_error(self):
        """对照组：精确协议与明文不一致**必须**仍然是 error。

        没有这条，"带噪不升级"就可能被误改成"一律不升级"，真出缺陷时反而静默。
        """

        run = run_psi_intersection(
            ROUTE, ZONE, op="Intersects", protocol="PROTOCOL_ECDH",
            curve="CURVE_SM2",
            reference_fn=lambda left, right: self._NEVER_MATCHES,
        )
        assert run.status == "error"
        assert run.agreement is False
        assert "与明文不一致" in (run.error or "")


# --------------------------------------------------------------------------
# 2.3 协议与椭圆曲线的关系：三分类，且只说自己做过的事
# --------------------------------------------------------------------------


class TestCurveRelation:
    """回归：`PROTOCOL_DP` 曾被登记为"不读曲线"，CLI 据此告诉使用者
    "--psi-curve 不生效"。

    源码核对后这是**不成立的断言**：上游
    `RunDpEcdhPsiAlice/Bob(..., CurveType curve = CurveType::CURVE_25519)`
    说明 DP 是 ECDH 系协议、曲线是它的形参。本项目的口径因此改成三分类，
    并且只陈述自己做过的事——本后端没把曲线传进 DP，所以不说"不生效"。
    """

    def test_relation_table_covers_every_protocol(self):
        missing = [n for n in PSI_PROTOCOLS if n not in PSI_CURVE_RELATION]
        assert missing == [], f"以下协议未登记曲线关系：{missing}"

    def test_relation_values_come_from_a_closed_set(self):
        allowed = {"required", "ignored", "implicit"}
        bad = {n: r for n, r in PSI_CURVE_RELATION.items() if r not in allowed}
        assert bad == {}, f"未知关系取值：{bad}"

    def test_dp_is_implicit_not_ignored(self):
        assert psi_curve_relation("PROTOCOL_DP") == "implicit"
        assert "PROTOCOL_DP" not in PSI_PROTOCOLS_WITHOUT_CURVE
        assert PSI_PROTOCOLS_WITHOUT_CURVE == (
            "PROTOCOL_KKRT",
            "PROTOCOL_RR22",
            "PROTOCOL_KKRT_NPC",
        )

    def test_dp_is_not_declared_curve_free(self):
        """DP 不**要求**曲线（自带默认），因此 needs_curve 为 False；
        但这与"不读曲线"是两回事，不能合并成一个判据。"""

        assert protocol_needs_curve("PROTOCOL_DP") is False
        assert psi_curve_relation("PROTOCOL_DP") == "implicit"

    def test_needs_curve_matches_the_required_relation(self):
        """两个入口不许漂移：needs_curve 必须恰好等于 relation == "required"。"""

        for name in PSI_PROTOCOLS:
            assert protocol_needs_curve(name) is (
                psi_curve_relation(name) == "required"
            ), name

    def test_curve_required_list_is_the_ecdh_family(self):
        assert PSI_PROTOCOLS_CURVE_REQUIRED == (
            "PROTOCOL_ECDH",
            "PROTOCOL_ECDH_3PC",
            "PROTOCOL_ECDH_NPC",
        )

    def test_unregistered_protocol_falls_back_to_required(self, monkeypatch):
        """防漏登记：将来加了协议却忘了进表时，按"必须给曲线"处理而不是静默放行。"""

        from backends.psi_backend import capability as capability_module

        reduced = {
            key: value
            for key, value in PSI_CURVE_RELATION.items()
            if key != "PROTOCOL_KKRT"
        }
        monkeypatch.setattr(capability_module, "PSI_CURVE_RELATION", reduced)
        assert capability_module.psi_curve_relation("PROTOCOL_KKRT") == "required"


# --------------------------------------------------------------------------
# 3. 泄漏面必须如实登记
# --------------------------------------------------------------------------


class TestLeakDisclosure:
    def test_every_psi_op_has_a_leak_entry(self):
        for op in PSI_OPS:
            assert op in PSI_OP_LEAKS, f"{op} 未登记泄漏面"

    def test_leak_entries_are_specific_not_boilerplate(self):
        texts = [PSI_OP_LEAKS[op] for op in PSI_OPS]
        assert len(set(texts)) == len(texts), "泄漏面描述不允许复制粘贴"

    @needs_psi
    def test_run_result_carries_leak_disclosure(self):
        run = run_psi_intersection(ROUTE, ZONE, op="Intersects")
        assert run.reveals == PSI_OP_LEAKS["Intersects"]
        assert run.semantics

    def test_intersects_discloses_that_it_reveals_more_than_a_bool(self):
        """Intersects 业务上只要布尔，但 PSI 会把交集本体交出去——必须写明。"""

        text = PSI_OP_LEAKS["Intersects"]
        assert "交集本体" in text

    def test_contains_leak_entry_points_at_the_subset_disclosure(self):
        """`Contains` 的静态表只描述核心，子集判定的形态是独立披露面。

        这里刻意**禁止**静态表出现"明文比较"：默认路径已经是 MPC 基数等值
        （见 backends.psi_backend.subset_mpc）。把旧结论留在静态表里，
        等于用一句过期的话去换一张好看的表。
        """

        text = PSI_OP_LEAKS["Contains"]
        assert "交集本体" in text
        assert "subset" in text
        assert "明文比较" not in text


# --------------------------------------------------------------------------
# 4. 真实求交（三算子）
# --------------------------------------------------------------------------


@needs_psi
class TestRealPsiIntersection:
    def test_intersects_matches_plaintext(self):
        run = run_psi_intersection(
            ROUTE, ZONE, op="Intersects",
            reference_fn=lambda l, r: plain.plain_intersects(l, r).value,
        )
        assert run.status == "ok"
        assert run.value is True
        assert run.agreement is True
        assert run.reference is True

    def test_cellset_intersect_returns_the_intersection_body(self):
        run = run_psi_intersection(
            ROUTE, ZONE, op="CellSetIntersect",
            reference_fn=lambda l, r: plain.plain_cellset_intersect(l, r).value,
        )
        assert run.status == "ok"
        assert run.value == EXPECTED_INTERSECTION
        assert run.agreement is True

    def test_psi_intersection_count_matches_the_grid_codes(self):
        """交集基数必须等于手工算出的真值，不能只看布尔。"""

        run = run_psi_intersection(ROUTE, ZONE, op="CellSetIntersect")
        assert run.intersection_count == len(EXPECTED_INTERSECTION) == 2
        assert run.original_count == len(ROUTE) == 3

    def test_contains_true_direction(self):
        """Contains(outer=ZONE, inner=ROUTE∩ZONE) 应为 True。"""

        outer, inner = ZONE, EXPECTED_INTERSECTION
        run = run_psi_intersection(
            outer, inner, op="Contains",
            reference_fn=lambda l, r: plain.plain_contains(l, r).value,
        )
        assert run.status == "ok"
        assert run.value is True
        assert run.agreement is True

    def test_contains_false_when_inner_is_larger(self):
        outer = EXPECTED_INTERSECTION
        inner = ZONE
        run = run_psi_intersection(
            outer, inner, op="Contains",
            reference_fn=lambda l, r: plain.plain_contains(l, r).value,
        )
        assert run.status == "ok"
        assert run.value is False
        assert run.agreement is True

    def test_contains_direction_matches_plain_on_both_orientations(self):
        """回归：旧实现写成 `outer <= 交集`，方向与集合都取反。

        它实际在问 "inner ⊇ outer"，会把 True 判成 False。这里两个方向都钉死。
        """

        cases = [
            (ZONE, EXPECTED_INTERSECTION),
            (EXPECTED_INTERSECTION, ZONE),
            (ROUTE, ROUTE),
            (ROUTE, ZONE),
        ]
        for outer, inner in cases:
            expected = plain.plain_contains(outer, inner).value
            run = run_psi_intersection(outer, inner, op="Contains")
            assert run.value == expected, (
                f"Contains(outer={outer}, inner={inner}) "
                f"PSI={run.value} 明文={expected}"
            )

    def test_contains_does_not_include_itself_when_disjoint(self):
        expected = plain.plain_contains(ROUTE, ZONE).value
        run = run_psi_intersection(ROUTE, ZONE, op="Contains")
        assert expected is False
        assert run.value is False

    def test_all_protocols_agree_on_the_same_input(self):
        """跨协议一致性：结果不允许随协议变化。"""

        observed = {}
        for protocol, curve in (
            ("PROTOCOL_ECDH", "CURVE_SM2"),
            ("PROTOCOL_ECDH", "CURVE_25519"),
            ("PROTOCOL_KKRT", None),
            ("PROTOCOL_RR22", None),
        ):
            run = run_psi_intersection(
                ROUTE, ZONE, op="CellSetIntersect",
                protocol=protocol, curve=curve,
            )
            assert run.status == "ok", run.error
            observed[protocol + "/" + str(curve)] = run.value
        assert len(set(observed.values())) == 1, observed

    def test_high_bit_grid_codes_are_not_truncated(self):
        """回归：64 位码可能 >= 2**63；CSV 通道不得被当成有符号整数截断。"""

        high = _code(x=131071)
        assert high >= 2 ** 63
        run = run_psi_intersection([high], [high], op="CellSetIntersect")
        assert run.status == "ok", run.error
        assert run.value == (high,)

    def test_run_psi_operation_dispatches_named_arguments(self):
        run = run_psi_operation("Intersects", {"route": ROUTE, "no_fly_zone": ZONE})
        assert run.status == "ok"
        assert run.value is True

    def test_receiver_rank_may_be_either_party(self):
        run = run_psi_intersection(ROUTE, ZONE, op="CellSetIntersect", receiver_rank=1)
        assert run.status == "ok"
        assert run.value == EXPECTED_INTERSECTION


# --------------------------------------------------------------------------
# 5. 空输入：显式判定，不以协议异常收场
# --------------------------------------------------------------------------


@needs_psi
class TestEmptyInput:
    """回归：官方 PSI 的 CSV 读取要求至少"表头 + 1 行"。

    空集合会抛 `Enforce fail at arrow_helper.cc:87 ... read csv file second line`。
    那是输入枚举口径问题，不是隐私计算失败，必须前置判掉。
    """

    def test_empty_right_is_resolved_without_protocol_error(self):
        run = run_psi_intersection(ROUTE, [], op="Intersects")
        assert run.status == "empty-input"
        assert run.value is False
        assert "second line" not in (run.error or "")

    def test_empty_left_is_resolved_without_protocol_error(self):
        run = run_psi_intersection([], ROUTE, op="Intersects")
        assert run.status == "empty-input"
        assert run.value is False

    def test_cellset_intersect_with_empty_side_returns_empty_body(self):
        run = run_psi_intersection(ROUTE, [], op="CellSetIntersect")
        assert run.status == "empty-input"
        assert run.value == ()

    def test_contains_with_empty_sets_follows_plaintext_convention(self):
        """∅ ⊆ 任意集合为 True；这必须与明文口径一致，不能自设特例。"""

        for outer, inner in (([], []), ([_code(1)], []), (ROUTE, [])):
            expected = plain.plain_contains(outer, inner).value
            run = run_psi_intersection(outer, inner, op="Contains")
            assert run.value == expected, (
                f"outer={outer} inner={inner} PSI={run.value} 明文={expected}"
            )

    def test_empty_outer_is_not_a_contains(self):
        run = run_psi_intersection([], ROUTE, op="Contains")
        assert run.status == "empty-input"
        assert run.value is False

    def test_empty_input_is_disclosed_in_notes(self):
        run = run_psi_intersection(ROUTE, [], op="Intersects")
        assert any("空集合" in note for note in run.notes)


# --------------------------------------------------------------------------
# 6. 诚实留空：环境不具备时绝不给推测值
# --------------------------------------------------------------------------


class TestNeverFakesValues:
    def _blocked_report(self) -> PsiCapabilityReport:
        report = PsiCapabilityReport()
        report.installed = True
        report.version = "0.0.0-fake"
        report.runnable = False
        report.blockers = ("人为构造的阻断项",)
        return report

    def test_unavailable_environment_leaves_value_empty(self):
        run = run_psi_intersection(
            ROUTE, ZONE, op="Intersects", report=self._blocked_report()
        )
        assert run.status == "unavailable"
        assert run.value is None
        assert run.intersection == ()
        assert run.intersection_count is None
        assert run.blockers == ("人为构造的阻断项",)

    def test_unavailable_environment_still_explains_itself(self):
        run = run_psi_intersection(
            ROUTE, ZONE, op="CellSetIntersect", report=self._blocked_report()
        )
        assert run.error is None, "环境缺失不是执行错误，不应记成 error"
        assert any("留有" in n or "空缺" in n for n in run.notes)

    def test_unavailable_never_claims_agreement(self):
        run = run_psi_intersection(
            ROUTE, ZONE, op="Intersects",
            report=self._blocked_report(),
            reference_fn=lambda l, r: True,
        )
        assert run.agreement is None

    def test_unknown_operator_is_rejected_explicitly(self):
        run = run_psi_intersection(ROUTE, ZONE, op="NotAnOperator")
        assert run.status == "error"
        assert "未登记" in (run.error or "")

    def test_bad_receiver_rank_is_rejected(self):
        run = run_psi_intersection(ROUTE, ZONE, op="Intersects", receiver_rank=5)
        assert run.status == "error"
        assert "receiver_rank" in (run.error or "")

    def test_run_psi_operation_rejects_non_psi_operator(self):
        with pytest.raises(ValueError):
            run_psi_operation("WeightedSum", {"values": [1], "weights": [1]})


# --------------------------------------------------------------------------
# 6b. 原生日志：默认静默，且不在工作目录落文件
# --------------------------------------------------------------------------


@needs_psi
class TestNativeLogHygiene:
    @pytest.mark.parametrize("quiet", [True, False])
    def test_no_spu_log_is_written_into_cwd(self, tmp_path, monkeypatch, quiet):
        """回归：`libspu.logging` 的 `system_log_path` 默认是相对路径 'spu.log'。

        原生库会在**当前工作目录**落一个 spu.log——编译器不该往用户的项目里丢文件。
        静默与详谈两种模式都必须干净。
        """

        monkeypatch.chdir(tmp_path)
        run_psi_intersection([1, 2, 3], [2, 3, 4], op="Intersects", quiet=quiet)
        leftovers = [p.name for p in tmp_path.iterdir()]
        assert leftovers == [], f"工作目录被写入文件：{leftovers}"

    def test_quiet_is_the_default(self):
        """默认不把官方协商日志倾泻到 stderr。"""

        import inspect

        signature = inspect.signature(run_psi_intersection)
        assert signature.parameters["quiet"].default is True

    def test_loud_mode_still_runs_correctly(self):
        """打开日志不能影响正确性。"""

        run = run_psi_intersection(ROUTE, ZONE, op="Intersects", quiet=False)
        assert run.status == "ok"
        assert run.value is True


# --------------------------------------------------------------------------
# 7. 与明文的一致性由协议无关地保持
# --------------------------------------------------------------------------


@needs_psi
class TestAgreementDiscipline:
    def test_mismatch_with_plaintext_is_reported_as_error(self):
        """如果 PSI 与明文不一致，必须报错而不是静默返回。"""

        run = run_psi_intersection(
            ROUTE, ZONE, op="Intersects",
            reference_fn=lambda l, r: False if True else None,  # 故意给错
        )
        assert run.value is True
        assert run.agreement is False
        assert run.status == "error"
        assert "不一致" in (run.error or "")

    def test_agreement_is_none_when_no_reference_supplied(self):
        run = run_psi_intersection(ROUTE, ZONE, op="Intersects")
        assert run.status == "ok"
        assert run.agreement is None
        assert run.reference is None

    def test_reference_exception_does_not_break_the_run(self):
        def boom(l, r):
            raise RuntimeError("参考实现故意抛错")

        run = run_psi_intersection(ROUTE, ZONE, op="Intersects", reference_fn=boom)
        assert run.status == "ok"
        assert run.value is True
        assert run.reference is None


# --------------------------------------------------------------------------
# 9. RR22：协议声明 + 参数能力 + 真实执行（课题 §18 / §19）
# --------------------------------------------------------------------------


class TestRr22ProtocolDeclaration:
    """RR22 是协议清单里的正式成员：两方、精确、不读 curve。"""

    def test_rr22_is_registered(self):
        assert "PROTOCOL_RR22" in PSI_PROTOCOLS

    def test_rr22_is_two_party(self):
        assert protocol_world_size("PROTOCOL_RR22") == 2

    def test_rr22_needs_no_curve(self):
        assert protocol_needs_curve("PROTOCOL_RR22") is False
        assert psi_curve_relation("PROTOCOL_RR22") == "ignored"
        assert protocol_is_exact("PROTOCOL_RR22") is True


@needs_psi
class TestRr22ParamsAndExecution:
    """参数必须真的进 Rr22Rarams 并跑出与明文一致的结果，而非登记在纸上。"""

    def test_rr22_param_capability_probe_matches_the_installed_spu(self):
        from spu import psi

        report = check_psi_capabilities()
        probe = report.rr22_params
        assert probe["protocol"] == "PROTOCOL_RR22"
        assert probe["protocol_present"] is True
        assert "rr22_params" in report.to_dict()
        if not hasattr(psi, "Rr22Rarams"):
            # 诚实分支：本版本没有参数类 → 探测必须如实置 False，不许伪装
            assert probe["runnable"] is False
            assert probe["params"]["rr22_params"] is False
            pytest.skip("当前 SPU 版本没有 Rr22Rarams，参数注入能力不完整")
        # 直接验证官方构造（§19 测试 4）
        conf = psi.PsiProtocolConfig(
            protocol=psi.PsiProtocol.PROTOCOL_RR22,
            receiver_rank=0,
            broadcast_result=False,
            rr22_params=psi.Rr22Rarams(low_comm_mode=False),
        )
        assert hasattr(conf, "rr22_params")
        assert probe["params"]["low_comm_mode"] is True
        assert probe["params"]["rr22_params"] is True
        assert probe["runnable"] is True

    def test_rr22_intersects_matches_plaintext(self):
        run = run_psi_intersection(
            ROUTE, ZONE, op="Intersects", protocol="PROTOCOL_RR22",
            reference_fn=lambda left, right: plain.plain_intersects(left, right).value,
        )
        assert run.status == "ok", run.error
        assert run.value is True
        assert run.agreement is True
        assert dict(run.protocol_params) == {"low_comm_mode": False}
        assert run.protocol_params["low_comm_mode"] is False

    def test_rr22_cellset_intersect_returns_the_intersection_body(self):
        run = run_psi_intersection(
            ROUTE, ZONE, op="CellSetIntersect", protocol="PROTOCOL_RR22",
            reference_fn=lambda left, right: tuple(sorted(set(left) & set(right))),
        )
        assert run.status == "ok", run.error
        assert run.value == EXPECTED_INTERSECTION
        assert run.agreement is True

    def test_rr22_does_not_truncate_high_bit_codes(self):
        high = _code(x=131071)
        assert high >= 2 ** 63
        run = run_psi_intersection(
            [high], [high], op="CellSetIntersect", protocol="PROTOCOL_RR22",
            reference_fn=lambda left, right: tuple(sorted(set(left) & set(right))),
        )
        assert run.status == "ok", run.error
        assert run.value == (high,)
        assert run.agreement is True

    def test_rr22_low_comm_mode_both_values_run_and_agree(self):
        results = {}
        for mode in (False, True):
            run = run_psi_intersection(
                ROUTE, ZONE, op="Intersects", protocol="PROTOCOL_RR22",
                rr22_low_comm_mode=mode,
                reference_fn=lambda left, right: plain.plain_intersects(left, right).value,
            )
            assert run.status == "ok", f"low_comm_mode={mode}: {run.error}"
            assert run.value is True
            assert run.agreement is True
            assert dict(run.protocol_params) == {"low_comm_mode": mode}
            results[mode] = run.value
        assert len(set(results.values())) == 1

    def test_rr22_param_is_not_recorded_for_other_protocols(self):
        """别的协议带出空参数档：不把"传了"写成"协议读了"。"""

        run = run_psi_intersection(
            ROUTE, ZONE, op="Intersects", protocol="PROTOCOL_KKRT",
            rr22_low_comm_mode=True,
        )
        assert run.status == "ok", run.error
        assert dict(run.protocol_params) == {}
        assert any("rr22_low_comm_mode" in note for note in run.notes)
