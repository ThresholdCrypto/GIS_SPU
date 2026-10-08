# -*- coding: utf-8 -*-
"""统一协议请求校验（Phase 3）：

    protocol → family → world_size → operation → field
    → security/result semantics → parameters

入口只有一个：`validate_protocol_request()`。

分层纪律
========
- 本模块只依赖两个后端的**协议元数据与静态校验函数**（protocol_registry /
  capability 里的纯函数），不执行任何协议、不探测环境；
- 它回答的问题是"这个显式请求在本仓库当前能力登记下是否成立"，
  不回答"现在这台机器能不能跑"（那是 capability probe / runtime 的职责）；
- Planner 在函数内延迟导入本模块（planner 不在导入期依赖 backends，保持分层
  无环）；Compiler 在构造期用它做 field / world_size 的前置拒绝；
- 协议名不存在只是"问题"之一，由调用方决定呈现形式：Planner 产出带位置/
  建议/代价的诊断，Compiler 对显式协议的请求直接抛 ValueError。

与 Runtime 的关系
=================
Runtime（run_spu_simulation / psi runtime）保持各自既有的 fail-fast 契约，
本模块不改变执行路径——只把"必然失败的显式请求"挪到编译期更早报错。

登记出处的纪律
==============
本模块自己不登记任何协议事实：world_size / field / 语义 / 参数全部读
`backends/spu_backend/protocol_registry.py` 与 `backends/psi_backend/`，
拒绝信息里给出可放行路径（补真机扫描 / 登记候选并补测试），不静默放宽。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from backends.psi_backend import protocol_registry as _psi_registry
from backends.psi_backend.capability import (
    PSI_DEFAULT_CURVE,
    PSI_RUNTIME_WORLD_SIZE,
    normalize_curve,
    normalize_psi_protocol,
    protocol_world_size,
    psi_curve_relation,
    psi_protocol_spec,
    runnable_protocols_hint,
    validate_psi_protocol_params,
)
from backends.spu_backend import protocol_registry as _mpc_registry
from backends.spu_backend.capability import normalize_field, normalize_protocol

#: 协议族清单（顺序稳定：错误信息与测试按此渲染）
PROTOCOL_FAMILIES: tuple[str, ...] = ("PSI", "MPC")


@dataclass(frozen=True)
class ProtocolValidationResult:
    """一次协议请求的静态校验结论；`ok=False` 表示**不应放行**。"""

    ok: bool
    family: str | None
    protocol: str | None
    field: str | None = None
    world_size: int | None = None
    problems: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "family": self.family,
            "protocol": self.protocol,
            "field": self.field,
            "world_size": self.world_size,
            "problems": list(self.problems),
            "notes": list(self.notes),
        }


def _normalize_in_family(family: str, protocol: str) -> str:
    if family == "PSI":
        return normalize_psi_protocol(protocol)
    return normalize_protocol(protocol)


def _family_names(family: str) -> tuple[str, ...]:
    if family == "PSI":
        return _psi_registry.PSI_PROTOCOL_NAMES
    return _mpc_registry.MPC_PROTOCOL_NAMES


def _resolve_protocol(
    norm_family: str | None, protocol: str
) -> tuple[str | None, str | None, tuple[str, ...]]:
    """解析协议名 → (family, 归一化名, problems)。

    - 显式 family：在该族里归一化；失败时若名字属于另一族，给出定向提示；
    - family=None：按名字在两个族里推断（协议名不重叠，没有歧义）。
    """

    if norm_family is not None:
        other = "MPC" if norm_family == "PSI" else "PSI"
        try:
            return norm_family, _normalize_in_family(norm_family, protocol), ()
        except ValueError as exc:
            try:
                other_name = _normalize_in_family(other, protocol)
            except ValueError:
                return norm_family, None, (str(exc),)
            return norm_family, None, (
                f"协议 {protocol!r} 属于 {other} 族（归一化名 {other_name}），"
                f"不能按 {norm_family} 校验；{norm_family} 支持："
                f"{list(_family_names(norm_family))}",
            )

    for fam in PROTOCOL_FAMILIES:
        try:
            return fam, _normalize_in_family(fam, protocol), ()
        except ValueError:
            continue
    return None, None, (
        f"未注册的协议名 {protocol!r}：既不在 PSI 支持清单"
        f"（{list(_psi_registry.PSI_PROTOCOL_NAMES)}），"
        f"也不在 MPC 支持清单（{list(_mpc_registry.MPC_PROTOCOL_NAMES)}）",
    )


def _check_world_size_value(world_size: Any, problems: list[str]) -> bool:
    """world_size 的整数闸（两个族共用）；bool 不算 int。"""

    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 2
    ):
        problems.append(f"world_size 必须是 ≥2 的整数，实得 {world_size!r}")
        return False
    return True


def _semantics_checks(
    protocol: str,
    semantics: str,
    require_exact: bool | None,
    problems: list[str],
    notes: list[str],
) -> None:
    if require_exact is True and semantics != "exact":
        problems.append(
            f"业务要求精确结果（require_exact=True），协议 {protocol} 的结果语义为 "
            f"{semantics}，不能作为与明文一致的一致性验证依据"
        )
    elif require_exact is None and semantics != "exact":
        notes.append(
            f"协议 {protocol} 结果语义为 {semantics}（带噪），"
            "不能作为与明文一致的一致性验证依据"
        )


def validate_protocol_request(
    *,
    protocol: str | None,
    family: str | None = None,
    operation: str | None = None,
    field: str | int | None = None,
    world_size: int | None = None,
    curve: str | None = None,
    protocol_params: Mapping[str, Any] | None = None,
    require_exact: bool | None = None,
) -> ProtocolValidationResult:
    """对一次**显式**协议请求做完整的静态能力校验（§六检查清单）。

    Args:
        protocol:        协议名；None = 未显式选择（由默认值/自动选择决定，
                         本层无可校验内容，直接放行并如实加注）。
        family:          "PSI" / "MPC"；None = 按协议名推断（两族命名空间不重叠）。
        operation:       可选：该协议被哪个算子使用；不属于该族算子 → 拒绝。
        field:           环宽（MPC 专用，32/64/128 或 FM32/FM64/FM128）；
                         PSI 族给 field 只加注"不适用"，不报错。
        world_size:      参与方数量；必须 ≥2 且满足协议下限。
        curve:           椭圆曲线（PSI 专用）；MPC 族给 curve 只加注"不适用"。
        protocol_params: 协议级参数（PSI：receiver_rank 等，走既有静态校验；
                         MPC：当前没有登记参数，收到非空即拒绝）。
        require_exact:   业务是否要求精确结果；True 且协议带噪 → 拒绝。

    Returns:
        ProtocolValidationResult。`ok=False` 时 `problems` 是完整的拒绝理由；
        `ok=True` 时 `notes` 是需要随行披露的事实（不生效参数 / 缺省值 / 带噪等）。
    """

    problems: list[str] = []
    notes: list[str] = []

    norm_family: str | None = None
    if family is not None:
        norm_family = str(family).upper().strip()
        if norm_family not in PROTOCOL_FAMILIES:
            return ProtocolValidationResult(
                ok=False,
                family=None,
                protocol=str(protocol) if protocol is not None else None,
                problems=(f"未知协议族 {family!r}；可选：{list(PROTOCOL_FAMILIES)}",),
            )

    if protocol is None:
        notes.append(
            "未显式指定协议：由登记默认值 / 自动选择决定；本层只校验显式请求"
        )
        return ProtocolValidationResult(
            ok=True,
            family=norm_family,
            protocol=None,
            world_size=world_size if isinstance(world_size, int) and not isinstance(world_size, bool) else None,
            notes=tuple(notes),
        )

    resolved_family, resolved_name, resolve_problems = _resolve_protocol(
        norm_family, str(protocol)
    )
    if resolve_problems:
        return ProtocolValidationResult(
            ok=False,
            family=resolved_family,
            protocol=None,
            problems=resolve_problems,
        )
    assert resolved_name is not None

    norm_field: str | None = None
    if resolved_family == "PSI":
        required = protocol_world_size(resolved_name)
        if required > PSI_RUNTIME_WORLD_SIZE:
            problems.append(
                f"协议 {resolved_name} 需要 {required} 个参与方，"
                f"本链路固定 {PSI_RUNTIME_WORLD_SIZE} 方，无法执行该协议；"
                f"请改用：{runnable_protocols_hint()}"
            )
        if world_size is not None:
            _check_world_size_value(world_size, problems)
        if field is not None:
            notes.append(
                "PSI 族不使用 SPU 环宽参数（集合元素走 CSV 键 / 64-bit grid code "
                "等值求交）；field 只对 MPC 族生效，本请求中不校验、也不生效"
            )
        if operation is not None and operation not in _psi_registry.PSI_CANDIDATE_OPS:
            problems.append(
                f"算子 {operation} 不经由 PSI 协议后端；协议 {resolved_name} 不会被它使用"
            )
        if curve is not None:
            try:
                curve_name = normalize_curve(curve)
            except ValueError as exc:
                problems.append(str(exc))
            else:
                relation = psi_curve_relation(resolved_name)
                if relation == "ignored":
                    notes.append(
                        f"协议 {resolved_name} 不基于椭圆曲线（curve_relation=ignored），"
                        f"curve={curve_name} 不会生效"
                    )
                elif relation == "implicit":
                    notes.append(
                        f"协议 {resolved_name} 自带内置默认曲线（curve_relation=implicit），"
                        f"本项目不覆盖传入的 curve={curve_name}"
                    )
        elif psi_curve_relation(resolved_name) == "required":
            notes.append(
                f"未显式指定曲线：协议 {resolved_name} 需要曲线，"
                f"缺省将使用 {PSI_DEFAULT_CURVE}"
            )
        param_check = validate_psi_protocol_params(resolved_name, protocol_params)
        problems.extend(param_check.problems)
        notes.extend(param_check.notes)
        _semantics_checks(
            resolved_name,
            psi_protocol_spec(resolved_name).result_semantics,
            require_exact,
            problems,
            notes,
        )
    else:
        spec = _mpc_registry.get_mpc_protocol_spec(resolved_name)
        if world_size is not None and _check_world_size_value(world_size, problems):
            if world_size < spec.world_size:
                alternatives = ", ".join(
                    name
                    for name, other in _mpc_registry.MPC_PROTOCOL_SPECS.items()
                    if other.world_size <= world_size
                )
                problems.append(
                    f"协议 {spec.name} 至少需要 {spec.world_size} 个参与方，"
                    f"world_size={world_size} 不满足；"
                    f"world_size={world_size} 下登记的协议：{alternatives}"
                )
        if field is not None:
            try:
                norm_field = normalize_field(field)
            except ValueError as exc:
                problems.append(str(exc))
            else:
                if norm_field not in spec.supported_fields:
                    problems.append(
                        f"协议 {spec.name} 未登记环宽 {norm_field}"
                        f"（本仓库实测支持：{', '.join(spec.supported_fields)}）；"
                        "如确需使用，请先补真机扫描（tests/test_spu_backend.py 的"
                        "协议×环宽矩阵）并更新 protocol_registry"
                    )
        if operation is not None and operation not in _mpc_registry.MPC_CANDIDATE_OPS:
            problems.append(
                f"算子 {operation} 不经由 SPU/MPC 协议后端；协议 {spec.name} 不会被它使用"
            )
        if curve is not None:
            notes.append(
                "MPC 族不使用椭圆曲线参数（curve 为 PSI 专属；SPU RuntimeConfig "
                "不接收曲线）"
            )
        params = dict(protocol_params or {})
        if params:
            problems.append(
                f"MPC 族当前没有登记协议参数（params_schema 为空）；收到未登记参数 "
                f"{sorted(params)}，不予放行（若确需新参数，请先登记并补测试）"
            )
        _semantics_checks(
            spec.name, spec.result_semantics, require_exact, problems, notes
        )

    return ProtocolValidationResult(
        ok=not problems,
        family=resolved_family,
        protocol=resolved_name,
        field=norm_field,
        world_size=(
            world_size
            if isinstance(world_size, int) and not isinstance(world_size, bool)
            else None
        ),
        problems=tuple(problems),
        notes=tuple(notes),
    )