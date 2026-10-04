# -*- coding: utf-8 -*-
"""真实格网集合输入适配层：CSV / JSON / CellSet / 64-bit 码列表 → 统一句柄。

背景（下一阶段任务 §3）
---------------------
编译器早期的 PSI 模拟只吃 `geosecure.compiler.DEFAULT_EXAMPLE_INPUTS` 里的
固定样例。那是"固定样例上的真实执行"，不是"用户真实数据绑定后的执行"。
本模块把外部数据转换为两方求交真正需要的形态：

    route.csv / route.json / CellSet / [64-bit codes]
        → ResolvedCellInput（codes + 可选布局清单 + 来源标注 + 参与方标签）

并负责跨方**布局握手**：两方都声明了布局清单且不一致时，PSI 之前就拒绝
（LAYOUT_MISMATCH）。CSV 通道本身不会发现布局差异，只会得到一个形式上合法
的错误交集——这条检查把"人工核对 README"换成"机器核对"。

输入格式
--------
CSV：至少含表头；键列名 `grid_code`（缺失时取第一列）；空单元格 / NULL 跳过；
     其余必须是 [0, 2^64) 的整数。
JSON：`[1, 2, 3]` 或
      {"grid_codes": [...], "layout": {...} 或 "manifest": {...},
       "label": "...", "party_id": "..."}；
      字符串形式的整数允许 "0x..." 十六进制写法。
CellSet（geo_privacy.core）：直接取 codes。
码序列（Iterable[int]）：逐项校验后使用。
"""

from __future__ import annotations

import csv
import json
import operator
import os
from dataclasses import dataclass, replace as _replace
from typing import Any, Iterable, Mapping, Sequence

from geo_privacy.core import CellSet as PlainCellSet

_KEY_COLUMN = "grid_code"
_JSON_CODE_KEYS = ("grid_codes", "codes")
_JSON_LAYOUT_KEYS = ("layout", "manifest")

#: 布局清单参与核对的字段（与 ir.grid_code_layout_manifest() 的键一一对应）
LAYOUT_COMPARE_KEYS: tuple[str, ...] = (
    "layout_id",
    "version",
    "bits",
    "x_bits",
    "y_bits",
    "z_bits",
    "level_bits",
    "toff_bits",
    "lt_bits",
)


@dataclass(frozen=True)
class PartyInput:
    """一方的一份数据：谁（party_id）、叫什么（name）、内容是什么（data）。

    两方求交里 left/right 的角色仍由算子的输入顺序决定；party_id 是"这份数据
    属于哪个参与方"的语义标签（uav_operator / airspace_authority / …），供后续
    多参与方扩展与审计使用，不改变当前两方执行的装配方式。
    """

    party_id: str
    name: str
    data: Any
    layout: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ResolvedCellInput:
    """一份完成校验的格网集合输入。"""

    name: str
    codes: tuple[int, ...]
    source: str = ""
    party_id: str | None = None
    label: str | None = None
    layout: Mapping[str, Any] | None = None
    duplicates: int = 0
    notes: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return len(self.codes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "source": self.source,
            "party_id": self.party_id,
            "label": self.label,
            "layout_id": (self.layout or {}).get("layout_id"),
            "duplicates": self.duplicates,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class LayoutAgreement:
    """两方布局清单的核对结果。

    - agreement=True/False：双方都声明了布局，逐字段核对后的一致 / 不一致；
    - agreement=None：只有一方或双方都没声明，**无法核对**——如实记为不确定，
      绝不假装通过。
    """

    agreement: bool | None
    left_id: str | None = None
    right_id: str | None = None
    mismatches: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "agreement": self.agreement,
            "left_layout_id": self.left_id,
            "right_layout_id": self.right_id,
            "mismatches": list(self.mismatches),
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# 基础校验
# --------------------------------------------------------------------------


def _validate_code(code: Any, *, where: str) -> int:
    """把一个格网码规范为 [0, 2^64) 的 int；失败给出具体位置。"""

    if isinstance(code, bool):
        raise ValueError(f"{where}: 布尔值不是合法格网码")
    if isinstance(code, str):
        text = code.strip()
        try:
            value = int(text, 16) if text.lower().startswith("0x") else int(text)
        except ValueError as exc:
            raise ValueError(f"{where}: 无法解析格网码 {code!r}") from exc
    else:
        try:
            value = int(operator.index(code))
        except TypeError as exc:
            raise TypeError(
                f"{where}: 格网码必须是整数，实得 {type(code).__name__}"
            ) from exc
    if not 0 <= value < 2 ** 64:
        raise ValueError(f"{where}: 格网码必须在 [0, 2^64) 内，实得 {value}")
    return value


def validate_grid_code(code: Any, *, where: str = "<grid_code>") -> int:
    """公开的格网码校验入口。

    与 `_validate_code` 同一实现：解析层（CSV / JSON / 代码序列）与
    Geo-RR22 预处理（geosot_optimizer）共用一套规则，避免"两处校验、
    两套口径"的漂移。
    """

    return _validate_code(code, where=where)


def _finalize(
    codes: Sequence[int],
    *,
    name: str,
    source: str,
    layout: Mapping[str, Any] | None = None,
    label: str | None = None,
    party_id: str | None = None,
) -> ResolvedCellInput:
    codes = tuple(codes)
    duplicates = len(codes) - len(set(codes))
    notes: list[str] = []
    if duplicates:
        notes.append(
            f"输入含 {duplicates} 个重复格网码：本层原样保留"
            "（排序/去重属 Geo-RR22 预处理阶段，不在适配层做）"
        )
    if layout is not None and layout.get("layout_id") is None:
        notes.append("布局清单缺少 layout_id：只能逐字段核对，无法靠 id 快速断言")
    return ResolvedCellInput(
        name=name,
        codes=codes,
        source=source,
        party_id=party_id,
        label=label,
        layout=dict(layout) if layout is not None else None,
        duplicates=duplicates,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# 各输入形态
# --------------------------------------------------------------------------


def _load_csv(path: str, *, name: str) -> ResolvedCellInput:
    if not os.path.exists(path):
        raise FileNotFoundError(f"输入文件不存在：{path}")
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"CSV 无数据：{path}（连表头都没有）")
    header = [cell.strip().strip('"') for cell in rows[0]]
    header_has_key = _KEY_COLUMN in header
    try:
        idx = header.index(_KEY_COLUMN)
    except ValueError:
        idx = 0
    codes: list[int] = []
    for line_no, row in enumerate(rows[1:], start=2):
        if not row or idx >= len(row):
            continue
        cell = row[idx].strip().strip('"')
        if not cell or cell.upper() == "NULL":
            continue
        codes.append(_validate_code(cell, where=f"{path}:{line_no}"))
    resolved = _finalize(codes, name=name, source=f"csv:{path}")
    if not header_has_key:
        resolved = _replace(
            resolved,
            notes=resolved.notes
            + (
                f"CSV 表头未找到 {_KEY_COLUMN} 列：按第一列取值；"
                "若文件没有表头，首行会被当作表头——请补表头以免丢码",
            ),
        )
    if len(rows) == 1:
        resolved = _replace(
            resolved,
            notes=resolved.notes
            + ("CSV 只有表头：空集合（执行时按集合论直接给结果，不启动协议）",),
        )
    return resolved


def _from_payload(
    payload: Any, *, name: str, source: str
) -> ResolvedCellInput:
    layout: Mapping[str, Any] | None = None
    label: str | None = None
    party_id: str | None = None

    if isinstance(payload, list):
        raw: Any = payload
    elif isinstance(payload, Mapping):
        raw = None
        for key in _JSON_CODE_KEYS:
            if key in payload:
                raw = payload[key]
                break
        if raw is None:
            raise ValueError(
                f"{source}: JSON 对象缺少 {'/'.join(_JSON_CODE_KEYS)} 字段"
            )
        for key in _JSON_LAYOUT_KEYS:
            if key in payload:
                candidate = payload[key]
                if not isinstance(candidate, Mapping):
                    raise ValueError(f"{source}: 布局清单必须是 JSON 对象")
                layout = dict(candidate)
                break
        label = payload.get("label")
        party_id = payload.get("party_id")
    else:
        raise ValueError(
            f"{source}: JSON 顶层必须是格网码数组或带 grid_codes 的对象"
        )

    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{source}: 格网码数组必须是列表")
    codes = [_validate_code(code, where=source) for code in raw]
    return _finalize(
        codes, name=name, source=source, layout=layout, label=label, party_id=party_id
    )


def _load_json(path: str, *, name: str) -> ResolvedCellInput:
    if not os.path.exists(path):
        raise FileNotFoundError(f"输入文件不存在：{path}")
    with open(path, encoding="utf-8-sig") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 解析失败：{path}: {exc}") from exc
    return _from_payload(payload, name=name, source=f"json:{path}")


def load_cellset(source: Any, *, name: str = "") -> ResolvedCellInput:
    """把一种受支持的输入形态转换为校验完毕的 `ResolvedCellInput`。

    支持：ResolvedCellInput（原样）/ PartyInput / CellSet（geo_privacy.core）/
    路径（.csv / .json）/ JSON 结构（Mapping / list）/ 整数码序列。
    """

    if isinstance(source, ResolvedCellInput):
        return source

    if isinstance(source, PartyInput):
        resolved = load_cellset(source.data, name=name or source.name)
        return _replace(
            resolved,
            name=name or source.name,
            party_id=resolved.party_id or source.party_id,
            label=resolved.label,
            layout=source.layout or resolved.layout,
        )

    if isinstance(source, PlainCellSet):
        return _finalize(
            list(source.codes), name=name, source="CellSet", label=source.label
        )

    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if str(path).lower().endswith(".json"):
            return _load_json(str(path), name=name)
        return _load_csv(str(path), name=name)

    if isinstance(source, Mapping):
        return _from_payload(dict(source), name=name, source="mapping")

    if isinstance(source, list):
        return _from_payload(source, name=name, source="codes")

    try:
        iterator = iter(source)
    except TypeError as exc:
        raise TypeError(
            f"不支持的输入形态 {type(source).__name__}："
            "期望 CSV/JSON 路径、CellSet、PartyInput 或 64-bit 码序列"
        ) from exc
    codes = [_validate_code(code, where=name or "<codes>") for code in iterator]
    return _finalize(codes, name=name, source="codes")


def normalize_inputs(inputs: Mapping[str, Any]) -> dict[str, ResolvedCellInput]:
    """把 Compiler/CLI 收到的 `{名字: 输入}` 映射整体规范化。"""

    return {str(name): load_cellset(value, name=str(name)) for name, value in inputs.items()}


# --------------------------------------------------------------------------
# 布局握手
# --------------------------------------------------------------------------


def check_layout_agreement(
    left: Mapping[str, Any] | None, right: Mapping[str, Any] | None
) -> LayoutAgreement:
    """核对两方的格网布局清单（PSI 之前必须通过）。"""

    if left is None and right is None:
        return LayoutAgreement(
            agreement=None,
            notes=("两方都未声明布局清单：无法核对（声明后本检查才会生效）",),
        )
    left_id = (left or {}).get("layout_id")
    right_id = (right or {}).get("layout_id")
    if left is None or right is None:
        missing = "left" if left is None else "right"
        return LayoutAgreement(
            agreement=None,
            left_id=left_id,
            right_id=right_id,
            notes=(f"仅 {missing} 侧未声明布局清单：无法核对，已如实记录",),
        )
    mismatches: list[str] = []
    for key in LAYOUT_COMPARE_KEYS:
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is None and right_value is None:
            continue
        if left_value != right_value:
            mismatches.append(f"{key}: left={left_value!r} right={right_value!r}")
    return LayoutAgreement(
        agreement=not mismatches,
        left_id=left_id,
        right_id=right_id,
        mismatches=tuple(mismatches),
    )