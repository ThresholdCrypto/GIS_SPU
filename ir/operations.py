"""Geo-IR 算子与程序。

GeoOperation 的构造签名按课题要求固定为：
    GeoOperation(op="Intersects", inputs=["route", "no_fly_zone"], output_type="Relation")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .types import GeoType, Sensitivity, max_sensitivity, requires_crypto
from .values import GeoEntity, GeoRelation, GeoValue

#: 支持的算子名（与 planner 注册表同源）
SUPPORTED_OPS: tuple[str, ...] = (
    "Intersects",
    "Contains",
    "DistanceLE",
    "CellSetIntersect",
    "WeightedSum",
    "TemporalOverlap",
)


def _coerce_geo_type(value: Any) -> GeoType:
    if isinstance(value, GeoType):
        return value
    if isinstance(value, str):
        try:
            return GeoType(value)
        except ValueError:
            normalized = value.strip().lower()
            for member in GeoType:
                if member.value.lower() == normalized or member.name.lower() == normalized:
                    return member
            raise
    raise TypeError(f"无法把 {value!r} 解释为 GeoType")


@dataclass(frozen=True)
class GeoOperation:
    """Geo-IR 的一条算子指令。"""

    op: str
    inputs: tuple[str, ...]
    output_type: GeoType = GeoType.UNKNOWN
    output_name: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    #: 静态推导出的敏感级别；None 表示由 planner 依据输入重新推导
    sensitivity: Sensitivity | None = None
    #: 源码位置，用于错误定位（file / line / col）
    location: Mapping[str, Any] | None = None
    #: 用户源码里的原始表达式文本，便于报错时回显
    source_expr: str | None = None

    def __init__(
        self,
        op: str,
        inputs: Sequence[str],
        output_type: Any = GeoType.UNKNOWN,
        output_name: str | None = None,
        params: Mapping[str, Any] | None = None,
        sensitivity: Sensitivity | str | None = None,
        location: Mapping[str, Any] | None = None,
        source_expr: str | None = None,
    ) -> None:
        object.__setattr__(self, "op", str(op))
        object.__setattr__(self, "inputs", tuple(inputs))
        object.__setattr__(self, "output_type", _coerce_geo_type(output_type))
        object.__setattr__(
            self, "output_name", output_name if output_name is not None else self._default_output_name()
        )
        object.__setattr__(self, "params", dict(params or {}))
        if sensitivity is None or isinstance(sensitivity, Sensitivity):
            object.__setattr__(self, "sensitivity", sensitivity)
        else:
            object.__setattr__(self, "sensitivity", Sensitivity(str(sensitivity)))
        object.__setattr__(self, "location", dict(location) if location else None)
        object.__setattr__(self, "source_expr", source_expr)

    def _default_output_name(self) -> str:
        name = "".join(part.capitalize() for part in self.op.split("_"))
        return f"{name.lower()}_0"

    @property
    def is_relation_output(self) -> bool:
        return self.output_type is GeoType.RELATION

    @property
    def location_str(self) -> str:
        if not self.location:
            return "<unknown>"
        loc = self.location
        head = str(loc.get("file", "<source>"))
        if "line" in loc:
            head += f":{loc['line']}"
            if "col" in loc:
                head += f":{loc['col']}"
        return head

    def __str__(self) -> str:
        return f"{self.op}({', '.join(self.inputs)}) -> {self.output_type}"


@dataclass
class GeoProgram:
    """一个完整的 Geo-IR 程序：输入、算子序列、输出。"""

    name: str = "<module>"
    source_file: str | None = None
    entity_inputs: dict[str, GeoValue] = field(default_factory=dict)
    operations: list[GeoOperation] = field(default_factory=list)
    relations: list[GeoRelation] = field(default_factory=list)
    returns: tuple[str, ...] = ()
    diagnostics: list[Any] = field(default_factory=list)

    # ---------------- 构造 ----------------

    def declare_input(
        self,
        name: str,
        *,
        geo_type: GeoType = GeoType.ENTITY_SET,
        sensitivity: Sensitivity = Sensitivity.SENSITIVE,
        entity: GeoEntity | None = None,
        const: Any = None,
        is_constant: bool = False,
    ) -> GeoValue:
        value = GeoValue(
            name=name,
            geo_type=geo_type,
            sensitivity=sensitivity,
            entity=entity,
            const=const,
            is_constant=is_constant,
            source="<input>",
        )
        self.entity_inputs[name] = value
        return value

    def add_operation(self, operation: GeoOperation) -> GeoOperation:
        self.operations.append(operation)
        return operation

    def add_relation(self, relation: GeoRelation) -> GeoRelation:
        self.relations.append(relation)
        return relation

    # ---------------- 查询 ----------------

    def lookup(self, name: str) -> GeoValue | None:
        """按名字解析值：先查输入，再查算子输出。"""

        if name in self.entity_inputs:
            return self.entity_inputs[name]
        for operation in self.operations:
            if operation.output_name == name:
                return GeoValue(
                    name=name,
                    geo_type=operation.output_type,
                    sensitivity=operation.sensitivity or Sensitivity.INTERNAL,
                    source=operation.op,
                )
        return None

    def value_names(self) -> tuple[str, ...]:
        names = list(self.entity_inputs)
        names.extend(op.output_name for op in self.operations if op.output_name)
        return tuple(names)

    def input_sensitivity(self, operation: GeoOperation) -> Sensitivity:
        """算子输入的敏感级别。

        解析不出来的输入按 SECRET 计入——这是有意的保守选择：
        "不知道它是什么" 与 "它是公开的" 是两回事，把前者当后者
        会让接触密态数据的算子被判成可走明文（真实的隐私泄漏）。
        """

        levels = [
            value.effective_sensitivity
            if (value := self.lookup(name)) is not None
            else Sensitivity.SECRET
            for name in operation.inputs
        ]
        return max_sensitivity(*levels) if levels else Sensitivity.SECRET

    def requires_crypto(self, operation: GeoOperation) -> bool:
        return requires_crypto(self.input_sensitivity(operation))

    def get_operation(self, output_name: str) -> GeoOperation | None:
        for operation in self.operations:
            if operation.output_name == output_name:
                return operation
        return None

    # ---------------- 序列化 ----------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_file": self.source_file,
            "inputs": {
                name: {
                    "geo_type": str(value.geo_type),
                    "sensitivity": str(value.effective_sensitivity),
                    "is_constant": value.is_constant,
                }
                for name, value in self.entity_inputs.items()
            },
            "operations": [
                {
                    "op": op.op,
                    "inputs": list(op.inputs),
                    "output_type": str(op.output_type),
                    "output_name": op.output_name,
                    "params": dict(op.params),
                    "sensitivity": str(op.sensitivity) if op.sensitivity else None,
                    "location": dict(op.location) if op.location else None,
                }
                for op in self.operations
            ],
            "relations": [r.to_dict() for r in self.relations],
            "returns": list(self.returns),
        }