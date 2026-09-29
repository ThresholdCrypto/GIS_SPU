"""Geo-IR 的文本渲染：给 CLI 与报告用的可读输出。"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .operations import GeoOperation, GeoProgram
from .values import GeoRelation


def _wrap_relation(relation: GeoRelation) -> str:
    return f"{relation.subject} | {relation.predicate} | {relation.object}"


def render_program(program: GeoProgram, *, indent: str = "  ") -> str:
    """渲染整个 Geo-IR 程序。"""

    lines: list[str] = []
    lines.append(f"GeoProgram: {program.name}")
    if program.source_file:
        lines.append(f"{indent}source: {program.source_file}")

    if program.entity_inputs:
        lines.append(f"{indent}inputs:")
        for name, value in program.entity_inputs.items():
            flags = []
            if value.is_constant:
                flags.append("const")
            if value.geo_type is not None:
                flags.append(str(value.geo_type))
            lines.append(
                f"{indent * 2}{name}: {' / '.join(flags)} "
                f"[sensitivity={value.effective_sensitivity}]"
            )
            if value.entity is not None and value.entity.cell_count:
                lines.append(
                    f"{indent * 3}cells={value.entity.cell_count} "
                    f"attrs={dict(value.entity.attrs)}"
                )

    if program.operations:
        lines.append(f"{indent}operations:")
        for index, operation in enumerate(program.operations):
            lines.append(
                f"{indent * 2}[{index}] GeoOperation(op={operation.op!r}, "
                f"inputs={list(operation.inputs)!r}, output_type={str(operation.output_type)!r})"
            )
            details = []
            if operation.params:
                details.append(f"params={dict(operation.params)}")
            if operation.sensitivity is not None:
                details.append(f"sensitivity={operation.sensitivity}")
            if operation.source_expr:
                details.append(f"source={operation.source_expr}")
            if details:
                lines.append(f"{indent * 3}{' ; '.join(details)}")

    if program.relations:
        lines.append(f"{indent}relations:")
        for relation in program.relations:
            lines.append(f"{indent * 2}{_wrap_relation(relation)}")

    if program.returns:
        lines.append(f"{indent}returns: {list(program.returns)}")

    return "\n".join(lines)


def render_table(
    rows: Iterable[Sequence[Any]],
    headers: Sequence[str],
    *,
    pad: str = "  ",
) -> str:
    """极简表格渲染，避免引入 tabulate 之类的外部依赖。"""

    rows = [[("" if cell is None else str(cell)) for cell in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            if index < len(widths):
                widths[index] = max(widths[index], len(cell))

    def fmt(cells: Sequence[str]) -> str:
        return pad.join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    lines = [fmt([str(h) for h in headers])]
    lines.append(pad.join("-" * w for w in widths))
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def render_operations_table(operations: Sequence[GeoOperation]) -> str:
    rows = [
        [op.op, ", ".join(op.inputs), str(op.output_type), op.output_name, op.location_str]
        for op in operations
    ]
    return render_table(rows, ["Operation", "Inputs", "OutputType", "Output", "Location"])


def render_relations(relations: Sequence[GeoRelation]) -> str:
    rows = [
        [r.subject, r.predicate, r.object, r.time, r.spatial_scope, str(r.sensitivity)]
        for r in relations
    ]
    return render_table(rows, ["Subject", "Predicate", "Object", "Time", "SpatialScope", "Sensitivity"])