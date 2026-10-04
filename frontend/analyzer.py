"""frontend：把用户 Python 源码静态解析为 Geo-IR。

设计原则（刻意保守）：
- 只识别 `geo.<op>(...)` 这一种调用形态，不做通用 Python/Shapely/GeoPandas 转换。
- 只做静态分析，**从不执行用户代码**。
- 解析不出确定性结论时，报诊断而不是猜。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ir import (
    GeoEntity,
    GeoOperation,
    GeoProgram,
    GeoType,
    Sensitivity,
    encode_grid_code,
)
from ir.operations import SUPPORTED_OPS

from .dialect import (
    GEO_DIALECT,
    DialectEntry,
    is_plaintext_utility,
    resolve_dialect,
)

# --------------------------------------------------------------------------
# 诊断
# --------------------------------------------------------------------------

#: 六类失败模式（与 validator 共享编号）
DIAG_UNSUPPORTED_GEO_OP = "GEO_OP_UNSUPPORTED"
DIAG_UNKNOWN_CALL = "GEO_CALL_UNRECOGNIZED"
DIAG_DYNAMIC_CONTROL_FLOW = "DYNAMIC_CONTROL_FLOW"
DIAG_NOT_TRACEABLE = "JAX_NOT_TRACEABLE"
DIAG_BACKEND_MISSING = "BACKEND_OP_MISSING"
#: 第 6 类：高度层号装不进 Z 位域（国标附录 B 的层号随层级变化）
DIAG_HEIGHT_LAYER = "HEIGHT_LAYER_UNSUPPORTED"
DIAG_SPU_UNSUPPORTED = "SPU_UNSUPPORTED"
DIAG_ARITY = "GEO_ARITY_MISMATCH"
DIAG_STATIC_ASSERT = "STATIC_CHECK_FAILED"
#: 实参无法静态解析为 Geo-IR 值（表达式实参 / 未声明名字 / 非方言调用）
DIAG_UNRESOLVED_INPUT = "GEO_INPUT_UNRESOLVED"
#: 明文工具调用（geo.quantize 等）：不是算子，不产生算子诊断
DIAG_PLAINTEXT_UTILITY = "GEO_PLAINTEXT_UTILITY"
#: 算子 × 协议组合校验失败（不在候选清单 / 协议本身跑不了 / 非 PSI 算子配协议）
DIAG_PROTOCOL_UNSUPPORTED = "PROTOCOL_UNSUPPORTED"


@dataclass
class Diagnostic:
    """一条可定位的诊断。绝不自动改用户代码，只陈述问题与替代方案。"""

    code: str
    message: str
    location: dict[str, Any] = field(default_factory=dict)
    cause: str = ""
    suggestion: str = ""
    suggested_op: str | None = None
    estimated_cost: dict[str, Any] | None = None
    severity: str = "error"

    @property
    def location_str(self) -> str:
        loc = self.location
        out = str(loc.get("file", "<source>"))
        if "line" in loc:
            out += f":{loc['line']}"
            if "col" in loc:
                out += f":{loc['col']}"
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "location": dict(self.location),
            "cause": self.cause,
            "suggestion": self.suggestion,
            "suggested_op": self.suggested_op,
            "estimated_cost": self.estimated_cost,
        }

    def __str__(self) -> str:
        head = f"[{self.severity.upper()}] {self.code} @ {self.location_str}: {self.message}"
        if self.cause:
            head += f"\n    原因: {self.cause}"
        if self.suggestion:
            head += f"\n    建议: {self.suggestion}"
        if self.suggested_op:
            head += f"\n    替代算子: {self.suggested_op}"
        if self.estimated_cost:
            head += f"\n    预计代价: {self.estimated_cost}"
        return head


class FrontendError(Exception):
    """前端致命错误（语法层面无法继续）。"""

    def __init__(self, diagnostic: Diagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


# --------------------------------------------------------------------------
# 参数与函数信息
# --------------------------------------------------------------------------


@dataclass
class FunctionInfo:
    name: str
    args: list[str]
    returns: str | None
    node: ast.FunctionDef
    lineno: int
    docstring: str | None = None


@dataclass
class ParseResult:
    program: GeoProgram
    diagnostics: list[Diagnostic]
    functions: list[FunctionInfo]
    module_node: ast.Module
    entry_function: str | None = None
    source: str = ""

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity != "error"]


# --------------------------------------------------------------------------
# AST 分析器
# --------------------------------------------------------------------------


class FrontendAnalyzer(ast.NodeVisitor):
    """把单个模块转换为 Geo-IR。

    参数类型来自三个来源，优先级由高到低：
        1. 源码里的类型注解（`route: CellSet`）
        2. 显式声明的 sensitivity 注解（`#: sensitivity=secret`）
        3. 方言默认值（DIALECT 表）
    """

    #: 判定为"动态控制流"的语句类型
    CONTROL_NODES = (ast.If, ast.For, ast.While, ast.AsyncFor, ast.AsyncWith, ast.Try)

    def __init__(
        self,
        source: str,
        filename: str = "<source>",
        *,
        type_hints: dict[str, GeoType] | None = None,
        sensitivities: dict[str, Sensitivity] | None = None,
    ) -> None:
        self.source = source
        self.filename = filename
        self.lines = source.splitlines()
        self.tree = ast.parse(source, filename=filename)
        # 显式声明一律规范化：允许调用方写字符串（"secret" / "CellSet"），
        # 但不允许把无法识别的值原样带进 IR（那会在判定密态时静默失真）。
        self.type_hints = {k: _coerce_geotype(v) for k, v in (type_hints or {}).items()}
        self.sensitivities = {
            k: _coerce_sensitivity(v) for k, v in (sensitivities or {}).items()
        }

        self.diagnostics: list[Diagnostic] = []
        self.program = GeoProgram(name="<module>", source_file=None if filename == "<source>" else filename)
        self.functions: list[FunctionInfo] = []
        self._current_function: FunctionInfo | None = None
        self._nested_control_depth = 0
        self._seen_ops: set[str] = set()
        #: 合成名计数器：保证每个算子结果在 GeoProgram 内唯一
        self._temp_counter = 0
        #: id(ast.Call) → 该调用的输出名，供外层实参引用
        self._call_outputs: dict[int, str] = {}
        #: 已被显式变量承载过的结果名，用于检出重复赋值
        self._emitted_names: set[str] = set()

    # ---------------- 主入口 ----------------

    def run(self, entry: str | None = None) -> ParseResult:
        self.program.name = self.filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        self.visit(self.tree)

        # 选定入口函数
        analysis_targets = self.functions
        if entry is not None:
            matched = [f for f in self.functions if f.name == entry]
            if not matched:
                self._error(
                    DIAG_STATIC_ASSERT,
                    f"找不到入口函数 {entry!r}",
                    node=self.tree,
                    cause=f"模块内已定义函数：{[f.name for f in self.functions]}",
                    suggestion="检查入口函数名，或省略 --entry 以分析全部函数",
                )
                analysis_targets = []
            else:
                analysis_targets = matched

        for function in analysis_targets:
            self._analyze_function(function)

        self._warn_on_unused_sensitivity_keys()

        result = ParseResult(
            program=self.program,
            diagnostics=self.diagnostics,
            functions=self.functions,
            module_node=self.tree,
            entry_function=entry or (self.functions[0].name if len(self.functions) == 1 else None),
            source=source_or_empty(self.source),
        )
        self.program.diagnostics = result.diagnostics
        return result

    # ---------------- 收集函数 ----------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        info = FunctionInfo(
            name=node.name,
            args=[a.arg for a in node.args.args],
            returns=_unparse(node.returns),
            node=node,
            lineno=node.lineno,
            docstring=ast.get_docstring(node),
        )
        self.functions.append(info)
        # 函数体延迟分析（run 里按入口筛选）
        # 但仍需遍历嵌套 def 以便收集
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child is not node:
                pass
        return

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    # ---------------- 模块级语句 ----------------

    def visit_Module(self, node: ast.Module) -> None:
        """遍历模块顶层语句。

        此前**没有**这个方法：NodeVisitor 的默认 generic_visit 未实现，
        于是模块级 geo 调用（README 7.1.1 展示的写法）连"未识别的调用"
        提示都不会出现——整条语句连同它承载的算子被静默丢弃，
        而编译结果照报成功。这里改为显式下降，把这类调用暴露成诊断。
        """

        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit_FunctionDef(statement)
                continue
            self._scan_module_statement(statement)

    def _scan_module_statement(self, node: ast.stmt) -> None:
        """模块顶层只找 geo 方言调用，**不做控制流 lint**。

        模块顶层的 `if __name__ == "__main__":`、`try: import ...` 之类是
        正常的 Python 脚手架，不是"密态分支"——控制流检查针对的是被分析的
        函数体。对脚手架报 DYNAMIC_CONTROL_FLOW 属误报，而"宁可少报，
        不可误报"是本模块的既定纪律（见 _type_matches 的同名注释）。

        但**函数调用**（`if compute(): ...`）是真的动态：分支条件依赖运行期
        数据，与函数体内的同类写法同罪。因此这里不做语句级 lint，
        却把语句嵌套计入 _nested_control_depth——它对函数调用、对
        `__name__` 字面量比较都成立，不会误伤脚手架。

        收集方式用 _collect_geo_calls，只收方言调用：
        非方言调用（print / 自建 helper）连"未识别调用"都不报，减少噪音。
        """

        targets: tuple[str, ...] = ()
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = tuple(t.id for t in node.targets if isinstance(t, ast.Name))
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target.id,) if isinstance(node.target, ast.Name) else ()
            value = node.value
        elif isinstance(node, ast.Expr):
            value = node.value

        dynamic = _is_dynamic_module_control(node)
        self._nested_control_depth += 1 if dynamic else 0
        try:
            if value is not None:
                self._emit_module_calls(value, targets=targets)
                return
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    self._scan_module_statement(child)
                elif isinstance(child, ast.ExceptHandler):
                    for inner in child.body:
                        self._scan_module_statement(inner)
                else:
                    self._emit_module_calls(child)
        finally:
            self._nested_control_depth -= 1 if dynamic else 0

    def _emit_module_calls(self, expr: ast.expr, *, targets: Sequence[str] = ()) -> None:
        """发射模块顶层表达式里的方言调用（内层先发射，与函数体一致）。"""

        calls: list[ast.Call] = []
        self._collect_geo_calls(expr, calls)
        for call in calls:
            self._handle_call(call, targets=targets if call is expr else ())

    # ---------------- 逐函数分析 ----------------

    def _analyze_function(self, function: FunctionInfo) -> None:
        self._current_function = function

        # 1) 先扫描 geo 调用，从方言反推参数类型（地理语义识别）
        inferred = self._infer_param_types(function)

        # 2) 声明输入值
        for arg in function.args:
            self._declare_input(arg, function, inferred.get(arg))

        # 3) 逐语句遍历
        for statement in function.node.body:
            self._visit_statement(statement)

        # 4) 记录返回
        returns = self._collect_returns(function.node)
        self.program.returns = tuple(returns)
        if not returns and function.returns is not None:
            self._warn(
                DIAG_STATIC_ASSERT,
                f"函数 {function.name} 有返回类型注解 {function.returns}，但未发现 return 语句",
                node=function.node,
                suggestion="补上 return，或去掉类型注解",
            )

    def _infer_param_types(self, function: FunctionInfo) -> dict[str, GeoType]:
        """从函数体内 geo.<op>(...) 的调用形态反推参数类型。

        这是"地理语义识别"的核心：用户不写类型注解，类型由调用的算子隐含决定。
        多在多处出现时取"更具体"的类型（Scalar < 集合类）。
        """

        inferred: dict[str, GeoType] = {}
        for node in ast.walk(function.node):
            if not isinstance(node, ast.Call):
                continue
            resolved = self._resolve_geo_call_quiet(node)
            if resolved is None:
                continue
            entry, _ = resolved
            for arg_node, raw_type in _typed_arguments(node, entry):
                if not isinstance(arg_node, ast.Name):
                    continue
                expected = _geotype_of_name(raw_type)
                current = inferred.get(arg_node.id)
                inferred[arg_node.id] = _more_specific(current, expected)
        return inferred

    def _resolve_geo_call_quiet(self, call: ast.Call) -> tuple[DialectEntry, str] | None:
        """与 _resolve_geo_call 相同，但不产生任何诊断（用于预扫描）。"""

        func = call.func
        if isinstance(func, ast.Name) and func.id in GEO_DIALECT:
            return GEO_DIALECT[func.id], func.id
        if isinstance(func, ast.Attribute):
            receiver_name = _unparse(func.value)
            if receiver_name in ("geo", "geo_privacy.geo"):
                resolved = resolve_dialect(func.attr)
                if resolved is not None:
                    return resolved, func.attr
        return None

    def _declare_input(
        self, arg: str, function: FunctionInfo, inferred: GeoType | None = None
    ) -> None:
        """声明一个函数参数为 Geo-IR 输入值。

        类型来源优先级：显式 type_hints > 源码注解 > 方言推断 > 默认 EntitySet。
        """

        annotation = self._annotation_of(arg, function)
        geo_type = (
            self.type_hints.get(arg)
            or _annot_to_geotype(annotation)
            or inferred
            or GeoType.ENTITY_SET
        )
        sensitivity = self.sensitivities.get(arg, _default_sensitivity_for(geo_type))
        self.program.declare_input(
            arg,
            geo_type=geo_type,
            sensitivity=sensitivity,
            entity=(
                None
                if geo_type in (GeoType.SCALAR, GeoType.BOOL)
                else GeoEntity(name=arg, entity_type=geo_type, sensitivity=sensitivity)
            ),
        )

    def _annotation_of(self, arg: str, function: FunctionInfo) -> str | None:
        """从函数签名中取该参数的注解文本。

        注意：注解挂在每个 ast.arg 上，arg.annotation，而不是 arguments 对象上。
        """

        for node in function.node.args.args:
            if node.arg == arg and node.annotation is not None:
                return _unparse(node.annotation)
        return None

    # ---------------- 语句遍历（控制流检测） ----------------

    def _visit_statement(self, node: ast.stmt) -> None:
        if isinstance(node, self.CONTROL_NODES):
            self._report_control_flow(node)
            # 仍需下降，找出里面的 geo 调用（以便同时给出算子建议）。
            # 既要下到 body/orelse，也要扫条件表达式本身——
            # `if geo.intersects(a, b):` 这种写法以前会被整条丢掉。
            self._nested_control_depth += 1
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    self._visit_statement(child)
                elif isinstance(child, ast.ExceptHandler):
                    for inner in child.body:
                        self._visit_statement(inner)
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, ast.stmt) and not isinstance(child, ast.ExceptHandler):
                    self._emit_expression_calls(child)
            self._nested_control_depth -= 1
            return

        # 取出本语句里"承载结果"的表达式与它对应的变量名。
        # 只有最外层那个方言调用会拿到变量名，内层调用一律用合成名。
        targets: tuple[str, ...] = ()
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = tuple(t.id for t in node.targets if isinstance(t, ast.Name))
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target.id,) if isinstance(node.target, ast.Name) else ()
            value = node.value
        elif isinstance(node, ast.Return):
            value = node.value
        elif isinstance(node, ast.Expr):
            value = node.value

        if value is not None:
            self._emit_expression_calls(value, targets=targets)
            return

        # 其余语句类型（AugAssign 等）：仍扫一遍表达式，别再丢算子
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                self._visit_statement(child)
            elif isinstance(child, ast.ExceptHandler):
                for inner in child.body:
                    self._visit_statement(inner)
            else:
                self._emit_expression_calls(child)

    def _emit_expression_calls(self, expr: ast.expr, *, targets: Sequence[str] = ()) -> None:
        """扫描一段表达式子树，按"内层先发射"的顺序处理其中的方言调用。

        这是"嵌套调用被静默丢弃"的修复点：识别不再依赖 node.value 是不是
        直接的 ast.Call，而是把整棵表达式树走完。走不完就等于把算子连同
        它的密态输入一起丢掉——那是真实的隐私泄漏，而不是少报一行日志。
        """

        if isinstance(expr, ast.Call) and self._resolve_geo_call_quiet(expr) is None:
            # 表达式根部的非方言调用：仍走一遍识别，保留"未识别调用"
            # 与"不支持的地理算子"两类提示（内部按名字去重）。
            self._resolve_geo_call(expr)

        calls: list[ast.Call] = []
        self._collect_geo_calls(expr, calls)
        for call in calls:
            # 只有表达式最外层的调用能被变量名承载
            self._handle_call(call, targets=targets if call is expr else ())

    def _collect_geo_calls(self, node: ast.AST, out: list[ast.Call]) -> None:
        """后序收集方言调用：子调用先入列，保证内层算子先发射。

        语句边界交给 _visit_statement 递归处理，这里不下钻，
        避免同一个算子被发射两次。
        """

        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                continue
            self._collect_geo_calls(child, out)
        if isinstance(node, ast.Call) and self._is_geo_relevant_call(node):
            out.append(node)

    def _is_geo_relevant_call(self, call: ast.Call) -> bool:
        """该调用是否值得进识别流程。

        除方言算子外还包括**明文工具**（geo.quantize）：它虽然没有算子语义，
        但必须给用户一句明确说明（"不进入 Geo-IR"），否则用户会以为
        它算出来的东西受了保护。不收进这里就等于把它静默丢掉——
        与 geo.height_band 当初的问题同源。
        """

        if self._resolve_geo_call_quiet(call) is not None:
            return True
        func = call.func
        if isinstance(func, ast.Attribute) and _unparse(func.value) in (
            "geo",
            "geo_privacy.geo",
        ):
            return is_plaintext_utility(func.attr)
        return False

    def _report_control_flow(self, node: ast.stmt) -> None:
        kind = type(node).__name__.lower()
        static_ok = _is_statically_decidable(node)
        severity = "error"
        cause = (
            "该分支条件的取值依赖运行期数据，编译期无法展开为静态计算图，"
            "因此既不能进入 jax.jit 追踪，也不能映射为 MPC 电路。"
        )
        if static_ok:
            severity = "warning"
            cause = "该分支条件在编译期可判定，将被静态展开；但请确认它确实不依赖密态数据。"

        self.diagnostics.append(
            Diagnostic(
                code=DIAG_DYNAMIC_CONTROL_FLOW,
                severity=severity,
                message=f"检测到 {kind} 语句，密态分支无法编译",
                location=self._loc(node),
                cause=cause,
                suggestion=(
                    "改为数据并行写法：用 jnp.where / jnp.maximum 等无分支原语表达条件选择；"
                    "或把分支拆成多个独立 geo 任务，由业务侧按需组合。"
                ),
                suggested_op="（分支选择 → WeightedSum 或 DistanceLE 的阈值形式）",
                estimated_cost={
                    "note": "分支展开使密文条数按路径数线性放大",
                    "multiplier": "路径数 × N_ct",
                },
            )
        )

    # ---------------- 调用识别 ----------------

    def _handle_call(
        self, call: ast.Call, targets: Sequence[str] | None = None
    ) -> str | None:
        """识别并发射一个方言调用，返回它的输出名（未识别时为 None）。"""

        resolved = self._resolve_geo_call(call)
        if resolved is None:
            return None
        entry, method_name = resolved
        return self._emit_operation(call, entry, method_name, targets)

    def _resolve_geo_call(self, call: ast.Call) -> tuple[DialectEntry, str] | None:
        """判断是否为 `geo.<op>(...)` 形态。"""

        func = call.func
        if not isinstance(func, ast.Attribute):
            # 可能是 `intersects(...)` 直接导入形式
            if isinstance(func, ast.Name) and func.id in GEO_DIALECT:
                return GEO_DIALECT[func.id], func.id
            # 完全无法识别 → 仅在函数体内首次出现时提示
            if isinstance(func, (ast.Name, ast.Attribute)):
                self._maybe_unknown_call(call, func)
            return None

        receiver = func.value
        receiver_name = _unparse(receiver)
        if receiver_name not in ("geo", "geo_privacy.geo"):
            # 其他对象的属性调用不在方言范围内，静默跳过（避免噪音）
            return None

        method = func.attr
        resolved = resolve_dialect(method)
        if resolved is None:
            if is_plaintext_utility(method):
                # 明文工具（geo.quantize）：当场算标量，不进 Geo-IR。
                # 报"算子不支持"是假错误——它本来就不是算子。
                self._report_plaintext_utility(call, method)
                return None
            self._unsupported_geo_op(call, method)
            return None
        return resolved, method

    def _report_plaintext_utility(self, call: ast.Call, method: str) -> None:
        """`geo.<util>()` 是明文工具，不是算子：说明一次即可，不阻塞编译。"""

        name = f"geo.{method}"
        if name in self._seen_ops:
            return
        self._seen_ops.add(name)
        self.diagnostics.append(
            Diagnostic(
                code=DIAG_PLAINTEXT_UTILITY,
                severity="warning",
                message=f"{name}() 是明文工具，不进入 Geo-IR",
                location=self._loc(call),
                cause=(
                    "它返回普通标量（箱号、阈值），不是地理实体，"
                    "因此没有算子语义、也不会被降级为隐私算子。"
                ),
                suggestion=(
                    "正常使用即可；若要让它参与密态判定，"
                    "请把结果作为阈值传给 geo.distance_le 之类的算子"
                ),
                estimated_cost={"note": "该调用不产生隐私计算开销，也不会被密态执行"},
            )
        )

    def _maybe_unknown_call(self, call: ast.Call, func: ast.expr) -> None:
        name = _unparse(func)
        if name in self._seen_ops:
            return
        self._seen_ops.add(name)
        self.diagnostics.append(
            Diagnostic(
                code=DIAG_UNKNOWN_CALL,
                severity="warning",
                message=f"未识别的调用 {name}()，已忽略",
                location=self._loc(call),
                cause="编译器只识别 geo_privacy 方言；其他调用不会被降级为隐私算子。",
                suggestion=f"可用的地理算子：{sorted(GEO_DIALECT)}",
                estimated_cost={"note": "该调用不产生隐私计算开销，也不会被密态执行"},
            )
        )

    def _unsupported_geo_op(self, call: ast.Call, method: str) -> None:
        suggestion = _nearest_ops(method)
        self.diagnostics.append(
            Diagnostic(
                code=DIAG_UNSUPPORTED_GEO_OP,
                severity="error",
                message=f"geo.{method}() 不是受支持的地理算子",
                location=self._loc(call),
                cause=f"当前方言只覆盖 {len(GEO_DIALECT)} 个算子：{sorted(GEO_DIALECT)}",
                suggestion="改用下方的等价算子名，或在 dialect 中登记新算子（需同时补后端实现与测试）",
                suggested_op=" / ".join(suggestion) if suggestion else None,
                estimated_cost=_cost_hint_for(suggestion[0]) if suggestion else None,
            )
        )

    def _emit_operation(
        self,
        call: ast.Call,
        entry: DialectEntry,
        method_name: str,
        targets: Sequence[str] | None,
    ) -> str | None:
        if entry.materializes:
            return self._emit_materialize(call, entry, method_name, targets)

        arity = len(call.args)
        if arity != entry.arity:
            self.diagnostics.append(
                Diagnostic(
                    code=DIAG_ARITY,
                    severity="error",
                    message=f"geo.{method_name}() 需要 {entry.arity} 个参数，实得 {arity}",
                    location=self._loc(call),
                    cause="算子签名固定；参数数量不符会使后端无法选择表示（representation）。",
                    suggestion=f"按 geo.{method_name}({', '.join(entry.geotypes)}) 的签名补齐参数",
                    suggested_op=entry.op,
                )
            )

        input_names = self._resolve_inputs(call, entry, method_name)

        # 校验输入类型是否与方言声明一致
        self._check_input_types(call, entry, method_name, input_names)

        output_name = targets[0] if targets else None
        if output_name is not None:
            self._warn_on_name_reuse(output_name, call)

        operation = GeoOperation(
            op=entry.op,
            inputs=input_names,
            # 用"产出类型"而非"供应物"：下游算子按它做类型检查，
            # 填错等于把链式类型错误一路放行。
            output_type=entry.output_geo_type or entry.returns,
            output_name=output_name,
            params=dict(entry.params),
            location=self._loc(call),
            source_expr=_unparse(call),
        )
        # 注意：GeoOperation 会把 output_name=None 兜底成 "<op>_0"，所以这里
        # 不能用 output_name is None 判断，必须看 targets。旧版正是踩了这个兜底：
        # 全部同类算子都叫 "<op>_0"，lookup 解析到的是**别的那一个**结果，
        # 于是敏感度算错，密态输入被判成 plaintext-ok。
        if output_name is None:
            operation = _rename_operation(operation, self._next_temp_name(entry.op))

        # 敏感度：取输入最大者；解析不到的输入按 SECRET 计入
        operation = _set_sensitivity(operation, self.program.input_sensitivity(operation))

        # 若在动态控制流内，标注该算子被不可编译结构包裹
        if self._nested_control_depth > 0:
            operation = _add_param(operation, "_inside_control_flow", True)

        self.program.add_operation(operation)
        self._register_relation(operation, entry)
        # 记下"这个调用产出哪个名字"，外层实参才能解析到它
        self._call_outputs[id(call)] = operation.output_name
        return operation.output_name

    # ---------------- 物化算子（在本方明文算出码集合，不进密态） ----------------

    def _emit_materialize(
        self,
        call: ast.Call,
        entry: DialectEntry,
        method_name: str,
        targets: Sequence[str] | None,
    ) -> str | None:
        """发射一个物化算子（`geo.height_band(...)`）。

        实参是**命名参数**：`x=` / `height_min=` 等。被编码的数据（x, y）进
        `inputs`；高度带与层级是物化配置，进 `params`。

        配置进 params 而非 inputs 是有意的：敏感度判定只看 `inputs`，
        把 `level=15` 混进去会让"公开的编码参数"参与密态需求判定。
        层号容量校验（第 6 类失败）则正好从 params 里取 level / height_max。
        """

        values, params, problems = self._collect_arguments(call, entry, method_name)
        if problems:
            return None

        # 输入类型同样要核：把格网集合当成 XY 单元索引用是链式类型错误，
        # 与位置实参族走同一套判据，不在物化这条路上开例外。
        self._check_input_types(call, entry, method_name, values)

        output_name = targets[0] if targets else None
        if output_name is not None:
            self._warn_on_name_reuse(output_name, call)

        operation = GeoOperation(
            op=entry.op,
            inputs=tuple(values),
            output_type=entry.output_geo_type or entry.returns,
            output_name=output_name,
            params=params,
            location=self._loc(call),
            source_expr=_unparse(call),
        )
        if output_name is None:
            operation = _rename_operation(operation, self._next_temp_name(entry.op))

        # 物化在本方明文完成，但**产出值的敏感级别必须由输入推导**，不能写死 PUBLIC：
        # 若调用方把 x/y 标注为 secret，写死 PUBLIC 会让消费它的 PSI 算子
        # 被判成"可走明文"——那是一条真实的泄漏路径。
        # 编码本身不进密态（由 planner 的 backend=Plaintext 表达），
        # 与"结果值有多敏感"是两件事。
        operation = _set_sensitivity(operation, self.program.input_sensitivity(operation))

        if self._nested_control_depth > 0:
            operation = _add_param(operation, "_inside_control_flow", True)

        self.program.add_operation(operation)
        self._call_outputs[id(call)] = operation.output_name
        return operation.output_name

    def _collect_arguments(
        self, call: ast.Call, entry: DialectEntry, method_name: str
    ) -> tuple[list[str], dict[str, Any], bool]:
        """收集物化算子的实参；返回 (inputs, params, 是否出错)。

        位置与命名两种写法都支持：facade 的签名顺序恰好是
        `value_params + config_params`（x, y, height_min, height_max, level,
        toff, lt），因此位置实参可以无歧义地按序对号入座。
        拒绝位置写法会误报——业务侧那样写在本方明文里本来就能跑通。
        """

        problems = False
        positional = list(entry.value_params) + list(entry.config_params)

        if len(call.args) > len(positional):
            self._error(
                DIAG_ARITY,
                f"geo.{method_name}() 最多接受 {len(positional)} 个参数，实得 {len(call.args)}",
                node=call,
                cause=f"参数依次为 {list(positional)}。",
                suggestion="核对参数个数；或用命名参数只写需要的那些",
                suggested_op=entry.op,
            )
            return [], {}, True

        given: dict[str, ast.expr] = dict(zip(positional, call.args))
        for keyword in call.keywords:
            if keyword.arg is None:
                self._error(
                    DIAG_UNRESOLVED_INPUT,
                    f"geo.{method_name}() 不支持 **kwargs 展开",
                    node=call,
                    cause="展开式实参在编译期无法对应到参数名。",
                    suggestion="逐个写出命名参数（x= / y= / height_min= ...）",
                    suggested_op=entry.op,
                )
                problems = True
                continue
            if keyword.arg not in entry.value_params + entry.config_params:
                self._error(
                    DIAG_STATIC_ASSERT,
                    f"geo.{method_name}() 收到未知参数 {keyword.arg!r}",
                    node=call,
                    cause=f"可用参数：{sorted(entry.value_params + entry.config_params)}",
                    suggestion="核对参数名拼写",
                    suggested_op=entry.op,
                )
                problems = True
                continue
            given[keyword.arg] = keyword.value

        missing = [name for name in entry.required_params if name not in given]
        if missing and not problems:
            self._error(
                DIAG_ARITY,
                f"geo.{method_name}() 缺少必需参数 {missing}",
                node=call,
                cause=f"必需参数为 {list(entry.required_params)}（其余有默认值）。",
                suggestion="补齐后重试",
                suggested_op=entry.op,
            )
            problems = True

        values: list[str] = []
        params: dict[str, Any] = {}

        for name in entry.value_params:
            node = given.get(name)
            if node is None:
                continue
            values.append(self._resolve_one_input(node, entry, method_name, call))

        for name in entry.config_params:
            node = given.get(name)
            if node is None:
                if name in entry.defaults:
                    params[name] = entry.defaults[name]
                continue
            literal = _literal_value(node)
            if literal is None:
                self._error(
                    DIAG_STATIC_ASSERT,
                    f"geo.{method_name}() 的参数 {name} 必须是编译期字面量",
                    node=node,
                    cause=(
                        "高度带与层级决定格网码本身，必须在编译期确定；"
                        "运行期才定的层号会让码集合无法静态展开。"
                    ),
                    suggestion=f"把 {name} 改成字面量（例如 {name}=15）",
                    suggested_op=entry.op,
                )
                problems = True
                continue
            params[name] = literal

        if not problems:
            # 层号容量（第 6 类失败）由 validator 从这两个键读取；
            # 这里保证物化出来的算子确实带上了它们。
            params.setdefault("height_level", params.get("level"))

        return values, params, problems

    def _resolve_inputs(
        self, call: ast.Call, entry: DialectEntry, method_name: str
    ) -> list[str]:
        """把实参 AST 解析成 Geo-IR 里**能解析到值**的名字。

        这是敏感度判定的地基：inputs 里的名字必须能被 GeoProgram.lookup 解析，
        否则算子会被当成不接触密态数据（旧版直接落回 PUBLIC，属真实泄漏）。
        """

        return [self._resolve_one_input(arg, entry, method_name, call) for arg in call.args]

    def _resolve_one_input(
        self, arg: ast.expr, entry: DialectEntry, method_name: str, call: ast.Call
    ) -> str:
        if isinstance(arg, ast.Name):
            if self.program.lookup(arg.id) is not None:
                return arg.id
            self._unresolved_input_diagnostic(arg, entry, method_name, kind="name")
            return arg.id

        if isinstance(arg, ast.Constant):
            return self._register_constant(arg)

        if isinstance(arg, ast.Call):
            if self._resolve_geo_call_quiet(arg) is not None:
                # 内层算子已按后序先发射，这里直接引用它的输出名
                name = self._call_outputs.get(id(arg))
                if name is not None:
                    return name
            self._unresolved_input_diagnostic(arg, entry, method_name, kind="call")
            return _unparse(arg)

        self._unresolved_input_diagnostic(arg, entry, method_name, kind="expr")
        return _unparse(arg)

    def _register_constant(self, node: ast.Constant) -> str:
        """把字面量实参登记为 PUBLIC 常量，使其在 Geo-IR 里可解析。

        常量必须可解析：否则会被保守规则按 SECRET 计入，
        把纯明文任务（如阈值 2）误判成必须进密态。
        """

        name = _unparse(node)
        if self.program.lookup(name) is None:
            value = node.value
            self.program.declare_input(
                name,
                geo_type=GeoType.BOOL if isinstance(value, bool) else GeoType.SCALAR,
                sensitivity=Sensitivity.PUBLIC,
                const=value,
                is_constant=True,
            )
        return name

    def _unresolved_input_diagnostic(
        self, arg: ast.expr, entry: DialectEntry, method_name: str, *, kind: str
    ) -> None:
        """实参无法静态解析时的诊断：位置 / 原因 / 替代写法 / 预计代价。"""

        text = _unparse(arg)
        if kind == "call":
            cause = (
                f"{text!r} 是一次普通 Python 调用，不是受支持的地理算子；"
                "编译器不执行用户代码，因此拿不到它的返回值与敏感度。"
            )
        elif kind == "name":
            cause = (
                f"名字 {text!r} 既不是函数形参，也不是已知算子的输出，"
                "无法确定它承载的数据类型与敏感级别。"
            )
        else:
            cause = (
                f"{text!r} 是表达式实参（切片、属性、下标等），"
                "Geo-IR 要求输入是有名字的值，无法静态确定其类型与敏感级别。"
            )

        self.diagnostics.append(
            Diagnostic(
                code=DIAG_UNRESOLVED_INPUT,
                severity="warning",
                message=f"geo.{method_name}() 的实参 {text!r} 无法解析为 Geo-IR 值",
                location=self._loc(arg),
                cause=(
                    cause
                    + " 为避免静默降级，该实参按最敏感级别（SECRET）计入，"
                    "本算子会被安排进密态路径。"
                ),
                suggestion=(
                    "先把它赋值给一个变量再传入"
                    f"（例如 tmp = <表达式>; geo.{method_name}(..., tmp)），"
                    "使 Geo-IR 能解析出它的来源与敏感度"
                ),
                suggested_op=entry.op,
                estimated_cost=_cost_hint_for(entry.op),
            )
        )

    def _next_temp_name(self, op: str) -> str:
        """给没有变量承载的算子结果生成唯一且确定的名字。

        唯一性是硬要求：GeoProgram.lookup 按名字解析值，重名会让下游算子
        取到同名的另一个结果，把敏感度算错。名字只依赖源码顺序，
        因此同一份源码每次编译得到同样的 IR。
        """

        while True:
            self._temp_counter += 1
            name = f"geo_{op.lower()}_{self._temp_counter}"
            if self.program.lookup(name) is None:
                return name

    def _warn_on_name_reuse(self, name: str, call: ast.Call) -> None:
        if name in self._emitted_names:
            self._warn(
                DIAG_STATIC_ASSERT,
                f"算子结果名 {name!r} 被重复赋值，前一个结果将被遮蔽",
                node=call,
                cause="Geo-IR 按名字解析值；同名结果会让下游算子引用到不确定的那一个。",
                suggestion="给每个算子结果换一个不同的变量名",
            )
        self._emitted_names.add(name)

    def _check_input_types(
        self, call: ast.Call, entry: DialectEntry, method_name: str, input_names: Sequence[str]
    ) -> None:
        """逐实参核对类型。

        既查函数形参（由方言反推或注解给出），也查**中间结果**——
        后者以前被整段跳过（"交给后续阶段"），于是
        `geo.weighted_sum(geo.cellset_intersect(a, b), w)` 这种把格网集合
        当定点向量用的链式错误一路无诊断地通过。
        """

        for index, (name, expected) in enumerate(zip(input_names, entry.geotypes)):
            value = self.program.lookup(name)
            if value is None:
                continue  # 无法解析的实参已在 _resolve_one_input 里报过
            if _type_matches(value.geo_type, expected):
                continue
            expected_type = _geotype_of_name(expected)
            self.diagnostics.append(
                Diagnostic(
                    code=DIAG_STATIC_ASSERT,
                    severity="error",
                    message=(
                        f"geo.{method_name}() 第 {index + 1} 个参数 {name!r} 的类型是 "
                        f"{value.geo_type}，但算子要求 {expected}"
                    ),
                    location=self._loc(call),
                    cause=(
                        "类型不符会使后端选错计算表征，密态下无法在运行期补救。"
                        + (
                            "该实参是中间结果，其类型由上游算子的产出决定。"
                            if name not in self.program.entity_inputs
                            else ""
                        )
                    ),
                    suggestion=f"把 {name!r} 转换为 {expected}，或改用更适合该类型的算子",
                    suggested_op=_nearest_ops_for_types(expected_type, method_name),
                )
            )

    def _register_relation(self, operation: GeoOperation, entry: DialectEntry) -> None:
        """登记关系三元组。

        判据用 entry.returns（算子**供应给业务侧**的东西）而不是
        operation.output_type（它在 IR 里产出的值）：CellSetIntersect 供应
        一个关系，但产出的值是格网集合本体。两者混用会让它悄悄不再登记关系。
        """

        if entry.returns is not GeoType.RELATION:
            return
        from ir import relation_from_operation

        subject, obj = _split_subject_object(operation.inputs)
        relation = relation_from_operation(
            subject=subject,
            predicate=entry.predicate or operation.op,
            obj=obj,
            time=_scope_of(operation, "time"),
            spatial_scope=_scope_of(operation, "spatial_scope"),
            sensitivity=operation.sensitivity or Sensitivity.INTERNAL,
            meta={"source_op": operation.op, "output_name": operation.output_name},
        )
        self.program.add_relation(relation)

    # ---------------- 返回收集 ----------------

    def _collect_returns(self, node: ast.FunctionDef) -> list[str]:
        out: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                out.append(_unparse(child.value))
        return out

    # ---------------- 工具 ----------------

    def _loc(self, node: ast.AST) -> dict[str, Any]:
        return {
            "file": self.filename,
            "line": getattr(node, "lineno", None),
            "col": getattr(node, "col_offset", None),
            "text": self._source_line(getattr(node, "lineno", None)),
        }

    def _source_line(self, lineno: int | None) -> str | None:
        if lineno is None or not 0 < lineno <= len(self.lines):
            return None
        return self.lines[lineno - 1].strip()

    def _warn_on_unused_sensitivity_keys(self) -> None:
        """显式敏感度声明必须命中输入；命中不了就报出来，不许静默丢。"""

        if not self.sensitivities:
            return
        declared = set(self.program.entity_inputs)
        unused = sorted(key for key in self.sensitivities if key not in declared)
        if unused:
            self._warn(
                DIAG_STATIC_ASSERT,
                f"sensitivities 里的 {unused} 没有匹配到任何输入，已被忽略",
                node=self.tree,
                cause="只有形参名（或已声明的值名）才能被敏感度声明命中。",
                suggestion="核对拼写；或确认这些名字确实出现在被分析的函数签名里",
            )

    def _error(self, code: str, message: str, *, node: ast.AST, **kwargs: Any) -> None:
        self.diagnostics.append(
            Diagnostic(code=code, severity="error", message=message, location=self._loc(node), **kwargs)
        )

    def _warn(self, code: str, message: str, *, node: ast.AST, **kwargs: Any) -> None:
        self.diagnostics.append(
            Diagnostic(code=code, severity="warning", message=message, location=self._loc(node), **kwargs)
        )


# --------------------------------------------------------------------------
# 顶层 API
# --------------------------------------------------------------------------


def parse_source(
    source: str,
    filename: str = "<source>",
    *,
    entry: str | None = None,
    type_hints: dict[str, GeoType] | None = None,
    sensitivities: dict[str, Sensitivity] | None = None,
) -> ParseResult:
    """解析源码字符串为 Geo-IR 程序。"""

    analyzer = FrontendAnalyzer(
        source, filename, type_hints=type_hints, sensitivities=sensitivities
    )
    return analyzer.run(entry=entry)


def parse_file(path: str, *, entry: str | None = None, **kwargs: Any) -> ParseResult:
    """解析 .py 文件为 Geo-IR 程序。"""

    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    return parse_source(source, filename=path, entry=entry, **kwargs)


# --------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover
        return f"<{type(node).__name__}>"


def source_or_empty(text: str) -> str:
    return text


def _rename_operation(operation: GeoOperation, name: str) -> GeoOperation:
    return GeoOperation(
        op=operation.op,
        inputs=operation.inputs,
        output_type=operation.output_type,
        output_name=name,
        params=operation.params,
        sensitivity=operation.sensitivity,
        location=operation.location,
        source_expr=operation.source_expr,
    )


def _set_sensitivity(operation: GeoOperation, sensitivity: Sensitivity) -> GeoOperation:
    return GeoOperation(
        op=operation.op,
        inputs=operation.inputs,
        output_type=operation.output_type,
        output_name=operation.output_name,
        params=operation.params,
        sensitivity=sensitivity,
        location=operation.location,
        source_expr=operation.source_expr,
    )


def _add_param(operation: GeoOperation, key: str, value: Any) -> GeoOperation:
    params = dict(operation.params)
    params[key] = value
    return GeoOperation(
        op=operation.op,
        inputs=operation.inputs,
        output_type=operation.output_type,
        output_name=operation.output_name,
        params=params,
        sensitivity=operation.sensitivity,
        location=operation.location,
        source_expr=operation.source_expr,
    )


def _split_subject_object(inputs: Sequence[str]) -> tuple[str, str]:
    """委托给 semantic：那里按命名约定判定宾语（`no_fly_zone` 无论在第几个
    参数都应是宾语），本模块不再自持一份更粗糙的位置规则。"""

    from semantic import split_subject_object

    return split_subject_object(inputs)


def _scope_of(operation: GeoOperation, key: str) -> str | None:
    """取作用域：显式参数优先，其次按命名约定推断，推断不出就留空。

    推断一律交给 semantic 的命名约定表。本模块以前自带一份
    "取第一个下划线之后的部分"的粗糙实现，会把 `public_set` 变成 `set`、
    `risk_factors` 变成 `factors`——那是编造出来的作用域，
    属最好留空也不能给的那类假数据。
    """

    params = operation.params or {}
    if params.get(key) is not None:
        return str(params[key])
    if key == "spatial_scope":
        from semantic import infer_spatial_scope

        return infer_spatial_scope(operation.inputs)
    return None


def _annot_to_geotype(annotation: str | None) -> GeoType | None:
    if not annotation:
        return None
    base = annotation.split("[")[0].strip()
    mapping = {
        "CellSet": GeoType.CELL_SET,
        "EntitySet": GeoType.ENTITY_SET,
        "QuantVector": GeoType.VECTOR,
        "TimeInterval": GeoType.TIME_INTERVAL,
        "Point": GeoType.POINT,
        "int": GeoType.SCALAR,
        "float": GeoType.SCALAR,
        "Scalar": GeoType.SCALAR,
        "bool": GeoType.BOOL,
        "GeoRelation": GeoType.RELATION,
    }
    return mapping.get(base)


#: 类型的"具体度"排序：后者比前者更具体，冲突时取更具体者
_TYPE_SPECIFICITY = {
    GeoType.UNKNOWN: 0,
    GeoType.SCALAR: 1,
    GeoType.BOOL: 1,
    GeoType.ENTITY_SET: 2,
    GeoType.CELL_SET: 3,
    GeoType.VECTOR: 3,
    GeoType.POINT: 3,
    GeoType.TIME_INTERVAL: 3,
    GeoType.RELATION: 4,
}


def _more_specific(current: GeoType | None, candidate: GeoType) -> GeoType:
    if current is None:
        return candidate
    if _TYPE_SPECIFICITY.get(candidate, 0) > _TYPE_SPECIFICITY.get(current, 0):
        return candidate
    return current


def _geotype_of_name(name: str) -> GeoType:
    try:
        return GeoType(name)
    except ValueError:
        return GeoType.UNKNOWN


#: 视为可互操作的类型对（格网集合的两种视图；标量与布尔同属数值）
_INTERCHANGEABLE = frozenset(
    {
        frozenset({GeoType.ENTITY_SET, GeoType.CELL_SET}),
        frozenset({GeoType.SCALAR, GeoType.BOOL}),
    }
)

#: 每种产出类型最接近的"替代算子"，用于链式类型错误时给出可执行建议
_ALTERNATIVES_FOR_TYPE: dict[GeoType, str] = {
    GeoType.CELL_SET: "CellSetIntersect / Intersects",
    GeoType.ENTITY_SET: "Intersects / Contains",
    GeoType.VECTOR: "WeightedSum（要求定长数值向量）",
    GeoType.TIME_INTERVAL: "TemporalOverlap",
    GeoType.SCALAR: "WeightedSum",
}


def _type_matches(actual: GeoType, expected: str) -> bool:
    """实际类型能否满足方言声明的类型。"""

    if str(actual) == expected:
        return True
    expected_type = _geotype_of_name(expected)
    if actual is GeoType.UNKNOWN or expected_type is GeoType.UNKNOWN:
        # 判不出来就不报错：宁可少报，不可误报（用户代码不允许被我们乱改）
        return True
    return frozenset({actual, expected_type}) in _INTERCHANGEABLE


def _nearest_ops_for_types(expected: GeoType, method_name: str) -> str | None:
    """类型不符时给出替代算子：优先按期望类型给，其次按名字相似度。"""

    typed = _ALTERNATIVES_FOR_TYPE.get(expected)
    if typed:
        return typed
    alternatives = _nearest_ops(method_name)
    return alternatives[0] if alternatives else None


def _default_sensitivity_for(geo_type: GeoType) -> Sensitivity:
    if geo_type is GeoType.SCALAR:
        return Sensitivity.PUBLIC
    return Sensitivity.SENSITIVE


def _coerce_sensitivity(value: Any) -> Sensitivity:
    """把显式敏感度声明统一成 Sensitivity（允许写字符串形式）。"""

    if isinstance(value, Sensitivity):
        return value
    try:
        return Sensitivity(str(value))
    except ValueError as exc:
        raise ValueError(
            f"无法把 {value!r} 解释为敏感级别；可用：{[s.value for s in Sensitivity]}"
        ) from exc


def _coerce_geotype(value: Any) -> GeoType:
    """把显式类型声明统一成 GeoType（大小写不敏感）。"""

    if isinstance(value, GeoType):
        return value
    text = str(value).strip()
    for member in GeoType:
        if member.value.lower() == text.lower() or member.name.lower() == text.lower():
            return member
    raise ValueError(
        f"无法把 {value!r} 解释为 Geo 类型；可用：{[t.value for t in GeoType]}"
    )


def _literal_value(node: ast.expr) -> Any:
    """取编译期字面量的值；非字面量返回 None。

    支持 `-5` 这类一元负号（AST 里是 UnaryOp，字面上仍是常量）。
    """

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _literal_value(node.operand)
        if isinstance(operand, (int, float)) and not isinstance(operand, bool):
            return -operand if isinstance(node.op, ast.USub) else operand
    return None


def _is_dynamic_module_control(node: ast.stmt) -> bool:
    """模块顶层的这条语句是否构成"动态控制流"。

    只有条件**依赖运行期数据**才算：`if __name__ == "__main__":` 是字面量
    比较、`try: import ...` 是导入兜底，两者都不该被当成密态分支误报。
    这是 _is_statically_decidable 在模块层的对应物，二者判据必须一致。
    """

    if isinstance(node, (ast.If, ast.While)):
        test = node.test
        if isinstance(test, ast.Constant):
            return False
        # __name__ / __file__ 之类由解释器给出的常量，编译期语义确定
        if isinstance(test, ast.Compare):
            operands = [test.left, *test.comparators]
            return not all(
                isinstance(x, ast.Constant)
                or (isinstance(x, ast.Name) and x.id.startswith("__"))
                for x in operands
            )
        return True
    # for / try / with 在函数体里的判据返回 False（见 _is_statically_decidable）；
    # 模块层同样不把它们当"动态分支"，避免与函数体口径分叉。
    return False


def _typed_arguments(
    call: ast.Call, entry: DialectEntry
) -> list[tuple[ast.expr, str]]:
    """把一次方言调用的实参按**类型声明顺序**配对成 (实参节点, 期望类型)。

    位置实参自然对号入座；命名实参必须按参数名回到它在 `geotypes` 里的
    位置——只 zip(call.args, geotypes) 会让 `geo.height_band(x=x, ...)`
    这类写法完全不被推断，x/y 便停留在默认的 EntitySet，
    进而报出"第 1 个参数类型是 EntitySet，但算子要求 Scalar"这种**误报**。
    """

    geotypes = list(entry.geotypes)
    if len(geotypes) != entry.arity:
        # arity 与 geotypes 长度不一致时不猜，退回位置配对
        return list(zip(call.args, geotypes))

    names = list(entry.value_params + entry.config_params)
    pairs: list[tuple[ast.expr, str]] = list(zip(call.args, geotypes))
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg not in names:
            continue
        index = names.index(keyword.arg)
        if index < len(geotypes):
            pairs.append((keyword.value, geotypes[index]))
    return pairs


def _is_statically_decidable(node: ast.stmt) -> bool:
    """判断控制流条件是否在编译期可判定（字面量 / True / False）。"""

    if isinstance(node, (ast.If, ast.While)):
        test = node.test
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        return False
    elif isinstance(node, (ast.Try, ast.AsyncWith)):
        return False
    else:
        return False
    return isinstance(test, ast.Constant) and isinstance(test.value, bool)


def _nearest_ops(name: str) -> list[str]:
    """给不认识的算子名推荐最接近的受支持算子。

    委托给 semantic.suggest_ops，避免别名表在两处漂移。
    """

    from semantic import suggest_ops

    return suggest_ops(name)


def _cost_hint_for(op: str) -> dict[str, Any] | None:
    from planner.registry import OPERATOR_REGISTRY

    rule = OPERATOR_REGISTRY.get(op)
    if rule is None:
        return None
    return {
        "representation": rule.representation,
        "backend": rule.backend,
        "security_level": rule.security_level,
        "estimated_cost": dict(rule.cost_profile),
    }
