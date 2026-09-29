"""JAX 后端：把隐私方案中的算子生成为可追踪的 jax.numpy 代码。

硬性约束（逐条对应课题要求）：
    1. 只用 jax.numpy（jnp），不引入任何 Python 运行期依赖；
    2. 函数体是纯表达式，保证 jax.jit 可追踪（静态形状、无 Python 分支）；
    3. 幂等：同一方案生成的源码与 IR 都不依赖调用时机；
    4. 不允许出现 Python 级 if/for/while —— 那是"动态控制流无法编译"的根源。

只对具备 JAX 实现的算子生成代码（DistanceLE / WeightedSum / TemporalOverlap）。
PSI 族算子（Intersects / Contains / CellSetIntersect）在 JAX 中没有逐元素原语：
交集性是组合问题，不是逐元素算子，故明确标注为不具备 JAX 实现，而不是硬凑。
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from planner.planner import PlannedStep, PrivacyPlan
from planner.registry import OPERATOR_REGISTRY, get_rule

# --------------------------------------------------------------------------
# 代码生成
# --------------------------------------------------------------------------


@dataclass
class GeneratedFunction:
    """一段生成的 JAX 代码。"""

    op: str
    name: str
    source: str
    inputs: tuple[str, ...] = ()
    signature: str = ""
    tolerance: float = 0.0
    notes: tuple[str, ...] = ()
    file_name: str = ""
    dtype: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "name": self.name,
            "signature": self.signature,
            "inputs": list(self.inputs),
            "tolerance": self.tolerance,
            "dtype": self.dtype,
            "notes": list(self.notes),
            "source": self.source,
        }


#: 每个算子的容忍度：定点/浮点误差的显式上限。
#: 判定型算子（DistanceLE / TemporalOverlap）输出布尔，整数路径下应为精确；
#: 但若被强制走浮点定点，量化误差会累积在比较边界，故给出边界裕度。
TOLERANCES: Mapping[str, float] = {
    "DistanceLE": 0.0,       # 整数平方和：精确
    "WeightedSum": 0.0,      # 定点整数乘加：精确
    "TemporalOverlap": 0.0,  # 整数区间比较：精确
}

#: 若上游把输入浮点化，允许的相对误差上限（用于容差测试）
FLOAT_TOLERANCE = 1e-4


def generate_distance_le(fn_name: str, *, dtype: str = "int32") -> GeneratedFunction:
    """生成 DistanceLE 的 JAX 实现。

    形态：平方和与阈值平方比较，无 sqrt，无分支。
    """

    source = textwrap.dedent(
        f'''
        import jax.numpy as jnp

        def {fn_name}(left, right, threshold):
            """距离是否不超过阈值。比较距离平方与阈值平方，避免 sqrt。"""
            delta = left - right
            dist_sq = jnp.sum(jnp.square(delta))
            return dist_sq <= jnp.square(threshold)
        '''
    ).strip()
    return GeneratedFunction(
        op="DistanceLE",
        name=fn_name,
        source=source,
        inputs=("left", "right", "threshold"),
        signature=f"({fn_name}(left: f32[N], right: f32[N], threshold: f32 scalar) -> bool)",
        tolerance=TOLERANCES["DistanceLE"],
        dtype=dtype,
        notes=(
            "无 sqrt：乘法深度 d=1，避开 MPC 除法电路",
            "jnp.sum 归约是 SPU 已适配的原语",
            "形状静态：N 由输入张量决定，无需 Python 分支",
        ),
    )


def generate_weighted_sum(fn_name: str, *, dtype: str = "int32") -> GeneratedFunction:
    """生成 WeightedSum 的 JAX 实现。"""

    source = textwrap.dedent(
        f'''
        import jax.numpy as jnp


        def {fn_name}(values, weights, scale):
            """定点加权和：sum(w * v) / scale。"""
            acc = jnp.sum(values * weights)
            return acc // scale
        '''
    ).strip()
    return GeneratedFunction(
        op="WeightedSum",
        name=fn_name,
        source=source,
        inputs=("values", "weights", "scale"),
        signature=f"({fn_name}(values: f32[K], weights: f32[K], scale: f32 scalar) -> f32 scalar)",
        tolerance=TOLERANCES["WeightedSum"],
        dtype=dtype,
        notes=(
            "定点乘加 d=1；位宽按 8+8+ceil(log2 K) 预留累加余量",
            "scale 作为参数传入而非闭包捕获，保证 jax.jit 可追踪",
        ),
    )


def generate_temporal_overlap(fn_name: str, *, dtype: str = "int32") -> GeneratedFunction:
    """生成 TemporalOverlap 的 JAX 实现。

    段式编码：节点展开为 [start, end) 后逐对比较，用向量化替代双重循环。
    """

    source = textwrap.dedent(
        f'''
        import jax.numpy as jnp


        def {fn_name}(left_toff, left_lt, right_toff, right_lt):
            """是否存在重叠时段。段式节点按 [Toff, Toff + 2^Lt) 展开后两两比较。"""
            one = jnp.asarray(1, jnp.int32)
            left_end = left_toff + jnp.left_shift(one, left_lt)
            right_end = right_toff + jnp.left_shift(one, right_lt)
            # 广播为 [N, M] 逐对比较：重叠判据 l.start < r.end 且 r.start < l.end
            overlapping = (left_toff[:, None] < right_end[None, :]) & (
                right_toff[None, :] < left_end[:, None]
            )
            return jnp.any(overlapping)
        '''
    ).strip()
    return GeneratedFunction(
        op="TemporalOverlap",
        name=fn_name,
        source=source,
        inputs=("left_toff", "left_lt", "right_toff", "right_lt"),
        signature=f"({fn_name}(left_toff: i32[N], left_lt: i32[N], right_toff: i32[M], right_lt: i32[M]) -> bool)",
        tolerance=TOLERANCES["TemporalOverlap"],
        dtype=dtype,
        notes=(
            "用位运算与向量比较替代双重循环，避免动态控制流",
            "jnp.all 归约；无需排序，故不触碰 SPU 的 sort 补丁路径",
            "节点必须按 2^Lt 对齐（审计发现 F3）",
        ),
    )


GENERATORS: Mapping[str, Any] = {
    "DistanceLE": generate_distance_le,
    "WeightedSum": generate_weighted_sum,
    "TemporalOverlap": generate_temporal_overlap,
}


@dataclass
class JaxGenerationResult:
    """一次 JAX 代码生成的结果。"""

    functions: list[GeneratedFunction] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[Any] = field(default_factory=list)

    def function_for(self, op: str) -> GeneratedFunction | None:
        for function in self.functions:
            if function.op == op:
                return function
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "functions": [f.to_dict() for f in self.functions],
            "skipped": list(self.skipped),
        }


def generate_for_plan(plan: PrivacyPlan, *, prefix: str = "geo") -> JaxGenerationResult:
    """为隐私方案的每个步骤生成 JAX 代码。

    PSI 族算子不生成代码，而是登记为 skipped 并说明原因——
    这比"生成了一个跑不通的函数"诚实得多。
    """

    result = JaxGenerationResult()

    for index, step in enumerate(plan.steps):
        generator = GENERATORS.get(step.operation)
        if generator is None:
            rule = OPERATOR_REGISTRY.get(step.operation)
            result.skipped.append(
                {
                    "operation": step.operation,
                    "representation": step.representation,
                    "backend": step.backend,
                    "reason": _skip_reason(step, rule),
                }
            )
            continue
        fn_name = f"{prefix}_{step.operation.lower()}_{index}"
        result.functions.append(generator(fn_name))

    return result


def _skip_reason(step: PlannedStep, rule: Any) -> str:
    if rule is not None and not rule.has_jax_impl:
        return (
            f"{step.operation} 属于 {step.backend} 族，"
            "在 jax.numpy 中没有逐元素对应原语；"
            "集合交是组合问题而非逐元素算子，无法用张量表达式表达。"
            f"该算子由 {step.backend} 直接执行，不经过 JAX 代码生成。"
        )
    return f"算子 {step.operation} 尚未登记 JAX 代码生成器"


# --------------------------------------------------------------------------
# 源码渲染
# --------------------------------------------------------------------------

GENERATED_HEADER = '''"""由 geo-secure 编译器自动生成的 JAX 代码。请勿手工修改。

生成依据：Geo-IR → Privacy Plan → JAX 代码生成
约束：仅使用 jax.numpy；无 Python 运行期依赖；保证 jax.jit 可追踪。
"""
'''


def render_module(result: JaxGenerationResult, *, source_plan: PrivacyPlan | None = None) -> str:
    """把生成结果渲染成一个 .py 模块文本。"""

    chunks = [GENERATED_HEADER]
    if source_plan is not None:
        chunks.append(f"# 源程序: {source_plan.program_name}")
        chunks.append(f"# 规划步骤数: {len(source_plan.steps)}")
        chunks.append("")

    for function in result.functions:
        chunks.append(function.source)
        chunks.append("")

    if result.skipped:
        chunks.append("# 以下算子不生成 JAX 代码（原因见注释）：")
        for item in result.skipped:
            chunks.append(f"#   - {item['operation']} [{item['backend']}]: {item['reason']}")
        chunks.append("")

    return "\n".join(chunks).rstrip() + "\n"


def static_check_source(source: str) -> list[str]:
    """对生成的源码做静态自检：确保没有禁止的模式。

    这是"编译前 capability check"的一部分，不依赖 jax 是否安装。
    """

    problems: list[str] = []
    tree_source = source

    if "import jax.numpy as jnp" not in tree_source:
        problems.append("生成的代码必须导入 jax.numpy as jnp")

    for banned in ("import numpy as np", "import numpy\n", "import spu", "import jax "):
        if banned in tree_source:
            problems.append(f"生成代码不得使用 {banned.strip()}（会引入运行期依赖）")

    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"生成代码存在语法错误: {exc}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for statement in node.body:
                if isinstance(statement, (ast.If, ast.For, ast.While)):
                    problems.append(
                        f"函数 {node.name} 含 Python 级 {type(statement).__name__}，"
                        "动态控制流无法被 jax.jit 追踪"
                    )
    return problems