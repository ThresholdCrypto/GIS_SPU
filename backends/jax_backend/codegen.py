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


def generate_weighted_sum(
    fn_name: str, *, scale: int = 1, dtype: str = "int32"
) -> GeneratedFunction:
    """生成 WeightedSum 的 JAX 实现。

    `scale` 是**编译期常量**，不是运行时输入。理由（都有实测依据，
    见 `docs/MPC_BENCHMARK_PROTOCOL.md` §4.2 / §4.3）：

    - SPU 的整数除法是迭代近似实现（`div_goldschmidt`），`scale == 1` 也不是
      恒等映射：K≥1024 实测与明文不一致（偏差率 43%–93%），而 `jnp.sum(w*v)`
      本身精确且确定；
    - 除法路径内部要求 **64 位环**：`WeightedSum × FM32` 直接报
      `ring=FM32 could not represent PT_I64`，起不来；
    - 代价：`//` 在 HLO 里展开为 divide + remainder + select + sign，
      实测占该算子 PPHLO 字节数的 61%、通信量的 56%（ABY3, K=256）。

    三种取值三条路径：

    - `scale == 1`（管线里的唯一路径）→ 直接返回累加和，**电路里没有除法**；
    - `scale == 2^s` → 右移代替整除，精确；
    - 其它 → 保留整除，并在 notes 里明确写出"SPU 上是近似除法、需要 ≥FM64"。
    """

    scale = int(scale)
    if scale < 1:
        raise ValueError("scale 必须为正整数（定点缩放因子）")

    if scale == 1:
        body = "return acc"
        scale_note = "scale=1：直接返回累加和（无除法电路）"
    elif scale & (scale - 1) == 0:
        shift = scale.bit_length() - 1
        body = f"return jnp.right_shift(acc, {shift})"
        scale_note = f"scale={scale}=2^{shift}：右移代替整除，精确"
    else:
        body = f"return acc // {scale}"
        scale_note = (
            f"scale={scale} 不是 2 的幂：保留整除——SPU 上为近似除法"
            "（结果可能与明文不一致），且除法路径需要 ≥FM64"
        )

    source = textwrap.dedent(
        f'''
        import jax.numpy as jnp


        def {fn_name}(values, weights):
            """定点加权和：sum(w * v)，scale 已折进生成代码。"""
            acc = jnp.sum(values * weights)
            {body}
        '''
    ).strip()
    return GeneratedFunction(
        op="WeightedSum",
        name=fn_name,
        source=source,
        inputs=("values", "weights"),
        signature=f"({fn_name}(values: f32[K], weights: f32[K]) -> f32 scalar)",
        tolerance=TOLERANCES["WeightedSum"],
        dtype=dtype,
        notes=(
            "定点乘加 d=1；位宽按 8+8+ceil(log2 K) 预留累加余量",
            "scale 是编译期常量（不占运行时输入），因此 jax.jit 可追踪；"
            "与 Geo-IR 的 2 输入形态一致",
            scale_note,
        ),
    )


#: TemporalOverlap 的电路形态：
#:   - `pairwise`：广播成 [N, M] 逐对比较，O(N·M)；
#:   - `sweep`   ：事件排序归并 + 前缀扫描，O((N+M)·log(N+M))；
#:   - `auto`    ：按规模在两者间选（见 TEMPORAL_OVERLAP_SWEEP_MIN_SIZE），
#:                 拿不到规模时退回 `pairwise`（保守：不改动既有成本口径）。
TEMPORAL_OVERLAP_STRATEGIES: tuple[str, ...] = ("auto", "pairwise", "sweep")

#: `auto` 的切换阈值：节点数 ≥ 此值时用 `sweep`。
#:
#: 依据是**实测的成对 A/B**（ABY3/FM64，×5 中位 + 区间，
#: `docs/mpc_temporal_ab.json`）：排序本身在 MPC 里很贵，所以扫描版只有在
#: 规模足够大时才划算——
#:
#: | K | pairwise 通信量 | sweep 通信量 | 谁更省 |
#: |---:|---:|---:|---|
#: | 8 | 95 872 B | 317 968 B | pairwise |
#: | 32 | 496 896 B | 1 265 424 B | pairwise |
#: | 64 | 1 296 896 B | 2 497 552 B | pairwise |
#: | 128 | 3 853 312 B | 5 438 480 B | pairwise |
#: | 256 | 12 546 048 B | **10 889 232 B** | sweep |
#:
#: 交叉点落在 128–256 之间（K=256 上同一次运行内两侧区间不重叠），取 256 作为
#: 阈值：只在实测确认更省的一侧切换。跨运行有 8% 量级漂移（sweep 抖动更大，
#: 见 `docs/MPC_BENCHMARK_PROTOCOL.md` §4.1），故阈值是保守取整，不是精确门限。
TEMPORAL_OVERLAP_SWEEP_MIN_SIZE = 256

#: sweep 形态的函数体（未缩进；组装时整体缩进到函数体内）
_TEMPORAL_OVERLAP_SWEEP_BODY = textwrap.dedent(
    """
    one = jnp.asarray(1, jnp.int32)
    zero = jnp.asarray(0, jnp.int32)
    left_end = left_toff + jnp.left_shift(one, left_lt)
    right_end = right_toff + jnp.left_shift(one, right_lt)
    # 事件打包成一个整数一起排序：(pos << 2) | (is_start << 1) | side
    #   is_start=1 起点 / 0 终点；side=1 左侧 / 0 右侧。
    # 起终点占**高位** ⇒ 同一位置上所有终点排在所有起点之前，
    # 半开区间 [0,2) 与 [2,4) 因此不会被误判成重叠。
    left_events = jnp.concatenate(
        [
            jnp.left_shift(left_toff, 2) + 3,
            jnp.left_shift(left_end, 2) + 1,
        ]
    )
    right_events = jnp.concatenate(
        [
            jnp.left_shift(right_toff, 2) + 2,
            jnp.left_shift(right_end, 2),
        ]
    )
    keys = jnp.sort(jnp.concatenate([left_events, right_events]))
    is_start = jnp.bitwise_and(jnp.right_shift(keys, 1), one)
    side = jnp.bitwise_and(keys, one)
    delta = jnp.where(is_start == 1, one, -one)
    active_left = jnp.cumsum(jnp.where(side == 1, delta, zero))
    active_right = jnp.cumsum(jnp.where(side == 1, zero, delta))
    return jnp.any((active_left > 0) & (active_right > 0))
    """
).strip()

#: pairwise 形态的函数体（P3 之前的形态，保留作 A/B 基线与 oracle）
_TEMPORAL_OVERLAP_PAIRWISE_BODY = textwrap.dedent(
    """
    one = jnp.asarray(1, jnp.int32)
    left_end = left_toff + jnp.left_shift(one, left_lt)
    right_end = right_toff + jnp.left_shift(one, right_lt)
    # 广播为 [N, M] 逐对比较：重叠判据 l.start < r.end 且 r.start < l.end
    overlapping = (left_toff[:, None] < right_end[None, :]) & (
        right_toff[None, :] < left_end[:, None]
    )
    return jnp.any(overlapping)
    """
).strip()

_TEMPORAL_OVERLAP_NOTES: Mapping[str, tuple[str, ...]] = {
    "sweep": (
        "排序归并 + 前缀扫描：代价 O((N+M)·log(N+M))，不物化 [N, M] 比较矩阵",
        "事件打包进单个整数，归并由排序隐含完成（无 Python 级控制流）",
        "排序在 MPC 里代价高：K<256 时比逐对比较更贵（实测交叉点，见 §4.1）",
        "节点必须按 2^Lt 对齐（审计发现 F3）",
    ),
    "pairwise": (
        "广播成 [N, M] 逐对比较：代价 O(N·M)；K≥256 起换 sweep 电路更省",
        "节点必须按 2^Lt 对齐（审计发现 F3）",
    ),
}


def generate_temporal_overlap(
    fn_name: str,
    *,
    dtype: str = "int32",
    strategy: str = "auto",
    size_hint: int | None = None,
) -> GeneratedFunction:
    """生成 TemporalOverlap 的 JAX 实现。

    两种电路形态（都无 Python 级控制流、都只用 jnp）：

    - `pairwise`：广播成 [N, M] 逐对比较，O(N·M)。**K=4096 会展开 1 670 万
      元素、实测直接拖死进程**，这是"K 上限被锁死"的来源；
    - `sweep`：把节点的起止点打包成事件、**排序归并**后做前缀扫描，
      两侧活跃数同时 > 0 即存在重叠，O((N+M)·log(N+M))，不物化 [N, M] 矩阵。

    `strategy="auto"`（默认）按 `size_hint` 在两者间选：≥
    `TEMPORAL_OVERLAP_SWEEP_MIN_SIZE` 用 `sweep`，否则（拿不到规模时也）用
    `pairwise`。切换阈值来自实测交叉点，不是拍的——见常量处的对照表。

    Args:
        size_hint: 调用方声明的节点数（N 与 M 取较大者即可）。仅
            `strategy="auto"` 用；给不出就保持既有（pairwise）成本口径。
    """

    if strategy not in TEMPORAL_OVERLAP_STRATEGIES:
        raise ValueError(
            f"未知 TemporalOverlap 电路形态 {strategy!r}；"
            f"可用：{TEMPORAL_OVERLAP_STRATEGIES}"
        )

    if strategy == "auto":
        strategy = (
            "sweep"
            if size_hint is not None and int(size_hint) >= TEMPORAL_OVERLAP_SWEEP_MIN_SIZE
            else "pairwise"
        )

    if strategy == "sweep":
        body = _TEMPORAL_OVERLAP_SWEEP_BODY
    else:
        body = _TEMPORAL_OVERLAP_PAIRWISE_BODY
    source = (
        "import jax.numpy as jnp\n\n\n"
        f"def {fn_name}(left_toff, left_lt, right_toff, right_lt):\n"
        '    """是否存在重叠时段。节点按 [Toff, Toff + 2^Lt) 展开（半开区间）。"""\n'
        + textwrap.indent(body, "    ")
        + "\n"
    )
    return GeneratedFunction(
        op="TemporalOverlap",
        name=fn_name,
        source=source,
        inputs=("left_toff", "left_lt", "right_toff", "right_lt"),
        signature=f"({fn_name}(left_toff: i32[N], left_lt: i32[N], right_toff: i32[M], right_lt: i32[M]) -> bool)",
        tolerance=TOLERANCES["TemporalOverlap"],
        dtype=dtype,
        notes=_TEMPORAL_OVERLAP_NOTES[strategy],
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
