# 位平面布局与打包（D3）：成本层落地

> 状态：**已落地（预测层 + 前提实测）**，2026-10-06。实现 `planner/layout.py`，
> 测试 `tests/test_bitplane_layout.py`（23 条，含"拿课题产物逐项复算"）；
> 打包**前提**（通信量按环元素计费、随元素数线性）已由 P6 探针实测，见 §5
> （产物 `docs/mpc_packing_probe.json`）。
> **未落地**：打包电路本身——JAX 生成器与 SPU 执行路径一行未改。
> 因此本文只谈**密文条数**与由实测单价换算的**上界**，不谈打包后的实测通信量。

## 1. 口径出处与复算

D3 的决策与数字都来自课题交付物，不是本项目自拟：

| 出处 | 内容 |
|---|---|
| `outputs/格网数据样例_明文与密态映射_v5.json` → `cost_prediction` | 场景的四个数（v6 与之逐项相同） |
| `outputs/三维格网接入层口径勘误与修正说明.docx` → 表 3-1 | 课题组自己的复算与验证方式 |
| `README.md` §8.4 | 本项目的落点说明（"layout planner + 把 N_ct 的下降量写回 `estimated_cost`"） |

场景形状：**1000 候选 × 49 缓冲格网 × 4 属性 × b = 8**。载明的四个数：

| 项 | 课题记载 | 本模块公式 | 复算 |
|---|---|---|---|
| 逐点值数 / 条数 | 196000 | `values = candidates × cells × attributes` | 1000×49×4 = 196000 ✓ |
| L1（同格网跨属性） | 49000 | `ceil(values / attributes)` | 196000/4 = 49000 ✓ |
| L2（跨格网位平面） | 192 | `ceil(values / slots) × bits` | ceil(196000/8192)×8 = 192 ✓ |
| L2 相对逐点下降 | 1020.8× | `values / L2` | 196000/192 = 1020.8 ✓ |

`slots`（每密文位平面槽位数）课题载明 **8192**。本模块不硬编这个数，而是复算出
同一条关系：

```
slots = 2^ceil(log2(candidates)) × bits = 下一 2 的幂(1000) × 8 = 1024 × 8 = 8192
```

`tests/test_bitplane_layout.py::TestReconcilesWithTheCaseStudyArtifact` 直接读该 JSON 逐项
核对上表——**不另抄一份数字**：抄出来的第二份没人再验，与代码各说各话时也发现不了。

## 2. 归约轴 → 布局

选布局的依据是算子的**归约方向**（README §8.4 的 D3 决策）：

| 算子 | 归约轴 | 布局 | 理由 |
|---|---|---|---|
| `WeightedSum` | 属性轴 | **L1**（同格网跨属性） | Σ_a w_a·x_a 就是在同一格网内跨属性归约，把 A 个属性并进一条密文即对齐归约方向 |
| `DistanceLE` | 候选轴 | **L2**（跨格网位平面） | 逐候选判定后要跨候选聚合（是否存在越界候选），位平面把同一属性位跨候选排布 |
| `TemporalOverlap` | 候选/节点轴 | **L2** | 事件排序归并 + 前缀扫描本身就沿节点轴 |
| `Intersects` / `Contains` / `CellSetIntersect` | 无 | 不适用 | 集合交由 PSI 承担，不经 MPC 值布局 |
| `HeightBand` | 无 | 不适用 | 明文物化，不进密态 |

`REDUCTION_AXIS` 与 `OPERATOR_REGISTRY` 的键集合由测试锁定：**新增算子时这里会红**，
逼着显式回答"它的归约轴是什么"，而不是让它默默按逐点计。

## 3. 三条边界（都是"不许编"的具体化）

1. **不给形状就不给数字**。没有 `LayoutShape` 时只输出轴向决策
   （"该走 L1/L2"），`selected_ciphertexts` / `reduction` 一律 `None`。
   没有一个"默认规模"可以顶替——K、格网数、属性数都随业务变。
2. **没收益就退回逐点**。L2 在极小规模上条数反而更多（如 1 个值：`ceil(1/8192)×8 = 8 > 1`）；
   L1 在属性数为 1 时也没有收益。两种都如实退回 `naive` 并写明原因，
   **不报小于 1 的"收益"**。
3. **未登记归约轴的算子不替它猜**。返回 `naive` + "未登记归约轴"的说明，
   不抛异常也不假装有布局。

## 4. 模型与实测的关系（只说站得住的那半句）

模型的**前提**是"通信量由密文条数（环元素个数）主导，输入位宽不影响单价"。
这条前提有**两路**实测支持：

- `docs/MPC_BENCHMARK_PROTOCOL.md` §8.4 结论 3 实测到 ABY3 × `DistanceLE` 的通信量
  对 K 线性，约 **16 B/元素**（`tests/test_bitplane_layout.py::TestMeasuredPremise` 直接读
  `docs/mpc_comm_baseline.json` 核验：14–20 B/元素区间 + 线性比在 10% 内）；
- **P6 打包探针**（§5）：秘密×秘密逐元素乘法在 int8 / int32 / int64 三档位宽、
  同一元素数下通信量**完全相同**，且 B/元素 在 512 / 1024 / 4096 三档规模上恒定。

但**把"元素数 ÷8"直接读成"打包后通信量 ÷8"仍只是上界**——打包电路还没写，
槽内归约的通信代价没有测过。所以 `LayoutPlan.caveats` 里始终带着这两句：

- 本项是结构代价（密文条数）预测，不是实测通信量；
- 通信量 ∝ 条数 的前提已实测，但打包电路未实现，此处收益只能是**上界**。

## 5. 打包前提实测（P6 探针）

产物 `docs/mpc_packing_probe.json` / `.csv`，代码 `backends/spu_backend/packing_probe.py`：

```bash
~/.spuenv/bin/python tests/benchmarks/benchmark_mpc.py --packing-probe --repeat 3
```

**探针算子**是"秘密 × 秘密"的逐元素乘法。（第一版用 `x * 2`——乘公开常数，实测
通信量恒为 **0 字节**：SPU 把它编译成本地线性运算。探针因此换成真会通信的两方
乘法；这条教训写进了 `tests/test_packing_probe.py`，防止回退。）

实测（ABY3 / FM64，3 次重复取中位数）：

| 用例 | 元素数 | 通信量 | B/元素 |
|---|---|---|---|
| `mul_ss int8  N=4096` | 4096 | 65 536 B | 16.00 |
| `mul_ss int32 N=4096` | 4096 | 65 536 B | 16.00 |
| `mul_ss int64 N=4096` | 4096 | 65 536 B | 16.00 |
| `mul_ss int64 N=512` | 512 | 8 192 B | 16.00 |
| `mul_ss int64 N=1024` | 1024 | 16 384 B | 16.00 |

三条结论：

1. **按环元素计费，不按输入位**：int8 / int32 / int64 在 N=4096 上通信量之比全部为
   1.000（逐原语明细里唯一通信的原语是 `multiply`）。把 8 个 8 位值装进 1 个 64 位
   环元素**不额外收费**——打包的收益空间由此成立；
2. **随元素数线性**：512 → 1024 → 4096 的 B/元素 恒为 16.00（相对离散度 0），
   所以"元素数 ÷8"确实对应"通信量 ÷8"这个**上界**；
3. **与既有基线互相印证**：P3 的 `docs/mpc_comm_baseline.json` 里 ABY3 ×
   `DistanceLE` 也是 16 B/元素——不同算子、不同输入构造、不同测量脚本，落到同一
   单价。

**它不证明什么**（不许越界）：探针跑的是逐元素乘法，**不是打包电路**。槽内归约
（L1 的加权和、L2 的跨候选聚合）要额外通信，那部分必须另行实测。所以
`planner/layout.py` 给出的收益一律标注为**上界**。

## 6. 用法

```bash
# 只做轴向决策（不预测条数）
geo-secure build examples/distance_check.py

# 给出规模形状 → 把条数下降写回 estimated_cost
geo-secure build examples/distance_check.py \
    --layout-shape candidates=1000,cells=49,attributes=4,bits=8
```

输出（Privacy planning 段）：

```
  MPC 协议: DistanceLE → ABY3（按实测代价自动选择）
  位平面布局: DistanceLE → L2（逐点 196000 条 → 192 条，1020.8×）
```

Python API：

```python
from planner import LayoutShape, plan_layout, plan_program
plan_layout("WeightedSum", LayoutShape(1000, 49, 4, 8))   # → L1，49000 条
plan_program(program, layout_shape=LayoutShape(1000, 49, 4, 8))
```

写进 `PlannedStep.estimated_cost` 的键（与四量 `N_ct` / `b` / `d` / `R` 并存，
**不覆盖**它们——`N_ct` 是"每元素条数"的描述串，这里是**总条数**）：

| 键 | 含义 |
|---|---|
| `layout` | 选中的布局：`L1` / `L2` / `naive` |
| `layout_axis` | 归约轴：`attribute` / `candidate` / `None` |
| `layout_basis` | 一句话依据（含"未给形状则不预测条数"） |
| `N_ct_naive` / `N_ct_layout` / `layout_reduction` | 仅在给了形状时出现 |

## 7. 下一步（真正落地打包需要什么）

1. **槽位打包算子**：把 A 个属性（或跨候选的同一位）按位并进一个环元素，
   需要 `jax.numpy` 侧的移位/掩码原语——先过 SPU 能力核查（`shift_left` /
   `bitwise_and` 一类原语是否已被 `libspu` 适配）；
2. **槽内归约**：L1 的加权和、L2 的跨候选聚合要在槽内完成，
   需要掩码 + 旋转（或按位求和树），这条路径的通信量必须**重新实测**，
   不能沿用本文的条数比例；
3. **三份实现对拍**：打包后的明文/JAX/SPU 三份实现必须逐位一致，
   容差与未打包路径分开登记（打包引入的是位运算，不该有浮点误差）。
