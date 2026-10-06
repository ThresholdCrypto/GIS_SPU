# 位平面布局与打包（D3）：成本层落地

> 状态：**已落地（预测层）**，2026-10-06。实现 `planner/layout.py`，
> 测试 `tests/test_bitplane_layout.py`（23 条，含"拿课题产物逐项复算"）。
> **未落地**：打包电路本身——JAX 生成器与 SPU 执行路径一行未改。
> 因此本文只谈**密文条数**，不谈打包后的通信量或墙钟。

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

模型的**前提**是"通信量 ∝ 密文条数"。这条前提有实测支持：
`docs/MPC_BENCHMARK_PROTOCOL.md` §8.4 结论 3 实测到 ABY3 × `DistanceLE` 的通信量
对 K 线性，约 **16 B/元素**（`tests/test_bitplane_layout.py::TestMeasuredPremise` 直接读
`docs/mpc_comm_baseline.json` 核验：14–20 B/元素区间 + 线性比在 10% 内）。

但**据此外推"打包后通信量按同比例下降"仍属未验证**——打包电路还没写，
槽位打包与槽内归约的通信代价没有测过。所以 `LayoutPlan.caveats` 里始终带着这两句：

- 本项是结构代价（密文条数）预测，不是实测通信量；
- 通信量 ∝ 条数 的前提有实测支持，但外推属未验证。

## 5. 用法

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

## 6. 下一步（真正落地打包需要什么）

1. **槽位打包算子**：把 A 个属性（或跨候选的同一位）按位并进一个环元素，
   需要 `jax.numpy` 侧的移位/掩码原语——先过 SPU 能力核查（`shift_left` /
   `bitwise_and` 一类原语是否已被 `libspu` 适配）；
2. **槽内归约**：L1 的加权和、L2 的跨候选聚合要在槽内完成，
   需要掩码 + 旋转（或按位求和树），这条路径的通信量必须**重新实测**，
   不能沿用本文的条数比例；
3. **三份实现对拍**：打包后的明文/JAX/SPU 三份实现必须逐位一致，
   容差与未打包路径分开登记（打包引入的是位运算，不该有浮点误差）。
