# geo-secure：面向地理信息行业的低门槛隐私计算编译器（MVP）

让地理业务开发者**不需要学 JAX、SPU 或 MPC**。

开发者只写普通 Python 地理业务逻辑：

```python
from geo_privacy import geo

def check_conflict(route, no_fly_zone):
    return geo.intersects(route, no_fly_zone)
```

编译器自动完成：

```
Python 地理业务代码
  → 地理语义识别（frontend）
  → Geo-IR（ir）
  → 隐私计算算子选择（planner）
  → JAX 代码生成（backends/jax_backend）
  → SPU 模拟验证（backends/spu_backend）   ← 本次已真实跑通
  → 后续可扩展到 PSI / FHE / TEE
```

> **项目进展说明**（完成度、里程碑时间线、已验证 / 未落地、关键实测结论、复现方式）
> 见 [`docs/PROGRESS.md`](docs/PROGRESS.md)。当前 `main` 共 **28 个提交**、
> **1151 项测试全部通过、0 跳过**（2026-10-10 于 WSL2 + spu 0.9.5 复跑）。

## 快速开始

**第一步（任意环境，零依赖）**：

```bash
python -m geosecure.cli build examples/route_conflict.py
python -m geosecure.cli ops        # 算子注册表
python -m geosecure.cli check      # SPU / JAX 能力核查
```

**第二步（真实 SPU 验证，需 WSL2 / Linux + Python 3.10 或 3.11）**：

```bash
bash scripts/setup_wsl_spu.sh         # 一键：系统依赖 + Python 3.11 + 装 SPU + 全部验证
```

等价手工步骤：

```bash
pip install -r requirements-spu.txt   # spu==0.9.5 / jax<=0.4.34 / numpy<2
python -m pytest tests/ -q            # 1151 项全部通过（0 跳过）
python -m geosecure.cli build examples/distance_check.py
```

> 系统必须有 `libgomp1`（libspu 的 OpenMP 运行时），否则 `import spu` 即报
> `ImportError: libgomp.so.1`。`scripts/setup_wsl_spu.sh` 会自动安装。

验证强度现已到达端到端：实际在 **WSL2 + Python 3.11.16 + jax 0.4.34 +
spu 0.9.5** 上跑通 `Simulator.simple` + `sim_jax`，三个可生成 JAX 的算子
**实跑误差均为 0.0**（详见 §6）。

安装为命令后也可用 `geo-secure build examples/route_conflict.py`。

---

## 1. 架构

### 1.1 目录结构

```
GIS_SPU/
├── geo_privacy/            用户侧 API（业务开发者唯一接触的包）
│   ├── geo.py              geo 门面：六个地理算子 + 一个物化算子的标记与明文参考语义
│   └── core.py             CellSet / QuantVector / TimeInterval / quantize
├── frontend/               源码 → Geo-IR
│   ├── analyzer.py         基于 AST 的静态分析（从不执行用户代码）
│   └── dialect.py          geo.<method> → 算子的方言表
├── semantic/               地理语义识别
│   └── semantics.py        谓词表、作用域推断、敏感度规则、算子纠错
├── ir/                     Geo-IR
│   ├── types.py            GeoType / Sensitivity
│   ├── values.py           GeoEntity / GeoValue / GeoRelation + 格网编码
│   ├── operations.py       GeoOperation / GeoProgram
│   └── render.py           可读渲染
├── planner/                隐私计算方案
│   ├── registry.py         Operator Registry（算子→表征→后端）+ MPC 协议候选与选择
│   ├── layout.py           位平面布局（D3）：归约轴 → L1 / L2
│   └── planner.py          Planner，输出五元组
├── backends/
│   ├── plain/              明文参考实现（语义基准）
│   ├── jax_backend/        代码生成 + 追踪验证 + SPU 编译桥
│   │   ├── codegen.py      三个算子的 JAX 代码生成器
│   │   ├── reference.py    与生成代码逐行对应的 jax.numpy 实现
│   │   ├── runner.py       jax.jit 可追踪性检查与执行
│   │   └── spu_bridge.py   HLO 降级（SPU 编译前端第一步，无需 libspu）
│   ├── spu_backend/
│   │   ├── capability.py   能力探测与原语核查
│   │   ├── cost_baseline.py 读实测通信量基线（docs/mpc_comm_baseline.json）
│   │   ├── packing_probe.py 打包前提探针（按环元素计费？）与收益上界
│   │   ├── slot_cost_probe.py 取槽步单价探针（掩码/右移 vs 纯乘法）
│   │   └── runtime.py      run_spu_simulation
│   ├── psi_backend/        PSI（官方 spu.psi.psi_execute 两方求交）
│   │   ├── capability.py   协议/曲线/IO 形态探测（运行期，不预设版本）
│   │   └── runtime.py      run_psi_intersection / run_psi_operation
│   ├── psi_ca_backend/     PSI-Cardinality 计数档（OpenMined PSI，只出交集基数；
│   │   │                   与 psi_backend 并列的第二条 PSI 路径，泄漏承诺不同）
│   │   ├── capability.py   环境与 API 形态核对（运行期，不预设版本）
│   │   └── runtime.py      run_psi_cardinality（只承接 CellSetIntersect 计数）
│   └── psi_sum_backend/    PI-Sum 交集内求和（private-join-and-compute，只出
│       │                   「基数 + 交集内关联值之和」；第三条 PSI 路径）
│       ├── capability.py   构建产物 + flag 形态核对（运行期跑 --help，不预设）
│       └── runtime.py      run_psi_intersection_sum（只承接 CellSetIntersect）
├── validator/              编译前验证 + 六类失败模式报告
├── geosecure/              编译器主流程与 CLI（八阶段）
├── examples/               九个示例（含三维 vertical_conflict.py / altitude_band.py）
├── tests/                  三十九个测试文件
└── docs/
    ├── SPU_CAPABILITY.md   SPU 核对结论（含证据出处）
    ├── BITPLANE_LAYOUT.md  位平面布局（D3）的口径出处、模型与边界
    ├── PSI_CAPABILITY.md   PSI 核对结论 + 实测坑与泄漏面
    └── PSI_SUM_CAPABILITY.md  PI-Sum 核对结论、拒绝清单与复跑命令
```

### 1.2 数据流

```
源码字符串
   │  ast.parse（只解析，不执行）
   ▼
frontend/analyzer.py ── Diagnostic（可定位错误）
   │  识别 geo.<op>(...)，从方言反推参数类型
   ▼
GeoProgram { GeoValue 输入, GeoOperation 序列, GeoRelation 关系 }
   │  谓词、作用域、敏感度由 semantic 决定
   ▼
planner/planner.py ── PlannedStep（五元组）
   │  算子 → 计算表征 → 隐私后端
   ▼
backends/jax_backend/codegen.py ── 纯 jnp 函数源码
   │  static_check_source → check_traceable → lower_to_hlo
   ▼
backends/spu_backend/runtime.py ── run_spu_simulation
   │  能力门控 → Simulator.simple → sim_jax → 容差对拍
   ▼
Operator Status 表
```

### 1.3 分层原则

- **`geo_privacy` 不导入 jax / spu / secretflow**，可脱离隐私计算栈独立运行。
- **Geo-IR 不承载坐标与浮点**：空间一律格网编码，属性一律量化整数。
- **`validator` 只报告、不改代码**。编译器的诊断是建议，不是补丁。
- **SPU API 全部经 `docs/SPU_CAPABILITY.md` 核对后才使用**，不猜测。

---

## 2. 支持的算子

| 算子 | 方言写法 | 语义 | 输出 |
|---|---|---|---|
| Intersects | `geo.intersects(a, b)` | 两个格网集合是否存在公共格网 | Relation |
| Contains | `geo.contains(a, b)` | a 是否包含 b 的全部格网 | Relation |
| DistanceLE | `geo.distance_le(p, q, t)` | 量化向量距离是否 ≤ 阈值 | Relation |
| CellSetIntersect | `geo.cellset_intersect(a, b)` | 格网集合求交，返回交集本体 | Relation |
| WeightedSum | `geo.weighted_sum(v, w)` | 定点加权和 | Scalar |
| TemporalOverlap | `geo.temporal_overlap(a, b)` | 两个段式时间区间是否重叠 | Relation |
| HeightBand | `geo.height_band(x=, y=, ...)` | 把 (XY 单元, 高度带) 物化成 3D 层集合（**明文**，不进密态） | CellSet |

别名（便于业务写法更自然）：`intersect` / `within` / `near` / `score` /
`time_overlap` / `cell_intersect` 等，见 `semantic/semantics.py::ALIAS_TABLE`。

`HeightBand` 与上面六个**不是一类**：它是**物化算子**——在本方明文把高度带
展开成格网码集合（每层一个 64 位码），不经过任何密态后端。登记它的理由是
让"高度带参与判定"在 Geo-IR 里可见，详见 §7.1.1。
另有明文工具 `geo.quantize()`：返回普通标量，**不是算子**，不进 Geo-IR。

用户**不写类型注解**也可以：参数类型由调用的算子反推（见 §3.4）。

---

## 3. Geo-IR 格式

### 3.1 类型系统

```python
GeoType:      EntitySet | CellSet | Vector | Point | TimeInterval
              | Scalar | Relation | Bool | Unknown
Sensitivity:  public | internal | sensitive | secret
```

敏感度决定 planner 的下限：`sensitive` 及以上必须进密态。
判定门槛常量：`ir.CRYPTO_REQUIRED_LEVEL`。**默认值是保守的**
（地理数据默认 `sensitive`，未知情况归 `internal`）。

### 3.2 数据模型

```python
GeoEntity(name, entity_type, sensitivity, cell_codes, attrs,
          time_segments, spatial_scope, temporal_scope, meta)
GeoValue(name, geo_type, sensitivity, const, entity, source, is_constant)
GeoRelation(subject, predicate, object, time, spatial_scope,
            sensitivity, confidence, meta)
```

`GeoRelation` 的六个字段是**必填语义**，`relation_from_operation()` 是唯一构造入口。

### 3.3 算子与程序

课题指定的构造签名原样可用：

```python
GeoOperation(op="Intersects", inputs=["route", "no_fly_zone"], output_type="Relation")
```

`GeoProgram` 持有输入、算子序列、关系列表、返回值：

```python
program.declare_input("route_A", sensitivity=Sensitivity.SENSITIVE)
program.add_operation(GeoOperation(...))
program.add_relation(relation_from_operation("Route_A", "Intersects", "NoFlyZone_B"))
```

### 3.3.1 表达式级调用识别（嵌套调用也算）

frontend 不要求用户把每个算子调用都单独赋值给变量。识别按**整棵表达式树**做，
内层算子先发射、外层算子再引用内层的输出名：

```python
# 这两种写法给出同一套 Geo-IR 与同一套安全结论
return geo.intersects(geo.cellset_intersect(a, b), c)     # 嵌套

hit = geo.cellset_intersect(a, b)                          # 显式变量
return geo.intersects(hit, c)
```

覆盖的写法：`return`、赋值（含注解赋值）、裸表达式语句、`tuple` / `list` / `dict`
字面量、以及 `if` / `while` 条件表达式内部。

**为什么这件事必须是硬要求**：Geo-IR 的算子若被丢掉，它的密态输入就不再参与
敏感度判定，planner 会把该算子当成"只接触公开数据"而放进明文路径。
这不是少报一行日志，是真实的隐私泄漏——由
`tests/test_frontend.py::TestNestedCallRecognition` 与
`TestSensitivityIsNeverSilentlyDowngraded` 锁定。

没有变量承载的算子结果会拿到**确定且唯一**的合成名
（`geo_<op小写>_<序号>`）；唯一性由测试锁定，因为重名会让下游算子
通过 `GeoProgram.lookup` 解析到同名的**另一个**结果，进而把敏感度算错。

### 3.4 类型推断

参数类型按优先级确定：

1. `parse_source(..., type_hints={...})` 显式给出
2. 源码注解 `def f(route: CellSet)`
3. **从方言反推**：`geo.distance_le(p1, p2, t)` 说明 `p1, p2` 是 Vector、`t` 是 Scalar
4. 默认 `EntitySet`

第 3 条是"低门槛"的关键：用户不写注解也能被正确编译。
多处调用冲突时取**更具体**的类型；真的矛盾时报类型错误（不静默取值）。

**产出的类型 vs 供应的类型。** 方言表里 `returns` 记的是算子**供应给业务侧**
的东西（例如 `CellSetIntersect` 供应一个关系），而 `output_geo_type` 记的是它在
Geo-IR 里**产出**的值类型（格网集合本体）。下游算子按后者检查类型：

```python
cells = geo.cellset_intersect(a, b)      # output_geo_type = CellSet
geo.weighted_sum(cells, weights)         # 报错：要求 Vector，实得 CellSet
```

旧版一律按 `returns` 记成 `Relation`，且中间结果完全不检查类型，
于是"格网集合喂给定点向量"这类链式错误无诊断通过。现由
`tests/test_frontend.py::TestChainedResultTypes` 锁定。

**无法解析的实参按 SECRET 计入。** 遇到表达式实参（切片、下标、非方言调用）
或未知名字时，frontend 报 `GEO_INPUT_UNRESOLVED` 诊断，并让该输入以最敏感级别
参与判定。判不出来与判成公开是两回事：把前者当后者会静默降级。
反过来，字面量实参会被登记为 **PUBLIC 常量**，否则纯明文任务会被误判必须进密态。

### 3.5 格网编码口径

```
grid_code = X'17 | Y'17 | Z7 | L5 | Toff14 | Lt4  = 64 bit
time_tree_origin = 2026-09-06T22:00:00+08:00
dt_code = 1 min,  Lt_max = 14
```

各分量语义（本版补齐 `Z` 与 `L` 的口径，此前只作为位域存在）：

| 分量 | 位宽 | 语义 |
|------|------|------|
| `X'` / `Y'` | 17 / 17 | 面片因子化后的水平坐标残差（11 位分区键外提） |
| `Z` | 7 | **GB/T 40087-2021 附录 B 的高度层号** `height_index(大地高, L)` |
| `L` | 5 | 剖分层级。位域可表示 0–31（本版由 4 位扩到 5 位，见 7.1.1） |
| `Toff` | 14 | 相对时间树原点的分钟偏移 |
| `Lt` | 4 | 二叉时间树的节点层级，`dt = 2^Lt` 分钟 |

上版的 `ver`（1 位，恒为 0）已并入 `L` 高位，不再是独立位段。

`Z` 是**层号**而非米值；层号到物理高度是等比剖分（层厚随层号增长，不是常数）：

```
H(h, L) = r0 * ((1 + theta0)^(h * cell_deg(L)) - 1)      式(B.4)
h(H, L) = floor( ln(1 + H / r0) / ln(1 + theta0) / cell_deg(L) )   式(B.7)
```

公式实现见 `ir/geosot.py`，符合性由 `tests/test_geosot.py` 对拍国标特征值锁定。

本版把 L 位域由 4 位扩到 5 位（见 7.1.1），码位重排，
**所有已生成的码都变了**。新布局下由测试锁定：

```
encode_grid_code(x=21861, y=27702, z=15, level=9, toff=2160, lt=4)
  == 0x2AB29B0D87A48704   # L4 布局下的旧值是 0x2AB29B0D87C90E08
```

> **外部样例已重算为 v6**：接入层已定版采用 L5，交付样例
> `outputs/格网数据样例_明文与密态映射_v6.json`（`supersedes: v5`）已按新布局重算，
> 其中 `grid_code.hex` 与 `grid_code.segments` 都不含 `ver`。
> 上版 `outputs/格网数据样例_明文与密态映射_v5.json` 保留作 **L4 基线**，
> 供双方对拍新旧差异，不再是现行交付件。
> 给对端的变更说明见 `outputs/键布局对齐确认单_蚂蚁密算.md`。
> 历史报告 `outputs/阶段三改进报告_高度层Z语义接入与位域容量约束.md` 仍是 L4 布局下的
> 产物，作为阶段记录保留，其中的码值**不可用于对拍**。

### 3.6 渲染

```python
render_program(program)      # 整程序
render_relations(relations)  # 关系表
render_operations_table(ops) # 算子表
```

---

## 4. Privacy Planner 规则

### 4.1 默认注册表

| 算子 | 表征 | 后端 | 安全级别 |
|---|---|---|---|
| Intersects | CompactCellSet | PSI | high |
| Contains | CompactCellSet | PSI/MPC | high |
| DistanceLE | QuantizedVector | MPC/SPU | high |
| WeightedSum | FixedPointVector | SPU/MPC | medium |
| TemporalOverlap | TimeInterval | MPC/SPU | high |
| CellSetIntersect | CompactCellSet | PSI | high |
| HeightBand | CompactCellSet | Plaintext | low |

后端字段里的顺序表示优先级：`PSI/MPC` 表示首选 PSI，退化时用 MPC。

`HeightBand` 的 `Backend = Plaintext` 是有意的：层集合的编码在本方完成，
不进任何密态后端。它出现在规划表里，是为了让代价账和最终状态表**完整**——
看到 `plaintext-local` 就知道这一步没有也不需要有密态执行。

### 4.2 输出五元组

每个算子输出：

```python
{
  "operation":       "DistanceLE",
  "representation":  "QuantizedVector",
  "backend":         "MPC/SPU",
  "estimated_cost":  {"N_ct": ..., "b": ..., "d": ..., "R": ..., "basis": ...},
  "security_level":  "high",
}
```

`estimated_cost` 用课题的四量模型：

- `N_ct` 密文条数
- `b` 位宽
- `d` 乘法深度
- `R` 旋转／通信轮次

每条规则都带 `basis`，说明代价是怎么推出来的（例如"避 sqrt：比较距离平方与阈值平方，d=1"），
以及 `notes` 说明该表征的约束。

经 MPC 值布局的算子（`DistanceLE` / `WeightedSum` / `TemporalOverlap`）还会在
`estimated_cost` 里多带位平面布局键（`layout` / `layout_axis` / `layout_basis`，
给了规模形状时再加 `N_ct_naive` / `N_ct_layout` / `layout_reduction`）——
它与四量并存、**不覆盖** `N_ct`：`N_ct` 是"每元素条数"的描述串，布局键给的是**总条数**。
见 8.4 与 `docs/BITPLANE_LAYOUT.md`。

### 4.3 敏感度对规划的影响

- 输入含 `sensitive` / `secret` → 走密态后端，`needs_crypto = True`
- 输入全为 `public` → 状态标 `plaintext-ok`，但仍保留密态路径备选
  （不删除候选方案，因为上游若把该值密态化，方案必须立刻可切换）
- **解析不出来的输入按 `secret` 计入**（`GeoProgram.input_sensitivity`）。
  无输入的算子同样按 `secret` 处理。"不知道它是什么"不等于"它是公开的"。

代价四量里的位宽 `b` 按元素数实算，不写死：`WeightedSum` 用
`b(K) = 8 + 8 + ceil(log2 K)`（见 `planner.registry.resolve_cost`）。
旧版档案里写死 `b=16` 而备注里的公式是 `16 + ceil(log2 K)`，K>1 时自相矛盾。

### 4.4 规划无副作用

`plan_program()` 不修改传入的 `GeoProgram`，由测试锁定
（`test_planner.py::TestPlannerErrors::test_plan_does_not_mutate_program`）。

---

## 5. 当前 SPU 版本/API

> 完整核对记录与证据出处见 `docs/SPU_CAPABILITY.md`，机器可读版见
> `docs/spu_capability_report.json`。**以下每一条都实际核对过。**

### 5.1 版本与分发

- 官方最新发布版 **`spu==0.9.5`**（PyPI）。
- `requires-python = ">=3.10,<3.12"`。
- wheel 仅 macOS(`arm64`) 与 Linux(`manylinux_2_17_x86_64` / `manylinux_2_28_aarch64`)，**无 Windows wheel**。
- 依赖硬钉 `jax[cpu]>=0.4.16,<=0.4.34`，且要求 `numpy<2`。
- Windows 上使用需 WSL2 或 Linux 容器。

### 5.2 API 形态（已从 0.9.5 源码核对）

```python
from spu.utils import simulation
sim = simulation.Simulator.simple(wsize, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)
spu_fn = simulation.sim_jax(sim, my_jax_fn)
result = spu_fn(*inputs)
```

- **SPU 0.9.5 没有 `run_spu_simulation`**。本项目的 `run_spu_simulation(...)` 是在官方
  `Simulator.simple` + `sim_jax` 之上的封装（课题要求的命名）。
- 编译桥 `spu.utils.frontend.compile(Kind.JAX, ...)` 的流程：
  注册 dummy `interpreter` 后端 → `jax.jit().trace().lower(('interpreter',))`
  → `compiler_ir('hlo').as_serialized_hlo_module_proto()` → `spu_api.compile(...)`。
- 枚举取值：
  - `ProtocolKind`: `REF2K | SEMI2K | ABY3 | CHEETAH | SECURENN`（**无 SPDZ2K**）
  - `FieldType`: `FM32 | FM64 | FM128`
  - 参与方下限：`ABY3 = 3`、`SECURENN = 3`、其余 = 2

### 5.3 已知跨版本脆弱点

SPU 0.9.5 依赖以下 **jax 私有接口**：

| 接口 | 形态 | jax 0.4.34 | jax 0.11.2 |
|---|---|---|---|
| `jax._src.lib.xla_extension_version` | **属性** | 存在（值 289） | **已移除** |
| `jax._src.lib.xla_client` | **属性** | 存在 | 已移除 |
| `jax._src.api_util` | 子模块 | 存在 | 存在 |
| `jax._src.lax.lax._canonicalize_float_for_sort` | 函数 | 存在 | 存在 |
| `jax._src.xla_bridge.register_backend_factory` | 函数 | 存在 | 存在 |

**形态差别很重要**：`xla_extension_version` / `xla_client` 是 `jax._src.lib` 下的
**属性**，而非可 `import` 的子模块（SPU 源码写法：
`from jax._src.lib import xla_client, xla_extension_version`）。早期能力核查直接对这些
名字调 `import_module`，会把**已存在的属性误判为缺失**，从而把真实可运行的
环境标记成 `runnable=False`。现已修正为“模块优先；导入失败则拆出父模块
取属性”，并配两条回归测试锁定（正例 + 逆例）。

`xla_extension_version` 在 jax ≥ 0.5 被删除，SPU 0.9.5 用它分支选择新旧 API，
因此在较新 jax 上 SPU 无法工作。能力核查会检测并报告这一点。

### 5.4 本项目在 SPU 之上做的抽象

`geosecure` **不假设 SPU API**，把 SPU 相关的一切集中在一处：

- `backends/spu_backend/capability.py` —— 版本/平台/原语探测，全部对照官方源码
- `backends/spu_backend/runtime.py` —— `run_spu_simulation`，能力门控后才导入官方 API
- `backends/jax_backend/spu_bridge.py` —— SPU 编译前端前三步（不需要 libspu，可独立验证）

**未修改任何 SPU 源码。** 也未在本项目中重新实现任何 MPC 协议。

### 5.5 PSI API（spu 0.9.5 实测）

SPU 除 MPC 之外还提供 **PSI**（隐私集合求交），入口与 MPC 完全不同：

```python
import spu.libspu as libspu
from spu import psi

lctx = libspu.link.create_mem(desc, rank)      # 每个参与方一个线程
cfg = psi.PsiExecuteConfig(
    protocol_conf=psi.PsiProtocolConfig(
        protocol=psi.PsiProtocol.PROTOCOL_ECDH,
        receiver_rank=0,
        ecdh_params=psi.EcdhParams(curve=psi.EllipticCurveType.CURVE_SM2),
    ),
    input_params=psi.InputParams(
        type=psi.SourceType.SOURCE_TYPE_FILE_CSV,
        path="<csv>", selected_keys=["grid_code"], keys_unique=True,
    ),
    output_params=psi.OutputParams(
        type=psi.SourceType.SOURCE_TYPE_FILE_CSV, path="<out.csv>",
    ),
    join_conf=psi.ResultJoinConfig(
        type=psi.ResultJoinType.JOIN_TYPE_INNER_JOIN, left_side_rank=0,
    ),
)
report = psi.psi_execute(cfg, lctx)
```

| 项目 | 结论 |
|------|------|
| 输入输出 | 仅 **CSV 文件**，无内存张量接口 |
| 枚举中的协议 | `ECDH` `KKRT` `RR22` `ECDH_3PC` `ECDH_NPC` `KKRT_NPC` `DP` |
| 本链路可执行协议 | `ECDH` `KKRT` `RR22` `ECDH_NPC` `KKRT_NPC` `DP`（**不含 `ECDH_3PC`**） |
| 可用曲线 | `25519` `FOURQ` `SM2` `SECP256K1` `25519_ELLIGATOR2` |
| 默认选择 | `PROTOCOL_ECDH` + `CURVE_SM2`（国密，涉密测绘首选） |
| 哨位值 | `PROTOCOL_UNSPECIFIED` / `CURVE_INVALID_TYPE` 存在但不可用，已剔除 |
| 参与方约束 | `ECDH_3PC` 要求 3 方；本后端链路固定 2 方，故必然失败 |
| 结果精度 | 除 `DP` 外都是**精确**交集；`DP` 是差分隐私协议，结果**带噪**（见下） |
| 编译期放行 | 候选（自动/建议）：`ECDH` `KKRT` `RR22`；**显式放行**（只允许显式选择、不进建议清单）：`ECDH_NPC` `KKRT_NPC` `DP`；其余一律编译期拒绝 |

**"枚举里有"不等于"这里能跑"**。`PROTOCOL_ECDH_3PC` 是三方协议，而本项目的
进程内链路是两方（`Intersects` / `Contains` / `CellSetIntersect` 都是两方求交），
照名字配上去只会得到一段 libpsi 的 C++ 栈回溯：

```
[Enforce fail at external/psi~/psi/legacy/memory_psi.cc:44]
lctx_->WorldSize() == 3. psi_type:4, only three parties supported, got 2
```

现在它在进入协议**之前**就被拦下，并给出可读原因与替代协议清单。
能力报告同时列出两张清单：`protocols`（枚举中有什么）与可执行协议
（`protocols_runnable_here()`）。

**"能跑"也不等于"结果准"**。`PROTOCOL_DP` 是 **DP-PSI（差分隐私 PSI）**：
上游 `psi/legacy/dp_psi/dp_psi.h` 的 `DpPsiOptions(bob_p=0.9, epsilon=3.0)` 由 ε
推出 Alice 侧子采样 `p2 = e^ε/(1+e^ε) ≈ 0.953` 与上采样 `q = 1-p2 ≈ 0.047`
（往交集中**注入假元素**），Bob 侧另有 0.9 的子采样。噪声是协议的**设计目标**。

原生库自己把这些参数打了出来（`quiet=False`，2026-09-28 实测），无需推断：

```
LEGACY PSI config: {"psi_type":"DP_PSI_2PC", ..., "dppsi_params":{"bob_sub_sampling":0.9,"epsilon":3}}
[dp_psi.h:37] DpPsiOptions p1:0.9 epsilon:3 p2:0.9525741268224333, q:0.047425873177566746
[dp_psi.cc:49] sample bernoulli_distribution: 0.047425873177566746
[dp_psi.cc:60] bernoulli_items_idx:0 ratio:0      <- 本次上采样抽到 0 个，故未注入
```

本机 spu 0.9.5 实测（2026-09），两个方向都观测到，且**都是间歇性的**：

| 观测 | 结果 |
|------|------|
| 交集本体被注入非成员 | `A=[11,22,33]`、`B=[22,33,44]`（真交集 `{22,33}`）；`CellSetIntersect` 10 次里 1 次返回 `(11, 22, 33)`——`11` 根本不在 `B` 里 |
| `Intersects` 漏报真实冲突 | 真交集 2 个元素 → 30 次里 3~4 次给出 `False`（约 10%）；真交集 15 个元素 → 20 次里 0 次 |
| 同批对照 | `PROTOCOL_ECDH` 5/5、`PROTOCOL_KKRT` 12/12 均为 `True` |

对禁飞区判定这是**安全事故级别**的语义变化：既可能把不冲突判成冲突，也可能把
冲突判成不冲突；而且**小交集——恰恰是偶发冲突、最需要警惕的那一类——正是更容易
被漏报的一类**。因此本项目：

- 在 `PSI_PROTOCOLS_WITH_NOISE` 里登记它，`psi-check` 与 PSI 能力块都如实带出；
- 每个 DP 的 `PsiRunResult` 都带一条带噪说明（`notes`），不靠调用方转述；
- DP 下 `agreement=False` **不**升级为 `error`（那是协议语义；升级会让一次正常
  执行被随机报成失败）；
- 状态表记 `executed-noisy` 而**不是** `verified`——本项目里 `verified` 专指
  "与明文一致"。

**PSI 协议开关（CLI）**

```bash
# 换用已有的另一个两方协议（层面 1：不改 SPU，也不改本项目代码）
geo-secure build examples/route_conflict.py --psi-protocol KKRT
geo-secure build examples/route_conflict.py --psi-protocol RR22
geo-secure build examples/route_conflict.py --psi-protocol ECDH_NPC   # NPC 族：显式放行档
geo-secure build examples/route_conflict.py --psi-curve CURVE_25519
# Contains 的子集判定：默认走 MPC，可显式退回明文（退回会被状态词标出来）
geo-secure build examples/vertical_conflict.py --psi-subset plaintext
```

| 参数 | 缺省 | 说明 |
|------|------|------|
| `--psi-protocol` | `PROTOCOL_ECDH` | 取值见上表"本链路可执行协议"。候选 `ECDH`/`KKRT`/`RR22` 之外的 `ECDH_NPC`/`KKRT_NPC`/`DP` 属**显式放行**：可显式选择、不进自动候选/替代建议；`KKRT`/`RR22`/`KKRT_NPC` 不基于椭圆曲线（给了也不读）；`DP` 能跑但结果带噪 |
| `--psi-curve` | `CURVE_SM2` | 只有 ECDH 族由本项目注入。**即使当前协议用不上也会校验拼写**，拼错即报错而非静默忽略。`DP` 自带内置默认曲线（上游源码默认 25519），本项目**不覆盖**它——所以提示语写"未传入"而**不写**"不生效" |
| `--psi-subset` | `mpc` | `Contains` 的**子集判定**走哪条路：`mpc`（MPC 基数等值）/ `plaintext`（显式明文）。只接受这两个值；MPC 不可用时自动退回明文并把模式标为 `plaintext-fallback`，状态词记 `subset-plaintext`（见 7.6） |
| `--psi-rr22-low-comm-mode` | 关 | **RR22 专用参数**：开 → `Rr22Rarams.low_comm_mode=True`（低通信模式）。只对 `--psi-protocol RR22` 生效（配其它协议会打印提示且不注入），不是通用 curve 参数 |

三条纪律（都有测试守着）：

1. 非法协议名 → 退出码 `2` + 可用清单，**不是 traceback**；
2. 选了本链路跑不了的协议（如 `ECDH_3PC`）→ 能力阶段直接报 error，
   状态表记 `backend-direct` 而**不是** `verified`，模拟阶段跳过。
3. 选了结果带噪的协议（`DP`）→ 能跑，但两个 PSI 阶段都记 warning，
   状态表记 `executed-noisy`，且**不得**出现"经真实 PSI 求交验证"这类字样。
4. 选了**显式放行**协议（`ECDH_NPC` / `KKRT_NPC`）→ 编译期放行，方案理由带
   "显式放行（不进候选/替代建议清单）"；这两族已真机执行并与明文对拍。
   既未登记候选、也未显式放行的协议（含未来上游新增）仍被编译期拒绝，
   提示"先登记候选或显式放行并补测试"。


**MPC（SPU）协议开关（P0，本版新增）**

`--protocol` 作用于 SPU/MPC 阶段（两者是**不同的协议命名空间**：
PSI 侧是 `PROTOCOL_ECDH` / `RR22`，MPC 侧是 `REF2K` / `SEMI2K` / `ABY3` /
`CHEETAH` / `SECURENN`，合用一个字段会让两种协议互相被当成"未知协议"）。

```bash
geo-secure build examples/distance_check.py --protocol CHEETAH   # 合法，且真跑验证
geo-secure build examples/distance_check.py --protocol SPDZ2K    # 规划期即拒（退出码 1）
```

规划期校验（`planner.registry.validate_mpc_protocol_for_operation`）：
每个算子登记 `default_mpc_protocol`（`ABY3`）与 `mpc_protocol_candidates`
（5 个协议）；显式选的协议不在该清单里 → 产出 `PROTOCOL_UNSUPPORTED`
编译期诊断，**不再等到运行时才崩**。`PlannedStep.mpc_protocol` 与编译 JSON
都带该值，可逐层对拍。

**MPC 协议按代价选择（P4，本版新增）**

P0 那条开关解决的是"能不能选"，P4 解决的是"**凭什么选**"。此前自动路径直接取
登记默认值 `MPC_RULE_DEFAULT_PROTOCOL`（`ABY3`）——"协议选择"在编译期其实没有
选择依据。现在补上三段：

1. **候选集带实测代价**。`planner.mpc_protocol_candidates_for(op)` 从
   `docs/mpc_comm_baseline.json`（§8.4 的 `--comm --repeat 5` 产物）读每个协议在
   **同一比价规模**上的发送+接收字节数。比价规模取"能覆盖最多协议的 K"
   （FM64、默认电路），不取各协议各自的最大 K——否则 ABY3 会取到 K=4096、
   其余协议只有 K=256，比的就不是同一个量了；
2. **自动选择取实测最省**。`--protocol` 缺省=**自动**
   （`planner.select_mpc_protocol`）；显式给了协议才覆盖。选到什么、凭什么，
   写进 `PlannedStep.mpc_protocol_basis`（`measured-comm` / `declared-default` /
   `explicit`）与 `PlannedStep.reasons`，CLI 的 Privacy planning 段直接印出来：

   ```
   MPC 协议: DistanceLE → ABY3（按实测代价自动选择）
   ```
3. **无密码学保护的协议不参与自动选择**。`REF2K` 实测 send+recv 恒为 **0 B**
   （§8.4），把它当"等价但更快的隐私后端"不成立。它仍留在
   `mpc_protocol_candidates` 里——对拍/排查要能显式选——但被
   `SPU_PROTOCOLS_WITHOUT_CRYPTO` 挡在自动选择之外，且显式选它时规划里会
   带一条披露。

| 算子（比价 K） | 实测发送+接收（升序，可自动选） | 排除 |
|---|---|---|
| `DistanceLE`（K=256） | **ABY3** 4 200 B < SEMI2K 8 336 B < SECURENN 12 032 B < CHEETAH 2 415 124 B | REF2K 0 B |
| `WeightedSum`（K=256） | **ABY3** 4 112 B < SEMI2K 8 192 B = SECURENN 8 192 B < CHEETAH 1 473 970 B | REF2K 0 B |
| `TemporalOverlap`（K=32） | **ABY3** 508 672 B < SEMI2K 696 320 B < CHEETAH 1 758 220 B < SECURENN 25 837 512 B | REF2K 0 B |

- **现有实测下 ABY3 恰好是最省的**，所以自动选择与 P0 的登记默认值一致——
  行为不变，但**依据换成了实测**；一旦排序翻转，选择会跟着翻转，并在
  `reasons` 里写明"与登记默认值不同、如需保守可显式指定"；
- **除 ABY3 外没有稳定的全局顺序**：`SECURENN` 与 `CHEETAH` 在两个算子上位置
  互换（`DistanceLE` 上 SECURENN 更省，`TemporalOverlap` 上反过来差 14.7×）
  ——代价由电路形态主导，所以**按算子各排各的**，不搞一张全局协议优先级表；
- **产物缺失时不编数字**：退回登记默认值，`basis="declared-default"`，
  理由里明说"这是登记默认值，不是实测结论"；
- **拒绝信息要可操作**：非法协议名 / 候选清单外的协议，`PROTOCOL_UNSUPPORTED`
  的 `suggestion` 会带上这行排序（含"已排除 REF2K"的原因），而不是只报"不支持"。

口径细节见 `docs/MPC_BENCHMARK_PROTOCOL.md` §8.5，测试见
`tests/test_protocol_selection.py`（21 条，含"拿产物核对候选上的每个数字"与
"排序第二的协议也真跑通"）。

**四个实测坑**（详见 `docs/PSI_CAPABILITY.md`）：

1. `PROTOCOL_ECDH` 不显式给 `curve` → `RuntimeError: Curve type is not specified.`
2. 官方枚举是 pybind11 类型，成员**不是 `int`**，只判 `isinstance(int)` 会静默得到空清单。
3. 空集合（只有表头的 CSV）会让 arrow 读表失败 → 现在前置判掉。
4. 只有 `receiver_rank` 拿到交集与计数，另一侧 `intersection_count` 为 `-1`。
5. `LogOptions.system_log_path` 默认是相对路径 `spu.log`，原生库会在**当前工作目录**
   落文件。本后端一律改指 `/dev/null`（需要日志时用 `quiet=False` 走 console），
   不让编译器往用户项目里丢日志。
6. `PROTOCOL_DP` 的 `curve` 是**带默认值的形参**（上游源码默认 25519），不是"不读曲线"。
   早期把它与 `KKRT` / `RR22` 并列，于是 CLI 告诉使用者"--psi-curve 不生效"——一条
   没核对过的断言。现在按 `required` / `ignored` / `implicit` 三分类分别措辞。

### 5.6 RR22 支持（本版新增）

**接入方式**：复用 SPU/libpsi 已提供的 RR22 实现，**不在 GIS_SPU 内重新实现 RR22
密码学算法**（OPRF / OKVS 等一概不落地到本仓库）。本仓库负责：协议选择 → 参数装配
→ 运行验证。链路：

```text
GeoSOT-3D 64-bit grid_code → CellSet → Geo-IR（Intersects / CellSetIntersect）
    → CompactCellSet（Planner）→ PrivacyPlan（protocol=PROTOCOL_RR22 + protocol_params）
    → PSI Backend → psi.PsiProtocolConfig(rr22_params=psi.Rr22Rarams(low_comm_mode=...))
    → spu.psi.psi_execute → libpsi RR22
```

```bash
geo-secure build examples/rr22_grid_intersection.py --psi-protocol RR22
geo-secure build examples/rr22_grid_intersection.py --psi-protocol RR22 --psi-rr22-low-comm-mode
```

```text
[7/8] PSI capability check   + OK   ...；选用 PROTOCOL_RR22（不读曲线）（low_comm_mode=True）...
PSI simulation
  Intersects: ok  [PROTOCOL_RR22]
      params    : low_comm_mode=True
      |A|=3  |A∩B|=2
      result    : True
      reference : True   agree=True
```

| 项目 | 结论（spu 0.9.5 实测，2026-09-30） |
|------|------|
| 协议登记 | `PsiProtocol.PROTOCOL_RR22`，两方、精确协议 |
| 参数类 | `psi.Rr22Rarams(low_comm_mode=False/True)` 均可构造 |
| 参数注入 | `PsiProtocolConfig(..., rr22_params=...)` 实测接受；runtime 在 worker 内注入 |
| 曲线 | `RR22 = ignored`：不读 curve，绝不把 SM2 注入 RR22 |
| 真实执行 | `Intersects` / `CellSetIntersect` 与明文一致；两种 `low_comm_mode` 均验证 |
| 64 位码 | `>= 2^63` 的格网码不被截断（同 ECDH/KKRT 的回归口径） |
| 参数可追踪 | 运行结果 / 计划步骤 / JSON / CLI 四处都带 `protocol`、`protocol_params` |
| 能力探测 | `check_psi_capabilities().rr22_params`：`protocol_present` / `rr22_params` / `low_comm_mode` / `runnable` 逐项如实探测 |

已完成与未完成必须分开写：

```text
已完成：RR22 协议接入 —— 可配置 / 可编译 / 可执行 / 可验证（本版）
未完成：RR22 密码算法改造、Geo-RR22 优化 —— 下一阶段（见 8.0 与 docs/RR22_INTEGRATION.md）
```

- 默认协议**仍是** `PROTOCOL_ECDH + CURVE_SM2`；RR22 由 `--psi-protocol RR22` 显式选择。
  是否改为默认，等真实 RR22 完成性能/正确性对拍后再定（`planner/registry.py`
  的 `PSI_RULE_DEFAULT_PROTOCOL`）。
- `--psi-rr22-low-comm-mode` 是 RR22 专用；配其它协议会打印提示且不注入。
- 若上游版本没有 `Rr22Rarams`：能力探测如实报"不完整"；运行期显式请求
  `low_comm_mode=True` 会失败而不是静默忽略（详见 `docs/RR22_INTEGRATION.md`）。

### 5.7 真实输入 / 链式执行 / 布局握手（本版新增）

**真实输入**：`--input NAME=PATH`（可重复）把业务量绑定到真实格网集合；
输入适配层（`backends/psi_backend/input_adapter.py`）支持：

```text
CSV（需 grid_code 列；缺表头直接报错，不静默丢码）
JSON（[codes...] 或 {"grid_codes": [...], "layout": {...}}）
CellSet / 代码序列
```

```bash
geo-secure build examples/route_zone_chain.py \
    --input route=examples/route_cells.csv \
    --input no_fly_zone=examples/nofly_cells.csv \
    --input sensitive_area=examples/sensitive_cells.csv
```

**链式执行**：`CellSetIntersect` 的真实 PSI 输出作为下游算子的运行时输入
（不再读样例）。每个 run 的 notes 标注输入来源（`输入绑定` / `上一步 PSI 输出` /
`样例兜底`）；链式上游失败时下游**不允许**回退样例，报"输入无法装配"。

**布局握手**：`--input-layout NAME=PATH`（或 `Compiler(input_layouts=...)`）为每方
声明 `ir.grid_code_layout_manifest()`；双方都声明且不一致 → **进入 PSI 前**以
`LAYOUT_MISMATCH` 拒绝执行；只声明单方 = 无法核对（如实标注，不假装通过）。

**配置闭环**：`psi_protocol_params`（`receiver_rank` / `broadcast_result` /
`low_comm_mode` 等）沿 Planner → Compiler → Runtime → `PsiRunResult` 单链传递；
编译 JSON 带 `psi_runtime_config` 与每步 `protocol_params`，四层可直接对拍。

**MPC 协议闭环（本版 Phase 4）**：MPC 侧唯一由方案**逐步决定**的参数就是协议本身
（环宽 / 参与方数量同源于编译器配置），执行期按**步骤**解析
（`Compiler._step_protocol`：显式指定 > 方案实测选择 > 登记默认值）——同一程序里
两个 MPC 步骤各选不同协议时**各跑各的**，不再把“第一个 MPC 步骤的协议”套到所有
步骤上。`_check_mpc_closure` 在阶段 6 兜底核对，漂移即阶段报错
（验收口径：Runtime 不允许静默更换协议）。编译 JSON 的 plan 步带出 `mpc_protocol`
与 `mpc_protocol_basis`，与 `spu_runs[op].protocol` 可直接对拍；
见 `tests/test_mpc_execution_chain.py`。

### 5.8 性能基线（本版新增）

`tests/benchmarks/benchmark_psi.py`：确定性 GeoSOT-3D 测试集，真机测量
ECDH / KKRT / RR22 / RR22+low_comm（N=2^10…2^24），输出
`docs/psi_benchmark_baseline.json` / `.csv`。

```bash
/opt/miniconda3/envs/spu311/bin/python tests/benchmarks/benchmark_psi.py            # 标准扫描
/opt/miniconda3/envs/spu311/bin/python tests/benchmarks/benchmark_psi.py --quick    # 快扫（2^10/2^12）
```

诚实规则：未实测的组合记 `unavailable`（不填数字）；重复键用例按**实测**记录
（RR22/KKRT 报错、ECDH 计数含重复乘数）——这是"去重预处理是正确性前提"的证据，
见 `docs/BENCHMARK_PROTOCOL.md` 与 `docs/GEO_RR22_DESIGN.md`。

**MPC 侧基线（P1 新增）**：`tests/benchmarks/benchmark_mpc.py` 把
`DistanceLE / WeightedSum / TemporalOverlap` × 5 个 SPU 协议
（REF2K / SEMI2K / ABY3 / CHEETAH / SECURENN）× 3 个环宽实测，
输出 `docs/mpc_benchmark_baseline.json` / `.csv`。
对账轴是**预测位宽 b(K) ↔ 平台环宽**，**不**给"预测时间 ↔ 实测时间"误差
（planner 不预测时间，硬凑口径等于伪造）。

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib"                      # libgomp.so.1 所在
~/.spuenv/bin/python tests/benchmarks/benchmark_mpc.py         # 标准扫描
~/.spuenv/bin/python tests/benchmarks/benchmark_mpc.py --quick # 快扫
```

基线找出的两处缺口（**都已在 P2-1 修复**，见下）：

- `WeightedSum` 的定点整除 `acc // scale` 在 SPU 上是**近似且非确定**的
  （`scale == 1` 亦然；K=1024 偏差率 43%、K=4096 为 93%）；
- `WeightedSum × FM32` **完全起不来**（除法路径内部需要 64 位环：
  `ring=FM32 could not represent PT_I64`），而 planner 的位宽预测
  b(K=256)=24 ≤ 32 会判"够用"——**预测 ≠ 实测**的实例。
- （另有）`TemporalOverlap` 的**默认**电路是 N×M 两两比较（二次），规模上限只能压到 K≈32——**P3 已解除**：新增 sweep 电路后 K=1024 真机跑通。

**P2-1：把定点 scale 改成编译期常量（本版新增）**

根因是**一步除法**——它在 HLO 里展开成 `divide + remainder + select + sign`。
`scale` 在所有路径上恒为 1，所以这步除法纯粹是负担：删掉后

| 指标（`WeightedSum` K=256） | 修前 | 修后 |
|---|---:|---:|
| 与明文一致（K=4096，×30） | 偏差率 93% | **0/30，逐位精确** |
| `FM32` 环宽 | 崩溃 | **ok** |
| PPHLO 字节数 | 2 400 | **938** |
| 通信量（ABY3 / SEMI2K / CHEETAH，探针口径） | 9 286 / 51 544 / 5 246 783 B | **4 112 / 16 384 / 1 474 286 B** |
| 通信量（同上，P2-2 基线口径：SEMI2K 改 2 方） | — | **4 112 / 8 192 / 1 474 167 B** |
| CHEETAH 墙钟中位数 | 1 149.8 ms | **85.2 ms** |

生成规则：`scale == 1` → 直接返回累加和；`scale == 2^s` → 右移（精确）；
其它 → 保留整除并在 notes 里写明"SPU 上是近似除法、需要 ≥FM64"。

**重复实验与精确性（P1.5 / P2-1 重跑）**

```bash
~/.spuenv/bin/python tests/benchmarks/benchmark_mpc.py --repeat 5 \
    --json docs/mpc_repeat_baseline.json --csv docs/mpc_repeat_baseline.csv
~/.spuenv/bin/python tests/benchmarks/benchmark_mpc.py --ops WeightedSum --repeat 30 \
    --json docs/mpc_exactness_repeat.json --csv docs/mpc_exactness_repeat.csv
```

判断标尺：

- 同一会话内 `p75/p25` 多数 ≤ 1.05 ⇒ **只认差异大于 5% 的协议排序**；
- 4 个非 CHEETAH 协议在三个算子上差异都与抖动同量级，**本基线不支持
  给它们排序**；唯一超出噪声的是 `TemporalOverlap` 上 SECURENN / CHEETAH 明显更慢；
- CHEETAH 仍慢一个数量级，但 SPU 模拟器**不含真实网络**，省通信量的优势
  在带宽受限部署下**不可外推**——通信量口径见下。

**P2-2：通信量已入基线（本版新增）**

`RuntimeConfig.enable_pphlo_profile = True` 后 SPU 自己的日志会给出逐算子
`send bytes / recv bytes` 与 `Link details: total send bytes N, recv bytes M`
（需在 fd 级重定向，Python 的 `redirect_stdout` 抓不到），
现已接进 `run_spu_simulation(capture_comm=True)` 与每条基线记录。

```bash
~/.spuenv/bin/python tests/benchmarks/benchmark_mpc.py --comm --repeat 5 \
    --json docs/mpc_comm_baseline.json --csv docs/mpc_comm_baseline.csv
```

实测（send+recv 合计，K=256 / `TemporalOverlap` K=32）：

| 协议 | DistanceLE | WeightedSum | TemporalOverlap |
|---|---:|---:|---:|
| REF2K | 0 B | 0 B | 0 B |
| ABY3 | 4 200 B | 4 112 B | 0.51 MB |
| SEMI2K | 8 336 B | 8 192 B | 0.70 MB |
| SECURENN | 12 032 B | 8 192 B | **25.84 MB** |
| CHEETAH | **2.42 MB** | **1.47 MB** | 1.76 MB |

三个结论（详见 `docs/MPC_BENCHMARK_PROTOCOL.md` §8.4）：

- **墙钟排不出协议**：除 CHEETAH 外四个协议墙钟都在 45–49 ms / 36–39 ms，
  通信量却从 0 B 跨到 2.42 MB（3 个数量级）——REF2K 的 0 B 说明它不做
  密码学保护（"最快"没有意义）；
- **`TemporalOverlap` 上墙钟排序被通信量反转**：SECURENN 墙钟 633 ms
  < CHEETAH 1242 ms，通信量却是 25.84 MB ≫ 1.76 MB（差 14.7×）——
  二次电路把 SECURENN 的通信量推爆；
- **取不到就是 `None`**：没开 profiling / 解析不到 / 原生 logger 打不开，
  一律留空并在 `note` 写明原因，**不用推测值填充**。

参与方数量按各协议下限取默认值（ABY3/SECURENN 3 方、SEMI2K/CHEETAH/REF2K 2 方）：
**这个数必须跟着通信量一起报**——实测 SEMI2K 同一组合 2 方 8 192 B、3 方 16 384 B。

三个实测坑（都会让通信量静默变成"取不到"）：CHEETAH 必须 2 方，否则
`[yacl] Get data timeout`；profile 行必须 **fd 级**重定向才拿得到；
SPU 原生日志是**进程级**开关，PSI 路径会把它关掉，因此 `capture_comm=True`
会先把它打开。多数组合 ×5 完全确定，但 `TemporalOverlap` 的 ABY3 会抖
（pairwise 3.0%–6.1%；`sweep` 电路抖得更大：9.5%–12.2%），所以一律报**中位 + 区间**。

> ⚠️ profiling 有开销：**带 profiling 的墙钟不可与不带 profiling 的墙钟比**。
> `docs/mpc_repeat_baseline.*` 是没开 profiling 的，`docs/mpc_comm_baseline.*` 是开了的。

**P3：`TemporalOverlap` 两套电路的 A/B（本版新增）**

`TemporalOverlap` 现有两套电路：默认的 N×M 两两比较（pairwise，二次）与
排序归并 + 前缀扫描（`sweep`，O((N+M)log(N+M))）。**谁更省要在同规模上成对比**，
所以单独开了一个 A/B 模式（混进主扫描会被不同 K 的行稀释掉）：

```bash
~/.spuenv/bin/python tests/benchmarks/benchmark_mpc.py --temporal-ab --comm --repeat 5 \
    --json docs/mpc_temporal_ab.json --csv docs/mpc_temporal_ab.csv
```

实测（ABY3 / FM64，send+recv，×5 中位，带 profiling）：

| K | pairwise 通信量 | sweep 通信量 | 谁省 |
|---:|---:|---:|---|
| 8 | 95 872 B | 317 968 B | pairwise |
| 32 | 496 896 B | 1 265 424 B | pairwise |
| 64 | 1 296 896 B | 2 497 552 B | pairwise |
| 128 | 3 853 312 B | 5 438 480 B | pairwise |
| 256 | 12 546 048 B（下界 12.50 MB） | **10 889 232 B**（上界 11.18 MB） | **sweep（区间不重叠）** |

- **交叉点在 K=128 与 256 之间**：K=256 上 sweep 的上界低于 pairwise 的下界，
  这是排序结论而不是"看着差不多"；故 `strategy="auto"` 以 K=256 为阈值，
  **拿不到规模时退回 pairwise**（不擅自改既有成本口径）；
- **换电路省的是通信量，不是时间**：K=256 墙钟 sweep 354 ms vs pairwise 177 ms，
  区间同样不重叠；
- **真正的收益是"能跑"**：K≥256 逐对版会拖死进程，sweep 在 K=1024 真机跑通
  （单次探针 1.8 s / 43.7 MB，**不成对故不参与比价**）；
- 两套电路**都保留**：`sweep` 的 `sort` 依赖 SPU frontend 补丁（跨 jax 版本易碎），
  出问题时还有退路。

详见 `docs/MPC_BENCHMARK_PROTOCOL.md` §4.1。

### 5.9 PSI-Cardinality 计数档（`--psi-count psi-ca`，第四个后端族）

缺省的 PSI 走 libpsi 求交（接收方拿到交集本体）；`--psi-count psi-ca` 换成
**OpenMined PSI-Cardinality**（`openmined-psi==2.0.6`，Apache-2.0）：协议只计算并返回
**交集基数**，交集本体不交给任一方。两条路径并列、泄漏承诺不同，刻意不合并
（libpsi 的 `REVEAL_COUNT` 只是业务层收窄，协议内部仍把交集本体交给接收方）。

```bash
geo-secure build examples/conflict_count.py --psi-count psi-ca   # 只出计数
geo-secure psi-ca-check                                          # 环境核查
```

- 只承接 `CellSetIntersect` 的计数（`REVEAL_COUNT`）：`Intersects` / `Contains`
  等算子、以及"计数喂给下游当集合"的链式用法，在编译期**显式拒绝**
  （逐条给出错误位置、原因、建议替代与预计代价）；
- 只用上游 `RAW` 数据结构（精确计数）；`GCS` / `BloomFilter` 近似档未接；
- 状态词是 `count-only`（不是 `verified`——产出只有基数，没有交集本体可比对）；
- 接口是进程内 protobuf（不落盘）；角色映射 client=左（获得计数）/ server=右；
- API 不预设：运行期核对上游 API 形态，不符即 `runnable=false` 并列出缺失项；
- **真机已验证（2026-10-09，WSL2 + `openmined-psi==2.0.6`）**：`psi-ca-check`
  `runnable=true`（blockers 为空）；`build examples/conflict_count.py --psi-count psi-ca`
  真实执行 `|A|=3  |A∩B|=2`、状态词 `count-only`、`agree=True`；命令与实测输出见
  `docs/PSI_CA_CAPABILITY.md` §8，环境快照 `docs/psi_ca_capability_report_wsl.json`。

### 5.10 PI-Sum 交集内求和档（`--psi-sum pjc`，第五个后端族）

缺省的 PSI 走 libpsi 求交、`--psi-count psi-ca` 只出基数；`--psi-sum pjc` 是
**第三条 PSI 路径**：接入 Google **private-join-and-compute**（Apache-2.0，
Private Intersection-Sum）。协议一次算出**两个数**——交集基数 +
**交集内关联值之和**——交集本体同样不交给任一方。

```bash
geo-secure psi-sum-check                                         # 环境核查
geo-secure build examples/intersection_sum.py --psi-sum pjc \
    --psi-sum-weights route=examples/route_risk_weights.csv       # 基数 + 权重和
```

- 只承接 `CellSetIntersect` 的 `REVEAL_INTERSECTION_SUM`：其余算子、以及
  "两个数喂给下游当集合"的链式用法，在编译期**显式拒绝**（逐条给出错误位置、
  原因、建议替代与预计代价）；
- **左侧（client）输入必须带关联值**：`psi_sum_weights`（CLI
  `--psi-sum-weights 输入名=PATH`，CSV `grid_code,value` 或 JSON）。缺值
  **不按 0 补齐**——补齐会把"数据没接上"伪装成"和为 0"；
- 关联值须为非负 int64（上游口径），和值须落在 int64 内，否则上游报错；
- 状态词是 `count-and-sum`（既不是 `verified`——两个数没有交集本体可比对；
  也不是 `count-only`——本档多交了一个和）；
- **接入对象是 Bazel 构建的两个可执行文件**（上游无官方 PyPI 包）：能力核查
  跑 `<binary> --help` 逐项核对 flag 形态，缺项即 `runnable=false`。
  PyPI 上的同名包与本项目无关，不得顶替；
- 传输面：上游两侧都走 gRPC `LocalCredentials(LOCAL_TCP)`，**只能同机**、无 TLS、
  无身份认证；本项目把监听地址从 `0.0.0.0` 收紧到 `127.0.0.1`；
- **诚实登记**：本档在测试桩下已验证接入层；上游需 Bazel 构建，真机执行未做，
  Linux/WSL 复跑命令见 `docs/PSI_SUM_CAPABILITY.md` §8。

---

## 6. 已验证算子

验证强度分三级，下表如实标注：

| 算子 | 明文 | JAX 生成 | jax.jit 可追踪 | HLO 可降级 | SPU 实跑 |
|---|---|---|---|---|---|
| DistanceLE | ✓ | ✓ | ✓ | ✓ | **✓ ABY3/FM64，误差 0.0** |
| WeightedSum | ✓ | ✓ | ✓ | ✓ | **✓ ABY3/FM64，误差 0.0** |
| TemporalOverlap | ✓ | ✓ | ✓ | ✓ | **✓ ABY3/FM64，误差 0.0** |
| Intersects | ✓ | 不生成（PSI 族，无逐元素 JAX 原语） | — | — | **✓ PSI ECDH/SM2，与明文一致** |
| Contains | ✓ | 不生成（同上） | — | — | **✓ PSI ECDH/SM2 + 子集判定 ABY3/FM64 MPC 基数等值，与明文一致** |
| CellSetIntersect | ✓ | 不生成（同上） | — | — | **✓ PSI ECDH/SM2，与明文一致** |
| HeightBand | ✓ | 不生成（明文物化，非逐元素算子） | — | — | — （不进密态） |

> PSI 走的是**独立的执行路径**（`spu.psi.psi_execute`），不是 `jax.jit → SPU 虚拟机`。
> 因此表中"SPU 实跑"一列对 PSI 族记的是**真机 PSI 求交**结果；
> 详见 `docs/PSI_CAPABILITY.md`。

> `HeightBand` 一列全为"不适用"是**结论而不是缺口**：它在本方明文把高度带
> 展开成层集合，密态边界落在消费它的 `Intersects` / `Contains` / `CellSetIntersect`
> 上（那些行已验证）。它的明文语义与 `CellSet.from_height_band` 有对拍测试。

> `CellSetIntersect` 另有一条**计数档**（`--psi-count psi-ca`）：走 OpenMined
> PSI-Cardinality，协议只出交集基数、不交交集本体。它属"接入层已验证（测试桩）、
> 真机待复跑"状态——见 §5.9 与 `docs/PSI_CA_CAPABILITY.md`。

### 6.1 什么叫"已验证"

- **明文**：`backends/plain/` 的参考实现，六个算子齐全。它是 JAX／SPU 结果的语义基准。
- **JAX 生成 + 可追踪**：生成代码通过静态自检（只用 `jnp`、无 Python 级分支），
  且 `jax.jit` 追踪成功。
- **HLO 可降级**：走通 SPU 编译前端前三步，并实测提取出 StableHLO 原语。
- **SPU 实跑**：在 WSL2 + Python 3.11 + `spu==0.9.5` + `jax==0.4.34` 下
  走官方 `Simulator.simple` + `sim_jax` 真实执行，并与明文输出对拍。
  当前三个可生成 JAX 的算子均**已实跑通过**（误差 0.0）。
- **PSI 实跑**：走官方 `spu.psi.psi_execute` 做**真实两方求交**，
  交集基数与手工真值逐位核对，并与 `backends.plain` 对拍。
  已在 `PROTOCOL_ECDH/SM2`、`ECDH/25519`、`KKRT`、`RR22` 四种配置下
  得到一致结果。三个 PSI 算子均**已实跑通过**。

  PSI 与 SPU 的"已验证"**互不顶替**：`_status_word` 分别判定两条路径，
  没有真实执行记录时一律不写 `verified`。

### 6.1.1 状态词口径

| 状态词 | 含义 | 触发条件 |
|--------|------|----------|
| `verified` | 主后端**真实执行过**且与明文一致 | PSI `status=ok` / SPU `ok` 且在容差内 |
| `executed-noisy` | 真跑了，但所用协议**结果带噪**，不一致是设计行为 | PSI `status=ok` 且协议在 `PSI_PROTOCOLS_WITH_NOISE` 中（当前仅 `DP`） |
| `subset-plaintext` | PSI 段真跑了，但 `Contains` 的子集判定落在**明文**上 | `--psi-subset plaintext`；或 MPC 不可用时自动退回（mode=`plaintext-fallback`） |
| `count-only` | PSI-CA 计数档真实执行过且与明文计数一致；产出只有基数，不与 `verified` 混用 | `psi_run.protocol == "PSI-CA"` 且 `status=ok` |
| `count-and-sum` | PI-Sum 求和档真实执行过且与明文一致；产出是「基数 + 交集内关联值之和」两个数，同样不与 `verified` 混用 | `psi_run.protocol == "PJC-PI-SUM"` 且 `status=ok` |
| `tolerance-exceeded` | SPU 跑通但超出容差 | `within_tolerance is False` |
| `backend-direct` | 有直连后端，本次未真实执行 | PSI 未跑（环境缺失 / 空输入） |
| `jax-verified` | JAX 生成且可追踪，无密态实跑 | `jax.jit` 追踪成功 |
| `jax-generated` | 生成了 JAX 但未做追踪验证 | 追踪为 `None` |
| `not-traceable` | JAX 代码无法被 `jax.jit` 追踪 | `traceable is False` |
| `plaintext-local` | 该算子在本方明文完成，本就**不该**有密态执行 | 物化算子（`HeightBand`） |
| `planned` | 仅规划，未执行 | 兜底 |
| `error` | 执行失败或与明文不一致 | 见 `PsiRunResult.error` |

PSI 的 `empty-input`（空集合）与 `unavailable`（环境缺失）**都不进入** `verified`：
前者没启动协议，后者根本没跑。

`executed-noisy` 是新增的一档，理由与 `backend-direct` 同源：不许用一个不能成立的
词。`verified` 在本项目专指"与明文一致"，而带噪协议**按定义**不保证这一点——把它
写成 `verified` 就是拿一个读过源码、测过 30 次的结论去换一个好看的状态词。

### 6.2 实测原语（防止"登记靠猜"）

| 算子 | 实测发射的 StableHLO 原语 |
|---|---|
| DistanceLE | `subtract, multiply, reduce, add, convert, compare, constant` |
| WeightedSum | `multiply, reduce, add, constant`（P2-1 前为 `multiply, reduce, add, divide, remainder, compare, select, sign, and, convert, subtract, constant`） |
| TemporalOverlap（默认 / pairwise） | `shift_left, add, broadcast_in_dim, compare, and, or, reduce, constant` |
| TemporalOverlap（`sweep` 电路，P3） | `shift_left, shift_right_arithmetic, add, negate, broadcast_in_dim, compare, and, or, select, concatenate, sort, reduce, reduce_window, return, constant` |

**重要发现（已闭合）**：源码中一步写法的定点整除 `acc // scale`，在 HLO 里展开为
`divide + remainder + select + sign`。即"看起来一步"的操作在密态下代价高得多。
`divide` / `remainder` 已列入高代价原语并在能力核查时告警。
P2-1 把 `scale` 改成编译期常量后，该算子生成代码里**不再有除法**——
这条发现的用处从"告警"变成"设计纪律"：**别把能编译期消解的运算放进密态电路**。

该表由测试锁定：生成代码若引入新原语，`test_measured_primitives_match_registry` 会失败。

### 6.3 容差

| 算子 | 容差 | 说明 |
|---|---|---|
| DistanceLE | 0.0 | 整数平方和与阈值平方比较，精确 |
| WeightedSum | 0.0 | 定点整数乘加，精确（P2-1 后生成代码无除法；K=64…4096 × 30 次实测全部逐位一致） |
| TemporalOverlap | 0.0 | 整数区间比较，精确 |

三份实现（明文 / JAX / SPU）在**整数路径下要求完全一致**，容差 0 是刻意选择：
`QuantizedVector` 与 `FixedPointVector` 的设计目的就是把浮点误差挡在密态之外。

若上游把输入浮点化，`backends/jax_backend.TOLERANCES` 旁提供
`FLOAT_TOLERANCE = 1e-4` 作为相对误差上限，并有测试验证误差只来自表示精度。

### 6.4 测试覆盖

```
tests/test_ir.py                  38 项   类型系统、格网口径、算子/程序/关系
tests/test_frontend.py            41 项   表达式级调用识别、输入可解析性、敏感度不降级、链式类型、语义
tests/test_planner.py             58 项   注册表、五元组、敏感度策略、代价模型、无副作用、PSI/MPC 两套协议候选校验
tests/test_bitplane_layout.py     23 项   位平面布局（D3）：课题产物逐项复算、归约轴决策、不给形状不给数字、与实测前提对账
tests/test_packing_probe.py       30 项   打包前提探针（P6）：位宽扫描/规模扫描、按环元素计费、收益上界、缺数诚实留空、真机位宽等价
tests/test_slot_cost_probe.py     25 项   取槽步单价（P7-P0）：掩码/右移/全量提取 vs 纯乘法的 A/B、位运算语义、比价基准不许拿 0 B 变体
tests/test_primitive_probe.py     27 项   逐原语真机核验（P0）：D3 点名原语全覆盖、登记名与 capability 白名单对账、计费维度对账（收缩类按输出计费、收缩长度免费）、只有同维度的不可能读数才标可疑
tests/test_slot_reduction_probe.py 19 项   免逐槽提取路线筛选（P0）：路线 a/c 必须继续对拍失败（纯 jax 复算）、批间差参与判定、SWAR 树同语义比价
tests/test_geo_rr22_coverage.py   12 项   Geo-RR22 覆盖关系判定实证：国标前缀性质（构造 + 21973 条真实码）、本项目码解不出真实层级、剪枝入口不得出现
tests/test_protocol_registry.py   23 项   协议元数据单一来源、结果语义、Planner 与后端交叉一致
tests/test_protocol_coverage.py   13 项   协议覆盖镜像：登记协议必须显式归类（已验证/不可执行/无密码学保护）、登记即有真机用例（扫描用例的 parametrize 源须恰为 SPU_PROTOCOLS），不许静默滑过
tests/test_protocol_selection.py  21 项   MPC 协议按实测代价选择、REF2K 不自动选中、拒绝信息可操作、排序第二协议真跑
tests/test_protocol_validation.py 47 项   统一 capability validation（Phase 3）：协议/family/world_size/field/语义/参数编译期拒绝、自动选中协议同样复核、Runtime fail-fast 契约
tests/test_mpc_execution_chain.py  9 项   Planner → Runtime 的 MPC 协议闭环（Phase 4）：逐步协议一致、两步不同协议各跑各的、显式覆盖、闭环核对器正反例
tests/test_jax_backend.py         48 项   生成器、可追踪性、原语核对、与明文对拍、TemporalOverlap 两套电路等价
tests/test_spu_backend.py         52 项   协议/环宽规范化、能力门控、私有接口与共享库回归、SPU 实跑、sweep 电路真机
tests/test_spu_profile.py         13 项   SPU 通信量剖析：pphlo 日志解析与 fd 级捕获（P2-2）
tests/test_benchmark_mpc.py       70 项   MPC 基线：用例构造、协议×算子×环宽矩阵、通信量采集、策略 A/B、诚实留空、四个探针开关的产物路径与互斥
tests/test_benchmark.py           33 项   PSI 基线生成器确定性/合法性、记录 schema、unavailable 诚实规则、写入器、真实 RR22 记录
tests/test_benchmark_schema.py   48 项   统一 benchmark metadata（Phase 7）：共有 13 字段投影、缺口在真实产物上锁死、跨协议比较（只比 ok 行 / 不同规模不比 / 重复取中位数）
tests/test_psi_backend.py         82 项   PSI 能力/协议归一化/真实求交/空输入/泄漏面/诚实留空/日志卫生/带噪与精确披露/RR22 参数链路
tests/test_psi_ca_backend.py      16 项   PSI-Cardinality 计数档（第四后端族）：能力/API 核对、只出交集基数、编译期拒绝契约、执行装配与真机判读
tests/test_psi_sum_backend.py     32 项   PI-Sum 交集内求和档（第五后端族）：上游 flag 形态核对、编译期拒绝清单、执行装配与真机结果判读
tests/test_cli_psi_ca.py          13 项   CLI 计数档 `--psi-count`：开关生效、与 `--psi-protocol`/`--psi-curve` 互斥提示、报告
tests/test_cli_psi_sum.py         20 项   CLI 求和档 `--psi-sum`：开关生效、互斥提示、报告
tests/test_psi_runtime_config.py  11 项   PsiRuntimeConfig 拆分/注入/单一配置源/非法 rank/曲线/协议
tests/test_psi_capability.py      14 项   参数级校验（bool 型 rank 拒绝）、三层能力核查、RR22 低通信探测
tests/test_subset_mpc.py          30 项   Contains 密态子集比较：电路原语与注册表一致、模式口径、逐点精确、只有基数进 MPC、退路披露
tests/test_result_policy.py       10 项   结果策略（§15）：业务层暴露与协议内部泄漏分开登记，拒绝伪造
tests/test_party_manager.py        9 项   两方抽象（§16）：WorldConfig / PartyManager 确定性 + 显式拒绝多方
tests/test_party_binding.py        5 项   Party Binding（§14）：PartyInput → notes / JSON；两方上限（§16）
tests/test_runtime_hardening.py    8 项   临时 CSV 工程化（§18）：文件权限 / 规模与磁盘检查 / 清理登记
tests/test_geosot.py              43 项   国标附录 A/B 特征值、层号与层区间互逆、Z/L 位域容量、低空可分辨性、高度带集合语义
tests/test_geosot_layout.py        9 项   GridCode 布局身份：manifest 字段与跨方握手（Phase 4 / 验收 D）
tests/test_geosot_optimizer.py    19 项   Geo-RR22 预处理（§8/§9）：排序 / 去重 / 前缀压缩 / 分桶 + 真实 PSI 对拍
tests/test_height_materialize.py  50 项   height band 方言注册/别名/模块级遍历/参数形式/materialize 诊断/明文一致/示例
tests/test_height_planner.py      12 项   第 6 类失败模式、三维工作流、规划器的高度语义诚实性
tests/test_input_adapter.py       18 项   CSV/JSON/CellSet 输入适配、错误定位到行、布局加载
tests/test_execution_chain.py     15 项   链式执行使用上一步 PSI 输出、样例兜底披露、布局不一致阻断、CLI --input
tests/test_rr22_geosot.py          8 项   GeoSOT-3D 编码 → CellSet → CompactCellSet → RR22 链路（相交/不相交/相同/空集/高位码/重复/排序）
tests/test_end_to_end.py          77 项   全流程、状态表、CLI（协议/曲线/子集/RR22/布局形状/MPC 协议）、六类失败报告、编译入口参数、诊断聚合、确定性
                                  ─────
                                  1151 通过 / 0 跳过
```

在 **WSL2 + Linux + Python 3.11.16 + jax 0.4.34 + spu 0.9.5** 上，
**1151 项全部通过，无跳过**。真实执行隐私协议的用例：

| 类别 | 数量 | 说明 |
|------|------|------|
| SPU(MPC) 真跑 | 15 项 | `test_spu_backend` 3 项 + 子集比较电路 2 项 + 打包探针 3 项 + 取槽单价探针 2 项 + **MPC 协议闭环 5 项**；整数路径误差 0.0 |
| PSI 真机求交（`@needs_psi`） | 23 项函数 | 真调 `psi_execute`，非 mock；含链式执行 / 真实输入绑定 / 布局握手 / 配置闭环 / RR22 / benchmark 记录（grep 口径：`grep -rc needs_psi tests/*.py`） |
| PSI 空输入路径 | 6 项 | 前置判定，不启动协议 |
| PSI 原生日志卫生 | 4 项 | 真机执行 + 校验不落 CWD 日志 |

在未装 SPU 的环境上，相关用例会明确 skip 并说明缺失项。性能基线另有独立运行器
（`tests/benchmarks/benchmark_psi.py`，见 5.8）。

---

## 7. 不支持的算子与当前限制

### 7.1 明确不做的事

- **不做通用 GeoPandas / Shapely 源码自动转换。** 只识别 `geo_privacy` 方言。
  理由：任意几何库的算子集与语义远超密态可行范围，硬转会产出跑不通的代码。
- **不实现任意 Python 控制流的编译。** 依赖运行期数据的 `if/for/while` 直接报错，
  而不是尝试展开（展开会让密文条数按路径数爆炸）。
- **不做浮点坐标表示。** 密态下浮点不可靠，且坐标可反演恢复，等同于未脱敏。
- **不做向量嵌入。** 嵌入可反演恢复坐标，在涉密测绘口径下不可接受；
  唯一可接受的向量是**分箱整数向量**。
- **不修改 SPU 源码，也不重新实现 MPC 协议。**
- **不做未经验证的"完整性填充"。** 每个进入注册表的算子都有对应实现与测试。

### 7.1.1 高度层（Z）语义与 L 位域（本版由 4 位扩到 5 位）

Z 语义缺口已补齐；L 位域的容量缺口在本版也一并闭合（由接入层拍板采用方案 B，
层级上限由 15 抬到 31）。两者放在同一节，因为它们是同一个问题的两面：
高度带要能分层，既要有正确的层号语义，也要有位域装得下那个层级。

依据是 `geosot_work-master` 对国标 **GB/T 40087-2021《地球空间网格编码规则》**
的实现（附录 A 表 A.1 水平剖分、附录 B 式(B.4)/(B.7) 高度域等比剖分）。
本版把其中可直接核验的公式落成 `ir/geosot.py` 的纯函数，不引入任何第三方依赖。

#### 已解决：Z 不再是"无关的整数分量"

上一版的问题是 `plain_intersects([z=15], [z=99]) == False`——Z 只是位域，
同一 XY 上不同高度层被判为不相交，编译产物里也没有任何高度相关参数。
本版把 `Z` 定义为 **GB/T 40087-2021 附录 B 的高度层号**：

```
ir/geosot.py     cell_deg / cells_per_deg / equator_scale
                 height_cell / height_index / height_layer_lower / height_layer_bounds
                 height_layers / height_layer_count / height_layer_fits / validate_height_layer
```

**层边界必须用式(B.4) 的精确等比形式，不能用线性近似。** 参考实现
`geosot_work-master/src/geosot_core.py::scope_geo_num3d` 用 `(h*hc, (h+1)*hc)`
近似层区间（`hc = height_cell(L)` 只是第 0 层层厚）；层厚实际按
`(1+theta0)^(h*cell_deg(L))` 等比增长，于是线性边界在大高度上不再包含原高度：

| 口径 | `height_layer_bounds(height_index(H,L), L)` 包含 H 的失败数 |
|---|---|
| 精确式(B.4) | **0 / 297** |
| 线性近似 `(h·hc, (h+1)·hc)` | **48 / 297**（L=10..32，H=300 m 起就出现，1e6 m 一项占 23） |

（297 = 33 个层级 × 9 个高度，`test_height_index_is_strict_inverse_of_layer_bounds`
逐组断言"返回的层必须真包含原高度"。）

偏差不是可以忽略的小量，随层号迅速放大：

| 层号 @L=9 | 精确下底面 | 线性近似 | 线性偏小 |
|---|---|---|---|
| h=15 | 1890064.8 m | 1669792.4 m | 11.7% |
| h=100 | 29608560.7 m | 11131949.1 m | 62.4% |
| h=255 | 519501834.2 m | 28386470.1 m | 94.5% |

国标特征值逐项对拍通过（与 `geosot_work-master/tests/test_gbt40087.py` 独立复算同一组数字）：

| 量 | 本实现 | 国标值 |
|---|---|---|
| `height_cell(9)` | 111319.49079327373 | `r0·theta0` = 111319.49079327357 |
| `H_255` | 519501834.15823954 | 519501834.1582395 |
| `r_255` | 525879971.15823954 | 525879971.1582395 |
| `H_-256` | -6302106.722602183 | -6302106.722602182 |
| `H_256` | 528680171.1252437 | 528680171.1252437 |
| `floor(n(H_255))` | 255 | 255 |

（最后一行的浮点原始值是 255.00000000000037，第 14 位起为浮点噪声；
取 `floor` 而非 `round` 是刻意的——层号是左闭右开区间的下界，
`round` 会把 `h=0 @L9` 的 0.95 层误判为第 1 层。）

**关键在于高度带如何参与判定。** 把高度带物化成**层集合**（每层一个 64 位码）
之后，高度带重叠就退化为集合交：

```
航线 0-20000 m (11 层, Z=0..10)  ∩  管控 5000-9000 m (3 层, Z=2..4)  ->  公共层 3，相交
同一 XY、层号不同                                                      ->  不相交（三维语义生效）
```

因此**不需要引入 `height_band: [z_lo, z_hi]` 区间参数，也不需要 MPC 区间比较**。
上一版 README 把"带重叠需要额外的 MPC 比较"当作待解问题；实测表明那是把一个
可被 PSI 直接承担的集合问题换成了更贵的算术问题。这一结论使第 6 节 PSI 族的
泄漏面无需扩大。

反面对照（`test_endpoint_only_representation_is_a_false_negative`）：若只把带的
两端各编一个码，中间层被漏掉，`0-20000 m` 与 `5000-9000 m` 会得到 `False`
这一**假阴性**。这是"必须展开成层集合"的直接证据。

业务侧写法（不需要知道层号，也不需要枚举层）：

```python
from geo_privacy import geo

route = geo.height_band(x=21861, y=27702, height_min=0,     height_max=20000, level=15)
zone  = geo.height_band(x=21861, y=27702, height_min=5000,  height_max=9000,  level=15)
geo.intersects(route, zone)          # True（11 层 vs 3 层，公共层 3）
route.intersection(zone).cardinality()   # 3
```

上面这段在**业务侧直接跑**（明文语义）与**送进编译器**是同一个写法，
两者给出同一套层集合。编译结果（`geo-secure build` 的最终表）：

```
Operation   Representation  Backend    Status
----------  --------------  ---------  ---------------
HeightBand  CompactCellSet  Plaintext  plaintext-local
HeightBand  CompactCellSet  Plaintext  plaintext-local
Intersects  CompactCellSet  PSI        verified
```

读法：两个 `HeightBand` 在本方明文完成（`plaintext-local`，本就不该有密态
执行），密态边界落在 `Intersects` 上，由真实 PSI 求交验证（`verified`）。

**高度带重叠 = 层集合有交**，因此这一步不需要任何密态区间比较；
若把高度带写成 `[z_lo, z_hi]` 区间参数，反而要把一个可被 PSI 承担的
集合问题换成更贵的算术问题。

##### 一条曾存在的**假验证**（已修）

本节此前展示的写法有个隐蔽后果，必须记下来：

* 模块顶层的 `geo.height_band(...)` 调用**根本不被遍历**，方言表里也没有它。
  于是编译**报成功**、`Intersects` 显示 `verified`，而高度带从未进入 Geo-IR：
  判定退化成"整条航线格网集合 vs 管控区格网集合"，高度信息被丢掉却是静默的。
* 同样写法放进函数体则报 `GEO_OP_UNSUPPORTED`；在内联位置又会被当成
  SECRET 实参。**同一种文档写法有三种不同结果**，其中一种还是假成功。

三处根因与对应修法：

| 根因 | 修法 |
|---|---|
| 方言表没有 `height_band` 登记 | 登记为 `HeightBand`（物化算子），并补 `altitude_band` / `elevation_band` / `高度带` 别名 |
| `suggest_ops` 对它返回空 → 报错时给不出替代算子 | 在 `ALIAS_TABLE` 补高度带语汇 |
| `FrontendAnalyzer` 缺 `visit_Module`，模块顶层语句从不被遍历 | 加 `visit_Module`，只收方言调用；不 lint `__main__` / `try: import` 之类脚手架 |

顺带闭合的一条检查：第 6 类失败（层号超出 Z 位域）此前**读不到 params**，
从用户源码无法触发；物化算子现在会把 `level` / `height_max` 写进 `params`，
于是 `level=23` 这类写法会被如实拦下：

```
[ERROR] HEIGHT_LAYER_UNSUPPORTED @ vertical.py:2:8:
  算子 HeightBand 的高度带 [0, 1000] m 在层级 L=23 上需要 8 位层号，超出 Z7 位域
    建议: 把层级降到 L<=22（该高度带在 L=22 上只需 7 位），或把高度带收窄到实际管辖区间
```

#### 已解决（本版）：L 位域 4 → 5 位，低空段终于有层分辨率

上一版把这一条标为"不在本编译器可修范围"，因为它要动已固化的主键布局——那是
落库即难回改的决策。本版由接入层拍板**采用方案 B**：

```
旧: X17 | Y17 | Z7 | L4 | Toff14 | Lt4 | ver1   -> 层级上限 15
新: X17 | Y17 | Z7 | L5 | Toff14 | Lt4          -> 层级上限 31
```

`ver` 恒为 0，其位并入 `L` 的高位，因此这是**唯一不付出任何代价**的方案：
Z7 与 Toff14 都不动。四个候选的实测对比（数字由 `ir.geosot` 的真实公式算出）：

| 方案 | 布局 | 层级上限 | 0–1000 m 可用最高层级 | 该带层数 | 第 0 层层厚 | Toff 跨度 |
|---|---|---|---|---|---|---|
| 旧 | `X17\|Y17\|Z7\|L4\|Toff14\|Lt4\|ver1` | L15 | L15 | 1 层 | 1839.6 m | 11.38 天 |
| A | `X17\|Y17\|Z6\|L5\|Toff14\|Lt4\|ver1` | L31 | L21 | 33 层 | 30.7 m | 11.38 天 |
| **B（采用）** | `X17\|Y17\|Z7\|L5\|Toff14\|Lt4` | L31 | **L22** | 66 层 | 15.3 m | 11.38 天 |
| C | `X17\|Y17\|Z7\|L5\|Toff13\|Lt4\|ver1` | L31 | L22 | 66 层 | 15.3 m | 5.69 天 |

A 严格更差（牺牲 Z7 只换来 L21）；B 与 C 的高度能力等价，而 C 要多付一半的
时间跨度。**真正的绑定者随后从 L 换成了 Z**：0–1000 m 在 L23 上需 8 位层号，
Z7 装不下，所以该带的可用上限是 L22。

**收益是语义的，不只是"能编更高层级"。** 旧上限 L15 上，`0–100 m` 与
`800–1000 m` 都塌缩成第 0 层、得到**同一个码**，于是判为相交——一个真实存在的
**假阳性**：竖直升离的两段空域会被报成冲突。

```
L15  航线 0-100 m   -> 1 层，层号 [0, 0]
     管控 800-1000 m -> 1 层，层号 [0, 0]    -> 同一个码      -> intersects = True    （错）
L22  航线 0-100 m   -> 7 层，层号 [0, 6]
     管控 800-1000 m -> 14 层，层号 [52, 65] -> 层号区间不相交 -> intersects = False  （对）
```

固化在 `test_L15_collapses_two_vertically_separated_bands` 与
`test_L22_separates_two_vertically_separated_bands`。

**代价必须写清楚：所有已生成的码都变了。**

```
encode_grid_code(x=21861, y=27702, z=15, level=9, toff=2160, lt=4)
  L4 布局: 0x2AB29B0D87C90E08
  L5 布局: 0x2AB29B0D87A48704
```

码位重排不是向后兼容的改动。PSI 走 CSV 交换这些 64 位码，两方布局不一致
**不会报错、只会静默算错**。因此本版把布局从"注释"抬成"可核对的量"：

1. `ir.values.GRID_CODE_LAYOUT` 给出规范写法（`X17|Y17|Z7|L5|Toff14|Lt4`），
   `MAX_ENCODABLE_LEVEL` 给出层级上限（31）；
   `geo-secure build` 的 PSI 能力块会把它**打出来**，便于和对端逐字比对。
2. 位段位移**由布局推导**（`ir.values._derive_shifts`），不再手写第二份——
   两份并行定义正是布局漂移的来源：改了位宽却忘改位移，编码会**静默错位**。
   `test_field_bitmasks_partition_the_key` 断言各段的位掩码两两不重叠、
   并集恰好 64 位。
3. 交付样例已按新布局重算：`outputs/格网数据样例_明文与密态映射_v6.json`
   （`supersedes: v5`）。生成器 `work/build_sample_v5b.py` 的位段编码改为直接调用
   `ir.values.encode_grid_code`，不再手写位移表达式——布局只剩一个来源，
   从根上消除「改了布局、忘了改生成器」这类漂移。

### 7.2 环境状态（已解除）

**Windows 原生环境**（仍不可运行，无需处理）：

```
状态   : unavailable
平台   : Windows / AMD64
spu    : 未安装
阻断项 :
  - 原生 Windows 无 libspu 原生库（需 WSL2 / Linux）
```

**WSL2 / Linux 环境（已验证）**：

```
状态   : available
平台   : Linux / x86_64 / Python 3.11.16
jax    : 0.4.34（私有接口全部就位）
spu    : 0.9.5（libspu 已加载）
阻断项 : 无
```

本地真实环境报告：`docs/spu_capability_report_wsl.json`。
下面是实际执行结果（节选）：

```
Operation   Representation   Backend  Status
DistanceLE  QuantizedVector  MPC/SPU  verified
```

与文档保持一致的约束仍然生效：在**无法运行 SPU 的环境**上，
`run_spu_simulation` 依然返回 `status="unavailable"` 并附完整阻断清单，
**不返回任何推测数值**——宁可留空，不给假数据。

在受支持环境重现：

```bash
# WSL2 / Linux 上（Python 3.10 或 3.11）
pip install -r requirements-spu.txt      # spu==0.9.5 / jax<=0.4.34 / numpy<2
python -m pytest tests/ -q               # 1151 项全部通过（0 跳过）
python -m geosecure.cli build examples/distance_check.py
```

### 7.3 PSI 族算子为何没有 JAX 实现

`Intersects` / `Contains` / `CellSetIntersect` 属于集合运算。交集性是**组合问题**，
不是逐元素算子，`jax.numpy` 里没有对应原语。

- 在 planner 中它们被登记为 `has_jax_impl = False`；
- 在 JAX 代码生成阶段被记录进 `skipped` 并附上原因，
  而不是硬凑一个"能跑但语义不对"的张量表达式；
- 改由 `backends/psi_backend/` 调用官方 `spu.psi.psi_execute` 真实执行，
  在流水线中占两个独立阶段：`PSI capability check` / `PSI simulation`；
- `backends/jax_backend/reference.py` 里保留了 O(N·M) 广播比较版本，
  仅用于**语义对照与测试**，不作为密态路径。

一图说明两条路径的分工：

```
                       ┌─ DistanceLE / WeightedSum / TemporalOverlap
Python → Geo-IR → Plan ┤      → JAX 代码生成 → jax.jit → SPU(MPC) 模拟
                       └─ Intersects / Contains / CellSetIntersect
                              → PSI（CSV 两方求交）→ 语义判定
                                   └─ 只有 Contains 多一轮：
                                      子集判定 → MPC 基数等值（见 7.6）
```

### 7.6 PSI 的泄漏面（必须知情）

PSI 的标准语义是"接收方得到**交集本体**"，这比布尔结果泄露更多：

| 算子 | 业务想要 | 实际暴露给接收方 |
|------|----------|------------------|
| `Intersects` | 一个布尔 | 交集**本体** + 交集基数 |
| `Contains` | 一个布尔 | 交集本体 + 一个布尔（子集判定走 **MPC 基数等值**） |
| `CellSetIntersect` | 交集本体 | 交集本体（无额外泄露） |

`Contains` 在交集之上还需一次子集比较。它现在**默认在 MPC 里完成**
（`backends/psi_backend/subset_mpc.py`）：

```
Contains(outer, inner) ≜ inner ⊆ outer  ⟺  |outer ∩ inner| == |inner|

参与方 0 私有输入  k = |outer ∩ inner|   ← 它本来就从 PSI 拿到交集，不新增泄露
参与方 1 私有输入  n = |inner|
MPC 电路输出一个比特  k == n
```

这样做的理由是**旧写法装不进两方部署**：`inner <= set(交集)` 需要把 `inner` 与交集
放进同一个进程，而 `inner` 属于参与方 1——真装机上要么把 `inner` 整个交给接收方，
要么把交集交给参与方 1。换成 MPC 后两侧各只提供**一个整数**，
输出只有一个布尔（已由 `tests/test_subset_mpc.py` 用 spy 钉住：进电路的只有两个基数，
格网码一个都没进去）。

**这不是把泄漏面清零，只是换掉其中一条。** 交集本体仍然按 PSI 的标准语义交给接收方。
两种形态各自登记、不合并：静态表写核心，`PsiRunResult.subset` 带出这一次实际用的是
哪条路（`mpc` / `plaintext` / `plaintext-fallback`），`reveals` 是两者拼接。

退路也如实说：MPC 不可用（没装 SPU、环宽/协议不匹配）时会自动退回明文，
但 mode 标成 `plaintext-fallback`，状态词记 `subset-plaintext`，notes 里写明
「两方部署下该退路需要把 inner 交给接收方」。显式选择明文用 `--psi-subset plaintext`。

每个算子的泄漏面登记在 `PSI_OP_LEAKS`，随 `PsiRunResult.reveals` 输出，
并出现在 CLI 的 `PSI simulation` 段与编译结果的 JSON 里。

**计数档是第三条泄漏语义。** `--psi-count psi-ca`（PSI-Cardinality，§5.9）不产出
交集本体：协议只计算并返回基数（登记码 `count-only`，与上表的 `intersection-body`
并列且互不相同）。它把 `CellSetIntersect` 的暴露面收窄为"一个整数"，代价是下游不能
再消费集合；`Intersects` / `Contains` 这类需要布尔解释的算子被显式拒绝。

**求和档是第四条泄漏语义。** `--psi-sum pjc`（PI-Sum，§5.10）多交出一个
**交集内关联值之和**（登记码 `count+sum`）。它比计数档多一个数，也就多一条反推
通道：上游 README 自陈，某个唯一的大值、或过小的交集，都能由这个和推断出成员
是否在交集里，而其缓解措施（加噪、剪除离群值、过小交集中止）**上游未实现**。
因此"只出一个整数"与"出两个数"分开登记、不共用状态词（`count-only` /
`count-and-sum`）。

**精度是第二条独立披露面。** 泄漏面说的是"暴露了什么"，精度说的是"推出来的结论
可不可信"。选用 `PROTOCOL_DP` 时两者同时变差：它既把 A 的元素当成交集元素交出去，
又会漏报真实交集。两者各自登记（`PSI_OP_LEAKS` / `PSI_PROTOCOLS_WITH_NOISE`），
不合并——合并后总有人只读到一半。

### 7.4 环宽与溢出

`grid_code` 是 64 位定长键。在 `FM64` 环上做**大小比较**（而非等值比较）时，
靠近 2⁶⁴ 的值会翻转为负数导致判定错误。

- 因此 `Intersects` / `CellSetIntersect` 的能力核查会输出 FM128 告警（它们没有 MPC 段，
  这条告警是"一旦改走 MPC 就要注意"的条件提示）。
- 用 PSI 时不存在此问题（等值比较不涉及大小语义）。
- **`Contains` 不再吃这条告警**：它的 `grid_code` 全程不出 PSI，进 MPC 的只有
  基数 `k`/`n` 两个小整数，照抄溢出告警属于误导。能力核查对它改说的原话是
  "grid_code 不进 MPC"。
- 若确实要把 64 位键放进 MPC 做大小比较，应切 `FM128`，代价是 `b` 翻倍。

### 7.5 时间语义的历史修正

`TimeInterval.overlap` 早期实现用"区间元组集合求交"判断重叠，
那只能识别**完全相同**的区间，部分重叠会漏判。
现已改为标准判据 `l.start < r.end and r.start < l.end`，
在明文 / JAX / `geo_privacy` 三处保持一致，并有参数化测试覆盖
部分重叠 / 相邻 / 嵌套 / 远离四种情形。

---

## 8. 下一步扩展点

### 8.0 已知待办（本版遗留，按优先级）

| 优先级 | 待办 | 说明 |
|--------|------|------|
| 高 | **Geo-RR22 输入预处理** | 接入闭环、性能基线、输入预处理均已完成（见 5.6 / 5.8）；去重（正确性前提）/ 排序 / 无损前缀压缩 / bucket 分区已实现并**默认接入** PSI 输入路径（`geosot_optimizer`，`GEOSOT_OPTIMIZER_VERSION = "0.1.0"`）；**候选集剪枝仍不实现**：前置的「覆盖关系判定」实证报告已出（本版 P0，`docs/GEO_RR22_COVERAGE.md`），结论是**当前口径下不可实现**——国标码有前缀覆盖规则（21973 条真实码实测），但本项目 64 位码是字段切分布局、解不出真实层级，且 `(X, Y)` 与 `L` 的口径未声明，同一条 x 在 9 级与 15 级上被复用。剪枝在没有无损规则前等于把近似引进精确路径（任务文档 §10）。设计见 `docs/GEO_RR22_DESIGN.md` |
| 中 | ~~编译期按方案实算元素数 K~~ | **已闭合（本版 P1）**：K 现在有两条来路——算子的 `params.K`（调用方显式声明）与 `--layout-shape` 的 `attributes`（与位平面布局同一口径），并在 `estimated_cost` 里写明 `K_basis`。修掉了"布局按 attributes=4 算、位宽按 K=3 算"这一处同报告内自相矛盾；`TemporalOverlap` 的电路选择也接上了 `size_hint`（`--layout-shape candidates=N` → `sweep`） |
| 低 | 三维示例接入真实 3D 码集 | `examples/vertical_conflict.py` 已跑通编译与 PSI 真实执行，但 `DEFAULT_EXAMPLE_INPUTS` 仍是二维码；换码集即可 |

> 已闭合（上一版遗留）：**高度层（Z）语义**。Z 已定义为 GB/T 40087 附录 B 的
> 高度层号，层边界取式(B.4) 精确口径，高度带按层集合参与判定、由 PSI 承担。
> 见 7.1.1。

> 已闭合（本版）：**L 位域 4→5 位**。原先被标为"接入层规范决策、不在本编译器
> 可修范围"的容量缺口，已按方案 B 落地：`ver` 位并入 L 高位，层级上限由 15 抬到
> 31，低空 0–1000 m 最高可用 L22（66 层）。顺带修掉一个真实的**假阳性**——
> L15 上竖直升离的两段空域会被编成同一个码。见 7.1.1。

> 已闭合（本版）：**`geo.height_band` 的假验证**。文档写法此前在模块顶层
> 会被静默丢弃（编译报成功、状态显示 verified，高度带却没进 IR）。
> 现在它登记为 `HeightBand` 物化算子，三种写法（模块顶层 / 函数体内 / 内联）
> 都进 Geo-IR，并有 `examples/altitude_band.py` 与 `tests/test_height_materialize.py`
> 守着。顺带闭合：第 6 类失败从用户源码可触发。见 7.1.1。

> 已闭合（本版）：**`Contains` 的密态子集比较**。第二次比较从明文集合运算改为
> MPC 基数等值（`k == n`），新状态词 `subset-plaintext` 与 `--psi-subset` 开关见 7.6。
> 同时修掉一条错话：`Contains` 的能力核查不再声称 `grid_code` 有 FM64 溢出风险。

> 已闭合（本版）：**层面 1 的 PSI 协议开关**。此前 `--protocol` 只作用于
> SPU（MPC）阶段，PSI 阶段永远走默认 `ECDH + SM2`，无法切换。
> 现在 `--psi-protocol` / `--psi-curve` 一路贯到 `run_psi_intersection`。
> 顺带闭合两个相邻缺陷：`--protocol SPDZ2K` 这类非法 SPU 协议名会让整条
> 流水线带 traceback 崩掉（现为可读错误 + 退出码 1）；失败时总结行
> 报"0 个错误"的自相矛盾表述。见 5.5。

> 已闭合（本版）：**"能跑"与"结果准"混为一谈**。`PROTOCOL_DP` 是差分隐私 PSI，
> 上游源码与实测都确认它**会**向交集注入非成员、**会**漏报真实冲突（真交集 2 个
> 时约 10%）。它此前被登记为"不读曲线"，并被一条"所有可执行协议都应与明文一致"
> 的用例顺带断言，造成约 2/12 的**随机失败**——而它掩盖的是语义差异，不是缺陷。
> 现在拆成两条独立披露：`PSI_PROTOCOL_WORLD_SIZE`（能不能跑）与
> `PSI_PROTOCOLS_WITH_NOISE`（跑出来准不准）；状态词新增 `executed-noisy`；
> 曲线关系改为 `required` / `ignored` / `implicit` 三分类。见 5.5 与 6.1.1。

> 已闭合（本版）：**RR22 协议接入**（课题下一阶段的接入闭环）。`PROTOCOL_RR22` 从
> "协议名已登记"升级为"可配置 → 可编译 → 可执行 → 可验证"：`PlannedStep` 携带
> `protocol` / `protocol_params`；`OperatorRule` 登记 `default_protocol` /
> `protocol_candidates`；Compiler / CLI 新增 `psi_rr22_low_comm_mode` /
> `--psi-rr22-low-comm-mode`；runtime 向 `PsiProtocolConfig` 注入 `Rr22Rarams`；
> 运行结果 / 计划 / JSON / CLI 四处都带协议参数。RR22 仍走 PSI 后端（`spu.psi`），
> JAX Backend 与 SPU MPC `ProtocolKind` 均未改动；新增 `tests/test_rr22_geosot.py`
> 与 `examples/rr22_grid_intersection.py`。见 5.6 与 `docs/RR22_INTEGRATION.md`。

> 已闭合（本版）：**下一阶段任务 Phase 1–5** —— 配置闭环（`PsiRuntimeConfig` 单一配置源 + 三层
> 协议能力核查 + 协议候选校验）、真实 GeoSOT 输入（`--input` / InputAdapter）、链式执行
> （CellSetIntersect 输出喂下游）、布局握手（LAYOUT_MISMATCH 前置拒绝）、性能基线
> （`tests/benchmarks/benchmark_psi.py` → `docs/psi_benchmark_baseline.json`）。
> 版本矩阵见 `docs/VERSION_COMPATIBILITY.md`；重复键实测（RR22/KKRT 报错）与新阶段设计
> 见 `docs/GEO_RR22_DESIGN.md`。

> 已闭合（本版）：**MPC 协议"按代价/能力选"**。`--protocol` 缺省从固定
> `ABY3` 改为**自动**（`planner.select_mpc_protocol` 按 `docs/mpc_comm_baseline.json`
> 的实测发送+接收排序）；候选集带实测代价（`mpc_protocol_candidates_for`）；
> 无密码学保护的 `REF2K` 留在候选里但不参与自动选择，显式选它会附披露；
> 选择依据落到 `PlannedStep.mpc_protocol_basis` 与 CLI 输出；拒绝信息带
> "能用什么、多贵、为什么排除"。见 5.5 与 `docs/MPC_BENCHMARK_PROTOCOL.md` §8.5。

> 已闭合（本版）：**位平面布局（D3）的预测层**。`planner/layout.py` 按算子归约轴
> 选 L1 / L2，条数公式与课题交付物（`outputs/格网数据样例_明文与密态映射_v5.json`
> 的 `cost_prediction`）逐项复算一致（196000 / 49000 / 192 / 1020.8×）；
> `estimated_cost` 增补 `layout*` 键；CLI 增 `--layout-shape`。
> **未落地的是打包电路本身**——本项只调整密文条数，不是实测通信量。见 8.4 与
> `docs/BITPLANE_LAYOUT.md`。

> 已闭合（本版）：**打包收益的前提实测（P6 探针）**。位平面布局的条数模型有一个
> 此前只能标"未验证"的前提——"通信量按环元素计费、与输入位宽无关"。现在用
> "秘密 × 秘密"的逐元素乘法实测：int8 / int32 / int64 在**同一元素数**下通信量之比
> 全为 1.000（唯一通信原语是 `multiply`），**16 B/元素** 在 512 / 1024 / 4096 元素上
> 恒定。产物 `docs/mpc_packing_probe.json`，跑法 `--packing-probe`。
> **仍不是打包收益的实测**：探针不是打包电路，槽内归约未测，收益只作上界。
> 见 8.4 与 `docs/BITPLANE_LAYOUT.md` §5。

> 已闭合（本版）：**打包电路的取槽代价（P7-P0 探针）**。P6 之后紧接着量了打包电路
> 必须付的那一步——把 8 位值从环元素里取出来（右移 + 掩码）：四个变体 A/B 实测
> （产物 `docs/mpc_slot_cost_probe.json`）`mul_only`（秘密 × 公开常数）**0 B**、
> `mask_only` 624 B/元素、`shift_only` 1888 B/元素、`shift_mask`（全量提取）
> **1424–1496 B/元素 = 纯乘法 16 B/元素 的 89–94 倍**（区间来自批间差）。结论是**前置判据**：元素数下降的
> 倍数必须先超过这个量级，打包才可能划算。见 8.4 与 `docs/BITPLANE_LAYOUT.md` §5.1。

> 已闭合（本版 Phase 4）：**Planner → Runtime 的 MPC 协议闭环**。执行期协议此前只取
> **方案里第一个** MPC 步骤的协议并套到所有步骤上——同一程序里两个 MPC 步骤各选
> 不同协议时，第 2 步会被**静默换协议**（与验收口径冲突）。现在改为**逐步解析**
> （`Compiler._step_protocol`：显式指定 > 方案实测选择 > 登记默认值），并新增
> `_check_mpc_closure` 在阶段 6 兜底核对，漂移即阶段报错。对拍覆盖
> `planned.mpc_protocol == runtime.protocol`、环宽、参与方数量；编译 JSON 的 plan 步
> 带出 `mpc_protocol` / `mpc_protocol_basis`。真机证据：收窄候选构造出两步不同协议
> （WeightedSum→SEMI2K〔2 方〕/ TemporalOverlap→ABY3〔3 方〕）后**各跑各的**、
> 结果与明文逐位一致（err=0.0）。见 5.7 与 `tests/test_mpc_execution_chain.py`（9 项）。

### 8.1 接入新增隐私后端

`planner.registry.OperatorRule` 的 `backend` 字段是自由字符串，
`backends/` 下按目录扩展即可：

```
backends/
├── psi/        ← 新：真实 PSI（OPRF / RR22 类协议）
├── fhe/        ← 新：FHE（BFV / CKKS，做"公开参数 × 密态数据"的线性部分）
└── tee/        ← 新：TEE
```

需要同时补：
1. `OperatorRule.backends` 里的后端声明；
2. 该后端的 `run_*` 执行入口；
3. `validator` 里对应的能力核查；
4. 测试（明文 / JAX / 新后端三方对拍，明确容差）。

分工原则（来自课题既有结论）：**FHE 做线性部分，MPC 做非线性判定与保密权重。**

### 8.2 新增地理算子

1. 在 `geo_privacy/geo.py::GEO_OPERATIONS` 登记方言（op 名、arity、参数类型、返回类型）；
2. 在 `planner/registry.py` 登记表征、后端、安全级别、四量代价；
3. 在 `backends/plain/` 写明文实现；
4. 若能降级为张量表达式，在 `backends/jax_backend/codegen.py` 加生成器；
   否则在 `skipped` 逻辑里说明原因；
5. 在 `semantic/semantics.py` 补谓词与别名；
6. 加测试。

方言表是**单一真源**：`frontend/dialect.py` 从 `GEO_OPERATIONS` 自动构建，
两处不会漂移。

若新算子是**物化算子**（在本方明文产出码集合，不进密态），登记方式多三项：

```python
"altitude_band": {
    "op": "AltitudeBand",
    "arity": 5,
    "materializes": True,          # 声明为物化算子
    "value_params": ("x", "y"),    # 进 Geo-IR inputs：被编码的数据
    "config_params": (...),        # 进 Geo-IR params：物化配置，不参与敏感度判定
    "required_params": (...),      # 无默认值的那些
    "defaults": {"toff": 0, ...},  # 必须与 facade 签名逐字一致（有测试对拍）
    ...
}
```

同时要在 `planner/registry.py` 里把 `backend` 设为 `"Plaintext"`——
能力核查、JAX 生成、模拟阶段都按这个字段跳过它，并在最终状态表里记
`plaintext-local`。**漏掉这一步会让它被当成密态算子**，状态表也就不可信了。

两条硬约束（都有测试守着）：

1. facade 上的每个 `geo.<method>` 必须有归属——算子 / 物化 / 明文工具
   （`PLAINTEXT_UTILITIES`）三者之一，不允许存在"文档里有、方言里没有"的灰色状态；
2. 物化算子的**产出敏感度必须由 `inputs` 推导**，不得写死级别。

### 8.3 提升 SPU 侧覆盖

- ~~在 WSL2 / Linux + Python 3.11 环境下接通真实 SPU 模拟，
  把 `tests/test_spu_backend.py::TestRealSpuSimulation` 从 skip 变为通过~~
  ——**已闭合**：本机 `1151 passed / 0 failed / 0 skipped`，无一条真实执行用例被跳过。
- 补 `FM128` 路径测试（64 位键的溢出场景）。
- ~~补不同协议的代价实测（`semi2k` / `aby3` / `cheetah`）~~ ——**已闭合（本版 P1）**：
  `tests/benchmarks/benchmark_mpc.py` 已把 3 个 MPC 算子 × 5 个 SPU 协议 × 3 个环宽
  实测入库，并与 planner 的**预测位宽**对账（`docs/MPC_BENCHMARK_PROTOCOL.md`）。
  对账同时找出两处真实缺口：近似除法（非确定偏差）与 `WeightedSum × FM32`
  的环宽下限；重复实验（×5 / ×30）已完成。
  **两处缺口已在 P2-1 闭合**（`scale` 改编译期常量 → 生成代码无除法：
  与明文逐位一致、FM32 可用、PPHLO 2400→938、通信量降 56–93%、CHEETAH 快 13×）。
  ~~把通信量接进基线~~ ——**已闭合（P2-2）**：`--comm` + `capture_comm=True`
  采 SPU 的 pphlo profiling，落到每条记录，产物 `docs/mpc_comm_baseline.json`；
  实测确认墙钟排不出协议优劣、`TemporalOverlap` 上墙钟排序被通信量反转。
  ~~`TemporalOverlap` 二次电路改造~~ ——**已闭合（P3）**：新增 `sweep` 电路（排序归并 + 前缀扫描，O((N+M)·log(N+M))），K=1024 真机跑通，原来那条「K 上限被锁在 32 量级」的限制因此解除。但排序在 MPC 里贵（实测交叉点 K≈128–256），故默认 `auto` 以 K=256 为阈值、拿不到规模就退回 pairwise，两套电路都保留（见 `docs/MPC_BENCHMARK_PROTOCOL.md` §4.1）。
  ~~未闭合的余项：把电路选择接到实际输入规模上（`size_hint` 目前要调用方给）~~
  ——**已闭合（本版 P1）**：`--layout-shape candidates=N` 现在同时喂给
  `TemporalOverlap` 的电路选择（`generate_for_plan(size_hint=...)`），
  `N ≥ 256` 自动走 `sweep`、否则 `pairwise`；拿不到规模时仍退回既有口径并披露。
  实测：同一示例在 `candidates=1000` 下生成 `sweep`（HLO 4257 → 9364 B，含 `sort`）。
  `size_hint` 仍可被调用方显式覆盖（`generate_for_plan(..., size_hint=…)`）。

### 8.4 位平面与打包布局（D3）

课题的 D3 决策是"位平面双布局：L1（同格网跨属性）与 L2（跨格网位平面）按算子归约方向选择"。

**已落地（本版 P5，预测层）**：`planner/layout.py` 按算子归约轴选布局，
并把条数下降写回 `estimated_cost`。口径来自课题交付物、并由本仓库逐项复算：

| 项 | 课题记载 | 公式 | 复算（1000 候选 × 49 格网 × 4 属性 × b=8） |
|---|---|---|---|
| 逐点条数 | 196000 | `candidates × cells × attributes` | 196000 ✓ |
| L1（同格网跨属性） | 49000 | `ceil(values / attributes)` | 49000 ✓ |
| L2（跨格网位平面） | 192 | `ceil(values / slots) × b`，`slots = 2^ceil(log2 候选) × b` | 192 ✓ |
| L2 下降 | 1020.8× | `values / L2` | 1020.8 ✓ |

```bash
geo-secure build examples/distance_check.py \
    --layout-shape candidates=1000,cells=49,attributes=4,bits=8
#   MPC 协议: DistanceLE → ABY3（按实测代价自动选择）
#   位平面布局: DistanceLE → L2（逐点 196000 条 → 192 条，1020.8×）
```

三条边界（都有测试守着）：**不给形状就不给数字**（只做轴向决策）；
**没收益就退回逐点**（不报小于 1 的"收益"）；**未登记归约轴的算子不替它猜**。

**未落地**：打包电路本身。JAX 生成器与 SPU 执行路径一行未改，因此本项
**只是密文条数的结构调整**。

打包的**前提**已在 P6 实测（产物 `docs/mpc_packing_probe.json`，跑法
`tests/benchmarks/benchmark_mpc.py --packing-probe --repeat 3`，ABY3 / FM64）：
用"秘密 × 秘密"的逐元素乘法（第一版用 `x * 2`——乘公开常数，实测通信量恒为
0 字节，什么也测不出）分别扫位宽与规模——

| 用例 | 元素数 | 通信量 | B/元素 |
|---|---|---|---|
| `mul_ss int8 / int32 / int64 N=4096` | 4096 | 65 536 B（三者相同） | 16.00 |
| `mul_ss int64 N=512` | 512 | 8 192 B | 16.00 |
| `mul_ss int64 N=1024` | 1024 | 16 384 B | 16.00 |

即 **SPU 按环元素计费、与输入位宽无关**（8 位值装进 64 位环元素不额外收费），
且通信量对元素数线性。"8 个 8 位值挤进 1 个环元素"因此省下 8 倍条数，
上界通信量 65 536 B → 8 192 B。

**仍然不是打包收益的实测**：探针跑的是逐元素乘法，**不是打包电路**；槽内归约
（L1 的加权和、L2 的跨候选聚合）要额外通信，那部分必须另行实测。所以
`planner/layout.py` 给出的收益一律标注为**上界**。详见 `docs/BITPLANE_LAYOUT.md` §5。

**再进一步（P7-P0）**：打包电路还要把每个 8 位值**取出来**（右移 + 掩码），
这一步在 MPC 里走协议，不免费。四变体 A/B 实测（产物 `docs/mpc_slot_cost_probe.json`，
`tests/benchmarks` 之外由 `tests/test_slot_cost_probe.py` 守着）：

| 变体 | 电路 | 通信量 | B/元素 |
|---|---|---|---|
| `mul_only` | `Σ x × c_s`（秘密 × 公开常数） | 0 B | 0.00 |
| `mask_only` | `Σ (x & 0xFF) × c_s` | 210 944 B | 824.00 |
| `shift_only` | `Σ (x >> 8s) × c_s` | 483 328 B | 1888.00 |
| `shift_mask` | 全量取槽（打包电路的那一步） | 382 976 B | **1496.00** |

（2026-10-08 以 `--repeat 3` 取中位数重跑；P7-P0 首次为单次读数 624 / 1888 / 1424。
同一条电路批间差实测 5%–15%，所以下表只做量级判断。**本版把重复次数变成了真的重复
执行**：此前 `--repeat` 只记进记录、没真跑。）

取槽步是纯乘法的 **89–94 倍**单价。所以"元素数 ÷8"这个上界**只在取槽能摊薄时才兑现**：
打包要划算，元素数下降的倍数必须先超过这个量级——L2 的 1020.8× 是**条数比**，
与 89–94 倍同量级，不能当成通信量比。这条把下一步从"照着公式写电路"改成
"先做免逐槽提取的设计筛选"（见 `docs/BITPLANE_LAYOUT.md` §7）。

**本版 P0：把这三条路线筛掉，并逐原语真机核验。** 探针
`tests/benchmarks/benchmark_mpc.py --slot-reduction-probe --repeat 3`（产物
`docs/mpc_slot_reduction_probe.json`，`tests/test_slot_reduction_probe.py` 守着）把
"能不能不逐槽取数"变成可执行的电路，并**拿目标语义对拍**（参考是"应该算出什么"，
不是"它自己那套公式"）：

| 路线 | 变体 | 电路 | 对拍 | 通信量 |
|---|---|---|---|---|
| —（基准） | `extract_seq` | 逐槽提取 + 加权 | **通过** | 1424 B/元素 |
| —（批间差） | `extract_seq_dup` | 与基准**逐位相同** | 通过 | 1552 B/元素 |
| (b) 公开权重 | `extract_seq_secret_weight` | 同提取、权重取秘密 | 通过 | 1432 B/元素 |
| (a) 整元素一次乘法 | `fullmul` | `(X · ONES_packed) & 0xFF` | **失败** | 224 B/元素 |
| (c) 不掩码移位 | `maskless_shift` | `Σ_s (X >> 8s) · w_s` | **失败** | 1888 B/元素 |
| 槽内归并 | `swar_tree` | 对数深度掩码+移位+加法树 | 通过 | 1888 B/元素 |

三条结论，都有实测支撑：

1. **路线 (a) 不成立**：一次整元素乘法拿不到 `Σ_s v_s`（槽间交叉项/进位），
   实测算错（`max_abs_error=22`）。要分离各槽，**必须先逐槽提取**；
2. **路线 (c) 不成立**：省掉掩码后高位槽会串进来（`max_abs_error≈1.7e8`），
   而且并不便宜（1888 > 1424）；
3. **路线 (b) 免的是"加权"，不是"提取"**：权重放公开侧与放秘密侧差别
   （1432 → 1424）**小于本探针自测的批间差（8.6%）**，不构成结论。SWAR 树
   也没有换来更少的按位原语（1888 > 1496，同语义对照）。

即：**"免逐槽提取"在三条路线下都不兑现**，打包只能照付取槽单价（1424–1552 B/元素，
与 P7-P0 同量级）。这正是 `docs/BITPLANE_LAYOUT.md` §7 第 5 步的门槛：
打包要划算，元素数下降的倍数必须先超过这个量级。

**逐原语真机核验（P0）**：`--primitive-probe`（产物 `docs/mpc_primitive_probe.json`，
`tests/test_primitive_probe.py` 守着）把 capability 表"登记为已适配"的原语逐个写成最小
电路真机跑一遍。**22/22 全部跑通**（0 error）。单价**按计费维度分两栏读**——
逐元素电路按输入元素计费，收缩类按**输出个数**计费：

逐元素电路（ABY3 / FM64，N=64，3 次中位）：

| 原语 | B/元素 | 原语 | B/元素 | 原语 | B/元素 |
|---|---|---|---|---|---|
| `add` / `sub` / `neg` / `not` / `shift_left` | 0.00 | `mul`（秘密×秘密） | 16.00 | `compare` | 88.25 |
| `max` / `min` / `select` | 104.00 | `shift_right` | 224.00 | `xor` | 448.00 |
| `and` | 464.00 | `or` | 912.00 | `top_k` | 121.31 |
| `div` | 5214.00 | `sort` | 6032.00 | | |

收缩类电路（另一个计费维度，2026-10-08 复测）：

| 用例 | 收缩长度 | 输出个数 | 通信量 | B/输出 |
|---|---|---|---|---|
| `dot` / `dot_long` | 64 / 512 | 1 | 16 B | 16.00 |
| `sum` / `sum_long` | 64 / 512 | 1 | 16 B | 16.00 |
| `matmul`（16×64 · 64×16） | 64 | 256 | 4096 B | 16.00 |

三条**必须记住**的读数：

- `shift_left` / `not` / `add` / `sub` / `neg` 实测 **0 B**——它们是本地线性/逐位运算，
  "打包取槽要付钱"这件事只发生在 `>>`（224 B/元素）与 `&`/`|`/`^`（448–912 B/元素）上；
- **收缩类按输出个数计费、收缩长度免费**：`dot` / `sum` 输出 1 个、实测恒为 **16 B**，
  收缩长度 64 → 512 读数**逐字节不变**；`matmul` 账也对得上（256 × 16 B = 4096 B）。
  单价 16 B 与 P6 的环元素单价一致；
- **一条自我更正**：第一版把 `dot` 的 16 B 除以**输入元素数**得到 **0.25 B/元素**，
  看着"比乘法便宜 64 倍"，于是标成可疑的下限违例。打开 `pphlo` 追踪后确认是
  **归一化选错了维度**——读数没错，**尺子错了**（与 `x * 2` 实测 0 B 同族教训）。
  现在对账按计费维度走（`cost_driver` / `cost_per_unit`），
  `below_multiply_floor` 只在**同一维度**上拦真·不可能读数，
  `contraction_is_free` 固定"收缩长度免费"这条实测结论。
  **但它不是"dot 免费"**：逐槽提取仍按元素计费，§8.4 的打包门槛不变。

> **批间差**：`top_k` / `sort` 的读数在两次探针之间会明显漂移
> （`top_k` 本轮 3 次区间 5840–10650 B，上一轮中位 16348 B）。
> 这两条只做量级判断，不用于排序。

### 8.5 关系稀疏表示与 ZKP

`semantic` 已能产出标准关系三元组。下一步可：
- 按课题口径把三元组编码为定长 `rel_bitset`（关系号 r → 2^r，长度与挂接数 k 无关）；
- 在 PSI 后端上用关系集合求交替代逐条比较；
- 为离散关系输出补 ZKP 证明（审计与血缘）。

### 8.6 CLI 与工程化

- `geo-secure build --emit-jax <path>` 已可导出生成模块；
- `geo-secure build --psi-protocol / --psi-curve` 已可切换 PSI 协议（见 5.5）；
- `geo-secure build --psi-subset` 已可切换 `Contains` 的子集判定路径，
  降级时会反映到状态词（`subset-plaintext`）与泄漏面文本（见 7.6）；
- 切换后**协议语义也跟着变**：带噪协议（`DP`）会被降级为 `executed-noisy` 并
  在两个 PSI 阶段记 warning，替代协议清单里也带 `*` 标注（见 5.5）；
- `geo-secure build --input NAME=PATH` / `--input-layout NAME=PATH` 已可绑定真实
  格网集合与布局握手（见 5.7）；
- PSI 性能基线：`tests/benchmarks/benchmark_psi.py`（见 5.8 与 `docs/BENCHMARK_PROTOCOL.md`）；
- 可扩展：批量编译、代价报告导出、与 CI 集成（把 `geo-secure check` 作为前置门禁）。

### 8.7 统一 benchmark metadata（Phase 7，已落地）

任务文档 §九 要求「PSI / MPC 都能输出统一的 benchmark metadata」，
且「不要为了统一而丢失 PSI 特有指标」。本版把两族记录投影到同一张表：

- 模块 `backends/benchmark_schema.py`：`CommonBenchmarkRecord`（13 个共有字段）
  + `PSIBenchmarkMetadata` / `MPCBenchmarkMetadata` + `metadata.raw`（原记录保真）；
- §九 清单 21 项逐条登记去向（`common:` / `metadata:` / `missing`），
  某族**确实没采集**的指标显式登记成缺口，并由测试在真实产物上锁死；
- 投影器 `scripts/unify_benchmark.py`：只读既有基线，不重跑、不改写产物；
- 跨协议比较 `compare_protocols`：只比 `ok` 行、不同族/算子/规模直接报错、
  同用例重测取中位数并带出样本数与极差。

详见 `docs/BENCHMARK_SCHEMA.md`；测试 `tests/test_benchmark_schema.py`（48 项）。

---

## 附：六类失败模式的错误报告

编译器支持的六类失败，每类都给出
**错误位置 / 问题原因 / 建议替代算子 / 预计隐私计算代价**。

| 编号 | 类型 | 诊断码 | 触发示例 | 严重级别 |
|---|---|---|---|---|
| 1 | 地理算子不支持 | `GEO_OP_UNSUPPORTED` | `geo.buffer_zone(...)` | error |
| 2 | JAX 算子无法追踪 | `JAX_NOT_TRACEABLE` | 依赖数据取值的分支、混用 numpy | error |
| 3 | SPU 当前版本不支持 | `SPU_UNSUPPORTED` | 环境阻断 warning；用到未适配原语 error | 视情况 |
| 4 | 动态 Python 控制流无法编译 | `DYNAMIC_CONTROL_FLOW` | `if flag:` 包裹 geo 调用 | error / warning |
| 5 | 后端没有对应隐私算子 | `BACKEND_OP_MISSING` | IR 中出现未登记算子 | error |
| 6 | 高度层号超出 Z 位域 | `HEIGHT_LAYER_UNSUPPORTED` | 高度带在层级 L 上需要的层号位数 > 7 | error |

实际输出示例（第 1 类）：

```
▸ 地理算子不支持（GEO_OP_UNSUPPORTED）
    [ERROR] GEO_OP_UNSUPPORTED @ user_code.py:3:11: geo.buffer_zone() 不是受支持的地理算子
    原因: 当前方言只覆盖 6 个算子：[...]
    建议: 改用下方的等价算子名，或在 dialect 中登记新算子（需同时补后端实现与测试）
    替代算子: Contains
    预计代价: {'representation': 'CompactCellSet', 'backend': 'PSI/MPC',
               'security_level': 'high', 'N_ct': '1 条/格网', 'b': '64', ...}
```

**严重级别为何要区分**（第 3 类）：环境阻断（未装 SPU / 平台不支持 / 版本不匹配）
不是用户代码的问题，标为 `warning`，用户仍可拿到 IR、方案与 JAX 结果去取证；
原语阻断（生成代码用到未适配原语）是代码问题，标为 `error`。

**编译器绝不自动修改用户代码**，由测试锁定
（`test_compiler_never_modifies_user_source`）。
