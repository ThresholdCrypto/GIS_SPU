# 阶段三改进报告：高度层（Z）语义接入与位域容量约束

**课题**：三维格网数据统一接入隐私计算体系（GeoSOT-3D / DQG-4D）
**交付物**：`GIS_SPU/`（面向地理信息行业的低门槛隐私计算编译器 MVP）
**本轮主题**：参考 `geosot_work-master` 对国标三维格网编码的实现，补齐格网语义中的高度维
**日期**：2026-09-24
**验证环境**：WSL2 / Ubuntu / Linux x86_64 / Python 3.11.16 / `jax==0.4.34` / `spu==0.9.5`

---

## 0. 一句话结论

高度维（Z）此前只是 64 位键里的一个**位域**，不参与任何算子语义；本轮把它定义为
**GB/T 40087-2021 附录 B 的高度层号**，层边界取式(B.4) 的精确等比形式，
高度带**物化为层集合**因而可以直接由 PSI 承担——上一版认为"高度带重叠需要额外
MPC 区间比较"的判断被证伪。

与此同时发现一个**更硬的容量约束**：64 位键里 `L` 只有 4 位，可编码层级被封在 15；
而 L≤15 上整个 0–1000 m 低空带**恒为 1 层**（层号恒为 0）。也就是说，
在当前键布局下三维语义在低空带上退化为"XY 相交即相交"。该约束属**接入层规范决策**，
不在本编译器可修范围内；本编译器改为在编译期把它**显示出来**（新增第 6 类失败模式）。

---

## 1. 参考实现的贡献与分歧

参考实现：`geosot_work-master/src/geosot_core.py`（1369 行）+ `tests/test_gbt40087.py`。

### 1.1 直接采用的部分

| 国标出处 | 公式 | 本项目落点 |
|---|---|---|
| 附录 A 表 A.1 | `cell_deg(L)`、`cells_per_deg(L)` | `ir/geosot.py` |
| 附录 B 式(B.4) | `H(h,L) = r0·((1+theta0)^(h·cell_deg(L)) − 1)` | `height_layer_lower` / `height_layer_bounds` |
| 附录 B 式(B.7) | `h(H,L) = floor(ln(1+H/r0) / ln(1+theta0) / cell_deg(L))` | `height_index` |
| 附录 B 特征值 | `H_255`、`r_255`、`H_-256`、`H_256`、`n(H_255)` | `tests/test_geosot.py`（独立复算，逐项对拍） |

特征值对拍（本实现 vs 国标给定值，两边独立算出同一组数字）：

| 量 | 本实现 | 国标值 |
|---|---|---|
| `height_cell(9)` | 111319.49079327373 | `r0·theta0` = 111319.49079327357 |
| `H_255` | 519501834.15823954 | 519501834.1582395 |
| `r_255` | 525879971.15823954 | 525879971.1582395 |
| `H_-256` | -6302106.722602183 | -6302106.722602182 |
| `H_256` | 528680171.1252437 | 528680171.1252437 |
| `floor(n(H_255))` | 255 | 255 |

### 1.2 明确分歧：层边界不能用线性近似

参考实现的 `scope_geo_num3d`（@L910）用 `(h*hc, (h+1)*hc)` 近似层区间，
其中 `hc = height_cell(L)`。但 `height_cell(L)` 只是**第 0 层**的厚度——
层厚按公比 `(1+theta0)^(h·cell_deg(L))` 等比增长。于是线性边界在大高度上
不再包含原高度：

| 口径 | `bounds(height_index(H,L), L)` 真包含 `H` 的失败数 |
|---|---|
| 精确式(B.4) | **0 / 297** |
| 线性近似 | **48 / 297**（L=10..32；H=300 m 起就出现，1e6 m 一项占 23 个） |

（297 = 33 个层级 × 9 个高度 `{0, 1, 10, 120, 300, 1000, 3000, 10000, 1e6}` m。）

偏差随层号迅速放大，不是可以忽略的小量：

| 层号 @L=9 | 精确下底面 | 线性近似 | 线性偏小 |
|---|---|---|---|
| h=15 | 1890064.77 m | 1669792.36 m | 11.7% |
| h=100 | 29608560.66 m | 11131949.08 m | 62.4% |
| h=255 | 519501834.16 m | 28386470.15 m | 94.5% |

**结论**：本项目采用精确式(B.4)。测试 `test_height_index_is_strict_inverse_of_layer_bounds`
在 33 个层级 × 9 个高度上逐组断言"返回的层必须真包含原高度"。

### 1.3 两处必须说清的细节

- **`floor` 而非 `round`**：层号是左闭右开区间的下界。实测 `h=0 @L=9` 取带内
  0.95 位置时 `raw = 0.950409`，`round` 会给 1（错层），`floor` 给 0（正确）。
- **浮点噪声**：`n(H_255)` 的原始值是 `255.00000000000037`，第 14 位起为噪声，
  所以特征值对拍记录的是 `floor(n(H_255)) = 255`，不宣称"精确等于 255"。
  这一行是本报告对上一轮草稿的**主动更正**。

---

## 2. 关键设计决策：高度带物化为层集合，而不是区间参数

### 2.1 决策

高度带 `[h_min, h_max]` 在层级 `L` 上展开为**覆盖到的全部层**，每层编一个 64 位码；
于是"高度带重叠"退化为**集合交**，由 PSI 直接承担。

```
航线 0-20000 m (11 层, Z=0..10)  ∩  管控 5000-9000 m (3 层, Z=2..4)  ->  公共层 3，相交
同一 XY、层号不同                                                     ->  不相交
```

### 2.2 被推翻的旧判断

上一版 README（§7.1.1）把"PSI 只做集合交，带重叠需要额外的 MPC 比较"列为待解问题。
实测表明该判断是错的：把高度带展开成层集合后，这是一个可被 PSI 直接承担的
**集合问题**，不需要引入 `height_band: [z_lo, z_hi]` 区间参数，也不需要 MPC 区间比较。
该结论使 §6 记录的 PSI 族泄漏面**无需扩大**。

### 2.3 反面对照（为什么必须展开成层集合）

`test_endpoint_only_representation_is_a_false_negative`：若只把带的两端各编一个码，
中间层被漏掉，`0-20000 m` 与 `5000-9000 m` 会得到 `False` 这一**假阴性**。

真实终端输出：

```
航线 0-20000m vs 管控 5000-9000m         A=11层 B= 3层 相交=True  公共层数=3
整带展开 (11 层)  与 中间带 相交 = True   <- 正确
只取两端 (2 层)   与 中间带 相交 = False   <- 假阴性
```

### 2.4 业务侧写法（低门槛目标）

业务开发者不需要知道层号，也不需要枚举层：

```python
from geo_privacy import geo

route = geo.height_band(x=21861, y=27702, height_min=0,    height_max=20000, level=15)
zone  = geo.height_band(x=21861, y=27702, height_min=5000, height_max=9000,  level=15)
geo.intersects(route, zone)                  # True
route.intersection(zone).cardinality()       # 3
```

该示例已在终端验证：`route=11 层`、`zone=3 层`、`intersects=True`、`公共层数=3`。

### 2.5 新增能力：物理高度可从单个码反解

旧版 Z 只是位域，从码里反解不出物理高度。现在：

```
0x2AB29B0D87C90E08 -> Z=15  L=9 -> [1890064.8, 2034372.1) m
```

---

## 3. 本轮实测发现的新约束：4 位 L 位域封住了层分辨率

### 3.1 现象

```
L 位域 = 4 位  ->  层级上限 = 15
encode_grid_code(..., level=16, ...)  ->  ValueError: grid_code 分量 L=16 超出 4 位上限 15
```

参考实现没有这个约束——它用 96 位 3D 码 + 独立的层级字节，最高到 L32。

### 3.2 后果：低空带坍缩为一层

| 层级 | 可编码 | 第 0 层层厚 | 赤道边长 | 0–1000 m 覆盖层数 | 该带所需 Z 位宽 |
|---|---|---|---|---|---|
| L15 | 是 | 1839.6 m | 1855.3 m | **1** | 1 |
| L16 | **否** | 981.0 m | 989.5 m | 2 | 1 |
| L17 | **否** | 490.5 m | 494.8 m | 3 | 2 |
| L19 | **否** | 122.6 m | 123.7 m | 9 | 4 |
| L22 | **否** | 15.3 m | 15.5 m | 66 | 7（Z7 恰好够） |
| L23 | **否** | 7.7 m | 7.8 m | 131 | 8（Z7 溢出） |

L≤15 上 `height_cell(L)` 从 111319 m 降到 1839.6 m，**始终大于 1000 m 的低空带宽度**。
实测（`test_low_altitude_band_is_single_layer_at_encodable_levels`）：

```
L=9..15 上，0-1000 m / 0-100 m / 800-1000 m  全部 -> 层号恒为 0，均为 1 层
航线 0-100 m  vs  管控 800-1000 m  ->  1 层 vs 1 层  ->  相交（同层，无法分辨）
```

**处置：本版不擅自改键布局。** 布局是落库即难回改的决策，必须由接入层规范统一。
本编译器改为把约束**显示出来**：

- `encode_grid_code` 越界立即失败，绝不静默截断；
- validator 新增**第 6 类失败模式** `HEIGHT_LAYER_UNSUPPORTED`，在编译期报出
  "该高度带在该层级上需要 N 位层号"并给出可用层级。

### 3.3 位域重划的定量方案（仅供参考，本版未采用）

| 方案 | 布局 | 层级上限 | 0–1000 m 可用最高层级 | 第 0 层层厚 |
|---|---|---|---|---|
| 现状 | `X17\|Y17\|Z7\|L4\|Toff14\|Lt4\|ver1` | L15 | L15 | 1839.6 m |
| A | `X17\|Y17\|Z6\|L5\|Toff14\|Lt4\|ver1` | L31 | L21（需 Z6 位，够） | 30.7 m |
| B | `X17\|Y17\|Z7\|L5\|Toff14\|Lt4\|ver0` | L31 | L22（需 Z7 位，够） | 15.3 m |
| C | `X17\|Y17\|Z7\|L5\|Toff13\|Lt4\|ver1` | L31 | L22 | 15.3 m |

方案 B 代价最小（`ver` 并入 `L` 高位即可，`ver` 目前恒为 0），但四者都改动了
已固化的主键布局。**这是需要与合作方（蚂蚁密算）及接入层规范共同决策的事项。**

### 3.4 业务侧立刻可做的降级

收窄高度带（低空管理方通常只管辖一段，如 0–300 m）——即便层级不变，
带边界与层边界的对齐关系也可用 `height_layer_fits` 复算。
代价是：仍需 L≥17 才有米级分辨力。

---

## 4. 代码改动清单

### 新增

| 文件 | 说明 |
|---|---|
| `GIS_SPU/ir/geosot.py` | GB 附录 A/B 公式的纯函数实现（8642 B） |
| `GIS_SPU/tests/test_geosot.py` | 39 项：国标特征值、层号/层区间互逆、Z 与 L 位域容量、高度带集合语义（17637 B） |
| `GIS_SPU/tests/test_height_planner.py` | 12 项：第 6 类失败模式、三维工作流、规划器高度语义诚实性 |
| `GIS_SPU/examples/vertical_conflict.py` | 三维示例（`Intersects` + `Contains`，带高度参数） |

### 修改

| 文件 | 改动 |
|---|---|
| `GIS_SPU/ir/values.py` | Z 语义说明；新增 `layer_codes_for_height` / `height_band_codes` / `grid_code_height_interval` |
| `GIS_SPU/ir/__init__.py` | 导出新符号 |
| `GIS_SPU/geo_privacy/core.py` | `CellSet.from_height_band` classmethod；三维文档串（本轮修正：原示例用了不可编码的 `level=19` 且参数名有误） |
| `GIS_SPU/geo_privacy/geo.py` | `_GeoFacade.height_band(...)` |
| `GIS_SPU/frontend/analyzer.py` | 新增诊断码 `HEIGHT_LAYER_UNSUPPORTED` |
| `GIS_SPU/validator/checks.py` | 第 6 类 `FailureClass` + `validate_height_layer_capacity()`，接入 `validate_all` |
| `GIS_SPU/validator/__init__.py` | 导出 |
| `GIS_SPU/planner/registry.py` | Z 语义说明；`Intersects`/`Contains`/`CellSetIntersect` 的高度注记 |
| `GIS_SPU/tests/test_end_to_end.py` | 改为 `test_all_six_failure_classes_are_registered` |
| `GIS_SPU/README.md` | §3.5 分量语义表；§7.1.1 重写；§8.0 待办；§6.4 计数；附录改为六类失败模式 |

**未改动**：`geosot_work-master/`（只读参考）、SPU 源码（项目自带抽象层）。

---

## 5. 验证证据

### 5.1 测试

```
tests/test_ir.py              37 项
tests/test_planner.py         31 项
tests/test_jax_backend.py     37 项
tests/test_spu_backend.py     31 项
tests/test_psi_backend.py     47 项
tests/test_frontend.py        41 项
tests/test_end_to_end.py      55 项
tests/test_geosot.py          39 项
tests/test_height_planner.py  12 项
                            ─────
                            330 通过 / 0 跳过
```

终端实测：`pytest tests/ -q` → `exit code 0`、`330 passed`（**0 skipped**）。
上一版为 279 项，本轮 +51。

### 5.2 反证测试（7 个变异，全部致目标测试失败）

| 变异 | 变异生效 | 目标测试失败 |
|---|---|---|
| M1 层边界改成线性近似 `h*height_cell` | YES | True |
| M2 `height_index` 改 `round`（去掉 `floor`） | YES | True |
| M3 `height_layers` 丢掉端点层 | YES | True |
| M4 Z 位宽改成 32 位 | YES | True |
| M5 高度带物化退化为只取两端层 | YES | True |
| M6 第 6 类检查不查容量（直接返回空） | n/a（返回空即 no-op 的做法本身被检出） | True |
| M7 高度带冲突改判为 MPC | YES | True |

变异还原后逐项 sha256 断言**字节一致**。

### 5.3 CLI 端到端（真实执行）

```
geo-secure build examples/vertical_conflict.py
[1/8] Parsing                + OK
[2/8] IR generation          + OK
[3/8] Privacy planning       + OK
[5/8] SPU capability check   + OK   available
[7/8] PSI capability check   + OK   spu 0.9.5
[8/8] PSI simulation         + OK   2 个算子经真实 PSI 求交验证

Operation   Representation  Backend  Status
Intersects  CompactCellSet  PSI      verified
Contains    CompactCellSet  PSI/MPC  verified
```

`[8/8]` 走的是**真实** `spu.psi.psi_execute`（`PROTOCOL_ECDH / CURVE_SM2`），非 mock。

### 5.4 第三方独立地复算

GB 附录 A/B 特征值在**两处独立实现**（本项目 `ir/geosot.py` 与参考实现
`geosot_work-master/src/geosot_core.py`）上算出同一组数字。

---

## 6. 诚实的边界：本轮未解决什么

1. **L 位域容量（最高优先级）**。4 位封在 L15，低空 0–1000 m 恒为 1 层。
   本编译器**不擅自改键布局**；需要接入层规范决策（见 §3.3）。
2. **`Contains` 的密态子集比较**。PSI 求交后仍用明文比较判定 `inner ⊆ (A∩B)`，
   泄漏面已在 §7.6 登记。
3. **PSI 结果链式传递**。`CellSetIntersect` 的交集本体尚未喂给下游算子。
4. **编译期按方案实算元素数 K**。`resolve_cost` 已支持按 K 实例化，但 K 目前需显式传入。
5. **三维示例的样例输入仍是二维码**。`examples/vertical_conflict.py` 已跑通编译与
   PSI 真实执行，但 `DEFAULT_EXAMPLE_INPUTS` 未换成真实 3D 码集（低优先级）。
6. **关系稀疏表示与 ZKP**（README §8.5）未动。
7. **v5 样例的 Z 值自洽性**：`格网数据样例_明文与密态映射_v5.json` 里
   `Z=15 @L=9` 按 GB 口径对应 1890064.8 m，与其 `alt=120 m` 输入矛盾
   （120 m @L=9 正确的层号是 0）。Z 在过去没有定义语义，因此该字段此前无法校验；
   现在可以校验了，但**样例文件的订正属数据侧工作，不在本次代码改动范围内**。

---

## 7. 对课题主线的意义

- **与合作方对接**：本项目 PSI 走真实的 `spu.psi.psi_execute`；参考实现的 PSI 用的是
  一个 Windows 二进制 `psi/frontend.exe`，与本课题要求的蚂蚁密算技术栈无关。
  即本项目的密态路径比参考实现更贴合合作方要求。
- **低空导航主场景**：三维语义已具备且可检验，但**低空 0–1000 m 的分辨力受键布局
  限制**。在键布局重划之前，低空三维冲突判定在密态下与二维同效——这是当前最需要
  与合作方对齐的一项技术约束。
- **低门槛目标达成**：业务侧仅用 `geo.height_band(...)` + `geo.intersects(...)`
  即可描述三维任务，不需接触 JAX、SPU 或 MPC API，也不需知道层号。

---

## 8. 复现命令

```bash
# WSL2 / Linux，Python 3.11.16
cd GIS_SPU
python -m pytest tests/ -q                 # 330 passed
python -m geosecure.cli build examples/vertical_conflict.py
python -m geosecure.cli build examples/distance_check.py
```

---

## 附：本报告的自我更正记录

上一轮草稿（未交付）中有两处表述经复算后**证伪并已更正**，在此留痕：

1. 草稿称 `n(H_255)` 为 `254.99999999999997`。复算得原始值 `255.00000000000037`，
   `floor` 后为 255。已在 §1.3 按 `floor(n(H_255)) = 255` 记录。
2. 草稿称"线性近似集中在 H=1e6 m，L=10..21"。复算得失败点分布在
   L=10..32，H=300 m 起就出现，1e6 m 一项占 23 个（共 48 个）。
   已在 §1.2 按实测分布更正。

这两处都是**先写作、后台复算发现不符**才纠正的，记录在此以便审阅。
