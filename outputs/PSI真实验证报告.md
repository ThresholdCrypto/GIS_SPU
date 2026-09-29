# 真实 PSI 接入与验证报告（蚂蚁隐语 SPU 0.9.5）

- 日期：2026-09-24
- 环境：WSL2 Ubuntu / Linux 6.18 / Python 3.11.16 / `spu==0.9.5` / `jax==0.4.34` / `numpy==1.26.4`
- 课题：「三维格网数据统一接入隐私计算体系」（GeoSOT-3D / DQG-4D）
- 项目：`GIS_SPU/` —— 面向地理信息行业的低门槛隐私计算编译器（geo-secure）
- 结论：**`Intersects` / `Contains` / `CellSetIntersect` 三个 PSI 族算子已从
  `backend-direct` 推进到 `verified`——真的调用官方 `psi_execute`，真的两方求交。**

---

## 一、这一步要解决的问题

上一阶段结束时，流水线只有 **一条** 密态执行路径：

```
DistanceLE / WeightedSum / TemporalOverlap  →  JAX  →  jax.jit  →  SPU(MPC) 模拟
```

而 planner 里早已登记的 `Intersects` / `Contains` / `CellSetIntersect` 三个算子
（backend = `PSI`）**从未真正执行过**：它们进不了 JAX（交集性是组合问题，
不是逐元素张量算子），状态词只能停在 `backend-direct`。
也就是说，"PSI"当时只是方案上的一个字，没有任何执行证据。

本次把这条路径补上，并在流水线里为它开了两个独立阶段。

## 二、两条路径的分工（架构结论）

```
                       ┌─ DistanceLE / WeightedSum / TemporalOverlap
Python → Geo-IR → Plan ┤      → JAX 代码生成 → jax.jit → SPU(MPC) 模拟
                       └─ Intersects / Contains / CellSetIntersect
                              → PSI（CSV 两方求交）→ 语义判定
```

流水线由 6 阶段扩为 **8 阶段**：

```
Parsing → IR generation → Privacy planning → JAX generation
  → SPU capability check → SPU simulation
  → PSI capability check → PSI simulation        ← 新增
```

关键纪律：**PSI 与 SPU 的"已验证"互不顶替**。
`_status_word` 分别判定两条路径，没有真实执行记录时一律不写 `verified`。

## 三、官方 PSI API 实测形态

```python
import spu.libspu as libspu
from spu import psi

desc = libspu.link.Desc()
for r in range(2):
    desc.add_party(f"id_{r}", f"thread_{r}")
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
report.original_count          # 本方原始条数
report.intersection_count      # 交集条数（仅 receiver 侧有效）
report.intersection_unique_count
```

| 项目 | 实测结论 |
|------|----------|
| 入口 | `spu.psi.psi_execute(config, lctx) -> PsiExecuteReport` |
| 输入输出 | **仅 CSV 文件**（`SOURCE_TYPE_FILE_CSV`），无内存张量接口 |
| 可用协议 | `ECDH` `KKRT` `RR22` `ECDH_3PC` `ECDH_NPC` `KKRT_NPC` `DP` |
| 可用曲线 | `25519` `FOURQ` `SM2` `SECP256K1` `25519_ELLIGATOR2` |
| 本后端默认 | `PROTOCOL_ECDH` + `CURVE_SM2`（国密，涉密测绘合规首选） |
| 链接方式 | `libspu.link.create_mem` 可建**进程内**多方链路（模拟器） |
| 系统依赖 | 与 SPU 共用 `libpsi.so`，只需 `libgomp1` |

## 四、实测踩到的 6 个真问题（全部已修，全部有回归测试）

### 问题 1：官方枚举成员不是 `int`，能力探测**静默返回空清单**

`PsiProtocol` / `EllipticCurveType` 是 pybind11 类型：

```python
isinstance(psi.PsiProtocol.PROTOCOL_ECDH, int)   # → False
psi.PsiProtocol.PROTOCOL_ECDH.value              # → 1
```

早期 `_enum_members` 只判 `isinstance(value, int)`，于是协议/曲线清单全是 `()`。
**不报错**，但"默认协议在当前版本是否可用"这类判据全部失去依据——
这是最危险的一类 bug：看起来一切正常，实际所有校验都在空转。

修法：同时接受 `int` 与带 `.value` 的枚举成员；并把"清单非空"写成测试契约。

### 问题 2：`PROTOCOL_ECDH` 必须显式指定曲线

`EcdhParams.curve` 默认是 `CURVE_INVALID_TYPE`：

```
RuntimeError: Curve type is not specified.
```

`KKRT` / `RR22` 走 OT 扩展，不读曲线，给了也不报错。
修法：`protocol_needs_curve()` 区分两类，只对 ECDH 族注入曲线。

### 问题 3：空集合会让 PSI 在读表阶段崩溃

PSI 的 CSV 通道走 `arrow_csv_batch_provider`，要求文件至少"表头 + 1 行"。
空集合只能写出 `grid_code\r\n`，于是：

```
Enforce fail at external/psi~/psi/utils/arrow_helper.cc:87
  std::getline(file, line). read csv file second line failed
```

这是**输入枚举口径问题，不是隐私计算失败**。
修法：进入协议前前置判掉，按集合论直接给值，标 `status="empty-input"` 并说明原因。
（`∅ ⊆ 任意集合` 的口径严格沿用 `backends.plain`，不另设特例。）

### 问题 4：`Contains` 的判定方向与集合都取反了

业务语义（与明文一致）：`Contains(outer, inner) ≜ inner ⊆ outer`。
PSI 只回答"交集是什么"，即 `outer ∩ inner`，故正确判据是 `inner ⊆ (outer ∩ inner)`。

早期实现写成 `outer <= 交集`——它实际在问 `inner ⊇ outer`，
**把 `True` 判成 `False`**。实测：

| 输入 | 明文 | 修复前 | 修复后 |
|------|------|--------|--------|
| outer={A,B,C}, inner={A,B} | `True` | `False` ❌ | `True` ✅ |
| outer={A,B}, inner={A,B,C} | `False` | `True` ❌ | `False` ✅ |

### 问题 5：64 位格网码可能 ≥ 2⁶³

`X=131071` 时码值为 `18446633081147752456`（> 2⁶³）。
实测 CSV 通道未被当成有符号整数截断，`2⁶⁴-1` 也能正确往返。
仍保留回归测试盯住这个边界。

### 问题 6：原生库会在**当前工作目录**落一个 `spu.log`

`libspu.logging.LogOptions.system_log_path` 默认值是**相对路径** `spu.log`。
原生库据此在进程 cwd 建文件——cwd 是用户项目目录时，就会凭空多出一个 `spu.log`。
只关掉 `enable_console_logger` **并不能**阻止这件事（试过一次，文件照样出现，只是 0 字节）。

修法：`system_log_path` 一律指向 `os.devnull`；需要看协商日志时用 `quiet=False`
走 console。回归测试在 `tmp_path` 下跑，断言工作目录**一个多余文件都没有**。

## 五、验证结果（真机）

### 5.1 首要验证目标

`examples/route_conflict.py` 的 `route_A | Intersects | NoFlyZone_B`：

```
[7/8] PSI capability check   + OK   1 个算子需 PSI（Intersects）；当前环境可执行（spu 0.9.5）
[8/8] PSI simulation         + OK   1 个算子经真实 PSI 求交验证

PSI simulation
  Intersects: ok  [PROTOCOL_ECDH / CURVE_SM2]
      |A|=3  |A∩B|=2
      result    : True
      reference : True   agree=True
      leaks     : 接收方获得交集本体（不只是布尔值）；交集基数由 recipient 可见

Result
Operation   Representation  Backend  Status
Intersects  CompactCellSet  PSI      verified
```

使用真实 64 位格网码（`encode_grid_code` 生成）：

| 侧 | 格网码 |
|----|--------|
| route_A | `3076691977862381576`, `3076832715350736904`, `3076973452839092232` |
| NoFlyZone_B | `3076832715350736904`, `3076973452839092232`, `3236851239610744840` |
| **交集** | `3076832715350736904`, `3076973452839092232` ← 恰好 2 个，与手工真值一致 |

### 5.2 跨协议 / 跨曲线一致性

同一输入在四种配置下结果**完全一致**（交集基数均为 2）：

| 协议 | 曲线 | 结果 |
|------|------|------|
| `PROTOCOL_ECDH` | `CURVE_SM2` | ok，交集 2 |
| `PROTOCOL_ECDH` | `CURVE_25519` | ok，交集 2 |
| `PROTOCOL_KKRT` | — | ok，交集 2 |
| `PROTOCOL_RR22` | — | ok，交集 2 |

### 5.3 与明文对拍

三个算子全部与 `backends.plain` 一致（`agree=True`），
包括 `Contains` 的两个方向、集合相等、完全不相交、空集等边界。

## 六、泄漏面（如实登记，不淡化）

PSI 的标准语义是"接收方得到**交集本体**"，比布尔结果泄露更多：

| 算子 | 业务想要 | 实际暴露给接收方 |
|------|----------|------------------|
| `Intersects` | 一个布尔 | 交集**本体** + 交集基数 |
| `Contains` | 一个布尔 | 交集本体 + 一次**明文**比较 |
| `CellSetIntersect` | 交集本体 | 交集本体（无额外泄露） |

**必须向业务方点明**：`Contains` 在交集之上还有一次子集比较，
当前实现用的是**明文比较**，不是密态比较。若这一步也须密态，
应改走 MPC 路径（注册表中记为 `PSI/MPC`）。

每个算子的泄漏面登记在 `PSI_OP_LEAKS`，随 `PsiRunResult.reveals` 带出，
出现在 CLI 的 `PSI simulation` 段与结果 JSON 中；测试断言各算子描述**互不相同**，
防止复制粘贴式敷衍。

## 七、对抗性验证（护栏是否真的有效）

不写"看起来对"的测试。做法：逐个把修复**回滚**，确认对应测试必然失败。

| 变异 | 回滚内容 | 结果 |
|------|----------|------|
| M1 | `Contains` 判定改回 `outer <= 交集` | 1 failed ✅ 护栏有效 |
| M2 | 枚举探测改回只判 `isinstance(int)` | 1 failed ✅ 护栏有效 |
| M3 | 空输入前置检查关掉 | 2 failed（耗时 60s，真去跑协议了）✅ |
| M4 | `_status_word` 的 PSI 分支关掉 | 1 failed ✅ 护栏有效 |
| M5 | `system_log_path` 改回默认（落 `spu.log`） | 2 failed ✅ 护栏有效 |
| M6 | `quiet` 默认值改为 `False` | 1 failed ✅ 护栏有效 |

六次变异后还原，全量复跑 **219 passed**，环境干净。

> 记一笔方法论：M6 第一次变异**没有真正改变行为**（只是插了段死代码），
> 测试自然照过——"变异后仍通过"当时无法区分"护栏无效"与"变异无效"。
> 换成真正改动默认值后护栏立刻生效。**变异测试自身也要被验证。**

## 八、测试与状态词

```
tests/test_ir.py           37 项
tests/test_planner.py      26 项
tests/test_jax_backend.py  37 项
tests/test_spu_backend.py  31 项
tests/test_psi_backend.py  47 项   PSI 能力/协议归一化/真实求交/空输入/泄漏面/诚实留空/日志卫生
tests/test_end_to_end.py   41 项   全流程、状态表、CLI、五类失败报告、确定性
                          ─────
全量                        219 通过 / 0 跳过
```

真实执行隐私协议的用例：

| 类别 | 数量 | 说明 |
|------|------|------|
| SPU(MPC) 实跑 | 3 项 | `Simulator.simple`，误差 0.0 |
| PSI 真机求交 | 16 项 | 真的调用 `psi_execute`，非 mock |
| PSI 空输入路径 | 6 项 | 前置判定，不启动协议 |

状态词口径：

| 状态词 | 含义 |
|--------|------|
| `verified` | 该算子声明的主后端**真实执行过**且与明文一致 |
| `backend-direct` | 有直连后端，但本次未真实执行 |
| `planned` | 仅规划，未执行 |
| `error` | 执行失败或与明文不一致 |
| `tolerance-exceeded` | SPU 结果超出容差 |
| `empty-input` | 空集合：按集合论直接给值，未启动协议 |

## 九、交付物

| 文件 | 说明 |
|------|------|
| `GIS_SPU/backends/psi_backend/` | PSI 后端（能力探测 + 运行时），**未改 SPU 源码** |
| `GIS_SPU/docs/PSI_CAPABILITY.md` | PSI 核对结论、5 个坑、泄漏面、状态词 |
| `GIS_SPU/docs/psi_capability_report_wsl.json` | 机器可读能力报告（真实执行生成） |
| `GIS_SPU/tests/test_psi_backend.py` | 47 项 PSI 测试 |
| `GIS_SPU/README.md` | §5.5 PSI API、§6 已验证算子、§7.3/§7.6 路径与泄漏面 |
| `outputs/PSI真实验证_终端逐字记录.txt` | 终端逐字记录（环境 / 能力 / 三个示例 / 测试） |
| `outputs/PSI能力核查快照_wsl.json` | 能力报告快照 |

## 十、下一步扩展点

1. **`Contains` 的密态化比较**：把 `inner ⊆ (outer∩inner)` 这一步从明文比较换成
   MPC 基数比较，消除当前的明文泄露（注册表已预留 `PSI/MPC`）。
2. **PSI 规模化**：当前样例是 3×3 条；需要测 10⁴–10⁶ 级别的格网码集合，
   并记录 `R`（轮次）与通信量，标定注册表里的代价四量模型。
3. **三方 / 多方 PSI**：`PROTOCOL_ECDH_3PC` / `ECDH_NPC` / `KKRT_NPC` 已在能力清单中，
   但本项目当前只做两方；多方场景（航线 × 空域 × 时窗）需要另行验证语义。
4. **`receiver_rank` 的业务化**：当前默认 rank 0 拿交集。
   实际业务里"谁有权知道冲突结果"应成为规划器的一个显式输入。
5. **PSI 结果接入链式算子**：`CellSetIntersect` 的交集本体目前只输出，
   尚未作为输入喂给 `WeightedSum` / `TemporalOverlap` 等后续算子。
6. **`GEOSOT-3D` 三维语义补齐**：Z 维目前只是格网码的一个位段，
   尚未参与算子语义；低空导航需要高度层的显式参与。
