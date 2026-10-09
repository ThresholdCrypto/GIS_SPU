# PSI 能力核查与实现说明（spu 0.9.5 实测）

> 本文件只记录**核过源码、跑过真机**的结论。
> 未经实测的 API 一律不写；能力探测在运行期完成，不预设版本。

## 1. 结论速览

| 项目 | 结论 |
|------|------|
| 入口 | `spu.psi.psi_execute(config, lctx) -> PsiExecuteReport` |
| 运行环境 | WSL2 Ubuntu / Python 3.11.16 / `spu==0.9.5` |
| 可运行 | 是（`runnable=True`，无阻断项） |
| 输入输出 | 仅 **CSV 文件**（`SOURCE_TYPE_FILE_CSV`），无内存张量接口 |
| 已接入算子 | `Intersects` / `Contains` / `CellSetIntersect` |
| 默认协议 | `PROTOCOL_ECDH` |
| 默认曲线 | `CURVE_SM2`（国密，涉密测绘场景合规首选） |
| 实测结果 | 3 个算子全部与明文一致；跨 4 种协议结果一致 |
| 子集判定 | `Contains` 的第二次比较走 **MPC 基数等值**（`|outer∩inner| == |inner|`），
非明文；见 3.7 |

## 2. 官方 API 形态（已核对 0.9.5 实测）

```python
import spu.libspu as libspu
from spu import psi

desc = libspu.link.Desc()
for r in range(2):
    desc.add_party(f"id_{r}", f"thread_{r}")
lctx = libspu.link.create_mem(desc, r)          # 每个参与方一个线程

cfg = psi.PsiExecuteConfig(
    protocol_conf=psi.PsiProtocolConfig(
        protocol=psi.PsiProtocol.PROTOCOL_ECDH,
        receiver_rank=0,
        broadcast_result=False,
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
report.original_count, report.intersection_count, report.intersection_unique_count
```

### 枚举清单（已剔除哨位值）

| 枚举 | 可用成员 |
|------|----------|
| `PsiProtocol` | `PROTOCOL_ECDH` `PROTOCOL_KKRT` `PROTOCOL_RR22` `PROTOCOL_ECDH_3PC` `PROTOCOL_ECDH_NPC` `PROTOCOL_KKRT_NPC` `PROTOCOL_DP` |
| `EllipticCurveType` | `CURVE_25519` `CURVE_FOURQ` `CURVE_SM2` `CURVE_SECP256K1` `CURVE_25519_ELLIGATOR2` |
| `SourceType` | `SOURCE_TYPE_FILE_CSV` |
| `ResultJoinType` | `JOIN_TYPE_INNER_JOIN` `LEFT` `RIGHT` `FULL` `DIFFERENCE` |

`PROTOCOL_UNSPECIFIED` / `CURVE_INVALID_TYPE` / `SOURCE_TYPE_UNSPECIFIED` /
`JOIN_TYPE_UNSPECIFIED` 是 pybind11 枚举的默认占位值：**存在但不可用**。
能力探测显式剔除它们，避免"看起来有这个选项"的误导。

### `PROTOCOL_ECDH_3PC`：枚举成员，但本链路不可执行

上表的 "可用成员" 指的是**该版本枚举里有什么**，不是**本项目能跑什么**。
两者对 `PROTOCOL_ECDH_3PC` 给出相反答案：

| | |
|---|---|
| 枚举中 | 存在（`PsiProtocol.PROTOCOL_ECDH_3PC = 4`） |
| 要求的参与方 | **3** |
| 本项目链路 | 固定 2 方（进程内 `create_mem` 只建两条 link） |
| 实测结果 | 必然失败 |

失败形态是 libpsi 的 C++ 强制校验，而不是可读原因：

```
[Enforce fail at external/psi~/psi/legacy/memory_psi.cc:44]
lctx_->WorldSize() == 3. psi_type:4, only three parties supported, got 2
```

因此后端登记了 `PSI_PROTOCOL_WORLD_SIZE`，并在进入协议**之前**拦截：

```python
from backends.psi_backend import protocols_runnable_here, protocol_world_size

protocol_world_size("PROTOCOL_ECDH_3PC")   # 3
protocol_world_size("PROTOCOL_ECDH")       # 2
protocols_runnable_here()                  # 6 个两方协议，不含 ECDH_3PC
```

要使用三方协议，需要先把 PSI 后端的链路从两方推广到三方——
那属于新增能力，不在本版范围内。当前拦截只保证：**不会把 C++ 栈当成原因抛出去**。

### `PROTOCOL_DP`：能跑，但结果**带噪**

上一节说的是"枚举里有、这里跑不了"。`PROTOCOL_DP` 是**另一类**：跑得起来，
而且状态是 `ok`，但结果**不保证与明文一致**。

它是 **DP-PSI（差分隐私 PSI）**。上游源码 `psi/legacy/dp_psi/dp_psi.h`：

```cpp
struct DpPsiOptions {
  explicit DpPsiOptions(double bob_p = 0.9, double epsilon = 3.0)
      : p1(bob_p), alice_epsilon(epsilon) {
    double e_epsilon = std::exp(alice_epsilon);
    p2 = e_epsilon / (1 + e_epsilon);   // Alice 子采样
    q = 1 - p2;                          // Alice 上采样：往交集中注入假元素
  }
  double p1;            // Bob 子采样 0.9
  double p2;            // ≈ 0.953
  double q;             // ≈ 0.047
};

size_t RunDpEcdhPsiAlice(..., CurveType curve = CurveType::CURVE_25519);
```

噪声是协议的**设计目标**。本机 spu 0.9.5 实测（2026-09），两个方向都观测到，
且**都是间歇性的**：

原生库自己把参数打了出来（`quiet=False`，实测 2026-09-28）：

```
LEGACY PSI config: {"psi_type":"DP_PSI_2PC", ...,
                    "dppsi_params":{"bob_sub_sampling":0.9,"epsilon":3}}
[dp_psi.h:37]  DpPsiOptions p1:0.9 epsilon:3 p2:0.9525741268224333, q:0.047425873177566746
[dp_psi.cc:74] sample bernoulli_distribution: 0.9
[dp_psi.cc:87] bernoulli_items: 2, bernoulli_idx:2 ratio:0.6666666666666666
[dp_psi.cc:49] sample bernoulli_distribution: 0.9525741268224333
[dp_psi.cc:60] bernoulli_items_idx:1 ratio:1
[dp_psi.cc:49] sample bernoulli_distribution: 0.047425873177566746
[dp_psi.cc:60] bernoulli_items_idx:0 ratio:0      <- 本次上采样抽到 0 个，故未注入
```

最后两行解释了为什么注入是间歇的：`q ≈ 0.047` 的伯努利抽样**这一次抽到 0 个**，
于是没有假元素进交集；抽到非 0 个时交集里就多出 A 的元素。

| 观测 | 实测结果 |
|------|----------|
| 交集本体被注入非成员 | `A=[11,22,33]`、`B=[22,33,44]`（真交集 `{22,33}`）；`CellSetIntersect` 10 次里 1 次返回 `(11, 22, 33)`——`11` 根本不在 `B` 里 |
| `Intersects` 漏报真实冲突 | 真交集 2 个元素 → 30 次里 3~4 次 `False`（约 10%）；真交集 15 个元素 → 20 次里 0 次 |
| 同批对照 | `PROTOCOL_ECDH` 5/5、`PROTOCOL_KKRT` 12/12 均为 `True` |
| 官方计数 | DP 下 `intersection_unique_count` 恒为 0（ECDH 同数据为 2），不能当交集基数读 |

**小交集恰恰是更容易被漏报的一类**，而小交集正是"偶发冲突"——地理围栏场景里
最需要发现的那种。因此本后端把它登记进 `PSI_PROTOCOLS_WITH_NOISE`，并且：

- 每个 DP 的 `PsiRunResult.notes` 都带一条带噪说明；
- DP 下 `agreement=False` **不**升级为 `error`；
- 编译状态记 `executed-noisy`，两个 PSI 阶段记 warning；
- `psi-check` 的 `notes` 与替代协议清单都带出这件事。

这张表与 `PSI_PROTOCOL_WORLD_SIZE` 是**两条独立披露**：
前者回答"能不能跑"，后者回答"跑出来准不准"。

### `PROTOCOL_DP` 的曲线：`implicit`，不是"不读曲线"

早期实现把 `PROTOCOL_DP` 与 `KKRT` / `RR22` 并列进 `PSI_PROTOCOLS_WITHOUT_CURVE`，
CLI 随之告诉使用者"--psi-curve 不生效"。源码核对后这是**不成立的断言**：
`RunDpEcdhPsiAlice(..., CurveType curve = CurveType::CURVE_25519)` 说明 DP 是
ECDH 系协议，曲线是它的形参，只是**带了默认值**。

现在改成三分类（`PSI_CURVE_RELATION`）：

| 关系 | 含义 | 协议 |
|------|------|------|
| `required` | 不显式指定必失败（`Curve type is not specified.`） | `ECDH` `ECDH_3PC` `ECDH_NPC` |
| `ignored` | 不基于椭圆曲线，给了也不读 | `KKRT` `RR22` `KKRT_NPC` |
| `implicit` | 基于椭圆曲线且**自带默认值**（DP 默认 25519） | `DP` |

对 `implicit` 一类，本项目**不覆盖**它的默认值（覆盖是否在协议内部生效没核对过），
因此提示语写"未传入"而**不写**"不生效"——只陈述自己做过的事。

### 编译期放行分档：候选 / 显式放行 / 拒绝

"枚举里有"≠"能跑"，"能跑"也≠"编译器放行"。规划期把协议分成三档，单一来源是
`backends/psi_backend/protocol_registry.PSI_PROTOCOLS_EXPLICIT_ONLY`，判定在
`planner.registry.validate_protocol_for_operation`：

| 分档 | 协议 | 行为 |
|------|------|------|
| 候选 | `ECDH` `KKRT` `RR22` | 可显式选择，也进自动候选与替代建议清单 |
| 显式放行 | `ECDH_NPC` `KKRT_NPC`（NPC 族，精确） | 可显式选择并带"显式放行"披露；**不进**候选/建议清单 |
| 显式放行 | `DP`（带噪） | 同上，但披露写成"结果带噪" |
| 拒绝 | `ECDH_3PC` | 参与方数量不满足本链路（3 方） |
| 拒绝 | 其余已登记协议 | 提示"先登记候选或显式放行并补测试" |

**新协议默认落在"拒绝"档**：登记进枚举清单不会自动获得放行，避免"登记即放行"的
静默滑过。NPC 族已真机执行并与明文对拍（`tests/test_psi_backend.py::`
`TestRealPsiIntersection::test_npc_protocols_are_really_executed`）。

NPC 族的性能基线经 benchmark 的显式开关单独产出（不与三候选主基线混跑，
避免默认扫描变慢）：

```bash
python tests/benchmarks/benchmark_psi.py --protocols ecdh-npc,kkrt-npc --sizes 10,12,14
```

## 3. 落地后实测踩到的坑（全部已修）

### 3.1 `PROTOCOL_ECDH` 必须显式指定曲线

`EcdhParams.curve` 默认值是 `CURVE_INVALID_TYPE`，直接跑会：

```
RuntimeError: Curve type is not specified.
```

`PROTOCOL_KKRT` / `PROTOCOL_RR22` 走 OT 扩展，不读曲线，给了也不报错。
本后端因此在 `protocol_needs_curve()` 中区分协议，只对 `required` 一类注入曲线。
`PROTOCOL_DP` 属 `implicit`（自带默认曲线），见上一节——它曾被错归进"不读曲线"。

### 3.2 官方枚举成员不是 `int`

`PsiProtocol` / `EllipticCurveType` 是 pybind11 类型，
`isinstance(psi.PsiProtocol.PROTOCOL_ECDH, int) is False`，但带 `.value`。

早期实现只判 `isinstance(value, int)`，导致协议/曲线清单**静默为空**——
不报错，但"默认协议是否可用"这类判据全部失效。
现在两种形态都接受，清单非空作为测试契约。

### 3.3 空集合会让 PSI 在读表阶段崩溃

PSI 的 CSV 通道走 `arrow_csv_batch_provider`，要求文件至少"表头 + 1 行"。
空集合只能写出 `grid_code\r\n`，于是：

```
Enforce fail at external/psi~/psi/utils/arrow_helper.cc:87
  std::getline(file, line). read csv file second line failed
```

这是**输入枚举口径问题，不是隐私计算失败**。现在在进入协议前就判掉，
按集合论直接给值，并标注 `status="empty-input"` 与原因。

### 3.4 `Contains` 的判定方向

PSI 只回答"交集是什么"，`Contains` 的子集判定要在交集之上再做一步。
正确的业务语义是 `inner ⊆ outer`（与 `backends.plain.plain_contains` 对齐），
而 PSI 出的是 `outer ∩ inner`，故判据为 `inner ⊆ (outer ∩ inner)`。

早期实现写成 `outer <= 交集`，方向与集合都取反了——它实际在问
"inner ⊇ outer"，会把 `True` 判成 `False`。现已修正，并在测试中把两个
方向都钉死。

### 3.5 高位置位格网码

64 位格网编码可能 `>= 2^63`（如 X=131071 时恰为 `18446633081147752456`）。
实测 CSV 通道未被当成有符号整数截断，`2^64-1` 也能正确往返。
仍保留一条回归测试盯住这个边界。

### 3.6 日志会落在当前工作目录

`libspu.logging.LogOptions.system_log_path` 默认值是**相对路径** `spu.log`。
原生库会据此在进程的 cwd 里建文件——如果 cwd 是用户的项目目录，就会凭空多出一个
`spu.log`。仅关掉 `enable_console_logger` 并不能阻止这件事。

本后端一律把 `system_log_path` 指向 `os.devnull`；
需要查看协商日志时用 `quiet=False`，日志走 console 而不是文件。
回归测试在 `tmp_path` 里跑，断言工作目录**一个多余文件都没有**（静默/详谈两种模式）。

### 3.7 `Contains` 的子集判定：从明文比较改为 MPC（本版）

旧写法是在交集之上做**本地集合运算** `inner <= set(交集)`。它有两个问题：

1. 那是明文比较，不是密态比较；
2. 更硬的：它**装不进两方部署**。这一步需要 `inner` 与交集同处一个进程，
   而 `inner` 属于参与方 1——真装机上要么把 `inner` 整个交给接收方，
   要么把交集交给参与方 1。两者都与"子集判定"的本意相悖。

现在改用基数等值（`backends/psi_backend/subset_mpc.py`）：

```
inner ⊆ outer  ⟺  |outer ∩ inner| == |inner|

参与方 0 私有输入  k = |outer ∩ inner|   ← 它本来就从 PSI 拿到交集，不新增泄露
参与方 1 私有输入  n = |inner|
电路（jax.numpy，仅 compare + convert）：k == n  → 一个比特
```

两侧各只提供**一个整数**，格网码一个都不进电路——`tests/test_subset_mpc.py`
用一个 spy 把这条断言钉住（捕获到的入参必须是 `(2, 2)` 而不是任何 64 位码）。

执行路径与本项目其它 MPC 算子一致，沿用编译器的协议/环宽：

```
spu.utils.simulation.Simulator.simple(wsize, ProtocolKind, FieldType)
→ sim_jax → out = spu_fn(k, n)
```

`Simulator.__call__` 对每个输入做 `io.make_shares(x, VIS_SECRET)` 后再分发给各
参与方，因此**没有任何一方看到明文输入**——这正是本功能需要的语义。
实测在 `SEMI2K` / `ABY3` / `CHEETAH` × `FM32` / `FM64` 上均可执行，
整数路径误差 0.0，与明文逐点一致（0..5 的 36 个格点全过）。

**没有改变的事**：PSI 的标准语义没变，接收方**仍然**拿到交集本体。
本版消掉的是"第二次明文比较"这一条，不是交集本体那一条。

### 3.8 退路：MPC 不可用时会退回明文（并如实说）

MPC 不可用（没装 SPU、环宽/协议不匹配、mock 环境）时不静默改语义，
而是退回明文并三处标记：

| 位置 | 标记 |
|------|------|
| `PsiRunResult.subset.mode` | `plaintext-fallback`（显式选择明文则是 `plaintext`） |
| `PsiRunResult.reveals` | "子集判定**回退为明文比较**：MPC 电路未能执行" |
| 状态词 | `subset-plaintext`，不是 `verified` |

为什么保留退路而不是直接留空？旧行为就是明文，留空会让 `Contains` 在没有 SPU 的
环境上从"能跑"变成"不能跑"。退路保留，但把"我退路了"写进结果里，
调用方读得到，也读得懂。

## 4. 泄漏面（如实登记，不淡化）

PSI 的标准语义是"接收方得到**交集本体**"。即使业务只需要一个布尔值，
交集元素也已经出现在接收方侧：

| 算子 | 业务语义 | 实际暴露给接收方 |
|------|----------|------------------|
| `Intersects` | `\|A∩B\| > 0` | 交集**本体**（不只是布尔）；交集基数可见 |
| `Contains` | `inner ⊆ outer` | 交集本体 + 一个布尔（子集判定走 **MPC 基数等值**） |
| `CellSetIntersect` | 交集本体 | 交集本体（即业务所需结果） |

`Contains` 的第二次比较默认是**密态**的（MPC 基数等值，见 3.7），
但它仍然按 PSI 的标准语义把**交集本体**交给接收方——这条没被消掉。

两种形态各自登记、不合并：静态表（`PSI_OP_LEAKS`）只写核心，
这一次实际用的是哪条路由 `PsiRunResult.subset.mode` 带出
（`mpc` / `plaintext` / `plaintext-fallback`），`reveals` 是两者拼接。
合并成一句话就会有人只读到一半。

每个算子的泄漏面都在 `PSI_OP_LEAKS` 中登记，并由 `PsiRunResult.reveals`
随结果带出；测试断言各算子描述**互不相同**，防止复制粘贴式敷衍。

## 5. 为什么 PSI 不进 JAX/SPU 那条路

`Intersects` 是**集合层面的组合问题**，不是逐元素张量算子：
它的输出取决于"哪些格网同时在两侧出现"，无法表达成对固定形状张量的
`jnp` 运算。因此：

- `planner` 中这三个算子 `has_jax_impl=False`；
- JAX 代码生成阶段显式标记为 skipped（不是失败）；
- 改由独立的 `psi_simulation` 阶段真实执行。

这也是 `_status_word` 要把 PSI 与 SPU 两条路径分开判定的原因：
二者的"已验证"不可互相顶替。

## 6. 状态词口径

| 状态词 | 含义 |
|--------|------|
| `verified` | 该算子声明的主后端**真实执行过**且与明文一致 |
| `executed-noisy` | 真跑了，但所用协议**结果带噪**，不一致是设计行为（当前仅 `DP`） |
| `subset-plaintext` | PSI 段真跑过，但 `Contains` 的子集判定落在明文上 |
| `backend-direct` | 有直连后端，但本次未真实执行 |
| `planned` | 仅规划，未执行 |
| `error` | 执行失败或与明文不一致 |
| `tolerance-exceeded` | SPU 结果超出容差 |

对 PSI：`status="ok"` → `verified`；`empty-input` / `unavailable` →
**不进入** `verified`（未启动协议或环境不具备）。
但当协议在 `PSI_PROTOCOLS_WITH_NOISE` 中时，`status="ok"` 记的是
`executed-noisy` 而**不是** `verified`——"与明文一致"这个条件对带噪协议不成立。

`Contains` 另有一条独立降级：即便协议是精确的，只要子集判定落在明文上
（显式选择或 MPC 退路），状态词记 `subset-plaintext`。原因是 `verified` 会被
读成"这一步也是密态完成的"，而那句话在退路上不成立。
带噪与明文子集两档同时成立时，`executed-noisy` 优先——结论不可信比少一层保护更严重。

## 7. 一键复现

在 WSL2（Python 3.11）中：

```bash
cd GIS_SPU
python -m pytest tests/test_psi_backend.py -v
```

实测：71 项通过，0 跳过（说明真实 PSI 确实跑了，不是被 skip 掩盖）。

只看子集比较（含「只有基数进 MPC」那条断言）：

```bash
python -m pytest tests/test_subset_mpc.py -v
```

实测：30 项通过。

只看协议本身与带噪披露：

```bash
python -m pytest tests/test_psi_backend.py -v -k "ProtocolWorldSize or ProtocolNoise or CurveRelation"

# CLI 侧：切换协议 / 看带噪降级
python -m geosecure.cli build examples/route_conflict.py --psi-protocol KKRT
python -m geosecure.cli build examples/route_conflict.py --psi-protocol DP    # 状态记为 executed-noisy
python -m geosecure.cli build examples/route_conflict.py --psi-protocol ECDH_NPC   # 显式放行档
python -m geosecure.cli psi-check | grep 带噪

# CLI 侧：含有 Contains 的例子，看子集判定的两种路径
python -m geosecure.cli build examples/vertical_conflict.py                    # subset: mode=mpc, ABY3/FM64
python -m geosecure.cli build examples/vertical_conflict.py --psi-subset plaintext   # 状态记为 subset-plaintext
```

实测报告见 `docs/psi_capability_report_wsl.json`。
