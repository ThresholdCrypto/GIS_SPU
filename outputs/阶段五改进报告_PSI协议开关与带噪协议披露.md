# 阶段五改进报告：PSI 协议开关打通 + 带噪协议（DP）披露

- 课题：三维格网数据统一接入隐私计算体系（GeoSOT-3D / DQG-4D，低空导航验证）
- 项目：`GIS_SPU` 面向地理信息行业的低门槛隐私计算编译器 MVP
- 环境：WSL2 Ubuntu / Python 3.11.16 / spu 0.9.5 / jax 0.4.34
- 日期：2026-09-28
- 逐字终端记录：`outputs/阶段五验证_PSI协议开关与带噪协议_终端逐字记录.txt`

---

## 1. 本阶段目标与结论

**目标**：把 PSI 的 CLI 协议开关打通——即"层面 1 唯一剩下的口子"。

"层面 1"指：**不改 SPU 源码、不新增后端**，只把 SPU 已经提供的另一个协议接出来用。
此前 `--protocol` 只作用于 SPU（MPC）阶段，PSI 阶段永远走默认 `ECDH + SM2`，
无法切换。

**结论**：已打通。`--psi-protocol` / `--psi-curve` 一路贯到
`backends.psi_backend.run_psi_intersection`，并在 CLI 上对三类情况分别给出可读结论
（非法名 / 跑不了 / 跑得起来但结果不准）。过程中**顺带发现并修掉了两个相邻缺陷**，
以及一个**语义级别的重大披露缺口**（见第 4、5 节）。

---

## 2. 交付物

| 文件 | 改动 |
|------|------|
| `geosecure/cli.py` | 新增 `--psi-protocol` / `--psi-curve`；构造 `Compiler` **之前**先解析协议/曲线，非法名以可读错误退出码 2 结束而非 traceback；打印 `选用` / `带噪` 行；曲线提示按三分类分别措辞 |
| `geosecure/compiler.py` | `Compiler.__init__` 接收并解析 PSI 协议/曲线；`CompileResult` 新增 `psi_protocol` / `psi_curve` 并序列化；`_stage_psi_capability` / `_stage_psi_simulation` 分别处理"跑不了"与"带噪"两种情况；新增 `curve_suffix()`；`_status_word` 新增 `executed-noisy` |
| `backends/psi_backend/capability.py` | 新增 `PSI_PROTOCOLS_WITH_NOISE` / `protocol_is_exact()` / `PSI_CURVE_RELATION` / `psi_curve_relation()` / `PSI_PROTOCOLS_CURVE_REQUIRED` / `runnable_protocols_hint()`；`PSI_PROTOCOLS_WITHOUT_CURVE` 改为派生，修正 DP 的归类；能力报告新增带噪披露 note；新增 `resolve_psi_protocol()` 的曲线语义 |
| `backends/psi_backend/runtime.py` | 接收 `protocol` / `curve`；带噪协议结果附带说明；带噪协议的 `agreement=False` **不**升级为 `error`；三方协议错误的替代清单带 `*` 标注 |
| `backends/psi_backend/__init__.py` | 导出新增常量与函数 |
| `tests/test_psi_backend.py` | 修正一条**随机失败**用例；新增 `TestProtocolNoise`（9 项）与 `TestCurveRelation`（7 项） |
| `tests/test_end_to_end.py` | `TestCli` 新增 5 项：DP 曲线提示、带噪状态词、两个 PSI 阶段状态、替代清单标注、`--help` 带噪标注 |
| `README.md` | §5.5 新增 DP 披露与"三条纪律"；§6.1.1 新增 `executed-noisy`；§7.6 新增"精度是第二条披露面"；§8.0 / §8.6 更新；§6.4 测试覆盖表按 10 个文件重算（330 → 417） |
| `docs/PSI_CAPABILITY.md` | 新增 `PROTOCOL_DP` 两节（带噪语义 + 曲线三分类）；状态词口径补充；复现命令更新 |

---

## 3. 层面 1 的收口：协议开关

```bash
geo-secure build examples/route_conflict.py --psi-protocol KKRT
geo-secure build examples/route_conflict.py --psi-protocol RR22
geo-secure build examples/route_conflict.py --psi-protocol DP
geo-secure build examples/route_conflict.py --psi-protocol ECDH --psi-curve CURVE_25519
```

三条纪律（均有测试守着）：

1. **非法协议名** → 退出码 `2` + 可用清单，**不是 traceback**；
2. **本链路跑不了的协议**（`ECDH_3PC`，需 3 方）→ PSI 能力阶段报 `error`，
   状态表记 `backend-direct` 而**不是** `verified`，模拟阶段跳过；
3. **结果带噪的协议**（`DP`）→ 能跑，但两个 PSI 阶段都记 `warning`，
   状态表记 `executed-noisy`，且不得出现"经真实 PSI 求交验证"这类字样。

---

## 4. 顺带修掉的两个相邻缺陷

上一阶段遗留的改动让两个缺陷进了主干，本阶段一并收口：

| 缺陷 | 现象 | 现状 |
|------|------|------|
| 非法 SPU 协议名冲掉整份输出 | `--protocol SPDZ2K`（spu 0.9.5 无此协议）让整条流水线带 traceback 崩掉 | 编译器在调用点接住 `run_spu_simulation` 的 `ValueError`，转成可读错误 + 退出码 1 |
| 总结行自相矛盾 | 阶段失败但不产生 diagnostics 时，输出"编译未通过：0 个错误" | 改为"编译未通过：阶段失败：<阶段名>" |

注意这两处都**没有**改 `run_spu_simulation` 的 fail-fast 契约——它有独立的测试
守着（`test_rejects_invalid_protocol_before_anything_else`）。修复落在调用方。

---

## 5. 关键发现：`PROTOCOL_DP` 是**带噪**协议（本阶段最重要的结论）

### 5.1 起因

一条既有用例把 `protocols_runnable_here()` 里的**每个**协议都跑一遍，并断言
"结果必须与明文一致"。它出现约 **2/12 的随机失败**，失败协议恒为 `PROTOCOL_DP`。

第一反应是"这是并发/不确定性 bug"。往下查才发现：**不是 bug，是语义差异**——
而那条用例把它掩盖了。

### 5.2 证据链（三重）

**（a）上游源码** `secretflow/psi` → `psi/legacy/dp_psi/dp_psi.h`：

```cpp
struct DpPsiOptions {
  explicit DpPsiOptions(double bob_p = 0.9, double epsilon = 3.0)
      : p1(bob_p), alice_epsilon(epsilon) {
    double e_epsilon = std::exp(alice_epsilon);
    p2 = e_epsilon / (1 + e_epsilon);   // Alice 子采样
    q = 1 - p2;                          // Alice 上采样：往交集中注入假元素
  }
  double p1;  // Bob 子采样 0.9
  double p2;  // ≈ 0.953
  double q;   // ≈ 0.047
};
size_t RunDpEcdhPsiAlice(..., CurveType curve = CurveType::CURVE_25519);
```

**（b）原生库运行时自证**（`quiet=False`，2026-09-28 实测）——无需推断：

```
LEGACY PSI config: {"psi_type":"DP_PSI_2PC", ...,
                    "dppsi_params":{"bob_sub_sampling":0.9,"epsilon":3}}
[dp_psi.h:37] DpPsiOptions p1:0.9 epsilon:3 p2:0.9525741268224333, q:0.047425873177566746
[dp_psi.cc:74] sample bernoulli_distribution: 0.9
[dp_psi.cc:87] bernoulli_items: 2, bernoulli_idx:2 ratio:0.6666666666666666
[dp_psi.cc:49] sample bernoulli_distribution: 0.9525741268224333
[dp_psi.cc:60] bernoulli_items_idx:1 ratio:1
[dp_psi.cc:49] sample bernoulli_distribution: 0.047425873177566746
[dp_psi.cc:60] bernoulli_items_idx:0 ratio:0      <- 本次上采样抽到 0 个，故未注入
```

最后两行解释了为什么注入是**间歇**的：`q ≈ 0.047` 的伯努利抽样这一次抽到 0 个。

**（c）行为实测**——两个方向都观测到，且都是间歇性的：

| 观测 | 结果 |
|------|------|
| 交集本体被**注入非成员** | `A=[11,22,33]`、`B=[22,33,44]`（真交集 `{22,33}`）；`CellSetIntersect` 10 次里 1 次返回 `(11, 22, 33)`——`11` 根本不在 `B` 里 |
| `Intersects` **漏报真实冲突** | 真交集 2 个元素 → 30 次里 4 次 `False`（约 13%）；真交集 15 个元素 → 20 次里 0 次 |
| 同批对照 | `PROTOCOL_ECDH` 10/10、`PROTOCOL_KKRT` 10/10 均为 `True` |
| 官方计数 | DP 下 `intersection_unique_count` 恒为 0（ECDH 同数据为 2） |

### 5.3 为什么这条对课题重要

对禁飞区判定这是**安全事故级别**的语义变化：既可能把不冲突判成冲突，也可能把
冲突判成不冲突。而且——

> **小交集恰恰是更容易被漏报的一类**，而小交集正是"偶发冲突"，
> 也就是地理围栏场景里最需要发现的那种。

`DP` 的噪声是协议的**设计目标**（差分隐私），不是缺陷；但把它当成 `ECDH` 的
等价替代去跑冲突判定，结论就是不可信的。

### 5.4 处理方式

拆成**两条独立披露**，不合并——合并后总有人只读到一半：

- `PSI_PROTOCOL_WORLD_SIZE`（+ `protocols_runnable_here()`）回答"**能不能跑**"
  ——既有披露，对应 `ECDH_3PC`；
- `PSI_PROTOCOLS_WITH_NOISE`（+ `protocol_is_exact()`）回答"**跑出来准不准**"
  ——本阶段新增，对应 `DP`。

具体落地：

1. 登记进 `PSI_PROTOCOLS_WITH_NOISE`，`psi-check` 的 `notes` 与 CLI 的 PSI 能力块
   都如实带出；
2. 每个 DP 的 `PsiRunResult.notes` 都带一条带噪说明，不靠调用方转述；
3. DP 下 `agreement=False` **不**升级为 `error`——那是协议语义；升级会让一次正常
   执行被**随机**报成失败，而且错误消息会把"协议本就带噪"说成"PSI 结果与明文不一致"；
4. 状态词新增 `executed-noisy`，两个 PSI 阶段记 `warning`。

### 5.5 修掉自身的测量错误

初测用 40 次抽样得到"1/40（2.5%）"，据此写成文档。换输入后实测是 **3~4/30
（约 10~13%）**——同一个量在不同输入下差别很大，因为漏报率随真交集大小急剧变化。
文档已按更完整的阶梯（真交集 2 → 约 10%；15 → 20 次里 0 次）重写。

**教训**：小样本单批次得出的比率不要直接当结论写进文档。

---

## 6. 纠正一条**不成立**的断言：DP 的曲线关系

早期实现把 `PROTOCOL_DP` 与 `KKRT` / `RR22` 并列进 `PSI_PROTOCOLS_WITHOUT_CURVE`，
CLI 随之告诉使用者"`--psi-curve` 不生效"。

源码核对后这是**不成立的断言**：

```cpp
size_t RunDpEcdhPsiAlice(..., CurveType curve = CurveType::CURVE_25519);
```

DP 是 ECDH 系协议，曲线是它的**形参**，只是**带了默认值**。于是改为三分类：

| 关系 | 含义 | 协议 | 本项目行为 |
|------|------|------|-----------|
| `required` | 不显式指定必失败（`Curve type is not specified.`） | `ECDH` `ECDH_3PC` `ECDH_NPC` | 注入默认曲线 `CURVE_SM2` |
| `ignored` | 不基于椭圆曲线，给了也不读 | `KKRT` `RR22` `KKRT_NPC` | 丢弃并明说"不读椭圆曲线" |
| `implicit` | 基于椭圆曲线且**自带默认值**（DP 默认 25519） | `DP` | **不覆盖**；提示写"未传入"而**不写**"不生效" |

`implicit` 一类的措辞刻意保守：只陈述**本项目做过的事**（没把曲线传进去），
不预言协议内部行为（覆盖是否生效没核对过）。

`protocol_needs_curve()` 与 `psi_curve_relation()` 之间加了一条不变量测试，
防止两个入口各自漂移。

---

## 7. 验证证据

| 项目 | 结果 |
|------|------|
| 全量测试 | **417 passed, 0 failed**（本阶段新增 `TestProtocolNoise` 9 项、`TestCurveRelation` 7 项、`TestCli` DP 披露 5 项，另修正 1 条随机失败用例） |
| 连续 14 轮全量测试 | **0 轮失败**（修复前同一循环 14 轮里有 5 轮假失败） |
| 变异测试 | **8/8 全部被捕获**（见下表） |
| 五个示例回归 | 全部 `exit=0`，状态词正确 |
| 字节卫生 | 9 个改动文件全部 LF、无 BOM |

变异测试明细（每条都是"故意改坏实现，看测试是否报警"）：

| 变异 | 被捕获 |
|------|--------|
| `runtime`：带噪协议的不一致重新升级为 `error` | ✅ 1 项失败 |
| `compiler._status_word`：不再区分带噪协议 | ✅ 2 项失败 |
| `compiler._stage_psi_capability`：去掉带噪分支 | ✅ 1 项失败 |
| `compiler._stage_psi_simulation`：去掉带噪分支 | ✅ 2 项失败 |
| `capability`：DP 关系退回 `ignored` | ✅ 3 项失败 |
| `capability`：不再在能力报告里披露带噪协议 | ✅ 1 项失败 |
| `capability.runnable_protocols_hint`：不再标注带噪协议 | ✅ 2 项失败 |
| `cli`：DP 曲线提示改回"不生效" | ✅ 1 项失败 |

> 修掉一条"看起来确定性、实际概率性"的用例：最初用 `reference_fn = lambda: False`
> 造不一致，但 DP 的噪声**偶尔恰好**等于 `False`，于是 14 轮里 5 轮假失败。
> 改成用一个**不可能**等于布尔/元组的参考值后，才真正确定。

---

## 8. 仍未闭合 / 下一步

| 优先级 | 待办 | 说明 |
|--------|------|------|
| 高 | **键布局重划（L 位域 4→5 位）** | `L4` 把层级封在 15，低空 0–1000 m 恒为 1 层，三维分辨力退化为"XY 相交即相交"。属接入层规范决策，不在本编译器可修范围 |
| 高 | `Contains` 的密态子集比较 | 当前是明文比较，泄漏面已在 §7.6 登记 |
| 中 | PSI 结果链式传递 | `CellSetIntersect` 的交集本体尚未喂给下游算子 |
| 中 | 编译期按方案实算元素数 K | `resolve_cost` 已支持按 K 实例化，K 仍需显式传入 |
| 低 | 三方 PSI 链路 | 要用 `ECDH_3PC` 需把后端链路从两方推广到三方，属新增能力 |
| 低 | `implicit` 曲线的生效性 | DP 传曲线是否在协议内部生效未核对；要核对需读上游 C++ 或做可区分实验 |

---

## 9. 复现命令

```bash
# 在 WSL2（Python 3.11 + spu 0.9.5）中
cd GIS_SPU

# 全量测试
python -m pytest tests/ -q

# 只看本次相关的三组
python -m pytest tests/test_psi_backend.py -v -k "ProtocolWorldSize or ProtocolNoise or CurveRelation"

# 命令行
python -m geosecure.cli build examples/route_conflict.py --psi-protocol KKRT   # verified
python -m geosecure.cli build examples/route_conflict.py --psi-protocol DP     # executed-noisy
python -m geosecure.cli build examples/route_conflict.py --psi-protocol NOPE   # exit 2
python -m geosecure.cli build examples/route_conflict.py --psi-protocol ECDH_3PC  # error + 替代清单
python -m geosecure.cli psi-check | grep 带噪
```

> 想看 `DP_PSI_2PC` 的原生日志，把 `quiet=False` 传给
> `backends.psi_backend.run_psi_intersection`（见第 5.2 节 b 项）。