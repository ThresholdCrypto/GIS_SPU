# SPU 隐私计算协议清单与扩展路径

> 本文所有结论均为**实测**：在 WSL2 + `/opt/miniconda3/envs/spu311` 上
> 对已安装的 `spu 0.9.5` 逐项运行取得，不采用记忆或文档推测。
> 核对时间：2026-09-28。

## 一、SPU 提供两大类协议

SPU 里"协议"是**两个互不相通的东西**，入口、枚举、能力边界都不同：

| | MPC | PSI |
|---|---|---|
| 枚举 | `spu.libspu.ProtocolKind` | `spu.libpsi.PsiProtocol` |
| 入口 | `spu.utils.simulation.Simulator.simple` → `sim_jax` | `spu.psi.psi_execute(config, lctx)` |
| 原生库 | `libspu.so` | `libpsi.so` |
| 输入形态 | 内存张量（JAX 数组） | **仅 CSV 文件** |
| 本项目封装 | `backends/spu_backend/` | `backends/psi_backend/` |
| 本项目 CLI | `--protocol` / `--field` / `--world-size` | 暂无（固定默认值） |

### 1.1 MPC 协议（`ProtocolKind`，实测枚举值）

| 协议 | 枚举值 | 参与方下限 | 实测执行 |
|---|---|---|---|
| `PROT_INVALID` | 0 | —（哨位值） | 不可用 |
| `REF2K` | 1 | 2 | ok |
| `SEMI2K` | 2 | 2 | ok |
| `ABY3` | 3 | 3 | ok（本项目默认） |
| `CHEETAH` | 4 | 2 | ok |
| `SECURENN` | 5 | 3 | ok |
| `SWIFT` | （6） | — | **未发布**：`libspu.pyi` 里该行被注释掉 |

注意两点：

1. **没有 `SPDZ2K`。** 按名字配置会直接 `ValueError`。
2. `SWIFT = 6` 在存根里以注释形式占位，说明上游规划过但 0.9.5 未发布；
   不要按它写代码。

### 1.2 环宽（`FieldType`）

`FM32` / `FM64` / `FM128`（+ `FT_INVALID` 哨位）。三档**全部实测可跑**。

`DistanceLE` 的 协议 × 环宽 全矩阵（5 × 3 = 15 组）实测全部 `ok`：

| | FM32 | FM64 | FM128 |
|---|---|---|---|
| REF2K | ok | ok | ok |
| SEMI2K | ok | ok | ok |
| ABY3 | ok | ok | ok |
| CHEETAH | ok | ok | ok |
| SECURENN | ok | ok | ok |

`WeightedSum` 在 5 个协议 × `FM64`/`FM128` 上也全部 `ok` 且结果与明文一致（15）。

### 1.3 PSI 协议（`PsiProtocol`，实测枚举值）

| 协议 | 枚举值 | 需要曲线 | 参与方 | 本项目链路实测 |
|---|---|---|---|---|
| `PROTOCOL_UNSPECIFIED` | 0 | — | —（哨位值） | 不可用 |
| `PROTOCOL_ECDH` | 1 | 是 | 2 | ok（默认） |
| `PROTOCOL_KKRT` | 2 | 否 | 2 | ok |
| `PROTOCOL_RR22` | 3 | 否 | 2 | ok |
| `PROTOCOL_ECDH_3PC` | 4 | 是 | **3** | **失败**（见 §三） |
| `PROTOCOL_ECDH_NPC` | 5 | 是 | 2 | ok |
| `PROTOCOL_KKRT_NPC` | 6 | 否 | 2 | ok |
| `PROTOCOL_DP` | 7 | 否 | 2 | ok |

椭圆曲线（`EllipticCurveType`）：
`CURVE_25519` / `CURVE_FOURQ` / `CURVE_SM2` / `CURVE_SECP256K1` /
`CURVE_25519_ELLIGATOR2`（+ `CURVE_INVALID_TYPE` 哨位）。

本项目默认 `PROTOCOL_ECDH` + `CURVE_SM2`：涉密测绘场景优先国密。
`ECDH` 族**必须显式给曲线**，否则 `RuntimeError: Curve type is not specified.`

## 二、如何"增加"隐私计算协议

"增加协议"在三个不同层面上含义完全不同，代价也差三个数量级。

### 层面 1：换用 SPU 已有的另一个协议（成本最低，今天就能做）

**MPC 侧已经通了**，无需改代码：

```bash
geo-secure build examples/distance_check.py --protocol CHEETAH --field FM128
geo-secure build examples/risk_score.py     --protocol SECURENN --world-size 3
```

后台能力核查会跟着走：`check_operation_capability` 用
`OP_PROTOCOL_HINT` 给出算子推荐协议与环宽，`run_spu_simulation` 在
参与方数量不足时**先拦后跑**，不会拿推测值填充结果。

**PSI 侧目前没有 CLI 通道** —— `compiler._stage_psi_simulation` 调用
`run_psi_operation` 时未传 `protocol`/`curve`，因此永远走默认
`ECDH + SM2`。要按协议切换，需要把参数从 CLI 一路贯到
`run_psi_intersection`。这是当前最实际的一个待补口子。

### 层面 2：接入新的隐私后端族（FHE / TEE）

不改 SPU 源码，在 `backends/` 下按目录扩展即可（README §8.1 已规划）：

```
backends/
├── psi_backend/    ← 已有
├── spu_backend/    ← 已有
├── fhe/            ← 新：BFV / CKKS，做"公开参数 × 密态数据"的线性部分
└── tee/            ← 新：TEE
```

四件事缺一不可：

1. `planner.registry.OperatorRule.backend` 里声明新后端；
2. 该后端的 `run_*` 执行入口；
3. `validator` 里对应的能力核查（**没有可运行 → 留空，不填推测值**）；
4. 测试：明文 / JAX / 新后端三方对拍，并明确容差。

分工原则（课题既有结论）：**FHE 做线性部分，MPC 做非线性判定与保密权重。**

### 层面 3：给 SPU 本体增加一个新协议内核（成本最高，且本项目不做）

需要改 SPU 的 C++ 源码并重新编译 `libspu.so`。两个硬约束：

1. **wheel 是纯二进制的**——实测 `spu` 包内不含任何 `.h` / `.hpp` / `.cc`，
   只有 `libspu.so`、`libpsi.so` 与 `.pyi` 存根；
2. 本项目明确约定**不修改 SPU 源码**（`backends/psi_backend/__init__.py`
   的设计边界里写死了这一条）。

因此新增协议本体属于**上游 SPU 仓库**的工作，不在本项目范围内。
本项目只做"在 SPU 已有协议之上选择与封装"。

## 三、本次核对发现并修复的缺陷：`ECDH_3PC` 的假可用

### 缺陷

`PROTOCOL_ECDH_3PC` 在官方枚举中存在，被列进了 `PSI_PROTOCOLS`，
能力报告与 README 都把它算作"可用协议"。但它是**三方**协议，
而本后端进程内链路固定两方：

```python
for rank in range(2):
    desc.add_party(f"id_{rank}", f"thread_{rank}")
```

于是选中它必然失败，且失败形态是 libpsi 的 C++ 强制校验而非可读原因：

```
[Enforce fail at external/psi~/psi/legacy/memory_psi.cc:44]
lctx_->WorldSize() == 3. psi_type:4, only three parties supported, got 2
```

这与项目已经确立的纪律冲突——"看起来有这个选项"不等于"能用"，
`ENUM_SENTINELS` 的处理方式就是先例。**"枚举里有"与"这里能跑"必须分开说。**

### 修复内容

| 文件 | 变更 |
|---|---|
| `backends/psi_backend/capability.py` | 新增 `PSI_PROTOCOL_WORLD_SIZE` 表（7 个协议逐一登记参与方数量）、`PSI_RUNTIME_WORLD_SIZE = 2`、`protocol_world_size()`、`protocols_runnable_here()`；能力探测新增第 7 步，在 `notes` 里披露不可执行协议 |
| `backends/psi_backend/runtime.py` | `run_psi_intersection` 增加**前置**参与方检查：给出错误位置、原因、替代协议清单，不再抛 C++ 栈 |
| `backends/psi_backend/__init__.py` | 导出新符号 |
| `tests/test_psi_backend.py` | 新增 `TestProtocolWorldSize`（8 个用例），含"漏登记即失败"与"清单里每个协议都必须真能跑"两条契约 |
| `README.md` / `docs/PSI_CAPABILITY.md` | 拆成"枚举中的协议"与"本链路可执行协议"两张清单，并记录失败形态 |

### 修复后行为

```
枚举协议（probe）：DP, ECDH, ECDH_3PC, ECDH_NPC, KKRT, KKRT_NPC, RR22   （7）
可执行   （2 方） ：DP, ECDH,        ECDH_NPC, KKRT, KKRT_NPC, RR22   （6）

PROTOCOL_ECDH_3PC -> error
  协议 PROTOCOL_ECDH_3PC 需要 3 个参与方，本后端的进程内链路固定 2 方，
  无法执行该协议；请改用两方可执行协议：PROTOCOL_ECDH, PROTOCOL_KKRT, ...
```

6 个两方协议逐一实跑，全部 `ok` 且 `value=True`。

### 验证

- 全量测试 `388 passed`（基线 380 + 新增 8），0 失败；
- 变异验证：把前置检查改成恒不生效，8 个新用例中
  `test_three_party_protocol_is_refused_with_a_readable_reason` 立刻失败，
  证明该用例真的在守这条约束，不是空断言；
- 5 个示例（`route_conflict` / `distance_check` / `risk_score` /
  `vertical_conflict` / `altitude_band`）全部 `exit=0`，最终状态表无回退；
- 字节卫生：改动文件均 LF、无 BOM、无尾随 CR。

## 四、明确未做的事

1. **没有为 PSI 加 CLI 协议开关。** 这是层面 1 的待补口子，
   需要改 `geosecure/cli.py` / `compiler.py` 的参数贯通，本次未做。
2. **没有把 PSI 链路推广到三方。** 那需要构造 3 条 link、3 份 CSV 输入
   与三方 receiver 语义，属新增能力；本次只保证三方协议被**可读地拒绝**。
3. **没有改 SPU 源码**，也没有新增协议内核。
4. 未对 `PROTOCOL_DP`（差分隐私）与其它协议的**语义差异**做业务侧评估——
   当前只验证"能跑通且与明文一致"，不代表适合替代 ECDH 用于围栏判定。