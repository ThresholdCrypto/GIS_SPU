# 结果策略（ResultPolicy）设计说明：PSI 与 MPC 两族

> 任务文档 §15。统一入口（Phase 10 起，PSI / MPC 共用）：
> `backends/result_policy.py`；旧路径 `backends/psi_backend/result_policy.py`
> 保留为再导出兼容层（单一来源，测试锁定 identity）。运行时接入：
> `backends/psi_backend/runtime.py`（`run_psi_intersection(..., result_policy=...)`）与
> `backends/spu_backend/runtime.py`（`run_spu_simulation(..., op=..., result_policy=...)`，
> 策略在**任何执行之前**解析，fail-fast）。测试：`tests/test_result_policy.py`
> （含真实 PSI 的 COUNT 口径对拍与 MPC 侧策略表 / 运行时纪律）。

## 1. 问题：PSI 的标准语义不是“布尔”

PSI 的标准语义是 **receiver gets intersection body**——接收方获得交集本体。
即使业务只需要 `True / False`，协议内部也已经把交集元素交给了接收方。
所以本设计把两件事**分开登记**：

| 概念 | 含义 | 谁能改变 |
|---|---|---|
| `protocol_leak`（协议内部泄漏） | 协议真实施加给接收方的信息（当前唯一现实值：交集本体） | 不能——换策略不改协议语义 |
| `business_value`（业务暴露） | 编译器最终交还给业务层的数据 | 由 ResultPolicy 控制 |

**硬性纪律**：任何输出都不得把 PSI 说成“只返回布尔值”。
`REVEAL_BOOLEAN` 的含义是“业务层字段是布尔”，不是“协议只泄漏布尔”；
两句披露必须同时出现（`ResultPolicy.disclosure` 与 CLI `policy` 行就是按这个
口径生成的）。这条纪律来自任务文档 §25-9。

## 2. 五种策略与当前状态（按执行族登记）

| 策略 | 业务层暴露 | PSI 协议内部泄漏 | MPC 协议内部输出面 | 本链路状态 |
|---|---|---|---|---|
| `REVEAL_INTERSECTION` | 交集本体 | 交集本体 | —— | 可执行（PSI 独有；`CellSetIntersect` 默认） |
| `REVEAL_BOOLEAN` | 布尔判定 | 交集本体 | 输出面（`output-only`） | 可执行（PSI：`Intersects` / `Contains` 默认；MPC：`DistanceLE` / `TemporalOverlap` 默认） |
| `REVEAL_COUNT` | 交集基数 | 交集本体 | —— | 可执行（PSI 独有；`CellSetIntersect` 可选） |
| `REVEAL_VALUE` | 数值聚合 | —— | 输出面（`output-only`） | 可执行（MPC 独有；`WeightedSum` 默认） |
| `REVEAL_TO_REGULATOR` | 最小化报告 | 交集本体 | 输出面（`output-only`） | **显式拒绝**：部署期模式，两族同判，进程内两方模拟不提供该执行语义（不伪造实现） |

适用性矩阵（`APPLICABLE_POLICIES_BY_OP`）：

| 算子 | 执行族 | 默认策略 | 允许显式选择 |
|---|---|---|---|
| `Intersects` | PSI | `REVEAL_BOOLEAN` | 仅 `REVEAL_BOOLEAN` |
| `Contains` | PSI | `REVEAL_BOOLEAN` | 仅 `REVEAL_BOOLEAN` |
| `CellSetIntersect` | PSI | `REVEAL_INTERSECTION` | `REVEAL_INTERSECTION` / `REVEAL_COUNT` |
| `DistanceLE` | MPC | `REVEAL_BOOLEAN` | 仅 `REVEAL_BOOLEAN` |
| `WeightedSum` | MPC | `REVEAL_VALUE` | 仅 `REVEAL_VALUE` |
| `TemporalOverlap` | MPC | `REVEAL_BOOLEAN` | 仅 `REVEAL_BOOLEAN` |

为什么 `Intersects` 不能选 `REVEAL_COUNT`：它的业务语义就是布尔，返回计数会
**扩大**暴露面而不是收缩；“登记为可用”等于把最小化设计反向使用。当算子不存在
比默认更小的合法策略时，`resolve_result_policy` 直接拒绝并列出可用清单。

## 2.1 MPC 侧的登记口径（Phase 10）

MPC（SPU 模拟）类算子的输出面由 `OP_RESULT_FAMILY` 划入 MPC 族，泄漏码是
`output-only`（与 PSI 的 `intersection-body` 并列且互不相同）。登记句说三件事：

1. 计算结果在**指定输出方**揭示，输入与中间值在半诚实模型下不进输出面；
2. 这是对**本项目配置**的登记，不是对上游 SPU 协议安全性证明的转述；
3. **不得默认把原始计算结果广播给所有参与方**（任务书 §十）——当前进程内
   模拟不建模按方隔离（输出直接返回给调用方 / 编译报告），真实部署必须显式
   指定输出方。

每个 MPC 算子的输出面句子另登记在 `MPC_OP_REVEALS`（`DistanceLE` /
`WeightedSum` / `TemporalOverlap`），随 `SpuRunResult.reveals` 输出，并出现在
CLI 的 SPU simulation 段。运行时纪律：`run_spu_simulation(..., op=...)` 在
**任何执行之前**解析策略，未知算子 / 部署期策略直接 fail-fast——策略是编译
期属性，随结果返回，不随执行成败改变。CLI 输出形如（`examples/distance_check.py`，节选）：

```text
      policy    : REVEAL_BOOLEAN（业务层暴露 boolean；结果策略 REVEAL_BOOLEAN：业务层暴露 boolean；协议内部输出面（本仓库按配置登记）：计算结果在指定输出方之间揭示，……不得默认广播给所有参与方。）
      reveals   : 输出面：距离判定布尔值在指定输出方揭示；输入坐标与中间差值不进输出面（半诚实模型）
```

（`policy` 行里的「……」是本文档的节选省略，完整句子由
`ResultPolicy.disclosure` 生成。）

## 3. 执行口径示例（业务暴露 vs 协议泄漏，并排出现）

以 `CellSetIntersect` + `REVEAL_COUNT` 为例：业务层拿到 `k = |A∩B|` 而不是集合
本体；但 `protocol_leak` 不随策略改变——接收方在协议内部**仍然**拿到交集本体。
CLI 输出形如：

```text
      policy    : REVEAL_COUNT（业务层暴露 count；协议内部泄漏面不随策略改变：接收方仍获得交集本体）
```

这既是最小化披露的工程化落地，也是审计演练的现成工具：用 `REVEAL_COUNT`
代替 `REVEAL_INTERSECTION` 重跑，对比“业务层看到的数据”缩小了多少——
而 `protocol_leak` 一栏保持不变。

## 4. 对拍口径（一处刻意的“不折算”）

运行时（`_reference_under_policy`）只对**业务暴露**做折算：COUNT 与参考基数
比较；其余策略**不折算**——例如 `Intersects` 的参考值保持原样。这是刻意行为：
把任何参考值都 `bool()` 折算会**掩盖**测试中故意注入的错误参考，
让“该失败的对拍通过”。对拍口径不允许这种软化。

## 5. 与部署期设计的关系（REVEAL_TO_REGULATOR）

`REVEAL_TO_REGULATOR` 描述的是“结果只形成给监管方的最小化报告”这类
**部署期**执行模式：真正的落地需要监管方链路、审计与治理设计，远超本 MVP
进程内两方模拟的范围。本仓库的处置是**显式拒绝**（不伪造实现；PSI 与 MPC
两族同判），并在登记表里保留该策略名，供后续部署设计对接。

## 6. 复现

```bash
cd /mnt/c/Users/DELL/Documents/Codex/2026-09-20/geosot-3d-dqg-4d-c-users-2/GIS_SPU
/opt/miniconda3/envs/spu311/bin/python -m pytest tests/test_result_policy.py -q
# 编译期端到端（策略随步骤带出、CLI 打印 policy / reveals）
/opt/miniconda3/envs/spu311/bin/python -m pytest tests/test_end_to_end.py -k MpcResultPolicy -q
```