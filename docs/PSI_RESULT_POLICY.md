# PSI 结果策略（ResultPolicy）设计说明

> 任务文档 §15。代码：`backends/psi_backend/result_policy.py`；运行时接入：
> `backends/psi_backend/runtime.py`（`run_psi_intersection(..., result_policy=...)`）；
> 测试：`tests/test_result_policy.py`（含真实 PSI 的 COUNT 口径对拍）。

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

## 2. 四种策略与当前状态

| 策略 | 业务层暴露 | 协议内部泄漏 | 本链路状态 |
|---|---|---|---|
| `REVEAL_INTERSECTION` | 交集本体 | 交集本体 | 可执行（`CellSetIntersect` 默认） |
| `REVEAL_BOOLEAN` | 布尔判定 | 交集本体 | 可执行（`Intersects` / `Contains` 默认） |
| `REVEAL_COUNT` | 交集基数 | 交集本体 | 可执行（`CellSetIntersect` 可选） |
| `REVEAL_TO_REGULATOR` | 最小化报告 | 交集本体 | **显式拒绝**：部署期模式，进程内两方模拟不提供该执行语义（不伪造实现） |

适用性矩阵（`APPLICABLE_POLICIES_BY_OP`）：

| 算子 | 默认策略 | 允许显式选择 |
|---|---|---|
| `Intersects` | `REVEAL_BOOLEAN` | 仅 `REVEAL_BOOLEAN` |
| `Contains` | `REVEAL_BOOLEAN` | 仅 `REVEAL_BOOLEAN` |
| `CellSetIntersect` | `REVEAL_INTERSECTION` | `REVEAL_INTERSECTION` / `REVEAL_COUNT` |

为什么 `Intersects` 不能选 `REVEAL_COUNT`：它的业务语义就是布尔，返回计数会
**扩大**暴露面而不是收缩；“登记为可用”等于把最小化设计反向使用。当算子不存在
比默认更小的合法策略时，`resolve_result_policy` 直接拒绝并列出可用清单。

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
进程内两方模拟的范围。本仓库的处置是**显式拒绝**（不伪造实现），并在登记表
里保留该策略名，供后续部署设计对接。

## 6. 复现

```bash
cd /mnt/c/Users/DELL/Documents/Codex/2026-09-20/geosot-3d-dqg-4d-c-users-2/GIS_SPU
/opt/miniconda3/envs/spu311/bin/python -m pytest tests/test_result_policy.py -q
```