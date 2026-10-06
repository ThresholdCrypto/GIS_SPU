# MPC 性能基线规程与结果（P1：预测代价 ↔ 实测代价对账；P2-2：通信量入基线）

> 状态：已执行，并在 P2-1 / P2-2 后重跑（2026-10-06）。产物：
> `docs/mpc_benchmark_baseline.json` / `.csv`（单次扫描）、
> `docs/mpc_repeat_baseline.json` / `.csv`（全量 ×5）、
> `docs/mpc_exactness_repeat.json` / `.csv`（`WeightedSum` ×30，精确性）、
> `docs/mpc_comm_baseline.json` / `.csv`（全量 ×5，**带 profiling 的通信量**，见 §8.4）。
>
> **P2-1（2026-10-06）**：`WeightedSum` 的定点 scale 改为编译期常量，
> 生成代码不再含除法 ⇒ §4.2 的近似除法与 §4.3 的 FM32 崩溃**均已闭合**，
> 两处"已知偏差登记"作废；修前/修后的对照留在各节内。
>
> **P2-2（2026-10-06）**：通信量接入基线（`capture_comm=True`）。墙钟排不出
> 协议优劣（§8.1），通信量可以；`TemporalOverlap` 上两者排序相反（§8.4）。
> **注意**：带 profiling 的墙钟不可与不带 profiling 的墙钟混用。
> 运行器：`tests/benchmarks/benchmark_mpc.py`；模块：`backends/spu_backend/benchmark.py`。
> 环境：WSL2 / Ubuntu / x86_64 / Python 3.11.16 / spu 0.9.5 / jax 0.4.34
> （见 `docs/VERSION_COMPATIBILITY.md`）。
> 与 PSI 基线（`docs/BENCHMARK_PROTOCOL.md`）同构；**对账轴不同**，见 §3。

## 1. 运行方法

```bash
cd GIS_SPU
export LD_LIBRARY_PATH="$HOME/.local/lib"          # libgomp.so.1 所在
~/.spuenv/bin/python tests/benchmarks/benchmark_mpc.py              # 标准扫描
~/.spuenv/bin/python tests/benchmarks/benchmark_mpc.py --quick      # 快扫（冒烟）
~/.spuenv/bin/python tests/benchmarks/benchmark_mpc.py \
    --ops WeightedSum --protocols ABY3,CHEETAH                      # 显式组合
~/.spuenv/bin/python tests/benchmarks/benchmark_mpc.py --comm       # 采通信量（见 §8.4）
```

> 解释器口径：本仓库既有文档（`docs/LOCAL_TEST.md`、`docs/BENCHMARK_PROTOCOL.md`）
> 写作 `/opt/miniconda3/envs/spu311/bin/python`；本基线**本轮实际使用**的是
> `~/.spuenv`（uv 建的 Python 3.11 环境，依赖组合相同，见 `docs/VERSION_COMPATIBILITY.md`）。
> 两者可互换，命令里的解释器换成任一个装了 SPU 的 3.11 环境都成立。

libspu 的原生日志会大量写入 stdout/stderr，建议重定向：`... > ~/bench_mpc.log 2>&1`；
JSON/CSV 产物不受影响。退出码：`0` = 无预期外记录；`1` = 存在预期外记录
（含"预期失败的重复行没有失败"这种上游行为变化）。

## 2. 扫描策略（standard_cases）

| 类别 | 内容 |
|---|---|
| 主扫描 | `DistanceLE` / `WeightedSum` / `TemporalOverlap` × 5 协议（REF2K / SEMI2K / ABY3 / CHEETAH / SECURENN） |
| 环宽矩阵 | 三算子 × FM32 / FM64 / FM128（协议固定 ABY3） |
| 规模扫描 | `DistanceLE` / `WeightedSum` 到 K=4096；`TemporalOverlap` 只到 K=32（二次电路，见 §4.1） |
| 位宽对账 | K ∈ {1, 8, 2^20} 的预测位宽探针；预测 b(K) > 环宽的组合**不实测** |

规模上限**按算子给**，不强行统一量纲：`TemporalOverlap` 的电路是 N×M 两两比较
（`generate_temporal_overlap` 的 `[:, None] < [None, :]`），对节点数是**二次**的。

## 3. 对账轴与字段口径

planner 预测的是**结构代价**（`N_ct` / `b` / `d` / `R`），**不是墙钟时间**。因此本基线
记录两侧，但只对**能对账的那一项**下结论：

| 轴 | 口径 |
|---|---|
| **位宽 `b(K)` ↔ 平台环宽** | `predicted_bits`（planner）vs `field_bits`（`FIELD_BITS[field]`）→ `bit_width_covered` |
| 墙钟 / 峰值内存 / PPHLO 字节数 | **平台侧回填**，仅供回归参考；**不**给"预测时间 ↔ 实测时间"误差 |

不给时间误差的理由：planner 不预测时间。硬凑一个口径等于伪造
（任务文档 §25-10）。

沿用 PSI 基线的名字与诚实规则：`status` / `note` / `error` / `wall_ms` / `peak_rss_mb` /
`memory_mb` / `agreement`。MPC 侧新增：

- `predicted_cost` / `predicted_bits` / `field_bits` / `bit_width_covered`：对账三件套；
- `pphlo_bytes`：平台侧回填的真实编译产物规模；
- `expect_status`：预期状态标记，三值语义见 §4.4。

`memory_mb` = 本次运行前后进程**峰值 RSS 增量**（高水位差，≥0），是下界性质的量；
绝对值看 `peak_rss_mb`。所有行的 `agreement=True` 都来自真实明文对拍
（`backends.plain.run_plain`）；未执行（`unavailable`）的行**不填任何数字**。

## 4. 实测发现（本阶段新增证据）

### 4.1 `TemporalOverlap` 的电路是二次的

生成代码把两段式区间比较展开成 N×M 两两比较，节点数（元素数）∝ K²：

| K | 元素数 | 实测 |
|---:|---:|---|
| 32（主扫描） | 1 024 | ok，~90–130 ms |
| 256 | 65 536 | **CHEETAH 单条 > 90 s 不返回**（未完成，不计入基线） |
| 4096 | 16 777 216 | 展开即拖死进程 |

结论：`TemporalOverlap` 的规模上限必须单独压低（本基线取 K=32）；要往上走，
先改生成电路的形态（分段扫描 / 排序归并），而不是调大规模参数。

### 4.2 `WeightedSum` 的定点整除在 SPU 上是**近似且非确定**的

> **已闭合（P2-1，2026-10-06）**：`scale` 已改为**编译期常量**，`scale == 1`
> 时生成代码里不再有除法。修前/修后（K=256，`WeightedSum`）：
>
> | 指标 | 修前 | 修后 |
> |---|---:|---:|
> | PPHLO 字节数 | 2 400 | **938** |
> | 通信量 ABY3（3 方） | 9 286 B | **4 112 B**（−56%） |
> | 通信量 SEMI2K（3 方） | 51 544 B | **16 384 B**（−68%） |
> | 通信量 SECURENN（3 方） | 125 368 B | **8 192 B**（−93%） |
> | 通信量 CHEETAH（2 方） | 5 246 783 B | **1 474 286 B**（−72%） |
> | CHEETAH 墙钟中位数 | 1 149.8 ms | **85.2 ms**（−93%） |
> | K=4096 与明文一致 | 偏差率 93% | **0/30** |
>
> ⚠️ **口径更正（P2-2 重测）**：这张表里的通信量是**探针**量的，其中 SEMI2K
> 跑的是 **3 方**；§8.4 的基线用每协议的**参与方下限**（SEMI2K → 2 方），
> 同一组合实测 **8 192 B**。已实测确认差异来源就是参与方数量：
> SEMI2K 2 方 8 192 B / 3 方 16 384 B。⇒ **比通信量必须同一参与方数量**，
> 引用本表时请认括号里的方数，或直接用 §8.4 的基线口径。
> 
> 下面这段是**修复前的实测记录**，作为"为什么不能把除法留在电路里"的证据保留。

生成代码的 `acc // scale` 在 HLO 里展开为 `divide + remainder + select + sign`
（见 README §6.2），SPU 的除法是迭代近似实现（栈里是 `div_goldschmidt`）。
`jnp.sum` 与 `sum(w*v)` 本身**精确且确定**——偏差**只**来自 `//`：

| K | `scale` | 明文真值 | 实测 | 结论 |
|---:|---:|---:|---|---|
| 256 | 1 | 46 370 | 逐位一致（`max_abs_error=0`） | ok |
| 1024 | 1 | 202 971 | 一轮偏 1、另一轮逐位一致 | **非确定** |
| 4096 | 1 | 826 416 | 在 826412–826416 间抖动，`max_abs_error` 最大 4 | error |

两个要点：

1. **`scale == 1` 也不是恒等映射**——`acc // 1` 在密态下依旧是近似除法；
2. 偏差**随累加量级上升**，且同一组合重跑可能恰好对上——所以它不能写成
   `expect_status="error"`（那会让基线变成抽奖：同一份代码两次运行给出不同退出码），
   而写成 `expect_status="deviation"`（§4.4）。

### 4.3 新发现的缺陷：`WeightedSum` × FM32 起不来（预测 ≠ 实测）

> **已闭合（P2-1，2026-10-06）**：除法移除后 `WeightedSum × FM32` 实测为
> `ok` 且与明文逐位一致（`tests/test_spu_backend.py` 有回归用例）。
> 注意"预测 ≠ 实测"这个**教训**仍然成立：planner 的位宽预测 b(K) 描述的是
> **数据**位宽，永远覆盖不了"实现路径对环宽的额外要求"这类第二约束；
> 这次的实例是除法路径内部要 64 位环。下面保留当时的现场记录。

```
WeightedSum ABY3 FM32 K=256 → error
[Enforce fail at libspu/core/encoding.cc:106]
integer encoding failed, ring=FM32 could not represent PT_I64
```

原因是定点除法路径内部要用 **64 位环**表示中间量。而 planner 的预测位宽
`b(K) = 8 + 8 + ceil(log2 K)` 只算**数据**位宽：K=256 → b=24 ≤ 32 →
`bit_width_covered=True`，看起来"够用"，实测却不可用。

这正是 P1 要找的**预测 ≠ 实测**实例：位宽覆盖判定**不足以**代表组合可用。
处理方式（本轮落地）：

- 在 `backends/spu_backend/benchmark.py` 把 `WeightedSum` × 窄环（< FM64）
  登记为**稳定失败**（`expect_status="error"`），并把原因写进 `note`；
- 在 planner 规则（`planner/registry.py` 的 `WeightedSum.notes`）加下限说明；
- **不改**位宽公式——`b(K)` 描述的是数据位宽，语义没错；缺的是"环宽下限"
  这个第二约束，属于规划器后续要补的一类约束（§7）。

### 4.4 三类特殊记录（`expect_status`）

| 标记 | 语义 | 哪些行 |
|---|---|---|
| `""`（空） | 正常用例：非 `ok` / `unavailable` 即预期外 | 绝大多数行 |
| `"error"` | **稳定**失败，已定位原因 | `WeightedSum` × FM32 |
| `"deviation"` | **非确定**偏差：`ok` 与 `error` **都算符合**，第三种状态才算预期外 | `WeightedSum` × FM64，K ≥ 1024 |
| `execute=False` | 越界占位：`unavailable` + 原因，不填数字 | `WeightedSum` × FM32 × K=2^20 |

保留 `"deviation"` 而不是直接删掉这些行：一旦上游修好除法，这些行会开始
**稳定** ok，`unexpected_records` 不会报警——但文档与 `note` 会一直写着
"这是非确定偏差"，需要人工复核后清理。**登记而不掩盖**是基线的作用。

> **P2-1 之后**：`standard_cases()` **不再产出任何**带 `expect_status` 的行——
> 上表里的 `error` / `deviation` 两类登记都已随除法移除而作废（§4.2 / §4.3）。
> 机制保留：`EXPECT_DEVIATION` 与 `unexpected_records` 的处理逻辑仍在，
> 遇到非确定路径时不必重新发明；测试用合成行覆盖它的语义。
> 反过来，**若有人把运行时除法加回生成代码**，基线会立刻以
> "非 ok 记录"报出来（而不是被提前登记成预期失败）。

## 5. 结果摘要

| 用例 | 状态 | 预测 b | 环宽 | 覆盖 | wall_ms | mem_MB | max_err | agreement |
|---|---|---:|---:|---|---:|---:|---:|---|
| DistanceLE REF2K FM64 K=256 | ok | 8 | 64 | True | 69.7 | 39.5 | 0 | True |
| DistanceLE SEMI2K FM64 K=256 | ok | 8 | 64 | True | 46.9 | 1.4 | 0 | True |
| DistanceLE ABY3 FM64 K=256 | ok | 8 | 64 | True | 47.3 | 2.0 | 0 | True |
| DistanceLE CHEETAH FM64 K=256 | ok | 8 | 64 | True | 196.8 | 115.4 | 0 | True |
| DistanceLE SECURENN FM64 K=256 | ok | 8 | 64 | True | 47.2 | 0.0 | 0 | True |
| WeightedSum REF2K FM64 K=256 | ok | 24 | 64 | True | 35.0 | 0.0 | 0 | True |
| WeightedSum SEMI2K FM64 K=256 | ok | 24 | 64 | True | 35.9 | 0.0 | 0 | True |
| WeightedSum ABY3 FM64 K=256 | ok | 24 | 64 | True | 35.9 | 0.0 | 0 | True |
| WeightedSum CHEETAH FM64 K=256 | ok | 24 | 64 | True | 85.6 | 0.0 | 0 | True |
| WeightedSum SECURENN FM64 K=256 | ok | 24 | 64 | True | 35.5 | 0.0 | 0 | True |
| TemporalOverlap REF2K FM64 K=32 | ok | 18 | 64 | True | 77.1 | 0.0 | 0 | True |
| TemporalOverlap SEMI2K FM64 K=32 | ok | 18 | 64 | True | 94.5 | 0.0 | 0 | True |
| TemporalOverlap ABY3 FM64 K=32 | ok | 18 | 64 | True | 86.5 | 0.0 | 0 | True |
| TemporalOverlap CHEETAH FM64 K=32 | ok | 18 | 64 | True | 1440.9 | 647.7 | 0 | True |
| TemporalOverlap SECURENN FM64 K=32 | ok | 18 | 64 | True | 643.8 | 0.0 | 0 | True |
| DistanceLE ABY3 FM32 K=256 | ok | 8 | 32 | True | 43.8 | 0.0 | 0 | True |
| DistanceLE ABY3 FM64 K=256 | ok | 8 | 64 | True | 44.2 | 0.0 | 0 | True |
| DistanceLE ABY3 FM128 K=256 | ok | 8 | 128 | True | 44.0 | 0.0 | 0 | True |
| WeightedSum ABY3 FM32 K=256 | ok | 24 | 32 | True | 35.3 | 0.0 | 0 | True |
| WeightedSum ABY3 FM64 K=256 | ok | 24 | 64 | True | 33.0 | 0.0 | 0 | True |
| WeightedSum ABY3 FM128 K=256 | ok | 24 | 128 | True | 32.5 | 0.0 | 0 | True |
| TemporalOverlap ABY3 FM32 K=32 | ok | 18 | 32 | True | 85.0 | 0.0 | 0 | True |
| TemporalOverlap ABY3 FM64 K=32 | ok | 18 | 64 | True | 89.0 | 0.0 | 0 | True |
| TemporalOverlap ABY3 FM128 K=32 | ok | 18 | 128 | True | 84.9 | 0.0 | 0 | True |
| DistanceLE ABY3 FM64 K=64 | ok | 8 | 64 | True | 52.2 | 0.0 | 0 | True |
| DistanceLE ABY3 FM64 K=256 | ok | 8 | 64 | True | 44.5 | 0.0 | 0 | True |
| DistanceLE ABY3 FM64 K=1024 | ok | 8 | 64 | True | 53.4 | 0.0 | 0 | True |
| DistanceLE ABY3 FM64 K=4096 | ok | 8 | 64 | True | 57.8 | 0.0 | 0 | True |
| WeightedSum ABY3 FM64 K=64 | ok | 22 | 64 | True | 33.6 | 0.0 | 0 | True |
| WeightedSum ABY3 FM64 K=256 | ok | 24 | 64 | True | 33.3 | 0.0 | 0 | True |
| WeightedSum ABY3 FM64 K=1024 | ok | 26 | 64 | True | 35.6 | 0.0 | 0 | True |
| WeightedSum ABY3 FM64 K=4096 | ok | 28 | 64 | True | 34.5 | 0.0 | 0 | True |
| TemporalOverlap ABY3 FM64 K=8 | ok | 18 | 64 | True | 89.5 | 0.0 | 0 | True |
| TemporalOverlap ABY3 FM64 K=16 | ok | 18 | 64 | True | 93.0 | 0.0 | 0 | True |
| TemporalOverlap ABY3 FM64 K=32 | ok | 18 | 64 | True | 83.5 | 0.0 | 0 | True |
| WeightedSum ABY3 FM32 K=1048576 | unavailable | 36 | 32 | False | - | - | - | - |
（每行数值来自 `docs/mpc_benchmark_baseline.json`，本表为可读渲染；字段口径见 §3。）

本次汇总（P2-1 修复后重跑）：**36 条 → ok 35 / error 0 / unavailable 1 / 预期外 0**。

## 6. 观察与解读（不得过度引申）

- **CHEETAH 的"慢"几乎全部来自那一步除法**（P2-1 修前 1.15–1.26 s、
  含 649 MB 内存抬升；删掉除法后中位数 **85 ms**，−93%）。这一条同时说明：
  拿"某协议更慢"当结论之前，要先确认电路本身有没有可去掉的高代价原语；
- **REF2K / SEMI2K / ABY3 / SECURENN 在本量级仍难以排序**（33–40 ms，
  差异 ≤ 10%）；要排序必须用重复实验的 p25/p75（§8）与通信量（§8.4）。
  **P2-2 已经兑现后半句**：同一次运行里这四个协议的墙钟差不到 3 ms，
  通信量却从 0 B（REF2K）到 12 032 B（SECURENN）到 2.41 MB（CHEETAH）——
  **墙钟不是"协议代价"，通信量才是**；
- **PPHLO 字节数只取决于算子与 K**（`DistanceLE` 1526、`TemporalOverlap` 2653；
  `WeightedSum` 由 2400 降到 **938**——降幅来自删除法），与协议无关——
  印证"协议选择不改电路，只改执行方案"；
- 本基线是**单次运行**，无重复实验与方差数据；作为方向性参考，
  回归门禁需在重复实验与阈值讨论之后再启用（口径与 PSI 基线一致）；
- `WeightedSum` K=1024 这一行本轮恰好逐位一致：**它不代表该组合可用**，
  只代表这一轮的舍入恰好对上（§4.2）。

## 7. 下一步

1. ~~通信量入基线~~ ——**已完成（P2-2）**：`RuntimeConfig.enable_pphlo_profile = True`
   后 SPU 日志里就有逐算子 `send bytes / recv bytes` 与
   `Link details: total send bytes N, recv bytes M`，现已接进
   `run_spu_simulation(capture_comm=True)` 并落到每条记录
   （`comm_total_bytes` 等，见 §8.4）。落地后协议选型才有站得住的判据
   ——§8.1 已经证明本机墙钟不能替代通信量结论；
2. ~~除法替代方案实测~~ ——**已完成（P2-1）**：`scale` 改编译期常量，
   `scale == 1` 无除法、`scale == 2^s` 用右移、其它情形保留并显式告警。
   证据见 §4.2 / §4.3 的修前修后对照；
3. **环宽下限约束**：P2-1 之后 `WeightedSum` 已不需要它，但"实现路径对环宽的
   额外要求"这类**第二约束**仍然没有地方表达。落地方式候选：
   (a) `OperatorRule` 增加 `min_field_bits`，(b) 只写文档。
   当前**没有**已知实例，建议**不先实现**，等第二个实例出现再抽公共机制；
4. ~~重复实验~~ ——**已完成（P1.5 + P2-1 重跑）**：见 §8；
5. **`TemporalOverlap` 电路改造**：把 N×M 两两比较换成分段扫描 / 排序归并，
   否则 K 上限被锁死在 32 量级（P2-1 的经验说明：电路形态比协议选择更能决定代价）。

## 8. 重复实验（P1.5 起，P2-1 后重跑）

单次运行不足以支撑任何排序结论（§6 已声明）。本节把主扫描重复 5 次、
把 `WeightedSum` 重复 30 次，产出两个独立产物：

```bash
# 方差：全量标准扫描 × 5
~/.spuenv/bin/python tests/benchmarks/benchmark_mpc.py --repeat 5 \
    --json docs/mpc_repeat_baseline.json --csv docs/mpc_repeat_baseline.csv
# 精确性：WeightedSum × 30（K 覆盖 64 / 256 / 1024 / 4096，三个环宽）
~/.spuenv/bin/python tests/benchmarks/benchmark_mpc.py \
    --ops WeightedSum --repeat 30 \
    --json docs/mpc_exactness_repeat.json --csv docs/mpc_exactness_repeat.csv
```

记录口径（`summarize_repeats`）与前几节一致，新增：

- `status`：多次一致取该状态，**不一致记 `mixed`**（不取多数票）；
- `status_counts`：各状态出现次数；
- `wall_ms_stats` / `wall_ms_samples`：min / p25 / **median** / p75 / max 与原始样本；
  平面列 `wall_ms` 取中位数；
- `value_runs`：**真正得出数值**的次数（崩掉的不计）；
- `deviation_rate` = 与明文不一致的次数 / `value_runs`。

### 8.1 会话内抖动：全量 × 5

| 用例 | 状态 | median | p25 | p75 | p75/p25 |
|---|---|---:|---:|---:|---:|
| DistanceLE REF2K FM64 K=256 | ok | 46.7 | 45.9 | 48.1 | 1.046 |
| DistanceLE SEMI2K FM64 K=256 | ok | 47.5 | 46.1 | 48.4 | 1.048 |
| DistanceLE ABY3 FM64 K=256 | ok | 50.0 | 48.3 | 50.4 | 1.044 |
| DistanceLE CHEETAH FM64 K=256 | ok | 174.2 | 172.4 | 175.6 | 1.019 |
| DistanceLE SECURENN FM64 K=256 | ok | 47.5 | 46.3 | 48.0 | 1.036 |
| WeightedSum REF2K FM64 K=256 | ok | 33.4 | 33.4 | 35.1 | 1.052 |
| WeightedSum SEMI2K FM64 K=256 | ok | 35.9 | 34.2 | 36.0 | 1.052 |
| WeightedSum ABY3 FM64 K=256 | ok | 33.4 | 33.2 | 35.7 | 1.076 |
| WeightedSum CHEETAH FM64 K=256 | ok | 86.9 | 84.7 | 91.2 | 1.078 |
| WeightedSum SECURENN FM64 K=256 | ok | 36.4 | 36.0 | 36.6 | 1.019 |
| TemporalOverlap REF2K FM64 K=32 | ok | 69.1 | 68.6 | 69.7 | 1.015 |
| TemporalOverlap SEMI2K FM64 K=32 | ok | 101.1 | 98.4 | 103.2 | 1.049 |
| TemporalOverlap ABY3 FM64 K=32 | ok | 95.8 | 93.3 | 97.7 | 1.047 |
| TemporalOverlap CHEETAH FM64 K=32 | ok | 1239.9 | 1132.8 | 1313.8 | 1.160 |
| TemporalOverlap SECURENN FM64 K=32 | ok | 827.2 | 815.6 | 835.9 | 1.025 |
| DistanceLE ABY3 FM32 K=256 | ok | 49.7 | 49.2 | 50.0 | 1.015 |
| DistanceLE ABY3 FM64 K=256 | ok | 47.9 | 47.7 | 48.2 | 1.011 |
| DistanceLE ABY3 FM128 K=256 | ok | 48.0 | 47.6 | 53.4 | 1.120 |
| WeightedSum ABY3 FM32 K=256 | ok | 36.4 | 36.1 | 37.9 | 1.052 |
| WeightedSum ABY3 FM64 K=256 | ok | 36.7 | 36.6 | 37.9 | 1.035 |
| WeightedSum ABY3 FM128 K=256 | ok | 35.8 | 34.7 | 35.8 | 1.034 |
| TemporalOverlap ABY3 FM32 K=32 | ok | 93.5 | 91.4 | 94.0 | 1.028 |
| TemporalOverlap ABY3 FM64 K=32 | ok | 106.1 | 94.5 | 107.7 | 1.139 |
| TemporalOverlap ABY3 FM128 K=32 | ok | 111.1 | 110.6 | 116.5 | 1.053 |
| DistanceLE ABY3 FM64 K=64 | ok | 54.1 | 52.4 | 54.2 | 1.034 |
| DistanceLE ABY3 FM64 K=256 | ok | 53.3 | 47.0 | 54.3 | 1.157 |
| DistanceLE ABY3 FM64 K=1024 | ok | 48.7 | 45.1 | 50.0 | 1.108 |
| DistanceLE ABY3 FM64 K=4096 | ok | 49.5 | 47.2 | 53.3 | 1.128 |
| WeightedSum ABY3 FM64 K=64 | ok | 36.3 | 34.6 | 38.4 | 1.110 |
| WeightedSum ABY3 FM64 K=256 | ok | 35.3 | 33.6 | 36.2 | 1.078 |
| WeightedSum ABY3 FM64 K=1024 | ok | 36.0 | 35.5 | 36.4 | 1.024 |
| WeightedSum ABY3 FM64 K=4096 | ok | 36.6 | 35.3 | 37.1 | 1.052 |
| TemporalOverlap ABY3 FM64 K=8 | ok | 97.5 | 90.4 | 98.8 | 1.092 |
| TemporalOverlap ABY3 FM64 K=16 | ok | 108.8 | 108.0 | 111.4 | 1.031 |
| TemporalOverlap ABY3 FM64 K=32 | ok | 110.7 | 108.7 | 114.4 | 1.053 |
| WeightedSum ABY3 FM32 K=1048576 | unavailable | — | — | — | — |

（来自 `docs/mpc_repeat_baseline.json`；36 条汇总 = ok 35 / unavailable 1 / 预期外 0。）

结论：

- **除 CHEETAH 外，同一会话内的抖动很小**：`p75/p25` 基本 ≤ 1.05。
  ⇒ **差异大于 5% 的协议排序才值得采信**，小于 5% 的不是；
- CHEETAH 的抖动更大（1.06–1.15）且绝对值仍高一个数量级
  （`WeightedSum` K=256：87 ms vs ABY3 33 ms）。**注意**：SPU 模拟器把各参与方
  跑在同一进程里，**不含真实网络**，而 CHEETAH 的设计目标是省**通信量**——
  这个倍数在带宽受限的真实部署下**不可外推**（§8.4 给了通信量口径）；
  §8.4 的实测支持这个警告的方向：CHEETAH 在本机的通信量确实最大
  （`WeightedSum` K=256 为 1.47 MB，是 ABY3 的 359×）。
- **4 个非 CHEETAH 协议在 3 个算子上都难以排序**（差异与抖动同量级）：
  `WeightedSum` K=256 上是 33.4/33.4/35.9/36.4/36.7 ms（ABY3/REF2K/SEMI2K/
  SECURENN…），差 10% 也就是 3 ms，**不足以支撑选型**；
- `TemporalOverlap` 上 SECURENN（827 ms）与 CHEETAH（1240 ms）明显更高——
  这是**本轮唯一**超出抖动的排序结论（K=32，二次电路，见 §4.1）。

### 8.2 精确性：`WeightedSum` × 30（P2-1 之后）

| K | ok 次数 | 得出数值次数 | 偏差率 | median ms |
|---:|---:|---:|---:|---:|
| 64 | 30 / 30 | 30/30 | 0% | 34.7 |
| 256 | 30 / 30 | 30/30 | 0% | 39.6 |
| 1024 | 30 / 30 | 30/30 | 0% | 35.5 |
| 4096 | 30 / 30 | 30/30 | 0% | 36.5 |

结论：

- **K = 64 / 256 / 1024 / 4096 全部 30/30 与明文逐位一致**（`deviation_rate = 0`）；
  修前 K=1024 是 43%、K=4096 是 93%（§4.2）；
- 环宽 FM32 / FM64 / FM128 三档同样 30/30 一致（含修前必崩的 `FM32 × K=256`）；
- **注意这不是"除法变准了"**：是除法已经不在电路里了。`scale == 1` 时
  生成代码直接返回 `jnp.sum(w*v)`；`jnp.sum` 与 `w*v` 本来就是精确且确定的。

### 8.3 本节能得出什么、不能得出什么

| 能得出 | 不能得出 |
|---|---|
| `WeightedSum` 的整除偏差**已随 P2-1 消失**（0/30，全 K 全环宽） | 其它算子是否也有隐藏的近似路径（未逐个探测） |
| 会话内 `p75/p25` 多数 ≤ 1.05，给了"多小算噪声"的标尺 | 跨机器 / 跨负载的抖动（只有一台机器） |
| `TemporalOverlap` 上 SECURENN / CHEETAH 显著更慢 | 其余 4 个协议之间的排序（差异在噪声内） |
| 协议选择**不改电路**（PPHLO 字节数与协议无关） | CHEETAH 在真实网络下的表现（模拟器不含网络） |

### 8.4 通信量测量（P2-2 已落地）

**为什么必须有这一节**：§8.1 已经证明本机墙钟排不出协议优劣（4 个非 CHEETAH
协议的差值与抖动同量级）。通信量是**唯一能把它们分开**的代价轴，因为
SPU 模拟器把参与方跑在同一进程里，**不含真实网络**——墙钟测不到"谁在通信"。

取数方式（实测，不是推断）
------------------------

`RuntimeConfig.enable_pphlo_profile = True` 之后，SPU 自己会把逐算子与链路级
通信量写进日志：

```
pphlo.multiply, executed 1 times, duration ..., send bytes 2048 recv bytes 2048, ...
Link details: total send bytes 4681, recv bytes 4605, send actions 321, recv actions 316
```

这三个坑都是实测踩出来的，缺一条就会拿到 0 或假数：

1. **行由 C 层 spdlog 写出**，`contextlib.redirect_stdout` 抓不到 ——
   必须 **fd 级重定向**（`os.dup2`），见 `backends/spu_backend/profile.py`
   的 `capture_native_logs`；解析必须在 `with` 块**内**完成（出块即关 fd）；
2. **原生日志是进程级开关**：`backends/psi_backend/runtime.py::_set_native_log(quiet=True)`
   会把它关掉（PSI 路径每次求交都关）。被关过一次之后，同一进程里后续所有
   SPU 运行的 profile 行都不再出现 —— 实测表现为"fd 重定向拿到空文本，
   通信量静默变成取不到"。所以 `capture_comm=True` 会先调
   `enable_native_console_log()` 把它打开（`system_log_path` 一律改指
   `/dev/null`，否则默认的相对路径 `'spu.log'` 会在 CWD 落盘）；
3. **CHEETAH 必须用 2 方**（`Simulator(2, ...)`）；用 3 方会
   `[yacl] Get data timeout, key=root:P2P-2:1->0`。`protocol_min_world_size`
   已经给出了每协议的下限，基线按它取默认值。

落地形态（P2-2 改动清单）
------------------------

- `run_spu_simulation(..., capture_comm=True)`：开 profiling、fd 捕获、解析，
  在 `SpuRunResult` 上填 `comm_*` 与 `profiled`；
- 基线每条记录新增 `profiled` / `comm_send_bytes` / `comm_recv_bytes` /
  `comm_total_bytes` / `comm_send_actions` / `comm_recv_actions` /
  `comm_by_primitive` / `comm_total_bytes_stats`；
- 重复实验按**中位 + 区间**汇总（`comm_total_bytes_stats` 给 min/p25/median/p75/max，
  逐原语明细取"总量最接近中位那次"的快照）；
- **取不到就是 `None`**：没开 profiling、解析不到、logger 打不开，
  一律留空并在 `note` 里写明原因，绝不用推测值填充。

新产物：`docs/mpc_comm_baseline.json` / `.csv`（全量标准扫描 × 5，带 profiling）。

```bash
~/.spuenv/bin/python tests/benchmarks/benchmark_mpc.py --comm --repeat 5 \
    --json docs/mpc_comm_baseline.json --csv docs/mpc_comm_baseline.csv
```

> ⚠️ **profiled 运行的墙钟不能与 §8.1 的墙钟比**：profiling 本身有开销，
> 两个口径的应用例分开产（`mpc_repeat_baseline` 是没开 profiling 的，
> `mpc_comm_baseline` 是开了的）。

实测（`docs/mpc_comm_baseline.json`，send+recv 合计；同一次运行内取中位）
----------------------------------------------------------------------

参与方数量按 `protocol_min_world_size` 取默认值：ABY3 3、SECURENN 3、
SEMI2K 2、CHEETAH 2、REF2K 2。**这个数必须跟着通信量一起报**——已实测：
SEMI2K 同一组合 2 方 8 192 B、3 方 16 384 B（同样 `WeightedSum` K=256 / FM64）。


| 协议 | DistanceLE K=256 | WeightedSum K=256 | TemporalOverlap K=32 |
|---|---:|---:|---:|
| REF2K | 0 B | 0 B | 0 B |
| ABY3（3 方） | 4 200 B | 4 112 B | 0.52 MB（0.505–0.531） |
| SEMI2K（2 方） | 8 336 B | 8 192 B | 0.70 MB |
| SECURENN（3 方） | 12 032 B | 8 192 B | **25.84 MB** |
| CHEETAH（2 方） | **2.41 MB**（2.414–2.415） | **1.47 MB**（1.474–1.475） | 1.76 MB |

同一次运行里的墙钟（ms），用来对照"墙钟看不出差别、通信量差 3 个数量级"：

| 协议 | DistanceLE K=256 | WeightedSum K=256 | TemporalOverlap K=32 |
|---|---:|---:|---:|
| REF2K | 44.6 | 34.7 | 66.7 |
| ABY3 | 44.2 | 35.2 | 84.1 |
| SEMI2K | 46.2 | 38.8 | 94.5 |
| SECURENN | 47.9 | 35.4 | 608.3 |
| CHEETAH | 169.2 | 85.3 | 1220.8 |

三条站得住的结论
----------------

1. **`DistanceLE` / `WeightedSum` 上墙钟完全排不出协议**：除 CHEETAH 外
   四个协议墙钟都在 44–48 ms / 35–39 ms，而通信量从 **0 B 到 2.41 MB**
   跨 3 个数量级。REF2K 的 0 B 直接说明它不做密码学保护（"最快"没有意义），
   SECURENN 的 12 032 B 说明它并不是"和 ABY3 差不多"；
2. **`TemporalOverlap` 上墙钟与通信量给出相反的排序**：墙钟 SECURENN
   608 ms < CHEETAH 1221 ms（差 2.0×），通信量却是 SECURENN **25.84 MB**
   ≫ CHEETAH 1.76 MB（差 **14.7×**）。这是本轮**唯一**一处墙钟排序被
   通信量彻底反转的地方，也正是 §7 里"电路形态比协议选择更能决定代价"的
   量化证据：二次电路（N×M 两两比较）把 SECURENN 的通信量推爆；
3. **通信量对 K 线性**（ABY3 × `DistanceLE`）：K=64→1 128 B、256→4 200 B、
   1 024→16 488 B、4 096→65 640 B，即 ~16 B/元素。这与 planner 的结构代价
   模型（`N_ct` 随 K 线性）方向一致 —— 但**不是**对 planner 的验证：
   planner 预测的是电路规模，这里量的是字节数。

抖动：不是每条都确定
--------------------

`--repeat 5` 的实测结果分成两类（`comm_total_bytes_stats` 的 min/max）：

| 组合 | 抖动 | 说明 |
|---|---|---|
| REF2K（全部算子） | 0.000% | 恒为 0 B |
| ABY3 / SEMI2K / SECURENN × `DistanceLE`、`WeightedSum` | 0.000% | 五次逐字节一致 |
| SEMI2K / SECURENN / CHEETAH × `TemporalOverlap` | 0.000% | 五次逐字节一致 |
| CHEETAH × `DistanceLE` / `WeightedSum` | 0.03%–0.12% | 批间抖动 |
| ABY3 × `TemporalOverlap` | 3.6%–10.8% | 两次重跑实测区间；FM32 3.9–4.9%、FM64 5.0–6.2%、FM128 7.1–10.1% |

⇒ 报数**一律用中位 + 区间**，不拿单次值下结论；ABY3 × `TemporalOverlap`
这种抖动超过 4% 的组合，两个协议之间的差别要大于抖动才能算数。
（上表是**同一次**运行的快照；换一次运行中位会小幅变动，这是抖动本身，不是笔误。）

> 历史更正：P2-2 的初版探针记过"~1–2% 批间抖动（9 286 B vs 9 366 B）"。
> 那是在 **P2-1 之前**量的，当时生成代码里还有 `acc // scale` 的近似除法路径；
> 除法移除后重测（上表）显示绝大多数组合**完全确定**。旧口径已作废。
