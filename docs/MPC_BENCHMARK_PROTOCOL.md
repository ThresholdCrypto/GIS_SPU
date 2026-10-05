# MPC 性能基线规程与结果（P1：预测代价 ↔ 实测代价对账）

> 状态：已执行（2026-10-05）。产物：`docs/mpc_benchmark_baseline.json` / `.csv`（单次扫描）、
> `docs/mpc_repeat_baseline.json` / `.csv`（全量 ×5）、
> `docs/mpc_deviation_repeat.json` / `.csv`（`WeightedSum` ×30）。
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

## 5. 结果摘要

| 用例 | 状态 | 预测 b | 环宽 | 覆盖 | wall_ms | mem_MB | max_err | agreement |
|---|---|---:|---:|---|---:|---:|---:|---|
| DistanceLE REF2K FM64 K=256 | ok | 8 | 64 | True | 77.7 | 39.5 | 0 | True |
| DistanceLE SEMI2K FM64 K=256 | ok | 8 | 64 | True | 53.6 | 1.4 | 0 | True |
| DistanceLE ABY3 FM64 K=256 | ok | 8 | 64 | True | 52.2 | 2.0 | 0 | True |
| DistanceLE CHEETAH FM64 K=256 | ok | 8 | 64 | True | 210.7 | 116.5 | 0 | True |
| DistanceLE SECURENN FM64 K=256 | ok | 8 | 64 | True | 49.8 | 0.0 | 0 | True |
| WeightedSum REF2K FM64 K=256 | ok | 24 | 64 | True | 79.2 | 0.0 | 0 | True |
| WeightedSum SEMI2K FM64 K=256 | ok | 24 | 64 | True | 83.6 | 0.0 | 0 | True |
| WeightedSum ABY3 FM64 K=256 | ok | 24 | 64 | True | 83.1 | 0.0 | 0 | True |
| WeightedSum CHEETAH FM64 K=256 | ok | 24 | 64 | True | 1263.5 | 649.3 | 0 | True |
| WeightedSum SECURENN FM64 K=256 | ok | 24 | 64 | True | 126.3 | 0.0 | 0 | True |
| TemporalOverlap REF2K FM64 K=32 | ok | 18 | 64 | True | 77.8 | 0.0 | 0 | True |
| TemporalOverlap SEMI2K FM64 K=32 | ok | 18 | 64 | True | 102.4 | 0.0 | 0 | True |
| TemporalOverlap ABY3 FM64 K=32 | ok | 18 | 64 | True | 99.3 | 0.0 | 0 | True |
| TemporalOverlap CHEETAH FM64 K=32 | ok | 18 | 64 | True | 1334.4 | 12.0 | 0 | True |
| TemporalOverlap SECURENN FM64 K=32 | ok | 18 | 64 | True | 733.0 | 0.0 | 0 | True |
| DistanceLE ABY3 FM32 K=256 | ok | 8 | 32 | True | 52.8 | 0.0 | 0 | True |
| DistanceLE ABY3 FM64 K=256 | ok | 8 | 64 | True | 51.8 | 0.0 | 0 | True |
| DistanceLE ABY3 FM128 K=256 | ok | 8 | 128 | True | 49.1 | 0.0 | 0 | True |
| WeightedSum ABY3 FM32 K=256 | error | 24 | 32 | True | 183.5 | 0.0 | — | — |
| WeightedSum ABY3 FM64 K=256 | ok | 24 | 64 | True | 90.0 | 0.0 | 0 | True |
| WeightedSum ABY3 FM128 K=256 | ok | 24 | 128 | True | 81.4 | 0.0 | 0 | True |
| TemporalOverlap ABY3 FM32 K=32 | ok | 18 | 32 | True | 86.0 | 0.0 | 0 | True |
| TemporalOverlap ABY3 FM64 K=32 | ok | 18 | 64 | True | 103.4 | 0.0 | 0 | True |
| TemporalOverlap ABY3 FM128 K=32 | ok | 18 | 128 | True | 106.3 | 0.0 | 0 | True |
| DistanceLE ABY3 FM64 K=64 | ok | 8 | 64 | True | 58.2 | 0.0 | 0 | True |
| DistanceLE ABY3 FM64 K=256 | ok | 8 | 64 | True | 52.6 | 0.0 | 0 | True |
| DistanceLE ABY3 FM64 K=1024 | ok | 8 | 64 | True | 58.5 | 0.0 | 0 | True |
| DistanceLE ABY3 FM64 K=4096 | ok | 8 | 64 | True | 57.8 | 0.0 | 0 | True |
| WeightedSum ABY3 FM64 K=64 | ok | 22 | 64 | True | 83.9 | 0.0 | 0 | True |
| WeightedSum ABY3 FM64 K=256 | ok | 24 | 64 | True | 82.4 | 0.0 | 0 | True |
| WeightedSum ABY3 FM64 K=1024 | ok | 26 | 64 | True | 86.2 | 0.0 | 0 | True |
| WeightedSum ABY3 FM64 K=4096 | error | 28 | 64 | True | 85.1 | 0.0 | 4 | — |
| TemporalOverlap ABY3 FM64 K=8 | ok | 18 | 64 | True | 110.3 | 0.0 | 0 | True |
| TemporalOverlap ABY3 FM64 K=16 | ok | 18 | 64 | True | 103.3 | 0.0 | 0 | True |
| TemporalOverlap ABY3 FM64 K=32 | ok | 18 | 64 | True | 87.0 | 0.0 | 0 | True |
| WeightedSum ABY3 FM32 K=1048576 | unavailable | 36 | 32 | False | — | — | — | — |

（每行数值来自 `docs/mpc_benchmark_baseline.json`，本表为可读渲染；字段口径见 §3。）

本次汇总：**36 条 → ok 33 / error 2 / unavailable 1 / 预期外 0**。

## 6. 观察与解读（不得过度引申）

- **同规模下 CHEETAH 明显更慢**（`WeightedSum` K=256：CHEETAH 1.26 s vs
  ABY3 0.08 s），且 `memory_mb` 抬升 649 MB——但本基线**未测通信量**，
  单次时间差不足以断言优劣；`CHEETAH` 的目标是低通信，不在本次测量范围；
- **REF2K / SEMI2K / ABY3 / SECURENN 在本量级差别不大**（50–130 ms 量级），
  差异小于单次运行的噪声；要排序必须做重复实验（§7）；
- **所有 executor 的 PPHLO 字节数只取决于算子与 K**（`DistanceLE` 1526、
  `WeightedSum` 2400、`TemporalOverlap` 2653 量级），与协议无关——
  印证"协议选择不改电路，只改执行方案"；
- 本基线是**单次运行**，无重复实验与方差数据；作为方向性参考，
  回归门禁需在重复实验与阈值讨论之后再启用（口径与 PSI 基线一致）；
- `WeightedSum` K=1024 这一行本轮恰好逐位一致：**它不代表该组合可用**，
  只代表这一轮的舍入恰好对上（§4.2）。

## 7. 下一步

1. **环宽下限约束**：planner 需要一类"第二约束"——位宽覆盖之外，
   还要表达"该算子的实现路径要求环宽 ≥ N 位"（`WeightedSum` ≥ FM64）。
   落地方式候选：(a) `OperatorRule` 增加 `min_field_bits` 字段，
   编译期对显式 `--field` 给诊断；(b) 只在文档与 `notes` 里声明。
   **建议 (a)**，但需先在 validator 侧定义"用户显式指定 field"这条路径；
2. **除法替代方案实测**：用 `jax.lax.div` 之外的写法（先按位截断、或把
   `scale` 折进权重避免除）消除近似除法——**先实测**再改生成器；
   `docs/mpc_deviation_repeat.json` 提供了"改之前"的基线；
3. ~~重复实验~~ ——**已完成（P1.5）**：主扫描 ×5 与 `WeightedSum` ×30
   见 §8。**余项**：偏差率跨批次不一致（§8.2），需要定位它是
   抽样波动还是进程内状态；
4. **通信量测量**：CHEETAH / SEMI2K / ABY3 的优劣判定必须落到通信量上，
   当前 SPU 模拟器不暴露该指标，需要另行接入（独立工作项）；
   §8.1 已证明"本机墙钟不能替代通信量结论"；
5. **`TemporalOverlap` 电路改造**：把 N×M 两两比较换成分段扫描 / 排序归并，
   否则 K 上限被锁死在 32 量级。

## 8. 重复实验（P1.5：先拿方差，再谈排序）

单次运行不足以支撑任何排序结论（§6 已声明）。本节把主扫描重复 5 次、
把偏差最可疑的 `WeightedSum` 重复 30 次，产出两个独立产物：

```bash
# 方差：全量标准扫描 × 5
~/.spuenv/bin/python tests/benchmarks/benchmark_mpc.py --repeat 5 \
    --json docs/mpc_repeat_baseline.json --csv docs/mpc_repeat_baseline.csv
# 偏差率：WeightedSum × 30（K 覆盖 64 / 256 / 1024 / 4096）
~/.spuenv/bin/python tests/benchmarks/benchmark_mpc.py \
    --ops WeightedSum --repeat 30 \
    --json docs/mpc_deviation_repeat.json --csv docs/mpc_deviation_repeat.csv
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
| DistanceLE REF2K FM64 K=256 | ok | 44.1 | 43.8 | 45.1 | 1.029 |
| DistanceLE SEMI2K FM64 K=256 | ok | 45.7 | 45.4 | 46.2 | 1.016 |
| DistanceLE ABY3 FM64 K=256 | ok | 44.7 | 44.5 | 45.1 | 1.014 |
| DistanceLE CHEETAH FM64 K=256 | ok | 171.6 | 168.0 | 173.8 | 1.035 |
| DistanceLE SECURENN FM64 K=256 | ok | 46.2 | 46.1 | 46.3 | 1.005 |
| WeightedSum REF2K FM64 K=256 | ok | 63.1 | 63.0 | 64.0 | 1.016 |
| WeightedSum SEMI2K FM64 K=256 | ok | 77.1 | 76.3 | 77.4 | 1.014 |
| WeightedSum ABY3 FM64 K=256 | ok | 75.2 | 74.9 | 76.3 | 1.020 |
| WeightedSum CHEETAH FM64 K=256 | ok | 1232.6 | 1186.3 | 1378.9 | 1.162 |
| WeightedSum SECURENN FM64 K=256 | ok | 115.2 | 115.1 | 118.1 | 1.027 |
| TemporalOverlap REF2K FM64 K=32 | ok | 63.4 | 62.9 | 65.0 | 1.034 |
| TemporalOverlap SEMI2K FM64 K=32 | ok | 92.1 | 90.5 | 93.7 | 1.036 |
| TemporalOverlap ABY3 FM64 K=32 | ok | 89.9 | 89.4 | 91.6 | 1.025 |
| TemporalOverlap CHEETAH FM64 K=32 | ok | 1237.8 | 1122.8 | 1255.8 | 1.118 |
| TemporalOverlap SECURENN FM64 K=32 | ok | 623.8 | 612.8 | 631.1 | 1.030 |
| DistanceLE ABY3 FM32 K=256 | ok | 44.0 | 43.6 | 45.6 | 1.047 |
| DistanceLE ABY3 FM64 K=256 | ok | 44.0 | 42.9 | 44.1 | 1.028 |
| DistanceLE ABY3 FM128 K=256 | ok | 43.5 | 43.3 | 43.5 | 1.007 |
| WeightedSum ABY3 FM32 K=256 | error | 67.4 | 65.7 | 163.5 | 2.487 |
| WeightedSum ABY3 FM64 K=256 | ok | 75.5 | 75.2 | 76.2 | 1.014 |
| WeightedSum ABY3 FM128 K=256 | ok | 76.1 | 75.8 | 78.3 | 1.034 |
| TemporalOverlap ABY3 FM32 K=32 | ok | 83.9 | 82.4 | 84.6 | 1.027 |
| TemporalOverlap ABY3 FM64 K=32 | ok | 83.8 | 82.5 | 85.3 | 1.034 |
| TemporalOverlap ABY3 FM128 K=32 | ok | 86.0 | 85.0 | 86.3 | 1.015 |
| DistanceLE ABY3 FM64 K=64 | ok | 47.5 | 45.5 | 51.6 | 1.134 |
| DistanceLE ABY3 FM64 K=256 | ok | 46.4 | 46.2 | 46.7 | 1.012 |
| DistanceLE ABY3 FM64 K=1024 | ok | 45.9 | 45.7 | 46.9 | 1.026 |
| DistanceLE ABY3 FM64 K=4096 | ok | 46.2 | 45.8 | 49.7 | 1.086 |
| WeightedSum ABY3 FM64 K=64 | ok | 78.3 | 77.7 | 79.1 | 1.017 |
| WeightedSum ABY3 FM64 K=256 | ok | 79.8 | 78.7 | 80.8 | 1.027 |
| WeightedSum ABY3 FM64 K=1024 | ok | 79.2 | 78.6 | 80.7 | 1.027 |
| WeightedSum ABY3 FM64 K=4096 | mixed | 81.5 | 78.7 | 87.7 | 1.114 |
| TemporalOverlap ABY3 FM64 K=8 | ok | 88.9 | 86.4 | 99.0 | 1.146 |
| TemporalOverlap ABY3 FM64 K=16 | ok | 89.3 | 86.2 | 89.8 | 1.041 |
| TemporalOverlap ABY3 FM64 K=32 | ok | 86.6 | 86.4 | 86.7 | 1.003 |
| WeightedSum ABY3 FM32 K=1048576 | unavailable | — | — | — | — |

（来自 `docs/mpc_repeat_baseline.json`；36 条汇总 = ok 33 / error 1 / unavailable 1 / 预期外 0。）

结论：

- **除 CHEETAH 外，同一会话内的抖动很小**：`p75/p25` 基本 ≤ 1.05。
  这意味着**差异大于 5% 的协议排序是可信的**，小于 5% 的不是；
- `CHEETAH` 的抖动更大（1.12–1.16），且绝对值高一个数量级（`WeightedSum`
  K=256：1232 ms vs ABY3 75 ms，≈16×）——**注意**：SPU 模拟器把各参与方跑在
  同一进程里，**不含真实网络**，而 CHEETAH 的设计目标是省**通信量**；
  这个倍数在带宽受限的真实部署下**不可外推**；
- 4 个非 CHEETAH 协议在 `DistanceLE` / `TemporalOverlap` 上差异
  （44–92 ms 量级）**与本机抖动同量级**，本基线**不支持**给它们排序；
- `WeightedSum` 上 REF2K（63.1 ms）比 ABY3 / SEMI2K（75.2 / 77.1 ms）低约 16%，
  超出抖动范围，属**本机本量级**的观察值；REF2K 的安全假设与其它协议不同，
  不能只按这一个数字选型；
- `WeightedSum ABY3 FM32 K=256` 5 次全失败（`p75/p25 = 2.49` 是"失败耗时"的抖动，
  不是性能），与 §4.3 的稳定失败一致。

### 8.2 偏差率：`WeightedSum` × 30

| K | ok 次数 | 得出数值次数 | 偏差率 | 最大偏差 | median ms |
|---:|---:|---:|---:|---:|---:|
| 64 | 30 / 30 | 30/30 | 0.0% | 0 | 76.3 |
| 256 | 30 / 30 | 30/30 | 0.0% | 0 | 76.3 |
| 1024 | 17 / 30 | 30/30 | 43.3% | 1 | 77.0 |
| 4096 | 2 / 30 | 30/30 | 93.3% | 4 | 77.5 |

结论：

- **K ≤ 256：30/30 逐位一致**——小累加量级下整数路径是精确的；
- **K = 1024：偏差率 43.3%（13/30）**，最大偏差 1；
- **K = 4096：偏差率 93.3%（28/30）**，最大偏差 4；
- 偏差率**随 K 上升**，方向明确；
- **但偏差率不是一个稳定常数**：全量 ×5 那一批里 K=4096 是 3/5 偏差（60%），
  这一批是 28/30（93%）。若真值是 93%，5 次里只出 3 次偏差的概率约 0.3%——
  两批的差异**超出抽样噪声**。可能的解释是进程内状态（JIT 缓存 / 参与方
  初始化顺序）会影响舍入路径，但**本轮没有证据定位**，故只记录、不解释。
  ⇒ 对外只给"偏差会发生、且随 K 上升"，**不给精确概率**。

### 8.3 本节能得出什么、不能得出什么

| 能得出 | 不能得出 |
|---|---|
| `WeightedSum` 的整除偏差**确实发生**，且随 K 上升 | 偏差的精确概率（两批不一致，见 §8.2） |
| CHEETAH 在本机模拟器里慢一个数量级 | CHEETAH 在真实网络下更慢（省的是通信量） |
| `p75/p25` 给了"多小的差异是噪声"的标尺 | 跨机器 / 跨负载的抖动（只有一台机器一台次） |
| 协议选择**不改电路**（§6 的 PPHLO 观察） | 任何协议的绝对性能承诺 |
