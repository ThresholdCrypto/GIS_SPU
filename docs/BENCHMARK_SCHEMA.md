# 统一 benchmark metadata（Phase 7 / 任务文档 §九）

> 状态：已落地（2026-10-10）。
> 模块：`backends/benchmark_schema.py`；投影器：`scripts/unify_benchmark.py`；
> 测试：`tests/test_benchmark_schema.py`（48 项）。
> 输入产物：`docs/psi_benchmark_baseline.json`、`docs/mpc_benchmark_baseline.json`、
> `docs/mpc_comm_baseline.json`（三份都在仓库里，可直接复算）。

## 1. 为什么要有这一层

任务文档 §九 要求「未来 PSI / MPC 都应该能够输出统一的 benchmark metadata」，
同时「不要为了统一而丢失 PSI 特有指标」。两族记录本来就长得不一样：

- **PSI**：文件（CSV）接口，一条记录 = 一次真实求交（协议 × 规模 × 变量矩阵）；
- **MPC**：`jax.jit → SPU` 进程内模拟，一条记录 = 协议 × 算子 × 环宽 × 规模的一个格子。

本层是**只读投影**：不改两族运行器、不重跑基线、不改写产物，
把两族记录摆到同一张表上。

## 2. 三层形状

| 类型 | 内容 |
|---|---|
| `CommonBenchmarkRecord` | 两族共有的 13 个字段（逐字对应 §九 的 `BenchmarkRecord`） |
| `PSIBenchmarkMetadata` | PSI 特有指标（交集比例 / 协议耗时拆段 / 曲线关系 / 平台对拍 …） |
| `MPCBenchmarkMetadata` | MPC 特有指标（环宽 / 重复实验统计 / 通信量拆原语 …） |
| `metadata.raw` | 原始记录**原样**带出——统一层不丢字段，provenance 可追 |

## 3. 共有字段口径

| 字段 | 单位 | PSI 来路 | MPC 来路 |
|---|---|---|---|
| `family` | — | `"PSI"` | `"MPC"` |
| `protocol` | — | 记录的 `protocol` | 记录的 `protocol` |
| `operation` | — | `CellSetIntersect`（基线只跑这条执行路径） | 记录的 `op` |
| `world_size` | 方 | `psi_backend` 协议注册表 | `spu_backend` 协议注册表 |
| `field` | — | 椭圆曲线；曲线关系为 `ignored` 的协议记 `None`（给了也不读） | 环宽 `FM32` / `FM64` / `FM128` |
| `input_size` | 条 | `n_left + n_right` | 可归约元素数 `k` |
| `unique_size` | 条 | `n_left_unique + n_right_unique` | `None`（MPC 无去重概念） |
| `result_size` | 条 | `intersection_count` | `None`（基线未记录输出元素数） |
| `compute_time` | ms | `psi_execute_ms` | `wall_ms - setup_ms` |
| `communication_bytes` | B | `None`（该基线未采集通信量） | `comm_total_bytes` |
| `total_time` | ms | `total_ms` | `wall_ms`（`repeat>1` 时是中位数） |
| `memory_bytes` | B | `peak_rss_mb × 2^20` | 同左 |
| `status` | — | `ok` / `unavailable` / `error` | 同左 |

两处**容易看错**的口径，特意写在这里：

1. 记录里的 `memory_mb` 是「本次运行把进程峰值 RSS 高水位抬升了多少」（下界性质），
   不是绝对占用；`memory_bytes` 取的是 `peak_rss_mb`（绝对值），
   两个数都留在 `metadata.raw` 里。
2. PSI 的 `field` 不是环宽而是椭圆曲线；`PROTOCOL_KKRT` / `PROTOCOL_RR22`
   的曲线关系是 `ignored`，故 `field` 记 `None`——不拿「记录里有个曲线名」
   冒充「这条曲线生效了」。

## 4. §九 清单逐项去向

`TASK_DOC_METRICS` 把 §九「建议至少记录」的 21 项逐条登记了去向
（`common:<字段>` / `metadata:<字段>` / `missing`），测试锁定它逐字等于清单本身。

| §九 指标 | 去向 | 备注 |
|---|---|---|
| `family` `protocol` `operation` `world_size` `field` `status` | `common:*` | 同名共有字段 |
| `N` / `unique_N` | `common:input_size` / `common:unique_size` | §九 的写法与共有字段名不同 |
| `protocol_time` | `common:compute_time` | 同一件事的两个名字 |
| `peak_memory` | `common:memory_bytes` | 同上 |
| `intersection_ratio` | `metadata:intersection_ratio` | PSI |
| `input_io_time` | `metadata:input_io_time` | PSI = `io_write_ms + io_read_ms`；MPC 缺口 |
| `semantic_processing_time` | `metadata:semantic_processing_time` | PSI；MPC 缺口 |
| `send_bytes` `recv_bytes` `total_bytes` | `metadata:*_bytes` | MPC；PSI 缺口 |
| `result_semantics` | `metadata:result_semantics` | 由协议注册表派生（不在原记录里） |
| `encode_time` `dedup_time` `layout_agreement` | `missing` | 见 §5 |

## 5. 缺口（登记出来，而不是静默为空）

| 指标 | PSI | MPC | 原因 |
|---|---|---|---|
| `encode_time` | 缺 | 缺 | 编码耗时在编译器侧，两族基线都不测 |
| `dedup_time` | 缺 | 缺 | 去重在编译器侧（`backends/psi_backend/geosot_optimizer.py`），不在基线记录里 |
| `input_io_time` | 有 | 缺 | MPC 是进程内模拟，没有文件 IO |
| `semantic_processing_time` | 有 | 缺 | MPC 记录里没有语义段 |
| `send_bytes` / `recv_bytes` / `total_bytes` | 缺 | 有 | PSI 基线未装通信量计量（`docs/BENCHMARK_PROTOCOL.md` §7 已列为待办） |
| `layout_agreement` | 缺 | 缺 | PSI 记录 `layout_id` 但不记握手结论；MPC 无布局概念 |

`MISSING_METRICS` / `MISSING_RAW_KEYS` 是这两张声明的单一来源，
`tests/test_benchmark_schema.py::TestMissingMetricsAreReal` 会拿真实产物
逐条核对「声明的缺口必须真的不在记录里」。

## 6. 比较不同协议（验收 §9）

```python
compare_protocols(records, metric, *, family=None, operation=None, input_size=None)
```

纪律：

- **只比较 `status == ok` 的行**；`unavailable` / `error` 的行进 `skipped`，不给数字；
- 记录里没有该指标的行走 `skipped`（不拿别的数凑）；
- **不同族 / 不同算子 / 不同规模不比**：筛完还剩多个就直接报错，
  宁可让调用方显式收窄，也不把不可比的读数排成一张榜；
- 同一用例名有多条 `ok` 记录（基线里确有同用例重测的多行）时取**中位数**——
  与 MPC 运行器对重复实验的聚合口径一致（`backends/spu_backend/benchmark.py`
  的 `wall_ms` 取各次中位数），并把 `samples`（样本数）与 `spread`（极差）
  一并带出：读数稳不稳要看得见，不是抹平。

实测输出（`total_time`，PSI N=2^12，每个参与方 4096 条）：

| 用例 | total_ms |
|---|---:|
| RR22 N=2^12 intr=1.0 | 27.687 |
| RR22 N=2^12 intr=0.0 | 35.661 |
| RR22+low_comm N=2^12 | 40.156 |
| RR22 N=2^12 | 43.271 |
| KKRT N=2^12 | 44.418 |
| RR22 N=2^12 high_bits | 44.454 |
| RR22 N=2^12 L=6 | 44.620 |
| RR22 N=2^12 recv=1 | 48.740 |
| ECDH N=2^12 | 377.626 |

（与 `docs/BENCHMARK_PROTOCOL.md` §6 的结论一致：ECDH 在该规模上远慢于 KKRT / RR22。）

`compute_time`，MPC `TemporalOverlap` K=32：

| 用例 | 中位 ms | 样本 | 极差 |
|---|---:|---:|---:|
| TemporalOverlap REF2K FM64 K=32 | 72.938 | 2 | 14.549 |
| TemporalOverlap ABY3 FM32 K=32 | 85.278 | 2 | 8.370 |
| TemporalOverlap ABY3 FM64 K=32 | 85.738 | 6 | 35.973 |
| TemporalOverlap ABY3 FM128 K=32 | 90.952 | 2 | 2.346 |
| TemporalOverlap SEMI2K FM64 K=32 | 96.384 | 2 | 2.158 |
| TemporalOverlap ABY3 FM64 K=32 [sweep] | 226.206 | 2 | 1.784 |
| TemporalOverlap SECURENN FM64 K=32 | 622.253 | 2 | 20.961 |
| TemporalOverlap CHEETAH FM64 K=32 | 1222.761 | 2 | 37.896 |

（`[sweep]` 是第二套电路，与 pairwise 分行——同一协议名的两种配置不混成一行。）

## 7. 运行方法

```bash
cd GIS_SPU
# 投影成统一 JSON（+ 可选 CSV，只含共有字段）
python scripts/unify_benchmark.py docs/psi_benchmark_baseline.json \
    --json docs/psi_benchmark_unified.json
python scripts/unify_benchmark.py docs/mpc_benchmark_baseline.json \
    docs/mpc_comm_baseline.json --json docs/mpc_benchmark_unified.json

# 直接比较
python scripts/unify_benchmark.py docs/psi_benchmark_baseline.json \
    --compare total_time --input-size 8192
python scripts/unify_benchmark.py docs/mpc_benchmark_baseline.json \
    docs/mpc_comm_baseline.json \
    --compare compute_time --operation TemporalOverlap --input-size 32
```

`docs/*_unified.json` 是**派生**产物，不进仓库（可随时按上面命令重算）；
仓库里留的是两族原始基线。

## 8. 下一步

- Phase 8（任务文档 §十）：result policy / regulator-only disclosure 与密码协议解耦
  （已有 `backends/psi_backend/result_policy.py` 与 `tests/test_result_policy.py` 的基础，
  待补齐 MPC 侧与统一入口）；
- 若要让两族运行器**直接**落统一格式（而不是事后投影），可在
  `tests/benchmarks/benchmark_psi.py` / `benchmark_mpc.py` 加 `--unified-json` 开关——
  本版**未改**运行器，投影器已覆盖该需求。
