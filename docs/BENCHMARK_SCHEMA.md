# 统一 benchmark metadata（Phase 7 / 任务文档 §九）

> 状态：已落地（2026-10-10；Phase 9 同日扩到四个协议族；Phase 11 补齐外部两档
> 的 PSI-CA 通信量与 PI-Sum 输入 I/O 两处缺口）。
> 模块：`backends/benchmark_schema.py`；投影器：`scripts/unify_benchmark.py`；
> 测试：`tests/test_benchmark_schema.py`（69 项）。
> 输入产物：`docs/psi_benchmark_baseline.json`、`docs/psi_ca_benchmark_baseline.json`、
> `docs/psi_sum_benchmark_baseline.json`、`docs/mpc_benchmark_baseline.json`、
> `docs/mpc_comm_baseline.json`（五份都在仓库里，可直接复算）。

## 1. 为什么要有这一层

任务文档 §九 要求「未来 PSI / MPC 都应该能够输出统一的 benchmark metadata」，
同时「不要为了统一而丢失 PSI 特有指标」。四个协议族的记录本来就长得不一样：

- **PSI**（libpsi 求交）：文件（CSV）接口，一条记录 = 一次真实求交
  （协议 × 规模 × 变量矩阵）；协议内部泄漏面 = **交集本体**；
- **PSI-CA**（openmined-psi 计数）：进程内库、全程内存 protobuf（无落盘），
  一条记录 = 一次真实计数；协议内部泄漏面 = **只有基数**（`count-only`）；
- **PI-Sum**（private-join-and-compute 交集内求和）：上游两个可执行文件
  （client / server 子进程 + 本机回环 gRPC），一条记录 = 一次真实求和；
  协议内部泄漏面 = **基数 + 交集内关联值之和**（`count+sum`）；
- **MPC**：`jax.jit → SPU` 进程内模拟，一条记录 = 协议 × 算子 × 环宽 × 规模的一个格子。

三条 PSI 路径**独立执行、泄漏承诺互不相同**——记录里的 `metadata.protocol_leak`
就是这条事实的落点；比较读数前先看它，别把三档混成一个“PSI”。

本层是**只读投影**：不改各族运行器、不重跑基线、不改写产物，
把各族记录摆到同一张表上。

## 2. 三层形状

| 类型 | 内容 |
|---|---|
| `CommonBenchmarkRecord` | 各族共有的 13 个字段（逐字对应 §九 的 `BenchmarkRecord`） |
| `PSIBenchmarkMetadata` | PSI（libpsi）特有指标（交集比例 / 协议耗时拆段 / 曲线关系 / 平台对拍 …） |
| `PsiCaBenchmarkMetadata` | PSI-CA 特有指标（结构 `RAW` / 泄漏码 `count-only` / 进程内存 …） |
| `PsiSumBenchmarkMetadata` | PI-Sum 特有指标（上游提交 / Paillier 模数 / 关联值规则 / 和值 …） |
| `MPCBenchmarkMetadata` | MPC 特有指标（环宽 / 重复实验统计 / 通信量拆原语 …） |
| `metadata.raw` | 原始记录**原样**带出——统一层不丢字段，provenance 可追 |

## 3. 共有字段口径

| 字段 | 单位 | PSI 来路 | PSI-CA 来路 | PI-Sum 来路 | MPC 来路 |
|---|---|---|---|---|---|
| `family` | — | `"PSI"` | `"PSI-CA"` | `"PI-SUM"` | `"MPC"` |
| `protocol` | — | 记录的 `protocol` | 记录的 `protocol`（`PSI-CA`） | 记录的 `protocol`（`PJC-PI-SUM`） | 记录的 `protocol` |
| `operation` | — | `CellSetIntersect`（基线只跑这条执行路径） | 同左 | 同左 | 记录的 `op` |
| `world_size` | 方 | `psi_backend` 协议注册表 | 记录列（运行时写入） | 记录列（运行时写入） | `spu_backend` 协议注册表 |
| `field` | — | 椭圆曲线；曲线关系为 `ignored` 的协议记 `None`（给了也不读） | `None`（无曲线 / 环宽概念） | `None`（无曲线 / 环宽概念） | 环宽 `FM32` / `FM64` / `FM128` |
| `input_size` | 条 | `n_left + n_right` | 同左 | 同左 | 可归约元素数 `k` |
| `unique_size` | 条 | `n_left_unique + n_right_unique` | 同左 | 同左 | `None`（MPC 无去重概念） |
| `result_size` | 条 | `intersection_count` | `intersection_count` | `intersection_count`（和值在 metadata） | `None`（基线未记录输出元素数） |
| `compute_time` | ms | `psi_execute_ms` | `psi_ca_execute_ms` | `pi_sum_execute_ms`（含两子进程往返） | `wall_ms - setup_ms` |
| `communication_bytes` | B | `None`（该基线未采集通信量） | `total_bytes`（协议消息载荷合计，Phase 11） | `total_bytes`（回环中继合计，Phase 10） | `comm_total_bytes` |
| `total_time` | ms | `total_ms` | `total_ms` | `total_ms` | `wall_ms`（`repeat>1` 时是中位数） |
| `memory_bytes` | B | `peak_rss_mb × 2^20` | `peak_rss_mb × 2^20`（运行器进程，覆盖协议） | `peak_rss_mb × 2^20`（两个子进程 VmHWM 探针，取较大者） | `peak_rss_mb × 2^20` |
| `status` | — | `ok` / `unavailable` / `error` | 同左 | 同左 | 同左 |

两处**容易看错**的口径，特意写在这里：

1. 记录里的 `memory_mb` 是「本次运行把进程峰值 RSS 高水位抬升了多少」（下界性质），
   不是绝对占用；`memory_bytes` 取的是 `peak_rss_mb`（绝对值），
   两个数都留在 `metadata.raw` 里。
2. PSI 的 `field` 不是环宽而是椭圆曲线；`PROTOCOL_KKRT` / `PROTOCOL_RR22`
   的曲线关系是 `ignored`，故 `field` 记 `None`——不拿「记录里有个曲线名」
   冒充「这条曲线生效了」。
3. PSI-CA 的 `memory_bytes` 是**运行器进程**峰值 RSS——openmined-psi 进程内
   加载，该读数覆盖协议；PI-Sum 的协议跑在两个子进程里，运行器进程 RSS
   **不含**它们，所以该档用 procfs `VmHWM` 采样探针直接读**子进程**峰值
   （Phase 10，client / server 各一条、取较大者；窗口误差 ≤ 采样间隔 20 ms），
   不用运行器进程的读数冒充。

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
| `input_io_time` | `metadata:input_io_time` | PSI = `io_write_ms + io_read_ms`；PI-Sum = `io_write_ms`（只计输入 CSV 落盘段，Phase 11）；PSI-CA 全程内存；MPC 无 IO（缺口见 §5） |
| `semantic_processing_time` | `metadata:semantic_processing_time` | 只有 PSI 记录 `semantic_ms`；其余三档缺口 |
| `send_bytes` `recv_bytes` `total_bytes` | `metadata:*_bytes` | MPC = `comm_*`；PI-Sum = 回环中继（Phase 10）；PSI-CA = 协议消息载荷（Phase 11，进程内链路 ⇒ **下界**）；libpsi 一档仍缺口 |
| `result_semantics` | `metadata:result_semantics` | PSI / MPC 由协议注册表派生；PSI-CA / PI-Sum 取记录列（运行时的声明，均为 `exact`） |
| `encode_time` `dedup_time` `layout_agreement` | `missing` | 见 §5 |

## 5. 缺口（登记出来，而不是静默为空）

| 指标 | PSI | PSI-CA | PI-Sum | MPC | 原因 |
|---|---|---|---|---|---|
| `encode_time` | 缺 | 缺 | 缺 | 缺 | 编码耗时在编译器侧，各族基线都不测 |
| `dedup_time` | 缺 | 缺 | 缺 | 缺 | 去重在运行时内部执行（三条 PSI 路径都去重但未单独计时；MPC 无此步） |
| `input_io_time` | 有 | 缺 | 有 | 缺 | PSI 拆 `io_write_ms + io_read_ms`；PI-Sum = `io_write_ms`（只计输入 CSV 落盘段；结果走 stdout 无读取段，Phase 11）；PSI-CA 全程内存（无 IO 步骤）；MPC 没有文件 IO |
| `semantic_processing_time` | 有 | 缺 | 缺 | 缺 | 只有 PSI 路径记录 `semantic_ms` |
| `peak_memory` | 有 | 有 | 有 | 有 | PI-Sum 子进程各挂 procfs VmHWM 采样探针（Phase 10，取 client / server 较大者；窗口误差 ≤ 20 ms 采样间隔） |
| `send_bytes` / `recv_bytes` / `total_bytes` | 缺 | 有 | 有 | 有 | PI-Sum = 回环 TCP 中继逐字节（Phase 10，gRPC 往返可观测）；PSI-CA = 协议消息 protobuf 载荷（Phase 11，进程内链路、**不是**网络观测，故是下界）；libpsi（求交档）缺计量钩子，仍开放 |
| `layout_agreement` | 缺 | 缺 | 缺 | 缺 | PSI 三档记录 `layout_id` 但不记握手结论；MPC 无布局概念 |

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

# 外部两条 PSI 路径（PSI-CA / PI-Sum）合并投影
python scripts/unify_benchmark.py docs/psi_ca_benchmark_baseline.json \
    docs/psi_sum_benchmark_baseline.json --json docs/psi_external_unified.json

# 直接比较
python scripts/unify_benchmark.py docs/psi_benchmark_baseline.json \
    --compare total_time --input-size 8192
python scripts/unify_benchmark.py docs/mpc_benchmark_baseline.json \
    docs/mpc_comm_baseline.json \
    --compare compute_time --operation TemporalOverlap --input-size 32
```

`docs/*_unified.json` 是**派生**产物，不进仓库（可随时按上面命令重算）；
仓库里留的是各族原始基线。

外部两档基线的重算（真机 WSL；PI-Sum 需要 `GIS_SPU_PJC_BIN_DIR`，
见 `docs/PSI_SUM_CAPABILITY.md` §8）：`bash scripts/verify_external_baselines_wsl.sh`
（该脚本第 1/5 步先核对 PSI-CA 依赖是不是 `requirements-psi-ca.txt` 钉死的版本，
再按哈希清单校验——不一致直接停，不会用错版本的产物产出基线）。

PI-Sum 运行器（Phase 10 起）默认开计量：通信量走回环中继（`send_bytes` /
`recv_bytes` / `total_bytes` + `comm_meter`），峰值内存走 procfs 探针
（`peak_rss_mb` / `client_peak_rss_mb` / `server_peak_rss_mb` + `memory_probe`）；
可用 `--no-measure-comm` / `--no-measure-memory` 关闭（关闭后对应键留空，
不填估计值）。计量口径与读数见 `docs/PSI_SUM_CAPABILITY.md` §7.1。

## 8. 下一步

已落地（Phase 10，2026-10-10）：

- **通信量计量**：PI-Sum 已接入回环 TCP 中继（client 与 server 之间插入中继，
  双向逐字节、应用层字节），基线读数与口径见 `docs/PSI_SUM_CAPABILITY.md` §7.1；
- **通信量计量（Phase 11）**：PSI-CA 已接入协议消息载荷计量（`Request` /
  `ServerSetup` / `Response` 的 `SerializeToString()` 长度，方向见
  `docs/PSI_CA_CAPABILITY.md` §7.1），`MISSING_METRICS["PSI-CA"]` 缩为 5 项；
- **输入 I/O 计时（Phase 11）**：PI-Sum 的 `io_write_ms`（输入 CSV 落盘段），
  读数见 `docs/PSI_SUM_CAPABILITY.md` §7.2；
- **PI-Sum 子进程内存**：procfs `VmHWM` 采样探针（20 ms 间隔，client / server
  各一条，取较大者）已接入运行器；加上 Phase 11 的输入 I/O 计时，
  `MISSING_METRICS["PI-SUM"]` 相应缩为 4 项
  （`encode_time` / `dedup_time` / `semantic_processing_time` /
  `layout_agreement`）；
- **result policy 与密码协议解耦（MPC 侧）**：统一入口 `backends/result_policy.py`
  （PSI / MPC 共用；旧路径保留为再导出兼容层），新增 `REVEAL_VALUE` 数值档与
  MPC 泄漏码 `output-only`；`REVEAL_TO_REGULATOR` 维持显式拒绝。

仍开放：

- 通信量计量还差 **libpsi 求交**一档——需要该 SDK 的计量钩子，属下一阶段
  （PSI-CA 已在 Phase 11 补齐：协议消息载荷，见 `docs/PSI_CA_CAPABILITY.md` §7.1）。
- 若要让各族运行器**直接**落统一格式（而不是事后投影），可在
  `tests/benchmarks/benchmark_psi.py` / `benchmark_psi_ca.py` /
  `benchmark_psi_sum.py` / `benchmark_mpc.py` 加 `--unified-json` 开关——
  本版**未改**运行器，投影器已覆盖该需求。
