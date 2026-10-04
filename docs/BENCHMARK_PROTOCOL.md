# PSI 性能基线规程与结果（Phase 5 / 验收 F）

> 状态：已执行（2026-09-30）。产物：`docs/psi_benchmark_baseline.json`（机器可读主格式）、
> `docs/psi_benchmark_baseline.csv`。运行器：`tests/benchmarks/benchmark_psi.py`；
> 模块：`backends/psi_backend/benchmark.py`。
> 环境：WSL2 / Ubuntu / x86_64 / Python 3.11.16 / spu 0.9.5 / jax 0.4.34
> （见 `docs/VERSION_COMPATIBILITY.md`）。

## 1. 运行方法

```bash
cd GIS_SPU
/opt/miniconda3/envs/spu311/bin/python tests/benchmarks/benchmark_psi.py            # 标准扫描
/opt/miniconda3/envs/spu311/bin/python tests/benchmarks/benchmark_psi.py --quick    # 快扫（2^10/2^12）
/opt/miniconda3/envs/spu311/bin/python tests/benchmarks/benchmark_psi.py \
    --sizes 10,12 --protocols rr22,rr22-low                                         # 显式扫描
```

libspu 的原生日志会大量写入 stdout/stderr，建议重定向：`... > /tmp/bench.log 2>&1`；
JSON/CSV 产物不受影响。退出码：`0` = 无预期外记录；`1` = 存在预期外记录
（含"预期失败的重复键行没有失败"这种上游行为变化）。

## 2. 扫描策略（standard_cases）

| 类别 | 内容 |
|---|---|
| 主扫描 | ECDH / KKRT / RR22 / RR22+low_comm × N=2^10…2^18；KKRT / RR22 两族延伸 2^20 / 2^22 / 2^24；ECDH 实测到 2^20，2^22 / 2^24 记 `unavailable`（键数超线性增长，见 §6） |
| 变量矩阵（RR22，N=2^12） | 交集比例 0 / 1、高位码（≥2^63）、L=6、Z=5、receiver_rank=1 |
| 重复键矩阵（N=2^12，两边各 25%） | ECDH / RR22 / KKRT，预期行为各不相同（见 §4） |

用例排布纪律：**可完成的行排在会失败的行之前**。官方 PSI 在一次失败后可能留下影响
同进程后续运行的全局状态——曾观察到失败行之后紧邻的 ECDH 行出现一次 AllGather
超时（单条复测通过），之后把失败行统一排到末尾，结果稳定。详见 §4。

## 3. 字段口径

- 任务文档 §7.3 要求字段齐全：`protocol` / `low_comm_mode` / `n_left` / `n_right` /
  `intersection_ratio` / `psi_execute_ms` / `total_ms` / `memory_mb` / `agreement`；
- 补充字段：`setup_ms`（造数）、`io_write_ms` / `io_read_ms` / `semantic_ms`
  （运行时 `timings_ms` 拆段）、`peak_rss_mb`、`intersection_count` /
  `intersection_ratio_actual`、`status` / `note` / `error` / `expect_status`；
- `memory_mb` = 本次运行前后进程**峰值 RSS 增量**（ru_maxrss 高水位差，≥0）——
  语义是"这次运行把高水位抬升了多少"，是下界性质的量；绝对值看 `peak_rss_mb`。
  串行扫描中，较小行可能因为前面的大行已抬高水位而记 0，这是预期口径；
- `agreement` 只来自真实对拍（明文参考 `plain_intersects_reference`）；
  未执行（`unavailable`）的行**不填任何数字**，也不给 `agreement`。

## 4. 重复键的实测结论（本阶段新增证据）

| 协议 | 实测行为（N=2^12，两边各 25% 重复） | 错误 / 现象 |
|---|---|---|
| `PROTOCOL_RR22` | **error**（~35 s 重试后失败） | `Paxos error, Duplicate keys were detected`（okvs/paxos.cc） |
| `PROTOCOL_KKRT` | **error**（~35 s 重试后失败） | `Cannot find empty bin in stash`（cuckoo_index.cc） |
| `PROTOCOL_ECDH` | ok，但 `intersection_count` **含重复乘数**（3072 vs 唯一 2048；N=2^8 探针 192 vs 128） | 布尔语义不受影响，计数语义不可按唯一集合解读 |

结论（写入 `docs/GEO_RR22_DESIGN.md`）：

1. **去重不是"优化"，是进入 RR22 / KKRT 的正确性前提**；
2. ECDH 虽能完成，但计数含重复乘数——跨协议统一口径必须先去重；
3. 失败行之后同进程的后续运行偶发超时（见 §2 顺序纪律），复测通过——
   记录在案，不掩盖。

## 5. 结果摘要

| 用例 | 状态 | psi_execute_ms | total_ms | mem_MB | agreement |
|---|---|---:|---:|---:|---|
| ECDH N=2^10 | ok | 185.5 | 186.5 | 24.2 | True |
| ECDH N=2^12 | ok | 374.9 | 377.6 | 14.9 | True |
| ECDH N=2^14 | ok | 1299.4 | 1310.6 | 43.9 | True |
| ECDH N=2^16 | ok | 5295.3 | 5341.7 | 174.1 | True |
| ECDH N=2^18 | ok | 18768.4 | 19010.2 | 199.7 | True |
| ECDH N=2^20 | ok | 76772.3 | 77952.6 | 228.6 | True |
| KKRT N=2^10 | ok | 26.9 | 27.8 | 0.0 | True |
| KKRT N=2^12 | ok | 41.8 | 44.4 | 0.0 | True |
| KKRT N=2^14 | ok | 86.3 | 99.0 | 0.0 | True |
| KKRT N=2^16 | ok | 237.5 | 278.8 | 0.0 | True |
| KKRT N=2^18 | ok | 1009.4 | 1220.5 | 0.0 | True |
| KKRT N=2^20 | ok | 4189.8 | 5284.4 | 648.5 | True |
| KKRT N=2^22 | ok | 15522.0 | 21005.3 | 568.5 | True |
| KKRT N=2^24 | ok | 60643.9 | 85289.3 | 2238.5 | True |
| RR22 N=2^10 | ok | 31.1 | 32.1 | 0.0 | True |
| RR22 N=2^12 | ok | 40.7 | 43.3 | 0.0 | True |
| RR22 N=2^14 | ok | 69.1 | 78.9 | 0.0 | True |
| RR22 N=2^16 | ok | 171.9 | 216.0 | 0.0 | True |
| RR22 N=2^18 | ok | 577.1 | 765.5 | 0.0 | True |
| RR22 N=2^20 | ok | 2116.7 | 2996.6 | 0.0 | True |
| RR22 N=2^22 | ok | 6582.3 | 11797.8 | 0.0 | True |
| RR22 N=2^24 | ok | 23944.7 | 49729.9 | 713.6 | True |
| RR22+low_comm N=2^10 | ok | 31.8 | 32.7 | 0.0 | True |
| RR22+low_comm N=2^12 | ok | 37.3 | 40.2 | 0.0 | True |
| RR22+low_comm N=2^14 | ok | 70.2 | 80.6 | 0.0 | True |
| RR22+low_comm N=2^16 | ok | 193.8 | 234.8 | 0.0 | True |
| RR22+low_comm N=2^18 | ok | 707.8 | 884.9 | 0.0 | True |
| RR22+low_comm N=2^20 | ok | 2933.4 | 3759.6 | 0.0 | True |
| RR22+low_comm N=2^22 | ok | 8888.2 | 13405.1 | 0.0 | True |
| RR22+low_comm N=2^24 | ok | 31670.8 | 58026.4 | 163.5 | True |
| ECDH N=2^22 | unavailable | — | — | — | — |
| ECDH N=2^24 | unavailable | — | — | — | — |
| RR22 N=2^12 intr=0.0 | ok | 33.5 | 35.7 | 0.0 | True |
| RR22 N=2^12 intr=1.0 | ok | 24.6 | 27.7 | 0.0 | True |
| RR22 N=2^12 high_bits | ok | 41.8 | 44.5 | 0.0 | True |
| RR22 N=2^12 L=6 | ok | 42.0 | 44.6 | 0.0 | True |
| RR22 N=2^12 Z=5 | ok | 38.5 | 41.3 | 0.0 | True |
| RR22 N=2^12 recv=1 | ok | 46.0 | 48.7 | 0.0 | True |
| ECDH N=2^12 dup=0.25 | ok | 480.1 | 483.3 | 0.0 | True |
| RR22 N=2^12 dup=0.25 | error | 35031.4 | 35033.5 | 0.0 | — |
| KKRT N=2^12 dup=0.25 | error | 35019.5 | 35021.6 | 0.0 | — |

（每行数值来自 `docs/psi_benchmark_baseline.json`，本表为可读渲染；字段口径见 §3。）

## 6. 观察与解读（不得过度引申）

- **ECDH 超线性增长**：2^10 186 ms → 2^18 18.8 s → 2^20 76.8 s（规模 ×1024 → 时间 ×414），
  故本机预算不覆盖 2^22 / 2^24（如实记 `unavailable`，可在专用环境用
  `--sizes 22,24 --protocols ecdh` 补测）；
- **KKRT / RR22 增长远慢**：2^24 单次实测 RR22 求交 23.9 s / KKRT 60.6 s
  （端到端含 IO 49.7 s / 85.3 s）；
- **RR22+low_comm**：2^24 单次 31.7 s vs RR22 23.9 s；`low_comm` 的目标是**通信量**，
  本基线**未测量通信量**，单次时间差不足以断言优劣（任务文档 §25-10）；
- 本基线是**单次运行**，无重复实验与方差数据；作为方向性参考，
  回归门禁（§19）需在重复实验与阈值讨论之后再启用；
- 所有行的 `agreement=True` 都来自真实明文对拍；带噪协议（DP）不在本基线范围内。

## 7. 下一步（Phase 6 输入）

- 去重预处理优先落地（正确性，见 `docs/GEO_RR22_DESIGN.md`）；
- 以 2^24 RR22 端到端 ~50 s 为参考上限：排序 / 前缀压缩 / 分桶的收益必须实测再主张；
- 候选集剪枝先出"覆盖关系判定"实证报告；
- 如需通信量对比：另建通信计量（libspu 侧无现成计数器，需先做能力核查）。