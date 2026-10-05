# 版本兼容矩阵（GIS_SPU ↔ SPU / JAX / Python）

> 纪律（任务文档 §21）：**不靠版本号推断兼容**。升级 SPU 或 JAX 后，
> 必须重新跑能力核查与真机测试，再更新本文件。
> 能力核查依据与快照：`docs/SPU_CAPABILITY.md`、`docs/PSI_CAPABILITY.md`、
> `docs/spu_capability_report_wsl.json`、`docs/psi_capability_report_wsl.json`。

## 1. 当前验证矩阵

| GIS_SPU | 平台 | Python | SPU | JAX | RR22 参数 | 状态 |
|---|---|---|---|---|---|---|
| 当前仓库（本阶段基线） | WSL2 Ubuntu 26.04.1 / x86_64 | 3.11.16 | 0.9.5 | 0.4.34 | `low_comm_mode` | **verified**（真机跑通） |
| 当前仓库 | 任意平台（无 SPU） | 3.10 / 3.11 | — | 0.4.34 | — | 基础编译器测试可跑；SPU/PSI 用例 skip 并说明缺失项 |

本行"verified"的真机证据（2026-09-30，WSL2）：

| 项 | 证据 |
|---|---|
| PSI 真机求交 | `ECDH(SM2) / KKRT / RR22 / RR22+low_comm / DP` 全部真跑（`tests/test_psi_backend.py` 等，本阶段共数百项断言） |
| RR22 参数注入 | `Rr22Rarams.low_comm_mode` 真注入（`PsiRunResult.protocol_params` 记录实际值） |
| RR22 性能 | 见 `docs/psi_benchmark_baseline.json`（本阶段新增基线） |
| `Contains` 密态子集判定 | MPC 基数等值（SPU），退路披露为 `plaintext-fallback` |
| MPC 协议矩阵 | REF2K / SEMI2K / ABY3 / CHEETAH / SECURENN × 3 个 MPC 算子，逐条真跑（`tests/test_spu_backend.py::TestProtocolFieldSweep`、`tests/test_protocol_coverage.py`） |
| MPC 代价基线 | 见 `docs/mpc_benchmark_baseline.json`（本阶段新增，规程 `docs/MPC_BENCHMARK_PROTOCOL.md`） |
| 环境快照 | `docs/psi_capability_report_wsl.json` |

## 2. 已知的版本脆弱点（升级前先看）

- **jax 私有接口**：SPU 0.9.5 依赖 `jax._src.lib.xla_extension_version`
  （属性，不是子模块）等私有接口；jax ≥ 0.5 已移除，SPU 无法工作。
  详见 `README.md` §5.3 的对照表与两条回归测试。
- **`EcdhParams` 默认曲线无效**：`CURVE_INVALID_TYPE`，ECDH 必须显式给曲线
  （本项目缺省注入 `CURVE_SM2`）。升级后检查默认值是否变化。
- **PSI 是 CSV 文件接口**：没有内存张量接口。输入 CSV 必须有表头且至少 1 行；
  空集合（只有表头）会由本项目前置判定处理，不启动协议。
- **`keys_unique` 假设**：官方 `InputParams` 宣称键唯一（`keys_unique=true`）；
  重复键的真实行为由基线脚本实测记录，见 `docs/BENCHMARK_PROTOCOL.md`。

## 3. 升级 SPU 后的重验清单（逐项打勾）

> 说明：下列条目是**下次升级 SPU 后**的逐项重验清单，此处保持未勾选作为动作列表。
> 当前基线（SPU 0.9.5）已全部满足，见第 4 节 2026-10-04 记录。

```text
[ ] spu.psi.PsiProtocol 枚举存在且含 PROTOCOL_RR22 / PROTOCOL_DP
[ ] spu.psi.Rr22Rarams 存在，low_comm_mode 字段可传
[ ] spu.psi.PsiProtocolConfig 字段：protocol / receiver_rank /
    broadcast_result / ecdh_params / rr22_params
[ ] spu.psi.psi_execute 返回 report.original_count /
    intersection_count / intersection_unique_count
[ ] InputParams keys_unique / SourceType CSV 行为
[ ] libspu.link.Desc / create_mem 两方进程内链路
[ ] jax 私有接口对照（README §5.3）与版本门槛
```

重跑命令（WSL / spu311）：

```bash
cd GIS_SPU
/opt/miniconda3/envs/spu311/bin/python -m pytest tests/test_psi_capability.py tests/test_psi_backend.py tests/test_rr22_geosot.py tests/test_spu_backend.py -q
```

## 4. 记录模板（每次升级补一行）

```text
日期 | GIS_SPU commit | 平台 | Python | SPU | JAX | 真机项 | 结论（verified / 失败项）
```

已按模板记录：

> 2026-10-05 这一行的解释器是 `~/.spuenv/bin/python`（uv 建，Python 3.11.16；
> 依赖同为 `spu==0.9.5` / `jax 0.4.34` / `numpy<2`），不是本文件其他段落写的
> `/opt/miniconda3/envs/spu311`。两者可互换，能力核查结论一致。
> 本轮同时记录两条**负面**结论（真机实测，不粉饰）：
> `WeightedSum` 结尾的 `//` 在 SPU 上是近似且非确定的除法；
> `WeightedSum × FM32` 因除法内部需要 64 位环而**不可用**（见
> `docs/MPC_BENCHMARK_PROTOCOL.md` §4.2 / §4.3）。

| 日期 | GIS_SPU commit | 平台 | Python | SPU | JAX | 真机项 | 结论 |
|---|---|---|---|---|---|---|---|
| 2026-10-04 | `5daf60e`（工作区含未提交改动） | WSL2 Ubuntu 26.04.1 / x86_64 | 3.11.16 | 0.9.5 | 0.4.34 | PSI 真机全协议 + SPU/MPC + 全量测试 | **verified**（`653 passed`，0 failed，0 skipped） |
| 2026-10-05 | `423946d` + 工作区未提交改动（P0 MPC 协议入 planner / P1 MPC 代价基线 / P1.5 重复实验） | WSL2 Ubuntu 26.04.1 / x86_64 | 3.11.16 | 0.9.5 | 0.4.34 | 全量测试 + MPC 协议矩阵（5 协议 × 3 算子 × 3 环宽）+ MPC 代价基线 + 重复实验（×5 / ×30） | **verified**（`748 passed`，0 failed，0 skipped） |

> 同日第二验证（无 SPU 环境，Python 3.10 / 3.14）：`555 passed / 87 skipped / 11 failed`。
> 11 项失败全部为"SPU 不可用"的环境门断言（期望 `error`、实得 `unavailable`），
> 非功能缺陷；SPU/PSI 用例 skip 并说明缺失项。
