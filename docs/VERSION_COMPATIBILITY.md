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
| MPC 通信量基线 | 见 `docs/mpc_comm_baseline.json`（P2-2 新增，规程 §8.4） |
| `TemporalOverlap` 电路 A/B | 见 `docs/mpc_temporal_ab.json`（P3 新增，规程 §4.1） |
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
> 2026-10-06 一行的要点：除法从 `WeightedSum` 生成代码里移除后，
> `FM32 × K=256` 由崩溃转为可用，K=64…4096 由 43%–93% 偏差转为
> **0/30 逐位一致**；通信量（ABY3/SEMI2K/SECURENN/CHEETAH, K=256）
> 分别降 56% / 68% / 93% / 72%。见 `docs/MPC_BENCHMARK_PROTOCOL.md` §4.2 / §4.3。
>
> 上一轮记录的两条**负面**结论（真机实测，不粉饰）：
> `WeightedSum` 结尾的 `//` 在 SPU 上是近似且非确定的除法；
> `WeightedSum × FM32` 因除法内部需要 64 位环而**不可用**（见
> `docs/MPC_BENCHMARK_PROTOCOL.md` §4.2 / §4.3）。

| 日期 | GIS_SPU commit | 平台 | Python | SPU | JAX | 真机项 | 结论 |
|---|---|---|---|---|---|---|---|
| 2026-10-04 | `5daf60e`（工作区含未提交改动） | WSL2 Ubuntu 26.04.1 / x86_64 | 3.11.16 | 0.9.5 | 0.4.34 | PSI 真机全协议 + SPU/MPC + 全量测试 | **verified**（`653 passed`，0 failed，0 skipped） |
| 2026-10-05 | `423946d` + 工作区未提交改动（P0 MPC 协议入 planner / P1 MPC 代价基线 / P1.5 重复实验） | WSL2 Ubuntu 26.04.1 / x86_64 | 3.11.16 | 0.9.5 | 0.4.34 | 全量测试 + MPC 协议矩阵（5 协议 × 3 算子 × 3 环宽）+ MPC 代价基线 + 重复实验（×5 / ×30） | **verified**（`748 passed`，0 failed，0 skipped） |
| 2026-10-06 | `bd151ae` + 工作区未提交改动（P2-1：`WeightedSum` 定点 scale 改编译期常量、生成代码去除法） | WSL2 Ubuntu 26.04.1 / x86_64 | 3.11.16 | 0.9.5 | 0.4.34 | 全量测试 + MPC 基线重跑（单次 / ×5 / ×30）+ 三环宽精确性 + 通信量探针 | **verified**（`753 passed`，0 failed，0 skipped） |
| 2026-10-06 | `5a79c2c` + 工作区未提交改动（P2-2：通信量入基线，`capture_comm` / `--comm`） | WSL2 Ubuntu 26.04.1 / x86_64 | 3.11.16 | 0.9.5 | 0.4.34 | 全量测试 + 通信量基线（全量 ×5，带 profiling）+ MPC 基线重跑（单次 / ×5 / ×30） | **verified**（`773 passed`，0 failed，0 skipped） |
| 2026-10-06 | `4716f4b` + 工作区未提交改动（P3：`TemporalOverlap` 第二套电路 `sweep` + 两套电路的成对 A/B） | WSL2 Ubuntu 26.04.1 / x86_64 | 3.11.16 | 0.9.5 | 0.4.34 | 全量测试 + MPC 基线重跑（单次 / ×5 / ×30）+ 通信量基线（×5）+ `sweep` A/B（×5，含 `sort` / `reduce_window` 真机实测） | **verified**（`793 passed`，0 failed，0 skipped） |
| 2026-10-06 | `b79da30` + 工作区未提交改动（P4：MPC 协议按实测代价选择，`REF2K` 不再自动选中） | WSL2 Ubuntu 26.04.1 / x86_64 | 3.11.16 | 0.9.5 | 0.4.34 | 全量测试 + `DistanceLE` 自动协议真跑 + 排序第二协议（SEMI2K）真跑 | **verified**（`815 passed`，0 failed，0 skipped） |
| 2026-10-06 | `498e12f` + 工作区未提交改动（P5：位平面布局 D3 预测层，`planner/layout.py` + CLI `--layout-shape`） | WSL2 Ubuntu 26.04.1 / x86_64 | 3.11.16 | 0.9.5 | 0.4.34 | 全量测试 + `--layout-shape` 端到端（`distance_check` / `risk_score` 两个示例）+ 课题产物逐项复算 | **verified**（`838 passed`，0 failed，0 skipped） |
| 2026-10-06 | `a811246`（P6：打包收益上界实测，SPU 按环元素计费） | WSL2 Ubuntu 26.04.1 / x86_64 | 3.11.16 | 0.9.5 | 0.4.34 | 全量测试 + 打包前提探针（位宽扫描 / 规模扫描，`--packing-probe`） | **verified**（全量用例在 P7-P0 后为 `893 passed`，0 failed，0 skipped） |
| 2026-10-06 | `d91da69`（P7-P0：打包电路取槽步单价实测） | WSL2 Ubuntu 26.04.1 / x86_64 | 3.11.16 | 0.9.5 | 0.4.34 | 全量测试（2026-10-08 复跑）+ 取槽四变体 A/B（`test_slot_cost_probe.py` 25 项） | **verified**（`893 passed`，0 failed，0 skipped） |
| 2026-10-08 | `66c823a` + 工作区改动（CI 修复：`requirements-spu.txt` 补 `pytest`；PSI 纯输入校验提到能力门之前；无 SPU 环境下的 CLI 用例改按原因 skip） | WSL2 Ubuntu 26.04.1 / x86_64 | 3.11.16 | 0.9.5 | 0.4.34 | 全量测试（有 SPU）+ 全量测试（无 SPU 环境） | **verified**（有 SPU `893 passed`；无 SPU `766 passed / 127 skipped / 0 failed`） |
| 2026-10-08 | `52bfb5d`（P0：免逐槽提取路线筛选 + 逐原语真机核验；P1：K/size_hint 编译期口径 + 探针 CLI） | WSL2 Ubuntu 26.04.1 / x86_64 | 3.11.16 | 0.9.5 | 0.4.34 | 全量测试 + 路线筛选探针（6 变体 ×3 次，与目标语义对拍）+ 逐原语探针（19 原语 ×3 次）+ 取槽探针重跑 + Geo-RR22 覆盖关系实证（21973 条真实码） | **verified**（`949 passed`，0 failed，0 skipped） |
| 2026-10-08 | 工作区未提交改动（P0 收尾：`dot` 读数根因定位——计费维度从输入元素改按输出个数；新增收缩类用例 `dot_long`/`sum`/`sum_long`/`matmul`；探针产物含 `cost_driver`/`cost_per_unit`。协议覆盖镜像：`SPU_PROTOCOLS_VERIFIED` 由「等于全集」改为**显式字面量**，并新增「登记即有真机用例」机检） | WSL2 Ubuntu 26.04.1 / x86_64 | 3.11.16 | 0.9.5 | 0.4.34 | 全量测试（有 SPU）+ 全量测试（无 SPU 环境）+ 逐原语探针重跑（22 原语 ×3 次，`pphlo` 追踪核对 `dot`） | **verified**（有 SPU `957 passed`，0 failed，0 skipped；无 SPU `829 passed / 127 skipped / 1 failed`，该 1 条为**先前已存在**、与本版改动无关，见附注） |

> 2026-10-06 第二行（P2-2）的要点：通信量接进 `run_spu_simulation(capture_comm=True)`
> 与每条基线记录，新增产物 `docs/mpc_comm_baseline.json` / `.csv`。
> 关键实测：SPU 原生日志是**进程级**开关，PSI 路径会关掉它（`_set_native_log(quiet=True)`），
> 关过之后同一进程里 pphlo profile 行不再出现——所以 `capture_comm=True` 必须先
> 用 `libspu.logging.setup_logging` 把它打开（`system_log_path` 指 `/dev/null`）。
> 这条"跨模块的全局状态耦合"有回归测试守（`test_comm_capture_survives_a_prior_native_log_shutdown`）。

> 2026-10-06 末行（P4）的要点：`--protocol` 缺省从固定 `ABY3` 改为**自动**，
> 依据是 `docs/mpc_comm_baseline.json` 的实测发送+接收（`planner.select_mpc_protocol`）；
> 候选集带实测代价（`mpc_protocol_candidates_for`）；`REF2K` 留在候选里但被
> `SPU_PROTOCOLS_WITHOUT_CRYPTO` 挡在自动选择之外。**注意口径**：这不是
> "实测跑赢/跑输"的新测量，而是把已有产物接进规划层——本轮**没有**重跑基线，
> 排序数字全部来自既有产物（`tests/test_protocol_selection.py` 直接读产物核对）。
> 见 `docs/MPC_BENCHMARK_PROTOCOL.md` §8.5 与 `README.md` §5.5。

> 2026-10-06 末行（P5）的要点：位平面布局（D3）落到**预测层**——
> `planner/layout.py` 按归约轴选 L1/L2，条数公式与课题交付物
> `outputs/格网数据样例_明文与密态映射_v5.json` 的 `cost_prediction`
> 逐项复算一致（196000 / 49000 / 192 / 1020.8×），测试直接读那份 JSON 核对。
> **本轮没有新测量**：模型前提"通信量 ∝ 条数"取自既有产物
> `docs/mpc_comm_baseline.json`（§8.4 结论 3，ABY3 × `DistanceLE` 约 16 B/元素），
> 打包电路**未实现**，故不对打包后的通信量或墙钟作任何断言。
> 见 `docs/BITPLANE_LAYOUT.md` 与 `README.md` §8.4。

> 2026-10-06 末行（P6）的要点：**打包前提实测**。探针
> `backends/spu_backend/packing_probe.py`（产物 `docs/mpc_packing_probe.json`，
> 跑法 `tests/benchmarks/benchmark_mpc.py --packing-probe`）用"秘密 × 秘密"
> 逐元素乘法实测：int8 / int32 / int64 在**同一元素数**（N=4096）下通信量之比全为
> 1.000（唯一通信原语 `multiply`），**16 B/元素** 在 512 / 1024 / 4096 元素上恒定
> ——SPU 按**环元素**计费、与输入位宽无关。**本轮仍不是打包电路的实测**：
> 槽内归约未测，收益只作上界。第一版探针用 `x * 2`（乘公开常数）实测恒为 0 字节，
> 已换掉并写进回归测试。见 `docs/MPC_BENCHMARK_PROTOCOL.md` §8.6、
> `docs/BITPLANE_LAYOUT.md` §5 与 `README.md` §8.4。

> 同日第三段（P7-P0）的要点：**取槽步单价实测**。P6 之后紧接着量打包电路必须付的
> 那一步——把 8 位值从环元素取出来（右移 + 掩码）：四变体 A/B 实测
> （`backends/spu_backend/slot_cost_probe.py`，产物 `docs/mpc_slot_cost_probe.json`）
> 秘密 × 公开常数 **0 B**、`&` 624 B/元素、`>>` 1888 B/元素、全量取槽
> **1424 B/元素 = 纯乘法 16 B/元素 的 89 倍**（重跑后 1496 / 94 倍，批间差 5%–15%）。口径：这是**前置判据**（打包划不划算），
> 不是打包电路的收益；L2 的 1020.8× 是条数比，不能当通信量比。
> 见 `docs/MPC_BENCHMARK_PROTOCOL.md` §8.7、`docs/BITPLANE_LAYOUT.md` §5.1、
> `README.md` §8.4。

> 同日第二验证（无 SPU 环境，Python 3.10 / 3.14）：`555 passed / 87 skipped / 11 failed`。
> 11 项失败全部为"SPU 不可用"的环境门断言（期望 `error`、实得 `unavailable`），
> 非功能缺陷；SPU/PSI 用例 skip 并说明缺失项。

> 同日第四段（P0 收尾）的要点：**`dot` 读数根因定位 + 计费维度更正**。上一轮把 `dot`
> 实测的 16 B 除以**输入元素数**（N=64）得到 0.25 B/元素，看着"比乘法便宜 64 倍"，
> 遂标为 `below_multiply_floor`（可疑）。打开 `pphlo` 追踪后真相是**归一化选错了维度**：
> `pphlo.dot` 是一条专用算子（1 轮、1 个环元素），实测**按输出个数计费、收缩长度免费**。
> 本轮把收缩类用例补齐（`dot` / `dot_long` / `matmul` / `sum` / `sum_long`，22 条），
> 并给每条记录加 `cost_driver` / `cost_per_unit` / `comm_per_output`；
> 对账只在**同一计费维度**上进行，`contraction_is_free` 固定"收缩长度免费"这条实测结论
> （两条读数不一致则报"存疑"）。**读数没错，尺子错了**——与 P6 `x * 2` 实测 0 B 同族。
> 这不改变 D3 结论：收缩免费省不掉按元素计费的逐槽提取。见
> `docs/MPC_BENCHMARK_PROTOCOL.md` §8.8、`docs/BITPLANE_LAYOUT.md` §7、`README.md` §8.4。

> **附注（先前已存在、与本版改动无关）**：无 SPU 环境（Python 3.14 + jax 0.11.2）下有
> 1 条失败：`tests/test_spu_backend.py::TestCapabilityProbe::test_private_dep_check_handles_attribute_form`。
> 经 `git stash` 对照确认在本版改动**之前**即失败，非本版引入；本项目未做修复，
> 也未把它算进"已验证"口径。权威口径以 SPU 环境（`957 passed / 0 failed / 0 skipped`）为准。

> 2026-10-09 的要点：**Phase 5 定界 + NPC 族编译期显式放行**。核查确认
> `spu 0.9.5` 的 `PsiProtocol` 7 个真实协议本项目**全部已登记**、6 个已真机执行，
> 唯一"SPU 已支持但编译器全链路未走完"的对象是 **NPC 族**（`ECDH_NPC`/`KKRT_NPC`）。
> 本版把放行规则从"候选 / DP 特例 / 其余拒绝"改成三档显式登记
> （`PsiProtocolSpec.explicit_only` + `PSI_PROTOCOLS_EXPLICIT_ONLY`）：候选
> `ECDH`/`KKRT`/`RR22`；显式放行 `ECDH_NPC`/`KKRT_NPC`/`DP`；其余（含未来新登记）
> 一律编译期拒绝。配套测试：`test_protocol_registry.py::TestExplicitOnlyClassification`
> 锁定"每个已登记协议恰好落入一类"，`test_planner.py` 锁定 NPC 放行与"未来协议仍被拒"，
> `test_end_to_end.py` 锁定 `--psi-protocol ECDH_NPC` 的真实执行。
> benchmark 侧新增 `--protocols ecdh-npc,kkrt-npc`（显式扫描，不进默认标准扫描）。
