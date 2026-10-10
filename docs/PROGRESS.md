# 项目进展说明：geo-secure 低门槛隐私计算编译器（MVP）

> **口径对齐**：2026 年 9 月工作月报（任务 3）。
> **截至 2026-10-10**，本地与远程 `main` 一致：**32 个提交**、
> **1212 项自动化测试全部通过（0 失败 / 0 跳过）**。
> **验证环境**：WSL2 Ubuntu 26.04.1 / x86_64，Python 3.11.16，spu 0.9.5，jax 0.4.34。

---

## 1. 一句话结论

地信开发者只用少量业务 API 描述任务，编译器自动完成
「语义识别 → Geo-IR → 隐私计算算子选择 → JAX 生成 → SPU 模拟验证」；
**用户代码中不出现任何 JAX / SPU / MPC 调用**。

```python
from geo_privacy import geo

def check_conflict(route, no_fly_zone):
    return geo.intersects(route, no_fly_zone)
```

---

## 2. 交付物与完成度

| 模块 | 交付物 | 状态 |
|---|---|---|
| `frontend` | 业务 API → Geo-IR 的语义识别（含嵌套调用、链式类型、敏感度不降级） | 完成 |
| `ir` | `GeoProgram` / `GeoOperation` / `GeoValue` / `GeoType` / `GeoEntity` / `GeoRelation`；关系六元组（subject / predicate / object / time / spatial_scope / sensitivity） | 完成 |
| `planner` | Operator Registry + Privacy Planner，输出 `{operation, representation, backend, estimated_cost, security_level}`；MPC 协议按实测代价自动选择；位平面布局（D3）预测层；外部 PSI 执行档（PSI-CA / PI-Sum）的后端族标注（Phase 8） | 完成 |
| `backends/jax_backend` | `DistanceLE` / `WeightedSum` / `TemporalOverlap` 的 JAX 生成（可 `jax.jit` 追踪、无 Python 运行时依赖） | 完成 |
| `backends/spu_backend` | SPU 能力核查、模拟执行、通信量采集、打包前提与取槽单价探针；结果策略随 `op=` 在**执行前**解析并带出（Phase 10） | 完成 |
| `backends/psi_backend` | PSI 协议 ECDH(SM2) / KKRT / RR22(+`low_comm`) / DP / NPC 族接入与真机执行；NPC 与 DP 属"显式放行"档 | 完成 |
| `backends/psi_ca_backend` | PSI-Cardinality 计数档（OpenMined PSI，只出交集基数）：能力/API 核对、`--psi-count psi-ca` 接入、编译期拒绝契约 | 完成（真机已验证，2026-10-09 WSL2） |
| `backends/psi_sum_backend` | PI-Sum 交集内求和档（Google private-join-and-compute）：能力/flags 核对、`--psi-sum pjc` 接入、编译期拒绝清单；Phase 10 起带通信量（回环中继）与峰值内存（procfs VmHWM）计量（默认开启） | 完成（真机已验证，2026-10-09 WSL2） |
| `semantic` / `validator` | 关系三元组产出；六类失败模式的错误报告（错误位置 + 原因 + 建议替代算子 + 预计隐私计算代价） | 完成 |
| CLI | `geo-secure build`，八阶段输出 + 「算子-表征-后端-状态」表 | 完成 |

---

## 3. 里程碑时间线

| 日期 | 提交 | 内容 | 全量测试 |
|---|---|---|---|
| 2026-09-29 | `b1fde97` | Initial import：框架、6 个地理算子、Geo-IR、Planner、JAX 生成、SPU 模拟验证 | — |
| 2026-09-30 | `5daf60e` | RR22 接入闭环：协议可选、参数可传、真实执行可验证 | — |
| 2026-10-04 | `5daf60e` + 工作区改动 | 接入闭环第二阶段：配置闭环、真实码集输入、协议候选校验、Geo-RR22 预处理层 | `653 passed` |
| 2026-10-05 | `423946d` + 工作区改动 | P0/P1/P1.5：MPC 协议进入 planner、MPC 代价基线、重复实验 | `748 passed` |
| 2026-10-06 | `bd151ae` + 工作区改动 | P2-1：`WeightedSum` 定点 `scale` 改编译期常量，生成代码去除法 | `753 passed` |
| 2026-10-06 | `5a79c2c` + 工作区改动 | P2-2：通信量入基线（`capture_comm` / `--comm`） | `773 passed` |
| 2026-10-06 | `4716f4b` + 工作区改动 | P3：`TemporalOverlap` 第二套电路 `sweep` + 两套电路的成对 A/B | `793 passed` |
| 2026-10-06 | `b79da30` + 工作区改动 | P4：MPC 协议按实测代价 / 能力自动选择（`REF2K` 不再被自动选中） | `815 passed` |
| 2026-10-06 | `498e12f` + 工作区改动 | P5：位平面布局（D3）预测层（`planner/layout.py` + CLI `--layout-shape`） | `838 passed` |
| 2026-10-06 | `a811246` | P6：打包收益上界实测（SPU 按环元素而非输入位计费） | — |
| 2026-10-06 | `d91da69` | P7-P0：打包电路取槽步的单价实测（按位操作不免费） | **`893 passed`**（2026-10-08 复跑确认） |
| 2026-10-08 | `1dbc8c8` | P0 收尾：`dot` 读数根因定位（计费维度改按输出个数）+ 协议覆盖镜像修正 | `957 passed` |
| 2026-10-08 | `ec0b4ee` | P2/P3：`MpcProtocolSpec` 协议元数据 + 统一 capability validation（field / world_size / 语义 / 参数编译期前置拒绝） | **`1004 passed`** |
| 2026-10-09 | （工作区） | Phase 5 定界 + NPC 族**编译期显式放行**（`explicit_only`）+ E2E / benchmark 开关 | 待 WSL 复跑 |
| 2026-10-09 | （工作区） | 层面 3 起步：PSI-Cardinality 计数档接入（`--psi-count psi-ca` / `psi-ca-check` / 只接 `CellSetIntersect` + `REVEAL_COUNT` 的拒绝契约 / 测试桩 29 项） | 桩下全绿；全量 `912 passed`（Windows 离线环境） |
| 2026-10-09 | （工作区） | 管控层：复跑脚本**供应链固定**（上游钉 commit + bazelisk `v1.29.0` sha256 校验）+ 细粒度沙箱档草案（`docs/SANDBOX_AND_SUPPLY_CHAIN.md` / `scripts/codex_permissions.example.toml`） | 脚本已实测：sha256 正/反例、按 sha fetch 得 `950c5e4`；沙箱档已实测：本机 Windows 后端兑现不了（deny-glob 需 elevated、schannel TLS 不可用、白名单未见生效）——**暂不启用** |
| 2026-10-09 | （工作区） | 真机确认：SPU 模拟路径 `distance_check` / `risk_score` 全 `verified`（err=0.0）+ PSI-CA 计数档真机验证（`openmined-psi==2.0.6`，解除该档「未验证」） | 全量 `1094 passed`（0 failed / 0 skipped，WSL spu311）；PSI-CA 两文件 `29 passed` |
| 2026-10-10 | （本提交） | **Phase 4：Planner → Runtime 的 MPC 协议闭环**——执行期协议改为**逐步解析**（`_step_protocol`：显式指定 > 方案实测选择 > 登记默认值），修掉“两个 MPC 步骤各选不同协议时被统一成第一个协议”的静默换协议；新增 `_check_mpc_closure` 兜底核对与 9 项对拍测试；编译 JSON 的 plan 步带出 `mpc_protocol` / `mpc_protocol_basis` | **`1103 passed`**（WSL2 真机，0 跳过） |
| 2026-10-10 | （本提交） | **Phase 7：统一 benchmark metadata**——新增 `backends/benchmark_schema.py`（共有 13 字段 + 两族指标 + 缺口登记）与投影器 `scripts/unify_benchmark.py`，48 项测试；跨协议比较只比 `ok` 行、不同族/算子/规模直接报错、同用例重测取中位数（见 `docs/BENCHMARK_SCHEMA.md`） | **`1151 passed`**（WSL2 真机，0 跳过） |
| 2026-10-10 | （本提交） | **Phase 8：外部 PSI 执行档接入方案层**——`--psi-count psi-ca` / `--psi-sum pjc` 的计划步骤带出 `execution_backend`（`PSI-CA` / `PI-Sum`），计划表与最终状态表的 Backend 列不再写 `PSI`（与状态词 `count-only` / `count-and-sum` 同一口径）；登记表 `EXTERNAL_PSI_FAMILY_*` 与 backends 逐项交叉断言（21 项测试） | **`1172 passed`**（WSL2 真机，0 跳过） |
| 2026-10-10 | （本提交） | **Phase 9：统一 benchmark metadata 扩到四族**——`BENCHMARK_FAMILIES = PSI / PSI-CA / PI-SUM / MPC`；两条外部路径基线运行器（`benchmark_psi_ca.py` / `benchmark_psi_sum.py`）与真实基线产物入库（各 3 条、全部 ok）；缺口登记补三处（PI-Sum `peak_memory`、外部两档 `input_io_time` / `semantic_processing_time`）；统一层测试 48 → 64 项 | **`1188 passed`**（WSL2 真机，0 跳过） |
| 2026-10-10 | （本提交） | **Phase 10：结果策略与密码协议解耦（MPC 侧）+ PI-Sum 计量（通信量 / 内存）**——统一入口 `backends/result_policy.py`（PSI / MPC 共用，`psi_backend` 旧路径再导出兼容）；策略表扩到 5 种（新增 `REVEAL_VALUE`，`WeightedSum` 默认）与族泄漏码（PSI `intersection-body` / MPC `output-only`，不宣称零泄漏、不得默认广播）；`run_spu_simulation(op=...)` 任何执行前解析策略并随结果 / CLI 带出 `policy` / `reveals`；PI-Sum 回环 TCP 中继通信量 + procfs `VmHWM` 峰值内存（默认开启，`--no-measure-*` 可关），基线三条重跑入库 | **`1205 passed`**（WSL2 真机，0 失败 / 0 跳过） |
| 2026-10-10 | （本提交） | **Phase 11：外部两档计量补齐 + PSI-CA 依赖哈希固定**——①PSI-CA 通信量定口径并落地：量**协议消息的 protobuf 载荷**（`Request` / `ServerSetup` / `Response`，进程内链路 ⇒ 是网络字节的**下界**；产物带 `comm_meter="protobuf-payload"` 与方向），基线三条实测 107 524 / 430 084 / 1 720 324 B（严格线性：send 35 B/元素、recv 70 B/元素）；②PI-Sum 补 `input_io_time`（= `io_write_ms`，只计输入 CSV 落盘段，结果走 stdout 无读取段）；③`openmined-psi==2.0.6` + `protobuf==6.30.2` 按 `requirements-psi-ca.txt` **哈希固定**（`--require-hashes`，反例实测 rc=1），复跑脚本第 1/5 步强制核对版本；缺口收缩：PSI-CA 通信量、PI-Sum `input_io_time` 由「缺」转「有」 | **`1212 passed`**（WSL2 真机，0 失败 / 0 跳过） |

> 「+ 工作区改动」表示该轮结果记录于提交前后的工作区状态，逐轮明细见
> `docs/VERSION_COMPATIBILITY.md`。

---

## 4. 关键实测结论（可复算）

1. **协议选择由实测驱动，而非登记默认值**（产物 `docs/mpc_comm_baseline.json`，发送+接收，K=256）
   - `DistanceLE`：ABY3 4 200 B < SEMI2K 8 336 B < SECURENN 12 032 B < CHEETAH 2 415 124 B
   - `REF2K` 实测 send+recv 恒为 **0 B**（无密码学保护）→ 留在候选清单里但被排除在自动选择之外
   - `--protocol` 缺省从固定 `ABY3` 改为**自动**；排序数字全部取自既有产物（本轮未重跑基线）
2. **`TemporalOverlap` 两套计算方案的成对 A/B**（`docs/mpc_temporal_ab.json`）
   - 默认两两比较（二次）vs `sweep` 排序归并（O((N+M)·log(N+M))）
   - **交叉点在 K=128 与 256 之间**：K=256 时 `sweep` 上界 10 889 232 B < pairwise 下界 12 546 048 B（区间不重叠）
   - K=1024 时 `sweep` 真机跑通（单次探针 1.8 s / 43.7 MB），原先「K 上限被锁在 32 量级」的限制解除
3. **D3 位平面布局预测层复算课题交付物四个数**：196000 / 49000 / 192 / 1020.8×
   （`planner/layout.py`，与 `outputs/格网数据样例_明文与密态映射_v5.json` 逐项一致）
4. **打包前提实测**（`docs/mpc_packing_probe.json`）：SPU **按环元素计费**，
   16 B/元素，**与输入位宽无关**（int8 / int32 / int64 在 N=4096 上通信量之比 1.000）
5. **取槽步单价**（`docs/mpc_slot_cost_probe.json`）：秘密×公开常数 0 B、`&` 624 B/元素、
   `>>` 1888 B/元素、**全量取槽 1424 B/元素 = 纯乘法 16 B/元素的 89 倍**
   （2026-10-08 以 `--repeat 3` 重跑后为 1496 B/元素 / 94 倍；同一电路批间差 5%–15%）
6. **`WeightedSum` 定点 `scale` 改编译期常量**：`FM32 × K=256` 由崩溃转为可用；
   K=64…4096 由 43%–93% 偏差转为 **0/30 逐位一致**；通信量（ABY3/SEMI2K/SECURENN/CHEETAH，
   K=256）分别降 56% / 68% / 93% / 72%
7. **三份对拍**：每个算子建立「明文 / JAX / SPU」三份实现，**整数路径误差 0.0**，定点按明确容差
8. **PI-Sum 计量（Phase 10）**：通信量（回环中继逐字节，应用层字节含 gRPC 封装）
   随规模近似线性：2^8 → 2^12 合计 178 656 → 2 832 275 B；峰值 RSS 13.4 → 23.9 MB；
   耗时读数方差大（同机同规模 1.4 s–27.9 s），计量开 / 关差异被本机噪声淹没。
   产物 `docs/psi_sum_benchmark_baseline.json`（三条均 `agree=true`）

---

## 5. 已验证 / 未落地

**已验证（真机）**

- 6 个地理算子：`Intersects` / `Contains` / `DistanceLE` / `CellSetIntersect` / `WeightedSum` / `TemporalOverlap`
- MPC 协议：ABY3 / SEMI2K / SECURENN / CHEETAH（`REF2K` 可显式指定，不参与自动选择）
- MPC（SPU）协议闭环：方案**逐步**选什么、执行就**逐步**跑什么
  （`Compiler._step_protocol` + `_check_mpc_closure`）；构造出“两步不同协议”
  （SEMI2K / ABY3）后真机**各跑各的**，结果与明文逐位一致
  （`tests/test_mpc_execution_chain.py`，9 项）
- PSI 协议：ECDH(SM2) / KKRT / RR22（+ `low_comm`）/ DP-PSI（带噪语义单独披露）/ NPC 族（ECDH_NPC、KKRT_NPC，显式放行）
- 真机用例：SPU(MPC) 真跑 15 项、PSI 真机求交 23 项函数等（统计口径见 `README.md` §6.4）

**未落地（如实标注）**

- **打包电路本身**：只做到「预测层 + 前提实测」，JAX 生成器与 SPU 执行路径一行未改
  ——而本版 P0 把前置条件量清楚了，且**结论是不利的**：免逐槽提取的三条路线
  （整元素乘法 / 公开权重 / 不掩码移位）都不兑现，取槽单价仍在 1424–1496 B/元素
  （纯乘法的 89–94 倍），与 L2 的条数比 1020.8× 同量级。故打包电路**暂不实现**，
  先回课题侧核「L2 的条数比能否兑现」；
- ~~电路选择接到实际输入规模~~ ——**已闭合（本版 P1）**：
  `--layout-shape candidates=N` 现在同时喂给 `TemporalOverlap` 的电路选择；
- ~~Geo-RR22 候选集剪枝~~ ——**前置实证已出，结论为「当前口径下不可实现」**：
  `docs/GEO_RR22_COVERAGE.md`（国标码有前缀覆盖规则，21973 条真实码实测；
  本项目 64 位码是字段切分布局、解不出真实层级，且 `(X, Y)` 与 `L` 的口径未声明）。
  剪枝在没有无损规则前等于把近似引进精确路径，**不实现**；
- ~~PSI-Cardinality 真机执行~~ ——**已闭合（2026-10-09，WSL2）**：装上
  `openmined-psi==2.0.6` 后真机执行 `|A∩B|=2`、状态词 `count-only`、`agree=True`；
  快照 `docs/psi_ca_capability_report_wsl.json`，命令与输出见 `docs/PSI_CA_CAPABILITY.md` §8；
- 通用 GeoPandas / Shapely 源码自动转换（**明确不做**，只支持登记的业务 API）
- 三维示例的默认输入仍是二维码集（换码集即可）

---

## 6. 下一步（对齐月报下月计划）

1. ~~打包方案前置设计筛选~~ / ~~逐项底层运算真机核查 + 槽内归并实测~~ ——
   **已闭合（本版 P0）**：三条路线真机对拍全部不兑现；22 个原语逐项真机核验 22/22 跑通。
   其中 `dot` 那一读数**本轮已定位根因并更正**：它是**按输出个数计费**（1 个输出 = 16 B，
   收缩长度免费），不是"低于秘密乘法下限"——第一版拿输入元素数当尺子，读数没错、**尺子错了**。
   对账口径已按计费维度修正（`cost_driver` / `cost_per_unit` / `contraction_is_free`）。
   据实测**决定打包方案暂不落地**（收缩免费省不掉按元素计费的逐槽提取）；
   D3 升级为「通信量比」的条件（同规模成对实测）已写明，但前置结论不利，
   需先回课题侧核条数比。
2. **继续扩展协议接入面**（新 PSI / MPC 协议按同一口径接入），补齐协议覆盖镜像测试，
   做到「登记即有测试」。本版已把「登记即有测试」扩到**原语**层与**协议**层：
   `tests/test_primitive_probe.py` 断言探针表里声明的 jax/HLO 登记名必须真在
   capability 白名单里（`xor` / `top_k` 显式登记为「不在表里」）；
   `tests/test_protocol_coverage.py` 修掉一处**静默滑过**——`SPU_PROTOCOLS_VERIFIED`
   原先写成 `frozenset(SPU_PROTOCOLS)`（等于"新协议一登记就自动算已验证"），
   现改为显式字面量，并新增 `test_every_registered_protocol_has_a_real_execution_case`：
   从 `TestProtocolFieldSweep` 的 parametrize 源里**读出**被真机扫描的协议集合，
   断言它**恰好**等于 `SPU_PROTOCOLS`。
3. ~~收敛 CLI 与文档~~ ——**已闭合（本版 P1）**：四个探针各有一个 CLI 开关
   （`--packing-probe` / `--slot-cost-probe` / `--slot-reduction-probe` /
   `--primitive-probe`），产物文件名从单一张表派生、同时开两个探针直接报错、
   与 `--ops`/`--protocols` 组合直接拒绝；`--repeat` 从「只记录」变成**真的重复执行并报中位数**
   （此前 P7-P0 的读数其实是单次）。README 扩展点、能力矩阵、版本兼容表同步更新。
4. **协议扩展架构：Phase 2 / Phase 3 已闭合（本版 P2/P3）**——
   两族协议元数据（`PsiProtocolSpec.family` + `MpcProtocolSpec`：world_size /
   security_model（含出处）/ 语义 / `supported_fields` 实测矩阵）齐备；
   统一校验入口 `backends.protocol_validation.validate_protocol_request()`，
   `field` / `world_size` / 协议参数在编译期前置拒绝（自动选中协议同样复核，
   拒绝信息给出可操作的替代候选与放宽路径）。
   **Phase 5 的"新协议"对象已定界（本版）**：核查 `spu 0.9.5` 的 `PsiProtocol`
   枚举共 7 个真实协议，本项目**全部已登记**、6 个已真机执行；唯一"SPU 已支持、
   但编译器全链路未走完"的是 **NPC 族**（`ECDH_NPC` / `KKRT_NPC`）。本版把它收尾为
   **显式放行**档（`PsiProtocolSpec.explicit_only`；不进候选/建议清单，显式选择
   可通过编译期并带披露），并补齐编译器路径测试与 benchmark 显式开关。
   因此原先"选一个 SPU 已支持的**新**协议走完整接入流程"字面对象已不存在——
   扩展机制改由 NPC 族演练的后半段（编译期放行 + 基准 + E2E）验收。
   ~~Planner → Runtime 协议参数一致性对拍测试~~ ——**已闭合（本版 Phase 4）**：
   PSI 侧原有 `_check_config_closure` 逐字段对拍；MPC 侧补上**逐步协议解析**
   （`_step_protocol`：显式指定 > 方案实测选择 > 登记默认值——修掉“两个 MPC 步骤
   各选不同协议时被统一成第一个协议”的静默换协议）与 `_check_mpc_closure` 兜底
   核对（漂移即阶段报错）；对拍覆盖协议 / 环宽 / 参与方数量，并带反面用例；
   编译 JSON 的 plan 步带出 `mpc_protocol` / `mpc_protocol_basis`，四层可直接对拍。
   下一步（Phase 4 / 5 余项）：自动协议选择按 `world_size` **过滤候选**（当前策略是
   拒绝并给出替代，过滤属下一阶段）；`runtime_adapter` / `benchmark_profile`
   元数据位尚未登记（两族执行接口保持独立）；**另起"SPU 未实现过的新协议"工程**
   （层面 3：选一个开源 PSI/MPC 协议内核，按 `backends/` 新族接入，不改 SPU 源码）。
   **层面 3 已起步（本版）**：首个外部协议内核落地为 `backends/psi_ca_backend`
   （PSI-Cardinality 计数档）——与 libpsi 路径并列为第二条 PSI 路径，
   接入层（编译期契约 + 执行装配 + 测试桩）完成；**真机已验证（2026-10-09，WSL2）**：`|A∩B|=2`、状态词 `count-only`、`agree=True`。
   **层面 3 第二个内核（本版）**：`backends/psi_sum_backend`（PI-Sum 交集内求和，
   Google private-join-and-compute，Apache-2.0）——第三条 PSI 路径，泄漏承诺是
   「基数 + 交集内关联值之和」（登记码 `count+sum`）。接入层（能力核查跑上游
   `--help` 核 flag 形态、编译期拒绝清单、执行装配、测试桩）完成；上游需 Bazel
   构建且无官方 PyPI 包，**真机已复跑（2026-10-09，WSL2）**：Bazel 构建成功、PI-Sum 协议跑通，`result = (2, 13)` 与登记期望一致——命令见
   `docs/PSI_SUM_CAPABILITY.md` §8。

5. **~~统一 benchmark metadata~~ ——已闭合（本版 Phase 7）**：两族记录投影到同一张表
   （`backends/benchmark_schema.py`：共有 13 字段 + `PSIBenchmarkMetadata` /
   `MPCBenchmarkMetadata` + `metadata.raw` 保真）；§九 清单 21 项逐条登记去向，
   某族确实没采集的指标登记成缺口并由测试在真实产物上锁死；跨协议比较只比 `ok` 行、
   不同族/算子/规模直接报错、同用例重测取中位数并带出样本数与极差。
   见 `docs/BENCHMARK_SCHEMA.md`。
   **Phase 8 前半已闭合（本提交）**：外部 PSI 执行档接入方案层——计划步骤带出
   `execution_backend`（`PSI-CA` / `PI-Sum`）与该档强制的结果策略、协议泄漏面
   （`planner/registry.py` 的 `EXTERNAL_PSI_FAMILY_*`，与两个 backends 逐项交叉断言）。
   **Phase 9 已闭合（本提交）**：`benchmark_schema` 并入两条外部路径——族扩为
   `PSI / PSI-CA / PI-SUM / MPC` 四族，新增两套 metadata / 适配器 / 族推断与
   缺口登记（PI-Sum `peak_memory`、外部两档 `input_io_time` /
   `semantic_processing_time`）；两条路径各配一个基线运行器
   （`tests/benchmarks/benchmark_psi_ca.py` / `benchmark_psi_sum.py`），
   真机基线产物入库（各 3 条、全部 ok），统一层测试 48 → 64 项。
   **Phase 10 已闭合**：MPC 侧的策略↔协议解耦（统一入口
   `backends/result_policy.py` + `REVEAL_VALUE` + 泄漏码 `output-only`）、PI-Sum 的
   通信量计量（回环 TCP 中继）与子进程峰值内存（procfs `VmHWM` 探针）。
   **Phase 11 已闭合（本提交）**：PSI-CA 通信量（协议消息载荷，进程内链路 ⇒
   下界，读数见 `docs/PSI_CA_CAPABILITY.md` §7.1）与 PI-Sum 输入 I/O
   （`io_write_ms`，见 `docs/PSI_SUM_CAPABILITY.md` §7.2）两处缺口补齐；
   PSI-CA 依赖按 `requirements-psi-ca.txt` 哈希固定（含正反例实测）。
   **仍开放**：`REVEAL_TO_REGULATOR`（部署期模式，两族同判显式拒绝）、
   libpsi（求交档）的通信量计量（见 `docs/BENCHMARK_SCHEMA.md` §8）。

---

## 7. 复现

```bash
git clone git@github.com:ThresholdCrypto/GIS_SPU.git && cd GIS_SPU
pip install -r requirements-spu.txt   # spu==0.9.5 / jax<=0.4.34 / numpy<2
python -m pytest tests/ -q            # 1212 项全部通过（0 跳过）
python -m geosecure.cli build examples/distance_check.py
```

WSL2 / Linux 上一键脚本：`bash scripts/setup_wsl_spu.sh`。
外部 PSI 执行档（PSI-CA / PI-Sum）在方案层的标注复核：
`bash scripts/verify_external_family_plan.sh`（三种档位的最终状态表 + 全量回归）。
外部两档基准基线重算：`bash scripts/verify_external_baselines_wsl.sh`
（PSI-CA + PI-Sum 真机基线 + 统一层测试；PI-Sum 需 `GIS_SPU_PJC_BIN_DIR`）。

---

**相关文档**：`README.md`（架构 / 算子 / Geo-IR / Planner / 能力矩阵）、
`docs/VERSION_COMPATIBILITY.md`（版本兼容与逐轮记录）、
`docs/MPC_BENCHMARK_PROTOCOL.md`（MPC 基线规程）、`docs/BITPLANE_LAYOUT.md`（D3 布局）、
`docs/SPU_CAPABILITY.md` / `docs/PSI_CAPABILITY.md`（能力核查）。
