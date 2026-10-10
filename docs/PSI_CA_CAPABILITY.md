# PSI-Cardinality（计数档）：能力、边界与复跑

> 代码：`backends/psi_ca_backend/`；CLI：`geo-secure build … --psi-count psi-ca`、
> `geo-secure psi-ca-check`；选型依据：`outputs/新协议选型报告.md`。

## 1. 定位：第二条 PSI 路径（与 libpsi 求交并列）

| | libpsi 求交（默认） | PSI-Cardinality（`--psi-count psi-ca`） |
|---|---|---|
| 实现 | 官方 `spu.psi.psi_execute`（libpsi） | **OpenMined PSI**（`openmined-psi==2.0.6`，Apache-2.0） |
| 产出 | 交集本体；`Intersects` / `Contains` 由它派生布尔 | **只有交集基数**（一个整数） |
| 协议内部泄漏 | `intersection-body`：接收方拿到交集本体 | `count-only`：交集本体不交给任一方 |
| 接口形态 | CSV 文件（落盘） | 进程内 protobuf 消息（不落盘、无临时文件） |
| 承接算子 | `Intersects` / `Contains` / `CellSetIntersect` | **只承接 `CellSetIntersect` 的 `REVEAL_COUNT`** |
| 触发 | 缺省 | `--psi-count psi-ca` |
| 规划层后端名 | `PSI` | `PSI-CA`（Phase 8 起；计划表与最终状态表同口径） |

两条路径**刻意不合并**：泄漏承诺不同。若把计数档并进 libpsi 的 `REVEAL_COUNT`
策略，会把"业务层收窄"误读成"协议层不再交交集本体"——后者才是本档的真实差别。

## 2. 上游依据（2026-10-09 从源码核对，非推测）

- `private_set_intersection/python/__init__.py`：
  `client` / `server` 子入口；`CreateWithNewKey(reveal_intersection)`；
  `server.CreateSetupMessage(fpr, num_client_inputs, inputs, ds)`；
  `client.CreateRequest` / `server.ProcessRequest` / `client.GetIntersectionSize`。
- `private_set_intersection/cpp/psi_server.cpp`：
  - `fpr` 对 `Raw` 数据结构**被忽略**（源码原文注释）；
  - `ProcessRequest` 拒绝两侧 `reveal_intersection` 不一致的请求——本项目两侧
    都以 `False`（不要交集本体）创建。
- 发行版：`openmined-psi==2.0.6`（PyPI），依赖 `protobuf==6.30.2`；
  轮子只有 macOS 与 manylinux（cp39–cp313），**Windows 无轮子**。
- **供应链固定（Phase 11）**：依赖按 `requirements-psi-ca.txt` **哈希固定**安装
  （`--require-hashes`：拿不到钉死的产物、或哈希对不上，pip 直接失败）；钉死的
  wheel sha256 与实测正反例见 §8.1。

本项目不预设 API：`backends/psi_ca_backend/capability.py` 在运行期核对模块属性与
方法清单；形态不符时 `runnable=False` 并列出缺失项，不做"猜 API"的执行。

## 3. 角色映射（必须知情）

OpenMined PSI 是**不对称**协议：client 发起查询、server 持库，**计数由 client 侧得到**。
本后端固定：

```
client = 左侧输入（对应 libpsi 的 receiver_rank=0 口径；获得计数）
server = 右侧输入
```

威胁模型与 libpsi 路径一致：半诚实（semi-honest）两方，见 `docs/ADVERSARY_MODEL.md`。
补充知情项：**计数本身对 client 可见**——小集合场景下，计数可能逼近交集的信息量，
"只出计数"不等于"无信息可得"。

## 4. 精确与近似（只接精确档）

- `RAW`（本档唯一）：**精确计数**，`result_semantics = exact`；
- `GCS` / `BLOOM_FILTER`：近似档（上游口径偏差可达 ±10% 量级），**未接入**——
  近似结果不能与明文计数对拍，接进来只会混淆"验证"的含义。

`CreateSetupMessage` 的 fpr 固定传 `0.0` 且**不提供旋钮**：RAW 档忽略该参数，
提供旋钮会制造"已配置精度"的错觉。

## 5. 泄漏面登记（与 libpsi 并列、互不相同）

- 协议内部泄漏：`PSI_CA_PROTOCOL_LEAK = "count-only"`（libpsi 是 `intersection-body`）；
- 业务层暴露：`REVEAL_COUNT`（`result_policy.policy`）；
- 整句披露：`result_policy.leak_disclosure` 由 CLI 直接打印——libpsi 档那句
  「接收方仍获得交集本体」在本档是**错的**，必须替换；
- 逐算子登记：`PSI_CA_OP_LEAKS`（当前只有 `CellSetIntersect` 一项）。

## 6. 编译期契约：拒绝清单

`--psi-count psi-ca` 是**整档切换**——不允许"一步走计数、另一步走求交"的混合语义：

| 情形 | 结论 | 建议（拒绝报告里逐条给出） |
|---|---|---|
| 算子非 `CellSetIntersect`（如 `Intersects`） | 编译期 error | 改用 `CellSetIntersect` 取基数；或去掉开关走 libpsi |
| 计数输出被下游 PSI 输入消费 | 编译期 error | 拆分为独立求交；或走 libpsi 链式路径 |
| 两方布局清单不一致 | 进入协议前 `LAYOUT_MISMATCH` 拒绝 | 对齐 `ir.grid_code_layout_manifest()` |
| 环境缺 `openmined-psi` / API 漂移 | 阶段 warning，**计数留空**（不给推测值） | `pip install openmined-psi==2.0.6`（Linux/macOS/WSL2） |

拒绝报告包含：错误位置（算子 + 第几步）、原因、建议替代、预计隐私计算代价（b/d/R）。
报告不修改用户代码；同一份源码去掉 `--psi-count psi-ca` 即可回到缺省求交路径。

## 7. 已验证 / 未验证（诚实登记）

- **已验证（测试桩 + Windows 离线环境）**：能力探测（缺包 / API 漂移 / 版本偏差
  只提示不阻断）、`RAW` 调用序列（两侧 `reveal_intersection=False`、fpr=0.0、
  计数取回）、排序去重、空输入不启协议、全部拒绝路径、泄漏登记一致性、
  CLI 全链路（`tests/test_psi_ca_backend.py` / `tests/test_cli_psi_ca.py`）。
- **已验证（真机，2026-10-09，WSL2 + `openmined-psi==2.0.6`）**：
  `psi-ca-check` → `installed=true` / `version=2.0.6` / `runnable=true`，
  `blockers` 与 `api_problems` 均为空；`build examples/conflict_count.py --psi-count psi-ca`
  8 阶段全绿（第 4/6 阶段按设计 WARNING / SKIPPED），PSI 仿真真实执行：
  `|A|=3  |A∩B|=2`、`result=2`、`reference=(3076832715348377604, 3076973452836732932)`、
  `agree=True`、`result-sems=exact`、`policy=REVEAL_COUNT`，状态词 `count-only`；
  独立最小验证 `GetIntersectionSize=2`；`tests/test_psi_ca_backend.py` +
  `tests/test_cli_psi_ca.py` 29 项通过（真机分支：`runnable=true` ⇒ 断言
  `status=ok` / `value=2` / `agree=True` / `count-only`）。
- **规划层标注（Phase 8，2026-10-10）**：开档后计划表与最终状态表的 Backend 列
  为 `PSI-CA`——此前两处都写 `PSI`，与状态词 `count-only` 自相矛盾。步骤的
  `backend` 保留登记值、实际执行族存 `execution_backend`，档强制的 `REVEAL_COUNT`
  与泄漏码 `count-only` 随 `reasons` 留痕；由 `tests/test_external_psi_planning.py`
  锁定（含环境缺失时 Backend 列仍为 `PSI-CA` 的分支）。
- **通信量计量（Phase 11，2026-10-10，真机 WSL2）**：量**协议消息的
  protobuf 载荷**（口径与依据见 §7.1）。三条基线全部 `ok`、`agree=true` 保持：
  载荷合计 107 524 / 430 084 / 1 720 324 B（N=2^10 / 2^12 / 2^14），
  与元素数严格线性（send 35 B/元素、recv 70 B/元素）。
- **仍未验证**：`GCS` / `BloomFilter` 近似档（未接）、跨机部署（当前是进程内链路）、
  近似档的 fpr 语义（RAW 档上游注明忽略 fpr）。

### 7.1 通信量口径与读数（Phase 11，2026-10-10）

**口径：量「协议消息的 protobuf 载荷」，不是网络观测。**

本档是**进程内链路**——client / server 对象同进程、没有 socket，可观测的网络
为零；而真实部署里过网的就是这三条消息。所以量它们的序列化长度
（上游消息对象都有 `SerializeToString()`）：

| 方向 | 由哪些消息构成 | 依据 |
|---|---|---|
| `send_bytes`（client→server） | `Request` | 上游 proto 注释：Request「sent to the server」 |
| `recv_bytes`（server→client） | `ServerSetup` + `Response` | Response「sent back to client」；`ServerSetup` 只由 `server.CreateSetupMessage` 产出、只由 `client.GetIntersectionSize` 消费，必须传到 client |

**不含**任何传输封装（gRPC / HTTP2 / TLS）与 TCP/IP 头——真实部署的网络字节
**≥** 这里报的数：这是一条**下界**，不是等号。计量名随数走：产物里
`comm_meter = "protobuf-payload"`、`comm_direction` 写明方向；上游 API 漂移
（消息对象没有 `SerializeToString`）时**留空 + 注明**，不填 0、不推测。

实测（WSL2 + `openmined-psi==2.0.6`，产物 `docs/psi_ca_benchmark_baseline.json`）：

| 规模 | send | recv | 合计 | 每元素（send / recv） |
|---|---:|---:|---:|---|
| N=2^10 | 35 840 B | 71 684 B | 107 524 B | 35.0 / 70.0 B |
| N=2^12 | 143 360 B | 286 724 B | 430 084 B | 35.0 / 70.0 B |
| N=2^14 | 573 440 B | 1 146 884 B | 1 720 324 B | 35.0 / 70.0 B |

逐条消息（另跑 N=1024 / 4096 取证，RAW 档一条元素一个条目）：`Request = 35n`、
`Response = 35n`、`ServerSetup = 35n + 4`，即 `send = 35n`、`recv = 70n + 4`
——三档读数与公式完全吻合。

与另外两条 PSI 路径的口径差别（别混着比）：

| 路径 | 量的东西 | 含传输封装？ |
|---|---|---|
| PSI（libpsi 求交） | **未采集**（需 SDK 计量钩子，仍开放） | — |
| PSI-CA（本档） | 协议消息 protobuf 载荷 | 否（进程内链路 ⇒ 下界） |
| PI-Sum | 回环 TCP 中继逐字节（应用层） | 含 gRPC / HTTP2，不含 TCP/IP 头 |

## 8. WSL2 / Linux 复跑（已于 2026-10-09 真机执行）

```bash
# 1) 安装（Linux / WSL2；**哈希固定**：版本 + wheel sha256，拿不到钉死的产物
#    或哈希对不上，pip 直接失败——不会静默换个产物装上）
/opt/miniconda3/envs/spu311/bin/python -m pip install -r requirements-psi-ca.txt

# 2) 环境核查（期望 runnable=true、installed=true）
/opt/miniconda3/envs/spu311/bin/python -m geosecure.cli psi-ca-check

# 3) 端到端（样例兜底输入；期望 |A∩B|=2、状态词 count-only）
/opt/miniconda3/envs/spu311/bin/python -m geosecure.cli build examples/conflict_count.py --psi-count psi-ca

# 4) 测试（真机下 TestCountModeExecution 走 "ok + count-only" 分支）
/opt/miniconda3/envs/spu311/bin/python -m pytest tests/test_psi_ca_backend.py tests/test_cli_psi_ca.py -q

# 5) 基准基线（可选；期望 3 条 ok、agree=true，产物写 docs/psi_ca_benchmark_baseline.json）
/opt/miniconda3/envs/spu311/bin/python tests/benchmarks/benchmark_psi_ca.py
```

**实测（2026-10-09，WSL2 Ubuntu + Python 3.11.16）**
- 安装落点：`/opt/miniconda3/envs/spu311` 的 site-packages 属 root，pip 自动走
  **user site**（`~/.local/lib/python3.11/site-packages`，`ENABLE_USER_SITE=True`），
  导入正常；同时带入 `protobuf==6.30.2`（该环境原本没有 protobuf，spu 也不依赖它，无冲突）。
- `psi-ca-check`：`installed=true` / `version=2.0.6` / `runnable=true`，
  `blockers=[]`、`api_problems=[]`；快照 `docs/psi_ca_capability_report_wsl.json`。
- `build examples/conflict_count.py --psi-count psi-ca`：8 阶段全绿（第 4/6 阶段按设计
  WARNING / SKIPPED），PSI 仿真关键行 `|A|=3  |A∩B|=2`、`result : 2`、
  `reference : (3076832715348377604, 3076973452836732932)   agree=True`、
  `result-sems: exact`、`policy : REVEAL_COUNT`，最终表
  `CellSetIntersect | CompactCellSet | PSI | count-only`。
- 测试：两个文件 `29 passed`；同环境全量 `python -m pytest tests/ -q` →
  **`1094 passed`（0 failed / 0 skipped）**。

最小独立验证（不依赖本仓库装配）：

```python
import private_set_intersection.python as psi

client = psi.client.CreateWithNewKey(False)
server = psi.server.CreateWithNewKey(False)
setup = server.CreateSetupMessage(0.0, 3, ["1", "2", "3", "4"], psi.DataStructure.RAW)
request = client.CreateRequest(["2", "3", "9"])
response = server.ProcessRequest(request)
print(client.GetIntersectionSize(setup, response))   # 期望 2
```

### 8.1 依赖哈希固定（Phase 11，2026-10-10 实测）

`requirements-psi-ca.txt` 用 `--require-hashes` 把本档依赖钉到**产物**级：版本号
没变、产物被换，同样装不上（不是"提示一下"）。哈希取自 PyPI 官方 JSON API 的
`digests.sha256`（来源 URL 写在 requirements 文件头）。

| 包 | 版本 | 轮子 | sha256 |
|---|---|---|---|
| openmined-psi | 2.0.6 | cp311 manylinux_2_39 x86_64 | `346b44c634aa487cc4a7928d39995382a582f832cb3a2b5e10efaf278ac4279b` |
| openmined-psi | 2.0.6 | cp311 manylinux_2_35 x86_64 | `2981aa1b02358d996e6874cfb76e9b833c52f16cc832d3917b9333c8b0bc985e` |
| openmined-psi | 2.0.6 | cp310 manylinux_2_39 x86_64 | `57992c9f968f9123f34d5d9504a322b8e6449dddac91e17b2a7cad7eedfd7efa` |
| openmined-psi | 2.0.6 | cp310 manylinux_2_35 x86_64 | `ae39769a37b997868894c432a9b1a84ba46d7f3277078a5927902753344baf06` |
| protobuf | 6.30.2 | cp39-abi3-manylinux2014_x86_64 | `4f6c687ae8efae6cf6093389a596548214467778146b7245e886f35e1485315d` |
| protobuf | 6.30.2 | py3-none-any | `ae86b030e69a98e08c77beab574cbcb9fff6d031d57209f574a5aea1445f4b51` |

同一版本列了多份轮子：pip 只会在这几份**已核过**的产物里按平台挑（本仓库 WSL2
实际命中 `manylinux_2_39`），挑不到就失败——不会放宽成"不校验"。

实测证据（WSL2，pip 26.2.1）：

- 正例 `pip install -r requirements-psi-ca.txt`：两条都 `Requirement already satisfied`，
  版本与钉死值一致；
- 正例（强制重解析、只下载不安装）：`--dry-run --force-reinstall --no-cache-dir`
  → 下载 `openmined_psi-2.0.6-cp311-cp311-manylinux_2_39_x86_64.whl` 与
  `protobuf-6.30.2-cp39-abi3-manylinux2014_x86_64.whl`，哈希校验通过；
- 反例（把 openmined-psi 的 4 个哈希全改掉）：pip 报
  `THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE`、`rc=1`，
  并回显实际值 `346b44c6…4279b`——与上表逐字一致（顺手复核了 pin 的真实性）。

复跑脚本已强制这一步：`scripts/verify_external_baselines_wsl.sh` 的 **1/5**
核对「本环境装的版本 == 钉死版本」，不一致直接停（不给"用了别的版本还能出基线"的口子）。
