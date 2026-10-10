# PI-Sum（交集内求和）：能力、边界与复跑

> 代码：`backends/psi_sum_backend/`；CLI：`geo-secure build … --psi-sum pjc`、
> `geo-secure psi-sum-check`；选型依据：`outputs/新协议选型报告.md`（备选 A）。
> 核对日期：2026-10-09；核对对象：上游 `master@950c5e4c`（`2026-03-09`）。

## 1. 定位：第三条 PSI 路径（与 libpsi 求交、PSI-CA 计数并列）

| | libpsi 求交（默认） | PSI-CA（`--psi-count psi-ca`） | PI-Sum（`--psi-sum pjc`） |
|---|---|---|---|
| 实现 | 官方 `spu.psi.psi_execute` | OpenMined PSI | **Google `private-join-and-compute`** |
| 接入对象 | SPU 自带 `libpsi` | PyPI 轮子（`openmined-psi==2.0.6`） | **Bazel 构建的两个可执行文件** |
| 产出 | 交集本体 | **只有交集基数**（一个整数） | **基数 + 交集内关联值之和**（两个数） |
| 协议内部泄漏 | `intersection-body` | `count-only` | `count+sum` |
| 承接算子 | `Intersects` / `Contains` / `CellSetIntersect` | `CellSetIntersect` + `REVEAL_COUNT` | `CellSetIntersect` + `REVEAL_INTERSECTION_SUM` |
| 输入形态 | CSV 文件 | 进程内 protobuf | CSV 文件（本机回环 gRPC 传消息） |
| 触发 | 缺省 | `--psi-count psi-ca` | `--psi-sum pjc`（另需关联值） |
| 规划层后端名 | `PSI` | `PSI-CA` | `PI-Sum`（Phase 8 起；计划表与最终状态表同口径） |

三条路径**刻意不合并**：泄漏承诺互不相同。把求和档并进 libpsi 或 PSI-CA，
都会把"协议层多交了一个和"这件事抹平成"又一个输出开关"——而多出来的那个和
正是本档的用途，也是本档的新风险面。

## 2. 上游依据（2026-10-09 从源码逐字核对，非推测）

- `README.md`：功能定义（server 持标识符；client 持标识符 + 非负整数值；
  **client** 得知交集大小与交集内值之和）、honest-but-curious 安全模型、
  泄漏告警（唯一值或过小交集可由 intersection-sum 反推成员）。
- `private_join_and_compute/client.cc`：flag `--client_data_file` / `--port`
  （默认 `0.0.0.0:10501`）/ `--paillier_modulus_size`（默认 `1536`）；失败返回 1。
- `private_join_and_compute/server.cc`：flag `--server_data_file` / `--port`；
  就绪语句 `Server: listening on <port>`。
- `private_join_and_compute/client_impl.cc`：结果行原文
  `Client: The intersection size is <N> and the intersection-sum is <S>`；
  和值经 `ToIntValue()` 取出。
- `private_join_and_compute/data_util.cc`：server CSV 每行**恰好 1 列**；
  client CSV 每行**恰好 2 列**且第 2 列为**非负 int64**（负数直接报错）；无表头。
- `server.cc` / `client.cc`：两侧都用 gRPC `LocalCredentials(LOCAL_TCP)`
  ——**仅本机回环，无 TLS、无身份认证**。
- 许可：Apache-2.0；构建工具链 `.bazelversion` = `8.0.1`。

**上游没有官方 PyPI 包。** PyPI 上的同名包 `private-join-and-compute`（0.0.1，
17 KB 纯 Python 轮子，无作者无主页）与 `pjc`（无关的脚手架工具）**都不是**
Google 的实现，不得用来顶替——能力核查的 `notes` 里带这句点名。

本项目不预设 flag 形态：`capability.py` 在运行期跑 `<binary> --help` 逐项核对
flag 清单；缺失即 `runnable=False` 并列出缺失项，不按未核对的形态执行。

## 3. 角色映射（必须知情）

上游协议**不对称**：server 持标识符，client 持标识符 + 关联值，
**结果（基数与和）只由 client 得知**。本后端固定：

```
client = 左侧输入（带关联值 left_values，获得结果；对应 libpsi 的 receiver_rank=0）
server = 右侧输入（只持标识符）
```

威胁模型与另两条 PSI 路径一致：半诚实（semi-honest）两方，见
`docs/ADVERSARY_MODEL.md`。补充知情项：**和值本身可反推成员**——上游 README
自陈"若某标识符的值特别大，或交集过小，可由 intersection-sum 推断出它是否在
交集里"，并明言其缓解措施（加噪、剪除离群值、过小交集中止）**本开源库未实现**。

**传输面**：两侧都是 gRPC `LocalCredentials(LOCAL_TCP)`，只能同机执行；
本项目把监听地址从上游默认的 `0.0.0.0:10501` 收紧到 `127.0.0.1:10501`。
跨机部署须自行加通道保护（上游 README 同样提示"非本地运行请考虑 SSL"）。

## 4. 精确与范围（只接精确档）

- 结果语义 `exact`：Paillier 同态求和无噪声，基数与和都是精确值；
- 关联值必须落在 `[0, 2**63-1]`（上游按**非负 int64** 解析，负数直接报错）；
- 和值须落在 int64 内（上游 `ToIntValue`），超界由上游报错、本项目原样转述；
- **缺值不按 0 补齐**：关联值覆盖面不完整时直接拒绝执行。补齐会把"数据没接上"
  伪装成"和为 0"——那是假结果，不是容差。

## 5. 泄漏面登记（三条路径并列、互不相同）

- 协议内部泄漏：`PSI_SUM_PROTOCOL_LEAK = "count+sum"`
  （libpsi 是 `intersection-body`，PSI-CA 是 `count-only`）；
- 业务层暴露：`REVEAL_INTERSECTION_SUM`（`result_policy.policy`）；
- 整句披露：`result_policy.leak_disclosure` 由 CLI 直接打印——libpsi 那句
  「接收方仍获得交集本体」在本档是**错的**，必须替换；
- 逐算子登记：`PSI_SUM_OP_LEAKS`（当前只有 `CellSetIntersect` 一项）。

结果策略名 `REVEAL_INTERSECTION_SUM` 登记在**本后端**（不进统一策略模块
`backends/result_policy.py`）：统一模块的 PSI 族 `protocol_leak` 对所有策略
都固定为 `intersection-body`，把本档的策略塞进去会把两栏说成同一件事
（本档的泄漏码是 `count+sum`）。

## 6. 编译期契约：拒绝清单

`--psi-sum pjc` 是**整档切换**——不允许"一步求和、另一步求交"的混合语义：

| 情形 | 结论 | 建议（拒绝报告里逐条给出） |
|---|---|---|
| 算子非 `CellSetIntersect`（如 `Intersects`） | 编译期 error | 改用 `CellSetIntersect` 取基数 + 和；只需基数走 `--psi-count psi-ca`；要交集本体走 libpsi |
| 求和输出被下游 PSI 输入消费 | 编译期 error | 拆分为独立求交；或去掉开关走 libpsi |
| 左侧输入没有关联值 | 编译期 error | 用 `psi_sum_weights`（CLI：`--psi-sum-weights 输入名=PATH`）声明 |
| 关联值覆盖不全（输入已绑定时可比对） | 编译期 error | 补齐覆盖，或改用只出基数的 PSI-CA |
| 与 `--psi-count psi-ca` 同时启用 | 构造期 `ValueError` | 二选一 |
| 关联值形态非法（非 Mapping） | 构造期 `ValueError` | 修成 `{码: 非负整数}` |

每条拒绝都带：**错误位置**（算子 + 步序）、**原因**、**建议替代算子**、
**预计隐私计算代价**。编译期违例一律 `error`，不降级成 warning——否则
"这一档跑不了这个程序"会被读成"环境问题、装好就能跑"。

状态词是 `count-and-sum`（不是 `verified`：产出是两个数，没有交集本体可比对；
也不是 `count-only`：本档多交了一个和）。

## 7. 诚实登记：已验证 / 未验证

1. **真机已复跑（2026-10-09，WSL2 Ubuntu + Python 3.11.16）**：
   `bazel build //private_join_and_compute:all` 成功（Bazel 8.0.1，与登记的上游
   `.bazelversion` 一致），两个二进制启动、协议跑通，**结果行解析与登记原文一致**：
   `|A|=3  |A∩B|=2`、`result = (2, 13)`、
   `reference = (3076832715348377604, 3076973452836732932)`、`agree=True`、
   `result-sems = exact`、client 退出码 0；能力核查 `runnable=true`，实际使用
   `paillier_modulus_size=1536`（与上游默认一致）。命令见 §8，能力核查原文见
   `docs/psi_sum_capability_report_wsl.json`。
   **2026-10-10 从零复跑复核**（WSL 重装后 /tmp 被清空）：clone → Bazel 构建
   （5,683 个 action）→ `psi-sum-check` → `build examples/intersection_sum.py`
   全链重跑，结果与上一致（`(2, 13)`、`agree=True`）；同日产出性能基线
   `docs/psi_sum_benchmark_baseline.json`（N=2^8 / 2^10 / 2^12，三条均 `agree=true`）。
   注意耗时读数的**重复性有限**：同机多轮复核里 2^8 档从 1.4 s 到 22.6 s 都出现过
   （机器负载与 Paillier 固定开销主导，规模不是唯一因素）——引用读数时带上轮次
   与机器上下文，别当容量规划的绝对值。
   **2026-10-10 补充（Phase 10）**：进程内存与通信量两条计量已接入运行器
   （默认开启）——峰值内存用 procfs `VmHWM` 采样探针（client / server 各一条，
   取较大者），通信量用回环 TCP 中继逐字节计数；三条基线读数见 §7.1。
   **仍未实测**：跨机部署（两侧 `LocalCredentials(LOCAL_TCP)` 只允许同机，
   跨机须自行加通道保护）。

### 7.1 Phase 10 计量读数（2026-10-10，WSL2，基线三条）

| 规模 | send（client→server） | recv（server→client） | 合计 | peak RSS（取 client / server 较大者） |
|---|---:|---:|---:|---:|
| N=2^8 | 168 183 B | 10 473 B | 178 656 B | 13.363 MB（client 13.137 / server 13.363） |
| N=2^10 | 670 490 B | 38 892 B | 709 382 B | 15.230 MB（client 14.809 / server 15.230） |
| N=2^12 | 2 679 694 B | 152 581 B | 2 832 275 B | 23.934 MB（client 21.773 / server 23.934） |

口径（与产物字段逐字对应）：通信量是**应用层字节**（含 gRPC / HTTP2 封装，
不含 TCP/IP 头），方向 `send=client→server；recv=server→client`；中继多一跳，
`timings_ms` 含该跳开销（A/B 实测：计量开 / 关的耗时差被本机噪声淹没——耗时
本身重复性有限，同机 1.4 s–27.9 s 都出现过，见上面第 1 条）。峰值内存是内核
`VmHWM` 水位，采样间隔 20 ms（最后一次采样到退出的窗口 ≤ 20 ms 记不到）；
两个子进程并发采样，`peak_rss_mb` 取较大者、不是合计。三条读数与
`agreement=true` 一同入库 `docs/psi_sum_benchmark_baseline.json`。

2. **接入层已验证**：`tests/test_psi_sum_backend.py`（32 项）与
   `tests/test_cli_psi_sum.py`（20 项）在**测试桩**下全绿。桩替换的是
   runtime 的"唯一进程启动点"`spawn_pjc` 与 capability 的 `probe_binary_flags`，
   并按真实输入 CSV 做明文复算后打印上游格式的结果行——它证明的是
   编译期契约、CSV 落盘形态、角色映射、结果解析与折算口径，
   **不证明上游密码学语义**。
3. **未接的档**：上游没有"只出和不出基数"或"带噪和"的开关，本项目也不提供
   （不制造"已配置精度"的错觉）。
4. **规划层标注（Phase 8，2026-10-10）**：开档后计划表与最终状态表的 Backend 列
   为 `PI-Sum`（此前写 `PSI`）；步骤带出 `execution_backend`，档强制的
   `REVEAL_INTERSECTION_SUM` 与泄漏码 `count+sum` 随 `reasons` 留痕；由
   `tests/test_external_psi_planning.py` 锁定。上游产物未构建时，环境缺失如实报
   `backend-direct` 且求和栏位留空，**不改后端名**。

## 8. Linux / WSL 复跑命令（已于 2026-10-09 真机执行）

> 这串步骤已整理成脚本：`bash scripts/verify_pi_sum_wsl.sh`
> （自动按上游 `.bazelversion` 取 Bazel 版本、clone 失败回退 SSH、对 `(2, 13)`
> 做断言；日志落在 `/tmp/pjc_setup/`）。
> 若本机需经代理出网（直连 443 不通），先 `export https_proxy=http://127.0.0.1:<端口>`——
> clone 与 Bazel 拉依赖都读 `http_proxy`/`https_proxy`。
> 供应链管控（2026-10-09 起）：脚本把上游钉到 commit `950c5e4c…`、把 bazelisk
> 钉到 `v1.29.0` 并校验 sha256（不再用 `releases/latest`）；
> 依据与实测见 `docs/SANDBOX_AND_SUPPLY_CHAIN.md` §1。

```bash
git init /tmp/pjc && git -C /tmp/pjc remote add origin \
    https://github.com/google/private-join-and-compute.git
git -C /tmp/pjc fetch --depth 1 origin 950c5e4c88d7effe85147beb7856152f7c53394b
git -C /tmp/pjc checkout FETCH_HEAD   # 钉死的 commit（见 §1 供应链固定）
cd /tmp/pjc && bazel build //private_join_and_compute:all
export GIS_SPU_PJC_BIN_DIR=/tmp/pjc/bazel-bin/private_join_and_compute

cd /mnt/c/Users/DELL/Documents/Codex/2026-09-20/geosot-3d-dqg-4d-c-users-2/GIS_SPU
geo-secure psi-sum-check                       # 能力核查：应 runnable=true
geo-secure build examples/intersection_sum.py --psi-sum pjc \
    --psi-sum-weights route=examples/route_risk_weights.csv

# 5) 基准基线（可选；期望 3 条 ok、agree=true，产物写 docs/psi_sum_benchmark_baseline.json）
#    默认开通信量 + 内存计量；--no-measure-comm / --no-measure-memory 可关（关后留空，不填估计值）
GIS_SPU_PJC_BIN_DIR=/tmp/pjc/bazel-bin/private_join_and_compute \
    /opt/miniconda3/envs/spu311/bin/python tests/benchmarks/benchmark_psi_sum.py
```

期望：状态表里 `CellSetIntersect` 的 Status 为 `count-and-sum`，
结果行打印 `(2, 13)`（交集码 B、C，权重 4 + 9）。

**实测（2026-10-09，WSL2 Ubuntu + Python 3.11.16）**
- 8 阶段全 OK；`|A|=3  |A∩B|=2`；`result = (2, 13)`；
  `reference = (3076832715348377604, 3076973452836732932)`，`agree=True`；
  `result-sems = exact`；`policy = REVEAL_INTERSECTION_SUM`；client 退出码 0；
  状态表 `CellSetIntersect | CompactCellSet | PSI | count-and-sum`。
- 上游 clone 得到 `950c5e4（2026-03-09）`、`.bazelversion = 8.0.1`，与登记一致。
- 首次构建在这台 20 核机器上约 11 分钟（Bazel 日志 16:22:19 起、脚本 16:34 结束，5,683 个 action）。
- 同一环境 `spu 0.9.5` / `jax 0.4.34`，SPU 能力核查 `runnable=true`；
  但 `openmined-psi` 未安装（PSI-CA 档在该环境不可执行，需 `pip install openmined-psi==2.0.6`）——与本档无关，如实记录。
- **2026-10-10（Phase 10）**：基线三条以计量重跑（通信量 + 峰值内存），
  读数与口径见 §7.1；`agree=true` 保持。
