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
- **未验证**：**`openmined-psi` 真机执行**（本仓库当前开发环境为 Windows，
  该发行版无 Windows 轮子）。真机复跑前，不得对外宣称"PSI-Cardinality 已验证"。

## 8. WSL2 / Linux 复跑

```bash
# 1) 安装（Linux / WSL2）
/opt/miniconda3/envs/spu311/bin/python -m pip install openmined-psi==2.0.6

# 2) 环境核查（期望 runnable=true、installed=true）
/opt/miniconda3/envs/spu311/bin/python -m geosecure.cli psi-ca-check

# 3) 端到端（样例兜底输入；期望 |A∩B|=2、状态词 count-only）
/opt/miniconda3/envs/spu311/bin/python -m geosecure.cli build examples/conflict_count.py --psi-count psi-ca

# 4) 测试（真机下 TestCountModeExecution 走 "ok + count-only" 分支）
/opt/miniconda3/envs/spu311/bin/python -m pytest tests/test_psi_ca_backend.py tests/test_cli_psi_ca.py -q
```

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
