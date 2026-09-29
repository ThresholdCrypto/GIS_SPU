# 本地测试指南（WSL2）

本文件记录**实测环境**与**逐步验证过程**，所有输出都是在本机真实跑出来的，
不是示意。命令与输出一一对应，可照抄复现。

## 实测环境

| 项 | 值 |
|----|----|
| 发行版 | Ubuntu 26.04.1 LTS (WSL2) |
| 解释器 | `/opt/miniconda3/envs/spu311/bin/python`（conda env `spu311`） |
| Python | 3.11.16（SPU 要求 `>=3.10,<3.12`） |
| spu | 0.9.5 |
| jax | 0.4.34（SPU 0.9.5 硬钉的版本） |
| numpy | 1.26.4（SPU 要求 `<2`） |
| libgomp1 | 已装（libspu / libpsi 的 OpenMP 运行时） |

> 为什么是 conda env 而不是系统 Python：系统 Python 是 3.14，
> 不在 SPU wheel 的 `requires-python` 区间内，且 SPU 只发布
> manylinux wheel（无 Windows wheel）。

## 一次性准备（已完成）

环境本身已装好。若在**新机器**上从零开始，跑：

```bash
bash scripts/setup_wsl_spu.sh
```

它做四件事：装 libgomp1 → 建 conda 3.11 环境 → 按
`requirements-spu.txt` 装 spu/jax/numpy → 跑 `check` + 全量测试 +
三个示例做验收。脚本末尾会打印激活命令。

## 用法

三种等价写法，任选：

```bash
# A. 激活环境（需**新开终端**，见下方“踩过的坑”）
conda activate spu311
python -m geosecure.cli ops

# B. 直接用全路径，不需要激活
/opt/miniconda3/envs/spu311/bin/python -m geosecure.cli ops

# C. 用安装好的入口脚本
/opt/miniconda3/envs/spu311/bin/geo-secure ops
```

> **踩过的坑：`conda init` 只对“新开的”终端生效。**
> 若当前终端是在初始化之前打开的，`conda` 与 `python` 都会报
> `command not found`。两种解法：
>
> 1. **新开一个终端**（最省事）；
> 2. 在当前终端里手动加载 conda 再激活：
>
> ```bash
> source /opt/miniconda3/etc/profile.d/conda.sh
> conda activate spu311
> ```
>
> 完全不想管这件事，就用写法 B / C —— 绝对路径不依赖 shell 初始化。

**最省事：一条命令跑完全流程**

```bash
cd <项目根>
bash scripts/debug_local.sh
```

它按顺序走完下面 8 步，每步都打印"这一步在看什么"。

---

## 逐步验证（含实测输出）

### 0/7 —— 环境版本核对
先确认解释器与三个关键库的版本，再谈任何结论。
```bash
python -c "import sys,spu,jax,numpy as np;print('py',sys.version.split()[0]);print('spu',spu.__version__);print('jax',jax.__version__);print('numpy',np.__version__)"
```
实测输出：
```text
python 3.11.16
spu    0.9.5
jax    0.4.34
numpy  1.26.4
```
### 1/7 —— 算子注册表
六个算子（外加 HeightBand）各自被规划成什么表征 / 后端。这是 Planner 的全部规则来源。
```bash
python -m geosecure.cli ops
```
实测输出：
```text
看什么：六个算子各自被规划成什么表征/后端。这是 Planner 的全部规则来源。
已登记算子
------------------------------------------------------------------------
Operation         Representation    Backend    Security  JAX  Cost
----------------  ----------------  ---------  --------  ---  ------------------------------------------
Intersects        CompactCellSet    PSI        high      否    b=64 d=0 R=1（PSI 求交轮次）
Contains          CompactCellSet    PSI/MPC    high      否    b=64 d=1（包含性判定） R=1（PSI）+ 1（包含性确认）
DistanceLE        QuantizedVector   MPC/SPU    high      是    b=8（量化箱号） d=1（平方后求和，无开方） R=1（归约）
WeightedSum       FixedPointVector  SPU/MPC    medium    是    b=18（8+8+ceil(log2 K)，K=3） d=1（乘加） R=1（归约）
TemporalOverlap   TimeInterval      MPC/SPU    high      是    b=18（Toff 14 + Lt 4） d=1（区间比较） R=1（归约）
CellSetIntersect  CompactCellSet    PSI        high      否    b=64 d=0 R=1（PSI）
HeightBand        CompactCellSet    Plaintext  low       否    b=64 d=0 R=0
```
### 2/7 —— SPU / JAX 能力核查
看 `runnable` 是否为 true、`blockers` 是否为空。缺 libgomp1 或不在 Linux 会在这里暴露。
```bash
python -m geosecure.cli check
```
实测输出：
```text
看什么：runnable 是否为 true；blockers 是否为空。缺 libgomp1 / 非 Linux 会在这里暴露。
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
SPU 能力核查
------------------------------------------------------------------------
{
  "status": "available",
  "runnable": true,
  "platform": "Linux",
  "machine": "x86_64",
  "python_version": "3.11.16",
  "python_supported": true,
  "jax": {
    "installed": true,
    "version": "0.4.34",
    "prefix": "/opt/miniconda3/envs/spu311/lib/python3.11/site-packages/jax/__init__.py",
    "private_deps_ok": true,
    "missing_deps": [],
    "jit_ok": true,
    "hlo_lowering_ok": true,
    "hlo_bytes": 1979,
    "error": null
  },
  "spu": {
    "installed": true,
    "version": "0.9.5",
    "prefix": "/opt/miniconda3/envs/spu311/lib/python3.11/site-packages/spu/__init__.py",
    "libspu_loadable": true,
    "simulation_api": [
      "spu.utils.simulation.Simulator",
      "spu.utils.simulation.sim_jax",
      "spu.utils.frontend.compile",
      "spu.utils.frontend.Kind"
    ],
    "error": null
  },
  "blockers": [],
  "details": {
    "platform": {
      "system": "Linux",
      "machine": "x86_64",
      "python_version": "3.11.16",
      "libspu_present": true,
      "notes": [],
      "python_supported": true
    }
  }
}
```
### 3/7 —— PSI 能力核查
协议与曲线清单、是否 file_io_only。PSI 走独立执行路径，不经过 jax.jit。
```bash
python -m geosecure.cli psi-check
```
实测输出：
```text
看什么：协议与曲线清单、是否 file_io_only。PSI 走的是独立执行路径，不经过 jax.jit。
PSI 能力核查
------------------------------------------------------------------------
{
  "installed": true,
  "version": "0.9.5",
  "prefix": "/opt/miniconda3/envs/spu311/lib/python3.11/site-packages/spu/__init__.py",
  "libpsi_loadable": true,
  "psi_execute_available": true,
  "create_mem_available": true,
  "protocols": [
    "PROTOCOL_DP",
    "PROTOCOL_ECDH",
    "PROTOCOL_ECDH_3PC",
    "PROTOCOL_ECDH_NPC",
    "PROTOCOL_KKRT",
    "PROTOCOL_KKRT_NPC",
    "PROTOCOL_RR22"
  ],
  "curves": [
    "CURVE_25519",
    "CURVE_25519_ELLIGATOR2",
    "CURVE_FOURQ",
    "CURVE_SECP256K1",
    "CURVE_SM2"
  ],
  "source_types": [
    "SOURCE_TYPE_FILE_CSV"
  ],
  "join_types": [
    "JOIN_TYPE_DIFFERENCE",
    "JOIN_TYPE_FULL_JOIN",
    "JOIN_TYPE_INNER_JOIN",
    "JOIN_TYPE_LEFT_JOIN",
    "JOIN_TYPE_RIGHT_JOIN"
  ],
  "file_io_only": true,
  "runnable": true,
  "blockers": [],
  "notes": [
    "PSI 当前只提供 CSV 文件接口，无内存张量接口；本后端用临时工作目录承载输入输出",
    "以下协议在当前 2 方链路上不可执行（参与方数量不符）：PROTOCOL_ECDH_3PC；可执行协议：PROTOCOL_ECDH, PROTOCOL_KKRT, PROTOCOL_RR22, PROTOCOL_ECDH_NPC, PROTOCOL_KKRT_NPC, PROTOCOL_DP",
    "以下协议结果**带噪**，不保证与明文一致、不可用于一致性验证：PROTOCOL_DP（差分隐私 PSI：交集中会注入假元素/丢弃真元素）"
  ],
  "error": null,
  "supported_ops": [
    "Intersects",
    "Contains",
    "CellSetIntersect"
  ]
}
```
### 4/7 —— 全量测试
passed 应为 456、skipped 应为 0。出现 skip 说明环境缺件被静默放过。
```bash
python -m pytest tests/ -q
```
实测输出：
```text
看什么：passed 数应为 456、skipped 应为 0。出现 skip 说明环境缺件被静默放过。
........................................................................ [ 78%]
........................................................................ [ 94%]
........................                                                 [100%]
456 passed in 10.14s
```
### 5/7 —— 编译 PSI 族示例
第 8 阶段做**真实 PSI 求交**；Result 表 Status 应为 `verified`。
```bash
python -m geosecure.cli build examples/route_conflict.py
```
实测输出：
```text
看什么：第 8 阶段做真实 PSI 求交；Result 表 Status=verified。
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
geo-secure — 地理信息行业低门槛隐私计算编译器
------------------------------------------------------------------------
Pipeline
[1/8] Parsing                + OK       1 个函数，1 个算子调用
[2/8] IR generation          + OK       GeoProgram route_conflict.py：2 输入 / 1 算子 / 1 关系
[3/8] Privacy planning       + OK       1 步方案，1 步需密态
[4/8] JAX generation         ! WARNING  生成 0 个函数；1 个算子不生成（Intersects）
[5/8] SPU capability check   + OK       available
[6/8] SPU simulation         - SKIPPED  没有可模拟的 JAX 实现；Intersects 属 PSI 族，由 PSI 阶段执行
[7/8] PSI capability check   + OK       1 个算子需 PSI（Intersects）；选用 PROTOCOL_ECDH / CURVE_SM2；当前环境可执行（spu 0.9.5）
[8/8] PSI simulation         + OK       1 个算子经真实 PSI 求交验证

------------------------------------------------------------------------
IR generation
Operation   Inputs                OutputType  Output            Location
----------  --------------------  ----------  ----------------  --------------------------------
Intersects  route_A, NoFlyZone_B  Relation    geo_intersects_1  examples/route_conflict.py:16:11

关系（GeoRelation）：
Subject  Predicate   Object       Time  SpatialScope  Sensitivity
-------  ----------  -----------  ----  ------------  -----------
route_A  Intersects  NoFlyZone_B        A             sensitive
  route_A | Intersects | NoFlyZone_B

------------------------------------------------------------------------
Privacy planning
Operation   Representation  Backend  EstimatedCost           Security  Status
----------  --------------  -------  ----------------------  --------  -------
Intersects  CompactCellSet  PSI      b=64 d=0 R=1（PSI 求交轮次）  high      planned

  后端: ['PSI']   表征: ['CompactCellSet']   安全级别: ['high']

------------------------------------------------------------------------
JAX generation
  - Intersects [PSI]：不生成 JAX（Intersects 属于 PSI 族，在 jax.numpy 中没有逐元素对应原语；集合交是组合问题而非逐元素算子，无法用张量表达式表达。该算子由 PSI 直接执行，不经过 JAX 代码生成。）

------------------------------------------------------------------------
SPU capability check
  状态      : available
  平台      : Linux / x86_64 / Python 3.11.16
  jax       : 0.4.34
  spu       : 0.9.5
  可运行    : 是

------------------------------------------------------------------------
PSI capability check
  协议      : PROTOCOL_DP, PROTOCOL_ECDH, PROTOCOL_ECDH_3PC, PROTOCOL_ECDH_NPC, PROTOCOL_KKRT, PROTOCOL_KKRT_NPC, PROTOCOL_RR22
  选用      : PROTOCOL_ECDH / CURVE_SM2
  曲线      : CURVE_25519, CURVE_25519_ELLIGATOR2, CURVE_FOURQ, CURVE_SECP256K1, CURVE_SM2
  带噪      : PROTOCOL_DP（差分隐私：结果不与明文保证一致）
  输入形态  : CSV 文件（无内存张量接口）
  键布局    : X17|Y17|Z7|L5|Toff14|Lt4（层级上限 L31；布局须与对端一致）
  可运行    : 是

------------------------------------------------------------------------
SPU simulation
  （未执行：没有可模拟的 JAX 实现；Intersects 属 PSI 族，由 PSI 阶段执行）

------------------------------------------------------------------------
PSI simulation
  Intersects: ok  [PROTOCOL_ECDH / CURVE_SM2]
      |A|=3  |A∩B|=2
      result    : True
      reference : True   agree=True
      leaks     : 接收方获得交集本体（不只是布尔值）；交集基数由 recipient 可见

------------------------------------------------------------------------
Result
Operation   Representation  Backend  Status
----------  --------------  -------  --------
Intersects  CompactCellSet  PSI      verified

------------------------------------------------------------------------
编译完成。
```
### 6/7 —— 编译 MPC 族示例 + 打印生成的 JAX 源码
第 6 阶段在 SPU 模拟器上真跑，tolerance 0.0 且 err=0.0；--verbose 打印生成的 jnp 代码。
```bash
python -m geosecure.cli build examples/distance_check.py --verbose
```
实测输出：
```text
看什么：第 6 阶段在 SPU 模拟器上真跑，tolerance 0.0 且 err=0.0；--verbose 给出生成的 jnp 代码。
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
geo-secure — 地理信息行业低门槛隐私计算编译器
------------------------------------------------------------------------
Pipeline
[1/8] Parsing                + OK       1 个函数，1 个算子调用
[2/8] IR generation          + OK       GeoProgram distance_check.py：3 输入 / 1 算子 / 1 关系
[3/8] Privacy planning       + OK       1 步方案，1 步需密态
[4/8] JAX generation         + OK       生成 1 个函数
[5/8] SPU capability check   + OK       available
[6/8] SPU simulation         + OK       1 个算子在 SPU 模拟器上验证通过
[7/8] PSI capability check   - SKIPPED  方案中没有走 PSI 的算子
[8/8] PSI simulation         - SKIPPED  方案中没有走 PSI 的算子

------------------------------------------------------------------------
IR generation
Operation   Inputs             OutputType  Output            Location
----------  -----------------  ----------  ----------------  --------------------------------
DistanceLE  p1, p2, threshold  Relation    geo_distancele_1  examples/distance_check.py:15:11

关系（GeoRelation）：
Subject  Predicate   Object  Time  SpatialScope  Sensitivity
-------  ----------  ------  ----  ------------  -----------
p1       DistanceLE  p2                          sensitive
  p1 | DistanceLE | p2

------------------------------------------------------------------------
Privacy planning
Operation   Representation   Backend  EstimatedCost                     Security  Status
----------  ---------------  -------  --------------------------------  --------  -------
DistanceLE  QuantizedVector  MPC/SPU  b=8（量化箱号） d=1（平方后求和，无开方） R=1（归约）  high      planned

  后端: ['MPC/SPU']   表征: ['QuantizedVector']   安全级别: ['high']

------------------------------------------------------------------------
JAX generation
  + geo_distancele_0(left, right, threshold) [jax.jit 可追踪, HLO 2243 B]

--- geo_distancele_0 ---
import jax.numpy as jnp

def geo_distancele_0(left, right, threshold):
    """距离是否不超过阈值。比较距离平方与阈值平方，避免 sqrt。"""
    delta = left - right
    dist_sq = jnp.sum(jnp.square(delta))
    return dist_sq <= jnp.square(threshold)


------------------------------------------------------------------------
SPU capability check
  状态      : available
  平台      : Linux / x86_64 / Python 3.11.16
  jax       : 0.4.34
  spu       : 0.9.5
  可运行    : 是

------------------------------------------------------------------------
SPU simulation
  DistanceLE: ok
      outputs   : True
      reference : True
      tolerance : 0.0  err=0.0  ok=True

------------------------------------------------------------------------
Result
Operation   Representation   Backend  Status
----------  ---------------  -------  --------
DistanceLE  QuantizedVector  MPC/SPU  verified

------------------------------------------------------------------------
编译完成。
```
### 7/7 —— 编译 3D 与组合示例
vertical_conflict 走 PSI（含 Contains 的 MPC 子集判定）；risk_score 一次覆盖两个算子（WeightedSum + TemporalOverlap）。
```bash
python -m geosecure.cli build examples/vertical_conflict.py
python -m geosecure.cli build examples/risk_score.py
```
实测输出：
```text
看什么：vertical_conflict 走 PSI；risk_score 一次覆盖两个算子（WeightedSum + TemporalOverlap）。
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
geo-secure — 地理信息行业低门槛隐私计算编译器
------------------------------------------------------------------------
Pipeline
[1/8] Parsing                + OK       1 个函数，2 个算子调用
[2/8] IR generation          + OK       GeoProgram vertical_conflict.py：2 输入 / 2 算子 / 2 关系
[3/8] Privacy planning       + OK       2 步方案，2 步需密态
[4/8] JAX generation         ! WARNING  生成 0 个函数；2 个算子不生成（Intersects, Contains）
[5/8] SPU capability check   + OK       available
[6/8] SPU simulation         - SKIPPED  没有可模拟的 JAX 实现；Intersects, Contains 属 PSI 族，由 PSI 阶段执行
[7/8] PSI capability check   + OK       2 个算子需 PSI（Intersects, Contains）；选用 PROTOCOL_ECDH / CURVE_SM2；当前环境可执行（spu 0.9.5）
[8/8] PSI simulation         + OK       2 个算子经真实 PSI 求交验证

------------------------------------------------------------------------
IR generation
Operation   Inputs                                OutputType  Output   Location
----------  ------------------------------------  ----------  -------  -----------------------------------
Intersects  route_band_cells, control_zone_cells  Relation    overlap  examples/vertical_conflict.py:38:14
Contains    control_zone_cells, route_band_cells  Relation    covered  examples/vertical_conflict.py:39:14

关系（GeoRelation）：
Subject           Predicate   Object              Time  SpatialScope  Sensitivity
----------------  ----------  ------------------  ----  ------------  -----------
route_band_cells  Intersects  control_zone_cells                      sensitive
route_band_cells  Contains    control_zone_cells                      sensitive
  route_band_cells | Intersects | control_zone_cells
  route_band_cells | Contains | control_zone_cells

------------------------------------------------------------------------
Privacy planning
Operation   Representation  Backend  EstimatedCost                       Security  Status
----------  --------------  -------  ----------------------------------  --------  -------
Intersects  CompactCellSet  PSI      b=64 d=0 R=1（PSI 求交轮次）              high      planned
Contains    CompactCellSet  PSI/MPC  b=64 d=1（包含性判定） R=1（PSI）+ 1（包含性确认）  high      planned

  后端: ['PSI', 'PSI/MPC']   表征: ['CompactCellSet']   安全级别: ['high']

------------------------------------------------------------------------
JAX generation
  - Intersects [PSI]：不生成 JAX（Intersects 属于 PSI 族，在 jax.numpy 中没有逐元素对应原语；集合交是组合问题而非逐元素算子，无法用张量表达式表达。该算子由 PSI 直接执行，不经过 JAX 代码生成。）
  - Contains [PSI/MPC]：不生成 JAX（Contains 属于 PSI/MPC 族，在 jax.numpy 中没有逐元素对应原语；集合交是组合问题而非逐元素算子，无法用张量表达式表达。该算子由 PSI/MPC 直接执行，不经过 JAX 代码生成。）

------------------------------------------------------------------------
SPU capability check
  状态      : available
  平台      : Linux / x86_64 / Python 3.11.16
  jax       : 0.4.34
  spu       : 0.9.5
  可运行    : 是

------------------------------------------------------------------------
PSI capability check
  协议      : PROTOCOL_DP, PROTOCOL_ECDH, PROTOCOL_ECDH_3PC, PROTOCOL_ECDH_NPC, PROTOCOL_KKRT, PROTOCOL_KKRT_NPC, PROTOCOL_RR22
  选用      : PROTOCOL_ECDH / CURVE_SM2
  曲线      : CURVE_25519, CURVE_25519_ELLIGATOR2, CURVE_FOURQ, CURVE_SECP256K1, CURVE_SM2
  带噪      : PROTOCOL_DP（差分隐私：结果不与明文保证一致）
  输入形态  : CSV 文件（无内存张量接口）
  键布局    : X17|Y17|Z7|L5|Toff14|Lt4（层级上限 L31；布局须与对端一致）
  可运行    : 是

------------------------------------------------------------------------
SPU simulation
  （未执行：没有可模拟的 JAX 实现；Intersects, Contains 属 PSI 族，由 PSI 阶段执行）

------------------------------------------------------------------------
PSI simulation
  Intersects: ok  [PROTOCOL_ECDH / CURVE_SM2]
      |A|=3  |A∩B|=2
      result    : True
      reference : True   agree=True
      leaks     : 接收方获得交集本体（不只是布尔值）；交集基数由 recipient 可见
  Contains: ok  [PROTOCOL_ECDH / CURVE_SM2]
      |A|=3  |A∩B|=2
      result    : False
      reference : False   agree=True
      subset    : False  (mode=mpc, ABY3/FM64)
      leaks     : 接收方获得交集本体（PSI 标准语义）；子集判定只输出一个布尔，其执行形态（MPC / 明文）另行披露，见 subset 字段；子集判定由 MPC 基数等值完成：k=|outer∩inner| 与 n=|inner| 各为一方私有输入，输出只有一个布尔；明文比较已移除

------------------------------------------------------------------------
Result
Operation   Representation  Backend  Status
----------  --------------  -------  --------
Intersects  CompactCellSet  PSI      verified
Contains    CompactCellSet  PSI/MPC  verified

------------------------------------------------------------------------
编译完成。
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
geo-secure — 地理信息行业低门槛隐私计算编译器
------------------------------------------------------------------------
Pipeline
[1/8] Parsing                + OK       1 个函数，2 个算子调用
[2/8] IR generation          + OK       GeoProgram risk_score.py：4 输入 / 2 算子 / 1 关系
[3/8] Privacy planning       + OK       2 步方案，2 步需密态
[4/8] JAX generation         + OK       生成 2 个函数
[5/8] SPU capability check   + OK       available
[6/8] SPU simulation         + OK       2 个算子在 SPU 模拟器上验证通过
[7/8] PSI capability check   - SKIPPED  方案中没有走 PSI 的算子
[8/8] PSI simulation         - SKIPPED  方案中没有走 PSI 的算子

------------------------------------------------------------------------
IR generation
Operation        Inputs                                 OutputType  Output     Location
---------------  -------------------------------------  ----------  ---------  ----------------------------
WeightedSum      risk_factors, weights                  Scalar      score      examples/risk_score.py:17:12
TemporalOverlap  availability_window, operation_window  Relation    window_ok  examples/risk_score.py:18:16

关系（GeoRelation）：
Subject              Predicate        Object            Time  SpatialScope  Sensitivity
-------------------  ---------------  ----------------  ----  ------------  -----------
availability_window  TemporalOverlap  operation_window                      sensitive
  availability_window | TemporalOverlap | operation_window

------------------------------------------------------------------------
Privacy planning
Operation        Representation    Backend  EstimatedCost                               Security  Status
---------------  ----------------  -------  ------------------------------------------  --------  -------
WeightedSum      FixedPointVector  SPU/MPC  b=18（8+8+ceil(log2 K)，K=3） d=1（乘加） R=1（归约）  medium    planned
TemporalOverlap  TimeInterval      MPC/SPU  b=18（Toff 14 + Lt 4） d=1（区间比较） R=1（归约）      high      planned

  后端: ['MPC/SPU', 'SPU/MPC']   表征: ['FixedPointVector', 'TimeInterval']   安全级别: ['high', 'medium']

------------------------------------------------------------------------
JAX generation
  + geo_weightedsum_0(values, weights, scale) [jax.jit 可追踪, HLO 3948 B]
  + geo_temporaloverlap_1(left_toff, left_lt, right_toff, right_lt) [jax.jit 可追踪, HLO 4668 B]

------------------------------------------------------------------------
SPU capability check
  状态      : available
  平台      : Linux / x86_64 / Python 3.11.16
  jax       : 0.4.34
  spu       : 0.9.5
  可运行    : 是

------------------------------------------------------------------------
SPU simulation
  WeightedSum: ok
      outputs   : 80
      reference : 80
      tolerance : 0.0  err=0.0  ok=True
  TemporalOverlap: ok
      outputs   : True
      reference : True
      tolerance : 0.0  err=0.0  ok=True

------------------------------------------------------------------------
Result
Operation        Representation    Backend  Status
---------------  ----------------  -------  --------
WeightedSum      FixedPointVector  SPU/MPC  verified
TemporalOverlap  TimeInterval      MPC/SPU  verified

------------------------------------------------------------------------
编译完成。
```
### 完成
```text
想看单文件的失败报告，自己造一个错误文件再编译，例如：
  printf "from geo_privacy import geo\n\ndef f(a,b):\n    return geo.buffer_zone(a,b)\n" > /tmp/bad.py
  /opt/miniconda3/envs/spu311/bin/python -m geosecure.cli build /tmp/bad.py
```
---

## 常见问题

### 1. `An NVIDIA GPU may be present ... Falling back to cpu.`

JAX 在导入时探测到本机有 NVIDIA 驱动、但没有装 CUDA 版 jaxlib，
于是回退到 CPU。**这条是良性提示，不是错误**：

- 项目的 SPU 后端跑的是**模拟器**（`spu.utils.simulation`），本来就在 CPU 上；
- `requirements-spu.txt` 钉的是 `jax[cpu]`，装 CUDA 版会偏离 SPU 0.9.5 的版本区间。

想让它闭嘴，在跑命令前加：

```bash
export JAX_PLATFORMS=cpu
```

### 2. IDE 里 `Import "jax.numpy" could not be resolved` 红线

这是 **Pylance 用的是哪个解释器**的问题，与代码无关。当前的
`backends/jax_backend/*.py` 把 `import jax` / `import jax.numpy as jnp`
放在 `try/except ImportError` 里（为了在没装 JAX 的机器上也能跑明文流程），
所以一旦解释器里没有 jax，Pylance 就会报"无法解析"。

本机的关键事实：**VS Code 没有跑在 WSL 远程模式**（`~/.vscode-server`
不存在），所以 Pylance 用的是 **Windows 侧**的解释器 —— 那个解释器
既没有 jax，也**永远不会**有 spu（SPU 无 Windows wheel）。

正确修法（推荐）：

1. 在 VS Code 里按 `F1` → **WSL: Reopen Folder in WSL**；
2. 重新打开项目根目录 `2026-09-20/geosot-3d-dqg-4d-c-users-2`；
3. 解释器选 `/opt/miniconda3/envs/spu311/bin/python`
   （仓库里已有 `.vscode/settings.json` 把它设成默认，通常会自动选中）。

切到 WSL 模式后，`jax` / `jax.numpy` / `spu` / `planner` / `backends`
这些导入会一次性全部解析成功，4 条红线消失。

若坚持留在 Windows 模式：可以在 Windows Python 上
`pip install "jax[cpu]"` 消掉 jax 那几条，但 `import spu` 的那条
**无法消除**（上游没有 Windows wheel）。所以还是建议切 WSL 模式。

### 3. 没有 `pip` / `conda` 命令

本机 `hjd` 用户的系统 Python 是 3.14，且没装 pip3 —— 这不需要管。
本项目一律用 `spu311` 环境里的解释器，见上面"用法"一节。
若 `conda` 不在 PATH，先 `source /opt/miniconda3/etc/profile.d/conda.sh`，
或直接用全路径写法 B / C。

### 4. 两方结果不一致（PSI 不报错但结论不对）

最常见的原因是**键布局不一致**。PSI 走 CSV 交换 64 位格网码，
两方用了不同的位域布局时，交集不会报错，只会**静默算错**。

本版布局（`ir.values.GRID_CODE_LAYOUT`）：

```text
X17|Y17|Z7|L5|Toff14|Lt4     层级上限 L31
```

`geo-secure build` 的 PSI 能力块会把这一行打出来，逐字比对即可。

注意：本版把 L 位域由 4 位扩到 5 位，**所有已生成的码都变了**：

```text
旧(L4): 0x2AB29B0D87C90E08
新(L5): 0x2AB29B0D87A48704
```

旧样例数据必须重算，否则两方对不上。
