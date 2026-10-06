# SPU 能力核查（含证据出处）

> 本文件记录**实际核对过**的事实。凡未经核对的内容一律标注为"未核实"，
> 不写进实现。核查时间：2026-09-23。

## 一、核对方法

| 项 | 手段 |
|---|---|
| SPU 版本与依赖 | 下载官方 PyPI 发布包 `spu==0.9.5` 的 wheel 与 `METADATA`，直接读文件 |
| SPU API 形态 | 解包 wheel，读 `spu/utils/simulation.py`、`spu/utils/frontend.py`、`spu/libspu.pyi` |
| 枚举取值 | 读 `spu/libspu.pyi` 的类型存根，提取枚举成员 |
| JAX 兼容性 | 在两个解释器中实际导入并探测私有接口：jax 0.11.2 与 jax 0.4.34 |
| 生成代码的原语 | 实跑 `jax.jit(...).lower(('interpreter',)).as_text()`，从文本 HLO 提取 `stablehlo.*` |

核对产物留存在 `docs/spu_capability_report.json`（机器可读）与本文（人可读）。

## 二、SPU 0.9.5 事实

### 2.1 分发约束

- PyPI 包名 `spu`，最新发布版 **0.9.5**。
- `requires-python = ">=3.10,<3.12"` —— **3.12 及以上不在支持区间**。
- wheel 平台仅三档：
  - `macosx_14_0_arm64`（cp310 / cp311）
  - `manylinux_2_17_x86_64`（cp310 / cp311）
  - `manylinux_2_28_aarch64`（cp310 / cp311）
- **无 Windows wheel**。原生 Windows x64 上 `libspu` 是 Linux ELF，无法加载。
- 结论：Windows 上要用 SPU，必须走 WSL2 或 Linux 容器。

### 2.2 硬依赖

来自 0.9.5 的 `METADATA`：

```
numpy<2,>=1.22.0
cloudpickle>=2.0.0
multiprocess>=0.70.12.2
cachetools>=5.0.0
jax[cpu]<=0.4.34,>=0.4.16
termcolor>=2.0.0
```

注意两点：
1. `jax` 被**硬钉在 0.4.16 ~ 0.4.34**；装更新版本（如 0.11.2）会破坏 SPU。
2. `numpy<2`；而 jax 0.4.34 要求 `numpy>=2`（0.4.34 的元数据）——
   这在实践中由 pip 解析器处理，**未在本机验证过二者共存**（标注：未核实）。

### 2.3 私有 JAX 接口依赖

`spu/utils/frontend.py` 与 `spu/utils/simulation.py` 直接引用以下非公开接口：

| 接口 | jax 0.4.34 | jax 0.11.2 |
|---|---|---|
| `jax.extend.linear_util` | 存在 | 存在 |
| `jax._src.api_util` | 存在 | 存在 |
| `jax._src.lib.xla_extension_version` | 存在（值 289） | **已移除** |
| `jax._src.xla_bridge._backend_lock / _backends / register_backend_factory` | 存在 | 存在 |
| `jax.interpreters.xla.Backend` | 存在 | 存在 |
| `jax._src.lax.lax._canonicalize_float_for_sort` | 存在 | 存在 |

**这是最关键的兼容性风险**：SPU 0.9.5 内部按
`xla_extension_version` 分支选择新旧 API，该符号在 jax ≥ 0.5 已被删除。
在 jax 0.11.2 上，`from jax._src.lib import xla_extension_version` 会 `ImportError`。

> **实现提示（已修正）**：`xla_extension_version` 与 `xla_client` 是 `jax._src.lib` 下的
> **属性**，不是可 `import` 的子模块。若核查代码直接对它调
> `importlib.import_module`，会把**已存在的属性误判为缺失**，从而把真实可运行的
> 环境标成 `runnable=False`（本次实际遇到并已修复）。现改为
> “模块优先，导入失败则拆出父模块取属性”，并配正/逆两条回归测试。

### 2.4 枚举取值（来自 `libspu.pyi`）

```
ProtocolKind : REF2K | SEMI2K | ABY3 | CHEETAH | SECURENN     （无 SPDZ2K）
FieldType    : FM32 | FM64 | FM128
Visibility   : VIS_SECRET | VIS_PUBLIC | VIS_PRIVATE
SourceIRType : XLA | STABLEHLO
```

参与方数量下限（按协议定义）：`ABY3 = 3`、`SECURENN = 3`、`SEMI2K = 2`、`CHEETAH = 2`、`REF2K = 2`。

> **实测补充（2026-10-05，P1）**：5 个协议 × 3 个环宽（FM32 / FM64 / FM128）
> 已逐条真跑（`DistanceLE` / `WeightedSum` / `TemporalOverlap`，
> 数据见 `docs/mpc_benchmark_baseline.json`）。两条**平台侧限制**（真机实测，
> 不粉饰）：
>
> 1. `WeightedSum` 的定点整除（生成代码的 `acc // scale`）在 SPU 上是
>    **迭代近似**实现（栈里是 `div_goldschmidt`）：`scale == 1` 也不保证恒等映射，
>    K=4096 实测偏差最大 4，且**非确定**——同一组合重跑可能恰好逐位一致；
> 2. `WeightedSum × FM32` **起不来**：除法路径内部需要 64 位环
>    （`integer encoding failed, ring=FM32 could not represent PT_I64`）。
>    即该算子的环宽下限是 **FM64**，这与 planner 的位宽预测 b(K) 是两件事。
>
> 规程、扫描策略与全部数据：`docs/MPC_BENCHMARK_PROTOCOL.md`。
>
> **补充（2026-10-06，P2-1 已闭合）**：上面第 1、2 条的根因是**生成代码里的
> 一步整数除法**。把 `WeightedSum` 的定点 scale 改成编译期常量、`scale == 1`
> 时不再发射除法后：除法原语（`divide/remainder/select/sign`）从生成代码里消失，
> `WeightedSum × FM32` 实测可用，K=64…4096 全部与明文逐位一致，
> PPHLO 字节数 2 400 → 938、通信量（ABY3, K=256）9 286 B → 4 112 B、
> CHEETAH 墙钟中位数 1 150 ms → 85 ms。
> 结论保留的意义：**"SPU 的除法是近似实现"这条平台事实仍然成立**，
> 只是不再落在该算子的电路里。
>
> **补充（2026-10-06，P2-2）：通信量只能从原生日志读，且原生日志是进程级开关**
> `Context` 上**没有**任何字节计数接口，Python 侧也拿不到执行统计对象——
> 通信量唯一的来源是把 `RuntimeConfig.enable_pphlo_profile` 打开、再读
> C 层 spdlog 写出的 `Link details: total send bytes N, recv bytes M`。
> 两个实测约束：其一，这些行**只有 fd 级重定向**拿得到
> （`contextlib.redirect_stdout` 不行），见
> `backends/spu_backend/profile.py::capture_native_logs`；其二，
> `libspu.logging.setup_logging` 是**进程级单例**配置——本仓库 PSI 路径会用
> `quiet=True` 把它关掉，**关过之后同一进程里再也不出 profile 行**，
> 因此采集通信量前必须重新打开（`enable_native_console_log()`），
> 且 `system_log_path` 要显式改指 `/dev/null`（默认的相对路径 `'spu.log'`
> 会在当前工作目录落盘）。
> 数据见 `docs/mpc_comm_baseline.json`；规程见 `docs/MPC_BENCHMARK_PROTOCOL.md` §8.4。

### 2.5 没有 `run_spu_simulation`

**SPU 0.9.5 不存在 `run_spu_simulation` 这个函数。** 官方执行入口是：

```python
from spu.utils import simulation
sim = simulation.Simulator.simple(wsize, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)
spu_fn = simulation.sim_jax(sim, my_jax_fn)
result = spu_fn(*inputs)
```

`sim_jax` 内部调用 `spu.utils.frontend.compile(Kind.JAX, fn, ...)`，其流程为：

1. 注册一个 dummy `interpreter` 后端（`register_backend_factory('interpreter', ...)`）；
2. `jax.jit(fn, ...).trace(*args).lower(lowering_platforms=('interpreter',))`；
3. `lowered.compiler_ir('hlo').as_serialized_hlo_module_proto()`；
4. `spu_api.compile(CompilationSource(XLA, ir_text, vis), copts)` 得到 MLIR executable；
5. 每个参与方起一个线程，经 mem link 执行，`io.reconstruct` 还原明文结果。

本项目把第 1~3 步单独抽为 `backends/jax_backend/spu_bridge.py`，
因为这三步**不需要 libspu**，可在没有 SPU 的环境里独立验证
"生成的 JAX 代码能否被 SPU 的编译前端吃下"。

`run_spu_simulation(jax_fn, inputs, protocol, field)` 是本项目在官方 API 之上的封装
（课题要求的命名），实现见 `backends/spu_backend/runtime.py`。

### 2.6 前端补丁点（跨版本脆弱处）

`spu/utils/frontend.py` 在编译前会 patch `jax._src.lax.lax`：

- 目标函数名尝试两个候选：`_float_to_int_for_sort`、`_canonicalize_float_for_sort`
- 若无一个存在，直接抛
  `RuntimeError: Failed to patch sort, current jax version is not compatible with SPU`

本机实测：两个 jax 版本上都存在 `_canonicalize_float_for_sort`（后者），
所以**该补丁点当前不会失败**；但它与 `xla_extension_version` 一样属于私有接口，
升级 jax 时需重新核对。

## 三、本项目生成代码的原语核查（实测）

对 `DistanceLE / WeightedSum / TemporalOverlap` 的生成代码实跑 lower，提取文本 HLO：

| 算子 | 实测发射的 StableHLO 原语 |
|---|---|
| DistanceLE | `subtract, multiply, reduce, add, convert, compare, constant` |
| WeightedSum | `multiply, reduce, add, constant`（P2-1 后；修前为 `multiply, reduce, add, divide, remainder, compare, select, sign, and, convert, subtract, constant`） |
| TemporalOverlap | `shift_left, add, broadcast_in_dim, compare, and, or, reduce, constant` |

**重要发现（正是"不要假设 SPU API"的实例）**：

源码里的定点整除 `acc // scale` 是一步写法，但 HLO 展开成
`divide + remainder + select + sign` 四类原语的组合。
也就是说，"看起来一步"的操作在密态下的乘法深度与通信轮次远高于直觉。
`backends/spu_backend/capability.py` 因此把 `divide` / `remainder` 列入
**高代价原语**并在能力核查时输出告警。

> **P2-1 落地（2026-10-06）**：`WeightedSum` 的定点 `scale` 改成**编译期常量**，
> 生成代码里不再有除法 ⇒ 上表该行的原语集合收窄为
> `multiply, reduce, add, constant`，除法告警也随之消失
> （有反向回归用例守着：把运行时除法加回去就会失败）。

该表由 `tests/test_jax_backend.py::TestHloAndPrimitives::test_measured_primitives_match_registry`
锁定：若生成代码变动导致原语集合超出登记清单，测试会失败。

## 四、环宽与整数溢出

本项目 `grid_code` 为 **64 位定长键**。在 `FM64` 环上把它当无符号整数参与运算，
靠近 \(2^{64}\) 的值会翻转为负数，使比较结果错误。

- 因此 `Intersects` / `Contains` / `CellSetIntersect` 的能力核查会输出 FM128 告警。
- 为什么仍推荐 PSI 而非 MPC 做集合交：PSI 只比较等值不比较大小，
  不存在溢出语义问题（等值比较在两边一致即可）。
- 若必须用 MPC 做 64 位键的大小/包含判定，应切 `FM128`，
  代价是位宽翻倍（四量里的 `b` 直接翻倍，通信量随之上升）。

## 五、当前环境的核查结论

### 5.1 已验证可运行环境（WSL2 / Linux）

```
状态      : available
平台      : Linux / x86_64 / Python 3.11.16
jax       : 0.4.34（私有接口全部就位，值 289）
spu       : 0.9.5（libspu 可加载）
阻断项：无
可运行    : 是
```

在此环境下：

- **已验证**：Geo-IR、隐私规划、JAX 代码生成、`jax.jit` 可追踪性、
  HLO 降级、明文/JAX 结果一致性，以及 **真实的 SPU 多方模拟执行**。
- 实跑算子（ABY3 / FM64，wsize=3）：

| 算子 | SPU 输出 | 明文输出 | max_abs_error | pphlo 字节 |
|---|---|---|---|---|
| DistanceLE | True | True | 0.0 | 1508 |
| WeightedSum | 80 | 80 | 0.0 | **938**（P2-1 前 2390） |
| TemporalOverlap | True | True | 0.0 | 2593 |

实际运行报告：`docs/spu_capability_report_wsl.json`。

> **跑通 SPU 后才暴露的第二个 bug**：CLI 构造 SPU 参考值时，
> `reference_fn` 直接把**原始样例数组**交给 `run_plain`，漏了
> `PLAIN_ARG_BUILDERS` 的参数拆解。`TemporalOverlap` 需把 4 个节点数组
> 还原为 2 个节点列表，参数不符 -> 抛异常 -> 被
> `_plain_reference` 的 `except` 吞掉成 `None` -> 参考值为空 -> 误报为
> **SPU 执行失败**。实际上 SPU 跑对了（输出 True、误差 0.0）。
> 已修正为参考侧同样走参数拆解；新增两条回归测试锁定，
> 并用“回退修复则测试必失败”做过对抗性验证。
>
> 这是一个 **只有真实跑通 SPU 才会暴露** 的缺陷：在无 SPU 环境下
> 模拟阶段直接 unavailable，参考值根本不会被消费。

> **第三个 bug（探测口径错误）**：`probe_platform()` 用
> `shutil.which("libspu.so")` 判断原生库是否存在。`shutil.which` 只扫 `PATH`
> 上的**可执行**文件，而 `.so` 是动态库，永远命中不了，
> 因此 `libspu_present` 恒为 `false`，与同一份报告里的
> `libspu_loadable: true` 直接矛盾。已改用
> `importlib.util.find_spec("spu.libspu")`（问 Python 自己能不能找到该扩展模块），
> 并新增两条回归测试（含一条一致性断言）。

### 5.2 Windows 原生环境（仍不可运行）

```
状态      : unavailable
平台      : Windows / AMD64
spu       : 未安装
阻断项：
  - 原生 Windows 无 libspu 原生库（需 WSL2 / Linux）
```

`run_spu_simulation` 在此类环境下返回 `status="unavailable"`
并附带完整阻断清单，**不返回任何推测数值**。

## 六、备忘：SPU 官方仓库中的相关版本信息

（来自上一轮对官方仓库 `spu/pyproject.toml` 的读取，供交叉参考）

- 开发分支声明 `jax[cpu]==0.8.0`，`requires-python = ">=3.11,<3.13"`。
  这与已发布的 0.9.5（钉 0.4.34）不一致，说明主干与发布版存在差距；
  **本项目按已发布版 0.9.5 的约束实现**，并在能力核查中同时报告
  "jax 版本超出绑定区间"这一事实。
- 仓库版本快照为 `0.10.0.dev20251211`（开发版）。
