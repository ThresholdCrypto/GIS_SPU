# SPU 源码获取与真实执行验证报告

> 课题：三维格网数据统一接入隐私计算体系（GeoSOT-3D / DQG-4D）
> 项目：`GIS_SPU/` 地理信息行业低门槛隐私计算编译器 MVP
> 日期：2026-09-23
> 结论：**端到端流水线已跑通到真实 SPU 模拟执行**。`python → Geo-IR → Privacy Plan → JAX → SPU simulation` 全链路可用；
> 3 个可生成 JAX 的算子在真实 SPU 模拟器上实跑，与明文误差 0.0。测试 164 项全部通过、无跳过。

---

## 一、本轮做了什么

| # | 事项 | 结果 |
|---|---|---|
| 1 | 打通 WSL2 访问 | 沙箱内 `E_ACCESSDENIED`；本轮改为**沙箱外以你本人账户执行**后打通 |
| 2 | 获取 SPU 运行时 | 装 `spu==0.9.5`（manylinux cp311 wheel，75,496,690 B） |
| 3 | 解决 Python 版本问题 | WSL 是 Ubuntu 26.04，自带 Python **3.14**；SPU 只支持 3.10/3.11 → 用 Miniconda 建 3.11 环境 |
| 4 | 补齐系统依赖 | 缺 `libgomp.so.1`（OpenMP）→ 装 `libgomp1` |
| 5 | 跑通真实 SPU | `geosecure.cli build` 全 6 阶段通过，3 算子 `verified` |
| 6 | 修掉 3 个真实 bug | 详见第三节 |
| 7 | 补回归测试 | 6 条新测试，并做**对抗性验证**（回退修复则测试必失败） |
| 8 | 更新文档与报告 | README / SPU_CAPABILITY / requirements 全部改为实测口径 |

---

## 二、实际验证环境（逐字核对）

```
platform : Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43
python   : 3.11.16
numpy    : 1.26.4
jax      : 0.4.34
spu      : 0.9.5
libspu   : /opt/miniconda3/envs/spu311/lib/python3.11/site-packages/spu/libspu.so
```

**能力核查**（`geosecure check`）：

```json
{
  "status": "available", "runnable": true,
  "python_version": "3.11.16", "python_supported": true,
  "jax": {"version": "0.4.34", "private_deps_ok": true, "missing_deps": [],
          "jit_ok": true, "hlo_lowering_ok": true, "hlo_bytes": 1979},
  "spu": {"installed": true, "version": "0.9.5", "libspu_loadable": true,
          "simulation_api": ["spu.utils.simulation.Simulator",
                             "spu.utils.simulation.sim_jax",
                             "spu.utils.frontend.compile",
                             "spu.utils.frontend.Kind"]},
  "blockers": []
}
```

**执行路径**（严格走官方 API，未改 SPU 源码）：

```python
sim    = spu.utils.simulation.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)
spu_fn = spu.utils.simulation.sim_jax(sim, jax_fn)     # 内部 frontend.compile(Kind.JAX, ...)
out    = spu_fn(*inputs)
```

---

## 三、真实 SPU 算子一致性

协议 ABY3、环宽 FM64、参与方 3（ABY3 的最小 world size）。

| 算子 | SPU 输出 | 明文输出 | max_abs_error | 容差 | pphlo 字节 |
|---|---|---|---|---|---|
| DistanceLE | `True` | `True` | 0.0 | 0.0 | 1508 |
| WeightedSum | `80` | `80` | 0.0 | 0.0 | 2390 |
| TemporalOverlap | `True` | `True` | 0.0 | 0.0 | 2593 |

TemporalOverlap 另做负例（`[0,2)` vs `[100,102)` 不相交）→ SPU 与明文同为 `False`，误差 0.0。

**容差为何取 0**：三个算子在设计上都是整数/定点路径
（`QuantizedVector` 平方和比较、`FixedPointVector` 定点乘加、`TimeInterval` 整数区间比较），
把浮点误差挡在密态之外。整数路径下密文与明文应**完全一致**，容差 0 是刻意选择而非放松。

---

## 四、端到端编译产物（最终状态表）

```
--- examples/route_conflict.py ---
Intersects   CompactCellSet  PSI      backend-direct

--- examples/distance_check.py ---
DistanceLE   QuantizedVector MPC/SPU  verified

--- examples/risk_score.py ---
WeightedSum      FixedPointVector  SPU/MPC  verified
TemporalOverlap  TimeInterval      MPC/SPU  verified
```

- `verified` = 生成 JAX 且**在真实 SPU 模拟器上跑通并与明文一致**。
- `backend-direct` = 该算子属 PSI 族，无逐元素 JAX 原语，由 PSI 后端直接执行（不硬凑张量表达式）。
- `Intersects` 的关系对象 `route_A | Intersects | NoFlyZone_B` 已按标准 Geo-IR 表示。

---

## 五、本轮修掉的 3 个真实 bug

跑通真实 SPU 前，这些问题在无 SPU 环境里都不会暴露。

### Bug 1：能力核查把「属性」当「子模块」（误判环境不可用）

`SPU_JAX_PRIVATE_DEPS` 里含 `jax._src.lib.xla_extension_version`。
SPU 源码写法是 `from jax._src.lib import xla_client, xla_extension_version` —— 这是
**祖先模块的属性**，不是可 `import` 的子模块。核查代码直接对它调 `importlib.import_module`，
必然抛异常，于是把**本可运行的环境**标记为 `runnable=False`。

实测证据：`jax._src.lib.xla_extension_version = 289`（确实存在），
但 `import_module("jax._src.lib.xla_extension_version")` → `ModuleNotFoundError`。

修复：改为「模块优先；导入失败则拆出父模块 `getattr`」。
影响范围：这正是上一轮「环境阻断」结论的直接来源之一。

### Bug 2：SPU 参考值构造漏了参数拆解（把通过误报为失败）

CLI 构造 SPU 参考值时，`reference_fn` 直接把**原始样例数组**交给 `run_plain`，
漏了 `PLAIN_ARG_BUILDERS`。`TemporalOverlap` 需把 4 个节点数组还原为 2 个节点列表，
参数不符 → 抛异常 → 被 `_plain_reference` 的 `except` 吞成 `None` → 参考值为空 →
**误报为 SPU 执行失败**。实际上 SPU 跑对了（输出 `True`、误差 0.0）。

修复：参考侧同样走参数拆解。这是一个**只有真实跑通 SPU 才会暴露**的缺陷：
无 SPU 时模拟阶段直接 `unavailable`，参考值根本不会被消费。

### Bug 3：用 `shutil.which` 判断共享库是否存在（字段自相矛盾）

`probe_platform()` 用 `shutil.which("libspu.so")` 判断原生库是否存在。
`shutil.which` 只扫 `PATH` 上的**可执行**文件，`.so` 是动态库，永远命中不了，
所以 `libspu_present` 恒为 `false`，与同一份报告里的 `libspu_loadable: true` 直接矛盾。

修复：改用 `importlib.util.find_spec("spu.libspu")`（问 Python 自己能否找到该扩展模块）。
修复后 `libspu_present = libspu_loadable = True`。

---

## 六、测试

```
tests/test_ir.py           37 项   类型系统、格网口径、算子/程序/关系
tests/test_planner.py      26 项   注册表、五元组、敏感度策略、无副作用
tests/test_jax_backend.py  37 项   生成器、可追踪性、原语核对、与明文对拍
tests/test_spu_backend.py  31 项   协议/环宽规范化、能力门控、私有接口与共享库回归、SPU 实跑
tests/test_end_to_end.py   33 项   全流程、状态表、CLI、五类失败报告、确定性
                          ─────
                          164 通过 / 0 跳过
```

本轮新增 6 条回归测试，全部做**对抗性验证**（把修复回退，测试必须失败）：

| 测试 | 锁定 |
|---|---|
| `test_private_dep_check_handles_attribute_form` | Bug 1 正例 |
| `test_private_dep_check_rejects_missing_names` | Bug 1 逆例（真缺失必须判为缺失） |
| `test_simulation_reference_is_not_none_for_every_operator` | Bug 2 根因 |
| `test_temporal_overlap_never_reported_as_error` | Bug 2 现象 |
| `test_libspu_present_agrees_with_libspu_loadable` | Bug 3 一致性 |
| `test_module_spec_available_handles_shared_library` | Bug 3 探测口径 |

---

## 七、可复现路径

项目内新增一键脚本：

```bash
bash scripts/setup_wsl_spu.sh
```

它会：装 `libgomp1` → 必要时用 conda 建 Python 3.11 → 装 `requirements-spu.txt`
→ 核对版本与 API → 跑 `tests/` → 逐个编译 3 个示例。**已实测幂等**（重复执行会复用已有环境）。
支持用环境变量覆盖：`SPU_ENV_NAME` / `CONDA_HOME` / `PY_INDEX` / `CONDA_MIRROR`。

手工等价步骤：

```bash
pip install -r requirements-spu.txt      # spu==0.9.5 / jax<=0.4.34 / numpy<2
python -m pytest tests/ -q               # 164 项全部通过
python -m geosecure.cli build examples/risk_score.py
```

---

## 八、仍然存在的限制（未变的部分）

1. **Windows 原生不可运行 SPU**。`libspu` 只发布 manylinux/macOS wheel，需 WSL2 或 Linux 容器。
   这是上游发布策略，不是本项目问题。
2. **PSI 族占位为后端直连**。`Intersects` / `Contains` / `CellSetIntersect`
   在方案里标 `PSI` / `backend-direct`，**尚未接入真实 PSI 协议**（OPRF / RR22 类）。
   这是下一阶段的第一优先扩展点。
3. **只认 `geo_privacy` 方言**。不做通用 GeoPandas / Shapely 源码自动转换——这是刻意的边界。
4. **不编译依赖运行期数据的动态控制流**，直接报错并给替代算子建议，不尝试展开。
5. **未验证非 ABY3 协议**。本轮只实测 `ABY3`/`FM64`；
   `SEMI2K` / `CHEETAH` / `REF2K` / `SECURENN` 与 `FM32` / `FM128` 尚未实跑。
6. **`grid_code` 64 位在 FM64 上做大小比较有溢出风险**，核查会提示改用 FM128。

---

## 九、下一步建议

1. **接真实 PSI**：为 `Intersects` / `Contains` / `CellSetIntersect` 接 OPRF 类 PSI，
   把 `backend-direct` 推进到 `verified`。这是覆盖格网求交场景的关键一步。
2. **扩协议/环宽矩阵**：至少补 `SEMI2K`（2 方，通信更低）与 `FM128`（消除 64 位比较溢出），
   形成「算子 × 协议 × 环宽」的实测矩阵。
3. **补浮点路径的容差验证**：当前整数路径容差 0；若上游把坐标浮点化，
   需启用 `FLOAT_TOLERANCE = 1e-4` 并实测密态误差来源。
4. **性能基线**：记录各算子 pphlo 字节与密态轮次，对齐课题的「四量」（`N_ct` / `b` / `d` / `R`）口径。
5. **低空导航场景闭环**：把 Route_A / NoFlyZone_B 样例扩展成多时段、多空域的批量判定，
   验证 PSI 与 MPC 混合调度。

---

## 十、附件

- `outputs/SPU真实验证_终端逐字记录.txt` —— 本轮终端原始输出（环境 / check / pytest / 算子一致性 / 状态表）
- `GIS_SPU/docs/spu_capability_report_wsl.json` —— WSL 真实环境的能力核查 JSON
- `GIS_SPU/docs/SPU_CAPABILITY.md` —— 证据链文档（已按实测更新）
- `GIS_SPU/README.md` —— 项目说明（§5 API、§6 已验证算子、§7 限制）