# RR22 Integration（Geo-IR → PSI Backend → SPU/libpsi RR22）

> 本文件只记录**跑过真机**的接入闭环。环境：WSL2 Ubuntu / Python 3.11.16 /
> `spu==0.9.5`；实测日期 2026-09-30。
> 边界：GIS_SPU **不实现** RR22 密码学算法（OPRF/OKVS 等都由上游 libpsi 提供），
> 也不修改 SPU 源码；本仓库负责协议选择、参数装配与运行验证。

## 1. Architecture

```text
Geo Application           geo.intersects / geo.cellset_intersect（业务 API）
    ↓
Geo-IR                     Intersects / CellSetIntersect（不含任何协议字段）
    ↓
Planner                    representation=CompactCellSet, backend=PSI,
                           protocol=PROTOCOL_RR22, protocol_params={...}
    ↓
PSI Backend                backends/psi_backend/runtime.py
    ↓
SPU 调用层                 spu.libspu.link.create_mem（两方进程内链路）
                           psi.PsiProtocolConfig(rr22_params=psi.Rr22Rarams(...))
    ↓
libpsi                     spu.psi.psi_execute → RR22（日志可见 Rr22PsiSender / Rr22PsiReceiver）
```

职责边界（与课题 §33 一致）：Geo-IR 描述"算什么"；Planner 决定"用什么表示、
什么后端、什么协议"；PSI Backend 负责"怎么调用 PSI"；SPU/libpsi 负责"怎么执行"；
RR22 是"具体密码协议"。Geo-RR22（三维格网优化）是下一阶段，不属于本文件的接入层。

## 2. Current Capability

| 协议 | 两方可执行 | 结果精度 | 曲线关系 |
|------|-----------|----------|----------|
| `PROTOCOL_ECDH` | 是 | 精确 | `required`（本项目注入 SM2） |
| `PROTOCOL_KKRT` | 是 | 精确 | `ignored` |
| **`PROTOCOL_RR22`** | **是（本版接入）** | **精确** | **`ignored`** |
| `PROTOCOL_DP` | 是 | 带噪（不可做一致性验证） | `implicit` |

能力探测（`check_psi_capabilities().rr22_params`，结论来自**实际构造**）：

```json
{
  "protocol": "PROTOCOL_RR22",
  "protocol_present": true,
  "params": {"rr22_params": true, "low_comm_mode": true},
  "runnable": true
}
```

若上游版本没有 `Rr22Rarams`：探测如实置 `false` 并给出原因；运行期显式请求
`low_comm_mode=True` 会**失败**（不能假装设置成功），未请求则按协议内部默认配置
执行并在结果里加注、清空参数档。

## 3. RR22 Parameters

| 参数 | 取值 | 含义 | 本项目入口 |
|------|------|------|------------|
| `low_comm_mode` | `false`（默认） | RR22 默认通信模式 | `--psi-protocol RR22` |
| `low_comm_mode` | `true` | 低通信模式 | `--psi-protocol RR22 --psi-rr22-low-comm-mode` |

本机两种取值均真实执行且与明文一致（`agreement=True`）。`--psi-rr22-low-comm-mode`
是 **RR22 专用**开关：配其它协议时会打印提示，且参数不注入、不登记。

## 4. Runtime Path

```python
# backends/psi_backend/runtime.py（worker 内，RR22 分支）
if protocol_name == "PROTOCOL_RR22":
    protocol_conf.rr22_params = psi.Rr22Rarams(
        low_comm_mode=bool(rr22_low_comm_mode)
    )
```

顺序纪律（不猜 API）：

1. 进入线程前用 `hasattr(psi, "Rr22Rarams")` 核对参数类真实存在；
2. 存在 → 按调用方给的实值注入，并写入 `PsiRunResult.protocol_params`
   （`{"low_comm_mode": ...}`）；
3. 不存在且显式请求 `True` → `status="error"`，原因指向能力报告；
4. 不存在且未显式请求 → 不注入、清空参数档、`notes` 如实说明。

RR22 与 ECDH curve 严格分开：`RR22 = ignored`，**不创建 `EcdhParams`**，绝不把
SM2 注入 RR22；`PSI_CURVE_RELATION` 表未改动。

## 5. Input Representation

```text
GeoSOT-3D 坐标/层号 → 64 位 grid_code（X17|Y17|Z7|L5|Toff14|Lt4）
    → CellSet（业务层集合对象）
    → CompactCellSet（Planner 给集合族算子的计算表征）
    → PSI（CSV 交换：keys=["grid_code"], keys_unique=True, INNER_JOIN）
```

- 64 位**无符号**交换：`>= 2^63` 的格网码不截断（有回归测试）；
- `CellSet` 是业务对象、`CompactCellSet` 是规划表征，不合并成两个数据模型；
- 对端布局必须与本机逐位一致（`GRID_CODE_LAYOUT`，见 README 3.5）。

## 6. Leakage

沿用 PSI 标准语义：**接收方获得交集本体**（不只是布尔值），交集基数对 recipient
可见。RR22 不改变泄漏面：`reveals` 字段照常带出（协议换名不等于披露消失）。
`Intersects` 的说明保持"本可比布尔值泄露更多"的原话。

## 7. Verification

对拍 = 明文参考（`CellSet` 集合交 / plain 后端）vs RR22 真实结果：

| 文件 | 覆盖 |
|------|------|
| `tests/test_psi_backend.py` | RR22 声明、参数能力探测、`Intersects` / `CellSetIntersect` 与明文一致、`>= 2^63` 高位码、`low_comm_mode` 双取值（共 9 项） |
| `tests/test_rr22_geosot.py` | GeoSOT-3D → grid_code → CellSet → CompactCellSet → RR22（相交/不相交/相同/空集/高位码/重复/排序，共 8 项） |
| `tests/test_end_to_end.py` | CLI `--psi-protocol RR22 [--psi-rr22-low-comm-mode]` 与参数披露（2 项） |

RR22 属**精确**协议：与明文不一致会被判 `error`（`protocol_is_exact`）；
带噪的 `PROTOCOL_DP` 行为不受影响。

## 8. Future Work

Geo-RR22 优化（**下一阶段**，与协议接入分开）：

1. 64-bit GridCode 专用编码
2. 排序 / 去重
3. 前缀压缩
4. 分层格网集合压缩
5. bucket partition
6. 地理候选集剪枝
7. communication cost estimation
8. RR22 与 GeoSOT 层级结构联合优化

当前不该做的事（课题 §27）：不改 JAX Backend、不改 SPU MPC `ProtocolKind`、
不在 GIS_SPU 重写 OPRF/OKVS、不给 Geo-IR 加协议字段。