# Geo-RR22 输入预处理说明（去重 / 排序 / 压缩 / 分桶已实现，候选集剪枝待实现）

> 状态：实现说明（更新于 2026-10-04）。任务文档 §8 / §9 / §23 Phase 6。
> 已实现：`backends/psi_backend/geosot_optimizer.py`（`GEOSOT_OPTIMIZER_VERSION = "0.1.0"`）——
> 排序 / 去重 / 无损前缀压缩 / bucket 分区；执行路径组合 `prepare_grid_codes`
> （校验 → 去重 → 排序）已接入 PSI 输入（`runtime.py` 的 `optimize_input=True`，默认开启）。
> 未实现：候选集剪枝（任务文档 §10），见第 7 节。
> 顺序纪律：**先基线，后优化**。基线见 `docs/BENCHMARK_PROTOCOL.md` 与
> `docs/psi_benchmark_baseline.json`（本仓库真机数据）。
> 本文件不声称任何性能收益——收益必须在本仓库基线上实测证明（任务文档 §25-10）。

## 1. 现状：通用 RR22 + 64-bit GridCode

RR22 接入闭环已完成（`docs/RR22_INTEGRATION.md`）：协议选择 → 参数装配 →
真机执行 → 对拍。输入预处理层（`geosot_optimizer`）已落地：去重（正确性前提）→
排序 → 无损前缀压缩 → bucket 分区，默认接入 PSI 输入路径；唯一尚未实现的是
候选集剪枝（§10）。

2026-09-30 基线实测（详见 `docs/BENCHMARK_PROTOCOL.md`）：

- RR22 在 N=2^24 端到端 ~50 s（求交本体 23.9 s）——"是否需要优化"有真实数据可依；
- **重复键**（N=2^12，两边各 25% 重复）：
  - `PROTOCOL_RR22` → 报错 `Paxos error, Duplicate keys were detected`；
  - `PROTOCOL_KKRT` → 报错 `Cannot find empty bin in stash`（cuckoo）；
  - `PROTOCOL_ECDH` → 可完成，但 `intersection_count` 含重复乘数
    （N=2^12 实测 3072 vs 唯一 2048；N=2^8 探针 192 vs 128），布尔语义不受影响；
    本环境还观察到紧邻失败用例之后的一次偶发 AllGather 超时（单独复测通过）。

> 结论：**去重不是"优化"，而是进入 RR22 / KKRT 的正确性前提**。预处理层
> （`geosot_optimizer`）必须存在，且其输出必须成为"双方约定的输入口径"。

## 2. 目标流水线（任务文档 §8）

```text
GeoSOT 层级结构
    ↓ GridCode 分布特征
  排序 → 去重 → 前缀 / 层级压缩 → bucket partition → 候选集剪枝 → RR22
```

## 3. 模块接口（`backends/psi_backend/geosot_optimizer.py`，已实现）

```python
sort_grid_codes(codes) -> tuple[int, ...]                # 确定性全序（保留重复）
deduplicate_grid_codes(codes) -> tuple[int, ...]         # 去重，保留首次出现顺序
prepare_grid_codes(codes) -> GridCodePreparation         # 校验 → 去重 → 排序（执行路径）
fingerprint_grid_codes(codes) -> str                     # sorted unique 集合 SHA-256 前 16 hex
compress_grid_codes(codes) -> GridCodeCompression        # 无损前缀压缩（expand() 可还原）
partition_grid_codes(codes, *, num_buckets=16) -> tuple[tuple[int, ...], ...]  # 按最高位分桶
```

**先于实现确定的硬约束**：

1. **确定性**：同一输入在任何机器上给出逐位一致的输出——跨方不一致的预处理
   等于静默算错（与 GridCode 布局握手同性质的教训，需引入 optimizer 版本 +
   指纹握手）。
2. **无损**：排序 / 去重 / 压缩 / 分桶不得改变集合语义：
   `optimized 交集 == baseline 交集`（逐元素）。
3. **剪枝严格**：候选剪枝只允许删除"双方可证不相交"的部分；近似不得进入
   精确路径（任务文档 §10）。
4. **去重是语义决定**：`deduplicate` 的输出即新的输入口径；对已有语义
   （集合交）不影响业务定义，但必须与对端同时启用（见第 1 条握手）。

> 现状：约束 1（确定性 + 版本 / 指纹）与约束 2（无损）已由
> `GridCodePreparation.to_dict()`（含 `version` / `fingerprint`）与
> `GridCodeCompression.expand()` 的还原对拍覆盖；约束 3（剪枝严格）尚未实现
> ——候选集剪枝是本层唯一未落地项。`partition_grid_codes` 只做划分、不丢弃任何码。

## 4. 结构信号与开放问题

可利用：

- 64 位码布局 `X17|Y17|Z7|L5|Toff14|Lt4`（`ir.values.GRID_CODE_BITS`，单一来源）；
- 同 XY 不同高度层已是不同码，"高度交叠"已归约为集合交（已验证，README §7.1.1）；
- 排序键天然存在（整型全序），对拍容易。

待研究（**不得先假设**）：

- 跨 L 的父 / 子 / 覆盖关系在本编码下的判定规则（是否存在前缀性质？
  需先对国标实现与实测验证，再谈候选剪枝）；
- bucket 划分（按 X/Y 高位）双方**私有集合**上的命中率 / 通信代价分布；
- 压缩表示（前缀 / 区间 / 位图）与 libpsi CSV 文件接口的衔接方式。

风险：

- 跨方不一致的压缩 = 静默错交集（历史教训同 GridCode 布局）；optimizer 产物
  必须带版本与指纹，进入 PSI 前校验；
- 过早优化会造成"结果对拍不上、原因难定位"——因此 Phase 6 的每一步都必须
  单独对拍、单独记录。

## 5. 对拍与验收（Phase 6）

```text
optimized_result == baseline_result   （逐元素）
```

- 优化路径与基线路径共用同一记录 schema（`benchmark.py`），
  新增字段只允许附录式扩展；
- 每个优化开关单独做正例 / 反例 / 边界（空集、`,`, ``, `）对拍；
- 无实测收益不得写进文档（§25-10）。

## 6. 边界（任务文档 §25，全部保留）

- 不改 libpsi / SPU 源码；不重新实现 OPRF / OKVS；
- 不改 JAX 后端；不把 protocol / RR22 参数放进 Geo-IR；
- 不模拟结果冒充真实 PSI；不把 DP 当精确。

## 7. 下一步执行清单

1. ~~`deduplicate_grid_codes` + 接入 PSI 输入路径~~——已完成（`prepare_grid_codes`，默认开启）；
2. ~~`sort_grid_codes` / `partition_grid_codes` 实现 + 对拍测试~~——已完成；
3. ~~评估"压缩 / 分桶"方向~~——压缩 / 分桶已实现并有无损还原测试；真实收益仍待基线实测；
4. **候选集剪枝（唯一剩余项）**：~~先出"覆盖关系判定"实证报告~~ ——**报告已出
   （2026-10-08，P0）**：`docs/GEO_RR22_COVERAGE.md`。结论是**当前口径下不可实现**：
   国标码确有前缀覆盖规则（21973 条真实码实测，低 22 位全零），但本项目 64 位码是
   `X17|Y17|Z7|L5|Toff14|Lt4` 的字段切分布局，`decode_grid_code` 解课题样例码得到
   `L=0/16`（不是 21）；且本项目没有声明 `(X, Y)` 是"本级索引"还是"共同分辨率索引"
   （同一个 `x=21861` 在 9 级与 15 级上被复用），覆盖规则因此**不可定义**。
   按任务文档 §10「近似不得进入精确路径」，**剪枝不实现**；
   下一步是接入层拍板口径，再用同一套检查复算。守卫测试 `tests/test_geo_rr22_coverage.py`
   （含"剪枝入口不得出现"一条）。
