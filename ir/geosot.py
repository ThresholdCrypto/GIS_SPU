"""GB/T 40087-2021 网格剖分与高度域公式：本项目格网语义的底座。

本模块只做一件事——把国标 GB/T 40087-2021《地球空间网格编码规则》里
可直接核验的剖分与高度公式落成纯函数，供 Geo-IR 与算子层引用：

    附录 A 表 A.1   层级单元跨度 cell_deg(L)、每度网格数 cells_per_deg(L)
    附录 B 式(B.4)  高度域等比剖分：层 h 的下底面 H(h) = r0*((1+theta0)^(h*cell_deg) - 1)
    附录 B 式(B.7)  大地高 -> 层号（(B.4) 的严格逆）

为什么单独成模块
----------------
高度不是"附加属性"：低空空域分层管辖，同一 XY 上的不同高度带可能分属不同
管理方。本项目的 64 位定长键里 Z 占 7 位，此前只作为位域存在、不参与任何算子
语义（见 README 7.1.1 节）。把国标公式落成可调用、可复算的纯函数，是让 Z
参与语义的前提；公式全部有标准特征值可对拍，且不依赖任何第三方库。

口径说明（与 geosot_work-master 参考实现逐条对齐，分歧处已注明）
--------------------------------------------------------------
* cell_deg：L<=9 取 2^(9-L) 度；10~15 取 2^(15-L)/60 度（1 度 = 60 真实分口径）；
  16~21 取 2^(21-L)/3600 度；22~32 取 2^(32-L)/2048/3600 度。
* height_cell(L) 是【第 0 层】的厚度，不是层厚常数——层厚按 (1+theta0) 等比增长。
  参考实现 scope_geo_num3d 用 (h*hc, (h+1)*hc) 近似层边界，在 h>0 时偏小约
  h 个千分点；本模块按式(B.4)/(B.7) 取精确边界，保证 height_index 与
  height_layer_bounds 严格互逆。
* height_index 按式(B.7) 取真实 floor，可为负（地下层）。国标特征值 H_{-256}
  即是负层面；写入无符号 Z 位域时必须显式校验（见 validate_height_layer）。
  参考实现把 height<0 直接钳到 0，本模块保留真实负值，钳位责任交给编码侧。
"""

from __future__ import annotations

import math
from typing import Iterator

#: 赤道半径（米，WGS84），国标 (B.4)/(B.7) 中的 r0
R0 = 6378137.0

#: theta0 = pi/180，国标高度域等比剖分的公比底数
THETA0 = math.pi / 180.0

#: 国标最大剖分层级（全球尺度 L0 至厘米级 L32）
MAX_LEVEL = 32

#: 本题 64 位定长键里 Z 位域的位宽（X'17 | Y'17 | Z7 | L5 | Toff14 | Lt4）
HEIGHT_LAYER_BITS = 7
HEIGHT_LAYER_MAX = (1 << HEIGHT_LAYER_BITS) - 1  # 127


# --------------------------------------------------------------------------
# 附录 A 表 A.1：水平剖分
# --------------------------------------------------------------------------


def cell_deg(level: int) -> float:
    """层级单元在真实度单位下的跨度（2D 网格与高度公式共用）。

        L<=9     : 2^(9-L)                  度
        10 <= L <= 15 : 2^(15-L)/60          度（扩展分 -> 度）
        16 <= L <= 21 : 2^(21-L)/3600        度（实秒 -> 度）
        22 <= L <= 32 : 2^(32-L)/2048/3600   度（秒下 -> 度）
    """

    if not 0 <= level <= MAX_LEVEL:
        raise ValueError(f"层级必须落在 [0, {MAX_LEVEL}]，实得 {level!r}")
    if level <= 9:
        return 2.0 ** (9 - level)
    if level <= 15:
        return 2.0 ** (15 - level) / 60.0
    if level <= 21:
        return 2.0 ** (21 - level) / 3600.0
    return 2.0 ** (32 - level) / 2048.0 / 3600.0


def cells_per_deg(level: int) -> int:
    """每度网格数：L<=15 为 2^(L-9)，L>=16 为 3600*2^(L-21)。"""

    if not 0 <= level <= MAX_LEVEL:
        raise ValueError(f"层级必须落在 [0, {MAX_LEVEL}]，实得 {level!r}")
    if level <= 15:
        return 2 ** (level - 9)
    return 3600 * 2 ** (level - 21)


def equator_scale(level: int) -> float:
    """赤道附近的近似单元边长（米）：r0*theta0*cell_deg(L)。

    只用于量级报告与能力核查，不作为精确边长——精确边长随纬度变化。
    """

    return R0 * THETA0 * cell_deg(level)


# --------------------------------------------------------------------------
# 附录 B：高度域等比剖分
# --------------------------------------------------------------------------


def height_cell(level: int) -> float:
    """第 0 层的高度单元厚度（米）：r0*((1+theta0)^cell_deg - 1)，式(B.4) 的 h=1 情形。

    注意：这是"地面起算第 0 层"的厚度，不是层厚常数。层厚随层号等比增长
    （公比 (1+theta0)^cell_deg），低空段增长极小，高段不可忽略。
    """

    return R0 * ((1.0 + THETA0) ** cell_deg(level) - 1.0)


def height_layer_lower(layer: int, level: int) -> float:
    """层 `layer` 的下底面大地高（米），式(B.3)/(B.4) 精确口径。"""

    return R0 * ((1.0 + THETA0) ** (layer * cell_deg(level)) - 1.0)


def height_layer_bounds(layer: int, level: int) -> tuple[float, float]:
    """层 `layer` 的 [下底面, 上底面) 大地高区间（米）。"""

    return height_layer_lower(layer, level), height_layer_lower(layer + 1, level)


def height_index(height: float, level: int) -> int:
    """大地高（米） -> 高度层号，式(B.7)：floor(ln(1+H/r0)/ln(1+theta0)/cell_deg)。

    这是 (B.4) 的严格逆：height_layer_bounds(height_index(H, L), L) 恰好包含 H。
    返回值可为负（地下层）；H < -r0 无定义。
    """

    if height <= -R0:
        raise ValueError(f"大地高 {height} m 达到或低于 -r0，超出式(B.7) 定义域")
    k = math.log1p(height / R0) / math.log1p(THETA0) / cell_deg(level)
    return int(math.floor(k))


def height_layers(height_min: float, height_max: float, level: int) -> tuple[int, ...]:
    """高度带 [height_min, height_max]（米）覆盖的层号序列（含两端所在层）。

    这是"高度带 -> 层集合"的唯一入口。低空 0-1000 m 在 L=19 上得到 9 层，
    与国标三维格网按层枚举的口径一致。

    注意两端均按"所在层"计入：与区间端点同层的网格必须参与判定，否则
    0-300 m 与 300-600 m 这类首尾相接的带会被判为不相交。
    """

    if height_max < height_min:
        height_min, height_max = height_max, height_min
    lo = height_index(height_min, level)
    hi = height_index(height_max, level)
    return tuple(range(lo, hi + 1))


def iter_height_layers(height_min: float, height_max: float, level: int) -> Iterator[int]:
    """height_layers 的惰性版本（层数很大时用）。"""

    if height_max < height_min:
        height_min, height_max = height_max, height_min
    lo = height_index(height_min, level)
    hi = height_index(height_max, level)
    return iter(range(lo, hi + 1))


def height_layer_count(height_min: float, height_max: float, level: int) -> int:
    """高度带覆盖的层数（= 每个 XY 单元需要展开的 3D 网格条数）。"""

    if height_max < height_min:
        height_min, height_max = height_max, height_min
    return height_index(height_max, level) - height_index(height_min, level) + 1


# --------------------------------------------------------------------------
# 位域容量校验：7 位 Z 的适用边界
# --------------------------------------------------------------------------


def height_layer_bits(level: int, height_max: float) -> int:
    """把 [0, height_max] 带在层级 L 上编成层号所需的最小位宽。"""

    top = max(0, height_index(max(0.0, height_max), level))
    return max(1, top.bit_length())


def height_layer_fits(
    level: int, height_max: float, *, bits: int = HEIGHT_LAYER_BITS
) -> bool:
    """高度带 [0, height_max] 在层级 L 上是否装得进 `bits` 位 Z 位域。"""

    return height_layer_bits(level, height_max) <= bits


def validate_height_layer(
    layer: int, level: int, *, bits: int = HEIGHT_LAYER_BITS
) -> None:
    """校验层号可写入 Z 位域；越界即失败，绝不静默截断。"""

    limit = (1 << bits) - 1
    if layer < 0:
        raise ValueError(
            f"层号 {layer} 为负（地下层），无符号 Z{bits} 位域无法表示；"
            "本项目口径不支持负高度层"
        )
    if layer > limit:
        raise ValueError(
            f"层号 {layer} 超出 Z{bits} 位域上限 {limit}；"
            f"层级 L={level} 上的该高度需要更多位（见 height_layer_bits）"
        )


def max_supported_height(level: int, *, bits: int = HEIGHT_LAYER_BITS) -> float:
    """`bits` 位 Z 位域在层级 L 上能覆盖的最大大地高（米）。

    Z 位域是层号位域，层厚随层号增长，所以高度上限不是线性外推。
    """

    return height_layer_lower((1 << bits) - 1, level)
