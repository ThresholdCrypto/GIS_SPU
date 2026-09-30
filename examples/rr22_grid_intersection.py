"""示例：GeoSOT-3D 格网集合冲突判定，走 RR22（PSI 精确协议）。

链路（与 tests/test_rr22_geosot.py 中的链路对拍一致）：

    GeoSOT-3D 坐标/层号 → 64 位 grid_code（X17|Y17|Z7|L5|Toff14|Lt4）
        → CellSet（业务对象，geo_privacy.core）
        → Geo-IR（Intersects）
        → CompactCellSet（Planner 表征）
        → PSI 后端 → spu.libpsi RR22 → 交集 / 布尔

编译（显式选择 RR22；低通信模式是 RR22 专用参数）：

    geo-secure build examples/rr22_grid_intersection.py --psi-protocol RR22
    geo-secure build examples/rr22_grid_intersection.py --psi-protocol RR22 --psi-rr22-low-comm-mode

下方 64 位码由项目编码器生成（ir.values.encode_grid_code，toff=0 lt=4，L9），
可用 decode_grid_code 逐位核对；与对端对拍时布局必须逐位一致（见 README 3.5）。
业务侧通常不手写编码，而由平台编码器或 geo.height_band 物化产出。
"""

from geo_privacy import geo
from geo_privacy.core import CellSet

#: 航线格网：x=21861..21863, y=27702, 高度层号 z=15, L9
route_A_cells = CellSet(
    [
        3076691977860022276,  # x=21861
        3076832715348377604,  # x=21862
        3076973452836732932,  # x=21863
    ],
    label="Route_A",
)

#: 禁飞区格网：与航线共享 x=21862/21863（真交集 2 个），另有 x=22999
NoFlyZone_B_cells = CellSet(
    [
        3076832715348377604,  # x=21862
        3076973452836732932,  # x=21863
        3236851239608385540,  # x=22999
    ],
    label="NoFlyZone_B",
)


def check_conflict(route_A, NoFlyZone_B):
    """航线 A 与禁飞区 B 是否存在公共 64 位格网（三维：高度体现为 Z 层号位域）。"""
    return geo.intersects(route_A, NoFlyZone_B)