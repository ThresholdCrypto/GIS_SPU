"""示例：航线与禁飞区冲突检查（课题首要验证目标）。

业务语义：
    Route_A | Intersects | NoFlyZone_B
必须能被表示为标准 Geo-IR。

编译：
    geo-secure build examples/route_conflict.py
"""

from geo_privacy import geo


def check_conflict(route_A, NoFlyZone_B):
    """航线 A 与禁飞区 B 是否冲突。"""
    return geo.intersects(route_A, NoFlyZone_B)