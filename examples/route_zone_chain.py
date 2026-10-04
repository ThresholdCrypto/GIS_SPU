"""示例：链式执行——PSI 交集结果喂给下游算子（Phase 3 验收场景）。

业务语义：
    conflict = Route ∩ NoFlyZone        （CellSetIntersect，PSI 输出交集本体）
    result   = conflict ∩ SensitiveArea （Intersects，消费上一步的真实输出）

编译（绑定真实数据）：
    geo-secure build examples/route_zone_chain.py \
        --input route=examples/route_cells.csv \
        --input no_fly_zone=examples/nofly_cells.csv \
        --input sensitive_area=examples/sensitive_cells.csv
"""

from geo_privacy import geo


def check_zone_chain(route, no_fly_zone, sensitive_area):
    """先求航线与禁飞区的冲突格网，再看冲突格网是否触及敏感区。"""
    conflict = geo.cellset_intersect(route, no_fly_zone)
    return geo.intersects(conflict, sensitive_area)
