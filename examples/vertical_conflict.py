"""示例：低空三维高度带冲突与覆盖判定（GeoSOT-3D / GB/T 40087 附录 B）。

与 route_conflict.py 的区别：这里的两方都**带高度带**，因此判定是三维的。

业务语义（低空导航里真实会问的两个问题）：
    1. 航线占用的高度带与临时管控区的高度带是否重叠？   → Intersects
    2. 管控区是否完整覆盖航线的高度带（含上下余量）？   → Contains

关键机制：高度带在明文侧被物化成**层集合**（每层一个 64 位码），因此
"高度带重叠"退化为集合交，交给 PSI 即可，不需要额外的密态区间比较。

    同一 XY、不同高度层 = 不同的 64 位码 → 判为不相交（三维语义生效）
    同一 XY、高度带相交     = 两个层集合有交 → 判为相交

物化高度带的写法（业务侧不需要知道 GB 的层号，也直接写进可编译的源码）：

    from geo_privacy import geo
    route = geo.height_band(x=21861, y=27702, height_min=100, height_max=200,
                            level=15)
    zone  = geo.height_band(x=21861, y=27702, height_min=150, height_max=300,
                            level=15)

这两个物化结果是**明文算出的层集合**；编译后被登记为 Geo-IR 的
`HeightBand` 物化算子（Backend=Plaintext），密态边界落在下面
`geo.intersects` / `geo.contains` 这两步上（PSI）。
若把高度带写成 `level=23` 这种装不进 Z7 位域的层级，编译器会给出
第 6 类失败（HEIGHT_LAYER_UNSUPPORTED）而不是静默通过。

编译：
    geo-secure build examples/vertical_conflict.py
"""

from geo_privacy import geo


def check_height_conflict(route_band_cells, control_zone_cells):
    """航线高度带与管控区高度带是否冲突，以及管控区是否完整覆盖航线。"""
    overlap = geo.intersects(route_band_cells, control_zone_cells)
    covered = geo.contains(control_zone_cells, route_band_cells)
    return overlap, covered
