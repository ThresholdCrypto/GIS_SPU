"""示例：文档写法（README 7.1.1）的高度带物化，直接写进可编译的源码。

与 vertical_conflict.py 的区别：那里把两个格网集合当**函数参数**收进来
（适合"数据在别处、由调用方注入"的形态）；这里直接在**模块顶层**用
`geo.height_band(...)` 物化，是文档里给业务侧看的那种写法。

这条路径曾经是**假验证**：模块顶层语句从不被遍历，方言表里也没有
`geo.height_band`，于是编译报成功、Intersects 显示 verified，
而高度带根本没进 Geo-IR。本示例的存在就是为了让它保持在"能跑通且有据"。

编译：
    geo-secure build examples/altitude_band.py

预期最终表：
    Operation   Representation  Backend    Status
    HeightBand  CompactCellSet  Plaintext  plaintext-local
    HeightBand  CompactCellSet  Plaintext  plaintext-local
    Intersects  CompactCellSet  PSI        verified
"""

from geo_privacy import geo

# 0-20000 m 机关航线占用带（L15 上 11 层）——明文在本方物化成层集合
route = geo.height_band(
    x=21861, y=27702, height_min=0, height_max=20000, level=15
)

# 5000-9000 m 临时管控带（L15 上 3 层）
control_zone = geo.height_band(
    x=21861, y=27702, height_min=5000, height_max=9000, level=15
)

# 高度带重叠 = 层集合有交 → 交给 PSI，不需要密态区间比较
conflict = geo.intersects(route, control_zone)

# 管控区是否完整覆盖航线带（含上下余量）
covered = geo.contains(control_zone, route)