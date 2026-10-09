"""示例：只问"冲突格网有多少个"——PSI-Cardinality 计数档。

业务语义：
    conflict_count = |Route_A ∩ NoFlyZone_B|   （基数；不产出交集本体）

它走的是与 libpsi 求交并列的第二条 PSI 路径（OpenMined PSI-Cardinality）：
协议只计算并返回计数，交集本体不交给任一方。

编译（计数档）：
    geo-secure build examples/conflict_count.py --psi-count psi-ca

同一份代码在缺省（libpsi 求交）档下返回交集本体；两种档位的泄漏承诺不同，
按业务需要选择（见 docs/PSI_CA_CAPABILITY.md）。
"""

from geo_privacy import geo


def conflict_count(route, no_fly_zone):
    """航线与禁飞区的冲突格网基数。"""

    return geo.cellset_intersect(route, no_fly_zone)
