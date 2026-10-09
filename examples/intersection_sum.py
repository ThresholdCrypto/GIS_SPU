"""示例：冲突格网数 + 冲突格网的风险权重和——PI-Sum 交集内求和档。

业务语义：
    (count, weight_sum) = (|Route_A ∩ NoFlyZone_B|, Σ Route_A 在交集内的权重)

它走的是与 libpsi 求交、PSI-CA 计数档并列的第三条 PSI 路径
（Google private-join-and-compute，Private Intersection-Sum）：协议只输出
「基数 + 和」两个数，交集本体不交给任一方。

编译（求和档；左侧输入必须带关联值）：
    geo-secure build examples/intersection_sum.py --psi-sum pjc \
        --psi-sum-weights route=examples/route_risk_weights.csv

缺值不会按 0 补齐——直接拒绝执行（理由见 docs/PSI_SUM_CAPABILITY.md）。
需要跨机部署时注意：上游两侧都走 gRPC LocalCredentials(LOCAL_TCP)，
本路径只在本机回环内跑。
"""

from geo_privacy import geo


def conflict_weight_sum(route, no_fly_zone):
    """航线与禁飞区的冲突格网基数，以及交集内的风险权重之和。"""

    return geo.cellset_intersect(route, no_fly_zone)
