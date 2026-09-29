"""示例：两点距离是否小于安全间隔。

业务语义：DistanceLE → QuantizedVector → MPC/SPU
这是首个端到端验证的算子（明文 / JAX / SPU 三方对拍）。

编译：
    geo-secure build examples/distance_check.py
"""

from geo_privacy import geo


def check_distance(p1, p2, threshold):
    """两架飞行器的水平间隔是否不超过安全阈值。"""
    return geo.distance_le(p1, p2, threshold)