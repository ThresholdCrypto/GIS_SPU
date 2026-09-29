"""示例：低空起降场选址风险评估打分。

业务语义：
    WeightedSum  → FixedPointVector → SPU/MPC（多属性加权评分）
    TemporalOverlap → TimeInterval  → MPC/SPU（时段可用性）
两者组合体现"链式算子"：打分结果与时间窗口共同决定候选可行性。

编译：
    geo-secure build examples/risk_score.py
"""

from geo_privacy import geo


def risk_score(risk_factors, weights, availability_window, operation_window):
    """风险加权总分 + 运行时段是否与可用窗口重叠。"""
    score = geo.weighted_sum(risk_factors, weights)
    window_ok = geo.temporal_overlap(availability_window, operation_window)
    return score, window_ok