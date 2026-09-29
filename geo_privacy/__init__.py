"""geo_privacy：面向地理信息行业的低门槛隐私计算前端 API。

开发者只需：

    from geo_privacy import geo

    def check_conflict(route, no_fly_zone):
        return geo.intersects(route, no_fly_zone)

编译器负责：地理语义识别 → Geo-IR → 隐私计算算子选择 → JAX 代码生成 → SPU 模拟验证。
本包不导入 jax / spu / secretflow，可脱离隐私计算栈独立运行。
"""

from .core import CellSet, QuantVector, TimeInterval, quantize
from .geo import GEO_OPERATIONS, geo

__all__ = ["geo", "CellSet", "QuantVector", "TimeInterval", "quantize", "GEO_OPERATIONS"]
__version__ = "0.1.0"