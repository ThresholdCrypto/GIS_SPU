"""pytest 根配置：把 GIS_SPU 加入导入路径，使 ir / frontend / geosecure 等可直接导入。"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)