"""
arrange_library 顶层包。

对外暴露的核心函数:
- arrange_library: 端到端排机 + 调用 prediction_delivery 进行 Pooling 预测
"""

from .arrange_library_model6 import arrange_library

__all__ = ["arrange_library"]

