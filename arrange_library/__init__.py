"""
arrange_library 顶层包。

对外暴露的核心函数:
- arrange_library: 端到端排机 + 调用 prediction_delivery 进行 Pooling 预测

对外暴露的异常:
- SchedulingTimeoutError: 排机超时（超过 10 分钟）时抛出，调用方可捕获并作为失败原因回推
"""

from .arrange_library_model6 import arrange_library, SchedulingTimeoutError

__all__ = ["arrange_library", "SchedulingTimeoutError"]

