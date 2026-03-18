"""
调度系统共享数据类型 - 避免循环导入
创建时间：2025-08-20 00:00:00
更新时间：2026-03-06 15:26:39
"""

import time
import math
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum

# 导入基础模型
# from liblane_paths import setup_liblane_paths
# setup_liblane_paths()

from arrange_library.models.library_info import EnhancedLibraryInfo, MachineType

class OptimizationObjective(Enum):
    """优化目标枚举"""
    MAXIMIZE_UTILIZATION = "maximize_utilization"
    MINIMIZE_LOAD_IMBALANCE = "minimize_load_imbalance"
    MAXIMIZE_PRIORITY_SCORE = "maximize_priority_score"

class SchedulingPhase(Enum):
    """调度阶段枚举"""
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    MULTI_OBJECTIVE_OPTIMIZATION = "multi_objective_optimization"
    HEURISTIC_IMPROVEMENT = "heuristic_improvement"
    QUALITY_VALIDATION = "quality_validation"

@dataclass
class MachineInfo:
    """机器信息模型"""
    machine_id: str
    machine_type: MachineType
    lane_capacity_gb: float
    total_lanes: int
    available_lanes: int = field(default_factory=lambda: 0)
    utilization_target: float = 0.9
    
    def __post_init__(self):
        if self.available_lanes == 0:
            self.available_lanes = self.total_lanes

@dataclass
class LaneAssignment:
    """Lane分配结果"""
    lane_id: str
    machine_id: str
    machine_type: MachineType
    libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    total_data_gb: float = 0.0
    utilization_rate: float = 0.0
    customer_library_ratio: float = 0.0
    priority_score: float = 0.0
    constraint_violations: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)  # 红线校验错误
    lane_capacity_gb: float = 250.0  # 添加实际Lane容量
    metadata: Dict[str, Any] = field(default_factory=dict)
    pooling_coefficients: Dict[str, float] = field(default_factory=dict)
    
    def add_library(self, library: EnhancedLibraryInfo):
        """添加文库到Lane"""
        self.libraries.append(library)
        self.total_data_gb += library.get_data_amount_gb()
        self.calculate_metrics()
    
    def remove_library(self, library: EnhancedLibraryInfo):
        """从Lane中移除文库"""
        if library in self.libraries:
            self.libraries.remove(library)
            self.total_data_gb -= library.get_data_amount_gb()
            self.calculate_metrics()
    
    def calculate_metrics(self):
        """计算Lane指标"""
        self._recalculate_metrics()
    
    def _recalculate_metrics(self):
        """重新计算Lane指标"""
        if not self.libraries:
            self.utilization_rate = 0.0
            self.customer_library_ratio = 0.0
            self.priority_score = 0.0
            return
            
        # 计算利用率 - 使用实际Lane容量
        self.utilization_rate = min(1.0, self.total_data_gb / self.lane_capacity_gb)
        
        # 计算客户文库占比
        customer_data = sum(lib.get_data_amount_gb() for lib in self.libraries if lib.is_customer_library())
        self.customer_library_ratio = customer_data / self.total_data_gb if self.total_data_gb > 0 else 0.0
        
        # 计算优先级分数
        urgent_libraries = sum(1 for lib in self.libraries if lib.is_urgent_priority())
        self.priority_score = urgent_libraries / len(self.libraries)

@dataclass
class SchedulingSolution:
    """调度解决方案"""
    lane_assignments: List[LaneAssignment] = field(default_factory=list)
    unassigned_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    total_libraries: int = 0
    assigned_libraries: int = 0
    overall_utilization: float = 0.0
    overall_balance_score: float = 0.0
    overall_priority_score: float = 0.0
    constraint_violations: List[str] = field(default_factory=list)
    analysis_summary: Dict[str, Any] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    feasible: bool = True
    
    def calculate_overall_metrics(self):
        """计算整体指标"""
        if not self.lane_assignments:
            return
            
        # 计算整体利用率
        total_utilization = sum(lane.utilization_rate for lane in self.lane_assignments)
        self.overall_utilization = total_utilization / len(self.lane_assignments)
        
        # 计算负载均衡分数
        utilizations = [lane.utilization_rate for lane in self.lane_assignments]
        mean_util = sum(utilizations) / len(utilizations)
        variance = sum((u - mean_util) ** 2 for u in utilizations) / len(utilizations)
        self.overall_balance_score = 1.0 / (1.0 + variance)  # 方差越小，均衡分数越高
        
        # 计算整体优先级分数
        total_priority = sum(lane.priority_score * len(lane.libraries) for lane in self.lane_assignments)
        total_libs = sum(len(lane.libraries) for lane in self.lane_assignments)
        self.overall_priority_score = total_priority / total_libs if total_libs > 0 else 0.0
        
        # 更新分配统计
        self.assigned_libraries = total_libs
        self.total_libraries = self.assigned_libraries + len(self.unassigned_libraries)

@dataclass
class SchedulingResult:
    """最终调度结果"""
    solution: SchedulingSolution
    accuracy_rate: float
    constraint_satisfaction_rate: float
    quality_score: float
    processing_time_seconds: float
    phase_timing: Dict[SchedulingPhase, float] = field(default_factory=dict)
    validation_details: Dict[str, Any] = field(default_factory=dict)
    improvement_suggestions: List[str] = field(default_factory=list)

@dataclass
class AsyncLaneAssignment:
    """异步调度Lane分配结果 - 测试兼容接口"""
    lane_id: str
    machine_id: str
    assigned_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)

@dataclass
class AsyncSchedulingResult:
    """异步调度结果 - 测试兼容接口"""
    success: bool = False
    lane_assignments: List[AsyncLaneAssignment] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None