"""
逐Lane贪心排机器 - 简化高效的排机算法
采用逐Lane填充策略，一条Lane排满后再排下一条
创建时间：2025-12-24 13:30:00
更新时间：2026-04-07 15:21:20

核心思想：
- 按优先级排序文库
- 逐条Lane进行填充
- 每加入一个文库就进行约束检查（单Lane内）
- 集成碱基不均衡处理器和AI规则检查器
- 复杂度：O(n × m)，其中n是文库数，m是Lane数

集成模块：
- BaseImbalanceHandler: 碱基不均衡文库分组识别和混排规则检查
- RuleChecker: 完整的24条排机规则检测（文库对级别+Lane级别）

变更记录：
- 2026-01-30: 调整碱基不均衡和特殊文库限制：
  - max_imbalance_ratio: 35%（从30%调整）
  - max_special_library_data_gb: 350G（从240G调整）
- 2025-12-31: GreedyLaneConfig按机型从scheduling_config自动加载容量/阈值，默认回归规则文档取值
- 2025-12-26: 调整Lane容量范围为严格配置：
  - min_utilization: 0.99（965GB）
  - max_utilization: 1.01（985GB）
- 2025-12-25: 根据人工排机数据分析，调整了以下参数：
  - max_special_library_data_gb: 从240G提高到350G（人工排机碱基不均衡占比最高达30%）
  - max_imbalance_ratio: 新增参数，控制碱基不均衡占比上限（默认40%）
  - enable_dedicated_imbalance_lane: 新增参数，支持碱基不均衡专用Lane策略
"""

import random
import time
from typing import Any, List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field, replace
from loguru import logger

# from liblane_paths import setup_liblane_paths
# setup_liblane_paths()

from arrange_library.core.ai.pooling_coefficient_optimizer import (
    PoolingCoefficientOptimizer,
    PoolingOptimizationResult,
)
from arrange_library.core.config.scheduling_config import get_scheduling_config, SchedulingMode
from arrange_library.models.library_info import EnhancedLibraryInfo, MachineType
from arrange_library.core.constraints.index_validator_verified import IndexConflictValidator
from arrange_library.core.constraints.lane_validator import LaneValidator
from arrange_library.core.scheduling.scheduling_types import LaneAssignment, SchedulingSolution
from arrange_library.core.preprocessing.base_imbalance_handler import BaseImbalanceHandler
from arrange_library.core.preprocessing.batch_rule_analyzer import BatchRuleAnalyzer, BatchAnalysisReport
from arrange_library.core.preprocessing.library_splitter import LibrarySplitter
from arrange_library.core.preprocessing.rule_constrained_strategy_planner import (
    RuleConstrainedStrategyPlanner,
    StrategyExecutionPlan,
)
from arrange_library.core.validation.rule_checker import RuleChecker


@dataclass
class GreedyLaneConfig:
    """贪心排机配置（遵循红线规则）

    默认值对齐《排机规则文档》，并支持按机器类型从 scheduling_config 获取容量与阈值。
    """
    # 是否按机器类型自动加载配置
    use_machine_config: bool = True

    # Lane容量配置（单位：GB）- 默认使用Nova X-25B PE150策略作为回退值
    lane_capacity_gb: float = 985.0
    
    # ===== Lane容量利用率范围 =====
    # 容量下限：980G（≈99.5%利用率，基于985G标称容量）
    min_utilization: float = 0.995
    
    # 容量上限：990G（≈100.5%利用率，基于985G标称容量）
    max_utilization: float = 1.005
    
    # 客户文库最大占比（红线规则：<=50%）
    max_customer_ratio: float = 0.50
    # 10bp Index最小占比（红线规则：>=40%，混排时）
    min_10bp_index_ratio: float = 0.40
    # 特殊文库类型最大数量
    max_special_library_types: int = 3
    
    # ===== 碱基不均衡数据量限制 =====
    # [2026-01-30 修正] 调整为：Nova X-25B 特殊文库总量 350G
    max_special_library_data_gb: float = 350.0
    
    # ===== 碱基不均衡占比上限 =====
    # [2026-01-30 修正] 调整为：3.6T-NEW模式碱基不均衡占比 ≤35%
    max_imbalance_ratio: float = 0.35
    
    # 是否启用Index冲突检查
    enable_index_check: bool = True
    # 是否启用碱基不均衡检查（使用BaseImbalanceHandler）
    enable_imbalance_check: bool = True
    # 是否启用完整规则检查（使用RuleChecker）
    enable_rule_checker: bool = True
    # 碱基不均衡文库同Lane最大类型数（规则文档：<=3种）
    max_imbalance_types_per_lane: int = 3
    
    # ===== [2025-12-25 新增] 碱基不均衡专用Lane策略 =====
    # [人工排机实际] 5条Lane为100%碱基不均衡专用Lane，消化4860GB数据
    enable_dedicated_imbalance_lane: bool = True
    
    # ===== [2025-12-25 新增] 小文库聚类专用Lane策略 =====
    # [人工排机实际] 大量同尺寸小文库（如886个5-6G文库）集中到专用Lane
    # 策略：识别同数据量区间的小文库群，优先为它们形成专用Lane
    enable_small_library_clustering: bool = True
    # 小文库阈值（<=此值视为小文库）
    small_library_threshold_gb: float = 20.0
    # 聚类最小文库数（同尺寸文库数达到此值才形成专用Lane）
    # 降低阈值以捕获更多聚类（从50降到30）
    clustering_min_count: int = 30
    # 聚类数据量区间宽度（如2.0表示5-7G为一个区间，更宽松）
    clustering_bin_width_gb: float = 2.0
    
    # ===== [2025-12-25 新增] 非10bp专用Lane策略 =====
    # [规则文档] 规则4b: Lane中只含非10碱基文库时，也不做限制
    # [人工排机实际] 5条Lane是100%非10bp文库（6/8碱基Index）
    enable_non_10bp_dedicated_lane: bool = True
    
    # ===== [2025-12-25 新增] 大带小预留策略 =====
    # [人工排机实际] 人工排机会预留大文库作为"骨架"，用于携带小文库
    # 策略：
    # 1. 统计小文库总量，计算需要预留的大文库数据量
    # 2. 在普通排机时，保留部分大文库不参与
    # 3. 最后阶段，用预留大文库+剩余小文库组成Lane
    enable_backbone_reservation: bool = True
    # 小文库阈值（用于计算预留量）- 与小文库聚类策略共用
    # small_library_threshold_gb: float = 20.0  # 已在上面定义
    # 预留骨架的安全系数（预留量 = 小文库缺口 × 系数）
    backbone_safety_factor: float = 1.2

    def resolve_for_machine(
        self,
        machine_type: str,
        mode: SchedulingMode = SchedulingMode.NON_1_0,
    ) -> "GreedyLaneConfig":
        """按机型生成有效配置（容量/阈值来自 scheduling_config）"""
        if not self.use_machine_config:
            return self

        scheduling_config = get_scheduling_config()
        lane_capacity = scheduling_config.get_lane_capacity(machine_type, mode)
        tolerance = scheduling_config.lane_capacity.standard_tolerance

        if lane_capacity > 0:
            min_utilization = (lane_capacity - tolerance) / lane_capacity
            max_utilization = (lane_capacity + tolerance) / lane_capacity
        else:
            min_utilization = self.min_utilization
            max_utilization = self.max_utilization

        validation_limits = scheduling_config.validation_limits
        return replace(
            self,
            lane_capacity_gb=lane_capacity or self.lane_capacity_gb,
            min_utilization=min_utilization,
            max_utilization=max_utilization,
            max_customer_ratio=validation_limits.customer_ratio_limit,
            min_10bp_index_ratio=validation_limits.index_10bp_ratio_min,
            max_special_library_types=validation_limits.special_library_type_limit,
            max_special_library_data_gb=scheduling_config.get_special_library_limit(machine_type),
            max_imbalance_ratio=validation_limits.base_imbalance_ratio_limit,
            max_imbalance_types_per_lane=validation_limits.special_library_type_limit,
        )


class GreedyLaneScheduler:
    """
    逐Lane贪心排机器
    
    算法流程：
    1. 按优先级和数据量对文库排序
    2. 创建空Lane
    3. 遍历文库，尝试放入当前Lane
       - 使用LaneValidator进行完整红线规则校验
       - 使用BaseImbalanceHandler进行碱基不均衡混排检查
       - 使用RuleChecker进行文库对兼容性检查
    4. 当前Lane满了，创建新Lane继续
    5. 所有文库处理完毕，返回结果
    
    优点：
    - 复杂度低：O(n × m)
    - 使用与后置校验相同的验证器，确保规则一致
    - 集成碱基不均衡处理，确保分组混排规则
    - 集成完整规则检查，覆盖24条排机规则
    - 逻辑清晰，易于调试
    """
    
    def __init__(self, config: Optional[GreedyLaneConfig] = None):
        """初始化贪心排机器
        
        初始化以下验证器和处理器：
        1. IndexConflictValidator - Index冲突检测
        2. LaneValidator - Lane红线规则校验
        3. BaseImbalanceHandler - 碱基不均衡分组和混排规则
        4. RuleChecker - 完整24条规则检测
        """
        self._base_config = config or GreedyLaneConfig()
        self.config = self._base_config
        
        # Lane ID计数器（按前缀和机器类型分别计数，避免多轮排机时ID重复）
        # 格式: {"GL_Nova X-25B": 5, "DL_Nova X-25B": 3, ...}
        self._lane_counters: Dict[str, int] = {}
        self.scheduling_config = get_scheduling_config()
        
        # 核心验证器
        self.index_validator = IndexConflictValidator()
        self.lane_validator = LaneValidator(strict_mode=True)
        
        # 碱基不均衡处理器 - 用于识别文库分组和检查混排规则
        self.imbalance_handler = BaseImbalanceHandler() if self.config.enable_imbalance_check else None
        
        # 完整规则检查器 - 用于文库对兼容性和Lane级别规则检测
        self.rule_checker = RuleChecker() if self.config.enable_rule_checker else None

        # Pooling优化器
        self.pooling_optimizer = PoolingCoefficientOptimizer()
        self.library_splitter = LibrarySplitter()
        
        # 记录配置信息
        config_mode = "按机型自动" if self._base_config.use_machine_config else "固定"
        logger.info(f"逐Lane贪心排机器初始化完成 - 基础容量: {self.config.lane_capacity_gb}GB, "
                   f"利用率区间: [{self.config.min_utilization:.0%}, {self.config.max_utilization:.0%}], "
                   f"配置模式: {config_mode}")
        logger.info(f"启用检查: Index={self.config.enable_index_check}, "
                   f"碱基不均衡={self.config.enable_imbalance_check}, "
                   f"规则检查器={self.config.enable_rule_checker}")

        self._batch_analysis_report: Optional[BatchAnalysisReport] = None
        self._strategy_plan: Optional[StrategyExecutionPlan] = None

    def _run_batch_analysis(
        self, libraries: List[EnhancedLibraryInfo]
    ) -> Optional[BatchAnalysisReport]:
        """在排机前执行全局批次分析和策略规划

        如果外部已通过设置 _batch_analysis_report 和 _strategy_plan 传入全局分析结果，
        则跳过内部分析（避免在部分数据上重复分析导致结论不准确）。
        """
        if self._batch_analysis_report is not None:
            logger.info("使用外部传入的全局分析报告（全量文库），跳过内部分析")
            return self._batch_analysis_report
        if not libraries:
            return None
        try:
            analyzer = BatchRuleAnalyzer(self.scheduling_config)
            machine_type = str(getattr(libraries[0], "eq_type", "") or "")
            report = analyzer.analyze(libraries, machine_type)
            for line in report.summary_lines():
                logger.info(line)

            planner = RuleConstrainedStrategyPlanner()
            self._strategy_plan = planner.plan(report)
            return report
        except Exception as exc:
            logger.warning(f"批次全局分析异常，不影响排机: {exc}")
            return None

    def _resolve_lane_capacity_rule(
        self,
        libraries: Optional[List[EnhancedLibraryInfo]],
        machine_type: str,
        lane: Optional[LaneAssignment] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """按统一配置表解析当前Lane上下文的容量规则。"""
        validation_metadata: Dict[str, Any] = {}
        if lane is not None:
            validation_metadata.update(self._build_lane_validation_metadata(lane))
        if metadata:
            validation_metadata.update(metadata)
        if libraries:
            first_lib = libraries[0]
            validation_metadata.setdefault('process_code', getattr(first_lib, 'process_code', None))
            validation_metadata.setdefault('test_code', getattr(first_lib, 'test_code', None))
            validation_metadata.setdefault('test_no', getattr(first_lib, 'test_no', ''))
        return self._get_scheduling_lane_capacity_range(
            libraries=libraries or [],
            machine_type=machine_type,
            metadata=validation_metadata,
        )

    @staticmethod
    def _apply_scheduling_capacity_cap(selection: Any) -> Any:
        """仅在排机阶段收紧标准25B规则上限，不影响LaneValidator。"""
        rule_code = str(getattr(selection, "rule_code", "") or "")
        if rule_code in {"tj_1595_standard_pe150_25b", "tj_1595_standard_pe150_25b_other"}:
            selection = replace(
                selection,
                max_target_gb=min(float(selection.max_target_gb), 1100.0),
                effective_max_gb=min(float(selection.effective_max_gb), 1105.0),
            )
        return selection

    def _get_scheduling_lane_capacity_range(
        self,
        libraries: List[EnhancedLibraryInfo],
        machine_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """获取仅用于排机的容量规则，允许比校验规则更保守。"""
        selection = self.scheduling_config.get_lane_capacity_range(
            libraries=libraries,
            machine_type=machine_type,
            metadata=metadata,
        )
        return self._apply_scheduling_capacity_cap(selection)

    def _resolve_lane_capacity_limits(
        self,
        libraries: Optional[List[EnhancedLibraryInfo]],
        machine_type: str,
        lane: Optional[LaneAssignment] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """获取统一配置表解析后的Lane有效容量上下限。"""
        selection = self._resolve_lane_capacity_rule(
            libraries=libraries,
            machine_type=machine_type,
            lane=lane,
            metadata=metadata,
        )
        return float(selection.effective_min_gb), float(selection.effective_max_gb)
    
    def _get_next_lane_id(self, prefix: str, machine_type: str) -> str:
        """获取下一个Lane ID，确保唯一性
        
        Args:
            prefix: Lane前缀，如 "GL", "DL", "NB", "SL", "BL"
            machine_type: 机器类型，如 "Nova X-25B"
            
        Returns:
            唯一的Lane ID，格式如 "GL_Nova X-25B_001"
        """
        counter_key = f"{prefix}_{machine_type}"
        if counter_key not in self._lane_counters:
            self._lane_counters[counter_key] = 0
        self._lane_counters[counter_key] += 1
        return f"{prefix}_{machine_type}_{self._lane_counters[counter_key]:03d}"
    
    def _reset_lane_counters(self):
        """重置Lane计数器（每次schedule调用前重置）"""
        self._lane_counters = {}

    @staticmethod
    def _get_library_runtime_key(lib: EnhancedLibraryInfo) -> str:
        """获取调度阶段识别文库对象的稳定键，拆分子文库必须彼此不同。"""
        for attr_name in ("_detail_output_key", "fragment_id", "wkaidbid", "aidbid"):
            value = str(getattr(lib, attr_name, "") or "").strip()
            if value:
                return value
        source_key = str(
            getattr(lib, "_origrec_key", None) or getattr(lib, "origrec", "") or ""
        ).strip()
        if source_key:
            return source_key
        return str(id(lib))

    def _resolve_machine_type_enum(
        self,
        machine_type: str,
        libraries: Optional[List[EnhancedLibraryInfo]] = None,
    ) -> MachineType:
        """解析机器类型枚举，兼容不同eq_type写法"""
        if machine_type:
            try:
                return MachineType(machine_type)
            except ValueError:
                pass

        if libraries:
            enum = libraries[0].get_machine_type_enum()
            if enum != MachineType.UNKNOWN:
                return enum

        return MachineType.UNKNOWN

    def _split_imbalance_by_customer(
        self, libraries: List[EnhancedLibraryInfo]
    ) -> Tuple[List[EnhancedLibraryInfo], List[EnhancedLibraryInfo]]:
        """按客户/内部拆分文库列表"""
        customer_libs: List[EnhancedLibraryInfo] = []
        internal_libs: List[EnhancedLibraryInfo] = []
        for lib in libraries:
            if lib.is_customer_library():
                customer_libs.append(lib)
            else:
                internal_libs.append(lib)
        return customer_libs, internal_libs

    def _build_imbalance_dedicated_lanes(
        self, failed: List[EnhancedLibraryInfo], machine_type: MachineType
    ) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo]]:
        """
        将碱基不均衡文库优先组专用Lane，并回退中性文库。

        该方法复用 `_schedule_dedicated_imbalance_lanes` 的统一规则：
        - 目标碱基不均衡数据量 = Lane容量 × 分组数据量占比
        - 目标平衡文库数据量 = Lane容量 - 碱基不均衡数据量
        """
        imbalance_libs = [lib for lib in failed if lib.is_base_imbalance()]
        neutral_libs = [lib for lib in failed if not lib.is_base_imbalance()]
        if not imbalance_libs:
            return [], failed

        machine_type_str = machine_type.value if isinstance(machine_type, MachineType) else str(machine_type)
        lanes, remaining_imbalance = self._schedule_dedicated_imbalance_lanes(
            imbalance_libs=imbalance_libs,
            machine_type=machine_type_str,
        )
        remaining = remaining_imbalance + neutral_libs
        logger.info(f"专用Lane创建完成: {len(lanes)}条Lane, {len(remaining_imbalance)}个碱基不均衡文库待分配, {len(neutral_libs)}个中性文库待分配")
        return lanes, remaining

    def schedule(
        self,
        libraries: List[EnhancedLibraryInfo],
        keep_failed_lanes: bool = False,
        libraries_already_split: bool = False,
        perform_presplit_family_rollback: bool = True,
    ) -> SchedulingSolution:
        """
        执行逐Lane贪心排机
        
        Args:
            libraries: 待排机文库列表
            keep_failed_lanes: 是否保留验证失败的Lane（不拆解），供外层矫正处理
            libraries_already_split: 外层是否已完成预拆分
            perform_presplit_family_rollback: 是否在调度器内部执行拆分家族回滚
            
        Returns:
            SchedulingSolution: 排机结果
        """
        start_time = time.time()
        
        # 重置Lane计数器
        self._reset_lane_counters()
        
        if not libraries:
            logger.warning("输入文库列表为空")
            return SchedulingSolution(lane_assignments=[], unassigned_libraries=[])
        
        logger.info(f"开始逐Lane贪心排机 - 文库数: {len(libraries)}")

        # 0. 先按业务规则执行文库拆分，再进入后续排机分析与调度
        if libraries_already_split:
            working_libraries = list(libraries)
            split_records: List[dict] = []
        else:
            working_libraries, split_records = self.library_splitter.split_libraries(libraries)
        presplit_family_context = self._build_presplit_family_context(working_libraries)
        if libraries_already_split:
            logger.info("预拆分完成: 使用外层已拆分文库")
        elif split_records:
            logger.info(
                "预拆分完成: 原始文库{}个，触发拆分{}个，拆分后文库{}个",
                len(libraries),
                len(split_records),
                len(working_libraries),
            )
        else:
            logger.info("预拆分完成: 无需拆分")
        
        # 1. 全局规则分析 - 在排机前对整批文库做全局特征分析
        self._batch_analysis_report = self._run_batch_analysis(working_libraries)
        
        # 2. 按优先级和数据量排序文库
        sorted_libs = self._sort_libraries(working_libraries)
        
        # 3. 按机器类型分组
        machine_groups = self._group_by_machine_type(sorted_libs)
        
        all_lanes: List[LaneAssignment] = []
        unassigned: List[EnhancedLibraryInfo] = []
        
        # 4. 对每个机器类型组进行排机
        for machine_type, libs in machine_groups.items():
            machine_type_enum = self._resolve_machine_type_enum(machine_type, libs)
            machine_type_key = (
                machine_type_enum.value
                if machine_type_enum != MachineType.UNKNOWN
                else (machine_type or MachineType.NOVA_X_25B.value)
            )
            self.config = (
                self._base_config.resolve_for_machine(machine_type_key)
                if self._base_config.use_machine_config
                else self._base_config
            )
            logger.info(
                f"处理机器类型 {machine_type} - 文库数: {len(libs)} - "
                f"Lane容量: {self.config.lane_capacity_gb}GB, "
                f"利用率区间: [{self.config.min_utilization:.0%}, {self.config.max_utilization:.0%}], "
                f"特殊文库上限: {self.config.max_special_library_data_gb:.0f}GB"
            )
            
            # ===== 碱基不均衡专用Lane策略（规则约束决策） =====
            enable_dedicated = (
                self._strategy_plan.enable_dedicated_imbalance_lane
                if self._strategy_plan is not None
                else getattr(self.config, 'enable_dedicated_imbalance_lane', False)
            )
            if enable_dedicated and self.imbalance_handler:
                # 分离碱基不均衡文库和碱基均衡文库
                imbalance_libs = []
                balanced_libs = []
                for lib in libs:
                    jjbj = getattr(lib, 'jjbj', None)
                    is_imbalance = (jjbj is not None and str(jjbj).strip() == '是')
                    if is_imbalance:
                        imbalance_libs.append(lib)
                    else:
                        balanced_libs.append(lib)
                
                if imbalance_libs:
                    logger.info(f"碱基不均衡专用Lane策略: {len(imbalance_libs)}个碱基不均衡文库")
                    # 先为碱基不均衡文库形成专用Lane
                    dedicated_lanes, remaining_imbalance = self._schedule_dedicated_imbalance_lanes(
                        imbalance_libs, machine_type
                    )
                    all_lanes.extend(dedicated_lanes)
                    logger.info(f"形成{len(dedicated_lanes)}条专用Lane，剩余{len(remaining_imbalance)}个碱基不均衡文库")
                    
                    # 剩余的碱基不均衡文库和碱基均衡文库合并，进行混排
                    libs = remaining_imbalance + balanced_libs
                    # 重新排序
                    libs = self._sort_libraries(libs)
            
            # ===== 非10bp专用Lane策略（规则约束决策） =====
            enable_non_10bp = (
                self._strategy_plan.enable_non_10bp_dedicated_lane
                if self._strategy_plan is not None
                else getattr(self.config, 'enable_non_10bp_dedicated_lane', False)
            )
            if enable_non_10bp:
                non_10bp_lanes, remaining_after_non_10bp = self._schedule_non_10bp_dedicated_lanes(
                    libs, machine_type
                )
                if non_10bp_lanes:
                    all_lanes.extend(non_10bp_lanes)
                    libs = remaining_after_non_10bp
                    logger.info(f"非10bp专用Lane策略: 形成{len(non_10bp_lanes)}条专用Lane")
            
            # ===== 大带小预留策略（规则约束决策） =====
            enable_backbone = (
                self._strategy_plan.enable_backbone_reservation
                if self._strategy_plan is not None
                else getattr(self.config, 'enable_backbone_reservation', False)
            )
            reserved_backbone: List[EnhancedLibraryInfo] = []
            if enable_backbone:
                # 计算需要预留的骨架数据量
                backbone_needed, _ = self._calculate_backbone_reservation(libs)
                if backbone_needed > 0:
                    # 选择骨架文库并预留
                    reserved_backbone, libs = self._select_backbone_libraries(libs, backbone_needed)
                    logger.info(f"骨架预留: 预留{len(reserved_backbone)}个大文库, "
                               f"总量{sum(lib.get_data_amount_gb() for lib in reserved_backbone):.1f}GB")
            
            # 第一轮排机（混排）- 使用排除骨架后的文库
            lanes, failed = self._schedule_machine_group(libs, machine_type)
            all_lanes.extend(lanes)
            
            # 第二轮：打乱输入池做探索，但真正构Lane前仍会按临检/SJ > YC > 其他重排。
            if failed and len(failed) >= 10:
                random.shuffle(failed)
                logger.info(f"第二轮排机: {len(failed)} 个未分配文库（随机顺序）")
                lanes2, failed2 = self._schedule_machine_group(failed, machine_type)
                all_lanes.extend(lanes2)
                failed = failed2
            
            # 第三轮：小文库优先探索，但单条Lane内仍由高优先级重排逻辑主导。
            if failed and len(failed) >= 10:
                failed.sort(key=lambda lib: lib.get_data_amount_gb())
                logger.info(f"第三轮排机: {len(failed)} 个未分配文库（小文库优先）")
                lanes3, failed3 = self._schedule_machine_group(failed, machine_type)
                all_lanes.extend(lanes3)
                failed = failed3
            
            # 多轮尝试：继续用不同随机顺序排机
            consecutive_failures = 0
            for round_num in range(4, 50):  # 最多尝试到第50轮，增加尝试次数
                if failed and len(failed) >= 10:
                    random.shuffle(failed)
                    logger.info(f"第{round_num}轮排机: {len(failed)} 个未分配文库")
                    lanes_n, failed_n = self._schedule_machine_group(failed, machine_type)
                    # [2025-12-29 修复] 无论是否形成新Lane，都要更新failed
                    # 因为failed_n中包含了验证失败Lane的文库
                    failed = failed_n
                    if lanes_n:
                        all_lanes.extend(lanes_n)
                        consecutive_failures = 0  # 重置连续失败计数
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= 8:  # 连续8轮没有新Lane，停止
                            logger.info(f"连续{consecutive_failures}轮无新Lane，停止尝试")
                            break

            # 碱基不均衡残余专用Lane拆分（容量需满足利用率红线，按专用Lane验证）
            if failed and getattr(self.config, 'enable_dedicated_imbalance_lane', False):
                imbalance_lanes, failed = self._build_imbalance_dedicated_lanes(failed, machine_type)
                if imbalance_lanes:
                    all_lanes.extend(imbalance_lanes)
                    logger.info(f"碱基不均衡专用Lane: 形成{len(imbalance_lanes)}条, 剩余{len(failed)}个未分配文库")
            
            # ===== [2025-12-25 新增] 小文库聚类专用Lane策略 =====
            # 等大文库排完之后，用剩余的同尺寸小文库形成专用Lane
            # 根据人工排机分析，大量同尺寸小文库（如886个5-6G）应集中到专用Lane
            enable_clustering = getattr(self.config, 'enable_small_library_clustering', False)
            if enable_clustering and failed and len(failed) >= 50:
                logger.info(f"小文库聚类策略: 尝试处理剩余 {len(failed)} 个未分配文库")
                clustering_lanes, remaining_after_clustering = self._schedule_small_library_clustering_lanes(
                    failed, machine_type
                )
                if clustering_lanes:
                    all_lanes.extend(clustering_lanes)
                    failed = remaining_after_clustering
                    logger.info(f"小文库聚类策略: 形成{len(clustering_lanes)}条专用Lane, "
                               f"剩余{len(failed)}个未分配文库")
            
            # ===== [2025-12-25 新增] 大带小预留策略（最后执行） =====
            # 使用预留的骨架大文库来携带剩余的小文库
            if enable_backbone and failed:
                # 分离小文库和大文库
                threshold = getattr(self.config, 'small_library_threshold_gb', 20.0)
                small_remaining = [lib for lib in failed if lib.get_data_amount_gb() <= threshold]
                large_remaining = [lib for lib in failed if lib.get_data_amount_gb() > threshold]
                
                if small_remaining:
                    small_total = sum(lib.get_data_amount_gb() for lib in small_remaining)
                    logger.info(f"大带小策略: {len(small_remaining)}个小文库待分配, 总量{small_total:.1f}GB")
                    
                    # 优先使用预留的骨架大文库
                    available_backbone = reserved_backbone + large_remaining
                    
                    if available_backbone:
                        backbone_total = sum(lib.get_data_amount_gb() for lib in available_backbone)
                        logger.info(f"大带小策略: 可用骨架{len(available_backbone)}个, 总量{backbone_total:.1f}GB")
                        
                        backbone_lanes, remaining_small = self._schedule_backbone_with_small_lanes(
                            available_backbone, small_remaining, machine_type
                        )
                        if backbone_lanes:
                            all_lanes.extend(backbone_lanes)
                            # 更新failed：用掉的大文库 + 剩余小文库
                            used_backbone_ids = set()
                            for lane in backbone_lanes:
                                for lib in lane.libraries:
                                    if lib.get_data_amount_gb() > threshold:
                                        used_backbone_ids.add(lib.origrec)
                            # 未用完的骨架也放入failed
                            unused_backbone = [lib for lib in available_backbone if lib.origrec not in used_backbone_ids]
                            failed = unused_backbone + remaining_small
                            logger.info(f"大带小策略: 形成{len(backbone_lanes)}条骨架Lane, "
                                       f"剩余{len(failed)}个未分配文库")
                        else:
                            # 骨架Lane形成失败，将预留的骨架加回failed
                            failed = available_backbone + small_remaining
                    else:
                        # 没有可用骨架大文库，无法执行大带小策略
                        # [2025-12-29 修复] 将文库加回failed
                        failed = small_remaining + large_remaining + reserved_backbone
                        logger.info(f"大带小策略: 无可用骨架大文库，{len(failed)}个文库无法分配")
                else:
                    # 没有小文库，将预留的骨架加回failed
                    # [2025-12-29 修复] 同时保留large_remaining
                    failed = reserved_backbone + large_remaining
            elif reserved_backbone:
                # 骨架预留但未启用大带小策略，将骨架加回failed
                failed = reserved_backbone + failed
            
            unassigned.extend(failed)
        
        # ===== [2025-12-25 新增] 最后填充策略 =====
        # 尝试将剩余未分配文库塞入已有Lane的剩余空间
        if unassigned and all_lanes:
            logger.info(f"最后填充策略: 尝试将{len(unassigned)}个未分配文库塞入已有Lane")
            still_unassigned: List[EnhancedLibraryInfo] = []
            filled_count = 0
            removed_libs: List[EnhancedLibraryInfo] = []  # 记录被踢出的文库
            
            for lib in unassigned:
                placed = False
                for lane in all_lanes:
                    # 检查容量
                    machine_type_str = lane.machine_type.value if lane.machine_type else (lib.eq_type or "Nova X-25B")
                    test_libs_for_capacity = lane.libraries + [lib]
                    _, max_capacity = self._resolve_lane_capacity_limits(
                        test_libs_for_capacity,
                        machine_type_str,
                        lane=lane,
                    )
                    new_total = lane.total_data_gb + lib.get_data_amount_gb()
                    if new_total > max_capacity:
                        continue
                    
                    # [2025-12-25 新增] NB Lane类型约束：只允许非10bp文库
                    if lane.lane_id.startswith('NB_'):
                        # 检查待添加文库是否有10bp Index
                        if self._library_has_10bp_index(lib):
                            continue  # 跳过，不允许10bp文库进入NB Lane
                        
                        # [2025-12-31 新增] NB Lane客户占比严格策略：如果已有内部文库，完全禁止添加客户文库
                        # 如果当前Lane已有内部文库，则禁止添加客户文库（避免客户占比超过50%）
                        # 注意：这个检查在后续的动态调整逻辑之前执行，确保NB Lane的严格规则
                        if lane.libraries:
                            has_internal = any(not l.is_customer_library() for l in lane.libraries)
                            if has_internal and lib.is_customer_library():
                                # Lane中已有内部文库，禁止添加客户文库（不进行动态调整，因为NB Lane应该保持纯内部或纯客户）
                                logger.debug(f"填充阶段: NB Lane {lane.lane_id} 已有内部文库，禁止添加客户文库 {lib.origrec}")
                                continue
                        
                        # [2025-12-31 说明] NB Lane客户占比规则：
                        # - 如果NB Lane已有内部文库，完全禁止添加客户文库（已在上面检查）
                        # - 如果NB Lane没有内部文库（全部是客户文库），允许添加客户文库（会形成100%客户占比的Lane，符合规则）
                        # - 动态调整逻辑在后续执行，但NB Lane的特殊规则优先
                        
                        # [2025-12-31 新增] NB Lane碱基不均衡占比严格检查
                        if not self._check_base_imbalance_ratio_near_limit(
                            lane, lib, threshold=self.config.max_imbalance_ratio
                        ):
                            logger.debug(f"填充阶段: NB Lane {lane.lane_id} 碱基不均衡占比接近限制，拒绝添加碱基不均衡文库 {lib.origrec}")
                            continue
                    
                    # 检查Index兼容性
                    if self.config.enable_index_check:
                        test_libs = lane.libraries + [lib]
                        if not self.index_validator.validate_lane_quick(test_libs):
                            continue
                    
                    # [2025-12-25 新增] Peak Size约束检查
                    test_libs_for_peak = lane.libraries + [lib]
                    if not self._check_peak_size_compatible(test_libs_for_peak):
                        continue
                    
                    # [2025-12-31 新增] 动态调整策略：计算需要踢出的违反规则的文库数据量
                    # 注意：对于NB Lane，如果已有内部文库，完全禁止添加客户文库（已在前面检查）
                    # 对于其他Lane，如果添加文库会导致违反规则，尝试踢出违反规则的文库
                    removal_needed = False
                    to_remove = []
                    
                    # 1. 检查客户占比，计算需要踢出的客户文库数据量
                    # 注意：对于NB Lane，如果已有内部文库，已在前面禁止添加客户文库，这里不会执行
                    test_libs_for_customer = lane.libraries + [lib]
                    if not self._check_customer_ratio_compatible_by_data(test_libs_for_customer):
                        required_customer_removal = self._calculate_required_removal_for_customer_ratio(lane, lib, max_ratio=0.50)
                        logger.info(f"填充阶段: Lane {lane.lane_id} 添加文库 {lib.origrec} 会导致客户占比超过50%，需要踢出{required_customer_removal:.1f}GB客户文库")
                        
                        if required_customer_removal > 0:
                            # 尝试踢出足够的客户文库
                            customer_libs_to_remove = []
                            removal_data = 0.0
                            for l in lane.libraries:
                                if l.is_customer_library() and removal_data < required_customer_removal:
                                    lib_data = float(getattr(l, 'contract_data_raw', 0) or 0)
                                    if removal_data + lib_data <= required_customer_removal * 1.1:  # 允许10%的误差
                                        customer_libs_to_remove.append(l)
                                        removal_data += lib_data
                            
                            if removal_data >= required_customer_removal * 0.9:  # 至少踢出90%的所需数据量
                                to_remove.extend(customer_libs_to_remove)
                                removal_needed = True
                                logger.info(f"填充阶段: Lane {lane.lane_id} 可以踢出{removal_data:.1f}GB客户文库，满足要求，将执行踢出操作")
                            else:
                                logger.warning(f"填充阶段: Lane {lane.lane_id} 只能踢出{removal_data:.1f}GB客户文库，不足要求{required_customer_removal:.1f}GB，拒绝添加")
                                continue  # 无法踢出足够的客户文库，拒绝添加
                    
                    # 2. 检查碱基不均衡占比，计算需要踢出的碱基不均衡文库数据量
                    test_libs_for_imbalance = lane.libraries + [lib]
                    if not self._check_base_imbalance_compatible(test_libs_for_imbalance):
                        required_imbalance_removal = self._calculate_required_removal_for_base_imbalance_ratio(
                            lane, lib, max_ratio=self.config.max_imbalance_ratio
                        )
                        if required_imbalance_removal > 0:
                            # 尝试踢出足够的碱基不均衡文库
                            imbalance_libs_to_remove = []
                            removal_data = 0.0
                            for l in lane.libraries:
                                if l.is_base_imbalance() and l not in to_remove and removal_data < required_imbalance_removal:
                                    lib_data = float(getattr(l, 'contract_data_raw', 0) or 0)
                                    if removal_data + lib_data <= required_imbalance_removal * 1.1:
                                        imbalance_libs_to_remove.append(l)
                                        removal_data += lib_data
                            
                            if removal_data >= required_imbalance_removal * 0.9:
                                to_remove.extend(imbalance_libs_to_remove)
                                removal_needed = True
                            else:
                                continue  # 无法踢出足够的碱基不均衡文库，拒绝添加
                    
                    # 3. 检查10bp Index占比（仅非NB Lane），计算需要踢出的非10bp文库数据量
                    if not lane.lane_id.startswith('NB_'):
                        test_libs_for_10bp = lane.libraries + [lib]
                        if not self._check_10bp_index_ratio_compatible(test_libs_for_10bp):
                            required_non_10bp_removal = self._calculate_required_removal_for_10bp_index_ratio(
                                lane, lib, min_ratio=self.config.min_10bp_index_ratio
                            )
                            if required_non_10bp_removal > 0:
                                # 判断待添加文库是否为10bp
                                lib_is_10bp = False
                                ten_bp_data = getattr(lib, 'ten_bp_data', None)
                                if ten_bp_data is not None and ten_bp_data > 0:
                                    lib_is_10bp = True
                                else:
                                    lib_is_10bp = self._library_has_10bp_index(lib)
                                
                                # 如果待添加的是非10bp文库，需要踢出非10bp文库
                                if not lib_is_10bp:
                                    non_10bp_libs_to_remove = []
                                    removal_data = 0.0
                                    for l in lane.libraries:
                                        if l not in to_remove and removal_data < required_non_10bp_removal:
                                            l_is_10bp = False
                                            l_ten_bp_data = getattr(l, 'ten_bp_data', None)
                                            if l_ten_bp_data is not None and l_ten_bp_data > 0:
                                                l_is_10bp = True
                                            else:
                                                l_is_10bp = self._library_has_10bp_index(l)
                                            
                                            if not l_is_10bp:
                                                lib_data = float(getattr(l, 'contract_data_raw', 0) or 0)
                                                if removal_data + lib_data <= required_non_10bp_removal * 1.1:
                                                    non_10bp_libs_to_remove.append(l)
                                                    removal_data += lib_data
                                    
                                    if removal_data >= required_non_10bp_removal * 0.9:
                                        to_remove.extend(non_10bp_libs_to_remove)
                                        removal_needed = True
                                    else:
                                        continue  # 无法踢出足够的非10bp文库，拒绝添加
                    
                    # 4. 如果需要踢出文库，先执行踢出操作
                    if removal_needed and to_remove:
                        # 验证踢出后Lane是否仍然有效（容量检查）
                        removal_data_total = sum(l.get_data_amount_gb() for l in to_remove)
                        remaining_data = lane.total_data_gb - removal_data_total + lib.get_data_amount_gb()
                        remaining_libs_after_removal = [l for l in lane.libraries if l not in to_remove] + [lib]
                        min_lane_data, max_capacity = self._resolve_lane_capacity_limits(
                            remaining_libs_after_removal,
                            machine_type_str,
                            lane=lane,
                        )
                        
                        # 检查容量下限和上限
                        if remaining_data < min_lane_data:
                            continue  # 踢出后容量不足，拒绝添加
                        
                        if remaining_data > max_capacity:
                            continue  # 踢出后容量仍然超过上限，拒绝添加
                        
                        # 执行踢出操作
                        for l in to_remove:
                            lane.remove_library(l)
                            removed_libs.append(l)  # 记录被踢出的文库，后续重新分配
                            logger.debug(f"填充阶段: 从Lane {lane.lane_id} 踢出违反规则的文库 {l.origrec} ({l.get_data_amount_gb():.1f}GB)")
                    
                    # 5. 最终容量检查（确保添加后不超过上限）
                    final_total = lane.total_data_gb + lib.get_data_amount_gb()
                    _, max_capacity = self._resolve_lane_capacity_limits(
                        lane.libraries + [lib],
                        machine_type_str,
                        lane=lane,
                    )
                    if final_total > max_capacity:
                        continue  # 最终容量超过上限，拒绝添加
                    
                    # 6. 最终Index冲突检查（确保添加后没有Index冲突）
                    if self.config.enable_index_check:
                        final_test_libs = lane.libraries + [lib]
                        if not self.index_validator.validate_lane_quick(final_test_libs):
                            continue  # 最终Index冲突检查失败，拒绝添加
                    
                    # 7. 最终规则检查（确保添加后所有规则都满足）
                    final_test_libs = lane.libraries + [lib]
                    
                    # 7.0 NB Lane特殊规则检查（优先于其他检查）
                    if lane.lane_id.startswith('NB_'):
                        # NB Lane：如果已有内部文库，完全禁止添加客户文库
                        if lane.libraries:
                            has_internal = any(not l.is_customer_library() for l in lane.libraries)
                            if has_internal and lib.is_customer_library():
                                logger.debug(f"填充阶段最终检查: NB Lane {lane.lane_id} 已有内部文库，禁止添加客户文库 {lib.origrec}")
                                continue
                    
                    # 7.1 客户占比检查
                    if not self._check_customer_ratio_compatible_by_data(final_test_libs):
                        continue  # 最终客户占比检查失败，拒绝添加
                    
                    # 7.2 碱基不均衡占比检查
                    if not self._check_base_imbalance_compatible(final_test_libs):
                        continue  # 最终碱基不均衡占比检查失败，拒绝添加
                    
                    # 7.3 Peak Size检查
                    if not self._check_peak_size_compatible(final_test_libs):
                        continue  # 最终Peak Size检查失败，拒绝添加
                    
                    # 7.4 10bp Index占比检查（仅非NB Lane）
                    if not lane.lane_id.startswith('NB_'):
                        if not self._check_10bp_index_ratio_compatible(final_test_libs):
                            continue  # 最终10bp Index占比检查失败，拒绝添加
                    
                    # 8. 通过所有检查，加入Lane
                    lane.add_library(lib)
                    placed = True
                    filled_count += 1
                    
                    # [2025-12-31 新增] 填充阶段日志：记录向NB Lane添加文库的情况
                    if lane.lane_id.startswith('NB_'):
                        current_total = sum(float(getattr(l, 'contract_data_raw', 0) or 0) for l in lane.libraries)
                        current_customer = sum(float(getattr(l, 'contract_data_raw', 0) or 0) for l in lane.libraries if l.is_customer_library())
                        current_customer_ratio = current_customer / current_total if current_total > 0 else 0.0
                        logger.debug(f"填充阶段: 向NB Lane {lane.lane_id} 添加文库 {lib.origrec} (客户={lib.is_customer_library()}), "
                                   f"当前客户占比={current_customer_ratio:.1%}, 踢出{len(to_remove)}个违反规则的文库")
                    
                    break
                
                if not placed:
                    still_unassigned.append(lib)
            
            if filled_count > 0:
                logger.info(f"最后填充策略: 成功塞入{filled_count}个文库，踢出{len(removed_libs)}个违反规则的文库，剩余{len(still_unassigned)}个无法分配")
            
            # 将被踢出的文库添加到未分配列表，以便后续重新分配
            unassigned = still_unassigned + removed_libs
        
        # ===== [2025-12-25 新增] 挪移优化策略 =====
        # 当有未分配文库且无法新开Lane时，从现有Lane挪出部分文库合并成新Lane
        if unassigned and all_lanes:
            unassigned, new_lanes = self._optimize_by_redistribution(
                unassigned, all_lanes, machine_type
            )
            all_lanes.extend(new_lanes)

        # 统一回滚预拆分文库：任一拆分家族未全部成Lane，则整组回滚为原始文库
        if perform_presplit_family_rollback and presplit_family_context:
            all_lanes, unassigned, rollback_records = self._rollback_incomplete_presplit_families(
                lanes=all_lanes,
                unassigned=unassigned,
                family_context=presplit_family_context,
            )
            if rollback_records:
                logger.info(
                    "预拆分回滚完成: {}个原始文库因子片段未全部成Lane而回滚",
                    len(rollback_records),
                )

        # 3.5 Pooling优化（不改变Lane组成，仅调整系数）
        if all_lanes:
            all_lanes = self._apply_pooling_optimization(all_lanes)
        
        # 4. 计算统计信息
        total_assigned = sum(len(lane.libraries) for lane in all_lanes)
        total_data = sum(lane.total_data_gb for lane in all_lanes)
        total_capacity = sum(lane.lane_capacity_gb for lane in all_lanes)
        avg_utilization = total_data / total_capacity if total_capacity > 0 else 0
        
        elapsed = time.time() - start_time
        
        logger.info(f"逐Lane贪心排机完成 - "
                   f"Lane数: {len(all_lanes)}, "
                   f"已分配: {total_assigned}/{len(libraries)}, "
                   f"未分配: {len(unassigned)}, "
                   f"平均利用率: {avg_utilization:.1%}, "
                   f"耗时: {elapsed:.2f}秒")
        
        # 5. 构建返回结果
        solution = SchedulingSolution(
            lane_assignments=all_lanes,
            unassigned_libraries=unassigned
        )
        solution.calculate_overall_metrics()
        
        # 6. [2025-12-26 新增] 使用LaneValidator对所有Lane进行最终验证
        self._validate_all_lanes(solution, keep_failed_lanes=keep_failed_lanes)

        # 恢复基础配置，避免跨次排机污染
        self.config = self._base_config
        return solution

    def _build_presplit_family_context(
        self,
        libraries: List[EnhancedLibraryInfo],
    ) -> Dict[str, Dict[str, object]]:
        """从预拆分后的文库列表中提取拆分家族上下文。"""
        family_context: Dict[str, Dict[str, object]] = {}
        for lib in libraries:
            family_id = str(getattr(lib, "original_library_id", "") or "").strip()
            if not family_id:
                continue
            total_fragments = int(getattr(lib, "total_fragments", 0) or 0)
            if total_fragments <= 1:
                continue
            family_entry = family_context.setdefault(
                family_id,
                {
                    "expected_fragments": total_fragments,
                    "source_library": getattr(lib, "_split_source_library", None),
                },
            )
            if family_entry.get("source_library") is None:
                family_entry["source_library"] = getattr(lib, "_split_source_library", None)
            family_entry["expected_fragments"] = max(
                int(family_entry.get("expected_fragments", 0) or 0),
                total_fragments,
            )
        return family_context

    def _rollback_incomplete_presplit_families(
        self,
        lanes: List[LaneAssignment],
        unassigned: List[EnhancedLibraryInfo],
        family_context: Dict[str, Dict[str, object]],
    ) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo], List[Dict[str, object]]]:
        """若预拆分家族未全部成Lane，则移除其片段并恢复原始文库。"""
        if not family_context:
            return lanes, unassigned, []

        assigned_fragments: Dict[str, List[Tuple[LaneAssignment, EnhancedLibraryInfo]]] = {}
        for lane in lanes:
            for lib in list(lane.libraries):
                family_id = str(getattr(lib, "original_library_id", "") or "").strip()
                if family_id and family_id in family_context:
                    assigned_fragments.setdefault(family_id, []).append((lane, lib))

        unassigned_fragments: Dict[str, List[EnhancedLibraryInfo]] = {}
        remaining_unassigned: List[EnhancedLibraryInfo] = []
        for lib in unassigned:
            family_id = str(getattr(lib, "original_library_id", "") or "").strip()
            if family_id and family_id in family_context:
                unassigned_fragments.setdefault(family_id, []).append(lib)
                continue
            remaining_unassigned.append(lib)

        rollback_records: List[Dict[str, object]] = []
        restored_originals: List[EnhancedLibraryInfo] = []
        for family_id, context in family_context.items():
            expected_fragments = int(context.get("expected_fragments", 0) or 0)
            assigned_count = len(assigned_fragments.get(family_id, []))
            pending_count = len(unassigned_fragments.get(family_id, []))
            if expected_fragments <= 0 or (assigned_count == expected_fragments and pending_count == 0):
                continue

            for lane, lib in assigned_fragments.get(family_id, []):
                lane.remove_library(lib)

            source_library = context.get("source_library")
            if source_library is not None:
                restored_originals.append(source_library)

            rollback_records.append(
                {
                    "family_id": family_id,
                    "expected_fragments": expected_fragments,
                    "assigned_fragments": assigned_count,
                    "pending_fragments": pending_count,
                }
            )

        filtered_lanes = [lane for lane in lanes if lane.libraries]
        final_unassigned = remaining_unassigned + restored_originals
        return filtered_lanes, final_unassigned, rollback_records

    def _try_assign_split_fragments_atomically(
        self,
        split_libraries: List[EnhancedLibraryInfo],
        lanes: List[LaneAssignment],
    ) -> bool:
        """尝试原子分配拆分子文库：全部成功且不落同Lane才提交，否则整体回滚。"""
        if not split_libraries or not lanes:
            return False
        if len(split_libraries) > len(lanes):
            return False

        placed_records: List[Tuple[LaneAssignment, EnhancedLibraryInfo]] = []
        used_lane_ids: Set[str] = set()
        for fragment in split_libraries:
            target_lane: Optional[LaneAssignment] = None
            for lane in lanes:
                if lane.lane_id in used_lane_ids:
                    continue
                if self._can_add_to_lane(lane, fragment):
                    lane.add_library(fragment)
                    target_lane = lane
                    used_lane_ids.add(lane.lane_id)
                    placed_records.append((lane, fragment))
                    break

            if target_lane is None:
                for placed_lane, placed_lib in reversed(placed_records):
                    placed_lane.remove_library(placed_lib)
                return False
        return True

    def _get_split_family_id(self, lib: EnhancedLibraryInfo) -> Optional[str]:
        """提取拆分家族标识，用于同Lane互斥约束。"""
        explicit_family_id = str(getattr(lib, "original_library_id", "") or "").strip()
        if explicit_family_id:
            return explicit_family_id

        is_split = bool(getattr(lib, "is_split", False))
        split_flag = str(getattr(lib, "wkissplit", "") or "").strip()
        if is_split or split_flag.lower() in {"yes", "true", "1"} or split_flag == "是":
            fallback_family_id = str(getattr(lib, "origrec", "") or "").strip()
            if fallback_family_id:
                return fallback_family_id
        return None

    def _apply_deferred_atomic_split(
        self,
        unassigned: List[EnhancedLibraryInfo],
        lanes: List[LaneAssignment],
    ) -> Tuple[List[EnhancedLibraryInfo], List[Dict[str, object]]]:
        """对未分配文库执行按需拆分：仅全部子文库可入Lane时才提交拆分。"""
        if not unassigned or not lanes:
            return unassigned, []

        remaining_unassigned: List[EnhancedLibraryInfo] = []
        committed_records: List[Dict[str, object]] = []

        for lib in unassigned:
            if not self.library_splitter._should_split(lib):
                remaining_unassigned.append(lib)
                continue

            split_libs = self.library_splitter._perform_split(lib)
            if len(split_libs) <= 1:
                remaining_unassigned.append(lib)
                continue
            if not all(
                float(getattr(fragment, "contract_data_raw", 0) or 0)
                > self.library_splitter.min_split_size
                for fragment in split_libs
            ):
                remaining_unassigned.append(lib)
                continue

            committed = self._try_assign_split_fragments_atomically(
                split_libraries=split_libs,
                lanes=lanes,
            )
            if not committed:
                remaining_unassigned.append(lib)
                continue

            committed_records.append(
                {
                    "original_id": str(getattr(lib, "origrec", "") or ""),
                    "original_size": float(getattr(lib, "contract_data_raw", 0) or 0),
                    "split_count": len(split_libs),
                    "new_ids": [str(getattr(item, "origrec", "") or "") for item in split_libs],
                }
            )

        if committed_records:
            logger.info(
                "按需拆分执行: 触发提交{}个，保留未拆分{}个",
                len(committed_records),
                len(remaining_unassigned),
            )
        return remaining_unassigned, committed_records

    def _apply_pooling_optimization(self, lanes: List[LaneAssignment]) -> List[LaneAssignment]:
        """为已形成的Lane计算Pooling系数，失败时使用规则降级"""
        if not self.pooling_optimizer or not self.pooling_optimizer.enabled:
            return lanes

        for lane in lanes:
            try:
                result: PoolingOptimizationResult = self.pooling_optimizer.optimize_pooling(lane)
                lane.pooling_coefficients = result.coefficients
                lane.metadata["predicted_output"] = result.predicted_total_output
                lane.metadata["predicted_cv"] = result.predicted_cv
                lane.metadata["order_cv_ratio"] = result.order_cv_ratio
                if result.lane_cv is not None:
                    lane.metadata["lane_cv"] = result.lane_cv
                else:
                    lane.metadata.pop("lane_cv", None)
                lane.metadata["pooling_reason"] = result.reason
                if result.warnings:
                    lane.metadata["pooling_warnings"] = result.warnings
            except Exception as exc:
                logger.exception(f"Lane {lane.lane_id} Pooling优化失败，使用规则降级: {exc}")
                lane.pooling_coefficients = self.pooling_optimizer.calculate_simple_coefficients(
                    lane.libraries
                )
                lane.metadata.pop("lane_cv", None)
        return lanes
    
    def _validate_all_lanes(self, solution: SchedulingSolution, keep_failed_lanes: bool = False) -> None:
        """
        [2025-12-26 新增] [2025-12-29 优化] 使用LaneValidator对所有Lane进行最终验证
        
        验证流程优化：
        1. 验证所有Lane，记录验证失败的Lane
        2. 将验证失败Lane中的文库移到未分配列表
        3. 尝试从新的未分配列表中组建新Lane（残余精选）
        4. 对新Lane进行验证
        
        Args:
            solution: 排机结果
            keep_failed_lanes: 是否保留验证失败的Lane（不拆解），供外层处理
        """
        from arrange_library.core.constraints.lane_validator import LaneValidator
        
        validator = LaneValidator(strict_mode=False)
        
        # 第一轮验证
        valid_lanes = []
        failed_lane_objects = []
        
        for lane in solution.lane_assignments:
            lane_id = lane.lane_id
            libs = lane.libraries
            metadata = self._build_lane_validation_metadata(lane)
            
            # 获取机器类型
            machine_type = lane.machine_type.value if lane.machine_type else "Nova X-25B"
            
            # 调用LaneValidator验证
            result = validator.validate_lane(
                libraries=libs,
                lane_id=lane_id,
                machine_type=machine_type,
                metadata=metadata
            )
            
            if result.is_valid:
                valid_lanes.append(lane)
            else:
                failed_lane_objects.append((lane, result.errors))
                logger.warning(f"Lane {lane_id} 验证失败: {[e.message for e in result.errors]}")
        
        # 如果有验证失败的Lane，将其文库移到未分配列表
        if failed_lane_objects:
            logger.info(f"有{len(failed_lane_objects)}条Lane验证失败，尝试挪移优化...")
            
            # keep_failed_lanes=True：不拆解、不残余精选，外层自行矫正
            if keep_failed_lanes:
                logger.info(f"keep_failed_lanes=True，保留{len(failed_lane_objects)}条验证失败的Lane供外层矫正（未分配池不变）")
                for lane, errors in failed_lane_objects:
                    lane.validation_errors = [e.message for e in errors]
                    valid_lanes.append(lane)
                solution.lane_assignments = valid_lanes
                logger.info(f"最终验证结果（含待矫正Lane）: {len(solution.lane_assignments)}条Lane，{len(solution.unassigned_libraries)}个文库未分配")
                return
            
            # 收集验证失败Lane中的所有文库
            failed_libs = []
            for lane, _ in failed_lane_objects:
                failed_libs.extend(lane.libraries)
            
            # 合并到未分配列表
            all_unassigned = list(solution.unassigned_libraries) + failed_libs
            logger.info(f"未分配文库更新: 原{len(solution.unassigned_libraries)}个 + 验证失败{len(failed_libs)}个 = {len(all_unassigned)}个")
            
            # 尝试从新的未分配列表中组建新Lane（残余精选）
            machine_type = "Nova X-25B"
            remaining, new_lanes = self._try_form_lane_from_unassigned(all_unassigned, machine_type)
            
            if new_lanes:
                # 验证新创建的Lane
                for new_lane in new_lanes:
                    new_result = validator.validate_lane(
                        libraries=new_lane.libraries,
                        lane_id=new_lane.lane_id,
                        machine_type=machine_type
                    )
                    if new_result.is_valid:
                        valid_lanes.append(new_lane)
                        logger.info(f"残余精选成功: 新Lane {new_lane.lane_id} 验证通过")
                    else:
                        # 新Lane也失败，文库退回未分配
                        remaining.extend(new_lane.libraries)
                        logger.warning(f"残余精选的Lane {new_lane.lane_id} 验证失败: {[e.message for e in new_result.errors]}")
            
            # 更新solution
            solution.lane_assignments = valid_lanes
            solution.unassigned_libraries = remaining
        else:
            # 没有验证失败的Lane，直接使用原结果
            solution.lane_assignments = valid_lanes
        
        # 最终记录
        logger.info(f"最终验证结果: {len(solution.lane_assignments)}条Lane通过，{len(solution.unassigned_libraries)}个文库未分配")
    
    def _sort_libraries(self, libraries: List[EnhancedLibraryInfo]) -> List[EnhancedLibraryInfo]:
        """
        普通排序：按优先级分层后，再按数据量降序。
        """
        return sorted(
            libraries,
            key=lambda lib: (
                self._get_scattered_mix_priority_rank(lib),
                self._get_scattered_mix_delete_date_sort_value(lib),
                -lib.get_data_amount_gb(),
            ),
        )

    def _get_scattered_mix_priority_rank(self, lib: EnhancedLibraryInfo) -> int:
        """散样混排优先级：临检和SJ > YC > 其他。"""
        data_type = str(getattr(lib, "data_type", "") or "").strip()
        if data_type in {"临检", "SJ"} or lib.is_clinical_by_code() or lib.is_s_level_customer():
            return 0
        if data_type == "YC" or lib.is_yc_library():
            return 1
        return 2

    def _parse_scattered_mix_delete_date(self, lib: EnhancedLibraryInfo) -> Optional[float]:
        """解析散样混排的delete_date天数字段，数值越小表示越临近越优先。"""
        raw_value = getattr(lib, "_delete_date_raw", None)
        if raw_value in (None, ""):
            raw_value = getattr(lib, "deduction_time", None)
        if raw_value in (None, ""):
            return None

        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return None

    def _get_scattered_mix_delete_date_sort_value(self, lib: EnhancedLibraryInfo) -> float:
        """其他文库按delete_date排序，越临近越优先；缺失值排最后。"""
        if self._get_scattered_mix_priority_rank(lib) < 2:
            return 0.0
        parsed = self._parse_scattered_mix_delete_date(lib)
        if parsed is None:
            return float("inf")
        return parsed

    def _sort_remaining_for_scattered_mix_lane(
        self,
        libraries: List[EnhancedLibraryInfo],
    ) -> List[EnhancedLibraryInfo]:
        """散样混排成Lane顺序：优先聚拢高优先级同类文库。"""
        if not libraries:
            return libraries

        board_sorted = self._sort_by_board_preference(libraries)
        board_order = {id(lib): idx for idx, lib in enumerate(board_sorted)}
        return sorted(
            libraries,
            key=lambda lib: (
                self._get_scattered_mix_priority_rank(lib),
                self._get_scattered_mix_delete_date_sort_value(lib),
                board_order.get(id(lib), len(board_order)),
                -lib.get_data_amount_gb(),
            ),
        )

    def _sort_remaining_for_lane_seed(
        self,
        libraries: List[EnhancedLibraryInfo],
        seed_lib: EnhancedLibraryInfo,
    ) -> List[EnhancedLibraryInfo]:
        """单条Lane内优先吞同级高优先级文库，降低临检/SJ/YC被打散概率。"""
        if not libraries:
            return libraries

        seed_rank = self._get_scattered_mix_priority_rank(seed_lib)
        base_sorted = self._sort_remaining_for_scattered_mix_lane(libraries)
        if seed_rank >= 2:
            return base_sorted
        base_order = {id(lib): idx for idx, lib in enumerate(base_sorted)}
        return sorted(
            libraries,
            key=lambda lib: (
                0 if self._get_scattered_mix_priority_rank(lib) == seed_rank else 1,
                self._get_scattered_mix_priority_rank(lib),
                self._get_scattered_mix_delete_date_sort_value(lib),
                base_order.get(id(lib), len(base_order)),
                -lib.get_data_amount_gb(),
            ),
        )

    @staticmethod
    def _normalize_profile_text(value: Any) -> str:
        """统一规则画像中的文本口径。"""
        if value is None:
            return ""
        return str(value).strip().replace("＋", "+").replace("×", "X").upper()

    def _is_customer_library(self, lib: EnhancedLibraryInfo) -> bool:
        """判断文库是否为客户文库。"""
        checker = getattr(lib, "is_customer_library", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                pass
        customer_flag = self._normalize_profile_text(getattr(lib, "customer_library", "") or "")
        if customer_flag in {"是", "客户", "Y", "YES", "TRUE", "1"}:
            return True
        if customer_flag in {"否", "N", "NO", "FALSE", "0"}:
            return False
        lab_type = self._normalize_profile_text(getattr(lib, "lab_type", "") or "")
        return "客户" in lab_type

    def _is_soft_single_lane_target_library(
        self,
        lib: EnhancedLibraryInfo,
        profile: Dict[str, Any],
    ) -> bool:
        """判断文库是否命中“软优先单独成lane”目标集合。"""
        customer_complex_targets = {
            self._normalize_profile_text(item)
            for item in profile.get("customer_complex_results_any", set()) or set()
            if self._normalize_profile_text(item)
        }
        internal_risk_targets = {
            self._normalize_profile_text(item)
            for item in profile.get("internal_risk_build_flags_any", set()) or set()
            if self._normalize_profile_text(item)
        }
        if not customer_complex_targets and not internal_risk_targets:
            return False

        if self._is_customer_library(lib):
            complex_result = self._normalize_profile_text(
                getattr(lib, "complex_result", None) or getattr(lib, "wkcomplexresult", None) or ""
            )
            return complex_result in customer_complex_targets

        risk_build_flag = self._normalize_profile_text(
            getattr(lib, "risk_build_flag", None) or getattr(lib, "wkriskbuildflag", None) or ""
        )
        return risk_build_flag in internal_risk_targets

    def _sort_by_soft_single_lane_preference(
        self,
        libraries: List[EnhancedLibraryInfo],
        seed_lib: EnhancedLibraryInfo,
        seed_rule: Any,
    ) -> List[EnhancedLibraryInfo]:
        """软优先：命中规则时，优先将同类风险文库聚拢到当前Lane。"""
        if not libraries:
            return libraries
        if not bool(getattr(seed_rule, "soft_single_lane_preferred", False)):
            return libraries

        profile = dict(getattr(seed_rule, "profile", {}) or {})
        seed_is_target = self._is_soft_single_lane_target_library(seed_lib, profile)
        if not seed_is_target:
            return libraries

        base_order = {id(lib): idx for idx, lib in enumerate(libraries)}
        return sorted(
            libraries,
            key=lambda lib: (
                0 if self._is_soft_single_lane_target_library(lib, profile) else 1,
                base_order.get(id(lib), len(base_order)),
            ),
        )

    def _sort_by_board_preference(
        self, libraries: List[EnhancedLibraryInfo]
    ) -> List[EnhancedLibraryInfo]:
        """
        软约束排序：在满足规则的前提下，尽量将同板号(wkboardnumber)的文库聚合在一起。

        策略：
        - 统计当前待排文库中各板号的出现次数
        - 出现次数越多的板号排越靠前（整板优先进入当前Lane）
        - 同板号内部按数据量降序（保持填充效率）
        - 无板号文库排在最后，按数据量降序

        此方法仅调整排列顺序，不修改任何硬约束，不影响 _can_add_to_lane 校验逻辑。
        每条Lane开始前调用一次，使贪心算法自然倾向于聚合同板文库。
        """
        if not libraries:
            return libraries

        board_count: Dict[str, int] = {}
        for lib in libraries:
            board = getattr(lib, 'board_number', '') or ''
            if board:
                board_count[board] = board_count.get(board, 0) + 1

        if not board_count:
            return libraries

        def _sort_key(lib: EnhancedLibraryInfo) -> tuple:
            board = getattr(lib, 'board_number', '') or ''
            freq = board_count.get(board, 0)
            # 无板号文库频次设为0，排最后；有板号按频次降序，板内数据量降序
            return (-freq, board, -lib.get_data_amount_gb())

        return sorted(libraries, key=_sort_key)

    def _schedule_dedicated_imbalance_lanes(
        self,
        imbalance_libs: List[EnhancedLibraryInfo],
        machine_type: str
    ) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo]]:
        """
        [2025-12-25 新增] 为碱基不均衡文库形成专用Lane
        
        根据人工排机数据分析：
        - 人工排机有5条100%碱基不均衡专用Lane
        - 每条Lane: 36个文库，972GB数据
        - 文库类型: 主要是G27分组（10X转录组-3'+墨卓转录组-3端）
        
        策略（严格）：
        1. 目标碱基不均衡量 = Lane容量 × 分组占比（0.8/0.99/1.0等）
        2. 平衡文库补量 = Lane容量 - 目标碱基不均衡量
        3. 满足混排规则、Index与Peak约束后形成专用Lane
        
        Args:
            imbalance_libs: 碱基不均衡文库列表
            machine_type: 机器类型
            
        Returns:
            (专用Lane列表, 剩余未分配的碱基不均衡文库)
        """
        if not imbalance_libs:
            return [], []
        if not self.imbalance_handler:
            logger.warning("imbalance_handler未初始化，无法执行碱基不均衡专用Lane策略")
            return [], list(imbalance_libs)

        # 将字符串机器类型转换为枚举（兼容不同写法）
        machine_type_enum = self._resolve_machine_type_enum(machine_type, imbalance_libs)
        if machine_type_enum == MachineType.UNKNOWN:
            machine_type_enum = MachineType.NOVA_X_25B
        machine_type_str = machine_type_enum.value if isinstance(machine_type_enum, MachineType) else str(machine_type)
        
        customer_imbalance, internal_imbalance = self._split_imbalance_by_customer(imbalance_libs)

        def _build_group(
            group_libs: List[EnhancedLibraryInfo],
        ) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo]]:
            if not group_libs:
                return [], []

            grouped: Dict[str, List[EnhancedLibraryInfo]] = {}
            unmatched: List[EnhancedLibraryInfo] = []
            for lib in group_libs:
                group_id = self.imbalance_handler.identify_imbalance_type(lib)
                if not group_id or group_id == "G_UNKNOWN":
                    unmatched.append(lib)
                    continue
                target_group = group_id
                grouped.setdefault(target_group, []).append(lib)

            generated_lanes: List[LaneAssignment] = []
            remaining_all: List[EnhancedLibraryInfo] = []

            for target_group, libs in grouped.items():
                remaining = self._sort_remaining_for_scattered_mix_lane(libs)
                target_ratio = self.imbalance_handler.get_group_data_ratio(target_group)
                target_imbalance = self.config.lane_capacity_gb * target_ratio
                min_imbalance = target_imbalance * 0.95
                if target_imbalance <= 0:
                    remaining_all.extend(remaining)
                    continue

                while remaining:
                    lane_id = self._get_next_lane_id("DL", machine_type)
                    picked: List[EnhancedLibraryInfo] = []
                    next_remaining: List[EnhancedLibraryInfo] = []
                    picked_data = 0.0

                    for lib in remaining:
                        lib_data = lib.get_data_amount_gb()
                        if picked_data + lib_data > target_imbalance + 1e-6:
                            next_remaining.append(lib)
                            continue

                        test_libs = picked + [lib]
                        if self.config.enable_index_check and not self.index_validator.validate_lane_quick(test_libs):
                            next_remaining.append(lib)
                            continue
                        if not self._check_peak_size_compatible(test_libs):
                            next_remaining.append(lib)
                            continue
                        is_compatible, _ = self.imbalance_handler.check_mix_compatibility(
                            test_libs,
                            enforce_total_limit=False,
                        )
                        if not is_compatible:
                            next_remaining.append(lib)
                            continue

                        picked.append(lib)
                        picked_data += lib_data

                    if not picked or picked_data < min_imbalance:
                        remaining_all.extend(picked + next_remaining)
                        break

                    lane = LaneAssignment(
                        lane_id=lane_id,
                        machine_id=f"M_{lane_id[3:]}",
                        machine_type=machine_type_enum,
                        lane_capacity_gb=self.config.lane_capacity_gb,
                    )
                    for lib in picked:
                        lane.add_library(lib)

                    balance_data = max(self.config.lane_capacity_gb - picked_data, 0.0)
                    lane.metadata["required_balance_data_gb"] = round(balance_data, 3)
                    lane.metadata["wkbalancedata"] = round(balance_data, 3)

                    lane.metadata["is_dedicated_imbalance_lane"] = True
                    lane.metadata["dedicated_group"] = target_group
                    lane.metadata["dedicated_target_ratio"] = round(target_ratio, 4)
                    lane.metadata["dedicated_imbalance_data_gb"] = round(picked_data, 3)
                    lane.metadata["dedicated_target_imbalance_gb"] = round(target_imbalance, 3)

                    effective_total = lane.total_data_gb + balance_data
                    utilization = effective_total / self.config.lane_capacity_gb if self.config.lane_capacity_gb > 0 else 0.0
                    lane_machine_type_str = (
                        lane.machine_type.value if lane.machine_type else machine_type_str
                    )
                    _, max_allowed = self._resolve_lane_capacity_limits(
                        lane.libraries,
                        lane_machine_type_str,
                        lane=lane,
                    )
                    if effective_total > max_allowed + 1e-6:
                        remaining_all.extend(picked + next_remaining)
                        break

                    if self._validate_dedicated_lane(lane):
                        generated_lanes.append(lane)
                        logger.info(
                            f"专用Lane {lane.lane_id} 形成成功 - 分组={target_group}, "
                            f"碱基不均衡={picked_data:.1f}G, 平衡文库={balance_data:.1f}G, "
                            f"总量(含平衡)={effective_total:.1f}G"
                        )
                        remaining = next_remaining
                    else:
                        remaining_all.extend(picked + next_remaining)
                        break

            return generated_lanes, unmatched + remaining_all

        customer_lanes, remaining_customer = _build_group(customer_imbalance)
        internal_lanes, remaining_internal = _build_group(internal_imbalance)

        dedicated_lanes = customer_lanes + internal_lanes
        remaining = remaining_customer + remaining_internal

        return dedicated_lanes, remaining
    
    def _validate_dedicated_lane(self, lane: LaneAssignment) -> bool:
        """
        验证专用Lane（简化版，只检查Index冲突和容量）
        
        专用Lane不受碱基不均衡占比限制，因为它本身就是100%碱基不均衡
        """
        # [2026-01-28] 专用Lane也必须检查碱基不均衡混排规则
        if self.imbalance_handler:
            is_compatible, reason = self.imbalance_handler.check_mix_compatibility(
                lane.libraries,
                enforce_total_limit=False,
            )
            if not is_compatible:
                logger.info(f"专用Lane {lane.lane_id} 混排规则不兼容: {reason}")
                return False
        
        machine_type_str = lane.machine_type.value if lane.machine_type else "Nova X-25B"
        
        # 创建一个非严格模式的验证器，专用Lane只检查错误不检查警告
        from arrange_library.core.constraints.lane_validator import LaneValidator
        lenient_validator = LaneValidator(strict_mode=False)
        
        # [2025-12-26 修复] 传入metadata指示这是碱基不均衡专用Lane，跳过碱基不均衡占比检查
        validation_metadata: Dict[str, object] = {"is_dedicated_imbalance_lane": True}
        balance_data = lane.metadata.get("wkbalancedata")
        if balance_data is None:
            balance_data = lane.metadata.get("wkadd_balance_data")
        if balance_data is not None:
            validation_metadata["wkbalancedata"] = balance_data
        result = lenient_validator.validate_lane(
            libraries=lane.libraries,
            lane_id=lane.lane_id,
            machine_type=machine_type_str,
            metadata=validation_metadata,
        )
        
        # 专用Lane只关注严重错误，忽略警告
        if result.errors:
            error_types = [e.rule_type.value for e in result.errors]
            logger.debug(f"专用Lane {lane.lane_id} 验证失败: {error_types}")
            return False
        
        return True
    
    def _check_peak_size_compatible(self, libraries: List[EnhancedLibraryInfo]) -> bool:
        """
        检查文库列表的Peak Size是否满足约束
        
        规则6要求（满足任一即可）：
        - 条件1: Peak Size 最大值-最小值 <= 150bp
        - 条件2: 150bp窗口内数据量 >= 总数据量的75%
        
        Args:
            libraries: 待检查的文库列表
            
        Returns:
            True表示满足Peak Size约束，False表示不满足
        """
        if len(libraries) < 2:
            return True  # 文库数不足，无需检查
        
        # 收集Peak Size数据
        peak_data = {}  # {peak_size: total_data}
        total_data = 0.0
        
        for lib in libraries:
            peak_size = getattr(lib, 'peak_size', None)
            data = lib.get_data_amount_gb()
            total_data += data
            
            if peak_size and peak_size > 0:
                ps = float(peak_size)
                peak_data[ps] = peak_data.get(ps, 0) + data
        
        if len(peak_data) < 2:
            return True  # 有效Peak Size不足，无需检查
        
        min_peak = min(peak_data.keys())
        max_peak = max(peak_data.keys())
        diff = max_peak - min_peak
        
        # 条件1: 差值 <= 150bp
        if diff <= 150:
            return True
        
        # 条件2: 150bp窗口覆盖率 >= 75%
        best_coverage = 0.0
        for window_start in range(int(min_peak), int(max_peak) - 150 + 1):
            window_end = window_start + 150
            covered_data = sum(
                data for ps, data in peak_data.items()
                if window_start <= ps <= window_end
            )
            coverage = covered_data / total_data if total_data > 0 else 0
            best_coverage = max(best_coverage, coverage)
        
        return best_coverage >= 0.75
    
    def _check_customer_ratio_compatible(self, libraries: List[EnhancedLibraryInfo]) -> bool:
        """
        检查文库列表的客户占比是否满足约束（按文库数量）
        
        [2025-12-26 修复] 使用is_customer_library()方法识别客户文库
        规则：客户占比 <=50% 或 =100% 都通过
        - 客户占比 <= 50% → 通过（混排Lane）
        - 客户占比 = 100% → 通过（客户专用Lane）
        - 50% < 客户占比 < 100% → 不通过（中间比例不允许）
        
        Args:
            libraries: 待检查的文库列表
            
        Returns:
            True表示满足客户占比约束，False表示不满足
        """
        total_libs = len(libraries)
        if total_libs < 1:
            return True
        
        # 统计客户文库（使用is_customer_library方法）
        customer_count = 0
        for lib in libraries:
            if lib.is_customer_library():
                customer_count += 1
        
        # 计算客户占比（按文库数量）
        customer_ratio = customer_count / total_libs
        
        # 规则：<=50% 或 =100% 都通过
        # 只有在 >50% 且 <100% 时才不通过
        if customer_ratio <= 0.50 or customer_ratio == 1.0:
            return True
        return False
    
    def _check_customer_ratio_compatible_by_data(self, libraries: List[EnhancedLibraryInfo]) -> bool:
        """
        检查文库列表的客户占比是否满足约束（按数据量计算）
        
        [2025-12-31 新增] 按数据量计算客户占比，与验证规则保持一致
        规则：客户占比 <=50% 或 =100% 都通过
        - 客户占比 <= 50% → 通过（混排Lane）
        - 客户占比 = 100% → 通过（客户专用Lane）
        - 50% < 客户占比 < 100% → 不通过（中间比例不允许）
        
        Args:
            libraries: 待检查的文库列表
            
        Returns:
            True表示满足客户占比约束，False表示不满足
        """
        if len(libraries) < 1:
            return True
        
        # 使用contract_data_raw字段，与验证器保持一致
        total_data = sum(float(getattr(lib, 'contract_data_raw', 0) or 0) for lib in libraries)
        if total_data == 0:
            return True
        
        # 统计客户文库数据量（使用is_customer_library方法）
        customer_data = 0.0
        for lib in libraries:
            if lib.is_customer_library():
                lib_data = float(getattr(lib, 'contract_data_raw', 0) or 0)
                customer_data += lib_data
        
        # 计算客户占比（按数据量）
        customer_ratio = customer_data / total_data if total_data > 0 else 0.0
        
        # 规则：<=50% 或 =100% 都通过
        # 只有在 >50% 且 !=100% 时才不通过
        if customer_ratio <= 0.50 or customer_ratio == 1.0:
            return True
        return False
    
    def _check_customer_ratio_near_limit(self, lane: LaneAssignment, lib: EnhancedLibraryInfo, threshold: float = 0.50) -> bool:
        """
        检查添加文库后客户占比是否接近限制
        
        [2025-12-31 新增] 严格策略：如果当前客户占比已接近50%，且待添加的是客户文库，拒绝
        [2026-02-06 修改] 只有Lane数据量达到容量下限后才检查占比，填充阶段不拦截
        
        Args:
            lane: 当前Lane
            lib: 待添加的文库
            threshold: 阈值，默认50%（接近50%限制）
            
        Returns:
            True表示可以添加，False表示接近限制，应该拒绝
        """
        if not lane.libraries:
            return True
        
        # 计算当前Lane数据总量
        current_total = sum(float(getattr(l, 'contract_data_raw', 0) or 0) for l in lane.libraries)
        if current_total == 0:
            return True
        
        # 还没装到容量下限之前，不检查占比，让文库先进来
        machine_type_str = lane.machine_type.value if lane.machine_type else "Nova X-25B"
        min_capacity, _ = self._resolve_lane_capacity_limits(
            lane.libraries,
            machine_type_str,
            lane=lane,
        )
        if current_total < min_capacity:
            return True
        
        current_customer = sum(float(getattr(l, 'contract_data_raw', 0) or 0) for l in lane.libraries if l.is_customer_library())
        current_customer_ratio = current_customer / current_total
        
        # 达到容量后才检查：如果客户占比已>=阈值，且待添加的是客户文库，拒绝
        if current_customer_ratio >= threshold and lib.is_customer_library():
            return False
        
        return True
    
    def _check_base_imbalance_ratio_near_limit(
        self,
        lane: LaneAssignment,
        lib: EnhancedLibraryInfo,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        检查添加文库后碱基不均衡占比是否接近限制
        
        [2025-12-31 新增] 严格策略：如果当前碱基不均衡占比已接近上限，且待添加的是碱基不均衡文库，拒绝
        
        Args:
            lane: 当前Lane
            lib: 待添加的文库
        threshold: 阈值，默认使用配置的碱基不均衡占比上限
            
        Returns:
            True表示可以添加，False表示接近限制，应该拒绝
        """
        if not lane.libraries:
            return True

        threshold = self.config.max_imbalance_ratio if threshold is None else threshold
        
        # 计算当前碱基不均衡占比
        current_total = sum(float(getattr(l, 'contract_data_raw', 0) or 0) for l in lane.libraries)
        if current_total == 0:
            return True
        
        current_imbalance = sum(float(getattr(l, 'contract_data_raw', 0) or 0) for l in lane.libraries if l.is_base_imbalance())
        current_imbalance_ratio = current_imbalance / current_total if current_total > 0 else 0.0
        
        # 如果当前碱基不均衡占比已>=阈值，且待添加的是碱基不均衡文库，拒绝
        if current_imbalance_ratio >= threshold and lib.is_base_imbalance():
            return False
        
        return True
    
    def _check_10bp_index_ratio_near_limit(
        self,
        lane: LaneAssignment,
        lib: EnhancedLibraryInfo,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        检查添加文库后10bp Index占比是否接近限制
        
        [2025-12-31 新增] 严格策略：如果当前10bp Index占比<阈值，且待添加的是非10bp文库，拒绝
        用于确保10bp Index占比始终>=配置下限
        
        Args:
            lane: 当前Lane
            lib: 待添加的文库
        threshold: 阈值，默认使用配置的10bp Index占比下限
            
        Returns:
            True表示可以添加，False表示接近限制，应该拒绝
        """
        if not lane.libraries:
            return True

        threshold = self.config.min_10bp_index_ratio if threshold is None else threshold
        
        # 计算当前10bp Index占比
        current_total = sum(float(getattr(l, 'contract_data_raw', 0) or 0) for l in lane.libraries)
        if current_total == 0:
            return True
        
        current_10bp = 0.0
        current_non_10bp = 0.0
        
        for l in lane.libraries:
            lib_data = float(getattr(l, 'contract_data_raw', 0) or 0)
            ten_bp_data = getattr(l, 'ten_bp_data', None)
            if ten_bp_data is not None and ten_bp_data > 0:
                current_10bp += lib_data
            elif self._library_has_10bp_index(l):
                current_10bp += lib_data
            else:
                current_non_10bp += lib_data
        
        # 如果当前没有10bp或非10bp，不需要检查
        if current_10bp == 0 or current_non_10bp == 0:
            return True
        
        current_10bp_ratio = current_10bp / current_total if current_total > 0 else 0.0
        
        # 如果当前10bp Index占比<阈值，且待添加的是非10bp文库，拒绝
        if current_10bp_ratio < threshold:
            lib_is_10bp = False
            ten_bp_data = getattr(lib, 'ten_bp_data', None)
            if ten_bp_data is not None and ten_bp_data > 0:
                lib_is_10bp = True
            else:
                lib_is_10bp = self._library_has_10bp_index(lib)
            
            if not lib_is_10bp:
                return False
        
        return True
    
    def _calculate_required_removal_for_customer_ratio(
        self, 
        lane: LaneAssignment, 
        lib: EnhancedLibraryInfo, 
        max_ratio: float = 0.50
    ) -> float:
        """
        计算添加文库后，需要踢出多少数据量的客户文库才能保持客户占比符合规则
        
        [2025-12-31 新增] 动态调整策略：当添加文库会导致客户占比违反规则时，计算需要踢出的客户文库数据量
        规则：客户占比 <=50% 或 =100% 都可以，其他则不行
        
        Args:
            lane: 当前Lane
            lib: 待添加的文库
            max_ratio: 最大允许的客户占比（混排场景），默认50%
            
        Returns:
            需要踢出的客户文库数据量（GB），如果不需要踢出则返回0.0
        """
        if not lane.libraries:
            return 0.0
        
        # 计算添加后的总数据量和客户数据量
        current_total = sum(float(getattr(l, 'contract_data_raw', 0) or 0) for l in lane.libraries)
        current_customer = sum(float(getattr(l, 'contract_data_raw', 0) or 0) for l in lane.libraries if l.is_customer_library())
        
        lib_data = float(getattr(lib, 'contract_data_raw', 0) or 0)
        new_total = current_total + lib_data
        new_customer = current_customer + (lib_data if lib.is_customer_library() else 0)
        
        # 如果添加后客户占比<=max_ratio，不需要踢出
        if new_total == 0:
            return 0.0
        
        new_customer_ratio = new_customer / new_total
        
        # 规则：<=50% 或 =100% 都可以，其他则不行
        # 如果客户占比<=50%，不需要踢出
        if new_customer_ratio <= max_ratio:
            return 0.0
        
        # 如果客户占比=100%（或接近100%），允许，不需要踢出
        if abs(new_customer_ratio - 1.0) < 1e-6:
            return 0.0
        
        # 如果客户占比在50%和100%之间，需要踢出客户文库使其<=50%或=100%
        # 策略：尝试踢出客户文库，使客户占比<=50%
        # 设需要踢出x GB的客户文库数据
        # (new_customer - x) / (new_total - x) <= max_ratio
        # new_customer - x <= max_ratio * (new_total - x)
        # new_customer - x <= max_ratio * new_total - max_ratio * x
        # new_customer - max_ratio * new_total <= x - max_ratio * x
        # new_customer - max_ratio * new_total <= x * (1 - max_ratio)
        # x >= (new_customer - max_ratio * new_total) / (1 - max_ratio)
        
        required_removal = (new_customer - max_ratio * new_total) / (1 - max_ratio)
        
        # 确保返回值非负
        return max(0.0, required_removal)
    
    def _calculate_required_removal_for_base_imbalance_ratio(
        self,
        lane: LaneAssignment,
        lib: EnhancedLibraryInfo,
        max_ratio: Optional[float] = None,
    ) -> float:
        """
        计算添加文库后，需要踢出多少数据量的碱基不均衡文库才能保持碱基不均衡占比<=max_ratio
        
        Args:
            lane: 当前Lane
            lib: 待添加的文库
        max_ratio: 最大允许的碱基不均衡占比，默认使用配置值
            
        Returns:
            需要踢出的碱基不均衡文库数据量（GB），如果不需要踢出则返回0.0
        """
        if not lane.libraries:
            return 0.0

        max_ratio = self.config.max_imbalance_ratio if max_ratio is None else max_ratio
        
        # 计算添加后的总数据量和碱基不均衡数据量
        current_total = sum(float(getattr(l, 'contract_data_raw', 0) or 0) for l in lane.libraries)
        current_imbalance = sum(float(getattr(l, 'contract_data_raw', 0) or 0) for l in lane.libraries if l.is_base_imbalance())
        
        lib_data = float(getattr(lib, 'contract_data_raw', 0) or 0)
        new_total = current_total + lib_data
        new_imbalance = current_imbalance + (lib_data if lib.is_base_imbalance() else 0)
        
        # 如果添加后碱基不均衡占比<=max_ratio，不需要踢出
        if new_total == 0:
            return 0.0
        
        new_imbalance_ratio = new_imbalance / new_total
        if new_imbalance_ratio <= max_ratio:
            return 0.0
        
        # 计算需要踢出的碱基不均衡文库数据量
        required_removal = (new_imbalance - max_ratio * new_total) / (1 - max_ratio)
        
        # 确保返回值非负
        return max(0.0, required_removal)
    
    def _calculate_required_removal_for_10bp_index_ratio(
        self,
        lane: LaneAssignment,
        lib: EnhancedLibraryInfo,
        min_ratio: Optional[float] = None,
    ) -> float:
        """
        计算添加文库后，需要踢出多少数据量的非10bp文库才能保持10bp Index占比>=min_ratio
        
        Args:
            lane: 当前Lane
            lib: 待添加的文库
        min_ratio: 最小允许的10bp Index占比，默认使用配置值
            
        Returns:
            需要踢出的非10bp文库数据量（GB），如果不需要踢出则返回0.0
        """
        if not lane.libraries:
            return 0.0

        min_ratio = self.config.min_10bp_index_ratio if min_ratio is None else min_ratio
        
        # 计算当前10bp和非10bp数据量
        current_total = sum(float(getattr(l, 'contract_data_raw', 0) or 0) for l in lane.libraries)
        current_10bp = 0.0
        current_non_10bp = 0.0
        
        for l in lane.libraries:
            lib_data = float(getattr(l, 'contract_data_raw', 0) or 0)
            ten_bp_data = getattr(l, 'ten_bp_data', None)
            if ten_bp_data is not None and ten_bp_data > 0:
                current_10bp += lib_data
            elif self._library_has_10bp_index(l):
                current_10bp += lib_data
            else:
                current_non_10bp += lib_data
        
        # 判断待添加文库是否为10bp
        lib_data = float(getattr(lib, 'contract_data_raw', 0) or 0)
        lib_is_10bp = False
        ten_bp_data = getattr(lib, 'ten_bp_data', None)
        if ten_bp_data is not None and ten_bp_data > 0:
            lib_is_10bp = True
        else:
            lib_is_10bp = self._library_has_10bp_index(lib)
        
        # 计算添加后的数据量
        new_total = current_total + lib_data
        new_10bp = current_10bp + (lib_data if lib_is_10bp else 0)
        new_non_10bp = current_non_10bp + (lib_data if not lib_is_10bp else 0)
        
        # 如果添加后10bp Index占比>=min_ratio，不需要踢出
        if new_total == 0:
            return 0.0
        
        new_10bp_ratio = new_10bp / new_total
        if new_10bp_ratio >= min_ratio:
            return 0.0
        
        # 计算需要踢出的非10bp文库数据量
        # 设需要踢出x GB的非10bp文库数据
        # new_10bp / (new_total - x) >= min_ratio
        # new_10bp >= min_ratio * (new_total - x)
        # new_10bp >= min_ratio * new_total - min_ratio * x
        # min_ratio * x >= min_ratio * new_total - new_10bp
        # x >= (min_ratio * new_total - new_10bp) / min_ratio
        # x >= new_total - new_10bp / min_ratio
        
        required_removal = new_total - new_10bp / min_ratio
        
        # 确保返回值非负且不超过当前非10bp数据量
        return max(0.0, min(required_removal, new_non_10bp))
    
    def _check_10bp_index_ratio_compatible(self, libraries: List[EnhancedLibraryInfo]) -> bool:
        """
        检查文库列表的10bp Index占比是否满足约束
        
        [2025-12-31 新增] 严格遵守验证规则：非NB Lane需要10bp Index占比>=配置下限
        与LaneValidator的验证逻辑保持一致
        
        Args:
            libraries: 待检查的文库列表
            
        Returns:
            True表示满足10bp Index占比约束，False表示不满足
        """
        if len(libraries) < 1:
            return True
        
        total_data = sum(lib.get_data_amount_gb() for lib in libraries)
        if total_data == 0:
            return True
        
        # 统计10bp Index和非10bp Index数据量（与验证器逻辑一致）
        data_10bp = 0.0
        data_non_10bp = 0.0
        
        for lib in libraries:
            lib_data = lib.get_data_amount_gb()
            is_10bp = False
            
            # 优先使用预处理好的ten_bp_data字段（与验证器一致）
            ten_bp_data = getattr(lib, 'ten_bp_data', None)
            if ten_bp_data is not None and ten_bp_data > 0:
                is_10bp = True
            else:
                # 回退到检查index_seq
                is_10bp = self._library_has_10bp_index(lib)
            
            if is_10bp:
                data_10bp += lib_data
            else:
                data_non_10bp += lib_data
        
        # 仅当混排时才检查（纯10bp或纯非10bp都通过）
        if data_10bp == 0 or data_non_10bp == 0:
            return True
        
        # 计算10bp Index占比（按数据量）
        ratio_10bp = data_10bp / total_data if total_data > 0 else 0.0
        
        # 规则：10bp Index占比 >= 配置下限
        return ratio_10bp >= self.config.min_10bp_index_ratio
    
    def _check_base_imbalance_compatible(self, libraries: List[EnhancedLibraryInfo]) -> bool:
        """
        检查文库列表的碱基不均衡占比是否满足约束
        
        [2025-12-26 新增] 根据规则文档：
        - 碱基不均衡专用Lane（DL Lane）：允许100%，不受限制
        - 普通Lane/混排Lane：碱基不均衡占比 < 配置上限
        - NB Lane（非10bp专用）也需要受此限制
        
        Args:
            libraries: 待检查的文库列表
            
        Returns:
            True表示满足碱基不均衡占比约束，False表示不满足
        """
        if len(libraries) < 1:
            return True
        
        total_data = sum(lib.get_data_amount_gb() for lib in libraries)
        if total_data == 0:
            return True
        
        # 统计碱基不均衡文库数据量
        imbalance_data = 0.0
        for lib in libraries:
            if lib.is_base_imbalance():
                imbalance_data += lib.get_data_amount_gb()
        
        # 如果没有碱基不均衡文库，通过
        if imbalance_data == 0:
            return True
        
        # 碱基不均衡占比 <= 配置上限
        imbalance_ratio = imbalance_data / total_data
        return imbalance_ratio <= self.config.max_imbalance_ratio

    def _check_single_end_ratio_compatible(self, libraries: List[EnhancedLibraryInfo]) -> bool:
        """
        检查单端Index占比是否满足约束（按数据量计算）

        规则：单端Index占比 < 30%，与LaneValidator逻辑保持一致
        """
        if len(libraries) < 1:
            return True

        total_data = sum(float(getattr(lib, 'contract_data_raw', 0) or 0) for lib in libraries)
        if total_data == 0:
            return True

        single_end_data = 0.0
        for lib in libraries:
            lib_data = float(getattr(lib, 'contract_data_raw', 0) or 0)
            is_single_end = False

            # 优先使用预处理字段
            single_index_data = getattr(lib, 'single_index_data', None)
            if single_index_data is not None and single_index_data > 0:
                is_single_end = True
            else:
                index_seq = getattr(lib, 'index_seq', '') or ''
                is_single_end = self._is_single_end_index(index_seq)

            if is_single_end:
                single_end_data += lib_data

        if single_end_data == 0:
            return True

        single_end_ratio = single_end_data / total_data
        return single_end_ratio < self.lane_validator.single_end_ratio_limit

    def _is_single_end_index(self, index_seq: str) -> bool:
        """判断是否为单端Index（与LaneValidator保持一致）"""
        if not index_seq:
            return True

        sequences = index_seq.split(',')
        for seq in sequences:
            if ';' in seq:
                return False
        return True

    def _check_special_library_type_compatible(self, libraries: List[EnhancedLibraryInfo]) -> bool:
        """特殊文库种类数不再作为排机约束。"""
        _ = libraries
        return True

    def _check_add_test_ratio_compatible(self, libraries: List[EnhancedLibraryInfo]) -> bool:
        """
        检查加测文库占比是否满足约束（严格模式下视为硬约束）
        """
        if len(libraries) < 1:
            return True

        total_data = sum(float(getattr(lib, 'contract_data_raw', 0) or 0) for lib in libraries)
        if total_data == 0:
            return True

        add_test_data = 0.0
        for lib in libraries:
            remark = getattr(lib, 'remark', '') or getattr(lib, '上机备注', '') or ''
            add_test_remark = getattr(lib, 'add_test_remark', '') or ''
            if '加测' in remark or '加测' in add_test_remark:
                add_test_data += float(getattr(lib, 'contract_data_raw', 0) or 0)

        if add_test_data == 0:
            return True

        effective_limit = self.lane_validator.add_test_ratio_limit + (
            self.lane_validator.add_test_buffer_gb / total_data
        )
        add_test_ratio = add_test_data / total_data
        return add_test_ratio <= effective_limit
    
    def _library_has_10bp_index(self, lib: EnhancedLibraryInfo) -> bool:
        """
        检查文库是否有10bp Index
        
        [2025-12-25 新增] 用于NB Lane类型约束检查
        
        Args:
            lib: 待检查的文库
            
        Returns:
            True表示有10bp Index，False表示没有
        """
        # 优先使用预处理好的ten_bp_data字段
        ten_bp_data = getattr(lib, 'ten_bp_data', None)
        if ten_bp_data is not None and ten_bp_data > 0:
            return True
        
        # 回退到解析index_seq
        index_seq = getattr(lib, 'index_seq', '') or ''
        if not index_seq:
            return False
        
        # 只看P7端长度（分号前的部分）
        if ';' in index_seq:
            p7_part = index_seq.split(';')[0]
        else:
            p7_part = index_seq
        
        # 检查是否有任何一个Index达到10碱基
        for idx in p7_part.split(','):
            idx = idx.strip()
            if idx and len(idx) >= 10:
                return True
        
        return False
    
    def _schedule_small_library_clustering_lanes(
        self,
        libraries: List[EnhancedLibraryInfo],
        machine_type: str
    ) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo]]:
        """
        [2025-12-25 新增] 小文库聚类专用Lane策略
        [2025-12-26 优化] 预处理分流：高客户占比区间与低客户占比区间混合组Lane
        
        根据人工排机数据分析：
        - 886个5-6G文库被集中到3-5条专用Lane
        - Lane 328692: 163个6G文库，978GB
        - Lane 328694: 151个6-8G文库，976GB
        - Lane 328755: 164个6G文库，977GB
        
        [2025-12-26 问题分析]
        - 8-10G区间客户占比高达87.7%，无法单独组Lane通过客户<=50%验证
        - 解决方案：将高客户占比区间的文库与低客户占比区间混合组Lane
        
        策略：
        1. 识别数据量<=20G的小文库
        2. 按数据量区间聚类，并分析各区间客户占比
        3. 高客户占比区间（>50%）不单独形成专用Lane
        4. 将高客户占比区间文库与低客户占比区间文库混合组Lane
        5. 混合组Lane时确保客户占比<=50%
        
        Args:
            libraries: 文库列表
            machine_type: 机器类型
            
        Returns:
            (专用Lane列表, 剩余未分配的文库)
        """
        # 获取配置参数
        threshold = getattr(self.config, 'small_library_threshold_gb', 20.0)
        min_count = getattr(self.config, 'clustering_min_count', 100)
        bin_width = getattr(self.config, 'clustering_bin_width_gb', 1.0)
        
        # 分离小文库和大文库
        small_libs = []
        large_libs = []
        for lib in libraries:
            data_gb = lib.get_data_amount_gb()
            if data_gb <= threshold:
                small_libs.append(lib)
            else:
                large_libs.append(lib)
        
        if not small_libs:
            return [], libraries
        
        logger.info(f"小文库聚类分析: {len(small_libs)}个小文库(<=20G), {len(large_libs)}个大文库(>20G)")
        
        # 按数据量区间聚类
        clusters: Dict[int, List[EnhancedLibraryInfo]] = {}
        for lib in small_libs:
            data_gb = lib.get_data_amount_gb()
            bin_idx = int(data_gb / bin_width)
            if bin_idx not in clusters:
                clusters[bin_idx] = []
            clusters[bin_idx].append(lib)
        
        # 找出足够大的聚类（文库数>=min_count）
        large_clusters = {k: v for k, v in clusters.items() if len(v) >= min_count}
        
        if not large_clusters:
            logger.info(f"未找到足够大的聚类（最小要求{min_count}个文库）")
            return [], libraries
        
        # [2025-12-26 新增] 分析各区间客户占比，识别高客户占比区间
        high_customer_ratio_bins = []  # 高客户占比区间（>50%）
        low_customer_ratio_bins = []   # 低客户占比区间（<=50%）
        
        for bin_idx, cluster_libs in sorted(large_clusters.items(), key=lambda x: -len(x[1])):
            bin_start = bin_idx * bin_width
            bin_end = (bin_idx + 1) * bin_width
            total_data = sum(lib.get_data_amount_gb() for lib in cluster_libs)
            
            # 计算客户占比
            customer_data = sum(lib.get_data_amount_gb() for lib in cluster_libs if lib.is_customer_library())
            customer_ratio = customer_data / total_data if total_data > 0 else 0
            
            logger.info(f"  聚类 {bin_start:.0f}-{bin_end:.0f}G: {len(cluster_libs)}个文库, "
                       f"总量{total_data:.1f}GB, 可组{total_data/self.config.lane_capacity_gb:.1f}条Lane, "
                       f"客户占比{customer_ratio:.1%}")
            
            if customer_ratio > 0.50:
                high_customer_ratio_bins.append(bin_idx)
            else:
                low_customer_ratio_bins.append(bin_idx)
        
        # 将字符串机器类型转换为枚举
        machine_type_enum = self._resolve_machine_type_enum(machine_type, libraries)
        if machine_type_enum == MachineType.UNKNOWN:
            machine_type_enum = MachineType.NOVA_X_25B
        
        clustering_lanes: List[LaneAssignment] = []
        used_libs: Set[str] = set()
        
        # ========== 阶段1：为低客户占比区间形成专用Lane ==========
        for bin_idx in low_customer_ratio_bins:
            cluster_libs = large_clusters[bin_idx]
            bin_start = bin_idx * bin_width
            bin_end = (bin_idx + 1) * bin_width
            
            sorted_cluster = self._sort_remaining_for_scattered_mix_lane(cluster_libs)
            available = [
                lib for lib in sorted_cluster
                if self._get_library_runtime_key(lib) not in used_libs
            ]
            
            if len(available) < min_count:
                continue
            
            remaining_cluster = list(available)
            
            while len(remaining_cluster) >= 50:
                lane_id = self._get_next_lane_id("SL", machine_type)
                current_lane = LaneAssignment(
                    lane_id=lane_id,
                    machine_id=f"M_{lane_id[3:]}",
                    machine_type=machine_type_enum,
                    lane_capacity_gb=self.config.lane_capacity_gb
                )
                
                next_remaining: List[EnhancedLibraryInfo] = []
                for lib in remaining_cluster:
                    test_libs_for_capacity = current_lane.libraries + [lib]
                    _, max_capacity = self._resolve_lane_capacity_limits(
                        test_libs_for_capacity,
                        machine_type,
                        lane=current_lane,
                    )
                    new_total = current_lane.total_data_gb + lib.get_data_amount_gb()
                    
                    if new_total > max_capacity:
                        next_remaining.append(lib)
                        continue
                    
                    if self.config.enable_index_check:
                        test_libs = current_lane.libraries + [lib]
                        if not self.index_validator.validate_lane_quick(test_libs):
                            next_remaining.append(lib)
                            continue
                    
                    test_libs_for_peak = current_lane.libraries + [lib]
                    if not self._check_peak_size_compatible(test_libs_for_peak):
                        next_remaining.append(lib)
                        continue
                    
                    # [2025-12-31 新增] 客户占比接近限制检查（严格策略）
                    if not self._check_customer_ratio_near_limit(current_lane, lib, threshold=0.50):
                        next_remaining.append(lib)
                        continue
                    
                    # [2025-12-31 新增] 客户占比检查（按数据量计算，严格遵守规则）
                    test_libs_for_customer = current_lane.libraries + [lib]
                    if not self._check_customer_ratio_compatible_by_data(test_libs_for_customer):
                        next_remaining.append(lib)
                        continue
                    
                    # [2025-12-31 新增] 碱基不均衡占比接近限制检查（严格策略）
                    if not self._check_base_imbalance_ratio_near_limit(
                        current_lane, lib, threshold=self.config.max_imbalance_ratio
                    ):
                        next_remaining.append(lib)
                        continue
                    
                    # [2025-12-31 新增] 碱基不均衡占比检查
                    test_libs_for_imbalance = current_lane.libraries + [lib]
                    if not self._check_base_imbalance_compatible(test_libs_for_imbalance):
                        next_remaining.append(lib)
                        continue
                    
                    # [2025-12-31 新增] 10bp Index占比接近限制检查（严格策略）
                    if not self._check_10bp_index_ratio_near_limit(
                        current_lane, lib, threshold=self.config.min_10bp_index_ratio
                    ):
                        next_remaining.append(lib)
                        continue
                    
                    # [2025-12-31 新增] 10bp Index占比检查
                    test_libs_for_10bp = current_lane.libraries + [lib]
                    if not self._check_10bp_index_ratio_compatible(test_libs_for_10bp):
                        next_remaining.append(lib)
                        continue
                    
                    current_lane.add_library(lib)
                    used_libs.add(self._get_library_runtime_key(lib))
                
                if not current_lane.libraries:
                    break
                
                utilization = current_lane.total_data_gb / self.config.lane_capacity_gb
                min_allowed, _ = self._resolve_lane_capacity_limits(
                    current_lane.libraries,
                    machine_type,
                    lane=current_lane,
                )
                if current_lane.total_data_gb >= min_allowed:
                    # [2025-12-31 新增] 最终完整验证：检查所有规则
                    validation_passed = True
                    
                    # 1. Index冲突检查
                    if self.config.enable_index_check:
                        if not self.index_validator.validate_lane_quick(current_lane.libraries):
                            logger.warning(f"小文库专用Lane {current_lane.lane_id} Index冲突，拒绝形成")
                            validation_passed = False
                    
                    # 2. 客户占比检查（按数据量计算，严格遵守规则）
                    if validation_passed and not self._check_customer_ratio_compatible_by_data(current_lane.libraries):
                        logger.warning(f"小文库专用Lane {current_lane.lane_id} 客户占比不符合规则，拒绝形成")
                        validation_passed = False
                    
                    # 3. 碱基不均衡占比检查
                    if validation_passed and not self._check_base_imbalance_compatible(current_lane.libraries):
                        logger.warning(f"小文库专用Lane {current_lane.lane_id} 碱基不均衡占比不符合规则，拒绝形成")
                        validation_passed = False
                    
                    # 4. Peak Size检查
                    if validation_passed and not self._check_peak_size_compatible(current_lane.libraries):
                        logger.warning(f"小文库专用Lane {current_lane.lane_id} Peak Size不符合规则，拒绝形成")
                        validation_passed = False
                    
                    # 5. 10bp Index占比检查
                    if validation_passed and not self._check_10bp_index_ratio_compatible(current_lane.libraries):
                        logger.warning(f"小文库专用Lane {current_lane.lane_id} 10bp Index占比不符合规则，拒绝形成")
                        validation_passed = False
                    
                    if validation_passed:
                        # 所有检查通过，添加Lane
                        clustering_lanes.append(current_lane)
                        logger.info(f"小文库专用Lane {current_lane.lane_id} 形成成功 - "
                                   f"聚类区间: {bin_start:.0f}-{bin_end:.0f}G, "
                                   f"文库数: {len(current_lane.libraries)}, "
                                   f"数据量: {current_lane.total_data_gb:.1f}GB, "
                                   f"利用率: {utilization:.1%}")
                        remaining_cluster = next_remaining
                    else:
                        # 验证失败，回退
                        for lib in current_lane.libraries:
                            used_libs.discard(self._get_library_runtime_key(lib))
                        break
                else:
                    for lib in current_lane.libraries:
                        used_libs.discard(self._get_library_runtime_key(lib))
                    break
        
        # ========== 阶段2：预处理分流 - 高客户占比区间与低客户占比区间混合 ==========
        if high_customer_ratio_bins:
            logger.info(f"预处理分流: 发现{len(high_customer_ratio_bins)}个高客户占比区间，尝试跨区间混合组Lane")
            
            # 收集高客户占比区间的未使用文库
            high_ratio_libs = []
            for bin_idx in high_customer_ratio_bins:
                for lib in large_clusters[bin_idx]:
                    if self._get_library_runtime_key(lib) not in used_libs:
                        high_ratio_libs.append(lib)
            
            # 收集低客户占比区间的剩余内部文库（非客户文库）
            low_ratio_internal_libs = []
            for bin_idx in low_customer_ratio_bins:
                for lib in large_clusters[bin_idx]:
                    if self._get_library_runtime_key(lib) not in used_libs and not lib.is_customer_library():
                        low_ratio_internal_libs.append(lib)
            
            # 收集其他未使用的小文库中的内部文库
            other_internal_libs = []
            for lib in small_libs:
                if self._get_library_runtime_key(lib) not in used_libs and not lib.is_customer_library():
                    # 检查是否不在大聚类中
                    data_gb = lib.get_data_amount_gb()
                    bin_idx = int(data_gb / bin_width)
                    if bin_idx not in large_clusters:
                        other_internal_libs.append(lib)
            
            # 合并所有可用的内部文库
            all_internal_libs = low_ratio_internal_libs + other_internal_libs
            
            if high_ratio_libs and all_internal_libs:
                logger.info(f"预处理分流: {len(high_ratio_libs)}个高客户区间文库, "
                           f"{len(all_internal_libs)}个可用内部文库")
                
                # 按数据量排序
                high_ratio_libs = self._sort_remaining_for_scattered_mix_lane(high_ratio_libs)
                all_internal_libs = self._sort_remaining_for_scattered_mix_lane(all_internal_libs)
                
                # 尝试混合组Lane（确保客户占比<=50%）
                mixed_lanes = self._form_mixed_customer_lanes(
                    high_ratio_libs, 
                    all_internal_libs, 
                    machine_type_enum,
                    machine_type,
                    used_libs
                )
                
                clustering_lanes.extend(mixed_lanes)
                logger.info(f"预处理分流: 成功形成{len(mixed_lanes)}条混合Lane")
        
        # 收集剩余文库
        remaining_libs = [
            lib for lib in libraries
            if self._get_library_runtime_key(lib) not in used_libs
        ]
        
        return clustering_lanes, remaining_libs
    
    def _form_mixed_customer_lanes(
        self,
        high_customer_libs: List[EnhancedLibraryInfo],
        internal_libs: List[EnhancedLibraryInfo],
        machine_type_enum: MachineType,
        machine_type: str,
        used_libs: Set[str]
    ) -> List[LaneAssignment]:
        """
        [2025-12-26 新增] 形成混合Lane，确保客户占比<=50%
        
        策略：
        1. 优先加入内部文库（非客户文库）
        2. 然后加入客户文库，但确保客户数据占比不超过50%
        3. 同时检查Index冲突和Peak Size约束
        
        Args:
            high_customer_libs: 高客户占比区间的文库（可能包含客户和内部文库）
            internal_libs: 可用的内部文库
            machine_type_enum: 机器类型枚举
            machine_type: 机器类型字符串
            used_libs: 已使用的文库ID集合
            
        Returns:
            混合Lane列表
        """
        mixed_lanes: List[LaneAssignment] = []
        
        # 分离高客户区间中的客户文库和内部文库
        high_customer_customer_libs = [lib for lib in high_customer_libs if lib.is_customer_library()]
        high_customer_internal_libs = [lib for lib in high_customer_libs if not lib.is_customer_library()]
        
        # 合并所有内部文库
        all_internal = internal_libs + high_customer_internal_libs
        all_internal = [
            lib for lib in all_internal
            if self._get_library_runtime_key(lib) not in used_libs
        ]
        
        # 客户文库
        customer_libs = [
            lib for lib in high_customer_customer_libs
            if self._get_library_runtime_key(lib) not in used_libs
        ]
        
        if not customer_libs:
            return []
        
        logger.info(f"混合组Lane: {len(customer_libs)}个客户文库待分配, {len(all_internal)}个内部文库可用")
        
        # 尝试形成混合Lane
        max_attempts = 10  # 最多尝试形成10条混合Lane
        for attempt in range(max_attempts):
            # 检查是否还有足够的文库
            remaining_customer = [
                lib for lib in customer_libs
                if self._get_library_runtime_key(lib) not in used_libs
            ]
            remaining_internal = [
                lib for lib in all_internal
                if self._get_library_runtime_key(lib) not in used_libs
            ]
            remaining_customer = self._sort_remaining_for_scattered_mix_lane(remaining_customer)
            remaining_internal = self._sort_remaining_for_scattered_mix_lane(remaining_internal)
            
            if not remaining_customer:
                break
            
            # 计算需要多少内部文库数据来平衡客户文库
            customer_total_data = sum(lib.get_data_amount_gb() for lib in remaining_customer)
            
            # 如果剩余内部文库不足以平衡，只处理部分客户文库
            internal_total_data = sum(lib.get_data_amount_gb() for lib in remaining_internal)
            
            if internal_total_data < customer_total_data * 0.5:
                # 内部文库不足，无法再形成更多混合Lane
                logger.info(f"混合组Lane: 内部文库数据量不足（{internal_total_data:.0f}GB），停止")
                break
            
            # 创建新Lane
            lane_id = self._get_next_lane_id("SL", machine_type)
            current_lane = LaneAssignment(
                lane_id=lane_id,
                machine_id=f"M_{lane_id[3:]}",
                machine_type=machine_type_enum,
                lane_capacity_gb=self.config.lane_capacity_gb
            )
            
            target_capacity = self.config.lane_capacity_gb * 1.15  # 目标115%利用率
            
            # 第一步：加入内部文库（目标：占Lane的50%以上）
            internal_target = target_capacity * 0.55  # 内部文库占55%
            for lib in remaining_internal:
                if current_lane.total_data_gb >= internal_target:
                    break
                
                test_libs_for_capacity = current_lane.libraries + [lib]
                _, max_capacity = self._resolve_lane_capacity_limits(
                    test_libs_for_capacity,
                    machine_type,
                    lane=current_lane,
                )
                new_total = current_lane.total_data_gb + lib.get_data_amount_gb()
                if new_total > max_capacity:
                    continue
                
                # Index冲突检查
                if self.config.enable_index_check:
                    test_libs = current_lane.libraries + [lib]
                    if not self.index_validator.validate_lane_quick(test_libs):
                        continue
                
                # Peak Size检查
                test_libs_for_peak = current_lane.libraries + [lib]
                if not self._check_peak_size_compatible(test_libs_for_peak):
                    continue
                
                # [2025-12-31 新增] 碱基不均衡占比接近限制检查（严格策略）
                if not self._check_base_imbalance_ratio_near_limit(
                    current_lane, lib, threshold=self.config.max_imbalance_ratio
                ):
                    continue
                
                # [2025-12-31 新增] 碱基不均衡占比检查
                test_libs_for_imbalance = current_lane.libraries + [lib]
                if not self._check_base_imbalance_compatible(test_libs_for_imbalance):
                    continue
                
                # [2025-12-31 新增] 10bp Index占比接近限制检查（严格策略）
                if not self._check_10bp_index_ratio_near_limit(
                    current_lane, lib, threshold=self.config.min_10bp_index_ratio
                ):
                    continue
                
                # [2025-12-31 新增] 10bp Index占比检查
                test_libs_for_10bp = current_lane.libraries + [lib]
                if not self._check_10bp_index_ratio_compatible(test_libs_for_10bp):
                    continue
                
                current_lane.add_library(lib)
                used_libs.add(self._get_library_runtime_key(lib))
            
            internal_data_in_lane = current_lane.total_data_gb
            
            if internal_data_in_lane < self.config.lane_capacity_gb * 0.3:
                # 内部文库太少，无法形成有效的混合Lane
                for lib in current_lane.libraries:
                    used_libs.discard(self._get_library_runtime_key(lib))
                break
            
            # 第二步：加入客户文库（确保客户占比<=50%）
            for lib in remaining_customer:
                test_libs_for_capacity = current_lane.libraries + [lib]
                _, max_capacity = self._resolve_lane_capacity_limits(
                    test_libs_for_capacity,
                    machine_type,
                    lane=current_lane,
                )
                new_total = current_lane.total_data_gb + lib.get_data_amount_gb()
                if new_total > max_capacity:
                    continue
                
                # 计算加入后的客户占比
                customer_data_in_lane = sum(
                    l.get_data_amount_gb() for l in current_lane.libraries if l.is_customer_library()
                ) + lib.get_data_amount_gb()
                customer_ratio = customer_data_in_lane / new_total
                
                if customer_ratio > 0.50:
                    # 客户占比会超过50%，跳过
                    continue
                
                # Index冲突检查
                if self.config.enable_index_check:
                    test_libs = current_lane.libraries + [lib]
                    if not self.index_validator.validate_lane_quick(test_libs):
                        continue
                
                # Peak Size检查
                test_libs_for_peak = current_lane.libraries + [lib]
                if not self._check_peak_size_compatible(test_libs_for_peak):
                    continue
                
                # [2025-12-31 新增] 客户占比接近限制检查（严格策略）
                if not self._check_customer_ratio_near_limit(current_lane, lib, threshold=0.50):
                    continue
                
                # [2025-12-31 新增] 碱基不均衡占比接近限制检查（严格策略）
                if not self._check_base_imbalance_ratio_near_limit(
                    current_lane, lib, threshold=self.config.max_imbalance_ratio
                ):
                    continue
                
                # [2025-12-31 新增] 碱基不均衡占比检查
                test_libs_for_imbalance = current_lane.libraries + [lib]
                if not self._check_base_imbalance_compatible(test_libs_for_imbalance):
                    continue
                
                # [2025-12-31 新增] 10bp Index占比接近限制检查（严格策略）
                if not self._check_10bp_index_ratio_near_limit(
                    current_lane, lib, threshold=self.config.min_10bp_index_ratio
                ):
                    continue
                
                # [2025-12-31 新增] 10bp Index占比检查
                test_libs_for_10bp = current_lane.libraries + [lib]
                if not self._check_10bp_index_ratio_compatible(test_libs_for_10bp):
                    continue
                
                current_lane.add_library(lib)
                used_libs.add(self._get_library_runtime_key(lib))
            
            # 检查Lane是否有效
            utilization = current_lane.total_data_gb / self.config.lane_capacity_gb
            min_allowed, _ = self._resolve_lane_capacity_limits(
                current_lane.libraries,
                machine_type,
                lane=current_lane,
            )
            if current_lane.total_data_gb >= min_allowed:
                # [2025-12-31 新增] 最终完整验证：检查所有规则
                # 1. 客户占比检查（按数据量计算，与验证器一致）
                final_total = sum(float(getattr(lib, 'contract_data_raw', 0) or 0) for lib in current_lane.libraries)
                final_customer = sum(float(getattr(lib, 'contract_data_raw', 0) or 0) for lib in current_lane.libraries if lib.is_customer_library())
                final_customer_ratio = final_customer / final_total if final_total > 0 else 0.0
                
                # 规则：客户占比 <=50% 或 =100% 都可以，其他则不行
                if final_customer_ratio > 0.50 and abs(final_customer_ratio - 1.0) > 1e-6:
                    logger.warning(f"混合Lane {current_lane.lane_id} 客户占比{final_customer_ratio:.1%}不符合规则，拒绝形成")
                    for lib in current_lane.libraries:
                        used_libs.discard(self._get_library_runtime_key(lib))
                    continue
                
                # 2. Index冲突检查
                if self.config.enable_index_check:
                    if not self.index_validator.validate_lane_quick(current_lane.libraries):
                        logger.warning(f"混合Lane {current_lane.lane_id} Index冲突，拒绝形成")
                        for lib in current_lane.libraries:
                            used_libs.discard(self._get_library_runtime_key(lib))
                        continue
                
                # 3. 碱基不均衡占比检查
                if not self._check_base_imbalance_compatible(current_lane.libraries):
                    logger.warning(f"混合Lane {current_lane.lane_id} 碱基不均衡占比不符合规则，拒绝形成")
                    for lib in current_lane.libraries:
                        used_libs.discard(self._get_library_runtime_key(lib))
                    continue
                
                # 4. Peak Size检查
                if not self._check_peak_size_compatible(current_lane.libraries):
                    logger.warning(f"混合Lane {current_lane.lane_id} Peak Size不符合规则，拒绝形成")
                    for lib in current_lane.libraries:
                        used_libs.discard(self._get_library_runtime_key(lib))
                    continue
                
                # 5. 10bp Index占比检查
                if not self._check_10bp_index_ratio_compatible(current_lane.libraries):
                    logger.warning(f"混合Lane {current_lane.lane_id} 10bp Index占比不符合规则，拒绝形成")
                    for lib in current_lane.libraries:
                        used_libs.discard(self._get_library_runtime_key(lib))
                    continue
                
                # 所有检查通过，添加Lane
                mixed_lanes.append(current_lane)
                customer_count = sum(1 for l in current_lane.libraries if l.is_customer_library())
                internal_count = len(current_lane.libraries) - customer_count
                logger.info(f"混合Lane {current_lane.lane_id} 形成成功 - "
                           f"文库数: {len(current_lane.libraries)} (客户{customer_count}+内部{internal_count}), "
                           f"数据量: {current_lane.total_data_gb:.1f}GB, "
                           f"客户占比: {final_customer_ratio:.1%}, "
                           f"利用率: {utilization:.1%}")
            else:
                # 容量不足，回退
                for lib in current_lane.libraries:
                    used_libs.discard(self._get_library_runtime_key(lib))
        
        return mixed_lanes
    
    def _validate_small_library_lane(self, lane: LaneAssignment) -> bool:
        """
        验证小文库专用Lane（宽松版）
        
        小文库专用Lane的特点：
        - 文库数量多（100-200个）
        - 单个文库数据量小（5-20G）
        - 主要需要检查Index冲突和容量
        
        [2025-12-25] 根据人工排机分析，小文库专用Lane可以放宽以下规则：
        - customer_ratio: 小文库可能大部分是客户文库，放宽此限制
        - index_10bp_ratio: 同尺寸小文库可能Index类型单一，放宽此限制
        """
        # 只做基本的Index冲突检查
        if self.config.enable_index_check:
            if not self.index_validator.validate_lane_quick(lane.libraries):
                logger.debug(f"小文库专用Lane {lane.lane_id} Index冲突检查失败")
                return False
        
        # 容量检查
        machine_type_str = lane.machine_type.value if lane.machine_type else "Nova X-25B"
        min_allowed, max_allowed = self._resolve_lane_capacity_limits(
            lane.libraries,
            machine_type_str,
            lane=lane,
        )
        total_data = lane.total_data_gb
        if total_data < min_allowed:
            logger.debug(f"小文库专用Lane {lane.lane_id} 容量不足: {total_data:.1f}GB < {min_allowed:.1f}GB")
            return False
        
        if total_data > max_allowed:
            logger.debug(f"小文库专用Lane {lane.lane_id} 容量超限: {total_data:.1f}GB > {max_allowed:.1f}GB")
            return False
        
        # 小文库专用Lane通过验证（放宽customer_ratio和index_10bp_ratio）
        logger.debug(f"小文库专用Lane {lane.lane_id} 验证通过 (宽松模式)")
        return True
    
    def _schedule_non_10bp_dedicated_lanes(
        self,
        libraries: List[EnhancedLibraryInfo],
        machine_type: str
    ) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo]]:
        """
        [2025-12-25 新增] 非10bp专用Lane策略
        
        根据规则文档（规则4b）：
        - a、Lane中只含10碱基文库时，不做限制
        - b、Lane中只含非10碱基文库时，也不做限制
        - c、Lane中含10碱基和非10碱基时，10碱基Index的文库数据量 > 40%
        
        人工排机分析（2025-07-01 Nova X-25B）：
        - 5条Lane是100%非10bp文库（6/8碱基Index）
        - Lane 328629, 328634, 328667, 328703, 328704
        - 主要是自建库项目（清华、康龙化成等）
        
        策略：
        1. 识别非10bp文库（6碱基或8碱基Index）
        2. 将非10bp文库集中形成专用Lane
        3. 专用Lane不受10bp>=40%规则限制
        
        Args:
            libraries: 文库列表
            machine_type: 机器类型
            
        Returns:
            (专用Lane列表, 剩余未分配的文库)
        """
        # 分离10bp和非10bp文库
        non_10bp_libs = []
        bp_10_libs = []
        
        for lib in libraries:
            is_10bp = self._is_10bp_library(lib)
            if is_10bp:
                bp_10_libs.append(lib)
            else:
                non_10bp_libs.append(lib)
        
        if not non_10bp_libs:
            return [], libraries
        
        # 计算非10bp文库总量
        non_10bp_total = sum(lib.get_data_amount_gb() for lib in non_10bp_libs)
        logger.info(f"非10bp专用Lane分析: {len(non_10bp_libs)}个非10bp文库, 总量{non_10bp_total:.1f}GB, "
                   f"可组{non_10bp_total/self.config.lane_capacity_gb:.1f}条Lane")
        
        # 如果非10bp文库不足以形成一条Lane，不处理
        min_lane_data, _ = self._resolve_lane_capacity_limits(
            non_10bp_libs,
            machine_type,
            metadata={'is_pure_non_10bp_lane': True},
        )
        if non_10bp_total < min_lane_data:
            logger.info(f"非10bp文库总量{non_10bp_total:.1f}GB不足以形成专用Lane（需要>={min_lane_data:.1f}GB）")
            return [], libraries
        
        # 将字符串机器类型转换为枚举
        machine_type_enum = self._resolve_machine_type_enum(machine_type, non_10bp_libs)
        if machine_type_enum == MachineType.UNKNOWN:
            machine_type_enum = MachineType.NOVA_X_25B
        
        # 按数据量排序（大文库优先）
        sorted_non_10bp = self._sort_remaining_for_scattered_mix_lane(non_10bp_libs)
        
        non_10bp_lanes: List[LaneAssignment] = []
        used_libs: Set[str] = set()
        remaining_non_10bp = list(sorted_non_10bp)
        
        while remaining_non_10bp:
            remaining_non_10bp = self._sort_remaining_for_scattered_mix_lane(remaining_non_10bp)
            # 检查剩余文库是否足以形成一条Lane
            remaining_total = sum(lib.get_data_amount_gb() for lib in remaining_non_10bp)
            min_lane_data, _ = self._resolve_lane_capacity_limits(
                remaining_non_10bp,
                machine_type,
                metadata={'is_pure_non_10bp_lane': True},
            )
            if remaining_total < min_lane_data:
                break
            
            # 使用全局计数器获取唯一Lane ID
            lane_id = self._get_next_lane_id("NB", machine_type)
            current_lane = LaneAssignment(
                lane_id=lane_id,  # NB = Non-10Bp
                machine_id=f"M_{lane_id[3:]}",
                machine_type=machine_type_enum,
                lane_capacity_gb=self.config.lane_capacity_gb
            )
            
            # 贪心填充Lane
            next_remaining: List[EnhancedLibraryInfo] = []
            for lib in remaining_non_10bp:
                # 容量检查
                test_libs_for_capacity = current_lane.libraries + [lib]
                _, max_capacity = self._resolve_lane_capacity_limits(
                    test_libs_for_capacity,
                    machine_type,
                    lane=current_lane,
                )
                new_total = current_lane.total_data_gb + lib.get_data_amount_gb()
                
                if new_total > max_capacity:
                    next_remaining.append(lib)
                    continue
                
                # Index冲突检查
                if self.config.enable_index_check:
                    test_libs = current_lane.libraries + [lib]
                    if not self.index_validator.validate_lane_quick(test_libs):
                        next_remaining.append(lib)
                        continue
                
                # [2025-12-25 新增] Peak Size约束检查
                test_libs_for_peak = current_lane.libraries + [lib]
                if not self._check_peak_size_compatible(test_libs_for_peak):
                    next_remaining.append(lib)
                    continue
                
                # [2025-12-31 修复] NB Lane客户占比严格策略：如果已有内部文库，完全禁止添加客户文库
                # 规则：客户占比要么 <= 50%，要么 = 100%
                # 如果当前Lane已有内部文库，则禁止添加客户文库（避免客户占比超过50%）
                if current_lane.libraries:
                    has_internal = any(not l.is_customer_library() for l in current_lane.libraries)
                    if has_internal and lib.is_customer_library():
                        # Lane中已有内部文库，禁止添加客户文库
                        logger.debug(f"NB Lane形成: {current_lane.lane_id} 已有内部文库，禁止添加客户文库 {lib.origrec}")
                        next_remaining.append(lib)
                        continue
                
                # [2025-12-31 修复] 客户占比约束检查（按数据量计算，与验证器一致）
                # 规则：客户占比要么 <= 50%，要么 = 100%
                # 严格策略：如果当前Lane客户占比已接近40%，只允许添加内部文库（NB Lane更严格）
                if not self._check_customer_ratio_near_limit(current_lane, lib, threshold=0.50):
                    next_remaining.append(lib)
                    continue
                
                # 检查添加后的客户占比
                test_libs_for_customer = current_lane.libraries + [lib]
                if not self._check_customer_ratio_compatible_by_data(test_libs_for_customer):
                    # 调试：记录为什么被拒绝
                    test_total = sum(float(getattr(l, 'contract_data_raw', 0) or 0) for l in test_libs_for_customer)
                    test_customer = sum(float(getattr(l, 'contract_data_raw', 0) or 0) for l in test_libs_for_customer if l.is_customer_library())
                    test_ratio = test_customer / test_total if test_total > 0 else 0
                    logger.debug(f"NB Lane {current_lane.lane_id} 拒绝文库 {lib.origrec}: 客户占比{test_ratio:.1%}超过50%且不等于100%")
                    next_remaining.append(lib)
                    continue
                
                # [2025-12-31 新增] 碱基不均衡占比接近限制检查（严格策略）
                if not self._check_base_imbalance_ratio_near_limit(
                    current_lane, lib, threshold=self.config.max_imbalance_ratio
                ):
                    next_remaining.append(lib)
                    continue
                
                # [2025-12-26 新增] 碱基不均衡占比检查（NB Lane仍受此限制）
                test_libs_for_imbalance = current_lane.libraries + [lib]
                if not self._check_base_imbalance_compatible(test_libs_for_imbalance):
                    next_remaining.append(lib)
                    continue
                
                # 通过检查，加入Lane
                current_lane.add_library(lib)
                used_libs.add(self._get_library_runtime_key(lib))
            
            # 检查Lane是否有效
            if not current_lane.libraries:
                break
            
            utilization = current_lane.total_data_gb / self.config.lane_capacity_gb
            min_allowed, _ = self._resolve_lane_capacity_limits(
                current_lane.libraries,
                machine_type,
                lane=current_lane,
            )
            if current_lane.total_data_gb >= min_allowed:
                # [2025-12-31 新增] 最终验证：检查客户占比是否符合规则
                final_total = sum(float(getattr(lib, 'contract_data_raw', 0) or 0) for lib in current_lane.libraries)
                final_customer = sum(float(getattr(lib, 'contract_data_raw', 0) or 0) for lib in current_lane.libraries if lib.is_customer_library())
                final_customer_ratio = final_customer / final_total if final_total > 0 else 0.0
                
                # 如果客户占比不符合规则（>50%且<100%），拒绝这个Lane
                if final_customer_ratio > 0.50 and abs(final_customer_ratio - 1.0) > 1e-6:
                    logger.warning(f"NB Lane {current_lane.lane_id} 客户占比{final_customer_ratio:.1%}不符合规则，拒绝形成")
                    for lib in current_lane.libraries:
                        used_libs.discard(self._get_library_runtime_key(lib))
                    break
                
                # 验证Lane（非10bp专用Lane放宽10bp占比规则）
                if self._validate_non_10bp_lane(current_lane):
                    non_10bp_lanes.append(current_lane)
                    # 统计客户文库数量（用于日志）
                    customer_count = sum(1 for lib in current_lane.libraries if lib.is_customer_library())
                    logger.info(f"非10bp专用Lane {current_lane.lane_id} 形成成功 - "
                               f"文库数: {len(current_lane.libraries)} (客户{customer_count}+内部{len(current_lane.libraries)-customer_count}), "
                               f"数据量: {current_lane.total_data_gb:.1f}GB, "
                               f"客户占比: {final_customer_ratio:.1%}, "
                               f"利用率: {utilization:.1%}")
                    remaining_non_10bp = next_remaining
                else:
                    # 验证失败，回退
                    for lib in current_lane.libraries:
                        used_libs.discard(self._get_library_runtime_key(lib))
                    break
            else:
                # 容量不足，回退
                for lib in current_lane.libraries:
                    used_libs.discard(self._get_library_runtime_key(lib))
                break
        
        # 收集剩余文库（未被使用的非10bp文库 + 10bp文库）
        remaining_libs = [
            lib for lib in libraries
            if self._get_library_runtime_key(lib) not in used_libs
        ]
        
        return non_10bp_lanes, remaining_libs
    
    def _is_10bp_library(self, lib: EnhancedLibraryInfo) -> bool:
        """判断文库是否为10bp Index文库"""
        # 优先使用预处理的ten_bp_data字段
        ten_bp_data = getattr(lib, 'ten_bp_data', None)
        if ten_bp_data is not None and ten_bp_data > 0:
            return True
        
        # 回退到解析index_seq
        index_seq = getattr(lib, 'index_seq', '') or ''
        if not index_seq:
            return False
        
        # 只看P7端长度（分号前的部分）
        if ';' in index_seq:
            p7_part = index_seq.split(';')[0]
        else:
            p7_part = index_seq
        
        # 检查是否有任何一个Index达到10碱基
        for idx in p7_part.split(','):
            idx = idx.strip()
            if idx and len(idx) >= 10:
                return True
        
        return False
    
    def _validate_non_10bp_lane(self, lane: LaneAssignment) -> bool:
        """
        验证非10bp专用Lane（宽松版）
        
        非10bp专用Lane的特点：
        - 全部是非10碱基Index（6/8碱基）
        - 根据规则4b，不受10bp>=40%限制
        - 主要检查Index冲突和容量
        """
        # 只做基本的Index冲突检查
        if self.config.enable_index_check:
            if not self.index_validator.validate_lane_quick(lane.libraries):
                logger.debug(f"非10bp专用Lane {lane.lane_id} Index冲突检查失败")
                return False
        
        # 容量检查
        machine_type_str = lane.machine_type.value if lane.machine_type else "Nova X-25B"
        min_allowed, max_allowed = self._resolve_lane_capacity_limits(
            lane.libraries,
            machine_type_str,
            lane=lane,
        )
        total_data = lane.total_data_gb
        if total_data < min_allowed:
            logger.debug(f"非10bp专用Lane {lane.lane_id} 容量不足: {total_data:.1f}GB < {min_allowed:.1f}GB")
            return False
        
        if total_data > max_allowed:
            logger.debug(f"非10bp专用Lane {lane.lane_id} 容量超限: {total_data:.1f}GB > {max_allowed:.1f}GB")
            return False
        
        # 非10bp专用Lane通过验证（根据规则4b，不检查10bp占比）
        logger.debug(f"非10bp专用Lane {lane.lane_id} 验证通过 (规则4b: 纯非10bp不限制)")
        return True
    
    def _calculate_backbone_reservation(
        self,
        libraries: List[EnhancedLibraryInfo]
    ) -> Tuple[float, int]:
        """
        [2025-12-25 新增] 计算需要预留的大文库骨架数据量
        
        分析思路：
        - 实际排机中，小文库会被部分消化到混排Lane中
        - 但总会有一些"尾巴"小文库无法凑成完整Lane
        - 因此，我们需要保守预留，确保最后有骨架可用
        
        策略：
        1. 小文库总量如果不足下限容量，预留一个骨架（约500G）
        2. 否则，预留小文库总量的10%作为安全冗余（至少500G）
        
        Args:
            libraries: 全部文库列表
            
        Returns:
            (需要预留的大文库数据量GB, 预计可以携带的小文库形成的Lane数)
        """
        threshold = getattr(self.config, 'small_library_threshold_gb', 20.0)
        safety_factor = getattr(self.config, 'backbone_safety_factor', 1.2)
        min_lane_data = self.config.lane_capacity_gb * self.config.min_utilization  # 下限容量
        
        # 统计小文库
        small_libs = [lib for lib in libraries if lib.get_data_amount_gb() <= threshold]
        small_total = sum(lib.get_data_amount_gb() for lib in small_libs)
        
        if not small_libs:
            return 0.0, 0
        
        # 统计大文库
        large_libs = [lib for lib in libraries if lib.get_data_amount_gb() > threshold]
        large_total = sum(lib.get_data_amount_gb() for lib in large_libs)
        
        # 策略：预留足够的骨架，确保能携带足够的小文库
        # 
        # 核心问题：小文库不能自行组成Lane（需要达到下限容量），必须依赖大文库骨架
        # 
        # 从实际测试中发现：
        # - 1401个小文库(7363GB)，排机后约剩余200-500G无法分配
        # - 需要足够的骨架来形成2条Lane，才能消化这些"尾巴"
        # 
        # 计算逻辑：
        # 1. 保守估计，剩余小文库约500-600G
        # 2. 每条骨架Lane配置：500G骨架 + 500G小文库 = 1000G
        # 3. 需要2条骨架Lane = 1000G骨架
        
        # 保守估计：预留足够形成2条骨架Lane（约1000-1200G）
        # 但考虑到大文库总量有限，限制在大文库的50%以内
        backbone_needed = min(1200.0 * safety_factor, large_total * 0.50)
        
        # 至少预留500G（能形成1条Lane）
        if backbone_needed < 500.0 and large_total >= 500.0:
            backbone_needed = 500.0
        
        estimated_lanes_needed = max(1, int(backbone_needed / 500))
        
        logger.info(f"大带小预留分析: {len(small_libs)}个小文库({small_total:.0f}GB), "
                   f"{len(large_libs)}个大文库({large_total:.0f}GB), "
                   f"预留骨架{backbone_needed:.0f}GB（可带{estimated_lanes_needed}条Lane）")
        
        return backbone_needed, estimated_lanes_needed
    
    def _select_backbone_libraries(
        self,
        libraries: List[EnhancedLibraryInfo],
        target_data_gb: float
    ) -> Tuple[List[EnhancedLibraryInfo], List[EnhancedLibraryInfo]]:
        """
        [2025-12-25 新增] 从大文库中挑选骨架预留
        
        挑选策略：
        1. 从大文库（>20G）中挑选
        2. 优先选择中等大小的文库（50-100G），因为太大的文库可能是核心项目
        3. 选择足够的文库，直到总量>=target_data_gb
        
        Args:
            libraries: 全部文库列表
            target_data_gb: 需要预留的数据量
            
        Returns:
            (预留的骨架文库列表, 剩余参与普通排机的文库列表)
        """
        if target_data_gb <= 0:
            return [], libraries
        
        threshold = getattr(self.config, 'small_library_threshold_gb', 20.0)
        
        # 分离大文库和小文库
        large_libs = [lib for lib in libraries if lib.get_data_amount_gb() > threshold]
        small_libs = [lib for lib in libraries if lib.get_data_amount_gb() <= threshold]
        
        if not large_libs:
            return [], libraries
        
        # 优先选择中等大小的文库（30-100G），按数据量排序
        # 中等文库更适合作为骨架：不会太大影响核心项目，也不会太小凑不够
        mid_sized = [lib for lib in large_libs if 30 <= lib.get_data_amount_gb() <= 100]
        large_sized = [lib for lib in large_libs if lib.get_data_amount_gb() > 100]
        small_large = [lib for lib in large_libs if 20 < lib.get_data_amount_gb() < 30]

        # 按优先级排序：中等 > 小大 > 大
        # 中等文库内部按数据量升序（用小的凑数，保留大的给核心项目）
        candidates = sorted(mid_sized, key=lambda lib: lib.get_data_amount_gb())
        candidates.extend(sorted(small_large, key=lambda lib: lib.get_data_amount_gb()))
        candidates.extend(sorted(large_sized, key=lambda lib: lib.get_data_amount_gb()))
        
        # 贪心选择骨架文库
        backbone: List[EnhancedLibraryInfo] = []
        backbone_total = 0.0
        backbone_ids: Set[str] = set()
        
        for lib in candidates:
            if backbone_total >= target_data_gb:
                break
            backbone.append(lib)
            backbone_total += lib.get_data_amount_gb()
            backbone_ids.add(lib.origrec)
        
        # 剩余的大文库和小文库参与普通排机
        remaining_large = [lib for lib in large_libs if lib.origrec not in backbone_ids]
        remaining = remaining_large + small_libs
        
        logger.info(f"骨架预留: 选择{len(backbone)}个大文库作为骨架, 总量{backbone_total:.1f}GB "
                   f"(目标{target_data_gb:.1f}GB), 剩余{len(remaining)}个文库参与普通排机")
        
        return backbone, remaining
    
    def _schedule_backbone_with_small_lanes(
        self,
        backbone_libs: List[EnhancedLibraryInfo],
        small_libs: List[EnhancedLibraryInfo],
        machine_type: str
    ) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo]]:
        """
        [2025-12-25 新增] 用骨架大文库+小文库组成Lane
        
        策略（改进版）：
        1. 可以使用多个骨架大文库来组成一条Lane的基础
        2. 用小文库填充剩余空间，直到Lane满
        3. 验证Lane，如果通过则保留
        
        关键改进：支持"多骨架+小文库"模式，例如：
        - 骨架1(200G) + 骨架2(300G) + 小文库(200G) = 700G Lane
        
        Args:
            backbone_libs: 预留的骨架大文库
            small_libs: 待分配的小文库
            machine_type: 机器类型
            
        Returns:
            (形成的Lane列表, 剩余未分配的小文库)
        """
        if not backbone_libs or not small_libs:
            return [], small_libs
        
        # 将字符串机器类型转换为枚举
        machine_type_enum = self._resolve_machine_type_enum(machine_type, backbone_libs)
        if machine_type_enum == MachineType.UNKNOWN:
            machine_type_enum = MachineType.NOVA_X_25B
        
        # 按数据量排序骨架（大的优先）
        sorted_backbone = self._sort_remaining_for_scattered_mix_lane(backbone_libs)
        # 按数据量排序小文库（大的优先，更快填满）
        sorted_small = self._sort_remaining_for_scattered_mix_lane(small_libs)
        
        backbone_lanes: List[LaneAssignment] = []
        remaining_backbone: List[EnhancedLibraryInfo] = list(sorted_backbone)
        remaining_small: List[EnhancedLibraryInfo] = list(sorted_small)
        used_backbone_ids: Set[str] = set()
        used_small_ids: Set[str] = set()
        
        # 循环尝试形成骨架Lane
        while remaining_backbone and remaining_small:
            remaining_backbone = [
                lib for lib in remaining_backbone
                if lib.origrec not in used_backbone_ids
            ]
            remaining_small = [
                lib for lib in remaining_small
                if lib.origrec not in used_small_ids
            ]
            if not remaining_backbone or not remaining_small:
                break

            remaining_backbone = self._sort_remaining_for_scattered_mix_lane(remaining_backbone)

            # 使用全局计数器获取唯一Lane ID
            lane_id = self._get_next_lane_id("BL", machine_type)
            current_lane = LaneAssignment(
                lane_id=lane_id,  # BL = Backbone Lane
                machine_id=f"M_{lane_id[3:]}",
                machine_type=machine_type_enum,
                lane_capacity_gb=self.config.lane_capacity_gb
            )
            
            # 策略（优化版）：
            # 1. 先放入骨架，但限制骨架总量在500-700G（为小文库留空间）
            # 2. 然后用小文库填充剩余空间
            # 这样可以确保每条Lane都能携带足够的小文库
            
            backbone_in_lane: List[EnhancedLibraryInfo] = []
            next_remaining_backbone: List[EnhancedLibraryInfo] = []
            
            # 目标：骨架占用Lane的60-70%，留30-40%给小文库
            # 对于975G的Lane，骨架约600-700G，小文库约300-400G
            target_backbone_data = self.config.lane_capacity_gb * 0.65  # 约630G
            current_backbone_data = 0.0
            
            for lib in remaining_backbone:
                if lib.origrec in used_backbone_ids:
                    continue
                
                # 骨架量达到目标后，停止添加骨架（为小文库留空间）
                if current_backbone_data >= target_backbone_data:
                    next_remaining_backbone.append(lib)
                    continue
                
                # 容量检查
                test_libs_for_capacity = current_lane.libraries + [lib]
                _, max_capacity = self._resolve_lane_capacity_limits(
                    test_libs_for_capacity,
                    machine_type,
                    lane=current_lane,
                )
                new_total = current_lane.total_data_gb + lib.get_data_amount_gb()
                
                if new_total > max_capacity:
                    next_remaining_backbone.append(lib)
                    continue
                
                # Index冲突检查
                if self.config.enable_index_check:
                    test_libs = current_lane.libraries + [lib]
                    if not self.index_validator.validate_lane_quick(test_libs):
                        next_remaining_backbone.append(lib)
                        continue
                
                # [2025-12-25 新增] Peak Size约束检查
                test_libs_for_peak = current_lane.libraries + [lib]
                if not self._check_peak_size_compatible(test_libs_for_peak):
                    next_remaining_backbone.append(lib)
                    continue
                
                # [2025-12-31 新增] 客户占比接近限制检查（严格策略）
                if not self._check_customer_ratio_near_limit(current_lane, lib, threshold=0.50):
                    next_remaining_backbone.append(lib)
                    continue
                
                # [2025-12-31 新增] 客户占比检查（按数据量计算）
                test_libs_for_customer = current_lane.libraries + [lib]
                if not self._check_customer_ratio_compatible_by_data(test_libs_for_customer):
                    next_remaining_backbone.append(lib)
                    continue
                
                # [2025-12-31 新增] 碱基不均衡占比接近限制检查（严格策略）
                if not self._check_base_imbalance_ratio_near_limit(
                    current_lane, lib, threshold=self.config.max_imbalance_ratio
                ):
                    next_remaining_backbone.append(lib)
                    continue
                
                # [2025-12-31 新增] 碱基不均衡占比检查（BL Lane不是专用Lane，需要检查）
                test_libs_for_imbalance = current_lane.libraries + [lib]
                if not self._check_base_imbalance_compatible(test_libs_for_imbalance):
                    next_remaining_backbone.append(lib)
                    continue
                
                # [2025-12-31 新增] 10bp Index占比接近限制检查（严格策略）
                if not self._check_10bp_index_ratio_near_limit(
                    current_lane, lib, threshold=self.config.min_10bp_index_ratio
                ):
                    next_remaining_backbone.append(lib)
                    continue
                
                # [2025-12-31 新增] 10bp Index占比检查（BL Lane需要检查）
                test_libs_for_10bp = current_lane.libraries + [lib]
                if not self._check_10bp_index_ratio_compatible(test_libs_for_10bp):
                    next_remaining_backbone.append(lib)
                    continue
                
                # 通过检查，加入Lane
                current_lane.add_library(lib)
                backbone_in_lane.append(lib)
                used_backbone_ids.add(lib.origrec)
                current_backbone_data += lib.get_data_amount_gb()
            
            # 如果没有任何骨架能放入，退出循环
            if not backbone_in_lane:
                break
            
            # 用小文库填充剩余空间
            next_remaining_small: List[EnhancedLibraryInfo] = []
            small_in_lane: List[EnhancedLibraryInfo] = []

            lane_seed = self._sort_remaining_for_scattered_mix_lane(current_lane.libraries)[0]
            candidate_small_pool = self._sort_remaining_for_lane_seed(remaining_small, lane_seed)

            for lib in candidate_small_pool:
                if lib.origrec in used_small_ids:
                    continue
                
                # 容量检查
                test_libs_for_capacity = current_lane.libraries + [lib]
                _, max_capacity = self._resolve_lane_capacity_limits(
                    test_libs_for_capacity,
                    machine_type,
                    lane=current_lane,
                )
                new_total = current_lane.total_data_gb + lib.get_data_amount_gb()
                
                if new_total > max_capacity:
                    next_remaining_small.append(lib)
                    continue
                
                # Index冲突检查
                if self.config.enable_index_check:
                    test_libs = current_lane.libraries + [lib]
                    if not self.index_validator.validate_lane_quick(test_libs):
                        next_remaining_small.append(lib)
                        continue
                
                # [2025-12-25 新增] Peak Size约束检查
                test_libs_for_peak = current_lane.libraries + [lib]
                if not self._check_peak_size_compatible(test_libs_for_peak):
                    next_remaining_small.append(lib)
                    continue
                
                # [2025-12-31 新增] 客户占比接近限制检查（严格策略）
                if not self._check_customer_ratio_near_limit(current_lane, lib, threshold=0.50):
                    next_remaining_small.append(lib)
                    continue
                
                # [2025-12-31 新增] 客户占比检查（按数据量计算）
                test_libs_for_customer = current_lane.libraries + [lib]
                if not self._check_customer_ratio_compatible_by_data(test_libs_for_customer):
                    next_remaining_small.append(lib)
                    continue
                
                # [2025-12-31 新增] 碱基不均衡占比接近限制检查（严格策略）
                if not self._check_base_imbalance_ratio_near_limit(
                    current_lane, lib, threshold=self.config.max_imbalance_ratio
                ):
                    next_remaining_small.append(lib)
                    continue
                
                # [2025-12-31 新增] 碱基不均衡占比检查（BL Lane不是专用Lane，需要检查）
                test_libs_for_imbalance = current_lane.libraries + [lib]
                if not self._check_base_imbalance_compatible(test_libs_for_imbalance):
                    next_remaining_small.append(lib)
                    continue
                
                # [2025-12-31 新增] 10bp Index占比接近限制检查（严格策略）
                if not self._check_10bp_index_ratio_near_limit(
                    current_lane, lib, threshold=self.config.min_10bp_index_ratio
                ):
                    next_remaining_small.append(lib)
                    continue
                
                # [2025-12-31 新增] 10bp Index占比检查（BL Lane需要检查）
                test_libs_for_10bp = current_lane.libraries + [lib]
                if not self._check_10bp_index_ratio_compatible(test_libs_for_10bp):
                    next_remaining_small.append(lib)
                    continue
                
                # 通过检查，加入Lane
                current_lane.add_library(lib)
                small_in_lane.append(lib)
                used_small_ids.add(lib.origrec)
                
                # 检查是否已满
                _, max_allowed = self._resolve_lane_capacity_limits(
                    current_lane.libraries,
                    machine_type,
                    lane=current_lane,
                )
                if current_lane.total_data_gb >= max_allowed - 1e-6:
                    break
            
            # 检查Lane是否有效
            utilization = current_lane.total_data_gb / self.config.lane_capacity_gb
            min_allowed, _ = self._resolve_lane_capacity_limits(
                current_lane.libraries,
                machine_type,
                lane=current_lane,
            )
            if current_lane.total_data_gb >= min_allowed:
                # 验证Lane
                if self._validate_backbone_lane(current_lane):
                    backbone_lanes.append(current_lane)
                    backbone_total = sum(lib.get_data_amount_gb() for lib in backbone_in_lane)
                    small_total = sum(lib.get_data_amount_gb() for lib in small_in_lane)
                    logger.info(f"骨架Lane {current_lane.lane_id} 形成成功 - "
                               f"骨架: {len(backbone_in_lane)}个({backbone_total:.0f}G), "
                               f"小文库: {len(small_in_lane)}个({small_total:.0f}G), "
                               f"总量: {current_lane.total_data_gb:.1f}GB, "
                               f"利用率: {utilization:.1%}")
                    remaining_backbone = next_remaining_backbone
                    remaining_small = [lib for lib in remaining_small if lib.origrec not in used_small_ids]
                else:
                    # 验证失败，回退所有文库
                    for lib in backbone_in_lane:
                        used_backbone_ids.discard(lib.origrec)
                    for lib in small_in_lane:
                        used_small_ids.discard(lib.origrec)
                    break
            else:
                # 容量不足，回退所有文库
                for lib in backbone_in_lane:
                    used_backbone_ids.discard(lib.origrec)
                for lib in small_in_lane:
                    used_small_ids.discard(lib.origrec)
                break
        
        # 返回剩余未分配的小文库
        final_remaining = [lib for lib in small_libs if lib.origrec not in used_small_ids]
        
        return backbone_lanes, final_remaining
    
    def _validate_backbone_lane(self, lane: LaneAssignment) -> bool:
        """
        验证骨架Lane（适度宽松版）
        
        骨架Lane的特点：
        - 1个大文库作为骨架 + 多个小文库填充
        - 主要检查Index冲突和容量
        - 客户占比由最终验证阶段检查（允许后期调整）
        """
        # Index冲突检查
        if self.config.enable_index_check:
            if not self.index_validator.validate_lane_quick(lane.libraries):
                logger.debug(f"骨架Lane {lane.lane_id} Index冲突检查失败")
                return False
        
        # 容量检查
        machine_type_str = lane.machine_type.value if lane.machine_type else "Nova X-25B"
        min_allowed, max_allowed = self._resolve_lane_capacity_limits(
            lane.libraries,
            machine_type_str,
            lane=lane,
        )
        total_data = lane.total_data_gb
        if total_data < min_allowed:
            logger.debug(f"骨架Lane {lane.lane_id} 容量不足: {total_data:.1f}GB < {min_allowed:.1f}GB")
            return False
        
        if total_data > max_allowed:
            logger.debug(f"骨架Lane {lane.lane_id} 容量超限: {total_data:.1f}GB > {max_allowed:.1f}GB")
            return False
        
        # 骨架Lane通过验证
        logger.debug(f"骨架Lane {lane.lane_id} 验证通过")
        return True
    
    def _group_by_machine_type(self, libraries: List[EnhancedLibraryInfo]) -> Dict[str, List[EnhancedLibraryInfo]]:
        """按机器类型分组"""
        groups: Dict[str, List[EnhancedLibraryInfo]] = {}
        
        for lib in libraries:
            machine_type = lib.eq_type or "Unknown"
            if machine_type not in groups:
                groups[machine_type] = []
            groups[machine_type].append(lib)
        
        return groups
    
    def _schedule_machine_group(
        self, 
        libraries: List[EnhancedLibraryInfo],
        machine_type: str
    ) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo]]:
        """
        对单个机器类型组进行逐Lane排机
        
        核心逻辑（简单直接）：
        1. 创建第1条Lane，遍历所有文库，能放就放
        2. Lane满了（>=90%） → 保存，用剩余文库排第2条
        3. Lane填不满（<90%） → 剩余文库无法形成有效Lane，结束
        
        注意：一条Lane排完验证通过后，再排下一条。不同时排多条。
        """
        valid_lanes: List[LaneAssignment] = []
        failed_libraries: List[EnhancedLibraryInfo] = []  # 验证失败的文库
        remaining: List[EnhancedLibraryInfo] = list(libraries)
        
        # 将字符串机器类型转换为枚举
        machine_type_enum = self._resolve_machine_type_enum(machine_type, libraries)
        if machine_type_enum == MachineType.UNKNOWN:
            machine_type_enum = MachineType.NOVA_X_25B
        
        # 逐条Lane排机
        while remaining:
            # 散样混排策略优先于一般排序：临检 > YC > delete_date > 其他，尽量集中到连续Lane。
            remaining = self._sort_remaining_for_scattered_mix_lane(remaining)
            lane_candidate_order = self._sort_remaining_for_lane_seed(remaining, remaining[0])

            seed_rule = self._get_scheduling_lane_capacity_range(
                libraries=[remaining[0]],
                machine_type=machine_type_enum.value,
            )
            lane_candidate_order = self._sort_by_soft_single_lane_preference(
                lane_candidate_order,
                remaining[0],
                seed_rule,
            )
            # 为当前Lane抽取一个软目标容量（在下限和上限之间随机），硬上限仍由验证控制
            target_capacity_gb = self._sample_target_capacity(seed_rule)

            # 使用全局计数器获取唯一Lane ID
            lane_id = self._get_next_lane_id("GL", machine_type)
            current_lane = LaneAssignment(
                lane_id=lane_id,
                machine_id=f"M_{lane_id[3:]}",  # 去掉"GL_"前缀
                machine_type=machine_type_enum,
                lane_capacity_gb=seed_rule.soft_target_gb or self.config.lane_capacity_gb
            )
            current_lane.metadata["rule_code"] = seed_rule.rule_code
            current_lane.metadata["seq_mode"] = seed_rule.sequencing_mode
            current_lane.metadata["sequencing_mode"] = seed_rule.sequencing_mode
            current_lane.metadata["loading_method"] = seed_rule.loading_method
            current_lane.metadata["target_capacity_gb"] = seed_rule.soft_target_gb
            
            # 遍历剩余文库，尝试放入当前Lane
            next_remaining: List[EnhancedLibraryInfo] = []
            deferred_remaining: List[EnhancedLibraryInfo] = []
            placed_ids: Set[str] = set()
            break_index: Optional[int] = None
            for lib in lane_candidate_order:
                # 检查是否能放入
                if self._can_add_to_lane(current_lane, lib):
                    current_lane.add_library(lib)
                    placed_ids.add(str(getattr(lib, "origrec", "") or id(lib)))
                    
                    # 检查是否达到软目标容量（非硬上限）
                    if current_lane.total_data_gb >= target_capacity_gb:
                        # 达到软目标，剩余文库留给下一条Lane
                        break_index = lane_candidate_order.index(lib)
                        break
                else:
                    # 放不进去，留给下一条Lane
                    deferred_remaining.append(lib)

            if break_index is not None:
                trailing_candidates = lane_candidate_order[break_index + 1 :]
                deferred_remaining.extend(trailing_candidates)

            if lane_candidate_order:
                seen_ids = {id(lib) for lib in deferred_remaining}
                for lib in remaining:
                    lib_key = str(getattr(lib, "origrec", "") or id(lib))
                    if lib_key in placed_ids:
                        continue
                    if id(lib) in seen_ids:
                        continue
                    deferred_remaining.append(lib)
                    seen_ids.add(id(lib))

            next_remaining = deferred_remaining
            
            # 当前Lane排完，检查是否有效
            if not current_lane.libraries:
                # 没有任何文库能放进去，结束
                logger.warning(f"剩余 {len(remaining)} 个文库无法分配（全部有约束冲突）")
                break
            
            completed_rule = self._get_scheduling_lane_capacity_range(
                libraries=current_lane.libraries,
                machine_type=machine_type_enum.value,
                metadata=self._build_lane_validation_metadata(current_lane),
            )
            if current_lane.total_data_gb >= completed_rule.effective_min_gb:
                # 容量达标，进行完整红线规则验证
                is_valid, error_types = self._validate_completed_lane(current_lane)
                if is_valid:
                    # 验证通过，保存并继续排下一条
                    valid_lanes.append(current_lane)
                    logger.info(f"Lane {current_lane.lane_id} 通过验证 - "
                               f"文库数: {len(current_lane.libraries)}, "
                               f"数据量: {current_lane.total_data_gb:.1f}GB, "
                               f"规则: {completed_rule.rule_code}")
                    remaining = next_remaining
                else:
                    # 验证失败，当前Lane的文库标记为失败，用剩余文库继续排
                    error_detail = f" ({', '.join(error_types)})" if error_types else ""
                    logger.warning(
                        f"Lane {current_lane.lane_id} 红线验证失败{error_detail}，"
                        f"{len(current_lane.libraries)} 个文库标记为无法分配"
                    )
                    failed_libraries.extend(current_lane.libraries)  # 标记为失败
                    remaining = next_remaining  # 用剩余文库继续排
            else:
                # Lane利用率不足，无法形成有效Lane，结束
                # 当前Lane的文库 + 剩余文库 = 所有未分配文库
                all_unassigned = current_lane.libraries + next_remaining
                logger.warning(
                    f"Lane {current_lane.lane_id} 数据量{current_lane.total_data_gb:.1f}G"
                    f"低于下限{completed_rule.effective_min_gb:.1f}G，"
                    f"{len(all_unassigned)} 个文库无法形成有效Lane"
                )
                remaining = all_unassigned
                break
        
        # 合并剩余未分配 + 验证失败的文库
        all_unassigned = remaining + failed_libraries
        return valid_lanes, all_unassigned

    def _sample_target_capacity(self, lane_rule_selection: Any) -> float:
        """
        在硬约束范围内随机抽取本条Lane的软目标容量。
        - 下限：规则配置的目标下限
        - 上限：规则配置的目标上限
        """
        min_total = float(
            getattr(lane_rule_selection, "min_target_gb", 0.0)
            or getattr(lane_rule_selection, "effective_min_gb", 0.0)
            or self.config.lane_capacity_gb * self.config.min_utilization
        )
        max_total = float(
            getattr(lane_rule_selection, "max_target_gb", 0.0)
            or getattr(lane_rule_selection, "effective_max_gb", 0.0)
            or self.config.lane_capacity_gb * self.config.max_utilization
        )
        if max_total <= min_total:
            return min_total
        return random.uniform(min_total, max_total)
    
    def _can_add_to_lane(self, lane: LaneAssignment, lib: EnhancedLibraryInfo) -> bool:
        """
        检查文库是否可以加入Lane（完整验证规则检查）
        
        [2025-12-31 优化] 增强实时验证，确保严格遵守验证规则：
        检查项目：
        1. 容量上限检查
        2. 机器类型兼容性检查
        3. Index冲突检查
        4. 客户占比检查（按数据量，<=50%或=100%）
        5. 10bp Index占比检查（非NB Lane需要>=40%）
        6. 单端Index占比检查（按数据量，<30%）
        7. 碱基不均衡占比检查（非DL Lane需要<=40%）
        8. Peak Size兼容性检查
        9. 特殊文库类型数量限制（非DL/NB/BL Lane）
        10. 加测文库占比检查（严格模式下视为硬约束）
        11. 碱基不均衡混排规则检查（使用BaseImbalanceHandler）
        12. 文库对兼容性检查（使用RuleChecker）
        """
        test_libraries = lane.libraries + [lib]
        lib_data = lib.get_data_amount_gb()

        # 拆分家族互斥：同一原始文库拆出的子文库不可进入同一Lane
        new_split_family_id = self._get_split_family_id(lib)
        if new_split_family_id:
            for existing_lib in lane.libraries:
                if self._get_split_family_id(existing_lib) == new_split_family_id:
                    return False
        
        # 判断Lane类型
        is_nb_lane = lane.lane_id.startswith('NB_')
        is_dl_lane = lane.lane_id.startswith('DL_')
        is_bl_lane = lane.lane_id.startswith('BL_')
        is_sl_lane = lane.lane_id.startswith('SL_')
        machine_type_str = lane.machine_type.value if lane.machine_type else (lib.eq_type or "Nova X-25B")
        lane_metadata = self._build_lane_validation_metadata(lane)
        
        # 1. 容量上限检查
        new_total = lane.total_data_gb + lib_data
        lane_rule = self._get_scheduling_lane_capacity_range(
            libraries=test_libraries,
            machine_type=machine_type_str,
            metadata=lane_metadata,
        )
        max_capacity = lane_rule.effective_max_gb
        if new_total > max_capacity:
            return False

        if self.scheduling_config.validate_lane_constraints(
            libraries=test_libraries,
            machine_type=machine_type_str,
            metadata=lane_metadata,
        ):
            return False
        
        # 2. 机器类型兼容性检查
        if lane.libraries:
            existing_type_str = lane.machine_type.value if lane.machine_type else ""
            new_type_str = lib.eq_type or ""
            if existing_type_str and new_type_str and existing_type_str != new_type_str:
                return False
        
        # 3. Index冲突检查（硬性约束）
        # 使用增量检查：只验证新文库与已有文库的冲突（O(n)），不重复检查现有文库对（O(n²)）
        if self.config.enable_index_check:
            if not self.index_validator.validate_new_lib_quick(lane.libraries, lib):
                return False
        
        # 4. 客户占比检查（按数据量计算，严格遵守规则）
        if not self._check_customer_ratio_compatible_by_data(test_libraries):
            return False
        
        # 5. 10bp Index占比检查（非NB Lane需要>=40%）
        if not is_nb_lane:
            if not self._check_10bp_index_ratio_compatible(test_libraries):
                return False
        
        # 6. 单端Index占比检查（按数据量，<30%）
        if not self._check_single_end_ratio_compatible(test_libraries):
            return False

        # 7. 碱基不均衡占比检查（非DL Lane需要<=40%，按数据量计算）
        if not is_dl_lane:
            if not self._check_base_imbalance_compatible(test_libraries):
                return False
        
        # 8. Peak Size兼容性检查
        if not self._check_peak_size_compatible(test_libraries):
            return False

        # 9. 特殊文库类型数量限制已取消
        if not (is_dl_lane or is_nb_lane or is_bl_lane):
            if not self._check_special_library_type_compatible(test_libraries):
                return False

        # 10. 加测文库占比检查（严格模式下视为硬约束）
        if not self._check_add_test_ratio_compatible(test_libraries):
            return False
        
        # 11. 碱基不均衡混排规则检查
        if self.config.enable_imbalance_check and self.imbalance_handler:
            if not self._check_imbalance_compatibility(lane, lib):
                return False
        
        # 12. 文库对兼容性检查（使用RuleChecker检查新文库与现有文库的兼容性）
        if self.config.enable_rule_checker and self.rule_checker:
            if not self._check_pairwise_compatibility(lane, lib):
                return False
        
        return True
    
    def _check_imbalance_compatibility(self, lane: LaneAssignment, lib: EnhancedLibraryInfo) -> bool:
        """
        检查碱基不均衡混排兼容性
        
        [2025-12-25 待讨论] 根据人工排机数据分析，简化了原有的严格限制：
        1. 原规则限制碱基不均衡总量不超过240G（约25%占比）
        2. 人工排机实际允许到30%（最高29.6%），且有5条100%碱基不均衡专用Lane
        3. 当前策略：
           - 支持100%碱基不均衡专用Lane（enable_dedicated_imbalance_lane=True时）
           - 混排Lane检查总量和占比，不检查分组间混排限制
        
        规则：
        1. 专用Lane：允许100%碱基不均衡（不受占比限制，仅受总量限制）
        2. 混排Lane：碱基不均衡文库占比不超过 max_imbalance_ratio（默认40%）
        
        Args:
            lane: 当前Lane
            lib: 待加入的文库
            
        Returns:
            True表示可以混排，False表示不兼容
        """
        if not self.imbalance_handler:
            return True

        # 收集Lane内所有文库（包括待添加的）
        test_libraries = lane.libraries + [lib]
        is_compatible, reason = self.imbalance_handler.check_mix_compatibility(test_libraries)
        if not is_compatible:
            logger.debug(f"碱基不均衡混排不兼容: {reason}")
        return is_compatible
    
    def _check_pairwise_compatibility(self, lane: LaneAssignment, lib: EnhancedLibraryInfo) -> bool:
        """
        检查新文库与Lane内现有文库的两两兼容性
        
        使用RuleChecker的check_all_rules方法检查文库对级别规则（规则0-14）
        
        Args:
            lane: 当前Lane
            lib: 待加入的文库
            
        Returns:
            True表示兼容，False表示存在规则冲突
        """
        if not self.rule_checker or not lane.libraries:
            return True
        
        # 将EnhancedLibraryInfo转换为RuleChecker所需的字典格式
        new_lib_dict = self._convert_lib_to_dict(lib)
        
        # 检查新文库与每个现有文库的兼容性
        for existing_lib in lane.libraries:
            existing_lib_dict = self._convert_lib_to_dict(existing_lib)
            
            # 检查15条规则
            violations = self.rule_checker.check_all_rules(existing_lib_dict, new_lib_dict)
            
            # 检查是否有硬约束违反（规则0,1,2,10是硬约束）
            hard_constraint_indices = [0, 1, 2, 10]  # 机器类型、工序编码、Index冲突、测序策略
            has_hard_violation = False
            for idx in hard_constraint_indices:
                if violations[idx] == 1:
                    logger.debug(f"规则{idx}冲突: {lib.origrec} vs {existing_lib.origrec}")
                    has_hard_violation = True
                    break
            if has_hard_violation:
                return False
        
        return True
    
    def _convert_lib_to_dict(self, lib: EnhancedLibraryInfo) -> Dict:
        """
        将EnhancedLibraryInfo转换为RuleChecker所需的字典格式
        
        映射关系：
        - 机器类型 <- eq_type
        - 工序编码 <- process_code
        - 样本类型 <- data_type / lab_type
        - Index序列 <- index_seq
        - PeakSize <- peak_size
        - 测序策略 <- seq_strategy
        - 合同数据量_文库 <- contract_data_raw
        - 是否双端index测序 <- 从index_seq推断
        - 是否是客户文库 <- 从sample_id推断
        """
        # 判断是否为客户文库
        sample_id = getattr(lib, 'sample_id', '') or ''
        is_customer = '是' if sample_id.startswith('FKDL') or '客户' in (getattr(lib, 'lab_type', '') or '') else '否'
        
        # 判断是否为双端Index
        index_seq = getattr(lib, 'index_seq', '') or ''
        is_dual_index = '是' if ';' in index_seq else '否'
        
        return {
            '机器类型': getattr(lib, 'eq_type', '') or '',
            '工序编码': getattr(lib, 'process_code', None),
            '样本类型': getattr(lib, 'sample_type_code', '') or getattr(lib, 'data_type', '') or getattr(lib, 'lab_type', '') or '',
            'Index序列': index_seq,
            'PeakSize': getattr(lib, 'peak_size', None),
            '测序策略': getattr(lib, 'seq_strategy', '') or '',
            '合同数据量_文库': float(getattr(lib, 'contract_data_raw', 0) or 0),
            '是否双端index测序': is_dual_index,
            '是否是客户文库': is_customer,
            '数据类型': getattr(lib, 'data_priority', '') or '',
            '库检综合结果': getattr(lib, 'qc_result', '') or '',
            '产线标识': getattr(lib, 'production_line', '') or '',
            'index碱基数目': len(index_seq.split(';')[0]) if index_seq and ';' in index_seq else len(index_seq),
        }
    
    def _validate_completed_lane(self, lane: LaneAssignment) -> tuple[bool, list[str]]:
        """
        对已填满的Lane进行完整红线规则验证
        
        验证流程：
        1. 使用LaneValidator进行基础红线规则校验
        2. 使用RuleChecker进行Lane级别完整规则检查
        3. 使用BaseImbalanceHandler进行最终碱基不均衡验证
        
        只有所有验证都通过，Lane才被认为是有效的。
        """
        machine_type_str = lane.machine_type.value if lane.machine_type else "Nova X-25B"
        metadata = self._build_lane_validation_metadata(lane)
        
        # 1. 基础红线规则验证（LaneValidator）
        result = self.lane_validator.validate_lane(
            libraries=lane.libraries,
            lane_id=lane.lane_id,
            machine_type=machine_type_str,
            metadata=metadata
        )
        
        if not result.is_valid:
            error_types = [e.rule_type.value for e in result.errors]
            logger.debug(f"Lane {lane.lane_id} LaneValidator验证失败: {error_types}")
            return False, error_types
        
        # 2. 使用RuleChecker进行Lane级别规则检查
        if self.config.enable_rule_checker and self.rule_checker:
            rule_check_result = self._validate_lane_with_rule_checker(lane, machine_type_str)
            if not rule_check_result:
                return False, ["rule_checker"]
        
        # 3. 使用BaseImbalanceHandler进行碱基不均衡最终验证
        if self.config.enable_imbalance_check and self.imbalance_handler:
            is_compatible, reason = self.imbalance_handler.check_mix_compatibility(lane.libraries)
            if not is_compatible:
                logger.debug(f"Lane {lane.lane_id} 碱基不均衡验证失败: {reason}")
                return False, ["imbalance_check"]
        
        return True, []
    
    def _validate_lane_with_rule_checker(self, lane: LaneAssignment, machine_type: str) -> bool:
        """
        使用RuleChecker进行Lane级别的完整规则检查
        
        检查内容包括：
        - 规则4: 10碱基Index占比
        - 规则5: 单端Index占比
        - 规则6: 客户/诺禾占比
        - 规则15: Lane容量上限
        - 规则17: 碱基不均衡占比
        - 规则23: 碱基不均衡文库类型数量
        - 规则24: 分组27/28混排占比
        
        Args:
            lane: 待验证的Lane
            machine_type: 机器类型
            
        Returns:
            True表示验证通过，False表示存在规则违反
        """
        if not self.rule_checker:
            return True
        
        # 将Lane内所有文库转换为字典格式
        lane_libraries_dict = [self._convert_lib_to_dict(lib) for lib in lane.libraries]
        
        # 获取工序编码（从第一个文库获取）
        process_code = 1595  # 默认Nova X-25B工序
        if lane.libraries:
            first_lib = lane.libraries[0]
            process_code = getattr(first_lib, 'process_code', 1595) or 1595
        
        # 执行Lane级别规则检查
        results = self.rule_checker.check_all_lane_rules(
            lane_libraries=lane_libraries_dict,
            machine_type=machine_type,
            process_code=process_code,
            load_method='25B' if '25B' in machine_type else '10B',
            priority='其他'
        )
        
        if not results['is_valid']:
            # 收集违反的规则
            violated_rules = []
            rule_mapping = {
                'rule4_10base_ratio': '规则4(10碱基占比)',
                'rule5_single_index_ratio': '规则5(单端占比)',
                'rule6_customer_ratio': '规则6(客户占比)',
                'rule15_capacity': '规则15(容量上限)',
                'rule17_imbalance_ratio': '规则17(碱基不均衡占比)',
                'rule23_type_count': '规则23(文库类型数)',
                'rule24_group29_ratio': '规则24(G27/G28占比)',
            }
            
            for rule_key, rule_name in rule_mapping.items():
                if results.get(rule_key, 0) == 1:
                    violated_rules.append(rule_name)
            
            logger.debug(f"Lane {lane.lane_id} RuleChecker验证失败: {violated_rules}")
            return False
        
        return True
    
    def _optimize_by_redistribution(
        self,
        unassigned: List[EnhancedLibraryInfo],
        all_lanes: List[LaneAssignment],
        machine_type: str
    ) -> Tuple[List[EnhancedLibraryInfo], List[LaneAssignment]]:
        """
        [2025-12-25 新增] 挪移优化策略
        
        当有未分配文库但数据量不足以新开Lane时，从现有Lane挪出部分文库，
        与未分配文库合并成新Lane，实现100%分配率。
        
        策略：
        1. 计算未分配文库总量，判断需要从现有Lane挪出多少数据
        2. 优先从NB Lane挪出客户文库（保持新Lane为100%客户）
        3. 确保被挪出的Lane剩余数据量>=下限容量
        4. 验证新Lane符合所有规则
        
        Args:
            unassigned: 未分配文库列表
            all_lanes: 现有Lane列表
            machine_type: 机器类型
            
        Returns:
            (仍未分配的文库列表, 新创建的Lane列表)
        """
        if not unassigned:
            return [], []
        
        unassigned_data = sum(lib.get_data_amount_gb() for lib in unassigned)
        min_lane_data, _ = self._resolve_lane_capacity_limits(unassigned, machine_type)
        
        # [2025-12-26 新增] 如果未分配数据量>=下限，尝试从中精选组建新Lane
        if unassigned_data >= min_lane_data:
            logger.info(f"挪移优化: 未分配数据量{unassigned_data:.0f}GB已足够，尝试精选组建新Lane")
            remaining, new_lanes = self._try_form_lane_from_unassigned(unassigned, machine_type)
            if new_lanes:
                return remaining, new_lanes
            logger.info(f"挪移优化: 精选组建失败，继续尝试挪移策略")
        
        needed_from_lanes = min_lane_data - unassigned_data
        logger.info(f"挪移优化: 未分配{len(unassigned)}个文库({unassigned_data:.0f}GB)，需要从现有Lane挪出{needed_from_lanes:.0f}GB")
        
        # 收集可挪移的文库（记录每个文库来自哪个Lane，用于还原）
        moved_libs: List[EnhancedLibraryInfo] = []
        moved_libs_source: Dict[str, LaneAssignment] = {}  # lib.origrec -> source_lane
        moved_data = 0.0
        
        # 优先从NB Lane挪出（未分配的大多是客户文库、非10bp）
        lane_priority = []
        for lane in all_lanes:
            if lane.lane_id.startswith('NB_'):
                lane_priority.append((0, lane))
            elif lane.lane_id.startswith('GL_'):
                lane_priority.append((1, lane))
            elif lane.lane_id.startswith('DL_'):
                lane_priority.append((2, lane))
            else:
                lane_priority.append((3, lane))
        lane_priority.sort(key=lambda x: (x[0], x[1].lane_id))
        
        for _, lane in lane_priority:
            if moved_data >= needed_from_lanes:
                break
            
            # 计算该Lane可挪出的最大数据量（保持剩余>=下限）
            lane_machine_type = lane.machine_type.value if lane.machine_type else machine_type
            lane_min_allowed, _ = self._resolve_lane_capacity_limits(
                lane.libraries,
                lane_machine_type,
                lane=lane,
            )
            can_move = lane.total_data_gb - lane_min_allowed
            if can_move <= 0:
                continue
            
            # 挑选可挪出的文库（优先客户文库，Index兼容）
            candidates = []
            for lib in lane.libraries:
                if lib.is_customer_library:
                    test_libs = unassigned + moved_libs + [lib]
                    if self.index_validator.validate_lane_quick(test_libs):
                        candidates.append(lib)
            
            # 按数据量从小到大排序（优先挪小文库）
            candidates.sort(key=lambda x: x.get_data_amount_gb())
            
            # 挪出文库（需要验证原Lane被挪出后仍然有效）
            lane_moved = 0.0
            to_remove = []
            for lib in candidates:
                if moved_data >= needed_from_lanes:
                    break
                lib_data = lib.get_data_amount_gb()
                if lane_moved + lib_data > can_move:
                    continue
                
                # 临时移除，验证原Lane是否仍然有效
                temp_remaining = [l for l in lane.libraries if l not in to_remove and l != lib]
                
                # 验证原Lane（移除后）是否仍然符合规则
                if not self._validate_lane_after_removal(temp_remaining, lane):
                    logger.debug(f"挪移优化: 从{lane.lane_id}挪出{lib.origrec}会导致原Lane规则不通过，跳过")
                    continue
                
                moved_libs.append(lib)
                moved_libs_source[lib.origrec] = lane  # 记录来源Lane
                moved_data += lib_data
                lane_moved += lib_data
                to_remove.append(lib)
                logger.debug(f"挪移优化: 从{lane.lane_id}挪出{lib.origrec}({lib_data:.1f}GB)")
            
            # 从原Lane移除
            for lib in to_remove:
                lane.libraries.remove(lib)
            
            # 更新Lane数据量
            lane.total_data_gb = sum(lib.get_data_amount_gb() for lib in lane.libraries)
            
            # 最终验证原Lane是否通过
            if to_remove and not self._validate_lane_after_removal(lane.libraries, lane):
                logger.warning(f"挪移优化: {lane.lane_id}被挪出后验证不通过，还原")
                # 还原
                for lib in to_remove:
                    lane.libraries.append(lib)
                    moved_libs.remove(lib)
                    moved_data -= lib.get_data_amount_gb()
                lane.total_data_gb = sum(lib.get_data_amount_gb() for lib in lane.libraries)
        
        # 如果挪出的数据量不足，放弃挪移（还原）
        if moved_data < needed_from_lanes * 0.9:  # 允许10%误差
            logger.warning(f"挪移优化: 挪出数据量{moved_data:.0f}GB不足，放弃挪移")
            # 还原（这里简化处理，实际应该还原到原Lane）
            return unassigned, []
        
        # 合并成新Lane
        new_lane_libs = unassigned + moved_libs
        new_lane_data = sum(lib.get_data_amount_gb() for lib in new_lane_libs)
        
        # [2025-12-26 修复] 根据实际文库内容决定Lane类型前缀
        # 判断是否为纯非10bp Lane
        bp10_data = sum(lib.get_data_amount_gb() for lib in new_lane_libs if self._library_has_10bp_index(lib))
        non_10bp_data = new_lane_data - bp10_data
        
        if bp10_data == 0 and non_10bp_data > 0:
            # 纯非10bp → NB Lane
            lane_prefix = 'NB'
        elif bp10_data > 0 and non_10bp_data == 0:
            # 纯10bp → GL Lane
            lane_prefix = 'GL'
        else:
            # 混排Lane需要检查10bp占比>=配置下限
            bp10_ratio = bp10_data / new_lane_data if new_lane_data > 0 else 0
            min_10bp_ratio = self.config.min_10bp_index_ratio
            if bp10_ratio < min_10bp_ratio:
                # 10bp占比不足阈值，需要从现有Lane补充10bp文库
                logger.warning(
                    f"挪移优化: 新Lane 10bp占比{bp10_ratio:.1%}不足{min_10bp_ratio:.0%}，尝试补充10bp文库"
                )
                
                # 从现有Lane找10bp文库补充
                for _, lane in lane_priority:
                    if bp10_ratio >= min_10bp_ratio:
                        break
                    for lib in list(lane.libraries):
                        if self._library_has_10bp_index(lib):
                            # 检查Index兼容性
                            if self.index_validator.validate_lane_quick(new_lane_libs + [lib]):
                                # 检查移除后原Lane是否仍然有效
                                temp_remaining = [l for l in lane.libraries if l != lib]
                                if len(temp_remaining) > 0 and self._validate_lane_after_removal(temp_remaining, lane):
                                    new_lane_libs.append(lib)
                                    moved_libs.append(lib)
                                    moved_libs_source[lib.origrec] = lane  # 记录来源Lane
                                    lane.libraries.remove(lib)
                                    lane.total_data_gb = sum(l.get_data_amount_gb() for l in lane.libraries)
                                    new_lane_data = sum(l.get_data_amount_gb() for l in new_lane_libs)
                                    bp10_data = sum(l.get_data_amount_gb() for l in new_lane_libs if self._library_has_10bp_index(l))
                                    bp10_ratio = bp10_data / new_lane_data if new_lane_data > 0 else 0
                                    logger.debug(f"挪移优化: 补充10bp文库{lib.origrec}，新10bp占比{bp10_ratio:.1%}")
                                    if bp10_ratio >= min_10bp_ratio:
                                        break
                
                # 如果仍然不足40%，无法形成有效Lane
                if bp10_ratio < min_10bp_ratio:
                    logger.warning("挪移优化: 无法补充足够10bp文库，放弃创建新Lane")
                    return unassigned, []
            
            lane_prefix = 'GL'
        
        new_lane_id = self._get_next_lane_id(lane_prefix, machine_type)
        machine_type_enum = self._resolve_machine_type_enum(machine_type, new_lane_libs)
        if machine_type_enum == MachineType.UNKNOWN:
            machine_type_enum = MachineType.NOVA_X_25B
        
        new_lane = LaneAssignment(
            lane_id=new_lane_id,
            machine_id=f"M_{new_lane_id[3:]}",
            machine_type=machine_type_enum,
            lane_capacity_gb=self.config.lane_capacity_gb,
            libraries=new_lane_libs,
            total_data_gb=new_lane_data
        )
        
        # [2025-12-31 新增] 最终完整验证：检查所有规则
        validation_passed = True
        validation_errors = []
        
        # 1. 容量检查
        min_allowed, max_allowed = self._resolve_lane_capacity_limits(
            new_lane_libs,
            machine_type,
            lane=new_lane,
        )
        utilization = new_lane_data / self.config.lane_capacity_gb
        if new_lane_data < min_allowed:
            validation_errors.append(f"容量不足: {new_lane_data:.1f}GB < {min_allowed:.1f}GB")
            validation_passed = False
        if new_lane_data > max_allowed:
            validation_errors.append(f"容量超限: {new_lane_data:.1f}GB > {max_allowed:.1f}GB")
            validation_passed = False
        
        # 2. Index冲突检查
        if validation_passed and self.config.enable_index_check:
            if not self.index_validator.validate_lane_quick(new_lane_libs):
                validation_errors.append("Index冲突")
                validation_passed = False
        
        # 3. 客户占比检查（按数据量计算，严格遵守规则）
        if validation_passed:
            if not self._check_customer_ratio_compatible_by_data(new_lane_libs):
                validation_errors.append("客户占比不符合规则")
                validation_passed = False
        
        # 4. 碱基不均衡占比检查
        if validation_passed:
            if not self._check_base_imbalance_compatible(new_lane_libs):
                validation_errors.append("碱基不均衡占比不符合规则")
                validation_passed = False
        
        # 5. Peak Size检查
        if validation_passed:
            if not self._check_peak_size_compatible(new_lane_libs):
                validation_errors.append("Peak Size不符合规则")
                validation_passed = False
        
        # 6. 10bp Index占比检查（仅非NB Lane）
        if validation_passed and not new_lane_id.startswith('NB_'):
            if not self._check_10bp_index_ratio_compatible(new_lane_libs):
                validation_errors.append("10bp Index占比不符合规则")
                validation_passed = False
        
        if validation_passed:
            logger.info(f"挪移优化: 创建新Lane {new_lane_id}，{len(new_lane_libs)}个文库，{new_lane_data:.0f}GB")
            return [], [new_lane]
        else:
            logger.warning(f"挪移优化: 新Lane {new_lane_id} 验证失败: {', '.join(validation_errors)}，放弃创建")
            # 还原挪出的文库到原Lane
            for lib in moved_libs:
                source_lane = moved_libs_source.get(lib.origrec)
                if source_lane and lib not in source_lane.libraries:
                    source_lane.libraries.append(lib)
                    source_lane.total_data_gb = sum(l.get_data_amount_gb() for l in source_lane.libraries)
                    logger.debug(f"挪移优化: 还原{lib.origrec}到{source_lane.lane_id}")
            return unassigned, []
    
    def _try_form_lane_from_unassigned(
        self,
        unassigned: List[EnhancedLibraryInfo],
        machine_type: str
    ) -> Tuple[List[EnhancedLibraryInfo], List[LaneAssignment]]:
        """
        [2025-12-26 新增] 尝试从未分配文库中精选组建新Lane
        
        当未分配文库总数据量>=下限但因约束（如客户占比）无法直接成Lane时，
        贪心选择符合约束的子集组建新Lane。
        
        策略：
        1. 先选内部文库（降低客户占比）
        2. 再选客户文库（保证客户占比<=50%）
        3. 确保总数据量在[下限, 上限]范围内
        4. 验证Index兼容性和其他规则
        
        Args:
            unassigned: 未分配文库列表
            machine_type: 机器类型
            
        Returns:
            (仍未分配的文库列表, 新创建的Lane列表)
        """
        from arrange_library.core.constraints.lane_validator import LaneValidator
        
        min_lane_data, max_lane_data = self._resolve_lane_capacity_limits(unassigned, machine_type)
        
        # 分离客户和内部文库
        internal_libs = [lib for lib in unassigned if not lib.is_customer_library()]
        customer_libs = [lib for lib in unassigned if lib.is_customer_library()]
        
        # 按数据量降序排序（优先选大文库）
        internal_libs.sort(key=lambda x: x.get_data_amount_gb(), reverse=True)
        customer_libs.sort(key=lambda x: x.get_data_amount_gb(), reverse=True)
        
        logger.info(f"残余精选: 内部{len(internal_libs)}个, 客户{len(customer_libs)}个")
        
        # 贪心选择
        selected: List[EnhancedLibraryInfo] = []
        total_data = 0.0
        customer_count = 0
        
        # 第一步：选内部文库
        for lib in internal_libs:
            new_data = total_data + lib.get_data_amount_gb()
            _, candidate_max = self._resolve_lane_capacity_limits(selected + [lib], machine_type)
            if new_data <= candidate_max:
                # 检查Index兼容性
                if self.config.enable_index_check and selected:
                    test_libs = selected + [lib]
                    if not self.index_validator.validate_lane_quick(test_libs):
                        continue
                selected.append(lib)
                total_data = new_data
        
        # 第二步：选客户文库（保证客户占比<=50%，按数据量计算）
        for lib in customer_libs:
            new_data = total_data + lib.get_data_amount_gb()
            
            _, candidate_max = self._resolve_lane_capacity_limits(selected + [lib], machine_type)
            if new_data <= candidate_max:
                # 计算客户占比（按数据量，与验证器一致）
                test_libs = selected + [lib]
                test_total = sum(float(getattr(l, 'contract_data_raw', 0) or 0) for l in test_libs)
                test_customer = sum(float(getattr(l, 'contract_data_raw', 0) or 0) for l in test_libs if l.is_customer_library())
                test_customer_ratio = test_customer / test_total if test_total > 0 else 0.0
                
                # [2025-12-31 新增] 严格策略：如果客户占比已接近50%，拒绝添加客户文库
                if test_customer_ratio > 0.50:
                    continue
                
                # 规则：客户占比<=50%或=100%都通过
                if test_customer_ratio > 0.50 and abs(test_customer_ratio - 1.0) > 1e-6:
                    continue
                
                # 检查Index兼容性
                if self.config.enable_index_check:
                    if not self.index_validator.validate_lane_quick(test_libs):
                        continue
                
                selected.append(lib)
                total_data = new_data
                customer_count += 1
        
        # 检查是否达到下限
        min_lane_data, _ = self._resolve_lane_capacity_limits(selected, machine_type)
        if total_data < min_lane_data:
            logger.info(f"残余精选: 数据量{total_data:.0f}GB不足下限{min_lane_data:.0f}GB，放弃")
            return unassigned, []
        
        # 使用LaneValidator进行完整验证
        validator = LaneValidator()
        
        imbalance_data = sum(
            lib.get_data_amount_gb() for lib in selected if lib.is_base_imbalance()
        )
        imbalance_ratio = imbalance_data / total_data if total_data > 0 else 0.0
        max_imbalance_ratio = self.config.max_imbalance_ratio
        use_dedicated_imbalance = (
            getattr(self.config, "enable_dedicated_imbalance_lane", False)
            and imbalance_ratio > max_imbalance_ratio
        )
        metadata = {"is_dedicated_imbalance_lane": True} if use_dedicated_imbalance else {}

        result = validator.validate_lane(
            selected, lane_id="TEMP_RESIDUAL", machine_type=machine_type, metadata=metadata
        )
        if not result.is_valid:
            error_msgs = [str(e) for e in result.errors]
            logger.info(f"残余精选: 验证失败 - {', '.join(error_msgs[:2])}")
            return unassigned, []
        
        # 确定Lane类型前缀
        bp10_data = sum(lib.get_data_amount_gb() for lib in selected if self._library_has_10bp_index(lib))
        non_10bp_data = total_data - bp10_data
        
        # [2025-12-31 新增] 如果选中的文库中有客户文库，且全部是非10bp，则不能创建NB Lane
        # NB Lane应该只包含内部文库（非客户文库）
        has_customer = any(lib.is_customer_library() for lib in selected)
        
        if use_dedicated_imbalance:
            lane_prefix = "DL"
        elif bp10_data == 0 and non_10bp_data > 0:
            if has_customer:
                # 有客户文库，不能创建NB Lane，改为GL Lane
                lane_prefix = "GL"
            else:
                lane_prefix = "NB"
        elif bp10_data > 0 and non_10bp_data == 0:
            lane_prefix = "GL"
        else:
            lane_prefix = "GL"
        
        # [2025-12-31 新增] 最终完整验证：检查所有规则（按数据量计算客户占比）
        final_total = sum(float(getattr(lib, 'contract_data_raw', 0) or 0) for lib in selected)
        final_customer = sum(float(getattr(lib, 'contract_data_raw', 0) or 0) for lib in selected if lib.is_customer_library())
        final_customer_ratio = final_customer / final_total if final_total > 0 else 0.0
        
        # 规则：客户占比 <=50% 或 =100% 都可以，其他则不行
        if final_customer_ratio > 0.50 and abs(final_customer_ratio - 1.0) > 1e-6:
            logger.warning(f"残余精选: 客户占比{final_customer_ratio:.1%}不符合规则，拒绝创建Lane")
            return unassigned, []
        
        # 创建新Lane
        new_lane_id = self._get_next_lane_id(lane_prefix, machine_type)
        machine_type_enum = self._resolve_machine_type_enum(machine_type, selected)
        if machine_type_enum == MachineType.UNKNOWN:
            machine_type_enum = MachineType.NOVA_X_25B
        
        new_lane = LaneAssignment(
            lane_id=new_lane_id,
            machine_id=f"M_{new_lane_id[3:]}",
            machine_type=machine_type_enum,
            lane_capacity_gb=self.config.lane_capacity_gb,
            libraries=selected,
            total_data_gb=total_data
        )
        
        # [2025-12-31 新增] 最终完整验证：检查所有规则
        validation_passed = True
        
        # 1. Index冲突检查
        if self.config.enable_index_check:
            if not self.index_validator.validate_lane_quick(selected):
                logger.warning(f"残余精选: {new_lane_id} Index冲突，拒绝创建")
                validation_passed = False
        
        # 2. 碱基不均衡占比检查
        if (
            validation_passed
            and not use_dedicated_imbalance
            and not self._check_base_imbalance_compatible(selected)
        ):
            logger.warning(f"残余精选: {new_lane_id} 碱基不均衡占比不符合规则，拒绝创建")
            validation_passed = False
        
        # 3. Peak Size检查
        if validation_passed and not self._check_peak_size_compatible(selected):
            logger.warning(f"残余精选: {new_lane_id} Peak Size不符合规则，拒绝创建")
            validation_passed = False
        
        # 4. 10bp Index占比检查（仅非NB Lane）
        if validation_passed and not new_lane_id.startswith('NB_'):
            if not self._check_10bp_index_ratio_compatible(selected):
                logger.warning(f"残余精选: {new_lane_id} 10bp Index占比不符合规则，拒绝创建")
                validation_passed = False
        
        if not validation_passed:
            return unassigned, []
        
        # 计算剩余未分配
        selected_ids = {lib.origrec for lib in selected}
        remaining = [lib for lib in unassigned if lib.origrec not in selected_ids]
        
        customer_ratio = final_customer_ratio * 100  # 转换为百分比
        logger.info(f"残余精选: 成功创建{new_lane_id}，{len(selected)}个文库，{total_data:.0f}GB，客户占比{customer_ratio:.1f}%")
        logger.info(f"残余精选: 剩余{len(remaining)}个文库无法分配")
        
        return remaining, [new_lane]
    
    def _validate_lane_after_removal(
        self,
        remaining_libs: List[EnhancedLibraryInfo],
        lane: LaneAssignment
    ) -> bool:
        """
        验证Lane被挪出部分文库后是否仍然有效
        
        检查项：
        1. 容量是否在有效范围内（>=下限容量）
        2. Index是否有冲突
        3. 其他规则验证
        
        Args:
            remaining_libs: 剩余的文库列表
            lane: 原Lane对象（用于获取元信息）
            
        Returns:
            True表示仍然有效，False表示无效
        """
        if not remaining_libs:
            return False
        
        # 1. 容量检查
        remaining_data = sum(lib.get_data_amount_gb() for lib in remaining_libs)
        machine_type = lane.machine_type.value if lane.machine_type else "Nova X-25B"
        min_lane_data, _ = self._resolve_lane_capacity_limits(
            remaining_libs,
            machine_type,
            lane=lane,
        )
        if remaining_data < min_lane_data:
            return False
        
        # 2. Index检查
        if not self.index_validator.validate_lane_quick(remaining_libs):
            return False
        
        # 3. 使用LaneValidator进行完整规则验证
        metadata = self._build_lane_validation_metadata(lane)
        
        result = self.lane_validator.validate_lane(
            remaining_libs, 
            lane_id=lane.lane_id, 
            metadata=metadata
        )
        
        return result.is_valid

    def _build_lane_validation_metadata(self, lane: LaneAssignment) -> Dict[str, Any]:
        """构造校验元信息，只保留当前Lane真实需要的上下文。"""
        lane_metadata = dict(getattr(lane, "metadata", {}) or {})
        metadata: Dict[str, Any] = {}

        # 只透传会影响校验行为、且不会因种子文库变化而过期的元数据。
        if lane_metadata.get("is_package_lane"):
            metadata["is_package_lane"] = True
        balance_data = lane_metadata.get("wkbalancedata")
        if balance_data is None:
            balance_data = lane_metadata.get("wkadd_balance_data")
        if balance_data is None:
            balance_data = lane_metadata.get("required_balance_data_gb")
        if balance_data is not None:
            metadata["wkbalancedata"] = balance_data

        metadata.update({
            'is_dedicated_imbalance_lane': lane.lane_id.startswith('DL_'),
            'is_pure_non_10bp_lane': lane.lane_id.startswith('NB_'),
            'is_backbone_lane': lane.lane_id.startswith('BL_')
        })
        if lane.libraries:
            first_lib = lane.libraries[0]
            process_code = getattr(first_lib, 'process_code', None)
            if process_code is not None:
                metadata['process_code'] = process_code
            test_code = getattr(first_lib, 'test_code', None)
            if test_code is not None:
                metadata['test_code'] = test_code
            test_no = getattr(first_lib, 'test_no', '')
            if test_no is not None and str(test_no).strip() != '':
                metadata['test_no'] = test_no
        return metadata


def create_greedy_scheduler(
    lane_capacity_gb: float = 1000.0,
    enable_index_check: bool = True
) -> GreedyLaneScheduler:
    """
    创建贪心排机器的便捷方法
    
    Args:
        lane_capacity_gb: Lane容量（GB）
        enable_index_check: 是否启用Index冲突检查
        
    Returns:
        GreedyLaneScheduler实例
    """
    config = GreedyLaneConfig(
        lane_capacity_gb=lane_capacity_gb,
        enable_index_check=enable_index_check
    )
    return GreedyLaneScheduler(config)
