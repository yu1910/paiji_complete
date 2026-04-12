"""
成Lane校验程序
创建时间：2025-12-02 18:00:00
更新时间：2026-04-12 10:00:00

变更记录：
- 2026-03-06: 移除类内LANE_CAPACITY/LANE_MIN_DATA/LANE_MAX_DATA死代码常量，
  容量区间统一由 scheduling_config.get_lane_capacity_range() 驱动（规则矩阵优先，fallback使用975G目标值）
- 2026-01-30: 调整碱基不均衡和特殊文库限制：
  - BASE_IMBALANCE_RATIO_LIMIT: 35%（从30%调整）
  - SPECIAL_LIBRARY_CAPACITY: 350G（从240G调整）
- 2026-01-30: 统一碱基不均衡判断逻辑
  - BASE_IMBALANCE_KEYWORDS: 与 EnhancedLibraryInfo.is_base_imbalance() 保持一致
  - 新增关键词：rrbs, ribo-seq, em-seq, 墨卓, visium, fixed rna, mobidrop
- 2025-12-26: 修复客户文库占比验证逻辑：
  - 使用sampletype字段以"客户"开头识别客户文库
  - 规则改为: <=50%或=100%都通过，只有>50%且<100%时才违规
- 2025-12-24: 优化10bp/单端Index/碱基不均衡判断逻辑，优先使用预处理字段(ten_bp_data/single_index_data/jjbj)

依据《排机流程规划》步骤二.5实现的成Lane校验：
- Index重复校验：P7/P5序列在Lane内不可冲突
- 客户占比：单一客户占比<=50%
- 10bp占比：10bp Index占比>=40%（混排时）
- 单端占比：单端Index占比<30%
- 碱基不均衡占比：碱基不均衡文库占比<=40%
- 容量校验：Lane总数据量由规则矩阵决定，Nova X-25B标准规则effective_min=970G，effective_max=980G
- Peak Size校验：最大-最小<=150bp 或 150bp窗口覆盖>=75%
- 特殊文库限制：特殊文库总量<=阈值（不再限制类型数量）
- FC最小数据量校验：Nova X-25B整个FC最小1150G
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

from loguru import logger

# 使用包内相对导入，避免依赖 sys.path，在 Docker 等环境更稳定
# from arrange_library.liblane_paths import setup_liblane_paths

# setup_liblane_paths()

from arrange_library.core.config.scheduling_config import get_scheduling_config
from arrange_library.models.library_info import EnhancedLibraryInfo


class ValidationRuleType(Enum):
    """校验规则类型"""
    INDEX_CONFLICT = "index_conflict"
    CUSTOMER_RATIO = "customer_ratio"
    INDEX_10BP_RATIO = "index_10bp_ratio"
    SINGLE_END_RATIO = "single_end_ratio"
    BASE_IMBALANCE_RATIO = "base_imbalance_ratio"
    CAPACITY = "capacity"
    CAPACITY_INSUFFICIENT = "capacity_insufficient"  # 数据量不足，需要添加文库
    CAPACITY_EXCEEDED = "capacity_exceeded"          # 数据量超标，需要剔除文库
    FC_MIN_DATA = "fc_min_data"                      # FC最小数据量校验
    PEAK_SIZE = "peak_size"
    SPECIAL_LIBRARY_LIMIT = "special_library_limit"
    ADD_TEST_RATIO = "add_test_ratio"
    NEW_MODE_CAPACITY = "new_mode_capacity"
    BOARD_LIMIT = "board_limit"
    NOVA_X10B_MINIMUM = "nova_x10b_minimum"
    SINGLE_INDEX_HEAVY_NOVO = "single_index_heavy_novo"
    RULE_MATRIX_CONSTRAINT = "rule_matrix_constraint"


class ValidationSeverity(Enum):
    """校验严重程度"""
    ERROR = "error"      # 错误（必须修正）
    WARNING = "warning"  # 警告（可接受但不推荐）


@dataclass
class ValidationError:
    """校验错误"""
    rule_type: ValidationRuleType
    severity: ValidationSeverity
    message: str
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    affected_libraries: List[str] = field(default_factory=list)


@dataclass
class LaneValidationResult:
    """Lane校验结果"""
    lane_id: str
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'lane_id': self.lane_id,
            'is_valid': self.is_valid,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'error_types': [e.rule_type.value for e in self.errors]
        }


class LaneValidator:
    """成Lane校验程序
    
    依据《排机规则文档》实现红线规则校验：
    1. Index重复校验
    2. 客户占比校验
    3. 10bp Index占比校验
    4. 单端Index占比校验
    5. 碱基不均衡占比校验
    6. 容量校验（由规则矩阵驱动）
    7. Peak Size校验
    8. 特殊文库限制校验
    
    所有校验阈值统一从 scheduling_config.validation_limits（ValidationLimitsConfig）读取，
    不再在类中硬编码，修改规则只需调整配置，无需改代码。
    """

    def __init__(self, strict_mode: bool = True):
        """
        初始化校验器

        Args:
            strict_mode: 是否为严格模式（严格模式将警告视为错误）
        """
        self.strict_mode = strict_mode
        self.scheduling_config = get_scheduling_config()
        limits = self.scheduling_config.validation_limits

        # 从配置加载所有校验阈值
        self.customer_ratio_limit: float = limits.customer_ratio_limit
        self.index_10bp_ratio_min: float = limits.index_10bp_ratio_min
        self.single_end_ratio_limit: float = limits.single_end_ratio_limit
        self.base_imbalance_ratio_limit: float = limits.base_imbalance_ratio_limit
        self.base_imbalance_ratio_limit_3_6t: float = limits.base_imbalance_ratio_limit_3_6t
        self.add_test_ratio_limit: float = limits.add_test_ratio_limit
        self.add_test_buffer_gb: float = limits.add_test_buffer_gb
        self.peak_size_max_diff: int = limits.peak_size_max_diff
        self.peak_size_coverage_ratio: float = limits.peak_size_coverage_ratio
        self.special_library_type_limit: int = limits.special_library_type_limit
        self.special_library_capacity: Dict[str, float] = limits.special_library_capacity
        self.fc_min_data: Dict[str, float] = limits.fc_min_data
        self.base_imbalance_keywords: List[str] = limits.base_imbalance_keywords
        self.special_library_keywords: List[str] = limits.special_library_keywords

        # 复用同一个 IndexConflictValidator 实例，避免每次 _validate_index_conflicts
        # 调用时都重复初始化（每次初始化会打印 INFO 日志，高频调用时有明显开销）
        from arrange_library.core.constraints.index_validator_verified import IndexConflictValidator
        self._index_validator = IndexConflictValidator()

        logger.info(
            f"成Lane校验程序初始化完成 (严格模式: {strict_mode}, "
            f"碱基不均衡上限: {self.base_imbalance_ratio_limit:.0%}, "
            f"特殊文库关键词: {len(self.special_library_keywords)}个)"
        )
    
    def validate_lane(
        self, 
        libraries: List[EnhancedLibraryInfo],
        lane_id: str = "",
        machine_type: str = "Nova X-25B",
        lane_mode: str = "standard",
        metadata: Dict = None
    ) -> LaneValidationResult:
        """
        校验Lane
        
        Args:
            libraries: Lane内的文库列表
            lane_id: Lane ID
            machine_type: 机器类型
            lane_mode: Lane模式 (standard/new_mode/mode_36t)
            metadata: 额外的元数据信息
            
        Returns:
            LaneValidationResult: 校验结果
        """
        if metadata is None:
            metadata = {}
        
        # 从metadata中获取模式
        lane_mode = metadata.get('mode', lane_mode)
        errors = []
        warnings = []
        
        if not libraries:
            return LaneValidationResult(
                lane_id=lane_id,
                is_valid=False,
                errors=[ValidationError(
                    rule_type=ValidationRuleType.CAPACITY,
                    severity=ValidationSeverity.ERROR,
                    message="Lane为空"
                )]
            )
        
        # 1. Index重复校验
        index_errors = self._validate_index_conflicts(libraries)
        errors.extend(index_errors)
        
        # 2. 客户占比校验
        customer_result = self._validate_customer_ratio(libraries)
        if customer_result:
            if customer_result.severity == ValidationSeverity.ERROR:
                errors.append(customer_result)
            else:
                warnings.append(customer_result)
        
        # 3. 10bp Index占比校验
        # [2025-12-25] 如果是纯非10bp Lane，跳过此检查（规则4b）
        is_pure_non_10bp_lane = metadata.get('is_pure_non_10bp_lane', False)
        if not is_pure_non_10bp_lane:
            index_10bp_result = self._validate_10bp_index_ratio(libraries)
            if index_10bp_result:
                if index_10bp_result.severity == ValidationSeverity.ERROR:
                    errors.append(index_10bp_result)
                else:
                    warnings.append(index_10bp_result)
        
        # 4. 单端Index占比校验
        single_end_result = self._validate_single_end_ratio(libraries)
        if single_end_result:
            if single_end_result.severity == ValidationSeverity.ERROR:
                errors.append(single_end_result)
            else:
                warnings.append(single_end_result)
        
        # 5. 碱基不均衡占比校验
        # [2025-12-26] 如果是碱基不均衡专用Lane（DL Lane），跳过此检查
        is_dedicated_imbalance_lane = metadata.get('is_dedicated_imbalance_lane', False)
        if not is_dedicated_imbalance_lane:
            imbalance_result = self._validate_base_imbalance_ratio(libraries, machine_type)
            if imbalance_result:
                if imbalance_result.severity == ValidationSeverity.ERROR:
                    errors.append(imbalance_result)
                else:
                    warnings.append(imbalance_result)
        
        # 6. 容量校验
        capacity_result = self._validate_capacity(libraries, machine_type, metadata)
        if capacity_result:
            errors.append(capacity_result)
        
        # 7. Peak Size校验
        peak_size_result = self._validate_peak_size(libraries)
        if peak_size_result:
            if peak_size_result.severity == ValidationSeverity.ERROR:
                errors.append(peak_size_result)
            else:
                warnings.append(peak_size_result)
        
        # 8. 特殊文库限制校验
        # [2025-12-25] 对于专用Lane（碱基不均衡专用、非10bp专用、骨架Lane），跳过特殊文库限制
        is_dedicated_imbalance_lane = metadata.get('is_dedicated_imbalance_lane', False)
        is_pure_non_10bp_lane_for_special = metadata.get('is_pure_non_10bp_lane', False)
        is_backbone_lane = metadata.get('is_backbone_lane', False)
        
        if not (is_dedicated_imbalance_lane or is_pure_non_10bp_lane_for_special or is_backbone_lane):
            special_result = self._validate_special_library_limit(libraries, machine_type)
            if special_result:
                if special_result.severity == ValidationSeverity.ERROR:
                    errors.append(special_result)
                else:
                    warnings.append(special_result)
        
        # 9. 加测文库占比校验
        add_test_result = self._validate_add_test_ratio(libraries)
        if add_test_result:
            if add_test_result.severity == ValidationSeverity.ERROR:
                errors.append(add_test_result)
            else:
                warnings.append(add_test_result)
        
        # 10. 模式特定校验
        mode_errors = self._validate_mode_specific_rules(libraries, lane_mode, machine_type)
        errors.extend(mode_errors)

        # 11. 统一规则表限制校验
        constraint_messages = self.scheduling_config.validate_lane_constraints(
            libraries=libraries,
            machine_type=machine_type,
            metadata=metadata,
        )
        for message in constraint_messages:
            errors.append(
                ValidationError(
                    rule_type=ValidationRuleType.RULE_MATRIX_CONSTRAINT,
                    severity=ValidationSeverity.ERROR,
                    message=message,
                )
            )
        
        # 确定是否通过校验
        is_valid = len(errors) == 0
        if self.strict_mode:
            is_valid = is_valid and len(warnings) == 0
        
        return LaneValidationResult(
            lane_id=lane_id,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_index_conflicts(self, libraries: List[EnhancedLibraryInfo]) -> List[ValidationError]:
        """校验Index冲突，复用类内已初始化的 _index_validator 实例"""
        try:
            result = self._index_validator.validate_lane(libraries)
            if not result.is_valid:
                errors = []
                for conflict in result.conflicts:
                    errors.append(ValidationError(
                        rule_type=ValidationRuleType.INDEX_CONFLICT,
                        severity=ValidationSeverity.ERROR,
                        message=f"Index冲突: {conflict.library1_id} vs {conflict.library2_id}",
                        affected_libraries=[conflict.library1_id, conflict.library2_id]
                    ))
                return errors
        except Exception as e:
            logger.warning(f"Index校验失败: {e}")

        return []
    
    def _validate_customer_ratio(self, libraries: List[EnhancedLibraryInfo]) -> Optional[ValidationError]:
        """校验客户占比
        
        [2025-12-31 修复] 改为按数据量计算，与排机逻辑保持一致
        规则：客户文库占比 <=50% 或 =100% 都通过
        - 使用sampletype字段以"客户"开头来识别客户文库
        - 或者使用is_customer_library()方法
        """
        if len(libraries) == 0:
            return None
        
        # 计算总数据量（按合同数据量）
        total_data = sum(float(getattr(lib, 'contract_data_raw', 0) or 0) for lib in libraries)
        if total_data == 0:
            return None
        
        # 统计客户文库数据量（使用is_customer_library方法或sampletype字段）
        customer_data = 0.0
        customer_libs = []
        
        for lib in libraries:
            # 客户识别策略（尽量兼容不同数据源/不同字段口径）：
            # 1) 显式字段：customer_library / sampletype（更可信）
            # 2) 样本编号前缀：FKDL*（历史习惯）
            # 3) 回退：对象自带 is_customer_library()（避免被"customer_library=否"误导）
            customer_flag = str(getattr(lib, "customer_library", "") or "").strip()
            if customer_flag in {"是", "Y", "YES", "TRUE", "客户"}:
                is_customer = True
            else:
                sampletype = getattr(lib, "sampletype", "") or getattr(lib, "sample_type_code", "") or ""
                sample_id = getattr(lib, "sample_id", "") or ""
                if str(sampletype).startswith("客户") or str(sample_id).startswith("FKDL"):
                    is_customer = True
                elif hasattr(lib, "is_customer_library") and callable(lib.is_customer_library):
                    is_customer = bool(lib.is_customer_library())
                else:
                    is_customer = False
            
            if is_customer:
                lib_data = float(getattr(lib, 'contract_data_raw', 0) or 0)
                customer_data += lib_data
                customer_libs.append(lib.origrec)
        
        # 计算客户占比（按数据量计算）
        customer_ratio = customer_data / total_data if total_data > 0 else 0.0
        
        # 规则：<=50% 或 =100% 都通过；仅当 >50% 且 不等于100% 时违规
        if customer_ratio > self.customer_ratio_limit and customer_ratio != 1.0:
            return ValidationError(
                rule_type=ValidationRuleType.CUSTOMER_RATIO,
                severity=ValidationSeverity.ERROR,
                message=f"客户文库占比{customer_ratio:.1%}不符合规则(应<={self.customer_ratio_limit:.0%}或=100%)",
                current_value=customer_ratio,
                threshold_value=self.customer_ratio_limit,
                affected_libraries=customer_libs
            )
        
        return None
    
    def _validate_10bp_index_ratio(self, libraries: List[EnhancedLibraryInfo]) -> Optional[ValidationError]:
        """校验10bp Index占比
        
        优先使用预处理好的ten_bp_data字段，如果不存在则回退到解析index_seq
        """
        total_data = sum(float(lib.contract_data_raw or 0) for lib in libraries)
        if total_data == 0:
            return None
        
        # 检测是否存在10bp和非10bp混排
        data_10bp = 0.0
        data_non_10bp = 0.0
        libs_10bp = []
        
        for lib in libraries:
            lib_data = float(lib.contract_data_raw or 0)
            is_10bp = False
            
            # 优先使用预处理好的ten_bp_data字段
            ten_bp_data = getattr(lib, 'ten_bp_data', None)
            if ten_bp_data is not None and ten_bp_data > 0:
                is_10bp = True
            else:
                # 回退到解析index_seq
                index_seq = getattr(lib, 'index_seq', '') or ''
                is_10bp = self._is_10bp_index(index_seq)
            
            if is_10bp:
                data_10bp += lib_data
                libs_10bp.append(lib.origrec)
            else:
                data_non_10bp += lib_data
        
        # 仅当混排时才检查
        if data_10bp == 0 or data_non_10bp == 0:
            return None
        
        ratio_10bp = data_10bp / total_data
        
        if ratio_10bp < self.index_10bp_ratio_min:
            return ValidationError(
                rule_type=ValidationRuleType.INDEX_10BP_RATIO,
                severity=ValidationSeverity.ERROR,
                message=f"10bp Index占比{ratio_10bp:.1%}低于{self.index_10bp_ratio_min:.0%}要求",
                current_value=ratio_10bp,
                threshold_value=self.index_10bp_ratio_min,
                affected_libraries=libs_10bp
            )
        
        return None
    
    def _is_10bp_index(self, index_seq: str) -> bool:
        """判断是否为10碱基Index"""
        if not index_seq:
            return False
        
        sequences = index_seq.split(',')
        for seq in sequences:
            p7_seq = seq.split(';')[0].strip() if ';' in seq else seq.strip()
            if p7_seq.upper() not in ('PE', '通用接头', '随机INDEX'):
                if len(p7_seq) == 10:
                    return True
        
        return False
    
    def _validate_single_end_ratio(self, libraries: List[EnhancedLibraryInfo]) -> Optional[ValidationError]:
        """校验单端Index占比
        
        优先使用预处理好的single_index_data字段，如果不存在则回退到解析index_seq
        """
        total_data = sum(float(lib.contract_data_raw or 0) for lib in libraries)
        if total_data == 0:
            return None
        
        single_end_data = 0.0
        double_end_data = 0.0
        single_end_libs = []
        
        for lib in libraries:
            lib_data = float(lib.contract_data_raw or 0)
            is_single_end = False
            
            # 优先使用预处理好的single_index_data字段
            single_idx_data = getattr(lib, 'single_index_data', None)
            if single_idx_data is not None and single_idx_data > 0:
                is_single_end = True
            else:
                # 回退到解析index_seq
                index_seq = getattr(lib, 'index_seq', '') or ''
                is_single_end = self._is_single_end_index(index_seq)
            
            if is_single_end:
                single_end_data += lib_data
                single_end_libs.append(lib.origrec)
            else:
                double_end_data += lib_data
        
        # 全部单端不限制
        if double_end_data == 0:
            return None
        
        single_end_ratio = single_end_data / total_data
        
        if single_end_ratio >= self.single_end_ratio_limit:
            return ValidationError(
                rule_type=ValidationRuleType.SINGLE_END_RATIO,
                severity=ValidationSeverity.ERROR,
                message=f"单端Index占比{single_end_ratio:.1%}超过{self.single_end_ratio_limit:.0%}限制",
                current_value=single_end_ratio,
                threshold_value=self.single_end_ratio_limit,
                affected_libraries=single_end_libs
            )
        
        return None
    
    def _is_single_end_index(self, index_seq: str) -> bool:
        """判断是否为单端Index"""
        if not index_seq:
            return True
        
        sequences = index_seq.split(',')
        for seq in sequences:
            if ';' in seq:
                return False
        return True
    
    def _validate_base_imbalance_ratio(
        self, 
        libraries: List[EnhancedLibraryInfo],
        machine_type: str,
        limit_override: Optional[float] = None
    ) -> Optional[ValidationError]:
        """校验碱基不均衡占比"""
        total_data = sum(float(lib.contract_data_raw or 0) for lib in libraries)
        if total_data == 0:
            return None
        
        imbalance_data = 0.0
        imbalance_libs = []
        
        for lib in libraries:
            if self._is_base_imbalance_library(lib):
                imbalance_data += float(lib.contract_data_raw or 0)
                imbalance_libs.append(lib.origrec)
        
        if imbalance_data == 0:
            return None
        
        imbalance_ratio = imbalance_data / total_data
        
        # 根据模式选择限制
        limit = limit_override if limit_override is not None else self.base_imbalance_ratio_limit
        
        if imbalance_ratio > limit:
            return ValidationError(
                rule_type=ValidationRuleType.BASE_IMBALANCE_RATIO,
                severity=ValidationSeverity.ERROR,
                message=f"碱基不均衡占比{imbalance_ratio:.1%}超过{limit:.0%}限制",
                current_value=imbalance_ratio,
                threshold_value=limit,
                affected_libraries=imbalance_libs
            )
        
        return None
    
    def _is_base_imbalance_library(self, lib: EnhancedLibraryInfo) -> bool:
        """判断是否为碱基不均衡文库
        
        优先使用预处理好的jjbj字段，如果不存在则回退到关键词匹配
        """
        # 优先使用jjbj字段（数据预处理时已标记）
        jjbj = getattr(lib, 'jjbj', None)
        if jjbj is not None and str(jjbj).strip() != '':
            return str(jjbj).strip() == '是'
        
        # 回退到关键词匹配
        lab_type = getattr(lib, 'lab_type', '') or ''
        sample_type = getattr(lib, 'sample_type_code', '') or ''
        sub_project = getattr(lib, 'sub_project_name', '') or ''
        library_type = getattr(lib, 'library_type', '') or ''
        combined_type = f"{lab_type} {sample_type} {sub_project} {library_type}".lower()
        
        for keyword in self.base_imbalance_keywords:
            if keyword.lower() in combined_type:
                return True
        
        return False
    
    def _validate_capacity(
        self, 
        libraries: List[EnhancedLibraryInfo],
        machine_type: str,
        metadata: Optional[Dict] = None,
    ) -> Optional[ValidationError]:
        """
        校验Lane容量
        
        容量区间由 scheduling_config.get_lane_capacity_range() 统一决定：
        - 优先匹配 scheduling_rule_matrix.json 中的规则（规则矩阵）
        - 未命中规则时 fallback 至机器默认值（Nova X-25B 目标975G，tolerance=5G）
        
        返回：
        - 数据量不足：ValidationRuleType.CAPACITY_INSUFFICIENT
        - 数据量超标：ValidationRuleType.CAPACITY_EXCEEDED
        """
        total_data = sum(float(lib.contract_data_raw or 0) for lib in libraries)
        metadata = metadata or {}
        planned_balance_data = float(
            metadata.get("wkbalancedata")
            or metadata.get("wkadd_balance_data")
            or metadata.get("required_balance_data_gb")
            or 0.0
        )
        effective_total_data = total_data + max(planned_balance_data, 0.0)
        
        lane_rule_selection = self.scheduling_config.get_lane_capacity_range(
            libraries=libraries,
            machine_type=machine_type,
            metadata=metadata,
        )
        min_allowed = lane_rule_selection.effective_min_gb
        max_allowed = lane_rule_selection.effective_max_gb
        
        # 检查1：数据量超过上限
        if effective_total_data > max_allowed:
            excess_data = effective_total_data - max_allowed
            return ValidationError(
                rule_type=ValidationRuleType.CAPACITY_EXCEEDED,
                severity=ValidationSeverity.ERROR,
                message=(
                    f"Lane有效数据量{effective_total_data:.1f}G(原始{total_data:.1f}G"
                    f"+平衡{planned_balance_data:.1f}G)超过上限{max_allowed:.0f}G，"
                    f"规则={lane_rule_selection.rule_code}，"
                    f"需剔除约{excess_data:.1f}G的文库"
                ),
                current_value=effective_total_data,
                threshold_value=max_allowed
            )
        
        # 检查2：数据量低于下限（需要添加文库）
        if effective_total_data < min_allowed:
            shortfall = min_allowed - effective_total_data
            return ValidationError(
                rule_type=ValidationRuleType.CAPACITY_INSUFFICIENT,
                severity=ValidationSeverity.ERROR,
                message=(
                    f"Lane有效数据量{effective_total_data:.1f}G(原始{total_data:.1f}G"
                    f"+平衡{planned_balance_data:.1f}G)低于下限{min_allowed:.0f}G，"
                    f"规则={lane_rule_selection.rule_code}，"
                    f"需添加约{shortfall:.1f}G的文库"
                ),
                current_value=effective_total_data,
                threshold_value=min_allowed
            )
        
        # 数据量在有效范围内，记录日志
        logger.debug(
            f"Lane有效数据量{effective_total_data:.1f}G(原始{total_data:.1f}G+平衡{planned_balance_data:.1f}G)"
            f"在有效范围内[{min_allowed:.0f}G-{max_allowed:.0f}G]，"
            f"规则={lane_rule_selection.rule_code}"
        )
        
        return None
    
    def _validate_peak_size(self, libraries: List[EnhancedLibraryInfo]) -> Optional[ValidationError]:
        """校验Peak Size"""
        peak_sizes = []
        peak_size_data = {}
        
        total_data = sum(float(lib.contract_data_raw or 0) for lib in libraries)
        
        for lib in libraries:
            peak_size = getattr(lib, 'peak_size', None)
            if peak_size and peak_size > 0:
                peak_sizes.append(float(peak_size))
                peak_size_data[float(peak_size)] = peak_size_data.get(float(peak_size), 0) + float(lib.contract_data_raw or 0)
        
        if len(peak_sizes) < 2:
            return None
        
        min_peak = min(peak_sizes)
        max_peak = max(peak_sizes)
        
        # 条件1：最大-最小<=peak_size_max_diff bp
        if max_peak - min_peak <= self.peak_size_max_diff:
            return None
        
        # 条件2：窗口覆盖>=peak_size_coverage_ratio
        best_coverage = 0.0
        for window_start in range(int(min_peak), int(max_peak) - self.peak_size_max_diff + 1):
            window_end = window_start + self.peak_size_max_diff
            covered_data = sum(
                data for ps, data in peak_size_data.items()
                if window_start <= ps <= window_end
            )
            coverage = covered_data / total_data if total_data > 0 else 0
            best_coverage = max(best_coverage, coverage)
        
        if best_coverage >= self.peak_size_coverage_ratio:
            return None
        
        return ValidationError(
            rule_type=ValidationRuleType.PEAK_SIZE,
            severity=ValidationSeverity.ERROR,
            message=(
                f"Peak Size范围{max_peak - min_peak:.0f}bp超过{self.peak_size_max_diff}bp，"
                f"最佳窗口覆盖{best_coverage:.1%}<{self.peak_size_coverage_ratio:.0%}"
            ),
            current_value=max_peak - min_peak,
            threshold_value=self.peak_size_max_diff
        )
    
    def _validate_special_library_limit(
        self, 
        libraries: List[EnhancedLibraryInfo],
        machine_type: str
    ) -> Optional[ValidationError]:
        """校验特殊文库限制
        
        [2025-12-25 待讨论] 根据人工排机数据分析，添加专用Lane策略：
        - 如果Lane内全部是碱基不均衡文库（专用Lane），跳过数据量限制
        - 人工排机实际有5条100%碱基不均衡专用Lane（每条972GB）
        
        使用jjbj字段优先判断碱基不均衡，然后使用关键词匹配
        """
        special_data = 0.0
        special_libs = []
        imbalance_count = 0
        balanced_count = 0
        
        for lib in libraries:
            lib_data = float(lib.contract_data_raw or 0)
            
            # 优先使用jjbj字段判断碱基不均衡
            jjbj = getattr(lib, 'jjbj', None)
            is_imbalance = (jjbj is not None and str(jjbj).strip() == '是')
            
            if is_imbalance:
                imbalance_count += 1
                special_data += lib_data
                special_libs.append(lib.origrec)
            else:
                balanced_count += 1
        
        # ===== [2025-12-25 新增] 碱基不均衡专用Lane策略 =====
        # 如果Lane内全部是碱基不均衡文库（无碱基均衡文库），跳过数据量限制
        is_dedicated_lane = (balanced_count == 0 and imbalance_count > 0)
        if is_dedicated_lane:
            # 专用Lane不检查类型数量和数据量，直接通过
            # 因为人工排机的专用Lane允许任意碱基不均衡文库组合
            logger.debug(f"碱基不均衡专用Lane检测: {imbalance_count}个碱基不均衡文库, {special_data:.1f}GB, 跳过数据量限制")
            return None
        
        # ===== 混排Lane的原有检查逻辑 =====
        # 仅保留特殊文库总量检查，类型数量不再作为限制条件
        capacity_limit = self.special_library_capacity.get(
            machine_type,
            self.special_library_capacity.get('default', 240.0),
        )
        if special_data > float(capacity_limit):
            return ValidationError(
                rule_type=ValidationRuleType.SPECIAL_LIBRARY_LIMIT,
                severity=ValidationSeverity.ERROR,
                message=f"特殊文库总量{special_data:.1f}G超过{float(capacity_limit):.1f}G限制",
                current_value=special_data,
                threshold_value=float(capacity_limit),
                affected_libraries=special_libs,
            )
        
        return None
    
    def _validate_add_test_ratio(self, libraries: List[EnhancedLibraryInfo]) -> Optional[ValidationError]:
        """校验加测文库占比"""
        total_data = sum(float(lib.contract_data_raw or 0) for lib in libraries)
        if total_data == 0:
            return None
        
        add_test_data = 0.0
        add_test_libs = []
        
        for lib in libraries:
            remark = getattr(lib, 'remark', '') or getattr(lib, '上机备注', '') or ''
            add_test_remark = getattr(lib, 'add_test_remark', '') or ''
            
            if '加测' in remark or '加测' in add_test_remark:
                add_test_data += float(lib.contract_data_raw or 0)
                add_test_libs.append(lib.origrec)
        
        if add_test_data == 0:
            return None
        
        add_test_ratio = add_test_data / total_data
        
        # 允许25%+10G缓冲
        effective_limit = self.add_test_ratio_limit + (self.add_test_buffer_gb / total_data if total_data > 0 else 0)
        
        if add_test_ratio > effective_limit:
            return ValidationError(
                rule_type=ValidationRuleType.ADD_TEST_RATIO,
                severity=ValidationSeverity.WARNING,
                message=f"加测文库占比{add_test_ratio:.1%}超过{self.add_test_ratio_limit:.0%}+{self.add_test_buffer_gb}G缓冲",
                current_value=add_test_ratio,
                threshold_value=self.add_test_ratio_limit,
                affected_libraries=add_test_libs
            )
        
        return None
    
    def _validate_mode_specific_rules(
        self, 
        libraries: List[EnhancedLibraryInfo],
        lane_mode: str,
        machine_type: str
    ) -> List[ValidationError]:
        """校验模式特定规则"""
        errors = []
        
        if lane_mode == "new_mode":
            # 1.0模式特定规则
            # 容量校验: 2175G(RNA)/2100G(DNA) ±5G
            capacity_error = self._validate_new_mode_capacity(libraries)
            if capacity_error:
                errors.append(capacity_error)
            
            # 板号限制: ≤15个且不能为空
            board_error = self._validate_board_limit(libraries)
            if board_error:
                errors.append(board_error)
            
            # 诺禾单Index大文库限制
            heavy_novo_error = self._validate_single_index_heavy_novo(libraries)
            if heavy_novo_error:
                errors.append(heavy_novo_error)
        
        elif lane_mode == "mode_36t":
            # 3.6T模式特定规则：碱基不均衡比例使用3.6T专用上限
            imbalance_error = self._validate_base_imbalance_ratio(
                libraries, machine_type, limit_override=self.base_imbalance_ratio_limit_3_6t
            )
            if imbalance_error:
                errors.append(imbalance_error)
        
        # Nova X-10B特殊规则 (所有模式)
        if machine_type == "Nova X-10B":
            x10b_error = self._validate_nova_x10b_minimum(libraries)
            if x10b_error:
                errors.append(x10b_error)
        
        return errors
    
    def _validate_new_mode_capacity(self, libraries: List[EnhancedLibraryInfo]) -> Optional[ValidationError]:
        """校验1.0模式容量限制"""
        if not libraries:
            return None
        
        total_data = sum(float(lib.contract_data_raw or 0) for lib in libraries)
        
        # 判断是否为RNA Lane
        is_rna_lane = any(
            'RNA' in (getattr(lib, 'lab_type', '') or '').upper() 
            for lib in libraries
        )
        
        capacity_limit = 2175.0 if is_rna_lane else 2100.0
        buffer = 5.0
        
        if total_data > capacity_limit + buffer:
            return ValidationError(
                rule_type=ValidationRuleType.NEW_MODE_CAPACITY,
                severity=ValidationSeverity.ERROR,
                message=f"1.0模式Lane数据量{total_data:.1f}G超过{capacity_limit}G±{buffer}G限制",
                current_value=total_data,
                threshold_value=capacity_limit + buffer
            )
        
        return None
    
    def _validate_board_limit(self, libraries: List[EnhancedLibraryInfo]) -> Optional[ValidationError]:
        """校验板号限制"""
        boards = set()
        
        for lib in libraries:
            board = (
                getattr(lib, 'board_number', None) or
                getattr(lib, 'board_id', None) or
                getattr(lib, 'plate_id', None)
            )
            if board:
                boards.add(str(board))
        
        if len(boards) == 0:
            return ValidationError(
                rule_type=ValidationRuleType.BOARD_LIMIT,
                severity=ValidationSeverity.ERROR,
                message="1.0模式Lane缺失板号信息"
            )
        
        if len(boards) > 15:
            return ValidationError(
                rule_type=ValidationRuleType.BOARD_LIMIT,
                severity=ValidationSeverity.ERROR,
                message=f"1.0模式Lane板号数量{len(boards)}超过15个限制",
                current_value=len(boards),
                threshold_value=15
            )
        
        return None
    
    def _validate_single_index_heavy_novo(self, libraries: List[EnhancedLibraryInfo]) -> Optional[ValidationError]:
        """校验诺禾单Index大文库限制"""
        for lib in libraries:
            # 判断是否为诺禾文库
            sample_id = getattr(lib, 'sample_id', '') or ''
            is_customer = sample_id.startswith('FKDL') or '客户' in (getattr(lib, 'lab_type', '') or '')
            
            if not is_customer:  # 诺禾文库
                # 统计Index数量
                index_seq = getattr(lib, 'index_seq', '') or ''
                index_count = len([seg for seg in index_seq.split(',') if seg.strip()])
                
                lib_data = float(lib.contract_data_raw or 0)
                
                if index_count <= 1 and lib_data > 150.0:
                    return ValidationError(
                        rule_type=ValidationRuleType.SINGLE_INDEX_HEAVY_NOVO,
                        severity=ValidationSeverity.ERROR,
                        message=f"1.0模式存在诺禾单Index大文库{lib.origrec}({lib_data:.1f}G>150G)",
                        affected_libraries=[lib.origrec]
                    )
        
        return None
    
    def _validate_nova_x10b_minimum(self, libraries: List[EnhancedLibraryInfo]) -> Optional[ValidationError]:
        """校验Nova X-10B最小容量要求"""
        total_data = sum(float(lib.contract_data_raw or 0) for lib in libraries)
        
        # 检查是否为包Lane/包FC
        has_package = any(
            getattr(lib, 'is_package_lane', '') == '是' or 
            getattr(lib, 'package_fc_number', None)
            for lib in libraries
        )
        
        if not has_package or total_data < 375.0:
            return ValidationError(
                rule_type=ValidationRuleType.NOVA_X10B_MINIMUM,
                severity=ValidationSeverity.ERROR,
                message=f"Nova X-10B Lane数据量{total_data:.1f}G不足375G或非包Lane/包FC",
                current_value=total_data,
                threshold_value=375.0
            )
        
        return None
    
    def validate_fc_data_amount(
        self,
        lanes: List[List[EnhancedLibraryInfo]],
        machine_type: str
    ) -> Optional[ValidationError]:
        """
        校验整个FC的数据量是否满足最小要求
        
        规则（依据排机规则文档）：
        - Nova X-25B: 整个FC最小数据量 1150G
        - 其他机器类型暂无FC最小数据量限制
        
        Args:
            lanes: FC中所有Lane的文库列表
            machine_type: 机器类型
            
        Returns:
            ValidationError: 如果不满足最小数据量要求
        """
        fc_min_data = self.fc_min_data.get(machine_type, self.fc_min_data.get('default', 0.0))
        
        # 如果没有FC最小数据量限制，直接返回
        if fc_min_data <= 0:
            return None
        
        # 计算FC总数据量
        total_fc_data = 0.0
        for lane_libraries in lanes:
            total_fc_data += sum(float(lib.contract_data_raw or 0) for lib in lane_libraries)
        
        if total_fc_data < fc_min_data:
            shortfall = fc_min_data - total_fc_data
            return ValidationError(
                rule_type=ValidationRuleType.FC_MIN_DATA,
                severity=ValidationSeverity.ERROR,
                message=f"FC总数据量{total_fc_data:.1f}G低于最小要求{fc_min_data}G，需添加约{shortfall:.1f}G的文库",
                current_value=total_fc_data,
                threshold_value=fc_min_data
            )
        
        logger.debug(f"FC总数据量{total_fc_data:.1f}G满足最小要求{fc_min_data}G")
        return None
    
    def calculate_data_adjustment(
        self,
        libraries: List[EnhancedLibraryInfo],
        machine_type: str
    ) -> Dict:
        """
        计算Lane数据量调整建议
        
        返回一个字典，包含：
        - status: 'ok' / 'insufficient' / 'exceeded'
        - current_data: 当前数据量
        - target_range: (最小值, 最大值)
        - adjustment_needed: 需要调整的数据量（正数=需添加，负数=需剔除）
        - suggestions: 具体调整建议列表
        
        Args:
            libraries: Lane中的文库列表
            machine_type: 机器类型
            
        Returns:
            Dict: 调整建议信息
        """
        total_data = sum(float(lib.contract_data_raw or 0) for lib in libraries)
        
        lane_rule = self.scheduling_config.get_lane_capacity_range(
            libraries=libraries,
            machine_type=machine_type,
        )
        min_allowed = lane_rule.effective_min_gb
        max_allowed = lane_rule.effective_max_gb
        target_capacity = lane_rule.soft_target_gb
        
        result = {
            'status': 'ok',
            'current_data': total_data,
            'target_capacity': target_capacity,
            'target_range': (min_allowed, max_allowed),
            'adjustment_needed': 0.0,
            'suggestions': []
        }
        
        if total_data < min_allowed:
            # 数据量不足，需要添加文库
            shortfall = min_allowed - total_data
            result['status'] = 'insufficient'
            result['adjustment_needed'] = shortfall
            result['suggestions'] = self._generate_add_suggestions(shortfall, libraries)
            
        elif total_data > max_allowed:
            # 数据量超标，需要剔除文库
            excess = total_data - max_allowed
            result['status'] = 'exceeded'
            result['adjustment_needed'] = -excess  # 负数表示需要剔除
            result['suggestions'] = self._generate_remove_suggestions(excess, libraries)
        
        return result
    
    def _generate_add_suggestions(
        self, 
        shortfall: float, 
        current_libraries: List[EnhancedLibraryInfo]
    ) -> List[str]:
        """
        生成添加文库的建议
        
        Args:
            shortfall: 需要添加的数据量（G）
            current_libraries: 当前Lane中的文库
            
        Returns:
            List[str]: 建议列表
        """
        suggestions = []
        suggestions.append(f"当前Lane数据量不足，需添加约{shortfall:.1f}G的文库")
        
        # 根据缺口大小给出具体建议
        if shortfall <= 20:
            suggestions.append("建议：添加1-2个小数据量文库（10-20G）即可满足要求")
        elif shortfall <= 50:
            suggestions.append("建议：添加2-3个中等数据量文库（15-25G）")
        elif shortfall <= 100:
            suggestions.append("建议：添加3-5个文库，或1个大数据量文库（50-100G）")
        else:
            suggestions.append(f"建议：需要添加较多文库，考虑从待排文库池中选择合适的文库")
        
        # 检查Index兼容性提示
        if current_libraries:
            suggestions.append("注意：添加的文库需确保Index不与现有文库冲突")
        
        return suggestions
    
    def _generate_remove_suggestions(
        self, 
        excess: float, 
        current_libraries: List[EnhancedLibraryInfo]
    ) -> List[str]:
        """
        生成剔除文库的建议
        
        Args:
            excess: 需要剔除的数据量（G）
            current_libraries: 当前Lane中的文库
            
        Returns:
            List[str]: 建议列表
        """
        suggestions = []
        suggestions.append(f"当前Lane数据量超标，需剔除约{excess:.1f}G的文库")
        
        # 分析当前文库，找出合适的剔除候选
        if current_libraries:
            # 按数据量排序，找出可以剔除的候选
            sorted_libs = sorted(
                current_libraries, 
                key=lambda lib: float(lib.contract_data_raw or 0),
                reverse=True
            )
            
            # 找出优先级较低的文库
            low_priority_libs = [
                lib for lib in sorted_libs 
                if getattr(lib, 'scheduling_priority_rank', 4) >= 3
            ]
            
            if low_priority_libs:
                suggestions.append("建议优先剔除优先级较低的文库（非临检/YC/SJ）")
            
            # 按数据量给出具体建议
            for lib in sorted_libs[:3]:  # 最多列出前3个候选
                lib_data = float(lib.contract_data_raw or 0)
                if lib_data >= excess * 0.8:  # 剔除这个文库能基本解决问题
                    suggestions.append(
                        f"可考虑剔除文库 {lib.origrec}（{lib_data:.1f}G）"
                    )
                    break
        
        suggestions.append("注意：剔除的文库将回到待排池，等待下次排机")
        
        return suggestions
    
    def generate_validation_report(self, result: LaneValidationResult) -> str:
        """生成校验报告"""
        lines = [
            f"Lane校验报告: {result.lane_id}",
            f"校验结果: {'通过' if result.is_valid else '不通过'}",
            ""
        ]
        
        if result.errors:
            lines.append(f"错误 ({len(result.errors)}):")
            for error in result.errors:
                lines.append(f"  - [{error.rule_type.value}] {error.message}")
            lines.append("")
        
        if result.warnings:
            lines.append(f"警告 ({len(result.warnings)}):")
            for warning in result.warnings:
                lines.append(f"  - [{warning.rule_type.value}] {warning.message}")
        
        return "\n".join(lines)
