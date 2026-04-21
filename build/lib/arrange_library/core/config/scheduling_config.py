"""
排机系统统一配置管理模块
创建时间：2025-12-03 14:00:00
更新时间：2026-04-17 14:30:00

集中管理所有排机相关的配置常量，避免硬编码分散在各个模块中。
配置来源：docs/排机规则文档.md、config/business_rules.yaml
"""

import json
import os
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from pathlib import Path

import yaml
from loguru import logger

# from liblane_paths import setup_liblane_paths
# setup_liblane_paths()


class SchedulingMode(Enum):
    """排机模式"""
    MODE_1_0 = "1.0_mode"           # 1.0模式（手工/常规产线S）
    MODE_3_6T_NEW = "3.6T-NEW"      # 3.6T-NEW模式（自动产线Z/ZS）
    NON_1_0 = "non_1.0_mode"        # 非1.0模式


class PriorityLevel(Enum):
    """优先级等级"""
    CLINICAL = 1        # 临检
    YC = 2              # YC
    S_LEVEL = 3         # S级客户
    URGENT = 4          # 交付<72h
    NORMAL = 5          # 普通


@dataclass
class LaneCapacityConfig:
    """Lane容量配置"""
    # 标准Lane容量（非1.0模式，Nova X-25B PE150策略）
    # 目标975G，tolerance=5G，有效区间[970G, 980G]
    standard_capacity: float = 975.0
    standard_tolerance: float = 5.0
    
    # 1.0模式Lane容量
    mode_1_0_rna_capacity: float = 2175.0       # RNA专用Lane上限
    mode_1_0_mixed_capacity: float = 2100.0     # 混排Lane上限
    mode_1_0_tolerance: float = 5.0
    
    # 包Lane容量
    package_lane_target: float = 1000.0
    package_lane_tolerance: float = 5.0
    package_lane_min: float = 995.0
    package_lane_max: float = 1005.0
    
    # 不同机器类型的容量（fallback值，优先由规则矩阵决定）
    machine_capacities: Dict[str, float] = field(default_factory=lambda: {
        'Nova X-25B': 975.0,        # 目标975G，有效区间[970G, 980G]
        'Nova X-10B': 380.0,
        'Novaseq': 880.0,
        'T7': 1670.0,
        'T7-Methylation': 1580.0,
        'SURFSEQ-5000': 1200.0,
        'SURFSEQ-Q': 750.0,
    })



@dataclass
class ValidationLimitsConfig:
    """成Lane校验阈值配置（统一真值来源，LaneValidator 从此处读取，不再使用类常量）"""

    # 客户文库占比上限（>50% 且 <100% 时违规）
    customer_ratio_limit: float = 0.50

    # 10bp Index占比下限（混排时）
    index_10bp_ratio_min: float = 0.40

    # 单端Index占比上限
    single_end_ratio_limit: float = 0.30

    # 碱基不均衡占比上限（当前调度与严格校验统一按35%口径）
    base_imbalance_ratio_limit: float = 0.35
    # 3.6T-NEW 模式碱基不均衡上限（与常规场景保持一致）
    base_imbalance_ratio_limit_3_6t: float = 0.35

    # 加测文库占比上限及缓冲
    add_test_ratio_limit: float = 0.25
    add_test_buffer_gb: float = 10.0

    # Peak Size 校验参数
    peak_size_max_diff: int = 150           # bp，最大-最小差值
    peak_size_coverage_ratio: float = 0.75  # 150bp 窗口最低覆盖率

    # 特殊文库类型数量上限（1-57规则：组合混排最多5种）
    special_library_type_limit: int = 5

    # 特殊文库总量上限（按机器类型，单位 G）
    special_library_capacity: Dict[str, float] = field(default_factory=lambda: {
        'NovaSeq X Plus': 350.0,
        'Nova X-25B': 350.0,
        'NovaSeq X Plus-25B': 350.0,
        'Nova X-10B': 150.0,
        'NovaSeq X Plus-10B': 150.0,
        'Novaseq-S2': 500.0,
        'Novaseq-S1': 400.0,
        'Novaseq-S4': 400.0,
        'Novaseq-S4 XP': 400.0,
        'Novaseq': 400.0,
        'Novaseq-T7': 175.0,
        'T7': 175.0,
        'default': 350.0,
    })

    # FC 最小数据量（整个 Flow Cell，单位 G；0 表示不限制）
    fc_min_data: Dict[str, float] = field(default_factory=lambda: {
        'Nova X-25B': 1150.0,
        'Nova X-10B': 0.0,
        'Novaseq': 0.0,
        'T7': 0.0,
        'default': 0.0,
    })

    # 碱基不均衡文库识别关键词（与 EnhancedLibraryInfo.is_base_imbalance() 保持一致）
    base_imbalance_keywords: List[str] = field(default_factory=lambda: [
        'ATAC', 'CUT Tag', 'Methylation', 'small RNA', '单细胞', '10X',
        '简化基因组', 'GBS', 'RAD', 'circRNA', '甲基化',
        'rrbs', 'ribo-seq', 'em-seq', '墨卓', 'visium', 'fixed rna', 'mobidrop',
    ])

    # 特殊文库类型识别关键词
    special_library_keywords: List[str] = field(default_factory=lambda: [
        '甲基化', 'Methylation', '10X', 'ATAC', '外显子', 'HI-C',
        'small RNA', 'CUT Tag', '空间转录组',
    ])


@dataclass
class PoolingConfig:
    """Pooling配置"""
    # CV校验阈值
    cv_threshold: float = 1.15  # CV<1.15，最大误差15%，越小越好

    # Lane质量阈值（基于预测产出CV）
    lane_cv_threshold: float = 0.25
    lane_cv_penalty_weight: float = 200.0
    
    # Pooling系数范围
    default_coefficient_min: float = 1.0
    default_coefficient_max: float = 2.0
    
    # 最小下单量
    min_ordered_gb: float = 1.0
    
    # 最大下单/合同比
    max_ordered_ratio: float = 2.0
    
    # 下单量<1G时的系数范围
    small_data_coefficient_min: float = 1.5
    small_data_coefficient_max: float = 2.0
    
    # 下单量>2倍合同量时的系数范围
    large_ratio_coefficient_min: float = 1.3
    large_ratio_coefficient_max: float = 2.0
    
    # 下单量与合同量的最大差值
    max_order_contract_diff: float = 30.0
    
    # CV优化参数
    cv_optimization_adjustment_ratio: float = 0.2      # CV超标时每次调整幅度（20%）
    cv_optimization_deviation_threshold: float = 0.3  # 系数偏离均值阈值，超过才调整
    
    # Lane下单量/合同量最大比例
    lane_order_ratio_limit: float = 1.15
    
    # 上机浓度配置
    concentration_base_imbalance_package: float = 2.5   # 碱基不均衡包Lane
    concentration_scattered_mixed: float = 1.78         # 散样混排
    concentration_clinical_lane: float = 2.3            # 临检Lane
    concentration_default: float = 2.0


@dataclass
class PriorityConfig:
    """优先级配置"""
    # 临检文库前缀
    clinical_prefixes: List[str] = field(default_factory=lambda: [
        'FDYE', 'FDYT', 'FDYG', 'FDYP', 'FDYK', 'FDYX', 'FICR', 'FIPM'
    ])
    
    # 国际临检前缀
    international_clinical_prefixes: Dict[str, str] = field(default_factory=lambda: {
        'UK': 'E',    # 英国
        'US': 'C',    # 美国
    })
    
    # YC文库前缀
    yc_prefixes: List[str] = field(default_factory=lambda: ['FKDL'])
    
    # 紧急交付时间阈值（小时）
    urgent_delivery_hours: int = 72
    
    # 大数据量阈值
    large_data_threshold_gb: float = 70.0
    
    # 优先级权重
    weight_order_time: float = 0.40
    weight_delivery_time: float = 0.35
    weight_data_type: float = 0.25


@dataclass
class Mode1_0Config:
    """1.0模式配置"""
    # 排除关键词
    exclude_keywords: List[str] = field(default_factory=lambda: [
        '药企', '医检所', '医学', 'YC', 'SJ'
    ])
    
    # 排除文库类型
    exclude_library_types: List[str] = field(default_factory=lambda: [
        '外显子文库', 'HI-C文库', '10X HD', '10X ATAC'
    ])
    
    # 板号限制
    max_board_count_per_run: int = 15
    
    # 交付时间阈值（小时）- 与临检/YC同级优先处理
    priority_delivery_hours: int = 72


@dataclass
class GeneticAlgorithmConfig:
    """遗传算法参数配置"""
    # 变异率：控制解的探索能力，越高越容易跳出局部最优
    mutation_rate: float = 0.18
    # 交叉率：控制解的组合能力
    crossover_rate: float = 0.92
    # 最大改进迭代次数
    max_improvement_iterations: int = 150
    # 最大重平衡轮数
    max_rebalance_rounds: int = 3
    # 质量改进最小阈值（低于此值认为收敛）
    quality_improvement_threshold: float = 0.01
    # 质量评分权重
    weight_utilization: float = 0.4     # 利用率权重
    weight_balance: float = 0.3         # 平衡性权重
    weight_priority: float = 0.3        # 优先级权重


@dataclass
class LibrarySplitConfig:
    """文库拆分配置"""
    # 1.1模式（兼容旧名1.0）文库不参与预拆分；以下阈值仅适用于3.6T-NEW模式
    # 3.6T-NEW模式 + 单index：单文库合同量 >100G 触发拆分
    single_index_non_1_0_threshold_gb: float = 100.0
    # 兼容历史配置保留，当前文库拆分逻辑不再使用该阈值
    single_index_mode_1_0_threshold_gb: float = 200.0
    # 3.6T-NEW模式 + 多index：单文库合同量 >300G 触发拆分
    multi_index_threshold_gb: float = 300.0
    # 拆分后最小数据量
    min_split_size_gb: float = 2.0
    # 大数据量文库阈值
    large_data_threshold_gb: float = 70.0


@dataclass
class IndexValidationConfig:
    """Index校验配置"""
    # 模糊碱基比例阈值（超过此比例认为序列质量差）
    ambiguous_base_ratio_limit: float = 0.5
    # Index相似度阈值（高于此值可能冲突）
    similarity_threshold_high: float = 0.9
    similarity_threshold_medium: float = 0.8
    similarity_threshold_low: float = 0.75
    # 序列长度比例（用于计算最大允许重复位）
    sequence_length_ratio: float = 0.6
    # 最小允许重复位数
    min_overlap_positions: int = 7


@dataclass
class ConstraintSolverConfig:
    """约束求解器配置"""
    # 默认Pooling因子
    default_pooling_factor: float = 1.1
    # 重叠比例严重程度阈值
    overlap_critical_ratio: float = 2.0    # 超过允许值2倍以上
    overlap_warning_ratio: float = 1.5     # 超过允许值1.5倍
    overlap_caution_ratio: float = 0.8     # 接近允许值
    # 松弛求解线程数
    relaxed_solver_threads: int = 2


@dataclass
class LaneRuleSelection:
    """统一规则表匹配结果。"""

    rule_code: str
    soft_target_gb: float
    min_target_gb: float
    max_target_gb: float
    tolerance_gb: float
    effective_min_gb: float
    effective_max_gb: float
    lane_count: int
    loading_method: str
    sequencing_mode: str
    fc_min_data_gb: float = 0.0
    additional_balance_ratio: float = 0.0
    soft_single_lane_preferred: bool = False
    profile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """重试配置"""
    # 最大重试次数
    max_retries: int = 3
    
    # 基础重试延迟（秒）
    base_delay: float = 1.0
    
    # 最大重试延迟（秒）
    max_delay: float = 30.0
    
    # 指数退避因子
    exponential_factor: float = 2.0
    
    # 熔断器配置
    circuit_breaker_threshold: int = 5      # 失败次数阈值
    circuit_breaker_timeout: float = 60.0   # 熔断恢复时间（秒）
    circuit_breaker_half_open_max: int = 1  # 半开状态最大尝试次数


@dataclass
class BaseImbalanceConfig:
    """碱基不均衡文库配置"""
    # 碱基不均衡文库分组关键词
    group_keywords: List[str] = field(default_factory=lambda: [
        'ATAC', 'CUT Tag', 'Methylation', 'small RNA', '单细胞', '10X',
        '简化基因组', 'GBS', 'RAD', 'circRNA', '甲基化', 'EM-Seq'
    ])
    
    # 特殊文库关键词
    special_library_keywords: List[str] = field(default_factory=lambda: [
        '甲基化', 'Methylation', '10X', 'ATAC', '外显子', 'HI-C',
        'small RNA', 'CUT Tag', '空间转录组'
    ])
    
    # 不可混排的文库组合
    forbidden_combinations: List[Dict[str, List[str]]] = field(default_factory=lambda: [
        {
            'library': '10x HD Visium空间转录组文库',
            'forbidden_with': ["10X转录组-3'文库", "10X转录组-3'膜蛋白文库", "墨卓转录组-3端文库"]
        },
        {
            'library': '10X ATAC文库',
            'forbidden_with': ['10x_cellranger拆分方式的文库'],
            'restriction': 'not_same_run'
        }
    ])


class SchedulingConfigManager:
    """排机配置管理器（线程安全单例）"""
    
    _instance = None
    _lock = threading.Lock()
    _NORMALIZATION_CACHE_MAX_SIZE = 8192
    _RUNTIME_CACHE_MAX_SIZE = 4096
    _NORMALIZED_TEXT_CACHE: Dict[str, str] = {}
    _NORMALIZED_SEQ_KEYWORD_CACHE: Dict[str, str] = {}
    _CANONICAL_SEQ_MODE_CACHE: Dict[str, str] = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # 双重检查锁定
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            self._initialized = True
        
        # 原始YAML配置存储
        self._raw_config: Dict[str, Any] = {}
        self._rule_matrix_config: Dict[str, Any] = {}
        
        # 初始化各配置模块
        self.lane_capacity = LaneCapacityConfig()
        self.validation_limits = ValidationLimitsConfig()
        self.pooling = PoolingConfig()
        self.priority = PriorityConfig()
        self.mode_1_0 = Mode1_0Config()
        self.retry = RetryConfig()
        self.base_imbalance = BaseImbalanceConfig()
        self.genetic_algorithm = GeneticAlgorithmConfig()
        self.library_split = LibrarySplitConfig()
        self.index_validation = IndexValidationConfig()
        self.constraint_solver = ConstraintSolverConfig()
        
        # 1.1模式专属规则配置
        self._mode_1_1_rules: Dict[str, Any] = {}
        self._seq_strategy_resolution_cache: Dict[Tuple[str, ...], str] = {}
        self._seq_mode_resolution_cache: Dict[Tuple[Any, ...], str] = {}
        self._lane_rule_selection_cache: Dict[Tuple[Any, ...], Optional[LaneRuleSelection]] = {}
        self._lane_context_feature_cache: Dict[
            Tuple[str, ...],
            Tuple[str, Tuple[str, ...], Tuple[Tuple[str, ...], Tuple[str, ...]]],
        ] = {}
        
        # 从统一配置文件加载
        self._load_unified_config()
        self._load_rule_matrix_config()
        self._load_mode_1_1_rules()
        
        # 尝试从旧配置文件加载
        self._load_from_yaml()
        
        logger.info("排机配置管理器初始化完成")
    
    def _load_unified_config(self):
        """从统一配置文件加载配置"""
        config_dir = Path(__file__).parent.parent.parent / 'config'
        unified_config_path = config_dir / 'unified_rule_parameters.yaml'
        
        if not unified_config_path.exists():
            logger.warning(f"统一配置文件不存在: {unified_config_path}")
            return
        
        try:
            with open(unified_config_path, 'r', encoding='utf-8') as f:
                self._raw_config = yaml.safe_load(f) or {}
            
            # AI模型配置（Qwen已移除）：忽略 ai_models 配置段
            
            # 加载重试配置
            retry_config = self._raw_config.get('retry_settings', {})
            if retry_config:
                self.retry.max_retries = retry_config.get('max_retries', self.retry.max_retries)
                self.retry.base_delay = retry_config.get('initial_delay', self.retry.base_delay)
                self.retry.exponential_factor = retry_config.get('backoff_factor', self.retry.exponential_factor)
                self.retry.max_delay = retry_config.get('max_delay', self.retry.max_delay)
                
                cb_config = retry_config.get('circuit_breaker', {})
                if cb_config:
                    self.retry.circuit_breaker_threshold = cb_config.get('failure_threshold', self.retry.circuit_breaker_threshold)
                    self.retry.circuit_breaker_timeout = cb_config.get('recovery_timeout', self.retry.circuit_breaker_timeout)
            
            # 加载优先级规则
            priority_rules = self._raw_config.get('priority_rules', {})
            if priority_rules:
                self.priority.clinical_prefixes = priority_rules.get('clinical_prefixes', self.priority.clinical_prefixes)
                self.priority.yc_prefixes = priority_rules.get('yc_prefixes', self.priority.yc_prefixes)
                self.priority.urgent_delivery_hours = priority_rules.get('urgent_threshold_hours', self.priority.urgent_delivery_hours)
            
            # 加载Lane容量配置
            lane_capacities = self._raw_config.get('lane_capacities', {})
            if lane_capacities:
                if 'non_1_0_mode' in lane_capacities:
                    self.lane_capacity.standard_capacity = lane_capacities['non_1_0_mode'].get('max', self.lane_capacity.standard_capacity)
                if '1_0_mode_rna' in lane_capacities:
                    self.lane_capacity.mode_1_0_rna_capacity = lane_capacities['1_0_mode_rna'].get('max', self.lane_capacity.mode_1_0_rna_capacity)
                if '1_0_mode_mixed' in lane_capacities:
                    self.lane_capacity.mode_1_0_mixed_capacity = lane_capacities['1_0_mode_mixed'].get('max', self.lane_capacity.mode_1_0_mixed_capacity)
                if 'package_lane' in lane_capacities:
                    pkg = lane_capacities['package_lane']
                    self.lane_capacity.package_lane_min = pkg.get('min', self.lane_capacity.package_lane_min)
                    self.lane_capacity.package_lane_max = pkg.get('max', self.lane_capacity.package_lane_max)
                    self.lane_capacity.package_lane_target = pkg.get('target', self.lane_capacity.package_lane_target)
            
            # 加载1.0模式规则
            mode_1_0_rules = self._raw_config.get('mode_1_0_rules', {})
            if mode_1_0_rules:
                self.mode_1_0.exclude_keywords = mode_1_0_rules.get('exclude_keywords', self.mode_1_0.exclude_keywords)
                self.mode_1_0.exclude_library_types = mode_1_0_rules.get('exclude_types', self.mode_1_0.exclude_library_types)
            
            # 加载碱基不均衡关键词
            base_imbalance_kw = self._raw_config.get('base_imbalance_keywords', {})
            if base_imbalance_kw:
                primary = base_imbalance_kw.get('primary', [])
                additional = base_imbalance_kw.get('additional', [])
                if primary or additional:
                    self.base_imbalance.group_keywords = primary + additional
            
            # 加载Pooling规则
            pooling_rules = self._raw_config.get('pooling_rules', {})
            if pooling_rules:
                coef_range = pooling_rules.get('default_coefficient_range', {})
                if coef_range:
                    self.pooling.default_coefficient_min = coef_range.get('min', self.pooling.default_coefficient_min)
                    self.pooling.default_coefficient_max = coef_range.get('max', self.pooling.default_coefficient_max)
                self.pooling.lane_cv_threshold = pooling_rules.get(
                    'lane_cv_threshold', self.pooling.lane_cv_threshold
                )
                self.pooling.lane_cv_penalty_weight = pooling_rules.get(
                    'lane_cv_penalty_weight', self.pooling.lane_cv_penalty_weight
                )
            
            # 加载客户比例配置
            customer_ratio = self._raw_config.get('customer_ratio', {})
            if customer_ratio:
                self.validation_limits.customer_ratio_limit = customer_ratio.get('max_ratio', self.validation_limits.customer_ratio_limit)
            
            logger.debug("统一配置文件已加载")
            
        except Exception as e:
            logger.warning(f"加载统一配置文件失败: {e}")

    @classmethod
    def _store_bounded_cache(
        cls,
        cache: Dict[Any, Any],
        key: Any,
        value: Any,
        *,
        max_size: int,
    ) -> None:
        """写入有上限的运行时缓存，超过上限时整体清空。"""
        if len(cache) >= max_size:
            cache.clear()
        cache[key] = value

    def _clear_runtime_caches(self) -> None:
        """清理排机过程中的运行时缓存。"""
        self._seq_strategy_resolution_cache.clear()
        self._seq_mode_resolution_cache.clear()
        self._lane_rule_selection_cache.clear()
        self._lane_context_feature_cache.clear()
        type(self)._NORMALIZED_TEXT_CACHE.clear()
        type(self)._NORMALIZED_SEQ_KEYWORD_CACHE.clear()
        type(self)._CANONICAL_SEQ_MODE_CACHE.clear()

    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        """统一文本匹配口径。"""
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        cached = cls._NORMALIZED_TEXT_CACHE.get(text)
        if cached is not None:
            return cached
        normalized = (
            text.replace("’", "'")
            .replace("‘", "'")
            .replace("＇", "'")
            .replace("＋", "+")
            .replace("×", "X")
            .upper()
        )
        cls._store_bounded_cache(
            cls._NORMALIZED_TEXT_CACHE,
            text,
            normalized,
            max_size=cls._NORMALIZATION_CACHE_MAX_SIZE,
        )
        return normalized

    @classmethod
    def _normalize_seq_keyword(cls, value: Any) -> str:
        """统一测序策略/模式关键字。"""
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        cached = cls._NORMALIZED_SEQ_KEYWORD_CACHE.get(text)
        if cached is not None:
            return cached
        normalized = cls._normalize_text(text).replace("BP", "").replace(" ", "")
        cls._store_bounded_cache(
            cls._NORMALIZED_SEQ_KEYWORD_CACHE,
            text,
            normalized,
            max_size=cls._NORMALIZATION_CACHE_MAX_SIZE,
        )
        return normalized

    def _build_normalized_text_signature(self, values: List[Any]) -> Tuple[str, ...]:
        """构建去空后的标准化文本签名。"""
        signature: List[str] = []
        for value in values:
            normalized = self._normalize_text(value)
            if normalized:
                signature.append(normalized)
        return tuple(signature)

    def _build_normalized_keyword_signature(self, values: List[Any]) -> Tuple[str, ...]:
        """构建去空后的标准化关键字签名。"""
        signature: List[str] = []
        for value in values:
            normalized = self._normalize_seq_keyword(value)
            if normalized:
                signature.append(normalized)
        return tuple(signature)

    @classmethod
    def _canonicalize_seq_mode_value(cls, value: Any) -> str:
        """将单个测序模式值规整到规则矩阵使用的统一口径。"""
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        cached = cls._CANONICAL_SEQ_MODE_CACHE.get(text)
        if cached is not None:
            return cached
        normalized_text = cls._normalize_text(text)
        normalized_keyword = cls._normalize_seq_keyword(text)
        if "LANESEQ" in normalized_keyword:
            canonical = cls._normalize_text("Lane seq")
        elif "3.6T" in normalized_keyword:
            canonical = cls._normalize_text("3.6T-NEW")
        elif normalized_keyword in {"1.0", "1.1", "1.0MODE", "1.1MODE"}:
            canonical = cls._normalize_text("1.1")
        else:
            canonical = normalized_text
        cls._store_bounded_cache(
            cls._CANONICAL_SEQ_MODE_CACHE,
            text,
            canonical,
            max_size=cls._NORMALIZATION_CACHE_MAX_SIZE,
        )
        return canonical

    def _build_canonical_seq_mode_signature(self, values: List[Any]) -> Tuple[str, ...]:
        """构建测序模式候选的统一签名。"""
        signature: List[str] = []
        for value in values:
            canonical = self._canonicalize_seq_mode_value(value)
            if canonical:
                signature.append(canonical)
        return tuple(signature)

    def _get_library_cached_signature(
        self,
        lib: Any,
        *,
        raw_attrs: Tuple[str, ...],
        cache_attr: str,
        builder,
    ) -> Tuple[str, ...]:
        """按文库缓存常用签名，避免在选Lane时重复标准化同一批字段。"""
        raw_values = tuple(getattr(lib, attr_name, None) for attr_name in raw_attrs)
        cached = getattr(lib, cache_attr, None)
        if cached is not None and cached[0] == raw_values:
            return cached[1]
        signature = builder(list(raw_values))
        setattr(lib, cache_attr, (raw_values, signature))
        return signature

    def _load_rule_matrix_config(self) -> None:
        """加载统一排机规则总配置。"""
        config_dir = Path(__file__).parent.parent.parent / 'config'
        rule_matrix_path = config_dir / 'scheduling_rule_matrix.json'
        if not rule_matrix_path.exists():
            logger.warning(f"统一排机规则配置不存在: {rule_matrix_path}")
            return

        try:
            with open(rule_matrix_path, 'r', encoding='utf-8') as file:
                raw_config = json.load(file)
        except Exception as exc:
            logger.warning(f"加载统一排机规则配置失败: {exc}")
            return

        sample_type_groups: Dict[str, Dict[str, Any]] = {}
        for group_code, group_data in (raw_config.get('sample_type_groups') or {}).items():
            sample_types = {
                self._normalize_text(item)
                for item in group_data.get('sample_types', [])
                if self._normalize_text(item)
            }
            sample_type_groups[group_code] = {
                **group_data,
                'sample_types': sample_types,
            }

        def _normalize_scope_list(values: List[Any]) -> Set[str]:
            return {
                self._normalize_text(item)
                for item in values
                if self._normalize_text(item)
            }

        lane_rule_profiles: List[Dict[str, Any]] = []
        for profile in raw_config.get('lane_rule_profiles', []):
            normalized_profile = dict(profile)
            normalized_profile['machine_types'] = _normalize_scope_list(profile.get('machine_types', []))
            normalized_profile['test_nos'] = _normalize_scope_list(profile.get('test_nos', []))
            normalized_profile['project_types'] = _normalize_scope_list(profile.get('project_types', []))
            normalized_profile['seq_modes'] = _normalize_scope_list(profile.get('seq_modes', []))
            normalized_profile['seq_strategy_keywords_any'] = {
                self._normalize_seq_keyword(item)
                for item in profile.get('seq_strategy_keywords_any', [])
                if self._normalize_seq_keyword(item)
            }
            normalized_profile['seq_strategy_keywords_exclude'] = {
                self._normalize_seq_keyword(item)
                for item in profile.get('seq_strategy_keywords_exclude', [])
                if self._normalize_seq_keyword(item)
            }
            normalized_profile['process_codes'] = {
                int(item) for item in profile.get('process_codes', []) if item is not None
            }
            normalized_profile['sample_type_group_codes_any'] = list(
                profile.get('sample_type_group_codes_any', [])
            )
            normalized_profile['customer_complex_results_any'] = _normalize_scope_list(
                profile.get('customer_complex_results_any', [])
            )
            normalized_profile['internal_risk_build_flags_any'] = _normalize_scope_list(
                profile.get('internal_risk_build_flags_any', [])
            )
            soft_single_lane_preferred = profile.get('soft_single_lane_preferred', False)
            if isinstance(soft_single_lane_preferred, str):
                normalized_profile['soft_single_lane_preferred'] = soft_single_lane_preferred.strip().lower() in {
                    '1', 'true', 'yes', 'y', '是'
                }
            else:
                normalized_profile['soft_single_lane_preferred'] = bool(soft_single_lane_preferred)
            normalized_profile['priority'] = int(profile.get('priority', 0))
            normalized_profile['soft_target_gb'] = float(profile.get('soft_target_gb', 0.0) or 0.0)
            normalized_profile['min_target_gb'] = float(profile.get('min_target_gb', 0.0) or 0.0)
            normalized_profile['max_target_gb'] = float(profile.get('max_target_gb', 0.0) or 0.0)
            normalized_profile['tolerance_gb'] = float(profile.get('tolerance_gb', 0.0) or 0.0)
            normalized_profile['lane_count'] = int(profile.get('lane_count', 0) or 0)
            normalized_profile['fc_min_data_gb'] = float(profile.get('fc_min_data_gb', 0.0) or 0.0)
            normalized_profile['additional_balance_ratio'] = float(
                profile.get('additional_balance_ratio', 0.0) or 0.0
            )
            lane_rule_profiles.append(normalized_profile)

        lane_rule_profiles.sort(key=lambda item: item['priority'], reverse=True)

        lane_constraints: List[Dict[str, Any]] = []
        for constraint in raw_config.get('lane_constraints', []):
            normalized_constraint = dict(constraint)
            normalized_constraint['machine_types'] = _normalize_scope_list(constraint.get('machine_types', []))
            normalized_constraint['test_nos'] = _normalize_scope_list(constraint.get('test_nos', []))
            normalized_constraint['project_types'] = _normalize_scope_list(constraint.get('project_types', []))
            normalized_constraint['seq_modes'] = _normalize_scope_list(constraint.get('seq_modes', []))
            normalized_constraint['process_codes'] = {
                int(item) for item in constraint.get('process_codes', []) if item is not None
            }
            normalized_constraint['priority'] = int(constraint.get('priority', 0))
            lane_constraints.append(normalized_constraint)
        lane_constraints.sort(key=lambda item: item['priority'], reverse=True)

        loading_rules: List[Dict[str, Any]] = []
        for rule in raw_config.get('loading_concentration_rules', []):
            normalized_rule = dict(rule)
            normalized_rule['machine_types'] = _normalize_scope_list(rule.get('machine_types', []))
            normalized_rule['test_nos'] = _normalize_scope_list(rule.get('test_nos', []))
            normalized_rule['seq_modes'] = _normalize_scope_list(rule.get('seq_modes', []))
            normalized_rule['process_codes'] = {
                int(item) for item in rule.get('process_codes', []) if item is not None
            }
            normalized_rule['priority'] = int(rule.get('priority', 0))
            normalized_rule['concentration'] = float(rule.get('concentration', 0.0) or 0.0)
            normalized_rule['seq_strategy_keyword'] = self._normalize_seq_keyword(
                rule.get('seq_strategy_keyword', '')
            )
            loading_rules.append(normalized_rule)
        loading_rules.sort(key=lambda item: item['priority'], reverse=True)

        self._rule_matrix_config = {
            'sample_type_groups': sample_type_groups,
            'lane_rule_profiles': lane_rule_profiles,
            'lane_constraints': lane_constraints,
            'loading_concentration_rules': loading_rules,
        }
        logger.info(
            "统一排机规则配置已加载: profiles={}, constraints={}, loading_rules={}".format(
                len(lane_rule_profiles),
                len(lane_constraints),
                len(loading_rules),
            )
        )

    def _get_library_sample_type(self, lib: Any) -> str:
        """获取文库类型标准值。"""
        signature = self._get_library_cached_signature(
            lib,
            raw_attrs=('sample_type_code', 'sampletype'),
            cache_attr='_scheduling_sample_type_signature_cache',
            builder=self._build_normalized_text_signature,
        )
        return signature[0] if signature else ""

    def _get_library_project_data_type(self, lib: Any) -> str:
        """获取项目类型判定所需的标准化 data_type。"""
        signature = self._get_library_cached_signature(
            lib,
            raw_attrs=('data_type',),
            cache_attr='_scheduling_project_data_type_signature_cache',
            builder=self._build_normalized_text_signature,
        )
        return signature[0] if signature else ""

    def _get_lane_sample_types(self, libraries: List[Any]) -> Set[str]:
        """获取Lane内文库类型集合。"""
        return {
            sample_type
            for sample_type in (self._get_library_sample_type(lib) for lib in libraries)
            if sample_type
        }

    def _is_customer_library(self, lib: Any) -> bool:
        """判断文库是否为客户文库。"""
        raw_values = (
            getattr(lib, 'customer_library', None),
            getattr(lib, 'sample_type_code', None),
            getattr(lib, 'sampletype', None),
            getattr(lib, 'sample_id', None),
            getattr(lib, 'lab_type', None),
        )
        cached = getattr(lib, '_scheduling_customer_library_cache', None)
        if cached is not None and cached[0] == raw_values:
            return cached[1]

        checker = getattr(lib, 'is_customer_library', None)
        if callable(checker):
            try:
                resolved = bool(checker())
                setattr(lib, '_scheduling_customer_library_cache', (raw_values, resolved))
                return resolved
            except Exception:
                pass

        customer_flag = self._normalize_text(getattr(lib, 'customer_library', '') or '')
        if customer_flag in {
            self._normalize_text('是'),
            self._normalize_text('客户'),
            'Y',
            'YES',
            'TRUE',
            '1',
        }:
            setattr(lib, '_scheduling_customer_library_cache', (raw_values, True))
            return True
        if customer_flag in {
            self._normalize_text('否'),
            'N',
            'NO',
            'FALSE',
            '0',
        }:
            setattr(lib, '_scheduling_customer_library_cache', (raw_values, False))
            return False
        lab_type_text = self._normalize_text(getattr(lib, 'lab_type', '') or '')
        resolved = self._normalize_text('客户') in lab_type_text
        setattr(lib, '_scheduling_customer_library_cache', (raw_values, resolved))
        return resolved

    def _get_library_quality_risk_signature(self, lib: Any) -> Tuple[bool, str, str]:
        """获取规则选择所需的单库质量风险签名。"""
        raw_values = (
            getattr(lib, 'customer_library', None),
            getattr(lib, 'sample_type_code', None),
            getattr(lib, 'sampletype', None),
            getattr(lib, 'sample_id', None),
            getattr(lib, 'lab_type', None),
            getattr(lib, 'complex_result', None),
            getattr(lib, 'wkcomplexresult', None),
            getattr(lib, 'risk_build_flag', None),
            getattr(lib, 'wkriskbuildflag', None),
        )
        cached = getattr(lib, '_scheduling_quality_risk_signature_cache', None)
        if cached is not None and cached[0] == raw_values:
            return cached[1]

        signature = (
            self._is_customer_library(lib),
            self._normalize_text(
                getattr(lib, 'complex_result', None) or getattr(lib, 'wkcomplexresult', None) or ''
            ),
            self._normalize_text(
                getattr(lib, 'risk_build_flag', None) or getattr(lib, 'wkriskbuildflag', None) or ''
            ),
        )
        setattr(lib, '_scheduling_quality_risk_signature_cache', (raw_values, signature))
        return signature

    def _matches_lane_quality_risk_scope(self, libraries: List[Any], profile: Dict[str, Any]) -> bool:
        """评估规则中客户库检/诺禾风险建库组合条件。"""
        customer_complex_targets = set(profile.get('customer_complex_results_any', set()) or set())
        internal_risk_targets = set(profile.get('internal_risk_build_flags_any', set()) or set())

        if not customer_complex_targets and not internal_risk_targets:
            return True

        has_customer_complex_match = not customer_complex_targets
        has_internal_risk_match = not internal_risk_targets

        for lib in libraries:
            is_customer = self._is_customer_library(lib)
            if not has_customer_complex_match and is_customer:
                complex_result = self._normalize_text(
                    getattr(lib, 'complex_result', None) or getattr(lib, 'wkcomplexresult', None) or ''
                )
                if complex_result in customer_complex_targets:
                    has_customer_complex_match = True

            if not has_internal_risk_match and not is_customer:
                risk_build_flag = self._normalize_text(
                    getattr(lib, 'risk_build_flag', None) or getattr(lib, 'wkriskbuildflag', None) or ''
                )
                if risk_build_flag in internal_risk_targets:
                    has_internal_risk_match = True

            if has_customer_complex_match and has_internal_risk_match:
                return True

        return has_customer_complex_match and has_internal_risk_match

    def _classify_lane_project_type(self, libraries: List[Any]) -> str:
        """按wkdatatype(data_type)识别Lane项目类型，不再按样本编号推断。"""
        has_clinical = False
        has_yc = False
        for lib in libraries:
            data_type = self._get_library_project_data_type(lib)
            if data_type == '临检':
                has_clinical = True
            if data_type == 'YC':
                has_yc = True
        if has_clinical:
            return self._normalize_text('临检')
        if has_yc:
            return self._normalize_text('YC')
        return self._normalize_text('其他')

    def _resolve_process_code(self, libraries: List[Any], metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """解析Lane工序编码。"""
        metadata = metadata or {}
        inferred = self._infer_process_code_from_test_no(
            self._resolve_test_no(libraries, metadata)
        )
        if inferred is not None:
            return inferred
        for key in ('process_code', 'test_code'):
            parsed = self._parse_valid_process_code(metadata.get(key))
            if parsed is not None:
                return parsed
        for lib in libraries:
            for attr_name in ('process_code', 'test_code'):
                parsed = self._parse_valid_process_code(getattr(lib, attr_name, None))
                if parsed is not None:
                    return parsed
        return None

    @staticmethod
    def _parse_valid_process_code(value: Any) -> Optional[int]:
        """解析有效工序码（>0）；0、空值、非法值视为缺失。"""
        if value is None:
            return None
        text = str(value).strip()
        if text == '' or text.lower() in {'none', 'nan', 'null'}:
            return None
        try:
            parsed = int(float(text))
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _infer_process_code_from_test_no(self, normalized_test_no: str) -> Optional[int]:
        """按规则矩阵中 test_no -> process_code 映射反推工序码。仅在唯一映射时返回。"""
        if not normalized_test_no:
            return None
        candidates: Set[int] = set()
        for profile in self._rule_matrix_config.get('lane_rule_profiles', []):
            test_nos = set(profile.get('test_nos', set()) or set())
            if normalized_test_no in test_nos:
                candidates.update(set(profile.get('process_codes', set()) or set()))
        if len(candidates) == 1:
            return next(iter(candidates))
        return None

    def _resolve_test_no(self, libraries: List[Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """解析Lane工序文本。"""
        metadata = metadata or {}
        if metadata.get('test_no'):
            return self._normalize_text(metadata.get('test_no'))
        for lib in libraries:
            value = getattr(lib, 'test_no', None)
            if value is not None and str(value).strip() != '':
                return self._normalize_text(value)
        return ""

    def _resolve_seq_mode(self, libraries: List[Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """解析Lane测序模式。"""
        metadata = metadata or {}
        def _resolve_from_canonical_candidates(values: Tuple[str, ...]) -> str:
            if not values:
                return ""
            unique_candidates = list(dict.fromkeys(values))
            if len(unique_candidates) == 1:
                return unique_candidates[0]
            if self._normalize_text("Lane seq") in unique_candidates:
                return self._normalize_text("Lane seq")
            if (
                self._normalize_text("3.6T-NEW") in unique_candidates
                and self._normalize_text("1.1") not in unique_candidates
            ):
                return self._normalize_text("3.6T-NEW")
            if (
                self._normalize_text("1.1") in unique_candidates
                and self._normalize_text("3.6T-NEW") not in unique_candidates
            ):
                return self._normalize_text("1.1")
            return ""

        metadata_candidates = [metadata.get(key) for key in ('seq_mode', 'lcxms', 'sequencing_mode')]
        metadata_mode_signature = self._build_canonical_seq_mode_signature(metadata_candidates)
        resolved_metadata_mode = _resolve_from_canonical_candidates(metadata_mode_signature)
        if resolved_metadata_mode:
            return resolved_metadata_mode

        current_mode_signature = tuple(
            candidate
            for lib in libraries
            for candidate in self._get_library_cached_signature(
                lib,
                raw_attrs=('seq_mode', 'lcxms', '_current_seq_mode_raw'),
                cache_attr='_scheduling_seq_mode_signature_cache',
                builder=self._build_canonical_seq_mode_signature,
            )
        )

        seq_mode_cache_key = (
            metadata_mode_signature,
            current_mode_signature,
            self._resolve_process_code(libraries, metadata),
            self._resolve_seq_strategy(libraries, metadata),
        )
        cached_seq_mode = self._seq_mode_resolution_cache.get(seq_mode_cache_key)
        if cached_seq_mode is not None:
            return cached_seq_mode

        resolved_current_mode = _resolve_from_canonical_candidates(current_mode_signature)
        if resolved_current_mode:
            self._store_bounded_cache(
                self._seq_mode_resolution_cache,
                seq_mode_cache_key,
                resolved_current_mode,
                max_size=self._RUNTIME_CACHE_MAX_SIZE,
            )
            return resolved_current_mode

        process_code = self._resolve_process_code(libraries, metadata)
        seq_strategy = self._resolve_seq_strategy(libraries, metadata)
        if process_code == 1595 and seq_strategy == '10+24':
            resolved_mode = self._normalize_text('Lane seq')
            self._store_bounded_cache(
                self._seq_mode_resolution_cache,
                seq_mode_cache_key,
                resolved_mode,
                max_size=self._RUNTIME_CACHE_MAX_SIZE,
            )
            return resolved_mode
        if process_code == 1595:
            resolved_mode = self._normalize_text('3.6T-NEW')
            self._store_bounded_cache(
                self._seq_mode_resolution_cache,
                seq_mode_cache_key,
                resolved_mode,
                max_size=self._RUNTIME_CACHE_MAX_SIZE,
            )
            return resolved_mode
        self._store_bounded_cache(
            self._seq_mode_resolution_cache,
            seq_mode_cache_key,
            "",
            max_size=self._RUNTIME_CACHE_MAX_SIZE,
        )
        return ""

    def _collect_seq_strategy_candidate_values(
        self,
        libraries: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """收集解析测序策略所需的候选值。"""
        metadata = metadata or {}
        candidate_values: List[Any] = []
        for key in ('seq_strategy', 'seq_scheme', 'test_no', 'run_cycle'):
            if key in metadata:
                candidate_values.append(metadata.get(key))
        for lib in libraries:
            candidate_values.extend(
                [
                    getattr(lib, 'seq_strategy', None),
                    getattr(lib, 'seq_scheme', None),
                    getattr(lib, 'test_no', None),
                    getattr(lib, 'machine_note', None),
                    getattr(lib, '_lane_sj_mode_raw', None),
                ]
            )
        return candidate_values

    def _resolve_seq_strategy(self, libraries: List[Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """解析Lane测序策略。"""
        metadata = metadata or {}
        metadata_signature = self._build_normalized_keyword_signature(
            [metadata.get(key) for key in ('seq_strategy', 'seq_scheme', 'test_no', 'run_cycle') if key in metadata]
        )
        library_signature = tuple(
            candidate
            for lib in libraries
            for candidate in self._get_library_cached_signature(
                lib,
                raw_attrs=('seq_strategy', 'seq_scheme', 'test_no', 'machine_note', '_lane_sj_mode_raw'),
                cache_attr='_scheduling_seq_strategy_signature_cache',
                builder=self._build_normalized_keyword_signature,
            )
        )
        cache_key = metadata_signature + library_signature
        cached_seq_strategy = self._seq_strategy_resolution_cache.get(cache_key)
        if cached_seq_strategy is not None:
            return cached_seq_strategy
        normalized_candidates = list(cache_key)
        for candidate in normalized_candidates:
            if '10+24' in candidate:
                self._store_bounded_cache(
                    self._seq_strategy_resolution_cache,
                    cache_key,
                    '10+24',
                    max_size=self._RUNTIME_CACHE_MAX_SIZE,
                )
                return '10+24'
        for candidate in normalized_candidates:
            if 'PE150' in candidate:
                self._store_bounded_cache(
                    self._seq_strategy_resolution_cache,
                    cache_key,
                    'PE150',
                    max_size=self._RUNTIME_CACHE_MAX_SIZE,
                )
                return 'PE150'
        resolved_strategy = normalized_candidates[0] if normalized_candidates else ""
        self._store_bounded_cache(
            self._seq_strategy_resolution_cache,
            cache_key,
            resolved_strategy,
            max_size=self._RUNTIME_CACHE_MAX_SIZE,
        )
        return resolved_strategy

    def _match_scope(self, normalized_value: str, allowed_values: Set[str]) -> bool:
        """判断标准化单值是否命中作用域。"""
        if not allowed_values:
            return True
        return normalized_value in allowed_values

    def _match_process_scope(self, process_code: Optional[int], allowed_process_codes: Set[int]) -> bool:
        """判断工序编码是否命中作用域。"""
        if not allowed_process_codes:
            return True
        return process_code is not None and process_code in allowed_process_codes

    def _lane_matches_sample_type_group(self, lane_sample_types: Set[str], group_code: str) -> bool:
        """判断Lane文库类型是否命中某个文库组合组。"""
        if not lane_sample_types:
            return False
        group_data = (self._rule_matrix_config.get('sample_type_groups') or {}).get(group_code, {})
        sample_types = set(group_data.get('sample_types', set()) or set())
        return bool(sample_types) and set(lane_sample_types).issubset(sample_types)

    def _build_lane_quality_risk_signature(
        self,
        libraries: List[Any],
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """抽取规则选择所需的客户复杂度/内部风险签名。"""
        customer_complex_results: Set[str] = set()
        internal_risk_build_flags: Set[str] = set()
        for lib in libraries:
            is_customer, complex_result, risk_build_flag = self._get_library_quality_risk_signature(lib)
            if is_customer:
                if complex_result:
                    customer_complex_results.add(complex_result)
                continue
            if risk_build_flag:
                internal_risk_build_flags.add(risk_build_flag)
        return tuple(sorted(customer_complex_results)), tuple(sorted(internal_risk_build_flags))

    def _get_library_runtime_identity(self, lib: Any) -> str:
        """获取当前进程内用于Lane特征缓存的稳定文库键。"""
        for attr_name in ('_detail_output_key', 'fragment_id', 'wkaidbid', 'aidbid', '_origrec_key', 'origrec', 'sample_id'):
            value = getattr(lib, attr_name, None)
            if value is not None and str(value).strip():
                return str(value).strip()
        return str(id(lib))

    def _resolve_lane_context_features(
        self,
        libraries: List[Any],
    ) -> Tuple[str, Tuple[str, ...], Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        """解析规则选择所需的Lane上下文特征，并按等价文库集合缓存。"""
        cache_key = tuple(sorted(self._get_library_runtime_identity(lib) for lib in libraries))
        cached = self._lane_context_feature_cache.get(cache_key)
        if cached is not None:
            return cached

        context_features = (
            self._classify_lane_project_type(libraries),
            tuple(sorted(self._get_lane_sample_types(libraries))),
            self._build_lane_quality_risk_signature(libraries),
        )
        self._store_bounded_cache(
            self._lane_context_feature_cache,
            cache_key,
            context_features,
            max_size=self._RUNTIME_CACHE_MAX_SIZE,
        )
        return context_features

    def resolve_lane_rule_selection(
        self,
        libraries: List[Any],
        machine_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[LaneRuleSelection]:
        """按统一规则表解析Lane容量配置。"""
        if not libraries:
            return None

        metadata = metadata or {}
        normalized_machine_type = self._normalize_text(machine_type)
        process_code = self._resolve_process_code(libraries, metadata)
        test_no = self._resolve_test_no(libraries, metadata)
        seq_mode = self._resolve_seq_mode(libraries, metadata)
        seq_strategy = self._resolve_seq_strategy(libraries, metadata)
        project_type, lane_sample_types, quality_risk_signature = self._resolve_lane_context_features(libraries)
        cache_key = (
            normalized_machine_type,
            process_code,
            test_no,
            seq_mode,
            seq_strategy,
            project_type,
            lane_sample_types,
            quality_risk_signature,
        )
        if cache_key in self._lane_rule_selection_cache:
            cached_selection = self._lane_rule_selection_cache[cache_key]
            return deepcopy(cached_selection) if cached_selection is not None else None

        for profile in self._rule_matrix_config.get('lane_rule_profiles', []):
            if not self._match_scope(normalized_machine_type, set(profile.get('machine_types', set()) or set())):
                continue
            if not self._match_process_scope(process_code, set(profile.get('process_codes', set()) or set())):
                continue
            if not self._match_scope(test_no, set(profile.get('test_nos', set()) or set())):
                continue
            if not self._match_scope(project_type, set(profile.get('project_types', set()) or set())):
                continue
            if not self._match_scope(seq_mode, set(profile.get('seq_modes', set()) or set())):
                continue

            include_keywords = set(profile.get('seq_strategy_keywords_any', set()) or set())
            if include_keywords and seq_strategy not in include_keywords:
                continue
            exclude_keywords = set(profile.get('seq_strategy_keywords_exclude', set()) or set())
            if exclude_keywords and seq_strategy in exclude_keywords:
                continue

            group_codes = list(profile.get('sample_type_group_codes_any', []) or [])
            if group_codes and not any(
                self._lane_matches_sample_type_group(lane_sample_types, group_code)
                for group_code in group_codes
            ):
                continue
            if not self._matches_lane_quality_risk_scope(libraries, profile):
                continue

            min_target_gb = float(profile.get('min_target_gb', 0.0) or 0.0)
            max_target_gb = float(profile.get('max_target_gb', 0.0) or 0.0)
            tolerance_gb = float(profile.get('tolerance_gb', 0.0) or 0.0)
            selection = LaneRuleSelection(
                rule_code=str(profile.get('rule_code', 'unknown_rule')),
                soft_target_gb=float(profile.get('soft_target_gb', 0.0) or 0.0),
                min_target_gb=min_target_gb,
                max_target_gb=max_target_gb,
                tolerance_gb=tolerance_gb,
                effective_min_gb=min_target_gb - tolerance_gb,
                effective_max_gb=max_target_gb + tolerance_gb,
                lane_count=int(profile.get('lane_count', 0) or 0),
                loading_method=str(profile.get('loading_method', '') or ''),
                sequencing_mode=str(profile.get('sequencing_mode', '') or ''),
                fc_min_data_gb=float(profile.get('fc_min_data_gb', 0.0) or 0.0),
                additional_balance_ratio=float(
                    profile.get('additional_balance_ratio', 0.0) or 0.0
                ),
                soft_single_lane_preferred=bool(profile.get('soft_single_lane_preferred', False)),
                profile=profile,
            )
            self._store_bounded_cache(
                self._lane_rule_selection_cache,
                cache_key,
                deepcopy(selection),
                max_size=self._RUNTIME_CACHE_MAX_SIZE,
            )
            return selection

        self._store_bounded_cache(
            self._lane_rule_selection_cache,
            cache_key,
            None,
            max_size=self._RUNTIME_CACHE_MAX_SIZE,
        )
        return None

    def get_lane_capacity_range(
        self,
        libraries: List[Any],
        machine_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LaneRuleSelection:
        """获取Lane容量区间，优先使用统一规则表，失败时回退到旧配置。"""
        metadata = metadata or {}
        normalized_machine_type = self._normalize_text(machine_type)

        if self._has_package_lane_context(libraries, metadata):
            loading_method = ''
            if '25B' in normalized_machine_type or normalized_machine_type == self._normalize_text('NovaSeq X Plus'):
                loading_method = '25B'
            return LaneRuleSelection(
                rule_code="package_lane_capacity",
                soft_target_gb=float(self.lane_capacity.package_lane_target),
                min_target_gb=float(self.lane_capacity.package_lane_min),
                max_target_gb=float(self.lane_capacity.package_lane_max),
                tolerance_gb=0.0,
                effective_min_gb=float(self.lane_capacity.package_lane_min),
                effective_max_gb=float(self.lane_capacity.package_lane_max),
                lane_count=8,
                loading_method=loading_method,
                sequencing_mode=self._resolve_seq_mode(libraries, metadata),
                fc_min_data_gb=0.0,
                profile={},
            )

        selection = self.resolve_lane_rule_selection(libraries, machine_type, metadata)
        if selection is not None:
            return selection

        target_capacity = self.get_lane_capacity(machine_type, SchedulingMode.NON_1_0)
        tolerance_gb = self.lane_capacity.standard_tolerance
        fallback_loading_method = ''
        if '25B' in normalized_machine_type or normalized_machine_type == self._normalize_text('NovaSeq X Plus'):
            fallback_loading_method = '25B'

        return LaneRuleSelection(
            rule_code="fallback_machine_capacity",
            soft_target_gb=float(target_capacity),
            min_target_gb=float(target_capacity),
            max_target_gb=float(target_capacity),
            tolerance_gb=float(tolerance_gb),
            effective_min_gb=float(target_capacity - tolerance_gb),
            effective_max_gb=float(target_capacity + tolerance_gb),
            lane_count=8,
            loading_method=fallback_loading_method,
            sequencing_mode='',
            fc_min_data_gb=0.0,
            profile={},
        )

    def get_loading_rule_scope(self) -> Dict[str, Set[str]]:
        """获取上机浓度规则的工序和机型作用域。"""
        machine_types: Set[str] = set()
        test_nos: Set[str] = set()
        for rule in self._rule_matrix_config.get('loading_concentration_rules', []):
            machine_types.update(set(rule.get('machine_types', set()) or set()))
            test_nos.update(set(rule.get('test_nos', set()) or set()))
        return {
            'machine_types': machine_types,
            'test_nos': test_nos,
        }

    def _condition_seq_strategy_and_sample_type(
        self,
        libraries: List[Any],
        condition: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """判断是否命中“测序策略+文库类型”条件。"""
        metadata = metadata or {}
        strategy_keyword = self._normalize_seq_keyword(condition.get('seq_strategy_keyword', ''))
        sample_type_group_code = str(condition.get('sample_type_group_code', '') or '')
        target_sample_types = set()
        if sample_type_group_code:
            target_sample_types = set(
                ((self._rule_matrix_config.get('sample_type_groups') or {}).get(sample_type_group_code, {}))
                .get('sample_types', set())
                or set()
            )

        seq_strategy = self._resolve_seq_strategy(libraries, metadata)
        if strategy_keyword and seq_strategy != strategy_keyword:
            return False

        return any(
            self._get_library_sample_type(lib) in target_sample_types for lib in libraries
        )

    def _condition_has_package_lane_or_fc(self, lib: Any) -> bool:
        """判断文库是否已有包Lane或包FC。"""
        has_package_lane = str(getattr(lib, 'is_package_lane', '') or '').strip() == '是'
        package_lane_number = str(getattr(lib, 'package_lane_number', '') or '').strip()
        package_fc_number = str(
            getattr(lib, 'package_fc_number', None)
            or getattr(lib, 'fc_number', None)
            or getattr(lib, 'flowcell_id', None)
            or ''
        ).strip()
        return has_package_lane or bool(package_lane_number) or bool(package_fc_number)

    def _has_package_lane_context(
        self,
        libraries: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """判断当前Lane是否为包Lane上下文。"""
        metadata = metadata or {}
        if bool(metadata.get('is_dedicated_imbalance_lane')):
            return False
        if bool(metadata.get('is_package_lane')):
            return True

        _invalid_strs = {'', 'nan', 'none', 'null', 'na'}
        for lib in libraries:
            has_package_lane = str(getattr(lib, 'is_package_lane', '') or '').strip() == '是'
            raw_pkg_num = (
                getattr(lib, 'package_lane_number', None)
                or getattr(lib, 'baleno', None)
            )
            # NaN / None / 空字符串统一视为"无包Lane号"，避免 str(nan)='nan' 被误判
            if raw_pkg_num is None:
                package_lane_number = ''
            else:
                try:
                    import math
                    if isinstance(raw_pkg_num, float) and math.isnan(raw_pkg_num):
                        package_lane_number = ''
                    else:
                        package_lane_number = str(raw_pkg_num).strip()
                        if package_lane_number.lower() in _invalid_strs:
                            package_lane_number = ''
                except Exception:
                    package_lane_number = ''
            if has_package_lane or bool(package_lane_number):
                return True
        return False

    def _evaluate_forbid_condition(
        self,
        libraries: List[Any],
        condition: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """评估单条禁止条件。"""
        metadata = metadata or {}
        condition_type = str(condition.get('condition_type', '') or '').strip()
        project_type = self._classify_lane_project_type(libraries)

        if condition_type == 'lane_project_type_not_in':
            allowed_values = {
                self._normalize_text(item)
                for item in condition.get('project_types', [])
                if self._normalize_text(item)
            }
            return project_type not in allowed_values

        if condition_type == 'library_add_test_or_mixed':
            keywords = condition.get('keywords', [])
            for lib in libraries:
                text_values = [
                    str(getattr(lib, 'add_tests_remark', '') or ''),
                    str(getattr(lib, 'machine_note', '') or ''),
                    str(getattr(lib, 'remarks', '') or ''),
                ]
                combined = " ".join(text_values)
                if any(keyword in combined for keyword in keywords):
                    return True
            return False

        if condition_type == 'library_sample_type_contains_any':
            keywords = {
                self._normalize_text(item)
                for item in condition.get('keywords', [])
                if self._normalize_text(item)
            }
            for lib in libraries:
                sample_type = self._get_library_sample_type(lib)
                if any(keyword in sample_type for keyword in keywords):
                    return True
            return False

        if condition_type == 'has_package_lane_or_fc':
            return any(self._condition_has_package_lane_or_fc(lib) for lib in libraries)

        if condition_type == 'seq_strategy_and_sample_type':
            return self._condition_seq_strategy_and_sample_type(libraries, condition, metadata)

        if condition_type == 'has_special_split':
            for lib in libraries:
                raw_value = (
                    getattr(lib, 'special_splits', None)
                    or getattr(lib, 'wkspecialsplits', None)
                    or ''
                )
                if str(raw_value).strip() not in {'', '-', 'nan', 'None', 'NONE', 'null', 'NULL'}:
                    return True
            return False

        logger.warning(f"未知统一规则禁止条件类型: {condition_type}")
        return False

    def validate_lane_constraints(
        self,
        libraries: List[Any],
        machine_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """校验统一规则表中的Lane限制条件。"""
        if not libraries:
            return []

        metadata = metadata or {}
        normalized_machine_type = self._normalize_text(machine_type)
        process_code = self._resolve_process_code(libraries, metadata)
        test_no = self._resolve_test_no(libraries, metadata)
        seq_mode = self._resolve_seq_mode(libraries, metadata)
        project_type = self._classify_lane_project_type(libraries)
        lane_sample_types = self._get_lane_sample_types(libraries)
        messages: List[str] = []

        for constraint in self._rule_matrix_config.get('lane_constraints', []):
            if not self._match_scope(normalized_machine_type, set(constraint.get('machine_types', set()) or set())):
                continue
            if not self._match_process_scope(process_code, set(constraint.get('process_codes', set()) or set())):
                continue
            if not self._match_scope(test_no, set(constraint.get('test_nos', set()) or set())):
                continue
            if not self._match_scope(seq_mode, set(constraint.get('seq_modes', set()) or set())):
                continue
            if not self._match_scope(project_type, set(constraint.get('project_types', set()) or set())):
                continue

            constraint_type = str(constraint.get('constraint_type', '') or '').strip()
            if constraint_type == 'mutually_exclusive_sample_type_groups':
                group_codes = list(constraint.get('group_codes', []) or [])
                matched_group_count = sum(
                    1
                    for group_code in group_codes
                    if lane_sample_types
                    and lane_sample_types & set(
                        ((self._rule_matrix_config.get('sample_type_groups') or {}).get(group_code, {}))
                        .get('sample_types', set())
                        or set()
                    )
                )
                if matched_group_count > 1:
                    messages.append(str(constraint.get('message', 'Lane命中互斥文库组合限制')))
            elif constraint_type == 'forbid_mode_when_any_condition_matches':
                conditions = list(constraint.get('conditions', []) or [])
                if any(
                    self._evaluate_forbid_condition(libraries, condition, metadata)
                    for condition in conditions
                ):
                    messages.append(str(constraint.get('message', 'Lane命中模式禁止条件')))
            else:
                logger.warning(f"未知统一规则Lane限制类型: {constraint_type}")

        return messages

    def resolve_loading_concentration(
        self,
        libraries: List[Any],
        machine_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[float], str]:
        """按统一规则表解析lsjnd上机浓度。"""
        if not libraries:
            return None, "empty_lane"

        metadata = metadata or {}
        if not machine_type:
            machine_type = str(getattr(libraries[0], 'eq_type', '') or '')
        normalized_machine_type = self._normalize_text(machine_type)
        process_code = self._resolve_process_code(libraries, metadata)
        test_no = self._resolve_test_no(libraries, metadata)
        seq_mode = self._resolve_seq_mode(libraries, metadata)
        lane_sample_types = self._get_lane_sample_types(libraries)

        for rule in self._rule_matrix_config.get('loading_concentration_rules', []):
            if not self._match_scope(normalized_machine_type, set(rule.get('machine_types', set()) or set())):
                continue
            if not self._match_process_scope(process_code, set(rule.get('process_codes', set()) or set())):
                continue
            if not self._match_scope(test_no, set(rule.get('test_nos', set()) or set())):
                continue
            if not self._match_scope(seq_mode, set(rule.get('seq_modes', set()) or set())):
                continue

            rule_type = str(rule.get('rule_type', '') or '').strip()
            if rule_type == 'sample_type_subset':
                group_code = str(rule.get('sample_type_group_code', '') or '')
                if self._lane_matches_sample_type_group(lane_sample_types, group_code):
                    return float(rule.get('concentration', 0.0) or 0.0), str(
                        rule.get('reason', rule.get('rule_code', 'matched_rule'))
                    )
            elif rule_type == 'medical_commission_threshold':
                threshold = float(rule.get('data_threshold_gb', 0.0) or 0.0)
                medical_data = sum(
                    float(getattr(lib, 'contract_data_raw', 0.0) or 0.0)
                    for lib in libraries
                    if (
                        ('医学' in self._normalize_text(getattr(lib, 'sub_project_name', '')))
                        or ('医检所' in self._normalize_text(getattr(lib, 'sub_project_name', '')))
                    )
                    and ('委托' in self._normalize_text(getattr(lib, 'sub_project_name', '')))
                )
                if medical_data > threshold:
                    return float(rule.get('concentration', 0.0) or 0.0), str(
                        rule.get('reason', rule.get('rule_code', 'matched_rule'))
                    )
            elif rule_type == 'seq_strategy_and_sample_type':
                if self._condition_seq_strategy_and_sample_type(libraries, rule, metadata):
                    return float(rule.get('concentration', 0.0) or 0.0), str(
                        rule.get('reason', rule.get('rule_code', 'matched_rule'))
                    )

        return None, "no_special_rule_matched"
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取原始配置值
        
        支持嵌套键访问，如 'ai_models.scheduling_model.host'
        
        Args:
            key: 配置键，支持点分隔的嵌套路径
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._raw_config
        
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                    if value is None:
                        return default
                else:
                    return default
            return value if value is not None else default
        except (KeyError, TypeError):
            return default
    
    def _load_mode_1_1_rules(self) -> None:
        """加载1.1模式专属业务规则配置。"""
        config_dir = Path(__file__).parent.parent.parent / 'config'
        rules_path = config_dir / 'mode_1_1_rules.json'
        if not rules_path.exists():
            logger.debug(f"1.1模式规则配置不存在，跳过加载: {rules_path}")
            return

        try:
            with open(rules_path, 'r', encoding='utf-8') as f:
                self._mode_1_1_rules = json.load(f)
            logger.info("1.1模式规则配置加载完成: {}", rules_path)
        except Exception as exc:
            logger.warning("加载1.1模式规则配置失败: {}", exc)
            self._mode_1_1_rules = {}

    def get_mode_1_1_config(self) -> Dict[str, Any]:
        """获取1.1模式专属规则配置（只读副本）。"""
        return dict(self._mode_1_1_rules)

    def get_mode_1_1_value(self, key: str, default: Any = None) -> Any:
        """获取1.1模式配置中的单个参数值。"""
        return self._mode_1_1_rules.get(key, default)

    def _load_from_yaml(self):
        """从YAML配置文件加载配置"""
        config_dir = Path(__file__).parent.parent.parent / 'config'
        
        # 加载机器配置
        machine_config_path = config_dir / 'machine_configs.yaml'
        if machine_config_path.exists():
            try:
                with open(machine_config_path, 'r', encoding='utf-8') as f:
                    machine_config = yaml.safe_load(f)
                    
                # 更新机器容量
                if 'nova_x_25b' in machine_config:
                    self.lane_capacity.machine_capacities['Nova X-25B'] = machine_config['nova_x_25b'].get('capacity', 975.0)
                if 'nova_x_10b' in machine_config:
                    self.lane_capacity.machine_capacities['Nova X-10B'] = machine_config['nova_x_10b'].get('capacity', 380.0)
                    
                logger.debug("机器配置已从YAML加载")
            except Exception as e:
                logger.warning(f"加载机器配置失败: {e}")
        
        # 加载业务规则配置
        rules_config_path = config_dir / 'business_rules.yaml'
        if rules_config_path.exists():
            try:
                with open(rules_config_path, 'r', encoding='utf-8') as f:
                    rules_config = yaml.safe_load(f)
                
                # 更新Pooling配置
                if 'pooling_adjustment' in rules_config:
                    pooling_cfg = rules_config['pooling_adjustment']
                    self.pooling.min_ordered_gb = pooling_cfg.get('min_order_amount', 1.0)
                    self.pooling.max_ordered_ratio = pooling_cfg.get('max_order_ratio', 2.0)
                
                # 更新浓度配置
                if 'loading_concentration' in rules_config:
                    conc_cfg = rules_config['loading_concentration']
                    self.pooling.concentration_base_imbalance_package = conc_cfg.get('base_imbalance_package_lane', 2.5)
                    self.pooling.concentration_scattered_mixed = conc_cfg.get('base_imbalance_mixed', 1.78)
                    self.pooling.concentration_clinical_lane = conc_cfg.get('clinical_lane', 2.3)
                    self.pooling.concentration_default = conc_cfg.get('default', 2.0)
                    
                logger.debug("业务规则配置已从YAML加载")
            except Exception as e:
                logger.warning(f"加载业务规则配置失败: {e}")
    
    def get_lane_capacity(self, machine_type: str, mode: SchedulingMode = SchedulingMode.NON_1_0) -> float:
        """获取Lane容量"""
        if mode == SchedulingMode.MODE_1_0:
            return self.lane_capacity.mode_1_0_mixed_capacity
        
        return self.lane_capacity.machine_capacities.get(
            machine_type, 
            self.lane_capacity.standard_capacity
        )
    
    def get_special_library_limit(self, machine_type: str) -> float:
        """获取特殊文库总量限制"""
        return self.validation_limits.special_library_capacity.get(
            machine_type,
            self.validation_limits.special_library_capacity.get('Nova X-25B', 350.0)
        )
    
    def get_concentration(self, lane_type: str) -> float:
        """获取上机浓度"""
        concentration_map = {
            'base_imbalance_package_lane': self.pooling.concentration_base_imbalance_package,
            'scattered_mixed': self.pooling.concentration_scattered_mixed,
            'clinical_lane': self.pooling.concentration_clinical_lane,
            'default': self.pooling.concentration_default,
        }
        return concentration_map.get(lane_type, self.pooling.concentration_default)
    
    def is_clinical_library(self, sample_id: str) -> bool:
        """判断是否为临检文库"""
        if not sample_id:
            return False
        return any(sample_id.startswith(prefix) for prefix in self.priority.clinical_prefixes)
    
    def is_yc_library(self, sample_id: str, sub_project_name: str = '') -> bool:
        """判断是否为YC文库"""
        if not sample_id:
            return False
        if any(sample_id.startswith(prefix) for prefix in self.priority.yc_prefixes):
            return True
        if sub_project_name and 'YC' in sub_project_name:
            return True
        return False
    
    def is_mode_1_0_excluded(self, sub_project_name: str, library_type: str, remark: str = '') -> bool:
        """判断是否排除在1.0模式外"""
        # 检查关键词
        for keyword in self.mode_1_0.exclude_keywords:
            if keyword in (sub_project_name or ''):
                return True
        
        # 检查文库类型
        for exclude_type in self.mode_1_0.exclude_library_types:
            if exclude_type in (library_type or ''):
                return True
        
        # 检查备注
        if remark:
            if '加测' in remark or '混合' in remark or '不参与新模式排机' in remark:
                return True
        
        return False
    
    def is_base_imbalance_library(self, library_type: str) -> bool:
        """判断是否为碱基不均衡文库"""
        if not library_type:
            return False
        return any(keyword in library_type for keyword in self.base_imbalance.group_keywords)
    
    def to_dict(self) -> Dict[str, Any]:
        """导出配置为字典"""
        return {
            'lane_capacity': {
                'standard_capacity': self.lane_capacity.standard_capacity,
                'mode_1_0_rna_capacity': self.lane_capacity.mode_1_0_rna_capacity,
                'mode_1_0_mixed_capacity': self.lane_capacity.mode_1_0_mixed_capacity,
                'package_lane_target': self.lane_capacity.package_lane_target,
                'machine_capacities': self.lane_capacity.machine_capacities,
            },
            'validation_limits': {
                'customer_ratio_limit': self.validation_limits.customer_ratio_limit,
                'index_10bp_ratio_min': self.validation_limits.index_10bp_ratio_min,
                'single_end_ratio_limit': self.validation_limits.single_end_ratio_limit,
                'base_imbalance_ratio_limit': self.validation_limits.base_imbalance_ratio_limit,
                'special_library_type_limit': self.validation_limits.special_library_type_limit,
            },
            'pooling': {
                'cv_threshold': self.pooling.cv_threshold,
                'lane_cv_threshold': self.pooling.lane_cv_threshold,
                'lane_cv_penalty_weight': self.pooling.lane_cv_penalty_weight,
                'min_ordered_gb': self.pooling.min_ordered_gb,
                'max_ordered_ratio': self.pooling.max_ordered_ratio,
            },
            'priority': {
                'clinical_prefixes': self.priority.clinical_prefixes,
                'yc_prefixes': self.priority.yc_prefixes,
                'urgent_delivery_hours': self.priority.urgent_delivery_hours,
                'large_data_threshold_gb': self.priority.large_data_threshold_gb,
            },
            'retry': {
                'max_retries': self.retry.max_retries,
                'base_delay': self.retry.base_delay,
                'circuit_breaker_threshold': self.retry.circuit_breaker_threshold,
            },
        }


# 全局配置实例
def get_scheduling_config() -> SchedulingConfigManager:
    """获取排机配置管理器实例"""
    return SchedulingConfigManager()


# 便捷访问函数
def get_lane_capacity(machine_type: str = 'Nova X-25B', mode: str = 'non_1.0_mode') -> float:
    """获取Lane容量"""
    config = get_scheduling_config()
    mode_enum = SchedulingMode.MODE_1_0 if mode == '1.0_mode' else SchedulingMode.NON_1_0
    return config.get_lane_capacity(machine_type, mode_enum)


def get_validation_limits() -> ValidationLimitsConfig:
    """获取校验限制配置"""
    return get_scheduling_config().validation_limits


def get_pooling_config() -> PoolingConfig:
    """获取Pooling配置"""
    return get_scheduling_config().pooling


def get_priority_config() -> PriorityConfig:
    """获取优先级配置"""
    return get_scheduling_config().priority


def get_retry_config() -> RetryConfig:
    """获取重试配置"""
    return get_scheduling_config().retry


def get_validation_limits_config() -> ValidationLimitsConfig:
    """获取校验限制配置"""
    return get_scheduling_config().validation_limits



def get_genetic_algorithm_config() -> GeneticAlgorithmConfig:
    """获取遗传算法参数配置"""
    return get_scheduling_config().genetic_algorithm


def get_library_split_config() -> LibrarySplitConfig:
    """获取文库拆分配置"""
    return get_scheduling_config().library_split


def get_index_validation_config() -> IndexValidationConfig:
    """获取Index校验配置"""
    return get_scheduling_config().index_validation


def get_constraint_solver_config() -> ConstraintSolverConfig:
    """获取约束求解器配置"""
    return get_scheduling_config().constraint_solver
