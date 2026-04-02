"""
包Lane/FC排机程序
创建时间：2025-12-02 17:40:00
更新时间：2026-04-02 00:00:00

依据《排机流程规划》步骤二.1实现：
- 根据包Lane编号/Lane ID/FC/RunCycle进行固定分组
- 容量校验：包Lane数据量1000G±0.01G（999.99G-1000.01G）
- Index校验：校验Lane内Index冲突
- 成Lane：生成Lane分配结果
- Pooling：计算Pooling系数
- 计算：计算取样体积
"""

import copy
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

from loguru import logger

# from arrange_library.liblane_paths import setup_liblane_paths
# setup_liblane_paths()

from arrange_library.core.ai.pooling_coefficient_optimizer import (
    PoolingCoefficientOptimizer,
    PoolingOptimizationResult,
)
from arrange_library.core.config.scheduling_config import LaneRuleSelection, get_scheduling_config
from arrange_library.models.library_info import EnhancedLibraryInfo
from arrange_library.core.preprocessing.base_imbalance_handler import BaseImbalanceHandler


class PackageType(Enum):
    """包类型枚举"""
    PACKAGE_LANE = "package_lane"  # 包Lane
    LANE_ID = "lane_id"            # Lane ID
    FC = "fc"                      # 包FC
    RUN_CYCLE = "run_cycle"        # RunCycle


@dataclass
class LaneResult:
    """Lane分配结果"""
    lane_id: str
    libraries: List[EnhancedLibraryInfo]
    total_data_gb: float
    package_type: PackageType
    package_id: str
    index_valid: bool = True
    index_conflicts: List[str] = field(default_factory=list)
    pooling_coefficients: Dict[str, float] = field(default_factory=dict)
    volumes: Dict[str, float] = field(default_factory=dict)
    
    @property
    def library_count(self) -> int:
        return len(self.libraries)


@dataclass
class RunResult:
    """Run分配结果"""
    run_id: str
    fc_id: str
    lanes: List[LaneResult]
    machine_type: str
    run_cycle: str
    total_data_gb: float
    
    @property
    def lane_count(self) -> int:
        return len(self.lanes)
    
    @property
    def library_count(self) -> int:
        return sum(lane.library_count for lane in self.lanes)


@dataclass
class PackageSchedulingResult:
    """包排机结果"""
    total_runs: int
    total_lanes: int
    total_libraries: int
    runs: List[RunResult] = field(default_factory=list)
    remaining_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    failed_packages: Dict[str, str] = field(default_factory=dict)  # {package_id: failure_reason}
    
    def to_dict(self) -> Dict:
        return {
            'total_runs': self.total_runs,
            'total_lanes': self.total_lanes,
            'total_libraries': self.total_libraries,
            'failed_packages_count': len(self.failed_packages),
            'remaining_libraries_count': len(self.remaining_libraries)
        }


@dataclass
class MultiPackageLaneSplitFamily:
    """单个文库多包Lane编号拆分后的家族信息。"""
    family_id: str
    source_library: EnhancedLibraryInfo
    target_package_lane_numbers: List[str]
    raw_package_lane_number: str

    @property
    def total_fragments(self) -> int:
        return len(self.target_package_lane_numbers)


class PackageLaneScheduler:
    """包Lane/FC排机程序
    
    依据《排机流程规划》实现包Lane、包FC的固定分组排机：
    1. 根据包Lane编号/Lane ID/FC/RunCycle进行固定分组
    2. 容量校验：包Lane数据量1000G±5G（995G-1005G）
    3. Index校验：校验Lane内Index冲突
    4. 生成Lane分配结果
    """
    
    # 包Lane容量配置（1000G±0.01G）
    PACKAGE_LANE_TARGET_GB = 1000.0
    PACKAGE_LANE_TOLERANCE_GB = 0.01
    PACKAGE_LANE_MIN_GB = 999.99
    PACKAGE_LANE_MAX_GB = 1000.01
    
    # 普通Lane最大数据量（GB）- 用于其他类型的Lane
    MAX_LANE_CAPACITY_GB = 1000.0
    
    # FC容量配置（每种机器类型）
    FC_CAPACITY_CONFIG = {
        'Nova X-25B': {'lanes': 8, 'max_data_gb': 7800, 'lane_capacity': 975},
        'Nova X-10B': {'lanes': 8, 'max_data_gb': 3040, 'lane_capacity': 380},
        'Novaseq': {'lanes': 4, 'max_data_gb': 3520, 'lane_capacity': 880},
        'T7': {'lanes': 1, 'max_data_gb': 1670, 'lane_capacity': 1670},
        'SURFSEQ-5000': {'lanes': 1, 'max_data_gb': 1200, 'lane_capacity': 1200},
        'SURFSEQ-Q': {'lanes': 8, 'max_data_gb': 6000, 'lane_capacity': 750},
    }
    
    def __init__(self, pooling_optimizer: Optional[PoolingCoefficientOptimizer] = None):
        """初始化排机程序"""
        self._lane_counter = 0
        self._run_counter = 0
        self.pooling_optimizer = pooling_optimizer or PoolingCoefficientOptimizer()
        self.scheduling_config = get_scheduling_config()
        # 包Lane规则由当前调度器强制收紧，避免受外部配置漂移影响。
        self.scheduling_config.lane_capacity.package_lane_target = self.PACKAGE_LANE_TARGET_GB
        self.scheduling_config.lane_capacity.package_lane_tolerance = self.PACKAGE_LANE_TOLERANCE_GB
        self.scheduling_config.lane_capacity.package_lane_min = self.PACKAGE_LANE_MIN_GB
        self.scheduling_config.lane_capacity.package_lane_max = self.PACKAGE_LANE_MAX_GB
        self.imbalance_handler = BaseImbalanceHandler()
        logger.info("包Lane/FC排机程序初始化完成")

    def _get_package_lane_capacity_range(self, machine_type: str) -> LaneRuleSelection:
        """包lane优先走专属容量规则，不复用普通混排容量矩阵。"""
        normalized_machine_type = self.scheduling_config._normalize_text(machine_type)
        loading_method = ''
        if '25B' in normalized_machine_type or normalized_machine_type == self.scheduling_config._normalize_text('NovaSeq X Plus'):
            loading_method = '25B'

        return LaneRuleSelection(
            rule_code="package_lane_capacity",
            soft_target_gb=float(self.scheduling_config.lane_capacity.package_lane_target),
            min_target_gb=float(self.scheduling_config.lane_capacity.package_lane_min),
            max_target_gb=float(self.scheduling_config.lane_capacity.package_lane_max),
            tolerance_gb=0.0,
            effective_min_gb=float(self.scheduling_config.lane_capacity.package_lane_min),
            effective_max_gb=float(self.scheduling_config.lane_capacity.package_lane_max),
            lane_count=8,
            loading_method=loading_method,
            sequencing_mode='',
            fc_min_data_gb=0.0,
            profile={'source': 'lane_capacities.package_lane'},
        )

    @staticmethod
    def _apply_scheduling_capacity_cap(selection):
        """仅在排机阶段收紧标准25B规则上限，不影响LaneValidator。"""
        rule_code = str(getattr(selection, 'rule_code', '') or '')
        if rule_code in {'tj_1595_standard_pe150_25b', 'tj_1595_standard_pe150_25b_other'}:
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
    ):
        """获取仅用于排机的容量规则，允许比校验规则更保守。"""
        selection = self.scheduling_config.get_lane_capacity_range(
            libraries=libraries,
            machine_type=machine_type,
        )
        return self._apply_scheduling_capacity_cap(selection)
    
    def schedule(self, libraries: List[EnhancedLibraryInfo]) -> PackageSchedulingResult:
        """
        执行包排机
        
        Args:
            libraries: 文库列表
            
        Returns:
            PackageSchedulingResult: 排机结果
        """
        logger.info("=" * 60)
        logger.info(" 开始包Lane/FC排机")
        logger.info("=" * 60)
        logger.info(f"待排机文库总数: {len(libraries)}")

        prepared_libraries, split_families, preprocessing_failed, preprocessing_unprocessed = (
            self._expand_multi_package_lane_libraries(libraries)
        )
        
        # 分离包文库和普通文库
        package_lane_libs, lane_id_libs, fc_libs, run_cycle_libs, remaining_libs = (
            self._separate_libraries(prepared_libraries)
        )
        remaining_libs.extend(preprocessing_unprocessed)
        
        runs = []
        failed_packages = dict(preprocessing_failed)
        
        # 1. 处理包Lane
        lane_results, failed_lane_packages, unprocessed_lane_libs = self._process_package_lanes(package_lane_libs)
        lane_results, failed_lane_packages, unprocessed_lane_libs = (
            self._rollback_incomplete_multi_package_lane_splits(
                lane_results,
                failed_lane_packages,
                unprocessed_lane_libs,
                split_families,
            )
        )
        remaining_libs.extend(unprocessed_lane_libs)
        failed_packages.update(failed_lane_packages)
        
        # 2. 处理Lane ID
        lane_id_results, failed_lane_id, unprocessed_lane_id = self._process_lane_ids(lane_id_libs)
        lane_results.extend(lane_id_results)
        remaining_libs.extend(unprocessed_lane_id)
        failed_packages.update(failed_lane_id)
        
        # 3. 处理包FC
        fc_runs, failed_fc, unprocessed_fc = self._process_fc_packages(fc_libs)
        runs.extend(fc_runs)
        remaining_libs.extend(unprocessed_fc)
        failed_packages.update(failed_fc)
        
        # 4. 处理RunCycle
        run_cycle_results, failed_rc, unprocessed_rc = self._process_run_cycles(run_cycle_libs)
        lane_results.extend(run_cycle_results)
        remaining_libs.extend(unprocessed_rc)
        failed_packages.update(failed_rc)
        
        # 将独立Lane打包成Run
        if lane_results:
            standalone_runs = self._pack_lanes_to_runs(lane_results)
            runs.extend(standalone_runs)
        
        # 统计结果
        total_lanes = sum(run.lane_count for run in runs)
        total_libraries = sum(run.library_count for run in runs)
        
        result = PackageSchedulingResult(
            total_runs=len(runs),
            total_lanes=total_lanes,
            total_libraries=total_libraries,
            runs=runs,
            remaining_libraries=remaining_libs,
            failed_packages=failed_packages
        )
        
        self._print_scheduling_summary(result)
        
        return result

    @staticmethod
    def _parse_package_lane_numbers(raw_value: Optional[str]) -> List[str]:
        """解析包Lane编号，支持英文逗号和中文逗号。"""
        text = str(raw_value or '').strip()
        if not text:
            return []

        normalized = text.replace('，', ',').replace('；', ',').replace(';', ',')
        return [token.strip() for token in normalized.split(',') if token.strip()]

    @staticmethod
    def _split_contract_data_evenly(total_data: float, split_count: int) -> List[float]:
        """按份数均分合同数据量，并让最后一份兜底总和误差。"""
        if split_count <= 1:
            return [float(total_data)]

        average = float(total_data) / split_count
        split_values = [average] * split_count
        split_values[-1] = float(total_data) - average * (split_count - 1)
        return split_values

    def _build_multi_package_lane_family_id(
        self,
        lib: EnhancedLibraryInfo,
        family_index: int,
    ) -> str:
        """构造多包Lane拆分家族唯一标识。"""
        origrec = str(getattr(lib, 'origrec', '') or '').strip()
        library_code = str(getattr(lib, 'library_code', '') or '').strip()
        source_key = origrec or library_code or f"UNKNOWN_{family_index}"
        return f"PKG_MULTI_{source_key}_{family_index:04d}"

    def _expand_multi_package_lane_libraries(
        self,
        libraries: List[EnhancedLibraryInfo],
    ) -> Tuple[
        List[EnhancedLibraryInfo],
        Dict[str, MultiPackageLaneSplitFamily],
        Dict[str, str],
        List[EnhancedLibraryInfo],
    ]:
        """对逗号分隔的多包Lane编号执行专用拆分。

        仅此场景允许带包Lane编号文库拆分。若配置非法，则原文库保持未排机。
        """
        expanded_libraries: List[EnhancedLibraryInfo] = []
        split_families: Dict[str, MultiPackageLaneSplitFamily] = {}
        failed_packages: Dict[str, str] = {}
        unprocessed_libraries: List[EnhancedLibraryInfo] = []
        family_index = 0

        for lib in libraries:
            raw_package_lane = getattr(lib, 'package_lane_number', None) or getattr(lib, 'baleno', None)
            package_lane_numbers = self._parse_package_lane_numbers(raw_package_lane)

            if len(package_lane_numbers) <= 1:
                if package_lane_numbers:
                    normalized_pkg = package_lane_numbers[0]
                    lib.package_lane_number = normalized_pkg
                    lib.baleno = normalized_pkg
                    lib.is_package_lane = '是'
                expanded_libraries.append(lib)
                continue

            lib.package_lane_number = str(raw_package_lane).strip()
            lib.baleno = lib.package_lane_number
            lib.is_package_lane = '是'

            source_label = (
                str(getattr(lib, 'origrec', '') or '').strip()
                or str(getattr(lib, 'library_code', '') or '').strip()
                or f"UNKNOWN_{family_index + 1}"
            )
            if len(set(package_lane_numbers)) != len(package_lane_numbers):
                failed_packages[f"package_lane_multi_{source_label}"] = (
                    f"文库{source_label}配置了重复包Lane编号{package_lane_numbers}，"
                    "无法拆分到不同Lane，按原文库不拆分且不排机处理"
                )
                unprocessed_libraries.append(lib)
                logger.warning(
                    "文库 {} 多包Lane编号存在重复，跳过包Lane拆分: {}",
                    source_label,
                    package_lane_numbers,
                )
                continue

            family_index += 1
            family_id = self._build_multi_package_lane_family_id(lib, family_index)
            split_families[family_id] = MultiPackageLaneSplitFamily(
                family_id=family_id,
                source_library=lib,
                target_package_lane_numbers=list(package_lane_numbers),
                raw_package_lane_number=str(raw_package_lane).strip(),
            )

            total_contract_data = float(getattr(lib, 'contract_data_raw', 0.0) or 0.0)
            split_values = self._split_contract_data_evenly(total_contract_data, len(package_lane_numbers))
            original_library_id = (
                str(getattr(lib, 'origrec', '') or '').strip()
                or str(getattr(lib, 'library_code', '') or '').strip()
                or family_id
            )
            original_aidbid = str(
                getattr(lib, 'wkaidbid', None) or getattr(lib, 'aidbid', None) or ''
            ).strip()

            logger.info(
                "文库 {} 检测到多个包Lane编号 {}，按{}份执行专用拆分",
                source_label,
                package_lane_numbers,
                len(package_lane_numbers),
            )

            for idx, (package_lane_number, split_value) in enumerate(
                zip(package_lane_numbers, split_values),
                start=1,
            ):
                fragment = copy.deepcopy(lib)
                fragment.contract_data_raw = split_value
                fragment.package_lane_number = package_lane_number
                fragment.baleno = package_lane_number
                fragment.is_package_lane = '是'
                fragment.is_split = True
                fragment.wkissplit = 'yes'
                fragment.split_status = 'completed'
                fragment.wktotalcontractdata = total_contract_data
                fragment.total_contract_data = total_contract_data
                fragment.original_library_id = original_library_id
                fragment.fragment_index = idx
                fragment.total_fragments = len(package_lane_numbers)
                fragment.fragment_id = f"{original_library_id}_PKG{idx:03d}"
                fragment.split_reason = 'package_lane_multi_number'
                fragment.split_strategy = 'package_lane_multi_number'
                fragment._split_source_library = lib
                fragment._package_lane_multi_split = True
                fragment._package_lane_multi_split_family_id = family_id
                fragment._package_lane_original_numbers = tuple(package_lane_numbers)
                fragment._package_lane_original_value = str(raw_package_lane).strip()

                if idx == 1 and original_aidbid:
                    new_aidbid = original_aidbid
                else:
                    new_aidbid = str(uuid.uuid4())
                fragment.wkaidbid = new_aidbid
                fragment.aidbid = new_aidbid

                expanded_libraries.append(fragment)

        return expanded_libraries, split_families, failed_packages, unprocessed_libraries

    def _rollback_incomplete_multi_package_lane_splits(
        self,
        lane_results: List[LaneResult],
        failed_packages: Dict[str, str],
        unprocessed: List[EnhancedLibraryInfo],
        split_families: Dict[str, MultiPackageLaneSplitFamily],
    ) -> Tuple[List[LaneResult], Dict[str, str], List[EnhancedLibraryInfo]]:
        """若多包Lane拆分未全部成功，则整包回滚为原文库且不排机。"""
        if not split_families:
            return lane_results, failed_packages, unprocessed

        lane_results_by_package_id = {
            str(lane.package_id): lane
            for lane in lane_results
            if lane.package_type == PackageType.PACKAGE_LANE
        }
        rollback_families: Set[str] = set()

        for family_id, family in split_families.items():
            family_errors: List[str] = []
            for target_package_id in family.target_package_lane_numbers:
                lane = lane_results_by_package_id.get(target_package_id)
                if lane is None:
                    family_errors.append(f"目标包Lane {target_package_id} 未成功成Lane")
                    continue

                non_package_lane_libs = [
                    lib
                    for lib in lane.libraries
                    if not str(
                        getattr(lib, 'package_lane_number', None) or getattr(lib, 'baleno', None) or ''
                    ).strip()
                ]
                if non_package_lane_libs:
                    family_errors.append(
                        f"目标包Lane {target_package_id} 混入了{len(non_package_lane_libs)}个非包Lane文库"
                    )
                    continue

                matched_fragments = [
                    lib
                    for lib in lane.libraries
                    if getattr(lib, '_package_lane_multi_split_family_id', None) == family_id
                    and str(getattr(lib, 'package_lane_number', '') or '').strip() == target_package_id
                ]
                if len(matched_fragments) != 1:
                    family_errors.append(
                        f"目标包Lane {target_package_id} 中匹配片段数异常: {len(matched_fragments)}"
                    )

            if family_errors:
                rollback_families.add(family_id)
                source_label = (
                    str(getattr(family.source_library, 'origrec', '') or '').strip()
                    or str(getattr(family.source_library, 'library_code', '') or '').strip()
                    or family.family_id
                )
                failed_packages[f"package_lane_multi_{source_label}"] = (
                    f"文库{source_label}多包Lane拆分后未全部成功进入目标Lane，"
                    f"原因: {'；'.join(family_errors)}；按原文库不拆分且不排机处理"
                )

        changed = True
        while changed and rollback_families:
            changed = False
            rollback_package_ids = {
                pkg_id
                for family_id in rollback_families
                for pkg_id in split_families[family_id].target_package_lane_numbers
            }
            for family_id, family in split_families.items():
                if family_id in rollback_families:
                    continue
                if rollback_package_ids.intersection(family.target_package_lane_numbers):
                    rollback_families.add(family_id)
                    source_label = (
                        str(getattr(family.source_library, 'origrec', '') or '').strip()
                        or str(getattr(family.source_library, 'library_code', '') or '').strip()
                        or family.family_id
                    )
                    failed_packages[f"package_lane_multi_{source_label}"] = (
                        f"文库{source_label}关联的目标包Lane与回滚包重叠，"
                        "按规则整家族回滚且不排机"
                    )
                    changed = True

        if not rollback_families:
            return lane_results, failed_packages, unprocessed

        rollback_package_ids = {
            pkg_id
            for family_id in rollback_families
            for pkg_id in split_families[family_id].target_package_lane_numbers
        }

        retained_lane_results: List[LaneResult] = []
        rollback_lane_results: List[LaneResult] = []
        for lane in lane_results:
            if lane.package_type == PackageType.PACKAGE_LANE and str(lane.package_id) in rollback_package_ids:
                rollback_lane_results.append(lane)
            else:
                retained_lane_results.append(lane)

        final_unprocessed: List[EnhancedLibraryInfo] = []
        seen_object_ids: Set[int] = set()
        restored_family_ids: Set[str] = set()

        def add_library(lib: EnhancedLibraryInfo) -> None:
            object_id = id(lib)
            if object_id in seen_object_ids:
                return
            seen_object_ids.add(object_id)
            final_unprocessed.append(lib)

        def restore_source_library(family_id: str) -> None:
            if family_id in restored_family_ids:
                return
            restored_family_ids.add(family_id)
            family = split_families[family_id]
            source_library = family.source_library
            source_library.package_lane_number = family.raw_package_lane_number
            source_library.baleno = family.raw_package_lane_number
            source_library.is_package_lane = '是'
            source_library.is_split = False
            source_library.wkissplit = ''
            source_library.split_status = 'rolled_back'
            source_library.fragment_index = None
            source_library.total_fragments = None
            source_library.fragment_id = None
            source_library.split_reason = 'package_lane_multi_number_rollback'
            source_library.split_strategy = None
            add_library(source_library)

        def collect_rollback_libraries(libs: List[EnhancedLibraryInfo]) -> None:
            for lib in libs:
                family_id = getattr(lib, '_package_lane_multi_split_family_id', None)
                package_id = str(getattr(lib, 'package_lane_number', '') or '').strip()
                if family_id in rollback_families:
                    restore_source_library(family_id)
                elif package_id in rollback_package_ids:
                    add_library(lib)
                else:
                    add_library(lib)

        collect_rollback_libraries(unprocessed)
        for lane in rollback_lane_results:
            collect_rollback_libraries(lane.libraries)

        for package_id in rollback_package_ids:
            failed_packages.setdefault(
                f"package_lane_{package_id}",
                "关联多包Lane编号文库拆分未全部成功，整包回滚且不排机",
            )

        logger.warning(
            "多包Lane拆分触发回滚: 家族数={}，影响包Lane数={}",
            len(rollback_families),
            len(rollback_package_ids),
        )

        return retained_lane_results, failed_packages, final_unprocessed
    
    def _separate_libraries(self, libraries: List[EnhancedLibraryInfo]) -> Tuple:
        """分离不同类型的包文库"""
        package_lane_libs = []
        lane_id_libs = []
        fc_libs = []
        run_cycle_libs = []
        remaining_libs = []
        
        for lib in libraries:
            is_package_lane = getattr(lib, 'is_package_lane', '') == '是'
            package_lane_number = getattr(lib, 'package_lane_number', None)
            lane_id = getattr(lib, 'lane_id', None)
            fc_number = getattr(lib, 'fc_number', None) or getattr(lib, 'flowcell_id', None)
            run_cycle = getattr(lib, 'run_cycle', None)
            
            if package_lane_number:
                lib.is_package_lane = '是'
                package_lane_libs.append(lib)
            elif lane_id:
                lane_id_libs.append(lib)
            elif fc_number:
                fc_libs.append(lib)
            elif run_cycle:
                run_cycle_libs.append(lib)
            else:
                remaining_libs.append(lib)
        
        logger.info(f"  包Lane文库: {len(package_lane_libs)}")
        logger.info(f"  Lane ID文库: {len(lane_id_libs)}")
        logger.info(f"  包FC文库: {len(fc_libs)}")
        logger.info(f"  RunCycle文库: {len(run_cycle_libs)}")
        logger.info(f"  普通文库: {len(remaining_libs)}")
        
        return package_lane_libs, lane_id_libs, fc_libs, run_cycle_libs, remaining_libs
    
    def _process_package_lanes(self, libraries: List[EnhancedLibraryInfo]) -> Tuple[List[LaneResult], Dict[str, str], List[EnhancedLibraryInfo]]:
        """处理包Lane文库"""
        lane_results = []
        failed_packages = {}
        unprocessed = []
        
        # 按包Lane编号分组
        groups = defaultdict(list)
        for lib in libraries:
            pkg_number = str(getattr(lib, 'package_lane_number', ''))
            groups[pkg_number].append(lib)
        
        for pkg_id, libs in groups.items():
            total_data = self._calculate_total_contract_data(libs)
            machine_types = set(getattr(lib, 'eq_type', '') or '' for lib in libs)
            machine_types.discard('')
            machine_type = list(machine_types)[0] if machine_types else 'Nova X-25B'
            lane_rule = self._get_package_lane_capacity_range(machine_type)
            
            # 包Lane容量校验：优先使用统一规则配置
            if total_data < lane_rule.effective_min_gb or total_data > lane_rule.effective_max_gb:
                failed_packages[f"package_lane_{pkg_id}"] = (
                    f"数据量{total_data:.1f}G不在允许范围{lane_rule.effective_min_gb:.1f}G-"
                    f"{lane_rule.effective_max_gb:.1f}G内，规则={lane_rule.rule_code}"
                )
                unprocessed.extend(libs)
                continue

            constraint_messages = self.scheduling_config.validate_lane_constraints(
                libraries=libs,
                machine_type=machine_type,
            )
            if constraint_messages:
                failed_packages[f"package_lane_{pkg_id}"] = "；".join(constraint_messages)
                unprocessed.extend(libs)
                continue

            # 包Lane规则：按整条Lane统计Index对总数，需≥5。
            total_index_pairs = self._count_lane_index_pairs(libs)
            if total_index_pairs < 5:
                failed_packages[f"package_lane_{pkg_id}"] = (
                    f"Index对数不足：当前共{total_index_pairs}对，要求整条包Lane≥5对"
                )
                unprocessed.extend(libs)
                logger.warning(f"包Lane {pkg_id} Index对数不足：当前共{total_index_pairs}对（要求整条包Lane≥5对）")
                continue
            
            # 碱基不均衡限制校验（规则文档Line 81-82）
            imbalance_valid, imbalance_reason = self._validate_package_lane_imbalance(libs)
            if not imbalance_valid:
                failed_packages[f"package_lane_{pkg_id}"] = imbalance_reason
                unprocessed.extend(libs)
                logger.warning(f"包Lane {pkg_id} 碱基不均衡限制不满足：{imbalance_reason}")
                continue
            
            # Index冲突校验
            index_valid, conflicts = self._validate_index_conflicts(libs)
            
            if not index_valid:
                failed_packages[f"package_lane_{pkg_id}"] = f"Index冲突: {', '.join(conflicts[:3])}"
                unprocessed.extend(libs)
                continue
            
            # 创建Lane结果
            lane_result = LaneResult(
                lane_id=self._generate_lane_id(),
                libraries=libs,
                total_data_gb=total_data,
                package_type=PackageType.PACKAGE_LANE,
                package_id=pkg_id,
                index_valid=True,
                index_conflicts=[]
            )
            
            # 计算Pooling系数
            lane_result.pooling_coefficients = self._calculate_pooling_coefficients(
                libs, lane_result.lane_id
            )
            
            lane_results.append(lane_result)
            logger.debug(f"包Lane {pkg_id} 排机成功: {len(libs)}个文库, {total_data:.1f}G")
        
        return lane_results, failed_packages, unprocessed
    
    def _process_lane_ids(self, libraries: List[EnhancedLibraryInfo]) -> Tuple[List[LaneResult], Dict[str, str], List[EnhancedLibraryInfo]]:
        """处理Lane ID文库"""
        lane_results = []
        failed_packages = {}
        unprocessed = []
        
        # 按Lane ID分组
        groups = defaultdict(list)
        for lib in libraries:
            lane_id = str(getattr(lib, 'lane_id', ''))
            groups[lane_id].append(lib)
        
        for lane_id, libs in groups.items():
            total_data = sum(float(lib.contract_data_raw or 0) for lib in libs)
            machine_type = str(getattr(libs[0], 'eq_type', '') or 'Nova X-25B')
            lane_rule = self._get_scheduling_lane_capacity_range(libraries=libs, machine_type=machine_type)
            
            # 容量校验
            if total_data > lane_rule.effective_max_gb:
                failed_packages[f"lane_id_{lane_id}"] = (
                    f"数据量{total_data:.1f}G超过上限{lane_rule.effective_max_gb:.1f}G，"
                    f"规则={lane_rule.rule_code}"
                )
                unprocessed.extend(libs)
                continue
            
            # Index校验
            index_valid, conflicts = self._validate_index_conflicts(libs)
            
            if not index_valid:
                failed_packages[f"lane_id_{lane_id}"] = f"Index冲突"
                unprocessed.extend(libs)
                continue
            
            lane_result = LaneResult(
                lane_id=lane_id,
                libraries=libs,
                total_data_gb=total_data,
                package_type=PackageType.LANE_ID,
                package_id=lane_id,
                index_valid=True
            )
            
            lane_result.pooling_coefficients = self._calculate_pooling_coefficients(
                libs, lane_result.lane_id
            )
            lane_results.append(lane_result)
        
        return lane_results, failed_packages, unprocessed
    
    def _process_fc_packages(self, libraries: List[EnhancedLibraryInfo]) -> Tuple[List[RunResult], Dict[str, str], List[EnhancedLibraryInfo]]:
        """处理包FC文库"""
        runs = []
        failed_packages = {}
        unprocessed = []
        
        # 按FC编号分组
        groups = defaultdict(list)
        for lib in libraries:
            fc_number = str(getattr(lib, 'fc_number', None) or getattr(lib, 'flowcell_id', ''))
            groups[fc_number].append(lib)
        
        for fc_id, libs in groups.items():
            total_data = sum(float(lib.contract_data_raw or 0) for lib in libs)
            
            # 获取机器类型
            machine_types = set(getattr(lib, 'eq_type', '') or '' for lib in libs)
            machine_types.discard('')
            machine_type = list(machine_types)[0] if machine_types else 'Novaseq'
            
            # 获取FC容量配置
            lane_rule = self._get_scheduling_lane_capacity_range(libraries=libs, machine_type=machine_type)
            fc_config = {
                'max_data_gb': lane_rule.effective_max_gb * max(lane_rule.lane_count, 1),
                'lanes': max(lane_rule.lane_count, 1),
                'lane_capacity': lane_rule.effective_max_gb,
            }
            
            # 容量校验
            if total_data > fc_config['max_data_gb']:
                failed_packages[f"fc_{fc_id}"] = f"数据量{total_data:.1f}G超过FC容量{fc_config['max_data_gb']}G"
                unprocessed.extend(libs)
                continue
            
            # 分配到Lane
            lanes = self._distribute_to_lanes(libs, fc_config['lane_capacity'])
            
            if not lanes:
                failed_packages[f"fc_{fc_id}"] = "Lane分配失败"
                unprocessed.extend(libs)
                continue
            
            # 创建Run
            run_cycle = getattr(libs[0], 'run_cycle', '') or ''
            run = RunResult(
                run_id=self._generate_run_id(),
                fc_id=fc_id,
                lanes=lanes,
                machine_type=machine_type,
                run_cycle=run_cycle,
                total_data_gb=total_data
            )
            
            runs.append(run)
            logger.debug(f"包FC {fc_id} 排机成功: {len(lanes)}条Lane, {total_data:.1f}G")
        
        return runs, failed_packages, unprocessed
    
    def _process_run_cycles(self, libraries: List[EnhancedLibraryInfo]) -> Tuple[List[LaneResult], Dict[str, str], List[EnhancedLibraryInfo]]:
        """处理RunCycle文库"""
        lane_results = []
        failed_packages = {}
        unprocessed = []
        
        # 按RunCycle分组
        groups = defaultdict(list)
        for lib in libraries:
            run_cycle = str(getattr(lib, 'run_cycle', ''))
            groups[run_cycle].append(lib)
        
        for run_cycle, libs in groups.items():
            total_data = sum(float(lib.contract_data_raw or 0) for lib in libs)
            machine_type = str(getattr(libs[0], 'eq_type', '') or 'Nova X-25B')
            lane_rule = self._get_scheduling_lane_capacity_range(libraries=libs, machine_type=machine_type)
            
            # 如果数据量超过单Lane容量，需要分多Lane
            if total_data <= lane_rule.effective_max_gb:
                # 单Lane
                index_valid, conflicts = self._validate_index_conflicts(libs)
                
                if not index_valid:
                    failed_packages[f"run_cycle_{run_cycle}"] = f"Index冲突"
                    unprocessed.extend(libs)
                    continue
                
                lane_result = LaneResult(
                    lane_id=self._generate_lane_id(),
                    libraries=libs,
                    total_data_gb=total_data,
                    package_type=PackageType.RUN_CYCLE,
                    package_id=run_cycle,
                    index_valid=True
                )
                
                lane_result.pooling_coefficients = self._calculate_pooling_coefficients(
                    libs, lane_result.lane_id
                )
                lane_results.append(lane_result)
            else:
                # 需要分多Lane
                lanes = self._distribute_to_lanes(libs, lane_rule.effective_max_gb)
                lane_results.extend(lanes)
        
        return lane_results, failed_packages, unprocessed
    
    def _distribute_to_lanes(self, libraries: List[EnhancedLibraryInfo], lane_capacity: float) -> List[LaneResult]:
        """将文库分配到多个Lane"""
        lanes = []
        current_lane_libs = []
        current_data = 0.0
        
        # 按数据量排序
        sorted_libs = sorted(libraries, key=lambda x: float(x.contract_data_raw or 0), reverse=True)
        
        for lib in sorted_libs:
            lib_data = float(lib.contract_data_raw or 0)
            
            if current_data + lib_data <= lane_capacity:
                current_lane_libs.append(lib)
                current_data += lib_data
            else:
                # 当前Lane满，创建新Lane
                if current_lane_libs:
                    lane = LaneResult(
                        lane_id=self._generate_lane_id(),
                        libraries=current_lane_libs,
                        total_data_gb=current_data,
                        package_type=PackageType.FC,
                        package_id=""
                    )
                    lane.pooling_coefficients = self._calculate_pooling_coefficients(
                        current_lane_libs, lane.lane_id
                    )
                    lanes.append(lane)
                
                current_lane_libs = [lib]
                current_data = lib_data
        
        # 处理最后一个Lane
        if current_lane_libs:
            lane = LaneResult(
                lane_id=self._generate_lane_id(),
                libraries=current_lane_libs,
                total_data_gb=current_data,
                package_type=PackageType.FC,
                package_id=""
            )
            lane.pooling_coefficients = self._calculate_pooling_coefficients(
                current_lane_libs, lane.lane_id
            )
            lanes.append(lane)
        
        return lanes
    
    def _pack_lanes_to_runs(self, lanes: List[LaneResult]) -> List[RunResult]:
        """将独立Lane打包成Run"""
        runs = []
        
        # 按机器类型分组
        lanes_by_machine = defaultdict(list)
        for lane in lanes:
            if lane.libraries:
                machine_type = getattr(lane.libraries[0], 'eq_type', '') or 'Novaseq'
                lanes_by_machine[machine_type].append(lane)
        
        for machine_type, machine_lanes in lanes_by_machine.items():
            first_lane_libs = machine_lanes[0].libraries if machine_lanes else []
            lane_rule = self._get_scheduling_lane_capacity_range(
                libraries=first_lane_libs,
                machine_type=machine_type,
            )
            lanes_per_run = max(lane_rule.lane_count, 1)
            
            # 每N条Lane组成一个Run
            for i in range(0, len(machine_lanes), lanes_per_run):
                run_lanes = machine_lanes[i:i+lanes_per_run]
                total_data = sum(lane.total_data_gb for lane in run_lanes)
                
                run = RunResult(
                    run_id=self._generate_run_id(),
                    fc_id=f"AUTO_FC_{self._run_counter}",
                    lanes=run_lanes,
                    machine_type=machine_type,
                    run_cycle="",
                    total_data_gb=total_data
                )
                runs.append(run)
        
        return runs
    
    def _validate_index_conflicts(self, libraries: List[EnhancedLibraryInfo]) -> Tuple[bool, List[str]]:
        """验证Index冲突"""
        try:
            from arrange_library.core.constraints.index_validator_verified import IndexConflictValidator
            
            validator = IndexConflictValidator()
            result = validator.validate_lane(libraries)
            
            conflicts = [f"{c.library1_id} vs {c.library2_id}" for c in result.conflicts]
            return result.is_valid, conflicts
        except Exception as e:
            logger.warning(f"Index校验失败: {e}")
            return True, []
    
    def _calculate_pooling_coefficients(
        self, libraries: List[EnhancedLibraryInfo], lane_id: Optional[str] = None
    ) -> Dict[str, float]:
        """计算Pooling系数（优先使用loutput优化器，失败降级到规则）"""
        lane_identifier = lane_id or "PACKAGE_LANE"
        if self.pooling_optimizer and self.pooling_optimizer.enabled:
            try:
                result: PoolingOptimizationResult = self.pooling_optimizer.optimize_for_libraries(
                    libraries, lane_identifier
                )
                return result.coefficients
            except Exception as exc:
                logger.exception(f"Lane {lane_identifier} Pooling优化失败，使用规则降级: {exc}")
        return self._calculate_pooling_simple(libraries)

    def _calculate_pooling_simple(self, libraries: List[EnhancedLibraryInfo]) -> Dict[str, float]:
        """计算Pooling系数 - 规则降级路径"""
        coefficients: Dict[str, float] = {}
        for lib in libraries:
            contract_data = float(lib.contract_data_raw or 0)
            coefficient = 1.0
            if contract_data < 1:
                coefficient = 1.5
            elif contract_data > 150:
                coefficient = 1.3
            coefficients[lib.origrec] = coefficient
        return coefficients
    
    def _generate_lane_id(self) -> str:
        """生成Lane ID"""
        self._lane_counter += 1
        return f"LANE_{self._lane_counter:04d}"
    
    def _generate_run_id(self) -> str:
        """生成Run ID"""
        self._run_counter += 1
        return f"RUN_{self._run_counter:04d}"
    
    def _print_scheduling_summary(self, result: PackageSchedulingResult):
        """打印排机摘要"""
        logger.info("-" * 60)
        logger.info(" 包Lane/FC排机结果:")
        logger.info(f"  生成Run数: {result.total_runs}")
        logger.info(f"  生成Lane数: {result.total_lanes}")
        logger.info(f"  排机文库数: {result.total_libraries}")
        logger.info(f"  剩余文库数: {len(result.remaining_libraries)}")
        
        if result.failed_packages:
            logger.info(f"  失败包数: {len(result.failed_packages)}")
            for pkg_id, reason in list(result.failed_packages.items())[:5]:
                logger.info(f"    - {pkg_id}: {reason}")
        
        logger.info("=" * 60)
    
    def _count_index_pairs(self, lib: EnhancedLibraryInfo) -> int:
        """计算文库的Index对数
        
        根据规则文档（Line 225）：Index序列中，逗号","分隔多对Index
        
        Args:
            lib: 文库信息
            
        Returns:
            Index对数
        """
        if not lib.index_seq or lib.index_seq.strip() == "":
            return 0
        
        # 按逗号分隔计算对数
        return lib.index_seq.count(',') + 1

    def _count_lane_index_pairs(self, libraries: List[EnhancedLibraryInfo]) -> int:
        """统计整条Lane的Index对总数。"""
        return sum(self._count_index_pairs(lib) for lib in libraries)

    def _calculate_total_contract_data(self, libraries: List[EnhancedLibraryInfo]) -> float:
        """统计整条Lane合同数据量。"""
        return sum(float(getattr(lib, 'contract_data_raw', 0.0) or 0.0) for lib in libraries)
    
    def _validate_package_lane_imbalance(
        self, 
        libraries: List[EnhancedLibraryInfo]
    ) -> Tuple[bool, Optional[str]]:
        """校验包Lane中碱基不均衡文库的限制
        
        规则文档（Line 76-83）要求：
        - 碱基不均衡文库若申请包Lane，需同时满足所属分组最大数据量（统一240G）
        - 需满足类型混排规则（仅允许文档规定的混排组合）
        - 需严格满足分组数据量占比（0.8/0.99/1.0等）
        
        Args:
            libraries: 文库列表
            
        Returns:
            (是否通过校验, 失败原因)
        """
        # 按分组统计碱基不均衡文库数据量
        group_data: Dict[str, float] = defaultdict(float)
        group_limits: Dict[str, float] = {}
        
        for lib in libraries:
            # 使用BaseImbalanceHandler识别碱基不均衡文库
            if not self.imbalance_handler.is_imbalance_library(lib):
                continue
            
            # 获取文库所属分组
            group_id = self.imbalance_handler.identify_imbalance_type(lib)
            if not group_id:
                return False, f"文库{getattr(lib, 'origrec', 'UNKNOWN')}未识别到碱基不均衡分组"
            
            group_data[group_id] += float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
            
            # 获取该分组的最大数据量限制
            if group_id not in group_limits:
                group_def = self.imbalance_handler.groups.get(group_id)
                if group_def:
                    group_limits[group_id] = float(group_def.max_data_gb or 240.0)
                else:
                    group_limits[group_id] = 240.0
        
        if not group_data:
            return True, None

        # 检查每个分组是否超限
        for group_id, data_gb in group_data.items():
            limit = group_limits.get(group_id, 240.0)
            if data_gb > limit:
                return False, f"分组{group_id}碱基不均衡文库数据量{data_gb:.1f}G超过限制{limit}G"

        # 严格检查分组类型混排规则（G27/G28/G29等）
        mix_ok, mix_reason = self.imbalance_handler.check_mix_compatibility(libraries)
        if not mix_ok:
            return False, f"碱基不均衡类型混排不满足: {mix_reason}"

        # 严格检查分组数据量占比（0.8/0.99/1.0等）
        ratio_ok, ratio_reason = self.imbalance_handler.check_group_data_ratio(libraries)
        if not ratio_ok:
            return False, ratio_reason
        
        return True, None
