"""
1.1模式对外服务封装

对外提供稳定调用入口，复用现有主流程中的分流、排机和第二轮识别能力，
避免外部程序直接耦合 arrange_library_model6.py 内部编排细节。
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
from loguru import logger

from arrange_library.arrange_library_model6 import (
    _build_detail_output,
    _find_best_peak_size_window,
    _build_priority_36t_preconsume_inputs,
    _build_origrec_key,
    _collect_detail_output_libraries,
    _collect_lanes_with_split,
    _collect_prediction_rows,
    _is_machine_supported_for_arrangement,
    _read_csv_with_encoding_fallback,
    _resolve_machine_type_enum_simple,
    _run_prediction_delivery,
    _run_priority_36t_preconsume_stage,
    _safe_str,
    arrange_library,
    load_test_libraries,
    test_with_model,
)
from arrange_library.core.config.scheduling_config import get_scheduling_config
from arrange_library.core.constraints.lane_validator import LaneValidator
from arrange_library.core.scheduling.greedy_lane_scheduler import (
    GreedyLaneConfig,
    GreedyLaneScheduler,
)
from arrange_library.core.scheduling.mode_1_1_round2 import (
    Mode11Round2Handler,
    Round2IdentificationResult,
    Round2SchedulingResult,
)
from arrange_library.core.scheduling.mode_allocator import ModeAllocator, ModeDispatchResult
from arrange_library.core.scheduling.scheduling_types import LaneAssignment
from arrange_library.models.library_info import EnhancedLibraryInfo


@dataclass
class Mode11PreparedLibraries:
    """1.1 service 标准化后的输入文库集合。"""

    source_path: Optional[Path] = None
    all_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    ai_schedulable_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    non_ai_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    excluded_machine_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)


@dataclass
class Mode11Round1ServiceResult:
    """1.1首轮 service 执行结果。"""

    prepared_input: Mode11PreparedLibraries
    dispatch_result: ModeDispatchResult
    deferred_round2_identification: Round2IdentificationResult = field(
        default_factory=Round2IdentificationResult
    )
    first_round_pool: List[EnhancedLibraryInfo] = field(default_factory=list)
    priority_36t_lanes: List[LaneAssignment] = field(default_factory=list)
    priority_36t_remaining_for_1_1: List[EnhancedLibraryInfo] = field(default_factory=list)
    priority_36t_scheduling_stats: Dict[str, Any] = field(default_factory=dict)
    priority_36t_scheduling_error: Optional[str] = None
    lanes: List[LaneAssignment] = field(default_factory=list)
    fallback_libraries_for_36t: List[EnhancedLibraryInfo] = field(default_factory=list)
    normal_libraries_for_36t: List[EnhancedLibraryInfo] = field(default_factory=list)
    scheduling_stats: Dict[str, Any] = field(default_factory=dict)
    solution: Any = None
    scheduling_succeeded: bool = False
    scheduling_error: Optional[str] = None


@dataclass
class Mode11Round2ServiceResult:
    """1.1第二轮 service 执行结果。"""

    prepared_input: Mode11PreparedLibraries
    identification_result: Round2IdentificationResult
    scheduling_result: Optional[Round2SchedulingResult] = None
    schedule_invoked: bool = False


@dataclass
class Mode11FullServiceResult:
    """1.1全流程 service 执行结果。"""

    output_path: Path
    mode: str = "arrange"


@dataclass
class Mode11Round2ExportResult:
    """1.1第二轮轻量导出结果。"""

    output_path: Path
    prepared_input: Mode11PreparedLibraries
    identification_result: Round2IdentificationResult
    scheduling_result: Optional[Round2SchedulingResult] = None
    exported_lane_count: int = 0
    exported_library_count: int = 0
    predicted_row_count: int = 0


def _resolve_mode_1_1_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """解析1.1配置，优先使用显式传入值。"""
    resolved = dict(config or get_scheduling_config().get_mode_1_1_config())
    if not resolved:
        raise ValueError("未加载到1.1模式配置，无法执行 mode_1_1_service")
    return resolved


def _build_mode_1_1_first_round_balanced_25b_lanes(
    libraries: Sequence[EnhancedLibraryInfo],
) -> tuple[List[LaneAssignment], List[EnhancedLibraryInfo], Dict[str, Any]]:
    """按人工首轮思路，先把 Nova X-25B 主峰文库均摊成若干条 1.1 lane。"""
    remaining = list(libraries or [])
    stats: Dict[str, Any] = {
        "balanced_25b_preset_lane_count": 0,
        "balanced_25b_target_lane_count": 0,
        "balanced_25b_dominant_peak_bin": None,
        "balanced_25b_used_gb": 0.0,
    }
    machine_type = "Nova X-25B"
    machine_pool = [
        lib for lib in remaining if str(getattr(lib, "eq_type", "") or "").strip() == machine_type
    ]
    if not machine_pool:
        return [], remaining, stats

    scheduler = GreedyLaneScheduler(
        GreedyLaneConfig(
            use_machine_config=True,
            max_customer_ratio=0.50,
            min_10bp_index_ratio=0.40,
            max_special_library_types=0,
            max_special_library_data_gb=350.0,
            enable_index_check=True,
            enable_imbalance_check=True,
            enable_rule_checker=False,
            max_imbalance_types_per_lane=0,
            max_imbalance_ratio=0.35,
            enable_dedicated_imbalance_lane=True,
            enable_small_library_clustering=False,
            clustering_min_count=30,
            enable_non_10bp_dedicated_lane=False,
            enable_backbone_reservation=False,
        )
    )
    if scheduler.pooling_optimizer:
        scheduler.pooling_optimizer.enabled = False

    validator = LaneValidator(strict_mode=True)
    total_machine_gb = sum(lib.get_data_amount_gb() for lib in machine_pool)
    min_lane_data, max_lane_data = scheduler._resolve_lane_capacity_limits(machine_pool, machine_type)
    target_lane_count = int(total_machine_gb // min_lane_data)
    stats["balanced_25b_target_lane_count"] = target_lane_count
    if target_lane_count < 3:
        return [], remaining, stats

    target_lane_data = min(max_lane_data, total_machine_gb / target_lane_count)
    peak_bin_totals: Dict[int, float] = defaultdict(float)
    for lib in machine_pool:
        peak_size = float(getattr(lib, "peak_size", 0) or 0)
        peak_bin_totals[int(peak_size // 25) * 25] += lib.get_data_amount_gb()
    if not peak_bin_totals:
        return [], remaining, stats

    dominant_peak_bin = max(peak_bin_totals.items(), key=lambda item: item[1])[0]
    dominant_peak_total = peak_bin_totals[dominant_peak_bin]
    stats["balanced_25b_dominant_peak_bin"] = (dominant_peak_bin, dominant_peak_bin + 24)
    if dominant_peak_total + 1e-6 < target_lane_count * min_lane_data * 0.75:
        return [], remaining, stats

    dominant_center = dominant_peak_bin + 12
    dominant_window_min, dominant_window_max, _ = _find_best_peak_size_window(machine_pool)
    dominant_libraries = [
        lib
        for lib in machine_pool
        if dominant_peak_bin <= float(getattr(lib, "peak_size", 0) or 0) <= dominant_peak_bin + 24
    ]
    other_libraries = [lib for lib in machine_pool if id(lib) not in {id(item) for item in dominant_libraries}]
    dominant_libraries.sort(
        key=lambda lib: (
            -lib.get_data_amount_gb(),
            abs((float(getattr(lib, "peak_size", 0) or 0.0)) - dominant_center),
            _safe_str(getattr(lib, "origrec", ""), default=""),
        )
    )
    other_libraries.sort(
        key=lambda lib: (
            abs((float(getattr(lib, "peak_size", 0) or 0.0)) - dominant_center),
            -lib.get_data_amount_gb(),
            _safe_str(getattr(lib, "origrec", ""), default=""),
        )
    )

    groups: List[List[EnhancedLibraryInfo]] = [[] for _ in range(target_lane_count)]

    def _group_total(group: Sequence[EnhancedLibraryInfo]) -> float:
        return sum(lib.get_data_amount_gb() for lib in group)

    def _can_add_to_group(
        group: Sequence[EnhancedLibraryInfo],
        lib: EnhancedLibraryInfo,
        *,
        hard_target: bool,
    ) -> bool:
        candidate = list(group) + [lib]
        candidate_total = _group_total(candidate)
        max_allowed = target_lane_data if hard_target else max_lane_data
        if candidate_total > max_allowed + 1e-6:
            return False
        if scheduler.config.enable_index_check and not scheduler.index_validator.validate_new_lib_quick(list(group), lib):
            return False
        if not scheduler._check_customer_ratio_compatible_by_data(candidate):
            return False
        if not scheduler._check_peak_size_compatible(candidate):
            return False
        return True

    def _place_library(lib: EnhancedLibraryInfo) -> bool:
        ordered_groups = sorted(range(target_lane_count), key=lambda idx: _group_total(groups[idx]))
        for hard_target in (True, False):
            for idx in ordered_groups:
                if _can_add_to_group(groups[idx], lib, hard_target=hard_target):
                    groups[idx].append(lib)
                    return True
        return False

    for lib in dominant_libraries:
        _place_library(lib)
    for lib in other_libraries:
        _place_library(lib)

    preset_lanes: List[LaneAssignment] = []
    used_library_ids: set[int] = set()
    for lane_idx, group in enumerate(groups, start=1):
        if not group:
            return [], remaining, stats
        validation_result = validator.validate_lane(
            list(group),
            lane_id=f"GL_{machine_type}_BAL_{lane_idx:03d}",
            machine_type=machine_type,
            metadata={"seq_mode": "1.1", "sequencing_mode": "1.1"},
        )
        if not validation_result.is_valid:
            return [], remaining, stats

        group_total = _group_total(group)
        if group_total + 1e-6 < min_lane_data or group_total > max_lane_data + 1e-6:
            return [], remaining, stats

        lane = LaneAssignment(
            lane_id=f"GL_{machine_type}_BAL_{lane_idx:03d}",
            machine_id=f"M_GL_{machine_type}_BAL_{lane_idx:03d}",
            machine_type=_resolve_machine_type_enum_simple(machine_type),
            libraries=list(group),
            total_data_gb=group_total,
            lane_capacity_gb=max_lane_data,
            metadata={
                "seq_mode": "1.1",
                "sequencing_mode": "1.1",
                "dominant_peak_window_bp": (dominant_window_min, dominant_window_max),
                "balance_strategy": "dominant_peak_bin_round_robin",
            },
        )
        preset_lanes.append(lane)
        used_library_ids.update(id(lib) for lib in group)

    remaining = [lib for lib in remaining if id(lib) not in used_library_ids]
    stats["balanced_25b_preset_lane_count"] = len(preset_lanes)
    stats["balanced_25b_used_gb"] = round(sum(lane.total_data_gb for lane in preset_lanes), 1)
    return preset_lanes, remaining, stats


def _resolve_round2_export_output_path(
    *,
    data_path: Path,
    output_detail_dir: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
) -> Path:
    """解析1.1第二轮轻量导出路径。"""
    if output_file is not None:
        return Path(output_file)

    output_dir = Path(output_detail_dir) if output_detail_dir is not None else data_path.parent
    return output_dir / f"{data_path.stem}_mode11_round2_output.csv"


def _build_round2_export_key_series(df: pd.DataFrame) -> pd.Series:
    """为第二轮轻量导出构造稳定合并键。"""
    if "origrec_key" in df.columns:
        return df["origrec_key"].astype(str).str.strip()
    return _build_origrec_key(df).astype(str).str.strip()


def _clear_current_arrangement_fields_for_round2_export(df: pd.DataFrame) -> pd.DataFrame:
    """清空当前排机结果字段，避免历史回放值污染第二轮轻量导出结果。"""
    cleared = df.copy()
    for column_name in [
        "lrunid",
        "llaneid",
        "lcxms",
        "lsjfs",
        "laneround",
        "lanecreatetype",
        "排机规则",
        "index查重规则",
    ]:
        if column_name in cleared.columns:
            cleared[column_name] = ""
    return cleared


def _merge_round2_export_rows(
    *,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """将第二轮轻量导出结果按 origrec_key 合并回原始明细。"""
    if target_df.empty:
        return source_df.copy()

    merged = source_df.copy()
    merged["_mode11_round2_merge_key"] = _build_round2_export_key_series(merged)

    target_copy = target_df.copy()
    target_copy["_mode11_round2_merge_key"] = _build_round2_export_key_series(target_copy)
    target_copy = target_copy.drop_duplicates(subset=["_mode11_round2_merge_key"], keep="last")
    target_by_key = target_copy.set_index("_mode11_round2_merge_key")

    for column_name in target_copy.columns:
        if column_name == "_mode11_round2_merge_key":
            continue
        if column_name not in merged.columns:
            merged[column_name] = pd.NA
        elif merged[column_name].dtype != object and target_copy[column_name].dtype == object:
            merged[column_name] = merged[column_name].astype(object)

    target_mask = merged["_mode11_round2_merge_key"].isin(target_by_key.index)
    for column_name in target_copy.columns:
        if column_name == "_mode11_round2_merge_key":
            continue
        merged.loc[target_mask, column_name] = merged.loc[
            target_mask, "_mode11_round2_merge_key"
        ].map(target_by_key[column_name])

    merged = merged.drop(columns=["_mode11_round2_merge_key"], errors="ignore")
    return merged


def _clone_or_load_libraries(
    *,
    data_file: Optional[Union[str, Path]] = None,
    libraries: Optional[Sequence[EnhancedLibraryInfo]] = None,
) -> Mode11PreparedLibraries:
    """统一加载或克隆文库输入，避免 service 修改调用方对象。"""
    if (data_file is None) == (libraries is None):
        raise ValueError("data_file 和 libraries 必须二选一传入")

    if libraries is not None:
        cloned_libraries = deepcopy(list(libraries))
        return Mode11PreparedLibraries(
            source_path=None,
            all_libraries=cloned_libraries,
        )

    source_path = Path(data_file)  # type: ignore[arg-type]
    if not source_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {source_path}")

    loaded_libraries = load_test_libraries(str(source_path))
    return Mode11PreparedLibraries(
        source_path=source_path,
        all_libraries=loaded_libraries,
    )


def _is_library_ai_schedulable(lib: EnhancedLibraryInfo) -> bool:
    """判断 service 输入文库是否视为 AI 可排。"""
    raw_value = str(getattr(lib, "_aiavailable_raw", "") or "").strip().lower()
    if not raw_value:
        return True
    return raw_value in {"yes", "y", "true", "1", "是"}


def prepare_mode_1_1_libraries(
    *,
    data_file: Optional[Union[str, Path]] = None,
    libraries: Optional[Sequence[EnhancedLibraryInfo]] = None,
) -> Mode11PreparedLibraries:
    """将文件或内存中的文库统一整理为 1.1 service 输入。"""
    prepared = _clone_or_load_libraries(data_file=data_file, libraries=libraries)

    for lib in prepared.all_libraries:
        machine_type = getattr(lib, "machine_type", None) or _resolve_machine_type_enum_simple(
            getattr(lib, "eq_type", "")
        )
        lib.machine_type = machine_type
        if not _is_machine_supported_for_arrangement(machine_type):
            prepared.excluded_machine_libraries.append(lib)
            continue
        if _is_library_ai_schedulable(lib):
            prepared.ai_schedulable_libraries.append(lib)
        else:
            prepared.non_ai_libraries.append(lib)

    logger.info(
        "mode_1_1_service 输入准备完成: 全量={}, AI可排={}, 非AI={}, 机型排除={}",
        len(prepared.all_libraries),
        len(prepared.ai_schedulable_libraries),
        len(prepared.non_ai_libraries),
        len(prepared.excluded_machine_libraries),
    )
    return prepared


def run_mode_1_1_round1(
    *,
    data_file: Optional[Union[str, Path]] = None,
    libraries: Optional[Sequence[EnhancedLibraryInfo]] = None,
    config: Optional[Dict[str, Any]] = None,
    existing_lanes: Optional[Sequence[LaneAssignment]] = None,
    enable_priority_36t_preconsume: bool = True,
) -> Mode11Round1ServiceResult:
    """执行1.1首轮分流与排机，返回可供外部程序消费的结构化结果。"""
    resolved_config = _resolve_mode_1_1_config(config)
    prepared = prepare_mode_1_1_libraries(data_file=data_file, libraries=libraries)
    round2_handler = Mode11Round2Handler(resolved_config)
    deferred_round2_identification = round2_handler.identify_round2_candidates(
        prepared.ai_schedulable_libraries
    )
    round1_input_libraries = list(deferred_round2_identification.non_candidates)
    allocator = ModeAllocator(resolved_config)
    dispatch_result = allocator.allocate(round1_input_libraries)

    normal_libraries_for_36t: List[EnhancedLibraryInfo] = []
    priority_36t_remaining_for_1_1: List[EnhancedLibraryInfo] = []
    if enable_priority_36t_preconsume:
        preconsume_inputs = _build_priority_36t_preconsume_inputs(
            allocator=allocator,
            dispatch_result=dispatch_result,
        )
        preconsume_filler_candidates = list(preconsume_inputs.all_filler_1_1_libraries or [])
        priority_preconsume_result = _run_priority_36t_preconsume_stage(
            list(preconsume_inputs.priority_libraries),
            priority_fallback_to_36t_libraries=list(preconsume_inputs.priority_forbidden_libraries),
            filler_libraries_for_36t=list(preconsume_inputs.filler_forbidden_libraries),
            filler_libraries_from_1_1=preconsume_filler_candidates,
            max_target_lanes=int(getattr(preconsume_inputs, "max_priority_lanes", 0) or 0),
            max_filler_gb_per_lane=float(getattr(preconsume_inputs, "max_filler_gb_per_lane", 0.0) or 0.0),
        )
        if priority_preconsume_result.scheduling_succeeded:
            priority_36t_remaining_for_1_1 = list(
                getattr(priority_preconsume_result, "remaining_libraries", []) or []
            )
            priority_36t_remaining_for_1_1.extend(
                list(getattr(preconsume_inputs, "deferred_priority_libraries", []) or [])
            )
            normal_libraries_for_36t.extend(
                list(getattr(preconsume_inputs, "deferred_priority_forbidden_libraries", []) or [])
            )
            normal_libraries_for_36t.extend(
                list(getattr(priority_preconsume_result, "remaining_priority_forbidden_libraries", []) or [])
            )
            normal_libraries_for_36t.extend(
                list(getattr(priority_preconsume_result, "remaining_filler_forbidden_libraries", []) or [])
            )
            normal_libraries_for_36t.extend(
                lib
                for lib in list(dispatch_result.pool_1_1_forbidden or [])
                if id(lib) not in {
                    id(filler)
                    for filler in list(getattr(preconsume_inputs, "filler_forbidden_libraries", []) or [])
                }
            )
        else:
            normal_libraries_for_36t = (
                list(dispatch_result.pool_36t_priority) + list(dispatch_result.pool_1_1_forbidden)
            )
    else:
        logger.info(
            "mode_1_1_service 首轮快速测试模式: 跳过3.6T-NEW预消耗，仅验证1.1首轮可否成Lane"
        )
        preconsume_filler_candidates = []
        priority_preconsume_result = SimpleNamespace(
            lanes=[],
            remaining_libraries=[],
            remaining_priority_forbidden_libraries=list(dispatch_result.pool_36t_priority),
            remaining_filler_forbidden_libraries=list(dispatch_result.pool_1_1_forbidden),
            remaining_filler_1_1_libraries=[],
            scheduling_stats={},
            scheduling_succeeded=False,
            scheduling_error=None,
        )
        normal_libraries_for_36t = (
            list(dispatch_result.pool_36t_priority) + list(dispatch_result.pool_1_1_forbidden)
        )

    borrowed_fillers_from_1_1 = (
        {
            id(lib)
            for lib in list(preconsume_filler_candidates or [])
        }
        if priority_preconsume_result.scheduling_succeeded
        else set()
    )
    first_round_pool = list(priority_36t_remaining_for_1_1)
    first_round_pool.extend(
        lib for lib in list(dispatch_result.pool_1_1_normal or [])
        if id(lib) not in borrowed_fillers_from_1_1
    )
    first_round_pool.extend(
        lib for lib in list(dispatch_result.pool_1_1_quality_risk or [])
        if id(lib) not in borrowed_fillers_from_1_1
    )
    # 1.1兜底池仍属于1.1首轮池，只是不作为3.6T-NEW预消耗补料。
    first_round_pool.extend(
        lib for lib in list(dispatch_result.pool_1_1_quality_other or [])
        if id(lib) not in borrowed_fillers_from_1_1
    )
    if priority_preconsume_result.scheduling_succeeded:
        first_round_pool.extend(
            list(getattr(priority_preconsume_result, "remaining_filler_1_1_libraries", []) or [])
        )

    result = Mode11Round1ServiceResult(
        prepared_input=prepared,
        dispatch_result=dispatch_result,
        deferred_round2_identification=deferred_round2_identification,
        first_round_pool=first_round_pool,
        priority_36t_lanes=list(priority_preconsume_result.lanes or []),
        priority_36t_remaining_for_1_1=priority_36t_remaining_for_1_1,
        priority_36t_scheduling_stats=dict(priority_preconsume_result.scheduling_stats or {}),
        priority_36t_scheduling_error=priority_preconsume_result.scheduling_error,
        normal_libraries_for_36t=normal_libraries_for_36t,
        solution=SimpleNamespace(lane_assignments=[], unassigned_libraries=[]),
    )

    if deferred_round2_identification.total_candidates > 0:
        logger.info(
            "mode_1_1_service 首轮排机前置剥离第二轮候选: 文库={}, 分组={}",
            deferred_round2_identification.total_candidates,
            len(deferred_round2_identification.candidate_groups),
        )

    if not first_round_pool:
        result.scheduling_succeeded = True
        logger.info("mode_1_1_service 首轮排机跳过: 无可进入1.1首轮池的文库")
        return result

    for lib in first_round_pool:
        lib._current_seq_mode_raw = "1.1"

    first_round_label = str(resolved_config.get("first_round_label", "1.1第一轮"))
    try:
        balanced_existing_lanes, balanced_remaining_pool, balanced_lane_stats = (
            _build_mode_1_1_first_round_balanced_25b_lanes(first_round_pool)
        )
        scheduling_input_pool = (
            balanced_remaining_pool if balanced_existing_lanes else list(first_round_pool)
        )
        scheduling_existing_lanes = list(existing_lanes or []) + list(balanced_existing_lanes)
        scheduling_stats, solution = test_with_model(
            deepcopy(scheduling_input_pool),
            existing_lanes=deepcopy(scheduling_existing_lanes),
            enable_expensive_rescue=False,
            enable_peak_window_mixed_lanes=True,
        )
        scheduling_stats = dict(scheduling_stats or {})
        scheduling_stats.update(balanced_lane_stats)
        for lane in solution.lane_assignments:
            if not isinstance(lane.metadata, dict):
                lane.metadata = {}
            lane.metadata["dispatch_stage"] = "first_round_1_1"
            lane.metadata["selected_seq_mode"] = "1.1"
            lane.metadata["selected_round_label"] = first_round_label

        result.lanes = list(solution.lane_assignments or [])
        result.fallback_libraries_for_36t = list(solution.unassigned_libraries or [])
        result.normal_libraries_for_36t.extend(result.fallback_libraries_for_36t)
        result.scheduling_stats = dict(scheduling_stats or {})
        result.solution = solution
        result.scheduling_succeeded = True
        logger.info(
            "mode_1_1_service 首轮排机完成: 1.1 Lane={}, 回流3.6T文库={}",
            len(result.lanes),
            len(result.fallback_libraries_for_36t),
        )
    except Exception as exc:
        result.scheduling_error = str(exc)
        result.fallback_libraries_for_36t = list(first_round_pool)
        result.normal_libraries_for_36t.extend(first_round_pool)
        logger.error("mode_1_1_service 首轮排机异常，已回退到3.6T-NEW: {}", exc)
    finally:
        for lib in first_round_pool:
            lib._current_seq_mode_raw = ""

    return result


def run_mode_1_1_round2(
    *,
    data_file: Optional[Union[str, Path]] = None,
    libraries: Optional[Sequence[EnhancedLibraryInfo]] = None,
    config: Optional[Dict[str, Any]] = None,
    schedule_candidates: bool = False,
) -> Mode11Round2ServiceResult:
    """执行1.1第二轮候选识别，按需触发当前预留的第二轮排机入口。"""
    resolved_config = _resolve_mode_1_1_config(config)
    prepared = prepare_mode_1_1_libraries(data_file=data_file, libraries=libraries)
    handler = Mode11Round2Handler(resolved_config)
    identification_result = handler.identify_round2_candidates(prepared.ai_schedulable_libraries)
    scheduling_result: Optional[Round2SchedulingResult] = None

    if schedule_candidates and identification_result.total_candidates > 0:
        scheduling_result = handler.schedule_round2(identification_result.candidate_groups)

    return Mode11Round2ServiceResult(
        prepared_input=prepared,
        identification_result=identification_result,
        scheduling_result=scheduling_result,
        schedule_invoked=bool(schedule_candidates and identification_result.total_candidates > 0),
    )


def run_mode_1_1_round2_export(
    *,
    data_file: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    output_detail_dir: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
) -> Mode11Round2ExportResult:
    """仅执行1.1第二轮识别、直出与预测后处理，避免触发整条主流程。"""
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    resolved_config = _resolve_mode_1_1_config(config)
    prepared = prepare_mode_1_1_libraries(data_file=data_path)
    handler = Mode11Round2Handler(resolved_config)
    identification_result = handler.identify_round2_candidates(prepared.ai_schedulable_libraries)
    output_path = _resolve_round2_export_output_path(
        data_path=data_path,
        output_detail_dir=output_detail_dir,
        output_file=output_file,
    )

    source_df = _read_csv_with_encoding_fallback(data_path)
    source_had_origrec_key = "origrec_key" in source_df.columns
    source_df["origrec_key"] = _build_origrec_key(source_df)

    scheduling_result: Optional[Round2SchedulingResult] = None
    if identification_result.total_candidates > 0:
        scheduling_result = handler.schedule_round2(identification_result.candidate_groups)

    exported_lanes = list(getattr(scheduling_result, "lanes", []) or [])
    exported_lane_count = len(exported_lanes)
    exported_libraries = _collect_detail_output_libraries(
        SimpleNamespace(lane_assignments=exported_lanes, unassigned_libraries=[])
    )
    exported_library_count = len(exported_libraries)

    if not exported_lanes:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        passthrough_df = source_df.copy()
        if not source_had_origrec_key:
            passthrough_df = passthrough_df.drop(columns=["origrec_key"], errors="ignore")
        passthrough_df.to_csv(output_path, index=False)
        logger.info("1.1第二轮轻量导出跳过: 无可导出的第二轮Lane, 原始文件已原样写出 {}", output_path)
        return Mode11Round2ExportResult(
            output_path=output_path,
            prepared_input=prepared,
            identification_result=identification_result,
            scheduling_result=scheduling_result,
            exported_lane_count=0,
            exported_library_count=0,
            predicted_row_count=0,
        )

    loutput_series = (
        pd.to_numeric(source_df["loutput"], errors="coerce")
        if "loutput" in source_df.columns
        else pd.Series([pd.NA] * len(source_df))
    )
    loutput_by_origrec = dict(zip(source_df["origrec_key"], loutput_series))
    pred_df = _collect_prediction_rows(exported_lanes, loutput_by_origrec, "mode_1_1_round2_export")
    target_keys = {
        _safe_str(getattr(lib, "_source_origrec_key", getattr(lib, "origrec", "")), default="")
        for lib in exported_libraries
    }
    target_keys = {item for item in target_keys if item}
    target_raw_df = source_df.loc[source_df["origrec_key"].astype(str).isin(target_keys)].copy()

    temp_prediction_output = output_path.parent / f".{output_path.stem}.mode11_round2_targets.tmp.csv"
    lanes_with_split = _collect_lanes_with_split(exported_lanes)
    _build_detail_output(
        df_raw=target_raw_df,
        pred_df=pred_df,
        output_path=temp_prediction_output,
        ai_schedulable_keys=target_keys,
        lanes_with_split=lanes_with_split,
        detail_libraries=exported_libraries,
    )

    target_detail_df = _read_csv_with_encoding_fallback(temp_prediction_output)
    target_prediction_df = _run_prediction_delivery(
        input_data=target_detail_df,
        output_path=temp_prediction_output,
    )

    base_output_df = _clear_current_arrangement_fields_for_round2_export(source_df)
    merged_df = _merge_round2_export_rows(source_df=base_output_df, target_df=target_prediction_df)
    if not source_had_origrec_key:
        merged_df = merged_df.drop(columns=["origrec_key"], errors="ignore")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    if temp_prediction_output.exists():
        temp_prediction_output.unlink()

    logger.info(
        "1.1第二轮轻量导出完成: 分组={}, Lane={}, 文库={}, 输出文件={}",
        len(identification_result.candidate_groups),
        exported_lane_count,
        exported_library_count,
        output_path,
    )
    return Mode11Round2ExportResult(
        output_path=output_path,
        prepared_input=prepared,
        identification_result=identification_result,
        scheduling_result=scheduling_result,
        exported_lane_count=exported_lane_count,
        exported_library_count=exported_library_count,
        predicted_row_count=len(target_prediction_df),
    )


def run_mode_1_1_full(
    *,
    data_file: Union[str, Path],
    output_detail_dir: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
) -> Mode11FullServiceResult:
    """执行当前已集成到主流程中的1.1全链路编排。"""
    output_path = arrange_library(
        data_file=data_file,
        mode="arrange",
        output_detail_dir=output_detail_dir,
        output_file=output_file,
    )
    return Mode11FullServiceResult(output_path=Path(output_path), mode="arrange")


__all__ = [
    "Mode11FullServiceResult",
    "Mode11PreparedLibraries",
    "Mode11Round1ServiceResult",
    "Mode11Round2ExportResult",
    "Mode11Round2ServiceResult",
    "prepare_mode_1_1_libraries",
    "run_mode_1_1_full",
    "run_mode_1_1_round1",
    "run_mode_1_1_round2",
    "run_mode_1_1_round2_export",
]
