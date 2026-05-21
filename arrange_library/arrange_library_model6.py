"""
端到端排机流程测试 - 排机与 Pooling 预测
创建时间：2026-04-10 16:06:41
更新时间：2026-05-12 16:35:00

功能：
- 支持完整排机流程（GreedyLaneScheduler）
- 支持仅执行 Pooling 预测，不再重复排机
- 排机模式下先完成排机，再调用 prediction_delivery 输出下单量与产出量
- 预测模式下直接对已排机文件调用 prediction_delivery

变更记录：
- 2026-03-16: 移除脚本内置 Pooling 预测实现，统一改为调用 prediction_delivery
- 2026-03-16: 新增 mode 参数
             - arrange：加载数据、排机、预测，全流程执行
             - pooling：仅对已排机结果执行预测
- 2026-01-30: 字段映射精简，以表中实际字段名为准（wk前缀）
"""

import argparse
import json
import math
import random
import re
import signal
import sys
from collections import Counter
from dataclasses import dataclass, field
from copy import deepcopy
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd
import warnings

# 全局关闭 pandas 的 DataFrame 高度碎片化性能告警（来自 prediction_delivery 内部）
warnings.filterwarnings(
    "ignore",
    category=pd.errors.PerformanceWarning,
    message="DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.",
)

# 添加项目路径（包外：将 arrange_library 的上一级目录加入 sys.path）
# 这样 `from arrange_library...` 的绝对导入在脚本直跑场景下也稳定可用。
package_parent_dir = str(Path(__file__).resolve().parent.parent)
if package_parent_dir not in sys.path:
    sys.path.insert(0, package_parent_dir)

# from arrange_library.liblane_paths import setup_liblane_paths

# setup_liblane_paths()

from loguru import logger

from arrange_library.models.library_info import EnhancedLibraryInfo, MachineType
from arrange_library.core.config.scheduling_config import get_scheduling_config
from arrange_library.core.scheduling.mode_allocator import ModeAllocator, ModeDispatchResult
from arrange_library.core.scheduling.mode_1_1_round2 import Mode11Round2Handler
from arrange_library.core.constraints.lane_validator import (
    LaneValidator,
    LaneValidationResult,
    ValidationRuleType,
    ValidationError,
    ValidationSeverity,
)
from arrange_library.core.constraints.index_validator_verified import IndexConflictValidator as _IndexConflictValidator

# 模块级单例，避免在 _attempt_build_lane_from_pool 等高频函数中反复初始化
_MODULE_IDX_VALIDATOR = _IndexConflictValidator()
_AUTO_LANE_SERIAL_COUNTERS: Dict[Tuple[str, str], int] = {}
_IMBALANCE_LIBRARY_CANDIDATE_CACHE: Dict[str, bool] = {}
_LANE_IMBALANCE_SUMMARY_CACHE: Dict[Tuple[str, ...], Tuple[float, float, float]] = {}
_LANE_57_MIX_RULE_CACHE: Dict[Tuple[Tuple[str, ...], bool, str], Tuple[bool, str]] = {}
_LANE_CAPACITY_SELECTION_CACHE: Dict[
    Tuple[str, Tuple[int, ...], Tuple[Tuple[str, str], ...]],
    Any,
] = {}
_QUICK_INDEX_VALIDATION_RESULT_CACHE: Dict[
    Tuple[Tuple[str, ...], str],
    Tuple[bool, Tuple[Tuple[str, Optional[str]], ...]],
] = {}
_SCATTERED_MIX_SORT_CACHE: Dict[Tuple[int, ...], Tuple[EnhancedLibraryInfo, ...]] = {}
_RESCUE_RULE_BUCKET_CACHE: Dict[Tuple[str, Tuple[str, ...]], List[Tuple[str, List[EnhancedLibraryInfo]]]] = {}
_MAIN_RULE_BUCKET_SCHEDULE_CACHE: Dict[
    Tuple[Tuple[str, ...], bool],
    Tuple[List[Any], List[EnhancedLibraryInfo], Set[str], int],
] = {}
_BUCKET_TOTAL_DATA_CACHE: Dict[Tuple[str, ...], float] = {}
_PRIORITY_36T_PROFILE_SUMMARY_CACHE: Dict[Tuple[int, ...], Dict[str, float]] = {}
_PRIORITY_36T_PEAK_COMPAT_CACHE: Dict[Tuple[int, ...], bool] = {}
_LIGHT_POOL_FEASIBILITY_CACHE: Dict[
    Tuple[str, Tuple[Tuple[str, str], ...], Tuple[str, ...]],
    Tuple[bool, str],
] = {}
_RESCUE_VARIANT_ATTEMPT_CACHE: Dict[
    Tuple[str, str, str, Tuple[Tuple[str, str], ...], Tuple[int, ...], bool],
    Tuple[bool, str],
] = {}
_RESCUE_STRICT_FAILURE_SIGNATURES: Dict[
    Tuple[str, str, str, Tuple[Tuple[str, str], ...]],
    List[Tuple[int, ...]],
] = {}
_TERMINAL_DEDICATED_HARD_SKIP_CACHE: Dict[
    Tuple[str, str, Tuple[Tuple[str, str], ...], Tuple[int, ...]],
    str,
] = {}
_COMPACT_LIBRARY_IDENTITY_BY_KEY: Dict[str, int] = {}
_COMPACT_LIBRARY_IDENTITY_NEXT = 0
from arrange_library.core.data import load_libraries_from_csv
from arrange_library.core.preprocessing.base_imbalance_handler import BaseImbalanceHandler
from arrange_library.core.preprocessing.library_splitter import LibrarySplitter
from arrange_library.core.preprocessing.rule_constrained_strategy_planner import StrategyExecutionPlan
from arrange_library.core.preprocessing.batch_rule_analyzer import BatchAnalysisReport
from arrange_library.core.scheduling.greedy_lane_scheduler import GreedyLaneScheduler, GreedyLaneConfig
from arrange_library.core.scheduling.package_lane_scheduler import PackageLaneScheduler
from arrange_library.core.scheduling.scheduling_types import LaneAssignment

# prediction_delivery 作为独立包依赖，由 pip 安装后直接导入
from prediction_delivery import MODELS_DIR, predict_pooling

# ==================== 排机超时控制 ====================
# 排机最长允许运行时间（秒）。超过此时间视为异常，强制中断并返回失败。
SCHEDULING_TIMEOUT_SECONDS = 600  # 10 分钟


class SchedulingTimeoutError(Exception):
    """排机超时异常：排机耗时超过允许上限，强制终止。"""
    pass


@dataclass
class RollbackMode11ScheduleResult:
    """拆分回滚原始文库回流1.1排机结果。"""

    lanes: List[LaneAssignment] = field(default_factory=list)
    remaining_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)


def _scheduling_timeout_handler(signum: int, frame: object) -> None:
    """SIGALRM 信号处理器，超时时抛出 SchedulingTimeoutError。"""
    raise SchedulingTimeoutError(
        f"排机超时：超过 {SCHEDULING_TIMEOUT_SECONDS // 60} 分钟仍未完成，已强制终止"
    )


# ==================== Lane上机浓度规则 ====================
LANE_ORDERDATA_FLOOR = 1.0
SPECIAL_SPLIT_GROUP_A: Set[str] = {
    "10x_longranger",
    "10x_longranger_indexset",
    "10x_cellranger",
    "10x_cellranger_indexset",
}
SPECIAL_SPLIT_GROUP_B: Set[str] = {
    "10x_cellranger-atac_indexset",
    "10x_cellranger-atac",
}
SCHEDULING_MAX_TARGET_CAP_GB = 1100.0
SCHEDULING_MAX_EFFECTIVE_CAP_GB = 1105.0
ROLLBACK_SPLIT_LIBRARY_MODE_1_1_MAX_GB = 500.0
LANE_SEQ_10_PLUS_24_BALANCE_RATIO = 0.05
LANE_SEQ_10_PLUS_24_LANE_PREFIX = "LS"
LANE_SEQ_10_PLUS_24_TARGET_TOTAL_GB = 1000.0
LANE_SEQ_10_PLUS_24_TOLERANCE_GB = 5.0
LANE_SEQ_10_PLUS_24_BALANCE_DENOMINATOR = 1.0 - LANE_SEQ_10_PLUS_24_BALANCE_RATIO
MIN_BALANCE_RATIO_DENOMINATOR = 1e-9
PRIORITY_36T_PRECONSUME_LANE_FLOOR_GB = 995.0
PRIORITY_36T_PRECONSUME_FILLER_MAX_GB_PER_LANE = 250.0
PRIORITY_36T_PRECONSUME_MULTI_LANE_CLINICAL_MAX_GROUP_SHARE = 0.6
PRIORITY_36T_PRECONSUME_RETRY_WINDOW_TARGETS_GB: Tuple[float, ...] = (
    700.0,
    500.0,
    850.0,
    950.0,
    1100.0,
    1200.0,
    1500.0,
    1800.0,
    2200.0,
)
PRIORITY_36T_PRECONSUME_RETRY_FILLER_TARGETS_GB: Tuple[float, ...] = (
    250.0,
    400.0,
    600.0,
    800.0,
    1000.0,
    0.0,
)
SCHEDULING_CAP_RULE_CODES: Set[str] = {
    "tj_1595_standard_pe150_25b",
    "tj_1595_standard_pe150_25b_other",
}
DEFAULT_INDEX_CONFLICT_ATTEMPTS = 10
DEFAULT_OTHER_FAILURE_ATTEMPTS = 20
DEFAULT_EX_RESCUE_MAX_NEW_LANES = 2
DEFAULT_RB_RESCUE_MAX_NEW_LANES = 1
ZERO_LANE_RESCUE_SKIP_LIB_THRESHOLD = 200
LARGE_POOL_RESCUE_SKIP_LIB_THRESHOLD = 1500
LARGE_POOL_RESCUE_SKIP_DATA_GB = 15000.0
MODE_1_1_POST_RESCUE_SKIP_LIB_THRESHOLD = 1000
MODE_1_1_POST_RESCUE_SKIP_DATA_GB = 50000.0
SCATTERED_MIX_IMBALANCE_TARGET_RATIO = 0.35
SCATTERED_MIX_IMBALANCE_TARGET_EPSILON = 1e-6
SPECIAL_LIBRARY_LIMIT_EPSILON = 1e-6
SCATTERED_MIX_VARIANT_ATTEMPTS = 1
RM_SCATTERED_MIX_VARIANT_ATTEMPTS = 4
INDEX_RULE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "merge_deal" / "config"
BALANCE_LIBRARY_CONFIG_PATH = Path(__file__).resolve().parent / "AI排机-平衡文库.csv"
BALANCE_LIBRARY_MARKER_COLUMN = "_is_ai_balance_library"
REDUNDANT_OUTPUT_COLUMNS: Tuple[str, ...] = (
    "predicted_lorderdata",
    "ai_predicted_lorderdata",
    "ai_predicted_loutput",
)
PACKAGE_LANE_TARGET_GB = 1000.0
PACKAGE_LANE_TOLERANCE_GB = 0.01
PACKAGE_LANE_MIN_GB = PACKAGE_LANE_TARGET_GB - PACKAGE_LANE_TOLERANCE_GB
PACKAGE_LANE_MAX_GB = PACKAGE_LANE_TARGET_GB + PACKAGE_LANE_TOLERANCE_GB
PACKAGE_LANE_MIN_INDEX_PAIRS = 5
AI_LANE_MIN_INDEX_PAIRS = 5
_BASE_IMBALANCE_HANDLER = BaseImbalanceHandler()


def _normalize_text_for_match(value: Any) -> str:
    """统一文本匹配口径，消除引号、加号、大小写等格式差异。"""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("＇", "'")
        .replace("＋", "+")
        .replace("×", "X")
        .upper()
    )


def _machine_type_to_text(machine_type: Any, default: str = "") -> str:
    """将机型对象统一转换为业务文本，兼容不同模块中的同名枚举。"""
    if machine_type is None:
        return default
    value = getattr(machine_type, "value", machine_type)
    text = str(value).strip()
    return text or default


def _reset_auto_lane_serial_counters() -> None:
    """重置自动Lane编号计数器。"""
    _AUTO_LANE_SERIAL_COUNTERS.clear()


def _drop_redundant_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """移除不再对外输出的预测中间列。"""
    return df.drop(columns=list(REDUNDANT_OUTPUT_COLUMNS), errors="ignore")


def _reserve_auto_lane_serial(
    lane_id_prefix: str,
    machine_type: MachineType | str,
) -> int:
    """为未显式指定 serial 的Lane分配全局唯一编号。"""
    key = (str(lane_id_prefix).strip(), _machine_type_to_text(machine_type, default="Nova X-25B"))
    _AUTO_LANE_SERIAL_COUNTERS[key] = _AUTO_LANE_SERIAL_COUNTERS.get(key, 0) + 1
    return _AUTO_LANE_SERIAL_COUNTERS[key]


def _ensure_unique_lane_ids(lanes: List[Any]) -> int:
    """确保最终Lane ID全局唯一，避免导出时不同Lane被同一个llaneid合并。"""
    seen_ids: Set[str] = set()
    renamed = 0

    for lane in lanes:
        original_lane_id = str(getattr(lane, "lane_id", "") or "").strip()
        if not original_lane_id:
            continue
        if original_lane_id not in seen_ids:
            seen_ids.add(original_lane_id)
            continue

        parts = original_lane_id.rsplit("_", 2)
        candidate_lane_id = ""
        if len(parts) == 3 and parts[2].isdigit():
            prefix, machine_text, _ = parts
            next_serial = 1
            while True:
                candidate_lane_id = f"{prefix}_{machine_text}_{next_serial:03d}"
                if candidate_lane_id not in seen_ids:
                    break
                next_serial += 1
        else:
            suffix = 2
            while True:
                candidate_lane_id = f"{original_lane_id}__dup{suffix}"
                if candidate_lane_id not in seen_ids:
                    break
                suffix += 1

        logger.warning(
            "检测到重复Lane ID，已自动重命名: {} -> {}",
            original_lane_id,
            candidate_lane_id,
        )
        lane.lane_id = candidate_lane_id
        if getattr(lane, "machine_id", None):
            lane.machine_id = f"M_{candidate_lane_id}"
        seen_ids.add(candidate_lane_id)
        renamed += 1

    return renamed


def _deduplicate_solution_libraries(solution: Any) -> Dict[str, int]:
    """按稳定文库身份键去重，防止同一文库在多个Lane或未分配池重复出现。"""
    lane_assignments = list(getattr(solution, "lane_assignments", []) or [])
    unassigned_libraries = list(getattr(solution, "unassigned_libraries", []) or [])

    seen_assigned_keys: Set[str] = set()
    cleaned_lanes: List[Any] = []
    removed_assigned_duplicates = 0
    removed_empty_lanes = 0

    for lane in lane_assignments:
        original_libraries = list(getattr(lane, "libraries", []) or [])
        unique_libraries: List[EnhancedLibraryInfo] = []
        for lib in original_libraries:
            identity_key = _safe_str(_get_library_identity_key(lib), default="")
            if not identity_key:
                identity_key = f"obj:{id(lib)}"
            if identity_key in seen_assigned_keys:
                removed_assigned_duplicates += 1
                continue
            seen_assigned_keys.add(identity_key)
            unique_libraries.append(lib)
        lane.libraries = unique_libraries
        if unique_libraries:
            cleaned_lanes.append(lane)
        else:
            removed_empty_lanes += 1

    seen_unassigned_keys: Set[str] = set()
    cleaned_unassigned: List[EnhancedLibraryInfo] = []
    removed_unassigned_assigned_overlap = 0
    removed_unassigned_duplicates = 0
    for lib in unassigned_libraries:
        identity_key = _safe_str(_get_library_identity_key(lib), default="")
        if not identity_key:
            identity_key = f"obj:{id(lib)}"
        if identity_key in seen_assigned_keys:
            removed_unassigned_assigned_overlap += 1
            continue
        if identity_key in seen_unassigned_keys:
            removed_unassigned_duplicates += 1
            continue
        seen_unassigned_keys.add(identity_key)
        cleaned_unassigned.append(lib)

    solution.lane_assignments = cleaned_lanes
    solution.unassigned_libraries = cleaned_unassigned

    return {
        "removed_assigned_duplicates": removed_assigned_duplicates,
        "removed_empty_lanes": removed_empty_lanes,
        "removed_unassigned_assigned_overlap": removed_unassigned_assigned_overlap,
        "removed_unassigned_duplicates": removed_unassigned_duplicates,
    }


def _normalize_seq_strategy_keyword(value: Any) -> str:
    """统一测序策略匹配口径。"""
    return _normalize_text_for_match(value).replace("BP", "").replace(" ", "")


def _clear_imbalance_helper_caches() -> None:
    """清空混排不均衡辅助缓存，供测试和长流程边界场景复用。"""
    _IMBALANCE_LIBRARY_CANDIDATE_CACHE.clear()
    _LANE_IMBALANCE_SUMMARY_CACHE.clear()
    _LANE_57_MIX_RULE_CACHE.clear()
    _LANE_CAPACITY_SELECTION_CACHE.clear()
    _QUICK_INDEX_VALIDATION_RESULT_CACHE.clear()
    _SCATTERED_MIX_SORT_CACHE.clear()
    _COMPACT_LIBRARY_IDENTITY_BY_KEY.clear()


def _is_imbalance_library_candidate(lib: EnhancedLibraryInfo) -> bool:
    """统一判断碱基不均衡文库，避免多处直接访问处理器细节。"""
    cache_key = _safe_str(
        getattr(lib, "_detail_output_key", getattr(lib, "_origrec_key", getattr(lib, "origrec", ""))),
        default="",
    ) or str(id(lib))
    cached = _IMBALANCE_LIBRARY_CANDIDATE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        result = bool(_BASE_IMBALANCE_HANDLER.is_imbalance_library(lib))
    except Exception:
        result = False
    _IMBALANCE_LIBRARY_CANDIDATE_CACHE[cache_key] = result
    return result


def _summarize_lane_imbalance(
    libraries: List[EnhancedLibraryInfo],
) -> Tuple[float, float, float]:
    """统计当前Lane总量、不均衡总量及占比。"""
    if not libraries:
        return 0.0, 0.0, 0.0
    cache_key = tuple(
        _safe_str(
            getattr(lib, "_detail_output_key", getattr(lib, "_origrec_key", getattr(lib, "origrec", ""))),
            default="",
        ) or str(id(lib))
        for lib in libraries
    )
    cached = _LANE_IMBALANCE_SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    total_data = 0.0
    imbalance_data = 0.0
    for lib in libraries:
        data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
        total_data += data
        if _is_imbalance_library_candidate(lib):
            imbalance_data += data
    ratio = imbalance_data / total_data if total_data > 0 else 0.0
    result = (total_data, imbalance_data, ratio)
    _LANE_IMBALANCE_SUMMARY_CACHE[cache_key] = result
    return result


def _resolve_lane_imbalance_summary(
    libraries: List[EnhancedLibraryInfo],
    lane_summary: Optional[Tuple[float, float, float]] = None,
) -> Tuple[float, float, float]:
    if lane_summary is not None:
        return lane_summary
    return _summarize_lane_imbalance(libraries)


def _resolve_candidate_imbalance_flag(
    lib: EnhancedLibraryInfo,
    candidate_is_imbalance: Optional[bool] = None,
) -> bool:
    if candidate_is_imbalance is not None:
        return bool(candidate_is_imbalance)
    return _is_imbalance_library_candidate(lib)


def _resolve_special_library_data_limit(machine_type: MachineType | str) -> float:
    """获取当前机型的特殊文库总量上限。"""
    machine_key = machine_type.value if isinstance(machine_type, MachineType) else str(machine_type)
    return float(get_scheduling_config().get_special_library_limit(machine_key))


def _project_lane_special_data(
    current_libraries: List[EnhancedLibraryInfo],
    candidate: EnhancedLibraryInfo,
    *,
    lane_summary: Optional[Tuple[float, float, float]] = None,
    candidate_is_imbalance: Optional[bool] = None,
) -> float:
    """计算加入候选文库后的特殊文库总量。"""
    _, imbalance_data, _ = _resolve_lane_imbalance_summary(current_libraries, lane_summary)
    if _resolve_candidate_imbalance_flag(candidate, candidate_is_imbalance):
        imbalance_data += float(getattr(candidate, "contract_data_raw", 0.0) or 0.0)
    return imbalance_data


def _project_lane_imbalance_ratio(
    current_libraries: List[EnhancedLibraryInfo],
    candidate: EnhancedLibraryInfo,
    *,
    lane_summary: Optional[Tuple[float, float, float]] = None,
    candidate_is_imbalance: Optional[bool] = None,
) -> float:
    """计算加入候选文库后的碱基不均衡占比。"""
    total_data, imbalance_data, _ = _resolve_lane_imbalance_summary(current_libraries, lane_summary)
    candidate_data = float(getattr(candidate, "contract_data_raw", 0.0) or 0.0)
    projected_total = total_data + candidate_data
    if projected_total <= 0:
        return 0.0
    if _resolve_candidate_imbalance_flag(candidate, candidate_is_imbalance):
        imbalance_data += candidate_data
    return imbalance_data / projected_total


def _imbalance_target_distance(
    ratio: float,
    target_ratio: float = SCATTERED_MIX_IMBALANCE_TARGET_RATIO,
) -> float:
    """返回当前占比距离目标占比的绝对偏差。"""
    return abs(float(ratio) - float(target_ratio))


def _candidate_improves_imbalance_target(
    current_libraries: List[EnhancedLibraryInfo],
    candidate: EnhancedLibraryInfo,
    *,
    target_ratio: float = SCATTERED_MIX_IMBALANCE_TARGET_RATIO,
    machine_type: Optional[MachineType | str] = None,
    special_data_limit: Optional[float] = None,
    lane_summary: Optional[Tuple[float, float, float]] = None,
    candidate_is_imbalance: Optional[bool] = None,
) -> bool:
    """判断候选文库是否会让Lane的不均衡占比更接近目标值。"""
    effective_special_limit = (
        float(special_data_limit)
        if special_data_limit is not None
        else (
            _resolve_special_library_data_limit(machine_type)
            if machine_type is not None
            else None
        )
    )
    resolved_lane_summary = _resolve_lane_imbalance_summary(current_libraries, lane_summary)
    projected_special_data = _project_lane_special_data(
        current_libraries,
        candidate,
        lane_summary=resolved_lane_summary,
        candidate_is_imbalance=candidate_is_imbalance,
    )
    if (
        effective_special_limit is not None
        and projected_special_data > effective_special_limit + SPECIAL_LIBRARY_LIMIT_EPSILON
    ):
        return False
    _, _, current_ratio = resolved_lane_summary
    projected_ratio = _project_lane_imbalance_ratio(
        current_libraries,
        candidate,
        lane_summary=resolved_lane_summary,
        candidate_is_imbalance=candidate_is_imbalance,
    )
    return (
        _imbalance_target_distance(projected_ratio, target_ratio)
        + SCATTERED_MIX_IMBALANCE_TARGET_EPSILON
        < _imbalance_target_distance(current_ratio, target_ratio)
    )


def _build_imbalance_fill_sort_key(
    current_libraries: List[EnhancedLibraryInfo],
    lib: EnhancedLibraryInfo,
    *,
    target_ratio: float = SCATTERED_MIX_IMBALANCE_TARGET_RATIO,
    machine_type: Optional[MachineType | str] = None,
    special_data_limit: Optional[float] = None,
    lane_summary: Optional[Tuple[float, float, float]] = None,
    candidate_is_imbalance: Optional[bool] = None,
) -> Tuple[int, int, float, float, float]:
    """构建“尽量逼近不均衡目标占比”的排序键。"""
    data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
    if not current_libraries:
        return (0, 1, 0.0, 0.0, -data)
    resolved_lane_summary = _resolve_lane_imbalance_summary(current_libraries, lane_summary)
    _, _, current_ratio = resolved_lane_summary
    projected_ratio = _project_lane_imbalance_ratio(
        current_libraries,
        lib,
        lane_summary=resolved_lane_summary,
        candidate_is_imbalance=candidate_is_imbalance,
    )
    resolved_candidate_is_imbalance = _resolve_candidate_imbalance_flag(
        lib,
        candidate_is_imbalance,
    )
    return (
        0 if _candidate_improves_imbalance_target(
            current_libraries,
            lib,
            target_ratio=target_ratio,
            machine_type=machine_type,
            special_data_limit=special_data_limit,
            lane_summary=resolved_lane_summary,
            candidate_is_imbalance=resolved_candidate_is_imbalance,
        ) else 1,
        0 if (resolved_candidate_is_imbalance and current_ratio < target_ratio) else 1,
        _imbalance_target_distance(projected_ratio, target_ratio),
        max(projected_ratio - target_ratio, 0.0),
        -data,
    )


LANE_LOADING_COMBO_GROUP_A = {
    _normalize_text_for_match(item)
    for item in [
        "10X转录组-5'文库",
        "10X转录组文库-5V3文库",
        "10X转录组V(D)J-BCR文库",
        "10X转录组V(D)J-TCR文库",
        "客户-10X VDJ文库",
        "10X转录组-5‘膜蛋白文库",
        "客户-10X 5 Feature Barcode文库",
        "客户-10X 5 单细胞转录组文库",
        "客户-10X转录组V(D)J-BCR文库",
        "客户-10X转录组V(D)J-TCR文库",
    ]
}
LANE_LOADING_COMBO_GROUP_B = {
    _normalize_text_for_match(item)
    for item in [
        "10X Visium FFPEV2空间转录组文库(V2)",
        "10X Visium空间转录组文库",
        "10X转录组-3‘膜蛋白文库",
        "客户-10X 3 Feature Barcode文库",
        "客户-10X 3 单细胞转录组文库",
        "客户文库-10X Visium 文库",
        "客户-10X Visium FFPEV2空间转录组文库(V2)",
        "客户-10X Visium空间转录组文库",
        "墨卓转录组-3端文库",
        "10X转录组-3'文库",
        "10X转录组文库-3V4文库",
    ]
}
LANE_LOADING_10_PLUS_24_ATAC_TYPES = {
    _normalize_text_for_match(item)
    for item in [
        "客户-10X ATAC文库",
        "客户-10X ATAC (Multiome)文库",
        "10xATAC-seq文库",
    ]
}


class ConflictType(Enum):
    """Index冲突类型。"""

    SINGLE_SINGLE = "single_single"
    DUAL_DUAL = "dual_dual"
    SINGLE_DUAL = "single_dual"


@dataclass
class LatestIndexConflict:
    """最新Index冲突详情。"""

    record_id_1: str
    record_id_2: str
    conflict_type: ConflictType
    same_count_left: int
    same_count_right: Optional[int] = None


def _parse_index_pairs_latest(index_seq: str) -> List[Tuple[str, Optional[str]]]:
    """按最新规则解析Index字符串为[(P7, P5), ...]。"""
    if not index_seq:
        return []

    text = str(index_seq).strip()
    if not text or text.upper() == "NO INDEX":
        return []

    parsed: List[Tuple[str, Optional[str]]] = []
    items = [item.strip() for item in text.split(",") if item.strip()]
    for item in items:
        if ";" in item:
            parts = [part.strip() for part in item.split(";") if part.strip()]
            if len(parts) == 2:
                parsed.append((parts[0], parts[1]))
            elif len(parts) == 1:
                parsed.append((parts[0], None))
        else:
            parsed.append((item, None))
    return parsed


def _side_is_repeated_aligned_latest(seq_1: str, seq_2: str) -> Tuple[bool, int]:
    """最新规则：L<=8时same>(L-2)，L>8时same>7。"""
    s1 = (seq_1 or "").strip().upper()
    s2 = (seq_2 or "").strip().upper()
    if not s1 or not s2:
        return False, 0

    length = min(len(s1), len(s2))
    s1_cut = s1[:length]
    s2_cut = s2[:length]
    same = sum(1 for a, b in zip(s1_cut, s2_cut) if a == b)
    threshold = min(length - 2, 7)
    return same > threshold, same


def _side_is_repeated_left_latest(seq_1: str, seq_2: str) -> Tuple[bool, int]:
    """P7按左对齐比较。"""
    return _side_is_repeated_aligned_latest(seq_1, seq_2)


def _side_is_repeated_right_latest(seq_1: str, seq_2: str) -> Tuple[bool, int]:
    """P5按右对齐比较。"""
    s1 = (seq_1 or "").strip().upper()
    s2 = (seq_2 or "").strip().upper()
    if not s1 or not s2:
        return False, 0

    length = min(len(s1), len(s2))
    s1_cut = s1[-length:]
    s2_cut = s2[-length:]
    return _side_is_repeated_aligned_latest(s1_cut, s2_cut)


def _check_index_pair_repeat_latest(
    left_1: str,
    right_1: Optional[str],
    left_2: str,
    right_2: Optional[str],
) -> Tuple[bool, Optional[ConflictType], int, Optional[int]]:
    """按最新规则检查两个Index对是否重复。"""
    left_repeat, same_left = _side_is_repeated_left_latest(left_1, left_2)
    if not left_repeat:
        return False, None, same_left, None

    if right_1 is None and right_2 is None:
        return True, ConflictType.SINGLE_SINGLE, same_left, None

    if right_1 is not None and right_2 is not None:
        right_repeat, same_right = _side_is_repeated_right_latest(right_1, right_2)
        if right_repeat:
            return True, ConflictType.DUAL_DUAL, same_left, same_right
        return False, None, same_left, same_right

    return True, ConflictType.SINGLE_DUAL, same_left, None


def _validate_index_conflicts_latest(libraries: List[EnhancedLibraryInfo]) -> List[LatestIndexConflict]:
    """对Lane内文库执行最新Index冲突检查。"""
    if len(libraries) < 2:
        return []

    parsed_records: List[Tuple[EnhancedLibraryInfo, str, List[Tuple[str, Optional[str]]]]] = []
    for lib in libraries:
        record_id = _get_library_identity_key(lib)
        index_seq = str(getattr(lib, "index_seq", "") or "")
        pairs = _parse_index_pairs_latest(index_seq)
        if pairs:
            parsed_records.append((lib, record_id, pairs))

    conflicts: List[LatestIndexConflict] = []
    for i in range(len(parsed_records)):
        lib_1, record_id_1, pairs_1 = parsed_records[i]
        for j in range(i + 1, len(parsed_records)):
            lib_2, record_id_2, pairs_2 = parsed_records[j]
            for left_1, right_1 in pairs_1:
                for left_2, right_2 in pairs_2:
                    is_repeat, conflict_type, same_left, same_right = _check_index_pair_repeat_latest(
                        left_1=left_1,
                        right_1=right_1,
                        left_2=left_2,
                        right_2=right_2,
                    )
                    if is_repeat and conflict_type is not None:
                        conflicts.append(
                            LatestIndexConflict(
                                record_id_1=record_id_1,
                                record_id_2=record_id_2,
                                conflict_type=conflict_type,
                                same_count_left=same_left,
                                same_count_right=same_right,
                            )
                        )
    return conflicts


def _validate_lane_with_latest_index(
    validator,
    libraries: List[EnhancedLibraryInfo],
    lane_id: str,
    machine_type: str,
    metadata: Dict[str, Any],
):
    """在原有Lane校验结果上，强制覆盖为最新Index冲突规则。"""
    result = validator.validate_lane(
        libraries=libraries,
        lane_id=lane_id,
        machine_type=machine_type,
        metadata=metadata,
    )

    try:
        latest_conflicts = _validate_index_conflicts_latest(libraries)
    except Exception as exc:
        logger.exception(f"Lane {lane_id} 最新Index校验失败，沿用原校验结果: {exc}")
        return result

    non_index_errors = [
        err for err in result.errors
        if err.rule_type != ValidationRuleType.INDEX_CONFLICT
    ]
    special_split_valid = True
    special_split_tokens: Set[str] = set()
    special_split_reason = ""
    if not bool((metadata or {}).get("skip_special_split_rule", False)):
        try:
            special_split_valid, special_split_tokens, special_split_reason = _validate_lane_special_split_rule(
                libraries
            )
        except Exception as exc:
            logger.exception(f"Lane {lane_id} wkspecialsplits规则校验失败，沿用原校验结果: {exc}")
            special_split_valid = True
            special_split_tokens = set()
            special_split_reason = "special_split_check_failed"
    imbalance_mix_valid = True
    imbalance_mix_reason = ""
    if bool((metadata or {}).get("check_56_57_mix_rule", False)):
        try:
            imbalance_mix_valid, imbalance_mix_reason = _validate_lane_57_mix_rules(
                libraries,
                enforce_total_limit=False,
                lane_id=lane_id,
                lane_metadata=metadata,
            )
        except Exception as exc:
            logger.exception(f"Lane {lane_id} 57组合规则校验失败，沿用原校验结果: {exc}")
            imbalance_mix_valid = True
            imbalance_mix_reason = ""
    if not special_split_valid:
        affected_ids: List[str] = []
        for lib in libraries:
            if _get_library_special_split_tokens(lib):
                affected_ids.append(str(getattr(lib, "origrec", "")))
        non_index_errors.append(
            ValidationError(
                rule_type=ValidationRuleType.SPECIAL_LIBRARY_LIMIT,
                severity=ValidationSeverity.ERROR,
                message=(
                    "wkspecialsplits组合不合法: 仅允许"
                    "{10x_longranger,10x_longranger_indexset,10x_cellranger,10x_cellranger_indexset}"
                    "任意子集同Lane，或"
                    "{10x_cellranger-atac_indexset,10x_cellranger-atac}同Lane且不得与其他类型混排"
                    f" | 当前={sorted(special_split_tokens)} | reason={special_split_reason}"
                ),
                affected_libraries=affected_ids,
            )
        )
    if not imbalance_mix_valid:
        affected_ids = [str(getattr(lib, "origrec", "")) for lib in libraries]
        non_index_errors.append(
            ValidationError(
                rule_type=ValidationRuleType.BASE_IMBALANCE_RATIO,
                severity=ValidationSeverity.ERROR,
                message=f"57组合规则不合法: {imbalance_mix_reason}",
                affected_libraries=affected_ids,
            )
        )
    latest_index_errors: List[ValidationError] = []
    for conflict in latest_conflicts:
        latest_index_errors.append(
            ValidationError(
                rule_type=ValidationRuleType.INDEX_CONFLICT,
                severity=ValidationSeverity.ERROR,
                message=(
                    f"Index冲突(最新规则): {conflict.record_id_1} vs {conflict.record_id_2} "
                    f"| 类型={conflict.conflict_type.value} | P7相同位数={conflict.same_count_left}"
                ),
                affected_libraries=[conflict.record_id_1, conflict.record_id_2],
            )
        )

    result.errors = non_index_errors + latest_index_errors
    result.is_valid = len(result.errors) == 0
    if getattr(validator, "strict_mode", False):
        result.is_valid = result.is_valid and len(result.warnings) == 0
    return result


def _resolve_machine_type_enum_simple(eq_type: Optional[str]) -> MachineType:
    """将机型字符串转换为MachineType枚举"""
    if not eq_type:
        return MachineType.NOVA_X_25B
    text = str(eq_type).strip()
    text_lower = text.lower()
    text_upper = text.upper()
    if "novaseq x plus" in text_lower or "nova seq x plus" in text_lower:
        return MachineType.NOVASEQ_X_PLUS
    if "10B" in text_upper and "25B" not in text_upper:
        return MachineType.NOVA_X_10B
    return MachineType.NOVA_X_25B


def _is_machine_supported_for_arrangement(machine_type: MachineType) -> bool:
    """当前V6排机主流程仅支持25B与NovaSeq X Plus，显式排除10B。"""
    return machine_type in {MachineType.NOVA_X_25B, MachineType.NOVASEQ_X_PLUS}


def _get_machine_arrangement_exclusion_reason(machine_type: MachineType) -> str:
    """返回按机型禁排的规则原因；空字符串表示该机型可进入排机。"""
    if machine_type == MachineType.NOVA_X_10B:
        return "规则禁排: Nova X-10B不参与AI排机"
    if not _is_machine_supported_for_arrangement(machine_type):
        return f"规则禁排: {_machine_type_to_text(machine_type, default='Unknown')}不参与AI排机"
    return ""


def _lane_capacity_for_machine(machine_type: MachineType) -> float:
    """获取机器类型对应的Lane容量，对于25B使用更新后的容量基准"""
    if machine_type == MachineType.NOVA_X_10B:
        return 380.0
    # 25B机器：使用975G作为基准容量（2026-02-06调整）
    return 975.0


def _resolve_lane_capacity_selection(
    libraries: List[EnhancedLibraryInfo],
    machine_type: MachineType | str,
    lane_id: str = "",
    lane_metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """按统一配置表解析Lane容量范围，未命中时自动回退到系统默认配置。"""
    machine_type_text = _machine_type_to_text(machine_type, default="Nova X-25B")
    metadata = _build_lane_metadata_for_validator(lane_id, lane_metadata, libraries=libraries)
    cache_key = (
        machine_type_text,
        _build_library_compact_identity_signature(list(libraries or []), canonicalize=True),
        tuple(
            sorted(
                (
                    str(key),
                    _safe_str(value, default=""),
                )
                for key, value in (metadata or {}).items()
                if value not in (None, "")
            )
        ),
    )
    cached_selection = _LANE_CAPACITY_SELECTION_CACHE.get(cache_key)
    if cached_selection is not None:
        return deepcopy(cached_selection)
    selection = get_scheduling_config().get_lane_capacity_range(
        libraries=libraries,
        machine_type=machine_type_text,
        metadata=metadata,
    )
    if getattr(selection, "rule_code", "") in SCHEDULING_CAP_RULE_CODES:
        selection.max_target_gb = min(float(selection.max_target_gb), SCHEDULING_MAX_TARGET_CAP_GB)
        selection.effective_max_gb = min(float(selection.effective_max_gb), SCHEDULING_MAX_EFFECTIVE_CAP_GB)
    selection = _apply_balance_reservation_to_capacity_selection(
        selection=selection,
        libraries=libraries,
        machine_type=machine_type,
        lane_id=lane_id,
        lane_metadata=lane_metadata,
    )
    _LANE_CAPACITY_SELECTION_CACHE[cache_key] = deepcopy(selection)
    return selection


def _resolve_lane_capacity_limits(
    libraries: List[EnhancedLibraryInfo],
    machine_type: MachineType | str,
    lane_id: str = "",
    lane_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """获取Lane有效容量上下限。"""
    selection = _resolve_lane_capacity_selection(
        libraries=libraries,
        machine_type=machine_type,
        lane_id=lane_id,
        lane_metadata=lane_metadata,
    )
    return float(selection.effective_min_gb), float(selection.effective_max_gb)


def _total_lane_data(libraries: List[EnhancedLibraryInfo]) -> float:
    """计算文库列表的总数据量"""
    if not libraries:
        return 0.0
    signature = _build_library_identity_signature(libraries, canonicalize=True)
    cached = _BUCKET_TOTAL_DATA_CACHE.get(signature)
    if cached is not None:
        return cached
    total = sum(lib.get_data_amount_gb() for lib in libraries)
    if len(_BUCKET_TOTAL_DATA_CACHE) >= 8192:
        _BUCKET_TOTAL_DATA_CACHE.clear()
    _BUCKET_TOTAL_DATA_CACHE[signature] = total
    return total


def _library_identity_key_for_pool_removal(lib: EnhancedLibraryInfo) -> Tuple[str, str, str]:
    """构建从后续普通排机池扣除预抽取文库的稳定身份键。"""
    detail_key = _safe_str(
        getattr(lib, "_detail_output_key", None)
        or getattr(lib, "wkaidbid", None)
        or getattr(lib, "aidbid", None),
        default="",
    )
    origrec_key = _safe_str(
        getattr(lib, "_origrec_key", None)
        or getattr(lib, "origrec", None)
        or getattr(lib, "wkorigrec", None),
        default="",
    )
    sample_id = _safe_str(getattr(lib, "sample_id", None), default="")
    return detail_key, origrec_key, sample_id


def _remove_libraries_used_by_lanes(
    libraries: List[EnhancedLibraryInfo],
    lanes: List[LaneAssignment],
) -> List[EnhancedLibraryInfo]:
    """从文库池中移除已进入预构建Lane的文库。"""
    if not libraries or not lanes:
        return list(libraries or [])
    used_keys = {
        _library_identity_key_for_pool_removal(lib)
        for lane in lanes
        for lib in list(getattr(lane, "libraries", []) or [])
    }
    return [
        lib
        for lib in libraries
        if _library_identity_key_for_pool_removal(lib) not in used_keys
    ]


def _extract_global_dedicated_imbalance_lanes(
    libraries: List[EnhancedLibraryInfo],
    *,
    mode_name: Optional[str] = None,
    dispatch_stage: str = "global_dedicated_imbalance_preextract",
) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo]]:
    """在普通1.1/3.6T排机前，全局预抽取碱基不均衡专Lane。"""
    if not libraries:
        return [], []

    imbalance_libraries = [
        lib
        for lib in libraries
        if _is_imbalance_library_candidate(lib)
        and not _is_priority_library_for_36t_policy(lib)
    ]
    if not imbalance_libraries:
        return [], list(libraries)

    scheduler_config = GreedyLaneConfig(
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
    scheduler = GreedyLaneScheduler(scheduler_config)
    if scheduler.pooling_optimizer:
        scheduler.pooling_optimizer.enabled = False

    generated_lanes: List[LaneAssignment] = []
    machine_type = MachineType.NOVA_X_25B.value
    scheduler.config = (
        scheduler._base_config.resolve_for_machine(machine_type)
        if scheduler._base_config.use_machine_config
        else scheduler._base_config
    )
    lanes, remaining_imbalance = scheduler._schedule_dedicated_imbalance_lanes(
        imbalance_libs=imbalance_libraries,
        machine_type=machine_type,
    )
    for lane in lanes:
        if not isinstance(lane.metadata, dict):
            lane.metadata = {}
        lane.metadata["dispatch_stage"] = dispatch_stage
        lane.metadata["is_dedicated_imbalance_lane"] = True
        selected_seq_mode = _safe_str(
            mode_name
            or lane.metadata.get("selected_seq_mode")
            or lane.metadata.get("seq_mode")
            or lane.metadata.get("lcxms"),
            default="",
        )
        if selected_seq_mode:
            lane.metadata["selected_seq_mode"] = selected_seq_mode
            lane.metadata["seq_mode"] = selected_seq_mode
            lane.metadata["lcxms"] = selected_seq_mode
            for lib in list(getattr(lane, "libraries", []) or []):
                lib._current_seq_mode_raw = selected_seq_mode
                lib.selected_seq_mode = selected_seq_mode
                lib.current_seq_mode = selected_seq_mode
                lib.lcxms = selected_seq_mode
    generated_lanes.extend(lanes)
    if lanes:
        used_data = _total_lane_data(
            [lib for lane in lanes for lib in list(getattr(lane, "libraries", []) or [])]
        )
        logger.info(
            "全局跨机型碱基不均专Lane预抽取: 承载机型={}, 生成Lane={}, 消耗{:.1f}G, 剩余不均文库={}个/{:.1f}G",
            machine_type,
            len(lanes),
            used_data,
            len(remaining_imbalance),
            _total_lane_data(remaining_imbalance),
        )

    remaining_libraries = _remove_libraries_used_by_lanes(libraries, generated_lanes)
    if generated_lanes:
        logger.info(
            "全局碱基不均专Lane预抽取完成: 生成Lane={}, 后续普通排机池 {} -> {}",
            len(generated_lanes),
            len(libraries),
            len(remaining_libraries),
        )
    return generated_lanes, remaining_libraries


def _merge_partial_schedule_solutions(partial_solutions: List[Any]) -> Any:
    """合并多个分桶排机结果。"""
    from types import SimpleNamespace

    merged = SimpleNamespace(
        lane_assignments=[],
        unassigned_libraries=[],
        split_rollback_mode_1_1_libraries=[],
    )
    for solution in partial_solutions:
        if solution is None:
            continue
        merged.lane_assignments.extend(list(getattr(solution, "lane_assignments", []) or []))
        merged.unassigned_libraries.extend(list(getattr(solution, "unassigned_libraries", []) or []))
        merged.split_rollback_mode_1_1_libraries.extend(
            list(getattr(solution, "split_rollback_mode_1_1_libraries", []) or [])
        )
    return merged


def _schedule_single_bucket_with_cache(
    *,
    scheduler: GreedyLaneScheduler,
    bucket_libraries: List[EnhancedLibraryInfo],
    post_fill_optimization_enabled: bool,
) -> Any:
    """单个规则桶调度，复用主流程分桶缓存。"""
    bucket_signature = _build_library_identity_signature(bucket_libraries, canonicalize=True)
    cache_key = (bucket_signature, bool(post_fill_optimization_enabled))
    cached = _MAIN_RULE_BUCKET_SCHEDULE_CACHE.get(cache_key)
    if cached is not None:
        cached_solutions, _, _, _ = cached
        if cached_solutions:
            return deepcopy(cached_solutions[0])

    bucket_solution = scheduler.schedule(
        bucket_libraries,
        keep_failed_lanes=True,
        libraries_already_split=True,
        perform_presplit_family_rollback=False,
        enable_post_fill_optimization=post_fill_optimization_enabled,
    )
    _MAIN_RULE_BUCKET_SCHEDULE_CACHE[cache_key] = (
        [deepcopy(bucket_solution)],
        list(getattr(bucket_solution, "unassigned_libraries", []) or []),
        {
            _get_library_identity_key(lib)
            for lane in list(getattr(bucket_solution, "lane_assignments", []) or [])
            for lib in list(getattr(lane, "libraries", []) or [])
        },
        1,
    )
    return bucket_solution


def _schedule_with_rule_bucket_prescheduling(
    *,
    scheduler: GreedyLaneScheduler,
    libraries: List[EnhancedLibraryInfo],
    post_fill_optimization_enabled: bool,
) -> Any:
    """主流程规则分桶预调度：先按规则桶独立排，再汇总尾货统一补排。"""
    from types import SimpleNamespace

    if not libraries:
        return SimpleNamespace(lane_assignments=[], unassigned_libraries=[])

    machine_type = MachineType.NOVA_X_25B
    rule_buckets = _build_rescue_rule_buckets(libraries, "MAIN")
    if not rule_buckets:
        return scheduler.schedule(
            libraries,
            keep_failed_lanes=True,
            libraries_already_split=True,
            perform_presplit_family_rollback=False,
            enable_post_fill_optimization=post_fill_optimization_enabled,
        )

    partial_solutions: List[Any] = []
    tail_pool: List[EnhancedLibraryInfo] = []
    scheduled_library_keys: Set[str] = set()
    bucket_count = 0

    for bucket_name, bucket_libraries in rule_buckets:
        if not bucket_libraries:
            continue
        bucket_count += 1
        feasible, reason = _quick_check_pool_feasibility(
            pool=bucket_libraries,
            machine_type=machine_type,
            lane_metadata=None,
            stage_label=f"MAIN_{bucket_name}",
        )
        if not feasible:
            logger.info(
                "主流程规则分桶跳过: bucket={}, count={}, reason={}".format(
                    bucket_name,
                    len(bucket_libraries),
                    reason,
                )
            )
            tail_pool.extend(bucket_libraries)
            continue

        min_allowed, _ = _resolve_lane_capacity_limits(bucket_libraries, machine_type)
        total_data = _total_lane_data(bucket_libraries)
        if total_data + 1e-6 < min_allowed:
            logger.info(
                "主流程规则分桶尾货回收: bucket={}, count={}, data={:.1f}G不足门槛{:.1f}G".format(
                    bucket_name,
                    len(bucket_libraries),
                    total_data,
                    min_allowed,
                )
            )
            tail_pool.extend(bucket_libraries)
            continue

        logger.info(
            "主流程规则分桶预调度: bucket={}, count={}, data={:.1f}G".format(
                bucket_name,
                len(bucket_libraries),
                total_data,
            )
        )
        bucket_solution = _schedule_single_bucket_with_cache(
            scheduler=scheduler,
            bucket_libraries=bucket_libraries,
            post_fill_optimization_enabled=post_fill_optimization_enabled,
        )
        partial_solutions.append(bucket_solution)
        for lane in list(getattr(bucket_solution, "lane_assignments", []) or []):
            for lib in list(getattr(lane, "libraries", []) or []):
                scheduled_library_keys.add(_get_library_identity_key(lib))
        tail_pool.extend(list(getattr(bucket_solution, "unassigned_libraries", []) or []))

    dedup_tail_pool: List[EnhancedLibraryInfo] = []
    seen_tail_keys: Set[str] = set()
    for lib in tail_pool:
        lib_key = _get_library_identity_key(lib)
        if lib_key in scheduled_library_keys or lib_key in seen_tail_keys:
            continue
        seen_tail_keys.add(lib_key)
        dedup_tail_pool.append(lib)

    logger.info(
        "主流程规则分桶预调度完成: 分桶={}，桶后尾货={}个/{:.1f}G".format(
            bucket_count,
            len(dedup_tail_pool),
            _total_lane_data(dedup_tail_pool),
        )
    )

    if dedup_tail_pool:
        logger.info(
            "主流程尾货统一补排启动: count={}, data={:.1f}G".format(
                len(dedup_tail_pool),
                _total_lane_data(dedup_tail_pool),
            )
        )
        tail_solution = _schedule_single_bucket_with_cache(
            scheduler=scheduler,
            bucket_libraries=dedup_tail_pool,
            post_fill_optimization_enabled=post_fill_optimization_enabled,
        )
        partial_solutions.append(tail_solution)

    merged_solution = _merge_partial_schedule_solutions(partial_solutions)
    _deduplicate_solution_libraries(merged_solution)
    return merged_solution


def _is_priority_36t_preconsume_lane_capacity_valid(
    lane: LaneAssignment,
) -> Tuple[bool, Dict[str, Any]]:
    """判断高优先级预消耗Lane是否满足正式3.6T容量规则。"""
    lane_libraries = list(getattr(lane, "libraries", []) or [])
    total_data_gb = float(_total_lane_data(lane_libraries))
    lane_metadata = dict(getattr(lane, "metadata", {}) or {})
    lane_metadata.setdefault("selected_seq_mode", "3.6T-NEW")

    machine_type = getattr(lane, "machine_type", "") or ""
    if not machine_type and lane_libraries:
        machine_type = str(getattr(lane_libraries[0], "eq_type", "") or "")

    if not lane_libraries:
        return False, {
            "lane_id": str(getattr(lane, "lane_id", "") or ""),
            "rule_code": "",
            "total_data_gb": round(total_data_gb, 1),
            "min_allowed_gb": 0.0,
            "max_allowed_gb": 0.0,
            "reason": "empty_lane",
        }

    selection = _resolve_lane_capacity_selection(
        libraries=lane_libraries,
        machine_type=machine_type,
        lane_id=str(getattr(lane, "lane_id", "") or ""),
        lane_metadata=lane_metadata,
    )
    min_allowed_gb = float(getattr(selection, "effective_min_gb", 0.0) or 0.0)
    max_allowed_gb = float(getattr(selection, "effective_max_gb", 0.0) or 0.0)
    info = {
        "lane_id": str(getattr(lane, "lane_id", "") or ""),
        "rule_code": str(getattr(selection, "rule_code", "") or ""),
        "total_data_gb": round(total_data_gb, 1),
        "min_allowed_gb": round(min_allowed_gb, 1),
        "max_allowed_gb": round(max_allowed_gb, 1),
        "reason": "ok",
    }
    if total_data_gb + 1e-6 < min_allowed_gb:
        info["reason"] = "below_min_capacity"
        return False, info
    if total_data_gb - 1e-6 > max_allowed_gb:
        info["reason"] = "above_max_capacity"
        return False, info
    return True, info


@dataclass
class Priority36TPreconsumeResult:
    """3.6T-NEW高优先级预消耗结果。"""

    lanes: List[LaneAssignment] = field(default_factory=list)
    remaining_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    remaining_priority_forbidden_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    remaining_filler_forbidden_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    remaining_filler_1_1_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    consumed_priority_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    consumed_priority_forbidden_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    consumed_filler_forbidden_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    consumed_filler_1_1_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    scheduling_stats: Dict[str, Any] = field(default_factory=dict)
    scheduling_succeeded: bool = False
    scheduling_error: Optional[str] = None


@dataclass
class Priority36TPreconsumeInputs:
    """3.6T-NEW高优先级预消耗的分层输入。"""

    priority_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    priority_forbidden_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    filler_forbidden_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    all_filler_1_1_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    deferred_priority_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    deferred_priority_forbidden_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    max_priority_lanes: int = 0
    max_filler_gb_per_lane: float = PRIORITY_36T_PRECONSUME_FILLER_MAX_GB_PER_LANE


def _get_priority_36t_preconsume_priority_score(lib: EnhancedLibraryInfo) -> float:
    """统一计算首轮1.1前置3.6T预消耗的优先级评分。"""
    try:
        return float(lib.calculate_priority_score())
    except Exception:
        return 0.0


def _priority_36t_preconsume_candidate_sort_key(
    bucket: str,
    lib: EnhancedLibraryInfo,
) -> Tuple[int, int, float, float, str]:
    """构造优先级预消耗窗口的排序键。"""
    bucket_rank = 0 if bucket == "priority_36t" else 1
    preferred_data_types = tuple(
        getattr(lib, "_priority_preconsume_preferred_data_types", ()) or ()
    )

    data_type = str(getattr(lib, "data_type", "") or "")
    data_type_rank = len(preferred_data_types)
    for index, preferred_data_type in enumerate(preferred_data_types):
        if data_type == preferred_data_type:
            data_type_rank = index
            break

    data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
    return (
        bucket_rank,
        data_type_rank,
        -_get_priority_36t_preconsume_priority_score(lib),
        -data,
        _safe_str(getattr(lib, "origrec", ""), default=""),
    )


def _priority_36t_preconsume_group_key(
    bucket: str,
    lib: EnhancedLibraryInfo,
) -> Tuple[str, str]:
    """按文库身份分组，避免少数高重复文库霸占首轮3.6T预消耗窗口。"""
    origrec = _safe_str(getattr(lib, "origrec", ""), default="")
    if origrec:
        return bucket, origrec

    sample_id = _safe_str(getattr(lib, "sample_id", ""), default="")
    if sample_id:
        return bucket, sample_id

    index_seq = _safe_str(getattr(lib, "index_seq", ""), default="")
    if index_seq:
        return bucket, index_seq

    return bucket, str(id(lib))


def _normalize_priority_36t_preconsume_data_type(lib: EnhancedLibraryInfo) -> str:
    """统一首轮3.6T预消耗窗口内的数据类型口径。"""
    data_type = str(getattr(lib, "data_type", "") or "").strip()
    if data_type in {"YC", "SJ", "临检"}:
        return data_type
    return "OTHER"


PRIORITY_36T_REDLINE_DATA_TYPE_ORDER: Tuple[str, ...] = (
    "临检",
    "YC",
    "SJ",
)


def _select_priority_36t_preconsume_window(
    bucketed_candidates: List[Tuple[str, EnhancedLibraryInfo]],
    *,
    max_total_gb: float,
    preferred_machine_types: Tuple[str, ...] = (),
    preferred_data_types: Tuple[str, ...] = (),
    spread_lane_count: int = 0,
) -> Tuple[Dict[str, List[EnhancedLibraryInfo]], Dict[str, List[EnhancedLibraryInfo]], float]:
    """按lane上限截取优先级预消耗窗口，按代表负载控容量、按整组文库进窗口。"""
    selected_by_bucket: Dict[str, List[EnhancedLibraryInfo]] = {
        "priority_1_1": [],
        "priority_36t": [],
    }
    deferred_by_bucket: Dict[str, List[EnhancedLibraryInfo]] = {
        "priority_1_1": [],
        "priority_36t": [],
    }
    if not bucketed_candidates:
        return selected_by_bucket, deferred_by_bucket, 0.0

    grouped_candidates: Dict[Tuple[str, str], List[EnhancedLibraryInfo]] = {}
    for bucket, lib in list(bucketed_candidates):
        group_key = _priority_36t_preconsume_group_key(bucket, lib)
        grouped_candidates.setdefault(group_key, []).append(lib)

    ordered_groups = []
    tagged_libraries: List[EnhancedLibraryInfo] = []
    try:
        for group_key, grouped_libraries in grouped_candidates.items():
            bucket = group_key[0]
            for lib in grouped_libraries:
                setattr(lib, "_priority_preconsume_preferred_machine_types", preferred_machine_types)
                setattr(lib, "_priority_preconsume_preferred_data_types", preferred_data_types)
                tagged_libraries.append(lib)
            grouped_libraries.sort(
                key=lambda item: _priority_36t_preconsume_candidate_sort_key(bucket, item),
            )
            ordered_groups.append(group_key)

        ordered_groups.sort(
            key=lambda group_key: (
                *_priority_36t_preconsume_candidate_sort_key(
                    group_key[0],
                    grouped_candidates[group_key][0],
                )[:3],
                -sum(
                    float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
                    for lib in grouped_candidates[group_key]
                ),
                *_priority_36t_preconsume_candidate_sort_key(
                    group_key[0],
                    grouped_candidates[group_key][0],
                )[3:],
            ),
        )

        selected_total_gb = 0.0
        selected_group_type_counts: Counter[str] = Counter()

        def _select_or_defer_group(group_key: Tuple[str, str]) -> None:
            nonlocal selected_total_gb

            bucket = group_key[0]
            group_libraries = list(grouped_candidates.get(group_key, []))
            if not group_libraries:
                return

            representative = group_libraries[0]
            representative_data = float(getattr(representative, "contract_data_raw", 0.0) or 0.0)
            if (
                selected_total_gb > 0
                and selected_total_gb + representative_data > max_total_gb + 1e-6
            ):
                deferred_by_bucket.setdefault(bucket, []).extend(group_libraries)
                return

            selected_by_bucket.setdefault(bucket, []).extend(group_libraries)
            selected_total_gb += representative_data
            selected_group_type_counts[
                _normalize_priority_36t_preconsume_data_type(representative)
            ] += 1

        for group_key in ordered_groups:
            _select_or_defer_group(group_key)
    finally:
        for lib in tagged_libraries:
            if hasattr(lib, "_priority_preconsume_preferred_machine_types"):
                delattr(lib, "_priority_preconsume_preferred_machine_types")
            if hasattr(lib, "_priority_preconsume_preferred_data_types"):
                delattr(lib, "_priority_preconsume_preferred_data_types")

    return selected_by_bucket, deferred_by_bucket, selected_total_gb


def _priority_36t_fill_candidate_sort_key(
    lib: EnhancedLibraryInfo,
    *,
    allocator: ModeAllocator,
) -> Tuple[int, int, float, float, str]:
    """构造规则11补位文库排序键。"""
    prefix = str(getattr(lib, "sample_number_prefix", "") or "").strip().upper()
    if not prefix:
        sample_id = str(getattr(lib, "sample_id", "") or "").strip().upper()
        prefix = sample_id[:4] if len(sample_id) >= 4 else sample_id
    has_fdhe_prefix = any(
        prefix.startswith(str(eligible_prefix or "").upper())
        for eligible_prefix in list(getattr(allocator, "_eligible_prefixes", []) or [])
    )
    remark = str(getattr(lib, "add_tests_remark", "") or "").strip()
    has_add_test_flag = remark in set(getattr(allocator, "_eligible_add_test_kw", set()) or set())
    data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
    return (
        0 if has_add_test_flag else 1,
        0 if has_fdhe_prefix else 1,
        -_get_priority_36t_preconsume_priority_score(lib),
        -data,
        _safe_str(getattr(lib, "origrec", ""), default=""),
    )


def _priority_1_1_overflow_sort_key(
    lib: EnhancedLibraryInfo,
    *,
    allocator: ModeAllocator,
) -> Tuple[int, int, float, float, str]:
    """构造临检/YC/SJ 少量溢出到1.1时的优先级。"""
    prefix = allocator._resolve_sample_prefix(lib)
    has_fdhe_prefix = any(
        prefix.startswith(str(eligible_prefix or "").upper())
        for eligible_prefix in list(getattr(allocator, "_eligible_prefixes", []) or [])
    )
    remark = str(getattr(lib, "add_tests_remark", "") or "").strip()
    has_add_test_flag = remark in set(getattr(allocator, "_eligible_add_test_kw", set()) or set())
    data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
    return (
        0 if has_add_test_flag else 1,
        0 if has_fdhe_prefix else 1,
        data,
        -_get_priority_36t_preconsume_priority_score(lib),
        _safe_str(getattr(lib, "origrec", ""), default=""),
    )


def _get_priority_36t_preconsume_library_identity(lib: EnhancedLibraryInfo) -> str:
    """提取首轮3.6T预消耗文库的稳定身份键，用于多轮重试时扣减已使用文库。"""
    origrec = _safe_str(getattr(lib, "origrec", ""), default="")
    sample_id = _safe_str(getattr(lib, "sample_id", ""), default="")
    if origrec or sample_id:
        return f"origrec:{origrec}|sample_id:{sample_id}"

    if sample_id:
        return f"sample_id:{sample_id}"

    index_seq = _safe_str(getattr(lib, "index_seq", ""), default="")
    if index_seq:
        return f"index:{index_seq}"

    return f"object:{id(lib)}"


def _select_priority_36t_preconsume_fillers(
    filler_candidates: List[EnhancedLibraryInfo],
    *,
    target_gb: float,
    candidate_window_gb: Optional[float] = None,
) -> List[EnhancedLibraryInfo]:
    """从预排序补位候选中截取单次尝试所需的候选窗口。

    target_gb 是本次理论补料量；candidate_window_gb 是实际给调度器试配的候选量。
    高优预消耗需要足够普通文库做兼容性试探，不能只按缺口截断，否则候选被
    占比/PeakSize/10bp 规则拒绝后就没有替补可试。
    """
    if target_gb <= 0 and not candidate_window_gb:
        return []

    window_gb = float(candidate_window_gb if candidate_window_gb is not None else target_gb)
    if window_gb <= 0:
        return []

    selected_fillers: List[EnhancedLibraryInfo] = []
    selected_total_gb = 0.0
    for lib in list(filler_candidates or []):
        data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
        if (
            selected_fillers
            and selected_total_gb + data > window_gb + 1e-6
        ):
            continue
        if not selected_fillers and data > window_gb + 1e-6:
            continue
        selected_fillers.append(lib)
        selected_total_gb += data
        if selected_total_gb >= window_gb - 1e-6:
            break

    return selected_fillers


def _resolve_priority_36t_preconsume_filler_limit(
    *,
    configured_limit_gb: Optional[float],
    has_priority_libraries: bool,
) -> float:
    """高优预消耗补料上限。

    只要本批确实在消耗高优文库，普通补料不再受配置的单Lane补料上限限制；
    上限交给正式3.6T容量与红线校验控制。
    """
    if has_priority_libraries:
        return SCHEDULING_MAX_TARGET_CAP_GB
    return max(
        0.0,
        float(
            PRIORITY_36T_PRECONSUME_FILLER_MAX_GB_PER_LANE
            if configured_limit_gb is None
            else configured_limit_gb
        ),
    )


def _priority_36t_stage_fill_candidate_sort_key(
    lib: EnhancedLibraryInfo,
    *,
    preferred_machine_types: Tuple[str, ...] = (),
) -> Tuple[int, int, int, int, int, float, float, str]:
    """为首轮3.6T重试阶段重排补位文库，优先选择红线友好的全局补料。"""
    data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
    sample_type_text = " ".join(
        _safe_str(value, default="")
        for value in (
            getattr(lib, "sample_type", None),
            getattr(lib, "sample_type_code", None),
            getattr(lib, "lab_type", None),
            getattr(lib, "project_type", None),
            getattr(lib, "project_name", None),
        )
    )
    redline_risky_keywords = (
        "ATAC",
        "10X",
        "10x",
        "Visium",
        "Methylation",
        "甲基化",
        "small RNA",
        "单细胞",
        "外显子",
        "HI-C",
        "CUT Tag",
    )
    is_imbalance = bool(getattr(lib, "is_base_imbalance", lambda: False)()) or any(
        keyword in sample_type_text for keyword in redline_risky_keywords
    )
    is_customer = bool(getattr(lib, "is_customer_library", lambda: False)())
    ten_bp_data = float(getattr(lib, "ten_bp_data", 0.0) or 0.0)
    index_seq = str(getattr(lib, "index_seq", "") or "")
    index_parts = [part.strip() for part in index_seq.replace(";", ",").split(",") if part.strip()]
    has_10bp_index = ten_bp_data > 0 or any(len(part) >= 10 for part in index_parts)
    has_single_end_index = len(index_parts) == 1
    peak_size = float(getattr(lib, "peak_size", 0.0) or 0.0)
    peak_distance = abs(peak_size - 400.0) if peak_size > 0 else 9999.0
    return (
        1 if is_imbalance else 0,
        1 if is_customer else 0,
        0 if has_10bp_index else 1,
        1 if has_single_end_index else 0,
        0 if 250.0 <= peak_size <= 500.0 else 1,
        peak_distance,
        -data,
        _safe_str(getattr(lib, "origrec", ""), default=""),
    )


def _resolve_priority_36t_preconsume_routing_machine_type(
    priority_libraries: List[EnhancedLibraryInfo],
    fallback_libraries: Optional[List[EnhancedLibraryInfo]] = None,
) -> str:
    """高优预消耗只用一个调度机型承载全局补料池，避免按原始机型切分。"""
    for lib in list(priority_libraries or []) + list(fallback_libraries or []):
        eq_type = _safe_str(getattr(lib, "eq_type", ""), default="")
        if eq_type:
            return eq_type
    return MachineType.NOVA_X_25B.value


def _priority_36t_preconsume_data_gb(lib: EnhancedLibraryInfo) -> float:
    return float(getattr(lib, "contract_data_raw", 0.0) or 0.0)


def _priority_36t_preconsume_bucket_rank(lib: EnhancedLibraryInfo) -> int:
    bucket = str(getattr(lib, "_priority_preconsume_bucket", "") or "")
    if bucket == "priority_36t":
        return 0
    if bucket == "priority_1_1":
        return 1
    if bucket == "filler_36t":
        return 2
    if bucket == "filler_1_1":
        return 3
    return 4


def _priority_36t_preconsume_type_rank(lib: EnhancedLibraryInfo) -> int:
    data_type = _normalize_priority_36t_preconsume_data_type(lib)
    order = {"临检": 0, "YC": 1, "SJ": 2}
    return order.get(data_type, 9)


def _priority_36t_preconsume_core_sort_key(
    scheduler: GreedyLaneScheduler,
    lib: EnhancedLibraryInfo,
) -> Tuple[int, int, int, int, int, float, float, str]:
    """高优核心排序：先保临检，再在同类型内优先选择更容易过红线的文库。"""
    data = _priority_36t_preconsume_data_gb(lib)
    is_imbalance = bool(getattr(lib, "is_base_imbalance", lambda: False)())
    has_10bp = scheduler._library_has_10bp_index(lib)
    is_single = scheduler._is_single_end_index(str(getattr(lib, "index_seq", "") or ""))
    peak_size = float(getattr(lib, "peak_size", 0.0) or 0.0)
    peak_distance = abs(peak_size - 400.0) if peak_size > 0 else 9999.0
    return (
        _priority_36t_preconsume_bucket_rank(lib),
        _priority_36t_preconsume_type_rank(lib),
        1 if is_imbalance else 0,
        0 if has_10bp else 1,
        1 if is_single else 0,
        peak_distance,
        -data,
        _safe_str(getattr(lib, "origrec", ""), default=""),
    )


def _is_priority_36t_preconsume_core_library(lib: EnhancedLibraryInfo) -> bool:
    """高优预消耗Lane的起始/core文库只能是临检、YC、SJ。"""
    return _normalize_priority_36t_preconsume_data_type(lib) in PRIORITY_36T_REDLINE_DATA_TYPE_ORDER


def _priority_36t_preconsume_create_lane(
    scheduler: GreedyLaneScheduler,
    *,
    machine_type: str,
    machine_type_enum: MachineType,
) -> LaneAssignment:
    lane_id = scheduler._get_next_lane_id("GL", machine_type)
    lane = LaneAssignment(
        lane_id=lane_id,
        machine_id=f"M_{lane_id[3:]}",
        machine_type=machine_type_enum,
        lane_capacity_gb=SCHEDULING_MAX_TARGET_CAP_GB,
    )
    lane.metadata["dispatch_stage"] = "priority_preconsume_36t"
    lane.metadata["selected_seq_mode"] = "3.6T-NEW"
    lane.metadata["seq_mode"] = "3.6T-NEW"
    lane.metadata["lcxms"] = "3.6T-NEW"
    lane.metadata["priority_preconsume_defer_terminal_ratio_checks"] = True
    return lane


def _priority_36t_preconsume_add_if_possible(
    scheduler: GreedyLaneScheduler,
    lane: LaneAssignment,
    lib: EnhancedLibraryInfo,
) -> bool:
    if not scheduler._can_add_to_lane(lane, lib):
        return False
    lane.add_library(lib)
    return True


def _priority_36t_preconsume_profile_summary(
    scheduler: GreedyLaneScheduler,
    libraries: List[EnhancedLibraryInfo],
) -> Dict[str, float]:
    cache_key = _build_library_compact_identity_signature(libraries, canonicalize=True)
    cached = _PRIORITY_36T_PROFILE_SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    total = sum(_priority_36t_preconsume_data_gb(lib) for lib in libraries)
    if total <= 0:
        result = {
            "total": 0.0,
            "customer": 0.0,
            "customer_ratio": 0.0,
            "ten_bp": 0.0,
            "ten_bp_ratio": 0.0,
            "single": 0.0,
            "single_ratio": 0.0,
            "imbalance": 0.0,
            "imbalance_ratio": 0.0,
        }
        _PRIORITY_36T_PROFILE_SUMMARY_CACHE[cache_key] = dict(result)
        return result

    base_libraries = [
        lib for lib in libraries
        if not bool(getattr(lib, "_is_ai_balance_library", False))
    ]
    base_total = sum(_priority_36t_preconsume_data_gb(lib) for lib in base_libraries) or total
    customer = sum(
        _priority_36t_preconsume_data_gb(lib)
        for lib in base_libraries
        if scheduler._is_customer_library(lib)
    )
    ten_bp = sum(
        _priority_36t_preconsume_data_gb(lib)
        for lib in base_libraries
        if scheduler._library_has_10bp_index(lib)
    )
    single = 0.0
    imbalance = 0.0
    for lib in libraries:
        data = _priority_36t_preconsume_data_gb(lib)
        single_index_data = getattr(lib, "single_index_data", None)
        if single_index_data is not None and single_index_data > 0:
            single += data
        elif scheduler._is_single_end_index(str(getattr(lib, "index_seq", "") or "")):
            single += data
        if bool(getattr(lib, "is_base_imbalance", lambda: False)()):
            imbalance += data

    result = {
        "total": total,
        "customer": customer,
        "customer_ratio": customer / base_total if base_total > 0 else 0.0,
        "ten_bp": ten_bp,
        "ten_bp_ratio": ten_bp / base_total if base_total > 0 else 0.0,
        "single": single,
        "single_ratio": single / total if total > 0 else 0.0,
        "imbalance": imbalance,
        "imbalance_ratio": imbalance / total if total > 0 else 0.0,
    }
    if len(_PRIORITY_36T_PROFILE_SUMMARY_CACHE) >= 20000:
        _PRIORITY_36T_PROFILE_SUMMARY_CACHE.clear()
    _PRIORITY_36T_PROFILE_SUMMARY_CACHE[cache_key] = dict(result)
    return result


def _priority_36t_preconsume_peak_compatible(
    scheduler: GreedyLaneScheduler,
    libraries: List[EnhancedLibraryInfo],
) -> bool:
    cache_key = _build_library_compact_identity_signature(libraries, canonicalize=True)
    cached = _PRIORITY_36T_PEAK_COMPAT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    result = bool(scheduler._check_peak_size_compatible(libraries))
    if len(_PRIORITY_36T_PEAK_COMPAT_CACHE) >= 20000:
        _PRIORITY_36T_PEAK_COMPAT_CACHE.clear()
    _PRIORITY_36T_PEAK_COMPAT_CACHE[cache_key] = result
    return result


def _priority_36t_preconsume_filler_sort_key(
    scheduler: GreedyLaneScheduler,
    lane_libraries: List[EnhancedLibraryInfo],
    lib: EnhancedLibraryInfo,
    *,
    current_summary: Optional[Dict[str, float]] = None,
) -> Tuple[int, int, int, float, float, float, str, float]:
    current = current_summary or _priority_36t_preconsume_profile_summary(scheduler, lane_libraries)
    projected = _priority_36t_preconsume_profile_summary(scheduler, lane_libraries + [lib])
    data = _priority_36t_preconsume_data_gb(lib)

    ten_bp_needed = (
        current["ten_bp"] > 0
        and current["ten_bp_ratio"] < 0.40
    )
    single_near_limit = current["single"] > 0 and current["single_ratio"] >= 0.25
    imbalance_near_limit = current["imbalance"] > 0 and current["imbalance_ratio"] >= 0.30
    customer_middle = 0.50 < current["customer_ratio"] < 1.0

    has_10bp = scheduler._library_has_10bp_index(lib)
    single_index_data = getattr(lib, "single_index_data", None)
    is_single = (
        single_index_data is not None and single_index_data > 0
    ) or scheduler._is_single_end_index(str(getattr(lib, "index_seq", "") or ""))
    is_imbalance = bool(getattr(lib, "is_base_imbalance", lambda: False)())
    is_customer = scheduler._is_customer_library(lib)

    peak_penalty = 0.0
    existing_peaks = [
        float(getattr(item, "peak_size", 0.0) or 0.0)
        for item in lane_libraries
        if float(getattr(item, "peak_size", 0.0) or 0.0) > 0
    ]
    candidate_peak = float(getattr(lib, "peak_size", 0.0) or 0.0)
    if existing_peaks and candidate_peak > 0:
        sorted_peaks = sorted(existing_peaks)
        median_peak = sorted_peaks[len(sorted_peaks) // 2]
        peak_penalty = abs(candidate_peak - median_peak)

    return (
        0 if (not ten_bp_needed or has_10bp) else 1,
        0 if (not single_near_limit or not is_single) else 1,
        0 if (not imbalance_near_limit or not is_imbalance) else 1,
        abs(projected["customer_ratio"] - (0.45 if customer_middle else projected["customer_ratio"])),
        max(projected["imbalance_ratio"] - 0.35, 0.0),
        peak_penalty,
        _safe_str(getattr(lib, "origrec", ""), default=""),
        -data,
    )


def _priority_36t_preconsume_core_profile_can_add(
    scheduler: GreedyLaneScheduler,
    lane_libraries: List[EnhancedLibraryInfo],
    lib: EnhancedLibraryInfo,
    *,
    final_max_gb: float = SCHEDULING_MAX_EFFECTIVE_CAP_GB,
) -> bool:
    """判断高优核心加入后，是否仍有机会通过后续补料稀释到正式红线内。"""
    projected = list(lane_libraries) + [lib]
    summary = _priority_36t_preconsume_profile_summary(scheduler, projected)
    total = summary["total"]
    if total <= 0 or total > final_max_gb + 1e-6:
        return False

    imbalance_limit = min(
        float(getattr(scheduler.config, "max_special_library_data_gb", 350.0) or 350.0),
        float(getattr(scheduler.config, "max_imbalance_ratio", 0.35) or 0.35) * final_max_gb,
    )
    if summary["imbalance"] > imbalance_limit + 1e-6:
        return False

    single_limit = float(getattr(scheduler.lane_validator, "single_end_ratio_limit", 0.30) or 0.30)
    if summary["single"] > single_limit * final_max_gb + 1e-6:
        return False

    customer = summary["customer"]
    if customer > 0 and customer < total - 1e-6:
        if customer > 0.50 * final_max_gb + 1e-6:
            return False

    ten_bp = summary["ten_bp"]
    non_10bp = total - ten_bp
    if ten_bp > 0 and non_10bp > 0:
        min_10bp_ratio = float(getattr(scheduler.config, "min_10bp_index_ratio", 0.40) or 0.40)
        max_possible_10bp = ten_bp + max(0.0, final_max_gb - total)
        if max_possible_10bp / final_max_gb + 1e-12 < min_10bp_ratio:
            return False

    if not _priority_36t_preconsume_peak_compatible(scheduler, projected):
        return False

    return True


def _priority_36t_preconsume_core_variants(
    core_libraries: List[EnhancedLibraryInfo],
) -> List[List[EnhancedLibraryInfo]]:
    if not core_libraries:
        return []
    variants: List[List[EnhancedLibraryInfo]] = []
    thresholds = (1100.0, 950.0, 850.0, 700.0, 500.0, 350.0, 250.0, 0.0)
    for threshold in thresholds:
        selected: List[EnhancedLibraryInfo] = []
        total = 0.0
        for lib in core_libraries:
            data = _priority_36t_preconsume_data_gb(lib)
            if selected and threshold > 0 and total + data > threshold + 1e-6:
                continue
            selected.append(lib)
            total += data
            if threshold > 0 and total >= threshold - 1e-6:
                break
        if not selected:
            selected = [core_libraries[0]]
        selected_ids = tuple(id(lib) for lib in selected)
        if any(tuple(id(item) for item in variant) == selected_ids for variant in variants):
            continue
        variants.append(selected)
    return variants


def _priority_36t_preconsume_fill_and_validate_lane(
    scheduler: GreedyLaneScheduler,
    *,
    core_libraries: List[EnhancedLibraryInfo],
    filler_libraries: List[EnhancedLibraryInfo],
    machine_type: str,
    machine_type_enum: MachineType,
) -> Optional[LaneAssignment]:
    filler_ids = {id(lib) for lib in filler_libraries}
    lane = _priority_36t_preconsume_create_lane(
        scheduler,
        machine_type=machine_type,
        machine_type_enum=machine_type_enum,
    )
    for lib in core_libraries:
        if not _priority_36t_preconsume_add_if_possible(scheduler, lane, lib):
            return None

    while True:
        selection = scheduler._get_scheduling_lane_capacity_range(
            libraries=lane.libraries,
            machine_type=machine_type_enum.value,
            metadata=scheduler._build_lane_validation_metadata(lane),
        )
        if lane.total_data_gb >= selection.effective_min_gb:
            is_valid, _ = scheduler._validate_completed_lane(lane)
            if is_valid:
                return lane

        if lane.total_data_gb >= selection.effective_max_gb - 1e-6:
            return None

        added = False
        used_ids = {id(item) for item in lane.libraries}
        available_fillers = [
            lib for lib in filler_libraries
            if id(lib) not in used_ids
        ][:160]
        if not available_fillers:
            return None
        lane_libraries_snapshot = list(lane.libraries)
        lane_profile_summary = _priority_36t_preconsume_profile_summary(
            scheduler,
            lane_libraries_snapshot,
        )
        ordered_fillers = sorted(
            available_fillers,
            key=lambda lib: _priority_36t_preconsume_filler_sort_key(
                scheduler,
                lane_libraries_snapshot,
                lib,
                current_summary=lane_profile_summary,
            ),
        )
        for filler in ordered_fillers:
            if id(filler) not in filler_ids:
                continue
            projected_libraries = lane_libraries_snapshot + [filler]
            projected_summary = _priority_36t_preconsume_profile_summary(
                scheduler,
                projected_libraries,
            )
            if projected_summary["imbalance"] > scheduler.config.max_special_library_data_gb + 1e-6:
                continue
            if not _priority_36t_preconsume_add_if_possible(scheduler, lane, filler):
                continue
            added = True
            break

        if not added:
            return None


def _build_priority_36t_preconsume_lanes_by_profile(
    scheduler: GreedyLaneScheduler,
    libraries: List[EnhancedLibraryInfo],
    *,
    machine_type: str,
) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo]]:
    """专用3.6T高优预消耗：先按红线画像尽量放高优，再放普通补料。"""
    machine_type_enum = _resolve_machine_type_enum_simple(machine_type)
    priority_libraries = [
        lib for lib in libraries
        if str(getattr(lib, "_priority_preconsume_bucket", "") or "").startswith("priority_")
        and _is_priority_36t_preconsume_core_library(lib)
    ]
    priority_core_ids = {id(lib) for lib in priority_libraries}
    filler_libraries = [
        lib for lib in libraries
        if id(lib) not in priority_core_ids
    ]
    priority_libraries = sorted(
        priority_libraries,
        key=lambda lib: _priority_36t_preconsume_core_sort_key(scheduler, lib),
    )

    lanes: List[LaneAssignment] = []
    remaining_priority = list(priority_libraries)
    remaining_fillers = list(filler_libraries)

    while remaining_priority:
        best_lane: Optional[LaneAssignment] = None
        best_score: Optional[Tuple[float, float, float, int]] = None

        seed_candidates = remaining_priority[: min(12, len(remaining_priority))]
        for seed in seed_candidates:
            probe_lane = _priority_36t_preconsume_create_lane(
                scheduler,
                machine_type=machine_type,
                machine_type_enum=machine_type_enum,
            )
            probe_libraries: List[EnhancedLibraryInfo] = []
            if not _priority_36t_preconsume_core_profile_can_add(
                scheduler,
                probe_libraries,
                seed,
            ):
                continue
            if not _priority_36t_preconsume_add_if_possible(scheduler, probe_lane, seed):
                continue
            probe_libraries.append(seed)

            for lib in remaining_priority[:80]:
                if id(lib) == id(seed):
                    continue
                if not _priority_36t_preconsume_core_profile_can_add(
                    scheduler,
                    probe_libraries,
                    lib,
                ):
                    continue
                if not _priority_36t_preconsume_add_if_possible(scheduler, probe_lane, lib):
                    continue
                probe_libraries.append(lib)

            for core_variant in _priority_36t_preconsume_core_variants(probe_libraries):
                candidate_lane = _priority_36t_preconsume_fill_and_validate_lane(
                    scheduler,
                    core_libraries=core_variant,
                    filler_libraries=remaining_fillers,
                    machine_type=machine_type,
                    machine_type_enum=machine_type_enum,
                )
                if candidate_lane is None:
                    continue
                lane_priority_libs = [
                    lib for lib in candidate_lane.libraries
                    if str(getattr(lib, "_priority_preconsume_bucket", "") or "").startswith("priority_")
                ]
                priority_gb = sum(_priority_36t_preconsume_data_gb(lib) for lib in lane_priority_libs)
                clinical_gb = sum(
                    _priority_36t_preconsume_data_gb(lib)
                    for lib in lane_priority_libs
                    if _normalize_priority_36t_preconsume_data_type(lib) == "临检"
                )
                total_gb = sum(_priority_36t_preconsume_data_gb(lib) for lib in candidate_lane.libraries)
                score = (
                    priority_gb,
                    clinical_gb,
                    total_gb,
                    -len(candidate_lane.libraries),
                )
                if best_score is None or score > best_score:
                    best_lane = candidate_lane
                    best_score = score

        if best_lane is None:
            break

        lanes.append(best_lane)
        used_ids = {id(lib) for lib in best_lane.libraries}
        remaining_priority = [lib for lib in remaining_priority if id(lib) not in used_ids]
        remaining_fillers = [lib for lib in remaining_fillers if id(lib) not in used_ids]

        lane_priority_summary = Counter(
            _normalize_priority_36t_preconsume_data_type(lib)
            for lib in best_lane.libraries
            if str(getattr(lib, "_priority_preconsume_bucket", "") or "").startswith("priority_")
        )
        lane_profile = _priority_36t_preconsume_profile_summary(
            scheduler,
            list(best_lane.libraries),
        )
        logger.info(
            "3.6T-NEW高优预消耗画像成Lane: lane={}, 高优={}个/{:.1f}G, 临检={}个, "
            "总量={:.1f}G, 10bp={:.1%}, 单端={:.1%}, 碱基不均={:.1%}, 客户={:.1%}",
            best_lane.lane_id,
            sum(lane_priority_summary.values()),
            sum(
                _priority_36t_preconsume_data_gb(lib)
                for lib in best_lane.libraries
                if str(getattr(lib, "_priority_preconsume_bucket", "") or "").startswith("priority_")
            ),
            lane_priority_summary.get("临检", 0),
            best_lane.total_data_gb,
            lane_profile["ten_bp_ratio"],
            lane_profile["single_ratio"],
            lane_profile["imbalance_ratio"],
            lane_profile["customer_ratio"],
        )

    used_ids = {id(lib) for lane in lanes for lib in lane.libraries}
    recycled_libraries = [lib for lib in libraries if id(lib) not in used_ids]
    return lanes, recycled_libraries


def _remove_priority_36t_preconsume_used_libraries(
    pool: List[EnhancedLibraryInfo],
    used_libraries: List[EnhancedLibraryInfo],
) -> List[EnhancedLibraryInfo]:
    """按文库身份和出现次数从原始候选池扣减已用于预成lane的文库。"""
    if not used_libraries:
        return list(pool or [])

    used_counts = Counter(
        _get_priority_36t_preconsume_library_identity(lib)
        for lib in list(used_libraries or [])
    )
    remaining_pool: List[EnhancedLibraryInfo] = []
    for lib in list(pool or []):
        identity = _get_priority_36t_preconsume_library_identity(lib)
        if used_counts.get(identity, 0) > 0:
            used_counts[identity] -= 1
            continue
        remaining_pool.append(lib)
    return remaining_pool


def _summarize_priority_36t_preconsume_libraries(
    libraries: List[EnhancedLibraryInfo],
) -> Dict[str, Any]:
    """汇总首轮3.6T预消耗文库的行数、数据量与类型分布。"""
    type_row_count: Counter[str] = Counter()
    type_data_gb: Dict[str, float] = {}
    total_data_gb = 0.0

    for lib in list(libraries or []):
        data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
        data_type = str(getattr(lib, "data_type", "") or "").strip() or "UNKNOWN"
        total_data_gb += data
        type_row_count[data_type] += 1
        type_data_gb[data_type] = round(type_data_gb.get(data_type, 0.0) + data, 1)

    return {
        "row_count": len(list(libraries or [])),
        "data_gb": round(total_data_gb, 1),
        "type_row_count": dict(type_row_count),
        "type_data_gb": dict(sorted(type_data_gb.items())),
    }


def _build_priority_36t_preconsume_inputs(
    *,
    allocator: ModeAllocator,
    dispatch_result: ModeDispatchResult,
) -> Priority36TPreconsumeInputs:
    """将1.1分流结果拆成3.6T-NEW预消耗所需的主料和补料。"""
    inputs = Priority36TPreconsumeInputs()
    priority_total_gb = 0.0
    bucketed_priority_candidates: List[Tuple[str, EnhancedLibraryInfo]] = []
    overflow_candidates: List[EnhancedLibraryInfo] = []

    for lib in list(dispatch_result.pool_36t_priority or []):
        priority_total_gb += float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
        forbidden_reason = allocator._check_1_1_forbidden(lib)
        if forbidden_reason:
            bucketed_priority_candidates.append(("priority_36t", lib))
            continue
        if allocator._is_priority_overflow_candidate_for_1_1(lib):
            overflow_candidates.append(lib)
            continue
        bucketed_priority_candidates.append(("priority_36t", lib))

    allowed_overflow_total_gb = 0.0
    if (
        getattr(allocator, "_priority_overflow_enabled", False)
        and priority_total_gb >= float(getattr(allocator, "_priority_overflow_trigger_min_pool_gb", 0.0) or 0.0)
    ):
        configured_max_overflow_gb = float(
            getattr(allocator, "_priority_overflow_max_total_gb", 0.0) or 0.0
        )
        if configured_max_overflow_gb > 0:
            allowed_overflow_total_gb = configured_max_overflow_gb

    selected_overflow_ids: Set[int] = set()
    selected_overflow_total_gb = 0.0
    if allowed_overflow_total_gb > 0:
        for lib in sorted(
            overflow_candidates,
            key=lambda item: _priority_1_1_overflow_sort_key(item, allocator=allocator),
        ):
            data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
            if selected_overflow_total_gb + data > allowed_overflow_total_gb + 1e-6:
                continue
            selected_overflow_ids.add(id(lib))
            selected_overflow_total_gb += data

    for lib in overflow_candidates:
        if id(lib) in selected_overflow_ids:
            allocator._apply_mode_1_1_quality_seed_hint(lib)
            bucketed_priority_candidates.append(("priority_1_1", lib))
        else:
            bucketed_priority_candidates.append(("priority_36t", lib))

    inputs.max_priority_lanes = allocator._resolve_priority_36t_lane_count(priority_total_gb)
    inputs.max_filler_gb_per_lane = allocator._get_priority_36t_preconsume_max_filler_gb_per_lane()
    if inputs.max_priority_lanes <= 0 or not bucketed_priority_candidates:
        return inputs

    max_preconsume_total_gb = inputs.max_priority_lanes * SCHEDULING_MAX_TARGET_CAP_GB
    preferred_data_types: Tuple[str, ...] = PRIORITY_36T_REDLINE_DATA_TYPE_ORDER
    selected_by_bucket, deferred_by_bucket, selected_priority_total_gb = (
        _select_priority_36t_preconsume_window(
            bucketed_priority_candidates,
            max_total_gb=max_preconsume_total_gb,
            preferred_data_types=preferred_data_types,
            spread_lane_count=inputs.max_priority_lanes,
        )
    )
    inputs.priority_libraries = list(selected_by_bucket["priority_1_1"])
    inputs.priority_forbidden_libraries = list(selected_by_bucket["priority_36t"])
    inputs.deferred_priority_libraries = list(deferred_by_bucket["priority_1_1"])
    inputs.deferred_priority_forbidden_libraries = list(deferred_by_bucket["priority_36t"])

    # 高优先级预消耗默认只消耗高优先级文库，不提前借走1.1普通池；
    # 普通文库先交给1.1首轮，只有1.1未排走时才回流后续3.6T。
    inputs.filler_forbidden_libraries = []
    if allocator._should_borrow_1_1_fillers_for_priority_36t_preconsume():
        filler_candidates = [
            lib
            for lib in (
                list(dispatch_result.pool_1_1_normal or [])
                + list(dispatch_result.pool_1_1_quality_risk or [])
                + list(dispatch_result.pool_1_1_quality_other or [])
            )
            if not allocator._is_priority_for_36t(lib)
        ]

        sorted_filler_candidates = sorted(
            filler_candidates,
            key=lambda item: _priority_36t_fill_candidate_sort_key(item, allocator=allocator),
        )
        inputs.all_filler_1_1_libraries = list(sorted_filler_candidates)
    else:
        inputs.all_filler_1_1_libraries = []

    logger.info(
        "3.6T-NEW高优先级预消耗窗口: lane上限={}, 原始高优={}个/{:.1f}G, 窗口高优={}个/{:.1f}G, "
        "回流1.1高优={}个, 留后续3.6高优={}个, 候选补位={}个/{:.1f}G",
        inputs.max_priority_lanes,
        len(dispatch_result.pool_36t_priority or []),
        priority_total_gb,
        len(inputs.priority_libraries) + len(inputs.priority_forbidden_libraries),
        selected_priority_total_gb,
        len(inputs.deferred_priority_libraries),
        len(inputs.deferred_priority_forbidden_libraries),
        len(inputs.all_filler_1_1_libraries),
        sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in inputs.all_filler_1_1_libraries),
    )
    return inputs


def _run_priority_36t_preconsume_scheduler_attempt(
    priority_libraries: List[EnhancedLibraryInfo],
    *,
    priority_fallback_to_36t_libraries: Optional[List[EnhancedLibraryInfo]] = None,
    filler_libraries_for_36t: Optional[List[EnhancedLibraryInfo]] = None,
    filler_libraries_from_1_1: Optional[List[EnhancedLibraryInfo]] = None,
) -> Priority36TPreconsumeResult:
    """执行单次首轮3.6T预消耗调度尝试。"""
    result = Priority36TPreconsumeResult(
        remaining_libraries=list(priority_libraries or []),
        remaining_priority_forbidden_libraries=list(priority_fallback_to_36t_libraries or []),
        remaining_filler_forbidden_libraries=list(filler_libraries_for_36t or []),
        remaining_filler_1_1_libraries=list(filler_libraries_from_1_1 or []),
        scheduling_succeeded=True,
    )
    priority_fallback_to_36t_libraries = list(priority_fallback_to_36t_libraries or [])
    filler_libraries_for_36t = list(filler_libraries_for_36t or [])
    filler_libraries_from_1_1 = list(filler_libraries_from_1_1 or [])

    grouped_libraries_by_bucket: Dict[str, Dict[str, List[EnhancedLibraryInfo]]] = {
        "priority_1_1": {},
        "priority_36t": {},
        "filler_36t": {},
        "filler_1_1": {},
    }

    def _group_libraries(
        libraries: List[EnhancedLibraryInfo],
        bucket: str,
    ) -> List[EnhancedLibraryInfo]:
        grouped_bucket = grouped_libraries_by_bucket.setdefault(bucket, {})
        grouped_bucket.clear()
        representatives: List[EnhancedLibraryInfo] = []
        for lib in list(libraries or []):
            identity = _get_priority_36t_preconsume_library_identity(lib)
            if identity not in grouped_bucket:
                grouped_bucket[identity] = [lib]
                representatives.append(lib)
            else:
                grouped_bucket[identity].append(lib)
        return representatives

    representative_priority_libraries = _group_libraries(
        list(priority_libraries or []),
        "priority_1_1",
    )
    representative_priority_fallback_to_36t_libraries = _group_libraries(
        priority_fallback_to_36t_libraries,
        "priority_36t",
    )
    representative_filler_libraries_for_36t = _group_libraries(
        filler_libraries_for_36t,
        "filler_36t",
    )
    representative_filler_libraries_from_1_1 = _group_libraries(
        filler_libraries_from_1_1,
        "filler_1_1",
    )

    all_input_libraries = (
        list(representative_priority_libraries)
        + representative_priority_fallback_to_36t_libraries
        + representative_filler_libraries_for_36t
        + representative_filler_libraries_from_1_1
    )
    if not all_input_libraries:
        return result

    bucket_by_key: Dict[str, str] = {}
    tagged_libraries: List[EnhancedLibraryInfo] = []

    def _tag_libraries(libraries: List[EnhancedLibraryInfo], bucket: str) -> None:
        for index, lib in enumerate(libraries):
            tag_key = f"{bucket}:{index}:{getattr(lib, 'origrec', '')}"
            setattr(lib, "_priority_preconsume_bucket", bucket)
            setattr(lib, "_priority_preconsume_key", tag_key)
            lib._current_seq_mode_raw = "3.6T-NEW"
            bucket_by_key[tag_key] = bucket
            tagged_libraries.append(lib)

    _tag_libraries(list(representative_priority_libraries), "priority_1_1")
    _tag_libraries(representative_priority_fallback_to_36t_libraries, "priority_36t")
    _tag_libraries(representative_filler_libraries_for_36t, "filler_36t")
    _tag_libraries(representative_filler_libraries_from_1_1, "filler_1_1")

    for lib in tagged_libraries:
        lib._current_seq_mode_raw = "3.6T-NEW"

    try:
        preconsume_libraries = deepcopy(list(all_input_libraries))
        priority_machine_type = _resolve_priority_36t_preconsume_routing_machine_type(
            representative_priority_libraries,
            representative_priority_fallback_to_36t_libraries,
        )
        # 高优预消耗阶段按业务规则从全局未排池补料，不再按原始机型切分补料池。
        # 这里只改调度尝试用的深拷贝对象，正式输出仍保留原始文库信息。
        for lib in preconsume_libraries:
            lib.eq_type = priority_machine_type
            lib.machine_type = _resolve_machine_type_enum_simple(priority_machine_type)
        scheduler_config = GreedyLaneConfig(
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
            enable_dedicated_imbalance_lane=False,
            enable_small_library_clustering=False,
            clustering_min_count=30,
            enable_non_10bp_dedicated_lane=False,
            enable_backbone_reservation=False,
        )
        scheduler = GreedyLaneScheduler(scheduler_config)
        if scheduler.pooling_optimizer:
            scheduler.pooling_optimizer.enabled = False

        machine_type = str(getattr(preconsume_libraries[0], "eq_type", "") or "")
        scheduler._batch_analysis_report = BatchAnalysisReport(
            total_library_count=len(preconsume_libraries),
            total_data_gb=_total_lane_data(preconsume_libraries),
            machine_type=machine_type,
        )
        disabled_plan = StrategyExecutionPlan()
        disabled_plan.enable_dedicated_imbalance_lane = False
        disabled_plan.enable_non_10bp_dedicated_lane = False
        disabled_plan.enable_backbone_reservation = False
        disabled_plan.enable_small_library_clustering = False
        scheduler._strategy_plan = disabled_plan

        profile_lanes, recycled_libraries = _build_priority_36t_preconsume_lanes_by_profile(
            scheduler,
            preconsume_libraries,
            machine_type=machine_type,
        )
        scheduling_stats = {
            "lane_count": len(profile_lanes),
            "unassigned_count": len(recycled_libraries),
        }
        kept_lanes: List[LaneAssignment] = []
        accepted_lane_capacities: List[Dict[str, Any]] = []
        rejected_lane_capacities: List[Dict[str, Any]] = []

        for lane in list(profile_lanes or []):
            lane_libraries = list(getattr(lane, "libraries", []) or [])
            has_priority = any(
                str(getattr(lib, "_priority_preconsume_bucket", "") or "").startswith("priority_")
                for lib in lane_libraries
            )
            if not has_priority:
                recycled_libraries.extend(lane_libraries)
                continue

            lane_is_valid, capacity_info = _is_priority_36t_preconsume_lane_capacity_valid(lane)
            if lane_is_valid:
                if not isinstance(lane.metadata, dict):
                    lane.metadata = {}
                lane.metadata["preconsume_capacity_validation"] = dict(capacity_info)
                accepted_lane_capacities.append(dict(capacity_info))
                kept_lanes.append(lane)
                continue

            rejected_lane_capacities.append(dict(capacity_info))
            recycled_libraries.extend(lane_libraries)

        remaining_by_bucket: Dict[str, List[EnhancedLibraryInfo]] = {
            "priority_1_1": [],
            "priority_36t": [],
            "filler_36t": [],
            "filler_1_1": [],
        }
        for lib in recycled_libraries:
            bucket = str(getattr(lib, "_priority_preconsume_bucket", "") or "")
            if bucket in remaining_by_bucket:
                identity = _get_priority_36t_preconsume_library_identity(lib)
                original_group = list(
                    grouped_libraries_by_bucket.get(bucket, {}).get(identity, [])
                )
                if original_group:
                    remaining_by_bucket[bucket].extend(original_group)
                else:
                    remaining_by_bucket[bucket].append(lib)

        result.lanes = kept_lanes
        result.remaining_libraries = list(remaining_by_bucket["priority_1_1"])
        result.remaining_priority_forbidden_libraries = list(remaining_by_bucket["priority_36t"])
        result.remaining_filler_forbidden_libraries = list(remaining_by_bucket["filler_36t"])
        result.remaining_filler_1_1_libraries = list(remaining_by_bucket["filler_1_1"])
        result.scheduling_stats = dict(scheduling_stats or {})
        if accepted_lane_capacities:
            result.scheduling_stats["accepted_lane_capacities"] = list(accepted_lane_capacities)
        if rejected_lane_capacities:
            result.scheduling_stats["rejected_lane_capacities"] = list(rejected_lane_capacities)

        for lane in result.lanes:
            if not isinstance(lane.metadata, dict):
                lane.metadata = {}
            lane.metadata["dispatch_stage"] = "priority_preconsume_36t"
            lane.metadata["selected_seq_mode"] = "3.6T-NEW"
            lane.metadata["seq_mode"] = "3.6T-NEW"
            lane.metadata["lcxms"] = "3.6T-NEW"
            lane.metadata["resolved_seq_mode"] = "3.6T-NEW"
            for lib in list(getattr(lane, "libraries", []) or []):
                lib._current_seq_mode_raw = "3.6T-NEW"
                lib.selected_seq_mode = "3.6T-NEW"
                lib.current_seq_mode = "3.6T-NEW"
                lib.lcxms = "3.6T-NEW"

        log_fn = logger.info if result.lanes else logger.debug
        log_fn(
            "3.6T-NEW高优先级预消耗完成: 高优输入={}, 36T禁排高优={}, 36T补料={}, 1.1补料={}, "
            "保留Lane={}, 剩余高优回1.1={}, 剩余高优留36T={}, 剩余36T补料={}, 剩余1.1补料={}",
            len(priority_libraries),
            len(priority_fallback_to_36t_libraries),
            len(filler_libraries_for_36t),
            len(filler_libraries_from_1_1),
            len(result.lanes),
            len(result.remaining_libraries),
            len(result.remaining_priority_forbidden_libraries),
            len(result.remaining_filler_forbidden_libraries),
            len(result.remaining_filler_1_1_libraries),
        )
    except Exception as exc:
        result.lanes = []
        result.remaining_libraries = list(priority_libraries)
        result.remaining_priority_forbidden_libraries = list(priority_fallback_to_36t_libraries)
        result.remaining_filler_forbidden_libraries = list(filler_libraries_for_36t)
        result.remaining_filler_1_1_libraries = list(filler_libraries_from_1_1)
        result.scheduling_stats = {}
        result.scheduling_succeeded = False
        result.scheduling_error = str(exc)
        logger.error("3.6T-NEW高优先级预消耗异常，优先文库保留到后续3.6T池: {}", exc)
    finally:
        for lib in tagged_libraries:
            lib._current_seq_mode_raw = ""
            if hasattr(lib, "_priority_preconsume_bucket"):
                delattr(lib, "_priority_preconsume_bucket")
            if hasattr(lib, "_priority_preconsume_key"):
                delattr(lib, "_priority_preconsume_key")

    return result


def _run_priority_36t_preconsume_stage(
    priority_libraries: List[EnhancedLibraryInfo],
    *,
    priority_fallback_to_36t_libraries: Optional[List[EnhancedLibraryInfo]] = None,
    filler_libraries_for_36t: Optional[List[EnhancedLibraryInfo]] = None,
    filler_libraries_from_1_1: Optional[List[EnhancedLibraryInfo]] = None,
    max_target_lanes: Optional[int] = None,
    max_filler_gb_per_lane: Optional[float] = None,
) -> Priority36TPreconsumeResult:
    """按首轮lane目标做迭代选池与补位重试，尽量先消耗3.6T高优先级文库。"""
    result = Priority36TPreconsumeResult(
        remaining_libraries=list(priority_libraries or []),
        remaining_priority_forbidden_libraries=list(priority_fallback_to_36t_libraries or []),
        remaining_filler_forbidden_libraries=list(filler_libraries_for_36t or []),
        remaining_filler_1_1_libraries=list(filler_libraries_from_1_1 or []),
        scheduling_succeeded=True,
    )

    remaining_priority_1_1 = list(priority_libraries or [])
    remaining_priority_36t = list(priority_fallback_to_36t_libraries or [])
    remaining_filler_36t = list(filler_libraries_for_36t or [])
    remaining_filler_1_1 = list(filler_libraries_from_1_1 or [])

    total_priority_gb = sum(
        float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
        for lib in list(remaining_priority_1_1) + list(remaining_priority_36t)
    )
    if total_priority_gb <= 0 and not remaining_filler_36t and not remaining_filler_1_1:
        return result

    resolved_max_target_lanes = 0
    resolved_max_filler_gb_per_lane = _resolve_priority_36t_preconsume_filler_limit(
        configured_limit_gb=max_filler_gb_per_lane,
        has_priority_libraries=(
            bool(remaining_priority_1_1)
            or bool(remaining_priority_36t)
        ),
    )
    if total_priority_gb > 0:
        derived_lane_target = max(
            1,
            min(
                len(remaining_priority_1_1) + len(remaining_priority_36t),
                math.ceil(total_priority_gb / SCHEDULING_MAX_TARGET_CAP_GB),
            ),
        )
        if max_target_lanes is not None and max_target_lanes > 0:
            resolved_max_target_lanes = min(
                int(max_target_lanes),
                len(remaining_priority_1_1) + len(remaining_priority_36t),
            )
        else:
            resolved_max_target_lanes = derived_lane_target

    aggregated_stats: Dict[str, Any] = {
        "attempted_lane_target": resolved_max_target_lanes,
        "configured_lane_target": int(max_target_lanes or 0),
        "attempt_batches": 0,
        "successful_batches": 0,
        "lane_count": 0,
    }
    consecutive_empty_batches = 0
    preconsume_stop_requested = False

    def _consume_used_bucket_libraries(
        pool_libraries: List[EnhancedLibraryInfo],
        used_bucket_libraries: List[EnhancedLibraryInfo],
    ) -> List[EnhancedLibraryInfo]:
        used_identities = {
            _get_priority_36t_preconsume_library_identity(lib)
            for lib in list(used_bucket_libraries or [])
        }
        if not used_identities:
            return []
        return [
            lib
            for lib in list(pool_libraries or [])
            if _get_priority_36t_preconsume_library_identity(lib) in used_identities
        ]

    def _try_attempt_plan(
        *,
        attempt_priority_1_1: List[EnhancedLibraryInfo],
        attempt_priority_36t: List[EnhancedLibraryInfo],
        attempted_fill_targets: List[float],
        strategy_name: str,
        selected_priority_representative_gb: float,
    ) -> bool:
        nonlocal remaining_priority_1_1
        nonlocal remaining_priority_36t
        nonlocal remaining_filler_36t
        nonlocal remaining_filler_1_1
        nonlocal consecutive_empty_batches
        nonlocal preconsume_stop_requested

        if not attempt_priority_1_1 and not attempt_priority_36t:
            return False

        ordered_filler_1_1 = sorted(
            list(remaining_filler_1_1),
            key=lambda lib: _priority_36t_stage_fill_candidate_sort_key(
                lib,
                preferred_machine_types=(),
            ),
        )

        for filler_target_gb in list(attempted_fill_targets or []):
            aggregated_stats["attempt_batches"] = int(aggregated_stats["attempt_batches"]) + 1
            if attempt_priority_1_1 or attempt_priority_36t:
                # 高优文库前置消耗时，补料允许从当前全局未排普通文库中任意挑选；
                # 不能先按缺口截断候选，否则前段候选被红线拒绝后就没有替补可试。
                attempt_filler_1_1 = list(ordered_filler_1_1)
            else:
                attempt_filler_1_1 = _select_priority_36t_preconsume_fillers(
                    ordered_filler_1_1,
                    target_gb=filler_target_gb,
                )
            attempt_result = _run_priority_36t_preconsume_scheduler_attempt(
                list(attempt_priority_1_1),
                priority_fallback_to_36t_libraries=list(attempt_priority_36t),
                filler_libraries_for_36t=list(remaining_filler_36t),
                filler_libraries_from_1_1=list(attempt_filler_1_1),
            )
            if not attempt_result.scheduling_succeeded:
                result.scheduling_succeeded = False
                result.scheduling_error = attempt_result.scheduling_error
                result.scheduling_stats = dict(attempt_result.scheduling_stats or {})
                return False

            attempt_stats = dict(attempt_result.scheduling_stats or {})
            if attempt_stats.get("accepted_lane_capacities"):
                aggregated_stats.setdefault("accepted_lane_capacities", []).extend(
                    deepcopy(list(attempt_stats.get("accepted_lane_capacities", []) or []))
                )
            if attempt_stats.get("rejected_lane_capacities"):
                aggregated_stats.setdefault("rejected_lane_capacities", []).extend(
                    deepcopy(list(attempt_stats.get("rejected_lane_capacities", []) or []))
                )

            if not attempt_result.lanes:
                consecutive_empty_batches += 1
                if consecutive_empty_batches >= 1:
                    logger.info(
                        "3.6T-NEW高优先级预消耗提前停止: 连续{}轮未新增Lane",
                        consecutive_empty_batches,
                    )
                    preconsume_stop_requested = True
                    return False
                continue

            remaining_lane_slots = resolved_max_target_lanes - len(result.lanes)
            accepted_lanes = list(attempt_result.lanes[:remaining_lane_slots])
            if not accepted_lanes:
                return True
            consecutive_empty_batches = 0

            used_by_bucket: Dict[str, List[EnhancedLibraryInfo]] = {
                "priority_1_1": [],
                "priority_36t": [],
                "filler_36t": [],
                "filler_1_1": [],
            }
            for lane in accepted_lanes:
                result.lanes.append(lane)
                for lib in list(getattr(lane, "libraries", []) or []):
                    bucket = str(getattr(lib, "_priority_preconsume_bucket", "") or "")
                    if bucket in used_by_bucket:
                        used_by_bucket[bucket].append(lib)

            consumed_batch_by_bucket: Dict[str, List[EnhancedLibraryInfo]] = {
                "priority_1_1": _consume_used_bucket_libraries(
                    remaining_priority_1_1,
                    used_by_bucket["priority_1_1"],
                ),
                "priority_36t": _consume_used_bucket_libraries(
                    remaining_priority_36t,
                    used_by_bucket["priority_36t"],
                ),
                "filler_36t": _consume_used_bucket_libraries(
                    remaining_filler_36t,
                    used_by_bucket["filler_36t"],
                ),
                "filler_1_1": _consume_used_bucket_libraries(
                    remaining_filler_1_1,
                    used_by_bucket["filler_1_1"],
                ),
            }

            result.consumed_priority_libraries.extend(
                deepcopy(consumed_batch_by_bucket["priority_1_1"])
            )
            result.consumed_priority_forbidden_libraries.extend(
                deepcopy(consumed_batch_by_bucket["priority_36t"])
            )
            result.consumed_filler_forbidden_libraries.extend(
                deepcopy(consumed_batch_by_bucket["filler_36t"])
            )
            result.consumed_filler_1_1_libraries.extend(
                deepcopy(consumed_batch_by_bucket["filler_1_1"])
            )

            remaining_priority_1_1 = _remove_priority_36t_preconsume_used_libraries(
                remaining_priority_1_1,
                consumed_batch_by_bucket["priority_1_1"],
            )
            remaining_priority_36t = _remove_priority_36t_preconsume_used_libraries(
                remaining_priority_36t,
                consumed_batch_by_bucket["priority_36t"],
            )
            remaining_filler_36t = _remove_priority_36t_preconsume_used_libraries(
                remaining_filler_36t,
                consumed_batch_by_bucket["filler_36t"],
            )
            remaining_filler_1_1 = _remove_priority_36t_preconsume_used_libraries(
                remaining_filler_1_1,
                consumed_batch_by_bucket["filler_1_1"],
            )
            aggregated_stats["successful_batches"] = int(aggregated_stats["successful_batches"]) + 1
            aggregated_stats["lane_count"] = len(result.lanes)

            logger.info(
                "3.6T-NEW高优先级预消耗批次成功: 策略={}, 代表高优={:.1f}G, 补位目标={:.0f}G, "
                "新增Lane={}, 累计Lane={}, 剩余高优={:.1f}G",
                strategy_name,
                selected_priority_representative_gb,
                filler_target_gb,
                len(accepted_lanes),
                len(result.lanes),
                sum(
                    float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
                    for lib in list(remaining_priority_1_1) + list(remaining_priority_36t)
                ),
            )
            return True

        return False

    while len(result.lanes) < resolved_max_target_lanes:
        if preconsume_stop_requested:
            break
        bucketed_priority_candidates: List[Tuple[str, EnhancedLibraryInfo]] = [
            ("priority_1_1", lib) for lib in list(remaining_priority_1_1)
        ]
        bucketed_priority_candidates.extend(
            ("priority_36t", lib) for lib in list(remaining_priority_36t)
        )
        if not bucketed_priority_candidates:
            break

        batch_succeeded = False
        remaining_lane_slots = max(0, resolved_max_target_lanes - len(result.lanes))
        # 高优预消耗的补料来自全局未排池，不按原始机型分桶或设置机型偏好。
        attempt_machine_preferences: List[Tuple[str, ...]] = [()]

        preferred_data_types: Tuple[str, ...] = PRIORITY_36T_REDLINE_DATA_TYPE_ORDER
        remaining_priority_total_gb = sum(
            float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
            for _, lib in bucketed_priority_candidates
        )
        data_type_attempt_orders: List[Tuple[str, ...]] = []
        present_priority_data_types = {
            _normalize_priority_36t_preconsume_data_type(lib)
            for _, lib in bucketed_priority_candidates
        }
        for data_type in preferred_data_types:
            if data_type in present_priority_data_types:
                data_type_attempt_orders.append((data_type,))
        data_type_attempt_orders.append(preferred_data_types)
        data_type_attempt_orders.sort(
            key=lambda item: (
                0 if len(item) > 1 else 1,
                min(
                    (
                        preferred_data_types.index(data_type)
                        for data_type in item
                        if data_type in preferred_data_types
                    ),
                    default=len(preferred_data_types),
                ),
            )
        )
        seen_data_type_orders: Set[Tuple[str, ...]] = set()
        data_type_attempt_orders = [
            item
            for item in data_type_attempt_orders
            if not (item in seen_data_type_orders or seen_data_type_orders.add(item))
        ]

        for preferred_machine_types in attempt_machine_preferences:
            machine_strategy_prefix = (
                preferred_machine_types[0] if preferred_machine_types else "generic"
            )
            for attempt_data_types in data_type_attempt_orders:
                filtered_priority_candidates = [
                    (bucket, lib)
                    for bucket, lib in bucketed_priority_candidates
                    if _normalize_priority_36t_preconsume_data_type(lib) in set(attempt_data_types)
                ]
                if not filtered_priority_candidates:
                    continue
                filtered_priority_total_gb = sum(
                    float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
                    for _, lib in filtered_priority_candidates
                )
                data_type_strategy_prefix = "_".join(attempt_data_types) or "mixed"
                for priority_window_target_gb in PRIORITY_36T_PRECONSUME_RETRY_WINDOW_TARGETS_GB:
                    selected_by_bucket, _, selected_priority_total_gb = _select_priority_36t_preconsume_window(
                        filtered_priority_candidates,
                        max_total_gb=min(priority_window_target_gb, filtered_priority_total_gb),
                        preferred_machine_types=preferred_machine_types,
                        preferred_data_types=attempt_data_types,
                        spread_lane_count=remaining_lane_slots,
                    )
                    attempt_priority_1_1 = list(selected_by_bucket["priority_1_1"])
                    attempt_priority_36t = list(selected_by_bucket["priority_36t"])
                    if not attempt_priority_1_1 and not attempt_priority_36t:
                        continue

                    dynamic_fill_target_gb = max(
                        0.0,
                        min(
                            resolved_max_filler_gb_per_lane,
                            PRIORITY_36T_PRECONSUME_LANE_FLOOR_GB - selected_priority_total_gb,
                        ),
                    )
                    attempted_fill_targets = list(PRIORITY_36T_PRECONSUME_RETRY_FILLER_TARGETS_GB)
                    if dynamic_fill_target_gb > 0:
                        attempted_fill_targets.insert(0, dynamic_fill_target_gb)
                    attempted_fill_targets = list(dict.fromkeys(attempted_fill_targets))

                    positive_fill_targets = sorted(
                        [
                            target
                            for target in attempted_fill_targets
                            if target > 0 and target <= resolved_max_filler_gb_per_lane + 1e-6
                        ],
                        reverse=remaining_lane_slots > 1,
                    )
                    attempted_fill_targets = list(positive_fill_targets)
                    if 0.0 in PRIORITY_36T_PRECONSUME_RETRY_FILLER_TARGETS_GB:
                        attempted_fill_targets.append(0.0)

                    batch_succeeded = _try_attempt_plan(
                        attempt_priority_1_1=list(attempt_priority_1_1),
                        attempt_priority_36t=list(attempt_priority_36t),
                        attempted_fill_targets=list(attempted_fill_targets),
                        strategy_name=(
                            f"machine_{machine_strategy_prefix}_{data_type_strategy_prefix}_window_"
                            f"{int(min(priority_window_target_gb, filtered_priority_total_gb))}"
                        ),
                        selected_priority_representative_gb=float(selected_priority_total_gb),
                    )
                    if result.scheduling_succeeded is False:
                        return result
                    if batch_succeeded:
                        break
                    if preconsume_stop_requested:
                        break

                if batch_succeeded:
                    break
                if preconsume_stop_requested:
                    break

            if batch_succeeded:
                break
            if preconsume_stop_requested:
                break

        if not batch_succeeded:
            break

    result.remaining_libraries = list(remaining_priority_1_1)
    result.remaining_priority_forbidden_libraries = list(remaining_priority_36t)
    result.remaining_filler_forbidden_libraries = list(remaining_filler_36t)
    result.remaining_filler_1_1_libraries = list(remaining_filler_1_1)
    aggregated_stats["lane_count"] = len(result.lanes)
    aggregated_stats["physical_lane_sizes_gb"] = [
        round(_total_lane_data(list(getattr(lane, "libraries", []) or [])), 1)
        for lane in result.lanes
    ]
    aggregated_stats["consumed_priority_1_1"] = _summarize_priority_36t_preconsume_libraries(
        result.consumed_priority_libraries
    )
    aggregated_stats["consumed_priority_36t"] = _summarize_priority_36t_preconsume_libraries(
        result.consumed_priority_forbidden_libraries
    )
    aggregated_stats["consumed_priority_total"] = _summarize_priority_36t_preconsume_libraries(
        list(result.consumed_priority_libraries)
        + list(result.consumed_priority_forbidden_libraries)
    )
    aggregated_stats["remaining_priority_1_1"] = _summarize_priority_36t_preconsume_libraries(
        result.remaining_libraries
    )
    aggregated_stats["remaining_priority_36t"] = _summarize_priority_36t_preconsume_libraries(
        result.remaining_priority_forbidden_libraries
    )
    aggregated_stats["remaining_priority_total"] = _summarize_priority_36t_preconsume_libraries(
        list(result.remaining_libraries) + list(result.remaining_priority_forbidden_libraries)
    )
    result.scheduling_stats = aggregated_stats

    logger.info(
        "3.6T-NEW高优先级预消耗汇总: 累计Lane={}, 剩余高优回1.1={}, 剩余高优留36T={}, "
        "剩余36T补料={}, 剩余1.1补料={}",
        len(result.lanes),
        len(result.remaining_libraries),
        len(result.remaining_priority_forbidden_libraries),
        len(result.remaining_filler_forbidden_libraries),
        len(result.remaining_filler_1_1_libraries),
    )
    return result


def _drain_remaining_priority_to_1_1_first_round(
    normal_libs_for_36t: List[EnhancedLibraryInfo],
    *,
    allocator: ModeAllocator,
    max_total_gb: float,
) -> Tuple[List[EnhancedLibraryInfo], List[EnhancedLibraryInfo], float]:
    """将3.6T池剩余高优文库按总量上限回灌到1.1首轮。"""
    if max_total_gb <= 0:
        return [], list(normal_libs_for_36t or []), 0.0

    normal_pool = list(normal_libs_for_36t or [])
    priority_candidates = [
        lib for lib in normal_pool
        if allocator._is_priority_for_36t(lib)
    ]
    if not priority_candidates:
        return [], normal_pool, 0.0

    selected_ids: Set[int] = set()
    selected_libs: List[EnhancedLibraryInfo] = []
    selected_total_gb = 0.0

    sorted_desc = sorted(
        priority_candidates,
        key=lambda lib: float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
        reverse=True,
    )
    for lib in sorted_desc:
        lib_data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
        if lib_data <= 0:
            continue
        if selected_total_gb + lib_data <= max_total_gb + 1e-6:
            selected_ids.add(id(lib))
            selected_libs.append(lib)
            selected_total_gb += lib_data

    remaining_budget = max_total_gb - selected_total_gb
    if remaining_budget > 1e-6:
        sorted_asc = sorted(
            priority_candidates,
            key=lambda lib: float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
        )
        for lib in sorted_asc:
            lib_id = id(lib)
            if lib_id in selected_ids:
                continue
            lib_data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
            if lib_data <= 0:
                continue
            if selected_total_gb + lib_data <= max_total_gb + 1e-6:
                selected_ids.add(lib_id)
                selected_libs.append(lib)
                selected_total_gb += lib_data
                remaining_budget = max_total_gb - selected_total_gb
                if remaining_budget <= 1e-6:
                    break

    remaining_pool = [lib for lib in normal_pool if id(lib) not in selected_ids]
    return selected_libs, remaining_pool, selected_total_gb


def _cap_priority_for_1_1_first_round_pool(
    libraries: List[EnhancedLibraryInfo],
    *,
    allocator: ModeAllocator,
    max_total_gb: float,
) -> Tuple[List[EnhancedLibraryInfo], List[EnhancedLibraryInfo], float]:
    """限制1.1首轮池中的高优文库总量，超出部分剔除出首轮池。"""
    pool = list(libraries or [])
    if not pool:
        return [], [], 0.0

    normal_pool: List[EnhancedLibraryInfo] = []
    priority_pool: List[EnhancedLibraryInfo] = []
    for lib in pool:
        if allocator._is_priority_for_36t(lib):
            priority_pool.append(lib)
        else:
            normal_pool.append(lib)

    if not priority_pool:
        return pool, [], 0.0
    if max_total_gb <= 0:
        return normal_pool, list(priority_pool), 0.0

    selected_priority_ids: Set[int] = set()
    selected_total_gb = 0.0
    sorted_desc = sorted(
        priority_pool,
        key=lambda lib: float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
        reverse=True,
    )
    for lib in sorted_desc:
        lib_data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
        if lib_data <= 0:
            continue
        if selected_total_gb + lib_data <= max_total_gb + 1e-6:
            selected_priority_ids.add(id(lib))
            selected_total_gb += lib_data

    remaining_budget = max_total_gb - selected_total_gb
    if remaining_budget > 1e-6:
        sorted_asc = sorted(
            priority_pool,
            key=lambda lib: float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
        )
        for lib in sorted_asc:
            lib_id = id(lib)
            if lib_id in selected_priority_ids:
                continue
            lib_data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
            if lib_data <= 0:
                continue
            if selected_total_gb + lib_data <= max_total_gb + 1e-6:
                selected_priority_ids.add(lib_id)
                selected_total_gb += lib_data
                remaining_budget = max_total_gb - selected_total_gb
                if remaining_budget <= 1e-6:
                    break

    selected_pool: List[EnhancedLibraryInfo] = []
    overflow_priority: List[EnhancedLibraryInfo] = []
    for lib in pool:
        if not allocator._is_priority_for_36t(lib):
            selected_pool.append(lib)
            continue
        if id(lib) in selected_priority_ids:
            selected_pool.append(lib)
        else:
            overflow_priority.append(lib)

    return selected_pool, overflow_priority, selected_total_gb


def _enforce_mode_1_1_priority_cap_per_lane(
    solution: Any,
    *,
    allocator: ModeAllocator,
    max_priority_gb_per_lane: float,
) -> Dict[str, float]:
    """限制1.1首轮单条Lane可承载的高优文库总量，超出部分回退到未分配池。"""
    lanes = list(getattr(solution, "lane_assignments", []) or [])
    if not lanes:
        return {
            "adjusted_lanes": 0,
            "overflow_libraries": 0,
            "kept_priority_gb": 0.0,
            "removed_priority_gb": 0.0,
        }

    unassigned = list(getattr(solution, "unassigned_libraries", []) or [])
    adjusted_lanes = 0
    overflow_libraries = 0
    kept_priority_gb = 0.0
    removed_priority_gb = 0.0

    for lane in lanes:
        lane_libraries = list(getattr(lane, "libraries", []) or [])
        if not lane_libraries:
            continue

        secondary_36t_only_libraries = [
            lib for lib in lane_libraries
            if allocator._is_priority_for_36t(lib)
            and _is_36t_only_secondary_priority(lib, allocator)
        ]
        priority_libraries = [
            lib for lib in lane_libraries
            if allocator._is_priority_for_36t(lib)
            and not _is_36t_only_secondary_priority(lib, allocator)
        ]
        if not priority_libraries and not secondary_36t_only_libraries:
            continue

        selected_priority_ids: Set[int] = set()
        selected_priority_gb = 0.0
        if max_priority_gb_per_lane > 0:
            sorted_desc = sorted(
                priority_libraries,
                key=lambda lib: float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
                reverse=True,
            )
            for lib in sorted_desc:
                lib_data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
                if lib_data <= 0:
                    continue
                if selected_priority_gb + lib_data <= max_priority_gb_per_lane + 1e-6:
                    selected_priority_ids.add(id(lib))
                    selected_priority_gb += lib_data

            remaining_budget = max_priority_gb_per_lane - selected_priority_gb
            if remaining_budget > 1e-6:
                sorted_asc = sorted(
                    priority_libraries,
                    key=lambda lib: float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
                )
                for lib in sorted_asc:
                    lib_id = id(lib)
                    if lib_id in selected_priority_ids:
                        continue
                    lib_data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
                    if lib_data <= 0:
                        continue
                    if selected_priority_gb + lib_data <= max_priority_gb_per_lane + 1e-6:
                        selected_priority_ids.add(lib_id)
                        selected_priority_gb += lib_data
                        remaining_budget = max_priority_gb_per_lane - selected_priority_gb
                        if remaining_budget <= 1e-6:
                            break

        kept_libraries: List[EnhancedLibraryInfo] = []
        overflow_for_lane: List[EnhancedLibraryInfo] = []
        for lib in lane_libraries:
            if _is_36t_only_secondary_priority(lib, allocator):
                overflow_for_lane.append(lib)
                continue
            if not allocator._is_priority_for_36t(lib):
                kept_libraries.append(lib)
                continue
            if id(lib) in selected_priority_ids:
                kept_libraries.append(lib)
            else:
                overflow_for_lane.append(lib)

        kept_priority_gb += selected_priority_gb
        removed_priority_gb += _total_lane_data(overflow_for_lane)
        if not overflow_for_lane:
            continue

        adjusted_lanes += 1
        overflow_libraries += len(overflow_for_lane)
        lane.libraries = kept_libraries
        lane.total_data_gb = _total_lane_data(kept_libraries)
        lane.calculate_metrics()
        unassigned.extend(overflow_for_lane)

        logger.info(
            "1.1首轮Lane高优封顶生效: lane={}, 保留高优{:.1f}G, 回退{}个/{:.1f}G到未分配池 (单Lane上限{:.1f}G)",
            getattr(lane, "lane_id", ""),
            selected_priority_gb,
            len(overflow_for_lane),
            _total_lane_data(overflow_for_lane),
            max_priority_gb_per_lane,
        )

    removed_empty_lanes = 0
    kept_lanes: List[LaneAssignment] = []
    for lane in lanes:
        if list(getattr(lane, "libraries", []) or []):
            kept_lanes.append(lane)
        else:
            removed_empty_lanes += 1
    if removed_empty_lanes:
        logger.info("1.1首轮高优封顶后移除空Lane: {}条", removed_empty_lanes)

    solution.lane_assignments = kept_lanes
    solution.unassigned_libraries = unassigned
    return {
        "adjusted_lanes": adjusted_lanes,
        "overflow_libraries": overflow_libraries,
        "kept_priority_gb": kept_priority_gb,
        "removed_priority_gb": removed_priority_gb,
    }


def _enforce_mode_1_1_add_test_cap_per_lane(
    solution: Any,
    *,
    max_add_test_gb_per_lane: float,
) -> Dict[str, float]:
    """限制1.1单条Lane内加测/混合文库总量，超出部分回退到未分配池。"""
    lanes = list(getattr(solution, "lane_assignments", []) or [])
    if not lanes or max_add_test_gb_per_lane <= 0:
        return {
            "adjusted_lanes": 0,
            "overflow_libraries": 0,
            "kept_add_test_gb": 0.0,
            "removed_add_test_gb": 0.0,
        }

    unassigned = list(getattr(solution, "unassigned_libraries", []) or [])
    adjusted_lanes = 0
    overflow_libraries = 0
    kept_add_test_gb = 0.0
    removed_add_test_gb = 0.0

    for lane in lanes:
        lane_libraries = list(getattr(lane, "libraries", []) or [])
        if not lane_libraries:
            continue
        if not _is_mode_1_1_lane_context(lane, lane_libraries):
            continue
        add_test_libraries = [
            lib for lib in lane_libraries
            if _is_mode_1_1_add_test_limited_library(lib)
        ]
        if not add_test_libraries:
            continue

        selected_ids: Set[int] = set()
        selected_gb = 0.0
        for lib in sorted(
            add_test_libraries,
            key=lambda item: float(getattr(item, "contract_data_raw", 0.0) or 0.0),
            reverse=True,
        ):
            lib_data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
            if lib_data <= 0:
                continue
            if selected_gb + lib_data <= max_add_test_gb_per_lane + 1e-6:
                selected_ids.add(id(lib))
                selected_gb += lib_data

        remaining_budget = max_add_test_gb_per_lane - selected_gb
        if remaining_budget > 1e-6:
            for lib in sorted(
                add_test_libraries,
                key=lambda item: float(getattr(item, "contract_data_raw", 0.0) or 0.0),
            ):
                lib_id = id(lib)
                if lib_id in selected_ids:
                    continue
                lib_data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
                if lib_data <= 0:
                    continue
                if selected_gb + lib_data <= max_add_test_gb_per_lane + 1e-6:
                    selected_ids.add(lib_id)
                    selected_gb += lib_data
                    remaining_budget = max_add_test_gb_per_lane - selected_gb
                    if remaining_budget <= 1e-6:
                        break

        kept_libraries: List[EnhancedLibraryInfo] = []
        overflow_for_lane: List[EnhancedLibraryInfo] = []
        for lib in lane_libraries:
            if not _is_mode_1_1_add_test_limited_library(lib) or id(lib) in selected_ids:
                kept_libraries.append(lib)
            else:
                overflow_for_lane.append(lib)

        kept_add_test_gb += selected_gb
        removed_add_test_gb += _total_lane_data(overflow_for_lane)
        if not overflow_for_lane:
            continue

        adjusted_lanes += 1
        overflow_libraries += len(overflow_for_lane)
        lane.libraries = kept_libraries
        lane.total_data_gb = _total_lane_data(kept_libraries)
        lane.calculate_metrics()
        unassigned.extend(overflow_for_lane)
        logger.info(
            "1.1首轮Lane加测/混合封顶生效: lane={}, 保留{:.1f}G, 回退{}个/{:.1f}G到未分配池 (单Lane上限{:.1f}G)",
            getattr(lane, "lane_id", ""),
            selected_gb,
            len(overflow_for_lane),
            _total_lane_data(overflow_for_lane),
            max_add_test_gb_per_lane,
        )

    kept_lanes: List[LaneAssignment] = []
    removed_empty_lanes = 0
    for lane in lanes:
        if list(getattr(lane, "libraries", []) or []):
            kept_lanes.append(lane)
        else:
            removed_empty_lanes += 1
    if removed_empty_lanes:
        logger.info("1.1首轮加测/混合封顶后移除空Lane: {}条", removed_empty_lanes)

    solution.lane_assignments = kept_lanes
    solution.unassigned_libraries = unassigned
    return {
        "adjusted_lanes": adjusted_lanes,
        "overflow_libraries": overflow_libraries,
        "kept_add_test_gb": kept_add_test_gb,
        "removed_add_test_gb": removed_add_test_gb,
    }


def _enforce_mode_1_1_add_test_cap_and_cleanup(
    solution: Any,
    validator: Any,
    *,
    max_add_test_gb_per_lane: float,
    stage_label: str,
) -> Dict[str, float]:
    """执行1.1加测/混合封顶，并立即复核容量等终态规则。"""
    cap_stats = _enforce_mode_1_1_add_test_cap_per_lane(
        solution,
        max_add_test_gb_per_lane=max_add_test_gb_per_lane,
    )
    if cap_stats["adjusted_lanes"] > 0:
        logger.info(
            "{}单Lane加测/混合封顶完成: 调整Lane={}, 回退文库={}个/{:.1f}G",
            stage_label,
            int(cap_stats["adjusted_lanes"]),
            int(cap_stats["overflow_libraries"]),
            cap_stats["removed_add_test_gb"],
        )
        cleanup_stats = _final_non_package_validation_cleanup(solution, validator)
        cap_stats["cleanup_removed_lanes"] = cleanup_stats["removed_lanes"]
        cap_stats["cleanup_recovered_libs"] = cleanup_stats["recovered_libs"]
        if cleanup_stats["removed_lanes"] > 0:
            logger.warning(
                "{}封顶后二次总复核: 淘汰{}条不合规Lane，回收{}个文库",
                stage_label,
                cleanup_stats["removed_lanes"],
                cleanup_stats["recovered_libs"],
            )
    else:
        cap_stats["cleanup_removed_lanes"] = 0
        cap_stats["cleanup_recovered_libs"] = 0
    return cap_stats


def _apply_mode_1_1_add_test_cap_to_prebuilt_lanes(
    lanes: List[LaneAssignment],
    remaining_libraries: List[EnhancedLibraryInfo],
    *,
    max_add_test_gb_per_lane: float,
    stage_label: str,
) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo], Dict[str, float]]:
    """对预构建的1.1 Lane执行加测/混合封顶，溢出文库回到后续候选池。"""
    from types import SimpleNamespace

    if not lanes:
        return list(lanes or []), list(remaining_libraries or []), {
            "adjusted_lanes": 0,
            "overflow_libraries": 0,
            "kept_add_test_gb": 0.0,
            "removed_add_test_gb": 0.0,
        }
    cap_solution = SimpleNamespace(
        lane_assignments=list(lanes or []),
        unassigned_libraries=list(remaining_libraries or []),
    )
    cap_stats = _enforce_mode_1_1_add_test_cap_per_lane(
        cap_solution,
        max_add_test_gb_per_lane=max_add_test_gb_per_lane,
    )
    if cap_stats["adjusted_lanes"] > 0:
        logger.info(
            "{}单Lane加测/混合封顶完成: 调整Lane={}, 回退文库={}个/{:.1f}G",
            stage_label,
            int(cap_stats["adjusted_lanes"]),
            int(cap_stats["overflow_libraries"]),
            cap_stats["removed_add_test_gb"],
        )
    return (
        list(cap_solution.lane_assignments or []),
        list(cap_solution.unassigned_libraries or []),
        cap_stats,
    )


def _consume_mode_1_1_priority_from_unassigned(
    solution: Any,
    *,
    allocator: ModeAllocator,
    max_priority_gb_per_lane: float,
) -> Dict[str, float]:
    """尝试将1.1首轮未分配池中的高优文库回填到已有1.1 Lane。"""
    if max_priority_gb_per_lane <= 0:
        return {
            "consumed_libraries": 0,
            "consumed_gb": 0.0,
            "changed_lanes": 0,
            "remaining_priority_libraries": 0,
        }

    lanes = list(getattr(solution, "lane_assignments", []) or [])
    unassigned = list(getattr(solution, "unassigned_libraries", []) or [])
    if not lanes or not unassigned:
        return {
            "consumed_libraries": 0,
            "consumed_gb": 0.0,
            "changed_lanes": 0,
            "remaining_priority_libraries": len(
                [lib for lib in unassigned if allocator._is_priority_for_36t(lib)]
            ),
        }

    validator = LaneValidator(strict_mode=True)
    priority_candidates = [
        lib for lib in unassigned
        if allocator._is_priority_for_36t(lib)
        and not _is_36t_only_secondary_priority(lib, allocator)
        and not _should_library_split_by_rules(lib)
    ]
    if not priority_candidates:
        return {
            "consumed_libraries": 0,
            "consumed_gb": 0.0,
            "changed_lanes": 0,
            "remaining_priority_libraries": 0,
        }

    priority_candidates = sorted(
        priority_candidates,
        key=lambda lib: float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
        reverse=True,
    )
    consumed_ids: Set[int] = set()
    changed_lane_ids: Set[str] = set()
    consumed_gb = 0.0

    def _current_priority_gb(lane: LaneAssignment) -> float:
        return _total_lane_data(
            [
                lib for lib in list(getattr(lane, "libraries", []) or [])
                if allocator._is_priority_for_36t(lib)
            ]
        )

    for lib in priority_candidates:
        lib_data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
        if lib_data <= 0:
            continue

        candidate_lanes = sorted(
            [
                lane
                for lane in lanes
                if not _is_explicit_dedicated_imbalance_lane(lane)
            ],
            key=lambda lane: (
                _current_priority_gb(lane),
                -float(getattr(lane, "total_data_gb", 0.0) or 0.0),
            ),
        )
        for lane in candidate_lanes:
            if str(getattr(lane, "lane_id", "") or "").startswith("NB_"):
                lib_10bp, _ = _split_10bp_and_non_10bp([lib], validator)
                if lib_10bp:
                    continue
            lane_priority_gb = _current_priority_gb(lane)
            if lane_priority_gb + lib_data > max_priority_gb_per_lane + 1e-6:
                continue

            trial_libraries = list(getattr(lane, "libraries", []) or []) + [lib]
            validation_result = _validate_lane_state(
                validator,
                lane,
                trial_libraries,
            )
            if not getattr(validation_result, "is_valid", False):
                continue

            lane.libraries = trial_libraries
            lane.total_data_gb = _total_lane_data(trial_libraries)
            lane.calculate_metrics()
            consumed_ids.add(id(lib))
            consumed_gb += lib_data
            changed_lane_ids.add(getattr(lane, "lane_id", ""))
            logger.info(
                "1.1首轮高优回填成功: lane={}, 文库={}, 数据量{:.1f}G, lane内高优累计{:.1f}G",
                getattr(lane, "lane_id", ""),
                getattr(lib, "origrec", "") or getattr(lib, "_origrec_key", ""),
                lib_data,
                _current_priority_gb(lane),
            )
            break

    if consumed_ids:
        solution.lane_assignments = lanes
        solution.unassigned_libraries = [
            lib for lib in unassigned
            if id(lib) not in consumed_ids
        ]

    remaining_priority_libraries = len(
        [
            lib for lib in list(getattr(solution, "unassigned_libraries", []) or [])
            if allocator._is_priority_for_36t(lib)
        ]
    )
    return {
        "consumed_libraries": len(consumed_ids),
        "consumed_gb": consumed_gb,
        "changed_lanes": len([lane_id for lane_id in changed_lane_ids if lane_id]),
        "remaining_priority_libraries": remaining_priority_libraries,
    }


def _merge_dedicated_imbalance_lanes_into_mode_1_1(
    solution: Any,
) -> Dict[str, int]:
    """将1.1阶段已形成的同组3.6T碱基不均专lane，优先尝试合并为1.1专lane。"""
    lanes = list(getattr(solution, "lane_assignments", []) or [])
    if not lanes:
        return {"merged_groups": 0, "removed_lanes": 0, "new_lanes": 0}

    validator = LaneValidator(strict_mode=True)
    grouped: Dict[Tuple[str, str], List[LaneAssignment]] = {}
    for lane in lanes:
        lane_id = str(getattr(lane, "lane_id", "") or "")
        if not lane_id.startswith("DL_"):
            continue
        if not _is_explicit_dedicated_imbalance_lane(lane):
            continue
        metadata = dict(getattr(lane, "metadata", {}) or {})
        selected_seq_mode = str(
            metadata.get("selected_seq_mode")
            or metadata.get("seq_mode")
            or metadata.get("lcxms")
            or ""
        )
        if selected_seq_mode == "1.1":
            continue
        dedicated_group = str(metadata.get("dedicated_group") or "")
        if not dedicated_group:
            group_ids, _ = _resolve_lane_imbalance_groups_and_ratio(
                [
                    lib for lib in list(getattr(lane, "libraries", []) or [])
                    if not _is_ai_balance_library(lib)
                ]
            )
            if len(group_ids) == 1:
                dedicated_group = next(iter(group_ids))
        if not dedicated_group:
            continue
        machine_type_text = _machine_type_to_text(getattr(lane, "machine_type", None), default="Nova X-25B")
        grouped.setdefault((machine_type_text, dedicated_group), []).append(lane)

    if not grouped:
        return {"merged_groups": 0, "removed_lanes": 0, "new_lanes": 0}

    remaining_lanes = list(lanes)
    merged_groups = 0
    removed_lanes = 0
    new_lanes = 0

    for (machine_type_text, dedicated_group), lane_group in grouped.items():
        if len(lane_group) < 2:
            continue
        machine_type = lane_group[0].machine_type or MachineType.NOVA_X_25B
        selected_libraries: List[EnhancedLibraryInfo] = []
        for lane in lane_group:
            for lib in list(getattr(lane, "libraries", []) or []):
                if _is_ai_balance_library(lib):
                    continue
                selected_libraries.append(lib)
        if not selected_libraries:
            continue
        real_sample_types = {
            _safe_str(
                getattr(lib, "sample_type_code", None)
                or getattr(lib, "sampletype", None)
                or getattr(lib, "wksampletype", None),
                default="",
            )
            for lib in selected_libraries
            if _safe_str(
                getattr(lib, "sample_type_code", None)
                or getattr(lib, "sampletype", None)
                or getattr(lib, "wksampletype", None),
                default="",
            )
        }
        if len(real_sample_types) > 1:
            logger.info(
                "DL同组合并到1.1跳过(文库类型不一致): 分组={}, lane数={}, types={}".format(
                    dedicated_group,
                    len(lane_group),
                    sorted(real_sample_types),
                )
            )
            continue

        lane_metadata = {
            "selected_seq_mode": "1.1",
            "seq_mode": "1.1",
            "lcxms": "1.1",
            "dispatch_stage": "post_priority_fill_dedicated_imbalance_merge",
            "is_dedicated_imbalance_lane": True,
            "dedicated_group": dedicated_group,
        }
        total_gb = _total_lane_data(selected_libraries)
        if _count_lane_index_pairs(selected_libraries) < AI_LANE_MIN_INDEX_PAIRS:
            logger.info(
                "DL同组合并到1.1跳过(Index对数不足): 分组={}, lane数={}, index_pairs={}".format(
                    dedicated_group,
                    len(lane_group),
                    _count_lane_index_pairs(selected_libraries),
                )
            )
            continue
        if _validate_index_conflicts_latest(selected_libraries):
            logger.info(
                "DL同组合并到1.1跳过(最新Index冲突): 分组={}, lane数={}".format(
                    dedicated_group,
                    len(lane_group),
                )
            )
            continue

        merged_lane = LaneAssignment(
            lane_id="DL_M11_TMP",
            machine_id="M_DL_M11_TMP",
            machine_type=machine_type,
            lane_capacity_gb=_lane_capacity_for_machine(machine_type),
        )
        merged_lane.metadata.update(lane_metadata)
        for lib in selected_libraries:
            lib._current_seq_mode_raw = "1.1"
            lib.selected_seq_mode = "1.1"
            lib.current_seq_mode = "1.1"
            lib.lcxms = "1.1"
            merged_lane.add_library(lib)
        required_balance = _resolve_lane_balance_data_gb(merged_lane)
        if required_balance > 0:
            merged_lane.metadata["wkbalancedata"] = round(required_balance, 3)
            merged_lane.metadata["required_balance_data_gb"] = round(required_balance, 3)
        min_allowed, max_allowed = _resolve_lane_capacity_limits(
            libraries=selected_libraries,
            machine_type=machine_type,
            lane_id=merged_lane.lane_id,
            lane_metadata=merged_lane.metadata,
        )
        effective_total_gb = total_gb + max(required_balance, 0.0)
        if effective_total_gb < min_allowed - 1e-6 or effective_total_gb > max_allowed + 1e-6:
            logger.info(
                "DL同组合并到1.1跳过(容量): 分组={}, lane数={}, 裸量={:.3f}G, 平衡={:.3f}G, 有效总量={:.3f}G, 区间=[{:.3f}, {:.3f}]G".format(
                    dedicated_group,
                    len(lane_group),
                    total_gb,
                    required_balance,
                    effective_total_gb,
                    min_allowed,
                    max_allowed,
                )
            )
            continue
        if required_balance <= 0:
            logger.info(
                "DL同组合并到1.1跳过(无平衡量): 分组={}, lane数={}".format(
                    dedicated_group,
                    len(lane_group),
                )
            )
            continue
        new_lane_id = f"DL_{machine_type.value}_{_reserve_auto_lane_serial('DL', machine_type):03d}"
        merged_lane.lane_id = new_lane_id
        merged_lane.machine_id = f"M_{new_lane_id}"
        if not _materialize_balance_library_for_lane(
            lane=merged_lane,
            all_lanes=lane_group,
            unassigned_pool=[],
            validator=validator,
        ):
            logger.info(
                "DL同组合并到1.1跳过(平衡文库物化失败): 分组={}, lane数={}, 需补平衡={:.3f}G".format(
                    dedicated_group,
                    len(lane_group),
                    required_balance,
                )
            )
            continue
        validation_result = _validate_lane_state(
            validator,
            merged_lane,
            list(getattr(merged_lane, "libraries", []) or []),
            balance_already_in_libs=True,
            skip_peak_size=True,
            skip_balance_injection_context_rules=True,
        )
        if not getattr(validation_result, "is_valid", False):
            logger.info(
                "DL同组合并到1.1跳过(终态校验失败): 分组={}, lane数={}, errors={}".format(
                    dedicated_group,
                    len(lane_group),
                    [
                        _safe_str(getattr(err, "message", None), default="")
                        for err in list(getattr(validation_result, "errors", []) or [])
                    ],
                )
            )
            continue

        lane_group_ids = {id(lane) for lane in lane_group}
        remaining_lanes = [lane for lane in remaining_lanes if id(lane) not in lane_group_ids]
        remaining_lanes.append(merged_lane)
        merged_groups += 1
        removed_lanes += len(lane_group)
        new_lanes += 1
        logger.info(
            "DL同组合并到1.1成功: 分组={}, 来源={}, 新lane={}, 总量={:.3f}G".format(
                dedicated_group,
                [lane.lane_id for lane in lane_group],
                merged_lane.lane_id,
                float(getattr(merged_lane, "total_data_gb", 0.0) or 0.0),
            )
        )

    if merged_groups > 0:
        solution.lane_assignments = remaining_lanes
    return {
        "merged_groups": merged_groups,
        "removed_lanes": removed_lanes,
        "new_lanes": new_lanes,
    }


def _should_skip_expensive_rescue_stage(
    solution: Any,
    *,
    stage_name: str,
    zero_lane_lib_threshold: int = ZERO_LANE_RESCUE_SKIP_LIB_THRESHOLD,
    pool_lib_threshold: int = LARGE_POOL_RESCUE_SKIP_LIB_THRESHOLD,
    pool_data_threshold_gb: float = LARGE_POOL_RESCUE_SKIP_DATA_GB,
) -> bool:
    """在低收益的大池场景下跳过高成本救援流程。"""
    unassigned = list(getattr(solution, "unassigned_libraries", []) or [])
    if not unassigned:
        return False

    unassigned_count = len(unassigned)
    unassigned_data = _total_lane_data(unassigned)
    lane_count = len(getattr(solution, "lane_assignments", []) or [])

    has_mode_1_1_lane = False
    for lane in list(getattr(solution, "lane_assignments", []) or []):
        metadata = dict(getattr(lane, "metadata", {}) or {})
        lane_mode = _normalize_mode_1_1_alias(
            metadata.get("seq_mode")
            or metadata.get("sequencing_mode")
            or metadata.get("lcxms")
            or ""
        )
        rule_code = str(metadata.get("rule_code", "") or "")
        if lane_mode == "1.1" or "mode_1_1" in rule_code:
            has_mode_1_1_lane = True
            break

    if (
        has_mode_1_1_lane
        and (
            unassigned_count >= MODE_1_1_POST_RESCUE_SKIP_LIB_THRESHOLD
            or unassigned_data >= MODE_1_1_POST_RESCUE_SKIP_DATA_GB
        )
    ):
        logger.info(
            "{}跳过: 已有1.1 Lane成功落地，剩余未分配池={}个/{:.1f}G，跳过高成本救援以保证主流程输出".format(
                stage_name,
                unassigned_count,
                unassigned_data,
            )
        )
        return True

    if lane_count == 0 and unassigned_count >= zero_lane_lib_threshold:
        logger.info(
            "{}跳过: 当前0条Lane通过验证，未分配池={}个/{:.1f}G，继续做高成本救援收益过低".format(
                stage_name,
                unassigned_count,
                unassigned_data,
            )
        )
        return True

    if (
        unassigned_count >= pool_lib_threshold
        or unassigned_data >= pool_data_threshold_gb
    ):
        logger.info(
            "{}跳过: 未分配池过大={}个/{:.1f}G，优先保证主排机时效".format(
                stage_name,
                unassigned_count,
                unassigned_data,
            )
        )
        return True

    return False


def _validate_lane_57_mix_rules(
    libraries: List[EnhancedLibraryInfo],
    enforce_total_limit: bool = False,
    lane_id: str = "",
    lane_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """按 lane 上下文校验碱基不均衡组合规则。

    规则边界：
    - 仅对碱基不均组合中的 56/57 混排组合生效
    - 其他场景均不因57规则做约束
    - 包 lane 走自身规则，不在这里叠加通用组合校验
    """
    if not libraries:
        return True, ""

    if _is_package_lane_context(libraries, lane_metadata=lane_metadata):
        return True, "package lane skips generic imbalance mix rules"

    imbalance_flags = [
        _is_imbalance_library_candidate(lib)
        for lib in libraries
    ]
    has_imbalance = any(imbalance_flags)
    if not has_imbalance:
        return True, "no imbalance libraries"

    has_balanced = any(not flag for flag in imbalance_flags)
    if not has_balanced:
        return True, "57 rule only applies to 56/57 imbalance mixed lanes"

    cache_key = (
        tuple(sorted(_get_library_identity_key(lib) for lib in libraries)),
        bool(enforce_total_limit),
        "mixed_56_57",
    )
    cached = _LANE_57_MIX_RULE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    result = _BASE_IMBALANCE_HANDLER.check_mix_compatibility(
        libraries,
        enforce_total_limit=enforce_total_limit,
    )
    _LANE_57_MIX_RULE_CACHE[cache_key] = result
    return result


def _count_library_index_pairs(lib: EnhancedLibraryInfo) -> int:
    """统计单个文库的Index对数。"""
    index_seq = str(getattr(lib, "index_seq", "") or "").strip()
    if not index_seq:
        return 0
    return len([item for item in index_seq.split(",") if str(item).strip()])


def _count_lane_index_pairs(libraries: List[EnhancedLibraryInfo]) -> int:
    """统计整条Lane的真实文库Index对总数，不把AI平衡文库计入下限。"""
    return sum(
        _count_library_index_pairs(lib)
        for lib in libraries
        if not _is_ai_balance_library(lib)
    )


def _get_package_lane_number_from_library(lib: EnhancedLibraryInfo) -> str:
    """提取单个文库的包Lane编号。"""
    return _safe_str(
        getattr(lib, "package_lane_number", None)
        or getattr(lib, "baleno", None)
        or getattr(lib, "wkbaleno", None),
        default="",
    )


def _get_non_balance_libraries(libraries: List[EnhancedLibraryInfo]) -> List[EnhancedLibraryInfo]:
    """过滤掉AI平衡文库，返回真实合同文库。"""
    return [lib for lib in list(libraries or []) if not _is_ai_balance_library(lib)]


def _is_all_non_balance_libraries_package_numbered(libraries: List[EnhancedLibraryInfo]) -> bool:
    """判断Lane内除平衡文库外是否全部带包Lane编号。"""
    non_balance_libraries = _get_non_balance_libraries(libraries)
    if not non_balance_libraries:
        return False
    return all(_get_package_lane_number_from_library(lib) for lib in non_balance_libraries)


def _is_10_plus_24_library(lib: EnhancedLibraryInfo) -> bool:
    """判断单个文库是否为10+24测序策略文库。"""
    values = [
        getattr(lib, "seq_scheme", None),
        getattr(lib, "wkseqscheme", None),
        getattr(lib, "seq_notes", None),
        getattr(lib, "wkseqnotes", None),
        getattr(lib, "current_seq_mode", None),
        getattr(lib, "_current_seq_mode_raw", None),
    ]
    return any("10+24" in _safe_str(value, default="") for value in values)


def _is_all_non_balance_libraries_10_plus_24(libraries: List[EnhancedLibraryInfo]) -> bool:
    """判断Lane内除平衡文库外是否全部为10+24文库。"""
    non_balance_libraries = _get_non_balance_libraries(libraries)
    if not non_balance_libraries:
        return False
    return all(_is_10_plus_24_library(lib) for lib in non_balance_libraries)


def _get_package_lane_number_from_lane(lane: LaneAssignment) -> str:
    """提取真正包Lane对应的包Lane编号。"""
    package_id = _safe_str(getattr(lane, "metadata", {}).get("package_id", ""), default="")
    if package_id:
        return package_id
    libraries = list(getattr(lane, "libraries", []) or [])
    if not _is_all_non_balance_libraries_package_numbered(libraries):
        return ""
    for lib in _get_non_balance_libraries(libraries):
        package_lane_number = _get_package_lane_number_from_library(lib)
        if package_lane_number:
            return package_lane_number
    return ""


def _is_package_lane_assignment(lane: LaneAssignment) -> bool:
    """判断当前Lane是否为真正包Lane。"""
    metadata = getattr(lane, "metadata", {}) or {}
    if isinstance(metadata, dict) and bool(metadata.get("is_package_lane")):
        return True
    if bool(_get_package_lane_number_from_lane(lane)):
        return True
    libraries = list(getattr(lane, "libraries", []) or [])
    non_balance_libraries = _get_non_balance_libraries(libraries)
    package_numbered_libraries = [
        lib for lib in non_balance_libraries if _get_package_lane_number_from_library(lib)
    ]
    if package_numbered_libraries and len(package_numbered_libraries) == len(non_balance_libraries):
        return True
    lane_id = _safe_str(getattr(lane, "lane_id", None), default="")
    return lane_id.startswith("LANE_") and bool(package_numbered_libraries)


def _is_3_6t_new_lane_context(
    lane: LaneAssignment,
    libraries: List[EnhancedLibraryInfo],
) -> bool:
    """判断Lane是否属于3.6T-NEW排机上下文。"""
    metadata = dict(getattr(lane, "metadata", {}) or {})
    metadata_values = [
        metadata.get("selected_seq_mode"),
        metadata.get("seq_mode"),
        metadata.get("sequencing_mode"),
        metadata.get("lcxms"),
        metadata.get("resolved_seq_mode"),
    ]
    for value in metadata_values:
        if _normalize_text_for_match(value) == "3.6T-NEW":
            return True

    for lib in libraries:
        lib_values = [
            getattr(lib, "_current_seq_mode_raw", None),
            getattr(lib, "current_seq_mode", None),
            getattr(lib, "lcxms", None),
        ]
        if any(_normalize_text_for_match(value) == "3.6T-NEW" for value in lib_values):
            return True
    return False


def _is_mode_1_1_lane_context(
    lane: LaneAssignment,
    libraries: List[EnhancedLibraryInfo],
) -> bool:
    """判断Lane是否属于1.1排机上下文。"""
    if _is_package_lane_assignment(lane):
        return False

    metadata = dict(getattr(lane, "metadata", {}) or {})
    metadata_values = [
        metadata.get("selected_seq_mode"),
        metadata.get("seq_mode"),
        metadata.get("sequencing_mode"),
        metadata.get("lcxms"),
        metadata.get("resolved_seq_mode"),
    ]
    for value in metadata_values:
        if _normalize_mode_1_1_alias(value) == "1.1":
            return True

    lane_id = _safe_str(getattr(lane, "lane_id", None), default="")
    if lane_id.startswith(("GL_", "MG_", "RM_")):
        for lib in libraries:
            if _normalize_mode_1_1_alias(getattr(lib, "_current_seq_mode_raw", None)) == "1.1":
                return True

    for lib in libraries:
        lib_values = [
            getattr(lib, "_current_seq_mode_raw", None),
            getattr(lib, "selected_seq_mode", None),
            getattr(lib, "current_seq_mode", None),
            getattr(lib, "lcxms", None),
        ]
        if any(_normalize_mode_1_1_alias(value) == "1.1" for value in lib_values):
            return True
    return False


def _validate_ai_lane_index_pair_rules(
    lane: LaneAssignment,
    libraries: Optional[List[EnhancedLibraryInfo]] = None,
) -> List[str]:
    """校验所有非包AI Lane的Index对数下限。"""
    if _is_package_lane_assignment(lane):
        return []

    libraries = libraries if libraries is not None else (getattr(lane, "libraries", []) or [])
    if not libraries:
        return []

    total_index_pairs = _count_lane_index_pairs(libraries)
    if total_index_pairs >= AI_LANE_MIN_INDEX_PAIRS:
        return []

    return [
        "AI Lane {} Index对数不足: 当前{}对, 要求>={}对".format(
            lane.lane_id,
            total_index_pairs,
            AI_LANE_MIN_INDEX_PAIRS,
        )
    ]


def _validate_package_lane_rules(
    lane: LaneAssignment,
    libraries: Optional[List[EnhancedLibraryInfo]] = None,
) -> List[str]:
    """仅对带包Lane编号的Lane执行专项规则校验。"""
    package_lane_number = _get_package_lane_number_from_lane(lane)
    if not package_lane_number:
        return []

    libraries = libraries if libraries is not None else (getattr(lane, "libraries", []) or [])
    total_contract_data = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in libraries)
    explicit_balance_data = 0.0
    if any(_is_ai_balance_library(lib) for lib in libraries):
        explicit_balance_data = 0.0
    else:
        explicit_balance_data = _get_explicit_balance_data_from_context(
            libraries,
            lane_metadata=getattr(lane, "metadata", None),
        )
    effective_total_data = total_contract_data + explicit_balance_data
    total_index_pairs = _count_lane_index_pairs(libraries)
    conflicts = _validate_index_conflicts_latest(libraries)

    errors: List[str] = []
    if total_index_pairs < PACKAGE_LANE_MIN_INDEX_PAIRS:
        errors.append(
            f"包Lane {package_lane_number} Index对数不足: 当前{total_index_pairs}对, 要求>={PACKAGE_LANE_MIN_INDEX_PAIRS}对"
        )
    if conflicts:
        preview = [
            f"{conflict.record_id_1} vs {conflict.record_id_2}"
            for conflict in conflicts[:3]
        ]
        errors.append(
            f"包Lane {package_lane_number} 存在Index重复: {', '.join(preview)}"
        )
    if effective_total_data < PACKAGE_LANE_MIN_GB or effective_total_data > PACKAGE_LANE_MAX_GB:
        errors.append(
            "包Lane {} 有效数据量不满足1000G±0.01G: 当前{:.3f}G(合同{:.3f}G+平衡{:.3f}G)".format(
                package_lane_number,
                effective_total_data,
                total_contract_data,
                explicit_balance_data,
            )
        )
    return errors


def _cleanup_stale_package_lanes_before_validation(solution: Any) -> Dict[str, int]:
    """移除拆分回滚后只剩平衡文库/无真实包Lane文库的残留包Lane。"""
    lanes = list(getattr(solution, "lane_assignments", []) or [])
    if not lanes:
        return {"removed_lanes": 0, "recovered_libraries": 0}

    kept_lanes: List[LaneAssignment] = []
    recovered_libraries: List[EnhancedLibraryInfo] = []
    removed_lane_ids: List[str] = []

    for lane in lanes:
        package_lane_number = _get_package_lane_number_from_lane(lane)
        if not package_lane_number:
            kept_lanes.append(lane)
            continue

        libraries = list(getattr(lane, "libraries", []) or [])
        non_balance_libraries = _get_non_balance_libraries(libraries)
        package_numbered_libraries = [
            lib for lib in non_balance_libraries if _get_package_lane_number_from_library(lib)
        ]
        if package_numbered_libraries:
            kept_lanes.append(lane)
            continue

        removed_lane_ids.append(_safe_str(getattr(lane, "lane_id", ""), default=""))
        recovered_libraries.extend(non_balance_libraries)

    if not removed_lane_ids:
        return {"removed_lanes": 0, "recovered_libraries": 0}

    existing_unassigned_ids = {
        id(lib) for lib in list(getattr(solution, "unassigned_libraries", []) or [])
    }
    for lib in recovered_libraries:
        if id(lib) not in existing_unassigned_ids:
            solution.unassigned_libraries.append(lib)
            existing_unassigned_ids.add(id(lib))
    solution.lane_assignments = kept_lanes
    logger.warning(
        "包Lane残留清理: 移除{}条无真实包Lane文库的残留Lane，回收{}个非平衡文库: {}".format(
            len(removed_lane_ids),
            len(recovered_libraries),
            ", ".join(lane_id or "<empty>" for lane_id in removed_lane_ids[:10]),
        )
    )
    return {"removed_lanes": len(removed_lane_ids), "recovered_libraries": len(recovered_libraries)}


def _is_lane_seq_10_plus_24_lane_assignment(lane: LaneAssignment) -> bool:
    """判断是否为10+24 Lane seq专项Lane。"""
    metadata = getattr(lane, "metadata", None) or {}
    if bool(metadata.get("is_lane_seq_10_plus_24_lane")):
        return _is_all_non_balance_libraries_10_plus_24(list(getattr(lane, "libraries", []) or []))
    if str(getattr(lane, "lane_id", "") or "").startswith(f"{LANE_SEQ_10_PLUS_24_LANE_PREFIX}_"):
        return _is_all_non_balance_libraries_10_plus_24(list(getattr(lane, "libraries", []) or []))
    return False


def _validate_lane_seq_10_plus_24_rules(
    lane: LaneAssignment,
    libraries: Optional[List[EnhancedLibraryInfo]] = None,
) -> List[str]:
    """10+24 Lane seq只校验Index对数下限与Index不重复。"""
    libraries = libraries if libraries is not None else (getattr(lane, "libraries", []) or [])
    total_index_pairs = _count_lane_index_pairs(libraries)
    conflicts = _validate_index_conflicts_latest(libraries)
    total_contract_data = sum(
        float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
        for lib in libraries
    )
    metadata = getattr(lane, "metadata", None) or {}
    if not any(_is_ai_balance_library(lib) for lib in libraries):
        total_contract_data += _safe_float(
            metadata.get("wkbalancedata")
            or metadata.get("wkadd_balance_data")
            or metadata.get("required_balance_data_gb"),
            default=0.0,
        )
    min_total = LANE_SEQ_10_PLUS_24_TARGET_TOTAL_GB - LANE_SEQ_10_PLUS_24_TOLERANCE_GB
    max_total = LANE_SEQ_10_PLUS_24_TARGET_TOTAL_GB + LANE_SEQ_10_PLUS_24_TOLERANCE_GB

    errors: List[str] = []
    if total_contract_data < min_total - 1e-6 or total_contract_data > max_total + 1e-6:
        errors.append(
            "10+24 Lane seq {} 有效数据量不满足容量规则: 当前{:.3f}G, 要求[{:.3f}, {:.3f}]G".format(
                lane.lane_id,
                total_contract_data,
                min_total,
                max_total,
            )
        )
    if total_index_pairs < AI_LANE_MIN_INDEX_PAIRS:
        errors.append(
            "10+24 Lane seq {} Index对数不足: 当前{}对, 要求>={}对".format(
                lane.lane_id,
                total_index_pairs,
                AI_LANE_MIN_INDEX_PAIRS,
            )
        )
    if conflicts:
        preview = [
            f"{conflict.record_id_1} vs {conflict.record_id_2}"
            for conflict in conflicts[:3]
        ]
        errors.append(
            "10+24 Lane seq {} 存在Index重复: {}".format(
                lane.lane_id,
                ", ".join(preview),
            )
        )
    return errors


def _validate_final_package_lanes(solution: Any) -> None:
    """排机完成后复核所有包Lane规则。"""
    _cleanup_stale_package_lanes_before_validation(solution)
    all_errors: List[str] = []
    for lane in getattr(solution, "lane_assignments", []) or []:
        all_errors.extend(_validate_package_lane_rules(lane))

    if all_errors:
        for error in all_errors:
            logger.error(error)
        raise ValueError("排后包Lane校验失败，请检查日志中的包Lane规则明细")

    logger.info("排后包Lane校验通过")


def _validate_no_split_for_package_lane_libraries(solution: Any) -> None:
    """复核带包Lane编号文库仅允许多包Lane编号特例拆分。"""
    errors: List[str] = []
    multi_split_lane_ids: Dict[str, Set[str]] = {}
    multi_split_expected_package_ids: Dict[str, Set[str]] = {}
    lane_purity_checked: Set[str] = set()

    for lane in getattr(solution, "lane_assignments", []) or []:
        lane_id = _safe_str(getattr(lane, "lane_id", ""), default="")
        lane_package_lane_number = _get_package_lane_number_from_lane(lane)
        for lib in getattr(lane, "libraries", []) or []:
            package_lane_number = _safe_str(
                getattr(lib, "package_lane_number", None) or getattr(lib, "baleno", None),
                default="",
            )
            if not package_lane_number:
                continue
            is_split = _is_split_library(lib) or int(getattr(lib, "total_fragments", 0) or 0) > 1
            if not is_split:
                continue

            is_allowed_multi_pkg_split = bool(getattr(lib, "_package_lane_multi_split", False))
            original_numbers = getattr(lib, "_package_lane_original_numbers", None) or ()
            expected_package_ids = {
                _safe_str(item, default="")
                for item in original_numbers
                if _safe_str(item, default="")
            }
            family_id = _safe_str(
                getattr(lib, "_package_lane_multi_split_family_id", None),
                default="",
            )

            if lane_package_lane_number != package_lane_number:
                errors.append(
                    "包Lane {} 文库 {} 拆分后进入了不匹配的Lane包号 {}".format(
                        package_lane_number,
                        _safe_str(getattr(lib, "origrec", ""), default="UNKNOWN"),
                        lane_package_lane_number or "EMPTY",
                    )
                )
                continue

            if lane_id and lane_id not in lane_purity_checked:
                lane_purity_checked.add(lane_id)
                non_package_lane_libs = [
                    lane_lib
                    for lane_lib in getattr(lane, "libraries", []) or []
                    if not _is_ai_balance_library(lane_lib)
                    if not _safe_str(
                        getattr(lane_lib, "package_lane_number", None) or getattr(lane_lib, "baleno", None),
                        default="",
                    )
                ]
                if non_package_lane_libs:
                    errors.append(
                        f"包Lane拆分目标Lane {lane_id} 混入了{len(non_package_lane_libs)}个非包Lane文库"
                    )

            if not is_allowed_multi_pkg_split:
                continue

            if not (
                family_id
                and len(expected_package_ids) > 1
                and package_lane_number in expected_package_ids
                and int(getattr(lib, "total_fragments", 0) or 0) == len(expected_package_ids)
            ):
                errors.append(
                    "多包Lane拆分文库 {} 的包号家族信息不完整".format(
                        _safe_str(getattr(lib, "origrec", ""), default="UNKNOWN"),
                    )
                )
                continue

            multi_split_lane_ids.setdefault(family_id, set()).add(str(lane.lane_id))
            multi_split_expected_package_ids.setdefault(family_id, set()).update(expected_package_ids)

    for family_id, expected_package_ids in multi_split_expected_package_ids.items():
        actual_lane_ids = multi_split_lane_ids.get(family_id, set())
        if len(actual_lane_ids) != len(expected_package_ids):
            errors.append(
                f"多包Lane拆分家族 {family_id} 未分配到足够多的不同Lane: "
                f"目标包Lane数={len(expected_package_ids)}, 实际Lane数={len(actual_lane_ids)}"
            )

    if errors:
        for error in errors:
            logger.error(error)
        raise ValueError("排后校验失败：包Lane编号文库拆分规则不满足")

    logger.info("排后校验通过：包Lane编号文库拆分规则满足约束")


def _is_index_conflict_only(result: Any) -> bool:
    """判断失败是否仅由Index冲突导致。"""
    errors = getattr(result, "errors", None) or []
    if not errors:
        return False
    return all(err.rule_type == ValidationRuleType.INDEX_CONFLICT for err in errors)


def _get_scattered_mix_priority_rank(lib: EnhancedLibraryInfo) -> int:
    """历史兼容接口：高优文库逻辑已停用，所有文库同一档。"""
    return 2


def _get_priority_rank_label(rank: int) -> str:
    """将优先级档位转换为便于日志阅读的文本。"""
    if rank == 0:
        return "P0(临检/SJ)"
    if rank == 1:
        return "P1(YC)"
    return "P2(其他)"


def _get_current_hard_priority_rank(
    libraries: List[EnhancedLibraryInfo],
) -> Optional[int]:
    """返回当前待排池中最高优先级档位（数值越小优先级越高）。"""
    if not libraries:
        return None
    return min(_get_scattered_mix_priority_rank(lib) for lib in libraries)


def _get_priority_gate_label(max_rank: int) -> str:
    """返回当前允许参与成Lane的优先级范围描述。"""
    if max_rank <= 0:
        return "P0(仅临检/SJ)"
    if max_rank == 1:
        return "P0/P1(临检/SJ+YC)"
    return "P0/P1/P2(全部)"


def _resolve_priority_gate_rank(
    libraries: List[EnhancedLibraryInfo],
    machine_type: MachineType | str,
    lane_id: str = "",
    lane_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """解析当前可放开的最高优先级档位。

    规则：
    1. 先尽量只使用 P0(临检/SJ) 文库成Lane
    2. 若 P0 总量不足以独立成Lane，再放开 P1(YC) 作为补位
    3. 若 P0+P1 仍不足，再放开 P2(其他)
    """
    if not libraries:
        return None

    min_allowed, _ = _resolve_lane_capacity_limits(
        libraries=libraries,
        machine_type=machine_type,
        lane_id=lane_id,
        lane_metadata=lane_metadata,
    )
    rank_totals: Dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0}
    highest_present_rank = 0
    for lib in libraries:
        rank = _get_scattered_mix_priority_rank(lib)
        rank_totals[rank] += float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
        highest_present_rank = max(highest_present_rank, rank)

    cumulative = 0.0
    for rank in range(highest_present_rank + 1):
        cumulative += rank_totals.get(rank, 0.0)
        if cumulative >= min_allowed:
            return rank
    return highest_present_rank


def _filter_libraries_by_hard_priority(
    libraries: List[EnhancedLibraryInfo],
    machine_type: MachineType | str,
    *,
    lane_id: str = "",
    lane_metadata: Optional[Dict[str, Any]] = None,
    stage_name: str = "",
    emit_log: bool = False,
) -> List[EnhancedLibraryInfo]:
    """历史兼容接口：高优门禁已停用，原样返回候选池。"""
    return list(libraries or [])


def _filter_priority_across_pools(
    primary_pool: List[EnhancedLibraryInfo],
    secondary_pool: List[EnhancedLibraryInfo],
    machine_type: MachineType | str,
    *,
    lane_id: str = "",
    lane_metadata: Optional[Dict[str, Any]] = None,
    stage_name: str = "",
    emit_log: bool = False,
) -> Tuple[List[EnhancedLibraryInfo], List[EnhancedLibraryInfo], Optional[int]]:
    """历史兼容接口：高优门禁已停用，原样返回两个候选池。"""
    return list(primary_pool or []), list(secondary_pool or []), 2


def _can_rebuild_lane_from_priority_pool(
    *,
    candidate_pool: List[EnhancedLibraryInfo],
    validator: Any,
    machine_type: MachineType,
    lane_id: str,
    lane_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """判断当前高优先级池是否还能真正重组出一条合法Lane。

    仅当高优先级文库不仅“总量足够”，而且确实能通过打包与严格校验时，
    才允许在最终收口阶段继续回退较低优先级Lane。
    """
    if not candidate_pool:
        return False
    lane, _ = _attempt_build_lane_from_pool(
        pool=list(candidate_pool),
        validator=validator,
        machine_type=machine_type,
        lane_id_prefix=f"PGCHK_{lane_id}",
        lane_serial=1,
        index_conflict_attempts=3,
        other_failure_attempts=6,
        extra_metadata=dict(lane_metadata or {}),
        prioritize_scattered_mix=True,
    )
    return lane is not None


def _parse_scattered_mix_delete_date(lib: EnhancedLibraryInfo) -> Optional[float]:
    """解析散样混排的delete_date天数字段，数值越小表示越临近越优先。"""
    cache_key = (
        getattr(lib, "_delete_date_raw", None),
        getattr(lib, "deduction_time", None),
    )
    cached = getattr(lib, "_scattered_mix_delete_date_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    raw_value = getattr(lib, "_delete_date_raw", None)
    if raw_value in (None, ""):
        raw_value = getattr(lib, "deduction_time", None)
    if raw_value in (None, ""):
        setattr(lib, "_scattered_mix_delete_date_cache", (cache_key, None))
        return None
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        parsed = None
    setattr(lib, "_scattered_mix_delete_date_cache", (cache_key, parsed))
    return parsed


def _get_scattered_mix_delete_date_sort_value(lib: EnhancedLibraryInfo) -> float:
    """其他文库按delete_date排序，越临近越优先；缺失值排最后。"""
    if _get_scattered_mix_priority_rank(lib) < 2:
        return 0.0
    parsed = _parse_scattered_mix_delete_date(lib)
    if parsed is None:
        return float("inf")
    return parsed


def _sort_by_board_preference_for_scattered_mix(
    libraries: List[EnhancedLibraryInfo],
) -> List[EnhancedLibraryInfo]:
    """软约束：尽量让同板号文库在散样混排时优先聚拢。"""
    if not libraries:
        return libraries

    board_count: Dict[str, int] = {}
    for lib in libraries:
        board = getattr(lib, "board_number", "") or ""
        if board:
            board_count[board] = board_count.get(board, 0) + 1

    if not board_count:
        return libraries

    return sorted(
        libraries,
        key=lambda lib: (
            -board_count.get(getattr(lib, "board_number", "") or "", 0),
            getattr(lib, "board_number", "") or "",
            -lib.get_data_amount_gb(),
        ),
    )


def _sort_remaining_for_scattered_mix_lane(
    libraries: List[EnhancedLibraryInfo],
) -> List[EnhancedLibraryInfo]:
    """散样混排成Lane顺序：优先聚拢临检/SJ，其次YC，再考虑delete_date。"""
    if not libraries:
        return libraries

    cache_key = _build_library_compact_identity_signature(libraries, canonicalize=True)
    cached = _SCATTERED_MIX_SORT_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    board_sorted = _sort_by_board_preference_for_scattered_mix(libraries)
    board_order = {id(lib): idx for idx, lib in enumerate(board_sorted)}
    sorted_libraries = sorted(
        libraries,
        key=lambda lib: (
            _get_scattered_mix_priority_rank(lib),
            _get_scattered_mix_delete_date_sort_value(lib),
            board_order.get(id(lib), len(board_order)),
            -lib.get_data_amount_gb(),
        ),
    )
    if len(_SCATTERED_MIX_SORT_CACHE) >= 4096:
        _SCATTERED_MIX_SORT_CACHE.clear()
    _SCATTERED_MIX_SORT_CACHE[cache_key] = tuple(sorted_libraries)
    return sorted_libraries


def _sort_remaining_for_lane_seed(
    libraries: List[EnhancedLibraryInfo],
    seed_lib: EnhancedLibraryInfo,
    current_lane_libs: Optional[List[EnhancedLibraryInfo]] = None,
    machine_type: Optional[MachineType | str] = None,
    special_data_limit: Optional[float] = None,
) -> List[EnhancedLibraryInfo]:
    """单条Lane内优先吞同级高优先级文库，降低临检/SJ/YC被打散概率。"""
    if not libraries:
        return libraries

    seed_rank = _get_scattered_mix_priority_rank(seed_lib)
    base_sorted = _sort_remaining_for_scattered_mix_lane(libraries)
    base_order = {id(lib): idx for idx, lib in enumerate(base_sorted)}
    lane_context = list(current_lane_libs or [])
    lane_summary = _resolve_lane_imbalance_summary(lane_context) if lane_context else None
    return sorted(
        libraries,
        key=lambda lib: (
            0 if _get_scattered_mix_priority_rank(lib) == seed_rank else 1,
            _get_scattered_mix_priority_rank(lib),
            *_build_imbalance_fill_sort_key(
                lane_context,
                lib,
                machine_type=machine_type,
                special_data_limit=special_data_limit,
                lane_summary=lane_summary,
                candidate_is_imbalance=_is_imbalance_library_candidate(lib),
            ),
            _get_scattered_mix_delete_date_sort_value(lib),
            base_order.get(id(lib), len(base_order)),
            -lib.get_data_amount_gb(),
        ),
    )


def _build_scattered_mix_candidate_order(
    libraries: List[EnhancedLibraryInfo],
    current_lane_libs: Optional[List[EnhancedLibraryInfo]] = None,
    seed_offset: int = 0,
    machine_type: Optional[MachineType | str] = None,
    special_data_limit: Optional[float] = None,
) -> List[EnhancedLibraryInfo]:
    """为散样混排构建稳定候选顺序。"""
    if not libraries:
        return libraries

    ordered = _sort_remaining_for_scattered_mix_lane(libraries)
    lane_context = list(current_lane_libs or [])
    if lane_context:
        seed_lib = lane_context[0]
    else:
        seed_index = min(max(int(seed_offset), 0), len(ordered) - 1)
        seed_lib = ordered[seed_index]
    seen: Set[int] = set()
    prioritized: List[EnhancedLibraryInfo] = []
    for lib in _sort_remaining_for_lane_seed(
        ordered,
        seed_lib,
        current_lane_libs=lane_context,
        machine_type=machine_type,
        special_data_limit=special_data_limit,
    ):
        object_id = id(lib)
        if object_id in seen:
            continue
        seen.add(object_id)
        prioritized.append(lib)
    return prioritized


def _is_truthy_flag(value: Any) -> bool:
    """宽松识别业务布尔标记。"""
    normalized = _normalize_text_for_match(value)
    return normalized in {"Y", "YES", "TRUE", "1", "是", "需", "需要", "包LANE", "包FC"}


def _safe_library_text(lib: EnhancedLibraryInfo, *field_names: str) -> str:
    """按候选字段名顺序提取文库文本字段。"""
    for field_name in field_names:
        value = getattr(lib, field_name, None)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _is_manual_or_risk_library(lib: EnhancedLibraryInfo) -> bool:
    """识别风险建库、手工模板、客户自建库等高约束文库。"""
    combined_text = " ".join(
        _normalize_text_for_match(text)
        for text in [
            getattr(lib, "risk_build_flag", None),
            getattr(lib, "wkjkhj", None),
            getattr(lib, "remarks", None),
            getattr(lib, "machine_note", None),
        ]
        if text not in (None, "")
    )
    return any(keyword in combined_text for keyword in ["风险", "RISK", "手工", "模板", "客户自建", "自建库"])


def _is_customer_library_candidate(lib: EnhancedLibraryInfo) -> bool:
    """识别客户文库。"""
    customer_flag = _normalize_text_for_match(getattr(lib, "customer_library", None))
    lab_type = _normalize_text_for_match(getattr(lib, "lab_type", None))
    return _is_truthy_flag(getattr(lib, "customer_library", None)) or "客户" in customer_flag or "CUSTOMER" in lab_type


def _is_scattered_library_candidate(lib: EnhancedLibraryInfo) -> bool:
    """识别散样/混排优先处理文库。"""
    if _get_scattered_mix_priority_rank(lib) < 2:
        return True
    text = " ".join(
        _normalize_text_for_match(_safe_library_text(lib, field_name))
        for field_name in ["sub_project_name", "remarks", "machine_note", "add_test_note", "add_tests_remark"]
    )
    return any(keyword in text for keyword in ["散", "混排", "SCATTERED", "MIX"])


def _derive_rescue_rule_bucket_name(lib: EnhancedLibraryInfo) -> str:
    """为补Lane/救援阶段生成粗粒度规则桶名称。"""
    special_split = _normalize_text_for_match(
        getattr(lib, "special_splits", None) or getattr(lib, "wkspecialsplits", None)
    )
    special_group_a = {_normalize_text_for_match(value) for value in SPECIAL_SPLIT_GROUP_A}
    special_group_b = {_normalize_text_for_match(value) for value in SPECIAL_SPLIT_GROUP_B}
    if special_split in special_group_a:
        return "special_combo_a"
    if special_split in special_group_b:
        return "special_combo_b"
    if _is_truthy_flag(getattr(lib, "is_package_lane", None)):
        return "package_lane"
    try:
        if callable(getattr(lib, "is_10x_library", None)) and lib.is_10x_library():
            return "ten_x"
    except Exception:
        logger.exception("识别10X文库异常，回退到普通规则分桶")
    if _is_manual_or_risk_library(lib):
        return "manual_or_risk"
    if _is_customer_library_candidate(lib):
        return "customer_library"
    if _is_imbalance_library_candidate(lib):
        return "imbalance"
    if _is_scattered_library_candidate(lib):
        return "scattered"
    product_line = _normalize_text_for_match(_safe_library_text(lib, "wkproductline", "product_line"))
    if product_line in {"WGS", "RNA", "MRNA", "单细胞", "SINGLECELL"}:
        return f"product_{product_line}"
    return "general"


def _derive_rescue_pre_group_key(lib: EnhancedLibraryInfo) -> str:
    """构Lane前更强预分组，提升候选池纯度。"""
    task_group = _normalize_text_for_match(_safe_library_text(lib, "wktaskgroupname", "sub_project_name")) or "NA"
    product_line = _normalize_text_for_match(_safe_library_text(lib, "wkproductline", "product_line")) or "NA"
    sample_type = _normalize_text_for_match(_safe_library_text(lib, "wksampletype", "sample_type_code")) or "NA"
    split_type = _normalize_text_for_match(_safe_library_text(lib, "wkspecialsplits", "special_splits")) or "NA"
    is_ten_x = False
    try:
        is_ten_x = bool(callable(getattr(lib, "is_10x_library", None)) and lib.is_10x_library())
    except Exception:
        is_ten_x = False
    flags = [
        "TENX" if is_ten_x else "",
        "PACKAGE" if _is_truthy_flag(getattr(lib, "is_package_lane", None)) else "",
        "SCATTERED" if _is_scattered_library_candidate(lib) else "",
        "CUSTOMER" if _is_customer_library_candidate(lib) else "",
        "MANUAL_RISK" if _is_manual_or_risk_library(lib) else "",
        "IMBALANCE" if _is_imbalance_library_candidate(lib) else "",
    ]
    compact_flags = "+".join(flag for flag in flags if flag) or "NORMAL"
    return "|".join([task_group, product_line, sample_type, split_type, compact_flags])


def _group_rescue_bucket_candidates(bucket_name: str, libraries: List[EnhancedLibraryInfo]) -> List[Tuple[str, List[EnhancedLibraryInfo]]]:
    """按细粒度规则预分组，并将大组优先返回。"""
    grouped: Dict[str, List[EnhancedLibraryInfo]] = {}
    for lib in libraries:
        group_key = _derive_rescue_pre_group_key(lib)
        grouped.setdefault(group_key, []).append(lib)
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (-_total_lane_data(item[1]), -len(item[1]), item[0]),
    )
    return [(f"{bucket_name}:{group_key}", libs) for group_key, libs in ordered_groups]


def _build_rescue_rule_buckets(pool: List[EnhancedLibraryInfo], lane_id_prefix: str) -> List[Tuple[str, List[EnhancedLibraryInfo]]]:
    """构建救援阶段规则分桶，缩小无效搜索空间。"""
    if not pool:
        return []
    cache_key = (lane_id_prefix, _build_library_identity_signature(pool, canonicalize=True))
    cached = _RESCUE_RULE_BUCKET_CACHE.get(cache_key)
    if cached is not None:
        return [(bucket_name, list(bucket_libraries)) for bucket_name, bucket_libraries in cached]

    rough_buckets: Dict[str, List[EnhancedLibraryInfo]] = {}
    for lib in pool:
        rough_buckets.setdefault(_derive_rescue_rule_bucket_name(lib), []).append(lib)

    ordered_bucket_names = sorted(
        rough_buckets.keys(),
        key=lambda name: (
            0 if name.startswith("special_combo") else 1,
            0 if name in {"package_lane", "ten_x", "imbalance", "manual_or_risk"} else 1,
            -_total_lane_data(rough_buckets[name]),
            -len(rough_buckets[name]),
            name,
        ),
    )
    refined_buckets: List[Tuple[str, List[EnhancedLibraryInfo]]] = []
    for bucket_name in ordered_bucket_names:
        refined_buckets.extend(_group_rescue_bucket_candidates(bucket_name, rough_buckets[bucket_name]))

    if len(_RESCUE_RULE_BUCKET_CACHE) >= 512:
        _RESCUE_RULE_BUCKET_CACHE.clear()
    _RESCUE_RULE_BUCKET_CACHE[cache_key] = [(bucket_name, list(bucket_libraries)) for bucket_name, bucket_libraries in refined_buckets]
    return refined_buckets


def _quick_check_pool_feasibility(
    *,
    pool: List[EnhancedLibraryInfo],
    machine_type: MachineType,
    lane_metadata: Optional[Dict[str, Any]] = None,
    stage_label: str,
) -> Tuple[bool, str]:
    """救援前轻校验：明显不可能成Lane的候选池直接跳过。"""
    if not pool:
        return False, "空候选池"

    metadata = _build_lane_metadata_for_validator(f"{stage_label}_TMP", lane_metadata, libraries=pool)
    cache_key = (
        machine_type.value if isinstance(machine_type, MachineType) else str(machine_type),
        tuple(sorted((str(key), repr(value)) for key, value in metadata.items())),
        _build_library_identity_signature(pool, canonicalize=True),
    )
    cached = _LIGHT_POOL_FEASIBILITY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    min_allowed, max_allowed = _resolve_lane_capacity_limits(
        libraries=pool,
        machine_type=machine_type,
        lane_metadata=lane_metadata,
    )
    total_data = _total_lane_data(pool)
    if total_data + 1e-6 < min_allowed:
        result = (False, f"总量{total_data:.1f}G低于最小门槛{min_allowed:.1f}G")
    else:
        has_package = any(_is_truthy_flag(getattr(lib, "is_package_lane", None)) for lib in pool)
        has_non_package = any(not _is_truthy_flag(getattr(lib, "is_package_lane", None)) for lib in pool)
        normalized_special_splits = {
            _normalize_text_for_match(getattr(lib, "special_splits", None) or getattr(lib, "wkspecialsplits", None))
            for lib in pool
        }
        special_group_a = {_normalize_text_for_match(value) for value in SPECIAL_SPLIT_GROUP_A}
        special_group_b = {_normalize_text_for_match(value) for value in SPECIAL_SPLIT_GROUP_B}
        has_group_a = any(split_value in special_group_a for split_value in normalized_special_splits)
        has_group_b = any(split_value in special_group_b for split_value in normalized_special_splits)
        if has_package and has_non_package:
            result = (False, "包Lane文库与普通文库混入同池")
        elif has_group_a and has_group_b:
            result = (False, "10X special combo A/B混池")
        elif total_data > max_allowed * 6 and len(pool) > 256:
            result = (True, f"超大池{len(pool)}个/{total_data:.1f}G，允许进入分桶后再细化")
        else:
            result = (True, f"通过轻校验: pool={len(pool)}个/{total_data:.1f}G")

    if len(_LIGHT_POOL_FEASIBILITY_CACHE) >= 4096:
        _LIGHT_POOL_FEASIBILITY_CACHE.clear()
    _LIGHT_POOL_FEASIBILITY_CACHE[cache_key] = result
    return result


def _quick_check_lane_candidate(
    *,
    libraries: List[EnhancedLibraryInfo],
    machine_type: MachineType,
    lane_metadata: Optional[Dict[str, Any]] = None,
    stage_label: str,
) -> Tuple[bool, str]:
    """候选Lane轻校验：在重校验前统一过滤明显失败组合。"""
    feasible, reason = _quick_check_pool_feasibility(
        pool=libraries,
        machine_type=machine_type,
        lane_metadata=lane_metadata,
        stage_label=stage_label,
    )
    if not feasible:
        return feasible, reason

    cache_key = (
        machine_type.value if isinstance(machine_type, MachineType) else str(machine_type),
        tuple(sorted((str(key), repr(value)) for key, value in _build_lane_metadata_for_validator(stage_label, lane_metadata, libraries=libraries).items())),
        _build_library_identity_signature(libraries, canonicalize=True),
    )
    cached = _LIGHT_POOL_FEASIBILITY_CACHE.get(cache_key)
    if cached is not None and cached[1].startswith("candidate:"):
        return cached[0], cached[1][10:]

    ss_valid, _, ss_reason = _validate_lane_special_split_rule(libraries)
    if not ss_valid:
        result = (False, f"candidate:{ss_reason}")
    else:
        imbalance_mix_valid, imbalance_reason = _validate_lane_57_mix_rules(
            libraries,
            enforce_total_limit=False,
            lane_metadata=lane_metadata,
        )
        if not imbalance_mix_valid:
            result = (False, f"candidate:{imbalance_reason}")
        else:
            result = (True, "candidate:通过候选轻校验")

    if len(_LIGHT_POOL_FEASIBILITY_CACHE) >= 4096:
        _LIGHT_POOL_FEASIBILITY_CACHE.clear()
    _LIGHT_POOL_FEASIBILITY_CACHE[cache_key] = result
    return result[0], result[1][10:]


def _attempt_build_lane_from_pool(
    pool: List[EnhancedLibraryInfo],
    validator,
    machine_type: MachineType,
    lane_id_prefix: str,
    lane_serial: Optional[int] = None,
    index_conflict_attempts: int = DEFAULT_INDEX_CONFLICT_ATTEMPTS,
    other_failure_attempts: int = DEFAULT_OTHER_FAILURE_ATTEMPTS,
    extra_metadata: Optional[Dict[str, Any]] = None,
    prioritize_scattered_mix: bool = False,
    deterministic_candidate_order: Optional[str] = None,
    lane_validation_cache: Optional[
        Dict[Tuple[str, Tuple[Tuple[str, str], ...], Tuple[str, ...]], Any]
    ] = None,
) -> Tuple[LaneAssignment | None, List[EnhancedLibraryInfo]]:
    """尝试从未分配池构建新Lane

    打包阶段同时检查容量、special_split和Index冲突，
    大幅降低后续全量验证失败概率。

    Args:
        extra_metadata: 额外元数据（如is_pure_non_10bp_lane等），
                        在验证前注入Lane.metadata，使验证器正确识别Lane类型。
    """
    if not pool:
        return None, []
    if index_conflict_attempts <= 0 or other_failure_attempts <= 0:
        return None, []
    active_pool = _filter_libraries_by_hard_priority(
        list(pool),
        machine_type=machine_type,
        lane_metadata=extra_metadata,
        stage_name=f"{lane_id_prefix}_build",
        emit_log=True,
    )
    if not active_pool:
        return None, []

    # 复用模块级单例，不重复初始化（每次 new 会打印 INFO 日志，高频调用有明显开销）
    _idx_validator = _MODULE_IDX_VALIDATOR
    allocated_lane_serial = (
        int(lane_serial)
        if lane_serial is not None
        else _reserve_auto_lane_serial(lane_id_prefix, machine_type)
    )

    index_conflict_retry_count = 0
    other_failure_retry_count = 0
    attempt_idx = 0
    seen_selected_signatures: Set[Tuple[str, ...]] = set()
    cached_lane_validations = lane_validation_cache if lane_validation_cache is not None else {}
    special_data_limit = _resolve_special_library_data_limit(machine_type)
    max_scattered_mix_attempts = (
        RM_SCATTERED_MIX_VARIANT_ATTEMPTS
        if prioritize_scattered_mix and lane_id_prefix == "RM"
        else SCATTERED_MIX_VARIANT_ATTEMPTS
    )
    while (
        index_conflict_retry_count < index_conflict_attempts
        and other_failure_retry_count < other_failure_attempts
    ):
        attempt_idx += 1
        if prioritize_scattered_mix and attempt_idx > max_scattered_mix_attempts:
            break
        selected: List[EnhancedLibraryInfo] = []
        selected_ids: Set[int] = set()
        # 与 selected 平行维护的预解析索引缓存，避免在 validate_new_lib_quick_with_cache
        # 内部对同一文库反复调用 _parse_library_indices（逐个候选检查时累计百万次调用）
        selected_idx_cache: List = []
        total = 0.0
        random_target: Optional[float] = None
        if prioritize_scattered_mix:
            while True:
                selected_min_allowed = 0.0
                if selected:
                    selected_min_allowed, _ = _resolve_lane_capacity_limits(
                        libraries=selected,
                        machine_type=machine_type,
                        lane_id=f"{lane_id_prefix}_TMP",
                        lane_metadata=extra_metadata,
                    )
                remaining_candidates = _filter_libraries_excluding_object_ids(
                    active_pool,
                    selected_ids,
                )
                if not remaining_candidates:
                    break
                selected_lane_summary = _resolve_lane_imbalance_summary(selected) if selected else None
                lane_context_for_sort = selected if selected else None
                candidates = _build_scattered_mix_candidate_order(
                    remaining_candidates,
                    current_lane_libs=lane_context_for_sort,
                    seed_offset=attempt_idx - 1,
                    machine_type=machine_type,
                    special_data_limit=special_data_limit,
                )
                added_candidate = False
                for lib in candidates:
                    if _shares_split_family_with_selected(selected, lib):
                        continue
                    candidate_is_imbalance = _is_imbalance_library_candidate(lib)
                    projected_special_data = _project_lane_special_data(
                        selected,
                        lib,
                        lane_summary=selected_lane_summary,
                        candidate_is_imbalance=candidate_is_imbalance,
                    )
                    if projected_special_data > special_data_limit + SPECIAL_LIBRARY_LIMIT_EPSILON:
                        continue
                    if (
                        selected
                        and total >= selected_min_allowed
                        and not _candidate_improves_imbalance_target(
                            selected,
                            lib,
                            machine_type=machine_type,
                            special_data_limit=special_data_limit,
                            lane_summary=selected_lane_summary,
                            candidate_is_imbalance=candidate_is_imbalance,
                        )
                    ):
                        continue
                    data = lib.get_data_amount_gb()
                    trial_libs = selected + [lib]
                    trial_min_allowed, trial_max_allowed = _resolve_lane_capacity_limits(
                        libraries=trial_libs,
                        machine_type=machine_type,
                        lane_id=f"{lane_id_prefix}_TMP",
                        lane_metadata=extra_metadata,
                    )
                    if total + data > trial_max_allowed:
                        continue
                    candidate_light_valid, _ = _quick_check_lane_candidate(
                        libraries=trial_libs,
                        machine_type=machine_type,
                        lane_metadata=extra_metadata,
                        stage_label=f"{lane_id_prefix}_candidate",
                    )
                    if not candidate_light_valid:
                        continue
                    idx_valid, lib_indices = _validate_new_lib_quick_with_result_cache(
                        idx_validator=_idx_validator,
                        selected_libraries=selected,
                        selected_indices_cache=selected_idx_cache,
                        new_lib=lib,
                    )
                    if not idx_valid:
                        continue
                    selected.append(lib)
                    selected_ids.add(id(lib))
                    selected_idx_cache.append(lib_indices)
                    total += data
                    added_candidate = True
                    break
                if not added_candidate:
                    break
        else:
            if deterministic_candidate_order == "high_constraint":
                candidates = _sort_remaining_for_high_constraint_lane(
                    list(active_pool),
                    current_lane_libs=selected if selected else None,
                )
            else:
                candidates = list(active_pool)
                random.shuffle(candidates)
            for lib in candidates:
                if _shares_split_family_with_selected(selected, lib):
                    continue
                data = lib.get_data_amount_gb()
                trial_libs = selected + [lib]
                trial_min_allowed, trial_max_allowed = _resolve_lane_capacity_limits(
                    libraries=trial_libs,
                    machine_type=machine_type,
                    lane_id=f"{lane_id_prefix}_TMP",
                    lane_metadata=extra_metadata,
                )
                if total + data > trial_max_allowed:
                    continue
                ss_valid, _, _ = _validate_lane_special_split_rule(trial_libs)
                if not ss_valid:
                    continue
                imbalance_mix_valid, _ = _validate_lane_57_mix_rules(
                    trial_libs,
                    enforce_total_limit=False,
                    lane_metadata=extra_metadata,
                )
                if not imbalance_mix_valid:
                    continue
                # 带缓存增量检查：new_lib 的 index 解析结果同步写入 selected_idx_cache，
                # 后续再判断其他候选时不再重复解析 selected 中已有文库的索引
                idx_valid, lib_indices = _validate_new_lib_quick_with_result_cache(
                    idx_validator=_idx_validator,
                    selected_libraries=selected,
                    selected_indices_cache=selected_idx_cache,
                    new_lib=lib,
                )
                if not idx_valid:
                    continue
                selected.append(lib)
                selected_idx_cache.append(lib_indices)
                total += data
                if random_target is None and total >= trial_min_allowed:
                    random_target = random.uniform(trial_min_allowed, trial_max_allowed)
                    logger.debug(
                        f"Lane打包随机目标: {random_target:.1f}G "
                        f"(范围={trial_min_allowed:.0f}~{trial_max_allowed:.0f}G, 当前={total:.1f}G)"
                    )
                if random_target is not None and total >= random_target:
                    break
        if not selected:
            other_failure_retry_count += 1
            # 大池结构性快速失败：若大量随机尝试均无法选出任何文库，说明约束将整个候选
            # 集锁死，继续随机尝试不会有改善，提前退出避免无效重试（仅在没有 index 冲突
            # 且池较大时触发，保留对 index 冲突的多次随机重试机会）
            if (
                not prioritize_scattered_mix
                and other_failure_retry_count >= 3
                and index_conflict_retry_count == 0
                and len(active_pool) > 200
            ):
                break
            continue
        selected_min_allowed, _ = _resolve_lane_capacity_limits(
            libraries=selected,
            machine_type=machine_type,
            lane_metadata=extra_metadata,
        )
        if total < selected_min_allowed:
            other_failure_retry_count += 1
            # 同理：数据够但始终不达下限，结构性约束问题，大池时早退出
            if (
                not prioritize_scattered_mix
                and other_failure_retry_count >= 5
                and index_conflict_retry_count == 0
                and len(active_pool) > 200
            ):
                break
            continue
        selected_signature = _build_library_identity_signature(
            selected,
            canonicalize=True,
        )
        if prioritize_scattered_mix and selected_signature in seen_selected_signatures:
            logger.debug(
                "散样混排补Lane命中重复候选组合: prefix={}, machine={}, size={}, attempt={}",
                lane_id_prefix,
                machine_type.value,
                len(selected_signature),
                attempt_idx,
            )
            continue
        seen_selected_signatures.add(selected_signature)
        lane_id = f"{lane_id_prefix}_{machine_type.value}_{allocated_lane_serial:03d}"
        lane = LaneAssignment(
            lane_id=lane_id,
            machine_id=f"M_{lane_id}",
            machine_type=machine_type,
            lane_capacity_gb=_lane_capacity_for_machine(machine_type),
        )
        if extra_metadata:
            lane.metadata.update(extra_metadata)
        for lib in selected:
            lane.add_library(lib)
        ai_lane_index_errors = _validate_ai_lane_index_pair_rules(lane, libraries=list(lane.libraries or []))
        if ai_lane_index_errors:
            other_failure_retry_count += 1
            logger.debug(
                "补Lane候选Index对数不足: lane_id={}, errors={}",
                lane.lane_id,
                ai_lane_index_errors,
            )
            continue
        metadata = _build_lane_metadata_for_validator(lane.lane_id, lane.metadata, libraries=lane.libraries)
        cache_key = _build_lane_validation_cache_key(
            machine_type=lane.machine_type.value,
            lane_id=lane.lane_id,
            lane_metadata=lane.metadata,
            libraries=lane.libraries,
        )
        result = cached_lane_validations.get(cache_key)
        if result is None:
            result = _validate_lane_with_latest_index(
                validator=validator,
                libraries=lane.libraries,
                lane_id=lane.lane_id,
                machine_type=lane.machine_type.value,
                metadata=metadata,
            )
            cached_lane_validations[cache_key] = result
        else:
            logger.debug(
                "补Lane全量校验命中缓存: prefix={}, machine={}, lane_id={}, lib_count={}",
                lane_id_prefix,
                machine_type.value,
                lane.lane_id,
                len(selected_signature),
            )
        if result.is_valid:
            return lane, selected
        if _is_index_conflict_only(result):
            index_conflict_retry_count += 1
        else:
            other_failure_retry_count += 1
        if prioritize_scattered_mix:
            logger.debug(
                "散样混排补Lane候选组合验证失败: prefix={}, machine={}, lane_id={}, attempt={}/{}",
                lane_id_prefix,
                machine_type.value,
                lane_id,
                attempt_idx,
                max_scattered_mix_attempts,
            )
            if attempt_idx >= max_scattered_mix_attempts:
                break
            continue
    return None, []


def _attempt_build_rescue_lane_from_pool(
    pool: List[EnhancedLibraryInfo],
    validator,
    machine_type: MachineType,
    lane_id_prefix: str,
    lane_serial: Optional[int] = None,
    index_conflict_attempts: int = DEFAULT_INDEX_CONFLICT_ATTEMPTS,
    other_failure_attempts: int = DEFAULT_OTHER_FAILURE_ATTEMPTS,
    extra_metadata: Optional[Dict[str, Any]] = None,
    lane_validation_cache: Optional[
        Dict[Tuple[str, Tuple[Tuple[str, str], ...], Tuple[str, ...]], Any]
    ] = None,
) -> Tuple[LaneAssignment | None, List[EnhancedLibraryInfo]]:
    """仅在 RB/EX 补Lane阶段启用的窄范围回退。

    先按临检/SJ/YC优先顺序尝试；失败后只退到非临检/SJ池，
    避免为了救援Lane打散更高优先级文库。
    """
    if not pool:
        return None, []

    machine_type_text = machine_type.value if isinstance(machine_type, MachineType) else str(machine_type)
    metadata_key = tuple(
        sorted((str(key), repr(value)) for key, value in dict(extra_metadata or {}).items())
    )
    pool_total_data = sum(lib.get_data_amount_gb() for lib in pool)
    pool_min_lane_data, _ = _resolve_lane_capacity_limits(pool, machine_type)
    if pool_total_data + 1e-6 < pool_min_lane_data:
        logger.info(
            "补Lane尝试跳过: lane_prefix={}, pool_size={}, pool_data={:.3f}G不足最小门槛{:.3f}G".format(
                lane_id_prefix,
                len(pool),
                pool_total_data,
                pool_min_lane_data,
            )
        )
        return None, []

    # Stage 3 的 OG 普通尾货聚簇补Lane本身已经是同类、低优先级、窄池候选，
    # 再走散样混排优先路径收益很低，却会显著放大排序与重试成本。
    if lane_id_prefix == "OG":
        candidate_pool = list(pool)
        candidate_total_data = sum(lib.get_data_amount_gb() for lib in candidate_pool)
        candidate_min_lane_data, _ = _resolve_lane_capacity_limits(candidate_pool, machine_type)
        if candidate_total_data + 1e-6 < candidate_min_lane_data:
            logger.info(
                "补Lane尝试跳过: lane_prefix={}, pool_size={}, pool_data={:.3f}G不足最小门槛{:.3f}G".format(
                    lane_id_prefix,
                    len(candidate_pool),
                    candidate_total_data,
                    candidate_min_lane_data,
                )
            )
            return None, []
        logger.info(
            "补Lane尝试: variant=cluster_fast_path, lane_prefix={}, pool_size={}, pool_data={:.3f}G, prioritize_scattered_mix=False".format(
                lane_id_prefix,
                len(candidate_pool),
                candidate_total_data,
            )
        )
        return _attempt_build_lane_from_pool(
            pool=candidate_pool,
            validator=validator,
            machine_type=machine_type,
            lane_id_prefix=lane_id_prefix,
            lane_serial=lane_serial,
            index_conflict_attempts=index_conflict_attempts,
            other_failure_attempts=other_failure_attempts,
            extra_metadata=extra_metadata,
            prioritize_scattered_mix=False,
            deterministic_candidate_order=None,
            lane_validation_cache=lane_validation_cache,
        )

    seen_variants: Set[Tuple[bool, Tuple[int, ...]]] = set()
    variants: List[Tuple[str, List[EnhancedLibraryInfo], bool]] = []
    quick_skip_stats: Counter[str] = Counter()

    bucket_variants = _build_rescue_rule_buckets(pool, lane_id_prefix)
    for bucket_name, bucket_pool in bucket_variants:
        feasible, reason = _quick_check_pool_feasibility(
            pool=bucket_pool,
            machine_type=machine_type,
            lane_metadata=extra_metadata,
            stage_label=f"{lane_id_prefix}_{bucket_name}",
        )
        if not feasible:
            quick_skip_stats[reason] += 1
            continue
        variants.append((bucket_name, list(bucket_pool), True))
        if len(bucket_pool) >= 32:
            variants.append((f"{bucket_name}:relaxed", list(bucket_pool), False))

    non_clinical_pool = [
        lib for lib in pool
        if _get_scattered_mix_priority_rank(lib) >= 1
    ]
    other_only_pool = [
        lib for lib in non_clinical_pool
        if _get_scattered_mix_priority_rank(lib) == 2
    ]

    variants.append(("full_priority", list(pool), True))
    if non_clinical_pool and len(non_clinical_pool) < len(pool):
        variants.append(("non_clinical_priority", list(non_clinical_pool), True))
    if non_clinical_pool:
        variants.append(("non_clinical_relaxed", list(non_clinical_pool), False))
    if other_only_pool and len(other_only_pool) < len(non_clinical_pool):
        variants.append(("other_only_relaxed", list(other_only_pool), False))

    for variant_name, candidate_pool, prioritize_scattered_mix in variants:
        candidate_compact_signature = _build_library_compact_identity_signature(
            candidate_pool,
            canonicalize=True,
        )
        key = (prioritize_scattered_mix, candidate_compact_signature)
        if key in seen_variants:
            continue
        seen_variants.add(key)
        failure_cache_key = (
            machine_type_text,
            lane_id_prefix,
            variant_name,
            metadata_key,
        )
        failed_signatures = _RESCUE_STRICT_FAILURE_SIGNATURES.get(failure_cache_key, [])
        if any(
            _is_near_subset_signature(candidate_compact_signature, failed_signature)
            for failed_signature in failed_signatures
        ):
            quick_skip_stats["同池严格失败缓存命中"] += 1
            continue

        variant_cache_key = (
            machine_type_text,
            lane_id_prefix,
            variant_name,
            metadata_key,
            candidate_compact_signature,
            prioritize_scattered_mix,
        )
        cached_variant_result = _RESCUE_VARIANT_ATTEMPT_CACHE.get(variant_cache_key)
        if cached_variant_result is not None:
            cached_success, cached_reason = cached_variant_result
            if not cached_success:
                quick_skip_stats[cached_reason or "variant缓存失败命中"] += 1
                continue

        feasible, reason = _quick_check_pool_feasibility(
            pool=candidate_pool,
            machine_type=machine_type,
            lane_metadata=extra_metadata,
            stage_label=f"{lane_id_prefix}_{variant_name}",
        )
        if not feasible:
            quick_skip_stats[reason] += 1
            _RESCUE_VARIANT_ATTEMPT_CACHE[variant_cache_key] = (False, reason)
            continue
        candidate_total_data = sum(lib.get_data_amount_gb() for lib in candidate_pool)
        candidate_min_lane_data, _ = _resolve_lane_capacity_limits(candidate_pool, machine_type)
        if candidate_total_data + 1e-6 < candidate_min_lane_data:
            reason = "总量{:.3f}G不足最小门槛{:.3f}G".format(
                candidate_total_data,
                candidate_min_lane_data,
            )
            quick_skip_stats[reason] += 1
            _RESCUE_VARIANT_ATTEMPT_CACHE[variant_cache_key] = (False, reason)
            continue
        logger.info(
            "补Lane尝试: variant={}, lane_prefix={}, pool_size={}, pool_data={:.3f}G, prioritize_scattered_mix={}".format(
                variant_name,
                lane_id_prefix,
                len(candidate_pool),
                candidate_total_data,
                prioritize_scattered_mix,
            )
        )
        lane, used = _attempt_build_lane_from_pool(
            pool=candidate_pool,
            validator=validator,
            machine_type=machine_type,
            lane_id_prefix=lane_id_prefix,
            lane_serial=lane_serial,
            index_conflict_attempts=index_conflict_attempts,
            other_failure_attempts=other_failure_attempts,
            extra_metadata=extra_metadata,
            prioritize_scattered_mix=prioritize_scattered_mix,
            deterministic_candidate_order=None,
            lane_validation_cache=lane_validation_cache,
        )
        if lane:
            _RESCUE_VARIANT_ATTEMPT_CACHE[variant_cache_key] = (True, "ok")
            return lane, used
        _RESCUE_VARIANT_ATTEMPT_CACHE[variant_cache_key] = (False, "严格构Lane失败")
        if not prioritize_scattered_mix:
            failed_signatures.append(candidate_compact_signature)
            if len(failed_signatures) > 16:
                failed_signatures[:] = failed_signatures[-16:]
            _RESCUE_STRICT_FAILURE_SIGNATURES[failure_cache_key] = failed_signatures

    if quick_skip_stats:
        top_reasons = ", ".join(
            "{}x{}".format(reason, count)
            for reason, count in quick_skip_stats.most_common(4)
        )
        logger.info(
            "补Lane跳过汇总: lane_prefix={}, machine={}, reason_top=[{}]",
            lane_id_prefix,
            machine_type_text,
            top_reasons,
        )
    return None, []


def _attempt_build_lane_from_prioritized_pool(
    primary_pool: List[EnhancedLibraryInfo],
    secondary_pool: List[EnhancedLibraryInfo],
    validator,
    machine_type: MachineType,
    lane_id_prefix: str,
    lane_serial: int,
    index_conflict_attempts: int = 10,
    other_failure_attempts: int = 20,
    match_fn: Optional[Any] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    lane_validation_cache: Optional[
        Dict[Tuple[str, Tuple[Tuple[str, str], ...], Tuple[str, ...]], Any]
    ] = None,
) -> Tuple[LaneAssignment | None, List[EnhancedLibraryInfo]]:
    """优先消化失败Lane回收池，其次用未分配池补齐的构Lane逻辑。"""
    if not primary_pool and not secondary_pool:
        return None, []
    active_primary_pool, active_secondary_pool, _ = _filter_priority_across_pools(
        list(primary_pool),
        list(secondary_pool),
        machine_type=machine_type,
        stage_name=f"{lane_id_prefix}_prioritized_build",
        emit_log=True,
    )
    if not active_primary_pool and not active_secondary_pool:
        return None, []

    # 复用模块级单例，不重复初始化
    idx_validator = _MODULE_IDX_VALIDATOR
    index_conflict_retry_count = 0
    other_failure_retry_count = 0
    seen_selected_signatures: Set[Tuple[str, ...]] = set()
    cached_lane_validations = lane_validation_cache if lane_validation_cache is not None else {}
    special_data_limit = _resolve_special_library_data_limit(machine_type)
    while (
        index_conflict_retry_count < index_conflict_attempts
        and other_failure_retry_count < other_failure_attempts
    ):
        selected: List[EnhancedLibraryInfo] = []
        selected_ids: Set[int] = set()
        # 与 selected 平行维护的预解析索引缓存，同 _attempt_build_lane_from_pool 的优化逻辑
        selected_idx_cache: List = []
        total = 0.0
        random_target: Optional[float] = None

        def _ordered_candidates(
            pool: List[EnhancedLibraryInfo],
            prefer_balanced: bool,
            current_lane_libs: Optional[List[EnhancedLibraryInfo]] = None,
        ) -> List[EnhancedLibraryInfo]:
            filtered_pool = [lib for lib in pool if match_fn(lib)] if match_fn is not None else list(pool)
            lane_context = list(current_lane_libs or [])
            lane_summary = _resolve_lane_imbalance_summary(lane_context) if lane_context else None
            return sorted(
                filtered_pool,
                key=lambda lib: (
                    0 if (_is_imbalance_library_candidate(lib) is (not prefer_balanced)) else 1,
                    *_build_imbalance_fill_sort_key(
                        lane_context,
                        lib,
                        machine_type=machine_type,
                        special_data_limit=special_data_limit,
                        lane_summary=lane_summary,
                        candidate_is_imbalance=_is_imbalance_library_candidate(lib),
                    ),
                    -float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
                    str(getattr(lib, "origrec", "") or ""),
                ),
            )

        while True:
            selected_min_allowed = 0.0
            if selected:
                selected_min_allowed, _ = _resolve_lane_capacity_limits(
                    libraries=selected,
                    machine_type=machine_type,
                    lane_id=f"{lane_id_prefix}_TMP",
                )
            selected_lane_summary = _resolve_lane_imbalance_summary(selected) if selected else None
            lane_context_for_sort = selected if selected else None
            candidate_buckets = [
                _ordered_candidates(
                    _filter_libraries_excluding_object_ids(
                        active_primary_pool,
                        selected_ids,
                    ),
                    prefer_balanced=False,
                    current_lane_libs=lane_context_for_sort,
                ),
                _ordered_candidates(
                    _filter_libraries_excluding_object_ids(
                        active_secondary_pool,
                        selected_ids,
                    ),
                    prefer_balanced=True,
                    current_lane_libs=lane_context_for_sort,
                ),
            ]

            added_candidate = False
            for candidates in candidate_buckets:
                for lib in candidates:
                    candidate_is_imbalance = _is_imbalance_library_candidate(lib)
                    projected_special_data = _project_lane_special_data(
                        selected,
                        lib,
                        lane_summary=selected_lane_summary,
                        candidate_is_imbalance=candidate_is_imbalance,
                    )
                    if projected_special_data > special_data_limit + SPECIAL_LIBRARY_LIMIT_EPSILON:
                        continue
                    if (
                        selected
                        and total >= selected_min_allowed
                        and not _candidate_improves_imbalance_target(
                            selected,
                            lib,
                            machine_type=machine_type,
                            special_data_limit=special_data_limit,
                            lane_summary=selected_lane_summary,
                            candidate_is_imbalance=candidate_is_imbalance,
                        )
                    ):
                        continue
                    data = lib.get_data_amount_gb()
                    trial_libs = selected + [lib]
                    trial_min_allowed, trial_max_allowed = _resolve_lane_capacity_limits(
                        libraries=trial_libs,
                        machine_type=machine_type,
                        lane_id=f"{lane_id_prefix}_TMP",
                    )
                    if total + data > trial_max_allowed:
                        continue
                    candidate_light_valid, _ = _quick_check_lane_candidate(
                        libraries=trial_libs,
                        machine_type=machine_type,
                        lane_metadata=extra_metadata,
                        stage_label=f"{lane_id_prefix}_candidate",
                    )
                    if not candidate_light_valid:
                        continue
                    # 带缓存增量检查，避免对 selected 中已有文库的索引反复解析
                    idx_valid, lib_indices = _validate_new_lib_quick_with_result_cache(
                        idx_validator=idx_validator,
                        selected_libraries=selected,
                        selected_indices_cache=selected_idx_cache,
                        new_lib=lib,
                    )
                    if not idx_valid:
                        continue
                    selected.append(lib)
                    selected_ids.add(id(lib))
                    selected_idx_cache.append(lib_indices)
                    total += data
                    added_candidate = True
                    break
                if added_candidate:
                    break
            if not added_candidate:
                break

        if not selected:
            other_failure_retry_count += 1
            # 大池结构性快速失败（同 _attempt_build_lane_from_pool 的逻辑）
            if (
                other_failure_retry_count >= 3
                and index_conflict_retry_count == 0
                and (len(active_primary_pool) + len(active_secondary_pool)) > 200
            ):
                break
            continue

        selected_min_allowed, _ = _resolve_lane_capacity_limits(
            libraries=selected,
            machine_type=machine_type,
        )
        if total < selected_min_allowed:
            other_failure_retry_count += 1
            if (
                other_failure_retry_count >= 5
                and index_conflict_retry_count == 0
                and (len(active_primary_pool) + len(active_secondary_pool)) > 200
            ):
                break
            continue
        selected_signature = _build_library_identity_signature(
            selected,
            canonicalize=True,
        )
        if selected_signature in seen_selected_signatures:
            logger.debug(
                "优先池补Lane命中重复候选组合，终止无效重试: prefix={}, machine={}, size={}",
                lane_id_prefix,
                machine_type.value,
                len(selected_signature),
            )
            break
        seen_selected_signatures.add(selected_signature)

        lane_id = f"{lane_id_prefix}_{machine_type.value}_{lane_serial:03d}"
        lane = LaneAssignment(
            lane_id=lane_id,
            machine_id=f"M_{lane_id}",
            machine_type=machine_type,
            lane_capacity_gb=_lane_capacity_for_machine(machine_type),
        )
        if extra_metadata:
            lane.metadata.update(extra_metadata)
        for lib in selected:
            lane.add_library(lib)
        cache_key = _build_lane_validation_cache_key(
            machine_type=lane.machine_type.value,
            lane_id=lane.lane_id,
            lane_metadata=lane.metadata,
            libraries=lane.libraries,
        )
        result = cached_lane_validations.get(cache_key)
        if result is None:
            result = _validate_lane_state(validator, lane, lane.libraries)
            cached_lane_validations[cache_key] = result
        else:
            logger.debug(
                "优先池补Lane全量校验命中缓存: prefix={}, machine={}, lane_id={}, lib_count={}",
                lane_id_prefix,
                machine_type.value,
                lane.lane_id,
                len(selected_signature),
            )
        if result.is_valid:
            return lane, selected
        if _is_index_conflict_only(result):
            index_conflict_retry_count += 1
        else:
            other_failure_retry_count += 1
        logger.debug(
            "优先池补Lane候选组合验证失败且路径确定，停止重复尝试: prefix={}, machine={}, lane_id={}",
            lane_id_prefix,
            machine_type.value,
            lane_id,
        )
        break

    return None, []


def _remove_used_libraries_from_pools(
    primary_pool: List[EnhancedLibraryInfo],
    secondary_pool: List[EnhancedLibraryInfo],
    used: List[EnhancedLibraryInfo],
) -> Tuple[List[EnhancedLibraryInfo], List[EnhancedLibraryInfo]]:
    used_keys = {_get_library_identity_key(lib) for lib in used}
    primary_pool = [
        lib for lib in primary_pool
        if _get_library_identity_key(lib) not in used_keys
    ]
    secondary_pool = [
        lib for lib in secondary_pool
        if _get_library_identity_key(lib) not in used_keys
    ]
    return primary_pool, secondary_pool


def _drain_rescue_lanes_for_match(
    primary_pool: List[EnhancedLibraryInfo],
    secondary_pool: List[EnhancedLibraryInfo],
    validator: Any,
    machine_type: MachineType,
    lane_prefix: str,
    serial_start: int,
    match_fn: Any,
    extra_metadata: Optional[Dict[str, Any]] = None,
    lane_validation_cache: Optional[
        Dict[Tuple[str, Tuple[Tuple[str, str], ...], Tuple[str, ...]], Any]
    ] = None,
) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo], List[EnhancedLibraryInfo], int]:
    """对指定匹配条件持续抽取救援Lane。"""
    lanes: List[LaneAssignment] = []
    serial = serial_start
    while True:
        lane, used = _attempt_build_lane_from_prioritized_pool(
            primary_pool=primary_pool,
            secondary_pool=secondary_pool,
            validator=validator,
            machine_type=machine_type,
            lane_id_prefix=lane_prefix,
            lane_serial=serial,
            match_fn=match_fn,
            extra_metadata=extra_metadata,
            lane_validation_cache=lane_validation_cache,
        )
        if lane is None or not used:
            break
        lanes.append(lane)
        serial += 1
        primary_pool, secondary_pool = _remove_used_libraries_from_pools(
            primary_pool,
            secondary_pool,
            used,
        )
    return lanes, primary_pool, secondary_pool, serial


def _get_library_identity_key(lib: EnhancedLibraryInfo) -> str:
    """获取文库在当前流程中的稳定唯一键。"""
    cache_key = (
        getattr(lib, "_detail_output_key", None),
        bool(_is_split_library(lib)),
        getattr(lib, "fragment_id", None),
        getattr(lib, "wkaidbid", None),
        getattr(lib, "aidbid", None),
        getattr(lib, "_origrec_key", None),
        getattr(lib, "origrec", None),
    )
    cached = getattr(lib, "_library_identity_key_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    detail_output_key = _safe_str(
        getattr(lib, "_detail_output_key", None),
        default="",
    )
    if detail_output_key:
        setattr(lib, "_library_identity_key_cache", (cache_key, detail_output_key))
        return detail_output_key

    if _is_split_library(lib):
        for attr_name in ("fragment_id", "wkaidbid", "aidbid"):
            candidate = _safe_str(getattr(lib, attr_name, None), default="")
            if candidate:
                setattr(lib, "_library_identity_key_cache", (cache_key, candidate))
                return candidate

    origrec_key = _safe_str(
        getattr(lib, "_origrec_key", getattr(lib, "origrec", "")),
        default="",
    )
    if origrec_key:
        setattr(lib, "_library_identity_key_cache", (cache_key, origrec_key))
        return origrec_key
    fallback_key = str(id(lib))
    setattr(lib, "_library_identity_key_cache", (cache_key, fallback_key))
    return fallback_key


def _get_compact_library_identity_value(lib: EnhancedLibraryInfo) -> int:
    """将稳定文库键映射为紧凑整数，降低高频缓存键比较成本。"""
    global _COMPACT_LIBRARY_IDENTITY_NEXT
    identity_key = _get_library_identity_key(lib)
    cached = _COMPACT_LIBRARY_IDENTITY_BY_KEY.get(identity_key)
    if cached is not None:
        return cached
    _COMPACT_LIBRARY_IDENTITY_NEXT += 1
    _COMPACT_LIBRARY_IDENTITY_BY_KEY[identity_key] = _COMPACT_LIBRARY_IDENTITY_NEXT
    return _COMPACT_LIBRARY_IDENTITY_NEXT


def _build_library_compact_identity_signature(
    libraries: List[EnhancedLibraryInfo],
    *,
    canonicalize: bool = False,
) -> Tuple[int, ...]:
    """构建更紧凑的文库组合签名，用于运行时高频缓存。"""
    signature = tuple(_get_compact_library_identity_value(lib) for lib in libraries)
    if canonicalize:
        return tuple(sorted(signature))
    return signature


def _build_library_identity_signature(
    libraries: List[EnhancedLibraryInfo],
    *,
    canonicalize: bool = False,
) -> Tuple[str, ...]:
    """构建文库组合签名；canonicalize=True 时忽略候选顺序。"""
    signature = tuple(_get_library_identity_key(lib) for lib in libraries)
    if canonicalize:
        return tuple(sorted(signature))
    return signature


def _is_near_subset_signature(
    candidate_signature: Tuple[int, ...],
    cached_signature: Tuple[int, ...],
    *,
    allowed_delta_count: int = 2,
) -> bool:
    """判断两个候选池签名是否几乎未变化，用于跳过重复失败重试。"""
    if candidate_signature == cached_signature:
        return True
    candidate_counts = Counter(candidate_signature)
    cached_counts = Counter(cached_signature)
    keys = set(candidate_counts) | set(cached_counts)
    delta = sum(abs(candidate_counts.get(key, 0) - cached_counts.get(key, 0)) for key in keys)
    return delta <= allowed_delta_count


def _build_lane_validation_cache_key(
    *,
    machine_type: MachineType | str,
    lane_id: str,
    lane_metadata: Optional[Dict[str, Any]],
    libraries: List[EnhancedLibraryInfo],
) -> Tuple[str, Tuple[Tuple[str, str], ...], Tuple[str, ...]]:
    """构建Lane校验缓存键，按机型、校验上下文和等价文库组合收口。"""
    validator_metadata = _build_lane_metadata_for_validator(lane_id, lane_metadata, libraries=libraries)
    return (
        _machine_type_to_text(machine_type, default="Nova X-25B"),
        tuple(sorted((str(key), repr(value)) for key, value in validator_metadata.items())),
        _build_library_identity_signature(libraries, canonicalize=True),
    )


def _validate_new_lib_quick_with_result_cache(
    *,
    idx_validator,
    selected_libraries: List[EnhancedLibraryInfo],
    selected_indices_cache: List[List[Tuple[str, Optional[str]]]],
    new_lib: EnhancedLibraryInfo,
) -> Tuple[bool, List[Tuple[str, Optional[str]]]]:
    """对增量 index 快校验增加“等价已选集合 + 候选文库”结果缓存。"""
    cache_key = (
        _build_library_identity_signature(selected_libraries, canonicalize=True),
        _get_library_identity_key(new_lib),
    )
    cached = _QUICK_INDEX_VALIDATION_RESULT_CACHE.get(cache_key)
    if cached is not None:
        cached_valid, cached_indices = cached
        return cached_valid, list(cached_indices)

    idx_valid, parsed_indices = idx_validator.validate_new_lib_quick_with_cache(
        selected_indices_cache,
        new_lib,
    )
    _QUICK_INDEX_VALIDATION_RESULT_CACHE[cache_key] = (
        idx_valid,
        tuple(parsed_indices),
    )
    return idx_valid, parsed_indices


def _build_library_object_id_set(libraries: List[EnhancedLibraryInfo]) -> Set[int]:
    """按对象身份构建文库集合，避免 dataclass 深度相等比较。"""
    return {id(lib) for lib in libraries}


def _filter_libraries_excluding_object_ids(
    libraries: List[EnhancedLibraryInfo],
    excluded_ids: Set[int],
) -> List[EnhancedLibraryInfo]:
    """按对象身份过滤文库列表。"""
    if not excluded_ids:
        return list(libraries)
    return [lib for lib in libraries if id(lib) not in excluded_ids]


def _remove_library_by_identity_in_place(
    libraries: List[EnhancedLibraryInfo],
    target: EnhancedLibraryInfo,
) -> bool:
    """按对象身份删除单个文库。"""
    target_id = id(target)
    for index, lib in enumerate(libraries):
        if id(lib) == target_id:
            libraries.pop(index)
            return True
    return False


def _remove_libraries_by_identity_in_place(
    libraries: List[EnhancedLibraryInfo],
    targets: List[EnhancedLibraryInfo],
) -> int:
    """按对象身份批量删除文库。"""
    target_ids = _build_library_object_id_set(targets)
    if not target_ids:
        return 0
    original_len = len(libraries)
    libraries[:] = [lib for lib in libraries if id(lib) not in target_ids]
    return original_len - len(libraries)


def _get_library_source_origrec_key(lib: EnhancedLibraryInfo) -> str:
    """获取文库对应原始输入行的归属键。"""
    source_key = _safe_str(
        getattr(lib, "_source_origrec_key", None) or getattr(lib, "_origrec_key", None),
        default="",
    )
    if source_key:
        return source_key
    return _safe_str(getattr(lib, "origrec", ""), default="")


def _get_split_family_id_for_lane_build(lib: EnhancedLibraryInfo) -> str:
    """提取拆分家族标识，用于补Lane阶段禁止同家族片段进入同一Lane。"""
    explicit_family_id = _safe_str(getattr(lib, "original_library_id", None), default="")
    if explicit_family_id:
        return explicit_family_id

    if _is_split_library(lib) or int(getattr(lib, "total_fragments", 0) or 0) > 1:
        return _safe_str(getattr(lib, "origrec", None), default="")
    return ""


def _get_split_family_expected_count(
    family_id: str,
    family_counts: Dict[str, int],
    lib: EnhancedLibraryInfo,
) -> int:
    """拆分家族期望片段数；输入仅标记wkissplit时用当前家族实际片段数兜底。"""
    explicit_count = int(getattr(lib, "total_fragments", 0) or 0)
    if explicit_count > 1:
        return explicit_count
    if _is_split_library(lib) and family_id:
        return max(1, int(family_counts.get(family_id, 0) or 0))
    return explicit_count


def _shares_split_family_with_selected(
    selected: List[EnhancedLibraryInfo],
    candidate: EnhancedLibraryInfo,
) -> bool:
    """判断候选文库是否与当前Lane已选文库属于同一拆分家族。"""
    candidate_family_id = _get_split_family_id_for_lane_build(candidate)
    if not candidate_family_id:
        return False
    for existing_lib in selected:
        if _get_split_family_id_for_lane_build(existing_lib) == candidate_family_id:
            return True
    return False


def _get_library_detail_output_key(lib: EnhancedLibraryInfo) -> str:
    """获取明细输出按子文库展开时使用的稳定键。"""
    return _get_library_identity_key(lib)


def _is_ai_balance_library(lib: Any) -> bool:
    """判断是否为排机后新增的AI平衡文库。"""
    if bool(getattr(lib, BALANCE_LIBRARY_MARKER_COLUMN, False)):
        return True
    sample_id = _safe_str(getattr(lib, "sample_id", None) or getattr(lib, "wksampleid", None), default="")
    return sample_id.lower() == "phix"


def _calculate_balance_amount_for_final_ratio(non_balance_amount: float, balance_ratio: float) -> float:
    """按最终Lane总量占比计算应补平衡文库量。"""
    non_balance_amount = float(non_balance_amount or 0.0)
    balance_ratio = float(balance_ratio or 0.0)
    if non_balance_amount <= 0 or balance_ratio <= 0:
        return 0.0
    denominator = max(1.0 - balance_ratio, MIN_BALANCE_RATIO_DENOMINATOR)
    return non_balance_amount * balance_ratio / denominator


def _parse_balance_library_config_rows() -> List[Dict[str, Any]]:
    """读取平衡文库配置表并跳过说明行。"""
    if not BALANCE_LIBRARY_CONFIG_PATH.exists():
        logger.warning(f"平衡文库配置表不存在: {BALANCE_LIBRARY_CONFIG_PATH}")
        return []

    df = _read_csv_with_encoding_fallback(BALANCE_LIBRARY_CONFIG_PATH)
    rows: List[Dict[str, Any]] = []
    for order, (_, row) in enumerate(df.iterrows()):
        row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        sample_id = _safe_str(row_dict.get("wksampleid"), default="")
        dept = _safe_str(row_dict.get("wkdept"), default="")
        test_no = _safe_str(row_dict.get("wktestno"), default="") or _safe_str(
            row_dict.get("wktestno.1"), default=""
        )
        index_seq = _safe_str(row_dict.get("wkindexseq"), default="")
        if sample_id in {"文库ID", ""} and dept in {"实验室名称", ""} and test_no in {"工序名称", ""}:
            continue
        if not dept or not test_no or not index_seq:
            continue
        row_dict["wktestno"] = test_no
        row_dict["_template_order"] = order
        rows.append(row_dict)
    return rows


@lru_cache(maxsize=1)
def _load_balance_library_templates() -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """按实验室+工序缓存平衡文库模板，保留CSV原始优先级。"""
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in _parse_balance_library_config_rows():
        dept_key = _normalize_text_for_match(row.get("wkdept"))
        test_key = _normalize_text_for_match(row.get("wktestno"))
        if not dept_key or not test_key:
            continue
        buckets.setdefault((dept_key, test_key), []).append(row)
    return buckets


def _index_seq_contains_pe(index_seq: Any) -> bool:
    """判断index字符串中是否存在字面值PE。"""
    text = _safe_str(index_seq, default="")
    if not text:
        return False
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        parts = [part.strip().upper() for part in item.split(";") if part.strip()]
        if any(part == "PE" for part in parts):
            return True
    return False


def _derive_balance_base_type(index_seq: str) -> str:
    return "单" if ";" not in _safe_str(index_seq, default="") else "双"


def _derive_balance_index_bases(index_seq: str) -> int:
    text = _safe_str(index_seq, default="")
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        p7 = item.split(";")[0].strip()
        if p7.upper() in {"PE", "通用接头", "随机INDEX"}:
            continue
        return len(p7)
    return 0


def _get_lane_lab_name(lane: LaneAssignment) -> str:
    """获取lane所属实验室名称。"""
    for lib in list(getattr(lane, "libraries", []) or []):
        if _is_ai_balance_library(lib):
            continue
        for attr_name in ("wkdept", "_wkdept_raw", "dept"):
            value = _safe_str(getattr(lib, attr_name, None), default="")
            if value:
                return value
    return ""


def _get_lane_process_name(lane: LaneAssignment) -> str:
    """获取lane所属工序名称。"""
    for lib in list(getattr(lane, "libraries", []) or []):
        if _is_ai_balance_library(lib):
            continue
        value = _safe_str(getattr(lib, "test_no", None) or getattr(lib, "testno", None), default="")
        if value:
            return value
    return ""


def _get_lane_explicit_balance_data(lane: LaneAssignment) -> float:
    """读取lane已明确给定的平衡文库补量。"""
    if isinstance(lane.metadata, dict):
        for key in ("wkbalancedata", "wkadd_balance_data", "required_balance_data_gb"):
            value = _safe_float(lane.metadata.get(key), default=0.0)
            if value > 0:
                return value
    lane_level_values: List[float] = []
    for lib in list(getattr(lane, "libraries", []) or []):
        for attr_name in ("balance_data", "balancedata"):
            value = _safe_float(getattr(lib, attr_name, None), default=0.0)
            if value > 0:
                lane_level_values.append(value)
    return max(lane_level_values) if lane_level_values else 0.0


def _get_library_jjbj_flag(lib: EnhancedLibraryInfo) -> str:
    """读取文库碱基不均标记，兼容模型字段和原始输出字段。"""
    return _safe_str(
        getattr(lib, "jjbj", None) or getattr(lib, "wk_jjbj", None),
        default="",
    ).strip()


def _is_all_real_libraries_base_imbalanced(libraries: List[EnhancedLibraryInfo]) -> bool:
    """判断lane内真实文库是否全部明确标记为碱基不均。"""
    real_libraries = [
        lib
        for lib in list(libraries or [])
        if not _is_ai_balance_library(lib)
    ]
    if not real_libraries:
        return False
    for lib in real_libraries:
        flag = _get_library_jjbj_flag(lib)
        if flag:
            if flag != "是":
                return False
            continue
        group_id = _BASE_IMBALANCE_HANDLER.identify_imbalance_type(lib)
        if not group_id:
            return False
    return True


def _resolve_imbalance_group_balance_ratio(group_id: str) -> float:
    """按碱基不均组别解析最终lane中应占的平衡文库比例。"""
    if not group_id or group_id == "G_UNKNOWN":
        return 0.0
    group_info = _BASE_IMBALANCE_HANDLER.get_group_info(group_id)
    if not group_info:
        return 0.0
    explicit_ratio = float(getattr(group_info, "phix_ratio", 0.0) or 0.0)
    if explicit_ratio > 0:
        return explicit_ratio
    return float(_BASE_IMBALANCE_HANDLER.get_group_balance_ratio(group_id) or 0.0)


def _resolve_lane_imbalance_groups_and_ratio(
    libraries: List[EnhancedLibraryInfo],
) -> Tuple[Set[str], float]:
    """识别lane内碱基不均组别组合，并取该组合需要的平衡文库比例。"""
    group_ids: Set[str] = set()
    ratio = 0.0
    for lib in list(libraries or []):
        if _is_ai_balance_library(lib):
            continue
        group_id = _BASE_IMBALANCE_HANDLER.identify_imbalance_type(lib)
        if not group_id:
            continue
        group_ids.add(group_id)
        ratio = max(ratio, _resolve_imbalance_group_balance_ratio(group_id))
    return group_ids, ratio


def _is_dedicated_imbalance_lane_context(
    libraries: List[EnhancedLibraryInfo],
    lane_id: str = "",
    lane_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """统一判断专用碱基不均lane，避免只认DL_前缀。"""
    lane_id_text = _safe_str(lane_id, default="")
    metadata = lane_metadata or {}
    if lane_id_text.startswith("DL_") or bool(metadata.get("is_dedicated_imbalance_lane")):
        return True
    return _is_all_real_libraries_base_imbalanced(libraries)


def _is_explicit_dedicated_imbalance_lane(lane: LaneAssignment) -> bool:
    """判断lane是否被明确标记为碱基不均衡专用lane。"""
    return _is_dedicated_imbalance_lane_context(
        libraries=list(getattr(lane, "libraries", []) or []),
        lane_id=_safe_str(getattr(lane, "lane_id", None), default=""),
        lane_metadata=getattr(lane, "metadata", None),
    )


def _is_replaceable_normal_library(lib: EnhancedLibraryInfo) -> bool:
    """判断文库是否属于平衡文库腾挪时允许调整的普通文库。"""
    if _is_ai_balance_library(lib):
        return False
    return not _BASE_IMBALANCE_HANDLER.is_imbalance_library(lib)


def _get_explicit_balance_data_from_context(
    libraries: List[EnhancedLibraryInfo],
    lane_metadata: Optional[Dict[str, Any]] = None,
) -> float:
    """从 lane metadata 或 lane 内文库属性提取显式平衡补量。"""
    if isinstance(lane_metadata, dict):
        for key in ("wkbalancedata", "wkadd_balance_data", "required_balance_data_gb"):
            value = _safe_float(lane_metadata.get(key), default=0.0)
            if value > 0:
                return value

    lane_level_values: List[float] = []
    for lib in libraries:
        if _is_ai_balance_library(lib):
            continue
        for attr_name in ("balance_data", "balancedata"):
            value = _safe_float(getattr(lib, attr_name, None), default=0.0)
            if value > 0:
                lane_level_values.append(value)
    return max(lane_level_values) if lane_level_values else 0.0


def _is_package_lane_context(
    libraries: List[EnhancedLibraryInfo],
    lane_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """在无 LaneAssignment 上下文时判断是否属于包 lane。"""
    for lib in libraries:
        package_lane_number = _safe_str(
            getattr(lib, "package_lane_number", None) or getattr(lib, "baleno", None),
            default="",
        )
        if package_lane_number:
            return True
    if isinstance(lane_metadata, dict) and bool(lane_metadata.get("is_package_lane")):
        return bool(
            _safe_str(
                lane_metadata.get("package_id")
                or lane_metadata.get("package_lane_number")
                or lane_metadata.get("baleno"),
                default="",
            )
        )
    return False


def _resolve_lane_balance_ratio_from_libraries(libraries: List[EnhancedLibraryInfo]) -> float:
    """按 lane 内普通文库解析平衡文库比例。"""
    _, ratio = _resolve_lane_imbalance_groups_and_ratio(libraries)
    return ratio


def _resolve_balance_reservation_context(
    libraries: List[EnhancedLibraryInfo],
    machine_type: MachineType | str,
    lane_id: str = "",
    lane_metadata: Optional[Dict[str, Any]] = None,
    selection: Any = None,
) -> Dict[str, Any]:
    """解析当前 lane 是否需要为平衡文库预留容量，以及预留方式。"""
    metadata = lane_metadata or {}
    preserve_reservation = bool(metadata.get("preserve_balance_reservation"))
    if metadata.get("materialized_balance_library") and not preserve_reservation:
        return {"applied": False}
    if any(_is_ai_balance_library(lib) for lib in libraries) and not preserve_reservation:
        return {"applied": False}
    lane_id_text = _safe_str(lane_id, default="")
    is_dedicated = _is_dedicated_imbalance_lane_context(
        libraries=libraries,
        lane_id=lane_id_text,
        lane_metadata=metadata,
    )
    known_non_package_lane = (
        not bool(metadata.get("is_package_lane"))
        and lane_id_text.startswith(("GL_", "MX_", "EX_", "RB_", "RM_", "OG_", "M11R2_"))
    )
    if known_non_package_lane and not is_dedicated:
        return {"applied": False}

    explicit_balance_gb = _get_explicit_balance_data_from_context(libraries, metadata)
    if _is_package_lane_context(libraries, lane_metadata=metadata):
        if explicit_balance_gb <= 0:
            return {"applied": False}
        return {
            "applied": True,
            "mode": "absolute",
            "reserve_gb": round(explicit_balance_gb, 3),
            "reserve_ratio": 0.0,
        }

    if not is_dedicated:
        return {"applied": False}

    reserve_ratio = _resolve_lane_balance_ratio_from_libraries(libraries)
    if reserve_ratio <= 0 and explicit_balance_gb > 0:
        reference_gb = 0.0
        if selection is not None:
            reference_gb = float(
                getattr(selection, "max_target_gb", 0.0)
                or getattr(selection, "soft_target_gb", 0.0)
                or getattr(selection, "effective_max_gb", 0.0)
                or 0.0
            )
        if reference_gb <= 0:
            machine_enum = _resolve_machine_type_enum_simple(_machine_type_to_text(machine_type))
            reference_gb = _lane_capacity_for_machine(machine_enum)
        if reference_gb > 0:
            reserve_ratio = explicit_balance_gb / reference_gb

    if reserve_ratio <= 0:
        return {"applied": False}

    return {
        "applied": True,
        "mode": "ratio",
        "reserve_gb": round(explicit_balance_gb, 3) if explicit_balance_gb > 0 else 0.0,
        "reserve_ratio": min(max(float(reserve_ratio), 0.0), 0.999999),
    }


def _apply_balance_reservation_to_capacity_selection(
    selection: Any,
    libraries: List[EnhancedLibraryInfo],
    machine_type: MachineType | str,
    lane_id: str = "",
    lane_metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """按平衡文库补量/比例动态扣减 lane 容量上下限。"""
    reserve = _resolve_balance_reservation_context(
        libraries=libraries,
        machine_type=machine_type,
        lane_id=lane_id,
        lane_metadata=lane_metadata,
        selection=selection,
    )
    if not reserve.get("applied"):
        return selection

    profile = dict(getattr(selection, "profile", {}) or {})
    mode = reserve.get("mode")

    if mode == "absolute":
        reserve_gb = float(reserve.get("reserve_gb", 0.0) or 0.0)
        selection.soft_target_gb = max(PACKAGE_LANE_TARGET_GB - reserve_gb, 0.0)
        selection.min_target_gb = max(PACKAGE_LANE_TARGET_GB - reserve_gb, 0.0)
        selection.max_target_gb = max(PACKAGE_LANE_TARGET_GB - reserve_gb, 0.0)
        selection.effective_min_gb = max(PACKAGE_LANE_MIN_GB - reserve_gb, 0.0)
        selection.effective_max_gb = max(PACKAGE_LANE_MAX_GB - reserve_gb, 0.0)
    elif mode == "ratio":
        factor = max(0.0, 1.0 - float(reserve.get("reserve_ratio", 0.0) or 0.0))
        selection.soft_target_gb = max(float(selection.soft_target_gb) * factor, 0.0)
        selection.min_target_gb = max(float(selection.min_target_gb) * factor, 0.0)
        selection.max_target_gb = max(float(selection.max_target_gb) * factor, 0.0)
        selection.effective_min_gb = max(float(selection.effective_min_gb) * factor, 0.0)
        selection.effective_max_gb = max(float(selection.effective_max_gb) * factor, 0.0)
    else:
        return selection

    profile["balance_reserve_applied"] = True
    profile["balance_reserve_mode"] = mode
    profile["balance_reserve_gb"] = float(reserve.get("reserve_gb", 0.0) or 0.0)
    profile["balance_reserve_ratio"] = float(reserve.get("reserve_ratio", 0.0) or 0.0)
    selection.profile = profile
    return selection


def _resolve_lane_balance_ratio(lane: LaneAssignment) -> float:
    """按碱基不均衡分组模板解析lane平衡文库占比。"""
    return _resolve_lane_balance_ratio_from_libraries(list(getattr(lane, "libraries", []) or []))


def _is_mode_1_1_second_round_lane(lane: LaneAssignment) -> bool:
    """判断是否为1.1第二轮直出lane。"""
    metadata = getattr(lane, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    second_round_label = str(
        get_scheduling_config().get_mode_1_1_config().get("second_round_label", "1.1第二轮")
    )
    return str(metadata.get("selected_round_label", "") or "").strip() == second_round_label


def _resolve_mode_1_1_round2_last_phix_ratio(lane: LaneAssignment) -> float:
    """读取1.1第二轮lane的历史平衡文库占比。

    `wklastphix` 为历史lane的平衡文库占比（小数），业务口径按“占最终总量比例”解释。
    """
    ratios: List[float] = []
    for lib in list(getattr(lane, "libraries", []) or []):
        for attr_name in ("_last_phix_raw", "last_phix", "wklastphix"):
            value = _safe_float(getattr(lib, attr_name, None), default=None)
            if value is None:
                continue
            if 0.0 < float(value) < 1.0:
                ratios.append(float(value))
                break

    if not ratios:
        return 0.0

    resolved_ratio = max(ratios)
    if len({round(item, 6) for item in ratios}) > 1:
        logger.warning(
            "1.1第二轮lane {} 的 wklastphix 存在多个取值，按最大值使用: {} -> {:.6f}",
            getattr(lane, "lane_id", ""),
            sorted({round(item, 6) for item in ratios}),
            resolved_ratio,
        )
    return resolved_ratio


def _resolve_mode_1_1_round2_order_for_balance_library(
    lib: EnhancedLibraryInfo,
    lane: Optional[LaneAssignment] = None,
) -> Optional[float]:
    """为1.1第二轮平衡文库补量解析普通文库的下单量。"""
    contract_data = _get_lib_attr_float(lib, ["contract_data_raw", "contractdata"])
    if contract_data is None or contract_data <= 0:
        return None

    pooling_factor = None
    lane_meta = getattr(lane, "metadata", None)
    if isinstance(lane_meta, dict):
        low_output_origrecs = {
            str(item).strip()
            for item in (lane_meta.get("mode_1_1_round2_low_output_origrecs") or [])
            if str(item).strip()
        }
        origrec = str(getattr(lib, "origrec", "") or "").strip()
        if origrec and origrec in low_output_origrecs:
            pooling_factor = _safe_float(
                lane_meta.get("mode_1_1_round2_pooling_factor"),
                default=None,
            )
    if pooling_factor is not None and pooling_factor > 0:
        return round(float(contract_data) * float(pooling_factor), 6)

    add_test_output_rate = _normalize_rate_to_decimal(
        _get_lib_attr_float(lib, ["output_rate", "outputrate", "wkoutputrate"])
    )
    if add_test_output_rate is not None and add_test_output_rate > 0:
        return round(float(contract_data) / float(add_test_output_rate), 6)

    return None


def _resolve_lane_balance_data_gb(lane: LaneAssignment) -> float:
    """确定lane需要补充的平衡文库量。"""
    explicit_value = _get_lane_explicit_balance_data(lane)
    if _is_package_lane_assignment(lane):
        return round(explicit_value, 3) if explicit_value > 0 else 0.0
    if _is_mode_1_1_second_round_lane(lane):
        history_ratio = _resolve_mode_1_1_round2_last_phix_ratio(lane)
        if history_ratio <= 0:
            return 0.0
        non_balance_order = 0.0
        for lib in list(getattr(lane, "libraries", []) or []):
            if _is_ai_balance_library(lib):
                continue
            resolved_order = _resolve_mode_1_1_round2_order_for_balance_library(lib, lane)
            if resolved_order is None or resolved_order <= 0:
                continue
            non_balance_order += float(resolved_order)
        if non_balance_order <= 0:
            return 0.0
        if explicit_value > 0:
            return round(explicit_value, 3)
        denominator = max(1.0 - history_ratio, MIN_BALANCE_RATIO_DENOMINATOR)
        return round(non_balance_order * history_ratio / denominator, 3)
    if not _is_explicit_dedicated_imbalance_lane(lane):
        metadata = getattr(lane, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("is_lane_seq_10_plus_24_lane"):
            if explicit_value > 0:
                return round(explicit_value, 3)
            non_balance_data = sum(
                float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
                for lib in list(getattr(lane, "libraries", []) or [])
                if not _is_ai_balance_library(lib)
            )
            return round(
                _calculate_balance_amount_for_final_ratio(
                    non_balance_data,
                    LANE_SEQ_10_PLUS_24_BALANCE_RATIO,
                ),
                3,
            )
        return 0.0
    ratio = _resolve_lane_balance_ratio(lane)
    if ratio <= 0:
        return 0.0
    if explicit_value > 0:
        return round(explicit_value, 3)
    non_balance_data = sum(
        float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
        for lib in list(getattr(lane, "libraries", []) or [])
        if not _is_ai_balance_library(lib)
    )
    return round(_calculate_balance_amount_for_final_ratio(non_balance_data, ratio), 3)


def _get_lane_balance_templates(lane: LaneAssignment) -> List[Dict[str, Any]]:
    """按实验室+工序匹配lane可用平衡文库模板，并应用PE/phix优先级规则。"""
    dept = _get_lane_lab_name(lane)
    test_no = _get_lane_process_name(lane)
    if not dept or not test_no:
        return []

    templates = list(
        _load_balance_library_templates().get(
            (_normalize_text_for_match(dept), _normalize_text_for_match(test_no)),
            [],
        )
    )
    if not templates:
        return []

    lane_has_pe = any(
        _index_seq_contains_pe(getattr(lib, "index_seq", ""))
        for lib in list(getattr(lane, "libraries", []) or [])
        if not _is_ai_balance_library(lib)
    )
    filtered_templates: List[Dict[str, Any]] = []
    for template in templates:
        sample_id = _safe_str(template.get("wksampleid"), default="")
        if lane_has_pe and sample_id.lower() == "phix":
            continue
        filtered_templates.append(template)

    if lane_has_pe:
        return filtered_templates

    return sorted(
        filtered_templates,
        key=lambda item: (
            0 if _safe_str(item.get("wksampleid"), default="").lower() == "phix" else 1,
            int(item.get("_template_order", 0) or 0),
        ),
    )


def _build_balance_library_output_payload(
    template: Dict[str, Any],
    balance_amount_gb: float,
    aidbid: str,
    internal_origrec: str,
) -> Dict[str, Any]:
    """构建平衡文库输出行基础字段。"""
    payload = {
        "wkaidbid": aidbid,
        "wkorigrec": template.get("wkorigrec"),
        "wksid": template.get("wksid"),
        "wkpid": template.get("wkpid"),
        "wkproductline": template.get("wkproductline"),
        "lcontainerstate": template.get("lcontainerstate"),
        "wktestno": template.get("wktestno"),
        "wkqpcr": template.get("wkqpcr"),
        "wksampleid": template.get("wksampleid"),
        "wkdept": template.get("wkdept"),
        "lsjfs": template.get("lsjfs"),
        "wkindexseq": template.get("wkindexseq"),
        "wkcontractdata": round(balance_amount_gb, 3),
        "lorderdata": round(balance_amount_gb, 3),
        "origrec": internal_origrec,
        "origrec_key": internal_origrec,
        "detail_row_key": aidbid,
        BALANCE_LIBRARY_MARKER_COLUMN: True,
        "wkissplit": "",
        "wktotalcontractdata": pd.NA,
    }
    if "wktestno.1" in template:
        payload["wktestno.1"] = template.get("wktestno.1")
    return payload


def _create_balance_library_from_template(
    lane: LaneAssignment,
    template: Dict[str, Any],
    balance_amount_gb: float,
) -> EnhancedLibraryInfo:
    """按模板实例化一条真实平衡文库。"""
    aidbid = str(uuid4())
    internal_origrec = f"AI_BALANCE_{lane.lane_id}_{aidbid[:12]}"
    index_seq = _safe_str(template.get("wkindexseq"), default="")
    lib = EnhancedLibraryInfo(
        origrec=internal_origrec,
        sample_id=_safe_str(template.get("wksampleid"), default=""),
        sample_type_code="平衡文库",
        data_type="",
        customer_library="否",
        base_type=_derive_balance_base_type(index_seq),
        number_of_bases=_derive_balance_index_bases(index_seq),
        index_number=1,
        index_seq=index_seq,
        add_tests_remark="",
        product_line=_safe_str(template.get("wkproductline"), default=""),
        peak_size=0,
        eq_type=_machine_type_to_text(lane.machine_type, default="Nova X-25B"),
        contract_data_raw=round(balance_amount_gb, 3),
        test_code=None,
        test_no=_safe_str(template.get("wktestno"), default=""),
        sub_project_name="",
        create_date="",
        delivery_date="",
        lab_type="",
        data_volume_type="",
        board_number="",
    )
    lib.machine_type = lane.machine_type
    lib.sid = _safe_str(template.get("wksid"), default="")
    lib.qpcr_concentration = _safe_float(template.get("wkqpcr"), default=None)
    lib.balance_data = round(balance_amount_gb, 3)
    lib.is_add_balance = "是"
    lib.aidbid = aidbid
    lib.wkaidbid = aidbid
    lib._origrec_key = internal_origrec
    lib._source_origrec_key = internal_origrec
    lib._detail_output_key = aidbid
    lib._wkdept_raw = _safe_str(template.get("wkdept"), default="")
    lib._balance_output_payload = _build_balance_library_output_payload(
        template=template,
        balance_amount_gb=balance_amount_gb,
        aidbid=aidbid,
        internal_origrec=internal_origrec,
    )
    setattr(lib, BALANCE_LIBRARY_MARKER_COLUMN, True)
    return lib


def _find_conflicting_lane_libraries(
    lane_libraries: List[EnhancedLibraryInfo],
    candidate: EnhancedLibraryInfo,
) -> List[EnhancedLibraryInfo]:
    """返回与候选平衡文库产生最新index冲突的lane内普通文库。"""
    conflict_ids = _collect_candidate_balance_index_conflict_ids(lane_libraries, candidate)
    return [
        lib
        for lib in lane_libraries
        if getattr(lib, "origrec", "") in conflict_ids and _is_replaceable_normal_library(lib)
    ]


def _collect_candidate_balance_index_conflict_ids(
    lane_libraries: List[EnhancedLibraryInfo],
    candidate: EnhancedLibraryInfo,
) -> Set[str]:
    """收集候选平衡文库与lane内文库的latest index冲突记录ID。"""
    candidate_origrec = str(getattr(candidate, "origrec", "") or "").strip()
    if not candidate_origrec:
        return set()

    conflict_ids: Set[str] = set()
    for conflict in _validate_index_conflicts_latest(list(lane_libraries) + [candidate]):
        if conflict.record_id_1 == candidate_origrec:
            conflict_ids.add(str(conflict.record_id_2))
        elif conflict.record_id_2 == candidate_origrec:
            conflict_ids.add(str(conflict.record_id_1))
    return conflict_ids


def _validate_balance_injection_lane_state(
    validator: Any,
    lane: LaneAssignment,
    libraries: List[EnhancedLibraryInfo],
    candidate_balance_lib: EnhancedLibraryInfo,
    balance_already_in_libs: bool = False,
    skip_peak_size: bool = False,
) -> LaneValidationResult:
    """平衡文库注入时，额外补做候选平衡文库的latest index冲突校验。"""
    result = _validate_lane_state(
        validator,
        lane,
        libraries,
        balance_already_in_libs=balance_already_in_libs,
        skip_peak_size=skip_peak_size,
        skip_balance_injection_context_rules=skip_peak_size,
    )
    if not result.is_valid:
        logger.info(
            "Lane {} 候选平衡文库 {} 注入校验失败: {}",
            lane.lane_id,
            getattr(candidate_balance_lib, "sample_id", ""),
            [getattr(error, "message", str(error)) for error in list(getattr(result, "errors", []) or [])],
        )
        return result

    candidate_origrec = str(getattr(candidate_balance_lib, "origrec", "") or "").strip()
    other_libs = [
        lib
        for lib in libraries
        if str(getattr(lib, "origrec", "") or "").strip() != candidate_origrec
    ]
    conflict_ids = sorted(_collect_candidate_balance_index_conflict_ids(other_libs, candidate_balance_lib))
    if not conflict_ids:
        return result

    candidate_sample_id = str(getattr(candidate_balance_lib, "sample_id", "") or "").strip() or candidate_origrec
    return LaneValidationResult(
        lane_id=lane.lane_id,
        is_valid=False,
        errors=[
            ValidationError(
                rule_type=ValidationRuleType.INDEX_CONFLICT,
                severity=ValidationSeverity.ERROR,
                message=(
                    f"候选平衡文库 {candidate_sample_id} 与lane内文库存在latest index冲突，"
                    f"冲突记录: {', '.join(conflict_ids)}"
                ),
                affected_libraries=conflict_ids,
            )
        ],
        warnings=list(getattr(result, "warnings", []) or []),
    )


def _pick_replacement_from_pool(
    lane: LaneAssignment,
    candidate_balance_lib: EnhancedLibraryInfo,
    working_libs: List[EnhancedLibraryInfo],
    removed_libs: List[EnhancedLibraryInfo],
    unassigned_pool: List[EnhancedLibraryInfo],
    validator: Any,
) -> Optional[Tuple[List[EnhancedLibraryInfo], List[EnhancedLibraryInfo]]]:
    """优先从未分配池中选择可补入lane的普通文库。"""
    if not removed_libs or not unassigned_pool:
        return None

    removed_total = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in removed_libs)
    current_total = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in working_libs)
    candidates = sorted(
        [lib for lib in unassigned_pool if _is_replaceable_normal_library(lib)],
        key=lambda item: float(getattr(item, "contract_data_raw", 0.0) or 0.0),
        reverse=True,
    )

    for lib in candidates:
        data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
        if data <= 0 or data > removed_total + 1e-6:
            continue
        trial_libs = working_libs + [lib, candidate_balance_lib]
        result = _validate_balance_injection_lane_state(
            validator, lane, trial_libs,
            candidate_balance_lib=candidate_balance_lib,
            balance_already_in_libs=True,
            skip_peak_size=_is_explicit_dedicated_imbalance_lane(lane),
        )
        if result.is_valid:
            return trial_libs, [lib]
        # 容忍轻微不足，但保留后续lane间交换兜底
        if abs((current_total + data + float(candidate_balance_lib.contract_data_raw or 0.0)) - current_total) <= removed_total + 1e-6:
            continue
    return None


def _pick_replacement_from_other_lanes(
    current_lane: LaneAssignment,
    candidate_balance_lib: EnhancedLibraryInfo,
    working_libs: List[EnhancedLibraryInfo],
    removed_libs: List[EnhancedLibraryInfo],
    all_lanes: List[LaneAssignment],
    validator: Any,
) -> Optional[Tuple[List[EnhancedLibraryInfo], LaneAssignment, EnhancedLibraryInfo]]:
    """当未分配池无合适文库时，尝试跨lane做单文库交换。"""
    if not removed_libs:
        return None

    removed_total = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in removed_libs)
    for other_lane in all_lanes:
        if other_lane is current_lane:
            continue
        other_libs = list(getattr(other_lane, "libraries", []) or [])
        for other_lib in other_libs:
            if not _is_replaceable_normal_library(other_lib):
                continue
            data = float(getattr(other_lib, "contract_data_raw", 0.0) or 0.0)
            if data <= 0 or data > removed_total + 1e-6:
                continue
            current_trial_libs = working_libs + [other_lib, candidate_balance_lib]
            if not _validate_balance_injection_lane_state(
                validator, current_lane, current_trial_libs,
                candidate_balance_lib=candidate_balance_lib,
                balance_already_in_libs=True,
                skip_peak_size=_is_explicit_dedicated_imbalance_lane(current_lane),
            ).is_valid:
                continue
            other_trial_libs = [lib for lib in other_libs if lib is not other_lib] + removed_libs
            if _validate_lane_state(validator, other_lane, other_trial_libs).is_valid:
                return current_trial_libs, other_lane, other_lib
    return None


def _trim_lane_for_balance_capacity(
    lane: LaneAssignment,
    working_libs: List[EnhancedLibraryInfo],
    candidate_balance_lib: EnhancedLibraryInfo,
    validator: Any,
) -> Optional[Tuple[List[EnhancedLibraryInfo], List[EnhancedLibraryInfo]]]:
    """为平衡文库腾挪容量，必要时剔除部分普通文库。

    trial_libs 里已包含 candidate_balance_lib，校验时：
    - balance_already_in_libs=True：防止容量校验器二次叠加平衡文库数据量
    - skip_peak_size=True（专用不均衡 lane）：peak_size 分布是排机时既成事实，
      不应成为平衡文库注入的阻碍
    """
    is_dedicated = _is_explicit_dedicated_imbalance_lane(lane)

    def _check(libs: List[EnhancedLibraryInfo]) -> Any:
        return _validate_balance_injection_lane_state(
            validator, lane, libs,
            candidate_balance_lib=candidate_balance_lib,
            balance_already_in_libs=True,
            skip_peak_size=is_dedicated,
        )

    if _collect_candidate_balance_index_conflict_ids(working_libs, candidate_balance_lib):
        return None

    trial_libs = list(working_libs) + [candidate_balance_lib]
    if _check(trial_libs).is_valid:
        return trial_libs, []

    non_balance_libs = [
        lib for lib in sorted(
            working_libs,
            key=lambda item: float(getattr(item, "contract_data_raw", 0.0) or 0.0),
            reverse=True,
        )
        if _is_replaceable_normal_library(lib)
    ]
    trimmed_libs = list(working_libs)
    removed_libs: List[EnhancedLibraryInfo] = []
    for lib in non_balance_libs:
        trimmed_libs = [item for item in trimmed_libs if item is not lib]
        removed_libs.append(lib)
        trial_libs = trimmed_libs + [candidate_balance_lib]
        if _check(trial_libs).is_valid:
            return trial_libs, removed_libs

    return None


def _materialize_balance_library_for_lane(
    lane: LaneAssignment,
    all_lanes: List[LaneAssignment],
    unassigned_pool: List[EnhancedLibraryInfo],
    validator: Any,
) -> bool:
    """为单条lane补充真实平衡文库，必要时执行未分配补位或跨lane交换。"""
    lane_libraries = list(getattr(lane, "libraries", []) or [])
    existing_balance_libraries = [lib for lib in lane_libraries if _is_ai_balance_library(lib)]
    balance_amount = _resolve_lane_balance_data_gb(lane)
    if existing_balance_libraries:
        if not isinstance(lane.metadata, dict):
            lane.metadata = {}
        if balance_amount > 0:
            lane.metadata["wkbalancedata"] = round(balance_amount, 3)
            lane.metadata["required_balance_data_gb"] = round(balance_amount, 3)
        lane.metadata["materialized_balance_library"] = True
        return True

    if balance_amount <= 0:
        return False

    templates = _get_lane_balance_templates(lane)
    if not templates:
        logger.warning(
            "Lane {} 需要补平衡文库 {:.3f}G，但未匹配到实验室={} 工序={} 的配置模板",
            lane.lane_id,
            balance_amount,
            _get_lane_lab_name(lane) or "",
            _get_lane_process_name(lane) or "",
        )
        return False

    original_libs = list(getattr(lane, "libraries", []) or [])

    for template in templates:
        logger.info(
            "Lane {} 尝试补平衡文库模板: sample_id={}, test_no={}, index_seq={}",
            lane.lane_id,
            _safe_str(template.get("wksampleid"), default=""),
            _safe_str(template.get("wktestno"), default=""),
            _safe_str(template.get("wkindexseq"), default=""),
        )
        candidate_balance_lib = _create_balance_library_from_template(lane, template, balance_amount)
        trimmed_result = _trim_lane_for_balance_capacity(
            lane=lane,
            working_libs=original_libs,
            candidate_balance_lib=candidate_balance_lib,
            validator=validator,
        )
        if trimmed_result is not None:
            trial_libs, removed_libs = trimmed_result
            unassigned_pool.extend(removed_libs)
            lane.libraries = list(trial_libs)
            lane.total_data_gb = sum(lib.get_data_amount_gb() for lib in lane.libraries)
            lane.calculate_metrics()
            lane.metadata["wkbalancedata"] = round(balance_amount, 3)
            lane.metadata["materialized_balance_library"] = True
            logger.info(
                "Lane {} 平衡文库补充成功: sample_id={}, 数据量={:.3f}G",
                lane.lane_id,
                getattr(candidate_balance_lib, "sample_id", ""),
                balance_amount,
            )
            return True

    for template in templates:
        candidate_balance_lib = _create_balance_library_from_template(lane, template, balance_amount)
        conflicting_libs = _find_conflicting_lane_libraries(original_libs, candidate_balance_lib)
        if not conflicting_libs:
            continue
        conflicting_ids = _build_library_object_id_set(conflicting_libs)
        working_libs = [lib for lib in original_libs if id(lib) not in conflicting_ids]

        picked = _pick_replacement_from_pool(
            lane=lane,
            candidate_balance_lib=candidate_balance_lib,
            working_libs=working_libs,
            removed_libs=conflicting_libs,
            unassigned_pool=unassigned_pool,
            validator=validator,
        )
        if picked is not None:
            trial_libs, added_libs = picked
            for lib in conflicting_libs:
                unassigned_pool.append(lib)
            _remove_libraries_by_identity_in_place(unassigned_pool, added_libs)
            lane.libraries = trial_libs
            lane.total_data_gb = sum(lib.get_data_amount_gb() for lib in lane.libraries)
            lane.calculate_metrics()
            lane.metadata["wkbalancedata"] = round(balance_amount, 3)
            lane.metadata["materialized_balance_library"] = True
            logger.info(
                "Lane {} 平衡文库补充成功(未分配池替换): sample_id={}, 替换普通文库={}个",
                lane.lane_id,
                getattr(candidate_balance_lib, "sample_id", ""),
                len(conflicting_libs),
            )
            return True

        swapped = _pick_replacement_from_other_lanes(
            current_lane=lane,
            candidate_balance_lib=candidate_balance_lib,
            working_libs=working_libs,
            removed_libs=conflicting_libs,
            all_lanes=all_lanes,
            validator=validator,
        )
        if swapped is not None:
            trial_libs, other_lane, other_lib = swapped
            other_lane.libraries = [
                lib for lib in list(getattr(other_lane, "libraries", []) or []) if lib is not other_lib
            ] + conflicting_libs
            other_lane.total_data_gb = sum(lib.get_data_amount_gb() for lib in other_lane.libraries)
            other_lane.calculate_metrics()
            lane.libraries = trial_libs
            lane.total_data_gb = sum(lib.get_data_amount_gb() for lib in lane.libraries)
            lane.calculate_metrics()
            lane.metadata["wkbalancedata"] = round(balance_amount, 3)
            lane.metadata["materialized_balance_library"] = True
            logger.info(
                "Lane {} 平衡文库补充成功(跨lane交换): sample_id={}, 对端lane={}",
                lane.lane_id,
                getattr(candidate_balance_lib, "sample_id", ""),
                other_lane.lane_id,
            )
            return True

    logger.warning(
        "Lane {} 平衡文库补充失败: 需补 {:.3f}G，实验室={}，工序={}",
        lane.lane_id,
        balance_amount,
        _get_lane_lab_name(lane) or "",
        _get_lane_process_name(lane) or "",
    )
    return False


def _library_has_10_plus_24_seq_scheme(lib: EnhancedLibraryInfo) -> bool:
    """判断文库当前测序方案是否包含10+24。"""
    return any(
        "10+24" in _safe_str(value, default="")
        for value in (
            getattr(lib, "seq_scheme", None),
            getattr(lib, "wkseqscheme", None),
        )
    )


def _split_10_plus_24_libraries(
    libraries: List[EnhancedLibraryInfo],
) -> Tuple[List[EnhancedLibraryInfo], List[EnhancedLibraryInfo]]:
    """从普通池中前置剥离10+24文库。"""
    lane_seq_libraries: List[EnhancedLibraryInfo] = []
    remaining_libraries: List[EnhancedLibraryInfo] = []
    for lib in libraries:
        if _library_has_10_plus_24_seq_scheme(lib):
            lane_seq_libraries.append(lib)
        else:
            remaining_libraries.append(lib)
    return lane_seq_libraries, remaining_libraries


def _pick_10_plus_24_lane_seq_subset(
    remaining: List[EnhancedLibraryInfo],
    min_contract_data: float,
    max_contract_data: float,
    target_contract_data: float,
) -> Tuple[List[EnhancedLibraryInfo], List[EnhancedLibraryInfo]]:
    """为10+24 Lane seq选择一组落入容量窗口的文库。"""
    if not remaining:
        return [], []

    scale = 1000
    min_units = int(math.ceil(min_contract_data * scale - 1e-6))
    max_units = int(math.floor(max_contract_data * scale + 1e-6))
    target_units = int(round(target_contract_data * scale))
    if min_units > max_units:
        return [], list(remaining)

    values = [
        max(1, int(round(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) * scale)))
        for lib in remaining
    ]
    states: Dict[int, Tuple[int, int]] = {0: (-1, -1)}

    for idx, value in enumerate(values):
        if value <= 0:
            continue
        for current_sum in sorted(states.keys(), reverse=True):
            new_sum = current_sum + value
            if new_sum > max_units or new_sum in states:
                continue
            states[new_sum] = (current_sum, idx)

    candidate_sums = [item for item in states.keys() if min_units <= item <= max_units]
    if not candidate_sums:
        return [], list(remaining)

    candidate_sums.sort(key=lambda item: (abs(item - target_units), -item))
    selected_sum = candidate_sums[0]
    selected_indices: Set[int] = set()
    cursor = selected_sum
    while cursor > 0:
        previous_sum, idx = states[cursor]
        if idx < 0:
            break
        selected_indices.add(idx)
        cursor = previous_sum

    selected = [lib for idx, lib in enumerate(remaining) if idx in selected_indices]
    next_remaining = [lib for idx, lib in enumerate(remaining) if idx not in selected_indices]
    return selected, next_remaining


def _build_10_plus_24_lane_seq_lanes(
    libraries: List[EnhancedLibraryInfo],
    machine_type: MachineType = MachineType.NOVA_X_25B,
) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo]]:
    """按10+24容量规则前置生成Lane seq lane。"""
    if not libraries:
        return [], []

    remaining = sorted(
        list(libraries),
        key=lambda lib: (
            str(getattr(lib, "delivery_date", "") or getattr(lib, "wkdeliverydate", "") or ""),
            str(getattr(lib, "sample_id", "") or ""),
            -float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
        ),
    )
    lanes: List[LaneAssignment] = []
    lane_index = 1

    target_total = LANE_SEQ_10_PLUS_24_TARGET_TOTAL_GB
    min_total = target_total - LANE_SEQ_10_PLUS_24_TOLERANCE_GB
    max_total = target_total + LANE_SEQ_10_PLUS_24_TOLERANCE_GB
    min_contract_data = min_total * LANE_SEQ_10_PLUS_24_BALANCE_DENOMINATOR
    max_contract_data = max_total * LANE_SEQ_10_PLUS_24_BALANCE_DENOMINATOR
    target_contract_data = target_total * LANE_SEQ_10_PLUS_24_BALANCE_DENOMINATOR

    while remaining:
        selected, next_remaining = _pick_10_plus_24_lane_seq_subset(
            remaining=remaining,
            min_contract_data=min_contract_data,
            max_contract_data=max_contract_data,
            target_contract_data=target_contract_data,
        )
        if not selected:
            break

        selected_data = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in selected)
        balance_data = round(
            _calculate_balance_amount_for_final_ratio(
                selected_data,
                LANE_SEQ_10_PLUS_24_BALANCE_RATIO,
            ),
            3,
        )
        effective_total = selected_data + balance_data
        if effective_total < min_total - 1e-6 or effective_total > max_total + 1e-6:
            logger.warning(
                "10+24 Lane seq候选组合超出容量窗口: 合同量={:.3f}G, 平衡文库={:.3f}G, 有效总量={:.3f}G, 窗口=[{:.3f}, {:.3f}]G",
                selected_data,
                balance_data,
                effective_total,
                min_total,
                max_total,
            )
            break

        lane_id = f"{LANE_SEQ_10_PLUS_24_LANE_PREFIX}_{_machine_type_to_text(machine_type, default='Nova X-25B').replace(' ', '')}_{lane_index:03d}"
        for lib in selected:
            lib._current_seq_mode_raw = "Lane seq"
            lib._lane_sj_mode_raw = "Lane seq"
            lib.current_seq_mode = "Lane seq"
            lib.lane_sj_mode = "Lane seq"
            lib.seq_strategy = "10+24"

        lane = LaneAssignment(
            lane_id=lane_id,
            machine_id=f"M_{lane_id}",
            machine_type=machine_type,
            libraries=selected,
            total_data_gb=selected_data,
            metadata={
                "is_lane_seq_10_plus_24_lane": True,
                "selected_seq_mode": "Lane seq",
                "seq_mode": "Lane seq",
                "seq_strategy": "10+24",
                "wkbalancedata": balance_data,
                "required_balance_data_gb": balance_data,
                "additional_balance_ratio": LANE_SEQ_10_PLUS_24_BALANCE_RATIO,
                "target_total_gb": target_total,
                "capacity_tolerance_gb": LANE_SEQ_10_PLUS_24_TOLERANCE_GB,
                "dispatch_stage": "pre_10_plus_24_lane_seq",
            },
        )
        lanes.append(lane)
        logger.info(
            "10+24 Lane seq成Lane成功: lane={}, 文库数={}, 合同量={:.3f}G, 平衡文库={:.3f}G, 有效总量={:.3f}G",
            lane_id,
            len(selected),
            selected_data,
            balance_data,
            effective_total,
        )
        remaining = next_remaining
        lane_index += 1

    if remaining:
        logger.warning("10+24 Lane seq剩余{}个文库未成Lane", len(remaining))
    return lanes, remaining


def _materialize_balance_libraries_for_solution(solution: Any) -> Dict[str, int]:
    """对最终成lane结果补真实平衡文库。

    普通Lane补平衡失败仍回退未分配；碱基不均衡专Lane保留，仅保留所需平衡量
    元数据，最终是否有效以后续总体验证和最终输出为准。
    """
    from arrange_library.core.constraints.lane_validator import LaneValidator

    validator = LaneValidator(strict_mode=True)
    unassigned_pool = list(getattr(solution, "unassigned_libraries", []) or [])
    lanes = list(getattr(solution, "lane_assignments", []) or [])

    success_count = 0
    required_count = 0
    removed_lanes = 0
    preserved_dedicated_lanes = 0
    recovered_libraries = 0
    kept_lanes: List[LaneAssignment] = []

    for lane in lanes:
        required_balance_data = _resolve_lane_balance_data_gb(lane)
        if required_balance_data <= 0:
            kept_lanes.append(lane)
            continue

        required_count += 1
        if _materialize_balance_library_for_lane(
            lane=lane,
            all_lanes=lanes,
            unassigned_pool=unassigned_pool,
            validator=validator,
        ):
            success_count += 1
            kept_lanes.append(lane)
            continue

        lane_id = _safe_str(getattr(lane, "lane_id", ""), default="")
        if _is_explicit_dedicated_imbalance_lane(lane):
            if not isinstance(lane.metadata, dict):
                lane.metadata = {}
            lane.metadata["wkbalancedata"] = round(required_balance_data, 3)
            lane.metadata["required_balance_data_gb"] = round(required_balance_data, 3)
            kept_lanes.append(lane)
            preserved_dedicated_lanes += 1
            continue

        removed_lanes += 1
        recovered_libraries += len(list(getattr(lane, "libraries", []) or []))
        unassigned_pool.extend(
            lib
            for lib in list(getattr(lane, "libraries", []) or [])
            if not _is_ai_balance_library(lib)
        )

    solution.lane_assignments = kept_lanes
    solution.unassigned_libraries = unassigned_pool
    return {
        "required_lanes": required_count,
        "success_lanes": success_count,
        "removed_lanes": removed_lanes,
        "preserved_dedicated_lanes": preserved_dedicated_lanes,
        "recovered_libraries": recovered_libraries,
    }


def _extract_dedicated_10bp_lanes(
    libraries: List[EnhancedLibraryInfo],
    validator,
    machine_type: MachineType = MachineType.NOVA_X_25B,
    max_lanes: Optional[int] = None,
    index_conflict_attempts_per_lane: int = DEFAULT_INDEX_CONFLICT_ATTEMPTS,
    other_failure_attempts_per_lane: int = DEFAULT_OTHER_FAILURE_ATTEMPTS,
) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo]]:
    """优先从10bp文库中抽取纯10bp专Lane，再返回剩余待排文库。"""
    if not libraries:
        return [], []
    original_libraries = list(libraries)
    libraries = _filter_libraries_by_hard_priority(
        original_libraries,
        machine_type=machine_type,
        stage_name="dedicated_10bp",
        emit_log=True,
    )
    if not libraries:
        return [], original_libraries
    max_allowed_rank = _resolve_priority_gate_rank(
        libraries=original_libraries,
        machine_type=machine_type,
    )
    deferred_libraries = [
        lib for lib in original_libraries
        if max_allowed_rank is not None
        and _get_scattered_mix_priority_rank(lib) > max_allowed_rank
    ]

    libs_10bp, _ = _split_10bp_and_non_10bp(libraries, validator)
    if not libs_10bp:
        return [], original_libraries

    min_allowed, _ = _resolve_lane_capacity_limits(
        libraries=libs_10bp,
        machine_type=machine_type,
    )
    total_10bp_data = _total_lane_data(libs_10bp)
    theoretical_max_lanes = int(total_10bp_data // min_allowed)
    if theoretical_max_lanes <= 0:
        logger.info(
            "10bp专Lane预抽取跳过: 10bp总量{:.1f}G，不足以形成1条{}机型Lane".format(
                total_10bp_data,
                machine_type.value,
            )
        )
        return [], original_libraries

    target_lane_count = theoretical_max_lanes
    if max_lanes is not None:
        target_lane_count = min(target_lane_count, max_lanes)

    dedicated_lanes: List[LaneAssignment] = []
    remaining_10bp_pool: List[EnhancedLibraryInfo] = list(libs_10bp)
    used_library_keys: Set[str] = set()

    for _ in range(target_lane_count):
        lane, used = _attempt_build_lane_from_pool(
            pool=remaining_10bp_pool,
            validator=validator,
            machine_type=machine_type,
            lane_id_prefix="TB",
            index_conflict_attempts=index_conflict_attempts_per_lane,
            other_failure_attempts=other_failure_attempts_per_lane,
            prioritize_scattered_mix=True,
        )
        if lane is None or not used:
            break
        lane.metadata["is_pure_10bp_lane"] = True
        dedicated_lanes.append(lane)
        used_keys_current = {_get_library_identity_key(lib) for lib in used}
        used_library_keys.update(used_keys_current)
        remaining_10bp_pool = [
            lib for lib in remaining_10bp_pool
            if _get_library_identity_key(lib) not in used_keys_current
        ]

    remaining_libraries = [
        lib for lib in libraries
        if _get_library_identity_key(lib) not in used_library_keys
    ]
    remaining_libraries.extend(deferred_libraries)
    logger.info(
        "10bp专Lane预抽取完成: 新增专Lane={}，抽取10bp文库={}个/{:.1f}G，剩余待排={}个/{:.1f}G".format(
            len(dedicated_lanes),
            len(used_library_keys),
            sum(lane.total_data_gb for lane in dedicated_lanes),
            len(remaining_libraries),
            _total_lane_data(remaining_libraries),
        )
    )
    return dedicated_lanes, remaining_libraries


def _find_best_peak_size_window(
    libraries: List[EnhancedLibraryInfo],
    window_bp: int = 150,
) -> Tuple[float, float, List[EnhancedLibraryInfo]]:
    """找到数据量最大的 window_bp 范围的 Peak Size 窗口。

    Returns:
        (窗口下限peak, 窗口上限peak, 窗口内文库列表)
    """
    if not libraries:
        return 0.0, 0.0, []

    sorted_libs = sorted(
        libraries,
        key=lambda lib: float(getattr(lib, 'peak_size', 0) or 0),
    )
    best_start = 0
    best_end = 0
    best_data = 0.0

    for i in range(len(sorted_libs)):
        ps_start = float(getattr(sorted_libs[i], 'peak_size', 0) or 0)
        count = 0
        total_data = 0.0
        for j in range(i, len(sorted_libs)):
            ps_j = float(getattr(sorted_libs[j], 'peak_size', 0) or 0)
            if ps_j - ps_start <= window_bp:
                total_data += sorted_libs[j].get_data_amount_gb()
                count += 1
            else:
                break
        if total_data > best_data:
            best_data = total_data
            best_start = i
            best_end = i + count

    window_libs = sorted_libs[best_start:best_end]
    if not window_libs:
        return 0.0, 0.0, []
    ps_min = float(getattr(window_libs[0], 'peak_size', 0) or 0)
    ps_max = float(getattr(window_libs[-1], 'peak_size', 0) or 0)
    return ps_min, ps_max, window_libs


def _extract_mixed_lanes_by_peak_window(
    libraries: List[EnhancedLibraryInfo],
    validator,
    machine_type: MachineType = MachineType.NOVA_X_25B,
    max_lanes: Optional[int] = None,
    index_conflict_attempts_per_lane: int = DEFAULT_INDEX_CONFLICT_ATTEMPTS,
    other_failure_attempts_per_lane: int = DEFAULT_OTHER_FAILURE_ATTEMPTS,
) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo]]:
    """专Lane优先 + 混样排兜底：

    1. 先找最大的 Peak Size 150bp 兼容窗口
    2. 在窗口内混合 10bp + 非10bp 文库构建 Lane
    3. 使用随机重试策略通过全量验证

    由于纯非10bp专Lane受碱基不均衡占比和Peak Size约束无法成功，
    本函数混合不同Index碱基数的文库，自然满足10bp>=40%要求。
    """
    if not libraries:
        return [], []
    original_libraries = list(libraries)
    libraries = _filter_libraries_by_hard_priority(
        original_libraries,
        machine_type=machine_type,
        stage_name="mixed_peak_window",
        emit_log=True,
    )
    if not libraries:
        return [], original_libraries
    max_allowed_rank = _resolve_priority_gate_rank(
        libraries=original_libraries,
        machine_type=machine_type,
    )
    deferred_libraries = [
        lib for lib in original_libraries
        if max_allowed_rank is not None
        and _get_scattered_mix_priority_rank(lib) > max_allowed_rank
    ]

    ps_min, ps_max, window_libs = _find_best_peak_size_window(libraries)
    if not window_libs:
        return [], original_libraries

    window_total = _total_lane_data(window_libs)
    min_allowed, _ = _resolve_lane_capacity_limits(
        libraries=window_libs,
        machine_type=machine_type,
    )
    theoretical_max_lanes = int(window_total // min_allowed)
    if theoretical_max_lanes <= 0:
        logger.info(
            "混排窗口内总量{:.0f}G不足以形成Lane(最小{:.0f}G)".format(
                window_total, min_allowed,
            )
        )
        return [], original_libraries

    target_lane_count = theoretical_max_lanes
    if max_lanes is not None:
        target_lane_count = min(target_lane_count, max_lanes)

    logger.info(
        "混排Peak窗口: {:.0f}-{:.0f}bp, {}个文库/{:.0f}G, 理论最多{}条Lane".format(
            ps_min, ps_max, len(window_libs), window_total, target_lane_count,
        )
    )

    mixed_lanes: List[LaneAssignment] = []
    remaining_pool: List[EnhancedLibraryInfo] = list(window_libs)
    used_library_keys: Set[str] = set()

    for lane_serial in range(1, target_lane_count + 1):
        lane, used = _attempt_build_lane_from_pool(
            pool=remaining_pool,
            validator=validator,
            machine_type=machine_type,
            lane_id_prefix="MX",
            lane_serial=lane_serial,
            index_conflict_attempts=index_conflict_attempts_per_lane,
            other_failure_attempts=other_failure_attempts_per_lane,
            prioritize_scattered_mix=True,
        )
        if lane is None or not used:
            break
        mixed_lanes.append(lane)
        used_keys_current = {_get_library_identity_key(lib) for lib in used}
        used_library_keys.update(used_keys_current)
        remaining_pool = [
            lib for lib in remaining_pool
            if _get_library_identity_key(lib) not in used_keys_current
        ]

    remaining_libraries = [
        lib for lib in libraries
        if _get_library_identity_key(lib) not in used_library_keys
    ]
    remaining_libraries.extend(deferred_libraries)
    logger.info(
        "混排Lane预抽取完成: 新增Lane={}，使用文库={}个/{:.1f}G，剩余待排={}个/{:.1f}G".format(
            len(mixed_lanes),
            len(used_library_keys),
            sum(lane.total_data_gb for lane in mixed_lanes),
            len(remaining_libraries),
            _total_lane_data(remaining_libraries),
        )
    )
    return mixed_lanes, remaining_libraries


def _try_increase_lane_count(
    solution,
    validator,
    max_new_lanes: int = 3,
    index_conflict_attempts_per_lane: int = DEFAULT_INDEX_CONFLICT_ATTEMPTS,
    other_failure_attempts_per_lane: int = DEFAULT_OTHER_FAILURE_ATTEMPTS,
    donor_limit: int = 3,
) -> int:
    """尝试增加Lane数量（从现有Lane中匀出文库构建新Lane）"""
    lanes = solution.lane_assignments
    unassigned = solution.unassigned_libraries
    if not lanes and not unassigned:
        return 0

    machine_types = set()
    for lane in lanes:
        if lane.machine_type:
            machine_types.add(lane.machine_type)
    if not machine_types:
        machine_types.add(MachineType.NOVA_X_25B)

    added = 0
    for machine_type in machine_types:
        for _ in range(max_new_lanes - added):
            if added >= max_new_lanes:
                break
            pool: List[EnhancedLibraryInfo] = list(unassigned)
            min_allowed, _ = _resolve_lane_capacity_limits(
                libraries=pool,
                machine_type=machine_type,
            )

            if len(pool) < 5:
                donations = _collect_donations_for_pool(
                    lanes, validator, machine_type, min_allowed,
                    max_donations=40, max_per_lane=donor_limit,
                )
                for donor_lane, lib in donations:
                    donor_lane.remove_library(lib)
                    unassigned.append(lib)
                    pool.append(lib)

            new_lane, used = _attempt_build_rescue_lane_from_pool(
                pool=pool,
                validator=validator,
                machine_type=machine_type,
                lane_id_prefix="EX",
                index_conflict_attempts=index_conflict_attempts_per_lane,
                other_failure_attempts=other_failure_attempts_per_lane,
            )
            if new_lane:
                _remove_libraries_by_identity_in_place(unassigned, used)
                lanes.append(new_lane)
                added += 1
            else:
                logger.info(
                    "Lane数量提升停止: machine_type={}, 当前池首次构Lane失败，跳过同池重复尝试".format(
                        machine_type.value if isinstance(machine_type, MachineType) else machine_type
                    )
                )
                break
    return added


def _collect_donations_for_pool(
    lanes: List[LaneAssignment],
    validator,
    machine_type: MachineType,
    target_data: float,
    max_donations: int = 40,
    max_per_lane: int = 4,
) -> List[Tuple[LaneAssignment, EnhancedLibraryInfo]]:
    """从现有Lane中收集可捐赠的文库"""
    donations: List[Tuple[LaneAssignment, EnhancedLibraryInfo]] = []
    collected_data = 0.0

    for lane in lanes:
        if not lane.libraries or len(lane.libraries) <= 2:
            continue
        lane_machine_type = lane.machine_type if lane.machine_type else machine_type
        lane_min_allowed, _ = _resolve_lane_capacity_limits(
            libraries=lane.libraries,
            machine_type=lane_machine_type,
            lane_id=lane.lane_id,
            lane_metadata=lane.metadata,
        )
        per_lane = 0
        for lib in sorted(lane.libraries, key=lambda x: x.get_data_amount_gb()):
            remaining = lane.total_data_gb - lib.get_data_amount_gb()
            if remaining < lane_min_allowed:
                continue
            test_libs = [l for l in lane.libraries if l is not lib]
            metadata = _build_lane_metadata_for_validator(lane.lane_id, lane.metadata, libraries=test_libs)
            result = _validate_lane_with_latest_index(
                validator=validator,
                libraries=test_libs,
                lane_id=lane.lane_id,
                machine_type=lane.machine_type.value if lane.machine_type else "Nova X-25B",
                metadata=metadata,
            )
            if result.is_valid:
                donations.append((lane, lib))
                collected_data += lib.get_data_amount_gb()
                per_lane += 1
                if per_lane >= max_per_lane:
                    break
        if len(donations) >= max_donations or collected_data >= target_data * 1.5:
            break
    return donations


def _enforce_global_priority_hard_constraint(
    solution: Any,
    validator: Any,
) -> Dict[str, int]:
    """历史兼容接口：高优文库最终收口已停用。"""
    return {"adjusted_lanes": 0, "removed_lanes": 0, "deferred_libraries": 0}


def _should_skip_final_priority_gate_for_hybrid_mode_1_1(solution: Any) -> bool:
    """混合流程下保留后续阶段已落地的Lane，避免被最终优先级收口再次裁掉。"""
    has_priority_preconsume_lane = False
    has_non_preconsume_lane = False

    for lane in list(getattr(solution, "lane_assignments", []) or []):
        metadata = dict(getattr(lane, "metadata", {}) or {})
        dispatch_stage = str(metadata.get("dispatch_stage", "") or "")
        if dispatch_stage == "priority_preconsume_36t":
            has_priority_preconsume_lane = True
        else:
            has_non_preconsume_lane = True

        if has_priority_preconsume_lane and has_non_preconsume_lane:
            return True

    return False


def try_multi_lib_swap_rebalance(
    solution,
    validator,
    max_new_lanes: int = 2,
    max_donations: int = 40,
    index_conflict_max_trials: int = DEFAULT_INDEX_CONFLICT_ATTEMPTS,
    other_failure_max_trials: int = DEFAULT_OTHER_FAILURE_ATTEMPTS,
    max_per_lane: int = 4,
) -> Dict[str, int]:
    """跨Lane多文库交换再平衡"""
    lanes = solution.lane_assignments
    unassigned = solution.unassigned_libraries
    if not unassigned:
        return {"new_lanes": 0, "remaining_unassigned": len(unassigned)}

    machine_types = set()
    for lane in lanes:
        if lane.machine_type:
            machine_types.add(lane.machine_type)
    if not machine_types:
        machine_types.add(MachineType.NOVA_X_25B)

    new_lanes_count = 0
    for machine_type in machine_types:
        for _ in range(max_new_lanes - new_lanes_count):
            if new_lanes_count >= max_new_lanes:
                break
            pool = list(unassigned)
            min_allowed, _ = _resolve_lane_capacity_limits(
                libraries=pool,
                machine_type=machine_type,
            )
            pool_data = sum(lib.get_data_amount_gb() for lib in pool)

            if pool_data < min_allowed:
                donations = _collect_donations_for_pool(
                    lanes, validator, machine_type, min_allowed - pool_data,
                    max_donations=max_donations, max_per_lane=max_per_lane,
                )
                for donor_lane, lib in donations:
                    donor_lane.remove_library(lib)
                    unassigned.append(lib)
                    pool.append(lib)
                    pool_data += lib.get_data_amount_gb()
                    if pool_data >= min_allowed:
                        break

            new_lane, used = _attempt_build_rescue_lane_from_pool(
                pool=pool,
                validator=validator,
                machine_type=machine_type,
                lane_id_prefix="RB",
                index_conflict_attempts=index_conflict_max_trials,
                other_failure_attempts=other_failure_max_trials,
            )
            if new_lane:
                _remove_libraries_by_identity_in_place(unassigned, used)
                lanes.append(new_lane)
                new_lanes_count += 1
            else:
                logger.info(
                    "跨Lane多文库交换停止: machine_type={}, 当前池首次构Lane失败，跳过同池重复尝试".format(
                        machine_type.value if isinstance(machine_type, MachineType) else machine_type
                    )
                )
                break

    return {"new_lanes": new_lanes_count, "remaining_unassigned": len(unassigned)}


def try_targeted_imbalance_upgrade(
    solution,
    validator,
    *,
    max_successful_swaps: int = 6,
    max_candidate_libraries: int = 24,
    max_lanes_per_candidate: int = 6,
    max_donors_per_lane: int = 10,
) -> Dict[str, float]:
    """在不减少Lane数的前提下，局部替换已成Lane以提升不均衡消耗。"""
    lanes = list(getattr(solution, "lane_assignments", []) or [])
    unassigned = list(getattr(solution, "unassigned_libraries", []) or [])
    if not lanes or not unassigned or max_successful_swaps <= 0:
        return {
            "successful_swaps": 0,
            "changed_lanes": 0,
            "consumed_imbalance_gb": 0.0,
            "remaining_unassigned": len(unassigned),
        }

    candidate_lanes = [
        lane for lane in lanes
        if not _is_package_lane_assignment(lane)
        and str(getattr(lane, "lane_id", "") or "").startswith(("RM_", "RB_", "EX_", "MX_"))
    ]
    if not candidate_lanes:
        return {
            "successful_swaps": 0,
            "changed_lanes": 0,
            "consumed_imbalance_gb": 0.0,
            "remaining_unassigned": len(unassigned),
        }

    imbalance_candidates = [
        lib for lib in unassigned
        if _is_imbalance_library_candidate(lib)
    ]
    if not imbalance_candidates:
        return {
            "successful_swaps": 0,
            "changed_lanes": 0,
            "consumed_imbalance_gb": 0.0,
            "remaining_unassigned": len(unassigned),
        }

    def _lane_upgrade_sort_key(
        lane: LaneAssignment,
        candidate: EnhancedLibraryInfo,
    ) -> Tuple[float, float, int]:
        lane_libraries = list(lane.libraries or [])
        lane_summary = _summarize_lane_imbalance(lane_libraries)
        _, imbalance_data, _ = lane_summary
        candidate_data = float(getattr(candidate, "contract_data_raw", 0.0) or 0.0)
        special_limit = _resolve_special_library_data_limit(
            lane.machine_type or MachineType.NOVA_X_25B
        )
        headroom = max(special_limit - imbalance_data, 0.0)
        projected_ratio = _project_lane_imbalance_ratio(
            lane_libraries,
            candidate,
            lane_summary=lane_summary,
            candidate_is_imbalance=_is_imbalance_library_candidate(candidate),
        )
        return (
            0.0 if headroom + SPECIAL_LIBRARY_LIMIT_EPSILON >= candidate_data else 1.0,
            _imbalance_target_distance(projected_ratio),
            len(lane.libraries or []),
        )

    successful_swaps = 0
    changed_lanes: Set[str] = set()
    consumed_imbalance_gb = 0.0
    prioritized_candidates = sorted(
        imbalance_candidates,
        key=lambda lib: (
            _get_scattered_mix_priority_rank(lib),
            -float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
            _get_scattered_mix_delete_date_sort_value(lib),
            str(getattr(lib, "origrec", "") or ""),
        ),
    )[:max_candidate_libraries]

    for candidate in prioritized_candidates:
        if successful_swaps >= max_successful_swaps:
            break
        if candidate not in unassigned:
            continue

        candidate_rank = _get_scattered_mix_priority_rank(candidate)
        candidate_data = float(getattr(candidate, "contract_data_raw", 0.0) or 0.0)
        ordered_lanes = sorted(
            candidate_lanes,
            key=lambda lane: _lane_upgrade_sort_key(lane, candidate),
        )
        upgraded = False

        for lane in ordered_lanes[:max_lanes_per_candidate]:
            lane_libraries = list(lane.libraries or [])
            if not lane_libraries:
                continue
            donor_candidates = [
                lib for lib in lane_libraries
                if not _is_imbalance_library_candidate(lib)
                and _get_scattered_mix_priority_rank(lib) >= candidate_rank
            ]
            if not donor_candidates:
                continue
            donor_candidates = sorted(
                donor_candidates,
                key=lambda lib: (
                    _get_scattered_mix_priority_rank(lib),
                    abs(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) - candidate_data),
                    float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
                    str(getattr(lib, "origrec", "") or ""),
                ),
            )[:max_donors_per_lane]

            for donor in donor_candidates:
                trial_libraries = [lib for lib in lane_libraries if lib is not donor] + [candidate]
                result = _validate_lane_state(validator, lane, trial_libraries)
                if not result.is_valid:
                    continue

                lane.libraries = trial_libraries
                lane.total_data_gb = _total_lane_data(trial_libraries)
                lane.calculate_metrics()
                unassigned.remove(candidate)
                unassigned.append(donor)
                successful_swaps += 1
                changed_lanes.add(str(lane.lane_id))
                consumed_imbalance_gb += candidate_data
                upgraded = True
                break
            if upgraded:
                break

    solution.unassigned_libraries = unassigned
    return {
        "successful_swaps": successful_swaps,
        "changed_lanes": len(changed_lanes),
        "consumed_imbalance_gb": round(consumed_imbalance_gb, 3),
        "remaining_unassigned": len(unassigned),
    }


def _get_residual_regroup_cluster_key(lib: EnhancedLibraryInfo) -> str:
    """提取剩余文库重组搜索使用的聚簇键。"""
    sample_type = (
        getattr(lib, "sample_type_code", "")
        or getattr(lib, "sampletype", "")
        or getattr(lib, "data_type", "")
        or getattr(lib, "lab_type", "")
        or ""
    )
    return _normalize_text_for_match(sample_type)


def _normalize_residual_major_cluster_project(lib: EnhancedLibraryInfo) -> str:
    """提取未分配大簇 regroup 使用的项目名。"""
    return _normalize_text_for_match(
        _safe_library_text(lib, "sub_project_name", "wksubprojectname")
    ) or "EMPTY"


def _build_residual_major_cluster_key(lib: EnhancedLibraryInfo) -> str:
    """为未分配大簇专项 regroup 构建稳定聚类键。"""
    project_name = _normalize_residual_major_cluster_project(lib)
    sample_type = _normalize_text_for_match(
        _safe_library_text(lib, "sample_type_code", "wksampletype")
    ) or "EMPTY"
    data_type = _normalize_text_for_match(
        _safe_library_text(lib, "data_type", "wkdatatype")
    ) or "EMPTY"
    task_group = _normalize_text_for_match(
        _safe_library_text(lib, "task_group_name", "wktaskgroupname")
    ) or "EMPTY"
    return "|".join([project_name, sample_type, data_type, task_group])


def _is_major_residual_cluster_candidate(cluster_pool: List[EnhancedLibraryInfo]) -> bool:
    """判断未分配簇是否值得优先做专项 regroup。"""
    if not cluster_pool:
        return False
    total_gb = _total_lane_data(cluster_pool)
    if len(cluster_pool) >= 24 and total_gb >= 300.0:
        return True
    if len(cluster_pool) >= 12 and total_gb >= 900.0:
        return True
    if total_gb >= 2000.0:
        return True
    return False


def _rescue_remaining_lanes_by_layered_regroup_search(
    solution,
    validator,
    *,
    max_priority_cluster_lanes_per_machine: int = 8,
    max_mixed_rescue_lanes_per_machine: int = 8,
    max_normal_cluster_lanes_per_machine: int = 12,
    index_conflict_attempts_per_lane: int = DEFAULT_INDEX_CONFLICT_ATTEMPTS * 4,
    other_failure_attempts_per_lane: int = DEFAULT_OTHER_FAILURE_ATTEMPTS * 4,
    skip_split_rule_libraries: bool = False,
) -> Dict[str, int]:
    """对剩余文库执行“专lane -> 混排lane -> 普通lane”分层重组搜索。

    目标：
    1. 剩余临检/YC/SJ先尝试按同类聚簇专Lane；
    2. 若仍有高优先级尾货，再允许其主导混排Lane，普通文库仅按门禁补位；
    3. 最后再对纯普通文库做聚簇补Lane。

    说明：
    - 为避免重新引入拆分家族半成Lane问题，这里跳过拆分文库，仅处理非拆分尾货。
    - 所有新增Lane仍走现有严格校验与优先级门禁逻辑。
    """
    unassigned = list(getattr(solution, "unassigned_libraries", []) or [])
    if not unassigned:
        return {
            "new_lanes": 0,
            "major_cluster_lanes": 0,
            "priority_cluster_lanes": 0,
            "mixed_rescue_lanes": 0,
            "normal_cluster_lanes": 0,
            "remaining_unassigned": 0,
            "skipped_split_libraries": 0,
        }

    serials: Dict[Tuple[str, str], int] = {}

    def _next_lane_serial(prefix: str, machine_type: MachineType) -> int:
        key = (prefix, machine_type.value)
        serials[key] = serials.get(key, 0) + 1
        return serials[key]

    priority_cluster_lanes = 0
    mixed_rescue_lanes = 0
    normal_cluster_lanes = 0
    major_cluster_lanes = 0
    skipped_split_libraries = 0
    new_lanes: List[LaneAssignment] = []
    lane_validation_cache: Dict[
        Tuple[str, Tuple[Tuple[str, str], ...], Tuple[str, ...]],
        Any,
    ] = {}

    machine_type = MachineType.NOVA_X_25B
    remaining_pool: List[EnhancedLibraryInfo] = []
    passthrough: List[EnhancedLibraryInfo] = []
    for lib in unassigned:
        should_hold_split_rule_library = (
            _is_split_library(lib)
            or _is_split_rule_original_blocked_from_1_1(lib)
            or (
                skip_split_rule_libraries
                and (not _is_split_library(lib))
                and _is_split_rule_original_allowed_to_split_in_36t(lib)
            )
        )
        if should_hold_split_rule_library:
            skipped_split_libraries += 1
            passthrough.append(lib)
            continue
        remaining_pool.append(lib)

    if remaining_pool:
        machine_pool = list(remaining_pool)
        stage_priority_cluster_lanes = 0
        stage_mixed_rescue_lanes = 0
        stage_normal_cluster_lanes = 0
        stage_major_cluster_lanes = 0

        # Stage 0: 对同项目/同类型且总量大的尾货簇做专项 regroup。
        major_clusters: Dict[str, List[EnhancedLibraryInfo]] = {}
        for lib in machine_pool:
            cluster_key = _build_residual_major_cluster_key(lib)
            if not cluster_key:
                continue
            major_clusters.setdefault(cluster_key, []).append(lib)

        for _, cluster_pool in sorted(
            major_clusters.items(),
            key=lambda item: (_total_lane_data(item[1]), len(item[1])),
            reverse=True,
        ):
            if stage_major_cluster_lanes >= max(1, max_normal_cluster_lanes_per_machine // 2):
                break
            machine_pool_ids = _build_library_object_id_set(machine_pool)
            active_cluster = [lib for lib in cluster_pool if id(lib) in machine_pool_ids]
            if not active_cluster or not _is_major_residual_cluster_candidate(active_cluster):
                continue
            feasible, reason = _quick_check_pool_feasibility(
                pool=active_cluster,
                machine_type=machine_type,
                lane_metadata=None,
                stage_label="MAJOR_CLUSTER_RESCUE",
            )
            if not feasible:
                logger.info(
                    "未分配大簇专项regroup跳过: machine={}, count={}, data={:.1f}G, reason={}".format(
                        machine_type.value,
                        len(active_cluster),
                        _total_lane_data(active_cluster),
                        reason,
                    )
                )
                continue

            while active_cluster and stage_major_cluster_lanes < max(1, max_normal_cluster_lanes_per_machine // 2):
                lane, used = _attempt_build_rescue_lane_from_pool(
                    pool=active_cluster,
                    validator=validator,
                    machine_type=machine_type,
                    lane_id_prefix="MG",
                    lane_serial=_next_lane_serial("MG", machine_type),
                    index_conflict_attempts=index_conflict_attempts_per_lane,
                    other_failure_attempts=other_failure_attempts_per_lane,
                    lane_validation_cache=lane_validation_cache,
                )
                if not lane:
                    break
                new_lanes.append(lane)
                stage_major_cluster_lanes += 1
                major_cluster_lanes += 1
                used_ids = {id(lib) for lib in used}
                machine_pool = [lib for lib in machine_pool if id(lib) not in used_ids]
                active_cluster = [lib for lib in active_cluster if id(lib) not in used_ids]

        # Stage 1: 高优先级尾货优先做专Lane（跨机型，只按文库属性聚簇）。
        priority_clusters: Dict[str, List[EnhancedLibraryInfo]] = {}
        for lib in machine_pool:
            if _get_scattered_mix_priority_rank(lib) >= 2:
                continue
            cluster_key = _get_residual_regroup_cluster_key(lib)
            if not cluster_key:
                continue
            priority_clusters.setdefault(cluster_key, []).append(lib)

        for _, cluster_pool in sorted(
            priority_clusters.items(),
            key=lambda item: sum(lib.get_data_amount_gb() for lib in item[1]),
            reverse=True,
        ):
            if stage_priority_cluster_lanes >= max_priority_cluster_lanes_per_machine:
                break
            machine_pool_ids = _build_library_object_id_set(machine_pool)
            active_cluster = [lib for lib in cluster_pool if id(lib) in machine_pool_ids]
            if not active_cluster:
                continue
            min_allowed, _ = _resolve_lane_capacity_limits(active_cluster, machine_type)
            if sum(lib.get_data_amount_gb() for lib in active_cluster) + 1e-6 < min_allowed:
                continue

            while active_cluster and stage_priority_cluster_lanes < max_priority_cluster_lanes_per_machine:
                lane, used = _attempt_build_rescue_lane_from_pool(
                    pool=active_cluster,
                    validator=validator,
                    machine_type=machine_type,
                    lane_id_prefix="PG",
                    lane_serial=_next_lane_serial("PG", machine_type),
                    index_conflict_attempts=index_conflict_attempts_per_lane,
                    other_failure_attempts=other_failure_attempts_per_lane,
                    lane_validation_cache=lane_validation_cache,
                )
                if not lane:
                    break
                new_lanes.append(lane)
                stage_priority_cluster_lanes += 1
                priority_cluster_lanes += 1
                used_ids = {id(lib) for lib in used}
                machine_pool = [lib for lib in machine_pool if id(lib) not in used_ids]
                active_cluster = [lib for lib in active_cluster if id(lib) not in used_ids]

        # Stage 2: 仍有剩余高优先级时，允许高优先级主导混排，普通文库按门禁补位。
        while stage_mixed_rescue_lanes < max_mixed_rescue_lanes_per_machine:
            if not machine_pool:
                break
            current_top_rank = _get_current_hard_priority_rank(machine_pool)
            if current_top_rank is None:
                break
            lane, used = _attempt_build_rescue_lane_from_pool(
                pool=machine_pool,
                validator=validator,
                machine_type=machine_type,
                lane_id_prefix="RM",
                lane_serial=_next_lane_serial("RM", machine_type),
                index_conflict_attempts=index_conflict_attempts_per_lane,
                other_failure_attempts=other_failure_attempts_per_lane,
                lane_validation_cache=lane_validation_cache,
            )
            if not lane:
                break
            lane_top_rank = _get_current_hard_priority_rank(list(lane.libraries or []))
            if lane_top_rank is None:
                break
            if lane_top_rank > current_top_rank:
                break
            new_lanes.append(lane)
            stage_mixed_rescue_lanes += 1
            mixed_rescue_lanes += 1
            used_ids = {id(lib) for lib in used}
            machine_pool = [lib for lib in machine_pool if id(lib) not in used_ids]

        # Stage 2.5: 剩余普通尾货先尝试全局1.1 regroup，失败后再落到3.6T普通尾货Lane。
        global_mode_1_1_pool = [
            lib for lib in machine_pool
            if not _is_forbidden_in_mode_1_1_by_secondary_36t_policy(lib)
            and (
                _get_scattered_mix_priority_rank(lib) == 2
                or (
                    (not _is_split_library(lib))
                    and _should_library_split_in_3_6t(lib)
                    and _is_split_rule_original_allowed_in_1_1(lib)
                )
            )
        ]
        while global_mode_1_1_pool and stage_normal_cluster_lanes < max_normal_cluster_lanes_per_machine:
            lane, used = _try_build_global_mode_1_1_rescue_lane_from_pool(
                pool=global_mode_1_1_pool,
                validator=validator,
                machine_type=machine_type,
                lane_serial=_next_lane_serial("GL", machine_type),
                lane_validation_cache=lane_validation_cache,
            )
            if not lane or not used:
                break
            new_lanes.append(lane)
            stage_normal_cluster_lanes += 1
            normal_cluster_lanes += 1
            used_ids = {id(lib) for lib in used}
            machine_pool = [lib for lib in machine_pool if id(lib) not in used_ids]
            global_mode_1_1_pool = [lib for lib in global_mode_1_1_pool if id(lib) not in used_ids]
            logger.info(
                "终态普通尾货全局1.1 regroup成功: lane={}, 机型={}, 文库数={}, 数据量={:.1f}G".format(
                    lane.lane_id,
                    machine_type.value,
                    len(used),
                    float(getattr(lane, "total_data_gb", 0.0) or 0.0),
                )
            )

        # Stage 3: 对剩余普通尾货做同类聚簇补Lane。
        normal_clusters: Dict[str, List[EnhancedLibraryInfo]] = {}
        for lib in machine_pool:
            if _get_scattered_mix_priority_rank(lib) != 2:
                continue
            cluster_key = _get_residual_regroup_cluster_key(lib)
            if not cluster_key:
                continue
            normal_clusters.setdefault(cluster_key, []).append(lib)

        for _, cluster_pool in sorted(
            normal_clusters.items(),
            key=lambda item: sum(lib.get_data_amount_gb() for lib in item[1]),
            reverse=True,
        ):
            if stage_normal_cluster_lanes >= max_normal_cluster_lanes_per_machine:
                break
            machine_pool_ids = _build_library_object_id_set(machine_pool)
            active_cluster = [lib for lib in cluster_pool if id(lib) in machine_pool_ids]
            if not active_cluster:
                continue
            min_allowed, _ = _resolve_lane_capacity_limits(active_cluster, machine_type)
            if sum(lib.get_data_amount_gb() for lib in active_cluster) + 1e-6 < min_allowed:
                continue

            while active_cluster and stage_normal_cluster_lanes < max_normal_cluster_lanes_per_machine:
                lane, used = _attempt_build_rescue_lane_from_pool(
                    pool=active_cluster,
                    validator=validator,
                    machine_type=machine_type,
                    lane_id_prefix="OG",
                    lane_serial=_next_lane_serial("OG", machine_type),
                    index_conflict_attempts=index_conflict_attempts_per_lane,
                    other_failure_attempts=other_failure_attempts_per_lane,
                    lane_validation_cache=lane_validation_cache,
                )
                if not lane:
                    break
                new_lanes.append(lane)
                stage_normal_cluster_lanes += 1
                normal_cluster_lanes += 1
                used_ids = {id(lib) for lib in used}
                machine_pool = [lib for lib in machine_pool if id(lib) not in used_ids]
                active_cluster = [lib for lib in active_cluster if id(lib) not in used_ids]

        remaining_pool = machine_pool

    if new_lanes:
        solution.lane_assignments.extend(new_lanes)

    remaining_ids = {id(lib) for lib in passthrough}
    remaining_ids.update(id(lib) for lib in remaining_pool)
    final_unassigned = [lib for lib in unassigned if id(lib) in remaining_ids]
    solution.unassigned_libraries = final_unassigned

    return {
        "new_lanes": len(new_lanes),
        "major_cluster_lanes": major_cluster_lanes,
        "priority_cluster_lanes": priority_cluster_lanes,
        "mixed_rescue_lanes": mixed_rescue_lanes,
        "normal_cluster_lanes": normal_cluster_lanes,
        "remaining_unassigned": len(final_unassigned),
        "skipped_split_libraries": skipped_split_libraries,
    }


def _try_build_global_mode_1_1_rescue_lane_from_pool(
    *,
    pool: List[EnhancedLibraryInfo],
    validator: Any,
    machine_type: MachineType,
    lane_serial: int,
    lane_validation_cache: Optional[
        Dict[Tuple[str, Tuple[Tuple[str, str], ...], Tuple[str, ...]], Any]
    ] = None,
) -> Tuple[Optional[LaneAssignment], List[EnhancedLibraryInfo]]:
    lane_metadata = {
        "selected_seq_mode": "1.1",
        "seq_mode": "1.1",
        "lcxms": "1.1",
        "dispatch_stage": "terminal_global_mode_1_1_rescue",
    }
    lane, used = _attempt_build_rescue_lane_from_pool(
        pool=pool,
        validator=validator,
        machine_type=machine_type,
        lane_id_prefix="GL",
        lane_serial=lane_serial,
        extra_metadata=lane_metadata,
        lane_validation_cache=lane_validation_cache,
    )
    if (not lane or not used) and pool:
        lane, used, _ = _attempt_build_terminal_dedicated_lane_from_group(
            pool=pool,
            validator=validator,
            machine_type=machine_type,
            lane_id_prefix="GL",
            extra_metadata=lane_metadata,
            max_candidates=240,
        )
    if lane and used:
        for lib in list(getattr(lane, "libraries", []) or []):
            lib._current_seq_mode_raw = "1.1"
            lib.selected_seq_mode = "1.1"
            lib.current_seq_mode = "1.1"
            lib.lcxms = "1.1"
    return lane, used


def _try_convert_plain_36t_tail_lanes_to_mode_1_1(
    solution,
    validator: Any,
) -> Dict[str, int]:
    """将纯普通3.6T尾货Lane整体回收，尝试重组成1.1 Lane。

    只做全量成功替换：同一聚类里的候选3.6T Lane如果不能全部被合法1.1
    Lane消耗，则保持原3.6T结果，避免产生半残Lane。
    """
    lane_assignments = list(getattr(solution, "lane_assignments", []) or [])
    if not lane_assignments:
        return {"converted_lanes": 0, "new_lanes": 0, "used_libraries": 0}

    def _is_plain_convertible_36t_lane(lane: LaneAssignment) -> bool:
        lane_id = _safe_str(getattr(lane, "lane_id", ""), default="")
        if _is_package_lane_assignment(lane):
            return False
        if not _is_3_6t_new_lane_context(lane, list(getattr(lane, "libraries", []) or [])):
            return False
        if lane_id.startswith(("DL_", "PKG_", "LS_")):
            return False
        libraries = _get_non_balance_libraries(list(getattr(lane, "libraries", []) or []))
        if not libraries:
            return False
        if any(_is_priority_library_for_36t_policy(lib) for lib in libraries):
            return False
        if any(bool(getattr(lib, "is_base_imbalance", lambda: False)()) for lib in libraries):
            return False
        if any(_get_scattered_mix_priority_rank(lib) != 2 for lib in libraries):
            return False
        if any(_is_forbidden_in_mode_1_1_by_secondary_36t_policy(lib) for lib in libraries):
            return False
        if any(_is_split_library(lib) or _is_split_rule_original_blocked_from_1_1(lib) for lib in libraries):
            return False
        return True

    candidate_lanes = [
        lane for lane in lane_assignments
        if _is_plain_convertible_36t_lane(lane)
    ]
    if len(candidate_lanes) < 2:
        return {"converted_lanes": 0, "new_lanes": 0, "used_libraries": 0}

    clusters: Dict[Tuple[str, str], List[LaneAssignment]] = {}
    for lane in candidate_lanes:
        libraries = _get_non_balance_libraries(list(getattr(lane, "libraries", []) or []))
        cluster_key = _build_residual_major_cluster_key(libraries[0])
        if not cluster_key:
            cluster_key = _get_residual_regroup_cluster_key(libraries[0])
        if not cluster_key:
            continue
        machine_type = getattr(lane, "machine_type", MachineType.NOVA_X_25B)
        machine_value = getattr(machine_type, "value", str(machine_type))
        clusters.setdefault((machine_value, cluster_key), []).append(lane)

    if not clusters:
        return {"converted_lanes": 0, "new_lanes": 0, "used_libraries": 0}

    existing_gl_serials: List[int] = []
    for lane in lane_assignments:
        lane_id = _safe_str(getattr(lane, "lane_id", ""), default="")
        match = re.search(r"GL_[^_]+_(\d+)$", lane_id)
        if match:
            try:
                existing_gl_serials.append(int(match.group(1)))
            except ValueError:
                pass
    next_serial = max(existing_gl_serials or [0]) + 1

    committed_source_lane_ids: Set[int] = set()
    committed_new_lanes: List[LaneAssignment] = []
    committed_used_library_ids: Set[int] = set()

    for (_, _), source_lanes in sorted(
        clusters.items(),
        key=lambda item: _total_lane_data(
            [
                lib
                for lane in item[1]
                for lib in _get_non_balance_libraries(list(getattr(lane, "libraries", []) or []))
            ]
        ),
        reverse=True,
    ):
        if len(source_lanes) < 2:
            continue
        source_libraries = [
            lib
            for lane in source_lanes
            for lib in _get_non_balance_libraries(list(getattr(lane, "libraries", []) or []))
        ]
        source_total = _total_lane_data(source_libraries)
        if source_total <= 0:
            continue

        machine_type = getattr(source_lanes[0], "machine_type", MachineType.NOVA_X_25B)
        min_allowed, _ = _resolve_lane_capacity_limits(
            source_libraries,
            machine_type,
            lane_metadata={"selected_seq_mode": "1.1", "lcxms": "1.1"},
        )
        if source_total + 1e-6 < min_allowed:
            continue

        working_pool = list(source_libraries)
        candidate_new_lanes: List[LaneAssignment] = []
        candidate_used: List[EnhancedLibraryInfo] = []
        while working_pool:
            lane, used = _try_build_global_mode_1_1_rescue_lane_from_pool(
                pool=working_pool,
                validator=validator,
                machine_type=machine_type,
                lane_serial=next_serial,
            )
            if not lane or not used:
                break
            next_serial += 1
            candidate_new_lanes.append(lane)
            candidate_used.extend(used)
            used_ids = {id(lib) for lib in used}
            working_pool = [lib for lib in working_pool if id(lib) not in used_ids]

        if working_pool:
            continue
        source_ids = {id(lib) for lib in source_libraries}
        used_ids = {id(lib) for lib in candidate_used}
        if source_ids != used_ids:
            continue

        committed_source_lane_ids.update(id(lane) for lane in source_lanes)
        committed_new_lanes.extend(candidate_new_lanes)
        committed_used_library_ids.update(used_ids)

    if not committed_new_lanes:
        return {"converted_lanes": 0, "new_lanes": 0, "used_libraries": 0}

    solution.lane_assignments = [
        lane for lane in lane_assignments
        if id(lane) not in committed_source_lane_ids
    ] + committed_new_lanes
    return {
        "converted_lanes": len(committed_source_lane_ids),
        "new_lanes": len(committed_new_lanes),
        "used_libraries": len(committed_used_library_ids),
    }


def _consume_small_split_rule_originals_as_mode_1_1_lanes(
    *,
    pool: List[EnhancedLibraryInfo],
    validator: Any,
    max_lanes: int = 8,
    stage_label: str = "小拆分原始文库1.1 regroup",
) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo], Dict[str, int]]:
    """命中3.6拆分规则但<=500G的原始文库，在回3.6前最后尝试以原始形态排1.1。"""
    remaining = list(pool or [])
    lanes: List[LaneAssignment] = []
    used_total = 0
    serial = 1

    while len(lanes) < max_lanes:
        candidates = [
            lib for lib in remaining
            if (not _is_split_library(lib))
            and not _is_forbidden_in_mode_1_1_by_secondary_36t_policy(lib)
            and _should_library_split_in_3_6t(lib)
            and _is_split_rule_original_allowed_in_1_1(lib)
        ]
        if not candidates:
            break
        total_data = _total_lane_data(candidates)
        if total_data + 1e-6 < 2095.0:
            break
        lane, used = _try_build_global_mode_1_1_rescue_lane_from_pool(
            pool=candidates,
            validator=validator,
            machine_type=MachineType.NOVA_X_25B,
            lane_serial=serial,
        )
        if not lane or not used:
            break
        if not isinstance(lane.metadata, dict):
            lane.metadata = {}
        lane.metadata["dispatch_stage"] = "small_split_rule_original_mode_1_1_regroup"
        lane.metadata["selected_round_label"] = stage_label
        lanes.append(lane)
        used_ids = {id(lib) for lib in used}
        used_total += len(used_ids)
        remaining = [lib for lib in remaining if id(lib) not in used_ids]
        serial += 1

    return (
        lanes,
        remaining,
        {
            "new_lanes": len(lanes),
            "used_libraries": used_total,
            "remaining_libraries": len(remaining),
        },
    )


def _consume_tail_libraries_as_mode_1_1_lanes(
    *,
    pool: List[EnhancedLibraryInfo],
    validator: Any,
    max_lanes: int = 8,
    stage_label: str = "尾货1.1普通Lane二次抽取",
) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo], Dict[str, int]]:
    """尾货进入普通3.6T前，再尝试抽取可进1.1的普通Lane。"""
    remaining = list(pool or [])
    lanes: List[LaneAssignment] = []
    used_total = 0
    serial = 1

    while len(lanes) < max_lanes:
        candidates = [
            lib for lib in remaining
            if (not _is_split_library(lib))
            and not _is_split_rule_original_blocked_from_1_1(lib)
            and not _is_forbidden_in_mode_1_1_by_secondary_36t_policy(lib)
        ]
        if not candidates:
            break
        total_data = _total_lane_data(candidates)
        if total_data + 1e-6 < 2095.0:
            break
        lane, used = _try_build_global_mode_1_1_rescue_lane_from_pool(
            pool=candidates,
            validator=validator,
            machine_type=MachineType.NOVA_X_25B,
            lane_serial=serial,
        )
        if not lane or not used:
            break
        if not isinstance(lane.metadata, dict):
            lane.metadata = {}
        lane.metadata["dispatch_stage"] = "tail_mode_1_1_second_extract"
        lane.metadata["selected_seq_mode"] = "1.1"
        lane.metadata["seq_mode"] = "1.1"
        lane.metadata["lcxms"] = "1.1"
        lane.metadata["selected_round_label"] = stage_label
        for lib in list(lane.libraries or []):
            lib._current_seq_mode_raw = "1.1"
            lib.selected_seq_mode = "1.1"
            lib.current_seq_mode = "1.1"
            lib.lcxms = "1.1"
        lanes.append(lane)
        used_ids = {id(lib) for lib in used}
        used_total += len(used_ids)
        remaining = [lib for lib in remaining if id(lib) not in used_ids]
        serial += 1

    return (
        lanes,
        remaining,
        {
            "new_lanes": len(lanes),
            "used_libraries": used_total,
            "remaining_libraries": len(remaining),
        },
    )


# ==================== 辅助工具函数 ====================


def _safe_float(value, default: float = 0.0) -> float:
    """安全转换为float，处理NaN"""
    if pd.isna(value) or value is None or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value, default: int = 0) -> int:
    """安全转换为int，处理NaN"""
    if pd.isna(value) or value is None or value == '':
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _safe_str(value, default: str = '') -> str:
    """安全转换为str，处理NaN"""
    if pd.isna(value) or value is None:
        return default
    return str(value).strip()


def _normalize_mode_1_1_alias(value: Any) -> str:
    """统一1.1模式的历史别名口径，兼容 1 / 1.0 / 1.1 等老数据表达。"""
    text = _safe_str(value, default="")
    if not text:
        return ""

    try:
        numeric_value = float(text)
    except (TypeError, ValueError):
        numeric_value = None

    if numeric_value is not None and abs(numeric_value - 1.0) < 1e-9:
        return "1.1"

    normalized_text = _normalize_text_for_match(text).replace("模式", "").replace("MODE", "")
    if normalized_text in {"1.0", "1.1"}:
        return "1.1"

    return text


def _resolve_aiavailable_raw(row_dict: Dict[str, Any]) -> str:
    """兼容历史回放数据缺失 aiavailable 列的场景。"""
    if "aiavailable" in row_dict:
        return _safe_str(row_dict.get("aiavailable"), default="")
    return "yes"


def _apply_lane_orderdata_floor(order_value: Optional[float]) -> Optional[float]:
    """成Lane后下单量兜底：小于1统一抬升到1。"""
    if order_value is None:
        return None
    try:
        value = float(order_value)
    except (TypeError, ValueError):
        return order_value
    if value < LANE_ORDERDATA_FLOOR:
        return float(LANE_ORDERDATA_FLOOR)
    return value


def _is_yes_value(value: Any) -> bool:
    """判断字段值是否表达为yes。"""
    text = _safe_str(value, default="").upper()
    return text in {"Y", "YES", "TRUE", "1", "是"}


def _is_non_empty_value(value: Any) -> bool:
    """判断字段值是否为有效非空文本。"""
    text = _safe_str(value, default="")
    return text not in {"", "nan", "None", "NONE", "null", "NULL"}


def _is_split_rollback_unassigned_only(lib: EnhancedLibraryInfo) -> bool:
    """判断文库是否为拆分失败后仅允许未分配输出的原始文库。"""
    return bool(getattr(lib, "_split_family_rollback_unassigned_only", False))


def _is_split_rollback_mode_1_1_eligible(lib: EnhancedLibraryInfo) -> bool:
    """判断拆分回滚原始文库是否允许重新进入1.1排机。"""
    if not _is_split_rollback_unassigned_only(lib):
        return False
    if _is_yes_value(getattr(lib, "wkissplit", "")):
        return False
    contract_data = _safe_float(getattr(lib, "contract_data_raw", None), default=0.0)
    return contract_data <= ROLLBACK_SPLIT_LIBRARY_MODE_1_1_MAX_GB


def _is_split_library(lib: EnhancedLibraryInfo) -> bool:
    """判断文库是否为本轮拆分器生成且未回滚的拆分文库。"""
    if bool(getattr(lib, "_split_family_rollback_unassigned_only", False)):
        return False
    if _is_yes_value(getattr(lib, "wkissplit", "")):
        return True
    return bool(getattr(lib, "is_split", False)) or int(getattr(lib, "total_fragments", 0) or 0) > 1


def _should_library_split_by_rules(lib: EnhancedLibraryInfo) -> bool:
    """按当前拆分规则判断文库是否应拆分，不读取 wkissplit 标记。"""
    eval_lib = deepcopy(lib)
    current_data = _safe_float(getattr(eval_lib, "contract_data_raw", None), default=0.0)
    total_data_candidates = [
        _safe_float(getattr(eval_lib, "total_contract_data", None), default=0.0),
        _safe_float(getattr(eval_lib, "wktotalcontractdata", None), default=0.0),
    ]
    total_data = max([current_data] + total_data_candidates)
    if total_data > current_data:
        eval_lib.contract_data_raw = total_data
    eval_lib._current_seq_mode_raw = ""
    eval_lib.selected_seq_mode = ""
    eval_lib.current_seq_mode = ""
    eval_lib.lcxms = ""
    eval_lib.wkissplit = ""
    return bool(LibrarySplitter()._should_split(eval_lib))


def _should_library_split_in_3_6t(lib: EnhancedLibraryInfo) -> bool:
    """判断文库在3.6T-NEW当前行合同量口径下是否必须拆分。"""
    if _is_split_library(lib):
        return False
    eval_lib = deepcopy(lib)
    eval_lib._current_seq_mode_raw = "3.6T-NEW"
    eval_lib.selected_seq_mode = "3.6T-NEW"
    eval_lib.current_seq_mode = "3.6T-NEW"
    eval_lib.lcxms = "3.6T-NEW"
    eval_lib.wkissplit = ""
    return bool(LibrarySplitter()._should_split(eval_lib))


def _is_priority_library_for_36t_policy(lib: EnhancedLibraryInfo) -> bool:
    """历史兼容接口：高优文库逻辑已停用。"""
    return False


def _is_36t_only_secondary_priority(lib: EnhancedLibraryInfo, allocator: ModeAllocator) -> bool:
    """历史兼容接口：DHE/加测/混合不再禁止进入1.1。"""
    return False


def _is_mode_1_1_add_test_limited_library(lib: EnhancedLibraryInfo) -> bool:
    """识别1.1单Lane 150G封顶的加测/混合文库。"""
    remark = _safe_str(
        getattr(lib, "add_tests_remark", None)
        or getattr(lib, "wkaddtestsremark", None)
        or getattr(lib, "wkjcbz", None)
        or getattr(lib, "remark", None),
        default="",
    ).strip()
    negative_remarks = {"非加测", "无加测", "不加测", "无需加测", "未加测"}
    if remark and remark not in negative_remarks:
        return any(keyword in remark for keyword in ("加测", "混合"))
    return False


def _is_forbidden_in_mode_1_1_by_secondary_36t_policy(lib: EnhancedLibraryInfo) -> bool:
    """历史兼容接口：DHE/加测/混合不再禁止进入1.1。"""
    return False


def _is_split_rule_original_allowed_to_split_in_36t(lib: EnhancedLibraryInfo) -> bool:
    """原始文库只有不适合1.1，或1.1多轮失败后打标，才允许进入3.6T拆分。"""
    if _is_ai_balance_library(lib):
        return False
    if _is_split_library(lib):
        return True
    if bool(getattr(lib, "_mode_1_1_exhausted_allow_36t_split", False)):
        return _should_library_split_in_3_6t(lib)
    if not _should_library_split_in_3_6t(lib):
        return False
    return not _is_split_rule_original_allowed_in_1_1(lib)


def _is_small_unsplit_original_reserved_for_mode_1_1(lib: EnhancedLibraryInfo) -> bool:
    """<=500G未拆分原始文库只作为1.1候选，不作为普通3.6T补料。"""
    if _is_ai_balance_library(lib):
        return False
    if _is_split_library(lib):
        return False
    current_data = _safe_float(getattr(lib, "contract_data_raw", None), default=0.0)
    total_data_candidates = [
        current_data,
        _safe_float(getattr(lib, "total_contract_data", None), default=0.0),
        _safe_float(getattr(lib, "wktotalcontractdata", None), default=0.0),
        _safe_float(getattr(lib, "wkcontractdata", None), default=0.0),
    ]
    total_data = max(total_data_candidates)
    return 0.0 < total_data <= 500.0 + 1e-6


def _split_small_unsplit_originals_reserved_for_mode_1_1(
    libraries: List[EnhancedLibraryInfo],
) -> Tuple[List[EnhancedLibraryInfo], List[EnhancedLibraryInfo]]:
    """拆出只能继续尝试1.1、不能直接进入普通3.6T候选池的<=500G原始文库。"""
    allowed_for_36t: List[EnhancedLibraryInfo] = []
    reserved_for_1_1: List[EnhancedLibraryInfo] = []
    for lib in list(libraries or []):
        if _is_small_unsplit_original_reserved_for_mode_1_1(lib):
            reserved_for_1_1.append(lib)
        else:
            allowed_for_36t.append(lib)
    return allowed_for_36t, reserved_for_1_1


def _split_libraries_for_3_6t_allowed_sources(
    splitter: LibrarySplitter,
    libraries: List[EnhancedLibraryInfo],
) -> Tuple[List[EnhancedLibraryInfo], List[dict]]:
    """只对允许走3.6T拆分的原始文库做预拆分，<=500G保留原始文库给1.1/未分配。"""
    split_source_libraries: List[EnhancedLibraryInfo] = []
    passthrough_libraries: List[EnhancedLibraryInfo] = []
    for lib in list(libraries or []):
        if (not _is_split_library(lib)) and _should_library_split_in_3_6t(lib):
            if _is_split_rule_original_allowed_to_split_in_36t(lib):
                split_source_libraries.append(lib)
            else:
                passthrough_libraries.append(lib)
            continue
        passthrough_libraries.append(lib)

    if not split_source_libraries:
        return list(libraries or []), []

    split_libraries, split_records = splitter.split_libraries(split_source_libraries)
    return passthrough_libraries + split_libraries, split_records


def _has_completed_normal_1_1_attempt(lib: EnhancedLibraryInfo) -> bool:
    """判断普通文库是否已经完成1.1尝试并回流3.6T。"""
    return bool(getattr(lib, "_normal_1_1_attempted_before_36t", False))


def _is_eligible_for_36t_after_mode_priority(lib: EnhancedLibraryInfo) -> bool:
    """判断文库是否可作为3.6T主料。

    高优和>500G且命中拆分规则的原始文库可直接进3.6T；普通文库必须先完整尝试1.1，
    1.1仍排不完时才可作为3.6T主料继续消耗。
    """
    if _is_ai_balance_library(lib):
        return False
    if _is_split_library(lib):
        return True
    if _is_priority_library_for_36t_policy(lib):
        return True
    if _is_split_rule_original_allowed_to_split_in_36t(lib):
        return True
    if _is_small_unsplit_original_reserved_for_mode_1_1(lib):
        return False
    if _should_library_split_in_3_6t(lib):
        return False
    return _has_completed_normal_1_1_attempt(lib)


def _is_terminal_36t_candidate_after_1_1_gate(lib: EnhancedLibraryInfo) -> bool:
    """终态3.6T跨桶补Lane候选。

    3.6T优先/专属文库可直接进入；普通1.1可用文库必须已经完整尝试1.1，
    且未能成1.1 Lane 后才允许参与终态3.6T跨桶组合。
    """
    if _is_ai_balance_library(lib):
        return False
    if _is_split_library(lib):
        return True
    if _is_priority_library_for_36t_policy(lib):
        return True
    if _is_split_rule_original_allowed_to_split_in_36t(lib):
        return True
    if _is_small_unsplit_original_reserved_for_mode_1_1(lib):
        return False
    if _should_library_split_in_3_6t(lib):
        return False
    return _has_completed_normal_1_1_attempt(lib)


def _is_normal_filler_allowed_for_36t(lib: EnhancedLibraryInfo) -> bool:
    """判断文库是否可作为3.6T高优/拆分/1.1回流主料的普通补料。"""
    if _is_ai_balance_library(lib):
        return False
    if _is_split_library(lib):
        return False
    if _is_small_unsplit_original_reserved_for_mode_1_1(lib):
        return False
    if _should_library_split_in_3_6t(lib):
        return False
    return _is_terminal_36t_candidate_after_1_1_gate(lib)


def _is_split_rule_original_allowed_in_1_1(lib: EnhancedLibraryInfo) -> bool:
    """未拆分原始文库是否允许先排1.1。

    命中拆分规则且当前合同量<=500G的原始文库先尝试1.1；如果1.1未消耗，
    后续回到3.6T流程时再按拆分规则拆分成Lane。已拆分片段不回流1.1。
    """
    if _is_ai_balance_library(lib):
        return True
    if _is_split_library(lib):
        return False
    if not _should_library_split_by_rules(lib):
        return True
    current_data = _safe_float(getattr(lib, "contract_data_raw", None), default=0.0)
    return current_data <= 500.0


def _is_split_rule_original_blocked_from_1_1(lib: EnhancedLibraryInfo) -> bool:
    """判断文库是否因拆分状态或当前合同量上限不能进入1.1。"""
    if _is_ai_balance_library(lib):
        return False
    if _is_split_library(lib):
        return True
    return not _is_split_rule_original_allowed_in_1_1(lib)


def _infer_existing_split_fragment_count(
    fragment_data: float,
    total_contract_data: float,
) -> int:
    """按线上已拆分行的片段量和总量推断拆分份数。"""
    if fragment_data <= 0 or total_contract_data <= fragment_data + 1e-6:
        return 1
    ratio = total_contract_data / fragment_data
    rounded = int(round(ratio))
    if rounded > 1 and math.isclose(ratio, rounded, rel_tol=1e-6, abs_tol=1e-6):
        return rounded
    return max(2, int(math.ceil(ratio - 1e-9)))


def _normalize_existing_split_fragments(libraries: List[EnhancedLibraryInfo]) -> None:
    """为线上已拆分输入补齐家族元数据，保证终态拆分原子性复核生效。"""
    split_groups: Dict[str, List[EnhancedLibraryInfo]] = {}
    for lib in libraries:
        if not _is_yes_value(getattr(lib, "wkissplit", "")):
            continue
        if _get_package_lane_number_from_library(lib):
            continue
        fragment_data = _safe_float(getattr(lib, "contract_data_raw", None), default=0.0)
        total_data = max(
            fragment_data,
            _safe_float(getattr(lib, "total_contract_data", None), default=0.0),
            _safe_float(getattr(lib, "wktotalcontractdata", None), default=0.0),
        )
        if total_data <= fragment_data + 1e-6:
            continue
        family_id = _safe_str(
            getattr(lib, "original_library_id", None)
            or getattr(lib, "sample_id", None)
            or getattr(lib, "_source_origrec_key", None)
            or getattr(lib, "_origrec_key", None)
            or getattr(lib, "origrec", None),
            default="",
        )
        if not family_id:
            continue
        split_groups.setdefault(family_id, []).append(lib)

    for family_id, group in split_groups.items():
        if not group:
            continue
        max_total = max(
            _safe_float(getattr(lib, "total_contract_data", None), default=0.0)
            or _safe_float(getattr(lib, "wktotalcontractdata", None), default=0.0)
            or _safe_float(getattr(lib, "contract_data_raw", None), default=0.0)
            for lib in group
        )
        fragment_values = [
            _safe_float(getattr(lib, "contract_data_raw", None), default=0.0)
            for lib in group
        ]
        positive_fragment_values = [value for value in fragment_values if value > 0]
        if not positive_fragment_values:
            continue
        min_fragment = min(positive_fragment_values)
        expected_count = max(
            len(group),
            _infer_existing_split_fragment_count(min_fragment, max_total),
        )
        source_library = deepcopy(group[0])
        source_library.contract_data_raw = max_total
        source_library.wktotalcontractdata = max_total
        source_library.total_contract_data = max_total
        source_library.is_split = False
        source_library.wkissplit = ""
        source_library.split_status = "rolled_back"
        source_library.original_library_id = ""
        source_library.fragment_index = 0
        source_library.total_fragments = 0
        source_library.fragment_id = ""
        source_library._source_origrec_key = _safe_str(
            getattr(group[0], "_source_origrec_key", None)
            or getattr(group[0], "_origrec_key", None)
            or getattr(group[0], "origrec", None),
            default="",
        )
        source_library._detail_output_key = _safe_str(
            getattr(group[0], "_detail_output_key", None)
            or getattr(group[0], "wkaidbid", None)
            or getattr(group[0], "aidbid", None)
            or getattr(group[0], "origrec", None),
            default="",
        )

        ordered_group = sorted(
            group,
            key=lambda lib: (
                _safe_str(getattr(lib, "_source_origrec_key", None) or getattr(lib, "_origrec_key", None), default=""),
                _safe_str(getattr(lib, "wkaidbid", None) or getattr(lib, "aidbid", None), default=""),
            ),
        )
        for fragment_index, lib in enumerate(ordered_group, start=1):
            lib.is_split = True
            lib.wkissplit = "yes"
            lib.split_status = _safe_str(getattr(lib, "split_status", None), default="") or "completed"
            lib.original_library_id = family_id
            lib.total_fragments = expected_count
            lib.fragment_index = int(getattr(lib, "fragment_index", 0) or fragment_index)
            lib.fragment_id = _safe_str(getattr(lib, "fragment_id", None), default="") or f"{family_id}_F{lib.fragment_index:03d}"
            lib.wktotalcontractdata = max_total
            lib.total_contract_data = max_total
            lib._split_source_library = source_library
            lib._source_origrec_key = _safe_str(
                getattr(lib, "_source_origrec_key", None)
                or getattr(lib, "_origrec_key", None)
                or getattr(lib, "origrec", None),
                default="",
            )
            lib._detail_output_key = _safe_str(
                getattr(lib, "_detail_output_key", None)
                or getattr(lib, "wkaidbid", None)
                or getattr(lib, "aidbid", None)
                or lib.fragment_id,
                default="",
            )
            if hasattr(lib, "_library_identity_key_cache"):
                delattr(lib, "_library_identity_key_cache")


def _collect_lanes_with_split(lanes: List[LaneAssignment]) -> Set[str]:
    """收集包含拆分文库的lane_id集合。"""
    lane_ids: Set[str] = set()
    for lane in lanes:
        libs = list(getattr(lane, "libraries", []) or [])
        if any(_is_split_library(lib) for lib in libs):
            lane_ids.add(str(lane.lane_id))
    return lane_ids


def _collect_detail_output_libraries(solution: Any) -> List[EnhancedLibraryInfo]:
    """收集最终输出明细所需的全部文库，包含成Lane与未分配文库。"""
    detail_libraries: List[EnhancedLibraryInfo] = []
    for lane in getattr(solution, "lane_assignments", []) or []:
        detail_libraries.extend(list(getattr(lane, "libraries", []) or []))
    detail_libraries.extend(list(getattr(solution, "unassigned_libraries", []) or []))
    return detail_libraries


def _collect_split_family_state(solution: Any) -> Dict[str, Dict[str, Any]]:
    """收集当前解中拆分家族的成Lane/未分配状态。"""
    family_state: Dict[str, Dict[str, Any]] = {}
    all_split_libraries: List[EnhancedLibraryInfo] = []

    def ensure_entry(family_id: str) -> Dict[str, Any]:
        return family_state.setdefault(
            family_id,
            {
                "assigned": [],
                "unassigned": [],
                "expected": 0,
                "source": None,
            },
        )

    for lane in list(getattr(solution, "lane_assignments", []) or []):
        all_split_libraries.extend(
            lib
            for lib in list(getattr(lane, "libraries", []) or [])
            if _is_split_library(lib) and not bool(getattr(lib, "_package_lane_multi_split", False))
        )
    all_split_libraries.extend(
        lib
        for lib in list(getattr(solution, "unassigned_libraries", []) or [])
        if _is_split_library(lib) and not bool(getattr(lib, "_package_lane_multi_split", False))
    )
    family_actual_counts: Dict[str, int] = {}
    for lib in all_split_libraries:
        family_id = _get_split_family_id_for_lane_build(lib)
        if not family_id:
            continue
        family_actual_counts[family_id] = family_actual_counts.get(family_id, 0) + 1

    for lane in list(getattr(solution, "lane_assignments", []) or []):
        for lib in list(getattr(lane, "libraries", []) or []):
            family_id = _get_split_family_id_for_lane_build(lib)
            if not family_id or bool(getattr(lib, "_package_lane_multi_split", False)):
                continue
            expected_count = _get_split_family_expected_count(family_id, family_actual_counts, lib)
            if expected_count <= 1:
                continue
            entry = ensure_entry(family_id)
            entry["assigned"].append((lane, lib))
            entry["expected"] = max(int(entry["expected"] or 0), expected_count)
            source_library = getattr(lib, "_split_source_library", None)
            if source_library is not None:
                entry["source"] = source_library

    for lib in list(getattr(solution, "unassigned_libraries", []) or []):
        family_id = _get_split_family_id_for_lane_build(lib)
        if not family_id or bool(getattr(lib, "_package_lane_multi_split", False)):
            continue
        expected_count = _get_split_family_expected_count(family_id, family_actual_counts, lib)
        if expected_count <= 1:
            continue
        entry = ensure_entry(family_id)
        entry["unassigned"].append(lib)
        entry["expected"] = max(int(entry["expected"] or 0), expected_count)
        source_library = getattr(lib, "_split_source_library", None)
        if source_library is not None:
            entry["source"] = source_library

    return family_state


def _get_max_split_fragment_family_ids(
    family_state: Dict[str, Dict[str, Any]],
) -> Set[str]:
    """返回当前解中拆分份数最大的文库家族ID。"""
    max_expected = 0
    for entry in family_state.values():
        expected_count = int(entry.get("expected", 0) or 0)
        if expected_count > max_expected:
            max_expected = expected_count
    if max_expected <= 1:
        return set()
    return {
        family_id
        for family_id, entry in family_state.items()
        if int(entry.get("expected", 0) or 0) == max_expected
    }


def _lane_is_valid_after_split_repair(lane: LaneAssignment, validator: Any) -> bool:
    """判断拆分修复后的lane是否仍符合终态校验。"""
    if _is_package_lane_assignment(lane):
        return True
    metadata = _build_lane_metadata_for_validator(
        lane.lane_id,
        lane.metadata,
        libraries=list(getattr(lane, "libraries", []) or []),
    )
    result = _validate_lane_with_latest_index(
        validator=validator,
        libraries=list(getattr(lane, "libraries", []) or []),
        lane_id=str(lane.lane_id),
        machine_type=lane.machine_type.value if lane.machine_type else "Nova X-25B",
        metadata=metadata,
    )
    return bool(result.is_valid)


def _try_reorder_lanes_to_keep_split_families_in_same_run(solution: Any) -> int:
    """尝试通过调整lane顺序让完整拆分家族落入同一个8-lane run。"""
    lanes = list(getattr(solution, "lane_assignments", []) or [])
    if not lanes:
        return 0
    family_state = _collect_split_family_state(solution)
    max_split_family_ids = _get_max_split_fragment_family_ids(family_state)
    priority_lane_ids: List[str] = []
    ordered_family_items = sorted(
        family_state.items(),
        key=lambda item: (
            item[0] not in max_split_family_ids,
            -int(item[1].get("expected", 0) or 0),
            item[0],
        ),
    )
    for family_id, entry in ordered_family_items:
        expected_count = int(entry.get("expected", 0) or 0)
        assigned_items = list(entry.get("assigned", []) or [])
        if expected_count <= 1 or len(assigned_items) != expected_count or entry.get("unassigned"):
            continue
        current_runids = {
            index // 8
            for index, lane in enumerate(lanes)
            if any(lane is assigned_lane for assigned_lane, _ in assigned_items)
        }
        if len(current_runids) <= 1:
            continue
        if family_id in max_split_family_ids:
            logger.warning(
                "最大拆分份数文库家族{}当前跨{}个run窗口，按红线规则优先重排到同一runid",
                family_id,
                len(current_runids),
            )
        for lane, _ in assigned_items:
            lane_id = str(lane.lane_id)
            if lane_id not in priority_lane_ids:
                priority_lane_ids.append(lane_id)

    if not priority_lane_ids:
        return 0

    priority_set = set(priority_lane_ids)
    priority_lanes = [lane for lane in lanes if str(lane.lane_id) in priority_set]
    other_lanes = [lane for lane in lanes if str(lane.lane_id) not in priority_set]
    solution.lane_assignments = priority_lanes + other_lanes
    return len(priority_lanes)


def _try_place_unassigned_split_fragments_into_existing_run(solution: Any, validator: Any) -> int:
    """尝试把未分配拆分片段补回同一run内的合规lane。"""
    lanes = list(getattr(solution, "lane_assignments", []) or [])
    if not lanes:
        return 0
    family_state = _collect_split_family_state(solution)
    placed_count = 0
    for family_id, entry in family_state.items():
        expected_count = int(entry.get("expected", 0) or 0)
        assigned_items = list(entry.get("assigned", []) or [])
        pending_items = list(entry.get("unassigned", []) or [])
        if expected_count <= 1 or not pending_items:
            continue
        if len(assigned_items) + len(pending_items) != expected_count:
            continue
        assigned_lane_ids = {str(lane.lane_id) for lane, _ in assigned_items}
        if assigned_items:
            lane_indices = [idx for idx, lane in enumerate(lanes) if str(lane.lane_id) in assigned_lane_ids]
            if not lane_indices:
                continue
            target_run_index = min(lane_indices) // 8
        else:
            target_run_index = 0
        run_lanes = lanes[target_run_index * 8:(target_run_index + 1) * 8]
        for fragment in list(pending_items):
            placed = False
            for lane in run_lanes:
                if _is_package_lane_assignment(lane):
                    continue
                lane_id = str(lane.lane_id)
                if lane_id in assigned_lane_ids:
                    continue
                if any(_get_split_family_id_for_lane_build(lib) == family_id for lib in list(lane.libraries or [])):
                    continue
                lane.add_library(fragment)
                if _lane_is_valid_after_split_repair(lane, validator):
                    solution.unassigned_libraries = [
                        lib for lib in list(getattr(solution, "unassigned_libraries", []) or [])
                        if id(lib) != id(fragment)
                    ]
                    assigned_lane_ids.add(lane_id)
                    placed_count += 1
                    placed = True
                    break
                lane.remove_library(fragment)
            if not placed:
                break
    return placed_count


def _repair_split_families_before_final_rollback(
    solution: Any,
    validator: Any,
    max_attempts: int = 2,
) -> Dict[str, int]:
    """终态回滚前尝试修复拆分家族原子性问题。"""
    stats = {"attempts": 0, "reordered_lanes": 0, "placed_fragments": 0}
    for _ in range(max(0, int(max_attempts))):
        stats["attempts"] += 1
        reordered_lanes = _try_reorder_lanes_to_keep_split_families_in_same_run(solution)
        placed_fragments = _try_place_unassigned_split_fragments_into_existing_run(solution, validator)
        stats["reordered_lanes"] += reordered_lanes
        stats["placed_fragments"] += placed_fragments
        if reordered_lanes == 0 and placed_fragments == 0:
            break
    return stats


def _enforce_split_family_atomicity_for_stage(
    lane_assignments: List[LaneAssignment],
    unassigned_libraries: List[EnhancedLibraryInfo],
    validator: Any,
    stage_label: str,
) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo], Dict[str, int]]:
    """阶段内即时修复拆分家族，优先修复重建，失败后再回退到未分配池。"""
    from types import SimpleNamespace

    stage_solution = SimpleNamespace(
        lane_assignments=list(lane_assignments or []),
        unassigned_libraries=list(unassigned_libraries or []),
    )
    repair_stats = _repair_split_families_before_final_rollback(
        solution=stage_solution,
        validator=validator,
        max_attempts=2,
    )
    rollback_stats = _rollback_incomplete_split_families_in_final_solution(stage_solution)
    matrix_split_stats = {"added_lanes": 0, "used_originals": 0, "added_fragments": 0}
    mixed_matrix_split_stats = {"added_lanes": 0, "used_originals": 0, "added_fragments": 0}
    cross_split_stats = {"added_lanes": 0, "used_originals": 0, "added_fragments": 0}
    second_pass_rollback_stats = {
        "rollback_families": 0,
        "incomplete_families": 0,
        "cross_run_families": 0,
        "max_split_cross_run_families": 0,
        "removed_fragments": 0,
        "restored_originals": 0,
    }
    if int(rollback_stats.get("rollback_families", 0) or 0) > 0:
        matrix_split_stats = _try_add_matrix_split_lanes_from_unassigned(
            solution=stage_solution,
            validator=validator,
        )
        mixed_matrix_split_stats = _try_add_mixed_matrix_split_lanes_from_unassigned(
            solution=stage_solution,
            validator=validator,
        )
        cross_split_stats = _try_add_cross_split_fragment_lanes_from_unassigned(
            solution=stage_solution,
            validator=validator,
        )
        if (
            int(matrix_split_stats.get("added_lanes", 0) or 0) > 0
            or int(mixed_matrix_split_stats.get("added_lanes", 0) or 0) > 0
            or int(cross_split_stats.get("added_lanes", 0) or 0) > 0
        ):
            second_pass_rollback_stats = _rollback_incomplete_split_families_in_final_solution(
                stage_solution
            )
    combined_stats = {
        "repair_attempts": int(repair_stats.get("attempts", 0) or 0),
        "reordered_lanes": int(repair_stats.get("reordered_lanes", 0) or 0),
        "placed_fragments": int(repair_stats.get("placed_fragments", 0) or 0),
        "rollback_families": int(rollback_stats.get("rollback_families", 0) or 0),
        "incomplete_families": int(rollback_stats.get("incomplete_families", 0) or 0),
        "cross_run_families": int(rollback_stats.get("cross_run_families", 0) or 0),
        "max_split_cross_run_families": int(rollback_stats.get("max_split_cross_run_families", 0) or 0),
        "removed_fragments": int(rollback_stats.get("removed_fragments", 0) or 0),
        "restored_originals": int(rollback_stats.get("restored_originals", 0) or 0),
        "matrix_added_lanes": int(matrix_split_stats.get("added_lanes", 0) or 0),
        "matrix_used_originals": int(matrix_split_stats.get("used_originals", 0) or 0),
        "mixed_matrix_added_lanes": int(mixed_matrix_split_stats.get("added_lanes", 0) or 0),
        "mixed_matrix_used_originals": int(mixed_matrix_split_stats.get("used_originals", 0) or 0),
        "cross_split_added_lanes": int(cross_split_stats.get("added_lanes", 0) or 0),
        "cross_split_used_originals": int(cross_split_stats.get("used_originals", 0) or 0),
        "post_rebuild_rollback_families": int(second_pass_rollback_stats.get("rollback_families", 0) or 0),
    }
    if (
        combined_stats["reordered_lanes"] > 0
        or combined_stats["placed_fragments"] > 0
        or combined_stats["rollback_families"] > 0
        or combined_stats["max_split_cross_run_families"] > 0
        or combined_stats["matrix_added_lanes"] > 0
        or combined_stats["mixed_matrix_added_lanes"] > 0
        or combined_stats["cross_split_added_lanes"] > 0
    ):
        logger.info(
            "{}拆分即时校验完成: 重排Lane={}，补入片段={}，回滚家族={}，最大份数跨run红线={}，恢复原始文库={}，矩阵重建Lane={}，混合矩阵Lane={}，跨份数Lane={}，重建后残余回滚家族={}".format(
                stage_label,
                combined_stats["reordered_lanes"],
                combined_stats["placed_fragments"],
                combined_stats["rollback_families"],
                combined_stats["max_split_cross_run_families"],
                combined_stats["restored_originals"],
                combined_stats["matrix_added_lanes"],
                combined_stats["mixed_matrix_added_lanes"],
                combined_stats["cross_split_added_lanes"],
                combined_stats["post_rebuild_rollback_families"],
            )
        )
    return (
        list(getattr(stage_solution, "lane_assignments", []) or []),
        list(getattr(stage_solution, "unassigned_libraries", []) or []),
        combined_stats,
    )


def _rollback_incomplete_split_families_in_final_solution(solution: Any) -> Dict[str, int]:
    """终态复核拆分家族，未全部成Lane或跨runid时恢复为原始文库。"""
    family_state = _collect_split_family_state(solution)
    max_split_family_ids = _get_max_split_fragment_family_ids(family_state)
    family_assigned: Dict[str, List[Tuple[LaneAssignment, EnhancedLibraryInfo]]] = {
        family_id: list(entry.get("assigned", []) or [])
        for family_id, entry in family_state.items()
    }
    family_unassigned: Dict[str, List[EnhancedLibraryInfo]] = {
        family_id: list(entry.get("unassigned", []) or [])
        for family_id, entry in family_state.items()
    }
    expected_counts: Dict[str, int] = {
        family_id: int(entry.get("expected", 0) or 0)
        for family_id, entry in family_state.items()
    }
    source_libraries: Dict[str, EnhancedLibraryInfo] = {
        family_id: entry.get("source")
        for family_id, entry in family_state.items()
        if entry.get("source") is not None
    }

    runid_by_lane = _build_runid_by_lane(list(getattr(solution, "lane_assignments", []) or []))
    rollback_family_ids: Set[str] = set()
    cross_run_family_ids: Set[str] = set()
    max_split_cross_run_family_ids: Set[str] = set()
    incomplete_family_ids: Set[str] = set()
    for family_id, expected_count in expected_counts.items():
        assigned_items = family_assigned.get(family_id, [])
        assigned_count = len(assigned_items)
        unassigned_count = len(family_unassigned.get(family_id, []))
        if expected_count <= 1:
            continue
        is_complete_package_lane_split = (
            assigned_count == expected_count
            and unassigned_count == 0
            and all(
                _is_package_lane_assignment(lane)
                and _get_package_lane_number_from_library(lib)
                and _get_package_lane_number_from_lane(lane) == _get_package_lane_number_from_library(lib)
                for lane, lib in assigned_items
            )
        )
        if is_complete_package_lane_split:
            continue
        if assigned_count != expected_count or unassigned_count > 0:
            rollback_family_ids.add(family_id)
            incomplete_family_ids.add(family_id)
            continue
        assigned_runids = {
            str(runid_by_lane.get(lane.lane_id, "") or "").strip()
            for lane, _ in assigned_items
            if str(runid_by_lane.get(lane.lane_id, "") or "").strip()
        }
        if len(assigned_runids) > 1:
            rollback_family_ids.add(family_id)
            cross_run_family_ids.add(family_id)
            if family_id in max_split_family_ids:
                max_split_cross_run_family_ids.add(family_id)
                logger.warning(
                    "最大拆分份数文库家族{}违反同runid红线: assigned_runids={}",
                    family_id,
                    sorted(assigned_runids),
                )

    if not rollback_family_ids:
        return {
            "rollback_families": 0,
            "removed_fragments": 0,
            "restored_originals": 0,
            "incomplete_families": 0,
            "cross_run_families": 0,
            "max_split_cross_run_families": 0,
        }

    removed_fragment_ids: Set[int] = set()
    for family_id in rollback_family_ids:
        for lane, lib in family_assigned.get(family_id, []):
            lane.remove_library(lib)
            removed_fragment_ids.add(id(lib))

    solution.lane_assignments = [
        lane for lane in list(getattr(solution, "lane_assignments", []) or [])
        if list(getattr(lane, "libraries", []) or [])
    ]
    solution.unassigned_libraries = [
        lib for lib in list(getattr(solution, "unassigned_libraries", []) or [])
        if id(lib) not in removed_fragment_ids
        and _get_split_family_id_for_lane_build(lib) not in rollback_family_ids
    ]

    existing_source_keys: Set[str] = {
        _get_library_source_origrec_key(lib)
        for lib in list(getattr(solution, "unassigned_libraries", []) or [])
    }
    restored_originals = 0
    for family_id in rollback_family_ids:
        source_library = source_libraries.get(family_id)
        if source_library is None:
            continue
        source_library.is_split = False
        source_library.wkissplit = ""
        source_library.split_status = "rolled_back"
        source_key = _get_library_source_origrec_key(source_library)
        if source_key and source_key in existing_source_keys:
            continue
        solution.unassigned_libraries.append(source_library)
        if source_key:
            existing_source_keys.add(source_key)
        restored_originals += 1

    return {
        "rollback_families": len(rollback_family_ids),
        "removed_fragments": len(removed_fragment_ids),
        "restored_originals": restored_originals,
        "incomplete_families": len(incomplete_family_ids),
        "cross_run_families": len(cross_run_family_ids),
        "max_split_cross_run_families": len(max_split_cross_run_family_ids),
    }


def _normalize_special_split_token(value: Any) -> List[str]:
    """将wkspecialsplits值规范化为token列表。"""
    raw = _safe_str(value, default="").lower()
    if raw in {"", "-", "nan", "none", "null"}:
        return []
    normalized = raw.replace(";", ",").replace("|", ",").replace("/", ",")
    return [token.strip() for token in normalized.split(",") if token.strip()]


def _get_library_special_split_tokens(lib: EnhancedLibraryInfo) -> Set[str]:
    """获取单个文库的wkspecialsplits token集合。"""
    raw = getattr(lib, "special_splits", None)
    if raw is None:
        raw = getattr(lib, "wkspecialsplits", None)
    return set(_normalize_special_split_token(raw))


def _classify_library_special_split_mode(lib: EnhancedLibraryInfo) -> str:
    """将文库按wkspecialsplits归类到A/B/EMPTY/OTHER。"""
    tokens = _get_library_special_split_tokens(lib)
    if not tokens:
        return "EMPTY"
    if tokens.issubset(SPECIAL_SPLIT_GROUP_A):
        return "A"
    if tokens.issubset(SPECIAL_SPLIT_GROUP_B):
        return "B"
    return "OTHER"


def _collect_lane_special_split_tokens(libraries: List[EnhancedLibraryInfo]) -> Set[str]:
    """汇总Lane内wkspecialsplits token集合。"""
    tokens: Set[str] = set()
    for lib in libraries:
        if _is_ai_balance_library(lib):
            continue
        raw = getattr(lib, "special_splits", None)
        if raw is None:
            raw = getattr(lib, "wkspecialsplits", None)
        for token in _normalize_special_split_token(raw):
            tokens.add(token)
    return tokens


def _validate_lane_special_split_rule(
    libraries: List[EnhancedLibraryInfo],
) -> Tuple[bool, Set[str], str]:
    """校验Lane内wkspecialsplits组合规则。"""
    tokens = _collect_lane_special_split_tokens(libraries)
    mode_counter: Dict[str, int] = {"A": 0, "B": 0, "EMPTY": 0, "OTHER": 0}
    for lib in libraries:
        if _is_ai_balance_library(lib):
            continue
        mode = _classify_library_special_split_mode(lib)
        mode_counter[mode] = mode_counter.get(mode, 0) + 1

    if mode_counter["OTHER"] > 0:
        return False, tokens, "contains_unknown_special_split_token"

    if mode_counter["A"] > 0 and mode_counter["B"] > 0:
        return False, tokens, "group_a_and_group_b_mixed"

    # B组：仅允许B组内部混排，不可与其他任何类型同Lane（含空值）。
    if mode_counter["B"] > 0 and (mode_counter["A"] > 0 or mode_counter["EMPTY"] > 0):
        return False, tokens, "group_b_mixed_with_non_group_b"

    if mode_counter["B"] > 0:
        return True, tokens, "special_split_group_b_only"
    if mode_counter["A"] > 0:
        return True, tokens, "special_split_group_a"
    return True, tokens, "empty_special_splits"


def _get_lane_sample_types(libraries: List[EnhancedLibraryInfo]) -> Set[str]:
    """提取Lane内文库类型集合（统一匹配口径）。"""
    sample_types: Set[str] = set()
    for lib in libraries:
        sample_type = getattr(lib, "sample_type_code", "") or getattr(lib, "sampletype", "")
        normalized = _normalize_text_for_match(sample_type)
        if normalized:
            sample_types.add(normalized)
    return sample_types


def _lane_contains_customer_prefixed_sample_type(
    lane_sample_types: Set[str],
) -> bool:
    """判断Lane文库类型中是否存在客户前缀。"""
    return any("客户-" in sample_type for sample_type in lane_sample_types)


def _resolve_explicit_lane_loading_concentration(
    libraries: List[EnhancedLibraryInfo],
    lane_sample_types: Set[str],
) -> Tuple[Optional[float], str]:
    """按业务显式规则优先解析Lane排机浓度。"""
    if not libraries:
        return None, "empty_lane"

    has_10_plus_24 = any(
        _matches_lane_seq_strategy_keyword(lib, "10+24")
        for lib in libraries
    )
    has_atac_sample_type = any(
        _library_sample_type_matches_rule(lib, LANE_LOADING_10_PLUS_24_ATAC_TYPES)
        for lib in libraries
    )
    if has_10_plus_24 or has_atac_sample_type:
        return 1.9, "10_plus_24_or_atac_1_9"

    if lane_sample_types and lane_sample_types.issubset(LANE_LOADING_COMBO_GROUP_B):
        if _lane_contains_customer_prefixed_sample_type(lane_sample_types):
            return 2.5, "special_10x_combo_group_b_customer_2_5"
        return 1.78, "special_10x_combo_group_b_non_customer_1_78"

    clinical_data = sum(
        float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
        for lib in libraries
        if _is_clinical_data_type_library(lib)
    )
    if clinical_data > 100.0:
        return 2.3, "clinical_data_over_100g_2_3"

    return None, "no_explicit_loading_rule_matched"


def _is_clinical_data_type_library(lib: EnhancedLibraryInfo) -> bool:
    """判断文库数据类型是否为临检。"""
    data_type = _normalize_text_for_match(getattr(lib, "data_type", ""))
    return data_type == "临检"


def _matches_lane_seq_strategy_keyword(
    lib: EnhancedLibraryInfo, strategy_keyword: str
) -> bool:
    """判断文库是否命中指定测序策略关键字。"""
    normalized_keyword = _normalize_seq_strategy_keyword(strategy_keyword)
    if not normalized_keyword:
        return False

    # wkseqnotes/machine_note 是备注，不能作为 Lane seq 策略判定来源。
    strategy_candidates = [
        getattr(lib, "_lane_sj_mode_raw", ""),
        getattr(lib, "test_no", ""),
        getattr(lib, "seq_scheme", ""),
    ]
    for candidate in strategy_candidates:
        normalized = _normalize_seq_strategy_keyword(candidate)
        if normalized_keyword in normalized:
            return True
    return False


def _library_sample_type_matches_rule(
    lib: EnhancedLibraryInfo, sample_types: Set[str]
) -> bool:
    """判断文库类型是否命中规则配置中的文库类型集合。"""
    sample_type = _normalize_text_for_match(
        getattr(lib, "sample_type_code", "") or getattr(lib, "sampletype", "")
    )
    if not sample_type:
        return False
    return sample_type in sample_types


def _match_lane_loading_concentration_rule(
    rule: Dict[str, Any],
    libraries: List[EnhancedLibraryInfo],
    lane_sample_types: Set[str],
) -> bool:
    """判断Lane是否命中单条上机浓度规则。"""
    rule_type = str(rule.get("rule_type", "") or "").strip()
    sample_types: Set[str] = set(rule.get("sample_types", set()) or set())

    if rule_type == "sample_type_subset":
        return bool(lane_sample_types) and lane_sample_types.issubset(sample_types)

    if rule_type == "clinical_data_threshold":
        threshold = float(rule.get("data_threshold_gb", 0.0) or 0.0)
        clinical_data = sum(
            float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
            for lib in libraries
            if _is_clinical_data_type_library(lib)
        )
        return clinical_data > threshold

    if rule_type == "seq_strategy_and_sample_type":
        strategy_keyword = str(rule.get("seq_strategy_keyword", "") or "")
        has_strategy = any(
            _matches_lane_seq_strategy_keyword(lib, strategy_keyword)
            for lib in libraries
        )
        has_target_sample_type = any(
            _library_sample_type_matches_rule(lib, sample_types)
            for lib in libraries
        )
        return has_strategy and has_target_sample_type

    logger.warning(f"未知Lane上机浓度规则类型: {rule_type}")
    return False


def _resolve_lane_loading_concentration(
    libraries: List[EnhancedLibraryInfo],
    lane_id: str = "",
    lane_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], str]:
    """严格按显式业务规则解析上机浓度，未命中规则时不赋值。"""
    if not libraries:
        return None, "empty_lane"
    lane_sample_types = _get_lane_sample_types(libraries)
    explicit_concentration, explicit_rule = _resolve_explicit_lane_loading_concentration(
        libraries,
        lane_sample_types,
    )
    if explicit_concentration is not None:
        return explicit_concentration, explicit_rule

    return None, "no_loading_concentration_rule_matched"


def _resolve_lane_output_rule_fields(
    libraries: List[EnhancedLibraryInfo],
    machine_type: MachineType | str,
    lane_id: str = "",
    lane_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, str]:
    """解析Lane输出字段所需的规则结果。

    Returns:
        (loading_method, sequencing_mode, rule_code)
    """
    if not libraries:
        return "", "", "empty_lane"
    selection = _resolve_lane_capacity_selection(
        libraries=libraries,
        machine_type=machine_type,
        lane_id=lane_id,
        lane_metadata=lane_metadata,
    )
    machine_type_text = _machine_type_to_text(machine_type, default="")
    normalized_machine_type = _normalize_text_for_match(machine_type_text)
    loading_method = str(getattr(selection, "loading_method", "") or "").strip()
    sequencing_mode = str(getattr(selection, "sequencing_mode", "") or "").strip()
    rule_code = str(getattr(selection, "rule_code", "") or "").strip()

    if (
        _normalize_text_for_match(sequencing_mode) == _normalize_text_for_match("3.6T-NEW")
        and not _is_lane_seq_10_plus_24_lane_assignment(
            LaneAssignment(
                lane_id=lane_id or "OUTPUT_MODE_CHECK",
                machine_id="OUTPUT_MODE_CHECK",
                machine_type=(
                    machine_type
                    if isinstance(machine_type, MachineType)
                    else _resolve_machine_type_enum_simple(_machine_type_to_text(machine_type, default=""))
                ),
                lane_capacity_gb=_lane_capacity_for_machine(
                    machine_type
                    if isinstance(machine_type, MachineType)
                    else _resolve_machine_type_enum_simple(_machine_type_to_text(machine_type, default=""))
                ),
                libraries=list(libraries or []),
                metadata=dict(lane_metadata or {}),
            )
        )
        ):
        mode_1_1_metadata = dict(lane_metadata or {})
        mode_1_1_metadata.pop("capacity_rule_code", None)
        mode_1_1_metadata["selected_seq_mode"] = "1.1"
        mode_1_1_metadata["seq_mode"] = "1.1"
        mode_1_1_metadata["lcxms"] = "1.1"
        mode_1_1_metadata["sequencing_mode"] = "1.1"
        mode_1_1_selection = _resolve_lane_capacity_selection(
            libraries=libraries,
            machine_type=machine_type,
            lane_id=lane_id,
            lane_metadata=mode_1_1_metadata,
        )
        total_data_gb = _total_lane_data(list(libraries or []))
        mode_1_1_min_gb = float(getattr(mode_1_1_selection, "effective_min_gb", 0.0) or 0.0)
        mode_1_1_max_gb = float(getattr(mode_1_1_selection, "effective_max_gb", 0.0) or 0.0)
        mode_1_1_rule_code = str(getattr(mode_1_1_selection, "rule_code", "") or "").strip()
        if (
            mode_1_1_rule_code.startswith("tj_1595_mode_1_1")
            and total_data_gb + 1e-6 >= mode_1_1_min_gb
            and total_data_gb <= mode_1_1_max_gb + 1e-6
        ):
            loading_method = str(getattr(mode_1_1_selection, "loading_method", "") or "").strip() or loading_method
            sequencing_mode = "1.1"
            rule_code = mode_1_1_rule_code

    # Nova X-25B 与 NovaSeq X Plus 业务上统一按 25B 上机方式输出。
    if not loading_method and normalized_machine_type in {
        _normalize_text_for_match("Nova X-25B"),
        _normalize_text_for_match("NovaSeq X Plus"),
    }:
        loading_method = "25B"

    return loading_method, sequencing_mode, rule_code


@lru_cache(maxsize=1)
def _load_lane_index_rule_mapping() -> Tuple[Dict[Tuple[str, str], str], Dict[str, str]]:
    """加载显式排机规则映射。

    返回:
        (
            {(标准化工序名称, 标准化上机方式): 排机规则},
            {标准化工序名称: 唯一排机规则}
        )
    """
    pair_map: Dict[Tuple[str, str], str] = {}
    test_rule_candidates: Dict[str, Set[str]] = {}

    if not INDEX_RULE_CONFIG_PATH.exists():
        logger.warning(f"排机规则映射文件不存在，跳过显式排机规则解析: {INDEX_RULE_CONFIG_PATH}")
        return pair_map, {}

    try:
        df = pd.read_csv(INDEX_RULE_CONFIG_PATH, sep="\t", dtype=str).fillna("")
    except Exception as exc:
        logger.warning(f"加载排机规则映射失败: {exc}")
        return pair_map, {}

    for _, row in df.iterrows():
        test_no = _normalize_text_for_match(row.get("工序名称", ""))
        loading_method = _normalize_text_for_match(row.get("上机方式", ""))
        index_rule = str(row.get("排机规则", "") or "").strip().upper()
        if not test_no or not index_rule:
            continue
        if loading_method:
            pair_map[(test_no, loading_method)] = index_rule
        test_rule_candidates.setdefault(test_no, set()).add(index_rule)

    unique_test_rule_map = {
        test_no: next(iter(rule_values))
        for test_no, rule_values in test_rule_candidates.items()
        if len(rule_values) == 1
    }
    return pair_map, unique_test_rule_map


def _resolve_lane_index_rule_display(
    libraries: List[EnhancedLibraryInfo],
    loading_method: str,
) -> str:
    """解析Lane显式排机规则显示值，不再依赖wkindexseq是否含分号。"""
    if not libraries:
        return ""

    pair_map, unique_test_rule_map = _load_lane_index_rule_mapping()
    normalized_loading_method = _normalize_text_for_match(loading_method)

    matched_rules: Set[str] = set()
    for lib in libraries:
        test_no = _normalize_text_for_match(getattr(lib, "test_no", "") or getattr(lib, "testno", ""))
        if not test_no:
            continue

        pair_key = (test_no, normalized_loading_method)
        if normalized_loading_method and pair_key in pair_map:
            matched_rules.add(pair_map[pair_key])
            continue

        unique_rule = unique_test_rule_map.get(test_no, "")
        if unique_rule:
            matched_rules.add(unique_rule)

    if len(matched_rules) == 1:
        return next(iter(matched_rules))

    # NovaSeq X Plus / 25B / 10B 业务上默认走双端查重规则。
    lane_test_nos = {
        _normalize_text_for_match(getattr(lib, "test_no", "") or getattr(lib, "testno", ""))
        for lib in libraries
        if _normalize_text_for_match(getattr(lib, "test_no", "") or getattr(lib, "testno", ""))
    }
    if lane_test_nos == {_normalize_text_for_match("Novaseq X Plus-PE150")} and normalized_loading_method in {
        "10B",
        "25B",
    }:
        return "P7P5"

    if matched_rules:
        resolved_rule = sorted(matched_rules)[0]
        logger.warning(
            "Lane排机规则存在多个候选，使用排序后首个值: loading_method={}, rules={}",
            loading_method,
            sorted(matched_rules),
        )
        return resolved_rule

    return ""


def _get_lib_attr_float(
    lib: EnhancedLibraryInfo,
    attr_names: List[str],
    default: Optional[float] = None,
) -> Optional[float]:
    """按候选属性名顺序读取文库浮点值。"""
    for attr_name in attr_names:
        value = getattr(lib, attr_name, None)
        if value is None:
            continue
        try:
            value_float = float(value)
            if pd.isna(value_float):
                continue
            return value_float
        except (TypeError, ValueError):
            continue
    return default


def _get_row_attr_float(
    row: pd.Series,
    column_names: List[str],
    default: Optional[float] = None,
) -> Optional[float]:
    """按候选列名顺序读取DataFrame行中的浮点值。"""
    for column_name in column_names:
        if column_name not in row.index:
            continue
        value = row[column_name]
        if value is None:
            continue
        try:
            value_float = float(value)
            if pd.isna(value_float):
                continue
            return value_float
        except (TypeError, ValueError):
            continue
    return default


def _get_row_attr_text(
    row: pd.Series,
    column_names: List[str],
    default: str = "",
) -> str:
    """按候选列名顺序读取DataFrame行中的文本值。"""
    for column_name in column_names:
        if column_name not in row.index:
            continue
        value = row[column_name]
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return text
    return default


def _normalize_rate_to_decimal(rate_value: Optional[float]) -> Optional[float]:
    """将产出率统一换算为小数；百分数口径如50会转换为0.5。"""
    if rate_value is None:
        return None
    try:
        rate_float = float(rate_value)
    except (TypeError, ValueError):
        return None
    if pd.isna(rate_float):
        return None
    if rate_float > 1.0:
        rate_float = rate_float / 100.0
    return rate_float


def _resolve_historical_outrate(
    last_outrate: Optional[float],
    last_output: Optional[float],
    last_order: Optional[float],
) -> Optional[float]:
    """解析历史产出率，优先使用上一轮产出/上一轮下单反算，缺失时回退到显式字段。"""
    if last_output is not None and last_order is not None:
        try:
            output_value = float(last_output)
            order_value = float(last_order)
        except (TypeError, ValueError):
            output_value = None
            order_value = None
        if (
            output_value is not None
            and order_value is not None
            and not pd.isna(output_value)
            and not pd.isna(order_value)
            and order_value > 0
        ):
            derived_outrate = output_value / order_value
            if derived_outrate > 0:
                return derived_outrate

    if last_outrate is None:
        return None
    try:
        outrate = _normalize_rate_to_decimal(float(last_outrate))
        if not pd.isna(outrate) and outrate > 0:
            return outrate
    except (TypeError, ValueError):
        return None
    return None


def _is_add_test_library(lib: EnhancedLibraryInfo) -> bool:
    """判断是否为加测文库。"""
    remark = str(getattr(lib, "add_tests_remark", "") or "").strip()
    return "加测" in remark


def _apply_add_test_output_rate_rule(
    lib: EnhancedLibraryInfo,
    ai_predicted_order: Optional[float],
    ai_predicted_output: Optional[float],
    contract_data: float,
) -> Dict[str, Any]:
    """加测产出率规则：在AI预测后，对下单量进行二次修正。"""
    result: Dict[str, Any] = {
        "applied": False,
        "rule_reason": "not_add_test",
        "selected_order": ai_predicted_order,
        "selected_output": ai_predicted_output,
        "ai_predicted_order": ai_predicted_order,
        "ai_predicted_output": ai_predicted_output,
        "qpcr_within_15pct": None,
        "qpcr_deviation_ratio": None,
        "historical_based_order": None,
        "effective_last_outrate": None,
        "wklastqpcr": None,
        "wklastorderdata": None,
        "wklastoutput": None,
        "wklastoutrate": None,
    }
    if not _is_add_test_library(lib):
        return result

    result["applied"] = True
    current_qpcr = _get_lib_attr_float(lib, ["qpcr_molar", "qpcr_concentration"])
    last_qpcr = _get_lib_attr_float(lib, ["_last_qpcr_raw", "last_qpcr", "wklastqpcr", "wklistqpcr"])
    last_outrate = _get_lib_attr_float(lib, ["_last_outrate_raw", "last_outrate", "wklastoutrate"])
    last_order = _get_lib_attr_float(lib, ["_last_order_data_raw", "last_order_data", "wklastorderdata"])
    last_output = _get_lib_attr_float(lib, ["_last_output_raw", "last_output", "wklastoutput"])
    historical_outrate = _resolve_historical_outrate(
        last_outrate=last_outrate,
        last_output=last_output,
        last_order=last_order,
    )

    result["wklastqpcr"] = last_qpcr
    result["wklastorderdata"] = last_order
    result["wklastoutput"] = last_output
    result["wklastoutrate"] = historical_outrate

    if ai_predicted_order is None:
        result["rule_reason"] = "ai_order_missing"
        return result

    qpcr_within = False
    qpcr_deviation_ratio: Optional[float] = None
    if current_qpcr is not None and last_qpcr is not None and last_qpcr > 0:
        qpcr_deviation_ratio = abs(current_qpcr - last_qpcr) / last_qpcr
        qpcr_within = qpcr_deviation_ratio <= 0.15
    result["qpcr_within_15pct"] = qpcr_within
    result["qpcr_deviation_ratio"] = qpcr_deviation_ratio

    if not qpcr_within:
        # QPCR偏差超阈值时，完全采用AI预测值。
        result["rule_reason"] = "qpcr_outside_15pct_use_ai"
        return result

    if historical_outrate is None:
        result["rule_reason"] = "historical_outrate_missing_use_ai"
        return result

    effective_last_outrate = max(historical_outrate, 0.3)
    result["effective_last_outrate"] = effective_last_outrate
    historical_based_order = contract_data / effective_last_outrate if effective_last_outrate > 0 else ai_predicted_order
    result["historical_based_order"] = historical_based_order

    selected_order = max(ai_predicted_order, historical_based_order)
    result["selected_order"] = selected_order

    # 该规则仅修正下单量，不修正产出量，产出量始终保持AI预测结果。
    result["selected_output"] = ai_predicted_output
    result["rule_reason"] = "qpcr_within_15pct_compare_ai_vs_historical"
    return result


def _apply_add_test_output_rate_rule_to_prediction_df(
    prediction_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """对prediction_delivery结果应用加测产出率规则。"""
    if prediction_df is None or prediction_df.empty:
        return prediction_df

    df = prediction_df.copy()

    applied_count = 0
    override_count = 0

    for idx, row in df.iterrows():
        remark = _get_row_attr_text(row, ["wkaddtestsremark", "addtestsremark"])
        if "加测" not in remark:
            continue

        ai_order = _get_row_attr_float(
            row,
            ["lorderdata", "ai_predicted_lorderdata", "predicted_lorderdata"],
        )
        current_qpcr = _get_row_attr_float(row, ["wkqpcr", "qpcrmolar", "qpcr_molar"])
        last_qpcr = _get_row_attr_float(row, ["wklastqpcr", "wklistqpcr"])
        last_outrate = _get_row_attr_float(row, ["wklastoutrate"])
        last_output = _get_row_attr_float(row, ["wklastoutput"])
        last_order = _get_row_attr_float(row, ["wklastorderdata"])
        add_test_output_rate = _normalize_rate_to_decimal(
            _get_row_attr_float(row, ["wkoutputrate", "outputrate", "output_rate"])
        )
        contract_data = _get_row_attr_float(row, ["wkcontractdata", "contractdata", "wkcontractdata_raw"])

        if ai_order is None or contract_data is None or contract_data <= 0:
            continue

        selected_order = ai_order
        rule_applied = False

        if current_qpcr is not None and last_qpcr is not None and last_qpcr > 0:
            rule_applied = True
            qpcr_deviation_ratio = abs(current_qpcr - last_qpcr) / last_qpcr
            qpcr_within = qpcr_deviation_ratio <= 0.15

            if qpcr_within:
                historical_outrate = _resolve_historical_outrate(
                    last_outrate=last_outrate,
                    last_output=last_output,
                    last_order=last_order,
                )
                if historical_outrate is not None:
                    effective_last_outrate = max(historical_outrate, 0.3)
                    historical_based_order = contract_data / effective_last_outrate
                    selected_order = max(selected_order, historical_based_order)

        if add_test_output_rate is not None:
            rule_applied = True
            effective_add_test_output_rate = max(add_test_output_rate, 0.3)
            add_test_rate_based_order = contract_data / effective_add_test_output_rate
            selected_order = max(selected_order, add_test_rate_based_order)

        if not rule_applied:
            continue

        applied_count += 1
        if selected_order > ai_order:
            override_count += 1
        rounded_order = round(float(selected_order), 6)
        df.at[idx, "lorderdata"] = rounded_order
        if "predicted_lorderdata" in df.columns:
            df.at[idx, "predicted_lorderdata"] = rounded_order

    logger.info(
        "加测产出率规则应用完成: 评估{}条，覆盖{}条".format(
            applied_count, override_count
        )
    )

    df = df.drop(
        columns=[
            "ai_predicted_lorderdata",
            "ai_predicted_lai_output",
            "add_test_rule_applied",
            "add_test_rule_reason",
            "qpcr_within_15pct",
            "qpcr_deviation_ratio",
            "historical_based_lorderdata",
            "effective_last_outrate",
        ],
        errors="ignore",
    )

    if output_path is not None:
        df.to_csv(output_path, index=False)
        logger.info(f"已写回加测产出率修正结果: {output_path}")

    return df


def _resolve_mode_1_1_sample_prefix_from_row(row: pd.Series) -> str:
    """从输出行中解析样本编号前缀，优先读显式字段，缺失时回退到 sample_id 前四位。"""
    prefix = _normalize_text_for_match(
        _get_row_attr_text(row, ["wksample_number", "sample_number_prefix"])
    )
    if prefix:
        return prefix

    sample_id = _normalize_text_for_match(_get_row_attr_text(row, ["wksampleid", "sample_id"]))
    if not sample_id:
        return ""
    return sample_id[:4]


def _apply_mode_1_1_first_round_order_halving_to_prediction_df(
    prediction_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """对1.1首轮普通文库应用“模型正常预测后下单量除2”规则。"""
    if prediction_df is None or prediction_df.empty:
        return prediction_df

    df = prediction_df.copy()
    mode_1_1_config = get_scheduling_config().get_mode_1_1_config()
    halving_cfg = dict(mode_1_1_config.get("first_round_order_halving", {}) or {})
    if not bool(halving_cfg.get("enabled", False)):
        return df

    divisor = float(halving_cfg.get("divisor", 0) or 0)
    if divisor <= 0:
        return df

    first_round_label = str(mode_1_1_config.get("first_round_label", "1.1第一轮"))
    excluded_data_types = {
        str(item).strip()
        for item in (halving_cfg.get("excluded_data_types", []) or [])
        if str(item).strip()
    }
    excluded_prefixes = {
        _normalize_text_for_match(item)
        for item in (halving_cfg.get("excluded_sample_prefixes", []) or [])
        if _normalize_text_for_match(item)
    }
    excluded_add_test_keywords = {
        str(item).strip()
        for item in (halving_cfg.get("excluded_add_test_keywords", []) or [])
        if str(item).strip()
    }

    applied_count = 0
    balance_marker = None
    if BALANCE_LIBRARY_MARKER_COLUMN in df.columns:
        balance_marker = (
            df[BALANCE_LIBRARY_MARKER_COLUMN]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"})
        )
    for idx, row in df.iterrows():
        lane_round = _get_row_attr_text(row, ["laneround", "resolved_round_label"])
        if lane_round != first_round_label:
            continue

        if balance_marker is not None and bool(balance_marker.loc[idx]):
            continue

        current_order = _get_row_attr_float(row, ["lorderdata", "predicted_lorderdata"])
        if current_order is None or current_order <= 0:
            continue

        data_type = _get_row_attr_text(row, ["wkdatatype", "datatype"])
        if data_type in excluded_data_types:
            continue

        sample_prefix = _resolve_mode_1_1_sample_prefix_from_row(row)
        if sample_prefix and any(sample_prefix.startswith(prefix) for prefix in excluded_prefixes):
            continue

        add_test_remark = _get_row_attr_text(row, ["wkaddtestsremark", "addtestsremark"])
        if any(keyword and keyword in add_test_remark for keyword in excluded_add_test_keywords):
            continue

        halved_order = round(float(current_order) / divisor, 6)
        df.at[idx, "lorderdata"] = halved_order
        if "predicted_lorderdata" in df.columns:
            df.at[idx, "predicted_lorderdata"] = halved_order
        applied_count += 1

    logger.info("1.1首轮普通文库下单量除2规则应用完成: 覆盖{}条".format(applied_count))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"已写回1.1首轮下单量除2结果: {output_path}")

    return df


def _apply_mode_1_1_first_round_balance_rule_to_prediction_df(
    prediction_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """对1.1首轮平衡文库应用“合同量保持、下单量除2”规则。"""
    if prediction_df is None or prediction_df.empty:
        return prediction_df

    df = prediction_df.copy()
    if BALANCE_LIBRARY_MARKER_COLUMN not in df.columns:
        return df

    mode_1_1_config = get_scheduling_config().get_mode_1_1_config()
    first_round_label = str(mode_1_1_config.get("first_round_label", "1.1第一轮"))
    marker = (
        df[BALANCE_LIBRARY_MARKER_COLUMN]
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    if not marker.any():
        return df

    applied_count = 0
    for idx, row in df.iterrows():
        if not bool(marker.loc[idx]):
            continue

        lane_round = _get_row_attr_text(row, ["laneround", "resolved_round_label"])
        if lane_round != first_round_label:
            continue

        contract_data = _get_row_attr_float(row, ["wkcontractdata", "contractdata"])
        if contract_data is None or contract_data <= 0:
            continue

        halved_order = round(float(contract_data) / 2.0, 6)
        df.at[idx, "wkcontractdata"] = round(float(contract_data), 6)
        df.at[idx, "lorderdata"] = halved_order
        if "predicted_lorderdata" in df.columns:
            df.at[idx, "predicted_lorderdata"] = halved_order
        if "lai_output" in df.columns:
            df.at[idx, "lai_output"] = pd.NA
        applied_count += 1

    logger.info("1.1首轮平衡文库规则应用完成: 覆盖{}条".format(applied_count))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"已写回1.1首轮平衡文库结果: {output_path}")

    return df


def _apply_mode_1_1_round2_historical_order_rule_to_prediction_df(
    prediction_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """对1.1第二轮普通样本按 wkoutputrate 重算下单量。"""
    if prediction_df is None or prediction_df.empty:
        return prediction_df

    df = prediction_df.copy()
    mode_1_1_config = get_scheduling_config().get_mode_1_1_config()
    second_round_label = str(mode_1_1_config.get("second_round_label", "1.1第二轮"))

    balance_marker = None
    if BALANCE_LIBRARY_MARKER_COLUMN in df.columns:
        balance_marker = (
            df[BALANCE_LIBRARY_MARKER_COLUMN]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"})
        )

    applied_count = 0
    missing_history_count = 0

    for idx, row in df.iterrows():
        lane_round = _get_row_attr_text(row, ["laneround"])
        if lane_round != second_round_label:
            continue

        if balance_marker is not None and bool(balance_marker.loc[idx]):
            continue

        pooling_factor = _get_row_attr_float(row, ["resolved_round2_pooling_factor"])
        if pooling_factor is not None and pooling_factor > 0:
            continue

        contract_data = _get_row_attr_float(row, ["wkcontractdata", "contractdata"])
        if contract_data is None or contract_data <= 0:
            continue

        output_rate = _normalize_rate_to_decimal(
            _get_row_attr_float(row, ["wkoutputrate", "outputrate", "output_rate"])
        )
        if output_rate is None or output_rate <= 0:
            missing_history_count += 1
            continue

        rounded_order = round(float(contract_data) / float(output_rate), 6)
        df.at[idx, "lorderdata"] = rounded_order
        if "predicted_lorderdata" in df.columns:
            df.at[idx, "predicted_lorderdata"] = rounded_order
        applied_count += 1

    logger.info(
        "1.1第二轮wkoutputrate规则应用完成: 覆盖{}条, wkoutputrate缺失{}条".format(
            applied_count, missing_history_count
        )
    )

    if output_path is not None:
        df.to_csv(output_path, index=False)
        logger.info(f"已写回1.1第二轮wkoutputrate修正结果: {output_path}")

    return df


def _apply_mode_1_1_round2_pooling_rule_to_prediction_df(
    prediction_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """对1.1第二轮低产出率文库应用默认pooling系数规则。"""
    if prediction_df is None or prediction_df.empty:
        return prediction_df

    df = prediction_df.copy()
    mode_1_1_config = get_scheduling_config().get_mode_1_1_config()
    second_round_label = str(mode_1_1_config.get("second_round_label", "1.1第二轮"))

    applied_count = 0
    for idx, row in df.iterrows():
        lane_round = _get_row_attr_text(row, ["laneround"])
        if lane_round != second_round_label:
            continue

        pooling_factor = _get_row_attr_float(row, ["resolved_round2_pooling_factor"])
        contract_data = _get_row_attr_float(row, ["wkcontractdata", "contractdata"])
        if pooling_factor is None or pooling_factor <= 0 or contract_data is None or contract_data <= 0:
            continue

        rounded_order = round(float(contract_data) * float(pooling_factor), 6)
        df.at[idx, "lorderdata"] = rounded_order
        if "predicted_lorderdata" in df.columns:
            df.at[idx, "predicted_lorderdata"] = rounded_order
        applied_count += 1

    df = df.drop(columns=["resolved_round2_pooling_factor"], errors="ignore")
    logger.info("1.1第二轮pooling规则应用完成: 覆盖{}条".format(applied_count))

    if output_path is not None:
        df.to_csv(output_path, index=False)
        logger.info(f"已写回1.1第二轮pooling修正结果: {output_path}")

    return df


def _resolve_mode_1_1_round2_order_for_balance_row(row: pd.Series) -> Optional[float]:
    """为第二轮平衡文库占比计算解析普通文库的下单量。

    计算优先级：
    1. 命中第二轮默认 pooling 系数时，按 `合同量 * pooling系数`
    2. 存在加测产出率时，按 `合同量 / (加测产出率 / 100)`
    3. 否则回退到当前 `lorderdata`

    这里的值只用于得到 lane 级总下单量，不直接覆盖普通文库展示字段。
    """
    contract_data = _get_row_attr_float(row, ["wkcontractdata", "contractdata"])
    if contract_data is None or contract_data <= 0:
        return None

    pooling_factor = _get_row_attr_float(row, ["resolved_round2_pooling_factor"])
    if pooling_factor is not None and pooling_factor > 0:
        return round(float(contract_data) * float(pooling_factor), 6)

    add_test_output_rate = _normalize_rate_to_decimal(
        _get_row_attr_float(row, ["wkoutputrate", "outputrate", "output_rate"])
    )
    if add_test_output_rate is not None and add_test_output_rate > 0:
        return round(float(contract_data) / float(add_test_output_rate), 6)

    return None


def _apply_mode_1_1_round2_balance_rule_to_prediction_df(
    prediction_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """按1.1第二轮规则回写平衡文库合同量/下单量。"""
    if prediction_df is None or prediction_df.empty:
        return prediction_df

    df = prediction_df.copy()
    if "resolved_round2_balance_ratio" not in df.columns:
        return df
    if BALANCE_LIBRARY_MARKER_COLUMN not in df.columns:
        df = df.drop(columns=["resolved_round2_balance_ratio"], errors="ignore")
        return df

    mode_1_1_config = get_scheduling_config().get_mode_1_1_config()
    second_round_label = str(mode_1_1_config.get("second_round_label", "1.1第二轮"))
    marker = df[BALANCE_LIBRARY_MARKER_COLUMN].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    if not marker.any():
        df = df.drop(columns=["resolved_round2_balance_ratio"], errors="ignore")
        return df

    applied_lane_count = 0
    lane_id_series = df.get("llaneid")
    if lane_id_series is None:
        lane_id_series = df.get("laneid")
    if lane_id_series is None:
        df = df.drop(columns=["resolved_round2_balance_ratio"], errors="ignore")
        return df

    for lane_id in lane_id_series.fillna("").astype(str).unique():
        lane_id = str(lane_id).strip()
        if not lane_id:
            continue

        lane_mask = lane_id_series.fillna("").astype(str).eq(lane_id)
        balance_mask = lane_mask & marker
        if not balance_mask.any():
            continue

        lane_round_values = df.loc[lane_mask, "laneround"] if "laneround" in df.columns else pd.Series(dtype=object)
        if second_round_label not in {str(item).strip() for item in lane_round_values.dropna().tolist()}:
            continue

        ratio_series = pd.to_numeric(
            df.loc[lane_mask, "resolved_round2_balance_ratio"] if "resolved_round2_balance_ratio" in df.columns else pd.Series(dtype=float),
            errors="coerce",
        ).dropna()
        if ratio_series.empty:
            continue

        balance_ratio = float(ratio_series.max())
        if balance_ratio <= 0:
            continue

        non_balance_order_sum = 0.0
        non_balance_rows = df.loc[lane_mask & (~marker)]
        for _, non_balance_row in non_balance_rows.iterrows():
            resolved_order = _resolve_mode_1_1_round2_order_for_balance_row(non_balance_row)
            if resolved_order is None or resolved_order <= 0:
                continue
            non_balance_order_sum += float(resolved_order)
        if non_balance_order_sum <= 0:
            continue

        denominator = 1.0 - balance_ratio
        if denominator <= 0:
            continue

        lane_total_order = non_balance_order_sum / denominator
        balance_amount = round(lane_total_order * balance_ratio, 6)
        balance_row_count = int(balance_mask.sum())
        if balance_row_count <= 0:
            continue
        per_row_balance_amount = round(balance_amount / balance_row_count, 6)

        df.loc[balance_mask, "wkcontractdata"] = per_row_balance_amount
        df.loc[balance_mask, "lorderdata"] = per_row_balance_amount
        if "predicted_lorderdata" in df.columns:
            df.loc[balance_mask, "predicted_lorderdata"] = per_row_balance_amount
        if "lai_output" in df.columns:
            df.loc[balance_mask, "lai_output"] = pd.NA
        applied_lane_count += 1

    df = df.drop(columns=["resolved_round2_balance_ratio"], errors="ignore")
    logger.info("1.1第二轮平衡文库规则应用完成: 覆盖{}条Lane".format(applied_lane_count))

    if output_path is not None:
        df.to_csv(output_path, index=False)
        logger.info(f"已写回1.1第二轮平衡文库修正结果: {output_path}")

    return df


def _build_origrec_key(df: pd.DataFrame) -> pd.Series:
    """构建origrec唯一键"""
    keys: List[str] = []
    for idx, row in df.iterrows():
        raw = _safe_str(row.get("wkorigrec", row.get("origrec", "")))
        lane_unique = _safe_str(row.get("lane_unique_id", row.get("lane_unique", "")))
        llaneid = _safe_str(row.get("llaneid", ""))
        key = raw or lane_unique or llaneid or f"LIB_{idx}"
        keys.append(key)
    return pd.Series(keys, index=df.index)


def _build_runid_by_lane(
    lanes: List[LaneAssignment], lanes_per_run: int = 8
) -> Dict[str, str]:
    """为Lane生成runid映射。

    普通场景每个run最多包含指定数量的Lane；包含拆分文库时，把通过
    共享Lane直接或间接连接的拆分家族涉及的Lane作为不可拆组件，避免
    同一连通组件跨run。
    """
    if lanes_per_run <= 0:
        raise ValueError("lanes_per_run必须大于0")
    runid_by_lane: Dict[str, str] = {}
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if not lanes:
        return runid_by_lane

    lane_count = len(lanes)
    family_indices: Dict[str, List[int]] = {}
    family_expected_counts: Dict[str, int] = {}
    for idx, lane in enumerate(lanes):
        for lib in list(getattr(lane, "libraries", []) or []):
            if not _is_split_library(lib):
                continue
            family_id = _get_split_family_id_for_lane_build(lib)
            if not family_id:
                continue
            family_indices.setdefault(family_id, []).append(idx)
            family_expected_counts[family_id] = max(
                int(family_expected_counts.get(family_id, 0) or 0),
                int(getattr(lib, "total_fragments", 0) or 0),
            )
    for family_id, indices in family_indices.items():
        if int(family_expected_counts.get(family_id, 0) or 0) <= 1:
            family_expected_counts[family_id] = len(indices)
    max_split_count = max(family_expected_counts.values(), default=0)
    max_split_family_ids = {
        family_id
        for family_id, expected_count in family_expected_counts.items()
        if expected_count == max_split_count and expected_count > 1
    }

    split_lane_indices: Set[int] = {
        idx
        for indices in family_indices.values()
        for idx in set(indices)
    }
    parent: Dict[int, int] = {idx: idx for idx in split_lane_indices}

    def find(idx: int) -> int:
        root = parent[idx]
        if root != idx:
            parent[idx] = find(root)
        return parent[idx]

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    for indices in family_indices.values():
        unique_indices = sorted(set(indices))
        if not unique_indices:
            continue
        first_index = unique_indices[0]
        for idx in unique_indices[1:]:
            union(first_index, idx)

    component_by_root: Dict[int, Set[int]] = {}
    for idx in split_lane_indices:
        component_by_root.setdefault(find(idx), set()).add(idx)

    component_family_ids: Dict[int, Set[str]] = {
        root: set() for root in component_by_root
    }
    for family_id, indices in family_indices.items():
        unique_indices = sorted(set(indices))
        if not unique_indices:
            continue
        root = find(unique_indices[0])
        component_family_ids.setdefault(root, set()).add(family_id)

    split_components = [
        sorted(indices)
        for _, indices in sorted(
            component_by_root.items(),
            key=lambda item: (
                not bool(component_family_ids.get(item[0], set()) & max_split_family_ids),
                -max(
                    (
                        int(family_expected_counts.get(family_id, 0) or 0)
                        for family_id in component_family_ids.get(item[0], set())
                    ),
                    default=0,
                ),
                -len(item[1]),
                min(item[1]),
            ),
        )
    ]

    assigned_indices: Set[int] = {
        idx
        for component in split_components
        for idx in component
    }
    non_split_indices: List[int] = [
        idx for idx in range(lane_count)
        if idx not in assigned_indices
    ]

    oversized_components = [item for item in split_components if len(item) > lanes_per_run]
    if oversized_components:
        logger.warning(
            "runid分组发现{}个拆分组件超过{}条Lane，按拆分原子性规则保持同一runid，不按run容量切分",
            len(oversized_components),
            lanes_per_run,
        )

    run_groups: List[List[int]] = []

    # 每个run最多承载一个拆分连通组件；剩余位置只允许用非拆分lane填充。
    # run不要求凑满8条，避免把没有直接/间接交集的拆分lane放进同一run。
    for component in split_components:
        group = list(component)
        filler_count = min(lanes_per_run - len(group), len(non_split_indices))
        if filler_count > 0:
            group.extend(non_split_indices[:filler_count])
            non_split_indices = non_split_indices[filler_count:]
        run_groups.append(sorted(group))

    for start in range(0, len(non_split_indices), lanes_per_run):
        run_groups.append(non_split_indices[start:start + lanes_per_run])

    for run_index, group in enumerate(run_groups, start=1):
        for idx in group:
            runid_by_lane[lanes[idx].lane_id] = f"RUN_{timestamp}_{run_index:03d}"
    return runid_by_lane


def _partition_remaining_package_libraries(
    libraries: List[EnhancedLibraryInfo],
) -> Tuple[List[EnhancedLibraryInfo], List[EnhancedLibraryInfo]]:
    """拆分包Lane失败文库与可回流普通排机的文库。

    包Lane失败后的文库必须保持失败状态，不允许回流普通排机混排，也不允许拆分。
    这里额外兜底区分，避免后续流程误把仍带包Lane编号的文库塞回 normal_libs。
    """
    failed_package_libraries: List[EnhancedLibraryInfo] = []
    remaining_normal_libraries: List[EnhancedLibraryInfo] = []

    for lib in libraries:
        baleno = _get_package_lane_number_from_library(lib)
        if baleno:
            lib.package_lane_number = baleno
            lib.baleno = baleno
            lib.is_package_lane = "是"
            failed_package_libraries.append(lib)
        else:
            remaining_normal_libraries.append(lib)

    return failed_package_libraries, remaining_normal_libraries


def _build_lane_metadata_for_validator(
    lane_id: str,
    lane_metadata: Optional[Dict[str, Any]] = None,
    libraries: Optional[List[EnhancedLibraryInfo]] = None,
) -> Dict[str, Any]:
    """根据Lane前缀构建验证所需的metadata"""
    metadata: Dict[str, Any] = {}
    if _is_dedicated_imbalance_lane_context(
        libraries=list(libraries or []),
        lane_id=lane_id,
        lane_metadata=lane_metadata,
    ):
        metadata["is_dedicated_imbalance_lane"] = True
    if lane_id.startswith("NB_"):
        metadata["is_pure_non_10bp_lane"] = True
    if lane_id.startswith("BL_"):
        metadata["is_backbone_lane"] = True
    if lane_id.startswith(f"{LANE_SEQ_10_PLUS_24_LANE_PREFIX}_"):
        metadata["is_lane_seq_10_plus_24_lane"] = True
        metadata["mode"] = "lane_seq"
        metadata["seq_mode"] = "Lane seq"
        metadata["seq_strategy"] = "10+24"
    if lane_metadata:
        capacity_rule_code = str(lane_metadata.get("capacity_rule_code") or "").strip()
        mode_locked_by_capacity_rule = False
        if capacity_rule_code == "tj_1595_standard_pe150_25b":
            metadata["seq_mode"] = "3.6T-NEW"
            metadata["lcxms"] = "3.6T-NEW"
            metadata["selected_seq_mode"] = "3.6T-NEW"
            mode_locked_by_capacity_rule = True
        elif capacity_rule_code.startswith("tj_1595_mode_1_1"):
            metadata["seq_mode"] = "1.1"
            metadata["lcxms"] = "1.1"
            metadata["selected_seq_mode"] = "1.1"
            mode_locked_by_capacity_rule = True
        if not mode_locked_by_capacity_rule:
            for mode_key in ("selected_seq_mode", "seq_mode", "lcxms", "sequencing_mode"):
                mode_value = lane_metadata.get(mode_key)
                if mode_value:
                    metadata["seq_mode"] = str(mode_value).strip()
                    metadata["lcxms"] = str(mode_value).strip()
                    metadata["selected_seq_mode"] = str(mode_value).strip()
                    break
        if lane_metadata.get("is_package_lane"):
            metadata["is_package_lane"] = True
        if lane_metadata.get("is_dedicated_imbalance_lane"):
            metadata["is_dedicated_imbalance_lane"] = True
        if lane_metadata.get("is_pure_non_10bp_lane"):
            metadata["is_pure_non_10bp_lane"] = True
        if lane_metadata.get("is_backbone_lane"):
            metadata["is_backbone_lane"] = True
        if lane_metadata.get("is_lane_seq_10_plus_24_lane"):
            metadata["is_lane_seq_10_plus_24_lane"] = True
            metadata["mode"] = "lane_seq"
            metadata["seq_mode"] = "Lane seq"
            metadata["seq_strategy"] = "10+24"
        if not lane_metadata.get("materialized_balance_library"):
            balance_data = lane_metadata.get("wkbalancedata")
            if balance_data is None:
                balance_data = lane_metadata.get("wkadd_balance_data")
            if balance_data is None:
                balance_data = lane_metadata.get("required_balance_data_gb")
            if balance_data is not None:
                metadata["wkbalancedata"] = float(balance_data)
    return metadata


def _is_customer_like_validator(lib: EnhancedLibraryInfo) -> bool:
    """按LaneValidator的口径识别客户文库"""
    customer_flag = str(getattr(lib, "customer_library", "") or "").strip()
    if customer_flag in {"是", "Y", "YES", "TRUE", "客户"}:
        return True
    sampletype = getattr(lib, "sampletype", "") or getattr(lib, "sample_type_code", "") or ""
    sample_id = getattr(lib, "sample_id", "") or ""
    if str(sampletype).startswith("客户") or str(sample_id).startswith("FKDL"):
        return True
    if hasattr(lib, "is_customer_library") and callable(lib.is_customer_library):
        return bool(lib.is_customer_library())
    return False


def _split_customer_and_non_customer(
    libraries: List[EnhancedLibraryInfo],
) -> tuple[List[EnhancedLibraryInfo], List[EnhancedLibraryInfo]]:
    """按LaneValidator口径拆分客户/非客户文库"""
    customers: List[EnhancedLibraryInfo] = []
    non_customers: List[EnhancedLibraryInfo] = []
    for lib in libraries:
        if _is_customer_like_validator(lib):
            customers.append(lib)
        else:
            non_customers.append(lib)
    return customers, non_customers


def _normalize_wkjkhj_for_mix(value: Any) -> str:
    """归一化建库环节字段，用于手工/客户侧与产线侧混合规则。"""
    if value is None:
        return ""
    return str(value).strip().replace("＋", "+").replace("×", "X").upper()


def _is_manual_or_customer_side_for_mix(lib: EnhancedLibraryInfo) -> bool:
    """按LaneValidator新增混合规则口径识别手工/客户侧。"""
    wkjkhj = _normalize_wkjkhj_for_mix(getattr(lib, "wkjkhj", "") or "")
    return wkjkhj in {"客户自建", "诺禾手工"}


def _is_production_side_for_mix(lib: EnhancedLibraryInfo) -> bool:
    """按LaneValidator新增混合规则口径识别产线侧。"""
    wkjkhj = _normalize_wkjkhj_for_mix(getattr(lib, "wkjkhj", "") or "")
    return wkjkhj == "诺禾自动"


def _split_manual_customer_and_production_for_mix(
    libraries: List[EnhancedLibraryInfo],
) -> tuple[List[EnhancedLibraryInfo], List[EnhancedLibraryInfo]]:
    """拆分手工/客户侧与产线侧文库，AI平衡文库不参与占比。"""
    manual_or_customer: List[EnhancedLibraryInfo] = []
    production: List[EnhancedLibraryInfo] = []
    for lib in libraries:
        if bool(getattr(lib, "_is_ai_balance_library", False)):
            continue
        if _is_manual_or_customer_side_for_mix(lib):
            manual_or_customer.append(lib)
        elif _is_production_side_for_mix(lib):
            production.append(lib)
    return manual_or_customer, production


def _is_same_mix_side_as(reference_libs: List[EnhancedLibraryInfo]):
    """生成与参考文库同侧的选择器，用于从未分配池补同侧文库。"""
    has_manual_customer = any(_is_manual_or_customer_side_for_mix(lib) for lib in reference_libs)
    has_production = any(_is_production_side_for_mix(lib) for lib in reference_libs)
    if has_manual_customer and not has_production:
        return _is_manual_or_customer_side_for_mix
    if has_production and not has_manual_customer:
        return _is_production_side_for_mix
    return lambda lib: False


def _split_10bp_and_non_10bp(
    libraries: List[EnhancedLibraryInfo], validator
) -> tuple[List[EnhancedLibraryInfo], List[EnhancedLibraryInfo]]:
    """按10bp/非10bp拆分文库列表（与校验逻辑一致）"""
    libs_10bp: List[EnhancedLibraryInfo] = []
    libs_non_10bp: List[EnhancedLibraryInfo] = []
    for lib in libraries:
        ten_bp_data = getattr(lib, "ten_bp_data", None)
        if ten_bp_data is not None and ten_bp_data > 0:
            is_10bp = True
        else:
            index_seq = getattr(lib, "index_seq", "") or ""
            is_10bp = validator._is_10bp_index(index_seq)
        if is_10bp:
            libs_10bp.append(lib)
        else:
            libs_non_10bp.append(lib)
    return libs_10bp, libs_non_10bp


def _infer_terminal_lane_constraint_metadata(
    libraries: List[EnhancedLibraryInfo],
    validator: Any,
) -> Dict[str, Any]:
    """按候选Lane实际组成补充专用Lane校验标记。"""
    real_libraries = [lib for lib in list(libraries or []) if not _is_ai_balance_library(lib)]
    if not real_libraries:
        return {}

    inferred: Dict[str, Any] = {}
    libs_10bp, libs_non_10bp = _split_10bp_and_non_10bp(real_libraries, validator)
    if libs_non_10bp and not libs_10bp:
        inferred["is_pure_non_10bp_lane"] = True
    if all(
        bool(getattr(validator, "_is_base_imbalance_library", lambda _: False)(lib))
        for lib in real_libraries
    ):
        inferred["is_dedicated_imbalance_lane"] = True
    return inferred


def _validate_lane_state(
    validator: Any,
    lane: LaneAssignment,
    libraries: List[EnhancedLibraryInfo],
    balance_already_in_libs: bool = False,
    skip_peak_size: bool = False,
    skip_balance_injection_context_rules: bool = False,
) -> Any:
    """校验给定文库列表在当前Lane上下文中的合法性。

    balance_already_in_libs=True：libraries 已含平衡文库本体，去掉 metadata 里的
    wkbalancedata 避免容量校验器二次叠加。

    skip_peak_size=True：跳过 peak_size 错误/警告的判断。专用不均衡 lane 注入平衡文库
    时使用——该 lane 的 peak_size 分布是排机时就已形成的既成事实，平衡文库不应因此被阻止。
    """
    if _is_package_lane_assignment(lane):
        package_errors = _validate_package_lane_rules(lane, libraries=libraries)
        return LaneValidationResult(
            lane_id=lane.lane_id,
            is_valid=not package_errors,
            errors=[
                ValidationError(
                    rule_type=ValidationRuleType.CAPACITY,
                    severity=ValidationSeverity.ERROR,
                    message=message,
                )
                for message in package_errors
            ],
        )

    if _is_lane_seq_10_plus_24_lane_assignment(lane):
        lane_seq_errors = _validate_lane_seq_10_plus_24_rules(lane, libraries=libraries)
        return LaneValidationResult(
            lane_id=lane.lane_id,
            is_valid=not lane_seq_errors,
            errors=[
                ValidationError(
                    rule_type=ValidationRuleType.INDEX_CONFLICT,
                    severity=ValidationSeverity.ERROR,
                    message=message,
                )
                for message in lane_seq_errors
            ],
            warnings=[],
        )

    ai_lane_index_errors = _validate_ai_lane_index_pair_rules(lane, libraries=libraries)
    if ai_lane_index_errors:
        return LaneValidationResult(
            lane_id=lane.lane_id,
            is_valid=False,
            errors=[
                ValidationError(
                    rule_type=ValidationRuleType.INDEX_CONFLICT,
                    severity=ValidationSeverity.ERROR,
                    message=message,
                )
                for message in ai_lane_index_errors
            ],
            warnings=[],
        )

    if bool(getattr(lane, "metadata", {}).get("skip_strict_validation")):
        return LaneValidationResult(
            lane_id=lane.lane_id,
            is_valid=True,
            errors=[],
            warnings=[],
        )

    has_balance_library = any(_is_ai_balance_library(lib) for lib in libraries)
    metadata = _build_lane_metadata_for_validator(lane.lane_id, lane.metadata, libraries=libraries)
    if balance_already_in_libs:
        metadata.pop("wkbalancedata", None)
        metadata.pop("wkadd_balance_data", None)
        metadata.pop("required_balance_data_gb", None)
        if _is_explicit_dedicated_imbalance_lane(lane):
            # 候选平衡文库已真实加入 libraries，但专用不均衡lane在容量判定时
            # 仍需沿用“预留平衡容量”的窗口，而不是退回普通 lane 的裸容量下限。
            metadata["preserve_balance_reservation"] = True
    if skip_balance_injection_context_rules:
        metadata["skip_special_split_rule"] = True
    # 对专用不均衡 Lane，若 metadata 里没有显式平衡量，尝试从文库自身 balance_data 字段补全。
    # 这能修复一类情况：调度时 wkbalancedata 计算为 0 或记录缺失，导致有效容量被低估。
    if (
        _is_explicit_dedicated_imbalance_lane(lane)
        and not metadata.get("wkbalancedata")
        and not has_balance_library
        and not balance_already_in_libs
        and not (lane.metadata or {}).get("materialized_balance_library")
    ):
        raw_balance_vals = [
            float(getattr(lib, "balance_data", None) or 0)
            for lib in libraries
            if not _is_ai_balance_library(lib)
        ]
        top_balance = max(raw_balance_vals, default=0.0)
        if top_balance > 0:
            metadata["wkbalancedata"] = top_balance
            logger.debug(
                "DL_ lane {} metadata 中无 wkbalancedata，从文库 balance_data 补全: {:.1f}G",
                lane.lane_id,
                top_balance,
            )
    machine_type = lane.machine_type.value if lane.machine_type else "Nova X-25B"
    result = _validate_lane_with_latest_index(
        validator=validator,
        libraries=libraries,
        lane_id=lane.lane_id,
        machine_type=machine_type,
        metadata=metadata,
    )
    if (skip_peak_size or skip_balance_injection_context_rules) and not result.is_valid:
        # 平衡文库注入时，容量和Index冲突仍必须满足；单端占比、peak size、wkspecialsplits
        # 是原专用不均lane既有状态，不应因为补入平衡文库而阻止物化。
        skipped_rule_types = {ValidationRuleType.PEAK_SIZE}
        if skip_balance_injection_context_rules:
            skipped_rule_types.update(
                {
                    ValidationRuleType.SINGLE_END_RATIO,
                    ValidationRuleType.SPECIAL_LIBRARY_LIMIT,
                }
            )
        filtered_errors = [e for e in result.errors if e.rule_type not in skipped_rule_types]
        filtered_warnings = [w for w in result.warnings if w.rule_type not in skipped_rule_types]
        is_valid = len(filtered_errors) == 0 and (not validator.strict_mode or len(filtered_warnings) == 0)
        result = LaneValidationResult(
            lane_id=result.lane_id,
            is_valid=is_valid,
            errors=filtered_errors,
            warnings=filtered_warnings,
        )

    return result


def _get_lane_selected_mode(lane: LaneAssignment) -> str:
    """获取Lane选择的测序模式。"""
    metadata = dict(getattr(lane, "metadata", {}) or {})
    lane_mode = str(
        metadata.get("selected_seq_mode")
        or metadata.get("seq_mode")
        or metadata.get("lcxms")
        or metadata.get("sequencing_mode")
        or ""
    ).strip()
    if not lane_mode:
        lane_mode = str(metadata.get("mode", "") or "").strip()
    return lane_mode


def _is_split_lane_forbidden_by_mode(lane: LaneAssignment) -> bool:
    """判断Lane是否违反按规则应拆分文库的终态硬约束。"""
    if _is_mode_1_1_second_round_lane(lane):
        return False

    lane_libraries = [
        lib for lib in list(getattr(lane, "libraries", []) or [])
        if not _is_ai_balance_library(lib)
    ]
    split_libraries = [lib for lib in lane_libraries if _is_split_library(lib)]
    if _is_package_lane_assignment(lane):
        return False

    lane_mode = _get_lane_selected_mode(lane)
    if split_libraries and lane_mode != "3.6T-NEW":
        return True
    if lane_mode == "1.1":
        if any(_is_forbidden_in_mode_1_1_by_secondary_36t_policy(lib) for lib in lane_libraries):
            return True
        return any(
            (not _is_split_library(lib)) and _is_split_rule_original_blocked_from_1_1(lib)
            for lib in lane_libraries
        )
    if lane_mode == "3.6T-NEW":
        if any(
            (not _is_split_library(lib)) and _should_library_split_in_3_6t(lib)
            for lib in lane_libraries
        ):
            return True
        return any(
            _is_small_unsplit_original_reserved_for_mode_1_1(lib)
            for lib in lane_libraries
        )
    return False


def _is_capacity_shortage_only(result: LaneValidationResult) -> bool:
    """判断校验失败是否仅由容量不足导致。"""
    if result.is_valid:
        return False
    errors = list(getattr(result, "errors", []) or [])
    if not errors:
        return False
    shortage_tokens = ("容量不足", "低于", "不足", "未达到")
    for error in errors:
        message = str(getattr(error, "message", "") or "")
        if getattr(error, "rule_type", None) != ValidationRuleType.CAPACITY:
            return False
        if not any(token in message for token in shortage_tokens):
            return False
    return True


def _is_normal_replacement_library_for_split_repair(lib: EnhancedLibraryInfo) -> bool:
    """判断文库是否可作为1.1 Lane剔除拆分子文库后的普通补位候选。"""
    if _is_ai_balance_library(lib):
        return False
    if _is_split_library(lib):
        return False
    if _is_split_rollback_unassigned_only(lib):
        return False
    return True


def _repair_split_libraries_in_non_36t_lanes(solution: Any, validator: Any) -> Dict[str, int]:
    """剔除非3.6T Lane中的拆分子文库，并用普通未分配文库补位修复。"""
    stats = {
        "affected_lanes": 0,
        "removed_split_libraries": 0,
        "replacement_libraries": 0,
        "repaired_lanes": 0,
    }
    lanes = list(getattr(solution, "lane_assignments", []) or [])
    unassigned_pool = list(getattr(solution, "unassigned_libraries", []) or [])
    if not lanes:
        return stats

    replacement_pool = sorted(
        [lib for lib in unassigned_pool if _is_normal_replacement_library_for_split_repair(lib)],
        key=lambda item: _safe_float(getattr(item, "contract_data_raw", None), default=0.0),
        reverse=True,
    )
    replacement_pool_ids = {id(lib) for lib in replacement_pool}
    used_replacement_ids: Set[int] = set()
    removed_split_libraries: List[EnhancedLibraryInfo] = []

    for lane in lanes:
        if _is_package_lane_assignment(lane):
            continue
        if _get_lane_selected_mode(lane) == "3.6T-NEW":
            continue
        lane_libraries = list(getattr(lane, "libraries", []) or [])
        split_libraries = [lib for lib in lane_libraries if _is_split_library(lib)]
        if not split_libraries:
            continue

        stats["affected_lanes"] += 1
        stats["removed_split_libraries"] += len(split_libraries)
        split_ids = {id(lib) for lib in split_libraries}
        candidate_libraries = [lib for lib in lane_libraries if id(lib) not in split_ids]
        removed_split_libraries.extend(split_libraries)

        current_result = _validate_lane_state(validator, lane, candidate_libraries)
        selected_replacements: List[EnhancedLibraryInfo] = []
        lane_repaired = False
        if current_result.is_valid:
            lane.libraries = candidate_libraries
            lane.total_data_gb = sum(lib.get_data_amount_gb() for lib in candidate_libraries)
            stats["repaired_lanes"] += 1
            continue

        for candidate in replacement_pool:
            candidate_id = id(candidate)
            if candidate_id in used_replacement_ids:
                continue
            trial_libraries = candidate_libraries + selected_replacements + [candidate]
            trial_result = _validate_lane_state(validator, lane, trial_libraries)
            if trial_result.is_valid:
                selected_replacements.append(candidate)
                used_replacement_ids.add(candidate_id)
                lane.libraries = trial_libraries
                lane.total_data_gb = sum(lib.get_data_amount_gb() for lib in trial_libraries)
                stats["replacement_libraries"] += len(selected_replacements)
                stats["repaired_lanes"] += 1
                lane_repaired = True
                break
            if _is_capacity_shortage_only(trial_result):
                selected_replacements.append(candidate)
                used_replacement_ids.add(candidate_id)

        if selected_replacements and not lane_repaired:
            trial_libraries = candidate_libraries + selected_replacements
            if _validate_lane_state(validator, lane, trial_libraries).is_valid:
                lane.libraries = trial_libraries
                lane.total_data_gb = sum(lib.get_data_amount_gb() for lib in trial_libraries)
                stats["replacement_libraries"] += len(selected_replacements)
                stats["repaired_lanes"] += 1
            else:
                for lib in selected_replacements:
                    used_replacement_ids.discard(id(lib))
                lane.libraries = candidate_libraries
                lane.total_data_gb = sum(lib.get_data_amount_gb() for lib in candidate_libraries)
        elif not selected_replacements:
            lane.libraries = candidate_libraries
            lane.total_data_gb = sum(lib.get_data_amount_gb() for lib in candidate_libraries)

    if stats["removed_split_libraries"] <= 0:
        return stats

    unused_replacements = [
        lib for lib in replacement_pool
        if id(lib) not in used_replacement_ids
    ]
    kept_non_replacement = [
        lib for lib in unassigned_pool
        if id(lib) not in replacement_pool_ids and id(lib) not in used_replacement_ids
    ]
    solution.unassigned_libraries = kept_non_replacement + unused_replacements + removed_split_libraries
    solution.lane_assignments = [
        lane for lane in lanes if list(getattr(lane, "libraries", []) or [])
    ]
    logger.info(
        "非3.6T Lane拆分子文库即时修复完成: 影响Lane={}，剔除拆分子文库={}，补位普通文库={}，修复Lane={}",
        stats["affected_lanes"],
        stats["removed_split_libraries"],
        stats["replacement_libraries"],
        stats["repaired_lanes"],
    )
    return stats


def _filter_valid_lanes(
    lanes: List[LaneAssignment],
    validator: Any,
) -> Tuple[List[LaneAssignment], List[LaneAssignment]]:
    """过滤出通过严格校验且满足终态硬约束的Lane。"""
    valid_lanes: List[LaneAssignment] = []
    failed_lanes: List[LaneAssignment] = []
    for lane in lanes:
        if _is_split_lane_forbidden_by_mode(lane):
            failed_lanes.append(lane)
            logger.warning(
                "Lane {} 终态过滤淘汰: 含按规则应拆分文库但不是3.6T-NEW模式，应拆分文库只能排入3.6T-NEW Lane".format(
                    lane.lane_id
                )
            )
            continue

        result = _validate_lane_state(validator, lane, list(lane.libraries or []))
        if result.is_valid:
            valid_lanes.append(lane)
        else:
            failed_lanes.append(lane)
            logger.warning(
                "Lane {} 终态过滤淘汰: {}".format(
                    lane.lane_id,
                    [err.message for err in result.errors],
                )
            )
    return valid_lanes, failed_lanes


def _is_terminal_repair_progress_only_failure(result: Any) -> bool:
    """判断候选修复是否只剩容量不足/Index对数不足这类可继续补库的问题。"""
    errors = list(getattr(result, "errors", []) or [])
    if not errors:
        return True
    progress_markers = ("低于下限", "Index对数不足")
    for err in errors:
        message = str(getattr(err, "message", "") or "")
        if not any(marker in message for marker in progress_markers):
            return False
    return True


def _candidate_can_join_terminal_repair_lane(
    *,
    lane: LaneAssignment,
    current_libs: List[EnhancedLibraryInfo],
    candidate: EnhancedLibraryInfo,
    validator: Any,
    max_allowed: float,
) -> bool:
    """终态淘汰前补库候选轻校验，避免把明显冲突的库塞进失败Lane。"""
    if _is_ai_balance_library(candidate):
        return False
    if _is_package_lane_assignment(lane):
        return False
    if _is_truthy_flag(getattr(candidate, "is_package_lane", None)):
        return False
    if _get_package_lane_number_from_library(candidate):
        return False
    if _shares_split_family_with_selected(current_libs, candidate):
        return False
    candidate_data = float(getattr(candidate, "contract_data_raw", 0.0) or 0.0)
    if _total_lane_data(current_libs) + candidate_data > max_allowed + 1e-6:
        return False

    trial_libs = list(current_libs) + [candidate]
    if _is_mode_1_1_lane_context(lane, current_libs):
        add_test_total = sum(
            float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
            for lib in trial_libs
            if _is_mode_1_1_add_test_limited_library(lib)
        )
        mode_1_1_config = get_scheduling_config().get_mode_1_1_config()
        max_add_test_gb = float(
            (mode_1_1_config or {}).get("first_round_add_test_max_gb_per_lane", 150.0)
            or 0.0
        )
        if max_add_test_gb > 0 and add_test_total > max_add_test_gb + 1e-6:
            return False
    ss_valid, _, _ = _validate_lane_special_split_rule(trial_libs)
    if not ss_valid:
        return False
    imbalance_mix_valid, _ = _validate_lane_57_mix_rules(
        trial_libs,
        enforce_total_limit=False,
        lane_id=lane.lane_id,
        lane_metadata=lane.metadata,
    )
    if not imbalance_mix_valid:
        return False
    trial_result = _validate_lane_state(validator, lane, trial_libs)
    return _is_terminal_repair_progress_only_failure(trial_result)


def _try_repair_failed_lane_with_unassigned_pool(
    *,
    lane: LaneAssignment,
    unassigned_pool: List[EnhancedLibraryInfo],
    validator: Any,
) -> Dict[str, int]:
    """终态淘汰前，尝试从未分配池补库修复低容量或Index对数不足Lane。"""
    stats = {"repaired_lanes": 0, "added_libraries": 0}
    if _is_package_lane_assignment(lane) or _is_split_lane_forbidden_by_mode(lane):
        return stats

    current_libs = list(getattr(lane, "libraries", []) or [])
    if not current_libs:
        return stats

    initial_result = _validate_lane_state(validator, lane, current_libs)
    if initial_result.is_valid:
        return stats
    if not _is_terminal_repair_progress_only_failure(initial_result):
        return stats

    _, max_allowed = _resolve_lane_capacity_limits(
        libraries=current_libs,
        machine_type=lane.machine_type.value if lane.machine_type else "Nova X-25B",
        lane_id=lane.lane_id,
        lane_metadata=lane.metadata,
    )
    additions: List[EnhancedLibraryInfo] = []
    used_ids: Set[int] = set()
    candidate_pool = sorted(
        list(unassigned_pool),
        key=lambda lib: (
            _count_lane_index_pairs([lib]),
            float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
        ),
        reverse=True,
    )

    for candidate in candidate_pool:
        if id(candidate) in used_ids:
            continue
        if not _candidate_can_join_terminal_repair_lane(
            lane=lane,
            current_libs=current_libs,
            candidate=candidate,
            validator=validator,
            max_allowed=max_allowed,
        ):
            continue
        trial_libs = current_libs + [candidate]
        trial_result = _validate_lane_state(validator, lane, trial_libs)
        current_libs = trial_libs
        additions.append(candidate)
        used_ids.add(id(candidate))
        if trial_result.is_valid:
            break

    final_result = _validate_lane_state(validator, lane, current_libs)
    if not final_result.is_valid:
        return stats

    for candidate in additions:
        lane.add_library(candidate)
        _remove_library_by_identity_in_place(unassigned_pool, candidate)
    lane.calculate_metrics()
    stats["repaired_lanes"] = 1
    stats["added_libraries"] = len(additions)
    logger.info(
        "终态淘汰前修复Lane {}: 补入{}个未分配文库，数据量={:.1f}G".format(
            lane.lane_id,
            len(additions),
            _total_lane_data(list(getattr(lane, "libraries", []) or [])),
        )
    )
    return stats


def _repair_failed_lanes_before_final_cleanup(
    solution: Any,
    failed_lanes: List[LaneAssignment],
    validator: Any,
) -> Dict[str, int]:
    """终态过滤淘汰前，对低容量/Index对数不足Lane做一次保守补库修复。"""
    stats = {"attempted_lanes": 0, "repaired_lanes": 0, "added_libraries": 0}
    unassigned_pool = list(getattr(solution, "unassigned_libraries", []) or [])
    if not failed_lanes or not unassigned_pool:
        return stats
    for lane in failed_lanes:
        stats["attempted_lanes"] += 1
        repair_stats = _try_repair_failed_lane_with_unassigned_pool(
            lane=lane,
            unassigned_pool=unassigned_pool,
            validator=validator,
        )
        stats["repaired_lanes"] += repair_stats["repaired_lanes"]
        stats["added_libraries"] += repair_stats["added_libraries"]
    solution.unassigned_libraries = unassigned_pool
    return stats


def _clear_split_rollback_unassigned_only_flag(lib: EnhancedLibraryInfo) -> None:
    """允许拆分回滚原始文库在终态矩阵补救中重新参与3.6T拆分排机。"""
    if hasattr(lib, "_split_family_rollback_unassigned_only"):
        delattr(lib, "_split_family_rollback_unassigned_only")
    lib.is_split = False
    lib.wkissplit = ""
    lib.split_status = ""


def _try_build_matrix_split_lanes_for_group(
    *,
    source_libraries: List[EnhancedLibraryInfo],
    split_count: int,
    validator: Any,
    lane_id_prefix: str,
    force_split_count: bool = False,
) -> List[LaneAssignment]:
    """把同份数拆分文库按矩阵方式重组成Lane。"""
    if split_count <= 1 or not source_libraries:
        return []
    fragments_by_source: List[List[EnhancedLibraryInfo]] = []
    splitter = LibrarySplitter()
    for source in source_libraries:
        _clear_split_rollback_unassigned_only_flag(source)
        if force_split_count:
            split_source = deepcopy(source)
            split_source.contract_data_raw = float(getattr(source, "contract_data_raw", 0.0) or 0.0)
            fragments = []
            split_data_amount = split_source.contract_data_raw / split_count
            split_single_index_data = splitter._split_optional_float_value(
                getattr(split_source, "single_index_data", None),
                split_count,
            )
            split_ten_bp_data = splitter._split_optional_float_value(
                getattr(split_source, "ten_bp_data", None),
                split_count,
            )
            original_aidbid = str(
                getattr(split_source, "wkaidbid", None)
                or getattr(split_source, "aidbid", None)
                or ""
            ).strip()
            raw_total_contract = (
                getattr(split_source, "wktotalcontractdata", None)
                if getattr(split_source, "wktotalcontractdata", None) not in (None, "")
                else getattr(split_source, "total_contract_data", None)
            )
            try:
                original_total_contract = float(raw_total_contract)
            except (TypeError, ValueError):
                original_total_contract = float(split_source.contract_data_raw or 0.0)
            for i in range(split_count):
                new_lib = deepcopy(split_source)
                new_lib.contract_data_raw = split_data_amount
                new_lib.single_index_data = split_single_index_data
                new_lib.ten_bp_data = split_ten_bp_data
                new_lib.is_split = True
                new_lib.wkissplit = "yes"
                new_lib.split_status = "completed"
                new_lib.wktotalcontractdata = original_total_contract
                new_lib.total_contract_data = original_total_contract
                new_lib.original_library_id = str(getattr(split_source, "origrec", "") or "")
                new_lib.fragment_index = i + 1
                new_lib.total_fragments = split_count
                new_lib.fragment_id = f"{new_lib.original_library_id}_F{new_lib.fragment_index:03d}"
                if i == 0 and original_aidbid:
                    new_aidbid = original_aidbid
                else:
                    new_aidbid = str(uuid4())
                new_lib.wkaidbid = new_aidbid
                new_lib.aidbid = new_aidbid
                new_lib._split_source_library = source
                source_origrec_key = str(
                    getattr(source, "_source_origrec_key", None)
                    or getattr(source, "_origrec_key", None)
                    or getattr(source, "origrec", "")
                    or ""
                ).strip()
                new_lib._source_origrec_key = source_origrec_key
                new_lib._detail_output_key = str(new_lib.fragment_id or new_aidbid or source_origrec_key).strip()
                fragments.append(new_lib)
        else:
            fragments = splitter._perform_split(source)
        if len(fragments) != split_count or not all(_is_split_library(fragment) for fragment in fragments):
            return []
        fragments_by_source.append(fragments)

    lanes: List[LaneAssignment] = []
    for fragment_idx in range(split_count):
        lane_libs = [fragments[fragment_idx] for fragments in fragments_by_source]
        lane_id = f"{lane_id_prefix}_{MachineType.NOVA_X_25B.value}_{_reserve_auto_lane_serial(lane_id_prefix, MachineType.NOVA_X_25B):03d}"
        lane = LaneAssignment(
            lane_id=lane_id,
            machine_id=f"M_{lane_id}",
            machine_type=MachineType.NOVA_X_25B,
            lane_capacity_gb=_lane_capacity_for_machine(MachineType.NOVA_X_25B),
        )
        lane.metadata.update({"selected_seq_mode": "3.6T-NEW", "lcxms": "3.6T-NEW"})
        for lib in lane_libs:
            lib._current_seq_mode_raw = "3.6T-NEW"
            lane.add_library(lib)
        result = _validate_lane_state(validator, lane, list(lane.libraries or []))
        if not result.is_valid:
            return []
        lanes.append(lane)
    return lanes


def _try_add_matrix_split_lanes_from_unassigned(
    solution: Any,
    validator: Any,
) -> Dict[str, int]:
    """终态回滚后，对可完整矩阵拆分的未分配原始文库补建3.6T Lane。"""
    stats = {"added_lanes": 0, "used_originals": 0, "added_fragments": 0}
    unassigned = list(getattr(solution, "unassigned_libraries", []) or [])
    candidates: List[Tuple[EnhancedLibraryInfo, int]] = []
    splitter = LibrarySplitter()
    for lib in unassigned:
        if _is_split_library(lib):
            continue
        if not _is_split_rule_original_allowed_to_split_in_36t(lib):
            continue
        if _is_truthy_flag(getattr(lib, "is_package_lane", None)):
            continue
        if _get_package_lane_number_from_library(lib):
            continue
        eval_lib = deepcopy(lib)
        eval_lib._current_seq_mode_raw = "3.6T-NEW"
        eval_lib.selected_seq_mode = "3.6T-NEW"
        eval_lib.current_seq_mode = "3.6T-NEW"
        eval_lib.lcxms = "3.6T-NEW"
        if not splitter._should_split(eval_lib):
            continue
        split_count = len(splitter._perform_split(eval_lib))
        if split_count <= 1:
            continue
        candidates.append((lib, split_count))

    if not candidates:
        return stats

    grouped: Dict[int, List[EnhancedLibraryInfo]] = {}
    for lib, split_count in candidates:
        grouped.setdefault(split_count, []).append(lib)

    used_ids: Set[int] = set()
    added_lanes: List[LaneAssignment] = []
    for split_count, libs in sorted(grouped.items(), key=lambda item: (-item[0], -len(item[1]))):
        remaining = [lib for lib in libs if id(lib) not in used_ids]
        while len(remaining) >= 2:
            best_lanes: List[LaneAssignment] = []
            best_group: List[EnhancedLibraryInfo] = []
            for group_size in range(len(remaining), 1, -1):
                group = remaining[:group_size]
                data_per_lane = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in group) / split_count
                if data_per_lane < 995.0 - 1e-6 or data_per_lane > 1105.0 + 1e-6:
                    continue
                lanes = _try_build_matrix_split_lanes_for_group(
                    source_libraries=group,
                    split_count=split_count,
                    validator=validator,
                    lane_id_prefix="MS",
                )
                if lanes:
                    best_lanes = lanes
                    best_group = group
                    break
            if not best_lanes:
                break
            added_lanes.extend(best_lanes)
            for lib in best_group:
                used_ids.add(id(lib))
            remaining = [lib for lib in remaining if id(lib) not in used_ids]

    if not added_lanes:
        return stats

    solution.lane_assignments.extend(added_lanes)
    solution.unassigned_libraries = [lib for lib in unassigned if id(lib) not in used_ids]
    stats["added_lanes"] = len(added_lanes)
    stats["used_originals"] = len(used_ids)
    stats["added_fragments"] = sum(len(lane.libraries or []) for lane in added_lanes)
    logger.info(
        "终态矩阵拆分补Lane完成: 新增Lane={}，使用原始文库={}，拆分片段={}".format(
            stats["added_lanes"],
            stats["used_originals"],
            stats["added_fragments"],
        )
    )
    return stats


def _try_add_mixed_matrix_split_lanes_from_unassigned(
    solution: Any,
    validator: Any,
) -> Dict[str, int]:
    """终态修复后，对剩余原始文库尝试统一份数混合矩阵拆分成Lane。"""
    stats = {"added_lanes": 0, "used_originals": 0, "added_fragments": 0}
    unassigned = list(getattr(solution, "unassigned_libraries", []) or [])
    splitter = LibrarySplitter()
    candidates: List[EnhancedLibraryInfo] = []
    for lib in unassigned:
        if _is_split_library(lib):
            continue
        if not _is_split_rule_original_allowed_to_split_in_36t(lib):
            continue
        if _is_truthy_flag(getattr(lib, "is_package_lane", None)):
            continue
        if _get_package_lane_number_from_library(lib):
            continue
        eval_lib = deepcopy(lib)
        eval_lib._current_seq_mode_raw = "3.6T-NEW"
        eval_lib.selected_seq_mode = "3.6T-NEW"
        eval_lib.current_seq_mode = "3.6T-NEW"
        eval_lib.lcxms = "3.6T-NEW"
        if splitter._should_split(eval_lib):
            candidates.append(lib)
    mixed_lanes, mixed_sources = _try_build_mixed_matrix_split_lanes_from_sources(
        source_libraries=candidates,
        validator=validator,
    )
    if not mixed_lanes:
        return stats
    used_ids = {id(lib) for lib in mixed_sources}
    solution.lane_assignments.extend(mixed_lanes)
    solution.unassigned_libraries = [lib for lib in unassigned if id(lib) not in used_ids]
    stats["added_lanes"] = len(mixed_lanes)
    stats["used_originals"] = len(used_ids)
    stats["added_fragments"] = sum(len(lane.libraries or []) for lane in mixed_lanes)
    logger.info(
        "终态混合矩阵拆分补Lane完成: 新增Lane={}，使用原始文库={}，拆分片段={}".format(
            stats["added_lanes"],
            stats["used_originals"],
            stats["added_fragments"],
        )
    )
    return stats


def _collect_split_source_candidates(
    libraries: List[EnhancedLibraryInfo],
) -> List[Tuple[EnhancedLibraryInfo, int, List[EnhancedLibraryInfo]]]:
    """收集可在3.6T上下文拆分的原始文库及其完整片段。"""
    splitter = LibrarySplitter()
    candidates: List[Tuple[EnhancedLibraryInfo, int, List[EnhancedLibraryInfo]]] = []
    for lib in list(libraries or []):
        if _is_split_library(lib):
            continue
        if not _is_split_rule_original_allowed_to_split_in_36t(lib):
            continue
        if _is_truthy_flag(getattr(lib, "is_package_lane", None)):
            continue
        if _get_package_lane_number_from_library(lib):
            continue
        eval_lib = deepcopy(lib)
        eval_lib._current_seq_mode_raw = "3.6T-NEW"
        eval_lib.selected_seq_mode = "3.6T-NEW"
        eval_lib.current_seq_mode = "3.6T-NEW"
        eval_lib.lcxms = "3.6T-NEW"
        if not splitter._should_split(eval_lib):
            continue
        fragments = splitter._perform_split(eval_lib)
        split_count = len(fragments)
        if split_count <= 1:
            continue
        if not all(_is_split_library(fragment) for fragment in fragments):
            continue
        candidates.append((lib, split_count, fragments))
    return candidates


def _terminal_split_lane_metadata(
    mode_name: str = "3.6T-NEW",
    *,
    is_dedicated_imbalance_lane: bool = False,
) -> Dict[str, Any]:
    """终态拆分专池Lane的统一metadata。"""
    metadata = {
        "selected_seq_mode": mode_name,
        "seq_mode": mode_name,
        "lcxms": mode_name,
        "dispatch_stage": "terminal_sample_type_split_fragment",
    }
    if is_dedicated_imbalance_lane:
        metadata["is_dedicated_imbalance_lane"] = True
    return metadata


def _is_split_source_subset_dedicated_imbalance(
    source_records: List[Tuple[EnhancedLibraryInfo, int, List[EnhancedLibraryInfo]]],
) -> bool:
    """判断拆分源集合是否应按碱基不均专池预留平衡文库。"""
    sources = [source for source, _, _ in list(source_records or [])]
    return bool(sources) and all(_is_imbalance_library_candidate(source) for source in sources)


def _resolve_terminal_split_lane_capacity_window(
    *,
    sample_libraries: List[EnhancedLibraryInfo],
    machine_type: MachineType,
    lane_id: str,
    mode_name: str = "3.6T-NEW",
    is_dedicated_imbalance_lane: bool = False,
) -> Tuple[float, float]:
    """按目标模式和平衡预留解析拆分专池原始合同量窗口。"""
    metadata = _terminal_split_lane_metadata(
        mode_name,
        is_dedicated_imbalance_lane=is_dedicated_imbalance_lane,
    )
    return _resolve_lane_capacity_limits(
        libraries=sample_libraries,
        machine_type=machine_type,
        lane_id=lane_id,
        lane_metadata=metadata,
    )


def _find_cross_split_source_subset(
    candidates: List[Tuple[EnhancedLibraryInfo, int, List[EnhancedLibraryInfo]]],
    lane_count: int,
    min_lane_gb: Optional[float] = None,
    max_lane_gb: Optional[float] = None,
    machine_type: MachineType = MachineType.NOVA_X_25B,
    mode_name: str = "3.6T-NEW",
    max_candidates: int = 80,
) -> List[Tuple[EnhancedLibraryInfo, int, List[EnhancedLibraryInfo]]]:
    """按原始文库总量窗口寻找跨拆分份数子集。"""
    if lane_count <= 0 or not candidates:
        return []
    if min_lane_gb is None or max_lane_gb is None:
        sample_libraries = [item[2][0] for item in candidates if item[2]]
        if not sample_libraries:
            sample_libraries = [item[0] for item in candidates]
        is_dedicated_imbalance_lane = _is_split_source_subset_dedicated_imbalance(candidates)
        min_lane_gb, max_lane_gb = _resolve_terminal_split_lane_capacity_window(
            sample_libraries=sample_libraries,
            machine_type=machine_type,
            lane_id="XS_TMP",
            mode_name=mode_name,
            is_dedicated_imbalance_lane=is_dedicated_imbalance_lane,
        )
    min_total = min_lane_gb * lane_count
    max_total = max_lane_gb * lane_count
    ordered = sorted(
        candidates[:max_candidates],
        key=lambda item: (
            -float(getattr(item[0], "contract_data_raw", 0.0) or 0.0),
            -item[1],
            _safe_str(getattr(item[0], "origrec", ""), default=""),
        ),
    )
    best_subset: List[int] = []
    best_score: Tuple[float, int] = (float("inf"), 0)
    state_count = 0
    max_states = 120000

    def search(
        start_index: int,
        selected_indices: List[int],
        selected_total: float,
        selected_max_split: int,
    ) -> None:
        nonlocal best_subset, best_score, state_count
        if state_count >= max_states:
            return
        state_count += 1

        if selected_total > max_total + 1e-6:
            return
        if selected_total >= min_total - 1e-6 and selected_max_split <= lane_count:
            score = (abs(selected_total - ((min_total + max_total) / 2.0)), -len(selected_indices))
            if not best_subset or score < best_score:
                best_subset = list(selected_indices)
                best_score = score

        if start_index >= len(ordered):
            return

        remaining_total = sum(
            float(getattr(ordered[idx][0], "contract_data_raw", 0.0) or 0.0)
            for idx in range(start_index, len(ordered))
        )
        if selected_total + remaining_total < min_total - 1e-6:
            return

        for idx in range(start_index, len(ordered)):
            source, split_count, _ = ordered[idx]
            if split_count > lane_count:
                continue
            data = float(getattr(source, "contract_data_raw", 0.0) or 0.0)
            search(
                idx + 1,
                selected_indices + [idx],
                selected_total + data,
                max(selected_max_split, split_count),
            )

    search(0, [], 0.0, 0)
    return [ordered[idx] for idx in best_subset]


def _try_pack_cross_split_fragments_into_lanes(
    source_records: List[Tuple[EnhancedLibraryInfo, int, List[EnhancedLibraryInfo]]],
    lane_count: int,
    validator: Any,
    machine_type: MachineType = MachineType.NOVA_X_25B,
    mode_name: str = "3.6T-NEW",
) -> List[LaneAssignment]:
    """将不同拆分份数的完整片段装入同一组3.6T Lane。"""
    if lane_count <= 0 or not source_records:
        return []

    is_dedicated_imbalance_lane = _is_split_source_subset_dedicated_imbalance(source_records)
    lane_metadata = _terminal_split_lane_metadata(
        mode_name,
        is_dedicated_imbalance_lane=is_dedicated_imbalance_lane,
    )
    lanes: List[LaneAssignment] = []
    for lane_index in range(lane_count):
        lane_id = f"XS_TMP_{lane_index + 1:03d}"
        lane = LaneAssignment(
            lane_id=lane_id,
            machine_id=f"M_{lane_id}",
            machine_type=machine_type,
            lane_capacity_gb=_lane_capacity_for_machine(machine_type),
        )
        lane.metadata.update(lane_metadata)
        lane.metadata["dispatch_stage"] = "cross_split_fragment_binpack"
        lanes.append(lane)

    for source, split_count, fragments in sorted(
        source_records,
        key=lambda item: (
            -float(getattr(item[0], "contract_data_raw", 0.0) or 0.0) / max(item[1], 1),
            -item[1],
        ),
    ):
        if split_count > lane_count or len(fragments) != split_count:
            return []
        selected_lane_indices: List[int] = []
        for fragment in fragments:
            ranked_lane_indices = sorted(
                [
                    idx for idx in range(lane_count)
                    if idx not in selected_lane_indices
                ],
                key=lambda idx: float(getattr(lanes[idx], "total_data_gb", 0.0) or 0.0),
            )
            chosen_idx: Optional[int] = None
            for lane_idx in ranked_lane_indices:
                if _validate_index_conflicts_latest(list(lanes[lane_idx].libraries or []) + [fragment]):
                    continue
                chosen_idx = lane_idx
                break
            if chosen_idx is None:
                return []
            selected_lane_indices.append(chosen_idx)
            fragment._current_seq_mode_raw = mode_name
            fragment.selected_seq_mode = mode_name
            fragment.current_seq_mode = mode_name
            fragment.lcxms = mode_name
            lanes[chosen_idx].add_library(fragment)

    for lane in lanes:
        total_data = float(getattr(lane, "total_data_gb", 0.0) or 0.0)
        min_allowed, max_allowed = _resolve_terminal_split_lane_capacity_window(
            sample_libraries=list(lane.libraries or []),
            machine_type=machine_type,
            lane_id=lane.lane_id,
            mode_name=mode_name,
            is_dedicated_imbalance_lane=is_dedicated_imbalance_lane,
        )
        if total_data < min_allowed - 1e-6 or total_data > max_allowed + 1e-6:
            return []
        if not _validate_lane_state(validator, lane, list(lane.libraries or [])).is_valid:
            return []
    for lane in lanes:
        lane_id = f"XS_{machine_type.value}_{_reserve_auto_lane_serial('XS', machine_type):03d}"
        lane.lane_id = lane_id
        lane.machine_id = f"M_{lane_id}"
    return lanes


def _try_pack_sample_type_split_fragments_greedy(
    source_records: List[Tuple[EnhancedLibraryInfo, int, List[EnhancedLibraryInfo]]],
    validator: Any,
    machine_type: MachineType,
    lane_id_prefix: str = "TS",
    max_lanes: int = 12,
) -> Tuple[List[LaneAssignment], Set[int]]:
    """同文库类型内把可拆文库片段直接装成3.6T Lane，完整使用被选原始文库。"""
    if not source_records:
        return [], set()

    candidate_records = sorted(
        list(source_records),
        key=lambda item: (
            -float(getattr(item[0], "contract_data_raw", 0.0) or 0.0),
            -item[1],
            _safe_str(getattr(item[0], "origrec", ""), default=""),
        ),
    )
    remaining_records = list(candidate_records)
    added_lanes: List[LaneAssignment] = []
    used_source_ids: Set[int] = set()
    used_fragment_ids: Set[int] = set()
    touched_source_ids: Set[int] = set()

    while remaining_records and len(added_lanes) < max_lanes:
        lane_id = f"{lane_id_prefix}_TMP_{len(added_lanes) + 1:03d}"
        lane = LaneAssignment(
            lane_id=lane_id,
            machine_id=f"M_{lane_id}",
            machine_type=machine_type,
            lane_capacity_gb=_lane_capacity_for_machine(machine_type),
        )
        lane.metadata.update(
            {
                "selected_seq_mode": "3.6T-NEW",
                "seq_mode": "3.6T-NEW",
                "lcxms": "3.6T-NEW",
                "dispatch_stage": "terminal_sample_type_split_fragment_greedy",
            }
        )

        picked_fragments: List[EnhancedLibraryInfo] = []
        picked_source_keys: Set[int] = set()
        for source, _, fragments in remaining_records:
            source_key = id(source)
            if source_key in picked_source_keys:
                continue
            ranked_fragments = sorted(
                list(fragments or []),
                key=lambda fragment: (
                    -float(getattr(fragment, "contract_data_raw", 0.0) or 0.0),
                    _safe_str(getattr(fragment, "fragment_id", ""), default=""),
                ),
            )
            for fragment in ranked_fragments:
                if id(fragment) in used_fragment_ids:
                    continue
                trial_libs = list(lane.libraries or []) + [fragment]
                trial_total = sum(
                    float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
                    for lib in trial_libs
                )
                min_allowed, max_allowed = _resolve_lane_capacity_limits(
                    libraries=trial_libs,
                    machine_type=machine_type,
                    lane_id=lane_id,
                    lane_metadata=lane.metadata,
                )
                if trial_total > max_allowed + 1e-6:
                    continue
                if _validate_index_conflicts_latest(trial_libs):
                    continue
                fragment._current_seq_mode_raw = "3.6T-NEW"
                fragment.selected_seq_mode = "3.6T-NEW"
                fragment.current_seq_mode = "3.6T-NEW"
                fragment.lcxms = "3.6T-NEW"
                lane.add_library(fragment)
                picked_fragments.append(fragment)
                picked_source_keys.add(source_key)
                touched_source_ids.add(source_key)
                break
            total_data = float(getattr(lane, "total_data_gb", 0.0) or 0.0)
            if total_data >= min_allowed - 1e-6:
                break

        lane_libs = list(lane.libraries or [])
        if not lane_libs:
            break
        min_allowed, max_allowed = _resolve_lane_capacity_limits(
            libraries=lane_libs,
            machine_type=machine_type,
            lane_id=lane_id,
            lane_metadata=lane.metadata,
        )
        total_data = float(getattr(lane, "total_data_gb", 0.0) or 0.0)
        if total_data < min_allowed - 1e-6 or total_data > max_allowed + 1e-6:
            break
        if _count_lane_index_pairs(lane_libs) < AI_LANE_MIN_INDEX_PAIRS:
            break
        validation_result = _validate_lane_state(validator, lane, lane_libs)
        if not getattr(validation_result, "is_valid", False):
            filtered_errors = [
                err for err in list(getattr(validation_result, "errors", []) or [])
                if getattr(err, "rule_type", None) != ValidationRuleType.SPECIAL_LIBRARY_LIMIT
            ]
            filtered_warnings = list(getattr(validation_result, "warnings", []) or [])
            if filtered_errors or (getattr(validator, "strict_mode", False) and filtered_warnings):
                break

        lane_id = f"{lane_id_prefix}_{machine_type.value}_{_reserve_auto_lane_serial(lane_id_prefix, machine_type):03d}"
        lane.lane_id = lane_id
        lane.machine_id = f"M_{lane_id}"
        added_lanes.append(lane)

        used_fragment_keys = {id(fragment) for fragment in picked_fragments}
        used_fragment_ids.update(used_fragment_keys)
        next_records: List[Tuple[EnhancedLibraryInfo, int, List[EnhancedLibraryInfo]]] = []
        for source, split_count, fragments in remaining_records:
            left_fragments = [
                fragment for fragment in list(fragments or [])
                if id(fragment) not in used_fragment_keys
            ]
            if left_fragments:
                next_records.append((source, split_count, left_fragments))
            else:
                used_source_ids.add(id(source))
        remaining_records = next_records

    incomplete_touched_sources = touched_source_ids - used_source_ids
    if incomplete_touched_sources:
        return [], set()
    return added_lanes, used_source_ids


def _try_pack_sample_type_split_fragments_with_fillers(
    split_source_records: List[Tuple[EnhancedLibraryInfo, int, List[EnhancedLibraryInfo]]],
    filler_libraries: List[EnhancedLibraryInfo],
    validator: Any,
    machine_type: MachineType,
    lane_id_prefix: str = "TS",
    max_lanes: int = 12,
) -> Tuple[List[LaneAssignment], Set[int]]:
    """同文库类型内用拆分片段加无需拆的小文库共同补足3.6T Lane。"""
    split_records = list(split_source_records or [])
    fillers = [
        lib for lib in list(filler_libraries or [])
        if not _is_split_library(lib)
        and not _is_small_unsplit_original_reserved_for_mode_1_1(lib)
        and not _should_library_split_in_3_6t(lib)
    ]
    if not split_records or not fillers:
        return [], set()

    split_total = sum(
        float(getattr(source, "contract_data_raw", 0.0) or 0.0)
        for source, _, _ in split_records
    )
    filler_total = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in fillers)
    total_data = split_total + filler_total
    max_split_count = max(split_count for _, split_count, _ in split_records)
    candidate_lane_counts = [
        lane_count
        for lane_count in range(min(max_lanes, max_split_count + len(fillers)), max_split_count - 1, -1)
        if total_data >= 995.0 * lane_count - 1e-6
    ]
    if not candidate_lane_counts:
        return [], set()

    def select_split_records_for_lane_count(
        lane_count: int,
    ) -> List[Tuple[EnhancedLibraryInfo, int, List[EnhancedLibraryInfo]]]:
        selected_records: List[Tuple[EnhancedLibraryInfo, int, List[EnhancedLibraryInfo]]] = []
        index_fragment_counts: Dict[str, int] = {}
        selected_split_total = 0.0
        max_total = 1105.0 * lane_count
        for source, split_count, fragments in sorted(
            split_records,
            key=lambda item: (
                -float(getattr(item[0], "contract_data_raw", 0.0) or 0.0),
                -item[1],
                _safe_str(getattr(item[0], "origrec", ""), default=""),
            ),
        ):
            if split_count > lane_count or len(fragments or []) != split_count:
                continue
            index_key = _safe_str(getattr(source, "index_seq", None), default="")
            if index_fragment_counts.get(index_key, 0) + split_count > lane_count:
                continue
            source_data = float(getattr(source, "contract_data_raw", 0.0) or 0.0)
            if selected_split_total + source_data > max_total + 1e-6:
                continue
            selected_records.append((source, split_count, fragments))
            index_fragment_counts[index_key] = index_fragment_counts.get(index_key, 0) + split_count
            selected_split_total += source_data
        if selected_split_total + filler_total < 995.0 * lane_count - 1e-6:
            return []
        return selected_records

    def build_lanes(lane_count: int) -> List[LaneAssignment]:
        lanes: List[LaneAssignment] = []
        for lane_index in range(lane_count):
            lane_id = f"{lane_id_prefix}_TMP_{lane_index + 1:03d}"
            lane = LaneAssignment(
                lane_id=lane_id,
                machine_id=f"M_{lane_id}",
                machine_type=machine_type,
                lane_capacity_gb=_lane_capacity_for_machine(machine_type),
            )
            lane.metadata.update(
                {
                    "selected_seq_mode": "3.6T-NEW",
                    "seq_mode": "3.6T-NEW",
                    "lcxms": "3.6T-NEW",
                    "dispatch_stage": "terminal_sample_type_split_fragment_with_fillers",
                }
            )
            lanes.append(lane)
        return lanes

    def try_lane_count(lane_count: int) -> List[LaneAssignment]:
        selected_split_records = select_split_records_for_lane_count(lane_count)
        if not selected_split_records:
            return []
        lanes = build_lanes(lane_count)
        used_fillers: List[EnhancedLibraryInfo] = []
        for source, split_count, fragments in sorted(
            selected_split_records,
            key=lambda item: (
                -float(getattr(item[0], "contract_data_raw", 0.0) or 0.0) / max(item[1], 1),
                -item[1],
                _safe_str(getattr(item[0], "origrec", ""), default=""),
            ),
        ):
            if split_count > lane_count or len(fragments or []) != split_count:
                return []
            used_lane_indices: Set[int] = set()
            for fragment in list(fragments or []):
                ranked_indices = sorted(
                    [idx for idx in range(lane_count) if idx not in used_lane_indices],
                    key=lambda idx: float(getattr(lanes[idx], "total_data_gb", 0.0) or 0.0),
                )
                placed = False
                for lane_idx in ranked_indices:
                    trial_libs = list(lanes[lane_idx].libraries or []) + [fragment]
                    if _validate_index_conflicts_latest(trial_libs):
                        continue
                    fragment._current_seq_mode_raw = "3.6T-NEW"
                    fragment.selected_seq_mode = "3.6T-NEW"
                    fragment.current_seq_mode = "3.6T-NEW"
                    fragment.lcxms = "3.6T-NEW"
                    lanes[lane_idx].add_library(fragment)
                    used_lane_indices.add(lane_idx)
                    placed = True
                    break
                if not placed:
                    return []

        for filler in sorted(
            fillers,
            key=lambda lib: (
                -float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
                -_count_library_index_pairs(lib),
                _safe_str(getattr(lib, "origrec", ""), default=""),
            ),
        ):
            placed = False
            for lane_idx in sorted(
                range(lane_count),
                key=lambda idx: float(getattr(lanes[idx], "total_data_gb", 0.0) or 0.0),
            ):
                trial_libs = list(lanes[lane_idx].libraries or []) + [filler]
                trial_total = sum(
                    float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
                    for lib in trial_libs
                )
                _, max_allowed = _resolve_lane_capacity_limits(
                    libraries=trial_libs,
                    machine_type=machine_type,
                    lane_id=lanes[lane_idx].lane_id,
                    lane_metadata=lanes[lane_idx].metadata,
                )
                if trial_total > max_allowed + 1e-6:
                    continue
                if _validate_index_conflicts_latest(trial_libs):
                    continue
                filler._current_seq_mode_raw = "3.6T-NEW"
                filler.selected_seq_mode = "3.6T-NEW"
                filler.current_seq_mode = "3.6T-NEW"
                filler.lcxms = "3.6T-NEW"
                lanes[lane_idx].add_library(filler)
                used_fillers.append(filler)
                placed = True
                break
            if not placed:
                continue

        for lane in lanes:
            lane_libs = list(lane.libraries or [])
            min_allowed, max_allowed = _resolve_lane_capacity_limits(
                libraries=lane_libs,
                machine_type=machine_type,
                lane_id=lane.lane_id,
                lane_metadata=lane.metadata,
            )
            total = float(getattr(lane, "total_data_gb", 0.0) or 0.0)
            if total < min_allowed - 1e-6 or total > max_allowed + 1e-6:
                return []
            if _count_lane_index_pairs(lane_libs) < AI_LANE_MIN_INDEX_PAIRS:
                return []
            lane.metadata.update(_infer_terminal_lane_constraint_metadata(lane_libs, validator))
            validation_result = _validate_lane_state(validator, lane, lane_libs)
            if not getattr(validation_result, "is_valid", False):
                filtered_errors = [
                    err for err in list(getattr(validation_result, "errors", []) or [])
                    if getattr(err, "rule_type", None) != ValidationRuleType.SPECIAL_LIBRARY_LIMIT
                ]
                filtered_warnings = list(getattr(validation_result, "warnings", []) or [])
                if filtered_errors or (getattr(validator, "strict_mode", False) and filtered_warnings):
                    return []

        for lane in lanes:
            lane_id = f"{lane_id_prefix}_{machine_type.value}_{_reserve_auto_lane_serial(lane_id_prefix, machine_type):03d}"
            lane.lane_id = lane_id
            lane.machine_id = f"M_{lane_id}"
        for lane in lanes:
            lane.metadata["_terminal_used_filler_ids"] = [id(lib) for lib in used_fillers]
            lane.metadata["_terminal_used_source_ids"] = [id(source) for source, _, _ in selected_split_records]
        return lanes

    for lane_count in candidate_lane_counts:
        lanes = try_lane_count(lane_count)
        if lanes:
            used_ids: Set[int] = set()
            used_filler_ids: Set[int] = set()
            for lane in lanes:
                used_filler_ids.update(
                    int(value)
                    for value in list(lane.metadata.pop("_terminal_used_filler_ids", []) or [])
                )
                used_ids.update(
                    int(value)
                    for value in list(lane.metadata.pop("_terminal_used_source_ids", []) or [])
                )
            used_ids.update(used_filler_ids)
            return lanes, used_ids
    return [], set()


def _try_build_single_36t_mixed_lane_from_items(
    *,
    source_records: List[Tuple[EnhancedLibraryInfo, int, List[EnhancedLibraryInfo]]],
    filler_libraries: List[EnhancedLibraryInfo],
    validator: Any,
    machine_type: MachineType,
    lane_id_prefix: str,
    max_candidates: int = 220,
    require_36t_main_library: bool = False,
) -> Tuple[Optional[LaneAssignment], Set[int]]:
    """从普通文库中挑一条3.6T混排Lane。

    拆分文库必须以完整家族跨多条Lane提交，且同源片段不能进入同一条Lane；
    单条增量Lane无法满足该原子性要求，因此这里只处理普通补料。
    """
    items: List[Tuple[EnhancedLibraryInfo, int, EnhancedLibraryInfo]] = []
    for filler in list(filler_libraries or []):
        items.append((filler, id(filler), filler))
    if not items:
        return None, set()

    ordered_items = sorted(
        items[:max_candidates],
        key=lambda item: (
            -float(getattr(item[2], "contract_data_raw", 0.0) or 0.0),
            -_count_library_index_pairs(item[2]),
            _safe_str(getattr(item[2], "origrec", ""), default=""),
        ),
    )
    selected: List[Tuple[EnhancedLibraryInfo, int, EnhancedLibraryInfo]] = []
    selected_sources: Set[int] = set()
    best_total = 0.0
    best_index_pairs = 0
    validation_failures: Dict[str, int] = {}
    lane_metadata = {
        "selected_seq_mode": "3.6T-NEW",
        "seq_mode": "3.6T-NEW",
        "lcxms": "3.6T-NEW",
        "dispatch_stage": "terminal_global_36t_mixed_incremental",
    }

    def build_lane(lane_id: str) -> LaneAssignment:
        lane = LaneAssignment(
            lane_id=lane_id,
            machine_id=f"M_{lane_id}",
            machine_type=machine_type,
            lane_capacity_gb=_lane_capacity_for_machine(machine_type),
        )
        lane.metadata.update(lane_metadata)
        for _, _, lib in selected:
            lib._current_seq_mode_raw = "3.6T-NEW"
            lib.selected_seq_mode = "3.6T-NEW"
            lib.current_seq_mode = "3.6T-NEW"
            lib.lcxms = "3.6T-NEW"
            lane.add_library(lib)
        return lane

    def selected_total() -> float:
        return sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for _, _, lib in selected)

    best_lane: Optional[LaneAssignment] = None
    best_used_sources: Set[int] = set()
    for source, item_key, lib in ordered_items:
        source_key = id(source)
        if source_key in selected_sources:
            continue
        trial_libs = [item[2] for item in selected] + [lib]
        trial_total = sum(float(getattr(item, "contract_data_raw", 0.0) or 0.0) for item in trial_libs)
        _, max_allowed = _resolve_lane_capacity_limits(
            libraries=trial_libs,
            machine_type=machine_type,
            lane_id=f"{lane_id_prefix}_TMP",
            lane_metadata=lane_metadata,
        )
        if trial_total > max_allowed + 1e-6:
            continue
        if _validate_index_conflicts_latest(trial_libs):
            continue
        selected.append((source, item_key, lib))
        selected_sources.add(source_key)
        best_total = max(best_total, selected_total())
        best_index_pairs = max(best_index_pairs, _count_lane_index_pairs(trial_libs))

        min_allowed, max_allowed = _resolve_lane_capacity_limits(
            libraries=trial_libs,
            machine_type=machine_type,
            lane_id=f"{lane_id_prefix}_TMP",
            lane_metadata=lane_metadata,
        )
        if selected_total() < min_allowed - 1e-6:
            continue
        if _count_lane_index_pairs(trial_libs) < AI_LANE_MIN_INDEX_PAIRS:
            continue
        if require_36t_main_library and not any(
            _is_eligible_for_36t_after_mode_priority(item) for item in trial_libs
        ):
            continue
        trial_lane = build_lane(f"{lane_id_prefix}_TMP")
        validation_result = _validate_lane_state(validator, trial_lane, trial_libs)
        if not getattr(validation_result, "is_valid", False):
            filtered_errors = [
                err for err in list(getattr(validation_result, "errors", []) or [])
                if getattr(err, "rule_type", None) != ValidationRuleType.SPECIAL_LIBRARY_LIMIT
            ]
            filtered_warnings = list(getattr(validation_result, "warnings", []) or [])
            if filtered_errors or (getattr(validator, "strict_mode", False) and filtered_warnings):
                for err in filtered_errors:
                    key = _safe_str(getattr(err, "message", None), default="validation_error")
                    validation_failures[key] = validation_failures.get(key, 0) + 1
                if getattr(validator, "strict_mode", False):
                    for warn in filtered_warnings:
                        key = _safe_str(getattr(warn, "message", None), default="validation_warning")
                        validation_failures[key] = validation_failures.get(key, 0) + 1
                continue
        lane_id = f"{lane_id_prefix}_{machine_type.value}_{_reserve_auto_lane_serial(lane_id_prefix, machine_type):03d}"
        best_lane = build_lane(lane_id)
        best_used_sources = set(selected_sources)
        break

    if best_lane is None:
        top_failures = sorted(validation_failures.items(), key=lambda item: -item[1])[:3]
        logger.info(
            "终态全局3.6增量混排未成Lane明细: 机型={}, 候选items={}, best_total={:.1f}G, best_index_pairs={}, validation_top={}".format(
                machine_type.value,
                len(ordered_items),
                best_total,
                best_index_pairs,
                top_failures,
            )
        )
        return None, set()
    return best_lane, best_used_sources


def _try_add_cross_split_fragment_lanes_from_unassigned(
    solution: Any,
    validator: Any,
) -> Dict[str, int]:
    """跨不同拆分份数做片段级装箱，完整使用原始文库所有片段后补建Lane。"""
    stats = {"added_lanes": 0, "used_originals": 0, "added_fragments": 0}
    unassigned = list(getattr(solution, "unassigned_libraries", []) or [])
    candidates = _collect_split_source_candidates(
        [
            lib for lib in unassigned
            if _is_split_rule_original_allowed_to_split_in_36t(lib)
        ]
    )
    if not candidates:
        return stats

    used_ids: Set[int] = set()
    added_lanes: List[LaneAssignment] = []
    while True:
        remaining = [record for record in candidates if id(record[0]) not in used_ids]
        if len(remaining) < 2:
            break

        best_lanes: List[LaneAssignment] = []
        best_sources: List[Tuple[EnhancedLibraryInfo, int, List[EnhancedLibraryInfo]]] = []
        for lane_count in range(8, 1, -1):
            subset = _find_cross_split_source_subset(remaining, lane_count)
            if not subset:
                continue
            lanes = _try_pack_cross_split_fragments_into_lanes(
                source_records=subset,
                lane_count=lane_count,
                validator=validator,
            )
            if lanes:
                best_lanes = lanes
                best_sources = subset
                break

        if not best_lanes:
            break
        added_lanes.extend(best_lanes)
        for source, _, _ in best_sources:
            used_ids.add(id(source))

    if not added_lanes:
        return stats

    solution.lane_assignments.extend(added_lanes)
    solution.unassigned_libraries = [
        lib for lib in unassigned
        if id(lib) not in used_ids
    ]
    stats["added_lanes"] = len(added_lanes)
    stats["used_originals"] = len(used_ids)
    stats["added_fragments"] = sum(len(lane.libraries or []) for lane in added_lanes)
    logger.info(
        "终态跨份数片段装箱补Lane完成: 新增Lane={}，使用原始文库={}，拆分片段={}".format(
            stats["added_lanes"],
            stats["used_originals"],
            stats["added_fragments"],
        )
    )
    return stats


def _terminal_sample_type_group_key(
    lib: EnhancedLibraryInfo,
) -> Tuple[MachineType, str]:
    """终态专池增量只按机型和文库类型聚合。"""
    machine_type = _resolve_machine_type_enum_simple(
        _safe_str(getattr(lib, "eq_type", None), default="")
    )
    sample_type = (
        _safe_str(getattr(lib, "sample_type_code", None), default="")
        or _safe_str(getattr(lib, "wksampletype", None), default="")
        or _safe_str(getattr(lib, "lab_type", None), default="")
        or "UNKNOWN"
    )
    return machine_type, sample_type


def _terminal_dedicated_pool_hard_skip_reason(
    pool: List[EnhancedLibraryInfo],
    *,
    machine_type: MachineType,
    lane_id_prefix: str,
    lane_metadata: Dict[str, Any],
    max_candidates: int,
) -> Optional[str]:
    """只做单调硬失败判断，避免终态专池进入必失败的贪心枚举。"""
    candidates = list(pool or [])[:max_candidates]
    if not candidates:
        return "empty_pool"

    metadata_key = tuple(
        sorted((str(key), repr(value)) for key, value in dict(lane_metadata or {}).items())
    )
    cache_key = (
        machine_type.value if isinstance(machine_type, MachineType) else str(machine_type),
        lane_id_prefix,
        metadata_key,
        _build_library_compact_identity_signature(candidates, canonicalize=True),
    )
    cached = _TERMINAL_DEDICATED_HARD_SKIP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    min_allowed, _ = _resolve_lane_capacity_limits(
        libraries=candidates,
        machine_type=machine_type,
        lane_id=f"{lane_id_prefix}_TMP",
        lane_metadata=lane_metadata,
    )
    total_gb = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in candidates)
    reason: Optional[str] = None
    if total_gb + 1e-6 < min_allowed:
        reason = "hard_skip_total_below_min(total={:.1f}G,min={:.1f}G)".format(
            total_gb,
            min_allowed,
        )
    else:
        index_pairs = _count_lane_index_pairs(candidates)
        if index_pairs < AI_LANE_MIN_INDEX_PAIRS:
            reason = "hard_skip_index_pairs({}<{})".format(
                index_pairs,
                AI_LANE_MIN_INDEX_PAIRS,
            )

    if reason is not None:
        if len(_TERMINAL_DEDICATED_HARD_SKIP_CACHE) >= 4096:
            _TERMINAL_DEDICATED_HARD_SKIP_CACHE.clear()
        _TERMINAL_DEDICATED_HARD_SKIP_CACHE[cache_key] = reason
    return reason


def _attempt_build_terminal_dedicated_lane_from_group(
    pool: List[EnhancedLibraryInfo],
    validator: Any,
    machine_type: MachineType,
    lane_id_prefix: str,
    extra_metadata: Dict[str, Any],
    max_candidates: int = 120,
) -> Tuple[Optional[LaneAssignment], List[EnhancedLibraryInfo], str]:
    """在单一文库类型池内做有限顺序贪心搜索，避免全DFS组合爆炸。"""
    hard_skip_reason = _terminal_dedicated_pool_hard_skip_reason(
        pool,
        machine_type=machine_type,
        lane_id_prefix=lane_id_prefix,
        lane_metadata=extra_metadata,
        max_candidates=max_candidates,
    )
    if hard_skip_reason:
        return None, [], hard_skip_reason

    base_candidates = list(pool or [])[:max_candidates]
    candidates = sorted(
        base_candidates,
        key=lambda lib: (
            -float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
            -_count_library_index_pairs(lib),
            _safe_str(getattr(lib, "origrec", ""), default=""),
        ),
    )
    if not candidates:
        return None, [], "empty_pool"

    best_shortage = float("inf")
    best_index_pairs = 0
    validation_failures: Dict[str, int] = {}

    def build_lane(selected: List[EnhancedLibraryInfo], lane_id: str) -> LaneAssignment:
        lane = LaneAssignment(
            lane_id=lane_id,
            machine_id=f"M_{lane_id}",
            machine_type=machine_type,
            lane_capacity_gb=_lane_capacity_for_machine(machine_type),
        )
        lane.metadata.update(extra_metadata or {})
        lane.metadata.update(_infer_terminal_lane_constraint_metadata(selected, validator))
        for lib in selected:
            lane.add_library(lib)
        return lane

    def validate_selected(selected: List[EnhancedLibraryInfo]) -> Optional[LaneAssignment]:
        if not selected:
            return None
        total_gb = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in selected)
        min_allowed, max_allowed = _resolve_lane_capacity_limits(
            libraries=selected,
            machine_type=machine_type,
            lane_id=f"{lane_id_prefix}_TMP",
            lane_metadata=extra_metadata,
        )
        nonlocal best_shortage, best_index_pairs
        best_shortage = min(best_shortage, max(0.0, min_allowed - total_gb))
        best_index_pairs = max(best_index_pairs, _count_lane_index_pairs(selected))
        if total_gb < min_allowed - 1e-6 or total_gb > max_allowed + 1e-6:
            return None
        if _count_lane_index_pairs(selected) < AI_LANE_MIN_INDEX_PAIRS:
            return None
        if _validate_index_conflicts_latest(selected):
            return None
        trial_lane = build_lane(selected, f"{lane_id_prefix}_TMP")
        if _is_split_lane_forbidden_by_mode(trial_lane):
            return None
        result = _validate_lane_state(validator, trial_lane, selected)
        if not getattr(result, "is_valid", False):
            filtered_errors = [
                err for err in list(getattr(result, "errors", []) or [])
                if getattr(err, "rule_type", None) != ValidationRuleType.SPECIAL_LIBRARY_LIMIT
            ]
            filtered_warnings = list(getattr(result, "warnings", []) or [])
            if filtered_errors or (getattr(validator, "strict_mode", False) and filtered_warnings):
                for err in filtered_errors:
                    key = _safe_str(getattr(err, "message", None), default="validation_error")
                    validation_failures[key] = validation_failures.get(key, 0) + 1
                if getattr(validator, "strict_mode", False):
                    for warn in filtered_warnings:
                        key = _safe_str(getattr(warn, "message", None), default="validation_warning")
                        validation_failures[key] = validation_failures.get(key, 0) + 1
                return None
        lane_id = f"{lane_id_prefix}_{machine_type.value}_{_reserve_auto_lane_serial(lane_id_prefix, machine_type):03d}"
        return build_lane(selected, lane_id)

    def greedy_from_order(order: List[EnhancedLibraryInfo]) -> Tuple[Optional[LaneAssignment], List[EnhancedLibraryInfo]]:
        selected: List[EnhancedLibraryInfo] = []
        total_gb = 0.0
        for lib in order:
            if _shares_split_family_with_selected(selected, lib):
                continue
            trial_selected = selected + [lib]
            trial_total = total_gb + float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
            _, max_allowed = _resolve_lane_capacity_limits(
                libraries=trial_selected,
                machine_type=machine_type,
                lane_id=f"{lane_id_prefix}_TMP",
                lane_metadata=extra_metadata,
            )
            if trial_total > max_allowed + 1e-6:
                continue
            if _validate_index_conflicts_latest(trial_selected):
                continue
            selected = trial_selected
            total_gb = trial_total
            lane = validate_selected(selected)
            if lane is not None:
                final_used = [
                    lib for lib in list(getattr(lane, "libraries", []) or [])
                    if not _is_ai_balance_library(lib)
                ]
                return lane, final_used
        return None, []

    orders: List[List[EnhancedLibraryInfo]] = []
    orders.append(candidates)
    orders.append(sorted(candidates, key=lambda lib: (float(getattr(lib, "contract_data_raw", 0.0) or 0.0), _safe_str(getattr(lib, "origrec", ""), default=""))))
    orders.append(sorted(candidates, key=lambda lib: (-_count_library_index_pairs(lib), -float(getattr(lib, "contract_data_raw", 0.0) or 0.0))))
    orders.append(sorted(candidates, key=lambda lib: (_count_library_index_pairs(lib), -float(getattr(lib, "contract_data_raw", 0.0) or 0.0))))
    for offset in range(min(24, len(candidates))):
        rotated = candidates[offset:] + candidates[:offset]
        orders.append(rotated)
    for order in orders:
        lane, used = greedy_from_order(order)
        if lane is not None:
            return lane, used, "success"

    if best_shortage < float("inf"):
        top_failures = sorted(validation_failures.items(), key=lambda item: -item[1])[:3]
        return None, [], "no_valid_subset(best_shortage={:.1f}G,best_index_pairs={})".format(
            best_shortage,
            best_index_pairs,
        ) + (", validation_top={}".format(top_failures) if top_failures else "")
    return None, [], "no_valid_subset"


def _try_build_terminal_dedicated_imbalance_lane_from_group(
    pool: List[EnhancedLibraryInfo],
    validator: Any,
    machine_type: MachineType,
    all_lanes: List[LaneAssignment],
    unassigned_pool: List[EnhancedLibraryInfo],
    mode_name: str = "3.6T-NEW",
    max_candidates: int = 120,
) -> Tuple[Optional[LaneAssignment], List[EnhancedLibraryInfo], str]:
    """终态全碱基不均文库专Lane补位，复用现有平衡文库实例化规则。"""
    lane_metadata = {
        "selected_seq_mode": mode_name,
        "seq_mode": mode_name,
        "lcxms": mode_name,
        "dispatch_stage": "terminal_sample_type_dedicated_imbalance",
        "is_dedicated_imbalance_lane": True,
    }
    candidates = [
        lib for lib in list(pool or [])[:max_candidates]
        if _is_imbalance_library_candidate(lib)
    ]
    if not candidates:
        return None, [], "empty_imbalance_pool"
    if len(candidates) != len(list(pool or [])[:max_candidates]):
        return None, [], "mixed_imbalance_pool"

    hard_skip_reason = _terminal_dedicated_pool_hard_skip_reason(
        candidates,
        machine_type=machine_type,
        lane_id_prefix="DL",
        lane_metadata=lane_metadata,
        max_candidates=max_candidates,
    )
    if hard_skip_reason:
        return None, [], hard_skip_reason

    candidates = sorted(
        candidates,
        key=lambda lib: (
            -float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
            -_count_library_index_pairs(lib),
            _safe_str(getattr(lib, "origrec", ""), default=""),
        ),
    )
    best_shortage = float("inf")
    best_index_pairs = 0
    validation_failures: Dict[str, int] = {}

    def build_lane(selected: List[EnhancedLibraryInfo], lane_id: str) -> LaneAssignment:
        lane = LaneAssignment(
            lane_id=lane_id,
            machine_id=f"M_{lane_id}",
            machine_type=machine_type,
            lane_capacity_gb=_lane_capacity_for_machine(machine_type),
        )
        lane.metadata.update(lane_metadata)
        for lib in selected:
            lib._current_seq_mode_raw = mode_name
            lib.selected_seq_mode = mode_name
            lib.current_seq_mode = mode_name
            lib.lcxms = mode_name
            lane.add_library(lib)
        return lane

    def validate_selected(selected: List[EnhancedLibraryInfo]) -> Optional[LaneAssignment]:
        if not selected:
            return None
        total_gb = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in selected)
        min_allowed, max_allowed = _resolve_lane_capacity_limits(
            libraries=selected,
            machine_type=machine_type,
            lane_id="DL_TMP",
            lane_metadata=lane_metadata,
        )
        nonlocal best_shortage, best_index_pairs
        best_shortage = min(best_shortage, max(0.0, min_allowed - total_gb))
        best_index_pairs = max(best_index_pairs, _count_lane_index_pairs(selected))
        if total_gb < min_allowed - 1e-6 or total_gb > max_allowed + 1e-6:
            return None
        if _count_lane_index_pairs(selected) < AI_LANE_MIN_INDEX_PAIRS:
            return None
        if _validate_index_conflicts_latest(selected):
            return None
        lane = build_lane(selected, "DL_TMP")
        required_balance = _resolve_lane_balance_data_gb(lane)
        if required_balance <= 0:
            validation_failures["no_balance_ratio"] = validation_failures.get("no_balance_ratio", 0) + 1
            return None
        if not _materialize_balance_library_for_lane(
            lane=lane,
            all_lanes=all_lanes,
            unassigned_pool=unassigned_pool,
            validator=validator,
        ):
            validation_failures["balance_materialization_failed"] = validation_failures.get("balance_materialization_failed", 0) + 1
            return None
        validation_result = _validate_lane_state(
            validator,
            lane,
            list(lane.libraries or []),
            balance_already_in_libs=True,
            skip_peak_size=True,
        )
        if not getattr(validation_result, "is_valid", False):
            for err in list(getattr(validation_result, "errors", []) or []):
                key = _safe_str(getattr(err, "message", None), default="validation_error")
                validation_failures[key] = validation_failures.get(key, 0) + 1
            return None
        lane_id = f"DL_{machine_type.value}_{_reserve_auto_lane_serial('DL', machine_type):03d}"
        lane.lane_id = lane_id
        lane.machine_id = f"M_{lane_id}"
        return lane

    def greedy_from_order(order: List[EnhancedLibraryInfo]) -> Tuple[Optional[LaneAssignment], List[EnhancedLibraryInfo]]:
        selected: List[EnhancedLibraryInfo] = []
        total_gb = 0.0
        for lib in order:
            trial_selected = selected + [lib]
            trial_total = total_gb + float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
            _, max_allowed = _resolve_lane_capacity_limits(
                libraries=trial_selected,
                machine_type=machine_type,
                lane_id="DL_TMP",
                lane_metadata=lane_metadata,
            )
            if trial_total > max_allowed + 1e-6:
                continue
            if _validate_index_conflicts_latest(trial_selected):
                continue
            selected = trial_selected
            total_gb = trial_total
            lane = validate_selected(selected)
            if lane is not None:
                return lane, selected
        return None, []

    orders = [
        candidates,
        sorted(candidates, key=lambda lib: (float(getattr(lib, "contract_data_raw", 0.0) or 0.0), _safe_str(getattr(lib, "origrec", ""), default=""))),
    ]
    for order in orders:
        lane, used = greedy_from_order(order)
        if lane is not None:
            return lane, used, "success"

    top_failures = sorted(validation_failures.items(), key=lambda item: -item[1])[:3]
    return None, [], "no_valid_dedicated_imbalance_subset(best_shortage={:.1f}G,best_index_pairs={}, validation_top={})".format(
        best_shortage if best_shortage < float("inf") else 0.0,
        best_index_pairs,
        top_failures,
    )


def _resolve_g53_g54_combination_group(libraries: List[EnhancedLibraryInfo]) -> Optional[str]:
    """仅识别G53/G54组合混排，其他1-57组合不作为兜底候选。"""
    types: Set[str] = set()
    for lib in list(libraries or []):
        group_id = _BASE_IMBALANCE_HANDLER.identify_imbalance_type(lib)
        if not group_id or group_id == "G_UNKNOWN":
            return None
        lib_type = _BASE_IMBALANCE_HANDLER._get_library_type(lib)
        if lib_type:
            types.add(lib_type)
    group_id = _BASE_IMBALANCE_HANDLER._resolve_combination_group_for_types(types)
    if group_id in {"G53", "G54"}:
        return group_id
    return None


def _is_terminal_g53_g54_fill_candidate(lib: EnhancedLibraryInfo) -> bool:
    """终态1.1 G53/G54补位候选：未拆分原始文库，当前合同量<=500G。"""
    if _is_ai_balance_library(lib):
        return False
    if _is_split_library(lib):
        return False
    current_data = _safe_float(getattr(lib, "contract_data_raw", None), default=0.0)
    return current_data <= 500.0


def _try_build_terminal_g53_g54_fill_imbalance_lane(
    *,
    base_pool: List[EnhancedLibraryInfo],
    filler_pool: List[EnhancedLibraryInfo],
    validator: Any,
    machine_type: MachineType,
    all_lanes: List[LaneAssignment],
    unassigned_pool: List[EnhancedLibraryInfo],
    mode_name: str = "1.1",
    max_candidates: int = 160,
) -> Tuple[Optional[LaneAssignment], List[EnhancedLibraryInfo], str]:
    """终态1.1不均专Lane差容量时，只用G53/G54组合不均文库补位。"""
    base_candidates = [
        lib for lib in list(base_pool or [])[:max_candidates]
        if _is_imbalance_library_candidate(lib)
        and _is_split_rule_original_allowed_in_1_1(lib)
        and not _is_split_library(lib)
    ]
    if not base_candidates:
        return None, [], "empty_g53_g54_base_pool"

    base_ids = {id(lib) for lib in base_candidates}
    filler_candidates = [
        lib for lib in list(filler_pool or [])
        if id(lib) not in base_ids
        and _is_imbalance_library_candidate(lib)
        and _is_terminal_g53_g54_fill_candidate(lib)
    ][:max_candidates]
    if not filler_candidates:
        return None, [], "empty_g53_g54_filler_pool"

    lane_metadata = {
        "selected_seq_mode": mode_name,
        "seq_mode": mode_name,
        "lcxms": mode_name,
        "dispatch_stage": "terminal_sample_type_g53_g54_imbalance_fill",
        "is_dedicated_imbalance_lane": True,
    }
    base_order = sorted(
        base_candidates,
        key=lambda lib: (
            -float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
            -_count_library_index_pairs(lib),
            _safe_str(getattr(lib, "origrec", ""), default=""),
        ),
    )
    filler_order = sorted(
        filler_candidates,
        key=lambda lib: (
            -float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
            -_count_library_index_pairs(lib),
            _safe_str(getattr(lib, "origrec", ""), default=""),
        ),
    )

    validation_failures: Dict[str, int] = {}
    best_total = 0.0
    best_index_pairs = 0

    def build_lane(selected: List[EnhancedLibraryInfo], lane_id: str) -> LaneAssignment:
        lane = LaneAssignment(
            lane_id=lane_id,
            machine_id=f"M_{lane_id}",
            machine_type=machine_type,
            lane_capacity_gb=_lane_capacity_for_machine(machine_type),
        )
        lane.metadata.update(lane_metadata)
        for lib in selected:
            lib._current_seq_mode_raw = mode_name
            lib.selected_seq_mode = mode_name
            lib.current_seq_mode = mode_name
            lib.lcxms = mode_name
            lane.add_library(lib)
        return lane

    def validate_selected(selected: List[EnhancedLibraryInfo]) -> Optional[LaneAssignment]:
        nonlocal best_total, best_index_pairs
        if not selected:
            return None
        combination_group = _resolve_g53_g54_combination_group(selected)
        if combination_group is None:
            return None
        compatible, reason = _BASE_IMBALANCE_HANDLER.check_mix_compatibility(
            selected,
            enforce_total_limit=False,
        )
        if not compatible:
            validation_failures[reason or "g53_g54_incompatible"] = validation_failures.get(reason or "g53_g54_incompatible", 0) + 1
            return None
        total_gb = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in selected)
        min_allowed, max_allowed = _resolve_lane_capacity_limits(
            libraries=selected,
            machine_type=machine_type,
            lane_id="DLG_TMP",
            lane_metadata=lane_metadata,
        )
        best_total = max(best_total, total_gb)
        best_index_pairs = max(best_index_pairs, _count_lane_index_pairs(selected))
        if total_gb < min_allowed - 1e-6 or total_gb > max_allowed + 1e-6:
            return None
        if _count_lane_index_pairs(selected) < AI_LANE_MIN_INDEX_PAIRS:
            return None
        if _validate_index_conflicts_latest(selected):
            validation_failures["index_conflict"] = validation_failures.get("index_conflict", 0) + 1
            return None
        lane = build_lane(selected, "DLG_TMP")
        required_balance = _resolve_lane_balance_data_gb(lane)
        if required_balance <= 0:
            validation_failures["no_balance_ratio"] = validation_failures.get("no_balance_ratio", 0) + 1
            return None
        if not _materialize_balance_library_for_lane(
            lane=lane,
            all_lanes=all_lanes,
            unassigned_pool=unassigned_pool,
            validator=validator,
        ):
            validation_failures["balance_materialization_failed"] = validation_failures.get("balance_materialization_failed", 0) + 1
            return None
        validation_result = _validate_lane_state(
            validator,
            lane,
            list(lane.libraries or []),
            balance_already_in_libs=True,
            skip_peak_size=True,
        )
        if not getattr(validation_result, "is_valid", False):
            for err in list(getattr(validation_result, "errors", []) or []):
                key = _safe_str(getattr(err, "message", None), default="validation_error")
                validation_failures[key] = validation_failures.get(key, 0) + 1
            return None
        lane_id = f"DLG_{machine_type.value}_{_reserve_auto_lane_serial('DLG', machine_type):03d}"
        lane.lane_id = lane_id
        lane.machine_id = f"M_{lane_id}"
        lane.metadata["g53_g54_combination_group"] = combination_group
        return lane

    selected: List[EnhancedLibraryInfo] = []
    for lib in base_order:
        trial = selected + [lib]
        _, max_allowed = _resolve_lane_capacity_limits(
            libraries=trial,
            machine_type=machine_type,
            lane_id="DLG_TMP",
            lane_metadata=lane_metadata,
        )
        if sum(float(getattr(item, "contract_data_raw", 0.0) or 0.0) for item in trial) > max_allowed + 1e-6:
            continue
        if _validate_index_conflicts_latest(trial):
            continue
        selected = trial
    if not selected:
        return None, [], "no_base_selection_for_g53_g54"

    for filler in filler_order:
        trial = selected + [filler]
        _, max_allowed = _resolve_lane_capacity_limits(
            libraries=trial,
            machine_type=machine_type,
            lane_id="DLG_TMP",
            lane_metadata=lane_metadata,
        )
        if sum(float(getattr(item, "contract_data_raw", 0.0) or 0.0) for item in trial) > max_allowed + 1e-6:
            continue
        if _resolve_g53_g54_combination_group(trial) is None:
            continue
        if _validate_index_conflicts_latest(trial):
            continue
        selected = trial
        lane = validate_selected(selected)
        if lane is not None:
            used = [
                lib for lib in list(getattr(lane, "libraries", []) or [])
                if not _is_ai_balance_library(lib)
            ]
            return lane, used, "success"

    top_failures = sorted(validation_failures.items(), key=lambda item: -item[1])[:3]
    return None, [], "no_valid_g53_g54_fill_subset(best_total={:.1f}G,best_index_pairs={}, validation_top={})".format(
        best_total,
        best_index_pairs,
        top_failures,
    )


def _try_add_terminal_sample_type_dedicated_lanes(
    solution: Any,
    validator: Any,
    max_lanes_per_group: int = 12,
) -> Dict[str, int]:
    """终态未分配池按文库类型增量补Lane，成功才提交，不扰动既有Lane。"""
    stats = {
        "new_lanes": 0,
        "used_libraries": 0,
        "skipped_split_libraries": 0,
        "skipped_unsupported_machine": 0,
        "failed_groups": 0,
        "remaining_unassigned": len(list(getattr(solution, "unassigned_libraries", []) or [])),
    }
    unassigned = list(getattr(solution, "unassigned_libraries", []) or [])
    if not unassigned:
        return stats

    grouped: Dict[Tuple[MachineType, str], List[EnhancedLibraryInfo]] = {}
    passthrough: List[EnhancedLibraryInfo] = []
    for lib in unassigned:
        if _is_split_library(lib):
            stats["skipped_split_libraries"] += 1
            passthrough.append(lib)
            continue
        machine_type, sample_type = _terminal_sample_type_group_key(lib)
        if not _is_machine_supported_for_arrangement(machine_type):
            stats["skipped_unsupported_machine"] += 1
            passthrough.append(lib)
            continue
        grouped.setdefault((machine_type, sample_type), []).append(lib)

    if not grouped:
        solution.unassigned_libraries = passthrough
        return stats

    added_lanes: List[LaneAssignment] = []
    terminal_used_ids: Set[int] = set()
    lane_validation_cache: Dict[
        Tuple[str, Tuple[Tuple[str, str], ...], Tuple[str, ...]],
        Any,
    ] = {}
    mode_options: List[Tuple[str, Dict[str, str]]] = [
        ("1.1", {"selected_seq_mode": "1.1", "seq_mode": "1.1", "lcxms": "1.1"}),
        (
            "3.6T-NEW",
            {
                "selected_seq_mode": "3.6T-NEW",
                "seq_mode": "3.6T-NEW",
                "lcxms": "3.6T-NEW",
            },
        ),
    ]

    def commit_split_fragment_lanes(
        group_pool: List[EnhancedLibraryInfo],
    ) -> Tuple[List[LaneAssignment], Set[int]]:
        split_candidates = _collect_split_source_candidates(
            [
                lib for lib in group_pool
                if _is_split_rule_original_allowed_to_split_in_36t(lib)
            ]
        )
        if not split_candidates:
            return [], set()
        for lane_count in range(8, 1, -1):
            subset = _find_cross_split_source_subset(split_candidates, lane_count)
            if not subset:
                continue
            lanes = _try_pack_cross_split_fragments_into_lanes(
                source_records=subset,
                lane_count=lane_count,
                validator=validator,
                machine_type=machine_type,
                mode_name="3.6T-NEW",
            )
            if not lanes:
                lanes, used_ids = _try_pack_sample_type_split_fragments_greedy(
                    source_records=split_candidates,
                    validator=validator,
                    machine_type=machine_type,
                    lane_id_prefix="TS",
                    max_lanes=max_lanes_per_group,
                )
                if lanes:
                    for lane in lanes:
                        if not isinstance(lane.metadata, dict):
                            lane.metadata = {}
                        lane.metadata["dispatch_stage"] = "terminal_sample_type_split_fragment_greedy"
                        lane.metadata["selected_seq_mode"] = "3.6T-NEW"
                        lane.metadata["seq_mode"] = "3.6T-NEW"
                        lane.metadata["lcxms"] = "3.6T-NEW"
                    return lanes, used_ids
                continue
            used_ids = {id(source) for source, _, _ in subset}
            for lane in lanes:
                if not isinstance(lane.metadata, dict):
                    lane.metadata = {}
                lane.metadata["dispatch_stage"] = "terminal_sample_type_split_dedicated"
                lane.metadata["selected_seq_mode"] = "3.6T-NEW"
                lane.metadata["seq_mode"] = "3.6T-NEW"
                lane.metadata["lcxms"] = "3.6T-NEW"
            return lanes, used_ids
        lanes, used_ids = _try_pack_sample_type_split_fragments_greedy(
            source_records=split_candidates,
            validator=validator,
            machine_type=machine_type,
            lane_id_prefix="TS",
            max_lanes=max_lanes_per_group,
        )
        if lanes:
            return lanes, used_ids
        lanes, used_ids = _try_pack_sample_type_split_fragments_with_fillers(
            split_source_records=split_candidates,
            filler_libraries=[
                lib for lib in group_pool
                if not _is_small_unsplit_original_reserved_for_mode_1_1(lib)
                and not _should_library_split_in_3_6t(lib)
            ],
            validator=validator,
            machine_type=machine_type,
            lane_id_prefix="TS",
            max_lanes=max_lanes_per_group,
        )
        if lanes:
            return lanes, used_ids
        return [], set()

    for (machine_type, sample_type), group_libraries in sorted(
        grouped.items(),
        key=lambda item: (
            -sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in item[1]),
            item[0][0].value,
            item[0][1],
        ),
    ):
        remaining_group = sorted(
            [lib for lib in list(group_libraries) if id(lib) not in terminal_used_ids],
            key=lambda lib: (
                -float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
                _safe_str(getattr(lib, "origrec", ""), default=""),
            ),
        )
        if not remaining_group:
            continue
        group_added = 0
        for mode_name, mode_metadata in mode_options:
            if mode_name == "1.1":
                mode_group = [
                    lib for lib in remaining_group
                    if _is_split_rule_original_allowed_in_1_1(lib)
                ]
            else:
                mode_group = list(remaining_group)
            while remaining_group and group_added < max_lanes_per_group:
                if (
                    mode_name == "1.1"
                    and mode_group
                    and all(_is_imbalance_library_candidate(lib) for lib in mode_group)
                ):
                    lane, used_libraries, failure_reason = _try_build_terminal_dedicated_imbalance_lane_from_group(
                        pool=mode_group,
                        validator=validator,
                        machine_type=machine_type,
                        all_lanes=list(getattr(solution, "lane_assignments", []) or []) + added_lanes,
                        unassigned_pool=passthrough,
                        mode_name=mode_name,
                    )
                    if lane is None or not used_libraries:
                        g53_g54_pool = [
                            lib
                            for _, other_libraries in grouped.items()
                            for lib in other_libraries
                            if id(lib) not in terminal_used_ids
                            and id(lib) not in {id(item) for item in mode_group}
                            and _is_imbalance_library_candidate(lib)
                            and _is_terminal_g53_g54_fill_candidate(lib)
                        ]
                        g53_g54_lane, g53_g54_used, g53_g54_reason = _try_build_terminal_g53_g54_fill_imbalance_lane(
                            base_pool=mode_group,
                            filler_pool=g53_g54_pool,
                            validator=validator,
                            machine_type=machine_type,
                            all_lanes=list(getattr(solution, "lane_assignments", []) or []) + added_lanes,
                            unassigned_pool=passthrough,
                            mode_name=mode_name,
                        )
                        if g53_g54_lane is not None and g53_g54_used:
                            lane = g53_g54_lane
                            used_libraries = g53_g54_used
                            failure_reason = "success_g53_g54_fill"
                        else:
                            failure_reason = f"{failure_reason}; g53_g54_fill={g53_g54_reason}"

                    if lane is None or not used_libraries:
                        if group_added == 0:
                            stats["failed_groups"] += 1
                            logger.info(
                                "终态文库类型碱基不均专Lane未成Lane: 文库类型={}, 机型={}, 剩余文库={}, 剩余数据量={:.1f}G, reason={}".format(
                                    sample_type,
                                    machine_type.value,
                                    len(remaining_group),
                                    sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in remaining_group),
                                    failure_reason,
                                )
                            )
                        break
                    used_ids = {id(lib) for lib in used_libraries}
                    terminal_used_ids.update(used_ids)
                    passthrough = [
                        lib for lib in passthrough
                        if id(lib) not in used_ids
                    ]
                    added_lanes.append(lane)
                    remaining_group = [
                        lib for lib in remaining_group
                        if id(lib) not in used_ids
                    ]
                    mode_group = [
                        lib for lib in mode_group
                        if id(lib) not in used_ids
                    ]
                    group_added += 1
                    stats["new_lanes"] += 1
                    stats["used_libraries"] += len(used_libraries)
                    logger.info(
                        "终态文库类型碱基不均专Lane补Lane成功: lane={}, 文库类型={}, 文库数={}, 数据量={:.1f}G".format(
                            lane.lane_id,
                            sample_type,
                            len(used_libraries),
                            float(getattr(lane, "total_data_gb", 0.0) or 0.0),
                        )
                    )
                    continue
                if mode_name == "3.6T-NEW":
                    split_lanes, split_used_ids = commit_split_fragment_lanes(remaining_group)
                    if split_lanes:
                        added_lanes.extend(split_lanes)
                        terminal_used_ids.update(split_used_ids)
                        remaining_group = [
                            lib for lib in remaining_group
                            if id(lib) not in split_used_ids
                        ]
                        group_added += len(split_lanes)
                        stats["new_lanes"] += len(split_lanes)
                        stats["used_libraries"] += len(split_used_ids)
                        logger.info(
                            "终态文库类型拆分专池补Lane成功: 文库类型={}, 机型={}, 新增Lane={}, 使用原始文库={}, 剩余文库={}".format(
                                sample_type,
                                machine_type.value,
                                len(split_lanes),
                                len(split_used_ids),
                                len(remaining_group),
                            )
                        )
                        continue
                    mode_group = [
                        lib for lib in remaining_group
                        if not _should_library_split_in_3_6t(lib)
                        and not _is_small_unsplit_original_reserved_for_mode_1_1(lib)
                    ]
                if not mode_group:
                    if group_added == 0:
                        stats["failed_groups"] += 1
                        logger.info(
                            "终态文库类型专池未成Lane: 文库类型={}, 机型={}, 模式={}, 剩余文库={}, 剩余数据量={:.1f}G, reason={}".format(
                                sample_type,
                                machine_type.value,
                                mode_name,
                                len(remaining_group),
                                sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in remaining_group),
                                "no_mode_eligible_libraries",
                            )
                        )
                    break
                lane_metadata = {
                    **mode_metadata,
                    "dispatch_stage": "terminal_sample_type_dedicated",
                    "terminal_sample_type": sample_type,
                }
                lane, used_libraries, failure_reason = _attempt_build_terminal_dedicated_lane_from_group(
                    pool=mode_group,
                    validator=validator,
                    machine_type=machine_type,
                    lane_id_prefix="TG",
                    extra_metadata=lane_metadata,
                )
                if lane is None or not used_libraries:
                    if group_added == 0:
                        stats["failed_groups"] += 1
                        logger.info(
                            "终态文库类型专池未成Lane: 文库类型={}, 机型={}, 模式={}, 剩余文库={}, 剩余数据量={:.1f}G, reason={}".format(
                                sample_type,
                                machine_type.value,
                                mode_name,
                                len(remaining_group),
                                sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in remaining_group),
                                failure_reason,
                            )
                        )
                    break
                if _is_split_lane_forbidden_by_mode(lane):
                    break
                validation_result = _validate_lane_state(
                    validator,
                    lane,
                    list(lane.libraries or []),
                )
                if not getattr(validation_result, "is_valid", False):
                    filtered_errors = [
                        err for err in list(getattr(validation_result, "errors", []) or [])
                        if getattr(err, "rule_type", None) != ValidationRuleType.SPECIAL_LIBRARY_LIMIT
                    ]
                    filtered_warnings = list(getattr(validation_result, "warnings", []) or [])
                    if filtered_errors or (getattr(validator, "strict_mode", False) and filtered_warnings):
                        break

                used_ids = {id(lib) for lib in used_libraries}
                terminal_used_ids.update(used_ids)
                for lib in list(lane.libraries or []):
                    lib._current_seq_mode_raw = mode_name
                    lib.selected_seq_mode = mode_name
                    lib.current_seq_mode = mode_name
                    lib.lcxms = mode_name
                added_lanes.append(lane)
                remaining_group = [
                    lib for lib in remaining_group
                    if id(lib) not in used_ids
                ]
                mode_group = [
                    lib for lib in mode_group
                    if id(lib) not in used_ids
                ]
                group_added += 1
                stats["new_lanes"] += 1
                stats["used_libraries"] += len(used_libraries)
                logger.info(
                    "终态文库类型专池补Lane成功: lane={}, 文库类型={}, 模式={}, 文库数={}, 数据量={:.1f}G".format(
                        lane.lane_id,
                        sample_type,
                        mode_name,
                        len(used_libraries),
                        float(getattr(lane, "total_data_gb", 0.0) or 0.0),
                    )
                )
        passthrough.extend(remaining_group)

    if added_lanes:
        solution.lane_assignments.extend(added_lanes)
    solution.unassigned_libraries = passthrough
    stats["remaining_unassigned"] = len(solution.unassigned_libraries)
    return stats


def _try_add_terminal_global_36t_mixed_lanes(
    solution: Any,
    validator: Any,
    max_lanes: int = 32,
) -> Dict[str, int]:
    """终态全局3.6混排救援：跨桶组合已通过1.1门禁的3.6T候选。"""
    stats = {
        "new_lanes": 0,
        "used_originals": 0,
        "used_fillers": 0,
        "remaining_unassigned": len(list(getattr(solution, "unassigned_libraries", []) or [])),
    }
    unassigned = list(getattr(solution, "unassigned_libraries", []) or [])
    if not unassigned:
        return stats

    pool: List[EnhancedLibraryInfo] = []
    passthrough: List[EnhancedLibraryInfo] = []
    for lib in unassigned:
        if _is_split_library(lib):
            passthrough.append(lib)
            continue
        pool.append(lib)

    added_lanes: List[LaneAssignment] = []
    used_ids: Set[int] = set()

    def resolve_terminal_lane_machine_type(libraries: List[EnhancedLibraryInfo]) -> MachineType:
        for item in libraries:
            machine_type = _resolve_machine_type_enum_simple(
                _safe_str(getattr(item, "eq_type", None), default="")
            )
            if _is_machine_supported_for_arrangement(machine_type):
                return machine_type
        return MachineType.NOVA_X_25B

    if pool:
        machine_type = resolve_terminal_lane_machine_type(pool)
        remaining_pool = [lib for lib in pool if id(lib) not in used_ids]
        split_candidates = _collect_split_source_candidates(
            [
                lib for lib in remaining_pool
                if _is_split_rule_original_allowed_to_split_in_36t(lib)
                and _is_terminal_36t_candidate_after_1_1_gate(lib)
            ]
        )
        filler_libraries = [
            lib for lib in remaining_pool
            if _is_normal_filler_allowed_for_36t(lib)
        ]
        if not split_candidates and not filler_libraries:
            solution.unassigned_libraries = passthrough + [
                lib for lib in unassigned
                if id(lib) not in used_ids and id(lib) not in {id(item) for item in passthrough}
            ]
            stats["remaining_unassigned"] = len(solution.unassigned_libraries)
            return stats

        split_pack_added = 0
        while split_pack_added < max_lanes:
            remaining_pool = [lib for lib in pool if id(lib) not in used_ids]
            machine_type = resolve_terminal_lane_machine_type(remaining_pool)
            split_candidates = _collect_split_source_candidates(
                [
                    lib for lib in remaining_pool
                    if _is_split_rule_original_allowed_to_split_in_36t(lib)
                    and _is_terminal_36t_candidate_after_1_1_gate(lib)
                ]
            )
            filler_libraries = [
                lib for lib in remaining_pool
                if _is_normal_filler_allowed_for_36t(lib)
            ]
            if not split_candidates:
                break
            split_lanes, split_used_ids = _try_pack_sample_type_split_fragments_with_fillers(
                split_source_records=split_candidates,
                filler_libraries=filler_libraries,
                validator=validator,
                machine_type=machine_type,
                lane_id_prefix="GM",
                max_lanes=max_lanes - split_pack_added,
            )
            if not split_lanes or not split_used_ids:
                break
            for lane in split_lanes:
                if not isinstance(lane.metadata, dict):
                    lane.metadata = {}
                lane.metadata["dispatch_stage"] = "terminal_global_36t_split_mixed"
                lane.metadata["selected_seq_mode"] = "3.6T-NEW"
                lane.metadata["seq_mode"] = "3.6T-NEW"
                lane.metadata["lcxms"] = "3.6T-NEW"
            added_lanes.extend(split_lanes)
            used_ids.update(split_used_ids)
            stats["new_lanes"] += len(split_lanes)
            split_source_ids = {id(source) for source, _, _ in split_candidates}
            stats["used_originals"] += len(split_used_ids & split_source_ids)
            stats["used_fillers"] += len(split_used_ids - split_source_ids)
            split_pack_added += len(split_lanes)
            logger.info(
                "终态全局3.6跨机型混排补Lane成功: 承载机型={}, 新增Lane={}, 使用原始/补料文库={}，剩余候选={}".format(
                    machine_type.value,
                    len(split_lanes),
                    len(split_used_ids),
                    len(remaining_pool) - len(split_used_ids),
                )
            )

        incremental_added = 0
        while split_pack_added <= 0 and incremental_added < max_lanes:
            remaining_pool = [lib for lib in pool if id(lib) not in used_ids]
            machine_type = resolve_terminal_lane_machine_type(remaining_pool)
            split_candidates = _collect_split_source_candidates(
                [
                    lib for lib in remaining_pool
                    if _is_split_rule_original_allowed_to_split_in_36t(lib)
                    and _is_terminal_36t_candidate_after_1_1_gate(lib)
                ]
            )
            filler_libraries = [
                lib for lib in remaining_pool
                if _is_normal_filler_allowed_for_36t(lib)
            ]
            if not split_candidates:
                filler_libraries = [
                    lib for lib in filler_libraries
                    if _is_terminal_36t_candidate_after_1_1_gate(lib)
                ]
            lane, lane_used_ids = _try_build_single_36t_mixed_lane_from_items(
                source_records=split_candidates,
                filler_libraries=filler_libraries,
                validator=validator,
                machine_type=machine_type,
                lane_id_prefix="GM",
                require_36t_main_library=True,
            )
            if lane is None or not lane_used_ids:
                break
            added_lanes.append(lane)
            used_ids.update(lane_used_ids)
            split_source_ids = {id(source) for source, _, _ in split_candidates}
            stats["new_lanes"] += 1
            stats["used_originals"] += len(lane_used_ids & split_source_ids)
            stats["used_fillers"] += len(lane_used_ids - split_source_ids)
            incremental_added += 1
            logger.info(
                "终态全局3.6跨机型增量混排补Lane成功: lane={}, 承载机型={}, 文库数={}, 数据量={:.1f}G".format(
                    lane.lane_id,
                    machine_type.value,
                    len(list(lane.libraries or [])),
                    float(getattr(lane, "total_data_gb", 0.0) or 0.0),
                )
            )

        dedicated_imbalance_pool = [
            lib for lib in filler_libraries
            if id(lib) not in used_ids and _is_imbalance_library_candidate(lib)
        ]
        if split_pack_added <= 0 and incremental_added <= 0 and dedicated_imbalance_pool:
            dedicated_lanes, remaining_after_dedicated = _extract_global_dedicated_imbalance_lanes(
                dedicated_imbalance_pool
            )
            dedicated_used_ids = {
                id(lib)
                for lane in dedicated_lanes
                for lib in list(getattr(lane, "libraries", []) or [])
            }
            materialized_dedicated_lanes: List[LaneAssignment] = []
            balance_pool = [
                lib for lib in unassigned
                if id(lib) not in used_ids and id(lib) not in dedicated_used_ids
            ]
            for lane in dedicated_lanes:
                if not isinstance(lane.metadata, dict):
                    lane.metadata = {}
                lane.metadata["dispatch_stage"] = "terminal_global_dedicated_imbalance"
                required_balance = _resolve_lane_balance_data_gb(lane)
                if required_balance > 0 and not _materialize_balance_library_for_lane(
                    lane=lane,
                    all_lanes=list(getattr(solution, "lane_assignments", []) or []) + added_lanes + materialized_dedicated_lanes,
                    unassigned_pool=balance_pool,
                    validator=validator,
                ):
                    logger.warning(
                        "终态全局碱基不均专Lane补平衡失败，跳过提交: lane={}, 需补平衡={:.3f}G".format(
                            lane.lane_id,
                            required_balance,
                        )
                    )
                    continue
                validation_result = _validate_lane_state(
                    validator,
                    lane,
                    list(getattr(lane, "libraries", []) or []),
                    balance_already_in_libs=any(
                        _is_ai_balance_library(lib)
                        for lib in list(getattr(lane, "libraries", []) or [])
                    ),
                    skip_peak_size=True,
                )
                if not getattr(validation_result, "is_valid", False):
                    logger.warning(
                        "终态全局碱基不均专Lane终态校验失败，跳过提交: lane={}, errors={}".format(
                            lane.lane_id,
                            [
                                _safe_str(getattr(err, "message", None), default="")
                                for err in list(getattr(validation_result, "errors", []) or [])
                            ],
                        )
                    )
                    continue
                materialized_dedicated_lanes.append(lane)

            if materialized_dedicated_lanes:
                lane_used_ids = {
                    id(lib)
                    for lane in materialized_dedicated_lanes
                    for lib in list(getattr(lane, "libraries", []) or [])
                    if not _is_ai_balance_library(lib)
                }
                added_lanes.extend(materialized_dedicated_lanes)
                used_ids.update(lane_used_ids)
                stats["new_lanes"] += len(materialized_dedicated_lanes)
                stats["used_fillers"] += len(lane_used_ids)
                logger.info(
                    "终态全局跨机型碱基不均专Lane补Lane成功: 承载机型={}, 新增Lane={}, 使用文库={}, 剩余不均衡候选={}".format(
                        machine_type.value,
                        len(materialized_dedicated_lanes),
                        len(lane_used_ids),
                        len(remaining_after_dedicated),
                    )
                )

        filler_pool = [
            lib for lib in filler_libraries
            if id(lib) not in used_ids
            and not _is_imbalance_library_candidate(lib)
            and _is_terminal_36t_candidate_after_1_1_gate(lib)
        ]
        group_added = 0
        while split_pack_added <= 0 and incremental_added <= 0 and filler_pool and group_added < max_lanes:
            machine_type = resolve_terminal_lane_machine_type(filler_pool)
            lane_metadata = {
                "selected_seq_mode": "3.6T-NEW",
                "seq_mode": "3.6T-NEW",
                "lcxms": "3.6T-NEW",
                "dispatch_stage": "terminal_global_36t_mixed",
            }
            lane, used_libraries, _ = _attempt_build_terminal_dedicated_lane_from_group(
                pool=filler_pool,
                validator=validator,
                machine_type=machine_type,
                lane_id_prefix="GM",
                extra_metadata=lane_metadata,
                max_candidates=360,
            )
            if lane is None or not used_libraries:
                break
            lane_used_ids = {id(lib) for lib in used_libraries}
            added_lanes.append(lane)
            used_ids.update(lane_used_ids)
            filler_pool = [lib for lib in filler_pool if id(lib) not in lane_used_ids]
            group_added += 1
            stats["new_lanes"] += 1
            stats["used_fillers"] += len(lane_used_ids)
            logger.info(
                "终态全局3.6跨机型普通混排补Lane成功: lane={}, 承载机型={}, 文库数={}, 数据量={:.1f}G".format(
                    lane.lane_id,
                    machine_type.value,
                    len(used_libraries),
                    float(getattr(lane, "total_data_gb", 0.0) or 0.0),
                )
            )

    if added_lanes:
        solution.lane_assignments.extend(added_lanes)
    solution.unassigned_libraries = [
        lib for lib in unassigned
        if id(lib) not in used_ids
    ]
    stats["remaining_unassigned"] = len(solution.unassigned_libraries)
    return stats


def _proactively_build_split_family_lanes_from_pool(
    libraries: List[EnhancedLibraryInfo],
    validator: Any,
    stage_label: str = "3.6T-NEW前置拆分",
) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo], Dict[str, int]]:
    """在普通排机前优先为应拆分原始文库整组构建完整Lane。"""
    from types import SimpleNamespace

    working_solution = SimpleNamespace(
        lane_assignments=[],
        unassigned_libraries=list(libraries or []),
    )
    stats = {
        "matrix_added_lanes": 0,
        "matrix_used_originals": 0,
        "mixed_matrix_added_lanes": 0,
        "mixed_matrix_used_originals": 0,
        "cross_split_added_lanes": 0,
        "cross_split_used_originals": 0,
        "rounds": 0,
    }

    while True:
        stats["rounds"] += 1
        matrix_stats = _try_add_matrix_split_lanes_from_unassigned(
            solution=working_solution,
            validator=validator,
        )
        mixed_stats = _try_add_mixed_matrix_split_lanes_from_unassigned(
            solution=working_solution,
            validator=validator,
        )
        cross_stats = _try_add_cross_split_fragment_lanes_from_unassigned(
            solution=working_solution,
            validator=validator,
        )
        stats["matrix_added_lanes"] += int(matrix_stats.get("added_lanes", 0) or 0)
        stats["matrix_used_originals"] += int(matrix_stats.get("used_originals", 0) or 0)
        stats["mixed_matrix_added_lanes"] += int(mixed_stats.get("added_lanes", 0) or 0)
        stats["mixed_matrix_used_originals"] += int(mixed_stats.get("used_originals", 0) or 0)
        stats["cross_split_added_lanes"] += int(cross_stats.get("added_lanes", 0) or 0)
        stats["cross_split_used_originals"] += int(cross_stats.get("used_originals", 0) or 0)
        if (
            int(matrix_stats.get("added_lanes", 0) or 0) <= 0
            and int(mixed_stats.get("added_lanes", 0) or 0) <= 0
            and int(cross_stats.get("added_lanes", 0) or 0) <= 0
        ):
            stats["rounds"] -= 1
            break

    if working_solution.lane_assignments:
        for lane in working_solution.lane_assignments:
            if not isinstance(lane.metadata, dict):
                lane.metadata = {}
            lane.metadata["dispatch_stage"] = "proactive_split_family_build"
            lane.metadata["selected_seq_mode"] = "3.6T-NEW"
            lane.metadata["lcxms"] = "3.6T-NEW"
            for lib in list(lane.libraries or []):
                lib._current_seq_mode_raw = "3.6T-NEW"
        logger.info(
            "{}完成: 新增Lane={}，矩阵Lane={}，混合矩阵Lane={}，跨份数Lane={}，使用原始文库={}",
            stage_label,
            len(working_solution.lane_assignments),
            stats["matrix_added_lanes"],
            stats["mixed_matrix_added_lanes"],
            stats["cross_split_added_lanes"],
            stats["matrix_used_originals"] + stats["mixed_matrix_used_originals"] + stats["cross_split_used_originals"],
        )
    else:
        logger.info("{}未构建出完整拆分Lane", stage_label)

    return (
        list(working_solution.lane_assignments or []),
        list(working_solution.unassigned_libraries or []),
        stats,
    )


def _resolve_library_split_count_for_3_6t(lib: EnhancedLibraryInfo) -> int:
    """按拆分器规则解析3.6T-NEW上下文下应拆份数。"""
    splitter = LibrarySplitter()
    eval_lib = deepcopy(lib)
    eval_lib._current_seq_mode_raw = "3.6T-NEW"
    eval_lib.selected_seq_mode = "3.6T-NEW"
    eval_lib.current_seq_mode = "3.6T-NEW"
    eval_lib.lcxms = "3.6T-NEW"
    if not splitter._should_split(eval_lib):
        return 1
    fragments = splitter._perform_split(eval_lib)
    return len(fragments) if len(fragments) > 1 else 1


def _try_build_mixed_matrix_split_lanes_from_sources(
    *,
    source_libraries: List[EnhancedLibraryInfo],
    validator: Any,
) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo]]:
    """按拆分器给出的份数混合矩阵成Lane。"""
    if len(source_libraries) < 2:
        return [], []

    grouped: Dict[int, List[EnhancedLibraryInfo]] = {}
    for lib in source_libraries:
        split_count = _resolve_library_split_count_for_3_6t(lib)
        if split_count <= 1:
            continue
        grouped.setdefault(split_count, []).append(lib)

    for split_count, libs in sorted(grouped.items(), key=lambda item: (-item[0], -len(item[1]))):
        ordered_sources = sorted(
            list(libs),
            key=lambda lib: float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
            reverse=True,
        )
        for group_size in range(len(ordered_sources), 1, -1):
            group = ordered_sources[:group_size]
            data_per_lane = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in group) / split_count
            if data_per_lane < 995.0 - 1e-6 or data_per_lane > 1105.0 + 1e-6:
                continue
            lanes = _try_build_matrix_split_lanes_for_group(
                source_libraries=group,
                split_count=split_count,
                validator=validator,
                lane_id_prefix="MS",
            )
            if lanes:
                return lanes, group
    return [], []


def _final_non_package_validation_cleanup(
    solution: Any,
    validator: Any,
) -> Dict[str, int]:
    """平衡文库实例化完成后，对所有非包Lane做一次终态严格校验。

    经过优先级收口、rescue、平衡文库注入等多轮后处理之后，部分Lane的实际合同量
    可能已偏离最初校验时的状态，导致输出存在不合规Lane。这一步作为最后一道门，
    确保进入输出的非包Lane全部符合规则；不合规的Lane整体回退到未分配池（去掉
    AI平衡文库本体，只保留原始合同文库）。

    返回统计字典：
    - removed_lanes: 被淘汰的Lane数
    - recovered_libs: 回收到未分配池的原始文库数
    """
    package_lanes = [l for l in (solution.lane_assignments or []) if _is_package_lane_assignment(l)]
    non_package_lanes = [l for l in (solution.lane_assignments or []) if not _is_package_lane_assignment(l)]

    if not non_package_lanes:
        return {"removed_lanes": 0, "recovered_libs": 0}

    valid_lanes, failed_lanes = _filter_valid_lanes(non_package_lanes, validator)
    repair_stats = _repair_failed_lanes_before_final_cleanup(solution, failed_lanes, validator)
    if repair_stats["repaired_lanes"] > 0:
        logger.info(
            "终态淘汰前修复完成: 尝试{}条，修复{}条，补入{}个文库".format(
                repair_stats["attempted_lanes"],
                repair_stats["repaired_lanes"],
                repair_stats["added_libraries"],
            )
        )
        valid_lanes, failed_lanes = _filter_valid_lanes(non_package_lanes, validator)
    recovered_libs = 0
    recovered_split_family_ids: Set[str] = set()
    recovered_source_keys: Set[str] = {
        _get_library_source_origrec_key(lib)
        for lib in list(getattr(solution, "unassigned_libraries", []) or [])
        if not _is_split_library(lib)
    }
    for lane in failed_lanes:
        # 只回收原始合同文库，AI生成的平衡文库不放回未分配池
        original_libs: List[EnhancedLibraryInfo] = []
        for lib in list(lane.libraries or []):
            if _is_ai_balance_library(lib):
                continue
            if _is_split_library(lib):
                family_id = _get_split_family_id_for_lane_build(lib)
                source_library = getattr(lib, "_split_source_library", None)
                if source_library is not None:
                    source_library.is_split = False
                    source_library.wkissplit = ""
                    source_library.split_status = "rolled_back"
                    source_library.original_library_id = ""
                    source_library.fragment_index = 0
                    source_library.total_fragments = 0
                    source_library.fragment_id = ""
                    if family_id:
                        recovered_split_family_ids.add(family_id)
                    source_key = _get_library_source_origrec_key(source_library)
                    if not source_key or source_key not in recovered_source_keys:
                        original_libs.append(source_library)
                        if source_key:
                            recovered_source_keys.add(source_key)
                    continue
                # 没有 source 上下文时才保留片段，后续拆分兜底复核继续处理。
            original_libs.append(lib)

        solution.unassigned_libraries.extend(original_libs)
        recovered_libs += len(original_libs)
        logger.warning(
            "终态总复核淘汰Lane {}: 回收{}个原始文库到未分配池".format(
                lane.lane_id, len(original_libs)
            )
        )

    solution.lane_assignments = package_lanes + valid_lanes
    if recovered_split_family_ids:
        solution.unassigned_libraries = [
            lib for lib in list(getattr(solution, "unassigned_libraries", []) or [])
            if not (
                _is_split_library(lib)
                and _get_split_family_id_for_lane_build(lib) in recovered_split_family_ids
            )
        ]
    if failed_lanes:
        logger.info(
            "终态总复核完成: 淘汰{}条不合规非包Lane，回收{}个文库到未分配池，"
            "当前有效Lane={}，未分配={}".format(
                len(failed_lanes),
                recovered_libs,
                len(solution.lane_assignments),
                len(getattr(solution, "unassigned_libraries", []) or []),
            )
        )
    return {"removed_lanes": len(failed_lanes), "recovered_libs": recovered_libs}


def _rescue_failed_lanes_by_57_rules(
    failed_lanes: List[LaneAssignment],
    solution: Any,
    validator: Any,
    machine_type: MachineType = MachineType.NOVA_X_25B,
) -> Dict[str, int]:
    """对57规则失败Lane回收文库后做定向二次改排。"""
    if not failed_lanes:
        return {"failed_lanes": 0, "rescued_lanes": 0, "recovered_libraries": 0, "remaining_unassigned": len(solution.unassigned_libraries)}

    failed_lane_ids = {lane.lane_id for lane in failed_lanes}
    # 失败Lane在严格校验阶段可能已经从 solution.lane_assignments 中剔除了，
    # 此处必须直接以 failed_lanes 参数为准回收文库，否则会导致整条失败Lane的文库漏出结果文件。
    recovered_libraries: List[EnhancedLibraryInfo] = []
    for lane in failed_lanes:
        recovered_libraries.extend(list(lane.libraries or []))

    solution.lane_assignments = [
        lane for lane in solution.lane_assignments
        if lane.lane_id not in failed_lane_ids
    ]
    rescue_primary_pool = list(recovered_libraries)
    rescue_secondary_pool = list(solution.unassigned_libraries)
    solution.unassigned_libraries = []

    rescued_lanes: List[LaneAssignment] = []
    rescue_index = 1
    lane_validation_cache: Dict[
        Tuple[str, Tuple[Tuple[str, str], ...], Tuple[str, ...]],
        Any,
    ] = {}

    def _gid_matcher(group_id: str):
        return lambda lib: _BASE_IMBALANCE_HANDLER.identify_imbalance_type(lib) == group_id

    def _single_imbalance_type_match(lib: EnhancedLibraryInfo) -> bool:
        if not _BASE_IMBALANCE_HANDLER.is_imbalance_library(lib):
            return False
        gid = _BASE_IMBALANCE_HANDLER.identify_imbalance_type(lib)
        return gid not in {"G53", "G54", None, "G_UNKNOWN"}

    g53_lanes, rescue_primary_pool, rescue_secondary_pool, rescue_index = _drain_rescue_lanes_for_match(
        primary_pool=rescue_primary_pool,
        secondary_pool=rescue_secondary_pool,
        validator=validator,
        machine_type=machine_type,
        lane_prefix="RG53",
        serial_start=rescue_index,
        match_fn=_gid_matcher("G53"),
        extra_metadata={"is_dedicated_imbalance_lane": True},
        lane_validation_cache=lane_validation_cache,
    )
    rescued_lanes.extend(g53_lanes)

    g54_lanes, rescue_primary_pool, rescue_secondary_pool, rescue_index = _drain_rescue_lanes_for_match(
        primary_pool=rescue_primary_pool,
        secondary_pool=rescue_secondary_pool,
        validator=validator,
        machine_type=machine_type,
        lane_prefix="RG54",
        serial_start=rescue_index,
        match_fn=_gid_matcher("G54"),
        extra_metadata={"is_dedicated_imbalance_lane": True},
        lane_validation_cache=lane_validation_cache,
    )
    rescued_lanes.extend(g54_lanes)

    single_type_lanes, rescue_primary_pool, rescue_secondary_pool, rescue_index = _drain_rescue_lanes_for_match(
        primary_pool=rescue_primary_pool,
        secondary_pool=rescue_secondary_pool,
        validator=validator,
        machine_type=machine_type,
        lane_prefix="RG1",
        serial_start=rescue_index,
        match_fn=_single_imbalance_type_match,
        extra_metadata={"is_dedicated_imbalance_lane": True},
        lane_validation_cache=lane_validation_cache,
    )
    rescued_lanes.extend(single_type_lanes)

    while True:
        rescue_lane, used = _attempt_build_lane_from_prioritized_pool(
            primary_pool=rescue_primary_pool,
            secondary_pool=rescue_secondary_pool,
            validator=validator,
            machine_type=machine_type,
            lane_id_prefix="RS",
            lane_serial=rescue_index,
            extra_metadata={"is_dedicated_imbalance_lane": True},
            lane_validation_cache=lane_validation_cache,
        )
        if rescue_lane is None or not used:
            break
        rescued_lanes.append(rescue_lane)
        rescue_index += 1
        rescue_primary_pool, rescue_secondary_pool = _remove_used_libraries_from_pools(
            rescue_primary_pool,
            rescue_secondary_pool,
            used,
        )

    valid_rescued_lanes, failed_rescued_lanes = _filter_valid_lanes(rescued_lanes, validator)
    if failed_rescued_lanes:
        for lane in failed_rescued_lanes:
            rescue_secondary_pool.extend(list(lane.libraries or []))

    solution.lane_assignments.extend(valid_rescued_lanes)
    solution.unassigned_libraries = rescue_primary_pool + rescue_secondary_pool
    return {
        "failed_lanes": len(failed_lanes),
        "rescued_lanes": len(valid_rescued_lanes),
        "recovered_libraries": len(recovered_libraries),
        "remaining_unassigned": len(solution.unassigned_libraries),
    }


def _pick_special_split_removals(
    libraries: List[EnhancedLibraryInfo],
) -> List[EnhancedLibraryInfo]:
    """根据wkspecialsplits规则选择需剔除的文库（尽量最小移除）。"""
    if not libraries:
        return []

    mode_records: List[Tuple[EnhancedLibraryInfo, str]] = [
        (lib, _classify_library_special_split_mode(lib)) for lib in libraries
    ]
    data_by_mode: Dict[str, float] = {"A": 0.0, "B": 0.0, "EMPTY": 0.0, "OTHER": 0.0}
    for lib, mode in mode_records:
        data_by_mode[mode] += float(getattr(lib, "contract_data_raw", 0.0) or 0.0)

    to_remove: List[EnhancedLibraryInfo] = [lib for lib, mode in mode_records if mode == "OTHER"]
    removal_ids = _build_library_object_id_set(to_remove)
    remaining = [lib for lib in libraries if id(lib) not in removal_ids]
    remaining_modes = {_classify_library_special_split_mode(lib) for lib in remaining}

    has_a = "A" in remaining_modes
    has_b = "B" in remaining_modes
    has_empty = "EMPTY" in remaining_modes

    # A/B混排：优先保留数据量更大的组
    if has_a and has_b:
        keep_mode = "A" if data_by_mode["A"] >= data_by_mode["B"] else "B"
        to_remove.extend(
            [lib for lib in remaining if _classify_library_special_split_mode(lib) != keep_mode]
        )
        return to_remove

    # B组不能与其他同Lane：比较“保留B”与“剔除B”两种成本，选移除量更小者
    if has_b and has_empty:
        remove_non_b = [lib for lib in remaining if _classify_library_special_split_mode(lib) != "B"]
        remove_b = [lib for lib in remaining if _classify_library_special_split_mode(lib) == "B"]
        remove_non_b_data = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in remove_non_b)
        remove_b_data = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in remove_b)
        if remove_non_b_data <= remove_b_data:
            to_remove.extend(remove_non_b)
        else:
            to_remove.extend(remove_b)
        return to_remove

    return to_remove


def _auto_fix_lane_for_special_splits(
    lane: LaneAssignment,
    strict_validator: Any,
    all_lanes: List[LaneAssignment],
    unassigned_pool: List[EnhancedLibraryInfo],
) -> Dict[str, int]:
    """在排机过程中对wkspecialsplits违规Lane执行剔除+局部交换。"""
    stats = {"changed": 0, "removed": 0, "swapped_in": 0}
    is_valid, _, reason = _validate_lane_special_split_rule(lane.libraries)
    if is_valid:
        return stats

    removals = _pick_special_split_removals(list(lane.libraries))
    if not removals:
        logger.warning(f"Lane {lane.lane_id} special_splits违规({reason})但未选出可剔除文库")
        return stats

    for lib in removals:
        if lane.remove_library(lib):
            unassigned_pool.append(lib)
            stats["removed"] += 1

    lane.calculate_metrics()
    stats["changed"] = 1

    min_allowed, max_allowed = _resolve_lane_capacity_limits(
        libraries=lane.libraries,
        machine_type=lane.machine_type.value if lane.machine_type else "Nova X-25B",
        lane_id=lane.lane_id,
        lane_metadata=lane.metadata,
    )

    def can_add(candidate: EnhancedLibraryInfo) -> bool:
        candidate_data = float(getattr(candidate, "contract_data_raw", 0.0) or 0.0)
        if lane.total_data_gb + candidate_data > max_allowed:
            return False
        trial_libs = list(lane.libraries) + [candidate]
        ss_valid, _, _ = _validate_lane_special_split_rule(trial_libs)
        if not ss_valid:
            return False
        imbalance_mix_valid, _ = _validate_lane_57_mix_rules(
            trial_libs,
            enforce_total_limit=False,
            lane_id=lane.lane_id,
            lane_metadata=lane.metadata,
        )
        if not imbalance_mix_valid:
            return False
        trial_result = _validate_lane_state(strict_validator, lane, trial_libs)
        return bool(trial_result.is_valid)

    # 先从未分配池补齐
    for lib in sorted(list(unassigned_pool), key=lambda x: float(getattr(x, "contract_data_raw", 0.0) or 0.0), reverse=True):
        if lane.total_data_gb >= min_allowed:
            break
        if not can_add(lib):
            continue
        lane.add_library(lib)
        _remove_library_by_identity_in_place(unassigned_pool, lib)
        stats["swapped_in"] += 1

    # 再做局部交换：从其他Lane借可兼容文库
    if lane.total_data_gb < min_allowed:
        for donor_lane in all_lanes:
            if donor_lane is lane:
                continue
            for lib in sorted(list(donor_lane.libraries), key=lambda x: float(getattr(x, "contract_data_raw", 0.0) or 0.0), reverse=True):
                if lane.total_data_gb >= min_allowed:
                    break
                if not can_add(lib):
                    continue
                donor_trial = [x for x in donor_lane.libraries if x is not lib]
                donor_result = _validate_lane_state(strict_validator, donor_lane, donor_trial)
                if not donor_result.is_valid:
                    continue
                donor_lane.remove_library(lib)
                donor_lane.calculate_metrics()
                lane.add_library(lib)
                stats["swapped_in"] += 1
            if lane.total_data_gb >= min_allowed:
                break

    lane.calculate_metrics()
    final_valid, _, final_reason = _validate_lane_special_split_rule(lane.libraries)
    if not final_valid:
        logger.warning(f"Lane {lane.lane_id} special_splits局部交换后仍违规: {final_reason}")
    return stats


def _enforce_special_split_constraints_with_local_swap(
    solution: Any,
    strict_validator: Any,
    max_passes: int = 2,
) -> Dict[str, int]:
    """全局执行wkspecialsplits边排边检查，违规即剔除并尝试局部交换。"""
    summary = {"changed_lanes": 0, "removed_libraries": 0, "swapped_in_libraries": 0}
    lanes = solution.lane_assignments
    unassigned = solution.unassigned_libraries
    for _ in range(max_passes):
        pass_changed = False
        for lane in lanes:
            fix_stats = _auto_fix_lane_for_special_splits(
                lane=lane,
                strict_validator=strict_validator,
                all_lanes=lanes,
                unassigned_pool=unassigned,
            )
            if fix_stats["changed"] > 0:
                pass_changed = True
                summary["changed_lanes"] += 1
                summary["removed_libraries"] += fix_stats["removed"]
                summary["swapped_in_libraries"] += fix_stats["swapped_in"]
        if not pass_changed:
            break
    return summary


def _auto_fix_lane_for_customer_and_10bp(
    lane: LaneAssignment,
    strict_validator,
    unassigned_pool: List[EnhancedLibraryInfo],
) -> tuple[LaneAssignment | None, List[EnhancedLibraryInfo]]:
    """针对客户占比/10bp占比违规的Lane做矫正：可剔除+从未分配池补齐容量/占比"""
    metadata = _build_lane_metadata_for_validator(lane.lane_id, lane.metadata, libraries=lane.libraries)
    initial_result = _validate_lane_with_latest_index(
        validator=strict_validator,
        libraries=lane.libraries,
        lane_id=lane.lane_id,
        machine_type=lane.machine_type.value if lane.machine_type else "Nova X-25B",
        metadata=metadata,
    )
    error_types = {err.rule_type for err in initial_result.errors}
    fix_customer_ratio = ValidationRuleType.CUSTOMER_RATIO in error_types
    fix_10bp_ratio = ValidationRuleType.INDEX_10BP_RATIO in error_types
    if not (fix_customer_ratio or fix_10bp_ratio):
        return None, []

    working_libs: List[EnhancedLibraryInfo] = list(lane.libraries)
    removed_libs: List[EnhancedLibraryInfo] = []
    added_libs: List[EnhancedLibraryInfo] = []
    metadata_after = dict(metadata)
    machine_type = lane.machine_type.value if lane.machine_type else "Nova X-25B"
    customer_ratio_limit = 0.50
    index_10bp_ratio_min = 0.40
    min_allowed, max_allowed = _resolve_lane_capacity_limits(
        libraries=working_libs,
        machine_type=machine_type,
        lane_id=lane.lane_id,
        lane_metadata=lane.metadata,
    )

    def _total_data(libs: List[EnhancedLibraryInfo]) -> float:
        return sum(float(getattr(lib, "contract_data_raw", 0) or 0) for lib in libs)

    def _pick_from_pool(
        pool: List[EnhancedLibraryInfo],
        selector,
        need_data: float,
        current_total: float,
    ) -> List[EnhancedLibraryInfo]:
        if need_data <= 0:
            return []
        candidates = [lib for lib in pool if selector(lib)]
        candidates.sort(key=lambda x: float(getattr(x, "contract_data_raw", 0) or 0), reverse=True)
        picked: List[EnhancedLibraryInfo] = []
        acc = 0.0
        for lib in candidates:
            data = float(getattr(lib, "contract_data_raw", 0) or 0)
            if data <= 0:
                continue
            if current_total + acc + data > max_allowed:
                continue
            picked.append(lib)
            acc += data
            if acc >= need_data * 0.95:
                break
        return picked

    if fix_customer_ratio:
        customers, non_customers = _split_customer_and_non_customer(working_libs)
        data_customers = _total_data(customers)
        data_non_customers = _total_data(non_customers)
        data_total = _total_data(working_libs)
        need_non_cust = 0.0
        if data_customers > 0 or data_non_customers > 0:
            target_non_cust = max(data_non_customers, data_customers / customer_ratio_limit - data_customers)
            need_non_cust = max(0.0, target_non_cust - data_non_customers)
        picked = []
        if need_non_cust > 0:
            room = max_allowed - data_total
            if room < need_non_cust * 0.9 and customers:
                customers_sorted = sorted(customers, key=lambda x: float(getattr(x, "contract_data_raw", 0) or 0))
                freed = 0.0
                removed_cust: List[EnhancedLibraryInfo] = []
                for lib in customers_sorted:
                    if room + freed >= need_non_cust * 0.9:
                        break
                    lib_data = float(getattr(lib, "contract_data_raw", 0) or 0)
                    if data_total - freed - lib_data < min_allowed:
                        continue
                    removed_cust.append(lib)
                    freed += lib_data
                if removed_cust:
                    _remove_libraries_by_identity_in_place(working_libs, removed_cust)
                    _remove_libraries_by_identity_in_place(customers, removed_cust)
                    removed_libs.extend(removed_cust)
                    data_customers -= freed
                    data_total -= freed
                    room += freed
                    data_non_customers = _total_data(non_customers)
                    target_non_cust = max(data_non_customers, data_customers / customer_ratio_limit - data_customers)
                    need_non_cust = max(0.0, target_non_cust - data_non_customers)
            picked = _pick_from_pool(unassigned_pool, lambda x: not _is_customer_like_validator(x), need_non_cust, data_total)
        if picked:
            cand_libs = working_libs + picked
            re_res = _validate_lane_with_latest_index(
                validator=strict_validator,
                libraries=cand_libs, lane_id=lane.lane_id,
                machine_type=machine_type, metadata=metadata_after,
            )
            if re_res.is_valid:
                added_libs.extend(picked)
                _remove_libraries_by_identity_in_place(unassigned_pool, picked)
                working_libs = cand_libs
                logger.info(f"Lane {lane.lane_id} 客户占比矫正：补充{len(picked)}个非客户文库后通过校验")
            else:
                logger.debug(f"Lane {lane.lane_id} 稀释客户占比后仍未通过校验")
        current_validation = _validate_lane_with_latest_index(
            validator=strict_validator,
            libraries=working_libs,
            lane_id=lane.lane_id,
            machine_type=machine_type,
            metadata=metadata_after,
        )
        if ValidationRuleType.CUSTOMER_RATIO in {err.rule_type for err in current_validation.errors}:
            if customers:
                cand_libs = list(customers)
                need_data = max(0.0, min_allowed - _total_data(cand_libs))
                picked_cust = []
                if need_data > 0:
                    picked_cust = _pick_from_pool(unassigned_pool, lambda x: _is_customer_like_validator(x), need_data, _total_data(cand_libs))
                    cand_libs = cand_libs + picked_cust
                re_res = _validate_lane_with_latest_index(
                    validator=strict_validator,
                    libraries=cand_libs, lane_id=lane.lane_id,
                    machine_type=machine_type, metadata=metadata_after,
                )
                if re_res.is_valid:
                    removed_libs.extend(non_customers)
                    working_lib_ids = _build_library_object_id_set(working_libs)
                    added_libs.extend([lib for lib in cand_libs if id(lib) not in working_lib_ids])
                    _remove_libraries_by_identity_in_place(unassigned_pool, picked_cust)
                    working_libs = cand_libs
                    logger.info(
                        f"Lane {lane.lane_id} 客户占比矫正：转纯客户（补{len(picked_cust)}个客户文库）后通过校验，移出{len(non_customers)}个非客户文库"
                    )
                else:
                    logger.debug(f"Lane {lane.lane_id} 纯客户矫正失败")

    if fix_10bp_ratio:
        libs_10bp, libs_non_10bp = _split_10bp_and_non_10bp(working_libs, strict_validator)
        data_10bp = _total_data(libs_10bp)
        data_non_10bp = _total_data(libs_non_10bp)
        total_data = data_10bp + data_non_10bp
        if libs_10bp:
            cand_libs = list(libs_10bp)
            need_data = max(0.0, min_allowed - _total_data(cand_libs))
            picked = _pick_from_pool(unassigned_pool, lambda x: strict_validator._is_10bp_index(getattr(x, "index_seq", "") or "") or (getattr(x, "ten_bp_data", None) or 0) > 0, need_data, _total_data(cand_libs))
            cand_libs = cand_libs + picked
            re_res = _validate_lane_with_latest_index(
                validator=strict_validator,
                libraries=cand_libs, lane_id=lane.lane_id,
                machine_type=machine_type, metadata=metadata_after,
            )
            if re_res.is_valid and _total_data(cand_libs) >= min_allowed:
                libs_10bp_ids = _build_library_object_id_set(libs_10bp)
                dropped = [lib for lib in working_libs if id(lib) not in libs_10bp_ids]
                removed_libs.extend(dropped)
                added_libs.extend(picked)
                _remove_libraries_by_identity_in_place(unassigned_pool, picked)
                working_libs = cand_libs
                logger.info(
                    f"Lane {lane.lane_id} 10bp占比矫正：转纯10bp（补{len(picked)}个10bp文库）后通过校验，移出{len(dropped)}个非10bp文库"
                )
                return working_libs, removed_libs
        if libs_non_10bp:
            cand_libs = list(libs_non_10bp)
            need_data = max(0.0, min_allowed - _total_data(cand_libs))
            picked = _pick_from_pool(unassigned_pool, lambda x: not (strict_validator._is_10bp_index(getattr(x, "index_seq", "") or "") or (getattr(x, "ten_bp_data", None) or 0) > 0), need_data, _total_data(cand_libs))
            cand_libs = cand_libs + picked
            meta_non10 = dict(metadata_after)
            meta_non10["is_pure_non_10bp_lane"] = True
            re_res = _validate_lane_with_latest_index(
                validator=strict_validator,
                libraries=cand_libs, lane_id=lane.lane_id,
                machine_type=machine_type, metadata=meta_non10,
            )
            if re_res.is_valid and _total_data(cand_libs) >= min_allowed:
                libs_non_10bp_ids = _build_library_object_id_set(libs_non_10bp)
                dropped = [lib for lib in working_libs if id(lib) not in libs_non_10bp_ids]
                removed_libs.extend(dropped)
                added_libs.extend(picked)
                _remove_libraries_by_identity_in_place(unassigned_pool, picked)
                working_libs = cand_libs
                metadata_after = meta_non10
                logger.info(
                    f"Lane {lane.lane_id} 10bp占比矫正：转纯非10bp（补{len(picked)}个非10bp文库）后通过校验，移出{len(dropped)}个10bp文库"
                )
                return working_libs, removed_libs
        if data_10bp > 0 and data_non_10bp > 0:
            ratio = data_10bp / (data_10bp + data_non_10bp)
            if ratio < index_10bp_ratio_min:
                need_extra_10bp = max(0.0, index_10bp_ratio_min * data_non_10bp / (1 - index_10bp_ratio_min) - data_10bp)
                picked = _pick_from_pool(unassigned_pool, lambda x: strict_validator._is_10bp_index(getattr(x, "index_seq", "") or "") or (getattr(x, "ten_bp_data", None) or 0) > 0, need_extra_10bp, total_data)
                if picked:
                    cand_libs = working_libs + picked
                    re_res = _validate_lane_with_latest_index(
                        validator=strict_validator,
                        libraries=cand_libs, lane_id=lane.lane_id,
                        machine_type=machine_type, metadata=metadata_after,
                    )
                    if re_res.is_valid and _total_data(cand_libs) >= min_allowed:
                        added_libs.extend(picked)
                        _remove_libraries_by_identity_in_place(unassigned_pool, picked)
                        working_libs = cand_libs
                        logger.info(
                            f"Lane {lane.lane_id} 10bp占比矫正：补充{len(picked)}个10bp文库后通过校验"
                        )
                    else:
                        logger.debug(f"Lane {lane.lane_id} 补10bp后仍未通过校验")

    if not removed_libs and not added_libs:
        return None, []

    recheck_result = _validate_lane_with_latest_index(
        validator=strict_validator,
        libraries=working_libs, lane_id=lane.lane_id,
        machine_type=machine_type, metadata=metadata_after,
    )
    if not recheck_result.is_valid:
        logger.debug(f"Lane {lane.lane_id} 矫正后最终校验仍失败: {[e.message for e in recheck_result.errors]}")
        return None, []

    lane.libraries = working_libs
    lane.total_data_gb = sum(lib.get_data_amount_gb() for lib in lane.libraries)
    lane.calculate_metrics()
    lane.metadata.update(metadata_after)
    lane.metadata["auto_fix_customer_10bp"] = True
    logger.info(f"Lane {lane.lane_id} 矫正成功：移出{len(removed_libs)}个，补入{len(added_libs)}个文库")
    return lane, removed_libs


# ==================== 排机结果收集与输出 ====================


def _collect_prediction_rows(
    lanes: List[LaneAssignment],
    loutput_by_origrec: Dict[str, float],
    tag: str,
) -> pd.DataFrame:
    """收集排机结果到DataFrame

    Args:
        lanes: Lane列表
        loutput_by_origrec: origrec到实际产出的映射（用于计算误差）
        tag: 标签（用于日志）

    Returns:
        排机结果DataFrame
    """
    rows: List[Dict[str, Any]] = []
    logger.info(f"{tag} 收集排机结果，用于后续 prediction_delivery 预测")

    runid_by_lane = _build_runid_by_lane(lanes)
    for lane_sorter, lane in enumerate(lanes, start=1):
        libs = list(lane.libraries or [])
        if not libs:
            continue
        lane_loading_concentration, lane_concentration_rule = _resolve_lane_loading_concentration(
            libs,
            lane_id=lane.lane_id,
            lane_metadata=lane.metadata,
        )
        lane_loading_method, lane_sequencing_mode, lane_rule_code = _resolve_lane_output_rule_fields(
            libraries=libs,
            machine_type=lane.machine_type,
            lane_id=lane.lane_id,
            lane_metadata=lane.metadata,
        )
        lane_index_rule = _resolve_lane_index_rule_display(
            libraries=libs,
            loading_method=lane_loading_method,
        )
        logger.info(
            f"{tag} Lane {lane.lane_id} 排机浓度规则命中: {lane_concentration_rule}, "
            f"lsjnd={'' if lane_loading_concentration is None else format(lane_loading_concentration, '.3f')}"
        )
        logger.info(
            f"{tag} Lane {lane.lane_id} 输出规则命中: {lane_rule_code or 'unknown_rule'}, "
            f"lsjfs={lane_loading_method or ''}, lcxms={lane_sequencing_mode or ''}"
        )
        logger.info(
            f"{tag} Lane {lane.lane_id} 显式排机规则: {lane_index_rule or ''}"
        )
        runid = runid_by_lane.get(lane.lane_id)
        lane_balance_data = None
        lane_meta = lane.metadata if isinstance(lane.metadata, dict) else {}
        if lane_meta:
            lane_balance_data = lane_meta.get("wkbalancedata")
            if lane_balance_data is None:
                lane_balance_data = lane_meta.get("wkadd_balance_data")
            if lane_balance_data is None:
                lane_balance_data = lane_meta.get("required_balance_data_gb")
        lane_balance_data_value = None
        if lane_balance_data is not None:
            lane_balance_data_value = round(float(lane_balance_data), 3)
        # 从 lane metadata 中读取模式与轮次标记（编排器注入）
        lane_selected_seq_mode = str(lane_meta.get("selected_seq_mode", "") or "").strip()
        lane_selected_round_label = str(lane_meta.get("selected_round_label", "") or "").strip()
        second_round_label = str(
            get_scheduling_config().get_mode_1_1_config().get("second_round_label", "1.1第二轮")
        ).strip()
        is_mode_1_1_round2_lane = bool(
            lane_selected_round_label
            and second_round_label
            and lane_selected_round_label == second_round_label
        )
        is_true_package_lane = _is_package_lane_assignment(lane)
        is_true_10_plus_24_lane = _is_lane_seq_10_plus_24_lane_assignment(lane)
        if is_mode_1_1_round2_lane:
            lane_selected_seq_mode = "1.1"
            lane_sequencing_mode = "1.1"
        elif is_true_package_lane or is_true_10_plus_24_lane:
            lane_selected_seq_mode = "Lane seq"
            lane_sequencing_mode = "Lane seq"
        elif lane_rule_code == "tj_1595_standard_pe150_25b":
            lane_selected_seq_mode = "3.6T-NEW"
            lane_sequencing_mode = "3.6T-NEW"
        elif lane_rule_code.startswith("tj_1595_mode_1_1"):
            lane_selected_seq_mode = "1.1"
            lane_sequencing_mode = "1.1"
        elif _normalize_text_for_match(lane_selected_seq_mode) == _normalize_text_for_match("Lane seq"):
            lane_selected_seq_mode = ""
        if (
            not (is_true_package_lane or is_true_10_plus_24_lane)
            and _normalize_text_for_match(lane_sequencing_mode) == _normalize_text_for_match("Lane seq")
        ):
            lane_sequencing_mode = ""
        if _is_explicit_dedicated_imbalance_lane(lane) and not is_mode_1_1_round2_lane:
            lane_selected_round_label = ""
        round2_low_output_origrecs = {
            str(item).strip()
            for item in (lane_meta.get("mode_1_1_round2_low_output_origrecs") or [])
            if str(item).strip()
        }
        round2_pooling_factor = _safe_float(
            lane_meta.get("mode_1_1_round2_pooling_factor"),
            default=None,
        )
        round2_balance_ratio = None
        if lane_selected_round_label == str(
            get_scheduling_config().get_mode_1_1_config().get("second_round_label", "1.1第二轮")
        ):
            resolved_ratio = _resolve_lane_balance_ratio(lane)
            if resolved_ratio > 0:
                round2_balance_ratio = resolved_ratio

        # 排机阶段不做下单/产出预测，统一在 prediction_delivery 阶段落地。
        for lib in libs:
            contract = float(lib.contract_data_raw or 0.0)
            loutput = loutput_by_origrec.get(lib.origrec)
            last_qpcr = _get_lib_attr_float(lib, ["_last_qpcr_raw", "last_qpcr", "wklastqpcr", "wklistqpcr"])
            last_order = _get_lib_attr_float(lib, ["_last_order_data_raw", "last_order_data", "wklastorderdata"])
            last_output = _get_lib_attr_float(lib, ["_last_output_raw", "last_output", "wklastoutput"])
            last_phix = _get_lib_attr_float(lib, ["_last_phix_raw", "last_phix", "wklastphix"])
            last_outrate = _resolve_historical_outrate(
                last_outrate=_get_lib_attr_float(lib, ["_last_outrate_raw", "last_outrate", "wklastoutrate"]),
                last_output=last_output,
                last_order=last_order,
            )
            is_balance_lib = _is_ai_balance_library(lib)

            rows.append(
                {
                    "origrec": lib.origrec,
                    "origrec_key": _get_library_source_origrec_key(lib),
                    "detail_row_key": _get_library_detail_output_key(lib),
                    "runid": runid,
                    "virtualrunid": None,
                    "virtuallaneid": None,
                    "lane_id": lane.lane_id,
                    "lanesorter": lane_sorter,
                    "lsjnd": (
                        None
                        if lane_loading_concentration is None
                        else round(float(lane_loading_concentration), 3)
                    ),
                    "resolved_lsjfs": lane_loading_method or None,
                    "resolved_lcxms": lane_selected_seq_mode or lane_sequencing_mode or None,
                    "resolved_index_check_rule": lane_index_rule or None,
                    "resolved_round2_pooling_factor": (
                        round2_pooling_factor
                        if round2_pooling_factor is not None
                        and str(getattr(lib, "origrec", "") or "").strip() in round2_low_output_origrecs
                        else None
                    ),
                    "resolved_round2_balance_ratio": round2_balance_ratio,
                    "wkcontractdata": contract,
                    "wkbalancedata": (
                        lane_balance_data_value
                        if is_balance_lib
                        else None
                    ),
                    "predicted_lorderdata": contract if is_balance_lib else None,
                    "lai_output": None,
                    "ai_predicted_lorderdata": None,
                    "ai_predicted_loutput": None,
                    "add_test_rule_applied": False,
                    "add_test_rule_reason": "deferred_to_prediction_delivery",
                    "qpcr_within_15pct": None,
                    "qpcr_deviation_ratio": None,
                    "historical_based_lorderdata": None,
                    "wklastqpcr": last_qpcr,
                    "wklastorderdata": last_order,
                    "wklastoutput": last_output,
                    "wklastoutrate": last_outrate,
                    "wklastphix": last_phix,
                    "loutput": loutput,
                    "resolved_seq_mode": lane_selected_seq_mode or lane_sequencing_mode or None,
                    "resolved_round_label": lane_selected_round_label or None,
                    BALANCE_LIBRARY_MARKER_COLUMN: is_balance_lib,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["lsjnd"] = pd.to_numeric(df["lsjnd"], errors="coerce").round(3)
        df["wkbalancedata"] = pd.to_numeric(df["wkbalancedata"], errors="coerce").round(3)
        df["predicted_lorderdata"] = pd.to_numeric(df["predicted_lorderdata"], errors="coerce").round(3)
        df["lai_output"] = pd.to_numeric(df["lai_output"], errors="coerce").round(3)
    return df


def _expand_detail_output_rows(
    df_raw: pd.DataFrame,
    detail_libraries: List[EnhancedLibraryInfo],
    ai_schedulable_keys: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """按最终文库粒度展开原始明细，支持拆分子文库单独落表。"""
    expanded_source = df_raw.copy()
    if "origrec_key" not in expanded_source.columns:
        expanded_source["origrec_key"] = _build_origrec_key(expanded_source)

    raw_records: List[Dict[str, Any]] = []
    for raw_order, row in enumerate(expanded_source.to_dict(orient="records")):
        row_copy = dict(row)
        row_copy["_raw_order"] = raw_order
        raw_records.append(row_copy)

    ai_schedulable_keys = set(ai_schedulable_keys or set())
    non_ai_rows = [
        row
        for row in raw_records
        if str(row.get("origrec_key") or "").strip() not in ai_schedulable_keys
    ]

    ai_row_buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in raw_records:
        source_key = str(row.get("origrec_key") or "").strip()
        if source_key in ai_schedulable_keys:
            ai_row_buckets.setdefault(source_key, []).append(row)

    used_bucket_indices: Dict[str, int] = {}
    expanded_ai_rows: List[Dict[str, Any]] = []
    for expand_order, lib in enumerate(detail_libraries):
        source_key = _get_library_source_origrec_key(lib)
        bucket = ai_row_buckets.get(source_key)
        if not bucket:
            if _is_ai_balance_library(lib):
                template = dict(getattr(lib, "_balance_output_payload", {}) or {})
            else:
                logger.warning("明细展开缺少原始模板行，跳过文库 {}", source_key or getattr(lib, "origrec", ""))
                continue
        else:
            bucket_index = used_bucket_indices.get(source_key, 0)
            template = dict(bucket[min(bucket_index, len(bucket) - 1)])
            used_bucket_indices[source_key] = bucket_index + 1

        contract_data = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
        total_contract_data = getattr(lib, "total_contract_data", None)
        if total_contract_data in (None, ""):
            total_contract_data = getattr(lib, "wktotalcontractdata", None)
        if total_contract_data in (None, "") and _is_split_library(lib):
            total_contract_data = contract_data

        template["origrec_key"] = source_key
        template["detail_row_key"] = _get_library_detail_output_key(lib)
        if not _is_ai_balance_library(lib):
            template["wkorigrec"] = source_key or template.get("wkorigrec")
            template["origrec"] = source_key or template.get("origrec")
        template["wkcontractdata"] = contract_data
        single_index_data = getattr(lib, "single_index_data", None)
        if single_index_data not in (None, ""):
            template["wk_single_index_data"] = float(single_index_data)
        ten_bp_data = getattr(lib, "ten_bp_data", None)
        if ten_bp_data not in (None, ""):
            template["wk_10bp_data"] = float(ten_bp_data)
        if total_contract_data not in (None, ""):
            template["wktotalcontractdata"] = float(total_contract_data)

        package_lane_number = _safe_str(
            getattr(lib, "_package_lane_output_baleno", None)
            or getattr(lib, "baleno", None)
            or getattr(lib, "package_lane_number", None),
            default="",
        )
        if package_lane_number:
            template["wkbaleno"] = package_lane_number

        aidbid = _safe_str(getattr(lib, "wkaidbid", None) or getattr(lib, "aidbid", None), default="")
        if aidbid:
            template["wkaidbid"] = aidbid

        if _is_ai_balance_library(lib):
            template[BALANCE_LIBRARY_MARKER_COLUMN] = True
            template["wkissplit"] = ""
            template["wktotalcontractdata"] = pd.NA
        elif _is_split_library(lib):
            template["wkissplit"] = "yes"
        else:
            template["wkissplit"] = ""

        template["_expanded_order"] = int(getattr(lib, "fragment_index", 0) or 0)
        template["_library_expand_order"] = expand_order
        expanded_ai_rows.append(template)

    combined_rows = non_ai_rows + expanded_ai_rows
    expanded_df = pd.DataFrame(combined_rows)
    if expanded_df.empty:
        return expanded_df

    sort_columns = [
        column_name
        for column_name in ["_raw_order", "_expanded_order", "_library_expand_order"]
        if column_name in expanded_df.columns
    ]
    if sort_columns:
        expanded_df = expanded_df.sort_values(sort_columns, kind="stable")
    expanded_df = expanded_df.drop(
        columns=["_raw_order", "_expanded_order", "_library_expand_order"],
        errors="ignore",
    ).reset_index(drop=True)
    return expanded_df


def _build_detail_output(
    df_raw: pd.DataFrame,
    pred_df: pd.DataFrame,
    output_path: Path,
    ai_schedulable_keys: Optional[Set[str]] = None,
    lanes_with_split: Optional[Set[str]] = None,
    detail_libraries: Optional[List[EnhancedLibraryInfo]] = None,
    excluded_machine_reasons: Optional[Dict[str, str]] = None,
) -> None:
    """生成明细输出文件"""
    def _ensure_object_column(df: pd.DataFrame, column_name: str) -> None:
        """在写入字符串前显式转为object列，避免pandas类型告警。"""
        if column_name in df.columns:
            df[column_name] = df[column_name].astype(object)

    def _normalize_output_seq_mode(value: Any) -> Any:
        """统一输出口径中的测序模式展示值。"""
        text = str(value).strip()
        if not text:
            return value
        normalized_keyword = "".join(text.upper().split())
        if normalized_keyword == "LANESEQ":
            return "Lane seq"
        return value

    def _fill_balance_rows_from_lane_context(df: pd.DataFrame) -> None:
        """让平衡文库行继承同Lane其他文库的Lane级输出字段。"""
        if (
            df.empty
            or BALANCE_LIBRARY_MARKER_COLUMN not in df.columns
            or "llaneid" not in df.columns
        ):
            return

        balance_mask = df[BALANCE_LIBRARY_MARKER_COLUMN].fillna(False).astype(bool)
        lane_mask = df["llaneid"].map(_is_non_empty_value)
        reference_mask = (~balance_mask) & lane_mask
        if not reference_mask.any():
            return

        text_columns = [
            "wkdataunit",
            "wkuser",
            "wkdatadealbatch",
            "laneround",
            "lastlaneround",
            "task",
        ]
        for column_name in text_columns:
            if column_name not in df.columns:
                continue
            valid_reference_mask = reference_mask & df[column_name].map(_is_non_empty_value)
            if not valid_reference_mask.any():
                continue
            lane_values = (
                df.loc[valid_reference_mask, ["llaneid", column_name]]
                .drop_duplicates(subset=["llaneid"], keep="first")
                .set_index("llaneid")[column_name]
            )
            target_mask = balance_mask & lane_mask & (~df[column_name].map(_is_non_empty_value))
            if target_mask.any():
                df.loc[target_mask, column_name] = df.loc[target_mask, "llaneid"].map(lane_values)

        if "lsjnd" in df.columns:
            valid_reference_mask = reference_mask & df["lsjnd"].notna()
            if valid_reference_mask.any():
                lane_values = (
                    df.loc[valid_reference_mask, ["llaneid", "lsjnd"]]
                    .drop_duplicates(subset=["llaneid"], keep="first")
                    .set_index("llaneid")["lsjnd"]
                )
                target_mask = balance_mask & lane_mask & df["lsjnd"].isna()
                if target_mask.any():
                    df.loc[target_mask, "lsjnd"] = df.loc[target_mask, "llaneid"].map(lane_values)

    if detail_libraries is not None:
        merged = _expand_detail_output_rows(
            df_raw=df_raw,
            detail_libraries=detail_libraries,
            ai_schedulable_keys=ai_schedulable_keys,
        )
    else:
        merged = df_raw.copy()
    if "origrec_key" not in merged.columns:
        merged["origrec_key"] = _build_origrec_key(merged)
    if "detail_row_key" not in merged.columns:
        merged["detail_row_key"] = merged["origrec_key"].astype(str).str.strip()

    # 默认补齐预测相关字段，保证输出结构稳定
    merged["runid"] = pd.NA
    merged["laneid"] = pd.NA
    merged["lsjnd"] = pd.NA
    if "wkbalancedata" not in merged.columns:
        merged["wkbalancedata"] = pd.NA
    if BALANCE_LIBRARY_MARKER_COLUMN not in merged.columns:
        merged[BALANCE_LIBRARY_MARKER_COLUMN] = False
    merged["predicted_lorderdata"] = pd.NA
    merged["lai_output"] = pd.NA

    if not pred_df.empty:
        pred_for_merge = pred_df.copy()
        if "origrec_key" not in pred_for_merge.columns:
            pred_for_merge["origrec_key"] = pred_for_merge["origrec"].astype(str).str.strip()
        if "detail_row_key" not in pred_for_merge.columns:
            pred_for_merge["detail_row_key"] = pred_for_merge["origrec_key"].astype(str).str.strip()
        for missing_column in [
            "runid",
            "virtualrunid",
            "virtuallaneid",
            "lane_id",
            "lanesorter",
            "lsjnd",
            "resolved_lsjfs",
            "resolved_lcxms",
            "resolved_seq_mode",
            "resolved_round_label",
            "resolved_index_check_rule",
            "resolved_round2_pooling_factor",
            "resolved_round2_balance_ratio",
            "wklastoutrate",
            "wklastoutput",
            "wklastorderdata",
            "wklastphix",
            "wkbalancedata",
            BALANCE_LIBRARY_MARKER_COLUMN,
            "predicted_lorderdata",
            "lai_output",
        ]:
            if missing_column not in pred_for_merge.columns:
                pred_for_merge[missing_column] = pd.NA
        pred_for_merge = pred_for_merge[
            [
                "detail_row_key",
                "origrec_key",
                "runid",
                "virtualrunid",
                "virtuallaneid",
                "lane_id",
                "lanesorter",
                "lsjnd",
                "resolved_lsjfs",
                "resolved_lcxms",
                "resolved_seq_mode",
                "resolved_round_label",
                "resolved_index_check_rule",
                "resolved_round2_pooling_factor",
                "resolved_round2_balance_ratio",
                "wklastoutrate",
                "wklastoutput",
                "wklastorderdata",
                "wklastphix",
                "wkbalancedata",
                BALANCE_LIBRARY_MARKER_COLUMN,
                "predicted_lorderdata",
                "lai_output",
            ]
        ].copy()
        pred_for_merge.rename(columns={"lane_id": "laneid"}, inplace=True)

        merged = merged.drop(columns=["runid", "laneid", "lsjnd", "predicted_lorderdata", "lai_output"])
        merged = merged.merge(pred_for_merge, on=["detail_row_key", "origrec_key"], how="left", suffixes=("", "_pred"))
        if "lsjnd_pred" in merged.columns:
            merged["lsjnd"] = pd.to_numeric(
                merged["lsjnd_pred"], errors="coerce"
            ).combine_first(pd.to_numeric(merged["lsjnd"], errors="coerce"))
            merged.drop(columns=["lsjnd_pred"], inplace=True)
        if "wkbalancedata_pred" in merged.columns:
            merged["wkbalancedata"] = pd.to_numeric(
                merged["wkbalancedata_pred"], errors="coerce"
            ).combine_first(pd.to_numeric(merged["wkbalancedata"], errors="coerce"))
            merged.drop(columns=["wkbalancedata_pred"], inplace=True)
        if f"{BALANCE_LIBRARY_MARKER_COLUMN}_pred" in merged.columns:
            pred_marker = merged[f"{BALANCE_LIBRARY_MARKER_COLUMN}_pred"]
            base_marker = merged[BALANCE_LIBRARY_MARKER_COLUMN]
            merged[BALANCE_LIBRARY_MARKER_COLUMN] = pred_marker.where(pred_marker.notna(), base_marker)
            merged.drop(columns=[f"{BALANCE_LIBRARY_MARKER_COLUMN}_pred"], inplace=True)

    # 仅对已成Lane的数据填充测序模式，优先使用统一规则结果，未成Lane记录保持原值不改
    if "lcxms" not in merged.columns:
        merged["lcxms"] = pd.NA
    _ensure_object_column(merged, "lcxms")
    merged["lcxms"] = ""
    lane_assigned_mask = (
        merged["laneid"].notna()
        & ~merged["laneid"].astype(str).str.strip().isin({"", "nan", "None", "NONE", "null", "NULL"})
    )
    resolved_lcxms_mask = (
        "resolved_lcxms" in merged.columns
        and merged["resolved_lcxms"].notna()
        & ~merged["resolved_lcxms"].astype(str).str.strip().isin({"", "nan", "None", "NONE", "null", "NULL"})
    )
    if isinstance(resolved_lcxms_mask, pd.Series):
        merged.loc[lane_assigned_mask & resolved_lcxms_mask, "lcxms"] = merged.loc[
            lane_assigned_mask & resolved_lcxms_mask, "resolved_lcxms"
        ]
    # lcxms 二级回填：resolved_lcxms 缺失时，尝试从 resolved_seq_mode（lane metadata 注入）读取
    missing_lcxms_mask = (
        merged["lcxms"].isna()
        | merged["lcxms"].astype(str).str.strip().isin({"", "nan", "None", "NONE", "null", "NULL"})
    )
    resolved_seq_mode_mask = (
        "resolved_seq_mode" in merged.columns
        and merged["resolved_seq_mode"].notna()
        & ~merged["resolved_seq_mode"].astype(str).str.strip().isin({"", "nan", "None", "NONE", "null", "NULL"})
    )
    if isinstance(resolved_seq_mode_mask, pd.Series):
        fill_from_mode = lane_assigned_mask & missing_lcxms_mask & resolved_seq_mode_mask
        merged.loc[fill_from_mode, "lcxms"] = merged.loc[fill_from_mode, "resolved_seq_mode"]
    # 最终兜底：仅当 resolved_lcxms 和 resolved_seq_mode 都无法确定时，才回退到 3.6T-NEW
    still_missing_mask = (
        merged["lcxms"].isna()
        | merged["lcxms"].astype(str).str.strip().isin({"", "nan", "None", "NONE", "null", "NULL"})
    )
    merged.loc[lane_assigned_mask & still_missing_mask, "lcxms"] = "3.6T-NEW"
    merged["lcxms"] = merged["lcxms"].map(_normalize_output_seq_mode)

    # 1.1排机轮数：从 resolved_round_label（编排器注入）回填到 laneround 输出列
    if "laneround" not in merged.columns:
        merged["laneround"] = pd.NA
    _ensure_object_column(merged, "laneround")
    merged["laneround"] = ""
    if "resolved_round_label" in merged.columns:
        resolved_round_mask = (
            merged["resolved_round_label"].notna()
            & ~merged["resolved_round_label"].astype(str).str.strip().isin(
                {"", "nan", "None", "NONE", "null", "NULL"}
            )
        )
        resolved_round_text = merged["resolved_round_label"].astype(str).str.strip()
        valid_round_label_mask = resolved_round_text.isin(
            {
                str(get_scheduling_config().get_mode_1_1_config().get("first_round_label", "1.1第一轮")),
                str(get_scheduling_config().get_mode_1_1_config().get("second_round_label", "1.1第二轮")),
            }
        )
        merged.loc[lane_assigned_mask & resolved_round_mask & valid_round_label_mask, "laneround"] = merged.loc[
            lane_assigned_mask & resolved_round_mask & valid_round_label_mask,
            "resolved_round_label",
        ]
    # 兜底：若已明确成Lane且测序模式属于1.1，但上游遗漏了轮次标签，
    # 则按业务口径回填为1.1第一轮。第二轮会显式写 selected_round_label，不会走到这里。
    missing_round_mask = (
        merged["laneround"].isna()
        | merged["laneround"].astype(str).str.strip().isin({"", "nan", "None", "NONE", "null", "NULL"})
    )
    seq_mode_source = (
        merged["resolved_seq_mode"]
        if "resolved_seq_mode" in merged.columns
        else merged["lcxms"]
    )
    normalized_seq_mode = seq_mode_source.map(_normalize_mode_1_1_alias)
    first_round_label = str(
        get_scheduling_config().get_mode_1_1_config().get("first_round_label", "1.1第一轮")
    )
    inferred_first_round_mask = lane_assigned_mask & missing_round_mask & normalized_seq_mode.eq("1.1")
    merged.loc[inferred_first_round_mask, "laneround"] = first_round_label

    # 成Lane(有laneid)的文库，标记lanecreatetype为AI
    if "lanecreatetype" not in merged.columns:
        merged["lanecreatetype"] = pd.NA
    _ensure_object_column(merged, "lanecreatetype")
    merged["lanecreatetype"] = ""
    merged.loc[lane_assigned_mask, "lanecreatetype"] = "AI"

    # AI排机次数：默认0，仅对真正参与本轮排机的AI可排文库统一+1（无论是否成lane）
    if "aiarrangenumber" not in merged.columns:
        merged["aiarrangenumber"] = 0
    ai_arrange_series = pd.to_numeric(merged["aiarrangenumber"], errors="coerce").fillna(0).astype(int)
    ai_schedulable_keys = ai_schedulable_keys or set()
    if ai_schedulable_keys:
        ai_schedulable_mask = merged["origrec_key"].astype(str).isin(ai_schedulable_keys)
        ai_arrange_series.loc[ai_schedulable_mask] = ai_arrange_series.loc[ai_schedulable_mask] + 1
    merged["aiarrangenumber"] = ai_arrange_series

    excluded_machine_reasons = {
        str(key).strip(): str(value).strip()
        for key, value in (excluded_machine_reasons or {}).items()
        if str(key).strip() and str(value).strip()
    }
    if excluded_machine_reasons:
        excluded_machine_mask = merged["origrec_key"].astype(str).isin(excluded_machine_reasons)
        if excluded_machine_mask.any():
            for column_name in ["aiavailable", "unaireason"]:
                if column_name not in merged.columns:
                    merged[column_name] = pd.NA
                _ensure_object_column(merged, column_name)
            merged.loc[excluded_machine_mask, "aiavailable"] = "no"
            merged.loc[excluded_machine_mask, "unaireason"] = merged.loc[
                excluded_machine_mask, "origrec_key"
            ].astype(str).map(excluded_machine_reasons)

    # 将新生成的runid/laneid覆盖写回原始字段lrunid/llaneid，并移除runid/laneid输出列
    if "lrunid" not in merged.columns:
        merged["lrunid"] = pd.NA
    if "llaneid" not in merged.columns:
        merged["llaneid"] = pd.NA
    if "virtualrunid" not in merged.columns:
        merged["virtualrunid"] = pd.NA
    if "virtuallaneid" not in merged.columns:
        merged["virtuallaneid"] = pd.NA
    _ensure_object_column(merged, "lrunid")
    _ensure_object_column(merged, "llaneid")
    _ensure_object_column(merged, "virtualrunid")
    _ensure_object_column(merged, "virtuallaneid")
    # 输出文件中的排机字段只保留本轮最终成功Lane，先清空旧值，再按pred_df回填。
    merged["lrunid"] = ""
    merged["llaneid"] = ""
    merged["virtualrunid"] = ""
    merged["virtuallaneid"] = ""
    merged.loc[lane_assigned_mask, "lrunid"] = merged.loc[lane_assigned_mask, "runid"]
    merged.loc[lane_assigned_mask, "llaneid"] = merged.loc[lane_assigned_mask, "laneid"]
    for virtual_column in ["virtualrunid", "virtuallaneid"]:
        pred_column = f"{virtual_column}_pred"
        if pred_column in merged.columns:
            merged = merged.drop(columns=[pred_column])

    # 输入若来自历史排机结果，未成Lane行必须清空旧的Lane级字段，避免下游误判为仍在旧虚拟Lane中。
    unassigned_output_mask = ~lane_assigned_mask
    stale_lane_context_columns = [
        "virtuallaneid",
        "virtualrunid",
        "laneprojecttype",
        "lanepriority",
        "laneproductline",
        "lanecontractdata",
        "laneorderdata",
        "lanephix",
        "laneadaptortype",
        "lanespecialsplits",
        "laneseqscheme",
        "laneseqnotes",
        "laneindexnumber",
        "lanewknumber",
        "lanecreatetype",
        "lanebaleno",
        "lanebagfcno",
        "runid_laneid_raw",
        "laneupdatenumber",
        "lane_pooling_status",
        "lane_un_pooling_reason",
        "lane_calculate_error",
    ]
    for column_name in stale_lane_context_columns:
        if column_name in merged.columns:
            _ensure_object_column(merged, column_name)
            merged.loc[unassigned_output_mask, column_name] = ""

    # lanesorter用于下游按Lane稳定排序：同一Lane内所有行使用同一个顺序号，未成Lane留空。
    if "lanesorter" not in merged.columns:
        merged["lanesorter"] = pd.NA
    merged.loc[~lane_assigned_mask, "lanesorter"] = pd.NA
    merged["lanesorter"] = pd.to_numeric(merged["lanesorter"], errors="coerce").astype("Int64")

    # 修正Lane级合同量字段：以最终输出明细中同一Lane的wkcontractdata合计为准。
    # 旧输入或中间合并链路可能遗留1000G等历史值，不能作为本轮Lane实际合同量输出。
    if "lanecontractdata" not in merged.columns:
        merged["lanecontractdata"] = pd.NA
    if "wkcontractdata" not in merged.columns:
        merged["wkcontractdata"] = 0.0
    lane_contract_sum = pd.to_numeric(merged["wkcontractdata"], errors="coerce").fillna(0.0)
    merged["_lane_contract_sum_for_output"] = lane_contract_sum
    lane_contract_by_key = (
        merged.loc[lane_assigned_mask]
        .groupby(["lrunid", "llaneid"], dropna=False)["_lane_contract_sum_for_output"]
        .sum()
    )
    assigned_lane_keys = pd.MultiIndex.from_frame(
        merged.loc[lane_assigned_mask, ["lrunid", "llaneid"]]
    )
    lane_contract_values = pd.Series(assigned_lane_keys.map(lane_contract_by_key), index=merged.index[lane_assigned_mask])
    merged.loc[lane_assigned_mask, "lanecontractdata"] = pd.to_numeric(lane_contract_values, errors="coerce").round(3)
    merged.loc[unassigned_output_mask, "lanecontractdata"] = ""
    merged.drop(columns=["_lane_contract_sum_for_output"], inplace=True)

    # lsjfs优先读取统一规则表中的loading_method，未成Lane记录保持原值
    if "lsjfs" not in merged.columns:
        merged["lsjfs"] = pd.NA
    _ensure_object_column(merged, "lsjfs")
    merged["lsjfs"] = ""
    resolved_lsjfs_mask = (
        "resolved_lsjfs" in merged.columns
        and merged["resolved_lsjfs"].notna()
        & ~merged["resolved_lsjfs"].astype(str).str.strip().isin({"", "nan", "None", "NONE", "null", "NULL"})
    )
    if isinstance(resolved_lsjfs_mask, pd.Series):
        merged.loc[lane_assigned_mask & resolved_lsjfs_mask, "lsjfs"] = merged.loc[
            lane_assigned_mask & resolved_lsjfs_mask, "resolved_lsjfs"
        ]
    missing_lsjfs_mask = (
        merged["lsjfs"].isna()
        | merged["lsjfs"].astype(str).str.strip().isin({"", "nan", "None", "NONE", "null", "NULL"})
    )
    # 当前V6仅支持25B与NovaSeq X Plus，两类机型业务上统一按25B上机方式输出。
    merged.loc[lane_assigned_mask & missing_lsjfs_mask, "lsjfs"] = "25B"

    # 显式输出排机规则，避免下游再按wkindexseq是否含分号反推P7/P7P5。
    for column_name in ["排机规则", "index查重规则"]:
        if column_name not in merged.columns:
            merged[column_name] = pd.NA
        _ensure_object_column(merged, column_name)
        merged[column_name] = ""
    resolved_index_rule_mask = (
        "resolved_index_check_rule" in merged.columns
        and merged["resolved_index_check_rule"].notna()
        & ~merged["resolved_index_check_rule"].astype(str).str.strip().isin(
            {"", "nan", "None", "NONE", "null", "NULL"}
        )
    )
    if isinstance(resolved_index_rule_mask, pd.Series):
        for column_name in ["排机规则", "index查重规则"]:
            merged.loc[lane_assigned_mask & resolved_index_rule_mask, column_name] = merged.loc[
                lane_assigned_mask & resolved_index_rule_mask,
                "resolved_index_check_rule",
            ]

    # 某些历史输入或合并链路可能带入重名列；DataFrame 在 to_dict(orient="records")
    # 时会告警且后值覆盖前值。这里保留首个同名列，确保后续57规则复核口径稳定。
    if merged.columns.duplicated().any():
        duplicate_columns = merged.columns[merged.columns.duplicated()].tolist()
        logger.warning(
            "明细输出检测到重复列，已按首列保留去重: {}",
            duplicate_columns,
        )
        merged = merged.loc[:, ~merged.columns.duplicated()].copy()

    # lane_show规则：包FC+包Lane均有值 或 所在lane包含拆分文库
    if "lane_show" not in merged.columns:
        merged["lane_show"] = "no"
    has_bagfc = merged["wkbagfcno"].map(_is_non_empty_value) if "wkbagfcno" in merged.columns else pd.Series(False, index=merged.index)
    has_baleno = merged["wkbaleno"].map(_is_non_empty_value) if "wkbaleno" in merged.columns else pd.Series(False, index=merged.index)
    package_lane_show_mask = has_bagfc & has_baleno
    lanes_with_split = lanes_with_split or set()
    if lanes_with_split:
        split_lane_show_mask = merged["laneid"].astype(str).isin(lanes_with_split)
    else:
        split_lane_show_mask = pd.Series(False, index=merged.index)
    merged["lane_show"] = np.where(package_lane_show_mask | split_lane_show_mask, "yes", "no")

    # `wkuser` 字段保持输入原值，不随排机结果清空/覆盖
    if "wkuser" not in merged.columns:
        merged["wkuser"] = pd.NA
    _ensure_object_column(merged, "wkuser")

    _fill_balance_rows_from_lane_context(merged)

    # 输出字段改名：预测结果按业务字段名输出
    # 注意：这里使用预测值覆盖输出中的 lorderdata / lai_output
    merged["lorderdata"] = pd.to_numeric(merged.get("predicted_lorderdata"), errors="coerce")
    merged["lai_output"] = pd.to_numeric(merged.get("lai_output"), errors="coerce")

    # 显式排除中间列runid/laneid及预测中间列，避免重复
    merged = merged.drop(columns=["runid", "laneid"], errors="ignore")
    merged = merged.drop(columns=["predicted_lorderdata"], errors="ignore")
    merged = merged.drop(
        columns=["resolved_lsjfs", "resolved_lcxms", "resolved_index_check_rule",
                 "resolved_seq_mode", "resolved_round_label", "detail_row_key"],
        errors="ignore",
    )
    if "origrec_key" not in df_raw.columns:
        merged = merged.drop(columns=["origrec_key"], errors="ignore")
    if "origrec" not in df_raw.columns:
        merged = merged.drop(columns=["origrec"], errors="ignore")
    merged = _drop_redundant_output_columns(merged)

    if output_path.exists():
        logger.info(f"明细文件已存在，将覆盖: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info(f"明细输出完成: {output_path}")


# ==================== 数据加载 ====================


def _read_csv_with_encoding_fallback(csv_path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
    """读取CSV并对常见中文编码做兜底。"""
    if "encoding" in kwargs and kwargs["encoding"]:
        return pd.read_csv(csv_path, **kwargs)

    fallback_encodings = ("utf-8", "utf-8-sig", "gb18030", "gbk")
    last_error: Optional[UnicodeDecodeError] = None
    for encoding in fallback_encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding, **kwargs)
            if encoding != "utf-8":
                logger.warning(f"CSV编码非utf-8，已使用 {encoding} 读取文件: {csv_path}")
            return df
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(f"CSV读取失败，无法识别文件编码: {csv_path}，最后错误: {last_error}")


def load_standardized_csv(data_file: str, limit: int | None = None) -> List[EnhancedLibraryInfo]:
    """从标准化CSV文件加载文库数据（训练数据格式）
    
    使用 EnhancedLibraryInfo.from_csv_row() 完成字段映射，
    支持 wk 前缀和非 wk 前缀两种列名格式。
    """
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    logger.info(f"从标准化CSV文件加载数据: {data_path}")
    df = _read_csv_with_encoding_fallback(data_path, nrows=limit)
    logger.info(f"读取 {len(df)} 行数据")
    
    libraries: List[EnhancedLibraryInfo] = []
    
    for idx, row in df.iterrows():
        row_dict = {k: (v if not pd.isna(v) else None) for k, v in row.to_dict().items()}
        try:
            lib = EnhancedLibraryInfo.from_csv_row(row_dict)
            # 设置机型
            lib.machine_type = _resolve_machine_type_enum_simple(lib.eq_type)
            # 保留拆分相关原始字段仅用于明细透传；排机判断不得依赖 wkissplit。
            raw_wkissplit = _safe_str(row_dict.get("wkissplit"), default="")
            lib.wkissplit = raw_wkissplit
            raw_total_contract = row_dict.get("wktotalcontractdata")
            if raw_total_contract not in (None, ""):
                total_contract_value = _safe_float(raw_total_contract, default=None)
                if total_contract_value is not None:
                    lib.wktotalcontractdata = total_contract_value
                    lib.total_contract_data = total_contract_value
            raw_split_status = _safe_str(row_dict.get("split_status"), default="")
            if raw_split_status:
                lib.split_status = raw_split_status
            # 保存origrec_key与AI可排标识，供主流程与明细规则使用
            lib._origrec_key = _safe_str(
                row_dict.get("wkorigrec")
                or row_dict.get("origrec")
                or row_dict.get("lane_unique_id")
                or row_dict.get("lane_unique")
                or row_dict.get("llaneid")
                or f"LIB_{idx}"
            )
            raw_aidbid = _safe_str(
                row_dict.get("wkaidbid") or row_dict.get("aidbid"),
                default="",
            )
            if raw_aidbid:
                lib.wkaidbid = raw_aidbid
                lib.aidbid = raw_aidbid
            lib._source_origrec_key = lib._origrec_key
            lib._detail_output_key = raw_aidbid or lib._origrec_key
            lib._aiavailable_raw = _resolve_aiavailable_raw(row_dict)
            # 保存V6需要但EnhancedLibraryInfo不支持的额外字段
            jkhj_val = row_dict.get("wkjkhj") or row_dict.get("jkhj")
            lib._jkhj_raw = str(jkhj_val) if jkhj_val else "诺禾自动"
            lib._last_qpcr_raw = _safe_float(
                row_dict.get("wklastqpcr", row_dict.get("wklistqpcr")),
                default=None,
            )
            lib._last_order_data_raw = _safe_float(
                row_dict.get("wklastorderdata", row_dict.get("wklastlorderdata")),
                default=None,
            )
            lib._last_output_raw = _safe_float(row_dict.get("wklastoutput"), default=None)
            lib._last_outrate_raw = _safe_float(row_dict.get("wklastoutrate"), default=None)
            lib._last_phix_raw = _safe_float(row_dict.get("wklastphix"), default=None)
            lib._delete_date_raw = row_dict.get("delete_date", row_dict.get("扣减时间"))
            lib._wkdept_raw = _safe_str(row_dict.get("wkdept"), default="")
            raw_task_group_name = _safe_str(
                row_dict.get("wktaskgroupname") or row_dict.get("TASK_GROUP_NAME"),
                default="",
            )
            lib.task_group_name = raw_task_group_name
            lib._task_group_name_raw = raw_task_group_name
            # 保存测序模式相关原始字段，供拆分规则识别1.1/3.6T-NEW模式使用
            lib._lane_sj_mode_raw = _safe_str(row_dict.get("lsjfs"), default="")
            lib._current_seq_mode_raw = _normalize_mode_1_1_alias(row_dict.get("lcxms"))
            lib._last_cxms_raw = _normalize_mode_1_1_alias(
                row_dict.get("llastcxms") or row_dict.get("lastcxms")
            )
            if not getattr(lib, "last_cxms", None) and lib._last_cxms_raw:
                lib.last_cxms = lib._last_cxms_raw
            # 1.1模式轮次字段：上轮测序轮数（lims推送），供第二轮候选识别使用
            lib._last_lane_round_raw = _safe_str(
                row_dict.get("llastlaneround") or row_dict.get("lastlaneround"),
                default="",
            )
            if lib._last_lane_round_raw and getattr(lib, "last_laneid", None):
                if lib._last_order_data_raw is None:
                    lib._last_order_data_raw = _safe_float(row_dict.get("lorderdata"), default=None)
                if lib._last_output_raw is None:
                    lib._last_output_raw = _safe_float(row_dict.get("loutput"), default=None)
                if lib._last_outrate_raw is None:
                    derived_last_outrate = _resolve_historical_outrate(
                        last_outrate=_safe_float(row_dict.get("wkoutputrate"), default=None),
                        last_output=lib._last_output_raw,
                        last_order=lib._last_order_data_raw,
                    )
                    if derived_last_outrate is not None:
                        lib._last_outrate_raw = round(float(derived_last_outrate) * 100.0, 6)
            libraries.append(lib)
        except Exception as e:
            logger.warning(f"行 {idx} 创建文库对象失败: {e}")
    
    _normalize_existing_split_fragments(libraries)
    logger.info(f"成功创建 {len(libraries)} 个文库对象")
    return libraries


def load_test_libraries(data_file: str, limit: int | None = None) -> List[EnhancedLibraryInfo]:
    """加载测试文库（兼容多种数据格式）"""
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    try:
        return load_standardized_csv(data_file, limit=limit)
    except Exception as e:
        logger.warning(f"标准化CSV加载失败: {e}，尝试通用加载")
        return load_libraries_from_csv(data_path, limit=limit, enable_remark_recognition=False)


# ==================== 排机方案分析 ====================


def analyze_solution(solution: Any) -> Dict[str, Any]:
    """分析排机方案的质量指标
    
    Args:
        solution: 排机解决方案
    """
    lanes: List[LaneAssignment] = solution.lane_assignments
    
    # 基础统计
    total_lanes = len(lanes)
    total_libraries = sum(len(lane.libraries) for lane in lanes)
    total_contract_data = sum(lane.total_data_gb for lane in lanes)
    
    stats: Dict[str, Any] = {
        "total_lanes": total_lanes,
        "total_libraries": total_libraries,
        "total_contract_data_gb": total_contract_data,
        "avg_libraries_per_lane": total_libraries / total_lanes if total_lanes > 0 else 0,
        "avg_contract_data_per_lane": total_contract_data / total_lanes if total_lanes > 0 else 0,
    }
    
    # 统计Lane利用率
    lane_utilizations: List[float] = []
    for lane in lanes:
        total_data = lane.total_data_gb
        lane_capacity = lane.lane_capacity_gb
        if lane_capacity <= 0:
            if lane.machine_type == MachineType.NOVA_X_25B:
                lane_capacity = 975.0
            elif lane.machine_type == MachineType.NOVA_X_10B:
                lane_capacity = 380.0
            else:
                lane_capacity = 975.0
        
        utilization = total_data / lane_capacity if lane_capacity > 0 else 0.0
        lane_utilizations.append(utilization)
        
        if utilization > 1.5:
            logger.warning(
                f"Lane {lane.lane_id} 利用率异常: {utilization:.2%} "
                f"(数据量={total_data:.2f}GB, 容量={lane_capacity:.2f}GB)"
            )
    
    if lane_utilizations:
        stats["avg_utilization"] = sum(lane_utilizations) / len(lane_utilizations)
        stats["min_utilization"] = min(lane_utilizations)
        stats["max_utilization"] = max(lane_utilizations)
        
    return stats


# ==================== 排机主流程 ====================


def test_with_model(
    libraries: List[EnhancedLibraryInfo],
    existing_lanes: Optional[List[LaneAssignment]] = None,
    enable_expensive_rescue: bool = True,
    enable_peak_window_mixed_lanes: bool = True,
    enable_post_fill_optimization: Optional[bool] = None,
    enable_57_rescue: bool = False,
    enable_rule_bucket_prescheduling: bool = True,
) -> Tuple[Dict[str, Any], Any]:
    """排机流程

    流程：
    1. 优先尝试抽取纯10bp专Lane
    2. GreedyLaneScheduler 纯规则排机（不使用模型）
    3. 验证Lane合规性，矫正客户/10bp违规
    4. 尝试增加Lane数量、跨Lane交换再平衡
    Args:
        libraries: 待排机文库列表
        existing_lanes: 已存在的Lane（如包Lane），将被合并到最终结果中
        enable_expensive_rescue: 是否启用 EX/RB 与分层重组等高成本救援。
            默认保持原有行为；1.1 首轮会显式关闭，剩余文库直接回流 3.6T-NEW。
        enable_peak_window_mixed_lanes: 是否启用 Peak Size 窗口混排预构Lane。
            默认保持原有行为；1.1 首轮会显式关闭，避免在大批小库上做高成本预搜索。
        enable_post_fill_optimization: 是否启用最后填充与挪移优化。
            None 表示沿用 enable_peak_window_mixed_lanes 的历史行为。
        enable_57_rescue: 是否启用57规则二次改排救援，默认关闭。

    Returns:
        (排机统计, 排机方案)
    """
    logger.info("\n" + "=" * 80)
    logger.info("排机流程：纯规则排机")
    logger.info("=" * 80)
    
    # Lane容量配置（调度阶段）
    config = GreedyLaneConfig(
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
    
    scheduler = GreedyLaneScheduler(config)

    # 排机阶段不使用模型
    if scheduler.pooling_optimizer:
        scheduler.pooling_optimizer.enabled = False
        logger.info("排机阶段：Pooling优化器已禁用（V6流程不需要）")
    
    from arrange_library.core.constraints.lane_validator import LaneValidator
    strict_validator = LaneValidator(strict_mode=True)
    logger.info("严格校验容量区间改为按统一配置表动态解析")

    disabled_plan = StrategyExecutionPlan()
    disabled_plan.enable_dedicated_imbalance_lane = True
    disabled_plan.enable_non_10bp_dedicated_lane = False
    disabled_plan.enable_backbone_reservation = False
    scheduler._strategy_plan = disabled_plan

    # ===== 预拆分前置到所有预构建Lane之前 =====
    presplit_libraries, presplit_records = _split_libraries_for_3_6t_allowed_sources(
        scheduler.library_splitter,
        libraries,
    )
    presplit_family_context = scheduler._build_presplit_family_context(presplit_libraries)
    if presplit_records:
        logger.info(
            "主流程前置拆分完成: 原始文库{}个，触发拆分{}个，拆分后文库{}个",
            len(libraries),
            len(presplit_records),
            len(presplit_libraries),
        )
    else:
        logger.info("主流程前置拆分完成: 无需拆分")

    # ===== 混样排（Peak Size窗口内10bp+非10bp混排） =====
    # 不再先抽10bp专Lane：保留全部10bp文库参与混排，确保10bp>=40%
    dedicated_10bp_lanes: List[LaneAssignment] = []

    if enable_peak_window_mixed_lanes:
        mixed_lanes, remaining_libraries = _extract_mixed_lanes_by_peak_window(
            libraries=presplit_libraries,
            validator=strict_validator,
            machine_type=MachineType.NOVA_X_25B,
            index_conflict_attempts_per_lane=DEFAULT_INDEX_CONFLICT_ATTEMPTS,
            other_failure_attempts_per_lane=DEFAULT_OTHER_FAILURE_ATTEMPTS,
        )
    else:
        logger.info("Peak窗口混排预构Lane已按调用方要求关闭")
        mixed_lanes = []
        remaining_libraries = list(presplit_libraries)

    # 执行排机（剩余文库进入混样排机）
    post_fill_optimization_enabled = (
        enable_peak_window_mixed_lanes
        if enable_post_fill_optimization is None
        else bool(enable_post_fill_optimization)
    )
    if remaining_libraries:
        if enable_rule_bucket_prescheduling:
            solution = _schedule_with_rule_bucket_prescheduling(
                scheduler=scheduler,
                libraries=remaining_libraries,
                post_fill_optimization_enabled=post_fill_optimization_enabled,
            )
        else:
            solution = scheduler.schedule(
                remaining_libraries,
                keep_failed_lanes=True,
                libraries_already_split=True,
                perform_presplit_family_rollback=False,
                enable_post_fill_optimization=post_fill_optimization_enabled,
            )
    else:
        from types import SimpleNamespace

        solution = SimpleNamespace(lane_assignments=[], unassigned_libraries=[])
    special_split_stats = _enforce_special_split_constraints_with_local_swap(
        solution=solution,
        strict_validator=strict_validator,
        max_passes=2,
    )
    if special_split_stats["changed_lanes"] > 0:
        logger.info(
            "wkspecialsplits边排边检查完成: 调整Lane={}，剔除文库={}，局部交换补入={}".format(
                special_split_stats["changed_lanes"],
                special_split_stats["removed_libraries"],
                special_split_stats["swapped_in_libraries"],
            )
        )
    else:
        logger.info("wkspecialsplits边排边检查完成: 无需调整")

    # 严格验证
    passed_lanes: List[LaneAssignment] = []
    failed_lanes: List[LaneAssignment] = []
    for lane in solution.lane_assignments:
        metadata = _build_lane_metadata_for_validator(lane.lane_id, lane.metadata, libraries=lane.libraries)
        result = _validate_lane_with_latest_index(
            validator=strict_validator,
            libraries=lane.libraries,
            lane_id=lane.lane_id,
            machine_type=lane.machine_type.value if lane.machine_type else "Nova X-25B",
            metadata=metadata,
        )
        if result.is_valid:
            passed_lanes.append(lane)
        else:
            fixed_lane, removed_libs = _auto_fix_lane_for_customer_and_10bp(
                lane, strict_validator, solution.unassigned_libraries
            )
            if fixed_lane:
                passed_lanes.append(fixed_lane)
                solution.unassigned_libraries.extend(removed_libs)
                logger.info(
                    f"Lane {lane.lane_id} 客户/10bp占比矫正成功，移出{len(removed_libs)}个文库后通过严格校验"
                )
            else:
                failed_lanes.append(lane)
                error_types = [e.rule_type.value for e in result.errors]
                warning_types = [w.rule_type.value for w in result.warnings]
                logger.warning(
                    f"Lane {lane.lane_id} 验证失败 - 错误: {error_types}, 警告: {warning_types}"
                )

    solution.lane_assignments = passed_lanes
    logger.info(f"验证完成：{len(passed_lanes)}条Lane通过验证")
    if enable_57_rescue:
        rescue_stats = _rescue_failed_lanes_by_57_rules(
            failed_lanes=failed_lanes,
            solution=solution,
            validator=strict_validator,
            machine_type=MachineType.NOVA_X_25B,
        )
        if rescue_stats["failed_lanes"] > 0:
            logger.info(
                "57规则二次改排完成: 失败Lane={}，回收文库={}，新增成功Lane={}，剩余未分配={}".format(
                    rescue_stats["failed_lanes"],
                    rescue_stats["recovered_libraries"],
                    rescue_stats["rescued_lanes"],
                    rescue_stats["remaining_unassigned"],
                )
            )
    else:
        if failed_lanes:
            for lane in failed_lanes:
                solution.unassigned_libraries.extend(list(getattr(lane, "libraries", []) or []))
        logger.info("57规则二次改排救援已关闭，失败Lane文库回收到未分配池")

    if not enable_expensive_rescue:
        logger.info("高成本救援(EX/RB)已按调用方要求关闭，保留当前成Lane结果")
    elif not _should_skip_expensive_rescue_stage(
        solution,
        stage_name="高成本救援(EX/RB)",
    ):
        # 尝试增加Lane数量
        extra_lanes = _try_increase_lane_count(
            solution,
            strict_validator,
            max_new_lanes=DEFAULT_EX_RESCUE_MAX_NEW_LANES,
            index_conflict_attempts_per_lane=DEFAULT_INDEX_CONFLICT_ATTEMPTS,
            other_failure_attempts_per_lane=DEFAULT_OTHER_FAILURE_ATTEMPTS,
        )
        if extra_lanes:
            logger.info(f"Lane数量提升新增Lane数: {extra_lanes}")
        else:
            logger.info("Lane数量提升未新增Lane")
        extra_stage_split_stats = _enforce_special_split_constraints_with_local_swap(
            solution=solution,
            strict_validator=strict_validator,
            max_passes=1,
        )
        if extra_stage_split_stats["changed_lanes"] > 0:
            logger.info(
                "Lane提升后wkspecialsplits复检: 调整Lane={}，剔除文库={}，局部交换补入={}".format(
                    extra_stage_split_stats["changed_lanes"],
                    extra_stage_split_stats["removed_libraries"],
                    extra_stage_split_stats["swapped_in_libraries"],
                )
            )

        # 跨Lane多文库交换再平衡
        rebalance_result = try_multi_lib_swap_rebalance(
            solution,
            strict_validator,
            max_new_lanes=DEFAULT_RB_RESCUE_MAX_NEW_LANES,
            max_donations=80,
            index_conflict_max_trials=DEFAULT_INDEX_CONFLICT_ATTEMPTS,
            other_failure_max_trials=DEFAULT_OTHER_FAILURE_ATTEMPTS,
            max_per_lane=8,
        )
        if rebalance_result["new_lanes"] > 0:
            logger.info(
                "跨Lane多文库交换完成：新增{new_lanes}条，剩余未分配{remaining_unassigned}个".format(
                    **rebalance_result
                )
            )
        else:
            logger.info(
                "跨Lane多文库交换未新增Lane，剩余未分配{remaining_unassigned}个".format(
                    **rebalance_result
                )
            )
        rebalance_stage_split_stats = _enforce_special_split_constraints_with_local_swap(
            solution=solution,
            strict_validator=strict_validator,
            max_passes=1,
        )
        if rebalance_stage_split_stats["changed_lanes"] > 0:
            logger.info(
                "再平衡后wkspecialsplits复检: 调整Lane={}，剔除文库={}，局部交换补入={}".format(
                    rebalance_stage_split_stats["changed_lanes"],
                    rebalance_stage_split_stats["removed_libraries"],
                    rebalance_stage_split_stats["swapped_in_libraries"],
                )
            )
    passed_lanes = solution.lane_assignments

    # ===== 合并包Lane/10bp专Lane/混排Lane到最终结果 =====
    preset_lanes: List[LaneAssignment] = []
    if existing_lanes:
        preset_lanes.extend(existing_lanes)
    if dedicated_10bp_lanes:
        preset_lanes.extend(dedicated_10bp_lanes)
    if mixed_lanes:
        preset_lanes.extend(mixed_lanes)
    if preset_lanes:
        # 预构建Lane（尤其是碱基不均衡专Lane）在进入严格终态校验前，
        # 需要先把平衡文库真实挂入lane；否则像 G26 这类 800G+200G 的
        # 专Lane会在这里只按800G裸lane被提前淘汰。
        from types import SimpleNamespace

        preset_solution = SimpleNamespace(
            lane_assignments=list(preset_lanes),
            unassigned_libraries=list(getattr(solution, "unassigned_libraries", []) or []),
        )
        preset_balance_stats = _materialize_balance_libraries_for_solution(preset_solution)
        preset_lanes = list(getattr(preset_solution, "lane_assignments", []) or [])
        solution.unassigned_libraries = list(
            getattr(preset_solution, "unassigned_libraries", []) or []
        )
        if preset_balance_stats["required_lanes"] > 0:
            logger.info(
                "预构建Lane平衡文库前置处理完成: 需补平衡文库Lane={}，成功={}，失败移除={}，专Lane保留={}".format(
                    preset_balance_stats["required_lanes"],
                    preset_balance_stats["success_lanes"],
                    preset_balance_stats["removed_lanes"],
                    preset_balance_stats.get("preserved_dedicated_lanes", 0),
                )
            )
        valid_preset_lanes, failed_preset_lanes = _filter_valid_lanes(preset_lanes, strict_validator)
        logger.info(
            f"\n合并{len(valid_preset_lanes)}条预构建Lane到最终结果"
            f"（过滤掉{len(failed_preset_lanes)}条未通过严格校验的预构建Lane）"
        )
        if failed_preset_lanes:
            for lane in failed_preset_lanes:
                solution.unassigned_libraries.extend(
                    [
                        lib for lib in list(lane.libraries or [])
                        if not _is_ai_balance_library(lib)
                    ]
                )
        solution.lane_assignments = valid_preset_lanes + solution.lane_assignments
        logger.info(
            f"最终Lane总数: {len(solution.lane_assignments)} "
            f"(预构建Lane: {len(valid_preset_lanes)}, "
            f"普通Lane: {len(solution.lane_assignments) - len(valid_preset_lanes)})"
        )

    if presplit_family_context:
        solution.lane_assignments, solution.unassigned_libraries, rollback_records = (
            scheduler._rollback_incomplete_presplit_families(
                lanes=solution.lane_assignments,
                unassigned=solution.unassigned_libraries,
                family_context=presplit_family_context,
            )
        )
        if rollback_records:
            logger.info(
                "全局预拆分回滚完成: {}个原始文库因拆分子文库未全部成Lane而回滚",
                len(rollback_records),
            )

    blocked_rollback_libs: List[EnhancedLibraryInfo] = []
    split_rollback_unassigned_only = [
        lib for lib in list(solution.unassigned_libraries or [])
        if _is_split_rollback_unassigned_only(lib)
    ]
    if split_rollback_unassigned_only:
        mode_1_1_eligible_rollback_libs = [
            lib for lib in split_rollback_unassigned_only
            if _is_split_rollback_mode_1_1_eligible(lib)
        ]
        for lib in mode_1_1_eligible_rollback_libs:
            lib._current_seq_mode_raw = "1.1"
            setattr(lib, "_split_rollback_mode_1_1_candidate", True)
        blocked_rollback_libs = [
            lib for lib in split_rollback_unassigned_only
            if id(lib) not in {id(item) for item in mode_1_1_eligible_rollback_libs}
        ]
        rollback_ids = {id(lib) for lib in split_rollback_unassigned_only}
        solution.unassigned_libraries = [
            lib for lib in list(solution.unassigned_libraries or [])
            if id(lib) not in rollback_ids
        ]
        solution.split_rollback_mode_1_1_libraries = list(mode_1_1_eligible_rollback_libs)
        logger.info(
            "拆分失败回滚原始文库处理完成: 可回1.1候选={}个(合同量<={}G)，继续隔离={}个".format(
                len(mode_1_1_eligible_rollback_libs),
                ROLLBACK_SPLIT_LIBRARY_MODE_1_1_MAX_GB,
                len(blocked_rollback_libs),
            )
        )

    repair_split_1_1_stats = _repair_split_libraries_in_non_36t_lanes(solution, strict_validator)
    if repair_split_1_1_stats["removed_split_libraries"] > 0:
        logger.info(
            "非3.6T Lane拆分子文库即时修复统计: 影响Lane={}，剔除={}，补位={}，修复={}".format(
                repair_split_1_1_stats["affected_lanes"],
                repair_split_1_1_stats["removed_split_libraries"],
                repair_split_1_1_stats["replacement_libraries"],
                repair_split_1_1_stats["repaired_lanes"],
            )
        )

    if _should_skip_final_priority_gate_for_hybrid_mode_1_1(solution):
        logger.info(
            "检测到3.6T-NEW预消耗Lane与1.1 Lane并存，跳过最终全局优先级硬约束收口，保留已成Lane的1.1结果"
        )
    else:
        priority_gate_stats = _enforce_global_priority_hard_constraint(
            solution=solution,
            validator=strict_validator,
        )
        if priority_gate_stats["adjusted_lanes"] > 0:
            logger.info(
                "全局优先级硬约束收口完成: 调整Lane={}，移除Lane={}，暂缓较低优先级文库={}".format(
                    priority_gate_stats["adjusted_lanes"],
                    priority_gate_stats["removed_lanes"],
                    priority_gate_stats["deferred_libraries"],
                )
            )
            logger.info(
                "全局优先级硬约束收口后: 最终Lane数={}，未分配文库={}".format(
                    len(solution.lane_assignments),
                    len(solution.unassigned_libraries),
                )
            )

    # 大池 rescue 策略：不再整段跳过，而是按未分配池大小降配执行。
    # - 超小池（< 1200 个且 < 12000G）：全量执行，quotas 为默认值
    # - 中等池（>= 1200 个或 >= 12000G，但未达到极大）：降配执行，每机台每阶段最多 4 条
    # - 极大池（>= 4000 个或 >= 30000G）：仅执行一轮聚簇，每机台每阶段最多 2 条，节省时效
    # 特殊情况：0 条 Lane 通过验证且未分配池超过阈值 → 继续跳过，防止无效搜索
    _unassigned_rescue = list(getattr(solution, "unassigned_libraries", []) or [])
    _pool_lib_count = len(_unassigned_rescue)
    _pool_data_gb = _total_lane_data(_unassigned_rescue)
    _lane_count = len(getattr(solution, "lane_assignments", []) or [])

    if not enable_expensive_rescue:
        logger.info(
            "剩余库分层重组搜索已按调用方要求关闭: 未分配={}个/{:.1f}G".format(
                _pool_lib_count,
                _pool_data_gb,
            )
        )
        layered_regroup_stats = {
            "new_lanes": 0,
            "priority_cluster_lanes": 0,
            "mixed_rescue_lanes": 0,
            "normal_cluster_lanes": 0,
            "remaining_unassigned": _pool_lib_count,
            "skipped_split_libraries": 0,
        }
    elif _lane_count == 0 and _pool_lib_count >= ZERO_LANE_RESCUE_SKIP_LIB_THRESHOLD:
        # 完全没有 Lane 形成，继续 rescue 没有意义，跳过
        logger.info(
            "分层重组救援跳过: 当前0条Lane通过验证，未分配池={}个/{:.1f}G，跳过".format(
                _pool_lib_count, _pool_data_gb
            )
        )
        layered_regroup_stats = {
            "new_lanes": 0,
            "priority_cluster_lanes": 0,
            "mixed_rescue_lanes": 0,
            "normal_cluster_lanes": 0,
            "remaining_unassigned": _pool_lib_count,
            "skipped_split_libraries": 0,
        }
    elif _pool_lib_count >= 4000 or _pool_data_gb >= 30000.0:
        # 极大池：每机台每阶段只允许最多 2 条，快速覆盖最大聚簇
        logger.info(
            "分层重组救援（极大池降配）: 未分配={}个/{:.1f}G，每机台每阶段限 2 条".format(
                _pool_lib_count, _pool_data_gb
            )
        )
        layered_regroup_stats = _rescue_remaining_lanes_by_layered_regroup_search(
            solution=solution,
            validator=strict_validator,
            max_priority_cluster_lanes_per_machine=2,
            max_mixed_rescue_lanes_per_machine=2,
            max_normal_cluster_lanes_per_machine=2,
            skip_split_rule_libraries=True,
        )
    elif _pool_lib_count >= 1200 or _pool_data_gb >= 12000.0:
        # 中等大池：每机台每阶段最多 4 条，节省时效同时做有效搜索
        logger.info(
            "分层重组救援（大池降配）: 未分配={}个/{:.1f}G，每机台每阶段限 4 条".format(
                _pool_lib_count, _pool_data_gb
            )
        )
        layered_regroup_stats = _rescue_remaining_lanes_by_layered_regroup_search(
            solution=solution,
            validator=strict_validator,
            max_priority_cluster_lanes_per_machine=4,
            max_mixed_rescue_lanes_per_machine=4,
            max_normal_cluster_lanes_per_machine=4,
            skip_split_rule_libraries=True,
        )
    else:
        layered_regroup_stats = _rescue_remaining_lanes_by_layered_regroup_search(
            solution=solution,
            validator=strict_validator,
            skip_split_rule_libraries=True,
        )
    if layered_regroup_stats["new_lanes"] > 0:
        logger.info(
            "剩余库分层重组搜索完成: 新增Lane={} (大簇专项={}, 高约束专项={}, 专lane={}, 混排lane={}, 普通lane={}), "
            "跳过拆分文库={}, 剩余未分配={}".format(
                layered_regroup_stats["new_lanes"],
                layered_regroup_stats.get("major_cluster_lanes", 0),
                layered_regroup_stats.get("high_constraint_lanes", 0),
                layered_regroup_stats["priority_cluster_lanes"],
                layered_regroup_stats["mixed_rescue_lanes"],
                layered_regroup_stats["normal_cluster_lanes"],
                layered_regroup_stats["skipped_split_libraries"],
                layered_regroup_stats["remaining_unassigned"],
            )
        )
    else:
        logger.info(
            "剩余库分层重组搜索未新增Lane: 跳过拆分文库={}, 剩余未分配={}".format(
                layered_regroup_stats["skipped_split_libraries"],
                layered_regroup_stats["remaining_unassigned"],
            )
        )

    if blocked_rollback_libs:
        solution.unassigned_libraries.extend(blocked_rollback_libs)
        logger.info(
            "超过{}G的拆分失败回滚原始文库已恢复到未分配输出: {}个，不参与任何后续成Lane阶段".format(
                ROLLBACK_SPLIT_LIBRARY_MODE_1_1_MAX_GB,
                len(blocked_rollback_libs),
            )
        )

    dedup_stats = _deduplicate_solution_libraries(solution)
    if any(v > 0 for v in dedup_stats.values()):
        logger.warning(
            "终态文库去重: Lane内重复移除={}，空Lane移除={}，未分配与已分配冲突移除={}，未分配重复移除={}",
            dedup_stats["removed_assigned_duplicates"],
            dedup_stats["removed_empty_lanes"],
            dedup_stats["removed_unassigned_assigned_overlap"],
            dedup_stats["removed_unassigned_duplicates"],
        )

    renamed_lane_ids = _ensure_unique_lane_ids(solution.lane_assignments)
    if renamed_lane_ids > 0:
        logger.warning("最终收口发现重复Lane ID并已重命名: {}条", renamed_lane_ids)

    logger.info("脚本内置 Pooling 预测已停用，后续统一调用 prediction_delivery")
    stats = analyze_solution(solution)
    return stats, solution


def _schedule_rollback_libraries_in_mode_1_1(
    libraries: List[EnhancedLibraryInfo],
    mode_1_1_config: Optional[Dict[str, Any]],
) -> RollbackMode11ScheduleResult:
    """将合同量不超过阈值的拆分回滚原始文库回流1.1排机。"""
    result = RollbackMode11ScheduleResult(remaining_libraries=list(libraries or []))
    if not libraries:
        return result
    if not mode_1_1_config:
        logger.info("拆分回滚原始文库回流1.1跳过: 未加载1.1模式配置，候选{}个", len(libraries))
        return result

    first_round_label = str(mode_1_1_config.get("first_round_label", "1.1第一轮"))
    enable_expensive_rescue = bool(mode_1_1_config.get("rollback_enable_expensive_rescue", True))
    enable_peak_window = bool(mode_1_1_config.get("rollback_enable_peak_window_mixed_lanes", False))
    enable_post_fill = bool(mode_1_1_config.get("rollback_enable_post_fill_optimization", False))

    candidates: List[EnhancedLibraryInfo] = []
    for lib in libraries:
        lib.wkissplit = ""
        lib.is_split = False
        lib.split_status = "rolled_back"
        lib._current_seq_mode_raw = "1.1"
        if hasattr(lib, "_split_family_rollback_unassigned_only"):
            delattr(lib, "_split_family_rollback_unassigned_only")
        candidates.append(lib)

    logger.info(
        "拆分回滚原始文库回流1.1启动: 候选{}个，合同量阈值<={}G",
        len(candidates),
        ROLLBACK_SPLIT_LIBRARY_MODE_1_1_MAX_GB,
    )
    try:
        _, rollback_solution = test_with_model(
            deepcopy(candidates),
            existing_lanes=[],
            enable_expensive_rescue=enable_expensive_rescue,
            enable_peak_window_mixed_lanes=enable_peak_window,
            enable_post_fill_optimization=enable_post_fill,
            enable_57_rescue=False,
            enable_rule_bucket_prescheduling=False,
        )
    except Exception as exc:
        logger.error("拆分回滚原始文库回流1.1异常，保留为未分配输出: {}", exc)
        return result

    result.lanes = list(getattr(rollback_solution, "lane_assignments", []) or [])
    result.remaining_libraries = list(getattr(rollback_solution, "unassigned_libraries", []) or [])
    for lane in result.lanes:
        if not isinstance(lane.metadata, dict):
            lane.metadata = {}
        lane.metadata["dispatch_stage"] = "split_rollback_1_1"
        lane.metadata["selected_seq_mode"] = "1.1"
        lane.metadata["selected_round_label"] = first_round_label
        for lib in list(getattr(lane, "libraries", []) or []):
            lib.wkissplit = ""
            lib.is_split = False
            lib.split_status = "rolled_back"
            lib._current_seq_mode_raw = "1.1"
    for lib in result.remaining_libraries:
        lib.wkissplit = ""
        lib.is_split = False
        lib.split_status = "rolled_back"
        lib._current_seq_mode_raw = ""

    logger.info(
        "拆分回滚原始文库回流1.1完成: 新增Lane={}，剩余未分配={}",
        len(result.lanes),
        len(result.remaining_libraries),
    )
    return result


def _build_output_path(data_path: Path, output_dir: Path, mode: str) -> Path:
    """根据运行模式生成输出文件路径。"""
    suffix = "_lane_output_v6.csv" if mode == "arrange" else "_pooling_output_v6.csv"
    return output_dir / f"{data_path.stem}{suffix}"


def _run_prediction_delivery(input_data: Union[Path, pd.DataFrame], output_path: Path) -> pd.DataFrame:
    """跳过预测，仅保留下单量/预测产出字段为空的稳定输出。"""
    if isinstance(input_data, pd.DataFrame):
        prediction_df = input_data.copy()
    else:
        prediction_df = _read_csv_with_encoding_fallback(Path(input_data))

    for column_name in ["lorderdata", "lai_output", "predicted_lorderdata", "ai_predicted_lorderdata", "ai_predicted_loutput"]:
        if column_name not in prediction_df.columns:
            prediction_df[column_name] = pd.NA
        prediction_df[column_name] = pd.NA

    if "lanesorter" in prediction_df.columns:
        prediction_df["lanesorter"] = pd.to_numeric(
            prediction_df["lanesorter"], errors="coerce"
        ).astype("Int64")

    prediction_df = prediction_df.drop(
        columns=[
            BALANCE_LIBRARY_MARKER_COLUMN,
            "resolved_round2_pooling_factor",
            "resolved_round2_balance_ratio",
        ],
        errors="ignore",
    )
    prediction_df = _drop_redundant_output_columns(prediction_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(output_path, index=False)
    logger.info(
        "已跳过 prediction_delivery 模型预测，输出保留下单量/预测产出字段空值: {}",
        output_path,
    )
    return prediction_df


def arrange_library(
    data_file: Union[str, Path],
    mode: str = "arrange",
    output_detail_dir: Union[str, Path, None] = None,
    output_file: Union[str, Path, None] = None,
) -> Path:
    """
    封装的排机主函数：支持排机（arrange）与仅执行 Pooling 预测（pooling）。

    Args:
        data_file: 输入数据文件路径（CSV）。
        mode: 运行模式：
            - "arrange"：加载数据、排机、预测，全流程执行
            - "pooling"：仅对已排机结果执行预测
        output_detail_dir: 明细输出目录。当未显式提供 output_file 时，用于自动拼接输出文件名。
        output_file: 明细输出文件完整路径（包含文件名）。如果提供，则优先生效，忽略 output_detail_dir 的自动命名规则。

    Returns:
        实际写出的明细/预测结果文件路径。
    """
    random.seed(42)
    logger.info("=" * 80)
    logger.info("arrange_library: 端到端排机流程 - 排机与 Pooling 预测")
    logger.info("=" * 80)
    logger.info(f"调用时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"随机种子: 42 (固定，确保可复现)")
    logger.info(f"运行模式: {mode}")
    _reset_auto_lane_serial_counters()

    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    logger.info(f"\n使用数据文件: {data_path}")

    # 优先使用 output_file；未指定时退回到目录+自动命名逻辑
    if output_file is not None:
        output_path = Path(output_file)
        output_dir = output_path.parent
    else:
        if output_detail_dir is None:
            # 回退到原脚本中的默认目录
            output_dir = Path("/data/work/yuyongpeng/liblane_v2_deepseek/data/merge_data")
        else:
            output_dir = Path(output_detail_dir)
        output_path = _build_output_path(data_path, output_dir, mode)

    # 仅执行 Pooling 预测
    if mode == "pooling":
        logger.info("\n" + "=" * 80)
        logger.info("arrange_library: 跳过 Pooling 预测，仅保留空字段输出")
        logger.info("=" * 80)
        _run_prediction_delivery(input_data=data_path, output_path=output_path)
        logger.info(f"输出后处理完成，输出文件: {output_path}")
        return output_path

    # ===== 全流程排机模式 =====
    df_raw = _read_csv_with_encoding_fallback(data_path)
    df_raw["origrec_key"] = _build_origrec_key(df_raw)
    if "loutput" in df_raw.columns:
        loutput_series = pd.to_numeric(df_raw["loutput"], errors="coerce")
    else:
        loutput_series = pd.Series([pd.NA] * len(df_raw))
    loutput_by_origrec = dict(zip(df_raw["origrec_key"], loutput_series))

    libraries = load_test_libraries(str(data_path))
    logger.info(f"成功转换 {len(libraries)} 个测试文库")
    if not libraries:
        logger.error("未加载到任何文库数据")
        return

    ai_schedulable_libraries: List[EnhancedLibraryInfo] = []
    non_ai_libraries: List[EnhancedLibraryInfo] = []
    excluded_machine_libraries: List[EnhancedLibraryInfo] = []
    excluded_machine_reasons: Dict[str, str] = {}
    ai_schedulable_keys: Set[str] = set()
    for lib in libraries:
        machine_type = getattr(lib, "machine_type", None) or _resolve_machine_type_enum_simple(getattr(lib, "eq_type", ""))
        lib.machine_type = machine_type
        origrec_key = _safe_str(getattr(lib, "_origrec_key", getattr(lib, "origrec", "")))
        machine_exclusion_reason = _get_machine_arrangement_exclusion_reason(machine_type)
        if machine_exclusion_reason:
            lib._aiavailable_raw = "no"
            lib._unaireason_raw = machine_exclusion_reason
            excluded_machine_libraries.append(lib)
            if origrec_key:
                excluded_machine_reasons[origrec_key] = machine_exclusion_reason
            continue
        if _is_yes_value(getattr(lib, "_aiavailable_raw", "")):
            ai_schedulable_libraries.append(lib)
            ai_schedulable_keys.add(origrec_key)
        else:
            non_ai_libraries.append(lib)
    detail_ai_schedulable_keys = set(ai_schedulable_keys)
    logger.info(
        "AI可排文库筛选完成: 可排={}，不可排={}，机型排除={}".format(
            len(ai_schedulable_libraries), len(non_ai_libraries), len(excluded_machine_libraries)
        )
    )
    if excluded_machine_libraries:
        excluded_machine_summary = sorted(
            {
                _safe_str(getattr(lib, "eq_type", ""), default="Unknown")
                for lib in excluded_machine_libraries
            }
        )
        logger.warning(
            "以下机型已从V6排机主流程中显式排除: {}",
            ", ".join(excluded_machine_summary),
        )
    if not ai_schedulable_libraries:
        logger.warning("无AI可排文库（aiavailable!=yes），本次不执行排机，仅输出明细规则结果")

    # ===== 步骤0.9: 提前处理1.1第二轮历史固定组合文库 =====
    mode_1_1_config = get_scheduling_config().get_mode_1_1_config()
    mode_1_1_lanes: List[LaneAssignment] = []
    if ai_schedulable_libraries and mode_1_1_config:
        logger.info("\n" + "=" * 80)
        logger.info("步骤0.9: 提前处理1.1第二轮历史固定组合文库")
        logger.info("=" * 80)
        round2_handler = Mode11Round2Handler(mode_1_1_config)
        round2_result = round2_handler.identify_round2_candidates(ai_schedulable_libraries)
        if round2_result.total_candidates > 0:
            round2_schedule_result = round2_handler.schedule_round2(round2_result.candidate_groups)
            mode_1_1_lanes.extend(round2_schedule_result.lanes)
            ai_schedulable_libraries = (
                list(round2_result.non_candidates)
                + list(round2_schedule_result.fallback_libraries)
            )
            logger.info(
                "1.1第二轮历史Lane前置直出完成: 候选文库={}, 生成Lane={}, 剩余进入后续排机文库={}, 回流未成Lane池={}",
                round2_result.total_candidates,
                len(round2_schedule_result.lanes),
                len(ai_schedulable_libraries),
                len(round2_schedule_result.fallback_libraries),
            )
        else:
            logger.info("1.1第二轮历史Lane前置直出: 无候选文库")
    elif ai_schedulable_libraries:
        logger.info("未加载1.1模式配置，跳过1.1第二轮历史Lane前置直出")

    # ===== 步骤1: 处理包Lane文库 =====
    logger.info("\n" + "=" * 80)
    logger.info("步骤1: 处理包Lane文库（wkbaleno字段有值的文库）")
    logger.info("=" * 80)
    
    package_lanes = []
    package_libs = []
    normal_libs = []
    failed_package_libs = []
    deferred_after_1_1_libs: List[EnhancedLibraryInfo] = []
    
    for lib in ai_schedulable_libraries:
        baleno = _get_package_lane_number_from_library(lib)
        if baleno:
            lib.package_lane_number = baleno
            lib.baleno = baleno
            lib.is_package_lane = '是'
            package_libs.append(lib)
        else:
            normal_libs.append(lib)
    
    logger.info(f"包Lane文库数: {len(package_libs)}")
    logger.info(f"普通文库数: {len(normal_libs)}")
    
    if package_libs:
        logger.info("\n使用PackageLaneScheduler处理包Lane文库...")
        package_scheduler = PackageLaneScheduler()
        package_result = package_scheduler.schedule(package_libs)
        
        logger.info(f"包Lane处理结果:")
        logger.info(f"  - 成功生成Run数: {package_result.total_runs}")
        logger.info(f"  - 成功生成Lane数: {package_result.total_lanes}")
        logger.info(f"  - 已分配文库数: {package_result.total_libraries}")
        logger.info(f"  - 失败包数: {len(package_result.failed_packages)}")
        logger.info(f"  - 剩余未分配: {len(package_result.remaining_libraries)}")
        
        for run in package_result.runs:
            for lane_result in run.lanes:
                lane_metadata = {
                    'is_package_lane': True,
                    'package_id': lane_result.package_id,
                    'selected_seq_mode': 'Lane seq',
                    'seq_mode': 'Lane seq',
                    'lcxms': 'Lane seq',
                }
                if float(getattr(lane_result, "planned_balance_data_gb", 0.0) or 0.0) > 0:
                    lane_metadata["wkbalancedata"] = round(
                        float(getattr(lane_result, "planned_balance_data_gb", 0.0) or 0.0),
                        3,
                    )
                package_lane_id = f"PKG_{lane_result.lane_id}"
                lane_assignment = LaneAssignment(
                    lane_id=package_lane_id,
                    machine_id=f"M_{package_lane_id}",
                    machine_type=_resolve_machine_type_enum_simple(run.machine_type),
                    libraries=lane_result.libraries,
                    total_data_gb=lane_result.total_data_gb,
                    pooling_coefficients=lane_result.pooling_coefficients,
                    metadata=lane_metadata,
                )
                package_lanes.append(lane_assignment)

        failed_package_libs, recovered_normal_libs = _partition_remaining_package_libraries(
            package_result.remaining_libraries
        )
        normal_libs.extend(recovered_normal_libs)
        if failed_package_libs:
            logger.warning(
                "包Lane失败文库保持未分配，不进入普通排机: {}个文库，{}个失败包",
                len(failed_package_libs),
                len(package_result.failed_packages),
            )
        
        logger.info(f"\n包Lane处理完成，形成{len(package_lanes)}条包Lane")
        logger.info(f"剩余{len(normal_libs)}个文库进入普通排机流程")
    
    # ===== 步骤1.2: 处理10+24 Lane seq文库 =====
    lane_seq_10_plus_24_lanes: List[LaneAssignment] = []
    if normal_libs:
        lane_seq_10_plus_24_libs, normal_libs = _split_10_plus_24_libraries(normal_libs)
        if lane_seq_10_plus_24_libs:
            logger.info("\n" + "=" * 80)
            logger.info("步骤1.2: 处理10+24 Lane seq文库")
            logger.info("=" * 80)
            grouped_10_plus_24: Dict[MachineType, List[EnhancedLibraryInfo]] = {}
            for lib in lane_seq_10_plus_24_libs:
                machine_type = getattr(lib, "machine_type", None) or _resolve_machine_type_enum_simple(getattr(lib, "eq_type", ""))
                grouped_10_plus_24.setdefault(machine_type, []).append(lib)
            lane_seq_unassigned: List[EnhancedLibraryInfo] = []
            for machine_type, group_libs in grouped_10_plus_24.items():
                lanes, unassigned = _build_10_plus_24_lane_seq_lanes(
                    group_libs,
                    machine_type=machine_type,
                )
                lane_seq_10_plus_24_lanes.extend(lanes)
                lane_seq_unassigned.extend(unassigned)
            if lane_seq_unassigned:
                deferred_after_1_1_libs.extend(lane_seq_unassigned)
            logger.info(
                "10+24 Lane seq处理完成: 输入文库={}, 生成Lane={}, 未分配={}",
                len(lane_seq_10_plus_24_libs),
                len(lane_seq_10_plus_24_lanes),
                len(lane_seq_unassigned),
            )

    # ===== 步骤1.3: 1.1前碱基不均衡专Lane预抽取 =====
    dedicated_imbalance_lanes: List[LaneAssignment] = []
    first_round_add_test_max_gb_per_lane = float(
        (mode_1_1_config or {}).get("first_round_add_test_max_gb_per_lane", 150.0) or 0.0
    )
    if normal_libs:
        logger.info("\n" + "=" * 80)
        logger.info("步骤1.3: 1.1前碱基不均衡专Lane预抽取")
        logger.info("=" * 80)
        dedicated_imbalance_lanes, normal_libs = _extract_global_dedicated_imbalance_lanes(
            normal_libs,
            mode_name="1.1",
            dispatch_stage="pre_1_1_dedicated_imbalance",
        )
        dedicated_imbalance_lanes, normal_libs, _ = _apply_mode_1_1_add_test_cap_to_prebuilt_lanes(
            dedicated_imbalance_lanes,
            normal_libs,
            max_add_test_gb_per_lane=first_round_add_test_max_gb_per_lane,
            stage_label="步骤1.3 1.1前碱基不均衡专Lane",
        )
        logger.info(
            "1.1前碱基不均衡专Lane预抽取完成: 生成Lane={}, 剩余进入1.1普通尝试文库={}",
            len(dedicated_imbalance_lanes),
            len(normal_libs),
        )

    # ===== 步骤1.5: 1.1首轮排机 =====
    normal_libs_for_36t: List[EnhancedLibraryInfo] = []
    reserved_small_originals_for_1_1: List[EnhancedLibraryInfo] = []
    if mode_1_1_config:
        logger.info("\n" + "=" * 80)
        logger.info("步骤1.5: 1.1首轮排机")
        logger.info("=" * 80)

        allocator = ModeAllocator(mode_1_1_config)
        current_unlaned_pool = list(normal_libs)

        split_rule_blocked_original_libs: List[EnhancedLibraryInfo] = [
            lib for lib in current_unlaned_pool if _is_split_rule_original_blocked_from_1_1(lib)
        ]
        if split_rule_blocked_original_libs:
            blocked_ids = {id(lib) for lib in split_rule_blocked_original_libs}
            current_unlaned_pool = [
                lib for lib in current_unlaned_pool if id(lib) not in blocked_ids
            ]
            normal_libs_for_36t.extend(split_rule_blocked_original_libs)
            logger.info(
                "命中拆分规则且当前合同量>500G的原始文库{}个：从1.1候选池移出，保留到3.6T-NEW流程拆分与排机",
                len(split_rule_blocked_original_libs),
            )

        pool_1_1_all = list(current_unlaned_pool)

        if pool_1_1_all:
            logger.info("步骤1.5-1: 1.1首轮使用剩余未成Lane全集，候选{}个文库", len(pool_1_1_all))
            for lib in pool_1_1_all:
                lib._current_seq_mode_raw = "1.1"
                allocator._apply_mode_1_1_quality_seed_hint(lib)
            first_round_enable_expensive_rescue = bool(
                mode_1_1_config.get("first_round_enable_expensive_rescue", True)
            )
            first_round_enable_peak_window = bool(
                mode_1_1_config.get("first_round_enable_peak_window_mixed_lanes", False)
            )
            first_round_enable_post_fill_optimization = bool(
                mode_1_1_config.get("first_round_enable_post_fill_optimization", False)
            )
            first_round_enable_rule_bucket_prescheduling = bool(
                mode_1_1_config.get("first_round_enable_rule_bucket_prescheduling", True)
            )
            first_round_enable_second_pass_for_normal = bool(
                mode_1_1_config.get("first_round_enable_second_pass_for_normal", True)
            )
            second_pass_enable_expensive_rescue = bool(
                mode_1_1_config.get(
                    "first_round_second_pass_enable_expensive_rescue",
                    first_round_enable_expensive_rescue,
                )
            )
            second_pass_enable_peak_window = bool(
                mode_1_1_config.get("first_round_second_pass_enable_peak_window_mixed_lanes", True)
            )
            second_pass_enable_post_fill_optimization = bool(
                mode_1_1_config.get("first_round_second_pass_enable_post_fill_optimization", False)
            )
            second_pass_enable_rule_bucket_prescheduling = bool(
                mode_1_1_config.get(
                    "first_round_second_pass_enable_rule_bucket_prescheduling",
                    first_round_enable_rule_bucket_prescheduling,
                )
            )
            try:
                logger.info(
                    "1.1首轮策略: expensive_rescue={}, peak_window_mixed={}, post_fill_optimization={}, rule_bucket_prescheduling={}",
                    first_round_enable_expensive_rescue,
                    first_round_enable_peak_window,
                    first_round_enable_post_fill_optimization,
                    first_round_enable_rule_bucket_prescheduling,
                )
                _1_1_stats, _1_1_solution = test_with_model(
                    deepcopy(pool_1_1_all),
                    existing_lanes=[],
                    enable_expensive_rescue=first_round_enable_expensive_rescue,
                    enable_peak_window_mixed_lanes=first_round_enable_peak_window,
                    enable_post_fill_optimization=first_round_enable_post_fill_optimization,
                    enable_57_rescue=False,
                    enable_rule_bucket_prescheduling=first_round_enable_rule_bucket_prescheduling,
                )
                first_round_label = mode_1_1_config.get("first_round_label", "1.1第一轮")
                for lane in _1_1_solution.lane_assignments:
                    if not isinstance(lane.metadata, dict):
                        lane.metadata = {}
                    lane.metadata["dispatch_stage"] = "first_round_1_1"
                    lane.metadata["selected_seq_mode"] = "1.1"
                    lane.metadata["selected_round_label"] = first_round_label
                    for lib in list(lane.libraries or []):
                        lib._current_seq_mode_raw = "1.1"
                first_round_add_test_cap_stats = _enforce_mode_1_1_add_test_cap_per_lane(
                    _1_1_solution,
                    max_add_test_gb_per_lane=first_round_add_test_max_gb_per_lane,
                )
                if first_round_add_test_cap_stats["adjusted_lanes"] > 0:
                    logger.info(
                        "1.1首轮单Lane加测/混合封顶完成: 调整Lane={}, 回退文库={}个/{:.1f}G",
                        int(first_round_add_test_cap_stats["adjusted_lanes"]),
                        int(first_round_add_test_cap_stats["overflow_libraries"]),
                        first_round_add_test_cap_stats["removed_add_test_gb"],
                    )
                (
                    _1_1_solution.lane_assignments,
                    _1_1_solution.unassigned_libraries,
                    _1_1_stage_split_stats,
                ) = _enforce_split_family_atomicity_for_stage(
                    lane_assignments=list(_1_1_solution.lane_assignments or []),
                    unassigned_libraries=list(_1_1_solution.unassigned_libraries or []),
                    validator=LaneValidator(strict_mode=True),
                    stage_label="1.1首轮",
                )
                mode_1_1_lanes.extend(list(_1_1_solution.lane_assignments))
                fallback_libs = list(_1_1_solution.unassigned_libraries or [])
                split_rule_fallback_blocked_libs = [
                    lib for lib in fallback_libs if _is_split_rule_original_blocked_from_1_1(lib)
                ]
                if split_rule_fallback_blocked_libs:
                    blocked_ids = {id(lib) for lib in split_rule_fallback_blocked_libs}
                    fallback_libs = [lib for lib in fallback_libs if id(lib) not in blocked_ids]
                    for lib in split_rule_fallback_blocked_libs:
                        lib._current_seq_mode_raw = ""
                    normal_libs_for_36t.extend(split_rule_fallback_blocked_libs)
                    logger.info(
                        "1.1首轮剩余中{}个拆分规则且当前合同量>500G文库回3.6T-NEW，不参与1.1二次补排",
                        len(split_rule_fallback_blocked_libs),
                    )
                if fallback_libs and first_round_enable_second_pass_for_normal:
                    for lib in fallback_libs:
                        lib._current_seq_mode_raw = "1.1"
                    logger.info("1.1首轮二次补排启动: 首轮剩余文库={}个", len(fallback_libs))
                    logger.info(
                        "1.1首轮二次补排策略: expensive_rescue={}, peak_window_mixed={}, post_fill_optimization={}, rule_bucket_prescheduling={}",
                        second_pass_enable_expensive_rescue,
                        second_pass_enable_peak_window,
                        second_pass_enable_post_fill_optimization,
                        second_pass_enable_rule_bucket_prescheduling,
                    )
                    try:
                        _1_1_second_stats, _1_1_second_solution = test_with_model(
                            deepcopy(fallback_libs),
                            existing_lanes=[],
                            enable_expensive_rescue=second_pass_enable_expensive_rescue,
                            enable_peak_window_mixed_lanes=second_pass_enable_peak_window,
                            enable_post_fill_optimization=second_pass_enable_post_fill_optimization,
                            enable_57_rescue=False,
                            enable_rule_bucket_prescheduling=second_pass_enable_rule_bucket_prescheduling,
                        )
                        for lane in _1_1_second_solution.lane_assignments:
                            if not isinstance(lane.metadata, dict):
                                lane.metadata = {}
                            lane.metadata["dispatch_stage"] = "first_round_1_1_second_pass"
                            lane.metadata["selected_seq_mode"] = "1.1"
                            lane.metadata["selected_round_label"] = first_round_label
                            for lib in list(lane.libraries or []):
                                lib._current_seq_mode_raw = "1.1"
                        second_round_add_test_cap_stats = _enforce_mode_1_1_add_test_cap_per_lane(
                            _1_1_second_solution,
                            max_add_test_gb_per_lane=first_round_add_test_max_gb_per_lane,
                        )
                        if second_round_add_test_cap_stats["adjusted_lanes"] > 0:
                            logger.info(
                                "1.1首轮二次补排单Lane加测/混合封顶完成: 调整Lane={}, 回退文库={}个/{:.1f}G",
                                int(second_round_add_test_cap_stats["adjusted_lanes"]),
                                int(second_round_add_test_cap_stats["overflow_libraries"]),
                                second_round_add_test_cap_stats["removed_add_test_gb"],
                            )
                        (
                            _1_1_second_solution.lane_assignments,
                            _1_1_second_solution.unassigned_libraries,
                            _1_1_second_stage_split_stats,
                        ) = _enforce_split_family_atomicity_for_stage(
                            lane_assignments=list(_1_1_second_solution.lane_assignments or []),
                            unassigned_libraries=list(_1_1_second_solution.unassigned_libraries or []),
                            validator=LaneValidator(strict_mode=True),
                            stage_label="1.1二次补排",
                        )
                        mode_1_1_lanes.extend(list(_1_1_second_solution.lane_assignments))
                        fallback_libs = list(_1_1_second_solution.unassigned_libraries or [])
                        logger.info(
                            "1.1首轮二次补排完成: 新增Lane={}, 剩余文库={}",
                            len(_1_1_second_solution.lane_assignments),
                            len(fallback_libs),
                        )
                    except Exception as second_exc:
                        logger.error("1.1首轮二次补排异常，保留首轮剩余文库进入后续3.6T-NEW: {}", second_exc)
                if fallback_libs:
                    fallback_libs_for_36t, small_original_fallback_libs = (
                        _split_small_unsplit_originals_reserved_for_mode_1_1(fallback_libs)
                    )
                    if small_original_fallback_libs:
                        reserved_small_originals_for_1_1.extend(small_original_fallback_libs)
                        logger.info(
                            "1.1首轮剩余中{}个<=500G未拆分原始文库继续保留给1.1尾货抽取，不进入普通3.6T候选池",
                            len(small_original_fallback_libs),
                        )
                    for lib in fallback_libs_for_36t:
                        lib._current_seq_mode_raw = ""
                        setattr(lib, "_normal_1_1_attempted_before_36t", True)
                    normal_libs_for_36t.extend(fallback_libs_for_36t)
                    logger.info(
                        "步骤1.5-3候选池接收1.1剩余可进3.6T文库{}个，继续进入3.6T-NEW",
                        len(fallback_libs_for_36t),
                    )
                logger.info("1.1首轮排机完成: 生成{}条Lane", len(mode_1_1_lanes))
            except Exception as exc:
                logger.error("1.1首轮排机异常，全部剩余未成Lane文库回退到3.6T-NEW: {}", exc)
                pool_for_36t, small_original_fallback_libs = (
                    _split_small_unsplit_originals_reserved_for_mode_1_1(pool_1_1_all)
                )
                if small_original_fallback_libs:
                    reserved_small_originals_for_1_1.extend(small_original_fallback_libs)
                    logger.warning(
                        "1.1首轮异常后{}个<=500G未拆分原始文库仍保留给1.1尾货抽取，不回退普通3.6T",
                        len(small_original_fallback_libs),
                    )
                for lib in pool_for_36t:
                    lib._current_seq_mode_raw = ""
                    setattr(lib, "_normal_1_1_attempted_before_36t", True)
                normal_libs_for_36t.extend(pool_for_36t)
        else:
            logger.info("步骤1.5-1: 无剩余文库进入1.1首轮")

        normal_libs = normal_libs_for_36t
        if reserved_small_originals_for_1_1:
            normal_libs.extend(reserved_small_originals_for_1_1)
            logger.info(
                "步骤1.5后追加<=500G未拆分原始文库到尾货1.1池: {}个",
                len(reserved_small_originals_for_1_1),
            )

        logger.info("1.1首轮排机阶段完成: 3.6T-NEW后续候选池={}个文库, 1.1 Lane={}条",
                     len(normal_libs), len(mode_1_1_lanes))
    else:
        logger.info("未加载1.1模式配置，跳过模式分流，全部走3.6T-NEW排机")

    if normal_libs:
        small_split_1_1_lanes, normal_libs, small_split_1_1_stats = (
            _consume_small_split_rule_originals_as_mode_1_1_lanes(
                pool=normal_libs,
                validator=LaneValidator(strict_mode=True),
                max_lanes=8,
            )
        )
        if small_split_1_1_lanes:
            small_split_1_1_lanes, normal_libs, _ = _apply_mode_1_1_add_test_cap_to_prebuilt_lanes(
                small_split_1_1_lanes,
                normal_libs,
                max_add_test_gb_per_lane=first_round_add_test_max_gb_per_lane,
                stage_label="小拆分原始文库1.1 regroup",
            )
            mode_1_1_lanes.extend(small_split_1_1_lanes)
            logger.info(
                "小拆分原始文库1.1 regroup完成: 仅原始未拆分且<=500G参与，新增Lane={}, 消耗文库={}, 剩余3.6候选={}",
                len(small_split_1_1_lanes),
                sum(len(list(getattr(lane, "libraries", []) or [])) for lane in small_split_1_1_lanes),
                len(normal_libs),
            )

    # ===== 步骤1.6: 尾货1.1专Lane与普通Lane二次抽取 =====
    trailing_dedicated_imbalance_lanes: List[LaneAssignment] = []
    tail_mode_1_1_lanes: List[LaneAssignment] = []
    if normal_libs:
        logger.info("\n" + "=" * 80)
        logger.info("步骤1.6: 尾货1.1专Lane与普通Lane二次抽取")
        logger.info("=" * 80)
        trailing_dedicated_imbalance_lanes, normal_libs = _extract_global_dedicated_imbalance_lanes(
            normal_libs,
            mode_name="1.1",
            dispatch_stage="trailing_mode_1_1_dedicated_imbalance",
        )
        trailing_dedicated_imbalance_lanes, normal_libs, _ = _apply_mode_1_1_add_test_cap_to_prebuilt_lanes(
            trailing_dedicated_imbalance_lanes,
            normal_libs,
            max_add_test_gb_per_lane=first_round_add_test_max_gb_per_lane,
            stage_label="步骤1.6 尾货1.1碱基不均衡专Lane",
        )
        if trailing_dedicated_imbalance_lanes:
            logger.info(
                "尾货1.1碱基不均衡专Lane二次抽取完成: 新增Lane={}, 剩余待普通1.1二次抽取文库={}",
                len(trailing_dedicated_imbalance_lanes),
                len(normal_libs),
            )
        else:
            logger.info("尾货1.1碱基不均衡专Lane二次抽取未形成新Lane")
        tail_mode_1_1_lanes, normal_libs, tail_mode_1_1_stats = _consume_tail_libraries_as_mode_1_1_lanes(
            pool=normal_libs,
            validator=LaneValidator(strict_mode=True),
            max_lanes=8,
        )
        if tail_mode_1_1_lanes:
            tail_mode_1_1_lanes, normal_libs, _ = _apply_mode_1_1_add_test_cap_to_prebuilt_lanes(
                tail_mode_1_1_lanes,
                normal_libs,
                max_add_test_gb_per_lane=first_round_add_test_max_gb_per_lane,
                stage_label="步骤1.6 尾货1.1普通Lane",
            )
            mode_1_1_lanes.extend(tail_mode_1_1_lanes)
            logger.info(
                "尾货1.1普通Lane二次抽取完成: 新增Lane={}, 消耗文库={}, 剩余进入普通3.6T排机文库={}",
                len(tail_mode_1_1_lanes),
                sum(len(list(getattr(lane, "libraries", []) or [])) for lane in tail_mode_1_1_lanes),
                len(normal_libs),
            )
        else:
            logger.info(
                "尾货1.1普通Lane二次抽取未形成新Lane，剩余进入普通3.6T排机文库={}",
                len(normal_libs),
            )

    final_small_original_holdout: List[EnhancedLibraryInfo] = []
    if normal_libs:
        normal_libs, final_small_original_holdout = (
            _split_small_unsplit_originals_reserved_for_mode_1_1(normal_libs)
        )
        if final_small_original_holdout:
            split_after_mode_1_1_exhausted: List[EnhancedLibraryInfo] = []
            unsplittable_holdout: List[EnhancedLibraryInfo] = []
            for lib in final_small_original_holdout:
                lib._current_seq_mode_raw = "3.6T-NEW"
                lib.selected_seq_mode = "3.6T-NEW"
                lib.current_seq_mode = "3.6T-NEW"
                lib.lcxms = "3.6T-NEW"
                setattr(lib, "_mode_1_1_exhausted_allow_36t_split", True)
                if _should_library_split_in_3_6t(lib):
                    split_after_mode_1_1_exhausted.append(lib)
                else:
                    if hasattr(lib, "_mode_1_1_exhausted_allow_36t_split"):
                        delattr(lib, "_mode_1_1_exhausted_allow_36t_split")
                    lib._current_seq_mode_raw = "1.1"
                    lib.selected_seq_mode = "1.1"
                    lib.current_seq_mode = "1.1"
                    lib.lcxms = "1.1"
                    unsplittable_holdout.append(lib)
            normal_libs.extend(split_after_mode_1_1_exhausted)
            deferred_after_1_1_libs.extend(unsplittable_holdout)
            logger.info(
                "步骤2前置分流: {}个<=500G未拆分原始文库已完成1.1多轮尝试但未成Lane；其中{}个允许强制拆分后进3.6T，{}个因拆分后过小保留未分配输出",
                len(final_small_original_holdout),
                len(split_after_mode_1_1_exhausted),
                len(unsplittable_holdout),
            )

    # ===== 步骤2: 处理普通文库（包括包Lane处理失败的文库） =====
    logger.info("\n" + "=" * 80)
    logger.info("步骤2: 处理普通文库（使用GreedyLaneScheduler）")
    logger.info("=" * 80)

    proactive_split_lanes: List[LaneAssignment] = []
    if normal_libs:
        proactive_split_lanes, normal_libs, proactive_split_stats = _proactively_build_split_family_lanes_from_pool(
            libraries=normal_libs,
            validator=LaneValidator(strict_mode=True),
            stage_label="步骤2前置拆分家族构建",
        )
        if proactive_split_lanes:
            logger.info(
                "步骤2前置拆分家族构建完成: 新增Lane={}, 剩余普通排机文库={}",
                len(proactive_split_lanes),
                len(normal_libs),
            )

    has_prebuilt_lanes = any(
        (
            package_lanes,
            lane_seq_10_plus_24_lanes,
            dedicated_imbalance_lanes,
            proactive_split_lanes,
            mode_1_1_lanes,
            trailing_dedicated_imbalance_lanes,
        )
    )
    if normal_libs or has_prebuilt_lanes:
        random.seed(42)
        # 设置排机超时保护：超过 SCHEDULING_TIMEOUT_SECONDS 秒强制中断
        # signal.SIGALRM 仅在 Unix/Linux 下可用，且必须在主线程中调用
        _old_handler = None
        # SIG_ERR 是 C 层面的常量，Python signal 模块没有该属性，只需检查 SIGALRM 是否存在
        _use_signal_timeout = hasattr(signal, "SIGALRM")
        if _use_signal_timeout:
            _old_handler = signal.signal(signal.SIGALRM, _scheduling_timeout_handler)
            signal.alarm(SCHEDULING_TIMEOUT_SECONDS)
            logger.info(f"排机超时保护已启动，最大允许时间: {SCHEDULING_TIMEOUT_SECONDS // 60} 分钟")

        try:
            for lane in mode_1_1_lanes:
                if not isinstance(lane.metadata, dict):
                    lane.metadata = {}
                selected_seq_mode = str(lane.metadata.get("selected_seq_mode") or "").strip()
                if _is_explicit_dedicated_imbalance_lane(lane):
                    if not selected_seq_mode:
                        selected_seq_mode = "3.6T-NEW"
                    lane.metadata["selected_seq_mode"] = selected_seq_mode
                    lane.metadata["seq_mode"] = selected_seq_mode
                    lane.metadata["lcxms"] = selected_seq_mode
                elif not selected_seq_mode:
                    lane.metadata["selected_seq_mode"] = "1.1"
                    selected_seq_mode = "1.1"
                for lib in list(lane.libraries or []):
                    lib._current_seq_mode_raw = selected_seq_mode
            # 将包Lane、10+24 Lane seq和1.1模式Lane一起纳入最终结果
            all_existing_lanes = (
                list(package_lanes)
                + list(lane_seq_10_plus_24_lanes)
                + list(dedicated_imbalance_lanes)
                + list(proactive_split_lanes)
                + list(mode_1_1_lanes)
                + list(trailing_dedicated_imbalance_lanes)
            )
            stats, solution = test_with_model(
                deepcopy(normal_libs),
                existing_lanes=all_existing_lanes,
                enable_57_rescue=False,
            )
            (
                solution.lane_assignments,
                solution.unassigned_libraries,
                main_stage_split_stats,
            ) = _enforce_split_family_atomicity_for_stage(
                lane_assignments=list(solution.lane_assignments or []),
                unassigned_libraries=list(solution.unassigned_libraries or []),
                validator=LaneValidator(strict_mode=True),
                stage_label="主排机",
            )
            rollback_mode_1_1_libraries = list(
                getattr(solution, "split_rollback_mode_1_1_libraries", []) or []
            )
            if rollback_mode_1_1_libraries:
                rollback_mode_1_1_result = _schedule_rollback_libraries_in_mode_1_1(
                    rollback_mode_1_1_libraries,
                    mode_1_1_config=mode_1_1_config,
                )
                solution.lane_assignments.extend(rollback_mode_1_1_result.lanes)
                solution.unassigned_libraries.extend(rollback_mode_1_1_result.remaining_libraries)
        except SchedulingTimeoutError as exc:
            # 超时后取消闹钟、恢复旧信号处理器，再将异常继续向上抛出
            if _use_signal_timeout:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, _old_handler or signal.SIG_DFL)
            elapsed_min = SCHEDULING_TIMEOUT_SECONDS // 60
            logger.error(f"排机超时（{elapsed_min} 分钟），强制终止: {exc}")
            raise
        except Exception:
            # 其他异常：同样先清理超时保护，再原样抛出
            if _use_signal_timeout:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, _old_handler or signal.SIG_DFL)
            raise
        else:
            # 正常完成：取消闹钟、恢复旧信号处理器
            if _use_signal_timeout:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, _old_handler or signal.SIG_DFL)
    else:
        from types import SimpleNamespace
        stats = {}
        solution = SimpleNamespace(lane_assignments=[], unassigned_libraries=[])

    if failed_package_libs:
        solution.unassigned_libraries.extend(failed_package_libs)
    if deferred_after_1_1_libs:
        solution.unassigned_libraries.extend(deferred_after_1_1_libs)
    _validate_final_package_lanes(solution)
    _validate_no_split_for_package_lane_libraries(solution)

    balance_materialize_stats = _materialize_balance_libraries_for_solution(solution)
    if balance_materialize_stats["required_lanes"] > 0:
        logger.info(
            "平衡文库后处理完成: 需补平衡文库Lane={}，成功={}，失败移除={}，专Lane保留={}，回收文库={}".format(
                balance_materialize_stats["required_lanes"],
                balance_materialize_stats["success_lanes"],
                balance_materialize_stats["removed_lanes"],
                balance_materialize_stats.get("preserved_dedicated_lanes", 0),
                balance_materialize_stats["recovered_libraries"],
            )
        )
    _validate_final_package_lanes(solution)

    final_cleanup_validator = LaneValidator(strict_mode=True)
    split_repair_stats = _repair_split_families_before_final_rollback(
        solution=solution,
        validator=final_cleanup_validator,
        max_attempts=2,
    )
    if split_repair_stats["reordered_lanes"] > 0 or split_repair_stats["placed_fragments"] > 0:
        logger.info(
            "拆分原子性修复完成: 尝试{}轮，重排Lane={}，补入片段={}".format(
                split_repair_stats["attempts"],
                split_repair_stats["reordered_lanes"],
                split_repair_stats["placed_fragments"],
            )
        )
    _validate_final_package_lanes(solution)
    _validate_no_split_for_package_lane_libraries(solution)

    # 拆分原子性复核：终态过滤前先回滚不完整或跨runid的拆分家族，再让受影响Lane继续接受终态校验。
    final_split_stats = _rollback_incomplete_split_families_in_final_solution(solution)
    if final_split_stats["rollback_families"] > 0:
        logger.warning(
            "拆分原子性复核回滚: 家族={}，不完整={}，跨runid={}，最大份数跨run红线={}，撤回片段={}，恢复原始文库={}".format(
                final_split_stats["rollback_families"],
                final_split_stats["incomplete_families"],
                final_split_stats["cross_run_families"],
                final_split_stats.get("max_split_cross_run_families", 0),
                final_split_stats["removed_fragments"],
                final_split_stats["restored_originals"],
            )
        )

    # 终态总复核：平衡文库注入完成后，对所有非包Lane再走一遍严格校验。
    # 若 Lane 仍不合规（容量/混排等），整体回退到未分配池，防止不合格 Lane 流入输出。
    cleanup_stats = _final_non_package_validation_cleanup(solution, final_cleanup_validator)
    if cleanup_stats["removed_lanes"] > 0:
        logger.warning(
            "终态总复核: 淘汰{}条不合规Lane，回收{}个文库".format(
                cleanup_stats["removed_lanes"],
                cleanup_stats["recovered_libs"],
            )
        )
        terminal_regroup_stats = _rescue_remaining_lanes_by_layered_regroup_search(
            solution,
            final_cleanup_validator,
            max_priority_cluster_lanes_per_machine=2,
            max_mixed_rescue_lanes_per_machine=2,
            max_normal_cluster_lanes_per_machine=3,
            index_conflict_attempts_per_lane=DEFAULT_INDEX_CONFLICT_ATTEMPTS * 2,
            other_failure_attempts_per_lane=DEFAULT_OTHER_FAILURE_ATTEMPTS * 2,
            skip_split_rule_libraries=True,
        )
        if terminal_regroup_stats["new_lanes"] > 0:
            logger.info(
                "终态回收池分层重组补Lane完成: 新增Lane={} (大簇专项={}, 高约束专项={}, 混排lane={}, 普通lane={}), 跳过拆分规则文库={}, 剩余未分配={}".format(
                    terminal_regroup_stats["new_lanes"],
                    terminal_regroup_stats.get("major_cluster_lanes", 0),
                    terminal_regroup_stats.get("priority_cluster_lanes", 0),
                    terminal_regroup_stats.get("mixed_rescue_lanes", 0),
                    terminal_regroup_stats.get("normal_cluster_lanes", 0),
                    terminal_regroup_stats.get("skipped_split_libraries", 0),
                    terminal_regroup_stats.get("remaining_unassigned", 0),
                )
            )
            terminal_regroup_cleanup_stats = _final_non_package_validation_cleanup(
                solution,
                final_cleanup_validator,
            )
            if terminal_regroup_cleanup_stats["removed_lanes"] > 0:
                logger.warning(
                    "终态回收池重组后二次总复核: 淘汰{}条不合规Lane，回收{}个文库".format(
                        terminal_regroup_cleanup_stats["removed_lanes"],
                        terminal_regroup_cleanup_stats["recovered_libs"],
                    )
                )
        else:
            logger.info(
                "终态回收池分层重组未新增Lane: 跳过拆分规则文库={}，剩余未分配={}".format(
                    terminal_regroup_stats.get("skipped_split_libraries", 0),
                    terminal_regroup_stats.get("remaining_unassigned", 0),
                )
            )

    post_cleanup_split_stats = _rollback_incomplete_split_families_in_final_solution(solution)
    if post_cleanup_split_stats["rollback_families"] > 0:
        logger.warning(
            "终态过滤后拆分兜底回滚: 家族={}，不完整={}，跨runid={}，最大份数跨run红线={}，撤回片段={}，恢复原始文库={}".format(
                post_cleanup_split_stats["rollback_families"],
                post_cleanup_split_stats["incomplete_families"],
                post_cleanup_split_stats["cross_run_families"],
                post_cleanup_split_stats.get("max_split_cross_run_families", 0),
                post_cleanup_split_stats["removed_fragments"],
                post_cleanup_split_stats["restored_originals"],
            )
        )
        matrix_split_stats = _try_add_matrix_split_lanes_from_unassigned(
            solution=solution,
            validator=final_cleanup_validator,
        )
        if matrix_split_stats["added_lanes"] > 0:
            logger.info(
                "拆分兜底回滚后矩阵补Lane: 新增Lane={}，使用原始文库={}，拆分片段={}".format(
                    matrix_split_stats["added_lanes"],
                    matrix_split_stats["used_originals"],
                    matrix_split_stats["added_fragments"],
                )
            )
        second_cleanup_stats = _final_non_package_validation_cleanup(solution, final_cleanup_validator)
        if second_cleanup_stats["removed_lanes"] > 0:
            logger.warning(
                "拆分兜底回滚后二次终态总复核: 淘汰{}条不合规Lane，回收{}个文库".format(
                    second_cleanup_stats["removed_lanes"],
                    second_cleanup_stats["recovered_libs"],
                )
            )
            second_split_stats = _rollback_incomplete_split_families_in_final_solution(solution)
            if second_split_stats["rollback_families"] > 0:
                logger.warning(
                    "拆分兜底回滚后二次拆分复核: 家族={}，不完整={}，跨runid={}，最大份数跨run红线={}，撤回片段={}，恢复原始文库={}".format(
                        second_split_stats["rollback_families"],
                        second_split_stats["incomplete_families"],
                        second_split_stats["cross_run_families"],
                        second_split_stats.get("max_split_cross_run_families", 0),
                        second_split_stats["removed_fragments"],
                        second_split_stats["restored_originals"],
                    )
                )
        mixed_matrix_split_stats = _try_add_mixed_matrix_split_lanes_from_unassigned(
            solution=solution,
            validator=final_cleanup_validator,
        )
        if mixed_matrix_split_stats["added_lanes"] > 0:
            logger.info(
                "二次终态修复后混合矩阵补Lane: 新增Lane={}，使用原始文库={}，拆分片段={}".format(
                    mixed_matrix_split_stats["added_lanes"],
                    mixed_matrix_split_stats["used_originals"],
                    mixed_matrix_split_stats["added_fragments"],
                )
            )
            third_cleanup_stats = _final_non_package_validation_cleanup(solution, final_cleanup_validator)
            if third_cleanup_stats["removed_lanes"] > 0:
                logger.warning(
                    "混合矩阵补Lane后三次终态总复核: 淘汰{}条不合规Lane，回收{}个文库".format(
                        third_cleanup_stats["removed_lanes"],
                        third_cleanup_stats["recovered_libs"],
                    )
                )
                third_split_stats = _rollback_incomplete_split_families_in_final_solution(solution)
                if third_split_stats["rollback_families"] > 0:
                    logger.warning(
                        "混合矩阵补Lane后三次拆分复核: 家族={}，不完整={}，跨runid={}，最大份数跨run红线={}，撤回片段={}，恢复原始文库={}".format(
                            third_split_stats["rollback_families"],
                            third_split_stats["incomplete_families"],
                            third_split_stats["cross_run_families"],
                            third_split_stats.get("max_split_cross_run_families", 0),
                            third_split_stats["removed_fragments"],
                            third_split_stats["restored_originals"],
                        )
                    )
        cross_split_stats = _try_add_cross_split_fragment_lanes_from_unassigned(
            solution=solution,
            validator=final_cleanup_validator,
        )
        if cross_split_stats["added_lanes"] > 0:
            logger.info(
                "混合矩阵修复后跨份数片段装箱补Lane: 新增Lane={}，使用原始文库={}，拆分片段={}".format(
                    cross_split_stats["added_lanes"],
                    cross_split_stats["used_originals"],
                    cross_split_stats["added_fragments"],
                )
            )
            fourth_cleanup_stats = _final_non_package_validation_cleanup(solution, final_cleanup_validator)
            if fourth_cleanup_stats["removed_lanes"] > 0:
                logger.warning(
                    "跨份数片段装箱补Lane后四次终态总复核: 淘汰{}条不合规Lane，回收{}个文库".format(
                        fourth_cleanup_stats["removed_lanes"],
                        fourth_cleanup_stats["recovered_libs"],
                    )
                )
            fourth_split_stats = _rollback_incomplete_split_families_in_final_solution(solution)
            if fourth_split_stats["rollback_families"] > 0:
                logger.warning(
                    "跨份数片段装箱补Lane后四次拆分复核: 家族={}，不完整={}，跨runid={}，最大份数跨run红线={}，撤回片段={}，恢复原始文库={}".format(
                        fourth_split_stats["rollback_families"],
                        fourth_split_stats["incomplete_families"],
                        fourth_split_stats["cross_run_families"],
                        fourth_split_stats.get("max_split_cross_run_families", 0),
                        fourth_split_stats["removed_fragments"],
                        fourth_split_stats["restored_originals"],
                    )
                )

    terminal_global_36t_stats = _try_add_terminal_global_36t_mixed_lanes(
        solution=solution,
        validator=final_cleanup_validator,
    )
    if terminal_global_36t_stats["new_lanes"] > 0:
        logger.info(
            "终态全局3.6混排增量完成: 新增Lane={}，使用拆分原始文库={}，使用普通补料文库={}，剩余未分配={}".format(
                terminal_global_36t_stats["new_lanes"],
                terminal_global_36t_stats["used_originals"],
                terminal_global_36t_stats["used_fillers"],
                terminal_global_36t_stats["remaining_unassigned"],
            )
        )
    else:
        logger.info(
            "终态全局3.6混排未新增Lane: 剩余未分配={}".format(
                terminal_global_36t_stats["remaining_unassigned"],
            )
        )

    plain_36t_to_1_1_stats = _try_convert_plain_36t_tail_lanes_to_mode_1_1(
        solution,
        final_cleanup_validator,
    )
    if plain_36t_to_1_1_stats["converted_lanes"] > 0:
        logger.info(
            "终态纯普通3.6T Lane回收合并1.1完成: 回收3.6T Lane={}，新增1.1 Lane={}，使用文库={}".format(
                plain_36t_to_1_1_stats["converted_lanes"],
                plain_36t_to_1_1_stats["new_lanes"],
                plain_36t_to_1_1_stats["used_libraries"],
            )
        )
        plain_36t_cleanup_stats = _final_non_package_validation_cleanup(
            solution,
            final_cleanup_validator,
        )
        if plain_36t_cleanup_stats["removed_lanes"] > 0:
            logger.warning(
                "终态纯普通3.6T回收合并后二次总复核: 淘汰{}条不合规Lane，回收{}个文库".format(
                    plain_36t_cleanup_stats["removed_lanes"],
                    plain_36t_cleanup_stats["recovered_libs"],
                )
            )

    terminal_sample_type_stats = _try_add_terminal_sample_type_dedicated_lanes(
        solution=solution,
        validator=final_cleanup_validator,
    )
    if terminal_sample_type_stats["new_lanes"] > 0:
        logger.info(
            "终态文库类型专池增量完成: 新增Lane={}，使用文库={}，跳过拆分规则文库={}，跳过不支持机型={}，剩余未分配={}".format(
                terminal_sample_type_stats["new_lanes"],
                terminal_sample_type_stats["used_libraries"],
                terminal_sample_type_stats["skipped_split_libraries"],
                terminal_sample_type_stats["skipped_unsupported_machine"],
                terminal_sample_type_stats["remaining_unassigned"],
            )
        )
    else:
        logger.info(
            "终态文库类型专池增量未新增Lane: 跳过拆分规则文库={}，跳过不支持机型={}，剩余未分配={}".format(
                terminal_sample_type_stats["skipped_split_libraries"],
                terminal_sample_type_stats["skipped_unsupported_machine"],
                terminal_sample_type_stats["remaining_unassigned"],
            )
        )

    final_dedicated_merge_stats = _merge_dedicated_imbalance_lanes_into_mode_1_1(solution)
    if final_dedicated_merge_stats["merged_groups"] > 0:
        logger.info(
            "终态DL同组合并到1.1完成: 合并分组={}, 移除旧Lane={}, 新增1.1专Lane={}".format(
                int(final_dedicated_merge_stats["merged_groups"]),
                int(final_dedicated_merge_stats["removed_lanes"]),
                int(final_dedicated_merge_stats["new_lanes"]),
            )
        )
        renamed_lane_ids = _ensure_unique_lane_ids(solution.lane_assignments)
        if renamed_lane_ids > 0:
            logger.warning("终态DL同组合并后发现重复Lane ID并已重命名: {}条", renamed_lane_ids)
        final_dedicated_merge_cleanup_stats = _final_non_package_validation_cleanup(
            solution,
            final_cleanup_validator,
        )
        if final_dedicated_merge_cleanup_stats["removed_lanes"] > 0:
            logger.warning(
                "终态DL同组合并后二次总复核: 淘汰{}条不合规Lane，回收{}个文库".format(
                    final_dedicated_merge_cleanup_stats["removed_lanes"],
                    final_dedicated_merge_cleanup_stats["recovered_libs"],
                )
            )

    terminal_balance_materialize_stats = _materialize_balance_libraries_for_solution(solution)
    if terminal_balance_materialize_stats["required_lanes"] > 0:
        logger.info(
            "终态增量Lane平衡文库后处理完成: 需补平衡文库Lane={}，成功={}，失败移除={}，专Lane保留={}，回收文库={}".format(
                terminal_balance_materialize_stats["required_lanes"],
                terminal_balance_materialize_stats["success_lanes"],
                terminal_balance_materialize_stats["removed_lanes"],
                terminal_balance_materialize_stats.get("preserved_dedicated_lanes", 0),
                terminal_balance_materialize_stats["recovered_libraries"],
            )
        )
        terminal_balance_cleanup_stats = _final_non_package_validation_cleanup(solution, final_cleanup_validator)
        if terminal_balance_cleanup_stats["removed_lanes"] > 0:
            logger.warning(
                "终态增量Lane补平衡后二次总复核: 淘汰{}条不合规Lane，回收{}个文库".format(
                    terminal_balance_cleanup_stats["removed_lanes"],
                    terminal_balance_cleanup_stats["recovered_libs"],
                )
            )

    if mode_1_1_config:
        final_mode_1_1_cap_stats = _enforce_mode_1_1_add_test_cap_and_cleanup(
            solution,
            final_cleanup_validator,
            max_add_test_gb_per_lane=first_round_add_test_max_gb_per_lane,
            stage_label="最终导出前1.1",
        )
        if final_mode_1_1_cap_stats.get("cleanup_removed_lanes", 0) > 0:
            final_mode_1_1_rebuild_stats = _rescue_remaining_lanes_by_layered_regroup_search(
                solution,
                final_cleanup_validator,
                max_priority_cluster_lanes_per_machine=2,
                max_mixed_rescue_lanes_per_machine=2,
                max_normal_cluster_lanes_per_machine=8,
                index_conflict_attempts_per_lane=DEFAULT_INDEX_CONFLICT_ATTEMPTS * 2,
                other_failure_attempts_per_lane=DEFAULT_OTHER_FAILURE_ATTEMPTS * 2,
                skip_split_rule_libraries=True,
            )
            if final_mode_1_1_rebuild_stats["new_lanes"] > 0:
                logger.info(
                    "最终导出前1.1封顶回收池重组补Lane完成: 新增Lane={} (大簇专项={}, 混排lane={}, 普通lane={}), 剩余未分配={}".format(
                        final_mode_1_1_rebuild_stats["new_lanes"],
                        final_mode_1_1_rebuild_stats.get("major_cluster_lanes", 0),
                        final_mode_1_1_rebuild_stats.get("mixed_rescue_lanes", 0),
                        final_mode_1_1_rebuild_stats.get("normal_cluster_lanes", 0),
                        final_mode_1_1_rebuild_stats.get("remaining_unassigned", 0),
                    )
                )
                _enforce_mode_1_1_add_test_cap_and_cleanup(
                    solution,
                    final_cleanup_validator,
                    max_add_test_gb_per_lane=first_round_add_test_max_gb_per_lane,
                    stage_label="最终导出前1.1回收池重组后",
                )
            else:
                logger.info(
                    "最终导出前1.1封顶回收池重组未新增Lane: 剩余未分配={}".format(
                        final_mode_1_1_rebuild_stats.get("remaining_unassigned", 0),
                    )
                )

    final_export_split_stats = _rollback_incomplete_split_families_in_final_solution(solution)
    if final_export_split_stats["rollback_families"] > 0:
        logger.warning(
            "最终导出前拆分原子性兜底回滚: 家族={}，不完整={}，跨runid={}，最大份数跨run红线={}，撤回片段={}，恢复原始文库={}".format(
                final_export_split_stats["rollback_families"],
                final_export_split_stats["incomplete_families"],
                final_export_split_stats["cross_run_families"],
                final_export_split_stats.get("max_split_cross_run_families", 0),
                final_export_split_stats["removed_fragments"],
                final_export_split_stats["restored_originals"],
            )
        )
        final_export_cleanup_stats = _final_non_package_validation_cleanup(solution, final_cleanup_validator)
        if final_export_cleanup_stats["removed_lanes"] > 0:
            logger.warning(
                "最终导出前拆分回滚后二次总复核: 淘汰{}条不合规Lane，回收{}个文库".format(
                    final_export_cleanup_stats["removed_lanes"],
                    final_export_cleanup_stats["recovered_libs"],
                )
            )

    final_renamed_lane_ids = _ensure_unique_lane_ids(solution.lane_assignments)
    if final_renamed_lane_ids > 0:
        logger.warning("最终导出前发现重复Lane ID并已重命名: {}条", final_renamed_lane_ids)

    # 收集预测结果
    pred_df = _collect_prediction_rows(
        solution.lane_assignments, loutput_by_origrec, "arrange"
    )

    lanes_with_split = _collect_lanes_with_split(solution.lane_assignments)
    detail_libraries = _collect_detail_output_libraries(solution)

    # 输出明细
    _build_detail_output(
        df_raw=df_raw,
        pred_df=pred_df,
        output_path=output_path,
        ai_schedulable_keys=detail_ai_schedulable_keys,
        lanes_with_split=lanes_with_split,
        detail_libraries=detail_libraries,
        excluded_machine_reasons=excluded_machine_reasons,
    )

    logger.info("\n" + "=" * 80)
    logger.info("步骤3: 跳过 prediction_delivery，保留下单量/预测产出空字段")
    logger.info("=" * 80)
    prediction_df = _run_prediction_delivery(input_data=output_path, output_path=output_path)
    logger.info(
        "输出后处理完成: 记录数={}, 平均下单量={:.3f}G, 平均产出量={:.3f}G".format(
            len(prediction_df),
            float(pd.to_numeric(prediction_df["lorderdata"], errors="coerce").mean(skipna=True)),
            float(pd.to_numeric(prediction_df["lai_output"], errors="coerce").mean(skipna=True)),
        )
    )
    logger.info("\n端到端排机完成！")
    logger.info(f"最终输出文件: {output_path}")
    logger.info("=" * 80)

    return output_path


# ==================== 入口函数（CLI 封装） ====================


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="端到端排机流程测试 - 支持排机或仅执行 Pooling 预测"
    )
    parser.add_argument(
        "--mode",
        choices=["arrange", "pooling"],
        default="arrange",
        help="运行模式：arrange=加载数据、排机、预测；pooling=仅预测",
    )
    parser.add_argument(
        "--data-file",
        default="/data/work/yuyongpeng/liblane_v2_deepseek/data/pooling_test_data/2025-12-10_merged_standardized.csv",
        help="输入数据文件路径",
    )
    parser.add_argument(
        "--output-detail-dir",
        default="/data/work/yuyongpeng/liblane_v2_deepseek/data/merge_data",
        help="明细输出目录",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="明细输出文件完整路径（包含文件名）。如果提供，则优先生效，忽略 --output-detail-dir 的文件名拼接规则",
    )
    return parser.parse_args()


def main() -> None:
    """主入口：命令行包装 arrange_library 函数"""
    args = parse_args()
    try:
        arrange_library(
            data_file=args.data_file,
            mode=args.mode,
            output_detail_dir=args.output_detail_dir,
            output_file=args.output_file,
        )
    except Exception as exc:
        logger.error(f"测试过程发生错误: {exc}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("用户中断")
    except Exception as e:
        logger.error(f"测试过程发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
