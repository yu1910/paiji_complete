"""
端到端排机流程测试 - 排机与 Pooling 预测
创建时间：2026-01-12 11:23:30
更新时间：2026-03-16 17:26:55

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
import sys
from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import warnings

# 全局关闭 pandas 的 DataFrame 高度碎片化性能告警（来自 prediction_delivery 内部）
warnings.filterwarnings(
    "ignore",
    category=pd.errors.PerformanceWarning,
    message="DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.",
)

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from liblane_paths import setup_liblane_paths

setup_liblane_paths()

from loguru import logger

from models.library_info import EnhancedLibraryInfo, MachineType
from core.config.scheduling_config import get_scheduling_config
from core.constraints.lane_validator import ValidationRuleType, ValidationError, ValidationSeverity
from core.data import load_libraries_from_csv
from core.preprocessing.rule_constrained_strategy_planner import StrategyExecutionPlan
from core.scheduling.greedy_lane_scheduler import GreedyLaneScheduler, GreedyLaneConfig
from core.scheduling.package_lane_scheduler import PackageLaneScheduler
from core.scheduling.scheduling_types import LaneAssignment

# prediction_delivery 作为独立包依赖，由 pip 安装后直接导入
from prediction_delivery import MODELS_DIR, predict_pooling

# ==================== Lane上机浓度规则 ====================
LANE_ORDERDATA_FLOOR = 1.0
SPECIAL_SPLIT_GROUP_A: Set[str] = {
    "10x_longranger",
    "10x_longranger_indexset",
    "10x_cellranger",
    "10x_cellranger_indexset",
}
SPECIAL_SPLIT_GROUP_B: Set[str] = {
    "10x_cellranger_atac_indexset",
    "10x_cellranger_atac",
}


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


def _normalize_seq_strategy_keyword(value: Any) -> str:
    """统一测序策略匹配口径。"""
    return _normalize_text_for_match(value).replace("BP", "").replace(" ", "")


def _get_lane_loading_target_machine_texts() -> Set[str]:
    """从统一规则表获取Lane上机浓度规则适用机型集合。"""
    scope = get_scheduling_config().get_loading_rule_scope()
    return set(scope.get("machine_types", set()) or set())


def _get_lane_loading_target_testno_text() -> str:
    """从统一规则表获取Lane上机浓度规则适用工序。"""
    test_nos = sorted(set(get_scheduling_config().get_loading_rule_scope().get("test_nos", set()) or set()))
    if len(test_nos) == 1:
        return str(test_nos[0])
    return ""


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

    parsed_records: List[Tuple[str, List[Tuple[str, Optional[str]]]]] = []
    for lib in libraries:
        record_id = str(getattr(lib, "origrec", "") or str(id(lib)))
        index_seq = str(getattr(lib, "index_seq", "") or "")
        pairs = _parse_index_pairs_latest(index_seq)
        if pairs:
            parsed_records.append((record_id, pairs))

    conflicts: List[LatestIndexConflict] = []
    for i in range(len(parsed_records)):
        record_id_1, pairs_1 = parsed_records[i]
        for j in range(i + 1, len(parsed_records)):
            record_id_2, pairs_2 = parsed_records[j]
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
    try:
        special_split_valid, special_split_tokens, special_split_reason = _validate_lane_special_split_rule(
            libraries
        )
    except Exception as exc:
        logger.exception(f"Lane {lane_id} wkspecialsplits规则校验失败，沿用原校验结果: {exc}")
        special_split_valid = True
        special_split_tokens = set()
        special_split_reason = "special_split_check_failed"
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
                    "{10x_cellranger_atac_indexset,10x_cellranger_atac}同Lane且不得与其他类型混排"
                    f" | 当前={sorted(special_split_tokens)} | reason={special_split_reason}"
                ),
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
    text = str(eq_type)
    if "10B" in text:
        return MachineType.NOVA_X_10B
    return MachineType.NOVA_X_25B


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
    machine_type_text = (
        machine_type.value if isinstance(machine_type, MachineType) else str(machine_type or "Nova X-25B")
    )
    metadata = _build_lane_metadata_for_validator(lane_id, lane_metadata)
    return get_scheduling_config().get_lane_capacity_range(
        libraries=libraries,
        machine_type=machine_type_text,
        metadata=metadata,
    )


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
    return sum(lib.get_data_amount_gb() for lib in libraries)


def _is_index_conflict_only(result: Any) -> bool:
    """判断失败是否仅由Index冲突导致。"""
    errors = getattr(result, "errors", None) or []
    if not errors:
        return False
    return all(err.rule_type == ValidationRuleType.INDEX_CONFLICT for err in errors)


def _attempt_build_lane_from_pool(
    pool: List[EnhancedLibraryInfo],
    validator,
    machine_type: MachineType,
    lane_id_prefix: str,
    index_conflict_attempts: int = 3,
    other_failure_attempts: int = 5,
    extra_metadata: Optional[Dict[str, Any]] = None,
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

    from core.constraints.index_validator_verified import IndexConflictValidator
    _idx_validator = IndexConflictValidator()

    index_conflict_retry_count = 0
    other_failure_retry_count = 0
    attempt_idx = 0
    while (
        index_conflict_retry_count < index_conflict_attempts
        and other_failure_retry_count < other_failure_attempts
    ):
        attempt_idx += 1
        candidates = list(pool)
        random.shuffle(candidates)
        selected: List[EnhancedLibraryInfo] = []
        total = 0.0
        random_target: Optional[float] = None
        for lib in candidates:
            data = lib.get_data_amount_gb()
            trial_libs = selected + [lib]
            trial_min_allowed, trial_max_allowed = _resolve_lane_capacity_limits(
                libraries=trial_libs,
                machine_type=machine_type,
                lane_metadata=extra_metadata,
            )
            if total + data > trial_max_allowed:
                continue
            ss_valid, _, _ = _validate_lane_special_split_rule(trial_libs)
            if not ss_valid:
                continue
            if not _idx_validator.validate_lane_quick(trial_libs):
                continue
            selected.append(lib)
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
            continue
        selected_min_allowed, _ = _resolve_lane_capacity_limits(
            libraries=selected,
            machine_type=machine_type,
            lane_metadata=extra_metadata,
        )
        if total < selected_min_allowed:
            other_failure_retry_count += 1
            continue
        lane_id = f"{lane_id_prefix}_{machine_type.value}_{attempt_idx:03d}"
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
        metadata = _build_lane_metadata_for_validator(lane.lane_id, lane.metadata)
        result = _validate_lane_with_latest_index(
            validator=validator,
            libraries=lane.libraries,
            lane_id=lane.lane_id,
            machine_type=lane.machine_type.value,
            metadata=metadata,
        )
        if result.is_valid:
            return lane, selected
        if _is_index_conflict_only(result):
            index_conflict_retry_count += 1
        else:
            other_failure_retry_count += 1
    return None, []


def _get_library_identity_key(lib: EnhancedLibraryInfo) -> str:
    """获取文库在当前流程中的稳定唯一键。"""
    origrec_key = _safe_str(
        getattr(lib, "_origrec_key", getattr(lib, "origrec", "")),
        default="",
    )
    if origrec_key:
        return origrec_key
    return str(id(lib))


def _extract_dedicated_10bp_lanes(
    libraries: List[EnhancedLibraryInfo],
    validator,
    machine_type: MachineType = MachineType.NOVA_X_25B,
    max_lanes: Optional[int] = None,
    index_conflict_attempts_per_lane: int = 80,
    other_failure_attempts_per_lane: int = 160,
) -> Tuple[List[LaneAssignment], List[EnhancedLibraryInfo]]:
    """优先从10bp文库中抽取纯10bp专Lane，再返回剩余待排文库。"""
    if not libraries:
        return [], []

    libs_10bp, _ = _split_10bp_and_non_10bp(libraries, validator)
    if not libs_10bp:
        return [], list(libraries)

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
        return [], list(libraries)

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
    index_conflict_attempts_per_lane: int = 100,
    other_failure_attempts_per_lane: int = 200,
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

    ps_min, ps_max, window_libs = _find_best_peak_size_window(libraries)
    if not window_libs:
        return [], list(libraries)

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
        return [], list(libraries)

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

    for _ in range(target_lane_count):
        lane, used = _attempt_build_lane_from_pool(
            pool=remaining_pool,
            validator=validator,
            machine_type=machine_type,
            lane_id_prefix="MX",
            index_conflict_attempts=index_conflict_attempts_per_lane,
            other_failure_attempts=other_failure_attempts_per_lane,
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
    index_conflict_attempts_per_lane: int = 3,
    other_failure_attempts_per_lane: int = 5,
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

            new_lane, used = _attempt_build_lane_from_pool(
                pool, validator, machine_type, "EX",
                index_conflict_attempts=index_conflict_attempts_per_lane,
                other_failure_attempts=other_failure_attempts_per_lane,
            )
            if new_lane:
                for lib in used:
                    if lib in unassigned:
                        unassigned.remove(lib)
                lanes.append(new_lane)
                added += 1
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
            metadata = _build_lane_metadata_for_validator(lane.lane_id, lane.metadata)
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


def try_multi_lib_swap_rebalance(
    solution,
    validator,
    max_new_lanes: int = 2,
    max_donations: int = 40,
    index_conflict_max_trials: int = 3,
    other_failure_max_trials: int = 5,
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

            new_lane, used = _attempt_build_lane_from_pool(
                pool, validator, machine_type, "RB",
                index_conflict_attempts=index_conflict_max_trials,
                other_failure_attempts=other_failure_max_trials,
            )
            if new_lane:
                for lib in used:
                    if lib in unassigned:
                        unassigned.remove(lib)
                lanes.append(new_lane)
                new_lanes_count += 1

    return {"new_lanes": new_lanes_count, "remaining_unassigned": len(unassigned)}


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


def _is_split_library(lib: EnhancedLibraryInfo) -> bool:
    """判断文库是否为拆分文库。"""
    if bool(getattr(lib, "is_split", False)):
        return True
    wkissplit = _safe_str(getattr(lib, "wkissplit", ""), default="")
    if _is_yes_value(wkissplit):
        return True
    split_status = _safe_str(getattr(lib, "split_status", ""), default="").lower()
    return split_status == "completed"


def _collect_lanes_with_split(lanes: List[LaneAssignment]) -> Set[str]:
    """收集包含拆分文库的lane_id集合。"""
    lane_ids: Set[str] = set()
    for lane in lanes:
        libs = list(getattr(lane, "libraries", []) or [])
        if any(_is_split_library(lib) for lib in libs):
            lane_ids.add(str(lane.lane_id))
    return lane_ids


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


def _is_medical_commission_library(lib: EnhancedLibraryInfo) -> bool:
    """判断文库是否属于医学委托项目。"""
    sub_project_name = _normalize_text_for_match(getattr(lib, "sub_project_name", ""))
    if not sub_project_name:
        return False
    has_medical_keyword = ("医学" in sub_project_name) or ("医检所" in sub_project_name)
    return has_medical_keyword and ("委托" in sub_project_name)


def _matches_lane_seq_strategy_keyword(
    lib: EnhancedLibraryInfo, strategy_keyword: str
) -> bool:
    """判断文库是否命中指定测序策略关键字。"""
    normalized_keyword = _normalize_seq_strategy_keyword(strategy_keyword)
    if not normalized_keyword:
        return False

    strategy_candidates = [
        getattr(lib, "_lane_sj_mode_raw", ""),
        getattr(lib, "test_no", ""),
        getattr(lib, "seq_scheme", ""),
        getattr(lib, "machine_note", ""),
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

    if rule_type == "medical_commission_threshold":
        threshold = float(rule.get("data_threshold_gb", 0.0) or 0.0)
        medical_project_data = sum(
            float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
            for lib in libraries
            if _is_medical_commission_library(lib)
        )
        return medical_project_data > threshold

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
) -> Tuple[Optional[float], str]:
    """按统一规则表计算Lane上机浓度（未命中返回空）。"""
    if not libraries:
        return None, "empty_lane"
    return get_scheduling_config().resolve_loading_concentration(libraries)


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
        "wklistqpcr": None,
        "wklastorderdata": None,
        "wklastoutput": None,
        "wklastoutrate": None,
    }
    if not _is_add_test_library(lib):
        return result

    result["applied"] = True
    current_qpcr = _get_lib_attr_float(lib, ["qpcr_molar", "qpcr_concentration"])
    last_qpcr = _get_lib_attr_float(lib, ["_last_qpcr_raw", "last_qpcr", "wklistqpcr"])
    last_outrate = _get_lib_attr_float(lib, ["_last_outrate_raw", "last_outrate", "wklastoutrate"])
    last_order = _get_lib_attr_float(lib, ["_last_order_data_raw", "last_order_data", "wklastorderdata"])
    last_output = _get_lib_attr_float(lib, ["_last_output_raw", "last_output", "wklastoutput"])

    result["wklistqpcr"] = last_qpcr
    result["wklastorderdata"] = last_order
    result["wklastoutput"] = last_output
    result["wklastoutrate"] = last_outrate

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

    effective_last_outrate = max(last_outrate or 0.0, 0.3)
    result["effective_last_outrate"] = effective_last_outrate
    historical_based_order = contract_data / effective_last_outrate if effective_last_outrate > 0 else ai_predicted_order
    result["historical_based_order"] = historical_based_order

    selected_order = max(ai_predicted_order, historical_based_order)
    result["selected_order"] = selected_order

    # 该规则仅修正下单量，不修正产出量，产出量始终保持AI预测结果。
    result["selected_output"] = ai_predicted_output
    result["rule_reason"] = "qpcr_within_15pct_compare_ai_vs_historical"
    return result


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
    """为Lane生成runid映射，每个runid最多包含指定数量的Lane"""
    if lanes_per_run <= 0:
        raise ValueError("lanes_per_run必须大于0")
    runid_by_lane: Dict[str, str] = {}
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    for idx, lane in enumerate(lanes):
        run_index = idx // lanes_per_run + 1
        runid_by_lane[lane.lane_id] = f"RUN_{timestamp}_{run_index:03d}"
    return runid_by_lane


def _build_lane_metadata_for_validator(
    lane_id: str,
    lane_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """根据Lane前缀构建验证所需的metadata"""
    metadata: Dict[str, Any] = {}
    if lane_id.startswith("DL_"):
        metadata["is_dedicated_imbalance_lane"] = True
    if lane_id.startswith("NB_"):
        metadata["is_pure_non_10bp_lane"] = True
    if lane_id.startswith("BL_"):
        metadata["is_backbone_lane"] = True
    if lane_metadata:
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


def _validate_lane_state(
    validator: Any,
    lane: LaneAssignment,
    libraries: List[EnhancedLibraryInfo],
) -> Any:
    """校验给定文库列表在当前Lane上下文中的合法性。"""
    metadata = _build_lane_metadata_for_validator(lane.lane_id, lane.metadata)
    machine_type = lane.machine_type.value if lane.machine_type else "Nova X-25B"
    return _validate_lane_with_latest_index(
        validator=validator,
        libraries=libraries,
        lane_id=lane.lane_id,
        machine_type=machine_type,
        metadata=metadata,
    )


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
    remaining = [lib for lib in libraries if lib not in to_remove]
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
        if lib in lane.libraries:
            lane.remove_library(lib)
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
        trial_result = _validate_lane_state(strict_validator, lane, trial_libs)
        return bool(trial_result.is_valid)

    # 先从未分配池补齐
    for lib in sorted(list(unassigned_pool), key=lambda x: float(getattr(x, "contract_data_raw", 0.0) or 0.0), reverse=True):
        if lane.total_data_gb >= min_allowed:
            break
        if not can_add(lib):
            continue
        lane.add_library(lib)
        if lib in unassigned_pool:
            unassigned_pool.remove(lib)
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
    metadata = _build_lane_metadata_for_validator(lane.lane_id, lane.metadata)
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
            target_non_cust = max(data_non_customers, data_customers / strict_validator.CUSTOMER_RATIO_LIMIT - data_customers)
            need_non_cust = max(0.0, target_non_cust - data_non_customers)
        pool_non_cust_count = sum(1 for lib in unassigned_pool if not _is_customer_like_validator(lib))
        pool_non_cust_data = sum(float(getattr(lib, "contract_data_raw", 0) or 0) for lib in unassigned_pool if not _is_customer_like_validator(lib))
        logger.info(
            f"Lane {lane.lane_id} 稀释策略：客户={data_customers:.1f}G，非客户={data_non_customers:.1f}G，"
            f"需补非客户={need_non_cust:.1f}G，池中非客户={pool_non_cust_count}个/{pool_non_cust_data:.1f}G"
        )
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
                    for lib in removed_cust:
                        if lib in working_libs:
                            working_libs.remove(lib)
                        if lib in customers:
                            customers.remove(lib)
                    removed_libs.extend(removed_cust)
                    data_customers -= freed
                    data_total -= freed
                    room += freed
                    data_non_customers = _total_data(non_customers)
                    target_non_cust = max(data_non_customers, data_customers / strict_validator.CUSTOMER_RATIO_LIMIT - data_customers)
                    need_non_cust = max(0.0, target_non_cust - data_non_customers)
            picked = _pick_from_pool(unassigned_pool, lambda x: not _is_customer_like_validator(x), need_non_cust, data_total)
        logger.info(f"Lane {lane.lane_id} 实际挑选非客户文库: {len(picked)}个")
        if picked:
            cand_libs = working_libs + picked
            re_res = _validate_lane_with_latest_index(
                validator=strict_validator,
                libraries=cand_libs, lane_id=lane.lane_id,
                machine_type=machine_type, metadata=metadata_after,
            )
            if re_res.is_valid:
                added_libs.extend(picked)
                for lib in picked:
                    if lib in unassigned_pool:
                        unassigned_pool.remove(lib)
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
                    added_libs.extend([lib for lib in cand_libs if lib not in working_libs])
                    for lib in picked_cust:
                        if lib in unassigned_pool:
                            unassigned_pool.remove(lib)
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
                dropped = [lib for lib in working_libs if lib not in libs_10bp]
                removed_libs.extend(dropped)
                added_libs.extend(picked)
                for lib in picked:
                    if lib in unassigned_pool:
                        unassigned_pool.remove(lib)
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
                dropped = [lib for lib in working_libs if lib not in libs_non_10bp]
                removed_libs.extend(dropped)
                added_libs.extend(picked)
                for lib in picked:
                    if lib in unassigned_pool:
                        unassigned_pool.remove(lib)
                working_libs = cand_libs
                metadata_after = meta_non10
                logger.info(
                    f"Lane {lane.lane_id} 10bp占比矫正：转纯非10bp（补{len(picked)}个非10bp文库）后通过校验，移出{len(dropped)}个10bp文库"
                )
                return working_libs, removed_libs
        if data_10bp > 0 and data_non_10bp > 0:
            ratio = data_10bp / (data_10bp + data_non_10bp)
            if ratio < strict_validator.INDEX_10BP_RATIO_MIN:
                need_extra_10bp = max(0.0, strict_validator.INDEX_10BP_RATIO_MIN * data_non_10bp / (1 - strict_validator.INDEX_10BP_RATIO_MIN) - data_10bp)
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
                        for lib in picked:
                            if lib in unassigned_pool:
                                unassigned_pool.remove(lib)
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
    for lane in lanes:
        libs = list(lane.libraries or [])
        if not libs:
            continue
        lane_loading_concentration, lane_concentration_rule = _resolve_lane_loading_concentration(libs)
        logger.info(
            f"{tag} Lane {lane.lane_id} 排机浓度规则命中: {lane_concentration_rule}, "
            f"lsjnd={'' if lane_loading_concentration is None else format(lane_loading_concentration, '.3f')}"
        )
        runid = runid_by_lane.get(lane.lane_id)
        lane_balance_data = None
        if isinstance(lane.metadata, dict):
            lane_balance_data = lane.metadata.get("wkbalancedata")
            if lane_balance_data is None:
                lane_balance_data = lane.metadata.get("wkadd_balance_data")
            if lane_balance_data is None:
                lane_balance_data = lane.metadata.get("required_balance_data_gb")
        lane_balance_data_value = None
        if lane_balance_data is not None:
            lane_balance_data_value = round(float(lane_balance_data), 3)

        # 排机脚本只负责产出成Lane结果，预测统一交给 prediction_delivery 处理。
        predicted_orders: List[Optional[float]] = [None] * len(libs)
        predicted_outputs: List[Optional[float]] = [None] * len(libs)

        for idx, lib in enumerate(libs):
            pred_order = predicted_orders[idx]
            pred_output = predicted_outputs[idx]
            contract = float(lib.contract_data_raw or 0.0)
            loutput = loutput_by_origrec.get(lib.origrec)
            add_test_rule_result = _apply_add_test_output_rate_rule(
                lib=lib,
                ai_predicted_order=pred_order,
                ai_predicted_output=pred_output,
                contract_data=contract,
            )
            selected_order_raw = add_test_rule_result["selected_order"]
            selected_order = _apply_lane_orderdata_floor(selected_order_raw)
            selected_output = add_test_rule_result["selected_output"]

            rows.append(
                {
                    "origrec": lib.origrec,
                    "runid": runid,
                    "lane_id": lane.lane_id,
                    "lsjnd": (
                        None
                        if lane_loading_concentration is None
                        else round(float(lane_loading_concentration), 3)
                    ),
                    "wkcontractdata": contract,
                    "wkbalancedata": lane_balance_data_value,
                    "predicted_lorderdata": None if selected_order is None else round(float(selected_order), 3),
                    "lai_output": None if selected_output is None else round(float(selected_output), 3),
                    "ai_predicted_lorderdata": None if pred_order is None else round(float(pred_order), 3),
                    "ai_predicted_loutput": None if pred_output is None else round(float(pred_output), 3),
                    "add_test_rule_applied": add_test_rule_result["applied"],
                    "add_test_rule_reason": add_test_rule_result["rule_reason"],
                    "qpcr_within_15pct": add_test_rule_result["qpcr_within_15pct"],
                    "qpcr_deviation_ratio": add_test_rule_result["qpcr_deviation_ratio"],
                    "historical_based_lorderdata": (
                        None
                        if add_test_rule_result["historical_based_order"] is None
                        else round(float(add_test_rule_result["historical_based_order"]), 3)
                    ),
                    "wklistqpcr": add_test_rule_result["wklistqpcr"],
                    "wklastorderdata": add_test_rule_result["wklastorderdata"],
                    "wklastoutput": add_test_rule_result["wklastoutput"],
                    "wklastoutrate": add_test_rule_result["wklastoutrate"],
                    "loutput": loutput,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["lsjnd"] = pd.to_numeric(df["lsjnd"], errors="coerce").round(3)
        df["wkbalancedata"] = pd.to_numeric(df["wkbalancedata"], errors="coerce").round(3)
        df["predicted_lorderdata"] = pd.to_numeric(df["predicted_lorderdata"], errors="coerce").round(3)
        df["lai_output"] = pd.to_numeric(df["lai_output"], errors="coerce").round(3)
    return df
def _build_detail_output(
    df_raw: pd.DataFrame,
    pred_df: pd.DataFrame,
    output_path: Path,
    ai_schedulable_keys: Optional[Set[str]] = None,
    lanes_with_split: Optional[Set[str]] = None,
) -> None:
    """生成明细输出文件"""
    def _ensure_object_column(df: pd.DataFrame, column_name: str) -> None:
        """在写入字符串前显式转为object列，避免pandas类型告警。"""
        if column_name in df.columns:
            df[column_name] = df[column_name].astype(object)

    target_machine_types = _get_lane_loading_target_machine_texts()
    target_test_no = _get_lane_loading_target_testno_text()
    merged = df_raw.copy()
    if "origrec_key" not in merged.columns:
        merged["origrec_key"] = _build_origrec_key(merged)

    # 默认补齐预测相关字段，保证输出结构稳定
    merged["runid"] = pd.NA
    merged["laneid"] = pd.NA
    merged["lsjnd"] = pd.NA
    if "wkbalancedata" not in merged.columns:
        merged["wkbalancedata"] = pd.NA
    merged["predicted_lorderdata"] = pd.NA
    merged["lai_output"] = pd.NA

    if not pred_df.empty:
        pred_for_merge = pred_df.copy()
        pred_for_merge["origrec_key"] = pred_for_merge["origrec"].astype(str).str.strip()
        pred_for_merge = pred_for_merge[
            [
                "origrec_key",
                "runid",
                "lane_id",
                "lsjnd",
                "wkbalancedata",
                "predicted_lorderdata",
                "lai_output",
            ]
        ].copy()
        pred_for_merge.rename(columns={"lane_id": "laneid"}, inplace=True)

        merged = merged.drop(columns=["runid", "laneid", "lsjnd", "predicted_lorderdata", "lai_output"])
        merged = merged.merge(pred_for_merge, on="origrec_key", how="left", suffixes=("", "_pred"))
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

    # 仅对已成Lane的数据填充默认测序模式，未成Lane记录保持原值不改
    if "lcxms" not in merged.columns:
        merged["lcxms"] = pd.NA
    _ensure_object_column(merged, "lcxms")
    lane_assigned_mask = (
        merged["laneid"].notna()
        & ~merged["laneid"].astype(str).str.strip().isin({"", "nan", "None", "NONE", "null", "NULL"})
    )
    missing_lcxms_mask = (
        merged["lcxms"].isna()
        | merged["lcxms"].astype(str).str.strip().isin({"", "nan", "None", "NONE", "null", "NULL"})
    )
    merged.loc[lane_assigned_mask & missing_lcxms_mask, "lcxms"] = "3.6T-NEW"

    # AI排机次数：默认0，AI可排文库统一+1（无论是否成lane）
    if "aiarrangenumber" not in merged.columns:
        merged["aiarrangenumber"] = 0
    ai_arrange_series = pd.to_numeric(merged["aiarrangenumber"], errors="coerce").fillna(0).astype(int)
    ai_schedulable_keys = ai_schedulable_keys or set()
    if ai_schedulable_keys:
        ai_schedulable_mask = merged["origrec_key"].astype(str).isin(ai_schedulable_keys)
        ai_arrange_series.loc[ai_schedulable_mask] = ai_arrange_series.loc[ai_schedulable_mask] + 1
    merged["aiarrangenumber"] = ai_arrange_series

    # 将新生成的runid/laneid覆盖写回原始字段lrunid/llaneid，并移除runid/laneid输出列
    if "lrunid" not in merged.columns:
        merged["lrunid"] = pd.NA
    if "llaneid" not in merged.columns:
        merged["llaneid"] = pd.NA
    _ensure_object_column(merged, "lrunid")
    _ensure_object_column(merged, "llaneid")
    merged.loc[lane_assigned_mask, "lrunid"] = merged.loc[lane_assigned_mask, "runid"]
    merged.loc[lane_assigned_mask, "llaneid"] = merged.loc[lane_assigned_mask, "laneid"]

    # 工序+机型映射覆盖lsjfs：仅对已成Lane数据写回，未成Lane记录保持空值
    if "lsjfs" not in merged.columns:
        merged["lsjfs"] = pd.NA
    _ensure_object_column(merged, "lsjfs")
    if "wktestno" in merged.columns and "wkeqtype" in merged.columns:
        testno_norm = merged["wktestno"].map(_normalize_text_for_match)
        eqtype_norm = merged["wkeqtype"].map(_normalize_text_for_match)
        lsjfs_override_mask = (
            lane_assigned_mask
            &
            (testno_norm == target_test_no)
            & eqtype_norm.isin(target_machine_types)
        )
        merged.loc[lsjfs_override_mask, "lsjfs"] = "25B"

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

    # 对AI参与排机但未成lane的文库，将wkuser置空
    if "wkuser" not in merged.columns:
        merged["wkuser"] = pd.NA
    _ensure_object_column(merged, "wkuser")
    if ai_schedulable_keys:
        wkuser_clear_mask = merged["origrec_key"].astype(str).isin(ai_schedulable_keys) & (~lane_assigned_mask)
        merged.loc[wkuser_clear_mask, "wkuser"] = ""

    # 输出字段改名：预测结果按业务字段名输出
    # 注意：这里使用预测值覆盖输出中的 lorderdata / lai_output
    merged["lorderdata"] = pd.to_numeric(merged.get("predicted_lorderdata"), errors="coerce")
    merged["lai_output"] = pd.to_numeric(merged.get("lai_output"), errors="coerce")

    # 显式排除中间列runid/laneid及预测中间列，避免重复
    merged = merged.drop(columns=["runid", "laneid"], errors="ignore")
    merged = merged.drop(columns=["predicted_lorderdata"], errors="ignore")

    if output_path.exists():
        logger.info(f"明细文件已存在，将覆盖: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info(f"明细输出完成: {output_path}")


# ==================== 数据加载 ====================


def load_standardized_csv(data_file: str, limit: int | None = None) -> List[EnhancedLibraryInfo]:
    """从标准化CSV文件加载文库数据（训练数据格式）
    
    使用 EnhancedLibraryInfo.from_csv_row() 完成字段映射，
    支持 wk 前缀和非 wk 前缀两种列名格式。
    """
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    logger.info(f"从标准化CSV文件加载数据: {data_path}")
    df = pd.read_csv(data_path, nrows=limit)
    logger.info(f"读取 {len(df)} 行数据")
    
    libraries: List[EnhancedLibraryInfo] = []
    
    for idx, row in df.iterrows():
        row_dict = {k: (v if not pd.isna(v) else None) for k, v in row.to_dict().items()}
        try:
            lib = EnhancedLibraryInfo.from_csv_row(row_dict)
            # 设置机型
            lib.machine_type = _resolve_machine_type_enum_simple(lib.eq_type)
            # 保存origrec_key与AI可排标识，供主流程与明细规则使用
            lib._origrec_key = _safe_str(
                row_dict.get("wkorigrec")
                or row_dict.get("origrec")
                or row_dict.get("lane_unique_id")
                or row_dict.get("lane_unique")
                or row_dict.get("llaneid")
                or f"LIB_{idx}"
            )
            lib._aiavailable_raw = _safe_str(row_dict.get("aiavailable"), default="")
            # 保存V6需要但EnhancedLibraryInfo不支持的额外字段
            jkhj_val = row_dict.get("wkjkhj") or row_dict.get("jkhj")
            lib._jkhj_raw = str(jkhj_val) if jkhj_val else "诺禾自动"
            lib._last_qpcr_raw = _safe_float(row_dict.get("wklistqpcr"), default=None)
            lib._last_order_data_raw = _safe_float(
                row_dict.get("wklastorderdata", row_dict.get("wklastlorderdata")),
                default=None,
            )
            lib._last_output_raw = _safe_float(row_dict.get("wklastoutput"), default=None)
            lib._last_outrate_raw = _safe_float(row_dict.get("wklastoutrate"), default=None)
            # 保存测序模式相关原始字段，供拆分规则识别模式（1.0/非1.0）使用
            lib._lane_sj_mode_raw = _safe_str(row_dict.get("lsjfs"), default="")
            lib._current_seq_mode_raw = _safe_str(row_dict.get("lcxms"), default="")
            lib._last_cxms_raw = _safe_str(row_dict.get("llastcxms"), default="")
            libraries.append(lib)
        except Exception as e:
            logger.warning(f"行 {idx} 创建文库对象失败: {e}")
    
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
        max_special_library_types=3,
        max_special_library_data_gb=350.0,
        enable_index_check=True,
        enable_imbalance_check=True,
        enable_rule_checker=False,
        max_imbalance_types_per_lane=5,
        max_imbalance_ratio=0.35,
        enable_dedicated_imbalance_lane=False,
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
    
    from core.constraints.lane_validator import LaneValidator
    strict_validator = LaneValidator(strict_mode=True)
    logger.info("严格校验容量区间改为按统一配置表动态解析")

    disabled_plan = StrategyExecutionPlan()
    disabled_plan.enable_dedicated_imbalance_lane = False
    disabled_plan.enable_non_10bp_dedicated_lane = False
    disabled_plan.enable_backbone_reservation = False
    scheduler._strategy_plan = disabled_plan

    # ===== 混样排（Peak Size窗口内10bp+非10bp混排） =====
    # 不再先抽10bp专Lane：保留全部10bp文库参与混排，确保10bp>=40%
    dedicated_10bp_lanes: List[LaneAssignment] = []

    mixed_lanes, remaining_libraries = _extract_mixed_lanes_by_peak_window(
        libraries=libraries,
        validator=strict_validator,
        machine_type=MachineType.NOVA_X_25B,
        index_conflict_attempts_per_lane=100,
        other_failure_attempts_per_lane=200,
    )

    # 执行排机（剩余文库进入混样排机）
    if remaining_libraries:
        solution = scheduler.schedule(remaining_libraries, keep_failed_lanes=True)
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
    for lane in solution.lane_assignments:
        metadata = _build_lane_metadata_for_validator(lane.lane_id, lane.metadata)
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
                error_types = [e.rule_type.value for e in result.errors]
                warning_types = [w.rule_type.value for w in result.warnings]
                logger.warning(
                    f"Lane {lane.lane_id} 验证失败 - 错误: {error_types}, 警告: {warning_types}"
                )

    solution.lane_assignments = passed_lanes
    logger.info(f"验证完成：{len(passed_lanes)}条Lane通过验证")

    # 尝试增加Lane数量
    extra_lanes = _try_increase_lane_count(
        solution,
        strict_validator,
        max_new_lanes=3,
        index_conflict_attempts_per_lane=3,
        other_failure_attempts_per_lane=5,
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
        max_new_lanes=2,
        max_donations=80,
        index_conflict_max_trials=3,
        other_failure_max_trials=5,
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
        logger.info(f"\n合并{len(preset_lanes)}条预构建Lane到最终结果")
        solution.lane_assignments = preset_lanes + solution.lane_assignments
        logger.info(
            f"最终Lane总数: {len(solution.lane_assignments)} "
            f"(预构建Lane: {len(preset_lanes)}, "
            f"普通Lane: {len(solution.lane_assignments) - len(preset_lanes)})"
        )

    logger.info("脚本内置 Pooling 预测已停用，后续统一调用 prediction_delivery")
    stats = analyze_solution(solution)
    return stats, solution


def _build_output_path(data_path: Path, output_dir: Path, mode: str) -> Path:
    """根据运行模式生成输出文件路径。"""
    suffix = "_lane_output_v6.csv" if mode == "arrange" else "_pooling_output_v6.csv"
    return output_dir / f"{data_path.stem}{suffix}"


def _run_prediction_delivery(input_data: Union[Path, pd.DataFrame], output_path: Path) -> pd.DataFrame:
    """统一调用 prediction_delivery 执行 Pooling 预测。"""
    if output_path.exists():
        logger.info(f"预测输出文件已存在，将覆盖: {output_path}")
    logger.info(f"调用 prediction_delivery，模型目录: {MODELS_DIR}")
    return predict_pooling(input_data=input_data, output_file=output_path)


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
        logger.info("arrange_library: 仅执行 Pooling 预测")
        logger.info("=" * 80)
        _run_prediction_delivery(input_data=data_path, output_path=output_path)
        logger.info(f"Pooling 预测完成，输出文件: {output_path}")
        return output_path

    # ===== 全流程排机模式 =====
    df_raw = pd.read_csv(data_path)
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
    ai_schedulable_keys: Set[str] = set()
    for lib in libraries:
        if _is_yes_value(getattr(lib, "_aiavailable_raw", "")):
            ai_schedulable_libraries.append(lib)
            ai_schedulable_keys.add(_safe_str(getattr(lib, "_origrec_key", getattr(lib, "origrec", ""))))
        else:
            non_ai_libraries.append(lib)
    logger.info(
        "AI可排文库筛选完成: 可排={}，不可排={}".format(
            len(ai_schedulable_libraries), len(non_ai_libraries)
        )
    )
    if not ai_schedulable_libraries:
        logger.warning("无AI可排文库（aiavailable!=yes），本次不执行排机，仅输出明细规则结果")

    # ===== 步骤1: 处理包Lane文库 =====
    logger.info("\n" + "=" * 80)
    logger.info("步骤1: 处理包Lane文库（wkbaleno字段有值的文库）")
    logger.info("=" * 80)
    
    package_lanes = []
    package_libs = []
    normal_libs = []
    
    for lib in ai_schedulable_libraries:
        baleno = getattr(lib, 'package_lane_number', None) or getattr(lib, 'baleno', None)
        if baleno and str(baleno).strip():
            lib.package_lane_number = str(baleno).strip()
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
                lane_assignment = LaneAssignment(
                    lane_id=lane_result.lane_id,
                    machine_id=f"M_{lane_result.lane_id}",
                    machine_type=_resolve_machine_type_enum_simple(run.machine_type),
                    libraries=lane_result.libraries,
                    total_data_gb=lane_result.total_data_gb,
                    pooling_coefficients=lane_result.pooling_coefficients,
                    metadata={'is_package_lane': True, 'package_id': lane_result.package_id}
                )
                package_lanes.append(lane_assignment)
        
        normal_libs.extend(package_result.remaining_libraries)
        
        logger.info(f"\n包Lane处理完成，形成{len(package_lanes)}条包Lane")
        logger.info(f"剩余{len(normal_libs)}个文库进入普通排机流程")
    
    # ===== 步骤2: 处理普通文库（包括包Lane处理失败的文库） =====
    logger.info("\n" + "=" * 80)
    logger.info("步骤2: 处理普通文库（使用GreedyLaneScheduler）")
    logger.info("=" * 80)

    if ai_schedulable_libraries:
        random.seed(42)
        stats, solution = test_with_model(
            deepcopy(normal_libs), existing_lanes=package_lanes
        )
    else:
        from types import SimpleNamespace
        stats = {}
        solution = SimpleNamespace(lane_assignments=[], unassigned_libraries=[])

    # 收集预测结果
    pred_df = _collect_prediction_rows(
        solution.lane_assignments, loutput_by_origrec, "arrange"
    )

    lanes_with_split = _collect_lanes_with_split(solution.lane_assignments)

    # 输出明细
    _build_detail_output(
        df_raw=df_raw,
        pred_df=pred_df,
        output_path=output_path,
        ai_schedulable_keys=ai_schedulable_keys,
        lanes_with_split=lanes_with_split,
    )

    logger.info("\n" + "=" * 80)
    logger.info("步骤3: 调用 prediction_delivery 执行 Pooling 预测")
    logger.info("=" * 80)
    prediction_df = _run_prediction_delivery(input_data=output_path, output_path=output_path)
    logger.info(
        "prediction_delivery 预测完成: 记录数={}, 平均下单量={:.3f}G, 平均产出量={:.3f}G".format(
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
