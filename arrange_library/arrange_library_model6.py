"""
端到端排机流程测试 - 排机与 Pooling 预测
创建时间：2026-04-10 16:06:41
更新时间：2026-04-14 13:10:00

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
import signal
import sys
from dataclasses import dataclass
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
    LaneValidationResult,
    ValidationRuleType,
    ValidationError,
    ValidationSeverity,
)
from arrange_library.core.constraints.index_validator_verified import IndexConflictValidator as _IndexConflictValidator

# 模块级单例，避免在 _attempt_build_lane_from_pool 等高频函数中反复初始化
_MODULE_IDX_VALIDATOR = _IndexConflictValidator()
_AUTO_LANE_SERIAL_COUNTERS: Dict[Tuple[str, str], int] = {}
from arrange_library.core.data import load_libraries_from_csv
from arrange_library.core.preprocessing.base_imbalance_handler import BaseImbalanceHandler
from arrange_library.core.preprocessing.library_splitter import LibrarySplitter
from arrange_library.core.preprocessing.rule_constrained_strategy_planner import StrategyExecutionPlan
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
    "10x_cellranger_atac_indexset",
    "10x_cellranger_atac",
}
SCHEDULING_MAX_TARGET_CAP_GB = 1100.0
SCHEDULING_MAX_EFFECTIVE_CAP_GB = 1105.0
SCHEDULING_CAP_RULE_CODES: Set[str] = {
    "tj_1595_standard_pe150_25b",
    "tj_1595_standard_pe150_25b_other",
}
DEFAULT_INDEX_CONFLICT_ATTEMPTS = 10
DEFAULT_OTHER_FAILURE_ATTEMPTS = 20
DEFAULT_EX_RESCUE_MAX_NEW_LANES = 2
DEFAULT_RB_RESCUE_MAX_NEW_LANES = 1
INDEX_RULE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "merge_deal" / "config"
BALANCE_LIBRARY_CONFIG_PATH = Path(__file__).resolve().parent / "AI排机-平衡文库.csv"
BALANCE_LIBRARY_MARKER_COLUMN = "_is_ai_balance_library"
PACKAGE_LANE_TARGET_GB = 1000.0
PACKAGE_LANE_TOLERANCE_GB = 0.01
PACKAGE_LANE_MIN_GB = PACKAGE_LANE_TARGET_GB - PACKAGE_LANE_TOLERANCE_GB
PACKAGE_LANE_MAX_GB = PACKAGE_LANE_TARGET_GB + PACKAGE_LANE_TOLERANCE_GB
PACKAGE_LANE_MIN_INDEX_PAIRS = 5
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


def _normalize_seq_strategy_keyword(value: Any) -> str:
    """统一测序策略匹配口径。"""
    return _normalize_text_for_match(value).replace("BP", "").replace(" ", "")


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
    try:
        imbalance_mix_valid, imbalance_mix_reason = _validate_lane_57_mix_rules(
            libraries,
            enforce_total_limit=False,
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
                    "{10x_cellranger_atac_indexset,10x_cellranger_atac}同Lane且不得与其他类型混排"
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
    metadata = _build_lane_metadata_for_validator(lane_id, lane_metadata)
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
    return sum(lib.get_data_amount_gb() for lib in libraries)


def _validate_lane_57_mix_rules(
    libraries: List[EnhancedLibraryInfo],
    enforce_total_limit: bool = False,
) -> Tuple[bool, str]:
    """按 BaseImbalanceHandler 校验 57 组合混排规则。"""
    return _BASE_IMBALANCE_HANDLER.check_mix_compatibility(
        libraries,
        enforce_total_limit=enforce_total_limit,
    )


def _count_library_index_pairs(lib: EnhancedLibraryInfo) -> int:
    """统计单个文库的Index对数。"""
    index_seq = str(getattr(lib, "index_seq", "") or "").strip()
    if not index_seq:
        return 0
    return len([item for item in index_seq.split(",") if str(item).strip()])


def _count_lane_index_pairs(libraries: List[EnhancedLibraryInfo]) -> int:
    """统计整条Lane的Index对总数。"""
    return sum(_count_library_index_pairs(lib) for lib in libraries)


def _get_package_lane_number_from_lane(lane: LaneAssignment) -> str:
    """提取Lane对应的包Lane编号。"""
    package_id = _safe_str(getattr(lane, "metadata", {}).get("package_id", ""), default="")
    if package_id:
        return package_id
    for lib in getattr(lane, "libraries", []) or []:
        baleno = _safe_str(
            getattr(lib, "package_lane_number", None) or getattr(lib, "baleno", None),
            default="",
        )
        if baleno:
            return baleno
    return ""


def _is_package_lane_assignment(lane: LaneAssignment) -> bool:
    """判断当前Lane是否为预构建包Lane。"""
    if bool(getattr(lane, "metadata", {}).get("is_package_lane")):
        return True
    return bool(_get_package_lane_number_from_lane(lane))


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
    if total_contract_data < PACKAGE_LANE_MIN_GB or total_contract_data > PACKAGE_LANE_MAX_GB:
        errors.append(
            f"包Lane {package_lane_number} 合同数据量不满足1000G±0.01G: 当前{total_contract_data:.3f}G"
        )
    return errors


def _validate_final_package_lanes(solution: Any) -> None:
    """排机完成后复核所有包Lane规则。"""
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

            if not (
                is_allowed_multi_pkg_split
                and family_id
                and len(expected_package_ids) > 1
                and package_lane_number in expected_package_ids
                and lane_package_lane_number == package_lane_number
                and int(getattr(lib, "total_fragments", 0) or 0) == len(expected_package_ids)
            ):
                errors.append(
                    "包Lane {} 文库 {} 被拆分，但不属于允许的“多包Lane编号专用拆分”".format(
                        package_lane_number,
                        _safe_str(getattr(lib, "origrec", ""), default="UNKNOWN"),
                    )
                )
                continue

            if lane_id and lane_id not in lane_purity_checked:
                lane_purity_checked.add(lane_id)
                non_package_lane_libs = [
                    lane_lib
                    for lane_lib in getattr(lane, "libraries", []) or []
                    if not _safe_str(
                        getattr(lane_lib, "package_lane_number", None) or getattr(lane_lib, "baleno", None),
                        default="",
                    )
                ]
                if non_package_lane_libs:
                    errors.append(
                        f"多包Lane拆分目标Lane {lane_id} 混入了{len(non_package_lane_libs)}个非包Lane文库"
                    )

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
    """散样混排优先级：临检和SJ > YC > 其他。"""
    data_type = str(getattr(lib, "data_type", "") or "").strip()
    try:
        is_clinical = bool(lib.is_clinical_by_code())
    except Exception:
        is_clinical = False
    try:
        is_sj = bool(lib.is_s_level_customer())
    except Exception:
        is_sj = False
    try:
        is_yc = bool(lib.is_yc_library())
    except Exception:
        is_yc = False

    if data_type in {"临检", "SJ"} or is_clinical or is_sj:
        return 0
    if data_type == "YC" or is_yc:
        return 1
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
    """按分层补位规则过滤可参与当前阶段成Lane的文库。"""
    max_allowed_rank = _resolve_priority_gate_rank(
        libraries=libraries,
        machine_type=machine_type,
        lane_id=lane_id,
        lane_metadata=lane_metadata,
    )
    if max_allowed_rank is None:
        return []
    filtered = [
        lib for lib in libraries
        if _get_scattered_mix_priority_rank(lib) <= max_allowed_rank
    ]
    if emit_log and len(filtered) != len(libraries):
        logger.info(
            "全局优先级硬约束生效: stage={}, 当前允许{}参与，暂缓较低优先级文库{}个".format(
                stage_name or "unknown",
                _get_priority_gate_label(max_allowed_rank),
                len(libraries) - len(filtered),
            )
        )
    return filtered


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
    """在多个池之间统一执行分层补位版全局优先级门禁。"""
    max_allowed_rank = _resolve_priority_gate_rank(
        libraries=primary_pool + secondary_pool,
        machine_type=machine_type,
        lane_id=lane_id,
        lane_metadata=lane_metadata,
    )
    if max_allowed_rank is None:
        return [], [], None
    filtered_primary = [
        lib for lib in primary_pool
        if _get_scattered_mix_priority_rank(lib) <= max_allowed_rank
    ]
    filtered_secondary = [
        lib for lib in secondary_pool
        if _get_scattered_mix_priority_rank(lib) <= max_allowed_rank
    ]
    if emit_log and (
        len(filtered_primary) != len(primary_pool)
        or len(filtered_secondary) != len(secondary_pool)
    ):
        logger.info(
            "全局优先级硬约束生效: stage={}, 当前允许{}参与，主池暂缓{}个、辅池暂缓{}个较低优先级文库".format(
                stage_name or "unknown",
                _get_priority_gate_label(max_allowed_rank),
                len(primary_pool) - len(filtered_primary),
                len(secondary_pool) - len(filtered_secondary),
            )
        )
    return filtered_primary, filtered_secondary, max_allowed_rank


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
    raw_value = getattr(lib, "_delete_date_raw", None)
    if raw_value in (None, ""):
        raw_value = getattr(lib, "deduction_time", None)
    if raw_value in (None, ""):
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


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

    board_sorted = _sort_by_board_preference_for_scattered_mix(libraries)
    board_order = {id(lib): idx for idx, lib in enumerate(board_sorted)}
    return sorted(
        libraries,
        key=lambda lib: (
            _get_scattered_mix_priority_rank(lib),
            _get_scattered_mix_delete_date_sort_value(lib),
            board_order.get(id(lib), len(board_order)),
            -lib.get_data_amount_gb(),
        ),
    )


def _sort_remaining_for_lane_seed(
    libraries: List[EnhancedLibraryInfo],
    seed_lib: EnhancedLibraryInfo,
) -> List[EnhancedLibraryInfo]:
    """单条Lane内优先吞同级高优先级文库，降低临检/SJ/YC被打散概率。"""
    if not libraries:
        return libraries

    seed_rank = _get_scattered_mix_priority_rank(seed_lib)
    base_sorted = _sort_remaining_for_scattered_mix_lane(libraries)
    base_order = {id(lib): idx for idx, lib in enumerate(base_sorted)}
    return sorted(
        libraries,
        key=lambda lib: (
            0 if _get_scattered_mix_priority_rank(lib) == seed_rank else 1,
            _get_scattered_mix_priority_rank(lib),
            _get_scattered_mix_delete_date_sort_value(lib),
            base_order.get(id(lib), len(base_order)),
            -lib.get_data_amount_gb(),
        ),
    )


def _build_scattered_mix_candidate_order(
    libraries: List[EnhancedLibraryInfo],
) -> List[EnhancedLibraryInfo]:
    """为散样混排构建稳定候选顺序。"""
    if not libraries:
        return libraries

    ordered = _sort_remaining_for_scattered_mix_lane(libraries)
    seed_lib = ordered[0]
    seen: Set[int] = set()
    prioritized: List[EnhancedLibraryInfo] = []
    for lib in _sort_remaining_for_lane_seed(ordered, seed_lib):
        object_id = id(lib)
        if object_id in seen:
            continue
        seen.add(object_id)
        prioritized.append(lib)
    return prioritized


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
    while (
        index_conflict_retry_count < index_conflict_attempts
        and other_failure_retry_count < other_failure_attempts
    ):
        attempt_idx += 1
        if prioritize_scattered_mix:
            candidates = _build_scattered_mix_candidate_order(active_pool)
        else:
            candidates = list(active_pool)
            random.shuffle(candidates)
        selected: List[EnhancedLibraryInfo] = []
        # 与 selected 平行维护的预解析索引缓存，避免在 validate_new_lib_quick_with_cache
        # 内部对同一文库反复调用 _parse_library_indices（逐个候选检查时累计百万次调用）
        selected_idx_cache: List = []
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
            imbalance_mix_valid, _ = _validate_lane_57_mix_rules(
                trial_libs,
                enforce_total_limit=False,
            )
            if not imbalance_mix_valid:
                continue
            # 带缓存增量检查：new_lib 的 index 解析结果同步写入 selected_idx_cache，
            # 后续再判断其他候选时不再重复解析 selected 中已有文库的索引
            idx_valid, lib_indices = _idx_validator.validate_new_lib_quick_with_cache(
                selected_idx_cache, lib
            )
            if not idx_valid:
                continue
            selected.append(lib)
            selected_idx_cache.append(lib_indices)
            total += data
            if random_target is None and total >= trial_min_allowed:
                if prioritize_scattered_mix:
                    # 散样混排优先聚拢高优先级文库，达到成Lane下限后不再继续吞入更低优先级文库。
                    random_target = float(trial_min_allowed)
                else:
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


def _attempt_build_rescue_lane_from_pool(
    pool: List[EnhancedLibraryInfo],
    validator,
    machine_type: MachineType,
    lane_id_prefix: str,
    lane_serial: Optional[int] = None,
    index_conflict_attempts: int = DEFAULT_INDEX_CONFLICT_ATTEMPTS,
    other_failure_attempts: int = DEFAULT_OTHER_FAILURE_ATTEMPTS,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[LaneAssignment | None, List[EnhancedLibraryInfo]]:
    """仅在 RB/EX 补Lane阶段启用的窄范围回退。

    先按临检/SJ/YC优先顺序尝试；失败后只退到非临检/SJ池，
    避免为了救援Lane打散更高优先级文库。
    """
    if not pool:
        return None, []

    seen_variants: Set[Tuple[bool, Tuple[int, ...]]] = set()
    non_clinical_pool = [
        lib for lib in pool
        if _get_scattered_mix_priority_rank(lib) >= 1
    ]
    other_only_pool = [
        lib for lib in non_clinical_pool
        if _get_scattered_mix_priority_rank(lib) == 2
    ]

    variants: List[Tuple[str, List[EnhancedLibraryInfo], bool]] = [
        ("full_priority", list(pool), True),
    ]
    if non_clinical_pool and len(non_clinical_pool) < len(pool):
        variants.append(("non_clinical_priority", list(non_clinical_pool), True))
    if non_clinical_pool:
        variants.append(("non_clinical_relaxed", list(non_clinical_pool), False))
    if other_only_pool and len(other_only_pool) < len(non_clinical_pool):
        variants.append(("other_only_relaxed", list(other_only_pool), False))

    for variant_name, candidate_pool, prioritize_scattered_mix in variants:
        key = (prioritize_scattered_mix, tuple(sorted(id(lib) for lib in candidate_pool)))
        if key in seen_variants:
            continue
        seen_variants.add(key)
        logger.info(
            "补Lane尝试: variant={}, lane_prefix={}, pool_size={}, pool_data={:.3f}G, prioritize_scattered_mix={}".format(
                variant_name,
                lane_id_prefix,
                len(candidate_pool),
                sum(lib.get_data_amount_gb() for lib in candidate_pool),
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
        )
        if lane:
            return lane, used
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
    while (
        index_conflict_retry_count < index_conflict_attempts
        and other_failure_retry_count < other_failure_attempts
    ):
        selected: List[EnhancedLibraryInfo] = []
        # 与 selected 平行维护的预解析索引缓存，同 _attempt_build_lane_from_pool 的优化逻辑
        selected_idx_cache: List = []
        total = 0.0
        random_target: Optional[float] = None

        def _ordered_candidates(pool: List[EnhancedLibraryInfo], prefer_balanced: bool) -> List[EnhancedLibraryInfo]:
            filtered_pool = [lib for lib in pool if match_fn(lib)] if match_fn is not None else list(pool)
            return sorted(
                filtered_pool,
                key=lambda lib: (
                    0 if (_BASE_IMBALANCE_HANDLER.is_imbalance_library(lib) is (not prefer_balanced)) else 1,
                    -float(getattr(lib, "contract_data_raw", 0.0) or 0.0),
                    str(getattr(lib, "origrec", "") or ""),
                ),
            )

        candidate_buckets = [
            _ordered_candidates(active_primary_pool, prefer_balanced=False),
            _ordered_candidates(active_secondary_pool, prefer_balanced=True),
        ]

        for candidates in candidate_buckets:
            for lib in candidates:
                if lib in selected:
                    continue
                data = lib.get_data_amount_gb()
                trial_libs = selected + [lib]
                trial_min_allowed, trial_max_allowed = _resolve_lane_capacity_limits(
                    libraries=trial_libs,
                    machine_type=machine_type,
                )
                if total + data > trial_max_allowed:
                    continue
                ss_valid, _, _ = _validate_lane_special_split_rule(trial_libs)
                if not ss_valid:
                    continue
                imbalance_mix_valid, _ = _validate_lane_57_mix_rules(
                    trial_libs,
                    enforce_total_limit=False,
                )
                if not imbalance_mix_valid:
                    continue
                # 带缓存增量检查，避免对 selected 中已有文库的索引反复解析
                idx_valid, lib_indices = idx_validator.validate_new_lib_quick_with_cache(
                    selected_idx_cache, lib
                )
                if not idx_valid:
                    continue
                selected.append(lib)
                selected_idx_cache.append(lib_indices)
                total += data
                if random_target is None and total >= trial_min_allowed:
                    random_target = random.uniform(trial_min_allowed, trial_max_allowed)
                if random_target is not None and total >= random_target:
                    break
            if random_target is not None and total >= random_target:
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
        result = _validate_lane_state(validator, lane, lane.libraries)
        if result.is_valid:
            return lane, selected
        if _is_index_conflict_only(result):
            index_conflict_retry_count += 1
        else:
            other_failure_retry_count += 1

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
    detail_output_key = _safe_str(
        getattr(lib, "_detail_output_key", None),
        default="",
    )
    if detail_output_key:
        return detail_output_key

    if _is_split_library(lib):
        for attr_name in ("fragment_id", "wkaidbid", "aidbid"):
            candidate = _safe_str(getattr(lib, attr_name, None), default="")
            if candidate:
                return candidate

    origrec_key = _safe_str(
        getattr(lib, "_origrec_key", getattr(lib, "origrec", "")),
        default="",
    )
    if origrec_key:
        return origrec_key
    return str(id(lib))


def _get_library_source_origrec_key(lib: EnhancedLibraryInfo) -> str:
    """获取文库对应原始输入行的归属键。"""
    source_key = _safe_str(
        getattr(lib, "_source_origrec_key", None) or getattr(lib, "_origrec_key", None),
        default="",
    )
    if source_key:
        return source_key
    return _safe_str(getattr(lib, "origrec", ""), default="")


def _get_library_detail_output_key(lib: EnhancedLibraryInfo) -> str:
    """获取明细输出按子文库展开时使用的稳定键。"""
    return _get_library_identity_key(lib)


def _is_ai_balance_library(lib: Any) -> bool:
    """判断是否为排机后新增的AI平衡文库。"""
    return bool(getattr(lib, BALANCE_LIBRARY_MARKER_COLUMN, False))


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


def _is_explicit_dedicated_imbalance_lane(lane: LaneAssignment) -> bool:
    """判断lane是否被明确标记为碱基不均衡专用lane。"""
    lane_id = _safe_str(getattr(lane, "lane_id", None), default="")
    if lane_id.startswith("DL_"):
        return True
    metadata = getattr(lane, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("is_dedicated_imbalance_lane"):
        return True
    return False


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
    if isinstance(lane_metadata, dict) and bool(lane_metadata.get("is_package_lane")):
        return True
    for lib in libraries:
        package_lane_number = _safe_str(
            getattr(lib, "package_lane_number", None) or getattr(lib, "baleno", None),
            default="",
        )
        if package_lane_number:
            return True
    return False


def _resolve_lane_balance_ratio_from_libraries(libraries: List[EnhancedLibraryInfo]) -> float:
    """按 lane 内普通文库解析平衡文库比例。"""
    ratio = 0.0
    for lib in libraries:
        if _is_ai_balance_library(lib):
            continue
        group_id = _BASE_IMBALANCE_HANDLER.identify_imbalance_type(lib)
        if not group_id:
            continue
        group_info = _BASE_IMBALANCE_HANDLER.get_group_info(group_id)
        if not group_info:
            continue
        ratio = max(ratio, float(getattr(group_info, "phix_ratio", 0.0) or 0.0))
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
    if metadata.get("materialized_balance_library"):
        return {"applied": False}
    if any(_is_ai_balance_library(lib) for lib in libraries):
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

    is_dedicated = lane_id.startswith("DL_") or bool(metadata.get("is_dedicated_imbalance_lane"))
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
        selection.effective_min_gb = max(PACKAGE_LANE_MIN - reserve_gb, 0.0)
        selection.effective_max_gb = max(PACKAGE_LANE_MAX - reserve_gb, 0.0)
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


def _resolve_lane_balance_data_gb(lane: LaneAssignment) -> float:
    """确定lane需要补充的平衡文库量。"""
    explicit_value = _get_lane_explicit_balance_data(lane)
    if _is_package_lane_assignment(lane):
        return round(explicit_value, 3) if explicit_value > 0 else 0.0
    if not _is_explicit_dedicated_imbalance_lane(lane):
        return 0.0
    ratio = _resolve_lane_balance_ratio(lane)
    if ratio <= 0:
        return 0.0
    if explicit_value > 0:
        return round(explicit_value, 3)
    lane_capacity = _safe_float(getattr(lane, "lane_capacity_gb", None), default=0.0)
    if lane_capacity <= 0:
        lane_capacity = _lane_capacity_for_machine(getattr(lane, "machine_type", MachineType.NOVA_X_25B))
    return round(lane_capacity * ratio, 3)


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
    conflict_ids: Set[str] = set()
    for conflict in _validate_index_conflicts_latest(list(lane_libraries) + [candidate]):
        if conflict.record_id_1 == candidate.origrec:
            conflict_ids.add(conflict.record_id_2)
        elif conflict.record_id_2 == candidate.origrec:
            conflict_ids.add(conflict.record_id_1)
    return [
        lib
        for lib in lane_libraries
        if getattr(lib, "origrec", "") in conflict_ids and _is_replaceable_normal_library(lib)
    ]


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
        result = _validate_lane_state(
            validator, lane, trial_libs,
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
            if not _validate_lane_state(
                validator, current_lane, current_trial_libs,
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
        return _validate_lane_state(
            validator, lane, libs,
            balance_already_in_libs=True,
            skip_peak_size=is_dedicated,
        )

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
    if any(_is_ai_balance_library(lib) for lib in list(getattr(lane, "libraries", []) or [])):
        return False

    balance_amount = _resolve_lane_balance_data_gb(lane)
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
        candidate_balance_lib = _create_balance_library_from_template(lane, template, balance_amount)
        trimmed_result = _trim_lane_for_balance_capacity(
            lane=lane,
            working_libs=original_libs,
            candidate_balance_lib=candidate_balance_lib,
            validator=validator,
        )
        if trimmed_result is not None:
            trimmed_libs, removed_libs = trimmed_result
            unassigned_pool.extend(removed_libs)
            lane.libraries = trimmed_libs
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
        working_libs = [lib for lib in original_libs if lib not in conflicting_libs]

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
            for lib in added_libs:
                if lib in unassigned_pool:
                    unassigned_pool.remove(lib)
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


def _materialize_balance_libraries_for_solution(solution: Any) -> Dict[str, int]:
    """对最终成lane结果补真实平衡文库。"""
    from arrange_library.core.constraints.lane_validator import LaneValidator

    validator = LaneValidator(strict_mode=True)
    unassigned_pool = list(getattr(solution, "unassigned_libraries", []) or [])
    lanes = list(getattr(solution, "lane_assignments", []) or [])

    success_count = 0
    required_count = 0
    for lane in lanes:
        if _resolve_lane_balance_data_gb(lane) > 0:
            required_count += 1
        if _materialize_balance_library_for_lane(
            lane=lane,
            all_lanes=lanes,
            unassigned_pool=unassigned_pool,
            validator=validator,
        ):
            success_count += 1

    solution.unassigned_libraries = unassigned_pool
    return {
        "required_lanes": required_count,
        "success_lanes": success_count,
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


def _enforce_global_priority_hard_constraint(
    solution: Any,
    validator: Any,
) -> Dict[str, int]:
    """对最终结果做全局优先级收口，按“高优先级先独立成Lane，不足再放开补位”执行。"""
    unassigned = list(getattr(solution, "unassigned_libraries", []) or [])
    if not unassigned:
        return {"adjusted_lanes": 0, "removed_lanes": 0, "deferred_libraries": 0}

    adjusted_lanes = 0
    removed_lanes = 0
    deferred_libraries = 0
    kept_lanes: List[LaneAssignment] = []

    for lane in getattr(solution, "lane_assignments", []) or []:
        if _is_package_lane_assignment(lane):
            kept_lanes.append(lane)
            continue

        lane_machine_type = lane.machine_type if lane.machine_type else MachineType.NOVA_X_25B
        max_allowed_rank = _resolve_priority_gate_rank(
            libraries=unassigned,
            machine_type=lane_machine_type.value,
            lane_id=lane.lane_id,
            lane_metadata=lane.metadata,
        )
        if max_allowed_rank is None:
            kept_lanes.append(lane)
            continue

        kept_libs = [
            lib for lib in list(lane.libraries or [])
            if _get_scattered_mix_priority_rank(lib) <= max_allowed_rank
        ]
        removed_libs = [
            lib for lib in list(lane.libraries or [])
            if _get_scattered_mix_priority_rank(lib) > max_allowed_rank
        ]
        if not removed_libs:
            kept_lanes.append(lane)
            continue

        adjusted_lanes += 1
        deferred_libraries += len(removed_libs)
        unassigned.extend(removed_libs)

        if not kept_libs:
            rebuildable = _can_rebuild_lane_from_priority_pool(
                candidate_pool=[
                    lib for lib in unassigned
                    if _get_scattered_mix_priority_rank(lib) <= max_allowed_rank
                ],
                validator=validator,
                machine_type=lane_machine_type,
                lane_id=lane.lane_id,
                lane_metadata=lane.metadata,
            )
            if not rebuildable:
                deferred_libraries -= len(removed_libs)
                for _ in removed_libs:
                    unassigned.pop()
                kept_lanes.append(lane)
                logger.info(
                    "全局优先级硬约束跳过整Lane移除 {}: 当前允许{}参与，但高优先级池无法重组出合法Lane，保留原Lane".format(
                        lane.lane_id,
                        _get_priority_gate_label(max_allowed_rank),
                    )
                )
                continue

            removed_lanes += 1
            logger.warning(
                "全局优先级硬约束移除整条Lane {}: 当前待排仅允许{}参与，该Lane仅包含更低优先级文库{}个".format(
                    lane.lane_id,
                    _get_priority_gate_label(max_allowed_rank),
                    len(removed_libs),
                )
            )
            continue

        lane.libraries = kept_libs
        lane.total_data_gb = sum(lib.get_data_amount_gb() for lib in kept_libs)
        lane.calculate_metrics()
        validation_result = _validate_lane_state(validator, lane, kept_libs)
        if validation_result.is_valid:
            kept_lanes.append(lane)
            logger.warning(
                "全局优先级硬约束裁剪Lane {}: 暂缓{}个较低优先级文库，保留{}个文库继续成Lane".format(
                    lane.lane_id,
                    len(removed_libs),
                    len(kept_libs),
                )
            )
        else:
            rebuildable = _can_rebuild_lane_from_priority_pool(
                candidate_pool=[
                    lib for lib in unassigned + kept_libs
                    if _get_scattered_mix_priority_rank(lib) <= max_allowed_rank
                ],
                validator=validator,
                machine_type=lane_machine_type,
                lane_id=lane.lane_id,
                lane_metadata=lane.metadata,
            )
            if not rebuildable:
                deferred_libraries -= len(removed_libs)
                for _ in removed_libs:
                    unassigned.pop()
                lane.libraries = list(lane.libraries or []) + removed_libs
                lane.total_data_gb = sum(lib.get_data_amount_gb() for lib in lane.libraries)
                lane.calculate_metrics()
                kept_lanes.append(lane)
                logger.info(
                    "全局优先级硬约束跳过Lane裁剪 {}: 裁剪后会失效，且高优先级池无法重组合法Lane，保留原Lane".format(
                        lane.lane_id
                    )
                )
                continue

            removed_lanes += 1
            deferred_libraries += len(kept_libs)
            unassigned.extend(kept_libs)
            logger.warning(
                "全局优先级硬约束导致Lane {} 裁剪后失效，整Lane回退到未分配池: errors={}".format(
                    lane.lane_id,
                    [error.rule_type.value for error in validation_result.errors],
                )
            )

    solution.lane_assignments = kept_lanes
    solution.unassigned_libraries = unassigned
    return {
        "adjusted_lanes": adjusted_lanes,
        "removed_lanes": removed_lanes,
        "deferred_libraries": deferred_libraries,
    }


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
                for lib in used:
                    if lib in unassigned:
                        unassigned.remove(lib)
                lanes.append(new_lane)
                new_lanes_count += 1

    return {"new_lanes": new_lanes_count, "remaining_unassigned": len(unassigned)}


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


def _rescue_remaining_lanes_by_layered_regroup_search(
    solution,
    validator,
    *,
    max_priority_cluster_lanes_per_machine: int = 8,
    max_mixed_rescue_lanes_per_machine: int = 8,
    max_normal_cluster_lanes_per_machine: int = 12,
    index_conflict_attempts_per_lane: int = DEFAULT_INDEX_CONFLICT_ATTEMPTS * 4,
    other_failure_attempts_per_lane: int = DEFAULT_OTHER_FAILURE_ATTEMPTS * 4,
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
    skipped_split_libraries = 0
    new_lanes: List[LaneAssignment] = []

    remaining_by_machine: Dict[MachineType, List[EnhancedLibraryInfo]] = {}
    passthrough: List[EnhancedLibraryInfo] = []
    for lib in unassigned:
        if _is_split_library(lib):
            skipped_split_libraries += 1
            passthrough.append(lib)
            continue

        machine_type = getattr(lib, "machine_type", None) or _resolve_machine_type_enum_simple(
            getattr(lib, "eq_type", "")
        )
        if not _is_machine_supported_for_arrangement(machine_type):
            passthrough.append(lib)
            continue
        remaining_by_machine.setdefault(machine_type, []).append(lib)

    for machine_type, machine_pool in list(remaining_by_machine.items()):
        if not machine_pool:
            continue
        machine_priority_cluster_lanes = 0
        machine_mixed_rescue_lanes = 0
        machine_normal_cluster_lanes = 0

        # Stage 1: 高优先级尾货优先做专Lane（同机型、同文库类型聚簇）。
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
            if machine_priority_cluster_lanes >= max_priority_cluster_lanes_per_machine:
                break
            active_cluster = [lib for lib in cluster_pool if lib in machine_pool]
            if not active_cluster:
                continue
            min_allowed, _ = _resolve_lane_capacity_limits(active_cluster, machine_type)
            if sum(lib.get_data_amount_gb() for lib in active_cluster) + 1e-6 < min_allowed:
                continue

            while active_cluster and machine_priority_cluster_lanes < max_priority_cluster_lanes_per_machine:
                lane, used = _attempt_build_rescue_lane_from_pool(
                    pool=active_cluster,
                    validator=validator,
                    machine_type=machine_type,
                    lane_id_prefix="PG",
                    lane_serial=_next_lane_serial("PG", machine_type),
                    index_conflict_attempts=index_conflict_attempts_per_lane,
                    other_failure_attempts=other_failure_attempts_per_lane,
                )
                if not lane:
                    break
                new_lanes.append(lane)
                machine_priority_cluster_lanes += 1
                priority_cluster_lanes += 1
                used_ids = {id(lib) for lib in used}
                machine_pool = [lib for lib in machine_pool if id(lib) not in used_ids]
                active_cluster = [lib for lib in active_cluster if id(lib) not in used_ids]

        # Stage 2: 仍有剩余高优先级时，允许高优先级主导混排，普通文库按门禁补位。
        while machine_mixed_rescue_lanes < max_mixed_rescue_lanes_per_machine:
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
            )
            if not lane:
                break
            lane_top_rank = _get_current_hard_priority_rank(list(lane.libraries or []))
            if lane_top_rank is None:
                break
            if lane_top_rank > current_top_rank:
                break
            new_lanes.append(lane)
            machine_mixed_rescue_lanes += 1
            mixed_rescue_lanes += 1
            used_ids = {id(lib) for lib in used}
            machine_pool = [lib for lib in machine_pool if id(lib) not in used_ids]

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
            if machine_normal_cluster_lanes >= max_normal_cluster_lanes_per_machine:
                break
            active_cluster = [lib for lib in cluster_pool if lib in machine_pool]
            if not active_cluster:
                continue
            min_allowed, _ = _resolve_lane_capacity_limits(active_cluster, machine_type)
            if sum(lib.get_data_amount_gb() for lib in active_cluster) + 1e-6 < min_allowed:
                continue

            while active_cluster and machine_normal_cluster_lanes < max_normal_cluster_lanes_per_machine:
                lane, used = _attempt_build_rescue_lane_from_pool(
                    pool=active_cluster,
                    validator=validator,
                    machine_type=machine_type,
                    lane_id_prefix="OG",
                    lane_serial=_next_lane_serial("OG", machine_type),
                    index_conflict_attempts=index_conflict_attempts_per_lane,
                    other_failure_attempts=other_failure_attempts_per_lane,
                )
                if not lane:
                    break
                new_lanes.append(lane)
                machine_normal_cluster_lanes += 1
                normal_cluster_lanes += 1
                used_ids = {id(lib) for lib in used}
                machine_pool = [lib for lib in machine_pool if id(lib) not in used_ids]
                active_cluster = [lib for lib in active_cluster if id(lib) not in used_ids]

        remaining_by_machine[machine_type] = machine_pool

    if new_lanes:
        solution.lane_assignments.extend(new_lanes)

    remaining_ids = {id(lib) for lib in passthrough}
    for machine_pool in remaining_by_machine.values():
        remaining_ids.update(id(lib) for lib in machine_pool)
    final_unassigned = [lib for lib in unassigned if id(lib) in remaining_ids]
    solution.unassigned_libraries = final_unassigned

    return {
        "new_lanes": len(new_lanes),
        "priority_cluster_lanes": priority_cluster_lanes,
        "mixed_rescue_lanes": mixed_rescue_lanes,
        "normal_cluster_lanes": normal_cluster_lanes,
        "remaining_unassigned": len(final_unassigned),
        "skipped_split_libraries": skipped_split_libraries,
    }


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


def _collect_detail_output_libraries(solution: Any) -> List[EnhancedLibraryInfo]:
    """收集最终输出明细所需的全部文库，包含成Lane与未分配文库。"""
    detail_libraries: List[EnhancedLibraryInfo] = []
    for lane in getattr(solution, "lane_assignments", []) or []:
        detail_libraries.extend(list(getattr(lane, "libraries", []) or []))
    detail_libraries.extend(list(getattr(solution, "unassigned_libraries", []) or []))
    return detail_libraries


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

    medical_project_data = sum(
        float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
        for lib in libraries
        if _is_medical_commission_library(lib)
    )
    if medical_project_data > 100.0:
        return 2.3, "medical_commission_over_100g_2_3"

    has_10_plus_24 = any(
        _matches_lane_seq_strategy_keyword(lib, "10+24")
        for lib in libraries
    )
    if has_10_plus_24 and any(
        _library_sample_type_matches_rule(lib, LANE_LOADING_10_PLUS_24_ATAC_TYPES)
        for lib in libraries
    ):
        return 2.0, "10_plus_24_atac_2_0"

    if lane_sample_types and lane_sample_types.issubset(LANE_LOADING_COMBO_GROUP_A):
        if _lane_contains_customer_prefixed_sample_type(lane_sample_types):
            return 2.5, "special_10x_combo_group_a_customer_2_5"
        return 1.78, "special_10x_combo_group_a_non_customer_1_78"

    if lane_sample_types and lane_sample_types.issubset(LANE_LOADING_COMBO_GROUP_B):
        if _lane_contains_customer_prefixed_sample_type(lane_sample_types):
            return 2.5, "special_10x_combo_group_b_customer_2_5"
        return 1.78, "special_10x_combo_group_b_non_customer_1_78"

    return None, "no_explicit_loading_rule_matched"


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
    """按显式业务规则优先，其次走统一规则表计算Lane上机浓度。"""
    if not libraries:
        return None, "empty_lane"
    lane_sample_types = _get_lane_sample_types(libraries)
    explicit_concentration, explicit_rule = _resolve_explicit_lane_loading_concentration(
        libraries,
        lane_sample_types,
    )
    if explicit_concentration is not None:
        return explicit_concentration, explicit_rule
    return get_scheduling_config().resolve_loading_concentration(libraries)


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
    """解析历史产出率，优先显式字段，缺失时回退到上一轮产出/上一轮下单。"""
    if last_outrate is not None:
        try:
            outrate = float(last_outrate)
            if not pd.isna(outrate) and outrate > 0:
                return outrate
        except (TypeError, ValueError):
            pass

    if last_output is None or last_order is None:
        return None
    try:
        output_value = float(last_output)
        order_value = float(last_order)
    except (TypeError, ValueError):
        return None
    if pd.isna(output_value) or pd.isna(order_value) or order_value <= 0:
        return None
    derived_outrate = output_value / order_value
    if derived_outrate <= 0:
        return None
    return derived_outrate


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
        baleno = getattr(lib, "package_lane_number", None) or getattr(lib, "baleno", None)
        if baleno and str(baleno).strip():
            lib.package_lane_number = str(baleno).strip()
            lib.is_package_lane = "是"
            failed_package_libraries.append(lib)
        else:
            remaining_normal_libraries.append(lib)

    return failed_package_libraries, remaining_normal_libraries


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
        if lane_metadata.get("is_package_lane"):
            metadata["is_package_lane"] = True
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
    balance_already_in_libs: bool = False,
    skip_peak_size: bool = False,
) -> Any:
    """校验给定文库列表在当前Lane上下文中的合法性。

    balance_already_in_libs=True：libraries 已含平衡文库本体，去掉 metadata 里的
    wkbalancedata 避免容量校验器二次叠加。

    skip_peak_size=True：跳过 peak_size 错误/警告的判断。专用不均衡 lane 注入平衡文库
    时使用——该 lane 的 peak_size 分布是排机时就已形成的既成事实，平衡文库不应因此被阻止。
    """
    has_balance_library = any(_is_ai_balance_library(lib) for lib in libraries)
    if _is_package_lane_assignment(lane) and not (balance_already_in_libs or has_balance_library):
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

    metadata = _build_lane_metadata_for_validator(lane.lane_id, lane.metadata)
    if balance_already_in_libs:
        metadata.pop("wkbalancedata", None)
        metadata.pop("wkadd_balance_data", None)
        metadata.pop("required_balance_data_gb", None)
    machine_type = lane.machine_type.value if lane.machine_type else "Nova X-25B"
    result = _validate_lane_with_latest_index(
        validator=validator,
        libraries=libraries,
        lane_id=lane.lane_id,
        machine_type=machine_type,
        metadata=metadata,
    )
    if skip_peak_size and not result.is_valid:
        # 过滤掉 peak_size 相关的 errors/warnings，重新判断是否通过
        filtered_errors = [e for e in result.errors if e.rule_type != ValidationRuleType.PEAK_SIZE]
        filtered_warnings = [w for w in result.warnings if w.rule_type != ValidationRuleType.PEAK_SIZE]
        is_valid = len(filtered_errors) == 0 and (not validator.strict_mode or len(filtered_warnings) == 0)
        result = LaneValidationResult(
            lane_id=result.lane_id,
            is_valid=is_valid,
            errors=filtered_errors,
            warnings=filtered_warnings,
        )

    if _is_package_lane_assignment(lane) and (balance_already_in_libs or has_balance_library):
        package_errors = _validate_package_lane_rules(lane, libraries=libraries)
        if package_errors:
            result.errors.extend(
                ValidationError(
                    rule_type=ValidationRuleType.CAPACITY,
                    severity=ValidationSeverity.ERROR,
                    message=message,
                )
                for message in package_errors
            )
            result.is_valid = False
    return result


def _filter_valid_lanes(
    lanes: List[LaneAssignment],
    validator: Any,
) -> Tuple[List[LaneAssignment], List[LaneAssignment]]:
    """过滤出通过严格校验的Lane。"""
    valid_lanes: List[LaneAssignment] = []
    failed_lanes: List[LaneAssignment] = []
    for lane in lanes:
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
        imbalance_mix_valid, _ = _validate_lane_57_mix_rules(
            trial_libs,
            enforce_total_limit=False,
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
                    target_non_cust = max(data_non_customers, data_customers / customer_ratio_limit - data_customers)
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

        # 排机阶段不做下单/产出预测，统一在 prediction_delivery 阶段落地。
        for lib in libs:
            contract = float(lib.contract_data_raw or 0.0)
            loutput = loutput_by_origrec.get(lib.origrec)
            last_qpcr = _get_lib_attr_float(lib, ["_last_qpcr_raw", "last_qpcr", "wklastqpcr", "wklistqpcr"])
            last_order = _get_lib_attr_float(lib, ["_last_order_data_raw", "last_order_data", "wklastorderdata"])
            last_output = _get_lib_attr_float(lib, ["_last_output_raw", "last_output", "wklastoutput"])
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
                    "lane_id": lane.lane_id,
                    "lsjnd": (
                        None
                        if lane_loading_concentration is None
                        else round(float(lane_loading_concentration), 3)
                    ),
                    "resolved_lsjfs": lane_loading_method or None,
                    "resolved_lcxms": lane_sequencing_mode or None,
                    "resolved_index_check_rule": lane_index_rule or None,
                    "wkcontractdata": contract,
                    "wkbalancedata": (
                        lane_balance_data_value
                        if _is_package_lane_assignment(lane) and is_balance_lib
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


def _find_output_57_failed_lane_ids(df: pd.DataFrame) -> Set[str]:
    """按输出表字段口径复核57规则，返回失败的llaneid集合。"""
    if "llaneid" not in df.columns:
        return set()

    def _is_scheduled(value: Any) -> bool:
        text = str(value).strip()
        return bool(text) and text.lower() not in {"", "nan", "none", "null"} and text != "0"

    scheduled_df = df.loc[df["llaneid"].map(_is_scheduled)].copy()
    if scheduled_df.empty:
        return set()

    failed_lane_ids: Set[str] = set()
    for lane_id, sub in scheduled_df.groupby("llaneid", sort=False):
        libs: List[Any] = []
        for row in sub.to_dict(orient="records"):
            sample_type = str(row.get("wksampletype") or row.get("wkdatatype") or "")
            data_type = str(row.get("wkdatatype") or row.get("wksampletype") or "")
            sample_id = str(row.get("wksampleid") or "")
            contract_data = pd.to_numeric(row.get("wkcontractdata"), errors="coerce")
            libs.append(
                type(
                    "OutputLaneLib",
                    (),
                    {
                        "sample_type_code": sample_type,
                        "sampletype": sample_type,
                        "data_type": data_type,
                        "lab_type": data_type,
                        "sample_id": sample_id,
                        "customer_library": "是"
                        if sample_id.startswith("FKDL") or sample_type.startswith("客户")
                        else "否",
                        "contract_data_raw": 0.0 if pd.isna(contract_data) else float(contract_data),
                        "jjbj": "是" if str(row.get("wk_jjbj") or "").strip() == "是" else "否",
                    },
                )()
            )

        ok, reason = _validate_lane_57_mix_rules(libs, enforce_total_limit=False)
        if not ok:
            failed_lane_ids.add(str(lane_id))
            logger.warning(f"输出前57规则复核失败: lane={lane_id}, reason={reason}")

    return failed_lane_ids


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
        if total_contract_data not in (None, ""):
            template["wktotalcontractdata"] = float(total_contract_data)

        package_lane_number = _safe_str(
            getattr(lib, "package_lane_number", None) or getattr(lib, "baleno", None),
            default="",
        )
        if package_lane_number:
            template["wkbaleno"] = package_lane_number

        aidbid = _safe_str(getattr(lib, "wkaidbid", None) or getattr(lib, "aidbid", None), default="")
        if aidbid:
            template["wkaidbid"] = aidbid

        if _is_ai_balance_library(lib):
            template[BALANCE_LIBRARY_MARKER_COLUMN] = True

        if _is_split_library(lib):
            template["wkissplit"] = "yes"
        else:
            template["wkissplit"] = _safe_str(template.get("wkissplit"), default="")

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
) -> None:
    """生成明细输出文件"""
    def _ensure_object_column(df: pd.DataFrame, column_name: str) -> None:
        """在写入字符串前显式转为object列，避免pandas类型告警。"""
        if column_name in df.columns:
            df[column_name] = df[column_name].astype(object)

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
            "lane_id",
            "lsjnd",
            "resolved_lsjfs",
            "resolved_lcxms",
            "resolved_seq_mode",
            "resolved_round_label",
            "resolved_index_check_rule",
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
                "lane_id",
                "lsjnd",
                "resolved_lsjfs",
                "resolved_lcxms",
                "resolved_seq_mode",
                "resolved_round_label",
                "resolved_index_check_rule",
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
        merged.loc[lane_assigned_mask & resolved_round_mask, "laneround"] = merged.loc[
            lane_assigned_mask & resolved_round_mask, "resolved_round_label"
        ]

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

    # 将新生成的runid/laneid覆盖写回原始字段lrunid/llaneid，并移除runid/laneid输出列
    if "lrunid" not in merged.columns:
        merged["lrunid"] = pd.NA
    if "llaneid" not in merged.columns:
        merged["llaneid"] = pd.NA
    _ensure_object_column(merged, "lrunid")
    _ensure_object_column(merged, "llaneid")
    # 输出文件中的排机字段只保留本轮最终成功Lane，先清空旧值，再按pred_df回填。
    merged["lrunid"] = ""
    merged["llaneid"] = ""
    merged.loc[lane_assigned_mask, "lrunid"] = merged.loc[lane_assigned_mask, "runid"]
    merged.loc[lane_assigned_mask, "llaneid"] = merged.loc[lane_assigned_mask, "laneid"]

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

    failed_57_lane_ids = _find_output_57_failed_lane_ids(merged)
    if failed_57_lane_ids:
        failed_57_mask = merged["llaneid"].astype(str).isin(failed_57_lane_ids)
        logger.warning(
            "输出前57规则复核淘汰{}条Lane: {}".format(
                len(failed_57_lane_ids),
                sorted(failed_57_lane_ids),
            )
        )
        for column_name in [
            "lrunid",
            "llaneid",
            "lcxms",
            "lsjfs",
            "lanecreatetype",
            "排机规则",
            "index查重规则",
        ]:
            if column_name in merged.columns:
                merged.loc[failed_57_mask, column_name] = ""
        for column_name in ["runid", "laneid"]:
            if column_name in merged.columns:
                merged.loc[failed_57_mask, column_name] = pd.NA
        lane_assigned_mask = lane_assigned_mask & (~failed_57_mask)

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
            # 保留拆分相关原始字段，供lane_show与拆分Lane识别使用
            raw_wkissplit = _safe_str(row_dict.get("wkissplit"), default="")
            lib.wkissplit = raw_wkissplit
            if getattr(lib, "is_split", None) is None and _is_yes_value(raw_wkissplit):
                lib.is_split = True
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
            lib._aiavailable_raw = _safe_str(row_dict.get("aiavailable"), default="")
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
            lib._delete_date_raw = row_dict.get("delete_date", row_dict.get("扣减时间"))
            lib._wkdept_raw = _safe_str(row_dict.get("wkdept"), default="")
            # 保存测序模式相关原始字段，供拆分规则识别1.1/3.6T-NEW模式使用
            lib._lane_sj_mode_raw = _safe_str(row_dict.get("lsjfs"), default="")
            lib._current_seq_mode_raw = _safe_str(row_dict.get("lcxms"), default="")
            lib._last_cxms_raw = _safe_str(row_dict.get("llastcxms"), default="")
            # 1.1模式轮次字段：上轮测序轮数（lims推送），供第二轮候选识别使用
            lib._last_lane_round_raw = _safe_str(
                row_dict.get("llastlaneround") or row_dict.get("lastlaneround"),
                default="",
            )
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
    
    scheduler = GreedyLaneScheduler(config)

    # 排机阶段不使用模型
    if scheduler.pooling_optimizer:
        scheduler.pooling_optimizer.enabled = False
        logger.info("排机阶段：Pooling优化器已禁用（V6流程不需要）")
    
    from arrange_library.core.constraints.lane_validator import LaneValidator
    strict_validator = LaneValidator(strict_mode=True)
    logger.info("严格校验容量区间改为按统一配置表动态解析")

    disabled_plan = StrategyExecutionPlan()
    disabled_plan.enable_dedicated_imbalance_lane = False
    disabled_plan.enable_non_10bp_dedicated_lane = False
    disabled_plan.enable_backbone_reservation = False
    scheduler._strategy_plan = disabled_plan

    # ===== 预拆分前置到所有预构建Lane之前 =====
    presplit_libraries, presplit_records = scheduler.library_splitter.split_libraries(libraries)
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

    mixed_lanes, remaining_libraries = _extract_mixed_lanes_by_peak_window(
        libraries=presplit_libraries,
        validator=strict_validator,
        machine_type=MachineType.NOVA_X_25B,
        index_conflict_attempts_per_lane=DEFAULT_INDEX_CONFLICT_ATTEMPTS,
        other_failure_attempts_per_lane=DEFAULT_OTHER_FAILURE_ATTEMPTS,
    )

    # 执行排机（剩余文库进入混样排机）
    if remaining_libraries:
        solution = scheduler.schedule(
            remaining_libraries,
            keep_failed_lanes=True,
            libraries_already_split=True,
            perform_presplit_family_rollback=False,
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
                failed_lanes.append(lane)
                error_types = [e.rule_type.value for e in result.errors]
                warning_types = [w.rule_type.value for w in result.warnings]
                logger.warning(
                    f"Lane {lane.lane_id} 验证失败 - 错误: {error_types}, 警告: {warning_types}"
                )

    solution.lane_assignments = passed_lanes
    logger.info(f"验证完成：{len(passed_lanes)}条Lane通过验证")
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
        valid_preset_lanes, failed_preset_lanes = _filter_valid_lanes(preset_lanes, strict_validator)
        logger.info(
            f"\n合并{len(valid_preset_lanes)}条预构建Lane到最终结果"
            f"（过滤掉{len(failed_preset_lanes)}条未通过严格校验的预构建Lane）"
        )
        if failed_preset_lanes:
            for lane in failed_preset_lanes:
                solution.unassigned_libraries.extend(list(lane.libraries or []))
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

    layered_regroup_stats = _rescue_remaining_lanes_by_layered_regroup_search(
        solution=solution,
        validator=strict_validator,
    )
    if layered_regroup_stats["new_lanes"] > 0:
        logger.info(
            "剩余库分层重组搜索完成: 新增Lane={} (专lane={}, 混排lane={}, 普通lane={}), "
            "跳过拆分文库={}, 剩余未分配={}".format(
                layered_regroup_stats["new_lanes"],
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

    renamed_lane_ids = _ensure_unique_lane_ids(solution.lane_assignments)
    if renamed_lane_ids > 0:
        logger.warning("最终收口发现重复Lane ID并已重命名: {}条", renamed_lane_ids)

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
    predict_pooling(input_data=input_data, output_file=output_path)
    prediction_df = _read_csv_with_encoding_fallback(output_path)
    prediction_df = _apply_add_test_output_rate_rule_to_prediction_df(
        prediction_df=prediction_df,
        output_path=output_path,
    )
    if BALANCE_LIBRARY_MARKER_COLUMN in prediction_df.columns:
        marker = prediction_df[BALANCE_LIBRARY_MARKER_COLUMN].fillna(False)
        marker = marker.astype(str).str.lower().isin({"true", "1", "yes"})
        if marker.any():
            prediction_df.loc[marker, "lorderdata"] = pd.to_numeric(
                prediction_df.loc[marker, "wkcontractdata"], errors="coerce"
            )
            prediction_df.loc[marker, "lai_output"] = pd.NA
        prediction_df = prediction_df.drop(columns=[BALANCE_LIBRARY_MARKER_COLUMN], errors="ignore")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_df.to_csv(output_path, index=False)
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
        logger.info("arrange_library: 仅执行 Pooling 预测")
        logger.info("=" * 80)
        _run_prediction_delivery(input_data=data_path, output_path=output_path)
        logger.info(f"Pooling 预测完成，输出文件: {output_path}")
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
    ai_schedulable_keys: Set[str] = set()
    for lib in libraries:
        machine_type = getattr(lib, "machine_type", None) or _resolve_machine_type_enum_simple(getattr(lib, "eq_type", ""))
        lib.machine_type = machine_type
        origrec_key = _safe_str(getattr(lib, "_origrec_key", getattr(lib, "origrec", "")))
        if not _is_machine_supported_for_arrangement(machine_type):
            excluded_machine_libraries.append(lib)
            continue
        if _is_yes_value(getattr(lib, "_aiavailable_raw", "")):
            ai_schedulable_libraries.append(lib)
            ai_schedulable_keys.add(origrec_key)
        else:
            non_ai_libraries.append(lib)
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

    # ===== 步骤1: 处理包Lane文库 =====
    logger.info("\n" + "=" * 80)
    logger.info("步骤1: 处理包Lane文库（wkbaleno字段有值的文库）")
    logger.info("=" * 80)
    
    package_lanes = []
    package_libs = []
    normal_libs = []
    failed_package_libs = []
    
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
    
    # ===== 步骤1.5: 1.1模式分流编排 =====
    mode_1_1_config = get_scheduling_config().get_mode_1_1_config()
    mode_1_1_lanes: List[LaneAssignment] = []
    if mode_1_1_config:
        logger.info("\n" + "=" * 80)
        logger.info("步骤1.5: 1.1模式分流编排")
        logger.info("=" * 80)

        allocator = ModeAllocator(mode_1_1_config)
        dispatch_result = allocator.allocate(normal_libs)

        # 3.6T-NEW 优先池 + 1.1 禁排回退池 合并为普通排机池
        normal_libs_for_36t = dispatch_result.pool_36t_priority + dispatch_result.pool_1_1_forbidden
        # 1.1 可排文库按质量分组依次排机
        pool_1_1_all = (
            dispatch_result.pool_1_1_normal
            + dispatch_result.pool_1_1_quality_risk
            + dispatch_result.pool_1_1_quality_other
        )

        if pool_1_1_all:
            logger.info("1.1首轮池共{}个文库，进入1.1模式排机", len(pool_1_1_all))
            # 为1.1池文库注入模式标记，供后续规则矩阵命中1.1 profile
            for lib in pool_1_1_all:
                lib._current_seq_mode_raw = "1.1"
            try:
                _1_1_stats, _1_1_solution = test_with_model(
                    deepcopy(pool_1_1_all), existing_lanes=[]
                )
                # 给1.1产出的lane注入模式和轮次元数据
                first_round_label = mode_1_1_config.get("first_round_label", "1.1第一轮")
                for lane in _1_1_solution.lane_assignments:
                    if not isinstance(lane.metadata, dict):
                        lane.metadata = {}
                    lane.metadata["dispatch_stage"] = "first_round_1_1"
                    lane.metadata["selected_seq_mode"] = "1.1"
                    lane.metadata["selected_round_label"] = first_round_label
                mode_1_1_lanes = list(_1_1_solution.lane_assignments)
                # 1.1排不走的文库回流到3.6T-NEW，优先级不变
                fallback_libs = list(_1_1_solution.unassigned_libraries or [])
                if fallback_libs:
                    logger.info("1.1首轮未排走{}个文库，回流到3.6T-NEW候选池", len(fallback_libs))
                    # 清除1.1模式标记，恢复原始模式供3.6T-NEW规则矩阵使用
                    for lib in fallback_libs:
                        lib._current_seq_mode_raw = ""
                    normal_libs_for_36t.extend(fallback_libs)
                logger.info("1.1首轮排机完成: 生成{}条Lane", len(mode_1_1_lanes))
            except Exception as exc:
                logger.error("1.1首轮排机异常，全部回退到3.6T-NEW: {}", exc)
                for lib in pool_1_1_all:
                    lib._current_seq_mode_raw = ""
                normal_libs_for_36t.extend(pool_1_1_all)

        # 第二轮候选识别（首版仅做识别和日志，不执行真实排机）
        round2_handler = Mode11Round2Handler(mode_1_1_config)
        round2_result = round2_handler.identify_round2_candidates(normal_libs_for_36t)
        if round2_result.total_candidates > 0:
            round2_handler.schedule_round2(round2_result.candidate_groups)
            # 第二轮候选暂不从普通池移除，首版仍走3.6T-NEW排机

        normal_libs = normal_libs_for_36t
        logger.info("模式分流编排完成: 3.6T-NEW候选池={}个文库, 1.1 Lane={}条",
                     len(normal_libs), len(mode_1_1_lanes))
    else:
        logger.info("未加载1.1模式配置，跳过模式分流，全部走3.6T-NEW排机")

    # ===== 步骤2: 处理普通文库（包括包Lane处理失败的文库） =====
    logger.info("\n" + "=" * 80)
    logger.info("步骤2: 处理普通文库（使用GreedyLaneScheduler）")
    logger.info("=" * 80)

    if ai_schedulable_libraries:
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
            # 将包Lane和1.1模式Lane一起作为existing_lanes传入，在最终合并阶段统一纳入
            all_existing_lanes = list(package_lanes) + list(mode_1_1_lanes)
            stats, solution = test_with_model(
                deepcopy(normal_libs), existing_lanes=all_existing_lanes
            )
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

    _validate_final_package_lanes(solution)
    _validate_no_split_for_package_lane_libraries(solution)

    balance_materialize_stats = _materialize_balance_libraries_for_solution(solution)
    if balance_materialize_stats["required_lanes"] > 0:
        logger.info(
            "平衡文库后处理完成: 需补平衡文库Lane={}，成功={}".format(
                balance_materialize_stats["required_lanes"],
                balance_materialize_stats["success_lanes"],
            )
        )
    _validate_final_package_lanes(solution)

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
        ai_schedulable_keys=ai_schedulable_keys,
        lanes_with_split=lanes_with_split,
        detail_libraries=detail_libraries,
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
