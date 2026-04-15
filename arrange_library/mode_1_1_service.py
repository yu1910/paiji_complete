"""
1.1模式对外服务封装

对外提供稳定调用入口，复用现有主流程中的分流、排机和第二轮识别能力，
避免外部程序直接耦合 arrange_library_model6.py 内部编排细节。
"""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Union

from loguru import logger

from arrange_library.arrange_library_model6 import (
    _is_machine_supported_for_arrangement,
    _resolve_machine_type_enum_simple,
    arrange_library,
    load_test_libraries,
    test_with_model,
)
from arrange_library.core.config.scheduling_config import get_scheduling_config
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
    first_round_pool: List[EnhancedLibraryInfo] = field(default_factory=list)
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


def _resolve_mode_1_1_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """解析1.1配置，优先使用显式传入值。"""
    resolved = dict(config or get_scheduling_config().get_mode_1_1_config())
    if not resolved:
        raise ValueError("未加载到1.1模式配置，无法执行 mode_1_1_service")
    return resolved


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
) -> Mode11Round1ServiceResult:
    """执行1.1首轮分流与排机，返回可供外部程序消费的结构化结果。"""
    resolved_config = _resolve_mode_1_1_config(config)
    prepared = prepare_mode_1_1_libraries(data_file=data_file, libraries=libraries)
    allocator = ModeAllocator(resolved_config)
    dispatch_result = allocator.allocate(prepared.ai_schedulable_libraries)

    normal_libraries_for_36t = list(dispatch_result.pool_36t_priority) + list(
        dispatch_result.pool_1_1_forbidden
    )
    first_round_pool = (
        list(dispatch_result.pool_1_1_normal)
        + list(dispatch_result.pool_1_1_quality_risk)
        + list(dispatch_result.pool_1_1_quality_other)
    )

    result = Mode11Round1ServiceResult(
        prepared_input=prepared,
        dispatch_result=dispatch_result,
        first_round_pool=first_round_pool,
        normal_libraries_for_36t=normal_libraries_for_36t,
        solution=SimpleNamespace(lane_assignments=[], unassigned_libraries=[]),
    )

    if not first_round_pool:
        result.scheduling_succeeded = True
        logger.info("mode_1_1_service 首轮排机跳过: 无可进入1.1首轮池的文库")
        return result

    for lib in first_round_pool:
        lib._current_seq_mode_raw = "1.1"

    first_round_label = str(resolved_config.get("first_round_label", "1.1第一轮"))
    try:
        scheduling_stats, solution = test_with_model(
            deepcopy(first_round_pool),
            existing_lanes=list(existing_lanes or []),
        )
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
    "Mode11Round2ServiceResult",
    "prepare_mode_1_1_libraries",
    "run_mode_1_1_full",
    "run_mode_1_1_round1",
    "run_mode_1_1_round2",
]
