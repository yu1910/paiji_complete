"""
1.1模式第二轮候选识别与分组模块
创建时间：2026-04-14 13:40:00
更新时间：2026-04-14 13:40:00

首版只交付候选识别与分组框架，不含真实闭环排机。
真实排机（pooling 2.5 特例、二轮平衡文库口径等）预留接口，后续迭代补充。

职责：
- 从待排文库中识别 lastlaneround == "1.1第一轮" 的第二轮候选
- 按 llastlaneid 分组形成强绑定候选包
- 预留第二轮排机调用入口
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from arrange_library.models.library_info import EnhancedLibraryInfo


@dataclass
class Round2CandidateGroup:
    """第二轮候选分组：同一个 llastlaneid 下的文库集合"""
    last_lane_id: str
    libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    total_contract_gb: float = 0.0


@dataclass
class Round2IdentificationResult:
    """第二轮候选识别结果"""
    candidate_groups: List[Round2CandidateGroup] = field(default_factory=list)
    non_candidates: List[EnhancedLibraryInfo] = field(default_factory=list)
    total_candidates: int = 0


@dataclass
class Round2ScheduledLane:
    """第二轮排机产出的lane及其关联的原始分组信息。"""
    lane: Any
    source_last_lane_ids: List[str] = field(default_factory=list)
    strong_binding_kept: bool = True
    break_reason: str = ""


@dataclass
class Round2SchedulingResult:
    """第二轮排机执行结果。"""
    lanes: List[Any] = field(default_factory=list)
    scheduled_lane_results: List[Round2ScheduledLane] = field(default_factory=list)
    fallback_libraries: List[EnhancedLibraryInfo] = field(default_factory=list)
    scheduled_groups: int = 0
    broken_groups: int = 0
    total_scheduled_libraries: int = 0
    group_break_reasons: Dict[str, str] = field(default_factory=dict)


class Mode11Round2Handler:
    """1.1模式第二轮候选识别与分组

    按照规则15：
    - lastlaneround == "1.1第一轮" 的文库为第二轮候选
    - llastlaneid 相同的候选文库排在同 lane（强约束）
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._first_round_label = str(config.get("first_round_label", "1.1第一轮"))
        self._second_round_label = str(config.get("second_round_label", "1.1第二轮"))
        self._output_rate_threshold = float(config.get("second_round_output_rate_threshold", 40.0))
        self._default_pooling_factor = float(config.get("second_round_default_pooling_factor", 2.5))

    def identify_round2_candidates(
        self,
        libraries: List[EnhancedLibraryInfo],
    ) -> Round2IdentificationResult:
        """从待排文库中识别第二轮候选并按 llastlaneid 分组

        Args:
            libraries: 待排文库列表（可能包含首轮和非首轮文库）

        Returns:
            Round2IdentificationResult，包含分组后的候选和非候选文库
        """
        result = Round2IdentificationResult()
        groups_map: Dict[str, Round2CandidateGroup] = {}

        for lib in libraries:
            last_round = self._get_last_lane_round(lib)
            last_lid = self._get_last_lane_id(lib)

            if last_round == self._first_round_label and last_lid:
                if last_lid not in groups_map:
                    groups_map[last_lid] = Round2CandidateGroup(last_lane_id=last_lid)
                group = groups_map[last_lid]
                group.libraries.append(lib)
                group.total_contract_gb += float(getattr(lib, "contract_data_raw", 0) or 0)
                result.total_candidates += 1
            else:
                result.non_candidates.append(lib)

        result.candidate_groups = list(groups_map.values())

        if result.total_candidates > 0:
            logger.info(
                "1.1第二轮候选识别: 候选文库={}, 分组数={}, 非候选文库={}",
                result.total_candidates,
                len(result.candidate_groups),
                len(result.non_candidates),
            )
            for group in result.candidate_groups:
                logger.debug(
                    "  第二轮分组: llastlaneid={}, 文库数={}, 合同量={:.1f}G",
                    group.last_lane_id,
                    len(group.libraries),
                    group.total_contract_gb,
                )
        else:
            logger.info("1.1第二轮候选识别: 无符合条件的候选文库")

        return result

    def schedule_round2(
        self,
        candidate_groups: List[Round2CandidateGroup],
    ) -> Round2SchedulingResult:
        """执行第二轮真实排机并返回lane与回流结果。"""
        result = Round2SchedulingResult()
        if not candidate_groups:
            return result

        lane_buckets = self._build_lane_buckets(candidate_groups)
        logger.info(
            "1.1第二轮排机: 候选分组={}，装箱后待调度桶={}",
            len(candidate_groups),
            len(lane_buckets),
        )

        for bucket in lane_buckets:
            bucket_schedule = self._schedule_bucket(bucket)
            result.lanes.extend(bucket_schedule.lanes)
            result.scheduled_lane_results.extend(bucket_schedule.scheduled_lane_results)
            result.fallback_libraries.extend(bucket_schedule.fallback_libraries)
            result.scheduled_groups += bucket_schedule.scheduled_groups
            result.broken_groups += bucket_schedule.broken_groups
            result.total_scheduled_libraries += bucket_schedule.total_scheduled_libraries
            result.group_break_reasons.update(bucket_schedule.group_break_reasons)

        logger.info(
            "1.1第二轮排机完成: 生成Lane={}, 调度分组={}, 打破强绑定分组={}, 回流3.6T文库={}",
            len(result.lanes),
            result.scheduled_groups,
            result.broken_groups,
            len(result.fallback_libraries),
        )
        return result

    def _get_last_lane_round(self, lib: EnhancedLibraryInfo) -> str:
        """获取文库的上轮测序轮数"""
        # 优先从 dataclass 字段读取，其次从运行时附加的原始值读取
        value = getattr(lib, "last_lane_round", None)
        if not value:
            value = getattr(lib, "_last_lane_round_raw", None)
        return str(value or "").strip()

    def _get_last_lane_id(self, lib: EnhancedLibraryInfo) -> str:
        """获取文库的上次 lane ID"""
        value = getattr(lib, "last_laneid", None)
        if not value:
            value = getattr(lib, "lastlaneid", None)
        return str(value or "").strip()

    def _get_last_output_rate(self, lib: EnhancedLibraryInfo) -> Optional[float]:
        """获取历史产出率，兼容 dataclass 字段与运行时透传字段。"""
        for attr_name in ("last_outrate", "_last_outrate_raw", "wklastoutrate"):
            value = getattr(lib, attr_name, None)
            if value in (None, ""):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _group_requires_default_pooling(self, group: Round2CandidateGroup) -> bool:
        """判断分组是否需要应用第二轮默认pooling系数。"""
        for lib in group.libraries:
            last_outrate = self._get_last_output_rate(lib)
            if last_outrate is not None and last_outrate < self._output_rate_threshold:
                return True
        return False

    def _get_low_output_origrecs(self, group: Round2CandidateGroup) -> List[str]:
        """返回需要应用第二轮默认pooling系数的文库origrec。"""
        origrecs: List[str] = []
        for lib in group.libraries:
            last_outrate = self._get_last_output_rate(lib)
            if last_outrate is None or last_outrate >= self._output_rate_threshold:
                continue
            origrec = str(getattr(lib, "origrec", "") or "").strip()
            if origrec:
                origrecs.append(origrec)
        return origrecs

    def _build_lane_buckets(
        self,
        candidate_groups: List[Round2CandidateGroup],
    ) -> List[List[Round2CandidateGroup]]:
        """按1.1容量上限对第二轮分组进行贪心装桶。"""
        from arrange_library.arrange_library_model6 import _resolve_lane_capacity_limits

        sorted_groups = sorted(
            candidate_groups,
            key=lambda group: (
                0 if self._group_requires_default_pooling(group) else 1,
                -group.total_contract_gb,
                group.last_lane_id,
            ),
        )
        buckets: List[List[Round2CandidateGroup]] = []

        for group in sorted_groups:
            placed = False
            best_bucket_index = -1
            best_remaining = float("inf")

            for idx, bucket in enumerate(buckets):
                combined_groups = bucket + [group]
                combined_libraries = self._flatten_group_libraries(combined_groups)
                if not combined_libraries:
                    continue
                _, max_limit = _resolve_lane_capacity_limits(
                    libraries=combined_libraries,
                    machine_type=getattr(combined_libraries[0], "machine_type", None),
                    lane_id="M11R2_TMP",
                    lane_metadata={"lcxms": "1.1"},
                )
                total_gb = sum(item.total_contract_gb for item in combined_groups)
                if total_gb > max_limit + 1e-6:
                    continue
                remaining = max_limit - total_gb
                if remaining < best_remaining:
                    best_bucket_index = idx
                    best_remaining = remaining

            if best_bucket_index >= 0:
                buckets[best_bucket_index].append(group)
                placed = True

            if not placed:
                buckets.append([group])

        return buckets

    def _schedule_bucket(
        self,
        bucket_groups: List[Round2CandidateGroup],
    ) -> Round2SchedulingResult:
        """对单个装桶结果调用现有排机主能力，形成第二轮lane。"""
        from arrange_library.arrange_library_model6 import test_with_model

        result = Round2SchedulingResult()
        if not bucket_groups:
            return result

        source_last_lane_ids = [group.last_lane_id for group in bucket_groups]
        bucket_libraries = self._flatten_group_libraries(bucket_groups)
        for lib in bucket_libraries:
            lib._current_seq_mode_raw = "1.1"

        low_output_origrecs = sorted(
            {
                origrec
                for group in bucket_groups
                for origrec in self._get_low_output_origrecs(group)
            }
        )

        break_reason = ""
        strong_binding_kept = True
        try:
            _, solution = test_with_model(
                deepcopy(bucket_libraries),
                existing_lanes=[],
            )
        except Exception as exc:
            logger.error(
                "1.1第二轮排机失败，整桶回流3.6T-NEW: llastlaneid={}, error={}",
                source_last_lane_ids,
                exc,
            )
            result.fallback_libraries.extend(bucket_libraries)
            for group in bucket_groups:
                result.group_break_reasons[group.last_lane_id] = "round2_schedule_exception"
            return result
        finally:
            for lib in bucket_libraries:
                lib._current_seq_mode_raw = ""

        scheduled_lanes = list(getattr(solution, "lane_assignments", []) or [])
        fallback_libraries = list(getattr(solution, "unassigned_libraries", []) or [])
        if len(scheduled_lanes) != 1 or fallback_libraries:
            strong_binding_kept = False
            if len(scheduled_lanes) == 0:
                break_reason = "round2_scheduler_no_lane"
            elif fallback_libraries:
                break_reason = "round2_scheduler_partial_fallback"
            else:
                break_reason = "round2_scheduler_split_multiple_lanes"
            for group in bucket_groups:
                result.group_break_reasons[group.last_lane_id] = break_reason

        for lane in scheduled_lanes:
            if not isinstance(getattr(lane, "metadata", None), dict):
                lane.metadata = {}
            lane.metadata["dispatch_stage"] = "second_round_1_1"
            lane.metadata["selected_seq_mode"] = "1.1"
            lane.metadata["selected_round_label"] = self._second_round_label
            lane.metadata["mode_1_1_round2_source_last_lane_ids"] = list(source_last_lane_ids)
            lane.metadata["mode_1_1_round2_group_count"] = len(bucket_groups)
            if low_output_origrecs:
                lane.metadata["mode_1_1_round2_pooling_factor"] = self._default_pooling_factor
                lane.metadata["mode_1_1_round2_low_output_origrecs"] = list(low_output_origrecs)

            result.lanes.append(lane)
            result.scheduled_lane_results.append(
                Round2ScheduledLane(
                    lane=lane,
                    source_last_lane_ids=list(source_last_lane_ids),
                    strong_binding_kept=strong_binding_kept,
                    break_reason=break_reason,
                )
            )
            result.total_scheduled_libraries += len(list(getattr(lane, "libraries", []) or []))

        result.fallback_libraries.extend(fallback_libraries)
        result.scheduled_groups += len(bucket_groups)
        if not strong_binding_kept:
            result.broken_groups += len(bucket_groups)
        return result

    def _flatten_group_libraries(
        self,
        groups: List[Round2CandidateGroup],
    ) -> List[EnhancedLibraryInfo]:
        """将分组列表展开为文库列表。"""
        libraries: List[EnhancedLibraryInfo] = []
        for group in groups:
            libraries.extend(list(group.libraries or []))
        return libraries
