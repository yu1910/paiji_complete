"""
1.1模式第二轮候选识别与分组模块
创建时间：2026-04-14 13:40:00
更新时间：2026-04-16 12:25:00

职责：
- 从待排文库中识别 lastlaneround == "1.1第一轮" 且历史测序模式属于 1/1.0/1.1 的第二轮候选
- 按 llastlaneid 分组形成强绑定候选包
- 第二轮不重新排机，直接按历史 llastlaneid 复用第一轮已成型组合
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from arrange_library.models.library_info import EnhancedLibraryInfo
from arrange_library.models.library_info import MachineType
from arrange_library.core.scheduling.scheduling_types import LaneAssignment


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

    按照当前业务口径：
    - lastlaneround == "1.1第一轮" 时，当前这次排机属于第二轮
    - lastlaneround == "1.1第二轮" 时，说明上一轮已结束，当前重新回到第一轮
    - laneround 是当前排机结果字段，由程序根据 lastlaneround 回填，不作为输入识别条件
    - lcxms/lastcxms 属于 1/1.0/1.1，且 llastlaneid 有值
      满足以上条件的文库为第二轮候选
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
            last_lane_round = self._get_last_lane_round(lib)
            last_lid = self._get_last_lane_id(lib)
            seq_mode = self._get_seq_mode(lib)

            if (
                last_lane_round == self._first_round_label
                and last_lid
                and self._is_mode_1_1_family(seq_mode)
            ):
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
        """按历史 llastlaneid 直接生成第二轮 lane。

        业务口径：
        - 第二轮不需要重新排机
        - 同一个 llastlaneid 下的文库直接复用第一轮已成立的组合
        - 程序只负责生成 lane 结果，并在后续 prediction 阶段重算下单量/pooling
        """
        result = Round2SchedulingResult()
        if not candidate_groups:
            return result

        logger.info(
            "1.1第二轮直出Lane: 候选分组={}",
            len(candidate_groups),
        )

        for index, group in enumerate(candidate_groups, start=1):
            lane = self._build_round2_lane(group, lane_index=index)
            result.lanes.append(lane)
            result.scheduled_lane_results.append(
                Round2ScheduledLane(
                    lane=lane,
                    source_last_lane_ids=[group.last_lane_id],
                    strong_binding_kept=True,
                    break_reason="",
                )
            )
            result.scheduled_groups += 1
            result.total_scheduled_libraries += len(group.libraries)

        logger.info(
            "1.1第二轮直出完成: 生成Lane={}, 调度分组={}, 回流3.6T文库={}",
            len(result.lanes),
            result.scheduled_groups,
            len(result.fallback_libraries),
        )
        return result

    def _get_last_lane_round(self, lib: EnhancedLibraryInfo) -> str:
        """获取文库上轮轮次，兼容 lastlaneround/last_lane_round 多种口径。"""
        value = getattr(lib, "last_lane_round", None)
        if not value:
            value = getattr(lib, "lastlaneround", None)
        if not value:
            value = getattr(lib, "_last_lane_round_raw", None)
        return str(value or "").strip()

    def _get_seq_mode(self, lib: EnhancedLibraryInfo) -> str:
        """获取第二轮识别所需的参考测序模式。

        第二轮候选识别优先依据历史模式字段（llastcxms/lastcxms），
        只有历史值缺失时才回退到当前 lcxms。
        """
        for attr_name in (
            "_last_cxms_raw",
            "last_cxms",
            "lastcxms",
            "llastcxms",
            "_current_seq_mode_raw",
            "current_seq_mode",
            "seq_mode",
            "lcxms",
        ):
            value = getattr(lib, attr_name, None)
            if value not in (None, ""):
                return str(value).strip()
        return ""

    def _is_mode_1_1_family(self, value: str) -> bool:
        """判断测序模式是否属于 1.1 模式族。

        兼容历史字段中逗号拼接的模式串；只有所有非空片段均属于 1/1.0/1.1
        时，才认为该历史lane属于 1.1 模式族。
        """
        raw_value = str(value or "").strip()
        if not raw_value:
            return False
        tokens = [item.strip() for item in raw_value.split(",") if item and item.strip()]
        if not tokens:
            return False
        return all(token in {"1", "1.0", "1.1"} for token in tokens)

    def _get_last_lane_id(self, lib: EnhancedLibraryInfo) -> str:
        """获取文库的上次 lane ID（即系统推送的 llastlaneid）。"""
        value = getattr(lib, "last_laneid", None)
        if not value:
            value = getattr(lib, "lastlaneid", None)
        if not value:
            value = getattr(lib, "llastlaneid", None)
        return str(value or "").strip()

    def _get_last_output_rate(self, lib: EnhancedLibraryInfo) -> Optional[float]:
        """获取历史产出率，兼容 dataclass 字段与运行时透传字段。"""
        for attr_name in ("last_outrate", "_last_outrate_raw", "wklastoutrate"):
            value = getattr(lib, attr_name, None)
            if value in (None, ""):
                continue
            try:
                rate = float(value)
                if 0 < rate <= 1.0:
                    rate = rate * 100.0
                return rate
            except (TypeError, ValueError):
                continue
        return None

    def _is_add_test_like(self, lib: EnhancedLibraryInfo) -> bool:
        """判断是否属于规则14的加测/混合文库。"""
        remark = str(getattr(lib, "add_tests_remark", "") or "").strip()
        if not remark:
            return False
        return any(keyword in remark for keyword in ("加测", "混合"))

    def _group_requires_default_pooling(self, group: Round2CandidateGroup) -> bool:
        """判断分组是否需要应用第二轮默认pooling系数。"""
        for lib in group.libraries:
            if not self._is_add_test_like(lib):
                continue
            last_outrate = self._get_last_output_rate(lib)
            if last_outrate is not None and last_outrate < self._output_rate_threshold:
                return True
        return False

    def _get_low_output_origrecs(self, group: Round2CandidateGroup) -> List[str]:
        """返回需要应用第二轮默认pooling系数的文库origrec。"""
        origrecs: List[str] = []
        for lib in group.libraries:
            if not self._is_add_test_like(lib):
                continue
            last_outrate = self._get_last_output_rate(lib)
            if last_outrate is None or last_outrate >= self._output_rate_threshold:
                continue
            origrec = str(getattr(lib, "origrec", "") or "").strip()
            if origrec:
                origrecs.append(origrec)
        return origrecs

    def _build_round2_lane(
        self,
        group: Round2CandidateGroup,
        lane_index: int,
    ) -> LaneAssignment:
        """基于历史 llastlaneid 直接构造第二轮 lane。"""
        libraries = list(group.libraries or [])
        machine_type = self._resolve_machine_type(libraries)
        total_data_gb = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in libraries)
        lane = LaneAssignment(
            lane_id=f"M11R2_{lane_index:03d}",
            machine_id=f"M_M11R2_{lane_index:03d}",
            machine_type=machine_type,
            libraries=libraries,
            total_data_gb=total_data_gb,
            metadata={
                "dispatch_stage": "second_round_1_1",
                "selected_seq_mode": "1.1",
                "selected_round_label": self._second_round_label,
                "mode_1_1_round2_source_last_lane_ids": [group.last_lane_id],
                "mode_1_1_round2_group_count": 1,
                "skip_strict_validation": True,
            },
        )
        lane.calculate_metrics()

        low_output_origrecs = self._get_low_output_origrecs(group)
        if low_output_origrecs:
            lane.metadata["mode_1_1_round2_pooling_factor"] = self._default_pooling_factor
            lane.metadata["mode_1_1_round2_low_output_origrecs"] = sorted(low_output_origrecs)

        return lane

    def _resolve_machine_type(
        self,
        libraries: List[EnhancedLibraryInfo],
    ) -> MachineType:
        """从文库列表中解析机器类型，默认回退 Nova X-25B。"""
        if not libraries:
            return MachineType.NOVA_X_25B

        machine_type = getattr(libraries[0], "machine_type", None)
        if isinstance(machine_type, MachineType):
            return machine_type

        eq_type = str(getattr(libraries[0], "eq_type", "") or "").strip()
        for item in MachineType:
            if item.value == eq_type:
                return item
        return MachineType.NOVA_X_25B
