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
    ) -> None:
        """第二轮排机入口（首版空实现，预留接口）

        后续迭代需要补充：
        - 对每个 candidate_group 调度成 lane
        - 处理 pooling 2.5 特例（加测产出率 < 40%）
        - 处理二轮平衡文库口径（按下单数据量占比添加，合同量=下单量）
        """
        if not candidate_groups:
            return

        logger.info(
            "1.1第二轮排机: 当前为框架预留，共{}个分组待排（后续迭代接入真实排机）",
            len(candidate_groups),
        )

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
