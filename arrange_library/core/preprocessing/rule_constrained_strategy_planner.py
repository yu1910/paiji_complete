"""
规则约束策略规划器 - 基于全局分析结论制定排机执行计划
接收 BatchAnalysisReport，在规则可行域内决定策略开关与执行顺序，
每个决策附带规则证据，确保不尝试无效策略。

创建时间：2026-03-06 16:35:00
更新时间：2026-03-06 16:35:00
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger

from arrange_library.core.preprocessing.batch_rule_analyzer import (
    BatchAnalysisReport,
    DedicatedLaneRecommendation,
)


@dataclass
class StrategyDecision:
    """单条策略决策"""

    strategy_name: str
    enabled: bool
    priority: int
    rule_evidence: str
    parameters: Dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "启用" if self.enabled else "禁用"
        return f"[P{self.priority}] {self.strategy_name}: {status} | {self.rule_evidence}"


@dataclass
class StrategyExecutionPlan:
    """策略执行计划 - 排机器的完整执行蓝图"""

    decisions: List[StrategyDecision] = field(default_factory=list)

    enable_dedicated_imbalance_lane: bool = False
    enable_non_10bp_dedicated_lane: bool = False
    enable_backbone_reservation: bool = False
    enable_small_library_clustering: bool = True

    execution_order: List[str] = field(default_factory=list)
    estimated_dedicated_lanes: int = 0
    estimated_mixed_lanes_min: int = 0
    estimated_mixed_lanes_max: int = 0

    def summary_lines(self) -> List[str]:
        """生成人类可读的执行计划摘要"""
        lines = [
            "===== 策略执行计划 =====",
            f"预估专用Lane: {self.estimated_dedicated_lanes}",
            f"预估混排Lane: {self.estimated_mixed_lanes_min}~{self.estimated_mixed_lanes_max}",
            "--- 策略决策 ---",
        ]
        sorted_decisions = sorted(self.decisions, key=lambda d: d.priority, reverse=True)
        for d in sorted_decisions:
            lines.append(f"  {d}")
        if self.execution_order:
            lines.append("--- 执行顺序 ---")
            for i, step in enumerate(self.execution_order, 1):
                lines.append(f"  {i}. {step}")
        lines.append("===== 计划结束 =====")
        return lines


class RuleConstrainedStrategyPlanner:
    """规则约束策略规划器

    输入: BatchAnalysisReport (全局分析结论)
    输出: StrategyExecutionPlan (执行计划)

    设计原则:
    - 每个决策必须有规则证据支撑
    - 只在可行域内做选择，禁止尝试已知违规的策略
    - 优先级越高的策略越先执行
    """

    _STRATEGY_DEDICATED_IMBALANCE: str = "dedicated_imbalance_lane"
    _STRATEGY_DEDICATED_NON_10BP: str = "dedicated_non_10bp_lane"
    _STRATEGY_BACKBONE_RESERVATION: str = "backbone_reservation"
    _STRATEGY_SMALL_CLUSTERING: str = "small_library_clustering"
    _STRATEGY_PACKAGE_LANE: str = "package_lane_first"
    _STRATEGY_SPECIAL_COMBO_SPLIT: str = "special_combo_split"

    def plan(self, report: BatchAnalysisReport) -> StrategyExecutionPlan:
        """根据分析报告制定执行计划

        Args:
            report: 批次全局分析报告

        Returns:
            StrategyExecutionPlan: 策略执行计划
        """
        plan = StrategyExecutionPlan()

        self._decide_package_lane(report, plan)
        self._decide_special_combo_split(report, plan)
        self._decide_dedicated_imbalance(report, plan)
        self._decide_dedicated_non_10bp(report, plan)
        self._decide_backbone_reservation(report, plan)
        self._decide_small_clustering(report, plan)
        self._build_execution_order(plan)
        self._estimate_lane_counts(report, plan)

        for line in plan.summary_lines():
            logger.info(line)

        return plan

    def _decide_package_lane(
        self, report: BatchAnalysisReport, plan: StrategyExecutionPlan
    ) -> None:
        """决定包Lane优先策略"""
        if report.has_package_lane_libs:
            gp = report.group_profiles.get("package_lane")
            count = gp.library_count if gp else 0
            plan.decisions.append(
                StrategyDecision(
                    strategy_name=self._STRATEGY_PACKAGE_LANE,
                    enabled=True,
                    priority=100,
                    rule_evidence=f"存在{count}个包Lane/包FC文库，规则要求独占Lane优先处理",
                )
            )
        else:
            plan.decisions.append(
                StrategyDecision(
                    strategy_name=self._STRATEGY_PACKAGE_LANE,
                    enabled=False,
                    priority=100,
                    rule_evidence="无包Lane/包FC文库，跳过",
                )
            )

    def _decide_special_combo_split(
        self, report: BatchAnalysisReport, plan: StrategyExecutionPlan
    ) -> None:
        """决定特殊10X组合分离策略"""
        has_mutual_exclusive = any(
            mc.constraint_type == "mutually_exclusive_sample_type_groups"
            for mc in report.mix_constraints
        )
        gp_a = report.group_profiles.get("special_combo_a")
        gp_b = report.group_profiles.get("special_combo_b")

        if has_mutual_exclusive and gp_a and gp_b:
            plan.decisions.append(
                StrategyDecision(
                    strategy_name=self._STRATEGY_SPECIAL_COMBO_SPLIT,
                    enabled=True,
                    priority=90,
                    rule_evidence=(
                        f"5端/VDJ({gp_a.library_count}个, {gp_a.total_data_gb:.0f}GB) 与 "
                        f"3端/Visium({gp_b.library_count}个, {gp_b.total_data_gb:.0f}GB) "
                        f"互斥，必须分开排入不同Lane"
                    ),
                )
            )
        elif gp_a and gp_a.library_count > 0 and not gp_b:
            plan.decisions.append(
                StrategyDecision(
                    strategy_name=self._STRATEGY_SPECIAL_COMBO_SPLIT,
                    enabled=False,
                    priority=90,
                    rule_evidence="仅有5端/VDJ文库，无互斥冲突，无需分离",
                )
            )
        elif gp_b and gp_b.library_count > 0 and not gp_a:
            plan.decisions.append(
                StrategyDecision(
                    strategy_name=self._STRATEGY_SPECIAL_COMBO_SPLIT,
                    enabled=False,
                    priority=90,
                    rule_evidence="仅有3端/Visium文库，无互斥冲突，无需分离",
                )
            )

    def _decide_dedicated_imbalance(
        self, report: BatchAnalysisReport, plan: StrategyExecutionPlan
    ) -> None:
        """决定碱基不均衡专用Lane策略"""
        if not report.has_imbalance_libs:
            plan.enable_dedicated_imbalance_lane = False
            plan.decisions.append(
                StrategyDecision(
                    strategy_name=self._STRATEGY_DEDICATED_IMBALANCE,
                    enabled=False,
                    priority=80,
                    rule_evidence="无碱基不均衡文库，无需专用Lane",
                )
            )
            return

        gp_imb = report.group_profiles.get("imbalance")
        if not gp_imb:
            plan.enable_dedicated_imbalance_lane = False
            return

        rec = self._find_recommendation(report, "imbalance")
        ratio_warning = any(
            mc.constraint_type == "imbalance_ratio_warning"
            for mc in report.mix_constraints
        )

        if rec and rec.confidence in ("high", "medium"):
            plan.enable_dedicated_imbalance_lane = True
            plan.decisions.append(
                StrategyDecision(
                    strategy_name=self._STRATEGY_DEDICATED_IMBALANCE,
                    enabled=True,
                    priority=80,
                    rule_evidence=(
                        f"碱基不均衡文库{gp_imb.library_count}个共{gp_imb.total_data_gb:.0f}GB，"
                        f"可形成约{rec.estimated_lanes}条专用Lane（置信度{rec.confidence}）"
                    ),
                    parameters={"estimated_dedicated_lanes": rec.estimated_lanes},
                )
            )
        elif ratio_warning:
            plan.enable_dedicated_imbalance_lane = True
            plan.decisions.append(
                StrategyDecision(
                    strategy_name=self._STRATEGY_DEDICATED_IMBALANCE,
                    enabled=True,
                    priority=80,
                    rule_evidence=(
                        f"碱基不均衡全局占比超过30%红线，"
                        f"必须拆分为专用Lane以满足混排规则"
                    ),
                )
            )
        else:
            plan.enable_dedicated_imbalance_lane = False
            plan.decisions.append(
                StrategyDecision(
                    strategy_name=self._STRATEGY_DEDICATED_IMBALANCE,
                    enabled=False,
                    priority=80,
                    rule_evidence=(
                        f"碱基不均衡文库{gp_imb.total_data_gb:.0f}GB，"
                        f"不足以形成专用Lane，将混入散样"
                    ),
                )
            )

    def _decide_dedicated_non_10bp(
        self, report: BatchAnalysisReport, plan: StrategyExecutionPlan
    ) -> None:
        """决定非10bp专用Lane策略"""
        if not report.has_non_10bp_libs:
            plan.enable_non_10bp_dedicated_lane = False
            plan.decisions.append(
                StrategyDecision(
                    strategy_name=self._STRATEGY_DEDICATED_NON_10BP,
                    enabled=False,
                    priority=70,
                    rule_evidence="无非10bp文库，无需专用Lane",
                )
            )
            return

        gp_non10 = report.group_profiles.get("non_10bp")
        if not gp_non10:
            plan.enable_non_10bp_dedicated_lane = False
            return

        ratio_warning = any(
            mc.constraint_type == "10bp_ratio_warning"
            for mc in report.mix_constraints
        )
        rec = self._find_recommendation(report, "non_10bp")

        if ratio_warning and rec and rec.confidence in ("high", "medium"):
            plan.enable_non_10bp_dedicated_lane = True
            plan.decisions.append(
                StrategyDecision(
                    strategy_name=self._STRATEGY_DEDICATED_NON_10BP,
                    enabled=True,
                    priority=70,
                    rule_evidence=(
                        f"10bp全局占比不足40%，非10bp文库{gp_non10.total_data_gb:.0f}GB"
                        f"可形成约{rec.estimated_lanes}条专用Lane，"
                        f"避免混排违反10bp>=40%红线"
                    ),
                    parameters={"estimated_dedicated_lanes": rec.estimated_lanes},
                )
            )
        elif rec and rec.confidence == "high":
            plan.enable_non_10bp_dedicated_lane = True
            plan.decisions.append(
                StrategyDecision(
                    strategy_name=self._STRATEGY_DEDICATED_NON_10BP,
                    enabled=True,
                    priority=70,
                    rule_evidence=(
                        f"非10bp文库{gp_non10.total_data_gb:.0f}GB数据量充足，"
                        f"走专用Lane可提升排机效率"
                    ),
                )
            )
        else:
            plan.enable_non_10bp_dedicated_lane = False
            plan.decisions.append(
                StrategyDecision(
                    strategy_name=self._STRATEGY_DEDICATED_NON_10BP,
                    enabled=False,
                    priority=70,
                    rule_evidence=(
                        f"非10bp文库{gp_non10.total_data_gb:.0f}GB，"
                        f"不足以形成专用Lane或10bp占比未触发红线"
                    ),
                )
            )

    def _decide_backbone_reservation(
        self, report: BatchAnalysisReport, plan: StrategyExecutionPlan
    ) -> None:
        """决定骨架预留(大带小)策略"""
        high_tail_risks = [r for r in report.tail_risks if r.severity == "high"]
        medium_tail_risks = [r for r in report.tail_risks if r.severity == "medium"]

        if high_tail_risks:
            plan.enable_backbone_reservation = True
            tags = ", ".join(r.group_tag for r in high_tail_risks)
            plan.decisions.append(
                StrategyDecision(
                    strategy_name=self._STRATEGY_BACKBONE_RESERVATION,
                    enabled=True,
                    priority=60,
                    rule_evidence=(
                        f"存在{len(high_tail_risks)}个高风险尾部分组({tags})，"
                        f"需要预留骨架文库携带尾部碎片"
                    ),
                )
            )
        elif medium_tail_risks:
            plan.enable_backbone_reservation = True
            plan.decisions.append(
                StrategyDecision(
                    strategy_name=self._STRATEGY_BACKBONE_RESERVATION,
                    enabled=True,
                    priority=60,
                    rule_evidence=(
                        f"存在{len(medium_tail_risks)}个中等尾部风险，"
                        f"启用骨架预留以降低排不上风险"
                    ),
                )
            )
        else:
            plan.enable_backbone_reservation = False
            plan.decisions.append(
                StrategyDecision(
                    strategy_name=self._STRATEGY_BACKBONE_RESERVATION,
                    enabled=False,
                    priority=60,
                    rule_evidence="无显著尾部风险，无需骨架预留",
                )
            )

    def _decide_small_clustering(
        self, report: BatchAnalysisReport, plan: StrategyExecutionPlan
    ) -> None:
        """决定小文库聚类策略 - 默认启用"""
        plan.enable_small_library_clustering = True
        plan.decisions.append(
            StrategyDecision(
                strategy_name=self._STRATEGY_SMALL_CLUSTERING,
                enabled=True,
                priority=50,
                rule_evidence="小文库聚类始终启用，可提升同尺寸小文库Lane利用率",
            )
        )

    def _build_execution_order(self, plan: StrategyExecutionPlan) -> None:
        """根据已启用策略的优先级构建执行顺序"""
        enabled = [d for d in plan.decisions if d.enabled]
        enabled.sort(key=lambda d: d.priority, reverse=True)
        plan.execution_order = [d.strategy_name for d in enabled]

    def _estimate_lane_counts(
        self, report: BatchAnalysisReport, plan: StrategyExecutionPlan
    ) -> None:
        """估算专用Lane和混排Lane数"""
        dedicated_total = 0
        for d in plan.decisions:
            if d.enabled and "estimated_dedicated_lanes" in d.parameters:
                dedicated_total += int(d.parameters["estimated_dedicated_lanes"])
        plan.estimated_dedicated_lanes = dedicated_total

        remaining_min = max(0, report.estimated_total_lanes_min - dedicated_total)
        remaining_max = max(0, report.estimated_total_lanes_max - dedicated_total)
        plan.estimated_mixed_lanes_min = remaining_min
        plan.estimated_mixed_lanes_max = remaining_max

    @staticmethod
    def _find_recommendation(
        report: BatchAnalysisReport, group_tag: str
    ) -> Optional[DedicatedLaneRecommendation]:
        """查找指定分组的专用Lane推荐"""
        for rec in report.dedicated_lane_recommendations:
            if rec.group_tag == group_tag:
                return rec
        return None
