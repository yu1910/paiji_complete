"""
批次规则分析器 - 全局数据洞察与规则约束域分析
在排机前对整批文库进行全局特征分析，结合统一规则表，
输出可行域结论，指导后续调度策略选择。

创建时间：2026-03-06 16:22:00
更新时间：2026-03-06 16:50:00
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger


@dataclass
class LibraryGroupProfile:
    """单个文库分组的画像"""

    group_tag: str
    description: str
    library_count: int = 0
    total_data_gb: float = 0.0
    library_ids: List[str] = field(default_factory=list)

    estimated_lane_count_min: int = 0
    estimated_lane_count_max: int = 0
    capacity_rule_code: str = ""
    effective_min_gb: float = 0.0
    effective_max_gb: float = 0.0


@dataclass
class DedicatedLaneRecommendation:
    """专用Lane推荐"""

    group_tag: str
    reason: str
    estimated_lanes: int = 0
    total_data_gb: float = 0.0
    library_count: int = 0
    confidence: str = "medium"


@dataclass
class MixConstraint:
    """混排约束"""

    constraint_type: str
    group_a: str
    group_b: str
    rule_code: str
    message: str


@dataclass
class TailRiskItem:
    """尾部风险项"""

    group_tag: str
    reason: str
    remaining_data_gb: float = 0.0
    library_count: int = 0
    severity: str = "low"


@dataclass
class BatchAnalysisReport:
    """批次全局分析报告"""

    total_library_count: int = 0
    total_data_gb: float = 0.0
    machine_type: str = ""

    group_profiles: Dict[str, LibraryGroupProfile] = field(default_factory=dict)
    dedicated_lane_recommendations: List[DedicatedLaneRecommendation] = field(
        default_factory=list
    )
    mix_constraints: List[MixConstraint] = field(default_factory=list)
    tail_risks: List[TailRiskItem] = field(default_factory=list)

    estimated_total_lanes_min: int = 0
    estimated_total_lanes_max: int = 0
    has_package_lane_libs: bool = False
    has_imbalance_libs: bool = False
    has_customer_libs: bool = False
    has_10bp_libs: bool = False
    has_non_10bp_libs: bool = False
    has_special_combo_libs: bool = False

    strategy_hints: List[str] = field(default_factory=list)

    def summary_lines(self) -> List[str]:
        """生成人类可读的摘要文本行"""
        lines = [
            f"===== 批次全局分析报告 =====",
            f"机器类型: {self.machine_type}",
            f"文库总数: {self.total_library_count}",
            f"总数据量: {self.total_data_gb:.1f} GB",
            f"预估Lane数: {self.estimated_total_lanes_min}~{self.estimated_total_lanes_max}",
            f"--- 特征标记 ---",
            f"  包Lane文库: {'有' if self.has_package_lane_libs else '无'}",
            f"  碱基不均衡: {'有' if self.has_imbalance_libs else '无'}",
            f"  客户文库:   {'有' if self.has_customer_libs else '无'}",
            f"  10bp文库:   {'有' if self.has_10bp_libs else '无'}",
            f"  非10bp文库: {'有' if self.has_non_10bp_libs else '无'}",
            f"  特殊组合:   {'有' if self.has_special_combo_libs else '无'}",
        ]
        if self.group_profiles:
            lines.append("--- 分组画像 ---")
            for tag, gp in self.group_profiles.items():
                lines.append(
                    f"  [{tag}] {gp.description}: "
                    f"{gp.library_count}个文库, {gp.total_data_gb:.1f}GB, "
                    f"预估{gp.estimated_lane_count_min}~{gp.estimated_lane_count_max}条Lane"
                )
        if self.dedicated_lane_recommendations:
            lines.append("--- 专用Lane推荐 ---")
            for rec in self.dedicated_lane_recommendations:
                lines.append(
                    f"  [{rec.group_tag}] {rec.reason} - "
                    f"约{rec.estimated_lanes}条Lane, {rec.total_data_gb:.1f}GB, "
                    f"置信度: {rec.confidence}"
                )
        if self.mix_constraints:
            lines.append("--- 混排约束 ---")
            for mc in self.mix_constraints:
                lines.append(
                    f"  [{mc.rule_code}] {mc.group_a} vs {mc.group_b}: {mc.message}"
                )
        if self.tail_risks:
            lines.append("--- 尾部风险 ---")
            for tr in self.tail_risks:
                lines.append(
                    f"  [{tr.group_tag}] {tr.reason} - "
                    f"{tr.remaining_data_gb:.1f}GB, {tr.library_count}个, "
                    f"严重程度: {tr.severity}"
                )
        if self.strategy_hints:
            lines.append("--- 策略建议 ---")
            for hint in self.strategy_hints:
                lines.append(f"  - {hint}")
        lines.append("===== 报告结束 =====")
        return lines


class BatchRuleAnalyzer:
    """批次规则分析器

    在排机前对整批文库做全局特征分析，结合统一规则表，
    输出分组画像、专用Lane推荐、混排约束、尾部风险等结论。
    """

    _IMBALANCE_FIELD: str = "jjbj"
    _IMBALANCE_YES: str = "是"
    _10BP_SEQ_LEN: int = 10

    def __init__(self, scheduling_config: Any) -> None:
        self._scheduling_config = scheduling_config
        self._rule_matrix: Dict[str, Any] = {}
        if hasattr(scheduling_config, "_rule_matrix_config"):
            self._rule_matrix = scheduling_config._rule_matrix_config or {}

    def _is_10bp_library(self, lib: Any) -> bool:
        """判定文库是否为10bp文库（与LaneValidator._is_10bp_index逻辑一致）

        优先检查 ten_bp_data 字段，其次按 Index 序列 P7 端长度判定。
        """
        ten_bp_data = getattr(lib, "ten_bp_data", None)
        if ten_bp_data is not None and float(ten_bp_data or 0) > 0:
            return True
        index_seq = str(getattr(lib, "index_seq", "") or "")
        if not index_seq:
            return False
        for seq in index_seq.split(","):
            p7_seq = seq.split(";")[0].strip() if ";" in seq else seq.strip()
            if p7_seq.upper() not in ("PE", "通用接头", "随机INDEX"):
                if len(p7_seq) == self._10BP_SEQ_LEN:
                    return True
        return False

    def analyze(
        self,
        libraries: List[Any],
        machine_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchAnalysisReport:
        """执行全局分析

        Args:
            libraries: 待排机文库列表
            machine_type: 机器类型
            metadata: 额外元数据

        Returns:
            BatchAnalysisReport: 分析报告
        """
        metadata = metadata or {}
        report = BatchAnalysisReport(
            total_library_count=len(libraries),
            total_data_gb=sum(
                float(getattr(lib, "contract_data_raw", 0) or 0) for lib in libraries
            ),
            machine_type=machine_type,
        )

        if not libraries:
            report.strategy_hints.append("无文库输入，跳过分析")
            return report

        groups = self._classify_libraries(libraries)
        self._build_group_profiles(groups, report, machine_type, metadata)
        self._set_feature_flags(groups, report)
        self._detect_dedicated_lane_opportunities(report)
        self._detect_mix_constraints(report, libraries, machine_type, metadata)
        self._assess_tail_risk(report)
        self._estimate_total_lanes(report)
        self._generate_strategy_hints(report)

        return report

    def _classify_libraries(
        self, libraries: List[Any]
    ) -> Dict[str, List[Any]]:
        """将文库按业务特征分组"""
        groups: Dict[str, List[Any]] = defaultdict(list)

        sample_type_groups = self._rule_matrix.get("sample_type_groups", {})
        combo_a_types = set(
            sample_type_groups.get("special_combo_group_a", {}).get("sample_types", [])
        )
        combo_b_types = set(
            sample_type_groups.get("special_combo_group_b", {}).get("sample_types", [])
        )

        for lib in libraries:
            data_gb = float(getattr(lib, "contract_data_raw", 0) or 0)
            sample_type = str(getattr(lib, "sample_type_code", "") or "")
            jjbj = str(getattr(lib, self._IMBALANCE_FIELD, "") or "").strip()
            customer_lib = str(getattr(lib, "customer_library", "") or "").strip()
            is_package = bool(
                str(getattr(lib, "is_package_lane", "") or "").strip() == "是"
                or str(getattr(lib, "package_lane_number", "") or "").strip()
                or str(getattr(lib, "package_fc_number", "") or "").strip()
            )

            if is_package:
                groups["package_lane"].append(lib)
            if jjbj == self._IMBALANCE_YES:
                groups["imbalance"].append(lib)
            if customer_lib == "是" or str(
                getattr(lib, "lab_type", "") or ""
            ).startswith("FKDL"):
                groups["customer"].append(lib)

            is_10bp = self._is_10bp_library(lib)
            if is_10bp:
                groups["10bp"].append(lib)
            else:
                groups["non_10bp"].append(lib)

            if sample_type in combo_a_types:
                groups["special_combo_a"].append(lib)
            elif sample_type in combo_b_types:
                groups["special_combo_b"].append(lib)

            groups["all"].append(lib)

        return dict(groups)

    def _build_group_profiles(
        self,
        groups: Dict[str, List[Any]],
        report: BatchAnalysisReport,
        machine_type: str,
        metadata: Dict[str, Any],
    ) -> None:
        """为每个分组构建画像"""
        descriptions = {
            "all": "全部文库",
            "package_lane": "包Lane/包FC文库",
            "imbalance": "碱基不均衡文库",
            "customer": "客户文库",
            "10bp": "10碱基文库",
            "non_10bp": "非10碱基文库",
            "special_combo_a": "10X 5端/VDJ组合",
            "special_combo_b": "10X 3端/Visium组合",
        }

        for tag, libs in groups.items():
            total_data = sum(
                float(getattr(lib, "contract_data_raw", 0) or 0) for lib in libs
            )
            lib_ids = [str(getattr(lib, "origrec", "") or "") for lib in libs]

            cap_rule_code = ""
            eff_min = 0.0
            eff_max = 0.0
            if libs:
                try:
                    rule_sel = self._scheduling_config.get_lane_capacity_range(
                        libraries=libs[:1],
                        machine_type=machine_type,
                        metadata=metadata,
                    )
                    cap_rule_code = rule_sel.rule_code
                    eff_min = rule_sel.effective_min_gb
                    eff_max = rule_sel.effective_max_gb
                except Exception:
                    pass

            est_min, est_max = self._estimate_lane_count(total_data, eff_min, eff_max)

            profile = LibraryGroupProfile(
                group_tag=tag,
                description=descriptions.get(tag, tag),
                library_count=len(libs),
                total_data_gb=total_data,
                library_ids=lib_ids,
                estimated_lane_count_min=est_min,
                estimated_lane_count_max=est_max,
                capacity_rule_code=cap_rule_code,
                effective_min_gb=eff_min,
                effective_max_gb=eff_max,
            )
            report.group_profiles[tag] = profile

    @staticmethod
    def _estimate_lane_count(
        total_data: float, eff_min: float, eff_max: float
    ) -> Tuple[int, int]:
        """根据总数据和容量区间估算Lane数"""
        if eff_max <= 0:
            return (0, 0)
        est_min = max(1, math.ceil(total_data / eff_max))
        est_max = max(1, math.ceil(total_data / max(eff_min, 1.0)))
        return (est_min, est_max)

    @staticmethod
    def _set_feature_flags(
        groups: Dict[str, List[Any]], report: BatchAnalysisReport
    ) -> None:
        """设置批次特征标记"""
        report.has_package_lane_libs = bool(groups.get("package_lane"))
        report.has_imbalance_libs = bool(groups.get("imbalance"))
        report.has_customer_libs = bool(groups.get("customer"))
        report.has_10bp_libs = bool(groups.get("10bp"))
        report.has_non_10bp_libs = bool(groups.get("non_10bp"))
        report.has_special_combo_libs = bool(
            groups.get("special_combo_a") or groups.get("special_combo_b")
        )

    def _detect_dedicated_lane_opportunities(
        self, report: BatchAnalysisReport
    ) -> None:
        """检测可形成专用Lane的分组"""
        for tag in ("imbalance", "package_lane", "non_10bp", "special_combo_a", "special_combo_b"):
            gp = report.group_profiles.get(tag)
            if not gp or gp.library_count == 0:
                continue

            if gp.effective_min_gb <= 0:
                continue

            if gp.total_data_gb >= gp.effective_min_gb:
                est_lanes = max(1, math.floor(gp.total_data_gb / gp.effective_min_gb))
                confidence = "high" if gp.total_data_gb >= gp.effective_min_gb * 1.5 else "medium"
                reason_map = {
                    "imbalance": "碱基不均衡文库数据量充足，可包专用Lane避免混排占比限制",
                    "package_lane": "包Lane/包FC文库应优先独占Lane",
                    "non_10bp": "非10bp文库集中排入专用Lane，避免触发10bp>=40%混排规则",
                    "special_combo_a": "10X 5端/VDJ文库可独占Lane（与3端互斥）",
                    "special_combo_b": "10X 3端/Visium文库可独占Lane（与5端互斥）",
                }
                report.dedicated_lane_recommendations.append(
                    DedicatedLaneRecommendation(
                        group_tag=tag,
                        reason=reason_map.get(tag, "数据量充足可形成专用Lane"),
                        estimated_lanes=est_lanes,
                        total_data_gb=gp.total_data_gb,
                        library_count=gp.library_count,
                        confidence=confidence,
                    )
                )

    def _detect_mix_constraints(
        self,
        report: BatchAnalysisReport,
        libraries: List[Any],
        machine_type: str,
        metadata: Dict[str, Any],
    ) -> None:
        """检测混排约束"""
        gp_a = report.group_profiles.get("special_combo_a")
        gp_b = report.group_profiles.get("special_combo_b")
        has_a = bool(gp_a and gp_a.library_count > 0)
        has_b = bool(gp_b and gp_b.library_count > 0)
        if has_a and has_b:
            report.mix_constraints.append(
                MixConstraint(
                    constraint_type="mutually_exclusive_sample_type_groups",
                    group_a="special_combo_a",
                    group_b="special_combo_b",
                    rule_code="tj_1595_special_combo_groups_not_mixed",
                    message="10X 5端/VDJ 与 10X 3端/Visium 文库不可混排，必须分开",
                )
            )

        if report.has_10bp_libs and report.has_non_10bp_libs:
            gp_10 = report.group_profiles.get("10bp")
            gp_non10 = report.group_profiles.get("non_10bp")
            if gp_10 and gp_non10:
                total = gp_10.total_data_gb + gp_non10.total_data_gb
                ratio_10 = gp_10.total_data_gb / total if total > 0 else 0
                if ratio_10 < 0.40:
                    report.mix_constraints.append(
                        MixConstraint(
                            constraint_type="10bp_ratio_warning",
                            group_a="10bp",
                            group_b="non_10bp",
                            rule_code="10bp_min_40pct",
                            message=(
                                f"10bp数据量全局占比仅{ratio_10:.1%}，"
                                f"不足40%红线，若全部混排将违规，"
                                f"建议非10bp文库走专用Lane"
                            ),
                        )
                    )

        if report.has_imbalance_libs:
            gp_imb = report.group_profiles.get("imbalance")
            gp_all = report.group_profiles.get("all")
            if gp_imb and gp_all and gp_all.total_data_gb > 0:
                ratio_imb = gp_imb.total_data_gb / gp_all.total_data_gb
                if ratio_imb > 0.30:
                    report.mix_constraints.append(
                        MixConstraint(
                            constraint_type="imbalance_ratio_warning",
                            group_a="imbalance",
                            group_b="all",
                            rule_code="imbalance_max_30pct",
                            message=(
                                f"碱基不均衡文库全局占比{ratio_imb:.1%}，"
                                f"超过30%上限，必须拆分为专用Lane"
                            ),
                        )
                    )

    def _assess_tail_risk(self, report: BatchAnalysisReport) -> None:
        """评估尾部风险"""
        for tag in ("imbalance", "non_10bp", "special_combo_a", "special_combo_b"):
            gp = report.group_profiles.get(tag)
            if not gp or gp.library_count == 0 or gp.effective_max_gb <= 0:
                continue

            remainder = gp.total_data_gb % gp.effective_min_gb if gp.effective_min_gb > 0 else 0
            if 0 < remainder < gp.effective_min_gb * 0.5:
                severity = "high" if remainder < gp.effective_min_gb * 0.2 else "medium"
                report.tail_risks.append(
                    TailRiskItem(
                        group_tag=tag,
                        reason=(
                            f"{gp.description}排完整Lane后剩余{remainder:.1f}GB，"
                            f"不足半Lane({gp.effective_min_gb * 0.5:.0f}GB)，"
                            f"需要大文库骨架携带或混入散样Lane"
                        ),
                        remaining_data_gb=remainder,
                        library_count=gp.library_count,
                        severity=severity,
                    )
                )

    def _estimate_total_lanes(self, report: BatchAnalysisReport) -> None:
        """估算总Lane数"""
        gp_all = report.group_profiles.get("all")
        if gp_all:
            report.estimated_total_lanes_min = gp_all.estimated_lane_count_min
            report.estimated_total_lanes_max = gp_all.estimated_lane_count_max

    def _generate_strategy_hints(self, report: BatchAnalysisReport) -> None:
        """生成策略建议"""
        if report.has_package_lane_libs:
            report.strategy_hints.append(
                "存在包Lane/包FC文库，应最先处理，独占Lane"
            )

        if report.dedicated_lane_recommendations:
            tags = [r.group_tag for r in report.dedicated_lane_recommendations]
            if "imbalance" in tags:
                report.strategy_hints.append(
                    "碱基不均衡文库数据量充足，优先形成专用Lane"
                )
            if "non_10bp" in tags:
                report.strategy_hints.append(
                    "非10bp文库建议走专用Lane，避免10bp>=40%混排红线"
                )
            if "special_combo_a" in tags or "special_combo_b" in tags:
                report.strategy_hints.append(
                    "特殊10X组合文库应分别走专用Lane（互斥规则）"
                )

        if report.mix_constraints:
            for mc in report.mix_constraints:
                if mc.constraint_type == "10bp_ratio_warning":
                    report.strategy_hints.append(
                        "10bp全局占比不足40%，需要非10bp专用Lane或骨架预留策略"
                    )
                    break

        if report.tail_risks:
            high_risks = [r for r in report.tail_risks if r.severity == "high"]
            if high_risks:
                report.strategy_hints.append(
                    f"存在{len(high_risks)}个高风险尾部分组，建议启用骨架预留(大带小)"
                )

        if not report.strategy_hints:
            report.strategy_hints.append(
                "数据分布较均匀，可按常规贪心排机流程处理"
            )
