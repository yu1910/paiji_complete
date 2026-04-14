"""
1.1模式首轮分流器
创建时间：2026-04-14 13:30:00
更新时间：2026-04-14 13:30:00

负责将AI可排文库按规则切分为 3.6T-NEW 优先池、1.1 首轮池、1.1禁排回退池。
当前版本采用确定性规则，接口设计成将来可挂接AI决策器。

输入：全量AI可排文库 + 1.1配置
输出：ModeDispatchResult（包含各池子和每个文库的分流原因）
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from loguru import logger

from arrange_library.models.library_info import EnhancedLibraryInfo


@dataclass
class ModeDispatchResult:
    """模式分流结果"""
    pool_36t_priority: List[EnhancedLibraryInfo] = field(default_factory=list)
    pool_1_1_normal: List[EnhancedLibraryInfo] = field(default_factory=list)
    pool_1_1_quality_risk: List[EnhancedLibraryInfo] = field(default_factory=list)
    pool_1_1_quality_other: List[EnhancedLibraryInfo] = field(default_factory=list)
    pool_1_1_forbidden: List[EnhancedLibraryInfo] = field(default_factory=list)
    dispatch_reasons: Dict[str, str] = field(default_factory=dict)


class ModeAllocator:
    """首轮模式分流器

    按照1.1模式业务规则，将AI可排文库切分为多个池子：
    - 3.6T-NEW 优先池（临检/YC/SJ需要优先占用3.6T-NEW lane的文库）
    - 1.1 质量正常池（合格+正常建库，优先单独成lane）
    - 1.1 质量风险池（风险/不合格+风险建库，优先单独成lane）
    - 1.1 兜底池（不满足上述两个质量条件的1.1可排文库）
    - 1.1 禁排回退池（不满足1.1可排条件的文库，回退到3.6T-NEW）
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._contract_limit = float(config.get("single_library_contract_limit_gb", 500))
        self._priority_data_types = set(config.get("priority_data_types_for_36t", []))
        self._eligible_data_types = set(config.get("eligible_data_types_for_1_1", []))
        self._eligible_prefixes = [
            p.upper() for p in config.get("eligible_sample_prefixes_for_1_1", [])
        ]
        self._eligible_add_test_kw = set(config.get("eligible_add_test_keywords_for_1_1", []))
        self._lane_limit_rules = config.get("priority_36t_lane_limit_rules", [])
        qg = config.get("quality_grouping", {})
        ng = qg.get("normal_group", {})
        rg = qg.get("risk_group", {})
        self._normal_complex = set(ng.get("complex_results", []))
        self._normal_risk_flags = set(ng.get("risk_build_flags", []))
        self._risk_complex = set(rg.get("complex_results", []))
        self._risk_flags = set(rg.get("risk_build_flags", []))

    def allocate(
        self,
        libraries: List[EnhancedLibraryInfo],
    ) -> ModeDispatchResult:
        """执行首轮模式分流

        Args:
            libraries: 全量AI可排文库（不含包Lane文库）

        Returns:
            ModeDispatchResult 包含各池子和分流原因
        """
        result = ModeDispatchResult()

        priority_candidates: List[EnhancedLibraryInfo] = []
        eligible_for_1_1: List[EnhancedLibraryInfo] = []

        for lib in libraries:
            origrec = getattr(lib, "origrec", "") or ""
            reason = self._check_1_1_forbidden(lib)
            if reason:
                result.pool_1_1_forbidden.append(lib)
                result.dispatch_reasons[origrec] = reason
                continue

            if self._is_priority_for_36t(lib):
                priority_candidates.append(lib)
                result.dispatch_reasons[origrec] = "priority_data_type_for_36t"
            elif self._is_eligible_for_1_1(lib):
                eligible_for_1_1.append(lib)
                result.dispatch_reasons[origrec] = "eligible_for_1_1"
            else:
                result.pool_1_1_forbidden.append(lib)
                result.dispatch_reasons[origrec] = "not_eligible_for_1_1_fallback_36t"

        priority_total_gb = sum(float(lib.contract_data_raw or 0) for lib in priority_candidates)
        max_36t_lanes = self._resolve_priority_36t_lane_count(priority_total_gb)
        logger.info(
            "模式分流: 临检/YC/SJ数据量={:.1f}G, 建议3.6T-NEW lane上限={}, "
            "1.1可排文库={}, 1.1禁排回退={}",
            priority_total_gb, max_36t_lanes,
            len(eligible_for_1_1), len(result.pool_1_1_forbidden),
        )

        result.pool_36t_priority = priority_candidates

        for lib in eligible_for_1_1:
            origrec = getattr(lib, "origrec", "") or ""
            cr = str(getattr(lib, "complex_result", "") or "").strip()
            rf = str(getattr(lib, "risk_build_flag", "") or "").strip()

            if cr in self._normal_complex and rf in self._normal_risk_flags:
                result.pool_1_1_normal.append(lib)
                result.dispatch_reasons[origrec] = "1_1_quality_normal"
            elif cr in self._risk_complex and rf in self._risk_flags:
                result.pool_1_1_quality_risk.append(lib)
                result.dispatch_reasons[origrec] = "1_1_quality_risk"
            else:
                result.pool_1_1_quality_other.append(lib)
                result.dispatch_reasons[origrec] = "1_1_quality_other"

        self._warn_cross_mode_sample_split(result)

        logger.info(
            "模式分流完成: 3.6T-NEW优先池={}, 1.1正常池={}, 1.1风险池={}, "
            "1.1兜底池={}, 禁排回退={}",
            len(result.pool_36t_priority),
            len(result.pool_1_1_normal),
            len(result.pool_1_1_quality_risk),
            len(result.pool_1_1_quality_other),
            len(result.pool_1_1_forbidden),
        )
        return result

    def _check_1_1_forbidden(self, lib: EnhancedLibraryInfo) -> str:
        """检查文库是否禁止进入1.1模式，返回禁排原因（空字符串表示不禁排）"""
        contract = float(getattr(lib, "contract_data_raw", 0) or 0)
        if contract > self._contract_limit:
            return "contract_data_exceeds_{}g".format(int(self._contract_limit))

        baleno = str(getattr(lib, "package_lane_number", "") or "").strip()
        bagfcno = str(getattr(lib, "bagfcno", "") or "").strip()
        if baleno or bagfcno:
            return "has_package_lane_or_fc"

        special_splits = str(getattr(lib, "special_splits", "") or "").strip()
        if special_splits and special_splits.lower() not in ("", "nan", "none"):
            return "has_special_split"

        return ""

    def _is_priority_for_36t(self, lib: EnhancedLibraryInfo) -> bool:
        """判断文库是否属于临检/YC/SJ优先走3.6T-NEW的类型"""
        dt = str(getattr(lib, "data_type", "") or "").strip()
        return dt in self._priority_data_types

    def _is_eligible_for_1_1(self, lib: EnhancedLibraryInfo) -> bool:
        """判断文库是否满足1.1模式使用条件

        使用条件（三个中满足任一即可）：
        1. 数据类型为：临检、YC、SJ
        2. 样本编号前缀为：FDHE
        3. 加测备注为：加测或者混合
        """
        dt = str(getattr(lib, "data_type", "") or "").strip()
        if dt in self._eligible_data_types:
            return True

        prefix = str(getattr(lib, "sample_number_prefix", "") or "").strip().upper()
        if not prefix:
            sid = str(getattr(lib, "sample_id", "") or "").strip().upper()
            prefix = sid[:4] if len(sid) >= 4 else sid
        for ep in self._eligible_prefixes:
            if prefix.startswith(ep):
                return True

        remark = str(getattr(lib, "add_tests_remark", "") or "").strip()
        if remark in self._eligible_add_test_kw:
            return True

        return False

    def _resolve_priority_36t_lane_count(self, total_gb: float) -> int:
        """根据临检/YC/SJ数据量总和，确定3.6T-NEW lane数上限"""
        if total_gb <= 0:
            return 0
        max_lanes = 0
        for rule in self._lane_limit_rules:
            if total_gb <= rule.get("max_data_gb", 0):
                return rule.get("max_lanes", 1)
            max_lanes = rule.get("max_lanes", max_lanes)
        return max_lanes

    def _warn_cross_mode_sample_split(self, result: ModeDispatchResult) -> None:
        """检查同一个sample_id是否被分到了不同模式，记录警告日志"""
        sample_modes: Dict[str, Set[str]] = {}
        for label, pool in [
            ("36t", result.pool_36t_priority),
            ("1_1", result.pool_1_1_normal + result.pool_1_1_quality_risk + result.pool_1_1_quality_other),
            ("forbidden", result.pool_1_1_forbidden),
        ]:
            for lib in pool:
                sid = str(getattr(lib, "sample_id", "") or "").strip()
                if sid:
                    sample_modes.setdefault(sid, set()).add(label)

        for sid, modes in sample_modes.items():
            if len(modes) > 1:
                logger.warning(
                    "同sample_id跨模式分流: sample_id={}, 分布模式={}",
                    sid, sorted(modes),
                )
