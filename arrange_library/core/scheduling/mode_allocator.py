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
from typing import Any, Dict, List, Optional, Set, Tuple

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

    _QUALITY_SEED_HINTS = {
        "normal": (20, "mode_1_1_quality_normal"),
        "risk": (21, "mode_1_1_quality_risk"),
        "other": (22, "mode_1_1_quality_other"),
    }

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._contract_limit = float(config.get("single_library_contract_limit_gb", 500))
        self._priority_data_types = set(config.get("priority_data_types_for_36t", []))
        overflow_cfg = config.get("priority_data_types_to_1_1_overflow", {})
        self._priority_overflow_enabled = bool(overflow_cfg.get("enabled", False))
        self._priority_overflow_trigger_min_pool_gb = float(
            overflow_cfg.get("trigger_min_pool_gb", 0) or 0
        )
        self._priority_overflow_max_total_gb = float(
            overflow_cfg.get("max_total_gb", 0) or 0
        )
        self._eligible_data_types = set(config.get("eligible_data_types_for_1_1", []))
        self._eligible_prefixes = [
            p.upper() for p in config.get("eligible_sample_prefixes_for_1_1", [])
        ]
        self._eligible_add_test_kw = set(config.get("eligible_add_test_keywords_for_1_1", []))
        self._priority_36t_preconsume_borrow_fillers_from_1_1 = bool(
            config.get("priority_36t_preconsume_borrow_fillers_from_1_1", False)
        )
        self._priority_36t_preconsume_max_filler_gb_per_lane = float(
            config.get("priority_36t_preconsume_max_filler_gb_per_lane", 250.0) or 0.0
        )
        self._lane_limit_rules = config.get("priority_36t_lane_limit_rules", [])
        qg = config.get("quality_grouping", {})
        ng = qg.get("normal_group", {})
        rg = qg.get("risk_group", {})
        self._normal_complex = set(ng.get("complex_results", []))
        self._normal_risk_flags = set(ng.get("risk_build_flags", []))
        self._risk_complex = set(rg.get("complex_results", []))
        self._risk_flags = set(rg.get("risk_build_flags", []))
        manual_cfg = dict(config.get("manual_dispatch_overrides", {}) or {})
        self._manual_prefer_1_1_rules = [
            self._normalize_manual_dispatch_rule(rule)
            for rule in list(manual_cfg.get("prefer_1_1", []) or [])
        ]
        self._manual_prefer_36t_rules = [
            self._normalize_manual_dispatch_rule(rule)
            for rule in list(manual_cfg.get("prefer_36t", []) or [])
        ]

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
        first_round_1_1_candidates: List[EnhancedLibraryInfo] = []

        for lib in libraries:
            origrec = getattr(lib, "origrec", "") or ""
            reason = self._check_1_1_forbidden(lib)

            # 高优先级文库统一先进入3.6T-NEW预消耗候选池，即使其本身不满足1.1条件。
            if self._is_priority_for_36t(lib):
                priority_candidates.append(lib)
                if reason:
                    result.dispatch_reasons[origrec] = f"priority_data_type_for_36t|{reason}"
                else:
                    result.dispatch_reasons[origrec] = "priority_data_type_for_36t"
                continue

            if reason:
                result.pool_1_1_forbidden.append(lib)
                self._clear_mode_1_1_seed_hint(lib)
                result.dispatch_reasons[origrec] = reason
                continue

            manual_override = self._resolve_manual_dispatch_override(lib)
            if manual_override is not None:
                override_mode, override_reason, seed_rank, seed_group = manual_override
                if override_mode == "3.6T-NEW":
                    result.pool_1_1_forbidden.append(lib)
                    self._clear_mode_1_1_seed_hint(lib)
                    result.dispatch_reasons[origrec] = override_reason
                    continue

                self._set_mode_1_1_seed_hint(
                    lib,
                    seed_rank=seed_rank,
                    seed_group=seed_group,
                )
                first_round_1_1_candidates.append(lib)
                result.dispatch_reasons[origrec] = override_reason
                continue

            self._set_mode_1_1_seed_hint(lib)
            first_round_1_1_candidates.append(lib)
            result.dispatch_reasons[origrec] = "allowed_for_1_1_by_default"

        priority_total_gb = sum(float(lib.contract_data_raw or 0) for lib in priority_candidates)
        max_36t_lanes = self._resolve_priority_36t_lane_count(priority_total_gb)
        logger.info(
            "模式分流: 临检/YC/SJ数据量={:.1f}G, 建议3.6T-NEW lane上限={}, "
            "1.1可排文库={}, 1.1禁排回退={}",
            priority_total_gb, max_36t_lanes,
            len(first_round_1_1_candidates), len(result.pool_1_1_forbidden),
        )

        result.pool_36t_priority = priority_candidates

        for lib in first_round_1_1_candidates:
            origrec = getattr(lib, "origrec", "") or ""
            cr = str(getattr(lib, "complex_result", "") or "").strip()
            rf = str(getattr(lib, "risk_build_flag", "") or "").strip()
            existing_reason = str(result.dispatch_reasons.get(origrec, "") or "").strip()

            def _merge_quality_reason(quality_reason: str) -> str:
                if existing_reason.startswith("manual_prefer_1_1:"):
                    return f"{existing_reason}|{quality_reason}"
                return quality_reason

            if cr in self._normal_complex and rf in self._normal_risk_flags:
                self._apply_mode_1_1_quality_seed_hint(lib)
                result.pool_1_1_normal.append(lib)
                result.dispatch_reasons[origrec] = _merge_quality_reason("1_1_quality_normal")
            elif cr in self._risk_complex and rf in self._risk_flags:
                self._apply_mode_1_1_quality_seed_hint(lib)
                result.pool_1_1_quality_risk.append(lib)
                result.dispatch_reasons[origrec] = _merge_quality_reason("1_1_quality_risk")
            else:
                self._apply_mode_1_1_quality_seed_hint(lib)
                result.pool_1_1_quality_other.append(lib)
                result.dispatch_reasons[origrec] = _merge_quality_reason("1_1_quality_other")

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

    def _resolve_sample_prefix(self, lib: EnhancedLibraryInfo) -> str:
        """统一解析样本前缀，优先读显式字段，缺失时回退到 sample_id 前四位。"""
        prefix = str(getattr(lib, "sample_number_prefix", "") or "").strip().upper()
        if prefix:
            return prefix

        sample_id = str(getattr(lib, "sample_id", "") or "").strip().upper()
        if not sample_id:
            return ""
        return sample_id[:4] if len(sample_id) >= 4 else sample_id

    def _matches_eligible_sample_prefix(self, lib: EnhancedLibraryInfo) -> bool:
        prefix = self._resolve_sample_prefix(lib)
        if not prefix:
            return False
        return any(prefix.startswith(ep) for ep in self._eligible_prefixes)

    def _matches_eligible_add_test_keyword(self, lib: EnhancedLibraryInfo) -> bool:
        remark = str(getattr(lib, "add_tests_remark", "") or "").strip()
        return remark in self._eligible_add_test_kw

    def _has_1_1_secondary_eligibility(self, lib: EnhancedLibraryInfo) -> bool:
        """规则11次级条件：FDHE 或 加测/混合。"""
        return (
            self._matches_eligible_sample_prefix(lib)
            or self._matches_eligible_add_test_keyword(lib)
        )

    def _apply_mode_1_1_quality_seed_hint(self, lib: EnhancedLibraryInfo) -> None:
        """为规则12的优先组合打首轮聚簇提示。"""
        cr = str(getattr(lib, "complex_result", "") or "").strip()
        rf = str(getattr(lib, "risk_build_flag", "") or "").strip()
        if cr in self._normal_complex and rf in self._normal_risk_flags:
            self._set_mode_1_1_seed_hint(lib, *self._QUALITY_SEED_HINTS["normal"])
            return
        if cr in self._risk_complex and rf in self._risk_flags:
            self._set_mode_1_1_seed_hint(lib, *self._QUALITY_SEED_HINTS["risk"])
            return
        self._set_mode_1_1_seed_hint(lib, *self._QUALITY_SEED_HINTS["other"])

    def _is_eligible_for_1_1(self, lib: EnhancedLibraryInfo) -> bool:
        """判断高优先级文库是否满足规则11的小量溢出到1.1条件。

        注意：
        - 该条件不再作为1.1首轮的总准入条件
        - 其用途仅限于规则11场景下，识别哪些临检/YC/SJ文库允许少量混入1.1
        """
        dt = str(getattr(lib, "data_type", "") or "").strip()
        if dt not in self._eligible_data_types:
            return False
        return self._has_1_1_secondary_eligibility(lib)

    def _is_priority_overflow_candidate_for_1_1(self, lib: EnhancedLibraryInfo) -> bool:
        """优先池中允许少量溢出到1.1的候选。"""
        if not self._priority_overflow_enabled:
            return False
        if not self._is_priority_for_36t(lib):
            return False
        return self._is_eligible_for_1_1(lib)

    def _should_borrow_1_1_fillers_for_priority_36t_preconsume(self) -> bool:
        """首轮3.6T高优预消耗是否允许提前借用1.1普通池补料。"""
        return self._priority_36t_preconsume_borrow_fillers_from_1_1

    def _get_priority_36t_preconsume_max_filler_gb_per_lane(self) -> float:
        """首轮3.6T高优预消耗允许借用的1.1普通文库补位上限。"""
        return max(0.0, self._priority_36t_preconsume_max_filler_gb_per_lane)

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

    def _get_contract_data_gb(self, lib: EnhancedLibraryInfo) -> float:
        """读取文库合同量，统一成浮点数。"""
        return float(getattr(lib, "contract_data_raw", 0) or 0)

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

    @staticmethod
    def _normalize_rule_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().replace("＋", "+").replace("（", "(").replace("）", ")").upper()

    @staticmethod
    def _normalize_manual_dispatch_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
        def _normalize_list(values: Any) -> List[str]:
            normalized: List[str] = []
            for item in list(values or []):
                text = ModeAllocator._normalize_rule_text(item)
                if text:
                    normalized.append(text)
            return normalized

        min_contract = rule.get("min_contract_gb")
        max_contract = rule.get("max_contract_gb")
        peak_sizes = set()
        for value in list(rule.get("peak_sizes", []) or []):
            try:
                peak_sizes.add(int(value))
            except (TypeError, ValueError):
                continue

        raw_seed_rank = rule.get("seed_rank", 90)
        return {
            "name": str(rule.get("name", "") or "").strip() or "unnamed_rule",
            "task_group_keywords": _normalize_list(rule.get("task_group_keywords")),
            "sample_types": _normalize_list(rule.get("sample_types")),
            "sample_type_keywords": _normalize_list(rule.get("sample_type_keywords")),
            "data_types": _normalize_list(rule.get("data_types")),
            "sample_prefixes": _normalize_list(rule.get("sample_prefixes")),
            "peak_sizes": peak_sizes,
            "min_contract_gb": (
                float(min_contract) if min_contract not in (None, "", False) else None
            ),
            "max_contract_gb": (
                float(max_contract) if max_contract not in (None, "", False) else None
            ),
            "seed_rank": int(90 if raw_seed_rank in (None, "") else raw_seed_rank),
        }

    def _get_task_group_name(self, lib: EnhancedLibraryInfo) -> str:
        raw_value = getattr(lib, "task_group_name", None)
        if raw_value in (None, ""):
            raw_value = getattr(lib, "_task_group_name_raw", None)
        return str(raw_value or "").strip()

    def _matches_manual_dispatch_rule(
        self,
        lib: EnhancedLibraryInfo,
        rule: Dict[str, Any],
    ) -> bool:
        task_group = self._normalize_rule_text(self._get_task_group_name(lib))
        sample_type = self._normalize_rule_text(getattr(lib, "sample_type_code", "") or "")
        data_type = self._normalize_rule_text(getattr(lib, "data_type", "") or "")
        prefix = self._normalize_rule_text(
            getattr(lib, "sample_number_prefix", "") or str(getattr(lib, "sample_id", "") or "")[:4]
        )
        contract = self._get_contract_data_gb(lib)
        peak_size = getattr(lib, "peak_size", None)

        task_group_keywords = list(rule.get("task_group_keywords", []) or [])
        if task_group_keywords and not any(keyword in task_group for keyword in task_group_keywords):
            return False

        sample_types = list(rule.get("sample_types", []) or [])
        if sample_types and sample_type not in sample_types:
            return False

        sample_type_keywords = list(rule.get("sample_type_keywords", []) or [])
        if sample_type_keywords and not any(keyword in sample_type for keyword in sample_type_keywords):
            return False

        data_types = list(rule.get("data_types", []) or [])
        if data_types and data_type not in data_types:
            return False

        sample_prefixes = list(rule.get("sample_prefixes", []) or [])
        if sample_prefixes and not any(prefix.startswith(item) for item in sample_prefixes):
            return False

        min_contract = rule.get("min_contract_gb")
        if min_contract is not None and contract < float(min_contract):
            return False

        max_contract = rule.get("max_contract_gb")
        if max_contract is not None and contract > float(max_contract):
            return False

        peak_sizes = set(rule.get("peak_sizes", set()) or set())
        if peak_sizes:
            try:
                normalized_peak = int(peak_size)
            except (TypeError, ValueError):
                return False
            if normalized_peak not in peak_sizes:
                return False

        return True

    def _resolve_manual_dispatch_override(
        self,
        lib: EnhancedLibraryInfo,
    ) -> Optional[Tuple[str, str, Optional[int], str]]:
        for rule in self._manual_prefer_36t_rules:
            if self._matches_manual_dispatch_rule(lib, rule):
                return (
                    "3.6T-NEW",
                    f"manual_prefer_36t:{rule['name']}",
                    None,
                    "",
                )

        for rule in self._manual_prefer_1_1_rules:
            if self._matches_manual_dispatch_rule(lib, rule):
                raw_seed_rank = rule.get("seed_rank", 90)
                return (
                    "1.1",
                    f"manual_prefer_1_1:{rule['name']}",
                    int(90 if raw_seed_rank in (None, "") else raw_seed_rank),
                    str(rule.get("name", "") or "").strip(),
                )

        return None

    @staticmethod
    def _clear_mode_1_1_seed_hint(lib: EnhancedLibraryInfo) -> None:
        for attr_name in ("_mode_1_1_seed_rank", "_mode_1_1_seed_group"):
            if hasattr(lib, attr_name):
                delattr(lib, attr_name)

    def _set_mode_1_1_seed_hint(
        self,
        lib: EnhancedLibraryInfo,
        seed_rank: Optional[int] = None,
        seed_group: str = "",
    ) -> None:
        normalized_rank = 90 if seed_rank is None else int(seed_rank)
        current_rank = getattr(lib, "_mode_1_1_seed_rank", None)
        should_update_group = False
        if current_rank is None or normalized_rank < int(current_rank):
            lib._mode_1_1_seed_rank = normalized_rank
            should_update_group = True
        elif current_rank is not None and normalized_rank == int(current_rank):
            should_update_group = True
        if seed_group and should_update_group:
            lib._mode_1_1_seed_group = str(seed_group).strip()
