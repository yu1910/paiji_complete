"""
1.1模式分流器单元测试
创建时间：2026-04-14 13:50:00
更新时间：2026-04-14 13:50:00

测试覆盖：
- 单文库>500G禁排
- 包Lane/包FC禁排
- 特殊拆分方式禁排
- 临检/YC/SJ优先走3.6T-NEW
- 规则11的FDHE/加测/混合条件仅用于高优先级少量溢出到1.1
- 非高优先级文库默认进入1.1首轮池，除非命中禁排规则
- 质量分组：合格+正常建库、风险+风险建库、兜底
- 首轮lane数上限计算
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from arrange_library.core.config.scheduling_config import get_scheduling_config
from arrange_library.core.scheduling.mode_allocator import ModeAllocator, ModeDispatchResult
from arrange_library.models.library_info import EnhancedLibraryInfo


def _make_lib(
    origrec: str = "LIB_001",
    data_type: str = "其他",
    contract_data_raw: float = 50.0,
    sample_id: str = "S001",
    sample_type_code: str = "WES",
    sample_number_prefix: str = "",
    add_tests_remark: str = "-",
    complex_result: str = "",
    risk_build_flag: str = "",
    package_lane_number: str = "",
    bagfcno: str = "",
    special_splits: str = "",
    peak_size: int = 350,
    task_group_name: str = "",
) -> EnhancedLibraryInfo:
    """构造测试用文库对象（只填必要字段）"""
    lib = EnhancedLibraryInfo(
        origrec=origrec,
        sample_id=sample_id,
        sample_type_code=sample_type_code,
        data_type=data_type,
        customer_library="否",
        base_type="双",
        number_of_bases=10,
        index_number=1,
        contract_data_raw=contract_data_raw,
        index_seq="AACCGGTT;TTGGCCAA",
        add_tests_remark=add_tests_remark,
        product_line="S",
        eq_type="Nova X-25B",
        peak_size=peak_size,
        test_code=1595,
        test_no="Novaseq X Plus-PE150",
        sub_project_name="TEST_PROJECT",
        create_date="2026-04-14",
        delivery_date="2026-04-30",
        lab_type="诺禾-WES文库",
        data_volume_type="小数量",
        board_number="BN001",
    )
    lib.sample_number_prefix = sample_number_prefix
    lib.complex_result = complex_result
    lib.risk_build_flag = risk_build_flag
    lib.package_lane_number = package_lane_number
    lib.bagfcno = bagfcno
    lib.special_splits = special_splits
    if task_group_name:
        lib.task_group_name = task_group_name
        lib._task_group_name_raw = task_group_name
    return lib


_SAMPLE_CONFIG = {
    "single_library_contract_limit_gb": 500,
    "priority_data_types_for_36t": ["临检", "YC", "SJ"],
    "priority_data_types_to_1_1_overflow": {
        "enabled": True,
        "trigger_min_pool_gb": 1850,
        "max_total_gb": 100,
    },
    "eligible_data_types_for_1_1": ["临检", "YC", "SJ"],
    "eligible_sample_prefixes_for_1_1": ["FDHE"],
    "eligible_add_test_keywords_for_1_1": ["加测", "混合"],
    "priority_36t_lane_limit_rules": [
        {"max_data_gb": 1050, "max_lanes": 1},
        {"max_data_gb": 1600, "max_lanes": 2},
        {"max_data_gb": 2000, "max_lanes": 3},
        {"max_data_gb": 3000, "max_lanes": 4},
    ],
    "quality_grouping": {
        "normal_group": {
            "complex_results": ["合格"],
            "risk_build_flags": ["正常建库"],
        },
        "risk_group": {
            "complex_results": ["风险", "不合格"],
            "risk_build_flags": ["风险建库"],
        },
    },
}

_MANUAL_OVERRIDE_CONFIG = {
    **_SAMPLE_CONFIG,
    "manual_dispatch_overrides": {
        "prefer_1_1": [
            {
                "name": "auto_rna_vip_440",
                "task_group_keywords": ["RNA转录组建库-库检-Novaseq X Plus-PE150"],
                "sample_types": ["VIP-真核普通转录组文库"],
                "peak_sizes": [440],
                "seed_rank": 0,
            },
            {
                "name": "auto_wgs_10g_dna",
                "task_group_keywords": ["（自动化）WGS建库-库检-Novaseq X Plus-PE150"],
                "sample_types": ["DNA小片段文库"],
                "peak_sizes": [460],
                "min_contract_gb": 9.5,
                "max_contract_gb": 10.5,
                "seed_rank": 1,
            },
        ],
        "prefer_36t": [
            {
                "name": "capture_exome_strong_36t",
                "task_group_keywords": ["捕获建库+库检-天津Novaseq X Plus-PE150"],
            },
            {
                "name": "exome_like_sampletypes_36t",
                "sample_types": ["外显子文库"],
            },
        ],
    },
}


class TestForbiddenRules:
    """1.1模式禁排规则测试"""

    def test_contract_exceeds_500g_forbidden(self):
        lib = _make_lib(contract_data_raw=600.0, data_type="临检")
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        result = allocator.allocate([lib])
        assert len(result.pool_1_1_forbidden) == 0
        assert len(result.pool_36t_priority) == 1
        assert "exceeds_500g" in result.dispatch_reasons.get(lib.origrec, "")

    def test_contract_exactly_500g_not_forbidden(self):
        lib = _make_lib(contract_data_raw=500.0, data_type="临检")
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        result = allocator.allocate([lib])
        assert len(result.pool_1_1_forbidden) == 0

    def test_package_lane_forbidden(self):
        lib = _make_lib(data_type="临检", package_lane_number="PKG001")
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        result = allocator.allocate([lib])
        assert len(result.pool_1_1_forbidden) == 0
        assert len(result.pool_36t_priority) == 1
        assert "package_lane_or_fc" in result.dispatch_reasons.get(lib.origrec, "")

    def test_package_fc_forbidden(self):
        lib = _make_lib(data_type="临检", bagfcno="FC001")
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        result = allocator.allocate([lib])
        assert len(result.pool_1_1_forbidden) == 0
        assert len(result.pool_36t_priority) == 1

    def test_special_split_forbidden(self):
        lib = _make_lib(data_type="其他", special_splits="10x_cellranger", add_tests_remark="加测")
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        result = allocator.allocate([lib])
        assert len(result.pool_1_1_forbidden) == 1
        assert "special_split" in result.dispatch_reasons.get(lib.origrec, "")

    def test_priority_special_split_still_goes_to_36t_priority(self):
        lib = _make_lib(data_type="临检", special_splits="10x_cellranger")
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        result = allocator.allocate([lib])
        assert len(result.pool_36t_priority) == 1
        assert len(result.pool_1_1_forbidden) == 0
        assert "priority_data_type_for_36t" in result.dispatch_reasons.get(lib.origrec, "")
        assert "special_split" in result.dispatch_reasons.get(lib.origrec, "")


class TestPriorityDispatch:
    """临检/YC/SJ优先3.6T-NEW分流测试"""

    def test_clinical_goes_to_36t_priority(self):
        lib = _make_lib(data_type="临检")
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        result = allocator.allocate([lib])
        assert len(result.pool_36t_priority) == 1
        assert len(result.pool_1_1_normal) == 0

    def test_yc_goes_to_36t_priority(self):
        lib = _make_lib(data_type="YC")
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        result = allocator.allocate([lib])
        assert len(result.pool_36t_priority) == 1

    def test_sj_goes_to_36t_priority(self):
        lib = _make_lib(data_type="SJ")
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        result = allocator.allocate([lib])
        assert len(result.pool_36t_priority) == 1


class TestEligibilityConditions:
    """规则11的小量溢出与1.1默认准入测试"""

    def test_fdhe_prefix_eligible(self):
        lib = _make_lib(data_type="临检", sample_number_prefix="FDHE")
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        assert allocator._is_eligible_for_1_1(lib) is True

    def test_add_test_remark_eligible(self):
        lib = _make_lib(data_type="YC", add_tests_remark="加测")
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        assert allocator._is_eligible_for_1_1(lib) is True

    def test_mixed_remark_eligible(self):
        lib = _make_lib(data_type="SJ", add_tests_remark="混合")
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        assert allocator._is_eligible_for_1_1(lib) is True

    def test_add_test_remark_not_blocked_by_1_1_rule_matrix(self):
        lib = _make_lib(data_type="其他", add_tests_remark="加测")
        messages = get_scheduling_config().validate_lane_constraints(
            libraries=[lib],
            machine_type="Nova X-25B",
            metadata={"selected_seq_mode": "1.1", "lcxms": "1.1"},
        )
        assert "命中1.1模式禁止条件，不允许按1.1模式排机" not in messages

    def test_non_priority_library_defaults_to_1_1_when_not_forbidden(self):
        lib = _make_lib(data_type="其他", add_tests_remark="-")
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        result = allocator.allocate([lib])
        assert len(result.pool_1_1_forbidden) == 0
        assert len(result.pool_1_1_quality_other) == 1
        assert result.dispatch_reasons.get(lib.origrec) == "1_1_quality_other"

    def test_priority_data_types_remain_in_36t_pool_before_1_1_round(self):
        libs = [
            _make_lib(origrec="E1", contract_data_raw=450.0, add_tests_remark="加测"),
            _make_lib(origrec="E2", contract_data_raw=450.0, add_tests_remark="加测"),
            _make_lib(origrec="E3", contract_data_raw=450.0, add_tests_remark="加测"),
            _make_lib(origrec="E4", contract_data_raw=450.0, add_tests_remark="加测"),
            _make_lib(origrec="P_SMALL", contract_data_raw=60.0, data_type="临检"),
            _make_lib(origrec="P_LARGE", contract_data_raw=200.0, data_type="YC"),
        ]
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        result = allocator.allocate(libs)

        first_round_origrecs = {
            lib.origrec
            for lib in (
                result.pool_1_1_normal
                + result.pool_1_1_quality_risk
                + result.pool_1_1_quality_other
            )
        }
        priority_origrecs = {lib.origrec for lib in result.pool_36t_priority}

        assert "P_SMALL" not in first_round_origrecs
        assert "P_LARGE" not in first_round_origrecs
        assert {"P_SMALL", "P_LARGE"} <= priority_origrecs

    def test_priority_data_types_still_remain_in_36t_pool_when_gap_exceeds_100g(self):
        libs = [
            _make_lib(origrec="E1", contract_data_raw=425.0, add_tests_remark="加测"),
            _make_lib(origrec="E2", contract_data_raw=425.0, add_tests_remark="加测"),
            _make_lib(origrec="E3", contract_data_raw=425.0, add_tests_remark="加测"),
            _make_lib(origrec="E4", contract_data_raw=425.0, add_tests_remark="加测"),
            _make_lib(origrec="P_SMALL", contract_data_raw=60.0, data_type="临检"),
        ]
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        result = allocator.allocate(libs)

        first_round_origrecs = {
            lib.origrec
            for lib in (
                result.pool_1_1_normal
                + result.pool_1_1_quality_risk
                + result.pool_1_1_quality_other
            )
        }
        remaining_priority_origrecs = {lib.origrec for lib in result.pool_36t_priority}

        assert "P_SMALL" not in first_round_origrecs
        assert "P_SMALL" in remaining_priority_origrecs

    def test_manual_override_prefers_auto_rna_cluster_for_1_1(self):
        lib = _make_lib(
            data_type="其他",
            add_tests_remark="-",
            sample_type_code="VIP-真核普通转录组文库",
            peak_size=440,
            task_group_name="（自动化）RNA转录组建库-库检-Novaseq X Plus-PE150",
        )
        allocator = ModeAllocator(_MANUAL_OVERRIDE_CONFIG)
        result = allocator.allocate([lib])

        total_1_1 = (
            len(result.pool_1_1_normal)
            + len(result.pool_1_1_quality_risk)
            + len(result.pool_1_1_quality_other)
        )
        assert total_1_1 == 1
        assert "manual_prefer_1_1:auto_rna_vip_440" in result.dispatch_reasons.get(lib.origrec, "")
        assert getattr(lib, "_mode_1_1_seed_rank", None) == 0

    def test_manual_override_prefers_auto_wgs_10g_for_1_1(self):
        lib = _make_lib(
            data_type="其他",
            add_tests_remark="-",
            contract_data_raw=10.0,
            sample_type_code="DNA小片段文库",
            peak_size=460,
            task_group_name="（自动化）WGS建库-库检-Novaseq X Plus-PE150",
        )
        allocator = ModeAllocator(_MANUAL_OVERRIDE_CONFIG)
        result = allocator.allocate([lib])

        total_1_1 = (
            len(result.pool_1_1_normal)
            + len(result.pool_1_1_quality_risk)
            + len(result.pool_1_1_quality_other)
        )
        assert total_1_1 == 1
        assert "manual_prefer_1_1:auto_wgs_10g_dna" in result.dispatch_reasons.get(lib.origrec, "")
        assert getattr(lib, "_mode_1_1_seed_rank", None) == 1

    def test_manual_override_prefers_capture_exome_for_36t(self):
        lib = _make_lib(
            data_type="其他",
            add_tests_remark="加测",
            sample_type_code="外显子文库",
            peak_size=410,
            task_group_name="捕获建库+库检-天津Novaseq X Plus-PE150",
        )
        allocator = ModeAllocator(_MANUAL_OVERRIDE_CONFIG)
        result = allocator.allocate([lib])

        assert len(result.pool_1_1_forbidden) == 1
        assert not (
            result.pool_1_1_normal
            or result.pool_1_1_quality_risk
            or result.pool_1_1_quality_other
        )
        assert result.dispatch_reasons.get(lib.origrec) == "manual_prefer_36t:capture_exome_strong_36t"


class TestQualityGrouping:
    """1.1质量分组测试"""

    def test_normal_quality_group(self):
        lib = _make_lib(
            data_type="其他",
            add_tests_remark="-",
            complex_result="合格",
            risk_build_flag="正常建库",
            sample_type_code="VIP-真核普通转录组文库",
            peak_size=440,
            task_group_name="（自动化）RNA转录组建库-库检-Novaseq X Plus-PE150",
        )
        allocator = ModeAllocator(_MANUAL_OVERRIDE_CONFIG)
        result = allocator.allocate([lib])
        assert len(result.pool_1_1_normal) == 1

    def test_risk_quality_group(self):
        lib = _make_lib(
            data_type="其他",
            add_tests_remark="-",
            complex_result="风险",
            risk_build_flag="风险建库",
            sample_type_code="VIP-真核普通转录组文库",
            peak_size=440,
            task_group_name="（自动化）RNA转录组建库-库检-Novaseq X Plus-PE150",
        )
        allocator = ModeAllocator(_MANUAL_OVERRIDE_CONFIG)
        result = allocator.allocate([lib])
        assert len(result.pool_1_1_quality_risk) == 1

    def test_unqualified_risk_group(self):
        lib = _make_lib(
            data_type="其他",
            add_tests_remark="-",
            complex_result="不合格",
            risk_build_flag="风险建库",
            sample_type_code="VIP-真核普通转录组文库",
            peak_size=440,
            task_group_name="（自动化）RNA转录组建库-库检-Novaseq X Plus-PE150",
        )
        allocator = ModeAllocator(_MANUAL_OVERRIDE_CONFIG)
        result = allocator.allocate([lib])
        assert len(result.pool_1_1_quality_risk) == 1

    def test_other_quality_group(self):
        lib = _make_lib(
            data_type="其他",
            add_tests_remark="-",
            complex_result="合格",
            risk_build_flag="风险建库",
            sample_type_code="VIP-真核普通转录组文库",
            peak_size=440,
            task_group_name="（自动化）RNA转录组建库-库检-Novaseq X Plus-PE150",
        )
        allocator = ModeAllocator(_MANUAL_OVERRIDE_CONFIG)
        result = allocator.allocate([lib])
        assert len(result.pool_1_1_quality_other) == 1


class TestLaneLimitRules:
    """3.6T-NEW lane数上限测试"""

    def test_under_1050g_gives_1_lane(self):
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        assert allocator._resolve_priority_36t_lane_count(800) == 1

    def test_1050g_gives_2_lanes(self):
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        assert allocator._resolve_priority_36t_lane_count(1200) == 2

    def test_1600g_gives_3_lanes(self):
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        assert allocator._resolve_priority_36t_lane_count(1800) == 3

    def test_2000g_gives_4_lanes(self):
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        assert allocator._resolve_priority_36t_lane_count(2500) == 4

    def test_over_3000g_still_4_lanes(self):
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        assert allocator._resolve_priority_36t_lane_count(5000) == 4

    def test_zero_gives_0_lanes(self):
        allocator = ModeAllocator(_SAMPLE_CONFIG)
        assert allocator._resolve_priority_36t_lane_count(0) == 0
