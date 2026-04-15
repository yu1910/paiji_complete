"""
1.1模式 service 封装单元测试

测试覆盖：
- 输入准备与 AI 可排筛选
- 首轮 service 元数据注入与回流结果
- 第二轮 service 候选识别与预留排机调用
- 全流程 service 对 arrange_library 的委托
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from arrange_library import mode_1_1_service
from arrange_library.core.scheduling.scheduling_types import LaneAssignment
from arrange_library.models.library_info import EnhancedLibraryInfo, MachineType


def _make_lib(
    origrec: str = "LIB_001",
    data_type: str = "其他",
    contract_data_raw: float = 50.0,
    sample_id: str = "S001",
    sample_number_prefix: str = "",
    add_tests_remark: str = "-",
    complex_result: str = "合格",
    risk_build_flag: str = "正常建库",
    eq_type: str = "Nova X-25B",
    last_lane_round: str = "",
    last_laneid: str = "",
    aiavailable: str = "",
) -> EnhancedLibraryInfo:
    """构造测试用文库对象。"""
    lib = EnhancedLibraryInfo(
        origrec=origrec,
        sample_id=sample_id,
        sample_type_code="WES",
        data_type=data_type,
        customer_library="否",
        base_type="双",
        number_of_bases=10,
        index_number=1,
        contract_data_raw=contract_data_raw,
        index_seq="AACCGGTT;TTGGCCAA",
        add_tests_remark=add_tests_remark,
        product_line="S",
        eq_type=eq_type,
        peak_size=350,
        test_code=1595,
        test_no="Novaseq X Plus-PE150",
        sub_project_name="TEST_PROJECT",
        create_date="2026-04-15",
        delivery_date="2026-04-30",
        lab_type="诺禾-WES文库",
        data_volume_type="小数量",
        board_number="BN001",
    )
    lib.sample_number_prefix = sample_number_prefix
    lib.complex_result = complex_result
    lib.risk_build_flag = risk_build_flag
    lib.last_lane_round = last_lane_round if last_lane_round else None
    lib.last_laneid = last_laneid if last_laneid else None
    lib._aiavailable_raw = aiavailable
    return lib


_SAMPLE_CONFIG = {
    "single_library_contract_limit_gb": 500,
    "priority_data_types_for_36t": ["临检", "YC", "SJ"],
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
    "first_round_label": "1.1第一轮",
    "second_round_label": "1.1第二轮",
    "second_round_output_rate_threshold": 40.0,
    "second_round_default_pooling_factor": 2.5,
}


def test_prepare_mode_1_1_libraries_filters_machine_and_ai_flags():
    libs = [
        _make_lib(origrec="A", add_tests_remark="加测"),
        _make_lib(origrec="B", add_tests_remark="加测", aiavailable="no"),
        _make_lib(origrec="C", add_tests_remark="加测", eq_type="Nova X-10B"),
    ]

    result = mode_1_1_service.prepare_mode_1_1_libraries(libraries=libs)

    assert len(result.all_libraries) == 3
    assert len(result.ai_schedulable_libraries) == 1
    assert len(result.non_ai_libraries) == 1
    assert len(result.excluded_machine_libraries) == 1


def test_run_mode_1_1_round1_returns_lanes_and_fallback(monkeypatch):
    lane = LaneAssignment(
        lane_id="L_1_1_001",
        machine_id="M_L_1_1_001",
        machine_type=MachineType.NOVA_X_25B,
        libraries=[],
        metadata={},
    )
    fallback_lib = _make_lib(origrec="FB", add_tests_remark="加测")

    def _fake_test_with_model(libraries, existing_lanes=None):
        lane.libraries = [libraries[0]]
        return {"lane_count": 1}, SimpleNamespace(
            lane_assignments=[lane],
            unassigned_libraries=[fallback_lib],
        )

    monkeypatch.setattr(mode_1_1_service, "test_with_model", _fake_test_with_model)

    libs = [
        _make_lib(origrec="L1", add_tests_remark="加测"),
        _make_lib(origrec="L2", add_tests_remark="混合", complex_result="风险", risk_build_flag="风险建库"),
    ]
    result = mode_1_1_service.run_mode_1_1_round1(libraries=libs, config=_SAMPLE_CONFIG)

    assert result.scheduling_succeeded is True
    assert result.scheduling_error is None
    assert len(result.lanes) == 1
    assert result.lanes[0].metadata["selected_seq_mode"] == "1.1"
    assert result.lanes[0].metadata["selected_round_label"] == "1.1第一轮"
    assert len(result.fallback_libraries_for_36t) == 1
    assert any(lib.origrec == "FB" for lib in result.normal_libraries_for_36t)


def test_run_mode_1_1_round2_identifies_candidates_and_invokes_schedule(monkeypatch):
    called = {"value": False}

    def _fake_schedule(self, candidate_groups):
        called["value"] = True
        assert len(candidate_groups) == 1
        return SimpleNamespace(lanes=[], fallback_libraries=[], broken_groups=0)

    monkeypatch.setattr(mode_1_1_service.Mode11Round2Handler, "schedule_round2", _fake_schedule)

    libs = [
        _make_lib(origrec="R1", add_tests_remark="加测", last_lane_round="1.1第一轮", last_laneid="LANE_A"),
        _make_lib(origrec="R2", add_tests_remark="加测", last_lane_round="1.1第一轮", last_laneid="LANE_A"),
        _make_lib(origrec="R3", add_tests_remark="加测", last_lane_round="", last_laneid=""),
    ]

    result = mode_1_1_service.run_mode_1_1_round2(
        libraries=libs,
        config=_SAMPLE_CONFIG,
        schedule_candidates=True,
    )

    assert result.schedule_invoked is True
    assert result.scheduling_result is not None
    assert result.identification_result.total_candidates == 2
    assert len(result.identification_result.candidate_groups) == 1
    assert called["value"] is True


def test_run_mode_1_1_full_delegates_to_arrange_library(monkeypatch, tmp_path):
    output_file = tmp_path / "mode_1_1_output.csv"

    def _fake_arrange_library(**kwargs):
        assert kwargs["mode"] == "arrange"
        return output_file

    monkeypatch.setattr(mode_1_1_service, "arrange_library", _fake_arrange_library)

    result = mode_1_1_service.run_mode_1_1_full(data_file=tmp_path / "input.csv")

    assert result.output_path == output_file
    assert result.mode == "arrange"
