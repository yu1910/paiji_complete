"""
1.1模式第二轮候选识别单元测试
创建时间：2026-04-14 13:55:00
更新时间：2026-04-14 13:55:00

测试覆盖：
- lastlaneround 识别
- llastlaneid 分组
- 非候选文库正确分离
- 空输入处理
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from arrange_library.core.scheduling.mode_1_1_round2 import Mode11Round2Handler
from arrange_library.models.library_info import EnhancedLibraryInfo


def _make_lib(
    origrec: str = "LIB_001",
    lane_round: str = "",
    last_lane_round: str = "",
    last_laneid: str = "",
    contract_data_raw: float = 50.0,
    last_outrate: float | None = None,
    seq_mode: str = "1.1",
    last_seq_mode: str = "",
    add_tests_remark: str = "-",
) -> EnhancedLibraryInfo:
    """构造测试用文库对象"""
    lib = EnhancedLibraryInfo(
        origrec=origrec,
        sample_id="S001",
        sample_type_code="WES",
        data_type="其他",
        customer_library="否",
        base_type="双",
        number_of_bases=10,
        index_number=1,
        contract_data_raw=contract_data_raw,
        index_seq="AACCGGTT;TTGGCCAA",
        add_tests_remark=add_tests_remark,
        product_line="S",
        eq_type="Nova X-25B",
        peak_size=350,
        test_code=1595,
        test_no="Novaseq X Plus-PE150",
        sub_project_name="TEST_PROJECT",
        create_date="2026-04-14",
        delivery_date="2026-04-30",
        lab_type="诺禾-WES文库",
        data_volume_type="小数量",
        board_number="BN001",
    )
    lib.lane_round = lane_round if lane_round else None
    lib.last_lane_round = last_lane_round if last_lane_round else None
    lib.last_laneid = last_laneid if last_laneid else None
    lib._current_seq_mode_raw = seq_mode
    if last_seq_mode:
        lib.last_cxms = last_seq_mode
        lib._last_cxms_raw = last_seq_mode
    if last_outrate is not None:
        lib._last_outrate_raw = float(last_outrate)
    return lib


_SAMPLE_CONFIG = {
    "first_round_label": "1.1第一轮",
    "second_round_label": "1.1第二轮",
    "second_round_output_rate_threshold": 40.0,
    "second_round_default_pooling_factor": 2.5,
}


class TestRound2CandidateIdentification:
    """第二轮候选识别测试"""

    def test_first_round_lib_with_lastlaneid_is_candidate(self):
        lib = _make_lib(
            lane_round="",
            last_lane_round="1.1第一轮",
            last_laneid="LANE_A",
            seq_mode="1.0",
        )
        handler = Mode11Round2Handler(_SAMPLE_CONFIG)
        result = handler.identify_round2_candidates([lib])
        assert result.total_candidates == 1
        assert len(result.candidate_groups) == 1
        assert result.candidate_groups[0].last_lane_id == "LANE_A"

    def test_first_round_lib_without_lastlaneid_is_non_candidate(self):
        lib = _make_lib(lane_round="", last_lane_round="1.1第一轮", last_laneid="")
        handler = Mode11Round2Handler(_SAMPLE_CONFIG)
        result = handler.identify_round2_candidates([lib])
        assert result.total_candidates == 0
        assert len(result.non_candidates) == 1

    def test_non_first_round_lib_is_non_candidate(self):
        lib = _make_lib(lane_round="", last_lane_round="", last_laneid="LANE_A")
        handler = Mode11Round2Handler(_SAMPLE_CONFIG)
        result = handler.identify_round2_candidates([lib])
        assert result.total_candidates == 0
        assert len(result.non_candidates) == 1

    def test_second_round_lib_is_non_candidate(self):
        lib = _make_lib(lane_round="", last_lane_round="1.1第二轮", last_laneid="LANE_A")
        handler = Mode11Round2Handler(_SAMPLE_CONFIG)
        result = handler.identify_round2_candidates([lib])
        assert result.total_candidates == 0

    def test_non_1_1_mode_lib_is_non_candidate(self):
        lib = _make_lib(
            lane_round="",
            last_lane_round="1.1第一轮",
            last_laneid="LANE_A",
            seq_mode="3.6T-NEW",
        )
        handler = Mode11Round2Handler(_SAMPLE_CONFIG)
        result = handler.identify_round2_candidates([lib])
        assert result.total_candidates == 0

    def test_historical_1_1_mode_takes_priority_over_current_36t_mode(self):
        lib = _make_lib(
            lane_round="1.1第一轮",
            last_lane_round="1.1第一轮",
            last_laneid="LANE_A",
            seq_mode="3.6T-NEW",
            last_seq_mode="1",
        )
        handler = Mode11Round2Handler(_SAMPLE_CONFIG)
        result = handler.identify_round2_candidates([lib])
        assert result.total_candidates == 1
        assert len(result.candidate_groups) == 1

    def test_comma_joined_non_1_1_historical_mode_is_non_candidate(self):
        lib = _make_lib(
            lane_round="1.1第一轮",
            last_lane_round="1.1第一轮",
            last_laneid="LANE_A",
            seq_mode="3.6T-NEW",
            last_seq_mode="3.6T-NEW,3.6T-NEW",
        )
        handler = Mode11Round2Handler(_SAMPLE_CONFIG)
        result = handler.identify_round2_candidates([lib])
        assert result.total_candidates == 0

    def test_non_first_round_lastlaneround_is_non_candidate(self):
        lib = _make_lib(
            lane_round="1.1第一轮",
            last_lane_round="1.1第二轮",
            last_laneid="LANE_A",
            seq_mode="1.1",
        )
        handler = Mode11Round2Handler(_SAMPLE_CONFIG)
        result = handler.identify_round2_candidates([lib])
        assert result.total_candidates == 0

    def test_empty_input_returns_empty_result(self):
        handler = Mode11Round2Handler(_SAMPLE_CONFIG)
        result = handler.identify_round2_candidates([])
        assert result.total_candidates == 0
        assert len(result.candidate_groups) == 0
        assert len(result.non_candidates) == 0


class TestRound2Grouping:
    """第二轮按llastlaneid分组测试"""

    def test_same_lastlaneid_grouped_together(self):
        libs = [
            _make_lib(origrec="L1", lane_round="", last_lane_round="1.1第一轮", last_laneid="LANE_X"),
            _make_lib(origrec="L2", lane_round="", last_lane_round="1.1第一轮", last_laneid="LANE_X", seq_mode="1"),
        ]
        handler = Mode11Round2Handler(_SAMPLE_CONFIG)
        result = handler.identify_round2_candidates(libs)
        assert len(result.candidate_groups) == 1
        assert len(result.candidate_groups[0].libraries) == 2

    def test_different_lastlaneid_separate_groups(self):
        libs = [
            _make_lib(origrec="L1", lane_round="", last_lane_round="1.1第一轮", last_laneid="LANE_A"),
            _make_lib(origrec="L2", lane_round="", last_lane_round="1.1第一轮", last_laneid="LANE_B"),
        ]
        handler = Mode11Round2Handler(_SAMPLE_CONFIG)
        result = handler.identify_round2_candidates(libs)
        assert len(result.candidate_groups) == 2
        group_ids = {g.last_lane_id for g in result.candidate_groups}
        assert group_ids == {"LANE_A", "LANE_B"}

    def test_group_total_contract_gb_correct(self):
        libs = [
            _make_lib(origrec="L1", lane_round="", last_lane_round="1.1第一轮", last_laneid="LANE_X", contract_data_raw=100.0),
            _make_lib(origrec="L2", lane_round="", last_lane_round="1.1第一轮", last_laneid="LANE_X", contract_data_raw=200.0),
        ]
        handler = Mode11Round2Handler(_SAMPLE_CONFIG)
        result = handler.identify_round2_candidates(libs)
        assert abs(result.candidate_groups[0].total_contract_gb - 300.0) < 0.01

    def test_mixed_candidates_and_non_candidates(self):
        libs = [
            _make_lib(origrec="L1", lane_round="", last_lane_round="1.1第一轮", last_laneid="LANE_A"),
            _make_lib(origrec="L2", lane_round="", last_laneid=""),
            _make_lib(origrec="L3", lane_round="", last_lane_round="1.1第一轮", last_laneid="LANE_A"),
            _make_lib(origrec="L4", lane_round="", last_lane_round="1.1第一轮", last_laneid="LANE_B"),
        ]
        handler = Mode11Round2Handler(_SAMPLE_CONFIG)
        result = handler.identify_round2_candidates(libs)
        assert result.total_candidates == 3
        assert len(result.non_candidates) == 1
        assert len(result.candidate_groups) == 2


class TestRound2Scheduling:
    """第二轮按历史分组直出 lane 测试"""

    def test_schedule_round2_returns_lanes_with_round_metadata(self):
        libs = [
            _make_lib(origrec="L1", lane_round="", last_lane_round="1.1第一轮", last_laneid="LANE_X"),
            _make_lib(origrec="L2", lane_round="", last_lane_round="1.1第一轮", last_laneid="LANE_X"),
        ]
        handler = Mode11Round2Handler(_SAMPLE_CONFIG)
        identified = handler.identify_round2_candidates(libs)
        scheduled = handler.schedule_round2(identified.candidate_groups)

        assert len(scheduled.lanes) == 1
        assert scheduled.scheduled_groups == 1
        assert scheduled.broken_groups == 0
        assert len(scheduled.lanes[0].libraries) == 2
        assert scheduled.lanes[0].metadata["selected_seq_mode"] == "1.1"
        assert scheduled.lanes[0].metadata["selected_round_label"] == "1.1第二轮"
        assert scheduled.lanes[0].metadata["dispatch_stage"] == "second_round_1_1"
        assert scheduled.lanes[0].metadata["skip_strict_validation"] is True
        assert scheduled.lanes[0].metadata["mode_1_1_round2_source_last_lane_ids"] == ["LANE_X"]

    def test_schedule_round2_marks_low_output_pooling_factor(self):
        libs = [
            _make_lib(
                origrec="LOW_OUT",
                lane_round="",
                last_lane_round="1.1第一轮",
                last_laneid="LANE_Y",
                last_outrate=35.0,
                add_tests_remark="加测",
            ),
            _make_lib(
                origrec="NORMAL_OUT",
                lane_round="",
                last_lane_round="1.1第一轮",
                last_laneid="LANE_Y",
                last_outrate=55.0,
                add_tests_remark="混合",
            ),
        ]
        handler = Mode11Round2Handler(_SAMPLE_CONFIG)
        identified = handler.identify_round2_candidates(libs)
        scheduled = handler.schedule_round2(identified.candidate_groups)

        assert scheduled.lanes[0].metadata["mode_1_1_round2_pooling_factor"] == 2.5
        assert scheduled.lanes[0].metadata["mode_1_1_round2_low_output_origrecs"] == ["LOW_OUT"]

    def test_schedule_round2_marks_low_output_pooling_factor_for_decimal_rate(self):
        libs = [
            _make_lib(
                origrec="LOW_OUT_DECIMAL",
                lane_round="",
                last_lane_round="1.1第一轮",
                last_laneid="LANE_Z",
                last_outrate=0.35,
                add_tests_remark="加测",
            ),
        ]
        handler = Mode11Round2Handler(_SAMPLE_CONFIG)
        identified = handler.identify_round2_candidates(libs)
        scheduled = handler.schedule_round2(identified.candidate_groups)

        assert scheduled.lanes[0].metadata["mode_1_1_round2_pooling_factor"] == 2.5
        assert scheduled.lanes[0].metadata["mode_1_1_round2_low_output_origrecs"] == [
            "LOW_OUT_DECIMAL"
        ]

    def test_schedule_round2_does_not_mark_low_output_pooling_for_non_add_test(self):
        libs = [
            _make_lib(
                origrec="LOW_OUT_NON_ADD",
                lane_round="",
                last_lane_round="1.1第一轮",
                last_laneid="LANE_Q",
                last_outrate=35.0,
                add_tests_remark="-",
            ),
        ]
        handler = Mode11Round2Handler(_SAMPLE_CONFIG)
        identified = handler.identify_round2_candidates(libs)
        scheduled = handler.schedule_round2(identified.candidate_groups)

        assert "mode_1_1_round2_pooling_factor" not in scheduled.lanes[0].metadata
