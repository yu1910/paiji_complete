from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import types


PAIJI_ROOT = Path(__file__).resolve().parents[1]
if str(PAIJI_ROOT) not in sys.path:
    sys.path.insert(0, str(PAIJI_ROOT))

sys.modules.setdefault(
    "prediction_delivery",
    types.SimpleNamespace(MODELS_DIR=Path("."), predict_pooling=lambda *args, **kwargs: None),
)

import arrange_library.arrange_library_model6 as arrange_model6
from arrange_library.arrange_library_model6 import (
    _apply_add_test_output_rate_rule,
    _apply_add_test_output_rate_rule_to_prediction_df,
    _build_detail_output,
)
from arrange_library.models.library_info import EnhancedLibraryInfo


def test_ai_arrange_number_increments_for_schedulable_even_without_lane_assignment(
    tmp_path,
):
    output_path = tmp_path / "detail_output.csv"
    df_raw = pd.DataFrame(
        [
            {"wkorigrec": "A1", "origrec": "A1", "aiarrangenumber": 0, "wkuser": "user1"},
            {"wkorigrec": "A2", "origrec": "A2", "aiarrangenumber": 0, "wkuser": "user2"},
            {"wkorigrec": "A3", "origrec": "A3", "aiarrangenumber": 2, "wkuser": "user3"},
        ]
    )
    pred_df = pd.DataFrame(
        [
            {
                "origrec": "A1",
                "runid": "RUN_001",
                "lane_id": "LANE_001",
                "lsjnd": 1.5,
                "resolved_lsjfs": "25B",
                "resolved_lcxms": "3.6T-NEW",
                "resolved_index_check_rule": "P7P5",
                "wkbalancedata": 0.0,
                "predicted_lorderdata": 10.0,
                "lai_output": 9.0,
            }
        ]
    )

    _build_detail_output(
        df_raw=df_raw,
        pred_df=pred_df,
        output_path=output_path,
        ai_schedulable_keys={"A1", "A2"},
    )

    result = pd.read_csv(output_path)
    by_origrec = result.set_index("origrec_key").to_dict(orient="index")

    assert by_origrec["A1"]["aiarrangenumber"] == 1
    assert by_origrec["A2"]["aiarrangenumber"] == 1
    assert by_origrec["A3"]["aiarrangenumber"] == 2
    assert by_origrec["A1"]["llaneid"] == "LANE_001"
    assert pd.isna(by_origrec["A2"]["llaneid"])
    assert pd.isna(by_origrec["A2"]["wkuser"])


def test_add_test_output_rate_rule_uses_historical_rate_when_qpcr_within_15pct():
    lib = SimpleNamespace(
        add_tests_remark="这是加测",
        qpcr_molar=11.0,
        _last_qpcr_raw=10.0,
        _last_order_data_raw=400.0,
        _last_output_raw=100.0,
        _last_outrate_raw=None,
    )

    result = _apply_add_test_output_rate_rule(
        lib=lib,
        ai_predicted_order=350.0,
        ai_predicted_output=300.0,
        contract_data=120.0,
    )

    assert result["applied"] is True
    assert result["rule_reason"] == "qpcr_within_15pct_compare_ai_vs_historical"
    assert result["selected_order"] == 400.0
    assert result["historical_based_order"] == 400.0
    assert result["effective_last_outrate"] == 0.3
    assert result["wklastqpcr"] == 10.0
    assert result["wklastorderdata"] == 400.0
    assert result["wklastoutput"] == 100.0
    assert result["wklastoutrate"] == 0.25


def test_add_test_output_rate_rule_keeps_ai_prediction_when_qpcr_outside_15pct():
    lib = SimpleNamespace(
        add_tests_remark="加测",
        qpcr_molar=20.0,
        _last_qpcr_raw=10.0,
        _last_order_data_raw=400.0,
        _last_output_raw=300.0,
        _last_outrate_raw=None,
    )

    result = _apply_add_test_output_rate_rule(
        lib=lib,
        ai_predicted_order=350.0,
        ai_predicted_output=300.0,
        contract_data=120.0,
    )

    assert result["applied"] is True
    assert result["rule_reason"] == "qpcr_outside_15pct_use_ai"
    assert result["selected_order"] == 350.0
    assert result["historical_based_order"] is None


def test_prediction_df_add_test_output_rate_uses_independent_output_rate_floor():
    prediction_df = pd.DataFrame(
        [
            {
                "wkaddtestsremark": "加测",
                "lorderdata": 350.0,
                "wkqpcr": 20.0,
                "wklastqpcr": 10.0,
                "wklastorderdata": 400.0,
                "wklastoutput": 100.0,
                "wkoutputrate": 20,
                "wkcontractdata": 120.0,
            }
        ]
    )

    result = _apply_add_test_output_rate_rule_to_prediction_df(prediction_df)

    assert result.loc[0, "lorderdata"] == 400.0


def _build_library_for_split_test(origrec: str, contract_data_raw: float) -> EnhancedLibraryInfo:
    lib = EnhancedLibraryInfo(
        origrec=origrec,
        sample_id=f"SID_{origrec}",
        sample_type_code="普通文库",
        data_type="其他",
        customer_library="否",
        base_type="双",
        number_of_bases=10,
        index_number=1,
        index_seq="AACCGGTTAA;TTGGCCAATT",
        add_tests_remark="",
        product_line="S",
        peak_size=350,
        eq_type="Nova X-25B",
        contract_data_raw=contract_data_raw,
        test_code=1001,
        test_no="NovaSeq X Plus-PE150",
        sub_project_name="拆分测试项目",
        create_date="2026-04-08 09:00:00",
        delivery_date="2026-04-10 09:00:00",
        lab_type="普通文库",
        data_volume_type="标准",
        board_number="BOARD001",
        seq_scheme="3.6T-NEW",
    )
    lib.machine_type = arrange_model6.MachineType.NOVA_X_25B
    lib._origrec_key = origrec
    lib._source_origrec_key = origrec
    lib._detail_output_key = origrec
    return lib


def test_build_detail_output_expands_split_fragments_with_child_granularity(tmp_path):
    output_path = tmp_path / "split_detail_output.csv"
    df_raw = pd.DataFrame(
        [
            {
                "wkaidbid": "RAW_A1",
                "wkorigrec": "A1",
                "origrec": "A1",
                "aiarrangenumber": 0,
                "wkuser": "user1",
                "wkcontractdata": 120.0,
                "wktotalcontractdata": 120.0,
            }
        ]
    )
    pred_df = pd.DataFrame(
        [
            {
                "origrec": "A1",
                "origrec_key": "A1",
                "detail_row_key": "A1_F001",
                "runid": "RUN_001",
                "lane_id": "LANE_001",
                "lsjnd": 1.5,
                "resolved_lsjfs": "25B",
                "resolved_lcxms": "3.6T-NEW",
                "resolved_index_check_rule": "P7P5",
                "wkbalancedata": 0.0,
                "predicted_lorderdata": 60.0,
                "lai_output": 55.0,
            }
        ]
    )
    detail_libraries = [
        SimpleNamespace(
            origrec="A1",
            _origrec_key="A1",
            _source_origrec_key="A1",
            _detail_output_key="A1_F001",
            contract_data_raw=60.0,
            total_contract_data=120.0,
            wkaidbid="AID_001",
            aidbid="AID_001",
            is_split=True,
            wkissplit="yes",
            split_status="completed",
            fragment_id="A1_F001",
            fragment_index=1,
            package_lane_number="",
            baleno="",
        ),
        SimpleNamespace(
            origrec="A1",
            _origrec_key="A1",
            _source_origrec_key="A1",
            _detail_output_key="A1_F002",
            contract_data_raw=60.0,
            total_contract_data=120.0,
            wkaidbid="AID_002",
            aidbid="AID_002",
            is_split=True,
            wkissplit="yes",
            split_status="completed",
            fragment_id="A1_F002",
            fragment_index=2,
            package_lane_number="",
            baleno="",
        ),
    ]

    _build_detail_output(
        df_raw=df_raw,
        pred_df=pred_df,
        output_path=output_path,
        ai_schedulable_keys={"A1"},
        detail_libraries=detail_libraries,
    )

    result = pd.read_csv(output_path, keep_default_na=False)
    split_rows = result.loc[result["origrec_key"] == "A1"].copy()

    assert len(split_rows) == 2
    assert set(split_rows["wkaidbid"]) == {"AID_001", "AID_002"}
    assert set(split_rows["wkissplit"]) == {"yes"}
    assigned_rows = split_rows.loc[split_rows["llaneid"] == "LANE_001"]
    assert len(assigned_rows) == 1
    assert float(assigned_rows.iloc[0]["wkcontractdata"]) == 60.0
    unassigned_rows = split_rows.loc[split_rows["llaneid"] == ""]
    assert len(unassigned_rows) == 1
    assert unassigned_rows.iloc[0]["wkuser"] == ""


def test_test_with_model_splits_before_mixed_lane_prebuild(monkeypatch):
    source_lib = _build_library_for_split_test("PRE_SPLIT", 300.0)
    captured: dict[str, object] = {}

    def fake_extract(*, libraries, **kwargs):
        captured["mixed_input"] = [getattr(lib, "fragment_id", lib.origrec) for lib in libraries]
        captured["mixed_kwargs"] = dict(kwargs)
        return [], list(libraries)

    def fake_schedule(self, libraries, keep_failed_lanes=False, libraries_already_split=False, perform_presplit_family_rollback=True):
        captured["schedule_input"] = [getattr(lib, "fragment_id", lib.origrec) for lib in libraries]
        captured["libraries_already_split"] = libraries_already_split
        captured["perform_presplit_family_rollback"] = perform_presplit_family_rollback
        return SimpleNamespace(lane_assignments=[], unassigned_libraries=list(libraries))

    monkeypatch.setattr(arrange_model6, "_extract_mixed_lanes_by_peak_window", fake_extract)
    monkeypatch.setattr(arrange_model6.GreedyLaneScheduler, "schedule", fake_schedule)
    monkeypatch.setattr(
        arrange_model6,
        "_enforce_special_split_constraints_with_local_swap",
        lambda solution, strict_validator, max_passes: {"changed_lanes": 0, "removed_libraries": 0, "swapped_in_libraries": 0},
    )
    monkeypatch.setattr(arrange_model6, "_try_increase_lane_count", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        arrange_model6,
        "try_multi_lib_swap_rebalance",
        lambda *args, **kwargs: {"new_lanes": 0, "remaining_unassigned": len(args[0].unassigned_libraries)},
    )

    _, solution = arrange_model6.test_with_model([source_lib])

    assert captured["libraries_already_split"] is True
    assert captured["perform_presplit_family_rollback"] is False
    mixed_input = captured["mixed_input"]
    assert mixed_input == ["PRE_SPLIT_F001", "PRE_SPLIT_F002", "PRE_SPLIT_F003", "PRE_SPLIT_F004"]
    assert captured["mixed_kwargs"]["index_conflict_attempts_per_lane"] == 10
    assert captured["mixed_kwargs"]["other_failure_attempts_per_lane"] == 20
    assert captured["schedule_input"] == mixed_input
    assert len(solution.unassigned_libraries) == 1
    assert solution.unassigned_libraries[0].origrec == "PRE_SPLIT"


def test_extract_mixed_lanes_uses_scattered_mix_priority_and_new_retry_limits(monkeypatch):
    libraries = [
        _build_library_for_split_test("MIX_A", 60.0),
        _build_library_for_split_test("MIX_B", 60.0),
    ]
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        arrange_model6,
        "_find_best_peak_size_window",
        lambda libs: (300.0, 350.0, list(libs)),
    )
    monkeypatch.setattr(
        arrange_model6,
        "_resolve_lane_capacity_limits",
        lambda libraries, machine_type, lane_id="", lane_metadata=None: (50.0, 100.0),
    )

    def fake_attempt_build_lane_from_pool(
        *,
        pool,
        validator,
        machine_type,
        lane_id_prefix,
        lane_serial=None,
        index_conflict_attempts=0,
        other_failure_attempts=0,
        extra_metadata=None,
        prioritize_scattered_mix=False,
    ):
        captured["prioritize_scattered_mix"] = prioritize_scattered_mix
        captured["index_conflict_attempts"] = index_conflict_attempts
        captured["other_failure_attempts"] = other_failure_attempts
        return None, []

    monkeypatch.setattr(arrange_model6, "_attempt_build_lane_from_pool", fake_attempt_build_lane_from_pool)

    mixed_lanes, remaining_libraries = arrange_model6._extract_mixed_lanes_by_peak_window(
        libraries=libraries,
        validator=SimpleNamespace(),
        machine_type=arrange_model6.MachineType.NOVA_X_25B,
    )

    assert mixed_lanes == []
    assert remaining_libraries == libraries
    assert captured["prioritize_scattered_mix"] is True
    assert captured["index_conflict_attempts"] == 10
    assert captured["other_failure_attempts"] == 20
