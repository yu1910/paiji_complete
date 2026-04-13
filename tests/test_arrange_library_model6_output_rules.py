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
    BALANCE_LIBRARY_MARKER_COLUMN,
    _apply_add_test_output_rate_rule,
    _apply_add_test_output_rate_rule_to_prediction_df,
    _build_detail_output,
    _find_conflicting_lane_libraries,
    _get_lane_balance_templates,
    _materialize_balance_library_for_lane,
    _resolve_lane_balance_data_gb,
    _trim_lane_for_balance_capacity,
    _run_prediction_delivery,
)
from arrange_library.core.scheduling.scheduling_types import LaneAssignment
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


def _build_basic_lane_library(origrec: str, index_seq: str = "AACCGGTTAA;TTGGCCAATT", data: float = 950.0):
    lib = EnhancedLibraryInfo(
        origrec=origrec,
        sample_id=f"SID_{origrec}",
        sample_type_code="普通文库",
        data_type="其他",
        customer_library="否",
        base_type="双",
        number_of_bases=10,
        index_number=1,
        index_seq=index_seq,
        add_tests_remark="",
        product_line="S",
        peak_size=350,
        eq_type="Nova X-25B",
        contract_data_raw=data,
        test_code=1001,
        test_no="Novaseq X Plus-PE150",
        sub_project_name="平衡文库测试项目",
        create_date="2026-04-10 09:00:00",
        delivery_date="2026-04-12 09:00:00",
        lab_type="普通文库",
        data_volume_type="标准",
        board_number="BOARD001",
    )
    lib._wkdept_raw = "天津科技服务实验室"
    lib._origrec_key = origrec
    lib._source_origrec_key = origrec
    lib._detail_output_key = origrec
    lib.machine_type = arrange_model6.MachineType.NOVA_X_25B
    return lib


def test_get_lane_balance_templates_prefers_phix_without_pe_and_skips_with_pe(monkeypatch):
    templates = {
        (
            arrange_model6._normalize_text_for_match("天津科技服务实验室"),
            arrange_model6._normalize_text_for_match("Novaseq X Plus-PE150"),
        ): [
            {
                "wksampleid": "other_balance",
                "wkdept": "天津科技服务实验室",
                "wktestno": "Novaseq X Plus-PE150",
                "wkindexseq": "ACGTACGTAC;TGCATGCATG",
                "_template_order": 1,
            },
            {
                "wksampleid": "phix",
                "wkdept": "天津科技服务实验室",
                "wktestno": "Novaseq X Plus-PE150",
                "wkindexseq": "GGGGGGGGGG;ACCGAGATCT",
                "_template_order": 2,
            },
        ]
    }
    monkeypatch.setattr(arrange_model6, "_load_balance_library_templates", lambda: templates)

    lane = LaneAssignment(
        lane_id="NB_001",
        machine_id="M_001",
        machine_type=arrange_model6.MachineType.NOVA_X_25B,
        libraries=[_build_basic_lane_library("LIB001")],
    )
    ordered_templates = _get_lane_balance_templates(lane)
    assert [item["wksampleid"] for item in ordered_templates] == ["phix", "other_balance"]

    pe_lane = LaneAssignment(
        lane_id="NB_002",
        machine_id="M_002",
        machine_type=arrange_model6.MachineType.NOVA_X_25B,
        libraries=[_build_basic_lane_library("LIB002", index_seq="PE")],
    )
    ordered_templates = _get_lane_balance_templates(pe_lane)
    assert [item["wksampleid"] for item in ordered_templates] == ["other_balance"]


def test_resolve_lane_balance_data_gb_skips_non_dedicated_lane_even_with_imbalance_type():
    lane = LaneAssignment(
        lane_id="NB_003",
        machine_id="M_003",
        machine_type=arrange_model6.MachineType.NOVA_X_25B,
        lane_capacity_gb=975.0,
        libraries=[_build_basic_lane_library("LIB003", data=780.0)],
    )
    lane.libraries[0].sample_type_code = "ATAC-seq文库"
    lane.libraries[0].data_type = "ATAC-seq文库"

    assert _resolve_lane_balance_data_gb(lane) == 0.0


def test_resolve_lane_balance_data_gb_uses_lane_capacity_for_dedicated_imbalance_lane():
    lane = LaneAssignment(
        lane_id="DL_001",
        machine_id="M_004",
        machine_type=arrange_model6.MachineType.NOVA_X_25B,
        lane_capacity_gb=975.0,
        libraries=[_build_basic_lane_library("LIB004", data=780.0)],
        metadata={"is_dedicated_imbalance_lane": True},
    )
    lane.libraries[0].sample_type_code = "ATAC-seq文库"
    lane.libraries[0].data_type = "ATAC-seq文库"

    assert _resolve_lane_balance_data_gb(lane) == 195.0


def test_resolve_lane_balance_data_gb_does_not_use_explicit_value_when_ratio_is_zero():
    lane = LaneAssignment(
        lane_id="DL_002",
        machine_id="M_005",
        machine_type=arrange_model6.MachineType.NOVA_X_25B,
        lane_capacity_gb=975.0,
        libraries=[_build_basic_lane_library("LIB007", data=780.0)],
        metadata={"is_dedicated_imbalance_lane": True, "wkbalancedata": 123.0},
    )
    lane.libraries[0].sample_type_code = "10X转录组-3'文库"
    lane.libraries[0].data_type = "10X转录组-3'文库"

    assert _resolve_lane_balance_data_gb(lane) == 0.0


def test_find_conflicting_lane_libraries_only_returns_normal_libraries(monkeypatch):
    normal_lib = _build_basic_lane_library("LIB008", data=120.0)
    imbalance_lib = _build_basic_lane_library("LIB009", data=120.0, index_seq="CCAATTGGCC;GGTTAACCAA")
    imbalance_lib.sample_type_code = "ATAC-seq文库"
    imbalance_lib.data_type = "ATAC-seq文库"
    candidate = SimpleNamespace(origrec="AI_BALANCE_LANE_TEST")

    monkeypatch.setattr(
        arrange_model6,
        "_validate_index_conflicts_latest",
        lambda libraries: [
            SimpleNamespace(record_id_1="AI_BALANCE_LANE_TEST", record_id_2="LIB008"),
            SimpleNamespace(record_id_1="AI_BALANCE_LANE_TEST", record_id_2="LIB009"),
        ],
    )

    conflicts = _find_conflicting_lane_libraries([normal_lib, imbalance_lib], candidate)

    assert [lib.origrec for lib in conflicts] == ["LIB008"]


def test_trim_lane_for_balance_capacity_only_removes_normal_libraries(monkeypatch):
    imbalance_lib = _build_basic_lane_library("LIB010", data=600.0)
    imbalance_lib.sample_type_code = "ATAC-seq文库"
    imbalance_lib.data_type = "ATAC-seq文库"
    normal_lib = _build_basic_lane_library("LIB011", data=300.0, index_seq="GGAATTCCGG;AACCGGTTAA")

    lane = LaneAssignment(
        lane_id="DL_003",
        machine_id="M_006",
        machine_type=arrange_model6.MachineType.NOVA_X_25B,
        lane_capacity_gb=975.0,
        libraries=[imbalance_lib, normal_lib],
        metadata={"is_dedicated_imbalance_lane": True},
    )
    candidate_balance_lib = arrange_model6._create_balance_library_from_template(
        lane=lane,
        template={
            "wksampleid": "phix",
            "wkdept": "天津科技服务实验室",
            "wktestno": "Novaseq X Plus-PE150",
            "wkindexseq": "GGGGGGGGGG;ACCGAGATCT",
        },
        balance_amount_gb=195.0,
    )

    monkeypatch.setattr(
        arrange_model6,
        "_validate_lane_state",
        lambda validator, lane, libraries: SimpleNamespace(
            is_valid=sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in libraries) <= 1000.0
        ),
    )

    trimmed_result = _trim_lane_for_balance_capacity(
        lane=lane,
        working_libs=[imbalance_lib, normal_lib],
        candidate_balance_lib=candidate_balance_lib,
        validator=SimpleNamespace(),
    )

    assert trimmed_result is not None
    trimmed_libs, removed_libs = trimmed_result
    assert [lib.origrec for lib in removed_libs] == ["LIB011"]
    assert "LIB010" in [lib.origrec for lib in trimmed_libs]


def test_materialize_balance_library_for_dedicated_56_57_mix_lane(monkeypatch):
    templates = {
        (
            arrange_model6._normalize_text_for_match("天津科技服务实验室"),
            arrange_model6._normalize_text_for_match("Novaseq X Plus-PE150"),
        ): [
            {
                "wksampleid": "phix",
                "wkdept": "天津科技服务实验室",
                "wktestno": "Novaseq X Plus-PE150",
                "wkindexseq": "GGGGGGGGGG;ACCGAGATCT",
                "_template_order": 1,
            }
        ]
    }
    monkeypatch.setattr(arrange_model6, "_load_balance_library_templates", lambda: templates)
    monkeypatch.setattr(
        arrange_model6,
        "_validate_lane_state",
        lambda validator, lane, libraries: SimpleNamespace(is_valid=True),
    )
    monkeypatch.setattr(arrange_model6, "_validate_index_conflicts_latest", lambda libraries: [])

    imbalance_lib = _build_basic_lane_library("LIB005", data=180.0)
    imbalance_lib.sample_type_code = "ATAC-seq文库"
    imbalance_lib.data_type = "ATAC-seq文库"
    balanced_lib = _build_basic_lane_library("LIB006", data=600.0, index_seq="TTAACCGGTT;CCAATTGGCC")
    balanced_lib.sample_type_code = "真核普通转录组文库"
    balanced_lib.data_type = "真核普通转录组文库"

    is_compatible, _ = arrange_model6._BASE_IMBALANCE_HANDLER.check_mix_compatibility(
        [imbalance_lib, balanced_lib],
        enforce_total_limit=False,
    )
    assert is_compatible is True

    lane = LaneAssignment(
        lane_id="DL_056_001",
        machine_id="M_056_001",
        machine_type=arrange_model6.MachineType.NOVA_X_25B,
        lane_capacity_gb=975.0,
        libraries=[imbalance_lib, balanced_lib],
        metadata={"is_dedicated_imbalance_lane": True},
    )

    added = _materialize_balance_library_for_lane(
        lane=lane,
        all_lanes=[lane],
        unassigned_pool=[],
        validator=SimpleNamespace(),
    )

    assert added is True
    assert lane.metadata["wkbalancedata"] == 195.0
    assert lane.metadata["materialized_balance_library"] is True

    balance_libs = [lib for lib in lane.libraries if getattr(lib, BALANCE_LIBRARY_MARKER_COLUMN, False)]
    assert len(balance_libs) == 1
    balance_lib = balance_libs[0]
    assert balance_lib.sample_id == "phix"
    assert balance_lib.is_add_balance == "是"
    assert balance_lib.contract_data_raw == 195.0
    assert balance_lib._balance_output_payload["wkcontractdata"] == 195.0
    assert balance_lib._balance_output_payload["lorderdata"] == 195.0


def test_build_detail_output_generates_balance_library_row_without_raw_template(tmp_path):
    output_path = tmp_path / "balance_detail_output.csv"
    df_raw = pd.DataFrame(
        [{"wkorigrec": "RAW001", "origrec": "RAW001", "aiarrangenumber": 0, "wkuser": "user1"}]
    )
    pred_df = pd.DataFrame(
        [
            {
                "origrec": "AI_BALANCE_LANE_001",
                "origrec_key": "AI_BALANCE_LANE_001",
                "detail_row_key": "AID_BAL_001",
                "runid": "RUN_001",
                "lane_id": "LANE_001",
                "lsjnd": 1.5,
                "resolved_lsjfs": "25B",
                "resolved_lcxms": "3.6T-NEW",
                "resolved_index_check_rule": "P7P5",
                "wkbalancedata": 20.0,
                "predicted_lorderdata": 20.0,
                "lai_output": None,
                BALANCE_LIBRARY_MARKER_COLUMN: True,
            }
        ]
    )
    detail_libraries = [
        SimpleNamespace(
            origrec="AI_BALANCE_LANE_001",
            _origrec_key="AI_BALANCE_LANE_001",
            _source_origrec_key="AI_BALANCE_LANE_001",
            _detail_output_key="AID_BAL_001",
            contract_data_raw=20.0,
            wkaidbid="AID_BAL_001",
            aidbid="AID_BAL_001",
            _is_ai_balance_library=True,
            _balance_output_payload={
                "wkaidbid": "AID_BAL_001",
                "wksampleid": "phix",
                "wkdept": "天津科技服务实验室",
                "wktestno": "Novaseq X Plus-PE150",
                "wkindexseq": "GGGGGGGGGG;ACCGAGATCT",
                "wkcontractdata": 20.0,
                "origrec": "AI_BALANCE_LANE_001",
                "origrec_key": "AI_BALANCE_LANE_001",
                "detail_row_key": "AID_BAL_001",
                BALANCE_LIBRARY_MARKER_COLUMN: True,
            },
        )
    ]

    _build_detail_output(
        df_raw=df_raw,
        pred_df=pred_df,
        output_path=output_path,
        ai_schedulable_keys={"RAW001"},
        detail_libraries=detail_libraries,
    )

    result = pd.read_csv(output_path, keep_default_na=False)
    balance_rows = result.loc[result[BALANCE_LIBRARY_MARKER_COLUMN].astype(str).str.lower() == "true"].copy()
    assert len(balance_rows) == 1
    assert balance_rows.iloc[0]["wksampleid"] == "phix"
    assert float(balance_rows.iloc[0]["wkcontractdata"]) == 20.0
    assert float(balance_rows.iloc[0]["lorderdata"]) == 20.0
    assert balance_rows.iloc[0]["llaneid"] == "LANE_001"


def test_run_prediction_delivery_restores_balance_library_lorderdata_and_clears_output(tmp_path, monkeypatch):
    output_path = tmp_path / "prediction.csv"

    def _fake_predict_pooling(input_data, output_file):
        pd.DataFrame(
            [
                {
                    "wkcontractdata": 20.0,
                    "lorderdata": 999.0,
                    "lai_output": 888.0,
                    BALANCE_LIBRARY_MARKER_COLUMN: True,
                    "llaneid": "LANE_001",
                },
                {
                    "wkcontractdata": 10.0,
                    "lorderdata": 11.0,
                    "lai_output": 9.0,
                    BALANCE_LIBRARY_MARKER_COLUMN: False,
                    "llaneid": "LANE_001",
                },
            ]
        ).to_csv(output_file, index=False)

    monkeypatch.setattr(arrange_model6, "predict_pooling", _fake_predict_pooling)
    monkeypatch.setattr(
        arrange_model6,
        "_apply_add_test_output_rate_rule_to_prediction_df",
        lambda prediction_df, output_path=None: prediction_df,
    )

    result = _run_prediction_delivery(input_data=pd.DataFrame(), output_path=output_path)
    assert BALANCE_LIBRARY_MARKER_COLUMN not in result.columns
    assert float(result.loc[0, "lorderdata"]) == 20.0
    assert pd.isna(result.loc[0, "lai_output"])
    assert float(result.loc[1, "lorderdata"]) == 11.0

    persisted = pd.read_csv(output_path)
    assert BALANCE_LIBRARY_MARKER_COLUMN not in persisted.columns
    assert float(persisted.loc[0, "lorderdata"]) == 20.0
