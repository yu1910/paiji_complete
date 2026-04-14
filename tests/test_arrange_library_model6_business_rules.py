from pathlib import Path
import sys
import types
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import pytest


PAIJI_ROOT = Path(__file__).resolve().parents[1]
if str(PAIJI_ROOT) not in sys.path:
    sys.path.insert(0, str(PAIJI_ROOT))

sys.modules.setdefault(
    "prediction_delivery",
    types.SimpleNamespace(MODELS_DIR=Path("."), predict_pooling=lambda *args, **kwargs: None),
)

import arrange_library.arrange_library_model6 as arrange_model6
from arrange_library.arrange_library_model6 import (
    _collect_donations_for_pool,
    _apply_add_test_output_rate_rule_to_prediction_df,
    _attempt_build_lane_from_pool,
    _build_detail_output,
    _collect_prediction_rows,
    _ensure_unique_lane_ids,
    _filter_valid_lanes,
    _reset_auto_lane_serial_counters,
    _reserve_auto_lane_serial,
    _rescue_remaining_lanes_by_layered_regroup_search,
    _rescue_failed_lanes_by_57_rules,
    _resolve_lane_loading_concentration,
    try_multi_lib_swap_rebalance,
    _validate_final_package_lanes,
    _validate_no_split_for_package_lane_libraries,
    _validate_package_lane_rules,
)
from arrange_library.core.data.library_loader import load_libraries_from_csv
from arrange_library.core.constraints.lane_validator import LaneValidator
from arrange_library.core.config.scheduling_config import LaneRuleSelection
from arrange_library.core.config.scheduling_config import get_scheduling_config
from arrange_library.core.scheduling.greedy_lane_scheduler import GreedyLaneScheduler
from arrange_library.core.scheduling.package_lane_scheduler import PackageLaneScheduler, PackageType
from arrange_library.core.scheduling.scheduling_types import LaneAssignment, SchedulingSolution
from arrange_library.models.library_info import EnhancedLibraryInfo, MachineType


def _build_library(
    *,
    origrec: str,
    sample_type: str = "普通文库",
    data_type: str = "其他",
    contract_data: float = 100.0,
    index_seq: str = "AACCGGTTAA;TTGGCCAATT",
    add_tests_remark: str = "",
    sub_project_name: str = "常规项目",
    seq_scheme: str = "",
    machine_note: str = "",
    customer_library: str = "否",
    sample_id: str | None = None,
    board_number: str = "BOARD001",
    delete_date: float | None = None,
    jjbj: str = "否",
    package_lane_number: str = "",
    qpcr_molar: float | None = None,
    output_rate: float | None = None,
) -> EnhancedLibraryInfo:
    lib = EnhancedLibraryInfo(
        origrec=origrec,
        sample_id=sample_id or f"SID_{origrec}",
        sample_type_code=sample_type,
        data_type=data_type,
        customer_library=customer_library,
        base_type="双",
        number_of_bases=10,
        index_number=1,
        index_seq=index_seq,
        add_tests_remark=add_tests_remark,
        product_line="S",
        peak_size=350,
        eq_type="Nova X-25B",
        contract_data_raw=contract_data,
        test_code=1001,
        test_no="NovaSeq X Plus-PE150",
        sub_project_name=sub_project_name,
        create_date="2026-04-08 09:00:00",
        delivery_date="2026-04-10 09:00:00",
        lab_type=sample_type,
        data_volume_type="标准",
        board_number=board_number,
        package_lane_number=package_lane_number or None,
        is_package_lane="是" if package_lane_number else None,
        qpcr_molar=qpcr_molar,
        qpcr_concentration=qpcr_molar,
        seq_scheme=seq_scheme,
        machine_note=machine_note,
        output_rate=output_rate,
    )
    lib.jjbj = jjbj
    if package_lane_number:
        lib.package_lane_number = package_lane_number
        lib.baleno = package_lane_number
        lib.is_package_lane = "是"
    if delete_date is not None:
        lib._delete_date_raw = delete_date
    return lib


def _empty_prediction_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "origrec",
            "runid",
            "lane_id",
            "lsjnd",
            "resolved_lsjfs",
            "resolved_lcxms",
            "resolved_index_check_rule",
            "wkbalancedata",
            "predicted_lorderdata",
            "lai_output",
        ]
    )


def test_lane_capacity_rule_ignores_last_cxms_history_for_current_mode_resolution():
    scheduling_config = get_scheduling_config()
    lib_history = _build_library(origrec="LIB_HIST", sample_type="外显子文库")
    lib_history.test_code = 1595
    lib_history._current_seq_mode_raw = ""
    lib_history.last_cxms = "1.0"
    lib_history._last_cxms_raw = "1.0"

    lib_current = _build_library(origrec="LIB_CURR", sample_type="DNA小片段文库")
    lib_current.test_code = 1595
    lib_current._current_seq_mode_raw = "3.6T-NEW"
    lib_current.last_cxms = ""
    lib_current._last_cxms_raw = ""

    result = scheduling_config.get_lane_capacity_range(
        libraries=[lib_history, lib_current],
        machine_type="Nova X-25B",
    )

    assert result.rule_code == "tj_1595_standard_pe150_25b"
    assert result.effective_min_gb == 995.0
    assert result.effective_max_gb == 1105.0


def _build_lane_assignment(
    lane_id: str,
    libraries: Sequence[EnhancedLibraryInfo],
    *,
    package_id: str | None = None,
) -> LaneAssignment:
    lane = LaneAssignment(
        lane_id=lane_id,
        machine_id=f"M_{lane_id}",
        machine_type=MachineType.NOVA_X_25B,
        lane_capacity_gb=1000.0,
    )
    for lib in libraries:
        lane.add_library(lib)
    if package_id:
        lane.metadata["package_id"] = package_id
    return lane


def _lane_rule(
    *,
    min_gb: float = 100.0,
    max_gb: float = 100.0,
) -> LaneRuleSelection:
    return LaneRuleSelection(
        rule_code="unit_test_rule",
        soft_target_gb=max_gb,
        min_target_gb=min_gb,
        max_target_gb=max_gb,
        tolerance_gb=0.0,
        effective_min_gb=min_gb,
        effective_max_gb=max_gb,
        lane_count=8,
        loading_method="25B",
        sequencing_mode="3.6T-NEW",
        fc_min_data_gb=0.0,
        profile={},
    )


def _flatten_package_lanes(result) -> List[LaneAssignment]:
    flattened: List[LaneAssignment] = []
    for run in result.runs:
        for lane in run.lanes:
            if lane.package_type != PackageType.PACKAGE_LANE:
                continue
            flattened.append(
                _build_lane_assignment(
                    lane_id=lane.lane_id,
                    libraries=lane.libraries,
                    package_id=lane.package_id,
                )
            )
    return flattened


def test_build_detail_output_increments_aiarrangenumber_for_schedulable_unassigned_row(
    tmp_path: Path,
) -> None:
    df_raw = pd.DataFrame(
        [
            {
                "origrec": "LIB_UNASSIGNED",
                "wkorigrec": "LIB_UNASSIGNED",
                "aiarrangenumber": 0,
                "wkuser": "tester",
                "llaneid": "",
                "lrunid": "",
            }
        ]
    )

    output_path = tmp_path / "detail_unassigned.csv"
    _build_detail_output(
        df_raw=df_raw,
        pred_df=_empty_prediction_df(),
        output_path=output_path,
        ai_schedulable_keys={"LIB_UNASSIGNED"},
    )

    result = pd.read_csv(output_path)
    assert int(result.loc[0, "aiarrangenumber"]) == 1
    assert pd.isna(result.loc[0, "llaneid"]) or result.loc[0, "llaneid"] == ""
    assert pd.isna(result.loc[0, "lrunid"]) or result.loc[0, "lrunid"] == ""


def test_build_detail_output_increments_aiarrangenumber_for_schedulable_assigned_row(
    tmp_path: Path,
) -> None:
    df_raw = pd.DataFrame(
        [
            {
                "origrec": "LIB_ASSIGNED",
                "wkorigrec": "LIB_ASSIGNED",
                "aiarrangenumber": 0,
                "llaneid": "",
                "lrunid": "",
            }
        ]
    )
    pred_df = pd.DataFrame(
        [
            {
                "origrec": "LIB_ASSIGNED",
                "runid": "RUN_001",
                "lane_id": "LANE_001",
                "lsjnd": 2.3,
                "resolved_lsjfs": "25B",
                "resolved_lcxms": "3.6T-NEW",
                "resolved_index_check_rule": "P7P5",
                "wkbalancedata": None,
                "predicted_lorderdata": 180.0,
                "lai_output": 160.0,
            }
        ]
    )

    output_path = tmp_path / "detail_assigned.csv"
    _build_detail_output(
        df_raw=df_raw,
        pred_df=pred_df,
        output_path=output_path,
        ai_schedulable_keys={"LIB_ASSIGNED"},
    )

    result = pd.read_csv(output_path)
    assert int(result.loc[0, "aiarrangenumber"]) == 1
    assert result.loc[0, "llaneid"] == "LANE_001"
    assert result.loc[0, "lrunid"] == "RUN_001"


def test_reserve_auto_lane_serial_increments_per_prefix_and_machine() -> None:
    _reset_auto_lane_serial_counters()

    first = _reserve_auto_lane_serial("EX", MachineType.NOVA_X_25B)
    second = _reserve_auto_lane_serial("EX", MachineType.NOVA_X_25B)
    third = _reserve_auto_lane_serial("RB", MachineType.NOVA_X_25B)

    assert first == 1
    assert second == 2
    assert third == 1


def test_attempt_build_lane_from_pool_uses_unique_auto_serials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_auto_lane_serial_counters()
    libs_a = [_build_library(origrec="AUTO_1", contract_data=60.0)]
    libs_b = [_build_library(origrec="AUTO_2", contract_data=60.0)]

    monkeypatch.setattr(
        arrange_model6,
        "_filter_libraries_by_hard_priority",
        lambda libraries, **kwargs: list(libraries),
    )
    monkeypatch.setattr(
        arrange_model6,
        "_validate_lane_special_split_rule",
        lambda libraries: (True, set(), ""),
    )
    monkeypatch.setattr(
        arrange_model6,
        "_validate_lane_57_mix_rules",
        lambda libraries, enforce_total_limit=False: (True, ""),
    )
    monkeypatch.setattr(
        arrange_model6,
        "_resolve_lane_capacity_limits",
        lambda libraries, machine_type, lane_id="", lane_metadata=None: (50.0, 200.0),
    )
    monkeypatch.setattr(
        arrange_model6._MODULE_IDX_VALIDATOR,
        "validate_new_lib_quick_with_cache",
        lambda selected_idx_cache, lib: (True, []),
    )
    monkeypatch.setattr(
        arrange_model6,
        "_validate_lane_with_latest_index",
        lambda **kwargs: types.SimpleNamespace(is_valid=True, errors=[], warnings=[]),
    )

    lane_one, used_one = _attempt_build_lane_from_pool(
        pool=list(libs_a),
        validator=object(),
        machine_type=MachineType.NOVA_X_25B,
        lane_id_prefix="EX",
        lane_serial=None,
        index_conflict_attempts=1,
        other_failure_attempts=1,
    )
    lane_two, used_two = _attempt_build_lane_from_pool(
        pool=list(libs_b),
        validator=object(),
        machine_type=MachineType.NOVA_X_25B,
        lane_id_prefix="EX",
        lane_serial=None,
        index_conflict_attempts=1,
        other_failure_attempts=1,
    )

    assert lane_one is not None and used_one
    assert lane_two is not None and used_two
    assert lane_one.lane_id == "EX_Nova X-25B_001"
    assert lane_two.lane_id == "EX_Nova X-25B_002"


def test_ensure_unique_lane_ids_renames_duplicate_lane_assignments() -> None:
    lane_a = _build_lane_assignment(
        "EX_Nova X-25B_001",
        [_build_library(origrec="DUP_A", contract_data=60.0)],
    )
    lane_b = _build_lane_assignment(
        "EX_Nova X-25B_001",
        [_build_library(origrec="DUP_B", contract_data=60.0)],
    )

    renamed = _ensure_unique_lane_ids([lane_a, lane_b])

    assert renamed == 1
    assert lane_a.lane_id == "EX_Nova X-25B_001"
    assert lane_b.lane_id == "EX_Nova X-25B_002"
    assert lane_b.machine_id == "M_EX_Nova X-25B_002"


def test_collect_prediction_rows_keeps_wkbalancedata_empty_for_non_package_lane() -> None:
    lane = _build_lane_assignment(
        "RS_Nova X-25B_001",
        [
            _build_library(origrec="LIB_NORMAL", contract_data=950.0),
            _build_library(origrec="LIB_BALANCE", sample_type="平衡文库", sample_id="phix", contract_data=195.0),
        ],
    )
    lane.metadata["wkbalancedata"] = 195.0
    setattr(lane.libraries[1], arrange_model6.BALANCE_LIBRARY_MARKER_COLUMN, True)

    result = _collect_prediction_rows([lane], {}, "unit_test")

    normal_row = result.loc[result["origrec"] == "LIB_NORMAL"].iloc[0]
    balance_row = result.loc[result["origrec"] == "LIB_BALANCE"].iloc[0]
    assert pd.isna(normal_row["wkbalancedata"])
    assert pd.isna(balance_row["wkbalancedata"])


def test_collect_prediction_rows_sets_wkbalancedata_on_package_lane_balance_library() -> None:
    lane = _build_lane_assignment(
        "PKG_Nova X-25B_001",
        [
            _build_library(
                origrec="PKG_BALANCE",
                sample_type="平衡文库",
                sample_id="phix",
                contract_data=100.0,
                package_lane_number="PKG_001",
            ),
        ],
        package_id="PKG_001",
    )
    lane.metadata["wkbalancedata"] = 100.0
    setattr(lane.libraries[0], arrange_model6.BALANCE_LIBRARY_MARKER_COLUMN, True)

    result = _collect_prediction_rows([lane], {}, "unit_test")

    balance_row = result.loc[result["origrec"] == "PKG_BALANCE"].iloc[0]
    assert balance_row["wkbalancedata"] == 100.0


def test_build_detail_output_drops_internal_origrec_columns_when_raw_file_has_none(
    tmp_path: Path,
) -> None:
    df_raw = pd.DataFrame(
        [
            {
                "wkaidbid": "AID_001",
                "wkorigrec": "LIB_001",
                "aiarrangenumber": 0,
                "llaneid": "",
                "lrunid": "",
            }
        ]
    )

    output_path = tmp_path / "detail_no_internal_keys.csv"
    _build_detail_output(
        df_raw=df_raw,
        pred_df=_empty_prediction_df(),
        output_path=output_path,
        ai_schedulable_keys={"LIB_001"},
    )

    result = pd.read_csv(output_path)
    assert "origrec" not in result.columns
    assert "origrec_key" not in result.columns


def test_build_detail_output_copies_lane_level_fields_to_balance_library_row(
    tmp_path: Path,
) -> None:
    normal_lib = _build_library(origrec="LIB_001", contract_data=950.0, sample_id="NORMAL_001")
    normal_lib._origrec_key = "LIB_001"
    normal_lib._source_origrec_key = "LIB_001"

    lane = _build_lane_assignment("RS_Nova X-25B_001", [normal_lib])
    lane.metadata["selected_round_label"] = "ROUND_02"
    balance_lib = arrange_model6._create_balance_library_from_template(
        lane,
        {
            "wksampleid": "phix",
            "wkindexseq": "GGGGGGGGGG;ACCGAGATCT",
            "wkproductline": "S",
            "wktestno": "Novaseq X Plus-PE150",
            "wkdept": "天津科技服务实验室",
        },
        195.0,
    )
    pred_df = pd.DataFrame(
        [
            {
                "origrec": normal_lib.origrec,
                "origrec_key": arrange_model6._get_library_source_origrec_key(normal_lib),
                "detail_row_key": arrange_model6._get_library_detail_output_key(normal_lib),
                "runid": "RUN_001",
                "lane_id": lane.lane_id,
                "lsjnd": 2.345,
                "resolved_lsjfs": "25B",
                "resolved_lcxms": "3.6T-NEW",
                "resolved_seq_mode": "3.6T-NEW",
                "resolved_round_label": "ROUND_02",
                "resolved_index_check_rule": "P7P5",
                "wkbalancedata": None,
                "predicted_lorderdata": None,
                "lai_output": None,
                arrange_model6.BALANCE_LIBRARY_MARKER_COLUMN: False,
            },
            {
                "origrec": balance_lib.origrec,
                "origrec_key": arrange_model6._get_library_source_origrec_key(balance_lib),
                "detail_row_key": arrange_model6._get_library_detail_output_key(balance_lib),
                "runid": "RUN_001",
                "lane_id": lane.lane_id,
                "lsjnd": 2.345,
                "resolved_lsjfs": "25B",
                "resolved_lcxms": "3.6T-NEW",
                "resolved_seq_mode": "3.6T-NEW",
                "resolved_round_label": "ROUND_02",
                "resolved_index_check_rule": "P7P5",
                "wkbalancedata": None,
                "predicted_lorderdata": 195.0,
                "lai_output": None,
                arrange_model6.BALANCE_LIBRARY_MARKER_COLUMN: True,
            },
        ]
    )
    df_raw = pd.DataFrame(
        [
            {
                "wkaidbid": "AID_001",
                "wkorigrec": "LIB_001",
                "wksampleid": "NORMAL_001",
                "wkdataunit": "G",
                "wkuser": "USER_A",
                "wkdatadealbatch": "BATCH_01",
                "laneround": "",
                "lastlaneround": "ROUND_01",
                "task": "TASK_X",
                "lsjnd": "",
                "aiarrangenumber": 0,
                "llaneid": "",
                "lrunid": "",
            }
        ]
    )

    output_path = tmp_path / "detail_balance_lane_fields.csv"
    _build_detail_output(
        df_raw=df_raw,
        pred_df=pred_df,
        output_path=output_path,
        ai_schedulable_keys={"LIB_001"},
        detail_libraries=[normal_lib, balance_lib],
    )

    result = pd.read_csv(output_path)
    balance_row = result.loc[result["wksampleid"] == "phix"].iloc[0]
    assert balance_row["wkdataunit"] == "G"
    assert balance_row["wkuser"] == "USER_A"
    assert balance_row["wkdatadealbatch"] == "BATCH_01"
    assert balance_row["laneround"] == "ROUND_02"
    assert balance_row["lastlaneround"] == "ROUND_01"
    assert balance_row["task"] == "TASK_X"
    assert balance_row["lsjnd"] == pytest.approx(2.345, rel=0, abs=1e-6)


def test_validate_package_lane_rules_rejects_total_index_pairs_below_five() -> None:
    libs = [
        _build_library(
            origrec=f"PKG_IDX_{idx}",
            contract_data=250.0,
            package_lane_number="PKG_IDX",
            index_seq=f"AACCGGTTA{idx};TTGGCCAAT{idx}",
        )
        for idx in range(4)
    ]
    lane = _build_lane_assignment("LANE_IDX", libs, package_id="PKG_IDX")

    errors = _validate_package_lane_rules(lane)

    assert any("Index对数不足" in error for error in errors)


def test_validate_package_lane_rules_rejects_duplicate_indexes() -> None:
    libs = [
        _build_library(
            origrec="PKG_DUP_1",
            contract_data=200.0,
            package_lane_number="PKG_DUP",
            index_seq="AACCGGTTAA;TTGGCCAATT",
        ),
        _build_library(
            origrec="PKG_DUP_2",
            contract_data=200.0,
            package_lane_number="PKG_DUP",
            index_seq="AACCGGTTAA;TTGGCCAATT",
        ),
        _build_library(
            origrec="PKG_DUP_3",
            contract_data=200.0,
            package_lane_number="PKG_DUP",
            index_seq="CCGGAATTCC;GGAATTCCGG",
        ),
        _build_library(
            origrec="PKG_DUP_4",
            contract_data=200.0,
            package_lane_number="PKG_DUP",
            index_seq="TTAACCGGTT;AATTCCGGAA",
        ),
        _build_library(
            origrec="PKG_DUP_5",
            contract_data=200.0,
            package_lane_number="PKG_DUP",
            index_seq="CCAATTGGCC;GGTTCCAAGG",
        ),
    ]
    lane = _build_lane_assignment("LANE_DUP", libs, package_id="PKG_DUP")

    errors = _validate_package_lane_rules(lane)

    assert any("存在Index重复" in error for error in errors)


def test_validate_package_lane_rules_rejects_total_outside_1000g_tolerance() -> None:
    index_sequences = [
        "ACGTACGT;TGCATGCA",
        "AAAACCCC;GGGGTTTT",
        "TTTTAAAA;CCCCGGGG",
        "AGCTTCGA;TCGAAGCT",
        "CATGCATG;GTACGTAC",
    ]
    libs = [
        _build_library(
            origrec=f"PKG_RANGE_{idx}",
            contract_data=value,
            package_lane_number="PKG_RANGE",
            index_seq=index_seq,
        )
        for idx, (value, index_seq) in enumerate(
            zip([200.0, 200.0, 200.0, 200.0, 200.02], index_sequences),
            start=1,
        )
    ]
    lane = _build_lane_assignment("LANE_RANGE", libs, package_id="PKG_RANGE")

    errors = _validate_package_lane_rules(lane)

    assert any("1000G±0.01G" in error for error in errors)


def test_validate_package_lane_rules_accepts_exact_business_boundaries() -> None:
    index_sequences = [
        "ACGTACGT;TGCATGCA",
        "AAAACCCC;GGGGTTTT",
        "TTTTAAAA;CCCCGGGG",
        "AGCTTCGA;TCGAAGCT",
        "CATGCATG;GTACGTAC",
    ]
    libs = [
        _build_library(
            origrec=f"PKG_OK_{idx}",
            contract_data=value,
            package_lane_number="PKG_OK",
            index_seq=index_seq,
        )
        for idx, (value, index_seq) in enumerate(
            zip([200.0, 200.0, 200.0, 200.0, 200.01], index_sequences),
            start=1,
        )
    ]
    lane = _build_lane_assignment("LANE_OK", libs, package_id="PKG_OK")

    assert _validate_package_lane_rules(lane) == []


def test_multi_package_lane_split_success_puts_fragments_into_different_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = PackageLaneScheduler()
    monkeypatch.setattr(
        scheduler,
        "_calculate_pooling_coefficients",
        lambda libs, lane_id=None: {lib.origrec: 1.0 for lib in libs},
    )
    monkeypatch.setattr(scheduler, "_validate_index_conflicts", lambda _: (True, []))
    monkeypatch.setattr(
        scheduler.scheduling_config,
        "validate_lane_constraints",
        lambda libraries, machine_type: [],
    )

    libraries = [
        _build_library(
            origrec="PKG_MULTI_A",
            contract_data=1000.0,
            package_lane_number="PKG_A,PKG_B",
            index_seq="AACCGGTTAA;TTGGCCAATT,CCGGAATTCC;GGAATTCCGG,TTAACCGGTT;AATTCCGGAA,CCAATTGGCC;GGTTCCAAGG,TTAAGGCCAA;AACCGGTTCC",
        ),
        _build_library(
            origrec="PKG_MULTI_B",
            contract_data=1000.0,
            package_lane_number="PKG_A,PKG_B",
            index_seq="GGCCAATTGG;CCAATTGGCC,AATTGGCCAA;TTCCAAGGTT,CGTACGTACG;GCATGCATGC,ATGCATGCAT;TACGTACGTA,GGTTAACCGG;CCGGTTAACC",
        ),
    ]

    result = scheduler.schedule(libraries)

    assert not result.failed_packages
    assert result.total_lanes == 2
    assert len(result.remaining_libraries) == 0

    lane_assignments = _flatten_package_lanes(result)
    solution = SchedulingSolution(lane_assignments=lane_assignments)
    _validate_final_package_lanes(solution)
    _validate_no_split_for_package_lane_libraries(solution)

    family_to_lanes: Dict[str, set[str]] = {}
    for lane in lane_assignments:
        for lib in lane.libraries:
            family_id = getattr(lib, "_package_lane_multi_split_family_id", None)
            if family_id:
                family_to_lanes.setdefault(family_id, set()).add(lane.lane_id)

    assert family_to_lanes
    assert all(len(lane_ids) == 2 for lane_ids in family_to_lanes.values())


def test_multi_package_lane_split_rolls_back_when_any_target_lane_cannot_be_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = PackageLaneScheduler()
    monkeypatch.setattr(
        scheduler,
        "_calculate_pooling_coefficients",
        lambda libs, lane_id=None: {lib.origrec: 1.0 for lib in libs},
    )
    monkeypatch.setattr(scheduler, "_validate_index_conflicts", lambda _: (True, []))
    monkeypatch.setattr(
        scheduler.scheduling_config,
        "validate_lane_constraints",
        lambda libraries, machine_type: [],
    )

    original = _build_library(
        origrec="PKG_ROLLBACK",
        contract_data=1000.0,
        package_lane_number="PKG_R1,PKG_R2",
        index_seq="AACCGGTTAA;TTGGCCAATT,CCGGAATTCC;GGAATTCCGG,TTAACCGGTT;AATTCCGGAA,CCAATTGGCC;GGTTCCAAGG,TTAAGGCCAA;AACCGGTTCC",
    )

    result = scheduler.schedule([original])

    assert result.total_lanes == 0
    assert len(result.remaining_libraries) == 1
    restored = result.remaining_libraries[0]
    assert restored.origrec == "PKG_ROLLBACK"
    assert restored.package_lane_number == "PKG_R1,PKG_R2"
    assert restored.is_split is False
    assert restored.split_status == "rolled_back"
    assert "package_lane_multi_PKG_ROLLBACK" in result.failed_packages
    assert "不拆分且不排机" in result.failed_packages["package_lane_multi_PKG_ROLLBACK"]


def test_add_test_rule_keeps_ai_order_when_qpcr_outside_15pct_without_independent_rate() -> None:
    prediction_df = pd.DataFrame(
        [
            {
                "origrec": "ADD_AI_ONLY",
                "wkaddtestsremark": "加测",
                "lorderdata": 180.0,
                "wkqpcr": 13.0,
                "wklastqpcr": 10.0,
                "wklastoutput": 80.0,
                "wklastorderdata": 200.0,
                "wkcontractdata": 120.0,
            }
        ]
    )

    result = _apply_add_test_output_rate_rule_to_prediction_df(prediction_df)

    assert float(result.loc[0, "lorderdata"]) == 180.0


def test_add_test_rule_uses_historical_outrate_and_clamps_minimum_to_point_three() -> None:
    prediction_df = pd.DataFrame(
        [
            {
                "origrec": "ADD_HISTORY",
                "wkaddtestsremark": "加测",
                "lorderdata": 200.0,
                "wkqpcr": 9.5,
                "wklastqpcr": 10.0,
                "wklastoutput": 10.0,
                "wklastorderdata": 100.0,
                "wkcontractdata": 120.0,
            }
        ]
    )

    result = _apply_add_test_output_rate_rule_to_prediction_df(prediction_df)

    assert float(result.loc[0, "lorderdata"]) == 400.0


def test_add_test_rule_independent_output_rate_applies_even_when_qpcr_outside_range() -> None:
    prediction_df = pd.DataFrame(
        [
            {
                "origrec": "ADD_RATE",
                "wkaddtestsremark": "加测",
                "lorderdata": 180.0,
                "wkqpcr": 13.0,
                "wklastqpcr": 10.0,
                "wkoutputrate": 20,
                "wkcontractdata": 120.0,
            }
        ]
    )

    result = _apply_add_test_output_rate_rule_to_prediction_df(prediction_df)

    assert float(result.loc[0, "lorderdata"]) == 400.0


def test_collect_prediction_rows_carries_previous_round_fields_and_loading_concentration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "arrange_library.arrange_library_model6._resolve_lane_output_rule_fields",
        lambda **kwargs: ("25B", "3.6T-NEW", "unit_test_rule"),
    )
    monkeypatch.setattr(
        "arrange_library.arrange_library_model6._resolve_lane_index_rule_display",
        lambda libraries, loading_method: "P7P5",
    )

    lib = _build_library(
        origrec="HISTORY_ROW",
        sample_type="普通文库",
        contract_data=120.0,
        sub_project_name="肿瘤医学委托项目",
        qpcr_molar=9.8,
    )
    lib._last_qpcr_raw = 10.0
    lib._last_order_data_raw = 200.0
    lib._last_output_raw = 80.0
    lane = _build_lane_assignment("LANE_HISTORY", [lib])

    rows = _collect_prediction_rows([lane], {"HISTORY_ROW": 90.0}, "TEST")

    assert float(rows.loc[0, "lsjnd"]) == 2.3
    assert float(rows.loc[0, "wklastqpcr"]) == 10.0
    assert float(rows.loc[0, "wklastorderdata"]) == 200.0
    assert float(rows.loc[0, "wklastoutput"]) == 80.0
    assert float(rows.loc[0, "wklastoutrate"]) == 0.4


@pytest.mark.parametrize(
    ("libraries", "expected_concentration", "expected_reason"),
    [
        (
            [
                _build_library(
                    origrec="MED_1",
                    sample_type="普通文库",
                    contract_data=60.0,
                    sub_project_name="肿瘤医学委托项目",
                ),
                _build_library(
                    origrec="MED_2",
                    sample_type="普通文库",
                    contract_data=45.0,
                    sub_project_name="医检所委托项目",
                ),
            ],
            2.3,
            "medical_commission_over_100g_2_3",
        ),
        (
            [
                _build_library(
                    origrec="ATAC_1",
                    sample_type="客户-10X ATAC文库",
                    contract_data=50.0,
                    seq_scheme="151+10+24+151",
                ),
                _build_library(
                    origrec="ATAC_2",
                    sample_type="普通文库",
                    contract_data=50.0,
                    seq_scheme="151+10+24+151",
                ),
            ],
            2.0,
            "10_plus_24_atac_2_0",
        ),
        (
            [
                _build_library(origrec="GPA_1", sample_type="10X转录组-5'文库"),
                _build_library(origrec="GPA_2", sample_type="10X转录组V(D)J-BCR文库"),
            ],
            1.78,
            "special_10x_combo_group_a_non_customer_1_78",
        ),
        (
            [
                _build_library(origrec="GPA_C1", sample_type="10X转录组-5'文库"),
                _build_library(origrec="GPA_C2", sample_type="客户-10X VDJ文库"),
            ],
            2.5,
            "special_10x_combo_group_a_customer_2_5",
        ),
        (
            [
                _build_library(origrec="GPB_1", sample_type="10X Visium空间转录组文库"),
                _build_library(origrec="GPB_2", sample_type="10X转录组-3'文库"),
            ],
            1.78,
            "special_10x_combo_group_b_non_customer_1_78",
        ),
        (
            [
                _build_library(origrec="GPB_C1", sample_type="10X转录组-3'文库"),
                _build_library(origrec="GPB_C2", sample_type="客户-10X 3 单细胞转录组文库"),
            ],
            2.5,
            "special_10x_combo_group_b_customer_2_5",
        ),
    ],
)
def test_resolve_lane_loading_concentration_business_rules(
    libraries: Iterable[EnhancedLibraryInfo],
    expected_concentration: float,
    expected_reason: str,
) -> None:
    concentration, reason = _resolve_lane_loading_concentration(list(libraries))

    assert concentration == expected_concentration
    assert reason == expected_reason


def test_scattered_mix_sort_prefers_clinical_and_sj_then_yc_then_other_delete_date() -> None:
    scheduler = GreedyLaneScheduler()
    libraries = [
        _build_library(
            origrec="OTHER_FAR",
            data_type="其他",
            contract_data=30.0,
            delete_date=5.0,
            jjbj="是",
            board_number="B5",
        ),
        _build_library(
            origrec="YC",
            data_type="YC",
            contract_data=30.0,
            delete_date=2.0,
            jjbj="是",
            board_number="B3",
        ),
        _build_library(
            origrec="SJ",
            data_type="其他",
            contract_data=30.0,
            sub_project_name="单细胞SJ客户",
            delete_date=9.0,
            jjbj="是",
            board_number="B2",
        ),
        _build_library(
            origrec="CLINICAL",
            data_type="临检",
            contract_data=30.0,
            delete_date=8.0,
            jjbj="是",
            board_number="B1",
        ),
        _build_library(
            origrec="OTHER_NEAR",
            data_type="其他",
            contract_data=30.0,
            delete_date=1.0,
            jjbj="是",
            board_number="B4",
        ),
    ]

    ordered = scheduler._sort_remaining_for_scattered_mix_lane(libraries)
    ordered_ids = [lib.origrec for lib in ordered]

    assert set(ordered_ids[:2]) == {"CLINICAL", "SJ"}
    assert ordered_ids[2] == "YC"
    assert ordered_ids[-2:] == ["OTHER_NEAR", "OTHER_FAR"]


def test_schedule_machine_group_keeps_same_priority_libraries_in_first_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = GreedyLaneScheduler()
    monkeypatch.setattr(
        scheduler,
        "_get_scheduling_lane_capacity_range",
        lambda libraries, machine_type, metadata=None: _lane_rule(),
    )
    monkeypatch.setattr(
        scheduler,
        "_sample_target_capacity",
        lambda lane_rule_selection: 100.0,
    )
    monkeypatch.setattr(
        scheduler,
        "_validate_completed_lane",
        lambda lane: (True, []),
    )
    monkeypatch.setattr(
        scheduler,
        "_can_add_to_lane",
        lambda lane, lib: lane.total_data_gb + lib.get_data_amount_gb() <= 100.0,
    )

    libraries = [
        _build_library(
            origrec="CLINICAL_L1",
            data_type="临检",
            contract_data=60.0,
            board_number="A1",
        ),
        _build_library(
            origrec="SJ_L1",
            data_type="其他",
            contract_data=40.0,
            sub_project_name="SJ urgent project",
            board_number="A2",
        ),
        _build_library(
            origrec="YC_L2",
            data_type="YC",
            contract_data=60.0,
            board_number="A3",
        ),
        _build_library(
            origrec="OTHER_L2",
            data_type="其他",
            contract_data=40.0,
            delete_date=1.0,
            board_number="A4",
        ),
        _build_library(
            origrec="OTHER_L3",
            data_type="其他",
            contract_data=40.0,
            delete_date=2.0,
            board_number="A5",
        ),
    ]

    lanes, remaining = scheduler._schedule_machine_group(libraries, "Nova X-25B")

    assert [lib.origrec for lib in remaining] == ["OTHER_L3"]
    assert len(lanes) >= 2
    first_lane_ids = {lib.origrec for lib in lanes[0].libraries}
    assert first_lane_ids == {"CLINICAL_L1", "SJ_L1"}
    assert all(
        lib.origrec not in first_lane_ids
        for lane in lanes[1:]
        for lib in lane.libraries
    )


def test_schedule_machine_group_priority_seed_stops_at_min_and_preserves_next_priority_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = GreedyLaneScheduler()
    monkeypatch.setattr(
        scheduler,
        "_get_scheduling_lane_capacity_range",
        lambda libraries, machine_type, metadata=None: _lane_rule(min_gb=100.0, max_gb=200.0),
    )
    monkeypatch.setattr(
        scheduler,
        "_sample_target_capacity",
        lambda lane_rule_selection: 160.0,
    )
    monkeypatch.setattr(
        scheduler,
        "_validate_completed_lane",
        lambda lane: (True, []),
    )
    monkeypatch.setattr(
        scheduler,
        "_can_add_to_lane",
        lambda lane, lib: lane.total_data_gb + lib.get_data_amount_gb() <= 200.0,
    )

    libraries = [
        _build_library(
            origrec="CLINICAL_L1",
            data_type="临检",
            contract_data=60.0,
            board_number="A1",
        ),
        _build_library(
            origrec="SJ_L1",
            data_type="其他",
            contract_data=60.0,
            sub_project_name="SJ urgent project",
            board_number="A2",
        ),
        _build_library(
            origrec="YC_L2",
            data_type="YC",
            contract_data=60.0,
            board_number="A3",
        ),
        _build_library(
            origrec="OTHER_L2",
            data_type="其他",
            contract_data=40.0,
            delete_date=1.0,
            board_number="A4",
        ),
        _build_library(
            origrec="OTHER_L3",
            data_type="其他",
            contract_data=40.0,
            delete_date=2.0,
            board_number="A5",
        ),
    ]

    lanes, remaining = scheduler._schedule_machine_group(libraries, "Nova X-25B")

    assert len(lanes) == 2
    assert {lib.origrec for lib in lanes[0].libraries} == {"CLINICAL_L1", "SJ_L1"}
    assert {lib.origrec for lib in lanes[1].libraries} == {"YC_L2", "OTHER_L2"}
    assert [lib.origrec for lib in remaining] == ["OTHER_L3"]


def test_schedule_machine_group_continues_with_normal_lanes_when_priority_tail_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = GreedyLaneScheduler()
    monkeypatch.setattr(
        scheduler,
        "_get_scheduling_lane_capacity_range",
        lambda libraries, machine_type, metadata=None: _lane_rule(min_gb=100.0, max_gb=200.0),
    )
    monkeypatch.setattr(
        scheduler,
        "_sample_target_capacity",
        lambda lane_rule_selection: 160.0,
    )
    monkeypatch.setattr(
        scheduler,
        "_validate_completed_lane",
        lambda lane: (True, []),
    )

    def fake_can_add_to_lane(lane, lib):
        current_ids = {item.origrec for item in lane.libraries}
        if not current_ids:
            return True
        if current_ids == {"CLINICAL_BLOCKED"}:
            return False
        return lane.total_data_gb + lib.get_data_amount_gb() <= 200.0

    monkeypatch.setattr(scheduler, "_can_add_to_lane", fake_can_add_to_lane)

    libraries = [
        _build_library(
            origrec="CLINICAL_BLOCKED",
            data_type="临检",
            contract_data=60.0,
            board_number="B1",
        ),
        _build_library(
            origrec="OTHER_L1",
            data_type="其他",
            contract_data=60.0,
            delete_date=1.0,
            board_number="B2",
        ),
        _build_library(
            origrec="OTHER_L2",
            data_type="其他",
            contract_data=40.0,
            delete_date=2.0,
            board_number="B3",
        ),
    ]

    lanes, remaining = scheduler._schedule_machine_group(libraries, "Nova X-25B")

    assert len(lanes) == 1
    assert {lib.origrec for lib in lanes[0].libraries} == {"OTHER_L1", "OTHER_L2"}
    assert [lib.origrec for lib in remaining] == ["CLINICAL_BLOCKED"]


def test_select_backbone_libraries_prefers_mid_sized_candidates_before_priority_large_library() -> None:
    scheduler = GreedyLaneScheduler()
    scheduler.config.small_library_threshold_gb = 20.0

    libraries = [
        _build_library(
            origrec="SJ_BACKBONE",
            data_type="SJ",
            contract_data=80.0,
            sub_project_name="SJ priority project",
        ),
        _build_library(origrec="OTHER_31", contract_data=31.0, data_type="其他"),
        _build_library(origrec="OTHER_32", contract_data=32.0, data_type="其他"),
        _build_library(origrec="OTHER_33", contract_data=33.0, data_type="其他"),
    ]

    backbone, remaining = scheduler._select_backbone_libraries(libraries, target_data_gb=60.0)

    assert [lib.origrec for lib in backbone] == ["OTHER_31", "OTHER_32"]
    assert {lib.origrec for lib in remaining} == {"SJ_BACKBONE", "OTHER_33"}


def test_schedule_backbone_with_small_lanes_keeps_seed_priority_group_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = GreedyLaneScheduler()
    scheduler.config.lane_capacity_gb = 100.0

    monkeypatch.setattr(
        scheduler,
        "_resolve_lane_capacity_limits",
        lambda libraries, machine_type, lane=None, lane_id=None, lane_metadata=None, metadata=None: (90.0, 90.0),
    )
    monkeypatch.setattr(scheduler.index_validator, "validate_lane_quick", lambda libraries: True)
    monkeypatch.setattr(scheduler, "_check_peak_size_compatible", lambda libraries: True)
    monkeypatch.setattr(scheduler, "_check_customer_ratio_near_limit", lambda lane, lib, threshold=0.50: True)
    monkeypatch.setattr(scheduler, "_check_customer_ratio_compatible_by_data", lambda libraries: True)
    monkeypatch.setattr(scheduler, "_check_base_imbalance_ratio_near_limit", lambda lane, lib, threshold: True)
    monkeypatch.setattr(scheduler, "_check_base_imbalance_compatible", lambda libraries: True)
    monkeypatch.setattr(scheduler, "_check_10bp_index_ratio_near_limit", lambda lane, lib, threshold: True)
    monkeypatch.setattr(scheduler, "_check_10bp_index_ratio_compatible", lambda libraries: True)
    monkeypatch.setattr(scheduler, "_validate_backbone_lane", lambda lane: True)

    backbone_libs = [
        _build_library(
            origrec="YC_BACKBONE",
            data_type="YC",
            sample_id="FKDL0001-1a",
            contract_data=80.0,
        )
    ]
    small_libs = [
        _build_library(
            origrec="SJ_SMALL",
            data_type="SJ",
            sample_id="FKDL0002-1a",
            sub_project_name="SJ project",
            contract_data=10.0,
        ),
        _build_library(
            origrec="YC_SMALL",
            data_type="YC",
            sample_id="FKDL0003-1a",
            contract_data=10.0,
        ),
    ]

    lanes, remaining_small = scheduler._schedule_backbone_with_small_lanes(
        backbone_libs,
        small_libs,
        "Nova X-25B",
    )

    assert len(lanes) == 1
    lane_ids = [lib.origrec for lib in lanes[0].libraries]
    assert lane_ids == ["YC_BACKBONE", "YC_SMALL"]
    assert [lib.origrec for lib in remaining_small] == ["SJ_SMALL"]


def test_sort_remaining_for_lane_seed_keeps_scattered_mix_order_for_other_seed() -> None:
    scheduler = GreedyLaneScheduler()

    seed_lib = _build_library(origrec="OTHER_BACKBONE", data_type="其他", contract_data=80.0, delete_date=5.0)
    yc_small = _build_library(
        origrec="YC_SMALL",
        data_type="YC",
        sample_id="FKDL0100-1a",
        contract_data=10.0,
    )
    other_small = _build_library(
        origrec="OTHER_SMALL",
        data_type="其他",
        contract_data=10.0,
        delete_date=1.0,
    )

    ordered = scheduler._sort_remaining_for_lane_seed([other_small, yc_small], seed_lib)

    assert [lib.origrec for lib in ordered] == ["YC_SMALL", "OTHER_SMALL"]


def test_enhanced_library_info_special_splits_default_is_none() -> None:
    lib = EnhancedLibraryInfo(
        origrec="MODEL_SPECIAL_SPLIT_DEFAULT",
        sample_id="SID_MODEL_SPECIAL_SPLIT_DEFAULT",
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
        contract_data_raw=10.0,
        test_code=1001,
        test_no="NovaSeq X Plus-PE150",
        sub_project_name="常规项目",
        create_date="2026-04-08 09:00:00",
        delivery_date="2026-04-10 09:00:00",
        lab_type="普通文库",
        data_volume_type="标准",
        board_number="BOARD001",
    )

    assert lib.data_flag is None
    assert lib.special_splits is None


def test_load_libraries_from_csv_uses_arrange_library_model_class(tmp_path: Path) -> None:
    csv_path = tmp_path / "loader_minimal.csv"
    pd.DataFrame(
        [
            {
                "wkorigrec": "LOADER_001",
                "wksampleid": "SID_LOADER_001",
                "wksampletype": "普通文库",
                "wkdatatype": "YC",
                "wkindexseq": "AACCGGTTAA;TTGGCCAATT",
                "wkaddtestsremark": "",
                "wkproductline": "S",
                "wkpeaksize": 350,
                "wkeqtype": "Nova X-25B",
                "wkcontractdata": 10.0,
                "wktestno": "NovaSeq X Plus-PE150",
                "wksubprojectname": "常规项目",
                "wkcreatedate": "2026-04-08 09:00:00",
                "wkdeliverydate": "2026-04-10 09:00:00",
                "wkdataunit": "标准",
                "wkboardnumber": "BOARD001",
                "wkspecialsplits": "",
            }
        ]
    ).to_csv(csv_path, index=False)

    libs = load_libraries_from_csv(
        str(csv_path),
        enable_remark_recognition=False,
    )

    assert len(libs) == 1
    assert type(libs[0]).__module__ == "arrange_library.models.library_info"
    assert libs[0].data_type == "YC"
    assert libs[0].data_flag is None
    assert libs[0].special_splits is None


def test_schedule_stops_after_eight_consecutive_rounds_without_new_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = GreedyLaneScheduler()
    scheduler._strategy_plan = types.SimpleNamespace(
        enable_dedicated_imbalance_lane=False,
        enable_non_10bp_dedicated_lane=False,
        enable_backbone_reservation=False,
    )

    monkeypatch.setattr(scheduler, "_run_batch_analysis", lambda libraries: None)
    monkeypatch.setattr(scheduler, "_sort_libraries", lambda libraries: list(libraries))
    monkeypatch.setattr(
        scheduler,
        "_group_by_machine_type",
        lambda libraries: {"Nova X-25B": list(libraries)},
    )

    call_counter = {"count": 0}

    def fake_schedule_machine_group(libraries, machine_type):
        call_counter["count"] += 1
        return [], list(libraries)

    monkeypatch.setattr(scheduler, "_schedule_machine_group", fake_schedule_machine_group)

    libraries = [
        _build_library(
            origrec=f"ROUND_STOP_{idx}",
            contract_data=10.0,
            board_number=f"RS{idx:02d}",
        )
        for idx in range(10)
    ]

    solution = scheduler.schedule(libraries, libraries_already_split=True)

    assert call_counter["count"] == 11
    assert len(solution.lane_assignments) == 0
    assert len(solution.unassigned_libraries) == len(libraries)


def test_collect_donations_for_pool_prefers_smallest_removable_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_small = _build_library(origrec="OTHER_SMALL", contract_data=6.0, data_type="其他")
    other_large = _build_library(origrec="OTHER_LARGE", contract_data=10.0, data_type="其他")
    yc_small = _build_library(origrec="YC_SMALL", contract_data=5.0, data_type="YC", sample_id="FKDL0010-1a")
    lane = _build_lane_assignment("LANE_DONOR", [other_small, other_large, yc_small])

    monkeypatch.setattr(
        arrange_model6,
        "_resolve_lane_capacity_limits",
        lambda libraries, machine_type, lane_id=None, lane_metadata=None: (0.0, 100.0),
    )
    monkeypatch.setattr(
        arrange_model6,
        "_validate_lane_with_latest_index",
        lambda validator, libraries, lane_id, machine_type, metadata=None: types.SimpleNamespace(is_valid=True),
    )

    donations = _collect_donations_for_pool(
        lanes=[lane],
        validator=LaneValidator(strict_mode=True),
        machine_type=MachineType.NOVA_X_25B,
        target_data=5.0,
    )

    assert donations
    assert donations[0][1].origrec == "YC_SMALL"


def test_try_multi_lib_swap_rebalance_uses_scattered_mix_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        arrange_model6,
        "_resolve_lane_capacity_limits",
        lambda libraries, machine_type: (1.0, 100.0),
    )

    def fake_attempt(pool, validator, machine_type, lane_id_prefix, **kwargs):
        captured_calls.append(
            {
                "lane_id_prefix": lane_id_prefix,
                "prioritize_scattered_mix": kwargs.get("prioritize_scattered_mix"),
                "origrecs": [lib.origrec for lib in pool],
            }
        )
        return None, []

    monkeypatch.setattr(
        arrange_model6,
        "_attempt_build_lane_from_pool",
        fake_attempt,
    )

    solution = SchedulingSolution(
        lane_assignments=[],
        unassigned_libraries=[
            _build_library(origrec="YC_UNASSIGNED", data_type="YC", sample_id="FKDL1000-1a", contract_data=10.0),
            _build_library(origrec="OTHER_UNASSIGNED", data_type="其他", contract_data=10.0),
        ],
    )

    result = try_multi_lib_swap_rebalance(solution, LaneValidator(strict_mode=True), max_new_lanes=1)

    assert result["new_lanes"] == 0
    assert captured_calls
    assert captured_calls[0]["lane_id_prefix"] == "RB"
    assert captured_calls[0]["prioritize_scattered_mix"] is True


def test_try_multi_lib_swap_rebalance_falls_back_to_non_clinical_pool_without_scattering_priority_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        arrange_model6,
        "_resolve_lane_capacity_limits",
        lambda libraries, machine_type: (1.0, 100.0),
    )

    sj_lib = _build_library(
        origrec="SJ_UNASSIGNED",
        data_type="SJ",
        contract_data=10.0,
        sub_project_name="SJ urgent project",
    )
    yc_lib = _build_library(
        origrec="YC_UNASSIGNED",
        data_type="YC",
        contract_data=10.0,
        sample_id="FKDL1001-1a",
    )
    other_lib = _build_library(origrec="OTHER_UNASSIGNED", data_type="其他", contract_data=10.0)
    captured_calls: list[dict[str, object]] = []

    def fake_attempt(pool, validator, machine_type, lane_id_prefix, **kwargs):
        call = {
            "lane_id_prefix": lane_id_prefix,
            "prioritize_scattered_mix": kwargs.get("prioritize_scattered_mix"),
            "origrecs": [lib.origrec for lib in pool],
        }
        captured_calls.append(call)
        if len(captured_calls) < 3:
            return None, []
        lane = _build_lane_assignment("RB_TEST", [yc_lib, other_lib])
        return lane, [yc_lib, other_lib]

    monkeypatch.setattr(
        arrange_model6,
        "_attempt_build_lane_from_pool",
        fake_attempt,
    )

    solution = SchedulingSolution(
        lane_assignments=[],
        unassigned_libraries=[sj_lib, yc_lib, other_lib],
    )

    result = try_multi_lib_swap_rebalance(solution, LaneValidator(strict_mode=True), max_new_lanes=1)

    assert result["new_lanes"] == 1
    assert len(solution.lane_assignments) == 1
    assert {lib.origrec for lib in solution.lane_assignments[0].libraries} == {
        "YC_UNASSIGNED",
        "OTHER_UNASSIGNED",
    }
    assert [lib.origrec for lib in solution.unassigned_libraries] == ["SJ_UNASSIGNED"]
    assert captured_calls[0]["origrecs"] == ["SJ_UNASSIGNED", "YC_UNASSIGNED", "OTHER_UNASSIGNED"]
    assert captured_calls[0]["prioritize_scattered_mix"] is True
    assert captured_calls[1]["origrecs"] == ["YC_UNASSIGNED", "OTHER_UNASSIGNED"]
    assert captured_calls[1]["prioritize_scattered_mix"] is True
    assert captured_calls[2]["origrecs"] == ["YC_UNASSIGNED", "OTHER_UNASSIGNED"]
    assert captured_calls[2]["prioritize_scattered_mix"] is False


def test_rescue_remaining_lanes_by_layered_regroup_search_prefers_priority_then_mixed_then_normal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        arrange_model6,
        "_resolve_lane_capacity_limits",
        lambda libraries, machine_type, lane_id=None, lane_metadata=None: (100.0, 120.0),
    )

    clinical_a = _build_library(
        origrec="CLIN_A",
        sample_type="临检转录组",
        data_type="临检",
        contract_data=60.0,
    )
    clinical_b = _build_library(
        origrec="CLIN_B",
        sample_type="临检转录组",
        data_type="临检",
        contract_data=60.0,
    )
    yc_a = _build_library(
        origrec="YC_A",
        sample_type="YC文库",
        data_type="YC",
        contract_data=45.0,
        sample_id="FKDL_YC_A",
    )
    yc_b = _build_library(
        origrec="YC_B",
        sample_type="YC文库",
        data_type="YC",
        contract_data=45.0,
        sample_id="FKDL_YC_B",
    )
    other_a = _build_library(
        origrec="OTHER_A",
        sample_type="VIP-真核普通转录组文库",
        data_type="其他",
        contract_data=55.0,
    )
    other_b = _build_library(
        origrec="OTHER_B",
        sample_type="VIP-真核普通转录组文库",
        data_type="其他",
        contract_data=55.0,
    )
    split_other = _build_library(
        origrec="SPLIT_OTHER",
        sample_type="VIP-真核普通转录组文库",
        data_type="其他",
        contract_data=80.0,
    )
    split_other.is_split = True

    captured_calls: list[tuple[str, list[str]]] = []

    def fake_attempt(pool, validator, machine_type, lane_id_prefix, lane_serial=None, **kwargs):
        origrecs = [lib.origrec for lib in pool]
        captured_calls.append((lane_id_prefix, origrecs))
        if lane_id_prefix == "PG" and set(origrecs) == {"CLIN_A", "CLIN_B"}:
            lane = _build_lane_assignment("PG_Nova X-25B_001", [clinical_a, clinical_b])
            return lane, [clinical_a, clinical_b]
        if lane_id_prefix == "RM" and set(origrecs) == {"YC_A", "YC_B", "OTHER_A", "OTHER_B"}:
            lane = _build_lane_assignment("RM_Nova X-25B_001", [yc_a, yc_b, other_a])
            return lane, [yc_a, yc_b, other_a]
        if lane_id_prefix == "OG" and set(origrecs) == {"OTHER_B"}:
            return None, []
        return None, []

    monkeypatch.setattr(
        arrange_model6,
        "_attempt_build_rescue_lane_from_pool",
        fake_attempt,
    )

    solution = SchedulingSolution(
        lane_assignments=[],
        unassigned_libraries=[
            clinical_a,
            clinical_b,
            yc_a,
            yc_b,
            other_a,
            other_b,
            split_other,
        ],
    )

    stats = _rescue_remaining_lanes_by_layered_regroup_search(
        solution=solution,
        validator=LaneValidator(strict_mode=True),
        max_priority_cluster_lanes_per_machine=2,
        max_mixed_rescue_lanes_per_machine=2,
        max_normal_cluster_lanes_per_machine=2,
    )

    assert stats["new_lanes"] == 2
    assert stats["priority_cluster_lanes"] == 1
    assert stats["mixed_rescue_lanes"] == 1
    assert stats["normal_cluster_lanes"] == 0
    assert stats["skipped_split_libraries"] == 1
    assert [lane.lane_id for lane in solution.lane_assignments] == [
        "PG_Nova X-25B_001",
        "RM_Nova X-25B_001",
    ]
    assert [lib.origrec for lib in solution.unassigned_libraries] == [
        "OTHER_B",
        "SPLIT_OTHER",
    ]
    assert captured_calls[0] == ("PG", ["CLIN_A", "CLIN_B"])
    assert captured_calls[1] == ("RM", ["YC_A", "YC_B", "OTHER_A", "OTHER_B"])


def test_mixed_customer_lane_prefers_high_priority_customer_under_hard_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = GreedyLaneScheduler()
    scheduler.config.lane_capacity_gb = 100.0
    monkeypatch.setattr(
        scheduler,
        "_resolve_lane_capacity_limits",
        lambda libraries, machine_type, lane=None, metadata=None: (50.0, 100.0),
    )
    monkeypatch.setattr(
        scheduler.index_validator,
        "validate_lane_quick",
        lambda libraries: True,
    )
    monkeypatch.setattr(scheduler, "_check_peak_size_compatible", lambda libraries: True)
    monkeypatch.setattr(scheduler, "_check_base_imbalance_ratio_near_limit", lambda lane, lib, threshold: True)
    monkeypatch.setattr(scheduler, "_check_base_imbalance_compatible", lambda libraries: True)
    monkeypatch.setattr(scheduler, "_check_10bp_index_ratio_near_limit", lambda lane, lib, threshold: True)
    monkeypatch.setattr(scheduler, "_check_10bp_index_ratio_compatible", lambda libraries: True)
    monkeypatch.setattr(scheduler, "_check_customer_ratio_near_limit", lambda lane, lib, threshold=0.50: True)

    internal = _build_library(
        origrec="INTERNAL",
        customer_library="否",
        sample_id="FDHE_INTERNAL",
        contract_data=70.0,
    )
    other_customer = _build_library(
        origrec="OTHER_CUSTOMER",
        customer_library="是",
        sample_id="FKDL_OTHER",
        data_type="其他",
        contract_data=20.0,
        sub_project_name="普通客户项目",
    )
    clinical_customer = _build_library(
        origrec="CLINICAL_CUSTOMER",
        customer_library="是",
        sample_id="FKDL_CLIN",
        data_type="临检",
        contract_data=20.0,
        sub_project_name="临检客户项目",
    )

    lanes = scheduler._form_mixed_customer_lanes(
        high_customer_libs=[other_customer, clinical_customer],
        internal_libs=[internal],
        machine_type_enum=MachineType.NOVA_X_25B,
        machine_type="Nova X-25B",
        used_libs=set(),
    )

    assert len(lanes) == 1
    lane_member_ids = [lib.origrec for lib in lanes[0].libraries]
    assert lane_member_ids == ["INTERNAL", "CLINICAL_CUSTOMER"]


def test_filter_valid_lanes_keeps_1000g_package_lane_in_final_merge() -> None:
    validator = LaneValidator(strict_mode=True)
    libraries = [
        _build_library(
            origrec=f"PKG_LIB_{idx}",
            contract_data=200.0,
            index_seq=index_seq,
            package_lane_number="PKG_MERGE_OK",
        )
        for idx, index_seq in enumerate(
            [
                "AACCGGTTAA;TTGGCCAATT",
                "AAAACCCC;GGGGTTTT",
                "TTTTAAAA;CCCCGGGG",
                "AGCTTCGA;TCGAAGCT",
                "CATGCATG;GTACGTAC",
            ],
            start=1,
        )
    ]
    lane = _build_lane_assignment("LANE_PKG_OK", libraries, package_id="PKG_MERGE_OK")
    lane.total_data_gb = 1000.0
    lane.metadata["is_package_lane"] = True

    valid_lanes, failed_lanes = _filter_valid_lanes([lane], validator)

    assert valid_lanes == [lane]
    assert failed_lanes == []


def test_rescue_failed_lanes_recovers_libraries_even_if_failed_lane_already_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_lib = _build_library(
        origrec="FAILED_LIB",
        contract_data=120.0,
        sample_id="FAILED_SAMPLE",
    )
    failed_lane = _build_lane_assignment("FAILED_LANE", [failed_lib])

    passed_lib = _build_library(
        origrec="PASSED_LIB",
        contract_data=120.0,
        sample_id="PASSED_SAMPLE",
    )
    passed_lane = _build_lane_assignment("PASSED_LANE", [passed_lib])

    existing_unassigned = _build_library(
        origrec="UNASSIGNED_LIB",
        contract_data=80.0,
        sample_id="UNASSIGNED_SAMPLE",
    )
    solution = SchedulingSolution(
        lane_assignments=[passed_lane],
        unassigned_libraries=[existing_unassigned],
    )

    def _fake_drain_rescue_lanes_for_match(
        primary_pool,
        secondary_pool,
        validator,
        machine_type,
        lane_prefix,
        serial_start,
        match_fn=None,
        extra_metadata=None,
    ):
        return [], list(primary_pool), list(secondary_pool), serial_start

    monkeypatch.setattr(
        arrange_model6,
        "_drain_rescue_lanes_for_match",
        _fake_drain_rescue_lanes_for_match,
    )
    monkeypatch.setattr(
        arrange_model6,
        "_attempt_build_lane_from_prioritized_pool",
        lambda **kwargs: (None, []),
    )

    stats = _rescue_failed_lanes_by_57_rules(
        failed_lanes=[failed_lane],
        solution=solution,
        validator=object(),
    )

    assert solution.lane_assignments == [passed_lane]
    assert [lib.origrec for lib in solution.unassigned_libraries] == [
        "FAILED_LIB",
        "UNASSIGNED_LIB",
    ]
    assert stats == {
        "failed_lanes": 1,
        "rescued_lanes": 0,
        "recovered_libraries": 1,
        "remaining_unassigned": 2,
    }
