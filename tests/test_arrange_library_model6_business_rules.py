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
    _apply_balance_reservation_to_capacity_selection,
    _attempt_build_lane_from_pool,
    _attempt_build_lane_from_prioritized_pool,
    _attempt_build_rescue_lane_from_pool,
    _build_detail_output,
    _collect_prediction_rows,
    _consume_mode_1_1_priority_from_unassigned,
    _enforce_mode_1_1_priority_cap_per_lane,
    _enforce_global_priority_hard_constraint,
    _ensure_unique_lane_ids,
    _filter_valid_lanes,
    _reset_auto_lane_serial_counters,
    _reserve_auto_lane_serial,
    _rescue_remaining_lanes_by_layered_regroup_search,
    _rescue_failed_lanes_by_57_rules,
    _resolve_lane_loading_concentration,
    _try_increase_lane_count,
    try_targeted_imbalance_upgrade,
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


def test_attempt_build_rescue_lane_from_pool_skips_impossible_small_pool(monkeypatch) -> None:
    pool = [
        _build_library(origrec="R1", contract_data=30.0),
        _build_library(origrec="R2", contract_data=30.0),
        _build_library(origrec="R3", contract_data=30.0),
    ]

    attempted = {"count": 0}

    def _fake_attempt_build_lane_from_pool(**kwargs):
        attempted["count"] += 1
        return None, []

    monkeypatch.setattr(arrange_model6, "_resolve_lane_capacity_limits", lambda libraries, machine_type: (100.0, 120.0))
    monkeypatch.setattr(arrange_model6, "_attempt_build_lane_from_pool", _fake_attempt_build_lane_from_pool)

    lane, used = _attempt_build_rescue_lane_from_pool(
        pool=pool,
        validator=LaneValidator(strict_mode=True),
        machine_type=MachineType.NOVA_X_25B,
        lane_id_prefix="EX",
    )

    assert lane is None
    assert used == []
    assert attempted["count"] == 0


def test_try_increase_lane_count_stops_retrying_same_pool_after_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"count": 0}

    monkeypatch.setattr(
        arrange_model6,
        "_resolve_lane_capacity_limits",
        lambda libraries, machine_type, lane_id="", lane_metadata=None: (1.0, 100.0),
    )

    def _fake_attempt(**kwargs):
        attempts["count"] += 1
        return None, []

    monkeypatch.setattr(arrange_model6, "_attempt_build_rescue_lane_from_pool", _fake_attempt)

    solution = SchedulingSolution(
        lane_assignments=[],
        unassigned_libraries=[
            _build_library(origrec="EX_FAIL_1", contract_data=10.0),
            _build_library(origrec="EX_FAIL_2", contract_data=10.0),
        ],
    )

    added = _try_increase_lane_count(
        solution=solution,
        validator=LaneValidator(strict_mode=True),
        max_new_lanes=2,
    )

    assert added == 0
    assert attempts["count"] == 1


def test_try_multi_lib_swap_rebalance_stops_retrying_same_pool_after_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"count": 0}

    monkeypatch.setattr(
        arrange_model6,
        "_resolve_lane_capacity_limits",
        lambda libraries, machine_type, lane_id="", lane_metadata=None: (1.0, 100.0),
    )

    def _fake_attempt(**kwargs):
        attempts["count"] += 1
        return None, []

    monkeypatch.setattr(arrange_model6, "_attempt_build_rescue_lane_from_pool", _fake_attempt)

    solution = SchedulingSolution(
        lane_assignments=[],
        unassigned_libraries=[
            _build_library(origrec="RB_FAIL_1", contract_data=10.0),
            _build_library(origrec="RB_FAIL_2", contract_data=10.0),
        ],
    )

    result = try_multi_lib_swap_rebalance(
        solution,
        LaneValidator(strict_mode=True),
        max_new_lanes=2,
    )

    assert result["new_lanes"] == 0
    assert attempts["count"] == 1


def test_enforce_global_priority_hard_constraint_caches_rebuildability_for_same_priority_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clinical = _build_library(origrec="PG_CLIN", data_type="临检", contract_data=100.0)
    other_a = _build_library(origrec="PG_OTHER_A", data_type="其他", contract_data=40.0)
    other_b = _build_library(origrec="PG_OTHER_B", data_type="其他", contract_data=45.0)

    lane_a = _build_lane_assignment("GL_PG_001", [other_a])
    lane_b = _build_lane_assignment("GL_PG_002", [other_b])
    solution = SchedulingSolution(
        lane_assignments=[lane_a, lane_b],
        unassigned_libraries=[clinical],
    )

    monkeypatch.setattr(
        arrange_model6,
        "_resolve_lane_capacity_limits",
        lambda libraries, machine_type, lane_id="", lane_metadata=None: (100.0, 200.0),
    )

    attempts = {"count": 0}

    def _fake_can_rebuild(**kwargs):
        attempts["count"] += 1
        return False

    monkeypatch.setattr(arrange_model6, "_can_rebuild_lane_from_priority_pool", _fake_can_rebuild)

    stats = _enforce_global_priority_hard_constraint(
        solution=solution,
        validator=LaneValidator(strict_mode=True),
    )

    assert attempts["count"] == 1
    assert stats["adjusted_lanes"] == 2
    assert stats["removed_lanes"] == 0
    assert len(solution.lane_assignments) == 2


def test_attempt_build_lane_from_pool_dedupes_equivalent_scattered_mix_combinations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib_a = _build_library(origrec="EQ_A", contract_data=60.0)
    lib_b = _build_library(origrec="EQ_B", contract_data=50.0)
    validation_calls = {"count": 0}

    monkeypatch.setattr(
        arrange_model6,
        "_resolve_lane_capacity_limits",
        lambda libraries, machine_type, lane_id="", lane_metadata=None: (100.0, 200.0),
    )
    monkeypatch.setattr(
        arrange_model6,
        "_validate_lane_special_split_rule",
        lambda libraries: (True, set(), ""),
    )
    monkeypatch.setattr(
        arrange_model6,
        "_validate_lane_57_mix_rules",
        lambda libraries, enforce_total_limit=False, lane_id="", lane_metadata=None: (True, ""),
    )
    monkeypatch.setattr(
        arrange_model6._MODULE_IDX_VALIDATOR,
        "validate_new_lib_quick_with_cache",
        lambda selected_idx_cache, lib: (True, lib.origrec),
    )

    def _fake_candidate_order(
        libraries,
        current_lane_libs=None,
        seed_offset=0,
        machine_type=None,
        special_data_limit=None,
    ):
        ordered = list(libraries)
        return ordered if seed_offset % 2 == 0 else list(reversed(ordered))

    def _fake_validate_lane_with_latest_index(**kwargs):
        validation_calls["count"] += 1
        return types.SimpleNamespace(
            is_valid=False,
            errors=[types.SimpleNamespace(rule_type=arrange_model6.ValidationRuleType.CAPACITY)],
        )

    monkeypatch.setattr(arrange_model6, "_build_scattered_mix_candidate_order", _fake_candidate_order)
    monkeypatch.setattr(arrange_model6, "_validate_lane_with_latest_index", _fake_validate_lane_with_latest_index)

    lane, used = _attempt_build_lane_from_pool(
        pool=[lib_a, lib_b],
        validator=LaneValidator(strict_mode=True),
        machine_type=MachineType.NOVA_X_25B,
        lane_id_prefix="RM",
        prioritize_scattered_mix=True,
        index_conflict_attempts=1,
        other_failure_attempts=4,
    )

    assert lane is None
    assert used == []
    assert validation_calls["count"] == 1


def test_attempt_build_lane_from_pool_caches_equivalent_quick_index_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib_a = _build_library(origrec="IDX_EQ_A", contract_data=60.0, index_seq="AAAA;TTTT")
    lib_b = _build_library(origrec="IDX_EQ_B", contract_data=50.0, index_seq="CCCC;GGGG")
    quick_check_calls = {"count": 0}

    arrange_model6._clear_imbalance_helper_caches()
    monkeypatch.setattr(
        arrange_model6,
        "_resolve_lane_capacity_limits",
        lambda libraries, machine_type, lane_id="", lane_metadata=None: (100.0, 200.0),
    )
    monkeypatch.setattr(
        arrange_model6,
        "_validate_lane_special_split_rule",
        lambda libraries: (True, set(), ""),
    )
    monkeypatch.setattr(
        arrange_model6,
        "_validate_lane_57_mix_rules",
        lambda libraries, enforce_total_limit=False, lane_id="", lane_metadata=None: (True, ""),
    )

    def _fake_validate_new_lib_quick(selected_idx_cache, lib):
        quick_check_calls["count"] += 1
        return True, [(lib.origrec, None)]

    monkeypatch.setattr(
        arrange_model6._MODULE_IDX_VALIDATOR,
        "validate_new_lib_quick_with_cache",
        _fake_validate_new_lib_quick,
    )
    monkeypatch.setattr(
        arrange_model6,
        "_build_scattered_mix_candidate_order",
        lambda libraries, **kwargs: list(libraries),
    )
    monkeypatch.setattr(
        arrange_model6,
        "_validate_lane_with_latest_index",
        lambda **kwargs: types.SimpleNamespace(
            is_valid=False,
            errors=[types.SimpleNamespace(rule_type=arrange_model6.ValidationRuleType.CAPACITY)],
        ),
    )

    lane, used = _attempt_build_lane_from_pool(
        pool=[lib_a, lib_b],
        validator=LaneValidator(strict_mode=True),
        machine_type=MachineType.NOVA_X_25B,
        lane_id_prefix="RM",
        prioritize_scattered_mix=True,
        index_conflict_attempts=1,
        other_failure_attempts=4,
    )

    assert lane is None
    assert used == []
    assert quick_check_calls["count"] == 2


def test_validate_completed_lane_caches_equivalent_library_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = GreedyLaneScheduler()
    scheduler.config.enable_rule_checker = False
    scheduler.config.enable_imbalance_check = False

    lib_a = _build_library(origrec="CACHE_A", contract_data=60.0)
    lib_b = _build_library(origrec="CACHE_B", contract_data=50.0)
    lane_a = _build_lane_assignment("GL_CACHE_001", [lib_a, lib_b])
    lane_b = _build_lane_assignment("GL_CACHE_002", [lib_b, lib_a])

    validation_calls = {"count": 0}

    def _fake_validate_lane(*, libraries, lane_id, machine_type, metadata):
        validation_calls["count"] += 1
        return types.SimpleNamespace(is_valid=True, errors=[])

    monkeypatch.setattr(scheduler.lane_validator, "validate_lane", _fake_validate_lane)

    assert scheduler._validate_completed_lane(lane_a) == (True, [])
    assert scheduler._validate_completed_lane(lane_b) == (True, [])
    assert validation_calls["count"] == 1


def test_is_imbalance_library_candidate_caches_by_library_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib = _build_library(origrec="IMB_CACHE", sample_type="客户-PCR产物")
    calls = {"count": 0}

    def _fake_is_imbalance(target):
        calls["count"] += 1
        return target.origrec == "IMB_CACHE"

    arrange_model6._clear_imbalance_helper_caches()
    monkeypatch.setattr(arrange_model6._BASE_IMBALANCE_HANDLER, "is_imbalance_library", _fake_is_imbalance)

    assert arrange_model6._is_imbalance_library_candidate(lib) is True
    assert arrange_model6._is_imbalance_library_candidate(lib) is True
    assert calls["count"] == 1


def test_summarize_lane_imbalance_caches_same_lane_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib_a = _build_library(origrec="IMB_SUM_A", contract_data=60.0)
    lib_b = _build_library(origrec="IMB_SUM_B", contract_data=40.0)
    calls = {"count": 0}

    def _fake_is_candidate(lib):
        calls["count"] += 1
        return lib.origrec == "IMB_SUM_A"

    arrange_model6._clear_imbalance_helper_caches()
    monkeypatch.setattr(arrange_model6, "_is_imbalance_library_candidate", _fake_is_candidate)

    summary1 = arrange_model6._summarize_lane_imbalance([lib_a, lib_b])
    summary2 = arrange_model6._summarize_lane_imbalance([lib_a, lib_b])

    assert summary1 == (100.0, 60.0, 0.6)
    assert summary2 == summary1
    assert calls["count"] == 2


def test_validate_lane_57_mix_rules_caches_equivalent_library_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib_a = _build_library(origrec="MIX_CACHE_A", contract_data=60.0)
    lib_b = _build_library(origrec="MIX_CACHE_B", contract_data=40.0)
    calls = {"count": 0}

    def _fake_is_candidate(lib):
        return lib.origrec == "MIX_CACHE_A"

    def _fake_check_mix_compatibility(libraries, enforce_total_limit=False):
        calls["count"] += 1
        return True, f"checked_{len(libraries)}"

    arrange_model6._clear_imbalance_helper_caches()
    monkeypatch.setattr(arrange_model6, "_is_imbalance_library_candidate", _fake_is_candidate)
    monkeypatch.setattr(
        arrange_model6._BASE_IMBALANCE_HANDLER,
        "check_mix_compatibility",
        _fake_check_mix_compatibility,
    )

    result1 = arrange_model6._validate_lane_57_mix_rules([lib_a, lib_b])
    result2 = arrange_model6._validate_lane_57_mix_rules([lib_b, lib_a])

    assert result1 == (True, "checked_2")
    assert result2 == result1
    assert calls["count"] == 1


def test_base_imbalance_handler_identify_imbalance_type_caches_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = arrange_model6._BASE_IMBALANCE_HANDLER
    lib = _build_library(origrec="IMB_GROUP_CACHE", sample_type="客户-PCR产物", jjbj="")
    calls = {"count": 0}

    for attr_name in ("_imbalance_group_id_cache", "_imbalance_library_type_cache"):
        if hasattr(lib, attr_name):
            delattr(lib, attr_name)

    def _fake_get_library_type(target):
        calls["count"] += 1
        return "客户-PCR产物"

    monkeypatch.setattr(handler, "_get_library_type", _fake_get_library_type)

    assert handler.identify_imbalance_type(lib) == "G29"
    assert handler.identify_imbalance_type(lib) == "G29"
    assert calls["count"] == 1


def test_resolve_lane_capacity_selection_caches_equivalent_library_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib_a = _build_library(origrec="CAP_CACHE_A", contract_data=60.0)
    lib_b = _build_library(origrec="CAP_CACHE_B", contract_data=40.0)
    calls = {"count": 0}

    class _FakeSchedulingConfig:
        def get_lane_capacity_range(self, libraries, machine_type, metadata=None):
            calls["count"] += 1
            return types.SimpleNamespace(
                rule_code="unit_test_rule",
                max_target_gb=1000.0,
                effective_max_gb=1005.0,
                effective_min_gb=995.0,
            )

    arrange_model6._clear_imbalance_helper_caches()
    monkeypatch.setattr(arrange_model6, "get_scheduling_config", lambda: _FakeSchedulingConfig())
    monkeypatch.setattr(
        arrange_model6,
        "_apply_balance_reservation_to_capacity_selection",
        lambda selection, **kwargs: selection,
    )

    sel1 = arrange_model6._resolve_lane_capacity_selection([lib_a, lib_b], MachineType.NOVA_X_25B)
    sel2 = arrange_model6._resolve_lane_capacity_selection([lib_b, lib_a], MachineType.NOVA_X_25B)

    assert sel1.effective_min_gb == 995.0
    assert sel2.effective_max_gb == 1005.0
    assert calls["count"] == 1


def test_scheduling_config_resolve_seq_strategy_caches_same_candidate_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduling_config = get_scheduling_config()
    scheduling_config._clear_runtime_caches()
    lib = _build_library(
        origrec="SEQ_STRATEGY_CACHE",
        seq_scheme="NovaSeq X Plus-PE150",
        machine_note="PE150",
    )
    calls = {"count": 0}
    original_normalize = type(scheduling_config)._normalize_seq_keyword.__func__

    def _counting_normalize_seq_keyword(value):
        calls["count"] += 1
        return original_normalize(type(scheduling_config), value)

    monkeypatch.setattr(scheduling_config, "_normalize_seq_keyword", _counting_normalize_seq_keyword)

    result1 = scheduling_config._resolve_seq_strategy([lib], {})
    first_call_count = calls["count"]
    result2 = scheduling_config._resolve_seq_strategy([lib], {})

    assert result1 == "PE150"
    assert result2 == result1
    assert first_call_count > 0
    assert calls["count"] == first_call_count


def test_scheduling_config_is_customer_library_caches_method_result() -> None:
    scheduling_config = get_scheduling_config()
    scheduling_config._clear_runtime_caches()
    lib = _build_library(origrec="CUSTOMER_FLAG_CACHE", customer_library="")
    calls = {"count": 0}

    def _fake_is_customer_library():
        calls["count"] += 1
        return True

    lib.is_customer_library = _fake_is_customer_library

    assert scheduling_config._is_customer_library(lib) is True
    assert scheduling_config._is_customer_library(lib) is True
    assert calls["count"] == 1


def test_scheduling_config_resolve_lane_context_features_caches_equivalent_library_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduling_config = get_scheduling_config()
    scheduling_config._clear_runtime_caches()
    lib_a = _build_library(origrec="LANE_CTX_CACHE_A")
    lib_b = _build_library(origrec="LANE_CTX_CACHE_B", sample_type="DNA小片段文库")
    calls = {"count": 0}
    original_classify = scheduling_config._classify_lane_project_type

    def _counting_classify(libraries):
        calls["count"] += 1
        return original_classify(libraries)

    monkeypatch.setattr(scheduling_config, "_classify_lane_project_type", _counting_classify)

    result1 = scheduling_config._resolve_lane_context_features([lib_a, lib_b])
    result2 = scheduling_config._resolve_lane_context_features([lib_b, lib_a])

    assert result1 == result2
    assert calls["count"] == 1


def test_scheduling_config_resolve_lane_rule_selection_caches_equivalent_lane_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduling_config = get_scheduling_config()
    scheduling_config._clear_runtime_caches()

    lib_a1 = _build_library(origrec="RULE_SCOPE_A1", sample_type="DNA小片段文库")
    lib_a1.test_code = 1595
    lib_a1._current_seq_mode_raw = "3.6T-NEW"

    lib_a2 = _build_library(origrec="RULE_SCOPE_A2", sample_type="DNA小片段文库")
    lib_a2.test_code = 1595
    lib_a2._current_seq_mode_raw = "3.6T-NEW"

    calls = {"count": 0}
    original_match_scope = scheduling_config._match_scope

    def _counting_match_scope(normalized_value, allowed_values):
        calls["count"] += 1
        return original_match_scope(normalized_value, allowed_values)

    monkeypatch.setattr(scheduling_config, "_match_scope", _counting_match_scope)

    result1 = scheduling_config.resolve_lane_rule_selection([lib_a1], "Nova X-25B")
    first_call_count = calls["count"]
    result2 = scheduling_config.resolve_lane_rule_selection([lib_a2], "Nova X-25B")

    assert result1 is not None
    assert result2 is not None
    assert result2.rule_code == result1.rule_code
    assert first_call_count > 0
    assert calls["count"] == first_call_count


def test_get_scattered_mix_priority_rank_caches_by_library_signature() -> None:
    lib = _build_library(origrec="SCATTERED_PRIORITY_CACHE", customer_library="")
    calls = {"clinical": 0, "sj": 0, "yc": 0}

    def _fake_is_clinical():
        calls["clinical"] += 1
        return False

    def _fake_is_sj():
        calls["sj"] += 1
        return False

    def _fake_is_yc():
        calls["yc"] += 1
        return True

    lib.is_clinical_by_code = _fake_is_clinical
    lib.is_s_level_customer = _fake_is_sj
    lib.is_yc_library = _fake_is_yc

    assert arrange_model6._get_scattered_mix_priority_rank(lib) == 1
    assert arrange_model6._get_scattered_mix_priority_rank(lib) == 1
    assert calls == {"clinical": 1, "sj": 1, "yc": 1}


def test_sort_remaining_for_scattered_mix_lane_caches_equivalent_library_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib_a = _build_library(origrec="SCATTERED_SORT_CACHE_A", contract_data=60.0)
    lib_b = _build_library(origrec="SCATTERED_SORT_CACHE_B", contract_data=40.0)
    calls = {"count": 0}
    original_sort = arrange_model6._sort_by_board_preference_for_scattered_mix

    def _counting_sort(libraries):
        calls["count"] += 1
        return original_sort(libraries)

    arrange_model6._clear_imbalance_helper_caches()
    monkeypatch.setattr(arrange_model6, "_sort_by_board_preference_for_scattered_mix", _counting_sort)

    result1 = arrange_model6._sort_remaining_for_scattered_mix_lane([lib_a, lib_b])
    result2 = arrange_model6._sort_remaining_for_scattered_mix_lane([lib_b, lib_a])

    assert [lib.origrec for lib in result1] == [lib.origrec for lib in result2]
    assert calls["count"] == 1


def test_resolve_balance_reservation_context_fast_skips_known_non_package_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib = _build_library(origrec="BAL_FAST_SKIP", contract_data=60.0)

    monkeypatch.setattr(
        arrange_model6,
        "_get_explicit_balance_data_from_context",
        lambda libraries, lane_metadata=None: (_ for _ in ()).throw(AssertionError("should not scan explicit balance")),
    )
    monkeypatch.setattr(
        arrange_model6,
        "_is_package_lane_context",
        lambda libraries, lane_metadata=None: (_ for _ in ()).throw(AssertionError("should not check package context")),
    )

    result = arrange_model6._resolve_balance_reservation_context(
        libraries=[lib],
        machine_type=MachineType.NOVA_X_25B,
        lane_id="MX_TMP",
        lane_metadata={},
    )

    assert result == {"applied": False}


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


def test_attempt_build_lane_from_pool_continues_post_min_to_improve_imbalance_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libs = [
        _build_library(origrec="BAL_60", contract_data=60.0, jjbj="否"),
        _build_library(origrec="BAL_40", contract_data=40.0, jjbj="否"),
        _build_library(origrec="IMB_35", contract_data=35.0, jjbj="是"),
        _build_library(origrec="BAL_5", contract_data=5.0, jjbj="否"),
    ]

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
        lambda libraries, machine_type, lane_id="", lane_metadata=None: (100.0, 140.0),
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
    monkeypatch.setattr(
        arrange_model6._BASE_IMBALANCE_HANDLER,
        "is_imbalance_library",
        lambda lib: getattr(lib, "jjbj", "否") == "是",
    )

    lane, used = _attempt_build_lane_from_pool(
        pool=list(libs),
        validator=object(),
        machine_type=MachineType.NOVA_X_25B,
        lane_id_prefix="RM",
        lane_serial=1,
        index_conflict_attempts=1,
        other_failure_attempts=1,
        prioritize_scattered_mix=True,
    )

    assert lane is not None and used
    assert {lib.origrec for lib in used} == {"BAL_60", "IMB_35", "BAL_5"}


def test_attempt_build_lane_from_prioritized_pool_continues_post_min_to_improve_imbalance_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_pool = [_build_library(origrec="BAL_60", contract_data=60.0, jjbj="否")]
    secondary_pool = [
        _build_library(origrec="BAL_40", contract_data=40.0, jjbj="否"),
        _build_library(origrec="IMB_35", contract_data=35.0, jjbj="是"),
        _build_library(origrec="BAL_5", contract_data=5.0, jjbj="否"),
    ]

    monkeypatch.setattr(
        arrange_model6,
        "_filter_priority_across_pools",
        lambda primary_pool, secondary_pool, **kwargs: (list(primary_pool), list(secondary_pool), 2),
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
        lambda libraries, machine_type, lane_id="", lane_metadata=None: (100.0, 140.0),
    )
    monkeypatch.setattr(
        arrange_model6._MODULE_IDX_VALIDATOR,
        "validate_new_lib_quick_with_cache",
        lambda selected_idx_cache, lib: (True, []),
    )
    monkeypatch.setattr(
        arrange_model6,
        "_validate_lane_state",
        lambda validator, lane, libraries: types.SimpleNamespace(is_valid=True, errors=[], warnings=[]),
    )
    monkeypatch.setattr(
        arrange_model6._BASE_IMBALANCE_HANDLER,
        "is_imbalance_library",
        lambda lib: getattr(lib, "jjbj", "否") == "是",
    )

    lane, used = _attempt_build_lane_from_prioritized_pool(
        primary_pool=list(primary_pool),
        secondary_pool=list(secondary_pool),
        validator=object(),
        machine_type=MachineType.NOVA_X_25B,
        lane_id_prefix="RS",
        lane_serial=1,
    )

    assert lane is not None and used
    assert [lib.origrec for lib in used] == ["BAL_60", "BAL_40", "IMB_35"]


def test_attempt_build_lane_from_pool_does_not_repeat_same_scattered_mix_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libs = [
        _build_library(origrec="BAL_60", contract_data=60.0, jjbj="否"),
        _build_library(origrec="BAL_40", contract_data=40.0, jjbj="否"),
        _build_library(origrec="IMB_35", contract_data=35.0, jjbj="是"),
    ]
    validate_calls = {"count": 0}

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
        lambda libraries, machine_type, lane_id="", lane_metadata=None: (100.0, 140.0),
    )
    monkeypatch.setattr(
        arrange_model6._MODULE_IDX_VALIDATOR,
        "validate_new_lib_quick_with_cache",
        lambda selected_idx_cache, lib: (True, []),
    )
    monkeypatch.setattr(
        arrange_model6._BASE_IMBALANCE_HANDLER,
        "is_imbalance_library",
        lambda lib: getattr(lib, "jjbj", "否") == "是",
    )

    def _invalid_once(**kwargs):
        validate_calls["count"] += 1
        return types.SimpleNamespace(
            is_valid=False,
            errors=[types.SimpleNamespace(rule_type=arrange_model6.ValidationRuleType.BASE_IMBALANCE_RATIO)],
            warnings=[],
        )

    monkeypatch.setattr(arrange_model6, "_validate_lane_with_latest_index", _invalid_once)

    lane, used = _attempt_build_lane_from_pool(
        pool=list(libs),
        validator=object(),
        machine_type=MachineType.NOVA_X_25B,
        lane_id_prefix="EX",
        lane_serial=1,
        index_conflict_attempts=5,
        other_failure_attempts=5,
        prioritize_scattered_mix=True,
    )

    assert lane is None
    assert used == []
    assert validate_calls["count"] == 1


def test_attempt_build_lane_from_pool_allows_bounded_rm_scattered_mix_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libs = [
        _build_library(origrec="ALT_A", contract_data=60.0, jjbj="否"),
        _build_library(origrec="ALT_B", contract_data=40.0, jjbj="否"),
        _build_library(origrec="ALT_C", contract_data=60.0, jjbj="是"),
    ]
    validate_calls = {"count": 0}

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
        lambda libraries, machine_type, lane_id="", lane_metadata=None: (100.0, 120.0),
    )
    monkeypatch.setattr(
        arrange_model6._MODULE_IDX_VALIDATOR,
        "validate_new_lib_quick_with_cache",
        lambda selected_idx_cache, lib: (True, []),
    )

    def _fake_scattered_mix_order(
        libraries,
        current_lane_libs=None,
        seed_offset=0,
        machine_type=None,
        special_data_limit=None,
    ):
        remaining = tuple(lib.origrec for lib in libraries)
        if current_lane_libs:
            return list(libraries)
        if remaining == ("ALT_A", "ALT_B", "ALT_C") and seed_offset == 0:
            return [libs[0], libs[1], libs[2]]
        if remaining == ("ALT_A", "ALT_B", "ALT_C") and seed_offset == 1:
            return [libs[2], libs[0], libs[1]]
        if remaining == ("ALT_A", "ALT_B") and seed_offset == 1:
            return [libs[0], libs[1]]
        return list(libraries)

    def _fake_validate(**kwargs):
        validate_calls["count"] += 1
        selected = [lib.origrec for lib in kwargs["libraries"]]
        if selected == ["ALT_C", "ALT_A"]:
            return types.SimpleNamespace(is_valid=True, errors=[], warnings=[])
        return types.SimpleNamespace(
            is_valid=False,
            errors=[types.SimpleNamespace(rule_type=arrange_model6.ValidationRuleType.BASE_IMBALANCE_RATIO)],
            warnings=[],
        )

    monkeypatch.setattr(arrange_model6, "_build_scattered_mix_candidate_order", _fake_scattered_mix_order)
    monkeypatch.setattr(arrange_model6, "_validate_lane_with_latest_index", _fake_validate)

    lane, used = _attempt_build_lane_from_pool(
        pool=list(libs),
        validator=object(),
        machine_type=MachineType.NOVA_X_25B,
        lane_id_prefix="RM",
        lane_serial=1,
        index_conflict_attempts=5,
        other_failure_attempts=5,
        prioritize_scattered_mix=True,
    )

    assert lane is not None and used
    assert [lib.origrec for lib in used] == ["ALT_C", "ALT_A"]
    assert validate_calls["count"] == 2


def test_attempt_build_lane_from_pool_respects_special_library_cap_when_filling_imbalance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libs = [
        _build_library(origrec="BAL_60", contract_data=60.0, jjbj="否"),
        _build_library(origrec="BAL_40", contract_data=40.0, jjbj="否"),
        _build_library(origrec="IMB_35", contract_data=35.0, jjbj="是"),
        _build_library(origrec="IMB_20", contract_data=20.0, jjbj="是"),
    ]

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
        lambda libraries, machine_type, lane_id="", lane_metadata=None: (100.0, 140.0),
    )
    monkeypatch.setattr(
        arrange_model6,
        "_resolve_special_library_data_limit",
        lambda machine_type: 20.0,
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
    monkeypatch.setattr(
        arrange_model6._BASE_IMBALANCE_HANDLER,
        "is_imbalance_library",
        lambda lib: getattr(lib, "jjbj", "否") == "是",
    )

    lane, used = _attempt_build_lane_from_pool(
        pool=list(libs),
        validator=object(),
        machine_type=MachineType.NOVA_X_25B,
        lane_id_prefix="RM",
        lane_serial=1,
        index_conflict_attempts=1,
        other_failure_attempts=1,
        prioritize_scattered_mix=True,
    )

    assert lane is not None and used
    assert {lib.origrec for lib in used} == {"BAL_60", "BAL_40", "IMB_20"}


def test_attempt_build_lane_from_prioritized_pool_does_not_repeat_same_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_calls = {"count": 0}
    primary_pool = [_build_library(origrec="BAL_60", contract_data=60.0, jjbj="否")]
    secondary_pool = [
        _build_library(origrec="BAL_40", contract_data=40.0, jjbj="否"),
        _build_library(origrec="IMB_35", contract_data=35.0, jjbj="是"),
    ]

    monkeypatch.setattr(
        arrange_model6,
        "_filter_priority_across_pools",
        lambda primary_pool, secondary_pool, **kwargs: (list(primary_pool), list(secondary_pool), 2),
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
        lambda libraries, machine_type, lane_id="", lane_metadata=None: (100.0, 140.0),
    )
    monkeypatch.setattr(
        arrange_model6._MODULE_IDX_VALIDATOR,
        "validate_new_lib_quick_with_cache",
        lambda selected_idx_cache, lib: (True, []),
    )
    monkeypatch.setattr(
        arrange_model6._BASE_IMBALANCE_HANDLER,
        "is_imbalance_library",
        lambda lib: getattr(lib, "jjbj", "否") == "是",
    )

    def _invalid_state(validator, lane, libraries):
        validate_calls["count"] += 1
        return types.SimpleNamespace(
            is_valid=False,
            errors=[types.SimpleNamespace(rule_type=arrange_model6.ValidationRuleType.BASE_IMBALANCE_RATIO)],
            warnings=[],
        )

    monkeypatch.setattr(arrange_model6, "_validate_lane_state", _invalid_state)

    lane, used = _attempt_build_lane_from_prioritized_pool(
        primary_pool=list(primary_pool),
        secondary_pool=list(secondary_pool),
        validator=object(),
        machine_type=MachineType.NOVA_X_25B,
        lane_id_prefix="RS",
        lane_serial=1,
        index_conflict_attempts=5,
        other_failure_attempts=5,
    )

    assert lane is None
    assert used == []
    assert validate_calls["count"] == 1


def test_try_targeted_imbalance_upgrade_swaps_in_unassigned_imbalance_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = _build_lane_assignment(
        "RM_Nova X-25B_001",
        [
            _build_library(origrec="BAL_900", contract_data=900.0, jjbj="否", data_type="其他"),
            _build_library(origrec="BAL_100", contract_data=100.0, jjbj="否", data_type="其他"),
        ],
    )
    incoming = _build_library(origrec="IMB_100", contract_data=100.0, jjbj="是", data_type="其他")
    solution = types.SimpleNamespace(
        lane_assignments=[lane],
        unassigned_libraries=[incoming],
    )

    monkeypatch.setattr(
        arrange_model6,
        "_validate_lane_state",
        lambda validator, lane, libraries, **kwargs: types.SimpleNamespace(is_valid=True, errors=[], warnings=[]),
    )
    monkeypatch.setattr(
        arrange_model6,
        "_resolve_special_library_data_limit",
        lambda machine_type: 350.0,
    )

    result = try_targeted_imbalance_upgrade(
        solution=solution,
        validator=object(),
        max_successful_swaps=2,
        max_candidate_libraries=4,
        max_lanes_per_candidate=2,
        max_donors_per_lane=4,
    )

    assert result["successful_swaps"] == 1
    assert result["changed_lanes"] == 1
    assert result["consumed_imbalance_gb"] == 100.0
    assert {lib.origrec for lib in lane.libraries} == {"BAL_900", "IMB_100"}
    assert {lib.origrec for lib in solution.unassigned_libraries} == {"BAL_100"}


def test_attempt_build_rescue_lane_from_pool_uses_fast_path_for_og(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def _fake_attempt(**kwargs):
        calls.append(kwargs)
        return None, []

    monkeypatch.setattr(arrange_model6, "_attempt_build_lane_from_pool", _fake_attempt)

    _attempt_build_rescue_lane_from_pool(
        pool=[_build_library(origrec="OG_1", contract_data=100.0)],
        validator=object(),
        machine_type=MachineType.NOVA_X_25B,
        lane_id_prefix="OG",
        lane_serial=1,
        index_conflict_attempts=3,
        other_failure_attempts=4,
    )

    assert len(calls) == 1
    assert calls[0]["prioritize_scattered_mix"] is False
    assert calls[0]["lane_id_prefix"] == "OG"


def test_apply_balance_reservation_to_capacity_selection_uses_package_lane_bound_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _lane_rule(min_gb=995.0, max_gb=1105.0)

    monkeypatch.setattr(
        arrange_model6,
        "_resolve_balance_reservation_context",
        lambda **kwargs: {
            "applied": True,
            "mode": "absolute",
            "reserve_gb": 50.0,
            "reserve_ratio": 0.0,
        },
    )

    result = _apply_balance_reservation_to_capacity_selection(
        selection,
        libraries=[],
        machine_type=MachineType.NOVA_X_25B,
    )

    assert result.effective_min_gb == arrange_model6.PACKAGE_LANE_MIN_GB - 50.0
    assert result.effective_max_gb == arrange_model6.PACKAGE_LANE_MAX_GB - 50.0


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


def test_schedule_stops_after_stagnant_retry_pool_repeats(
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
    monkeypatch.setattr(scheduler, "_pool_has_enough_data_for_lane", lambda libraries, machine_type, lane=None: True)
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

    assert call_counter["count"] == 5
    assert len(solution.lane_assignments) == 0
    assert len(solution.unassigned_libraries) == len(libraries)


def test_schedule_keeps_retrying_when_failed_pool_is_still_changing(
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
    monkeypatch.setattr(scheduler, "_pool_has_enough_data_for_lane", lambda libraries, machine_type, lane=None: True)
    monkeypatch.setattr(
        scheduler,
        "_group_by_machine_type",
        lambda libraries: {"Nova X-25B": list(libraries)},
    )

    call_counter = {"count": 0}

    def fake_schedule_machine_group(libraries, machine_type):
        call_counter["count"] += 1
        if len(libraries) <= 1:
            return [], list(libraries)
        return [], list(libraries[1:])

    monkeypatch.setattr(scheduler, "_schedule_machine_group", fake_schedule_machine_group)

    libraries = [
        _build_library(
            origrec=f"ROUND_CHANGE_{idx}",
            contract_data=10.0,
            board_number=f"RC{idx:02d}",
        )
        for idx in range(20)
    ]

    solution = scheduler.schedule(libraries, libraries_already_split=True)

    assert call_counter["count"] == 11
    assert len(solution.lane_assignments) == 0
    assert len(solution.unassigned_libraries) == 9


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
        lane_validation_cache=None,
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


def test_enforce_mode_1_1_priority_cap_per_lane_moves_overflow_priority_to_unassigned() -> None:
    allocator = arrange_model6.ModeAllocator(
        {"priority_data_types_for_36t": ["临检", "YC", "SJ"]}
    )

    clin_110 = _build_library(origrec="CLIN_110", data_type="临检", contract_data=110.0)
    yc_90 = _build_library(origrec="YC_90", data_type="YC", contract_data=90.0)
    other_800 = _build_library(origrec="OTHER_800", data_type="其他", contract_data=800.0)
    normal_lane = _build_lane_assignment("L1", [clin_110, yc_90, other_800])

    sj_120 = _build_library(origrec="SJ_120", data_type="SJ", contract_data=120.0)
    other_700 = _build_library(origrec="OTHER_700", data_type="其他", contract_data=700.0)
    untouched_lane = _build_lane_assignment("L2", [sj_120, other_700])

    existing_unassigned = _build_library(origrec="UNASSIGNED_50", data_type="其他", contract_data=50.0)
    solution = SchedulingSolution(
        lane_assignments=[normal_lane, untouched_lane],
        unassigned_libraries=[existing_unassigned],
    )

    stats = _enforce_mode_1_1_priority_cap_per_lane(
        solution,
        allocator=allocator,
        max_priority_gb_per_lane=150.0,
    )

    assert stats == {
        "adjusted_lanes": 1,
        "overflow_libraries": 1,
        "kept_priority_gb": 230.0,
        "removed_priority_gb": 90.0,
    }
    assert [lib.origrec for lib in solution.lane_assignments[0].libraries] == [
        "CLIN_110",
        "OTHER_800",
    ]
    assert [lib.origrec for lib in solution.lane_assignments[1].libraries] == [
        "SJ_120",
        "OTHER_700",
    ]
    assert [lib.origrec for lib in solution.unassigned_libraries] == [
        "UNASSIGNED_50",
        "YC_90",
    ]


def test_consume_mode_1_1_priority_from_unassigned_fills_existing_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = arrange_model6.ModeAllocator(
        {"priority_data_types_for_36t": ["临检", "YC", "SJ"]}
    )

    lane1 = _build_lane_assignment(
        "L1",
        [_build_library(origrec="NORMAL_2000", data_type="其他", contract_data=2000.0)],
    )
    lane2 = _build_lane_assignment(
        "L2",
        [_build_library(origrec="NORMAL_2140", data_type="其他", contract_data=2140.0)],
    )
    clin_60 = _build_library(origrec="CLIN_60", data_type="临检", contract_data=60.0)
    yc_40 = _build_library(origrec="YC_40", data_type="YC", contract_data=40.0)
    sj_30 = _build_library(origrec="SJ_30", data_type="SJ", contract_data=30.0)
    other_20 = _build_library(origrec="OTHER_20", data_type="其他", contract_data=20.0)
    solution = SchedulingSolution(
        lane_assignments=[lane1, lane2],
        unassigned_libraries=[clin_60, yc_40, sj_30, other_20],
    )

    def _fake_validate_lane_state(validator, lane, libraries, **kwargs):
        total = sum(lib.get_data_amount_gb() for lib in libraries)
        return types.SimpleNamespace(is_valid=total <= 2205.0, errors=[], warnings=[])

    monkeypatch.setattr(arrange_model6, "_validate_lane_state", _fake_validate_lane_state)

    stats = _consume_mode_1_1_priority_from_unassigned(
        solution,
        allocator=allocator,
        max_priority_gb_per_lane=100.0,
    )

    assert stats == {
        "consumed_libraries": 3,
        "consumed_gb": 130.0,
        "changed_lanes": 2,
        "remaining_priority_libraries": 0,
    }
    assert [lib.origrec for lib in solution.lane_assignments[0].libraries] == [
        "NORMAL_2000",
        "YC_40",
        "SJ_30",
    ]
    assert [lib.origrec for lib in solution.lane_assignments[1].libraries] == [
        "NORMAL_2140",
        "CLIN_60",
    ]
    assert [lib.origrec for lib in solution.unassigned_libraries] == [
        "OTHER_20",
    ]
