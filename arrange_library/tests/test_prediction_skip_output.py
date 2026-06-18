import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from arrange_library.arrange_library_model6 import (
    BALANCE_LIBRARY_MARKER_COLUMN,
    _resolve_lane_output_rule_fields,
    _run_prediction_delivery,
)
from arrange_library.core.constraints.index_validator_verified import IndexConflictValidator
from arrange_library.models.library_info import EnhancedLibraryInfo, MachineType


def test_run_prediction_delivery_skips_prediction_and_clears_output_fields(tmp_path):
    output_path = tmp_path / "prediction_skipped.csv"
    input_df = pd.DataFrame(
        [
            {
                "origrec": "LIB001",
                "laneround": "1.1第二轮",
                "wkcontractdata": 12.0,
                "lorderdata": 8.0,
                "lai_output": 10.5,
                "predicted_lorderdata": 8.0,
                "ai_predicted_lorderdata": 8.1,
                "ai_predicted_loutput": 10.4,
                "resolved_round2_pooling_factor": 2.5,
                "resolved_round2_balance_ratio": 0.1,
                BALANCE_LIBRARY_MARKER_COLUMN: True,
            }
        ]
    )

    result_df = _run_prediction_delivery(input_data=input_df, output_path=output_path)

    assert output_path.exists()
    assert pd.isna(result_df.loc[0, "lorderdata"])
    assert pd.isna(result_df.loc[0, "lai_output"])
    assert "predicted_lorderdata" not in result_df.columns
    assert "ai_predicted_lorderdata" not in result_df.columns
    assert "ai_predicted_loutput" not in result_df.columns
    assert "resolved_round2_pooling_factor" not in result_df.columns
    assert "resolved_round2_balance_ratio" not in result_df.columns
    assert BALANCE_LIBRARY_MARKER_COLUMN not in result_df.columns

    written_df = pd.read_csv(output_path)
    assert pd.isna(written_df.loc[0, "lorderdata"])
    assert pd.isna(written_df.loc[0, "lai_output"])
    assert "predicted_lorderdata" not in written_df.columns
    assert "ai_predicted_lorderdata" not in written_df.columns
    assert "ai_predicted_loutput" not in written_df.columns


def test_index_conflict_uses_shorter_length_for_p7_left_align():
    validator = IndexConflictValidator()

    is_repeat, _, same_left, same_right = validator._check_index_pair_repeat(
        "AACCGGTT",
        None,
        "AACCGGTTAA",
        None,
    )

    assert is_repeat is True
    assert same_left == 8
    assert same_right is None


def test_index_conflict_uses_shorter_length_for_p5_right_align():
    validator = IndexConflictValidator()

    is_repeat, _, same_left, same_right = validator._check_index_pair_repeat(
        "AACCGGTT",
        "AACCGGTT",
        "AACCGGTTAA",
        "TTAACCGGTT",
    )

    assert is_repeat is True
    assert same_left == 8
    assert same_right == 8


def test_index_conflict_8bp_vs_10bp_does_not_pad_short_index():
    validator = IndexConflictValidator()

    left_repeat, left_same = validator._side_is_repeated_left("AACCGGTT", "AACCGGTTAA")
    right_repeat, right_same = validator._side_is_repeated_right("AACCGGTT", "TTAACCGGTT")

    assert left_repeat is True
    assert left_same == 8
    assert right_repeat is True
    assert right_same == 8


def _make_singapore_pe150_lib(origrec: str, contract_data_raw: float) -> EnhancedLibraryInfo:
    lib = EnhancedLibraryInfo(
        origrec=origrec,
        sample_id=f"SG_{origrec}",
        sample_type_code="人重测序文库",
        data_type="其他",
        customer_library="否",
        base_type="双",
        number_of_bases=10,
        index_number=1,
        index_seq="AACCGGTTAA;TTAACCGGTT",
        add_tests_remark="非加测",
        product_line="S",
        eq_type="Nova X-25B",
        contract_data_raw=contract_data_raw,
        peak_size=460,
        test_code=1738,
        test_no="Novaseq X Plus-PE150",
        sub_project_name="TEST",
        create_date="2026-06-18",
        delivery_date="2026-06-25",
        lab_type="诺禾自动",
        data_volume_type="G",
        board_number="BN001",
    )
    lib.process_code = 1738
    lib.wktestno = "Novaseq X Plus-PE150"
    lib._wkdept_raw = "新加坡科技服务实验室"
    lib._current_seq_mode_raw = "3.6T-NEW"
    lib.selected_seq_mode = "3.6T-NEW"
    lib.current_seq_mode = "3.6T-NEW"
    lib.seq_mode = "1.1"
    lib.lcxms = "3.6T-NEW"
    return lib


def test_singapore_pe150_hyphenated_test_no_matches_standard_25b_rule():
    libs = [_make_singapore_pe150_lib(str(idx), 150.0) for idx in range(7)]
    libs[0].contract_data_raw = 153.458

    loading_method, sequencing_mode, rule_code = _resolve_lane_output_rule_fields(
        libraries=libs,
        machine_type=MachineType.NOVA_X_25B,
        lane_id="GM_Nova X-25B_001",
        lane_metadata={
            "selected_seq_mode": "3.6T-NEW",
            "seq_mode": "3.6T-NEW",
            "lcxms": "3.6T-NEW",
            "dispatch_stage": "terminal_global_36t_split_mixed",
        },
    )

    assert loading_method == "25B"
    assert sequencing_mode == "3.6T"
    assert rule_code == "sg_1738_standard_pe150_25b"
