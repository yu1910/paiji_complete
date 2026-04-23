import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from arrange_library.arrange_library_model6 import _build_detail_output


def test_build_detail_output_backfills_mode_1_1_first_round_laneround_when_round_label_missing(tmp_path):
    output_path = tmp_path / "mode_1_1_laneround_backfill.csv"
    df_raw = pd.DataFrame(
        [
            {
                "wkorigrec": "10001",
                "origrec": "10001",
                "wksampleid": "FDHE260100-1A",
                "aiarrangenumber": 0,
            }
        ]
    )
    pred_df = pd.DataFrame(
        [
            {
                "origrec": "10001",
                "origrec_key": "10001",
                "detail_row_key": "10001",
                "runid": "RUN_001",
                "lane_id": "LANE_001",
                "resolved_lsjfs": "25B",
                "resolved_lcxms": "1.1",
                "resolved_seq_mode": "1.1",
                "resolved_round_label": None,
                "resolved_index_check_rule": "P7P5",
                "predicted_lorderdata": None,
                "lai_output": None,
            }
        ]
    )

    _build_detail_output(
        df_raw=df_raw,
        pred_df=pred_df,
        output_path=output_path,
        ai_schedulable_keys={"10001"},
    )

    result = pd.read_csv(output_path, keep_default_na=False)
    assert str(result.loc[0, "lcxms"]) == "1.1"
    assert result.loc[0, "laneround"] == "1.1第一轮"
