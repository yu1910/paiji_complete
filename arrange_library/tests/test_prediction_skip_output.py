import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from arrange_library.arrange_library_model6 import (
    BALANCE_LIBRARY_MARKER_COLUMN,
    _run_prediction_delivery,
)


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
