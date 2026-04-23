import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from arrange_library.arrange_library_model6 import (
    BALANCE_LIBRARY_MARKER_COLUMN,
    _build_detail_output,
)


def test_build_detail_output_keeps_package_lane_fields_when_57_recheck_runs(tmp_path):
    output_path = tmp_path / "package_lane_detail_output.csv"
    df_raw = pd.DataFrame(
        [
            {
                "wkorigrec": "11266224",
                "origrec": "11266224",
                "wksampleid": "FKDL250209501-1A",
                "wksampletype": "客户-10X ATAC文库",
                "wkdatatype": "客户-10X ATAC文库",
                "wkbaleno": "C2511044598",
                "aiarrangenumber": 0,
            },
            {
                "wkorigrec": "11266226",
                "origrec": "11266226",
                "wksampleid": "FKDL250209503-1A",
                "wksampletype": "客户-10X ATAC文库",
                "wkdatatype": "客户-10X ATAC文库",
                "wkbaleno": "C2511044598",
                "aiarrangenumber": 0,
            },
            {
                "wkorigrec": "11266225",
                "origrec": "11266225",
                "wksampleid": "FKDL250209502-1A",
                "wksampletype": "客户-10X ATAC文库",
                "wkdatatype": "客户-10X ATAC文库",
                "wkbaleno": "C2511044598",
                "aiarrangenumber": 0,
            },
            {
                "wkorigrec": "11266223",
                "origrec": "11266223",
                "wksampleid": "FKDL250209500-1A",
                "wksampletype": "客户-10X ATAC文库",
                "wkdatatype": "客户-10X ATAC文库",
                "wkbaleno": "C2511044598",
                "aiarrangenumber": 0,
            },
        ]
    )
    pred_df = pd.DataFrame(
        [
            {
                "origrec": "11266224",
                "origrec_key": "11266224",
                "detail_row_key": "PKG_LIB_001",
                "runid": "RUN_001",
                "lane_id": "LANE_0001",
                "lsjnd": 2.0,
                "resolved_lsjfs": "25B",
                "resolved_lcxms": "LANE SEQ",
                "resolved_index_check_rule": "P7P5",
                "wkbalancedata": None,
                "predicted_lorderdata": 225.0,
                "lai_output": None,
                BALANCE_LIBRARY_MARKER_COLUMN: False,
            },
            {
                "origrec": "11266226",
                "origrec_key": "11266226",
                "detail_row_key": "PKG_LIB_002",
                "runid": "RUN_001",
                "lane_id": "LANE_0001",
                "lsjnd": 2.0,
                "resolved_lsjfs": "25B",
                "resolved_lcxms": "LANE SEQ",
                "resolved_index_check_rule": "P7P5",
                "wkbalancedata": None,
                "predicted_lorderdata": 225.0,
                "lai_output": None,
                BALANCE_LIBRARY_MARKER_COLUMN: False,
            },
            {
                "origrec": "11266225",
                "origrec_key": "11266225",
                "detail_row_key": "PKG_LIB_003",
                "runid": "RUN_001",
                "lane_id": "LANE_0001",
                "lsjnd": 2.0,
                "resolved_lsjfs": "25B",
                "resolved_lcxms": "LANE SEQ",
                "resolved_index_check_rule": "P7P5",
                "wkbalancedata": None,
                "predicted_lorderdata": 225.0,
                "lai_output": None,
                BALANCE_LIBRARY_MARKER_COLUMN: False,
            },
            {
                "origrec": "11266223",
                "origrec_key": "11266223",
                "detail_row_key": "PKG_LIB_004",
                "runid": "RUN_001",
                "lane_id": "LANE_0001",
                "lsjnd": 2.0,
                "resolved_lsjfs": "25B",
                "resolved_lcxms": "LANE SEQ",
                "resolved_index_check_rule": "P7P5",
                "wkbalancedata": None,
                "predicted_lorderdata": 225.0,
                "lai_output": None,
                BALANCE_LIBRARY_MARKER_COLUMN: False,
            },
            {
                "origrec": "AI_BALANCE_LANE_0001",
                "origrec_key": "AI_BALANCE_LANE_0001",
                "detail_row_key": "PKG_BAL_001",
                "runid": "RUN_001",
                "lane_id": "LANE_0001",
                "lsjnd": 2.0,
                "resolved_lsjfs": "25B",
                "resolved_lcxms": "LANE SEQ",
                "resolved_index_check_rule": "P7P5",
                "wkbalancedata": 100.0,
                "predicted_lorderdata": 100.0,
                "lai_output": None,
                BALANCE_LIBRARY_MARKER_COLUMN: True,
            },
        ]
    )
    detail_libraries = [
        SimpleNamespace(
            origrec="11266224",
            _origrec_key="11266224",
            _source_origrec_key="11266224",
            _detail_output_key="PKG_LIB_001",
            contract_data_raw=225.0,
            package_lane_number="C2511044598",
            baleno="C2511044598",
        ),
        SimpleNamespace(
            origrec="11266226",
            _origrec_key="11266226",
            _source_origrec_key="11266226",
            _detail_output_key="PKG_LIB_002",
            contract_data_raw=225.0,
            package_lane_number="C2511044598",
            baleno="C2511044598",
        ),
        SimpleNamespace(
            origrec="11266225",
            _origrec_key="11266225",
            _source_origrec_key="11266225",
            _detail_output_key="PKG_LIB_003",
            contract_data_raw=225.0,
            package_lane_number="C2511044598",
            baleno="C2511044598",
        ),
        SimpleNamespace(
            origrec="11266223",
            _origrec_key="11266223",
            _source_origrec_key="11266223",
            _detail_output_key="PKG_LIB_004",
            contract_data_raw=225.0,
            package_lane_number="C2511044598",
            baleno="C2511044598",
        ),
        SimpleNamespace(
            origrec="AI_BALANCE_LANE_0001",
            _origrec_key="AI_BALANCE_LANE_0001",
            _source_origrec_key="AI_BALANCE_LANE_0001",
            _detail_output_key="PKG_BAL_001",
            contract_data_raw=100.0,
            _is_ai_balance_library=True,
            _balance_output_payload={
                "wkaidbid": "PKG_BAL_001",
                "wksampleid": "phix",
                "wkindexseq": "GGGGGGGGGG;ACCGAGATCT",
                "wkcontractdata": 100.0,
                "origrec": "AI_BALANCE_LANE_0001",
                "origrec_key": "AI_BALANCE_LANE_0001",
                "detail_row_key": "PKG_BAL_001",
                BALANCE_LIBRARY_MARKER_COLUMN: True,
            },
        ),
    ]

    _build_detail_output(
        df_raw=df_raw,
        pred_df=pred_df,
        output_path=output_path,
        ai_schedulable_keys={"11266224", "11266226", "11266225", "11266223"},
        detail_libraries=detail_libraries,
    )

    result = pd.read_csv(output_path, keep_default_na=False)
    package_rows = result.loc[
        result["wkbaleno"].astype(str).eq("C2511044598")
        | result["wksampleid"].astype(str).str.lower().eq("phix")
    ].copy()

    assert len(package_rows) == 5
    assert set(package_rows["llaneid"]) == {"LANE_0001"}
    assert set(package_rows["lrunid"]) == {"RUN_001"}
    assert set(package_rows["lanecreatetype"]) == {"AI"}
    assert set(package_rows["lcxms"]) == {"Lane seq"}
    assert set(package_rows["lsjfs"]) == {"25B"}
