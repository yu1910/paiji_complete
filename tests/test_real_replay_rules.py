from pathlib import Path
import sys
import types

import pandas as pd


PAIJI_ROOT = Path(__file__).resolve().parents[1]
if str(PAIJI_ROOT) not in sys.path:
    sys.path.insert(0, str(PAIJI_ROOT))

sys.modules.setdefault(
    "prediction_delivery",
    types.SimpleNamespace(MODELS_DIR=Path("."), predict_pooling=lambda *args, **kwargs: None),
)

import arrange_library.arrange_library_model6 as arrange_model6


REAL_REPLAY_DIR = Path(__file__).resolve().parent / "real_replay_data"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def _fake_predict_pooling_factory(
    order_map: dict[str, float],
    output_map: dict[str, float],
):
    def _fake_predict_pooling(input_data, output_file):
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            df = _read_csv(Path(input_data))

        key_column = "origrec_key"
        if key_column not in df.columns:
            for candidate in ("wkorigrec", "origrec"):
                if candidate in df.columns:
                    key_column = candidate
                    break

        default_order = pd.to_numeric(df.get("lorderdata"), errors="coerce")
        if default_order.isna().all():
            default_order = pd.to_numeric(df.get("wkcontractdata"), errors="coerce")

        default_output = pd.to_numeric(df.get("lai_output"), errors="coerce")
        if default_output.isna().all():
            default_output = pd.to_numeric(df.get("loutput"), errors="coerce")

        df["lorderdata"] = (
            df[key_column].astype(str).map(order_map).astype(float).combine_first(default_order)
        )
        df["lai_output"] = (
            df[key_column].astype(str).map(output_map).astype(float).combine_first(default_output)
        )

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return _fake_predict_pooling


def test_priority_medical_addtest_replay_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        arrange_model6,
        "predict_pooling",
        _fake_predict_pooling_factory(
            order_map={
                "N_CLIN_ADD": 350.0,
                "N_CLIN_2": 360.0,
                "N_SJ_1": 500.0,
                "N_YC_1": 500.0,
                "N_YC_2": 480.0,
                "N_OTHER_1": 500.0,
                "N_OTHER_2": 480.0,
                "N_UNASSIGNED": 100.0,
            },
            output_map={
                "N_CLIN_ADD": 300.0,
                "N_CLIN_2": 320.0,
                "N_SJ_1": 450.0,
                "N_YC_1": 470.0,
                "N_YC_2": 430.0,
                "N_OTHER_1": 460.0,
                "N_OTHER_2": 420.0,
                "N_UNASSIGNED": 90.0,
            },
        ),
    )

    output_path = tmp_path / "priority_medical_addtest_output.csv"
    result_path = arrange_model6.arrange_library(
        REAL_REPLAY_DIR / "replay_priority_medical_addtest.csv",
        mode="arrange",
        output_file=output_path,
    )

    result = _read_csv(result_path)

    clin_add_row = result.loc[result["origrec_key"] == "N_CLIN_ADD"].iloc[0]
    assert float(clin_add_row["lorderdata"]) == 466.666667
    assert float(clin_add_row["lsjnd"]) == 2.3

    first_priority_lane = str(clin_add_row["llaneid"])
    lane_members = set(result.loc[result["llaneid"] == first_priority_lane, "origrec_key"])
    assert lane_members == {"N_CLIN_ADD", "N_CLIN_2", "N_SJ_1"}

    yc_lanes = {
        str(value)
        for value in result.loc[result["origrec_key"].isin(["N_YC_1", "N_YC_2"]), "llaneid"]
        if str(value).strip()
    }
    assert len(yc_lanes) == 1

    unassigned_row = result.loc[result["origrec_key"] == "N_UNASSIGNED"].iloc[0]
    assert int(unassigned_row["aiarrangenumber"]) == 1
    assert str(unassigned_row["llaneid"]).strip() == ""


def test_package_multisplit_atac_replay_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        arrange_model6,
        "predict_pooling",
        _fake_predict_pooling_factory(
            order_map={"P_ATAC_OK": 950.0, "P_ATAC_FAIL": 900.0},
            output_map={"P_ATAC_OK": 880.0, "P_ATAC_FAIL": 820.0},
        ),
    )

    output_path = tmp_path / "package_multisplit_atac_output.csv"
    result_path = arrange_model6.arrange_library(
        REAL_REPLAY_DIR / "replay_package_multisplit_atac.csv",
        mode="arrange",
        output_file=output_path,
    )

    result = _read_csv(result_path)

    ok_rows = result.loc[result["origrec_key"] == "P_ATAC_OK"]
    assert len(ok_rows) == 2
    assert ok_rows["llaneid"].nunique() == 2
    assert set(pd.to_numeric(ok_rows["lsjnd"], errors="coerce")) == {2.0}

    fail_row = result.loc[result["origrec_key"] == "P_ATAC_FAIL"].iloc[0]
    assert str(fail_row["llaneid"]).strip() == ""
    assert int(fail_row["aiarrangenumber"]) == 1


def test_loading_combo_replay_data_hits_expected_concentration_rules() -> None:
    libraries = {
        lib.origrec: lib
        for lib in arrange_model6.load_test_libraries(
            str(REAL_REPLAY_DIR / "replay_loading_combo_packages.csv")
        )
    }

    cases = [
        (
            ["GA_CUST_1", "GA_CUST_2"],
            2.5,
            "special_10x_combo_group_a_customer_2_5",
        ),
        (
            ["GA_NON_1", "GA_NON_2"],
            1.78,
            "special_10x_combo_group_a_non_customer_1_78",
        ),
        (
            ["GB_CUST_1", "GB_CUST_2"],
            2.5,
            "special_10x_combo_group_b_customer_2_5",
        ),
        (
            ["GB_NON_1", "GB_NON_2"],
            1.78,
            "special_10x_combo_group_b_non_customer_1_78",
        ),
    ]

    for origrecs, expected_concentration, expected_rule in cases:
        selected = [libraries[origrec] for origrec in origrecs]
        concentration, rule = arrange_model6._resolve_lane_loading_concentration(selected)
        assert concentration == expected_concentration
        assert rule == expected_rule
