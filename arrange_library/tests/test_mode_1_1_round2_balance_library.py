"""
1.1第二轮平衡文库规则测试
创建时间：2026-04-22 18:10:04
更新时间：2026-04-23 11:04:15
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import arrange_library.arrange_library_model6 as arrange_library_model6
import arrange_library.mode_1_1_service as mode_1_1_service
from arrange_library.arrange_library_model6 import (
    _collect_prediction_rows,
    _apply_mode_1_1_round2_balance_rule_to_prediction_df,
    _normalize_text_for_match,
    _materialize_balance_libraries_for_solution,
    _resolve_lane_balance_data_gb,
    load_standardized_csv,
)
from arrange_library.core.data.library_loader import load_libraries_from_csv
from arrange_library.core.scheduling.scheduling_types import LaneAssignment
from arrange_library.models.library_info import EnhancedLibraryInfo, MachineType


def _make_round2_lane(
    *,
    contract_data_raw: float = 90.0,
    last_phix: float = 0.1,
    output_rate: float = 100.0,
    order_data_amount: float | None = None,
) -> LaneAssignment:
    lib = EnhancedLibraryInfo(
        origrec="ROUND2_LIB_001",
        sample_id="S001",
        sample_type_code="WES",
        data_type="其他",
        customer_library="否",
        base_type="双",
        number_of_bases=10,
        index_number=1,
        contract_data_raw=contract_data_raw,
        index_seq="AACCGGTT;TTGGCCAA",
        add_tests_remark="-",
        product_line="S",
        eq_type="Nova X-25B",
        peak_size=350,
        test_code=1595,
        test_no="Novaseq X Plus-PE150",
        sub_project_name="TEST_PROJECT",
        create_date="2026-04-22",
        delivery_date="2026-04-30",
        lab_type="诺禾-WES文库",
        data_volume_type="小数量",
        board_number="BN001",
    )
    lib._wkdept_raw = "天津科技服务实验室"
    lib._last_phix_raw = last_phix
    lib.output_rate = output_rate
    if order_data_amount is not None:
        lib.order_data_amount = order_data_amount

    lane = LaneAssignment(
        lane_id="M11R2_001",
        machine_id="M_M11R2_001",
        machine_type=MachineType.NOVA_X_25B,
        libraries=[lib],
        total_data_gb=contract_data_raw,
        metadata={
            "selected_seq_mode": "1.1",
            "selected_round_label": "1.1第二轮",
            "skip_strict_validation": True,
        },
    )
    lane.calculate_metrics()
    return lane


def test_resolve_lane_balance_data_uses_wklastphix_for_mode_1_1_round2():
    lane = _make_round2_lane(contract_data_raw=90.0, last_phix=0.1, output_rate=50.0)

    # 第二轮按lane总下单量计算：下单量=90/0.5=180G，平衡文库占最终总量10%，应补 180*0.1/0.9=20G
    assert _resolve_lane_balance_data_gb(lane) == 20.0


def test_resolve_lane_balance_data_uses_total_ratio_formula_for_2000g_and_5pct():
    lane = _make_round2_lane(contract_data_raw=2000.0, last_phix=0.05, output_rate=50.0)

    # 第二轮按lane总下单量计算：下单量=2000/0.5=4000G，x / (4000 + x) = 0.05，因此 x = 210.526...
    assert _resolve_lane_balance_data_gb(lane) == 210.526


def test_materialize_balance_library_for_mode_1_1_round2_uses_historical_phix_ratio():
    lane = _make_round2_lane(contract_data_raw=90.0, last_phix=0.1, output_rate=50.0)
    solution = SimpleNamespace(lane_assignments=[lane], unassigned_libraries=[])

    stats = _materialize_balance_libraries_for_solution(solution)

    assert stats["required_lanes"] == 1
    assert stats["success_lanes"] == 1
    assert len(lane.libraries) == 2
    balance_libs = [lib for lib in lane.libraries if getattr(lib, "_is_ai_balance_library", False)]
    assert len(balance_libs) == 1
    assert float(balance_libs[0].contract_data_raw) == 20.0
    assert lane.metadata["wkbalancedata"] == 20.0
    assert lane.metadata["materialized_balance_library"] is True


def test_materialize_balance_library_for_mode_1_1_round2_skips_conflicting_template(monkeypatch: pytest.MonkeyPatch):
    lane = _make_round2_lane(contract_data_raw=90.0, last_phix=0.1)
    solution = SimpleNamespace(lane_assignments=[lane], unassigned_libraries=[])
    template_key = (
        _normalize_text_for_match("天津科技服务实验室"),
        _normalize_text_for_match("Novaseq X Plus-PE150"),
    )

    monkeypatch.setattr(
        arrange_library_model6,
        "_load_balance_library_templates",
        lambda: {
            template_key: [
                {
                    "wksampleid": "phix",
                    "wkdept": "天津科技服务实验室",
                    "wktestno": "Novaseq X Plus-PE150",
                    "wkindexseq": "AACCGGTT;TTGGCCAA",
                    "_template_order": 0,
                },
                {
                    "wksampleid": "BAL_SAFE",
                    "wkdept": "天津科技服务实验室",
                    "wktestno": "Novaseq X Plus-PE150",
                    "wkindexseq": "TTTTAAAA;CCCCGGGG",
                    "_template_order": 1,
                },
            ]
        },
    )

    stats = _materialize_balance_libraries_for_solution(solution)

    assert stats["required_lanes"] == 1
    assert stats["success_lanes"] == 1
    balance_libs = [lib for lib in lane.libraries if getattr(lib, "_is_ai_balance_library", False)]
    assert len(balance_libs) == 1
    assert balance_libs[0].sample_id == "BAL_SAFE"


def test_materialize_balance_library_for_mode_1_1_round2_rejects_only_conflicting_template(
    monkeypatch: pytest.MonkeyPatch,
):
    lane = _make_round2_lane(contract_data_raw=90.0, last_phix=0.1)
    solution = SimpleNamespace(lane_assignments=[lane], unassigned_libraries=[])
    template_key = (
        _normalize_text_for_match("天津科技服务实验室"),
        _normalize_text_for_match("Novaseq X Plus-PE150"),
    )

    monkeypatch.setattr(
        arrange_library_model6,
        "_load_balance_library_templates",
        lambda: {
            template_key: [
                {
                    "wksampleid": "phix",
                    "wkdept": "天津科技服务实验室",
                    "wktestno": "Novaseq X Plus-PE150",
                    "wkindexseq": "AACCGGTT;TTGGCCAA",
                    "_template_order": 0,
                }
            ]
        },
    )

    stats = _materialize_balance_libraries_for_solution(solution)

    assert stats["required_lanes"] == 1
    assert stats["success_lanes"] == 0
    assert len(lane.libraries) == 1
    assert lane.metadata.get("materialized_balance_library") is not True


def test_apply_mode_1_1_round2_balance_rule_uses_lane_total_order_formula():
    prediction_df = pd.DataFrame(
        [
            {
                "llaneid": "LANE_R2_001",
                "laneround": "1.1第二轮",
                "wkcontractdata": 100.0,
                "wkoutputrate": 80.0,
                "lorderdata": 1.0,
                "resolved_round2_balance_ratio": 0.1,
                "_is_ai_balance_library": False,
            },
            {
                "llaneid": "LANE_R2_001",
                "laneround": "1.1第二轮",
                "wkcontractdata": 50.0,
                "wkoutputrate": 50.0,
                "lorderdata": 2.0,
                "resolved_round2_balance_ratio": 0.1,
                "_is_ai_balance_library": False,
            },
            {
                "llaneid": "LANE_R2_001",
                "laneround": "1.1第二轮",
                "wkcontractdata": 0.0,
                "wkoutputrate": None,
                "lorderdata": None,
                "predicted_lorderdata": None,
                "resolved_round2_balance_ratio": 0.1,
                "_is_ai_balance_library": True,
            },
        ]
    )

    result_df = _apply_mode_1_1_round2_balance_rule_to_prediction_df(prediction_df)

    balance_row = result_df.loc[result_df["_is_ai_balance_library"]].iloc[0]
    # 普通文库下单量分别为 100/0.8=125、50/0.5=100，非平衡总下单量 225。
    # 平衡文库占最终lane总下单量 10%，则最终lane总下单量 = 225 / 0.9 = 250，平衡文库=25。
    assert balance_row["wkcontractdata"] == pytest.approx(25.0, rel=0, abs=1e-6)
    assert balance_row["lorderdata"] == pytest.approx(25.0, rel=0, abs=1e-6)
    assert "resolved_round2_balance_ratio" not in result_df.columns


def test_apply_mode_1_1_round2_balance_rule_does_not_fallback_to_historical_outrate():
    prediction_df = pd.DataFrame(
        [
            {
                "llaneid": "LANE_R2_002",
                "laneround": "1.1第二轮",
                "wkcontractdata": 100.0,
                "wkoutputrate": None,
                "lorderdata": None,
                "predicted_lorderdata": None,
                "wklastoutrate": 80.0,
                "wklastoutput": 80.0,
                "wklastorderdata": 100.0,
                "resolved_round2_balance_ratio": 0.1,
                "_is_ai_balance_library": False,
            },
            {
                "llaneid": "LANE_R2_002",
                "laneround": "1.1第二轮",
                "wkcontractdata": 0.0,
                "wkoutputrate": None,
                "lorderdata": None,
                "predicted_lorderdata": None,
                "resolved_round2_balance_ratio": 0.1,
                "_is_ai_balance_library": True,
            },
        ]
    )

    result_df = _apply_mode_1_1_round2_balance_rule_to_prediction_df(prediction_df)

    balance_row = result_df.loc[result_df["_is_ai_balance_library"]].iloc[0]
    assert balance_row["wkcontractdata"] == pytest.approx(0.0, rel=0, abs=1e-6)
    assert pd.isna(balance_row["lorderdata"])


def test_load_standardized_csv_preserves_wklastphix(tmp_path: Path):
    csv_path = tmp_path / "mode_1_1_standardized.csv"
    pd.DataFrame(
        [
            {
                "wkorigrec": "STD_LIB_001",
                "wksampleid": "S001",
                "wksampletype": "WES",
                "wkdatatype": "其他",
                "CUSTOMERLIBRARY": "否",
                "BASETYPE": "双",
                "NUMBEROFBASES": 10,
                "INDEXNUMBER": 1,
                "wkindexseq": "AACCGGTT;TTGGCCAA",
                "wkaddtestsremark": "-",
                "wkproductline": "S",
                "wkeqtype": "Nova X-25B",
                "wkcontractdata": 100.0,
                "wkpeaksize": 350,
                "wktestno": "Novaseq X Plus-PE150",
                "wksubprojectname": "TEST_PROJECT",
                "wkcreatedate": "2026-04-22",
                "wkdeliverydate": "2026-04-30",
                "LABTYPE": "诺禾-WES文库",
                "wkdataunit": "小数量",
                "wkboardnumber": "BN001",
                "wkdept": "天津科技服务实验室",
                "wklastphix": 0.12,
            }
        ]
    ).to_csv(csv_path, index=False)

    libraries = load_standardized_csv(str(csv_path))

    assert len(libraries) == 1
    assert getattr(libraries[0], "_last_phix_raw", None) == pytest.approx(0.12, rel=0, abs=1e-6)


def test_load_libraries_from_csv_preserves_wklastphix(tmp_path: Path):
    csv_path = tmp_path / "mode_1_1_loader.csv"
    pd.DataFrame(
        [
            {
                "wkorigrec": "LOAD_LIB_001",
                "wksampleid": "S001",
                "wksampletype": "WES",
                "wkdatatype": "其他",
                "CUSTOMERLIBRARY": "否",
                "BASETYPE": "双",
                "NUMBEROFBASES": 10,
                "INDEXNUMBER": 1,
                "wkindexseq": "AACCGGTT;TTGGCCAA",
                "wkaddtestsremark": "-",
                "wkproductline": "S",
                "wkeqtype": "Nova X-25B",
                "wkcontractdata": 100.0,
                "wkpeaksize": 350,
                "wktestno": "Novaseq X Plus-PE150",
                "wksubprojectname": "TEST_PROJECT",
                "wkcreatedate": "2026-04-22",
                "wkdeliverydate": "2026-04-30",
                "LABTYPE": "诺禾-WES文库",
                "wkdataunit": "小数量",
                "wkboardnumber": "BN001",
                "wkdept": "天津科技服务实验室",
                "wklastphix": 0.08,
            }
        ]
    ).to_csv(csv_path, index=False)

    libraries = load_libraries_from_csv(csv_path, enable_remark_recognition=False, allow_missing=False)

    assert len(libraries) == 1
    assert getattr(libraries[0], "_last_phix_raw", None) == pytest.approx(0.08, rel=0, abs=1e-6)


def test_collect_prediction_rows_preserves_wklastphix_in_output():
    lane = _make_round2_lane(contract_data_raw=90.0, last_phix=0.15)

    prediction_df = _collect_prediction_rows([lane], loutput_by_origrec={}, tag="TEST")

    assert len(prediction_df) == 1
    assert prediction_df.loc[0, "wklastphix"] == pytest.approx(0.15, rel=0, abs=1e-6)


def test_run_mode_1_1_round2_export_materializes_balance_libraries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    input_path = tmp_path / "round2_input.csv"
    output_path = tmp_path / "round2_output.csv"
    pd.DataFrame([{"origrec": "LIB001", "loutput": 1.0}]).to_csv(input_path, index=False)

    lane = _make_round2_lane(contract_data_raw=90.0, last_phix=0.1, output_rate=50.0)

    class _FakeIdentificationResult:
        total_candidates = 1
        candidate_groups = [["group"]]

    class _FakeHandler:
        def __init__(self, config):
            self.config = config

        def identify_round2_candidates(self, libraries):
            return _FakeIdentificationResult()

        def schedule_round2(self, candidate_groups):
            return SimpleNamespace(lanes=[lane])

    materialize_called = {"value": False}

    def _fake_materialize(solution):
        materialize_called["value"] = True
        return {"required_lanes": 1, "success_lanes": 1}

    def _fake_prepare_mode_1_1_libraries(*, data_file=None, libraries=None):
        return SimpleNamespace(ai_schedulable_libraries=[object()])

    def _fake_collect_detail_output_libraries(solution):
        return [SimpleNamespace(_source_origrec_key="LIB001", origrec="LIB001")]

    def _fake_collect_prediction_rows(lanes, loutput_by_origrec, tag):
        return pd.DataFrame([{"origrec_key": "LIB001", "detail_row_key": "LIB001"}])

    def _fake_build_detail_output(
        *,
        df_raw,
        pred_df,
        output_path,
        ai_schedulable_keys,
        lanes_with_split,
        detail_libraries,
    ):
        pd.DataFrame([{"origrec_key": "LIB001", "detail_row_key": "LIB001"}]).to_csv(output_path, index=False)

    def _fake_run_prediction_delivery(*, input_data, output_path):
        return pd.DataFrame(
            [
                {
                    "origrec_key": "LIB001",
                    "detail_row_key": "LIB001",
                    "llaneid": "M11R2_001",
                    "laneround": "1.1第二轮",
                }
            ]
        )

    monkeypatch.setattr(mode_1_1_service, "_resolve_mode_1_1_config", lambda config=None: {})
    monkeypatch.setattr(mode_1_1_service, "prepare_mode_1_1_libraries", _fake_prepare_mode_1_1_libraries)
    monkeypatch.setattr(mode_1_1_service, "Mode11Round2Handler", _FakeHandler)
    monkeypatch.setattr(mode_1_1_service, "_materialize_balance_libraries_for_solution", _fake_materialize)
    def _fake_read_csv_with_encoding_fallback(path):
        path_obj = Path(path)
        if path_obj.exists():
            return pd.read_csv(path_obj)
        return pd.DataFrame([{"origrec": "LIB001", "origrec_key": "LIB001", "loutput": 1.0}])

    monkeypatch.setattr(
        mode_1_1_service,
        "_read_csv_with_encoding_fallback",
        _fake_read_csv_with_encoding_fallback,
    )
    monkeypatch.setattr(mode_1_1_service, "_collect_detail_output_libraries", _fake_collect_detail_output_libraries)
    monkeypatch.setattr(mode_1_1_service, "_collect_prediction_rows", _fake_collect_prediction_rows)
    monkeypatch.setattr(mode_1_1_service, "_collect_lanes_with_split", lambda lanes: set())
    monkeypatch.setattr(mode_1_1_service, "_build_detail_output", _fake_build_detail_output)
    monkeypatch.setattr(mode_1_1_service, "_run_prediction_delivery", _fake_run_prediction_delivery)

    result = mode_1_1_service.run_mode_1_1_round2_export(
        data_file=input_path,
        output_file=output_path,
    )

    assert materialize_called["value"] is True
    assert result.exported_lane_count == 1
    assert output_path.exists()


def test_run_mode_1_1_round2_export_keeps_appended_balance_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    input_path = tmp_path / "round2_input.csv"
    output_path = tmp_path / "round2_output.csv"
    pd.DataFrame([{"origrec": "LIB001", "loutput": 1.0}]).to_csv(input_path, index=False)

    class _FakeIdentificationResult:
        total_candidates = 1
        candidate_groups = [["group"]]

    class _FakeHandler:
        def __init__(self, config):
            self.config = config

        def identify_round2_candidates(self, libraries):
            return _FakeIdentificationResult()

        def schedule_round2(self, candidate_groups):
            return SimpleNamespace(lanes=[object()])

    def _fake_prepare_mode_1_1_libraries(*, data_file=None, libraries=None):
        return SimpleNamespace(ai_schedulable_libraries=[object()])

    def _fake_collect_detail_output_libraries(solution):
        return [
            SimpleNamespace(_source_origrec_key="LIB001", origrec="LIB001"),
            SimpleNamespace(_source_origrec_key="BAL_NEW_001", origrec="BAL_NEW_001"),
        ]

    def _fake_collect_prediction_rows(lanes, loutput_by_origrec, tag):
        return pd.DataFrame(
            [
                {"origrec_key": "LIB001", "detail_row_key": "LIB001"},
                {"origrec_key": "BAL_NEW_001", "detail_row_key": "BAL_NEW_001"},
            ]
        )

    def _fake_build_detail_output(
        *,
        df_raw,
        pred_df,
        output_path,
        ai_schedulable_keys,
        lanes_with_split,
        detail_libraries,
    ):
        pd.DataFrame(
            [
                    {"origrec": "LIB001", "origrec_key": "LIB001", "detail_row_key": "LIB001", "llaneid": "M11R2_001"},
                {
                        "origrec": "BAL_NEW_001",
                    "origrec_key": "BAL_NEW_001",
                    "detail_row_key": "BAL_NEW_001",
                    "llaneid": "M11R2_001",
                    "wkisaddbalance": "1",
                    "wkbalancedata": 20.0,
                    "wksampleid": "phix",
                },
            ]
        ).to_csv(output_path, index=False)

    def _fake_run_prediction_delivery(*, input_data, output_path):
        if isinstance(input_data, pd.DataFrame):
            return input_data.copy()
        return pd.read_csv(input_data)

    monkeypatch.setattr(mode_1_1_service, "_resolve_mode_1_1_config", lambda config=None: {})
    monkeypatch.setattr(mode_1_1_service, "prepare_mode_1_1_libraries", _fake_prepare_mode_1_1_libraries)
    monkeypatch.setattr(mode_1_1_service, "Mode11Round2Handler", _FakeHandler)
    monkeypatch.setattr(
        mode_1_1_service,
        "_materialize_balance_libraries_for_solution",
        lambda solution: {"required_lanes": 1, "success_lanes": 1},
    )
    def _fake_read_csv_with_encoding_fallback(path):
        path_obj = Path(path)
        if path_obj.exists():
            return pd.read_csv(path_obj)
        return pd.DataFrame([{"origrec": "LIB001", "origrec_key": "LIB001", "loutput": 1.0}])

    monkeypatch.setattr(
        mode_1_1_service,
        "_read_csv_with_encoding_fallback",
        _fake_read_csv_with_encoding_fallback,
    )
    monkeypatch.setattr(mode_1_1_service, "_collect_detail_output_libraries", _fake_collect_detail_output_libraries)
    monkeypatch.setattr(mode_1_1_service, "_collect_prediction_rows", _fake_collect_prediction_rows)
    monkeypatch.setattr(mode_1_1_service, "_collect_lanes_with_split", lambda lanes: set())
    monkeypatch.setattr(mode_1_1_service, "_build_detail_output", _fake_build_detail_output)
    monkeypatch.setattr(mode_1_1_service, "_run_prediction_delivery", _fake_run_prediction_delivery)

    mode_1_1_service.run_mode_1_1_round2_export(
        data_file=input_path,
        output_file=output_path,
    )

    written_df = pd.read_csv(output_path)
    assert "BAL_NEW_001" in written_df["origrec"].astype(str).tolist()
    balance_row = written_df.loc[written_df["origrec"].astype(str) == "BAL_NEW_001"].iloc[0]
    assert str(balance_row["wkisaddbalance"]) in {"1", "1.0"}
    assert float(balance_row["wkbalancedata"]) == pytest.approx(20.0, rel=0, abs=1e-6)
