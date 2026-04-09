from pathlib import Path
import sys

import pytest


PAIJI_ROOT = Path(__file__).resolve().parents[1]
if str(PAIJI_ROOT) not in sys.path:
    sys.path.insert(0, str(PAIJI_ROOT))

from arrange_library.arrange_library_model6 import _resolve_lane_loading_concentration
from arrange_library.core.scheduling.greedy_lane_scheduler import GreedyLaneScheduler
from arrange_library.models.library_info import EnhancedLibraryInfo


def _build_library(
    origrec: str,
    *,
    sample_type_code: str = "常规RNA文库",
    data_type: str = "其他",
    sample_id: str = "SAMPLE_001",
    sub_project_name: str = "普通项目",
    contract_data_raw: float = 50.0,
    test_no: str = "PE150",
    delete_date=None,
    jjbj: str = "否",
) -> EnhancedLibraryInfo:
    lib = EnhancedLibraryInfo(
        origrec=origrec,
        sample_id=sample_id,
        sample_type_code=sample_type_code,
        data_type=data_type,
        customer_library="否",
        base_type="双",
        number_of_bases=10,
        index_number=1,
        index_seq="ACGTACGT;TGCATGCA",
        add_tests_remark="",
        product_line="N",
        peak_size=300,
        eq_type="Nova X-25B",
        contract_data_raw=float(contract_data_raw),
        test_code=1595,
        test_no=test_no,
        sub_project_name=sub_project_name,
        create_date="2026-04-08",
        delivery_date="2026-04-10",
        lab_type=sample_type_code,
        data_volume_type="常规",
        board_number="B1",
    )
    lib._delete_date_raw = delete_date
    lib.jjbj = jjbj
    return lib


@pytest.fixture(scope="module")
def scheduler():
    return GreedyLaneScheduler()


def test_scattered_mix_priority_orders_clinical_and_sj_then_yc_then_delete_date(
    scheduler: GreedyLaneScheduler,
):
    libraries = [
        _build_library("other5", delete_date=5, contract_data_raw=50.0),
        _build_library("yc", data_type="YC", sample_id="FKDL001", contract_data_raw=40.0),
        _build_library("clinical", data_type="临检", sample_id="FDYE001", contract_data_raw=60.0),
        _build_library("sj", sub_project_name="重点SJ项目", contract_data_raw=55.0),
        _build_library("other2", delete_date=2, contract_data_raw=30.0),
    ]

    ordered = scheduler._sort_remaining_for_scattered_mix_lane(libraries)

    assert [lib.origrec for lib in ordered] == ["clinical", "sj", "yc", "other2", "other5"]


def test_scattered_mix_seed_prefers_same_priority_libraries_even_for_imbalance(
    scheduler: GreedyLaneScheduler,
):
    libraries = [
        _build_library(
            "clinical_seed",
            data_type="临检",
            sample_id="FDYE001",
            contract_data_raw=60.0,
            jjbj="是",
        ),
        _build_library(
            "sj_same_rank",
            sub_project_name="重点SJ项目",
            contract_data_raw=55.0,
            jjbj="是",
        ),
        _build_library(
            "yc_next",
            data_type="YC",
            sample_id="FKDL001",
            contract_data_raw=50.0,
            jjbj="是",
        ),
        _build_library("other_last", delete_date=1, contract_data_raw=45.0, jjbj="是"),
    ]

    ordered = scheduler._sort_remaining_for_lane_seed(libraries, libraries[0])

    assert [lib.origrec for lib in ordered] == [
        "clinical_seed",
        "sj_same_rank",
        "yc_next",
        "other_last",
    ]


@pytest.mark.parametrize(
    ("libraries", "expected_concentration", "expected_rule"),
    [
        (
            [
                _build_library("medical1", sub_project_name="医学委托项目A", contract_data_raw=60.0),
                _build_library("medical2", sub_project_name="医学委托项目B", contract_data_raw=50.0),
            ],
            2.3,
            "medical_commission_over_100g_2_3",
        ),
        (
            [
                _build_library(
                    "atac",
                    sample_type_code="客户-10X ATAC文库",
                    test_no="151+10+24+151",
                )
            ],
            2.0,
            "10_plus_24_atac_2_0",
        ),
        (
            [
                _build_library("group_a_customer_1", sample_type_code="10X转录组-5'文库"),
                _build_library("group_a_customer_2", sample_type_code="客户-10X VDJ文库"),
            ],
            2.5,
            "special_10x_combo_group_a_customer_2_5",
        ),
        (
            [
                _build_library("group_a_non_customer_1", sample_type_code="10X转录组-5'文库"),
                _build_library("group_a_non_customer_2", sample_type_code="10X转录组V(D)J-BCR文库"),
            ],
            1.78,
            "special_10x_combo_group_a_non_customer_1_78",
        ),
        (
            [
                _build_library("group_b_customer_1", sample_type_code="10X转录组-3'文库"),
                _build_library("group_b_customer_2", sample_type_code="客户-10X Visium空间转录组文库"),
            ],
            2.5,
            "special_10x_combo_group_b_customer_2_5",
        ),
        (
            [
                _build_library("group_b_non_customer_1", sample_type_code="10X转录组-3'文库"),
                _build_library("group_b_non_customer_2", sample_type_code="墨卓转录组-3端文库"),
            ],
            1.78,
            "special_10x_combo_group_b_non_customer_1_78",
        ),
    ],
)
def test_explicit_lane_loading_concentration_rules(
    libraries,
    expected_concentration,
    expected_rule,
):
    concentration, rule_name = _resolve_lane_loading_concentration(libraries)

    assert concentration == expected_concentration
    assert rule_name == expected_rule
