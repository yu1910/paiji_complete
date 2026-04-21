import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from arrange_library.core.scheduling.greedy_lane_scheduler import (
    GreedyLaneConfig,
    GreedyLaneScheduler,
)
from arrange_library.models.library_info import EnhancedLibraryInfo


def _make_lib(
    *,
    origrec: str,
    sample_type_code: str,
    contract_data_raw: float,
    peak_size: int,
) -> EnhancedLibraryInfo:
    lib = EnhancedLibraryInfo(
        origrec=origrec,
        sample_id=f"S_{origrec}",
        sample_type_code=sample_type_code,
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
        peak_size=peak_size,
        test_code=1595,
        test_no="Novaseq X Plus-PE150",
        sub_project_name="TEST_PROJECT",
        create_date="2026-04-17",
        delivery_date="2026-04-30",
        lab_type="诺禾-WES文库",
        data_volume_type="小数量",
        board_number="BN001",
    )
    lib._current_seq_mode_raw = "1.1"
    return lib


def _build_scheduler() -> GreedyLaneScheduler:
    return GreedyLaneScheduler(
        GreedyLaneConfig(
            use_machine_config=False,
            enable_rule_checker=False,
        )
    )


def test_sort_libraries_prefers_mode_1_1_seed_rank():
    scheduler = _build_scheduler()
    lib_seed = _make_lib(
        origrec="SEED",
        sample_type_code="VIP-真核普通转录组文库",
        contract_data_raw=6.0,
        peak_size=440,
    )
    lib_other = _make_lib(
        origrec="OTHER",
        sample_type_code="客户-PCR产物",
        contract_data_raw=12.0,
        peak_size=440,
    )
    lib_seed._mode_1_1_seed_rank = 0
    lib_seed._mode_1_1_seed_group = "rna_cluster"
    lib_other._mode_1_1_seed_rank = 2
    lib_other._mode_1_1_seed_group = "scatter_cluster"

    sorted_libs = scheduler._sort_libraries([lib_other, lib_seed])

    assert [lib.origrec for lib in sorted_libs] == ["SEED", "OTHER"]


def test_sort_remaining_for_lane_seed_prefers_same_mode_1_1_seed_group():
    scheduler = _build_scheduler()
    seed_lib = _make_lib(
        origrec="SEED",
        sample_type_code="VIP-真核普通转录组文库",
        contract_data_raw=6.0,
        peak_size=440,
    )
    same_group = _make_lib(
        origrec="SAME_GROUP",
        sample_type_code="VIP-真核普通转录组文库",
        contract_data_raw=5.5,
        peak_size=440,
    )
    other_group = _make_lib(
        origrec="OTHER_GROUP",
        sample_type_code="DNA小片段文库",
        contract_data_raw=10.0,
        peak_size=460,
    )
    seed_lib._mode_1_1_seed_rank = 0
    seed_lib._mode_1_1_seed_group = "rna_cluster"
    same_group._mode_1_1_seed_rank = 0
    same_group._mode_1_1_seed_group = "rna_cluster"
    other_group._mode_1_1_seed_rank = 1
    other_group._mode_1_1_seed_group = "wgs_cluster"

    sorted_candidates = scheduler._sort_remaining_for_lane_seed(
        [other_group, same_group],
        seed_lib,
    )

    assert [lib.origrec for lib in sorted_candidates] == ["SAME_GROUP", "OTHER_GROUP"]
