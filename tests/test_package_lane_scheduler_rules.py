from pathlib import Path
import sys
from types import SimpleNamespace


PAIJI_ROOT = Path(__file__).resolve().parents[1]
if str(PAIJI_ROOT) not in sys.path:
    sys.path.insert(0, str(PAIJI_ROOT))

from arrange_library.core.scheduling.package_lane_scheduler import PackageLaneScheduler
from arrange_library.models.library_info import EnhancedLibraryInfo


def _build_package_lane_library(
    origrec: str,
    package_lane_number: str,
    contract_data_raw: float,
    index_seq: str,
) -> EnhancedLibraryInfo:
    lib = EnhancedLibraryInfo(
        origrec=origrec,
        sample_id=f"SAMPLE_{origrec}",
        sample_type_code="常规RNA文库",
        data_type="其他",
        customer_library="否",
        base_type="双",
        number_of_bases=10,
        index_number=1,
        index_seq=index_seq,
        add_tests_remark="",
        product_line="N",
        peak_size=300,
        eq_type="Nova X-25B",
        contract_data_raw=float(contract_data_raw),
        test_code=1595,
        test_no="PE150",
        sub_project_name="普通项目",
        create_date="2026-04-08",
        delivery_date="2026-04-10",
        lab_type="常规RNA文库",
        data_volume_type="常规",
        board_number="B1",
    )
    lib.package_lane_number = package_lane_number
    lib.baleno = package_lane_number
    lib.is_package_lane = "是"
    return lib


def _build_scheduler() -> PackageLaneScheduler:
    return PackageLaneScheduler(pooling_optimizer=SimpleNamespace(enabled=False))


def test_package_lane_requires_at_least_five_index_pairs():
    scheduler = _build_scheduler()
    libraries = [
        _build_package_lane_library("LIB1", "PKG1", 250.0, "ACGTACGT;TGCATGCA"),
        _build_package_lane_library("LIB2", "PKG1", 250.0, "AAAACCCC;GGGGTTTT"),
        _build_package_lane_library("LIB3", "PKG1", 250.0, "TTTTAAAA;CCCCGGGG"),
        _build_package_lane_library("LIB4", "PKG1", 250.0, "AGCTTCGA;TCGAAGCT"),
    ]

    result = scheduler.schedule(libraries)

    assert result.total_lanes == 0
    assert "package_lane_PKG1" in result.failed_packages
    assert "Index对数不足" in result.failed_packages["package_lane_PKG1"]
    assert len(result.remaining_libraries) == 4


def test_package_lane_rejects_duplicate_indexes():
    scheduler = _build_scheduler()
    libraries = [
        _build_package_lane_library("LIB1", "PKG1", 200.0, "ACGTACGT;TGCATGCA"),
        _build_package_lane_library("LIB2", "PKG1", 200.0, "ACGTACGT;TGCATGCA"),
        _build_package_lane_library("LIB3", "PKG1", 200.0, "TTTTAAAA;CCCCGGGG"),
        _build_package_lane_library("LIB4", "PKG1", 200.0, "AGCTTCGA;TCGAAGCT"),
        _build_package_lane_library("LIB5", "PKG1", 200.0, "CATGCATG;GTACGTAC"),
    ]

    result = scheduler.schedule(libraries)

    assert result.total_lanes == 0
    assert "package_lane_PKG1" in result.failed_packages
    assert "Index冲突" in result.failed_packages["package_lane_PKG1"]


def test_package_lane_requires_contract_volume_within_1000g_tolerance():
    scheduler = _build_scheduler()
    libraries = [
        _build_package_lane_library("LIB1", "PKG1", 199.99, "ACGTACGT;TGCATGCA"),
        _build_package_lane_library("LIB2", "PKG1", 199.99, "AAAACCCC;GGGGTTTT"),
        _build_package_lane_library("LIB3", "PKG1", 199.99, "TTTTAAAA;CCCCGGGG"),
        _build_package_lane_library("LIB4", "PKG1", 199.99, "AGCTTCGA;TCGAAGCT"),
        _build_package_lane_library("LIB5", "PKG1", 199.99, "CATGCATG;GTACGTAC"),
    ]

    result = scheduler.schedule(libraries)

    assert result.total_lanes == 0
    assert "package_lane_PKG1" in result.failed_packages
    assert "数据量" in result.failed_packages["package_lane_PKG1"]


def test_multi_package_lane_numbers_split_into_different_lanes_when_targets_succeed():
    scheduler = _build_scheduler()
    source_library = _build_package_lane_library(
        "SOURCE_OK",
        "PKG_A,PKG_B",
        2000.0,
        "ACGTACGT;TGCATGCA,AAAACCCC;GGGGTTTT,TTTTAAAA;CCCCGGGG,AGCTTCGA;TCGAAGCT,CATGCATG;GTACGTAC",
    )

    result = scheduler.schedule([source_library])

    lane_results = [lane for run in result.runs for lane in run.lanes]
    package_ids = sorted(lane.package_id for lane in lane_results)
    lane_ids = {lane.lane_id for lane in lane_results}

    assert result.total_lanes == 2
    assert result.failed_packages == {}
    assert package_ids == ["PKG_A", "PKG_B"]
    assert len(lane_ids) == 2
    assert all(len(lane.libraries) == 1 for lane in lane_results)
    assert all(lane.total_data_gb == 1000.0 for lane in lane_results)


def test_multi_package_lane_numbers_roll_back_when_any_target_fails():
    scheduler = _build_scheduler()
    source_library = _build_package_lane_library(
        "SOURCE_FAIL",
        "PKG_A,PKG_B",
        1800.0,
        "ACGTACGT;TGCATGCA,AAAACCCC;GGGGTTTT,TTTTAAAA;CCCCGGGG,AGCTTCGA;TCGAAGCT,CATGCATG;GTACGTAC",
    )

    result = scheduler.schedule([source_library])

    assert result.total_lanes == 0
    assert len(result.remaining_libraries) == 1

    rolled_back_library = result.remaining_libraries[0]
    assert rolled_back_library.origrec == "SOURCE_FAIL"
    assert rolled_back_library.package_lane_number == "PKG_A,PKG_B"
    assert rolled_back_library.split_status == "rolled_back"
    assert any("不拆分且不排机处理" in reason for reason in result.failed_packages.values())
