import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from arrange_library.core.preprocessing.library_splitter import LibrarySplitter
from arrange_library.models.library_info import EnhancedLibraryInfo


def _make_lib(
    origrec: str = "LIB_SPLIT_001",
    contract_data_raw: float = 200.0,
    index_seq: str = "AACCGGTT;TTGGCCAA",
    test_code: int = 1595,
    test_no: str = "NovaSeq X Plus-PE150",
    seq_scheme: str = "",
) -> EnhancedLibraryInfo:
    lib = EnhancedLibraryInfo(
        origrec=origrec,
        sample_id="S001",
        sample_type_code="WES",
        data_type="其他",
        customer_library="否",
        base_type="双",
        number_of_bases=10,
        index_number=1,
        index_seq=index_seq,
        add_tests_remark="-",
        product_line="S",
        peak_size=350,
        eq_type="Nova X-25B",
        contract_data_raw=contract_data_raw,
        test_code=test_code,
        test_no=test_no,
        sub_project_name="TEST_PROJECT",
        create_date="2026-04-15",
        delivery_date="2026-04-30",
        lab_type="诺禾-WES文库",
        data_volume_type="小数量",
        board_number="BN001",
        seq_scheme=seq_scheme,
    )
    return lib


def test_splitter_ignores_last_cxms_for_36t_single_index():
    splitter = LibrarySplitter()
    lib = _make_lib(contract_data_raw=201.0)
    lib._last_cxms_raw = "1.0"
    lib.last_cxms = "1.0"

    rule_label, threshold = splitter._resolve_split_rule(lib)

    assert rule_label == "3.6t_new_single_index_pair"
    assert threshold == 200.0
    assert splitter._should_split(lib) is True


def test_splitter_infers_36t_from_test_no_when_test_code_missing():
    splitter = LibrarySplitter()
    lib = _make_lib(
        origrec="LIB_SPLIT_001A",
        contract_data_raw=201.0,
        test_code=0,
        test_no="Novaseq X Plus-PE150",
        seq_scheme="PE150",
    )
    lib._last_cxms_raw = "1.0"
    lib.last_cxms = "1.0"

    rule_label, threshold = splitter._resolve_split_rule(lib)

    assert rule_label == "3.6t_new_single_index_pair"
    assert threshold == 200.0
    assert splitter._should_split(lib) is True


def test_splitter_ignores_last_cxms_for_36t_multi_index():
    splitter = LibrarySplitter()
    lib = _make_lib(
        origrec="LIB_SPLIT_002",
        contract_data_raw=350.0,
        index_seq="AACCGGTT;TTGGCCAA,CCAATTGG;GGTTAACC",
    )
    lib._last_cxms_raw = "1.0"
    lib.last_cxms = "1.0"

    rule_label, threshold = splitter._resolve_split_rule(lib)

    assert rule_label == "3.6t_new_multi_index_pair"
    assert threshold == 300.0
    assert splitter._should_split(lib) is True


def test_splitter_keeps_current_1_1_mode_unsplit():
    splitter = LibrarySplitter()
    lib = _make_lib(origrec="LIB_SPLIT_003", contract_data_raw=200.0)
    lib._current_seq_mode_raw = "1.1"
    lib._last_cxms_raw = "3.6T-NEW"

    rule_label, threshold = splitter._resolve_split_rule(lib)

    assert rule_label == "mode_1.1_disabled_compat_1.0"
    assert threshold == float("inf")
    assert splitter._should_split(lib) is False


def test_splitter_evenly_splits_data_fields_and_generates_new_child_bids():
    splitter = LibrarySplitter()
    lib = _make_lib(origrec="LIB_SPLIT_OUTPUT", contract_data_raw=600.0)
    lib.wkaidbid = "ORIGINAL_BID"
    lib.aidbid = "ORIGINAL_BID"
    lib.single_index_data = 90.0
    lib.ten_bp_data = 30.0

    fragments = splitter._perform_split(lib)

    assert len(fragments) == 3
    assert {fragment.contract_data_raw for fragment in fragments} == {200.0}
    assert {fragment.single_index_data for fragment in fragments} == {30.0}
    assert {fragment.ten_bp_data for fragment in fragments} == {10.0}
    assert {fragment.wkissplit for fragment in fragments} == {"yes"}
    assert {fragment._split_source_bid for fragment in fragments} == {"ORIGINAL_BID"}
    child_bids = {fragment.wkaidbid for fragment in fragments}
    assert len(child_bids) == 3
    assert "ORIGINAL_BID" not in child_bids


def test_single_index_split_uses_strict_200g_boundary_and_ceiling_count():
    splitter = LibrarySplitter()
    boundary_lib = _make_lib(origrec="LIB_SPLIT_200", contract_data_raw=200.0)
    over_boundary_lib = _make_lib(origrec="LIB_SPLIT_401", contract_data_raw=401.0)

    assert splitter._should_split(boundary_lib) is False
    assert splitter._perform_split(boundary_lib) == [boundary_lib]

    fragments = splitter._perform_split(over_boundary_lib)
    assert len(fragments) == 3
    assert all(abs(fragment.contract_data_raw - (401.0 / 3.0)) < 1e-9 for fragment in fragments)
