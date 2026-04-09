from pathlib import Path
import sys
from types import SimpleNamespace


PAIJI_ROOT = Path(__file__).resolve().parents[1]
if str(PAIJI_ROOT) not in sys.path:
    sys.path.insert(0, str(PAIJI_ROOT))

from arrange_library.core.preprocessing.library_splitter import LibrarySplitter


def _build_library(mode: str, data_amount: float, index_seq: str = "ATCG;GCTA"):
    return SimpleNamespace(
        origrec="LIB001",
        contract_data_raw=data_amount,
        index_seq=index_seq,
        _lane_sj_mode_raw=mode,
        lane_sj_mode=mode,
        _current_seq_mode_raw=mode,
        current_seq_mode=mode,
        _last_cxms_raw=mode,
        last_cxms=mode,
        seq_scheme="",
        test_no="Novaseq X Plus-PE150",
        package_lane_number="",
        baleno="",
        package_fc_number="",
        lane_id="",
        fc_id="",
        runid="",
    )


def test_mode_1_1_single_index_is_not_split():
    splitter = LibrarySplitter()
    lib = _build_library(mode="1.1", data_amount=150.0)

    processed, split_records = splitter.split_libraries([lib])

    assert splitter._should_split(lib) is False
    assert len(processed) == 1
    assert split_records == []


def test_mode_1_1_multi_index_is_not_split():
    splitter = LibrarySplitter()
    lib = _build_library(mode="1.1", data_amount=400.0, index_seq="ATCG;GCTA,TTAA;GGCC")

    assert splitter._should_split(lib) is False


def test_mode_1_0_legacy_name_is_not_split():
    splitter = LibrarySplitter()
    lib = _build_library(mode="1.0", data_amount=250.0)

    assert splitter._should_split(lib) is False


def test_mode_3_6t_new_single_index_still_splits():
    splitter = LibrarySplitter()
    lib = _build_library(mode="3.6T-NEW", data_amount=120.0)

    assert splitter._should_split(lib) is True


def test_mode_3_6t_new_multi_index_still_splits():
    splitter = LibrarySplitter()
    lib = _build_library(mode="3.6T-NEW", data_amount=400.0, index_seq="ATCG;GCTA,TTAA;GGCC")

    assert splitter._should_split(lib) is True


def test_non_target_mode_is_not_split():
    splitter = LibrarySplitter()
    lib = _build_library(mode="Lane seq", data_amount=400.0, index_seq="ATCG;GCTA,TTAA;GGCC")

    assert splitter._should_split(lib) is False
