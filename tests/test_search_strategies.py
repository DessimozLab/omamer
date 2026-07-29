import numba
import numpy as np
import pytest

from omamer.alphabets import Alphabet, get_transform
from omamer.merge_search import (
    BetaBinomialModel,
    FAMILY_MODEL_BETA_BINOMIAL,
    FAMILY_MODEL_BINOMIAL,
    MergeSearch,
    SEARCH_SEQUENCE,
    SEARCH_SEQUENCE_THEN_STRUCTURE,
    SEARCH_STRUCTURE,
    SearchConfig,
    SearchDatabase,
    SearchIndex,
    SequenceBatch,
    resolve_family_model,
    resolve_search_mode,
)


@pytest.mark.parametrize(
    "mode,has_sequence,has_structure,expected",
    [
        ("auto", True, False, SEARCH_SEQUENCE),
        ("auto", False, True, SEARCH_STRUCTURE),
        ("auto", True, True, SEARCH_SEQUENCE_THEN_STRUCTURE),
        ("seq", True, True, SEARCH_SEQUENCE),
        ("ss", True, True, SEARCH_STRUCTURE),
        ("sqs", True, True, SEARCH_SEQUENCE_THEN_STRUCTURE),
    ],
)
def test_resolve_search_mode(
    mode,
    has_sequence,
    has_structure,
    expected,
):
    assert (
        resolve_search_mode(mode, has_sequence, has_structure)
        == expected
    )


def test_resolve_search_mode_rejects_unavailable_input():
    with pytest.raises(ValueError, match="requires matching"):
        resolve_search_mode("ss", True, False)


def test_resolve_family_model():
    assert (
        resolve_family_model("auto", False)
        == FAMILY_MODEL_BINOMIAL
    )
    assert (
        resolve_family_model("auto", True)
        == FAMILY_MODEL_BETA_BINOMIAL
    )
    assert (
        resolve_family_model("binomial", True)
        == FAMILY_MODEL_BINOMIAL
    )
    with pytest.raises(ValueError, match="requires beta-binomial"):
        resolve_family_model("bbinom", False)


def test_lookup_compiles_and_dispatches_search_strategies():
    family_dtype = np.dtype(
        [
            ("HOGoff", np.uint32),
            ("HOGnum", np.uint32),
            ("LevelOff", np.uint32),
            ("LevelNum", np.uint32),
        ]
    )
    hog_dtype = np.dtype(
        [("FamOff", np.uint32), ("ParentOff", np.int32)]
    )
    alphabet = Alphabet(21)
    k = 2
    database = SearchDatabase(
        get_transform(k, alphabet.DIGITS_AA),
        k,
        alphabet.DIGITS_AA_LOOKUP,
        np.array([(0, 1, 0, 0)], dtype=family_dtype),
        np.array([(0, -1)], dtype=hog_dtype),
        np.array([0, 1], dtype=np.uint32),
    )
    empty_f64 = np.empty(0, dtype=np.float64)
    model = BetaBinomialModel(
        FAMILY_MODEL_BINOMIAL,
        np.empty((0, 0), dtype=np.float64),
        np.empty((0, 0), dtype=np.float64),
        empty_f64,
        empty_f64,
        np.empty(0, dtype=np.bool_),
        np.empty(0, dtype=np.uint32),
        np.empty(0, dtype=np.uint32),
    )
    n_codes = alphabet.n**k
    full_index = SearchIndex(
        np.arange(n_codes + 1, dtype=np.uint32),
        np.zeros(n_codes, dtype=np.uint32),
        np.array([1e-6]),
        np.array([1e-6]),
        SEARCH_SEQUENCE,
        np.int64(0),
        np.int64(0),
        *model,
    )
    empty_index = full_index._replace(
        table_idx=np.zeros(n_codes + 1, dtype=np.uint32),
        table_buff=np.empty(0, dtype=np.uint32),
    )
    sequence = "RWVMWYYCLPGIPTNEQVFMHWCLVKDSYYWIRYNWNELP"
    query = np.frombuffer((sequence + " ").encode(), dtype=np.uint8)
    batch = SequenceBatch(
        query,
        np.array([0, len(sequence) + 1], dtype=np.uint64),
    )
    family_result_dtype = np.dtype(
        [
            ("id", np.uint32),
            ("pvalue", np.float64),
            ("ss_pvalue", np.float64),
            ("count", np.uint32),
            ("ss_count", np.uint32),
            ("score", np.uint32),
            ("ss_score", np.uint32),
            ("normcount", np.float64),
            ("overlap", np.float64),
            ("decision_source", np.uint8),
        ]
    )
    subfamily_result_dtype = np.dtype(
        [
            ("id", np.uint32),
            ("score", np.float64),
            ("count", np.uint32),
        ]
    )
    structure_index = full_index._replace(
        decision_source=SEARCH_STRUCTURE
    )
    kernel = object.__new__(MergeSearch)._lookup

    strategies = (
        (SEARCH_SEQUENCE, full_index, structure_index, SEARCH_SEQUENCE),
        (SEARCH_STRUCTURE, full_index, structure_index, SEARCH_STRUCTURE),
        (
            SEARCH_SEQUENCE_THEN_STRUCTURE,
            empty_index,
            structure_index,
            SEARCH_STRUCTURE,
        ),
    )
    for mode, sequence_index, ss_index, expected_source in strategies:
        family_results = np.zeros(
            (1, 1),
            dtype=family_result_dtype,
        )
        subfamily_results = np.zeros(
            (1, 1),
            dtype=subfamily_result_dtype,
        )
        config = SearchConfig(
            mode,
            1,
            1.0,
            0.0,
            0.1,
            True,
            numba.get_num_threads(),
        )
        kernel(
            family_results,
            subfamily_results,
            batch,
            batch,
            database,
            sequence_index,
            ss_index,
            config,
        )
        assert family_results["id"][0, 0] == 1
        assert (
            family_results["decision_source"][0, 0]
            == expected_source
        )

    assert kernel.nopython_signatures
