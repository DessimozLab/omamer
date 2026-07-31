import numba
import numpy as np
import pytest

from omamer.alphabets import Alphabet, get_transform
from omamer.merge_search import (
    SEARCH_SEQUENCE,
    SEARCH_SEQUENCE_THEN_STRUCTURE,
    SEARCH_STRUCTURE,
    MergeSearch,
    SearchConfig,
    SearchDatabase,
    SequenceBatch,
    dispatch_sequence,
    make_search_index,
    place_sequence,
    resolve_search_mode,
)
from omamer.stat_models import (
    FamilyModel,
    get_family_model,
    make_family_model,
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
        get_family_model("auto", False)
        == FamilyModel.BINOMIAL
    )
    assert (
        get_family_model("auto", True)
        == FamilyModel.BETA_BINOMIAL
    )
    assert (
        get_family_model("binomial", True)
        == FamilyModel.BINOMIAL
    )
    with pytest.raises(ValueError, match="requires beta-binomial"):
        get_family_model("bbinom", False)


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
    family_probability = np.array([1e-6])
    model = make_family_model("binomial", family_probability)
    n_codes = alphabet.n**k
    full_index = make_search_index(
        np.arange(n_codes + 1, dtype=np.uint32),
        np.zeros(n_codes, dtype=np.uint32),
        np.array([1e-6]),
        SEARCH_SEQUENCE,
        0,
        0,
        model,
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
            ("count", np.uint32),
            ("score", np.uint32),
            ("normcount", np.float64),
            ("overlap", np.float64),
            ("modality", np.uint8),
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
        modality=SEARCH_STRUCTURE
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
            0.0,
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
            family_results["modality"][0, 0]
            == expected_source
        )

    assert kernel.nopython_signatures
    assert kernel.targetoptions["nogil"]
    assert dispatch_sequence.targetoptions["nogil"]
    assert place_sequence.targetoptions["nogil"]
