import math
from types import SimpleNamespace

import numba
import numpy as np
import pytest

from omamer.alphabets import Alphabet, get_transform
from omamer.family_sort import FAMILY_SORT_NORMCOUNT, FAMILY_SORT_PVALUE
from omamer.merge_search import (
    SEARCH_HYBRID,
    SEARCH_SEQUENCE,
    SEARCH_SEQUENCE_THEN_STRUCTURE,
    SEARCH_STRUCTURE,
    SEARCH_SUM,
    MergeSearch,
    PlacementConfig,
    SearchModality,
    SequenceBatch,
    default_family_sorting,
    init_query_family_results,
    lookup_queries,
    make_kmer_index,
    make_search_database,
    make_search_results,
    place_sequence,
    resolve_search_mode,
    sum_family_scores,
)
from omamer.stat_models import (
    FamilyModel,
    FamilyScoringParameters,
    HogModelParameters,
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
        ("sum", True, True, SEARCH_SUM),
        ("hybrid", True, True, SEARCH_HYBRID),
    ],
)
def test_resolve_search_mode(
    mode,
    has_sequence,
    has_structure,
    expected,
):
    assert resolve_search_mode(mode, has_sequence, has_structure) == expected


def test_resolve_search_mode_rejects_unavailable_input():
    with pytest.raises(ValueError, match="requires matching"):
        resolve_search_mode("ss", True, False)
    with pytest.raises(ValueError, match="requires matching"):
        resolve_search_mode("sum", True, False)
    with pytest.raises(ValueError, match="requires matching"):
        resolve_search_mode("hybrid", False, True)


def test_default_family_sorting():
    assert default_family_sorting("auto", SEARCH_SUM) == "pvalue"
    assert default_family_sorting("auto", SEARCH_SEQUENCE) == "normcount"
    assert default_family_sorting("auto", SEARCH_SEQUENCE_THEN_STRUCTURE) == "normcount"
    assert default_family_sorting("normcount", SEARCH_SUM) == "normcount"


def test_resolve_family_model():
    assert get_family_model("auto", False) == FamilyModel.BINOMIAL
    assert get_family_model("auto", True) == FamilyModel.BETA_BINOMIAL
    assert get_family_model("binomial", True) == FamilyModel.BINOMIAL
    with pytest.raises(ValueError, match="requires beta-binomial"):
        get_family_model("bbinom", False)


def test_auto_model_reports_all_invalid_beta_binomial_fallback(caplog):
    search = SimpleNamespace(
        ss_ref_fam_prob=np.asarray([0.01]),
        ss_ref_fam_bbinom={
            "q_coef": np.zeros((1, 3)),
            "kappa_coef": np.zeros((1, 2)),
            "center": np.zeros(1),
            "scale": np.ones(1),
            "valid": np.zeros(1, dtype=np.bool_),
            "n_min": np.ones(1, dtype=np.uint32),
            "n_max": np.ones(1, dtype=np.uint32),
        },
        resolved_family_models={},
        _reported_family_models=set(),
    )

    model = MergeSearch._family_model(search, "ss", "auto")

    assert model.kind == FamilyModel.BINOMIAL
    assert search.resolved_family_models == {"ss": "binomial"}
    assert "stores beta-binomial arrays but has no valid fits" in caplog.text


@pytest.fixture
def search_data():
    alphabet = Alphabet(21)
    database = make_search_database(
        get_transform(2, alphabet.DIGITS_AA),
        2,
        alphabet.DIGITS_AA_LOOKUP,
        np.array(
            [(0, 2, 0, 2), (2, 1, 4, 1)],
            dtype=[
                ("HOGoff", "u4"),
                ("HOGnum", "u4"),
                ("LevelOff", "u4"),
                ("LevelNum", "u4"),
            ],
        ),
        np.array(
            [(0, -1), (0, 0), (1, -1)], dtype=[("FamOff", "u4"), ("ParentOff", "i4")]
        ),
        np.array([0, 1, 2, 2, 2, 3, 3], dtype=np.uint32),
    )
    model = make_family_model("binomial", np.full(2, 1e-6))
    hog_model = HogModelParameters(np.full(3, 1e-6))
    n_codes = alphabet.n**2
    sequence_index = make_kmer_index(
        np.arange(n_codes + 1, dtype=np.uint32),
        np.ones(n_codes, dtype=np.uint32),
        SEARCH_SEQUENCE,
        0,
    )
    structure_index = sequence_index._replace(
        table_buff=np.full(n_codes, 2, dtype=np.uint32),
        modality=SEARCH_STRUCTURE,
    )
    empty_index = sequence_index._replace(
        table_idx=np.zeros(n_codes + 1, dtype=np.uint32),
        table_buff=np.empty(0, dtype=np.uint32),
    )
    return SimpleNamespace(
        database=database,
        model=model,
        hog_model=hog_model,
        sequence_index=sequence_index,
        structure_index=structure_index,
        empty_index=empty_index,
    )


def query_batch(sequences):
    return SequenceBatch(
        np.frombuffer((" ".join(sequences) + " ").encode(), dtype=np.uint8),
        np.asarray(
            [0] + list(np.cumsum([len(seq) + 1 for seq in sequences])), dtype=np.uint64
        ),
    )


GOOD_QUERY = "RWVMWYYCLPGIPTNEQVFMHWCLVKDSYYWIRYNWNELP"


@pytest.mark.parametrize("family_only", [True, False])
@pytest.mark.parametrize("top_n_families", [1, 20])
def test_lookup_compiles_and_dispatches_search_strategies(
    search_data, family_only, top_n_families
):
    data = search_data
    batch = query_batch([GOOD_QUERY])
    placement = PlacementConfig(top_n_families, 0.1, family_only, FAMILY_SORT_NORMCOUNT)
    family_scoring = FamilyScoringParameters(0.0, 0.0)
    sequence_search = SearchModality(
        batch, data.sequence_index, data.model, data.hog_model
    )
    structure_search = SearchModality(
        batch, data.structure_index, data.model, data.hog_model
    )
    empty_search = SearchModality(batch, data.empty_index, data.model, data.hog_model)
    strategies = (
        ((sequence_search,), SEARCH_SEQUENCE, 1),
        ((structure_search,), SEARCH_STRUCTURE, 2),
        ((empty_search, structure_search), SEARCH_STRUCTURE, 2),
        ((sequence_search, structure_search), SEARCH_SEQUENCE, 1),
        ((empty_search,), 0, 0),
    )
    search = SimpleNamespace(_search_database=data.database)
    for modalities, expected_source, expected_family in strategies:
        results = make_search_results(1, top_n_families)
        MergeSearch._lookup(search, results, modalities, placement, family_scoring)
        assert results.families["id"][0, 0] == expected_family
        assert results.families["modality"][0, 0] == expected_source
        assert np.all(results.families["id"][0, 1:] == 0)
        assert np.all(results.subfamilies["id"][0, 1:] == 0)
        if expected_source:
            expected_hog = (1 if family_only else 2) if expected_family == 1 else 3
            assert results.subfamilies["id"][0, 0] == expected_hog
            if not family_only:
                assert results.subfamilies["score"][0, 0] > 0.9

    assert lookup_queries.nopython_signatures
    assert lookup_queries.targetoptions["nogil"]
    assert place_sequence.targetoptions["nogil"]


@pytest.mark.parametrize("family_model", ["binomial", "beta-binomial"])
@pytest.mark.parametrize("threads", [1, 2])
def test_fallback_preserves_sequence_hits_in_mixed_batch(
    search_data, family_model, threads
):
    data = search_data
    model = data.model
    if family_model == "beta-binomial":
        model = make_family_model(
            family_model,
            np.full(2, 1e-6),
            np.array([[-10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]]),
            np.array([[3.0, 0.0], [3.0, 0.0]]),
            np.zeros(2),
            np.ones(2),
            np.array([True, False]),
            np.ones(2, dtype=np.uint32),
            np.full(2, 100, dtype=np.uint32),
        )
    sequences = [GOOD_QUERY, "X" * 40, "A" * 40, "A", GOOD_QUERY, "X" * 40, ""]
    structures = [GOOD_QUERY] * 4 + ["X" * 40, "X" * 40, GOOD_QUERY]
    modalities = (
        SearchModality(
            query_batch(sequences), data.sequence_index, model, data.hog_model
        ),
        SearchModality(
            query_batch(structures), data.structure_index, model, data.hog_model
        ),
    )
    previous_threads = numba.get_num_threads()
    numba.set_num_threads(threads)
    try:
        results = make_search_results(len(sequences), 20)
        MergeSearch._lookup(
            SimpleNamespace(_search_database=data.database),
            results,
            modalities,
            PlacementConfig(20, 0.1, False, FAMILY_SORT_NORMCOUNT),
            FamilyScoringParameters(0.0, 0.0),
        )
    finally:
        numba.set_num_threads(previous_threads)
    np.testing.assert_array_equal(
        results.families["modality"][:, 0], [1, 2, 2, 2, 1, 0, 2]
    )
    np.testing.assert_array_equal(results.families["id"][:, 0], [1, 2, 2, 2, 1, 0, 2])
    np.testing.assert_array_equal(
        results.subfamilies["id"][:, 0], [2, 3, 3, 3, 2, 0, 3]
    )
    assert not np.any(results.families["id"][:, 1:])


def test_search_plan_loads_only_active_modalities(monkeypatch):
    search = object.__new__(MergeSearch)
    calls = []

    def load(modality, *args):
        calls.append(modality)
        return modality

    monkeypatch.setattr(search, "_search_modality", load)
    buffer = SimpleNamespace(idx=np.arange(3))
    for mode, expected in (
        (SEARCH_SEQUENCE, ("seq",)),
        (SEARCH_STRUCTURE, ("ss",)),
        (SEARCH_SEQUENCE_THEN_STRUCTURE, ("seq", "ss")),
    ):
        calls.clear()
        assert search._search_plan(mode, buffer, buffer, "auto") == expected
        assert tuple(calls) == expected
    with pytest.raises(ValueError, match="matching counts"):
        search._search_plan(
            SEARCH_SEQUENCE_THEN_STRUCTURE,
            buffer,
            SimpleNamespace(idx=np.arange(2)),
            "auto",
        )


@pytest.mark.parametrize("modality", ["seq", "ss"])
@pytest.mark.parametrize("policy_version", [0, 1])
def test_coefficient_loading_applies_legacy_validity_policy(
    tmp_path, modality, policy_version
):
    import tables
    from omamer.bbinom_coefficients import BBINOM_NODE_NAMES

    coefficients = {
        "q_coef": np.array(
            [[-3.0, 0.1, 0.0], [50.0, 0.1, 0.0], [-3.0, 0.0, 0.0], [-3.0, 0.1, 0.0]]
        ),
        "kappa_coef": np.array(
            [[3.0, 0.0], [3.0, 0.0], [np.log(1000.0), 0.0], [3.0, 0.0]]
        ),
        "center": np.zeros(4),
        "scale": np.ones(4),
        "valid": np.array([True, True, True, False]),
    }
    with tables.open_file(tmp_path / "coefficients.h5", mode="w") as handle:
        group = handle.create_group("/", "Index")
        group._v_attrs[modality + "_bbinom_validity_policy_version"] = policy_version
        for field, values in coefficients.items():
            handle.create_array(group, BBINOM_NODE_NAMES[modality][field], values)
        search = object.__new__(MergeSearch)
        search.db = SimpleNamespace(db=handle)
        loaded = (
            search.ref_fam_bbinom if modality == "seq" else search.ss_ref_fam_bbinom
        )
        np.testing.assert_array_equal(
            loaded["valid"], [True, False, bool(policy_version), False]
        )
        assert loaded["n_min"].dtype == np.uint32
        assert loaded["n_max"].size == 0
        np.testing.assert_array_equal(loaded["q_coef"], coefficients["q_coef"])
        assert loaded is (
            search.ref_fam_bbinom if modality == "seq" else search.ss_ref_fam_bbinom
        )


@pytest.mark.parametrize("modality", ["seq", "ss"])
def test_explicit_binomial_does_not_load_coefficients(modality):
    search = SimpleNamespace(
        ref_fam_prob=np.array([0.01]),
        ss_ref_fam_prob=np.array([0.02]),
        resolved_family_models={},
        _reported_family_models=set(),
    )
    model = MergeSearch._family_model(search, modality, "binomial")
    assert model.kind == FamilyModel.BINOMIAL
    assert search.resolved_family_models == {modality: "binomial"}


def test_index_infers_beta_binomial_model_for_legacy_database(tmp_path):
    import tables
    from omamer.index import Index

    with tables.open_file(tmp_path / "legacy.h5", mode="w") as handle:
        group = handle.create_group("/", "Index")
        group._v_attrs.k = 2
        group._v_attrs.alphabet_n = 21
        group._v_attrs.hidden_taxa = ()
        handle.create_array(group, "FamilyBBinomValid", np.array([True]))
        index = Index(SimpleNamespace(db=handle))
        assert index.models == ("binomial", "beta-binomial")


def _family_results(rows):
    qres = init_query_family_results(len(rows))
    for i, row in enumerate(rows):
        for field, value in zip(("id", "pvalue", "count", "normcount", "overlap"), row):
            qres[field][i] = value
    return qres


def test_sum_family_scores_unions_families():
    seq = _family_results([(3, 10.0, 7, 0.5, 0.9), (1, 2.0, 4, 0.2, 0.8)])
    ss = _family_results([(1, 5.0, 9, 0.7, 0.6), (5, 1.5, 3, 0.1, 0.5)])
    combined = sum_family_scores(seq, ss)
    by_id = {int(x["id"]): x for x in combined}
    assert sorted(by_id) == [1, 3, 5]
    assert by_id[3]["pvalue"] == 10.0
    assert by_id[1]["pvalue"] == 7.0
    assert by_id[5]["pvalue"] == 1.5
    # the other fields come from the sequence when it has the family
    assert by_id[1]["count"] == 4 and by_id[1]["overlap"] == 0.8
    assert by_id[5]["count"] == 3 and by_id[5]["normcount"] == 0.1
    assert len(sum_family_scores(seq[:0], ss[:0])) == 0
    np.testing.assert_array_equal(sum_family_scores(seq[:0], ss), ss)


@pytest.mark.parametrize("family_only", [True, False])
@pytest.mark.parametrize("threads", [1, 2])
def test_sum_mode_adds_scores_and_places_with_scoring_modality(
    search_data, family_only, threads
):
    data = search_data
    sequences = [GOOD_QUERY, "X" * 40, GOOD_QUERY, ""]
    structures = [GOOD_QUERY, GOOD_QUERY, "X" * 40, ""]
    # the sequence index hits family 1 (HOG 2), the structure one family 2
    sequence_search = SearchModality(
        query_batch(sequences), data.sequence_index, data.model, data.hog_model
    )
    structure_search = SearchModality(
        query_batch(structures), data.structure_index, data.model, data.hog_model
    )
    search = SimpleNamespace(_search_database=data.database)
    placement = PlacementConfig(20, 0.1, family_only, FAMILY_SORT_PVALUE)
    scoring = FamilyScoringParameters(0.0, 0.0)

    def run(modalities, mode):
        results = make_search_results(len(sequences), 20)
        previous_threads = numba.get_num_threads()
        numba.set_num_threads(threads)
        try:
            MergeSearch._lookup(search, results, modalities, placement, scoring, mode)
        finally:
            numba.set_num_threads(previous_threads)
        return results

    seq_only = run((sequence_search,), SEARCH_SEQUENCE)
    ss_only = run((structure_search,), SEARCH_STRUCTURE)
    summed = run((sequence_search, structure_search), SEARCH_SUM)
    fam = summed.families
    sub = summed.subfamilies
    seq_hog, ss_hog = (1, 3) if family_only else (2, 3)

    # query 0: both families, each from one modality
    assert sorted(fam["id"][0, :2]) == [1, 2]
    assert np.all(fam["modality"][0, :2] == SEARCH_SUM)
    for rank in range(2):
        if fam["id"][0, rank] == 1:
            assert fam["pvalue"][0, rank] == seq_only.families["pvalue"][0, 0]
            assert sub["id"][0, rank] == seq_hog
        else:
            assert fam["pvalue"][0, rank] == ss_only.families["pvalue"][0, 0]
            assert sub["id"][0, rank] == ss_hog
    # queries 1, 2: a single usable modality; query 3: nothing
    assert fam["id"][1, 0] == 2 and sub["id"][1, 0] == ss_hog
    assert fam["id"][2, 0] == 1 and sub["id"][2, 0] == seq_hog
    assert not np.any(fam["id"][1:, 1:])
    assert fam["id"][3, 0] == 0 and fam["modality"][3, 0] == 0

    # hybrid: the sequence results for the queries with a significant
    # sequence family (0, 2), the summed ones for the others (1, 3)
    hybrid = run((sequence_search, structure_search), SEARCH_HYBRID)
    for query, expected in ((0, seq_only), (1, summed), (2, seq_only), (3, summed)):
        np.testing.assert_array_equal(
            hybrid.families[query], expected.families[query]
        )
        np.testing.assert_array_equal(
            hybrid.subfamilies[query], expected.subfamilies[query]
        )
    assert hybrid.families["modality"][0, 0] == SEARCH_SEQUENCE
    assert hybrid.families["modality"][1, 0] == SEARCH_SUM

    # the same family in both modalities: the scores add up
    both = run((sequence_search, sequence_search), SEARCH_SUM)
    assert both.families["id"][0, 0] == 1
    np.testing.assert_allclose(
        both.families["pvalue"][0, 0], 2 * seq_only.families["pvalue"][0, 0]
    )
    assert both.subfamilies["id"][0, 0] == seq_hog

    # the threshold applies to the sum: neither modality alone passes
    single = seq_only.families["pvalue"][0, 0]
    strict = FamilyScoringParameters(1.5 * single, 0.0)
    results = make_search_results(len(sequences), 20)
    MergeSearch._lookup(
        search, results, (sequence_search, sequence_search), placement, strict,
        SEARCH_SUM,
    )
    assert results.families["id"][0, 0] == 1
    results = make_search_results(len(sequences), 20)
    MergeSearch._lookup(
        search, results, (sequence_search,), placement, strict, SEARCH_SEQUENCE
    )
    assert results.families["id"][0, 0] == 0

    # a separate threshold for summed scores (--summed_alpha): above the sum,
    # nothing is placed; at the single-modality threshold the sum passes
    summed_strict = FamilyScoringParameters(0.0, 0.0, 2.5 * single)
    results = make_search_results(len(sequences), 20)
    MergeSearch._lookup(
        search, results, (sequence_search, sequence_search), placement,
        summed_strict, SEARCH_SUM,
    )
    assert results.families["id"][0, 0] == 0
    summed_loose = FamilyScoringParameters(10 * single, 0.0, 1.5 * single)
    results = make_search_results(len(sequences), 20)
    MergeSearch._lookup(
        search, results, (sequence_search, sequence_search), placement,
        summed_loose, SEARCH_SUM,
    )
    assert results.families["id"][0, 0] == 1


def test_summed_threshold_and_ss_weight_defaults():
    from omamer.merge_search import (
        HYBRID_SS_WEIGHT, resolve_ss_weight, summed_neglog_alpha,
    )
    assert summed_neglog_alpha(math.exp(-50)) == pytest.approx(50.0)
    assert summed_neglog_alpha(None) == -1.0
    with pytest.raises(ValueError):
        summed_neglog_alpha(0.0)
    # the 3Di evidence is discounted in the hybrid mode, not in the sum mode
    assert resolve_ss_weight(None, SEARCH_HYBRID) == HYBRID_SS_WEIGHT
    assert resolve_ss_weight(None, SEARCH_SUM) == 1.0
    assert resolve_ss_weight(0.5, SEARCH_HYBRID) == 0.5
    with pytest.raises(ValueError):
        resolve_ss_weight(0.0, SEARCH_SUM)


def test_sum_family_scores_weights_3di():
    seq = _family_results([(3, 10.0, 7, 0.5, 0.9), (1, 2.0, 4, 0.2, 0.8)])
    ss = _family_results([(1, 5.0, 9, 0.7, 0.6), (5, 1.5, 3, 0.1, 0.5)])
    by_id = {int(x["id"]): x for x in sum_family_scores(seq, ss, 0.2)}
    assert by_id[3]["pvalue"] == 10.0
    assert by_id[1]["pvalue"] == pytest.approx(3.0)
    assert by_id[5]["pvalue"] == pytest.approx(0.3)
    assert by_id[5]["count"] == 3
