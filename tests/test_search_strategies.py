import math
from types import SimpleNamespace

import numba
import numpy as np
import pytest

from omamer.alphabets import Alphabet, get_transform
from omamer.merge_search import (
    FAMILY_SELECT_CHOOSE_SCORED,
    FAMILY_SELECT_FILTER,
    QUERY_FAMILY_RESULT_DTYPE,
    SEARCH_HYBRID,
    SEARCH_SEQUENCE,
    SEARCH_SEQUENCE_THEN_STRUCTURE,
    SEARCH_STRUCTURE,
    SEARCH_SUM,
    KmerIndex,
    SearchConfig,
    SearchDatabase,
    SearchScratch,
    SequenceBatch,
    binom_neglogccdf,
    count_family_hogs,
    kmer_df_reference,
    lookup_queries,
    make_search_results,
    query_probability_scale,
    resolve_search_mode,
    score_family_candidates,
    search_seq_kmers,
    select_family_candidates,
    sum_family_scores,
)


@pytest.mark.parametrize(
    "mode,has_sequence,has_structure,expected",
    [
        ("auto", True, False, SEARCH_SEQUENCE),
        ("auto", False, True, SEARCH_STRUCTURE),
        ("auto", True, True, SEARCH_HYBRID),
        ("seq", True, True, SEARCH_SEQUENCE),
        ("ss", True, True, SEARCH_STRUCTURE),
        ("sqs", True, True, SEARCH_SEQUENCE_THEN_STRUCTURE),
        ("sum", True, True, SEARCH_SUM),
        ("hybrid", True, True, SEARCH_HYBRID),
    ],
)
def test_resolve_search_mode(mode, has_sequence, has_structure, expected):
    assert resolve_search_mode(mode, has_sequence, has_structure) == expected


def test_resolve_search_mode_rejects_unavailable_input():
    with pytest.raises(ValueError, match="requires matching"):
        resolve_search_mode("ss", True, False)
    with pytest.raises(ValueError, match="requires matching"):
        resolve_search_mode("sum", True, False)
    with pytest.raises(ValueError, match="requires matching"):
        resolve_search_mode("hybrid", False, True)
    with pytest.raises(ValueError, match="search_mode"):
        resolve_search_mode("both", True, True)


def make_index(table_idx, table_buff, modality, family_probability,
               hog_probability, max_df=0, kmer_mask=None, df_reference=0.0,
               df_exponent=0.0):
    return KmerIndex(
        table_idx,
        table_buff,
        modality,
        np.int64(max_df),
        np.zeros(0, dtype=np.bool_) if kmer_mask is None else kmer_mask,
        family_probability,
        hog_probability,
        float(df_reference),
        float(df_exponent),
    )


def make_config(mode=SEARCH_SEQUENCE, top_n=1, family_only=False,
                selection=FAMILY_SELECT_FILTER, neglog_alpha=0.0,
                log_correction=0.0, summed_neglog_alpha=None, ss_weight=1.0):
    return SearchConfig(
        mode,
        top_n,
        0.1,
        family_only,
        selection,
        neglog_alpha,
        log_correction,
        neglog_alpha if summed_neglog_alpha is None else summed_neglog_alpha,
        ss_weight,
    )


@pytest.fixture
def search_data():
    # two families: family 0 with HOGs 0 (root) and 1, family 1 with HOG 2;
    # every k-mer of the sequence index hits HOG 1, of the 3Di index HOG 2
    alphabet = Alphabet(21)
    hogs = np.array(
        [(0, -1), (0, 0), (1, -1)], dtype=[("FamOff", "u4"), ("ParentOff", "i4")]
    )
    database = SearchDatabase(
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
        hogs,
        np.array([0, 1, 2, 2, 2, 3, 3], dtype=np.uint32),
        np.ascontiguousarray(hogs["FamOff"]),
    )
    n_codes = alphabet.n**2
    sequence_index = make_index(
        np.arange(n_codes + 1, dtype=np.uint32),
        np.ones(n_codes, dtype=np.uint32),
        SEARCH_SEQUENCE,
        np.full(2, 1e-6),
        np.full(3, 1e-6),
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


def lookup(data, n_queries, sequences, structures, seq_index, ss_index, config,
           threads=None):
    results = make_search_results(n_queries, config.top_n_families)
    previous_threads = numba.get_num_threads()
    if threads is not None:
        numba.set_num_threads(threads)
    try:
        lookup_queries(
            results.families, results.subfamilies, sequences, structures,
            data.database, seq_index, ss_index, config, numba.get_num_threads(),
        )
    finally:
        numba.set_num_threads(previous_threads)
    return results


GOOD_QUERY = "RWVMWYYCLPGIPTNEQVFMHWCLVKDSYYWIRYNWNELP"


@pytest.mark.parametrize(
    "family_selection", [FAMILY_SELECT_FILTER, FAMILY_SELECT_CHOOSE_SCORED]
)
@pytest.mark.parametrize("family_only", [True, False])
@pytest.mark.parametrize("top_n_families", [1, 20])
def test_lookup_dispatches_search_strategies(
    search_data, family_only, top_n_families, family_selection
):
    data = search_data
    batch = query_batch([GOOD_QUERY])
    strategies = (
        (SEARCH_SEQUENCE, data.sequence_index, data.sequence_index, 1, 1),
        (SEARCH_STRUCTURE, data.structure_index, data.structure_index, 2, 2),
        (SEARCH_SEQUENCE_THEN_STRUCTURE, data.empty_index, data.structure_index, 2, 2),
        (SEARCH_SEQUENCE_THEN_STRUCTURE, data.sequence_index, data.structure_index, 1, 1),
        (SEARCH_SEQUENCE, data.empty_index, data.empty_index, 0, 0),
    )
    for mode, seq_index, ss_index, expected_source, expected_family in strategies:
        config = make_config(
            mode, top_n_families, family_only, family_selection
        )
        results = lookup(data, 1, batch, batch, seq_index, ss_index, config)
        assert results.families["id"][0, 0] == expected_family
        assert results.families["modality"][0, 0] == expected_source
        # fewer significant families than -n: the other ranks stay empty
        assert np.all(results.families[0, 1:] == np.zeros_like(results.families[0, 1:]))
        assert np.all(results.subfamilies["id"][0, 1:] == 0)
        if expected_source:
            expected_hog = (1 if family_only else 2) if expected_family == 1 else 3
            assert results.subfamilies["id"][0, 0] == expected_hog
            if not family_only:
                assert results.subfamilies["score"][0, 0] > 0.9

    assert lookup_queries.nopython_signatures
    assert lookup_queries.targetoptions["nogil"]


@pytest.mark.parametrize("threads", [1, 2])
def test_fallback_preserves_sequence_hits_in_mixed_batch(search_data, threads):
    data = search_data
    sequences = [GOOD_QUERY, "X" * 40, "A" * 40, "A", GOOD_QUERY, "X" * 40, ""]
    structures = [GOOD_QUERY] * 4 + ["X" * 40, "X" * 40, GOOD_QUERY]
    results = lookup(
        data, len(sequences), query_batch(sequences), query_batch(structures),
        data.sequence_index, data.structure_index,
        make_config(SEARCH_SEQUENCE_THEN_STRUCTURE, 20), threads,
    )
    np.testing.assert_array_equal(
        results.families["modality"][:, 0], [1, 2, 2, 2, 1, 0, 2]
    )
    np.testing.assert_array_equal(results.families["id"][:, 0], [1, 2, 2, 2, 1, 0, 2])
    np.testing.assert_array_equal(
        results.subfamilies["id"][:, 0], [2, 3, 3, 3, 2, 0, 3]
    )
    assert not np.any(results.families["id"][:, 1:])


def _family_results(rows):
    qres = np.zeros(len(rows), dtype=QUERY_FAMILY_RESULT_DTYPE)
    for i, row in enumerate(rows):
        for field, value in zip(("id", "pvalue", "count", "normcount", "overlap"), row):
            qres[field][i] = value
    return qres


def test_sum_family_scores_unions_families():
    seq = _family_results([(3, 10.0, 7, 0.5, 0.9), (1, 2.0, 4, 0.2, 0.8)])
    ss = _family_results([(1, 5.0, 9, 0.7, 0.6), (5, 1.5, 3, 0.1, 0.5)])
    combined = sum_family_scores(seq, ss, 1.0)
    by_id = {int(x["id"]): x for x in combined}
    assert sorted(by_id) == [1, 3, 5]
    assert by_id[3]["pvalue"] == 10.0
    assert by_id[1]["pvalue"] == 7.0
    assert by_id[5]["pvalue"] == 1.5
    # the other fields come from the sequence when it has the family
    assert by_id[1]["count"] == 4 and by_id[1]["overlap"] == 0.8
    assert by_id[5]["count"] == 3 and by_id[5]["normcount"] == 0.1
    assert len(sum_family_scores(seq[:0], ss[:0], 1.0)) == 0
    np.testing.assert_array_equal(sum_family_scores(seq[:0], ss, 1.0), ss)

    # the 3Di scores are weighted
    by_id = {int(x["id"]): x for x in sum_family_scores(seq, ss, 0.2)}
    assert by_id[3]["pvalue"] == 10.0
    assert by_id[1]["pvalue"] == pytest.approx(3.0)
    assert by_id[5]["pvalue"] == pytest.approx(0.3)
    assert by_id[5]["count"] == 3


@pytest.mark.parametrize("family_only", [True, False])
@pytest.mark.parametrize("threads", [1, 2])
def test_sum_mode_adds_scores_and_places_with_scoring_modality(
    search_data, family_only, threads
):
    data = search_data
    sequences = query_batch([GOOD_QUERY, "X" * 40, GOOD_QUERY, ""])
    structures = query_batch([GOOD_QUERY, GOOD_QUERY, "X" * 40, ""])
    seq_index, ss_index = data.sequence_index, data.structure_index

    def run(mode, seq_index, ss_index, **options):
        config = make_config(mode, 20, family_only, **options)
        return lookup(
            data, 4, sequences, structures, seq_index, ss_index, config, threads
        )

    # the sequence index hits family 1 (HOG 2), the structure one family 2
    seq_only = run(SEARCH_SEQUENCE, seq_index, seq_index)
    ss_only = lookup(
        data, 4, structures, structures, ss_index, ss_index,
        make_config(SEARCH_STRUCTURE, 20, family_only), threads,
    )
    summed = run(SEARCH_SUM, seq_index, ss_index)
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
    hybrid = run(SEARCH_HYBRID, seq_index, ss_index)
    for query, expected in ((0, seq_only), (1, summed), (2, seq_only), (3, summed)):
        np.testing.assert_array_equal(hybrid.families[query], expected.families[query])
        np.testing.assert_array_equal(
            hybrid.subfamilies[query], expected.subfamilies[query]
        )
    assert hybrid.families["modality"][0, 0] == SEARCH_SEQUENCE
    assert hybrid.families["modality"][1, 0] == SEARCH_SUM

    # the same family in both modalities: the scores add up
    both = run(SEARCH_SUM, seq_index, seq_index)
    assert both.families["id"][0, 0] == 1
    np.testing.assert_allclose(
        both.families["pvalue"][0, 0], 2 * seq_only.families["pvalue"][0, 0]
    )
    assert both.subfamilies["id"][0, 0] == seq_hog

    # the threshold applies to the sum: neither modality alone passes
    single = seq_only.families["pvalue"][0, 0]
    strict = run(SEARCH_SUM, seq_index, seq_index, neglog_alpha=1.5 * single)
    assert strict.families["id"][0, 0] == 1
    strict = run(SEARCH_SEQUENCE, seq_index, seq_index, neglog_alpha=1.5 * single)
    assert strict.families["id"][0, 0] == 0

    # a separate threshold for summed scores (--summed_alpha)
    results = run(
        SEARCH_SUM, seq_index, seq_index, summed_neglog_alpha=2.5 * single
    )
    assert results.families["id"][0, 0] == 0
    results = run(
        SEARCH_SUM, seq_index, seq_index, neglog_alpha=10 * single,
        summed_neglog_alpha=1.5 * single,
    )
    assert results.families["id"][0, 0] == 1


def _selection_candidates():
    # n = 10 k-mers. Family 1 (p 0.5, 8 k-mers): normcount 0.6, P(X >= 8)
    # = 0.055. Family 2 (p 1e-4, 5 k-mers): normcount 0.5, significant.
    # Family 3 (p 1e-3, 2 k-mers): normcount 0.2, significant at a lower
    # level only.
    index = make_index(
        np.zeros(1, dtype=np.uint32), np.zeros(0, dtype=np.uint32),
        SEARCH_SEQUENCE, np.array([0.1, 0.5, 1e-4, 1e-3]), np.zeros(0),
    )
    qres = np.zeros(3, dtype=QUERY_FAMILY_RESULT_DTYPE)
    qres["id"][:] = [1, 2, 3]
    qres["count"][:] = [8, 5, 2]
    qres["overlap"][:] = 1.0
    return qres, index


@pytest.mark.parametrize("top_n", [1, 3])
@pytest.mark.parametrize(
    "selection,expected",
    [
        # the significant families, best by normcount first
        (FAMILY_SELECT_FILTER, [2]),
        # family 1 has p = 0.055 > 1/N (score 0): not ranked
        (FAMILY_SELECT_CHOOSE_SCORED, [2]),
    ],
)
def test_select_family_candidates(selection, expected, top_n):
    qres, index = _selection_candidates()
    # N = e^5 families (score = -ln p - 5), alpha = e^-10
    config = make_config(top_n=top_n, selection=selection, neglog_alpha=10.0,
                         log_correction=5.0)
    selected = select_family_candidates(qres.copy(), 10, index, config, 1.0)
    assert list(selected["id"]) == expected
    assert selected["pvalue"][0] == pytest.approx(
        -math.log(math.comb(10, 5)) - 5 * math.log(1e-4) - 5.0, rel=1e-3
    )
    assert selected["normcount"][0] == pytest.approx(0.5, rel=1e-3)
    # the candidates are not modified
    np.testing.assert_array_equal(qres, _selection_candidates()[0])


def test_choose_scored_keeps_the_significant_families_below_a_significant_top():
    qres, index = _selection_candidates()
    # without family 1, family 2 is the top one; family 3 (score
    # -ln(45e-6) - 5 = 5.0) is significant at alpha e^-4 only
    qres = qres[1:]
    for neglog_alpha, expected in ((10.0, [2]), (4.0, [2, 3])):
        config = make_config(top_n=3, selection=FAMILY_SELECT_CHOOSE_SCORED,
                             neglog_alpha=neglog_alpha, log_correction=5.0)
        selected = select_family_candidates(qres.copy(), 10, index, config, 1.0)
        assert list(selected["id"]) == expected


def test_choose_scored_ranks_unscaled_and_tests_scaled():
    qres, index = _selection_candidates()
    config = make_config(selection=FAMILY_SELECT_CHOOSE_SCORED, neglog_alpha=10.0,
                         log_correction=5.0)
    # p scaled by 0.1 gives family 1 (p 0.05, 8 of 10 k-mers) a score of
    # about 15 and the best normcount, but its unscaled score is 0: it is
    # not ranked; the chosen family 2 is tested with p = 1e-5
    selected = select_family_candidates(qres.copy(), 10, index, config, 0.1)
    assert list(selected["id"]) == [2]
    assert selected["pvalue"][0] == pytest.approx(
        -math.log(math.comb(10, 5)) - 5 * math.log(1e-5) - 5.0, rel=1e-3
    )
    # scaled up 1000x (p 0.1), family 2 is no longer significant
    selected = select_family_candidates(qres.copy(), 10, index, config, 1000.0)
    assert len(selected) == 0
    # the filter rule tests every candidate with the scaling: family 1 too
    config = config._replace(family_selection=FAMILY_SELECT_FILTER)
    selected = select_family_candidates(qres.copy(), 10, index, config, 0.1)
    assert sorted(selected["id"]) == [1, 2]


def test_kmer_df_reference_is_the_posting_weighted_mean():
    # document frequencies 1, 3, 0, 4
    table_idx = np.array([0, 1, 4, 4, 8])
    assert kmer_df_reference(table_idx) == pytest.approx((1 + 9 + 16) / 8)
    # df > 3 ignored; masked k-mers ignored
    assert kmer_df_reference(table_idx, max_df=3) == pytest.approx(10 / 4)
    mask = np.array([False, True, False, False])
    assert kmer_df_reference(table_idx, kmer_mask=mask) == pytest.approx(17 / 5)


def test_query_probability_scale_scores_as_a_scaled_probability():
    p = np.array([1e-3, 2e-3])
    index = make_index(
        np.zeros(1, dtype=np.uint32), np.zeros(0, dtype=np.uint32),
        SEARCH_SEQUENCE, p, np.zeros(0),
    )
    # off by default
    assert query_probability_scale(index, 500.0, 10) == 1.0
    index = index._replace(df_reference=25.0, df_exponent=1.5)
    # mean df 100 = 4 x the reference
    assert query_probability_scale(index, 1000.0, 10) == pytest.approx(8.0)
    assert query_probability_scale(index, 0.0, 10) == 1.0

    qres = np.zeros(2, dtype=QUERY_FAMILY_RESULT_DTYPE)
    qres["id"][:] = [0, 1]
    qres["count"][:] = [12, 20]
    plain = score_family_candidates(qres.copy(), 300, index, 0.0, 0.0)
    scaled = score_family_candidates(qres.copy(), 300, index, 0.0, 0.0, 8.0)
    for i, family in enumerate(scaled["id"]):
        assert scaled["pvalue"][i] == pytest.approx(
            binom_neglogccdf(int(qres["count"][family]), 300, 8.0 * p[family])
        )
        assert scaled["pvalue"][i] < plain["pvalue"][plain["id"] == family][0]
        # the ranking statistic is not scaled
        assert scaled["normcount"][i] == plain["normcount"][plain["id"] == family][0]
    # family 0 with 2 k-mers: E[X] = 0.3 unscaled, 2.4 scaled -> the count
    # filter removes it from the scaled scoring only
    qres["count"][:] = [2, 20]
    assert list(score_family_candidates(qres.copy(), 300, index, 0.0, 0.0)["id"]) == [0, 1]
    assert list(
        score_family_candidates(qres.copy(), 300, index, 0.0, 0.0, 8.0)["id"]
    ) == [1]


def test_binom_neglogccdf_matches_scipy():
    from scipy.stats import binom

    for (x, n, p) in [(1, 10, 0.1), (5, 37, 0.02), (30, 300, 1e-3), (37, 37, 0.5)]:
        expected = -np.log(binom.sf(x - 1, n, p))
        assert binom_neglogccdf(x, n, p) == pytest.approx(expected, rel=1e-9)
    assert binom_neglogccdf(0, 10, 0.1) == 0.0
    assert binom_neglogccdf(11, 10, 0.1) == np.inf
    # p = 1 (the only family of a database) is clipped below 1
    assert 0.0 <= binom_neglogccdf(5, 5, 1.0) < 1e-6


def _scratch(n_families):
    return SearchScratch(
        np.zeros((1, n_families), dtype=np.int32),
        np.zeros((1, n_families), dtype=np.uint16),
        np.full((1, n_families), -1, dtype=np.int32),
        np.full((1, n_families), -1, dtype=np.int32),
        np.zeros(1, dtype=np.uint32),
    )


def test_search_skips_capped_masked_and_x_kmers():
    # k-mer codes 0..3 with document frequencies 1, 2, 0, 1; code 4 is the
    # X flag
    table_idx = np.asarray([0, 1, 3, 3, 4], dtype=np.uint32)
    table_buff = np.asarray([0, 1, 2, 2], dtype=np.uint32)
    hog2fam = np.asarray([0, 1, 1], dtype=np.uint32)
    index = make_index(
        table_idx, table_buff, SEARCH_SEQUENCE, np.zeros(2), np.zeros(3), max_df=1
    )
    scratch = _scratch(2)
    (n_skipped, kmers, sum_df) = search_seq_kmers(
        np.asarray([0, 1, 2, 3, 4], dtype=np.uint32),
        np.asarray([0, 1, 2, 3, 4], dtype=np.uint32),
        hog2fam, np.uint32(4), index, scratch, 0,
    )
    # code 1 has df 2 and is capped, code 4 is the X flag; the absent code 2
    # counts towards n but has no postings
    assert (scratch.num_hit_families[0], n_skipped) == (2, 2)
    # the document frequencies of the counted codes 0, 2 and 3
    assert sum_df == 2.0
    np.testing.assert_array_equal(kmers, [0, 3])
    np.testing.assert_array_equal(scratch.family_counts[0], [1, 1])
    np.testing.assert_array_equal(scratch.family_low_location[0], [0, 3])
    np.testing.assert_array_equal(scratch.family_high_location[0], [0, 3])
    np.testing.assert_array_equal(count_family_hogs(kmers, index, 1, 3), [0, 1])
    np.testing.assert_array_equal(count_family_hogs(kmers, index, 0, 1), [1])

    # a masked k-mer is skipped as well, and the counters of the previous
    # query are reset
    index = index._replace(
        max_df=np.int64(0), kmer_mask=np.array([True, False, False, False])
    )
    (n_skipped, kmers, sum_df) = search_seq_kmers(
        np.asarray([0, 1], dtype=np.uint32), np.asarray([0, 1], dtype=np.uint32),
        hog2fam, np.uint32(4), index, scratch, 0,
    )
    assert n_skipped == 1 and sum_df == 2.0
    np.testing.assert_array_equal(kmers, [1])
    np.testing.assert_array_equal(scratch.family_counts[0], [0, 2])


def test_count_family_hogs_matches_posting_scan():
    rng = np.random.default_rng(1)
    n_hogs, n_kmers = 40, 30
    lists = [
        np.sort(rng.choice(n_hogs, size=rng.integers(0, 12), replace=False))
        for _ in range(n_kmers)
    ]
    table_idx = np.zeros(n_kmers + 1, dtype=np.uint32)
    table_idx[1:] = np.cumsum([len(x) for x in lists])
    table_buff = np.concatenate(lists).astype(np.uint32)
    index = make_index(table_idx, table_buff, SEARCH_SEQUENCE, np.zeros(1), np.zeros(1))
    kmers = np.arange(n_kmers, dtype=np.uint32)
    expected = np.zeros(n_hogs, dtype=np.uint16)
    np.add.at(expected, table_buff, 1)
    for (hog_s, hog_e) in [(0, 40), (0, 1), (5, 17), (39, 40), (12, 12)]:
        np.testing.assert_array_equal(
            count_family_hogs(kmers, index, hog_s, hog_e), expected[hog_s:hog_e]
        )
