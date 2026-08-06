import numpy as np
import numba
import pytest
import tables
from types import SimpleNamespace
from scipy.stats import betabinom
from scipy.optimize import check_grad
from omamer.alphabets import Alphabet, get_transform
from omamer.bbinom_coefficients import bbinom_coefficient_valid_mask
from omamer.bbinom_fit import (
    DenseFamilyHitHistogram,
    _bbinom_objective_and_gradient,
    _unique_valid_codes,
    choose_sequence_n_values,
    count_selected_family_hits,
    fit_family_bbinom_from_hist,
)
from omamer.database import DatabaseFromOMABrowser
from omamer.index import (
    filtered_hog_kmer_counts,
    select_kmer_max_df,
    validate_kmer_percentage,
)
from omamer.stat_models import (
    FamilyModel,
    FamilyModelParameters,
    bbinom_eval_n,
    beta_binomial_neglogccdf,
    beta_binomial_params_for_n,
    family_expected_count,
    family_log_correction,
    family_neglogccdf,
)
from omamer.compression import ctz, naive_ctz, popcount, select1_in_word
from omamer.compression import to_elias_fano, from_elias_fano
from omamer.merge_search import (
    SEARCH_SEQUENCE,
    SearchScratch,
    make_kmer_index,
    search_seq_kmers,
)


def popcount_naive(x):
    """
    Naive implementation of popcount, i.e. counting
    the number of 1s in the bitwise representation of x
    """
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count

def test_popcount():
    n = 500_000
    data = np.random.randint(0, np.iinfo(np.uint32).max,
                             size=n, dtype=np.uint32)

    for i in range(n):
        x = data[i]
        expected = popcount_naive(x)
        actual = popcount(x)
        assert expected == actual


def test_pmi_kmer_filter_prefers_family_specific_kmers_and_keeps_ties():
    # Four indexed codes have family document frequencies 1, 2, 3 and 1;
    # code 3 is absent.  At 50%, both df=1 codes are retained, while absent
    # codes remain valid because they do not add posting-list work.
    table_idx = np.asarray([0, 1, 3, 6, 6, 7], dtype=np.uint32)
    n_present, n_retained, max_df = select_kmer_max_df(
        table_idx, 6, 50.0
    )
    assert (n_present, n_retained, max_df) == (4, 2, 1)

    # This also verifies that conditional background construction counts only
    # retained postings, not an entry for every possible k-mer code.
    table_buff = np.asarray([0, 1, 2, 0, 1, 2, 2], dtype=np.uint32)
    np.testing.assert_array_equal(
        filtered_hog_kmer_counts(table_idx, table_buff, max_df, 3),
        [1, 0, 1],
    )


@pytest.mark.parametrize("value", [0, -1, 100.1])
def test_kmer_percentage_rejects_out_of_range_values(value):
    with pytest.raises(ValueError, match="kmer_percentage"):
        validate_kmer_percentage(value)


def test_search_uses_build_time_document_frequency_cutoff():
    table_idx = np.asarray([0, 1, 3, 3, 4], dtype=np.uint32)
    table_buff = np.asarray([0, 1, 2, 2], dtype=np.uint32)
    hog_tab = np.asarray([(0,), (1,), (1,)], dtype=[("FamOff", np.uint32)])

    hog_counts = np.zeros(3, dtype=np.uint16)
    fam_counts = np.zeros(2, dtype=np.uint16)
    fam_lowloc = np.full(2, -1, dtype=np.int32)
    fam_highloc = np.full(2, -1, dtype=np.int32)
    hit_fams = np.zeros(2, dtype=np.int32)
    hit_hogs = np.zeros(3, dtype=np.int32)

    index = make_kmer_index(
        table_idx,
        table_buff,
        SEARCH_SEQUENCE,
        1,
    )
    scratch = SearchScratch(
        hit_fams[np.newaxis, :],
        hit_hogs[np.newaxis, :],
        hog_counts[np.newaxis, :],
        fam_counts[np.newaxis, :],
        fam_lowloc[np.newaxis, :],
        fam_highloc[np.newaxis, :],
        np.zeros(1, dtype=np.uint32),
        np.zeros(1, dtype=np.uint32),
    )
    n_skipped = search_seq_kmers(
        np.asarray([0, 1, 2, 3], dtype=np.uint32),
        np.asarray([0, 1, 2, 3], dtype=np.uint32),
        hog_tab,
        np.uint32(4),
        index,
        scratch,
        0,
    )
    n_fams = scratch.num_hit_families[0]
    n_hogs = scratch.num_hit_hogs[0]

    # Code 1 has df=2 and is filtered. The absent code 2 remains a trial but
    # has no postings, matching the beta-binomial fitting convention.
    assert (n_fams, n_hogs, n_skipped) == (2, 2, 1)
    np.testing.assert_array_equal(hog_counts, [1, 0, 1])
    np.testing.assert_array_equal(fam_counts, [1, 1])


def test_bbinom_fitting_uses_build_time_document_frequency_cutoff():
    alphabet = Alphabet(n=21)
    dfs = np.zeros(21, dtype=np.uint32)
    # A, N and R have encoded values 0, 11 and 14. R is filtered at df=2.
    dfs[[0, 11, 14]] = [1, 1, 2]
    table_idx = np.zeros(22, dtype=np.uint32)
    table_idx[1:] = np.cumsum(dfs)

    codes = _unique_valid_codes(
        "ARN",
        1,
        get_transform(1, alphabet.DIGITS_AA),
        alphabet.DIGITS_AA_LOOKUP,
        21,
        table_idx,
        1,
    )
    np.testing.assert_array_equal(codes, [0, 11])


@numba.njit
def select1_in_word_naive(word, rank):
    count = 0
    for bit in range(32):
        if (word >> bit) & 1:
            if count == rank:
                return bit
            count += 1
    return -1


def test_select1_in_word():
    n = 100_000
    for _ in range(n):
        word = np.random.randint(0, 2**32, dtype=np.uint32)
        pop = bin(word).count("1")
        for r in range(pop):
            expected = select1_in_word_naive(word, r)
            actual = select1_in_word(word, r)
            assert expected == actual, f"FAIL: word={bin(word)}, rank={r}, expected={expected}, got={actual}"

        # Check rank overflow case
        assert select1_in_word(word, pop) == -1, f"FAIL: rank={pop} should be invalid"


def test_ctz():
    seed = 42
    num_tests = 10000
    np.random.seed(seed)

    for _ in range(num_tests):
        v = np.random.randint(0, 2**32, dtype=np.uint32)
        expected = naive_ctz(v)
        actual = ctz(v)
        assert expected == actual, f"Error for value {v}"


def test_elias_fano(num_tests=1000, max_len=200, max_val=10**6):
    for _ in range(num_tests):
        length = np.random.randint(1, max_len + 1)
        values = np.sort(np.random.choice(
            np.arange(max_val, dtype=np.uint32), size=length, replace=False))

        ef = to_elias_fano(values)

        # assert np.array_equal(from_elias_fano_correct(ef.l,
        #     ef.lower_packed,
        #     ef.upper_packed,
        #     ef.n), values)

        decoded = from_elias_fano(
            ef.l,
            ef.lower_packed,
            ef.upper_packed,
            ef.n
        )

        assert np.array_equal(decoded, values), f"FAILED:\nOriginal: {values}\nDecoded: {decoded}"



def test_decode_structure_sequence_from_uint8_ascii():
    seq = np.frombuffer(b"DEAEA ", dtype=np.uint8)
    decoded = DatabaseFromOMABrowser._decode_structure_sequence(seq)
    assert decoded == "DEAEA "


def test_normalise_structure_sequence_from_uint8_ascii():
    seq = np.frombuffer(b"DEAEA ", dtype=np.uint8)
    norm = DatabaseFromOMABrowser._normalise_structure_sequence(seq)
    assert bytes(norm).decode("ascii") == "DEAEA "


def test_normalise_structure_sequence_preserves_existing_string_path():
    seq = np.frombuffer(
        (Alphabet(n=21).sanitise_seq("MREIVL") + " ").encode("ascii"),
        dtype="S1",
    )
    norm = DatabaseFromOMABrowser._normalise_structure_sequence(seq)
    assert bytes(norm).decode("ascii") == "MREIVL "


def test_beta_binomial_tail_matches_scipy():
    n = 37
    alpha = 2.5
    beta = 8.0
    for x in [1, 5, 12, 25, 37]:
        actual = beta_binomial_neglogccdf(x, n, alpha, beta)
        expected = -np.log(betabinom.sf(x - 1, n, alpha, beta))
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_beta_binomial_length_aware_parameter_reconstruction():
    n = 100
    q_coef = np.asarray([0.0, 0.1, -0.05], dtype=np.float64)
    kappa_coef = np.asarray([np.log(20.0), 0.25], dtype=np.float64)
    center = np.log(float(n))
    scale = 1.0

    alpha, beta, q = beta_binomial_params_for_n(n, q_coef, kappa_coef, center, scale)

    np.testing.assert_allclose(q, 0.5)
    np.testing.assert_allclose(alpha, 10.0)
    np.testing.assert_allclose(beta, 10.0)


def test_choose_sequence_n_values_selects_eligible_buckets():
    n_unique = np.asarray([5] * 3 + [10] * 5 + [20] * 5 + [30] * 5, dtype=np.uint32)
    summary, selected = choose_sequence_n_values(n_unique, min_records_per_n=5, n_buckets=2)

    assert selected.size == 2
    assert set(selected.tolist()).issubset({10, 20, 30})
    assert not bool(summary.loc[summary["n_unique"].eq(5), "eligible"].iloc[0])


def test_count_selected_family_hits_counts_query_hits():
    codes = np.asarray([0, 1, 2], dtype=np.uint32)
    table_idx = np.asarray([0, 2, 3, 5], dtype=np.uint32)
    table_buff = np.asarray([0, 1, 2, 1, 3], dtype=np.uint32)
    hog_to_family = np.asarray([0, 1, 1, 2], dtype=np.uint32)
    target_rank = np.asarray([-1, 0, 1], dtype=np.int32)
    counts = np.zeros(2, dtype=np.uint32)
    touched = np.empty(2, dtype=np.int32)

    n_touched = count_selected_family_hits(
        codes,
        table_idx,
        table_buff,
        hog_to_family,
        target_rank,
        counts,
        touched,
    )

    assert n_touched == 2
    observed = {int(touched[i]): int(counts[touched[i]]) for i in range(n_touched)}
    assert observed == {0: 3, 1: 1}


def test_fit_family_bbinom_from_hist_returns_importable_row():
    n_query_counts = {20: 80, 40: 80, 80: 80}
    records = [
        (20, 1, 10),
        (20, 2, 8),
        (40, 2, 10),
        (40, 3, 9),
        (80, 4, 11),
        (80, 5, 7),
    ]

    row = fit_family_bbinom_from_hist(7, records, n_query_counts)

    assert row is not None
    assert row["family_offset"] == 7
    assert row["modality"] == "seq"
    assert row["kmer_percentage"] == 100.0
    assert row["q_degree"] == 2
    assert row["kappa_degree"] == 1
    for key in ["q_coef_0", "q_coef_1", "q_coef_2", "kappa_coef_0", "kappa_coef_1"]:
        assert np.isfinite(row[key])
    assert "fit_valid" in row


def test_failed_bbinom_optimizer_row_is_not_valid(monkeypatch):
    def failed_minimize(objective, theta0, method, bounds, jac):
        return SimpleNamespace(
            success=False,
            x=np.full_like(theta0, 3.0),
            fun=123.0,
            message="iteration limit",
            status=1,
            nit=5,
            nfev=10,
        )

    monkeypatch.setattr("omamer.bbinom_fit.minimize", failed_minimize)
    row = fit_family_bbinom_from_hist(
        7,
        [(20, 1, 10), (40, 2, 10)],
        {20: 80, 40: 80},
    )

    assert row["optimizer_success"] is False
    assert row["fit_valid"] is False
    assert "optimizer_failed" in row["fit_invalid_reason"].split(";")
    # Failed fits retain a diagnostic row, but use the explicit initial
    # coefficients rather than the optimizer's untrusted iterate.
    np.testing.assert_allclose(row["q_coef_1"], 0.0)
    np.testing.assert_allclose(row["kappa_coef_0"], np.log(1000.0))


def test_bbinom_valid_mask_rejects_rails_and_legacy_failed_initial_fit():
    valid = np.ones(3, dtype=bool)
    q_coef = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [50.0, 0.2, 0.3],
            [-2.0, 0.0, 0.0],
        ]
    )
    kappa_coef = np.asarray(
        [
            [2.0, 0.4],
            [2.0, 0.4],
            [np.log(1000.0), 0.0],
        ]
    )
    center = np.ones(3)
    scale = np.ones(3)

    np.testing.assert_array_equal(
        bbinom_coefficient_valid_mask(
            valid,
            q_coef,
            kappa_coef,
            center,
            scale,
            reject_initial_fallback=True,
        ),
        [True, False, False],
    )


def test_dense_family_histogram_reconstructs_nonzero_records():
    # N=2 occupies columns 0..2 and N=4 occupies columns 3..7.
    data = np.zeros((2, 8), dtype=np.uint16)
    data[0, 1] = 3
    data[0, 2] = 2
    data[0, 3 + 2] = 5
    data[1, 3 + 4] = 7
    hist = DenseFamilyHitHistogram(
        data,
        n_values=np.asarray([2, 4]),
        offsets=np.asarray([0, 3, 8]),
    )

    assert hist.records(0) == [(2, 1, 3), (2, 2, 2), (4, 2, 5)]
    assert hist.records(1) == [(4, 4, 7)]


def test_family_log_correction_policy():
    assert family_log_correction("bonferroni", 100) == pytest.approx(
        np.log(100)
    )
    assert family_log_correction("none", 100) == 0.0
    with pytest.raises(ValueError, match="family_correction"):
        family_log_correction("unknown", 100)


def test_bbinom_exact_gradient_matches_finite_difference():
    ns = np.asarray([20.0, 20.0, 40.0, 40.0, 80.0, 80.0])
    xs = np.asarray([0.0, 2.0, 1.0, 7.0, 5.0, 15.0])
    weights = np.asarray([50.0, 10.0, 45.0, 15.0, 40.0, 20.0])
    center = np.log(40.0)
    scale = 0.8
    xq = np.column_stack(
        [
            np.ones(ns.size),
            (np.log(ns) - center) / scale,
            ((np.log(ns) - center) / scale) ** 2,
        ]
    )
    xk = xq[:, :2]
    theta = np.asarray([-2.0, 0.2, -0.1, np.log(20.0), 0.1])

    def objective(value):
        return _bbinom_objective_and_gradient(
            value,
            xq,
            xk,
            ns,
            xs,
            weights,
        )[0]

    def gradient(value):
        return _bbinom_objective_and_gradient(
            value,
            xq,
            xk,
            ns,
            xs,
            weights,
        )[1]

    assert check_grad(objective, gradient, theta) < 1e-3


def test_fit_family_bbinom_from_hist_respects_modality():
    n_query_counts = {20: 80, 40: 80, 80: 80}
    records = [(20, 1, 10), (40, 2, 10), (80, 4, 11)]

    seq_row = fit_family_bbinom_from_hist(7, records, n_query_counts, modality="seq")
    ss_row = fit_family_bbinom_from_hist(7, records, n_query_counts, modality="ss")

    assert seq_row["modality"] == "seq"
    assert ss_row["modality"] == "ss"
    # Coefficients only depend on the data, not the modality label.
    np.testing.assert_allclose(seq_row["q_coef_0"], ss_row["q_coef_0"])


def test_modality_index_arrays_selects_structure_or_raises():
    from omamer.bbinom_fit import _modality_index_arrays, _modality_family_probability

    class FakeDB:
        def __init__(self, has_ss):
            self._has_ss = has_ss
            self._db_Index_TableIndex = np.asarray([0, 1], dtype=np.uint32)
            self._db_Index_TableBuffer = np.asarray([0], dtype=np.uint32)
            self._db_Index_SSTableIndex = np.asarray([0, 2], dtype=np.uint32)
            self._db_Index_SSTableBuffer = np.asarray([0, 0], dtype=np.uint32)
            self._db_Index_FamilyProbability = np.asarray([0.1, 0.2], dtype=np.float64)
            self._db_Index_SSFamilyProbability = np.asarray([0.3, 0.4], dtype=np.float64)

        def has_structure(self):
            return self._has_ss

    db = FakeDB(has_ss=True)
    seq_idx, seq_buff = _modality_index_arrays(db, "seq")
    ss_idx, ss_buff = _modality_index_arrays(db, "ss")
    assert seq_idx[1] == 1 and ss_idx[1] == 2
    np.testing.assert_array_equal(_modality_family_probability(db, "ss"), [0.3, 0.4])

    with pytest.raises(ValueError):
        _modality_index_arrays(db, "bogus")

    no_ss = FakeDB(has_ss=False)
    with pytest.raises(ValueError):
        _modality_index_arrays(no_ss, "ss")


def test_bbinom_eval_n_clamps_to_trained_range():
    n_min = np.asarray([0, 45], dtype=np.uint32)
    n_max = np.asarray([0, 1096], dtype=np.uint32)

    # family 1 has a trained range [45, 1096].
    assert bbinom_eval_n(1, 30, n_min, n_max) == 45      # below -> clamp up
    assert bbinom_eval_n(1, 500, n_min, n_max) == 500    # inside -> unchanged
    assert bbinom_eval_n(1, 5000, n_min, n_max) == 1096  # above -> clamp down
    # family 0 has no range (n_max == 0) -> never clamp.
    assert bbinom_eval_n(0, 5000, n_min, n_max) == 5000


def test_family_neglogccdf_clamps_out_of_range_n():
    q_coef = np.asarray([[0.0, 0.1, -0.05]], dtype=np.float64)
    kappa_coef = np.asarray([[np.log(20.0), 0.25]], dtype=np.float64)
    center = np.asarray([np.log(100.0)], dtype=np.float64)
    scale = np.asarray([1.0], dtype=np.float64)
    valid = np.asarray([True])
    ref = np.asarray([0.01])
    n_min = np.asarray([50], dtype=np.uint32)
    n_max = np.asarray([200], dtype=np.uint32)
    model = FamilyModelParameters(
        FamilyModel.BETA_BINOMIAL,
        ref,
        q_coef,
        kappa_coef,
        center,
        scale,
        valid,
        n_min,
        n_max,
    )

    # Query N=400 is above the trained max -> q/kappa evaluated at N=200, but the
    # Beta-Binomial tail and count still use the actual N=400. This must equal
    # scoring an in-range query of N=200 evaluated at the same boundary.
    out_of_range = family_neglogccdf(0, 30, 400, model)
    at_boundary = family_neglogccdf(0, 30, 200, model)
    alpha, beta, q = beta_binomial_params_for_n(200, q_coef[0], kappa_coef[0], center[0], scale[0])
    expected = beta_binomial_neglogccdf(30, 400, alpha, beta)
    np.testing.assert_allclose(out_of_range, expected, rtol=1e-9)
    # The boundary (in-range) query uses N=200 for the tail, so it differs.
    assert not np.isclose(out_of_range, at_boundary)

    # Expected count uses clamped q but actual N.
    ec = family_expected_count(0, 400, model)
    np.testing.assert_allclose(ec, q * 400, rtol=1e-9)
