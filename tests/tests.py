import numpy as np
import numba
import pytest
import tables
from scipy.stats import betabinom
from omamer.alphabets import Alphabet
from omamer.bbinom_coefficients import import_bbinom_coefficients
from omamer.bbinom_fit import (
    choose_sequence_n_values,
    count_selected_family_hits,
    fit_family_bbinom_from_hist,
)
from omamer.database import DatabaseFromOMABrowser
from omamer.stat_models import beta_binomial_neglogccdf, beta_binomial_params_for_n
from omamer.compression import ctz, naive_ctz, popcount, select1_in_word
from omamer.compression import to_elias_fano, from_elias_fano
from omamer.merge_search import family_result_sort


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



dtype = [("normcount", float), ("overlap", float), ("pvalue", float)]

def generate_random_data(seed, size=100):
    rng = np.random.default_rng(seed)
    data = np.zeros(size, dtype=dtype)
    data['normcount'] = rng.random(size) * 100
    data['overlap'] = rng.random(size) * 10
    data['pvalue'] = rng.random(size)
    return data


def naive_sort(arr, k=None):
    """
    Sort by descending normcount, then overlap, then pvalue.
    """
    # Convert to list of numpy records (so x['field'] works)
    records = list(arr)
    # Sort using tuple key, reverse for descending
    sorted_list = sorted(
        records,
        key=lambda x: (x['normcount'], x['overlap'], x['pvalue']),
        reverse=True
    )
    if k is not None:
        sorted_list = sorted_list[:k]
    # Convert back to structured array
    return np.array(sorted_list, dtype=dtype)


def assert_structs_close(a, b, atol=1e-8):
    """
    Compare two structured arrays field-by-field with tolerance.
    """
    for field in ['normcount', 'overlap', 'pvalue']:
        np.testing.assert_allclose(a[field], b[field], atol=atol,
                                   err_msg=f"Field '{field}' differs")

@pytest.mark.parametrize("seed", list(range(100)))
def test_family_sort(seed):
    x = generate_random_data(seed)
    # Full sort
    sorted_auto = family_result_sort(x, k=len(x))
    sorted_naive = naive_sort(x)
    assert_structs_close(sorted_auto, sorted_naive)

    # Top-k selection
    k = 10
    sorted_auto_k = family_result_sort(x, k=k)
    sorted_naive_k = naive_sort(x, k)
    assert_structs_close(sorted_auto_k, sorted_naive_k)


@pytest.mark.parametrize("seed", list(range(100)))
def test_ties(seed):
    random_data = generate_random_data(seed)
    # Make a tie with the best record
    sorted_full = naive_sort(random_data)
    sorted_full[-1]["normcount"] = sorted_full[0]["normcount"]
    sorted_full[-1]["overlap"] = sorted_full[0]["overlap"] + 1
    np.random.shuffle(sorted_full)
    random_data = sorted_full

    k = 10
    sorted_auto_k = family_result_sort(random_data, k=k)
    sorted_naive_k = naive_sort(random_data, k=k)
    assert_structs_close(sorted_auto_k, sorted_naive_k)

    # Make a tie by the 2nd parameter
    sorted_full = naive_sort(random_data)
    sorted_full[-1]["normcount"] = sorted_full[0]["normcount"]
    sorted_full[-1]["overlap"] = sorted_full[0]["overlap"]
    sorted_full[-1]["pvalue"] = sorted_full[0]["pvalue"] + 1
    np.random.shuffle(sorted_full)
    random_data = sorted_full

    sorted_auto_k = family_result_sort(random_data, k=k)
    sorted_naive_k = naive_sort(random_data, k=k)
    assert_structs_close(sorted_auto_k, sorted_naive_k)


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


def test_import_bbinom_coefficients_writes_modality_arrays(tmp_path):
    db_path = tmp_path / "test.h5"
    coeff_path = tmp_path / "coefficients.tsv"
    coeff_path.write_text(
        "\t".join(
            [
                "family_offset",
                "modality",
                "log_n_center",
                "log_n_scale",
                "q_coef_0",
                "q_coef_1",
                "q_coef_2",
                "kappa_coef_0",
                "kappa_coef_1",
                "n_train_min",
                "n_train_max",
            ]
        )
        + "\n"
        + "1\tseq\t4.0\t1.5\t0.1\t0.2\t0.3\t2.0\t0.4\t52\t277\n"
        + "2\tss\t4.1\t1.6\t0.5\t0.6\t0.7\t3.0\t0.8\t52\t277\n"
    )

    family_descr = {"ID": tables.UInt32Col()}
    filters = tables.Filters(complevel=0)
    with tables.open_file(db_path, "w") as h5:
        fam = h5.create_table("/", "Family", family_descr)
        for i in range(3):
            row = fam.row
            row["ID"] = i
            row.append()
        fam.flush()
        h5.create_group("/", "Index")

    class FakeDB:
        def __init__(self, h5):
            self.db = h5
            self.compression_filters = filters

        @property
        def family_table(self):
            return self.db.root.Family

    with tables.open_file(db_path, "a") as h5:
        written = import_bbinom_coefficients(FakeDB(h5), coeff_path)
        assert written == {"seq": 1, "ss": 1}
        assert h5.root.Index._v_attrs["bbinom_model"] == "length_aware_beta_binomial"
        np.testing.assert_array_equal(h5.root.Index.FamilyBBinomValid[:], [False, True, False])
        np.testing.assert_array_equal(h5.root.Index.SSFamilyBBinomValid[:], [False, False, True])
        np.testing.assert_allclose(h5.root.Index.FamilyBBinomQCoef[1], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(h5.root.Index.SSFamilyBBinomKappaCoef[2], [3.0, 0.8])


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
    assert row["q_degree"] == 2
    assert row["kappa_degree"] == 1
    for key in ["q_coef_0", "q_coef_1", "q_coef_2", "kappa_coef_0", "kappa_coef_1"]:
        assert np.isfinite(row[key])


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
