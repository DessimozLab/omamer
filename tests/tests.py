import numpy as np
<<<<<<< HEAD
import numba

<<<<<<< Updated upstream
from omamer.index import popcount, select1_in_word
=======
from omamer.index import ctz, naive_ctz, popcount, select1_in_word
from omamer.index import to_elias_fano, from_elias_fano, from_elias_fano_correct
>>>>>>> Stashed changes


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
>>>>>>> Stashed changes
=======
import pytest
from omamer.merge_search import (
    family_result_sort,
)

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
>>>>>>> orthoxml
