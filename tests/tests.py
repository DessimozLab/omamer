import numpy as np
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
