import numpy as np
import pytest
from omamer.family_sort import (
    FAMILY_SORT_NORMCOUNT,
    FAMILY_SORT_PVALUE,
    family_result_sort,
    resolve_family_sorting,
)


dtype = [("normcount", float), ("overlap", float), ("pvalue", float)]

SORTING_POLICIES = (
    ("normcount", FAMILY_SORT_NORMCOUNT, ("normcount", "overlap", "pvalue")),
    ("pvalue", FAMILY_SORT_PVALUE, ("pvalue", "overlap", "normcount")),
)


def naive_sort(arr, fields, k=None):
    """
    Sort by the requested descending field order.
    """
    # Convert to list of numpy records (so x['field'] works)
    records = list(arr)
    # Sort using tuple key, reverse for descending
    sorted_list = sorted(
        records,
        key=lambda x: tuple(x[field] for field in fields),
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


def generate_random_data(seed, size=100):
    rng = np.random.default_rng(seed)
    data = np.zeros(size, dtype=dtype)
    data['normcount'] = rng.random(size) * 100
    data['overlap'] = rng.random(size) * 10
    data['pvalue'] = rng.random(size)
    return data


@pytest.mark.parametrize(("policy", "sorting", "fields"), SORTING_POLICIES)
@pytest.mark.parametrize("seed", list(range(100)))
def test_family_sort(seed, policy, sorting, fields):
    assert resolve_family_sorting(policy) == sorting
    x = generate_random_data(seed)
    # Full sort
    sorted_auto = family_result_sort(x, k=len(x), sorting=sorting)
    sorted_naive = naive_sort(x, fields)
    assert_structs_close(sorted_auto, sorted_naive)

    # Top-k selection
    k = 10
    sorted_auto_k = family_result_sort(x, k=k, sorting=sorting)
    sorted_naive_k = naive_sort(x, fields, k)
    assert_structs_close(sorted_auto_k, sorted_naive_k)


@pytest.mark.parametrize(("policy", "sorting", "fields"), SORTING_POLICIES)
@pytest.mark.parametrize("seed", list(range(100)))
def test_ties(seed, policy, sorting, fields):
    random_data = generate_random_data(seed)
    primary, secondary, tertiary = fields
    # Make a tie with the best record
    sorted_full = naive_sort(random_data, fields)
    sorted_full[-1][primary] = sorted_full[0][primary]
    sorted_full[-1][secondary] = sorted_full[0][secondary] + 1
    np.random.shuffle(sorted_full)
    random_data = sorted_full

    k = 10
    sorted_auto_k = family_result_sort(random_data, k=k, sorting=sorting)
    sorted_naive_k = naive_sort(random_data, fields, k=k)
    assert_structs_close(sorted_auto_k, sorted_naive_k)

    # Make a tie by the 2nd parameter
    sorted_full = naive_sort(random_data, fields)
    sorted_full[-1][primary] = sorted_full[0][primary]
    sorted_full[-1][secondary] = sorted_full[0][secondary]
    sorted_full[-1][tertiary] = sorted_full[0][tertiary] + 1
    np.random.shuffle(sorted_full)
    random_data = sorted_full

    sorted_auto_k = family_result_sort(random_data, k=k, sorting=sorting)
    sorted_naive_k = naive_sort(random_data, fields, k=k)
    assert_structs_close(sorted_auto_k, sorted_naive_k)


def test_resolve_family_sorting():
    assert resolve_family_sorting("normcount") == FAMILY_SORT_NORMCOUNT
    assert resolve_family_sorting("pvalue") == FAMILY_SORT_PVALUE
    with pytest.raises(ValueError, match="family_sorting"):
        resolve_family_sorting("unknown")
