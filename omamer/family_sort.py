"""
    OMAmer - tree-driven and alignment-free protein assignment to sub-families

    (C) 2024-present Nikolai Romashchenko <nikolai.romashchenko@unil.ch>
    (C) 2022-2023 Alex Warwick Vesztrocy <alex.warwickvesztrocy@unil.ch>
    (C) 2019-2021 Victor Rossier <victor.rossier@unil.ch> and
                  Alex Warwick Vesztrocy <alex@warwickvesztrocy.co.uk>

    This file is part of OMAmer.

    OMAmer is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OMAmer is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with OMAmer. If not, see <http://www.gnu.org/licenses/>.
"""

import numba
import numpy as np


@numba.njit(nogil=True)
def fam_res_compare(x1, x2):
    """
    Compare two family results and order them according to normcount, overlap and pvalue.
    """
    # normalised count
    if x1["normcount"] != x2["normcount"]:
        # greater first
        return -1 if (x1["normcount"] > x2["normcount"]) else 1
    else:
        if x1["overlap"] != x2["overlap"]:
            # greater first
            return -1 if (x1["overlap"] > x2["overlap"]) else 1
        else:
            if x1["pvalue"] != x2["pvalue"]:
                # greater first. note, we use neglog units
                return -1 if (x1["pvalue"] > x2["pvalue"]) else 1
    # equal. take whichever.
    return 0


@numba.njit(nogil=True)
def fam_res_less(x1, x2):
    """
    Same as fam_res_compare, but operates as '<'
    """
    if x1["normcount"] != x2["normcount"]:
        return x1["normcount"] < x2["normcount"]

    if x1["overlap"] != x2["overlap"]:
        return x1["overlap"] < x2["overlap"]

    if x1["pvalue"] != x2["pvalue"]:
        return x1["pvalue"] < x2["pvalue"]

    return False


@numba.njit(nogil=True)
def fam_res_le(x1, x2):
    """
    Same as fam_res_compare, but operates as '<='
    """
    if not fam_res_less(x1, x2):
        return x1["normcount"] == x2["normcount"] and \
            x1["overlap"] == x2["overlap"] and \
            x1["pvalue"] == x2["pvalue"]

    return True


@numba.njit(nogil=True)
def fam_res_greater(x1, x2):
    """
    Same as fam_res_compare, but operates as '>'
    """
    return not fam_res_le(x1, x2)


@numba.njit(nogil=True)
def fam_res_ge(x1, x2):
    """
    Same as fam_res_compare, but operates as '>='
    """
    return not fam_res_less(x1, x2)


@numba.njit(nogil=True)
def family_result_argsort(x, ii):
    """
    argsort of family results using defined comparison above.
    uses an implementation of quicksort.
    note: np.argsort DOES NOT support struct type in numba. this code does.
    """
    bfs = []
    pvs = []
    afs = []
    if len(ii) <= 1:
        return ii  # [ii[0]]
    else:
        for i in ii:
            # need to implement the order here.
            j = fam_res_compare(x[i], x[ii[0]])
            if j < 0:
                # LHS of pivot
                bfs.append(i)
            elif j > 0:
                # RHS of pivot
                afs.append(i)
            else:
                # same
                pvs.append(i)

        if len(bfs) > 0:
            bfs = family_result_argsort(x, bfs)
        if len(afs) > 0:
            afs = family_result_argsort(x, afs)

        return bfs + pvs + afs


@numba.njit(nogil=True)
def family_result_sort(x, k):
    """
    Sort the family results according to normcount, overlap and pvalue (to break ties).
    this uses a quicksort implementation as np.argsort does not support struct type in numba.
    """

    # Quickselect top k results
    idx = np.arange(len(x))
    _ = _select(x, idx, k, 0, len(x) - 1)
    x = x[idx[:k]]

    # Now mergesort the selected results, because we need to
    # report them sorted
    idx = family_result_argsort(x, list(range(len(x))))
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = x[idx[i]]
    return y


@numba.njit(nogil=True)
def _swap(array, i, j):
    tmp = array[i]
    array[i] = array[j]
    array[j] = tmp


@numba.njit(nogil=True)
def _partition(x, idx, low, high):
    """
    Index-based version of the partition algorithm of
    quicksort. Juggles indexes of the idx array that
    indexes the x array, considering the range [low, high].
    """
    mid = (low + high) >> 1

    # Use median of three {low, middle, high} as the pivot
    if fam_res_greater(x[idx[mid]], x[idx[low]]):
        _swap(idx, mid, low)
    if fam_res_greater(x[idx[high]], x[idx[mid]]):
        _swap(idx, high, mid)
        if fam_res_greater(x[idx[mid]], x[idx[low]]):
            _swap(idx, low, mid)

    pivot = x[idx[mid]]
    # Put the pivot in the end of the array
    _swap(idx, mid, high)

    # Collect elements that are > pivot in the beginning
    i = low
    for j in range(low, high):
        if fam_res_greater(x[idx[j]], pivot):
            _swap(idx, i, j)
            i += 1

    # All indexes of idx in [0, i) are good now.
    # Let's place the pivot back where it should be
    _swap(idx, i, high)
    return i


@numba.njit(nogil=True)
def _select(x, idx, k, low, high):
    """
    Select the k'th largest element of the x array
    """
    if k >= len(x):
        return len(x)

    i = _partition(x, idx, low, high)
    while i != k:
        if i < k:
            low = i + 1
            i = _partition(x, idx, low, high)
        else:
            high = i - 1
            i = _partition(x, idx, low, high)
    return idx[k]
