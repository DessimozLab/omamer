"""
    OMAmer - tree-driven and alignment-free protein assignment to sub-families

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
import functools

import numba
import numpy as np

_EMPTY_MASK = np.zeros(0, dtype=np.bool_)


@numba.njit(cache=True, nogil=True)
def _count_distinct_below(k, alphabet_n, min_distinct):
    n_codes = alphabet_n**k
    mask = np.empty(n_codes, dtype=np.bool_)
    # the digits of the code, least significant first, counted up as an
    # odometer (no division per code)
    digits = np.zeros(k, dtype=np.int64)
    for code in range(n_codes):
        seen = np.uint64(0)
        distinct = 0
        for i in range(k):
            bit = np.uint64(1) << np.uint64(digits[i])
            if not (seen & bit):
                seen |= bit
                distinct += 1
        mask[code] = distinct < min_distinct
        i = 0
        while i < k:
            digits[i] += 1
            if digits[i] < alphabet_n:
                break
            digits[i] = 0
            i += 1
    return mask


@functools.lru_cache(maxsize=4)
def low_complexity_mask(k, alphabet_n, min_distinct):
    """
    Boolean array over all k-mer codes: True for k-mers with fewer than
    min_distinct different letters (e.g. 'DDDDDD' or 'VLVLVL' for
    min_distinct=3), which are ignored at search. Codes are base-alphabet_n
    numbers of k digits (alphabet_n <= 64). Empty (no mask) when
    min_distinct is None or <= 1.
    """
    if min_distinct is None or int(min_distinct) <= 1:
        return _EMPTY_MASK
    return _count_distinct_below(int(k), int(alphabet_n), int(min_distinct))


def stored_min_distinct(index_attrs, modality):
    """min_distinct stored in the /Index attributes for a modality, or None."""
    name = "{}_kmer_min_distinct".format(modality)
    return int(index_attrs[name]) if name in index_attrs._f_list() else None
