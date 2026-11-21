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
import os

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
    min_distinct=3). Codes are base-alphabet_n numbers of k digits.
    """
    if int(alphabet_n) <= 64:
        # one pass over the codes, letters as bits (~30x faster than below)
        return _count_distinct_below(int(k), int(alphabet_n), int(min_distinct))
    n_codes = int(alphabet_n) ** int(k)
    codes = np.arange(n_codes, dtype=np.int64)
    digits = np.empty((n_codes, int(k)), dtype=np.uint8)
    for i in range(int(k)):
        digits[:, i] = codes % alphabet_n
        codes //= alphabet_n
    del codes
    digits.sort(axis=1)
    distinct = 1 + np.count_nonzero(np.diff(digits, axis=1), axis=1)
    return distinct < int(min_distinct)


def modality_kmer_mask(modality, k, alphabet_n, min_distinct=None):
    """
    Mask of low-complexity k-mers of a modality: k-mers with fewer than
    min_distinct different letters are ignored at search and in the null
    training of the beta-binomial. min_distinct comes from the database (stored
    by mkdb / refit-bbinom); when it is None, the environment variables
    OMAMER_SS_MIN_DISTINCT (3Di) and OMAMER_SEQ_MIN_DISTINCT (sequences) are used.
    Returns an empty array when masking is off.
    """
    if min_distinct is None:
        variable = {"ss": "OMAMER_SS_MIN_DISTINCT", "seq": "OMAMER_SEQ_MIN_DISTINCT"}.get(modality)
        if variable is None:
            return _EMPTY_MASK
        min_distinct = int(os.environ.get(variable, "0") or 0)
    if int(min_distinct) <= 1:
        return _EMPTY_MASK
    return low_complexity_mask(int(k), int(alphabet_n), int(min_distinct))


def stored_min_distinct(index_attrs, modality):
    """min_distinct stored in the /Index attributes for a modality, or None."""
    name = "{}_kmer_min_distinct".format(modality)
    return int(index_attrs[name]) if name in index_attrs._f_list() else None
