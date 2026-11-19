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

import numpy as np

_EMPTY_MASK = np.zeros(0, dtype=np.bool_)


@functools.lru_cache(maxsize=4)
def low_complexity_mask(k, alphabet_n, min_distinct):
    """
    Boolean array over all k-mer codes: True for k-mers with fewer than
    min_distinct different letters (e.g. 'DDDDDD' or 'VLVLVL' for
    min_distinct=3). Codes are base-alphabet_n numbers of k digits.
    """
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


def modality_kmer_mask(modality, k, alphabet_n):
    """
    Experimental (model audit): mask of low-complexity 3Di k-mers, enabled with
    the environment variable OMAMER_SS_MIN_DISTINCT=<m> (k-mers with fewer than
    m distinct letters are ignored at search and in the null training).
    Returns an empty array when masking is off.
    """
    if modality != "ss":
        return _EMPTY_MASK
    m = int(os.environ.get("OMAMER_SS_MIN_DISTINCT", "0") or 0)
    if m <= 1:
        return _EMPTY_MASK
    return low_complexity_mask(int(k), int(alphabet_n), m)
