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
