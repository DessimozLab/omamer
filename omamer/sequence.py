"""
    OMAmer - tree-driven and alignment-free protein assignment to sub-families

    (C) 2024-2025 Nikolai Romashchenko <nikolai.romashchenko@unil.ch>
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
from numba.typed import List


@numba.njit
def custom_unique1d(ar):
    """
    adapted from np._unique1d for numba
    """
    perm = ar.argsort(kind="mergesort")  # if return_index else 'quicksort')
    aux = ar[perm]

    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]

    idx = np.concatenate((np.nonzero(mask)[0], np.array([mask.size])))

    return aux[mask], perm[mask], np.diff(idx)


@numba.njit
def unique1d_linear(array):
    """
    Find a set of unique elements in linear time
    """
    unique_list = List()
    index_list = List()
    seen = set()

    for i in range(len(array)):
        if array[i] not in seen:
            seen.add(array[i])
            unique_list.append(array[i])
            index_list.append(i)

    return np.asarray(unique_list), np.asarray(index_list, dtype=np.uint32), None


@numba.njit
def seq_to_kmers(s, DIGITS_AA_LOOKUP, k, trans, x_flag):
    """
    get the sequence unique k-mers and non ambiguous locations (when truly unique)
    """
    n_kmers = len(s) - k + 1

    s_norm = DIGITS_AA_LOOKUP[s]
    r = np.zeros(n_kmers, dtype=np.uint32)

    # compute the code of the first k-mer
    for j in range(k):
        r[0] += trans[j] * s_norm[j]

    # does k-mer contain any X?
    x_seen = np.any(s_norm[0:k] == DIGITS_AA_LOOKUP[88])
    # if yes, replace it by the x_flag
    r[0] = r[0] if not x_seen else x_flag

    # codes for other k-mers
    for i in range(1, n_kmers):
        if not x_seen:
            # if the previous k-mer was valid,
            # recompute the current code from the previous one

            # remove the first character from the code
            shared = r[i-1] - (trans[0] * s_norm[i - 1])

            # trans[-2] is the alphabet size
            r[i] = shared * trans[-2] + trans[-1] * s_norm[i + k - 1]

        else:
            # if the previous k-mer has Xs,
            # just compute the code from scratch
            for j in range(k):
                r[i] += trans[j] * s_norm[i + j]

        x_seen = np.any(s_norm[i: i + k] == DIGITS_AA_LOOKUP[88])
        r[i] = r[i] if not x_seen else x_flag

    return unique1d_linear(r)
