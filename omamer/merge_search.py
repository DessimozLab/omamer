"""
    OMAmer - tree-driven and alignment-free protein assignment to sub-families

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
from Rmath4 import pbinom, phyper
from numba.tests.support import captured_stdout
from numba.typed import List, Dict
from property_manager import lazy_property, cached_property
from time import time
import numba
import numpy as np
import pandas as pd


from ._utils import LOG
from .alphabets import get_transform
from .sequence_buffer import SequenceBuffer
from .hierarchy import (
    get_root_leaf_offsets,
    get_hog_member_prots,
    is_taxon_implied,
    get_children,
)
from ._clock import clock, as_seconds


# maximum neglogp to set
MAX_LOGP = 20000.0


@numba.njit(nogil=True)
def binom_neglogccdf(x, n, p):
    """
    Use pbinom from RMath4
    """
    # bdtrc does not support high precision (so for v small p fails)
    # bdtrc supports (float64, long, float64)
    # return -1.0 * np.log(sc.bdtrc(np.float64(x - 1), n, p))
    return -1.0 * pbinom(x - 1, n, p, 0, 1)

    # pbinom(n - 1, m, m/N, 0, 0)
    # phyper(n - 1, m, N - m, m, 0, 0)
    #return -1.0 * phyper(x - 1, n, n/p - n, n, 0, 1)


# ----
# family result sorting
@numba.njit
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


@numba.njit
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


@numba.njit
def fam_res_le(x1, x2):
    """
    Same as fam_res_compare, but operates as '<='
    """
    if not fam_res_less(x1, x2):
        return x1["normcount"] == x2["normcount"] and \
            x1["overlap"] == x2["overlap"] and \
            x1["pvalue"] == x2["pvalue"]

    return True


@numba.njit
def fam_res_greater(x1, x2):
    """
    Same as fam_res_compare, but operates as '>'
    """
    return not fam_res_le(x1, x2)


@numba.njit
def fam_res_ge(x1, x2):
    """
    Same as fam_res_compare, but operates as '>='
    """
    return not fam_res_less(x1, x2)


@numba.njit
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


@numba.njit
def family_result_sort(x, k):
    """
    Sort the family results according to normcount, overlap and pvalue (to break ties).
    this uses a quicksort implementation as np.argsort does not support struct type in numba.
    """
    idx = np.arange(len(x))
    _ = _select(x, idx, k, 0, len(x) - 1)
    return x[idx[:k]]


@numba.njit
def _swap(array, i, j):
    tmp = array[i]
    array[i] = array[j]
    array[j] = tmp


@numba.njit
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


@numba.njit
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


# ----
## generic functions
@numba.njit
def get_fam_hog2parent(fam_ent, hog_tab):
    """
    get HOG parent offsets of a single family
    """
    hog_off = fam_ent["HOGoff"]
    hog2parent_tmp = hog_tab["ParentOff"][hog_off : hog_off + fam_ent["HOGnum"]]
    if hog2parent_tmp.size > 1:
        return np.append(
            np.array([-1], dtype=np.int32), hog2parent_tmp[1:] - np.int32(hog_off)
        )
    else:
        return hog2parent_tmp


@numba.njit
def get_fam_level_offsets(fam_ent, level_arr):
    """
    get HOG level offsets of a single family
    """
    level_off = fam_ent["LevelOff"]
    level_num = fam_ent["LevelNum"]
    fam_level_offsets = level_arr[level_off : np.int32(level_off + level_num + 2)]

    # because specific for a single family, reinitizialize offsets of family levels
    return fam_level_offsets - fam_level_offsets[0]


## search functions
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
def parse_seq(s, DIGITS_AA_LOOKUP, n_kmers, k, trans, x_flag):
    """
    get the sequence unique k-mers and non ambiguous locations (when truly unique)
    """
    s_norm = DIGITS_AA_LOOKUP[s]
    r = np.zeros(n_kmers, dtype=np.uint32)  # max kmer 7

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


#@numba.njit
def search_seq_kmers(r1, p1, hog_tab, fam_tab, x_flag, table_idx, table_buff,
                     hog_counts, fam_counts, fam_lowloc, fam_highloc, ref_fam_prob,
                     hit_fams, num_hit_fams, hit_hogs, num_hit_hogs):
    """
    Perform the kmer search, using the index.
    """
    # Reinitialize counters modified by the previous call
    for i in range(num_hit_fams):
        family = hit_fams[i]
        fam_counts[family] = 0
        fam_lowloc[family] = -1
        fam_highloc[family] = -1

    for i in range(num_hit_hogs):
        hog = hit_hogs[i]
        hog_counts[hog] = 0

    num_hit_fams = 0
    num_hit_hogs = 0

    n = r1.shape[0]

    #c = np.zeros((fam_tab.size,), dtype=np.int32)
    c_max = 0

    # Look at the largest k-mer to have an estimate of
    # the # of candidate families needed
    max_candidates = 1
    for i in range(r1.shape[0] - 1, 0, -1):
        kmer = r1[i]
        if kmer == x_flag:
            continue

        x = table_idx[kmer: kmer + 2]
        hogs = table_buff[x[0]: x[1]]
        fams = hog_tab["FamOff"][hogs]

        max_candidates = int(np.ceil(np.log(len(fams))))
        break

    # Count frequency
    f = np.zeros((n,), dtype=np.int32)
    f_nc = np.zeros((n,), dtype=np.int32)
    f_candidates = np.zeros((max_candidates,), dtype=np.uint32)
    num_candidates = 0

    nc = np.zeros((fam_tab.size,), dtype=np.float32)
    nc_max = -np.inf
    #nc_opt = np.zeros((fam_tab.size,), dtype=np.float32)

    f_left_border = 0
    thorough = False
    middle = False

    mu = ref_fam_prob * n
    x = 1

    # iterate unique k-mers
    for i in range(r1.shape[0]):
        kmer = r1[i]
        loc = p1[i]

        # to ignore k-mers with X
        if kmer == x_flag:
            continue

        # get mapping to HOGs
        x = table_idx[kmer: kmer + 2]
        hogs = table_buff[x[0]: x[1]]
        fams = hog_tab["FamOff"][hogs]

        # Last stage
        if thorough and not middle:
            # Binary search candidates in the k-mer family list,
            # place them as usual
            for fam_off in f_candidates[:num_candidates]:
                idx = np.searchsorted(fams, fam_off)
                if idx < len(fams) and fams[idx] == fam_off:
                    hog = hogs[idx]
                    if not hog_counts[hog]:
                        hit_hogs[num_hit_hogs] = hog
                        num_hit_hogs += 1

                    hog_counts[hog] += 1

                    if not fam_counts[fam_off]:
                        hit_fams[num_hit_fams] = fam_off
                        num_hit_fams += 1

                    fam_counts[fam_off] += 1

                    # initiate first location
                    if fam_lowloc[fam_off] == -1:
                        fam_lowloc[fam_off] = loc
                        fam_highloc[fam_off] = loc

                    # update either lower or higher boundary
                    elif loc < fam_lowloc[fam_off]:
                        fam_lowloc[fam_off] = loc
                    elif loc > fam_highloc[fam_off]:
                        fam_highloc[fam_off] = loc

        else:
            for hog in hogs:
                if not hog_counts[hog]:
                    hit_hogs[num_hit_hogs] = hog
                    num_hit_hogs += 1

                hog_counts[hog] += 1

            for fam_off in fams:
                p = ref_fam_prob[fam_off]
                if not fam_counts[fam_off]:
                    hit_fams[num_hit_fams] = fam_off
                    num_hit_fams += 1

                # Descretized normcount
                if fam_counts[fam_off] > 0:
                    inc = - int(np.ceil((1 - fam_counts[fam_off]) / (1 - p)))
                    f_nc[inc] -= 1

                f[fam_counts[fam_off]] -= 1
                fam_counts[fam_off] += 1
                f[fam_counts[fam_off]] += 1

                nc[fam_off] = (fam_counts[fam_off] + 1 - n*p) / (n - n * p)
                nc_int = - int(np.ceil((1 - fam_counts[fam_off]) / (1 - p)))
                f_nc[nc_int] += 1

                if middle and fam_counts[fam_off] >= f_left_border:
                    if num_candidates == max_candidates:
                        max_candidates = 2 * max_candidates
                        new_candidates = np.zeros((max_candidates,), dtype=np.uint32)
                        for j in range(len(f_candidates)):
                            new_candidates[j] = f_candidates[j]
                        f_candidates = new_candidates

                    f_candidates[num_candidates] = fam_off
                    num_candidates += 1

                c_max = max(c_max, fam_counts[fam_off])
                nc_max = max(nc_max, nc[fam_off])

                # initiate first location
                if fam_lowloc[fam_off] == -1:
                    fam_lowloc[fam_off] = loc
                    fam_highloc[fam_off] = loc

                # update either lower or higher boundary
                elif loc < fam_lowloc[fam_off]:
                    fam_lowloc[fam_off] = loc
                elif loc > fam_highloc[fam_off]:
                    fam_highloc[fam_off] = loc

            #for fam_off in hit_fams[:num_hit_fams]:
                #p = ref_fam_prob[fam_off]
                #nc_opt[fam_off] = nc[fam_off] + (n - i - 1) / (n - n * p)

            #nc_opt_sorted = list(reversed(sorted(nc_opt)))
            #nc_opt_max = nc_opt.max()

            #c_opt = c + (n - i - 1)
            #c_opt_sorted = list(reversed(sorted(c_opt)))
            #c_opt_max = c_opt_sorted[1]
            #stop = c_opt_max < c_max

            # Middle stage is over if we have at least some
            # candidates' ids
            if thorough:
                #middle = False
                middle = num_candidates == 0
                #print(f"Start binary search at {i} / {n} with {num_candidates} candidates")
            else:
                sum_candidates = 0
                left_border = 0
                for left_border in range(c_max, 0, -1):
                    sum_candidates += f[left_border]
                    if sum_candidates >= max_candidates:
                        break

                c_opt_candidate = fam_counts[left_border] + (n - i - 1)
                #nc_opt_candidate = nc[left_border] + (n - i - 1)
                thorough = c_opt_candidate < c_max

                # Start the middle stage (1 iteration)
                if thorough:
                    f_left_border = left_border
                    middle = True
                    #print(f"Stop at {i} / {n}: {c_opt_candidate} < {c_max}. {max_candidates} candidates")

    return num_hit_fams, num_hit_hogs


## functions to cumulate HOG k-mer counts
@numba.njit
def cumulate_counts_1fam(hog_cum_counts, fam_level_offsets, hog2parent):
    current_best_child_count = np.zeros(hog_cum_counts.shape, dtype=np.uint32)

    # iterate over level offsets backward
    for i in range(fam_level_offsets.size - 2):
        x = fam_level_offsets[-i - 3 : -i - 1]

        # when reaching level, sum all hog counts with their best child count
        hog_cum_counts[x[0] : x[1]] = np.add(
            hog_cum_counts[x[0] : x[1]], current_best_child_count[x[0] : x[1]]
        )

        # update current_best_child_count of the parents of the current hogs
        for j in range(x[0], x[1]):
            parent_off = hog2parent[j]

            # only if parent exists
            if parent_off != -1:
                c = current_best_child_count[hog2parent[j]]
                current_best_child_count[hog2parent[j]] = max(c, hog_cum_counts[j])


## generic score functions
@numba.njit
def store_bestpath(hog_offsets, parent_offsets, fam_bestpath, fam_hog_scores):
    # keep HOGs descending from the best path
    cands = hog_offsets[fam_bestpath[parent_offsets]]

    # get the score of these candidates
    cands_scores = fam_hog_scores[cands]

    # find the candidate HOGs offsets with the higher count (>0) at this level
    if cands_scores.size > 0:
        # take max, >0
        cands_offsets = np.where(
            (cands_scores > 0) & (cands_scores == np.max(cands_scores))
        )[0]

        # if a single candidate, update the best path. Else, stop because of tie
        if cands_offsets.size == 1:
            fam_bestpath[cands[cands_offsets]] = True


@numba.njit
def hog_path_placement(
    fam_hog_cumcounts,
    query_nkmer,
    fam_level_offsets,
    fam_hog2parent,
    fam_hog_counts,
    fam_ref_hog_prob,
):
    fam_hog_scores = np.zeros(fam_hog_cumcounts.shape, dtype=np.float64)
    fam_bestpath = np.full(fam_hog_cumcounts.shape, False)
    revcum_counts = np.full(fam_hog_cumcounts.shape, query_nkmer, dtype=np.uint16)

    # initialise root HOG
    fam_bestpath[0] = True
    expect_count = fam_ref_hog_prob[0] * query_nkmer
    fam_hog_scores[0] = (fam_hog_cumcounts[0] - expect_count) / query_nkmer

    # loop through hog levels
    for i in range(1, fam_level_offsets.size - 2):
        x = fam_level_offsets[i : i + 2]
        hog_offsets = np.array(list(range(x[0], x[1])))

        # grab parents
        parent_offsets = fam_hog2parent[hog_offsets]

        # update query revcumcount, basically subtracting parent counts from query counts
        qh_count = revcum_counts[parent_offsets] - fam_hog_counts[parent_offsets]
        revcum_counts[hog_offsets] = qh_count

        ## HOG score
        # compute expected number of k-mer matches
        expect_count = fam_ref_hog_prob[hog_offsets] * qh_count
        fam_hog_scores[hog_offsets] = (
            fam_hog_cumcounts[hog_offsets] - expect_count
        ) / qh_count

        # store bestpath
        store_bestpath(hog_offsets, parent_offsets, fam_bestpath, fam_hog_scores)

        # also, if we have a reference taxon need to know when to STOP

    return fam_hog_scores, fam_bestpath


@numba.njit
def get_closest_taxa_from_ref(q2hog_off, ref_taxoff, tax_tab, hog_tab, chog_buff):
    """
    Based on the predicted HOG, we find the closest implied taxon from the reference taxon.
     - if the reference taxon is implied in the HOG (descendant of root-taxon and ancestor of child-HOG taxa), we report the reference taxon (at level).
     - if the root-taxon is one of its ancestors, we report the first ancestor that is defined within the HOG child-HOG taxa (more general).
       (basically the speciation before the duplication delimiting the HOG from the reference taxon)
     - otherwise, we simply report the HOG root-taxon ('more specific' or 'different lineage').
     - na if not placed
    """
    q2closest_taxon = np.zeros(q2hog_off.size, dtype=np.uint32)
    true_tax_lineage = get_root_leaf_offsets(ref_taxoff, tax_tab["ParentOff"])[::-1]

    for i, hog_off in enumerate(q2hog_off):
        # not placed
        if hog_off == -1:
            q2closest_taxon[i] == -1
            continue

        # reference taxon implied in HOG (at level)
        if is_taxon_implied(true_tax_lineage, hog_off, hog_tab, chog_buff):
            q2closest_taxon[i] = ref_taxoff

        # root-taxon in ancestors (more general)
        elif np.argwhere(true_tax_lineage[1:] == hog_tab["TaxOff"][hog_off]).size == 1:
            # get the closest taxon from the reference taxon among the HOG children
            child_hog_taxa = np.unique(
                hog_tab["TaxOff"][get_children(hog_off, hog_tab, chog_buff)]
            )

            # get the closer ancestral taxon in HOG
            j = 0
            while np.argwhere(child_hog_taxa == true_tax_lineage[j]).size == 0:
                j += 1

            # add 1 because we actually take the parent taxa of child_hog_taxa
            q2closest_taxon[i] = true_tax_lineage[j + 1]

        # root-taxon either in child taxa (more specific) or in a different clade
        else:
            q2closest_taxon[i] = hog_tab["TaxOff"][hog_off]

    return q2closest_taxon


@numba.njit
def unzip_dict(data: numba.typed.Dict):
    keys = np.zeros(len(data))
    values = np.zeros_like(keys)
    for i, key in enumerate(data):
        keys[i] = key
        values[i] = data[key]
    return keys, values


@numba.njit
def dict_slice(data: numba.typed.Dict, start: np.uint32, end: np.uint32) -> np.ndarray:
    result = np.zeros(end - start, dtype=np.uint32)
    for j in range(start, end):
        result[j - start] = data.get(j, np.uint32(0))
    return result

class MergeSearch(object):
    def __init__(self, ki, include_extant_genes=False):
        assert ki.db.db.mode == "r", "Database must be opened in read mode."

        # load ki and db
        self.db = ki.db
        self.ki = ki

        self.include_extant_genes = include_extant_genes

    # want to cache these, so that we don't load multiple times when chunking queries
    @cached_property
    def trans(self):
        return get_transform(self.ki.k, self.ki.alphabet.DIGITS_AA)

    @cached_property
    def kmer_table(self):
        # the kmer table requires caching so that we can use it in numba
        z = self.ki.kmer_table
        return {k: z[k][:] for k in z}

    @cached_property
    def fam_tab(self):
        return self.db._db_Family[:]

    @cached_property
    def hog_tab(self):
        return self.db._db_HOG[:]

    @cached_property
    def tax_tab(self):
        return self.db._db_Taxonomy[:]

    @cached_property
    def level_arr(self):
        return self.db._db_LevelOffsets[:]

    @lazy_property
    def ref_fam_prob(self):
        return self.db._db_Index_FamilyProbability[:]

    @lazy_property
    def ref_hog_prob(self):
        return self.db._db_Index_HOGProbability[:]

    def merge_search(
        self,
        seqs,
        ids,
        top_n_fams=1,
        alpha=1e-6,
        sst=0.1,
        family_only=False,
        ref_taxon_off=None,
    ):
        t0 = time()
        sbuff = SequenceBuffer(seqs=seqs, ids=ids)

        # allocate result arrays
        family_results = np.zeros(
            (len(sbuff.idx) - 1, top_n_fams),
            dtype=np.dtype(
                [
                    ("id", np.uint32),
                    ("pvalue", np.float64),
                    ("count", np.uint32),
                    ("normcount", np.float64),
                    ("overlap", np.float64),
                ]
            ),
        )
        subfam_results = np.zeros(
            (len(sbuff.idx) - 1, top_n_fams),
            dtype=np.dtype(
                [("id", np.uint32), ("score", np.float64), ("count", np.uint32)]
            ),
        )

        # perform the search. arguments are given like this as we are using numba.
        self._lookup(
            family_results,
            subfam_results,
            sbuff.buff,
            sbuff.idx,
            self.trans,
            self.kmer_table["idx"],
            self.kmer_table["buff"],
            self.ki.k,
            self.ki.alphabet.DIGITS_AA_LOOKUP,
            self.fam_tab,
            self.hog_tab,
            self.level_arr,
            top_n_fams=top_n_fams,
            ref_fam_prob=self.ref_fam_prob,
            ref_hog_prob=self.ref_hog_prob,
            alpha_cutoff=alpha,
            sst=sst,
            family_only=family_only,
        )

        t1 = time()
        td = max((t1 - t0), 1e-3)
        n = len(sbuff.ids)
        LOG.debug(
            "{:.02f} seconds for block of {:d} sequences (~{:.02f} queries/second)".format(
                td, n, n / td
            )
        )

        return self.output_results(
            family_results, subfam_results, sbuff, top_n_fams, ref_taxon_off
        )

    def output_results(
        self,
        family_results,
        subfam_results,
        sbuff,
        top_n_fams,
        ref_taxon_off,
    ):
        HEADER = [
            "qseqid",
            "hogid",
            "hoglevel",
            "family_p",
            "family_count",
            "family_normcount",
            "subfamily_score",
            "subfamily_count",
            "qseqlen",
            "subfamily_medianseqlen",
            "qseq_overlap",
        ]

        # Note: missing values are dealt differently by pandas and numpy

        def generate():
            for i in range(0, len(sbuff.idx) - 1):
                for j in range(top_n_fams):
                    if (j == 0) or subfam_results["id"][i, j] > 0:
                        yield {
                            "qseq_offset": i + 1,
                            "hog_offset": subfam_results["id"][i, j],
                            "qseq_overlap": family_results["overlap"][i, j],
                            "family_p": family_results["pvalue"][i, j],
                            "subfamily_score": subfam_results["score"][i, j],
                            "family_count": family_results["count"][i, j],
                            "family_normcount": family_results["normcount"][i, j],
                            "subfamily_count": subfam_results["count"][i, j],
                        }

        df = pd.DataFrame(generate())
        if len(df) == 0:
            return df

        # cast to pd dtype so that we can use pd.NA...
        df["qseq_offset"] = df["qseq_offset"].astype("UInt32")
        df["hog_offset"] = df["hog_offset"].astype("UInt32")
        df["family_count"] = df["family_count"].astype("UInt32")
        df["subfamily_count"] = df["subfamily_count"].astype("UInt32")

        # set empty as NA
        na_value = 0
        for k in df.keys():
            df.loc[df[k] == na_value, k] = pd.NA

        # set the query ids
        qseq_offsets = df["qseq_offset"].to_numpy(dtype=np.uint32)
        df["qseqid"] = sbuff.ids[qseq_offsets - 1]
        df["qseqlen"] = sbuff.get_seqlen(qseq_offsets)

        # load the hog ids
        hog_f = df["hog_offset"].notna()
        df.loc[hog_f, "subfamily_medianseqlen"] = (
            df.loc[hog_f, "hog_offset"]
            .apply(lambda i: self.hog_tab["MedianSeqLen"][i - 1])
            .astype("UInt32")
        )
        if self.include_extant_genes:
            # add extant gene list if necessary
            HEADER.append("subfamily_geneset")
            df.loc[hog_f, "subfamily_geneset"] = df.loc[hog_f, "hog_offset"].apply(
                lambda i: ",".join(
                    map(
                        self.db.get_prot_id,
                        get_hog_member_prots(
                            i-1,
                            self.hog_tab,
                                self.db._db_ChildrenHOG[:],
                                self.db._db_ChildrenProt[:],
                            ))))

        # add the hog id
        df.loc[hog_f, "hogid"] = df.loc[hog_f, "hog_offset"].apply(
            lambda i: self.db.get_hog_id(i - 1)
        )
        # add the hog level
        df.loc[hog_f, "hoglevel"] = df.loc[hog_f, "hog_offset"].apply(
            lambda i: self.tax_tab[self.hog_tab[i - 1]["TaxOff"]]["ID"].decode("ascii")
        )

        # compute taxonomic congruences
        if ref_taxon_off:
            q2hog_off = df.loc[hog_f, "hog_offset"].to_numpy(dtype=np.uint32)
            q2closest_taxon = get_closest_taxa_from_ref(
                q2hog_off,
                ref_taxon_off,
                self.tax_tab,
                self.hog_tab,
                self.db._db_ChildrenHOG[:],
            )
            HEADER.append("closest_taxa")
            df.loc[hog_f, "closest_taxa"] = list(
                map(
                    lambda x: self.tax_tab["ID"][x].decode("ascii")
                    if x != -1
                    else pd.NA,
                    q2closest_taxon,
                )
            )

        return df[HEADER]

    @lazy_property
    def _lookup(self):
        def func(
                family_results,
                subfam_results,
                seqs,
                seqs_idx,
                trans,
                table_idx,
                table_buff,
                k,
                DIGITS_AA_LOOKUP,
                fam_tab,
                hog_tab,
                level_arr,
                top_n_fams,
                ref_fam_prob,
                ref_hog_prob,
                alpha_cutoff,
                sst,
                family_only,
        ):
            """
            top_n_fams: number of family for which HOG scores are computed
            """
            # flags to ignore k-mers containing X
            x_flag = table_idx.size - 1

            num_threads = numba.get_num_threads()
            thread_hog_counts = np.zeros((num_threads, hog_tab.size), dtype=np.uint16)
            thread_fam_counts = np.zeros((num_threads, fam_tab.size), dtype=np.uint16)
            thread_fam_lowloc = np.full((num_threads, fam_tab.size), -1, dtype=np.int32)
            thread_fam_highloc = np.full((num_threads, fam_tab.size), -1, dtype=np.int32)
            thread_hit_fams = np.zeros((num_threads, fam_tab.size), dtype=np.int32)
            thread_hit_hogs = np.zeros((num_threads, hog_tab.size), dtype=np.int32)
            thread_num_hit_fams = np.zeros(num_threads, dtype=np.uint32)
            thread_num_hit_hogs = np.zeros(num_threads, dtype=np.uint32)

            parse_time = 0
            search_time = 0
            filter_time = 0
            pvalue_time = 0
            place_time = 0
            sort_time = 0

            for zz in numba.prange(len(seqs_idx) - 1):
                t0 = clock()

                # load seq
                s = seqs[seqs_idx[zz] : np.int64(seqs_idx[zz + 1] - 1)]
                query_len = s.shape[0]
                n_kmers = query_len - (k - 1)

                # double check we don't have short peptides (of len < k)
                if n_kmers == 0:
                    continue

                # seq -> bag of kmers
                (r1, p1, _) = parse_seq(s, DIGITS_AA_LOOKUP, n_kmers, k, trans, x_flag)

                # skip if only one k-mer with X
                if len(r1) > 1:
                    pass
                elif r1[0] == x_flag:
                    continue

                t1 = clock()
                parse_time += t1 - t0
                t0 = clock()

                # Get thread-local data structures for search
                hit_fams = thread_hit_fams[numba.get_thread_id()]
                hit_hogs = thread_hit_hogs[numba.get_thread_id()]
                hog_counts = thread_hog_counts[numba.get_thread_id()]
                fam_counts = thread_fam_counts[numba.get_thread_id()]
                fam_lowloc = thread_fam_lowloc[numba.get_thread_id()]
                fam_highloc = thread_fam_highloc[numba.get_thread_id()]
                num_hit_fams = thread_num_hit_fams[numba.get_thread_id()]
                num_hit_hogs = thread_num_hit_hogs[numba.get_thread_id()]

                # sort k-mers by popularity
                ranks = np.zeros(r1.shape[0], dtype=np.uint32)
                for i in range(r1.shape[0]):
                    kmer = r1[i]
                    if kmer == x_flag:
                        continue

                    idx = table_idx[kmer: kmer + 2]
                    ranks[i] = idx[1] - idx[0]
                perm = ranks.argsort(kind="mergesort")
                r1 = r1[perm]
                p1 = p1[perm]


                # search using kmers
                num_hit_fams, num_hit_hogs = search_seq_kmers(
                    r1, p1, hog_tab, fam_tab, x_flag, table_idx, table_buff,
                    hog_counts, fam_counts, fam_lowloc, fam_highloc, ref_fam_prob,
                    hit_fams, num_hit_fams, hit_hogs, num_hit_hogs
                )

                thread_num_hit_fams[numba.get_thread_id()] = num_hit_fams
                thread_num_hit_hogs[numba.get_thread_id()] = num_hit_hogs

                # Identify families of interest
                idx = hit_fams[:num_hit_fams]
                qres = np.repeat(np.zeros_like(family_results[zz, 0]), len(idx))
                qres["id"][:] = idx
                qres["count"][:] = fam_counts[idx]

                t1 = clock()
                search_time += t1 - t0
                t0 = clock()

                # 2. Fast family filtering
                #     - a. filter by count. We are only interested in families
                #     that have at least their expected number of k-mer matches,
                #     because that number should be already given by random
                #     (however is still insignificant).
                mu = ref_fam_prob[qres["id"]] * len(r1)
                qres = qres[qres["count"] >= mu]

                #     - b. filter by sequence coverage. There is no point to
                #     compute p-value for families that are going to be hard
                #     filtered by coverage later anyway.
                for i in range(len(qres)):
                    family_id = qres["id"][i]
                    qres["overlap"][i] = (fam_highloc[family_id] - fam_lowloc[family_id] + k) / query_len

                qres = qres[(qres["overlap"] >= (25 / query_len))]
                if len(qres) == 0:
                    continue

                # - c. Filter by predicted p-value: perform the Chernoff KL-div test,
                # that is, the Chernoff upper bound for the binomial X:
                #     P(X >= k) <= exp(-n D(k/n || p))
                # where D is Kullback-Leibler divergence.
                # First, compute k / n, the empirical proportion of Bernoulli successes.
                # We need to clip it a little for the case n = k, because it's used
                # in logarithms later. For the other edge case, we already have k > 0
                # guaranteed, so we're good here
                epsilon = 1e-10
                n = len(r1)
                k_n = np.clip(qres["count"] / n, epsilon, 1 - epsilon)

                # Now, by demanding exp(-n KL(k/n || p)) < alpha, we guarantee P < alpha too.
                # There is a theoretical chance of that P < alpha <= bound, and the test will
                # fail with a false negative. I could not observe any instances of this.
                p = ref_fam_prob[qres["id"]]
                kl_div = k_n * np.log(k_n / p) + (1 - k_n) * np.log((1 - k_n) / (1 - p))
                qres = qres[kl_div > -np.log(alpha_cutoff)/n]

                if len(qres) == 0:
                    continue

                t1 = clock()
                filter_time += t1 - t0
                t0 = clock()

                # 1. compute p-value for each family. note: in negative log units
                correction_factor = np.log(len(ref_fam_prob))
                for i in numba.prange(len(qres)):
                    qres["pvalue"][i] = min(
                        float(MAX_LOGP),
                        max(
                            0.0,
                            (
                                    binom_neglogccdf(
                                        qres["count"][i],
                                        len(r1),
                                        ref_fam_prob[qres["id"][i]],
                                    )
                                    - correction_factor
                            ),
                        ),
                    )

                t1 = clock()
                pvalue_time += t1 - t0
                t0 = clock()

                # Filter on the actual p-value
                alpha = -1.0 * np.log(alpha_cutoff)
                qres = qres[qres["pvalue"] >= alpha]
                # filter out 0 neg log p. alpha > 0 is normal. alpha = 0 is edge case.
                qres = qres if alpha > 0 else qres[qres["pvalue"] > 0]
                if len(qres) == 0:
                    continue

                # 3. Compute normalised count
                expected_count = ref_fam_prob[qres["id"]] * len(r1)
                qres["normcount"][:] = (qres["count"] - expected_count) / (
                        len(r1) - expected_count
                )

                t1 = clock()
                filter_time += t1 - t0
                t0 = clock()

                # 4. Store results
                # - a. sort by normcount, then overlap, then p-value for tie-breaking
                qres = family_result_sort(qres, top_n_fams)

                # - b. store results
                family_results["id"][zz, :top_n_fams] = qres["id"][:top_n_fams] + 1
                family_results["pvalue"][zz, :top_n_fams] = qres["pvalue"][:top_n_fams]
                family_results["count"][zz, :top_n_fams] = qres["count"][:top_n_fams]
                family_results["normcount"][zz, :top_n_fams] = qres["normcount"][:top_n_fams]
                family_results["overlap"][zz, :top_n_fams] = qres["overlap"][:top_n_fams]

                t1 = clock()
                sort_time += t1 - t0
                t0 = clock()

                # 5. Place within families
                for i in numba.prange(min(len(qres), top_n_fams)):
                    entry = fam_tab[qres["id"][i]]
                    hog_s = entry["HOGoff"]
                    hog_e = hog_s + entry["HOGnum"]

                    if family_only:
                        # early exit
                        subfam_results["id"][zz, i] = hog_s + 1
                        continue

                    fam_hog2parent = get_fam_hog2parent(entry, hog_tab)
                    fam_level_offsets = get_fam_level_offsets(entry, level_arr)

                    # cumulation of counts
                    c = hog_counts[hog_s:hog_e].copy()

                    cumulate_counts_1fam(c, fam_level_offsets, fam_hog2parent)

                    # new expected count, but using old cumulation
                    (fam_hog_scores, fam_bestpath) = hog_path_placement(
                        c,
                        r1.size,
                        fam_level_offsets,
                        fam_hog2parent,
                        hog_counts[hog_s:hog_e],
                        ref_hog_prob[hog_s:hog_e],
                    )

                    # place on path
                    choice = 0  # default root
                    choice_score = 0.0
                    best_score = -1
                    for j in np.argwhere(fam_bestpath).flatten():
                        sf_score = fam_hog_scores[j]
                        best_score = max(best_score, sf_score)
                        if sf_score >= sst:
                            choice = j
                            choice_score = sf_score

                    # store results
                    subfam_results["id"][zz, i] = choice + hog_s + 1
                    subfam_results["score"][zz, i] = choice_score
                    subfam_results["count"][zz, i] = (
                        c[int(choice)] if choice_score != 0.0 else 0
                    )

                t1 = clock()
                place_time += t1 - t0

            total_time = parse_time + search_time + filter_time + pvalue_time + place_time + sort_time

            print()
            print("Parse time\t", as_seconds(parse_time))
            print("Search time\t", as_seconds(search_time))
            print("Filter time\t", as_seconds(filter_time))
            print("Pvalue time\t", as_seconds(pvalue_time))
            print("Place time\t", as_seconds(place_time))
            print("Sort time\t", as_seconds(sort_time))
            print("Batch total\t", as_seconds(total_time))
            print()

        #return numba.jit(func, parallel=True, nopython=True, nogil=True)
        return func
