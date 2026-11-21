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
import math
from collections import namedtuple
from time import time

from numba.typed import List
from property_manager import lazy_property, cached_property
import numba
import numpy as np
import pandas as pd


from ._utils import LOG
from .alphabets import get_transform
from .kmer_mask import modality_kmer_mask, stored_min_distinct
from .sequence_buffer import SequenceBuffer
from .hierarchy import (
    get_root_leaf_offsets,
    get_hog_member_prots,
    is_taxon_implied,
    get_children,
)


# maximum neglogp to set
MAX_LOGP = 20000.0

# Search modes. SEARCH_SEQUENCE, SEARCH_STRUCTURE and SEARCH_SUM are also
# the values of the `modality` result field: the scores a family was
# selected with.
SEARCH_SEQUENCE = 1
SEARCH_STRUCTURE = 2
# the sequence, then the 3Di alone for the queries without a significant
# sequence family
SEARCH_SEQUENCE_THEN_STRUCTURE = 3
# family scores summed over the two modalities (sequence + w * 3Di)
SEARCH_SUM = 4
# the sequence, then the summed scores for the queries without a significant
# sequence family
SEARCH_HYBRID = 5

# How the family of a query is selected:
# - filter: test every candidate, rank the significant ones (OMAmer 2);
# - choose_scored: rank the candidates with a positive score (p < 1/N with
#   the Bonferroni correction), then test the chosen one: the query is
#   placed only when its top family is significant.
FAMILY_SELECT_FILTER = 0
FAMILY_SELECT_CHOOSE_SCORED = 1
FAMILY_SELECTIONS = {
    "filter": FAMILY_SELECT_FILTER,
    "choose_scored": FAMILY_SELECT_CHOOSE_SCORED,
}

# Bounds of a family hit probability in the tests: p = 1 (e.g. the only
# family of a database) or a scaled p above 1 would divide by zero.
MIN_PROBABILITY = 1e-100
MAX_PROBABILITY = 1.0 - 1e-9

MODALITY_NAMES = {
    SEARCH_SEQUENCE: "seq",
    SEARCH_STRUCTURE: "ss",
    SEARCH_SUM: "seq+ss",
}

# Array carriers passed to the compiled search (see `lookup_queries`).
SequenceBatch = namedtuple("SequenceBatch", ("buffer", "offsets"))

# The k-mer index of one modality and its binomial models. `max_df` > 0:
# k-mers in more families are ignored; `kmer_mask`: ignored k-mer codes (empty:
# none), e.g. low-complexity ones. Ignored k-mers do not count towards n.
# `df_exponent` b > 0 makes the family hit probabilities query-specific
# when a family is tested (see `query_probability_scale`).
KmerIndex = namedtuple(
    "KmerIndex",
    (
        "table_idx",
        "table_buff",
        "modality",
        "max_df",
        "kmer_mask",
        "family_probability",
        "hog_probability",
        "df_reference",
        "df_exponent",
    ),
)

# `hog2fam`: the family of each HOG, as a contiguous array: the search looks
# it up once per posting, and a strided read of the HOG table costs a cache
# line per lookup.
SearchDatabase = namedtuple(
    "SearchDatabase",
    ("trans", "k", "digits_lookup", "families", "hogs", "levels", "hog2fam"),
)

SearchConfig = namedtuple(
    "SearchConfig",
    (
        "mode",
        "top_n_families",
        "subfamily_score_threshold",
        "family_only",
        "family_selection",
        "neglog_alpha",
        "log_correction",
        # threshold on the summed scores and the weight w of the 3Di score
        "summed_neglog_alpha",
        "ss_weight",
    ),
)

# Per-thread family counters of the k-mer search. Only the entries of the
# families hit by the previous query are reset.
SearchScratch = namedtuple(
    "SearchScratch",
    (
        "hit_families",
        "family_counts",
        "family_low_location",
        "family_high_location",
        "num_hit_families",
    ),
)

def resolve_search_mode(mode, has_sequence, has_structure):
    """The search mode for the given queries ('auto': hybrid with both
    modalities, else the one given)."""
    if mode in (None, "auto"):
        if has_sequence and has_structure:
            return SEARCH_HYBRID
        if has_sequence:
            return SEARCH_SEQUENCE
        if has_structure:
            return SEARCH_STRUCTURE
        raise ValueError("Search requires sequence or structure queries")

    modes = {
        "seq": (SEARCH_SEQUENCE, has_sequence),
        "ss": (SEARCH_STRUCTURE, has_structure),
        "sqs": (SEARCH_SEQUENCE_THEN_STRUCTURE, has_sequence and has_structure),
        "sum": (SEARCH_SUM, has_sequence and has_structure),
        "hybrid": (SEARCH_HYBRID, has_sequence and has_structure),
    }
    try:
        selected, available = modes[mode]
    except KeyError:
        raise ValueError(
            "search_mode must be 'auto', 'seq', 'ss', 'sqs', 'sum' or 'hybrid'"
        ) from None
    if not available:
        raise ValueError(
            "search_mode={!r} requires matching query data and index".format(mode)
        )
    return selected


QUERY_FAMILY_RESULT_DTYPE = np.dtype(
    [
        ("id", np.uint32),
        ("pvalue", np.float64),
        ("count", np.uint32),
        ("normcount", np.float64),
        ("overlap", np.float64),
    ]
)

SearchResults = namedtuple("SearchResults", ("families", "subfamilies"))


def make_search_results(n_queries, top_n_families):
    family_dtype = np.dtype(
        [
            ("id", np.uint32),
            ("pvalue", np.float64),
            ("count", np.uint32),
            ("normcount", np.float64),
            ("overlap", np.float64),
            ("modality", np.uint8),
        ]
    )
    subfamily_dtype = np.dtype(
        [("id", np.uint32), ("score", np.float64), ("count", np.uint32)]
    )
    shape = (n_queries, top_n_families)
    return SearchResults(
        np.zeros(shape, dtype=family_dtype),
        np.zeros(shape, dtype=subfamily_dtype),
    )


@numba.njit(nogil=True)
def clip_probability(p):
    """p clipped to [MIN_PROBABILITY, MAX_PROBABILITY] (unchanged inside)."""
    return min(max(p, MIN_PROBABILITY), MAX_PROBABILITY)


@numba.njit(nogil=True)
def binom_neglogccdf(x, n, p):
    """
    Neg-log of the upper tail P(X >= x) of a Binomial(n, p), in pure numba
    (so that the compiled search can be cached). Stops early once the
    decaying tail is negligible.
    """
    if x <= 0:
        return 0.0
    if x > n:
        return np.inf

    p = clip_probability(p)
    log_px = (
        math.lgamma(n + 1.0)
        - math.lgamma(x + 1.0)
        - math.lgamma(n - x + 1.0)
        + x * math.log(p)
        + (n - x) * math.log1p(-p)
    )

    odds = p / (1.0 - p)
    acc = 1.0
    term = 1.0
    for k in range(x, n):
        ratio = ((n - k) / (k + 1.0)) * odds
        term *= ratio
        acc += term
        # tail is decaying and this term no longer moves the sum -> stop
        if ratio < 1.0 and term < acc * 1e-16:
            break

    return -(log_px + math.log(acc))


# ----
# family result sorting
@numba.njit(nogil=True)
def fam_res_compare(x1, x2, by_score):
    """
    Compare two family results and order them according to normcount, overlap
    and pvalue, or (by_score) pvalue, overlap and normcount.
    """
    if by_score:
        if x1["pvalue"] != x2["pvalue"]:
            # greater first. note, we use neglog units
            return -1 if x1["pvalue"] > x2["pvalue"] else 1
        if x1["overlap"] != x2["overlap"]:
            return -1 if x1["overlap"] > x2["overlap"] else 1
        if x1["normcount"] != x2["normcount"]:
            return -1 if x1["normcount"] > x2["normcount"] else 1
    else:
        if x1["normcount"] != x2["normcount"]:
            return -1 if x1["normcount"] > x2["normcount"] else 1
        if x1["overlap"] != x2["overlap"]:
            return -1 if x1["overlap"] > x2["overlap"] else 1
        if x1["pvalue"] != x2["pvalue"]:
            return -1 if x1["pvalue"] > x2["pvalue"] else 1
    # equal. take whichever.
    return 0


@numba.njit(nogil=True)
def family_result_argsort(x, ii, by_score):
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
            j = fam_res_compare(x[i], x[ii[0]], by_score)
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
            bfs = family_result_argsort(x, bfs, by_score)
        if len(afs) > 0:
            afs = family_result_argsort(x, afs, by_score)

        return bfs + pvs + afs


@numba.njit(nogil=True)
def family_result_sort(x, k, by_score=False):
    """
    Sort the family results according to normcount, overlap and pvalue (to
    break ties), or (by_score) pvalue, overlap and normcount.
    this uses a quicksort implementation as np.argsort does not support struct type in numba.
    """

    # Quickselect top k results
    idx = np.arange(len(x))
    _ = _select(x, idx, k, 0, len(x) - 1, by_score)
    x = x[idx[:k]]

    # Now mergesort the selected results, because we need to
    # report them sorted
    idx = family_result_argsort(x, list(range(len(x))), by_score)
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
def _partition(x, idx, low, high, by_score):
    """
    Index-based version of the partition algorithm of
    quicksort. Juggles indexes of the idx array that
    indexes the x array, considering the range [low, high].
    """
    mid = (low + high) >> 1

    # Use median of three {low, middle, high} as the pivot
    # (fam_res_compare(a, b) < 0: a goes before b)
    if fam_res_compare(x[idx[mid]], x[idx[low]], by_score) < 0:
        _swap(idx, mid, low)
    if fam_res_compare(x[idx[high]], x[idx[mid]], by_score) < 0:
        _swap(idx, high, mid)
        if fam_res_compare(x[idx[mid]], x[idx[low]], by_score) < 0:
            _swap(idx, low, mid)

    pivot = x[idx[mid]]
    # Put the pivot in the end of the array
    _swap(idx, mid, high)

    # Collect elements that are > pivot in the beginning
    i = low
    for j in range(low, high):
        if fam_res_compare(x[idx[j]], pivot, by_score) < 0:
            _swap(idx, i, j)
            i += 1

    # All indexes of idx in [0, i) are good now.
    # Let's place the pivot back where it should be
    _swap(idx, i, high)
    return i


@numba.njit(nogil=True)
def _select(x, idx, k, low, high, by_score):
    """
    Select the k'th largest element of the x array
    """
    if k >= len(x):
        return len(x)

    i = _partition(x, idx, low, high, by_score)
    while i != k:
        if i < k:
            low = i + 1
            i = _partition(x, idx, low, high, by_score)
        else:
            high = i - 1
            i = _partition(x, idx, low, high, by_score)
    return idx[k]


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


@numba.njit(nogil=True)
def search_seq_kmers(r1, p1, hog2fam, x_flag, index, scratch, thread_id):
    """
    Perform the k-mer search, using the index: count the k-mer hits of each
    family and record the first and last query locations of these hits.

    Returns the number of skipped k-mers (with X, masked, or above the
    document frequency cap), the searched k-mers that have postings (in
    query order), from which `count_family_hogs` recovers the per-HOG
    counts of a family, and the sum of the document frequencies of the
    k-mers counted (not skipped).
    """
    table_idx = index.table_idx
    table_buff = index.table_buff
    max_df = index.max_df

    thread_fam_counts = scratch.family_counts[thread_id]
    thread_fam_lowloc = scratch.family_low_location[thread_id]
    thread_fam_highloc = scratch.family_high_location[thread_id]
    thread_hit_fams = scratch.hit_families[thread_id]
    thread_num_hit_fams = scratch.num_hit_families[thread_id]

    # Reinitialize counters modified by the previous call
    for i in range(thread_num_hit_fams):
        family = thread_hit_fams[i]
        thread_fam_counts[family] = 0
        thread_fam_lowloc[family] = -1
        thread_fam_highloc[family] = -1

    thread_num_hit_fams = 0
    n_skipped = 0
    sum_df = 0.0
    kmers = np.empty(r1.shape[0], dtype=r1.dtype)
    n_searched = 0

    # iterate unique k-mers
    for m in range(r1.shape[0]):
        kmer = r1[m]
        loc = p1[m]

        # k-mers with X are ignored and do not count towards n (the unique
        # codes contain the X flag at most once)
        if kmer == x_flag:
            n_skipped += 1
            continue

        # masked k-mers (e.g. low-complexity 3Di) are ignored as well
        if index.kmer_mask.size > 0 and index.kmer_mask[kmer]:
            n_skipped += 1
            continue

        # get mapping to HOGs
        lo = table_idx[kmer]
        hi = table_idx[kmer + 1]
        df = hi - lo

        if 0 < max_df < df:
            n_skipped += 1
            continue

        sum_df += df
        if df > 0:
            kmers[n_searched] = kmer
            n_searched += 1

        # The locations p1 of the unique k-mers increase with m, so the low
        # location of a family is that of its first hit and the high one
        # that of its last hit.
        for j in range(lo, hi):
            fam_off = hog2fam[table_buff[j]]
            count = thread_fam_counts[fam_off]
            if not count:
                thread_hit_fams[thread_num_hit_fams] = fam_off
                thread_num_hit_fams += 1
                thread_fam_lowloc[fam_off] = loc
            thread_fam_counts[fam_off] = count + 1
            thread_fam_highloc[fam_off] = loc

    scratch.num_hit_families[thread_id] = thread_num_hit_fams
    return n_skipped, kmers[:n_searched], sum_df


@numba.njit(nogil=True)
def count_family_hogs(kmers, index, hog_s, hog_e):
    """
    Per-HOG k-mer counts of the query in the family with HOGs [hog_s, hog_e).

    A posting list is sorted by HOG offset and the HOGs of a family are
    contiguous, so the postings of the family in the list of each k-mer are
    found by binary search.
    """
    table_idx = index.table_idx
    table_buff = index.table_buff
    hog_s = np.int64(hog_s)
    hog_e = np.int64(hog_e)
    counts = np.zeros(hog_e - hog_s, dtype=np.uint16)
    for m in range(kmers.shape[0]):
        kmer = kmers[m]
        a = np.int64(table_idx[kmer])
        hi = np.int64(table_idx[kmer + 1])
        b = hi
        # first posting with HOG >= hog_s
        while a < b:
            mid = (a + b) >> 1
            if np.int64(table_buff[mid]) < hog_s:
                a = mid + 1
            else:
                b = mid
        while a < hi:
            hog = np.int64(table_buff[a])
            if hog >= hog_e:
                break
            counts[hog - hog_s] += 1
            a += 1
    return counts


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


@numba.njit(nogil=True)
def find_family_candidates(sequence, database, index, scratch):
    """
    Search the k-mers of one query and keep the family candidates with hits
    over at least 25 positions of the query, before scoring
    (`score_family_candidates`).

    Returns the candidates (unsorted), the number n of k-mers used for
    scoring, the searched k-mers and the sum of the document frequencies of
    the n k-mers (see `search_seq_kmers`).
    """
    k = database.k
    query_len = sequence.shape[0]
    n_kmers = query_len - (k - 1)
    x_flag = index.table_idx.size - 1

    no_kmers = np.empty(0, dtype=np.uint32)
    if n_kmers <= 0:
        return (
            np.zeros(0, dtype=QUERY_FAMILY_RESULT_DTYPE), np.int64(0), no_kmers, 0.0
        )

    # seq -> bag of kmers
    (r1, p1, _) = parse_seq(
        sequence, database.digits_lookup, n_kmers, k, database.trans, x_flag
    )

    # skip if only one k-mer with X
    if len(r1) == 1 and r1[0] == x_flag:
        return (
            np.zeros(0, dtype=QUERY_FAMILY_RESULT_DTYPE), np.int64(0), no_kmers, 0.0
        )

    thread_id = numba.get_thread_id()
    (n_skipped, kmers, sum_df) = search_seq_kmers(
        r1, p1, database.hog2fam, x_flag, index, scratch, thread_id
    )

    thread_hit_fams = scratch.hit_families[thread_id]
    thread_fam_counts = scratch.family_counts[thread_id]
    thread_fam_lowloc = scratch.family_low_location[thread_id]
    thread_fam_highloc = scratch.family_high_location[thread_id]
    thread_num_hit_fams = scratch.num_hit_families[thread_id]

    # Filter by sequence coverage: there is no point to compute p-values for
    # families that would be filtered by coverage later anyway. Coverage
    # removes about three in four of the hit families at the cost of a few
    # array reads each; the count filter (`score_family_candidates`) one in
    # five of the rest.
    min_overlap = 25 / query_len
    covered = np.empty(thread_num_hit_fams, dtype=np.uint32)
    n_covered = 0
    for i in range(thread_num_hit_fams):
        family_id = thread_hit_fams[i]
        overlap = (
            thread_fam_highloc[family_id] - thread_fam_lowloc[family_id] + k
        ) / query_len
        if overlap >= min_overlap:
            covered[n_covered] = family_id
            n_covered += 1
    idx = covered[:n_covered]
    qres = np.zeros(n_covered, dtype=QUERY_FAMILY_RESULT_DTYPE)
    qres["id"][:] = idx
    qres["count"][:] = thread_fam_counts[idx]
    for i in range(n_covered):
        family_id = idx[i]
        qres["overlap"][i] = (
            thread_fam_highloc[family_id] - thread_fam_lowloc[family_id] + k
        ) / query_len

    n = np.int64(len(r1) - n_skipped)
    return qres, n, kmers, sum_df


def kmer_df_reference(table_idx, max_df=0, kmer_mask=None):
    """
    The posting-weighted mean document frequency m0 = sum(df^2) / sum(df) of
    the k-mers a search counts (not masked, df <= max_df when max_df > 0):
    the mean df of a k-mer drawn like the postings, which is how the family
    probabilities p_F are defined. If the chance that a family contains a
    k-mer is proportional to its df, a query whose k-mers have mean df m
    expects n p_F m / m0 hits in family F.
    """
    (total, total_sq) = _kmer_df_sums(
        table_idx,
        np.int64(max_df),
        np.zeros(0, dtype=np.bool_) if kmer_mask is None else kmer_mask,
    )
    return float(total_sq) / float(total) if total > 0 else 0.0


@numba.njit(nogil=True)
def _kmer_df_sums(table_idx, max_df, kmer_mask):
    """sum(df) and sum(df^2) of the k-mers of `kmer_df_reference`, exact
    (in integers) and without a copy of the index."""
    total = np.int64(0)
    total_sq = np.int64(0)
    for i in range(len(table_idx) - 1):
        if kmer_mask.size > 0 and kmer_mask[i]:
            continue
        df = np.int64(table_idx[i + 1]) - np.int64(table_idx[i])
        if 0 < max_df < df:
            continue
        total += df
        total_sq += df * df
    return total, total_sq


@numba.njit(nogil=True)
def query_probability_scale(index, sum_df, n):
    """
    The factor (m / m0) ** b of the family hit probabilities of one query
    (1 when off), with m the mean document frequency of its n k-mers
    (`sum_df` / n), m0 = `index.df_reference` and b = `index.df_exponent`:
    queries made of common k-mers need more hits to be significant.
    """
    if index.df_exponent == 0.0 or index.df_reference <= 0.0 or n <= 0:
        return 1.0
    mean_df = sum_df / n
    if mean_df <= 0.0:
        return 1.0
    return (mean_df / index.df_reference) ** index.df_exponent


@numba.njit(nogil=True)
def scale_probability(p, p_scale):
    return clip_probability(p * p_scale)


@numba.njit(nogil=True)
def is_significant(score, neglog_alpha):
    """The significance test of a corrected score (with alpha = 1, a
    positive score)."""
    return score >= neglog_alpha and (neglog_alpha > 0.0 or score > 0.0)


@numba.njit(nogil=True)
def score_family_tails(qres, n, index, log_correction, p_scale=1.0):
    """
    Set the corrected score (-log tail probability minus the correction,
    clipped to [0, MAX_LOGP]) of every candidate, in place, with the family
    hit probabilities scaled by `p_scale` (see `query_probability_scale`).
    """
    for i in range(len(qres)):
        p = scale_probability(index.family_probability[qres["id"][i]], p_scale)
        neglog_tail = binom_neglogccdf(qres["count"][i], n, p)
        qres["pvalue"][i] = min(MAX_LOGP, max(0.0, neglog_tail - log_correction))


@numba.njit(nogil=True)
def score_family_candidates(
    qres, n, index, neglog_alpha, log_correction, p_scale=1.0
):
    """
    The significant family candidates of a query, with their scores (in
    corrected -log units) and normcounts. The scores use the family hit
    probabilities scaled by `p_scale` (see `query_probability_scale`), the
    normcounts, which rank the families of a query, the unscaled ones.

    Two fast filters come first, in one pass: the count filter (families
    with fewer k-mer hits than expected are not significant) and the
    Chernoff bound of the binomial tail.
    """
    epsilon = 1e-10
    keep = np.full(len(qres), True)
    for i in range(len(qres)):
        expected = scale_probability(
            index.family_probability[qres["id"][i]], p_scale
        )
        if not qres["count"][i] >= expected * n:
            keep[i] = False
            continue
        # The Chernoff upper bound for the binomial X:
        #     P(X >= k) <= exp(-n KL_div(k/n || p))
        #
        # The empirical proportion of observed k-mers, clipped for the
        # edge cases k = n, k = 0.
        observed = min(max(qres["count"][i] / n, epsilon), 1.0 - epsilon)
        kl_divergence = observed * math.log(observed / expected) + (
            1.0 - observed
        ) * math.log((1.0 - observed) / (1.0 - expected))
        # Now we check P <= exp(-n KL_div(k/n || p)) <= alpha
        #                   KL_div(k/n || p) >= -log(alpha) / n.
        # and if it holds, we almost guarantee P <= alpha
        # There is a very small chance of that P <= alpha < bound, and then this test
        # fails with a false negative. In practice, it does not happen.
        keep[i] = kl_divergence > neglog_alpha / n
    qres = qres[keep]
    if len(qres) == 0:
        return qres

    score_family_tails(qres, n, index, log_correction, p_scale)

    # normalised count
    expected_count = index.family_probability[qres["id"]] * n
    qres["normcount"][:] = (qres["count"] - expected_count) / (n - expected_count)

    keep = np.empty(len(qres), dtype=np.bool_)
    for i in range(len(qres)):
        keep[i] = is_significant(qres["pvalue"][i], neglog_alpha)
    return qres[keep]


@numba.njit(nogil=True)
def keep_if_chosen_significant(qres, neglog_alpha):
    """
    Ranked candidates (best first) -> the significant ones if the best one
    is significant, else none.
    """
    if len(qres) == 0 or not is_significant(qres["pvalue"][0], neglog_alpha):
        return qres[:0]
    keep = np.empty(len(qres), dtype=np.bool_)
    for i in range(len(qres)):
        keep[i] = is_significant(qres["pvalue"][i], neglog_alpha)
    return qres[keep]


@numba.njit(nogil=True)
def select_family_candidates(qres, n, index, config, p_scale):
    """
    The significant family candidates of a query under the selection rule
    `config.family_selection` (see FAMILY_SELECT_*), unsorted for the filter
    rule and ranked for choose_scored. The tests use the family hit
    probabilities scaled by `p_scale` (see `query_probability_scale`). With
    choose_scored, the candidates and their ranking do not: the scaling
    only decides whether the chosen families are significant.
    """
    if config.family_selection == FAMILY_SELECT_FILTER:
        return score_family_candidates(
            qres, n, index, config.neglog_alpha, config.log_correction, p_scale
        )

    qres = score_family_candidates(qres, n, index, 0.0, config.log_correction)
    if len(qres) == 0:
        return qres
    qres = family_result_sort(qres, config.top_n_families, False)
    if p_scale != 1.0:
        # only the ranked candidates are scored again, with the scaling
        score_family_tails(qres, n, index, config.log_correction, p_scale)
    return keep_if_chosen_significant(qres, config.neglog_alpha)


@numba.njit(nogil=True)
def store_family_results(family_results, sequence_id, qres, n_store, modality):
    family_results["id"][sequence_id, :n_store] = qres["id"][:n_store] + 1
    family_results["pvalue"][sequence_id, :n_store] = qres["pvalue"][:n_store]
    family_results["count"][sequence_id, :n_store] = qres["count"][:n_store]
    family_results["normcount"][sequence_id, :n_store] = qres["normcount"][:n_store]
    family_results["overlap"][sequence_id, :n_store] = qres["overlap"][:n_store]
    family_results["modality"][sequence_id, :n_store] = modality


@numba.njit(nogil=True)
def place_in_family(
    subfam_results, sequence_id, rank, family_id, n, kmers, index, database, config
):
    """
    Place a query on the HOG path of one family, from the per-HOG counts of
    the searched k-mers `kmers` (n k-mers used for scoring) in `index`.
    """
    entry = database.families[family_id]
    hog_s = entry["HOGoff"]
    hog_e = hog_s + entry["HOGnum"]

    if config.family_only:
        # early exit: the HOG counts are not needed
        subfam_results["id"][sequence_id, rank] = hog_s + 1
        return

    fam_hog2parent = get_fam_hog2parent(entry, database.hogs)
    fam_level_offsets = get_fam_level_offsets(entry, database.levels)

    fam_hog_counts = count_family_hogs(kmers, index, hog_s, hog_e)

    # cumulation of counts
    c = fam_hog_counts.copy()

    cumulate_counts_1fam(c, fam_level_offsets, fam_hog2parent)

    # new expected count, but using old cumulation
    (fam_hog_scores, fam_bestpath) = hog_path_placement(
        c,
        n,
        fam_level_offsets,
        fam_hog2parent,
        fam_hog_counts,
        index.hog_probability[hog_s:hog_e],
    )

    # place on path
    choice = 0  # default root
    choice_score = 0.0
    for j in np.argwhere(fam_bestpath).flatten():
        sf_score = fam_hog_scores[j]
        if sf_score >= config.subfamily_score_threshold:
            choice = j
            choice_score = sf_score

    # store results
    subfam_results["id"][sequence_id, rank] = choice + hog_s + 1
    subfam_results["score"][sequence_id, rank] = choice_score
    subfam_results["count"][sequence_id, rank] = (
        c[int(choice)] if choice_score != 0.0 else 0
    )


@numba.njit(nogil=True)
def place_family_results(
    family_results, subfam_results, sequence_id, qres, n, kmers, database, index,
    config, by_score=False,
):
    """Rank, store and place the significant family candidates of a query."""
    top_n_fams = config.top_n_families
    qres = family_result_sort(qres, top_n_fams, by_score)

    # store results (fewer than top_n_fams may be significant)
    n_store = min(len(qres), top_n_fams)
    store_family_results(family_results, sequence_id, qres, n_store, index.modality)

    # place within families
    for i in range(n_store):
        place_in_family(
            subfam_results, sequence_id, i, qres["id"][i], n, kmers, index,
            database, config,
        )


@numba.njit(nogil=True)
def place_sequence(
    family_results, subfam_results, sequence, sequence_id, database, index,
    config, scratch,
):
    """Search and place one query in one modality; False if not placed."""
    (qres, n, kmers, sum_df) = find_family_candidates(
        sequence, database, index, scratch
    )
    if len(qres) > 0:
        qres = select_family_candidates(
            qres, n, index, config, query_probability_scale(index, sum_df, n)
        )
    if len(qres) == 0:
        return False

    place_family_results(
        family_results, subfam_results, sequence_id, qres, n, kmers, database,
        index, config,
    )
    return True


@numba.njit(nogil=True)
def score_summed_candidates(qres, n, index, config, p_scale):
    """
    Score the family candidates of one modality for a sum: without a
    significance threshold (as with alpha = 1, every family with a positive
    corrected score is kept), so that a family weakly supported by both
    modalities is not lost. The candidates are those of the unscaled model,
    with the scores of the scaled one (see `query_probability_scale`; a
    score may then be 0).
    """
    qres = score_family_candidates(qres, n, index, 0.0, config.log_correction)
    if p_scale != 1.0 and len(qres) > 0:
        score_family_tails(qres, n, index, config.log_correction, p_scale)
    return qres


@numba.njit(nogil=True)
def search_families(sequence, database, index, config, scratch):
    """
    Search the k-mers of one query and score its family candidates for a
    sum (`score_summed_candidates`).
    """
    (qres, n, kmers, sum_df) = find_family_candidates(
        sequence, database, index, scratch
    )
    if len(qres) > 0:
        qres = score_summed_candidates(
            qres, n, index, config, query_probability_scale(index, sum_df, n)
        )
    return qres, n, kmers


@numba.njit(nogil=True)
def sum_family_scores(seq_res, ss_res, ss_weight):
    """
    Union of the family candidates of the two modalities, keyed by family.

    The score (`pvalue`, in corrected -log units) of a family is its sequence
    score plus `ss_weight` times its 3Di score; a family missing from one
    modality takes 0 from it. The other fields (count, normcount, overlap)
    are those of the sequence candidate when there is one, else of the 3Di
    one: the modality whose HOG counts are used for the placement.
    """
    n_seq = len(seq_res)
    combined = np.zeros(n_seq + len(ss_res), dtype=QUERY_FAMILY_RESULT_DTYPE)
    combined[:n_seq] = seq_res

    order = np.argsort(seq_res["id"])
    seq_ids = seq_res["id"][order]
    n_combined = n_seq
    for i in range(len(ss_res)):
        family_id = ss_res["id"][i]
        j = np.searchsorted(seq_ids, family_id)
        if j < n_seq and seq_ids[j] == family_id:
            combined["pvalue"][order[j]] += ss_weight * ss_res["pvalue"][i]
        else:
            combined[n_combined] = ss_res[i]
            combined["pvalue"][n_combined] = ss_weight * ss_res["pvalue"][i]
            n_combined += 1
    return combined[:n_combined]


@numba.njit(nogil=True)
def place_summed(
    family_results, subfam_results, sequence_id, seq_res, seq_n, seq_kmers,
    structure, database, seq_index, ss_index, config, scratch,
):
    """
    Search the 3Di of a query, given its sequence candidates `seq_res`
    (`search_families`), and rank families by the sum of their two scores
    (`sum_family_scores`); the threshold `config.summed_neglog_alpha`
    applies to the sum. Within a family, the query is placed with the
    sequence HOG counts when the family is a sequence candidate, else with
    the 3Di ones.
    """
    (ss_res, ss_n, ss_kmers) = search_families(
        structure, database, ss_index, config, scratch
    )
    qres = sum_family_scores(seq_res, ss_res, config.ss_weight)
    keep = np.empty(len(qres), dtype=np.bool_)
    for i in range(len(qres)):
        keep[i] = is_significant(qres["pvalue"][i], config.summed_neglog_alpha)
    qres = qres[keep]
    if len(qres) == 0:
        return False
    # ranked by the summed score
    top_n_fams = config.top_n_families
    qres = family_result_sort(qres, top_n_fams, True)
    n_store = min(len(qres), top_n_fams)
    store_family_results(family_results, sequence_id, qres, n_store, SEARCH_SUM)

    # placement modality: the sequence when it has the family as a candidate
    seq_ids = np.sort(seq_res["id"])
    for i in range(n_store):
        family_id = qres["id"][i]
        j = np.searchsorted(seq_ids, family_id)
        if j < len(seq_ids) and seq_ids[j] == family_id:
            place_in_family(
                subfam_results, sequence_id, i, family_id, seq_n, seq_kmers,
                seq_index, database, config,
            )
        else:
            place_in_family(
                subfam_results, sequence_id, i, family_id, ss_n, ss_kmers,
                ss_index, database, config,
            )
    return True


@numba.njit(nogil=True)
def place_sequence_hybrid(
    family_results, subfam_results, sequence, structure, sequence_id, database,
    seq_index, ss_index, config, scratch,
):
    """
    Place a query by its sequence when it has a significant sequence family
    (as `place_sequence`); otherwise search its 3Di and rank families by the
    sum of the sequence and 3Di scores (`place_summed`). The sequence search
    is shared: its candidates are scored twice, with the threshold and, for
    the sum, without. The 3Di is searched only for the queries without a
    significant sequence family.
    """
    (candidates, seq_n, seq_kmers, sum_df) = find_family_candidates(
        sequence, database, seq_index, scratch
    )
    p_scale = query_probability_scale(seq_index, sum_df, seq_n)
    if len(candidates) > 0:
        qres = select_family_candidates(
            candidates, seq_n, seq_index, config, p_scale
        )
        if len(qres) > 0:
            place_family_results(
                family_results, subfam_results, sequence_id, qres, seq_n,
                seq_kmers, database, seq_index, config,
            )
            return True
        candidates = score_summed_candidates(
            candidates, seq_n, seq_index, config, p_scale
        )
    return place_summed(
        family_results, subfam_results, sequence_id, candidates, seq_n,
        seq_kmers, structure, database, seq_index, ss_index, config, scratch,
    )


@numba.njit(nogil=True)
def select_from_batch(batch, i):
    return batch.buffer[batch.offsets[i] : np.int64(batch.offsets[i + 1] - 1)]


# inlined: the parallel lowering of `lookup_queries` does not resolve the
# namedtuple global otherwise
@numba.njit(nogil=True, inline="always")
def make_search_scratch(num_threads, n_families):
    return SearchScratch(
        np.zeros((num_threads, n_families), dtype=np.int32),
        np.zeros((num_threads, n_families), dtype=np.uint16),
        np.full((num_threads, n_families), -1, dtype=np.int32),
        np.full((num_threads, n_families), -1, dtype=np.int32),
        np.zeros(num_threads, dtype=np.uint32),
    )


@numba.njit(parallel=True, nogil=True, cache=True)
def lookup_queries(
    family_results, subfam_results, sequences, structures, database, seq_index,
    ss_index, config, num_threads,
):
    """
    Search and place every query of the batch in the mode `config.mode`
    (SEARCH_*): the sequences with `seq_index`, the 3Di with `ss_index` (a
    single-modality search may pass the same batch and index twice).

    num_threads is passed in (numba.get_num_threads() of the caller): that
    threading-layer intrinsic embeds a runtime function pointer, which
    would make this kernel non-cacheable.
    """
    scratch = make_search_scratch(num_threads, database.families.size)
    mode = config.mode
    n_queries = (
        structures.offsets.size - 1
        if mode == SEARCH_STRUCTURE
        else sequences.offsets.size - 1
    )

    for sequence_id in numba.prange(n_queries):
        placed = False
        if mode == SEARCH_HYBRID:
            place_sequence_hybrid(
                family_results, subfam_results,
                select_from_batch(sequences, sequence_id),
                select_from_batch(structures, sequence_id),
                sequence_id, database, seq_index, ss_index, config, scratch,
            )
        elif mode == SEARCH_SUM:
            (seq_res, seq_n, seq_kmers) = search_families(
                select_from_batch(sequences, sequence_id), database, seq_index,
                config, scratch,
            )
            place_summed(
                family_results, subfam_results, sequence_id, seq_res, seq_n,
                seq_kmers, select_from_batch(structures, sequence_id),
                database, seq_index, ss_index, config, scratch,
            )
        elif mode != SEARCH_STRUCTURE:
            placed = place_sequence(
                family_results, subfam_results,
                select_from_batch(sequences, sequence_id), sequence_id,
                database, seq_index, config, scratch,
            )

        # the 3Di alone: in the structure mode, or as the fallback of the
        # sequence-then-structure mode
        if (
            mode == SEARCH_STRUCTURE or mode == SEARCH_SEQUENCE_THEN_STRUCTURE
        ) and not placed:
            place_sequence(
                family_results, subfam_results,
                select_from_batch(structures, sequence_id), sequence_id,
                database, ss_index, config, scratch,
            )


class MergeSearch(object):
    def __init__(self, ki, include_extant_genes=False):
        assert ki.db.db.mode == "r", "Database must be opened in read mode."

        # load ki and db
        self.db = ki.db
        self.ki = ki
        self.has_structure = self.db.has_structure()
        # per modality: the k-mer df reference m0 of the index, and the
        # (exponent, reference) of the df correction in use
        self._df_reference = {}
        self.kmer_df_correction = {}

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

    @cached_property
    def hog_id_buff(self):
        # in memory: decoding the HOG ids of the results from it is far
        # cheaper than a read of the (compressed) HDF5 table per result row
        return self.db._db_HOGIDBuffer[:]

    def get_hog_id(self, i):
        ent = self.hog_tab[i]
        s = ent["IDBufferOff"]
        return self.hog_id_buff[s : s + ent["IDLen"]].tobytes().decode("ascii")

    @cached_property
    def ss_kmer_table(self):
        return {
            "idx": self.db._db_Index_SSTableIndex[:],
            "buff": self.db._db_Index_SSTableBuffer[:],
        }

    @lazy_property
    def ss_ref_fam_prob(self):
        return self.db._db_Index_SSFamilyProbability[:]

    @lazy_property
    def ss_ref_hog_prob(self):
        return self.db._db_Index_SSHOGProbability[:]

    @cached_property
    def _search_database(self):
        return SearchDatabase(
            self.trans,
            self.ki.k,
            self.ki.alphabet.DIGITS_AA_LOOKUP,
            self.fam_tab,
            self.hog_tab,
            self.level_arr,
            np.ascontiguousarray(self.hog_tab["FamOff"]),
        )

    def _kmer_index(self, modality, df_exponent=0.0, df_reference=None):
        """
        The k-mer index and models of a modality ('seq' or 'ss'), with the
        k-mer df correction of exponent `df_exponent` and reference
        `df_reference` (None: `kmer_df_reference` of the index).
        """
        attrs = self.db.db.root.Index._v_attrs
        if modality == "seq":
            (table, source, max_df, family_probability, hog_probability) = (
                self.kmer_table,
                SEARCH_SEQUENCE,
                getattr(attrs, "kmer_max_df", 0),
                self.ref_fam_prob,
                self.ref_hog_prob,
            )
        else:
            (table, source, max_df, family_probability, hog_probability) = (
                self.ss_kmer_table,
                SEARCH_STRUCTURE,
                getattr(attrs, "ss_kmer_max_df", 0),
                self.ss_ref_fam_prob,
                self.ss_ref_hog_prob,
            )
        kmer_mask = modality_kmer_mask(
            modality,
            self.ki.k,
            self.ki.alphabet.n,
            stored_min_distinct(attrs, modality),
        )
        if not df_exponent:
            (df_exponent, df_reference) = (0.0, 0.0)
            self.kmer_df_correction.pop(modality, None)
        else:
            if df_reference is None:
                if modality not in self._df_reference:
                    self._df_reference[modality] = kmer_df_reference(
                        table["idx"], max_df, kmer_mask
                    )
                    LOG.info(
                        "k-mer df correction for modality '%s': exponent %g, "
                        "reference df %.2f",
                        modality,
                        df_exponent,
                        self._df_reference[modality],
                    )
                df_reference = self._df_reference[modality]
            self.kmer_df_correction[modality] = (df_exponent, df_reference)
        return KmerIndex(
            table["idx"],
            table["buff"],
            source,
            np.int64(max_df),
            kmer_mask,
            family_probability,
            hog_probability,
            float(df_reference),
            float(df_exponent),
        )

    def merge_search(
        self,
        seqs,
        ids,
        top_n_fams=1,
        alpha=1e-6,
        sst=0.1,
        family_only=False,
        ref_taxon_off=None,
        struct_seqs=(),
        search_mode="auto",
        summed_alpha=None,
        ss_weight=0.5,
        family_selection="choose_scored",
        seq_kmer_df_exponent=1.5,
        ss_kmer_df_exponent=0.0,
        seq_kmer_df_reference=None,
        ss_kmer_df_reference=None,
    ):
        """
        seqs, struct_seqs: the sequences and the 3Di sequences of the queries
        (either may be empty), in the same order.
        search_mode: 'auto' (hybrid with both, else the one given), 'seq',
        'ss', 'sqs', 'sum' or 'hybrid' (see SEARCH_*).
        summed_alpha: threshold on the summed sequence + 3Di scores (sum
        mode, and the second stage of the hybrid mode); None: alpha.
        ss_weight: weight w of the 3Di score in these sums
        (sequence + w * 3Di).
        family_selection: 'filter' or 'choose_scored' (see FAMILY_SELECT_*).
        seq_kmer_df_exponent, ss_kmer_df_exponent: exponent b of the
        query-specific family hit probabilities p_F (m / m0) ** b, with m the
        mean document frequency (number of families) of the query's k-mers,
        used as `select_family_candidates` and `score_summed_candidates`
        describe (see `query_probability_scale`); 0: off.
        seq_kmer_df_reference, ss_kmer_df_reference: m0 (None: the
        posting-weighted mean df of the index, `kmer_df_reference`).
        """
        t0 = time()
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the interval (0, 1]")
        if summed_alpha is None:
            summed_alpha = alpha
        if not 0.0 < summed_alpha <= 1.0:
            raise ValueError("summed_alpha must be in the interval (0, 1]")
        if not ss_weight > 0.0:
            raise ValueError("ss_weight must be positive")
        if family_selection not in FAMILY_SELECTIONS:
            raise ValueError(
                "family_selection must be 'filter' or 'choose_scored', got "
                "{!r}".format(family_selection)
            )
        for exponent in (seq_kmer_df_exponent, ss_kmer_df_exponent):
            if not exponent >= 0.0:
                raise ValueError("kmer_df_exponent must be >= 0")
        for reference in (seq_kmer_df_reference, ss_kmer_df_reference):
            if reference is not None and not reference > 0.0:
                raise ValueError("kmer_df_reference must be positive")

        sbuff = SequenceBuffer(seqs=seqs, ids=ids)
        ssbuff = SequenceBuffer(seqs=struct_seqs, ids=ids)
        if len(struct_seqs) > 0 and not self.has_structure:
            raise ValueError("3Di queries need a database with a 3Di index")
        mode = resolve_search_mode(search_mode, len(seqs) > 0, len(struct_seqs) > 0)
        seq_index = ss_index = None
        if mode != SEARCH_STRUCTURE:
            seq_index = self._kmer_index(
                "seq", seq_kmer_df_exponent, seq_kmer_df_reference
            )
        if mode != SEARCH_SEQUENCE:
            ss_index = self._kmer_index(
                "ss", ss_kmer_df_exponent, ss_kmer_df_reference
            )
            if seq_index is not None and len(sbuff.idx) != len(ssbuff.idx):
                raise ValueError(
                    "Sequence and structure queries must have matching counts"
                )
        qbuff = ssbuff if mode == SEARCH_STRUCTURE else sbuff

        # perform the search
        results = make_search_results(len(qbuff.idx) - 1, top_n_fams)
        config = SearchConfig(
            mode,
            int(top_n_fams),
            float(sst),
            bool(family_only),
            FAMILY_SELECTIONS[family_selection],
            -math.log(alpha),
            math.log(self.fam_tab.size),
            -math.log(summed_alpha),
            float(ss_weight),
        )
        lookup_queries(
            results.families,
            results.subfamilies,
            SequenceBatch(sbuff.buff, sbuff.idx),
            SequenceBatch(ssbuff.buff, ssbuff.idx),
            self._search_database,
            seq_index if seq_index is not None else ss_index,
            ss_index if ss_index is not None else seq_index,
            config,
            numba.get_num_threads(),
        )
        (family_results, subfam_results) = results

        t1 = time()
        td = max((t1 - t0), 1e-3)
        n = len(qbuff.ids)
        LOG.debug(
            "{:.02f} seconds for block of {:d} sequences (~{:.02f} queries/second)".format(
                td, n, n / td
            )
        )

        return self.output_results(
            family_results, subfam_results, qbuff, top_n_fams, ref_taxon_off
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
            "modality",
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
                            "modality": family_results["modality"][i, j],
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
        df["modality"] = df["modality"].map(MODALITY_NAMES)

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
            # load the tables once, not for every result row
            hog_tab = self.hog_tab
            chog_buff = self.db._db_ChildrenHOG[:]
            cprot_buff = self.db._db_ChildrenProt[:]
            df.loc[hog_f, "subfamily_geneset"] = df.loc[hog_f, "hog_offset"].apply(
                lambda i: ",".join(
                    map(
                        self.db.get_prot_id,
                        get_hog_member_prots(i - 1, hog_tab, chog_buff, cprot_buff),
                    )
                )
            )

        # add the hog id
        df.loc[hog_f, "hogid"] = df.loc[hog_f, "hog_offset"].apply(
            lambda i: self.get_hog_id(i - 1)
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
