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
import math
from collections import namedtuple
from time import time

import numba
import numpy as np
import pandas as pd
from numba.typed import List
from property_manager import cached_property, lazy_property

from ._utils import LOG
from .alphabets import get_transform
from .bbinom_coefficients import (
    BBINOM_VALIDITY_POLICY_VERSION,
    bbinom_coefficient_valid_mask,
)
from .family_sort import family_result_sort
from .hierarchy import (
    get_children,
    get_hog_member_prots,
    get_root_leaf_offsets,
    is_taxon_implied,
)
from .index import cumulate_counts_1fam
from .sequence_buffer import SequenceBuffer
from .stat_models import (
    FamilyScoringParameters,
    HogModelParameters,
    family_log_correction,
    filter_family_candidates,
    make_family_model,
    score_family_candidates,
)

SEARCH_SEQUENCE = 1
SEARCH_STRUCTURE = 2
SEARCH_SEQUENCE_THEN_STRUCTURE = 3


################################################################################
# Definitions for namedtuples that are used as composite
# types in pure numba code. Those are to aggregate many of
# related parameters and dispatch the Strategy design pattern
# in numba code without passing dozens of parameters to
# the main search function.

SequenceBatch = namedtuple("SequenceBatch", ("buffer", "offsets"))


@numba.njit(nogil=True)
def select_from_batch(batch: SequenceBatch, seq_id: int):
    return batch.buffer[
        batch.offsets[seq_id] : np.int64(batch.offsets[seq_id + 1] - 1)
    ]


# Modality-specific k-mer data. Statistical models are separate arguments at
# the parallel-kernel boundary, keeping ownership non-overlapping without
# nesting array-bearing namedtuples (which Numba's gufunc lowering rejects).
KmerIndex = namedtuple(
    "KmerIndex",
    (
        "table_idx",
        "table_buff",
        "modality",
        "kmer_filter_max_df",
    ),
)


def make_kmer_index(
    table_idx,
    table_buff,
    modality,
    kmer_filter_max_df,
):
    return KmerIndex(
        table_idx,
        table_buff,
        modality,
        np.int64(kmer_filter_max_df),
    )


SearchDatabase = namedtuple(
    "SearchDatabase",
    ("trans", "k", "digits_lookup", "families", "hogs", "levels"),
)


LookupConfig = namedtuple(
    "LookupConfig",
    (
        "mode",
        "num_threads",
    ),
)


PlacementConfig = namedtuple(
    "PlacementConfig",
    (
        "top_n_families",
        "subfamily_score_threshold",
        "family_only",
    ),
)

# Namedtuple for thread-local arrays used in search.
# Every thread has these arrays allocated for their
# search needs beforehand. This is to avoid allocating
# O(|num_fams|) per-thread everytime search is called.
SearchScratch = namedtuple(
    "SearchScratch",
    (
        "hit_families",
        "hit_hogs",
        "hog_counts",
        "family_counts",
        "family_low_location",
        "family_high_location",
        "num_hit_families",
        "num_hit_hogs",
    ),
)
################################################################################


def resolve_search_mode(mode, has_sequence, has_structure):
    if mode in (None, "auto"):
        if has_sequence and has_structure:
            return SEARCH_SEQUENCE_THEN_STRUCTURE
        if has_sequence:
            return SEARCH_SEQUENCE
        if has_structure:
            return SEARCH_STRUCTURE
        raise ValueError("Search requires sequence or structure queries")

    modes = {
        "seq": (SEARCH_SEQUENCE, has_sequence),
        "ss": (SEARCH_STRUCTURE, has_structure),
        "sqs": (
            SEARCH_SEQUENCE_THEN_STRUCTURE,
            has_sequence and has_structure,
        ),
    }
    try:
        selected, available = modes[mode]
    except KeyError:
        raise ValueError(
            "search_mode must be 'auto', 'seq', 'ss', or 'sqs'"
        ) from None
    if not available:
        raise ValueError(
            "search_mode={!r} requires matching query data and index".format(
                mode
            )
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



## generic functions
@numba.njit(nogil=True)
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


@numba.njit(nogil=True)
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


@numba.njit(nogil=True)
def init_query_family_results(n):
    return np.zeros(n, dtype=QUERY_FAMILY_RESULT_DTYPE)


@numba.njit(nogil=True)
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


@numba.njit(nogil=True)
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
def search_seq_kmers(r1, p1, hog_tab, x_flag, index, scratch, thread_id):
    """
    Perform the kmer search, using the index.
    """
    table_idx = index.table_idx
    table_buff = index.table_buff
    kmer_filter_max_df = index.kmer_filter_max_df

    thread_hog_counts = scratch.hog_counts[thread_id]
    thread_fam_counts = scratch.family_counts[thread_id]
    thread_fam_lowloc = scratch.family_low_location[thread_id]
    thread_fam_highloc = scratch.family_high_location[thread_id]
    thread_hit_fams = scratch.hit_families[thread_id]
    thread_hit_hogs = scratch.hit_hogs[thread_id]
    thread_num_hit_fams = scratch.num_hit_families[thread_id]
    thread_num_hit_hogs = scratch.num_hit_hogs[thread_id]

    # Reinitialize counters modified by the previous call
    for i in range(thread_num_hit_fams):
        family = thread_hit_fams[i]
        thread_fam_counts[family] = 0
        thread_fam_lowloc[family] = -1
        thread_fam_highloc[family] = -1

    for i in range(thread_num_hit_hogs):
        hog = thread_hit_hogs[i]
        thread_hog_counts[hog] = 0

    thread_num_hit_fams = 0
    thread_num_hit_hogs = 0
    n_skipped = 0

    # define the HOG->family column view out of the loop
    # (to avoid refetching the struct field on every k-mer)
    hog2fam = hog_tab["FamOff"]

    # iterate unique k-mers
    for m in range(r1.shape[0]):
        kmer = r1[m]
        loc = p1[m]

        # to ignore k-mers with X
        if kmer == x_flag:
            continue

        # get mapping to HOGs
        lo = table_idx[kmer]
        hi = table_idx[kmer + 1]
        df = hi - lo

        # The build-time information filter is represented by its maximum
        # family document frequency. This is equivalent to the PMI threshold
        # selected by mkdb and keeps all ties at the cutoff.
        if 0 < kmer_filter_max_df < df:
            n_skipped += 1
            continue

        # Single fused pass over the k-mer's HOGs. Each HOG maps to exactly one
        # family, so we update the HOG and family counters together
        for j in range(lo, hi):
            hog = table_buff[j]

            if not thread_hog_counts[hog]:
                thread_hit_hogs[thread_num_hit_hogs] = hog
                thread_num_hit_hogs += 1
            thread_hog_counts[hog] += 1

            fam_off = hog2fam[hog]

            if not thread_fam_counts[fam_off]:
                thread_hit_fams[thread_num_hit_fams] = fam_off
                thread_num_hit_fams += 1
            thread_fam_counts[fam_off] += 1

            # initiate first location
            if thread_fam_lowloc[fam_off] == -1:
                thread_fam_lowloc[fam_off] = loc
                thread_fam_highloc[fam_off] = loc

            # update either lower or higher boundary
            elif loc < thread_fam_lowloc[fam_off]:
                thread_fam_lowloc[fam_off] = loc
            elif loc > thread_fam_highloc[fam_off]:
                thread_fam_highloc[fam_off] = loc

    scratch.num_hit_families[thread_id] = thread_num_hit_fams
    scratch.num_hit_hogs[thread_id] = thread_num_hit_hogs
    return n_skipped


## generic score functions
@numba.njit(nogil=True)
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


@numba.njit(nogil=True)
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
def filter_by_overlap(qres, query_len, fam_highloc, fam_lowloc, k):
    """
    Filters families by sequence coverage.
    """
    for i in range(len(qres)):
        family_id = qres["id"][i]
        qres["overlap"][i] = (
            fam_highloc[family_id] - fam_lowloc[family_id] + k
        ) / query_len

    return qres[qres["overlap"] >= (25 / query_len)]


@numba.njit(nogil=True)
def place_sequence(
    family_results,
    subfam_results,
    sequence,
    sequence_id,
    database,
    index,
    family_model,
    hog_model,
    placement,
    family_scoring,
    scratch,
) -> bool:
    trans = database.trans
    k = database.k
    digits_lookup = database.digits_lookup
    fam_tab = database.families
    hog_tab = database.hogs
    level_arr = database.levels

    table_idx = index.table_idx
    top_n_fams = placement.top_n_families
    sst = placement.subfamily_score_threshold
    family_only = placement.family_only

    query_len = sequence.shape[0]
    n_kmers = query_len - (k - 1)
    x_flag = table_idx.size - 1

    # Double check we don't have short peptides (of len < k)
    if n_kmers == 0:
        return False

    # get sequence k-mers
    (r1, p1, _) = parse_seq(sequence, digits_lookup, n_kmers, k, trans, x_flag)

    # Skip if only one k-mer with X
    if len(r1) > 1:
        pass
    elif r1[0] == x_flag:
        return False

    thread_id = numba.get_thread_id()
    n_skipped = search_seq_kmers(
        r1,
        p1,
        hog_tab,
        x_flag,
        index,
        scratch,
        thread_id,
    )

    # Only unpack the thread-local views needed after k-mer counting.
    thread_hit_fams = scratch.hit_families[thread_id]
    thread_hog_counts = scratch.hog_counts[thread_id]
    thread_fam_counts = scratch.family_counts[thread_id]
    thread_fam_lowloc = scratch.family_low_location[thread_id]
    thread_fam_highloc = scratch.family_high_location[thread_id]
    thread_num_hit_fams = scratch.num_hit_families[thread_id]

    # Identify families of interest
    idx = thread_hit_fams[:thread_num_hit_fams]
    qres = init_query_family_results(len(idx))
    qres["id"][:] = idx
    qres["count"][:] = thread_fam_counts[idx]

    # Effective query length: unique k-mers actually searched, i.e. excluding
    # k-mers dropped by the build-time information filter.
    n = len(r1) - n_skipped

    # cheap filter: is # of k-mers at the expected number?
    qres = filter_family_candidates(qres, n, family_model)
    if len(qres) == 0:
        return False

    # filter out by demanding at least 0.25x query coverage
    qres = filter_by_overlap(
        qres,
        query_len,
        thread_fam_highloc,
        thread_fam_lowloc,
        k,
    )
    if len(qres) == 0:
        return False

    # Apply the model-specific significance bound and exact family scoring
    qres = score_family_candidates(qres, n, family_model, family_scoring)
    if len(qres) == 0:
        return False

    # 5. Store results
    # - a. sort by normcount, then overlap, then p-value for tie-breaking
    qres = family_result_sort(qres, top_n_fams)

    # - b. store results
    family_results["id"][sequence_id, :top_n_fams] = qres["id"][:top_n_fams] + 1
    family_results["pvalue"][sequence_id, :top_n_fams] = qres["pvalue"][:top_n_fams]
    family_results["count"][sequence_id, :top_n_fams] = qres["count"][:top_n_fams]
    family_results["normcount"][sequence_id, :top_n_fams] = qres["normcount"][:top_n_fams]
    family_results["overlap"][sequence_id, :top_n_fams] = qres["overlap"][:top_n_fams]
    family_results["modality"][sequence_id, :top_n_fams] = index.modality

    # 5. Place within families
    for i in range(min(len(qres), top_n_fams)):
        entry = fam_tab[qres["id"][i]]
        hog_s = entry["HOGoff"]
        hog_e = hog_s + entry["HOGnum"]

        if family_only:
            # early exit
            subfam_results["id"][sequence_id, i] = hog_s + 1
            continue

        fam_hog2parent = get_fam_hog2parent(entry, hog_tab)
        fam_level_offsets = get_fam_level_offsets(entry, level_arr)

        # cumulation of counts
        c = thread_hog_counts[hog_s:hog_e].copy()

        cumulate_counts_1fam(c, fam_level_offsets, fam_hog2parent)

        # new expected count, but using old cumulation
        (fam_hog_scores, fam_bestpath) = hog_path_placement(
            c,
            n,
            fam_level_offsets,
            fam_hog2parent,
            thread_hog_counts[hog_s:hog_e],
            hog_model.hog_probability[hog_s:hog_e],
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
        subfam_results["id"][sequence_id, i] = choice + hog_s + 1
        subfam_results["score"][sequence_id, i] = choice_score
        subfam_results["count"][sequence_id, i] = c[int(choice)] if choice_score != 0.0 else 0

    return True


class MergeSearch(object):
    def __init__(
        self,
        ki,
        include_extant_genes=False,
        kmer_percentage=None,
    ):
        assert ki.db.db.mode == "r", "Database must be opened in read mode."

        # load ki and db
        self.db = ki.db
        self.ki = ki
        self.has_structure = self.db.has_structure()

        self.include_extant_genes = include_extant_genes
        if (
            kmer_percentage is not None
            and not np.isclose(float(kmer_percentage), self.ki.kmer_percentage)
        ):
            raise ValueError(
                "Requested kmer_percentage={} does not match the database's "
                "build-time kmer_percentage={}".format(
                    kmer_percentage,
                    self.ki.kmer_percentage,
                )
            )
        self._validate_bbinom_filter_compatibility()

    def _validate_bbinom_filter_compatibility(self):
        """Reject coefficient arrays calibrated for another index filter."""
        attrs = self.db.db.root.Index._v_attrs
        for modality, valid_node, attr_name in (
            ("seq", "/Index/FamilyBBinomValid", "seq_bbinom_kmer_percentage"),
            ("ss", "/Index/SSFamilyBBinomValid", "ss_bbinom_kmer_percentage"),
        ):
            if valid_node not in self.db.db:
                continue
            fitted_percentage = float(getattr(attrs, attr_name, 100.0))
            if not np.isclose(fitted_percentage, self.ki.kmer_percentage):
                raise ValueError(
                    "{} beta-binomial coefficients use kmer_percentage={}, "
                    "but the database index uses {}. Rebuild the database and "
                    "refit/import matching coefficients.".format(
                        modality,
                        fitted_percentage,
                        self.ki.kmer_percentage,
                    )
                )

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
    def ss_kmer_table(self):
        z = self.ki.ss_kmer_table
        return {k: z[k][:] for k in z}

    @lazy_property
    def ss_ref_fam_prob(self):
        return self.db._db_Index_SSFamilyProbability[:]

    @lazy_property
    def ss_ref_hog_prob(self):
        return self.db._db_Index_SSHOGProbability[:]

    @lazy_property
    def ref_fam_bbinom_q_coef(self):
        if "/Index/FamilyBBinomQCoef" in self.db.db:
            return self.db._db_Index_FamilyBBinomQCoef[:]
        return np.empty((0, 0), dtype=np.float64)

    @lazy_property
    def ref_fam_bbinom_kappa_coef(self):
        if "/Index/FamilyBBinomKappaCoef" in self.db.db:
            return self.db._db_Index_FamilyBBinomKappaCoef[:]
        return np.empty((0, 0), dtype=np.float64)

    @lazy_property
    def ref_fam_bbinom_center(self):
        if "/Index/FamilyBBinomLogNCenter" in self.db.db:
            return self.db._db_Index_FamilyBBinomLogNCenter[:]
        return np.empty(0, dtype=np.float64)

    @lazy_property
    def ref_fam_bbinom_scale(self):
        if "/Index/FamilyBBinomLogNScale" in self.db.db:
            return self.db._db_Index_FamilyBBinomLogNScale[:]
        return np.empty(0, dtype=np.float64)

    @lazy_property
    def ref_fam_bbinom_valid(self):
        if "/Index/FamilyBBinomValid" in self.db.db:
            stored = self.db._db_Index_FamilyBBinomValid[:]
            attrs = self.db.db.root.Index._v_attrs
            policy_version = getattr(
                attrs,
                "seq_bbinom_validity_policy_version",
                0,
            )
            valid = bbinom_coefficient_valid_mask(
                stored,
                self.ref_fam_bbinom_q_coef,
                self.ref_fam_bbinom_kappa_coef,
                self.ref_fam_bbinom_center,
                self.ref_fam_bbinom_scale,
                reject_initial_fallback=(
                    policy_version < BBINOM_VALIDITY_POLICY_VERSION
                ),
            )
            rejected = int(np.count_nonzero(stored) - np.count_nonzero(valid))
            if rejected:
                LOG.warning(
                    "Disabled %d invalid legacy sequence beta-binomial fits",
                    rejected,
                )
            return valid
        return np.empty(0, dtype=np.bool_)

    @lazy_property
    def ref_fam_bbinom_n_min(self):
        if "/Index/FamilyBBinomNTrainMin" in self.db.db:
            return self.db._db_Index_FamilyBBinomNTrainMin[:]
        return np.empty(0, dtype=np.uint32)

    @lazy_property
    def ref_fam_bbinom_n_max(self):
        if "/Index/FamilyBBinomNTrainMax" in self.db.db:
            return self.db._db_Index_FamilyBBinomNTrainMax[:]
        return np.empty(0, dtype=np.uint32)

    @lazy_property
    def ss_ref_fam_bbinom_q_coef(self):
        if "/Index/SSFamilyBBinomQCoef" in self.db.db:
            return self.db._db_Index_SSFamilyBBinomQCoef[:]
        return np.empty((0, 0), dtype=np.float64)

    @lazy_property
    def ss_ref_fam_bbinom_kappa_coef(self):
        if "/Index/SSFamilyBBinomKappaCoef" in self.db.db:
            return self.db._db_Index_SSFamilyBBinomKappaCoef[:]
        return np.empty((0, 0), dtype=np.float64)

    @lazy_property
    def ss_ref_fam_bbinom_center(self):
        if "/Index/SSFamilyBBinomLogNCenter" in self.db.db:
            return self.db._db_Index_SSFamilyBBinomLogNCenter[:]
        return np.empty(0, dtype=np.float64)

    @lazy_property
    def ss_ref_fam_bbinom_scale(self):
        if "/Index/SSFamilyBBinomLogNScale" in self.db.db:
            return self.db._db_Index_SSFamilyBBinomLogNScale[:]
        return np.empty(0, dtype=np.float64)

    @lazy_property
    def ss_ref_fam_bbinom_valid(self):
        if "/Index/SSFamilyBBinomValid" in self.db.db:
            stored = self.db._db_Index_SSFamilyBBinomValid[:]
            attrs = self.db.db.root.Index._v_attrs
            policy_version = getattr(
                attrs,
                "ss_bbinom_validity_policy_version",
                0,
            )
            valid = bbinom_coefficient_valid_mask(
                stored,
                self.ss_ref_fam_bbinom_q_coef,
                self.ss_ref_fam_bbinom_kappa_coef,
                self.ss_ref_fam_bbinom_center,
                self.ss_ref_fam_bbinom_scale,
                reject_initial_fallback=(
                    policy_version < BBINOM_VALIDITY_POLICY_VERSION
                ),
            )
            rejected = int(np.count_nonzero(stored) - np.count_nonzero(valid))
            if rejected:
                LOG.warning(
                    "Disabled %d invalid legacy structure beta-binomial fits",
                    rejected,
                )
            return valid
        return np.empty(0, dtype=np.bool_)

    @lazy_property
    def ss_ref_fam_bbinom_n_min(self):
        if "/Index/SSFamilyBBinomNTrainMin" in self.db.db:
            return self.db._db_Index_SSFamilyBBinomNTrainMin[:]
        return np.empty(0, dtype=np.uint32)

    @lazy_property
    def ss_ref_fam_bbinom_n_max(self):
        if "/Index/SSFamilyBBinomNTrainMax" in self.db.db:
            return self.db._db_Index_SSFamilyBBinomNTrainMax[:]
        return np.empty(0, dtype=np.uint32)

    def _family_model(self, modality, policy):
        probability = (
            self.ref_fam_prob
            if modality == "seq"
            else self.ss_ref_fam_prob
        )
        if policy == "binomial":
            return make_family_model(policy, probability)

        if modality == "seq":
            return make_family_model(
                policy,
                probability,
                self.ref_fam_bbinom_q_coef,
                self.ref_fam_bbinom_kappa_coef,
                self.ref_fam_bbinom_center,
                self.ref_fam_bbinom_scale,
                self.ref_fam_bbinom_valid,
                self.ref_fam_bbinom_n_min,
                self.ref_fam_bbinom_n_max,
            )
        return make_family_model(
            policy,
            probability,
            self.ss_ref_fam_bbinom_q_coef,
            self.ss_ref_fam_bbinom_kappa_coef,
            self.ss_ref_fam_bbinom_center,
            self.ss_ref_fam_bbinom_scale,
            self.ss_ref_fam_bbinom_valid,
            self.ss_ref_fam_bbinom_n_min,
            self.ss_ref_fam_bbinom_n_max,
        )

    def _search_components(self, modality, family_model_policy):
        """Build disjoint k-mer, family-model, and HOG-model carriers."""
        if modality == "seq":
            model = self._family_model("seq", family_model_policy)
            return (
                make_kmer_index(
                    self.kmer_table["idx"],
                    self.kmer_table["buff"],
                    SEARCH_SEQUENCE,
                    self.ki.kmer_max_df,
                ),
                model,
                HogModelParameters(self.ref_hog_prob),
            )
        model = self._family_model("ss", family_model_policy)
        return (
            make_kmer_index(
                self.ss_kmer_table["idx"],
                self.ss_kmer_table["buff"],
                SEARCH_STRUCTURE,
                self.ki.ss_kmer_max_df,
            ),
            model,
            HogModelParameters(self.ss_ref_hog_prob),
        )

    def merge_search(
        self,
        seqs,
        struct_seqs,
        ids,
        top_n_fams=1,
        alpha=1e-6,
        family_correction="bonferroni",
        sst=0.1,
        family_only=False,
        ref_taxon_off=None,
        search_mode="auto",
        family_model="auto",
    ):
        t0 = time()
        sbuff = SequenceBuffer(seqs=seqs, ids=ids)
        ssbuff = SequenceBuffer(seqs=struct_seqs, ids=ids)

        # resolve search mode
        has_sequence = len(seqs) > 0
        has_structure = self.has_structure and len(struct_seqs) > 0
        mode = resolve_search_mode(search_mode, has_sequence, has_structure)

        database = SearchDatabase(
            self.trans,
            self.ki.k,
            self.ki.alphabet.DIGITS_AA_LOOKUP,
            self.fam_tab,
            self.hog_tab,
            self.level_arr,
        )

        sequence_components = (
            self._search_components("seq", family_model)
            if mode != SEARCH_STRUCTURE
            else None
        )
        structure_components = (
            self._search_components("ss", family_model)
            if mode != SEARCH_SEQUENCE
            else None
        )

        if sequence_components is None:
            sequence_components = structure_components
        if structure_components is None:
            structure_components = sequence_components

        sequence_index, sequence_family_model, sequence_hog_model = (
            sequence_components
        )
        structure_index, structure_family_model, structure_hog_model = (
            structure_components
        )

        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the interval (0, 1]")

        lookup_config = LookupConfig(mode, numba.get_num_threads())
        placement = PlacementConfig(
            top_n_fams,
            sst,
            family_only,
        )
        family_scoring = FamilyScoringParameters(
            -math.log(alpha),
            family_log_correction(family_correction, self.fam_tab.size),
        )

        data_size = max(len(sbuff.idx) - 1, len(ssbuff.idx) - 1)

        # allocate result arrays
        family_results = np.zeros(
            (data_size, top_n_fams),
            dtype=np.dtype(
                [
                    ("id", np.uint32),
                    ("pvalue", np.float64),
                    ("count", np.uint32),
                    ("score", np.uint32),
                    ("normcount", np.float64),
                    ("overlap", np.float64),
                    ("modality", np.uint8),
                ]
            ),
        )
        subfam_results = np.zeros(
            (data_size, top_n_fams),
            dtype=np.dtype(
                [("id", np.uint32), ("score", np.float64), ("count", np.uint32)]
            ),
        )

        self._lookup(
            family_results,
            subfam_results,
            SequenceBatch(sbuff.buff, sbuff.idx),
            SequenceBatch(ssbuff.buff, ssbuff.idx),
            database,
            sequence_index,
            sequence_family_model,
            sequence_hog_model,
            structure_index,
            structure_family_model,
            structure_hog_model,
            lookup_config,
            placement,
            family_scoring,
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
            family_results, subfam_results, sbuff, ssbuff, top_n_fams, ref_taxon_off
        )

    @cached_property
    def _hog_id_buff(self):
        # the HOG id character buffer, kept in memory: decoding ids from it is
        # ~1000x cheaper than a per-row lookup in the (compressed) HDF5 table
        return self.db.hog_id_buffer[:]

    @cached_property
    def _hog_id_cache(self):
        return {}

    @cached_property
    def _tax_ids(self):
        # taxon ids are few and reused by almost every query: decode once
        return [x.decode("ascii") for x in self.tax_tab["ID"]]

    @cached_property
    def _children_hog(self):
        return self.db._db_ChildrenHOG[:]

    @cached_property
    def _children_prot(self):
        return self.db._db_ChildrenProt[:]

    @cached_property
    def _prot_id_cols(self):
        # (offset, length, buffer) of the protein ids, or None on databases
        # that predate the id buffer (<2.3.0) and store the id inline
        prot_tab = self.db.protein_table
        if "ID" in prot_tab.colinstances:
            return None
        return (
            prot_tab.col("IDBufferOff"),
            prot_tab.col("IDLen"),
            self.db.protein_id_buffer[:],
        )

    def get_hog_ids(self, hog_offs):
        """Decode the ids of the given (0-based) HOG offsets."""
        cache = self._hog_id_cache
        buff = self._hog_id_buff
        hog_tab = self.hog_tab
        ids = []
        for off in hog_offs:
            hog_id = cache.get(off)
            if hog_id is None:
                ent = hog_tab[off]
                s = ent["IDBufferOff"]
                hog_id = buff[s : s + ent["IDLen"]].tobytes().decode("ascii")
                cache[off] = hog_id
            ids.append(hog_id)
        return ids

    def get_prot_ids(self, prot_offs):
        """Decode the ids of the given (0-based) protein offsets."""
        cols = self._prot_id_cols
        if cols is None:
            return [self.db.get_prot_id(i) for i in prot_offs]
        (id_off, id_len, buff) = cols
        return [
            buff[id_off[i] : id_off[i] + id_len[i]].tobytes().decode("ascii")
            for i in prot_offs
        ]

    def output_results(
        self,
        family_results,
        subfam_results,
        sbuff,
        ssbuff,
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
        data_size = max(len(sbuff.idx) - 1, len(ssbuff.idx) - 1)

        # rows are (query, family rank) pairs: the best family is always
        # reported, the other ranks only when they carry a subfamily placement
        if top_n_fams == 1:
            qseq_off = np.arange(data_size, dtype=np.int64)
            rank_off = np.zeros(data_size, dtype=np.int64)
        else:
            keep = subfam_results["id"][:data_size] > 0
            keep[:, 0] = True
            (qseq_off, rank_off) = np.nonzero(keep)

        nrows = len(qseq_off)
        if nrows == 0:
            return pd.DataFrame()

        def nullable_uint(values, na_mask=None):
            arr = pd.array(values, dtype="UInt32")
            if na_mask is None:
                na_mask = values == 0
            if np.any(na_mask):
                arr[na_mask] = pd.NA
            return arr

        def float_with_na(values):
            values = values.copy()
            values[values == 0] = np.nan
            return values

        # query ids and lengths (structure-only searches have no sequence)
        qbuff = sbuff if len(sbuff.buff) else ssbuff
        qseqid = qbuff.ids[qseq_off].tolist()
        if qbuff.ids.dtype.kind != "U":
            qseqid = list(map(str, qseqid))
        qseqlen = qbuff.idx[qseq_off + 1] - qbuff.idx[qseq_off]

        # placed queries: everything below is only defined for those
        hog_off = subfam_results["id"][qseq_off, rank_off].astype(np.int64) - 1
        unplaced = hog_off < 0
        placed_rows = np.nonzero(~unplaced)[0]
        placed_hogs = hog_off[placed_rows]

        hogid = [None] * nrows
        hoglevel = [None] * nrows
        tax_ids = self._tax_ids
        tax_offs = self.hog_tab["TaxOff"][placed_hogs].tolist()
        hog_ids = self.get_hog_ids(placed_hogs.tolist())
        for (i, row) in enumerate(placed_rows.tolist()):
            hogid[row] = hog_ids[i]
            hoglevel[row] = tax_ids[tax_offs[i]]

        subfamily_medianseqlen = pd.array(np.zeros(nrows, dtype=np.uint32), dtype="UInt32")
        subfamily_medianseqlen[unplaced] = pd.NA
        subfamily_medianseqlen[placed_rows] = self.hog_tab["MedianSeqLen"][placed_hogs]

        decision_raw = family_results["modality"][qseq_off, rank_off]
        modality = np.empty(nrows, dtype=object)
        modality[:] = pd.NA
        modality[decision_raw == 1] = "seq"
        modality[decision_raw == 2] = "ss"

        data = {
            "qseqid": qseqid,
            "hogid": hogid,
            "hoglevel": hoglevel,
            "family_p": float_with_na(family_results["pvalue"][qseq_off, rank_off]),
            "family_count": nullable_uint(family_results["count"][qseq_off, rank_off]),
            "family_normcount": float_with_na(
                family_results["normcount"][qseq_off, rank_off]
            ),
            "modality": modality,
            "subfamily_score": float_with_na(
                subfam_results["score"][qseq_off, rank_off]
            ),
            "subfamily_count": nullable_uint(
                subfam_results["count"][qseq_off, rank_off]
            ),
            "qseqlen": qseqlen,
            "subfamily_medianseqlen": subfamily_medianseqlen,
            "qseq_overlap": float_with_na(
                family_results["overlap"][qseq_off, rank_off]
            ),
        }

        if self.include_extant_genes:
            # add extant gene list if necessary
            HEADER.append("subfamily_geneset")
            geneset = np.empty(nrows, dtype=object)
            geneset[:] = pd.NA
            for (i, row) in enumerate(placed_rows.tolist()):
                geneset[row] = ",".join(
                    self.get_prot_ids(
                        get_hog_member_prots(
                            placed_hogs[i].item(),
                            self.hog_tab,
                            self._children_hog,
                            self._children_prot,
                        )
                    )
                )
            data["subfamily_geneset"] = geneset

        # compute taxonomic congruences
        if ref_taxon_off:
            q2closest_taxon = get_closest_taxa_from_ref(
                (placed_hogs + 1).astype(np.uint32),
                ref_taxon_off,
                self.tax_tab,
                self.hog_tab,
                self._children_hog,
            )
            HEADER.append("closest_taxa")
            closest_taxa = np.empty(nrows, dtype=object)
            closest_taxa[:] = pd.NA
            for (i, row) in enumerate(placed_rows.tolist()):
                x = q2closest_taxon[i]
                if x != -1:
                    closest_taxa[row] = tax_ids[x]
            data["closest_taxa"] = closest_taxa

        return pd.DataFrame(data)[HEADER]

    @lazy_property
    def _lookup(self):
        def func(
            family_results,
            subfam_results,
            sequences,
            structures,
            database,
            sequence_index,
            sequence_family_model,
            sequence_hog_model,
            structure_index,
            structure_family_model,
            structure_hog_model,
            lookup_config,
            placement,
            family_scoring,
        ):
            # allocate a collection of thread-local data structures
            scratch = SearchScratch(
                np.zeros(
                    (lookup_config.num_threads, database.families.size),
                    dtype=np.int32,
                ),
                np.zeros(
                    (lookup_config.num_threads, database.hogs.size),
                    dtype=np.int32,
                ),
                np.zeros(
                    (lookup_config.num_threads, database.hogs.size),
                    dtype=np.uint16,
                ),
                np.zeros(
                    (lookup_config.num_threads, database.families.size),
                    dtype=np.uint16,
                ),
                np.full(
                    (lookup_config.num_threads, database.families.size),
                    -1,
                    dtype=np.int32,
                ),
                np.full(
                    (lookup_config.num_threads, database.families.size),
                    -1,
                    dtype=np.int32,
                ),
                np.zeros(lookup_config.num_threads, dtype=np.uint32),
                np.zeros(lookup_config.num_threads, dtype=np.uint32),
            )

            n_iter = (
                structures.offsets.size - 1
                if lookup_config.mode == SEARCH_STRUCTURE
                else sequences.offsets.size - 1
            )

            for sequence_id in numba.prange(n_iter):
                placed = False

                # search sequence
                if lookup_config.mode != SEARCH_STRUCTURE:
                    sequence = select_from_batch(sequences, sequence_id)
                    placed = place_sequence(
                        family_results,
                        subfam_results,
                        sequence,
                        sequence_id,
                        database,
                        sequence_index,
                        sequence_family_model,
                        sequence_hog_model,
                        placement,
                        family_scoring,
                        scratch,
                    )

                # if failed and we have structure, search structure
                if lookup_config.mode != SEARCH_SEQUENCE and not placed:
                    structure = select_from_batch(structures, sequence_id)
                    place_sequence(
                        family_results,
                        subfam_results,
                        structure,
                        sequence_id,
                        database,
                        structure_index,
                        structure_family_model,
                        structure_hog_model,
                        placement,
                        family_scoring,
                        scratch,
                    )

        return numba.jit(func, parallel=True, nopython=True, nogil=True, cache=True)
        #return func
