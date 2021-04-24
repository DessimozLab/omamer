'''
    OMAmer - tree-driven and alignment-free protein assignment to sub-families

    (C) 2019-2020 Victor Rossier <victor.rossier@unil.ch> and
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
'''
from property_manager import lazy_property, cached_property
import numba
import numpy as np
import tables
import os
import pandas as pd
import sys
from scipy import special, stats
from math import sqrt
from time import time

from ._utils import LOG
from .index import get_transform, SequenceBuffer, QuerySequenceBuffer
from .hierarchy import (
    get_root_leaf_offsets, 
    get_hog_taxon_levels, 
    get_hog_member_prots,
    is_taxon_implied,
    get_children
)

# (numba does not like nested methods)
## generic functions
@numba.njit
def get_fam_hog2parent(fam_ent, hog_tab):
    '''
    get HOG parent offsets of a single family
    '''
    hog_off = fam_ent['HOGoff']
    hog2parent_tmp = hog_tab['ParentOff'][hog_off:hog_off + fam_ent['HOGnum']]
    if hog2parent_tmp.size >1:
        return np.append(np.array([-1], dtype=np.int64), hog2parent_tmp[1:] - np.int64(hog_off))
    else:
        return hog2parent_tmp

@numba.njit
def get_fam_level_offsets(fam_ent, level_arr):
    '''
    get HOG level offsets of a single family
    '''
    level_off = fam_ent["LevelOff"]
    level_num = fam_ent["LevelNum"]
    fam_level_offsets = level_arr[level_off : np.int64(level_off + level_num + 2)]

    # because specific for a single family, reinitizialize offsets of family levels
    return fam_level_offsets - fam_level_offsets[0]


## search functions
@numba.njit
def custom_unique1d(ar):
    """
    adapted from np._unique1d for numba
    """
    perm = ar.argsort(kind='mergesort')# if return_index else 'quicksort')
    aux = ar[perm]

    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    
    idx = np.concatenate((np.nonzero(mask)[0], np.array([mask.size])))

    return aux[mask], perm[mask], np.diff(idx)

@numba.njit
def parse_seq(
    s, DIGITS_AA_LOOKUP, n_kmers, k, trans, x_char, x_flag):
    '''
    get the sequence unique k-mers and non ambiguous locations (when truly unique)
    '''
    s_norm = DIGITS_AA_LOOKUP[s]
    r = np.zeros(n_kmers, dtype=np.uint32)  # max kmer 7
    for i in numba.prange(n_kmers):
        # numba can't do np.dot with non-float
        for j in numba.prange(k):
            r[i] += trans[j] * s_norm[i + j]
        # does k-mer contain any X?
        x_seen = np.any(s_norm[i:i+k] == x_char)
        # if yes, replace it by the x_flag
        r[i] = r[i] if not x_seen else x_flag
    r1, idx, counts = custom_unique1d(r)
    
    # get locations of truly unique k-mers
    p1 = np.full(r1.size, -1, dtype=np.int32)
    idx_c1 = np.argwhere(counts == 1).flatten()
    p1[idx_c1] = idx[idx_c1]

    return r1, p1

@numba.njit
def search_seq_kmers(
    r1, p1, hog_tab, fam_tab, x_flag, table_idx, table_buff):
    
    ## search k-mer table
    hog_counts = np.zeros(hog_tab.size, dtype=np.uint16)
    fam_counts = np.zeros(fam_tab.size, dtype=np.uint16)

    # store total occurences of query k-mers
    query_occ = np.uint32(0)

    # store total occurence of query-hog k-mers
    hog_occurs = np.zeros(hog_tab.size, dtype=np.uint32)

    # store lowest and higher k-mer locations
    fam_lowloc = np.full(fam_tab.size, -1, dtype=np.int32)
    fam_highloc = np.full(fam_tab.size, -1, dtype=np.int32)

    # iterate unique k-mers
    for m in numba.prange(r1.shape[0]):
        kmer = r1[m]
        loc = p1[m]

        # to ignore k-mers with X
        if kmer == x_flag:
            continue
        else:
            pass

        # get mapping to HOGs
        x = table_idx[kmer : kmer + 2]
        hogs = table_buff[x[0] : x[1]]
        fams = hog_tab['FamOff'][hogs]
        hog_counts[hogs] += np.uint16(1)
        fam_counts[fams] += np.uint16(1)
        query_occ += np.uint32(hogs.size)
        hog_occurs[hogs] += np.uint32(hogs.size)

        # store lowest and highest locations
        for fam_off in fams:

            # initiate first location
            if fam_lowloc[fam_off] == -1:
                fam_lowloc[fam_off] = loc
                fam_highloc[fam_off] = loc

            # update either lower or higher boundary
            elif loc < fam_lowloc[fam_off]:
                fam_lowloc[fam_off] = loc
            elif loc > fam_highloc[fam_off]:
                fam_highloc[fam_off] = loc

    return hog_counts, fam_counts, query_occ, hog_occurs, fam_lowloc, fam_highloc

@numba.njit
def get_top_m_fams(
    fam_counts, top_m_fams, cum_mode, fam_tab, hog_counts, hog_tab, level_arr, fam_filter):
    '''
    1. get the top m families from summed k-mer counts
    2. recalculate best root-to-leaf counts if cum_mode==max
    option to filter some families for validation
    '''
    # filter families
    if fam_filter.size > 0:
        fam_offsets = np.arange(fam_counts.size)[fam_filter]
        
        # get top m families from summed k-mer counts
        top_fam = fam_offsets[np.argsort(fam_counts[fam_filter])[::-1][:top_m_fams]]        
    
    else:
        top_fam = np.argsort(fam_counts)[::-1][:top_m_fams]

    top_fam_counts = fam_counts[top_fam]

    # cumulated queryHOG counts for the top m families
    # small optimization: remember the fam_hog_cumcounts for the top n families for next step
    if cum_mode == 'max':

        # iterate over top n families
        for fam_rank in numba.prange(top_m_fams):
            fam_off = top_fam[fam_rank]
            fam_ent = fam_tab[fam_off]
            fam_hog_off = fam_ent['HOGoff']
            fam_hog_nr = fam_ent['HOGnum']

            # compute the cumulated HOG counts for that family
            fam_hog_cumcounts = hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr].copy()

            fam_hog2parent = get_fam_hog2parent(fam_ent, hog_tab)
            fam_level_offsets = get_fam_level_offsets(fam_ent, level_arr)

            cumulate_counts_1fam(fam_hog_cumcounts, fam_level_offsets, fam_hog2parent, _sum, _max)

            # replace summed counts by maxed counts (from highest scoring root-to-leaf path)
            top_fam_counts[fam_rank] = fam_hog_cumcounts[0]
    
    return top_fam, top_fam_counts


## functions to cumulate HOG k-mer counts
@numba.njit
def _max(x, y):
    return max(x, y)

@numba.njit
def _sum(x, y):
        return x + y 

@numba.njit
def cumulate_counts_1fam(
    hog_cum_counts, fam_level_offsets, hog2parent, cum_fun, prop_fun):

    current_best_child_count = np.zeros(hog_cum_counts.shape, dtype=np.uint64)

    # iterate over level offsets backward
    for i in range(fam_level_offsets.size - 2):
        x = fam_level_offsets[-i - 3 : -i - 1]

        # when reaching level, sum all hog counts with their best child count
        hog_cum_counts[x[0] : x[1]] = cum_fun(
            hog_cum_counts[x[0] : x[1]], current_best_child_count[x[0] : x[1]]
        )

        # update current_best_child_count of the parents of the current hogs
        for j in range(x[0], x[1]):
            parent_off = hog2parent[j]

            # only if parent exists
            if parent_off != -1:
                c = current_best_child_count[hog2parent[j]]
                current_best_child_count[hog2parent[j]] = prop_fun(
                    c, hog_cum_counts[j]
                )

@numba.njit(parallel=True, nogil=True)
def cumulate_counts_nfams(
    hog_counts, fam_tab, level_arr, hog2parent, main_fun, cum_fun, prop_fun):

    hog_cum_counts = hog_counts.copy()

    for fam_off in numba.prange(fam_tab.size):
        entry = fam_tab[fam_off]
        level_off = entry["LevelOff"]
        level_num = entry["LevelNum"]
        fam_level_offsets = level_arr[
            level_off : np.int64(level_off + level_num + 2)
        ]

        main_fun(hog_cum_counts, fam_level_offsets, hog2parent, cum_fun, prop_fun)

    return hog_cum_counts


## generic score functions
@numba.njit
def store_bestpath(hog_offsets, parent_offsets, fam_bestpath, fam_hog_scores, pv_score):
    
    # keep HOGs descending from the best path
    cands = hog_offsets[fam_bestpath[parent_offsets]]

    # get the score of these candidates
    cands_scores = fam_hog_scores[cands]

    # find the candidate HOGs offsets with the higher count (>0) at this level
    if cands_scores.size > 0:

        # need smallest
        if pv_score:
            cands_offsets = np.where(cands_scores == np.min(cands_scores))[0]
        # need > 0 and max
        else:
            cands_offsets = np.where(
                (cands_scores > 0) & (cands_scores == np.max(cands_scores))
            )[0]

        # if a single candidate, update the best path. Else, stop because of tie
        if cands_offsets.size == 1:
            fam_bestpath[cands[cands_offsets]] = True

# probability to observe the k-mer in a set of size 1
@numba.njit
def kmer_prob_mash(alphabet_n, k):
    '''theoretical probability inspired from Mash paper'''
    return (1 / alphabet_n) ** k

@numba.njit
def kmer_prob_freq(kmer_occurs, all_kmer_occurs):
    '''number of independent observation of k-mer divided by total number of independent all k-mers'''
    return kmer_occurs / all_kmer_occurs

@numba.njit
def query_kmer_prob_freq(query_occurs, query_counts, all_kmer_occurs):
    '''average of the kmer_prob_freq over query k-mer set'''
    av_kmer_occurs = query_occurs / query_counts
    return kmer_prob_freq(av_kmer_occurs, all_kmer_occurs)
      
# probability to observe the k-mer in a HOG k-mer set
@numba.njit
def get_kmer_bernoulli(kmer_prob, hog_counts):
    return 1 - (1 - kmer_prob) ** hog_counts

@numba.njit
def get_expected_kmer_counts(query_counts, kmer_bernoulli):
    return query_counts * kmer_bernoulli


## naive scoring schemes
@numba.njit
def norm_fam_querysize(fam_counts, query_counts):
    return fam_counts / query_counts

@numba.njit
def norm_fam_querysize_hogsize(fam_counts, alphabet_n, k, ref_fam_counts, query_counts):
    # compute expected number of k-mer matches
    kmer_prob = kmer_prob_mash(alphabet_n, k)
    kmer_bernoulli = get_kmer_bernoulli(kmer_prob, ref_fam_counts)
    exp_kmer_counts = get_expected_kmer_counts(query_counts, kmer_bernoulli)
    
    return (fam_counts - exp_kmer_counts) / query_counts

@numba.njit
def norm_fam_querysize_kmerfreq():
    '''
    place to weight each k-mer by frequency. maybe inspire here from TD-IDF
    '''
    pass

@numba.njit
def norm_fam_querysize_hogsize_kmerfreq(fam_counts, query_occurs, query_counts, all_kmer_occurs, ref_fam_counts):
    # compute expected number of k-mer matches
    kmer_prob = query_kmer_prob_freq(query_occurs, query_counts, all_kmer_occurs)
    kmer_bernoulli = get_kmer_bernoulli(kmer_prob, ref_fam_counts)
    exp_kmer_counts = get_expected_kmer_counts(query_counts, kmer_bernoulli)
    
    return (fam_counts - exp_kmer_counts) / query_counts

@numba.njit
def norm_hog_querysize(
    fam_hog_cumcounts, query_counts, fam_level_offsets, hog2parent, fam_hog_counts):

    fam_hog_scores = np.zeros(fam_hog_cumcounts.shape, dtype=np.float64)
    fam_bestpath = np.full(fam_hog_cumcounts.shape, False)
    revcum_counts = np.full(fam_hog_cumcounts.shape, query_counts, dtype=np.uint16)

    # compute and store root-HOG score
    fam_hog_scores[0] = np.float64(fam_hog_cumcounts[0]) / query_counts

    # set the root-HOG as best path
    fam_bestpath[0] = True
    
    # loop through hog levels
    for i in range(1, fam_level_offsets.size - 2):
        x = fam_level_offsets[i : i + 2]
        hog_offsets = np.arange(x[0], x[1])

        # grab parents
        parent_offsets = hog2parent[hog_offsets]

        # update query revcumcount, basically substracting parent counts from query count
        qh_count = revcum_counts[parent_offsets] - fam_hog_counts[parent_offsets]
        revcum_counts[hog_offsets] = qh_count

        # compute and store score
        fam_hog_scores[hog_offsets] = fam_hog_cumcounts[hog_offsets] / qh_count

        # store bestpath
        store_bestpath(hog_offsets, parent_offsets, fam_bestpath, fam_hog_scores, pv_score=False)

    return fam_hog_scores, fam_bestpath

@numba.njit
def norm_hog_querysize_hogsize(
    fam_hog_cumcounts, query_counts, alphabet_n, k, fam_ref_hog_counts, fam_level_offsets, hog2parent, fam_hog_counts):

    fam_hog_scores = np.zeros(fam_hog_cumcounts.shape, dtype=np.float64)
    fam_bestpath = np.full(fam_hog_cumcounts.shape, False)
    revcum_counts = np.full(fam_hog_cumcounts.shape, query_counts, dtype=np.uint16)
    
    ## root-HOG score
    # compute expected number of k-mer matches
    kmer_prob = kmer_prob_mash(alphabet_n, k)
    kmer_bernoulli = get_kmer_bernoulli(kmer_prob, fam_ref_hog_counts[0])
    exp_kmer_counts = get_expected_kmer_counts(query_counts, kmer_bernoulli)
    
    fam_hog_scores[0] = (np.float64(fam_hog_cumcounts[0]) - exp_kmer_counts) / query_counts

    # set the root-HOG as best path
    fam_bestpath[0] = True
    
    # loop through hog levels
    for i in range(1, fam_level_offsets.size - 2):
        x = fam_level_offsets[i : i + 2]
        hog_offsets = np.arange(x[0], x[1])

        # grab parents
        parent_offsets = hog2parent[hog_offsets]

        # update query revcumcount, basically substracting parent counts from query count
        qh_count = revcum_counts[parent_offsets] - fam_hog_counts[parent_offsets]
        revcum_counts[hog_offsets] = qh_count

        ## HOG score
        # compute expected number of k-mer matches
        kmer_prob = kmer_prob_mash(alphabet_n, k)
        kmer_bernoulli = get_kmer_bernoulli(kmer_prob, fam_ref_hog_counts[hog_offsets])
        exp_kmer_counts = get_expected_kmer_counts(qh_count, kmer_bernoulli)

        fam_hog_scores[hog_offsets] = (fam_hog_cumcounts[hog_offsets] - exp_kmer_counts) / qh_count

        # store bestpath
        store_bestpath(hog_offsets, parent_offsets, fam_bestpath, fam_hog_scores, pv_score=False)

    return fam_hog_scores, fam_bestpath

@numba.njit
def norm_hog_querysize_kmerfreq():
    pass
    
@numba.njit
def norm_hog_querysize_hogsize_kmerfreq(
    fam_hog_cumcounts, query_counts, query_occurs, all_kmer_occurs, fam_ref_hog_counts, fam_level_offsets, hog2parent, fam_hog_counts, fam_hog_occurs):

    fam_hog_scores = np.zeros(fam_hog_cumcounts.shape, dtype=np.float64)
    fam_bestpath = np.full(fam_hog_cumcounts.shape, False)
    revcum_counts = np.full(fam_hog_cumcounts.shape, query_counts, dtype=np.uint16)
    revcum_occurs = np.full(fam_hog_cumcounts.shape, query_occurs, dtype=np.uint32)

    ## root-HOG score
    # compute expected number of k-mer matches
    kmer_prob = query_kmer_prob_freq(query_occurs, query_counts, all_kmer_occurs)
    kmer_bernoulli = get_kmer_bernoulli(kmer_prob, fam_ref_hog_counts[0])
    exp_kmer_counts = get_expected_kmer_counts(query_counts, kmer_bernoulli)

    fam_hog_scores[0] = (np.float64(fam_hog_cumcounts[0]) - exp_kmer_counts) / query_counts

    # set the root-HOG as best path
    fam_bestpath[0] = True
    
    # loop through hog levels
    for i in range(1, fam_level_offsets.size - 2):
        x = fam_level_offsets[i : i + 2]
        hog_offsets = np.arange(x[0], x[1])

        # grab parents
        parent_offsets = hog2parent[hog_offsets]

        # update query revcumcount and revcumoccur, basically substracting parent counts/occurs from query counts/occurs
        qh_count = revcum_counts[parent_offsets] - fam_hog_counts[parent_offsets]
        qh_occur = revcum_occurs[parent_offsets] - fam_hog_occurs[parent_offsets]
        revcum_counts[hog_offsets] = qh_count
        revcum_occurs[hog_offsets] = qh_occur

        ## HOG score
        # compute expected number of k-mer matches
        kmer_prob = query_kmer_prob_freq(qh_occur, qh_count, all_kmer_occurs)
        kmer_bernoulli = get_kmer_bernoulli(kmer_prob, fam_ref_hog_counts[hog_offsets])
        exp_kmer_counts = get_expected_kmer_counts(qh_count, kmer_bernoulli)

        fam_hog_scores[hog_offsets] = (fam_hog_cumcounts[hog_offsets] - exp_kmer_counts) / qh_count

        # store bestpath
        store_bestpath(hog_offsets, parent_offsets, fam_bestpath, fam_hog_scores, pv_score=False)

    return fam_hog_scores, fam_bestpath


## probabilistic scoring schemes
def poisson_log_pmf(k, lda):
    return k * np.lib.scimath.log(lda) - lda - special.gammaln(k + 1)

def compute_log_poisson_pvalue(n, hog_cum_counts, lamb):
        
    # probabilities for tail x values
    tail_size = n - hog_cum_counts 
    tail_log_probs = poisson_log_pmf(np.arange(hog_cum_counts, n + 1), np.full(tail_size + 1, lamb))

    # sum of these tail probabilities
    return special.logsumexp(tail_log_probs)

def compute_log_normal_pvalue(hog_cum_counts, mean, sd):
    # actually more stable to use to use the log survival function
    return stats.norm.logsf(hog_cum_counts, loc=mean, scale=sd)

    #return np.lib.scimath.log(np.sum(stats.norm.pdf(np.arange(hog_cum_counts, n + 1), loc=mean, scale=sd)))

def compute_cdist_pvalue(cdist, perm_counts, hog_cum_counts):
    '''
    should work for any continuous function
    '''
    params = cdist.fit(perm_counts)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    return cdist.logsf(hog_cum_counts, loc=loc, scale=scale, *arg)

def compute_fam_mash_pvalue(alphabet_n, k, ref_fam_counts, query_counts, fam_counts):
    
    # probability to draw a k-mer in family k-mer set
    kmer_prob = kmer_prob_mash(alphabet_n, k)
    kmer_bernoulli = get_kmer_bernoulli(kmer_prob, ref_fam_counts)

    # log p-value
    fam_scores = np.zeros(fam_counts.size, np.float64)
    for i in range(fam_counts.size): 

        # !min(hog size, query size) instead of query size
        n = np.int64(min(query_counts, ref_fam_counts[i]))
        fam_scores[i] = compute_log_poisson_pvalue(n, fam_counts[i], n * kmer_bernoulli[i])
        
    return fam_scores

def compute_fam_kmerfreq_pvalue(query_occurs, query_counts, all_kmer_occurs, ref_fam_counts, fam_counts):
    
    # probability to draw a k-mer in family k-mer set
    kmer_prob = query_kmer_prob_freq(query_occurs, query_counts, all_kmer_occurs)
    kmer_bernoulli = get_kmer_bernoulli(kmer_prob, ref_fam_counts)

    # log p-value
    fam_scores = np.zeros(fam_counts.size, np.float64)
    for i in range(fam_counts.size): 

        # !min(hog size, query size) instead of query size
        n = np.int64(min(query_counts, ref_fam_counts[i]))
        fam_scores[i] = compute_log_poisson_pvalue(n, fam_counts[i], n * kmer_bernoulli[i])
        
    return fam_scores

def compute_hog_mash_pvalue(
    fam_hog_cumcounts, query_counts, alphabet_n, k, fam_ref_hog_counts, fam_level_offsets, hog2parent, fam_hog_counts):

    fam_hog_scores = np.zeros(fam_hog_cumcounts.shape, dtype=np.float64)
    fam_bestpath = np.full(fam_hog_cumcounts.shape, False)
    revcum_counts = np.full(fam_hog_cumcounts.shape, query_counts, dtype=np.uint16)
    
    ## root-HOG p-value
    kmer_prob = kmer_prob_mash(alphabet_n, k)
    kmer_bernoulli = get_kmer_bernoulli(kmer_prob, fam_ref_hog_counts[0])

    # !min(hog size, query size) instead of query size
    n = np.int64(min(query_counts, fam_ref_hog_counts[0]))
    fam_hog_scores[0] = compute_log_poisson_pvalue(n, fam_hog_cumcounts[0], n * kmer_bernoulli)
    
    # set the root-HOG as best path
    fam_bestpath[0] = True
    
    # loop through hog levels
    for i in range(1, fam_level_offsets.size - 2):
        x = fam_level_offsets[i : i + 2]
        hog_offsets = np.arange(x[0], x[1])

        # grab parents
        parent_offsets = hog2parent[hog_offsets]

        # update query revcumcount, basically substracting parent counts from query count
        qh_count = revcum_counts[parent_offsets] - fam_hog_counts[parent_offsets]
        revcum_counts[hog_offsets] = qh_count

        ## HOG score
        kmer_bernoulli = get_kmer_bernoulli(kmer_prob, fam_ref_hog_counts[hog_offsets])
        
        for j in range(hog_offsets.size):
            if fam_hog_cumcounts[hog_offsets[j]] > 0:
                
                # !min(hog size, query size) instead of query size
                n = np.int64(min(qh_count[j], fam_ref_hog_counts[hog_offsets[j]]))
                fam_hog_scores[hog_offsets[j]] = compute_log_poisson_pvalue(
                    n, fam_hog_cumcounts[hog_offsets[j]], n * kmer_bernoulli[j])
            else:
                fam_hog_scores[hog_offsets[j]] = 0.0
            
        # store bestpath
        store_bestpath(hog_offsets, parent_offsets, fam_bestpath, fam_hog_scores, pv_score=True)

    return fam_hog_scores, fam_bestpath

def compute_hog_kmerfreq_pvalue(
    fam_hog_cumcounts, query_counts, query_occurs, all_kmer_occurs, fam_ref_hog_counts, fam_level_offsets, hog2parent, fam_hog_counts, fam_hog_occurs):

    fam_hog_scores = np.zeros(fam_hog_cumcounts.shape, dtype=np.float64)
    fam_bestpath = np.full(fam_hog_cumcounts.shape, False)
    revcum_counts = np.full(fam_hog_cumcounts.shape, query_counts, dtype=np.uint16)
    revcum_occurs = np.full(fam_hog_cumcounts.shape, query_occurs, dtype=np.uint32)

    ## root-HOG score
    # compute expected number of k-mer matches
    kmer_prob = query_kmer_prob_freq(query_occurs, query_counts, all_kmer_occurs)
    kmer_bernoulli = get_kmer_bernoulli(kmer_prob, fam_ref_hog_counts[0])

    # !min(hog size, query size) instead of query size
    n = np.int64(min(query_counts, fam_ref_hog_counts[0]))
    fam_hog_scores[0] = compute_log_poisson_pvalue(n, fam_hog_cumcounts[0], n * kmer_bernoulli)

    # set the root-HOG as best path
    fam_bestpath[0] = True
    
    # loop through hog levels
    for i in range(1, fam_level_offsets.size - 2):
        x = fam_level_offsets[i : i + 2]
        hog_offsets = np.arange(x[0], x[1])

        # grab parents
        parent_offsets = hog2parent[hog_offsets]

        # update query revcumcount and revcumoccur, basically substracting parent counts/occurs from query counts/occurs
        qh_count = revcum_counts[parent_offsets] - fam_hog_counts[parent_offsets]
        qh_occur = revcum_occurs[parent_offsets] - fam_hog_occurs[parent_offsets]
        revcum_counts[hog_offsets] = qh_count
        revcum_occurs[hog_offsets] = qh_occur

        ## HOG score
        kmer_prob = query_kmer_prob_freq(qh_occur, qh_count, all_kmer_occurs)
        kmer_bernoulli = get_kmer_bernoulli(kmer_prob, fam_ref_hog_counts[hog_offsets])

        for j in range(hog_offsets.size):
            if fam_hog_cumcounts[hog_offsets[j]] > 0:
                
                # !min(hog size, query size) instead of query size
                n = np.int64(min(qh_count[j], fam_ref_hog_counts[hog_offsets[j]]))
                fam_hog_scores[hog_offsets[j]] = compute_log_poisson_pvalue(
                    n, fam_hog_cumcounts[hog_offsets[j]], n * kmer_bernoulli[j])
            else:
                fam_hog_scores[hog_offsets[j]] = 0.0

        # store bestpath
        store_bestpath(hog_offsets, parent_offsets, fam_bestpath, fam_hog_scores, pv_score=True)

    return fam_hog_scores, fam_bestpath


## non-parametric scoring schemes
@numba.njit
def permute_seq_buff(seq_buff, w_size):
    '''
    shuffle windows of sequence buffer and shuffle within windows
    any odd window is also shuffled within the permuted sequence buffer
    '''
    seq_perm = np.zeros_like(seq_buff)
    
    # initate window indexes of the permuted sequence
    perm_w_idx = np.arange(0, seq_perm.size, w_size)
    
    # in case of odd window, insert it randomly
    odd_w_size = seq_perm.size % w_size
    
    if odd_w_size > 0:
        
        # pick the index of the odd window 
        odd_wi = np.random.choice(perm_w_idx) 
        
        # shift offset of following window indexes by the difference between window size and odd_w_size
        perm_w_idx[np.argwhere(odd_wi == perm_w_idx)[0][0] + 1 :] -= (w_size - odd_w_size)
        
        # add the odd window and shuffle it
        seq_perm[odd_wi: odd_wi + odd_w_size] = seq_buff[-odd_w_size:]
        np.random.shuffle(seq_perm[odd_wi: odd_wi + odd_w_size])
    
    # shuffle the windows indexes of permuted sequence
    np.random.shuffle(perm_w_idx)
    
    # get window indexes from the original sequence
    seq_w_idx = np.arange(0, seq_buff.size - odd_w_size, w_size)
    
    # traverse windows of permuted sequence
    i = 0
    for perm_wi in perm_w_idx:

        # skip odd window
        if odd_w_size > 0 and perm_wi == odd_wi:
            continue

        # add window and shuffle it
        seq_wi = seq_w_idx[i]
        seq_perm[perm_wi: perm_wi + w_size] = seq_buff[seq_wi: seq_wi + w_size]
        np.random.shuffle(seq_perm[perm_wi: perm_wi + w_size])

        i += 1
    
    return seq_perm

@numba.njit
def search_one_seq_perms(
    seq_buff, n_kmers, x_char, x_flag, perm_nr, top_m_fams, max_hog_nr, w_size, 
    trans, table_idx, table_buff, k, alphabet_n, DIGITS_AA_LOOKUP, 
    fam_tab, hog_tab, level_arr, top_fam, cum_mode):

    top_fam_perm_counts = np.zeros((perm_nr, top_m_fams), dtype=np.uint16)
    top_fam_hog_perm_counts = np.zeros((perm_nr, top_m_fams, max_hog_nr), dtype=np.uint16)

    for p in numba.prange(perm_nr):

        # TO DO: skip computation of k-mer locations
        r1, p1 = parse_seq(
            permute_seq_buff(seq_buff, w_size), DIGITS_AA_LOOKUP, n_kmers, k, trans, x_char, x_flag)

        # skip if one k-mer with X
        if len(r1) > 1:
            pass
        elif r1[0] == x_flag:
            continue

        # TO DO: skip computation of k-mer locations
        perm_hog_counts, perm_fam_counts, perm_query_occ, perm_hog_occurs, fam_lowloc, fam_highloc = search_seq_kmers(
            r1, p1, hog_tab, fam_tab, x_flag, table_idx, table_buff)

        # cumulate family and store cumulated HOG counts
        for r in numba.prange(top_m_fams):
            fam_ent = fam_tab[top_fam[r]]
            fam_hog_off = fam_ent['HOGoff']
            fam_hog_nr = fam_ent['HOGnum']
            
            # select perm_hog_counts for the family
            fam_perm_hog_cumcounts = perm_hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr]
            
            # cumulate HOG counts
            if cum_mode == 'max':
                fam_hog2parent = get_fam_hog2parent(fam_ent, hog_tab)
                fam_level_offsets = get_fam_level_offsets(fam_ent, level_arr)
                
                cumulate_counts_1fam(fam_perm_hog_cumcounts, fam_level_offsets, fam_hog2parent, _sum, _max)
            else:
                cumulate_counts_1fam(fam_perm_hog_cumcounts, fam_level_offsets, fam_hog2parent, _sum, _sum)
            
            # use root-HOG count as fam count
            top_fam_perm_counts[p, r] = fam_perm_hog_cumcounts[0]
            
            # store HOG counts too
            top_fam_hog_perm_counts[p, r, :fam_hog_nr] = fam_perm_hog_cumcounts

    return top_fam_perm_counts, top_fam_hog_perm_counts

@numba.njit
def norm_fam_nonparametric(fam_counts, fam_perm_counts, query_counts):
    '''
    remove the mean non-parametric counts from permuted queries
    '''
    fam_scores = np.zeros(fam_counts.size, np.float64)
    for i in numba.prange(fam_counts.size):
        fc = fam_counts[i]
        # remove sample mean --> fitted poisson pdf mean 
        mean_fpc = np.mean(fam_perm_counts[:, i])
        if mean_fpc >= fc:
            fam_scores[i] = 0.0
        else:
            fam_scores[i] = (fc - mean_fpc) / query_counts
    return fam_scores

def poisson_mle(perm_counts):
    '''
    MLE of lambda for Poisson is the sample mean
    in absence of any permuted count, return almost 0
    '''
    s = np.sum(perm_counts)
    return (s / perm_counts.size) if s > 0 else (0.1 / perm_counts.size) 

# various distribution to fit
def compute_nonparametric_poinorm_pvalue(counts, query_counts, ref_counts, perm_counts):
    '''
    adaptively fit Poisson or normal distribution to permuted counts
    and choose theoretical standard deviation if less than two x values
    '''
    n = np.int64(min(query_counts, ref_counts))

    # requires at least once shared k-mer and n > 0
    if counts > 0 and n > 0:

        # sample mean
        lamb = poisson_mle(perm_counts)

        # derive p from sample mean (i.e. lambda = np)
        p = np.float64(min(lamb / n, 1))

        # criteria from Decker and Fitzgibbon (1991) 
        if (n * 0.31 * p) < 0.47:
            return compute_log_poisson_pvalue(n, counts, lamb)

        else:
            # use theoretical sd if sum of permuted count equal 0 or less than two different count values (e.g. 3, 3 --> sd of 0...)
            # dubious?
            if np.sum(perm_counts) == 0 or np.unique(perm_counts).size == 1:
                mean = lamb
                sd = sqrt(lamb * (1-p))
            else:
                params = stats.norm.fit(perm_counts)
                mean = params[-2]
                sd = params[-1]

            return compute_log_normal_pvalue(counts, mean, sd)
    else:
        return 0.0

def compute_nonparametric_poisson_pvalue(counts, query_counts, ref_counts, perm_counts):
    n = np.int64(min(query_counts, ref_counts))

    # requires at least once shared k-mer and n > 0
    if counts > 0 and n > 0:

        # sample mean
        lamb = poisson_mle(perm_counts)

        return compute_log_poisson_pvalue(n, counts, lamb)

    else:
        return 0.0

def compute_nonparametric_norm_pvalue(counts, query_counts, ref_counts, perm_counts):
    n = np.int64(min(query_counts, ref_counts))

    # requires at least once shared k-mer and n > 0
    if counts > 0 and n > 0:

        # sample mean
        lamb = poisson_mle(perm_counts)

        # derive p from sample mean (i.e. lambda = np)
        p = np.float64(min(lamb / n, 1))

        # use theoretical sd if sum of permuted count equal 0 or less than two different count values (e.g. 3, 3 --> sd of 0...)
        # dubious?
        if np.sum(perm_counts) == 0 or np.unique(perm_counts).size == 1:
            mean = lamb
            sd = sqrt(lamb * (1-p))
        else:
            params = stats.norm.fit(perm_counts)
            mean = params[-2]
            sd = params[-1]

        return compute_log_normal_pvalue(counts, mean, sd)

    else:
        return 0.0

def compute_nonparametric_gamma_pvalue(counts, query_counts, ref_counts, perm_counts):
    
    n = np.int64(min(query_counts, ref_counts))

    # requires at least once shared k-mer and n > 0
    if counts > 0 and n > 0:

        # need at least 2 datapoints
        if np.sum(perm_counts) > 0 and np.unique(perm_counts).size > 1:

            return compute_cdist_pvalue(stats.gamma, perm_counts, counts)

        # else fall back to a poisson or normal 
        else:
            return compute_nonparametric_poinorm_pvalue(counts, query_counts, ref_counts, perm_counts)
    
    else:
        return 0.0

def compute_nonparametric_pvalue(counts, query_counts, ref_counts, perm_counts, dist):

    if dist == 'poisson':
        return compute_nonparametric_poisson_pvalue(counts, query_counts, ref_counts, perm_counts)

    elif dist == 'normal':
        return compute_nonparametric_norm_pvalue(counts, query_counts, ref_counts, perm_counts)

    elif dist == 'poinorm':
        return compute_nonparametric_poinorm_pvalue(counts, query_counts, ref_counts, perm_counts)

    elif dist == 'gamma':
        return compute_nonparametric_gamma_pvalue(counts, query_counts, ref_counts, perm_counts)

def compute_fam_nonparametric_pvalue(fam_counts, query_counts, ref_fam_counts, fam_perm_counts, dist):
    
    fam_scores = np.zeros(fam_counts.size, np.float64)
    for i in range(fam_counts.size): 
        fc = fam_counts[i]
        fam_scores[i] = compute_nonparametric_pvalue(fc, query_counts, ref_fam_counts[i], fam_perm_counts[:, i], dist)
        
    return fam_scores

@numba.njit
def norm_hog_nonparametric(
    fam_hog_cumcounts, query_counts, fam_level_offsets, hog2parent, fam_hog_counts, fam_hog_perm_counts):
    '''
    substract HOG mean counts
    '''
    fam_hog_scores = np.zeros(fam_hog_cumcounts.shape, dtype=np.float64)
    fam_bestpath = np.full(fam_hog_cumcounts.shape, False)
    revcum_counts = np.full(fam_hog_cumcounts.shape, query_counts, dtype=np.uint16)

    # compute and store root-HOG score
    hc = fam_hog_cumcounts[0]
    mean_hpc = np.mean(fam_hog_perm_counts[:, 0])
    if hc > mean_hpc:
        fam_hog_scores[0] = np.float64(hc - mean_hpc) / query_counts
    else:
        fam_hog_scores[0] = 0.0

    # set the root-HOG as best path
    fam_bestpath[0] = True
    
    # loop through hog levels
    for i in range(1, fam_level_offsets.size - 2):
        x = fam_level_offsets[i : i + 2]
        hog_offsets = np.arange(x[0], x[1])

        # grab parents
        parent_offsets = hog2parent[hog_offsets]

        # update query revcumcount, basically substracting parent counts from query count
        qh_count = revcum_counts[parent_offsets] - fam_hog_counts[parent_offsets]
        revcum_counts[hog_offsets] = qh_count

        # compute and store score
        for j in range(hog_offsets.size):
            ho = hog_offsets[j]
            hc = fam_hog_cumcounts[ho]
            mean_hpc = np.mean(fam_hog_perm_counts[:, ho]) 
            if hc > mean_hpc:
                fam_hog_scores[ho] = np.float64(hc - mean_hpc) / qh_count[j]
            else:
                fam_hog_scores[ho] = 0.0

        # store bestpath
        store_bestpath(hog_offsets, parent_offsets, fam_bestpath, fam_hog_scores, pv_score=False)

    return fam_hog_scores, fam_bestpath

def compute_hog_nonparametric_pvalue(
    fam_hog_cumcounts, query_counts, fam_ref_hog_counts, fam_hog_perm_counts, 
    fam_level_offsets, hog2parent, fam_hog_counts, dist):
    
    fam_hog_scores = np.zeros(fam_hog_cumcounts.shape, dtype=np.float64)
    fam_bestpath = np.full(fam_hog_cumcounts.shape, False)
    revcum_counts = np.full(fam_hog_cumcounts.shape, query_counts, dtype=np.uint16)

    # compute and store root-HOG score
    hc = fam_hog_cumcounts[0]
    fam_hog_scores[0] = compute_nonparametric_pvalue(hc, query_counts, fam_ref_hog_counts[0], fam_hog_perm_counts[:, 0], dist)

    # set the root-HOG as best path
    fam_bestpath[0] = True
    
    # loop through hog levels
    for i in range(1, fam_level_offsets.size - 2):
        x = fam_level_offsets[i : i + 2]
        hog_offsets = np.arange(x[0], x[1])

        # grab parents
        parent_offsets = hog2parent[hog_offsets]

        # update query revcumcount, basically substracting parent counts from query count
        qh_count = revcum_counts[parent_offsets] - fam_hog_counts[parent_offsets]
        revcum_counts[hog_offsets] = qh_count

        # compute and store score
        for j in range(hog_offsets.size):
            ho = hog_offsets[j]
            hc = fam_hog_cumcounts[ho]
            fam_hog_scores[ho] = compute_nonparametric_pvalue(hc, qh_count[j], fam_ref_hog_counts[ho], fam_hog_perm_counts[:, ho], dist)

        # store bestpath
        store_bestpath(hog_offsets, parent_offsets, fam_bestpath, fam_hog_scores, pv_score=True)

    return fam_hog_scores, fam_bestpath

## sequence overlap
@numba.njit
def compute_overlap(fam_highloc, fam_lowloc, k, query_len):
    overlap = np.zeros(fam_highloc.shape, dtype=np.float64)
    for i in numba.prange(fam_highloc.size):
        if fam_highloc[i] == -1:
            continue
        else:
            overlap[i] = (fam_highloc[i] - fam_lowloc[i] + k) / query_len
    return overlap

## function to place queries and output results
@numba.njit
def _place_queries(
    query_offsets, q2fam_off, q2fam_score, q2fam_overlap, fam_tab, q2hog_bestpath, q2hog_scores, overlap, fst, sst,
    true_tax_off, tax_tab, hog_tab, chog_buff):
    '''
    For each query:
        Skip when low sequence overlap (overlap)
        Traverse the best scoring root-to-leaf HOG path to:
         1. keep track of the most specific HOG above the subfamily-score threshold (sst)
         2. keep track of the highest HOG score to use as family score
         3. option to stop placement at true taxon
        Skip if family score below family-score threshold or all subfamily-score below subfamily-score threshold
    Skipped queries are returned with a -1
    '''
    q2hog_off = np.full(q2fam_off.size, -1, dtype=np.int64)
    q2hog_score = np.full(q2fam_off.size, -1, dtype=np.float64)
    q2max_hog_score = np.full(q2fam_off.size, -1, dtype=np.float64)
    
    if true_tax_off is not None:
        true_tax_lineage = get_root_leaf_offsets(true_tax_off, tax_tab['ParentOff'])[::-1]
    
    for i in numba.prange(query_offsets.size):
        q = query_offsets[i]
        
        # if below overlap threshold, skip
        o = q2fam_overlap[q]
        if (o < overlap):
            continue       

        # find best scoring subfamily
        fam_off = q2fam_off[q]
        root_hog_off = fam_tab[fam_off]['HOGoff']
        fam_hog_nr = fam_tab[fam_off]['HOGnum']
        fam_bestpath = q2hog_bestpath[q, :fam_hog_nr]
        fam_hog_scores = q2hog_scores[q, :fam_hog_nr]
        best_j = None
        best_s = 0
        for j in np.argwhere(fam_bestpath).flatten():
            
            # update best subfamily (above threshold)
            ss = fam_hog_scores[j]  
            if ss >= sst:
                best_j = j
            
            # update best score
            if ss > best_s:
                best_s = ss
            
            # stop if subfamily implies the true taxon
            if true_tax_off is not None:
                hog_off = np.int(root_hog_off + j)
                if is_taxon_implied(true_tax_lineage, hog_off, hog_tab, chog_buff):
                    break
                #hog_taxa = get_hog_taxon_levels(hog_off, hog_tab, hog_taxa_buff)
                #if np.argwhere(hog_taxa == true_tax_off).size == 1:
                #    break
                        
        # skip if no subfamily-score > subfamily-score threshold or if best subfamily-score < family-score threshold
        if (best_j is None) or (best_s < fst):
            continue
        
        q2hog_off[i] = np.int(best_j + root_hog_off)
        q2hog_score[i] = ss
        q2max_hog_score[i] = best_s

    return q2hog_off, q2hog_score, q2max_hog_score

#@numba.njit
#def get_closest_taxa_from_ref(q2hog_off, ref_taxoff, tax_tab, hog_tab, hog_taxa_buff):
#    '''
#    Based on the predicted HOG root and descendant taxa, we find the closest taxon from reference taxon.
#     - if root-taxon is one of its descendants, we report the HOG root-taxon (older child) (should be root-HOGs when using such stopping criterion)
#     - if root-taxon is one of its ancestors, we report the first ancestor that is defined within the HOG (based on real hog2taxa)
#       (basically the speciation before the duplication or loss delimiting the HOG from the reference taxon)
#     - otherwise, we simply report the HOG root-taxon (still the closer from the reference taxon since on wrong lineage)
#     - na if not placed    
#    '''
#    q2closest_taxon = np.zeros(q2hog_off.size, dtype=np.uint64)
#    ref_lineage = get_root_leaf_offsets(ref_taxoff, tax_tab['ParentOff'])[::-1][1:]
#
#    for i, hog_off in enumerate(q2hog_off):
#
#        # not placed
#        if hog_off == -1:
#            q2closest_taxon[i] == -1
#            continue
#
#        # get the HOG taxa and the HOG root-taxon
#        hog_taxa = get_hog_taxon_levels(hog_off, hog_tab, hog_taxa_buff)
#        root_tax_off = hog_tab['TaxOff'][hog_off]    
#
#        # reference taxon in HOG taxa (at level)
#        if np.argwhere(hog_taxa == ref_taxoff).size == 1:
#            q2closest_taxon[i] = ref_taxoff
#
#        # root-taxon in taxon lineage (more general)
#        elif np.argwhere(ref_lineage == root_tax_off).size == 1:
#
#            # get the closer ancestral taxon in HOG
#            j = 0
#            while np.argwhere(hog_taxa == ref_lineage[j]).size == 0:
#                j += 1
#            q2closest_taxon[i] = ref_lineage[j]
#
#        # root-taxon either in child taxa (more specific) or in a different clade
#        else:
#            q2closest_taxon[i] = hog_tab['TaxOff'][hog_off]
#    
#    return q2closest_taxon
@numba.njit
def get_closest_taxa_from_ref(q2hog_off, ref_taxoff, tax_tab, hog_tab, chog_buff):
    '''
    Based on the predicted HOG, we find the closest implied taxon from the reference taxon.
     - if the reference taxon is implied in the HOG (descendant of root-taxon and ancestor of child-HOG taxa), we report the reference taxon (at level).
     - if the root-taxon is one of its ancestors, we report the first ancestor that is defined within the HOG child-HOG taxa (more general).
       (basically the speciation before the duplication delimiting the HOG from the reference taxon)
     - otherwise, we simply report the HOG root-taxon ('more specific' or 'different lineage').
     - na if not placed    
    '''
    q2closest_taxon = np.zeros(q2hog_off.size, dtype=np.uint64)
    true_tax_lineage = get_root_leaf_offsets(ref_taxoff, tax_tab['ParentOff'])[::-1]

    for i, hog_off in enumerate(q2hog_off):

        # not placed
        if hog_off == -1:
            q2closest_taxon[i] == -1
            continue 
        
        # reference taxon implied in HOG (at level)
        if is_taxon_implied(true_tax_lineage, hog_off, hog_tab, chog_buff):
            q2closest_taxon[i] = ref_taxoff

        # root-taxon in ancestors (more general)
        elif np.argwhere(true_tax_lineage[1:] == hog_tab['TaxOff'][hog_off]).size == 1:
            
            # get the closest taxon from the reference taxon among the HOG children
            child_hog_taxa = np.unique(hog_tab['TaxOff'][get_children(hog_off, hog_tab, chog_buff)])
            
            # get the closer ancestral taxon in HOG
            j = 0
            while np.argwhere(child_hog_taxa == true_tax_lineage[j]).size == 0:
                j += 1
            
            # add 1 because we actually take the parent taxa of child_hog_taxa
            q2closest_taxon[i] = true_tax_lineage[j + 1]

        # root-taxon either in child taxa (more specific) or in a different clade
        else:
            q2closest_taxon[i] = hog_tab['TaxOff'][hog_off]
    
    return q2closest_taxon


class MergeSearch(object):
    def __init__(self, ki, nthreads=None, low_mem=False, include_extant_genes=False):
    	assert ki.db.mode == "r", "Database must be opened in read mode."

    	# load ki and db
    	self.db = ki.db
    	self.ki = ki

    	# performance features
    	self.nthreads = nthreads if nthreads is not None else os.cpu_count()
    	self.low_mem = low_mem
    	self.include_extant_genes = include_extant_genes
    	self.query_sp = None

    @cached_property
    def trans(self):
        return get_transform(self.ki.k, self.ki.alphabet.DIGITS_AA)

    @cached_property
    def table_idx(self):
        x = self.ki._table_idx
        return x[:] if not self.low_mem else x

    @cached_property
    def table_buff(self):
        x = self.ki._table_buff
        return x[:] if not self.low_mem else x

    @cached_property
    def fam_tab(self):
        return self.db._fam_tab[:]

    @cached_property
    def hog_tab(self):
        return self.db._hog_tab[:]

    @cached_property
    def tax_tab(self):
        return self.db._tax_tab[:]

    @cached_property
    def level_arr(self):
        return self.db._level_arr[:]	    

    @cached_property
    def max_hog_nr(self):
        return np.int64(np.max(self.db._fam_tab.col('HOGnum')))

    # k-mer counts of reference HOGs and family
    @lazy_property
    def ref_hog_counts_sum(self):
    	return cumulate_counts_nfams(
    		self.ki._hog_count[:], self.db._fam_tab[:], self.db._level_arr[:], self.db._hog_tab.col('ParentOff'), 
    		cumulate_counts_1fam, _sum, _sum)            

    @lazy_property
    def ref_fam_counts_sum(self):
    	return self.ref_hog_counts_sum[self.db._fam_tab.col('HOGoff')]

    @lazy_property
    def ref_hog_counts_max(self):
    	return cumulate_counts_nfams(
    		self.ki._hog_count[:], self.db._fam_tab[:], self.db._level_arr[:], self.db._hog_tab.col('ParentOff'), 
    		cumulate_counts_1fam, _sum, _max) 

    @lazy_property
    def ref_fam_counts_max(self):
    	return self.ref_hog_counts_max[self.db._fam_tab.col('HOGoff')]

    def merge_search(self, seqs=None, ids=None, fasta_file=None, score='querysize_hogsize_kmerfreq', cum_mode='max', top_m_fams=100, 
        top_n_fams=1, perm_nr=1, w_size=6, dist='poisson', fam_filter=np.array([], dtype=np.int64)):
        
        # load query sequences
        t1 = time()
        if seqs:
            sbuff = SequenceBuffer(seqs=seqs, ids=ids)
        elif fasta_file:
            sbuff = SequenceBuffer(fasta_file=fasta_file)

        t2 = time()
        print('{} second to load query sequences'.format(int(t2 - t1)))

        # for parametric scores that normalize for family and subfamily size, compute cumulated counts of reference HOGs 
        # actually these are also required to compute have the n parameter in the binomial as min(|Q|, |H|))
        if score in {'querysize_hogsize', 'querysize_hogsize_kmerfreq', 'mash_pvalue', 'kmerfreq_pvalue', 'nonparam_pvalue'}:
            if cum_mode == 'sum':
            	ref_fam_counts = self.ref_fam_counts_sum
            	ref_hog_counts = self.ref_hog_counts_sum
            elif cum_mode == 'max':
            	ref_fam_counts = self.ref_fam_counts_max
            	ref_hog_counts = self.ref_hog_counts_max
        else:
            ref_fam_counts = np.array([], dtype=np.int64)
            ref_hog_counts = np.array([], dtype=np.int64)
        
        t3 = time()
        print('{} second to cumulate counts of reference HOGs'.format(int(t3 - t2)))

        # load OMAmer database and table in memory       
        trans = self.trans
        table_idx = self.table_idx
        table_buff = self.table_buff
        fam_tab = self.fam_tab
        hog_tab = self.hog_tab
        level_arr = self.level_arr
        max_hog_nr = self.max_hog_nr

        t4 = time()
        print('{} second to load the k-mer table'.format(int(t4 - t3)))
        
        # pick lookup function
        if score == 'querysize_hogsize_kmerfreq':
            lookup_fun = self._lookup_querysize_hogsize_kmerfreq

        elif score == 'nonparam_naive':
            lookup_fun = self._lookup_nonparam_naive

        elif score in {'mash_pvalue', 'kmerfreq_pvalue', 'nonparam_pvalue'}:
            # disable numba vectorisation for scipy
            self.low_mem = True
            lookup_fun = self._lookup_pvalue
        else:
            lookup_fun = self._lookup_naive

        # actual OMAmer search
        queryFam_ranked, queryFam_scores, queryFam_overlaps, queryRankHog_bestpath, queryRankHog_scores = lookup_fun(
            sbuff.buff,
            sbuff.idx,
            trans,
            table_idx,
            table_buff,
            self.ki.k,
            self.ki.alphabet.n,
            self.ki.alphabet.DIGITS_AA_LOOKUP,
            fam_tab,
            hog_tab,
            level_arr,
            max_hog_nr,
            fam_filter,
            top_n_fams = top_n_fams,
            top_m_fams = top_m_fams,
            cum_mode = cum_mode, 
            score = score,
            ref_fam_counts = ref_fam_counts,
            ref_hog_counts = ref_hog_counts,
            perm_nr = perm_nr,
            w_size = w_size,
            dist = dist)
        
        t5 = time()
        ts =  t5 - t4
        print('{} second for the actual search (~ {} query/second)'.format(int(ts), int(sbuff.prot_nr / ts)))

        self._queryFam_ranked = queryFam_ranked
        self._queryFam_scores = queryFam_scores
        self._queryFam_overlaps = queryFam_overlaps
        self._queryRankHog_bestpath = queryRankHog_bestpath
        self._queryRankHog_scores = queryRankHog_scores

        # store ids and lengths of sbuff
        self._query_ids = sbuff.ids.flatten()
        self._query_lengths = sbuff.idx[1:] - sbuff.idx[:-1]


    def store_results(self, filename):
        compr = tables.Filters(complevel=6, complib="blosc", fletcher32=True)
        h5 = tables.open_file(filename, 'w', filters=compr)
        h5.create_carray('/', 'query_ids', obj=self._query_ids, filters=compr)
        h5.create_carray('/', 'queryFam_ranked', obj=self._queryFam_ranked, filters=compr)
        h5.create_carray('/', 'queryFam_scores', obj=self._queryFam_scores, filters=compr)
        h5.create_carray('/', 'queryFam_overlaps', obj=self._queryFam_overlaps, filters=compr)
        h5.create_carray('/', 'queryRankHog_bestpath', obj=self._queryRankHog_bestpath, filters=compr)
        h5.create_carray('/', 'queryRankHog_scores', obj=self._queryRankHog_scores, filters=compr)
        h5.close()

    def load_results(self, filename):
        compr = tables.Filters(complevel=6, complib="blosc", fletcher32=True)
        h5 = tables.open_file(filename, 'r', filters=compr)
        self._query_ids = h5.root.query_ids[:]
        self._queryFam_ranked = h5.root.queryFam_ranked[:]
        self._queryFam_scores = h5.root.queryFam_scores[:]
        self._queryFam_overlaps = h5.root.queryFam_overlaps[:]
        self._queryRankHog_bestpath = h5.root.queryRankHog_bestpath[:]
        self._queryRankHog_scores = h5.root.queryRankHog_scores[:]
        h5.close()

    def place_queries(
        self, query_offsets, overlap, fst, sst, ref_taxoff=None):
        '''
        For each query:
            Skip when low sequence overlap (overlap)
            Traverse the best scoring root-to-leaf HOG path to:
             1. keep track of the most specific HOG above the subfamily-score threshold (sst)
             2. keep track of the highest HOG score to use as family score
             3. option to stop placement at true taxon
            Skip if family score below family-score threshold or all subfamily-score below subfamily-score threshold
        Skipped queries are returned with a -1
        '''
        return _place_queries(
            query_offsets, self._queryFam_ranked[:, 0], self._queryFam_scores[:, 0], self._queryFam_overlaps[:, 0], self.fam_tab, 
            self._queryRankHog_bestpath[0], self._queryRankHog_scores[0], overlap, fst, sst, ref_taxoff, self.tax_tab, self.hog_tab, self.db._chog_arr[:])

    def output_results(self, overlap, fst, sst, ref_taxon):

        if ref_taxon:
            tax_tab = self.db._tax_tab[:]
            ref_taxoff = np.searchsorted(tax_tab['ID'], ref_taxon.encode('ascii'))
        else:
            ref_taxoff = None

        # place queries
        query_offsets = np.arange(self._query_ids.size, dtype=np.int64)
        q2hog_off, q2hog_score, q2max_hog_score = self.place_queries(
            query_offsets, overlap, fst, sst, ref_taxoff)
        
        c = ['qseqid', 'hogid', 'overlap', 'family-score', 'subfamily-score', 'qseqlen', 'subfamily-medianseqlen']
        r = [[x.decode('ascii') if isinstance(x, bytes) else x for x in self._query_ids],
            map(lambda x: self.hog_tab['OmaID'][x].decode('ascii') if x != -1 else 'na', q2hog_off),
            [x if q2hog_off[i] != -1 else 'na' for i, x in enumerate(self._queryFam_overlaps.flatten())], 
            map(lambda x: x if x != -1 else 'na', q2max_hog_score),
            map(lambda x: x if x != -1 else 'na', q2hog_score),
            self._query_lengths,
            map(lambda x: self.hog_tab['MedianSeqLen'][x] if x != -1 else 'na', q2hog_off)]
        
        ## temporary to check PANTHER placement
        #c = ['qseqid', 'hogid', 'overlap', 'family-score', 'subfamily-score', 'qseqlen', 'subfamily-medianseqlen']
        #r = [[x.decode('ascii') if isinstance(x, bytes) else x for x in self._query_ids],
        #    map(lambda x: self.hog_tab['OmaID'][x].decode('ascii') if x != -1 else 'na', q2hog_off),
        #    [x if q2hog_off[i] != -1 else 'na' for i, x in enumerate(self._queryFam_overlaps.flatten())], 
        #    map(lambda x: x if x != -1 else 'na', q2max_hog_score),
        #    map(lambda x: x if x != -1 else 'na', q2hog_score),
        #    self._query_lengths,
        #    self._query_lengths]

        # compute taxonomic congruences 
        if ref_taxon:
            q2closest_taxon = get_closest_taxa_from_ref(
                q2hog_off, ref_taxoff, tax_tab, self.hog_tab, self.db._chog_arr[:])
            c.insert(2, 'closetax')
            r.insert(2, map(lambda x: tax_tab['ID'][x].decode('ascii') if x != -1 else 'na', q2closest_taxon)) 

        # add member proteins as csv
        if self.include_extant_genes:
            chog_buff = self.db._chog_arr[:]
            cprot_buff = self.db._cprot_arr[:]
            prot_tab = self.db._prot_tab[:]

            c.append('subfamily-geneset')
            r.append([','.join(map(lambda x: x.decode('ascii'), prot_tab['ID'][get_hog_member_prots(
                hog_off, self.hog_tab, chog_buff, cprot_buff)])) for hog_off in q2hog_off])

        return pd.DataFrame(zip(*r), columns=c)

    @lazy_property
    def _lookup_querysize_hogsize_kmerfreq(self):
        def func(
            seqs,
            seqs_idx,
            trans,
            table_idx,
            table_buff,
            k,
            alphabet_n,
            DIGITS_AA_LOOKUP,
            fam_tab,
            hog_tab,
            level_arr,
            max_hog_nr,
            fam_filter,
            top_n_fams, 
            top_m_fams,
            cum_mode,
            score,
            ref_fam_counts,
            ref_hog_counts,
            perm_nr,
            w_size,
            dist
        ):   
            '''
            top_n_fams: number of family for which HOG scores are computed
            top_m_fams: number of family for which family scores are computed before resorting
            '''
            # highest scoring families per query sorted by counts 
            queryFam_ranked = np.zeros((len(seqs_idx) - 1, top_n_fams), dtype=np.uint32)
            # corresponding k-mer counts
            queryFam_scores = np.zeros((len(seqs_idx) - 1, top_n_fams), dtype=np.float64)
            # scores and bestpath mask for HOGs of top_n_fam
            queryRankHog_scores = np.zeros((top_n_fams, len(seqs_idx) - 1, max_hog_nr), dtype=np.float64)
            queryRankHog_bestpath = np.zeros((top_n_fams, len(seqs_idx) - 1, max_hog_nr), dtype=np.bool8)    
            # OPTION: store only HOGs on the bestpath

            # query sequence overlaps with families
            queryFam_overlaps = np.zeros((len(seqs_idx) - 1, top_n_fams), dtype=np.float64)

            # to ignore k-mers with X (88 == b'X')
            x_char = DIGITS_AA_LOOKUP[88]
            # get a flag for k-mer with any X (= last k-mer + 1)
            x_flag = table_idx.size - 1

            # iterate of sequences
            for zz in numba.prange(len(seqs_idx) - 1):

                ## get the query sequence
                s = seqs[seqs_idx[zz] : np.int(seqs_idx[zz + 1] - 1)]
                query_len = s.shape[0]
                n_kmers = query_len - (k - 1)

                # double check we don't have short peptides (of len < k)
                # note: written in this order to provide loop-optimisation hint (?)
                if n_kmers > 0:
                    pass
                else:
                    continue
                
                ## get the sequence unique k-mers and non-ambiguous locations
                r1, p1 = parse_seq(
                    s, DIGITS_AA_LOOKUP, n_kmers, k, trans, x_char, x_flag)

                # skip if one k-mer with X
                if len(r1) > 1:
                    pass
                elif r1[0] == x_flag:
                    continue

                ## search sequence
                hog_counts, fam_counts, query_occ, hog_occurs, fam_lowloc, fam_highloc = search_seq_kmers(
                    r1, p1, hog_tab, fam_tab, x_flag, table_idx, table_buff)

                ## get the raw top m families from summed k-mer counts
                top_fam, top_fam_counts = get_top_m_fams(
                    fam_counts, top_m_fams, cum_mode, fam_tab, hog_counts, hog_tab, level_arr, fam_filter)
                
                ## normalize family counts
                top_fam_scores = norm_fam_querysize_hogsize_kmerfreq(
                    top_fam_counts, query_occ, r1.size, table_buff.size, ref_fam_counts[top_fam])     

                # resort by score
                idx = (-top_fam_scores).argsort()
                top_fam = top_fam[idx]
                top_fam_scores = top_fam_scores[idx]

                # store them with corresponding scores
                queryFam_ranked[zz, :top_n_fams] = top_fam[:top_n_fams]
                queryFam_scores[zz, :top_n_fams] = top_fam_scores[:top_n_fams]
                queryFam_overlaps[zz, :top_n_fams] = compute_overlap(fam_highloc[top_fam[:top_n_fams]], fam_lowloc[top_fam[:top_n_fams]], k, query_len)

                ### Compute HOG scores and bestpath for top n families
                # iterate over top n families
                for fam_rank in numba.prange(top_n_fams):
                    fam_off = top_fam[fam_rank]
                    fam_ent = fam_tab[fam_off]
                    fam_hog_off = np.int64(fam_ent['HOGoff'])  # slicing with int (vs. uint) enabled parallel=True
                    fam_hog_nr = np.int64(fam_ent['HOGnum'])

                    # compute the cumulated HOG counts for that family
                    fam_hog_counts = hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr].copy()

                    fam_hog2parent = get_fam_hog2parent(fam_ent, hog_tab)
                    fam_level_offsets = get_fam_level_offsets(fam_ent, level_arr)

                    fam_hog_cumcounts = fam_hog_counts.copy()

                    if cum_mode == 'max':
                        cumulate_counts_1fam(
                            fam_hog_cumcounts, fam_level_offsets, fam_hog2parent, _sum, _max)

                    elif cum_mode == 'sum':
                        cumulate_counts_1fam(
                            fam_hog_cumcounts, fam_level_offsets, fam_hog2parent, _sum, _sum)

                    # querysize_kmerfreq
                    fam_hog_scores, fam_bestpath = norm_hog_querysize_hogsize_kmerfreq(
                        fam_hog_cumcounts, r1.size, query_occ, table_buff.size, ref_hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr],
                        fam_level_offsets, fam_hog2parent, fam_hog_counts, hog_occurs[fam_hog_off:fam_hog_off + fam_hog_nr])  

                    queryRankHog_bestpath[fam_rank, zz, :fam_bestpath.size] = fam_bestpath
                    queryRankHog_scores[fam_rank, zz, :fam_hog_scores.size] = fam_hog_scores

            return queryFam_ranked, queryFam_scores, queryFam_overlaps, queryRankHog_bestpath, queryRankHog_scores

        if not self.low_mem:
            # Set nthreads, note: this only works before numba called first time!
            numba.set_num_threads(self.nthreads)
            return numba.jit(func, parallel=(True if self.nthreads > 1 else False), nopython=True, nogil=True)
        else:
            return func
    
    @lazy_property
    def _lookup_nonparam_naive(self):
        def func(
            seqs,
            seqs_idx,
            trans,
            table_idx,
            table_buff,
            k,
            alphabet_n,
            DIGITS_AA_LOOKUP,
            fam_tab,
            hog_tab,
            level_arr,
            max_hog_nr,
            fam_filter,
            top_n_fams, 
            top_m_fams,
            cum_mode,
            score,
            ref_fam_counts,
            ref_hog_counts,
            perm_nr,
            w_size,
            dist
        ):   
            '''
            top_n_fams: number of family for which HOG scores are computed
            top_m_fams: number of family for which family scores are computed before resorting
            '''
            # highest scoring families per query sorted by counts 
            queryFam_ranked = np.zeros((len(seqs_idx) - 1, top_n_fams), dtype=np.uint32)
            # corresponding k-mer counts
            queryFam_scores = np.zeros((len(seqs_idx) - 1, top_n_fams), dtype=np.float64)
            # scores and bestpath mask for HOGs of top_n_fam
            queryRankHog_scores = np.zeros((top_n_fams, len(seqs_idx) - 1, max_hog_nr), dtype=np.float64)
            queryRankHog_bestpath = np.zeros((top_n_fams, len(seqs_idx) - 1, max_hog_nr), dtype=np.bool8)    
            # OPTION: store only HOGs on the bestpath
            
            # query sequence overlaps with families
            queryFam_overlaps = np.zeros((len(seqs_idx) - 1, top_n_fams), dtype=np.float64)

            # to ignore k-mers with X (88 == b'X')
            x_char = DIGITS_AA_LOOKUP[88]
            # get a flag for k-mer with any X (= last k-mer + 1)
            x_flag = table_idx.size - 1

            # iterate of sequences
            for zz in numba.prange(len(seqs_idx) - 1):

                ## get the query sequence
                s = seqs[seqs_idx[zz] : np.int(seqs_idx[zz + 1] - 1)]
                query_len = s.shape[0]
                n_kmers = query_len - (k - 1)

                # double check we don't have short peptides (of len < k)
                # note: written in this order to provide loop-optimisation hint (?)
                if n_kmers > 0:
                    pass
                else:
                    continue

                ## get the sequence unique k-mers and non-ambiguous locations
                r1, p1 = parse_seq(
                    s, DIGITS_AA_LOOKUP, n_kmers, k, trans, x_char, x_flag)

                # skip if one k-mer with X
                if len(r1) > 1:
                    pass
                elif r1[0] == x_flag:
                    continue

                ## search sequence
                hog_counts, fam_counts, query_occ, hog_occurs, fam_lowloc, fam_highloc = search_seq_kmers(
                    r1, p1, hog_tab, fam_tab, x_flag, table_idx, table_buff)

                ## get the raw top m families from summed k-mer counts
                top_fam, top_fam_counts = get_top_m_fams(
                    fam_counts, top_m_fams, cum_mode, fam_tab, hog_counts, hog_tab, level_arr, fam_filter)
                
                ## search permuted sequences if non-parametric score
                top_fam_perm_counts, top_fam_hog_perm_counts = search_one_seq_perms(
                    s, n_kmers, x_char, x_flag, perm_nr, top_m_fams, max_hog_nr, w_size, 
                    trans, table_idx, table_buff, k, alphabet_n, DIGITS_AA_LOOKUP, 
                    fam_tab, hog_tab, level_arr, top_fam, cum_mode)
                
                ## normalize family counts
                top_fam_scores = norm_fam_nonparametric(
                    top_fam_counts, top_fam_perm_counts, r1.size)
                
                # resort by score
                idx = (-top_fam_scores).argsort()
                top_fam = top_fam[idx]
                top_fam_scores = top_fam_scores[idx]

                # store them with corresponding scores and overlaps
                queryFam_ranked[zz, :top_n_fams] = top_fam[:top_n_fams]
                queryFam_scores[zz, :top_n_fams] = top_fam_scores[:top_n_fams]
                queryFam_overlaps[zz, :top_n_fams] = compute_overlap(fam_highloc[top_fam[:top_n_fams]], fam_lowloc[top_fam[:top_n_fams]], k, query_len)

                ### Compute HOG scores and bestpath for top n families
                # iterate over top n families
                for fam_rank in numba.prange(top_n_fams):
                    fam_off = top_fam[fam_rank]
                    fam_ent = fam_tab[fam_off]
                    fam_hog_off = np.int64(fam_ent['HOGoff'])  # slicing with int (vs. uint) enabled parallel=True
                    fam_hog_nr = np.int64(fam_ent['HOGnum'])

                    # compute the cumulated HOG counts for that family
                    fam_hog_counts = hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr].copy()

                    fam_hog2parent = get_fam_hog2parent(fam_ent, hog_tab)
                    fam_level_offsets = get_fam_level_offsets(fam_ent, level_arr)

                    fam_hog_cumcounts = fam_hog_counts.copy()

                    if cum_mode == 'max':
                        cumulate_counts_1fam(
                            fam_hog_cumcounts, fam_level_offsets, fam_hog2parent, _sum, _max)

                    elif cum_mode == 'sum':
                        cumulate_counts_1fam(
                            fam_hog_cumcounts, fam_level_offsets, fam_hog2parent, _sum, _sum)

                    fam_hog_scores, fam_bestpath = norm_hog_nonparametric(
                        fam_hog_cumcounts, r1.size, fam_level_offsets, fam_hog2parent, fam_hog_counts, 
                        top_fam_hog_perm_counts[:, idx[fam_rank]])  # permuted HOG counts must be sorted by family-score too!

                    queryRankHog_bestpath[fam_rank, zz, :fam_bestpath.size] = fam_bestpath
                    queryRankHog_scores[fam_rank, zz, :fam_hog_scores.size] = fam_hog_scores

            return queryFam_ranked, queryFam_scores, queryFam_overlaps, queryRankHog_bestpath, queryRankHog_scores

        if not self.low_mem:
            # Set nthreads, note: this only works before numba called first time!
            numba.set_num_threads(self.nthreads)
            return numba.jit(func, parallel=(True if self.nthreads > 1 else False), nopython=True, nogil=True)
        else:
            return func

    @lazy_property
    def _lookup_naive(self):
        def func(
            seqs,
            seqs_idx,
            trans,
            table_idx,
            table_buff,
            k,
            alphabet_n,
            DIGITS_AA_LOOKUP,
            fam_tab,
            hog_tab,
            level_arr,
            max_hog_nr,
            fam_filter,
            top_n_fams, 
            top_m_fams,
            cum_mode,
            score,
            ref_fam_counts,
            ref_hog_counts,
            perm_nr,
            w_size,
            dist
        ):   
            '''
            top_n_fams: number of family for which HOG scores are computed
            top_m_fams: number of family for which family scores are computed before resorting
            '''
            # highest scoring families per query sorted by counts 
            queryFam_ranked = np.zeros((len(seqs_idx) - 1, top_n_fams), dtype=np.uint32)
            # corresponding k-mer counts
            queryFam_scores = np.zeros((len(seqs_idx) - 1, top_n_fams), dtype=np.float64)
            # scores and bestpath mask for HOGs of top_n_fam
            queryRankHog_scores = np.zeros((top_n_fams, len(seqs_idx) - 1, max_hog_nr), dtype=np.float64)
            queryRankHog_bestpath = np.zeros((top_n_fams, len(seqs_idx) - 1, max_hog_nr), dtype=np.bool8)    
            # OPTION: store only HOGs on the bestpath

            # query sequence overlaps with families
            queryFam_overlaps = np.zeros((len(seqs_idx) - 1, top_n_fams), dtype=np.float64)

            # to ignore k-mers with X (88 == b'X')
            x_char = DIGITS_AA_LOOKUP[88]
            # get a flag for k-mer with any X (= last k-mer + 1)
            x_flag = table_idx.size - 1

            # iterate of sequences
            for zz in numba.prange(len(seqs_idx) - 1):

                ## get the query sequence
                s = seqs[seqs_idx[zz] : np.int(seqs_idx[zz + 1] - 1)]
                query_len = s.shape[0]
                n_kmers = query_len - (k - 1)

                # double check we don't have short peptides (of len < k)
                # note: written in this order to provide loop-optimisation hint (?)
                if n_kmers > 0:
                    pass
                else:
                    continue

                ## get the sequence unique k-mers and non-ambiguous locations
                r1, p1 = parse_seq(
                    s, DIGITS_AA_LOOKUP, n_kmers, k, trans, x_char, x_flag)

                # skip if one k-mer with X
                if len(r1) > 1:
                    pass
                elif r1[0] == x_flag:
                    continue

                ## search sequence
                hog_counts, fam_counts, query_occ, hog_occurs, fam_lowloc, fam_highloc = search_seq_kmers(
                    r1, p1, hog_tab, fam_tab, x_flag, table_idx, table_buff)

                ## get the raw top m families from summed k-mer counts
                top_fam, top_fam_counts = get_top_m_fams(
                    fam_counts, top_m_fams, cum_mode, fam_tab, hog_counts, hog_tab, level_arr, fam_filter)
                
                ## search permuted sequences if non-parametric score
                if (score == 'nonparam_naive') or (score == 'nonparam_pvalue'):
                    top_fam_perm_counts, top_fam_hog_perm_counts = search_one_seq_perms(
                        s, n_kmers, x_char, x_flag, perm_nr, top_m_fams, max_hog_nr, w_size, 
                        trans, table_idx, table_buff, k, alphabet_n, DIGITS_AA_LOOKUP, 
                        fam_tab, hog_tab, level_arr, top_fam, cum_mode)
                else:
                    pass
                
                ## normalize family counts
                # naive normalization procedures
                if score == 'querysize':
                    top_fam_scores = norm_fam_querysize(top_fam_counts, r1.size)

                elif score == 'querysize_hogsize':
                    top_fam_scores = norm_fam_querysize_hogsize(
                        top_fam_counts, alphabet_n, k, ref_fam_counts[top_fam], r1.size)

                elif score == 'querysize_hogsize_kmerfreq':
                    top_fam_scores = norm_fam_querysize_hogsize_kmerfreq(
                        top_fam_counts, query_occ, r1.size, table_buff.size, ref_fam_counts[top_fam])

                # non-parametric normalization procedures
                elif score == 'nonparam_naive':    
                    top_fam_scores = norm_fam_nonparametric(
                        top_fam_counts, top_fam_perm_counts, r1.size)
                
                # to enable parallel loop, pick one score and such else-continue statement (or understand what is going on)
                else: 
                    continue     

                # resort by score
                if (score == 'mash_pvalue') or (score == 'kmerfreq_pvalue') or (score == 'nonparam_pvalue'):
                    idx = (top_fam_scores).argsort()
                else:
                    idx = (-top_fam_scores).argsort()

                top_fam = top_fam[idx]
                top_fam_scores = top_fam_scores[idx]

                # store them with corresponding scores
                queryFam_ranked[zz, :top_n_fams] = top_fam[:top_n_fams]
                queryFam_scores[zz, :top_n_fams] = top_fam_scores[:top_n_fams]
                queryFam_overlaps[zz, :top_n_fams] = compute_overlap(fam_highloc[top_fam[:top_n_fams]], fam_lowloc[top_fam[:top_n_fams]], k, query_len)

                ### Compute HOG scores and bestpath for top n families
                # iterate over top n families
                for fam_rank in numba.prange(top_n_fams):
                    fam_off = top_fam[fam_rank]
                    fam_ent = fam_tab[fam_off]
                    fam_hog_off = np.int64(fam_ent['HOGoff'])  # slicing with int (vs. uint) enabled parallel=True
                    fam_hog_nr = np.int64(fam_ent['HOGnum'])

                    # compute the cumulated HOG counts for that family
                    fam_hog_counts = hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr].copy()

                    fam_hog2parent = get_fam_hog2parent(fam_ent, hog_tab)
                    fam_level_offsets = get_fam_level_offsets(fam_ent, level_arr)

                    fam_hog_cumcounts = fam_hog_counts.copy()

                    if cum_mode == 'max':
                        cumulate_counts_1fam(
                            fam_hog_cumcounts, fam_level_offsets, fam_hog2parent, _sum, _max)

                    elif cum_mode == 'sum':
                        cumulate_counts_1fam(
                            fam_hog_cumcounts, fam_level_offsets, fam_hog2parent, _sum, _sum)

                    # naive normalization procedures
                    if score == 'querysize':
                        fam_hog_scores, fam_bestpath = norm_hog_querysize(
                            fam_hog_cumcounts, r1.size, fam_level_offsets, fam_hog2parent, fam_hog_counts)

                    elif score == 'querysize_hogsize':
                        fam_hog_scores, fam_bestpath = norm_hog_querysize_hogsize(
                            fam_hog_cumcounts,  r1.size, alphabet_n, k, ref_hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr], 
                            fam_level_offsets, fam_hog2parent, fam_hog_counts)

                    # querysize_kmerfreq
                    elif score == 'querysize_hogsize_kmerfreq':
                        fam_hog_scores, fam_bestpath = norm_hog_querysize_hogsize_kmerfreq(
                            fam_hog_cumcounts, r1.size, query_occ, table_buff.size, ref_hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr],
                            fam_level_offsets, fam_hog2parent, fam_hog_counts, hog_occurs[fam_hog_off:fam_hog_off + fam_hog_nr])
                    
                    # non-parametric normalization procedures
                    elif score == 'nonparam_naive':
                        fam_hog_scores, fam_bestpath = norm_hog_nonparametric(
                            fam_hog_cumcounts, r1.size, fam_level_offsets, fam_hog2parent, fam_hog_counts, 
                            top_fam_hog_perm_counts[:, idx[fam_rank]])
                                            
                    # to enable parallel loop, pick one score and such else-continue statement
                    else: 
                        continue    

                    queryRankHog_bestpath[fam_rank, zz, :fam_bestpath.size] = fam_bestpath
                    queryRankHog_scores[fam_rank, zz, :fam_hog_scores.size] = fam_hog_scores

            return queryFam_ranked, queryFam_scores, queryFam_overlaps, queryRankHog_bestpath, queryRankHog_scores

        if not self.low_mem:
            # Set nthreads, note: this only works before numba called first time!
            numba.set_num_threads(self.nthreads)
            return numba.jit(func, parallel=(True if self.nthreads > 1 else False), nopython=True, nogil=True)
        else:
            return func
    
    @lazy_property
    def _lookup_pvalue(self):
        def func(
            seqs,
            seqs_idx,
            trans,
            table_idx,
            table_buff,
            k,
            alphabet_n,
            DIGITS_AA_LOOKUP,
            fam_tab,
            hog_tab,
            level_arr,
            max_hog_nr,
            fam_filter,
            top_n_fams, 
            top_m_fams,
            cum_mode,
            score,
            ref_fam_counts,
            ref_hog_counts,
            perm_nr,
            w_size,
            dist
        ):   
            '''
            top_n_fams: number of family for which HOG scores are computed
            top_m_fams: number of family for which family scores are computed before resorting
            '''
            # highest scoring families per query sorted by counts 
            queryFam_ranked = np.zeros((len(seqs_idx) - 1, top_n_fams), dtype=np.uint32)
            # corresponding k-mer counts
            queryFam_scores = np.zeros((len(seqs_idx) - 1, top_n_fams), dtype=np.float64)
            # scores and bestpath mask for HOGs of top_n_fam
            queryRankHog_scores = np.zeros((top_n_fams, len(seqs_idx) - 1, max_hog_nr), dtype=np.float64)
            queryRankHog_bestpath = np.zeros((top_n_fams, len(seqs_idx) - 1, max_hog_nr), dtype=np.bool8)    
            # OPTION: store only HOGs on the bestpath

            # query sequence overlaps with families
            queryFam_overlaps = np.zeros((len(seqs_idx) - 1, top_n_fams), dtype=np.float64)

            # to ignore k-mers with X (88 == b'X')
            x_char = DIGITS_AA_LOOKUP[88]
            # get a flag for k-mer with any X (= last k-mer + 1)
            x_flag = table_idx.size - 1

            # iterate of sequences
            for zz in numba.prange(len(seqs_idx) - 1):

                ## get the query sequence
                s = seqs[seqs_idx[zz] : np.int(seqs_idx[zz + 1] - 1)]
                query_len = s.shape[0]
                n_kmers = query_len - (k - 1)

                # double check we don't have short peptides (of len < k)
                # note: written in this order to provide loop-optimisation hint (?)
                if n_kmers > 0:
                    pass
                else:
                    continue

                ## get the sequence unique k-mers and non-ambiguous locations
                r1, p1 = parse_seq(
                    s, DIGITS_AA_LOOKUP, n_kmers, k, trans, x_char, x_flag)

                # skip if one k-mer with X
                if len(r1) > 1:
                    pass
                elif r1[0] == x_flag:
                    continue

                ## search sequence
                hog_counts, fam_counts, query_occ, hog_occurs, fam_lowloc, fam_highloc = search_seq_kmers(
                    r1, p1, hog_tab, fam_tab, x_flag, table_idx, table_buff)

                ## get the raw top m families from summed k-mer counts
                top_fam, top_fam_counts = get_top_m_fams(
                    fam_counts, top_m_fams, cum_mode, fam_tab, hog_counts, hog_tab, level_arr, fam_filter)
                
                ## search permuted sequences if non-parametric score
                if (score == 'nonparam_naive') or (score == 'nonparam_pvalue'):
                    top_fam_perm_counts, top_fam_hog_perm_counts = search_one_seq_perms(
                        s, n_kmers, x_char, x_flag, perm_nr, top_m_fams, max_hog_nr, w_size, 
                        trans, table_idx, table_buff, k, alphabet_n, DIGITS_AA_LOOKUP, 
                        fam_tab, hog_tab, level_arr, top_fam, cum_mode)
                else:
                    pass
                
                ## normalize family counts
                # naive normalization procedures
                if score == 'querysize':
                    top_fam_scores = norm_fam_querysize(top_fam_counts, r1.size)

                elif score == 'querysize_hogsize':
                    top_fam_scores = norm_fam_querysize_hogsize(
                        top_fam_counts, alphabet_n, k, ref_fam_counts[top_fam], r1.size)

                elif score == 'querysize_hogsize_kmerfreq':
                    top_fam_scores = norm_fam_querysize_hogsize_kmerfreq(
                        top_fam_counts, query_occ, r1.size, table_buff.size, ref_fam_counts[top_fam])

                # probabilistic schemes: not supported by numba yet
                elif score == 'mash_pvalue':
                    top_fam_scores = compute_fam_mash_pvalue(
                        alphabet_n, k, ref_fam_counts[top_fam], r1.size, top_fam_counts)

                elif score == 'kmerfreq_pvalue':
                    top_fam_scores = compute_fam_kmerfreq_pvalue(
                        query_occ, r1.size, table_buff.size, ref_fam_counts[top_fam], top_fam_counts)

                # non-parametric normalization procedures
                elif score == 'nonparam_naive':    
                    top_fam_scores = norm_fam_nonparametric(
                        top_fam_counts, top_fam_perm_counts, r1.size)
                
                elif score == 'nonparam_pvalue':
                    top_fam_scores = compute_fam_nonparametric_pvalue(
                        top_fam_counts, r1.size, ref_fam_counts[top_fam], top_fam_perm_counts, dist)

                # to enable parallel loop, pick one score and such else-continue statement (or understand what is going on)
                else: 
                    continue     

                # resort by score
                if (score == 'mash_pvalue') or (score == 'kmerfreq_pvalue') or (score == 'nonparam_pvalue'):
                    idx = (top_fam_scores).argsort()
                else:
                    idx = (-top_fam_scores).argsort()

                top_fam = top_fam[idx]
                top_fam_scores = top_fam_scores[idx]

                # store them with corresponding scores
                queryFam_ranked[zz, :top_n_fams] = top_fam[:top_n_fams]
                queryFam_scores[zz, :top_n_fams] = top_fam_scores[:top_n_fams]
                queryFam_overlaps[zz, :top_n_fams] = compute_overlap(fam_highloc[top_fam[:top_n_fams]], fam_lowloc[top_fam[:top_n_fams]], k, query_len)

                ### Compute HOG scores and bestpath for top n families
                # iterate over top n families
                for fam_rank in numba.prange(top_n_fams):
                    fam_off = top_fam[fam_rank]
                    fam_ent = fam_tab[fam_off]
                    fam_hog_off = np.int64(fam_ent['HOGoff'])  # slicing with int (vs. uint) enabled parallel=True
                    fam_hog_nr = np.int64(fam_ent['HOGnum'])

                    # compute the cumulated HOG counts for that family
                    fam_hog_counts = hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr].copy()

                    fam_hog2parent = get_fam_hog2parent(fam_ent, hog_tab)
                    fam_level_offsets = get_fam_level_offsets(fam_ent, level_arr)

                    fam_hog_cumcounts = fam_hog_counts.copy()

                    if cum_mode == 'max':
                        cumulate_counts_1fam(
                            fam_hog_cumcounts, fam_level_offsets, fam_hog2parent, _sum, _max)

                    elif cum_mode == 'sum':
                        cumulate_counts_1fam(
                            fam_hog_cumcounts, fam_level_offsets, fam_hog2parent, _sum, _sum)

                    # naive normalization procedures
                    if score == 'querysize':
                        fam_hog_scores, fam_bestpath = norm_hog_querysize(
                            fam_hog_cumcounts, r1.size, fam_level_offsets, fam_hog2parent, fam_hog_counts)

                    elif score == 'querysize_hogsize':
                        fam_hog_scores, fam_bestpath = norm_hog_querysize_hogsize(
                            fam_hog_cumcounts,  r1.size, alphabet_n, k, ref_hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr], 
                            fam_level_offsets, fam_hog2parent, fam_hog_counts)

                    # querysize_kmerfreq
                    elif score == 'querysize_hogsize_kmerfreq':
                        fam_hog_scores, fam_bestpath = norm_hog_querysize_hogsize_kmerfreq(
                            fam_hog_cumcounts, r1.size, query_occ, table_buff.size, ref_hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr],
                            fam_level_offsets, fam_hog2parent, fam_hog_counts, hog_occurs[fam_hog_off:fam_hog_off + fam_hog_nr])

                    # probabilistic schemes: not supported by numba yet
                    elif score == 'mash_pvalue':
                        fam_hog_scores, fam_bestpath = compute_hog_mash_pvalue(
                            fam_hog_cumcounts,  r1.size, alphabet_n, k, ref_hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr], 
                            fam_level_offsets, fam_hog2parent, fam_hog_counts)

                    elif score == 'kmerfreq_pvalue':
                        fam_hog_scores, fam_bestpath = compute_hog_kmerfreq_pvalue(
                            fam_hog_cumcounts, r1.size, query_occ, table_buff.size, ref_hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr],
                            fam_level_offsets, fam_hog2parent, fam_hog_counts, hog_occurs[fam_hog_off:fam_hog_off + fam_hog_nr])
                    
                    # non-parametric normalization procedures
                    elif score == 'nonparam_naive':
                        fam_hog_scores, fam_bestpath = norm_hog_nonparametric(
                            fam_hog_cumcounts, r1.size, fam_level_offsets, fam_hog2parent, fam_hog_counts, 
                            top_fam_hog_perm_counts[:, idx[fam_rank]])
                    
                    elif score == 'nonparam_pvalue':
                        fam_hog_scores, fam_bestpath = compute_hog_nonparametric_pvalue(
                            fam_hog_cumcounts, r1.size, ref_hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr], 
                            top_fam_hog_perm_counts[:, fam_rank], fam_level_offsets, fam_hog2parent, fam_hog_counts, dist)
                        
                    # to enable parallel loop, pick one score and such else-continue statement
                    else: 
                        continue    

                    queryRankHog_bestpath[fam_rank, zz, :fam_bestpath.size] = fam_bestpath
                    queryRankHog_scores[fam_rank, zz, :fam_hog_scores.size] = fam_hog_scores

            return queryFam_ranked, queryFam_scores, queryFam_overlaps, queryRankHog_bestpath, queryRankHog_scores

        if not self.low_mem:
            # Set nthreads, note: this only works before numba called first time!
            numba.set_num_threads(self.nthreads)
            return numba.jit(func, parallel=(True if self.nthreads > 1 else False), nopython=True, nogil=True)
        else:
            return func