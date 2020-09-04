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
from scipy import special

from ._utils import LOG
from .index import get_transform, SequenceBuffer, QuerySequenceBuffer
from .hierarchy import get_descendant_hogs, get_descendant_prots

# --> will be renamed Search

# numba does not like nested methods 
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

@numba.njit
def _max(x, y):
    return max(x, y)

@numba.njit
def _sum(x, y):
        return x + y 
    
@numba.njit
def cumulate_counts_1fam(
    hog_cum_counts, fam_level_offsets, hog2parent, cum_fun, prop_fun):

    current_best_child_count = np.zeros(hog_cum_counts.shape, dtype=np.uint16)

    # iterate over level offsets backward
    for i in range(fam_level_offsets.size - 2):
        x = fam_level_offsets[-i - 3 : -i - 1]

        # when reaching level, sum all hog counts with their best child count
        hog_cum_counts[x[0] : x[1]] = cum_fun(
            hog_cum_counts[x[0] : x[1]], current_best_child_count[x[0] : x[1]]
        )

        # update current_best_child_count of the parents of the current hogs
        for i in range(x[0], x[1]):
            parent_off = hog2parent[i]

            # only if parent exists
            if parent_off != -1:
                c = current_best_child_count[hog2parent[i]]
                current_best_child_count[hog2parent[i]] = prop_fun(
                    c, hog_cum_counts[i]
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

## scoring scheme functions
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
    
    # makes sense to remove exp_kmer_counts query_counts too right?
    return (fam_counts - exp_kmer_counts) / (query_counts - exp_kmer_counts)

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
    
    # makes sense to remove exp_kmer_counts query_counts too right?
    return (fam_counts - exp_kmer_counts) / (query_counts - exp_kmer_counts)

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
    
    fam_hog_scores[0] = (np.float64(fam_hog_cumcounts[0]) - exp_kmer_counts) / (query_counts - exp_kmer_counts)

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

        fam_hog_scores[hog_offsets] = (fam_hog_cumcounts[hog_offsets] - exp_kmer_counts) / (qh_count - exp_kmer_counts)

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

    fam_hog_scores[0] = (np.float64(fam_hog_cumcounts[0]) - exp_kmer_counts) / (query_counts - exp_kmer_counts)

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

        fam_hog_scores[hog_offsets] = (fam_hog_cumcounts[hog_offsets] - exp_kmer_counts) / (qh_count - exp_kmer_counts)

        # store bestpath
        store_bestpath(hog_offsets, parent_offsets, fam_bestpath, fam_hog_scores, pv_score=False)

    return fam_hog_scores, fam_bestpath

## probabilistic scoring schemes
def poisson_log_pmf(k, lda):
    return k * np.lib.scimath.log(lda) - lda - special.gammaln(k + 1)

def compute_log_poisson_pvalue(query_counts, hog_cum_counts, kmer_bernoulli):
        
    # probabilities for tail x values
    tail_size = query_counts - hog_cum_counts 
    tail_log_probs = poisson_log_pmf(np.arange(hog_cum_counts, query_counts + 1), np.full(tail_size + 1, kmer_bernoulli * query_counts))

    # sum of these tail probabilities
    return special.logsumexp(tail_log_probs)

def compute_fam_mash_pvalue(alphabet_n, k, ref_fam_counts, query_counts, fam_counts):
    
    # probability to draw a k-mer in family k-mer set
    kmer_prob = kmer_prob_mash(alphabet_n, k)
    kmer_bernoulli = get_kmer_bernoulli(kmer_prob, ref_fam_counts)

    # log p-value
    fam_scores = np.zeros(fam_counts.size, np.float64)
    for i in range(fam_counts.size): 
        fam_scores[i] = compute_log_poisson_pvalue(query_counts, fam_counts[i], kmer_bernoulli[i])
        
    return fam_scores

def compute_fam_kmerfreq_pvalue(query_occurs, query_counts, all_kmer_occurs, ref_fam_counts, fam_counts):
    
    # probability to draw a k-mer in family k-mer set
    kmer_prob = query_kmer_prob_freq(query_occurs, query_counts, all_kmer_occurs)
    kmer_bernoulli = get_kmer_bernoulli(kmer_prob, ref_fam_counts)

    # log p-value
    fam_scores = np.zeros(fam_counts.size, np.float64)
    for i in range(fam_counts.size): 
        fam_scores[i] = compute_log_poisson_pvalue(query_counts, fam_counts[i], kmer_bernoulli[i])
        
    return fam_scores

def compute_hog_mash_pvalue(
    fam_hog_cumcounts, query_counts, alphabet_n, k, fam_ref_hog_counts, fam_level_offsets, hog2parent, fam_hog_counts):

    fam_hog_scores = np.zeros(fam_hog_cumcounts.shape, dtype=np.float64)
    fam_bestpath = np.full(fam_hog_cumcounts.shape, False)
    revcum_counts = np.full(fam_hog_cumcounts.shape, query_counts, dtype=np.uint16)
    
    ## root-HOG p-value
    kmer_prob = kmer_prob_mash(alphabet_n, k)
    kmer_bernoulli = get_kmer_bernoulli(kmer_prob, fam_ref_hog_counts[0])
    fam_hog_scores[0] = compute_log_poisson_pvalue(query_counts, fam_hog_cumcounts[0], kmer_bernoulli)
    
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
                fam_hog_scores[hog_offsets[j]] = compute_log_poisson_pvalue(
                    qh_count[j], fam_hog_cumcounts[hog_offsets[j]], kmer_bernoulli[j])
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
    fam_hog_scores[0] = compute_log_poisson_pvalue(query_counts, fam_hog_cumcounts[0], kmer_bernoulli)

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
                fam_hog_scores[hog_offsets[j]] = compute_log_poisson_pvalue(
                    qh_count[j], fam_hog_cumcounts[hog_offsets[j]], kmer_bernoulli[j])
            else:
                fam_hog_scores[hog_offsets[j]] = 0.0

        # store bestpath
        store_bestpath(hog_offsets, parent_offsets, fam_bestpath, fam_hog_scores, pv_score=True)

    return fam_hog_scores, fam_bestpath

def compute_hog_kmerfreqmin_pvalue(
    fam_hog_cumcounts, query_counts, query_occurs, all_kmer_occurs, fam_ref_hog_counts, fam_level_offsets, hog2parent, fam_hog_counts, fam_hog_occurs):

    fam_hog_scores = np.zeros(fam_hog_cumcounts.shape, dtype=np.float64)
    fam_bestpath = np.full(fam_hog_cumcounts.shape, False)
    revcum_counts = np.full(fam_hog_cumcounts.shape, query_counts, dtype=np.uint16)
    revcum_occurs = np.full(fam_hog_cumcounts.shape, query_occurs, dtype=np.uint32)

    ## root-HOG score
    # compute expected number of k-mer matches
    kmer_prob = query_kmer_prob_freq(query_occurs, query_counts, all_kmer_occurs)
    kmer_bernoulli = get_kmer_bernoulli(kmer_prob, fam_ref_hog_counts[0])
    fam_hog_scores[0] = compute_log_poisson_pvalue(query_counts, fam_hog_cumcounts[0], kmer_bernoulli)

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
                    n, fam_hog_cumcounts[hog_offsets[j]], kmer_bernoulli[j])
            else:
                fam_hog_scores[hog_offsets[j]] = 0.0

        # store bestpath
        store_bestpath(hog_offsets, parent_offsets, fam_bestpath, fam_hog_scores, pv_score=True)

    return fam_hog_scores, fam_bestpath

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

    def merge_search(self, seqs=None, ids=None, fasta_file=None, score='querysize', cum_mode='max', top_m_fams=10, top_n_fams=1):
    	# load query sequences
    	if seqs:
    	    sbuff = SequenceBuffer(seqs=seqs, ids=ids)
    	elif fasta_file:
    	    sbuff = SequenceBuffer(fasta_file=fasta_file)

    	if cum_mode == 'sum':
    		ref_fam_counts = self.ref_fam_counts_sum
    		ref_hog_counts = self.ref_hog_counts_sum
    	elif cum_mode == 'max':
    		ref_fam_counts = self.ref_fam_counts_max
    		ref_hog_counts = self.ref_hog_counts_max

    	# set to low mem if probabilistic score
    	if score in {'mash_pvalue', 'kmerfreq_pvalue', 'kmerfreqmin_pvalue'}:
    		self.low_mem = True
    		lookup_fun = self._lookup_pvalue
    	else:
    		self.low_mem = False
    		lookup_fun = self._lookup

    	queryFam_ranked, queryFam_scores, queryRankHog_bestpath, queryRankHog_scores = lookup_fun(
    	    sbuff.buff,
    	    sbuff.idx,
    	    self.trans,
    	    self.table_idx[:],
    	    self.table_buff[:],
    	    self.ki.k,
    	    self.ki.alphabet.n,
    	    self.ki.alphabet.DIGITS_AA_LOOKUP,
    	    self.fam_tab,
    	    self.hog_tab,
    	    self.level_arr,
    	    self.max_hog_nr,
    	    top_n_fams = top_n_fams,
    	    top_m_fams = top_m_fams,
    	    cum_mode = cum_mode, 
    	    score = score,
    	    ref_fam_counts = ref_fam_counts,
    	    ref_hog_counts = ref_hog_counts)

    	self._queryFam_ranked = queryFam_ranked
    	self._queryFam_scores = queryFam_scores
    	self._queryRankHog_bestpath = queryRankHog_bestpath
    	self._queryRankHog_scores = queryRankHog_scores

    	# store ids of sbuff
    	self._query_ids = sbuff.ids.flatten()

    def output_results(self, threshold=0.1):

    	def get_prot_ids(h):
    	    desc_hogs = list(get_descendant_hogs(h, self.db._hog_tab, self.db._chog_arr)) + [h]
    	    prot_ii = get_descendant_prots(desc_hogs, self.db._hog_tab, self.db._cprot_arr)
    	    return self.db._prot_tab.read_coordinates(prot_ii, 'ID')

    	qseqid = self._query_ids
    	family = self._queryFam_ranked[:, 0]
    	family_score = self._queryFam_scores[:, 0]

    	# generate the dataframe
    	def generate_results(threshold):
    	    for i in np.argwhere(~np.isnan(family_score)).flatten():
    	        # find best scoring subfamily
    	        fam_hog_nr = self.db._fam_tab[family[i]]['HOGnum']
    	        fam_bestpath = self._queryRankHog_bestpath[0, i, :fam_hog_nr]
    	        fam_hog_scores = self._queryRankHog_scores[0, i, :fam_hog_nr]
    	        best_j = None
    	        best_s = np.inf
    	        for j in np.argwhere(fam_bestpath).flatten():
    	            s = fam_hog_scores[j]
    	            if s < best_s and s >= threshold:
    	                best_j = j
    	                best_s = s
    	        if best_j is not None:
    	            hog_off = self.db._fam_tab[family[i]]['HOGoff']
    	            # update best_j to get the subfamily offset
    	            best_j = int(best_j + hog_off)
    	            z = {'qseqid': qseqid[i],
    	                 'family': self.db._hog_tab[self.db._fam_tab[family[i]]['HOGoff']]['OmaID'].decode('ascii'),
    	                 'family-score': family_score[i],
    	                 'subfamily': self.db._hog_tab[best_j]['OmaID'].decode('ascii'),
    	                 'subfamily-score': best_s}
    	            if self.include_extant_genes:
    	                z['subfamily-geneset'] = ','.join(map(lambda x: x.decode('ascii'), get_prot_ids(best_j)))
    	            yield z

    	h = ['qseqid', 'family', 'family-score', 'subfamily', 'subfamily-score']
    	if self.include_extant_genes:
    	    h.append('subfamily-geneset')
    	
    	# if >0 family hit
    	if np.max(family_score) >= threshold:
    		return pd.DataFrame(generate_results(threshold))[h]
    	else:
    		return pd.DataFrame()

    @lazy_property
    def _lookup(self):
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
            top_n_fams = 1, 
            top_m_fams = 1,
            cum_mode = 'max',
            score = 'querysize',
            ref_fam_counts = np.array([]),
            ref_hog_counts = np.array([])
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

            # to ignore k-mers with X (88 == b'X')
            x_char = DIGITS_AA_LOOKUP[88]
            # get a flag for k-mer with any X (= last k-mer + 1)
            x_flag = table_idx.size - 1

            # iterate of sequences
            for zz in numba.prange(len(seqs_idx) - 1):

                ## get the query sequence
                s = seqs[seqs_idx[zz] : np.int(seqs_idx[zz + 1] - 1)]
                n_kmers = s.shape[0] - (k - 1)

                # double check we don't have short peptides (of len < k)
                # note: written in this order to provide loop-optimisation hint (?)
                if n_kmers > 0:
                    pass
                else:
                    continue

                ## get the sequence unique k-mers
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
                r1 = np.unique(r)

                # skip if one k-mer with X
                if len(r1) > 1:
                    pass
                elif r1[0] == x_flag:
                    continue

                ## search k-mer table
                hog_counts = np.zeros(hog_tab.size, dtype=np.uint16)
                fam_counts = np.zeros(fam_tab.size, dtype=np.uint16)
                
                # store total occurences of query k-mers
                query_occ = np.uint32(0)
                
                # store total occurence of query-hog k-mers
                hog_occurs = np.zeros(hog_tab.size, dtype=np.uint32)
                
                # iterate unique k-mers
                for m in numba.prange(r1.shape[0]):
                    kmer = r1[m]

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
                    
                ### Get top n families
                # get top m families from summed k-mer counts
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

                # naive normalization procedures            
                if score == 'querysize':
                    top_fam_scores = norm_fam_querysize(
                        top_fam_counts, r1.size)
                    
                elif score == 'querysize_hogsize':
                    top_fam_scores = norm_fam_querysize_hogsize(
                        top_fam_counts, alphabet_n, k, ref_fam_counts[top_fam], r1.size)
                
                # querysize_kmerfreq            
                elif score == 'querysize_hogsize_kmerfreq':
                    top_fam_scores = norm_fam_querysize_hogsize_kmerfreq(
                        top_fam_counts, query_occ, r1.size, table_buff.size, ref_fam_counts[top_fam])

                # not supported by numba
                # # probabilistic schemes
                # elif score == 'mash_pvalue':
                #     top_fam_scores = compute_fam_mash_pvalue(
                #         alphabet_n, k, ref_fam_counts[top_fam], r1.size, top_fam_counts)
                    
                # elif score == 'kmerfreq_pvalue':
                #     top_fam_scores = compute_fam_kmerfreq_pvalue(
                #         query_occ, r1.size, table_buff.size, ref_fam_counts[top_fam], top_fam_counts)
                    
                # to enable parallel loop, pick one score and such else-continue statement (or understand what is going on)
                else: 
                    continue     
                
                # resort by score
                if (score == 'mash_pvalue') or (score == 'kmerfreq_pvalue'):
                    idx = (top_fam_scores).argsort()
                else:
                    idx = (-top_fam_scores).argsort()

                top_fam = top_fam[idx]
                top_fam_scores = top_fam_scores[idx]

                # store them with corresponding scores
                queryFam_ranked[zz, :top_n_fams] = top_fam[:top_n_fams]
                queryFam_scores[zz, :top_n_fams] = top_fam_scores[:top_n_fams]

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
                    
                    # not supported by numba
                    # # probabilistic schemes
                    # elif score == 'mash_pvalue':
                    #     fam_hog_scores, fam_bestpath = compute_hog_mash_pvalue(
                    #         fam_hog_cumcounts,  r1.size, alphabet_n, k, ref_hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr], 
                    #         fam_level_offsets, fam_hog2parent, fam_hog_counts)
                    
                    # elif score == 'kmerfreq_pvalue':
                    #     fam_hog_scores, fam_bestpath = compute_hog_kmerfreq_pvalue(
                    #         fam_hog_cumcounts, r1.size, query_occ, table_buff.size, ref_hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr],
                    #         fam_level_offsets, fam_hog2parent, fam_hog_counts, hog_occurs[fam_hog_off:fam_hog_off + fam_hog_nr])

                    # to enable parallel loop, pick one score and such else-continue statement
                    else: 
                        continue    
                        
                    queryRankHog_bestpath[fam_rank, zz, :fam_bestpath.size] = fam_bestpath
                    queryRankHog_scores[fam_rank, zz, :fam_hog_scores.size] = fam_hog_scores

            return queryFam_ranked, queryFam_scores, queryRankHog_bestpath, queryRankHog_scores

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
            top_n_fams = 1, 
            top_m_fams = 1,
            cum_mode = 'max',
            score = 'querysize',
            ref_fam_counts = np.array([]),
            ref_hog_counts = np.array([])
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

            # to ignore k-mers with X (88 == b'X')
            x_char = DIGITS_AA_LOOKUP[88]
            # get a flag for k-mer with any X (= last k-mer + 1)
            x_flag = table_idx.size - 1

            # iterate of sequences
            for zz in numba.prange(len(seqs_idx) - 1):

                ## get the query sequence
                s = seqs[seqs_idx[zz] : np.int(seqs_idx[zz + 1] - 1)]
                n_kmers = s.shape[0] - (k - 1)

                # double check we don't have short peptides (of len < k)
                # note: written in this order to provide loop-optimisation hint (?)
                if n_kmers > 0:
                    pass
                else:
                    continue

                ## get the sequence unique k-mers
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
                r1 = np.unique(r)

                # skip if one k-mer with X
                if len(r1) > 1:
                    pass
                elif r1[0] == x_flag:
                    continue

                ## search k-mer table
                hog_counts = np.zeros(hog_tab.size, dtype=np.uint16)
                fam_counts = np.zeros(fam_tab.size, dtype=np.uint16)
                
                # store total occurences of query k-mers
                query_occ = np.uint32(0)
                
                # store total occurence of query-hog k-mers
                hog_occurs = np.zeros(hog_tab.size, dtype=np.uint32)
                
                # iterate unique k-mers
                for m in numba.prange(r1.shape[0]):
                    kmer = r1[m]

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
                    
                ### Get top n families
                # get top m families from summed k-mer counts
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

                # naive normalization procedures            
                if score == 'querysize':
                    top_fam_scores = norm_fam_querysize(
                        top_fam_counts, r1.size)
                    
                elif score == 'querysize_hogsize':
                    top_fam_scores = norm_fam_querysize_hogsize(
                        top_fam_counts, alphabet_n, k, ref_fam_counts[top_fam], r1.size)
                
                # querysize_kmerfreq            
                elif score == 'querysize_hogsize_kmerfreq':
                    top_fam_scores = norm_fam_querysize_hogsize_kmerfreq(
                        top_fam_counts, query_occ, r1.size, table_buff.size, ref_fam_counts[top_fam])

                # probabilistic schemes
                elif score == 'mash_pvalue':
                    top_fam_scores = compute_fam_mash_pvalue(
                        alphabet_n, k, ref_fam_counts[top_fam], r1.size, top_fam_counts)
                    
                elif score == 'kmerfreq_pvalue' or score == 'kmerfreqmin_pvalue':
                    top_fam_scores = compute_fam_kmerfreq_pvalue(
                        query_occ, r1.size, table_buff.size, ref_fam_counts[top_fam], top_fam_counts)
                    
                # to enable parallel loop, pick one score and such else-continue statement (or understand what is going on)
                else: 
                    continue     
                
                # resort by score
                if (score == 'mash_pvalue') or (score == 'kmerfreq_pvalue') or (score == 'kmerfreqmin_pvalue'):
                    idx = (top_fam_scores).argsort()
                else:
                    idx = (-top_fam_scores).argsort()

                top_fam = top_fam[idx]
                top_fam_scores = top_fam_scores[idx]

                # store them with corresponding scores
                queryFam_ranked[zz, :top_n_fams] = top_fam[:top_n_fams]
                queryFam_scores[zz, :top_n_fams] = top_fam_scores[:top_n_fams]

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
                    
                    # probabilistic schemes
                    elif score == 'mash_pvalue':
                        fam_hog_scores, fam_bestpath = compute_hog_mash_pvalue(
                            fam_hog_cumcounts,  r1.size, alphabet_n, k, ref_hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr], 
                            fam_level_offsets, fam_hog2parent, fam_hog_counts)
                    
                    elif score == 'kmerfreq_pvalue':
                        fam_hog_scores, fam_bestpath = compute_hog_kmerfreq_pvalue(
                            fam_hog_cumcounts, r1.size, query_occ, table_buff.size, ref_hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr],
                            fam_level_offsets, fam_hog2parent, fam_hog_counts, hog_occurs[fam_hog_off:fam_hog_off + fam_hog_nr])

                    elif score == 'kmerfreqmin_pvalue':
                        fam_hog_scores, fam_bestpath = compute_hog_kmerfreqmin_pvalue(
                            fam_hog_cumcounts, r1.size, query_occ, table_buff.size, ref_hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr],
                            fam_level_offsets, fam_hog2parent, fam_hog_counts, hog_occurs[fam_hog_off:fam_hog_off + fam_hog_nr])

                    # to enable parallel loop, pick one score and such else-continue statement
                    else: 
                        continue    
                        
                    queryRankHog_bestpath[fam_rank, zz, :fam_bestpath.size] = fam_bestpath
                    queryRankHog_scores[fam_rank, zz, :fam_hog_scores.size] = fam_hog_scores

            return queryFam_ranked, queryFam_scores, queryRankHog_bestpath, queryRankHog_scores

        if not self.low_mem:
            # Set nthreads, note: this only works before numba called first time!
            numba.set_num_threads(self.nthreads)
            return numba.jit(func, parallel=(True if self.nthreads > 1 else False), nopython=True, nogil=True)
        else:
            return func

# v0.1.3
	# def _lookup(self):
	# 	def func(
	# 	    seqs,
	# 	    seqs_idx,
	# 	    trans,
	# 	    table_idx,
	# 	    table_buff,
	# 	    k,
	# 	    DIGITS_AA_LOOKUP,
	# 	    fam_tab,
	# 	    hog_tab,
	# 	    level_arr,
	# 	    max_hog_nr,
	# 	    top_n_fams = 1, 
	# 	    top_m_fams = 1,
	# 	    cum_mode = 'max',
	# 	    score = 'correct_querysize'
	# 	):   
	# 	    '''
	# 	    top_n_fams: number of family for which HOG scores are computed
	# 	    top_m_fams: number of family for which family scores are computed before resorting
	# 	    '''
	# 	    # highest scoring families per query sorted by counts 
	# 	    queryFam_ranked = np.zeros((len(seqs_idx) - 1, top_n_fams), dtype=np.uint32)
	# 	    # corresponding k-mer counts
	# 	    queryFam_scores = np.zeros((len(seqs_idx) - 1, top_n_fams), dtype=np.float64)
	# 	    # scores and bestpath mask for HOGs of top_n_fam
	# 	    queryRankHog_scores = np.zeros((top_n_fams, len(seqs_idx) - 1, max_hog_nr), dtype=np.float64)
	# 	    queryRankHog_bestpath = np.zeros((top_n_fams, len(seqs_idx) - 1, max_hog_nr), dtype=np.bool8)    
	# 	    # OPTION: store only HOGs on the bestpath

	# 	    # to ignore k-mers with X (88 == b'X')
	# 	    x_char = DIGITS_AA_LOOKUP[88]
	# 	    # get a flag for k-mer with any X (= last k-mer + 1)
	# 	    x_flag = table_idx.size - 1
	# 	    # last_char = DIGITS_AA_LOOKUP[-1]
	# 	    # x_flag = 0
	# 	    # for j in range(k):
	# 	    #     x_flag += trans[j] * last_char
	# 	    # x_flag += 1

	# 	    # iterate of sequences
	# 	    for zz in numba.prange(len(seqs_idx) - 1):

	# 	        ## get the query sequence
	# 	        s = seqs[seqs_idx[zz] : np.int(seqs_idx[zz + 1] - 1)]
	# 	        n_kmers = s.shape[0] - (k - 1)

	# 	        # double check we don't have short peptides (of len < k)
	# 	        # note: written in this order to provide loop-optimisation hint (?)
	# 	        if n_kmers > 0:
	# 	            pass
	# 	        else:
	# 	            continue

	# 	        ## get the sequence unique k-mers
	# 	        s_norm = DIGITS_AA_LOOKUP[s]
	# 	        r = np.zeros(n_kmers, dtype=np.uint32)  # max kmer 7
	# 	        for i in numba.prange(n_kmers):
	# 	            # numba can't do np.dot with non-float
	# 	            for j in numba.prange(k):
	# 	                r[i] += trans[j] * s_norm[i + j]
	# 	            # does k-mer contain any X?
	# 	            x_seen = np.any(s_norm[i:i+k] == x_char)
	# 	            # if yes, replace it by the x_flag
	# 	            r[i] = r[i] if not x_seen else x_flag
	# 	        r1 = np.unique(r)

	# 	        # skip if one k-mer with X
	# 	        if len(r1) > 1:
	# 	            pass
	# 	        elif r1[0] == x_flag:
	# 	            continue

	# 	        ## search k-mer table
	# 	        hog_counts = np.zeros(hog_tab.size, dtype=np.uint16)
	# 	        fam_counts = np.zeros(fam_tab.size, dtype=np.uint16)

	# 	        # iterate unique k-mers
	# 	        for m in numba.prange(r1.shape[0]):
	# 	            kmer = r1[m]

	# 	            # to ignore k-mers with X
	# 	            if kmer == x_flag:
	# 	                continue
	# 	            else:
	# 	                pass

	# 	            # get mapping to HOGs
	# 	            x = table_idx[kmer : kmer + 2]
	# 	            hogs = table_buff[x[0] : x[1]]
	# 	            fams = hog_tab['FamOff'][hogs]
	# 	            hog_counts[hogs] += np.uint16(1)
	# 	            fam_counts[fams] += np.uint16(1)

	# 	        ### Get top n families
	# 	        # get top m families from summed k-mer counts
	# 	        top_fam = np.argsort(fam_counts)[::-1][:top_m_fams]
	# 	        top_fam_counts = fam_counts[top_fam]

	# 	        # cumulated queryHOG counts for the top m families
	# 	        # small optimization: remember the fam_hog_cumcounts for the top n families for next step
	# 	        if cum_mode == 'max':

	# 	            # iterate over top n families
	# 	            for fam_rank in numba.prange(top_m_fams):
	# 	                fam_off = top_fam[fam_rank]
	# 	                fam_ent = fam_tab[fam_off]
	# 	                fam_hog_off = fam_ent['HOGoff']
	# 	                fam_hog_nr = fam_ent['HOGnum']

	# 	                # compute the cumulated HOG counts for that family
	# 	                fam_hog_cumcounts = hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr].copy()

	# 	                fam_hog2parent = get_fam_hog2parent(fam_ent, hog_tab)
	# 	                fam_level_offsets = get_fam_level_offsets(fam_ent, level_arr)

	# 	                cumulate_counts_1fam(fam_hog_cumcounts, fam_level_offsets, fam_hog2parent, _sum, _max)

	# 	                # replace summed counts by maxed counts (from highest scoring root-to-leaf path)
	# 	                top_fam_counts[fam_rank] = fam_hog_cumcounts[0]

	# 	        # compute family score (normalize by number of unique k-mers)
	# 	        top_fam_scores = top_fam_counts / r1.size

	# 	        # resort by score
	# 	        idx = (-top_fam_scores).argsort()
	# 	        top_fam = top_fam[idx]
	# 	        top_fam_scores = top_fam_scores[idx]

	# 	        # store them with corresponding scores
	# 	        queryFam_ranked[zz, :top_n_fams] = top_fam[:top_n_fams]
	# 	        queryFam_scores[zz, :top_n_fams] = top_fam_scores[:top_n_fams]

	# 	        ### Compute HOG scores and bestpath for top n families
	# 	        # iterate over top n families
	# 	        for fam_rank in numba.prange(top_n_fams):
	# 	            fam_off = top_fam[fam_rank]
	# 	            fam_ent = fam_tab[fam_off]
	# 	            fam_hog_off = np.int64(fam_ent['HOGoff'])  # slicing with int (vs. uint) enabled parallel=True
	# 	            fam_hog_nr = np.int64(fam_ent['HOGnum'])

	# 	            # compute the cumulated HOG counts for that family
	# 	            fam_hog_counts = hog_counts[fam_hog_off:fam_hog_off + fam_hog_nr].copy()

	# 	            fam_hog2parent = get_fam_hog2parent(fam_ent, hog_tab)
	# 	            fam_level_offsets = get_fam_level_offsets(fam_ent, level_arr)

	# 	            fam_hog_cumcounts = fam_hog_counts.copy()  # because I need both count and cum counts in correct query size

	# 	            # don't know why I cannot one line this within numba
	# 	            if cum_mode == 'max':
	# 	                cumulate_counts_1fam(
	# 	                    fam_hog_cumcounts, fam_level_offsets, fam_hog2parent, _sum, _max)
	# 	            elif cum_mode == 'sum':
	# 	                cumulate_counts_1fam(
	# 	                    fam_hog_cumcounts, fam_level_offsets, fam_hog2parent, _sum, _sum)

	# 	            if score == 'querysize':
	# 	                # compute HOG score (normalize by number of unique k-mers)
	# 	                fam_hog_scores = fam_hog_cumcounts / r1.size

	# 	                # compute highest scoring root-to-leaf HOG path within the family
	# 	                fam_bestpath = get_fam_bestpath(fam_hog_scores, fam_level_offsets, fam_hog2parent)

	# 	            # problem of using multiple if elif here! but working above >?!?!?
	# 	            #elif score == 'correct_querysize':
	# 	            else:
	# 	                fam_hog_scores, fam_bestpath = _norm_hog_correct_query_size(
	# 	                    fam_hog_cumcounts, r1.size, fam_level_offsets, fam_hog2parent, fam_hog_counts)

	# 	            queryRankHog_bestpath[fam_rank, zz, :fam_bestpath.size] = fam_bestpath
	# 	            queryRankHog_scores[fam_rank, zz, :fam_hog_scores.size] = fam_hog_scores

	# 	    return queryFam_ranked, queryFam_scores, queryRankHog_bestpath, queryRankHog_scores

	# 	if not self.low_mem:
	# 	    # Set nthreads, note: this only works before numba called first time!
	# 	    numba.set_num_threads(self.nthreads)
	# 	    return numba.jit(func, parallel=(True if self.nthreads > 1 else False), nopython=True, nogil=True)
	# 	else:
	# 	    return func
