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
                
@numba.njit
def get_fam_bestpath(fam_hog_counts, fam_level_offsets, hog2parent, pv_score=False):
    
    fam_bestpath = np.full(fam_hog_counts.shape, False)

    # root-HOG
    hog_score = fam_hog_counts[0]
    if hog_score > 0:
        fam_bestpath[0] = True

    # loop through hog levels
    for i in range(1, fam_level_offsets.size - 2):
        x = fam_level_offsets[i : i + 2]
        hog_offsets = np.arange(x[0], x[1])

        # grab parents
        parent_offsets = hog2parent[hog_offsets]
        
        # store best path
        store_bestpath(hog_offsets, parent_offsets, fam_bestpath, fam_hog_counts, pv_score)
    
    return fam_bestpath

@numba.njit
def _norm_hog_correct_query_size(
    fam_hog_cumcounts, query_count, fam_level_offsets, hog2parent, fam_hog_counts, pv_score=False):

    fam_hog_scores = np.zeros(fam_hog_cumcounts.shape, dtype=np.float64)
    fam_bestpath = np.full(fam_hog_cumcounts.shape, False)
    revcum_counts = np.full(fam_hog_cumcounts.shape, query_count, dtype=np.uint16)

    # compute and store root-HOG score
    fam_hog_scores[0] = np.float64(fam_hog_cumcounts[0]) / query_count

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
        store_bestpath(hog_offsets, parent_offsets, fam_bestpath, fam_hog_scores, pv_score)

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

	def merge_search(self, seqs=None, ids=None, fasta_file=None):
		# load query sequences
		if seqs:
		    sbuff = SequenceBuffer(seqs=seqs, ids=ids)
		elif fasta_file:
		    sbuff = SequenceBuffer(fasta_file=fasta_file)

		queryFam_ranked, queryFam_scores, queryRankHog_bestpath, queryRankHog_scores = self._lookup(
		    sbuff.buff,
		    sbuff.idx,
		    self.trans,
		    self.table_idx,
		    self.table_buff,
		    self.ki.k,
		    self.ki.alphabet.DIGITS_AA_LOOKUP,
		    self.fam_tab,
		    self.hog_tab,
		    self.level_arr,
		    self.max_hog_nr,
		    top_n_fams = 1,
		    top_m_fams = 10,
		    cum_mode = 'max', 
		    score = 'correct_querysize')

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
		    DIGITS_AA_LOOKUP,
		    fam_tab,
		    hog_tab,
		    level_arr,
		    max_hog_nr,
		    top_n_fams = 1, 
		    top_m_fams = 1,
		    cum_mode = 'max',
		    score = 'correct_querysize'
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
		    # last_char = DIGITS_AA_LOOKUP[-1]
		    # x_flag = 0
		    # for j in range(k):
		    #     x_flag += trans[j] * last_char
		    # x_flag += 1

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

		        # compute family score (normalize by number of unique k-mers)
		        top_fam_scores = top_fam_counts / r1.size

		        # resort by score
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

		            fam_hog_cumcounts = fam_hog_counts.copy()  # because I need both count and cum counts in correct query size

		            # don't know why I cannot one line this within numba
		            if cum_mode == 'max':
		                cumulate_counts_1fam(
		                    fam_hog_cumcounts, fam_level_offsets, fam_hog2parent, _sum, _max)
		            elif cum_mode == 'sum':
		                cumulate_counts_1fam(
		                    fam_hog_cumcounts, fam_level_offsets, fam_hog2parent, _sum, _sum)

		            if score == 'querysize':
		                # compute HOG score (normalize by number of unique k-mers)
		                fam_hog_scores = fam_hog_cumcounts / r1.size

		                # compute highest scoring root-to-leaf HOG path within the family
		                fam_bestpath = get_fam_bestpath(fam_hog_scores, fam_level_offsets, fam_hog2parent)

		            # problem of using multiple if elif here! but working above >?!?!?
		            #elif score == 'correct_querysize':
		            else:
		                fam_hog_scores, fam_bestpath = _norm_hog_correct_query_size(
		                    fam_hog_cumcounts, r1.size, fam_level_offsets, fam_hog2parent, fam_hog_counts)

		            queryRankHog_bestpath[fam_rank, zz, :fam_bestpath.size] = fam_bestpath
		            queryRankHog_scores[fam_rank, zz, :fam_hog_scores.size] = fam_hog_scores

		    return queryFam_ranked, queryFam_scores, queryRankHog_bestpath, queryRankHog_scores

		if not self.low_mem:
		    # Set nthreads, note: this only works before numba called first time!
		    numba.set_num_threads(self.nthreads)
		    return numba.jit(func, parallel=(True if self.nthreads > 1 else False), nopython=True, nogil=True)
		else:
		    return func
