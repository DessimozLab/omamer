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
from math import sqrt
from rvlib import Binomial
from time import time

from ._utils import LOG
from .index import get_transform, SequenceBuffer
from .hierarchy import (
    get_root_leaf_offsets,
    get_hog_member_prots,
    is_taxon_implied,
    get_children
)


# ----
# stats functions
@numba.njit
def binom_neglogccdf(x, n, p):
    '''
        Use rvlib to compute p-value
    '''
    return -1.0*Binomial(n, p).logccdf(x-1)


# ----
# family result sorting
@numba.njit
def fam_res_compare(x1, x2):
    '''
        Compare two family results and order them according to normcount, overlap and pvalue.
    '''
    # normalised count
    if x1['normcount'] != x2['normcount']:
        # greater first
        return -1 if (x1['normcount'] > x2['normcount']) else 1
    else:
        if x1['overlap'] != x2['overlap']:
            # greater first
            return -1 if (x1['overlap'] > x2['overlap']) else 1
        else:
            if x1['pvalue'] != x2['pvalue']:
                # greater first
                return -1 if (x1['pvalue'] > x2['pvalue']) else 1
    # equal. take whichever.
    return 0


@numba.njit
def family_result_argsort(x, ii):
    '''
        argsort of family results using defined comparison above.
        uses an implementation of quicksort.
        note: np.argsort DOES NOT support struct type in numba. this code does.
    '''
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

        return (bfs + pvs + afs)


@numba.njit
def family_result_sort(x):
    '''
        Sort the family results according to normcount, overlap and pvalue (to break ties).
        this uses a quicksort implementation as np.argsort does not support struct type in numba.
    '''
    # perform the family result sorting.
    idx = family_result_argsort(x, list(range(len(x))))
    y = np.zeros_like(x)
    for i in numba.prange(len(x)):
        y[i] = x[idx[i]]
    return y


# ----
## generic functions
@numba.njit
def get_fam_hog2parent(fam_ent, hog_tab):
    '''
    get HOG parent offsets of a single family
    '''
    hog_off = fam_ent['HOGoff']
    hog2parent_tmp = hog_tab['ParentOff'][hog_off:hog_off + fam_ent['HOGnum']]
    if hog2parent_tmp.size >1:
        return np.append(np.array([-1], dtype=np.int32), hog2parent_tmp[1:] - np.int32(hog_off))
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
    s, DIGITS_AA_LOOKUP, n_kmers, k, trans, x_flag):
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
        x_seen = np.any(s_norm[i:i+k] == DIGITS_AA_LOOKUP[88])  # 88==b'X'
        # if yes, replace it by the x_flag
        r[i] = r[i] if not x_seen else x_flag

    return custom_unique1d(r)


@numba.njit
def search_seq_kmers(
    r1, p1, hog_tab, fam_tab, x_flag, table_idx, table_buff):
    '''
        Perform the kmer search, using the index.
    '''
    hog_counts = np.zeros(hog_tab.size, dtype=np.uint16)
    fam_counts = np.zeros(fam_tab.size, dtype=np.uint16)

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

    return (hog_counts, fam_counts, fam_lowloc, fam_highloc)


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

'''
@numba.njit
def _propagate_counts(hog_count, hog2parent, levels, hog_offset):
    # propagate down
    res = hog_count.copy()
    # root already valid.
    for i in range(len(levels) - 1):
        x = levels[i:i+2]
        for j in range(x[0], x[1]):
            m = int(j-hog_offset)
            p = hog2parent[m]
            if p > -1:
                res[m] += res[int(p-hog_offset)]
            
    return res

@numba.njit(parallel=True, nogil=True)
def propagate_counts(hog_counts, fam_tab, hog2parent, levels):
    res = np.zeros_like(hog_counts)
    # parallel propagation
    for i in numba.prange(len(fam_tab)):
        entry = fam_tab[i]
        hog_s = entry["HOGoff"]
        hog_e = hog_s+entry["HOGnum"]
        level_s = int(entry["LevelOff"])
        level_e = int(level_s+entry["LevelNum"]+1)
        res[hog_s:hog_e] = _propagate_counts(hog_counts[hog_s:hog_e], hog2parent[hog_s:hog_e], levels[level_s:level_e], hog_s)
        
    return res


@numba.njit
def _cumulate_counts(hog_count, hog2parent, hog_offset):
    # NEW
    # propagate up by iterating backwards
    res = hog_count.copy()
    for i in range(len(hog2parent)-1, -1, -1):
        j = hog2parent[i]
        if j >= 0:
            res[j-hog_offset] += res[i]

    return res

@numba.njit(parallel=True, nogil=True)
def cumulate_counts(hog_counts, fam_tab, hog2parent):
    # NEW
    res = np.zeros_like(hog_counts)
    # parallel propagation
    for i in numba.prange(fam_tab.size):
        s = fam_tab["HOGoff"][i]
        e = s+fam_tab["HOGnum"][i]
        res[s:e] = _cumulate_counts(hog_counts[s:e], hog2parent[s:e], s)

    return res
'''

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
    fam_hog_cumcounts, query_nkmer, fam_level_offsets, fam_hog2parent, fam_hog_counts, fam_ref_hog_prob):
    # NEW USED TO PLACE ON PATH
    # TODO: switch this so that we compute and STOP going down the levels / place live.

    fam_hog_scores = np.zeros(fam_hog_cumcounts.shape, dtype=np.float64)
    fam_bestpath = np.full(fam_hog_cumcounts.shape, False)
    revcum_counts = np.full(fam_hog_cumcounts.shape, query_nkmer, dtype=np.uint16)

    # initialise root HOG
    fam_bestpath[0] = True
    expect_count = fam_ref_hog_prob[0] * query_nkmer
    fam_hog_scores[0] = (fam_hog_cumcounts[0] - expect_count) / query_nkmer

    # loop through hog levels
    # TODO: change this so that we only go down the greedy path!
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
        fam_hog_scores[hog_offsets] = (fam_hog_cumcounts[hog_offsets] - expect_count) / qh_count
        # store bestpath
        store_bestpath(hog_offsets, parent_offsets, fam_bestpath, fam_hog_scores)

        # also, if we have a reference taxon need to know when to STOP

    return fam_hog_scores, fam_bestpath


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
    def __init__(self, ki, include_extant_genes=False):
    	assert ki.db.db.mode == "r", "Database must be opened in read mode."

    	# load ki and db
    	self.db = ki.db
    	self.ki = ki

    	self.include_extant_genes = include_extant_genes

    @cached_property
    def trans(self):
        return get_transform(self.ki.k, self.ki.alphabet.DIGITS_AA)

    @cached_property
    def table_idx(self):
        return self.ki._table_idx[:]

    @cached_property
    def table_buff(self):
        return self.ki._table_buff[:]

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

    @lazy_property
    def ref_fam_prob(self):
        if 'FamilyProbability' in self.db.db.root.Index:
            return self.db.db.root.Index.FamilyProbability[:]
        else:
            raise ValueError('FamilyProbability not in db')

    @lazy_property
    def ref_hog_prob(self):
        if 'HOGProbability' in self.db.db.root.Index:
            return self.db.db.root.Index.HOGProbability[:]
        else:
            raise ValueError('HOGProbability not in db')

    def merge_search(self, seqs=None, ids=None, fasta_file=None,
        top_n_fams=10, alpha=0.05, sst=0.05, family_only=False):
        t1 = time()
        if seqs:
            sbuff = SequenceBuffer(seqs=seqs, ids=ids)
        elif fasta_file:
            sbuff = SequenceBuffer(fasta_file=fasta_file)

        t2 = time()
        # switch this to use logging
        print('{} second to load query sequences'.format(int(t2 - t1)))

        # load OMAmer database and tables into memory
        trans = self.trans
        table_idx = self.table_idx
        table_buff = self.table_buff
        fam_tab = self.fam_tab
        hog_tab = self.hog_tab
        level_arr = self.level_arr

        t4 = time()
        print('{} second to load the k-mer table'.format(int(t4 - t2)))

        # allocate result arrays
        family_results = np.zeros((len(sbuff.idx) - 1, top_n_fams),
                                  dtype=np.dtype([('id', np.uint32),
                                                  ('pvalue', np.float64),
                                                  ('count', np.uint32),
                                                  ('normcount', np.float64),
                                                  ('overlap', np.float64)]))
        subfam_results = np.zeros((len(sbuff.idx) - 1, top_n_fams),
                                  dtype=np.dtype([('id', np.uint32),
                                                  ('score', np.float64),
                                                  ('count', np.uint32)]))
        # perform the search 
        self._lookup(
            family_results,
            subfam_results,
            sbuff.buff,
            sbuff.idx,
            trans,
            table_idx,
            table_buff,
            self.ki.k,
            self.ki.alphabet.DIGITS_AA_LOOKUP,
            fam_tab,
            hog_tab,
            level_arr,
            top_n_fams=top_n_fams,
            ref_fam_prob=self.ref_fam_prob,
            ref_hog_prob=self.ref_hog_prob,
            alpha_cutoff=alpha,
            sst=sst,
            family_only=family_only)

        t5 = time()
        ts =  t5 - t4
        print('{} second for the actual search (~ {} query/second)'.format(int(ts), int(sbuff.prot_nr / ts)))

        return self.output_results(family_results, subfam_results, sbuff, top_n_fams)

    def output_results(self, family_results, subfam_results, sbuff, top_n_fams):
        #if ref_taxon:
        #    tax_tab = self.db._tax_tab[:]
        #    ref_taxoff = np.searchsorted(tax_tab['ID'], ref_taxon.encode('ascii'))
        #else:
        #    ref_taxoff = None

        # TODO: enable output of more than one result, probably by outputing all and then removing the duplicate NA,NA,NA lines (or keeping at least one if there are no other results?)

        # Note: missing values are dealt differently by pandas and numpy
        header = ['qseqid', 'hogid', 'overlap', 'family-score', 'subfamily-score', 'family-count', 'family-normcount', 'subfamily-count', 'qseqlen', 'subfamily-medianseqlen']
        def generate():
            for i in range(0, len(sbuff.idx)-1):
                for j in range(top_n_fams):
                    if (j == 0) or subfam_results['id'][i,j] > 0:
                        yield {'qseq_offset': i+1,
                               'hog_offset': subfam_results['id'][i,j],
                               'overlap': family_results['overlap'][i,j],
                               'family-score': family_results['pvalue'][i,j],
                               'subfamily-score': subfam_results['score'][i,j],
                               'family-count': family_results['count'][i,j],
                               'family-normcount': family_results['normcount'][i,j],
                               'subfamily-count': subfam_results['count'][i,j]}

        df = pd.DataFrame(generate())
        
        # so to pd dtype so that we can use pd.NA...
        df['qseq_offset'] = df['qseq_offset'].astype('UInt32')
        df['hog_offset'] = df['hog_offset'].astype('UInt32')
        df['family-count'] = df['family-count'].astype('UInt32')
        df['subfamily-count'] = df['subfamily-count'].astype('UInt32')

        # set empty as NA
        na_value = 0
        for k in df.keys():
            df.loc[df[k]==na_value, k] = pd.NA

        # TODO: update this within the SequenceBuffer (?)
        # set the query ids
        decode_if_necessary = lambda x: x.decode('ascii') if isinstance(x, bytes) else x
        df['qseqid'] = df['qseq_offset'].apply(lambda i: decode_if_necessary(sbuff.ids[i-1]))
        df['qseqlen'] = df['qseq_offset'].apply(lambda i: (sbuff.idx[i] - sbuff.idx[i-1]))

        # load the hog ids
        hog_f = df['hog_offset'].notna()
        df.loc[hog_f,'subfamily-medianseqlen'] = df.loc[hog_f,'hog_offset'].apply(lambda i: self.hog_tab['MedianSeqLen'][i-1]).astype('UInt32')
        if self.include_extant_genes:
            # add extant gene list if necessary
            chog_buff = self.db._chog_arr[:]
            cprot_buff = self.db._cprot_arr[:]
            prot_tab = self.db._prot_tab[:]

            header.append('subfamily-geneset')
            df.loc[hog_f, 'subfamily-geneset'] = df.loc[hog_f, 'hog_offset'].apply(lambda i: ','.join(map(lambda x: x.decode('ascii'), prot_tab['ID'][get_hog_member_prots(i-1, self.hog_tab, chog_buff, cprot_buff)])))

        # add the hog id
        df.loc[hog_f,'hogid'] = df.loc[hog_f,'hog_offset'].apply(lambda i: self.hog_tab['OmaID'][i-1].decode('ascii'))

        ## compute taxonomic congruences
        #if ref_taxon:
        #    q2closest_taxon = get_closest_taxa_from_ref(
        #        q2hog_off, ref_taxoff, tax_tab, self.hog_tab, self.db._chog_arr[:])
        #    c.insert(2, 'closetax')
        #    r.insert(2, map(lambda x: tax_tab['ID'][x].decode('ascii') if x != -1 else 'na', q2closest_taxon))
        
        return df[header]

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
            family_only
        ):
            '''
            top_n_fams: number of family for which HOG scores are computed
            '''
            # flags to ignore k-mers containing X
            x_flag = table_idx.size - 1

            for zz in numba.prange(len(seqs_idx) - 1):
                # load seq
                s = seqs[seqs_idx[zz] : np.int64(seqs_idx[zz + 1] - 1)]
                query_len = s.shape[0]
                n_kmers = (query_len - (k - 1))

                # double check we don't have short peptides (of len < k)
                if n_kmers == 0:
                    continue

                # seq -> bag of kmers
                (r1, p1, _) = parse_seq(s, DIGITS_AA_LOOKUP, n_kmers, k, trans, x_flag)

                # skip if only one k-mer with X
                if len(r1) > 1:
                    pass
                elif r1[0] == x_flag:
                    continue

                # search using kmers
                (hog_counts, fam_counts, fam_lowloc, fam_highloc) = search_seq_kmers(
                    r1, p1, hog_tab, fam_tab, x_flag, table_idx, table_buff)

                # identify families of interest. note: repeat(zeros_like..) numba hack
                idx = np.argwhere(fam_counts > 0).flatten()
                qres = np.repeat(np.zeros_like(family_results[zz,0]), len(idx))
                qres['id'][:] = idx
                qres['count'][:] = fam_counts[idx]

                # 1. compute p-value for each family. note: in negative log units
                correction_factor = np.log(len(ref_fam_prob))
                for i in numba.prange(len(qres)):
                    qres['pvalue'][i] = max(0.0, (binom_neglogccdf(qres['count'][i], len(r1), ref_fam_prob[qres['id'][i]]) - correction_factor))

                # 2. Filtering
                # - a. filter to significant families (on p-value)
                alpha = -1.0*np.log(alpha_cutoff)
                qres = qres[qres['pvalue'] >= alpha]
                if len(qres) == 0:
                    continue
                
                # - b. filter on sequence coverage
                qres['overlap'][:] = (fam_highloc[qres['id']] - fam_lowloc[qres['id']] + k) / query_len

                qres = qres[(qres['overlap'] >= (25/query_len))]
                if len(qres) == 0:
                    continue

                # 3. Compute normalised count
                # - a. compute the expected count
                top_fam_expect_counts = (ref_fam_prob[qres['id']] * len(r1))
                
                # - b. compute the normalised count
                qres['normcount'][:] = (
                        (qres['count'] - top_fam_expect_counts) /
                        (len(r1) - top_fam_expect_counts)
                        )

                # 4. Store results
                # - a. sort by normcount, then overlap, then p-value for tie-breaking
                qres = family_result_sort(qres)

                # - b. store results
                family_results['id'][zz, :top_n_fams] = qres['id'][:top_n_fams] + 1
                family_results['pvalue'][zz, :top_n_fams] = qres['pvalue'][:top_n_fams]
                family_results['count'][zz, :top_n_fams] = qres['count'][:top_n_fams]
                family_results['normcount'][zz, :top_n_fams] = qres['normcount'][:top_n_fams]
                family_results['overlap'][zz, :top_n_fams] = qres['overlap'][:top_n_fams]

                
                # 5. Place within families
                for i in numba.prange(min(len(qres), top_n_fams)):
                    entry = fam_tab[qres['id'][i]]
                    hog_s = entry["HOGoff"]
                    hog_e = hog_s+entry["HOGnum"]
                   
                    if family_only:
                        subfam_results['id'][zz,i] = hog_s + 1
                        continue

                    # TODO: work out whether we actually need to do this...
                    fam_hog2parent = get_fam_hog2parent(entry, hog_tab)
                    fam_level_offsets = get_fam_level_offsets(entry, level_arr)

                    # cumulation of counts
                    c = hog_counts[hog_s:hog_e].copy()
                    cumulate_counts_1fam(c, fam_level_offsets, fam_hog2parent, _sum, _max)

                    # new expected count, but using old cumulation
                    (fam_hog_scores, fam_bestpath) = hog_path_placement(
                            c, r1.size,
                            fam_level_offsets, fam_hog2parent, hog_counts[hog_s:hog_e], ref_hog_prob[hog_s:hog_e])

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
                    subfam_results['id'][zz,i] = choice + hog_s + 1
                    subfam_results['score'][zz,i] = choice_score
                    subfam_results['count'][zz,i] = c[int(choice)] if choice_score != 0.0 else 0

        return numba.jit(func, parallel=True, nopython=True, nogil=True)
