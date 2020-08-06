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
from property_manager import lazy_property
import numba
import numpy as np
import os
import pandas as pd
import sys

from ._utils import LOG
from .hierarchy import get_descendant_hogs, get_descendant_prots


class Search(object):
    def __init__(self, fs, include_extant_genes=False):
        assert fs.db.mode == "r", "Database must be opened in read mode."

        # load fs, ki and db
        self.db = fs.db
        self.ki = fs.ki
        self.fs = fs

        self.include_extant_genes = include_extant_genes

    #@lazy_property
    #def hog_cum_count(self):
    #    return self.cumulate_counts_nfams(
    #        self.ki._hog_count[:],
    #        self.db._fam_tab[:],
    #        self.db._level_arr[:],
    #        self.db._hog_tab.col("ParentOff"),
    #        self.cumulate_counts_1fam,
    #        self._sum,
    #        self._max,
    #    )

    #@lazy_property
    #def total_occur(self):
    #    return self.ki._idx_arr[:].size

    def search(
        self,
        norm_fam_fun,
        top_n,
        prop_fun,
        norm_hog_fun,
        threshold,
        norm_fam_args=[],
        norm_hog_args=[]):
        """
        args
         - norm_fam_fun: choice of function to normalize k-mer count of families
         - top_n: number of families to proceed with cumulation of HOG k-mer counts
         - prop_fun: function to decide how counts are cumulated (_max, _sum)
         - norm_hog_fun: choice of function to normalize k-mer count of HOGs

        additional args can be passed to norm_fam_fun and norm_hog_fun
        """
        top_n = min(top_n, len(self.db._fam_tab))
        #assert (
        #    top_n <= self.db._fam_tab.nrows
        #), "top n must be smaller than total number of families"

        # normalize at family level
        LOG.debug("compute family scores")
        queryFam_score = norm_fam_fun(*norm_fam_args)

        # resort families after normalization and keep top n
        fam_ranked = self.resort_ranked_fam(self.fs._fam_ranked[:], queryFam_score)[
            :, :top_n
        ]

        LOG.debug("propagate HOG counts bottom-up")
        queryHog_cum_count = self.cumulate_counts_nfams_nqueries(
            fam_ranked,
            self.fs._queryHog_count[:],
            self.db._fam_tab[:],
            self.db._level_arr[:],
            self.db._hog_tab.col("ParentOff"),
            len(self.fs._query_count),
            self.cumulate_counts_1fam,
            self._sum,
            self._max,
        )

        LOG.debug("compute HOG scores")
        queryHog_score, queryHog_bestpath = norm_hog_fun(
            fam_ranked, queryHog_cum_count, *norm_hog_args
        )

        LOG.debug("form results table")
        # get the results -- do something a bit stupid here to begin with...

        # - queryHog_score is matrix of scores to output
        # - bestpath_mask contains a mask of this? 
        def get_prot_ids(h):
            desc_hogs = list(get_descendant_hogs(h, self.db._hog_tab, self.db._chog_arr)) + [h]
            prot_ii = get_descendant_prots(desc_hogs, self.db._hog_tab, self.db._cprot_arr)
            return self.db._prot_tab.read_coordinates(prot_ii, 'ID')
        
        qseqid = self.fs._query_id[:].flatten()  # these will come from FASTA file
        family = fam_ranked[:, 0]                # family offset id
        family_score = queryFam_score[np.arange(len(qseqid)), family] 

        # generate the dataframe
        def generate_results(threshold):
            for i in np.argwhere(~np.isnan(family_score)).flatten():
                # find best scoring subfamily
                best_j = None
                best_s = np.inf
                for j in np.argwhere(queryHog_bestpath[i]).flatten():
                    s = queryHog_score[i,j]
                    if s < best_s and s >= threshold:
                        best_j = j
                        best_s = s
                if best_j is not None:
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

        return pd.DataFrame(generate_results(threshold))[h]

    ### normalize family ####################################################################################################################
    def norm_fam_query_size(self):
        """
        simple division by the length of query
        """
        # no need to broadcast because query_count is a vertical vector
        # query_count_bc = np.full((self.fs._queryFam_count[:].shape[1], self.fs._queryFam_count[:].shape[0]), self.fs._query_count[:]).T

        return self.fs._queryFam_count[:] / self.fs._query_count[:]

    ### sort families #######################################################################################################################
    @staticmethod
    def resort_ranked_fam(fam_ranked, queryFam_score, top_n=None):

        # ranked scores following fam_ranked until top_n
        ranked_score = queryFam_score[
            np.arange(fam_ranked.shape[0])[:, None], fam_ranked[:, :top_n]
        ]

        # get indices of the reverse sorted ranked scores
        idx = (-ranked_score).argsort()

        # back to the fam offsets by indexing fam_ranked with our indices
        return fam_ranked[np.arange(fam_ranked.shape[0])[:, None], idx]

    ### cumulate counts #####################################################################################################################
    @staticmethod
    @numba.njit
    def _max(x, y):
        return max(x, y)

    @staticmethod
    @numba.njit
    def _sum(x, y):
        return x + y

    @staticmethod
    @numba.njit
    def cumulate_counts_1fam(
        hog_cum_counts, fam_level_offsets, hog2parent, cum_fun, prop_fun
    ):

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

    @staticmethod
    @numba.njit(parallel=True, nogil=True)
    def cumulate_counts_nfams(
        hog_counts, fam_tab, level_arr, hog2parent, main_fun, cum_fun, prop_fun
    ):

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

    @staticmethod
    @numba.njit(parallel=True, nogil=True)
    def cumulate_counts_nfams_nqueries(
        fam_results,
        hog_counts,
        fam_tab,
        level_arr,
        hog2parent,
        q_num,
        main_fun,
        cum_fun,
        prop_fun,
    ):

        hog_cum_counts = hog_counts.copy()  # probably not necessary

        # iterate queries
        for q in numba.prange(q_num):

            # iterate families
            for fam_off in fam_results[q]:

                entry = fam_tab[fam_off]
                level_off = entry["LevelOff"]
                level_num = entry["LevelNum"]
                fam_level_offsets = level_arr[
                    level_off : np.int64(level_off + level_num + 2)
                ]
                main_fun(
                    hog_cum_counts[q], fam_level_offsets, hog2parent, cum_fun, prop_fun
                )

        return hog_cum_counts

    ### normalize hogs ######################################################################################################################
    def norm_hog_query_size(self, fam_ranked, queryHog_cum_count):
        """
        same but normalize first the queryHog_cum_count
        could be slightly more clever and remove part conserved in parent 
        """
        # query_count_bc = np.full((queryHog_cum_count.shape[1], queryHog_cum_count.shape[0]), self.fs._query_count[:]).T

        queryHog_score = queryHog_cum_count / self.fs._query_count[:]

        queryHog_bestpath = self.compute_bestpath_nqueries_nfams(
            len(self.fs._query_count),
            fam_ranked,
            queryHog_score,
            self.db._fam_tab[:],
            self.db._level_arr[:],
            self.db._hog_tab.col("ParentOff"),
        )

        return queryHog_score, queryHog_bestpath

    @staticmethod
    def store_bestpath(
        hog_offsets, parent_offsets, queryHog_bestpath, queryHog_score, pv=False
    ):

        # keep HOGs descending from the best path
        cands = hog_offsets[queryHog_bestpath[parent_offsets]]

        # get the score of these candidates
        cands_scores = queryHog_score[cands]

        # find the candidate HOGs offsets with the higher count (>0) at this level
        if cands_scores.size > 0:

            # need smallest
            if pv:
                cands_offsets = np.where(cands_scores == np.min(cands_scores))[0]
            # need > 0 and max
            else:
                cands_offsets = np.where(
                    (cands_scores > 0) & (cands_scores == np.max(cands_scores))
                )[0]

            # if a single candidate, update the best path. Else, stop because of tie
            if cands_offsets.size == 1:
                queryHog_bestpath[cands[cands_offsets]] = True

    def compute_bestpath_nqueries_nfams(
        self, q_num, fam_ranked, queryHog_cum_count, fam_tab, level_arr, hog2parent
    ):
        """
        this one is to compute the best path w/o computing the score simultaneously
        """
        queryHog_bestpath = np.full(queryHog_cum_count.shape, False)

        # loop through queries and families
        for q in range(q_num):
            for f in fam_ranked[q]:

                entry = fam_tab[f]
                level_off = entry["LevelOff"]
                level_num = entry["LevelNum"]
                fam_level_offsets = level_arr[
                    level_off : np.int64(level_off + level_num + 2)
                ]

                # use the cumulative counts to define the best path
                rh = fam_level_offsets[0]
                qh_score = queryHog_cum_count[q][rh]
                if qh_score > 0:
                    queryHog_bestpath[q][rh] = True

                # loop through hog levels
                for i in range(1, fam_level_offsets.size - 2):
                    x = fam_level_offsets[i : i + 2]
                    hog_offsets = np.arange(x[0], x[1])

                    # grab parents
                    parent_offsets = hog2parent[hog_offsets]

                    # store best path
                    self.store_bestpath(
                        hog_offsets,
                        parent_offsets,
                        queryHog_bestpath[q],
                        queryHog_cum_count[q],
                    )

        return queryHog_bestpath
