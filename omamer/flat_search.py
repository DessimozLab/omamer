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
import os
import sys
import tables

from ._utils import LOG
from .index import get_transform, SequenceBuffer, QuerySequenceBuffer


class FlatSearch(object):
    def __init__(self, ki, nthreads=None, low_mem=False):

        assert ki.db.mode == "r", "Database must be opened in read mode."

        # load ki and db
        self.db = ki.db
        self.ki = ki

        # performance features
        self.nthreads = nthreads if nthreads is not None else os.cpu_count()
        self.low_mem = low_mem

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.fs.close()

    # cached properties from Index
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
    def prot2hog(self):
        return self.db._prot_tab.col("HOGoff")

    @cached_property
    def hog2fam(self):
        return self.db._hog_tab.col("FamOff")
    
    # flat search function
    def flat_search(self, seqs=None, ids=None, fasta_file=None, query_sp=None):
        """
        point is to limit code duplication in FlatSearchValidation 
        """
        # load query sequences
        if seqs:
            sb = SequenceBuffer(seqs=seqs, ids=ids)
        elif fasta_file:
            sb = SequenceBuffer(fasta_file=fasta_file)
        elif query_sp:
            self.query_species = "".join(query_sp.split())
            sb = QuerySequenceBuffer(db=self.db, query_sp=self.query_species)

        return self._flat_search(sb)

    def _flat_search(self, sbuff):
        LOG.debug("look-up k-mer table")
        # fam_ranked, queryFam_count, queryFam_occur, queryHog_count, queryHog_occur, query_count, query_occur = self.lookup(sbuff)
        (
            fam_ranked,
            queryFam_count,
            queryFam_occur,
            queryHog_count,
            queryHog_occur,
            query_count,
            query_occur,
        ) = self._lookup(
            sbuff.buff,
            sbuff.idx,
            self.trans,
            self.table_idx,
            self.table_buff,
            self.db._hog_tab.col("FamOff"),
            self.db._fam_tab.nrows,
            self.db._hog_tab.nrows,
            self.ki.k,
            self.ki.alphabet.DIGITS_AA_LOOKUP,
        )

        self._fam_ranked = fam_ranked
        self._queryFam_count = queryFam_count
        self._queryFam_occur = queryFam_occur
        self._queryHog_count = queryHog_count
        self._queryHog_occur = queryHog_occur

        # store as vertical vectors
        self._query_count = query_count[:, None]
        self._query_occur = query_occur[:, None]

        # store ids of sbuff
        self._query_id = sbuff.ids.flatten()

    @lazy_property
    def _lookup(self):
        def func(
            seqs,
            seqs_idx,
            trans,
            table_idx,
            table_buff,
            hog2fam,
            n_fams,
            n_hogs,
            k,
            DIGITS_AA_LOOKUP,
        ):

            fam_results = np.zeros((len(seqs_idx) - 1, n_fams), dtype=np.uint32)
            fam_counts = np.zeros((len(seqs_idx) - 1, n_fams), dtype=np.uint16)
            fam_occur = np.zeros((len(seqs_idx) - 1, n_fams), dtype=np.uint32)  # ()
            hog_counts = np.zeros((len(seqs_idx) - 1, n_hogs), dtype=np.uint16)
            hog_occur = np.zeros((len(seqs_idx) - 1, n_hogs), dtype=np.uint32)  # ()
            query_counts = np.zeros((len(seqs_idx) - 1), dtype=np.uint16)  # ()
            query_occur = np.zeros((len(seqs_idx) - 1), dtype=np.uint32)  # ()

            x_char = DIGITS_AA_LOOKUP[88]  # 88 == b'X'
            x_kmer = 0
            for j in range(k):
                x_kmer += trans[j] * x_char
            x_kmer += 1

            for zz in numba.prange(len(seqs_idx) - 1):
                # grab query sequence
                s = seqs[seqs_idx[zz] : int(seqs_idx[zz + 1] - 1)]
                n_kmers = s.shape[0] - (k - 1)

                # double check we don't have short peptides
                # note: written in this order to provide loop-optimisation hint
                if n_kmers > 0:
                    pass
                else:
                    continue

                # parse into k-mers
                # TO DO: get locations simultaneously
                s_norm = DIGITS_AA_LOOKUP[s]

                r = np.zeros(n_kmers, dtype=np.uint32)  # max kmer 7
                for i in numba.prange(n_kmers):
                    # numba can't do np.dot with non-float
                    for j in range(k):
                        r[i] += trans[j] * s_norm[i + j]

                    x_seen = np.any(s_norm[i:i+k] == x_char)
                    r[i] = r[i] if not x_seen else x_kmer
                r1 = np.unique(r)

                if len(r1) > 1:
                    pass
                elif r1[0] == x_kmer:
                    continue

                # search hogs and fams
                hog_res = np.zeros(n_hogs, dtype=np.uint16)
                fam_res = np.zeros(n_fams, dtype=np.uint16)
                hog_occ = np.zeros(n_hogs, dtype=np.uint32)
                fam_occ = np.zeros(n_fams, dtype=np.uint32)
                query_occ = 0
                for m in numba.prange(r1.shape[0]):
                    kmer = r1[m]
                    if kmer < x_kmer:
                        pass
                    else:
                        continue
                    x = table_idx[kmer : kmer + 2]
                    hogs = table_buff[x[0] : x[1]]
                    fams = hog2fam[hogs]
                    hog_res[hogs] += np.uint16(1)
                    fam_res[fams] += np.uint16(1)
                    # kmer_occ is the nr of hogs/fams with the given kmer, used to compute its frequency
                    kmer_occ = (
                        hogs.size
                    )  # the built-in function len made a type error only when parallel was ON
                    # store for the set of query k-mers
                    query_occ += kmer_occ
                    # and for the set of intersecting k-mers between the query and every hogs and fams
                    hog_occ[hogs] += np.uint32(kmer_occ)
                    fam_occ[fams] += np.uint32(
                        kmer_occ
                    )  # I think here it should be len(fams) instead of len(hogs) ... # actually fine because one hog for one fam...

                # report results for families sorted by k-mer count
                t = np.argsort(fam_res)[::-1]
                fam_results[zz, :n_fams] = t
                # not sorting anymore the corresponding fam counts and occur
                fam_counts[zz, :n_fams] = fam_res
                fam_occur[zz, :n_fams] = fam_occ
                # fam_counts[zz, :n_fams] = fam_res[t]
                # fam_occur[zz, :n_fams] = fam_occ[t]
                # report raw results for hogs
                hog_counts[zz, :n_hogs] = hog_res
                hog_occur[zz, :n_hogs] = hog_occ
                # report results for the query
                query_counts[zz] = len(r1)
                query_occur[zz] = query_occ

            return (
                fam_results,
                fam_counts,
                fam_occur,
                hog_counts,
                hog_occur,
                query_counts,
                query_occur,
            )

        if not self.low_mem:
            # Set nthreads, note: this only works before numba called first time!
            numba.config.NUMBA_NUM_THREADS = self.nthreads
            return numba.jit(func, parallel=True, nopython=True, nogil=True)
        else:
            return func
