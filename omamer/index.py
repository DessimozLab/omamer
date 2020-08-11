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
from Bio import SeqIO, SearchIO
from ete3 import Tree
from multiprocessing import sharedctypes
from property_manager import lazy_property, cached_property
from PySAIS import sais
import ctypes
import gzip
import logging 
import multiprocessing as mp
import numba
import numpy as np
import os
import tables

from ._utils import LOG
from .alphabets import Alphabet
from .hierarchy import get_lca_hog_off, get_descendant_species, get_descendant_taxa


@numba.njit
def get_transform(k, DIGITS_AA):
    t = np.zeros(k, dtype=np.uint64)
    for i in numba.prange(k):
        t[i] = len(DIGITS_AA) ** (k - (i + 1))
    return t


class Index(object):
    def __init__(
        self, db, k=6, reduced_alphabet=False, nthreads=1, hidden_taxa=[]
    ):
        
        # load database object
        self.db = db

        # load k, alphabet size and hidden taxa
        if '/Index' in self.db.db:
            self.k = self.db.db.root.Index._v_attrs['k']
            alphabet_n = self.db.db.root.Index._v_attrs['alphabet_n']
            self.hidden_taxa = self.db.db.root.Index._v_attrs['hidden_taxa']
        else:
            self.k = k
            alphabet_n = 21 if not reduced_alphabet else 13
            self.hidden_taxa = hidden_taxa

        self.alphabet = Alphabet(n=alphabet_n)

        # performance features
        self.nthreads = nthreads

    ### useful for validation
    def set_species(self, sp_ii):
        self.sp_filter = np.full((len(self.db._sp_tab),), True)
        #self.sp_filter[:] = True
        for i in sp_ii:
            self.sp_filter[i] = False

    @property
    def sp_filter(self):
        sp_filter = np.full((len(self.db._sp_tab),), False)
        for hidden_taxon in self.hidden_taxa:
            descendant_species = get_descendant_species(
                np.searchsorted(self.db._tax_tab.col('ID'), hidden_taxon.encode('ascii')), self.db._tax_tab, self.db._ctax_arr)
            # case where hidden taxon is species
            if descendant_species.size == 0:
                sp_filter[np.searchsorted(self.db._sp_tab.col('ID'), hidden_taxon.encode('ascii'))] = True
            else:
                for sp_off in descendant_species:
                    sp_filter[sp_off] = True
        return sp_filter

    @property
    def tax_filter(self):
        tax_filter = np.full((len(self.db._tax_tab),), False)
        for hidden_taxon in self.hidden_taxa:
            htax_off = np.searchsorted(self.db._tax_tab.col('ID'), hidden_taxon.encode('ascii'))
            tax_filter[htax_off] = True
            descendant_taxa = get_descendant_taxa(htax_off, self.db._tax_tab, self.db._ctax_arr)
            for htax_off in descendant_taxa:
                tax_filter[htax_off] = True
        return tax_filter

    ### same as in database class; easy access to data ###
    def _get_node_if_exist(self, node):
        if node in self.db.db:
            return self.db.db.get_node(node)

    @property
    def _table_idx(self):
        return self._get_node_if_exist('/Index/TableIndex')

    @property
    def _table_buff(self):
        return self._get_node_if_exist('/Index/TableBuffer')

    @property
    def _fam_count(self):
        return self._get_node_if_exist('/Index/FamCount')

    @property
    def _hog_count(self):
        return self._get_node_if_exist('/Index/HOGcount')

    ### main function to build the index ###
    def build_kmer_table(self):

        #assert self.db.mode in {"w", "a"}, "Index must be opened in write mode."
        assert self.db.mode == 'w', "Index must be opened in write mode."
        assert ('/Index' not in self.db.db), 'Index has already been computed'

        # build suffix array with option to translate the sequence buffer first
        sa = self._build_suffixarray(
            self.alphabet.translate(self.db._seq_buff[:]), len(self.db._prot_tab)
        )

        # Set nthreads, note: this only works before numba called first time!
        numba.config.NUMBA_NUM_THREADS = self.nthreads

        self._build_kmer_table(sa)

    @staticmethod
    def _build_suffixarray(seqs, n):
        # Build suffix array
        sa = sais(seqs)
        sa[:n].sort()  # Sort delimiters by position
        return sa

    def _build_kmer_table(self, sa):
        #@numba.njit(parallel=True) --> was breaking for PANTHER database
        @numba.njit
        def _compute_mask_and_filter(
            sa, sa_mask, sa_filter, k, n, prot2spoff, prot2hogoff, sp_filter
        ):
            """
            1. compute a mapper between suffixes and HOGs
            2. simultaneously, compute suffix filter for species and suffixe < k
            """
            for i in numba.prange(n):
                # leverages the sorted protein delimiters at the beggining of the sa to get the suffix offsets by protein
                s = (sa[i - 1] if i > 0 else -1) + 1
                e = sa[i] + 1
                sa_mask[s:e] = prot2hogoff[i]
                if sp_filter[prot2spoff[i]]:
                    sa_filter[s:e] = True
                else:
                    sa_filter[(e - k) : e] = True

        @numba.njit
        def _same_kmer(seq_buff, sa, kmer, jj, k):
            kmer_jj = seq_buff[sa[jj] : (sa[jj] + k)].view(np.uint8)
            return np.all(np.equal(kmer, kmer_jj))

        @numba.njit
        def _compute_many_lca_hogs(hog_offsets, fam_offsets, hog_parents):
            """
            compute lca hogs for a list of hogs and their families
            """
            # as many lca hogs as unique families
            lca_hogs = np.zeros((np.unique(fam_offsets).size,), dtype=np.int64)

            # keep track of family, the corresponding hogs and the lca hog offset
            curr_fam = fam_offsets[0]
            curr_hogs = list(
                hog_offsets[0:1]
            )  # set the type of items in list to integers
            lca_off = 0

            for i in range(1, len(hog_offsets)):
                fam = fam_offsets[i]
                # wait to have all hogs of the family between computing the lca hog
                if fam == curr_fam:
                    curr_hogs.append(hog_offsets[i])
                else:
                    lca_hogs[lca_off] = get_lca_hog_off(curr_hogs, hog_parents)
                    curr_hogs = list(hog_offsets[i : i + 1])
                    curr_fam = fam
                    lca_off += 1

            # last family
            lca_hogs[lca_off] = get_lca_hog_off(curr_hogs, hog_parents)
            return lca_hogs

        @numba.njit
        def _compute_kmer_table(
            sa,
            seq_buff,
            sa_mask,
            hog_fams,
            hog_parents,
            table_idx,
            table_buff,
            fam_kmer_counts,
            hog_kmer_counts,
            k,
            DIGITS_AA,
            DIGITS_AA_LOOKUP,
        ):

            ii = 0  # pointer to sa offset of kk
            kk = 0  # pointer to k-mer (offset in table_idx)
            ii_table_buff = 0  # pointer to offset in table_buff (~rows of k-mer table)
            trans = get_transform(k, DIGITS_AA)
            while ii < len(sa):
                ## compute the new k-mer (kk1)
                kmer = seq_buff[sa[ii] : (sa[ii] + k)].view(np.uint8)
                kk1 = 0
                for i in range(k):
                    kk1 += int(DIGITS_AA_LOOKUP[kmer[i]] * trans[i])

                ## find offset of new k-mer in sa (jj)
                # first in windows of 100s
                jj = min(ii + 100, len(sa))
                while (jj < len(sa)) and _same_kmer(seq_buff, sa, kmer, jj, k):
                    jj = min(jj + 100, len(sa))

                # then, refine with binary search
                lo = max(ii, (jj - 100) + 1)
                hi = jj
                while lo < hi:
                    m = int(np.floor((lo + hi) / 2))
                    if _same_kmer(seq_buff, sa, kmer, m, k):
                        lo = m + 1
                    else:
                        hi = m
                jj = lo

                ## compute LCA HOGs containing current k-mer (kk)
                # get the hog offsets for each suffix containing kk
                hog_offsets = np.unique(sa_mask[ii:jj])

                # get the corresponding fam offsets
                fam_offsets = hog_fams[hog_offsets]

                # compute the LCA hog offsets
                lca_hog_offsets = _compute_many_lca_hogs(
                    hog_offsets, fam_offsets, hog_parents
                )

                # store kmer counts of fams and lca hogs
                fam_kmer_counts[np.unique(fam_offsets)] = (
                    fam_kmer_counts[np.unique(fam_offsets)] + 1
                )
                hog_kmer_counts[lca_hog_offsets] = hog_kmer_counts[lca_hog_offsets] + 1

                ## store LCA HOGs in table buffer
                nr_lca_hog_offsets = len(lca_hog_offsets)
                table_buff[
                    ii_table_buff : ii_table_buff + nr_lca_hog_offsets
                ] = lca_hog_offsets

                ## store buffer offset in table index at offset corresponding to k-mer integer encoding
                table_idx[kk : kk1 + 1] = ii_table_buff

                ## find buffer offset of new k-mer in table index
                ii_table_buff += nr_lca_hog_offsets
                ii = jj
                kk = kk1 + 1

            # fill until the end
            table_idx[kk:] = ii_table_buff
            return ii_table_buff

        LOG.debug(" - filter suffix array and compute its HOG mask")
        n = len(self.db._prot_tab)
        sa_mask = np.zeros(sa.shape, dtype=np.uint64)
        sa_filter = np.zeros(sa.shape, dtype=np.bool)

        _compute_mask_and_filter(
            sa,
            sa_mask,
            sa_filter,
            self.k,
            n,
            self.db._prot_tab.col("SpeOff"),
            self.db._prot_tab.col("HOGoff"),
            self.sp_filter,
        )

        # before filtering the sa, reorder and reverse the suffix filter
        sa = sa[~sa_filter[sa]]

        # filter and reorder the mask according to this filtered sa
        sa_mask = sa_mask[sa]

        LOG.debug(" - compute k-mer table")
        table_idx = np.zeros(
            (len(self.alphabet.DIGITS_AA) ** self.k + 1), dtype=np.uint64
        )

        # initiate buffer of size sa_mask, which is maximum size if all suffixes are from different HOGs
        table_buff = np.zeros((len(sa_mask)), dtype=np.uint64)

        fam_kmer_counts = np.zeros(len(self.db._fam_tab), dtype=np.uint64)
        hog_kmer_counts = np.zeros(len(self.db._hog_tab), dtype=np.uint64)

        ii_table_buff = _compute_kmer_table(
            sa,
            self.db._seq_buff[:],
            sa_mask,
            self.db._hog_tab.col("FamOff"),
            self.db._hog_tab.col("ParentOff"),
            table_idx,
            table_buff,
            fam_kmer_counts,
            hog_kmer_counts,
            self.k,
            self.alphabet.DIGITS_AA,
            self.alphabet.DIGITS_AA_LOOKUP,
        )

        # remove extra space
        table_buff = table_buff[:ii_table_buff]

        LOG.debug(" - write k-mer table")
        idx = self.db.db.create_group('/', 'Index', 'hog indexes')
        idx._f_setattr('k', self.k)
        idx._f_setattr('alphabet_n', self.alphabet.n)
        idx._f_setattr('hidden_taxa', self.hidden_taxa)
        self.db.db.create_carray(idx, "TableIndex", obj=table_idx, filters=self.db._compr)
        self.db.db.create_carray(
            idx, "TableBuffer", obj=table_buff, filters=self.db._compr
        )
        self.db.db.create_carray(
            idx, "FamCount", obj=fam_kmer_counts, filters=self.db._compr
        )  # for these, I can initialize them before...
        self.db.db.create_carray(
            idx, "HOGcount", obj=hog_kmer_counts, filters=self.db._compr
        )


class SequenceBuffer(object):
    """
    to load sequences from db or files
    adapted from alex
    """

    def __init__(self, seqs=None, ids=None, db=None, fasta_file=None):
        self.prot_off = 0
        if seqs is not None:
            self.add_seqs(*seqs)
            self.ids = np.array(ids) if ids else np.array(range(len(seqs)))
        elif db is not None:
            self._prot_tab = db._prot_tab
            self._seq_buff = db._seq_buff
            self.load_from_db()
        elif fasta_file is not None:
            seqs = self.parse_fasta(fasta_file)
            self.add_seqs(*seqs)
        else:
            raise ValueError(
                "need to pass array of seqs or db to load from to SequenceBuffer"
            )

    def __getstate__():
        return (self.prot_nr, self.n, self.buff_shr, self.buff_idx_shr)

    def __setstate__(state):
        (self.prot_nr, self.n, self.buff_shr, self.buff_idx_shr) = state

    def parse_fasta(self, fasta_file):
        ids = []
        seqs = []
        func = open if not fasta_file.endswith('.gz') else gzip.open
        with func(fasta_file, 'rt') as fp:
            for rec in SeqIO.parse(fp, "fasta"):
                ids.append(rec.id)
                seqs.append(str(rec.seq))
        self.ids = np.array(ids)
        return seqs

    def add_seqs(self, *seqs):
        self.prot_nr = len(seqs)
        self.n = self.prot_nr + sum(len(s) for s in seqs)

        self.buff_shr = sharedctypes.RawArray(ctypes.c_uint8, self.n)
        self.buff[:] = np.frombuffer(
            (" ".join(seqs) + " ").encode("ascii"), dtype=np.uint8
        )

        self.buff_idx_shr = sharedctypes.RawArray(ctypes.c_uint64, self.prot_nr + 1)
        for i in range(len(seqs)):
            self.idx[i + 1] = len(seqs[i]) + 1 + self.idx[i]

    def load_from_db(self):
        self.prot_nr = len(self._prot_tab)
        self.n = len(self._seq_buff)

        self.buff_shr = sharedctypes.RawArray(ctypes.c_uint8, self.n)
        self.buff[:] = self._seq_buff[:].view(np.uint8)

        self.buff_idx_shr = sharedctypes.RawArray(ctypes.c_uint64, self.prot_nr + 1)
        # self.idx[:-1] = self._prot_tab.cols.SeqOff[:]
        self.idx[:-1] = self._prot_tab[:]["SeqOff"]
        self.idx[-1] = self.n

        # store offsets as ids
        self.ids = np.arange(
            self.prot_off, self.prot_off + self.prot_nr, dtype=np.uint64
        )[:, None]

    @lazy_property
    def buff(self):
        return np.frombuffer(self.buff_shr, dtype=np.uint8).reshape(self.n)

    @lazy_property
    def idx(self):
        return np.frombuffer(self.buff_idx_shr, dtype=np.uint64).reshape(
            self.prot_nr + 1
        )

    # @lazy_property
    # def prot_offsets(self):
    #     return np.arange(self.prot_off, self.prot_off + self.prot_nr, dtype=np.uint64)

    def __getitem__(self, i):
        s = int(self.idx[i])
        e = int(self.idx[i + 1] - 1)
        return self.buff[s:e].tobytes().decode("ascii")


class QuerySequenceBuffer(SequenceBuffer):
    """
    get a sliced version of the sequence buffer object for a given species in db
    TO DO: put an __super__
    """

    def __init__(self, db, query_sp):
        self.query_sp = (
            query_sp if isinstance(query_sp, bytes) else query_sp.encode("ascii")
        )
        self.set_query_prot_tab(db)
        self.filter_query_seq_buff(db)
        self.load_from_db()

    def set_query_prot_tab(self, db):
        sp_off = np.searchsorted(db._sp_tab.col("ID"), self.query_sp)
        sp_ent = db._sp_tab[sp_off]
        self.prot_off = sp_ent["ProtOff"]
        self._prot_tab = db._prot_tab[self.prot_off : self.prot_off + sp_ent["ProtNum"]]

    def filter_query_seq_buff(self, db):
        self._seq_buff = db._seq_buff[
            self._prot_tab[0]["SeqOff"] : self._prot_tab[-1]["SeqOff"]
            + self._prot_tab[-1]["SeqLen"]
        ]
        # initialize sequence buffer offset
        self._prot_tab["SeqOff"] -= db._prot_tab[self.prot_off]["SeqOff"]
