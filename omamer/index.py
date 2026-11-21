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
from property_manager import lazy_property
import numba
import numpy as np

from ._utils import LOG
from .alphabets import Alphabet, get_transform
from .hierarchy import get_lca_off, get_leaves
from .merge_search import cumulate_counts_1fam


class Index(object):
    def __init__(
        self,
        db,
        k=6,
        reduced_alphabet=False,
        hidden_taxa=[],
        kmer_max_df=0,
        ss_kmer_max_df=0,
        seq_min_distinct=None,
        ss_min_distinct=None,
    ):
        # load database object
        self.db = db

        # load k, alphabet size and hidden taxa
        if "/Index" in self.db.db:
            self.k = self.db.db.root.Index._v_attrs["k"]
            alphabet_n = self.db.db.root.Index._v_attrs["alphabet_n"]
            self.hidden_taxa = self.db.db.root.Index._v_attrs["hidden_taxa"]
        else:
            self.k = k
            alphabet_n = 21 if not reduced_alphabet else 13
            self.hidden_taxa = hidden_taxa
            if kmer_max_df < 0 or ss_kmer_max_df < 0:
                raise ValueError("k-mer document-frequency caps must be >= 0")
            self.kmer_max_df = int(kmer_max_df)
            self.ss_kmer_max_df = int(ss_kmer_max_df)
            # low-complexity k-mer mask of each modality, stored for search
            self.min_distinct = {"seq": seq_min_distinct, "ss": ss_min_distinct}

        self.alphabet = Alphabet(n=alphabet_n)

    @lazy_property
    def sp_filter(self):
        sp_filter = np.full((len(self.db._db_Species),), False)
        if len(self.hidden_taxa) > 0:
            tax_tab = self.db._db_Taxonomy[:]
            child_tax = self.db._db_ChildrenTax[:]

            if len(self.hidden_taxa) > 0:
                LOG.debug(' - creating species filter to hide declared taxa')

            for hidden_taxon in self.hidden_taxa:
                LOG.debug('   - identifying: {}'.format(hidden_taxon))
                taxon = np.argwhere(tax_tab["ID"] == hidden_taxon.encode("ascii")).flatten()
                if len(taxon) == 0:
                    raise ValueError("Can't find {} in taxonomy.".format(hidden_taxon))
                elif len(taxon) > 1:
                    raise ValueError("Ambiguous taxon {}.".format(hidden_taxon))

                tax_ii = taxon[0]
                sp_ii = tax_tab[tax_ii]['SpeOff']

                if sp_ii >= 0:
                    # leaf (i.e., extant species listed)
                    LOG.debug('     - hiding {}'.format(self.db._db_Species[sp_ii]['ID'].decode('ascii')))
                    sp_filter[sp_ii] = True
                else:
                    # filter all leaves below declared taxon
                    for sp_jj in tax_tab["SpeOff"][get_leaves(tax_ii, tax_tab, child_tax)]:
                        LOG.debug('     - hiding {}'.format(self.db._db_Species[sp_jj]['ID'].decode('ascii')))
                        sp_filter[sp_jj] = True
        return sp_filter

    @property
    def kmer_table(self):
        return {
            "buff": self.db._db_Index_TableBuffer,
            "idx": self.db._db_Index_TableIndex,
        }

    ### main function to build the index ###
    def build_kmer_table(self, seq_buff, ss_buff=None):
        """
        Build the k-mer table of the sequences and, if ss_buff is given, the
        one of the 3Di sequences. Record i of both buffers is protein i.
        """
        assert self.db.mode in {"w", "a"}, "Index must be opened in write mode."
        assert "/Index" not in self.db.db, "Index has already been computed"

        idx = self.db.db.create_group("/", "Index", "hog indexes")
        idx._f_setattr("k", self.k)
        idx._f_setattr("alphabet_n", self.alphabet.n)
        idx._f_setattr("hidden_taxa", self.hidden_taxa)

        for modality, buff in (("seq", seq_buff), ("ss", ss_buff)):
            if buff is None:
                continue
            # build suffix array with option to translate the sequence buffer first
            translated = self.alphabet.translate(buff)
            sa = self._build_suffixarray(translated, len(self.db._db_Protein))
            self._build_kmer_table(buff, sa, translated, modality=modality)

    @staticmethod
    def _build_suffixarray(seqs, n):
        # Build suffix array
        LOG.debug(" - building suffix array for sequences")

        from PySAIS import sais
        # import sais here, otherwise we need it for search-time dependency

        sa = sais(seqs)
        sa[:n].sort()  # Sort delimiters by position
        return sa

    def _build_kmer_table(self, seq_buff, sa, translated=None, modality="seq"):
        """
        Build the k-mer table from the suffix array of the translated sequence
        buffer. The k-mer codes are computed from the original sequences, the
        suffixes of a k-mer are found by comparing the translated ones, as with
        a reduced alphabet different k-mers of the original sequences belong
        to the same k-mer.

        modality is "seq" (sequences) or "ss" (3Di sequences, stored with the
        prefix SS). The family and HOG probabilities ignore the k-mers found
        in more families than the document-frequency cap of the modality.
        """
        if translated is None:
            translated = self.alphabet.translate(seq_buff)
        max_df = self.kmer_max_df if modality == "seq" else self.ss_kmer_max_df

        @numba.njit(parallel=True, nogil=True)
        def _compute_mask_and_filter(
            sa, sa_mask, sa_filter, k, n, prot2spoff, prot2hogoff, sp_filter
        ):
            """
            1. compute a mapper between suffixes and HOGs
            2. simultaneously, compute suffix filter for species and suffixes < k
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
            lca_hogs = np.zeros((np.unique(fam_offsets).size,), dtype=np.int32)

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
                    lca_hogs[lca_off] = get_lca_off(curr_hogs, hog_parents)
                    curr_hogs = list(hog_offsets[i : i + 1])
                    curr_fam = fam
                    lca_off += 1

            # last family
            lca_hogs[lca_off] = get_lca_off(curr_hogs, hog_parents)
            return lca_hogs

        @numba.njit
        def _compute_kmer_table(
            sa,
            seq_buff,
            translated,
            sa_mask,
            hog_fams,
            hog_parents,
            table_idx,
            table_buff,
            hog_kmer_counts,
            k,
            DIGITS_AA,
            DIGITS_AA_LOOKUP,
            max_df,
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
                # the suffix array is sorted by the translated sequences, so
                # the suffixes of the k-mer are compared in translated form
                translated_kmer = translated[sa[ii] : (sa[ii] + k)].view(np.uint8)
                # first in windows of 50s
                # THIS MAY NEED TO BE INCREASED OVER TIME AND IS OPTIMISED FOR THE LUCA DB
                jj = min(ii + 50, len(sa))
                while (jj < len(sa)) and _same_kmer(translated, sa, translated_kmer, jj, k):
                    jj = min(jj + 50, len(sa))

                # then, refine with binary search
                lo = max(ii, (jj - 50) + 1)
                hi = jj
                while lo < hi:
                    m = int(np.floor((lo + hi) / 2))
                    if _same_kmer(translated, sa, translated_kmer, m, k):
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

                ## store kmer counts of fams and lca hogs, without the
                ## k-mers above the document-frequency cap
                nr_lca_hog_offsets = len(lca_hog_offsets)
                if max_df == 0 or nr_lca_hog_offsets <= max_df:
                    hog_kmer_counts[lca_hog_offsets] = hog_kmer_counts[lca_hog_offsets] + 1

                ## store LCA HOGs in table buffer
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

        def estimate_family_prob(tab, idx, h2f, max_df):
            @numba.njit
            def count_family_occurrence(tab, idx, h2f, max_df):
                c = np.zeros(h2f.max() + 1, dtype=np.int64)
                n = 0
                for i in range(len(idx) - 1):
                    df = np.int64(idx[i + 1]) - np.int64(idx[i])
                    if max_df != 0 and df > max_df:
                        continue
                    hogs = tab[idx[i] : idx[i + 1]]
                    c[h2f[hogs]] += df
                    n += df
                return c, n

            fam_occ, n = count_family_occurrence(tab, idx, h2f, max_df)
            return fam_occ / n

        def estimate_hog_prob(hog_counts, fam_tab, level_arr, hog2parent):
            @numba.njit(parallel=True, nogil=True)
            def cumulate_counts_nfams(
                hog_counts, fam_level_off, fam_level_num, level_arr, hog2parent
            ):
                hog_cum_counts = hog_counts.copy()

                for i in numba.prange(len(fam_level_off)):
                    s = fam_level_off[i]
                    e = np.int32(s + fam_level_num[i] + 2)
                    fam_level_offsets = level_arr[s:e]
                    cumulate_counts_1fam(hog_cum_counts, fam_level_offsets, hog2parent)

                return hog_cum_counts

            hog_occ = cumulate_counts_nfams(
                hog_counts,
                fam_tab.col("LevelOff"),
                fam_tab.col("LevelNum"),
                level_arr[:],
                hog2parent,
            )

            return hog_occ / hog_counts.sum()

        LOG.debug(" - filter suffix array and compute its HOG mask")
        n = len(self.db._db_Protein)
        sa_mask = np.zeros(sa.shape, dtype=np.uint32)
        sa_filter = np.zeros(sa.shape, dtype=np.bool_)

        _compute_mask_and_filter(
            sa,
            sa_mask,
            sa_filter,
            self.k,
            n,
            self.db._db_Protein.col("SpeOff"),
            self.db._db_Protein.col("HOGoff"),
            self.sp_filter,
        )

        # before filtering the sa, reorder and reverse the suffix filter
        sa = sa[~sa_filter[sa]]

        # filter and reorder the mask according to this filtered sa
        sa_mask = sa_mask[sa]

        LOG.debug(" - compute k-mer table")
        table_idx = np.zeros(
            (len(self.alphabet.DIGITS_AA) ** self.k + 1), dtype=np.uint32
        )

        # initiate buffer of size sa_mask, which is maximum size if all suffixes are from different HOGs
        table_buff = np.zeros((len(sa_mask)), dtype=np.uint32)

        hog_kmer_counts = np.zeros(len(self.db._db_HOG), dtype=np.uint64)

        h2f = self.db._db_HOG.col("FamOff")

        ii_table_buff = _compute_kmer_table(
            sa,
            seq_buff,
            translated,
            sa_mask,
            h2f,
            self.db._db_HOG.col("ParentOff"),
            table_idx,
            table_buff,
            hog_kmer_counts,
            self.k,
            self.alphabet.DIGITS_AA,
            self.alphabet.DIGITS_AA_LOOKUP,
            max_df,
        )

        # remove extra space
        table_buff = table_buff[:ii_table_buff]

        LOG.debug(" - write k-mer table")
        idx = self.db.db.root.Index
        prefix = "" if modality == "seq" else "SS"
        idx._f_setattr("kmer_max_df" if modality == "seq" else "ss_kmer_max_df", max_df)
        min_distinct = self.min_distinct.get(modality)
        if min_distinct is not None:
            idx._f_setattr("{}_kmer_min_distinct".format(modality), int(min_distinct))
        self.db.db.create_carray(
            idx, prefix + "TableIndex", obj=table_idx, filters=self.db._compr
        )
        self.db.db.create_carray(
            idx, prefix + "TableBuffer", obj=table_buff, filters=self.db._compr
        )

        # compute the family / hog probability estimates, assuming binomial distns
        fam_prob = estimate_family_prob(table_buff, table_idx, h2f, max_df)
        self.db.db.create_carray(
            idx, prefix + "FamilyProbability", obj=fam_prob, filters=self.db._compr
        )

        hog_prob = estimate_hog_prob(
            hog_kmer_counts,
            self.db._db_Family,
            self.db._db_LevelOffsets,
            self.db._db_HOG.col("ParentOff"),
        )
        self.db.db.create_carray(
            idx, prefix + "HOGProbability", obj=hog_prob, filters=self.db._compr
        )
