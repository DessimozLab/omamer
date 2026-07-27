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
import numpy as np
import numpy.typing as npt
import numba
from property_manager import lazy_property

from ._utils import LOG
from .alphabets import Alphabet, get_transform
from .hierarchy import get_lca_off, get_leaves
from .typing import OMAmerDBLike


def validate_kmer_percentage(value):
    """Validate and normalise the percentage of indexed k-mers to retain."""
    percentage = float(value)
    if not 0.0 < percentage <= 100.0:
        raise ValueError("kmer_percentage must be in (0, 100]")
    return percentage


@numba.njit(cache=True)
def select_kmer_max_df(table_idx, n_families, kmer_percentage):
    """Return the document-frequency cutoff for a PMI-ranked k-mer filter.

    A posting list contains one LCA HOG per family, so its length is the
    k-mer's family document frequency (df). Under a uniform family prior, an
    observed k-mer carries ``log2(n_families / df)`` bits about its family.
    The score is monotonic in df, and retaining every tie at the cutoff makes
    a single maximum df sufficient to reproduce the selection later.
    """
    df_hist = np.zeros(n_families + 1, dtype=np.uint64)
    n_present = 0
    for kmer in range(table_idx.size - 1):
        df = table_idx[kmer + 1] - table_idx[kmer]
        if df > 0:
            df_hist[df] += 1
            n_present += 1

    n_keep = int(np.ceil(n_present * kmer_percentage / 100.0))
    if n_keep < 1:
        n_keep = 1

    cumulative = 0
    max_df = 0
    for df in range(1, n_families + 1):
        cumulative += df_hist[df]
        if cumulative >= n_keep:
            max_df = df
            break

    n_retained = 0
    for df in range(1, max_df + 1):
        n_retained += df_hist[df]

    return n_present, n_retained, max_df


@numba.njit(cache=True, nogil=True)
def filtered_hog_kmer_counts(table_idx, table_buff, max_df, n_hogs):
    """Count index postings retained by a document-frequency cutoff."""
    hog_counts = np.zeros(n_hogs, dtype=np.uint64)
    for kmer in range(table_idx.size - 1):
        lo = table_idx[kmer]
        hi = table_idx[kmer + 1]
        if max_df == 0 or hi - lo <= max_df:
            for pos in range(lo, hi):
                hog_counts[table_buff[pos]] += 1
    return hog_counts


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


class Index(object):
    def __init__(
        self,
        db: OMAmerDBLike,
        k=6,
        reduced_alphabet=False,
        hidden_taxa=(),
        kmer_percentage=100.0,
    ):
        # load database object
        self.db = db

        # load k, alphabet size and hidden taxa
        if "/Index" in self.db.db:
            attrs = self.db.db.root.Index._v_attrs
            self.k = attrs["k"]
            alphabet_n = attrs["alphabet_n"]
            self.hidden_taxa = attrs["hidden_taxa"]
            # Databases created before build-time filtering are unfiltered.
            self.kmer_percentage = validate_kmer_percentage(
                getattr(attrs, "kmer_percentage", 100.0)
            )
            self.kmer_max_df = int(getattr(attrs, "kmer_max_df", 0))
            self.ss_kmer_max_df = int(getattr(attrs, "ss_kmer_max_df", 0))
            if self.kmer_percentage < 100.0 and self.kmer_max_df <= 0:
                raise ValueError(
                    "Database records a filtered kmer_percentage but has no "
                    "sequence k-mer document-frequency cutoff"
                )
            if (
                self.kmer_percentage < 100.0
                and "/Index/SSTableIndex" in self.db.db
                and self.ss_kmer_max_df <= 0
            ):
                raise ValueError(
                    "Database records a filtered kmer_percentage but has no "
                    "3Di k-mer document-frequency cutoff"
                )
        else:
            self.k = k
            alphabet_n = 21 if not reduced_alphabet else 13
            self.hidden_taxa = hidden_taxa
            self.kmer_percentage = validate_kmer_percentage(kmer_percentage)
            self.kmer_max_df = 0
            self.ss_kmer_max_df = 0

        self.alphabet = Alphabet(n=alphabet_n)

    @lazy_property
    def sp_filter(self) -> npt.NDArray[np.bool_]:
        sp_filter = np.full((len(self.db.species_table),), False)
        if len(self.hidden_taxa) > 0:
            tax_tab = self.db.taxonomy_table[:]
            child_tax = self.db.children_tax_carray[:]

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
                    text = bytes(self.db.species_table[sp_ii]['ID']).decode('ascii')
                    LOG.debug('     - hiding {}'.format(text))
                    sp_filter[sp_ii] = True
                else:
                    # filter all leaves below declared taxon
                    for sp_jj in tax_tab["SpeOff"][get_leaves(tax_ii, tax_tab, child_tax)]:
                        text = bytes(self.db.species_table[sp_jj]['ID']).decode('ascii')
                        LOG.debug('     - hiding {}'.format(text))
                        sp_filter[sp_jj] = True
        return sp_filter

    @property
    def kmer_table(self):
        return {
            "buff": self.db.seq_table_buffer_carray,
            "idx": self.db.seq_table_index_carray,
        }

    @property
    def ss_kmer_table(self):
        return {
            "buff": self.db._db_Index_SSTableBuffer,
            "idx": self.db._db_Index_SSTableIndex,
        }

    ### main function to build the index ###
    def build_kmer_table(self, seq_buff, ss_buff):
        assert self.db.access_mode in {"w", "a"}, "Index must be opened in write mode."
        assert "/Index" not in self.db.db, "Index has already been computed"

        # build suffix array with option to translate the sequence buffer first

        LOG.debug(" - building suffix array for sequences")
        sa = self._build_suffixarray(
            self.alphabet.translate(seq_buff), len(self.db.protein_table)
        )

        # structure suffix array
        ss_sa = None
        if ss_buff is not None:
            LOG.debug(" - building suffix array for structures")
            ss_sa = self._build_suffixarray(
                self.alphabet.translate(ss_buff), len(self.db.protein_table)
            )

        self._build_kmer_table(seq_buff, sa, ss_buff, ss_sa)

    @staticmethod
    def _build_suffixarray(seqs, n):
        # Build suffix array

        from PySAIS import sais
        # import sais here, otherwise we need it for search-time dependency

        sa = sais(seqs)
        sa[:n].sort()  # Sort delimiters by position
        return sa

    def _build_kmer_table(self, seq_buff, sa, ss_buff, ss_sa):
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
            sa_mask,
            hog_fams,
            hog_parents,
            table_idx,
            table_buff,
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
                # first in windows of 50s
                # THIS MAY NEED TO BE INCREASED OVER TIME AND IS OPTIMISED FOR THE LUCA DB
                jj = min(ii + 50, len(sa))
                while (jj < len(sa)) and _same_kmer(seq_buff, sa, kmer, jj, k):
                    jj = min(jj + 50, len(sa))

                # then, refine with binary search
                lo = max(ii, (jj - 50) + 1)
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

                ## store kmer counts of fams and lca hogs
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

        def estimate_family_prob(hog_counts, h2f):
            @numba.njit
            def count_family_occurrence(hog_counts, h2f):
                c = np.zeros(h2f.max() + 1, dtype=np.uint64)
                for hog in range(hog_counts.size):
                    c[h2f[hog]] += hog_counts[hog]
                return c

            fam_occ = count_family_occurrence(hog_counts, h2f)
            return fam_occ / hog_counts.sum()

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

        def apply_information_filter(table_idx, table_buff, hog_counts, modality):
            if self.kmer_percentage == 100.0:
                return hog_counts, 0

            n_present, n_retained, max_df = select_kmer_max_df(
                table_idx,
                len(self.db.family_table),
                self.kmer_percentage,
            )
            if n_retained == 0:
                raise RuntimeError(
                    "{} k-mer information filter retained no indexed k-mers".format(
                        modality
                    )
                )
            threshold = np.log2(len(self.db.family_table) / max_df)
            LOG.info(
                "{} information filter: retained {} of {} indexed k-mers "
                "({:.2f}%; df <= {}; PMI >= {:.3f} bits)".format(
                    modality,
                    n_retained,
                    n_present,
                    100.0 * n_retained / n_present,
                    max_df,
                    threshold,
                )
            )
            return (
                filtered_hog_kmer_counts(
                    table_idx,
                    table_buff,
                    max_df,
                    len(self.db.hog_table),
                ),
                max_df,
            )

        LOG.debug(" - filter suffix array and compute its HOG mask")
        n = len(self.db.protein_table)
        sa_mask = np.zeros(sa.shape, dtype=np.uint32)
        sa_filter = np.zeros(sa.shape, dtype=np.bool_)

        _compute_mask_and_filter(
            sa,
            sa_mask,
            sa_filter,
            self.k,
            n,
            self.db.protein_table.col("SpeOff"),
            self.db.protein_table.col("HOGoff"),
            np.asarray(self.sp_filter),
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

        hog_kmer_counts = np.zeros(len(self.db.hog_table), dtype=np.uint64)

        h2f = self.db.hog_table.col("FamOff")

        ii_table_buff = _compute_kmer_table(
            sa,
            seq_buff,
            sa_mask,
            h2f,
            self.db.hog_table.col("ParentOff"),
            table_idx,
            table_buff,
            hog_kmer_counts,
            self.k,
            self.alphabet.DIGITS_AA,
            self.alphabet.DIGITS_AA_LOOKUP,
        )

        # remove extra space
        table_buff = table_buff[:ii_table_buff]
        hog_kmer_counts, self.kmer_max_df = apply_information_filter(
            table_idx, table_buff, hog_kmer_counts, "Sequence"
        )

        LOG.debug(" - write k-mer table")
        idx = self.db.db.create_group("/", "Index", "hog indexes")
        idx._f_setattr("k", self.k)
        idx._f_setattr("alphabet_n", self.alphabet.n)
        idx._f_setattr("hidden_taxa", self.hidden_taxa)
        idx._f_setattr("kmer_percentage", self.kmer_percentage)
        idx._f_setattr("kmer_max_df", self.kmer_max_df)
        self.db.db.create_carray(
            idx, "TableIndex", obj=table_idx, filters=self.db.compression_filters
        )
        self.db.db.create_carray(
            idx, "TableBuffer", obj=table_buff, filters=self.db.compression_filters
        )

        # compute the family / hog probability estimates, assuming binomial distns
        fam_prob = estimate_family_prob(hog_kmer_counts, h2f)
        self.db.db.create_carray(
            idx, "FamilyProbability", obj=fam_prob, filters=self.db.compression_filters
        )

        hog_prob = estimate_hog_prob(
            hog_kmer_counts,
            self.db.family_table,
            self.db.level_offset_carray,
            self.db.hog_table.col("ParentOff"),
        )
        self.db.db.create_carray(
            idx, "HOGProbability", obj=hog_prob, filters=self.db.compression_filters
        )

        ############################
        # structure index

        if ss_sa is not None and ss_buff is not None:
            ss_sa_mask = np.zeros(ss_sa.shape, dtype=np.uint32)
            ss_sa_filter = np.zeros(ss_sa.shape, dtype=np.bool_)

            _compute_mask_and_filter(
                ss_sa,
                ss_sa_mask,
                ss_sa_filter,
                self.k,
                n,
                self.db.protein_table.col("SpeOff"),
                self.db.protein_table.col("HOGoff"),
                self.sp_filter,
            )
            ss_sa = ss_sa[~ss_sa_filter[ss_sa]]
            ss_sa_mask = ss_sa_mask[ss_sa]

            LOG.debug(" - compute 3di k-mer table")
            ss_table_idx = np.zeros(
                (len(self.alphabet.DIGITS_AA) ** self.k + 1), dtype=np.uint32
            )

            # initiate buffer of size sa_mask, which is maximum size if all suffixes are from different HOGs
            ss_table_buff = np.zeros((len(ss_sa_mask)), dtype=np.uint32)
            ss_hog_kmer_counts = np.zeros(len(self.db.hog_table), dtype=np.uint64)
            h2f = self.db.hog_table.col("FamOff")
            ss_ii_table_buff = _compute_kmer_table(
                ss_sa,
                ss_buff,
                ss_sa_mask,
                h2f,
                self.db.hog_table.col("ParentOff"),
                ss_table_idx,
                ss_table_buff,
                ss_hog_kmer_counts,
                self.k,
                self.alphabet.DIGITS_AA,
                self.alphabet.DIGITS_AA_LOOKUP,
            )

            ss_table_buff = ss_table_buff[:ss_ii_table_buff]
            ss_hog_kmer_counts, self.ss_kmer_max_df = apply_information_filter(
                ss_table_idx, ss_table_buff, ss_hog_kmer_counts, "3Di"
            )
            idx._f_setattr("ss_kmer_max_df", self.ss_kmer_max_df)

            LOG.debug(" - write structure k-mer table")
            self.db.db.create_carray(
                idx, "SSTableIndex", obj=ss_table_idx, filters=self.db.compression_filters
            )
            self.db.db.create_carray(
                idx, "SSTableBuffer", obj=ss_table_buff, filters=self.db.compression_filters
            )

            fam_ss_prob = estimate_family_prob(ss_hog_kmer_counts, h2f)
            self.db.db.create_carray(
                idx, "SSFamilyProbability", obj=fam_ss_prob, filters=self.db.compression_filters
            )

            hog_ss_prob = estimate_hog_prob(
                ss_hog_kmer_counts,
                self.db.family_table,
                self.db.level_offset_carray,
                self.db.hog_table.col("ParentOff"),
            )
            self.db.db.create_carray(
                idx, "SSHOGProbability", obj=hog_ss_prob, filters=self.db.compression_filters
            )
        ############################
