"""
    OMAmer - tree-driven and alignment-free protein assignment to sub-families

    (C) 2024 Nikolai Romashchenko <nikolai.romashchenko@unil.ch>
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
import math
import numba
from numba import uint32
from numba.experimental import jitclass
from property_manager import lazy_property

from ._utils import LOG
from .alphabets import Alphabet, get_transform
from .hierarchy import get_lca_off, get_leaves
from ._clock import clock


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
    def __init__(self, db, k=6, reduced_alphabet=False, hidden_taxa=[]):
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
    def build_kmer_table(self, seq_buff):
        assert self.db.mode in {"w", "a"}, "Index must be opened in write mode."
        assert "/Index" not in self.db.db, "Index has already been computed"

        # build suffix array with option to translate the sequence buffer first
        sa = self._build_suffixarray(
            self.alphabet.translate(seq_buff), len(self.db._db_Protein)
        )
        self._build_kmer_table(seq_buff, sa)

    @staticmethod
    def _build_suffixarray(seqs, n):
        # Build suffix array
        LOG.debug(" - building suffix array for sequences")

        from PySAIS import sais
        # import sais here, otherwise we need it for search-time dependency

        sa = sais(seqs)
        sa[:n].sort()  # Sort delimiters by position
        return sa

    def _build_kmer_table(self, seq_buff, sa):
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

        def estimate_family_prob(tab, idx, h2f):
            @numba.njit
            def count_family_occurrence(tab, idx, h2f):
                c = np.zeros(h2f.max() + 1, dtype=np.uint32)
                for i in range(len(idx) - 1):
                    hogs = tab[idx[i] : idx[i + 1]]
                    c[h2f[hogs]] += idx[i + 1] - idx[i]
                return c

            fam_occ = count_family_occurrence(tab, idx, h2f)
            return fam_occ / idx[-1]

        def estimate_hog_prob(idx, hog_counts, fam_tab, level_arr, hog2parent):
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

            return hog_occ / idx[-1]

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
            sa_mask,
            h2f,
            self.db._db_HOG.col("ParentOff"),
            table_idx,
            table_buff,
            hog_kmer_counts,
            self.k,
            self.alphabet.DIGITS_AA,
            self.alphabet.DIGITS_AA_LOOKUP,
        )

        # remove extra space
        table_buff = table_buff[:ii_table_buff]

        LOG.debug(" - write k-mer table")
        idx = self.db.db.create_group("/", "Index", "hog indexes")
        idx._f_setattr("k", self.k)
        idx._f_setattr("alphabet_n", self.alphabet.n)
        idx._f_setattr("hidden_taxa", self.hidden_taxa)
        self.db.db.create_carray(
            idx, "TableIndex", obj=table_idx, filters=self.db._compr
        )
        self.db.db.create_carray(
            idx, "TableBuffer", obj=table_buff, filters=self.db._compr
        )

        # compute the family / hog probability estimates, assuming binomial distns
        fam_prob = estimate_family_prob(table_buff, table_idx, h2f)
        self.db.db.create_carray(
            idx, "FamilyProbability", obj=fam_prob, filters=self.db._compr
        )

        hog_prob = estimate_hog_prob(
            table_idx,
            hog_kmer_counts,
            self.db._db_Family,
            self.db._db_LevelOffsets,
            self.db._db_HOG.col("ParentOff"),
        )
        self.db.db.create_carray(
            idx, "HOGProbability", obj=hog_prob, filters=self.db._compr
        )


@numba.njit
def set_bits(packed, index, value, bit_width):
    bit_pos = index * bit_width
    word_index = bit_pos // 32
    offset = bit_pos % 32

    packed[word_index] |= (value << offset) & 0xFFFFFFFF

    if offset + bit_width > 32:
        packed[word_index + 1] |= (value >> (32 - offset)) & 0xFFFFFFFF

@numba.njit
def get_bits(packed, index, bit_width):
    bit_pos = index * bit_width
    word_index = bit_pos // 32
    offset = bit_pos % 32

    value = (packed[word_index] >> offset) & 0xFFFFFFFF

    if offset + bit_width > 32:
        # Spans two words
        value |= (packed[word_index + 1] << (32 - offset)) & 0xFFFFFFFF

    return value & ((1 << bit_width) - 1)

@numba.njit
def select1_in_word(word, rank):
    count = 0
    for bit in range(32):
        if (word >> bit) & 1:
            if count == rank:
                return bit
            count += 1
    return -1

@numba.njit('int_(uint32)')
def popcount(v):
    """
    Counts the number of 1s in the bit representation of v.
    https://stackoverflow.com/questions/71097470/msb-lsb-popcount-in-numba

    If for some reason you want to understand it:
    https://www.chessprogramming.org/Population_Count
    """
    v = np.uint32(v)  # ensure 32-bit
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    c = np.uint32((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24
    return c

# @numba.njit
# def popcount(x):
#     x = x - ((x >> 1) & 0x55555555)
#     x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
#     x = (x + (x >> 4)) & 0x0F0F0F0F
#     x = x + (x >> 8)
#     x = x + (x >> 16)
#     return x & 0x0000003F


@numba.njit
def select1(packed, rank):
    count = 0
    for word_idx in range(len(packed)):
        word = packed[word_idx]
        # popcount, the number of 1s in the word
        ones = popcount(word)

        if rank < count + ones:
            # Found the word that contains the rank-th 1
            # Now extract it directly
            return word_idx * 32 + select1_in_word(word, rank - count)

        count += ones

    return -1  # not found


spec = [
    ("n", uint32),
    ("l", uint32),
    ("lower_packed", uint32[:]),
    ("upper_packed", uint32[:]),
    ("bit_length_upper", uint32),
]

@jitclass(spec)
class EliasFanoLite:
    def __init__(self, n, l, lower_packed, upper_packed):
        self.n = n
        self.l = l
        self.lower_packed = lower_packed
        self.upper_packed = upper_packed

@numba.njit
def to_elias_fano(values):
    n = len(values)
    max_value = values[-1]  # assume sorted
    l = int(math.floor(math.log2(max_value // n + 1)))
    low_mask = (1 << l) - 1

    u = (max_value >> l) + 1
    bitvector_len = n + u

    total_lower_bits = n * l
    lower_words = (total_lower_bits + 31) // 32
    upper_words = (bitvector_len + 31) // 32

    lower_packed = np.zeros(lower_words, dtype=np.uint32)
    upper_packed = np.zeros(upper_words, dtype=np.uint32)

    for i in range(n):
        v = values[i]
        lower = v & low_mask
        upper = v >> l

        set_bits(lower_packed, i, lower, l)
        pos = upper + i
        word = pos // 32
        offset = pos % 32
        upper_packed[word] |= (1 << offset)

    return EliasFanoLite(n, l, lower_packed, upper_packed)


@numba.njit
def ef_storage_size_bytes(ef: EliasFanoLite):
    lower_bits_used = ef.n * ef.l
    upper_bits_used = ef.bit_length_upper

    lower_bytes = ((lower_bits_used + 31) // 32) * 4
    upper_bytes = ((upper_bits_used + 31) // 32) * 4

    return lower_bytes + upper_bytes

@numba.njit
def ensure_capacity(buf, needed, current_size):
    """
    Dynamic memory allocator to ensure that input array has
    the required size. If needed, doubles the memory
    """
    if needed <= len(buf):
        return buf

    # dynamic allocation if needed
    new_size = max(len(buf) * 2, needed)
    new_buf = np.empty(new_size, dtype=np.uint32)
    for i in range(current_size):
        new_buf[i] = buf[i]
    return new_buf

@numba.njit
def update_with_elias_fano(idx, buff):
    """
    Encodes the inverted (kmers -> HOGs) index.
    Buffer containing the list of HOGs
    """
    n_lists = len(idx) - 1

    initial_capacity = len(buff) // 2
    flat_buffer = np.empty(initial_capacity, dtype=np.uint32)

    new_idx = np.empty(len(idx), dtype=np.uint32)
    raw_flags = np.zeros(n_lists, dtype=np.uint8)

    offset = 0

    for i in range(n_lists):
        start = idx[i]
        end = idx[i + 1]
        length = end - start

        new_idx[i] = offset

        if length == 0:
            continue

        values = buff[start:end]

        ef = to_elias_fano(values)
        total_bytes = ef_storage_size_bytes(ef)
        uncompressed_bytes = length * 4
        compression_ratio = total_bytes / uncompressed_bytes

        if compression_ratio > 1.0:
            # If we ended up with larger representation, keep it raw
            raw_flags[i] = 1
            flat_buffer = ensure_capacity(flat_buffer, offset + length, offset)
            for j in range(length):
                flat_buffer[offset + j] = values[j]
            offset += length
        else:
            # Otherwise store it compressed
            raw_flags[i] = 0
            l_len = len(ef.lower_packed)
            u_len = len(ef.upper_packed)
            total_len = 4 + l_len + u_len

            flat_buffer = ensure_capacity(flat_buffer, offset + total_len, offset)

            flat_buffer[offset] = ef.l
            flat_buffer[offset + 1] = l_len
            flat_buffer[offset + 2] = u_len
            flat_buffer[offset + 3] = length

            for j in range(l_len):
                flat_buffer[offset + 4 + j] = ef.lower_packed[j]
            for j in range(u_len):
                flat_buffer[offset + 4 + l_len + j] = ef.upper_packed[j]

            offset += total_len

    new_idx[-1] = offset
    return flat_buffer[:offset], new_idx, raw_flags

@numba.njit
def ctz(v):
    """
    Count Trailing Zeros: count the number of leading zeros from the right.
    See:
    https://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightModLookup
    """
    if v == 0:
        return 32

    mod37_bit_position = np.array([
        32, 0, 1, 26, 2, 23, 27, 0,
        3, 16, 24, 30, 28, 11, 0, 13,
        4, 7, 17, 0, 25, 22, 31, 15,
        29, 10, 12, 6, 0, 21, 14, 9,
        5, 20, 8, 19, 18
    ], dtype=np.uint32)

    isolated = (-v & v) % 37
    return mod37_bit_position[isolated]


def naive_ctz(v):
    if v == 0:
        return 32
    count = 0
    while (v & 1) == 0:
        v >>= 1
        count += 1
    return count

@numba.njit
def from_elias_fano_correct(l, lower, upper, n):
    """
    A slower version of Elias Fano decoding using select1
    (selecting the position i-th 1 in the word)
    For unit tests
    """
    out = np.empty(n, dtype=np.uint32)
    for i in range(n):
        upper_pos = select1(upper, i)
        upper_val = upper_pos - i
        lower_val = get_bits(lower, i, l)
        out[i] = (upper_val << l) | lower_val
    return out

import cppyy
import cppyy.numba_ext


cppyy.cppdef(r"""
#include <stdint.h>

extern "C" uint32_t ctz32(uint32_t x) {
    return __builtin_ctz(x);
}

extern "C"
uint32_t extract_lower_bits(uint32_t word, uint32_t next_word,
                            uint32_t offset, uint32_t bit_width) {
    uint32_t value = word >> offset;
    if (offset + bit_width > 32) {
        value |= next_word << (32 - offset);
    }
    return value & ((1U << bit_width) - 1);
}
""")

ctz_cpp = cppyy.gbl.ctz32
#decode_ef = cppyy.gbl.decode_ef_element


@numba.njit
def from_elias_fano(l, lower, upper, n):
    out = np.empty(n, dtype=np.uint32)
    word_idx = 0
    word = upper[0]

    for i in range(n):

        while word == 0:
            word_idx += 1
            word = upper[word_idx]

        pos = word_idx * 32 + ctz_cpp(word)
        upper_val = pos - i
        word &= word - 1

        lower_val = get_bits(lower, i, l)
        out[i] = (upper_val << l) | lower_val

    return out


@numba.njit
def retrieve_list(i, idx, buff, raw_flags):
    start = idx[i]
    end = idx[i + 1]

    select_time = 0
    bits_time = 0

    if end == start:
        return np.empty((0,), dtype=np.uint32), select_time, bits_time

    if raw_flags[i]:
        return buff[start:end], select_time, bits_time

    t0 = clock()

    # Retrieve from Elias-Fano representation
    l = buff[start]
    lenL = buff[start + 1]
    lenU = buff[start + 2]
    n = buff[start + 3]

    lower_start = start + 4
    upper_start = lower_start + lenL

    lower = buff[lower_start : lower_start + lenL]
    upper = buff[upper_start : upper_start + lenU]

    t1 = clock()
    bits_time += t1 - t0

    t0 = clock()
    out = from_elias_fano(l, lower, upper, n)
    t1 = clock()
    select_time += t1 - t0

    return out, select_time, bits_time


@numba.njit
def batch_decode(kmers, idx, buff, raw_flags, start_kmer,
                 x_flag, out, sizes, max_elements):
    """
    Elias-Fano decoder for HOGs associated with a batch of k-mers.
    More efficient than the per-k-mer decoder as it's more cache friendly.
    Outputs HOGs into out and lengths of HOG arrays into sizes.
    """
    i = start_kmer
    sizes[0] = 0
    current_n = 0

    t0 = clock()

    # First pass: determine how many HOG lists fit to the batch
    while i < len(kmers):
        kmer = kmers[i]
        if kmer == x_flag:
            i += 1
            sizes[i] = 0
            continue

        start = idx[kmer]
        end = idx[kmer + 1]

        if raw_flags[kmer]:
            n = end - start
        else:
            n = buff[start + 3]

        if current_n + n > max_elements:
            #sizes[i] = current_n + n
            break

        current_n += n
        sizes[i] = n
        i += 1


    t1 = clock()
    bits_time = t1 - t0

    # Second pass: decode into buffer
    out_offset = 0
    select_time = 0

    t0 = clock()
    for j in range(start_kmer, i):
        kmer = kmers[j]
        if kmer == x_flag:
            continue

        start = idx[kmer]
        end = idx[kmer + 1]

        if raw_flags[kmer]:
            n = end - start
            out[out_offset : out_offset + n] = buff[start:end]
        else:
            l = buff[start]
            lenL = buff[start + 1]
            lenU = buff[start + 2]
            n = buff[start + 3]

            lower_start = start + 4
            upper_start = lower_start + lenL

            lower = buff[lower_start : lower_start + lenL]
            upper = buff[upper_start : upper_start + lenU]

            out[out_offset : out_offset + n] = from_elias_fano(l, lower, upper, n)

        out_offset += n

    t1 = clock()
    select_time += t1 - t0

    return i, select_time, bits_time


