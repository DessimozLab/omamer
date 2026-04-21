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
import numba
import math
from numba import uint32
from numba.experimental import jitclass

from ._clock import clock
from .intrin import ctz_cpp


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


