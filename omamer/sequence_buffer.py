"""
    OMAmer - tree-driven and alignment-free protein assignment to sub-families

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
from multiprocessing import sharedctypes
from property_manager import lazy_property
import ctypes
import numpy as np

from .alphabets import Alphabet


class SequenceBuffer(object):
    """
    enables the use of numba for lists of sequences.
    """

    def __init__(self, seqs, ids, alphabet_n=21):
        """
        args:
         - seqs: list of sequences
         - ids: list of corresponding ids
        """
        self.alphabet = Alphabet(n=alphabet_n)
        self.add_seqs(*seqs)
        self.ids = np.array(ids) if ids else np.array(range(len(seqs)))

    # def __getstate__(self):
    #    return (self.prot_nr, self.n, self.buff_shr, self.buff_idx_shr)

    # def __setstate__(self, state):
    #    (self.prot_nr, self.n, self.buff_shr, self.buff_idx_shr) = state

    def add_seqs(self, *seqs):
        self.prot_nr = len(seqs)
        self.n = self.prot_nr + sum(len(s) for s in seqs)

        self.buff_shr = sharedctypes.RawArray(ctypes.c_uint8, self.n)
        self.buff[:] = self.alphabet.translate(
            np.frombuffer((" ".join(seqs) + " ").encode("ascii"), dtype="|S1")
        ).view(np.uint8)

        self.buff_idx_shr = sharedctypes.RawArray(ctypes.c_uint64, self.prot_nr + 1)
        for i in range(len(seqs)):
            self.idx[i + 1] = len(seqs[i]) + 1 + self.idx[i]

    @lazy_property
    def buff(self):
        return np.frombuffer(self.buff_shr, dtype=np.uint8).reshape(self.n)

    @lazy_property
    def idx(self):
        return np.frombuffer(self.buff_idx_shr, dtype=np.uint64).reshape(
            self.prot_nr + 1
        )

    # def __getitem__(self, i):
    #    s = int(self.idx[i])
    #    e = int(self.idx[i + 1] - 1)
    #    return self.buff[s:e].tobytes().decode("ascii")

    def get_seqlen(self, i):
        return self.idx[i] - self.idx[i - 1]
