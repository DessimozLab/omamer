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
import numpy as np


class Alphabet(object):
    def __init__(self, n=21):
        self.n = n
        self.setup()

    def setup(self):
        if self.n == 21:
            digits = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWXY", dtype=np.uint8)
            lookup = np.zeros(np.max(digits) + 1, dtype=np.uint8)
            lookup[digits] = np.arange(len(digits))
            trans = None
        elif self.n == 13:
            # Reduced from Linclust merges (A, S, T), (D, N), (E, Q), (F, Y), (I, V), (K, R) and (L, M)
            digits = np.frombuffer(b"ACDEFGHIKLPWX", dtype=np.uint8)
            lookup = np.zeros(ord(b"Y") + 1, dtype=np.uint8)
            lookup[digits] = np.arange(len(digits))

            # (A, S, T)
            lookup[ord(b"S")] = lookup[ord(b"A")]
            lookup[ord(b"T")] = lookup[ord(b"A")]
            # (D, N)
            lookup[ord(b"N")] = lookup[ord(b"D")]
            # (E, Q)
            lookup[ord(b"Q")] = lookup[ord(b"E")]
            # (F, Y)
            lookup[ord(b"Y")] = lookup[ord(b"F")]
            # (I, V)
            lookup[ord(b"V")] = lookup[ord(b"I")]
            # (K, R)
            lookup[ord(b"R")] = lookup[ord(b"K")]
            # (L, M)
            lookup[ord(b"M")] = lookup[ord(b"L")]

            # Translation table
            trans = np.zeros(ord(b"Y") + 1, dtype=np.uint8)
            trans[digits] = np.arange(1, len(digits) + 1)

            # (A, S, T)
            trans[ord(b"S")] = trans[ord(b"A")]
            trans[ord(b"T")] = trans[ord(b"A")]
            # (D, N)
            trans[ord(b"N")] = trans[ord(b"D")]
            # (E, Q)
            trans[ord(b"Q")] = trans[ord(b"E")]
            # (F, Y)
            trans[ord(b"Y")] = trans[ord(b"F")]
            # (I, V)
            trans[ord(b"V")] = trans[ord(b"I")]
            # (K, R)
            trans[ord(b"R")] = trans[ord(b"K")]
            # (L, M)
            trans[ord(b"M")] = trans[ord(b"L")]

        else:
            raise ValueError("Unknown reduced alphabet.")

        self.digits = digits
        self.lookup = lookup
        self.trans = trans

    def translate(self, x):
        if self.trans is None:
            return x
        else:
            return self.trans[x.view(np.uint8)].view("|S1")

    @property
    def DIGITS_AA_LOOKUP(self):
        return self.lookup

    @property
    def DIGITS_AA(self):
        return self.digits
