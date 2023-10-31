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
from Bio import SeqIO


class SequenceReader(object):
    @staticmethod
    def read(fp, k, format="fasta", chunksize=None, sanitiser=None):
        ids = []
        seqs = []
        for rec in filter(lambda x: (len(x.seq) >= k), SeqIO.parse(fp, format)):
            ids.append(rec.id)
            s = str(rec.seq).upper()
            seqs.append(sanitiser(s) if sanitiser is not None else s)

            if chunksize is not None and len(ids) == chunksize:
                yield (ids, seqs)
                ids = []
                seqs = []

        yield (ids, seqs)
