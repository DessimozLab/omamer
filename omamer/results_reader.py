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
import pandas as pd

from ._utils import auto_open


def results_reader(filename):
    # read the metadata first
    metadata = {}
    with auto_open(filename, "rt") as fp:
        for x in fp:
            if x.startswith("!"):
                x = x[1:].rstrip().split(": ")
                k = x[0]
                v = ": ".join(x[1:])
                metadata[k] = v
            else:
                break

    df = pd.read_csv(auto_open(filename), sep="\t", comment="!")
    return (df, metadata)
