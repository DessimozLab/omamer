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

from typing import Protocol
import tables


class OMAmerDBLike(Protocol):
    """
    Typed data interface for the default Database class,
    describing what tables and arrays an OMAmer database must have.

    This class is used only for static typing and
    does not describe the on-disk file format
    (e.g. field names in HDF5; for these details see `Database`).
    Instead, it only defines what properties `Database` should
    provide after loading the data from disk.

    See also: `OMAmerExtendedDBLike`
    """

    db: tables.File

    # The properties below correspond to Tables
    # defined in the Database class.
    # Anything that supports this protocol should
    # have those properties implemented.

    # 1. Tables

    @property
    def protein_table(self) -> tables.Table:
        ...

    @property
    def hog_table(self) -> tables.Table:
        ...

    @property
    def family_table(self) -> tables.Table:
        ...

    @property
    def species_table(self) -> tables.Table:
        ...

    @property
    def taxonomy_table(self) -> tables.Table:
        ...

    # 2. Arrays: CArrays (chunked layout arrays, support compression)

    @property
    def children_tax_carray(self) -> tables.CArray:
        ...

    @property
    def children_hog_carray(self) -> tables.CArray:
        ...

    @property
    def children_prot_carray(self) -> tables.CArray:
        ...

    @property
    def level_offset_carray(self) -> tables.CArray:
        ...

    @property
    def hog_taxa_carray(self) -> tables.CArray:
        ...

    # 2.1. Data structures computed by Index
    # from the protein sequence data.
    @property
    def seq_table_index_carray(self) -> tables.CArray:
        ...

    @property
    def seq_table_buffer_carray(self) -> tables.CArray:
        ...

    @property
    def seq_family_probability_carray(self) -> tables.CArray:
        ...

    @property
    def seq_hog_probability_carray(self) -> tables.CArray:
        ...

    # 3. Arrays: EArrays (extendable arrays, support append)
    @property
    def hog_id_buffer(self) -> tables.EArray:
        ...

    @property
    def protein_id_buffer(self) -> tables.EArray:
        ...

    # 4. Filters: compression details
    @property
    def compression_filters(self) -> tables.Filters:
        ...

    # 5. Other
    @property
    def access_mode(self) -> str:
        ...

class OMAmerExtendedDBLike(OMAmerDBLike, Protocol):
    """
    Typed data interface for an extended OMAmer database containing
    structural data together with sequence data.

    See also: `OMAmerDBLike`
    """

    @property
    def ss_table_index_carray(self) -> tables.CArray:
        ...

    @property
    def ss_table_buffer_carray(self) -> tables.CArray:
        ...

    @property
    def ss_family_probability_carray(self) -> tables.CArray:
        ...

    @property
    def ss_hog_probability_carray(self) -> tables.CArray:
        ...