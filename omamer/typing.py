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


class OMAmerDatabaseLike(Protocol):
    """
    Main class to describe HDFI OMAmer database format
    in terms of static type checker systems, i.e. "protocols".
    See: https://typing.python.org/en/latest/spec/protocol.html
    """

    db: tables.File

    # The properties below correspond to Tables
    # defined in the Database class.
    # Anything that supports this protocol should
    # have those properties implemented.

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
