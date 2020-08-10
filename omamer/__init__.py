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
from datetime import date

__packagename__ = "omamer"
__version__ = "0.1.2"
__copyright__ = "(C) 2019-{:d} Victor Rossier <victor.rossier@unil.ch> and Alex Warwick Vesztrocy <alex@warwickvesztrocy.co.uk>".format(
    date.today().year
)

from .hierarchy import *
from .alphabets import *
from .database import *
from .index import *
from .flat_search import *
from .search import *
from .prob_search import *
from .merge_search import *
from .validation import *