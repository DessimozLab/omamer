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

import ctypes
import sys


if sys.version_info[1] < 13:
    clock = ctypes.pythonapi._PyTime_GetSystemClock
    as_seconds = ctypes.pythonapi._PyTime_AsSecondsDouble
else:
    # in python 3.13 and later, PyTime C API was made public
    clock = ctypes.pythonapi.PyTime_TimeRaw
    as_seconds = ctypes.pythonapi.PyTime_AsSecondsDouble


# Set the argument types and return types of the functions
clock.argtypes = []
clock.restype = ctypes.c_int64

as_seconds.argtypes = [ctypes.c_int64]
as_seconds.restype = ctypes.c_double


