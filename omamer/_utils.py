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
from io import TextIOWrapper
import bz2
import gzip
import logging
import os
import sys


logging.basicConfig(format="%(levelname)s: %(message)s")
LOG = logging.getLogger(__name__)
SILENT = False

# File opening. This is based on the example on SO here:
# http://stackoverflow.com/a/26986344
fmagic = {b"\x1f\x8b\x08": gzip.open, b"\x42\x5a\x68": bz2.BZ2File}


def auto_open(fn, *args):
    """
    Opens files based on their "magic bytes". Supports bz2 and gzip. If it
    finds neither of these, presumption is it is a standard, uncompressed
    file.
    """
    if os.path.isfile(fn) and os.stat(fn).st_size > 0:
        with open(fn, "rb") as fp:
            fs = fp.read(max(len(x) for x in fmagic))
        for magic, _open in fmagic.items():
            if fs.startswith(magic):
                return _open(fn, *args)
    else:
        if fn.endswith(".gz"):
            return gzip.open(fn, *args)
        elif fn.endswith(".bz2"):
            return bz2.BZ2File(fn, *args)
        elif fn.endswith(".xz"):
            return lzma.open(fn, *args)

    return open(fn, *args)


def set_log_level(x):
    LOG.setLevel(getattr(logging, x.upper()))


def set_if_silent(x):
    if x is True:
        SILENT = TRUE


def is_progress_disabled():
    return LOG.getEffectiveLevel() != logging.DEBUG


def print_line(n, file=None):
    file = file if file is not None else sys.stderr
    if not SILENT:
        print("=" * n, file=file, flush=True)


def print_message(x, no_newline=None, file=None):
    file = file if file is not None else sys.stderr
    if not SILENT:
        print(x, file=file, end="\n" if no_newline is not True else "", flush=True)


def compute_file_md5(fn):
    from filehash import FileHash

    md5hasher = FileHash("md5")
    return md5hasher.hash_file(fn)
