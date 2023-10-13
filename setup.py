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
from setuptools import setup, find_packages


name = "omamer"
__version__ = None
with open("{:s}/__init__.py".format(name), "rt") as fp:
    for line in fp:
        if line.startswith("__version__"):
            exec(line.rstrip())

requirements = [
    "alive_progress",
    "biopython",
    "ete3",
    "filehash",
    "numba",
    "numpy",
    "pandas>2.0.0",
    "property_manager",
    "Rmath4",
    "scipy",
    "tables",
    "tqdm",
]
extra_requirements = {"build": ["pysais"]}

desc = "OMAmer - tree-driven and alignment-free protein assignment to sub-families"

setup(
    name=name,
    version=__version__,
    author="Victor Rossier and Alex Warwick Vesztrocy",
    email="alex@warwickvesztrocy.co.uk",
    url="https://github.com/DessimozLab/omamer",
    description=desc,
    packages=find_packages(),
    install_requires=requirements,
    extras_require=extra_requirements,
    python_requires=">=3.8",
    license="LGPLv3",
    #scripts=["bin/omamer"],
    entry_points={'console_scripts': ['omamer = omamer.main:main']}
)
