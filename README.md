# OMAmer

OMAmer is a novel alignment-free protein family assignment method, which limits over-specific subfamily assignments and is suited to phylogenomic databases with thousands of genomes. It is based on an innovative method using subfamily-informed k-mers for alignment-free mapping to ancestral protein subfamilies. Whilst able to reject non-homologous family-level assignments, it has provided better and quicker subfamily-level assignments than a method based on closest sequences (using DIAMOND).

# Installation
Requires Python >= 3.6. Download the package from the PyPI, resolving the dependencies by using ``pip install omamer``.

Alternatively, clone this repository and install manually.

# Pre-Built Database

Download a pre-built database from the links below. More databases to follow shortly -- if you have a request, please create an issue.

(add link to Metazoan database)


# omamer mkdb - Building a Database
This is currently reliant on the OMA browser's database file. Building using OrthoXML and FASTA files available shortly. Pre-built databases are available to download.

# omamer search - Searching a Database
Assign proteins to families in a pre-existing database.
## Usage
Required arguments: ``--db``, ``--query``

     usage: omamer search [-h] --db DB --query QUERY [--out OUT]
                     [--include_extant_genes] [--chunksize CHUNKSIZE]
                     [--nthreads NTHREADS] [--log_level {debug,info,warning}]
<!---
## Arguments
### Quick reference table

| Flag                 | Default                | Description |
|:--------------------|:----------------------|:-----------|
| [``--db``](#markdown-header--db)        |                        | Database filename.


### Descriptions
#### `--db`
Database
--->

# Change log
### Version 0.0.1
Initial release.


# License
OMAmer is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

OMAmer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with OMAmer. If not, see <http://www.gnu.org/licenses/>.
