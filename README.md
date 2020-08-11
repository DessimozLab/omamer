# OMAmer

OMAmer is a novel alignment-free protein family assignment method, which limits over-specific subfamily assignments and is suited to phylogenomic databases with thousands of genomes. It is based on an innovative method using subfamily-informed k-mers for alignment-free mapping to ancestral protein subfamilies. Whilst able to reject non-homologous family-level assignments, it has provided better and quicker subfamily-level assignments than a method based on closest sequences (using DIAMOND).

# Installation
Requires Python >= 3.6. Download the package from the PyPI, resolving the dependencies by using ``pip install omamer``.

Alternatively, clone this repository and install manually.

# Pre-Built Databases

Download pre-built databases from the links below. More databases to follow shortly -- if you have a request, please create an issue.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3978356.svg)](https://doi.org/10.5281/zenodo.3978356)
 - Metazoa (from January 2020 OMA release)
 - Hominidae (from January 2020 OMA release)

# omamer mkdb - Building a Database
This is currently reliant on the OMA browser's database file and the species phylogeny of HOGs. Building using OrthoXML files available shortly. 
 - https://omabrowser.org/All/OmaServer.h5
 - https://omabrowser.org/All/speciestree.nwk
## Usage
Required arguments: ``--db``, ``--root_taxon``, ``--oma_path``

    usage: omamer mkdb [-h] --db DB [--nthreads NTHREADS] [--k K]
                       [--hidden_taxa HIDDEN_TAXA] [--min_hog_size MIN_HOG_SIZE]
                       --root_taxon ROOT_TAXON --oma_path OMA_PATH
                       [--log_level {debug,info,warning}]
                     
## Arguments
| Flag                 | Default                | Description |
|:--------------------|:----------------------|:-----------|
| [``--db``](#markdown-header--db) || Database filename.
| [``--nthreads``](#markdown-header--nthreads)|Number of threads available|Number of threads to use.
| [``--k``](#markdown-header--k)|6|_k_-mer length.
| [``--hidden_taxa``](#markdown-header--hidden_taxa)||Proteins of these taxa are hidden from the _k_-mer table.
| [``--min_hog_size``](#markdown-header--min_hog_size)|6|HOGs passing this threshold are used.
| [``--root_taxon``](#markdown-header--root_taxon)||HOGs defined at, or descending from, this taxon are uses.
| [``--oma_path``](#markdown-header--oma_path)||Path to both OmaServer.h5 and speciestree.nwk.
| [``--log_level``](#markdown-header--log_level)|info|Logging level.

# omamer search - Searching a Database
Assign proteins to families and subfamilies in a pre-existing database.
## Usage
Required arguments: ``--db``, ``--query``

    usage: omamer search [-h] --db DB --query QUERY [--threshold THRESHOLD]
                         [--out OUT] [--include_extant_genes]
                         [--chunksize CHUNKSIZE] [--nthreads NTHREADS]
                         [--log_level {debug,info,warning}]
                     
## Arguments
### Quick reference table

| Flag                 | Default                | Description |
|:--------------------|:----------------------|:-----------|
| [``--db``](#markdown-header--db) || Database filename.
| [``--query``](#markdown-header--query) || FASTA formatted sequences.
| [``--threshold``](#markdown-header--threshold) |0.05| Subfamily-score threshold used to vary placement specificity.
| [``--out``](#markdown-header--db) |stdout| Path to output.
| [``--include_extant_genes``](#markdown-header--include_extant_genes)||Include extant gene IDs as comma separated entry in results.
| [``--chunksize``](#markdown-header--chunksize) |500| Number of sequences searched sequentially.
| [``--nthreads``](#markdown-header--db) |Number of threads available|Number of threads to use.
| [``--log_level``](#markdown-header--db) |info| Logging level.

# Output columns

#### Qseqid
The sequence identifier from the input fasta.

#### Family and Subfamily
The identifier of the hierarchical orthologous group (HOG) in OMA, which you can access through the OMA browser's REST API (https://omabrowser.org/api/docs). 

A HOG identifier is composed of the root-HOG identifier (following “HOG:” and before the first dot), which is followed by its sub-HOGs (before each subsequent dot). For example, for subfamily HOG:0487954.3l.27l, HOG:0487954 is the root-HOG (HOG without-parent), HOG:0487954.3l is its child and HOG:0487954.3l.27l its grandchild.

#### Family-score
The number of unique _k_-mers shared between the family and query divided by the number of unique query _k_-mers.

#### Subfamily-score
Similar but with the _k_-mers specific to the sub-family (~unique to that subtree in the family tree) instead of the family _k_-mers.

# Change log
#### Version 0.0.1
 - Initial release.

#### Version 0.1.0
 - Added hidden_taxa and threshold arguments.
 
#### Version 0.1.2
 - Debugging

# License
OMAmer is a free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

OMAmer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with OMAmer. If not, see <http://www.gnu.org/licenses/>.

# Citation
Rossier Victor, Alex Warwick Vesztrocy, Marc Robinson-Rechavi, and Christophe Dessimoz. 2020. “OMAmer: Tree-Driven and Alignment-Free Protein Assignment to Subfamilies Outperforms Closest Sequence Approaches.” <bioRxiv. https://doi.org/10.1101/2020.04.30.068296>.

Code used for that paper is available here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3978356.svg)](https://doi.org/10.5281/zenodo.3978356)

