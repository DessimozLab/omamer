# OMAmer

OMAmer is a novel alignment-free protein family assignment method, which limits over-specific subfamily assignments and is suited to phylogenomic databases with thousands of genomes. It is based on an innovative method using evolutionnary-informed k-mers for alignment-free mapping to ancestral protein subfamilies. Whilst able to reject non-homologous family-level assignments, it has provided better and quicker subfamily-level assignments than a method based on closest sequences (using DIAMOND).

# Installation
Requires Python >= 3.6. Download the package from the PyPI, resolving the dependencies by using ``pip install omamer``.

Alternatively, clone this repository and install manually.

# Pre-Built Databases

Download pre-built databases from the link below (from January 2020 OMA release).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4593702.svg)](https://doi.org/10.5281/zenodo.4593702)
 - LUCA.h5
 - Metazoa.h5
 - Viridiplantae.h5
 - Hominidae.h5

Their names indicate the root-taxon parameter used. Other non-required parameters were left to default. 

# omamer mkdb - Building a Database
This is currently reliant on the OMA browser's database file and the species phylogeny of HOGs. Building using OrthoXML files available shortly. 
 - https://omabrowser.org/All/OmaServer.h5
 - https://omabrowser.org/All/speciestree.nwk
## Usage
Required arguments: ``--db``, ``--oma_path``

    usage: omamer mkdb [-h] --db DB [--nthreads NTHREADS] [--min_fam_size MIN_FAM_SIZE] [--min_fam_completeness MIN_FAM_COMPLETENESS] [--logic {AND,OR}]
                       [--root_taxon ROOT_TAXON] [--hidden_taxa HIDDEN_TAXA] [--species SPECIES] [--reduced_alphabet] [--k K] --oma_path OMA_PATH
                       [--log_level {debug,info,warning}]
                   
## Arguments
| Flag                 | Default                | Description |
|:--------------------|:----------------------|:-----------|
| [``--db``](#markdown-header--db)||Path to new database (including filename)
| [``--nthreads``](#markdown-header--nthreads)|1|Number of threads to use
| [``--min_fam_size``](#markdown-header--min_fam_size)|6|Only root-HOGs with a protein count passing this threshold are used.
| [``--min_fam_completeness``](#markdown-header--min_hog_size)|0.0|Only root-HOGs passing this threshold are used. The completeness of a HOG is defined as the number of observed species divided by the expected number of species at the HOG taxonomic level.
| [``--logic``](#markdown-header--min_hog_size)|AND|Logic used between the two above arguments to filter root-HOGs. Options are "AND" or "OR".
| [``--root_taxon``](#markdown-header--root_taxon)|LUCA|HOGs defined at, or descending from, this taxon are uses as root-HOGs.
| [``--hidden_taxa``](#markdown-header--hidden_taxa)||The proteins from these taxa are removed before the database computation. Usage: a list of comma-separated taxa (scientific name) with underscore replacing spaces (_e.g._ Bacteria,Homo_sapiens).
| [``--species``](#markdown-header--species)||Temporary option
| [``--reduced_alphabet``](#markdown-header--reduced_alphabet)||Use reduced alphabet from Linclust paper
| [``--k``](#markdown-header--k)|6|_k_-mer length
| [``--oma_path``](#markdown-header--oma_path)||Path to a directory with both OmaServer.h5 and speciestree.nwk
| [``--log_level``](#markdown-header--log_level)|info|Logging level

# omamer search - Searching a Database
Assign proteins to families and subfamilies in a pre-existing database.
## Usage
Required arguments: ``--db``, ``--query``

    usage: omamer search [-h] --db DB --query QUERY [--score {default,sensitive}] [--threshold THRESHOLD] [--reference_taxon REFERENCE_TAXON] [--out OUT]
                         [--include_extant_genes] [--chunksize CHUNKSIZE] [--nthreads NTHREADS] [--log_level {debug,info,warning}]

## Arguments
### Quick reference table

| Flag                 | Default                | Description |
|:--------------------|:----------------------|:-----------|
| [``--db``](#markdown-header--db) || Path to existing database (including filename)
| [``--query``](#markdown-header--query) || Path to FASTA formatted sequences
| [``--score``](#markdown-header--score) |default| Type of OMAmer-score to use. Options are "default" and "sensitive".
| [``--threshold``](#markdown-header--threshold) |0.05| Threshold applied on the OMAmer-score that is used to vary the specificity of predicted HOGs. The lower the theshold the more (over-)specific predicted HOGs will be. 
| [``--reference_taxon``](#markdown-header--reference_taxon) || The placement is stopped when reaching a HOG with the reference taxon (must exist in the OMA database).  This is a complementary option to vary the specificity of predicted HOGs. 
| [``--out``](#markdown-header--db) |stdout| Path to output (default stdout)
| [``--include_extant_genes``](#markdown-header--include_extant_genes)||Include extant gene IDs as comma separated entry in results
| [``--chunksize``](#markdown-header--chunksize) |10000| Number of queries to process at once.
| [``--nthreads``](#markdown-header--db) |1|Number of threads to use
| [``--log_level``](#markdown-header--db) |info| Logging level

# Output columns

#### Query sequence identifier
The sequence identifier from the input fasta

#### Predicted HOG identifier
The identifier of the hierarchical orthologous group (HOG) in OMA, which you can access through the OMA browser search bar or its REST API (https://omabrowser.org/api/docs). 

A HOG identifier is composed of the root-HOG identifier (following “HOG:” and before the first dot), which is followed by its sub-HOGs (before each subsequent dot). For example, for subfamily HOG:0487954.3l.27l, HOG:0487954 is the root-HOG (HOG without-parent), HOG:0487954.3l is its child and HOG:0487954.3l.27l its grandchild.

#### Closest taxon from reference taxon
The taxon from the predicted HOG that is closest from the reference taxon (given one was provided). This option provides a mean to evaluate the performance of OMAmer placement given some knowledge of the query taxonomy is available.

#### Overlap-score
The fraction of the query sequence overlapping with k-mers of reference root-HOGs. This score aims to help reject partial homologous matches that are problematic in some applications.

#### Family-level OMAmer-score
The OMAmer-score of the predicted root-HOG. At the family level, this score measures the sequence similarity between the query and a given root-HOG.

#### Subfamily-level OMAmer-score
The OMAmer-score of the predicted HOG. At the subfamily level, this score captures the excess of similarity that is shared between the query and a given HOG, thus excluding the similarity with regions conserved in more ancestral HOGs.

#### Subfamily gene set
Extant gene IDs of predicted HOG, which you can look for in the OMA browser search bar or its REST API (https://omabrowser.org/api/docs). 

# Change log

#### Version 0.2.2
 - Automated deployment to PyPI
 - Removed PyHAM dependency

#### Version 0.2.0
 - Added ``--min_fam_completeness``, ``--logic``, ``--score`` and ``--reference_taxon`` options
 - New output format
 - Debugging

#### Version 0.1.2 - 0.1.3
 - Debugging

#### Version 0.1.0
 - Added hidden_taxa and threshold arguments

#### Version 0.0.1
 - Initial release

# License
OMAmer is a free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

OMAmer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with OMAmer. If not, see <http://www.gnu.org/licenses/>.

# Citation
Victor Rossier, Alex Warwick Vesztrocy, Marc Robinson-Rechavi, Christophe Dessimoz, OMAmer: tree-driven and alignment-free protein assignment to subfamilies outperforms closest sequence approaches, Bioinformatics, 2021;, btab219, https://doi.org/10.1093/bioinformatics/btab219

Code used for that paper is available here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4593702.svg)](https://doi.org/10.5281/zenodo.4593702)


