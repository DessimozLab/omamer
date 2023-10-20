# OMAmer - tree-driven and alignment-free protein assignment to subfamilies

OMAmer is a novel alignment-free protein family assignment method, which limits over-specific subfamily assignments and is suited to phylogenomic databases with thousands of genomes. It is based on an innovative method using evolutionary-informed _k_-mers for alignment-free mapping to ancestral protein subfamilies. Whilst able to reject non-homologous family-level assignments, it has provided better and quicker subfamily-level assignments than a method based on closest sequences (using DIAMOND).

# Installation
Requires Python >= 3.8. Download the package from the PyPI, resolving the dependencies by using ``pip install omamer``.

Alternatively, clone this repository and install manually.

Note: Python 3.12 is currently not supported, until the `numba` package is updated ([issue](https://github.com/numba/numba/issues/9197)).

# Pre-Built Databases

Pre-built databases are available for the latest OMA release from the [download section on the OMA Browser website](https://omabrowser.org/oma/current).

 - LUCA: https://omabrowser.org/All/LUCA.h5
 - _Metazoa: https://omabrowser.org/All/Metazoa.h5
 - _Viridiplantae_: https://omabrowser.org/All/Viridiplantae.h5
 - _Saccharomyceta_: https://omabrowser.org/All/Saccharomyceta.h5 
 - _Primates_: https://omabrowser.org/All/Primates.h5

Their names indicate the root-taxon parameter used. Other non-required parameters were left to default.

Note: databases included in the [Zenodo upload](https://zenodo.org/record/4593702) from the manuscript are not supported by the most recent version of OMAmer. We recommend using the most recent release with databases built on the most recent OMA browser release.



# omamer search - Searching a Database
Assign proteins to families and subfamilies in a pre-existing database.
## Usage
Required arguments: ``--db``, ``--query``

    usage: omamer search [-h] -d DB -q QUERY [--threshold THRESHOLD] [--family_alpha FAMILY_ALPHA] [-fo] [-n TOP_N_FAMS] [--reference_taxon REFERENCE_TAXON]
                     [-o OUT] [--include_extant_genes] [-c CHUNKSIZE] [-t {0,1,2,3,4,5,6,7,8}] [--log_level {debug,info,warning}] [--silent]

## Arguments
### Quick reference table

| Short Flag | Flag                 | Default                | Description |
|:-----------|:---------------------|:-----------------------|:------------|
| [``-d``](#markdown-header-d) | [``--db``](#markdown-header--db) || Path to existing database (including filename)
| [``-q``](#markdown-header-q) | [``--query``](#markdown-header--query) || Path to FASTA formatted sequences
| | [``--threshold``](#markdown-header--threshold) | 0.1 | Threshold applied on the OMAmer-score that is used to vary the specificity of predicted HOGs. The lower the theshold the more (over-)specific predicted HOGs will be.
| | [``--family_alpha``](#markdown-header--family_alpha) | 1e-6 | Significance threshold used when filtering families.
| [``-fo``](#markdown-header-fo) | [``--family_only``](#markdown-header--family_only) | False | If set, only place at the family level. Useful for certain analysis. Note: `subfamily_medianseqlen` in the results is for the family level.
| [``-n``](#markdown-header-n) | [``--family_only``](#markdown-header--top_n_fams) | 1 | Number of top level families to place into. By default, placed into only the best scoring family.
<!--| | [``--reference_taxon``](#markdown-header--reference_taxon) || The placement is stopped when reaching a HOG with the reference taxon (must exist in the OMA database).  This is a complementary option to vary the specificity of predicted HOGs.-->
| [``-o``](#markdown-header-o) | [``--out``](#markdown-header--db) | stdout | Path to output. If not set, defaults to stdout.
| | [``--include_extant_genes``](#markdown-header--include_extant_genes)||Include extant gene IDs as comma separated entry in results
| [``-c``](#markdown-header-c) | [``--chunksize``](#markdown-header--chunksize) |10000| Number of queries to process at once.
| [``-t``](#markdown-header-t) | [``--nthreads``](#markdown-header--db) |1|Number of threads to use
| | [``--log_level``](#markdown-header--db) |info| Logging level (options debug, info, warning)
| | [``--silent``](#markdown-header--silent) || Set to silence the output.

# Output

Output is in the form of a tab-seperated value file (TSV), with metadata added to the header using ``!<tag>: <value>``. A parser can be imported for further analysis in python as ``from omamer.results_reader import results_reader``.

## Output Columns

#### Query sequence identifier (`qseqid`)
The sequence identifier from the input FASTA-formatted sequences.

#### Predicted HOG identifier (`hogid`)
The identifier of the hierarchical orthologous group (HOG) in OMA, which you can access through the OMA browser search bar or its REST API (https://omabrowser.org/api/docs). 

A HOG identifier is composed of the root-HOG identifier (following “HOG:” and before the first dot), which is followed by its sub-HOGs (before each subsequent dot). For example, for subfamily HOG:0487954.3l.27l, HOG:0487954 is the root-HOG (HOG without-parent), HOG:0487954.3l is its child and HOG:0487954.3l.27l its grandchild.

#### Predicted HOG taxonomic level (`hoglevel`)
The taxonomic level that the predicted HOG is defined at.

#### Family p-value (`family_p`)
p-value of having as many or more of k-mers in common under a binomial distribution. Reported in negative natural log units.

#### Family count (`family_count`)
Count of _k_-mers in common with the family / root level HOG.

#### Family normalised count (`family_normcount`)
Family count, normalised by the expected number of hits for the query's sequence length, with the family's _k_-mer content.

#### Sub-Family score (`subfamily_score`)
The OMAmer-score of the predicted HOG. At the subfamily level, this score captures the excess of similarity that is shared between the query and a given HOG, thus excluding the similarity with regions conserved in more ancestral HOGs.

#### Sub-Family count (`subfamily_count`)
Count of _k_-mers in common with the sub-family / HOG.

#### Query sequence length (`qseqlen`)
Count of _k_-mers in common with the sub-family / HOG.

#### Sub-Family median sequence length (`subfamily_medianseqlen`)
Median length of the sequences that are present in the predicted HOG. In the case of family-only placement, this is instead reported at the root-HOG level.

#### Query sequence overlap (`qseq_overlap`)
The proportion of the query sequence overlapping with _k_-mers of reference root-HOGs. This may be helpful to reject partially homologous matches that are problematic in some applications.

#### Sub-Family gene set (`subfamily_geneset`)
Optionally printed (see ``--include_extant_genes``). Comma-seperated list of extant gene IDs of predicted HOG. The [OMA browser](https://omabrowser.org) can be used to find out more information. In particular, using the [REST API](https://omabrowser.org/api/docs), or via the [Python API Client](https://github.com/DessimozLab/pyomadb).

<!-- #### Closest taxon from reference taxon (`closest_taxa`)
The taxon from the predicted HOG that is closest from the reference taxon (given one was provided). This option provides a mean to evaluate the performance of OMAmer placement given some knowledge of the query taxonomy is available.
-->


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


# Change log

#### Version 2.0.0
 - Major update of database format and search code to improve overall memory useage. Most standard runs with LUCA-level database will run on a machine with 16GB RAM.
 - Update to the scoring algorithm for root-level HOG / family assignments, to allow for significance testing. This estimates a binomial distribution for each family, so that we can compute the probability of matching at least as many k-mers as we have observed by chance, for each family that has a match to a given query.
 - UX improvements - more feedback during interactive search runs, whilst maintaining small log files.

#### Version 0.2.5
 - Fixes an issue when storing the pre-conputed statistics

#### Version 0.2.4
 - Improved loading time for standard search by pre-computing statistics
 - Adding new command line option "info" to show the metadata of the 
   dataset used to build the omamer database.
   

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
