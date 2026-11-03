
# Change log

## Unreleased

### Changed

- Support numpy 2 (numpy >= 1.26 and pytables >= 3.10.1 are now required)
- Python >= 3.10 is now required (support for Python 3.8 and 3.9 is dropped,
  as pytables >= 3.10 no longer supports them)
- `mkdb` stores the HOGs of a family level in a fixed order, so repeated builds
  of the same input produce identical databases
- `search --include_extant_genes` no longer reloads the database tables for
  every result row, which made it very slow on large databases
- CI runs the test suite with Python 3.12 and 3.14 and with numpy 1 and 2,
  including search results checked against golden files
- `mkdb --orthoxml` adds the taxonomic levels that a HOG spans implicitly, as
  pyoma does for the OMA browser. A subHOG now starts at the level of the
  duplication that created it, and nested duplications get their intermediate
  HOG, so FastOMA output gives the same database as the OMA browser build

### Fixed

- Python 3.13+: install `legacy-cgi`, which ete3 still needs
- `mkdb` no longer runs out of memory on HOGs with a duplication nested
  directly in another duplication (e.g. FastOMA output), where an intermediate
  HOG such as `HOG:1.1a` is missing for `HOG:1.1a.1a`. Such HOGs were made
  their own parent, so the k-mer table computation looped forever.
- `mkdb --orthoxml` stores the NCBI taxon ids of the OrthoXML species
- `mkdb --reduced_alphabet` built a wrong k-mer table: k-mers that are equal
  in the reduced alphabet, but not in the original sequences (e.g. `AS` and
  `AT`), were stored as several runs, with duplicate and too specific HOG
  entries. Searches with such databases placed many queries in the wrong
  family. Databases built with `--reduced_alphabet` must be rebuilt.

## Version 2.1.2

### Fixed

- An important bugfix to the search function producing invalid results in 2.1.1: #57
- Fixed incompatibility with python 13 (#53) 
- Fixed a crash when empty fasta if provided (#58)


### Changed

- Updated dependencies to Github actions

## Version 2.1.1

- Performance improvements to the mkdb command with orthoxml input
- Added a check for non-unique protein IDs in the input fasta files. Now it gives a more informative error message
- fixed #49

## Version 2.1.0
- Significant improvements to classification speed 

## Version 2.0.4
- Fixes issue #34 (numpy2 incompatibility)
- Experimental support to build omamer databases from orthoxml/fasta files
- Updated github action to latest versions

## Version 2.0.3
- Fixes issue #30
- Update github action to latest versions

## Version 2.0.2
- changed method for hiding taxa in build process. Now takes a file containing taxa to hide on separate lines.
- checks and improved feedback for root taxon and requested taxa to hide.
- root taxon set by default to the root level in speciestree.nwk (previously hard-coded to default to LUCA)

## Version 2.0.1
 - remove dependency for filehash library
 - return better error message if build dependencies are not met, but trying to building an omamer database
 - minor fixes

## Version 2.0.0
 - Major update of database format and search code to improve overall memory useage. Most standard runs with LUCA-level database will run on a machine with 16GB RAM.
 - Update to the scoring algorithm for root-level HOG / family assignments, to allow for significance testing. This estimates a binomial distribution for each family, so that we can compute the probability of matching at least as many k-mers as we have observed by chance, for each family that has a match to a given query.
 - UX improvements - more feedback during interactive search runs, whilst maintaining small log files.

## Version 0.2.5
 - Fixes an issue when storing the pre-conputed statistics

## Version 0.2.4
 - Improved loading time for standard search by pre-computing statistics
 - Adding new command line option "info" to show the metadata of the 
   dataset used to build the omamer database.
   

## Version 0.2.2
 - Automated deployment to PyPI
 - Removed PyHAM dependency

## Version 0.2.0
 - Added ``--min_fam_completeness``, ``--logic``, ``--score`` and ``--reference_taxon`` options
 - New output format
 - Debugging

## Version 0.1.2 - 0.1.3
 - Debugging

## Version 0.1.0
 - Added hidden_taxa and threshold arguments

## Version 0.0.1
 - Initial release