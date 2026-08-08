
# Change log

## Unreleased

### Added

- `set-kmer-filter`, `refit-bbinom` and `import-bbinom`: re-derive family models
  on a built database instead of rebuilding it. Neither the PMI k-mer filter nor
  the beta-binomial design affects the k-mer tables, so one `mkdb` can serve a
  whole sweep of filters and N grids.
- `mkdb --training_buffers_out`: write the sequence and 3Di buffers to a sidecar
  so a later refit can rebuild each null query's k-mer set. Only `mkdb` can
  produce it -- the database stores k-mer tables, not sequences.
- `refit-bbinom --family_shard i/n`: fit one contiguous shard of families, for
  fanning a large build out over nodes. Shards draw the same null sample from
  the same seed, so sharded and whole runs agree exactly.
- `refit-bbinom --n_counts_cache`: cache the per-record exact-N scan. Keyed by
  the k-mer filter, and refuses to be reused across filters.
- `mkdb --bbinom_seq_n_values`, alongside the existing `--bbinom_ss_n_values`.

### Changed

- `import-bbinom` accepts several coefficient files and merges them, rejecting
  overlapping shards. Stored arrays cover every family, so a partial import
  would blank the rest.
- The stored `models` attribute is now maintained when coefficients are added or
  dropped, so `info` cannot advertise a model the database does not carry.

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