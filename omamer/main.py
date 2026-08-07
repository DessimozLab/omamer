"""
    OMAmer - tree-driven and alignment-free protein assignment to sub-families

    (C) 2024-present Nikolai Romashchenko <nikolai.romashchenko@unil.ch>
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


def main():
    from argparse import (
        ArgumentParser,
        HelpFormatter,
        _SubParsersAction,
        FileType,
        ArgumentDefaultsHelpFormatter,
    )
    from tables import PerformanceWarning
    import multiprocessing as mp
    import os
    import sys
    import warnings

    from . import __version__, __copyright__
    from ._runners import mkdb_oma, search, info_db

    class NoSubparsersMetavarFormatter(HelpFormatter):
        def _format_action(self, action):
            result = super()._format_action(action)
            if isinstance(action, _SubParsersAction):
                # fix indentation on first line
                return "%*s%s" % (self._current_indent, "", result.lstrip())
            return result

        def _format_action_invocation(self, action):
            if isinstance(action, _SubParsersAction):
                # remove metavar and help line
                return ""
            return super()._format_action_invocation(action)

        def _iter_indented_subactions(self, action):
            if isinstance(action, _SubParsersAction):
                try:
                    get_subactions = action._get_subactions
                except AttributeError:
                    pass
                else:
                    # remove indentation
                    yield from get_subactions()
            else:
                yield from super()._iter_indented_subactions(action)

    def get_thread_count():
        if hasattr(os, "sched_getaffinity"):
            #  works for schedulers, e.g., slurm
            return len(os.sched_getaffinity(0))
        else:
            return mp.cpu_count()

    desc = "OMAmer - tree-driven and alignment-free protein assignment to sub-families."
    parser = ArgumentParser(
        formatter_class=NoSubparsersMetavarFormatter,
        prog="omamer",
        description=desc,
        epilog=__copyright__,
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        help="Show version and exit.",
        version=__version__,
    )

    subparsers = parser.add_subparsers(title="Commands")

    mkdb_parser = subparsers.add_parser(
        "mkdb",
        formatter_class=ArgumentDefaultsHelpFormatter,
        help="Build a database.",
        description="Build a database, by providing an OMA HDF5 database file [BROWSERBUILD] or OrthoXML + FASTA + Newick files [OXMLBUILD].",
    )
    mkdb_parser.set_defaults(func=mkdb_oma)
    mkdb_parser.add_argument(
        "-d", "--db", required=True, help="Path to new database (including filename)."
    )
    mkdb_parser.add_argument(
        "-t",
        "--nthreads",
        default=1,
        type=int,
        choices=range(get_thread_count() + 1),
        help="Number of threads to use",
    )
    mkdb_parser.add_argument(
        "--min_fam_size",
        default=6,
        help="Only root-HOGs with a protein count passing this threshold are used.",
        type=int,
    )
    mkdb_parser.add_argument(
        "--min_fam_completeness",
        default=0.5,
        help="Only root-HOGs passing this threshold are used. The completeness of a HOG is defined as the number of observed species divided by the expected number of species at the HOG taxonomic level",
        type=float,
    )
    mkdb_parser.add_argument(
        "--logic",
        default="OR",
        help="Logic used between the two above arguments to filter root-HOGs. Options are AND or OR.",
        choices=["AND", "OR"],
    )
    mkdb_parser.add_argument(
        "--root_taxon",
        required=False,
        help="HOGs defined at, or descending from, this taxon are uses as root-HOGs. Default is the top level in species tree.",
    )
    mkdb_parser.add_argument(
        "--hidden_taxa",
        required=False,
        help="Optional -- path to a file giving a list of taxa to hide the proteins during index creation only. HOGs will still exist in the database, but it will not be possible to place to them. Names must match EXACTLY to those in the given newick species tree.",
        type=FileType("r"),
    )
    # mkdb_parser.add_argument(
    #    "--species", default="", help="Alternatively to --hidden_taxa, provide a file with species offsets in sp_tab (tmp option for scaling experiment)", type=str
    # )
    mkdb_parser.add_argument(
        "--reduced_alphabet",
        default=False,
        action="store_true",
        help="Use reduced alphabet from Linclust paper.",
    )
    mkdb_parser.add_argument("--k", default=6, help="k-mer length", type=int)
    mkdb_parser.add_argument(
        "--kmer_percentage",
        default=100.0,
        type=float,
        help="Retain the most informative percentage of indexed sequence and 3Di "
             "k-mer types, ranked by pointwise mutual information / inverse family "
             "document frequency. The setting is stored in the database and forced "
             "at search time; 100 disables filtering.",
    )
    mkdb_parser.add_argument(
        "--models",
        nargs="+",
        choices=("binomial", "beta-binomial"),
        default=("binomial", "beta-binomial"),
        help="Statistical family models to learn for every indexed modality. "
        "Beta-binomial requires binomial as its fallback.",
    )
    mkdb_parser.add_argument(
        "--bbinom_ss_n_values",
        help="Comma- or space-separated exact unique-k-mer counts used for "
        "3Di beta-binomial fitting. If omitted, buckets are selected from "
        "the database structures.",
    )
    mkdb_parser.add_argument(
        "--bbinom_n_buckets",
        default=24,
        type=int,
        help="Number of exact unique-k-mer-count buckets used for beta-binomial fitting.",
    )
    mkdb_parser.add_argument(
        "--bbinom_min_records_per_n",
        default=50,
        type=int,
        help="Minimum source proteins required for an exact-N fitting bucket.",
    )
    mkdb_parser.add_argument(
        "--bbinom_max_records_per_n",
        default=500,
        type=int,
        help="Maximum sampled source proteins per exact-N fitting bucket; 0 uses all.",
    )
    mkdb_parser.add_argument(
        "--bbinom_min_nonzero_queries",
        default=20,
        type=int,
        help="Minimum nonzero null hits required to fit one family.",
    )
    mkdb_parser.add_argument(
        "--bbinom_max_families",
        default=0,
        type=int,
        help="Optional random fitting cap for debugging; 0 fits every family.",
    )
    mkdb_parser.add_argument(
        "--bbinom_min_family_prob",
        default=0.0,
        type=float,
        help="Fit families whose binomial probability is at least this value.",
    )
    mkdb_parser.add_argument(
        "--bbinom_fit_workers",
        default=1,
        type=int,
        help="Worker processes used for per-family beta-binomial fits.",
    )
    mkdb_parser.add_argument(
        "--bbinom_max_histogram_gb",
        default=3.0,
        type=float,
        help="Maximum RAM for one beta-binomial histogram batch; 0 is unbounded.",
    )
    mkdb_parser.add_argument(
        "--bbinom_seed",
        default=42,
        type=int,
        help="Random seed for beta-binomial source-protein sampling.",
    )
    mkdb_parser.add_argument(
        "--oma_path",
        help="Path to OMA browser release (must include OmaServer.h5 and speciestree.nwk). [BROWSERBUILD]",
    )
    mkdb_parser.add_argument(
        "--orthoxml",
        help="Path to OrthoXML file containing HOGs. [OXMLBUILD]",
        type=FileType("r")
    )
    mkdb_parser.add_argument(
        "--species_tree",
        help="Path to newick file containing species tree. [OXMLBUILD]",
        type=FileType("r")
    )
    mkdb_parser.add_argument(
        "--sequences",
        nargs='*',
        help="Paths to sequence files (1 or multiple, only for non-browser build). [OXMLBUILD]",
        type=FileType("r")
    )
    mkdb_parser.add_argument(
        "--structures",
        nargs='*',
        help="""Paths to 3di structure files (1 or multiple) for OXMLBUILD. 
        Provide exactly one 3di HDF5 file for BROWSERBUILD.""",
        type=FileType("r")
    )
    mkdb_parser.add_argument(
        "--log_level",
        default="info",
        choices=["debug", "info", "warning"],
        help="Logging level.",
    )

    search_parser = subparsers.add_parser(
        "search",
        formatter_class=ArgumentDefaultsHelpFormatter,
        help="Search an existing database.",
        description="Search for protein sequences, given in FASTA format, against an existing database.",
    )
    search_parser.set_defaults(func=search)
    search_parser.add_argument(
        "-d",
        "--db",
        required=True,
        help="Path to existing database (including filename).",
    )
    search_parser.add_argument(
        "-q",
        "--query",
        required=False,
        help="Path to FASTA formatted sequences",
        type=str,
    )

    search_parser.add_argument(
        "-s",
        "--structure",
        required=False,
        help="Path to FASTA formatted 3di sequences",
        type=str,
    )
    search_parser.add_argument(
        "--search_mode",
        choices=("auto", "seq", "ss", "sqs"),
        default="auto",
        help="Query strategy: sequence, structure, or sequence with structure fallback.",
    )
    search_parser.add_argument(
        "--family_model",
        choices=("auto", "binomial", "beta-binomial"),
        default="auto",
        help="Family significance model. Auto uses stored beta-binomial coefficients when available.",
    )

    search_parser.add_argument(
        "--threshold",
        default=0.10,
        type=float,
        help="Threshold applied on the OMAmer-score that is used to vary the specificity of predicted HOGs. The lower the theshold the more (over-)specific predicted HOGs will be.",
    )
    search_parser.add_argument(
        "--family_alpha",
        default=1e-6,
        type=float,
        help="Significance threshold used when filtering families.",
    )
    search_parser.add_argument(
        "--family_correction",
        choices=("bonferroni", "none"),
        default="bonferroni",
        help="Multiple-testing correction applied to family p-values. "
        "'bonferroni' preserves the historical correction over all indexed "
        "families; 'none' filters raw family p-values.",
    )
    search_parser.add_argument(
        "--kmer_percentage",
        default=None,
        type=float,
        help="Optional compatibility check for the database's build-time k-mer "
             "percentage. Search always uses the value stored in the database "
             "and rejects a different value.",
    )
    search_parser.add_argument(
        "-fo",
        "--family_only",
        action="store_true",
        help="Set to only place at family level. Note: subfamily_medianseqlen in results is for the family level.",
    )
    search_parser.add_argument(
        "-n",
        "--top_n_fams",
        default=1,
        type=int,
        help="Number of top level families to place into. By default, placed into only in the best scoring family.",
    )

    search_parser.add_argument(
        "--reference_taxon",
        help="The placement is stopped when reaching the reference taxon (must exist in the OMA database).",
    )

    search_parser.add_argument(
        "-o",
        "--out",
        help="Path to output. If not set, defaults to stdout",
        type=FileType("w"),
    )
    search_parser.add_argument(
        "--include_extant_genes",
        action="store_true",
        help="Include extant gene IDs as comma separated entry in results.",
    )
    search_parser.add_argument(
        "-c",
        "--chunksize",
        default=10000,
        type=int,
        help="Number of queries to process at once.",
    )
    search_parser.add_argument(
        "-t",
        "--nthreads",
        default=1,
        type=int,
        choices=range(get_thread_count() + 1),
        help="Number of threads to use",
    )
    search_parser.add_argument(
        "--log_level",
        default="info",
        choices=["debug", "info", "warning"],
        help="Logging level.",
    )
    search_parser.add_argument("--silent", action="store_true", help="Silence output")

    info_parser = subparsers.add_parser(
        "info",
        help="Show metadata about an omamer database.",
        description="Show metadata about an existing omamer database",
    )

    info_parser.set_defaults(func=info_db)
    info_parser.add_argument(
        "-d",
        "--db",
        required=True,
        help="Path to an existing database (including filename).",
    )

    args = parser.parse_args()
    if hasattr(args, "func"):
        if hasattr(args, "log_level"):
            from omamer._utils import set_log_level, LOG

            set_log_level(args.log_level)

        if hasattr(args, "silent"):
            from omamer._utils import set_if_silent

            set_if_silent(args.silent)

        if not sys.warnoptions and not getattr(args, "log_level", "") == "debug":
            warnings.simplefilter("ignore", category=PerformanceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)

        if hasattr(args, "nthreads"):
            # set number of threads before we call any other code
            import numba

            nthreads = args.nthreads if args.nthreads > 0 else get_thread_count()
            numba.set_num_threads(nthreads)

            os.environ["MKL_NUM_THREADS"] = os.environ[
                "NUMEXPR_NUM_THREADS"
            ] = os.environ["OMP_NUM_THREADS"] = str(nthreads)

        # call the relevant runner func
        args.func(args)
    else:
        parser.print_usage()


if __name__ == "__main__":
    main()
