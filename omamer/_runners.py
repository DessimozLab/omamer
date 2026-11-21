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
import os
from ._utils import LOG, check_file_exists


def mkdb_oma(args):
    from .database import DatabaseFromOMABrowser, DatabaseFromOrthoXML
    from .index import Index
    from .species_tree import SpeciesTree, SpeciesTreeError

    import os

    _ensure_db_build_dependencies_available()

    assert args.k < 8, "Max k-mer size is 7."
    LOG.info("Create database from OMA build")
    LOG.info("arguments for build:")
    for (k, v) in vars(args).items():
        LOG.info(" - {}: {}".format(k, v))

    # work out mode
    browser_db_mode = (args.oma_path is not None)

    if browser_db_mode:
        Database = DatabaseFromOMABrowser

        # check directories and find files
        if not os.path.isdir(args.oma_path):
            raise ValueError('Can\'t find OMA PATH: {}'.format(args.oma_path))
        if os.path.isdir(os.path.join(args.oma_path, 'data')):
            # assume given root browser directory.
            args.oma_path = os.path.join(args.oma_path, 'data')

        oma_db_fn = os.path.join(args.oma_path, "OmaServer.h5")
        check_file_exists(oma_db_fn)
        nwk = os.path.join(args.oma_path, "speciestree.nwk")
        check_file_exists(nwk)
        structure_h5_fn = None
        if args.structures:
            if len(args.structures) != 1:
                raise ValueError(
                    "BROWSERBUILD expects exactly one path to 3Di HDF5 file via --structures"
                )
            structure_h5_fn = args.structures[0].name

    else:
        if args.orthoxml is None or not args.sequences:
            raise ValueError('Must pass either browser database (--oma_path) or all files for OrthoXML build (--orthoxml --sequences, and --species_tree unless the OrthoXML file contains the taxonomy)')

        # orthoxml mode
        Database = DatabaseFromOrthoXML

        # find files
        oxml_fn = args.orthoxml.name
        fasta_fns = list(map(lambda x: x.name, args.sequences))
        structure_fns = [x.name for x in args.structures] if args.structures else []
        nwk = args.species_tree.name if args.species_tree is not None else None

    # read the species tree, from the OrthoXML file if none is given
    if nwk is not None:
        full_stree = SpeciesTree.read(nwk)
    else:
        try:
            full_stree = SpeciesTree.from_orthoxml(oxml_fn)
        except SpeciesTreeError:
            raise SpeciesTreeError(
                "{} contains no taxonomy (OrthoXML >= 0.5), pass the species "
                "tree with --species_tree".format(oxml_fn)
            )
        LOG.info("Using the taxonomy of the OrthoXML file as species tree")

    # check the root taxon, and that the taxa below it have unique names
    stree = full_stree
    root_taxon = args.root_taxon if args.root_taxon is not None else stree.root
    if root_taxon is None:
        raise SpeciesTreeError(
            "The root of the species tree has no name, name it or set --root_taxon"
        )
    stree = stree.subtree(root_taxon)

    if args.hidden_taxa:
        # load and check for hidden taxa
        hidden_taxa = list(map(lambda x: x.rstrip(), args.hidden_taxa))

        for ht in hidden_taxa:
            stree.check_taxon(ht, "hidden taxon")
    else:
        hidden_taxa = []

    db = Database(
        args.db,
        root_taxon=root_taxon,
        min_fam_size=args.min_fam_size,
        logic=args.logic,
        min_fam_completeness=args.min_fam_completeness,
        include_younger_fams=True,
        mode="w",
    )

    if args.exclude_families:
        db.excluded_families = {
            line.strip().encode("ascii")
            for line in args.exclude_families
            if line.strip()
        }
        LOG.info("Read {} root-HOG ids to exclude".format(len(db.excluded_families)))

    # add sequences from database
    LOG.info("Loading sequences")
    if browser_db_mode:
        seq_buff, ss_buff = db.build_database(oma_db_fn, full_stree, structure_h5_fn)
    else:
        seq_buff, ss_buff = db.build_database(oxml_fn, fasta_fns, full_stree, structure_fns)

    LOG.info("Building index")
    db.ki = Index(
        db,
        k=args.k,
        reduced_alphabet=args.reduced_alphabet,
        hidden_taxa=hidden_taxa,
        kmer_max_df=args.seq_max_df,
        ss_kmer_max_df=args.ss_max_df,
        seq_min_distinct=args.seq_min_distinct,
        ss_min_distinct=args.ss_min_distinct,
    )
    db.ki.sp_filter
    db.ki.build_kmer_table(seq_buff, ss_buff)
    db.add_metadata()
    db.add_md5_hash()

    db.close()
    LOG.info("Done")


def search(args):
    from alive_progress import alive_bar
    from ._utils import print_message
    import sys

    if args.out is None:
        args.out = sys.stdout

    # display welcome info
    welcome()

    print_message("")
    with alive_bar(
        title="Loading required libraries",
        bar=None,
        monitor=False,
        stats=False,
        elapsed=False,
        receipt_text=1,
        file=sys.stderr,
    ) as bar:
        from datetime import datetime
        from time import time
        import os

        import numpy as np

        from . import __version__
        from ._utils import compute_file_md5
        from .database import Database
        from .merge_search import MergeSearch
        from .sequence_reader import SequenceReader

        bar.text(" [DONE]")

    print_run_data(args)
    check_args(args)

    t0 = time()

    # reload
    db = Database(args.db)
    if (
        args.structure or args.search_mode in ("ss", "sqs", "sum", "hybrid")
    ) and not db.has_structure():
        raise RuntimeError(
            "The database {} has no 3Di index (it was built without "
            "--structures): search it without --structure, in the search mode "
            "'auto' or 'seq'".format(args.db)
        )

    # setup search
    ms = MergeSearch(ki=db.ki, include_extant_genes=args.include_extant_genes)

    # only print header for file output
    print_header = args.out.name != sys.stdout.name

    # find reference taxon if set
    ref_taxon = (
        args.reference_taxon.encode("ascii")
        if args.reference_taxon is not None
        else None
    )
    if ref_taxon is not None:
        tax_ids = db._db_Taxonomy.col("ID")
        ref_taxoff = np.searchsorted(tax_ids, ref_taxon)
        assert tax_ids[ref_taxoff] == ref_taxon, "Cannot identify {}".format(ref_taxon)
    else:
        ref_taxoff = None

    _ensure_data_loaded(
        ms, load_structure=bool(args.structure) and args.search_mode != "seq"
    )

    search_times = []

    search_pbar_kwargs = {
        "title": "Searching - ",
        "unit": " queries",
        "disable": args.silent,
        "file": sys.stderr,
    }
    if not os.isatty(0):
        # non-interactive mode, attempt to give some feedback in log files
        search_pbar_kwargs["refresh_secs"] = 5
        search_pbar_kwargs["force_tty"] = True
        search_pbar_kwargs["bar"] = search_pbar_kwargs["spinner"] = False

    print_message("")
    with alive_bar(**search_pbar_kwargs) as pbar:
        def read(filename):
            return SequenceReader.read(
                filename,
                k=db.ki.k,
                format="fasta",
                chunksize=args.chunksize,
                sanitiser=db.ki.alphabet.sanitise_seq,
            )

        # the sequence and 3Di files are read in lock-step chunks
        query_iter = read(args.query) if args.query else None
        struct_iter = read(args.structure) if args.structure else None
        for ids, seqs in query_iter if query_iter is not None else struct_iter:
            struct_seqs = []
            if query_iter is None:
                (seqs, struct_seqs) = ([], seqs)
            elif struct_iter is not None:
                (struct_ids, struct_seqs) = next(struct_iter, (None, None))
                if struct_ids is None:
                    raise RuntimeError(
                        "The structure file has fewer records than the query file"
                    )
                if not np.array_equal(ids, struct_ids):
                    raise RuntimeError(
                        "Query and structure IDs must match, record by record "
                        "(records shorter than k are skipped in both files)"
                    )

            t_search0 = time()
            df = ms.merge_search(
                seqs=seqs,
                ids=ids,
                top_n_fams=args.top_n_fams,
                alpha=args.family_alpha,
                sst=args.threshold,
                family_only=args.family_only,
                ref_taxon_off=ref_taxoff,
                struct_seqs=struct_seqs,
                search_mode=args.search_mode,
                summed_alpha=args.summed_alpha,
                ss_weight=args.ss_weight,
                family_selection=args.family_selection,
                seq_kmer_df_exponent=args.seq_kmer_df_exponent,
                ss_kmer_df_exponent=args.ss_kmer_df_exponent,
                seq_kmer_df_reference=args.seq_kmer_df_reference,
                ss_kmer_df_reference=args.ss_kmer_df_reference,
            )
            t_search1 = time()

            pbar(len(ids))

            if df.size > 0:
                if print_header:
                    # write the top header
                    print("!omamer-version: {}".format(__version__), file=args.out)
                    print(
                        "!query-md5: {}".format(
                            compute_file_md5(args.query or args.structure)
                        ),
                        file=args.out,
                    )
                    print(
                        "!date-run: {}".format(datetime.fromtimestamp(t0).isoformat()),
                        file=args.out,
                    )
                    print("!db-path: {}".format(db.filename), file=args.out)
                    print(
                        "!family-selection: {}".format(args.family_selection),
                        file=args.out,
                    )
                    if ms.kmer_df_correction:
                        print(
                            "!kmer-df-correction: {}".format(
                                ",".join(
                                    "{}:b={:g},m0={:.2f}".format(m, b, m0)
                                    for m, (b, m0) in sorted(
                                        ms.kmer_df_correction.items()
                                    )
                                )
                            ),
                            file=args.out,
                        )

                    # include some of the db metadata
                    db_info = dict(_format_info_db(db))
                    for k in ["source", "root level", "database hash"]:
                        if k in db_info:
                            print(
                                "!db-info-{}: {}".format(
                                    "_".join(k.split(" ")), db_info[k]
                                ),
                                file=args.out,
                            )
                df.to_csv(
                    args.out, sep="\t", index=False, header=print_header, na_rep="N/A"
                )
                print_header = False
            search_times.append((len(ids), t_search1 - t_search0))

        if (
            query_iter is not None
            and struct_iter is not None
            and next(struct_iter, None) is not None
        ):
            raise RuntimeError(
                "The structure file has more records than the query file"
            )

    db.close()
    if args.out is not sys.stdout:
        args.out.close()

    search_rate = sum(map(lambda x: x[0], search_times)) / sum(
        map(lambda x: x[1], search_times)
    )
    goodbye(args, time() - t0, search_rate)


def _ensure_db_build_dependencies_available():
    try:
        from PySAIS import sais
    except ImportError:
        LOG.error("To build OMAmer databases, pysais must be installed. Please ensure you installed omamer with the 'build' extra, e.g. `pip install omamer[build]`")
        import sys
        sys.exit(1)


def _ensure_data_loaded(ms, load_structure=True):
    from alive_progress import alive_bar
    import sys

    from ._utils import print_message, print_line

    # ensure that the data is loaded, gives progress messages unless silenced.
    print_message("\nLoading data required for OMAmer search from database...")

    def _load(attr, title):
        with alive_bar(
            title=" - {}".format(title),
            bar=None,
            monitor=False,
            stats=False,
            elapsed="({elapsed})",
            receipt_text=1,
            file=sys.stderr,
        ) as bar:
            getattr(ms, attr)
            bar.text("[DONE]")

    ms.trans
    _load("tax_tab", "taxonomy information")
    _load("fam_tab", "family information")
    _load("hog_tab", "sub-family information")
    _load("hog_id_buff", "sub-family ids")
    _load("level_arr", "family hierarchy")
    _load("kmer_table", "k-mer index")
    _load("ref_fam_prob", "family probability estimates")
    _load("ref_hog_prob", "sub-family probability estimates")
    # a sequence-only search does not need the 3Di index of the database
    if load_structure:
        _load("ss_kmer_table", "3Di k-mer index")
        _load("ss_ref_fam_prob", "3Di family probability estimates")
        _load("ss_ref_hog_prob", "3Di sub-family probability estimates")

    print_message("\nFinished loading required data\n")
    print_line(80)


def _format_info_db(db):
    for k, v in db.get_metadata().items():
        if isinstance(v, list):
            if len(v) == 0:
                v = ["-"]
            v = ",".join(v)
        yield (k, v)


def info_db(args):
    from .database import Database
    from ._utils import print_line
    import sys

    with Database(args.db) as db:
        print_line(80, file=sys.stdout)
        for k, v in _format_info_db(db):
            print(f"  {k:23s}:{v!s:>40}")
        print_line(80, file=sys.stdout)


# welcome / goodbye messages for omamer search
def welcome():
    from . import __version__
    from ._utils import print_line, print_message

    welcome_message = """
   _____ _____ _____
  |     |     |  _  |_____ ___ ___
  |  |  | | | |     |     | -_|  _|
  |_____|_|_|_|__|__|_|_|_|___|_|   v{}

  OMAmer is licensed under the GNU Lesser General Public License 3.0 (LGPL-3.0).
    """.format(
        __version__
    )

    print_line(80)
    print_message(welcome_message)
    print_line(80)


def print_run_data(args):
    from . import __version__
    from ._utils import print_line, print_message
    import platform

    print_message("")
    print_line(80)
    print_message("\nRunning OMAmer on {}, using:".format(platform.node()))
    print_message(" - database: {}".format(args.db))
    if args.query:
        print_message(" - query: {}".format(args.query))
    if args.structure:
        print_message(" - 3Di query: {}".format(args.structure))
    print_message(" - version: {}".format(__version__))
    print_message("")
    print_line(80)

    # temporary removal for 2.0.0 release
    if args.reference_taxon is not None:
        raise RuntimeError("reference_taxon is not supported in release 2.0.0")


def goodbye(args, time_taken, search_rate):
    import sys

    from ._utils import print_line, print_message

    citation = "Victor Rossier, Alex Warwick Vesztrocy, Marc Robinson-Rechavi, Christophe Dessimoz, OMAmer: tree-driven and alignment-free protein assignment to subfamilies outperforms closest sequence approaches, Bioinformatics, Volume 37, Issue 18, September 2021, Pages 2866-2873, https://doi.org/10.1093/bioinformatics/btab219"

    print_message("")
    print_line(80)
    print_message("\nOMAmer search complete:")
    if args.out.name != sys.stdout.name:
        print_message(" - results written to: {}".format(args.out.name))
    print_message(f" - total {time_taken:.02f} seconds")
    print_message(f" - search phase only {search_rate:.02f} queries/s")
    print_message("\n\nNote: family p-values are stated in negative log units.")
    print_line(80)
    print_message(
        f"\nThank you for using OMAmer. If you use OMAmer in your research, please cite:\n\n{citation}\n\n"
    )
    print_message(
        "OMAmer uses data from the OMA browser. Results can be interpreted further using:"
    )
    print_message(" - OMA browser website (https://omabrowser.org)")
    print_message(
        " - PyOMADB, the Python OMA API client (https://github.com/DessimozLab/pyomadb)"
    )
    print_message("")
    print_line(80)


def check_args(args):
    if not args.query and not args.structure:
        raise RuntimeError("At least one of --query or --structure must be provided")
    for filename in (args.query, args.structure):
        if filename:
            # Enforce query existence check before loading DB
            with open(filename, "r") as _:
                pass

            if os.path.getsize(filename) == 0:
                raise RuntimeError(f"Input file {filename} is empty")
