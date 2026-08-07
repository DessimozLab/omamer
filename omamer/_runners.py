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
import numpy as np
import os
import psutil
from enum import Enum
from ._utils import LOG, check_file_exists


class Compression(Enum):
    NONE = 0
    ELIAS_FANO = 1
    HUFFMAN = 2

compression = Compression.NONE


def mkdb_oma(args):
    from .database import DatabaseFromOMABrowser, DatabaseFromOrthoXML
    from .index import Index, validate_kmer_percentage
    from .stat_models import validate_index_models

    from ete3 import Tree
    import os

    _ensure_db_build_dependencies_available()

    assert args.k < 8, "Max k-mer size is 7."
    args.kmer_percentage = validate_kmer_percentage(
        getattr(args, "kmer_percentage", 100.0)
    )
    args.models = validate_index_models(
        getattr(args, "models", ("binomial", "beta-binomial"))
    )
    LOG.info("Create database from OMA build")
    LOG.info("arguments for build:")
    for (k, v) in vars(args).items():
        LOG.info(" - {}: {}".format(k, v))

    # work out mode
    browser_db_mode = args.oma_path is not None

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
            check_file_exists(structure_h5_fn)

    else:
        if args.orthoxml is None or args.species_tree is None or len(args.sequences) == 0:
            raise ValueError('Must pass either browser database (--oma_path) or all files for OrthoXML build (--orthoxml --species_tree --sequences)')

        # orthoxml mode
        Database = DatabaseFromOrthoXML

        # find files
        oxml_fn = args.orthoxml.name
        sequence_files = list(map(lambda x: x.name, args.sequences))
        structure_files = list(map(lambda x: x.name, args.structures)) \
            if args.structures else []
        nwk = args.species_tree.name

    # check if root_taxon in tree
    t = Tree(nwk, format=1, quoted_node_names=True)
    if args.root_taxon is not None:
        pruned_t = t.search_nodes(name=args.root_taxon)
        if len(pruned_t) == 0:
            raise ValueError("Unable to find {} as root taxon".format(args.root_taxon))
        elif len(pruned_t) > 1:
            raise ValueError("Ambiguous root taxon {} ({})".format(args.root_taxon, ", ".join(map(lambda x: x.name, pruned_t))))
        t = pruned_t[0]
    root_taxon = t.name

    if args.hidden_taxa:
        # load and check for hidden taxa
        hidden_taxa = list(map(lambda x: x.rstrip(), args.hidden_taxa))

        for ht in hidden_taxa:
            r = t.search_nodes(name=ht)
            if len(r) == 0:
                raise ValueError("Unable to find {} as hidden taxon".format(ht))
            elif len(r) > 1:
                raise ValueError("Ambiguous hidden taxon {} ({})".format(ht, ", ".join(map(lambda x: x.name, r))))
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

    # add sequences from database
    LOG.info("Loading sequences")
    if browser_db_mode:
        seq_buff, ss_buff = db.build_database(oma_db_fn, nwk, structure_h5_fn)
    else:
        seq_buff, ss_buff = db.build_database(oxml_fn, sequence_files, structure_files, nwk)

    LOG.info("Building index")
    db.ki = Index(
        db,
        k=args.k,
        reduced_alphabet=args.reduced_alphabet,
        hidden_taxa=hidden_taxa,
        kmer_percentage=args.kmer_percentage,
        models=args.models,
        bbinom_options={
            "n_values_by_modality": {
                "ss": getattr(args, "bbinom_ss_n_values", None),
            },
            "n_buckets": getattr(args, "bbinom_n_buckets", 24),
            "min_records_per_n": getattr(
                args, "bbinom_min_records_per_n", 50
            ),
            "max_records_per_n": getattr(
                args, "bbinom_max_records_per_n", 500
            ),
            "min_nonzero_queries": getattr(
                args, "bbinom_min_nonzero_queries", 20
            ),
            "max_families": getattr(args, "bbinom_max_families", 0),
            "min_family_prob": getattr(args, "bbinom_min_family_prob", 0.0),
            "workers": getattr(args, "bbinom_fit_workers", 1),
            "max_histogram_gb": getattr(
                args, "bbinom_max_histogram_gb", 3.0
            ),
            "seed": getattr(args, "bbinom_seed", 42),
        },
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
    _check_db_kmer_percentage(db, getattr(args, "kmer_percentage", None))

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
        ms,
        load_structure=(
            bool(args.structure)
            and getattr(args, "search_mode", "auto") != "seq"
        ),
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
        has_sequence = args.query and os.path.exists(args.query)
        has_structure = args.structure and os.path.exists(args.structure)

        query_iter = None
        if has_sequence:
            query_iter = SequenceReader.read(
                args.query,
                k=db.ki.k,
                format="fasta",
                chunksize=args.chunksize,
                sanitiser=db.ki.alphabet.sanitise_seq,
            )

        struct_iter = None
        if has_structure:
            struct_iter = SequenceReader.read(
                args.structure,
                k=db.ki.k,
                format="fasta",
                chunksize=args.chunksize,
                sanitiser=db.ki.alphabet.sanitise_seq,
            )

        if not has_sequence and not has_structure:
            raise RuntimeError("At least one of --query or --structure must be provided")

        # main iterator over sequence records
        if has_sequence:
            main_iter = query_iter
        else:
            main_iter = struct_iter

        for i, (ids_q, seqs_q) in enumerate(main_iter):
            struct_seqs = []

            # if both exist, advance structure iterator and
            # check sequence record IDs match in both
            # the sequence and structure file
            if has_sequence and has_structure:
                ids_s, struct_seqs = next(struct_iter)
                if ids_q != ids_s:
                    raise RuntimeError("Query and structure IDs must match")

            # if only structure, swap inputs as the main iterator is
            # not over sequence
            elif has_structure and not has_sequence:
                struct_seqs = seqs_q
                seqs_q = []

            t_search0 = time()
            df = ms.merge_search(
                seqs=seqs_q,
                struct_seqs=struct_seqs,
                ids=ids_q,
                top_n_fams=args.top_n_fams,
                alpha=args.family_alpha,
                family_correction=args.family_correction,
                sst=args.threshold,
                family_only=args.family_only,
                ref_taxon_off=ref_taxoff,
                search_mode=getattr(args, "search_mode", "auto"),
                family_model=getattr(args, "family_model", "auto"),
                family_sorting=getattr(args, "family_sorting", "normcount"),
            )
            t_search1 = time()

            pbar(len(ids_q))

            if df.size > 0:
                if print_header:
                    # write the top header
                    print("!omamer-version: {}".format(__version__), file=args.out)
                    print(
                        "!query-md5: {}".format(compute_file_md5(args.query if args.query else args.structure)),
                        file=args.out,
                    )
                    print(
                        "!date-run: {}".format(datetime.fromtimestamp(t0).isoformat()),
                        file=args.out,
                    )
                    print("!db-path: {}".format(db.filename), file=args.out)
                    print(
                        "!family-model-requested: {}".format(
                            getattr(args, "family_model", "auto")
                        ),
                        file=args.out,
                    )
                    resolved_models = ",".join(
                        "{}:{}".format(modality, model)
                        for modality, model in sorted(
                            ms.resolved_family_models.items()
                        )
                    )
                    print(
                        "!family-model-resolved: {}".format(resolved_models),
                        file=args.out,
                    )
                    print(
                        "!family-sorting: {}".format(
                            getattr(args, "family_sorting", "normcount")
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

            search_times.append((len(ids_q), t_search1 - t_search0))

    db.close()

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
    _load("level_arr", "family hierarchy")

    _load("kmer_table", "k-mer index")
    _load("ref_fam_prob", "family probability estimates")
    _load("ref_hog_prob", "sub-family probability estimates")
    # Databases pre All.Jul2024 didn't have any structure
    # We keep it backward compatible to make it possible to load
    # older databases.
    # A sequence-only query does not need to materialize a structural index
    # merely because the database happens to contain one.
    if load_structure and ms.db.has_structure():
        _load("ss_kmer_table", "structural k-mer index")
        _load("ss_ref_fam_prob", "structural family probability estimates")
        _load("ss_ref_hog_prob", "structural sub-family probability estimates")

    process = psutil.Process()
    LOG.info(f"Memory after loading DB: "
                 f"{process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")

    if compression == Compression.ELIAS_FANO:
        # TODO:
        # Conditional import as cppyy can give problems on
        # certain architectures. Until we polish this code and
        # it works everywhere, keep it here
        from .compression import update_with_elias_fano

        # Replace the original kmer_index with a more
        # compact Elias-Fano representation.
        buff, new_idx, raw_flags = update_with_elias_fano(
            ms.kmer_table["idx"], ms.kmer_table["buff"]
        )

        ms.kmer_table["buff"] = buff
        ms.kmer_table["idx"] = new_idx
        ms.kmer_table["raw_flags"] = raw_flags

        LOG.info(f"Memory after replacing kmer_table: "
                 f"{process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")

    else:
        ms.kmer_table['raw_flags'] = np.empty((0,))


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


def _check_db_kmer_percentage(db, requested_percentage):
    """Ensure a compatibility assertion matches the build-time DB setting."""
    if requested_percentage is None:
        return
    from .index import validate_kmer_percentage

    requested_percentage = validate_kmer_percentage(requested_percentage)
    if not np.isclose(requested_percentage, db.ki.kmer_percentage):
        raise ValueError(
            "Requested kmer_percentage={} does not match the database's "
            "build-time kmer_percentage={}".format(
                requested_percentage,
                db.ki.kmer_percentage,
            )
        )


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
    print_message(" - query: {}".format(args.query))
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
    print_message(f" - search phase {search_rate:.02f} queries/s")
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
    for filename in [args.query, args.structure]:
        if filename:
            # Enforce query existence check before loading DB
            with open(filename, "r") as _:
                pass

            if os.path.getsize(filename) == 0:
                raise RuntimeError(f"Input file {filename} is empty")
