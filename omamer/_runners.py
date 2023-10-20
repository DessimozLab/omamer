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
from ._utils import LOG


def mkdb_oma(args):
    from .database import DatabaseFromOMA
    from .index import Index
    import os

    assert args.k < 8, "Max k-mer size is 7."
    LOG.info("Create database from OMA build")
    LOG.info("arguments for build:")
    for (k, v) in vars(args).items():
        LOG.info(" - {}: {}".format(k, v))

    db = DatabaseFromOMA(
        args.db,
        root_taxon=args.root_taxon,
        min_fam_size=args.min_fam_size,
        logic=args.logic,
        min_fam_completeness=args.min_fam_completeness,
        include_younger_fams=True,
        mode="w",
    )

    oma_db_fn = os.path.join(args.oma_path, "OmaServer.h5")
    nwk_fn = os.path.join(args.oma_path, "speciestree.nwk")

    # add sequences from database
    LOG.info("Adding sequences")
    seq_buff = db.build_database(oma_db_fn, nwk_fn)

    hidden_taxa = []
    if args.hidden_taxa:
        hidden_taxa = [" ".join(x.split("_")) for x in args.hidden_taxa.split(",")]

    LOG.info("Building index")
    db.ki = Index(
        db, k=args.k, reduced_alphabet=args.reduced_alphabet, hidden_taxa=hidden_taxa
    )
    db.ki.build_kmer_table(seq_buff)
    db.add_metadata()
    db.add_md5_hash()

    db.close()
    LOG.info("Done")


def search(args):
    from alive_progress import alive_bar
    from ._utils import print_message, print_line
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

    t0 = time()

    # reload
    db = Database(args.db)

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

    _ensure_data_loaded(ms)

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
        for ids, seqs in SequenceReader.read(
            args.query,
            k=db.ki.k,
            format="fasta",
            chunksize=args.chunksize,
            sanitiser=db.ki.alphabet.sanitise_seq,
        ):
            t_search0 = time()
            df = ms.merge_search(
                seqs=seqs,
                ids=ids,
                top_n_fams=args.top_n_fams,
                alpha=args.family_alpha,
                sst=args.threshold,
                family_only=args.family_only,
                ref_taxon_off=ref_taxoff,
            )
            t_search1 = time()

            pbar(len(ids))

            if df.size > 0:
                if print_header:
                    # write the top header
                    print("!omamer-version: {}".format(__version__), file=args.out)
                    print(
                        "!query-md5: {}".format(compute_file_md5(args.query.name)),
                        file=args.out,
                    )
                    print(
                        "!date-run: {}".format(datetime.fromtimestamp(t0).isoformat()),
                        file=args.out,
                    )
                    print("!db-path: {}".format(db.filename), file=args.out)

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

    db.close()

    search_rate = sum(map(lambda x: x[0], search_times)) / sum(
        map(lambda x: x[1], search_times)
    )
    goodbye(args, time() - t0, search_rate)


def _ensure_data_loaded(ms):
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
    print_message(" - query: {}".format(args.query.name))
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
        "OMAmer uses data from the OMA browser. Results can be interpretted further using:"
    )
    print_message(" - OMA browser website (https://omabrowser.org)")
    print_message(
        " - PyOMADB, the Python OMA API client (https://github.com/DessimozLab/pyomadb)"
    )
    print_message("")
    print_line(80)
