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

    db.close()
    LOG.info("Done")


def search(args):
    from tqdm.auto import tqdm
    import os
    import sys

    from .database import Database
    from .merge_search import MergeSearch
    from .sequence_reader import SequenceReader

    # reload
    db = Database(args.db)

    # setup search
    ms = MergeSearch(ki=db.ki, include_extant_genes=args.include_extant_genes)

    # only print header for file output
    args.out = sys.stdout if args.out is None else args.out
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

    pbar = tqdm(desc="Searching")
    for ids, seqs in SequenceReader.read(
        args.query,
        k=db.ki.k,
        format="fasta",
        chunksize=args.chunksize,
        sanitiser=db.ki.alphabet.sanitise_seq,
    ):
        df = ms.merge_search(
            seqs=seqs,
            ids=ids,
            top_n_fams=args.top_n_fams,
            alpha=args.family_alpha,
            sst=args.threshold,
            family_only=args.family_only,
            ref_taxon_off=ref_taxoff,
        )
        pbar.update(len(ids))
        if df.size > 0:
            df.to_csv(args.out, sep="\t", index=False, header=print_header)
            print_header = False

    pbar.close()
    db.close()


def info_db(args):
    from .database import Database

    with Database(args.db) as db:
        print("=" * 80)
        for k, v in db.get_metadata().items():
            if isinstance(v, list):
                if len(v) == 0:
                    v = ["-"]
                v = ",".join(v)
            print(f"  {k:23s}:{v!s:>40}")
        print("=" * 80)
