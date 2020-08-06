'''
    OMAmer - tree-driven and alignment-free protein assignment to sub-families

    (C) 2019-2020 Victor Rossier <victor.rossier@unil.ch> and
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
'''
from ._utils import LOG


def mkdb_oma(args):
    from .database import DatabaseFromOMA
    from .index import Index
    import os

    assert (args.k < 8), "Max k-mer size is 7."
    LOG.info('Create database from OMA build')
    # todo: remove the oma database dependency / just take the root level.
    db = DatabaseFromOMA(
        args.db, root_taxon=args.root_taxon, include_younger_fams=True, min_prot_nr=args.min_hog_size, mode='w'
    )

    oma_db_fn = os.path.join(args.oma_path, "OmaServer.h5")
    nwk_fn = os.path.join(args.oma_path, "speciestree.nwk")

    # add sequences from database
    LOG.info('Adding sequences')
    db.build_database(oma_db_fn, nwk_fn)

    # build index
    LOG.info('Building index')
    ki = Index(db, k=args.k, reduced_alphabet=False)
    hidden_taxa = args.hidden_taxa.split(',')
    if hidden_taxa[0]:
        ki.hide_taxa(hidden_taxa, nwk_fn)
    ki.build_kmer_table()

    db.close()
    LOG.info('Done')

# new search using merge_search.py
def search(args):
    from Bio import SeqIO
    from tqdm import tqdm
    import sys

    from .database import Database
    from .index import Index
    from .merge_search import MergeSearch

    # reload
    db = Database(args.db, nthreads=args.nthreads)

    # setup search
    ms = MergeSearch(ki=db.ki, nthreads=args.nthreads, include_extant_genes=args.include_extant_genes)

    # only print header for file output 
    print_header = (args.out.name != sys.stdout.name)

    # initialise query
    ids = []
    seqs = []
    
    pbar = tqdm(desc='Searching')
    for rec in filter(lambda x: len(x.seq) >= db.ki.k,
                      SeqIO.parse(args.query, 'fasta')):
        ids.append(rec.id)
        seqs.append(str(rec.seq))
        if len(ids) == args.chunksize:
            ms.merge_search(seqs=seqs, ids=ids)
            pbar.update(len(ids))
            df = ms.output_results(threshold=args.threshold)
            if df.size >0:
                df.to_csv(args.out, sep='\t', index=False, header=print_header)
            ids = []
            seqs = []
            print_header = False

    # final search
    if len(ids) > 0:
        ms.merge_search(ids=ids, seqs=seqs)
        df = ms.output_results(threshold=args.threshold)
        pbar.update(len(ids))
        if df.size >0:
            df.to_csv(args.out, sep='\t', index=False, header=print_header)

    pbar.close()
    db.close()

# def search(args):
#     from Bio import SeqIO
#     from tqdm import tqdm
#     import sys

#     from .database import Database
#     from .index import Index
#     from .flat_search import FlatSearch
#     from .search import Search

#     # reload
#     db = Database(args.db, nthreads=args.nthreads)

#     # setup search
#     fs = FlatSearch(db.ki, nthreads=args.nthreads)
#     se = Search(fs, include_extant_genes=args.include_extant_genes)

#     # only print header for file output 
#     print_header = (args.out.name != sys.stdout.name)

#     # initialise query
#     ids = []
#     seqs = []
    
#     pbar = tqdm(desc='Searching')
#     for rec in filter(lambda x: len(x.seq) >= db.ki.k,
#                       SeqIO.parse(args.query, 'fasta')):
#         ids.append(rec.id)
#         seqs.append(str(rec.seq))
#         if len(ids) == args.chunksize:
#             fs.flat_search(ids=ids, seqs=seqs)
#             df = se.search(se.norm_fam_query_size, 1, se._max, se.norm_hog_query_size, args.threshold)
#             pbar.update(len(ids))
#             df.to_csv(args.out, sep='\t', index=False, header=print_header)


#             ids = []
#             seqs = []
#             print_header = False


#     # final search
#     fs.flat_search(ids=ids, seqs=seqs)
#     df = se.search(se.norm_fam_query_size, 1, se._max, se.norm_hog_query_size, args.threshold)
#     if len(ids) > 0:
#         pbar.update(len(ids))
#     df.to_csv(args.out, sep='\t', index=False, header=print_header)

#     pbar.close()
#     db.close()
