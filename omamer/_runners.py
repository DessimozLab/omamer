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
        args.db, root_taxon=args.root_taxon, min_fam_size=args.min_fam_size, logic=args.logic, min_fam_completeness=args.min_fam_completeness, include_younger_fams=True, mode='w'
    )

    oma_db_fn = os.path.join(args.oma_path, "OmaServer.h5")
    nwk_fn = os.path.join(args.oma_path, "speciestree.nwk")

    # add sequences from database
    LOG.info('Adding sequences')
    db.build_database(oma_db_fn, nwk_fn)

    ## tmp argument of species to include (inverse of hidden taxa)
    #import numpy as np
    #hidden_taxa = []
    #if args.species:
    #    with open(args.species, 'r') as inf:
    #        sp_offsets = np.array([int(l.rstrip()) for l in inf.readlines()])
    #        sp_offsets.sort()
    #    f = np.full(db._sp_tab.nrows, False)
    #    f[sp_offsets] = True
    #    hidden_taxa = list(map(lambda x: x.decode('ascii'), db._sp_tab[:]['ID'][~f]))
    #elif args.hidden_taxa:
    #    hidden_taxa=[' '.join(x.split('_')) for x in args.hidden_taxa.split(',')]

    hidden_taxa = []
    if args.hidden_taxa:
        hidden_taxa = [' '.join(x.split('_')) for x in args.hidden_taxa.split(',')]

    LOG.info('Building index')
    ki = Index(db, k=args.k, reduced_alphabet=args.reduced_alphabet, nthreads=1, hidden_taxa=hidden_taxa)
    ki.build_kmer_table()

    db.close()
    LOG.info('Done')

def search(args):
    from Bio import SeqIO
    from tqdm import tqdm
    import numpy as np
    import os
    import sys

    from .database import Database
    from .index import Index
    from .merge_search import MergeSearch

    low_mem = False  # need to reintroduce the low memory
    if not low_mem:
        # Set number of threads for numba.
        # TODO: check whether we also need to check multiprocessing param for numpy.
        import numba
        nthreads = args.nthreads if args.nthreads > 0 else os.cpu_count()
        numba.set_num_threads(nthreads)

    # reload
    db = Database(args.db)

    # setup search
    ms = MergeSearch(ki=db.ki, nthreads=args.nthreads, low_mem=False, include_extant_genes=args.include_extant_genes)

    if args.score == 'default':
        score = 'querysize_hogsize_kmerfreq'
    elif args.score == 'sensitive':
        score = 'nonparam_naive'

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
            ms.merge_search(
                seqs=seqs, ids=ids, fasta_file=None, score=score, cum_mode='max', top_m_fams=100, top_n_fams=1, perm_nr=1, w_size=6, dist='poisson', fam_filter=np.array([], dtype=np.int64)
            )
            pbar.update(len(ids))
            df = ms.output_results(overlap=0, fst=0, sst=args.threshold, ref_taxon=args.reference_taxon)
            if df.size >0:
                df.to_csv(args.out, sep='\t', index=False, header=print_header)
            ids = []
            seqs = []
            print_header = False

    # final search
    if len(ids) > 0:
        ms.merge_search(
            seqs=seqs, ids=ids, fasta_file=None, score=score, cum_mode='max', top_m_fams=100, top_n_fams=1, perm_nr=1, w_size=6, dist='poisson', fam_filter=np.array([], dtype=np.int64)
        )
        df = ms.output_results(overlap=0, fst=0, sst=args.threshold, ref_taxon=args.reference_taxon)
        pbar.update(len(ids))
        if df.size >0:
            df.to_csv(args.out, sep='\t', index=False, header=print_header)

    pbar.close()
    db.close()
