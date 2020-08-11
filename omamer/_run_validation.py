import sys
omamer_path = sys.argv[1]
sys.path.insert(0, omamer_path)

import os
import numpy as np
from tqdm import tqdm

from omamer.database import DatabaseFromOMA
from omamer.index import Index, QuerySequenceBuffer
from omamer.merge_search import MergeSearch
from omamer.validation import Validation

def run_validation_pipeline(
    working_path, root_taxon, min_hog_size, include_younger_fams, oma_db_fn, nwk_fn, k, reduced_alphabet, query_sp2hidden_taxa,
    query_sp, subfamily_query_sp2thresholds, focal_taxon, bin_num, val_mode, neg_root_taxon, chunksize):
    '''
    including automatic filenames
    '''
    out_path = '{}{}_{}families/'.format(working_path, root_taxon, 'only_' if not include_younger_fams else '')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    db_fn = '{}{}_{}wo_{}.h5'.format(out_path, root_taxon, 'younfams_' if include_younger_fams else '', '_'.join(['_'.join(x.split()) for x in query_sp2hidden_taxa[query_sp]]))
    va_fn = '{}_{}_{}.h5'.format(db_fn.split('.h5')[0], neg_root_taxon if neg_root_taxon else 'Random', '_'.join(query_sp.split()))

    va = _run_validation_pipeline(
        db_fn, root_taxon, min_hog_size, include_younger_fams, oma_db_fn, nwk_fn, k, reduced_alphabet, query_sp2hidden_taxa[query_sp], 
        va_fn, query_sp, subfamily_query_sp2thresholds[query_sp], root_taxon, bin_num, val_mode, neg_root_taxon, chunksize)
    
    return va

def _run_validation_pipeline(
    db_fn, root_taxon, min_hog_size, include_younger_fams, oma_db_fn, nwk_fn, k, reduced_alphabet, hidden_taxa,
    va_fn, query_sp, thresholds, focal_taxon, bin_num, val_mode, neg_root_taxon, chunksize):
    
    # build or reload database
    if not os.path.isfile(db_fn):
        db = DatabaseFromOMA(
            filename=db_fn, root_taxon=root_taxon, min_prot_nr=min_hog_size, 
            include_younger_fams=include_younger_fams, mode='w')
        
        # load sequences from OMA database
        db.build_database(oma_db_fn, nwk_fn)
        
        # build k-mer table
        ki = Index(db, k=k, reduced_alphabet=reduced_alphabet, hidden_taxa=hidden_taxa)
        ki.build_kmer_table()

        db.close()
    
    # reload database
    db = DatabaseFromOMA(
        filename=db_fn, root_taxon=root_taxon, min_prot_nr=min_hog_size, 
        include_younger_fams=include_younger_fams)
    
    qbuff = QuerySequenceBuffer(db, query_sp)

    # run or reload validation
    if not os.path.isfile(va_fn):
                      
        # setup search and validation
        ms = MergeSearch(ki=db.ki, nthreads=1)  # fails for nthreads > 1 BECAUSE numba version is <0.50
        va = Validation(
            db, filename=va_fn, thresholds=thresholds, nwk_fn=nwk_fn, query_sp=query_sp, 
            focal_taxon=focal_taxon, bin_num=bin_num, val_mode=val_mode, neg_root_taxon=neg_root_taxon, 
            max_query_nr=qbuff.prot_nr, neg_query_file='{}.fa'.format(va_fn.split('.')[0]), 
            oma_db_fn=oma_db_fn, nthreads=1)
        
        # search and validate
        ids = []
        seqs = []

        pbar = tqdm(desc='Searching')
        for i, q in enumerate(qbuff.ids):
            ids.append(q[0])
            seqs.append(qbuff[i])
            if len(ids) == chunksize:
                # search and validate the chunk
                ms.merge_search(seqs=seqs, ids=ids)
                va.validate(ms)     

                pbar.update(len(ids))
                ids = []
                seqs = []

        # search and validate last chunk
        ms.merge_search(seqs=seqs, ids=ids)
        va.validate(ms)
        if len(ids) > 0:
            pbar.update(len(ids))

        # close stuff
        pbar.close()
        va.va.close()
        
    va = Validation(
        db, filename=va_fn, thresholds=thresholds, nwk_fn=nwk_fn, query_sp=query_sp, 
        focal_taxon=focal_taxon, bin_num=bin_num, val_mode=val_mode, neg_root_taxon=neg_root_taxon, 
        max_query_nr=qbuff.prot_nr, neg_query_file='{}.fa'.format(va_fn.split('.')[0]), 
        oma_db_fn=oma_db_fn, nthreads=1)
    
    return va

working_path = sys.argv[2]
root_taxon = sys.argv[3]
min_hog_size = 6
include_younger_fams = True if (sys.argv[4] == 'True') else False
oma_path = sys.argv[5]
oma_db_fn = os.path.join(oma_path, "OmaServer.h5")
nwk_fn = os.path.join(oma_path, "speciestree.nwk")
k = 6
reduced_alphabet = False

query_sp = ' '.join(sys.argv[6].split('_'))

query_species = ['Ornithorhynchus anatinus', 'Lepisosteus oculatus', 'Branchiostoma floridae', 'Branchiostoma lanceolatum', 'Homo sapiens']
query_hidden_taxa = [['Ornithorhynchus anatinus'], ['Lepisosteus oculatus'], ['Branchiostoma'], ['Branchiostoma'], ['Homo sapiens']]
query_sp2hidden_taxa = dict(zip(query_species, query_hidden_taxa))

omamer_thresholds = np.arange(0.001, 1.001, 0.005)
subfamily_query_sp2thresholds = {
    ('Ornithorhynchus anatinus'): omamer_thresholds,
    ('Lepisosteus oculatus'): omamer_thresholds,
    ('Branchiostoma floridae'): omamer_thresholds,
    ('Branchiostoma lanceolatum'): omamer_thresholds,
    ('Homo sapiens'): omamer_thresholds
}

root_taxon2focal_taxon = {
    "Metazoa": "Metazoa",
    "LUCA": "Metazoa",
    "Hominidae": "Hominidae"
}


focal_taxon = "Metazoa"
bin_num = 1
val_mode = 'golike' 
neg_root_taxon_arg = sys.argv[7]
neg_root_taxon = None if (neg_root_taxon_arg == 'Random') else neg_root_taxon_arg

chunksize = int(sys.argv[8])

run_validation_pipeline(
    working_path, root_taxon, min_hog_size, include_younger_fams, oma_db_fn, nwk_fn, k, reduced_alphabet, query_sp2hidden_taxa,
    query_sp, subfamily_query_sp2thresholds, focal_taxon, bin_num, val_mode, neg_root_taxon, chunksize)
