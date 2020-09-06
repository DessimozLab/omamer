import sys
omamer_path = sys.argv[1]
sys.path.insert(0, omamer_path)

from omamer.database import DatabaseFromOMA
from omamer.index import Index, QuerySequenceBuffer
from omamer.alphabets import Alphabet
from omamer.merge_search import MergeSearch
from omamer.validation import Validation

import os
import tables
import shutil
import numpy as np
from tqdm import tqdm

def is_complete(fn, db_path):
    cf = '{}COMPLETE.txt'.format(db_path)
    if not os.path.exists(cf):
        with open(cf, 'w') as inf:
            pass
    with open(cf, 'r') as inf:
        complete_fns = set(map(lambda x: x.rstrip(), inf.readlines()))
    return fn in complete_fns

def set_complete(fn, db_path):
    with open('{}COMPLETE.txt'.format(db_path), 'a') as inf:
        inf.write('{}\n'.format(fn))

def build_database_from_oma(db_path, root_taxon, min_fam_size, min_fam_completeness, include_younger_fams, oma_db_fn, nwk_fn):
    '''
    parse OMA HOGs
    '''
    db_fn = '{}{}_MinFamSize{}_MinFamComp0{}_{}.h5'.format(
        db_path, root_taxon, min_fam_size, str(min_fam_completeness).split('.')[-1], 'yf' if include_younger_fams else 'rf')

    if not is_complete(db_fn, db_path):
        if os.path.exists(db_fn):
            os.remove(db_fn)
            
        db = DatabaseFromOMA(
            filename=db_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, min_fam_completeness=min_fam_completeness,
            include_younger_fams=include_younger_fams, mode='w')

        # load sequences from OMA database
        db.build_database(oma_db_fn, nwk_fn)
        db.close()

        set_complete(db_fn, db_path)

def build_suffix_array(db_path, root_taxon, min_fam_size, min_fam_completeness, include_younger_fams, reduced_alphabet):
    '''
    compute SA 
    '''
    # reload database
    db_fn = '{}{}_MinFamSize{}_MinFamComp0{}_{}.h5'.format(
        db_path, root_taxon, min_fam_size, str(min_fam_completeness).split('.')[-1], 'yf' if include_younger_fams else 'rf')

    assert os.path.exists(db_fn), 'database missing'

    db = DatabaseFromOMA(
        filename=db_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, min_fam_completeness=min_fam_completeness,
        include_younger_fams=include_younger_fams, mode='r')

    # compute suffix array
    alphabet_n = 21 if not reduced_alphabet else 13    
    sa_fn = '{}SA_{}_MinFamSize{}_MinFamComp0{}_{}_A{}.h5'.format(
        db_path, root_taxon, min_fam_size, str(min_fam_completeness).split('.')[-1], 
        'yf' if include_younger_fams else 'rf', alphabet_n)

    if not is_complete(sa_fn, db_path):
        if os.path.exists(sa_fn):
            os.remove(sa_fn)

        alphabet = Alphabet(n=alphabet_n)
        sa = Index._build_suffixarray(alphabet.translate(db._seq_buff[:]), len(db._prot_tab))

        # store it in HDF5
        sa_h5 = tables.open_file(sa_fn, 'w', filters=db._compr)
        sa_h5.create_carray('/', 'SuffixArray', obj=sa, filters=db._compr)
        sa_h5.close()

        set_complete(sa_fn, db_path)

def build_kmer_table(
    db_path, root_taxon, min_fam_size, min_fam_completeness, include_younger_fams, reduced_alphabet, hidden_taxa, k):
    
    alphabet_n = 21 if not reduced_alphabet else 13

    db_fn = '{}{}_MinFamSize{}_MinFamComp0{}_{}.h5'.format(
        db_path, root_taxon, min_fam_size, str(min_fam_completeness).split('.')[-1], 'yf' if include_younger_fams else 'rf')
    
    sa_fn = '{}SA_{}_MinFamSize{}_MinFamComp0{}_{}_A{}.h5'.format(
        db_path, root_taxon, min_fam_size, str(min_fam_completeness).split('.')[-1], 
        'yf' if include_younger_fams else 'rf', alphabet_n) 

    assert os.path.exists(db_fn), 'database missing'
    assert os.path.exists(sa_fn), 'suffix array missing'

    ki_fn = '{}{}_MinFamSize{}_MinFamComp0{}_{}_A{}_k{}_wo_{}.h5'.format(
        db_path, root_taxon, min_fam_size, str(min_fam_completeness).split('.')[-1], 
        'yf' if include_younger_fams else 'rf', alphabet_n, k, '_'.join(['_'.join(x.split()) for x in hidden_taxa]))

    if not is_complete(ki_fn, db_path):
        if os.path.exists(ki_fn):
            os.remove(ki_fn)

        # copy database
        shutil.copyfile(db_fn, ki_fn)

        # load in append mode
        db = DatabaseFromOMA(
            filename=ki_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, min_fam_completeness=min_fam_completeness,
            include_younger_fams=include_younger_fams, mode='a')   

        # load suffix array
        sa_h5 = tables.open_file(sa_fn, 'r', filters=db._compr)
        sa = sa_h5.root.SuffixArray[:]
        
        # build index
        ki = Index(db, k=k, reduced_alphabet=reduced_alphabet, hidden_taxa=hidden_taxa)
        ki._build_kmer_table(sa)
        
        db.close()
        sa_h5.close()

        set_complete(ki_fn, db_path)

def search_validate(
    db_path, root_taxon, min_fam_size, min_fam_completeness, include_younger_fams, reduced_alphabet, hidden_taxa, k,
    thresholds, oma_db_fn, nwk_fn, score, cum_mode, top_m_fams, val_mode, neg_root_taxon, focal_taxon, fam_bin_num, hog_bin_num, 
    pvalue_score, query_sp, overwrite=False):
    
    alphabet_n = 21 if not reduced_alphabet else 13
    
    # reload k-mer table
    ki_fn = '{}{}_MinFamSize{}_MinFamComp0{}_{}_A{}_k{}_wo_{}.h5'.format(
        db_path, root_taxon, min_fam_size, str(min_fam_completeness).split('.')[-1], 
        'yf' if include_younger_fams else 'rf', alphabet_n, k, '_'.join(['_'.join(x.split()) for x in hidden_taxa]))
    
    assert os.path.exists(ki_fn), 'index missing'

    # load in append mode
    db = DatabaseFromOMA(
        filename=ki_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, min_fam_completeness=min_fam_completeness,
        include_younger_fams=include_younger_fams, mode='r')

    # load query sequences
    sbuff = QuerySequenceBuffer(db, query_sp)
    chunksize = sbuff.prot_nr

    # setup search and validation steps
    se_va_fn = '{}{}_MinFamSize{}_MinFamComp0{}_{}_A{}_k{}_wo_{}_{}_{}_top{}fams_{}_{}_{}_{}fbn_{}hbn.h5'.format(
        db_path, root_taxon, min_fam_size, str(min_fam_completeness).split('.')[-1], 
        'yf' if include_younger_fams else 'rf', alphabet_n, k, '_'.join(['_'.join(x.split()) for x in hidden_taxa]),
        score, cum_mode, top_m_fams, val_mode, neg_root_taxon, focal_taxon, fam_bin_num, hog_bin_num)

    if not is_complete(se_va_fn, db_path) or overwrite:
        if os.path.exists(se_va_fn):
            os.remove(se_va_fn)

        ms = MergeSearch(ki=db.ki, nthreads=1)
        va = Validation(db, se_va_fn, thresholds, oma_db_fn=oma_db_fn, nwk_fn=nwk_fn, 
                        neg_query_file='{}.fa'.format(se_va_fn.split('.')[0]), nthreads=1, query_sp=query_sp, 
                        max_query_nr=sbuff.prot_nr, val_mode=val_mode, neg_root_taxon=neg_root_taxon, focal_taxon=focal_taxon, 
                        fam_bin_num=fam_bin_num, hog_bin_num=fam_bin_num)

        assert va.mode == 'w'
        
        # search and validate
        ids = []
        seqs = []

        pbar = tqdm(desc='Searching')
        for i, q in enumerate(sbuff.ids):
            ids.append(q)
            seqs.append(sbuff[i])
            if len(ids) == chunksize:
                # search and validate the chunk
                ms.merge_search(seqs=seqs, ids=ids, score=score, cum_mode=cum_mode, top_m_fams=top_m_fams)
                va.validate(ms, pvalue_score=pvalue_score)     

                pbar.update(len(ids))
                ids = []
                seqs = []

        # search and validate last chunk
        if len(ids) > 0:
            ms.merge_search(seqs=seqs, ids=ids, score=score, cum_mode=cum_mode, top_m_fams=top_m_fams)
            va.validate(ms, pvalue_score=pvalue_score)
            pbar.update(len(ids))

        # close stuff
        pbar.close()
        va.va.close()

        set_complete(se_va_fn, db_path)

if __name__ == "__main__":
    
    step = sys.argv[2] 
    if step == 'db':
        db_path = sys.argv[3]
        root_taxon = sys.argv[4]
        min_fam_size = int(sys.argv[5])
        min_fam_completeness = float(sys.argv[6])
        include_younger_fams =  True if (sys.argv[7] == 'True') else False
        oma_path = sys.argv[8]
        oma_db_fn = os.path.join(oma_path, "OmaServer.h5")
        nwk_fn = os.path.join(oma_path, "speciestree.nwk")
        build_database_from_oma(
            db_path, root_taxon, min_fam_size, min_fam_completeness, include_younger_fams, oma_db_fn, nwk_fn)
    
    elif step == 'sa':
        db_path = sys.argv[3]
        root_taxon = sys.argv[4]
        min_fam_size = int(sys.argv[5])
        min_fam_completeness = float(sys.argv[6])
        include_younger_fams =  True if (sys.argv[7] == 'True') else False
        reduced_alphabet = True if (sys.argv[8] == 'True') else False
        build_suffix_array(
            db_path, root_taxon, min_fam_size, min_fam_completeness, include_younger_fams, reduced_alphabet)

    elif step == 'ki':
        db_path = sys.argv[3]
        root_taxon = sys.argv[4]
        min_fam_size = int(sys.argv[5])
        min_fam_completeness = float(sys.argv[6])
        include_younger_fams =  True if (sys.argv[7] == 'True') else False
        reduced_alphabet = True if (sys.argv[8] == 'True') else False
        hidden_taxa = [' '.join(x.split('_')) for x in sys.argv[9].split(',')]
        k = int(sys.argv[10])
        build_kmer_table(
            db_path, root_taxon, min_fam_size, min_fam_completeness, include_younger_fams, reduced_alphabet, hidden_taxa, k)    

    elif step == 'se_va':
        db_path = sys.argv[3]
        root_taxon = sys.argv[4]
        min_fam_size = int(sys.argv[5])
        min_fam_completeness = float(sys.argv[6])
        include_younger_fams =  True if (sys.argv[7] == 'True') else False
        reduced_alphabet = True if (sys.argv[8] == 'True') else False
        hidden_taxa = [' '.join(x.split('_')) for x in sys.argv[9].split(',')]
        k = int(sys.argv[10])
        thresholds = np.arange(*(float(x) for x in sys.argv[11].split(',')))  # e.g. "0,1.01,0.01"
        oma_path = sys.argv[12]
        oma_db_fn = os.path.join(oma_path, "OmaServer.h5")
        nwk_fn = os.path.join(oma_path, "speciestree.nwk")
        score = sys.argv[13]
        cum_mode = sys.argv[14]
        top_m_fams = int(sys.argv[15])
        val_mode = sys.argv[16]
        neg_root_taxon = sys.argv[17]
        focal_taxon = sys.argv[18]
        fam_bin_num = int(sys.argv[19])
        hog_bin_num = int(sys.argv[20])
        query_sp = ' '.join(sys.argv[21].split('_'))
        overwrite = True if (sys.argv[22] == 'True') else False

        if score in {'mash_pvalue', 'kmerfreq_pvalue', 'kmerfreqmin_pvalue'}:
            pvalue_score = True
        else:
            pvalue_score = False

        search_validate(
            db_path, root_taxon, min_fam_size, min_fam_completeness, include_younger_fams, reduced_alphabet, hidden_taxa, k,
            thresholds, oma_db_fn, nwk_fn, score, cum_mode, top_m_fams, val_mode, neg_root_taxon, focal_taxon, fam_bin_num, hog_bin_num, 
            pvalue_score, query_sp, overwrite)
    else:
        print('unknown step')





