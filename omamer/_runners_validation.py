import sys
omamer_path = sys.argv[1]
sys.path.insert(0, omamer_path)

from omamer.database import DatabaseFromOMA
from omamer.index import Index
from omamer.alphabets import Alphabet

import tables
import os

def build_database_from_oma(db_path, root_taxon, min_fam_size, min_completeness, include_younger_fams, oma_db_fn, nwk_fn):
    '''
    parse OMA HOGs
    '''
    db_fn = '{}{}_MinFamSize{}_MinFamComp0{}_{}.h5'.format(
        db_path, root_taxon, min_fam_size, str(min_completeness).split('.')[-1], '_yf' if include_younger_fams else '_rf')
    db = DatabaseFromOMA(
        filename=db_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, min_completeness=min_completeness,
        include_younger_fams=include_younger_fams, mode='w')

    # load sequences from OMA database
    db.build_database(oma_db_fn, nwk_fn)
    db.close()

def build_suffix_array(db_path, root_taxon, min_fam_size, min_completeness, include_younger_fams, reduced_alphabet):
    '''
    compute SA 
    '''
    # reload database
    db_fn = '{}{}_MinFamSize{}_MinFamComp0{}_{}.h5'.format(
        db_path, root_taxon, min_fam_size, str(min_completeness).split('.')[-1], '_yf' if include_younger_fams else '_rf')
    db = DatabaseFromOMA(
        filename=db_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, min_completeness=min_completeness,
        include_younger_fams=include_younger_fams, mode='r')

    # compute suffix array
    alphabet_n = 21 if not reduced_alphabet else 13
    alphabet = Alphabet(n=alphabet_n)
    sa = Index._build_suffixarray(alphabet.translate(db._seq_buff[:]), len(db._prot_tab))

    # store it in HDF5
    sa_fn = '{}SA_{}_MinFamSize{}_MinFamComp0{}_{}_A{}.h5'.format(
        db_path, root_taxon, min_fam_size, str(min_completeness).split('.')[-1], 
        '_yf' if include_younger_fams else '_rf', alphabet_n)
    sa_h5 = tables.open_file(sa_fn, 'w', filters=db._compr)
    sa_h5.create_carray('/', 'SuffixArray', obj=sa, filters=db._compr)
    sa_h5.close()

def build_kmer_table(
    db_path, root_taxon, min_fam_size, min_completeness, include_younger_fams, reduced_alphabet, hidden_taxa, k):
    
    alphabet_n = 21 if not reduced_alphabet else 13

    # copy database
    db_fn = '{}{}_MinFamSize{}_MinFamComp0{}_{}.h5'.format(
        db_path, root_taxon, min_fam_size, str(min_completeness).split('.')[-1], '_yf' if include_younger_fams else '_rf')    
    ki_fn = '{}{}_MinFamSize{}_MinFamComp0{}_{}_A{}_k{}_wo_{}.h5'.format(
        db_path, root_taxon, min_fam_size, str(min_completeness).split('.')[-1], 
        '_yf' if include_younger_fams else '_rf', alphabet_n, k, '_'.join(['_'.join(x.split()) for x in hidden_taxa]))
    
    shutil.copyfile(db_fn, ki_fn)

    # load in append mode
    db = DatabaseFromOMA(
        filename=ki_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, min_completeness=min_completeness,
        include_younger_fams=include_younger_fams, mode='a')   

    # load suffix array
    sa_fn = '{}SA_{}_MinFamSize{}_MinFamComp0{}_{}_A{}.h5'.format(
        db_path, root_taxon, min_fam_size, str(min_completeness).split('.')[-1], 
        '_yf' if include_younger_fams else '_rf', alphabet_n) 
    sa_h5 = tables.open_file(sa_fn, 'r', filters=db._compr)
    sa = sa_h5.root.SuffixArray[:]
    
    # build index
    ki = Index(db, k=k, reduced_alphabet=reduced_alphabet, hidden_taxa=hidden_taxa)
    ki._build_kmer_table(sa)
    
    db.close()
    sa_h5.close()

if __name__ == "__main__":
    
    step = sys.argv[2] 
    if step == 'db':
        db_path = sys.argv[3]
        root_taxon = sys.argv[4]
        min_fam_size = int(sys.argv[5])
        min_completeness = float(sys.argv[6])
        include_younger_fams =  True if (sys.argv[7] == 'True') else False
        oma_path = sys.argv[8]
        oma_db_fn = os.path.join(oma_path, "OmaServer.h5")
        nwk_fn = os.path.join(oma_path, "speciestree.nwk")
        build_database_from_oma(
            db_path, root_taxon, min_fam_size, min_completeness, include_younger_fams, oma_db_fn, nwk_fn)
    
    elif step == 'sa':
        db_path = sys.argv[3]
        root_taxon = sys.argv[4]
        min_fam_size = int(sys.argv[5])
        min_completeness = float(sys.argv[6])
        include_younger_fams =  True if (sys.argv[7] == 'True') else False
        reduced_alphabet = True if (sys.argv[8] == 'True') else False
        build_suffix_array(
            db_path, root_taxon, min_fam_size, min_completeness, include_younger_fams, reduced_alphabet)

    elif step == 'ki':
        db_path = sys.argv[3]
        root_taxon = sys.argv[4]
        min_fam_size = int(sys.argv[5])
        min_completeness = float(sys.argv[6])
        include_younger_fams =  True if (sys.argv[7] == 'True') else False
        reduced_alphabet = True if (sys.argv[8] == 'True') else False
        hidden_taxa = [' '.join(x.split('_')) for x in sys.argv[9].split(',')]
        k = int(sys.argv[10])
        build_kmer_table(
            db_path, root_taxon, min_fam_size, min_completeness, include_younger_fams, reduced_alphabet, hidden_taxa, k)    

    else:
        print('unknown step')





