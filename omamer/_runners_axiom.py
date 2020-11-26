import sys
omamer_path = sys.argv[1]
sys.path.insert(0, omamer_path)

from omamer.database import DatabaseFromOMA
from omamer.index import (
    Index, 
    QuerySequenceBuffer, 
    SequenceBufferFasta
)
from omamer.alphabets import Alphabet
from omamer.merge_search import MergeSearch
from omamer.validation import Validation

import os
import tables
import shutil
import numpy as np
from tqdm import tqdm

'''
    General functions to run OMAmer on axiom (e.g. from notebook).
'''

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

def is_in_oma(db_path, root_taxon, min_fam_size, min_fam_completeness, query_sp):
    ki_fn = '{}{}_MinFamSize{}_MinFamComp0{}.h5'.format(
        db_path, root_taxon, min_fam_size, str(min_fam_completeness).split('.')[-1])
    db = DatabaseFromOMA(
        filename=ki_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, min_fam_completeness=min_fam_completeness, include_younger_fams=True)
    return np.argwhere(db._sp_tab.col('ID') == query_sp.encode('ascii')).size > 0

def search(db_path, root_taxon, min_fam_size, min_fam_completeness, query_sp, overwrite, proteome_fn):

    # if proteome in OMA, hide it
    hidden_taxa = [query_sp] if is_in_oma(db_path, root_taxon, min_fam_size, min_fam_completeness, query_sp) else []

    # reload k-mer table
    ki_fn = '{}{}_MinFamSize{}_MinFamComp0{}{}.h5'.format(
        db_path, root_taxon, min_fam_size, str(min_fam_completeness).split('.')[-1], 
        '_wo_{}'.format('_'.join(['_'.join(x.split()) for x in hidden_taxa])) if hidden_taxa else '')
    assert os.path.exists(ki_fn), 'index missing'
    
    db = DatabaseFromOMA(
        filename=ki_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, min_fam_completeness=min_fam_completeness, include_younger_fams=True)

    # search
    ms_fn = '{}{}_MinFamSize{}_MinFamComp0{}{}_query_{}.h5'.format(
        db_path, root_taxon, min_fam_size, str(min_fam_completeness).split('.')[-1], 
        '_wo_{}'.format('_'.join(['_'.join(x.split()) for x in hidden_taxa])) if hidden_taxa else '', '_'.join(query_sp.split()))
    
    if not is_complete(ms_fn, db_path) or overwrite:
        if os.path.exists(ms_fn):
            os.remove(ms_fn)
        
        ms = MergeSearch(ki=db.ki, nthreads=1)
        if hidden_taxa:
            sbuff = QuerySequenceBuffer(db=db, query_sp=query_sp)
        else:
            sbuff = SequenceBufferFasta(proteome_fn)
        ms.merge_search(seqs=[s for s in sbuff], ids=list(sbuff.ids), fasta_file=None, score='nonparam_naive', cum_mode='max', top_m_fams=100, 
            top_n_fams=1, perm_nr=1, w_size=6, dist='poisson', fam_filter=np.array([], dtype=np.int64))
        ms.store_results(ms_fn)
        set_complete(ms_fn, db_path)

def write_axiom_script(step, tmp_path, mem, hour_nr, oe_path):

    if step == 'omamer_search':
        with open('{}run_{}.sh'.format(tmp_path, step), 'w') as inf:
            inf.write(
"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={}G
#SBATCH --time={}:00:00
#SBATCH --job-name={}
#SBATCH --partition=axiom
#SBATCH --output={}%x_%j.out
#SBATCH --error={}%x_%j.err

omamer_path=$1
db_path=$2
root_taxon=$3
min_fam_size=$4
min_completeness=$5
query_sp=$6
overwrite=$7
proteome_fn=$8

source /scratch/axiom/FAC/FBM/DBC/cdessim2/default/vrossie4/miniconda3/bin/activate omamer

python ${{omamer_path}}/_runners_axiom.py ${{omamer_path}} omamer_search ${{db_path}} ${{root_taxon}} ${{min_fam_size}} ${{min_completeness}} ${{query_sp}} ${{overwrite}} ${{proteome_fn}}

sacct -j %j --format=jobname,maxrss,elapsed""".format(mem, hour_nr, step, oe_path, oe_path))

if __name__ == "__main__":

    step = sys.argv[2]
    if step == 'omamer_search':
        db_path = sys.argv[3]
        root_taxon = sys.argv[4]
        min_fam_size = int(sys.argv[5])
        min_fam_completeness = float(sys.argv[6])
        query_sp = ' '.join(sys.argv[7].split('_'))
        overwrite = True if (sys.argv[8] == 'True') else False
        proteome_fn = sys.argv[9]

        search(db_path, root_taxon, min_fam_size, min_fam_completeness, query_sp, overwrite, proteome_fn)

