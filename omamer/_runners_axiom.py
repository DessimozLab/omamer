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
import pickle
import tables
import shutil
import numpy as np
from tqdm import tqdm

'''
General functions to run OMAmer on axiom (e.g. from notebook).
'''

def is_complete(fn, path):
    cf = '{}COMPLETE.txt'.format(path)
    if not os.path.exists(cf):
        with open(cf, 'w') as inf:
            pass
    with open(cf, 'r') as inf:
        complete_fns = set(map(lambda x: x.rstrip(), inf.readlines()))
    return fn in complete_fns

def set_complete(fn, path):
    with open('{}COMPLETE.txt'.format(path), 'a') as inf:
        inf.write('{}\n'.format(fn))

def build_database_from_oma(db_path, root_taxon, min_fam_size, logic, min_fam_completeness, oma_db_fn, nwk_fn, overwrite):
    '''
    Parse OMA HOGs.
    '''
    db_fn = '{}{}_MinFamSize{}_{}_MinFamComp0{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1])

    if not is_complete(db_fn, db_path) or overwrite:
        if os.path.exists(db_fn):
            os.remove(db_fn)
            
        db = DatabaseFromOMA(
            filename=db_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, logic=logic, min_fam_completeness=min_fam_completeness,
            include_younger_fams=True, mode='w')

        # load sequences from OMA database
        db.build_database(oma_db_fn, nwk_fn)
        db.close()

        set_complete(db_fn, db_path)

def build_suffix_array(db_path, root_taxon, min_fam_size, logic, min_fam_completeness, reduced_alphabet, overwrite):
    '''
    Compute Suffix Array. 
    '''
    # reload database
    db_fn = '{}{}_MinFamSize{}_{}_MinFamComp0{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1])

    assert os.path.exists(db_fn), 'database missing'

    db = DatabaseFromOMA(
        filename=db_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, logic=logic, min_fam_completeness=min_fam_completeness,
        include_younger_fams=True, mode='r')

    # compute suffix array
    alphabet_n = 21 if not reduced_alphabet else 13    
    sa_fn = '{}SA_{}_MinFamSize{}_{}_MinFamComp0{}_A{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], alphabet_n)

    if not is_complete(sa_fn, db_path) or overwrite:
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
    db_path, root_taxon, min_fam_size, logic, min_fam_completeness, reduced_alphabet, k, hidden_taxa, overwrite):
    
    alphabet_n = 21 if not reduced_alphabet else 13

    db_fn = '{}{}_MinFamSize{}_{}_MinFamComp0{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1])
    
    sa_fn = '{}SA_{}_MinFamSize{}_{}_MinFamComp0{}_A{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], alphabet_n)

    assert os.path.exists(db_fn), 'database missing'
    assert os.path.exists(sa_fn), 'suffix array missing'

    ki_fn = '{}{}_MinFamSize{}_{}_MinFamComp0{}_A{}_k{}{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], alphabet_n, k, 
        '_wo_{}'.format('_'.join(['_'.join(x.split()) for x in hidden_taxa])) if hidden_taxa else '')

    if not is_complete(ki_fn, db_path) or overwrite:
        if os.path.exists(ki_fn):
            os.remove(ki_fn)

        # copy database
        shutil.copyfile(db_fn, ki_fn)

        # load in append mode
        db = DatabaseFromOMA(
            filename=ki_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, logic=logic, min_fam_completeness=min_fam_completeness,
            include_younger_fams=True, mode='a')   

        # load suffix array
        sa_h5 = tables.open_file(sa_fn, 'r', filters=db._compr)
        sa = sa_h5.root.SuffixArray[:]
        
        # build index
        ki = Index(db, k=k, reduced_alphabet=reduced_alphabet, hidden_taxa=hidden_taxa)
        ki._build_kmer_table(sa)
        
        db.close()
        sa_h5.close()

        set_complete(ki_fn, db_path)

def is_in_oma(db_path, root_taxon, min_fam_size, logic, min_fam_completeness, query_sp):
    ki_fn = '{}{}_MinFamSize{}_{}_MinFamComp0{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1])
    db = DatabaseFromOMA(
        filename=ki_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, logic=logic, min_fam_completeness=min_fam_completeness, include_younger_fams=True)
    return np.argwhere(db._sp_tab.col('ID') == query_sp.encode('ascii')).size > 0

def search(
    db_path, root_taxon, min_fam_size, logic, min_fam_completeness, reduced_alphabet, k, query_sp, proteome_fn, store_hdf5, out_path, ref_taxon, overwrite):
    
    alphabet_n = 21 if not reduced_alphabet else 13

    # if proteome in OMA, hide it (NOTE sure about this because specific to taxonomic placement validation)
    hidden_taxa = [query_sp] if is_in_oma(db_path, root_taxon, min_fam_size, logic, min_fam_completeness, query_sp) else []

    # reload k-mer table
    ki_fn = '{}{}_MinFamSize{}_{}_MinFamComp0{}_A{}_k{}{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], alphabet_n, k, 
        '_wo_{}'.format('_'.join(['_'.join(x.split()) for x in hidden_taxa])) if hidden_taxa else '')

    assert os.path.exists(ki_fn), '{} missing'.format(ki_fn)
    
    db = DatabaseFromOMA(
        filename=ki_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, logic=logic, min_fam_completeness=min_fam_completeness, include_younger_fams=True)

    # search
    ms_fn = '{}_MinFamSize{}_{}_MinFamComp0{}_A{}_k{}{}_query_{}'.format(
        root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], alphabet_n, k, 
        '_wo_{}'.format('_'.join(['_'.join(x.split()) for x in hidden_taxa])) if hidden_taxa else '', '_'.join(query_sp.split()))
    
    tsv_fn = '{}{}.tsv'.format(out_path, ms_fn)
    hdf5_fn = '{}{}.h5'.format(db_path, ms_fn)
    if not is_complete(tsv_fn, out_path) or overwrite:
        if os.path.exists(tsv_fn):
            os.remove(tsv_fn)
        if os.path.exists(hdf5_fn):
            os.remove(hdf5_fn)

        # search
        ms = MergeSearch(ki=db.ki, nthreads=1)
        sbuff = SequenceBufferFasta(proteome_fn)
        ms.merge_search(seqs=[s for s in sbuff], ids=list(sbuff.ids), fasta_file=None, score='nonparam_naive', cum_mode='max', top_m_fams=100, 
            top_n_fams=1, perm_nr=1, w_size=6, dist='poisson', fam_filter=np.array([], dtype=np.int64))

        # export results
        df = ms.output_results(overlap=0, fst=0, sst=0, ref_taxon=ref_taxon)
        if df.size >0:
            df.to_csv(tsv_fn, sep='\t', index=False, header=False)
            if store_hdf5:
                ms.store_results(hdf5_fn)
            set_complete(tsv_fn, out_path)

def write_axiom_script(step, name, tmp_path, mem, hour_nr, oe_path):

    if step == 'parse_hogs':
        with open('{}run_{}.sh'.format(tmp_path, name), 'w') as inf:
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
logic=$5
min_completeness=$6
oma_path=$7
overwrite=$8

source /scratch/axiom/FAC/FBM/DBC/cdessim2/default/vrossie4/miniconda3/bin/activate omamer

python ${{omamer_path}}omamer/_runners_axiom.py ${{omamer_path}} parse_hogs ${{db_path}} ${{root_taxon}} ${{min_fam_size}} ${{logic}} ${{min_completeness}} ${{oma_path}} ${{overwrite}}

sstat -j ${{SLURM_JOBID}}.batch --format=MaxRSS
sacct -j ${{SLURM_JOBID}}.batch --format=elapsed""".format(mem, hour_nr, name, oe_path, oe_path))

    elif step == 'suffix_array':
        with open('{}run_{}.sh'.format(tmp_path, name), 'w') as inf:
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
logic=$5
min_completeness=$6
reduced_alphabet=$7
overwrite=$8

source /scratch/axiom/FAC/FBM/DBC/cdessim2/default/vrossie4/miniconda3/bin/activate omamer

python ${{omamer_path}}omamer/_runners_axiom.py ${{omamer_path}} suffix_array ${{db_path}} ${{root_taxon}} ${{min_fam_size}} ${{logic}} ${{min_completeness}} ${{reduced_alphabet}} ${{overwrite}}

sstat -j ${{SLURM_JOBID}}.batch --format=MaxRSS
sacct -j ${{SLURM_JOBID}}.batch --format=elapsed""".format(mem, hour_nr, name, oe_path, oe_path))

    elif step == 'kmer_table':
        with open('{}run_{}.sh'.format(tmp_path, name), 'w') as inf:
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
logic=$5
min_completeness=$6
reduced_alphabet=$7
k=$8
hidden_taxa=$9
overwrite=${{10}}

source /scratch/axiom/FAC/FBM/DBC/cdessim2/default/vrossie4/miniconda3/bin/activate omamer

python ${{omamer_path}}omamer/_runners_axiom.py ${{omamer_path}} kmer_table ${{db_path}} ${{root_taxon}} ${{min_fam_size}} ${{logic}} ${{min_completeness}} ${{reduced_alphabet}} ${{k}} ${{hidden_taxa}} ${{overwrite}}
sstat -j ${{SLURM_JOBID}}.batch --format=MaxRSS
sacct -j ${{SLURM_JOBID}}.batch --format=elapsed""".format(mem, hour_nr, name, oe_path, oe_path))

    elif step == 'omamer_search':
        with open('{}run_{}.sh'.format(tmp_path, name), 'w') as inf:
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
logic=$5
min_completeness=$6
reduced_alphabet=$7
k=$8
query_sp=$9
proteome_fn=${{10}}
store_hdf5=${{11}}
out_path=${{12}}
ref_taxon=${{13}}
overwrite=${{14}}

source /scratch/axiom/FAC/FBM/DBC/cdessim2/default/vrossie4/miniconda3/bin/activate omamer

python ${{omamer_path}}omamer/_runners_axiom.py ${{omamer_path}} omamer_search ${{db_path}} ${{root_taxon}} ${{min_fam_size}} ${{logic}} ${{min_completeness}} ${{reduced_alphabet}} ${{k}} ${{query_sp}} ${{proteome_fn}} ${{store_hdf5}} ${{out_path}} ${{ref_taxon}} ${{overwrite}}

sstat -j ${{SLURM_JOBID}}.batch --format=MaxRSS
sacct -j ${{SLURM_JOBID}}.batch --format=elapsed""".format(mem, hour_nr, name, oe_path, oe_path))

if __name__ == "__main__":

    step = sys.argv[2]
    if step == 'parse_hogs':
        db_path = sys.argv[3]
        root_taxon = sys.argv[4]
        min_fam_size = int(sys.argv[5])
        logic = sys.argv[6]
        min_fam_completeness = float(sys.argv[7])
        oma_path = sys.argv[8]
        oma_db_fn = os.path.join(oma_path, "OmaServer.h5")
        nwk_fn = os.path.join(oma_path, "speciestree.nwk")
        overwrite =  True if (sys.argv[9] == 'True') else False
        build_database_from_oma(
            db_path, root_taxon, min_fam_size, logic, min_fam_completeness, oma_db_fn, nwk_fn, overwrite)
    
    elif step == 'suffix_array':
        db_path = sys.argv[3]
        root_taxon = sys.argv[4]
        min_fam_size = int(sys.argv[5])
        logic = sys.argv[6]
        min_fam_completeness = float(sys.argv[7])
        reduced_alphabet = True if (sys.argv[8] == 'True') else False
        overwrite =  True if (sys.argv[9] == 'True') else False
        build_suffix_array(
            db_path, root_taxon, min_fam_size, logic, min_fam_completeness, reduced_alphabet, overwrite)

${{omamer_path}} kmer_table ${{db_path}} ${{root_taxon}} ${{min_fam_size}} ${{logic}} ${{min_completeness}} ${{reduced_alphabet}} ${{k}} ${{hidden_taxa}} ${{overwrite}}

    elif step == 'kmer_table':
        db_path = sys.argv[3]
        root_taxon = sys.argv[4]
        min_fam_size = int(sys.argv[5])
        logic = sys.argv[6]
        min_fam_completeness = float(sys.argv[7])
        reduced_alphabet = True if (sys.argv[8] == 'True') else False
        k = int(sys.argv[9])
        hidden_taxa = [' '.join(x.split('_')) for x in sys.argv[10].split(',')] if sys.argv[10] != 'na' else []
        overwrite =  True if (sys.argv[11] == 'True') else False
        build_kmer_table(
            db_path, root_taxon, min_fam_size, logic, min_fam_completeness, reduced_alphabet, k, hidden_taxa, overwrite)   

    elif step == 'omamer_search':
        db_path = sys.argv[3]
        root_taxon = sys.argv[4]
        min_fam_size = int(sys.argv[5])
        logic = sys.argv[6]
        min_fam_completeness = float(sys.argv[7])
        reduced_alphabet = True if (sys.argv[8] == 'True') else False
        k = int(sys.argv[9])
        query_sp = ' '.join(sys.argv[10].split('_'))
        proteome_fn = sys.argv[11]
        store_hdf5 = True if (sys.argv[12] == 'True') else False
        out_path = sys.argv[13]
        ref_taxon = sys.argv[14]
        overwrite = True if (sys.argv[15] == 'True') else False

        search(
            db_path, root_taxon, min_fam_size, logic, min_fam_completeness, reduced_alphabet, k, query_sp, proteome_fn, store_hdf5, 
            out_path, ref_taxon, overwrite)
