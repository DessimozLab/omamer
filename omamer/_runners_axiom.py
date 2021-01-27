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

def build_database_from_oma(
    db_path, root_taxon, min_fam_size, logic, min_fam_completeness, include_younger_fams, oma_db_fn, nwk_fn, overwrite):
    '''
    Parse OMA HOGs.
    '''
    db_fn = '{}{}_MinFamSize{}_{}_MinFamComp0{}{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], '' if include_younger_fams else '_woyf')

    if not is_complete(db_fn, db_path) or overwrite:
        if os.path.exists(db_fn):
            os.remove(db_fn)
            
        db = DatabaseFromOMA(
            filename=db_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, logic=logic, min_fam_completeness=min_fam_completeness,
            include_younger_fams=include_younger_fams, mode='w')

        # load sequences from OMA database
        db.build_database(oma_db_fn, nwk_fn)
        db.close()

        set_complete(db_fn, db_path)

def build_suffix_array(
    db_path, root_taxon, min_fam_size, logic, min_fam_completeness, include_younger_fams, reduced_alphabet, overwrite):
    '''
    Compute Suffix Array. 
    '''
    # reload database
    db_fn = '{}{}_MinFamSize{}_{}_MinFamComp0{}{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], '' if include_younger_fams else '_woyf')

    assert os.path.exists(db_fn), 'database missing'

    db = DatabaseFromOMA(
        filename=db_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, logic=logic, min_fam_completeness=min_fam_completeness,
        include_younger_fams=include_younger_fams, mode='r')

    # compute suffix array
    alphabet_n = 21 if not reduced_alphabet else 13    
    sa_fn = '{}SA_{}_MinFamSize{}_{}_MinFamComp0{}{}_A{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], '' if include_younger_fams else '_woyf', alphabet_n)

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
    db_path, root_taxon, min_fam_size, logic, min_fam_completeness, include_younger_fams, reduced_alphabet, k, hidden_taxa, overwrite):
    
    alphabet_n = 21 if not reduced_alphabet else 13

    db_fn = '{}{}_MinFamSize{}_{}_MinFamComp0{}{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], '' if include_younger_fams else '_woyf')
    
    sa_fn = '{}SA_{}_MinFamSize{}_{}_MinFamComp0{}{}_A{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], '' if include_younger_fams else '_woyf', alphabet_n)

    assert os.path.exists(db_fn), 'database missing'
    assert os.path.exists(sa_fn), 'suffix array missing'

    ki_fn = '{}{}_MinFamSize{}_{}_MinFamComp0{}{}_A{}_k{}{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], '' if include_younger_fams else '_woyf', alphabet_n, k, 
        '_wo_{}'.format('_'.join(['_'.join(x.split()) for x in hidden_taxa])) if hidden_taxa else '')

    if not is_complete(ki_fn, db_path) or overwrite:
        if os.path.exists(ki_fn):
            os.remove(ki_fn)

        # copy database
        shutil.copyfile(db_fn, ki_fn)

        # load in append mode
        db = DatabaseFromOMA(
            filename=ki_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, logic=logic, min_fam_completeness=min_fam_completeness,
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

# def is_in_oma(db_path, root_taxon, min_fam_size, logic, min_fam_completeness, alphabet_n, k, query_sp):
#     hidden_taxa = []
#     ki_fn = '{}{}_MinFamSize{}_{}_MinFamComp0{}_A{}_k{}{}.h5'.format(
#         db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], alphabet_n, k, 
#         '_wo_{}'.format('_'.join(['_'.join(x.split()) for x in hidden_taxa])) if hidden_taxa else '')
#     db = DatabaseFromOMA(
#         filename=ki_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, logic=logic, min_fam_completeness=min_fam_completeness, include_younger_fams=True)
#     return np.argwhere(db._sp_tab.col('ID') == query_sp.encode('ascii')).size > 0

def search(
    db_path, root_taxon, min_fam_size, logic, min_fam_completeness, reduced_alphabet, k, query_sp, proteome_fn, store_hdf5, out_path, ref_taxon, overwrite):
    
    alphabet_n = 21 if not reduced_alphabet else 13

    # # if proteome in OMA, hide it (NOTE sure about this because specific to taxonomic placement validation)
    # hidden_taxa = [query_sp] if is_in_oma(db_path, root_taxon, min_fam_size, logic, min_fam_completeness, query_sp) else []
    hidden_taxa = []

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

def search_validate(
    db_path, root_taxon, min_fam_size, logic, min_fam_completeness, include_younger_fams, reduced_alphabet, hidden_taxa, k,
    thresholds, oma_db_fn, nwk_fn, score, cum_mode, top_m_fams, val_mode, neg_root_taxon, focal_taxon, fam_bin_num, hog_bin_num, 
    pvalue_score, query_sp, overwrite, perm_nr, w_size, dist, comp_t, size_t):
    
    alphabet_n = 21 if not reduced_alphabet else 13
    
    # reload k-mer table
    ki_fn = '{}{}_MinFamSize{}_{}_MinFamComp0{}{}_A{}_k{}{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], '' if include_younger_fams else '_woyf', alphabet_n, k, 
        '_wo_{}'.format('_'.join(['_'.join(x.split()) for x in hidden_taxa])) if hidden_taxa else '')
    
    assert os.path.exists(ki_fn), 'index missing'

    # load 
    db = DatabaseFromOMA(
        filename=ki_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, min_fam_completeness=min_fam_completeness,
        include_younger_fams=include_younger_fams, mode='r')

    # setup search and validation steps
    se_va_fn = '{}{}_MinFamSize{}_{}_MinFamComp0{}{}_A{}_k{}{}_query_{}_{}_{}_top{}fams{}{}_{}_{}_{}_{}fbn_{}hbn_MinFamComp0{}_MinFamSize{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], 
        '' if include_younger_fams else '_woyf', alphabet_n, k, '_wo_{}'.format('_'.join(['_'.join(x.split()) for x in hidden_taxa])) if hidden_taxa else '',
        '_'.join(query_sp.split()), score, cum_mode, top_m_fams, '_{}perms_w{}'.format(perm_nr, w_size) if (score == 'nonparam_pvalue') or (score == 'nonparam_naive') else '', 
        '_{}'.format(dist) if (score == 'nonparam_pvalue') else '',
        val_mode, neg_root_taxon, focal_taxon, fam_bin_num, hog_bin_num, str(comp_t).split('.')[-1], size_t)    

    if not is_complete(se_va_fn, db_path) or overwrite:
        if os.path.exists(se_va_fn):
            os.remove(se_va_fn)
        fasta = '{}.fa'.format(se_va_fn.split('.h5')[0])
        if os.path.exists(fasta):
            os.remove(fasta)

        ms = MergeSearch(ki=db.ki, nthreads=1)

        # maximum number of queries (used for clade-specific negatives) is the proteome size
        sp_off = np.searchsorted(db._sp_tab.col('ID'), query_sp)
        max_query_nr = db._sp_tab[sp_off]['ProtNum']

        va = Validation(db, se_va_fn, thresholds, oma_db_fn=oma_db_fn, nwk_fn=nwk_fn, 
                        neg_query_file='{}.fa'.format(se_va_fn.split('.')[0]), nthreads=1, query_sp=query_sp, 
                        max_query_nr=max_query_nr, val_mode=val_mode, neg_root_taxon=neg_root_taxon, focal_taxon=focal_taxon, 
                        fam_bin_num=fam_bin_num, hog_bin_num=fam_bin_num, comp_t=comp_t, size_t=size_t)

        assert va.mode == 'w'

        # load query sequences (keep the ones from filtered families)
        fam_filter = va.fam_filter
        sbuff = QuerySequenceBuffer(db, query_sp, fam_filter=fam_filter)
        
        # search and validate
        chunksize = 10000

        ids = []
        seqs = []

        pbar = tqdm(desc='Searching')
        for i, q in enumerate(sbuff.ids):
            ids.append(q)
            seqs.append(sbuff[i])
            if len(ids) == chunksize:
                # search and validate the chunk
                ms.merge_search(seqs=seqs, ids=ids, score=score, cum_mode=cum_mode, top_m_fams=top_m_fams, perm_nr=perm_nr, w_size=w_size, dist=dist,
                    fam_filter=fam_filter) 
                va.validate(ms, score=score, cum_mode=cum_mode, top_m_fams=top_m_fams, pvalue_score=pvalue_score, 
                    perm_nr=perm_nr, w_size=w_size, dist=dist)     

                pbar.update(len(ids))
                ids = []
                seqs = []

        # search and validate last chunk
        if len(ids) > 0:
            ms.merge_search(seqs=seqs, ids=ids, score=score, cum_mode=cum_mode, top_m_fams=top_m_fams, perm_nr=perm_nr, w_size=w_size, dist=dist,
                fam_filter=fam_filter) 
            va.validate(ms, score=score, cum_mode=cum_mode, top_m_fams=top_m_fams, pvalue_score=pvalue_score, perm_nr=perm_nr, w_size=w_size, dist=dist) 
            pbar.update(len(ids))

        # close stuff
        pbar.close()
        va.va.close()

        set_complete(se_va_fn, db_path)


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
include_younger_fams=$7
oma_path=$8
overwrite=$9

source /scratch/axiom/FAC/FBM/DBC/cdessim2/default/vrossie4/miniconda3/bin/activate omamer

python ${{omamer_path}}omamer/_runners_axiom.py ${{omamer_path}} parse_hogs ${{db_path}} ${{root_taxon}} ${{min_fam_size}} ${{logic}} ${{min_completeness}} ${{include_younger_fams}} ${{oma_path}} ${{overwrite}}

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
include_younger_fams=$7
reduced_alphabet=$8
overwrite=$9

source /scratch/axiom/FAC/FBM/DBC/cdessim2/default/vrossie4/miniconda3/bin/activate omamer

python ${{omamer_path}}omamer/_runners_axiom.py ${{omamer_path}} suffix_array ${{db_path}} ${{root_taxon}} ${{min_fam_size}} ${{logic}} ${{min_completeness}} ${{include_younger_fams}} ${{reduced_alphabet}} ${{overwrite}}

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
include_younger_fams=$7
reduced_alphabet=$8
k=$9
hidden_taxa=${{10}}
overwrite=${{11}}

source /scratch/axiom/FAC/FBM/DBC/cdessim2/default/vrossie4/miniconda3/bin/activate omamer

python ${{omamer_path}}omamer/_runners_axiom.py ${{omamer_path}} kmer_table ${{db_path}} ${{root_taxon}} ${{min_fam_size}} ${{logic}} ${{min_completeness}} ${{include_younger_fams}} ${{reduced_alphabet}} ${{k}} ${{hidden_taxa}} ${{overwrite}}
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
    
    elif step == 'search_validate':
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
include_younger_fams=$7
reduced_alphabet=$8
hidden_taxa=$9
k=${{10}}
oma_path=${{11}}
score=${{12}}
cum_mode=${{13}}
top_m_fams=${{14}}
val_mode=${{15}}
neg_root_taxon=${{16}}
focal_taxon=${{17}}
fam_bin_num=${{18}}
hog_bin_num=${{19}}
query_sp=${{20}}
overwrite=${{21}}
perm_nr=${{22}}
w_size=${{23}}
dist=${{24}}
comp_t=${{25}}
size_t=${{26}}

source /scratch/axiom/FAC/FBM/DBC/cdessim2/default/vrossie4/miniconda3/bin/activate omamer

python ${{omamer_path}}omamer/_runners_axiom.py ${{omamer_path}} search_validate ${{db_path}} ${{root_taxon}} ${{min_fam_size}} ${{logic}} ${{min_completeness}} ${{include_younger_fams}} ${{reduced_alphabet}} ${{hidden_taxa}} ${{k}} ${{oma_path}} ${{score}} ${{cum_mode}} ${{top_m_fams}} ${{val_mode}} ${{neg_root_taxon}} ${{focal_taxon}} ${{fam_bin_num}} ${{hog_bin_num}} ${{query_sp}} ${{overwrite}} ${{perm_nr}} ${{w_size}} ${{dist}} ${{comp_t}} ${{size_t}}
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
        include_younger_fams = True if (sys.argv[8] == 'True') else False
        oma_path = sys.argv[9]
        oma_db_fn = os.path.join(oma_path, "OmaServer.h5")
        nwk_fn = os.path.join(oma_path, "speciestree.nwk")
        overwrite = True if (sys.argv[10] == 'True') else False
        build_database_from_oma(
            db_path, root_taxon, min_fam_size, logic, min_fam_completeness, include_younger_fams, oma_db_fn, nwk_fn, overwrite)
    
    elif step == 'suffix_array':
        db_path = sys.argv[3]
        root_taxon = sys.argv[4]
        min_fam_size = int(sys.argv[5])
        logic = sys.argv[6]
        min_fam_completeness = float(sys.argv[7])
        include_younger_fams = True if (sys.argv[8] == 'True') else False
        reduced_alphabet = True if (sys.argv[9] == 'True') else False
        overwrite =  True if (sys.argv[10] == 'True') else False
        build_suffix_array(
            db_path, root_taxon, min_fam_size, logic, min_fam_completeness, include_younger_fams, reduced_alphabet, overwrite)

    elif step == 'kmer_table':
        db_path = sys.argv[3]
        root_taxon = sys.argv[4]
        min_fam_size = int(sys.argv[5])
        logic = sys.argv[6]
        min_fam_completeness = float(sys.argv[7])
        include_younger_fams = True if (sys.argv[8] == 'True') else False
        reduced_alphabet = True if (sys.argv[9] == 'True') else False
        k = int(sys.argv[10])
        hidden_taxa = [' '.join(x.split('_')) for x in sys.argv[11].split(',')] if sys.argv[11] != 'na' else []
        overwrite =  True if (sys.argv[12] == 'True') else False
        build_kmer_table(
            db_path, root_taxon, min_fam_size, logic, min_fam_completeness, include_younger_fams, reduced_alphabet, k, hidden_taxa, overwrite)   

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

    elif step == 'search_validate':
        db_path = sys.argv[3]
        root_taxon = sys.argv[4]
        min_fam_size = int(sys.argv[5])
        logic = sys.argv[6]
        min_fam_completeness = float(sys.argv[7])
        include_younger_fams =  True if (sys.argv[8] == 'True') else False
        reduced_alphabet = True if (sys.argv[9] == 'True') else False
        hidden_taxa = [' '.join(x.split('_')) for x in sys.argv[10].split(',')]
        k = int(sys.argv[11])
        # thresholds = np.arange(*(float(x) for x in sys.argv[11].split(',')))  # e.g. "0,1.01,0.01"
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
        perm_nr = int(sys.argv[23])
        w_size = int(sys.argv[24])
        dist = sys.argv[25]
        comp_t = float(sys.argv[26])
        size_t = int(sys.argv[27])

        if score in {'mash_pvalue', 'kmerfreq_pvalue', 'nonparam_pvalue'}:
            thresholds = np.concatenate((np.arange(-1000, -9, 10), np.arange(-10, -0.9, 1), np.arange(-1, -0.09, 0.1), np.arange(-0.1, -0.009, 0.01)))
            pvalue_score = True
        else:
            thresholds = np.arange(0, 1.01, 0.01)
            #thresholds = np.arange(0.001, 1.001, 0.005) the one used in OMAmer paper
            pvalue_score = False

        search_validate(
            db_path, root_taxon, min_fam_size, logic, min_fam_completeness, include_younger_fams, reduced_alphabet, hidden_taxa, k,
            thresholds, oma_db_fn, nwk_fn, score, cum_mode, top_m_fams, val_mode, neg_root_taxon, focal_taxon, fam_bin_num, hog_bin_num, 
            pvalue_score, query_sp, overwrite, perm_nr, w_size, dist, comp_t, size_t)
    else:
        print('unknown step')
