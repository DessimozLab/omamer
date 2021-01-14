'''
Code used to benchmark closest sequence methods, i.e. DIAMOND and Smith-Waterman precomputed in OMA database during all-all.
Act also as _runners_axiom.py for these methods.ß
'''
import os
import sys
omamer_path = sys.argv[1]
sys.path.insert(0, omamer_path)

import tables
import numpy as np
from Bio import SeqIO, SearchIO

from omamer.database import DatabaseFromOMA
from omamer.index import SequenceBufferFasta
from omamer.hierarchy import get_root_leaf_offsets
from omamer.validation import Validation
from omamer._runners_axiom import (
	is_complete, 
	set_complete
)

class DIAMONDsearch():
    '''
    To parse DIAMOND output and mimic OMAmer MergeSearch object as input to validation.
    '''
    def __init__(self, db, query_prots, diamond_output, fasta=None, qr_name=None, mode=None):
                
        # load ki and db
        self.db = db
        self.ki = db.ki
        
        self.query_prots = query_prots
        self.diamond_output = diamond_output

    @staticmethod
    def parse_blast_result(diamond_output, prot_tab, score):

        qi_rows = []
        res_rows = []

        result = SearchIO.parse(diamond_output, 'blast-tab')

        for res in result:
            qi = res.id
            qi_rows.append(qi)
            ti = int(res[0].id)
            s = (res[0][0].bitscore if score=='bitscore' else res[0][0].evalue)
            res_rows.append((int(qi), prot_tab[ti]['FamOff'], prot_tab[ti]['HOGoff'], s))

        return qi_rows, res_rows

    @staticmethod
    def interleave_na_results(query_ids, tmp_res_rows, prob):
        '''
        some query might not have any hit with BLAST/DIAMOND but should still be present in the result table
        '''
        qi2res = dict([ (k, (x, y, z)) for k, x, y, z in tmp_res_rows])
        return [(int(qi), *qi2res.get(int(qi), (-1, -1, 1000000 if prob else -1))) for qi in query_ids.flatten()]

    def import_blast_result(self, score='evalue', prob=True):

        qi_rows, tmp_res_rows = self.parse_blast_result(self.diamond_output, self.db._prot_tab[:], 'evalue')

        sb = SequenceBufferFasta(fasta_file=self.query_prots)
        query_ids = sb.ids

        if len(query_ids) != len(tmp_res_rows):
            res_rows = self.interleave_na_results(query_ids, tmp_res_rows, True)
        else:
            res_rows = tmp_res_rows
        
        # convert rows into column arrays
        query_ids, best_fam, best_hog, score =  [list(x) for x in zip(*res_rows)]
        
        ## family arrays
        queryFam_ranked = np.zeros((len(query_ids), 1), dtype=np.int64)
        queryFam_scores = np.zeros((len(query_ids), 1), dtype=np.float64)
        queryFam_ranked[:, 0] = best_fam
        queryFam_scores[:, 0] = score
        
        ## sub-family 2D arrays
        max_hog_nr = np.int64(np.max(self.db._fam_tab.col('HOGnum')))
        fam_tab = self.db._fam_tab[:]
        hog_tab = self.db._hog_tab[:]

        # scores and bestpath mask for HOGs of top_n_fam
        queryRankHog_scores = np.zeros((1, len(query_ids), max_hog_nr), dtype=np.float64)
        queryRankHog_bestpath = np.zeros((1, len(query_ids), max_hog_nr), dtype=np.bool8)  

        for i, fam_off in enumerate(best_fam):

            # skip if no family assignment
            if fam_off == -1:
                continue

            # tag all ancestral HOGs of the best hog as on best path and with the common score
            fam_ent = fam_tab[fam_off]
            fam_hog_off = np.int64(fam_ent['HOGoff'])
            fam_hog_nr = np.int64(fam_ent['HOGnum'])

            fam_bestpath = get_root_leaf_offsets(best_hog[i], hog_tab['ParentOff']) - fam_hog_off

            queryRankHog_scores[0, i][fam_bestpath] = score[i]
            queryRankHog_bestpath[0, i][fam_bestpath] = True

        self._query_ids = np.array(query_ids, dtype=np.uint64)
        self._queryFam_ranked = queryFam_ranked
        self._queryFam_scores = queryFam_scores
        self._queryRankHog_bestpath = queryRankHog_bestpath
        self._queryRankHog_scores = queryRankHog_scores


class DIAMONDvalidation(Validation):
    '''
    Copy of the OMAmer Validation class with 2 function changes:
     - validate
     - validate_family (remove OMAmer search of negatives)
    '''
    def __init__(
        self, db, filename, thresholds, oma_db_fn=None, nwk_fn=None, neg_query_file=None, nthreads=1, query_sp=None, 
        max_query_nr=None, val_mode='golike', neg_root_taxon='random', focal_taxon=None, fam_bin_num=1, hog_bin_num=1, comp_t=0, size_t=0):
        super().__init__(
            db, filename, thresholds, oma_db_fn, nwk_fn, neg_query_file, nthreads, query_sp, max_query_nr, val_mode, 
            neg_root_taxon, focal_taxon, fam_bin_num, hog_bin_num, comp_t, size_t)
    
    def validate(self, se_pos, se_neg, pvalue_score, hog2bin=True):
        '''
        validate both family and subfamily levels
        '''
        self.validate_family(se_pos, se_neg, pvalue_score)
        self.validate_subfamily(se_pos, hog2bin, pvalue_score)

    def validate_family(self, se_pos, se_neg, pvalue_score):

        # validate negatives
        tn_query2tresh, fp_neg_query2tresh = self._validate_negative(
            self._thresholds[:], se_neg._queryFam_ranked, se_neg._queryFam_scores, pvalue_score)

        # validate positives
        tp_query2tresh, fn_query2tresh, fp_pos_query2tresh = self._validate_positive(
            se_pos._query_ids, self._thresholds[:], se_pos._queryFam_ranked, se_pos._queryFam_scores, self.db._prot_tab[:], pvalue_score, self.fam_filter_lca)

        # store results
        self._fam_tn.append(tn_query2tresh)
        self._fam_fp_neg.append(fp_neg_query2tresh)
        self._fam_tp.append(tp_query2tresh)
        self._fam_fn.append(fn_query2tresh)
        self._fam_fp_pos.append(fp_pos_query2tresh)
        self._fam_tn.flush()
        self._fam_fp_neg.flush()
        self._fam_tp.flush()
        self._fam_fn.flush()
        self._fam_fp_pos.flush()


def load_db_ki(
    db_path, root_taxon, min_fam_size, logic, min_fam_completeness, include_younger_fams, reduced_alphabet, k, hidden_taxa):
    alphabet_n = 21 if not reduced_alphabet else 13

    db_ki_fn = '{}{}_MinFamSize{}_{}_MinFamComp0{}{}_A{}_k{}{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], '' if include_younger_fams else '_woyf', alphabet_n, k, 
        '_wo_{}'.format('_'.join(['_'.join(x.split()) for x in hidden_taxa])) if hidden_taxa else '')
    
    # reload k-mer table
    db = DatabaseFromOMA(
        filename=db_ki_fn, root_taxon=root_taxon, min_fam_size=min_fam_size, min_fam_completeness=min_fam_completeness,
        include_younger_fams=include_younger_fams, mode='r')
    
    return db

def parse_validate_diamond(
	db_path, root_taxon, min_fam_size, logic, min_fam_completeness, include_younger_fams, reduced_alphabet, k, 
	hidden_taxa, query_sp, val_mode, neg_root_taxon, focal_taxon, fam_bin_num, hog_bin_num, comp_t, size_t, 
	diamond_thresholds, nwk_fn, overwrite):
	alphabet_n = 21 if not reduced_alphabet else 13

	# check if already computed
	se_va_fn = '{}{}_MinFamSize{}_{}_MinFamComp0{}{}_A{}_k{}{}_query_{}_DIAMOND_{}_{}_{}_{}fbn_{}hbn_MinFamComp0{}_MinFamSize{}.h5'.format(
	    db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], 
	    '' if include_younger_fams else '_woyf', alphabet_n, k, '_wo_{}'.format('_'.join(['_'.join(x.split()) for x in hidden_taxa])) if hidden_taxa else '',
	    '_'.join(query_sp.split()), val_mode, neg_root_taxon, focal_taxon, fam_bin_num, hog_bin_num, str(comp_t).split('.')[-1], size_t)

	if not is_complete(se_va_fn, db_path) or overwrite:
		if os.path.exists(se_va_fn):
		    os.remove(se_va_fn)

		# reload k-mer table
		db = load_db_ki(
		    db_path, root_taxon, min_fam_size, logic, min_fam_completeness, include_younger_fams, reduced_alphabet, k, hidden_taxa)

		# parse negatives
		query_prots = '{}{}_MinFamSize{}_{}_MinFamComp0{}{}_A{}_k{}{}.{}_prots'.format(
		    db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], 
		    '' if include_younger_fams else '_woyf', alphabet_n, k, '_wo_{}'.format('_'.join(['_'.join(x.split()) for x in hidden_taxa])) if hidden_taxa else '',
		    neg_root_taxon)

		diamond_out = '{}{}_MinFamSize{}_{}_MinFamComp0{}{}_A{}_k{}{}.diamond_neg_{}'.format(
		    db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], 
		    '' if include_younger_fams else '_woyf', alphabet_n, k, '_wo_{}'.format('_'.join(['_'.join(x.split()) for x in hidden_taxa])) if hidden_taxa else '',
		    neg_root_taxon)

		se_neg = DIAMONDsearch(db, query_prots=query_prots, diamond_output=diamond_out)
		se_neg.import_blast_result(score='evalue', prob=True)

		# parse positives
		query_prots = '{}{}_MinFamSize{}_{}_MinFamComp0{}{}_A{}_k{}{}.query_prots'.format(
		    db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], 
		    '' if include_younger_fams else '_woyf', alphabet_n, k, '_wo_{}'.format('_'.join(['_'.join(x.split()) for x in hidden_taxa])) if hidden_taxa else '',
		    neg_root_taxon)

		diamond_out = '{}{}_MinFamSize{}_{}_MinFamComp0{}{}_A{}_k{}{}.diamond_pos'.format(
		    db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], 
		    '' if include_younger_fams else '_woyf', alphabet_n, k, '_wo_{}'.format('_'.join(['_'.join(x.split()) for x in hidden_taxa])) if hidden_taxa else '',
		    neg_root_taxon)

		se_pos = DIAMONDsearch(db, query_prots=query_prots, diamond_output=diamond_out)
		se_pos.import_blast_result(score='evalue', prob=True)

		# validate 
		va = DIAMONDvalidation(
		    db, se_va_fn, diamond_thresholds, oma_db_fn=None, nwk_fn=nwk_fn, 
		    neg_query_file=None, nthreads=1, query_sp=query_sp, max_query_nr=None, val_mode=val_mode, 
		    neg_root_taxon=neg_root_taxon, focal_taxon=focal_taxon, fam_bin_num=fam_bin_num, hog_bin_num=fam_bin_num, 
		    comp_t=comp_t, size_t=size_t)

		va.validate(se_pos, se_neg, pvalue_score=True, hog2bin=True)
		va.va.close()
		set_complete(se_va_fn, db_path)

def write_cs_script(step, name, tmp_path, mem, hour_nr, oe_path):

    if step == 'parse_validate_diamond':
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
query_sp=${{11}}
val_mode=${{12}}
neg_root_taxon=${{13}}
focal_taxon=${{14}}
fam_bin_num=${{15}}
hog_bin_num=${{16}}
comp_t=${{17}}
size_t=${{18}}
nwk_fn=${{19}}
overwrite=${{20}}

source /scratch/axiom/FAC/FBM/DBC/cdessim2/default/vrossie4/miniconda3/bin/activate omamer

python ${{omamer_path}}omamer/benchmark.py ${{omamer_path}} parse_validate_diamond ${{db_path}} ${{root_taxon}} ${{min_fam_size}} ${{logic}} ${{min_completeness}} ${{include_younger_fams}} ${{reduced_alphabet}} ${{k}} ${{hidden_taxa}} ${{query_sp}} ${{val_mode}} ${{neg_root_taxon}} ${{focal_taxon}} ${{fam_bin_num}} ${{hog_bin_num}} ${{comp_t}} ${{size_t}} ${{nwk_fn}} ${{overwrite}}
sstat -j ${{SLURM_JOBID}}.batch --format=MaxRSS
sacct -j ${{SLURM_JOBID}}.batch --format=elapsed""".format(mem, hour_nr, name, oe_path, oe_path))

if __name__ == "__main__":

	step = sys.argv[2]
	if step == 'parse_validate_diamond':
		db_path = sys.argv[3]
		root_taxon = sys.argv[4]
		min_fam_size = int(sys.argv[5])
		logic = sys.argv[6]
		min_fam_completeness = float(sys.argv[7])
		include_younger_fams =  True if (sys.argv[8] == 'True') else False
		reduced_alphabet = True if (sys.argv[9] == 'True') else False
		k = int(sys.argv[10])
		hidden_taxa = [' '.join(x.split('_')) for x in sys.argv[11].split(',')]
		query_sp = ' '.join(sys.argv[12].split('_'))
		val_mode = sys.argv[13]
		neg_root_taxon = sys.argv[14]
		focal_taxon = sys.argv[15]
		fam_bin_num = int(sys.argv[16])
		hog_bin_num = int(sys.argv[17])
		comp_t = float(sys.argv[18])
		size_t = int(sys.argv[19])
		diamond_thresholds = np.logspace(-322, 6, base=10, num=165)
		nwk_fn = sys.argv[20]
		overwrite = True if (sys.argv[21] == 'True') else False

		parse_validate_diamond(
			db_path, root_taxon, min_fam_size, logic, min_fam_completeness, include_younger_fams, reduced_alphabet, k, 
			hidden_taxa, query_sp, val_mode, neg_root_taxon, focal_taxon, fam_bin_num, hog_bin_num, comp_t, size_t, 
			diamond_thresholds, nwk_fn, overwrite)

