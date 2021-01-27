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
import os
import sys
omamer_path = sys.argv[1]
sys.path.insert(0, omamer_path)

import tables
import random 
import numpy as np
from tqdm import tqdm
from Bio import SeqIO, SearchIO
from scipy.stats import rankdata
from pyoma.browser.db import Database as OmaDatabase
from pyoma.browser.db import OmaIdMapper

from omamer.database import DatabaseFromOMA
from omamer.index import SequenceBufferFasta, QuerySequenceBuffer
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


class ClosestEntries(object):
    def __init__(self, db_fn, as_enum=False):
        self.db = OmaDatabase(db_fn)
        self.as_enum = as_enum
    
    def ensure_enum(self, entry):
        if isinstance(entry, int):
            return entry
        else:
            return self.db.id_mapper['Oma'].omaid_to_entry_nr(entry)
    
    def by_dist(self, entry, rank=1):
        x = self.db.get_vpairs(self.ensure_enum(entry))
        r = rankdata(x['Distance'], method='dense')

        cand = set(x[r == rank]['EntryNr2']) 
        return cand, (x[r == rank]['Distance'][0] if cand else None)
    
    def by_score(self, entry, rank=1):
        x = self.db.get_vpairs(self.ensure_enum(entry))
        r = rankdata(-x['Score'], method='dense')

        cand = set(x[r == rank]['EntryNr2']) 
        return cand, (x[r == rank]['Score'][0] if cand else None)


def get_closest_sequence(prot_off, prot_tab, omap, closest, sp_filter, omaid2prot_off):
    '''
    find the closest sequence that is not hidden nor not-referenced by iterating over ranks
    '''
    def _filter_cand(cand, sp_filter, prot_tab, omap, omaid2prot_off):
        '''
        filter hidden species and non-referenced proteins
        '''
        f_cand = set()
        non_ref = False
        for x in cand:
            # get OMAmer protein offset
            omaid = omap.map_entry_nr(x)
            prot_off = omaid2prot_off.get(omaid.encode('ascii'), None)

            # filter out non-referenced proteins
            if prot_off:
                # filter out hidden proteins
                if not sp_filter[prot_tab[prot_off]['SpeOff']]:
                    f_cand.add(x)

            # set non_ref flag to True
            else:
                non_ref = True

        return f_cand, non_ref
    
    # get the OMA entry nr 
    oma_entry = int(omap.omaid_to_entry_nr(prot_tab[prot_off]['ID'].decode('ascii')))
    
    cs = None
    rank = 1
    non_ref_2 = False # (to count the number of missing closest sequences)
    cand, score = closest.by_score(oma_entry, rank=rank)
    
    while cand:
        # filter candidate by their presence in OMAmer reference HOGs
        f_cand, non_ref = _filter_cand(cand, sp_filter, prot_tab, omap, omaid2prot_off)
        if f_cand:
            # random tie breaking
            omaid = omap.map_entry_nr(random.choice(list(f_cand)))
            cs = omaid2prot_off[omaid.encode('ascii')]
            break
        # absence of closest sequence in reference sequences
        elif non_ref:
            non_ref_2 = True
            break
        else:
            rank += 1
            cand, score = closest.by_score(oma_entry, rank=rank)
    if not cs: non_ref_2 = True
    return cs, score, non_ref_2

        
class SWsearch():
    '''
    To get SW closest sequences precomputed in OMA.
    '''
    def __init__(self, db, query_sp):
                
        # load ki and db
        self.db = db
        self.ki = db.ki
        self.query_sp = query_sp
    
    def search(self, oma_db_fn):
        
        prob = False # bitscore (found to get better results than PAM)

        prot_tab = self.db._prot_tab[:]
        sp_filter = self.ki.sp_filter

        # load queries
        sb = QuerySequenceBuffer(db=self.db, query_sp=self.query_sp)
        ids = sb.ids.flatten()
        oma_entries = prot_tab['ID'][ids]

        # load OMA database and initiate associated objects
        omadb = OmaDatabase(oma_db_fn)
        omap = OmaIdMapper(omadb)
        closest = ClosestEntries(oma_db_fn)
        
        # search closest sequences
        omaid2prot_off = dict(zip(prot_tab['ID'], range(len(prot_tab))))
        
        res_rows = []

        non_ref_count = 0
        i = 0
        for prot_off in tqdm(ids):

            cs, score, non_ref_flag = get_closest_sequence(prot_off, prot_tab, omap, closest, sp_filter, omaid2prot_off)

            if cs:
                res_rows.append((ids[i], prot_tab['FamOff'][cs], prot_tab['HOGoff'][cs], score))
            else:
                res_rows.append((ids[i], -1, -1, 1000000 if prob else 0))

            if non_ref_flag:
                non_ref_count += 1

            i+=1

        print('{} queries with closest sequence not in reference proteins'.format(non_ref_count))

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


class SWvalidation(Validation):
    '''
    Same than DIAMONDvalidation without a negative query set
    '''
    def __init__(
        self, db, filename, thresholds, oma_db_fn=None, nwk_fn=None, neg_query_file=None, nthreads=1, query_sp=None, 
        max_query_nr=None, val_mode='golike', neg_root_taxon='random', focal_taxon=None, fam_bin_num=1, hog_bin_num=1, comp_t=0, size_t=0):
        super().__init__(
            db, filename, thresholds, oma_db_fn, nwk_fn, neg_query_file, nthreads, query_sp, max_query_nr, val_mode, 
            neg_root_taxon, focal_taxon, fam_bin_num, hog_bin_num, comp_t, size_t)
    
    def validate(self, se_pos, pvalue_score, hog2bin=True):
        '''
        validate both family and subfamily levels
        '''
        self.validate_family(se_pos, pvalue_score)
        self.validate_subfamily(se_pos, hog2bin, pvalue_score)

    def validate_family(self, se_pos, pvalue_score):

        # validate positives
        tp_query2tresh, fn_query2tresh, fp_pos_query2tresh = self._validate_positive(
            se_pos._query_ids, self._thresholds[:], se_pos._queryFam_ranked, se_pos._queryFam_scores, self.db._prot_tab[:], pvalue_score, self.fam_filter_lca)

        # store results
        self._fam_tp.append(tp_query2tresh)
        self._fam_fn.append(fn_query2tresh)
        self._fam_fp_pos.append(fp_pos_query2tresh)
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

def search_validate_sw(
    db_path, root_taxon, min_fam_size, logic, min_fam_completeness, include_younger_fams, reduced_alphabet, k, 
    hidden_taxa, query_sp, val_mode, neg_root_taxon, focal_taxon, fam_bin_num, hog_bin_num, comp_t, size_t, 
    sw_thresholds, oma_db_fn, nwk_fn, overwrite):
    alphabet_n = 21 if not reduced_alphabet else 13

    # check if already computed
    se_va_fn = '{}{}_MinFamSize{}_{}_MinFamComp0{}{}_A{}_k{}{}_query_{}_SW_{}_{}_{}_{}fbn_{}hbn_MinFamComp0{}_MinFamSize{}.h5'.format(
        db_path, root_taxon, min_fam_size, logic, str(min_fam_completeness).split('.')[-1], 
        '' if include_younger_fams else '_woyf', alphabet_n, k, '_wo_{}'.format('_'.join(['_'.join(x.split()) for x in hidden_taxa])) if hidden_taxa else '',
        '_'.join(query_sp.split()), val_mode, neg_root_taxon, focal_taxon, fam_bin_num, hog_bin_num, str(comp_t).split('.')[-1], size_t)

    if not is_complete(se_va_fn, db_path) or overwrite:
        if os.path.exists(se_va_fn):
            os.remove(se_va_fn)

        # reload k-mer table
        db = load_db_ki(
            db_path, root_taxon, min_fam_size, logic, min_fam_completeness, include_younger_fams, reduced_alphabet, k, hidden_taxa)

        # search positives
        se = SWsearch(db, query_sp)
        se.search(oma_db_fn)

        # validate 
        va = SWvalidation(
            db, se_va_fn, sw_thresholds, oma_db_fn=None, nwk_fn=nwk_fn, 
            neg_query_file=None, nthreads=1, query_sp=query_sp, max_query_nr=None, val_mode=val_mode, 
            neg_root_taxon=neg_root_taxon, focal_taxon=focal_taxon, fam_bin_num=fam_bin_num, hog_bin_num=fam_bin_num, 
            comp_t=comp_t, size_t=size_t)

        va.validate(se, pvalue_score=False, hog2bin=True)
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

    elif step == 'search_validate_sw':
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
oma_db_fn=${{19}}
nwk_fn=${{20}}
overwrite=${{21}}

source /scratch/axiom/FAC/FBM/DBC/cdessim2/default/vrossie4/miniconda3/bin/activate omamer

python ${{omamer_path}}omamer/benchmark.py ${{omamer_path}} search_validate_sw ${{db_path}} ${{root_taxon}} ${{min_fam_size}} ${{logic}} ${{min_completeness}} ${{include_younger_fams}} ${{reduced_alphabet}} ${{k}} ${{hidden_taxa}} ${{query_sp}} ${{val_mode}} ${{neg_root_taxon}} ${{focal_taxon}} ${{fam_bin_num}} ${{hog_bin_num}} ${{comp_t}} ${{size_t}} ${{oma_db_fn}} ${{nwk_fn}} ${{overwrite}}
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

	elif step == 'search_validate_sw':
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
		sw_thresholds = np.arange(1, 5001, 25)
		oma_db_fn = sys.argv[20]
		nwk_fn = sys.argv[21]
		overwrite = True if (sys.argv[22] == 'True') else False

		search_validate_sw(
			db_path, root_taxon, min_fam_size, logic, min_fam_completeness, include_younger_fams, reduced_alphabet, k, 
			hidden_taxa, query_sp, val_mode, neg_root_taxon, focal_taxon, fam_bin_num, hog_bin_num, comp_t, size_t, 
			sw_thresholds, oma_db_fn, nwk_fn, overwrite)