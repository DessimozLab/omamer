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
# validation based on merge_search results
from property_manager import lazy_property, cached_property
from itertools import repeat
from tqdm import tqdm
from ete3 import Tree
import numpy as np
import tables
import os

from .hierarchy import get_root_leaf_hog_offsets, get_lca_tax





class SubfamilyValidation():

	def __init__(self, db, thresholds, nwk_fn, query_sp, focal_taxon=None, bin_num=1, name=None, path=None, val_mode='golike'):

		self.db = db
		self.ki = db.ki

		# hdf5 file
		self.name = name #"{}_{}".format(self.se.name, name if name else 'subfamily_{}'.format(val_mode))
		self.path = path #if path else self.db.path
		self.file = "{}{}.h5".format(self.path, self.name)

		if os.path.isfile(self.file):
			self.mode = 'r'
			self.va = tables.open_file(self.file, self.mode)

		else:
			self.mode = 'w'
			self.va = tables.open_file(self.file, self.mode)

		# mode of validation and thresholds
		self.val_mode = val_mode
		self.va.create_carray('/', 'Thresholds', obj=np.array(thresholds, dtype=np.float64), filters=self.db._compr)
		self.nwk_fn = nwk_fn
		self.query_sp = query_sp
		self.focal_taxon = focal_taxon if focal_taxon else self.db.root_taxon
		self.bin_num = bin_num

	def __enter__(self):
	    return self

	def __exit__(self, *_):
	    self.va.close()
	    
	def clean(self):
	    '''
	    close and remove hdf5 file
	    '''
	    self.__exit__()
	    try:
	        os.remove(self.file)
	    except FileNotFoundError:
	        print("{} already cleaned".format(self.file))

	@lazy_property
	def hog_off2taxbin(self):
		tax_off2taxbin = self.bin_taxa(self.nwk_fn, self.focal_taxon, self.db._tax_tab[:], self.query_sp, self.ki.tax_filter, 
			bin_num=self.bin_num, focal_bin=False, merge_post_lca_taxa=True)
		return np.array([tax_off2taxbin.get(tax, -1) for tax in self.db._hog_tab.col('TaxOff')], np.int64)

	@lazy_property
	def fam_off2taxbin(self):
		'''
		one more been because of focal taxon bin
		'''
		tax_off2taxbin = self.bin_taxa(self.nwk_fn, self.focal_taxon, self.db._tax_tab[:], self.query_sp, self.ki.tax_filter, 
			bin_num=self.bin_num + 1, focal_bin=True, merge_post_lca_taxa=True)
		return np.array([tax_off2taxbin.get(tax, -1) for tax in self.db._fam_tab.col('TaxOff')], np.int64)
	
	@cached_property
	def hog_filter_lca(self):
		return self.ki.tax_filter[self.db._hog_tab.col('LCAtaxOff')]

	### same as in database class; easy access to data ###
	@property
	def _thresholds(self):
	    return self.va.get_node('/Thresholds')

	@property
	def _query_ids(self):
		if '/QueryIDs' not in self.va:
			return self.va.create_earray('/', 'QueryIDs', tables.UInt32Atom(), shape=(0,), filters=self.db._compr)
		else:
			return self.va.get_node('/QueryIDs')

	@property
	def _query_fambins(self):
		if '/QueryFamBins' not in self.va:
			return self.va.create_earray('/', 'QueryFamBins', tables.UInt32Atom(), shape=(0,), filters=self.db._compr)
		else:
			return self.va.get_node('/QueryFamBins')

	def _get_node(self, node):
		if '/{}'.format(node) in self.va:
		    return self.va.get_node('/{}'.format(node))
		else:
			# if root-taxon and focal taxon are not the same, there are two additional taxonomic bins for the ancestral and outgroup taxa
			bn = self.bin_num if self.focal_taxon == self.db.root_taxon else self.bin_num + 2
			return self.va.create_earray('/', node, tables.UInt16Atom(), shape=(0, bn, self._thresholds.nrows), filters=self.db._compr)
	@property      
	def _tp_pre(self):
	    return self._get_node('TP_pre')

	@property
	def _tp_rec(self):
	    return self._get_node('TP_rec')
	    
	@property
	def _fn(self):
	    return self._get_node('FN')

	@property
	def _fp(self):
	    return self._get_node('FP')

	####################################################################################################################################
	def validate(self, se, hog2bin=True, hf_lca=True, prob=False):
		assert (self.mode in {'w', 'a'}), 'Validation must be opened in write mode.'

		tp_pre_query2x2tresh, tp_rec_query2x2tresh, fn_query2x2tresh, fp_query2x2tresh = self._validate(
			self._thresholds[:], se._queryFam_ranked, se._query_ids, se._queryRankHog_bestpath, se._queryRankHog_scores, 
			self.db._prot_tab[:], self.db._fam_tab[:], self.db._hog_tab.col('ParentOff'), get_root_leaf_hog_offsets, prob, 
			self.hog_off2taxbin if hog2bin else np.array([]), self.hog_filter_lca if hf_lca else np.array([]), self.val_mode)

		# store results
		self._tp_pre.append(tp_pre_query2x2tresh)
		self._tp_rec.append(tp_rec_query2x2tresh)
		self._fn.append(fn_query2x2tresh)
		self._fp.append(fp_query2x2tresh)
		self._tp_pre.flush()
		self._tp_rec.flush()
		self._fn.flush()
		self._fp.flush()

		# and keep track of query ids and taxbin of predicted family
		self._query_ids.append(se._query_ids)
		self._query_fambins.append(self.fam_off2taxbin[se._queryFam_ranked.flatten()])
		self._query_ids.flush()
		self._query_fambins.flush()

	@staticmethod
	def _validate(
		thresholds, queryFam_ranked, query_prot_offsets, queryRankHog_bestpath, queryRankHog_scores,
		prot_tab, fam_tab, hog2parent, fun_root_leaf, prob, hog2bin, hog_filter_lca, val_mode):
		'''
		args:
		 - pv: whether p-value type of score (the lower the better)
		 - hog2x: mapper between HOGs and any HOG grouping such as taxonomy or amount of duplications in query sister
		 - x_num: number of group in hog2x mapper
		'''
		def _compute_tp_fp_fn(
			tp_hogs, fn_hogs, fp_hogs, hog2x, x_num, tp_pre_query2x2tresh, tp_rec_query2x2tresh, fn_query2x2tresh, fp_query2x2tresh, val_mode):
			'''
			3 options to define TPs, FNs and FPs
			'''
			# map to bins and keep track of HOG numbers
			tp_x, tp_nr = np.unique(hog2x[tp_hogs], return_counts=True) if x_num else tp_hogs
			fn_x, fn_nr = np.unique(hog2x[fn_hogs], return_counts=True) if x_num else fn_hogs
			fp_x, fp_nr = np.unique(hog2x[fp_hogs], return_counts=True) if x_num else fp_hogs

			# my custom approach: split TPs in to recall and precision TPs
			if val_mode == 'custom':
			    tp_x_pre = np.setdiff1d(tp_x, fp_x)
			    tp_x_rec = np.setdiff1d(tp_x, fn_x)

			    # and counts TPs, FPs and FNs by query
			    tp_pre_query2x2tresh[q, tp_x_pre, t_off] = 1
			    tp_rec_query2x2tresh[q, tp_x_rec, t_off] = 1
			    fn_query2x2tresh[q, fn_x, t_off] = 1
			    fp_query2x2tresh[q, fp_x, t_off] = 1

			# stringent approach ignoring hierarchy
			elif val_mode == 'stringent':   
			    tp_x = np.setdiff1d(tp_x, np.union1d(fn_x, fp_x))

			    #also counts TPs, FPs and FNs by query
			    tp_pre_query2x2tresh[q, tp_x, t_off] = 1
			    tp_rec_query2x2tresh[q, tp_x, t_off] = 1
			    fn_query2x2tresh[q, fn_x, t_off] = 1
			    fp_query2x2tresh[q, fp_x, t_off] = 1

			# approach where TPs, FPs and FNs are counted by HOG
			elif val_mode == 'golike':
			    
			    tp_pre_query2x2tresh[q, tp_x, t_off] = tp_nr
			    tp_rec_query2x2tresh[q, tp_x, t_off] = tp_nr
			    fn_query2x2tresh[q, fn_x, t_off] = fn_nr
			    fp_query2x2tresh[q, fp_x, t_off] = fp_nr

		thresholds = np.array(thresholds, dtype=np.float64)

		bin_num = (np.max(hog2bin) + 1) if hog2bin.size >0 else None
		# store validation results
		tp_pre_query2bin2tresh = np.zeros((query_prot_offsets.size, bin_num if bin_num else hog2parent.size, thresholds.size), dtype=np.uint16)
		tp_rec_query2bin2tresh = np.zeros((query_prot_offsets.size, bin_num if bin_num else hog2parent.size, thresholds.size), dtype=np.uint16)
		fn_query2bin2tresh = np.zeros((query_prot_offsets.size, bin_num if bin_num else hog2parent.size, thresholds.size), dtype=np.uint16)
		fp_query2bin2tresh = np.zeros((query_prot_offsets.size, bin_num if bin_num else hog2parent.size, thresholds.size), dtype=np.uint16)

		# iterage over queries
		for q in tqdm(range(query_prot_offsets.size)):

			# true data
			prot_off = query_prot_offsets[q]
			true_fam = prot_tab[prot_off]['FamOff']
			true_leafhog = prot_tab[prot_off]['HOGoff']
			true_hogs = fun_root_leaf(true_leafhog, hog2parent)[1:]  # ignore root-HOG

			# remove hogs specific to hidden taxa from true hogs
			if hog_filter_lca.size > 0:
			    true_hogs = true_hogs[~hog_filter_lca[true_hogs]]

			# pred fam
			pred_fam = queryFam_ranked[q, 0]
			    
			# hogs of pred fam
			hog_off = fam_tab[pred_fam]['HOGoff']
			hog_num = fam_tab[pred_fam]['HOGnum']
			hog_offsets = np.arange(hog_off, hog_off + hog_num, dtype=np.uint64)[1:] # ignore root-HOG

			# best path hogs and score        
			hogs_mask = queryRankHog_bestpath[0, q, 1:hog_num]  # ignore root-HOG
			hogs_bestpath = hog_offsets[hogs_mask]
			hogs_bestpath_score = queryRankHog_scores[0, q, 1:hog_num][hogs_mask]  # ignore root-HOG

			# iterate over thresholds
			for t_off in range(thresholds.size):
			    t_val = thresholds[t_off]
			    
			    # pred hogs
			    pred_hogs = hogs_bestpath[(hogs_bestpath_score < t_val) if prob else (hogs_bestpath_score >= t_val)]
			    
			    # confront true classes against predicted classes to get benchmark results
			    # not supported by numba ...
			    tp_hogs = np.intersect1d(true_hogs, pred_hogs, assume_unique=True)
			    fn_hogs = np.setdiff1d(true_hogs, pred_hogs, assume_unique=True)
			    fp_hogs = np.setdiff1d(pred_hogs, true_hogs, assume_unique=True)

			    _compute_tp_fp_fn(tp_hogs, fn_hogs, fp_hogs, hog2bin, bin_num, tp_pre_query2bin2tresh, tp_rec_query2bin2tresh,
			        fn_query2bin2tresh, fp_query2bin2tresh, val_mode)

		return tp_pre_query2bin2tresh, tp_rec_query2bin2tresh, fn_query2bin2tresh, fp_query2bin2tresh

	@staticmethod
	def bin_taxa(stree_tree, focal_taxon, tax_tab, query_sp, tax_filter, bin_num=2, focal_bin=True, merge_post_lca_taxa=True):

	    def _bin_taxa(bin_num, tax2root_dist, lca_tax, focal_bin, merge_post_lca_taxa):

	        # remove one bin if not merging taxa younger than lca taxon
	        bin_num = bin_num if merge_post_lca_taxa else bin_num - 1

	        # grab root_dist of lca taxon
	        lca_dist = tax2root_dist[lca_tax] if lca_tax else max(tax2root_dist.values())

	        # make a bin specific to the root taxon
	        if focal_bin:
	            dist_range_size = lca_dist / (bin_num - 1)
	            dist_ranges = [-1] + [dist_range_size*n for n in range(0, bin_num)]
	        else:
	            dist_range_size = lca_dist / bin_num
	            dist_ranges = [-1] + [dist_range_size*n for n in range(1, bin_num + 1)]

	        # fill bins with taxa within distance ranges
	        tax2taxbin = {}
	        for bn in range(bin_num):
	            bin_taxa = {k for k,v in tax2root_dist.items() if v > dist_ranges[bn] and v <= dist_ranges[bn + 1]}
	            tax2taxbin.update(dict(zip(bin_taxa, repeat(bn))))

	        # deal with taxa descending from lca taxon
	        max_dist = max(tax2root_dist.values())
	        post_lca_taxa = {k for k,v in tax2root_dist.items() if v > lca_dist and v <= max_dist}

	        if merge_post_lca_taxa:
	            tax2taxbin.update(dict(zip(post_lca_taxa, repeat(bin_num - 1))))
	        else:
	            tax2taxbin.update(dict(zip(post_lca_taxa, repeat(bin_num))))

	        return tax2taxbin
	    
	    # get focal species tree
	    stree = Tree(stree_tree, format=1, quoted_node_names=True)
	    focal_stree = stree&focal_taxon
	    
	    # get distance from each taxon to the root
	    tax2root_dist = {x.name:x.get_distance(focal_stree) for x in focal_stree.traverse()}
	    
	    # some mappers
	    tax_off2tax = tax_tab['ID'] 
	    tax2tax_off = dict(zip(tax_off2tax, range(tax_off2tax.size)))
	    
	    # get the lca taxon bewteen the query species and the reference (including hidden taxa with children propagation)
	    lca_tax = tax_off2tax[get_lca_tax(
	        tax2tax_off[query_sp.encode('ascii')], tax_tab['ParentOff'], np.argwhere(tax_filter).flatten())].decode('ascii')
	    
	    # bin taxa in the species tree
	    tax2taxbin = _bin_taxa(bin_num, tax2root_dist, lca_tax, focal_bin, merge_post_lca_taxa)
	    
	    # get taxa ancestor of focal taxon
	    older_taxa = tax_off2tax[get_root_leaf_hog_offsets(
	        tax2tax_off[focal_taxon.encode('ascii')], tax_tab['ParentOff'])][:-1]
	    
	    tax_off2taxbin = {}
	    
	    # bin taxa including older and outgroup taxa from focal taxon
	    if older_taxa.size > 0:
	        for tax_off, tax in enumerate(tax_off2tax):

	            # if ancestor of focal taxon (bin 1)
	            if tax in older_taxa:
	                tax_off2taxbin[tax_off] = 1

	            # from binned taxa (bin 2, ..., n) 
	            elif tax.decode('ascii') in tax2taxbin:
	                tax_off2taxbin[tax_off] = tax2taxbin[tax.decode('ascii')] + 2

	            # or outgroup of focal taxon(bin 0)
	            else:
	                tax_off2taxbin[tax_off] = 0
	                
	    # bin taxa without ancestor and outgroup bins
	    else:
	        for tax_off, tax in enumerate(tax_off2tax):
	            tax_off2taxbin[tax_off] = tax2taxbin[tax.decode('ascii')]

	    return tax_off2taxbin

	@staticmethod
	def partition_queries(thresholds, query_values, parameter_name):
		part_names = []
		partitions = []
		curr_thresh = -1
		for thresh in thresholds:
		    part = np.full(query_values.size, False)
		    part[(query_values > curr_thresh) & (query_values <= thresh)] = True
		    partitions.append(part)
		    part_names.append('{} < {} <= {}'.format(curr_thresh if curr_thresh != -1 else 0, parameter_name, thresh))
		    curr_thresh = thresh

		# last one
		part = np.full(query_values.size, False)
		part[query_values > curr_thresh] = True
		partitions.append(part)
		part_names.append('{} < {} <= {}'.format(curr_thresh if curr_thresh != -1 else 0, parameter_name, 1))

		return np.array(partitions), part_names

	@staticmethod
	def compute_precision_recall(tp_pre_query2bin2tresh, tp_rec_query2bin2tresh, fn_query2bin2tresh, fp_query2bin2tresh, partitions=np.array([])):

		if partitions.size == 0:
		    partitions = np.array([np.full(tp_pre_query2bin2tresh.shape[0], True)])

		part_num = partitions.shape[0]
		bin_num = tp_pre_query2bin2tresh.shape[1]
		thresh_num = tp_pre_query2bin2tresh.shape[2]

		part2bin2pre = np.zeros((part_num, bin_num, thresh_num), dtype=np.float64)
		part2bin2rec = np.zeros((part_num, bin_num, thresh_num), dtype=np.float64)
		part2bin2tp_pre_nr = np.zeros((part_num, bin_num, thresh_num), dtype=np.uint64)
		part2bin2tp_rec_nr = np.zeros((part_num, bin_num, thresh_num), dtype=np.uint64)
		part2bin2fn_nr = np.zeros((part_num, bin_num, thresh_num), dtype=np.uint64)
		part2bin2fp_nr = np.zeros((part_num, bin_num, thresh_num), dtype=np.uint64)

		for p in range(part_num):
		    part = partitions[p]
		    for b in range(bin_num):
		        for t in range(thresh_num):
		            tp_pre_nr = np.sum(tp_pre_query2bin2tresh[:, b, t][part])
		            tp_rec_nr = np.sum(tp_rec_query2bin2tresh[:, b, t][part])
		            fn_nr = np.sum(fn_query2bin2tresh[:, b, t][part])
		            fp_nr = np.sum(fp_query2bin2tresh[:, b, t][part])
		            part2bin2pre[p, b, t] = (tp_pre_nr/(tp_pre_nr + fp_nr)) if tp_pre_nr or fp_nr else 0
		            part2bin2rec[p, b, t] = (tp_rec_nr/(tp_rec_nr + fn_nr)) if tp_rec_nr or fn_nr else 0
		            part2bin2tp_pre_nr[p, b, t] = tp_pre_nr
		            part2bin2tp_rec_nr[p, b, t] = tp_rec_nr
		            part2bin2fn_nr[p, b, t] = fn_nr
		            part2bin2fp_nr[p, b, t] = fp_nr

		part2bin2query_nr = np.zeros((part_num, bin_num), dtype=np.uint64)

		for p in range(part_num):
		    for b in range(bin_num):
		        part2bin2query_nr[p, b] = part2bin2tp_rec_nr[p, b, 0] + part2bin2fn_nr[p, b, 0]
		        
		return part2bin2pre, part2bin2rec, part2bin2query_nr

	def F1(self, part2bin2pre, part2bin2rec):
		n = part2bin2pre * part2bin2rec
		d = part2bin2pre + part2bin2rec
		part2bin2f1 = 2 * np.divide(n, d, out=np.zeros_like(n), where=d!=0)
		part2bin2f1_max = np.max(part2bin2f1, axis=2)

		part2bin2f1_tval = np.zeros(part2bin2f1_max.shape)
		part2bin2f1_toff = np.zeros(part2bin2f1_max.shape, dtype=np.uint64)

		for p in range(part2bin2f1.shape[0]):
			for b in range(part2bin2f1.shape[1]):
				toff = np.where(part2bin2f1[p, b]==part2bin2f1_max[p, b])[0][0]
				part2bin2f1_tval[p, b] = self._thresholds[:][toff]
				part2bin2f1_toff[p, b] = toff
		return part2bin2f1_max, part2bin2f1_tval, part2bin2f1_toff

	def PREpro(self, part2bin2pre, part2bin2rec):
		n = part2bin2pre * part2bin2pre * part2bin2rec
		d = part2bin2pre + part2bin2pre + part2bin2rec
		part2bin2f1 = 3 * np.divide(n, d, out=np.zeros_like(n), where=d!=0)
		part2bin2f1_max = np.max(part2bin2f1, axis=2)

		part2bin2f1_tval = np.zeros(part2bin2f1_max.shape)
		part2bin2f1_toff = np.zeros(part2bin2f1_max.shape, dtype=np.uint64)

		for p in range(part2bin2f1.shape[0]):
			for b in range(part2bin2f1.shape[1]):
				toff = np.where(part2bin2f1[p, b]==part2bin2f1_max[p, b])[0][0]
				part2bin2f1_tval[p, b] = self._thresholds[:][toff]
				part2bin2f1_toff[p, b] = toff
		return part2bin2f1_max, part2bin2f1_tval, part2bin2f1_toff
