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
import numba
import numpy as np
from tqdm import tqdm

## Generic functions for tables and arrays
def get_seq(prot_off, prot_tab, seq_buff):
    seq_off = prot_tab['SeqOff'][prot_off]
    return seq_buff[seq_off: seq_off + prot_tab['SeqLen'][prot_off]].tobytes().decode("ascii")[:-1]

def row_id2off(row_id, tab):
    if isinstance(row_id, str):
        row_id = row_id.encode('ascii')
    x = np.argwhere(tab['ID'] == row_id)
    if x.size > 0:
        return x[0][0]

def row_off2id(row_off, tab):
    return tab['ID'][row_off].decode('ascii')

## Generic functions for hierarchical tables (hog_tab and tax_tab)
@numba.njit
def get_root_leaf_offsets(off, parent_arr):
    '''
    Leverages parent pointers to gather parents until root.
    '''
    leaf_root = [off]
    parent = parent_arr[off]
    while parent != -1:
        leaf_root.append(parent)
        parent = parent_arr[parent]
    return np.array(leaf_root[::-1], dtype=np.uint64)

@numba.njit
def get_lca_off(offsets, parent_arr):
    '''
    Compute the last common ancestor (LCA) of an offset list.
     - for HOGs, works if they all come from the same root-HOG
    '''
    off_nr = len(offsets)

    # lca of one off is itself
    if off_nr == 1:
        return np.uint64(offsets[0])

    else:
        # gather every root-to-leaf paths and keep track of the shortest one
        root_leaf_paths = []
        min_path = np.iinfo(np.int64).max
        for x in offsets:
            root_leaf = get_root_leaf_offsets(x, parent_arr)
            root_leaf_paths.append(root_leaf)
            if len(root_leaf) < min_path:
                min_path = len(root_leaf)

        # if one off is root, lca is root
        if min_path == 1:
            return np.uint64(root_leaf_paths[0][0])

        else:
            # off from root to leaves and stop at min_path
            mat = np.zeros((off_nr, min_path), dtype=np.int64)
            for i in range(off_nr):
                mat[i] = root_leaf_paths[i][:min_path]
            matT = mat.T

            # the lca off is the one before they start diverging
            i = 0
            while i < min_path and np.unique(matT[i]).size == 1:
                i += 1
            return np.uint64(matT[i - 1][0])

@numba.njit
def get_children(off, tab, c_buff):
    ent = tab[off]
    c_off = ent["ChildrenOff"]
    return c_buff[c_off : c_off + ent["ChildrenNum"]]

def traverse(off, tab, c_buff, acc, leaf_fun, prefix_fun, postfix_fun, **kwargs):
    if prefix_fun:
        prefix_fun(off, acc, **kwargs)

    for c in get_children(off, tab, c_buff):

        # come back when no more children (could add a stop_fun)
        if tab[c]["ChildrenOff"] == -1:
            if leaf_fun:
                leaf_fun(c, acc, **kwargs)
        else:
            traverse(c, tab, c_buff, acc, leaf_fun, prefix_fun, postfix_fun, **kwargs)
            
    if postfix_fun:
        postfix_fun(off, acc, **kwargs)
    
    return acc

def get_descendants(off, tab, c_buff):
    def append(off, list):
        list.append(off)
        return list

    descendants = traverse(off, tab, c_buff, [], append, append, None)
    descendants.remove(off)

    return np.array(descendants, dtype=np.uint64)

def get_leaves(off, tab, c_buff):
    def append(off, list):
        list.append(off)
        return list
    
    if tab['ChildrenOff'][off] == -1:
        return np.array([off], dtype=np.uint64)
    else:
        return np.array(traverse(off, tab, c_buff, [], append, None, None), dtype=np.uint64)

def is_ancestor(off1, off2, parent_arr):
    '''
    Is off1 ancestral to off2?
    '''
    return off1 in get_root_leaf_offsets(off2, parent_arr)

## HOG specific functions
def get_hog_child_prots(hog_off, hog_tab, cprot_buff):
    hog_ent = hog_tab[hog_off]
    cprot_off = hog_ent["ChildrenProtOff"]
    return cprot_buff[cprot_off : cprot_off + hog_ent["ChildrenProtNum"]]
    
def get_hog_member_prots(hog_off, hog_tab, chog_buff, cprot_buff):
    '''
    Collect proteins from HOG and its descendants.
    '''
    descendant_hogs = get_descendants(hog_off, hog_tab, chog_buff)
    descendant_prots = list(get_hog_child_prots(hog_off, hog_tab, cprot_buff))
    for dhog_off in descendant_hogs:
        descendant_prots.extend(list(get_hog_child_prots(dhog_off, hog_tab, cprot_buff)))
    return descendant_prots

@numba.njit
def get_hog_taxon_levels(hog_off, hog_tab, hog_taxa_buff):
    hog_ent = hog_tab[hog_off]
    hog_taxa_off = hog_ent['HOGtaxaOff']
    return hog_taxa_buff[hog_taxa_off : hog_taxa_off + hog_ent["HOGtaxaNum"]]

def get_hog_child_species(hog_off, hog_tab, hog_taxa_buff, tax_tab):
    '''
    Compute species from extant taxa (without children).
    '''
    sp_offsets = tax_tab['SpeOff'][get_hog_taxon_levels(hog_off, hog_tab, hog_taxa_buff)]
    return sp_offsets[np.argwhere(sp_offsets != -1).flatten()]

def get_hog_member_species(hog_off, hog_tab, chog_buff, hog_taxa_buff, tax_tab):
    '''
    Collect species from HOG and its descendants.
    '''
    descendant_hogs = get_descendants(hog_off, hog_tab, chog_buff)
    member_species = list(get_hog_child_species(hog_off, hog_tab, hog_taxa_buff, tax_tab))
    for dhog_off in descendant_hogs:
        member_species.extend(list(get_hog_child_species(dhog_off, hog_tab, hog_taxa_buff, tax_tab)))
    return member_species   

@numba.njit
def is_taxon_implied(true_tax_lineage, hog_off, hog_tab, chog_buff):
    '''
    Check whether the HOG implies the true taxon.
    '''
    implied = False
    
    # the HOG defined on the lineage of the true taxon
    if np.argwhere(true_tax_lineage == hog_tab['TaxOff'][hog_off]).size == 1:
        implied = True
    
    # child HOGs are defined on the lineage of the true taxon
    child_hog_taxa = np.unique(hog_tab['TaxOff'][get_children(hog_off, hog_tab, chog_buff)])
    for tax_off in true_tax_lineage:
        if np.argwhere(child_hog_taxa == tax_off).size == 1:
            implied = False
            break

    return implied

def get_hogs_at_level(ref_taxoff, tax_tab, fam_tab, ref_desc_taxoffs, hog_tab, chog_buff, include_younger_fams=False):
    '''
    Given a selected reference taxon, finds all HOGs implying that taxon.
    Option to include root-HOGs defined in descendants of the reference taxon too.
    TO DO: more filtering options
    '''
    cand_hogs = []

    # trace the taxon ancestors
    true_tax_lineage = get_root_leaf_offsets(ref_taxoff, tax_tab['ParentOff'])[::-1]

    # traverse reference families and skip the ones that lies outside the taxonomic scope of the reference taxon
    for fam_ent in tqdm(fam_tab):
        fam_taxoff = fam_ent['TaxOff']

        # include families defined in descendants of the reference taxon too.
        if fam_taxoff in ref_desc_taxoffs:
            if include_younger_fams:
                cand_hogs.append(fam_ent['HOGoff'])
            continue

        if fam_taxoff not in true_tax_lineage:
            continue

        # traverse the family and find the HOGs implying the reference taxon
        rh_off = fam_ent['HOGoff']
        for hog_off in np.arange(rh_off, rh_off + fam_ent['HOGnum'], dtype=int):
            if is_taxon_implied(true_tax_lineage, hog_off, hog_tab, chog_buff):
                cand_hogs.append(hog_off)

    return np.array(cand_hogs, dtype=np.int64)

# functions to precompute HOG taxonomic levels (implied or not) 
def get_hog_taxa(hog_off, sp_tab, prot_tab, hog_tab, cprot_buff, tax_tab, chog_buff):
    '''
    Compute all HOG taxonomic level induced by child HOGs or member proteins.
    '''
    taxa = set()
    
    # add taxa induced by member proteins
    cprot_taxa = np.unique(sp_tab[prot_tab[get_hog_child_prots(hog_off, hog_tab, cprot_buff)]['SpeOff']]['TaxOff'])
    for tax_off in cprot_taxa:
        taxa.update(get_root_leaf_offsets(tax_off, tax_tab['ParentOff']))
    
    # add taxa induced by child HOGs (thus exluding their own taxon)
    chog_taxa = np.unique(hog_tab[get_children(hog_off, hog_tab, chog_buff)]['TaxOff'])
    for tax_off in chog_taxa:
        taxa.update(get_root_leaf_offsets(tax_off, tax_tab['ParentOff'])[:-1])
    
    # remove taxa older than the HOG root-taxon
    hog_tax_off = hog_tab[hog_off]['TaxOff']
    taxa = taxa.difference(get_root_leaf_offsets(hog_tax_off, tax_tab['ParentOff'])[:-1])
    
    return taxa

def get_hog2taxa(hog_tab, sp_tab, prot_tab, cprot_buff, tax_tab, chog_buff):
    '''
    Precompute compact hog2taxa.
    '''
    buff_off = 0
    hog_taxa_idx = [buff_off]
    hog_taxa_buff = []
    for hog_off in tqdm(range(hog_tab.size)):
        taxa = get_hog_taxa(hog_off, sp_tab, prot_tab, hog_tab, cprot_buff, tax_tab, chog_buff)
        buff_off += len(taxa)
        hog_taxa_idx.append(buff_off)
        hog_taxa_buff.extend(taxa)
    return np.array(hog_taxa_idx, dtype=np.int64), np.array(hog_taxa_buff, dtype=np.int16)

def get_hog_implied_taxa(hog_off, hog_tab, tax_tab, ctax_buff, chog_buff):
    '''
    Collect the HOG implied taxa (i.e. any taxon descending from the HOG root-taxon and before child HOGs root-taxa).
    (implied because include taxa having lost their copy)
    '''
    tax_off = hog_tab[hog_off]['TaxOff']
    hog_taxa = set(get_descendants(tax_off, tax_tab, ctax_buff))
    hog_taxa.add(tax_off)
    chogs_taxa = set()
    for chog_off in get_children(hog_off, hog_tab, chog_buff):
        ctax_off = hog_tab[chog_off]['TaxOff']
        chogs_taxa.add(ctax_off)
        chogs_taxa.update(get_descendants(ctax_off, tax_tab, ctax_buff))
    return hog_taxa.difference(chogs_taxa)

def get_hog2implied_taxa(hog_tab, tax_tab, ctax_buff, chog_buff):
    '''
    Precompute compact hog2implied_taxa.
    '''
    buff_off = 0
    hog_taxa_idx = [buff_off]
    hog_taxa_buff = []
    for hog_off in tqdm(range(hog_tab.size)):
        taxa = get_hog_implied_taxa(hog_off, hog_tab, tax_tab, ctax_buff, chog_buff)
        buff_off += len(taxa)
        hog_taxa_idx.append(buff_off)
        hog_taxa_buff.extend(taxa)
    return np.array(hog_taxa_idx, dtype=np.int16), np.array(hog_taxa_buff, dtype=np.int16)

## Taxonomy specific functions
def get_closest_reference_taxon(tax_off, tax2parent, hidden_taxa):
    """
    Get taxon from which a tax_off has diverged (if not hidden, returns tax_off itself)
    """
    root_leaf = get_root_leaf_offsets(tax_off, tax2parent)
    for x in root_leaf[::-1][1:]:
        if x not in hidden_taxa:
            return x

## Prot tab
def get_seq(prot_off, prot_tab, seq_buff):
    seq_off = prot_tab['SeqOff'][prot_off]
    return seq_buff[seq_off: seq_off + prot_tab['SeqLen'][prot_off]].tobytes().decode("ascii")[:-1]

# TO UPDATE
def get_sister_taxa(tax_off, tax2parent, hidden_taxa, tax_tab, ctax_buff):

    lca_tax = get_lca_taxon(tax_off, tax2parent, hidden_taxa)
    if lca_tax != None:
        children = _children_tax(lca_tax, tax_tab, ctax_buff)
        return np.setdiff1d(children, tax_off)
    else:
        return np.array([], dtype=np.int64)