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
import itertools
import tables


# generic functions for hierarchy
@numba.njit
def get_root_leaf_hog_offsets(off, parent_arr):
    """
    leverages parent pointers to gather parents until root
    """
    leaf_root = [off]
    parent = parent_arr[off]
    while parent != -1:
        leaf_root.append(parent)
        parent = parent_arr[parent]
    return np.array(leaf_root[::-1], dtype=np.uint64)


@numba.njit
def get_lca_hog_off(offsets, parent_arr):
    """
    compute the last common ancestor (lca) within a list of hog offsets from the same family
    """
    off_nr = len(offsets)

    # lca of one hog is itself
    if off_nr == 1:
        return np.uint64(offsets[0])

    else:
        # gather every root-to-leaf paths and keep track of the shortest one
        root_leaf_paths = []
        min_path = np.iinfo(np.int64).max
        for x in offsets:
            root_leaf = get_root_leaf_hog_offsets(x, parent_arr)
            root_leaf_paths.append(root_leaf)
            if len(root_leaf) < min_path:
                min_path = len(root_leaf)

        # if one hog is root, lca is root
        if min_path == 1:
            return np.uint64(root_leaf_paths[0][0])

        else:
            # hogs from root to leaves and stop at min_path
            mat = np.zeros((off_nr, min_path), dtype=np.int64)
            for i in range(off_nr):
                mat[i] = root_leaf_paths[i][:min_path]
            matT = mat.T

            # the lca hog is the one before hogs start diverging
            i = 0
            while i < min_path and np.unique(matT[i]).size == 1:
                i += 1
            return np.uint64(matT[i - 1][0])


def is_ancestor(hog1, hog2, hog2parent):
    """
    is hog1 ancestral to hog2
    """
    return hog1 in get_root_leaf_hog_offsets(hog2, hog2parent)


# Taxonomy


def _children_tax(tax_off, tax_tab, ctax_buff):
    """
	collect direct children of a taxon
	"""
    tax_ent = tax_tab[tax_off]
    ctax_off = tax_ent["ChildrenOff"]
    return ctax_buff[ctax_off : ctax_off + tax_ent["ChildrenNum"]]


def get_lca_tax(tax_off, tax2parent, hidden_taxa):
    """
    get taxon from which tax_off has diverged
    """
    root_leaf = get_root_leaf_hog_offsets(tax_off, tax2parent)
    for x in root_leaf[::-1]:
        if x not in hidden_taxa:
            return x


def get_sister_taxa(tax_off, tax2parent, hidden_taxa, tax_tab, ctax_buff):

    lca_tax = get_lca_tax(tax_off, tax2parent, hidden_taxa)
    children = _children_tax(lca_tax, tax_tab, ctax_buff)

    return np.setdiff1d(children, tax_off)


def leaf_traverse(tax_off, tax_tab, ctax_buff, acc, leaf_fun):
    """
	extend to main traverse by adding postorder, preorder and midorder funs
	"""

    for ctax in _children_tax(tax_off, tax_tab, ctax_buff):
        # reach species
        spe_off = tax_tab[ctax]["SpeOff"]
        if spe_off != -1:
            leaf_fun(spe_off, acc)
        else:
            leaf_traverse(ctax, tax_tab, ctax_buff, acc, leaf_fun)
    return acc


def get_descendant_species(tax_off, tax_tab, ctax_buff):
    def append_species(spe_off, list):
        list.append(spe_off)
        return list

    return np.array(
        sorted(leaf_traverse(tax_off, tax_tab, ctax_buff, [], append_species)),
        dtype=np.uint64,
    )

def traverse_taxonomy(tax_off, tax_tab, ctax_buff, acc, leaf_fun, prefix_fun):

    prefix_fun(tax_off, acc)

    for ctax in _children_tax(tax_off, tax_tab, ctax_buff):

        # stop when no children
        if tax_tab[ctax]["ChildrenOff"] == -1:
            leaf_fun(ctax, acc)
        else:
            traverse_taxonomy(ctax, tax_tab, ctax_buff, acc, leaf_fun, prefix_fun)

    return acc

def get_descendant_taxa(tax_off, tax_tab, ctax_buff):
    def append_taxon(tax_off, list):
        list.append(tax_off)
        return list

    descendant_taxa = traverse_taxonomy(tax_off, tax_tab, ctax_buff, [], append_taxon, append_taxon)
    descendant_taxa.remove(tax_off)

    return np.array(np.unique(descendant_taxa), dtype=np.uint64)

# HOGs


def _children_prot(hog_off, hog_tab, cprot_buff):
    """
    simply collect the proteins of a HOG
    """
    hog_ent = hog_tab[hog_off]
    cprot_off = hog_ent["ChildrenProtOff"]
    return cprot_buff[cprot_off : cprot_off + hog_ent["ChildrenProtNum"]]


def _children_hog(hog_off, hog_tab, chog_buff):
    """
	3 functions for children is silly
	"""
    hog_ent = hog_tab[hog_off]
    chog_off = hog_ent["ChildrenHOGoff"]
    return chog_buff[chog_off : chog_off + hog_ent["ChildrenHOGnum"]]


def traverse(hog_off, hog_tab, chog_buff, acc, leaf_fun, prefix_fun):

    prefix_fun(hog_off, acc)

    for chog in _children_hog(hog_off, hog_tab, chog_buff):

        # stop when no children
        if hog_tab[chog]["ChildrenHOGoff"] == -1:
            leaf_fun(chog, acc)
        else:
            traverse(chog, hog_tab, chog_buff, acc, leaf_fun, prefix_fun)

    return acc


def get_descendant_hogs(hog_off, hog_tab, chog_buff):
    def append_hog(hog_off, list):
        list.append(hog_off)
        return list

    descendant_hogs = traverse(hog_off, hog_tab, chog_buff, [], append_hog, append_hog)
    descendant_hogs.remove(hog_off)

    return np.array(np.unique(descendant_hogs), dtype=np.uint64)


def get_descendant_prots(descendant_hogs, hog_tab, cprot_buff):
    descendant_prots = []
    for x in descendant_hogs:
        descendant_prots.extend(list(_children_prot(x, hog_tab, cprot_buff)))
    return np.array(descendant_prots, dtype=np.uint64)


def get_descendant_species_taxoffs(
    hog_off, hog_tab, chog_buff, cprot_buff, prot2speoff, speoff2taxoff
):
    """
	traverse the HOG to collect all species 
	"""
    prot_offs = get_descendant_prots(
        np.append(get_descendant_hogs(hog_off, hog_tab, chog_buff), np.uint64(hog_off)),
        hog_tab,
        cprot_buff,
    )
    return speoff2taxoff[prot2speoff[prot_offs]]


# inparalog coverage new
def get_sispecies_candidates(tax_off, tax_tab, ctax_buff, hidden_taxa):

    # get each possible LCA speciation node
    tax_offs = _get_root_leaf_offsets(tax_off, tax_tab["ParentOff"])[::-1]

    sispecies_cands = []
    for i, to in enumerate(tax_offs[:-1]):

        parent = tax_offs[i + 1]

        # skip if parent taxa is hidden
        if parent in hidden_taxa:
            continue
        else:
            sistaxa = np.setdiff1d(_children_tax(parent, tax_tab, ctax_buff), to)
            sispecies = []
            for t in sistaxa:
                sispecies.append(list(get_descendant_species(t, tax_tab, ctax_buff)))
            sispecies_cands.append(list(itertools.chain(*sispecies)))

    return sispecies_cands


def get_sister_hogs(hog_off, hog_tab, chog_buff):
    parent = hog_tab["ParentOff"][hog_off]
    return np.setdiff1d(_children_hog(parent, hog_tab, chog_buff), hog_off)


def find_sisspecies(ortholog_species, inparalog_species, sispecies_cands):
    sispecies = []
    for ss_cand in sispecies_cands:
        if (
            len(
                set(list(ortholog_species) + list(inparalog_species)).intersection(
                    ss_cand
                )
            )
            > 0
        ):
            sispecies = ss_cand
            break
    return sispecies


def filter_proteins(proteins, protein_species, sisspecies):
    mask = np.array([x in sisspecies for x in protein_species])
    if mask.size > 0:
        return proteins[mask]
    else:
        return proteins


def calculate_inparalog_coverage(orthologs_f, inparalogs_f):
    n = inparalogs_f.size
    d = n + orthologs_f.size
    return n / d if n != 0 else 0


def compute_inparalog_coverage_new(
    qoff,
    query_ids,
    prot_tab,
    cprot_buff,
    hog_tab,
    chog_buff,
    hidden_taxa,
    sispecies_cands,
    verbose=0,
):

    poff = query_ids[qoff]
    hog_off = prot_tab[poff]["HOGoff"]

    leaf_root_hogs = _get_root_leaf_offsets(hog_off, hog_tab["ParentOff"])[::-1]

    for hog_off in leaf_root_hogs:
        hog_tax = hog_tab["TaxOff"][hog_off]
        hog_lcatax = hog_tab["LCAtaxOff"][hog_off]

        # duplication inside hidden taxa; move to parent
        if hog_tax in hidden_taxa:
            continue

        # duplication outside hidden taxa but not knowloedgeable because of all species members are hidden
        elif hog_lcatax in hidden_taxa:

            # sister HOGs at same taxon and with LCA taxon not hidden
            sis_hogs = get_sister_hogs(hog_off, hog_tab, chog_buff)
            sis_hogs = [
                h
                for h in sis_hogs
                if hog_tab["TaxOff"][h] == hog_tax
                and hog_tab["LCAtaxOff"][h] not in hidden_taxa
            ]

            # LCA node is a duplication (or several); consider all orthologs and inparalogs to calculate IC
            if sis_hogs:
                if verbose == 1:
                    print(hog_tab["ID"][sis_hogs])

                # collect proteins from sister HOGs sub-HOGs
                orthologs = get_descendant_prots(sis_hogs, hog_tab, cprot_buff)
                sis_inhogs = list(
                    itertools.chain(
                        *[
                            list(get_descendant_hogs(sh, hog_tab, chog_buff))
                            for sh in sis_hogs
                        ]
                    )
                )
                inparalogs = get_descendant_prots(sis_inhogs, hog_tab, cprot_buff)

                return calculate_inparalog_coverage(orthologs, inparalogs)

            # sister HOGs are not knowloedgeable
            else:
                continue

        # HOG is knowloedgeable
        else:
            if verbose == 1:
                print(hog_tab["ID"][hog_off])

            # collect orthologs and paralogs
            orthologs = _children_prot(hog_off, hog_tab, cprot_buff)
            inhogs = get_descendant_hogs(hog_off, hog_tab, chog_buff)
            inparalogs = get_descendant_prots(inhogs, hog_tab, cprot_buff)

            # get the species of these orthologs and paralogs
            ortholog_species = prot_tab["SpeOff"][orthologs]
            inparalog_species = prot_tab["SpeOff"][inparalogs]

            # find sister species
            sisspecies = find_sisspecies(
                ortholog_species, inparalog_species, sispecies_cands
            )

            # then filter out protein not descending from sister taxa
            orthologs_f = filter_proteins(orthologs, ortholog_species, sisspecies)
            inparalogs_f = filter_proteins(inparalogs, inparalog_species, sisspecies)

            return calculate_inparalog_coverage(orthologs_f, inparalogs_f)
