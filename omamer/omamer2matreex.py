import ete3
import collections
import numpy as np
from .hierarchy import (
    traverse, 
    is_ancestor,
    get_hog_child_prots
)

'''
Functions to generate inputs for Matreex visualisation from an OMAmer database.
'''

def format_st(nwk_fn):
    '''
    Add speciation events and taxa to the species tree.
    Used as templates to build the HOG gene trees inbetween duplication events.
    '''
    st = ete3.Tree(nwk_fn, format=1, quoted_node_names=True)
    for node in st.traverse():

        # keep track of taxon and its level (distance from root)
        node.add_features(taxon = node.name, root_dist = node.get_distance(st))

        if node.is_leaf():
            continue

        node.add_features(S=node.name, Ev='0>1')

    return st

## Functions to export and format a gene tree from the OMAmer database (overly complicated but works...)
## (maybe straight from OMA database would be simpler)

def _convert_hog2gt(hog_off, hog2taxon2subtrees_leaves, hog_tab2, tax_tab, st, tax_id2tax_off, cprot_buff, sp_tab, prot_tab, keep_losses):

    # accumulators
    hog2taxon2subtrees = hog2taxon2subtrees_leaves['hog2taxon2subtrees']
    leaves = hog2taxon2subtrees_leaves['leaves']

    hog_ent = hog_tab2[hog_off]

    # create the HOG subtree as a copy the taxonomy (species tree) from the taxon where the HOG is defined
    taxon = tax_tab[hog_ent['TaxOff']]['ID'].decode('ascii')
    #print(taxon)
    hog_st = (st&taxon).copy()

    # add the HOG id at each node
    hog_id = hog_ent['OmaID'].decode('ascii')
    for node in hog_st.traverse():
        node.add_features(hog_name = hog_id)

    ### add the previously computed HOG subtrees to the current one
    if hog_off in hog2taxon2subtrees:

        ## start by grouping taxon on the same path (can happen when >1 subsequent dupl AND some losses)
        taxon2subtrees = hog2taxon2subtrees[hog_off]
        taxon_ids = list(taxon2subtrees.keys())
        tax_levels = tax_tab['Level'][[tax_id2tax_off[x] for x in taxon_ids]]

        postdupl_tax2taxa = collections.defaultdict(list)
        connected_taxa = set()

        # top-down traversal
        for postdupl_tax, l1 in sorted(zip(taxon_ids, tax_levels), key=lambda x: x[1]):

            # skip if already connected to a postdupl_tax
            if postdupl_tax in connected_taxa:
                continue

            # bottom-up
            for tax, l2 in sorted(zip(taxon_ids, tax_levels), key=lambda x: x[1], reverse=True):

                if is_ancestor(tax_id2tax_off[postdupl_tax], tax_id2tax_off[tax], tax_tab['ParentOff']):
                    postdupl_tax2taxa[postdupl_tax].append(tax)
                    connected_taxa.add(tax)

        ## add duplications and graph corresponding HOG subtrees
        for pdtax, taxa in postdupl_tax2taxa.items():

            # add the duplication node
            parent = (hog_st&pdtax).up
            dupl = parent.add_child(name='dupl')
            dupl.dist = 0.5
            dupl.add_features(Ev='1>0', hog_name=hog_id, taxon=pdtax, root_dist = parent.root_dist + 0.5)

            # add subtrees to the duplication node
            for tax in taxa:
                for t in taxon2subtrees[tax]:
                    t.dist = 0.5
                    dupl.add_child(t)

            # remove the original taxon subtree
            (hog_st&pdtax).detach()

    ### traverse the HOG subtree, relabel extant genes and 
    ### prune species without genes OR mark only loss events
    hog_prots = get_hog_child_prots(hog_off, hog_tab2, cprot_buff)
    hog_species = list(map(lambda x:x.decode('ascii'), sp_tab[prot_tab[hog_prots]['SpeOff']]['ID']))
    hog_sp2prot = dict(zip(hog_species, hog_prots))
    leaves.update(list(map(lambda x: str(x), hog_prots)))

    for leaf in hog_st.get_leaves():
        lname = leaf.name 

        # rename
        if lname in hog_species:
            leaf.name = str(hog_sp2prot[lname])

        # prune
        elif lname not in leaves:
            if keep_losses == True:
                leaf.name = 'loss'
            else:
                leaf.delete(preserve_branch_length=True)

    ### keep track of extant genes and return the hog gene tree
    parent_hog_off = hog_ent['ParentOff']
    if parent_hog_off not in hog2taxon2subtrees:
        hog2taxon2subtrees[parent_hog_off] = collections.defaultdict(set)
    hog2taxon2subtrees[parent_hog_off][taxon].add(hog_st)

    return hog2taxon2subtrees_leaves

def convert_hog2gt(
    self, hog_off, hog_tab, chog_buff, tax_tab, st, tax_id2tax_off, cprot_buff, sp_tab, prot_tab, keep_losses=False):
    '''
    Converts a HOG from OMAmer to a ete3 Tree object.
    '''    
    hog2taxon2subtrees_leaves = {
        'hog2taxon2subtrees':{},
        'leaves':set()
    }

    hog2taxon2subtrees_leaves = traverse(
        hog_off, hog_tab, chog_buff, hog2taxon2subtrees_leaves, _convert_hog2gt, None, _convert_hog2gt,
        hog_tab2=hog_tab, tax_tab=tax_tab, st=st, tax_id2tax_off=tax_id2tax_off, 
        cprot_buff=cprot_buff, sp_tab=sp_tab, prot_tab=prot_tab, keep_losses=keep_losses)

    hog_root_taxon = tax_tab[hog_tab[hog_off]['TaxOff']]['ID'].decode('ascii')
    hog_tree = list(hog2taxon2subtrees_leaves['hog2taxon2subtrees'][-1][hog_root_taxon])[0]

    # special case happening when only a duplication descents from the root node. this happens because of the parsing of the OMA HOGs at the Metazoa level.
    # in such case we keep an unifurcation.
    if len(hog_tree.children) == 1 and hog_tree.children[0].name == 'dupl':
        hog_tree.children[0].prune(hog_tree.get_leaves())
    else:
        # remove single child speciations
        hog_tree.prune(hog_tree.get_leaves(), preserve_branch_length=True)

    return hog_tree

def format_gene_tree(gt):
    '''
    Convert the gene tree exported from OMAmer to a Matreex-formated ete3.Tree.
    (same output as matreex.ham2gt)
    '''
    # format the gene tree
    for node in gt.traverse('postorder'):
        
        if node.is_leaf():
            node.event = 'loss' if node.name == 'loss' else ''
            node.add_features(
                gene = node.name,
                copy_nr = '0' if node.event == 'loss' else '1')
        
        else:
            node.event = 'duplication' if node.Ev == '1>0' else 'speciation'
        
        # common features
        node.add_features(
            HOG=node.hog_name,
            description=node.taxon,
            color=''
        )
        # human friendly tree name
        node.name = '{}_{}'.format(node.HOG, node.taxon)
        
        # delete useless features
        node.del_feature('hog_name')
        node.del_feature('Ev')
        node.del_feature('S')
        node.del_feature('root_dist')
    
    return gt

def propagate_losses(gt):
    '''
    Propagate loss events from leaves to internal nodes.
    '''
    for node in gt.traverse('postorder'):
        if node.is_leaf():
            continue
        cevents = set([c.event for c in node.children])
    
        if len(cevents) == 1 and list(cevents)[0] == 'loss':
            node.event = 'loss'
            
    return gt

def _diagonalize_matrix(gt, st):
    '''
    Diagonalize the matrix by sorting gene tree nodes as in species tree.
    '''
    for node in gt.traverse():
        if node.is_leaf():
            continue

        elif node.event == 'duplication':
            node.children = sorted(node.children, key=lambda x: x.HOG)

        else:
            gt_order = np.array([c.taxon for c in node.children])
            st_order = np.array([c.name for c in (st & node.taxon).children])
            node.children = [node.children[i] for i in [np.argwhere(gt_order == t)[0][0] for t in st_order]]