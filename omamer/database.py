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
from Bio import SeqIO
from ete3 import Tree
from itertools import repeat
from property_manager import lazy_property
from tqdm import tqdm
import collections
import logging
import numpy as np
import os
import tables

from ._utils import LOG, is_progress_disabled

from .hierarchy import (
    get_root_leaf_hog_offsets,
    get_lca_hog_off,
    get_descendant_species_taxoffs,
)
from .index import Index


class Database(object):
    # define the format of the tables
    class ProteinTableFormat(tables.IsDescription):
        ID = tables.StringCol(255, pos=1, dflt=b"")
        SeqOff = tables.UInt64Col(pos=2)
        SeqLen = tables.UInt64Col(pos=3)
        SpeOff = tables.UInt64Col(pos=4)
        HOGoff = tables.UInt64Col(pos=5)
        FamOff = tables.UInt64Col(pos=6)

    class HOGtableFormat(tables.IsDescription):
        ID = tables.StringCol(255, pos=1, dflt=b"")
        FamOff = tables.UInt64Col(pos=2)
        TaxOff = tables.UInt64Col(pos=3)
        ParentOff = tables.Int64Col(pos=4)
        ChildrenHOGoff = tables.Int64Col(pos=5)
        ChildrenHOGnum = tables.Int64Col(pos=6)
        ChildrenProtOff = tables.Int64Col(pos=7)
        ChildrenProtNum = tables.Int64Col(pos=8)
        OmaID = tables.StringCol(255, pos=9, dflt=b"")
        LCAtaxOff = tables.UInt64Col(pos=10)

    class FamilyTableFormat(tables.IsDescription):
        ID = tables.UInt64Col(pos=1)
        TaxOff = tables.UInt64Col(pos=2)
        HOGoff = tables.UInt64Col(pos=3)
        HOGnum = tables.UInt64Col(pos=4)
        LevelOff = tables.UInt64Col(pos=5)
        LevelNum = tables.UInt64Col(pos=6)

    class SpeciesTableFormat(tables.IsDescription):
        ID = tables.StringCol(255, pos=1, dflt=b"")
        ProtOff = tables.UInt64Col(pos=2)
        ProtNum = tables.UInt64Col(pos=3)
        TaxOff = tables.UInt64Col(pos=4)

    class TaxonomyTableFormat(tables.IsDescription):
        ID = tables.StringCol(255, pos=1, dflt=b"")
        ParentOff = tables.Int64Col(pos=2)
        ChildrenOff = tables.Int64Col(pos=3)
        ChildrenNum = tables.Int64Col(pos=4)
        SpeOff = tables.Int64Col(pos=5)
        Level = tables.Int64Col(pos=6)

    def __init__(self, filename, root_taxon=None, mode='r', nthreads=1):
        assert (mode != 'w' or root_taxon), "A root_taxon must be defined when building the database"
        self.root_taxon = root_taxon
        self.filename = filename
        self.mode = mode

        self.nthreads = nthreads
        self._compr = tables.Filters(complevel=6, complib="blosc", fletcher32=True)

        if os.path.isfile(self.filename) and self.mode in {'a', 'w'}:
            LOG.warning('Overwriting database file ({})'.format(self.filename))

        self.db = tables.open_file(self.filename, self.mode, filters=self._compr)

        if '/Index' in self.db:
            self.ki = Index(self, nthreads=self.nthreads)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def clean(self):
        self.__exit__()
        try:
            os.remove(self.filename)
        except FileNotFoundError:
            print("{} already cleaned".format(self.filename))

    ### method attributes to facilitate access to data stored in hdf5 file ###
    @property
    def _prot_tab(self):
        if "/Protein" in self.db:
            return self.db.root.Protein
        else:
            return self.db.create_table(
                "/", "Protein", self.ProteinTableFormat, filters=self._compr
            )

    @property
    def _hog_tab(self):
        if "/HOG" in self.db:
            return self.db.root.HOG
        else:
            return self.db.create_table(
                "/", "HOG", self.HOGtableFormat, filters=self._compr
            )

    @property
    def _fam_tab(self):
        if "/Family" in self.db:
            return self.db.root.Family
        else:
            return self.db.create_table(
                "/", "Family", self.FamilyTableFormat, filters=self._compr
            )

    @property
    def _sp_tab(self):
        if "/Species" in self.db:
            return self.db.root.Species
        else:
            return self.db.create_table(
                "/", "Species", self.SpeciesTableFormat, filters=self._compr
            )

    @property
    def _tax_tab(self):
        if "/Taxonomy" in self.db:
            return self.db.root.Taxonomy
        else:
            return self.db.create_table(
                "/", "Taxonomy", self.TaxonomyTableFormat, filters=self._compr
            )

    # arrays
    @property
    def _seq_buff(self):
        if "/SequenceBuffer" in self.db:
            return self.db.root.SequenceBuffer
        else:
            # initiate
            return self.db.create_earray(
                "/", "SequenceBuffer", tables.StringAtom(1), (0,), filters=self._compr
            )

    # The below arrays are of fixed length and created later as carray
    @property
    def _chog_arr(self):
        if "/ChildrenHOG" in self.db:
            return self.db.root.ChildrenHOG
        else:
            return None

    @property
    def _cprot_arr(self):
        if "/ChildrenProt" in self.db:
            return self.db.root.ChildrenProt
        else:
            return None

    @property
    def _ctax_arr(self):
        if "/ChildrenTax" in self.db:
            return self.db.root.ChildrenTax
        else:
            return None

    @property
    def _level_arr(self):
        if "/LevelOffsets" in self.db:
            return self.db.root.LevelOffsets
        else:
            return None

    ### functions common to DatabaseFromOMA and DatabaseFromFasta ###
    def initiate_tax_tab(self, stree_path):
        """
	except SpeOff column
        """
        assert self.mode in {"w", "a"}, "Database must be opened in write mode."

        def _parse_stree(stree_path, roottax):

            stree = Tree(stree_path, format=1, quoted_node_names=True)
            pruned_stree = [x for x in stree.traverse() if x.name == roottax][0]

            tax2parent = {}
            tax2children = {}
            tax2level = {}
            species = set()

            for tl in pruned_stree.traverse():
                tax = tl.name.encode('ascii')
                tax2parent[tax] = tl.up.name.encode('ascii') if tl.up else 'root'.encode('ascii')
                tax2children[tax] = [x.name.encode('ascii') for x in tl.children]
                tax2level[tax] = tl.get_distance(stree)
                if tl.is_leaf():
                    species.add(tl.name.encode('ascii'))

            return tax2parent, tax2children, tax2level, species

        # parse species tree
        tax2parent, tax2children, tax2level, species = _parse_stree(
            stree_path, self.root_taxon
        )

        # sort taxa and create mapper to their offsets
        sorted_taxa = sorted(tax2parent.keys())
        tax2taxoff = dict(zip(sorted_taxa, range(len(tax2parent))))

        # collect rows of taxonomy table and children buffer
        tax_rows = []
        children_buffer = []

        children_buffer_off = 0
        for tax in sorted_taxa:

            # must not be root
            par_off = (
                np.searchsorted(sorted_taxa, tax2parent[tax])
                if tax != self.root_taxon.encode("ascii")
                else -1
            )

            child_offsets = [np.searchsorted(sorted_taxa, x) for x in tax2children[tax]]

            # if no children, -1
            tax_rows.append(
                (
                    tax,
                    par_off,
                    children_buffer_off if child_offsets else -1,
                    len(child_offsets),
                    -1,
                    tax2level[tax],
                )
            )

            children_buffer.extend(child_offsets)
            children_buffer_off += len(child_offsets)

        # fill tax table
        self._tax_tab.append(tax_rows)
        self._tax_tab.flush()

        # store children taxa
        if not self._ctax_arr:
            self.db.create_carray(
                "/",
                "ChildrenTax",
                obj=np.array(children_buffer, dtype=np.int64),
                filters=self._compr,
            )

        return tax2taxoff, species

    def update_hog_and_fam_tabs(
        self, fam2hogs, hog2taxoff, hog2protoffs, hog2oma_hog=None
    ):
        assert self.mode in {"w", "a"}, "Database must be opened in write mode."

        def _get_parents(hog_off, hog_num, hogs):

            parents = []
            parent2child_hogs = collections.defaultdict(list)

            # sort alphabetically and track their true offsets
            alpha_hogs, alpha_hogs_offs = map(
                np.array, zip(*sorted(zip(hogs, range(hog_num))))
            )

            for i, hog_id in enumerate(hogs):
                child_hog = hog_off + i

                # use alphabetically sorted hogs here because np.searchsorted
                parent_hog = self.find_parent_hog(hog_id, alpha_hogs)
                if parent_hog != -1:
                    # map back to original hog offset and add the global hog_off
                    parent_hog = hog_off + alpha_hogs_offs[parent_hog]
                parents.append(parent_hog)

                if parent_hog != -1:
                    parent2child_hogs[parent_hog].append(child_hog)

            return parents, parent2child_hogs

        def _get_child_hogs(hog_off, hog_num, parent2child_hogs, child_hogs_off):

            children_offsets = []
            children_numbers = []
            children = []
            children_off = child_hogs_off

            for tmp_hog_off in range(hog_off, hog_off + hog_num):

                # get child hogs from the current HOG's offset
                tmp_children = parent2child_hogs.get(tmp_hog_off, [])
                children_num = len(tmp_children)

                # if >0 child hogs, keep track of offset, number and child hogs
                if children_num > 0:
                    children_offsets.append(children_off)
                    children.extend(tmp_children)

                # else, store -1 as offset and 0 as count
                else:
                    children_offsets.append(-1)

                # keep track of number of children
                children_numbers.append(children_num)

                # update children offset
                children_off += children_num

            return children_offsets, children_numbers, children, children_off

        def _get_child_prots(hogs, hog2protoffs, child_prots_off):

            children_offsets = []
            children_numbers = []
            children = []
            children_off = child_prots_off

            for hog in hogs:

                # get child prots from the current HOG id
                tmp_children = hog2protoffs.get(hog, [])
                children_num = len(tmp_children)

                # if >0 child prots, keep track of offset, number and child hogs
                if children_num > 0:
                    children_offsets.append(children_off)
                    children.extend(tmp_children)

                # else, store -1 as offset and 0 as count
                else:
                    children_offsets.append(-1)

                # keep track of number of children
                children_numbers.append(children_num)

                # update children offset
                children_off += children_num

            return children_offsets, children_numbers, children, children_off

        # main
        hog_rows = []
        fam_rows = []

        hog_off = 0
        fam_off = 0

        # initiate HOG and protein children arrays
        child_hogs = []
        child_prots = []
        child_hogs_off = 0
        child_prots_off = 0

        # initiate level array
        level_offsets = [0]
        level_offsets_off = 0

        for fam_id, hogs in sorted(fam2hogs.items(), key=lambda x: int(x[0])):

            ### hogs
            # sort by level
            hogs = sorted(hogs, key=lambda x: len(x.split(b".")))
            hog_num = len(hogs)
            hog_taxoffs = list(map(lambda x: hog2taxoff[x], hogs))

            # parents
            parents, parent2child_hogs = _get_parents(hog_off, hog_num, hogs)

            # levels
            hog_levels = list(map(lambda x: len(x.split(b".")), hogs))
            hog_level_offsets = (
                np.cumsum(np.unique(hog_levels, return_counts=True)[1]) + hog_off
            )
            hog_level_offsets_num = len(hog_level_offsets)
            level_offsets.extend(hog_level_offsets)

            # children hogs
            (
                child_hogs_offsets,
                child_hogs_numbers,
                tmp_child_hogs,
                child_hogs_off,
            ) = _get_child_hogs(hog_off, hog_num, parent2child_hogs, child_hogs_off)

            child_hogs.extend(tmp_child_hogs)

            # children prots
            (
                child_prots_offsets,
                child_prots_numbers,
                tmp_child_prots,
                child_prots_off,
            ) = _get_child_prots(hogs, hog2protoffs, child_prots_off)

            child_prots.extend(tmp_child_prots)

            # OMA hogs
            oma_hogs = (
                list(map(lambda x: hog2oma_hog[x], hogs))
                if hog2oma_hog
                else repeat(b"", hog_num)
            )

            hog_rows.extend(
                list(
                    zip(
                        hogs,
                        repeat(fam_off),
                        hog_taxoffs,
                        parents,
                        child_hogs_offsets,
                        child_hogs_numbers,
                        child_prots_offsets,
                        child_prots_numbers,
                        oma_hogs,
                        repeat(0),
                    )
                )
            )

            ### fams
            fam_rows.append(
                (
                    fam_id,
                    hog_taxoffs[0],
                    hog_off,
                    hog_num,
                    level_offsets_off,
                    hog_level_offsets_num,
                )
            )

            hog_off += hog_num
            fam_off += 1
            level_offsets_off += hog_level_offsets_num

        # fill family and HOG tables
        self._fam_tab.append(fam_rows)
        self._fam_tab.flush()
        self._hog_tab.append(hog_rows)
        self._hog_tab.flush()

        # store children simulaneously without previous initiation because I did not know the size of these
        if not self._chog_arr:
            self.db.create_carray(
                "/",
                "ChildrenHOG",
                obj=np.array(child_hogs, dtype=np.uint64),
                filters=self._compr,
            )
        if not self._cprot_arr:
            self.db.create_carray(
                "/",
                "ChildrenProt",
                obj=np.array(child_prots, dtype=np.uint64),
                filters=self._compr,
            )

        # store offsets of levels of hogs inside families. copy the last one to go reverse
        level_offsets.append(level_offsets[-1])
        if not self._level_arr:
            self.db.create_carray(
                "/",
                "LevelOffsets",
                obj=np.array(level_offsets, dtype=np.int64),
                filters=self._compr,
            )

    def add_speoff_col(self):
        """
	to the taxonomy table
	"""
        # load species and taxa
        species = self._sp_tab.col("ID")
        taxa = self._tax_tab.col("ID")

        # potential idx of each taxon in species
        species_idx = np.searchsorted(species, taxa)

        # add extra elem because np.searchsorted gave idx 1 too high
        species = np.append(species, b"")

        # mask for taxa being species
        species_mask = species[species_idx] == taxa

        # species offsets
        spe_offsets = np.arange(0, species.size - 1)

        # create SpeOff column
        speoff_col = np.full(taxa.size, -1)
        speoff_col[species_mask] = spe_offsets

        # update tax table
        self._tax_tab.modify_column(colname="SpeOff", column=speoff_col)

    def add_taxoff_col(self):
        """
		to the species table
		"""
        # load species and taxa
        species = self._sp_tab.col("ID")
        taxa = self._tax_tab.col("ID")

        tax_offsets = np.searchsorted(taxa, species)

        self._sp_tab.modify_column(colname="TaxOff", column=tax_offsets)

    def update_prot_tab(self, hog2protoffs):
        """
        add HOGoff and FamOff rows in the protein table
	"""
        assert self.mode in {"w", "a"}, "Database must be opened in write mode."

        def _get_protoffs2hogoff(hog2protoffs, hog2hogoff):
            protoff2hogoff = dict()
            for hog, protoffs in hog2protoffs.items():
                protoff2hogoff.update(dict(zip(protoffs, repeat(hog2hogoff[hog]))))
            return protoff2hogoff

        hog2hogoff = dict(zip(self._hog_tab.col("ID"), range(len(self._hog_tab))))
        protoffs2hogoff = _get_protoffs2hogoff(hog2protoffs, hog2hogoff)

        # get the hogoff column from it and the famoff col
        hogoff_col = list(zip(*sorted(protoffs2hogoff.items())))[1]
        famoff_col = self._hog_tab[np.array(hogoff_col)]["FamOff"]

        # replace the empty columns
        self._prot_tab.modify_column(colname="HOGoff", column=hogoff_col)
        self._prot_tab.modify_column(colname="FamOff", column=famoff_col)

    def add_lcataxoff_col(self):
        """
        compute the LCA taxon between the HOG species members
        """
        hog_tab = self._hog_tab[:]
        chog_buff = self._chog_arr[:]
        cprot_buff = self._cprot_arr[:]
        prot2speoff = self._prot_tab.col("SpeOff")
        speoff2taxoff = self._sp_tab.col("TaxOff")
        parent_arr = self._tax_tab.col("ParentOff")

        lcatax_offsets = np.zeros(hog_tab.size, dtype=np.uint64)

        for hog_off in range(hog_tab.size):
            species_tax_offs = get_descendant_species_taxoffs(
                hog_off, hog_tab, chog_buff, cprot_buff, prot2speoff, speoff2taxoff
            )
            lcatax_offsets[hog_off] = get_lca_hog_off(species_tax_offs, parent_arr)

        self._hog_tab.modify_column(colname="LCAtaxOff", column=lcatax_offsets)

    ### generic functions ###
    @staticmethod
    def parse_hogs(hog_id):
        if isinstance(hog_id, str):
            split_hog_id = hog_id.split(".")
            return [".".join(split_hog_id[: i + 1]) for i in range(len(split_hog_id))]
        elif isinstance(hog_id, bytes):
            split_hog_id = hog_id.split(b".")
            return [b".".join(split_hog_id[: i + 1]) for i in range(len(split_hog_id))]

    @staticmethod
    def find_parent_hog(hog_id, hogs):
        if isinstance(hog_id, str):
            split_hog = hog_id.split(".")
            parent_hog = -1
            if len(split_hog) > 1:
                parent_hog = np.searchsorted(hogs, ".".join(split_hog[:-1]))
            return parent_hog
        elif isinstance(hog_id, bytes):
            split_hog = hog_id.split(b".")
            parent_hog = -1
            if len(split_hog) > 1:
                parent_hog = np.searchsorted(hogs, b".".join(split_hog[:-1]))
            return parent_hog


class DatabaseFromOMA(Database):
    """
    Used to parse the OMA browser database file
    """
    def __init__(self, filename, root_taxon, min_prot_nr=6, include_younger_fams=True, mode='r'):
        super().__init__(filename, root_taxon, mode=mode)

        self.min_prot_nr = min_prot_nr
        self.include_younger_fams = include_younger_fams

    ### main function ###
    def build_database(self, oma_h5_path, stree_path):
        assert self.mode in {"w", "a"}, "Database must be opened in write mode."

        # build taxonomy table except the SpeOff column
        LOG.debug("initiate taxonomy table")
        tax2taxoff, species = self.initiate_tax_tab(stree_path)

        # load OMA h5 file
        h5file = tables.open_file(oma_h5_path, mode="r")

        LOG.debug("select and strip OMA HOGs")
        fam2hogs, hog2oma_hog, hog2tax = self.select_and_strip_OMA_HOGs(h5file)

        LOG.debug("fill sequence buffer, species table and initiate protein table")
        (
            fam2hogs,
            hog2protoffs,
            hog2tax,
            hog2oma_hog,
        ) = self.select_and_filter_OMA_proteins(
            h5file,
            fam2hogs,
            hog2oma_hog,
            hog2tax,
            species,
            self.min_prot_nr
        )

        LOG.debug(
            "add SpeOff and TaxOff columns in taxonomy and species tables, respectively"
        )
        self.add_speoff_col()
        self.add_taxoff_col()

        # mapper HOG to taxon offset
        hog2taxoff = {h: tax2taxoff.get(t, -1) for h, t in hog2tax.items()}

        LOG.debug("fill family and HOG tables")
        self.update_hog_and_fam_tabs(fam2hogs, hog2taxoff, hog2protoffs, hog2oma_hog)

        # add family and hog offsets
        LOG.debug("complete protein table")
        self.update_prot_tab(hog2protoffs)

        LOG.debug("compute LCA taxa")
        self.add_lcataxoff_col()

        # close and open in read mode
        h5file.close()

    ### functions to parse OMA database ###
    def select_and_strip_OMA_HOGs(self, h5file):
        def _process_oma_hog(
            tax2level,
            curr_oma_taxa,
            curr_oma_hog,
            curr_oma_roothog,
            fam,
            fam2hogs,
            hog2oma_hog,
            hog2tax,
            roottax,
            include_younger_fams
        ):
            """
			- decide whether an OMA HOG should be stored based on current root-HOG and HOG taxa
			- update root-HOG if necessary and keep track of HOG taxa
			"""

            def _is_descendant(hog1, hog2):
                """
			    True if hog1 descendant of hog2
			    """
                return hog2 in self.parse_hogs(hog1)

            def _store(
                fam, curr_oma_hog, curr_oma_roothog, fam2hogs, hog2oma_hog, hog2tax, tax
            ):
                """
			    - compute SPhog HOG id
			    - fill containers
			    """
                hog = b".".join(
                    [
                        str(fam).encode("ascii"),
                        *curr_oma_hog.split(b".")[len(curr_oma_roothog.split(b".")) :],
                    ]
                )

                fam2hogs[fam].add(hog)
                hog2oma_hog[hog] = curr_oma_hog
                hog2tax[hog] = tax

            # compute most ancestral taxon; when absent, flag it with 1000000
            tax_levels = list(map(lambda x: tax2level.get(x, 1000000), curr_oma_taxa))
            min_level = min(tax_levels)
            
            # skip HOGs outside of taxonomic scope defined by the root-taxon
            if min_level == 1000000:
                return fam, curr_oma_roothog
            else:
                tax = curr_oma_taxa[tax_levels.index(min_level)]  
                
                # whether HOG is descendant of current OMA root-HOG 
                if _is_descendant(curr_oma_hog, curr_oma_roothog):

                    _store(
                        fam,
                        curr_oma_hog,
                        curr_oma_roothog,
                        fam2hogs,
                        hog2oma_hog,
                        hog2tax,
                        tax,
                    )
                
                # or create a new family (include only family at root-taxon if include_younger_fams==False)
                elif (not include_younger_fams and tax == roottax) or include_younger_fams:

                    fam += 1
                    curr_oma_roothog = curr_oma_hog

                    # store after updating fam
                    _store(
                        fam,
                        curr_oma_hog,
                        curr_oma_roothog,
                        fam2hogs,
                        hog2oma_hog,
                        hog2tax,
                        tax,
                    )

            return fam, curr_oma_roothog
            
            # # compute most ancestral taxon; when absent, flag it with 1000000
            # tax_levels = list(map(lambda x: tax2level.get(x, 1000000), curr_oma_taxa))
            # min_level = min(tax_levels)

            # if min_level == 1000000:
            #     tax = None
            # else:
            #     tax = curr_oma_taxa[tax_levels.index(min_level)]

            # # store if descendant of current OMA root-HOG
            # if _is_descendant(curr_oma_hog, curr_oma_roothog):

            #     _store(
            #         fam,
            #         curr_oma_hog,
            #         curr_oma_roothog,
            #         fam2hogs,
            #         hog2oma_hog,
            #         hog2tax,
            #         tax,
            #     )

            # # if tax equal or younger that root taxon: store, update fam and current OMA root-HOG
            # elif tax:

            #     # include only family at root-taxon if include_younger_fams==False
            #     if (not include_younger_fams and tax == roottax) or include_younger_fams:

            #         fam += 1
            #         curr_oma_roothog = curr_oma_hog

            #         # store after updating fam
            #         _store(
            #             fam,
            #             curr_oma_hog,
            #             curr_oma_roothog,
            #             fam2hogs,
            #             hog2oma_hog,
            #             hog2tax,
            #             tax,
            #         )

            # return fam, curr_oma_roothog

        def _process_oma_fam(
            fam_tab_sort, tax2level, fam, fam2hogs, hog2oma_hog, hog2tax, roottax, include_younger_fams
        ):
            """
			apply _process_oma_hog to one OMA family
			 - fam_tab_sort: slice of the HogLevel table for one family sorted by HOG ids
			 - tax2level: mapper between taxa and their distance from roottax
			"""
            # bookeepers for HOGs
            curr_oma_hog = fam_tab_sort[0]["ID"]
            curr_oma_roothog = None
            curr_oma_taxa = []

            for r in fam_tab_sort:
                oma_hog, oma_tax = r[1], r[2] #b"".join(r[2].split())

                # evaluation at new oma HOG
                if oma_hog != curr_oma_hog:

                    fam, curr_oma_roothog = _process_oma_hog(
                        tax2level,
                        curr_oma_taxa,
                        curr_oma_hog,
                        curr_oma_roothog,
                        fam,
                        fam2hogs,
                        hog2oma_hog,
                        hog2tax,
                        roottax,
                        include_younger_fams
                    )

                    # reset for new HOG
                    curr_oma_taxa = []
                    curr_oma_hog = oma_hog

                # track taxa of current oma HOG
                curr_oma_taxa.append(oma_tax)

            # end
            fam, curr_oma_roothog = _process_oma_hog(
                tax2level,
                curr_oma_taxa,
                curr_oma_hog,
                curr_oma_roothog,
                fam,
                fam2hogs,
                hog2oma_hog,
                hog2tax,
                roottax,
                include_younger_fams
            )

            return fam

        #
        tax2level = dict(zip(self._tax_tab[:]["ID"], self._tax_tab[:]["Level"]))
        hog_tab = h5file.root.HogLevel

        # load families in memory
        families = hog_tab.col("Fam")

        # containers
        fam2hogs = collections.defaultdict(set)
        hog2oma_hog = dict()
        hog2tax = dict()

        # bookeepers for families
        fam = 0
        curr_fam = families[0]
        i = 0
        j = 0

        for oma_fam in tqdm(families, disable=is_progress_disabled()):
            if oma_fam != curr_fam:

                # load fam table and sort by HOG ids
                fam_tab = hog_tab[i:j]
                fam_tab = np.sort(fam_tab, order="ID")

                # select and strip HOGs of one family
                fam = _process_oma_fam(
                    fam_tab,
                    tax2level,
                    fam,
                    fam2hogs,
                    hog2oma_hog,
                    hog2tax,
                    self.root_taxon.encode("ascii"),
                    self.include_younger_fams
                )

                # move pointer and update current family
                i = j
                curr_fam = oma_fam
            j += 1

        # end
        fam_tab = hog_tab[i:j]
        fam_tab = np.sort(fam_tab, order="ID")
        fam = _process_oma_fam(
            fam_tab,
            tax2level,
            fam,
            fam2hogs,
            hog2oma_hog,
            hog2tax,
            self.root_taxon.encode("ascii"),
            self.include_younger_fams
        )

        del hog_tab, families

        return fam2hogs, hog2oma_hog, hog2tax

    def select_and_filter_OMA_proteins(
        self, h5file, fam2hogs, hog2oma_hog, hog2tax, species, min_prot_nr
    ):
        """
        One small diff compared to DatabaseFromFasta is that proteins in protein table are not following the species in species table. This should be fine.
        """
        genome_tab = h5file.root.Genome[:]
        ent_tab = h5file.root.Protein.Entries
        oma_seq_buffer = h5file.root.Protein.SequenceBuffer

        # temporary mappers
        oma_hog2hog = dict(zip(hog2oma_hog.values(), hog2oma_hog.keys()))
        spe2speoff = dict(
            zip(sorted(species), range(len(species)))
        )  # this is to sort the species table

        LOG.debug(" - select proteins from selected HOGs")
        # remember the size of families and bookkeep family of each protein for filtering
        fam2prot_nr = collections.Counter()
        prot_fams = []

        # bookkeep HOG of each protein to later build the hog2protoffs mapper
        prot_hogs = []

        # bookkeep sequence offset of each protein to later build the local seq_buff
        oma_seq_offsets = []

        # store temporary rows for species and protein tables
        oma_species = [b""] * len(species)  # to keep species sorted
        prot_rows = []

        for r in tqdm(genome_tab, disable=is_progress_disabled()):

            sp = r['SciName']
            sp_code = r['UniProtSpeciesCode']
            
            # use species code if scientific name is lacking
            # for ~27 cases the uniprot id replaces the scientific name in OMA species tree
            if sp not in species:
                sp = sp_code

            # filter if species outside root-taxon
            if sp in species:
                entry_off = r["EntryOff"]
                entry_num = r["TotEntries"]

                spe_off = spe2speoff[sp]
                oma_species[spe_off] = sp

                # sub entry table for species
                sp_ent_tab = ent_tab[entry_off : entry_off + entry_num]

                for rr in sp_ent_tab:
                    oma_hog = rr["OmaHOG"]
                    # select protein if member of selected OMA HOG
                    if oma_hog in oma_hog2hog:
                        # update counter and track hogs and families
                        hog = oma_hog2hog[oma_hog]
                        fam = int(hog.split(b".")[0].decode("ascii"))
                        fam2prot_nr[fam] += 1
                        prot_hogs.append(hog)
                        prot_fams.append(fam)

                        # track sequence length and offset
                        oma_seq_off = rr["SeqBufferOffset"]
                        seq_len = rr["SeqBufferLength"]
                        oma_seq_offsets.append(oma_seq_off)

                        # store protein row
                        oma_id = '{}{:05d}'.format(sp_code.decode('ascii'), rr['EntryNr'] - entry_off)
                        prot_rows.append((oma_id.encode('ascii'), 0, seq_len, spe_off, 0, 0))

        LOG.debug(" - filter by family protein number")
        # now filter the mappers
        f_fam2hogs = collections.defaultdict(set)
        f_hog2oma_hog = dict()
        f_hog2tax = dict()

        for fam, hogs in fam2hogs.items():

            prot_nr = fam2prot_nr[fam]

            # filter by size
            if prot_nr >= min_prot_nr:

                f_fam2hogs[fam] = hogs
                f_hog2oma_hog.update(
                    dict(zip(hogs, list(map(lambda x: hog2oma_hog[x], hogs))))
                )
                f_hog2tax.update(dict(zip(hogs, list(map(lambda x: hog2tax[x], hogs)))))

        LOG.debug(" - filter proteins based on filtered families")
        # bookkeeping for later
        hog2protoffs = collections.defaultdict(set)

        # pointer to species in species table
        curr_spe_off = prot_rows[0][3]

        # pointer to protein in protein table
        prot_off = self._prot_tab.nrows
        curr_prot_off = prot_off

        # pointer to sequence in buffer
        seq_off = self._seq_buff.nrows

        # store rows for species and protein tables	and sequence buffer
        spe_rows = [()] * len(species)  # keep sorted
        f_prot_rows = []
        seq_buff = []

        # iter trough temporary protein rows
        i = 0
        for r in tqdm(prot_rows, disable=is_progress_disabled()):
            if prot_fams[i] in f_fam2hogs:
                spe_off = r[3]

                # change of species --> store current species
                if spe_off != curr_spe_off:

                    spe_rows[curr_spe_off] = (
                        oma_species[curr_spe_off],
                        curr_prot_off,
                        prot_off - curr_prot_off,
                        0,
                    )

                    curr_prot_off = prot_off
                    curr_spe_off = spe_off

                    # take advantage to dump sequence buffer
                    self._seq_buff.append(seq_buff)
                    self._seq_buff.flush()
                    seq_buff = []

                # sequence
                oma_seq_off = oma_seq_offsets[i]
                seq_len = r[2]
                seq = oma_seq_buffer[oma_seq_off : oma_seq_off + seq_len]
                seq_buff.extend(list(seq))

                # store protein row
                f_prot_rows.append((r[0], seq_off, seq_len, spe_off, 0, 0))

                # book keeping
                hog2protoffs[prot_hogs[i]].add(prot_off)

                # update offset of protein sequence in buffer and of protein raw in table
                seq_off += seq_len
                prot_off += 1
            i += 1

        # end
        spe_rows[curr_spe_off] = (
            oma_species[curr_spe_off],
            curr_prot_off,
            prot_off - curr_prot_off,
            0,
        )

        self._seq_buff.append(seq_buff)
        self._seq_buff.flush()

        # fill species and protein tables
        self._sp_tab.append(spe_rows)
        self._sp_tab.flush()
        self._prot_tab.append(f_prot_rows)
        self._prot_tab.flush()

        del genome_tab, ent_tab, oma_seq_buffer

        return f_fam2hogs, hog2protoffs, f_hog2tax, f_hog2oma_hog
