"""
    OMAmer - tree-driven and alignment-free protein assignment to sub-families

    (C) 2022-2023 Alex Warwick Vesztrocy <alex.warwickvesztrocy@unil.ch>
    (C) 2019-2021 Victor Rossier <victor.rossier@unil.ch> and
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
"""
from collections import defaultdict
from itertools import repeat
from packaging.version import parse as parse_version
from tqdm.auto import tqdm
import numpy as np
import os
import tables

from . import __version__
from .hierarchy import (
    get_hog_child_prots,
    get_hog2taxa,
    traverse,
)
from .index import Index
from ._utils import LOG, is_progress_disabled, compute_file_md5


PROT_ID_LEN = 12


class Database(object):
    """
    Database definition for the OMAmer algorithm.
    """

    class ProteinTableFormat(tables.IsDescription):
        # Note, the ID field could be generalised and replaced with a SequenceBuffer,
        # for now use OMA formatted sequence identifiers. 5CHAR + max 7digit.
        ID = tables.StringCol(PROT_ID_LEN, pos=1, dflt=b"")
        SpeOff = tables.Int32Col(pos=2)
        HOGoff = tables.UInt32Col(pos=3)
        SeqLen = tables.UInt32Col(pos=4)

    class HOGTableFormat(tables.IsDescription):
        IDBufferOff = tables.UInt64Col(pos=1)
        IDLen = tables.UInt16Col(pos=2)
        FamOff = tables.UInt32Col(pos=3)
        TaxOff = tables.UInt32Col(pos=4)
        ParentOff = tables.Int32Col(pos=5)
        ChildrenOff = tables.Int32Col(pos=6)
        ChildrenNum = tables.Int32Col(pos=7)
        ChildrenProtOff = tables.Int32Col(pos=8)
        ChildrenProtNum = tables.Int32Col(pos=9)
        HOGtaxaOff = tables.UInt32Col(pos=10)
        HOGtaxaNum = tables.UInt32Col(pos=11)
        LCAtaxOff = tables.UInt32Col(pos=12)
        NrMemberGenes = tables.UInt32Col(pos=13)
        CompletenessScore = tables.Float64Col(pos=14)  # not used in search, but potentially useful 
        MedianSeqLen = tables.UInt32Col(pos=15)

    class FamilyTableFormat(tables.IsDescription):
        ID = tables.UInt32Col(pos=1)
        TaxOff = tables.UInt32Col(pos=2)
        HOGoff = tables.UInt32Col(pos=3)
        HOGnum = tables.UInt32Col(pos=4)
        LevelOff = tables.UInt32Col(pos=5)
        LevelNum = tables.UInt32Col(pos=6)

    class SpeciesTableFormat(tables.IsDescription):
        ID = tables.StringCol(255, pos=1, dflt=b"")
        TaxOff = tables.UInt32Col(pos=2)

    class TaxonomyTableFormat(tables.IsDescription):
        ID = tables.StringCol(255, pos=1, dflt=b"")
        TaxID = tables.Int32Col(pos=2)
        ParentOff = tables.Int32Col(pos=3)
        ChildrenOff = tables.Int32Col(pos=4)
        ChildrenNum = tables.Int32Col(pos=5)
        SpeOff = tables.Int32Col(pos=6)
        Level = tables.Int32Col(pos=7)

    def __init__(self, filename, root_taxon=None, mode="r"):
        assert mode in {
            "r",
            "w",
        }, "Databased must only be opened in read or write mode."
        assert (
            mode != "w" or root_taxon
        ), "A root_taxon must be defined when building the database"
        self.root_taxon = root_taxon
        self.filename = filename
        self.mode = mode

        self._compr = tables.Filters(complevel=6, complib="blosc", fletcher32=True)

        if os.path.isfile(self.filename) and self.mode == "w":
            LOG.warning("Overwriting database file ({})".format(self.filename))

        self.db = tables.open_file(self.filename, self.mode, filters=self._compr)

        if "/Index" in self.db:
            self.ki = Index(self)
        if self.mode == "r":
            self._check_db_version()

    def _check_db_version(self):
        # check that the database version is the same minor version as us.
        db_version = parse_version(self.get_metadata()["omamer version"])
        my_version = parse_version(__version__)
        if (db_version.major == my_version.major) and (
            db_version.minor == my_version.minor
        ):
            pass
        elif (db_version.major == my_version.major) and (
            db_version.minor < my_version.minor
        ):
            LOG.warning(
                "Database version mismatch: DB {} / OMAmer {}".format(
                    db_version, my_version
                )
            )
        else:
            raise RuntimeError(
                "Database major version mismatch: DB {} / OMAmer {}".format(
                    db_version, my_version
                )
            )

    def _check_open_writeable(self):
        assert self.mode == "w", "Database must be opened in write mode."

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __getattr__(self, attr):
        # make paths in the database accessible as attributes
        if attr.startswith("_db_"):
            path = "/" + "/".join(attr.split("_")[2:])
            if path in self.db:
                return self.db.get_node(path)
            else:
                raise tables.NoSuchNodeError(
                    "Path not found in database ({})".format(path)
                )

        return self.__getattribute__(attr)

    def get_hog_id(self, i):
        # extract the hog id from the hog id buffer
        x = self._db_HOG[i]
        s = x["IDBufferOff"]
        e = s + x["IDLen"]
        return self._db_HOGIDBuffer[s:e].tobytes().decode("ascii")

    def initiate_tax_tab(self, stree_path):
        """
        except SpeOff column
        """
        self._check_open_writeable()

        def _parse_stree(stree_path, roottax):
            from ete3 import Tree

            stree = Tree(stree_path, format=1, quoted_node_names=True)
            pruned_stree = [x for x in stree.traverse() if x.name == roottax][0]

            tax2parent = {}
            tax2children = {}
            tax2level = {}
            species = set()

            for tl in pruned_stree.traverse():
                tax = tl.name.encode("ascii")
                tax2parent[tax] = (
                    tl.up.name.encode("ascii") if tl.up else "root".encode("ascii")
                )
                tax2children[tax] = [x.name.encode("ascii") for x in tl.children]
                tax2level[tax] = tl.get_distance(stree)
                if tl.is_leaf():
                    species.add(tl.name.encode("ascii"))

            return tax2parent, tax2children, tax2level, species

        # create tax table
        tax_tab = self.db.create_table(
            "/", "Taxonomy", self.TaxonomyTableFormat, filters=self._compr
        )

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
                    -1,
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
        tax_tab.append(tax_rows)
        tax_tab.flush()

        # store children taxa
        self.db.create_carray(
            "/",
            "ChildrenTax",
            obj=np.array(children_buffer, dtype=np.int32),
            filters=self._compr,
        )

        return (tax2taxoff, species)

    def update_hog_and_fam_tabs(
        self,
        fam2hogs,
        hog2taxoff,
        hog2protoffs,
        hog2oma_hog,
        hog2gene_nr,
        hog2completeness,
    ):
        self._check_open_writeable()

        def _get_parents(hog_off, hog_num, hogs):
            parents = []
            parent2child_hogs = defaultdict(list)

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

        hog_tab = self.db.create_table(
            "/", "HOG", self.HOGTableFormat, filters=self._compr, expectedrows=1e7
        )
        fam_tab = self.db.create_table(
            "/", "Family", self.FamilyTableFormat, filters=self._compr, expectedrows=1e6
        )
        hog_tab.autoindex = fam_tab.autoindex = False

        # initiate HOG and protein children arrays
        hog_off = 0
        fam_off = 0
        child_hogs = []
        child_prots = []
        child_hogs_off = 0
        child_prots_off = 0
        hog2hogoff = {}

        # initiate level array
        level_offsets = [0]
        level_offsets_off = 0

        # initialise hog id buffer
        hog_id_buff = self.db.create_earray(
            "/", "HOGIDBuffer", tables.StringAtom(1), (0,), filters=self._compr
        )

        for fam_id, hogs in sorted(fam2hogs.items(), key=lambda x: int(x[0])):
            ### hogs
            # sort by level
            hogs = sorted(hogs, key=lambda x: len(x.split(b".")))
            hog_num = len(hogs)

            # parents
            (parents, parent2child_hogs) = _get_parents(hog_off, hog_num, hogs)

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

            # store family information
            fam_tab.append(
                [
                    (
                        fam_id,
                        hog2taxoff[hogs[0]],
                        hog_off,
                        hog_num,
                        level_offsets_off,
                        hog_level_offsets_num,
                    ),
                ]
            )

            # store sub-family information
            for i in range(len(hogs)):
                oma_hog_id = hog2oma_hog[hogs[i]]
                oma_hog_size = hog2gene_nr[hogs[i]]
                oma_hog_completeness = hog2completeness[hogs[i]]
                hog_taxoff = hog2taxoff[hogs[i]]

                # write record
                r = (
                    len(hog_id_buff),
                    len(oma_hog_id),
                    fam_off,
                    hog_taxoff,
                    parents[i],
                    child_hogs_offsets[i],
                    child_hogs_numbers[i],
                    child_prots_offsets[i],
                    child_prots_numbers[i],
                    -1,
                    -1,
                    0,
                    oma_hog_size,
                    oma_hog_completeness,
                    0,
                )
                hog_tab.append(
                    [
                        r,
                    ]
                )

                hog_id_buff.append(
                    np.frombuffer(oma_hog_id, dtype=tables.StringAtom(1))
                )

                # store hog2hogoff
                hog2hogoff[hogs[i]] = hog_off + i

            fam_off += 1
            level_offsets_off += hog_level_offsets_num
            hog_off += hog_num

        # flush the family and HOG tables and then reset autoindex
        fam_tab.flush()
        hog_tab.flush()
        hog_id_buff.flush()
        hog_tab.autoindex = fam_tab.autoindex = True

        # create child hog and protein tables
        self.db.create_carray(
            "/",
            "ChildrenHOG",
            obj=np.array(child_hogs, dtype=np.uint32),
            filters=self._compr,
        )
        self.db.create_carray(
            "/",
            "ChildrenProt",
            obj=np.array(child_prots, dtype=np.uint32),
            filters=self._compr,
        )

        # store offsets of levels of hogs inside families. copy the last one to go reverse
        level_offsets.append(level_offsets[-1])
        self.db.create_carray(
            "/",
            "LevelOffsets",
            obj=np.array(level_offsets, dtype=np.uint32),
            filters=self._compr,
        )
        return hog2hogoff

    def add_speoff_col(self):
        """
        to the taxonomy table
        """
        # load species and taxa
        species = self._db_Species.col("ID")
        taxa = self._db_Taxonomy.col("ID")

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
        self._db_Taxonomy.modify_column(colname="SpeOff", column=speoff_col)

    def add_taxoff_col(self):
        """
        to the species table
        """
        # load species and taxa
        species = self._db_Species.col("ID")
        taxa = self._db_Taxonomy.col("ID")

        tax_offsets = np.searchsorted(taxa, species)

        self._db_Species.modify_column(colname="TaxOff", column=tax_offsets)

    def update_prot_tab(self, hog2protoffs, hog2hogoff):
        """
        add HOGoff column in the protein table
        """
        self._check_open_writeable()

        # newer way of doing this
        hogoff_col = np.zeros(len(self._db_Protein), dtype=np.uint32)
        for hog, protoffs in hog2protoffs.items():
            hogoff_col[np.array(list(protoffs), dtype=np.uint64)] = hog2hogoff[hog]

        # replace the empty columns
        self._db_Protein.modify_column(colname="HOGoff", column=hogoff_col)

    def store_hog2taxa(self):
        """
        Store taxonomic levels of HOGs.
        """
        (hog_taxa_idx, hog_taxa_buff) = get_hog2taxa(
            self._db_HOG[:],
            self._db_Species[:],
            self._db_Protein.col("SpeOff"),
            self._db_ChildrenProt[:],
            self._db_Taxonomy[:],
            self._db_ChildrenHOG[:],
        )

        # this is already a uint32
        self.db.create_carray("/", "HOGtaxa", obj=hog_taxa_buff, filters=self._compr)

        self._db_HOG.modify_column(colname="HOGtaxaOff", column=hog_taxa_idx[:-1])
        self._db_HOG.modify_column(
            colname="HOGtaxaNum", column=hog_taxa_idx[1:] - hog_taxa_idx[:-1]
        )

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

    def add_median_seqlen_col(self):
        """
        Compute the median sequence length for each HOG
        """

        def compute_median_seq_len(
            hog_off, hog_tab, chog_buff, prot_seq_lens, cprot_buff, median_seq_lengths
        ):
            def _compute_median_seq_len(
                hog_off,
                parent2seq_lengths,
                prot_seq_lens,
                hog_tab2,
                cprot_buff,
                median_seq_lengths,
            ):
                # compute HOG sequence lengths
                seq_lengths = list(
                    map(
                        lambda x: prot_seq_lens[x],
                        get_hog_child_prots(hog_off, hog_tab2, cprot_buff),
                    )
                )

                # merge these with child sequence lentgths (stored in parent2seq_lengths)
                seq_lengths = seq_lengths + parent2seq_lengths[hog_off]

                # store in parent2seq_lengths
                parent2seq_lengths[hog_tab2["ParentOff"][hog_off]] += seq_lengths

                # store median
                median_seq_lengths[hog_off] = np.median(seq_lengths)

                return parent2seq_lengths

            parent2seq_lengths = defaultdict(list)

            parent2seq_lengths = traverse(
                hog_off,
                hog_tab,
                chog_buff,
                parent2seq_lengths,
                _compute_median_seq_len,
                None,
                _compute_median_seq_len,
                prot_seq_lens=prot_seq_lens,
                hog_tab2=hog_tab,
                cprot_buff=cprot_buff,
                median_seq_lengths=median_seq_lengths,
            )

        fam_tab = self._db_Family[:]
        hog_tab = self._db_HOG[:]
        chog_buff = self._db_ChildrenHOG[:]
        cprot_buff = self._db_ChildrenProt[:]
        prot_seq_lens = self._db_Protein.col("SeqLen")

        median_seq_lengths = np.zeros(hog_tab.size, dtype=np.uint32)
        for hog_off in fam_tab["HOGoff"]:
            compute_median_seq_len(
                hog_off,
                hog_tab,
                chog_buff,
                prot_seq_lens,
                cprot_buff,
                median_seq_lengths,
            )

        self._db_HOG.modify_column(colname="MedianSeqLen", column=median_seq_lengths)

    def add_metadata(self):
        """
        Store metadata within the database to track the versioning / build time.
        """
        from datetime import datetime

        from . import __version__

        self.db.set_node_attr("/", "omamer_version", __version__)
        self.db.set_node_attr("/", "root_level", self.root_taxon)
        self.db.set_node_attr("/", "create_timestamp", datetime.now().isoformat())

    def get_metadata(self):
        """
        Load metadata from the database.
        """
        attrs = self.db.root._v_attrs
        meta = {
            k.replace("_", " "): attrs[k] for k in attrs._f_list() if k != "oma_version"
        }
        if "source" in meta:
            src_version = meta["source"].lower() + "_version"
            meta["source"] += " / {}".format(attrs[src_version])
        if "omamer version" not in meta:
            meta["omamer version"] = "<= 0.2.3"
        meta["k-mer length"] = self.ki.k
        meta["alphabet size"] = self.ki.alphabet.n
        meta["nr species"] = len(self._db_Species)
        meta["hidden taxa"] = self.ki.hidden_taxa
        return meta

    def add_md5_hash(self):
        # generate unique hash for the database
        self._check_open_writeable()

        # temporarily close the database
        self.db.close()

        md5 = compute_file_md5(self.filename)

        # reopen the database and save the hash
        self.db = tables.open_file(self.filename, "a", filters=self._compr)
        self.db.set_node_attr("/", "database_hash", md5)


class DatabaseFromOMA(Database):
    """
    Used to parse the OMA browser database file
    """

    def __init__(
        self,
        filename,
        root_taxon,
        min_fam_size=6,
        min_fam_completeness=0.5,
        logic="OR",
        include_younger_fams=True,
        mode="r",
    ):
        super().__init__(filename, root_taxon, mode=mode)

        self.min_fam_size = min_fam_size
        self.min_fam_completeness = min_fam_completeness
        self.logic = logic
        self.include_younger_fams = include_younger_fams
        self.oma_version = ""

    ### main function ###
    def build_database(self, oma_h5_path, stree_path):
        self._check_open_writeable()

        # load OMA h5 file
        h5file = tables.open_file(oma_h5_path, mode="r")
        try:
            self.oma_version = h5file.get_node_attr("/", "oma_version")
        except AttributeError:
            raise Exception(
                "Provided file '{}' does not seem to be an OMA HDF5 database".format(
                    oma_h5_path
                )
            )

        # build taxonomy table except the SpeOff column
        LOG.debug("initiate taxonomy table")
        tax2taxoff, species = self.initiate_tax_tab(stree_path)
        self.add_taxid_col(h5file)

        LOG.debug("select and strip OMA HOGs")
        (
            fam2hogs,
            hog2oma_hog,
            hog2tax,
            hog2gene_nr,
            hog2completeness,
        ) = self.select_and_strip_OMA_HOGs(h5file)

        LOG.debug("fill sequence buffer, species table and initiate protein table")
        (
            fam2hogs,
            hog2protoffs,
            hog2tax,
            hog2oma_hog,
            seq_buff,
        ) = self.select_and_filter_OMA_proteins(
            h5file, fam2hogs, hog2oma_hog, hog2tax, species, self.min_fam_size
        )

        LOG.debug("add SpeOff and TaxOff columns in taxonomy and species tables")
        self.add_speoff_col()
        self.add_taxoff_col()

        # mapper HOG to taxon offset
        hog2taxoff = {h: tax2taxoff.get(t, -1) for h, t in hog2tax.items()}

        LOG.debug("fill family and HOG tables")
        hog2hogoff = self.update_hog_and_fam_tabs(
            fam2hogs,
            hog2taxoff,
            hog2protoffs,
            hog2oma_hog,
            hog2gene_nr,
            hog2completeness,
        )

        # add family and hog offsets
        LOG.debug("complete protein table")
        self.update_prot_tab(hog2protoffs, hog2hogoff)

        LOG.debug("store HOG taxa")
        self.store_hog2taxa()

        # compute the median sequence length of each HOG
        self.add_median_seqlen_col()

        # close and open in read mode
        h5file.close()

        return seq_buff

    def add_metadata(self):
        super().add_metadata()
        self.db.set_node_attr("/", "source", "OMA")
        self.db.set_node_attr("/", "oma_version", self.oma_version)
        self.db.set_node_attr("/", "min_fam_size", self.min_fam_size)
        self.db.set_node_attr("/", "min_fam_completeness", self.min_fam_completeness)
        self.db.set_node_attr("/", "filter_logic", self.logic)
        self.db.set_node_attr("/", "include_younger_fams", self.include_younger_fams)

    ### functions to parse OMA database ###
    def select_and_strip_OMA_HOGs(self, h5file):
        def _process_oma_hog(
            tax2level,
            curr_oma_taxa,
            curr_oma_hog,
            curr_oma_roothog,
            curr_oma_roothog_ok,
            fam,
            fam2hogs,
            hog2oma_hog,
            hog2tax,
            hog2gene_nr,
            hog2completeness,
            roottax,
            include_younger_fams,
            curr_oma_sizes,
            min_fam_size,
            curr_oma_comps,
            min_fam_completeness,
            logic,
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
                fam,
                curr_oma_hog,
                curr_oma_roothog,
                fam2hogs,
                hog2oma_hog,
                hog2tax,
                hog2gene_nr,
                hog2completeness,
                tax,
                hog_size,
                hog_comp,
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
                hog2gene_nr[hog] = hog_size
                hog2completeness[hog] = hog_comp

            # compute most ancestral taxon; when absent, flag it with 1000000
            tax_levels = list(map(lambda x: tax2level.get(x, 1000000), curr_oma_taxa))
            min_level = min(tax_levels)

            # if none of the HOG taxa within the taxonomic scope defined by the root-taxon, skip the HOG
            if min_level == 1000000:
                pass

            else:
                # the HOG taxon is the older taxon within the taxonimic scope defined by the root-taxon
                idx = tax_levels.index(min_level)
                hog_tax = curr_oma_taxa[idx]
                hog_size = curr_oma_sizes[idx]
                hog_comp = curr_oma_comps[idx]

                # if the HOG is descendant of current OMA root-HOG, new sub-HOG
                if _is_descendant(curr_oma_hog, curr_oma_roothog):
                    # but store only if root-HOG passed quality thresholds
                    if curr_oma_roothog_ok:
                        _store(
                            fam,
                            curr_oma_hog,
                            curr_oma_roothog,
                            fam2hogs,
                            hog2oma_hog,
                            hog2tax,
                            hog2gene_nr,
                            hog2completeness,
                            hog_tax,
                            hog_size,
                            hog_comp,
                        )

                # else, new root-HOG (include only root-HOG at root-taxon if include_younger_fams==False)
                elif (
                    not include_younger_fams and hog_tax == roottax
                ) or include_younger_fams:
                    curr_oma_roothog = curr_oma_hog

                    # but store it only if passes quality thresholds
                    ok = (
                        (hog_size >= min_fam_size or hog_comp >= min_fam_completeness)
                        if logic == "OR"
                        else (
                            hog_size >= min_fam_size
                            and hog_comp >= min_fam_completeness
                        )
                    )
                    if ok:
                        fam += 1
                        curr_oma_roothog_ok = True

                        # store after updating fam
                        _store(
                            fam,
                            curr_oma_hog,
                            curr_oma_roothog,
                            fam2hogs,
                            hog2oma_hog,
                            hog2tax,
                            hog2gene_nr,
                            hog2completeness,
                            hog_tax,
                            hog_size,
                            hog_comp,
                        )

                    else:
                        curr_oma_roothog_ok = False

            return (fam, curr_oma_roothog, curr_oma_roothog_ok)

        def _process_oma_fam(
            fam_tab_sort,
            tax2level,
            fam,
            fam2hogs,
            hog2oma_hog,
            hog2tax,
            hog2gene_nr,
            hog2completeness,
            roottax,
            include_younger_fams,
            min_fam_size,
            min_fam_completeness,
            logic,
        ):
            """
            apply _process_oma_hog to one OMA family
             - fam_tab_sort: slice of the HogLevel table for one family sorted by HOG ids
             - tax2level: mapper between taxa and their distance from roottax
            """
            # bookeepers for HOGs
            curr_oma_hog = fam_tab_sort[0]["ID"]
            curr_oma_roothog = None
            curr_oma_roothog_ok = False
            curr_oma_taxa = []
            curr_oma_comps = []
            curr_oma_sizes = []

            for r in fam_tab_sort:
                oma_hog, oma_tax, hog_completeness, hog_gene_nr = r[1], r[2], r[3], r[5]

                # evaluation at new oma HOG
                if oma_hog != curr_oma_hog:
                    fam, curr_oma_roothog, curr_oma_roothog_ok = _process_oma_hog(
                        tax2level,
                        curr_oma_taxa,
                        curr_oma_hog,
                        curr_oma_roothog,
                        curr_oma_roothog_ok,
                        fam,
                        fam2hogs,
                        hog2oma_hog,
                        hog2tax,
                        hog2gene_nr,
                        hog2completeness,
                        roottax,
                        include_younger_fams,
                        curr_oma_sizes,
                        min_fam_size,
                        curr_oma_comps,
                        min_fam_completeness,
                        logic,
                    )

                    # reset for new HOG
                    curr_oma_taxa = []
                    curr_oma_sizes = []
                    curr_oma_comps = []
                    curr_oma_hog = oma_hog

                # track taxa, completeness and size of current oma HOG
                curr_oma_taxa.append(oma_tax)
                curr_oma_comps.append(hog_completeness)
                curr_oma_sizes.append(hog_gene_nr)

            # end
            fam, curr_oma_roothog, curr_oma_roothog_ok = _process_oma_hog(
                tax2level,
                curr_oma_taxa,
                curr_oma_hog,
                curr_oma_roothog,
                curr_oma_roothog_ok,
                fam,
                fam2hogs,
                hog2oma_hog,
                hog2tax,
                hog2gene_nr,
                hog2completeness,
                roottax,
                include_younger_fams,
                curr_oma_sizes,
                min_fam_size,
                curr_oma_comps,
                min_fam_completeness,
                logic,
            )

            return fam

        #
        tax2level = dict(zip(self._db_Taxonomy[:]["ID"], self._db_Taxonomy[:]["Level"]))
        hog_tab = h5file.root.HogLevel

        # containers
        fam2hogs = defaultdict(set)
        hog2oma_hog = dict()
        hog2tax = dict()
        hog2gene_nr = dict()
        hog2completeness = dict()

        # bookeepers for families
        fam = 0
        curr_fam = hog_tab[0]["Fam"]
        i = 0
        j = 0

        for hog_ent in tqdm(
            hog_tab, disable=is_progress_disabled(), mininterval=10, desc="Parsing HOGs"
        ):
            oma_fam = hog_ent["Fam"]

            if oma_fam != curr_fam:
                # load fam table and sort by HOG ids
                fam_tab = hog_tab[i:j]
                fam_tab_sort = np.sort(fam_tab, order="ID")

                # select and strip HOGs of one family
                fam = _process_oma_fam(
                    fam_tab_sort,
                    tax2level,
                    fam,
                    fam2hogs,
                    hog2oma_hog,
                    hog2tax,
                    hog2gene_nr,
                    hog2completeness,
                    self.root_taxon.encode("ascii"),
                    self.include_younger_fams,
                    self.min_fam_size,
                    self.min_fam_completeness,
                    self.logic,
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
            hog2gene_nr,
            hog2completeness,
            self.root_taxon.encode("ascii"),
            self.include_younger_fams,
            self.min_fam_size,
            self.min_fam_completeness,
            self.logic,
        )

        del hog_tab

        return fam2hogs, hog2oma_hog, hog2tax, hog2gene_nr, hog2completeness

    def select_and_filter_OMA_proteins(
        self, h5file, fam2hogs, hog2oma_hog, hog2tax, species, min_fam_size
    ):
        genome_tab = h5file.root.Genome[:]
        ent_tab = h5file.root.Protein.Entries

        # load entire sequence buffer into memory if we are computing for more than 100 species
        oma_seq_buffer = h5file.root.Protein.SequenceBuffer
        if len(species) > 100:
            oma_seq_buffer = oma_seq_buffer[:]

        # temporary mappers and booking for latter
        sp2sp_off = dict(
            zip(sorted(species), range(len(species)))
        )  # this is to sort the species table
        oma_hog2hog = dict(zip(hog2oma_hog.values(), hog2oma_hog.keys()))
        hog2protoffs = defaultdict(set)

        LOG.debug(" - select proteins from selected HOGs")

        seq_off = 0  # pointer to sequence in buffer
        prot_off = 0  # pointer to protein in protein table

        # store rows for species and protein tables and sequence buffer
        sp_rows = [()] * len(species)  # keep sorted
        seq_buffs = []

        prot_tab = self.db.create_table(
            "/",
            "Protein",
            self.ProteinTableFormat,
            filters=self._compr,
            expectedrows=25e6,
        )

        for r in tqdm(
            genome_tab,
            disable=is_progress_disabled(),
            mininterval=1,
            desc="Parsing genomes",
        ):
            sp = r["SciName"]
            sp_code = r["UniProtSpeciesCode"]

            # If no scientific name is set in the OMA DB, use the species code.
            sp = sp_code if sp not in species else sp

            # filter if species outside root-taxon
            if sp in species:
                sp_off = sp2sp_off[sp]

                # load sub entry table for species
                entry_off = r["EntryOff"]
                entry_num = r["TotEntries"]
                sp_ent_tab = ent_tab[entry_off : entry_off + entry_num]

                for rr in sp_ent_tab:
                    oma_hog = rr["OmaHOG"]

                    # select protein if member of selected OMA HOG
                    if oma_hog not in oma_hog2hog:
                        continue

                    # sequence
                    oma_seq_off = rr["SeqBufferOffset"]
                    seq_len = rr["SeqBufferLength"]
                    seq = oma_seq_buffer[oma_seq_off : oma_seq_off + seq_len]
                    seq_buffs.append(seq)

                    # store protein row
                    oma_id = "{}{:05d}".format(
                        sp_code.decode("ascii"), rr["EntryNr"] - entry_off
                    )
                    assert (
                        len(oma_id) < PROT_ID_LEN
                    ), "Please check PROT_ID_LEN as {} is longer than {} chars.".format(
                        oma_id, PROT_ID_LEN
                    )

                    prot_tab.append([(oma_id.encode("ascii"), sp_off, 0, seq_len)])

                    # track hog and family
                    hog = oma_hog2hog[oma_hog]
                    hog2protoffs[hog].add(prot_off)

                    # update offset of protein sequence in buffer and of protein raw in table
                    seq_off += seq_len
                    prot_off += 1

                # store species info
                sp_rows[sp2sp_off[sp]] = (sp, 0)

        # fill species and protein tables
        sp_tab = self.db.create_table(
            "/", "Species", self.SpeciesTableFormat, filters=self._compr
        )
        sp_tab.append(sp_rows)
        sp_tab.flush()

        prot_tab.flush()

        # cleanup, ensure gc
        del genome_tab, ent_tab, oma_seq_buffer

        seq_buff = np.concatenate(seq_buffs)
        return (fam2hogs, hog2protoffs, hog2tax, hog2oma_hog, seq_buff)

    def add_taxid_col(self, h5file):
        """
        Add the NCBI taxon id from OMA hdf5.
        """
        oma_tax_tab = h5file.root.Taxonomy[:]
        oma_sp_tab = h5file.root.Genome[:]

        # unsure why there was this check. for jul23 we use -ve for GTDB
        # quick check that the values are positives
        # if (oma_tax_tab["NCBITaxonId"][0] >= 0) and (oma_sp_tab["NCBITaxonId"][0] >= 0):
        taxid_column = []
        for tax_name in self._db_Taxonomy.col("ID"):
            if tax_name == b"LUCA":
                taxid_column.append(0)  # -1)  # 0 is unused.
            else:
                try:
                    taxid_column.append(
                        oma_tax_tab["NCBITaxonId"][
                            np.argwhere(oma_tax_tab["Name"] == tax_name)[0][0]
                        ]
                    )
                # when the taxon name is a uniprot species code
                except IndexError:
                    taxid_column.append(
                        oma_sp_tab["NCBITaxonId"][
                            np.argwhere(oma_sp_tab["UniProtSpeciesCode"] == tax_name)[
                                0
                            ][0]
                        ]
                    )
        self._db_Taxonomy.modify_column(colname="TaxID", column=taxid_column)
