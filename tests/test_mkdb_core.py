from __future__ import annotations

import numpy as np
import pytest
import tables
from property_manager import set_property

from omamer.database import Database, DatabaseFromOMA
from omamer.hierarchy import get_root_leaf_offsets
from omamer.index import Index
from omamer.species_tree import SpeciesTree

HOG_LEVEL_DTYPE = np.dtype(
    [
        ("Fam", np.uint32),
        ("ID", "S32"),
        ("Level", "S32"),
        ("CompletenessScore", np.float64),
        ("NrMemberGenes", np.uint32),
    ]
)


def _hog_level(family, hog_id, taxon, completeness, members):
    return family, hog_id, taxon, completeness, members


class _StubHDF5File:
    """Stands in for the PyTables file behind the Database node properties."""

    def __init__(self, **nodes):
        self._nodes = {"/" + name: node for name, node in nodes.items()}

    def get_node(self, path):
        return self._nodes[path]

    def __contains__(self, path):
        return path in self._nodes


def _hog_selector(logic="OR"):
    """Build the minimal object required by select_and_strip_OMA_HOGs."""
    taxonomy = np.asarray(
        [
            (b"Metazoa", 0.0),
            (b"Primates", 1.0),
            (b"Homo sapiens", 2.0),
        ],
        dtype=[("ID", "S32"), ("Level", np.float64)],
    )
    selector = object.__new__(DatabaseFromOMA)
    selector.root_taxon = "Metazoa"
    selector.include_younger_fams = True
    selector.min_fam_size = 6
    selector.min_fam_completeness = 0.5
    selector.logic = logic
    selector.db = _StubHDF5File(Taxonomy=taxonomy)
    return selector


def test_select_mapping():
    """Tests that HOG selection preserves family filtering and subhog mapping"""
    hog_levels = np.asarray(
        [
            # Passing root family with two taxonomic observations and a child.
            _hog_level(10, b"HOG:0001", b"Metazoa", 0.9, 10),
            _hog_level(10, b"HOG:0001", b"Primates", 0.8, 8),
            _hog_level(10, b"HOG:0001.1a", b"Primates", 0.7, 4),
            # Fails both OR thresholds; its child must not leak into the DB.
            _hog_level(20, b"HOG:0002", b"Primates", 0.1, 2),
            _hog_level(20, b"HOG:0002.1a", b"Homo sapiens", 0.9, 1),
            # Passes only completeness and becomes the next internal family.
            _hog_level(30, b"HOG:0003", b"Primates", 0.8, 2),
            # Entirely outside the selected taxonomy.
            _hog_level(40, b"HOG:0004", b"Fungi", 1.0, 100),
        ],
        dtype=HOG_LEVEL_DTYPE,
    )

    (
        family_to_hogs,
        hog_to_oma_hog,
        hog_to_taxon,
        hog_to_members,
        hog_to_completeness,
    ) = _hog_selector().select_and_strip_OMA_HOGs(hog_levels)

    assert dict(family_to_hogs) == {
        1: {b"1", b"1.1a"},
        2: {b"2"},
    }
    assert hog_to_oma_hog == {
        b"1": b"HOG:0001",
        b"1.1a": b"HOG:0001.1a",
        b"2": b"HOG:0003",
    }
    assert hog_to_taxon == {
        b"1": b"Metazoa",
        b"1.1a": b"Primates",
        b"2": b"Primates",
    }
    assert hog_to_members == {b"1": 10, b"1.1a": 4, b"2": 2}
    assert hog_to_completeness == {b"1": 0.9, b"1.1a": 0.7, b"2": 0.8}


def test_logic_and():
    hog_levels = np.asarray(
        [
            _hog_level(10, b"HOG:0001", b"Metazoa", 0.9, 10),
            _hog_level(20, b"HOG:0002", b"Primates", 0.2, 10),
            _hog_level(30, b"HOG:0003", b"Primates", 0.9, 2),
        ],
        dtype=HOG_LEVEL_DTYPE,
    )

    family_to_hogs, hog_to_oma_hog, *_ = (
        _hog_selector(logic="AND").select_and_strip_OMA_HOGs(hog_levels)
    )

    assert dict(family_to_hogs) == {1: {b"1"}}
    assert hog_to_oma_hog == {b"1": b"HOG:0001"}


def test_construct_from_tiny_tree(tmp_path):
    tree_path = tmp_path / "species.nwk"
    tree_path.write_text("((spA:1,spB:1)Clade:1,spC:2)Root;\n")
    db = Database(tmp_path / "taxonomy.h5", root_taxon="Root", mode="w")
    try:
        taxon_offsets, species, species_below = db.initiate_tax_tab(str(tree_path))
        taxonomy = db.db.root.Taxonomy[:]
        children = db.db.root.ChildrenTax[:]

        assert set(species) == {b"spA", b"spB", b"spC"}
        assert species_below == {
            b"Root": 3,
            b"Clade": 2,
            b"spA": 1,
            b"spB": 1,
            b"spC": 1,
        }
        assert set(taxon_offsets) == set(species_below)

        by_id = {bytes(row["ID"]): row for row in taxonomy}
        assert int(by_id[b"Root"]["ParentOff"]) == -1
        assert int(by_id[b"Clade"]["ParentOff"]) == taxon_offsets[b"Root"]
        clade = by_id[b"Clade"]
        start = int(clade["ChildrenOff"])
        stop = start + int(clade["ChildrenNum"])
        assert {
            bytes(taxonomy[offset]["ID"])
            for offset in children[start:stop]
        } == {b"spA", b"spB"}
    finally:
        db.close()


def test_hog_and_family_tables(tmp_path):
    db = Database(tmp_path / "hogs.h5", root_taxon="Root", mode="w")
    try:
        hog_offsets = db.update_hog_and_fam_tabs(
            fam2hogs={
                1: [b"1", b"1.1a", b"1.1a.1b"],
                2: [b"2"],
            },
            hog2taxoff={b"1": 10, b"1.1a": 11, b"1.1a.1b": 12, b"2": 20},
            hog2protoffs={
                b"1": [0],
                b"1.1a": [1, 2],
                b"1.1a.1b": [],
                b"2": [3],
            },
            hog2oma_hog={
                b"1": b"HOG:0001",
                b"1.1a": b"HOG:0001.1a",
                b"1.1a.1b": b"HOG:0001.1a.1b",
                b"2": b"HOG:0002",
            },
            hog2gene_nr={b"1": 4, b"1.1a": 3, b"1.1a.1b": 1, b"2": 1},
            hog2completeness={
                b"1": 1.0,
                b"1.1a": 0.75,
                b"1.1a.1b": 0.5,
                b"2": 1.0,
            },
        )

        assert hog_offsets == {b"1": 0, b"1.1a": 1, b"1.1a.1b": 2, b"2": 3}
        assert [db.get_hog_id(i) for i in range(4)] == [
            "HOG:0001",
            "HOG:0001.1a",
            "HOG:0001.1a.1b",
            "HOG:0002",
        ]

        families = db._db_Family[:]
        np.testing.assert_array_equal(families["ID"], [1, 2])
        np.testing.assert_array_equal(families["TaxOff"], [10, 20])
        np.testing.assert_array_equal(families["HOGoff"], [0, 3])
        np.testing.assert_array_equal(families["HOGnum"], [3, 1])
        np.testing.assert_array_equal(families["LevelOff"], [0, 3])
        np.testing.assert_array_equal(families["LevelNum"], [3, 1])

        hogs = db._db_HOG[:]
        np.testing.assert_array_equal(hogs["FamOff"], [0, 0, 0, 1])
        np.testing.assert_array_equal(hogs["ParentOff"], [-1, 0, 1, -1])
        np.testing.assert_array_equal(hogs["ChildrenOff"], [0, 1, -1, -1])
        np.testing.assert_array_equal(hogs["ChildrenNum"], [1, 1, 0, 0])
        np.testing.assert_array_equal(hogs["ChildrenProtOff"], [0, 1, -1, 3])
        np.testing.assert_array_equal(hogs["ChildrenProtNum"], [1, 2, 0, 1])
        # HOG taxon lists are populated by the following mkdb stage. Their
        # unsigned offset and count fields must start from a valid empty state.
        np.testing.assert_array_equal(hogs["HOGtaxaOff"], [0, 0, 0, 0])
        np.testing.assert_array_equal(hogs["HOGtaxaNum"], [0, 0, 0, 0])

        np.testing.assert_array_equal(db._db_ChildrenHOG[:], [1, 2])
        np.testing.assert_array_equal(db._db_ChildrenProt[:], [0, 1, 2, 3])
        np.testing.assert_array_equal(db._db_LevelOffsets[:], [0, 1, 2, 3, 4, 4])
    finally:
        db.close()


@pytest.mark.parametrize("sep", [".", b"."])
def test_find_parent_hog_skips_missing_levels(sep):
    """A duplication nested in a duplication has no HOG for the middle level."""
    as_id = (lambda x: x) if isinstance(sep, str) else str.encode
    hogs = np.asarray(
        sorted(as_id(h) for h in ("1", "1.1a", "1.1b.1a", "1.1b.1b", "1.1b.1b.2a"))
    )

    def parent(hog_id):
        off = Database.find_parent_hog(as_id(hog_id), hogs)
        return None if off == -1 else hogs[off]

    assert parent("1") is None
    assert parent("1.1a") == as_id("1")
    # HOG 1.1b does not exist, the parent is the closest existing ancestor
    assert parent("1.1b.1a") == as_id("1")
    assert parent("1.1b.1b") == as_id("1")
    assert parent("1.1b.1b.2a") == as_id("1.1b.1b")


def test_hog_table_with_missing_intermediate_hog(tmp_path):
    db = Database(tmp_path / "hogs.h5", root_taxon="Root", mode="w")
    try:
        hogs = [b"1", b"1.1a", b"1.1b.1a", b"1.1b.1b", b"1.1b.1b.2a"]
        db.update_hog_and_fam_tabs(
            fam2hogs={1: hogs},
            hog2taxoff={h: 0 for h in hogs},
            hog2protoffs={h: [i] for i, h in enumerate(hogs)},
            hog2oma_hog={h: b"HOG:000" + h for h in hogs},
            hog2gene_nr={h: 1 for h in hogs},
            hog2completeness={h: 1.0 for h in hogs},
        )
        ids = [db.get_hog_id(i) for i in range(len(hogs))]
        parents = {
            ids[i]: ids[p] if p >= 0 else None
            for i, p in enumerate(db._db_HOG.col("ParentOff"))
        }
        assert parents == {
            "HOG:0001": None,
            "HOG:0001.1a": "HOG:0001",
            "HOG:0001.1b.1a": "HOG:0001",
            "HOG:0001.1b.1b": "HOG:0001",
            "HOG:0001.1b.1b.2a": "HOG:0001.1b.1b",
        }
    finally:
        db.close()


def test_root_leaf_offsets_detects_cycle():
    np.testing.assert_array_equal(
        get_root_leaf_offsets(2, np.asarray([-1, 0, 1], dtype=np.int32)), [0, 1, 2]
    )
    with pytest.raises(ValueError, match="cycle"):
        get_root_leaf_offsets(2, np.asarray([-1, 2, 1], dtype=np.int32))


class TinyIndexDatabase:
    """Minimal OMAmerDBLike adapter needed by the sequence index builder."""

    # (SpeOff, HOGoff), (FamOff, ParentOff), (LevelOff, LevelNum) rows and the
    # level offsets of the default three-protein, two-family database
    PROTEIN_ROWS = [(0, 1), (1, 0), (0, 2)]
    HOG_ROWS = [(0, -1), (0, 0), (1, -1)]
    FAMILY_ROWS = [(0, 1), (3, 0)]
    LEVEL_OFFSETS = [0, 1, 2, 2, 3]

    def __init__(
        self,
        path,
        protein_rows=None,
        hog_rows=None,
        family_rows=None,
        level_offsets=None,
    ):
        self._compr = tables.Filters(complevel=0)
        self.db = tables.open_file(path, mode="w", filters=self._compr)
        self._create_tables(
            self.PROTEIN_ROWS if protein_rows is None else protein_rows,
            self.HOG_ROWS if hog_rows is None else hog_rows,
            self.FAMILY_ROWS if family_rows is None else family_rows,
            self.LEVEL_OFFSETS if level_offsets is None else level_offsets,
        )

    def _create_tables(self, protein_rows, hog_rows, family_rows, level_offsets):
        protein = self.db.create_table(
            "/",
            "Protein",
            {
                "SpeOff": tables.Int32Col(pos=0),
                "HOGoff": tables.UInt32Col(pos=1),
            },
        )
        protein.append(protein_rows)
        protein.flush()

        hog = self.db.create_table(
            "/",
            "HOG",
            {
                "FamOff": tables.UInt32Col(pos=0),
                "ParentOff": tables.Int32Col(pos=1),
            },
        )
        hog.append(hog_rows)
        hog.flush()

        family = self.db.create_table(
            "/",
            "Family",
            {
                "LevelOff": tables.UInt32Col(pos=0),
                "LevelNum": tables.UInt32Col(pos=1),
            },
        )
        family.append(family_rows)
        family.flush()

        species = self.db.create_table(
            "/", "Species", {"ID": tables.StringCol(8)}
        )
        species.append([(b"sp0",), (b"sp1",)])
        species.flush()

        self.db.create_carray(
            "/",
            "LevelOffsets",
            obj=np.asarray(level_offsets, dtype=np.uint32),
        )

    # The subset of OMAmerDBLike that Index reads while building the k-mer table.

    @property
    def protein_table(self):
        return self.db.root.Protein

    @property
    def hog_table(self):
        return self.db.root.HOG

    @property
    def family_table(self):
        return self.db.root.Family

    @property
    def species_table(self):
        return self.db.root.Species

    @property
    def level_offset_carray(self):
        return self.db.root.LevelOffsets

    @property
    def compression_filters(self):
        return self._compr

    @property
    def access_mode(self):
        return "w"

    # What the refit and beta-binomial fitting code reads, as in Database.

    def __getattr__(self, attr):
        if attr.startswith("_db_"):
            return self.db.get_node("/" + "/".join(attr.split("_")[2:]))
        raise AttributeError(attr)

    def has_structure(self):
        return "/Index/SSTableIndex" in self.db

    def close(self):
        self.db.close()


def test_tiny_sequence_index(tmp_path):
    db = TinyIndexDatabase(tmp_path / "tiny-index.h5")
    try:
        index = Index(db, k=2, reduced_alphabet=False, hidden_taxa=())
        sequence_buffer = np.frombuffer(b"AAAA AAAC AAAG ", dtype="S1")
        # Without a 3Di buffer only the sequence half of the index is built.
        index.build_kmer_table(sequence_buffer, None)

        assert "/Index/SSTableIndex" not in db.db
        table_index = db.db.root.Index.TableIndex[:]
        table_buffer = db.db.root.Index.TableBuffer[:]
        nonempty_codes = np.flatnonzero(np.diff(table_index))

        np.testing.assert_array_equal(nonempty_codes, [0, 1, 5])
        np.testing.assert_array_equal(table_index[nonempty_codes], [0, 2, 3])
        np.testing.assert_array_equal(table_index[nonempty_codes + 1], [2, 3, 4])
        np.testing.assert_array_equal(table_buffer, [0, 2, 0, 2])
        np.testing.assert_allclose(
            db.db.root.Index.FamilyProbability[:],
            [0.75, 0.75],
        )
        # Family probabilities are independent, not a distribution over families.
        assert db.db.root.Index.FamilyProbability[:].sum() == 1.5
        np.testing.assert_allclose(
            db.db.root.Index.HOGProbability[:],
            [0.5, 0.0, 0.5],
        )
    finally:
        db.close()


def test_tiny_structure_index(tmp_path):
    """A 3Di buffer adds a structure half without disturbing the sequence half."""
    db = TinyIndexDatabase(tmp_path / "tiny-structure-index.h5")
    try:
        index = Index(db, k=2, reduced_alphabet=False, hidden_taxa=())
        sequence_buffer = np.frombuffer(b"AAAA AAAC AAAG ", dtype="S1")
        # The same proteins relabelled, so the two halves index disjoint k-mer
        # codes but must agree on every derived probability.
        structure_buffer = np.frombuffer(b"CCCC CCCD CCCE ", dtype="S1")
        index.build_kmer_table(sequence_buffer, structure_buffer)

        idx = db.db.root.Index
        assert "/Index/SSTableIndex" in db.db

        structure_index = idx.SSTableIndex[:]
        nonempty_codes = np.flatnonzero(np.diff(structure_index))
        # "CC", "CD" and "CE" over the 21-letter alphabet.
        np.testing.assert_array_equal(nonempty_codes, [22, 23, 24])
        np.testing.assert_array_equal(structure_index[nonempty_codes], [0, 2, 3])
        np.testing.assert_array_equal(structure_index[nonempty_codes + 1], [2, 3, 4])
        np.testing.assert_array_equal(idx.SSTableBuffer[:], [0, 2, 0, 2])

        np.testing.assert_allclose(idx.SSFamilyProbability[:], [0.75, 0.75])
        np.testing.assert_allclose(idx.SSHOGProbability[:], [0.5, 0.0, 0.5])
        np.testing.assert_allclose(idx.SSFamilyProbability[:], idx.FamilyProbability[:])
        np.testing.assert_allclose(idx.SSHOGProbability[:], idx.HOGProbability[:])
    finally:
        db.close()


def test_modalities_keep_independent_information_filters(tmp_path):
    db = TinyIndexDatabase(tmp_path / "filtered-index.h5")
    try:
        index = Index(db, k=2, kmer_percentage=50)
        index.build_kmer_table(
            np.frombuffer(b"AAAA AAAC AAAG ", dtype="S1"),
            np.frombuffer(b"CCCC CCCC CCCC ", dtype="S1"),
        )

        idx = db.db.root.Index
        assert index.kmer_max_df == idx._v_attrs.kmer_max_df == 1
        assert index.ss_kmer_max_df == idx._v_attrs.ss_kmer_max_df == 2
        np.testing.assert_array_equal(idx.TableBuffer[:], [0, 2, 0, 2])
        np.testing.assert_array_equal(idx.SSTableBuffer[:], [0, 2])
        np.testing.assert_allclose(idx.FamilyProbability[:], [0.5, 0.5])
        np.testing.assert_allclose(idx.SSFamilyProbability[:], [1.0, 1.0])
        np.testing.assert_allclose(idx.HOGProbability[:], [0.5, 0.0, 0.5])
        np.testing.assert_allclose(idx.SSHOGProbability[:], [0.5, 0.0, 0.5])
    finally:
        db.close()


def test_hidden_species_excluded_from_both_modalities(tmp_path):
    db = TinyIndexDatabase(tmp_path / "hidden-species-index.h5")
    try:
        index = Index(db, k=2)
        set_property(index, "sp_filter", np.asarray([True, False]))
        index.build_kmer_table(
            np.frombuffer(b"AAAA AAAC AAAG ", dtype="S1"),
            np.frombuffer(b"CCCC CCCD CCCE ", dtype="S1"),
        )

        idx = db.db.root.Index
        for prefix, expected_codes in (("", [0, 1]), ("SS", [22, 23])):
            table_index = getattr(idx, prefix + "TableIndex")[:]
            np.testing.assert_array_equal(
                np.flatnonzero(np.diff(table_index)), expected_codes
            )
            np.testing.assert_array_equal(getattr(idx, prefix + "TableBuffer")[:], [0, 0])
            np.testing.assert_allclose(
                getattr(idx, prefix + "FamilyProbability")[:], [1.0, 0.0]
            )
            np.testing.assert_allclose(
                getattr(idx, prefix + "HOGProbability")[:], [1.0, 0.0, 0.0]
            )
    finally:
        db.close()


def test_insert_missing_hog_levels():
    """OrthoXML HOGs get the implied levels that pyoma adds for the OMA browser."""
    from lxml import etree

    from omamer.orthoxml_parser import (
        HOG,
        NS,
        TaxonomyLevels,
        is_orthologGroup_node,
    )

    def og(level, *children):
        return (
            f'<orthologGroup><property name="TaxRange" value="{level}"/>'
            + "".join(children)
            + "</orthologGroup>"
        )

    def pg(*children):
        return "<paralogGroup>" + "".join(children) + "</paralogGroup>"

    def gene(i):
        return f'<geneRef id="{i}"/>'

    # a duplication in R, nested in it a second one in AB, and genes several
    # levels below their parent orthologGroup
    xml = og(
        "R",
        pg(pg(og("AB", gene(1), gene(2)), gene(3)), og("ABC", gene(4), gene(5))),
        gene(6),
    ).replace(
        "<orthologGroup>", f'<orthologGroup xmlns="{NS["OrthoXML"]}" id="1">', 1
    )
    species = {1: "A", 2: "B", 3: "A", 4: "A", 5: "C", 6: "D"}
    hog = HOG(etree.fromstring(xml))
    hog.insert_missing_levels(
        TaxonomyLevels(SpeciesTree.from_string("(((A,B)AB,C)ABC,D)R;").parents()), species.get
    )

    levels = sorted(
        (n.attrib["og"], hog.get_properties(n)["TaxRange"])
        for n in hog.struct.iter()
        if is_orthologGroup_node(n)
    )
    assert levels == [
        ("1", "D"),
        ("1", "R"),
        # the duplication in R has no orthologGroup for 1.1a in the OrthoXML
        ("1.1a", "ABC"),
        ("1.1a.1a", "A"),
        ("1.1a.1a", "AB"),
        ("1.1a.1a", "B"),
        ("1.1a.1b", "A"),
        ("1.1a.1b", "AB"),
        ("1.1b", "A"),
        ("1.1b", "AB"),
        ("1.1b", "ABC"),
        ("1.1b", "C"),
    ]
    # the genes keep their subHOGs
    assert {g.attrib["id"]: g.attrib["og"] for g in hog.geneRefs()} == {
        "1": "1.1a.1a",
        "2": "1.1a.1a",
        "3": "1.1a.1b",
        "4": "1.1b",
        "5": "1.1b",
        "6": "1",
    }

    # inserting again changes nothing
    before = etree.tostring(hog.struct)
    hog.insert_missing_levels(
        TaxonomyLevels(SpeciesTree.from_string("(((A,B)AB,C)ABC,D)R;").parents()), species.get
    )
    assert etree.tostring(hog.struct) == before


def test_reduced_alphabet_index_has_one_posting_per_family(tmp_path):
    """
    The suffix array is sorted on the reduced alphabet, where AS and AT are
    the same k-mer. Comparing the k-mers on the original letters split them
    into separate runs of the same code, so a family was listed once per run.
    """
    db = TinyIndexDatabase(tmp_path / "reduced.h5")
    try:
        index = Index(db, k=2, reduced_alphabet=True, hidden_taxa=())
        # proteins 0 and 1 are in family 0, protein 2 in family 1
        index.build_kmer_table(np.frombuffer(b"ASC ATA ASD ", dtype="S1"), None)

        table_index = db.db.root.Index.TableIndex[:]
        table_buffer = db.db.root.Index.TableBuffer[:]
        hog_families = db.hog_table.col("FamOff")
        alphabet = index.alphabet
        for code in np.flatnonzero(np.diff(table_index)):
            families = hog_families[table_buffer[table_index[code]:table_index[code + 1]]]
            assert len(families) == len(set(families)), code

        # AS == AT: one posting per family, families 0 and 1
        code = alphabet.DIGITS_AA_LOOKUP[ord("A")] * alphabet.n + alphabet.DIGITS_AA_LOOKUP[ord("S")]
        families = hog_families[table_buffer[table_index[code]:table_index[code + 1]]]
        assert sorted(families) == [0, 1]
        # family 0 has AA and AC, family 1 AA and AD: with each k-mer counted
        # once per family, the two families are symmetric
        np.testing.assert_allclose(db.db.root.Index.FamilyProbability[:], [0.75, 0.75])
    finally:
        db.close()
