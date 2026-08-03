from __future__ import annotations

import numpy as np
import tables

from omamer.database import Database, DatabaseFromOMA
from omamer.index import Index

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
    selector._db_Taxonomy = taxonomy
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


class TinyIndexDatabase:
    """Minimal database adapter needed by the sequence index builder."""

    def __init__(self, path):
        self.mode = "w"
        self._compr = tables.Filters(complevel=0)
        self.db = tables.open_file(path, mode="w", filters=self._compr)
        self._create_tables()

    def _create_tables(self):
        protein = self.db.create_table(
            "/",
            "Protein",
            {
                "SpeOff": tables.Int32Col(pos=0),
                "HOGoff": tables.UInt32Col(pos=1),
            },
        )
        protein.append([(0, 1), (1, 0), (0, 2)])
        protein.flush()

        hog = self.db.create_table(
            "/",
            "HOG",
            {
                "FamOff": tables.UInt32Col(pos=0),
                "ParentOff": tables.Int32Col(pos=1),
            },
        )
        hog.append([(0, -1), (0, 0), (1, -1)])
        hog.flush()

        family = self.db.create_table(
            "/",
            "Family",
            {
                "LevelOff": tables.UInt32Col(pos=0),
                "LevelNum": tables.UInt32Col(pos=1),
            },
        )
        family.append([(0, 1), (3, 0)])
        family.flush()

        species = self.db.create_table(
            "/", "Species", {"ID": tables.StringCol(8)}
        )
        species.append([(b"sp0",), (b"sp1",)])
        species.flush()

        self.db.create_carray(
            "/",
            "LevelOffsets",
            obj=np.asarray([0, 1, 2, 2, 3], dtype=np.uint32),
        )

        self._db_Protein = protein
        self._db_HOG = hog
        self._db_Family = family
        self._db_Species = species
        self._db_LevelOffsets = self.db.root.LevelOffsets

    def close(self):
        self.db.close()


def test_tiny_sequence_index(tmp_path):
    db = TinyIndexDatabase(tmp_path / "tiny-index.h5")
    try:
        index = Index(db, k=2, reduced_alphabet=False, hidden_taxa=())
        sequence_buffer = np.frombuffer(b"AAAA AAAC AAAG ", dtype="S1")
        suffix_array = index._build_suffixarray(
            index.alphabet.translate(sequence_buffer),
            len(db._db_Protein),
        )
        index._build_kmer_table(sequence_buffer, suffix_array)

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
