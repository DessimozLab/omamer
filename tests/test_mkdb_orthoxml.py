"""
mkdb --orthoxml on a tiny hand-made dataset: three species, two families.

Checks the sequence-only build and that 3Di records are stored in the order
of the protein table, whatever their order in the structure FASTA files.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
import tables

from omamer.database import Database, DatabaseFromOrthoXML
from omamer.main import main

SPECIES = ["A", "B", "C"]
FAMILIES = 2
SEQ_LEN = 40
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _protein_id(fam, sp):
    return "P{}_{}".format(fam, sp)


def _random_seq(rng):
    return "".join(rng.choice(list(AMINO_ACIDS), SEQ_LEN))


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("orthoxml")
    rng = np.random.default_rng(3)

    genes = {}
    species_xml = []
    for i, sp in enumerate(SPECIES):
        rows = []
        for fam in range(FAMILIES):
            gene_id = 1 + i * FAMILIES + fam
            genes[(fam, sp)] = gene_id
            rows.append(
                '<gene id="{}" protId="{}"/>'.format(gene_id, _protein_id(fam, sp))
            )
        species_xml.append(
            '<species name="{}" NCBITaxId="{}"><database name="db" version="1">'
            "<genes>{}</genes></database></species>".format(sp, i + 1, "".join(rows))
        )
    groups = []
    for fam in range(FAMILIES):
        groups.append(
            '<orthologGroup id="{}"><property name="TaxRange" value="R"/>'
            '<orthologGroup><property name="TaxRange" value="AB"/>'
            '<geneRef id="{}"/><geneRef id="{}"/></orthologGroup>'
            '<geneRef id="{}"/></orthologGroup>'.format(
                fam + 1, genes[(fam, "A")], genes[(fam, "B")], genes[(fam, "C")]
            )
        )
    (path / "hogs.orthoxml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<orthoXML xmlns="http://orthoXML.org/2011/" origin="test" '
        'originVersion="1" version="0.3">{}<groups>{}</groups></orthoXML>'.format(
            "".join(species_xml), "".join(groups)
        )
    )
    (path / "speciestree.nwk").write_text("((A,B)AB,C)R;\n")

    seqs = {}
    structures = {}
    for sp in SPECIES:
        with open(path / "{}.fa".format(sp), "w") as fp:
            for fam in range(FAMILIES):
                prot_id = _protein_id(fam, sp)
                seqs[prot_id] = _random_seq(rng)
                # a 3Di string of the same length, distinguishable per protein
                structures[prot_id] = _random_seq(rng)
                fp.write(">{}\n{}\n".format(prot_id, seqs[prot_id]))

    # the structures in reverse order, split over two files differently from
    # the proteomes, plus a record of a protein that is not in the OrthoXML
    ids = sorted(structures, reverse=True)
    with open(path / "ss1.fa", "w") as fp:
        fp.write(">unrelated\nACDE\n")
        for prot_id in ids[:2]:
            fp.write(">{} some description\n{}\n".format(prot_id, structures[prot_id]))
    with open(path / "ss2.fa", "w") as fp:
        for prot_id in ids[2:]:
            fp.write(">{}\n{}\n".format(prot_id, structures[prot_id]))

    # all proteins in one file, in an order unrelated to the protein table
    with open(path / "shuffled.fa", "w") as fp:
        for prot_id in sorted(seqs, key=lambda x: x[::-1], reverse=True):
            fp.write(">{}\n{}\n".format(prot_id, seqs[prot_id]))

    return path, seqs, structures


def _mkdb(db_path, dataset, *options):
    path = dataset[0]
    argv = [
        "omamer", "mkdb", "--db", str(db_path), "--log_level", "warning",
        "--orthoxml", str(path / "hogs.orthoxml"),
        "--species_tree", str(path / "speciestree.nwk"),
        "--sequences", *[str(path / "{}.fa".format(sp)) for sp in SPECIES],
        "--min_fam_size", "1",
        "--k", "3",
        *options,
    ]
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys, "argv", argv)
        main()


def _records(buff):
    """Split a space-separated S1 buffer into its records."""
    return buff.tobytes().decode("ascii").split(" ")[:-1]


def test_sequence_only_build(dataset, tmp_path):
    db_path = tmp_path / "seq.h5"
    _mkdb(db_path, dataset)
    db = Database(str(db_path))
    try:
        assert not db.has_structure()
        assert len(db._db_Protein) == len(SPECIES) * FAMILIES
        assert len(db._db_Family) == FAMILIES
    finally:
        db.close()


def test_structure_build_aligns_3di_with_proteins(dataset, tmp_path):
    path, seqs, structures = dataset
    db_path = tmp_path / "ss.h5"
    _mkdb(db_path, dataset, "--structures", str(path / "ss1.fa"), str(path / "ss2.fa"))
    db = Database(str(db_path))
    try:
        assert db.has_structure()
        prot_ids = [db.get_prot_id(i) for i in range(len(db._db_Protein))]
        assert sorted(prot_ids) == sorted(seqs)
    finally:
        db.close()

    # the buffers that mkdb indexes: record i of both is protein i
    selected = list(prot_ids)
    ss_buff = DatabaseFromOrthoXML._load_selected_3di_fasta(
        [str(path / "ss1.fa"), str(path / "ss2.fa")], selected
    )
    assert _records(ss_buff) == [structures[x] for x in selected]


def test_missing_3di_is_an_error(dataset, tmp_path):
    path = dataset[0]
    with pytest.raises(ValueError, match="Did not find 3Di sequences for 2 of 6"):
        _mkdb(tmp_path / "missing.h5", dataset, "--structures", str(path / "ss2.fa"))


def test_duplicate_3di_is_an_error(dataset):
    path = dataset[0]
    with pytest.raises(ValueError, match="multiple 3Di sequences"):
        DatabaseFromOrthoXML._load_selected_3di_fasta(
            [str(path / "ss1.fa"), str(path / "ss1.fa")],
            sorted(dataset[2]),
        )


def test_structure_index_follows_protein_order(dataset, tmp_path):
    # with the protein sequences as "3Di" input, in a different order, the
    # structure half of the index must be identical to the sequence half
    path = dataset[0]
    db_path = tmp_path / "seq-as-ss.h5"
    _mkdb(db_path, dataset, "--structures", str(path / "shuffled.fa"))
    with tables.open_file(str(db_path)) as h5:
        idx = h5.root.Index
        np.testing.assert_array_equal(idx.SSTableIndex[:], idx.TableIndex[:])
        np.testing.assert_array_equal(idx.SSTableBuffer[:], idx.TableBuffer[:])
        np.testing.assert_allclose(idx.SSFamilyProbability[:], idx.FamilyProbability[:])
        np.testing.assert_allclose(idx.SSHOGProbability[:], idx.HOGProbability[:])


@pytest.fixture(scope="module")
def structure_db(dataset, tmp_path_factory):
    path = dataset[0]
    db_path = tmp_path_factory.mktemp("ss-db") / "ss.h5"
    _mkdb(db_path, dataset, "--structures", str(path / "ss1.fa"), str(path / "ss2.fa"))
    db = Database(str(db_path))
    try:
        # the HOG of each protein, the expected placement of its own sequence
        hogs = {
            db.get_prot_id(i): db.get_hog_id(int(h))
            for i, h in enumerate(db._db_Protein.col("HOGoff"))
        }
    finally:
        db.close()
    return db_path, hogs


def test_structure_build_defaults(structure_db):
    """mkdb defaults from the model audit: 3Di low-complexity mask (< 3 distinct
    letters) and 3Di k-mers capped at 1000 families; sequences unfiltered."""
    from omamer.kmer_mask import stored_min_distinct

    db_path, _ = structure_db
    with tables.open_file(str(db_path)) as h5:
        attrs = h5.root.Index._v_attrs
        assert int(attrs["ss_kmer_max_df"]) == 1000
        assert int(attrs["kmer_max_df"]) == 0
        assert stored_min_distinct(attrs, "ss") == 3
        assert stored_min_distinct(attrs, "seq") == 0


def test_ss_max_df_caps_3di_kmers_at_build(dataset, structure_db, tmp_path):
    path = dataset[0]
    db_path = tmp_path / "capped.h5"
    _mkdb(
        db_path, dataset, "--structures", str(path / "ss1.fa"), str(path / "ss2.fa"),
        "--ss_max_df", "1",
    )
    with tables.open_file(str(db_path)) as capped, tables.open_file(str(structure_db[0])) as full:
        assert int(capped.root.Index._v_attrs["ss_kmer_max_df"]) == 1
        # the same k-mer tables, but family probabilities without the shared k-mers
        np.testing.assert_array_equal(
            capped.root.Index.SSTableIndex[:], full.root.Index.SSTableIndex[:]
        )
        p_capped = capped.root.Index.SSFamilyProbability[:]
        p_full = full.root.Index.SSFamilyProbability[:]
        assert np.all(p_capped <= p_full + 1e-12)
        assert np.any(p_capped < p_full)
        # sequences are not capped
        np.testing.assert_array_equal(
            capped.root.Index.FamilyProbability[:], full.root.Index.FamilyProbability[:]
        )
