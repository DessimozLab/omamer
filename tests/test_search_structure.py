"""
End-to-end searches with sequence and 3Di queries on a tiny database: three
species, two families, random 40-residue proteins and 3Di strings.

The 3Di index of the database is built with the sequence index builder from
the 3Di strings (a second database over the same OrthoXML file, whose index
nodes are copied under the 3Di names), so that these tests do not depend on
``mkdb --structures``.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
import tables

from omamer.database import Database
from omamer.main import main
from omamer.merge_search import MergeSearch

SPECIES = ["A", "B", "C"]
FAMILIES = 2
SEQ_LEN = 40
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SS_NODES = {
    "TableIndex": "SSTableIndex",
    "TableBuffer": "SSTableBuffer",
    "FamilyProbability": "SSFamilyProbability",
    "HOGProbability": "SSHOGProbability",
}


def _protein_id(fam, sp):
    return "P{}_{}".format(fam, sp)


def _write_fasta(path, records):
    with open(path, "w") as fp:
        for prot_id, seq in records:
            fp.write(">{}\n{}\n".format(prot_id, seq))
    return str(path)


def _run(argv):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys, "argv", ["omamer", *argv])
        main()


def _mkdb(db_path, path, fasta):
    _run([
        "mkdb", "--db", str(db_path), "--log_level", "warning",
        "--orthoxml", str(path / "hogs.orthoxml"),
        "--species_tree", str(path / "speciestree.nwk"),
        "--sequences", *map(str, fasta),
        "--min_fam_size", "1",
        "--k", "3",
    ])


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("structure-db")
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

    def random_seq():
        return "".join(rng.choice(list(AMINO_ACIDS), SEQ_LEN))

    seqs = {}
    structures = {}
    for sp in SPECIES:
        for fam in range(FAMILIES):
            prot_id = _protein_id(fam, sp)
            seqs[prot_id] = random_seq()
            structures[prot_id] = random_seq()
    _write_fasta(path / "seqs.fa", seqs.items())
    _write_fasta(path / "structures.fa", structures.items())

    # the sequence database, then its 3Di index from a database built on the
    # 3Di strings (same OrthoXML file, so the same HOG and protein tables)
    db_path = path / "db.h5"
    ss_path = path / "ss.h5"
    _mkdb(db_path, path, [path / "seqs.fa"])
    _mkdb(ss_path, path, [path / "structures.fa"])
    with tables.open_file(str(ss_path)) as ss, tables.open_file(
        str(db_path), "a"
    ) as db:
        np.testing.assert_array_equal(ss.root.HOG[:], db.root.HOG[:])
        for node, ss_node in SS_NODES.items():
            db.create_carray(
                "/Index", ss_node, obj=ss.get_node("/Index/" + node)[:]
            )

    with Database(str(db_path)) as db:
        # the HOG of each protein, the expected placement of its own sequence
        hogs = {
            db.get_prot_id(i): db.get_hog_id(int(h))
            for i, h in enumerate(db._db_Protein.col("HOGoff"))
        }
    return path, db_path, seqs, structures, hogs


def _search(db_path, out, *options):
    _run(["search", "--db", str(db_path), "--out", str(out), "--silent", *options])
    return pd.read_csv(out, sep="\t", comment="!", keep_default_na=False)


@pytest.mark.parametrize(
    "inputs, mode, modality",
    [
        ("seq", "auto", "seq"),
        ("ss", "auto", "ss"),
        ("both", "auto", "seq"),
        ("both", "seq", "seq"),
        ("both", "ss", "ss"),
        ("both", "sqs", "seq"),
        ("both", "sum", "seq+ss"),
        ("both", "hybrid", "seq"),
    ],
)
@pytest.mark.parametrize("family_only", [False, True])
def test_search_structure_db(dataset, tmp_path, inputs, mode, modality, family_only):
    _, db_path, seqs, structures, hogs = dataset
    ids = sorted(seqs)
    options = ["--search_mode", mode]
    if inputs in ("seq", "both"):
        options += ["--query", _write_fasta(tmp_path / "q.fa", [(x, seqs[x]) for x in ids])]
    if inputs in ("ss", "both"):
        options += [
            "--structure",
            _write_fasta(tmp_path / "s.fa", [(x, structures[x]) for x in ids]),
        ]
    if family_only:
        options.append("--family_only")
    df = _search(db_path, tmp_path / "out.tsv", *options)
    assert list(df["qseqid"]) == ids
    # every protein is placed into its family (its own HOG or an ancestor)
    for qseqid, hogid in zip(df["qseqid"], df["hogid"]):
        assert hogs[qseqid].startswith(hogid), (qseqid, hogid, hogs[qseqid])
        if family_only:
            assert "." not in hogid
    assert set(df["modality"]) == {modality}
    header = [
        line for line in (tmp_path / "out.tsv").read_text().splitlines()
        if line.startswith("!")
    ]
    resolved = {"seq": "seq", "ss": "ss", "both": "hybrid"}[inputs] if mode == "auto" else mode
    assert "!search-mode: {}".format(resolved) in header
    summed = resolved in ("sum", "hybrid")
    assert ("!ss-weight: 0.5" in header) == summed
    assert ("!summed-alpha: 1e-06" in header) == summed


def test_search_structure_fallbacks(dataset, tmp_path):
    """The 3Di places the queries whose sequence is not usable."""
    _, db_path, seqs, structures, hogs = dataset
    ids = sorted(seqs)
    # the sequence of the first protein is unrelated to the database
    query = [(x, seqs[x] if i else "W" * SEQ_LEN) for i, x in enumerate(ids)]
    # (with 2 families of random proteins, a 3Di score of 25.6 is halved by
    # the default --ss_weight 0.5 below the threshold of 13.8)
    options = [
        "--ss_weight", "1",
        "--query", _write_fasta(tmp_path / "q.fa", query),
        "--structure",
        _write_fasta(tmp_path / "s.fa", [(x, structures[x]) for x in ids]),
    ]
    for mode, modality in (("sqs", "ss"), ("hybrid", "seq+ss"), ("seq", "N/A")):
        df = _search(db_path, tmp_path / "out.tsv", "--search_mode", mode, *options)
        assert list(df["modality"]) == [modality] + ["seq"] * (len(ids) - 1)
        if modality != "N/A":
            assert hogs[ids[0]].startswith(df["hogid"][0])


def test_search_structure_requires_index(dataset, tmp_path):
    path, _, seqs, structures, _ = dataset
    # a database without 3Di index
    db_path = tmp_path / "seq.h5"
    _mkdb(db_path, path, [path / "seqs.fa"])
    query = _write_fasta(tmp_path / "q.fa", seqs.items())
    structure = _write_fasta(tmp_path / "s.fa", structures.items())
    df = _search(db_path, tmp_path / "out.tsv", "--query", query)
    assert set(df["modality"]) == {"seq"}
    # a 3Di input or a 3Di search mode is an error, not ignored
    for options in (
        ["--query", query, "--structure", structure],
        ["--structure", structure],
        ["--query", query, "--search_mode", "hybrid"],
        ["--query", query, "--structure", structure, "--search_mode", "seq"],
    ):
        with pytest.raises(RuntimeError, match="no 3Di index"):
            _search(db_path, tmp_path / "out.tsv", *options)
    with Database(str(db_path)) as db:
        with pytest.raises(ValueError, match="3Di index"):
            MergeSearch(db.ki).merge_search(
                seqs=[seqs[x] for x in sorted(seqs)], ids=sorted(seqs),
                struct_seqs=[structures[x] for x in sorted(seqs)],
            )


@pytest.mark.parametrize(
    "structure_ids, message",
    [
        ("fewer", "fewer records"),
        ("more", "more records"),
        ("other", "IDs must match"),
    ],
)
def test_search_query_and_structure_must_pair_up(
    dataset, tmp_path, structure_ids, message
):
    _, db_path, seqs, structures, _ = dataset
    ids = sorted(seqs)
    # in chunks of 3: a missing or extra chunk, or different ids in a chunk
    ss_ids = {
        "fewer": ids[:3],
        "more": ids + ["extra"],
        "other": ids[::-1],
    }[structure_ids]
    query = _write_fasta(tmp_path / "q.fa", [(x, seqs[x]) for x in ids])
    structure = _write_fasta(
        tmp_path / "s.fa", [(x, structures.get(x, structures[ids[0]])) for x in ss_ids]
    )
    with pytest.raises(RuntimeError, match=message):
        _search(
            db_path, tmp_path / "out.tsv",
            "--query", query, "--structure", structure, "--chunksize", "3",
        )
