"""
Opt-in check that ``mkdb`` builds the same database from an OMA browser HDF5
file (``--oma_path``) and from the OrthoXML + proteomes it was converted from
(``--orthoxml --species_tree --sequences``).

Set ``OMAMER_MKDB_SOURCES_DIR`` to a directory containing

- ``OmaServer.h5`` and ``speciestree.nwk`` (the browser build input),
- ``FastOMA_HOGs.orthoxml`` (the orthology the HDF5 file was made from),
- ``proteomes/*.fa`` (one FASTA file per species, headers as in the OrthoXML).

``OMAMER_MKDB_SOURCES_ROOT`` sets the root taxon (default: the root of the
species tree). Both databases are built once per module; this takes a few
minutes for a dataset of ~30 species. CI downloads such a dataset from
Zenodo, see the ``mkdb-sources`` job in ``.github/workflows/test_run.yml``.

The two databases store different protein IDs (OMA IDs vs. the FASTA IDs) and
may number HOGs and families in a different order, so everything is compared
via HOG IDs and the canonical protein IDs of the OMA HDF5 file. The NCBI taxon
ids are not compared, FastOMA does not write them to the OrthoXML.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import tables

from omamer.main import main

pytestmark = [
    pytest.mark.large_db,
    pytest.mark.skipif(
        not os.environ.get("OMAMER_MKDB_SOURCES_DIR"),
        reason="set OMAMER_MKDB_SOURCES_DIR to compare OMA and OrthoXML builds",
    ),
]

MAX_REPORTED = 10


@pytest.fixture(scope="module")
def source_dir():
    path = Path(os.environ["OMAMER_MKDB_SOURCES_DIR"])
    for name in ("OmaServer.h5", "speciestree.nwk", "FastOMA_HOGs.orthoxml"):
        if not (path / name).is_file():
            pytest.skip(f"missing input: {path / name}")
    if not list((path / "proteomes").glob("*.fa")):
        pytest.skip(f"no proteomes in {path / 'proteomes'}")
    return path


def _mkdb(db_path, *options):
    argv = ["omamer", "mkdb", "--db", str(db_path), "--log_level", "warning"]
    root_taxon = os.environ.get("OMAMER_MKDB_SOURCES_ROOT")
    if root_taxon:
        argv += ["--root_taxon", root_taxon]
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys, "argv", argv + list(options))
        main()


@pytest.fixture(scope="module")
def databases(source_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("mkdb_sources")
    oma_db = out / "oma.h5"
    oxml_db = out / "orthoxml.h5"
    _mkdb(oma_db, "--oma_path", str(source_dir))
    _mkdb(
        oxml_db,
        "--orthoxml",
        str(source_dir / "FastOMA_HOGs.orthoxml"),
        "--species_tree",
        str(source_dir / "speciestree.nwk"),
        "--sequences",
        *sorted(map(str, (source_dir / "proteomes").glob("*.fa"))),
    )
    with tables.open_file(oma_db) as oma, tables.open_file(oxml_db) as oxml:
        yield oma, oxml


def _buffered_ids(database, table_name, buffer_name):
    rows = database.get_node(table_name)[:]
    buffer = database.get_node(buffer_name)[:].tobytes()
    return [
        buffer[int(off) : int(off + n)].decode()
        for off, n in zip(rows["IDBufferOff"], rows["IDLen"])
    ]


def _hog_ids(database):
    return _buffered_ids(database, "/HOG", "/HOGIDBuffer")


def _differences(oma, oxml, what):
    """Compare two dicts and describe a readable sample of the differences."""
    only_oma = sorted(oma.keys() - oxml.keys())
    only_oxml = sorted(oxml.keys() - oma.keys())
    differ = sorted(k for k in oma.keys() & oxml.keys() if oma[k] != oxml[k])
    problems = []
    if only_oma:
        problems.append(f"{len(only_oma)} only in OMA build, e.g. {only_oma[:MAX_REPORTED]}")
    if only_oxml:
        problems.append(
            f"{len(only_oxml)} only in OrthoXML build, e.g. {only_oxml[:MAX_REPORTED]}"
        )
    if differ:
        sample = [(k, oma[k], oxml[k]) for k in differ[:MAX_REPORTED]]
        problems.append(f"{len(differ)} differ (key, OMA, OrthoXML), e.g. {sample}")
    return [f"{what}: {problem}" for problem in problems]


def _assert_same_mapping(oma, oxml, what):
    problems = _differences(oma, oxml, what)
    assert not problems, "\n".join(problems)


def _hog_properties(database):
    """HOG ID -> properties that do not depend on the HOG and family order."""
    ids = _hog_ids(database)
    hogs = database.root.HOG[:]
    taxa = database.root.Taxonomy.col("ID")
    hog_taxa = database.root.HOGtaxa[:]
    result = {}
    for i, hog_id in enumerate(ids):
        row = hogs[i]
        parent = int(row["ParentOff"])
        start = int(row["HOGtaxaOff"])
        result[hog_id] = {
            "parent": ids[parent] if parent >= 0 else None,
            "taxon": taxa[row["TaxOff"]].decode(),
            "members": int(row["NrMemberGenes"]),
            "completeness": round(float(row["CompletenessScore"]), 3),
            "median_seqlen": int(row["MedianSeqLen"]),
            "taxa": frozenset(
                t.decode()
                for t in taxa[hog_taxa[start : start + int(row["HOGtaxaNum"])]]
            ),
        }
    return result


def _oma_to_canonical_protein_ids(oma_h5):
    """Protein IDs of the --oma_path build (e.g. HUMAN00001) -> FASTA IDs."""
    with tables.open_file(oma_h5) as h5:
        genomes = h5.root.Genome[:]
        entries = h5.root.Protein.Entries.read(field="CanonicalId")
    mapping = {}
    for genome in genomes:
        code = genome["UniProtSpeciesCode"].decode()
        off = int(genome["EntryOff"])
        for nr in range(1, int(genome["TotEntries"]) + 1):
            mapping[f"{code}{nr:05d}"] = entries[off + nr - 1].decode()
    return mapping


def _proteins(database, protein_ids=None):
    """Protein ID -> (species, HOG ID, sequence length)."""
    ids = _buffered_ids(database, "/Protein", "/ProteinIDBuffer")
    if protein_ids is not None:
        ids = [protein_ids[i] for i in ids]
    hog_ids = _hog_ids(database)
    species = database.root.Species.col("ID")
    proteins = database.root.Protein[:]
    return {
        prot_id: (species[row["SpeOff"]].decode(), hog_ids[row["HOGoff"]], int(row["SeqLen"]))
        for prot_id, row in zip(ids, proteins)
    }


def test_same_build_parameters(databases):
    oma, oxml = databases
    for attribute in (
        "root_level",
        "min_fam_size",
        "min_fam_completeness",
        "filter_logic",
        "include_younger_fams",
    ):
        assert oma.root._v_attrs[attribute] == oxml.root._v_attrs[attribute], attribute
    for attribute in ("k", "alphabet_n"):
        assert oma.root.Index._v_attrs[attribute] == oxml.root.Index._v_attrs[attribute]


@pytest.mark.parametrize("node", ["/Taxonomy", "/Species", "/ChildrenTax"])
def test_same_taxonomy(databases, node):
    oma, oxml = databases
    oma_tab, oxml_tab = oma.get_node(node)[:], oxml.get_node(node)[:]
    assert oma_tab.dtype == oxml_tab.dtype
    if oma_tab.dtype.names is None:
        np.testing.assert_array_equal(oma_tab, oxml_tab)
        return
    for column in oma_tab.dtype.names:
        if column == "TaxID":
            # NCBI taxon ids are taken from the OMA HDF5 file and from the
            # OrthoXML species (FastOMA writes none). Other taxa get arbitrary
            # negative ids in both builds.
            continue
        np.testing.assert_array_equal(
            oma_tab[column], oxml_tab[column], err_msg=f"{node}/{column}"
        )


def test_same_families(databases):
    def families(database):
        ids = _hog_ids(database)
        taxa = database.root.Taxonomy.col("ID")
        return {
            ids[row["HOGoff"]]: (taxa[row["TaxOff"]].decode(), int(row["HOGnum"]))
            for row in database.root.Family[:]
        }

    _assert_same_mapping(*map(families, databases), "families (root HOG -> taxon, nr of HOGs)")


def test_same_hogs(databases):
    oma, oxml = map(_hog_properties, databases)
    problems = _differences(dict.fromkeys(oma), dict.fromkeys(oxml), "HOG IDs")
    for field in ("parent", "taxon", "members", "completeness", "median_seqlen", "taxa"):
        problems += _differences(
            {h: p[field] for h, p in oma.items() if h in oxml},
            {h: p[field] for h, p in oxml.items() if h in oma},
            f"HOG {field}",
        )
    assert not problems, "\n".join(problems)


def test_same_proteins(databases, source_dir):
    oma, oxml = databases
    _assert_same_mapping(
        _proteins(oma, _oma_to_canonical_protein_ids(source_dir / "OmaServer.h5")),
        _proteins(oxml),
        "proteins (species, HOG, length)",
    )


def test_same_kmer_table(databases):
    oma, oxml = databases
    oma_index, oxml_index = oma.root.Index, oxml.root.Index
    # number of HOGs per k-mer
    np.testing.assert_array_equal(oma_index.TableIndex[:], oxml_index.TableIndex[:])

    # the HOGs of each k-mer, compared via their IDs
    table_index = oma_index.TableIndex[:].astype(np.int64)
    kmer_of_entry = np.repeat(np.arange(len(table_index) - 1), np.diff(table_index))
    universe = sorted(set(_hog_ids(oma)) | set(_hog_ids(oxml)))
    rank = {hog_id: i for i, hog_id in enumerate(universe)}

    def postings(database, index):
        hog_rank = np.asarray([rank[h] for h in _hog_ids(database)], dtype=np.int64)
        entries = hog_rank[index.TableBuffer[:]]
        return entries[np.lexsort((entries, kmer_of_entry))]

    oma_postings = postings(oma, oma_index)
    oxml_postings = postings(oxml, oxml_index)
    differ = np.flatnonzero(oma_postings != oxml_postings)
    assert len(differ) == 0, (
        f"{len(differ)} of {len(oma_postings)} k-mer postings differ, e.g. "
        + str(
            [
                (int(kmer_of_entry[i]), universe[oma_postings[i]], universe[oxml_postings[i]])
                for i in differ[:MAX_REPORTED]
            ]
        )
    )


def test_same_probabilities(databases):
    oma, oxml = databases

    def hog_probability(database):
        return dict(zip(_hog_ids(database), database.root.Index.HOGProbability[:].round(6)))

    def family_probability(database):
        ids = _hog_ids(database)
        roots = [ids[off] for off in database.root.Family.col("HOGoff")]
        return dict(zip(roots, database.root.Index.FamilyProbability[:].round(6)))

    _assert_same_mapping(*map(family_probability, databases), "family probability")
    _assert_same_mapping(*map(hog_probability, databases), "HOG probability")
