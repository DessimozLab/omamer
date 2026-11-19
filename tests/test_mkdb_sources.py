"""
Opt-in check that ``mkdb`` builds the same database from an OMA browser HDF5
file (``--oma_path``) and from the OrthoXML + proteomes it was converted from
(``--orthoxml --species_tree --sequences``).

Set ``OMAMER_MKDB_SOURCES_DIR`` to a directory containing

- ``OmaServer.h5`` and ``speciestree.nwk`` (the browser build input),
- ``FastOMA_HOGs.orthoxml`` (the orthology the HDF5 file was made from),
- ``proteomes/*.fa`` (one FASTA file per species, headers as in the OrthoXML).

``OMAMER_MKDB_SOURCES_ROOT`` sets the root taxon (default: the root of the
species tree). Both databases are built once per module, together with a
third one from the OMA browser input with non-default index parameters
(reduced alphabet, shorter k, a hidden clade). All databases are also
searched with proteins sampled from them, and the OrthoXML input is also
built without the newick tree, from the OrthoXML <taxonomy>. This takes
~17 minutes for a dataset of ~30 species. CI downloads such a dataset from
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
def database_paths(source_dir, tmp_path_factory):
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
    return oma_db, oxml_db


@pytest.fixture(scope="module")
def databases(database_paths):
    oma_db, oxml_db = database_paths
    with tables.open_file(oma_db) as oma, tables.open_file(oxml_db) as oxml:
        yield oma, oxml


@pytest.fixture(scope="module")
def taxonomy_build_path(source_dir, tmp_path_factory):
    """OrthoXML build without --species_tree, from the OrthoXML <taxonomy>."""
    from omamer.species_tree import SpeciesTree, SpeciesTreeError

    try:
        SpeciesTree.from_orthoxml(source_dir / "FastOMA_HOGs.orthoxml")
    except SpeciesTreeError:
        pytest.skip("the OrthoXML file has no <taxonomy> (OrthoXML < 0.5)")
    db_path = tmp_path_factory.mktemp("mkdb_taxonomy") / "orthoxml_taxonomy.h5"
    _mkdb(
        db_path,
        "--orthoxml",
        str(source_dir / "FastOMA_HOGs.orthoxml"),
        "--sequences",
        *sorted(map(str, (source_dir / "proteomes").glob("*.fa"))),
    )
    return db_path


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


def _children_sets(database):
    """Taxon offset -> set of child taxon offsets."""
    taxonomy = database.root.Taxonomy[:]
    children = database.root.ChildrenTax[:]
    return {
        i: frozenset(children[row["ChildrenOff"] : row["ChildrenOff"] + row["ChildrenNum"]])
        for i, row in enumerate(taxonomy)
        if row["ChildrenNum"] > 0
    }


def test_taxonomy_from_orthoxml_gives_same_database(database_paths, taxonomy_build_path):
    # the <taxonomy> of the OrthoXML file is the species tree of the dataset,
    # so the build without --species_tree must give the same database. Only
    # the order of the children of a taxon may differ between the two files.
    _, oxml_path = database_paths
    with tables.open_file(oxml_path) as oxml, tables.open_file(taxonomy_build_path) as taxonomy:
        nodes = {
            node._v_pathname
            for node in oxml.walk_nodes("/")
            if isinstance(node, tables.Leaf)
        }
        assert nodes == {
            node._v_pathname
            for node in taxonomy.walk_nodes("/")
            if isinstance(node, tables.Leaf)
        }
        for node in sorted(nodes - {"/ChildrenTax"}):
            np.testing.assert_array_equal(
                oxml.get_node(node)[:], taxonomy.get_node(node)[:], err_msg=node
            )
        assert _children_sets(oxml) == _children_sets(taxonomy)


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


# --- non-default index parameters -------------------------------------------
#
# A third database is built from the OMA browser input with a reduced
# alphabet, a shorter k and a hidden clade. Only the index may differ from the
# default build.

VARIANT_K = 5
REDUCED_ALPHABET_N = 13

# The k-mer table of --reduced_alphabet builds splits the suffixes of one
# reduced k-mer into several runs, because the runs are found on the original
# instead of the translated sequences. Such k-mers get duplicate and too
# specific HOG entries, and searches place most queries in the wrong family.
reduced_alphabet_bug = pytest.mark.xfail(
    strict=True,
    reason="--reduced_alphabet k-mer table splits runs of the same reduced k-mer",
)


def _subtrees(database):
    """Taxon offset -> offsets of the taxa in its subtree, including itself."""
    parents = database.root.Taxonomy.col("ParentOff")
    subtree = {i: {i} for i in range(len(parents))}
    for i in range(len(parents)):
        parent = parents[i]
        while parent >= 0:
            subtree[parent].add(i)
            parent = parents[parent]
    return subtree


@pytest.fixture(scope="module")
def hidden_taxon(databases):
    """The internal taxon with the most HOGs, among those with at most half of
    the species. Its HOGs can only be indexed from its own species."""
    oma, _ = databases
    taxonomy = oma.root.Taxonomy[:]
    subtree = _subtrees(oma)
    is_species = taxonomy["SpeOff"] >= 0
    hogs_per_taxon = np.bincount(oma.root.HOG.col("TaxOff"), minlength=len(taxonomy))
    candidates = [
        (int(hogs_per_taxon[list(subtree[i])].sum()), taxonomy["ID"][i].decode(), i)
        for i in range(len(taxonomy))
        if taxonomy["ParentOff"][i] >= 0
        and not is_species[i]
        and is_species[list(subtree[i])].sum() <= is_species.sum() // 2
    ]
    if not candidates:
        pytest.skip("species tree has no suitable internal taxon to hide")
    nr_hogs, name, offset = max(candidates)
    assert nr_hogs > 0
    species = {
        taxonomy["ID"][t].decode() for t in subtree[offset] if is_species[t]
    }
    return name, subtree[offset], species


def _hidden_hogs(database, hidden_taxon):
    """Offsets of the HOGs defined at a level within the hidden clade."""
    _, clade, _ = hidden_taxon
    return np.flatnonzero(np.isin(database.root.HOG.col("TaxOff"), list(clade)))


@pytest.fixture(scope="module")
def variant_path(source_dir, hidden_taxon, tmp_path_factory):
    out = tmp_path_factory.mktemp("mkdb_variant")
    hidden_file = out / "hidden_taxa.txt"
    hidden_file.write_text(hidden_taxon[0] + "\n")
    db_path = out / "variant.h5"
    _mkdb(
        db_path,
        "--oma_path",
        str(source_dir),
        "--reduced_alphabet",
        "--k",
        str(VARIANT_K),
        "--hidden_taxa",
        str(hidden_file),
    )
    return db_path


@pytest.fixture(scope="module")
def variant(variant_path):
    with tables.open_file(variant_path) as h5:
        yield h5


def _check_index(database, k, alphabet_n):
    index = database.root.Index
    assert index._v_attrs["k"] == k
    assert index._v_attrs["alphabet_n"] == alphabet_n

    table_index = index.TableIndex[:].astype(np.int64)
    table_buffer = index.TableBuffer[:].astype(np.int64)
    assert len(table_index) == alphabet_n**k + 1
    assert table_index[0] == 0
    assert np.all(np.diff(table_index) >= 0)
    assert table_index[-1] == len(table_buffer) > 0

    nr_hogs = len(database.root.HOG)
    assert table_buffer.max() < nr_hogs

    # a k-mer lists at most one HOG per family (the LCA of its occurrences)
    hog_families = database.root.HOG.col("FamOff").astype(np.int64)
    kmer_of_entry = np.repeat(np.arange(len(table_index) - 1), np.diff(table_index))
    pairs = kmer_of_entry * (int(hog_families.max()) + 1) + hog_families[table_buffer]
    assert len(np.unique(pairs)) == len(pairs)

    # the probability of a family is the fraction of the table entries that
    # belong to k-mers listing the family (not a distribution over families)
    kmer_sizes = np.diff(table_index)
    unique_pairs = np.unique(pairs)
    pair_kmers = unique_pairs // (int(hog_families.max()) + 1)
    pair_families = unique_pairs % (int(hog_families.max()) + 1)
    expected = np.bincount(
        pair_families,
        weights=kmer_sizes[pair_kmers],
        minlength=len(database.root.Family),
    ) / table_index[-1]
    family_probability = index.FamilyProbability[:]
    assert len(family_probability) == len(database.root.Family)
    np.testing.assert_allclose(family_probability, expected, rtol=1e-12)
    hog_probability = index.HOGProbability[:]
    assert len(hog_probability) == nr_hogs
    assert np.all((hog_probability >= 0) & (hog_probability <= 1))


def test_default_index_structure(databases):
    for database in databases:
        _check_index(database, k=6, alphabet_n=21)


@reduced_alphabet_bug
def test_variant_index_structure(variant, hidden_taxon):
    _check_index(variant, k=VARIANT_K, alphabet_n=REDUCED_ALPHABET_N)
    assert list(variant.root.Index._v_attrs["hidden_taxa"]) == [hidden_taxon[0]]


@pytest.mark.parametrize(
    "node", ["/Family", "/HOG", "/Protein", "/Species", "/Taxonomy", "/HOGtaxa"]
)
def test_variant_same_tables(databases, variant, node):
    # index parameters must not change the HOGs, families and proteins
    oma, _ = databases
    np.testing.assert_array_equal(oma.get_node(node)[:], variant.get_node(node)[:])


def test_hidden_taxon_not_indexed(databases, variant, hidden_taxon):
    oma, _ = databases
    hidden_hogs = _hidden_hogs(oma, hidden_taxon)
    # the default build indexes them, otherwise the check below proves nothing
    assert np.isin(hidden_hogs, oma.root.Index.TableBuffer[:]).any()
    indexed = np.intersect1d(hidden_hogs, variant.root.Index.TableBuffer[:])
    assert len(indexed) == 0, (
        f"{len(indexed)} HOGs of the hidden taxon {hidden_taxon[0]} are in the "
        f"k-mer table, e.g. {[_hog_ids(variant)[i] for i in indexed[:MAX_REPORTED]]}"
    )


# --- search with the freshly built databases --------------------------------

NR_QUERIES = 150
NR_HIDDEN_QUERIES = 50


@pytest.fixture(scope="module")
def queries(databases, source_dir, hidden_taxon, tmp_path_factory):
    """Write a FASTA file with proteins sampled evenly from the database, some
    of them from the hidden clade. Returns the file, protein ID -> HOG ID and
    the IDs of the queries from the hidden clade."""
    from Bio import SeqIO

    _, oxml = databases
    proteins = _proteins(oxml)
    _, _, hidden_species = hidden_taxon
    in_clade = sorted(p for p, (sp, _, _) in proteins.items() if sp in hidden_species)
    outside = sorted(p for p, (sp, _, _) in proteins.items() if sp not in hidden_species)

    def sample(ids, n):
        return set(ids[:: max(1, len(ids) // n)][:n])

    hidden_queries = sample(in_clade, NR_HIDDEN_QUERIES)
    selected = sample(outside, NR_QUERIES) | hidden_queries
    path = tmp_path_factory.mktemp("queries") / "queries.fa"
    written = []
    with open(path, "w") as out:
        for fasta in sorted((source_dir / "proteomes").glob("*.fa")):
            for record in SeqIO.parse(fasta, "fasta"):
                if record.id in selected:
                    out.write(f">{record.id}\n{record.seq}\n")
                    written.append(record.id)
    assert set(written) == selected
    return path, {p: proteins[p][1] for p in written}, hidden_queries


def _search(db_path, query, tmp_path_factory):
    """Search and return the result rows as dicts (without the header)."""
    out = tmp_path_factory.mktemp("search") / "out.tsv"
    argv = ["omamer", "search", "--db", str(db_path), "--query", str(query), "--out", str(out), "--silent"]
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys, "argv", argv)
        main()
    table = [line for line in out.read_text().splitlines() if not line.startswith("!")]
    columns = table[0].split("\t")
    return [dict(zip(columns, line.split("\t"))) for line in table[1:]]


def _in_hog(hog_id, ancestor_id):
    return (hog_id + ".").startswith(ancestor_id + ".")


@pytest.fixture(scope="module")
def fresh_search(database_paths, queries, tmp_path_factory):
    query, _, _ = queries
    return [_search(path, query, tmp_path_factory) for path in database_paths]


def test_fresh_search_places_database_proteins(fresh_search, queries):
    _, protein_hogs, _ = queries
    _, oxml_rows = fresh_search
    assert {row["qseqid"] for row in oxml_rows} == set(protein_hogs)
    placed = [row for row in oxml_rows if row["hogid"] != "N/A"]
    misplaced = [
        (row["qseqid"], protein_hogs[row["qseqid"]], row["hogid"])
        for row in placed
        if not _in_hog(protein_hogs[row["qseqid"]], row["hogid"])
    ]
    # queries come from the database, so they belong to the predicted HOG
    assert len(placed) >= 0.9 * len(oxml_rows)
    assert len(misplaced) <= 0.02 * len(placed), misplaced[:MAX_REPORTED]


def test_fresh_search_same_for_both_builds(fresh_search):
    # the queries carry the FASTA IDs, which both builds share, and the two
    # databases are equivalent, so the search results must be the same
    oma_rows, oxml_rows = fresh_search
    assert oma_rows == oxml_rows


@reduced_alphabet_bug
def test_hidden_taxon_search(variant_path, databases, queries, hidden_taxon, tmp_path_factory):
    query, protein_hogs, hidden_queries = queries
    rows = _search(variant_path, query, tmp_path_factory)
    assert {row["qseqid"] for row in rows} == set(protein_hogs)

    oma, _ = databases
    hog_ids = _hog_ids(oma)
    hidden_hog_ids = {hog_ids[i] for i in _hidden_hogs(oma, hidden_taxon)}
    # no k-mer leads to a HOG of the hidden clade, so nothing is placed there
    assert not [row for row in rows if row["hogid"] in hidden_hog_ids]

    # proteins outside the hidden clade are placed as well as with the default
    # index, allowing for the less specific reduced alphabet and shorter k.
    # Most proteins of the hidden clade cannot be placed: their families are
    # often only defined within the clade.
    others = [row for row in rows if row["qseqid"] not in hidden_queries]
    placed = [row for row in others if row["hogid"] != "N/A"]
    misplaced = [
        (row["qseqid"], protein_hogs[row["qseqid"]], row["hogid"])
        for row in placed
        if not _in_hog(protein_hogs[row["qseqid"]], row["hogid"])
    ]
    assert len(placed) >= 0.8 * len(others)
    assert len(misplaced) <= 0.05 * len(placed), misplaced[:MAX_REPORTED]
