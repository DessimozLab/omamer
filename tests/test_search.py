"""
End-to-end checks of ``omamer search`` against a small published database.

Set ``OMAMER_SEARCH_DB`` to the All.May2026 ``Homininae.h5``
(https://omabrowser.org/All.May2026/Homininae.h5) to run them. The queries in
``data/homininae_queries.fa`` are 200 proteins sampled from that database and
10 bacterial proteins.

The search output is compared with the golden files in ``data/search_golden``.
They belong to one database; after moving to another release, regenerate them
with ``OMAMER_UPDATE_SEARCH_GOLDEN=1`` and review the differences.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import pytest

from omamer.database import Database
from omamer.main import main
from omamer.results_reader import results_reader

DATA = Path(__file__).parent / "data"
QUERIES = DATA / "homininae_queries.fa"
GOLDEN = DATA / "search_golden"

# header lines that change from run to run, or with every release
VOLATILE_HEADERS = ("!omamer-version:", "!date-run:", "!db-path:")
DATABASE_HASH_HEADER = "!db-info-database_hash:"

SEARCH_OPTIONS = {
    "default": [],
    "top3": ["-n", "3"],
    "family_only": ["--family_only"],
    "extant_genes": ["--include_extant_genes"],
    "threads": ["-t", "2"],
    # 210 queries in 5 chunks, the last one smaller than the others
    "chunks": ["-c", "50"],
}
# searches whose output must be identical to another golden file
SAME_GOLDEN = {"threads": "default", "chunks": "default"}


@pytest.fixture(scope="module")
def database_path():
    path = os.environ.get("OMAMER_SEARCH_DB")
    if not path:
        pytest.skip("set OMAMER_SEARCH_DB to the All.May2026 Homininae.h5")
    path = Path(path)
    if not path.is_file():
        pytest.skip(f"database is not available: {path}")
    return path


@pytest.fixture(scope="module")
def database_ids(database_path):
    """Return the HOG ID of each protein, and the IDs of the family root HOGs."""
    db = Database(str(database_path))
    try:
        protein_hogs = {
            db.get_prot_id(offset): db.get_hog_id(int(hog_offset))
            for offset, hog_offset in enumerate(db._db_Protein.col("HOGoff"))
        }
        family_hogs = {
            db.get_hog_id(int(hog_offset))
            for hog_offset in db._db_Family.col("HOGoff")
        }
        return protein_hogs, family_hogs
    finally:
        db.close()


@pytest.fixture(scope="module")
def search_output(database_path, tmp_path_factory):
    """Run each search once per module and return its output lines."""
    outputs = {}

    def run(name):
        if name not in outputs:
            out = tmp_path_factory.mktemp("search") / f"{name}.tsv"
            argv = [
                "omamer",
                "search",
                "--db",
                str(database_path),
                "--query",
                str(QUERIES),
                "--out",
                str(out),
                "--silent",
                *SEARCH_OPTIONS[name],
            ]
            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setattr(sys, "argv", argv)
                main()
            outputs[name] = [
                line
                for line in out.read_text().splitlines()
                if not line.startswith(VOLATILE_HEADERS)
            ]
        return outputs[name]

    return run


def _rows(lines):
    return list(
        csv.DictReader(
            (line for line in lines if not line.startswith("!")), delimiter="\t"
        )
    )


def _query_ids():
    return [
        line[1:].split()[0]
        for line in QUERIES.read_text().splitlines()
        if line.startswith(">")
    ]


def _in_hog(hog_id, ancestor_id):
    return (hog_id + ".").startswith(ancestor_id + ".")


def _header_value(lines, prefix):
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


@pytest.mark.parametrize("name", SEARCH_OPTIONS)
def test_search_matches_golden(name, search_output):
    lines = search_output(name)
    golden = GOLDEN / f"{SAME_GOLDEN.get(name, name)}.tsv"
    if os.environ.get("OMAMER_UPDATE_SEARCH_GOLDEN") == "1" and name not in SAME_GOLDEN:
        GOLDEN.mkdir(exist_ok=True)
        golden.write_text("\n".join(lines) + "\n")
        pytest.skip(f"updated {golden}")
    assert golden.is_file(), (
        f"missing {golden}; create it with OMAMER_UPDATE_SEARCH_GOLDEN=1"
    )
    expected = golden.read_text().splitlines()

    expected_hash = _header_value(expected, DATABASE_HASH_HEADER)
    database_hash = _header_value(lines, DATABASE_HASH_HEADER)
    if expected_hash != database_hash:
        pytest.skip(
            f"golden files belong to database {expected_hash}, not {database_hash}; "
            "regenerate them with OMAMER_UPDATE_SEARCH_GOLDEN=1"
        )
    assert len(lines) == len(expected)
    differences = [
        (nr, line, expected_line)
        for nr, (line, expected_line) in enumerate(zip(lines, expected), 1)
        if not _same_line(line, expected_line)
    ]
    assert not differences, "\n".join(
        f"line {nr}:\n  got      {line!r}\n  expected {expected_line!r}"
        for nr, line, expected_line in differences[:10]
    )


# floating point results differ in the last digits between platforms
FLOAT_REL_TOL = 1e-8


def _same_field(value, expected):
    if value == expected:
        return True
    try:
        return math.isclose(float(value), float(expected), rel_tol=FLOAT_REL_TOL)
    except ValueError:
        return False


def _same_line(line, expected):
    if line.startswith("!") or expected.startswith("!"):
        return line == expected
    fields, expected_fields = line.split("\t"), expected.split("\t")
    return len(fields) == len(expected_fields) and all(
        map(_same_field, fields, expected_fields)
    )


def test_search_places_database_proteins(search_output, database_ids):
    protein_hogs, _ = database_ids
    rows = _rows(search_output("default"))
    assert [row["qseqid"] for row in rows] == _query_ids()

    placed = misplaced = 0
    for row in rows:
        hog_id = protein_hogs.get(row["qseqid"])
        if hog_id is None or row["hogid"] == "N/A":
            continue
        placed += 1
        # queries come from the database, so they belong to the predicted HOG
        if not _in_hog(hog_id, row["hogid"]):
            misplaced += 1

    database_queries = sum(query in protein_hogs for query in _query_ids())
    assert placed >= 0.9 * database_queries
    assert misplaced <= 0.02 * placed


def test_search_family_only(search_output, database_ids):
    _, family_hogs = database_ids
    for row in _rows(search_output("family_only")):
        if row["hogid"] != "N/A":
            assert row["hogid"] in family_hogs, row


def test_search_extant_genes(search_output, database_ids):
    protein_hogs, _ = database_ids
    for row in _rows(search_output("extant_genes")):
        hog_id = protein_hogs.get(row["qseqid"])
        if hog_id is None or row["hogid"] == "N/A":
            continue
        genes = row["subfamily_geneset"].split(",")
        assert all(_in_hog(protein_hogs[gene], row["hogid"]) for gene in genes)
        if _in_hog(hog_id, row["hogid"]):
            assert row["qseqid"] in genes


def test_search_top_n_families(search_output):
    top3 = _rows(search_output("top3"))
    counts = {}
    for row in top3:
        counts[row["qseqid"]] = counts.get(row["qseqid"], 0) + 1
    assert set(counts) == set(_query_ids())
    assert max(counts.values()) <= 3

    # the best family is the same whether or not more are requested
    first = {}
    for row in top3:
        first.setdefault(row["qseqid"], row)
    for row in _rows(search_output("default")):
        assert first[row["qseqid"]] == row


@pytest.mark.parametrize("name", ["default", "top3", "extant_genes"])
def test_results_reader(name):
    # needs no database, the golden files are search output
    golden = GOLDEN / f"{name}.tsv"
    lines = golden.read_text().splitlines()
    df, metadata = results_reader(str(golden))

    expected_metadata = {}
    for line in lines:
        if not line.startswith("!"):
            break
        key, value = line[1:].split(": ", 1)
        expected_metadata[key] = value
    assert metadata == expected_metadata

    rows = _rows(lines)
    assert list(df.columns) == list(rows[0])
    assert list(df["qseqid"]) == [row["qseqid"] for row in rows]
    # unplaced queries are "N/A", which pandas reads as missing
    assert list(df["hogid"].isna()) == [row["hogid"] == "N/A" for row in rows]
    for column in ("family_p", "family_count", "subfamily_score", "qseqlen"):
        assert df[column].dtype.kind in "if", column
