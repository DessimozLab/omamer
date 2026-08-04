"""
Opt-in validation of complete mkdb output stored outside the repository.

Set ``OMAMER_RUN_LARGE_DB_TESTS=1`` and ``OMAMER_REFERENCE_DB_DIR`` to run
the reference-database checks. Set ``OMAMER_CANDIDATE_DB_DIR`` as well to
compare newly built databases with the trusted references.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tables

pytestmark = [
    pytest.mark.large_db,
    pytest.mark.skipif(
        os.environ.get("OMAMER_RUN_LARGE_DB_TESTS") != "1",
        reason="set OMAMER_RUN_LARGE_DB_TESTS=1 to inspect external databases",
    ),
]

DATABASE_NAMES = ("Primates.h5", "Metazoa.h5", "LUCA.h5")

CORE_NODES = (
    "/Family",
    "/HOG",
    "/HOGIDBuffer",
    "/Protein",
    "/Species",
    "/Taxonomy",
    "/ChildrenHOG",
    "/ChildrenProt",
    "/ChildrenTax",
    "/HOGtaxa",
    "/LevelOffsets",
    "/Index/TableIndex",
    "/Index/TableBuffer",
    "/Index/FamilyProbability",
    "/Index/HOGProbability",
)

STABLE_NODES = (
    "/Family",
    "/Species",
    "/Taxonomy",
    "/ChildrenTax",
    "/LevelOffsets",
    "/Index/TableIndex",
    "/Index/FamilyProbability",
)

ROOT_ATTRIBUTES = (
    "root_level",
    "source",
    "oma_version",
    "min_fam_size",
    "min_fam_completeness",
    "filter_logic",
    "include_younger_fams",
)
INDEX_ATTRIBUTES = ("k", "alphabet_n", "hidden_taxa")


def _database_path(directory_variable, name):
    directory = os.environ.get(directory_variable)
    if not directory:
        pytest.skip(f"set {directory_variable} to an external database directory")
    path = Path(directory) / name
    if not path.is_file():
        pytest.skip(f"database is not available: {path}")
    return path


def _sample_slices(length, chunk_size=65536):
    if length <= chunk_size:
        return [slice(0, length)]
    starts = (0, max(0, length // 2 - chunk_size // 2), length - chunk_size)
    return [slice(start, start + chunk_size) for start in sorted(set(starts))]


def _comparison_slices(length):
    if os.environ.get("OMAMER_DB_COMPARE", "sample") == "full":
        chunk_size = 1_000_000
        return [
            slice(start, min(length, start + chunk_size))
            for start in range(0, length, chunk_size)
        ]
    return _sample_slices(length)


def _comparison_indices(length):
    if os.environ.get("OMAMER_DB_COMPARE", "sample") == "full":
        return range(length)
    return np.linspace(0, length - 1, min(length, 1024), dtype=np.int64)


def _normalise_attribute(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _assert_node_equal(reference, candidate, path):
    reference_node = reference.get_node(path)
    candidate_node = candidate.get_node(path)
    assert reference_node.shape == candidate_node.shape, path
    assert reference_node.dtype == candidate_node.dtype, path
    for part in _comparison_slices(len(reference_node)):
        np.testing.assert_array_equal(
            reference_node[part],
            candidate_node[part],
            err_msg=f"database node differs: {path} at {part}",
        )


def _assert_buffer_offsets(offsets, counts, buffer_length, *, empty_offset=-1):
    assert np.all(counts >= 0)
    empty = counts == 0
    assert np.all(offsets[empty] == empty_offset)
    nonempty = ~empty
    assert np.all(offsets[nonempty] >= 0)
    assert np.all(offsets[nonempty] + counts[nonempty] <= buffer_length)


def _buffered_ids(database, table_name, buffer_name):
    rows = database.get_node(table_name)[:]
    buffer = database.get_node(buffer_name)[:].tobytes()
    return [
        buffer[int(row["IDBufferOff"]) : int(row["IDBufferOff"] + row["IDLen"])]
        for row in rows
    ]


def _protein_id_buffer(database):
    if "ID" in database.root.Protein.dtype.names:
        return None
    return database.root.ProteinIDBuffer[:].tobytes()


def _protein_ids(database, rows, id_buffer):
    if "ID" in rows.dtype.names:
        return rows["ID"]
    return np.asarray(
        [
            id_buffer[
                int(row["IDBufferOff"]) : int(row["IDBufferOff"] + row["IDLen"])
            ]
            for row in rows
        ]
    )


def _legacy_sequence_length_adjustment(database):
    # Databases with inline protein IDs stored the trailing sequence delimiter
    # in SeqLen. The buffered-ID schema stores the biological sequence length.
    return 1 if "ID" in database.root.Protein.dtype.names else 0


@pytest.mark.parametrize("name", DATABASE_NAMES)
def test_reference_db_relations(name):
    path = _database_path("OMAMER_REFERENCE_DB_DIR", name)

    with tables.open_file(path, mode="r") as database:
        missing = [node for node in CORE_NODES if node not in database]
        assert not missing, f"missing core nodes: {missing}"

        families = database.root.Family[:]
        hogs = database.root.HOG[:]
        proteins = database.root.Protein[:]
        species = database.root.Species[:]
        taxonomy = database.root.Taxonomy[:]

        family_starts = families["HOGoff"]
        family_sizes = families["HOGnum"]
        np.testing.assert_array_equal(
            family_starts,
            np.concatenate(([0], np.cumsum(family_sizes[:-1], dtype=np.uint64))),
        )
        assert int(family_starts[-1] + family_sizes[-1]) == len(hogs)
        np.testing.assert_array_equal(
            hogs["FamOff"],
            np.repeat(np.arange(len(families), dtype=np.uint32), family_sizes),
        )

        parents = hogs["ParentOff"]
        assert np.all((parents == -1) | ((parents >= 0) & (parents < len(hogs))))
        child_rows = parents >= 0
        np.testing.assert_array_equal(
            hogs["FamOff"][parents[child_rows]],
            hogs["FamOff"][child_rows],
        )
        assert np.all(parents[family_starts] == -1)

        child_hogs = database.root.ChildrenHOG[:]
        child_hog_count = int(hogs["ChildrenNum"].sum())
        _assert_buffer_offsets(
            hogs["ChildrenOff"],
            hogs["ChildrenNum"],
            len(child_hogs),
        )
        if child_hog_count:
            assert child_hog_count == len(child_hogs)
            owners = np.repeat(
                np.arange(len(hogs), dtype=np.int32), hogs["ChildrenNum"]
            )
            np.testing.assert_array_equal(hogs["ParentOff"][child_hogs], owners)

        child_proteins = database.root.ChildrenProt[:]
        _assert_buffer_offsets(
            hogs["ChildrenProtOff"],
            hogs["ChildrenProtNum"],
            len(child_proteins),
        )
        assert int(hogs["ChildrenProtNum"].sum()) == len(child_proteins)
        protein_owners = np.repeat(
            np.arange(len(hogs), dtype=np.uint32), hogs["ChildrenProtNum"]
        )
        np.testing.assert_array_equal(
            proteins["HOGoff"][child_proteins],
            protein_owners,
        )
        assert np.all(proteins["HOGoff"] < len(hogs))
        assert np.all((proteins["SpeOff"] >= 0) & (proteins["SpeOff"] < len(species)))

        children_taxa = database.root.ChildrenTax[:]
        _assert_buffer_offsets(
            taxonomy["ChildrenOff"],
            taxonomy["ChildrenNum"],
            len(children_taxa),
        )
        assert int(taxonomy["ChildrenNum"].sum()) == len(children_taxa)
        taxon_owners = np.repeat(
            np.arange(len(taxonomy), dtype=np.int32), taxonomy["ChildrenNum"]
        )
        np.testing.assert_array_equal(taxonomy["ParentOff"][children_taxa], taxon_owners)
        assert np.all(species["TaxOff"] < len(taxonomy))
        np.testing.assert_array_equal(taxonomy["ID"][species["TaxOff"]], species["ID"])
        extant = taxonomy["SpeOff"] >= 0
        assert np.all(taxonomy["SpeOff"][extant] < len(species))
        np.testing.assert_array_equal(
            species["ID"][taxonomy["SpeOff"][extant]],
            taxonomy["ID"][extant],
        )

        hog_taxa = database.root.HOGtaxa[:]
        assert np.all(hogs["HOGtaxaOff"] + hogs["HOGtaxaNum"] <= len(hog_taxa))
        assert np.all(hog_taxa < len(taxonomy))
        level_offsets = database.root.LevelOffsets[:]
        assert int(level_offsets[0]) == 0
        assert int(level_offsets[-1]) == len(hogs)
        assert np.all(level_offsets[1:] >= level_offsets[:-1])

        index = database.root.Index
        k = int(index._v_attrs["k"])
        alphabet_size = int(index._v_attrs["alphabet_n"])
        table_index = index.TableIndex
        table_buffer = index.TableBuffer
        assert len(table_index) == alphabet_size**k + 1
        assert int(table_index[0]) == 0
        assert int(table_index[-1]) == len(table_buffer)
        for part in _sample_slices(len(table_index)):
            values = table_index[part]
            assert np.all(values[1:] >= values[:-1])
        for part in _sample_slices(len(table_buffer)):
            assert np.all(table_buffer[part] < len(hogs))

        family_probability = index.FamilyProbability[:]
        hog_probability = index.HOGProbability[:]
        assert family_probability.shape == (len(families),)
        assert hog_probability.shape == (len(hogs),)
        assert np.all(np.isfinite(family_probability))
        assert np.all(np.isfinite(hog_probability))
        assert np.all((family_probability >= 0.0) & (family_probability <= 1.0))
        assert np.all((hog_probability >= 0.0) & (hog_probability <= 1.0))


@pytest.mark.parametrize("name", DATABASE_NAMES)
def test_candidate_db_matches_reference(name):
    reference_path = _database_path("OMAMER_REFERENCE_DB_DIR", name)
    candidate_path = _database_path("OMAMER_CANDIDATE_DB_DIR", name)

    with tables.open_file(reference_path, mode="r") as reference, tables.open_file(
        candidate_path, mode="r"
    ) as candidate:
        for attribute in ROOT_ATTRIBUTES:
            assert _normalise_attribute(reference.root._v_attrs[attribute]) == (
                _normalise_attribute(candidate.root._v_attrs[attribute])
            ), attribute
        for attribute in INDEX_ATTRIBUTES:
            assert _normalise_attribute(reference.root.Index._v_attrs[attribute]) == (
                _normalise_attribute(candidate.root.Index._v_attrs[attribute])
            ), attribute

        for path in STABLE_NODES:
            assert path in candidate, f"candidate is missing {path}"
            _assert_node_equal(reference, candidate, path)

        reference_hog_ids = _buffered_ids(reference, "/HOG", "/HOGIDBuffer")
        candidate_hog_ids = _buffered_ids(candidate, "/HOG", "/HOGIDBuffer")
        assert len(set(reference_hog_ids)) == len(reference_hog_ids)
        assert len(set(candidate_hog_ids)) == len(candidate_hog_ids)
        assert set(reference_hog_ids) == set(candidate_hog_ids)

        reference_hog_offset = {
            hog_id: offset for offset, hog_id in enumerate(reference_hog_ids)
        }
        candidate_to_reference = np.asarray(
            [reference_hog_offset[hog_id] for hog_id in candidate_hog_ids],
            dtype=np.uint32,
        )
        reference_to_candidate = np.empty_like(candidate_to_reference)
        reference_to_candidate[candidate_to_reference] = np.arange(
            len(candidate_to_reference), dtype=np.uint32
        )

        reference_hogs = reference.root.HOG[:]
        candidate_hogs = candidate.root.HOG[:][reference_to_candidate]
        for field in (
            "FamOff",
            "TaxOff",
            "NrMemberGenes",
            "CompletenessScore",
            "HOGtaxaNum",
        ):
            np.testing.assert_array_equal(candidate_hogs[field], reference_hogs[field])

        reference_adjustment = _legacy_sequence_length_adjustment(reference)
        candidate_adjustment = _legacy_sequence_length_adjustment(candidate)
        np.testing.assert_array_equal(
            candidate_hogs["MedianSeqLen"] - candidate_adjustment,
            reference_hogs["MedianSeqLen"] - reference_adjustment,
        )

        candidate_parents = candidate_hogs["ParentOff"].astype(np.int64)
        nonroot = candidate_parents >= 0
        candidate_parents[nonroot] = candidate_to_reference[
            candidate_parents[nonroot]
        ]
        np.testing.assert_array_equal(candidate_parents, reference_hogs["ParentOff"])

        reference_hog_taxa = reference.root.HOGtaxa
        candidate_hog_taxa = candidate.root.HOGtaxa
        for reference_offset in _comparison_indices(len(reference_hogs)):
            candidate_offset = int(reference_to_candidate[reference_offset])
            reference_row = reference_hogs[reference_offset]
            candidate_row = candidate.root.HOG[candidate_offset]
            reference_start = int(reference_row["HOGtaxaOff"])
            candidate_start = int(candidate_row["HOGtaxaOff"])
            count = int(reference_row["HOGtaxaNum"])
            np.testing.assert_array_equal(
                candidate_hog_taxa[candidate_start : candidate_start + count],
                reference_hog_taxa[reference_start : reference_start + count],
            )

        np.testing.assert_array_equal(
            candidate.root.Index.HOGProbability[:][reference_to_candidate],
            reference.root.Index.HOGProbability[:],
        )

        reference_protein_buffer = _protein_id_buffer(reference)
        candidate_protein_buffer = _protein_id_buffer(candidate)
        assert len(reference.root.Protein) == len(candidate.root.Protein)
        for part in _comparison_slices(len(reference.root.Protein)):
            reference_proteins = reference.root.Protein[part]
            candidate_proteins = candidate.root.Protein[part]
            np.testing.assert_array_equal(
                _protein_ids(candidate, candidate_proteins, candidate_protein_buffer),
                _protein_ids(reference, reference_proteins, reference_protein_buffer),
            )
            np.testing.assert_array_equal(
                candidate_proteins["SpeOff"],
                reference_proteins["SpeOff"],
            )
            np.testing.assert_array_equal(
                candidate_proteins["SeqLen"] - candidate_adjustment,
                reference_proteins["SeqLen"] - reference_adjustment,
            )
            np.testing.assert_array_equal(
                candidate_to_reference[candidate_proteins["HOGoff"]],
                reference_proteins["HOGoff"],
            )

        assert len(reference.root.Index.TableBuffer) == len(
            candidate.root.Index.TableBuffer
        )
        for part in _comparison_slices(len(reference.root.Index.TableBuffer)):
            np.testing.assert_array_equal(
                candidate_to_reference[candidate.root.Index.TableBuffer[part]],
                reference.root.Index.TableBuffer[part],
                err_msg=f"logical k-mer postings differ at {part}",
            )
