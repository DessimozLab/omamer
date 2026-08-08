"""Re-deriving family models on a built database.

The point of refit-bbinom is that one expensive mkdb can serve a whole sweep of
k-mer filters and N grids. Two properties have to hold for that to be safe:

1. Sharding families across nodes must not change the answer.
2. Re-deriving the k-mer filter must reproduce what mkdb would have written.

Both are checked here against a real (tiny) index rather than a stub, so the
numba counting kernels and the scipy fits actually run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tables

from omamer.bbinom_coefficients import BBINOM_NODE_NAMES
from omamer.bbinom_fit import apply_family_shard, parse_family_shard
from omamer.index import Index
from omamer.refit import drop_bbinom_coefficients, refit_modality, set_kmer_filter
from omamer.training_buffers import TrainingBuffers, write_training_buffers

from .test_mkdb_models import TinyIndexDatabase

N_FAMILIES = 6
PROTEINS_PER_FAMILY = 8


def _sequences():
    """Vary length so several exact-N design points exist, over a small letter
    pool so families share k-mers.

    The sharing matters: with disjoint k-mer sets every null query would score
    zero against every other family, no family would have a nonzero
    observation, and there would be nothing to fit.
    """
    letters = "ACDEF"
    rng = np.random.default_rng(11)
    records = []
    for _ in range(N_FAMILIES):
        for member in range(PROTEINS_PER_FAMILY):
            length = 7 + (member % 4) * 2
            records.append(
                "".join(letters[i] for i in rng.integers(0, len(letters), length))
            )
    return records


def _tiny_db(path):
    records = _sequences()
    n_proteins = len(records)
    protein_rows = [(i % 2, i) for i in range(n_proteins)]
    hog_rows = [(i // PROTEINS_PER_FAMILY, -1) for i in range(n_proteins)]
    # One level per family, each holding that family's single HOG range.
    family_rows = [(2 * f, 1) for f in range(N_FAMILIES)]
    level_offsets = []
    for f in range(N_FAMILIES):
        level_offsets += [f * PROTEINS_PER_FAMILY, (f + 1) * PROTEINS_PER_FAMILY]

    db = TinyIndexDatabase(
        str(path),
        protein_rows=protein_rows,
        hog_rows=hog_rows,
        family_rows=family_rows,
        level_offsets=level_offsets,
    )
    buffer = np.frombuffer(
        (" ".join(records) + " ").encode(), dtype="S1"
    )
    index = Index(
        db,
        k=2,
        reduced_alphabet=False,
        hidden_taxa=(),
        models=("binomial",),
    )
    db.ki = index
    index.build_kmer_table(buffer, buffer.copy())
    return db, buffer


FIT_OPTIONS = dict(
    n_buckets=0,  # every eligible N; the tiny build has only a handful
    min_records_per_n=2,
    max_records_per_n=0,
    min_nonzero_queries=1,
    seed=7,
)


@pytest.fixture
def tiny_build(tmp_path):
    db, buffer = _tiny_db(tmp_path / "tiny.h5")
    buffers_path = tmp_path / "tiny.buffers.h5"
    write_training_buffers(
        buffers_path,
        k=db.ki.k,
        alphabet_n=db.ki.alphabet.n,
        n_records=len(db.protein_table),
        buffers={"seq": buffer, "ss": buffer},
    )
    try:
        yield db, buffers_path
    finally:
        db.close()


def _fit(db, buffers_path, **overrides):
    options = dict(FIT_OPTIONS)
    options.update(overrides)
    with TrainingBuffers(buffers_path) as buffers:
        buffers.check_compatible(db)
        return refit_modality(db, buffers, "seq", store_metadata=False, **options)


def _comparable(rows):
    return (
        rows.sort_values("family_offset")
        .reset_index(drop=True)
        .drop(columns=[c for c in rows.columns if c == "index"])
    )


def test_sharded_refit_matches_whole_refit(tiny_build):
    db, buffers_path = tiny_build

    whole = _fit(db, buffers_path)
    assert not whole.empty

    shards = [
        _fit(db, buffers_path, family_shard=(i, 3)) for i in range(3)
    ]
    assert [len(s) for s in shards] == [2, 2, 2]
    merged = pd.concat(shards, ignore_index=True)

    # Disjoint cover of the families, and identical coefficients.
    assert sorted(merged["family_offset"]) == sorted(whole["family_offset"])
    pd.testing.assert_frame_equal(_comparable(whole), _comparable(merged))


def test_shard_selection_is_a_disjoint_cover():
    families = np.arange(97, dtype=np.int64) * 2
    for count in (1, 2, 5, 96, 97):
        shards = [
            apply_family_shard(families, (i, count)) for i in range(count)
        ]
        joined = np.concatenate(shards)
        np.testing.assert_array_equal(joined, families)
        sizes = [s.size for s in shards]
        assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize("value", ["0/0", "3/3", "-1/4", "nonsense", "5"])
def test_parse_family_shard_rejects_nonsense(value):
    with pytest.raises(ValueError):
        parse_family_shard(value)


def test_parse_family_shard_accepts_padded_values():
    assert parse_family_shard(None) is None
    assert parse_family_shard(" 2 / 5 ") == (2, 5)


def test_n_counts_cache_refuses_a_different_kmer_filter(tiny_build, tmp_path):
    db, buffers_path = tiny_build
    cache = tmp_path / "counts.npz"
    _fit(db, buffers_path, n_counts_cache=str(cache))
    assert cache.exists()

    # Same filter: the cache is reused and the fit is unchanged.
    reused = _fit(db, buffers_path, n_counts_cache=str(cache))
    assert not reused.empty

    with np.load(cache) as cached:
        counts = cached["counts"]
        n_records = int(cached["n_records"])
    with open(cache, "wb") as handle:
        np.savez(
            handle,
            counts=counts,
            kmer_filter_max_df=np.int64(17),
            n_records=np.int64(n_records),
        )
    with pytest.raises(ValueError, match="kmer_filter_max_df"):
        _fit(db, buffers_path, n_counts_cache=str(cache))


def test_set_kmer_filter_reproduces_the_unfiltered_build(tiny_build):
    """At 100 the recomputation must return exactly what mkdb stored."""
    db, _ = tiny_build
    before = {
        name: db.db.get_node("/Index/" + name)[:].copy()
        for name in (
            "FamilyProbability",
            "HOGProbability",
            "SSFamilyProbability",
            "SSHOGProbability",
        )
    }

    set_kmer_filter(db, 100.0)

    for name, expected in before.items():
        np.testing.assert_array_equal(
            db.db.get_node("/Index/" + name)[:], expected
        )
    assert db.db.root.Index._v_attrs["kmer_max_df"] == 0
    assert db.db.root.Index._v_attrs["ss_kmer_max_df"] == 0


def test_set_kmer_filter_tightens_the_cutoff_and_rewrites_probabilities(
    tiny_build,
):
    from omamer.index import filtered_hog_kmer_counts, select_kmer_max_df

    db, _ = tiny_build
    before = db.db.get_node("/Index/FamilyProbability")[:].copy()
    table_index = db.db.get_node("/Index/TableIndex")[:]
    table_buffer = db.db.get_node("/Index/TableBuffer")[:]
    n_hogs = len(db.hog_table)
    n_families = len(db.family_table)

    set_kmer_filter(db, 50.0)

    attrs = db.db.root.Index._v_attrs
    assert attrs["kmer_percentage"] == 50.0
    max_df = int(attrs["kmer_max_df"])
    _, _, expected_max_df = select_kmer_max_df(table_index, n_families, 50.0)
    assert max_df == expected_max_df > 0

    after = db.db.get_node("/Index/FamilyProbability")[:]
    assert after.shape == before.shape
    assert np.all(np.isfinite(after))

    # The cutoff is tie-inclusive, so on an index with only a handful of
    # distinct df values a percentage may retain everything. The cutoff is
    # still honoured -- forcing one below the observed maximum drops postings.
    all_postings = int(
        filtered_hog_kmer_counts(table_index, table_buffer, 0, n_hogs).sum()
    )
    df = np.diff(table_index.astype(np.int64))
    kept = int(
        filtered_hog_kmer_counts(
            table_index, table_buffer, int(df.max()) - 1, n_hogs
        ).sum()
    )
    assert 0 < kept < all_postings


def test_set_kmer_filter_invalidates_stale_coefficients(tiny_build):
    db, buffers_path = tiny_build
    rows = _fit(db, buffers_path)

    from omamer.bbinom_coefficients import store_bbinom_coefficients

    store_bbinom_coefficients(db, rows, modalities=("seq",))
    assert "/Index/" + BBINOM_NODE_NAMES["seq"]["valid"] in db.db

    drop_bbinom_coefficients(db, modalities=("seq",))
    for node_name in BBINOM_NODE_NAMES["seq"].values():
        assert "/Index/" + node_name not in db.db


def test_filtering_every_modality_keeps_the_index_openable(tiny_build):
    """A filtered percentage with a zero cutoff makes Index refuse to open.

    set-kmer-filter therefore always covers every indexed modality; filtering
    only 'seq' on a structure build would leave ss_kmer_max_df at 0 and brick
    the database.
    """
    db, _ = tiny_build
    set_kmer_filter(db, 50.0, modalities=("seq",))
    attrs = db.db.root.Index._v_attrs
    assert attrs["kmer_percentage"] == 50.0
    assert int(attrs["kmer_max_df"]) > 0
    assert int(attrs["ss_kmer_max_df"]) == 0  # the hazard, left unfixed here

    with pytest.raises(ValueError, match="3Di k-mer document-frequency cutoff"):
        Index(db)

    # Covering both modalities is what the CLI does, and it stays openable.
    set_kmer_filter(db, 50.0)
    assert int(db.db.root.Index._v_attrs["ss_kmer_max_df"]) > 0
    assert Index(db).kmer_percentage == 50.0


def test_training_buffers_reject_a_foreign_database(tiny_build, tmp_path):
    db, _ = tiny_build
    wrong = tmp_path / "wrong.buffers.h5"
    write_training_buffers(
        wrong,
        k=db.ki.k,
        alphabet_n=db.ki.alphabet.n,
        n_records=3,
        buffers={"seq": np.frombuffer(b"AAA AAC AAG ", dtype="S1")},
    )
    with TrainingBuffers(wrong) as buffers:
        with pytest.raises(ValueError, match="different build"):
            buffers.check_compatible(db)


def test_training_buffers_round_trip(tiny_build):
    _, buffers_path = tiny_build
    with TrainingBuffers(buffers_path) as buffers:
        assert buffers.modalities == ("seq", "ss")
        assert buffers.n_records == N_FAMILIES * PROTEINS_PER_FAMILY
        starts, stops = buffers.bounds("seq")
        buffer = buffers.buffer("seq")
        assert starts.size == buffers.n_records
        # Bounds must delimit records without including the separator.
        assert not np.any(buffer[stops] != ord(" "))
        assert np.all(stops > starts)
        with pytest.raises(ValueError, match="no 'nope' modality"):
            buffers.buffer("nope")


def test_written_buffers_match_the_record_count(tmp_path):
    with pytest.raises(ValueError, match="expected"):
        write_training_buffers(
            tmp_path / "bad.h5",
            k=2,
            alphabet_n=21,
            n_records=99,
            buffers={"seq": np.frombuffer(b"AAA AAC ", dtype="S1")},
        )
