"""Re-derive family models on an existing database.

Everything a beta-binomial sweep varies is cheap; the k-mer tables it varies
over are not. ``mkdb`` spends most of its time building suffix arrays, and
neither the k-mer filter nor the fitting design changes them:
``apply_information_filter`` only records a document-frequency cutoff that
search applies on the fly, and the fit only reads the tables.

So one ``mkdb`` run plus the two operations here cover a whole matrix:

  set-kmer-filter   rewrite kmer_percentage, the df cutoff, and the binomial
                    probabilities derived from it
  refit-bbinom      re-fit beta-binomial coefficients, optionally for one
                    family shard so a large build can fan out over nodes
"""

from __future__ import annotations

import numpy as np

from ._utils import LOG

MODALITY_PREFIX = {"seq": "", "ss": "SS"}


def _modality_table_nodes(db, modality):
    index_group = db.db.root.Index
    if modality == "seq":
        return index_group.TableIndex, index_group.TableBuffer
    return index_group.SSTableIndex, index_group.SSTableBuffer


def available_modalities(db):
    return ("seq", "ss") if db.has_structure() else ("seq",)


def _remove_nodes(db, names):
    index_group = db.db.root.Index
    for name in names:
        path = "{}/{}".format(index_group._v_pathname, name)
        if path in db.db:
            db.db.remove_node(path)


def set_kmer_filter(db, kmer_percentage, modalities=None):
    """Rewrite the PMI k-mer filter and everything derived from it.

    The k-mer tables themselves are untouched: the filter is a stored df
    cutoff that search applies while scanning postings. What has to be
    recomputed is the cutoff and the binomial family/HOG probabilities, which
    are estimated from postings that survive it.

    Beta-binomial coefficients are invalidated rather than kept, because they
    were fitted against a different background. Refit after calling this.
    """
    from .index import (
        filtered_hog_kmer_counts,
        select_kmer_max_df,
        validate_kmer_percentage,
    )
    from .stat_models import estimate_family_probability, estimate_hog_probability

    kmer_percentage = validate_kmer_percentage(kmer_percentage)
    index_group = db.db.root.Index
    n_families = len(db.family_table)
    n_hogs = len(db.hog_table)
    hog_to_family = db.hog_table.col("FamOff")
    if modalities is None:
        modalities = available_modalities(db)

    index_group._f_setattr("kmer_percentage", kmer_percentage)
    for modality in modalities:
        prefix = MODALITY_PREFIX[modality]
        table_index_node, table_buffer_node = _modality_table_nodes(db, modality)
        table_index = table_index_node[:]
        table_buffer = table_buffer_node[:]

        if kmer_percentage == 100.0:
            max_df = 0
            n_present = n_retained = 0
        else:
            n_present, n_retained, max_df = select_kmer_max_df(
                table_index, n_families, kmer_percentage
            )
            if n_retained == 0:
                raise RuntimeError(
                    "{} k-mer information filter retained no indexed "
                    "k-mers".format(modality)
                )
            LOG.info(
                "%s information filter: retained %d of %d indexed k-mers "
                "(%.2f%%; df <= %d; PMI >= %.3f bits)",
                modality,
                n_retained,
                n_present,
                100.0 * n_retained / n_present,
                max_df,
                float(np.log2(n_families / max_df)),
            )

        hog_kmer_counts = filtered_hog_kmer_counts(
            table_index, table_buffer, max_df, n_hogs
        )
        family_probability = estimate_family_probability(
            table_index,
            table_buffer,
            hog_to_family,
            max_df,
            n_families,
        )
        hog_probability = estimate_hog_probability(
            hog_kmer_counts,
            db.family_table.col("LevelOff"),
            db.family_table.col("LevelNum"),
            db.level_offset_carray[:],
            db.hog_table.col("ParentOff"),
        )

        attr = "kmer_max_df" if modality == "seq" else "ss_kmer_max_df"
        index_group._f_setattr(attr, int(max_df))
        _remove_nodes(
            db, (prefix + "FamilyProbability", prefix + "HOGProbability")
        )
        db.db.create_carray(
            index_group,
            prefix + "FamilyProbability",
            obj=family_probability,
            filters=db.compression_filters,
        )
        db.db.create_carray(
            index_group,
            prefix + "HOGProbability",
            obj=hog_probability,
            filters=db.compression_filters,
        )
        LOG.info(
            "Rewrote %s binomial probabilities for kmer_percentage=%s "
            "(df <= %d)",
            modality,
            kmer_percentage,
            max_df,
        )
    return kmer_percentage


def mark_models(db, with_bbinom):
    """Keep the stored ``models`` attribute honest about what is present.

    Search itself is safe either way -- the coefficient loaders are guarded and
    fall back to the binomial -- but ``omamer info`` reads this attribute, and a
    database advertising a model it does not carry is a trap for anything that
    selects databases by it.
    """
    from .stat_models import validate_index_models

    models = ("binomial", "beta-binomial") if with_bbinom else ("binomial",)
    db.db.root.Index._f_setattr("models", ",".join(validate_index_models(models)))
    return models


def drop_bbinom_coefficients(db, modalities=None):
    """Remove beta-binomial arrays so search falls back to the binomial."""
    from .bbinom_coefficients import BBINOM_NODE_NAMES

    if modalities is None:
        modalities = available_modalities(db)
    for modality in modalities:
        _remove_nodes(db, tuple(BBINOM_NODE_NAMES[modality].values()))
        LOG.info("Dropped %s beta-binomial coefficients", modality)

    remaining = any(
        "{}/{}".format(db.db.root.Index._v_pathname, node_name) in db.db
        for other in available_modalities(db)
        for node_name in BBINOM_NODE_NAMES[other].values()
    )
    mark_models(db, remaining)


def refit_modality(
    db,
    buffers,
    modality,
    *,
    n_values=None,
    n_buckets=24,
    min_records_per_n=50,
    max_records_per_n=500,
    min_nonzero_queries=20,
    max_families=0,
    min_family_prob=0.0,
    seed=42,
    workers=1,
    max_histogram_gb=3.0,
    family_offsets_path=None,
    family_shard=None,
    n_counts_cache=None,
    store_metadata=True,
):
    """Fit one modality from a training-buffer sidecar and return the rows."""
    from .bbinom_fit import fit_bbinom_coefficients_from_buffer

    sequence_buffer = buffers.buffer(modality)
    bounds = buffers.bounds(modality)
    return fit_bbinom_coefficients_from_buffer(
        db,
        sequence_buffer,
        modality=modality,
        n_values=n_values,
        n_buckets=n_buckets,
        min_records_per_n=min_records_per_n,
        max_records_per_n=max_records_per_n,
        min_nonzero_queries=min_nonzero_queries,
        max_families=max_families,
        min_family_prob=min_family_prob,
        seed=seed,
        workers=workers,
        max_histogram_gb=max_histogram_gb,
        family_offsets_path=family_offsets_path,
        family_shard=family_shard,
        bounds=bounds,
        n_counts_cache=n_counts_cache,
        store_metadata=store_metadata,
    )
