"""
    OMAmer - tree-driven and alignment-free protein assignment to sub-families

    (C) 2024-present Nikolai Romashchenko <nikolai.romashchenko@unil.ch>
    (C) 2022-2023 Alex Warwick Vesztrocy <alex.warwickvesztrocy@unil.ch>
    (C) 2019-2021 Victor Rossier <victor.rossier@unil.ch> and
                  Alex Warwick Vesztrocy <alex@warwickvesztrocy.co.uk>

    This file is part of OMAmer.

    OMAmer is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OMAmer is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with OMAmer. If not, see <http://www.gnu.org/licenses/>.
"""
import math
from collections import namedtuple
from enum import IntEnum

import numba
import numpy as np

from ._utils import LOG

MAX_FAMILY_NEGLOGP = 20000.0


class FamilyModel(IntEnum):
    BINOMIAL = 0
    BETA_BINOMIAL = 1


# These immutable carriers are built once at the Python/Numba boundary and
# passed unchanged to the compiled functions that consume them.
# `packed` holds the beta-binomial coefficients of each family in one row
# (see `pack_family_bbinom`), so that evaluating the model for a family reads
# one or two cache lines instead of one per array; the search evaluates it
# for every family hit by a query. Empty: the separate arrays are used.
FamilyModelParameters = namedtuple(
    "FamilyModelParameters",
    (
        "kind",
        "family_probability",
        "q_coef",
        "kappa_coef",
        "center",
        "scale",
        "valid",
        "n_min",
        "n_max",
        "packed",
    ),
    defaults=(np.empty((0, 0), dtype=np.float64),),
)

# Columns of a packed row; the q then the kappa coefficients follow.
PACKED_CENTER = 0
# > 0 if and only if the family has a beta-binomial fit (as
# `has_family_bbinom`); the row is all 0 otherwise
PACKED_SCALE = 1
PACKED_N_MIN = 2
PACKED_N_MAX = 3
# kappa when it does not depend on n (kappa coefficients of degree >= 1 are 0,
# as with a pooled kappa), else NaN; see `packed_bbinom_kappa`
PACKED_KAPPA = 4
PACKED_Q_COEF = 5
# |z| up to which the kappa of PACKED_KAPPA is the one computed from the
# coefficients (beyond, a power of z may overflow, and 0 * inf is NaN)
PACKED_KAPPA_MAX_Z = 1e30

# summed_neglog_alpha: threshold on summed sequence + 3Di scores (sum and hybrid
# modes); -1 means the same as neglog_alpha. ss_weight: weight of the 3Di score
# in that sum (score = sequence + ss_weight * 3Di).
FamilyScoringParameters = namedtuple(
    "FamilyScoringParameters",
    ("neglog_alpha", "log_correction", "summed_neglog_alpha", "ss_weight"),
    defaults=(-1.0, 1.0),
)

HogModelParameters = namedtuple(
    "HogModelParameters",
    ("hog_probability",),
)

IndexModelData = namedtuple(
    "IndexModelData",
    (
        "sequence_buffer",
        "table_index",
        "table_buffer",
        "hog_kmer_counts",
        "max_document_frequency",
    ),
)

_EMPTY_F64 = np.empty(0, dtype=np.float64)
_EMPTY_F64_2D = np.empty((0, 0), dtype=np.float64)
_EMPTY_BOOL = np.empty(0, dtype=np.bool_)
_EMPTY_U32 = np.empty(0, dtype=np.uint32)

SUPPORTED_INDEX_MODELS = ("binomial", "beta-binomial")


def validate_index_models(models):
    """Validate the statistical models requested while building an index."""
    if isinstance(models, str):
        models = models.replace(",", " ").split()
    selected = tuple(dict.fromkeys(models))
    unknown = sorted(set(selected).difference(SUPPORTED_INDEX_MODELS))
    if unknown:
        raise ValueError(
            "Unknown index model(s): {}".format(", ".join(unknown))
        )
    if not selected:
        raise ValueError("At least one index model is required")
    if "beta-binomial" in selected and "binomial" not in selected:
        raise ValueError(
            "The beta-binomial model requires the binomial fallback; "
            "request both 'binomial' and 'beta-binomial'"
        )
    return selected


@numba.njit(cache=True, nogil=True)
def _family_kmer_occurrence(
    table_index,
    table_buffer,
    hog_to_family,
    max_document_frequency,
    n_families,
):
    """Count family-weighted postings retained by the index filter."""
    family_occurrence = np.zeros(n_families, dtype=np.int64)
    n_postings = 0
    cutoff = np.int64(max_document_frequency)
    for kmer in range(table_index.size - 1):
        start = np.int64(table_index[kmer])
        stop = np.int64(table_index[kmer + 1])
        document_frequency = stop - start
        if document_frequency == 0 or (
            cutoff != 0 and document_frequency > cutoff
        ):
            continue
        for position in range(start, stop):
            family_occurrence[hog_to_family[table_buffer[position]]] += (
                document_frequency
            )
        n_postings += document_frequency
    return family_occurrence, n_postings


def estimate_family_probability(
    table_index,
    table_buffer,
    hog_to_family,
    max_document_frequency=0,
    n_families=None,
):
    """Estimate independent binomial hit probabilities for every family."""
    if n_families is None:
        n_families = int(np.max(hog_to_family)) + 1
    family_occurrence, n_postings = _family_kmer_occurrence(
        table_index,
        table_buffer,
        hog_to_family,
        max_document_frequency,
        n_families,
    )
    if n_postings == 0:
        raise RuntimeError("Cannot estimate family probabilities from an empty index")
    return family_occurrence / n_postings


@numba.njit(nogil=True)
def cumulate_counts_1fam(
    hog_cumulative_counts,
    family_level_offsets,
    hog_to_parent,
):
    current_best_child_count = np.zeros(
        hog_cumulative_counts.shape,
        dtype=np.uint32,
    )
    for level in range(family_level_offsets.size - 2):
        bounds = family_level_offsets[-level - 3 : -level - 1]
        hog_cumulative_counts[bounds[0] : bounds[1]] = np.add(
            hog_cumulative_counts[bounds[0] : bounds[1]],
            current_best_child_count[bounds[0] : bounds[1]],
        )
        for hog_offset in range(bounds[0], bounds[1]):
            parent_offset = hog_to_parent[hog_offset]
            if parent_offset != -1:
                current = current_best_child_count[parent_offset]
                current_best_child_count[parent_offset] = max(
                    current,
                    hog_cumulative_counts[hog_offset],
                )


@numba.njit(parallel=True, nogil=True)
def _cumulate_hog_counts(
    hog_counts,
    family_level_offset,
    family_level_count,
    level_offsets,
    hog_to_parent,
):
    cumulative = hog_counts.copy()
    for family in numba.prange(family_level_offset.size):
        start = family_level_offset[family]
        stop = np.int32(start + family_level_count[family] + 2)
        cumulate_counts_1fam(
            cumulative,
            level_offsets[start:stop],
            hog_to_parent,
        )
    return cumulative


def estimate_hog_probability(
    hog_counts,
    family_level_offset,
    family_level_count,
    level_offsets,
    hog_to_parent,
):
    """Estimate binomial HOG probabilities after hierarchy cumulation."""
    total = hog_counts.sum()
    if total == 0:
        raise RuntimeError("Cannot estimate HOG probabilities from an empty index")
    occurrence = _cumulate_hog_counts(
        hog_counts,
        family_level_offset,
        family_level_count,
        level_offsets,
        hog_to_parent,
    )
    return occurrence / total


def _write_binomial_model(db, index_group, modality, data):
    prefix = "" if modality == "seq" else "SS"
    hog_to_family = db.hog_table.col("FamOff")
    family_probability = estimate_family_probability(
        data.table_index,
        data.table_buffer,
        hog_to_family,
        data.max_document_frequency,
        len(db.family_table),
    )
    hog_probability = estimate_hog_probability(
        data.hog_kmer_counts,
        db.family_table.col("LevelOff"),
        db.family_table.col("LevelNum"),
        db.level_offset_carray[:],
        db.hog_table.col("ParentOff"),
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


def learn_index_models(
    db,
    index_group,
    modality_data,
    models=("binomial",),
    bbinom_options=None,
):
    """Learn and persist every requested model for every indexed modality."""
    models = validate_index_models(models)
    bbinom_options = dict(bbinom_options or {})
    n_values_by_modality = dict(
        bbinom_options.pop("n_values_by_modality", {}) or {}
    )
    kappa_scale_by_modality = dict(
        bbinom_options.pop("kappa_scale_by_modality", {}) or {}
    )
    min_distinct_by_modality = dict(
        bbinom_options.pop("min_distinct_by_modality", {}) or {}
    )

    for modality, data in modality_data.items():
        LOG.info("Learning binomial model for modality '%s'", modality)
        _write_binomial_model(db, index_group, modality, data)
        # search applies the low-complexity mask whatever the model (the
        # beta-binomial fit records it too, as it trains with the same mask)
        min_distinct = min_distinct_by_modality.get(modality)
        if min_distinct is not None:
            index_group._f_setattr(
                "{}_kmer_min_distinct".format(modality), int(min_distinct)
            )

    if "beta-binomial" in models:
        from .bbinom_coefficients import store_bbinom_coefficients
        from .bbinom_fit import (
            InsufficientTrainingData,
            fit_bbinom_coefficients_from_buffer,
        )

        unavailable_n_values = sorted(
            modality
            for modality, values in n_values_by_modality.items()
            if values is not None and modality not in modality_data
        )
        if unavailable_n_values:
            raise ValueError(
                "Beta-binomial fitting options supplied for unavailable "
                "modality: {}".format(", ".join(unavailable_n_values))
            )

        fitted = []
        for modality, data in modality_data.items():
            LOG.info("Learning beta-binomial model for modality '%s'", modality)
            fit_options = dict(bbinom_options)
            modality_n_values = n_values_by_modality.get(modality)
            if modality_n_values is not None:
                fit_options["n_values"] = modality_n_values
            if kappa_scale_by_modality.get(modality) is not None:
                fit_options["kappa_scale"] = float(kappa_scale_by_modality[modality])
            if min_distinct_by_modality.get(modality) is not None:
                fit_options["min_distinct"] = int(min_distinct_by_modality[modality])
            try:
                rows = fit_bbinom_coefficients_from_buffer(
                    db,
                    data.sequence_buffer,
                    modality=modality,
                    table_index=data.table_index,
                    table_buffer=data.table_buffer,
                    **fit_options,
                )
            except InsufficientTrainingData as e:
                # e.g. a small database: search uses the binomial model
                LOG.warning(
                    "Not fitting a beta-binomial model for modality '%s': %s",
                    modality,
                    e,
                )
                continue
            fitted.append(modality)
            store_bbinom_coefficients(
                db,
                rows,
                modalities=(modality,),
                kmer_percentage=db.ki.kmer_percentage,
            )

        if not fitted:
            models = tuple(m for m in models if m != "beta-binomial")

    index_group._f_setattr("models", ",".join(models))


def get_family_model(policy, has_coefficients):
    if policy == "auto":
        return (
            FamilyModel.BETA_BINOMIAL
            if has_coefficients
            else FamilyModel.BINOMIAL
        )
    if policy == "binomial":
        return FamilyModel.BINOMIAL
    if policy in ("bbinom", "beta-binomial"):
        if not has_coefficients:
            raise ValueError(
                "family_model={!r} requires beta-binomial coefficients".format(
                    policy
                )
            )
        return FamilyModel.BETA_BINOMIAL
    raise ValueError(
        "family_model must be 'auto', 'binomial', or 'bbinom'"
    )


def family_log_correction(policy, n_families):
    if policy == "bonferroni":
        return math.log(int(n_families))
    if policy == "none":
        return 0.0
    raise ValueError(
        "family_correction must be 'bonferroni' or 'none', got {!r}".format(
            policy
        )
    )


def make_family_model(
    policy,
    probability,
    q_coef=None,
    kappa_coef=None,
    center=None,
    scale=None,
    valid=None,
    n_min=None,
    n_max=None,
):
    has_coefficients = valid is not None and valid.size > 0 and np.any(valid)
    kind = get_family_model(policy, has_coefficients)
    if kind == FamilyModel.BINOMIAL:
        return FamilyModelParameters(
            kind,
            probability,
            _EMPTY_F64_2D,
            _EMPTY_F64_2D,
            _EMPTY_F64,
            _EMPTY_F64,
            _EMPTY_BOOL,
            _EMPTY_U32,
            _EMPTY_U32,
        )
    model = FamilyModelParameters(
        kind,
        probability,
        q_coef,
        kappa_coef,
        center,
        scale,
        valid,
        n_min,
        n_max,
    )
    return model._replace(packed=pack_family_bbinom(model))


def pack_family_bbinom(model):
    """
    One row per family: [center, scale (0: no beta-binomial fit, as
    `has_family_bbinom`), n_min, n_max (0: no clamp, as `bbinom_eval_n`),
    kappa if constant (NaN: depends on n), q coefficients, kappa
    coefficients], in float64 (exact for the uint32 n range). Rows are
    padded to 8 or 16 columns, so that the part of a row read with a
    constant kappa and up to 3 q coefficients lies in one cache line.

    When every kappa is constant (as with a pooled kappa), the kappa
    coefficients are left out and rows of up to 3 q coefficients take 8
    columns: one cache line per family, and half the memory. The search
    reads the row of every candidate family of a query, in no order, so
    that it mostly waits for memory; on Metazoa (215,667 families) the
    16-column rows (28 MB) do not fit a 16 MB L3 cache, the 8-column ones
    do, which halves the time of the count filter.
    """
    n_families = model.family_probability.size
    n_q = model.q_coef.shape[1]
    n_kappa = model.kappa_coef.shape[1]
    width = PACKED_Q_COEF + n_q + n_kappa
    width = 8 if width <= 8 else 16 * ((width + 15) // 16)
    packed = np.zeros((n_families, width), dtype=np.float64)

    def present(values):
        values = _EMPTY_F64 if values is None else np.asarray(values)
        m = min(n_families, values.shape[0])
        return m, values[:m]

    m = min(
        n_families,
        model.valid.size,
        model.q_coef.shape[0],
        model.kappa_coef.shape[0],
        model.center.size,
        model.scale.size,
    )
    packed[:m, PACKED_CENTER] = model.center[:m]
    packed[:m, PACKED_SCALE] = model.scale[:m]
    packed[:m, PACKED_Q_COEF : PACKED_Q_COEF + n_q] = model.q_coef[:m]
    packed[:m, PACKED_Q_COEF + n_q : PACKED_Q_COEF + n_q + n_kappa] = (
        model.kappa_coef[:m]
    )
    k, n_max = present(model.n_max)
    packed[:k, PACKED_N_MAX] = n_max
    k, n_min = present(model.n_min)
    packed[:k, PACKED_N_MIN] = n_min
    _fill_constant_kappa(packed, n_q, n_kappa)

    has_fit = np.zeros(n_families, dtype=np.bool_)
    has_fit[:m] = model.valid[:m] & (model.scale[:m] > 0.0)
    packed[~has_fit] = 0.0

    # Without the kappa coefficients, `packed_bbinom_params_q` must never
    # compute kappa from them: every kappa constant, and |z| within
    # PACKED_KAPPA_MAX_Z at any n (log n_eval is in [0, 45) for the int64 n).
    fit_rows = packed[has_fit]
    with np.errstate(divide="ignore", invalid="ignore"):
        max_z = (np.abs(fit_rows[:, PACKED_CENTER]) + 45.0) / fit_rows[
            :, PACKED_SCALE
        ]
    if (
        PACKED_Q_COEF + n_q <= 8
        and not np.isnan(fit_rows[:, PACKED_KAPPA]).any()
        and np.all(max_z <= PACKED_KAPPA_MAX_Z)
    ):
        packed = np.ascontiguousarray(packed[:, :8])
    return packed


@numba.njit(nogil=True)
def binom_neglogccdf(x, n, p):
    """
    Pure-numba implementation for the neg-log of the upper tail
    P(X >= x) of a Binomial(n, p).
    Stops early once the decaying tail is negligible.
    """
    if x <= 0:
        return 0.0
    if x > n:
        return np.inf

    log_px = (
        math.lgamma(n + 1.0)
        - math.lgamma(x + 1.0)
        - math.lgamma(n - x + 1.0)
        + x * math.log(p)
        + (n - x) * math.log1p(-p)
    )

    odds = p / (1.0 - p)
    acc = 1.0
    term = 1.0
    for k in range(x, n):
        ratio = ((n - k) / (k + 1.0)) * odds
        term *= ratio
        acc += term
        # tail is decaying and this term no longer moves the sum -> stop
        if ratio < 1.0 and term < acc * 1e-16:
            break

    return -(log_px + math.log(acc))


@numba.njit(nogil=True)
def has_family_bbinom(family_id, model):
    return (
        model.kind == FamilyModel.BETA_BINOMIAL
        and model.valid.size > family_id
        and model.q_coef.shape[0] > family_id
        and model.kappa_coef.shape[0] > family_id
        and model.center.size > family_id
        and model.scale.size > family_id
        and model.valid[family_id]
        and model.scale[family_id] > 0.0
    )


@numba.njit(nogil=True)
def bbinom_eval_n(family_id, n, n_min, n_max):
    # Clamp the query unique-kmer count to the family's trained N range
    # before evaluating the smooth q(N)/kappa(N) curve, so the length-aware
    # model is never extrapolated outside the range it was fitted on. The
    # Beta-Binomial support and observed count still use the actual N.
    n_eval = n
    if n_max.size > family_id and n_max[family_id] > 0:
        lo = n_min[family_id]
        hi = n_max[family_id]
        if n_eval < lo:
            n_eval = lo
        elif n_eval > hi:
            n_eval = hi
    return n_eval


@numba.njit(nogil=True)
def bbinom_expected_count(family_id, n, model):
    n_eval = bbinom_eval_n(family_id, n, model.n_min, model.n_max)
    _, _, q = beta_binomial_params_for_n(
        n_eval,
        model.q_coef[family_id],
        model.kappa_coef[family_id],
        model.center[family_id],
        model.scale[family_id],
    )
    return q * n


@numba.njit(nogil=True)
def family_expected_count(family_id, n, model):
    if has_family_bbinom(family_id, model):
        return bbinom_expected_count(family_id, n, model)
    return model.family_probability[family_id] * n


@numba.njit(nogil=True)
def family_neglogccdf(family_id, x, n, model):
    if has_family_bbinom(family_id, model):
        n_eval = bbinom_eval_n(family_id, n, model.n_min, model.n_max)
        alpha, beta, _ = beta_binomial_params_for_n(
            n_eval,
            model.q_coef[family_id],
            model.kappa_coef[family_id],
            model.center[family_id],
            model.scale[family_id],
        )
        return beta_binomial_neglogccdf(x, n, alpha, beta)
    return binom_neglogccdf(x, n, model.family_probability[family_id])


@numba.njit(nogil=True)
def packed_bbinom_z(row, n, log_n):
    """
    The z of `beta_binomial_params_for_n` at the clamped n (`bbinom_eval_n`)
    of a packed row; log_n = log(n), reused when n is not clamped.
    """
    n_eval = n
    if row[PACKED_N_MAX] > 0.0:
        lo = np.int64(row[PACKED_N_MIN])
        hi = np.int64(row[PACKED_N_MAX])
        if n_eval < lo:
            n_eval = lo
        elif n_eval > hi:
            n_eval = hi
    log_n_eval = log_n if n_eval == n else math.log(float(n_eval))
    return (log_n_eval - row[PACKED_CENTER]) / row[PACKED_SCALE]


@numba.njit(nogil=True)
def packed_bbinom_q(row, z, n_q):
    """q of `beta_binomial_params_for_n`, with the same operations."""
    q_linear = 0.0
    z_power = 1.0
    for i in range(n_q):
        q_linear += row[PACKED_Q_COEF + i] * z_power
        z_power *= z
    return min(max(sigmoid(q_linear), 1e-8), 1.0 - 1e-8)


@numba.njit(nogil=True)
def packed_bbinom_kappa(row, z, n_q, n_kappa):
    """kappa of `beta_binomial_params_for_n`, with the same operations."""
    log_kappa = 0.0
    z_power = 1.0
    for i in range(n_kappa):
        log_kappa += row[PACKED_Q_COEF + n_q + i] * z_power
        z_power *= z
    log_kappa = min(max(log_kappa, math.log(1e-4)), math.log(1e8))
    return math.exp(log_kappa)


@numba.njit(nogil=True, cache=True)
def _fill_constant_kappa(packed, n_q, n_kappa):
    """
    PACKED_KAPPA: kappa at z = 0 when the kappa coefficients of degree >= 1
    are 0. Then the terms of these degrees are +-0 at any |z| <=
    PACKED_KAPPA_MAX_Z, and kappa is bit-identical at every n.
    """
    for f in range(packed.shape[0]):
        row = packed[f]
        constant = True
        for i in range(1, n_kappa):
            if row[PACKED_Q_COEF + n_q + i] != 0.0:
                constant = False
        row[PACKED_KAPPA] = (
            packed_bbinom_kappa(row, 0.0, n_q, n_kappa) if constant else np.nan
        )


@numba.njit(nogil=True)
def packed_bbinom_params(row, n, log_n, n_q, n_kappa):
    """(alpha, beta) of `beta_binomial_params_for_n`, from a packed row."""
    z = packed_bbinom_z(row, n, log_n)
    q = packed_bbinom_q(row, z, n_q)
    return packed_bbinom_params_q(row, z, q, n_q, n_kappa)


@numba.njit(nogil=True)
def packed_bbinom_params_q(row, z, q, n_q, n_kappa):
    """`packed_bbinom_params` given z and q (`packed_bbinom_z`, `_q`)."""
    kappa = row[PACKED_KAPPA]
    if not (kappa == kappa and abs(z) <= PACKED_KAPPA_MAX_Z):
        kappa = packed_bbinom_kappa(row, z, n_q, n_kappa)

    alpha = q * kappa
    beta = (1.0 - q) * kappa
    return alpha, beta


@numba.njit(nogil=True)
def has_packed_bbinom(model):
    return (
        model.kind == FamilyModel.BETA_BINOMIAL
        and model.packed.shape[0] == model.family_probability.size
        and model.packed.shape[0] > 0
    )


@numba.njit(nogil=True)
def compute_expected_counts(qres, n, model):
    if model.kind == FamilyModel.BINOMIAL:
        expected_count = model.family_probability[qres["id"]] * n
    elif has_packed_bbinom(model):
        expected_count = np.empty(len(qres), dtype=np.float64)
        log_n = math.log(float(n)) if n > 0 else 0.0
        n_q = model.q_coef.shape[1]
        for i in range(len(qres)):
            family_id = qres["id"][i]
            row = model.packed[family_id]
            if row[PACKED_SCALE] > 0.0:
                z = packed_bbinom_z(row, n, log_n)
                expected_count[i] = packed_bbinom_q(row, z, n_q) * n
            else:
                expected_count[i] = model.family_probability[family_id] * n
    else:
        expected_count = np.empty(len(qres), dtype=np.float64)
        for i in range(len(qres)):
            expected_count[i] = family_expected_count(qres["id"][i], n, model)

    return expected_count


@numba.njit(nogil=True)
def filter_family_candidates(qres, n, model):
    """
    Keep families whose observed k-mer count reaches the model expectation.
    """
    expected_count = compute_expected_counts(qres, n, model)
    return qres[qres["count"] >= expected_count]


# Largest observed count for which `beta_binomial_pmf_test` uses the bound of
# `beta_binomial_neglogpmf_upper`: it costs one multiplication-division step
# per unit of the count.
PMF_BOUND_MAX_COUNT = 64


@numba.njit(nogil=True)
def beta_binomial_neglogpmf_upper(x, n, alpha, beta):
    """
    An upper bound of -log P(X = x) for X ~ BetaBinomial(n, alpha, beta),
    with 0 < x <= n, without lgamma: acc + log(prod) + extra (`acc` is 0
    unless the partial products leave [1e-100, 1e100]).

    With a = n - x and kappa = alpha + beta, `beta_binomial_logpmf` is

        -log P(X = x) = sum_{j<x} log[(j + 1) (a + beta + j)
                                       / ((j + alpha) (a + 1 + j))]
                        + E1 - E2,
        E1 = lgamma(n + beta + alpha) - lgamma(n + beta),
        E2 = lgamma(beta + alpha) - lgamma(beta),

    the integer-step lgamma differences written as products. lgamma is
    convex, so an increment lgamma(y + alpha) - lgamma(y) lies between
    alpha psi(y) and alpha psi(y + alpha), and the digamma function psi
    satisfies log(y) - 1/y < psi(y) < log(y) - 1/(2y) for y > 0. Hence

        E1 - E2 <= alpha [psi(n + kappa) - psi(beta)]
                <  alpha [log(r) + 1/beta]
                <= alpha [(r - 1) / sqrt(r) + 1/beta] = extra,

    with r = (n + kappa) / beta >= 1 (log(y) <= (y - 1) / sqrt(y) for
    y >= 1). The bound exceeds the exact value by about alpha (alpha + 1) /
    beta: little for the family models, where alpha = q kappa is small
    relative to beta.
    """
    a = n - x
    acc = 0.0
    num = 1.0
    den = 1.0
    for j in range(x):
        num *= (j + 1.0) * (a + beta + j)
        den *= (j + alpha) * (a + 1.0 + j)
        if num > 1e100 or den > 1e100 or num < 1e-100 or den < 1e-100:
            acc += math.log(num) - math.log(den)
            num = 1.0
            den = 1.0
    inv_beta = 1.0 / beta
    r = (n + alpha + beta) * inv_beta
    extra = alpha * ((n + alpha) * inv_beta / math.sqrt(r) + inv_beta)
    return acc, num / den, extra


@numba.njit(nogil=True)
def beta_binomial_pmf_test(x, n, alpha, beta, scoring):
    """
    The significance bound of the beta-binomial (see
    `_filter_by_significance_bound`): -log P(X = x), corrected, reaches
    -log(alpha).

    Rejected without lgamma when the upper bound of
    `beta_binomial_neglogpmf_upper` is below the threshold by more than a
    rounding margin, else decided with the exact `beta_binomial_logpmf`; the
    outcome is that of the exact test. The bound passes for about 1 in 200
    candidate families on Metazoa.
    """
    threshold = scoring.neglog_alpha + scoring.log_correction
    if 0 < x <= n and x <= PMF_BOUND_MAX_COUNT:
        acc, prod, extra = beta_binomial_neglogpmf_upper(
            x, float(n), alpha, beta
        )
        # The exact value is a sum of lgamma of magnitude up to about
        # (n + kappa) log(n + kappa), each within a few ulp; the margin
        # exceeds that rounding error by orders of magnitude (log(y) <
        # sqrt(y)), and the rounding of the bound as well.
        size = n + alpha + beta + 2.0
        margin = 1e-6 + 1e-12 * size * math.sqrt(size)
        if acc + math.log(prod) + extra < threshold - margin:
            return False
    neglogpmf = -beta_binomial_logpmf(float(x), float(n), alpha, beta)
    return neglogpmf - scoring.log_correction >= scoring.neglog_alpha


@numba.njit(nogil=True)
def _filter_by_significance_bound(qres, n, model, scoring):
    """
    Filters families by the count filter (`filter_family_candidates`), then
    by probabilistic bounds on p-value of the observed number of matches.
    Both in one pass: the beta-binomial q of a family serves both.
    """
    epsilon = 1e-10
    keep = np.full(len(qres), True)
    packed = has_packed_bbinom(model)
    log_n = math.log(float(n)) if n > 0 else 0.0
    n_q = model.q_coef.shape[1]
    n_kappa = model.kappa_coef.shape[1]

    for i in range(len(qres)):
        family_id = qres["id"][i]
        if packed:
            row = model.packed[family_id]
            is_bbinom = row[PACKED_SCALE] > 0.0
        else:
            is_bbinom = has_family_bbinom(family_id, model)
        if is_bbinom:
            # the count filter, as `compute_expected_counts`
            if packed:
                z = packed_bbinom_z(row, n, log_n)
                q = packed_bbinom_q(row, z, n_q)
            else:
                n_eval = bbinom_eval_n(family_id, n, model.n_min, model.n_max)
                alpha, beta, q = beta_binomial_params_for_n(
                    n_eval,
                    model.q_coef[family_id],
                    model.kappa_coef[family_id],
                    model.center[family_id],
                    model.scale[family_id],
                )
            if not qres["count"][i] >= q * n:
                keep[i] = False
                continue
            if packed:
                (alpha, beta) = packed_bbinom_params_q(row, z, q, n_q, n_kappa)
            # For the beta-binomial, compute the first member of the tail as
            # a bound to test significance.
            #    P(X = k) <= P(X >= k) <= alpha
            # we therefore test
            #    P(X = k) <= alpha
            #    -log P(X = k) >= -log(alpha)
            keep[i] = beta_binomial_pmf_test(
                qres["count"][i], n, alpha, beta, scoring
            )
        else:
            expected = model.family_probability[family_id]
            if not qres["count"][i] >= expected * n:
                keep[i] = False
                continue
            # For the binomial model, perform the Chernoff KL-div test.
            # The Chernoff upper bound for the binomial X:
            #     P(X >= k) <= exp(-n KL_div(k/n || p))
            #
            # The empirical proportion of observed k-mers, clipped for the
            # edge cases k = n, k = 0.
            observed = min(max(qres["count"][i] / n, epsilon), 1.0 - epsilon)
            kl_divergence = (
                observed * math.log(observed / expected)
                + (1.0 - observed)
                * math.log((1.0 - observed) / (1.0 - expected))
            )
            # Now we check P <= exp(-n KL_div(k/n || p)) <= alpha
            #                   KL_div(k/n || p) >= -log(alpha) / n.
            # and if it holds, we almost guarantee P <= alpha
            # There is a very small chance of that P <= alpha < bound, and then this test
            # fails with a false negative. In practice, it does not happen.
            keep[i] = kl_divergence > scoring.neglog_alpha / n

    return qres[keep]


@numba.njit(nogil=True)
def score_family_candidates(
    qres,
    n,
    model,
    scoring,
):
    """
    Apply the count filter (`filter_family_candidates`) and the fast bound,
    exact tail score, and final significance test.
    """
    qres = _filter_by_significance_bound(qres, n, model, scoring)
    if len(qres) == 0:
        return qres

    packed = has_packed_bbinom(model)
    log_n = math.log(float(n)) if n > 0 else 0.0
    n_q = model.q_coef.shape[1]
    n_kappa = model.kappa_coef.shape[1]
    for i in range(len(qres)):
        family_id = qres["id"][i]
        if packed:
            row = model.packed[family_id]
            if row[PACKED_SCALE] > 0.0:
                (alpha, beta) = packed_bbinom_params(row, n, log_n, n_q, n_kappa)
                neglog_tail = beta_binomial_neglogccdf(
                    qres["count"][i], n, alpha, beta
                )
            else:
                neglog_tail = binom_neglogccdf(
                    qres["count"][i], n, model.family_probability[family_id]
                )
        else:
            neglog_tail = family_neglogccdf(
                family_id,
                qres["count"][i],
                n,
                model,
            )
        qres["pvalue"][i] = min(
            MAX_FAMILY_NEGLOGP,
            max(0.0, neglog_tail - scoring.log_correction),
        )

    expected_count = compute_expected_counts(qres, n, model)
    qres["normcount"][:] = (qres["count"] - expected_count) / (n - expected_count)

    qres = qres[qres["pvalue"] >= scoring.neglog_alpha]
    if scoring.neglog_alpha <= 0.0:
        qres = qres[qres["pvalue"] > 0.0]
    return qres


@numba.njit(nogil=True)
def sigmoid(x):
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@numba.njit(nogil=True)
def beta_binomial_params_for_n(n, q_coef, kappa_coef, log_n_center, log_n_scale):
    z = (math.log(float(n)) - log_n_center) / log_n_scale

    q_linear = 0.0
    z_power = 1.0
    for i in range(q_coef.size):
        q_linear += q_coef[i] * z_power
        z_power *= z
    q = min(max(sigmoid(q_linear), 1e-8), 1.0 - 1e-8)

    log_kappa = 0.0
    z_power = 1.0
    for i in range(kappa_coef.size):
        log_kappa += kappa_coef[i] * z_power
        z_power *= z
    log_kappa = min(max(log_kappa, math.log(1e-4)), math.log(1e8))
    kappa = math.exp(log_kappa)

    alpha = q * kappa
    beta = (1.0 - q) * kappa
    return alpha, beta, q


@numba.njit(nogil=True)
def beta_binomial_logpmf(x, n, alpha, beta):
    return (
        math.lgamma(n + 1.0)
        - math.lgamma(x + 1.0)
        - math.lgamma(n - x + 1.0)
        + math.lgamma(x + alpha)
        + math.lgamma(n - x + beta)
        - math.lgamma(n + alpha + beta)
        + math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
    )


@numba.njit(nogil=True)
def beta_binomial_neglogccdf(x, n, alpha, beta):
    """
    Neg-log of the upper tail P(X >= x) of the Beta-Binomial(n, alpha, beta).

    We anchor on the single term P(x) -- one ``lgamma``-heavy call --
    and walk the rest of the tail with the closed-form PMF ratio

        P(k+1)/P(k) = (n - k)/(k + 1) * (k + alpha)/(n - k - 1 + beta)

    which costs a handful of multiplications per term. OMAmer only evaluates
    this in the upper tail (the count filter guarantees x >= expected mean,
    i.e. x is at or above the mode), so P(x) is the largest term and factoring
    it out keeps the running sum in [1, ~few) with no overflow. Once the tail
    is decaying and the next contribution is negligible we stop early.
    """
    if x <= 0:
        return 0.0
    if x > n:
        return np.inf

    log_px = beta_binomial_logpmf(float(x), float(n), alpha, beta)

    acc = 1.0
    term = 1.0
    for k in range(x, n):
        ratio = ((n - k) / (k + 1.0)) * ((k + alpha) / (n - k - 1.0 + beta))
        term *= ratio
        acc += term
        # tail is decaying and this term no longer moves the sum -> stop
        if ratio < 1.0 and term < acc * 1e-16:
            break

    return -(log_px + math.log(acc))
