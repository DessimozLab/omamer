"""
Import precomputed beta-binomial family model coefficients into an OMAmer DB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


BBINOM_COEFFICIENT_BOUND = 50.0
BBINOM_COEFFICIENT_BOUND_TOLERANCE = 1e-3
BBINOM_VALIDITY_POLICY_VERSION = 1
_THETA0_LOG_KAPPA = np.log(1000.0)


BBINOM_NODE_NAMES = {
    "seq": {
        "q_coef": "FamilyBBinomQCoef",
        "kappa_coef": "FamilyBBinomKappaCoef",
        "center": "FamilyBBinomLogNCenter",
        "scale": "FamilyBBinomLogNScale",
        "valid": "FamilyBBinomValid",
        "n_min": "FamilyBBinomNTrainMin",
        "n_max": "FamilyBBinomNTrainMax",
    },
    "ss": {
        "q_coef": "SSFamilyBBinomQCoef",
        "kappa_coef": "SSFamilyBBinomKappaCoef",
        "center": "SSFamilyBBinomLogNCenter",
        "scale": "SSFamilyBBinomLogNScale",
        "valid": "SSFamilyBBinomValid",
        "n_min": "SSFamilyBBinomNTrainMin",
        "n_max": "SSFamilyBBinomNTrainMax",
    },
}

REQUIRED_COLUMNS = {
    "family_offset",
    "modality",
    "log_n_center",
    "log_n_scale",
    "q_coef_0",
    "q_coef_1",
    "q_coef_2",
    "kappa_coef_0",
    "kappa_coef_1",
}


def _parse_bool(value, default=True):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError("Cannot interpret boolean value {!r}".format(value))


def bbinom_coefficient_valid_mask(
    valid,
    q_coef,
    kappa_coef,
    n_center,
    n_scale,
    *,
    reject_initial_fallback=False,
):
    """Apply numerical and legacy-fit quality checks to a validity mask."""
    valid = np.asarray(valid, dtype=bool).copy()
    q_coef = np.asarray(q_coef, dtype=np.float64)
    kappa_coef = np.asarray(kappa_coef, dtype=np.float64)
    n_center = np.asarray(n_center, dtype=np.float64)
    n_scale = np.asarray(n_scale, dtype=np.float64)

    finite = (
        np.all(np.isfinite(q_coef), axis=1)
        & np.all(np.isfinite(kappa_coef), axis=1)
        & np.isfinite(n_center)
        & np.isfinite(n_scale)
        & (n_scale > 0.0)
    )
    at_bound = (
        np.any(
            np.abs(q_coef)
            >= BBINOM_COEFFICIENT_BOUND - BBINOM_COEFFICIENT_BOUND_TOLERANCE,
            axis=1,
        )
        | np.any(
            np.abs(kappa_coef)
            >= BBINOM_COEFFICIENT_BOUND - BBINOM_COEFFICIENT_BOUND_TOLERANCE,
            axis=1,
        )
    )
    valid &= finite & ~at_bound

    if reject_initial_fallback:
        initial_fallback = (
            np.isclose(q_coef[:, 1], 0.0, rtol=0.0, atol=1e-12)
            & np.isclose(q_coef[:, 2], 0.0, rtol=0.0, atol=1e-12)
            & np.isclose(
                kappa_coef[:, 0],
                _THETA0_LOG_KAPPA,
                rtol=0.0,
                atol=1e-12,
            )
            & np.isclose(kappa_coef[:, 1], 0.0, rtol=0.0, atol=1e-12)
        )
        valid &= ~initial_fallback

    return valid


def _coefficient_row_is_valid(row):
    explicit_valid = _parse_bool(
        getattr(row, "fit_valid", None),
        default=True,
    )
    optimizer_success = _parse_bool(
        getattr(row, "optimizer_success", None),
        default=True,
    )
    coefficients = np.asarray(
        [
            row.q_coef_0,
            row.q_coef_1,
            row.q_coef_2,
            row.kappa_coef_0,
            row.kappa_coef_1,
        ],
        dtype=np.float64,
    )
    return bool(
        explicit_valid
        and optimizer_success
        and np.all(np.isfinite(coefficients))
        and np.isfinite(row.log_n_center)
        and np.isfinite(row.log_n_scale)
        and float(row.log_n_scale) > 0.0
        and np.all(
            np.abs(coefficients)
            < BBINOM_COEFFICIENT_BOUND - BBINOM_COEFFICIENT_BOUND_TOLERANCE
        )
    )


def read_bbinom_coefficients(path):
    df = pd.read_csv(path, sep=None, engine="python")
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError("Missing coefficient columns: {}".format(", ".join(sorted(missing))))

    df = df.copy()
    df["modality"] = df["modality"].astype(str)
    bad_modalities = sorted(set(df["modality"]).difference(BBINOM_NODE_NAMES))
    if bad_modalities:
        raise ValueError("Unknown modalities in coefficient file: {}".format(", ".join(bad_modalities)))

    if df.duplicated(["modality", "family_offset"]).any():
        dup = df.loc[df.duplicated(["modality", "family_offset"], keep=False), ["modality", "family_offset"]]
        raise ValueError("Duplicate coefficient rows: {}".format(dup.drop_duplicates().to_dict("records")[:5]))

    return df


def _remove_existing(index_group, names):
    h5 = index_group._v_file
    for name in names:
        path = index_group._v_pathname + "/" + name
        if path in h5:
            h5.remove_node(index_group, name)


def _write_modality_coefficients(db, index_group, modality, rows):
    names = BBINOM_NODE_NAMES[modality]
    n_families = db.family_table.nrows

    q_coef = np.full((n_families, 3), np.nan, dtype=np.float64)
    kappa_coef = np.full((n_families, 2), np.nan, dtype=np.float64)
    center = np.full(n_families, np.nan, dtype=np.float64)
    scale = np.full(n_families, np.nan, dtype=np.float64)
    valid = np.zeros(n_families, dtype=np.bool_)
    n_min = np.zeros(n_families, dtype=np.uint32)
    n_max = np.zeros(n_families, dtype=np.uint32)

    rejected = 0
    for row in rows.itertuples(index=False):
        family = int(row.family_offset)
        if family < 0 or family >= n_families:
            raise ValueError("family_offset {} is outside DB family range 0..{}".format(family, n_families - 1))
        log_n_scale = float(row.log_n_scale)
        if not np.isfinite(log_n_scale) or log_n_scale <= 0:
            raise ValueError("Invalid log_n_scale for family_offset {}".format(family))

        q_coef[family, :] = [float(row.q_coef_0), float(row.q_coef_1), float(row.q_coef_2)]
        kappa_coef[family, :] = [float(row.kappa_coef_0), float(row.kappa_coef_1)]
        center[family] = float(row.log_n_center)
        scale[family] = log_n_scale
        row_valid = _coefficient_row_is_valid(row)
        valid[family] = row_valid
        rejected += int(not row_valid)
        if hasattr(row, "n_train_min") and not pd.isna(row.n_train_min):
            n_min[family] = int(row.n_train_min)
        if hasattr(row, "n_train_max") and not pd.isna(row.n_train_max):
            n_max[family] = int(row.n_train_max)

    _remove_existing(index_group, names.values())
    filters = db.compression_filters
    h5 = db.db
    h5.create_carray(index_group, names["q_coef"], obj=q_coef, filters=filters)
    h5.create_carray(index_group, names["kappa_coef"], obj=kappa_coef, filters=filters)
    h5.create_carray(index_group, names["center"], obj=center, filters=filters)
    h5.create_carray(index_group, names["scale"], obj=scale, filters=filters)
    h5.create_carray(index_group, names["valid"], obj=valid, filters=filters)
    h5.create_carray(index_group, names["n_min"], obj=n_min, filters=filters)
    h5.create_carray(index_group, names["n_max"], obj=n_max, filters=filters)

    valid_count = int(valid.sum())
    unfitted_count = n_families - len(rows)
    return valid_count, rejected, unfitted_count


def _coefficient_kmer_percentage(rows):
    """Return the common build-time k-mer filter recorded by coefficients."""
    if "kmer_percentage" not in rows.columns:
        # Coefficients written before the information filter were unfiltered.
        return 100.0
    percentages = rows["kmer_percentage"].dropna().astype(float).unique()
    if len(percentages) != 1 or not 0.0 < percentages[0] <= 100.0:
        raise ValueError(
            "All coefficient rows for a modality must have one "
            "kmer_percentage in (0, 100]"
        )
    return float(percentages[0])


def store_bbinom_coefficients(
    db,
    coefficients,
    modalities=None,
    kmer_percentage=None,
):
    """Persist fitted coefficient rows without an intermediate text file."""
    if "/Index" not in db.db:
        raise ValueError("Database has no /Index group")

    df = coefficients.copy()
    index_group = db.db.root.Index
    db_percentage = float(
        getattr(index_group._v_attrs, "kmer_percentage", 100.0)
    )
    if kmer_percentage is not None and not np.isclose(
        float(kmer_percentage),
        db_percentage,
    ):
        raise ValueError(
            "Beta-binomial coefficients use kmer_percentage={}, but the "
            "database was built with kmer_percentage={}".format(
                kmer_percentage,
                db_percentage,
            )
        )

    grouped = {
        modality: rows
        for modality, rows in df.groupby("modality", sort=True)
    } if "modality" in df.columns else {}
    if modalities is None:
        modalities = tuple(sorted(grouped))
    else:
        modalities = tuple(dict.fromkeys(modalities))
    unknown = sorted(set(modalities).difference(BBINOM_NODE_NAMES))
    if unknown:
        raise ValueError(
            "Unknown modalities: {}".format(", ".join(unknown))
        )

    percentages = {
        modality: (
            _coefficient_kmer_percentage(grouped[modality])
            if modality in grouped and len(grouped[modality])
            else db_percentage
        )
        for modality in modalities
    }
    for modality, percentage in percentages.items():
        if not np.isclose(percentage, db_percentage):
            raise ValueError(
                "{} beta-binomial coefficients use kmer_percentage={}, "
                "but the database was built with kmer_percentage={}".format(
                    modality, percentage, db_percentage
                )
            )

    written = {}
    for modality in modalities:
        rows = grouped.get(modality, df.iloc[0:0])
        valid_count, invalid_count, unfitted_count = _write_modality_coefficients(
            db,
            index_group,
            modality,
            rows,
        )
        written[modality] = valid_count
        index_group._f_setattr(
            "{}_bbinom_kmer_percentage".format(modality),
            percentages[modality],
        )
        index_group._f_setattr(
            "{}_bbinom_validity_policy_version".format(modality),
            BBINOM_VALIDITY_POLICY_VERSION,
        )
        index_group._f_setattr(
            "{}_bbinom_valid_count".format(modality),
            valid_count,
        )
        index_group._f_setattr(
            "{}_bbinom_invalid_count".format(modality),
            invalid_count,
        )
        index_group._f_setattr(
            "{}_bbinom_unfitted_count".format(modality),
            unfitted_count,
        )

    index_group._f_setattr("bbinom_model", "length_aware_beta_binomial")
    index_group._f_setattr("bbinom_q_degree", 2)
    index_group._f_setattr("bbinom_kappa_degree", 1)
    return written
