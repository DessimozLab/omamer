"""
Import precomputed beta-binomial family model coefficients into an OMAmer DB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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
        valid[family] = True
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

    return int(valid.sum())


def _coefficient_kmer_percentage(rows, modality):
    """Return the common 3Di filter setting recorded by fitted coefficients."""
    if modality != "ss" or "kmer_percentage" not in rows.columns:
        # Coefficients written before the information filter were unfiltered.
        return 100.0
    percentages = rows["kmer_percentage"].dropna().astype(float).unique()
    if len(percentages) != 1 or not 0.0 < percentages[0] <= 100.0:
        raise ValueError(
            "All ss coefficient rows must have one kmer_percentage in (0, 100]"
        )
    return float(percentages[0])


def import_bbinom_coefficients(db, coefficient_path):
    if "/Index" not in db.db:
        raise ValueError("Database has no /Index group")

    df = read_bbinom_coefficients(coefficient_path)
    index_group = db.db.root.Index
    written = {}
    for modality, rows in df.groupby("modality", sort=True):
        written[modality] = _write_modality_coefficients(db, index_group, modality, rows)
        if modality == "ss":
            index_group._f_setattr(
                "ss_bbinom_kmer_percentage", _coefficient_kmer_percentage(rows, modality)
            )

    index_group._f_setattr("bbinom_model", "length_aware_beta_binomial")
    index_group._f_setattr("bbinom_q_degree", 2)
    index_group._f_setattr("bbinom_kappa_degree", 1)
    return written
