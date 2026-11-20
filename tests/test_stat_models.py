from types import SimpleNamespace

import numba
import numpy as np
import pytest

from omamer.stat_models import (
    FamilyModel,
    FamilyModelParameters,
    FamilyScoringParameters,
    filter_family_candidates,
    get_family_model,
    learn_index_models,
    make_family_model,
    score_family_candidates,
)

QUERY_RESULT_DTYPE = np.dtype(
    [
        ("id", np.uint32),
        ("pvalue", np.float64),
        ("count", np.uint32),
        ("normcount", np.float64),
        ("overlap", np.float64),
    ]
)


def test_family_model_factory():
    probability = np.array([0.01])
    assert get_family_model("auto", False) is FamilyModel.BINOMIAL
    assert get_family_model("auto", True) is FamilyModel.BETA_BINOMIAL
    model = make_family_model("binomial", probability)
    assert model.kind is FamilyModel.BINOMIAL
    assert model.family_probability is probability

    valid = np.array([True])
    model = make_family_model(
        "bbinom",
        probability,
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.zeros(1),
        np.ones(1),
        valid,
        np.ones(1, dtype=np.uint32),
        np.ones(1, dtype=np.uint32),
    )
    assert model.kind is FamilyModel.BETA_BINOMIAL
    assert model.valid is valid

    with pytest.raises(ValueError, match="requires beta-binomial"):
        make_family_model("bbinom", probability)


def test_family_model_enum_compiles_in_parallel_nogil():
    @numba.njit(parallel=True, nogil=True)
    def apply(model, result):
        for i in numba.prange(result.size):
            result[i] = model == FamilyModel.BETA_BINOMIAL

    result = np.zeros(8, dtype=np.bool_)
    apply(FamilyModel.BETA_BINOMIAL, result)

    assert np.all(result)
    assert apply.nopython_signatures


def test_index_model_learning_uses_modality_specific_n_values(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "omamer.stat_models._write_binomial_model",
        lambda *args, **kwargs: None,
    )

    def fake_buffer(db, sequence_buffer, **kwargs):
        calls.append(("buffer", sequence_buffer, kwargs))
        return None

    monkeypatch.setattr(
        "omamer.bbinom_fit.fit_bbinom_coefficients_from_buffer",
        fake_buffer,
    )
    monkeypatch.setattr(
        "omamer.bbinom_coefficients.store_bbinom_coefficients",
        lambda *args, **kwargs: None,
    )

    class FakeIndexGroup:
        def __init__(self):
            self.attrs = {}

        def _f_setattr(self, name, value):
            self.attrs[name] = value

    index_group = FakeIndexGroup()
    data = {
        modality: SimpleNamespace(
            sequence_buffer="{}-buffer".format(modality),
            table_index=np.asarray([0], dtype=np.uint32),
            table_buffer=np.empty(0, dtype=np.uint32),
        )
        for modality in ("seq", "ss")
    }
    db = SimpleNamespace(ki=SimpleNamespace(kmer_percentage=100.0))

    learn_index_models(
        db,
        index_group,
        data,
        models=("binomial", "beta-binomial"),
        bbinom_options={
            "n_values_by_modality": {"ss": "50,100"},
        },
    )

    assert [call[:2] for call in calls] == [
        ("buffer", "seq-buffer"),
        ("buffer", "ss-buffer"),
    ]
    assert "n_values" not in calls[0][2]
    assert calls[1][2]["n_values"] == "50,100"
    for _, _, kwargs in calls:
        assert "n_values_by_modality" not in kwargs
    assert index_group.attrs["models"] == "binomial,beta-binomial"


def test_family_candidate_operations_dispatch_and_fallback():
    qres = np.zeros(2, dtype=QUERY_RESULT_DTYPE)
    qres["id"] = [0, 1]
    qres["count"] = [20, 10]
    probabilities = np.array([0.01, 0.01])
    model = FamilyModelParameters(
        FamilyModel.BETA_BINOMIAL,
        probabilities,
        np.array([[-2.0], [-2.0]]),
        np.array([[3.0], [3.0]]),
        np.zeros(2),
        np.ones(2),
        np.array([True, False]),
        np.ones(2, dtype=np.uint32),
        np.full(2, 100, dtype=np.uint32),
    )

    filtered = filter_family_candidates(qres, 100, model)
    scored = score_family_candidates(
        filtered,
        100,
        model,
        FamilyScoringParameters(0.0, 0.0),
    )

    np.testing.assert_array_equal(scored["id"], [0, 1])
    assert np.all(scored["pvalue"] > 0.0)
    assert np.all(scored["normcount"] > 0.0)
    assert filter_family_candidates.nopython_signatures
    assert score_family_candidates.nopython_signatures
    assert filter_family_candidates.targetoptions["nogil"]
    assert score_family_candidates.targetoptions["nogil"]


@pytest.mark.parametrize("n_q, n_kappa", [(3, 2), (1, 1), (6, 5)])
def test_packed_beta_binomial_model_is_bit_identical(n_q, n_kappa):
    """The packed coefficient rows give the same floats as the arrays."""
    rng = np.random.default_rng(5)
    n_families = 300
    valid = rng.random(n_families) < 0.8
    scale = rng.uniform(0.5, 2.0, n_families)
    scale[:5] = 0.0  # disabled fits
    n_min = rng.integers(5, 60, n_families).astype(np.uint32)
    n_max = (n_min + rng.integers(0, 400, n_families)).astype(np.uint32)
    n_max[5:20] = 0  # no clamp
    model = make_family_model(
        "beta-binomial",
        rng.uniform(1e-4, 0.05, n_families),
        rng.normal(-3.0, 0.3, (n_families, n_q)) / np.arange(1, n_q + 1),
        rng.normal(3.0, 0.3, (n_families, n_kappa)) / np.arange(1, n_kappa + 1),
        rng.uniform(3.0, 6.0, n_families),
        scale,
        valid,
        n_min,
        n_max,
    )
    assert model.packed.shape[0] == n_families
    unpacked = model._replace(packed=np.empty((0, 0), dtype=np.float64))
    qres = np.zeros(n_families, dtype=QUERY_RESULT_DTYPE)
    qres["id"] = rng.permutation(n_families)
    for n in (1, 7, 40, 150, 1000):
        qres["count"] = rng.integers(0, n + 1, n_families)
        for scoring in (
            FamilyScoringParameters(0.0, 0.0),
            FamilyScoringParameters(-np.log(1e-6), np.log(n_families)),
        ):
            for search_model in (model, unpacked):
                filtered = filter_family_candidates(qres.copy(), n, search_model)
                scored = score_family_candidates(
                    filtered.copy(), n, search_model, scoring
                )
                if search_model is model:
                    expected = (filtered, scored)
            assert filtered.tobytes() == expected[0].tobytes()
            assert scored.tobytes() == expected[1].tobytes()
