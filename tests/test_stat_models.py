import numba
import numpy as np
import pytest

from omamer.stat_models import (
    FamilyModel,
    FamilyModelParameters,
    FamilyScoringParameters,
    filter_family_candidates,
    get_family_model,
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
