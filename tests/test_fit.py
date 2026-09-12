import numpy as np
import pytest

from sciflow.errors import DataValidationError, FitError
from sciflow.fit import compare_models, fit


def test_linear_recovers_parameters(line_data):
    x, y, sigma = line_data
    result = fit(x, y, "linear", sigma)
    assert abs(result.params["a"] - 1.5) < 3 * result.errors["a"]
    assert abs(result.params["b"] - 2.0) < 3 * result.errors["b"]
    assert result.dof == 28
    assert 0.3 < result.chi2_reduced < 2.0
    assert 0 <= result.p_value <= 1
    assert result.method.startswith("linear least squares")


def test_linear_matches_closed_form(line_data):
    """Exact solver must agree with polyfit's weighted solution."""
    x, y, sigma = line_data
    result = fit(x, y, "linear", sigma)
    coeffs, cov = np.polyfit(x, y, 1, w=1 / sigma, cov="unscaled")
    assert np.allclose([result.params["b"], result.params["a"]], coeffs)
    assert np.allclose(np.sqrt(np.diag(cov))[::-1], list(result.errors.values()))


def test_unweighted_fit_scales_errors_by_scatter(line_data):
    x, y, _ = line_data
    result = fit(x, y, "linear")
    assert result.sigma is None
    assert np.isnan(result.p_value)
    # Errors from scatter should be of the same order as the true sigma=0.2 case.
    weighted = fit(x, y, "linear", np.full_like(x, 0.2))
    assert 0.5 < result.errors["b"] / weighted.errors["b"] < 2.0


def test_nonlinear_exponential(rng):
    x = np.linspace(0, 5, 50)
    sigma = np.full_like(x, 0.05)
    y = 3.0 * np.exp(-0.8 * x) + rng.normal(0, sigma)
    result = fit(x, y, "exponential", sigma)
    assert abs(result.params["a"] - 3.0) < 4 * result.errors["a"]
    assert abs(result.params["b"] + 0.8) < 4 * result.errors["b"]
    assert result.n_evaluations > 0


def test_gaussian_peak(rng):
    x = np.linspace(-5, 5, 80)
    sigma = np.full_like(x, 0.02)
    y = 2.0 * np.exp(-0.5 * ((x - 1.0) / 0.7) ** 2) + rng.normal(0, sigma)
    result = fit(x, y, "gaussian", sigma)
    assert abs(result.params["center"] - 1.0) < 0.05
    assert abs(result.params["sigma"] - 0.7) < 0.05


def test_custom_expression_via_string(rng):
    from sciflow.models import expression_model

    x = np.linspace(0, 10, 40)
    y = 5 * np.exp(-x / 2.5) + 0.8 + rng.normal(0, 0.02, x.size)
    model = expression_model("a*exp(-x/tau) + c", initial_guess=[4, 2, 0.5])
    result = fit(x, y, model)
    assert abs(result.params["tau"] - 2.5) < 0.1


def test_r_squared_none_when_y_constant():
    x = np.arange(5.0)
    y = np.full(5, 3.0)
    result = fit(x, y, "linear")
    assert result.r_squared is None


def test_validation_errors():
    x = np.arange(5.0)
    with pytest.raises(DataValidationError, match="non-finite"):
        fit(x, np.array([1, 2, np.nan, 4, 5.0]), "linear")
    with pytest.raises(DataValidationError, match="positive"):
        fit(x, x, "linear", np.array([1, 1, 0, 1, 1.0]))
    with pytest.raises(DataValidationError, match="more than"):
        fit(x[:2], x[:2], "linear")
    with pytest.raises(DataValidationError):
        fit(x, x[:3], "linear")


def test_rank_deficient_design_raises():
    x = np.full(6, 2.0)
    y = np.arange(6.0)
    with pytest.raises(FitError, match="rank-deficient"):
        fit(x, y, "linear")


def test_nonlinear_failure_is_fit_error():
    x = np.linspace(1, 3, 6)
    y = np.array([1, 2, 1, 2, 1, 2.0])
    with pytest.raises(FitError):
        fit(x, y, "logistic", maxfev=5)


def test_compare_models_sorted_by_aic(line_data):
    x, y, sigma = line_data
    results = compare_models(x, y, ["poly3", "linear", "proportional"], sigma)
    aics = [r.aic for r in results]
    assert aics == sorted(aics)
    assert results[0].model.name in {"linear", "poly3"}


def test_result_serialisable(line_data):
    import json

    x, y, sigma = line_data
    result = fit(x, y, "linear", sigma, x_name="t", y_name="v")
    payload = json.loads(json.dumps(result.as_dict()))
    assert payload["x"] == "t" and payload["weighted"] is True
    assert "chi2_reduced" in payload
    assert "R^2" in result.summary()


def test_diagnostics_flag_wrong_model(rng):
    from sciflow.fit import diagnostics

    x = np.linspace(0, 10, 40)
    sigma = np.full_like(x, 0.2)
    good = fit(x, 1 + 2 * x + rng.normal(0, sigma), "linear", sigma)
    bad = fit(x, 1 + 2 * x + 0.15 * x**2 + rng.normal(0, sigma), "linear", sigma)
    keys_bad = {c.key: c.status for c in diagnostics(bad)}
    assert keys_bad["chi2_reduced"] == "warn" and keys_bad["runs_test"] == "warn"
    n_warn_good = sum(c.status == "warn" for c in diagnostics(good))
    assert n_warn_good <= 1
    unweighted = fit(x, 1 + 2 * x + rng.normal(0, sigma), "linear")
    assert {c.key: c.status for c in diagnostics(unweighted)}["chi2_reduced"] == "info"


def test_odr_with_x_errors_recovers_parameters(rng):
    xt = np.linspace(0, 10, 40)
    sx = np.full_like(xt, 0.15)
    sy = np.full_like(xt, 0.3)
    x = xt + rng.normal(0, sx)
    y = 1.5 + 2.0 * xt + rng.normal(0, sy)
    result = fit(x, y, "linear", sy, sigma_x=sx)
    assert result.method.startswith("orthogonal distance")
    assert result.sigma_x is not None and result.sigma_eff is not None
    assert abs(result.params["b"] - 2.0) < 3 * result.errors["b"]
    assert 0.4 < result.chi2_reduced < 1.8
    # Effective sigma includes the slope-propagated x error.
    assert np.allclose(result.sigma_eff, np.sqrt(0.3**2 + (result.params["b"] * 0.15) ** 2), rtol=1e-3)
    # Ignoring the x errors must give smaller (too optimistic) parameter errors.
    naive = fit(x, y, "linear", sy)
    assert naive.errors["b"] < result.errors["b"]
    assert naive.method.startswith("linear least squares")  # unchanged behaviour without sigma_x
    assert result.as_dict()["x_errors"] is True and naive.as_dict()["x_errors"] is False


def test_odr_nonlinear_and_unweighted(rng):
    xt = np.linspace(0, 5, 40)
    sx = np.full_like(xt, 0.05)
    x = xt + rng.normal(0, sx)
    y = 3.0 * np.exp(-0.8 * xt) + rng.normal(0, 0.02, xt.size)
    r = fit(x, y, "exponential", sigma_x=sx)          # x errors only: unweighted in y
    assert r.method.startswith("orthogonal distance") and np.isnan(r.p_value)
    assert abs(r.params["b"] + 0.8) < 4 * r.errors["b"]


def test_odr_validation():
    x = np.arange(6.0)
    with pytest.raises(DataValidationError, match="positive"):
        fit(x, 2 * x, "linear", sigma_x=np.zeros(6))


def test_durbin_watson_interval_depends_on_n(rng):
    from sciflow.fit.diagnostics import diagnostics

    for n in (15, 200):
        x = np.linspace(0, 1, n)
        y = 1 + 2 * x + rng.normal(0, 0.1, n)
        result = fit(x, y, "linear", np.full(n, 0.1))
        dw = [c for c in diagnostics(result) if c.key == "durbin_watson"][0]
        half = 1.96 * 2 / np.sqrt(n)
        assert f"{2 - half:.2f} – {2 + half:.2f}" in dw.expected


def test_exact_data_does_not_warn_or_crash():
    import warnings

    x = np.arange(10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a log(0) RuntimeWarning would fail here
        result = fit(x, 3 * x + 1, "linear")
        constant = fit(x, np.full(10, 2.0), "linear")
    assert np.isclose(result.params["b"], 3) and np.isclose(result.params["a"], 1)
    assert constant.r_squared is None and np.isfinite(list(constant.params.values())).all()
