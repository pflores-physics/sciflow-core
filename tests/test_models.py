import numpy as np
import pytest

from sciflow.errors import ModelError
from sciflow.models import Model, expression_model, get_model, list_models, polynomial, register_model


def test_builtin_names_resolve():
    for model in list_models():
        assert get_model(model.name) is model


def test_polyN_generated_on_demand():
    model = get_model("poly5")
    assert model.n_params == 6 and model.is_linear
    x = np.array([0.0, 1.0, 2.0])
    assert np.allclose(model(x, 1, 0, 0, 0, 0, 1), [1, 2, 33])


def test_unknown_model():
    with pytest.raises(ModelError, match="Unknown model"):
        get_model("does_not_exist")


def test_wrong_param_count():
    with pytest.raises(ModelError):
        get_model("linear")(np.array([1.0]), 1.0)


def test_expression_model_infers_parameters():
    model = expression_model("a*exp(-x/tau) + c")
    assert model.param_names == ["a", "tau", "c"]
    x = np.array([0.0, 1.0])
    assert np.allclose(model(x, 2.0, 1.0, 0.5), [2.5, 2 * np.exp(-1) + 0.5])


def test_expression_caret_means_power():
    model = expression_model("k*x^2", ["k"])
    assert np.allclose(model(np.array([3.0]), 2.0), [18.0])


def test_expression_rejects_unknown_names():
    with pytest.raises(ModelError):
        expression_model("a*__import__('os').system('x')", ["a"])
    with pytest.raises(ModelError):
        expression_model("a*foo(x)", ["a"])


def test_expression_needs_parameters():
    with pytest.raises(ModelError):
        expression_model("sin(x)")


def test_register_custom_model():
    model = Model("halfline", lambda x, m: m * x / 2, ["m"], "y = m*x/2")
    register_model(model)
    assert get_model("halfline") is model
    with pytest.raises(ModelError):
        register_model(model)


def test_polynomial_degree_check():
    with pytest.raises(ValueError):
        polynomial(0)


def test_guess_never_returns_nan():
    x = np.linspace(1, 5, 10)
    y = np.zeros_like(x)      # degenerate data
    for model in list_models():
        p0 = model.guess(x, y)
        assert p0.shape == (model.n_params,)
        assert np.all(np.isfinite(p0))


def test_multipeak_recovers_three_gaussians():
    from sciflow.fit import fit

    rng = np.random.default_rng(5)
    x = np.linspace(400, 700, 300)
    g = lambda x, A, x0, w: A * np.exp(-0.5 * ((x - x0) / w) ** 2)
    y = g(x, 1.0, 470, 8) + g(x, 0.6, 520, 12) + g(x, 0.8, 610, 6) + 0.05 + 2e-4 * (x - 400)
    y += rng.normal(0, 0.01, x.size)
    r = fit(x, y, "gaussian3+linear", np.full_like(x, 0.01))
    centres = sorted(r.params[f"x0{i}"] for i in (1, 2, 3))
    assert np.allclose(centres, [470, 520, 610], atol=0.5)
    assert set(r.derived) == {f"{q}_{i}" for q in ("FWHM", "area") for i in (1, 2, 3)}
    assert r.derived["area_1"][0] == pytest.approx(1.0 * 8 * np.sqrt(2 * np.pi), rel=0.02)
    assert 0.5 < r.chi2_reduced < 1.5


def test_multipeak_names_and_errors():
    m = get_model("lorentzian2+constant")
    assert m.param_names == ["A1", "x01", "w1", "A2", "x02", "w2", "bg0"]
    assert get_model("GAUSSIAN1").n_params == 3
    with pytest.raises(ModelError):
        get_model("gaussian0")
    with pytest.raises(ModelError):
        get_model("gaussian2+quadratic")


def test_derived_quantities_with_errors():
    from sciflow.fit import fit

    x = np.linspace(0, 10, 40)
    y = 3 * np.exp(-0.5 * x)
    r = fit(x, y + 1e-4 * np.sin(x), "exponential")
    tau, tau_err = r.derived["tau = -1/b"]
    assert tau == pytest.approx(2.0, rel=1e-3)
    # First-order propagation: sigma_tau = sigma_b / b^2
    assert tau_err == pytest.approx(r.errors["b"] / r.params["b"] ** 2, rel=1e-4)
    assert "derived" in r.as_dict() and "tau" in r.summary()
