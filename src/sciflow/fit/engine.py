"""Single fitting engine for every model.

* Models that expose a design matrix are solved exactly by weighted linear
  least squares (no iteration, no starting values, no convergence issues).
* Everything else goes through ``scipy.optimize.curve_fit`` with the model's
  initial guess and explicit convergence checks.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Union

import numpy as np
from scipy import stats
from scipy.optimize import OptimizeWarning, curve_fit

from sciflow.errors import DataValidationError, FitError
from sciflow.fit.result import FitResult
from sciflow.models.base import Model
from sciflow.models.registry import get_model

ArrayLike = Union[Sequence[float], np.ndarray]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_arrays(
    x: ArrayLike,
    y: ArrayLike,
    sigma: Optional[ArrayLike],
    n_params: int,
    *,
    x_name: str = "x",
    y_name: str = "y",
    sigma_name: str = "sigma",
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return float arrays after checking shapes, finiteness and sigma > 0."""

    def to_array(values, label):
        try:
            arr = np.asarray(values, dtype=float).ravel()
        except (TypeError, ValueError) as error:
            raise DataValidationError(
                f"Column '{label}' contains non-numeric values."
            ) from error
        if not np.all(np.isfinite(arr)):
            raise DataValidationError(
                f"Column '{label}' contains missing or non-finite values. "
                "Run clean_data first."
            )
        return arr

    x_arr = to_array(x, x_name)
    y_arr = to_array(y, y_name)
    if x_arr.shape != y_arr.shape:
        raise DataValidationError(
            f"'{x_name}' has {x_arr.size} values but '{y_name}' has {y_arr.size}."
        )

    sigma_arr = None
    if sigma is not None:
        sigma_arr = to_array(sigma, sigma_name)
        if sigma_arr.shape != y_arr.shape:
            raise DataValidationError(
                f"'{sigma_name}' has {sigma_arr.size} values but '{y_name}' has {y_arr.size}."
            )
        if np.any(sigma_arr <= 0):
            raise DataValidationError(
                f"Column '{sigma_name}' must be strictly positive."
            )

    if x_arr.size <= n_params:
        raise DataValidationError(
            f"Need more than {n_params} points to fit {n_params} parameters "
            f"(got {x_arr.size})."
        )
    return x_arr, y_arr, sigma_arr


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #

def _solve_linear(model: Model, x, y, sigma):
    """Exact weighted least squares: minimise sum(((y - A p) / sigma)^2)."""
    design = model.design_matrix(x)
    weights = 1.0 / sigma if sigma is not None else np.ones_like(y)
    aw = design * weights[:, None]
    yw = y * weights

    params, _, rank, _ = np.linalg.lstsq(aw, yw, rcond=None)
    if rank < design.shape[1]:
        raise FitError(
            f"Design matrix of '{model.name}' is rank-deficient: the data "
            "cannot determine all parameters (e.g. all x identical)."
        )
    covariance = np.linalg.inv(aw.T @ aw)
    return params, covariance, "linear least squares (exact)", 1


def _solve_nonlinear(model: Model, x, y, sigma, p0, maxfev):
    if p0 is None:
        p0 = model.guess(x, y)
    p0 = np.asarray(p0, dtype=float)
    if p0.shape != (model.n_params,):
        raise FitError(
            f"p0 must have {model.n_params} values for model '{model.name}'."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        try:
            params, covariance, info, message, ier = curve_fit(
                model.func,
                x,
                y,
                p0=p0,
                sigma=sigma,
                absolute_sigma=sigma is not None,
                bounds=model.bounds,
                maxfev=maxfev,
                ftol=1e-12,
                xtol=1e-12,
                full_output=True,
            )
        except OptimizeWarning as warning:
            raise FitError(
                f"Covariance could not be estimated for '{model.name}' "
                f"({warning}). The model is probably over-parameterised for "
                "this data or the fit hit a boundary."
            ) from warning
        except RuntimeError as error:
            raise FitError(
                f"Fit of '{model.name}' did not converge: {error}. "
                "Try a better initial guess (p0) or a different model."
            ) from error
        except (TypeError, ValueError) as error:
            raise FitError(f"Fit of '{model.name}' failed: {error}") from error

    if not np.all(np.isfinite(covariance)):
        raise FitError(
            f"Covariance of '{model.name}' contains non-finite values; "
            "parameters are not identifiable from this data."
        )
    n_eval = int(info.get("nfev", 0)) if isinstance(info, dict) else 0
    return params, covariance, "Levenberg-Marquardt (scipy.curve_fit)", n_eval


def _solve_odr(model: Model, x, y, sigma_y, sigma_x, p0, maxfev):
    """Orthogonal distance regression for data with x uncertainties.

    Same formulation as ODRPACK's explicit ODR: the true abscissae x* = x + d
    are extra unknowns and the objective is

        sum_i ((y_i - f(x_i + d_i; beta)) / sy_i)^2 + (d_i / sx_i)^2

    solved with scipy.optimize.least_squares (trust-region reflective, sparse
    Jacobian). Implemented here rather than with ``scipy.odr`` because that
    module is deprecated (scipy >= 1.17) and scheduled for removal.

    Returns params, the *unscaled* covariance of beta (absolute-sigma
    semantics when sigma_y is given), the weighted sum of squares at the
    optimum, and the effective per-point sigma sqrt(sy^2 + (f'(x) sx)^2).
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    n, p = x.size, model.n_params
    if p0 is None:
        p0 = _solve_linear(model, x, y, sigma_y)[0] if model.is_linear else model.guess(x, y)
    p0 = np.asarray(p0, dtype=float)
    if p0.shape != (p,):
        raise FitError(f"p0 must have {p} values for model '{model.name}'.")

    sy = sigma_y if sigma_y is not None else np.ones_like(y)
    sx = sigma_x

    def residuals(z):
        beta, d = z[:p], z[p:]
        return np.concatenate([(y - model.func(x + d, *beta)) / sy, d / sx])

    sparsity = lil_matrix((2 * n, p + n), dtype=int)
    sparsity[:n, :p] = 1
    sparsity[:n, p:] = np.eye(n, dtype=int)
    sparsity[n:, p:] = np.eye(n, dtype=int)

    z0 = np.concatenate([p0, np.zeros(n)])
    try:
        sol = least_squares(
            residuals, z0, jac_sparsity=sparsity, method="trf", x_scale="jac",
            ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=max(maxfev, 2000),
        )
    except (ValueError, FloatingPointError) as error:
        raise FitError(f"ODR fit of '{model.name}' failed: {error}") from error
    if not sol.success or not np.all(np.isfinite(sol.x)):
        raise FitError(
            f"ODR fit of '{model.name}' did not converge ({sol.message}). "
            "Try a better initial guess (p0) or a different model."
        )

    params = sol.x[:p]
    # Covariance of (beta, d) from the Gauss-Newton approximation, beta block.
    jac = sol.jac.toarray() if hasattr(sol.jac, "toarray") else np.asarray(sol.jac)
    try:
        cov_full = np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError as error:
        raise FitError(
            f"Covariance of '{model.name}' is singular (ODR); parameters are not identifiable."
        ) from error
    covariance = cov_full[:p, :p]
    if not np.all(np.isfinite(covariance)):
        raise FitError(f"Covariance of '{model.name}' contains non-finite values (ODR).")

    sum_square = float(2 * sol.cost)
    # Effective sigma of each point: propagate sigma_x through the local slope.
    eps = 1e-6 * np.maximum(np.abs(x), 1.0)
    slope = (model.func(x + eps, *params) - model.func(x - eps, *params)) / (2 * eps)
    sy0 = sigma_y if sigma_y is not None else np.zeros_like(x)
    sigma_eff = np.sqrt(sy0**2 + (slope * sx) ** 2)
    return (params, covariance, sum_square, sigma_eff,
            "orthogonal distance regression (least squares in x and y)", int(sol.nfev))


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def fit(
    x: ArrayLike,
    y: ArrayLike,
    model: Union[str, Model] = "linear",
    sigma: Optional[ArrayLike] = None,
    *,
    sigma_x: Optional[ArrayLike] = None,
    p0: Optional[Sequence[float]] = None,
    maxfev: int = 10_000,
    x_name: str = "x",
    y_name: str = "y",
) -> FitResult:
    """Fit ``model`` to ``(x, y)`` optionally weighted by ``sigma`` (and ``sigma_x``).

    Parameters
    ----------
    x, y
        Data arrays (or DataFrame columns).
    model
        Registered model name (``"linear"``, ``"gaussian"``, ``"poly3"``...)
        or a :class:`Model` instance.
    sigma
        1-sigma uncertainty of each ``y``. When given, the fit is weighted,
        the covariance is absolute and chi-square statistics are meaningful.
    sigma_x
        1-sigma uncertainty of each ``x``. When given, the fit switches to
        orthogonal distance regression (ODRPACK formulation), which accounts for errors
        in both variables. Without it, behaviour is exactly the classic
        least-squares fit.
    p0
        Starting parameters for non-linear models (overrides the model guess).
    maxfev
        Maximum function evaluations for the iterative solver.
    x_name, y_name
        Column labels used in reports and plots.

    Raises
    ------
    DataValidationError
        Bad input data.
    FitError
        The solver failed or the parameters are not identifiable.
    """
    if isinstance(model, str):
        model = get_model(model)

    x_arr, y_arr, sigma_arr = validate_arrays(
        x, y, sigma, model.n_params, x_name=x_name, y_name=y_name
    )
    sigma_x_arr = None
    if sigma_x is not None:
        _, _, sigma_x_arr = validate_arrays(
            x_arr, y_arr, sigma_x, model.n_params, x_name=x_name, y_name=y_name,
            sigma_name=f"sigma_{x_name}",
        )

    if model.prepare is not None:
        model.prepare(x_arr, y_arr)

    sigma_eff = None
    odr_ss = None
    if sigma_x_arr is not None:
        params, covariance, odr_ss, sigma_eff, method, n_eval = _solve_odr(
            model, x_arr, y_arr, sigma_arr, sigma_x_arr, p0, maxfev
        )
    elif model.is_linear:
        params, covariance, method, n_eval = _solve_linear(model, x_arr, y_arr, sigma_arr)
    else:
        params, covariance, method, n_eval = _solve_nonlinear(
            model, x_arr, y_arr, sigma_arr, p0, maxfev
        )

    fitted = model(x_arr, *params)
    residuals = y_arr - fitted
    n, p = x_arr.size, model.n_params
    dof = n - p

    ss_res = float(np.sum(residuals**2))
    rmse = float(np.sqrt(ss_res / n))
    residual_sd = float(np.sqrt(ss_res / dof))

    if sigma_arr is not None:
        # With x errors, chi2 is the ODR weighted sum of squares (orthogonal
        # distances); otherwise the classic sum of ((y - f) / sigma)^2.
        chi2 = odr_ss if odr_ss is not None else float(np.sum((residuals / sigma_arr) ** 2))
        p_value = float(stats.chi2.sf(chi2, dof))
        # Gaussian likelihood with known sigma (constants dropped).
        aic = chi2 + 2 * p
        bic = chi2 + p * np.log(n)
    else:
        # Unweighted: rescale covariance by the residual variance so the
        # parameter errors reflect the scatter of the data.
        variance = ss_res / dof
        if model.is_linear and sigma_x_arr is None:
            covariance = covariance * variance
        elif sigma_x_arr is not None:
            covariance = covariance * (odr_ss / dof)
        chi2 = ss_res
        p_value = float("nan")
        # A perfect fit (ss_res = 0) would give log(0); report -inf explicitly.
        with np.errstate(divide="ignore"):
            log_term = n * np.log(ss_res / n) if ss_res > 0 else -np.inf
        aic = log_term + 2 * p
        bic = log_term + p * np.log(n)

    chi2_reduced = chi2 / dof

    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    errors = np.sqrt(np.diag(covariance))
    derived = _derived_quantities(model, params, covariance)

    return FitResult(
        model=model,
        params=dict(zip(model.param_names, map(float, params))),
        errors=dict(zip(model.param_names, map(float, errors))),
        covariance=np.asarray(covariance, dtype=float),
        x=x_arr,
        y=y_arr,
        sigma=sigma_arr,
        sigma_x=sigma_x_arr,
        sigma_eff=sigma_eff,
        derived=derived,
        fitted=fitted,
        residuals=residuals,
        n_points=n,
        n_params=p,
        dof=dof,
        chi2=chi2,
        chi2_reduced=float(chi2_reduced),
        p_value=p_value,
        r_squared=r_squared,
        rmse=rmse,
        residual_sd=residual_sd,
        aic=float(aic),
        bic=float(bic),
        method=method,
        n_evaluations=n_eval,
        x_name=x_name,
        y_name=y_name,
    )


def _derived_quantities(model: Model, params, covariance) -> dict:
    """Evaluate ``model.derived`` and propagate the covariance (first order)."""
    if model.derived is None:
        return {}
    params = np.asarray(params, dtype=float)
    names = model.param_names

    def evaluate(vec):
        out = model.derived(dict(zip(names, vec)))
        return {k: float(v) for k, v in out.items()}

    try:
        base = evaluate(params)
    except Exception:  # a derived quantity undefined for these values
        return {}
    keys = list(base)
    jac = np.zeros((len(keys), params.size))
    for j in range(params.size):
        step = 1e-6 * max(1.0, abs(params[j]))
        plus, minus = params.copy(), params.copy()
        plus[j] += step
        minus[j] -= step
        try:
            vp, vm = evaluate(plus), evaluate(minus)
        except Exception:
            return {}
        jac[:, j] = [(vp[k] - vm[k]) / (2 * step) for k in keys]
    var = np.einsum("ij,jk,ik->i", jac, covariance, jac)
    return {k: (base[k], float(np.sqrt(max(v, 0.0)))) for k, v in zip(keys, var)}


def compare_models(
    x: ArrayLike,
    y: ArrayLike,
    models: Sequence[Union[str, Model]],
    sigma: Optional[ArrayLike] = None,
    **kwargs,
) -> list[FitResult]:
    """Fit several models and return the results sorted by AIC (best first).

    Models that fail to fit are skipped; if none succeed a FitError is raised.
    """
    results: list[FitResult] = []
    failures: list[str] = []
    for m in models:
        try:
            results.append(fit(x, y, m, sigma, **kwargs))
        except (FitError, DataValidationError) as error:
            name = m if isinstance(m, str) else m.name
            failures.append(f"{name}: {error}")
    if not results:
        raise FitError("No model could be fitted:\n" + "\n".join(failures))
    results.sort(key=lambda r: r.aic)
    for r in results:
        r.extra["compare_failures"] = failures
    return results
