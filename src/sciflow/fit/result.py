"""Container for the outcome of a fit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from sciflow.models.base import Model


@dataclass
class FitResult:
    """Everything produced by :func:`sciflow.fit.engine.fit`.

    Uncertainties are 1-sigma standard errors from the covariance matrix.
    When ``sigma`` was supplied, ``absolute_sigma=True`` semantics apply: the
    errors are taken at face value and the covariance is *not* rescaled by
    the reduced chi-square. ``chi2_reduced`` far from 1 is then a real
    statement about the data/model agreement (or about the error bars).
    """

    model: Model
    params: dict[str, float]
    errors: dict[str, float]
    covariance: np.ndarray
    x: np.ndarray
    y: np.ndarray
    sigma: Optional[np.ndarray]
    fitted: np.ndarray
    residuals: np.ndarray
    n_points: int
    n_params: int
    dof: int
    chi2: float
    chi2_reduced: float
    p_value: float
    r_squared: Optional[float]
    rmse: float
    residual_sd: float
    aic: float
    bic: float
    method: str
    success: bool = True
    message: str = ""
    n_evaluations: int = 0
    x_name: str = "x"
    y_name: str = "y"
    sigma_x: Optional[np.ndarray] = None
    sigma_eff: Optional[np.ndarray] = None
    derived: dict[str, tuple[float, float]] = field(default_factory=dict)
    """name -> (value, 1-sigma error) of quantities derived from the parameters."""
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def normalized_residuals(self) -> np.ndarray:
        """Residuals divided by the effective sigma.

        Effective sigma is ``sigma`` (y errors only), ``sqrt(sigma^2 +
        (f'(x) sigma_x)^2)`` when x errors were given, or the residual SD when
        no uncertainties were supplied at all.
        """
        if self.sigma_eff is not None and (self.sigma is not None):
            scale = self.sigma_eff
        elif self.sigma is not None:
            scale = self.sigma
        else:
            scale = self.residual_sd
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.residuals / scale

    @property
    def correlation(self) -> np.ndarray:
        d = np.sqrt(np.diag(self.covariance))
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.covariance / np.outer(d, d)

    def predict(self, x) -> np.ndarray:
        return self.model(np.asarray(x, dtype=float), *self.params.values())

    def as_dict(self) -> dict:
        """JSON-serialisable summary (arrays as lists)."""
        return {
            "model": self.model.name,
            "formula": self.model.formula,
            "x": self.x_name,
            "y": self.y_name,
            "params": {k: float(v) for k, v in self.params.items()},
            "errors": {k: float(v) for k, v in self.errors.items()},
            "covariance": self.covariance.tolist(),
            "n_points": int(self.n_points),
            "n_params": int(self.n_params),
            "dof": int(self.dof),
            "chi2": float(self.chi2),
            "chi2_reduced": float(self.chi2_reduced),
            "p_value": float(self.p_value),
            "r_squared": None if self.r_squared is None else float(self.r_squared),
            "rmse": float(self.rmse),
            "residual_sd": float(self.residual_sd),
            "aic": float(self.aic),
            "bic": float(self.bic),
            "method": self.method,
            "success": bool(self.success),
            "message": self.message,
            "weighted": self.sigma is not None,
            "x_errors": self.sigma_x is not None,
            "derived": {k: {"value": float(v), "error": float(e)} for k, (v, e) in self.derived.items()},
        }

    def summary(self) -> str:
        """Plain-text summary suitable for the terminal."""
        width = max(len(k) for k in self.params)
        lines = [
            f"Model: {self.model.name}   {self.model.formula}",
            f"Method: {self.method}   ({'weighted' if self.sigma is not None else 'unweighted'}"
            + (", x uncertainties)" if self.sigma_x is not None else ")"),
            "",
        ]
        for name in self.params:
            value, err = self.params[name], self.errors[name]
            rel = f"({100 * abs(err / value):.2f} %)" if value != 0 and np.isfinite(err) else ""
            lines.append(f"  {name:<{width}} = {value: .6g} +/- {err:.3g} {rel}")
        if self.derived:
            lines.append("")
            dw = max(len(k) for k in self.derived)
            for name, (value, err) in self.derived.items():
                lines.append(f"  {name:<{dw}} = {value: .6g} +/- {err:.3g}   (derived)")
        lines += [
            "",
            f"  N = {self.n_points}, parameters = {self.n_params}, dof = {self.dof}",
            f"  chi2 = {self.chi2:.4g}, chi2/dof = {self.chi2_reduced:.4g}, p-value = {self.p_value:.3g}",
            f"  RMSE = {self.rmse:.4g}, residual SD = {self.residual_sd:.4g}"
            + (f", R^2 = {self.r_squared:.6f}" if self.r_squared is not None else ""),
            f"  AIC = {self.aic:.4g}, BIC = {self.bic:.4g}",
        ]
        if not self.success:
            lines.append(f"  WARNING: {self.message}")
        return "\n".join(lines)
