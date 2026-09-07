"""Matplotlib figures for fit results.

Every function returns ``(fig, ax)`` and never calls ``plt.show()``: the
caller decides whether to display, save or embed. Pass ``path`` to save.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sciflow.fit.result import FitResult

PathLike = Union[str, Path]


def use_headless_backend() -> None:
    """Switch matplotlib to the non-interactive Agg backend (for batch runs)."""
    matplotlib.use("Agg", force=True)


def _save(fig, path: Optional[PathLike], dpi: int):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_fit(
    result: FitResult,
    *,
    path: Optional[PathLike] = None,
    title: Optional[str] = None,
    n_curve: int = 400,
    show_band: bool = True,
    dpi: int = 300,
    ax=None,
):
    """Data with error bars, fitted curve and (optionally) a 1-sigma band.

    The band is a first-order propagation of the parameter covariance through
    the model (numerical Jacobian), which is exact for linear models.
    """
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(8, 5.5))
    else:
        fig = ax.figure

    ax.errorbar(
        result.x,
        result.y,
        yerr=result.sigma,
        xerr=result.sigma_x,
        fmt="o",
        markersize=4,
        capsize=3,
        label="data",
        zorder=3,
    )

    x_curve = np.linspace(result.x.min(), result.x.max(), n_curve)
    y_curve = result.predict(x_curve)
    ax.plot(x_curve, y_curve, linewidth=2, label=f"fit: {result.model.name}", zorder=4)

    if show_band:
        band = prediction_band(result, x_curve)
        if band is not None:
            ax.fill_between(
                x_curve, y_curve - band, y_curve + band,
                alpha=0.2, linewidth=0, label="1σ band",
            )

    ax.set_xlabel(result.x_name)
    ax.set_ylabel(result.y_name)
    ax.set_title(title or f"Data and fit: {result.model.formula}")
    ax.grid(alpha=0.3)
    ax.legend()
    if own_figure:
        fig.tight_layout()
    _save(fig, path, dpi)
    return fig, ax


def plot_residuals(
    result: FitResult,
    *,
    path: Optional[PathLike] = None,
    normalized: bool = True,
    dpi: int = 300,
    ax=None,
):
    """Residuals versus x.

    With ``normalized=True`` (default) the residuals are divided by sigma and
    guide lines at ±1 and ±2 are drawn: about 68 % of points should fall
    inside ±1 if the error bars are honest and the model is right.
    """
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(8, 3.8))
    else:
        fig = ax.figure

    if normalized:
        values = result.normalized_residuals
        ax.errorbar(result.x, values, fmt="o", markersize=4)
        for level, style in ((1, "--"), (2, ":")):
            ax.axhline(level, linestyle=style, linewidth=1, color="gray")
            ax.axhline(-level, linestyle=style, linewidth=1, color="gray")
        if result.sigma_x is not None and result.sigma is not None:
            ax.set_ylabel("(y - fit) / σ_eff")
        elif result.sigma is not None:
            ax.set_ylabel("(y - fit) / σ")
        else:
            ax.set_ylabel("(y - fit) / s_res")
    else:
        ax.errorbar(result.x, result.residuals, yerr=result.sigma, fmt="o", markersize=4, capsize=3)
        ax.set_ylabel("y - fit")

    ax.axhline(0, linewidth=1.5, color="black")
    ax.set_xlabel(result.x_name)
    ax.set_title("Normalised residuals" if normalized else "Residuals")
    ax.grid(alpha=0.3)
    if own_figure:
        fig.tight_layout()
    _save(fig, path, dpi)
    return fig, ax


def plot_fit_with_residuals(
    result: FitResult, *, path: Optional[PathLike] = None, dpi: int = 300, **kwargs
):
    """Two stacked panels: fit on top, normalised residuals below."""
    fig, (ax_fit, ax_res) = plt.subplots(
        2, 1, figsize=(8, 10.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.28},
    )
    plot_fit(result, ax=ax_fit, **kwargs)
    plot_residuals(result, ax=ax_res)
    # Both panels keep their own title and x axis (sharex hides the top labels).
    ax_fit.tick_params(labelbottom=True)
    ax_fit.set_xlabel(result.x_name)
    _save(fig, path, dpi)
    return fig, (ax_fit, ax_res)


def prediction_band(result: FitResult, x: np.ndarray, eps: float = 1e-6) -> Optional[np.ndarray]:
    """1-sigma uncertainty of the fitted curve at ``x`` from the covariance."""
    params = np.array(list(result.params.values()), dtype=float)
    if not np.all(np.isfinite(result.covariance)):
        return None
    jac = np.empty((x.size, params.size))
    for j in range(params.size):
        step = eps * max(1.0, abs(params[j]))
        plus, minus = params.copy(), params.copy()
        plus[j] += step
        minus[j] -= step
        jac[:, j] = (result.model(x, *plus) - result.model(x, *minus)) / (2 * step)
    variance = np.einsum("ij,jk,ik->i", jac, result.covariance, jac)
    return np.sqrt(np.clip(variance, 0, None))


def plot_diagnostics(result: FitResult, *, path: Optional[PathLike] = None, dpi: int = 300):
    """Two panels: histogram of normalised residuals vs N(0,1), and a Q-Q plot.

    If the model and the uncertainties are right, the histogram follows the
    standard normal curve and the Q-Q points lie on the diagonal.
    """
    from scipy import stats

    z = result.normalized_residuals
    z = z[np.isfinite(z)]
    fig, (ax_h, ax_q) = plt.subplots(1, 2, figsize=(10, 4.2))

    bins = max(5, min(25, int(np.sqrt(z.size) * 1.5)))
    ax_h.hist(z, bins=bins, density=True, alpha=0.6, edgecolor="white", label="residuals")
    grid = np.linspace(-4, 4, 200)
    ax_h.plot(grid, stats.norm.pdf(grid), linewidth=2, label="N(0, 1)")
    for level, style in ((1, "--"), (2, ":")):
        ax_h.axvline(level, linestyle=style, linewidth=1, color="gray")
        ax_h.axvline(-level, linestyle=style, linewidth=1, color="gray")
    ax_h.set_xlabel("(y - fit) / σ" if result.sigma is not None else "(y - fit) / s_res")
    ax_h.set_ylabel("density")
    ax_h.set_title("Histogram of normalised residuals vs N(0,1)")
    ax_h.legend()
    ax_h.grid(alpha=0.3)

    (osm, osr), (slope, intercept, _) = stats.probplot(z, dist="norm")
    ax_q.plot(osm, osr, "o", markersize=4, label="residual quantiles")
    lim = np.array([min(osm.min(), osr.min()), max(osm.max(), osr.max())])
    ax_q.plot(lim, lim, linewidth=1.5, color="gray", label="y = x (ideal)")
    ax_q.set_xlabel("theoretical N(0,1) quantiles")
    ax_q.set_ylabel("observed quantiles")
    ax_q.set_title("Q-Q plot of normalised residuals")
    ax_q.legend()
    ax_q.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, path, dpi)
    return fig, (ax_h, ax_q)
