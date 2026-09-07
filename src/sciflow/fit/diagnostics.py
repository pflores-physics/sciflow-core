"""Goodness-of-fit diagnostics with explicit pass/warn criteria.

Every check returns a :class:`Check` with the measured value, the range that
is expected when the model and the uncertainties are both right, and a
status: ``"ok"``, ``"warn"`` or ``"info"`` (not applicable / informative).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

from sciflow.fit.result import FitResult


@dataclass
class Check:
    key: str
    name: str
    value: str
    expected: str
    status: str          # "ok" | "warn" | "info"
    note: str = ""

    def as_dict(self) -> dict:
        return {"key": self.key, "name": self.name, "value": self.value,
                "expected": self.expected, "status": self.status, "note": self.note}


def _runs_test(signs: np.ndarray) -> Optional[float]:
    """Wald–Wolfowitz runs test on the signs of the residuals (ordered by x).

    Returns the two-sided p-value, or None if it cannot be computed.
    Few runs = residuals cluster in stretches above/below the curve = the
    model misses a trend.
    """
    signs = signs[signs != 0]
    n_pos = int(np.sum(signs > 0))
    n_neg = int(np.sum(signs < 0))
    if n_pos < 2 or n_neg < 2:
        return None
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    n = n_pos + n_neg
    mean = 1 + 2 * n_pos * n_neg / n
    var = 2 * n_pos * n_neg * (2 * n_pos * n_neg - n) / (n**2 * (n - 1))
    if var <= 0:
        return None
    z = (runs - mean) / np.sqrt(var)
    return float(2 * stats.norm.sf(abs(z)))


def diagnostics(result: FitResult) -> list[Check]:
    """Run every diagnostic on a fit result."""
    checks: list[Check] = []
    weighted = result.sigma is not None
    order = np.argsort(result.x)
    r = result.residuals[order]
    z = result.normalized_residuals[order]
    n = result.n_points

    # 1. Reduced chi-square -------------------------------------------------
    if weighted:
        rc = result.chi2_reduced
        # 95 % interval of chi2/dof under H0.
        lo, hi = stats.chi2.ppf([0.025, 0.975], result.dof) / result.dof
        status = "ok" if lo <= rc <= hi else "warn"
        if rc > hi:
            note = "Too large: model misses structure and/or uncertainties are underestimated."
        elif rc < lo:
            note = "Too small: uncertainties are probably overestimated (fit is 'too good')."
        else:
            note = "Consistent with the quoted uncertainties."
        checks.append(Check("chi2_reduced", "Reduced chi-square χ²/dof", f"{rc:.3f}",
                            f"{lo:.2f} – {hi:.2f} (95 % interval for dof = {result.dof})", status, note))
        p = result.p_value
        checks.append(Check("p_value", "χ² p-value", f"{p:.3g}", "> 0.05 (and not ≈ 1)",
                            "ok" if 0.05 <= p <= 0.995 else "warn",
                            "Probability of a χ² at least this large if model and errors are right."))
    else:
        checks.append(Check("chi2_reduced", "Reduced chi-square χ²/dof", "n/a",
                            "requires a sigma column", "info",
                            "Without uncertainties χ² only measures scatter; parameter errors are "
                            "scaled by the residual standard deviation."))

    # 2. Coverage of normalised residuals -----------------------------------
    if np.all(np.isfinite(z)) and n >= 5:
        within1 = float(np.mean(np.abs(z) <= 1))
        within2 = float(np.mean(np.abs(z) <= 2))
        # Binomial tolerance around 68.3 % / 95.4 %.
        tol1 = 2 * np.sqrt(0.683 * 0.317 / n)
        tol2 = 2 * np.sqrt(0.954 * 0.046 / n)
        st1 = "ok" if abs(within1 - 0.683) <= max(tol1, 0.10) else "warn"
        st2 = "ok" if within2 >= 0.954 - max(tol2, 0.06) else "warn"
        if not weighted:
            st1 = st2 = "info"
        checks.append(Check("within_1sigma", "Points within ±1σ of the curve",
                            f"{100 * within1:.0f} %", "≈ 68 %", st1,
                            "Fraction of residuals smaller than one error bar."))
        checks.append(Check("within_2sigma", "Points within ±2σ of the curve",
                            f"{100 * within2:.0f} %", "≥ 95 %", st2,
                            "Points beyond ±2σ should be rare; beyond ±3σ suspect outliers."))

    # 3. Outliers -------------------------------------------------------------
    if np.all(np.isfinite(z)):
        worst = float(np.max(np.abs(z)))
        n_out = int(np.sum(np.abs(z) > 3))
        expected = 0.0027 * n
        tolerated = int(np.floor(expected + 2 * np.sqrt(expected) + 1))
        status = "ok" if n_out <= tolerated else "warn"
        checks.append(Check("outliers", "Residuals beyond ±3σ", f"{n_out} (max |z| = {worst:.2f})",
                            f"≤ {tolerated} (≈ {expected:.1f} expected for N = {n})", status if weighted else "info",
                            "Candidates for measurement errors; do not remove without a physical reason."))

    # 4. Trend: runs test ---------------------------------------------------
    p_runs = _runs_test(np.sign(r))
    if p_runs is not None:
        checks.append(Check("runs_test", "Runs test on residual signs (trend)", f"p = {p_runs:.3f}",
                            "p > 0.05", "ok" if p_runs > 0.05 else "warn",
                            "Low p: residuals form long stretches above/below the curve — "
                            "the model shape is wrong or a term is missing."))

    # 5. Autocorrelation: Durbin–Watson --------------------------------------
    zz = z if np.all(np.isfinite(z)) else r
    if n >= 12:
        dw = float(np.sum(np.diff(zz) ** 2) / np.sum(zz**2))
        checks.append(Check("durbin_watson", "Durbin–Watson statistic", f"{dw:.2f}",
                            "≈ 2 (1.5 – 2.5)", "ok" if 1.5 <= dw <= 2.5 else "warn",
                            "< 1.5: neighbouring residuals have the same sign (systematic deviation)."))

    # 6. Normality: Shapiro–Wilk ---------------------------------------------
    if 8 <= n <= 5000 and np.std(zz) > 0:
        try:
            p_sw = float(stats.shapiro(zz).pvalue)
            checks.append(Check("shapiro", "Shapiro–Wilk normality of residuals", f"p = {p_sw:.3f}",
                                "p > 0.05", "ok" if p_sw > 0.05 else "warn",
                                "Residuals should look Gaussian; failure suggests outliers or wrong model."))
        except Exception:  # pragma: no cover
            pass

    # 7. Parameter significance and correlations ----------------------------
    weak = [k for k, v in result.params.items()
            if result.errors[k] > 0 and abs(v) < 2 * result.errors[k]]
    checks.append(Check("significance", "Parameters compatible with zero (|value| < 2σ)",
                        ", ".join(weak) if weak else "none", "none", "warn" if weak else "ok",
                        "A parameter compatible with zero may be unnecessary: compare with a simpler model."))
    if result.n_params > 1:
        corr = result.correlation
        iu = np.triu_indices_from(corr, k=1)
        max_corr = float(np.max(np.abs(corr[iu])))
        names = list(result.params)
        i, j = iu[0][np.argmax(np.abs(corr[iu]))], iu[1][np.argmax(np.abs(corr[iu]))]
        checks.append(Check("correlation", "Strongest parameter correlation",
                            f"{corr[i, j]:+.2f} ({names[i]}, {names[j]})", "|ρ| < 0.9",
                            "ok" if max_corr < 0.9 else "warn",
                            "|ρ| near 1: the two parameters are nearly interchangeable — "
                            "their individual errors are large even if the curve is good."))
    return checks


def overall_status(checks: list[Check]) -> str:
    warns = [c for c in checks if c.status == "warn"]
    if not warns:
        return "ok"
    return "warn"
