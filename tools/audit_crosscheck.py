"""Cross-check sciflow against independent implementations.

Every number sciflow reports for a fit is recomputed with a different tool
(closed-form textbook formulas, numpy.polyfit, statsmodels WLS/OLS, lmfit,
scipy.stats) and the two are compared to tight tolerances.

    python tools/audit_crosscheck.py

Exit status 1 if any comparison fails. Optional packages (statsmodels,
lmfit) are skipped when not installed.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import stats

from sciflow.fit.engine import fit
from sciflow.fit.diagnostics import diagnostics
from sciflow.models.registry import expression_model

FAILS: list[str] = []
COUNT = 0


def check(name: str, a, b, rtol=1e-8, atol=0.0) -> None:
    global COUNT
    COUNT += 1
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.allclose(a, b, rtol=rtol, atol=atol)
    diff = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))) if a.size else 0.0
    print(f"  [{'OK ' if ok else 'BAD'}] {name}: max rel diff {diff:.1e}")
    if not ok:
        FAILS.append(name)


rng = np.random.default_rng(7)

# ---------------------------------------------------------------- 1. weighted line, closed form (Bevington 6.12–6.23)
print("1. Weighted straight line vs closed-form formulas (Bevington & Robinson)")
x = np.linspace(0, 10, 25)
sig = rng.uniform(0.2, 0.8, x.size)
y = 1.5 + 0.7 * x + rng.normal(0, sig)
r = fit(x, y, "linear", sig)
w = 1 / sig**2
S, Sx, Sy, Sxx, Sxy = w.sum(), (w * x).sum(), (w * y).sum(), (w * x * x).sum(), (w * x * y).sum()
D = S * Sxx - Sx**2
a_cf = (Sxx * Sy - Sx * Sxy) / D           # intercept
b_cf = (S * Sxy - Sx * Sy) / D             # slope
sa, sb = np.sqrt(Sxx / D), np.sqrt(S / D)
cov_ab = -Sx / D
names = list(r.params)
check("intercept, slope", [r.params[names[0]], r.params[names[1]]], [a_cf, b_cf])
check("intercept, slope errors", [r.errors[names[0]], r.errors[names[1]]], [sa, sb])
check("intercept-slope covariance", r.covariance[0, 1], cov_ab)
chi2_cf = np.sum(((y - a_cf - b_cf * x) / sig) ** 2)
check("chi2", r.chi2, chi2_cf)
check("chi2/dof", r.chi2_reduced, chi2_cf / (x.size - 2))
check("p-value", r.p_value, stats.chi2.sf(chi2_cf, x.size - 2))
check("AIC = chi2 + 2p", r.aic, chi2_cf + 2 * 2)
check("BIC = chi2 + p ln N", r.bic, chi2_cf + 2 * np.log(x.size))
ss = np.sum((y - a_cf - b_cf * x) ** 2)
check("RMSE = sqrt(SS/N)", r.rmse, np.sqrt(ss / x.size))
check("residual sd = sqrt(SS/(N-p))", r.residual_sd, np.sqrt(ss / (x.size - 2)))
# sciflow reports the ordinary (unweighted) coefficient of determination.
check("R^2 = 1 - SS_res/SS_tot (unweighted)", r.r_squared, 1 - ss / np.sum((y - y.mean()) ** 2))
check("correlation matrix", r.correlation[0, 1], cov_ab / (sa * sb))

# ---------------------------------------------------------------- 2. numpy.polyfit
print("2. Polynomial (poly3, weighted) vs numpy.polyfit(w = 1/sigma)")
y3 = 2 - 0.5 * x + 0.3 * x**2 - 0.02 * x**3 + rng.normal(0, sig)
r = fit(x, y3, "poly3", sig)
coef, cov = np.polyfit(x, y3, 3, w=1 / sig, cov="unscaled")
check("poly3 coefficients", list(r.params.values()), coef[::-1])
check("poly3 covariance", r.covariance, cov[::-1, ::-1], rtol=1e-6)

# ---------------------------------------------------------------- 3. statsmodels
try:
    import statsmodels.api as sm
    print("3. statsmodels WLS / OLS")
    X = np.column_stack([np.ones_like(x), x])
    wls = sm.WLS(y, X, weights=w).fit()
    r = fit(x, y, "linear", sig)
    check("WLS params", list(r.params.values()), wls.params)
    # statsmodels scales the covariance by the residual variance; undo it.
    check("WLS covariance (unscaled)", r.covariance, wls.cov_params() / wls.scale, rtol=1e-6)
    ols = sm.OLS(y, X).fit()
    r = fit(x, y, "linear")
    check("OLS params (unweighted)", list(r.params.values()), ols.params)
    check("OLS errors (unweighted) = scaled by residual sd", list(r.errors.values()), ols.bse, rtol=1e-6)
    check("OLS R^2", r.r_squared, ols.rsquared, rtol=1e-8)
    check("OLS residual sd", r.residual_sd, np.sqrt(ols.scale))
    dw_sm = sm.stats.durbin_watson(ols.resid)
    dw_mine = [c for c in diagnostics(r) if c.key == "durbin_watson"][0]
    check("Durbin-Watson (unweighted -> raw residuals)", float(dw_mine.value), dw_sm, rtol=5e-3)
except ImportError:
    print("3. statsmodels not installed: skipped")

# ---------------------------------------------------------------- 4. lmfit (non-linear, absolute sigma)
try:
    import lmfit
    print("4. lmfit (non-linear least squares, absolute sigma) on exp_offset and gaussian_offset")
    xe = np.linspace(0, 5, 40)
    se = np.full(xe.size, 0.05)
    ye = 3 * np.exp(-xe / 1.3) + 0.4 + rng.normal(0, se)
    r = fit(xe, ye, "exp_offset", se)

    def f_exp(x, A, k, c):  # sciflow's exp_offset is a*exp(b*x)+c, so b = k here
        return A * np.exp(k * x) + c

    m = lmfit.Model(f_exp)
    out = m.fit(ye, x=xe, A=2, k=-1, c=0, weights=1 / se, scale_covar=False)
    check("exp_offset params", list(r.params.values()), [out.params[k].value for k in ("A", "k", "c")], rtol=1e-6)
    check("exp_offset errors", list(r.errors.values()), [out.params[k].stderr for k in ("A", "k", "c")], rtol=1e-4)
    check("exp_offset chi2", r.chi2, out.chisqr, rtol=1e-8)
    check("exp_offset AIC/BIC (same definition up to constant)", r.aic - r.bic,
          2 * 3 - 3 * np.log(xe.size), rtol=1e-10)

    xg = np.linspace(-5, 5, 60)
    sg = np.full(xg.size, 0.1)
    yg = 4 * np.exp(-0.5 * ((xg - 0.7) / 1.1) ** 2) + 0.5 + rng.normal(0, sg)
    r = fit(xg, yg, "gaussian_offset", sg)

    def f_g(x, A, mu, s, c):
        return A * np.exp(-0.5 * ((x - mu) / s) ** 2) + c

    out = lmfit.Model(f_g).fit(yg, x=xg, A=3, mu=0, s=1, c=0, weights=1 / sg, scale_covar=False)
    check("gaussian_offset params", list(r.params.values()), [out.params[k].value for k in ("A", "mu", "s", "c")], rtol=1e-6)
    check("gaussian_offset errors", list(r.errors.values()), [out.params[k].stderr for k in ("A", "mu", "s", "c")], rtol=1e-4)
    # Derived FWHM and area, propagated by hand from the lmfit covariance.
    A, s = out.params["A"].value, out.params["s"].value
    cov = out.covar  # order A, mu, s, c
    fwhm = 2 * np.sqrt(2 * np.log(2)) * abs(s)
    d_fwhm = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(cov[2, 2])
    area = A * abs(s) * np.sqrt(2 * np.pi)
    g = np.array([abs(s), 0, A * np.sign(s), 0]) * np.sqrt(2 * np.pi)
    d_area = np.sqrt(g @ cov @ g)
    d = r.derived
    fk = [k for k in d if "fwhm" in k.lower()][0]
    ak = [k for k in d if "area" in k.lower()][0]
    check("derived FWHM (value, error)", d[fk], (fwhm, d_fwhm), rtol=1e-4)
    check("derived area (value, error)", d[ak], (area, d_area), rtol=1e-4)
except ImportError:
    print("4. lmfit not installed: skipped")

# ---------------------------------------------------------------- 5. ODR vs closed form (York 1966 / Deming) for a line
print("5. ODR straight line vs York (1966) iterative solution")
xt = np.linspace(1, 10, 20)
sx = np.full(xt.size, 0.15)
sy = np.full(xt.size, 0.3)
xo = xt + rng.normal(0, sx)
yo = 0.8 + 1.9 * xt + rng.normal(0, sy)
r = fit(xo, yo, "linear", sy, sigma_x=sx)


def york(x, y, sx, sy, iters=100):
    wx, wy = 1 / sx**2, 1 / sy**2
    b = np.polyfit(x, y, 1)[0]
    for _ in range(iters):
        W = wx * wy / (wx + b**2 * wy)
        xb, yb = np.sum(W * x) / W.sum(), np.sum(W * y) / W.sum()
        U, V = x - xb, y - yb
        beta = W * (U / wy + b * V / wx)
        b_new = np.sum(W * beta * V) / np.sum(W * beta * U)
        if abs(b_new - b) < 1e-14:
            b = b_new
            break
        b = b_new
    a = yb - b * xb
    xadj = xb + beta
    u = xadj - np.sum(W * xadj) / W.sum()
    sb = np.sqrt(1 / np.sum(W * u**2))
    sa = np.sqrt(1 / W.sum() + (np.sum(W * xadj) / W.sum()) ** 2 * sb**2)
    S = np.sum(W * (y - b * x - a) ** 2)
    return a, b, sa, sb, S


a_y, b_y, sa_y, sb_y, S_y = york(xo, yo, sx, sy)
n0, n1 = list(r.params)
# chi2 agrees to 1e-14, so the 1e-7 parameter difference is the flatness of the minimum
check("ODR line params", [r.params[n0], r.params[n1]], [a_y, b_y], rtol=1e-6)
check("ODR line errors", [r.errors[n0], r.errors[n1]], [sa_y, sb_y], rtol=1e-4)
check("ODR chi2 (weighted orthogonal SS)", r.chi2, S_y, rtol=1e-7)

# ---------------------------------------------------------------- 6. normalised residuals, p-value, coverage arithmetic
print("6. Bookkeeping: residuals, normalised residuals, coverage")
r = fit(x, y, "linear", sig)
pred = r.params[names[0]] + r.params[names[1]] * x
check("residuals = y - f(x)", r.residuals, y - pred)
check("normalised residuals = residual / sigma", r.normalized_residuals, (y - pred) / sig)
c = {k.key: k for k in diagnostics(r)}
check("within 1σ fraction", float(c["within_1sigma"].value.rstrip(" %")) / 100,
      np.mean(np.abs((y - pred) / sig) <= 1), atol=0.006)
check("Shapiro p", float(c["shapiro"].value.split("= ")[1]), stats.shapiro((y - pred) / sig).pvalue, atol=5e-4)

# ---------------------------------------------------------------- 7. prediction from the fitted model
print("7. Custom expression model equals hand-written function")
m = expression_model("a*x**2 + b*sin(x) + c", ["a", "b", "c"], initial_guess=[1, 1, 1])
yc = 0.3 * x**2 + 2 * np.sin(x) + 1 + rng.normal(0, sig)
r = fit(x, yc, m, sig)
X = np.column_stack([x**2, np.sin(x), np.ones_like(x)])
beta, *_ = np.linalg.lstsq(X / sig[:, None], yc / sig, rcond=None)
check("expression model (linear in params) vs lstsq", list(r.params.values()), beta, rtol=1e-6)

print("=" * 70)
print(f"{COUNT} comparisons, {len(FAILS)} failed")
for f in FAILS:
    print("  FAILED:", f)
sys.exit(1 if FAILS else 0)
