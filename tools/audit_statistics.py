"""Statistical audit of the sciflow engine by Monte Carlo simulation.

    python tools/audit_statistics.py [--reps 2000]

For each scenario the script generates many synthetic datasets from a known
model with known noise, fits them, and measures properties that a correct
implementation must have:

* coverage: the true parameter falls inside value +/- 1 sigma about 68.3 % of
  the time and inside +/- 2 sigma about 95.4 % of the time;
* chi2/dof has mean 1 and the p-value is uniform on [0, 1] when the model and
  the errors are right;
* derived quantities (tau, areas, ...) have correct propagated errors;
* each diagnostic raises a warning rarely on correct fits (false-positive
  rate) and often on wrong ones (power).

Exit code 1 if any hard check fails. Takes a few minutes with --reps 2000.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np
from scipy import stats

from sciflow.errors import FitError
from sciflow.fit import diagnostics, fit
from sciflow.models import expression_model

warnings.filterwarnings("ignore")

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'OK ' if ok else 'BAD'}] {name}: {detail}")


def coverage_stats(z: np.ndarray) -> tuple[float, float]:
    """Fraction of |z| <= 1 and <= 2 for pulls z = (fit - true) / error."""
    return float(np.mean(np.abs(z) <= 1)), float(np.mean(np.abs(z) <= 2))


def check_coverage(name, pulls, reps, tol_sd=4.0):
    """Compare coverage with 68.27 % / 95.45 % using binomial tolerance."""
    c1, c2 = coverage_stats(pulls)
    e1, e2 = 0.6827, 0.9545
    s1, s2 = np.sqrt(e1 * (1 - e1) / reps), np.sqrt(e2 * (1 - e2) / reps)
    ok = abs(c1 - e1) <= tol_sd * s1 and abs(c2 - e2) <= tol_sd * s2
    record(name, ok, f"within 1σ {100 * c1:.1f} % (exp 68.3), within 2σ {100 * c2:.1f} % (exp 95.5), "
                     f"pull mean {pulls.mean():+.3f}, pull sd {pulls.std():.3f} (exp 0, 1)")
    return ok


def check_chi2(name, chi2_red, pvals, dof):
    mean = chi2_red.mean()
    sd_expected = np.sqrt(2 / dof) / np.sqrt(len(chi2_red))
    ks = stats.kstest(pvals, "uniform").pvalue
    ok = abs(mean - 1) <= 4 * sd_expected and ks > 0.001
    record(name, ok, f"mean chi2/dof {mean:.4f} (exp 1 ± {sd_expected:.4f}), p-value uniformity KS p = {ks:.3f}")
    return ok


# --------------------------------------------------------------------------- #
def scenario_weighted_linear(rng, reps):
    print("\n1. Weighted linear fit, absolute sigma (exact solver)")
    x = np.linspace(0, 10, 25)
    sig = 0.3 + 0.05 * x
    pulls, rc, pv = [], [], []
    for _ in range(reps):
        y = 1.5 + 2.0 * x + rng.normal(0, sig)
        r = fit(x, y, "linear", sig)
        pulls.append([(r.params["a"] - 1.5) / r.errors["a"], (r.params["b"] - 2.0) / r.errors["b"]])
        rc.append(r.chi2_reduced)
        pv.append(r.p_value)
    pulls = np.array(pulls)
    check_coverage("linear: intercept coverage", pulls[:, 0], reps)
    check_coverage("linear: slope coverage", pulls[:, 1], reps)
    check_chi2("linear: chi2/dof and p-value", np.array(rc), np.array(pv), 23)


def scenario_unweighted(rng, reps):
    print("\n2. Unweighted fit (errors scaled by residual scatter)")
    x = np.linspace(0, 10, 25)
    pulls = []
    for _ in range(reps):
        y = 1.5 + 2.0 * x + rng.normal(0, 0.4, x.size)
        r = fit(x, y, "linear")
        pulls.append((r.params["b"] - 2.0) / r.errors["b"])
    # With estimated variance the pull follows Student t with dof=23, not N(0,1):
    # expected coverage of ±1 "sigma" is P(|t_23| <= 1) = 0.672, of ±2 is 0.942.
    pulls = np.array(pulls)
    c1, c2 = coverage_stats(pulls)
    e1, e2 = 2 * stats.t.cdf(1, 23) - 1, 2 * stats.t.cdf(2, 23) - 1
    ok = abs(c1 - e1) < 4 * np.sqrt(e1 * (1 - e1) / reps) and abs(c2 - e2) < 4 * np.sqrt(e2 * (1 - e2) / reps)
    record("unweighted: slope coverage (Student-t expectation)", ok,
           f"within 1σ {100 * c1:.1f} % (exp {100 * e1:.1f}), within 2σ {100 * c2:.1f} % (exp {100 * e2:.1f})")
    # Residual SD should be unbiased estimate of 0.4 (as a variance)
    sds = []
    for _ in range(400):
        y = 1.5 + 2.0 * x + rng.normal(0, 0.4, x.size)
        sds.append(fit(x, y, "linear").residual_sd ** 2)
    record("unweighted: residual variance unbiased", abs(np.mean(sds) - 0.16) < 0.01,
           f"mean residual_sd^2 = {np.mean(sds):.4f} (exp 0.1600)")


def scenario_nonlinear(rng, reps):
    print("\n3. Non-linear weighted fits (Levenberg-Marquardt)")
    x = np.linspace(0, 10, 40)
    sig = np.full_like(x, 0.05)
    pulls, rc, pv, tau_pulls = [], [], [], []
    for _ in range(reps):
        y = 5.0 * np.exp(-x / 2.5) + 0.8 + rng.normal(0, sig)
        r = fit(x, y, "exp_offset", sig)
        pulls.append([(r.params["a"] - 5.0) / r.errors["a"], (r.params["b"] + 0.4) / r.errors["b"],
                      (r.params["c"] - 0.8) / r.errors["c"]])
        tau, tau_err = r.derived["tau = -1/b"]
        tau_pulls.append((tau - 2.5) / tau_err)
        rc.append(r.chi2_reduced)
        pv.append(r.p_value)
    pulls = np.array(pulls)
    check_coverage("exp_offset: amplitude coverage", pulls[:, 0], reps)
    check_coverage("exp_offset: rate coverage", pulls[:, 1], reps)
    check_coverage("exp_offset: offset coverage", pulls[:, 2], reps)
    check_coverage("exp_offset: derived tau coverage (error propagation)", np.array(tau_pulls), reps)
    check_chi2("exp_offset: chi2/dof and p-value", np.array(rc), np.array(pv), 37)

    x = np.linspace(-5, 5, 60)
    sig = np.full_like(x, 0.03)
    area_pulls, w_pulls = [], []
    true_area = 2.0 * 0.7 * np.sqrt(2 * np.pi)
    for _ in range(reps // 2):
        y = 2.0 * np.exp(-0.5 * ((x - 1.0) / 0.7) ** 2) + 0.2 + rng.normal(0, sig)
        r = fit(x, y, "gaussian_offset", sig)
        a, ae = r.derived["area"]
        area_pulls.append((a - true_area) / ae)
        w_pulls.append((r.params["sigma"] - 0.7) / r.errors["sigma"])
    check_coverage("gaussian_offset: width coverage", np.array(w_pulls), reps // 2)
    check_coverage("gaussian_offset: derived area coverage", np.array(area_pulls), reps // 2)


def scenario_odr(rng, reps):
    print("\n4. Orthogonal distance regression (errors in x and y)")
    xt = np.linspace(0, 10, 30)
    sx = np.full_like(xt, 0.15)
    sy = np.full_like(xt, 0.3)
    pulls, rc, pv, naive_pulls = [], [], [], []
    for _ in range(reps):
        x = xt + rng.normal(0, sx)
        y = 1.5 + 2.0 * xt + rng.normal(0, sy)
        r = fit(x, y, "linear", sy, sigma_x=sx)
        pulls.append([(r.params["a"] - 1.5) / r.errors["a"], (r.params["b"] - 2.0) / r.errors["b"]])
        rc.append(r.chi2_reduced)
        pv.append(r.p_value)
        n = fit(x, y, "linear", sy)  # ignoring x errors: should be over-confident
        naive_pulls.append((n.params["b"] - 2.0) / n.errors["b"])
    pulls = np.array(pulls)
    check_coverage("ODR linear: intercept coverage", pulls[:, 0], reps)
    check_coverage("ODR linear: slope coverage", pulls[:, 1], reps)
    check_chi2("ODR linear: chi2/dof and p-value", np.array(rc), np.array(pv), 28)
    c1, _ = coverage_stats(np.array(naive_pulls))
    record("ODR: ignoring sigma_x is visibly over-confident (sanity)", c1 < 0.60,
           f"naive fit coverage within 1σ = {100 * c1:.1f} % (should be well below 68)")

    # non-linear ODR
    xt = np.linspace(0, 5, 40)
    sx = np.full_like(xt, 0.05)
    sy = np.full_like(xt, 0.03)
    pulls = []
    for _ in range(reps // 2):
        x = xt + rng.normal(0, sx)
        y = 3.0 * np.exp(-0.8 * xt) + rng.normal(0, sy)
        r = fit(x, y, "exponential", sy, sigma_x=sx)
        pulls.append((r.params["b"] + 0.8) / r.errors["b"])
    check_coverage("ODR exponential: rate coverage", np.array(pulls), reps // 2)


def scenario_multipeak(rng, reps):
    print("\n5. Multi-peak model with automatic initial guess")
    x = np.linspace(400, 700, 300)
    def g(x, A, x0, w):
        return A * np.exp(-0.5 * ((x - x0) / w) ** 2)

    sig = np.full_like(x, 0.01)
    fails, pulls, area_pulls = 0, [], []
    for _ in range(reps // 4):
        y = g(x, 1.0, 470, 8) + g(x, 0.6, 520, 12) + g(x, 0.8, 610, 6) + 0.05 + 2e-4 * (x - 400) + rng.normal(0, sig)
        try:
            r = fit(x, y, "gaussian3+linear", sig)
        except FitError:
            fails += 1
            continue
        order = np.argsort([r.params[f"x0{i}"] for i in (1, 2, 3)])
        centres = [r.params[f"x0{i + 1}"] for i in order]
        if np.max(np.abs(np.array(centres) - [470, 520, 610])) > 2:
            fails += 1
            continue
        i1 = order[0] + 1
        pulls.append((r.params[f"x0{i1}"] - 470) / r.errors[f"x0{i1}"])
        a, ae = r.derived[f"area_{i1}"]
        area_pulls.append((a - 1.0 * 8 * np.sqrt(2 * np.pi)) / ae)
    record("multipeak: automatic guess finds all three peaks", fails == 0, f"{fails} failures in {reps // 4} runs")
    check_coverage("multipeak: peak-1 centre coverage", np.array(pulls), len(pulls))
    check_coverage("multipeak: peak-1 derived area coverage", np.array(area_pulls), len(area_pulls))


def scenario_diagnostics(rng, reps):
    print("\n6. Diagnostics: false-positive rate on correct fits, power on wrong fits")
    x = np.linspace(0, 10, 40)
    sig = np.full_like(x, 0.2)
    warn_counts_good, warn_counts_bad = {}, {}
    any_warn_good = 0
    for _ in range(reps // 2):
        y = 1.0 + 2.0 * x + rng.normal(0, sig)
        checks = diagnostics(fit(x, y, "linear", sig))
        if any(c.status == "warn" for c in checks):
            any_warn_good += 1
        for c in checks:
            warn_counts_good[c.key] = warn_counts_good.get(c.key, 0) + (c.status == "warn")
        y = 1.0 + 2.0 * x + 0.06 * x**2 + rng.normal(0, sig)   # mild curvature
        for c in diagnostics(fit(x, y, "linear", sig)):
            warn_counts_bad[c.key] = warn_counts_bad.get(c.key, 0) + (c.status == "warn")
    n = reps // 2
    for key in warn_counts_good:
        fp = warn_counts_good[key] / n
        ok = fp <= 0.08
        record(f"diagnostic '{key}': false-positive rate", ok, f"{100 * fp:.1f} % (should be ≲ 5–7 %)")
    record("diagnostics: at least one warning on a correct fit (overall)", True,
           f"{100 * any_warn_good / n:.1f} % of correct fits show ≥1 warning — expected with 8–10 tests")
    for key in ("chi2_reduced", "runs_test", "durbin_watson"):
        power = warn_counts_bad.get(key, 0) / n
        record(f"diagnostic '{key}': power vs mild curvature", power > 0.5, f"{100 * power:.0f} % detection")


def scenario_extremes(rng, reps):
    print("\n7. Small samples and heteroscedastic noise")
    x = np.linspace(0, 1, 6)
    sig = np.full_like(x, 0.05)
    pulls, rc = [], []
    for _ in range(reps):
        y = 0.5 + 3.0 * x + rng.normal(0, sig)
        r = fit(x, y, "linear", sig)
        pulls.append((r.params["b"] - 3.0) / r.errors["b"])
        rc.append(r.chi2_reduced)
    check_coverage("N=6 weighted linear: slope coverage", np.array(pulls), reps)
    record("N=6: mean chi2/dof", abs(np.mean(rc) - 1) < 4 * np.sqrt(2 / 4) / np.sqrt(reps), f"{np.mean(rc):.3f}")

    x = np.linspace(1, 100, 50)
    sig = 0.02 * (10 + 0.5 * x)  # sigma grows with y
    pulls = []
    for _ in range(reps // 2):
        y = 10 + 0.5 * x + rng.normal(0, sig)
        r = fit(x, y, "linear", sig)
        pulls.append((r.params["b"] - 0.5) / r.errors["b"])
    check_coverage("heteroscedastic weighted linear: slope coverage", np.array(pulls), reps // 2)


def scenario_custom_vs_builtin(rng, reps):
    print("\n8. Custom expression gives the same result as the built-in model")
    x = np.linspace(0, 10, 40)
    sig = np.full_like(x, 0.05)
    worst = 0.0
    for _ in range(50):
        y = 5.0 * np.exp(-0.4 * x) + 0.8 + rng.normal(0, sig)
        r1 = fit(x, y, "exp_offset", sig)
        r2 = fit(x, y, expression_model("a*exp(b*x)+c", ["a", "b", "c"], initial_guess=[4, -0.3, 0.5]), sig)
        for k in ("a", "b", "c"):
            worst = max(worst, abs(r1.params[k] - r2.params[k]) / abs(r1.params[k]),
                        abs(r1.errors[k] - r2.errors[k]) / r1.errors[k])
    record("custom vs built-in exp_offset", worst < 1e-5, f"max relative difference {worst:.1e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    if args.reps < 1000:
        print(f"NOTE: {args.reps} repetitions is a quick look; the pass/fail tolerances assume >= 1000 "
              "(default 2000). Isolated failures at low --reps are usually statistical noise: re-run with more.")
    t0 = time.time()
    for scenario in (scenario_weighted_linear, scenario_unweighted, scenario_nonlinear, scenario_odr,
                     scenario_multipeak, scenario_diagnostics, scenario_extremes, scenario_custom_vs_builtin):
        scenario(rng, args.reps)
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{'=' * 70}\n{len(RESULTS)} checks, {len(bad)} failed, {time.time() - t0:.0f} s")
    for name, _, detail in bad:
        print(f"  FAILED: {name} — {detail}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
