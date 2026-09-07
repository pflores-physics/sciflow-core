"""Reproduce the NIST StRD certified values with sciflow — no code reading needed.

    python examples/validate_nist.py

For each dataset the script fits the NIST model from BOTH official starting
points and prints the certified parameter values next to the fitted ones,
with the number of agreeing significant figures. NIST's own criterion for a
correct implementation is agreement to 4-6 figures.
Reference: https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sciflow import fit
from sciflow.models import expression_model

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

DATASETS = [
    ("Misra1a", "lower", "b1*(1-exp(-b2*x))", [[500, 1e-4], [250, 5e-4]]),
    ("Chwirut1", "lower", "exp(-b1*x)/(b2+b3*x)", [[0.1, 0.01, 0.02], [0.15, 0.008, 0.010]]),
    ("MGH09", "higher", "b1*(x**2+x*b2)/(x**2+x*b3+b4)", [[25, 39, 41.5, 39], [0.25, 0.39, 0.415, 0.39]]),
    ("Thurber", "higher", "(b1+b2*x+b3*x**2+b4*x**3)/(1+b5*x+b6*x**2+b7*x**3)",
     [[1000, 1000, 400, 40, 0.7, 0.3, 0.03], [1300, 1500, 500, 75, 1, 0.4, 0.05]]),
    ("Eckerle4", "higher", "(b1/b2)*exp(-0.5*((x-b3)/b2)**2)", [[1, 10, 500], [1.5, 5, 450]]),
    ("Bennett5", "higher", "b1*(b2+x)**(-1/b3)", [[-2000, 50, 0.8], [-1500, 45, 0.85]]),
    ("Rat43", "higher", "b1/((1+exp(b2-b3*x))**(1/b4))", [[100, 10, 1, 1], [700, 5, 0.75, 1.3]]),
]


def certified_values(csv_path: Path) -> dict[str, tuple[float, float]]:
    """Parse the '# Certified:' header lines that ship with every CSV."""
    out = {}
    for line in csv_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        body = line.lstrip("# ").replace("Certified:", "").strip()
        if "+/-" in body and "=" in body and body.split("=")[0].strip().startswith("b"):
            name, rest = body.split("=", 1)
            value, sd = rest.split("+/-")
            out[name.strip()] = (float(value), float(sd))
    return out


def agreeing_figures(a: float, b: float) -> int:
    if a == b:
        return 12
    rel = abs(a - b) / abs(b)
    return 12 if rel == 0 else max(0, int(math.floor(-math.log10(rel))))


def main() -> int:
    worst_overall = 99
    for name, difficulty, expr, starts in DATASETS:
        csv_path = DATA / f"nist_{name.lower()}.csv"
        data = pd.read_csv(csv_path, comment="#")
        cert = certified_values(csv_path)
        params = list(cert)
        print(f"\n{'=' * 78}\n{name}  (difficulty: {difficulty}, N = {len(data)})   y = {expr}")
        for k, start in enumerate(starts, 1):
            result = fit(data["x"], data["y"], expression_model(expr, params, initial_guess=start), maxfev=50_000)
            print(f"  Start {k}: {start}")
            print(f"  {'param':<6} {'NIST certified':>18} {'sciflow':>18} {'figs':>5}   "
                  f"{'NIST std.dev.':>14} {'sciflow':>14} {'figs':>5}")
            worst = 99
            for p in params:
                cv, cs = cert[p]
                fv, fs = result.params[p], result.errors[p]
                fv_figs, fs_figs = agreeing_figures(fv, cv), agreeing_figures(fs, cs)
                worst = min(worst, fv_figs)
                print(f"  {p:<6} {cv:>18.10e} {fv:>18.10e} {fv_figs:>5}   {cs:>14.6e} {fs:>14.6e} {fs_figs:>5}")
            worst_overall = min(worst_overall, worst)
            print(f"  -> parameters agree to at least {worst} significant figures")
    print(f"\n{'=' * 78}\nWorst case over all datasets and starting points: {worst_overall} significant figures "
          f"(NIST pass criterion: 4-6).")
    return 0 if worst_overall >= 4 else 1


if __name__ == "__main__":
    sys.exit(main())
