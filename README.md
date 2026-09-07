# sciflow-core

[![tests](https://github.com/pflores-physics/sciflow-core/actions/workflows/tests.yml/badge.svg)](https://github.com/pflores-physics/sciflow-core/actions/workflows/tests.yml)

Curve fitting for experimental data that takes uncertainties seriously.
This is the open-source engine behind **sciflow**; the report generator,
the YAML batch pipeline and the web interface are distributed separately.

## What it does

* **Weighted least squares** with absolute σ (exact solver for models linear
  in the parameters, Levenberg–Marquardt otherwise).
* **Orthogonal distance regression** when x has uncertainties too
  (`sigma_x=`), implemented on `scipy.optimize.least_squares` with the
  ODRPACK formulation — no dependency on the deprecated `scipy.odr`.
* **Models**: line, proportional, polynomials, exponential (with/without
  offset), power law, Gaussian / Lorentzian peaks, multi-peak spectra on a
  background with automatic peak finding (`gaussian3+linear`), sine,
  logistic, and any user formula (`expression_model("a*exp(-x/tau)+c")`).
* **Derived quantities** with propagated errors: lifetimes and half-lives,
  peak FWHM and areas, periods.
* **Goodness-of-fit diagnostics** with explicit acceptance criteria: χ²/dof
  against its 95 % interval, p-value, ±1σ/±2σ coverage, outliers, runs test
  for trends, Durbin–Watson, Shapiro–Wilk, parameters compatible with zero,
  strongest correlation.
* **Model comparison** by AIC/BIC.
* **Figures**: data + fit + 1σ band with normalised residuals; histogram of
  residuals vs N(0,1) and Q-Q plot.

## Validation

Reproduces the certified values of seven NIST Statistical Reference Datasets
for nonlinear regression from both NIST starting points
(`tests/test_nist.py`):

| Dataset | Difficulty | Params | N | Agreement (params) |
|---|---|---:|---:|---|
| Misra1a | lower | 2 | 14 | 9 significant figures |
| Chwirut1 | lower | 3 | 214 | 6 |
| MGH09 | higher | 4 | 11 | 7 |
| Thurber | higher | 7 | 37 | 7 |
| Eckerle4 | higher | 3 | 35 | 10 |
| Bennett5 | higher | 3 | 154 | 6 |
| Rat43 | higher | 4 | 15 | 7 |

Residual sums of squares agree to 9–11 figures. NIST's own pass criterion is
4–6 figures.

## Install

```bash
pip install -e ".[dev]"
pytest
```

## Use

```python
import numpy as np
from sciflow import read_table, clean_data, fit, compare_models, diagnostics
from sciflow.plots import plot_fit_with_residuals

data = read_table("examples/data/pendulum_xy_errors.csv")
data, report = clean_data(data, ["L", "T2", "sigma_T2", "sigma_L"])

# uncertainties in both variables -> orthogonal distance regression
result = fit(data["L"], data["T2"], "proportional", data["sigma_T2"], sigma_x=data["sigma_L"])
print(result.summary())
for check in diagnostics(result):
    print(check.status, check.name, check.value, "expected:", check.expected)

fig, axes = plot_fit_with_residuals(result, path="fit.png")

# multi-peak spectrum with derived areas
spec = read_table("examples/data/gaussian_peak.csv")
r = fit(spec["x"], spec["counts"], "gaussian1+constant", spec["err"])
print(r.derived)  # {"FWHM_1": (value, error), "area_1": (value, error)}
```

## Licence

MIT.
