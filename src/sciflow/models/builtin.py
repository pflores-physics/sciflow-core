"""Built-in models.

Initial guesses are deliberately simple and robust rather than clever: their
only job is to land curve_fit in the right basin.
"""

from __future__ import annotations

import numpy as np

from sciflow.models.base import Model


# --------------------------------------------------------------------------- #
# Linear-in-parameters models (solved exactly)
# --------------------------------------------------------------------------- #

def _linear(x, a, b):
    return a + b * x


def _linear_design(x):
    return np.column_stack([np.ones_like(x), x])


LINEAR = Model(
    name="linear",
    func=_linear,
    param_names=["a", "b"],
    formula="y = a + b*x",
    design_matrix=_linear_design,
    description="Straight line (weighted least squares, exact).",
)


def _proportional(x, b):
    return b * x


PROPORTIONAL = Model(
    name="proportional",
    func=_proportional,
    param_names=["b"],
    formula="y = b*x",
    design_matrix=lambda x: x.reshape(-1, 1),
    description="Line through the origin.",
)


def polynomial(degree: int) -> Model:
    """Polynomial of the given degree, ``y = c0 + c1*x + ... + cn*x**n``."""
    if degree < 1:
        raise ValueError("degree must be >= 1")
    names = [f"c{i}" for i in range(degree + 1)]

    def func(x, *coeffs):
        return np.polyval(coeffs[::-1], x)

    def design(x):
        return np.vander(x, degree + 1, increasing=True)

    terms = " + ".join(
        ["c0", "c1*x"] + [f"c{i}*x^{i}" for i in range(2, degree + 1)]
    )
    return Model(
        name=f"poly{degree}",
        func=func,
        param_names=names,
        formula=f"y = {terms}",
        design_matrix=design,
        description=f"Polynomial of degree {degree} (exact solver).",
    )


# --------------------------------------------------------------------------- #
# Non-linear models
# --------------------------------------------------------------------------- #

def _exponential(x, a, b):
    return a * np.exp(b * x)


def _exponential_guess(x, y):
    # Log-linearise on the positive part of y.
    mask = y > 0
    if mask.sum() >= 2:
        slope, intercept = np.polyfit(x[mask], np.log(y[mask]), 1)
        return [np.exp(intercept), slope]
    return [y[0] if y[0] != 0 else 1.0, 0.0]


def _exp_derived(p):
    b = p["b"]
    return {"tau = -1/b": -1.0 / b, "half_life = ln2/|b|": np.log(2) / abs(b)}


EXPONENTIAL = Model(
    name="exponential",
    func=_exponential,
    param_names=["a", "b"],
    formula="y = a*exp(b*x)",
    initial_guess=_exponential_guess,
    description="Exponential growth/decay.",
    derived=_exp_derived,
)


def _exp_offset(x, a, b, c):
    return a * np.exp(b * x) + c


def _exp_offset_guess(x, y):
    c = y.min() if y[-1] < y[0] else y.max()
    c = c - 0.1 * abs(c) if c != 0 else 0.0
    a, b = _exponential_guess(x, np.abs(y - c) + 1e-12)
    if y[-1] < y[0]:
        b = -abs(b)
    return [a, b, c]


EXP_OFFSET = Model(
    name="exp_offset",
    func=_exp_offset,
    param_names=["a", "b", "c"],
    formula="y = a*exp(b*x) + c",
    initial_guess=_exp_offset_guess,
    description="Exponential with constant offset.",
    derived=_exp_derived,
)


def _power_law(x, a, b):
    return a * np.power(x, b)


def _power_law_guess(x, y):
    mask = (x > 0) & (y > 0)
    if mask.sum() >= 2:
        slope, intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
        return [np.exp(intercept), slope]
    return [1.0, 1.0]


POWER_LAW = Model(
    name="power_law",
    func=_power_law,
    param_names=["a", "b"],
    formula="y = a*x^b",
    initial_guess=_power_law_guess,
    description="Power law (x > 0).",
)


def _gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _peak_guess(x, y):
    i = int(np.argmax(np.abs(y)))
    amplitude = y[i]
    center = x[i]
    half = np.abs(y) >= 0.5 * np.abs(amplitude)
    width = (x[half].max() - x[half].min()) / 2.355 if half.sum() > 1 else (x.max() - x.min()) / 6
    return [amplitude, center, max(width, 1e-6)]


GAUSSIAN = Model(
    name="gaussian",
    func=_gaussian,
    param_names=["amplitude", "center", "sigma"],
    formula="y = A*exp(-(x-x0)^2 / (2*sigma^2))",
    initial_guess=_peak_guess,
    description="Gaussian peak.",
    derived=lambda p: {"FWHM": 2.354820045 * abs(p["sigma"]),
                       "area": p["amplitude"] * abs(p["sigma"]) * np.sqrt(2 * np.pi)},
)


def _gaussian_offset(x, amplitude, center, sigma, offset):
    return _gaussian(x, amplitude, center, sigma) + offset


def _peak_offset_guess(x, y):
    offset = np.median(y)
    amp, center, width = _peak_guess(x, y - offset)
    return [amp, center, width, offset]


GAUSSIAN_OFFSET = Model(
    name="gaussian_offset",
    func=_gaussian_offset,
    param_names=["amplitude", "center", "sigma", "offset"],
    formula="y = A*exp(-(x-x0)^2 / (2*sigma^2)) + c",
    initial_guess=_peak_offset_guess,
    description="Gaussian peak on a constant background.",
    derived=lambda p: {"FWHM": 2.354820045 * abs(p["sigma"]),
                       "area": p["amplitude"] * abs(p["sigma"]) * np.sqrt(2 * np.pi)},
)


def _lorentzian(x, amplitude, center, gamma):
    return amplitude * gamma**2 / ((x - center) ** 2 + gamma**2)


LORENTZIAN = Model(
    name="lorentzian",
    func=_lorentzian,
    param_names=["amplitude", "center", "gamma"],
    formula="y = A*gamma^2 / ((x-x0)^2 + gamma^2)",
    initial_guess=_peak_guess,
    description="Lorentzian peak (gamma = half width at half maximum).",
    derived=lambda p: {"FWHM": 2 * abs(p["gamma"]), "area": np.pi * p["amplitude"] * abs(p["gamma"])},
)


def _sine(x, amplitude, frequency, phase, offset):
    return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset


def _sine_guess(x, y):
    offset = np.mean(y)
    amplitude = (y.max() - y.min()) / 2
    # Dominant frequency from an FFT on a uniform resampling.
    n = len(x)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    if n >= 4 and xs[-1] > xs[0]:
        grid = np.linspace(xs[0], xs[-1], n)
        yg = np.interp(grid, xs, ys) - offset
        spectrum = np.abs(np.fft.rfft(yg))
        freqs = np.fft.rfftfreq(n, d=grid[1] - grid[0])
        spectrum[0] = 0
        frequency = freqs[int(np.argmax(spectrum))]
        if frequency == 0:
            frequency = 1.0 / (xs[-1] - xs[0])
    else:
        frequency = 1.0
    return [amplitude, frequency, 0.0, offset]


SINE = Model(
    name="sine",
    func=_sine,
    param_names=["amplitude", "frequency", "phase", "offset"],
    formula="y = A*sin(2*pi*f*x + phi) + c",
    initial_guess=_sine_guess,
    description="Sinusoid (frequency guessed from FFT).",
    derived=lambda p: {"period = 1/f": 1.0 / p["frequency"], "omega = 2*pi*f": 2 * np.pi * p["frequency"]},
)


def _logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


def _logistic_guess(x, y):
    L = y.max()
    x0 = x[int(np.argmin(np.abs(y - L / 2)))]
    k = 4.0 / (x.max() - x.min()) if x.max() > x.min() else 1.0
    return [L, k, x0]


LOGISTIC = Model(
    name="logistic",
    func=_logistic,
    param_names=["L", "k", "x0"],
    formula="y = L / (1 + exp(-k*(x - x0)))",
    initial_guess=_logistic_guess,
    description="Logistic / sigmoid curve.",
)


BUILTIN_MODELS = [
    LINEAR,
    PROPORTIONAL,
    polynomial(2),
    polynomial(3),
    EXPONENTIAL,
    EXP_OFFSET,
    POWER_LAW,
    GAUSSIAN,
    GAUSSIAN_OFFSET,
    LORENTZIAN,
    SINE,
    LOGISTIC,
]


# --------------------------------------------------------------------------- #
# Multi-peak models: "gaussian3", "lorentzian2+linear", ...
# --------------------------------------------------------------------------- #

def _bg_terms(background, state):
    if background == "none":
        return [], lambda x, *c: 0.0
    if background in ("const", "constant"):
        return ["bg0"], lambda x, c0: c0
    if background == "linear":
        # Centred at xc (median of x, fixed by prepare) so bg0 and bg1 are not
        # artificially correlated when x is far from zero.
        return ["bg0", "bg1"], lambda x, c0, c1: c0 + c1 * (x - state["xc"])
    raise ValueError("background must be none, constant or linear")


def multipeak(kind: str, n: int, background: str = "none") -> Model:
    """Sum of ``n`` Gaussian or Lorentzian peaks on an optional background.

    Parameters per peak: ``A_i`` (height), ``x0_i`` (centre), ``w_i`` (sigma
    for Gaussians, gamma = HWHM for Lorentzians). Background: ``bg0`` (and
    ``bg1`` for linear). Initial guesses come from ``scipy.signal.find_peaks``
    on a lightly smoothed copy of the data. Derived quantities: FWHM and area
    of every peak.
    """
    if kind not in ("gaussian", "lorentzian"):
        raise ValueError("kind must be gaussian or lorentzian")
    if n < 1:
        raise ValueError("n must be >= 1")
    state = {"xc": 0.0}
    bg_names, bg_func = _bg_terms(background, state)
    peak = _gaussian if kind == "gaussian" else _lorentzian
    names = [f"{p}{i + 1}" for i in range(n) for p in ("A", "x0", "w")] + bg_names

    def func(x, *params):
        total = np.zeros_like(np.asarray(x, dtype=float))
        for i in range(n):
            A, x0, w = params[3 * i: 3 * i + 3]
            total = total + peak(x, A, x0, w)
        return total + bg_func(x, *params[3 * n:])

    def guess(x, y):
        from scipy.signal import find_peaks, peak_widths

        order = np.argsort(x)
        xs, ys = x[order], y[order]
        # Background estimate: lower envelope (10th percentile) or linear edges.
        if background == "linear":
            k = max(2, len(xs) // 10)
            c1 = (np.median(ys[-k:]) - np.median(ys[:k])) / (np.median(xs[-k:]) - np.median(xs[:k]) or 1.0)
            c0 = np.median(ys[:k]) - c1 * (np.median(xs[:k]) - state["xc"])
            bg = c0 + c1 * (xs - state["xc"])
            bg_p = [c0, c1]
        elif background != "none":
            bg_p = [float(np.percentile(ys, 10))]
            bg = np.full_like(ys, bg_p[0])
        else:
            bg_p = []
            bg = np.zeros_like(ys)
        signal = ys - bg
        # Light smoothing for peak detection only.
        win = max(3, len(xs) // 50) | 1
        kernel = np.ones(win) / win
        smooth = np.convolve(signal, kernel, mode="same")
        idx, props = find_peaks(smooth, prominence=0.05 * (smooth.max() - smooth.min() or 1.0))
        if len(idx) < n:
            # Fall back: spread peaks evenly over the x range.
            extra = np.linspace(0, len(xs) - 1, n + 2)[1:-1].astype(int)
            idx = np.unique(np.concatenate([idx, extra]))[:n] if len(idx) else extra
        else:
            top = np.argsort(props["prominences"])[::-1][:n]
            idx = np.sort(idx[top])
        widths = peak_widths(smooth, idx, rel_height=0.5)[0] * np.median(np.diff(xs))
        p0 = []
        for i, j in enumerate(idx):
            fwhm = max(float(widths[i]), 2 * float(np.median(np.diff(xs))))
            w = fwhm / 2.3548 if kind == "gaussian" else fwhm / 2
            p0 += [float(max(signal[j], 1e-12)), float(xs[j]), w]
        return p0 + bg_p

    def derived(p):
        out = {}
        for i in range(n):
            A, w = p[f"A{i + 1}"], abs(p[f"w{i + 1}"])
            if kind == "gaussian":
                out[f"FWHM_{i + 1}"] = 2.354820045 * w
                out[f"area_{i + 1}"] = A * w * np.sqrt(2 * np.pi)
            else:
                out[f"FWHM_{i + 1}"] = 2 * w
                out[f"area_{i + 1}"] = np.pi * A * w
        return out

    term = "A_i*exp(-(x-x0_i)^2/(2*w_i^2))" if kind == "gaussian" else "A_i*w_i^2/((x-x0_i)^2+w_i^2)"
    bg_txt = {"none": "", "constant": " + bg0", "const": " + bg0", "linear": " + bg0 + bg1*(x - xc)"}[background]

    def prepare(x, y):
        if background == "linear":
            state["xc"] = float(np.median(x))
            model.formula = f"y = sum_i=1..{n} {term} + bg0 + bg1*(x - xc), xc = {state['xc']:.6g}"

    model = Model(
        name=f"{kind}{n}" + ("" if background == "none" else f"+{background}"),
        func=func,
        param_names=names,
        formula=f"y = sum_i=1..{n} {term}{bg_txt}",
        initial_guess=guess,
        description=f"{n} {kind} peak(s)" + ("" if background == "none" else f" on a {background} background")
                    + " (auto peak finding).",
        derived=derived,
        prepare=prepare,
    )
    return model
