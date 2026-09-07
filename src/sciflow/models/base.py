"""Model definition used by the fitting engine.

A model is a callable ``f(x, *params)`` plus metadata. Models that are linear
in their parameters (straight line, polynomial) also provide a design matrix
so the engine can solve them exactly by weighted linear least squares instead
of iterating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from sciflow.errors import ModelError

ModelFunc = Callable[..., np.ndarray]
GuessFunc = Callable[[np.ndarray, np.ndarray], Sequence[float]]
DesignFunc = Callable[[np.ndarray], np.ndarray]


@dataclass
class Model:
    """A parametric model ``y = f(x; params)``.

    Attributes
    ----------
    name
        Short identifier used in configs and the CLI (``"linear"``).
    func
        ``func(x, *params) -> y`` vectorised over ``x``.
    param_names
        Names in the same order ``func`` expects them.
    formula
        Human-readable expression for reports.
    initial_guess
        ``guess(x, y) -> params``. Required for non-linear models; ignored when
        ``design_matrix`` is given.
    design_matrix
        ``design(x) -> (n, p)`` matrix ``A`` such that ``y = A @ params``. When
        present the engine uses an exact linear solver.
    description
        One-line help text.
    """

    name: str
    func: ModelFunc
    param_names: list[str]
    formula: str
    initial_guess: Optional[GuessFunc] = None
    design_matrix: Optional[DesignFunc] = None
    description: str = ""
    bounds: tuple = field(default=(-np.inf, np.inf))
    derived: Optional[Callable[[dict], dict]] = None
    prepare: Optional[Callable[[np.ndarray, np.ndarray], None]] = None
    """``prepare(x, y)`` — called by the engine before fitting; lets a model fix
    data-dependent constants (e.g. the centre of a linear background)."""
    """``derived(params: dict) -> dict[name, value]`` — physical quantities computed
    from the parameters (e.g. tau = -1/b, FWHM, peak area). The engine propagates
    the parameter covariance to them numerically."""

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    @property
    def is_linear(self) -> bool:
        return self.design_matrix is not None

    def __call__(self, x, *params) -> np.ndarray:
        if len(params) != self.n_params:
            raise ModelError(
                f"Model '{self.name}' expects {self.n_params} parameters "
                f"{self.param_names}, got {len(params)}."
            )
        return np.asarray(self.func(np.asarray(x, dtype=float), *params), dtype=float)

    def guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.initial_guess is None:
            return np.ones(self.n_params)
        p0 = np.asarray(self.initial_guess(x, y), dtype=float)
        if p0.shape != (self.n_params,):
            raise ModelError(
                f"initial_guess of '{self.name}' returned {p0.shape}, "
                f"expected ({self.n_params},)."
            )
        # Replace anything non-finite with 1.0 so curve_fit can start.
        p0[~np.isfinite(p0)] = 1.0
        return p0
