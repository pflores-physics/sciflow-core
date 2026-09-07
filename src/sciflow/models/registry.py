"""Model registry: look models up by name, register your own, build from text.

Names understood by :func:`get_model`:

* any built-in name (``sciflow models`` lists them),
* ``polyN`` for a polynomial of degree N,
* a custom expression via :func:`expression_model`.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

import numpy as np

from sciflow.errors import ModelError
from sciflow.models.base import Model
from sciflow.models.builtin import BUILTIN_MODELS, multipeak, polynomial

_REGISTRY: dict[str, Model] = {m.name: m for m in BUILTIN_MODELS}
_POLY_RE = re.compile(r"^poly(\d+)$")
_PEAKS_RE = re.compile(r"^(gaussian|lorentzian)(\d+)(?:\+(constant|const|linear))?$")


def register_model(model: Model, *, overwrite: bool = False) -> Model:
    """Add a model to the registry so configs can refer to it by name."""
    if model.name in _REGISTRY and not overwrite:
        raise ModelError(f"A model named '{model.name}' already exists.")
    _REGISTRY[model.name] = model
    return model


def list_models() -> list[Model]:
    return sorted(_REGISTRY.values(), key=lambda m: m.name)


def get_model(name: str) -> Model:
    """Return the model registered under ``name`` (``polyN`` is generated)."""
    key = name.strip().lower()
    if key in _REGISTRY:
        return _REGISTRY[key]
    match = _POLY_RE.match(key)
    if match:
        model = polynomial(int(match.group(1)))
        _REGISTRY[model.name] = model
        return model
    match = _PEAKS_RE.match(key)
    if match:
        kind, n, bg = match.group(1), int(match.group(2)), match.group(3) or "none"
        try:
            model = multipeak(kind, n, "constant" if bg == "const" else bg)
        except ValueError as error:
            raise ModelError(str(error)) from error
        _REGISTRY[model.name] = model
        return model
    known = ", ".join(m.name for m in list_models())
    raise ModelError(
        f"Unknown model '{name}'. Available: {known}, polyN, "
        "gaussianN[+constant|+linear], lorentzianN[+constant|+linear]."
    )


# --------------------------------------------------------------------------- #
# Custom models from a text expression
# --------------------------------------------------------------------------- #

_SAFE_NAMES = {
    name: getattr(np, name)
    for name in (
        "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh",
        "tanh", "exp", "log", "log10", "log2", "sqrt", "abs", "pi", "e",
        "power", "where", "sign", "floor", "ceil",
    )
}
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def expression_model(
    expression: str,
    params: Optional[Sequence[str]] = None,
    *,
    name: str = "custom",
    initial_guess: Optional[Sequence[float]] = None,
    variable: str = "x",
) -> Model:
    """Build a model from a text formula such as ``"a*exp(-x/tau) + c"``.

    Parameters
    ----------
    expression
        Right-hand side written with NumPy semantics. ``**`` and ``^`` both
        mean power. Allowed functions: sin, cos, tan, exp, log, log10, sqrt,
        abs, tanh, ... plus the constants ``pi`` and ``e``.
    params
        Parameter names in order. If omitted, every identifier in the
        expression that is not ``x`` or a known function becomes a parameter,
        in order of first appearance.
    initial_guess
        Starting values (defaults to 1.0 for every parameter).
    variable
        Name of the independent variable inside ``expression``.

    Notes
    -----
    The expression is evaluated with Python's ``eval`` in a namespace that
    only contains NumPy functions and the parameters. Do not evaluate
    expressions from untrusted sources.
    """
    expr = expression.replace("^", "**")
    if params is None:
        seen: list[str] = []
        for ident in _IDENT_RE.findall(expr):
            if ident == variable or ident in _SAFE_NAMES or ident in seen:
                continue
            seen.append(ident)
        params = seen
    params = list(params)
    if not params:
        raise ModelError("Expression has no free parameters.")
    if variable in params:
        raise ModelError(f"'{variable}' is the independent variable, not a parameter.")

    try:
        code = compile(expr, "<sciflow-expression>", "eval")
    except SyntaxError as error:
        raise ModelError(f"Invalid expression '{expression}': {error}") from error

    for ident in code.co_names:
        if ident not in _SAFE_NAMES and ident not in params and ident != variable:
            raise ModelError(
                f"Unknown name '{ident}' in expression. Declare it as a "
                f"parameter or use one of: {sorted(_SAFE_NAMES)}"
            )

    def func(x, *values):
        namespace = dict(_SAFE_NAMES)
        namespace[variable] = x
        namespace.update(zip(params, values))
        return eval(code, {"__builtins__": {}}, namespace)  # noqa: S307

    p0 = list(initial_guess) if initial_guess is not None else [1.0] * len(params)
    if len(p0) != len(params):
        raise ModelError(
            f"initial_guess has {len(p0)} values for {len(params)} parameters."
        )

    return Model(
        name=name,
        func=func,
        param_names=params,
        formula=f"y = {expression}",
        initial_guess=lambda x, y: p0,
        description="User-defined expression.",
    )
