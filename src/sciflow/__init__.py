"""sciflow-core — validated curve fitting for experimental data.

Public API:

    from sciflow import read_table, clean_data, fit, compare_models, diagnostics
    from sciflow.models import get_model, expression_model, multipeak
    from sciflow.plots import plot_fit_with_residuals, plot_diagnostics

This is the open-source core of sciflow (engine, models, diagnostics,
figures). Report generation, the YAML pipeline and the web interface are
part of the full sciflow distribution.
"""

from sciflow.io.readers import read_table
from sciflow.clean.cleaner import clean_data
from sciflow.models.registry import get_model, list_models, register_model
from sciflow.fit.engine import compare_models, fit
from sciflow.fit.diagnostics import diagnostics
from sciflow.fit.result import FitResult

__version__ = "0.4.0"

__all__ = [
    "read_table", "clean_data", "get_model", "list_models", "register_model",
    "fit", "compare_models", "diagnostics", "FitResult", "__version__",
]
