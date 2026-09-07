"""Exception hierarchy for sciflow.

Every error raised on purpose by the package derives from ``SciflowError`` so
callers (and the CLI) can catch one type and print a clean message.
"""


class SciflowError(Exception):
    """Base class for all sciflow errors."""


class DataImportError(SciflowError):
    """The input file could not be read or parsed."""


class DataValidationError(SciflowError):
    """The data does not satisfy the requirements of the requested analysis."""


class ModelError(SciflowError):
    """Unknown model, bad parameters or a custom expression that fails."""


class FitError(SciflowError):
    """The fit could not be completed (no convergence, singular covariance...)."""


class ConfigError(SciflowError):
    """A pipeline configuration file is missing keys or has invalid values."""
