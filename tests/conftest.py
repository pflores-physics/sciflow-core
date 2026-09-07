import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def line_data(rng):
    """y = 1.5 + 2.0 x with known Gaussian noise."""
    x = np.linspace(0, 10, 30)
    sigma = np.full_like(x, 0.2)
    y = 1.5 + 2.0 * x + rng.normal(0, sigma)
    return x, y, sigma


@pytest.fixture
def dirty_frame():
    return pd.DataFrame({
        "X": [1, 2, 3, 4, 4, 5, 6, 7],
        "Y": [2.0, 4.1, "error", 8.0, 8.0, np.nan, np.inf, 14.2],
        "S": [0.1] * 8,
        "label": list("abcdefgh"),
    })
