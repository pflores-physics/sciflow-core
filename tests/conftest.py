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


@pytest.fixture
def mixed_files(tmp_path):
    """Three sources with the same physics and different headers/formats."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.csv").write_text(
        "Temp (C);Resistance (Ohm)\n20,5;100,1\n21,0;100,3\n", encoding="utf-8"
    )
    (data / "b.txt").write_text("temp_c\tR\n22.0\t100.5\nabc\t100.7\n", encoding="utf-8")
    book = data / "c.xlsx"
    with pd.ExcelWriter(book, engine="openpyxl") as writer:
        pd.DataFrame({"TEMP (c)": [23.0, 24.0], "Resistance (Ohm)": [100.9, np.nan]}).to_excel(
            writer, sheet_name="run1", index=False
        )
        pd.DataFrame({"Temp (C)": [25.0], "Resistance (Ohm)": [101.3]}).to_excel(
            writer, sheet_name="run2", index=False
        )
    return data
