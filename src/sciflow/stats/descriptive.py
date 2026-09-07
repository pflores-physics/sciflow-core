"""Descriptive statistics for one or more columns."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def describe(values: Sequence[float]) -> dict[str, float]:
    """Summary statistics of a 1-D sample (NaN ignored)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return {"n": 0}
    std = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    return {
        "n": int(n),
        "mean": float(np.mean(arr)),
        "std": std,
        "sem": std / np.sqrt(n) if n > 1 else float("nan"),
        "min": float(arr.min()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
    }


def describe_columns(
    data: pd.DataFrame, columns: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """One row of :func:`describe` per column."""
    if columns is None:
        columns = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    rows = {col: describe(data[col].to_numpy()) for col in columns}
    return pd.DataFrame(rows).T
