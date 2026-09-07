"""Data cleaning with a full, auditable report of what was removed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CleanReport:
    """What ``clean_data`` did. Row indices refer to the *original* frame."""

    original_rows: int
    final_rows: int
    non_numeric_rows: list[int] = field(default_factory=list)
    missing_rows: list[int] = field(default_factory=list)
    non_finite_rows: list[int] = field(default_factory=list)
    duplicate_rows: list[int] = field(default_factory=list)
    columns_checked: list[str] = field(default_factory=list)

    @property
    def removed_rows(self) -> int:
        return self.original_rows - self.final_rows

    def as_dict(self) -> dict:
        return {
            "original_rows": self.original_rows,
            "final_rows": self.final_rows,
            "removed_rows": self.removed_rows,
            "non_numeric_rows": self.non_numeric_rows,
            "missing_rows": self.missing_rows,
            "non_finite_rows": self.non_finite_rows,
            "duplicate_rows": self.duplicate_rows,
            "columns_checked": self.columns_checked,
        }

    def summary(self) -> str:
        lines = [
            f"Rows in: {self.original_rows}, rows out: {self.final_rows} "
            f"(removed {self.removed_rows})"
        ]
        for label, rows in (
            ("non-numeric", self.non_numeric_rows),
            ("missing", self.missing_rows),
            ("non-finite", self.non_finite_rows),
            ("duplicate", self.duplicate_rows),
        ):
            if rows:
                shown = ", ".join(str(r) for r in rows[:10])
                more = f" ... (+{len(rows) - 10})" if len(rows) > 10 else ""
                lines.append(f"  {label}: {len(rows)} row(s) [{shown}{more}]")
        return "\n".join(lines)


def clean_data(
    data: pd.DataFrame,
    columns: Optional[list[str]] = None,
    *,
    drop_duplicates: bool = False,
) -> tuple[pd.DataFrame, CleanReport]:
    """Return a numeric, finite copy of ``data`` plus a report.

    Parameters
    ----------
    data
        Input frame.
    columns
        Columns that must be numeric and finite. Defaults to all columns.
        Other columns are kept untouched.
    drop_duplicates
        Remove exactly repeated rows (over ``columns``). Off by default:
        repeated measurements are legitimate in experimental data and removing
        them changes the degrees of freedom of any subsequent fit.

    Notes
    -----
    Values that cannot be converted to a number (e.g. ``"error"``) are treated
    as missing and the row is reported under ``non_numeric_rows``.
    """
    if columns is None:
        columns = list(data.columns)
    missing_cols = [c for c in columns if c not in data.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in data: {missing_cols}")

    original_index = data.index.to_list()
    frame = data.copy()

    # 1. Coerce to numeric, tracking which rows were non-numeric strings.
    non_numeric = set()
    missing = set()
    for col in columns:
        original = frame[col]
        coerced = pd.to_numeric(original, errors="coerce")
        was_missing = original.isna()
        became_nan = coerced.isna() & ~was_missing
        non_numeric.update(frame.index[became_nan].to_list())
        missing.update(frame.index[was_missing].to_list())
        frame[col] = coerced.astype(float)

    bad_rows = non_numeric | missing
    frame = frame.drop(index=list(bad_rows))

    # 2. Non-finite values (inf, -inf).
    numeric_block = frame[columns].to_numpy(dtype=float)
    finite_mask = np.isfinite(numeric_block).all(axis=1)
    non_finite = frame.index[~finite_mask].to_list()
    frame = frame[finite_mask]

    # 3. Optional duplicates.
    duplicates: list[int] = []
    if drop_duplicates:
        dup_mask = frame.duplicated(subset=columns, keep="first")
        duplicates = frame.index[dup_mask].to_list()
        frame = frame[~dup_mask]

    frame = frame.reset_index(drop=True)

    report = CleanReport(
        original_rows=len(original_index),
        final_rows=len(frame),
        non_numeric_rows=sorted(non_numeric),
        missing_rows=sorted(missing - non_numeric),
        non_finite_rows=sorted(non_finite),
        duplicate_rows=sorted(duplicates),
        columns_checked=list(columns),
    )
    return frame, report
