"""Readers for tabular experimental data.

Supported formats: CSV / TSV / TXT (any delimiter, auto-detected by default),
and Excel (``.xlsx``/``.xls``, requires ``openpyxl``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from sciflow.errors import DataImportError

PathLike = Union[str, Path]

_EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
_TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".dat"}


def read_table(
    path: PathLike,
    *,
    sep: Optional[str] = None,
    decimal: str = ".",
    sheet: Union[str, int, None] = 0,
    skiprows: int = 0,
    comment: Optional[str] = "#",
    header: Union[int, None] = 0,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Read a data file into a DataFrame.

    Parameters
    ----------
    path
        File to read.
    sep
        Column delimiter for text files. ``None`` lets pandas sniff it
        (handles comma, semicolon, tab and whitespace-separated files).
    decimal
        Decimal marker (``","`` for many European instruments).
    sheet
        Sheet name or index for Excel files.
    skiprows
        Number of leading lines to skip (instrument headers, etc.).
    comment
        Lines starting with this character are ignored. ``None`` disables it.
    header
        Row index holding the column names, or ``None`` if the file has none.
    columns
        Names to assign to the columns (useful with ``header=None``).

    Raises
    ------
    DataImportError
        If the file does not exist, is empty, or cannot be parsed.
    """
    path = Path(path)
    if not path.is_file():
        raise DataImportError(f"File not found: '{path}'.")

    suffix = path.suffix.lower()
    try:
        if suffix in _EXCEL_SUFFIXES:
            data = pd.read_excel(
                path, sheet_name=sheet, skiprows=skiprows, header=header
            )
        else:
            if suffix not in _TEXT_SUFFIXES:
                # Unknown extension: try as delimited text anyway.
                pass
            data = pd.read_csv(
                path,
                sep=sep,
                engine="python" if sep is None else None,
                decimal=decimal,
                skiprows=skiprows,
                comment=comment,
                header=header,
            )
    except ImportError as error:  # e.g. openpyxl missing
        raise DataImportError(
            f"Missing optional dependency to read '{path.name}': {error}"
        ) from error
    except Exception as error:
        raise DataImportError(
            f"Could not parse '{path}': {error}"
        ) from error

    if data.empty:
        raise DataImportError(f"File '{path}' contains no data rows.")

    if columns is not None:
        if len(columns) != data.shape[1]:
            raise DataImportError(
                f"'columns' has {len(columns)} names but the file has "
                f"{data.shape[1]} columns."
            )
        data.columns = list(columns)
    else:
        # Normalise header whitespace ("  X " -> "X").
        data.columns = [str(c).strip() for c in data.columns]

    return data
