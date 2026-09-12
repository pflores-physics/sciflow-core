"""Readers for tabular experimental data.

Supported formats: CSV / TSV / TXT (any delimiter, auto-detected by default),
and Excel (``.xlsx``/``.xls``, requires ``openpyxl``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from sciflow.errors import DataImportError

PathLike = Union[str, Path]

_EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
_TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".dat"}
_CANDIDATE_SEPS = [",", ";", "\t", "|"]
_WHITESPACE = r"\s+"

_INT_RE = re.compile(r"^[+-]?\d+$")
_DOT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?$")
_COMMA_RE = re.compile(r"^[+-]?(\d+,\d*|,\d+)([eE][+-]?\d+)?$")


def _sample_lines(path: Path, skiprows: int, comment: Optional[str], limit: int = 20) -> list[str]:
    """First ``limit`` non-blank, non-comment lines after ``skiprows``."""
    lines: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for i, line in enumerate(handle):
            if i < skiprows:
                continue
            stripped = line.strip()
            if not stripped or (comment and stripped.startswith(comment)):
                continue
            lines.append(stripped)
            if len(lines) >= limit:
                break
    return lines


def _sniff_sep(path: Path, skiprows: int, comment: Optional[str], decimal: str) -> str:
    """Guess the delimiter from the first data lines.

    Comment lines and blank lines are ignored (pandas' own sniffer chokes on
    them). A candidate delimiter wins when it appears the same number of
    times (≥ 1) on every sampled line; otherwise the file is treated as
    whitespace-separated (one or more spaces/tabs), which also covers
    fixed-width instrument dumps.
    """
    lines = _sample_lines(path, skiprows, comment)
    for cand in _CANDIDATE_SEPS:
        if cand == decimal:
            continue
        counts = {line.count(cand) for line in lines}
        if len(counts) == 1 and counts.pop() >= 1:
            return cand
    return _WHITESPACE


def _split(line: str, sep: str) -> list[str]:
    parts = re.split(_WHITESPACE, line) if sep == _WHITESPACE else line.split(sep)
    return [p.strip().strip('"').strip("'") for p in parts]


def sniff_format(
    path: PathLike,
    *,
    skiprows: int = 0,
    comment: Optional[str] = "#",
    header: Optional[int] = 0,
) -> tuple[str, str]:
    """Guess ``(sep, decimal)`` of a text file by looking at its data lines.

    Every candidate delimiter that splits all sampled lines into the same
    number of fields is scored by the fraction of fields that look like
    numbers (integers, ``1.5``-style or ``1,5``-style). The best-scoring
    delimiter wins; ties go to the conventional order comma, semicolon, tab,
    bar, whitespace. The decimal marker is then whichever style is more
    frequent among the numeric fields. A file such as ``1,5;2,3`` is therefore
    read as semicolon-separated with decimal comma, not as three comma-separated
    columns.

    Returns ``(sep, decimal)``; ``sep`` is ``r"\\s+"`` for whitespace-separated
    files. If nothing can be decided (no data lines), returns ``(",", ".")``.
    """
    path = Path(path)
    lines = _sample_lines(path, skiprows, comment)
    if not lines:
        return ",", "."
    # Skip the header line (if any) when scoring: column names are not numbers.
    data_lines = lines[1:] if header is not None and len(lines) > 1 else lines

    def score(cand: str) -> tuple[float, int, int]:
        fields = [f for line in data_lines for f in _split(line, cand)]
        if not fields:
            return 0.0, 0, 0
        ints = sum(bool(_INT_RE.match(f)) for f in fields)
        dots = sum(bool(_DOT_RE.match(f)) for f in fields)
        commas = 0 if cand == "," else sum(bool(_COMMA_RE.match(f)) for f in fields)
        return (ints + dots + commas) / len(fields), commas, dots

    # Explicit delimiters first (whitespace is only the fallback, as in
    # _sniff_sep): a lone field such as "1,10" is otherwise indistinguishable
    # from a decimal comma.
    best: Optional[tuple[float, str, int, int]] = None  # (score, sep, comma_hits, dot_hits)
    for cand in _CANDIDATE_SEPS:
        counts = {line.count(cand) for line in lines}
        if not (len(counts) == 1 and counts.pop() >= 1):
            continue
        value, commas, dots = score(cand)
        if best is None or value > best[0] + 1e-9:
            best = (value, cand, commas, dots)
    if best is None:
        value, commas, dots = score(_WHITESPACE)
        best = (value, _WHITESPACE, commas, dots)
    _, sep, commas, dots = best
    decimal = "," if commas > dots else "."
    return sep, decimal


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
        Decimal marker (``","`` for many European instruments). ``"auto"``
        detects it together with the delimiter (see :func:`sniff_format`).
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
            if decimal == "auto":
                guessed_sep, decimal = sniff_format(
                    path, skiprows=skiprows, comment=comment, header=header
                )
                if sep is None:
                    sep = guessed_sep
            elif sep is None:
                sep = _sniff_sep(path, skiprows, comment, decimal)
            data = pd.read_csv(
                path,
                sep=sep,
                engine="python" if sep == _WHITESPACE else None,
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


def excel_sheets(path: PathLike) -> list[str]:
    """Names of the sheets in an Excel workbook (empty list for text files)."""
    path = Path(path)
    if path.suffix.lower() not in _EXCEL_SUFFIXES:
        return []
    try:
        with pd.ExcelFile(path) as book:
            return list(book.sheet_names)
    except ImportError as error:
        raise DataImportError(
            f"Missing optional dependency to read '{path.name}': {error}"
        ) from error
    except Exception as error:
        raise DataImportError(f"Could not open '{path}': {error}") from error
