"""Tidy: read several files or sheets, harmonise, combine, convert, write.

Everything that is renamed, converted or removed is recorded in a
:class:`TidyReport` so the delivered table can be audited row by row.

Steps: resolve inputs → read each unit (file or sheet) → rename → check
required columns → tag source → combine (concat or join) → select columns →
convert types → drop rows (missing / duplicates) → sort → write table +
report.
"""

from __future__ import annotations

import glob
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from sciflow.errors import ConfigError, DataValidationError, SciflowError
from sciflow.io.readers import excel_sheets, read_table, sniff_format
from sciflow.tidy.config import TidyConfig, TidyInput, load_tidy_config

log = logging.getLogger("sciflow")
PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #

@dataclass
class UnitReport:
    """One input unit: a text file, or one sheet of a workbook."""

    source: str
    sheet: Optional[str]
    rows: int
    columns_in: list[str]
    columns_out: list[str]
    renamed: dict[str, str]
    sep: Optional[str] = None
    decimal: Optional[str] = None
    skipped_reason: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.source}[{self.sheet}]" if self.sheet is not None else self.source


@dataclass
class TidyReport:
    units: list[UnitReport] = field(default_factory=list)
    combine: str = "concat"
    rows_combined: int = 0
    rows_out: int = 0
    columns_out: list[str] = field(default_factory=list)
    type_conversions: dict[str, str] = field(default_factory=dict)
    type_failures: dict[str, list[int]] = field(default_factory=dict)
    """column -> row positions (in the combined table, 0-based) where a
    non-empty value could not be converted and became missing."""
    dropped_missing: list[int] = field(default_factory=list)
    dropped_duplicates: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def rows_removed(self) -> int:
        return self.rows_combined - self.rows_out

    def as_dict(self) -> dict:
        return {
            "units": [
                {
                    "source": u.source, "sheet": u.sheet, "rows": u.rows,
                    "columns_in": u.columns_in, "columns_out": u.columns_out,
                    "renamed": u.renamed, "sep": u.sep, "decimal": u.decimal,
                    "skipped_reason": u.skipped_reason,
                }
                for u in self.units
            ],
            "combine": self.combine,
            "rows_combined": self.rows_combined,
            "rows_out": self.rows_out,
            "rows_removed": self.rows_removed,
            "columns_out": self.columns_out,
            "type_conversions": self.type_conversions,
            "type_failures": {k: list(map(int, v)) for k, v in self.type_failures.items()},
            "dropped_missing": list(map(int, self.dropped_missing)),
            "dropped_duplicates": list(map(int, self.dropped_duplicates)),
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        used = [u for u in self.units if u.skipped_reason is None]
        lines = [
            f"Inputs: {len(used)} unit(s) read"
            + (f", {len(self.units) - len(used)} skipped" if len(used) < len(self.units) else ""),
            f"Rows combined: {self.rows_combined}, rows out: {self.rows_out} "
            f"(removed {self.rows_removed})",
        ]
        for col, rows in self.type_failures.items():
            lines.append(f"  {col}: {len(rows)} value(s) could not be converted")
        if self.dropped_missing:
            lines.append(f"  missing values: {len(self.dropped_missing)} row(s) dropped")
        if self.dropped_duplicates:
            lines.append(f"  duplicates: {len(self.dropped_duplicates)} row(s) dropped")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


@dataclass
class TidyResult:
    frame: pd.DataFrame
    report: TidyReport
    files: dict[str, Path] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _norm(name: str) -> str:
    """Key used to match column names loosely: case- and whitespace-insensitive."""
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def _apply_rename(frame: pd.DataFrame, mapping: dict[str, str], loose: bool) -> dict[str, str]:
    """Rename columns in place; return the renames actually applied (old -> new)."""
    if not mapping:
        return {}
    applied: dict[str, str] = {}
    if loose:
        by_key = {_norm(old): new for old, new in mapping.items()}
        for col in list(frame.columns):
            new = by_key.get(_norm(col))
            if new is not None and new != col:
                applied[str(col)] = new
    else:
        for col in list(frame.columns):
            if col in mapping and mapping[col] != col:
                applied[str(col)] = mapping[col]
    if applied:
        frame.rename(columns=applied, inplace=True)
        dupes = frame.columns[frame.columns.duplicated()].tolist()
        if dupes:
            raise DataValidationError(
                f"Renaming produced duplicate column names: {sorted(set(map(str, dupes)))}. "
                "Two original columns map to the same new name."
            )
    return applied


def _resolve_units(cfg: TidyConfig) -> list[tuple[TidyInput, Path, Union[str, int, None]]]:
    """Expand every input into (input, file, sheet) units."""
    units: list[tuple[TidyInput, Path, Union[str, int, None]]] = []
    for inp in cfg.inputs:
        candidate = Path(inp.path)
        if not candidate.is_absolute():
            candidate = cfg.base_dir / candidate
        if candidate.is_file():
            files = [candidate]
        else:
            files = sorted(Path(p) for p in glob.glob(str(candidate)))
            files = [f for f in files if f.is_file()]
        if not files:
            raise ConfigError(f"inputs: path matches no file: '{candidate}'.")
        for file in files:
            if isinstance(inp.sheet, str) and inp.sheet.lower() == "all":
                names = excel_sheets(file)
                if not names:
                    units.append((inp, file, 0))  # text file: 'all' is meaningless
                else:
                    units.extend((inp, file, name) for name in names)
            else:
                units.append((inp, file, inp.sheet))
    return units


def _read_unit(cfg: TidyConfig, inp: TidyInput, file: Path, sheet) -> tuple[pd.DataFrame, UnitReport]:
    is_excel = file.suffix.lower() in (".xlsx", ".xlsm", ".xls")
    sep: Optional[str] = None
    decimal = "."
    if not is_excel:
        if inp.decimal == "auto":
            sep, decimal = sniff_format(
                file, skiprows=inp.skiprows, comment=inp.comment, header=inp.header
            )
            if inp.sep is not None:
                sep = inp.sep
        else:
            sep, decimal = inp.sep, inp.decimal
            if sep is None:  # let read_table sniff; record what it will use
                from sciflow.io.readers import _sniff_sep

                sep = _sniff_sep(file, inp.skiprows, inp.comment, decimal)
    frame = read_table(
        file, sep=sep, decimal=decimal, sheet=sheet, skiprows=inp.skiprows,
        comment=inp.comment, header=inp.header, columns=inp.names,
    )
    columns_in = [str(c) for c in frame.columns]
    renamed = _apply_rename(frame, inp.rename or {}, cfg.columns.normalize_names)
    renamed.update(_apply_rename(frame, cfg.rename, cfg.columns.normalize_names))
    report = UnitReport(
        source=file.name,
        sheet=(str(sheet) if sheet is not None else None) if is_excel else None,
        rows=len(frame),
        columns_in=columns_in, columns_out=[str(c) for c in frame.columns],
        renamed=renamed,
        sep=None if is_excel else {r"\s+": "whitespace", "\t": "tab"}.get(sep, sep),
        decimal=None if is_excel else decimal,
    )
    return frame, report


def _convert(series: pd.Series, kind: str, fmt: Optional[str]) -> tuple[pd.Series, list[int]]:
    """Convert a column; return the new series and positions that were lost."""
    before_missing = series.isna().to_numpy()
    if kind == "float":
        out = pd.to_numeric(series, errors="coerce").astype(float)
    elif kind == "int":
        num = pd.to_numeric(series, errors="coerce")
        whole = num.where(np.isclose(num, np.round(num), equal_nan=False))
        out = whole.round().astype("Int64")
    elif kind == "str":
        out = series.astype("string")
        out = out.where(~series.isna(), other=pd.NA)
    elif kind == "bool":
        mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False,
                   "si": True, "sí": True, "y": True, "n": False, "t": True, "f": False}
        as_text = series.astype("string").str.strip().str.lower()
        out = as_text.map(mapping).astype("boolean")
        out = out.where(~series.isna(), other=pd.NA)
    elif kind == "datetime":
        out = pd.to_datetime(series, errors="coerce", format=fmt)
    else:  # pragma: no cover - validated by config
        raise ConfigError(f"Unknown type '{kind}'.")
    after_missing = out.isna().to_numpy()
    lost = np.flatnonzero(after_missing & ~before_missing).tolist()
    return out, lost


def _select_columns(frame: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    missing = [c for c in keep if c not in frame.columns]
    if missing:
        raise DataValidationError(
            f"columns.keep lists {missing}, not present after combining. "
            f"Available: {[str(c) for c in frame.columns]}"
        )
    return frame.loc[:, keep].copy()


# --------------------------------------------------------------------------- #
# Main entry points
# --------------------------------------------------------------------------- #

def tidy(cfg: TidyConfig) -> TidyResult:
    """Run the tidy pipeline in memory (no files written). See :func:`run_tidy`."""
    report = TidyReport(combine=cfg.combine.how)
    frames: list[tuple[pd.DataFrame, UnitReport]] = []

    for inp, file, sheet in _resolve_units(cfg):
        frame, unit = _read_unit(cfg, inp, file, sheet)
        required = cfg.columns.require or []
        missing = [c for c in required if c not in frame.columns]
        if missing:
            msg = (f"{unit.label}: required column(s) {missing} not found after renaming. "
                   f"Columns: {unit.columns_out}")
            if cfg.columns.on_missing == "error":
                raise DataValidationError(msg)
            unit.skipped_reason = msg
            report.units.append(unit)
            report.warnings.append("skipped " + msg)
            log.warning("tidy: skipped %s", msg)
            continue
        report.units.append(unit)
        frames.append((frame, unit))
        log.info("tidy: %s: %d rows, columns %s", unit.label, len(frame), unit.columns_out)

    if not frames:
        raise DataValidationError("No input could be used (all skipped or empty).")

    # Combine ----------------------------------------------------------------
    comb = cfg.combine
    if comb.how == "concat":
        parts = []
        for frame, unit in frames:
            if comb.source_column:
                if comb.source_column in frame.columns:
                    raise DataValidationError(
                        f"{unit.label}: the source column name '{comb.source_column}' already "
                        "exists in the data. Choose another combine.source_column."
                    )
                frame = frame.copy()
                frame.insert(0, comb.source_column, unit.label)
            parts.append(frame)
        combined = pd.concat(parts, ignore_index=True, sort=False)
        columns_union = list(dict.fromkeys(str(c) for f, _ in frames for c in f.columns))
        for frame, unit in frames:
            absent = [c for c in columns_union if c not in frame.columns]
            if absent:
                report.warnings.append(
                    f"{unit.label}: has no column(s) {absent}; filled with missing values."
                )
    else:
        keys = comb.keys or []
        combined = None
        for frame, unit in frames:
            missing = [k for k in keys if k not in frame.columns]
            if missing:
                raise DataValidationError(
                    f"{unit.label}: join key(s) {missing} not found. Columns: {unit.columns_out}"
                )
            if combined is None:
                combined = frame
                continue
            stem = Path(unit.source).stem + (f"_{unit.sheet}" if unit.sheet else "")
            overlap = [c for c in frame.columns if c in combined.columns and c not in keys]
            combined = combined.merge(frame, on=keys, how=comb.join, suffixes=("", f"_{stem}"))
            if overlap:
                report.warnings.append(
                    f"{unit.label}: column(s) {overlap} also exist in earlier inputs; "
                    f"suffixed with _{stem}."
                )
        combined = combined.reset_index(drop=True)
    report.rows_combined = len(combined)

    # Columns ----------------------------------------------------------------
    if cfg.columns.keep:
        combined = _select_columns(combined, cfg.columns.keep)

    for col, kind in cfg.columns.types.items():
        if col not in combined.columns:
            raise DataValidationError(
                f"columns.types names '{col}', which is not in the table. "
                f"Available: {[str(c) for c in combined.columns]}"
            )
        fmt = cfg.columns.datetime_format
        if isinstance(fmt, dict):
            fmt = fmt.get(col)
        converted, lost = _convert(combined[col], kind, fmt if kind == "datetime" else None)
        combined[col] = converted
        report.type_conversions[col] = kind
        if lost:
            report.type_failures[col] = lost

    # Rows -------------------------------------------------------------------
    rows = cfg.rows
    if rows.dropna is not None:
        if rows.dropna in ("any", "all"):
            mask = combined.isna().any(axis=1) if rows.dropna == "any" else combined.isna().all(axis=1)
        else:
            missing_cols = [c for c in rows.dropna if c not in combined.columns]
            if missing_cols:
                raise DataValidationError(f"rows.dropna names unknown column(s) {missing_cols}.")
            mask = combined[rows.dropna].isna().any(axis=1)
        report.dropped_missing = np.flatnonzero(mask.to_numpy()).tolist()
        combined = combined.loc[~mask]

    if rows.drop_duplicates:
        subset = rows.duplicates_on
        if subset:
            missing_cols = [c for c in subset if c not in combined.columns]
            if missing_cols:
                raise DataValidationError(f"rows.duplicates_on names unknown column(s) {missing_cols}.")
        mask = combined.duplicated(subset=subset, keep="first")
        report.dropped_duplicates = combined.index[mask].tolist()
        combined = combined.loc[~mask]

    if rows.sort_by:
        missing_cols = [c for c in rows.sort_by if c not in combined.columns]
        if missing_cols:
            raise DataValidationError(f"rows.sort_by names unknown column(s) {missing_cols}.")
        combined = combined.sort_values(rows.sort_by, kind="stable")

    combined = combined.reset_index(drop=True)
    report.rows_out = len(combined)
    report.columns_out = [str(c) for c in combined.columns]
    return TidyResult(frame=combined, report=report)


def run_tidy(config: Union[TidyConfig, PathLike]) -> TidyResult:
    """Run the tidy pipeline and write the table and the report to disk."""
    cfg = load_tidy_config(config) if not isinstance(config, TidyConfig) else config
    result = tidy(cfg)

    out_path = Path(cfg.output.path)
    if not out_path.is_absolute():
        out_path = cfg.base_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = result.frame

    try:
        if out_path.suffix.lower() == ".xlsx":
            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                frame.to_excel(writer, sheet_name=cfg.output.sheet_name, index=False)
                if cfg.output.report:
                    _report_frame(result.report).to_excel(writer, sheet_name="report", index=False)
        else:
            sep = "\t" if out_path.suffix.lower() == ".tsv" else cfg.output.sep
            frame.to_csv(out_path, index=False, sep=sep, decimal=cfg.output.decimal)
    except ImportError as error:
        raise SciflowError(f"Cannot write '{out_path.name}': {error}") from error
    result.files["table"] = out_path

    if cfg.output.report:
        stem = out_path.with_suffix("")
        md_path = Path(f"{stem}_tidy_report.md")
        md_path.write_text(tidy_report_markdown(result.report, cfg.output.language), encoding="utf-8")
        json_path = Path(f"{stem}_tidy_report.json")
        json_path.write_text(json.dumps(result.report.as_dict(), indent=2, ensure_ascii=False),
                             encoding="utf-8")
        result.files["report"] = md_path
        result.files["json"] = json_path
    return result


def tidy_files(
    paths: list[PathLike],
    output: PathLike,
    *,
    rename: Optional[dict[str, str]] = None,
    keep: Optional[list[str]] = None,
    require: Optional[list[str]] = None,
    sheet: Union[str, int, None] = 0,
    decimal: str = "auto",
    sep: Optional[str] = None,
    skiprows: int = 0,
    source_column: Optional[str] = "source",
    drop_duplicates: bool = False,
    types: Optional[dict[str, str]] = None,
    language: str = "en",
    report: bool = True,
) -> TidyResult:
    """Quick concat of several files without a config file (used by the CLI)."""
    from sciflow.tidy.config import (
        ColumnsSection, CombineSection, RowsSection, TidyOutput, tidy_config_from_dict,
    )

    raw = {
        "inputs": [
            {"path": str(Path(p).resolve()), "sheet": sheet, "decimal": decimal,
             "sep": sep, "skiprows": skiprows}
            for p in paths
        ],
        "rename": rename or {},
        "columns": {"keep": keep, "require": require, "types": types or {}},
        "rows": {"drop_duplicates": drop_duplicates},
        "combine": {"how": "concat", "source_column": source_column},
        "output": {"path": str(Path(output).resolve()), "report": report, "language": language},
    }
    cfg = tidy_config_from_dict(raw, base_dir=Path.cwd())
    return run_tidy(cfg)


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #

_T = {
    "en": {
        "title": "Tidy report", "inputs": "Inputs", "unit": "Input", "rows": "Rows",
        "delimiter": "Delimiter", "decimal": "Decimal", "renamed": "Renamed columns",
        "skipped": "SKIPPED", "none": "none", "combine": "Combination",
        "concat": "rows stacked (concat)", "join": "merged on key columns (join)",
        "rows_combined": "Rows after combining", "rows_out": "Rows delivered",
        "removed": "removed", "columns_out": "Columns delivered", "types": "Type conversions",
        "type_failures": "Values that could not be converted (became missing)",
        "positions": "row positions in the combined table, 0-based",
        "dropped_missing": "Rows dropped for missing values", "dropped_dup": "Rows dropped as duplicates",
        "warnings": "Warnings", "no_warnings": "No warnings.",
        "note": "Row positions refer to the combined table before any row was dropped.",
    },
    "es": {
        "title": "Informe de unificación", "inputs": "Entradas", "unit": "Entrada", "rows": "Filas",
        "delimiter": "Separador", "decimal": "Decimal", "renamed": "Columnas renombradas",
        "skipped": "OMITIDA", "none": "ninguna", "combine": "Combinación",
        "concat": "filas apiladas (concat)", "join": "cruce por columnas clave (join)",
        "rows_combined": "Filas tras combinar", "rows_out": "Filas entregadas",
        "removed": "eliminadas", "columns_out": "Columnas entregadas", "types": "Conversiones de tipo",
        "type_failures": "Valores que no se pudieron convertir (pasan a vacío)",
        "positions": "posiciones de fila en la tabla combinada, desde 0",
        "dropped_missing": "Filas eliminadas por valores vacíos",
        "dropped_dup": "Filas eliminadas por duplicadas",
        "warnings": "Avisos", "no_warnings": "Sin avisos.",
        "note": "Las posiciones de fila se refieren a la tabla combinada antes de eliminar filas.",
    },
}


def _fmt_positions(rows: list[int], limit: int = 30) -> str:
    shown = ", ".join(str(r) for r in rows[:limit])
    return shown + (f" … (+{len(rows) - limit})" if len(rows) > limit else "")


def tidy_report_markdown(report: TidyReport, language: str = "en") -> str:
    t = _T.get(language, _T["en"])
    lines = [f"# {t['title']}", ""]
    lines.append(f"## {t['inputs']}")
    lines.append("")
    lines.append(f"| {t['unit']} | {t['rows']} | {t['delimiter']} | {t['decimal']} | {t['renamed']} |")
    lines.append("|---|---:|---|---|---|")
    for u in report.units:
        renamed = ", ".join(f"{k} → {v}" for k, v in u.renamed.items()) or t["none"]
        rows = f"{u.rows}" if u.skipped_reason is None else f"{t['skipped']}"
        lines.append(f"| {u.label} | {rows} | {u.sep or '—'} | {u.decimal or '—'} | {renamed} |")
    lines.append("")
    lines.append(f"## {t['combine']}")
    lines.append("")
    lines.append(f"- {t[report.combine]}")
    lines.append(f"- {t['rows_combined']}: {report.rows_combined}")
    lines.append(f"- {t['rows_out']}: {report.rows_out} ({report.rows_removed} {t['removed']})")
    lines.append(f"- {t['columns_out']}: {', '.join(report.columns_out)}")
    lines.append("")
    if report.type_conversions:
        lines.append(f"## {t['types']}")
        lines.append("")
        for col, kind in report.type_conversions.items():
            lines.append(f"- {col}: {kind}")
        lines.append("")
    if report.type_failures:
        lines.append(f"## {t['type_failures']}")
        lines.append("")
        lines.append(f"({t['positions']})")
        lines.append("")
        for col, rows in report.type_failures.items():
            lines.append(f"- {col}: {len(rows)} — {_fmt_positions(rows)}")
        lines.append("")
    if report.dropped_missing:
        lines.append(f"## {t['dropped_missing']}")
        lines.append("")
        lines.append(f"{len(report.dropped_missing)} — {_fmt_positions(report.dropped_missing)}")
        lines.append("")
    if report.dropped_duplicates:
        lines.append(f"## {t['dropped_dup']}")
        lines.append("")
        lines.append(f"{len(report.dropped_duplicates)} — {_fmt_positions(report.dropped_duplicates)}")
        lines.append("")
    lines.append(f"## {t['warnings']}")
    lines.append("")
    if report.warnings:
        lines.extend(f"- {w}" for w in report.warnings)
    else:
        lines.append(t["no_warnings"])
    lines.append("")
    lines.append(f"_{t['note']}_")
    lines.append("")
    return "\n".join(lines)


def _report_frame(report: TidyReport) -> pd.DataFrame:
    """Flat key/value table of the report (for the 'report' sheet of an xlsx)."""
    rows: list[tuple[str, str]] = []
    for u in report.units:
        status = "skipped" if u.skipped_reason else str(u.rows)
        rows.append((f"input {u.label}", f"rows={status}; sep={u.sep or '-'}; decimal={u.decimal or '-'}; "
                     f"renamed={u.renamed or 'none'}"))
    rows.append(("combine", report.combine))
    rows.append(("rows_combined", str(report.rows_combined)))
    rows.append(("rows_out", str(report.rows_out)))
    rows.append(("rows_removed", str(report.rows_removed)))
    rows.append(("columns_out", ", ".join(report.columns_out)))
    for col, kind in report.type_conversions.items():
        rows.append((f"type {col}", kind))
    for col, pos in report.type_failures.items():
        rows.append((f"type_failures {col}", f"{len(pos)}: {_fmt_positions(pos)}"))
    if report.dropped_missing:
        rows.append(("dropped_missing", f"{len(report.dropped_missing)}: {_fmt_positions(report.dropped_missing)}"))
    if report.dropped_duplicates:
        rows.append(("dropped_duplicates",
                     f"{len(report.dropped_duplicates)}: {_fmt_positions(report.dropped_duplicates)}"))
    for w in report.warnings:
        rows.append(("warning", w))
    return pd.DataFrame(rows, columns=["item", "value"])
