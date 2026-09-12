"""Configuration for ``sciflow tidy``: dataclasses + loader for YAML/JSON.

Example (YAML)::

    inputs:
      - path: data/*.csv          # a file or a glob pattern
        decimal: auto             # auto-detect delimiter and decimal marker
      - path: data/book.xlsx
        sheet: all                # every sheet of the workbook
    rename:                       # harmonise column names (old: new)
      "Temp (C)": temperature
      temp_c: temperature
    columns:
      keep: [source, temperature, resistance]
      require: [temperature, resistance]
      on_missing: error           # or skip: leave that file/sheet out
      types:
        temperature: float
        date: datetime
      datetime_format: "%d/%m/%Y"
    rows:
      dropna: [temperature]       # any | all | list of columns
      drop_duplicates: false
      sort_by: [temperature]
    combine:
      how: concat                 # concat (stack rows) | join (merge on keys)
      keys: null                  # key columns for join
      join: outer                 # inner | outer | left | right
      source_column: source       # column with the origin file (concat only)
    output:
      path: tidy/merged.csv       # .csv or .xlsx
      report: true
      language: en
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from sciflow.errors import ConfigError

PathLike = Union[str, Path]

VALID_TYPES = ("float", "int", "str", "datetime", "bool")


@dataclass
class TidyInput:
    path: str
    sheet: Union[str, int, None] = 0
    sep: Optional[str] = None
    decimal: str = "auto"
    skiprows: int = 0
    comment: Optional[str] = "#"
    header: Optional[int] = 0
    names: Optional[list[str]] = None
    rename: Optional[dict[str, str]] = None


@dataclass
class ColumnsSection:
    keep: Optional[list[str]] = None
    require: Optional[list[str]] = None
    on_missing: str = "error"
    types: dict[str, str] = field(default_factory=dict)
    datetime_format: Union[str, dict[str, str], None] = None
    normalize_names: bool = True


@dataclass
class RowsSection:
    dropna: Union[str, list[str], None] = None
    drop_duplicates: bool = False
    duplicates_on: Optional[list[str]] = None
    sort_by: Optional[list[str]] = None


@dataclass
class CombineSection:
    how: str = "concat"
    keys: Optional[list[str]] = None
    join: str = "outer"
    source_column: Optional[str] = "source"


@dataclass
class TidyOutput:
    path: str = "tidy/merged.csv"
    report: bool = True
    language: str = "en"
    sep: str = ","
    decimal: str = "."
    sheet_name: str = "data"


@dataclass
class TidyConfig:
    inputs: list[TidyInput]
    rename: dict[str, str] = field(default_factory=dict)
    columns: ColumnsSection = field(default_factory=ColumnsSection)
    rows: RowsSection = field(default_factory=RowsSection)
    combine: CombineSection = field(default_factory=CombineSection)
    output: TidyOutput = field(default_factory=TidyOutput)
    base_dir: Path = field(default_factory=Path, repr=False)

    def as_dict(self) -> dict:
        data = asdict(self)
        data.pop("base_dir", None)
        return data


# --------------------------------------------------------------------------- #

def _build(cls, section: Any, name: str):
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ConfigError(f"Section '{name}' must be a mapping, got {type(section).__name__}.")
    allowed = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(section) - allowed
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in '{name}': {sorted(unknown)}. Allowed: {sorted(allowed)}."
        )
    try:
        return cls(**section)
    except TypeError as error:
        raise ConfigError(f"Section '{name}': {error}") from error


def _as_str_list(value, name: str) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ConfigError(f"'{name}' must be a column name or a list of column names.")


def tidy_config_from_dict(raw: dict, base_dir: PathLike = ".") -> TidyConfig:
    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping.")
    if "inputs" not in raw:
        raise ConfigError("Missing required section 'inputs'.")
    known = {"inputs", "rename", "columns", "rows", "combine", "output"}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"Unknown top-level section(s): {sorted(unknown)}.")

    raw_inputs = raw["inputs"]
    if isinstance(raw_inputs, (str, dict)):
        raw_inputs = [raw_inputs]
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ConfigError("'inputs' must be a non-empty list of files (or mappings with 'path').")
    inputs: list[TidyInput] = []
    for i, item in enumerate(raw_inputs):
        if isinstance(item, str):
            item = {"path": item}
        inp = _build(TidyInput, item, f"inputs[{i}]")
        if inp.rename is not None and not isinstance(inp.rename, dict):
            raise ConfigError(f"inputs[{i}].rename must be a mapping old: new.")
        inputs.append(inp)

    rename = raw.get("rename") or {}
    if not isinstance(rename, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in rename.items()
    ):
        raise ConfigError("'rename' must be a mapping of old column name: new column name.")

    cfg = TidyConfig(
        inputs=inputs,
        rename=dict(rename),
        columns=_build(ColumnsSection, raw.get("columns"), "columns"),
        rows=_build(RowsSection, raw.get("rows"), "rows"),
        combine=_build(CombineSection, raw.get("combine"), "combine"),
        output=_build(TidyOutput, raw.get("output"), "output"),
        base_dir=Path(base_dir),
    )

    cols = cfg.columns
    cols.keep = _as_str_list(cols.keep, "columns.keep")
    cols.require = _as_str_list(cols.require, "columns.require")
    if cols.on_missing not in ("error", "skip"):
        raise ConfigError("columns.on_missing must be 'error' or 'skip'.")
    if not isinstance(cols.types, dict):
        raise ConfigError("columns.types must be a mapping column: type.")
    for col, kind in cols.types.items():
        if kind not in VALID_TYPES:
            raise ConfigError(
                f"columns.types['{col}'] = '{kind}' is not valid. Use one of {VALID_TYPES}."
            )
    if cols.datetime_format is not None and not isinstance(cols.datetime_format, (str, dict)):
        raise ConfigError("columns.datetime_format must be a string or a mapping column: format.")

    rows = cfg.rows
    if rows.dropna is not None and rows.dropna not in ("any", "all"):
        rows.dropna = _as_str_list(rows.dropna, "rows.dropna")
    rows.duplicates_on = _as_str_list(rows.duplicates_on, "rows.duplicates_on")
    rows.sort_by = _as_str_list(rows.sort_by, "rows.sort_by")

    comb = cfg.combine
    if comb.how not in ("concat", "join"):
        raise ConfigError("combine.how must be 'concat' or 'join'.")
    comb.keys = _as_str_list(comb.keys, "combine.keys")
    if comb.how == "join" and not comb.keys:
        raise ConfigError("combine.how is 'join' but combine.keys (key columns) is empty.")
    if comb.join not in ("inner", "outer", "left", "right"):
        raise ConfigError("combine.join must be inner, outer, left or right.")

    out = cfg.output
    if out.language not in ("en", "es"):
        raise ConfigError("output.language must be 'en' or 'es'.")
    if Path(out.path).suffix.lower() not in (".csv", ".xlsx", ".txt", ".tsv"):
        raise ConfigError("output.path must end in .csv, .tsv, .txt or .xlsx.")
    return cfg


def load_tidy_config(path: PathLike) -> TidyConfig:
    """Load a ``.yaml``/``.yml`` or ``.json`` tidy file.

    Relative paths inside the file are resolved against the file's folder.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: '{path}'.")
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml  # lazy: keeps pyyaml optional for library use

            raw = yaml.safe_load(text)
        elif path.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            raise ConfigError("Config file must be .yaml, .yml or .json.")
    except ImportError as error:
        raise ConfigError("PyYAML is required for YAML configs: pip install pyyaml") from error
    except ConfigError:
        raise
    except Exception as error:  # yaml/json parse errors
        raise ConfigError(f"Could not parse '{path}': {error}") from error
    return tidy_config_from_dict(raw, base_dir=path.parent)


EXAMPLE_TIDY_CONFIG = """\
# sciflow tidy configuration: read several files (or sheets), harmonise their
# columns, stack or join them, convert types and write ONE clean table plus a
# report of everything that was changed or removed.
# Run with:  sciflow tidy this_file.yaml

inputs:                          # one entry per file or pattern
  - path: data/*.csv             # relative to this YAML
    decimal: auto                # detects "1.5" vs "1,5" and the delimiter
    # sep: ";"                   # force the delimiter if detection fails
    # skiprows: 3                # instrument header lines to skip
    # rename: {"T[C]": temperature}   # renames applied to THIS input only
  # - path: data/book.xlsx
  #   sheet: all                 # all sheets; or a name / index (0 = first)

rename:                          # global harmonisation, old name: new name
  "Temp (C)": temperature        # matching ignores case and extra spaces
  temp_c: temperature
  "Resistance (Ohm)": resistance

columns:
  keep: [source, temperature, resistance]   # final columns, in this order
  require: [temperature, resistance]        # must exist in every file after rename
  on_missing: error              # error | skip (leave that file out, with a warning)
  types:                         # float | int | str | datetime | bool
    temperature: float
    resistance: float
    # date: datetime
  # datetime_format: "%d/%m/%Y"  # one format, or a mapping column: format

rows:
  dropna: [temperature, resistance]   # any | all | list of columns
  drop_duplicates: false
  # duplicates_on: [temperature]
  sort_by: [temperature]

combine:
  how: concat                    # concat: stack rows | join: merge on key columns
  # keys: [sample_id]            # key columns for join
  # join: outer                  # inner | outer | left | right
  source_column: source          # origin file (and sheet) of every row

output:
  path: tidy/merged.csv          # .csv (or .xlsx: adds a 'report' sheet)
  report: true                   # writes merged_tidy_report.md and .json
  language: en                   # en | es
"""
