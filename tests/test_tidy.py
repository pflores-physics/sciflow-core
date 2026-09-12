"""Tests for ``sciflow tidy`` and the automatic delimiter/decimal detection."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sciflow.errors import ConfigError, DataValidationError
from sciflow.io.readers import read_table, sniff_format
from sciflow.tidy import run_tidy, tidy, tidy_config_from_dict, tidy_files


# --------------------------------------------------------------------------- #
# Format detection
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text, expected",
    [
        ("x,y\n1.0,2.5\n2.0,3.5\n", (",", ".")),
        ("x;y\n1,0;2,5\n2,0;3,5\n", (";", ",")),          # European: semicolon + comma decimal
        ("x\ty\n1,0\t2,5\n2,0\t3,5\n", ("\t", ",")),       # tab + comma decimal
        ("x y\n1.0 2.5\n2.0   3.5\n", (r"\s+", ".")),      # whitespace, irregular
        ("# note\nx|y\n1|2\n3|4\n", ("|", ".")),           # comment line, integers
        ("a,b\n1,5\n2,5\n", (",", ".")),                   # commas are separators here
    ],
)
def test_sniff_format(tmp_path, text, expected):
    path = tmp_path / "f.txt"
    path.write_text(text, encoding="utf-8")
    assert sniff_format(path) == expected


def test_read_table_decimal_auto_reads_european_file(tmp_path):
    path = tmp_path / "eu.csv"
    path.write_text("T;R\n20,5;100,1\n21,0;100,3\n", encoding="utf-8")
    frame = read_table(path, decimal="auto")
    assert list(frame.columns) == ["T", "R"]
    assert frame["T"].tolist() == [20.5, 21.0]
    assert frame["R"].dtype.kind == "f"


def test_read_table_explicit_decimal_unchanged(tmp_path):
    path = tmp_path / "eu.csv"
    path.write_text("T;R\n20,5;100,1\n", encoding="utf-8")
    frame = read_table(path, decimal=",")
    assert frame["T"].tolist() == [20.5]


# --------------------------------------------------------------------------- #
# Concat  (fixture ``mixed_files`` lives in conftest.py)
# --------------------------------------------------------------------------- #

RENAME = {"Temp (C)": "temperature", "temp_c": "temperature",
          "Resistance (Ohm)": "resistance", "R": "resistance"}


# --------------------------------------------------------------------------- #
# Concat
# --------------------------------------------------------------------------- #

def test_concat_harmonises_and_tags_source(mixed_files, tmp_path):
    out = tmp_path / "out" / "merged.csv"
    result = tidy_files(
        [mixed_files / "a.csv", mixed_files / "b.txt", mixed_files / "c.xlsx"], out,
        rename=RENAME, sheet="all", require=["temperature", "resistance"],
        types={"temperature": "float", "resistance": "float"},
    )
    frame = result.frame
    assert list(frame.columns) == ["source", "temperature", "resistance"]
    assert len(frame) == 7
    assert frame["source"].tolist() == [
        "a.csv", "a.csv", "b.txt", "b.txt", "c.xlsx[run1]", "c.xlsx[run1]", "c.xlsx[run2]"
    ]
    # European decimals were understood, the 'abc' cell became missing and is reported.
    assert frame["temperature"].tolist()[:3] == [20.5, 21.0, 22.0]
    assert np.isnan(frame["temperature"].iloc[3])
    assert result.report.type_failures == {"temperature": [3]}
    # Loose matching: 'TEMP (c)' and 'Temp (C)' both hit the same rename entry.
    units = {u.label: u for u in result.report.units}
    assert units["c.xlsx[run1]"].renamed == {"TEMP (c)": "temperature",
                                             "Resistance (Ohm)": "resistance"}
    assert units["a.csv"].sep == ";" and units["a.csv"].decimal == ","
    assert units["b.txt"].sep == "tab"
    # Files written.
    assert out.is_file()
    written = pd.read_csv(out)
    assert len(written) == 7
    report_json = json.loads((tmp_path / "out" / "merged_tidy_report.json").read_text("utf-8"))
    assert report_json["rows_combined"] == 7
    assert report_json["type_failures"] == {"temperature": [3]}
    assert "Tidy report" in (tmp_path / "out" / "merged_tidy_report.md").read_text("utf-8")


def test_concat_missing_column_is_warned_not_hidden(tmp_path):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("x,z\n3,4\n", encoding="utf-8")
    result = tidy_files([tmp_path / "a.csv", tmp_path / "b.csv"], tmp_path / "m.csv", report=False)
    assert list(result.frame.columns) == ["source", "x", "y", "z"]
    assert any("has no column(s) ['z']" in w for w in result.report.warnings)
    assert any("has no column(s) ['y']" in w for w in result.report.warnings)


def test_require_error_and_skip(tmp_path):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("x,z\n3,4\n", encoding="utf-8")
    base = {"inputs": [str(tmp_path / "*.csv")], "columns": {"require": ["y"]},
            "output": {"path": str(tmp_path / "m.csv")}}
    with pytest.raises(DataValidationError, match="required column"):
        tidy(tidy_config_from_dict(base))
    base["columns"]["on_missing"] = "skip"
    result = tidy(tidy_config_from_dict(base))
    assert len(result.frame) == 1
    skipped = [u for u in result.report.units if u.skipped_reason]
    assert [u.source for u in skipped] == ["b.csv"]
    assert result.report.warnings and "skipped b.csv" in result.report.warnings[0]


def test_rename_collision_is_an_error(tmp_path):
    (tmp_path / "a.csv").write_text("T,temp\n1,2\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="duplicate column names"):
        tidy_files([tmp_path / "a.csv"], tmp_path / "m.csv",
                   rename={"T": "temperature", "temp": "temperature"}, report=False)


def test_source_column_clash_is_an_error(tmp_path):
    (tmp_path / "a.csv").write_text("source,y\n1,2\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="source column name"):
        tidy_files([tmp_path / "a.csv"], tmp_path / "m.csv", report=False)


def test_keep_unknown_column_lists_available(tmp_path):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="Available"):
        tidy_files([tmp_path / "a.csv"], tmp_path / "m.csv", keep=["x", "nope"], report=False)


# --------------------------------------------------------------------------- #
# Join, types, rows
# --------------------------------------------------------------------------- #

@pytest.fixture
def join_setup(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "temps.csv").write_text(
        "id,Temp (C),date\n1,20.5,01/03/2026\n2,21.0,02/03/2026\n3,x,03/03/2026\n2,21.0,02/03/2026\n",
        encoding="utf-8",
    )
    (data / "res.csv").write_text("id;Resistance (Ohm)\n1;100,1\n2;100,3\n4;100,9\n", encoding="utf-8")
    cfg = {
        "inputs": [str(data / "temps.csv"), str(data / "res.csv")],
        "rename": {"temp (c)": "temperature", "Resistance (Ohm)": "resistance"},
        "columns": {
            "require": ["id"],
            "types": {"id": "int", "temperature": "float", "resistance": "float", "date": "datetime"},
            "datetime_format": "%d/%m/%Y",
        },
        "rows": {"drop_duplicates": True, "dropna": ["temperature"], "sort_by": ["id"]},
        "combine": {"how": "join", "keys": ["id"], "join": "outer"},
        "output": {"path": str(tmp_path / "out" / "joined.xlsx"), "language": "es"},
    }
    return cfg, tmp_path


def test_join_types_and_row_filters(join_setup):
    cfg, tmp_path = join_setup
    result = run_tidy(tidy_config_from_dict(cfg))
    frame = result.frame
    assert list(frame.columns) == ["id", "temperature", "date", "resistance"]
    assert frame["id"].tolist() == [1, 2]
    assert str(frame["id"].dtype) == "Int64"
    assert frame["temperature"].tolist() == [20.5, 21.0]
    assert frame["resistance"].tolist() == [100.1, 100.3]
    assert frame["date"].dt.day.tolist() == [1, 2]
    rep = result.report
    assert rep.combine == "join"
    assert rep.rows_combined == 5          # ids 1, 2, 2, 3, 4 after the outer join
    assert rep.rows_out == 2
    assert rep.type_failures == {"temperature": [3]}   # the 'x'
    assert rep.dropped_missing == [3, 4]              # 'x' row and id 4 (no temperature)
    assert rep.dropped_duplicates == [2]
    # xlsx output with a report sheet, Spanish markdown report.
    book = pd.read_excel(result.files["table"], sheet_name=None)
    assert set(book) == {"data", "report"}
    assert len(book["data"]) == 2
    md = result.files["report"].read_text("utf-8")
    assert md.startswith("# Informe de unificación")
    assert "Filas eliminadas por duplicadas" in md


def test_join_overlapping_columns_get_suffix(tmp_path):
    (tmp_path / "a.csv").write_text("id,v\n1,10\n2,20\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("id,v\n1,11\n3,31\n", encoding="utf-8")
    cfg = {"inputs": [str(tmp_path / "a.csv"), str(tmp_path / "b.csv")],
           "combine": {"how": "join", "keys": ["id"], "join": "inner"},
           "output": {"path": str(tmp_path / "m.csv")}}
    result = tidy(tidy_config_from_dict(cfg))
    assert list(result.frame.columns) == ["id", "v", "v_b"]
    assert result.frame["id"].tolist() == [1]
    assert any("suffixed with _b" in w for w in result.report.warnings)


def test_type_conversions_bool_str_int(tmp_path):
    (tmp_path / "a.csv").write_text(
        "flag,label,n\nyes,alpha,3\nno,beta,2.5\nmaybe,,4\n", encoding="utf-8"
    )
    cfg = {"inputs": [str(tmp_path / "a.csv")],
           "columns": {"types": {"flag": "bool", "label": "str", "n": "int"}},
           "combine": {"source_column": None},
           "output": {"path": str(tmp_path / "m.csv")}}
    result = tidy(tidy_config_from_dict(cfg))
    frame = result.frame
    assert frame["flag"].tolist()[:2] == [True, False] and pd.isna(frame["flag"].iloc[2])
    assert frame["n"].tolist()[0] == 3 and pd.isna(frame["n"].iloc[1]) and frame["n"].iloc[2] == 4
    assert result.report.type_failures == {"flag": [2], "n": [1]}   # 'maybe' and 2.5
    assert pd.isna(frame["label"].iloc[2]) and frame["label"].iloc[0] == "alpha"


def test_dropna_any_and_duplicates_subset(tmp_path):
    (tmp_path / "a.csv").write_text("k,v\n1,10\n1,11\n2,\n3,30\n", encoding="utf-8")
    cfg = {"inputs": [str(tmp_path / "a.csv")],
           "rows": {"dropna": "any", "drop_duplicates": True, "duplicates_on": ["k"]},
           "combine": {"source_column": None},
           "output": {"path": str(tmp_path / "m.csv")}}
    result = tidy(tidy_config_from_dict(cfg))
    assert result.frame["k"].tolist() == [1, 3]
    assert result.report.dropped_missing == [2]
    assert result.report.dropped_duplicates == [1]


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #

def test_config_rejects_bad_values(tmp_path):
    good = {"inputs": ["x.csv"], "output": {"path": "m.csv"}}
    with pytest.raises(ConfigError, match="Unknown top-level"):
        tidy_config_from_dict({**good, "extra": 1})
    with pytest.raises(ConfigError, match="on_missing"):
        tidy_config_from_dict({**good, "columns": {"on_missing": "ignore"}})
    with pytest.raises(ConfigError, match="not valid"):
        tidy_config_from_dict({**good, "columns": {"types": {"a": "number"}}})
    with pytest.raises(ConfigError, match="combine.keys"):
        tidy_config_from_dict({**good, "combine": {"how": "join"}})
    with pytest.raises(ConfigError, match="must end in"):
        tidy_config_from_dict({"inputs": ["x.csv"], "output": {"path": "m.parquet"}})
    with pytest.raises(ConfigError, match="matches no file"):
        tidy(tidy_config_from_dict(good, base_dir=tmp_path))
