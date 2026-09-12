import numpy as np
import pandas as pd
import pytest

from sciflow.clean import clean_data


def test_reports_each_kind_of_bad_row(dirty_frame):
    clean, report = clean_data(dirty_frame, ["X", "Y", "S"])
    assert report.original_rows == 8
    assert report.non_numeric_rows == [2]
    assert report.missing_rows == [5]
    assert report.non_finite_rows == [6]
    assert report.duplicate_rows == []          # off by default
    assert report.final_rows == 5
    assert clean["Y"].dtype == float
    assert list(clean["label"]) == list("abdeh")  # untouched column preserved


def test_duplicates_only_when_requested(dirty_frame):
    clean, report = clean_data(dirty_frame, ["X", "Y", "S"], drop_duplicates=True)
    assert report.duplicate_rows == [4]
    assert report.final_rows == 4


def test_unknown_column_raises(dirty_frame):
    with pytest.raises(KeyError):
        clean_data(dirty_frame, ["X", "nope"])


def test_clean_frame_passes_through():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    clean, report = clean_data(frame)
    assert report.removed_rows == 0
    assert np.array_equal(clean.to_numpy(), frame.to_numpy())


def test_summary_mentions_counts(dirty_frame):
    _, report = clean_data(dirty_frame, ["X", "Y"])
    text = report.summary()
    assert "non-numeric: 1" in text and "missing: 1" in text and "non-finite: 1" in text


def test_non_positive_uncertainties_removed_when_requested():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0],
                          "s": [0.1, 0.0, -0.2, 0.1]})
    clean, report = clean_data(frame, ["x", "y", "s"], positive=["s"])
    assert len(clean) == 2
    assert report.non_positive_rows == [1, 2]
    assert "non-positive uncertainty" in report.summary()
    # Without ``positive`` the rows are kept (the engine will reject them).
    clean, report = clean_data(frame, ["x", "y", "s"])
    assert len(clean) == 4 and report.non_positive_rows == []


def test_reader_sniffs_delimiters(tmp_path):
    from sciflow.io.readers import read_table

    rows = [(0.0, 1.0, 0.1), (1.5, 2.25, 0.1), (3.0, 4.5, 0.1)]
    cases = {
        "comma.csv": ("x,y,s\n" + "\n".join(f"{a},{b},{c}" for a, b, c in rows), {}),
        "semicolon_decimal_comma.csv": ("x;y;s\n" + "\n".join(f"{a};{b};{c}".replace(".", ",") for a, b, c in rows),
                                        {"decimal": ","}),
        "tabs_with_comments.txt": ("# header\n\nx\ty\ts\n" + "\n".join(f"{a}\t{b}\t{c}" for a, b, c in rows) + "\n\n", {}),
        "spaces.dat": ("x   y  s\n" + "\n".join(f"{a}    {b}  {c}" for a, b, c in rows), {}),
    }
    for name, (text, kwargs) in cases.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        data = read_table(path, **kwargs)
        assert list(data.columns) == ["x", "y", "s"], name
        assert np.allclose(data.to_numpy(float), np.array(rows)), name
