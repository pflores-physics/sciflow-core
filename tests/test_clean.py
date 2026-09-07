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
