"""Lazy offset-index LogLabelDataset: correctness vs eager parse, multi-line
quoted fields, max_rows, drop_empty, and that it holds only offsets (not text)."""
from __future__ import annotations

import csv

from sequential_segmented_llm_training_inference.data.dataset import LogLabelDataset


def _write_csv(path, rows, header=("log_line", "label")):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(r)


def test_matches_eager_parse(tmp_path):
    rows = [(f"log line number {i}, with comma", f"Issue: {i % 3}") for i in range(50)]
    p = tmp_path / "d.csv"
    _write_csv(p, rows)

    ds = LogLabelDataset(p, data_column="log_line", label_column="label")
    assert len(ds) == 50
    for i, (d, l) in enumerate(rows):
        assert ds[i] == {"data": d, "label": l}


def test_multiline_quoted_field(tmp_path):
    p = tmp_path / "ml.csv"
    _write_csv(p, [
        ("first line\nsecond line inside quotes", "Issue: A"),
        ("plain", "Issue: B"),
    ])
    ds = LogLabelDataset(p, data_column="log_line", label_column="label")
    assert len(ds) == 2
    assert "\n" in ds[0]["data"]
    assert ds[1] == {"data": "plain", "label": "Issue: B"}


def test_max_rows(tmp_path):
    p = tmp_path / "m.csv"
    _write_csv(p, [(f"l{i}", f"y{i}") for i in range(20)])
    ds = LogLabelDataset(p, data_column="log_line", label_column="label", max_rows=5)
    assert len(ds) == 5
    assert ds[4] == {"data": "l4", "label": "y4"}


def test_drop_empty(tmp_path):
    p = tmp_path / "e.csv"
    _write_csv(p, [("good", "Issue: A"), ("", "Issue: B"), ("also good", "")])
    ds = LogLabelDataset(p, data_column="log_line", label_column="label", drop_empty=True)
    assert len(ds) == 1
    assert ds[0]["data"] == "good"


def test_holds_only_offsets_not_text(tmp_path):
    # Long text rows: a lazy index must not grow with text length.
    big = "x" * 5000
    p = tmp_path / "big.csv"
    _write_csv(p, [(f"{big} {i}", f"Issue: {i}") for i in range(200)])
    ds = LogLabelDataset(p, data_column="log_line", label_column="label")
    # Internal store is the offset list only — no materialised example text.
    assert not hasattr(ds, "examples")
    assert len(ds._offsets) == 200
    # Each entry is a small int (byte offset), not a ~5 KB string.
    assert all(isinstance(o, int) for o in ds._offsets)
    assert ds[199]["label"] == "Issue: 199"
