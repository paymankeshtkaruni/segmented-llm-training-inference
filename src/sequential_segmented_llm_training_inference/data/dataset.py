from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset


@dataclass
class LogLabelExample:
    data: str
    label: str


class LogLabelDataset(Dataset):
    """
    Generative text dataset.

    It returns raw text only:
        {"data": <input text>, "label": <target text>}

    It does not create class ids, issue ids, or level ids. The label is the
    textual target that the model must generate.

    Memory model
    ------------
    The dataset is **lazy**: ``__init__`` scans the CSV once and stores only the
    **byte offset** of each valid record (~8 bytes/row), not the text. Each
    ``__getitem__`` seeks to that offset and reads a single record on demand. So
    the whole corpus is never resident in RAM — only the offset index plus the
    current batch. Multi-line quoted fields are handled via quote-balance record
    detection (a CSV record is complete once its accumulated text has an even
    number of ``"`` characters).
    """

    def __init__(
        self,
        csv_path: str | Path,
        data_column: str = "data",
        label_column: str = "label",
        drop_empty: bool = True,
        max_rows: int | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_column = data_column
        self.label_column = label_column
        self.drop_empty = drop_empty
        # When set, stop indexing after this many valid examples.
        self.max_rows = max_rows

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self._offsets: List[int] = []
        self._data_idx: int = 0
        self._label_idx: int = 0
        self._build_index(self.csv_path)

        if not self._offsets:
            raise ValueError(f"No valid examples loaded from: {self.csv_path}")

    @staticmethod
    def _read_record(f: io.BufferedReader) -> Optional[bytes]:
        """Read one CSV record (possibly spanning several physical lines).

        Returns the raw record bytes, or None at EOF. A record is complete when
        the accumulated bytes contain an even number of double-quote characters
        (all quoted fields closed; escaped ``""`` keep the count even).
        """
        buf = b""
        while True:
            line = f.readline()
            if not line:
                return buf if buf else None
            buf += line
            if buf.count(b'"') % 2 == 0:
                return buf

    def _parse_record(self, raw: bytes) -> Optional[List[str]]:
        text = raw.decode("utf-8")
        try:
            return next(csv.reader(io.StringIO(text)))
        except StopIteration:
            return None

    def _build_index(self, csv_path: Path) -> None:
        with open(csv_path, "rb") as f:
            header_raw = self._read_record(f)
            if header_raw is None:
                raise ValueError(f"CSV has no header: {csv_path}")
            fieldnames = self._parse_record(header_raw) or []
            if self.data_column not in fieldnames or self.label_column not in fieldnames:
                raise ValueError(
                    f"CSV must contain columns '{self.data_column}' and '{self.label_column}'. "
                    f"Found: {fieldnames}"
                )
            self._data_idx = fieldnames.index(self.data_column)
            self._label_idx = fieldnames.index(self.label_column)

            record_start = f.tell()
            while True:
                raw = self._read_record(f)
                if raw is None:
                    break
                next_start = f.tell()
                values = self._parse_record(raw)
                if values is not None and len(values) > max(self._data_idx, self._label_idx):
                    data_text = str(values[self._data_idx] or "").strip()
                    label_text = str(values[self._label_idx] or "").strip()
                    if not (self.drop_empty and (not data_text or not label_text)):
                        self._offsets.append(record_start)
                        if self.max_rows is not None and len(self._offsets) >= self.max_rows:
                            break
                record_start = next_start

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        offset = self._offsets[idx]
        with open(self.csv_path, "rb") as f:
            f.seek(offset)
            raw = self._read_record(f)
        values = self._parse_record(raw) if raw is not None else None
        if values is None:
            return {"data": "", "label": ""}
        data_text = str(values[self._data_idx] if self._data_idx < len(values) else "").strip()
        label_text = str(values[self._label_idx] if self._label_idx < len(values) else "").strip()
        return {"data": data_text, "label": label_text}


# Backward-compatible aliases. They keep the same generative behavior.
LogGenerationDataset = LogLabelDataset
LogCsvDataset = LogLabelDataset
