"""
Data — lazy log-line dataset + generative collator (self-contained port).

Identical conventions to the full-model baseline so segmented and full are directly
comparable:
  * lazy byte-offset CSV dataset (corpus never in RAM — only an offset index + the
    current batch). Columns: log_line (input), label (target).
  * generative collator: sequence = prompt_ids + target_ids + [EOS], with
    prompt = "{log_line}\nLabel:", target = " {label}", add_bos=False, loss on label
    tokens only (prompt + padding masked with -100). Right-padding (keeps pos 0 real).
  * tokenizers: BPE-2k for the small model, GPT-2 for the large model.

These are DATA/tokenizer files (not code modules), loaded by path — consistent with
the self-containment rule (no imports from src/ or other project packages).
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from config import ModelConfig, PROMPT_TEMPLATE, TARGET_PREFIX, ADD_BOS, ADD_EOS

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "log_lines" / "generative_splits"
TOKENIZER_DIRS = {"bpe": REPO_ROOT / "bpe_tokenizer", "gpt2": REPO_ROOT / "gpt2_tokenizer"}
TRAIN_CSV, VAL_CSV, TEST_CSV = (DATA_DIR / f"{s}.csv" for s in ("train", "validation", "test"))


def build_tokenizer(which: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIRS[which]))
    if tok.pad_token is None:           # GPT-2 has no pad -> pad=eos (BPE-2k has its own)
        tok.pad_token = tok.eos_token
    return tok


class LazyLogDataset(Dataset):
    """Byte-offset-indexed CSV: __init__ scans once storing offsets (~8B/row);
    __getitem__ seeks and reads one record. Multi-line quoted fields handled by
    quote-balance (a record ends when its accumulated bytes have an even count of ")."""

    def __init__(self, csv_path: Path, max_rows: Optional[int] = None):
        self.csv_path = Path(csv_path)
        self._offsets: List[int] = []
        with open(self.csv_path, "rb") as f:
            header = self._read_record(f)
            fields = next(csv.reader(io.StringIO(header.decode())))
            self.li, self.la = fields.index("log_line"), fields.index("label")
            start = f.tell()
            while True:
                raw = self._read_record(f)
                if raw is None:
                    break
                nxt = f.tell()
                vals = self._parse(raw)
                if vals and len(vals) > max(self.li, self.la) \
                        and vals[self.li].strip() and vals[self.la].strip():
                    self._offsets.append(start)
                    if max_rows and len(self._offsets) >= max_rows:
                        break
                start = nxt
        if not self._offsets:
            raise ValueError(f"no rows in {self.csv_path}")

    @staticmethod
    def _read_record(f) -> Optional[bytes]:
        buf = b""
        while True:
            line = f.readline()
            if not line:
                return buf or None
            buf += line
            if buf.count(b'"') % 2 == 0:
                return buf

    @staticmethod
    def _parse(raw: bytes) -> Optional[List[str]]:
        try:
            return next(csv.reader(io.StringIO(raw.decode("utf-8"))))
        except StopIteration:
            return None

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        with open(self.csv_path, "rb") as f:
            f.seek(self._offsets[idx])
            vals = self._parse(self._read_record(f))
        return {"data": vals[self.li].strip(), "label": vals[self.la].strip()}


class GenerationCollator:
    """Builds prompt_ids + target_ids + [EOS]; masks prompt+padding with -100.
    Right-pads (so position 0 is always a real token -> no fully-masked attention row)."""

    def __init__(self, tokenizer, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length
        if self.tok.pad_token_id is None:
            raise ValueError("tokenizer needs a pad token")

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, Any]:
        ids_list, lab_list = [], []
        for item in batch:
            prompt = PROMPT_TEMPLATE.format(data=item["data"])
            target = f"{TARGET_PREFIX}{item['label']}"
            pid = self.tok.encode(prompt, add_special_tokens=False)
            tid = self.tok.encode(target, add_special_tokens=False)
            if ADD_BOS and self.tok.bos_token_id is not None:
                full = [self.tok.bos_token_id] + pid + tid
                plen = 1 + len(pid)
            else:
                full = pid + tid
                plen = len(pid)
            if ADD_EOS and self.tok.eos_token_id is not None:
                full = full + [self.tok.eos_token_id]
            full = full[: self.max_length]
            lab = full.copy()
            for i in range(min(plen, len(lab))):
                lab[i] = -100
            ids_list.append(full)
            lab_list.append(lab)
        width = max(len(x) for x in ids_list)
        pad = self.tok.pad_token_id
        ii = [x + [pad] * (width - len(x)) for x in ids_list]
        ll = [x + [-100] * (width - len(x)) for x in lab_list]
        return {"input_ids": torch.tensor(ii, dtype=torch.long),
                "labels": torch.tensor(ll, dtype=torch.long)}


if __name__ == "__main__":
    tok = build_tokenizer("bpe")
    ds = LazyLogDataset(TRAIN_CSV, max_rows=8)
    coll = GenerationCollator(tok, max_length=128)
    b = coll([ds[i] for i in range(4)])
    print(f"dataset len(>=)={len(ds)}  batch input_ids={tuple(b['input_ids'].shape)} "
          f"labels={tuple(b['labels'].shape)}  pad_id={tok.pad_token_id} eos={tok.eos_token_id}")
    sup = (b['labels'] != -100).sum(1).tolist()
    print("supervised label tokens per row:", sup, "(prompt+pad masked)")
