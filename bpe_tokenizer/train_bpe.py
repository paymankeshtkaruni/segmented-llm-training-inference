#!/usr/bin/env python
"""Train a small byte-level BPE tokenizer on the log corpus.

Why: the data uses only ~2,467 distinct GPT-2 tokens (4.9% of 50,257), so a
compact domain BPE shrinks the model embedding ~25x with no sequence-length
penalty. Byte-level (full 256-byte alphabet) => no unknown tokens ever.

The training corpus is the exact text the model sees per example:
    "{log_line}\nLabel: {label}"

Outputs (this folder): tokenizer.json + the PreTrainedTokenizerFast config files,
so `AutoTokenizer.from_pretrained("bpe_tokenizer")` works.

Usage:
    python bpe_tokenizer/train_bpe.py --vocab-size 2000
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from transformers import PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "log_lines" / "generative_splits"
OUT_DIR = Path(__file__).resolve().parent

PAD = "<|pad|>"
EOS = "<|endoftext|>"

# Must match the model's training format (full_model/_common.py).
PROMPT_TEMPLATE = "{data}\nLabel:"
TARGET_PREFIX = " "


def corpus_iter(csv_path: Path):
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            log_line = (row.get("log_line") or "").strip()
            label = (row.get("label") or "").strip()
            if not log_line or not label:
                continue
            yield PROMPT_TEMPLATE.format(data=log_line) + TARGET_PREFIX + label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=2000)
    ap.add_argument("--train-csv", type=Path, default=DATA_DIR / "train.csv")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    tok.post_processor = ByteLevelProcessor(trim_offsets=True)

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[PAD, EOS],          # ids 0,1 reserved
        initial_alphabet=ByteLevel.alphabet(),  # all 256 bytes -> never UNK
        show_progress=True,
    )

    print(f"[train] corpus = {args.train_csv}  vocab_size = {args.vocab_size}")
    tok.train_from_iterator(corpus_iter(args.train_csv), trainer=trainer)
    print(f"[train] done. vocab size = {tok.get_vocab_size()}")

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token=PAD,
        eos_token=EOS,
        bos_token=EOS,        # GPT-style: no dedicated BOS; collator uses add_bos=False anyway
        unk_token=None,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(args.out_dir))
    print(f"[save] -> {args.out_dir}")
    print(f"       pad_id={fast.pad_token_id}  eos_id={fast.eos_token_id}  "
          f"vocab_size={fast.vocab_size}  len={len(fast)}")


if __name__ == "__main__":
    main()
