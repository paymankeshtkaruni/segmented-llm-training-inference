#!/usr/bin/env python
"""Report stats for the trained BPE tokenizer vs GPT-2 on the log data.

Checks: vocab size, sequence-length distribution (full prompt+target+eos),
label-only lengths, and a round-trip sanity check (decode(encode(x)) == x).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "log_lines" / "generative_splits"
BPE_DIR = Path(__file__).resolve().parent
GPT2_DIR = REPO_ROOT / "gpt2_tokenizer"

PROMPT_TEMPLATE = "{data}\nLabel:"
TARGET_PREFIX = " "


def full_len(tok, log_line, label):
    p = tok.encode(PROMPT_TEMPLATE.format(data=log_line), add_special_tokens=False)
    t = tok.encode(TARGET_PREFIX + label, add_special_tokens=False)
    return len(p) + len(t) + 1  # +eos


def stats_for(tok, name):
    L, LL = [], []
    rt_ok = rt_n = 0
    for split in ["train", "validation", "test"]:
        with open(DATA_DIR / f"{split}.csv", newline="") as f:
            for r in csv.DictReader(f):
                ll, lab = r["log_line"].strip(), r["label"].strip()
                if not ll or not lab:
                    continue
                L.append(full_len(tok, ll, lab))
                LL.append(len(tok.encode(TARGET_PREFIX + lab, add_special_tokens=False)))
                if rt_n < 2000:  # round-trip check on a sample
                    s = PROMPT_TEMPLATE.format(data=ll) + TARGET_PREFIX + lab
                    ids = tok.encode(s, add_special_tokens=False)
                    rt_ok += int(tok.decode(ids) == s)
                    rt_n += 1
    L, LL = np.array(L), np.array(LL)
    p = lambda a, q: int(np.percentile(a, q))
    print(f"\n=== {name} (vocab={tok.vocab_size}) ===")
    print(f"full seq tokens : mean={L.mean():.0f} p50={p(L,50)} p90={p(L,90)} "
          f"p95={p(L,95)} p99={p(L,99)} max={L.max()}  >128:{(L>128).mean()*100:.2f}%")
    print(f"label tokens    : mean={LL.mean():.1f} p50={p(LL,50)} p95={p(LL,95)} max={LL.max()}")
    print(f"round-trip decode==encode: {rt_ok}/{rt_n}")


def main():
    bpe = AutoTokenizer.from_pretrained(str(BPE_DIR))
    print(f"BPE: pad={bpe.pad_token_id} eos={bpe.eos_token_id} vocab={bpe.vocab_size} len={len(bpe)}")
    stats_for(bpe, "BPE-2k")
    try:
        gpt2 = AutoTokenizer.from_pretrained(str(GPT2_DIR))
        if gpt2.pad_token is None:
            gpt2.pad_token = gpt2.eos_token
        stats_for(gpt2, "GPT-2 (50257)")
    except Exception as e:
        print("GPT-2 compare skipped:", e)


if __name__ == "__main__":
    main()
