# bpe_tokenizer — compact byte-level BPE for the log-line task

A small **byte-level BPE** tokenizer trained on the log corpus, replacing the
50,257-token GPT-2 tokenizer. The data uses only ~2,467 distinct GPT-2 tokens
(4.9%), so a compact domain vocab shrinks the model embedding ~25× **and**
tokenizes the logs more compactly — with no UNK (full 256-byte alphabet).

## Files
- `train_bpe.py` — trains the tokenizer on `log_lines/generative_splits/train.csv`
  (text = `"{log_line}\nLabel: {label}"`) and saves `tokenizer.json` +
  `tokenizer_config.json` here via `PreTrainedTokenizerFast`.
- `tokenizer_stats.py` — reports vocab size, sequence-length distribution, and a
  decode(encode(x))==x round-trip check, vs GPT-2.
- `tokenizer.json`, `tokenizer_config.json` — the trained tokenizer
  (load with `AutoTokenizer.from_pretrained("bpe_tokenizer")`).

## Properties (vocab_size = 2000)
- `pad_token = <|pad|>` (id 0), `eos_token = <|endoftext|>` (id 1),
  `bos_token = <|endoftext|>`. **pad ≠ eos** (cleaner than GPT-2; no NaN risk).
- full-seq tokens: mean 57, p95 84, p99 149, max 285 (`>128`: ~2.4%).
- round-trip: 2000/2000 exact.

## Retrain
```
python bpe_tokenizer/train_bpe.py --vocab-size 2000
python bpe_tokenizer/tokenizer_stats.py
```

## Used by
`final_paper_scripts_results/full_model/scripts/_common.py::build_tokenizer()`
points here. **The segmented side, ONNX export, and inference must use this same
tokenizer** for results to be comparable.
