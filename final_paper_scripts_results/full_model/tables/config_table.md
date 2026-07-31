### TBL-1 — Model configuration (small vs large)

| Parameter | Small (accuracy) | Large (cost) |
|---|---|---|
| tokenizer / vocab | BPE-2k / 2,000 | GPT-2 / 50,257 |
| max_seq_len | 128 | 512 |
| n_layers | 4 | 36 |
| d_model | 64 | 1280 |
| n_heads | 4 | 20 |
| d_ff | 256 | 5120 |
| dropout | 0.1 | 0.1 |
| params | 0.46M | 838M |
| used for | A1 accuracy, B1 inference | A2/B2/C cost |
