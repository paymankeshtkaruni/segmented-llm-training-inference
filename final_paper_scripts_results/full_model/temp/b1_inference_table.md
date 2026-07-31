### B1 — Real inference (small model, generation over full test set)

Honest generation accuracy (whole-label exact match) vs the teacher-forced token accuracy from A1.

| Device | Exact-match | Issue acc | Level acc | Teacher-forced token acc (A1) | Examples/s | Time (s) | Prompts truncated |
|--------|------------:|----------:|----------:|------------------------------:|-----------:|---------:|------------------:|
| GPU | 0.9344 | 0.9344 | 0.9840 | 0.9873 | 2205 | 5.3 | 303/11654 |
| CPU | 0.9337 | 0.9337 | 0.9933 | 0.9874 | 46 | 253.3 | 303/11654 |
