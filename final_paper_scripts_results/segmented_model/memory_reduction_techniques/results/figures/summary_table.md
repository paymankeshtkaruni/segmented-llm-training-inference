| category | metric | baseline | all-ON | reduction | time baseline→all-ON |
|---|---|---|---|---|---|
| gpu_train | VRAM | 18244 MB | 956 MB | **19.1×** | 6.08 → 164.53 s (27×) |
| cpu_train | RSS | 20611 MB | 1427 MB | **14.4×** | 97.02 → 352.17 s (4×) |
| gpu_inference | VRAM | 4094 MB | 634 MB | **6.5×** | 0.22 → 21.50 s (97×) |
| cpu_inference | RSS | 4003 MB | 942 MB | **4.2×** | 0.34 → 23.52 s (69×) |
