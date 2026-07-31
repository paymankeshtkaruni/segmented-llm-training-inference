### Cost comparison — large model (838M), batch 4, seq 512, n_steps 3

Memory is the *peak*; GPU host framework = python+torch(+transformers) or python+onnxruntime (C, no torch).

| Run | Runtime | GPU VRAM peak (MB) | GPU host framework (MB) | CPU host peak (MB) | GPU step | CPU step |
|-----|---------|-------------------:|------------------------:|-------------------:|---------:|---------:|
| A2 training | torch | 25424 | 473 | 33040 | 739 ms | 113.5 s |
| B2 inference | torch | 5266 | 475 | 5245 | 227 ms | 36.2 s |
| C inference | ONNX (torch-free) | 4566 | 225 | 3918 | 83 ms | 8.9 s |
