#!/usr/bin/env python
"""C — torch-free ONNX inference-cost PROFILER (GPU), LARGE model.
Imports only _onnx_cost_lib (onnxruntime/numpy/psutil) — NO torch."""
from __future__ import annotations
import _onnx_cost_lib as O

if __name__ == "__main__":
    O.run("cuda", profile_memory=True,
          out_dir=O.OUTPUTS / "onnx_infer_cost_profiler_gpu",
          prefix="onnx_infer_cost_profiler")
