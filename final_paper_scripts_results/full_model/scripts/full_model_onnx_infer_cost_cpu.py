#!/usr/bin/env python
"""C — torch-free ONNX inference-cost LIGHT timing (CPU), LARGE model.
Imports only _onnx_cost_lib (onnxruntime/numpy/psutil) — NO torch."""
from __future__ import annotations
import _onnx_cost_lib as O

if __name__ == "__main__":
    O.run("cpu", profile_memory=False,
          out_dir=O.OUTPUTS / "onnx_infer_cost_cpu",
          prefix="onnx_infer_cost")
