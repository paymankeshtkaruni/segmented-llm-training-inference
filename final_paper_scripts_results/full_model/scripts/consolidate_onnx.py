"""Consolidate the ONNX external-data (100+ scattered tensor files) into ONE
weights file. Cosmetic cleanup — the model is unchanged. Uses onnx only (no torch).

onnx/large_model.onnx (+ scattered tensors)  ->  onnx/large_model.onnx + large_model.weights
"""
import shutil
from pathlib import Path

import onnx

FULL_MODEL = Path(__file__).resolve().parent.parent
ONNX_DIR = FULL_MODEL / "onnx"
SRC = ONNX_DIR / "large_model.onnx"
TMP = FULL_MODEL / "onnx_tmp"

print(f"[consolidate] loading {SRC} (+ external data) ...")
m = onnx.load(str(SRC), load_external_data=True)

if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True)
print("[consolidate] saving with all tensors in one file ...")
onnx.save(m, str(TMP / "large_model.onnx"), save_as_external_data=True,
          all_tensors_to_one_file=True, location="large_model.weights", size_threshold=1024)

# verify the consolidated file re-loads
onnx.load(str(TMP / "large_model.onnx"), load_external_data=True)
print("[consolidate] verified re-load OK")

# swap: replace onnx/ with the consolidated version
shutil.rmtree(ONNX_DIR)
TMP.rename(ONNX_DIR)
files = sorted(p.name for p in ONNX_DIR.iterdir())
print(f"[consolidate] done. onnx/ now contains: {files}")
