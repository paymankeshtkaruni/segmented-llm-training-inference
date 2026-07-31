#!/usr/bin/env python
"""B2 inference cost — memory profiler (CPU), LARGE model, forward-only. Thin wrapper."""
from __future__ import annotations
import argparse
import _common as C
import _cost_lib as Q

DEFAULT_OUT = C.OUTPUTS_DIR / "infer_cost_profiler_cpu"


def main() -> None:
    p = argparse.ArgumentParser(description="B2 inference cost profiler (CPU)")
    Q.add_cost_args(p, default_device="cpu", default_out=DEFAULT_OUT, profile_memory=True)
    Q.run_cost(p.parse_args(), run_name="B2_inference_cost", mode="infer")


if __name__ == "__main__":
    main()
