#!/usr/bin/env python
"""A2 cost training — memory profiler (GPU). Thin wrapper over _cost_lib."""
from __future__ import annotations
import argparse
import _common as C
import _cost_lib as Q

DEFAULT_OUT = C.OUTPUTS_DIR / "cost_profiler_gpu"


def main() -> None:
    p = argparse.ArgumentParser(description="A2 cost training profiler (GPU)")
    Q.add_cost_args(p, default_device="cuda", default_out=DEFAULT_OUT, profile_memory=True)
    Q.run_cost(p.parse_args(), run_name="A2_cost_training")


if __name__ == "__main__":
    main()
