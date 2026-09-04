#!/usr/bin/env python
"""Run ONE cost measurement (one technique configuration) in a fresh process.

Process isolation guarantees every configuration starts with a pristine CUDA
context and an empty PyTorch caching-allocator pool, so no reserved-memory
residue from a previously measured configuration can leak into its no-miss
peak. Observed leak without isolation: the same tech code measured 1200 MB vs
4212 MB depending on which rung ran before it in the shared process.

Used by ladder_runner.py and loo_runner.py via run_isolated(); can also be run
directly for a single configuration.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))

from techniques import Tech   # noqa: E402  (dataclass only — no torch in the parent)


def tech_from_code(code: str) -> Tech:
    names = [f.name for f in fields(Tech)]
    if len(code) != len(names):
        raise ValueError(f"tech code length {len(code)} != {len(names)} flags")
    return Tech(**{n: c == "1" for n, c in zip(names, code)})


def run_isolated(kind: str, preset: str, device: str, out_dir, prefix: str, tech: Tech, **kw):
    """Spawn a fresh interpreter for one measurement; return its metrics dict."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(HERE / "measure_one.py"), "--kind", kind,
           "--preset", preset, "--device", str(device), "--out-dir", str(out_dir),
           "--prefix", prefix, "--tech-code", tech.code()]
    for k, v in kw.items():
        if v is not None and k != "from_scratch":      # from_scratch is always True here
            cmd += [f"--{k.replace('_', '-')}", str(v)]
    subprocess.run(cmd, check=True)
    return json.load(open(out_dir / f"{prefix}_met.json"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["train", "infer"], required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--tech-code", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--gen-tokens", type=int, default=8)
    ap.add_argument("--seg-override", default=None,
                    help="ExAxMxH partition override, e.g. 2x1x1x2 (default: preset's)")
    ap.add_argument("--update-style", default="after_full",
                    choices=["after_full", "immediate"],
                    help="immediate = segments updated during backward (no global clip)")
    ap.add_argument("--dropout", type=float, default=None,
                    help="override model dropout (e.g. 0.0); default = preset's (0.1)")
    a = ap.parse_args()

    seg = None
    if a.seg_override is not None:
        from config import SegmentationConfig
        e, at, mc, h = (int(x) for x in a.seg_override.split("x"))
        seg = SegmentationConfig(embedding_segments=e, attention_segments=at,
                                 mlp_chunks=mc, output_head_segments=h)
    tech = tech_from_code(a.tech_code)
    if a.kind == "train":
        from seg_cost_lib import run_cost
        met = run_cost(a.preset, a.device, a.out_dir, prefix=a.prefix, batch=a.batch,
                       n_steps=a.n_steps, seq_len=a.seq_len, from_scratch=True, tech=tech,
                       seg_override=seg, update_style=a.update_style, dropout=a.dropout)
    else:
        from infer_cost_lib import run_infer_cost
        met = run_infer_cost(a.preset, a.device, a.out_dir, prefix=a.prefix,
                             prompt_len=a.prompt_len, gen_tokens=a.gen_tokens,
                             tech=tech, from_scratch=True, seg_override=seg)
    out = a.out_dir / f"{a.prefix}_met.json"
    json.dump(met, open(out, "w"), indent=2)
    print(f"[measure_one] -> {out}")
    # the _work dir holds this run's param/grad/opt stores — regenerable scratch
    # that at 0.84B is ~GBs per run and blew the project quota when 72 runs kept
    # theirs. The results are the JSONs; always clean the scratch.
    import shutil
    work = a.out_dir / "_work"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
        print(f"[measure_one] cleaned scratch {work}")


if __name__ == "__main__":
    main()
