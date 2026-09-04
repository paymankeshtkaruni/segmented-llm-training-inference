#!/usr/bin/env python
"""A2 part 1 — inference exactness across the four torch inference modes, at 0.84B.

Modes (dials: streaming incl. output head, KV cache; base always on):
  I1_resident_cache      stream OFF, cache ON
  I2_resident_nocache    stream OFF, cache OFF
  I3_streaming_cache     stream ON,  cache ON
  I4_streaming_nocache   stream ON,  cache OFF

Per mode, against the reference full model with identical weights:
  - prefill hidden state max|D| (eval forward on one prompt)
  - greedy decode of N prompts x K tokens, compared token-by-token

Tech-code flag order: [sdpa, mlp_sum, ce, recompute, stream, records, park, adam,
segment_wo, free_device, no_kv_cache].
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))

from techniques import Tech                                              # noqa: E402
from config import get_preset                                            # noqa: E402
from modules import ReferenceGPTDecoder                                  # noqa: E402
from forward_engine import populate_from_reference, SegmentedForwardEngine  # noqa: E402
from loader import StrictSegmentLoader                                   # noqa: E402
from stores import make_store                                            # noqa: E402
from inference import SegmentedGenerator                                 # noqa: E402

MODES = {
    "I1_resident_cache":    "11100111000",
    "I2_resident_nocache":  "11100111001",
    "I3_streaming_cache":   "11101111100",
    "I4_streaming_nocache": "11101111101",
}


def tech_from_code(code: str) -> Tech:
    names = [f.name for f in fields(Tech)]
    return Tech(**{n: c == "1" for n, c in zip(names, code)})


class _Tok:
    eos_token_id = None   # random-id prompts never emit a meaningful EOS; decode fixed K


@torch.no_grad()
def reference_greedy(ref, prompt_ids, k, max_seq_len, device):
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = []
    for _ in range(k):
        logits, _ = ref(ids[:, -max_seq_len:], pad_token_id=None)
        nxt = int(logits[0, -1].argmax().item())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--preset", default="large_8x2x2x8")
    ap.add_argument("--n-prompts", type=int, default=5)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--new-tokens", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "verify_modes")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    preset = get_preset(a.preset)
    m, s = preset["model"], preset["seg"]

    torch.manual_seed(0)
    ref = ReferenceGPTDecoder(m).to(a.device).eval()

    g = torch.Generator().manual_seed(7)
    prompts = [torch.randint(0, m.vocab_size, (a.prompt_len,), generator=g).tolist()
               for _ in range(a.n_prompts)]
    ref_tokens = [reference_greedy(ref, p, a.new_tokens, m.max_seq_len, a.device)
                  for p in prompts]

    results = []
    for mode, code in MODES.items():
        tech = tech_from_code(code)
        store = make_store("cpu_ram")
        shared = populate_from_reference(ref.to("cpu"), m, s, store).to(a.device)
        ref.to(a.device)
        loader = StrictSegmentLoader(m, s, store, a.device, tech=tech)
        fwd = SegmentedForwardEngine(m, s, loader, shared, a.device).eval()
        loader.training = False

        x = torch.tensor([prompts[0]], dtype=torch.long, device=a.device)
        with torch.no_grad():
            _, h_ref = ref(x, pad_token_id=None)
            h_seg = fwd.forward_hidden(x, pad_token_id=None)
        d_hidden = (h_ref - h_seg).abs().max().item()
        del h_ref, h_seg

        gen = SegmentedGenerator(fwd, _Tok())
        agree, first_div = True, None
        for i, p in enumerate(prompts):
            seg_out = gen.generate(p, max_new_tokens=a.new_tokens)
            if seg_out != ref_tokens[i]:
                agree = False
                for j, (sa, ra) in enumerate(zip(seg_out, ref_tokens[i])):
                    if sa != ra:
                        first_div = {"prompt": i, "pos": j, "seg": sa, "ref": ra}
                        break
                else:
                    first_div = {"prompt": i, "pos": min(len(seg_out), len(ref_tokens[i])),
                                 "note": "length mismatch"}
                break
        results.append({"mode": mode, "tech_code": code,
                        "prefill_hidden_max_delta": d_hidden,
                        "greedy_tokens_identical": agree,
                        "first_divergence": first_div,
                        "n_prompts": a.n_prompts, "new_tokens": a.new_tokens})
        print(f"{mode:22} prefill max|D|={d_hidden:.3e}  tokens identical: {agree}"
              + (f"  first divergence: {first_div}" if first_div else ""), flush=True)
        del fwd, loader, shared, store, gen
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()

    tag = "gpu" if a.device.startswith("cuda") else "cpu"
    out = a.out_dir / f"infer_exact_{tag}.json"
    json.dump({"run": "verify_infer_modes", "preset": a.preset,
               "note": "greedy tokens vs reference full model; identical weights; "
                       "random-id prompts, fixed decode length",
               "results": results}, open(out, "w"), indent=2)
    print(f"\n-> {out}")
    if not all(r["greedy_tokens_identical"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
