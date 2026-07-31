"""
Segmented inference — greedy generation, one segment at a time.

Generation runs the segmented forward (forward_engine, one slice resident) on the
current sequence, then picks the next token by streaming the H output-head vocab
slices and taking a running argmax — so the full `[B,T,vocab]` logits are never built
even at decode time. No KV cache (the forward is recomputed each step), which keeps the
memory bound to one segment; fine for short label generation.

Identity: with the same weights, segmented greedy decoding emits the SAME tokens as the
reference model's greedy decoding (verified in the self-test). Per the plan, ONNX/real
inference for the paper uses the full model; this module is the segmented counterpart
and the memory-bounded decoder.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from forward_engine import SegmentedForwardEngine
from segments import vocab_range
from stores import SegmentKey


class SegmentedGenerator:
    def __init__(self, fwd: SegmentedForwardEngine, tokenizer):
        self.f = fwd.eval()
        self.tok = tokenizer

    @torch.no_grad()
    def _next_token(self, ids: torch.Tensor, pad_token_id=None) -> torch.Tensor:
        """Argmax next-token ids from the last position, streaming output-head slices.
        Batch-generic: ids [B,T] -> next [B,1] LongTensor (one segment resident throughout).
        pad_token_id enables batched generation over LEFT-padded variable-length prompts."""
        m, s, ld = self.f.m, self.f.s, self.f.loader
        if ids.size(1) > m.max_seq_len:                       # keep within context
            ids = ids[:, -m.max_seq_len:]
        hidden = self.f.forward_hidden(ids, pad_token_id=pad_token_id)  # all B examples together
        last = hidden[:, -1:, :]                              # [B,1,d_model]
        B = last.size(0); dev = ids.device                   # accumulators on-device (GPU-safe)
        best_val = torch.full((B, 1), torch.finfo(torch.float32).min, device=dev)
        best_idx = torch.zeros((B, 1), dtype=torch.long, device=dev)
        for h in range(s.output_head_segments):
            v0, v1 = vocab_range(m.vocab_size, s.output_head_segments, h)
            with ld.acquire_segment(SegmentKey(-1, "output_head", h)) as head:
                z = head(last)                               # [B,1,vsl]
            smax, sarg = z.max(dim=-1)                        # [B,1]
            upd = smax > best_val
            best_val = torch.where(upd, smax, best_val)
            best_idx = torch.where(upd, sarg + v0, best_idx)
            del z
        return best_idx                                      # [B,1] LongTensor

    @torch.no_grad()
    def generate(self, prompt_ids: List[int], max_new_tokens: int = 32) -> List[int]:
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.f.device)
        out: List[int] = []
        eos = self.tok.eos_token_id
        for _ in range(max_new_tokens):
            nxt = self._next_token(ids)                       # [1,1] LongTensor (B=1 here)
            tok = int(nxt[0, 0].item())
            if eos is not None and tok == eos:
                break
            out.append(tok)
            ids = torch.cat([ids, nxt], dim=1)
        return out

    @torch.no_grad()
    def generate_batch(self, prompts: List[List[int]], max_new_tokens: int = 32) -> List[List[int]]:
        """Batch greedy decode — all B prompts (same length) advanced together, still one
        segment resident at a time. No per-sequence EOS early-stop (fixed steps)."""
        ids = torch.tensor(prompts, dtype=torch.long, device=self.f.device)   # [B,T]
        outs = [[] for _ in prompts]
        for _ in range(max_new_tokens):
            nxt = self._next_token(ids)                       # [B,1]
            for b in range(ids.size(0)):
                outs[b].append(int(nxt[b, 0].item()))
            ids = torch.cat([ids, nxt], dim=1)
        return outs

    def generate_label(self, log_line: str, max_new_tokens: int = 32) -> str:
        from config import PROMPT_TEMPLATE
        pid = self.tok.encode(PROMPT_TEMPLATE.format(data=log_line), add_special_tokens=False)
        return self.tok.decode(self.generate(pid, max_new_tokens), skip_special_tokens=True).strip()


@torch.no_grad()
def _reference_greedy(ref, prompt_ids, max_new_tokens, eos, max_seq_len) -> List[int]:
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    out = []
    for _ in range(max_new_tokens):
        ctx = ids[:, -max_seq_len:]
        logits, _ = ref(ctx, pad_token_id=None)
        nxt = int(logits[0, -1].argmax().item())
        if nxt == eos:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return out


if __name__ == "__main__":
    import sys
    from config import SMALL_MODEL, SEG_8x2x2x8
    from modules import ReferenceGPTDecoder
    from forward_engine import populate_from_reference
    from loader import StrictSegmentLoader
    from stores import make_store
    torch.manual_seed(3)
    m, s = SMALL_MODEL, SEG_8x2x2x8
    ref = ReferenceGPTDecoder(m).eval()
    store = make_store("cpu_ram")
    shared = populate_from_reference(ref, m, s, store)
    fwd = SegmentedForwardEngine(m, s, StrictSegmentLoader(m, s, store, "cpu"), shared, "cpu").eval()

    class _Tok:  # minimal stand-in (eos id only)
        eos_token_id = 1
    gen = SegmentedGenerator(fwd, _Tok())

    prompt = [5, 17, 42, 8, 100, 3]
    seg_out = gen.generate(prompt, max_new_tokens=20)
    ref_out = _reference_greedy(ref, prompt, 20, eos=1, max_seq_len=m.max_seq_len)
    print(f"segmented greedy: {seg_out}")
    print(f"reference greedy: {ref_out}")
    ok = seg_out == ref_out
    print("INFERENCE IDENTITY OK (segmented greedy == reference greedy)" if ok else "INFERENCE MISMATCH")
    sys.exit(0 if ok else 1)
