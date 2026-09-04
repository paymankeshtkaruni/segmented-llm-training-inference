"""
Single source of truth for the incremental memory-reduction-technique ablation.

The SUBJECT is FIXED: the segmented LARGE cost model (`large_8x2x2x8`, 8x2x2x8 split).
We do NOT vary the segmentation axes. Instead we start from that segmented model with
every memory-reduction TECHNIQUE turned OFF (naive execution) and switch the techniques
ON one at a time, cumulatively, recording peak memory + step time at each rung. The top
rung (all ON) is the real cost model (~934 MB GPU).

`Tech` is the set of toggle flags. Default = all ON (the cost model). `baseline()` = all
OFF. The LADDERs below fix the LOGICAL order in which techniques are switched on, and
`cumulative(ladder)` expands that order into the per-rung flag sets the drivers run.

The engine copies in this folder read these flags to enable/disable each technique. The
shared modules are never touched (see memory: self-contained-experiment-folders).
"""
from __future__ import annotations

from dataclasses import dataclass, replace, fields


@dataclass(frozen=True)
class Tech:
    # ---- forward activation-memory techniques ----
    sdpa: bool = True            # memory-efficient attention (no full [B,H,T,T] scores)
    mlp_running_sum: bool = True # sum MLP chunk outputs (no full [B,T,d_ff] hidden)
    chunked_ce: bool = True      # streamed cross-entropy (no full [B,T,vocab] logits)
    # ---- backward technique ----
    recompute: bool = True       # backward by recomputation (no full autograd graph)
    # ---- streaming / offload techniques ----
    stream_segments: bool = True # one segment resident; the rest parked in the store
    offload_records: bool = True # layer-input snapshots streamed off-device (depth-indep.)
    park_grads_host: bool = True  # shared/W_o grads parked on host as computed
    offload_adam: bool = True    # shared Adam m,v parked on host, streamed in opt.step
    segment_wo: bool = True      # W_o (attn_out_proj) streamed as a segment, not resident
    # ---- allocator hygiene ----
    free_device: bool = True     # gc + empty_cache / malloc_trim on every segment release
    # ---- inference-only technique ----
    no_kv_cache: bool = True     # recompute each decode step instead of caching K/V

    def code(self) -> str:
        return "".join("1" if getattr(self, f.name) else "0" for f in fields(self))


def baseline() -> Tech:
    """The segmented large model with EVERY memory-reduction technique OFF (rung 0)."""
    return Tech(**{f.name: False for f in fields(Tech)})


# ---- logical order: each entry = (flag, rung name, one-line what/why) ----
# TRAIN: cheap-first — all time-free techniques, then streaming (moderate cost),
# then allocator release (expensive). Cost classes measured on A100 (ladder_train.json):
# free <2 s added; streaming x4.3; free_device x5.8 on top.
TRAIN_LADDER = [
    ("sdpa",            "r1_sdpa",           "SDPA attention (no full score matrix)"),
    ("mlp_running_sum", "r2_mlp_sum",        "MLP running-sum (no full d_ff hidden)"),
    ("chunked_ce",      "r3_chunked_ce",     "streamed chunked cross-entropy (no full logits)"),
    ("recompute",       "r4_recompute",      "backward by recomputation (no full autograd graph)"),
    ("offload_records", "r5_records",        "offload layer-input records (depth-independent)"),
    ("park_grads_host", "r6_park_grads",     "park shared/W_o grads on host"),
    ("offload_adam",    "r7_offload_adam",   "offload shared Adam state to host"),
    ("segment_wo",      "r8_segment_wo",     "stream W_o as a segment (resident floor 243->17 MB)"),
    ("stream_segments", "r9_stream",         "stream segments off-device (one resident)"),
    ("free_device",     "r10_free_device",   "free_device / allocator hygiene on release"),
]

# INFERENCE: no backward / optimizer -> drop recompute, records, grads, adam.
INFER_LADDER = [
    ("sdpa",            "r1_sdpa",           "SDPA attention (no full score matrix)"),
    ("mlp_running_sum", "r2_mlp_sum",        "MLP running-sum (no full d_ff hidden)"),
    ("chunked_ce",      "r3_chunked_ce",     "streamed chunked CE at eval (no full logits)"),
    ("stream_segments", "r4_stream",         "stream segments off-device (one resident)"),
    ("segment_wo",      "r5_segment_wo",     "stream W_o as a segment (resident floor down)"),
    ("free_device",     "r6_free_device",    "free_device / allocator hygiene on release"),
    ("no_kv_cache",     "r7_no_kv_cache",    "no KV cache (recompute each decode step)"),
]


def cumulative(ladder):
    """Expand a ladder into per-rung (name, desc, Tech) with flags accumulated from OFF.

    Rung 0 is the all-OFF baseline; each subsequent rung turns ON one more flag and keeps
    all previously-turned-on flags ON. Flags NOT named in `ladder` stay OFF throughout
    (e.g. no_kv_cache never appears in TRAIN_LADDER, recompute never in INFER_LADDER)."""
    rungs = [("r0_baseline", "naive segmented — all techniques OFF", baseline())]
    on = {f.name: False for f in fields(Tech)}
    for flag, name, desc in ladder:
        on[flag] = True
        rungs.append((name, desc, Tech(**on)))
    return rungs


if __name__ == "__main__":
    for tag, ladder in [("TRAIN", TRAIN_LADDER), ("INFER", INFER_LADDER)]:
        print(f"\n=== {tag} ladder ===")
        for name, desc, t in cumulative(ladder):
            print(f"  {name:16} {t.code()}  {desc}")
