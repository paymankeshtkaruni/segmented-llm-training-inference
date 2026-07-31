# segmentation_management — Design & Rationale

A clean, **self-contained** re-implementation of the segmented sequential
training/inference stack. Written for the paper: every module, option, and
consideration is justified here. Built by reading the existing `src/`
implementation, keeping what is correct, fixing the issues found, and re-housing
everything under this one package — **no imports from `src/` or elsewhere** (only
standard libraries: torch, numpy, transformers tokenizer, psutil).

This document grows one section per module as the package is built. Each module is
verified before the next is started.

---

## 1. Goal & central claim

**Claim:** segmenting a GPT decoder and training it **sequentially, one segment in
memory at a time**, under a memory limit, yields the **same trained model** as
standard full-model training — while drastically lowering peak memory.

So segmentation is a pure **execution** strategy layered on an **unchanged**
architecture. The model the segments reassemble into must be numerically identical
to a normally-trained model (verified by the export + audit module).

**Why it works (the principle):** every memory-heavy part of the model is sliced so
that only one slice is resident on the *constrained* device at any instant; the rest
lives in a cheap "store". Peak memory ≈ one slice, at the cost of extra compute
(reloading slices and recomputing activations in the backward pass).

## 2. Self-containment (why a fresh package)

The existing `src/` stack is correct in its core mechanics (its export is
audited numerically identical) but is spread across many packages and carries
options that, for this study, collapse to a single choice. We re-house a clean
version here so the paper's artifacts are reproducible from one directory and so
each design decision can be stated explicitly. Correct logic is ported; issues are
fixed (see §"Fixes").

## 3. Configurations (`config.py`)

Two config objects, deliberately separate because architecture is independent of
slicing:

- **`ModelConfig`** — the architecture (vocab, max_seq_len, n_layers, d_model,
  n_heads, d_ff, dropout). `head_dim = d_model/n_heads` must be integer.
- **`SegmentationConfig`** — the four slicing axes (E×A×M×H).

### Two models (why two)
| | small | large |
|---|---|---|
| use | accuracy (A1/B1-style) | cost/memory (A2/B2/C-style) |
| tokenizer / vocab | BPE-2k / 2,000 | GPT-2 / 50,257 |
| d_model · heads · layers · d_ff | 64 · 4 · 4 · 256 | 1280 · 20 · 36 · 5120 |
| params | ~0.46M | ~837M |

The task (log line → `Issue: …, level: …`, 154 labels) is easy, so a **small** model
keeps accuracy non-trivial (headroom). Memory cost, however, is only meaningful when
the **model's own** memory dominates the fixed runtime overhead (CUDA context, etc.),
which needs a **large** model — hence the split.

### The four segmentation axes (why each)
Notation **E×A×M×H** = `embedding × attention × mlp × output_head`.

| Axis | Splits | Reassembled by | Divisibility | Why (which tensor it bounds) |
|------|--------|----------------|--------------|------------------------------|
| **E** embedding | embedding along **d_model** | concat on d_model | E \| d_model | the large embedding table |
| **A** attention | **heads** into groups | concat head outputs → 1 out-proj | A \| n_heads | Q/K/V + per-head score tensors |
| **M** mlp | **d_ff** hidden into chunks | **sum** chunk outputs | M \| d_ff | the `[B,T,d_ff]` hidden activation |
| **H** output head | output proj along **vocab** | streamed/chunked cross-entropy | last slice takes remainder | the `[B,T,vocab]` logits |

A and M divide exactly (clean head/hidden groups). E divides d_model exactly. H need
**not** divide vocab — the last vocab slice absorbs the remainder, which matters for
the large model (50257 is not divisible by 8 or 16).

### The three presets (the only configs used)
- `small_8x2x2x8`   — small model, E8·A2·M2·H8 (4 heads/2 = 2 per group; 256/2 = 128
  hidden/chunk; 64/8 = 8 d_model/slice; 2000/8 = 250 vocab/slice).
- `large_8x2x2x8`   — large model, E8·A2·M2·H8 (1280/8, 20/2, 5120/2, 50257/8+rem).
- `large_16x4x4x16` — large model, E16·A4·M4·H16 (1280/16, 20/4, 5120/4, 50257/16+rem).
  Finer slicing → lower peak memory, more reload/recompute overhead. Having two large
  setups lets the paper show the memory/compute trade-off as slicing gets finer.

### Sequence/objective constants (why)
`add_bos=False`, `pad ≠ eos`, prompt `"{data}\nLabel:"`, loss on label tokens only.
With GPT-2 `bos==eos==pad`; prepending BOS would place the pad id at position 0, the
padding mask masks it, and the all-`-inf` attention row yields NaN. Omitting BOS (and
using a distinct pad in BPE-2k) avoids this. Same convention as the full-model baseline
so the two are comparable.

## 4. Device → offload-target principle (applies throughout)

The "store" that holds the inactive slices lives where the *non-constrained* memory
is:
- **GPU**: the constraint is **VRAM**; host RAM is free, so inactive slices /
  gradients / optimizer state are parked in **host RAM (`cpu_ram` store)** and only
  the active slice is on the GPU → VRAM peak = one slice.
- **CPU**: the constraint **is** host RAM, so it cannot be the parking spot — slices
  are parked on **disk**. On CPU, disk plays the role `cpu_ram` plays on GPU.

This single rule drives the store choice, the optimizer-step streaming, and the
gradient-clipping strategy (see the optimizer module).

## 4b. Memory discipline (hard rule at every step)

No step may allocate a large tensor when it can be avoided. Enforced throughout:
- **Attention** uses `F.scaled_dot_product_attention` (SDPA) — the `[B,n_heads,T,T]`
  score matrix is **never materialized**. With padding, only one `[B,1,T,T]` additive
  mask is built (finite large-negative value → fully-masked rows are uniform, not NaN).
- **MLP chunks** combined by a **running sum**; each `[B,T,d_ff/M]` chunk freed at once.
- **Output head** uses **streamed/chunked cross-entropy** — `[B,T,vocab]` never built;
  only `[B,seq_chunk,vocab/H]` at a time.
- **One active segment** on the constrained device; the rest in the store.
- **Backward** recomputes per segment (no full forward graph retained).
- **Optimizer step** streams one segment; **grad clipping** is two-pass streaming.
- Explicit `del` + `gc.collect()` (+ `empty_cache` CUDA / `malloc_trim` CPU) between segments.
Every module's doc states its peak tensor and why it is bounded.

## 5. Fixes applied while porting (running list)
- Attention via SDPA (no `[B,h,T,T]`); finite mask value (NaN-robust).
- Default the `after_full_backward` optimizer step + gradient clipping to the
  **streamed, device-correct** path (GPU→cpu_ram, CPU→disk) so there is never an
  all-gradients spike on the constrained device.
- Carry the `add_bos=False` / `pad≠eos` convention into the data/collator.
- Keep the **one-active-segment** invariant in the loader and assert it.

---

## Module log
*(each module documented here as it is built and verified)*

### `config.py` — ✅ built & verified
ModelConfig + SegmentationConfig + the 3 presets (above). Pure dataclasses, no torch;
validates divisibility and prints the preset summary on `python config.py`.

### `modules.py` — ✅ built & verified
The reference **un-segmented** pre-LN GPT decoder (`ReferenceGPTDecoder`) + its
building blocks (embedding, `CausalSelfAttention`, `GPTMLP`, `GPTBlock`), the canonical
init (`std=0.02`, residual projections scaled by `1/sqrt(2·n_layers)`), and the shared
`causal_lm_cross_entropy_loss` / `token_counts`.
- **Why it's here even though training never builds the full model:** it is the
  reassembly target for export/audit (segmented==full identity) and the numerical
  reference the segments must reproduce.
- **Layout choices that enable clean slicing:** one `[d_model,3·d_model]` qkv matrix
  (head groups are contiguous slices); MLP in/out projections both index d_ff (hidden
  chunks are contiguous slices, summed on output).
- **NaN consideration:** masking uses `-inf`+softmax (faithful to reference); a
  fully-masked query row would be NaN, avoided by convention (`add_bos=False`,
  right-padding) rather than by altering the math — required so segmented==full holds.
- Verified: small reference = 0.464M params, forward+loss healthy (≈ln 2000), no NaN.

### `segments.py` — ✅ built & verified
The four segment modules (`AttentionHeadSegment`, `MLPHiddenChunk`, `EmbeddingSlice`,
`OutputHeadSlice`) + deterministic slice-range helpers + the `[B,1,T,T]` causal∪padding
bias builder. Attention via SDPA; MLP down-proj bias excluded (added once on the sum).
- **Identity (segmented==full at layer level), measured exact:** concat of attention
  head-groups (+out-proj) and sum of MLP chunks (+shared bias) reproduce the full
  attention/MLP with **max|Δ|=0.00e+00**.
- **Memory:** per-segment peaks — attn `[B,n_heads/A,T,head_dim]` (SDPA, no scores),
  mlp `[B,T,d_ff/M]` (running-sum), emb `[B,T,d_model/E]`, head `[B,T,vocab/H]`.
  See MEMORY_INVESTIGATIONS.md #1–#3.

### `stores.py` — ✅ built & verified
`SegmentStore` with `CpuRamStore` (GPU runs, park in host RAM) and `DiskStore`
(CPU runs, one .pt/key, map_location=cpu, read-modify-write `accumulate`). `make_store`
+ `default_store_kind` enforce the device→target rule. Round-trip/accumulate/evict verified.

### `loader.py` — ✅ built & verified
`StrictSegmentLoader`: at most ONE segment on the device (asserted), `build_segment`
makes the right sliced module, `acquire_segment` loads (fresh-init persisted, or from
store) and releases with `free_device`. Carries train/eval mode onto each segment
(**bug found+fixed**: segments defaulted to train → dropout on → broke identity).
Verified: one-active over 20 segments; reload==store.

### `memory.py` — ✅ built & verified
`free_device` (gc unconditional; empty_cache CUDA-only; malloc_trim Linux/CPU-only),
point measurements (host RSS, nvidia-smi process VRAM, torch VRAM counters), background
`Sampler`. Each with the device rationale documented.

### `forward_engine.py` — ✅ built & verified
`SharedParams` (resident tiny: norms, attn out-proj, mlp bias, final norm) +
`SegmentedForwardEngine.forward_hidden` (embedding concat → per-layer attention concat
& MLP running-sum, one segment at a time) + `chunked_ce` (streamed, no full logits) +
`populate_from_reference` (slice a reference into store+shared, for tests/export).
- **Identity vs reference: hidden max|Δ|=0.00e+00, loss |Δ|=4.8e-7** — segmented
  forward == full forward.
- Memory peaks bounded per MEMORY_INVESTIGATIONS #1–#4; residual stream [B,T,d_model]
  is the one resident activation. eval()/train() propagate dropout mode to segments.

### `backward_engine.py` — ✅ built & verified
Recompute-based full backward: records only per-layer residual `[B,T,d_model]`;
analytic streamed CE grad (softmax−onehot, no full logits); per layer reload one
segment, recompute, `autograd.grad` for its params + input grad, stream param grads
to the grad store, free. Shared (tiny) grads accumulated and returned.
- **Gradient identity vs reference `loss.backward()`: max|Δ|=1.3e-7** (shared),
  ~1e-8 (segments) — segmented training == full training.
- Peak independent of depth (MEMORY_INVESTIGATIONS #5). Cleaned a dead placeholder line.

### `optimizer.py` — ✅ built & verified
`SegmentwiseAdamW` (AdamW only — betas (0.9,0.95), wd 0.1, lr 3e-4, GPT-style decay on
2D+ weights). Streamed `after_full_backward`: per segment load params+grad+(m,v), AdamW
in place, write back, free; two-pass streaming global-norm clip; shared params in-memory.
- **Step identity vs `torch.optim.AdamW`+`clip_grad_norm_`: max|Δ|=1.5e-8.**
- Peak = 4×one-segment, never 4×model; clip never holds all grads (MEMORY_INVESTIGATIONS #6).
- **Full train step proven identical end-to-end:** forward Δ=0, grads Δ=1.3e-7,
  updated weights Δ=1.5e-8 → segmented training == full training.

### `data.py` — ✅ built & verified
Self-contained port: lazy byte-offset `LazyLogDataset` (corpus never in RAM) +
`GenerationCollator` (prompt+target+EOS, add_bos=False, loss on label tokens, right-pad)
+ `build_tokenizer` (bpe/gpt2). IDENTICAL conventions to the full-model baseline
(same splits, prompt `"{data}\nLabel:"`, truncation behavior). Verified on real data.

### `trainer.py` — ✅ built & verified
`SegmentedTrainer`: epoch loop wiring forward-record → recompute backward → streamed
AdamW, aligned 1:1 with `_train_lib` (lr 3e-4, wd 0.1, betas (0.9,0.95), clip 1.0,
warmup 200, cosine, 5 epochs; train/val loss; val acc global+example). Init slices a
seeded `ReferenceGPTDecoder` so starting weights == a full model (makes seg==full
checkable). Store kind by device (cpu_ram GPU / disk CPU). Saves checkpoints/{best,last}.pt
+ metrics.json (full_model-compatible). Smoke: loss decreases from ≈ln 2000, eval+ckpt OK.

### `export.py` — ✅ built & verified  (CAPSTONE)
`reassemble_state_dict` / `reassemble_model`: inverse of the slicing — concat head
q/k/v→qkv, mlp chunks→full mlp, embedding slices→embedding, vocab slices→output proj,
shared→norms/bias/out-proj — into one `ReferenceGPTDecoder`.
- **(1) round-trip** slice→reassemble: max|Δ|=0.00e+00 (perfectly invertible).
- **(2) END-TO-END proof:** train the FULL model and the SEGMENTED model from the same
  seeded init for N steps on the same batches (real backward + clip + AdamW), reassemble,
  compare: **max|Δ|=4.9e-6 after 4 steps** → segmented training == full training,
  cumulatively, not just per step. This is the paper's central claim, verified.

---

## Headline result
Segmented sequential training (one slice resident at a time; bounded attention/MLP/CE,
recompute backward, streamed AdamW, two-pass clip) yields a model **numerically
identical** to standard full training:
  forward Δ=0 · grads Δ=1.3e-7 · one AdamW step Δ=1.5e-8 · 4 full steps Δ=4.9e-6.
All peak tensors measured/bounded (MEMORY_INVESTIGATIONS #1–#6). Self-contained under
`segmentation_management/`.

### `inference.py` — ✅ built & verified
`SegmentedGenerator`: greedy decoding via the segmented forward (one slice resident) +
streamed output-head argmax (no full logits at decode). Verified: segmented greedy ==
reference greedy (identical token sequence). Memory bounded to one segment; no KV cache
(recompute each step) — fine for short labels.

---

## Verification methodology — why these tests

The claim is **segmented == full**, so the only convincing test is *identity against
the reference full model at every level*. We do not test segments in isolation against
hand-expected numbers; we test that each segmented operation **reproduces the exact
quantity the un-segmented model produces**. Each module ships a self-test that runs on
`python <module>.py`; the full sweep is green.

The tests form a hierarchy, each isolating one failure mode, from the smallest unit up
to the end-to-end claim:

1. **Segment modules (`segments.py`) — why:** segmentation is only valid if the slices
   *recompose* into the original operation. So we test the algebra directly:
   concatenating the A attention head-group outputs (+ shared out-proj) must equal full
   attention; summing the M MLP hidden chunks (+ shared bias) must equal the full MLP.
   Result: **max|Δ| = 0.00e+00** (exact — these are deterministic rearrangements, not
   approximations). If this failed, every later test would be meaningless, so it is the
   foundation. It also pins the slice layout the export relies on.

2. **Forward engine — why:** composing the slices through the residual stream (with the
   resident shared params and the *streamed* cross-entropy) must equal a normal forward.
   Result: hidden **Δ=0**, loss **Δ=4.8e-7** (the only difference is logaddexp float
   re-association in chunked CE). This is also where the dropout-mode bug surfaced —
   proving the identity test catches real defects, not just confirms happy paths.

3. **Backward engine — why:** the recompute-based backward must yield the *same
   gradients* as a real `loss.backward()`; otherwise "same training" is false. Result:
   reassembled grads vs full backward **max|Δ|=1.3e-7** (float precision). This is the
   crux — it certifies the gradient is correct despite never building the full graph.

4. **Optimizer — why:** a correct gradient is not enough; the *update* must match too,
   including weight-decay grouping and clipping. We compare one streamed step to
   `torch.optim.AdamW` + `clip_grad_norm_`: **Δ=1.5e-8**.

5. **Capstone audit (`export.py`) — why it is the definitive test:** the per-stage tests
   above each hold *one* stage fixed. The real claim is that all stages, **composed and
   iterated**, do not drift. So we train the FULL model and the SEGMENTED model from the
   **same seeded init, on the same batches, for N steps** (real backward + clip + AdamW
   each step), then **reassemble** the segments and compare full weights. Result after
   4 steps: **max|Δ|=4.9e-6**. This compounds the per-step identity through the optimizer
   state and the residual stream across steps — the strongest evidence that segmented
   sequential training produces the *same trained model* as standard training. The
   round-trip slice→reassemble (Δ=0) confirms the reassembly itself is lossless.

6. **Inference — why:** decoding is a different code path (last-token logits, streamed
   argmax), so it gets its own identity test: segmented greedy decoding must emit the
   **same token sequence** as reference greedy. Result: identical.

**Tolerances, and why they differ:** slicing/forward/round-trip are deterministic
re-orderings → **exactly 0**. Backward/optimizer/multi-step involve floating-point
re-association (different summation order, chunked logaddexp, AdamW reductions) → equal
to **float precision (1e-8–1e-6)**, which is the expected and correct outcome, not a
defect. A real bug shows up as a *large* Δ (e.g. the dropout bug gave Δ=2.33), which the
tests flag immediately.

### Full verification sweep (all green)
| module | check | result |
|--------|-------|--------|
| segments | recompose == full attention/MLP | Δ = 0 |
| forward_engine | segmented forward == full | hidden Δ=0, loss Δ=4.8e-7 |
| backward_engine | grads == `loss.backward()` | Δ = 1.3e-7 |
| optimizer | step == `torch.optim.AdamW` | Δ = 1.5e-8 |
| export (capstone) | 4-step seg-train == full-train | Δ = 4.9e-6 |
| export | round-trip reassembly | Δ = 0 |
| inference | greedy == reference greedy | identical tokens |
| loader | one-active invariant | holds over 20 segments |
| stores / memory | round-trip / free | OK |
