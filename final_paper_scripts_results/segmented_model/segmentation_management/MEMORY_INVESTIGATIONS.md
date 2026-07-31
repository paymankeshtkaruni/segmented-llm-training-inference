# Memory Investigations (measured evidence)

Each operation in the segmented stack is evaluated for peak-memory risk **by
measurement and by understanding the kernel**, before it is committed. This file
records the evidence (for the paper). Method: exact `torch.cuda.max_memory_allocated`
on GPU; `ru_maxrss` high-water delta in a fresh process on CPU. Probe: `_mem_probe.py`.

Sizes use the large model's attention dims: B=4, T=512, head_dim=64, n_heads=20.

---

## #1 — Attention scores: SDPA vs explicit `q@kᵀ`+softmax  ✅ resolved

**Risk:** the explicit path materializes the score matrix `[B, n_heads, T, T]`
(83.9 MB at full heads) plus the softmax of the same size — an `O(T²)` activation,
the classic attention memory blow-up.

**Measured peak (extra over inputs):**

| device | heads | score `[B,H,T,T]` | explicit | **SDPA** |
|--------|------:|------------------:|---------:|---------:|
| GPU    | 20    | 83.9 MB | 176.3 MB | **10.5 MB** |
| GPU    | 10 (A=2) | 41.9 MB | 83.9 MB | **5.2 MB** |
| GPU    | 5 (A=4)  | 21.0 MB | 41.9 MB | **2.6 MB** |
| CPU    | 20    | 83.9 MB | 154.4 MB | **4.1 MB** |
| CPU    | 5 (A=4)  | 21.0 MB | 44.7 MB | **2.6 MB** |

**Understanding:** SDPA (`F.scaled_dot_product_attention`) uses fused flash /
mem-efficient kernels that compute attention in tiles and **never materialize the
full score matrix**; its peak ≈ the output tensor `[B,H,T,d]` only
(`4·20·512·64·4 B = 10.5 MB`, matching GPU). On torch 2.7 the **CPU** path also
fuses (4.1 MB, not the 154 MB MATH path) — confirmed by measurement, not assumed.

**Two independent guarantees, so the bound never depends on hoping a kernel fuses:**
1. **Segmentation** bounds it to `1/A`: explicit scales 176→84→42 MB and SDPA
   10.5→5.2→2.6 MB as A=1→2→4. Even a worst-case MATH fallback is bounded by `1/A`.
2. **SDPA** removes the score matrix entirely on both devices.

**Decision:** attention (reference + segments) uses SDPA; the padding case builds a
single `[B,1,T,T]` mask (broadcast over heads) with a finite large-negative value
(NaN-robust). Per-segment peak attention activation = `[B, n_heads/A, T, head_dim]`,
~2.6 MB at A=4 — negligible. **No further action needed.**

---
*(next investigations appended as each operation is built: MLP hidden activation,
output-head logits / chunked CE, recompute-backward intermediates, streamed
optimizer step + two-pass clip.)*

## #2 — MLP hidden activation: chunking + running-sum  ✅ resolved
**Risk:** the FFN hidden activation `[B,T,d_ff]` is usually the largest activation.
**Bounded form (by construction):** each `MLPHiddenChunk` only ever builds
`[B,T,d_ff/M]` (its own hidden), maps to d_model, and the engine **sums** chunk
outputs into one `[B,T,d_model]` accumulator, freeing each chunk before the next.
So MLP peak = `[B,T,d_ff/M] + [B,T,d_model]` instead of `[B,T,d_ff]`.
**Identity proven:** sum of chunks (+ shared bias) == full MLP, max|Δ| = 0.00e+00
(segments.py self-test). Down-projection bias is excluded from chunks and added once.
**Decision:** keep running-sum in the engine; never concatenate chunk hiddens.

## #3 — Attention head-group identity  ✅ resolved
Concatenating the A head-group segment outputs (+ shared output projection) ==
full attention, max|Δ| = 0.00e+00. Confirms segmentation is exact and the per-segment
attention peak is `[B, n_heads/A, T, head_dim]` (investigation #1).

## #4 — Output-head logits: streamed chunked cross-entropy  ✅ resolved
**Risk:** the logits `[B,T,vocab]` are huge (large model: `4·512·50257·4B ≈ 411 MB`,
and ×? for softmax/backward) — the single largest tensor if materialized.
**Bounded form (by construction, airtight):** `chunked_ce` iterates seq chunks ×
H vocab slices; the ONLY large temp is `z = head_slice(h)` of shape
`[B, seq_chunk, vocab/H]`, freed each inner step. The full `[B,T,vocab]` is never
allocated anywhere in the code path. Loss is assembled from a running `logaddexp`
(= full log-sum-exp) minus the true-class logit; accuracy from a running argmax.
Peak logits temp (large, H=8, seq_chunk=128): `4·128·6283·4B ≈ 12.9 MB` vs 411 MB.
**Identity proven:** chunked CE == reference `F.cross_entropy`, |Δ|=4.8e-7 (logaddexp
float rounding); token counts identical (forward_engine.py self-test).

## #5 — Backward by recomputation (no full graph)  ✅ resolved
**Risk:** a normal `loss.backward()` retains the whole forward graph (all layers'
activations) — `O(n_layers)` memory, the thing segmentation must avoid.
**Bounded form:** no global backward. Per layer (reverse), reload ONE segment,
recompute only its forward (grad on), `torch.autograd.grad` for its params + input
grad, write param grads to the grad store, free. Records store only the per-layer
residual `[B,T,d_model]` (in cpu_ram/disk), not activations. CE backward is analytic
& streamed (softmax−onehot over vocab slices), peak `[B,seq_chunk,vocab/H]`.
Peak ≈ one segment's recompute graph + a few `[B,T,d_model]` grad tensors —
**independent of depth**.
**Identity proven:** reassembled segmented grads == reference `loss.backward()`,
max|Δ|=1.3e-7 (shared) / 5.6e-9 (mlp) / 2.6e-8 (emb) / 3.7e-9 (head).

## #6 — Streamed AdamW step + two-pass clip  ✅ resolved
**Risk:** a normal optimizer step needs params + grads + Adam (m,v) for the WHOLE
model at once (≈4× model size) — and gradient clipping needs all grads to compute the
global norm. On the constrained device that is the all-at-once spike flagged in review.
**Bounded form:** the step iterates segments; each loads only its (params + grad +
m,v) ≈ 4× ONE segment, updates in place, writes back, frees. Clipping is TWO-PASS
STREAMING: pass 1 accumulates the global Σg² one segment at a time, pass 2 reloads &
scales — the full gradient is never on the constrained device. Adam (m,v) live in the
opt-state store (cpu_ram on GPU / disk on CPU). Shared (tiny) params: in-memory AdamW.
**Identity proven:** one segmented step == `torch.optim.AdamW` (same grouping) +
`clip_grad_norm_`, max|Δ|=1.5e-8.

## #7 — A2 cost-run VRAM reductions (large 838M, GPU, bs=4, seq=512)  ✅ resolved
These were found with the per-operation profiler (`scripts/seg_cost_lib.py`), which
measures every move on one clock and now reports a GUARANTEED no-miss peak via the
allocator high-water mark (`torch.cuda.max_memory_reserved` reset per phase; `ru_maxrss`
on CPU) alongside the 1 kHz sampled flow — the two agree (e.g. HW 1162 vs smi 1166 MB),
proving the sampler misses nothing. Each reduction was measured, then kept only after
its identity test still passed (segmented == full).

| step | overall VRAM peak | backward peak | note |
|------|-------------------|---------------|------|
| honest baseline | 3268 MB | 3368 MB | one-segment + empty_cache, but with hidden costs below |
| + CE backward `@torch.no_grad` | 2148 | 2158 | CE grad pass was building an autograd graph (`sh` inherited `requires_grad`) and accumulating it across all vocab slices — ~1.5 GB of needless VRAM. Analytic grad needs no autograd. |
| + layer-input records offloaded | 2112 (3-step) | — | `_record_layer_inputs` kept all `n_layers` `[B,T,d_model]` on device (O(depth)); now streamed to the records store → backward peak depth-INDEPENDENT. |
| + shared Adam state offloaded | 1364 | 1354 | shared params' m,v (~469 MB, mostly `attn_out_proj`) sat in VRAM the whole run but is only needed during `opt.step`; now parked on host, streamed per param. |
| + shared grads parked on host | **1166** | **1162** | `_add` moves each shared grad to host as computed, so the per-layer `attn_out_proj` grads (~226 MB) never accumulate on device during backward. Also removed a within-step held-grads measurement artifact. |
| + attn_out_proj segmentation | **934** | **934** | `attn_out_proj` (W_o) converted from a resident shared param to a streamed per-layer segment (kind `attn_out_proj`): params + grad + opt-state all flow through the segment machinery. Resident floor 243 -> ~17 MB (only norms/biases). Model now FULLY segmented. |

Net: **3268 → 934 MB (~27× below the full model's 25 GB)**, all identity-exact
(backward Δ=1.3e-7, end-to-end 4-step seg-vs-full Δ=4.9e-6). The residual floor is now ~17 MB (norms/biases only); the model is fully segmented.
Profiling overhead measured separately at 9.3% of a cost step (nvidia-smi polling the
largest part); real training (A1/D) uses no profiler → 0% overhead.

## #8 — A2 CPU cost (large 838M, disk store)  ✅ resolved
On CPU the constrained tier is host RAM, so the device→target rule parks all inactive
state on DISK (one segment's worth in RAM at a time). All the §7 engine reductions apply
unchanged (CE no_grad, layer-input + attn_out_proj streaming reduce the host working set;
the opt-state / shared-grads "offloads" are no-ops here since the host already IS the
device). Measured, per the same probe→worth→apply discipline:

| lever | effect (measured) | worth? |
|-------|-------------------|--------|
| disk store (vs cpu_ram) | RAM holds ONE segment, not the whole model (~20 GB → ~1 GB) | foundational |
| glibc malloc env (`MALLOC_ARENA_MAX=2`, MMAP/TRIM=131072) | backward step RSS 1372 → 1098 MB (−274), cross-round drift 64 → 38 MB, time neutral | yes |
| attn_out_proj → disk segment | W_o (226 MB) off host RAM; +disk I/O (step 212 → 248 s) | yes (RAM is the constraint) |

**Result:** per-step training RSS ≈ **1.1 GB (backward peak), flat across rounds** (no
leak). The overall peak (~4.4 GB) is the ONE-TIME build that instantiates the full 838M
reference to slice it for identical init — setup, not the per-step cost; a
too-large-to-instantiate model would use streamed per-segment init instead. Time: ~248
s/step on CPU (recompute + disk streaming) — the memory↔time tradeoff. RSS no-miss peak
(`ru_maxrss`) == sampled, confirming nothing is missed.

## #9 — B2 segmented torch INFERENCE cost (large 838M)  ✅ resolved
Inference is forward-only (eval, no_grad) — no backward graph / grads / optimizer state —
so it is far cheaper than A2 training. Profiled with the same MemFlow (no-miss high-water,
per-op, categories not blended). Applied the T05 (GPU) / T02 (CPU) findings and PROBED
the ones with a tradeoff, logging each action:

Results (large, bs=4, seq=512, 8 decode tokens):
| | GPU | CPU |
|--|-----|-----|
| peak (prefill) | **700 MB VRAM** (ctx 498 + resident 9 + op 132) | per-step **~800 MB RSS** |
| decode peak | 638 MB | ~810 MB |
| vs A2 train | 942 MB | ~1100 MB |
| time | 20.9 s/token (recompute, no KV) | 25 s/token |

Of the 700 MB GPU peak, **498 is just the CUDA context**; model+data ≈ 200 MB (resident is
only norms/biases — attn_out_proj streamed, no grads, no opt state). CPU overall peak (4.4
GB) is the one-time build, not the per-step cost (same build-vs-step distinction as A2).

Actions logged:
- **GPU device bug fixed:** `_next_token` built `best_val`/`best_idx` on CPU → device
  mismatch on CUDA (only surfaced on GPU; the inference self-test runs on CPU). Now on-device.
- **empty_cache probe (the T05 fix):** ON 702 vs OFF 726 MB = **only +24 MB**. On this
  fully-segmented engine the VRAM bound is STRUCTURAL (one small segment + residual +
  context), NOT from `empty_cache` — unlike T05's old code where it mattered a lot. Kept
  (free, 24 MB) but it is not what bounds us.
- **Batch decode added:** `_next_token` is now batch-generic ([B,T]→[B,1]) + `generate_batch`;
  B=1 greedy identity (segmented == reference) still holds. Decode memory ≈ prefill bound.
- No-miss high-water == sampled (700 ≈ 702), confirming nothing missed.

### #9 CPU double-check (T02) — probed
All T02 levers applied (disk store = one segment in RAM; in-memory generation state; T01
malloc env). **malloc on-vs-off probe on B2 CPU inference: peak RSS 833 vs 871 MB = only
+38 MB** (vs −274 MB on A2 CPU *training*). Inference churns far less host memory, so the
malloc tuning matters little here — same structural-bound conclusion as the GPU empty_cache
probe (+24 MB). Kept (free), not load-bearing for inference.

## #10 — C segmented torch-free ONNX inference cost (large 838M)  ✅ resolved (GPU)
Export (torch, one-time): each segment -> ONNX -> weights-as-inputs surgery (initializers
become graph inputs with SYMBOLIC dims) -> weightless .onnx, dedup to 4 signatures
(embedding/attention(SDPA is_causal)/mlp/output_head); per-segment lifted weights .npz;
glue.npz (norms, mlp_out_bias, final_norm, W_o). Verified: per-signature ONNX == torch
segment (Δ=2e-6, incl non-representative layers); composed torch-free forward == torch
forward_hidden (Δ=8.3e-7). Runtime imports ONLY onnxruntime+numpy+psutil (asserted; zero
torch).

All T06/T05 torch->ONNX techniques applied: preload_dlls (no silent CPU fallback), ONE
lazy CUDA session at a time (del+gc on signature switch), ONNX bytes in CPU RAM,
SessionOptions(enable_mem_pattern=False, intra_op=OMP, inter_op=1), cpu_ram-preload/disk
weight toggle, in-memory state, torch-free guard. (fp16/int8/IOBinding NOT used.)

GPU result (bs=4, seq=512): **peak VRAM 698 MB ≈ B2 torch 700** — torch-free does NOT cut
VRAM (the CUDA runtime context ~428 MB dominates, same as torch's ~498 MB). The win is
**host RAM: 3904 MB vs torch 7518 (~2× less)** — no PyTorch runtime, and only weights+graph
bytes resident. PROBE (lazy vs cache-all sessions): **698 vs 1020 MB = −322 MB** → the
one-CUDA-session-at-a-time policy IS load-bearing for ONNX VRAM (the T06 P3-for-sessions
mechanism), unlike the empty_cache trim (+24 MB, structural).

Bugs found+fixed: (a) torch exports Linear as MatMul with auto-named TRANSPOSED weight
initializers (onnx::MatMul_N), not param names -> save the LIFTED weights per segment;
(b) uneven output_head vocab slices (6282 vs 6283 when vocab%H!=0) -> SYMBOLIC weight dims
so one weightless graph serves all.

### #10 CPU (T02/T03) — measured + probed
CPU EP, all 4 sessions cached (no VRAM), weights streamed from disk per segment, malloc env.
Result: **peak RSS 526 MB (net 485) vs B2 torch CPU 833 (~37% less)** — no torch runtime +
one segment from disk at a time. PROBES: disk-weights vs preload = **526 vs 3505 MB**
(disk streaming saves 2979 MB / 85% — the T02 disk-store mechanism, the dominant CPU lever);
malloc on/off = **526 vs 611 MB** (−85 MB). Both load-bearing, applied. torch-free asserted.

**C summary:** torch-free segmented ONNX inference matches torch on VRAM (GPU 698≈700, CUDA
context dominates) but roughly halves host RAM (GPU 3904 vs 7518; CPU 526 vs 833). The
segment-streaming P3 holds torch-free: lazy CUDA session (−322 MB GPU) and disk weights
(−2979 MB CPU) are the load-bearing mechanisms.

## #11 — D: granularity scaling (8×2×2×8 vs 16×4×4×16, large)  — PAPER DISCUSSION
**Core thesis for the paper:** segmentation **granularity is a tunable memory↔time dial.**
Splitting into MORE, SMALLER segments (16×4×4×16 vs 8×2×2×8) shrinks the per-segment
working set, so **peak memory drops**; the price is **more time** (more segment
load/offload + more recompute boundaries). Therefore the granularity level is **chosen to
fit the AVAILABLE memory budget**:
  * little memory available  -> segment FINER (fits, but slower),
  * more memory available    -> segment COARSER (faster, fewer boundaries).

Consequence to argue in the paper: with sequential segmented training you can fit a given
model into *almost any* memory budget by trading compute time — granularity is the control
knob, and D quantifies the tradeoff curve (peak memory and step time at two granularities;
extendable to a curve over more granularities). Cost measured on the LARGE model only
(train cost, A2 harness), GPU + CPU, identity-exact (16×4×4×16 engine verified: forward
Δ=1e-6, backward Δ=6e-9).

Measured (to be filled when D runs land):
| config | GPU peak VRAM | GPU step time | CPU step RSS | CPU step time |
|--------|---------------|---------------|--------------|---------------|
| 8×2×2×8   | 942 MB (model+data 426) | 129 s | ~1.1 GB | 248 s |
| 16×4×4×16 | 786 MB (model+data 278) | 208 s | 1017 MB | 354 s |

## #12 — Full-model vs segmented: training equivalence & accuracy (PAPER DISCUSSION)

### Are any TRAINING OPTIONS different? No.
Every option/hyperparameter matches full_model by design (verified by diffing
full_model/scripts/_train_lib.py + _common.py against segmentation_management):
| option | full_model | segmented |
|--------|-----------|-----------|
| arch (small) | maxseq128, 4L, d64, 4h, ff256, drop0.1, vocab2000 bpe | identical |
| lr / wd / betas / eps | 3e-4 / 0.1 / (0.9,0.95) / 1e-8 | identical |
| grad-clip | 1.0 (`clip_grad_norm_`) | 1.0 (two-pass STREAMING global norm) |
| warmup / sched / epochs / batch / seed | 200 / cosine / 5 / 64 / 42 | identical |
| AdamW decay grouping | decay on dim>=2, none on biases/norms | identical |
| tokenizer / prompt / add_bos / add_eos / loss-mask / shuffle | bpe2000 / "{data}\nLabel:" / F / T / label-only / True | identical |

### Training MECHANISM (segmented), all identity-verified vs standard training:
- **Full backward by recomputation** (no global graph): per layer reload one segment,
  recompute its forward, autograd.grad for its grads + input grad, stream grads to store.
  Exact full gradient for EVERY param == loss.backward(), Δ=1.3e-7. Peak depth-independent.
- **Streamed AdamW (`after_full_backward`)**: full update of ALL params, one segment at a
  time (params+grad+m,v ~= 4x one segment). == torch.optim.AdamW, Δ=1.5e-8.
- **Global gradient clipping = two-pass streaming**: pass 1 accumulates the GLOBAL Σg^2
  across all segments (streamed); pass 2 reloads & scales. True global norm == clip_grad_norm_(1.0).
- **No multi-batch accumulation** (1 batch/step, == full_model). Within-step accumulation
  (shared grads, MLP running-sum) is intrinsic to one-segment processing.

### The only differences are IMPLEMENTATION-level (not options):
1. **Separate model reimplementation** — full_model uses src/.../GPTDecoder; segmented uses
   the clean ReferenceGPTDecoder. Same architecture SPEC, independent code -> can differ in
   weight-init scheme / LayerNorm eps, so they don't start from bit-identical weights even
   at the same seed.
2. **RNG / data order** — model construction consumes the RNG differently (different init
   code), so seeded shuffle=True yields a different batch order -> different trajectory.
Hence segmented A1 and full A1 are TWO INDEPENDENT runs of the same-spec model.

### Accuracy comparison (same small bpe-2k model, full vs segmented training)
| metric | full_model | segmented |
|--------|-----------|-----------|
| A1 token-acc (teacher-forced, global) GPU/CPU | 0.9873 / 0.9874 | 0.9817 / 0.9811 |
| B1 exact-match (generation, full test) GPU | 0.9344 | 0.9096 |
| B1 level-acc GPU | 0.9840 | 0.9760 |
- **A1 ~matches** (~0.5% gap = run variance from the two implementation differences above).
- **B1 wider gap (0.910 vs 0.934)**: likely a PROMPT-TRUNCATION difference (full_model caps
  the prompt to max_seq-max_new keeping the HEAD; segmented `_next_token` caps to max_seq at
  decode keeping the TAIL) — segmented accuracy drops specifically on the LONGEST prompts.
  Not a model defect; aligning truncation should close it. (Open item.)

### Metric definitions (for the paper)
- **Token accuracy (A1)**: teacher-forced, per-token — predict next token given the TRUE
  prefix; fraction of label tokens correct. Lenient (no error cascade); cheap (one forward).
- **Exact-match (B1)**: free greedy generation (model feeds its OWN tokens); the WHOLE
  generated label must equal the truth. Strict (errors cascade); the honest deployment metric.
  Label = "Issue: X, level: Y" -> exact = both right; issue-acc = X; level-acc = Y.

### Bottom line for the paper
No training option differs. The rigorous "segmented training == full training" claim rests
on the in-codebase IDENTITY tests (forward Δ=0, backward Δ=1.3e-7, AdamW Δ=1.5e-8,
end-to-end 4-step seg-vs-full Δ=4.9e-6, reassembly round-trip Δ=0) — NOT on matching two
independent runs. The cross-implementation A1/B1 numbers merely corroborate (both ~0.98).

## #13 — Why a SMALL model for accuracy (A1/B1) and a LARGE model for cost (A2/B2/C/D)
Two models on purpose:
- **SMALL (BPE-2k, ~0.46M; 4L, d64, 4h, ff256)** for the ACCURACY experiments (A1 train,
  B1 inference). Rationale: accuracy needs FULL multi-epoch training to converge, and the
  segmented path is compute-heavy (recompute + per-segment load/offload), so a small model
  keeps training TIME tractable while the task still has headroom (token-acc ~0.98). The
  point of A1/B1 is "does segmented training reach the same ACCURACY as full" — model size
  is irrelevant to that, so we use the cheapest model that learns the task.
- **LARGE (GPT-2 vocab, ~838M; 36L, d1280, 20h, ff5120)** for the COST experiments (A2
  train, B2/C inference, D scalability). Rationale: memory cost must be DOMINATED BY THE
  MODEL, not by fixed framework overhead (CUDA context ~0.5 GB, Python/torch baseline). On a
  small model the model+data is a few MB and the savings are invisible under the baseline; on
  838M the segmentable state is GBs, so the one-segment-at-a-time peak is clearly MEASURABLE
  and the savings (e.g. train 25 GB full -> ~0.9 GB segmented) are unambiguous. Cost is a
  per-step quantity (no convergence needed), so a few steps on the large model suffice —
  fast despite the model size.
So: small model = "is the trained model the SAME?" (accuracy, needs convergence, must be
cheap); large model = "how much memory does it save?" (cost, needs scale to be measurable,
needs only a few steps). This split keeps both questions answerable within compute budget.

## #14 — Which memory technique impacts TRAINING? Only recompute+dropout.  (KEY FINDING)
Question: of all the memory-reduction techniques, which actually change the trained model
(vs being pure memory-layout)? Probe: segmented-backward-vs-reference gradient agreement on
the 2x2 grid {dropout 0 / 0.1} x {eval / train} (scripts/seg_investigate_training_impact.py).

| dropout | mode | mlp-chunk grad max|Δ| |
|---|---|---|
| 0.0 | eval | 5.6e-9 (exact) |
| 0.0 | train | 5.6e-9 (exact) |
| 0.1 | eval | 5.6e-9 (exact) |
| 0.1 | **train** | **1.06e-2 — BREAKS** |

**Finding:** the **recompute backward** is the ONLY training-impacting technique, and ONLY
when the forward is stochastic (dropout in train mode). The backward recomputes the forward
and draws FRESH dropout masks != the recorded forward's masks, so the gradient is w.r.t. a
different dropout realization than the loss -> ~1% per-step gradient error.
- train mode alone (dropout=0): exact. dropout alone (eval): exact. Only both together break.
- ALL layout techniques (offload to store, empty_cache/malloc, opt-state offload,
  shared-grads-to-host, layer-input offload, attn_out_proj segmentation, streamed AdamW,
  two-pass streaming clip) are gradient-EXACT in every cell -> ZERO training impact (they
  change WHERE state lives, not WHAT is computed). This is why all eval-mode identity tests
  pass (Δ~1e-7) and the end-to-end no-dropout capstone is Δ=4.9e-6.

**Consequence:** explains the systematic train/infer gap measured against an in-codebase full
(non-segmented) ReferenceGPTDecoder (same arch/hparams/seed/data, dropout=0.1, 5 epochs,
5 seeds): full token-acc ~0.988 / B1 ~0.926 vs segmented 0.982 / 0.910 — full is consistently
~1.6% higher (directional, not variance). The dropout-recompute mask mismatch is the cause.

**Fix options (to decide):**
1. **RNG save/restore around recompute** (cf. torch.utils.checkpoint preserve_rng_state):
   capture the RNG state before each segment's forward; restore it before the recompute so the
   SAME dropout masks are drawn -> bit-exact training WITH dropout. Preferred (no extra memory).
2. **Store dropout masks** in the records and reuse in recompute (simple, but +memory per mask).
3. **Dropout-free training** (sidestep; loses regularization) — only if dropout isn't needed.

## #15 — Path B fix (RNG save/restore + faithful dropout): RESOLVED, segmented == full under dropout
Implemented three identity-restoring dropout fixes in the engine (see setup_segmented_model_plan.md §16):
(a) per-segment RNG save/restore so the backward recompute draws the SAME dropout masks as the recorded
forward (cf. torch.utils.checkpoint preserve_rng_state); (c) added the attention OUTPUT dropout the
reference has (was missing); (d) MLP dropout on the SUMMED output (acc+bias) instead of per-chunk.

Verification (cheap, before retrain):
- Eval identity unchanged: backward Δ=1.34e-7 (no-dropout path intact).
- Dropout grad self-consistency (segmented backward vs single-graph autograd through the SAME masks):
  worst param Δ=1.9e-8 on BOTH CPU and GPU (attn q_proj 1.0e-10). CUDA SDPA philox RNG IS reproducible
  via cuda.set_rng_state. Scripts: scripts/seg_verify_dropout_consistency.py, seg_investigate_training_impact.py.

Decision gate (retrain A1 GPU dropout=0.1, seed42, 5 epochs; full baseline seed42):
- segmented test token-acc = 0.9874  vs full 0.9877   (Δ=0.0003)
- segmented B1 exact-match = 0.9237   vs full 0.9244   (Δ=0.0007)   [old buggy segmented was 0.9096]
=> within +-0.01 gate -> SUCCESS. Path B kept; .path_b_backup deleted. New checkpoint:
   outputs/seg_train_gpu_pathb/checkpoints/best.pt ; B1: outputs/seg_b1_pathb/infer_metrics.json.

Caveat (paper): under dropout segmented is DISTRIBUTIONALLY equivalent + gradient-consistent, not
bit-exact (per-head-group SDPA dropout cannot reproduce a single all-heads SDPA mask). Bit-exact claim
holds at dropout=0 (Δ=5.6e-9). With Path B, accuracy matches full with dropout on.

## #16 — Cost reruns on the Path B engine (consistency check)
All cost experiments re-measured on the final (Path B) engine, 4 GPU nodes/batch. Result: training
backward peak rises a small, explainable amount (the faithful-dropout mask tensors [B,T,d_model] +
extra dropout activations in the recompute); inference is byte-identical (Path B is inert in eval).

| Experiment | metric | old | new (Path B) | Δ |
|---|---|---|---|---|
| A2 GPU (train) | backward peak VRAM | 934 | 964 MB | +30 |
| A2 CPU (train) | backward peak RSS  | 1136 | 1178 MB | +42 |
| D  GPU (train, 16x4x4x16) | backward peak VRAM | 786 | 832 MB | +46 |
| D  CPU (train, 16x4x4x16) | backward peak RSS  | 1017 | 1056 MB | +38 |
| B2 GPU (torch infer) | prefill/decode VRAM | 700/638 | 700/638 | 0 |
| B2 CPU (torch infer) | prefill/decode RSS  | 833/811 | 835/816 | ~0 (noise) |
| C  GPU (ONNX infer) | peak VRAM | 698 | 698 | 0 |
| C  CPU (ONNX infer) | net RSS | 485 | 514 | +29 (run variance; torch-free, Path-B-independent) |

Times all within shared-partition node noise (no systematic change). Headline holds: 838M segmented
training peaks ~964 MB VRAM vs ~25 GB full (~26.6x). Update §7e (934 -> 964) / §11 D-curve accordingly.
The +30-46 MB is the price of faithful dropout (training only); it does not change the memory story.

## #17 — Remove the full-model "build" spike: populate_from_scratch (one segment at a time)
Problem: the cost harness (via SegmentedTrainer) initialized by building the FULL ReferenceGPTDecoder
(838M) in RAM and slicing it (populate_from_reference) -> a one-time setup spike (CPU disk-store: RSS
~4.3 GB; GPU: full-model transient ~3 GB on top of the host store). That contradicts the one-segment
principle even though it was only setup.

Fix (forward_engine.populate_from_scratch + trainer from_scratch flag, default ON for cost via
seg_cost_lib --from-reference to opt back in): build each segment ALONE, GPT-init it (N(0,0.02);
residual proj x 1/sqrt(2L)), write to store, free it -> setup peak = ONE segment. The full model is
NEVER materialized. populate_from_reference kept only for the bit-exact identity proof + ONNX export.

NOT a measurement trick: the MemFlow sampler is unchanged (starts before build, samples continuously);
the build window is still fully recorded (11k-18k samples). The recorded build peak dropped because the
memory is genuinely never spent.

Reruns (from-scratch engine), build-phase peak vs per-step peak (per-step UNCHANGED):
| run | build peak (was) | build peak (now) | per-step peak |
|---|---|---|---|
| A2 CPU (RSS) | 4349 MB | 936 MB | 1175 MB (=1178) |
| D  CPU (RSS) | ~spike  | 886 MB | 1056 MB (=1056) |
| A2 GPU | RSS 6.4 GB / VRAM - | RSS 4296 / VRAM 510 | VRAM 964 (=964) |
| D  GPU | -                   | RSS 4292 / VRAM 510 | VRAM 832 (=832) |
CPU: spike eliminated (build peak ~= one-segment floor). GPU: full-model transient removed; remaining
~4.3 GB RSS is the host-RAM store itself (legit offload target on GPU), VRAM build = CUDA context only.
Plots (A2/D/scalability) regenerated from the spike-free runs.

## #18 — Speed of the memory↔time tradeoff (what streaming/offload/granularity cost)
Segmentation buys memory with TIME. There is ONE mode (tiny params resident, big params streamed one
segment at a time); the speed knobs are the offload medium and the granularity. Measured on the large
838M model (B=4, T=512, 3 measured steps), per-step and per-phase wall time:

| run | store/medium | peak | avg step | forward / backward / optimizer (s) |
|---|---|---|---|---|
| A2 GPU 8x2x2x8  | cpu_ram (PCIe) | 964 MB  | 125.1 s | 20.4 / 54.2 / 27.0 |
| D  GPU 16x4x4x16| cpu_ram (PCIe) | 832 MB  | 207.1 s | 34.5 / 90.5 / 42.1 |
| A2 CPU 8x2x2x8  | disk (I/O)     | 1175 MB | 245.7 s | 38.5 / 123.1 / 53.0 |
| D  CPU 16x4x4x16| disk (I/O)     | 1056 MB | 360.6 s | 56.1 / 175.9 / 83.4 |

Four cost factors:
1. **Streamed vs resident (per param):** a resident param is instant; a streamed one is loaded from the
   store on every use (W_o is loaded TWICE per layer in backward: hidden_half recompute + grad). So
   only tiny, constantly-used params (norms/biases) are kept resident — streaming them would cost time
   for negligible memory. Moving big params (W_o, shared opt-state) to streamed/offloaded saved memory
   at a small time cost.
2. **Offload medium = the biggest speed lever:** cpu_ram (host RAM over PCIe) vs disk. Same work, GPU
   (cpu_ram) 125 s/step vs CPU (disk) 246 s/step — ~2x slower, dominated by per-segment disk I/O.
3. **Granularity = the dial:** finer = lower peak, more time (more load/offload/recompute boundaries).
   GPU 8x2x2x8 -> 16x4x4x16: peak 964->832 MB (-14%), step 125->207 s (+66%). CPU 246->361 s.
4. **Recompute dominates the backward:** backward (54 s GPU A2) ~2.7x the forward (20 s) because it
   recomputes the forward AND re-streams every segment (no global autograd graph kept — that is what
   bounds the peak). Backward is the single largest time sink in every run.

Takeaway for the paper: frame the overhead as the price of fitting any model in any budget, not as
beating full training on speed. The two tunable knobs are offload medium (RAM >> disk for speed) and
granularity (coarser for speed, finer for memory). Numbers from outputs/seg_cost_{gpu,cpu}{,_16x4x4x16}.
