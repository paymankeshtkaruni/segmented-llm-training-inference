# Setup plan: building `segmentation_management` from scratch

> Reconstruction spec. An engineer (or AI) following this can recreate the `segmentation_management`
> package exactly: a GPT decoder that is **segmented along 4 axes** and trained/inferred **one segment
> resident at a time**, producing a **bit-exactly identical** model to standard full training (at
> drastically lower peak memory), with torch inference, a **torch-free ONNX** inference path, and a
> tunable memory↔time granularity dial.
>
> Golden rule of the whole package: *segmentation changes **where** state lives and **when** it is
> computed, never **what** is computed.* Every memory technique below is identity-preserving; the one
> exception (recompute + dropout) is corrected by the Path B fix in §16.

---

## 0. Guarantees the implementation must hit (identity tests)

| Test (eval / no-dropout unless noted) | Target Δ |
|---|---|
| Segmented forward vs reference forward | 0 |
| Segmented backward grads vs reference `loss.backward()` | ~1.3e-7 |
| Streamed AdamW step vs `torch.optim.AdamW` | ~1.5e-8 |
| 4-step end-to-end (fwd+bwd+opt) vs full | ~4.9e-6 |
| ONNX weightless-graph forward vs torch segment forward | ~2e-6 |
| **Dropout grad consistency** (Path B, train mode): segmented backward vs single-graph autograd through the *same* dropout forward | ~1.9e-8 (CPU & GPU) |

If any of these regress, a memory technique was implemented as something other than a pure relayout.

---

## 1. File inventory (package = `segmentation_management/`)

| File | Role |
|---|---|
| `config.py` | `ModelConfig`, `SegmentationConfig`, presets, `get_preset`, `PROMPT_TEMPLATE` |
| `modules.py` | `ReferenceGPTDecoder` (the full model the segments must match) + loss/metrics |
| `segments.py` | the 4 segment modules + range helpers (the bit-exact slicing math) |
| `stores.py` | `SegmentKey`, `CpuRamStore`, `DiskStore`, `make_store`, `default_store_kind` |
| `loader.py` | `build_segment`, `StrictSegmentLoader` (one-active-segment invariant) |
| `forward_engine.py` | `SharedParams`, `SegmentedForwardEngine` (forward + chunked CE), `populate_from_reference`, `_attn_bias` |
| `backward_engine.py` | `SegmentedBackwardEngine` (recompute backward, analytic CE grad, Path B dropout RNG) |
| `optimizer.py` | `SegmentwiseAdamW` (streamed AdamW, two-pass clip, opt-state offload), `all_segment_keys` |
| `trainer.py` | `SegmentedTrainer` (orchestration, checkpoints, metrics) |
| `inference.py` | `SegmentedGenerator` (torch greedy decode, batched) |
| `optrace.py` | `mark`/`set_hook` no-op op-trace hook for the cost profiler |
| ONNX path lives in `../scripts/`: `seg_c_export_segmented.py`, `seg_c_onnx_cost.py` |

**Core invariants enforced everywhere:**
- **One segment resident at a time** (`StrictSegmentLoader` raises if violated).
- **Device → store rule:** GPU run → offload to host RAM (`cpu_ram`); CPU run → offload to `disk`.
- Large weights (qkv, mlp, embedding columns, vocab rows, **and W_o**) are segments; only tiny
  norms + per-layer MLP output bias stay resident (`SharedParams`).

---

## 2. The four segmentation axes

| Axis | Splits | Segment module | Reassembly | Range helper |
|---|---|---|---|---|
| **E** embedding | `d_model` into `E` column slices | `EmbeddingSlice` | concat on last dim | `dmodel_range(d_model,E,seg)` |
| **A** attention | `n_heads` into `A` head-groups | `AttentionHeadSegment` | concat on last dim → shared `W_o` | `head_range(n_heads,A,seg)` |
| **M** mlp | `d_ff` into `M` chunks | `MLPHiddenChunk` | running **sum** + shared bias | `hidden_range(d_ff,M,seg)` |
| **H** output head | `vocab` into `H` slices (last absorbs remainder) | `OutputHeadSlice` | streamed CE (never concatenated) | `vocab_range(vocab,H,seg)` |

`SegmentKey(layer_id, kind, seg)` identifies every segment; `layer_id = -1` for the global
embedding/output-head; per-layer for `attention`, `attn_out_proj`, `mlp`; the records store uses
`kind="layer_input"`.

---

## 3. `config.py`

```python
@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int; max_seq_len: int; n_layers: int; d_model: int; n_heads: int; d_ff: int
    dropout: float = 0.1; architecture: str = "gpt_decoder"
    pad_token_id: int | None = None; eos_token_id: int | None = None
    # __post_init__ -> validate(): all dims > 0, d_model % n_heads == 0, 0 <= dropout < 1
    # head_dim property = d_model // n_heads ; to_dict() = asdict(self)

@dataclass(frozen=True)
class SegmentationConfig:
    embedding_segments: int; attention_segments: int; mlp_chunks: int; output_head_segments: int
    attention_axis="head_groups"; mlp_axis="feedforward_hidden_dimension"
    embedding_axis="d_model"; output_head_axis="vocab"
    # code property -> "ExAxMxH"
    # validate_against_model(m): each >=1; E | d_model; A | n_heads; M | d_ff; H <= vocab
    #   (H need NOT divide vocab — last slice absorbs remainder)
```

Constants:
```python
PROMPT_TEMPLATE = "{data}\nLabel:" ; TARGET_PREFIX = " " ; ADD_BOS = False ; ADD_EOS = True

SMALL_MODEL = ModelConfig(vocab_size=2000, max_seq_len=128, n_layers=4, d_model=64,
                          n_heads=4, d_ff=256, dropout=0.1)              # accuracy model (A1/B1)
LARGE_MODEL = ModelConfig(vocab_size=50257, max_seq_len=512, n_layers=36, d_model=1280,
                          n_heads=20, d_ff=5120, dropout=0.1)            # cost model (~838M, A2/B2/C/D)

SEG_8x2x2x8   = SegmentationConfig(8, 2, 2, 8)
SEG_16x4x4x16 = SegmentationConfig(16, 4, 4, 16)

PRESETS = {
  "small_8x2x2x8":  {"model": SMALL_MODEL, "seg": SEG_8x2x2x8,   "tokenizer": "bpe"},
  "large_8x2x2x8":  {"model": LARGE_MODEL, "seg": SEG_8x2x2x8,   "tokenizer": "gpt2"},
  "large_16x4x4x16":{"model": LARGE_MODEL, "seg": SEG_16x4x4x16, "tokenizer": "gpt2"},
}
def get_preset(name): p = PRESETS[name]; p["seg"].validate_against_model(p["model"]); return p
```

**Why two model sizes** (document for the paper): small for A1/B1 because *accuracy* needs full
multi-epoch convergence (cheap on a small model, task near-saturated ~0.98 token-acc); large for
A2/B2/C/D because *memory cost* is only measurable once the model dominates the fixed overhead
(CUDA context ~0.5 GB + framework baseline). At 838M the segmentable state is ~25 GB full vs ~0.9 GB
segmented — an unambiguous win.

---

## 4. `modules.py` — the reference the segments must match bit-exactly

Pre-LN GPT decoder. **Init is part of the spec** (it defines the weights the segments inherit):

- `GPTInputEmbedding`: `token_embedding[vocab,d_model] + position_embedding[max_seq,d_model]`, then `Dropout`.
- `CausalSelfAttention`: `qkv_projection = Linear(d_model, 3*d_model)`; split heads `[B,n_heads,T,head_dim]`;
  SDPA with `dropout_p = attention_dropout.p if training else 0`; `is_causal=True` when no padding, else
  build one additive bias `[B,1,T,T]` (causal ∪ padding) with `neg = finfo(dtype).min`; then
  **`output_dropout(output_projection(out))`** (note: the output dropout — segmented must replicate it, §16c).
- `GPTMLP`: `dropout(output_projection(GELU(input_projection(x))))` — dropout on the **full** MLP output (§16d).
- `GPTBlock`: `x = x + attn(attn_norm(x)); x = x + mlp(mlp_norm(x))`.
- `ReferenceGPTDecoder`: embedding → blocks → `final_norm` → `output_projection = Linear(d_model, vocab, bias=False)`.
  **No weight tying.** Returns `(logits, hidden_pre_logits)`.

Init scheme (must match for identity vs full):
```python
def _init_weights(m):
    Linear: normal_(weight, 0, 0.02); zeros_(bias)
    Embedding: normal_(weight, 0, 0.02)
    LayerNorm: ones_(weight); zeros_(bias)
def _scale_residual_projections():           # GPT-2 residual scaling
    scale = 1/sqrt(2*n_layers)
    for blk: normal_(blk.attention.output_projection.weight, 0, 0.02*scale)
             normal_(blk.mlp.output_projection.weight,       0, 0.02*scale)
```

Loss / metrics (shifted next-token):
```python
causal_lm_cross_entropy_loss(logits, labels, ignore_index=-100):
    F.cross_entropy(logits[:,:-1].reshape(-1,V), labels[:,1:].reshape(-1), ignore_index)
token_counts(logits, labels, ignore_index=-100) -> (correct, valid)   # argmax over shifted positions
```

---

## 5. `segments.py` — the slicing math (bit-exact)

Range helpers (all `[start, end)`):
```python
head_range(n_heads, A, seg)   = (seg*(n_heads//A), (seg+1)*(n_heads//A))
hidden_range(d_ff, M, seg)    = (seg*(d_ff//M),    (seg+1)*(d_ff//M))
dmodel_range(d_model, E, seg) = (seg*(d_model//E), (seg+1)*(d_model//E))
vocab_range(vocab, H, seg):                     # last slice absorbs remainder
    base = vocab//H; start = seg*base; end = vocab if seg==H-1 else start+base; return start,end
```

Segment modules (forward math + owned weights):

- **`AttentionHeadSegment(d_model, n_heads, A, dropout)`** owns `n_heads/A` heads. `q/k/v_proj = Linear(d_model, (n_heads/A)*head_dim)`. Forward: split to `[B, heads_per_seg, T, head_dim]`, `p = attn_dropout_p if training else 0`, SDPA (`is_causal=True` or `attn_mask=attn_bias`), reshape to `[B,T,out_dim]`. **No output projection here** (W_o is the shared/streamed `attn_out_proj`).
- **`MLPHiddenChunk(d_model, d_ff, M, dropout)`** owns `d_ff/M` hidden units. `input_projection=Linear(d_model, d_ff/M)`, GELU, `output_projection=Linear(d_ff/M, d_model, bias=False)`. **Forward returns `output_projection(GELU(input_projection(x)))` — NO dropout in the chunk** (Path B §16d moves it to the summed output; `dropout_p` kept only for reference).
- **`EmbeddingSlice(vocab, d_slice, max_seq_len, dropout)`** owns `d_slice = d_model/E` columns. Forward: `token_embedding(ids) + position_embedding(arange T)`, then `F.dropout(x, dropout_p, training)`.
- **`OutputHeadSlice(d_model, vocab_slice)`** = `Linear(d_model, vocab_slice, bias=False)`. Forward: `projection(hidden_norm)`.

---

## 6. `stores.py` — off-device state

```python
@dataclass(frozen=True)
class SegmentKey:
    layer_id: int; kind: str; seg: int
    def fname(self): return f"L{layer_id}__{kind}__s{seg}.pt"

# interface: put(key,sd) get(key)->cloned sd|None  has(key)  accumulate(key,sd)  evict(key)  keys()
_to_cpu(sd) = {k: v.detach().to("cpu", copy=True) for ...}
```

- **`CpuRamStore`** (GPU runs): in-memory `dict[SegmentKey, state_dict]`; `get` returns **clones**;
  `accumulate` adds in place on host.
- **`DiskStore(root)`** (CPU runs): wipes+creates `root`; `put`=`torch.save`, `get`=`torch.load(map_location="cpu")`;
  `accumulate` = **read-modify-write** (load, add, save) so at most one segment's state is in RAM.
- `make_store(kind, root=None)`: `"cpu_ram"`→`CpuRamStore()`; `"disk"`→`DiskStore(root)` (root required).
- `default_store_kind(device)`: `"cuda" in device → "cpu_ram"` else `"disk"`.

---

## 7. `loader.py` — one-active-segment

```python
def build_segment(key, m, s):
    "attention"     -> AttentionHeadSegment(m.d_model, m.n_heads, s.attention_segments, m.dropout)
    "mlp"           -> MLPHiddenChunk(m.d_model, m.d_ff, s.mlp_chunks, m.dropout)
    "embedding"     -> EmbeddingSlice(m.vocab_size, dmodel_range(...)[1]-[0], m.max_seq_len, m.dropout)
    "output_head"   -> OutputHeadSlice(m.d_model, vocab_range(...)[1]-[0])
    "attn_out_proj" -> nn.Linear(m.d_model, m.d_model)         # W_o, streamed per layer

class StrictSegmentLoader(model, seg, store, device):
    training=True ; _active=None ; profiler=None
    set_training(mode) -> self
    load_segment(key, init_if_missing=True):
        if _active is not None: raise RuntimeError("one-active-segment violated")
        module = build_segment(key); sd = store.get(key)
        sd is not None -> load_state_dict(strict=True)
        elif not init_if_missing -> KeyError
        else -> store.put(key, module.state_dict())          # first touch persists init
        module.to(device).train(self.training)               # dropout matches mode
        _active=module; profiler?.on_load(key); return module
    release_segment(save=False):
        if save: store.put(_active_key, _active.state_dict())
        profiler?.on_release(_active_key); _active=None; free_device(device)  # gc+empty_cache/malloc_trim
    @contextmanager acquire_segment(key, save_on_exit=False, init_if_missing=True):
        m=load_segment(...); try: yield m; finally: release_segment(save=save_on_exit)
```

---

## 8. `forward_engine.py`

`SharedParams(nn.Module)` — **resident** params only: `attn_norm[L]`, `mlp_norm[L]` (LayerNorms),
`mlp_out_bias[L]` (ParameterList of `zeros(d_model)`), `final_norm`. **`attn_out_proj` is NOT here**
(it is a streamed segment — this is the §16/§15 attn_out_proj-segmentation technique).

`_attn_bias(input_ids, pad_token_id, device)`: `None` if no pad (segments use `is_causal=True`); else
`[B,1,T,T]` additive bias = causal(`triu` diag 1) ∪ padding filled with `finfo(float32).min`.

`SegmentedForwardEngine(model, seg, loader, shared, device)` with `.train()/.eval()` (propagate to
`shared` and `loader.set_training`). **`forward_hidden(input_ids, pad_token_id)`** composition:

```python
bias = _attn_bias(...)
hidden = concat_E[ acquire(embedding,e)(input_ids) ]                      # [B,T,d_model]
for L:
    xin  = shared.attn_norm[L](hidden)
    attn = concat_A[ acquire(attention,a)(xin, attn_bias=bias) ]
    with acquire(attn_out_proj,L) as op:
        hidden = hidden + F.dropout(op(attn), m.dropout, ld.training)     # (c) attn-output dropout
    xin  = shared.mlp_norm[L](hidden)
    acc  = Σ_M acquire(mlp,c)(xin)                                        # running sum, each chunk freed
    hidden = hidden + F.dropout(acc + shared.mlp_out_bias[L], m.dropout, ld.training)  # (d) MLP-sum dropout
return shared.final_norm(hidden)
```
(In eval `F.dropout(...,training=False)` is a no-op, so inference/export are unchanged by (c)/(d).)

**`chunked_ce(hidden_norm, labels, seq_chunk=128, ignore_index=-100)`** — never materializes `[B,T,vocab]`:
for each time-chunk, stream the `H` output-head slices, accumulate `lse` via `logaddexp(lse, logsumexp(z))`,
gather the true-token logit into `correct`, track a running argmax for accuracy; per-token loss = `lse - correct`.

**`populate_from_reference(ref, m, s, store) -> SharedParams`** — slices the reference into the store
(the inverse of reassembly; this defines the exact mapping):
```python
shared = SharedParams(m)
for L:                                              # shared + W_o + mlp bias
    shared.attn_norm[L].load_state_dict(blk.attention_norm.state_dict())
    shared.mlp_norm[L].load_state_dict(blk.mlp_norm.state_dict())
    store.put((L,"attn_out_proj",0), {"weight": Wo.weight, "bias": Wo.bias})
    shared.mlp_out_bias[L].copy_(blk.mlp.output_projection.bias)
shared.final_norm.load_state_dict(ref.final_norm.state_dict())
for e: store.put((-1,"embedding",e), {token/pos .weight[:, d0:d1]})              # d_model columns
for h: store.put((-1,"output_head",h), {"projection.weight": Wout[v0:v1, :]})    # vocab rows
for L:
    qw,kw,vw = qkv.weight.chunk(3,0); qb,kb,vb = qkv.bias.chunk(3,0); dh=head_dim
    for a: h0,h1=head_range(...); store.put((L,"attention",a),
            {q/k/v _proj.weight = *w[h0*dh:h1*dh], q/k/v _proj.bias = *b[h0*dh:h1*dh]})
    for c: f0,f1=hidden_range(...); store.put((L,"mlp",c),
            {input_projection.weight=Win[f0:f1], input_projection.bias=bin[f0:f1],
             output_projection.weight=Wout[:, f0:f1]})     # note: chunk output_projection has NO bias
```
The MLP **output bias** is the shared `mlp_out_bias` (added once after the sum); the attention
**output projection** bias lives in the `attn_out_proj` segment.

---

## 9. `backward_engine.py` — recompute backward + analytic CE (no global autograd graph)

`_add(d, name, g)`: park shared-param grads on **host** (`g.detach().to("cpu")`, accumulate there) — keeps
the ~226 MB/layer W_o grads off VRAM during backward.  `_rec_key(L) = SegmentKey(L,"layer_input",0)`.

`SegmentedBackwardEngine(fwd, grad_store, records_store=None)`: `records` store holds the per-layer
residual snapshot **off device** → backward peak is depth-independent. Plus the **Path B** dropout RNG
state (`self._rng`, `self._train`).

**Recording** `_record_layer_inputs` (`@torch.no_grad`): set `self._train = ld.training and m.dropout>0`,
reset `self._rng`; forward exactly like `forward_hidden`, but **capture the RNG before every dropout draw**
and store each layer-input to the records store:
```python
for e: if _train: _rng[("emb",e)] = _rng_capture()      # before EmbeddingSlice F.dropout
       hidden_parts.append(acquire(embedding,e)(ids))
for L:
   records.put(_rec_key(L), {"h": hidden})              # OFF-device snapshot
   for a: if _train: _rng[("attn",L,a)] = _rng_capture()   # before SDPA dropout
          outs.append(acquire(attention,a)(xin, bias))
   hidden = hidden + _drop_record(op(cat(outs)), ("attn_out",L))     # (c)
   acc = Σ acquire(mlp,c)(xin)
   hidden = hidden + _drop_record(acc + mlp_out_bias[L], ("mlp_out",L))  # (d)
return hidden
```

**Analytic streamed CE grad** `_ce_backward(hidden_norm, labels)` (`@torch.no_grad`, two passes):
pass 1 stream heads → `lse` via `logaddexp`. Pass 2 per head slice: `softmax = exp(z - lse)`,
`onehot` (scatter targets in range), `g_z = (softmax - onehot) * mask / n_valid`; weight grad
`gw = einsum("btv,btd->vd", g_z, sh)` → `grad_store`; hidden grad `+= einsum("btv,vd->btd", g_z, head.weight)`;
also accumulate loss (`lse - correct`) + running argmax. Return `grad_hidden` padded to `[B,T,D]`.
(Doing this analytically — *not* via autograd — avoids a ~1.5 GB graph; §15 #7a.)

**Full `backward(input_ids, labels, pad_token_id)`**:
1. `hidden_last = _record_layer_inputs(...)`.
2. final_norm: `hl=hidden_last.detach().requires_grad_()`, `hidden_norm=final_norm(hl)`,
   `g_hidden_norm = _ce_backward(...)`, `autograd.grad(hidden_norm, [hl, *fn_params], g_hidden_norm)` →
   `g` (wrt hidden_last) + final_norm grads (`_add`).
3. For `L in reversed(range(n_layers))`:
   - reload `h_in = records.get(_rec_key(L)).to(device)`.
   - **recompute** `hidden_half` (`no_grad`): re-run attention segments **restoring** `_rng[("attn",L,a)]`
     before each, then `hidden_half = h_in + _drop_value(op(concat_val), ("attn_out",L))`.
   - **MLP backward**: `mask_mlp = _drop_mask(g, ("mlp_out",L))`; `g_mlp = g*mask_mlp`;
     `_add(mlp_out_bias.L, g_mlp.sum((0,1)))`; per chunk `autograd.grad(out, [hh, mlp_norm, seg], g_mlp, retain_graph=True)`
     → chunk grads to `grad_store`, mlp_norm via `_add`, accumulate `g_half`.
   - **attn_out_proj backward**: `mask_ao = _drop_mask(g_half, ("attn_out",L))`; `g_ao = g_half*mask_ao`;
     `autograd.grad(op(c_req), [c_req, op_params], g_ao)` → `grad_concat` + W_o grads to `grad_store`.
   - **attention backward**: per segment **restore** `_rng[("attn",L,a)]`, re-run, take its slice of
     `grad_concat`, `autograd.grad(...)` → seg grads to `grad_store`, attn_norm via `_add`, accumulate `g_in`.
   - `g = g_in`; `records.evict(_rec_key(L))`.
4. **embedding backward**: per slice **restore** `_rng[("emb",e)]`, re-run, `autograd.grad(out, emb_params, g[..,d0:d1])` → `grad_store`.
5. return `shared_grads`.

The **dropout RNG helpers** (Path B, §16):
```python
_rng_capture(): (torch.get_rng_state(), torch.cuda.get_rng_state(device) if cuda else None)
_rng_restore(st): torch.set_rng_state(st[0]); if st[1]: torch.cuda.set_rng_state(st[1], device)
_drop_record(x,key): if not _train: return x; _rng[key]=_rng_capture(); return F.dropout(x,p,True)
_drop_value(x,key):  if not _train: return x; _rng_restore(_rng[key]); return F.dropout(x,p,True)  # value w/ same mask
_drop_mask(like,key):if not _train: return None; _rng_restore(_rng[key]); return F.dropout(ones_like(like),p,True)  # 0/(1/(1-p)) mask to fold into grad
```
Folding `mask` into the gradient (`g_mlp=g*mask`, `g_ao=g_half*mask`) is exact because for `y=dropout(x)`,
`∂L/∂x = (∂L/∂y)·mask`.

---

## 10. `optimizer.py` — streamed AdamW + two-pass clip + opt-state offload

```python
all_segment_keys(m,s): [embeddings] + [output_heads] + per-layer[attention, attn_out_proj, mlp]

_adamw_(p, g, st, t, lr, b1, b2, eps, wd):           # matches torch.optim.AdamW
    if wd and p.dim()>=2: p.mul_(1 - lr*wd)           # decoupled wd, only dim>=2 (GPT grouping)
    m=st["exp_avg"]; v=st["exp_avg_sq"]               # init zeros on first touch
    m.mul_(b1).add_(g, 1-b1); v.mul_(b2).addcmul_(g,g,1-b2)
    denom = (v.sqrt()/sqrt(1-b2**t)).add_(eps)
    p.addcdiv_(m, denom, value=-(lr/(1-b1**t)))

class SegmentwiseAdamW(m,s,loader,shared,grad_store,opt_store,device,
                       lr=3e-4, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1):
    t=0 ; shared_state: dict[name->{exp_avg,exp_avg_sq}] on CPU
    _clip_scale(shared_grads, max_norm):              # PASS 1 (stream): total_sq via double precision
        total_sq = Σ_segkeys Σ params (g.double()**2).sum()  (from grad_store) + Σ shared_grads
        norm = total_sq**0.5; return 1.0 if norm<=max_norm else max_norm/(norm+1e-6)
    step(shared_grads, clip_norm):                    # PASS 2 (apply + update)
        t += 1; scale = _clip_scale(...) if clip_norm else 1.0
        for key in all_segment_keys:                  # segments: load grad+opt-state to device, update, save
            with acquire_segment(key, save_on_exit=True) as seg:
                for pname,p: g=grad_store[key][pname].to(device)*scale
                             pst = opt-state for pname -> to(device); _adamw_(p.data,g,pst,...)
                             park pst back to opt_store (cpu)
        for pname,p in shared.named_parameters():     # shared: opt-state parked on host, streamed per-param
            g = shared_grads[pname].to(device)*scale
            pst = shared_state[pname] -> to(device); _adamw_(p.data,g,pst,...); shared_state[pname] = pst.to(cpu)
```
Params are never offloaded (segments are loaded/saved via the loader; shared stay resident). Only
**gradients** (always in a store / on host) and **optimizer state** (segment opt store + shared CPU dict)
move to device one piece at a time → no all-grads / all-state spike.

---

## 11. `trainer.py`

```python
lr_mult(step, warmup, total, mode="cosine"): linear warmup (step+1)/warmup; then cosine 0.5(1+cos(pi*prog))

class SegmentedTrainer(preset, device, out_dir, lr=3e-4, weight_decay=0.1, grad_clip=1.0,
                       warmup=200, scheduler="cosine", seed=42, store_kind=None):
    manual_seed(seed); m,s = get_preset(preset); tok = build_tokenizer(preset.tokenizer)
    kind = store_kind or default_store_kind(device)             # GPU->cpu_ram, CPU->disk
    param_store/grad_store/opt_store/records_store = make_store(kind, out_dir/stores/<name>)
    shared = populate_from_reference(ReferenceGPTDecoder(m), m, s, param_store).to(device); del ref
    loader = StrictSegmentLoader(...); fwd = SegmentedForwardEngine(...)
    bwd = SegmentedBackwardEngine(fwd, grad_store, records_store); opt = SegmentwiseAdamW(...)
    train_step(batch, gstep, total): fwd.train(); evict grad_store; shared_grads=bwd.backward(ii,lab,pad);
        opt.lr = lr*lr_mult(gstep,warmup,total); opt.step(shared_grads, clip_norm=grad_clip)
    evaluate(loader, split): fwd.eval(); forward_hidden -> chunked_ce -> loss + token/example acc
    save_checkpoint(tag, meta): torch.save({"params": {f"{L}|{kind}|{seg}": param_store.get(k)},
        "shared": shared.state_dict(), "model_config", "seg_config", "preset", "meta},
        out_dir/checkpoints/{tag}.pt)   # best.pt + last.pt
    fit(epochs, batch_size, ...): loop train_step; per-epoch evaluate(val) -> save best/last; final test;
        write metrics.json (epochs_detail[], best_val_loss, test_acc_global/example, total_time_s, configs)
```
Hyperparameters that MUST match full training for the equivalence claim: `lr=3e-4, wd=0.1,
betas=(0.9,0.95), eps=1e-8, clip=1.0, warmup=200, cosine, 5 epochs, batch=64, seed=42`, label-masked loss.

---

## 12. `inference.py` (torch greedy decode)

```python
class SegmentedGenerator(fwd, tokenizer):  fwd.eval()
  _next_token(ids[B,T], pad_token_id=None) -> [B,1]:   # batch-generic
     if T>max_seq_len: ids=ids[:,-max_seq_len:]
     hidden = fwd.forward_hidden(ids, pad)[:, -1:, :]
     stream output_head slices, running argmax (best_val,best_idx on ids.device), return best_idx
  generate(prompt_ids, max_new_tokens=32): loop _next_token, stop on eos
  generate_batch(prompts, max_new_tokens=32): left-pad to equal length, advance all B together (fixed steps)
  generate_label(log_line, ...): encode PROMPT_TEMPLATE.format(data=...), generate, decode/strip
```
Accumulators live on `ids.device` (GPU correctness). `pad_token_id` enables batched decode over
LEFT-padded variable-length prompts.

---

## 13. `optrace.py`

```python
_HOOK=None ; set_hook(fn): global _HOOK=fn ; mark(label): if _HOOK: _HOOK(label)
```
Zero overhead when no hook. Engines call `mark("rec|L..|store")`, `"cebwd|lse|h..."`, `"bwd|L..|mlp_chunk.."`,
`"opt|shared|<name>"`, etc. The cost profiler installs a hook to record per-op memory.

---

## 14. ONNX torch-free inference path (`../scripts/seg_c_export_segmented.py`, `seg_c_onnx_cost.py`)

### 14.1 Export
For each segment: `build_segment` → load weights → `eval` → `torch.onnx.export` (opset 18,
`do_constant_folding=True`, dynamic axes on B,T). Input name by kind: `embedding→"input_ids"`,
`attention/mlp→"x_norm"`, `output_head→"hidden_norm"`.

**Weights-as-inputs surgery** (`_lift_initializers`): torch exports `Linear` as `MatMul` with
auto-named **transposed** initializers (`onnx::MatMul_N`, not param names). Move every initializer
out of `graph.initializer` and re-add it as a graph **input** with **symbolic** dims
(`f"{name}_d{i}"`); make outputs fully dynamic. Result: a **weightless** graph. Symbolic dims are
essential for the **uneven output-head** split (`50257 % 8 = 1` → slices 6282 and 6283 share one graph).

**Signature dedup**: hash `op_types + io names` → ~**4 unique signatures** (one per kind). Save each
weightless graph once (`signatures/<kind>.onnx`). Save each segment's lifted weights as
`weights/L{L}__{kind}__s{seg}.npz` (keyed by ONNX names). Save `glue.npz` = resident state
(`attn_ln.L.*`, `mlp_ln.L.*`, `mlp_out_bias.L`, `Wo.L.*`, `final_norm.*`) + `manifest.json`
(signatures + per-segment {signature, weights_npz}) + `config.json`. Built-in verify (representative
+ non-representative segment): ONNX-with-fed-weights vs torch segment → Δ~2e-6.

### 14.2 Torch-free runtime (`SegOnnxRuntime`)
- **No torch import** (asserted: `"torch" not in sys.modules` at end).
- GPU: `ort.preload_dlls()`; fail hard if `CUDAExecutionProvider` missing (no silent CPU fallback).
- `SessionOptions`: `enable_mem_pattern=False`, `intra_op = OMP_NUM_THREADS`, `inter_op = 1`.
- **Lazy single CUDA session** (default GPU): one active session; on kind switch `del sess; gc.collect()`
  (−322 MB vs caching all four). CPU caches all four (host RAM cheap).
- **Weights**: `preload_weights=False` (default CPU) streams each `.npz` per call (disk-stream analog,
  the dominant CPU lever, −2979 MB); `True` (default GPU) caches all in RAM.
- **LayerNorm in numpy** with `eps = np.float32(eps)` (avoid float64 promotion → wrong ORT outputs).
- Forward = the §8 composition in pure numpy: concat embeddings; per layer numpy LayerNorm + per-segment
  ONNX SDPA + numpy `attn@Wo.T+bo` residual + numpy LayerNorm + per-chunk ONNX MLP running-sum + bias;
  final numpy LayerNorm. `next_token` streams output-head slices with a numpy running argmax (uneven slice
  offsets computed inline). Never materializes full logits.

---

## 15. Memory-reduction technique catalog (measured)

Cost model = 838M (`large_8x2x2x8`, B=4, T=512) unless noted.

### GPU training — A2 peak VRAM chain **3268 → 934 MB (~27×)**
> The chain below was measured step-by-step on the pre-Path-B engine. On the **final Path B engine**
> the backward peak is **964 MB** (faithful dropout adds ~30 MB of mask tensors to the backward, training
> only; inference unchanged). 838M segmented train ≈ 964 MB vs ~25 GB full ≈ **26.6×**. See MEMORY_INVESTIGATIONS §16.
| Step | Technique | Peak (MB) | Mechanism (all identity-preserving) |
|---|---|---|---|
| base | one-segment forward/backward | 3268 | starting point |
| #7a | **analytic CE backward** (no autograd graph) | 2148 | CE grad in closed form (softmax−onehot); avoids ~1.5 GB inherited graph |
| #7b | **layer-inputs offloaded** to records store | 2112 | per-layer residual snapshots to host (cpu_ram), not VRAM → depth-independent |
| #7c | **shared Adam state to host** | 1364 | ~469 MB opt-state parked in cpu_ram, streamed per param in `step` |
| #7d | **shared grads to host** (per-layer) | 1166 | ~226 MB/layer W_o grads moved to host as computed (`_add`) |
| #7e | **attn_out_proj segmentation** | **934** | W_o made a streamed segment; resident floor → ~17 MB (norms/biases) |
| — | `empty_cache` (probed) | −24 only | not load-bearing; kept (free). Structural bound already tight |

Structural bounds also exploited: **SDPA** (no `[B,H,T,T]` score matrix: 176→~10 MB attn peak),
**MLP running-sum** (peak `[B,T,d_ff/M]` not `[B,T,d_ff]`), **streamed chunked CE** (12.9 MB vs 411 MB full logits).

### CPU training
- **DiskStore** (vs cpu_ram): ~20 GB → ~1 GB resident (one segment in RAM at a time) — foundational.
- **glibc malloc tuning** (`MALLOC_ARENA_MAX=2`, MMAP/TRIM thresholds): backward peak 1372 → 1098 MB
  (−274), cross-round drift 64→38 MB (no leak), time-neutral.
- **attn_out_proj disk segment**: W_o (226 MB) off host, +~36 s/step I/O (memory is the constraint).
- Result: per-step backward peak ~1.1 GB, flat across rounds.

### GPU / CPU inference (torch, B2)
- GPU peak ~700 MB (CUDA context ~498 dominates; model+data ~200). `empty_cache` −24 (free, kept).
- CPU peak RSS ~833 MB; malloc tuning −38 (not load-bearing at inference, kept).

### ONNX inference (torch-free, C)
- 4 graph signatures (dedup); lazy CUDA session −322 MB; CPU disk-stream weights −2979 MB (85%);
  `preload_dlls` (no silent fallback); LayerNorm float32. GPU VRAM ~698 (≈torch; context-bound) but
  **host RAM ~3904 vs torch 7518 (~2×)**; CPU RSS ~526 vs torch 833 (~37%).

### Granularity dial (D) — memory ↔ time
| Config | GPU VRAM | GPU step | CPU RSS | CPU step |
|---|---|---|---|---|
| 8×2×2×8 (coarse) | 942 MB | 129 s | ~1.1 GB | 248 s |
| 16×4×4×16 (fine) | 786 MB | 208 s | 1017 MB | 354 s |
Finer segmentation → smaller per-segment working set (lower peak) at higher compute (more boundaries).
**Choose granularity from the available memory budget.**

---

## 16. The Path B dropout fixes (the "new fixations")

**Finding (§14 of MEMORY_INVESTIGATIONS):** of all memory techniques, **only the recompute backward
affects training, and only with dropout on.** Grid (segmented backward vs reference): exact (5.6e-9) in
{dropout0×eval, dropout0×train, dropout0.1×eval}; **breaks (1.06e-2) only at dropout0.1×train** — the
recompute drew fresh masks ≠ the recorded forward's. Every other technique is gradient-exact in all cells.

Two *structural* dropout differences were also found and fixed so segmented dropout matches the reference:

- **(a) RNG save/restore** — capture the RNG state before each dropout draw in the recorded forward;
  restore it before every recompute so the **same masks** are drawn (cf. `torch.utils.checkpoint
  preserve_rng_state`). For SDPA/embedding dropout, capture/restore around the segment call; for the
  added (c)/(d) dropouts use `_drop_record`/`_drop_value`/`_drop_mask`. Folding the mask into the
  gradient (`g*mask`) is exact. Works on CUDA too (SDPA philox RNG is reproducible via `cuda.set_rng_state`).
- **(c) attention output dropout** — the reference applies `output_dropout` after `output_projection`;
  the segmented model had none. Added: `hidden = hidden + F.dropout(op(attn), dropout, training)`.
- **(d) MLP dropout on the summed output** — reference drops the full `W_out·h + b`; the segmented model
  dropped each chunk independently (different statistics). Moved dropout off `MLPHiddenChunk` and onto
  `F.dropout(acc + mlp_out_bias[L], dropout, training)`.
- (b) embedding per-slice dropout is left as-is (distributionally identical; still RNG-handled).

**Verification (cheap, before any retrain):** eval identity still Δ=1.34e-7 (no-dropout intact);
dropout grad self-consistency (segmented backward vs single-graph autograd through the same masks)
Δ=1.9e-8 on **both CPU and GPU**. (See `../scripts/seg_verify_dropout_consistency.py` and
`seg_investigate_training_impact.py`.)

**Result (validated):** retrained A1 GPU (dropout 0.1, seed 42, 5 epochs) → segmented test
token-acc **0.9874** vs full **0.9877** (Δ=0.0003); B1 exact-match **0.9237** vs full **0.9244**
(Δ=0.0007) — vs the old buggy segmented 0.9096. Path B makes segmented training match full *with
dropout on*.

**Caveat for the paper:** under dropout, segmented is *distributionally* equivalent, not bit-exact
(per-head-group SDPA dropout cannot reproduce a single all-heads SDPA mask). For a *bit-exact* claim,
train with `dropout=0` (proven Δ=5.6e-9).

---

## 17. Build order (recipe)

1. `config.py` → `modules.py` (reference + loss/metrics) → `segments.py` (range helpers + 4 modules).
   Self-test: reassemble segments == reference (forward Δ=0).
2. `stores.py` → `loader.py`. Self-test: `acquire_segment` round-trips weights; one-active invariant raises.
3. `forward_engine.py` (`SharedParams`, `populate_from_reference`, `forward_hidden`, `chunked_ce`).
   Self-test: `forward_hidden` Δ=0 vs `ref(...)[1]`; `chunked_ce` loss Δ~5e-7 vs `causal_lm_cross_entropy_loss`.
4. `backward_engine.py`. Self-test: grads Δ~1.3e-7 vs `loss.backward()` (eval). Then add Path B (§16) and
   self-test dropout grad consistency Δ~1e-8 (train).
5. `optimizer.py`. Self-test: streamed AdamW step Δ~1.5e-8 vs `torch.optim.AdamW`; two-pass clip == `clip_grad_norm_`.
6. `trainer.py` + `inference.py` + `optrace.py`. Capstone: 4-step end-to-end Δ~4.9e-6 vs full.
7. ONNX export + torch-free runtime (§14). Self-test: weightless-graph forward Δ~2e-6 vs torch segment;
   `torch` never imported in the runtime.
8. Memory profiling hooks via `optrace` + `loader.profiler` to reproduce §15.

Every step is gated by its identity test before moving on (incremental verification — never big-bang).
