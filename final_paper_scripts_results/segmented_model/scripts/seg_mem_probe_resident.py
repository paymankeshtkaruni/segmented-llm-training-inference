"""
GPU resident-composition probe — measure EXACTLY what occupies VRAM at the floor, so
any reduction is decided on evidence (not by copying T-series fixes blindly).

Isolates, by torch.cuda.memory_allocated at well-defined points:
  * shared params on device          = allocated right after build
  * shared Adam state (m,v)          = allocated after a step, minus held grads, minus params
  * held shared grads (last_grads)   = the bit freed when last_grads is dropped
Then measures the backward ALLOCATED peak and the resident floor, so we can predict the
peak reduction from offloading the shared opt state (it is resident during backward).
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

SEG_PKG = Path(__file__).resolve().parent.parent / "segmentation_management"
sys.path.insert(0, str(SEG_PKG))
from trainer import SegmentedTrainer        # noqa: E402

def MB(x): return x / (1024**2)

def main():
    dev = "cuda"
    torch.zeros(1, device=dev); torch.cuda.synchronize()
    ctx_alloc0 = MB(torch.cuda.memory_allocated())
    tr = SegmentedTrainer("large_8x2x2x8", dev, Path("/tmp/seg_resident_work"), store_kind="cpu_ram")
    m = tr.m; pad = tr.tok.pad_token_id
    B, T = 4, m.max_seq_len
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, m.vocab_size, (B, T), generator=g)
    if pad is not None: x[x == pad] = (int(pad)+1) % m.vocab_size
    y = x.clone(); y[:, 0] = -100
    x, y = x.to(dev), y.to(dev)
    torch.cuda.synchronize()
    a_build = MB(torch.cuda.memory_allocated())        # shared params resident on device

    # one full step to create grads (cpu_ram store, off-device) + shared opt state (on device)
    grads = tr.bwd.backward(x, y, pad_token_id=pad)
    tr.opt.step(grads, clip_norm=1.0)
    torch.cuda.synchronize()
    a_after = MB(torch.cuda.memory_allocated())        # params + shared opt state + held grads

    del grads
    import gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    a_noheld = MB(torch.cuda.memory_allocated())       # params + shared opt state

    shared_params = a_build - ctx_alloc0
    shared_opt    = a_noheld - a_build
    held_grads    = a_after - a_noheld

    # backward allocated peak on top of the (params + opt state) floor
    torch.cuda.reset_peak_memory_stats()
    floor = a_noheld
    g2 = tr.bwd.backward(x, y, pad_token_id=pad)
    torch.cuda.synchronize()
    bwd_peak = MB(torch.cuda.max_memory_allocated())

    print(f"[resident] large {m.n_params_estimate/1e6:.0f}M  B={B} T={T}")
    print(f"  ctx torch-alloc baseline ......... {ctx_alloc0:7.1f} MB")
    print(f"  shared params (resident) ......... {shared_params:7.1f} MB")
    print(f"  shared Adam state m,v (resident) . {shared_opt:7.1f} MB   <- offload candidate")
    print(f"  held last_grads (shared) ......... {held_grads:7.1f} MB   <- release candidate")
    print(f"  resident floor (params+optstate) . {floor:7.1f} MB")
    print(f"  backward ALLOCATED peak .......... {bwd_peak:7.1f} MB")
    print(f"  => predicted peak if opt-state offloaded: ~{bwd_peak - shared_opt:.0f} MB "
          f"(and if grads not held: ~{bwd_peak - shared_opt - held_grads:.0f} MB)")

if __name__ == "__main__":
    main()
