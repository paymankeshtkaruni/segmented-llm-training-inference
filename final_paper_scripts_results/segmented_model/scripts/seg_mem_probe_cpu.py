"""
CPU RAM unload probe — does the segmented CPU path TRULY hold only one segment at a
time, or does RAM stay elevated / climb (incomplete unload)?

Measures process RSS:
  * after build (resident floor),
  * at EVERY segment release within one backward (should return to a stable floor;
    a climb = segments / their loaded payloads are not being freed),
  * per phase and across several steps (a step-over-step climb = a leak),
  * after explicitly dropping last_grads + gc + malloc_trim (isolates what is
    resident-by-design: SharedParams + their Adam state + held grads).

No correctness path touched — read-only instrumentation.
"""
from __future__ import annotations
import ctypes, gc, sys
from pathlib import Path

import torch

SEG_PKG = Path(__file__).resolve().parent.parent / "segmentation_management"
sys.path.insert(0, str(SEG_PKG))
from trainer import SegmentedTrainer            # noqa: E402
from memory import host_rss_mb                  # noqa: E402


def trim():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass


class RssTrace:
    def __init__(self): self.phase="?"; self.rel=[]   # (phase, idx, rss)
    def set_phase(self,p): self.phase=p
    def on_load(self,k): pass
    def on_release(self,k): self.rel.append((self.phase, len(self.rel), host_rss_mb()))


def main():
    dev="cpu"
    tr=SegmentedTrainer("large_8x2x2x8", dev, Path("/tmp/seg_cpu_probe_work"), store_kind="disk")
    m=tr.m; pad=tr.tok.pad_token_id
    B,T=4,m.max_seq_len
    g=torch.Generator().manual_seed(0)
    x=torch.randint(0,m.vocab_size,(B,T),generator=g)
    if pad is not None: x[x==pad]=(int(pad)+1)%m.vocab_size
    y=x.clone(); y[:,0]=-100

    trim(); floor=host_rss_mb()
    print(f"[cpu-probe] large {m.n_params_estimate/1e6:.0f}M  B={B} T={T}  store=disk")
    print(f"  RSS after build (resident floor): {floor:.0f} MB")

    tr_prof=RssTrace(); tr.loader.profiler=tr_prof
    last=None
    for it in range(3):
        r0=host_rss_mb()
        tr_prof.set_phase(f"it{it}.fwd")
        with torch.no_grad(): _=tr.fwd.forward_hidden(x,pad_token_id=pad)
        r_f=host_rss_mb()
        tr_prof.set_phase(f"it{it}.bwd")
        gr=tr.bwd.backward(x,y,pad_token_id=pad); last=gr
        r_b=host_rss_mb()
        tr_prof.set_phase(f"it{it}.opt")
        tr.opt.step(gr,clip_norm=1.0)
        r_o=host_rss_mb()
        tr_prof.set_phase(f"it{it}.val")
        tr.fwd.eval()
        with torch.no_grad():
            h=tr.fwd.forward_hidden(x,pad_token_id=pad); tr.fwd.chunked_ce(h,y)
        tr.fwd.train()
        r_v=host_rss_mb()
        print(f"  step {it}: start={r0:.0f}  fwd={r_f:.0f}  bwd={r_b:.0f}  opt={r_o:.0f}  val={r_v:.0f} MB"
              f"   (Δstep={r_v-r0:+.0f})")

    # within-backward release trace (last step's bwd): does RSS return to a floor?
    bwd_rel=[r for ph,_,r in tr_prof.rel if ph=="it2.bwd"]
    if bwd_rel:
        print(f"  within last backward: {len(bwd_rel)} releases  RSS min={min(bwd_rel):.0f} "
              f"max={max(bwd_rel):.0f} first={bwd_rel[0]:.0f} last={bwd_rel[-1]:.0f} MB")

    # isolate resident-by-design
    peak=host_rss_mb()
    del last; trim(); after_drop=host_rss_mb()
    print(f"  RSS now={peak:.0f}  after drop(last_grads)+gc+malloc_trim={after_drop:.0f} MB "
          f"(freed {peak-after_drop:.0f})")
    print(f"  => resident floor was {floor:.0f} MB; step-over-step climb tells leak vs stable.")


if __name__=="__main__":
    main()
