"""Memory probe: PROVE (not assume) the attention memory behavior.

Question 1: does SDPA avoid materializing the [B, H, T, T] score matrix vs the
            explicit q@kᵀ+softmax@v path? Measure peak VRAM for both.
Question 2: does head-group SEGMENTATION (H -> H/A heads at a time) bound it?
Question 3: what SDPA backends are available on GPU and CPU (does CPU fuse)?
Measured with torch.cuda.max_memory_allocated (exact). For CPU we report the
available backend + the theoretical score size (CPU has no exact allocator probe).
"""
import math
import torch
import torch.nn.functional as F


def explicit(q, k, v):
    s = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    s = F.softmax(s, dim=-1)
    return s @ v


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def measure_cuda(fn, q, k, v):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn(q, k, v)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return (peak - base) / 1e6


def run(device):
    print(f"\n===== device={device} =====")
    print("available SDPA backends:")
    try:
        from torch.nn.attention import SDPBackend
        for b in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                  SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION):
            print("   ", b)
    except Exception as e:
        print("   (SDPBackend enum unavailable:", e, ")")

    B, T, Dh = 4, 512, 64
    for n_heads, label in [(20, "full (large, A=1)"), (10, "A=2 per-seg"), (5, "A=4 per-seg")]:
        q = torch.randn(B, n_heads, T, Dh, device=device)
        k = torch.randn(B, n_heads, T, Dh, device=device)
        v = torch.randn(B, n_heads, T, Dh, device=device)
        score_mb = B * n_heads * T * T * 4 / 1e6
        if device == "cuda":
            me = measure_cuda(explicit, q, k, v)
            ms = measure_cuda(sdpa, q, k, v)
            print(f"  H={n_heads:>2} {label:18} | score[B,H,T,T]={score_mb:6.1f}MB | "
                  f"explicit peak={me:6.1f}MB | SDPA peak={ms:6.1f}MB | "
                  f"saved={me-ms:6.1f}MB")
        else:
            # CPU: no exact allocator probe; report theoretical score size
            print(f"  H={n_heads:>2} {label:18} | score[B,H,T,T]={score_mb:6.1f}MB "
                  f"(materialized iff CPU SDPA uses MATH backend)")
        del q, k, v


def cpu_case(kind: str, n_heads: int):
    """Measure CPU peak RSS (ru_maxrss high-water) attributable to one attention op.
    Run in a FRESH process per case so one op's high-water can't hide the other."""
    import resource
    B, T, Dh = 4, 512, 64
    q = torch.randn(B, n_heads, T, Dh)
    k = torch.randn(B, n_heads, T, Dh)
    v = torch.randn(B, n_heads, T, Dh)
    _ = (q + 1).sum()  # touch, stabilize allocator
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB->MB
    out = explicit(q, k, v) if kind == "explicit" else sdpa(q, k, v)
    _ = out.sum()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    score_mb = B * n_heads * T * T * 4 / 1e6
    print(f"  CPU {kind:8} H={n_heads:>2} | score[B,H,T,T]={score_mb:6.1f}MB | "
          f"peak RSS delta over baseline = {peak-base:6.1f}MB")


if __name__ == "__main__":
    import sys
    print("torch:", torch.__version__)
    if len(sys.argv) >= 3 and sys.argv[1] == "--cpu-case":
        cpu_case(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    else:
        if torch.cuda.is_available():
            run("cuda")
        run("cpu")
