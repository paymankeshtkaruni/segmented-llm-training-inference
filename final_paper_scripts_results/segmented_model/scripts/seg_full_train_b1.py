"""
Full (NON-segmented) training of the SAME ReferenceGPTDecoder the segmented engine
implements, then the FULL-test B1 (batched greedy exact-match). Same hyperparameters as
segmented A1 (lr 3e-4, wd 0.1, betas (0.9,0.95), clip 1.0, warmup 200, cosine, 5 epochs,
batch 64). Purpose: show full training reaches the same accuracy as segmented (which is
bit-exactly the same model) — a clean in-codebase full-vs-segmented comparison.
"""
from __future__ import annotations
import argparse, math, re, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

SEG_PKG = Path(__file__).resolve().parent.parent / "segmentation_management"
sys.path.insert(0, str(SEG_PKG))
from config import get_preset, PROMPT_TEMPLATE                              # noqa: E402
from modules import ReferenceGPTDecoder, causal_lm_cross_entropy_loss, token_counts  # noqa: E402
from data import build_tokenizer, GenerationCollator, LazyLogDataset, TRAIN_CSV, VAL_CSV, TEST_CSV  # noqa: E402


def lr_mult(step, warmup, total, mode="cosine"):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))


def parse_il(s):
    m = re.match(r"\s*[Ii]ssue:\s*(.*?),\s*level:\s*(.*)", s.strip())
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, None)


@torch.no_grad()
def token_acc(model, loader, device, pad):
    model.eval(); gc = gv = 0
    for b in loader:
        ii = b["input_ids"].to(device); lab = b["labels"].to(device)
        logits, _ = model(ii, pad_token_id=pad)
        c, v = token_counts(logits, lab); gc += c; gv += v
    return gc / gv if gv else float("nan")


@torch.no_grad()
def b1_full_test(model, tok, device, pad, eos, batch=64, max_new=40):
    model.eval()
    test = LazyLogDataset(TEST_CSV)
    ex = [test[i] for i in range(len(test))]
    prompts = [tok.encode(PROMPT_TEMPLATE.format(data=e["data"]), add_special_tokens=False) for e in ex]
    order = sorted(range(len(ex)), key=lambda i: len(prompts[i]))
    n = exact = 0
    for b0 in range(0, len(order), batch):
        bidx = order[b0:b0 + batch]; bp = [prompts[i] for i in bidx]
        maxlen = max(len(p) for p in bp); B = len(bp)
        ids = torch.full((B, maxlen), pad, dtype=torch.long)
        for j, p in enumerate(bp):
            ids[j, maxlen - len(p):] = torch.tensor(p)
        ids = ids.to(device); outs = [[] for _ in bp]; done = [False] * B
        for _ in range(max_new):
            if ids.size(1) > model.cfg.max_seq_len:
                ids = ids[:, -model.cfg.max_seq_len:]
            logits, _ = model(ids, pad_token_id=pad)
            nxt = logits[:, -1].argmax(-1)
            for j in range(B):
                if not done[j]:
                    t = int(nxt[j])
                    if t == eos: done[j] = True
                    else: outs[j].append(t)
            ids = torch.cat([ids, nxt.view(B, 1)], dim=1)
            if all(done): break
        for j, i in enumerate(bidx):
            pred = tok.decode(outs[j], skip_special_tokens=True).strip()
            exact += int(pred == ex[i]["label"].strip()); n += 1
    return exact / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small_8x2x2x8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--save-ckpt", type=Path, default=None,
                    help="save the trained state_dict here after training")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    p = get_preset(a.preset); m = p["model"]
    tok = build_tokenizer(p["tokenizer"]); pad = tok.pad_token_id; eos = tok.eos_token_id
    model = ReferenceGPTDecoder(m).to(a.device)
    coll = GenerationCollator(tok, m.max_seq_len)
    g = torch.Generator(); g.manual_seed(a.seed)
    trl = DataLoader(LazyLogDataset(TRAIN_CSV), batch_size=a.batch_size, shuffle=True,
                     collate_fn=coll, generator=g)
    tel = DataLoader(LazyLogDataset(TEST_CSV), batch_size=a.batch_size, shuffle=False, collate_fn=coll)
    decay = [pp for pp in model.parameters() if pp.dim() >= 2]
    nodecay = [pp for pp in model.parameters() if pp.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    total = a.epochs * len(trl); gstep = 0
    t0 = time.perf_counter()
    for ep in range(a.epochs):
        model.train()
        for b in trl:
            mult = lr_mult(gstep, a.warmup, total)
            for pg in opt.param_groups: pg["lr"] = 3e-4 * mult
            ii = b["input_ids"].to(a.device); lab = b["labels"].to(a.device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(ii, pad_token_id=pad)
            loss = causal_lm_cross_entropy_loss(logits, lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); gstep += 1
        print(f"  epoch {ep+1}/{a.epochs} done ({time.perf_counter()-t0:.0f}s)", flush=True)
    if a.save_ckpt is not None:
        a.save_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), a.save_ckpt)
        print(f"[ckpt] saved -> {a.save_ckpt}", flush=True)
    tacc = token_acc(model, tel, a.device, pad)
    b1, n = b1_full_test(model, tok, a.device, pad, eos, batch=a.batch_size)
    print(f"[FULL seed={a.seed}] test token-acc={tacc:.4f}  B1 exact-match (full test n={n})={b1:.4f}  ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
