"""
Segmented trainer — the epoch loop, aligned 1:1 with the full-model baseline.

Same data, collator, objective, AdamW hyperparameters (lr 3e-4, wd 0.1,
betas (0.9,0.95), grad-clip 1.0, warmup 200, cosine, 5 epochs), and the same metrics
(train/val loss, val token-acc reported BOTH global and example-weighted, headline
global). The only difference is *how* the step is executed: one segment at a time via
the segmented engines (forward/backward/optimizer), so peak memory is one slice.

Init: a seeded `ReferenceGPTDecoder` is built once and sliced into the stores
(`populate_from_reference`) so the starting weights are IDENTICAL to a full model —
that is what makes "segmented training == full training" checkable. (For a model too
large to instantiate even once, a streamed per-segment init with the same
distribution would replace this; not needed for the study's models.)

Outputs (mirroring full_model): checkpoints/{best,last}.pt, metrics.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from config import ModelConfig, SegmentationConfig, get_preset
from modules import ReferenceGPTDecoder
from forward_engine import SegmentedForwardEngine, SharedParams, populate_from_reference, populate_from_scratch
from backward_engine import SegmentedBackwardEngine
from optimizer import SegmentwiseAdamW, all_segment_keys
from loader import StrictSegmentLoader
from stores import make_store, default_store_kind
from data import build_tokenizer, GenerationCollator, LazyLogDataset, TRAIN_CSV, VAL_CSV, TEST_CSV


def lr_mult(step: int, warmup: int, total: int, mode: str = "cosine") -> float:
    if total <= 0:
        return 1.0
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if mode == "none":
        return 1.0
    prog = (step - warmup) / max(1, total - warmup)
    if mode == "linear":
        return max(0.0, 1.0 - prog)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))


class SegmentedTrainer:
    def __init__(self, preset_name: str, device: str, out_dir: Path,
                 lr: float = 3e-4, weight_decay: float = 0.1, grad_clip: float = 1.0,
                 warmup: int = 200, scheduler: str = "cosine", seed: int = 42,
                 store_kind: Optional[str] = None, from_scratch: bool = False,
                 seg_override: Optional[SegmentationConfig] = None, tech=None,
                 update_style: str = "after_full",
                 dropout_override: Optional[float] = None):
        torch.manual_seed(seed)
        p = get_preset(preset_name)
        self.m: ModelConfig = p["model"]
        if dropout_override is not None:
            from dataclasses import replace as _replace
            self.m = _replace(self.m, dropout=dropout_override)
        # seg_override lets the ablation vary the E×A×M×H split while keeping the SAME
        # (large) model. None -> the preset's segmentation. Validated against the model.
        self.s: SegmentationConfig = seg_override if seg_override is not None else p["seg"]
        if seg_override is not None:
            self.s.validate_against_model(self.m)
        self.tok = build_tokenizer(p["tokenizer"])
        self.device = device
        self.out_dir = Path(out_dir); (self.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.lr, self.wd, self.grad_clip = lr, weight_decay, grad_clip
        self.warmup, self.scheduler, self.preset_name = warmup, scheduler, preset_name
        self.tech = tech    # memory-reduction technique toggles (None = all ON, unchanged)

        kind = store_kind or default_store_kind(device)            # GPU->cpu_ram, CPU->disk
        root = self.out_dir / "stores"
        self.param_store = make_store(kind, root / "params")
        self.grad_store = make_store(kind, root / "grads")
        self.opt_store = make_store(kind, root / "opt")
        self.records_store = make_store(kind, root / "records")    # off-device layer-input snapshots

        if from_scratch:
            # Cost / from-scratch: build segments ONE AT A TIME (setup peak = one segment).
            # Never materializes the full model -> no full-model "build" spike.
            self.shared: SharedParams = populate_from_scratch(self.m, self.s, self.param_store, seed=seed).to(device)
        else:
            # IDENTICAL init to a full model: build seeded reference, slice, then free it
            # (required only for the bit-exact identity proof; builds the full model once).
            ref = ReferenceGPTDecoder(self.m)
            self.shared = populate_from_reference(ref, self.m, self.s, self.param_store).to(device)
            del ref
        self.loader = StrictSegmentLoader(self.m, self.s, self.param_store, device, tech=tech)
        self.fwd = SegmentedForwardEngine(self.m, self.s, self.loader, self.shared, device)
        self.bwd = SegmentedBackwardEngine(self.fwd, self.grad_store, self.records_store)
        self.opt = SegmentwiseAdamW(self.m, self.s, self.loader, self.shared, self.grad_store,
                                    self.opt_store, device, lr=lr, weight_decay=weight_decay)
        # update style: "after_full" (default; separate optimizer sweep, supports
        # global clip) or "immediate" (segments updated during backward; NO global
        # clip — the engine applies updates the moment each gradient is final).
        self.update_style = update_style
        if update_style == "immediate":
            self.bwd.set_immediate_optimizer(self.opt)
        elif update_style != "after_full":
            raise ValueError(f"unknown update_style {update_style!r}")

    # ---- one optimizer step from a batch (forward-record + backward + AdamW) ----
    def train_step(self, batch, global_step: int, total_steps: int) -> Dict[str, float]:
        self.fwd.train()
        ii = batch["input_ids"].to(self.device); lab = batch["labels"].to(self.device)
        # clear previous grads
        for k in all_segment_keys(self.m, self.s):
            self.grad_store.evict(k)
        # lr must be current BEFORE backward: immediate style updates during it
        mult = lr_mult(global_step, self.warmup, total_steps, self.scheduler)
        self.opt.lr = self.lr * mult
        shared_grads = self.bwd.backward(ii, lab, pad_token_id=self.tok.pad_token_id)
        if self.update_style == "immediate":
            self.opt.step_shared(shared_grads)        # segments already updated in-backward
        else:
            self.opt.step(shared_grads, clip_norm=self.grad_clip)
        return {"loss": self.bwd.last_loss, "correct": self.bwd.last_correct,
                "valid": self.bwd.last_valid, "lr": self.opt.lr}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str, max_steps: Optional[int] = None) -> Dict[str, float]:
        self.fwd.eval()
        tot_loss = 0.0; tot_ex = 0; gc = gv = 0; ex_sum = 0.0; ex_n = 0
        for step, batch in enumerate(loader):
            if max_steps and step >= max_steps:
                break
            ii = batch["input_ids"].to(self.device); lab = batch["labels"].to(self.device)
            hidden = self.fwd.forward_hidden(ii, pad_token_id=self.tok.pad_token_id)
            loss, c, v = self.fwd.chunked_ce(hidden, lab)
            bsz = ii.size(0)
            tot_loss += float(loss) * bsz; tot_ex += bsz; gc += c; gv += v
            if v > 0:
                ex_sum += (c / v) * bsz; ex_n += bsz
        avg = tot_loss / max(1, tot_ex)
        g = gc / gv if gv else float("nan")
        e = ex_sum / ex_n if ex_n else float("nan")
        print(f"  [{split}] loss={avg:.4f} acc(global)={g:.4f} acc(ex)={e:.4f} ({tot_ex} ex)")
        return {"loss": avg, "acc_global": g, "acc_example": e}

    def save_checkpoint(self, tag: str, meta: dict) -> None:
        # gather the full segmented state (params store + shared + opt) for export/audit
        params = {f"{k.layer_id}|{k.kind}|{k.seg}": self.param_store.get(k)
                  for k in all_segment_keys(self.m, self.s)}
        torch.save({"params": params, "shared": self.shared.state_dict(),
                    "model_config": self.m.to_dict(), "seg_config": self.s.to_dict(),
                    "preset": self.preset_name, "meta": meta},
                   self.out_dir / "checkpoints" / f"{tag}.pt")

    def fit(self, epochs: int, batch_size: int, workers: int = 0, log_every: int = 50,
            max_train_steps=None, max_val_steps=None, max_rows=None):
        coll = GenerationCollator(self.tok, self.m.max_seq_len)
        tr = LazyLogDataset(TRAIN_CSV, max_rows=max_rows)
        va = LazyLogDataset(VAL_CSV, max_rows=max_rows)
        te = LazyLogDataset(TEST_CSV, max_rows=max_rows)
        trl = DataLoader(tr, batch_size=batch_size, shuffle=True, collate_fn=coll, num_workers=workers)
        val = DataLoader(va, batch_size=batch_size, shuffle=False, collate_fn=coll, num_workers=workers)
        tel = DataLoader(te, batch_size=batch_size, shuffle=False, collate_fn=coll, num_workers=workers)
        spe = min(len(trl), max_train_steps or len(trl))
        total = epochs * spe
        print(f"[seg-train] preset={self.preset_name} dev={self.device} bs={batch_size} "
              f"steps/epoch={spe} total={total} | train={len(tr)} val={len(va)}")

        best = float("inf"); gstep = 0; recs = []; t0 = time.time()
        for ep in range(epochs):
            ep_loss = 0.0; ep_c = ep_v = 0; n = 0; te0 = time.time()
            for step, batch in enumerate(trl):
                if max_train_steps and step >= max_train_steps:
                    break
                st = self.train_step(batch, gstep, total)
                ep_loss += st["loss"]; ep_c += st["correct"]; ep_v += st["valid"]; n += 1; gstep += 1
                if gstep % log_every == 0 or step == 0:
                    print(f"  E{ep+1} step {gstep} | loss={ep_loss/n:.4f} "
                          f"acc={ep_c/max(1,ep_v):.4f} lr={st['lr']:.2e}")
            tr_loss = ep_loss / max(1, n); tr_acc = ep_c / max(1, ep_v)
            print(f"Epoch {ep+1} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} t={time.time()-te0:.1f}s")
            v = self.evaluate(val, "validation", max_val_steps)
            self.save_checkpoint("last", {"epoch": ep + 1})
            if v["loss"] < best:
                best = v["loss"]; self.save_checkpoint("best", {"epoch": ep + 1, "val_loss": best})
                print(f"  *** new best val_loss={best:.4f} ***")
            recs.append({"epoch": ep + 1, "train_loss": tr_loss, "train_acc": tr_acc,
                         "val_loss": v["loss"], "val_acc": v["acc_global"],
                         "val_acc_global": v["acc_global"], "val_acc_example": v["acc_example"]})

        test = self.evaluate(tel, "test", max_val_steps)
        metrics = {"run": "segmented_training", "preset": self.preset_name, "device": self.device,
                   "model_config": self.m.to_dict(), "seg_config": self.s.to_dict(),
                   "batch_size": batch_size, "epochs": epochs, "optimizer": "segmentwise_adamw",
                   "lr": self.lr, "weight_decay": self.wd, "grad_clip": self.grad_clip,
                   "accuracy_aggregation": "headline=global; example also reported",
                   "epochs_detail": recs, "best_val_loss": best,
                   "test_loss": test["loss"], "test_acc": test["acc_global"],
                   "test_acc_global": test["acc_global"], "test_acc_example": test["acc_example"],
                   "total_time_s": time.time() - t0}
        json.dump(metrics, open(self.out_dir / "metrics.json", "w"), indent=2)
        print(f"[done] best_val={best:.4f} test_acc(global)={test['acc_global']:.4f} -> {self.out_dir/'metrics.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="small_8x2x2x8")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-train-steps", type=int, default=3)
    p.add_argument("--max-val-steps", type=int, default=2)
    p.add_argument("--max-rows", type=int, default=64)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--tech-code", default=None,
                   help="11-flag technique code (see techniques.py); default None = all ON")
    p.add_argument("--full-run", action="store_true",
                   help="lift the smoke-test caps: full dataset, no step limits")
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/seg_smoke"))
    a = p.parse_args()
    tech = None
    if a.tech_code is not None:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from techniques import Tech as _Tech
        from dataclasses import fields as _fields
        tech = _Tech(**{f.name: c == "1" for f, c in zip(_fields(_Tech), a.tech_code)})
        print(f"[tech] {a.tech_code} -> {tech}")
    tr = SegmentedTrainer(a.preset, a.device, a.out_dir, warmup=a.warmup,
                          store_kind="cpu_ram", tech=tech)
    if a.full_run:
        tr.fit(a.epochs, a.batch_size, log_every=50)
    else:
        tr.fit(a.epochs, a.batch_size, log_every=1, max_train_steps=a.max_train_steps,
               max_val_steps=a.max_val_steps, max_rows=a.max_rows)
