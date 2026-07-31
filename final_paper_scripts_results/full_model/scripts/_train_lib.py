"""
Shared A1 accuracy-training logic used by both the CPU and GPU train scripts.

The CPU and GPU entrypoints (full_model_train_cpu.py / _gpu.py) are thin wrappers
that only set the default device and output directory; all logic lives here so the
two variants can never drift.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

import _common as C
from sequential_segmented_llm_training_inference.model.full_model import (
    causal_lm_cross_entropy_loss,
)


def add_train_args(p: argparse.ArgumentParser, default_device: str, default_out: Path) -> None:
    p.add_argument("--device", default=default_device)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=C.TRAIN_BATCH_SIZE)  # 64
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--scheduler", choices=["cosine", "linear", "none"], default="cosine")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    # smoke knobs
    p.add_argument("--max-train-steps", type=int, default=None)
    p.add_argument("--max-val-steps", type=int, default=None)
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--max-val-rows", type=int, default=None)
    p.add_argument("--max-test-rows", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=default_out)


def lr_multiplier(step: int, warmup: int, total: int, mode: str) -> float:
    if total <= 0:
        return 1.0
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if mode == "none":
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    if mode == "linear":
        return max(0.0, 1.0 - progress)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def token_counts(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    """Return (num_correct, num_valid) over the shifted, non-ignored label tokens.

    Shift convention matches the package (training.metrics.token_accuracy,
    losses.shift_logits_and_labels) and causal_lm_cross_entropy_loss:
    logits[:, :-1] predicts labels[:, 1:].
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    valid = int(mask.sum().item())
    if valid == 0:
        return 0, 0
    preds = shift_logits.argmax(dim=-1)
    correct = int(((preds == shift_labels) & mask).sum().item())
    return correct, valid


@torch.no_grad()
def evaluate(model, loader, device, split, pad_token_id, max_steps=None):
    """Teacher-forced eval.

    Loss: example-weighted ( Σ(loss_batch*bsz) / Σbsz ) — matches both the
    segmented evaluate() and the framework Validator.

    Token accuracy: we report BOTH conventions (the repo is inconsistent):
      * acc_global   = Σcorrect / Σvalid            (matches train_segmented.py)
      * acc_example  = Σ(ratio_batch*bsz) / Σbsz    (matches train_normal.py /
                                                     framework Validator/Tester)
    `acc_global` is the HEADLINE/default. Returns a dict.
    """
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = total_valid = 0
    acc_ex_sum = 0.0
    acc_ex_count = 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits, _ = model(input_ids=input_ids, pad_token_id=pad_token_id)
        loss = causal_lm_cross_entropy_loss(logits, labels)
        c, v = token_counts(logits, labels)
        bsz = input_ids.size(0)
        total_loss += loss.item() * bsz
        total_examples += bsz
        total_correct += c
        total_valid += v
        if v > 0:  # mirror train_normal: skip all-masked batches in the mean
            acc_ex_sum += (c / v) * bsz
            acc_ex_count += bsz
    model.train()
    avg_loss = total_loss / total_examples if total_examples else float("inf")
    acc_global = total_correct / total_valid if total_valid else float("nan")
    acc_example = acc_ex_sum / acc_ex_count if acc_ex_count else float("nan")
    print(f"  [{split}] loss={avg_loss:.4f}  acc={acc_global:.4f} (global) "
          f"acc_ex={acc_example:.4f}  ({total_examples} ex, {total_valid} tokens)")
    return {"loss": avg_loss, "acc": acc_global,
            "acc_global": acc_global, "acc_example": acc_example}


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    out_dir = args.out_dir
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1] Tokenizer + model on {device}")
    tokenizer = C.build_tokenizer()
    pad_token_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size
    model = C.build_model(vocab_size).to(device)
    print(f"    params={C.count_params(model)/1e6:.2f}M  vocab={vocab_size}  pad={pad_token_id}")

    print("[2] Datasets")
    collator = C.build_collator(tokenizer)
    train_ds = C.build_dataset(C.TRAIN_CSV, max_rows=args.max_train_rows)
    val_ds = C.build_dataset(C.VAL_CSV, max_rows=args.max_val_rows)
    test_ds = C.build_dataset(C.TEST_CSV, max_rows=args.max_test_rows)
    print(f"    train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collator, num_workers=args.workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collator, num_workers=args.workers, pin_memory=pin)

    optimizer = model.configure_optimizers(
        learning_rate=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )

    steps_per_epoch = min(
        len(train_loader),
        args.max_train_steps if args.max_train_steps is not None else len(train_loader),
    )
    total_steps = args.epochs * steps_per_epoch
    print(f"[3] Training  steps/epoch={steps_per_epoch}  total_steps={total_steps}  "
          f"epochs={args.epochs}  bs={args.batch_size}  lr={args.lr}")

    best_val_loss = float("inf")
    global_step = 0
    epoch_records = []
    total_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_correct = epoch_valid = 0
        num_steps = 0
        model.train()

        for step, batch in enumerate(train_loader):
            if args.max_train_steps is not None and step >= args.max_train_steps:
                break
            mult = lr_multiplier(global_step, args.warmup_steps, total_steps, args.scheduler)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * mult
            current_lr = args.lr * mult

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids=input_ids, pad_token_id=pad_token_id)
            loss = causal_lm_cross_entropy_loss(logits, labels)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

            c, v = token_counts(logits.detach(), labels)
            epoch_loss += loss.item()
            epoch_correct += c
            epoch_valid += v
            num_steps += 1
            global_step += 1

            if global_step % args.log_every == 0 or step == 0:
                avg_l = epoch_loss / num_steps
                avg_a = epoch_correct / epoch_valid if epoch_valid else float("nan")
                print(f"  E{epoch+1} step {global_step:>6} | loss={avg_l:.4f} "
                      f"acc={avg_a:.4f} lr={current_lr:.2e}")

        epoch_time = time.time() - epoch_start
        avg_train_loss = epoch_loss / num_steps if num_steps else float("inf")
        avg_train_acc = epoch_correct / epoch_valid if epoch_valid else float("nan")
        print(f"\nEpoch {epoch+1} — train_loss={avg_train_loss:.4f} "
              f"train_acc={avg_train_acc:.4f} time={epoch_time:.1f}s")

        val = evaluate(model, val_loader, device, "validation",
                       pad_token_id, args.max_val_steps)
        val_loss = val["loss"]

        ckpt = {
            "epoch": epoch, "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss, "config": C.MODEL_CONFIG, "args": vars(args),
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  *** new best val_loss={best_val_loss:.4f} — saved ***")

        epoch_records.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc,  # global-token (headline)
            "val_loss": val_loss,
            "val_acc": val["acc_global"],          # headline = global-token
            "val_acc_global": val["acc_global"],
            "val_acc_example": val["acc_example"],
            "epoch_time_s": epoch_time, "best_val_loss": best_val_loss,
        })
        print("-" * 60)

    print("\n[4] Test on best checkpoint")
    best = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test = evaluate(model, test_loader, device, "test",
                    pad_token_id, args.max_val_steps)

    metrics = {
        "run": "A1_accuracy_training",
        "device": str(device),
        "model_config": C.MODEL_CONFIG,
        "vocab_size": vocab_size,
        "params_million": C.count_params(model) / 1e6,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": "adamw", "lr": args.lr, "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip, "warmup_steps": args.warmup_steps,
        "scheduler": args.scheduler,
        "accuracy_aggregation": "headline=global_token (Sigma correct/Sigma valid); "
                                "example-weighted also reported",
        "epochs_detail": epoch_records,
        "best_val_loss": best_val_loss,
        "best_epoch": int(best["epoch"]) + 1,
        "test_loss": test["loss"],
        "test_acc": test["acc_global"],            # headline = global-token
        "test_acc_global": test["acc_global"],
        "test_acc_example": test["acc_example"],
        "total_time_s": time.time() - total_start,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[done] best_val_loss={best_val_loss:.4f}  "
          f"test_acc(global)={test['acc_global']:.4f}  "
          f"test_acc(example)={test['acc_example']:.4f}  -> {out_dir/'metrics.json'}")
