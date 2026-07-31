#!/usr/bin/env python3
"""Torch-free resumable segmented ONNX inference.

Everything from scripts/infer_segmented.py's ResumableSegmentedGenerator applies
here, but with zero torch:
  - One segment's weights in memory at a time (P3 strict)
  - State persisted to disk after every segment execution — resume next day
  - No concat accumulation: embedding fills pre-allocated hidden column-by-column;
    attention head outputs saved per-segment then projected via chunked W_o matmul;
    MLP output accumulated in-place; output head argmax is incremental (no full
    logit tensor ever built)
  - All hidden states and intermediate tensors live as .npy files on disk

Imports: numpy, onnxruntime, json, pathlib, gc, ctypes. NO torch.

Usage:
    # First-time generation (runs to completion):
    python scripts/infer_onnx.py \\
        --onnx-dir onnx_export \\
        --prompt "your log line here" \\
        --max-new-tokens 40 \\
        --state-dir /tmp/onnx_state

    # Run only N segment steps then stop (resumable):
    python scripts/infer_onnx.py --onnx-dir onnx_export \\
        --state-dir /tmp/onnx_state --steps 10

    # Continue from saved state:
    python scripts/infer_onnx.py --onnx-dir onnx_export \\
        --state-dir /tmp/onnx_state --resume --steps 50
"""

from __future__ import annotations

import argparse
import csv as _csv_mod
import ctypes
import gc
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import psutil

_PROC = psutil.Process()


def _rss() -> float:
    return _PROC.memory_info().rss / 1024**2


def _trim() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ── Background VRAM poller (nvidia-smi; torch-free) ──────────────────────────

class _VramPoller:
    """Polls nvidia-smi in a background thread; records (elapsed_s, cpu_mb, gpu_mb)."""

    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._peak     = 0.0
        self._stop     = threading.Event()
        self._trace: list[tuple[float, float, float]] = []
        self._t0       = 0.0
        self._thread   = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._t0 = time.time()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def snapshot(self) -> None:
        gpu_mb  = self._query()
        cpu_mb  = _rss()
        elapsed = time.time() - self._t0
        self._trace.append((round(elapsed, 3), round(cpu_mb, 1), round(gpu_mb, 1)))
        self._peak = max(self._peak, gpu_mb)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.snapshot()

    def _query(self) -> float:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, timeout=2.0,
            )
            return float(out.strip().split(b"\n")[0])
        except Exception:
            return 0.0

    def query_once(self) -> float:
        """One-shot VRAM read that does NOT update the recorded peak/trace.

        Used to capture the VRAM baseline (CUDA runtime + weightless session
        graphs) after sessions are built but before any segment runs.
        """
        return self._query()

    @property
    def peak_mb(self) -> float:
        return self._peak

    def write_trace(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            w = _csv_mod.writer(f)
            w.writerow(["elapsed_s", "cpu_mb", "gpu_mb"])
            for row in self._trace:
                w.writerow(row)


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


# ── WInputsRunner: cached weightless sessions ─────────────────────────────────

class WInputsRunner:
    """~N_UNIQUE_SIG cached weightless ONNX sessions; streams one segment's weights.

    preload_weights=True: all segment .npz files are loaded into a CPU RAM dict at
    startup (analog of --storage cpu_ram in infer_segmented.py). During run(), weights
    are served from the dict — no Lustre reads during inference.
    """

    def __init__(self, manifest: dict, provider: str = "cpu",
                 preload_weights: bool = False):
        self.manifest = manifest
        self.providers = (["CUDAExecutionProvider"] if provider == "cuda"
                          else ["CPUExecutionProvider"])
        self._sessions:         dict[str, ort.InferenceSession] = {}
        self._act_inputs:       dict[str, list[str]]             = {}
        self._seg_to_sig:       dict[str, str]                   = {}
        self._seg_weights_npy:  dict[str, dict[str, str]]        = {}
        self._seg_weights_npz:  dict[str, str]                   = {}
        self._weights_cache:    dict[str, dict[str, np.ndarray]] = {}

        for name, info in manifest["segments"].items():
            self._seg_to_sig[name]      = info["signature"]
            self._seg_weights_npy[name] = info.get("weights_npy", {})
            self._seg_weights_npz[name] = info["weights_npz"]

        so = ort.SessionOptions()
        so.enable_mem_pattern = False
        n_threads = int(os.environ.get("OMP_NUM_THREADS", "4") or "4")
        so.intra_op_num_threads = max(1, n_threads)
        so.inter_op_num_threads = 1

        # For CUDAExecutionProvider: keep ONNX bytes in CPU RAM (no Lustre per step),
        # but only ONE compiled CUDA session in VRAM at a time (P3 for sessions).
        # For CPUExecutionProvider: pre-create all sessions (no VRAM cost).
        self._lazy_cuda = (provider == "cuda")
        self._so = so

        if self._lazy_cuda:
            # Load all ONNX model bytes to CPU RAM now (fast session creation later)
            self._sig_bytes: dict[str, bytes] = {}
            for sig, info in manifest["signatures"].items():
                self._sig_bytes[sig] = open(info["representative_weightless_onnx"], "rb").read()
                self._act_inputs[sig] = info["activation_inputs"]
            # One active CUDA session at a time
            self._session_sig: Optional[str] = None
            self._session_obj: Optional[ort.InferenceSession] = None
            # Get active_provider by creating the first session
            first_sig = next(iter(manifest["signatures"]))
            self._session_obj = ort.InferenceSession(
                self._sig_bytes[first_sig], sess_options=so, providers=self.providers,
            )
            self._session_sig = first_sig
            self.active_provider = self._session_obj.get_providers()[0]
        else:
            for sig, info in manifest["signatures"].items():
                sess = ort.InferenceSession(
                    info["representative_weightless_onnx"],
                    sess_options=so, providers=self.providers,
                )
                self._sessions[sig]    = sess
                self._act_inputs[sig]  = info["activation_inputs"]
            self.active_provider = next(iter(self._sessions.values())).get_providers()[0]

        self.unique_sessions  = len(manifest["signatures"])
        self.peak_weights_mb  = 0.0
        self.segments_run     = 0

        if preload_weights:
            print(f"[preload] Loading {len(manifest['segments'])} segment weight files "
                  f"to CPU RAM ...")
            for name in manifest["segments"]:
                npz = np.load(self._seg_weights_npz[name])
                self._weights_cache[name] = {k: npz[k].copy() for k in npz.files}
                npz.close()
            total_mb = sum(
                a.nbytes for ws in self._weights_cache.values() for a in ws.values()
            ) / 1024**2
            print(f"[preload] {total_mb:.1f} MB preloaded to CPU RAM")

    def _get_session(self, sig: str) -> ort.InferenceSession:
        if self._lazy_cuda:
            if self._session_sig != sig:
                # Destroy current CUDA session → its compiled kernels leave VRAM
                del self._session_obj
                gc.collect()
                # Create from CPU RAM bytes (no Lustre I/O)
                self._session_obj = ort.InferenceSession(
                    self._sig_bytes[sig], sess_options=self._so, providers=self.providers,
                )
                self._session_sig = sig
            return self._session_obj
        return self._sessions[sig]

    def run(self, name: str, act_feed: dict) -> np.ndarray:
        sig  = self._seg_to_sig[name]
        sess = self._get_session(sig)

        from_cache = name in self._weights_cache
        if from_cache:
            weights = self._weights_cache[name]
        else:
            npy_map = self._seg_weights_npy.get(name, {})
            if npy_map:
                weights = {k: np.load(p, mmap_mode="r") for k, p in npy_map.items()}
            else:
                npz = np.load(self._seg_weights_npz[name])
                weights = {k: npz[k] for k in npz.files}
                npz.close()

        wmb = sum(a.nbytes for a in weights.values()) / 1024**2
        self.peak_weights_mb = max(self.peak_weights_mb, wmb)

        feed = dict(act_feed)
        feed.update(weights)
        out = sess.run(None, feed)[0]
        out = np.array(out, copy=True)

        if not from_cache:
            del weights
        del feed
        self.segments_run += 1
        if self.segments_run % 16 == 0:
            gc.collect()
        return out


# ── Resumable generator ───────────────────────────────────────────────────────

_STATE_FILE  = "state.json"
_HIDDEN_FILE = "hidden.npy"
_ATTN_IN     = "attn_in.npy"
_MLP_IN      = "mlp_in.npy"
_MLP_ACC     = "mlp_acc.npy"
_LAST_H      = "last_h.npy"      # [1,1,D] final-norm input (last position only)
_HA_FMT      = "ha_{L}_{S}.npy"  # per-attention-segment output
SEQ_CHUNK    = 32


class ResumableOnnxGenerator:
    """Single-segment-at-a-time resumable autoregressive generation.

    State machine (phases for one token's forward pass):
        embed      → attn_norm → attn → attn_proj → mlp_norm → mlp → mlp_residual
        (loop over layers, then:)
        head_norm  → head → sample → (embed again, or done)

    After each segment execution the full generator state (phase, layer, seg,
    generated token ids, running argmax accumulators) is flushed to state_dir.
    The process can be killed and restarted at any phase.
    """

    def __init__(
        self,
        runner:    WInputsRunner,
        glue:      dict[str, np.ndarray],
        cfg:       dict,
        state_dir: Path,
        prompt_ids: list[int],
    ):
        self.runner    = runner
        self.glue      = glue
        self.cfg       = cfg
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self._D      = cfg["d_model"]
        self._n_attn = cfg["attention_segments"]
        self._n_mlp  = cfg["mlp_chunks"]
        self._n_emb  = cfg["n_emb_segs"]
        self._n_head = cfg["n_head_segs"]
        self._L      = cfg["n_layers"]
        self._eps    = cfg["layer_norm_eps"]
        self._prompt_ids = list(prompt_ids)

    # ── state I/O ─────────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        p = self.state_dir / _STATE_FILE
        if p.exists():
            return json.loads(p.read_text())
        return {
            "phase":          "embed",
            "layer":          0,
            "seg":            0,
            "generated_ids":  [],
            "done":           False,
            "head_best_val":  float("-inf"),
            "head_best_idx":  0,
            "head_vocab_off": 0,
        }

    def _save_state(self, st: dict) -> None:
        (self.state_dir / _STATE_FILE).write_text(json.dumps(st))

    def _np(self, fname: str) -> Path:
        return self.state_dir / fname

    # ── helpers ───────────────────────────────────────────────────────────────

    def _current_ids(self, st: dict) -> np.ndarray:
        ids = self._prompt_ids + st["generated_ids"]
        return np.array(ids, dtype=np.int64)[None, :]   # [1, S]

    def _ha_name(self, layer: int, seg: int) -> str:
        return _HA_FMT.format(L=layer, S=seg)

    # ── execute one segment step ──────────────────────────────────────────────

    def execute_one(self) -> dict:
        """Run one phase transition, save state. Returns current state dict."""
        st    = self._load_state()
        phase = st["phase"]
        L     = st["layer"]
        seg   = st["seg"]
        D     = self._D

        # ── embed ──────────────────────────────────────────────────────────────
        if phase == "embed":
            ids      = self._current_ids(st)
            pos_ids  = np.arange(ids.shape[1], dtype=np.int64)[None, :]
            S        = ids.shape[1]

            if seg == 0:
                hidden = np.zeros([1, S, D], dtype=np.float32)
            else:
                hidden = np.load(self._np(_HIDDEN_FILE))

            name = f"layer_-1__embedding__segment_{seg}"
            out  = self.runner.run(name, {"input_ids": ids, "position_ids": pos_ids})
            slice_d = out.shape[-1]
            hidden[:, :, seg * slice_d : (seg + 1) * slice_d] = out
            del out

            np.save(self._np(_HIDDEN_FILE), hidden)
            del hidden
            _trim()

            st["seg"] = seg + 1
            if st["seg"] >= self._n_emb:
                st["phase"] = "attn_norm"
                st["seg"]   = 0

        # ── attn_norm ──────────────────────────────────────────────────────────
        elif phase == "attn_norm":
            hidden  = np.load(self._np(_HIDDEN_FILE))
            attn_in = _layer_norm(
                hidden,
                self.glue[f"attn_ln.{L}.weight"],
                self.glue[f"attn_ln.{L}.bias"],
                self._eps,
            )
            np.save(self._np(_HIDDEN_FILE), hidden)
            np.save(self._np(_ATTN_IN),     attn_in)
            del hidden, attn_in
            _trim()
            st["phase"] = "attn"
            st["seg"]   = 0

        # ── attn ───────────────────────────────────────────────────────────────
        elif phase == "attn":
            attn_in = np.load(self._np(_ATTN_IN))
            name    = f"layer_{L}__attention__segment_{seg}"
            h_i     = self.runner.run(name, {"attention_input": attn_in})
            del attn_in
            _trim()
            np.save(self._np(self._ha_name(L, seg)), h_i)
            del h_i
            _trim()

            st["seg"] = seg + 1
            if st["seg"] >= self._n_attn:
                self._np(_ATTN_IN).unlink(missing_ok=True)
                st["phase"] = "attn_proj"
                st["seg"]   = 0

        # ── attn_proj (chunked W_o; no concat of head outputs) ────────────────
        elif phase == "attn_proj":
            W    = self.glue[f"proj_w.{L}"]    # [D, D]
            bias = self.glue[f"proj_b.{L}"]    # [D]
            col  = D // self._n_attn

            hidden   = np.load(self._np(_HIDDEN_FILE))
            S        = hidden.shape[1]
            proj_acc = np.zeros([1, S, D], dtype=np.float32)
            proj_acc[0, :, :] += bias

            for si in range(self._n_attn):
                h_i  = np.load(self._np(self._ha_name(L, si)))
                W_sl = W[:, si * col : (si + 1) * col]      # [D, col]
                for t0 in range(0, S, SEQ_CHUNK):
                    t1 = min(t0 + SEQ_CHUNK, S)
                    proj_acc[0, t0:t1, :] += h_i[0, t0:t1, :] @ W_sl.T
                del h_i
                self._np(self._ha_name(L, si)).unlink(missing_ok=True)

            hidden += proj_acc
            del proj_acc
            np.save(self._np(_HIDDEN_FILE), hidden)
            del hidden
            _trim()
            st["phase"] = "mlp_norm"
            st["seg"]   = 0

        # ── mlp_norm ───────────────────────────────────────────────────────────
        elif phase == "mlp_norm":
            hidden = np.load(self._np(_HIDDEN_FILE))
            mlp_in = _layer_norm(
                hidden,
                self.glue[f"mlp_ln.{L}.weight"],
                self.glue[f"mlp_ln.{L}.bias"],
                self._eps,
            )
            np.save(self._np(_HIDDEN_FILE), hidden)
            np.save(self._np(_MLP_IN),      mlp_in)
            del hidden, mlp_in
            _trim()
            st["phase"] = "mlp"
            st["seg"]   = 0

        # ── mlp ────────────────────────────────────────────────────────────────
        elif phase == "mlp":
            mlp_in = np.load(self._np(_MLP_IN))
            name   = f"layer_{L}__mlp__segment_{seg}"
            out    = self.runner.run(name, {"mlp_input": mlp_in})
            del mlp_in
            _trim()

            if seg == 0:
                np.save(self._np(_MLP_ACC), out)
            else:
                acc = np.load(self._np(_MLP_ACC))
                acc += out                                  # in-place: no third tensor
                np.save(self._np(_MLP_ACC), acc)
                del acc
            del out
            _trim()

            st["seg"] = seg + 1
            if st["seg"] >= self._n_mlp:
                self._np(_MLP_IN).unlink(missing_ok=True)
                st["phase"] = "mlp_residual"
                st["seg"]   = 0

        # ── mlp_residual ───────────────────────────────────────────────────────
        elif phase == "mlp_residual":
            hidden  = np.load(self._np(_HIDDEN_FILE))
            mlp_acc = np.load(self._np(_MLP_ACC))
            mlp_acc += self.glue[f"mlp_shared_bias.{L}"]
            hidden  += mlp_acc
            del mlp_acc
            self._np(_MLP_ACC).unlink(missing_ok=True)
            np.save(self._np(_HIDDEN_FILE), hidden)
            del hidden
            _trim()

            next_L = L + 1
            if next_L < self._L:
                st["layer"] = next_L
                st["phase"] = "attn_norm"
            else:
                st["phase"] = "head_norm"
            st["seg"] = 0

        # ── head_norm (last position only → [1,1,D]) ───────────────────────────
        elif phase == "head_norm":
            hidden    = np.load(self._np(_HIDDEN_FILE))
            last_pos  = hidden[:, -1:, :]                  # [1, 1, D]
            last_h    = _layer_norm(
                last_pos,
                self.glue["final_norm.weight"],
                self.glue["final_norm.bias"],
                self._eps,
            )
            del hidden, last_pos
            self._np(_HIDDEN_FILE).unlink(missing_ok=True)
            np.save(self._np(_LAST_H), last_h)
            del last_h
            _trim()
            st["phase"]         = "head"
            st["seg"]           = 0
            st["head_best_val"] = float("-inf")
            st["head_best_idx"] = 0
            st["head_vocab_off"]= 0

        # ── head (incremental argmax; no full logit tensor) ────────────────────
        elif phase == "head":
            last_h = np.load(self._np(_LAST_H))
            name   = f"layer_-1__output_head__segment_{seg}"
            part   = self.runner.run(name, {"hidden_norm": last_h})
            del last_h
            # part: [1, 1, vocab_slice]
            slice_logits = part[0, 0, :]                   # [vocab_slice]
            local_best   = int(np.argmax(slice_logits))
            local_val    = float(slice_logits[local_best])
            vocab_off    = st["head_vocab_off"]
            if local_val > st["head_best_val"]:
                st["head_best_val"] = local_val
                st["head_best_idx"] = vocab_off + local_best
            st["head_vocab_off"] = vocab_off + len(slice_logits)
            del part, slice_logits
            _trim()

            st["seg"] = seg + 1
            if st["seg"] >= self._n_head:
                self._np(_LAST_H).unlink(missing_ok=True)
                st["phase"] = "sample"
                st["seg"]   = 0

        # ── sample ─────────────────────────────────────────────────────────────
        elif phase == "sample":
            next_tok = st["head_best_idx"]
            st["generated_ids"].append(next_tok)
            eos = self.cfg.get("eos_token_id", None)
            max_new = st.get("max_new_tokens", None)
            done = (eos is not None and next_tok == eos)
            if max_new is not None:
                done = done or (len(st["generated_ids"]) >= max_new)
            if done:
                st["done"]  = True
                st["phase"] = "done"
            else:
                st["phase"] = "embed"
                st["layer"] = 0
                st["seg"]   = 0

        elif phase == "done":
            pass

        else:
            raise RuntimeError(f"Unknown phase: {phase!r}")

        self._save_state(st)
        return st

    # ── generate ──────────────────────────────────────────────────────────────

    def generate(self, max_new_tokens: int, steps: Optional[int] = None) -> list[int]:
        """Run the generation loop.

        Args:
            max_new_tokens: stop after this many tokens generated.
            steps:          if set, run at most this many segment steps then
                            return (allows resuming across process restarts).
        Returns:
            list of generated token ids so far.
        """
        st = self._load_state()
        if st.get("done"):
            return st["generated_ids"]
        if "max_new_tokens" not in st:
            st["max_new_tokens"] = max_new_tokens
            self._save_state(st)

        n = 0
        while not st.get("done"):
            st = self.execute_one()
            n += 1
            if steps is not None and n >= steps:
                break

        return st["generated_ids"]


# ── tokenization (torch-free) ─────────────────────────────────────────────────

_REPO_ROOT     = Path(__file__).resolve().parents[1]
_TOKENIZER_DIR = _REPO_ROOT / "gpt2_tokenizer"


def _load_fast_tokenizer():
    """Load tokenizer via `tokenizers` (Rust, no torch dependency)."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from tokenizers import Tokenizer
    tok_json = _TOKENIZER_DIR / "tokenizer.json"
    if not tok_json.exists():
        raise RuntimeError(f"tokenizer.json not found at {tok_json}")
    return Tokenizer.from_file(str(tok_json))


_FAST_TOK = None


def _get_tok():
    global _FAST_TOK
    if _FAST_TOK is None:
        _FAST_TOK = _load_fast_tokenizer()
    return _FAST_TOK


def _tokenize(text: str) -> list[int]:
    enc = _get_tok().encode(text, add_special_tokens=False)
    return list(enc.ids)


def _decode(ids: list[int]) -> str:
    return _get_tok().decode(ids, skip_special_tokens=True)


# ── Batch generation (B examples simultaneously, no state machine) ────────────

def _generate_batch_onnx(
    prompts_ids: list,
    *,
    runner: "WInputsRunner",
    glue: dict,
    cfg: dict,
    max_new_tokens: int,
    pad_id: int = 0,
) -> list:
    """Generate tokens for B prompts simultaneously using [B,S,D] numpy arrays.

    Satisfies P3: only one segment's weights resident at a time.
    Not resumable — use for batch evaluation only.
    Returns list of B lists of generated token ids (excluding prompt).
    """
    B      = len(prompts_ids)
    D      = cfg["d_model"]
    n_emb  = cfg["n_emb_segs"]
    n_attn = cfg["attention_segments"]
    n_mlp  = cfg["mlp_chunks"]
    n_head = cfg["n_head_segs"]
    n_L    = cfg["n_layers"]
    eps    = cfg["layer_norm_eps"]
    eos    = cfg.get("eos_token_id")
    col    = D // n_attn

    prompt_lens = [len(p) for p in prompts_ids]
    max_len = max(prompt_lens)
    # Left-pad so all sequences end at the same (real) last position
    seqs = [[pad_id] * (max_len - len(p)) + list(p) for p in prompts_ids]
    generated = [[] for _ in range(B)]
    done = [False] * B

    for _step in range(max_new_tokens):
        if all(done):
            break

        S = len(seqs[0])
        ids_np  = np.array(seqs,        dtype=np.int64)   # [B, S]
        pos_ids = np.broadcast_to(
            np.arange(S, dtype=np.int64)[None, :], (B, S)
        ).copy()

        # Embedding: accumulate slices into hidden [B, S, D]
        hidden = np.zeros([B, S, D], dtype=np.float32)
        for seg in range(n_emb):
            name     = f"layer_-1__embedding__segment_{seg}"
            out      = runner.run(name, {"input_ids": ids_np, "position_ids": pos_ids})
            slice_d  = out.shape[-1]
            hidden[:, :, seg * slice_d : (seg + 1) * slice_d] = out
            del out
        del ids_np, pos_ids

        # Transformer layers
        for L in range(n_L):
            # Attention norm
            attn_in = _layer_norm(
                hidden,
                glue[f"attn_ln.{L}.weight"],
                glue[f"attn_ln.{L}.bias"],
                eps,
            )

            # Attention segments → list of [B, S, Dn]
            ha_list = []
            for seg in range(n_attn):
                name = f"layer_{L}__attention__segment_{seg}"
                h_i  = runner.run(name, {"attention_input": attn_in})
                ha_list.append(h_i.copy())
                del h_i
            del attn_in

            # Attention output projection (chunked W_o; P3: one segment resident)
            W        = glue[f"proj_w.{L}"]   # [D, D]
            bias_vec = glue[f"proj_b.{L}"]   # [D]
            proj_acc = np.zeros([B, S, D], dtype=np.float32)
            proj_acc += bias_vec              # broadcast [D] → [B, S, D]
            for si, h_i in enumerate(ha_list):
                W_sl = W[:, si * col : (si + 1) * col]   # [D, col]
                for t0 in range(0, S, SEQ_CHUNK):
                    t1 = min(t0 + SEQ_CHUNK, S)
                    proj_acc[:, t0:t1, :] += h_i[:, t0:t1, :] @ W_sl.T
                del h_i
            del ha_list
            hidden += proj_acc
            del proj_acc
            _trim()

            # MLP norm
            mlp_in = _layer_norm(
                hidden,
                glue[f"mlp_ln.{L}.weight"],
                glue[f"mlp_ln.{L}.bias"],
                eps,
            )

            # MLP segments
            mlp_acc = None
            for seg in range(n_mlp):
                name = f"layer_{L}__mlp__segment_{seg}"
                out  = runner.run(name, {"mlp_input": mlp_in})
                if mlp_acc is None:
                    mlp_acc = out.copy()
                else:
                    mlp_acc += out
                del out
            del mlp_in
            mlp_acc += glue[f"mlp_shared_bias.{L}"]
            hidden  += mlp_acc
            del mlp_acc
            _trim()

        # Output head: last position only → [B, 1, D]
        last_pos = hidden[:, -1:, :]
        last_h   = _layer_norm(
            last_pos,
            glue["final_norm.weight"],
            glue["final_norm.bias"],
            eps,
        )
        del hidden, last_pos

        # Head segments: incremental argmax per example
        best_vals = np.full(B, float("-inf"), dtype=np.float32)
        best_idxs = np.zeros(B, dtype=np.int64)
        vocab_off = 0
        for seg in range(n_head):
            name         = f"layer_-1__output_head__segment_{seg}"
            part         = runner.run(name, {"hidden_norm": last_h})  # [B, 1, vocab_slice]
            slice_logits = part[:, 0, :]  # [B, vocab_slice]
            local_best   = slice_logits.argmax(axis=1)  # [B]
            local_vals   = slice_logits[np.arange(B), local_best]
            mask         = local_vals > best_vals
            best_vals[mask] = local_vals[mask]
            best_idxs[mask] = vocab_off + local_best[mask]
            vocab_off   += slice_logits.shape[1]
            del part, slice_logits
        del last_h
        _trim()

        next_toks = best_idxs.tolist()
        for b in range(B):
            if not done[b]:
                tok = next_toks[b]
                if eos is not None and tok == eos:
                    done[b] = True
                    seqs[b].append(pad_id)  # keep lengths uniform even on EOS
                else:
                    generated[b].append(tok)
                    seqs[b].append(tok)
            else:
                seqs[b].append(pad_id)

    return generated


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir",           required=True,
                    help="Directory produced by export_onnx.py")
    ap.add_argument("--state-dir",          default="/tmp/onnx_state",
                    help="Directory for resumable per-step state files")
    ap.add_argument("--prompt",             default=None)
    ap.add_argument("--prompt-ids",         default=None,
                    help="Space-separated token ids (use instead of --prompt)")
    ap.add_argument("--max-new-tokens",     type=int, default=40)
    ap.add_argument("--steps",              type=int, default=None,
                    help="Max segment steps per invocation (omit = run to completion)")
    ap.add_argument("--resume",             action="store_true")
    ap.add_argument("--provider",           default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--preload-weights",    action="store_true",
                    help="Preload all segment .npz weights to CPU RAM at startup "
                         "(analog of --storage cpu_ram in infer_segmented.py). "
                         "Recommended for GPU inference (T06): CPU RAM is the "
                         "backing store, VRAM holds one segment at a time.")
    # eval mode
    ap.add_argument("--evaluate-test",      action="store_true",
                    help="Evaluate on test.csv (50 examples by default)")
    ap.add_argument("--num-test-examples",  type=int, default=50)
    ap.add_argument("--eval-batch-size",    type=int, default=1,
                    help="Examples processed simultaneously in --evaluate-test "
                         "(batch inference via [B,S,D] numpy arrays). Default=1.")
    ap.add_argument("--data-dir",           default="log_lines/generative_splits",
                    help="Directory containing test.csv")
    # outputs
    ap.add_argument("--metrics-out",        default=None,
                    help="(legacy) single-example metrics JSON path")
    ap.add_argument("--metrics-output",     default=None,
                    help="Metrics JSON path (run_summary format, for both modes)")
    ap.add_argument("--trace-output",       default=None,
                    help="CSV trace path (elapsed_s, cpu_mb, gpu_mb)")
    args = ap.parse_args()

    onnx_dir  = Path(args.onnx_dir)
    state_dir = Path(args.state_dir)

    t0_wall = time.time()
    b0      = _rss()

    # ── load artifacts (torch-free) ───────────────────────────────────────────
    cfg      = json.loads((onnx_dir / "config.json").read_text())
    manifest = json.loads((onnx_dir / "manifest.json").read_text())
    npz      = np.load(str(onnx_dir / "glue.npz"))
    glue     = {k: npz[k].astype(np.float32) for k in npz.files}
    npz.close()

    b1 = _rss()
    print(f"[01] glue + config + manifest loaded  : {b1:.1f} MB  (+{b1-b0:.1f})")

    runner = WInputsRunner(manifest, args.provider,
                           preload_weights=args.preload_weights)
    b2 = _rss()
    print(f"[02] {runner.unique_sessions} cached weightless sessions : {b2:.1f} MB  "
          f"(+{b2-b1:.1f})  provider={runner.active_provider}")
    if args.preload_weights:
        print(f"     weights cache: {len(runner._weights_cache)} segments in CPU RAM")

    # ── VRAM baseline: after sessions compiled + weights preloaded, before inference
    # Analog of b0 on CPU path. For GPU: CUDA runtime + ORT session graphs reside
    # in VRAM from here; segment weights are streamed per step from CPU RAM → VRAM.
    vram_baseline = _VramPoller()._query() if args.provider == "cuda" else 0.0
    if args.provider == "cuda":
        print(f"[02b] VRAM baseline (sessions ready, pre-inference): {vram_baseline:.0f} MB")

    # ── start background VRAM poller ──────────────────────────────────────────
    poller = _VramPoller(interval=0.5)
    poller.start()
    peak_rss = b2

    metrics_path = args.metrics_output or args.metrics_out

    # ══════════════════════════════════════════════════════════════════════════
    # MODE A — evaluate-test: loop over test.csv
    # ══════════════════════════════════════════════════════════════════════════
    if args.evaluate_test:
        test_csv = Path(args.data_dir) / "test.csv"
        rows = []
        with open(test_csv) as f:
            for i, row in enumerate(_csv_mod.DictReader(f)):
                if i >= args.num_test_examples:
                    break
                rows.append(row)

        eval_bs = max(1, args.eval_batch_size)
        print(f"[03] evaluate-test: {len(rows)} examples from {test_csv}  "
              f"(batch_size={eval_bs})")
        print("-" * 70)

        n_correct = 0

        if eval_bs > 1:
            # ── Batch mode: B examples simultaneously ──────────────────────────
            for b_start in range(0, len(rows), eval_bs):
                batch = rows[b_start : b_start + eval_bs]
                prompts_ids_b = [_tokenize(r["log_line"]) for r in batch]
                labels_b      = [r.get("label", "").strip() for r in batch]

                generated_b = _generate_batch_onnx(
                    prompts_ids_b,
                    runner=runner, glue=glue, cfg=cfg,
                    max_new_tokens=args.max_new_tokens,
                )
                peak_rss = max(peak_rss, _rss())

                for pids, gen_ids, label, row in zip(
                    prompts_ids_b, generated_b, labels_b, batch
                ):
                    pred    = _decode(gen_ids).strip()
                    correct = pred == label
                    n_correct += correct
                    mark    = "✓" if correct else "✗"
                    print(f"[{mark}] LOG : {row['log_line'][:60]}...")
                    print(f"     TRUE: {label}")
                    print(f"     PRED: {pred}")

        else:
            # ── Sequential mode: one example at a time, fully in-memory ──────────
            # Uses _generate_batch_onnx with B=1 — hidden state stays in CPU RAM,
            # no Lustre I/O per segment step. Mirrors T05's in_memory=True behaviour.
            for i, row in enumerate(rows):
                prompt_ids = _tokenize(row["log_line"])
                label      = row.get("label", "").strip()

                generated_ids = _generate_batch_onnx(
                    [prompt_ids],
                    runner=runner, glue=glue, cfg=cfg,
                    max_new_tokens=args.max_new_tokens,
                )[0]
                peak_rss = max(peak_rss, _rss())

                pred    = _decode(generated_ids).strip()
                correct = pred == label
                n_correct += correct
                mark    = "✓" if correct else "✗"
                print(f"[{mark}] LOG : {row['log_line'][:60]}...")
                print(f"     TRUE: {label}")
                print(f"     PRED: {pred}")

        poller.stop()
        elapsed  = time.time() - t0_wall
        accuracy = n_correct / len(rows) if rows else 0.0

        print()
        print("-" * 70)
        print(f"Exact-match accuracy: {n_correct}/{len(rows)} = {accuracy*100:.1f}%")
        print()
        print("--- Memory stats (inference) ---")
        print(f"  Peak CPU RAM (RSS)          : {peak_rss:.1f} MB")
        print(f"  Peak CPU net (- bare base)  : {peak_rss - b0:.1f} MB")
        if args.provider == "cuda":
            vram_peak = poller.peak_mb
            vram_net  = vram_peak - vram_baseline
            print(f"  VRAM baseline (pre-inference) : {vram_baseline:.0f} MB")
            print(f"  Peak GPU VRAM (nvidia-smi)    : {vram_peak:.0f} MB")
            print(f"  Peak GPU VRAM net (- baseline): {vram_net:.0f} MB  ← single-segment working set")
        else:
            vram_peak = None
            vram_net  = None
            print(f"  Peak GPU VRAM              : N/A (cpu provider)")
        print(f"  Segments run               : {runner.segments_run}")
        print(f"  Peak segment weights       : {runner.peak_weights_mb:.2f} MB")

        if args.trace_output:
            poller.write_trace(args.trace_output)
            print(f"  Trace → {args.trace_output}")

        if metrics_path:
            summary = {
                "peak_cpu_ram_mb":        round(peak_rss, 2),
                "peak_cpu_net_mb":        round(peak_rss - b0, 2),
                "peak_gpu_vram_mb":       round(vram_peak, 1) if vram_peak is not None else None,
                "vram_baseline_mb":       round(vram_baseline, 1) if args.provider == "cuda" else None,
                "peak_gpu_vram_net_mb":   round(vram_net, 1) if vram_net is not None else None,
                "accuracy":               round(accuracy, 4),
                "n_correct":              n_correct,
                "n_examples":             len(rows),
                "elapsed_s":              round(elapsed, 2),
                "segments_run":           runner.segments_run,
                "peak_weights_mb":        round(runner.peak_weights_mb, 2),
                "unique_sessions":        runner.unique_sessions,
                "provider":               runner.active_provider,
            }
            Path(metrics_path).write_text(json.dumps({"run_summary": summary}, indent=2))
            print(f"  Metrics → {metrics_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # MODE B — single prompt (original behaviour, unchanged)
    # ══════════════════════════════════════════════════════════════════════════
    else:
        if args.resume:
            state_dir.mkdir(parents=True, exist_ok=True)
            st = json.loads((state_dir / _STATE_FILE).read_text())
            prompt_ids = st.get("prompt_ids", [])
            if not prompt_ids:
                raise RuntimeError("--resume but no prompt_ids in saved state")
        else:
            if args.prompt_ids:
                prompt_ids = [int(x) for x in args.prompt_ids.split()]
            elif args.prompt:
                prompt_ids = _tokenize(args.prompt)
            else:
                raise SystemExit("Provide --prompt or --prompt-ids (or --resume)")

        print(f"[03] prompt: {len(prompt_ids)} tokens  max_new={args.max_new_tokens}")

        gen = ResumableOnnxGenerator(
            runner=runner, glue=glue, cfg=cfg,
            state_dir=state_dir, prompt_ids=prompt_ids,
        )
        if not args.resume:
            st = gen._load_state()
            st["prompt_ids"]     = prompt_ids
            st["max_new_tokens"] = args.max_new_tokens
            gen._save_state(st)

        generated = gen.generate(max_new_tokens=args.max_new_tokens, steps=args.steps)
        peak_rss  = max(peak_rss, _rss())

        poller.stop()
        final_st = gen._load_state()
        done     = final_st.get("done", False)
        elapsed  = time.time() - t0_wall

        print()
        print("=" * 60)
        print("  ONNX segmented inference (torch-free)")
        print("=" * 60)
        print(f"  prompt tokens      : {len(prompt_ids)}")
        print(f"  generated tokens   : {len(generated)}")
        print(f"  done               : {done}")
        print(f"  phase              : {final_st['phase']}")
        print(f"  layer              : {final_st['layer']}")
        print(f"  peak RSS           : {peak_rss:.1f} MB")
        print(f"  net RSS (- bare)   : {peak_rss - b0:.1f} MB")
        print(f"  peak GPU VRAM      : {poller.peak_mb:.0f} MB"
              if args.provider == "cuda" else
              "  peak GPU VRAM      : N/A (cpu provider)")
        print(f"  elapsed            : {elapsed:.1f} s")
        print(f"  segments run       : {runner.segments_run}")
        print(f"  peak segment wts   : {runner.peak_weights_mb:.2f} MB")
        print(f"  unique sessions    : {runner.unique_sessions}")

        if generated:
            try:
                text = _decode(prompt_ids + generated)
                print(f"\n  Generated text:\n  {text}")
            except Exception:
                print(f"\n  Generated ids: {generated}")

        if args.trace_output:
            poller.write_trace(args.trace_output)
            print(f"\n  trace → {args.trace_output}")

        if metrics_path:
            summary = {
                "peak_cpu_ram_mb":   round(peak_rss, 2),
                "peak_cpu_net_mb":   round(peak_rss - b0, 2),
                "peak_gpu_vram_mb":  round(poller.peak_mb, 1) if args.provider == "cuda" else None,
                "prompt_tokens":     len(prompt_ids),
                "generated_tokens":  len(generated),
                "done":              done,
                "elapsed_s":         round(elapsed, 2),
                "segments_run":      runner.segments_run,
                "peak_weights_mb":   round(runner.peak_weights_mb, 2),
                "unique_sessions":   runner.unique_sessions,
                "provider":          runner.active_provider,
            }
            out = {"run_summary": summary}
            # legacy key for backward compat
            out.update({"peak_rss_mb": summary["peak_cpu_ram_mb"],
                        "net_rss_mb":  summary["peak_cpu_net_mb"],
                        "elapsed_s":   summary["elapsed_s"],
                        "segments_run": summary["segments_run"],
                        "peak_weights_mb": summary["peak_weights_mb"],
                        "unique_sessions": summary["unique_sessions"],
                        "provider": summary["provider"]})
            Path(metrics_path).write_text(json.dumps(out, indent=2))
            print(f"\n  metrics → {metrics_path}")

    assert "torch" not in sys.modules, "torch was imported — P3 violated!"
    print("\n  [OK] torch is NOT in sys.modules — inference is fully torch-free.")


if __name__ == "__main__":
    main()
