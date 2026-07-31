"""
Fixed GPT-style decoder for autoregressive log-line-to-label generation.

Task:
    log_line -> label

Training format expected from dataloader/collator:
    input_ids = source_ids + target_ids
    labels    = [-100] * len(source_ids) + target_ids

Recommended source/target format:
    source_text = f"{log_line}\\nLabel:"
    target_text = f" {label}"

Manual training loop should do:
    input_shifted = input_ids[:, :-1]
    labels_shifted = labels[:, 1:]

    logits, hidden_states = model(
        input_ids=input_shifted,
        pad_token_id=pad_token_id,
    )

    loss = cross_entropy(
        logits.reshape(-1, vocab_size),
        labels_shifted.reshape(-1),
        ignore_index=-100,
    )
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTInputEmbedding(nn.Module):
    """
    Token embedding + learned positional embedding.

    Input:
        input_ids: [batch_size, seq_len]

    Output:
        x: [batch_size, seq_len, d_model]
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(
            0,
            seq_len,
            dtype=torch.long,
            device=device,
        )

        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors
        x = self.dropout(x)

        return x


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    The causal mask makes sure position i can only attend to positions <= i.
    This is what prevents the model from seeing future label tokens.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_projection = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
            diagonal=1,
        )

        self.register_buffer(
            "causal_mask",
            causal_mask.view(1, 1, max_seq_len, max_seq_len),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:
            [batch_size, seq_len, d_model]

        padding_mask:
            Optional bool tensor [batch_size, 1, 1, seq_len]
            True means padding token.
        """
        batch_size, seq_len, d_model = x.shape

        qkv = self.qkv_projection(x)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        key = key.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        value = value.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]

        attention_scores = attention_scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        if padding_mask is not None:
            attention_scores = attention_scores.masked_fill(
                padding_mask,
                float("-inf"),
            )

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        output = attention_weights @ value

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, d_model)

        output = self.output_projection(output)
        output = self.output_dropout(output)

        return output


class GPTMLP(nn.Module):
    """
    Position-wise feed-forward network.

    Shape:
        [batch_size, seq_len, d_model]
        -> [batch_size, seq_len, d_ff]
        -> [batch_size, seq_len, d_model]
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.output_projection(x)
        x = self.dropout(x)

        return x


class GPTBlock(nn.Module):
    """
    One Pre-LayerNorm GPT block:

        x = x + CausalSelfAttention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = GPTMLP(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.attention_norm(x),
            padding_mask=padding_mask,
        )

        x = x + self.mlp(
            self.mlp_norm(x),
        )

        return x


class GPTDecoder(nn.Module):
    """
    GPT-style decoder-only model for log-line-to-label generation.

    Input:
        input_ids: [batch_size, seq_len]

    Output:
        logits:        [batch_size, seq_len, vocab_size]
        hidden_states: [batch_size, seq_len, d_model]
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        tie_weights: bool = False,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        self.embedding = GPTInputEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.blocks = nn.ModuleList(
            [
                GPTBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        self.output_projection = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

        if tie_weights:
            self.output_projection.weight = self.embedding.token_embedding.weight

        self.apply(self._init_weights)

        # GPT-style residual projection scaling.
        # This is optional but useful for training stability in deeper models.
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        """
        Scale residual projection weights similarly to GPT-2/minGPT style.

        The residual projections are:
            attention.output_projection
            mlp.output_projection
        """
        scale = 1.0 / math.sqrt(2 * self.n_layers)

        for block in self.blocks:
            nn.init.normal_(
                block.attention.output_projection.weight,
                mean=0.0,
                std=0.02 * scale,
            )

            nn.init.normal_(
                block.mlp.output_projection.weight,
                mean=0.0,
                std=0.02 * scale,
            )

    @staticmethod
    def build_padding_mask(
        input_ids: torch.Tensor,
        pad_token_id: int,
    ) -> torch.Tensor:
        """
        input_ids:
            [batch_size, seq_len]

        returns:
            [batch_size, 1, 1, seq_len]

        True means padding token.
        """
        return (input_ids == pad_token_id).view(
            input_ids.size(0),
            1,
            1,
            input_ids.size(1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        pad_token_id: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input_ids:
            [batch_size, seq_len]

        pad_token_id:
            optional tokenizer pad token id

        returns:
            logits:
                [batch_size, seq_len, vocab_size]

            hidden_states:
                [batch_size, seq_len, d_model]
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must have shape [batch_size, seq_len], "
                f"got {tuple(input_ids.shape)}"
            )

        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        padding_mask = None
        if pad_token_id is not None:
            padding_mask = self.build_padding_mask(
                input_ids=input_ids,
                pad_token_id=pad_token_id,
            )

        x = self.embedding(input_ids)

        for block in self.blocks:
            x = block(
                x,
                padding_mask=padding_mask,
            )

        hidden_states = self.final_norm(x)
        logits = self.output_projection(hidden_states)

        return logits, hidden_states

    @torch.no_grad()
    def generate_label_from_log_line(
        self,
        tokenizer,
        log_line: str,
        max_new_tokens: int = 32,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        device: Optional[torch.device] = None,
        use_label_prefix: bool = True,
    ) -> str:
        """
        Generate label text from one raw log line.

        If use_label_prefix=True, the generation prefix is:

            f"{log_line}\\nLabel:"

        This must match the collator/training format.
        """
        self.eval()

        if device is None:
            device = next(self.parameters()).device

        if pad_token_id is None:
            pad_token_id = tokenizer.pad_token_id

        if eos_token_id is None:
            eos_token_id = tokenizer.eos_token_id

        if use_label_prefix:
            prefix = f"{log_line}\nLabel:"
        else:
            prefix = log_line

        input_ids = tokenizer.encode(
            prefix,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)

        original_length = input_ids.size(1)
        generated = input_ids

        for _ in range(max_new_tokens):
            if generated.size(1) > self.max_seq_len:
                generated_context = generated[:, -self.max_seq_len:]
            else:
                generated_context = generated

            logits, _ = self.forward(
                input_ids=generated_context,
                pad_token_id=pad_token_id,
            )

            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

            generated = torch.cat(
                [generated, next_token_id],
                dim=1,
            )

            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

        generated_label_ids = generated[0, original_length:]

        generated_label = tokenizer.decode(
            generated_label_ids,
            skip_special_tokens=True,
        )

        return generated_label.strip()

    def configure_optimizers(
        self,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.AdamW:
        """
        GPT-style AdamW parameter grouping.

        Weight decay:
            applied to matrix-like weights

        No weight decay:
            biases and LayerNorm parameters
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optimizer_groups = [
            {
                "params": decay_params,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

        return torch.optim.AdamW(
            optimizer_groups,
            lr=learning_rate,
            betas=betas,
        )


def causal_lm_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Causal LM loss if you send the FULL sequence to the model.

    logits:
        [batch_size, seq_len, vocab_size]

    labels:
        [batch_size, seq_len]

    This function shifts internally:
        logits[:, :-1, :] predicts labels[:, 1:]

    Use this if your training loop does NOT shift before the model.
    """
    vocab_size = logits.size(-1)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )

    return loss


def shifted_cross_entropy_loss(
    logits: torch.Tensor,
    shifted_labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Causal LM loss if your training loop already shifted.

    Your current training style:

        input_shifted = input_ids[:, :-1]
        labels_shifted = labels[:, 1:]

        logits, hidden_states = model(input_ids=input_shifted)

        loss = shifted_cross_entropy_loss(logits, labels_shifted)

    logits:
        [batch_size, seq_len - 1, vocab_size]

    shifted_labels:
        [batch_size, seq_len - 1]
    """
    vocab_size = logits.size(-1)

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        shifted_labels.reshape(-1),
        ignore_index=ignore_index,
    )

    return loss


if __name__ == "__main__":
    # Simple sanity check
    vocab_size = 1000
    d_model = 64
    n_heads = 4
    n_layers = 2
    d_ff = 256
    max_seq_len = 32
    batch_size = 2
    seq_len = 10
    pad_token_id = 0

    model = GPTDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=0.1,
    )

    input_ids = torch.randint(
        low=1,
        high=vocab_size,
        size=(batch_size, seq_len),
    )

    logits, hidden_states = model(
        input_ids=input_ids,
        pad_token_id=pad_token_id,
    )

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert hidden_states.shape == (batch_size, seq_len, d_model)

    labels = input_ids.clone()
    labels[:, :3] = -100

    loss = causal_lm_cross_entropy_loss(
        logits=logits,
        labels=labels,
    )

    assert loss.dim() == 0

    input_shifted = input_ids[:, :-1]
    labels_shifted = labels[:, 1:]

    shifted_logits, _ = model(
        input_ids=input_shifted,
        pad_token_id=pad_token_id,
    )

    shifted_loss = shifted_cross_entropy_loss(
        logits=shifted_logits,
        shifted_labels=labels_shifted,
    )

    assert shifted_loss.dim() == 0

    print("Sanity check passed.")