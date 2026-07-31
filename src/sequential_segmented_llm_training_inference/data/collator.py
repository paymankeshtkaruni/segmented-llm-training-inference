from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class LogGenerationCollator:
    """
    Generative causal-LM collator.

    The batch items are raw text dictionaries:
        {"data": ..., "label": ...}

    The collator builds:
        prompt = prompt_template.format(data=data)
        target = target_prefix + label

    Loss is computed only on the target label tokens. Prompt tokens and padding
    tokens are masked with -100 in labels.
    """

    tokenizer: Any
    max_length: int = 256
    prompt_template: str = "Log line: {data}\nLabel description:"
    target_prefix: str = " "
    add_bos: bool = True
    add_eos: bool = True

    def __post_init__(self) -> None:
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have pad_token_id.")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor | List[str]]:
        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        attention_mask_list: List[List[int]] = []
        data_texts: List[str] = []
        label_texts: List[str] = []

        for item in batch:
            data_text = str(item["data"]).strip()
            label_text = str(item["label"]).strip()

            prompt = self.prompt_template.format(data=data_text)
            target = f"{self.target_prefix}{label_text}"

            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)

            if self.add_bos and self.tokenizer.bos_token_id is not None:
                full_ids = [self.tokenizer.bos_token_id] + prompt_ids + target_ids
                prompt_len = 1 + len(prompt_ids)
            else:
                full_ids = prompt_ids + target_ids
                prompt_len = len(prompt_ids)

            if self.add_eos and self.tokenizer.eos_token_id is not None:
                full_ids = full_ids + [self.tokenizer.eos_token_id]

            full_ids = full_ids[: self.max_length]

            labels = full_ids.copy()
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100

            attention_mask = [1] * len(full_ids)

            input_ids_list.append(full_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)
            data_texts.append(data_text)
            label_texts.append(label_text)

        max_seq_len = max(len(x) for x in input_ids_list)
        pad_id = self.tokenizer.pad_token_id

        padded_input_ids: List[List[int]] = []
        padded_labels: List[List[int]] = []
        padded_attention_masks: List[List[int]] = []

        for input_ids, labels, attention_mask in zip(
            input_ids_list,
            labels_list,
            attention_mask_list,
        ):
            pad_len = max_seq_len - len(input_ids)
            padded_input_ids.append(input_ids + [pad_id] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)
            padded_attention_masks.append(attention_mask + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "data_texts": data_texts,
            "label_texts": label_texts,
        }
