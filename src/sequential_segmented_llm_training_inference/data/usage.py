from __future__ import annotations

from pathlib import Path

import pandas as pd

from constants import DATA_DIR, TOKENIZER_DIR
from seqquential_segmented_llm_finetuning.scripts.data_modules.dataloader_builder import (
    build_train_validation_test_dataloaders,
    load_local_gpt2_tokenizer,
)


SPLIT_DIR = DATA_DIR / "generative_splits"


def print_split_file_status(split_dir: Path) -> None:
    print("\n" + "=" * 80)
    print("SPLIT FILE STATUS")
    print("=" * 80)

    required_files = [
        "train.csv",
        "train_before_balancing.csv",
        "validation.csv",
        "test.csv",
        "excluded_rare_issue_level.csv",
    ]

    for filename in required_files:
        path = split_dir / filename
        if path.exists():
            df = pd.read_csv(path)
            print(f"{filename:<35} exists | rows = {len(df)}")
        else:
            print(f"{filename:<35} MISSING")


def print_level_distribution(split_dir: Path, label_column: str = "label") -> None:
    print("\n" + "=" * 80)
    print("LEVEL DISTRIBUTION IN SPLIT FILES")
    print("=" * 80)

    files = [
        ("train_before_balancing", split_dir / "train_before_balancing.csv"),
        ("train_after_balancing", split_dir / "train.csv"),
        ("validation", split_dir / "validation.csv"),
        ("test", split_dir / "test.csv"),
    ]

    for name, path in files:
        print("\n" + "-" * 80)
        print(name.upper())
        print("-" * 80)

        if not path.exists():
            print(f"Missing file: {path}")
            continue

        df = pd.read_csv(path)

        if "level" not in df.columns:
            df["level"] = df[label_column].astype(str).str.extract(
                r"level\s*:\s*([^,\n\r]+)",
                expand=False,
            )

        counts = df["level"].fillna("UNKNOWN").value_counts()
        percentages = (counts / counts.sum() * 100).round(2)

        report = pd.DataFrame(
            {
                "count": counts,
                "percent": percentages,
            }
        )

        print(report)


def inspect_loader(loader, tokenizer, name: str, num_batches: int = 2) -> None:
    print("\n" + "=" * 80)
    print(f"{name.upper()} LOADER")
    print("=" * 80)

    print(f"Number of batches: {len(loader)}")
    print(f"Batch size: {loader.batch_size}")
    print(f"Dataset size: {len(loader.dataset)}")

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        print("\n" + "-" * 80)
        print(f"{name.upper()} BATCH {batch_idx}")
        print("-" * 80)

        print("input_ids shape:      ", batch["input_ids"].shape)
        print("attention_mask shape: ", batch["attention_mask"].shape)
        print("labels shape:         ", batch["labels"].shape)

        if "data_texts" in batch:
            print("\nRaw data example:")
            print(batch["data_texts"][0])

        if "label_texts" in batch:
            print("\nRaw label example:")
            print(batch["label_texts"][0])

        decoded_input = tokenizer.decode(
            batch["input_ids"][0],
            skip_special_tokens=False,
        )

        target_token_ids = batch["labels"][0]
        target_token_ids = target_token_ids[target_token_ids != -100]

        decoded_target = tokenizer.decode(
            target_token_ids,
            skip_special_tokens=False,
        )

        print("\nDecoded full model input:")
        print(decoded_input)

        print("\nDecoded training target only:")
        print(decoded_target)


def print_report_paths(split_dir: Path) -> None:
    report_dir = split_dir / "split_report"

    print("\n" + "=" * 80)
    print("REPORT / FIGURE PATHS")
    print("=" * 80)

    expected_reports = [
        "natural_issue_level_distribution_by_split.png",
        "natural_level_distribution_by_split.png",
        "final_loader_level_distribution.png",
        "train_level_distribution_before_after_balancing.png",
        "natural_split_ratio_deviation_by_issue_level.png",
        "excluded_rare_issue_level_groups.png",
        "natural_split_issue_level_distribution.csv",
        "natural_split_level_distribution.csv",
        "final_loader_level_distribution.csv",
        "split_summary.csv",
    ]

    for filename in expected_reports:
        path = report_dir / filename
        status = "exists" if path.exists() else "MISSING"
        print(f"{filename:<60} {status}")


def main() -> None:
    tokenizer = load_local_gpt2_tokenizer(TOKENIZER_DIR)

    print_split_file_status(SPLIT_DIR)
    print_level_distribution(SPLIT_DIR, label_column="label")
    print_report_paths(SPLIT_DIR)

    train_loader, validation_loader, test_loader = build_train_validation_test_dataloaders(
        split_dir=SPLIT_DIR,
        model_dir=TOKENIZER_DIR,
        data_column="log_line",
        label_column="label",
        batch_size=8,
        max_length=256,
    )

    inspect_loader(train_loader, tokenizer, "train", num_batches=2)
    inspect_loader(validation_loader, tokenizer, "validation", num_batches=2)
    inspect_loader(test_loader, tokenizer, "test", num_batches=2)


if __name__ == "__main__":
    main()