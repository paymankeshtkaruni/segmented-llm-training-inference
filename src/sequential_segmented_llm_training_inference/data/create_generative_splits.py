from __future__ import annotations

from pathlib import Path

try:
    from constants import DATA_DIR, PROCESSED_DATA_DIR, TOKENIZER_DIR
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data"
    PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
    TOKENIZER_DIR = PROJECT_ROOT / "gpt2_tokenizer"

try:
    from .dataloader_builder import split_csv_folder_by_issue_and_level
except ImportError:
    from dataloader_builder import split_csv_folder_by_issue_and_level


RAW_CSV_DIR = PROCESSED_DATA_DIR
SPLIT_OUTPUT_DIR = DATA_DIR / "generative_splits"


def main() -> None:
    paths = split_csv_folder_by_issue_and_level(
        csv_dir=RAW_CSV_DIR,
        output_dir=SPLIT_OUTPUT_DIR,
        data_column="log_line",
        label_column="label",
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        exclude_rare_issue_level_groups=True,
        balance_train_by_level=True,
        train_level_target_count=None,
    )

    print(f"balanced train.csv: {paths.train_csv}")
    print(f"natural validation.csv: {paths.validation_csv}")
    print(f"natural test.csv: {paths.test_csv}")
    print(f"train before balancing: {paths.train_before_balancing_csv}")
    print(f"excluded rare Issue × level groups: {paths.excluded_rare_csv}")
    print(f"all rows with split/key: {paths.all_with_split_csv}")
    print(f"figure report directory: {paths.report_dir}")


if __name__ == "__main__":
    main()
