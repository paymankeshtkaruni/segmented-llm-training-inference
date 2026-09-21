from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

try:
    from .collator import LogGenerationCollator
    from .dataset import LogLabelDataset
except ImportError:
    from collator import LogGenerationCollator
    from dataset import LogLabelDataset


_ISSUE_LEVEL_PATTERNS = [
    re.compile(r"Issue\s*:\s*(?P<issue>.*?)[,;\n]+\s*level\s*:\s*(?P<level>.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"issue\s*=\s*(?P<issue>.*?)[,;\n]+\s*level\s*=\s*(?P<level>.*)", re.IGNORECASE | re.DOTALL),
]


@dataclass(frozen=True)
class SplitPaths:
    train_csv: Path
    validation_csv: Path
    test_csv: Path
    train_before_balancing_csv: Path
    excluded_rare_csv: Path
    all_with_split_csv: Path
    report_dir: Path


def load_local_gpt2_tokenizer(model_dir: str | Path):
    model_dir = str(Path(model_dir).expanduser().resolve())

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    tokenizer.padding_side = "right"
    return tokenizer


def _read_all_csv_files(
    csv_dir: str | Path,
    data_column: str = "data",
    label_column: str = "label",
    drop_empty: bool = True,
) -> pd.DataFrame:
    csv_dir = Path(csv_dir)

    if csv_dir.is_file():
        csv_files = [csv_dir]
    elif csv_dir.is_dir():
        csv_files = sorted(csv_dir.glob("*.csv"))
    else:
        raise FileNotFoundError(f"CSV path not found: {csv_dir}")

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")

    frames: List[pd.DataFrame] = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        if data_column not in df.columns or label_column not in df.columns:
            raise ValueError(
                f"CSV must contain '{data_column}' and '{label_column}'. "
                f"File: {csv_file}. Found: {list(df.columns)}"
            )

        df = df.copy()
        df["source_file"] = csv_file.name
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged[data_column] = merged[data_column].fillna("").astype(str).str.strip()
    merged[label_column] = merged[label_column].fillna("").astype(str).str.strip()

    if drop_empty:
        merged = merged[(merged[data_column] != "") & (merged[label_column] != "")].copy()

    if merged.empty:
        raise ValueError("No valid rows after loading CSV files.")

    return merged.reset_index(drop=True)


def _clean_parsed_value(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" ,;.")
    return value or "Unknown"


def extract_issue_and_level_from_label(label: str) -> tuple[str, str]:
    """
    Extract Issue and level from a textual generative label.

    Supported examples:
        Issue: Login Problem, level: Error
        Issue: Login Problem\nlevel: Error
        {"Issue": "Login Problem", "level": "Error"}
    """
    text = str(label).strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(text.replace("'", '"'))
            except json.JSONDecodeError:
                parsed = None

        if isinstance(parsed, dict):
            issue = parsed.get("Issue") or parsed.get("issue") or parsed.get("ISSUE")
            level = parsed.get("level") or parsed.get("Level") or parsed.get("LEVEL")
            if issue is not None and level is not None:
                return _clean_parsed_value(issue), _clean_parsed_value(level)

    for pattern in _ISSUE_LEVEL_PATTERNS:
        match = pattern.search(text)
        if match:
            return (
                _clean_parsed_value(match.group("issue")),
                _clean_parsed_value(match.group("level")),
            )

    raise ValueError(
        "Could not extract Issue and level from label. "
        "Expected something like: 'Issue: <issue>, level: <level>'. "
        f"Bad label: {label!r}"
    )


def _add_issue_level_columns(
    df: pd.DataFrame,
    label_column: str = "label",
    issue_column: str = "Issue",
    level_column: str = "level",
) -> pd.DataFrame:
    df = df.copy()

    if issue_column not in df.columns or level_column not in df.columns:
        parsed = df[label_column].apply(extract_issue_and_level_from_label)
        df[issue_column] = parsed.apply(lambda x: x[0])
        df[level_column] = parsed.apply(lambda x: x[1])
    else:
        df[issue_column] = df[issue_column].fillna("Unknown").astype(str).str.strip()
        df[level_column] = df[level_column].fillna("Unknown").astype(str).str.strip()

    df["issue_level_key"] = df[issue_column] + " :: " + df[level_column]
    return df


def _minimum_group_count_for_three_splits(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> int:
    # At least one row must be available for every non-zero split.
    return int(train_ratio > 0) + int(validation_ratio > 0) + int(test_ratio > 0)


def _stable_sample_positions(n_rows: int, size: int, seed: int) -> np.ndarray:
    """Positions of a seeded sample of `size` rows out of `n_rows`, drawn
    directly from numpy.

    WHY not `DataFrame.sample(..., random_state=seed)`: the split files are a
    published artifact, and `sample`'s seed-to-row-positions mapping is a pandas
    implementation detail, whereas numpy guarantees the `RandomState` stream.
    Drawing the positions here and indexing with `.iloc` takes that one step out
    of pandas' hands. It is behaviour-preserving: `RandomState.permutation` and
    `choice(replace=False)` are exactly what `sample` calls, and the splits come
    out byte-identical on both pandas 2.3.3 and 3.0.6.

    This does NOT by itself make the splits reproducible across pandas majors —
    pandas 3 orders the balanced training rows differently for reasons upstream
    of the sampling — which is why `pyproject.toml` caps pandas below 3.0.
    """
    rng = np.random.RandomState(seed)
    if size >= n_rows:
        return rng.permutation(n_rows)
    return rng.choice(n_rows, size=size, replace=False)


def _split_group_indices(
    group: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split one Issue × level group. This function assumes the group is large
    enough to put at least one sample into every non-zero split.
    """
    positions = _stable_sample_positions(len(group), len(group), seed)
    shuffled_indices = list(group.index[positions])
    n = len(shuffled_indices)

    required = _minimum_group_count_for_three_splits(
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )
    if n < required:
        raise ValueError(
            f"Group has {n} rows but needs at least {required} rows for train/validation/test."
        )

    n_train = max(1, int(math.floor(n * train_ratio))) if train_ratio > 0 else 0
    n_validation = max(1, int(math.floor(n * validation_ratio))) if validation_ratio > 0 else 0
    n_test = max(1, n - n_train - n_validation) if test_ratio > 0 else 0

    # If floor allocation exceeded n because all splits were forced to at least 1,
    # reduce the largest split until the sum is valid.
    while n_train + n_validation + n_test > n:
        counts = {"train": n_train, "validation": n_validation, "test": n_test}
        largest = max(counts, key=counts.get)
        if largest == "train" and n_train > 1:
            n_train -= 1
        elif largest == "validation" and n_validation > 1:
            n_validation -= 1
        elif largest == "test" and n_test > 1:
            n_test -= 1
        else:
            break

    # If there are leftover rows, add them to train.
    while n_train + n_validation + n_test < n:
        n_train += 1

    train_idx = shuffled_indices[:n_train]
    validation_idx = shuffled_indices[n_train : n_train + n_validation]
    test_idx = shuffled_indices[n_train + n_validation : n_train + n_validation + n_test]

    return train_idx, validation_idx, test_idx


def _balance_training_set_by_level(
    train_df: pd.DataFrame,
    seed: int,
    level_column: str = "level",
    issue_level_column: str = "issue_level_key",
    target_per_level: Optional[int] = None,
) -> pd.DataFrame:
    """
    Balance only the training set by level using undersampling.

    This makes levels such as Error / Warning / Information comparable in the
    training loader. Within each level, sampling is proportional over
    Issue × level groups so the issue mixture inside the level is not destroyed.
    """
    level_counts = train_df[level_column].value_counts()
    if level_counts.empty:
        raise ValueError("Cannot balance an empty training set.")

    if target_per_level is None:
        target_per_level = int(level_counts.min())

    if target_per_level <= 0:
        raise ValueError("target_per_level must be positive.")

    balanced_parts: list[pd.DataFrame] = []

    for level_value, level_df in train_df.groupby(level_column, sort=True):
        level_df = level_df.copy()

        if len(level_df) <= target_per_level:
            balanced_parts.append(level_df)
            continue

        group_counts = level_df[issue_level_column].value_counts()
        raw_alloc = group_counts / group_counts.sum() * target_per_level
        alloc = raw_alloc.apply(math.floor).astype(int)

        # Keep at least one sample from each Issue × level group when possible.
        if target_per_level >= len(alloc):
            alloc[alloc == 0] = 1

        while alloc.sum() > target_per_level:
            reducible = alloc[alloc > 1]
            if reducible.empty:
                break
            key = reducible.idxmax()
            alloc.loc[key] -= 1

        fractional_order = (raw_alloc - raw_alloc.apply(math.floor)).sort_values(ascending=False)
        while alloc.sum() < target_per_level:
            added = False
            for key in fractional_order.index:
                if alloc.loc[key] < group_counts.loc[key]:
                    alloc.loc[key] += 1
                    added = True
                    break
            if not added:
                break

        for key, n_take in alloc.items():
            if n_take <= 0:
                continue
            part = level_df[level_df[issue_level_column] == key]
            positions = _stable_sample_positions(len(part), int(n_take), seed)
            balanced_parts.append(part.iloc[positions])

    balanced = pd.concat(balanced_parts, ignore_index=False)
    positions = _stable_sample_positions(len(balanced), len(balanced), seed)
    balanced = balanced.iloc[positions].reset_index(drop=True)
    balanced["train_balanced"] = True
    return balanced


def split_csv_folder_by_issue_and_level(
    csv_dir: str | Path,
    output_dir: str | Path,
    data_column: str = "data",
    label_column: str = "label",
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    drop_empty: bool = True,
    exclude_rare_issue_level_groups: bool = True,
    balance_train_by_level: bool = True,
    train_level_target_count: Optional[int] = None,
) -> SplitPaths:
    """
    Load CSV files and create generative train/validation/test CSVs.

    Correct behavior:
      1. Extract Issue and level from the textual generative label.
      2. Remove Issue × level groups that are too rare to appear in every split.
         These rows are saved to excluded_rare_issue_level.csv and are not used
         in train/validation/test.
      3. Split the remaining data inside each Issue × level group, so validation
         and test preserve the natural Issue × level distribution.
      4. Balance only the training set by level using undersampling, so the
         training dataloader is not dominated by Information.
      5. Save plot-based reports proving both the natural split and the training
         balancing.

    The training target remains the raw textual label. No classification ids are
    created.
    """
    total_ratio = train_ratio + validation_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(
            f"train_ratio + validation_ratio + test_ratio must be 1.0. Got {total_ratio}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "split_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    df = _read_all_csv_files(
        csv_dir=csv_dir,
        data_column=data_column,
        label_column=label_column,
        drop_empty=drop_empty,
    )
    df = _add_issue_level_columns(df, label_column=label_column)
    df["split"] = "unassigned"

    required_group_count = _minimum_group_count_for_three_splits(
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    group_sizes = df.groupby("issue_level_key").size()
    rare_keys = set(group_sizes[group_sizes < required_group_count].index)

    if exclude_rare_issue_level_groups:
        excluded_rare_df = df[df["issue_level_key"].isin(rare_keys)].copy()
        splittable_df = df[~df["issue_level_key"].isin(rare_keys)].copy()
    else:
        excluded_rare_df = df.iloc[0:0].copy()
        splittable_df = df.copy()

    if splittable_df.empty:
        raise ValueError(
            "No splittable rows remain after excluding rare Issue × level groups. "
            f"Minimum required group count is {required_group_count}."
        )

    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []

    for _, group in splittable_df.groupby("issue_level_key", sort=True):
        tr, va, te = _split_group_indices(
            group=group,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        train_indices.extend(tr)
        validation_indices.extend(va)
        test_indices.extend(te)

    df.loc[train_indices, "split"] = "train_before_balancing"
    df.loc[validation_indices, "split"] = "validation"
    df.loc[test_indices, "split"] = "test"
    if not excluded_rare_df.empty:
        df.loc[excluded_rare_df.index, "split"] = "excluded_rare_issue_level"

    train_before_balancing_df = df[df["split"] == "train_before_balancing"].copy()
    validation_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()
    excluded_rare_df = df[df["split"] == "excluded_rare_issue_level"].copy()

    if balance_train_by_level:
        train_df = _balance_training_set_by_level(
            train_df=train_before_balancing_df,
            seed=seed,
            target_per_level=train_level_target_count,
        )
        train_df["split"] = "train"
    else:
        train_df = train_before_balancing_df.copy()
        train_df["split"] = "train"

    train_csv = output_dir / "train.csv"
    validation_csv = output_dir / "validation.csv"
    test_csv = output_dir / "test.csv"
    train_before_balancing_csv = output_dir / "train_before_balancing.csv"
    excluded_rare_csv = output_dir / "excluded_rare_issue_level.csv"
    all_with_split_csv = output_dir / "all_with_split_and_issue_level_key.csv"

    train_df.to_csv(train_csv, index=False)
    validation_df.to_csv(validation_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    train_before_balancing_df.to_csv(train_before_balancing_csv, index=False)
    excluded_rare_df.to_csv(excluded_rare_csv, index=False)

    export_df = pd.concat(
        [train_df, validation_df, test_df, excluded_rare_df],
        ignore_index=True,
    )
    export_df.to_csv(all_with_split_csv, index=False)

    write_split_report_figures(
        full_df=df,
        train_before_balancing_df=train_before_balancing_df,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        excluded_rare_df=excluded_rare_df,
        report_dir=report_dir,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    return SplitPaths(
        train_csv=train_csv,
        validation_csv=validation_csv,
        test_csv=test_csv,
        train_before_balancing_csv=train_before_balancing_csv,
        excluded_rare_csv=excluded_rare_csv,
        all_with_split_csv=all_with_split_csv,
        report_dir=report_dir,
    )


def _distribution_table(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_columns + ["split", "count"])

    counts = (
        df.groupby(group_columns + ["split"])
        .size()
        .reset_index(name="count")
    )
    pivot = counts.pivot_table(
        index=group_columns,
        columns="split",
        values="count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()

    for split_name in ["train", "train_before_balancing", "validation", "test", "excluded_rare_issue_level"]:
        if split_name not in pivot.columns:
            pivot[split_name] = 0

    split_cols = [c for c in ["train", "train_before_balancing", "validation", "test", "excluded_rare_issue_level"] if c in pivot.columns]
    pivot["total"] = pivot[split_cols].sum(axis=1)
    return pivot


def write_split_report_figures(
    full_df: pd.DataFrame,
    train_before_balancing_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    excluded_rare_df: pd.DataFrame,
    report_dir: str | Path,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    """
    Write CSV tables and figures proving:
      - rare Issue × level groups were excluded,
      - validation/test preserve the natural distribution,
      - training was balanced by level.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    natural_split_df = pd.concat(
        [train_before_balancing_df, validation_df, test_df],
        ignore_index=True,
    )
    final_loader_df = pd.concat(
        [train_df, validation_df, test_df],
        ignore_index=True,
    )

    natural_issue_level = _distribution_table(natural_split_df, ["Issue", "level", "issue_level_key"])
    natural_issue = _distribution_table(natural_split_df, ["Issue"])
    natural_level = _distribution_table(natural_split_df, ["level"])
    final_level = _distribution_table(final_loader_df, ["level"])
    final_issue_level = _distribution_table(final_loader_df, ["Issue", "level", "issue_level_key"])

    natural_issue_level.to_csv(report_dir / "natural_split_issue_level_distribution.csv", index=False)
    natural_issue.to_csv(report_dir / "natural_split_issue_distribution.csv", index=False)
    natural_level.to_csv(report_dir / "natural_split_level_distribution.csv", index=False)
    final_level.to_csv(report_dir / "final_loader_level_distribution.csv", index=False)
    final_issue_level.to_csv(report_dir / "final_loader_issue_level_distribution.csv", index=False)

    summary_rows = []
    for name, part in [
        ("raw_all", full_df),
        ("excluded_rare_issue_level", excluded_rare_df),
        ("train_before_balancing", train_before_balancing_df),
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        summary_rows.append({"split": name, "count": len(part)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(report_dir / "split_summary.csv", index=False)

    _plot_stacked_distribution(
        table=natural_issue_level,
        label_column="issue_level_key",
        value_columns=["train_before_balancing", "validation", "test"],
        title="Natural Issue × level distribution before training balance",
        output_path=report_dir / "natural_issue_level_distribution_by_split.png",
    )

    _plot_stacked_distribution(
        table=natural_level,
        label_column="level",
        value_columns=["train_before_balancing", "validation", "test"],
        title="Natural level distribution before training balance",
        output_path=report_dir / "natural_level_distribution_by_split.png",
    )

    _plot_stacked_distribution(
        table=final_level,
        label_column="level",
        value_columns=["train", "validation", "test"],
        title="Final loader level distribution: balanced train, natural validation/test",
        output_path=report_dir / "final_loader_level_distribution.png",
    )

    _plot_train_level_before_after(
        train_before_balancing_df=train_before_balancing_df,
        train_df=train_df,
        output_path=report_dir / "train_level_distribution_before_after_balancing.png",
    )

    _plot_ratio_deviation(
        issue_level=natural_issue_level,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        output_path=report_dir / "natural_split_ratio_deviation_by_issue_level.png",
    )

    if not excluded_rare_df.empty:
        excluded_dist = excluded_rare_df.groupby(["Issue", "level", "issue_level_key"]).size().reset_index(name="count")
        excluded_dist.to_csv(report_dir / "excluded_rare_issue_level_distribution.csv", index=False)
        _plot_simple_counts(
            table=excluded_dist,
            label_column="issue_level_key",
            count_column="count",
            title="Excluded rare Issue × level groups",
            output_path=report_dir / "excluded_rare_issue_level_groups.png",
        )


def _plot_stacked_distribution(
    table: pd.DataFrame,
    label_column: str,
    value_columns: list[str],
    title: str,
    output_path: Path,
) -> None:
    if table.empty:
        return

    plot_df = table.copy()
    plot_df = plot_df.sort_values("total", ascending=False)
    plot_df = plot_df.set_index(label_column)[value_columns]

    height = max(6, min(30, 0.35 * len(plot_df) + 3))
    ax = plot_df.plot(kind="bar", stacked=True, figsize=(14, height))
    ax.set_title(title)
    ax.set_xlabel(label_column)
    ax.set_ylabel("Number of samples")
    ax.tick_params(axis="x", labelrotation=90)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=200)
    plt.close(ax.figure)


def _plot_simple_counts(
    table: pd.DataFrame,
    label_column: str,
    count_column: str,
    title: str,
    output_path: Path,
) -> None:
    if table.empty:
        return

    plot_df = table.sort_values(count_column, ascending=False).set_index(label_column)[[count_column]]
    height = max(6, min(30, 0.35 * len(plot_df) + 3))
    ax = plot_df.plot(kind="bar", legend=False, figsize=(14, height))
    ax.set_title(title)
    ax.set_xlabel(label_column)
    ax.set_ylabel("Number of samples")
    ax.tick_params(axis="x", labelrotation=90)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=200)
    plt.close(ax.figure)


def _plot_train_level_before_after(
    train_before_balancing_df: pd.DataFrame,
    train_df: pd.DataFrame,
    output_path: Path,
) -> None:
    before = train_before_balancing_df["level"].value_counts().rename("train_before_balancing")
    after = train_df["level"].value_counts().rename("train_balanced")
    plot_df = pd.concat([before, after], axis=1).fillna(0).astype(int)
    plot_df = plot_df.sort_values("train_before_balancing", ascending=False)

    ax = plot_df.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Training level distribution before and after balancing")
    ax.set_xlabel("level")
    ax.set_ylabel("Number of samples")
    ax.tick_params(axis="x", labelrotation=45)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=200)
    plt.close(ax.figure)


def _plot_ratio_deviation(
    issue_level: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    output_path: Path,
) -> None:
    if issue_level.empty:
        return

    table = issue_level.copy()
    table["train_pct"] = table["train_before_balancing"] / table["total"]
    table["validation_pct"] = table["validation"] / table["total"]
    table["test_pct"] = table["test"] / table["total"]

    table["max_abs_deviation"] = pd.concat(
        [
            (table["train_pct"] - train_ratio).abs(),
            (table["validation_pct"] - validation_ratio).abs(),
            (table["test_pct"] - test_ratio).abs(),
        ],
        axis=1,
    ).max(axis=1)

    plot_df = table.sort_values("max_abs_deviation", ascending=False)
    plot_df = plot_df.set_index("issue_level_key")[["max_abs_deviation"]]

    height = max(6, min(30, 0.35 * len(plot_df) + 3))
    ax = plot_df.plot(kind="bar", legend=False, figsize=(14, height))
    ax.set_title("Natural split-ratio deviation per Issue × level group")
    ax.set_xlabel("Issue × level")
    ax.set_ylabel("Maximum absolute deviation from requested ratio")
    ax.tick_params(axis="x", labelrotation=90)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=200)
    plt.close(ax.figure)


def build_dataloader_from_csv(
    csv_path: str | Path,
    model_dir: str | Path,
    data_column: str = "data",
    label_column: str = "label",
    batch_size: int = 8,
    max_length: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    tokenizer: Optional[Any] = None,
) -> DataLoader:
    if tokenizer is None:
        tokenizer = load_local_gpt2_tokenizer(model_dir)

    dataset = LogLabelDataset(
        csv_path=csv_path,
        data_column=data_column,
        label_column=label_column,
    )

    collator = LogGenerationCollator(
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )


def build_train_validation_test_dataloaders(
    split_dir: str | Path,
    model_dir: str | Path,
    data_column: str = "data",
    label_column: str = "label",
    batch_size: int = 8,
    max_length: int = 256,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build generative train/validation/test dataloaders from already split CSVs.

    The train loader reads train.csv, which is balanced by level if the split
    was created with balance_train_by_level=True. Validation and test read the
    natural distribution-preserving files.
    """
    split_dir = Path(split_dir)
    tokenizer = load_local_gpt2_tokenizer(model_dir)

    train_loader = build_dataloader_from_csv(
        csv_path=split_dir / "train.csv",
        model_dir=model_dir,
        data_column=data_column,
        label_column=label_column,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=True,
        num_workers=num_workers,
        tokenizer=tokenizer,
    )

    validation_loader = build_dataloader_from_csv(
        csv_path=split_dir / "validation.csv",
        model_dir=model_dir,
        data_column=data_column,
        label_column=label_column,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
        num_workers=num_workers,
        tokenizer=tokenizer,
    )

    test_loader = build_dataloader_from_csv(
        csv_path=split_dir / "test.csv",
        model_dir=model_dir,
        data_column=data_column,
        label_column=label_column,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
        num_workers=num_workers,
        tokenizer=tokenizer,
    )

    return train_loader, validation_loader, test_loader


def split_and_build_train_validation_test_dataloaders(
    csv_dir: str | Path,
    output_dir: str | Path,
    model_dir: str | Path,
    data_column: str = "data",
    label_column: str = "label",
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    batch_size: int = 8,
    max_length: int = 256,
    num_workers: int = 0,
    exclude_rare_issue_level_groups: bool = True,
    balance_train_by_level: bool = True,
    train_level_target_count: Optional[int] = None,
) -> tuple[DataLoader, DataLoader, DataLoader, SplitPaths]:
    """
    End-to-end function:
        1. Load all CSVs.
        2. Exclude Issue × level groups too rare for train/validation/test.
        3. Split remaining data by Issue × level.
        4. Balance only train by level.
        5. Generate proof figures.
        6. Return train/validation/test generative dataloaders.
    """
    paths = split_csv_folder_by_issue_and_level(
        csv_dir=csv_dir,
        output_dir=output_dir,
        data_column=data_column,
        label_column=label_column,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
        exclude_rare_issue_level_groups=exclude_rare_issue_level_groups,
        balance_train_by_level=balance_train_by_level,
        train_level_target_count=train_level_target_count,
    )

    train_loader, validation_loader, test_loader = build_train_validation_test_dataloaders(
        split_dir=output_dir,
        model_dir=model_dir,
        data_column=data_column,
        label_column=label_column,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
    )

    return train_loader, validation_loader, test_loader, paths
