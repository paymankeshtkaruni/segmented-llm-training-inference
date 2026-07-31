from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "log_lines"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
print("")

def clean_label(label: str) -> str:
    if label is None:
        return ""
    parts = [p.strip() for p in str(label).strip().split("_") if p.strip()]
    return " ".join(p.capitalize() for p in parts)


def clean_severity(severity: str) -> str:
    if severity is None:
        return ""
    return str(severity).strip().capitalize()


def build_label(label: str, severity: str) -> str:
    issue = clean_label(label)
    level = clean_severity(severity)
    return f"Issue: {issue}, level: {level}"


def detect_columns(fieldnames: list[str]) -> tuple[str, str, str]:
    normalized = {name.strip().lower(): name for name in fieldnames}

    log_col: Optional[str] = None
    label_col: Optional[str] = None
    severity_col: Optional[str] = None

    for c in ["log_line", "log", "text", "message"]:
        if c in normalized:
            log_col = normalized[c]
            break

    for c in ["label", "issue", "event"]:
        if c in normalized:
            label_col = normalized[c]
            break

    for c in ["severity", "level"]:
        if c in normalized:
            severity_col = normalized[c]
            break

    if log_col and label_col and severity_col:
        return log_col, label_col, severity_col

    if len(fieldnames) >= 3:
        return fieldnames[0], fieldnames[1], fieldnames[2]

    raise ValueError("Could not detect required columns.")


def convert_csv(input_csv: str | Path, output_csv: str | Path) -> None:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    with input_csv.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)

        if not reader.fieldnames:
            raise ValueError(f"CSV must have headers: {input_csv}")

        log_col, label_col, severity_col = detect_columns(reader.fieldnames)

        with output_csv.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=["log_line", "label"])
            writer.writeheader()

            for row in reader:
                log_line = (row.get(log_col) or "").strip()
                raw_label = (row.get(label_col) or "").strip()
                raw_severity = (row.get(severity_col) or "").strip()

                final_label = build_label(raw_label, raw_severity)

                writer.writerow(
                    {
                        "log_line": log_line,
                        "label": final_label,
                    }
                )


def process_all_csvs(data_dir: str | Path) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {data_dir}")

    output_dir = PROCESSED_DATA_DIR
    output_dir.mkdir(exist_ok=True)

    csv_files = sorted(
        p for p in data_dir.glob("*.csv")
        if p.is_file()
    )

    if not csv_files:
        print(f"No CSV files found in: {data_dir}")
        return

    for csv_file in csv_files:
        output_file = output_dir / csv_file.name
        try:
            convert_csv(csv_file, output_file)
            print(f"Processed: {csv_file.name} -> {output_file}")
        except Exception as e:
            print(f"Failed: {csv_file.name} -> {e}")


if __name__ == "__main__":

    process_all_csvs(DATA_DIR)