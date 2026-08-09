"""Clean freight-rate assessment data before feature engineering/modeling.

Inputs (repository root):
    data/train_test.csv
    data/validation.csv

Outputs:
    data/processed/train_test_clean.csv
    data/processed/validation_clean.csv
    data/processed/cleaning_metadata.json

Cleaning policy:
- Keep every row; never drop final validation loads.
- Strip categorical/text fields and normalize dates to YYYY-MM-DD.
- Correct negative weights with absolute value (treated as sign corruption).
- Impute missing weight using the equipment-specific median learned from
  development data only, with a global development median fallback.
- Impute missing market_index using the development-set median.
- Add data-quality flags so later modeling can retain information about
  corrected/imputed values.
- Do not alter, cap, or delete posted_rate outliers.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median


TEXT_COLS = ["load_id", "pickup", "delivery", "equipment"]


def _read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _to_float(value: str | None):
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def fit_cleaning_params(train_rows):
    """Learn imputation values from development data only."""
    corrected_weights_by_equipment = defaultdict(list)
    all_corrected_weights = []
    market_values = []

    for row in train_rows:
        equipment = (row.get("equipment") or "").strip()

        weight = _to_float(row.get("weight"))
        if weight is not None:
            weight = abs(weight)
            corrected_weights_by_equipment[equipment].append(weight)
            all_corrected_weights.append(weight)

        market = _to_float(row.get("market_index"))
        if market is not None:
            market_values.append(market)

    return {
        "weight_median_global": median(all_corrected_weights),
        "weight_median_by_equipment": {
            equipment: median(values)
            for equipment, values in sorted(corrected_weights_by_equipment.items())
        },
        "market_index_median": median(market_values),
    }


def clean_rows(rows, params, include_target: bool):
    cleaned = []
    counters = {
        "rows": len(rows),
        "negative_weight_corrected": 0,
        "missing_weight_imputed": 0,
        "missing_market_index_imputed": 0,
    }

    for source in rows:
        row = dict(source)

        for column in TEXT_COLS:
            row[column] = (row.get(column) or "").strip()

        row["date"] = (
            datetime.strptime(row["date"].strip(), "%Y-%m-%d")
            .date()
            .isoformat()
        )

        # Weight cleaning
        weight_raw = _to_float(row.get("weight"))
        weight_missing = int(weight_raw is None)
        weight_negative = int(weight_raw is not None and weight_raw < 0)

        if weight_negative:
            counters["negative_weight_corrected"] += 1

        if weight_missing:
            counters["missing_weight_imputed"] += 1
            equipment = row["equipment"]
            weight_clean = params["weight_median_by_equipment"].get(
                equipment, params["weight_median_global"]
            )
        else:
            weight_clean = abs(weight_raw)

        row["weight"] = f"{weight_clean:.6f}".rstrip("0").rstrip(".")

        # market_index cleaning
        market_raw = _to_float(row.get("market_index"))
        market_missing = int(market_raw is None)

        if market_missing:
            counters["missing_market_index_imputed"] += 1
            market_clean = params["market_index_median"]
        else:
            market_clean = market_raw

        row["market_index"] = f"{market_clean:.6f}".rstrip("0").rstrip(".")

        # Normalize remaining numeric fields without changing their values.
        for column in [
            "pickup_lat",
            "pickup_lon",
            "delivery_lat",
            "delivery_lon",
            "distance",
            "quote_signal",
        ]:
            value = _to_float(row.get(column))
            if value is None:
                raise ValueError(
                    f"Unexpected missing value in {column} for {row['load_id']}"
                )
            row[column] = f"{value:.6f}".rstrip("0").rstrip(".")

        if include_target:
            target = _to_float(row.get("posted_rate"))
            if target is None or target <= 0:
                raise ValueError(
                    f"Invalid posted_rate for {row['load_id']}: "
                    f"{row.get('posted_rate')}"
                )
            row["posted_rate"] = f"{target:.6f}".rstrip("0").rstrip(".")

        # Audit flags preserve information about original data quality.
        row["weight_missing_flag"] = str(weight_missing)
        row["weight_negative_flag"] = str(weight_negative)
        row["market_index_missing_flag"] = str(market_missing)

        cleaned.append(row)

    return cleaned, counters


def write_csv(path: Path, rows, include_target: bool):
    fields = [
        "load_id",
        "pickup",
        "delivery",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "equipment",
        "weight",
        "date",
        "market_index",
        "quote_signal",
    ]

    if include_target:
        fields.append("posted_rate")

    fields += [
        "weight_missing_flag",
        "weight_negative_flag",
        "market_index_missing_flag",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    output_dir = data_dir / "processed"

    train_path = data_dir / "train_test.csv"
    validation_path = data_dir / "validation.csv"

    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError(
            "Expected data/train_test.csv and data/validation.csv. "
            "Place the assessment CSV files in the data/ directory first."
        )

    train_rows = _read_csv(train_path)
    validation_rows = _read_csv(validation_path)

    # Fit all imputation parameters on development data only to avoid leakage.
    params = fit_cleaning_params(train_rows)

    train_clean, train_counts = clean_rows(
        train_rows, params, include_target=True
    )
    validation_clean, validation_counts = clean_rows(
        validation_rows, params, include_target=False
    )

    write_csv(
        output_dir / "train_test_clean.csv",
        train_clean,
        include_target=True,
    )
    write_csv(
        output_dir / "validation_clean.csv",
        validation_clean,
        include_target=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cleaning_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "params": params,
                "development": train_counts,
                "validation": validation_counts,
            },
            f,
            indent=2,
        )

    print("Cleaning complete.")
    print(
        json.dumps(
            {
                "params": params,
                "development": train_counts,
                "validation": validation_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
