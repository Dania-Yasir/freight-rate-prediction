# EDA and Data Cleaning

This document summarizes the data-quality findings and cleaning decisions used before feature engineering and modeling.

## Dataset Overview

- Development data: **48,000 loads**, covering **2025-01-01 to 2025-10-31**.
- Final validation data: **12,000 loads**, covering **2025-11-01 to 2025-12-31**.
- Prediction target: `posted_rate` in the development data.
- No duplicate `load_id` values or full duplicate rows were found.

## Data-Quality Findings

### Development data

- Missing `weight`: **300 rows**
- Negative `weight`: **292 rows**
- Missing `market_index`: **374 rows**

### Final validation data

- Missing `weight`: **165 rows**
- Negative `weight`: **145 rows**
- Missing `market_index`: **249 rows**

The magnitude distribution of negative weights was similar to the distribution of normal positive weights. For that reason, negative values were treated as sign corruption rather than meaningful negative cargo weights.

## Cleaning Decisions

1. **No rows were dropped.**  
   This is especially important for final validation because every supplied `load_id` requires a prediction.

2. **Negative weights were converted to absolute values.**

3. **Missing weights were imputed using equipment-specific medians learned only from development data:**
   - Dry Van: **31,444**
   - Flatbed: **31,532.5**
   - Reefer: **31,577**

4. **Missing `market_index` values were imputed using the development-set median:**
   - `market_index` median: **1.0558**

5. **Audit flags were added:**
   - `weight_missing_flag`
   - `weight_negative_flag`
   - `market_index_missing_flag`

6. Text fields, numeric formatting, and date representation were standardized.

7. `posted_rate` outliers were **not** capped or removed.

All imputation parameters are learned from the development dataset only, which avoids using future validation information during preprocessing.

## Exploratory Findings Relevant to Modeling

- `distance` has the strongest simple numeric relationship with `posted_rate`, with Pearson correlation of approximately **0.909**.
- Average freight rates differ by equipment type:
  - Reefer highest
  - Flatbed next
  - Dry Van lowest
- The final validation set represents a future time window, and `market_index` shifts lower:
  - Development mean: approximately **1.083**
  - Final validation mean: approximately **0.927**
- Final validation contains **8 cities not seen in development**.
- **1,447 of 12,000 loads (12.1%)** touch at least one unseen city.
- **736 pickup-to-delivery routes** are unseen in development, covering **1,461 validation loads**.
- `posted_rate` is right-skewed:
  - Median: approximately **$2,031**
  - P99: approximately **$5,973**
  - Maximum: **$25,533**

The high target values were retained rather than deleted because there was not enough evidence to treat them as invalid observations.

## Reproduce the Cleaning Step

Place the original assessment files at:

```text
data/train_test.csv
data/validation.csv
```

Then run:

```bash
python src/prepare_data.py
```

The script creates:

```text
data/processed/train_test_clean.csv
data/processed/validation_clean.csv
data/processed/cleaning_metadata.json
```

The generated cleaned CSV files are intermediate artifacts and do not need to be committed to the public repository.
