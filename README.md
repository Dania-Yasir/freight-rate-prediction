# Freight Rate Prediction

Machine Learning Engineer assessment project for predicting freight rates on future freight loads.

## Project Goal

The objective is to build and validate a regression model using the labeled development data, then generate freight-rate predictions for every load in the final validation dataset.

The final prediction file must contain exactly:

```text
load_id,predicted_rate
```

## Current Progress

Completed:

- Exploratory data analysis
- Data-quality checks
- Data cleaning and preprocessing
- Reproducible cleaning script
- Time-based train/validation split

In progress:

- Feature engineering
- Model comparison and selection

Still to complete:

- Final model training
- Predictions for `data/validation.csv`
- `validation_predictions.csv`
- December prediction chart using the provided scorer
- Final report and short Loom walkthrough

## Dataset

Development dataset:

- 48,000 labeled loads
- Date range: January 1, 2025 to October 31, 2025
- Target: `posted_rate`

Final validation dataset:

- 12,000 unlabeled loads
- Date range: November 1, 2025 to December 31, 2025

## Data Cleaning

The cleaning workflow is implemented in:

```text
src/prepare_data.py
```

Main cleaning decisions:

- No rows are dropped
- Negative weights are treated as sign errors and converted to absolute values
- Missing weights are imputed using equipment-specific medians learned from development data only
- Missing `market_index` values are imputed using the development-set median
- Data-quality flags are retained as model features
- `posted_rate` outliers are kept because there is not enough evidence to classify them as invalid

The cleaned files are written to:

```text
data/processed/train_test_clean.csv
data/processed/validation_clean.csv
data/processed/cleaning_metadata.json
```

More details are documented in:

```text
docs/EDA_CLEANING.md
```

## Validation Strategy

Because the final prediction dataset represents future loads, the development data is split chronologically instead of randomly.

Current local validation split:

- Training: January 1, 2025 to September 30, 2025
- Validation: October 1, 2025 to October 31, 2025

This produces:

- Training set: 43,147 rows
- Local validation set: 4,853 rows

This setup better represents the real task: training on past freight data and predicting rates for future loads.

## Exploratory Findings

Important findings from the development data include:

- `distance` has the strongest simple numeric relationship with `posted_rate`
- Freight rates differ by equipment type
- The final validation period has a lower average `market_index` than the development period
- Some cities and pickup-to-delivery routes in final validation are unseen during development
- `posted_rate` is right-skewed, but high values were retained rather than removed

These findings guide the feature-engineering and model-selection strategy.

## Feature Engineering

Feature engineering is performed after the time-based split and must be applied consistently to:

- Training data
- Local validation data
- Final validation data

Planned feature groups include:

- Core numeric features:
  - distance
  - weight
  - market_index
  - quote_signal
  - pickup/delivery coordinates

- Categorical features:
  - equipment
  - pickup city
  - delivery city

- Route features:
  - pickup-to-delivery route
  - coordinate differences

- Time features:
  - month
  - day of week
  - continuous time index

- Interaction features:
  - distance × market_index
  - distance × quote_signal

- Existing data-quality flags:
  - weight_missing_flag
  - weight_negative_flag
  - market_index_missing_flag

`load_id` is used only as an identifier and is not used as a predictive feature.

## Repository Structure

```text
freight-rate-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── processed/
│       ├── train_test_clean.csv
│       ├── validation_clean.csv
├── docs/
│   └── EDA_CLEANING.md
├── notebooks/
│   └── freight_rate_model.ipynb
├── src/
│   └── prepare_data.py
├── report/
└── scorer_results/
```

Some generated or assessment-provided files may not be committed to the public repository.

## Setup

Clone the repository:

```bash
git clone https://github.com/Dania-Yasir/freight-rate-prediction.git
cd freight-rate-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Reproduce Data Cleaning

Place the original assessment CSV files in:

```text
data/train_test.csv
data/validation.csv
```

Then run:

```bash
python src/prepare_data.py
```

## Modeling Notebook

The main modeling notebook is:

```text
notebooks/freight_rate_model.ipynb
```

Current notebook progress:

1. Setup and imports
2. Exploratory review of the cleaned data
3. Time-based train/validation split
4. Feature engineering section started

The next steps are to implement the feature-engineering pipeline, compare regression models, evaluate them on the October holdout, and generate final predictions.

## Final Submission

The assessment requires:

- An accessible GitHub repository with code, dependencies, and run instructions
- `validation_predictions.csv`
- A PDF or DOCX report explaining the validation strategy and containing the December prediction chart
- A 2–3 minute Loom walkthrough covering:
  - key EDA findings
  - data-quality issues and cleaning decisions
  - model-selection reasoning
  - training and validation strategy
  - the most important parts of the implementation

## Author

Dania Yasir
