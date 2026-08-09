# Freight Rate Prediction Challenge

This repository contains my solution for the Machine Learning Engineer freight rate prediction assessment.

## Objective

The goal is to build, validate, and apply a regression model that predicts freight rates for unseen loads.

The assessment requires:

1. Use `data/train_test.csv` as labeled development data.
2. Decide and document the train/validation split strategy.
3. Explore, clean, engineer features, validate, and model the data.
4. Predict all rows in `data/validation.csv`.
5. Save final validation predictions as `validation_predictions.csv` with exactly:
   - `load_id`
   - `predicted_rate`
6. Fill predictions for every row in `data/december_chart_inputs.csv`.
7. Run the provided scorer to validate outputs and generate the December chart.

## Repository Structure

```text
freight-rate-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── .gitkeep
├── src/
│   └── .gitkeep
├── report/
│   └── .gitkeep
└── scorer_results/
    └── .gitkeep
```

As the solution is completed, the repository will also include:

```text
├── score.py
├── validation_predictions.csv
├── src/
│   └── train_predict.py
├── notebooks/
│   └── freight_rate_model.ipynb
├── report/
│   └── Freight_Rate_Prediction_Report.pdf
└── scorer_results/
    └── candidate_december.png
```

## Setup

Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv .venv
```

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

## Expected Data Files

Place the assessment data files inside `data/`:

```text
data/
├── train_test.csv
├── validation.csv
├── validation_predictions_template.csv
└── december_chart_inputs.csv
```

## Modeling Workflow

The final solution will follow this workflow:

1. Data exploration and quality checks
2. Data cleaning and preprocessing
3. Feature engineering
4. Train/validation split
5. Candidate model comparison
6. Final model selection
7. Holdout validation
8. Retraining on available development data
9. Validation set prediction
10. December input prediction
11. Official scorer validation

## Run the Final Pipeline

After the final modeling script is added:

```bash
python src/train_predict.py
```

Then run the provided scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer should create:

```text
scorer_results/candidate_december.png
```

## Submission Deliverables

The final submission requires:

- Accessible GitHub repository containing solution code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing:
  - validation approach and data split strategy
  - the fixed December prediction chart produced by `score.py`
- 2–3 minute Loom walkthrough covering:
  - key exploratory findings
  - data-quality issues and how they were handled
  - model-selection reasoning
  - training and validation approach
  - the most important parts of the code

## Notes

The assessment data files are not included in this starter repository package. Add the provided assessment files locally before running the project.
