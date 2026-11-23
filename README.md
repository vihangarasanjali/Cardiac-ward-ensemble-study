# Ensemble Learning Dashboard: Cardiac Ward Mortality

A comparison of ensemble learning methods (bagging, boosting, stacking and a Random Forest + ANN hybrid) on real clinical data from a Sri Lankan teaching hospital, with a React dashboard to explore the results and score individual patients.

The task: predict in-hospital death for cardiac ward patients from their admission details and first few hourly readings.

> Research prototype built on 112 patients from a single hospital unit. It is not a clinical decision tool.

## Dataset

The Cardiac Patient Bed Head Ticket dataset, collected at the Teaching Hospital Karapitiya, Galle, Sri Lanka (112 patients, August 2018 to February 2020, 19 deaths).

- Dataset: https://zenodo.org/record/7603772 (CC BY 4.0)
- Paper: Rajapaksha, L.T.W.; Vidanagamachchi, S.M.; Gunawardena, S.; Thambawita, V. *An Open-Access Dataset of Hospitalized Cardiac-Arrest Patients: Machine-Learning-Based Predictions Using Clinical Documentation.* BioMedInformatics 2024, 4, 34-49. https://doi.org/10.3390/biomedinformatics4010003

## What the dashboard shows

| Page | Content |
|---|---|
| Summary | Cohort numbers, how the comparison was done, the top models |
| Data overview | Class balance, length of stay by outcome, why the first 6 readings, cleaning, predictor explorer |
| Model comparison | Ranking by any metric, spread across cross-validation folds, significance tests against a chosen reference model |
| Risk predictor | Score one patient with four saved ensemble models, or load a real example |

## Method

- **One row per patient (112 rows).** The raw file has about 5,900 hourly rows. Each patient is summarised by the mean, min, max, last value and slope of seven vital signs over their first 6 readings, plus admission labs and history. This keeps one patient from appearing in both training and test data.
- **Positive class = death** (19 patients). Recall therefore means deaths caught.
- **Cleaning.** Physiologically impossible values (for example GCS above 15) are set to missing before any summary is computed.
- **No synthetic data in evaluation.** Class imbalance is handled with class weights. Missing values are imputed with the median inside each fold.
- **Evaluation.** Repeated stratified 5-fold cross-validation (5 repeats, 25 splits). Metrics: ROC-AUC, PR-AUC, recall, precision, F1, balanced accuracy, Brier score.
- **Model comparison.** Nadeau-Bengio corrected resampled t-test, because cross-validation folds share training data.
- **Stay length is never a feature.** Patients who die have shorter stays, so it would leak the outcome.

### Models

| Family | Models |
|---|---|
| Bagging | Bagged trees, Random Forest, Extra Trees |
| Boosting | AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost |
| Stacking | Random Forest + XGBoost + LightGBM with a logistic regression combiner |
| Hybrid | Random Forest + ANN blend: `p = w * p_RF + (1 - w) * p_ANN`, with `w` chosen from {0.25, 0.5, 0.75} by inner cross-validation inside each training fold |
| Baselines (reference) | Logistic regression, single decision tree, ANN alone |

## Results

Mean ROC-AUC over 25 cross-validation folds from the reference run. Your numbers may differ slightly with different library versions.

| Model | Mean AUC |
|---|---|
| XGBoost | 0.828 |
| LightGBM | 0.815 |
| AdaBoost | 0.813 |
| Stacking (RF + XGB + LGBM) | 0.805 |
| Hybrid (RF + ANN blend) | 0.799 |
| Random Forest | 0.797 |
| CatBoost | 0.788 |
| Bagged trees | 0.783 |
| Extra Trees | 0.779 |
| Gradient Boosting | 0.778 |
| Logistic regression | 0.745 |
| ANN alone | 0.744 |
| Decision tree | 0.620 |

The main finding: ensembles clearly beat a single decision tree, but the leading ensembles cannot be told apart statistically. With only 19 deaths, AUC varies by roughly 0.15 or more between folds. The hybrid performs about the same as Random Forest on AUC, and catches more deaths at the 0.5 cutoff.

## Project structure

```
data/                 raw CardiacPatientData.csv
notebooks/            Ensemble_Cardiac_Study.ipynb, the full analysis
src/                  config, feature building, models, evaluation
scripts/              run_experiments.py, which reproduces everything
results/              CSV/JSON outputs read by the dashboard (results/models/ is git-ignored)
backend/              FastAPI service serving results and predictions
frontend/             React (Vite) dashboard
requirements.txt
```

## Setup and run

Requires Python 3.10 or newer. Node.js is only needed to modify the frontend.

```bash
pip install -r requirements.txt

# 1. Reproduce all results and saved models (about 7 minutes on one core, faster on several)
python scripts/run_experiments.py

# 2. Start the API (serves the built dashboard too)
uvicorn backend.main:app --port 8000
```

Open http://localhost:8000.

Build the frontend once (needed after cloning, since `frontend/dist/` is git-ignored, and after any change to the frontend source):

```bash
cd frontend
npm install --include=dev
npm run build
```

For development with live reload, run `npm run dev` in `frontend/` and open http://localhost:5173 (keep the API running on port 8000). Run the frontend tests with `npm test`.

The saved models in `results/models/` only load with the library versions that created them, so each machine should run step 1.

The notebook finds the CSV in `data/`. Start Jupyter from the project root and open `notebooks/Ensemble_Cardiac_Study.ipynb`. Its outputs go to `notebook_outputs/`.

## Limitations

- 112 patients and 19 deaths from one hospital unit, so results may not generalise and single-fold scores are very noisy.
- The 6-reading window was chosen because most deaths happen early; a sensitivity check with other window lengths is not yet included.
- Hyperparameters are fixed and modest, not tuned.
- Risk scores are not calibrated probabilities. Class weights push scores upward, so use them to compare patients, not as a chance of dying.

## Planned work

- Nested cross-validation for hyperparameter tuning
- Random Forest + ARIMA hybrid on a longer time series

