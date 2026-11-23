"""Central configuration shared by the notebook, the experiment script and the dashboard."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "CardiacPatientData.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"

SEED = 42
WINDOW = 6                  # first N readings per patient
N_SPLITS, N_REPEATS = 5, 5

VITALS = ["SBP", "DBP", "HR", "RR", "BT", "SpO2", "GCS"]
STATIC = ["Age", "Gender", "Na", "K", "Cl", "Urea", "Ceratinine",
          "Alcoholic", "Smoke", "FHCD", "TriageScore"]
# Physiologically plausible limits; values outside are set to NaN before aggregation
LIMITS = {"SBP": (40, 300), "DBP": (20, 200), "HR": (20, 250), "RR": (4, 60),
          "BT": (90, 108), "SpO2": (50, 100), "GCS": (3, 15)}
NON_FEATURES = ["ID", "Death", "n_window"]   # n_window leaks the outcome: never a feature

FAMILY = {  # used by the dashboard to colour/group models
    "Logistic Regression": "Baseline", "Decision Tree": "Baseline",
    "Bagging (trees)": "Bagging", "Random Forest": "Bagging", "Extra Trees": "Bagging",
    "AdaBoost": "Boosting", "Gradient Boosting": "Boosting", "XGBoost": "Boosting",
    "LightGBM": "Boosting", "CatBoost": "Boosting",
    "Stacking (RF+XGB+LGBM -> LR)": "Stacking",
    "Hybrid (RF + ANN blend)": "Hybrid", "ANN (MLP)": "Baseline",
}
