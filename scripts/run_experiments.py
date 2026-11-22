"""Precompute everything the dashboard reads. Run from the project root:

    python scripts/run_experiments.py            # full run (5x5 CV)
    python scripts/run_experiments.py --repeats 1  # quick smoke test
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import MODELS_DIR, N_REPEATS, N_SPLITS, RESULTS_DIR, SEED, WINDOW
from src.evaluate import run_cv, significance_table, summarize
from src.features import clean, get_xy, load_raw
from src.models import get_models

SAVE_MODELS = ["Random Forest", "XGBoost", "Stacking (RF+XGB+LGBM -> LR)"]   # one ensemble per family: bagging, boosting, stacking


def main(repeats: int):
    t0 = time.time()
    RESULTS_DIR.mkdir(exist_ok=True); MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. data -> patient-level table -------------------------------------------------
    raw = load_raw()
    pat, X, y, features, removed = get_xy(WINDOW)
    pat.to_csv(RESULTS_DIR / "patient_level.csv", index=False)
    per_pat = raw.groupby("ID").agg(n=("ID", "size"), outcome=("Outcome", "first"))
    report = {
        "raw_rows": int(len(raw)), "patients": int(raw.ID.nunique()),
        "survived": int((per_pat.outcome == 1).sum()), "died": int((per_pat.outcome == 0).sum()),
        "median_readings_survived": float(per_pat[per_pat.outcome == 1].n.median()),
        "median_readings_died": float(per_pat[per_pat.outcome == 0].n.median()),
        "died_with_at_least": {str(h): int((per_pat[per_pat.outcome == 0].n >= h).sum()) for h in (6, 12, 24)},
        "impossible_values_removed": removed, "window": WINDOW, "n_features": len(features),
        "window_length_counts": {str(k): int(v) for k, v in pat.n_window.value_counts().sort_index().items()},
        "raw_missing_per_column": {k: int(v) for k, v in raw.isna().sum().items() if v > 0},
        "readings_per_patient": per_pat.n.tolist(), "readings_outcome": per_pat.outcome.tolist(),
    }
    (RESULTS_DIR / "data_report.json").write_text(json.dumps(report, indent=2))
    print(f"[1/5] data ready: {len(pat)} patients, {len(features)} features, {y.sum()} deaths", flush=True)

    # 2. repeated CV for every model ---------------------------------------------------
    models = get_models(y)
    folds = run_cv(models, X, y, n_splits=N_SPLITS, n_repeats=repeats)
    folds.to_csv(RESULTS_DIR / "fold_level_scores.csv", index=False)
    summarize(folds).to_csv(RESULTS_DIR / "model_comparison_summary.csv", index=False)
    significance_table(folds, len(y)).to_csv(RESULTS_DIR / "significance.csv", index=False)
    print("[2/5] CV done", flush=True)

    # 3. out-of-fold probabilities (for ROC / PR curves) ----------------------------
    cv1 = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    oof = pd.DataFrame({"ID": pat.ID.values, "Death": y})
    for name, model in models.items():
        oof[name] = cross_val_predict(model, X, y, cv=cv1, method="predict_proba", n_jobs=-1)[:, 1]
    oof.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False)
    print("[3/5] out-of-fold predictions saved", flush=True)

    # 4. fit on all data: importances + saved models for the predictor page ------------
    imp = {}
    for name in ["Random Forest", "XGBoost"]:
        m = models[name]; m.fit(X, y)
        imp[name] = m.named_steps["clf"].feature_importances_
    pd.DataFrame(imp, index=features).rename_axis("feature").reset_index().to_csv(
        RESULTS_DIR / "feature_importance.csv", index=False)
    for name in SAVE_MODELS:
        m = models[name]
        if name not in imp:
            m.fit(X, y)
        joblib.dump(m, MODELS_DIR / (name.split(" (")[0].lower().replace(" ", "_") + ".joblib"))
    meta = {"features": features,
            "stats": {c: {"median": float(X[c].median()), "min": float(X[c].min()), "max": float(X[c].max()),
                          "p05": float(X[c].quantile(.05)), "p95": float(X[c].quantile(.95))} for c in features}}
    (MODELS_DIR / "feature_meta.json").write_text(json.dumps(meta, indent=2))
    print("[4/5] models + feature metadata saved", flush=True)

    # 5. run metadata ------------------------------------------------------------------------
    import xgboost, lightgbm, catboost
    (RESULTS_DIR / "run_meta.json").write_text(json.dumps({
        "seconds": round(time.time() - t0, 1), "cv": f"{N_SPLITS}-fold x {repeats} repeats", "seed": SEED,
        "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                     "lightgbm": lightgbm.__version__, "catboost": catboost.__version__}}, indent=2))
    print(f"[5/5] done in {time.time() - t0:.0f}s -> {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=N_REPEATS)
    main(ap.parse_args().repeats)
