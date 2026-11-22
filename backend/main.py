"""Read-only API over the precomputed results, plus (optionally) the built React app.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000
"""
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATA_PATH, FAMILY, LIMITS, MODELS_DIR, RESULTS_DIR, STATIC, VITALS, WINDOW  # noqa: E402
from src.features import clean, vitals_features  # noqa: E402

DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="Cardiac arrest ensemble study API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET"], allow_headers=["*"])


def _csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        raise HTTPException(503, f"{name} not found. Run `python scripts/run_experiments.py` first.")
    return pd.read_csv(path)


def _records(df: pd.DataFrame) -> list:
    return json.loads(df.to_json(orient="records"))      # NaN becomes null


def _json(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        raise HTTPException(503, f"{name} not found. Run `python scripts/run_experiments.py` first.")
    return json.loads(path.read_text())


@app.get("/api/health")
def health():
    return {"status": "ok", "results_ready": (RESULTS_DIR / "fold_level_scores.csv").exists()}


@app.get("/api/overview")
def overview():
    summary = _csv("model_comparison_summary.csv")
    summary["Family"] = summary["Model"].map(FAMILY)
    return {"report": _json("data_report.json"), "run": _json("run_meta.json"), "summary": _records(summary)}


@app.get("/api/patients")
def patients():
    return _records(_csv("patient_level.csv"))


@app.get("/api/folds")
def folds():
    df = _csv("fold_level_scores.csv")
    df["Family"] = df["Model"].map(FAMILY)
    return _records(df)


@app.get("/api/significance")
def significance():
    return _records(_csv("significance.csv"))


# ---------------------------------------------------------------------------------------------------
# Risk predictor
# ---------------------------------------------------------------------------------------------------
MODEL_FILES = {
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "Stacking (RF+XGB+LGBM -> LR)": "stacking.joblib",
}


class Reading(BaseModel):
    SBP: Optional[float] = None
    DBP: Optional[float] = None
    HR: Optional[float] = None
    RR: Optional[float] = None
    BT: Optional[float] = None
    SpO2: Optional[float] = None
    GCS: Optional[float] = None


class Admission(BaseModel):
    Age: Optional[float] = None
    Gender: Optional[float] = None
    Na: Optional[float] = None
    K: Optional[float] = None
    Cl: Optional[float] = None
    Urea: Optional[float] = None
    Ceratinine: Optional[float] = None
    Alcoholic: Optional[float] = None
    Smoke: Optional[float] = None
    FHCD: Optional[float] = None
    TriageScore: Optional[float] = None


class PredictIn(BaseModel):
    readings: list[Reading] = Field(min_length=1, max_length=24)
    admission: Admission


@lru_cache(maxsize=1)
def _meta() -> dict:
    path = MODELS_DIR / "feature_meta.json"
    if not path.exists():
        raise HTTPException(503, "Saved models not found. Run `python scripts/run_experiments.py` first.")
    return json.loads(path.read_text())


@lru_cache(maxsize=8)
def _model(name: str):
    path = MODELS_DIR / MODEL_FILES[name]
    if not path.exists():
        raise HTTPException(503, f"{path.name} not found. Run `python scripts/run_experiments.py` first.")
    try:
        return joblib.load(path)
    except Exception as e:                                   # pickles are tied to library versions
        raise HTTPException(503, "Saved models could not be loaded with the libraries installed here. "
                                 "Re-run `python scripts/run_experiments.py` to rebuild them. "
                                 f"({type(e).__name__})")


@lru_cache(maxsize=1)
def _oof() -> pd.DataFrame:
    return _csv("oof_predictions.csv")


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def build_feature_row(readings: list, admission: dict):
    """Same feature code as training: first WINDOW readings -> summaries, plus admission values."""
    feats = _meta()["features"]
    df = pd.DataFrame([r for r in readings[:WINDOW]], columns=VITALS).astype(float)
    ignored = []
    for col, (lo, hi) in LIMITS.items():                     # same physiological limits as cleaning
        bad = (df[col] < lo) | (df[col] > hi)
        for i in df.index[bad]:
            ignored.append({"reading": int(i) + 1, "field": col, "value": float(df.at[i, col]), "allowed": [lo, hi]})
        df.loc[bad, col] = np.nan
    if df.notna().sum().sum() == 0:
        raise HTTPException(422, "Enter at least one valid reading before predicting.")
    row = vitals_features(df)
    for c in STATIC:
        row[c] = admission.get(c)
        row[c] = np.nan if row[c] is None else row[c]
    X = pd.DataFrame([row])[feats]
    return X, ignored


@app.post("/api/predict")
def predict(body: PredictIn):
    X, ignored = build_feature_row([r.model_dump() for r in body.readings], body.admission.model_dump())
    oof = _oof()
    out = []
    for name in MODEL_FILES:
        score = float(_model(name).predict_proba(X)[0, 1])
        cohort = oof[name].values
        out.append({
            "model": name, "family": FAMILY[name], "score": score,
            "percentile": float((cohort < score).mean()),
            "cohort_median_survived": float(np.median(oof.loc[oof.Death == 0, name])),
            "cohort_median_died": float(np.median(oof.loc[oof.Death == 1, name])),
        })
    return {"models": out, "n_missing": int(X.isna().sum().sum()), "n_features": int(X.shape[1]),
            "readings_used": min(len(body.readings), WINDOW), "ignored": ignored}


@app.get("/api/example")
def example(outcome: str = "survived", i: int = 0):
    """A real patient's first readings and admission values, to try the predictor with."""
    if outcome not in ("survived", "died"):
        raise HTTPException(422, "outcome must be 'survived' or 'died'")
    df, _ = clean(pd.read_csv(DATA_PATH))
    target = 0 if outcome == "died" else 1
    ids = sorted(df.loc[df.Outcome == target, "ID"].unique())
    pid = ids[i % len(ids)]
    g = df[df.ID == pid]
    readings = [{k: _num(v) for k, v in r.items()} for r in g.head(WINDOW)[VITALS].to_dict("records")]
    admission = {c: (_num(g[c].dropna().iloc[0]) if g[c].notna().any() else None) for c in STATIC}
    oof = _oof()
    row = oof[oof.ID == pid].iloc[0]
    return {"id": int(pid), "outcome": outcome, "readings": readings, "admission": admission,
            "oof": {m: float(row[m]) for m in MODEL_FILES}}


if DIST.exists():                                          # production: serve the built React app
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
