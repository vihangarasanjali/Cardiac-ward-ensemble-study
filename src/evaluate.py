"""Repeated stratified CV, summary tables and the Nadeau-Bengio corrected t-test."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from .config import N_REPEATS, N_SPLITS, SEED

SCORING = {
    "AUC": "roc_auc", "PR_AUC": "average_precision",
    "Recall": make_scorer(recall_score, zero_division=0),
    "Precision": make_scorer(precision_score, zero_division=0),
    "F1": make_scorer(f1_score, zero_division=0),
    "BalAcc": "balanced_accuracy", "Brier": "neg_brier_score",
}
METRICS = list(SCORING)


def run_cv(models: dict, X, y, n_splits=N_SPLITS, n_repeats=N_REPEATS, verbose=True) -> pd.DataFrame:
    """Return a long DataFrame: one row per (Model, fold) with every metric. Brier is positive (lower = better)."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    frames = []
    for name, model in models.items():
        r = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=-1, error_score="raise")
        d = {m: r[f"test_{m}"] * (-1 if m == "Brier" else 1) for m in METRICS}
        d["Model"] = name; d["fold"] = np.arange(len(d["AUC"]))
        frames.append(pd.DataFrame(d))
        if verbose:
            print(f"{name:32s} AUC {d['AUC'].mean():.3f}  Recall {d['Recall'].mean():.3f}", flush=True)
    return pd.concat(frames, ignore_index=True)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and std per model, sorted by mean AUC."""
    g = folds.groupby("Model")[METRICS]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
    return out.sort_values("AUC_mean", ascending=False).reset_index()


def corrected_ttest(a, b, n_train, n_test):
    d = np.asarray(a) - np.asarray(b); k = len(d); var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=k - 1))


REFERENCES = ("Random Forest", "Hybrid (RF + ANN blend)")


def significance_table(folds: pd.DataFrame, n_patients: int, refs=REFERENCES, n_splits=N_SPLITS) -> pd.DataFrame:
    """Corrected paired t-tests of each reference model against every other model."""
    n_test = n_patients // n_splits; n_train = n_patients - n_test
    rows = []
    for ref in refs:
        if ref not in set(folds.Model):
            continue
        for other in folds.Model.unique():
            if other == ref:
                continue
            for m in ["AUC", "PR_AUC", "Recall", "F1"]:
                a = folds[folds.Model == ref].sort_values("fold")[m].values
                b = folds[folds.Model == other].sort_values("fold")[m].values
                t, p = corrected_ttest(a, b, n_train, n_test)
                rows.append({"reference": ref, "other": other, "metric": m,
                             "mean_diff": float(a.mean() - b.mean()), "t": t, "p": p})
    return pd.DataFrame(rows)
