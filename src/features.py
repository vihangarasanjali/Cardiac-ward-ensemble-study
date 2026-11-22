"""Raw CSV -> cleaned -> patient-level feature table (one row per patient)."""
import numpy as np
import pandas as pd
from .config import DATA_PATH, LIMITS, NON_FEATURES, STATIC, VITALS, WINDOW


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(raw: pd.DataFrame):
    """Set impossible values to NaN. Returns (cleaned_df, {column: n_removed})."""
    df = raw.sort_values("ID", kind="stable").copy()   # row order within patient = time order
    removed = {}
    for col, (lo, hi) in LIMITS.items():
        m = (df[col] < lo) | (df[col] > hi)
        removed[col] = int(m.sum())
        df.loc[m, col] = np.nan
    return df, removed


def vitals_features(window_df: pd.DataFrame) -> dict:
    """Summary features for one patient's window of readings."""
    r = {}
    for v in VITALS:
        s = window_df[v].dropna()
        r[f"{v}_mean"] = s.mean() if len(s) else np.nan
        r[f"{v}_min"] = s.min() if len(s) else np.nan
        r[f"{v}_max"] = s.max() if len(s) else np.nan
        r[f"{v}_last"] = s.iloc[-1] if len(s) else np.nan
        r[f"{v}_slope"] = np.polyfit(np.arange(len(s)), s.values, 1)[0] if len(s) >= 3 else np.nan
    return r


def build_patient_table(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby("ID", sort=True):
        w = g.head(window)
        r = {"ID": pid, "Death": int(g.Outcome.iloc[0] == 0)}     # 1 = died (positive class)
        r.update(vitals_features(w))
        for c in STATIC:                                          # admission = first recorded value
            s = g[c].dropna()
            r[c] = s.iloc[0] if len(s) else np.nan
        r["n_window"] = len(w)
        rows.append(r)
    return pd.DataFrame(rows)


def get_xy(window: int = WINDOW):
    """Convenience: raw file -> (patient_table, X, y, feature_names, removed_counts)."""
    df, removed = clean(load_raw())
    pat = build_patient_table(df, window)
    features = [c for c in pat.columns if c not in NON_FEATURES]
    return pat, pat[features], pat["Death"].values, features, removed
