"""Model zoo: baselines, bagging, boosting and stacking, all as imputing pipelines."""
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier, StackingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from .config import SEED


def _pipe(model, scale=False):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    steps.append(("clf", model))
    return Pipeline(steps)


def make_rf(**kw):
    p = dict(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=1, random_state=SEED)
    p.update(kw); return RandomForestClassifier(**p)


def make_xgb(spw, **kw):
    p = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=1, scale_pos_weight=spw, eval_metric="logloss", n_jobs=1, random_state=SEED, verbosity=0)
    p.update(kw); return XGBClassifier(**p)


def make_lgb(**kw):
    p = dict(n_estimators=300, learning_rate=0.05, num_leaves=8, min_child_samples=5, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, class_weight="balanced", n_jobs=1, random_state=SEED, verbose=-1)
    p.update(kw); return LGBMClassifier(**p)


def make_cat(**kw):
    p = dict(iterations=300, depth=4, learning_rate=0.05, auto_class_weights="Balanced",
             random_seed=SEED, verbose=0, thread_count=1, allow_writing_files=False)
    p.update(kw); return CatBoostClassifier(**p)


def get_models(y) -> dict:
    """y: binary target with 1 = death. scale_pos_weight = class ratio (no feature leakage)."""
    spw = round(float((y == 0).sum()) / float((y == 1).sum()), 2)
    return {
        "Logistic Regression": _pipe(LogisticRegression(max_iter=2000, class_weight="balanced"), scale=True),
        "Decision Tree": _pipe(DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, class_weight="balanced", random_state=SEED)),
        "Bagging (trees)": _pipe(BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=2, class_weight="balanced"),
                                                   n_estimators=300, random_state=SEED, n_jobs=1)),
        "Random Forest": _pipe(make_rf()),
        "Extra Trees": _pipe(ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2,
                                                  class_weight="balanced_subsample", n_jobs=1, random_state=SEED)),
        "AdaBoost": _pipe(AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=SEED)),
        "Gradient Boosting": _pipe(GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                              subsample=0.8, random_state=SEED)),
        "XGBoost": _pipe(make_xgb(spw)),
        "LightGBM": _pipe(make_lgb()),
        "CatBoost": _pipe(make_cat()),
        "Stacking (RF+XGB+LGBM -> LR)": _pipe(StackingClassifier(
            estimators=[("rf", make_rf()), ("xgb", make_xgb(spw)), ("lgb", make_lgb())],
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
            stack_method="predict_proba", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1)),
    }
