"""Model zoo: baselines, bagging, boosting and stacking, all as imputing pipelines."""
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
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


class OversampledMLP(ClassifierMixin, BaseEstimator):
    """Small neural network (ANN) for tiny, imbalanced tabular data.

    - inputs are standardised;
    - the minority class (deaths) is randomly oversampled to parity *inside the training data it is given*;
    - `n_seeds` networks with different initialisations are averaged, because one small net is very unstable.
    Expects imputed input (put it behind a SimpleImputer in a Pipeline).
    """

    def __init__(self, n_seeds=5, hidden_layer_sizes=(16, 8), alpha=0.01, max_iter=600, random_state=SEED):
        self.n_seeds = n_seeds
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        rng = np.random.RandomState(self.random_state)
        pos, neg = np.where(y == self.classes_[1])[0], np.where(y == self.classes_[0])[0]
        minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
        self.models_ = []
        for k in range(self.n_seeds):
            extra = rng.choice(minority, size=len(majority) - len(minority), replace=True)
            idx = np.concatenate([np.arange(len(y)), extra])
            net = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                                max_iter=self.max_iter, random_state=self.random_state + k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                net.fit(Xs[idx], y[idx])
            self.models_.append(net)
        return self

    def predict_proba(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        p = np.mean([m.predict_proba(Xs)[:, 1] for m in self.models_], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


class HybridRFANN(ClassifierMixin, BaseEstimator):
    """Blend of a Random Forest and an ANN: p = w * p_RF + (1 - w) * p_ANN.

    The RF weight w is chosen from `weights` using out-of-fold predictions made *inside the training data*
    (inner stratified CV), so the test fold never influences it. Near-ties go to the weight closest to 0.5.
    After the weight is chosen both models are refitted on all the training data.
    """

    def __init__(self, weights=(0.25, 0.5, 0.75), n_seeds=5, n_estimators=300, inner_splits=3, random_state=SEED):
        self.weights = weights
        self.n_seeds = n_seeds
        self.n_estimators = n_estimators
        self.inner_splits = inner_splits
        self.random_state = random_state

    def _rf(self):
        return RandomForestClassifier(n_estimators=self.n_estimators, min_samples_leaf=2, class_weight="balanced_subsample",
                                      n_jobs=1, random_state=self.random_state)

    def _ann(self):
        return OversampledMLP(n_seeds=self.n_seeds, random_state=self.random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        self.classes_ = np.unique(y)
        skf = StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.random_state)
        p_rf, p_ann = np.zeros(len(y)), np.zeros(len(y))
        for tr, va in skf.split(X, y):
            p_rf[va] = self._rf().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
            p_ann[va] = self._ann().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        self.inner_auc_ = {float(w): float(roc_auc_score(y, w * p_rf + (1 - w) * p_ann)) for w in self.weights}
        best = max(self.inner_auc_.values())
        near = [w for w, a in self.inner_auc_.items() if a >= best - 0.005]
        self.weight_ = min(near, key=lambda w: abs(w - 0.5))
        self.rf_ = self._rf().fit(X, y)
        self.ann_ = self._ann().fit(X, y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p = self.weight_ * self.rf_.predict_proba(X)[:, 1] + (1 - self.weight_) * self.ann_.predict_proba(X)[:, 1]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


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
        "Hybrid (RF + ANN blend)": _pipe(HybridRFANN()),
        "ANN (MLP)": _pipe(OversampledMLP()),          # the hybrid's neural-network half on its own, for reference
    }
