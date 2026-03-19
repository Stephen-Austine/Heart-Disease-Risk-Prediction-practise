"""
models.py
---------
Defines, trains, and tunes three classifiers:
  1. Logistic Regression  (interpretable baseline)
  2. Random Forest        (ensemble, bagging)
  3. XGBoost              (ensemble, gradient boosting)

Each model is wrapped in a consistent API: train(), predict(), predict_proba().
Hyperparameter tuning uses GridSearchCV with 5-fold stratified CV.
"""

import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)


from sklearn.metrics import roc_curve

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def find_optimal_threshold(model, X_val, y_val):
    """Find threshold that maximises Youden's J (sensitivity + specificity - 1)."""
    y_prob = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = float(thresholds[best_idx])
    print(f"  Optimal threshold: {best_thresh:.3f} (Youden's J = {j_scores[best_idx]:.3f})")
    return best_thresh


# ── Logistic Regression ───────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train, tune=True):
    print("\n── Training Logistic Regression ──")
    if tune:
        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [1000],
        }
        grid = GridSearchCV(
            LogisticRegression(random_state=42),
            param_grid, cv=CV, scoring="roc_auc", n_jobs=-1
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        print(f"  Best params: {grid.best_params_}")
        print(f"  CV AUC: {grid.best_score_:.4f}")
    else:
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
    return model


# ── Random Forest ─────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train, tune=True):
    print("\n── Training Random Forest ──")
    if tune:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [8, 10, 12, None],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt"],
        }
        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid, cv=CV, scoring="roc_auc", n_jobs=-1
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        print(f"  Best params: {grid.best_params_}")
        print(f"  CV AUC: {grid.best_score_:.4f}")
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
    return model


# ── XGBoost ───────────────────────────────────────────────────────────────────

def train_gradient_boosting(X_train, y_train, tune=True):
    print("\n── Training XGBoost ──")
    # Hold out 15% of training data as early-stopping eval set
    from sklearn.model_selection import train_test_split as tts
    X_tr, X_es, y_tr, y_es = tts(X_train, y_train, test_size=0.15,
                                   stratify=y_train, random_state=42)
    if tune:
        param_grid = {
            "n_estimators": [100, 150, 200],
            "learning_rate": [0.05, 0.1, 0.15],
            "max_depth": [3, 4, 6],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        grid = GridSearchCV(
            XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                early_stopping_rounds=20,
                random_state=42,
                verbosity=0,
            ),
            param_grid, cv=CV, scoring="roc_auc", n_jobs=-1
        )
        grid.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        model = grid.best_estimator_
        print(f"  Best params: {grid.best_params_}")
        print(f"  CV AUC: {grid.best_score_:.4f}")
    else:
        model = XGBClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            early_stopping_rounds=20,
            random_state=42, verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name="Model", threshold=0.5):
    """Return a dict of all evaluation metrics, using optimal threshold if provided."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "model": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    print(f"\n  [{model_name}] Test Results (threshold={threshold:.3f}):")
    print(f"    Accuracy : {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall   : {metrics['recall']:.4f}")
    print(f"    F1-Score : {metrics['f1']:.4f}")
    print(f"    AUC-ROC  : {metrics['auc']:.4f}")
    print(f"    Confusion Matrix:\n{metrics['confusion_matrix']}")
    return metrics


def cross_validate_model(model, X, y, model_name="Model"):
    """Run 5-fold CV and return mean ± std for each metric."""
    from sklearn.base import clone
    # XGBoost with early_stopping_rounds requires an eval_set at fit time,
    # which cross_validate never provides — clone without it for CV only.
    if hasattr(model, 'early_stopping_rounds') and model.early_stopping_rounds:
        cv_model = clone(model)
        cv_model.set_params(early_stopping_rounds=None)
    else:
        cv_model = model
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_results = cross_validate(cv_model, X, y, cv=CV, scoring=scoring,
                                return_train_score=False, n_jobs=-1)
    summary = {}
    print(f"\n  [{model_name}] 5-Fold CV Results:")
    for metric in scoring:
        key = f"test_{metric}"
        mean = cv_results[key].mean()
        std = cv_results[key].std()
        summary[metric] = {"mean": mean, "std": std, "values": cv_results[key]}
        print(f"    {metric:12s}: {mean:.4f} ± {std:.4f}")
    return summary


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"  Model saved → {path}")


def load_model(path):
    return joblib.load(path)
