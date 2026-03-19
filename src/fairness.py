"""
fairness.py
-----------
Runs the demographic fairness audit on the best-performing model (GBM).
Evaluates performance across age groups and gender subgroups.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def run_fairness_audit(model, preprocessor, raw_test_df, y_test,
                       age_col, sex_col, target_col):
    """
    Evaluate model performance across:
      - Age groups: <50, 50-64, 65+
      - Gender subgroups

    Parameters
    ----------
    model        : fitted sklearn model
    preprocessor : fitted ColumnTransformer
    raw_test_df  : original (un-preprocessed) test DataFrame
    y_test       : true labels (numpy array)
    age_col      : column name for age in raw_test_df
    sex_col      : column name for sex in raw_test_df
    target_col   : column name for target (will be dropped before transform)

    Returns
    -------
    pd.DataFrame with Subgroup, N, Precision, Recall, F1
    """
    # Reconstruct predictions on test set
    X_raw = raw_test_df.drop(columns=[target_col], errors="ignore")
    X_t   = preprocessor.transform(X_raw)
    y_pred = model.predict(X_t)

    age_vals = raw_test_df[age_col].values
    sex_vals = raw_test_df[sex_col].values

    # Normalise sex values to string for consistent comparison
    sex_vals_str = np.array([str(s).strip().upper()[0] for s in sex_vals])

    subgroups = []

    # Age groups
    for label, mask_fn in [
        ("Age < 50",   lambda a: a < 50),
        ("Age 50-64",  lambda a: (a >= 50) & (a < 65)),
        ("Age 65+",    lambda a: a >= 65),
    ]:
        mask = mask_fn(age_vals)
        n = mask.sum()
        if n < 5:
            continue
        yt, yp = y_test[mask], y_pred[mask]
        subgroups.append({
            "Subgroup":  label,
            "N":         int(n),
            "Precision": round(precision_score(yt, yp, zero_division=0), 4),
            "Recall":    round(recall_score(yt, yp,    zero_division=0), 4),
            "F1":        round(f1_score(yt, yp,        zero_division=0), 4),
        })

    # Gender
    for label, val in [("Male", "M"), ("Female", "F")]:
        mask = sex_vals_str == val
        n = mask.sum()
        if n < 5:
            continue
        yt, yp = y_test[mask], y_pred[mask]
        subgroups.append({
            "Subgroup":  label,
            "N":         int(n),
            "Precision": round(precision_score(yt, yp, zero_division=0), 4),
            "Recall":    round(recall_score(yt, yp,    zero_division=0), 4),
            "F1":        round(f1_score(yt, yp,        zero_division=0), 4),
        })

    df = pd.DataFrame(subgroups)
    print("\n── Fairness Audit Results ──")
    print(df.to_string(index=False))
    return df
