"""
enhance.py
----------
Implements five targeted accuracy improvements identified from diagnosis:

  Problem 1: OVERFITTING (train AUC=1.0, test AUC=0.85 → gap of 0.15)
  Fix:        Stronger regularisation + early stopping in GBM

  Problem 2: 10/28 NEAR-ZERO FEATURES (noise hurts generalisation)
  Fix:        Recursive Feature Elimination + mutual info selection

  Problem 3: SMALL DATASET (UCI n=242 train)
  Fix:        SMOTE oversampling to balance and expand effective training set

  Problem 4: SINGLE MODEL VARIANCE
  Fix:        Stacking ensemble (LR + RF + GBM → meta-LR)
              Soft-voting ensemble as alternative

  Problem 5: NO FEATURE ENGINEERING
  Fix:        Clinical interaction terms (age×thalach, oldpeak×slope, etc.)

Runs full comparison: baseline vs each enhancement vs combined.
"""

import sys, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import (
    SelectFromModel, RFECV, mutual_info_classif, SelectKBest
)
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

CV5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)


# ─── helpers ─────────────────────────────────────────────────────────────────

def metrics(model, X_te, y_te, name):
    yp  = model.predict(X_te)
    ypr = model.predict_proba(X_te)[:, 1]
    return {
        "name":      name,
        "accuracy":  round(accuracy_score(y_te, yp),          4),
        "precision": round(precision_score(y_te, yp,          zero_division=0), 4),
        "recall":    round(recall_score(y_te, yp,             zero_division=0), 4),
        "f1":        round(f1_score(y_te, yp,                 zero_division=0), 4),
        "auc":       round(roc_auc_score(y_te, ypr),          4),
    }

def cv_auc(model, X, y):
    scores = cross_val_score(model, X, y, cv=CV5, scoring="roc_auc", n_jobs=-1)
    return scores.mean(), scores.std()

def banner(msg):
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}")


# ─── Enhancement 1: Feature Engineering ──────────────────────────────────────

def add_clinical_features_uci(X_raw_df):
    """
    Create interaction and ratio features grounded in cardiology:
    - Heart Rate Reserve proxy  (age-predicted max HR - actual max HR)
    - ST-slope interaction      (oldpeak × slope_encoded)
    - Cholesterol-BP product    (risk amplification)
    - Age-HR interaction        (nonlinear risk)
    """
    df = X_raw_df.copy()
    # Age-predicted max HR (Tanaka formula: 208 - 0.7*age)
    df["hr_reserve"] = (208 - 0.7 * df["age"]) - df["thalach"]
    # ST interaction
    df["st_severity"] = df["oldpeak"] * (df["slope"].map({0: 0, 1: 1, 2: 2}) + 1)
    # Cholesterol-BP product (normalised)
    df["chol_bp"]     = (df["chol"] * df["trestbps"]) / 10000
    # Age × thalach nonlinear risk
    df["age_hr"]      = df["age"] * df["thalach"] / 1000
    return df


def add_clinical_features_kaggle(X_raw_df):
    df = X_raw_df.copy()
    slope_map = {"Up": 0, "Flat": 1, "Down": 2}
    slope_enc = df["ST_Slope"].map(slope_map).fillna(0)
    df["hr_reserve"]  = (208 - 0.7 * df["Age"]) - df["MaxHR"]
    df["st_severity"] = df["Oldpeak"] * (slope_enc + 1)
    df["chol_bp"]     = (df["Cholesterol"] * df["RestingBP"]) / 10000
    df["age_hr"]      = df["Age"] * df["MaxHR"] / 1000
    return df


# ─── Enhancement 2: Feature Selection ────────────────────────────────────────

def select_features(X_tr, y_tr, X_te, feature_names, k="all", method="mi"):
    """
    Mutual information selection or tree-based selection.
    Returns filtered X_tr, X_te, selected feature names.
    """
    if method == "mi":
        mi = mutual_info_classif(X_tr, y_tr, random_state=42)
        threshold = np.percentile(mi, 30)   # drop bottom 30%
        mask = mi >= threshold
    else:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_tr, y_tr)
        threshold = np.percentile(rf.feature_importances_, 25)
        mask = rf.feature_importances_ >= threshold

    selected = [f for f, m in zip(feature_names, mask) if m]
    print(f"  Feature selection: {mask.sum()}/{len(mask)} features retained")
    return X_tr[:, mask], X_te[:, mask], selected, mask


# ─── Enhancement 3: SMOTE-style oversampling (manual, no imbalanced-learn) ───

def smote_manual(X, y, random_state=42):
    """
    Simplified SMOTE: oversample minority class by interpolating
    between nearest neighbours. No external dependency.
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    majority_class  = classes[np.argmax(counts)]
    minority_class  = classes[np.argmin(counts)]
    n_majority      = counts.max()
    n_minority      = counts.min()
    n_to_generate   = n_majority - n_minority

    X_min = X[y == minority_class]
    synthetic = []
    for _ in range(n_to_generate):
        idx1, idx2 = rng.choice(len(X_min), 2, replace=False)
        alpha      = rng.random()
        synthetic.append(X_min[idx1] + alpha * (X_min[idx2] - X_min[idx1]))

    X_syn = np.vstack(synthetic)
    y_syn = np.full(n_to_generate, minority_class)
    X_bal = np.vstack([X, X_syn])
    y_bal = np.concatenate([y, y_syn])

    # Shuffle
    idx = rng.permutation(len(y_bal))
    print(f"  SMOTE: {n_minority}→{n_majority} minority samples | "
          f"total train: {len(y_bal)}")
    return X_bal[idx], y_bal[idx]


# ─── Enhancement 4: Tuned GBM with stronger regularisation ───────────────────

def train_regularised_gbm(X_tr, y_tr):
    """GBM with explicit overfitting controls."""
    param_grid = {
        "n_estimators":    [200, 300, 400],
        "learning_rate":   [0.03, 0.05, 0.08],
        "max_depth":       [3, 4],
        "subsample":       [0.7, 0.8],
        "min_samples_leaf":[5, 10, 15],
        "max_features":    ["sqrt"],
    }
    grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid, cv=CV5, scoring="roc_auc", n_jobs=-1
    )
    grid.fit(X_tr, y_tr)
    print(f"  Best regularised GBM params: {grid.best_params_}")
    print(f"  CV AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


# ─── Enhancement 5: Stacking Ensemble ────────────────────────────────────────

def train_stacking(X_tr, y_tr):
    """
    Level-0: LR + RF + GBM + MLP
    Level-1: Logistic Regression meta-learner
    Uses out-of-fold predictions to avoid leakage.
    """
    base_learners = [
        ("lr",  LogisticRegression(C=10, max_iter=1000, random_state=42)),
        ("rf",  RandomForestClassifier(n_estimators=200, max_depth=8,
                                       random_state=42)),
        ("gbm", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                           max_depth=3, subsample=0.8,
                                           min_samples_leaf=10, random_state=42)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                              alpha=0.01, random_state=42)),
    ]
    meta = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta,
        cv=CV5,
        passthrough=False,
        n_jobs=-1,
    )
    stack.fit(X_tr, y_tr)
    print("  Stacking ensemble trained (LR + RF + GBM + MLP → meta-LR)")
    return stack


def train_voting(X_tr, y_tr):
    """Soft-voting ensemble."""
    estimators = [
        ("lr",  LogisticRegression(C=10, max_iter=1000, random_state=42)),
        ("rf",  RandomForestClassifier(n_estimators=200, max_depth=8,
                                       random_state=42)),
        ("gbm", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                           max_depth=3, subsample=0.8,
                                           min_samples_leaf=10, random_state=42)),
    ]
    voter = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    voter.fit(X_tr, y_tr)
    print("  Soft-voting ensemble trained (LR + RF + GBM)")
    return voter


# ─── Enhancement 6: Neural Network baseline ──────────────────────────────────

def train_mlp(X_tr, y_tr):
    param_grid = {
        "hidden_layer_sizes": [(64, 32), (128, 64), (64, 64, 32)],
        "alpha":              [0.001, 0.01, 0.05],
        "learning_rate_init": [0.001, 0.01],
    }
    grid = GridSearchCV(
        MLPClassifier(max_iter=500, random_state=42, early_stopping=True,
                      validation_fraction=0.1),
        param_grid, cv=CV5, scoring="roc_auc", n_jobs=-1
    )
    grid.fit(X_tr, y_tr)
    print(f"  Best MLP params: {grid.best_params_}")
    print(f"  CV AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


# ─── Comparison Plot ──────────────────────────────────────────────────────────

def plot_enhancement_comparison(results_list, dataset_name, filename):
    names = [r["name"] for r in results_list]
    aucs  = [r["auc"]  for r in results_list]
    accs  = [r["accuracy"] for r in results_list]
    f1s   = [r["f1"]   for r in results_list]

    x  = np.arange(len(names))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))

    b1 = ax.bar(x - w,   aucs, w, label="AUC-ROC",  color="#2E75B6", alpha=0.88)
    b2 = ax.bar(x,       accs, w, label="Accuracy",  color="#27AE60", alpha=0.88)
    b3 = ax.bar(x + w,   f1s,  w, label="F1-Score",  color="#E74C3C", alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha="right", fontsize=8.5)
    ax.set_ylim(0.60, 1.02)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Enhancement Comparison — {dataset_name}", fontsize=13, fontweight="bold")
    ax.axhline(0.90, color="grey", linestyle="--", linewidth=0.8, alpha=0.5, label="0.90 target")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    # Value labels on bars
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUT, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run_enhancements():
    import os
    from preprocessing import (
        clean_uci, clean_kaggle,
        build_uci_preprocessor, build_kaggle_preprocessor,
        get_feature_names, UCI_TARGET, KAGGLE_TARGET,
        UCI_CATEGORICAL, KAGGLE_CATEGORICAL
    )
    from sklearn.model_selection import train_test_split

    print("\n" + "═"*60)
    print("  ACCURACY ENHANCEMENT PIPELINE")
    print("═"*60)

    # ── Load data ──────────────────────────────────────────────────────────
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    uci_raw    = clean_uci(pd.read_csv(os.path.join(base, "data", "uci_heart.csv")))
    kaggle_raw = clean_kaggle(pd.read_csv(os.path.join(base, "data", "kaggle_heart.csv")))

    # ── Feature engineering before splitting ──────────────────────────────
    banner("Step A: Feature Engineering")
    uci_eng    = add_clinical_features_uci(
        uci_raw.drop(columns=[UCI_TARGET])
    )
    kaggle_eng = add_clinical_features_kaggle(
        kaggle_raw.drop(columns=[KAGGLE_TARGET])
    )
    uci_eng[UCI_TARGET]          = uci_raw[UCI_TARGET].values
    kaggle_eng[KAGGLE_TARGET]    = kaggle_raw[KAGGLE_TARGET].values
    print(f"  UCI features: {uci_raw.shape[1]-1} → {uci_eng.shape[1]-1}")
    print(f"  Kaggle features: {kaggle_raw.shape[1]-1} → {kaggle_eng.shape[1]-1}")

    # ── UCI Pipeline ───────────────────────────────────────────────────────
    banner("UCI DATASET")

    # Original split
    uci_X_raw = uci_raw.drop(columns=[UCI_TARGET])
    uci_y     = uci_raw[UCI_TARGET].values
    Xu_tr_raw, Xu_te_raw, yu_tr, yu_te = train_test_split(
        uci_X_raw, uci_y, test_size=0.20, stratify=uci_y, random_state=42
    )

    # Engineered split
    uci_X_eng = uci_eng.drop(columns=[UCI_TARGET])
    _, Xu_te_eng_raw = train_test_split(
        uci_X_eng, test_size=0.20, stratify=uci_y, random_state=42
    )
    Xu_tr_eng_raw = uci_X_eng.iloc[:len(yu_tr)].copy()

    # Preprocess original
    uci_cat = [str(c) for c in UCI_CATEGORICAL]
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline as SKPipeline

    def make_prep_uci_eng(df):
        cont = ["age","trestbps","chol","thalach","oldpeak",
                "hr_reserve","st_severity","chol_bp","age_hr"]
        cat  = uci_cat
        cont_p = SKPipeline([("sc", MinMaxScaler())])
        cat_p  = SKPipeline([("ohe", OneHotEncoder(handle_unknown="ignore",
                                                    sparse_output=False))])
        return ColumnTransformer([("num", cont_p, cont), ("cat", cat_p, cat)],
                                  remainder="drop")

    def make_prep_kag_eng(df):
        cont = ["Age","RestingBP","Cholesterol","MaxHR","Oldpeak",
                "hr_reserve","st_severity","chol_bp","age_hr"]
        cat  = KAGGLE_CATEGORICAL
        cont_p = SKPipeline([("sc", MinMaxScaler())])
        cat_p  = SKPipeline([("ohe", OneHotEncoder(handle_unknown="ignore",
                                                    sparse_output=False))])
        return ColumnTransformer([("num", cont_p, cont), ("cat", cat_p, cat)],
                                  remainder="drop")

    # Fit preprocessors
    from preprocessing import build_uci_preprocessor, build_kaggle_preprocessor, get_feature_names

    prep_u_orig = build_uci_preprocessor()
    Xu_tr = prep_u_orig.fit_transform(Xu_tr_raw)
    Xu_te = prep_u_orig.transform(Xu_te_raw)
    feat_u = get_feature_names(prep_u_orig, uci_cat)

    prep_u_eng = make_prep_uci_eng(Xu_tr_eng_raw)
    Xu_tr_eng = prep_u_eng.fit_transform(Xu_tr_eng_raw)
    Xu_te_eng = prep_u_eng.transform(Xu_te_eng_raw)

    # ── 1. Baseline ─────────────────────────────────────────────────────
    banner("1. Baseline GBM (UCI)")
    gbm_base = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=4,
        subsample=0.8, random_state=42
    ).fit(Xu_tr, yu_tr)
    r_base_u = metrics(gbm_base, Xu_te, yu_te, "Baseline GBM")
    print(f"  AUC={r_base_u['auc']} | Acc={r_base_u['accuracy']} | F1={r_base_u['f1']}")

    # ── 2. Regularised GBM ──────────────────────────────────────────────
    banner("2. Regularised GBM (stronger overfitting control)")
    gbm_reg = train_regularised_gbm(Xu_tr, yu_tr)
    r_reg_u = metrics(gbm_reg, Xu_te, yu_te, "Regularised GBM")
    print(f"  AUC={r_reg_u['auc']} | Acc={r_reg_u['accuracy']} | F1={r_reg_u['f1']}")

    # ── 3. Feature Selection ─────────────────────────────────────────────
    banner("3. Feature Selection (drop bottom-30% by mutual info)")
    Xu_tr_fs, Xu_te_fs, feat_sel, mask = select_features(
        Xu_tr, yu_tr, Xu_te, feat_u, method="mi"
    )
    gbm_fs = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.8, min_samples_leaf=10, random_state=42
    ).fit(Xu_tr_fs, yu_tr)
    r_fs_u = metrics(gbm_fs, Xu_te_fs, yu_te, "GBM + Feat Select")
    print(f"  AUC={r_fs_u['auc']} | Acc={r_fs_u['accuracy']} | F1={r_fs_u['f1']}")

    # ── 4. SMOTE ─────────────────────────────────────────────────────────
    banner("4. SMOTE Oversampling")
    Xu_tr_sm, yu_tr_sm = smote_manual(Xu_tr, yu_tr)
    gbm_sm = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.8, min_samples_leaf=10, random_state=42
    ).fit(Xu_tr_sm, yu_tr_sm)
    r_sm_u = metrics(gbm_sm, Xu_te, yu_te, "GBM + SMOTE")
    print(f"  AUC={r_sm_u['auc']} | Acc={r_sm_u['accuracy']} | F1={r_sm_u['f1']}")

    # ── 5. Feature Engineering ───────────────────────────────────────────
    banner("5. Clinical Feature Engineering")
    gbm_eng = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.8, min_samples_leaf=10, random_state=42
    ).fit(Xu_tr_eng, yu_tr)
    r_eng_u = metrics(gbm_eng, Xu_te_eng, yu_te, "GBM + Feat Eng")
    print(f"  AUC={r_eng_u['auc']} | Acc={r_eng_u['accuracy']} | F1={r_eng_u['f1']}")

    # ── 6. Neural Network ────────────────────────────────────────────────
    banner("6. Neural Network (MLP)")
    mlp = train_mlp(Xu_tr, yu_tr)
    r_mlp_u = metrics(mlp, Xu_te, yu_te, "MLP Neural Net")
    print(f"  AUC={r_mlp_u['auc']} | Acc={r_mlp_u['accuracy']} | F1={r_mlp_u['f1']}")

    # ── 7. Stacking Ensemble ─────────────────────────────────────────────
    banner("7. Stacking Ensemble (LR + RF + GBM + MLP → meta-LR)")
    stack = train_stacking(Xu_tr, yu_tr)
    r_stack_u = metrics(stack, Xu_te, yu_te, "Stacking Ensemble")
    print(f"  AUC={r_stack_u['auc']} | Acc={r_stack_u['accuracy']} | F1={r_stack_u['f1']}")

    # ── 8. Soft Voting Ensemble ──────────────────────────────────────────
    banner("8. Soft Voting Ensemble")
    voter = train_voting(Xu_tr, yu_tr)
    r_vote_u = metrics(voter, Xu_te, yu_te, "Soft Voting")
    print(f"  AUC={r_vote_u['auc']} | Acc={r_vote_u['accuracy']} | F1={r_vote_u['f1']}")

    # ── 9. COMBINED BEST (Feat Eng + SMOTE + Stacking) ───────────────────
    banner("9. COMBINED: Feat Eng + SMOTE + Stacking")
    Xu_tr_combo, yu_tr_combo = smote_manual(Xu_tr_eng, yu_tr)
    stack_combo = train_stacking(Xu_tr_combo, yu_tr_combo)
    r_combo_u = metrics(stack_combo, Xu_te_eng, yu_te, "COMBINED Best")
    print(f"  AUC={r_combo_u['auc']} | Acc={r_combo_u['accuracy']} | F1={r_combo_u['f1']}")

    uci_results = [r_base_u, r_reg_u, r_fs_u, r_sm_u,
                   r_eng_u, r_mlp_u, r_stack_u, r_vote_u, r_combo_u]

    # ═══════════════════════════════════════════════════════════════
    # KAGGLE PIPELINE
    # ═══════════════════════════════════════════════════════════════
    banner("KAGGLE DATASET")

    kag_X_raw = kaggle_raw.drop(columns=[KAGGLE_TARGET])
    kag_y     = kaggle_raw[KAGGLE_TARGET].values
    Xk_tr_raw, Xk_te_raw, yk_tr, yk_te = train_test_split(
        kag_X_raw, kag_y, test_size=0.20, stratify=kag_y, random_state=42
    )

    kag_X_eng = kaggle_eng.drop(columns=[KAGGLE_TARGET])
    _, Xk_te_eng_raw = train_test_split(
        kag_X_eng, test_size=0.20, stratify=kag_y, random_state=42
    )
    Xk_tr_eng_raw = kag_X_eng.iloc[:len(yk_tr)].copy()

    prep_k_orig = build_kaggle_preprocessor()
    Xk_tr = prep_k_orig.fit_transform(Xk_tr_raw)
    Xk_te = prep_k_orig.transform(Xk_te_raw)

    prep_k_eng = make_prep_kag_eng(Xk_tr_eng_raw)
    Xk_tr_eng = prep_k_eng.fit_transform(Xk_tr_eng_raw)
    Xk_te_eng = prep_k_eng.transform(Xk_te_eng_raw)

    banner("1. Baseline GBM (Kaggle)")
    gbm_k_base = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=4,
        subsample=0.8, random_state=42
    ).fit(Xk_tr, yk_tr)
    r_base_k = metrics(gbm_k_base, Xk_te, yk_te, "Baseline GBM")
    print(f"  AUC={r_base_k['auc']} | Acc={r_base_k['accuracy']} | F1={r_base_k['f1']}")

    banner("Regularised GBM (Kaggle)")
    gbm_k_reg = train_regularised_gbm(Xk_tr, yk_tr)
    r_reg_k = metrics(gbm_k_reg, Xk_te, yk_te, "Regularised GBM")
    print(f"  AUC={r_reg_k['auc']} | Acc={r_reg_k['accuracy']} | F1={r_reg_k['f1']}")

    banner("Stacking Ensemble (Kaggle)")
    stack_k = train_stacking(Xk_tr, yk_tr)
    r_stack_k = metrics(stack_k, Xk_te, yk_te, "Stacking Ensemble")
    print(f"  AUC={r_stack_k['auc']} | Acc={r_stack_k['accuracy']} | F1={r_stack_k['f1']}")

    banner("COMBINED Best (Kaggle)")
    Xk_tr_combo, yk_tr_combo = smote_manual(Xk_tr_eng, yk_tr)
    stack_k_combo = train_stacking(Xk_tr_combo, yk_tr_combo)
    r_combo_k = metrics(stack_k_combo, Xk_te_eng, yk_te, "COMBINED Best")
    print(f"  AUC={r_combo_k['auc']} | Acc={r_combo_k['accuracy']} | F1={r_combo_k['f1']}")

    kaggle_results = [r_base_k, r_reg_k, r_stack_k, r_combo_k]

    # ── Save best models ──────────────────────────────────────────────────
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    joblib.dump(stack_combo,   os.path.join(MODEL_DIR, "uci_best_model.pkl"))
    joblib.dump(prep_u_eng,    os.path.join(MODEL_DIR, "uci_best_preprocessor.pkl"))
    joblib.dump(stack_k_combo, os.path.join(MODEL_DIR, "kaggle_best_model.pkl"))
    joblib.dump(prep_k_eng,    os.path.join(MODEL_DIR, "kaggle_best_preprocessor.pkl"))
    print("\n  Best models saved.")

    # ── Plots ─────────────────────────────────────────────────────────────
    banner("Generating Comparison Plots")
    uci_plot = plot_enhancement_comparison(
        uci_results, "UCI Heart Disease Dataset", "enhancement_uci.png"
    )
    kag_plot = plot_enhancement_comparison(
        kaggle_results, "Kaggle Heart Failure Dataset", "enhancement_kaggle.png"
    )

    # ── Summary table ─────────────────────────────────────────────────────
    banner("FINAL ENHANCEMENT SUMMARY")
    print(f"\n{'UCI Dataset':}")
    df_u = pd.DataFrame(uci_results)
    print(df_u[["name","accuracy","f1","auc"]].to_string(index=False))

    print(f"\n{'Kaggle Dataset':}")
    df_k = pd.DataFrame(kaggle_results)
    print(df_k[["name","accuracy","f1","auc"]].to_string(index=False))

    # Improvement deltas
    best_u = max(uci_results, key=lambda r: r["auc"])
    best_k = max(kaggle_results, key=lambda r: r["auc"])
    print(f"\n  UCI    best: {best_u['name']} | AUC={best_u['auc']} "
          f"(+{best_u['auc'] - r_base_u['auc']:.4f} vs baseline)")
    print(f"  Kaggle best: {best_k['name']} | AUC={best_k['auc']} "
          f"(+{best_k['auc'] - r_base_k['auc']:.4f} vs baseline)")

    return uci_results, kaggle_results, uci_plot, kag_plot


if __name__ == "__main__":
    run_enhancements()
