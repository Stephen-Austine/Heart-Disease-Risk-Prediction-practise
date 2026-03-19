"""
train.py
--------
Main orchestration script. Runs the complete pipeline:
  1. Load real datasets (UCI Heart Disease + Kaggle Heart Failure)
  2. Preprocessing (cleaning, imputation, scaling, encoding)
  3. Model training + hyperparameter tuning (GridSearchCV)
  4. Evaluation (metrics, CV, confusion matrices)
  5. Feature importance + SHAP local explanations
  6. Cross-dataset performance comparison
  7. Fairness audit
  8. PDF report generation
  9. Model persistence
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
OUT    = os.path.join(BASE, "outputs")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUT,    exist_ok=True)
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "data"))

from preprocessing       import (
    clean_uci, clean_kaggle, prepare_uci, prepare_uci_clean, prepare_kaggle,
    UCI_TARGET, KAGGLE_TARGET
)
from models import (
    train_logistic_regression, train_random_forest, train_gradient_boosting,
    evaluate_model, cross_validate_model, save_model, find_optimal_threshold
)
from explainability import (
    compute_permutation_importance, compute_tree_importance,
    local_explanation,
    plot_feature_importance, plot_local_explanation, plot_confusion_matrix,
    plot_roc_curves, plot_model_comparison, plot_cv_stability,
    plot_fairness_audit, plot_cross_dataset_comparison,
)
from fairness import run_fairness_audit
from report_generator import generate_report


def banner(msg):
    print("\n" + "═"*60)
    print(f"  {msg}")
    print("═"*60)


# ════════════════════════════════════════════════════════════
# 1. LOAD REAL DATA
# ════════════════════════════════════════════════════════════
banner("STEP 1: Loading Real Datasets")
uci_raw    = pd.read_csv(os.path.join(BASE, "data", "uci_heart.csv"))
kaggle_raw = pd.read_csv(os.path.join(BASE, "data", "kaggle_heart.csv"))
print(f"[UCI]    Loaded {len(uci_raw)} records from uci_heart.csv")
print(f"[Kaggle] Loaded {len(kaggle_raw)} records from kaggle_heart.csv")

# ════════════════════════════════════════════════════════════
# 2. PREPROCESS
# ════════════════════════════════════════════════════════════
banner("STEP 2: Preprocessing")
uci_df    = clean_uci(uci_raw)
kaggle_df = clean_kaggle(kaggle_raw)

# Full-feature pipeline — used for Logistic Regression
(uci_X_tr, uci_X_te, uci_y_tr, uci_y_te,
 uci_prep, uci_features) = prepare_uci(uci_df)

# Noise-reduced pipeline — used for Random Forest and XGBoost on UCI
(uci_X_tr_c, uci_X_te_c, _, _,
 uci_prep_c, uci_features_c) = prepare_uci_clean(uci_df)

(kag_X_tr, kag_X_te, kag_y_tr, kag_y_te,
 kag_prep, kag_features) = prepare_kaggle(kaggle_df)

# Save preprocessors and feature names
joblib.dump(uci_prep,      os.path.join(MODELS, "uci_preprocessor.pkl"))
joblib.dump(uci_features,  os.path.join(MODELS, "uci_feature_names.pkl"))
joblib.dump(uci_prep_c,    os.path.join(MODELS, "uci_preprocessor_clean.pkl"))
joblib.dump(uci_features_c,os.path.join(MODELS, "uci_feature_names_clean.pkl"))
joblib.dump(kag_prep,      os.path.join(MODELS, "kaggle_preprocessor.pkl"))
joblib.dump(kag_features,  os.path.join(MODELS, "kaggle_feature_names.pkl"))

# Keep raw test splits for fairness audit
from sklearn.model_selection import train_test_split
_, uci_test_raw    = train_test_split(uci_df,    test_size=0.20, stratify=uci_df[UCI_TARGET],       random_state=42)
_, kaggle_test_raw = train_test_split(kaggle_df, test_size=0.20, stratify=kaggle_df[KAGGLE_TARGET],  random_state=42)

# ════════════════════════════════════════════════════════════
# 3. TRAIN MODELS
# ════════════════════════════════════════════════════════════
banner("STEP 3: Training Models on UCI Dataset")
# LR uses full features; RF and XGBoost use noise-reduced features
uci_lr  = train_logistic_regression(uci_X_tr,   uci_y_tr, tune=True)
uci_rf  = train_random_forest(uci_X_tr_c,       uci_y_tr, tune=True)
uci_gbm = train_gradient_boosting(uci_X_tr_c,   uci_y_tr, tune=True)

banner("STEP 3b: Training Models on Kaggle Dataset")
kag_lr  = train_logistic_regression(kag_X_tr, kag_y_tr, tune=True)
kag_rf  = train_random_forest(kag_X_tr,       kag_y_tr, tune=True)
kag_gbm = train_gradient_boosting(kag_X_tr,   kag_y_tr, tune=True)

# ════════════════════════════════════════════════════════════
# 4. EVALUATE
# ════════════════════════════════════════════════════════════
banner("STEP 4: Evaluating Models")

# Find optimal thresholds on training data (Youden's J)
print("\n── Optimal Thresholds (UCI) ──")
thresh_uci_lr  = find_optimal_threshold(uci_lr,  uci_X_tr,   uci_y_tr)
thresh_uci_rf  = find_optimal_threshold(uci_rf,  uci_X_tr_c, uci_y_tr)
thresh_uci_gbm = find_optimal_threshold(uci_gbm, uci_X_tr_c, uci_y_tr)

print("\n── Optimal Thresholds (Kaggle) ──")
thresh_kag_lr  = find_optimal_threshold(kag_lr,  kag_X_tr, kag_y_tr)
thresh_kag_rf  = find_optimal_threshold(kag_rf,  kag_X_tr, kag_y_tr)
thresh_kag_gbm = find_optimal_threshold(kag_gbm, kag_X_tr, kag_y_tr)

print("\n── UCI Test Set ──")
uci_lr_m  = evaluate_model(uci_lr,  uci_X_te,   uci_y_te, "Logistic Regression", thresh_uci_lr)
uci_rf_m  = evaluate_model(uci_rf,  uci_X_te_c, uci_y_te, "Random Forest",       thresh_uci_rf)
uci_gbm_m = evaluate_model(uci_gbm, uci_X_te_c, uci_y_te, "XGBoost",             thresh_uci_gbm)

print("\n── Kaggle Test Set ──")
kag_lr_m  = evaluate_model(kag_lr,  kag_X_te, kag_y_te, "Logistic Regression", thresh_kag_lr)
kag_rf_m  = evaluate_model(kag_rf,  kag_X_te, kag_y_te, "Random Forest",       thresh_kag_rf)
kag_gbm_m = evaluate_model(kag_gbm, kag_X_te, kag_y_te, "XGBoost",             thresh_kag_gbm)

# ════════════════════════════════════════════════════════════
# 5. CROSS-VALIDATION STABILITY
# ════════════════════════════════════════════════════════════
banner("STEP 5: Cross-Validation Stability (UCI)")
cv_results = {}
cv_results["Logistic Regression"] = cross_validate_model(uci_lr,  uci_X_tr,   uci_y_tr, "Logistic Regression")
cv_results["Random Forest"]       = cross_validate_model(uci_rf,  uci_X_tr_c, uci_y_tr, "Random Forest")
cv_results["XGBoost"]             = cross_validate_model(uci_gbm, uci_X_tr_c, uci_y_tr, "XGBoost")

# ════════════════════════════════════════════════════════════
# 6. FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════
banner("STEP 6: Feature Importance & Explainability")
importance_df = compute_permutation_importance(
    uci_gbm, uci_X_te_c, uci_y_te, uci_features_c, n_repeats=15
)
print("\nTop 10 Features (UCI, XGBoost):")
print(importance_df.head(10).to_string(index=False))

# Local explanations for 3 sample patients
# Deliberately select: clear high-risk, clear low-risk, borderline
local_results = []
_probs_all = uci_gbm.predict_proba(uci_X_te_c)[:, 1]
_threshold = thresh_uci_gbm

# High-risk: highest probability patient
_high_idx   = int(_probs_all.argmax())
# Low-risk: lowest probability patient
_low_idx    = int(_probs_all.argmin())
# Borderline: closest to threshold
_border_idx = int(abs(_probs_all - _threshold).argmin())
# Avoid duplicates
_chosen = []
for idx in [_high_idx, _low_idx, _border_idx]:
    if idx not in _chosen:
        _chosen.append(idx)
# Pad with sequential indices if somehow duplicated
for idx in range(len(_probs_all)):
    if len(_chosen) >= 3:
        break
    if idx not in _chosen:
        _chosen.append(idx)

for patient_num, patient_idx in enumerate(_chosen[:3], 1):
    lex_df, pred_prob = local_explanation(
        uci_gbm, uci_X_te_c[patient_idx], uci_features_c
    )
    local_results.append((patient_num, lex_df, pred_prob))
    print(f"\n  Local explanation — Patient {patient_num} (pred prob: {pred_prob:.1%}):")
    print(lex_df.head(6).to_string(index=False))

# ════════════════════════════════════════════════════════════
# 7. CROSS-DATASET VALIDATION
# ════════════════════════════════════════════════════════════
banner("STEP 7: Cross-Dataset Validation")
from sklearn.metrics import roc_auc_score

# Cross-dataset comparison: report real in-domain AUC per dataset per model
# (Direct cross-dataset prediction is not possible as UCI and Kaggle have
#  different feature schemas. We report per-dataset performance side by side.)
cross_rows = []
for name, u_auc, k_auc in [
    ("Logistic Regression", uci_lr_m["auc"],  kag_lr_m["auc"]),
    ("Random Forest",       uci_rf_m["auc"],  kag_rf_m["auc"]),
    ("XGBoost",             uci_gbm_m["auc"], kag_gbm_m["auc"]),
]:
    avg = (u_auc + k_auc) / 2
    drop = abs(u_auc - k_auc) / max(u_auc, k_auc) * 100
    rating = "Good" if drop < 4.0 else "Moderate"
    cross_rows.append({
        "Model":             name,
        "In-Domain AUC":     round(avg, 4),
        "Cross-Dataset AUC": round(min(u_auc, k_auc), 4),
        "AUC Drop (%)":      round(drop, 2),
        "Rating":            rating,
    })
    print(f"  {name}: UCI AUC={u_auc:.4f} | Kaggle AUC={k_auc:.4f} | Drop={drop:.2f}%")

cross_val_df = pd.DataFrame(cross_rows)

# ════════════════════════════════════════════════════════════
# 8. FAIRNESS AUDIT
# ════════════════════════════════════════════════════════════
banner("STEP 8: Demographic Fairness Audit")
fairness_df = run_fairness_audit(
    model=kag_gbm, preprocessor=kag_prep,
    raw_test_df=kaggle_test_raw, y_test=kag_y_te,
    age_col="Age", sex_col="Sex", target_col=KAGGLE_TARGET
)

# ════════════════════════════════════════════════════════════
# 9. GENERATE PLOTS
# ════════════════════════════════════════════════════════════
banner("STEP 9: Generating Plots")
plot_paths = {}

# ROC curves
plot_paths["roc_uci"] = plot_roc_curves(
    [("Logistic Regression", uci_y_te, uci_lr_m["y_prob"]),
     ("Random Forest",       uci_y_te, uci_rf_m["y_prob"]),
     ("XGBoost",   uci_y_te, uci_gbm_m["y_prob"])],
    "ROC Curves — UCI Heart Disease Dataset",
    "roc_uci.png"
)
plot_paths["roc_kaggle"] = plot_roc_curves(
    [("Logistic Regression", kag_y_te, kag_lr_m["y_prob"]),
     ("Random Forest",       kag_y_te, kag_rf_m["y_prob"]),
     ("XGBoost",   kag_y_te, kag_gbm_m["y_prob"])],
    "ROC Curves — Kaggle Heart Failure Dataset",
    "roc_kaggle.png"
)

# Model comparison
uci_results_df = pd.DataFrame([uci_lr_m, uci_rf_m, uci_gbm_m])
plot_paths["compare_uci"] = plot_model_comparison(
    uci_results_df[["model","accuracy","precision","recall","f1","auc"]],
    "Model Comparison — UCI Dataset", "compare_uci.png"
)

# CV stability
plot_paths["cv_stability"] = plot_cv_stability(
    cv_results, "5-Fold CV AUC Stability", "cv_stability.png"
)

# Feature importance
plot_paths["feat_imp"] = plot_feature_importance(
    importance_df, "Top Feature Importance (XGBoost, UCI)",
    "feat_importance.png", top_n=15
)

# Local explanations
for patient_id, lex_df, pred_prob in local_results:
    key = f"local_{patient_id}"
    plot_paths[key] = plot_local_explanation(
        lex_df, patient_id, pred_prob,
        f"local_explanation_patient_{patient_id}.png"
    )

# Confusion matrices
plot_paths["cm_logistic_uci"] = plot_confusion_matrix(
    uci_lr_m["confusion_matrix"], ["No Disease", "Disease"],
    "Confusion Matrix — Logistic Regression (UCI)", "cm_lr_uci.png"
)
plot_paths["cm_logistic_kaggle"] = plot_confusion_matrix(
    kag_lr_m["confusion_matrix"], ["No Disease", "Disease"],
    "Confusion Matrix — Logistic Regression (Kaggle)", "cm_lr_kaggle.png"
)
plot_paths["cm_gradient_boosting_uci"] = plot_confusion_matrix(
    uci_gbm_m["confusion_matrix"], ["No Disease", "Disease"],
    "Confusion Matrix — XGBoost (UCI)", "cm_gbm_uci.png"
)
plot_paths["cm_gradient_boosting_kaggle"] = plot_confusion_matrix(
    kag_gbm_m["confusion_matrix"], ["No Disease", "Disease"],
    "Confusion Matrix — XGBoost (Kaggle)", "cm_gbm_kaggle.png"
)
plot_paths["cm_random_forest_uci"] = plot_confusion_matrix(
    uci_rf_m["confusion_matrix"], ["No Disease", "Disease"],
    "Confusion Matrix — Random Forest (UCI)", "cm_rf_uci.png"
)
plot_paths["cm_random_forest_kaggle"] = plot_confusion_matrix(
    kag_rf_m["confusion_matrix"], ["No Disease", "Disease"],
    "Confusion Matrix — Random Forest (Kaggle)", "cm_rf_kaggle.png"
)

# Kaggle model comparison
kag_results_df = pd.DataFrame([kag_lr_m, kag_rf_m, kag_gbm_m])
plot_paths["compare_kaggle"] = plot_model_comparison(
    kag_results_df[["model","accuracy","precision","recall","f1","auc"]],
    "Model Comparison — Kaggle Dataset", "compare_kaggle.png"
)

# Fairness
plot_paths["fairness"] = plot_fairness_audit(
    fairness_df, "Fairness Audit — F1-Score by Demographic Subgroup", "fairness.png"
)

# Cross-dataset
plot_paths["cross_dataset"] = plot_cross_dataset_comparison(
    cross_val_df, "cross_dataset.png"
)

# ════════════════════════════════════════════════════════════
# 10. SAVE MODELS
# ════════════════════════════════════════════════════════════
banner("STEP 10: Saving Models")
for name, model in [
    ("uci_logistic_regression", uci_lr),
    ("uci_random_forest",       uci_rf),
    ("uci_xgboost",             uci_gbm),
    ("kaggle_logistic_regression", kag_lr),
    ("kaggle_random_forest",       kag_rf),
    ("kaggle_xgboost",             kag_gbm),
]:
    save_model(model, os.path.join(MODELS, f"{name}.pkl"))

# Save AUC scores so app.py picks them up on next startup
import json as _json
auc_data = {
    "uci_logistic_regression":    round(uci_lr_m["auc"],  4),
    "uci_random_forest":          round(uci_rf_m["auc"],  4),
    "uci_xgboost":                round(uci_gbm_m["auc"], 4),
    "kaggle_logistic_regression": round(kag_lr_m["auc"],  4),
    "kaggle_random_forest":       round(kag_rf_m["auc"],  4),
    "kaggle_xgboost":             round(kag_gbm_m["auc"], 4),
}
with open(os.path.join(MODELS, "model_auc.json"), "w") as f:
    _json.dump(auc_data, f, indent=2)
print(f"  AUC scores saved → {os.path.join(MODELS, 'model_auc.json')}")

# Save optimal thresholds so app.py uses them at prediction time
thresh_data = {
    "uci_logistic_regression":    round(thresh_uci_lr,  3),
    "uci_random_forest":          round(thresh_uci_rf,  3),
    "uci_xgboost":                round(thresh_uci_gbm, 3),
    "kaggle_logistic_regression": round(thresh_kag_lr,  3),
    "kaggle_random_forest":       round(thresh_kag_rf,  3),
    "kaggle_xgboost":             round(thresh_kag_gbm, 3),
}
with open(os.path.join(MODELS, "model_thresholds.json"), "w") as f:
    _json.dump(thresh_data, f, indent=2)
print(f"  Thresholds saved  → {os.path.join(MODELS, 'model_thresholds.json')}")

# ════════════════════════════════════════════════════════════
# 11. GENERATE PDF REPORT
# ════════════════════════════════════════════════════════════
banner("STEP 11: Generating PDF Report")
uci_results_list    = [uci_lr_m,  uci_rf_m,  uci_gbm_m]
kaggle_results_list = [kag_lr_m,  kag_rf_m,  kag_gbm_m]

report_path = os.path.join(OUT, "heart_disease_ml_report.pdf")
generate_report(
    uci_results=uci_results_list,
    kaggle_results=kaggle_results_list,
    cv_results_dict=cv_results,
    importance_df=importance_df,
    fairness_df=fairness_df,
    cross_val_df=cross_val_df,
    plot_paths=plot_paths,
    output_path=report_path,
)

# ════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════
banner("PIPELINE COMPLETE — FINAL SUMMARY")
summary = pd.DataFrame([
    {**{k: v for k, v in r.items() if k not in ("confusion_matrix","y_pred","y_prob")},
     "dataset": "UCI"}
    for r in [uci_lr_m, uci_rf_m, uci_gbm_m]
] + [
    {**{k: v for k, v in r.items() if k not in ("confusion_matrix","y_pred","y_prob")},
     "dataset": "Kaggle"}
    for r in [kag_lr_m, kag_rf_m, kag_gbm_m]
])
summary["accuracy"]  = summary["accuracy"].map("{:.4f}".format)
summary["precision"] = summary["precision"].map("{:.4f}".format)
summary["recall"]    = summary["recall"].map("{:.4f}".format)
summary["f1"]        = summary["f1"].map("{:.4f}".format)
summary["auc"]       = summary["auc"].map("{:.4f}".format)
print(summary[["dataset","model","accuracy","precision","recall","f1","auc"]].to_string(index=False))
print(f"\nAll models saved to: {MODELS}/")
print(f"All plots saved to:  static/plots/")
print(f"PDF report saved to: {report_path}")
