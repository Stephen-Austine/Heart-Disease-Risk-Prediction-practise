"""
predict.py
----------
CLI prediction interface for the deployed artifact.
Usage:
    python predict.py --input patient_data.csv --model gbm --dataset uci
    python predict.py --demo

Outputs a risk classification, probability score, and top feature drivers
for each patient record.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
THRESHOLD  = 0.50  # Classification threshold


def load_artifacts(dataset="uci", model_type="gbm"):
    model_map = {"lr": "logistic_regression", "rf": "random_forest", "gbm": "xgboost"}
    model_name = model_map.get(model_type, model_type)
    model_path = os.path.join(MODEL_DIR, f"{dataset}_{model_name}.pkl")
    prep_path  = os.path.join(MODEL_DIR, f"{dataset}_preprocessor.pkl")
    feat_path  = os.path.join(MODEL_DIR, f"{dataset}_feature_names.pkl")

    if not os.path.exists(model_path):
        sys.exit(f"[ERROR] Model not found: {model_path}\nRun train.py first.")

    model       = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    feature_names = joblib.load(feat_path)
    return model, preprocessor, feature_names


def predict_batch(df_raw, model, preprocessor, feature_names, dataset="uci"):
    X_t = preprocessor.transform(df_raw)
    probs = model.predict_proba(X_t)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)

    # Compute feature contributions (perturbation-based)
    baseline = X_t.mean(axis=0)
    records = []
    for i, (prob, pred) in enumerate(zip(probs, preds)):
        contrib = []
        for j, fname in enumerate(feature_names):
            perturbed = X_t[i].copy()
            perturbed[j] = baseline[j]
            pert_prob = model.predict_proba(perturbed.reshape(1, -1))[0, 1]
            contrib.append((fname, float(prob - pert_prob)))
        contrib.sort(key=lambda x: abs(x[1]), reverse=True)
        top3 = contrib[:3]

        records.append({
            "patient_id":   i + 1,
            "risk_class":   "HIGH RISK" if pred == 1 else "LOW RISK",
            "probability":  round(float(prob), 4),
            "top_driver_1": f"{top3[0][0]} ({top3[0][1]:+.3f})" if len(top3) > 0 else "",
            "top_driver_2": f"{top3[1][0]} ({top3[1][1]:+.3f})" if len(top3) > 1 else "",
            "top_driver_3": f"{top3[2][0]} ({top3[2][1]:+.3f})" if len(top3) > 2 else "",
        })
    return pd.DataFrame(records)


def demo_prediction():
    """Run predictions on a small set of synthetic patient records."""
    print("\n" + "="*60)
    print("  HEART DISEASE PREDICTION SYSTEM — DEMO MODE")
    print("="*60)

    # Try to load trained model; fall back to a quick demo message
    if not os.path.exists(os.path.join(MODEL_DIR, "kaggle_xgboost.pkl")):
        print("\n[INFO] No trained models found. Run train.py first.\n")
        return

    model, preprocessor, feature_names = load_artifacts("kaggle", "gbm")

    # 5 synthetic demo patients
    demo_patients = pd.DataFrame({
        "Age":           [63, 45, 57, 38, 71],
        "Sex":           ["M", "F", "M", "F", "M"],
        "ChestPainType": ["ASY", "ATA", "NAP", "TA", "ASY"],
        "RestingBP":     [145, 110, 130, 122, 160],
        "Cholesterol":   [233, 198, 286, 175, 310],
        "FastingBS":     [1, 0, 0, 0, 1],
        "RestingECG":    ["Normal", "Normal", "ST", "Normal", "LVH"],
        "MaxHR":         [150, 168, 103, 175, 95],
        "ExerciseAngina":["N", "N", "Y", "N", "Y"],
        "Oldpeak":       [2.3, 0.0, 1.5, 0.0, 3.5],
        "ST_Slope":      ["Down", "Up", "Flat", "Up", "Flat"],
    })

    print("\nInput Patient Records:")
    print(demo_patients.to_string(index=False))

    results = predict_batch(demo_patients, model, preprocessor, feature_names, "kaggle")

    print("\n" + "─"*60)
    print("Prediction Results:")
    print("─"*60)
    for _, row in results.iterrows():
        flag = "🔴" if row["risk_class"] == "HIGH RISK" else "🟢"
        print(f"\n  Patient {row['patient_id']:>2} | {flag} {row['risk_class']:<12} "
              f"| Probability: {row['probability']:.1%}")
        print(f"    Top drivers: {row['top_driver_1']}")
        print(f"                 {row['top_driver_2']}")
        print(f"                 {row['top_driver_3']}")
    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Heart Disease Risk Prediction CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to input CSV file with patient records",
    )
    parser.add_argument(
        "--model", type=str, default="gbm",
        choices=["lr", "rf", "gbm"],
        help="Model to use: lr=Logistic Regression, rf=Random Forest, gbm=XGBoost",
    )
    parser.add_argument(
        "--dataset", type=str, default="kaggle",
        choices=["uci", "kaggle"],
        help="Which dataset's preprocessor to use",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo with synthetic patient records",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save predictions to this CSV path",
    )

    args = parser.parse_args()

    if args.demo or args.input is None:
        demo_prediction()
        return

    model, preprocessor, feature_names = load_artifacts(args.dataset, args.model)
    df_raw = pd.read_csv(args.input)
    print(f"\nLoaded {len(df_raw)} patient records from {args.input}")
    results = predict_batch(df_raw, model, preprocessor, feature_names, args.dataset)
    print(results.to_string(index=False))

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
