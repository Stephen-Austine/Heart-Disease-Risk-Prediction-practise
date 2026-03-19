"""
app.py  ─  CardioScan v4.0
"""
import os, sys, json, datetime, io, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")

from flask import Flask, request, jsonify, send_from_directory, send_file

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER

app = Flask(__name__, static_folder="static", static_url_path="")

BASE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")
DATA_DIR   = os.path.join(BASE, "data")
PLOTS_DIR  = os.path.join(BASE, "static", "plots")
LOG_UCI    = os.path.join(DATA_DIR, "new_uci.csv")
LOG_KAGGLE = os.path.join(DATA_DIR, "new_kaggle.csv")
PRED_LOG   = os.path.join(DATA_DIR, "predictions_log.json")

os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_AUC = {
    "uci_logistic_regression": 0.0,  "uci_random_forest": 0.0,
    "uci_xgboost":             0.0,  "kaggle_logistic_regression": 0.0,
    "kaggle_random_forest":    0.0,  "kaggle_xgboost":             0.0,
}

MODEL_THRESHOLDS = {
    "uci_logistic_regression": 0.5,  "uci_random_forest": 0.5,
    "uci_xgboost":             0.5,  "kaggle_logistic_regression": 0.5,
    "kaggle_random_forest":    0.5,  "kaggle_xgboost":             0.5,
}

BEST_ALGO = {
    "uci":    max(["logistic_regression","random_forest","xgboost"],
                  key=lambda a: MODEL_AUC[f"uci_{a}"]),
    "kaggle": max(["logistic_regression","random_forest","xgboost"],
                  key=lambda a: MODEL_AUC[f"kaggle_{a}"]),
}

MODELS, PREPROCESSORS = {}, {}
# Separate clean (noise-reduced) preprocessors for RF and XGBoost on UCI
PREPROCESSORS_CLEAN = {}

def load_assets():
    for ds in ["uci","kaggle"]:
        prep_path = os.path.join(MODELS_DIR, f"{ds}_preprocessor.pkl")
        if not os.path.exists(prep_path):
            print(f"[WARN] Preprocessor not found: {prep_path} — run train.py first")
            continue
        PREPROCESSORS[ds] = joblib.load(prep_path)
        for a in ["logistic_regression","random_forest","xgboost"]:
            path = os.path.join(MODELS_DIR, f"{ds}_{a}.pkl")
            if os.path.exists(path):
                MODELS[f"{ds}_{a}"] = joblib.load(path)
    # Load noise-reduced preprocessor for UCI RF/XGBoost
    clean_path = os.path.join(MODELS_DIR, "uci_preprocessor_clean.pkl")
    if os.path.exists(clean_path):
        PREPROCESSORS_CLEAN["uci"] = joblib.load(clean_path)
    # Load AUC scores saved by train.py, if available
    auc_path = os.path.join(MODELS_DIR, "model_auc.json")
    if os.path.exists(auc_path):
        with open(auc_path) as f:
            MODEL_AUC.update(json.load(f))
    # Load optimal thresholds saved by train.py
    thresh_path = os.path.join(MODELS_DIR, "model_thresholds.json")
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            MODEL_THRESHOLDS.update(json.load(f))
    # Recalculate best algo based on loaded AUC
    for ds in ["uci", "kaggle"]:
        BEST_ALGO[ds] = max(["logistic_regression","random_forest","xgboost"],
                             key=lambda a: MODEL_AUC.get(f"{ds}_{a}", 0))
    print(f"✓ Models loaded. Best: UCI={BEST_ALGO['uci']}, Kaggle={BEST_ALGO['kaggle']}")

load_assets()

MEDIANS = {
    "uci":    {"age":54,"sex":1,"cp":0,"trestbps":132,"chol":247,"fbs":0,
               "restecg":0,"thalach":150,"exang":0,"oldpeak":0.8,"slope":1,"ca":0,"thal":2},
    "kaggle": {"Age":53,"Sex":"M","ChestPainType":"NAP","RestingBP":131,"Cholesterol":207,
               "FastingBS":0,"RestingECG":"Normal","MaxHR":136,"ExerciseAngina":"N",
               "Oldpeak":0.8,"ST_Slope":"Up"},
}

FLABELS = {
    "age":"Age","sex":"Sex","cp":"Chest Pain","trestbps":"Resting BP","chol":"Cholesterol",
    "fbs":"Fasting BS","restecg":"Resting ECG","thalach":"Max Heart Rate",
    "exang":"Exercise Angina","oldpeak":"ST Depression","slope":"ST Slope",
    "ca":"Vessels Coloured","thal":"Thalassemia",
    "Age":"Age","Sex":"Sex","ChestPainType":"Chest Pain","RestingBP":"Resting BP",
    "Cholesterol":"Cholesterol","FastingBS":"Fasting BS","RestingECG":"Resting ECG",
    "MaxHR":"Max Heart Rate","ExerciseAngina":"Exercise Angina","Oldpeak":"ST Depression",
    "ST_Slope":"ST Slope",
}

UCI_DISPLAY = {
    "age":("Age","years"),"sex":("Sex","0=F 1=M"),"cp":("Chest Pain Type","0-3"),
    "trestbps":("Resting BP","mm Hg"),"chol":("Cholesterol","mg/dL"),
    "fbs":("Fasting BS >120","0=No 1=Yes"),"restecg":("Resting ECG","0-2"),
    "thalach":("Max Heart Rate","bpm"),"exang":("Exercise Angina","0=No 1=Yes"),
    "oldpeak":("ST Depression","mm"),"slope":("ST Slope","0-2"),
    "ca":("Vessels Coloured","0-3"),"thal":("Thalassemia","1-3"),
}
KAGGLE_DISPLAY = {
    "Age":("Age","years"),"Sex":("Sex","M/F"),
    "ChestPainType":("Chest Pain","ASY/ATA/NAP/TA"),
    "RestingBP":("Resting BP","mm Hg"),"Cholesterol":("Cholesterol","mg/dL"),
    "FastingBS":("Fasting BS >120","0/1"),"RestingECG":("Resting ECG","Normal/ST/LVH"),
    "MaxHR":("Max Heart Rate","bpm"),"ExerciseAngina":("Exercise Angina","N/Y"),
    "Oldpeak":("ST Depression","mm"),"ST_Slope":("ST Slope","Up/Flat/Down"),
}

def validate_inputs(dataset, inputs):
    """Validate inputs against reasonable ranges based on dataset statistics"""
    errors = []
    
    if dataset == "uci":
        # UCI dataset valid ranges
        ranges = {
            "age": (29, 77),
            "trestbps": (94, 186),
            "chol": (126, 394),
            "thalach": (71, 202),
            "oldpeak": (0.0, 3.4),
            "cp": (0, 3),
            "restecg": (0, 2),
            "exang": (0, 1),
            "slope": (0, 2),
            "ca": (0, 3),
            "thal": (1, 3),
            "fbs": (0, 1),
            "sex": (0, 1)
        }
    else:
        # Kaggle dataset valid ranges
        ranges = {
            "Age": (28, 77),
            "RestingBP": (76, 190),
            "Cholesterol": (0, 534),
            "MaxHR": (60, 202),
            "Oldpeak": (0.0, 4.0),
            "FastingBS": (0, 1)
        }
    
    for field, (min_val, max_val) in ranges.items():
        if field in inputs:
            # Check if input is empty
            if inputs[field] == "" or inputs[field] is None:
                errors.append(f"{FLABELS.get(field, field)} is required")
                continue
                
            try:
                value = float(inputs[field])
                if value < min_val or value > max_val:
                    errors.append(f"{FLABELS.get(field, field)} must be between {min_val} and {max_val}")
            except (ValueError, TypeError):
                errors.append(f"{FLABELS.get(field, field)} must be a valid number")
    
    return errors

def build_row(dataset, inputs):
    if dataset == "uci":
        return pd.DataFrame([{
            "age":int(inputs["age"]),"sex":int(inputs["sex"]),"cp":int(inputs["cp"]),
            "trestbps":int(inputs["trestbps"]),"chol":int(inputs["chol"]),
            "fbs":int(inputs["fbs"]),"restecg":int(inputs["restecg"]),
            "thalach":int(inputs["thalach"]),"exang":int(inputs["exang"]),
            "oldpeak":float(inputs["oldpeak"]),"slope":int(inputs["slope"]),
            "ca":int(inputs["ca"]),"thal":int(inputs["thal"]),
        }])
    return pd.DataFrame([{
        "Age":int(inputs["Age"]),"Sex":str(inputs["Sex"]),
        "ChestPainType":str(inputs["ChestPainType"]),
        "RestingBP":int(inputs["RestingBP"]),"Cholesterol":int(inputs["Cholesterol"]),
        "FastingBS":int(inputs["FastingBS"]),"RestingECG":str(inputs["RestingECG"]),
        "MaxHR":int(inputs["MaxHR"]),"ExerciseAngina":str(inputs["ExerciseAngina"]),
        "Oldpeak":float(inputs["Oldpeak"]),"ST_Slope":str(inputs["ST_Slope"]),
    }])

def get_drivers(model, prep, row, dataset):
    import shap
    X = prep.transform(row)
    pred_prob = float(model.predict_proba(X)[0][1])

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[1].flatten()
        else:
            sv = shap_values.flatten()
    except Exception:
        background = np.zeros((1, X.shape[1]))
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1], background
        )
        sv = explainer.shap_values(X, nsamples=100).flatten()

    # Map transformed feature indices back to original column names
    drivers = []
    feature_names = prep.get_feature_names_out() if hasattr(prep, 'get_feature_names_out') else row.columns.tolist()
    for i, (fname, shap_val) in enumerate(zip(feature_names, sv)):
        # Use original column name if we can match it, else use transformed name
        orig_col = fname.split("__")[-1] if "__" in fname else fname
        label = FLABELS.get(orig_col, FLABELS.get(fname, orig_col))
        drivers.append({"feature": orig_col, "label": label, "delta": round(float(shap_val), 4)})

    drivers.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return drivers[:6]

def log_prediction(dataset, inputs, prediction, probability):
    record = {"id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
              "timestamp": datetime.datetime.now().isoformat(),
              "dataset": dataset, "inputs": inputs,
              "prediction": prediction, "probability": round(probability,4)}
    log = []
    if os.path.exists(PRED_LOG):
        try:
            with open(PRED_LOG) as f: log = json.load(f)
        except: pass
    log.append(record)
    with open(PRED_LOG,"w") as f: json.dump(log, f, indent=2)
    row = dict(inputs)
    row["target" if dataset=="uci" else "HeartDisease"] = prediction
    csv = LOG_UCI if dataset=="uci" else LOG_KAGGLE
    df_new = pd.DataFrame([row])
    if os.path.exists(csv): df_new.to_csv(csv, mode="a", header=False, index=False)
    else: df_new.to_csv(csv, index=False)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index(): return send_from_directory("static", "index.html")

@app.route("/advanced")
def advanced(): return send_from_directory("static", "advanced.html")

@app.route("/settings")
def settings(): return send_from_directory("static", "settings.html")

@app.route("/plots/<path:filename>")
def serve_plot(filename): return send_from_directory(PLOTS_DIR, filename)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data    = request.get_json()
        dataset = data.get("dataset","uci")
        inputs  = data.get("inputs",{})
        
        # Validate inputs
        errors = validate_inputs(dataset, inputs)
        if errors:
            return jsonify({"success":False,"error":"Invalid inputs: " + "; ".join(errors)}), 400
            
        row     = build_row(dataset, inputs)
        algo    = BEST_ALGO[dataset]
        model   = MODELS[f"{dataset}_{algo}"]
        # Use noise-reduced preprocessor for UCI RF and XGBoost
        use_clean = (dataset == "uci" and algo in ("random_forest", "xgboost")
                     and "uci" in PREPROCESSORS_CLEAN)
        prep    = PREPROCESSORS_CLEAN["uci"] if use_clean else PREPROCESSORS[dataset]
        X       = prep.transform(row)
        thresh  = MODEL_THRESHOLDS.get(f"{dataset}_{algo}", 0.5)
        prob    = float(model.predict_proba(X)[0][1])
        pred    = int(prob >= thresh)
        risk    = "HIGH" if prob>=0.65 else "MODERATE" if prob>=0.40 else "LOW"

        # Each model needs its own preprocessor — RF and XGBoost use clean (20 features)
        def _prob_for(ds, a, raw_row):
            is_clean = (ds == "uci" and a in ("random_forest","xgboost")
                        and "uci" in PREPROCESSORS_CLEAN)
            p = PREPROCESSORS_CLEAN["uci"] if is_clean else PREPROCESSORS[ds]
            return round(float(MODELS[f"{ds}_{a}"].predict_proba(p.transform(raw_row))[0][1]), 3)

        all_probs = {a: _prob_for(dataset, a, row)
                     for a in ["logistic_regression","random_forest","xgboost"]}
        drivers = get_drivers(model, prep, row, dataset)
        vals    = list(all_probs.values())
        agreement = round(max(0,min(1, 1 - np.std(vals)*2)), 3)
        log_prediction(dataset, inputs, pred, prob)
        return jsonify({"success":True,"probability":round(prob,4),"prediction":pred,
                        "risk_level":risk,"best_model":algo,
                        "best_auc":MODEL_AUC[f"{dataset}_{algo}"],
                        "all_models":all_probs,"drivers":drivers,
                        "agreement":agreement,"dataset":dataset})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success":False,"error":str(e)}), 400

@app.route("/retrain", methods=["POST"])
def retrain():
    # Disabled in demo/cloud deployment - too resource-intensive for free tier
    # Run train.py locally to retrain. Set ENABLE_RETRAIN=1 env var to unlock.
    import os
    if not os.environ.get("ENABLE_RETRAIN"):
        return jsonify({
            "success": False,
            "error": "Retraining is disabled in the live demo. "
                     "Clone the repo and run train.py locally to retrain models."
        }), 403
    try:
        from preprocessing import (clean_uci, clean_kaggle, prepare_uci,
                                   prepare_uci_clean, prepare_kaggle)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = {}
        for ds in ["uci", "kaggle"]:
            clean_fn = clean_uci if ds == "uci" else clean_kaggle
            tgt      = "target" if ds == "uci" else "HeartDisease"
            csv      = LOG_UCI if ds == "uci" else LOG_KAGGLE
            base_csv = os.path.join(BASE, "data",
                                    "uci_heart.csv" if ds == "uci" else "kaggle_heart.csv")
            df = clean_fn(pd.read_csv(base_csv))
            n_new = 0
            if os.path.exists(csv):
                try:
                    df_new = pd.read_csv(csv)
                    if tgt in df_new.columns and len(df_new) > 0:
                        df = pd.concat([df, df_new], ignore_index=True)
                        n_new = len(df_new)
                except: pass

            if ds == "uci":
                Xtr, Xte, ytr, yte, preprocessor, _ = prepare_uci(df)
                Xtr_c, Xte_c, _, _, prep_c, _       = prepare_uci_clean(df)
                PREPROCESSORS[ds]       = preprocessor
                PREPROCESSORS_CLEAN[ds] = prep_c
                joblib.dump(prep_c, os.path.join(MODELS_DIR, "uci_preprocessor_clean.pkl"))
            else:
                Xtr, Xte, ytr, yte, preprocessor, _ = prepare_kaggle(df)
                PREPROCESSORS[ds] = preprocessor
            joblib.dump(preprocessor, os.path.join(MODELS_DIR, f"{ds}_preprocessor.pkl"))

            ALGOS = {
                "logistic_regression": LogisticRegression(C=10, max_iter=1000, random_state=42),
                "random_forest":       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                              min_samples_leaf=5, random_state=42),
                "xgboost":             XGBClassifier(n_estimators=150, learning_rate=0.1,
                                                     max_depth=4, subsample=0.8,
                                                     colsample_bytree=0.8,
                                                     use_label_encoder=False,
                                                     eval_metric="logloss",
                                                     random_state=42, verbosity=0),
            }
            ds_r = {"new_records": n_new, "total_records": len(df)}
            for algo, M in ALGOS.items():
                use_clean = (ds == "uci" and algo in ("random_forest", "xgboost"))
                X_tr_fit = Xtr_c if use_clean else Xtr
                X_te_fit = Xte_c if use_clean else Xte
                m   = M.fit(X_tr_fit, ytr)
                auc = roc_auc_score(yte, m.predict_proba(X_te_fit)[:, 1])
                cv  = cross_val_score(m, X_tr_fit, ytr, cv=CV, scoring="roc_auc")
                key = f"{ds}_{algo}"
                MODEL_AUC[key] = round(float(auc), 4)
                MODELS[key]    = m
                ds_r[algo]     = {"auc": round(float(auc), 4), "cv": round(float(cv.mean()), 4)}
                joblib.dump(m, os.path.join(MODELS_DIR, f"{key}.pkl"))
            BEST_ALGO[ds] = max(ALGOS.keys(), key=lambda a: MODEL_AUC[f"{ds}_{a}"])
            ds_r["best"] = BEST_ALGO[ds]
            results[ds]  = ds_r

        with open(os.path.join(MODELS_DIR, "model_auc.json"), "w") as f:
            json.dump(MODEL_AUC, f, indent=2)
        return jsonify({"success": True, "results": results, "best": BEST_ALGO, "auc": MODEL_AUC})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/analytics")
def analytics():
    log = []
    if os.path.exists(PRED_LOG):
        try:
            with open(PRED_LOG) as f: log = json.load(f)
        except: pass
    n_uci    = len(pd.read_csv(LOG_UCI))    if os.path.exists(LOG_UCI)    else 0
    n_kaggle = len(pd.read_csv(LOG_KAGGLE)) if os.path.exists(LOG_KAGGLE) else 0
    plots = {
        "Model Performance": [
            {"file":"roc_uci.png",       "title":"ROC Curves — UCI Dataset"},
            {"file":"roc_kaggle.png",     "title":"ROC Curves — Kaggle Dataset"},
            {"file":"cm_lr_uci.png",      "title":"Confusion Matrix — Logistic Regression (UCI)"},
            {"file":"cm_lr_kaggle.png",   "title":"Confusion Matrix — Logistic Regression (Kaggle)"},
            {"file":"cm_rf_uci.png",      "title":"Confusion Matrix — Random Forest (UCI)"},
            {"file":"cm_rf_kaggle.png",   "title":"Confusion Matrix — Random Forest (Kaggle)"},
            {"file":"cm_gbm_uci.png",     "title":"Confusion Matrix — XGBoost (UCI)"},
            {"file":"cm_gbm_kaggle.png",  "title":"Confusion Matrix — XGBoost (Kaggle)"},
            {"file":"cv_stability.png",   "title":"5-Fold Cross-Validation Stability"},
            {"file":"compare_uci.png",    "title":"Model Comparison — UCI Dataset"},
            {"file":"compare_kaggle.png", "title":"Model Comparison — Kaggle Dataset"},
            {"file":"cross_dataset.png",  "title":"Cross-Dataset Generalisation"},
        ],
        "Feature Analysis & Explainability": [
            {"file":"feat_importance.png",              "title":"Permutation Feature Importance (XGBoost)"},
            {"file":"local_explanation_patient_1.png",  "title":"SHAP — Patient 1 Risk Drivers"},
            {"file":"local_explanation_patient_2.png",  "title":"SHAP — Patient 2 Risk Drivers"},
            {"file":"local_explanation_patient_3.png",  "title":"SHAP — Patient 3 Risk Drivers"},
        ],
        "Exploratory Data Analysis": [
            {"file":"01_class_balance.png",             "title":"Class Balance"},
            {"file":"02_continuous_distributions.png",  "title":"Continuous Feature Distributions"},
            {"file":"03_categorical_rates.png",         "title":"Categorical Feature Disease Rates"},
            {"file":"04_outlier_boxplots.png",          "title":"Outlier Detection — Boxplots"},
            {"file":"05_correlation_heatmaps.png",      "title":"Correlation Heatmaps"},
            {"file":"06_feature_target_correlation.png","title":"Feature–Target Correlation"},
            {"file":"07_cholesterol_anomaly.png",       "title":"Cholesterol Zero-Value Anomaly"},
            {"file":"08_statistical_significance.png",  "title":"Statistical Significance Tests"},
            {"file":"09_pairplot_top_features.png",     "title":"Pairplot — Top Features"},
            {"file":"10_cross_dataset_alignment.png",   "title":"Cross-Dataset Feature Alignment"},
            {"file":"11_signal_strength.png",           "title":"Predictive Signal Strength"},
            {"file":"12_eda_findings_summary.png",      "title":"EDA Findings Summary"},
        ],
        "Clinical Fairness": [
            {"file":"fairness.png", "title":"Fairness Audit — F1 by Demographic Subgroup"},
        ],
    }
    for g in plots: plots[g] = [p for p in plots[g]
                                 if os.path.exists(os.path.join(PLOTS_DIR,p["file"]))]
    return jsonify({"success":True,"model_auc":MODEL_AUC,"best":BEST_ALGO,
                    "total_predictions":len(log),
                    "new_data":{"uci":n_uci,"kaggle":n_kaggle},
                    "recent_predictions":log[-10:][::-1],"plots":plots})

@app.route("/predictions_log")
def predictions_log():
    log = []
    if os.path.exists(PRED_LOG):
        try:
            with open(PRED_LOG) as f: log = json.load(f)
        except: pass
    return jsonify({"success":True,"log":log[::-1],"total":len(log)})

@app.route("/patients", methods=["GET"])
def get_patients():
    """Get all saved patient profiles"""
    patients = []
    patients_file = os.path.join(DATA_DIR, "patients.json")
    if os.path.exists(patients_file):
        try:
            with open(patients_file, "r") as f:
                patients = json.load(f)
        except:
            pass
    return jsonify({"success": True, "patients": patients})

@app.route("/patients", methods=["POST"])
def save_patient():
    """Save a new patient profile"""
    try:
        data = request.get_json()
        patient = data.get("patient")
        patients_file = os.path.join(DATA_DIR, "patients.json")
        
        patients = []
        if os.path.exists(patients_file):
            try:
                with open(patients_file, "r") as f:
                    patients = json.load(f)
            except:
                pass
        
        patient["id"] = str(len(patients) + 1)
        patient["created_at"] = datetime.datetime.now().isoformat()
        patients.append(patient)
        
        with open(patients_file, "w") as f:
            json.dump(patients, f, indent=2, default=str)
        
        return jsonify({"success": True, "patient": patient})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/patients/<patient_id>", methods=["GET"])
def get_patient(patient_id):
    """Get a specific patient profile"""
    try:
        patients_file = os.path.join(DATA_DIR, "patients.json")
        if os.path.exists(patients_file):
            try:
                with open(patients_file, "r") as f:
                    patients = json.load(f)
                
                for patient in patients:
                    if patient.get("id") == patient_id:
                        return jsonify({"success": True, "patient": patient})
            
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500
        
        return jsonify({"success": False, "error": "Patient not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/patients/<patient_id>", methods=["DELETE"])
def delete_patient(patient_id):
    """Delete a patient profile"""
    try:
        patients_file = os.path.join(DATA_DIR, "patients.json")
        if os.path.exists(patients_file):
            try:
                with open(patients_file, "r") as f:
                    patients = json.load(f)
                
                patients = [p for p in patients if p.get("id") != patient_id]
                
                with open(patients_file, "w") as f:
                    json.dump(patients, f, indent=2, default=str)
                
                return jsonify({"success": True})
            
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500
        
        return jsonify({"success": False, "error": "Patient not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    try:
        data       = request.get_json()
        inputs     = data.get("inputs",{})
        dataset    = data.get("dataset","uci")
        prob       = data.get("probability",0)
        risk_level = data.get("risk_level","UNKNOWN")
        best_model = data.get("best_model","")
        best_auc   = data.get("best_auc",0)
        all_models = data.get("all_models",{})
        drivers    = data.get("drivers",[])
        display    = UCI_DISPLAY if dataset=="uci" else KAGGLE_DISPLAY

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf,pagesize=A4,
                                leftMargin=2*cm,rightMargin=2*cm,
                                topMargin=2*cm,bottomMargin=2*cm)
        C_ACC  = colors.HexColor("#00ccf5")
        C_RISK = {"HIGH":colors.HexColor("#ff3360"),"MODERATE":colors.HexColor("#ffaa00"),
                  "LOW":colors.HexColor("#00df72")}.get(risk_level,C_ACC)
        C_GREY = colors.HexColor("#334466"); C_LIGHT = colors.HexColor("#0d1a2e")
        C_TEXT = colors.HexColor("#aaccee"); C_SUB  = colors.HexColor("#7494b8")

        TS = TableStyle([
            ("BACKGROUND",(0,0),(-1,0),C_LIGHT),
            ("TEXTCOLOR",(0,0),(-1,0),C_ACC),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,0),9),
            ("BOTTOMPADDING",(0,0),(-1,0),7),
            ("TOPPADDING",(0,0),(-1,0),7),
            ("TEXTCOLOR",(0,1),(-1,-1),C_TEXT),
            ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
            ("FONTSIZE",(0,1),(-1,-1),9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_LIGHT,colors.HexColor("#0a1520")]),
            ("BOTTOMPADDING",(0,1),(-1,-1),5),
            ("TOPPADDING",(0,1),(-1,-1),5),
            ("GRID",(0,0),(-1,-1),0.3,C_GREY),
            ("LEFTPADDING",(0,0),(-1,-1),8),
            ("RIGHTPADDING",(0,0),(-1,-1),8),
        ])
        H1 = ParagraphStyle("H1",fontSize=20,textColor=C_ACC,spaceAfter=4,
                             fontName="Helvetica-Bold",alignment=TA_CENTER)
        H2 = ParagraphStyle("H2",fontSize=12,textColor=C_ACC,spaceAfter=6,
                             spaceBefore=14,fontName="Helvetica-Bold")
        SM = ParagraphStyle("SM",fontSize=8,textColor=C_SUB,fontName="Helvetica")

        story = []
        story.append(Paragraph("CardioScan",H1))
        story.append(Paragraph("Clinical Heart Disease Risk Report",
                                ParagraphStyle("s",fontSize=10,textColor=C_GREY,
                                               alignment=TA_CENTER,fontName="Helvetica")))
        story.append(Spacer(1,6))
        now = datetime.datetime.now().strftime("%d %B %Y  %H:%M")
        story.append(Paragraph(f"Generated: {now}  |  Dataset: {dataset.upper()}  |  "
                                f"Model: {best_model.replace('_',' ').title()}  |  AUC: {best_auc:.4f}",
                                ParagraphStyle("m",fontSize=8,textColor=C_GREY,
                                               alignment=TA_CENTER,fontName="Helvetica")))
        story.append(HRFlowable(width="100%",thickness=1,color=C_GREY,spaceAfter=12))

        story.append(Paragraph("Risk Assessment",H2))
        rd = [["Risk Level","Probability","Prediction"],
              [risk_level, f"{prob*100:.1f}%",
               "Heart Disease Likely" if round(prob)==1 else "Heart Disease Unlikely"]]
        rt = Table(rd,colWidths=[5.5*cm,5.5*cm,7*cm]); rt.setStyle(TS)
        rt.setStyle(TableStyle([("TEXTCOLOR",(0,1),(0,1),C_RISK),
                                ("FONTNAME",(0,1),(0,1),"Helvetica-Bold"),
                                ("FONTSIZE",(0,1),(0,1),13)]))
        story.append(rt)

        story.append(Paragraph("Patient Clinical Inputs",H2))
        rows=[["Parameter","Value","Unit"]]
        for k,v in inputs.items():
            label,unit=display.get(k,(k,""))
            rows.append([label,str(v),unit])
        t=Table(rows,colWidths=[6*cm,5*cm,7*cm]); t.setStyle(TS); story.append(t)

        story.append(Paragraph("All Models Comparison",H2))
        am=[["Model","Probability","AUC (holdout)"]]
        amap={"logistic_regression":"Logistic Regression",
              "random_forest":"Random Forest","xgboost":"XGBoost"}
        for a,p in all_models.items():
            am.append([amap.get(a,a)+(" ★" if a==best_model else ""),
                       f"{p*100:.1f}%", f"{MODEL_AUC.get(f'{dataset}_{a}',0):.4f}"])
        t2=Table(am,colWidths=[7*cm,5*cm,6*cm]); t2.setStyle(TS); story.append(t2)

        story.append(Paragraph("Top Risk Drivers",H2))
        dr=[["Feature","Impact","Direction"]]
        for d in drivers:
            dr.append([d.get("label",d["feature"]),
                       f"{abs(d['delta'])*100:.1f}%",
                       "↑ Increases Risk" if d["delta"]>0 else "↓ Decreases Risk"])
        t3=Table(dr,colWidths=[6*cm,5*cm,7*cm]); t3.setStyle(TS); story.append(t3)

        story.append(Spacer(1,16))
        story.append(HRFlowable(width="100%",thickness=0.5,color=C_GREY))
        story.append(Spacer(1,6))
        story.append(Paragraph(
            "⚠ DISCLAIMER: This should be used alongside a medical professional and not as a standalone diagnostic tool."
            "Not a substitute for professional medical diagnosis. "
            "Always work together with a qualified cardiologist.",SM))

        doc.build(story)
        buf.seek(0)
        ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return send_file(buf,mimetype="application/pdf",
                         as_attachment=True,download_name=f"CardioScan_Report_{ts}.pdf")
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success":False,"error":str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status":"ok","best":BEST_ALGO,"models":len(MODELS),"auc":MODEL_AUC})

if __name__ == "__main__":
    print("\n"+"="*55)
    print("  CardioScan v4.0")
    print(f"  Best: UCI={BEST_ALGO['uci']}, Kaggle={BEST_ALGO['kaggle']}")
    print("  Main:     http://localhost:5000")
    print("  Advanced: http://localhost:5000/advanced")
    print("="*55+"\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port, host="0.0.0.0")
