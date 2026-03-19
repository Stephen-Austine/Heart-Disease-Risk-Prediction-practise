# CardioScan — Heart Disease Risk Prediction

A machine learning web application for early heart disease risk assessment.  
Built as a BSc Data Science capstone project at USIU-Africa.

## Live Demo

🔗 **[Update this link after deploying to Render]**

> The server may take ~30 seconds to wake up after inactivity (free tier).

---

## What it does

Enter a patient's clinical values and CardioScan returns:
- **Risk classification** — High / Low Risk with a probability score
- **Key risk drivers** — which features drove the prediction
- **Model choice** — Logistic Regression, Random Forest, or XGBoost
- **Dataset choice** — UCI Heart Disease or Kaggle Heart Failure

The **Advanced Analytics** tab shows ROC curves, confusion matrices,
cross-dataset generalisation, a demographic fairness audit, and
SHAP-style local explanations.

---

## Performance

| Dataset | Model | AUC-ROC |
|---|---|---|
| UCI Heart Disease | Logistic Regression | **0.882** |
| UCI Heart Disease | Random Forest | 0.827 |
| UCI Heart Disease | XGBoost | 0.793 |
| Kaggle Heart Failure | Logistic Regression | **0.916** |
| Kaggle Heart Failure | Random Forest | 0.892 |
| Kaggle Heart Failure | XGBoost | 0.892 |

---

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/cardioscan.git
cd cardioscan
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

To retrain from scratch:
```bash
python eda.py    # EDA plots (~10 sec)
python train.py  # All models (~5-10 min)
```

---

## Stack

Python · Flask · scikit-learn · XGBoost · pandas · matplotlib · ReportLab

---

## Datasets

- [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) — 303 records, 13 features
- [Kaggle Heart Failure](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) — 918 records, 11 features

---

## Disclaimer

Research prototype for academic purposes only. Not a medical device.  
Do not use for clinical diagnosis. Always consult a qualified clinician.
