"""
preprocessing.py
----------------
Handles all data cleaning, encoding, scaling, and feature engineering
for both the UCI and Kaggle datasets. Returns train/test splits ready
for model training.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os


# ── UCI Dataset ───────────────────────────────────────────────────────────────

UCI_CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
UCI_CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
UCI_TARGET = "target"

# Features confirmed as statistical noise by EDA (chi² p > 0.5 or AUC ≈ 0.50).
# Dropped before tree model training to reduce overfitting on small UCI dataset.
UCI_NOISY_FEATURES = ["chol", "fbs", "restecg", "sex"]
UCI_CONTINUOUS_CLEAN = [f for f in UCI_CONTINUOUS if f not in UCI_NOISY_FEATURES]
UCI_CATEGORICAL_CLEAN = [f for f in UCI_CATEGORICAL if f not in UCI_NOISY_FEATURES]

# ── Kaggle Dataset ────────────────────────────────────────────────────────────

KAGGLE_CONTINUOUS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
KAGGLE_CATEGORICAL = ["Sex", "ChestPainType", "FastingBS", "RestingECG",
                      "ExerciseAngina", "ST_Slope"]
KAGGLE_TARGET = "HeartDisease"


def load_uci(path):
    df = pd.read_csv(path)
    print(f"[UCI] Loaded {len(df)} records, {df.shape[1]} columns")
    return df


def load_kaggle(path):
    df = pd.read_csv(path)
    print(f"[Kaggle] Loaded {len(df)} records, {df.shape[1]} columns")
    return df


def clean_uci(df):
    """Drop rows with missing values; print audit."""
    before = len(df)
    df = df.dropna().copy()
    after = len(df)
    if before != after:
        print(f"[UCI] Dropped {before - after} rows with missing values")
    # Binarise target: >0 → 1
    df[UCI_TARGET] = (df[UCI_TARGET] > 0).astype(int)
    print(f"[UCI] Class distribution: {df[UCI_TARGET].value_counts().to_dict()}")
    return df


def clean_kaggle(df):
    """Replace zero cholesterol/BP (clinically impossible) with median."""
    before = len(df)
    # Zero cholesterol is missing, not real
    median_chol = df.loc[df["Cholesterol"] > 0, "Cholesterol"].median()
    df["Cholesterol"] = df["Cholesterol"].replace(0, median_chol)
    # Zero BP is also missing
    median_bp = df.loc[df["RestingBP"] > 0, "RestingBP"].median()
    df["RestingBP"] = df["RestingBP"].replace(0, median_bp)
    df = df.dropna().copy()
    print(f"[Kaggle] Imputed zero cholesterol → {median_chol:.0f}; zero BP → {median_bp:.0f}")
    print(f"[Kaggle] Class distribution: {df[KAGGLE_TARGET].value_counts().to_dict()}")
    return df


def build_uci_preprocessor():
    """Return a ColumnTransformer for UCI features."""
    numeric_pipe = Pipeline([("scaler", MinMaxScaler())])
    cat_pipe = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, UCI_CONTINUOUS),
        ("cat", cat_pipe, [str(c) for c in UCI_CATEGORICAL]),
    ], remainder="drop")
    return preprocessor


def build_uci_preprocessor_clean():
    """Preprocessor for UCI with noisy features removed (for RF and XGBoost)."""
    numeric_pipe = Pipeline([("scaler", MinMaxScaler())])
    cat_pipe = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, UCI_CONTINUOUS_CLEAN),
        ("cat", cat_pipe, [str(c) for c in UCI_CATEGORICAL_CLEAN]),
    ], remainder="drop")
    return preprocessor


def build_kaggle_preprocessor():
    """Return a ColumnTransformer for Kaggle features."""
    numeric_pipe = Pipeline([("scaler", MinMaxScaler())])
    cat_pipe = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, KAGGLE_CONTINUOUS),
        ("cat", cat_pipe, KAGGLE_CATEGORICAL),
    ], remainder="drop")
    return preprocessor


def get_feature_names(preprocessor, cat_cols):
    """Extract readable feature names after OHE."""
    num_names = preprocessor.transformers_[0][2]
    ohe = preprocessor.transformers_[1][1].named_steps["ohe"]
    cat_names = list(ohe.get_feature_names_out(cat_cols))
    return list(num_names) + cat_names


def prepare_uci(df, test_size=0.20, random_state=42):
    X = df.drop(columns=[UCI_TARGET])
    y = df[UCI_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    preprocessor = build_uci_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor, [str(c) for c in UCI_CATEGORICAL])
    print(f"[UCI] Train: {X_train_t.shape} | Test: {X_test_t.shape}")
    return X_train_t, X_test_t, y_train.values, y_test.values, preprocessor, feature_names


def prepare_uci_clean(df, test_size=0.20, random_state=42):
    """UCI pipeline with noisy features removed — used for RF and XGBoost."""
    X = df.drop(columns=[UCI_TARGET])
    y = df[UCI_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    preprocessor = build_uci_preprocessor_clean()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor, [str(c) for c in UCI_CATEGORICAL_CLEAN])
    print(f"[UCI-clean] Train: {X_train_t.shape} | Test: {X_test_t.shape}")
    return X_train_t, X_test_t, y_train.values, y_test.values, preprocessor, feature_names


def prepare_kaggle(df, test_size=0.20, random_state=42):
    X = df.drop(columns=[KAGGLE_TARGET])
    y = df[KAGGLE_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    preprocessor = build_kaggle_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor, KAGGLE_CATEGORICAL)
    print(f"[Kaggle] Train: {X_train_t.shape} | Test: {X_test_t.shape}")
    return X_train_t, X_test_t, y_train.values, y_test.values, preprocessor, feature_names


def save_preprocessor(preprocessor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved → {path}")


def load_preprocessor(path):
    return joblib.load(path)
