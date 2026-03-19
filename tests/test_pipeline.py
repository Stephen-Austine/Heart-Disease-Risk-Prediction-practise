"""
test_pipeline.py
----------------
Unit tests for preprocessing, model training, evaluation, and prediction
pipelines. Uses real CSV data from the data/ directory.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UCI_CSV    = os.path.join(BASE, "data", "uci_heart.csv")
KAGGLE_CSV = os.path.join(BASE, "data", "kaggle_heart.csv")

from preprocessing import (
    clean_uci, clean_kaggle, prepare_uci, prepare_uci_clean, prepare_kaggle,
    UCI_TARGET, KAGGLE_TARGET
)
from models import (
    train_logistic_regression, train_random_forest,
    evaluate_model, find_optimal_threshold
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.uci_df    = clean_uci(pd.read_csv(UCI_CSV))
        self.kaggle_df = clean_kaggle(pd.read_csv(KAGGLE_CSV))

    def test_uci_binary_target(self):
        vals = set(self.uci_df[UCI_TARGET].unique())
        self.assertTrue(vals.issubset({0, 1}))

    def test_kaggle_zero_cholesterol_imputed(self):
        self.assertEqual((self.kaggle_df["Cholesterol"] == 0).sum(), 0)

    def test_uci_split_shapes(self):
        X_tr, X_te, y_tr, y_te, _, _ = prepare_uci(self.uci_df)
        self.assertEqual(X_tr.shape[0] + X_te.shape[0], len(self.uci_df))
        self.assertEqual(X_tr.shape[0], y_tr.shape[0])
        self.assertEqual(X_tr.shape[1], 28, "Full UCI pipeline should produce 28 features")

    def test_uci_clean_split_shapes(self):
        X_tr, X_te, y_tr, y_te, _, _ = prepare_uci_clean(self.uci_df)
        self.assertEqual(X_tr.shape[1], 20, "Clean UCI pipeline should produce 20 features")

    def test_kaggle_split_shapes(self):
        X_tr, X_te, y_tr, y_te, _, _ = prepare_kaggle(self.kaggle_df)
        self.assertEqual(X_tr.shape[0] + X_te.shape[0], len(self.kaggle_df))

    def test_feature_range(self):
        X_tr, _, _, _, _, _ = prepare_uci(self.uci_df)
        # Continuous features should be in [0, 1] after MinMax
        self.assertGreaterEqual(X_tr[:, :5].min(), -0.01)
        self.assertLessEqual(X_tr[:, :5].max(),  1.01)

    def test_uci_record_count(self):
        self.assertEqual(len(self.uci_df), 303)

    def test_kaggle_record_count(self):
        self.assertEqual(len(self.kaggle_df), 918)


class TestModels(unittest.TestCase):

    def setUp(self):
        uci_df = clean_uci(pd.read_csv(UCI_CSV))
        (self.X_tr, self.X_te,
         self.y_tr, self.y_te, _, _) = prepare_uci(uci_df)

    def test_logistic_regression_trains(self):
        model = train_logistic_regression(self.X_tr, self.y_tr, tune=False)
        self.assertTrue(hasattr(model, "predict_proba"))

    def test_evaluate_returns_auc(self):
        model = train_logistic_regression(self.X_tr, self.y_tr, tune=False)
        metrics = evaluate_model(model, self.X_te, self.y_te, "LR")
        self.assertIn("auc", metrics)
        self.assertGreater(metrics["auc"], 0.7)

    def test_optimal_threshold_in_range(self):
        model = train_logistic_regression(self.X_tr, self.y_tr, tune=False)
        thresh = find_optimal_threshold(model, self.X_tr, self.y_tr)
        self.assertGreater(thresh, 0.0)
        self.assertLess(thresh, 1.0)

    def test_clean_vs_full_feature_count(self):
        uci_df = clean_uci(pd.read_csv(UCI_CSV))
        X_full, _, _, _, _, feats_full = prepare_uci(uci_df)
        X_clean, _, _, _, _, feats_clean = prepare_uci_clean(uci_df)
        self.assertGreater(len(feats_full), len(feats_clean))
        self.assertEqual(X_full.shape[1], 28)
        self.assertEqual(X_clean.shape[1], 20)


if __name__ == "__main__":
    unittest.main()
