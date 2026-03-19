"""
explainability.py
-----------------
Implements model interpretability using:
  - Permutation Importance (global, model-agnostic)
  - Tree-based Feature Importance (for RF and XGBoost)
  - SHAP values for local and global explanations (TreeExplainer)

All outputs feed into the report generator.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import shap
from sklearn.inspection import permutation_importance


PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

PALETTE = {
    "primary":   "#2E75B6",
    "secondary": "#E74C3C",
    "accent":    "#27AE60",
    "neutral":   "#7F8C8D",
    "light":     "#EBF3FB",
    "dark":      "#1A252F",
}


# ── Global Feature Importance ─────────────────────────────────────────────────

def compute_permutation_importance(model, X_test, y_test, feature_names,
                                   n_repeats=20, random_state=42):
    """Compute permutation importance (model-agnostic)."""
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, random_state=random_state,
        scoring="roc_auc", n_jobs=-1
    )
    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std":  result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


def compute_tree_importance(model, feature_names):
    """Extract built-in feature importances from tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return None
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# ── SHAP Local Explanation ────────────────────────────────────────────────────

def local_explanation(model, X_instance, feature_names, baseline_mean=None):
    """
    Compute real SHAP values for a single instance using TreeExplainer.
    Works with XGBoost, Random Forest, and Gradient Boosting models.
    Falls back to KernelExplainer for non-tree models (e.g. Logistic Regression).

    Returns a DataFrame of feature contributions and the predicted probability.
    """
    instance = X_instance.copy().reshape(1, -1)
    pred_prob = float(model.predict_proba(instance)[0, 1])

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(instance)
        # For binary classifiers, shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            sv = shap_values[1].flatten()
        else:
            sv = shap_values.flatten()
    except Exception:
        # Fallback for non-tree models
        background = np.zeros((1, instance.shape[1]))
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1], background
        )
        sv = explainer.shap_values(instance, nsamples=100).flatten()

    df = pd.DataFrame({
        "feature":      feature_names,
        "value":        instance.flatten(),
        "contribution": sv,
    }).sort_values("contribution", key=abs, ascending=False).reset_index(drop=True)

    return df, pred_prob


def compute_shap_global(model, X_data, feature_names):
    """
    Compute mean absolute SHAP values across a dataset for global importance.
    Returns a DataFrame sorted by mean |SHAP value|.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_data)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
    except Exception:
        background = X_data[:50]
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1], background
        )
        sv = explainer.shap_values(X_data, nsamples=100)

    mean_abs = np.abs(sv).mean(axis=0)
    df = pd.DataFrame({
        "feature":   feature_names,
        "shap_mean": mean_abs,
    }).sort_values("shap_mean", ascending=False).reset_index(drop=True)
    return df


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_feature_importance(importance_df, title, filename, top_n=15):
    """Horizontal bar chart of top-N feature importances."""
    df = importance_df.head(top_n).copy()
    df = df.sort_values("importance_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE["primary"] if i >= len(df) - 5 else PALETTE["neutral"]
              for i in range(len(df))]
    bars = ax.barh(df["feature"], df["importance_mean"], color=colors,
                   xerr=df.get("importance_std"), capsize=3,
                   error_kw={"ecolor": PALETTE["neutral"], "alpha": 0.7})

    ax.set_xlabel("Mean Permutation Importance (AUC drop)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_local_explanation(local_df, patient_id, pred_prob, filename, top_n=10):
    """Waterfall-style local explanation for one patient."""
    df = local_df.head(top_n).copy()
    df = df.sort_values("contribution", ascending=True)

    colors = [PALETTE["secondary"] if c < 0 else PALETTE["accent"]
              for c in df["contribution"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df["feature"], df["contribution"], color=colors)
    ax.axvline(0, color=PALETTE["dark"], linewidth=0.8)
    ax.set_xlabel("Contribution to Predicted Risk (SHAP-style)", fontsize=11)
    ax.set_title(
        f"Local Explanation — Patient {patient_id} | "
        f"Predicted Risk: {pred_prob:.1%}",
        fontsize=12, fontweight="bold"
    )
    red_patch  = mpatches.Patch(color=PALETTE["secondary"], label="Reduces risk")
    green_patch = mpatches.Patch(color=PALETTE["accent"],   label="Increases risk")
    ax.legend(handles=[green_patch, red_patch], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_confusion_matrix(cm, labels, title, filename):
    """Annotated confusion matrix heatmap."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=labels, yticklabels=labels,
           xlabel="Predicted Label", ylabel="True Label",
           title=title)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_roc_curves(models_data, title, filename):
    """
    Overlay ROC curves for multiple models.
    models_data: list of (model_name, y_test, y_prob) tuples
    """
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(7, 6))
    colors_list = [PALETTE["primary"], PALETTE["accent"], PALETTE["secondary"]]

    for (name, y_test, y_prob), color in zip(models_data, colors_list):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set(xlim=[0, 1], ylim=[0, 1.02],
           xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=title)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_model_comparison(results_df, title, filename):
    """Grouped bar chart comparing metrics across models."""
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    x = np.arange(len(metrics))
    width = 0.22
    model_names = results_df["model"].tolist()
    bar_colors = [PALETTE["primary"], PALETTE["accent"], PALETTE["secondary"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, (_, row) in enumerate(results_df.iterrows()):
        offset = (idx - 1) * width
        vals = [row[m] for m in metrics]
        ax.bar(x + offset, vals, width, label=row["model"],
               color=bar_colors[idx], alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    ax.axhline(0.9, color=PALETTE["neutral"], linestyle="--", linewidth=0.7, alpha=0.5)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_cv_stability(cv_results_dict, title, filename):
    """Box plots of 5-fold CV AUC scores across models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data   = [v["roc_auc"]["values"] for v in cv_results_dict.values()]
    labels = list(cv_results_dict.keys())
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    colors_list = [PALETTE["primary"], PALETTE["accent"], PALETTE["secondary"]]
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("AUC-ROC (5-fold CV)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_fairness_audit(fairness_df, title, filename):
    """Grouped bar chart of F1-Score by demographic subgroup."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(fairness_df))
    colors_list = [PALETTE["primary"] if "Age" in g else PALETTE["accent"]
                   for g in fairness_df["Subgroup"]]
    bars = ax.bar(x, fairness_df["F1"], color=colors_list, alpha=0.85, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(fairness_df["Subgroup"], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    for bar, val in zip(bars, fairness_df["F1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(fairness_df["F1"].mean(), color=PALETTE["secondary"],
               linestyle="--", linewidth=1, label="Mean F1")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_cross_dataset_comparison(cross_df, filename):
    """Heatmap of cross-dataset AUC results."""
    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(figsize=(7, 3.5))
    models = cross_df["Model"].tolist()
    cols   = ["In-Domain AUC", "Cross-Dataset AUC", "AUC Drop (%)"]
    data   = cross_df[cols].values.astype(float)
    im = ax.imshow(data.T, cmap="RdYlGn", aspect="auto", vmin=0.75, vmax=1.0)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=9)
    for i in range(len(models)):
        for j in range(len(cols)):
            val = data[i, j]
            ax.text(i, j, f"{val:.3f}" if j < 2 else f"{val:.2f}%",
                    ha="center", va="center", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Cross-Dataset Generalizability Results", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path
