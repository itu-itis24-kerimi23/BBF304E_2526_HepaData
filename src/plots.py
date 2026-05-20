from __future__ import annotations

import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import learning_curve
from sklearn.metrics import ConfusionMatrixDisplay

from .config import CLASS_LABELS


# ---------------------------------------------------------------------------
# Filename sanitiser
# ---------------------------------------------------------------------------

def _safe_filename(name: str) -> str:
    """
    Convert a model name (possibly containing spaces, parentheses, =) to a
    filesystem-safe lowercase string.
    """
    name = name.lower()
    name = re.sub(r"[^a-z0-9_]", "_", name)   # replace all non-alphanumeric with _
    name = re.sub(r"_+", "_", name)             # collapse repeated underscores
    return name.strip("_")


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def save_class_distribution_plot(class_counts, figures_dir, file_name="class_distribution.png"):
    plt.figure(figsize=(6, 4))
    bars = plt.bar(["No liver disease (0)", "Liver disease (1)"],
                   [class_counts.get(0, 0), class_counts.get(1, 0)],
                   color=["#378ADD", "#D85A30"])
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 str(int(bar.get_height())), ha="center", va="bottom", fontsize=11)
    plt.title("Class distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / file_name, dpi=150, bbox_inches="tight")
    plt.show()


def save_missing_values_plot(missing_values, figures_dir, file_name="missing_values.png"):
    plt.figure(figsize=(8, 4))
    missing_values.plot(kind="bar", color="#E24B4A")
    plt.title("Missing values by feature")
    plt.xlabel("Feature")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / file_name, dpi=150, bbox_inches="tight")
    plt.show()


def save_correlation_heatmap(df, figures_dir, file_name="correlation_matrix.png"):
    corr = df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    # Annotate cells
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    plt.title("Correlation matrix (numerical features)")
    plt.tight_layout()
    plt.savefig(figures_dir / file_name, dpi=150, bbox_inches="tight")
    plt.show()


def save_boxplots_by_class(df, target_col, figures_dir,
                            file_name="boxplots_by_class.png"):
    """
    Box-plots of every numerical feature split by class label.
    Reveals which features separate the two classes most clearly.
    """
    numeric_cols = [col for col in df.columns
                    if col not in (target_col, "Gender", "Selector")]
    n = len(numeric_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    colors = ["#378ADD", "#D85A30"]
    labels = ["No liver disease (0)", "Liver disease (1)"]

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        groups = [df.loc[df[target_col] == cls, col].dropna().values
                  for cls in [0, 1]]
        bp = ax.boxplot(groups, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(col, fontsize=10)
        ax.set_xticklabels(["No disease", "Disease"], fontsize=9)
        ax.set_ylabel("Value", fontsize=8)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Feature distributions by class", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(figures_dir / file_name, dpi=150, bbox_inches="tight")
    plt.show()


def save_outlier_summary(df, figures_dir, file_name="outlier_summary.png"):
    """
    IQR-based outlier counts per numerical feature.
    Visualises how many samples fall outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    """
    numeric_cols = [col for col in df.select_dtypes(include="number").columns
                    if col not in ("Selector", "target")]

    counts = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        counts[col] = n_out

    counts_s = pd.Series(counts).sort_values(ascending=False)

    plt.figure(figsize=(9, 4))
    bars = plt.bar(counts_s.index, counts_s.values, color="#EF9F27")
    for bar, val in zip(bars, counts_s.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 str(val), ha="center", va="bottom", fontsize=9)
    plt.title("Outlier count per feature (IQR rule)")
    plt.ylabel("Number of outliers")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / file_name, dpi=150, bbox_inches="tight")
    plt.show()

    return counts_s


# ---------------------------------------------------------------------------
# Model evaluation plots
# ---------------------------------------------------------------------------

def save_metric_bar_plot(results_df, metrics, figures_dir, file_name, title):
    results_df.set_index("Model")[metrics].plot(kind="bar", figsize=(11, 6))
    plt.title(title)
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=35, ha="right")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / file_name, dpi=150, bbox_inches="tight")
    plt.show()


def save_confusion_matrices(trained_estimators, X_test, y_test, figures_dir):
    for model_name, model in trained_estimators.items():
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, display_labels=CLASS_LABELS,
        )
        plt.title(f"Confusion matrix — {model_name}")
        plt.tight_layout()
        # FIXED: sanitise all special characters in filename
        safe_name = _safe_filename(model_name)
        plt.savefig(figures_dir / f"confusion_matrix_{safe_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()


def save_learning_curve_plot(estimator, X, y, model_name, figures_dir,
                              cv_splits, scoring="f1",
                              train_sizes=(0.25, 0.50, 0.75, 1.00)):
    """
    Learning curve using CV-based validation scores.
    Note: this uses eval_cv_splits for validation; it is presented as a
    learning curve (validation trend vs training size) — not repeated as a
    final performance estimate.
    """
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator=estimator,
        X=X, y=y,
        cv=cv_splits,
        scoring=scoring,
        train_sizes=np.array(train_sizes),
        n_jobs=-1,
    )

    tr_mean, tr_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    va_mean, va_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes_abs, tr_mean, "o-", label="Training score")
    plt.plot(train_sizes_abs, va_mean, "o-", label="Validation score (CV)")
    plt.fill_between(train_sizes_abs, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15)
    plt.fill_between(train_sizes_abs, va_mean - va_std, va_mean + va_std, alpha=0.15)
    plt.title(f"Learning curve — {model_name} ({scoring})")
    plt.xlabel("Training set size")
    plt.ylabel(scoring)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()

    safe_name = _safe_filename(model_name)
    plt.savefig(figures_dir / f"learning_curve_{safe_name}_{scoring}.png",
                dpi=150, bbox_inches="tight")
    plt.show()


def save_pvalue_heatmap(stats_df, model_col_a, model_col_b, pval_col,
                        figures_dir, file_name, title):
    """
    Heatmap of p-values from pairwise statistical tests.
    Cells below α=0.05 are highlighted.
    """
    model_names = sorted(
        set(stats_df[model_col_a].tolist() + stats_df[model_col_b].tolist())
    )
    n = len(model_names)
    idx_map = {name: i for i, name in enumerate(model_names)}
    mat = np.ones((n, n))

    for _, row in stats_df.iterrows():
        i = idx_map[row[model_col_a]]
        j = idx_map[row[model_col_b]]
        mat[i, j] = row[pval_col]
        mat[j, i] = row[pval_col]

    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="p-value")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(model_names, fontsize=9)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.axhline(y=-0.5, color="k", linewidth=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figures_dir / file_name, dpi=150, bbox_inches="tight")
    plt.show()
