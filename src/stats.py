"""
Statistical significance tests for model comparison.

Implements the course requirement of proving validity of results through
statistical tests beyond reporting mean ± std.

Two test types:
  1. Wilcoxon signed-rank test on paired 5×2 CV scores (non-parametric,
     appropriate for small n=10 samples from RepeatedStratifiedKFold).
  2. McNemar's test on binary test-set predictions (compares whether two
     models make the same classification errors).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test on CV fold scores
# ---------------------------------------------------------------------------

def run_wilcoxon_pairwise(raw_scores: dict, metric: str = "recall") -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank tests between all model pairs on a given metric.

    Parameters
    ----------
    raw_scores : dict
        Output of run_cross_validation: {model_name: {metric_name: np.array}}
    metric     : str
        Scoring key from get_scoring_dict (e.g. "recall", "f1", "balanced_accuracy").

    Returns
    -------
    DataFrame with columns: Model A, Model B, W Statistic, p-value, Significant (α=0.05)

    Notes
    -----
    With 10 paired observations (5×2 CV) the Wilcoxon test has limited power.
    Non-significant results should be interpreted as "insufficient evidence to
    distinguish" rather than "models are equivalent".
    """
    model_names = list(raw_scores.keys())
    rows = []

    for name_a, name_b in combinations(model_names, 2):
        scores_a = raw_scores[name_a][metric]
        scores_b = raw_scores[name_b][metric]
        diff = scores_a - scores_b

        if np.all(diff == 0):
            w_stat, p_val = np.nan, 1.0
        else:
            try:
                w_stat, p_val = stats.wilcoxon(diff, alternative="two-sided")
            except ValueError:
                w_stat, p_val = np.nan, np.nan

        rows.append({
            "Model A": name_a,
            "Model B": name_b,
            "Metric": metric,
            "Mean A": scores_a.mean(),
            "Mean B": scores_b.mean(),
            "W Statistic": w_stat,
            "p-value": p_val,
            "Significant (α=0.05)": (p_val < 0.05) if not np.isnan(p_val) else False,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("p-value").reset_index(drop=True)
    return df


def run_wilcoxon_multiple_metrics(raw_scores: dict, metrics=None) -> pd.DataFrame:
    """
    Run pairwise Wilcoxon tests across several metrics and stack the results.
    """
    if metrics is None:
        metrics = ["recall", "balanced_accuracy", "f1", "roc_auc"]

    all_rows = []
    for m in metrics:
        df = run_wilcoxon_pairwise(raw_scores, metric=m)
        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True).sort_values(["Metric", "p-value"])


# ---------------------------------------------------------------------------
# McNemar's test on test-set predictions
# ---------------------------------------------------------------------------

def _mcnemar_statistic(y_true, pred_a, pred_b):
    """
    Compute McNemar's test statistic and p-value (with continuity correction).

    Contingency table:
        b = A wrong,  B correct
        c = A correct, B wrong
    H0: both classifiers make the same number of errors.
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    b = np.sum(~correct_a & correct_b)   # A wrong, B correct
    c = np.sum(correct_a & ~correct_b)   # A correct, B wrong

    if (b + c) == 0:
        return np.nan, 1.0, int(b), int(c)

    # With continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = stats.chi2.sf(chi2, df=1)
    return chi2, p_val, int(b), int(c)


def run_mcnemar_pairwise(predictions: dict, y_test) -> pd.DataFrame:
    """
    Pairwise McNemar's tests between all model pairs.

    Parameters
    ----------
    predictions : dict
        {model_name: y_pred array} — output of run_test_evaluation.
    y_test      : array-like

    Returns
    -------
    DataFrame with columns: Model A, Model B, b, c, Chi2, p-value, Significant (α=0.05)
    """
    y_true = np.asarray(y_test)
    model_names = list(predictions.keys())
    rows = []

    for name_a, name_b in combinations(model_names, 2):
        pred_a = np.asarray(predictions[name_a])
        pred_b = np.asarray(predictions[name_b])

        chi2, p_val, b, c = _mcnemar_statistic(y_true, pred_a, pred_b)

        rows.append({
            "Model A": name_a,
            "Model B": name_b,
            "b (A wrong, B right)": b,
            "c (A right, B wrong)": c,
            "Chi2": chi2,
            "p-value": p_val,
            "Significant (α=0.05)": (p_val < 0.05) if not np.isnan(p_val) else False,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("p-value").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_pvalue(p: float, alpha: float = 0.05) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "< 0.001 *"
    sig = " *" if p < alpha else ""
    return f"{p:.3f}{sig}"


def print_significance_summary(wilcoxon_df: pd.DataFrame, mcnemar_df: pd.DataFrame):
    print("=" * 60)
    print("Wilcoxon pairwise tests (CV scores)")
    print("=" * 60)
    print(wilcoxon_df[["Model A", "Model B", "Metric", "p-value", "Significant (α=0.05)"]].to_string(index=False))
    print()
    print("=" * 60)
    print("McNemar's tests (test-set predictions)")
    print("=" * 60)
    print(mcnemar_df[["Model A", "Model B", "b (A wrong, B right)", "c (A right, B wrong)",
                       "p-value", "Significant (α=0.05)"]].to_string(index=False))
