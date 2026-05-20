from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    GridSearchCV,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    make_scorer,
)


# ---------------------------------------------------------------------------
# Cross-validation setup
# ---------------------------------------------------------------------------

def make_tuning_cv(n_splits: int = 5, random_state: int = 42):
    """
    Inner CV used exclusively inside GridSearchCV for hyperparameter tuning.

    This is a plain StratifiedKFold and is kept SEPARATE from the evaluation CV
    to avoid the double-dipping bias that arises when the same splits are used
    for both selecting hyperparameters and estimating generalisation performance.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def make_eval_cv_splits(X, y, n_splits: int = 2, n_repeats: int = 5, random_state: int = 42):
    """
    Outer CV used exclusively for performance evaluation (cross_validate).

    5 × 2 repeated stratified K-fold → 10 held-out scores per model.
    The resulting list of (train_idx, test_idx) tuples is generated once and
    reused across all models so that every model sees identical folds.

    These splits must NOT be passed to GridSearchCV (that uses make_tuning_cv).
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return list(cv.split(X, y))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def get_scoring_dict():
    return {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "mcc": make_scorer(matthews_corrcoef),
    }


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def build_pipeline(preprocessor, classifier):
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def _predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_score=None):
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }

    if y_score is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_score)
        metrics["PR-AUC"] = average_precision_score(y_true, y_score)
    else:
        metrics["ROC-AUC"] = np.nan
        metrics["PR-AUC"] = np.nan

    return metrics


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def tune_models(models, param_grids, preprocessor, X_train, y_train, n_jobs=-1):
    """
    Tune hyperparameters using the dedicated INNER cross-validation (make_tuning_cv).

    The tuning CV is created fresh inside this function and is completely
    independent of the evaluation CV splits produced by make_eval_cv_splits.
    This separation eliminates the double-dipping bias present in the original code.

    Returns
    -------
    tuning_results_df : DataFrame with best scores and params per model
    tuned_estimators  : dict of {model_name: fitted Pipeline with best hyperparams}
    best_params       : dict of {model_name: {bare_param_name: value}} for JSON export
    """
    tuning_cv = make_tuning_cv(random_state=42)
    tuning_rows = []
    tuned_estimators = {}
    best_params = {}

    for model_name, classifier in models.items():
        print("=" * 60)
        print(f"Tuning: {model_name}")

        pipeline = build_pipeline(clone(preprocessor), clone(classifier))
        param_grid = param_grids.get(model_name, {})

        if param_grid:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="balanced_accuracy",
                cv=tuning_cv,           # <-- inner CV only, never eval splits
                refit=True,
                n_jobs=n_jobs,
                return_train_score=False,
            )
            t0 = time.perf_counter()
            search.fit(X_train, y_train)
            elapsed = time.perf_counter() - t0

            tuned_estimators[model_name] = search.best_estimator_

            # Extract bare parameter names (strip "classifier__" prefix)
            raw = {
                k.replace("classifier__", ""): v
                for k, v in search.best_params_.items()
            }
            best_params[model_name] = raw

            tuning_rows.append({
                "Model": model_name,
                "Best CV Score (Balanced Acc)": search.best_score_,
                "Best Params": search.best_params_,
                "Tuning Time (s)": elapsed,
            })
        else:
            t0 = time.perf_counter()
            pipeline.fit(X_train, y_train)
            elapsed = time.perf_counter() - t0

            tuned_estimators[model_name] = pipeline
            best_params[model_name] = {}
            tuning_rows.append({
                "Model": model_name,
                "Best CV Score (Balanced Acc)": np.nan,
                "Best Params": {},
                "Tuning Time (s)": elapsed,
            })

    return pd.DataFrame(tuning_rows), tuned_estimators, best_params


# ---------------------------------------------------------------------------
# Evaluation CV
# ---------------------------------------------------------------------------

def run_cross_validation(estimators, X_train, y_train, eval_cv_splits, n_jobs=-1):
    """
    5×2 cross-validation using the evaluation splits (separate from tuning CV).

    Returns
    -------
    summary_df  : DataFrame with per-model mean and std for each metric
    raw_scores  : dict {model_name: {metric_name: np.array of 10 fold scores}}
                  Used by stats.py for significance testing.
    """
    scoring = get_scoring_dict()
    summary_rows = []
    raw_scores = {}

    for model_name, estimator in estimators.items():
        print("=" * 60)
        print(f"5×2 CV evaluation: {model_name}")

        scores = cross_validate(
            estimator=clone(estimator),   # clone so original estimator is unchanged
            X=X_train,
            y=y_train,
            cv=eval_cv_splits,            # <-- evaluation splits only
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=False,
        )

        raw_scores[model_name] = {}
        row = {"Model": model_name}
        for metric_name in scoring:
            values = scores[f"test_{metric_name}"]
            raw_scores[model_name][metric_name] = values
            pretty = metric_name.replace("_", " ").title()
            row[f"CV {pretty} Mean"] = values.mean()
            row[f"CV {pretty} Std"] = values.std()

        summary_rows.append(row)

    return pd.DataFrame(summary_rows), raw_scores


# ---------------------------------------------------------------------------
# Test set evaluation
# ---------------------------------------------------------------------------

def run_test_evaluation(estimators, X_train, X_test, y_train, y_test, verbose=True):
    """
    Fit each estimator on full training data, evaluate once on the held-out test set.
    Training and inference times are recorded.

    Returns
    -------
    results_df         : DataFrame of metrics per model
    trained_estimators : dict of {model_name: fitted Pipeline}
    predictions        : dict of {model_name: y_pred} for McNemar's test
    """
    rows = []
    trained_estimators = {}
    predictions = {}

    for model_name, estimator in estimators.items():
        print("=" * 60)
        print(f"Test evaluation: {model_name}")

        model = clone(estimator)

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = time.perf_counter() - t0

        y_score = _predict_scores(model, X_test)
        metrics = compute_metrics(y_test, y_pred, y_score)
        metrics = {"Model": model_name, **metrics,
                   "Training Time (s)": train_time,
                   "Inference Time (s)": infer_time}

        if verbose:
            print(classification_report(y_test, y_pred, zero_division=0))
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=[0, 1]))

        rows.append(metrics)
        trained_estimators[model_name] = model
        predictions[model_name] = y_pred

    return pd.DataFrame(rows), trained_estimators, predictions


# ---------------------------------------------------------------------------
# Selection score
# ---------------------------------------------------------------------------

def add_selection_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite score for candidate selection: equal weight on recall, F1,
    balanced accuracy, and ROC-AUC. Used only as a screening heuristic,
    not as a final clinical metric.
    """
    result = df.copy()
    if "CV Recall Mean" in result.columns:
        result["Selection Score"] = (
            0.25 * result["CV Recall Mean"]
            + 0.25 * result["CV F1 Mean"]
            + 0.25 * result["CV Balanced Accuracy Mean"]
            + 0.25 * result["CV Roc Auc Mean"]
        )
    else:
        result["Selection Score"] = (
            0.25 * result["Recall"]
            + 0.25 * result["F1-score"]
            + 0.25 * result["Balanced Accuracy"]
            + 0.25 * result["ROC-AUC"]
        )
    return result


# ---------------------------------------------------------------------------
# Saving / loading tuned estimators
# ---------------------------------------------------------------------------

def save_candidate_models(unfitted_candidates: dict, models_dir):
    """
    Save clones of the tuned-but-unfitted candidate pipelines so Notebook 04
    can load them without re-running GridSearchCV.
    """
    path = models_dir / "candidate_models.pkl"
    joblib.dump(unfitted_candidates, path)
    print(f"Candidate models saved → {path}")
    return path


def load_candidate_models(models_dir) -> dict:
    path = models_dir / "candidate_models.pkl"
    models = joblib.load(path)
    print(f"Candidate models loaded from {path}")
    return models


def save_best_params(best_params: dict, models_dir):
    """Save best hyperparameters as JSON for human inspection."""
    path = models_dir / "best_params.json"
    with open(path, "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"Best params saved → {path}")


# ---------------------------------------------------------------------------
# Seed robustness
# ---------------------------------------------------------------------------

def run_seed_robustness(models, preprocessor_factory, X, y, seeds, test_size=0.2):
    """
    Repeat train-test evaluation with multiple random splits.
    preprocessor_factory is called with X_train (not full X) for each seed.
    """
    from .data import create_train_test_split

    rows = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = create_train_test_split(
            X, y, test_size=test_size, random_state=seed,
        )
        # FIXED: pass X_train (not full X) to the preprocessor factory
        preprocessor, _, _ = preprocessor_factory(X_train)

        estimators = {
            name: build_pipeline(clone(preprocessor), clone(clf))
            for name, clf in models.items()
        }

        results_df, _, _ = run_test_evaluation(
            estimators, X_train, X_test, y_train, y_test, verbose=False,
        )
        results_df["Seed"] = seed
        rows.append(results_df)

    all_results = pd.concat(rows, ignore_index=True)
    numeric_cols = all_results.select_dtypes(include=[np.number]).columns
    summary = (
        all_results.groupby("Model")[numeric_cols]
        .agg(["mean", "std"])
    )
    summary.columns = [" ".join(col).strip() for col in summary.columns]
    summary = summary.reset_index()

    return all_results, summary
