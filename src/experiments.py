from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit

from .evaluation import build_pipeline, run_test_evaluation


def run_class_weight_comparison(model_variants, preprocessor, X_train, X_test, y_train, y_test):
    """Compare default candidate models against class-weighted variants."""
    estimators = {
        name: build_pipeline(clone(preprocessor), clone(clf))
        for name, clf in model_variants.items()
    }
    return run_test_evaluation(estimators, X_train, X_test, y_train, y_test, verbose=True)


def run_smote_comparison(models, preprocessor, X_train, X_test, y_train, y_test, random_state=42):
    """
    Compare candidate models with and without SMOTE oversampling.
    SMOTE is placed inside the imblearn Pipeline after preprocessing so it only
    operates on training data — no test-set leakage.
    """
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
    except ImportError:
        msg = "imbalanced-learn not installed. Run: pip install imbalanced-learn"
        print(msg)
        return pd.DataFrame({"Message": [msg]}), {}, {}

    estimators = {}
    for model_name, classifier in models.items():
        estimators[f"{model_name} (no SMOTE)"] = Pipeline(steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", clone(classifier)),
        ])
        estimators[f"{model_name} (SMOTE)"] = ImbPipeline(steps=[
            ("preprocessor", clone(preprocessor)),
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", clone(classifier)),
        ])

    return run_test_evaluation(estimators, X_train, X_test, y_train, y_test, verbose=True)


def run_ablation_study(models, preprocessor_factory, X_train, X_test, y_train, y_test, ablation_sets):
    """
    Controlled feature-group removal experiments.
    preprocessor_factory is called with the ablated X_train subset.
    """
    rows = []
    all_features = X_train.columns.tolist()

    for ablation_name, features_to_remove in ablation_sets.items():
        remaining = [col for col in all_features if col not in features_to_remove]

        X_tr = X_train[remaining].copy()
        X_te = X_test[remaining].copy()

        # FIXED: preprocessor sees only the ablated training columns
        preprocessor, _, _ = preprocessor_factory(X_tr)

        estimators = {
            name: build_pipeline(clone(preprocessor), clone(clf))
            for name, clf in models.items()
        }

        res_df, _, _ = run_test_evaluation(
            estimators, X_tr, X_te, y_train, y_test, verbose=False,
        )
        res_df["Ablation Setting"] = ablation_name
        res_df["Removed Features"] = ", ".join(features_to_remove) if features_to_remove else "None"
        rows.append(res_df)

    return pd.concat(rows, ignore_index=True)


def run_training_size_experiment(
    models,
    preprocessor_factory,
    X_train,
    X_test,
    y_train,
    y_test,
    train_fractions=(0.25, 0.50, 0.75, 1.00),
    random_state=42,
):
    """
    Train on increasing fractions of the training set (25 / 50 / 75 / 100 %).

    FIXED: subsets are drawn using StratifiedShuffleSplit to preserve the
    class ratio at every fraction. The original code used plain random sampling
    which could produce skewed class distributions on an imbalanced dataset.
    """
    rows = []
    n_train = len(X_train)

    for fraction in train_fractions:
        if fraction >= 1.0:
            X_sub = X_train.copy()
            y_sub = y_train.copy()
        else:
            subset_size = max(4, int(n_train * fraction))
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=n_train - subset_size,
                random_state=random_state,
            )
            idx_keep, _ = next(splitter.split(X_train, y_train))
            X_sub = X_train.iloc[idx_keep].copy()
            y_sub = y_train.iloc[idx_keep].copy()

        # FIXED: preprocessor built from the subset, not full X
        preprocessor, _, _ = preprocessor_factory(X_sub)
        estimators = {
            name: build_pipeline(clone(preprocessor), clone(clf))
            for name, clf in models.items()
        }

        res_df, _, _ = run_test_evaluation(
            estimators, X_sub, X_test, y_sub, y_test, verbose=False,
        )
        res_df["Training Fraction"] = fraction
        res_df["Training Size"] = len(X_sub)
        rows.append(res_df)

    return pd.concat(rows, ignore_index=True)
