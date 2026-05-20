from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

from .preprocessing import get_transformed_feature_names


def get_logistic_regression_coefficients(fitted_pipeline, numeric_features, categorical_features):
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]

    feature_names = get_transformed_feature_names(preprocessor, numeric_features, categorical_features)
    coefficients = classifier.coef_[0]

    return pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Absolute Coefficient": np.abs(coefficients),
    }).sort_values("Absolute Coefficient", ascending=False).reset_index(drop=True)


def get_random_forest_feature_importance(fitted_pipeline, numeric_features, categorical_features):
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]

    feature_names = get_transformed_feature_names(preprocessor, numeric_features, categorical_features)
    importances = classifier.feature_importances_

    return pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)


def get_permutation_importance(fitted_pipeline, X_test, y_test,
                                scoring="f1", random_state=42, n_repeats=20):
    """
    Permutation importance on the original (pre-preprocessing) feature columns.
    Measures how much test-set performance drops when each input feature is shuffled.
    """
    result = permutation_importance(
        fitted_pipeline, X_test, y_test,
        scoring=scoring, n_repeats=n_repeats,
        random_state=random_state, n_jobs=-1,
    )

    return pd.DataFrame({
        "Feature": X_test.columns.tolist(),
        "Importance Mean": result.importances_mean,
        "Importance Std": result.importances_std,
    }).sort_values("Importance Mean", ascending=False).reset_index(drop=True)


def save_feature_importance_plot(importance_df, value_col, title, figures_dir, file_name, top_n=10):
    from .plots import _safe_filename  # reuse sanitiser
    plot_df = importance_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["Feature"], plot_df[value_col], color="#378ADD")
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    plt.savefig(figures_dir / file_name, dpi=150, bbox_inches="tight")
    plt.show()


def get_misclassified_examples(fitted_pipeline, X_test, y_test, include_probability=True):
    y_pred = fitted_pipeline.predict(X_test)
    result = X_test.copy()
    result["true_label"] = y_test.values
    result["predicted_label"] = y_pred

    if include_probability and hasattr(fitted_pipeline, "predict_proba"):
        result["prob_liver_disease"] = fitted_pipeline.predict_proba(X_test)[:, 1]

    return result[result["true_label"] != result["predicted_label"]].copy().reset_index(drop=True)
