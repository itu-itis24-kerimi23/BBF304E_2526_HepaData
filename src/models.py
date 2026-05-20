from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from .config import RANDOM_STATE


def get_baseline_models(random_state: int = RANDOM_STATE):
    return {
        "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
    }


def get_comparison_models(random_state: int = RANDOM_STATE):
    """All five models listed in the proposal."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(50,), max_iter=2000,
            early_stopping=True, random_state=random_state,
        ),
    }


def get_comparison_param_grids():
    """
    Hyperparameter grids for Notebook 03 tuning.
    Parameter names assume a Pipeline step named 'classifier'.
    """
    return {
        "Logistic Regression": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        },
        "SVM": {
            "classifier__C": [0.5, 1.0, 5.0],
            "classifier__gamma": ["scale", "auto"],
            "classifier__class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10],
            "classifier__class_weight": [None, "balanced"],
        },
        "Gradient Boosting": {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [2, 3],
        },
        "MLP": {
            "classifier__hidden_layer_sizes": [(25,), (50,), (50, 25)],
            "classifier__alpha": [0.0001, 0.001],
        },
    }


def build_model_with_params(model_name: str, params: dict, random_state: int = RANDOM_STATE):
    """
    Reconstruct a classifier with specific hyperparameters loaded from saved tuning results.
    Used by Notebook 04 to avoid re-tuning from scratch.
    """
    if model_name == "Logistic Regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            C=params.get("C", 1.0),
            class_weight=params.get("class_weight", None),
        )
    elif model_name == "SVM":
        return SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state,
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            class_weight=params.get("class_weight", None),
        )
    elif model_name == "Random Forest":
        return RandomForestClassifier(
            random_state=random_state,
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            class_weight=params.get("class_weight", None),
        )
    elif model_name == "Gradient Boosting":
        return GradientBoostingClassifier(
            random_state=random_state,
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
        )
    elif model_name == "MLP":
        return MLPClassifier(
            max_iter=2000,
            early_stopping=True,
            random_state=random_state,
            hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (50,))),
            alpha=params.get("alpha", 0.0001),
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_class_weighted_variants(candidate_names: list, random_state: int = RANDOM_STATE):
    """
    Class-weighted versions of the candidate models for imbalance comparison.
    """
    models = {}
    for name in candidate_names:
        if name == "Logistic Regression":
            models[f"{name} (balanced)"] = LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=random_state,
            )
        elif name == "Random Forest":
            models[f"{name} (balanced)"] = RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=random_state,
            )
        elif name == "SVM":
            models[f"{name} (balanced)"] = SVC(
                kernel="rbf", probability=True,
                class_weight="balanced", random_state=random_state,
            )
        elif name == "Gradient Boosting":
            models[f"{name} (balanced)"] = GradientBoostingClassifier(
                random_state=random_state,
            )
        elif name == "MLP":
            models[f"{name} (balanced)"] = MLPClassifier(
                max_iter=2000, early_stopping=True, random_state=random_state,
            )
    return models
