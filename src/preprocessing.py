from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .config import CATEGORICAL_FEATURES


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


def create_preprocessor(X_train, categorical_features=None):
    """
    Build a preprocessing pipeline from the TRAINING set column structure.

    Always call this with X_train (not the full X) so that feature-name detection
    is based only on training data, even though no statistics are computed here.

    Numeric transformer : median imputation → StandardScaler
    Categorical transformer : most-frequent imputation → OneHotEncoder (drop first)
    """
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    categorical_features = [col for col in categorical_features if col in X_train.columns]
    numeric_features = [col for col in X_train.columns if col not in categorical_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_one_hot_encoder()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_features, categorical_features


def get_transformed_feature_names(preprocessor, numeric_features, categorical_features):
    feature_names = list(numeric_features)
    if categorical_features:
        cat_pipeline = preprocessor.named_transformers_["cat"]
        encoder = cat_pipeline.named_steps["onehot"]
        feature_names.extend(encoder.get_feature_names_out(categorical_features).tolist())
    return feature_names
