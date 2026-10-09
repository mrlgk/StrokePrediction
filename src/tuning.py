"""Hyperparameter tuning for Random Forest and XGBoost."""

from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from xgboost import XGBClassifier


def get_tuning_models(random_state: int = 42) -> dict:
    """Return models used for hyperparameter tuning."""
    return {
        "Random Forest": RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        ),
    }


def get_tuning_params() -> dict:
    """Return hyperparameter distributions for tuning."""
    return {
        "Random Forest": {
            "model__n_estimators": randint(100, 500),
            "model__max_depth": [None, 5, 10, 15, 20],
            "model__min_samples_split": randint(2, 10),
            "model__min_samples_leaf": randint(1, 5),
            "model__max_features": ["sqrt", "log2"],
        },
        "XGBoost": {
            "model__n_estimators": randint(100, 500),
            "model__max_depth": randint(2, 8),
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "model__min_child_weight": randint(1, 8),
            "model__gamma": [0, 0.1, 0.3, 0.5],
        },
    }