"""Chọn threshold trên OOF, đánh giá holdout một lần và lưu LR pipeline cuối."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import (  # noqa: E402
    compute_metrics,
    compute_permutation_importance,
    get_model_feature_importance,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
    plot_threshold_metrics,
)
from src.models import get_models  # noqa: E402
from src.preprocessing import RANDOM_STATE, get_preprocessor, load_data, split_data  # noqa: E402

RESULTS_DIR = ROOT / "results"
MODEL_PATH = ROOT / "models" / "stroke_pipeline.pkl"


def main() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    # Validate the training-only selection record before doing any work with holdout.
    cv_results_path = RESULTS_DIR / "cv_results.csv"
    if not cv_results_path.is_file():
        raise FileNotFoundError(f"CV results are required before final evaluation: {cv_results_path}")
    cv_results = pd.read_csv(cv_results_path)
    required_cv_columns = {"model", "average_precision_mean"}
    missing_cv_columns = required_cv_columns.difference(cv_results.columns)
    if missing_cv_columns:
        raise ValueError(f"CV results missing columns: {', '.join(sorted(missing_cv_columns))}")
    lr_rows = cv_results.loc[
        cv_results["model"].eq("Logistic Regression"), "average_precision_mean"
    ]
    if len(lr_rows) != 1 or not np.isfinite(float(lr_rows.iloc[0])):
        raise ValueError("CV results must contain exactly one finite Logistic Regression AP score.")
    lr_cv_ap = float(lr_rows.iloc[0])

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    selected = Pipeline(
        [
            ("preprocessor", get_preprocessor()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", get_models(random_state=RANDOM_STATE)["Logistic Regression"]),
        ]
    )

    # Chỉ dùng OOF trên train để khóa threshold. Tập test chưa được chấm điểm ở bước này.
    oof_proba = cross_val_predict(
        selected,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    threshold_rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        prediction = (oof_proba >= threshold).astype(int)
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "recall": recall_score(y_train, prediction, zero_division=0),
                "precision": precision_score(y_train, prediction, zero_division=0),
                "f1": f1_score(y_train, prediction, zero_division=0),
            }
        )
    threshold_results = pd.DataFrame(threshold_rows)
    threshold_results.to_csv(RESULTS_DIR / "threshold_results.csv", index=False)
    # Tie break: highest Recall, then the higher threshold (fewer predicted positives).
    chosen = max(
        threshold_rows,
        key=lambda row: (row["f1"], row["recall"], row["threshold"]),
    )
    threshold = chosen["threshold"]

    # Refit on all training data. The holdout is scored only after model and threshold are fixed.
    selected.fit(X_train, y_train)
    joblib.dump(selected, MODEL_PATH)
    y_proba = selected.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    test_row = {
        "model": "Logistic Regression",
        "threshold": threshold,
        **metrics,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    pd.DataFrame([test_row]).to_csv(RESULTS_DIR / "test_results.csv", index=False)

    plot_confusion_matrix(
        y_test, y_pred, title="Logistic Regression · Holdout", save_as="confusion_matrix.png"
    ).clear()
    plot_roc_curve(
        y_test, y_proba, title="Logistic Regression · Holdout ROC", save_as="roc_curve.png"
    ).clear()
    plot_pr_curve(
        y_test, y_proba, title="Logistic Regression · Holdout Precision-Recall", save_as="pr_curve.png"
    ).clear()
    plot_threshold_metrics(
        threshold_results,
        title="Logistic Regression · OOF threshold trên training",
        save_as="threshold_metrics.png",
    ).clear()

    feature_names = selected.named_steps["preprocessor"].get_feature_names_out()
    native_importance = get_model_feature_importance(selected, feature_names)
    native_importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    plot_feature_importance(
        native_importance,
        title="Logistic Regression · Trị tuyệt đối hệ số",
        save_as="feature_importance.png",
    ).clear()

    permutation = compute_permutation_importance(
        selected,
        X_train,
        y_train,
        scoring="average_precision",
        n_repeats=10,
        random_state=RANDOM_STATE,
    )
    permutation.to_csv(RESULTS_DIR / "permutation_importance.csv", index=False)
    plot_feature_importance(
        permutation,
        value_column="importance_mean",
        title="Logistic Regression · Permutation importance trên training",
        save_as="permutation_importance.png",
    ).clear()

    metadata = {
        "model": "Logistic Regression",
        "selection_metric": "average_precision_cv",
        "selection_value": lr_cv_ap,
        "threshold_method": "max F1 on stratified 5-fold out-of-fold predictions; ties prefer recall",
        "threshold": threshold,
        "random_state": RANDOM_STATE,
        "test_set_used_for_model_or_threshold_selection": False,
        "test_set_used_for_feature_importance": False,
    }
    (RESULTS_DIR / "final_model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Chosen threshold from training OOF:", threshold)
    print("OOF Recall / Precision / F1:", chosen["recall"], chosen["precision"], chosen["f1"])
    print("Holdout metrics:", json.dumps(test_row, ensure_ascii=False))
    print("Saved final pipeline:", MODEL_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
