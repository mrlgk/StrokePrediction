"""Đánh giá mô hình người phụ trách: C.

Dùng chung cho notebook 03_Modeling (B) và cho báo cáo. Mọi hàm nhận
y_true (0/1 thật) và y_proba (xác suất lớp 1, tức predict_proba(X)[:, 1]).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Tính 5 chỉ số chính. KHÔNG gồm accuracy (không đáng tin với dữ liệu mất cân bằng)."""
    return {
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix", save_as: str | None = None):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Không stroke", "Stroke"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    if save_as:
        RESULTS_DIR.mkdir(exist_ok=True)
        fig.savefig(RESULTS_DIR / save_as, dpi=150)
    return fig


def plot_roc_curve(y_true, y_proba, title: str = "ROC Curve", save_as: str | None = None):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    if save_as:
        RESULTS_DIR.mkdir(exist_ok=True)
        fig.savefig(RESULTS_DIR / save_as, dpi=150)
    return fig


def plot_pr_curve(y_true, y_proba, title: str = "Precision-Recall Curve", save_as: str | None = None):
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    if save_as:
        RESULTS_DIR.mkdir(exist_ok=True)
        fig.savefig(RESULTS_DIR / save_as, dpi=150)
    return fig


def evaluate_at_thresholds(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """Bảng Recall/Precision/F1 tại nhiều ngưỡng, để B chọn ngưỡng quyết định ở tuần 3.

    Ngưỡng mặc định 0.5 thường không phù hợp với bài toán mất cân bằng: hạ ngưỡng
    thường tăng Recall (bắt được nhiều ca dương hơn) nhưng giảm Precision.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 2),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })
    return pd.DataFrame(rows).round(3)


def save_cv_results(rows: list[dict], filename: str = "cv_results.csv") -> Path:
    """B gọi hàm này để lưu bảng kết quả cross-validation của 6 mô hình."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / filename
    pd.DataFrame(rows).round(3).to_csv(path, index=False)
    return path