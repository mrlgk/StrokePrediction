"""Tiền xử lý dữ liệu Stroke – người phụ trách: A.

"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "healthcare-dataset-stroke-data.csv"
RANDOM_STATE = 42

NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL_FEATURES = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
BINARY_FEATURES = ["hypertension", "heart_disease"]


def load_data() -> pd.DataFrame:
    """Đọc CSV gốc và bỏ cột id (chỉ là mã bệnh nhân, không có giá trị dự đoán)."""
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=["id"])


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Chia train/test có stratify theo stroke, tránh data leakage."""
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def get_preprocessor() -> ColumnTransformer:
    """Bộ tiền xử lý chưa fit: median+scale cho số, one-hot cho phân loại, ép kiểu số cho nhị phân."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    binary = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("to_float", FunctionTransformer(lambda X: np.asarray(X, dtype=float))),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("bin", binary, BINARY_FEATURES),
        ]
    )