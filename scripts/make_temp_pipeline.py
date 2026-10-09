"""Tạo pipeline TẠM để C thử app và các ca biên – người phụ trách: C.

KHÔNG phải mô hình cuối: dùng Logistic Regression đơn giản, không SMOTE.
Mô hình cuối do B chọn ở tuần 3.

Chạy (ở thư mục gốc repo, .venv đang bật):
    python scripts/make_temp_pipeline.py

File tạo ra: models/stroke_pipeline_temp.pkl  → KHÔNG commit file tạm này.
"""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import RANDOM_STATE, get_preprocessor, load_data, split_data

MODEL_PATH = ROOT / "models" / "stroke_pipeline_temp.pkl"

BASE = {
    "gender": "Male", "age": 50.0, "hypertension": 0, "heart_disease": 0,
    "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
    "avg_glucose_level": 100.0, "bmi": 25.0, "smoking_status": "never smoked",
}

# (tên ca, các trường ghi đè lên BASE)
EDGE_CASES = [
    ("Mặc định của app", {}),
    ("Tuổi rất nhỏ (trẻ em)", {"age": 1.0, "work_type": "children", "ever_married": "No"}),
    ("Tuổi rất lớn", {"age": 100.0}),
    ("Không biết BMI (NaN)", {"bmi": np.nan}),
    ("BMI thấp nhất của form", {"bmi": 10.0}),
    ("BMI cao nhất của form", {"bmi": 100.0}),
    ("Glucose cực cao", {"avg_glucose_level": 400.0}),
    ("Glucose bằng 0", {"avg_glucose_level": 0.0}),
    ("Giới tính Other", {"gender": "Other"}),
    ("Nguy cơ cao tổng hợp", {"age": 80.0, "hypertension": 1, "heart_disease": 1,
                              "avg_glucose_level": 250.0, "bmi": 35.0, "smoking_status": "smokes"}),
    ("Nguy cơ thấp tổng hợp", {"age": 20.0, "avg_glucose_level": 80.0, "bmi": 22.0}),
]


def main() -> int:
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    pipe = Pipeline([
        ("preprocessor", get_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ]).fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    try:
        joblib.dump(pipe, MODEL_PATH)
    except (pickle.PicklingError, AttributeError, TypeError) as e:
        print("KHÔNG lưu được pipeline:", e)
        print("Nguyên nhân thường gặp: src/preprocessing.py còn dùng lambda trong FunctionTransformer.")
        print("Cần A đưa hàm ra cấp module (xem nhận xét review PR).")
        return 1
    pipe = joblib.load(MODEL_PATH)  # thử luôn bước tải lại, giống app.py
    print(f"Đã lưu và tải lại được pipeline tạm: {MODEL_PATH}")
    print("Số cột sau tiền xử lý:", pipe.named_steps["preprocessor"].transform(X_test.head(2)).shape[1])

    rows = []
    for name, override in EDGE_CASES:
        row = {**BASE, **override}
        proba = float(pipe.predict_proba(pd.DataFrame([row]))[0, 1])
        assert 0.0 <= proba <= 1.0 and not np.isnan(proba), f"Xác suất không hợp lệ ở ca: {name}"
        rows.append({"Ca kiểm thử": name, "Xác suất": f"{proba:.1%}"})
    print("\nKết quả các ca biên (mô hình TẠM, chỉ để thử app chạy không lỗi):")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nTất cả ca chạy không lỗi. File tạm riêng, không ghi đè pipeline cuối.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
