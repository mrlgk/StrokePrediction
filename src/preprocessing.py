"""Tiền xử lý dữ liệu Stroke – người phụ trách: A.

HỢP ĐỒNG CẢ NHÓM: KHÔNG đổi tên hàm, tham số hay kiểu trả về khi đã đẩy lên main.
Chỉ được thêm tham số mới có giá trị mặc định.
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

# ---------------------------------------------------------------------------
# Ngưỡng y khoa tham chiếu (chỉ để kiểm tra dữ liệu, mô tả và hiển thị).
# KHÔNG đưa vào pipeline huấn luyện nên không làm đổi mô hình đã lưu.
# ---------------------------------------------------------------------------
# Glucose (mg/dL). Nguồn: CDC (hạ đường huyết <70, nặng <54); ADA (đói: <100
# bình thường, 100-125 tiền ĐTĐ, >=126 ĐTĐ). Dataset không ghi loại xét nghiệm
# hay thời điểm lấy mẫu nên các nhóm chỉ mang tính tham chiếu, không phải chẩn đoán.
GLUCOSE_SEVERE_LOW = 54.0
GLUCOSE_LOW = 70.0
GLUCOSE_PREDIABETES = 100.0
GLUCOSE_DIABETES = 126.0

# BMI (kg/m²). Nguồn: CDC. Người lớn từ 20 tuổi: <18,5 thiếu cân; 18,5-24,9
# bình thường; 25-29,9 thừa cân; >=30 béo phì. Dưới 20 tuổi phải dùng percentile
# theo tuổi và giới nên không áp ngưỡng người lớn.
BMI_UNDERWEIGHT = 18.5
BMI_OVERWEIGHT = 25.0
BMI_OBESITY = 30.0
ADULT_BMI_MIN_AGE = 20

# Giới hạn "không thể xảy ra" để bắt lỗi nhập liệu (lựa chọn kỹ thuật của nhóm,
# không phải khuyến cáo y khoa). Giá trị ngoài khoảng này bị coi là thiếu (NaN).
GLUCOSE_PLAUSIBLE_RANGE = (20.0, 800.0)
BMI_PLAUSIBLE_RANGE = (8.0, 100.0)

GLUCOSE_CATEGORY_ORDER = [
    "Hạ đường huyết nặng (<54)",
    "Hạ đường huyết (54–<70)",
    "Bình thường (70–<100)",
    "Tiền đái tháo đường (100–<126)",
    "Mức đái tháo đường (≥126)",
    "Thiếu",
]
BMI_CATEGORY_ORDER = [
    "Thiếu cân (<18,5)",
    "Bình thường (18,5–<25)",
    "Thừa cân (25–<30)",
    "Béo phì (≥30)",
    "Dưới 20 tuổi (cần percentile)",
    "Thiếu",
]


def load_data() -> pd.DataFrame:
    """Đọc CSV gốc và bỏ cột id (chỉ là mã bệnh nhân, không có giá trị dự đoán)."""
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=["id"])


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Chia train/test có stratify theo stroke, tránh data leakage."""
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def glucose_category(glucose) -> pd.Series:
    """Phân nhóm glucose theo ngưỡng CDC/ADA ở trên. Giá trị thiếu → "Thiếu"."""
    g = pd.Series(glucose, dtype=float).reset_index(drop=True)
    conditions = [
        g < GLUCOSE_SEVERE_LOW,
        g < GLUCOSE_LOW,
        g < GLUCOSE_PREDIABETES,
        g < GLUCOSE_DIABETES,
        g >= GLUCOSE_DIABETES,
    ]
    labels = GLUCOSE_CATEGORY_ORDER[:5]
    return pd.Series(np.select(conditions, labels, default="Thiếu"), index=g.index)


def bmi_category(bmi, age) -> pd.Series:
    """Phân nhóm BMI theo ngưỡng người lớn của CDC, có xét tuổi.

    Người dưới 20 tuổi (hoặc thiếu tuổi) không được xếp nhóm người lớn vì cần
    percentile theo tuổi/giới; BMI thiếu → "Thiếu".
    """
    b = pd.Series(bmi, dtype=float).reset_index(drop=True)
    a = pd.Series(age, dtype=float).reset_index(drop=True)
    adult = a >= ADULT_BMI_MIN_AGE
    conditions = [
        b.isna(),
        ~adult,
        b < BMI_UNDERWEIGHT,
        b < BMI_OVERWEIGHT,
        b < BMI_OBESITY,
        b >= BMI_OBESITY,
    ]
    labels = ["Thiếu", BMI_CATEGORY_ORDER[4], *BMI_CATEGORY_ORDER[:4]]
    return pd.Series(np.select(conditions, labels, default="Thiếu"), index=b.index)


def validate_clinical_ranges(df: pd.DataFrame):
    """Đưa glucose/BMI không thể xảy ra về NaN; trả về (bản sao đã làm sạch, báo cáo).

    Không sửa df đầu vào. Báo cáo là dict {cột: số giá trị bị đổi thành NaN}. Trên
    dữ liệu gốc của đồ án không có giá trị nào bị đổi, nên kết quả huấn luyện
    không thay đổi; hàm này bảo vệ khi nhận dữ liệu mới (ví dụ glucose = 0).
    """
    out = df.copy()
    report = {}
    for col, (low, high) in {
        "avg_glucose_level": GLUCOSE_PLAUSIBLE_RANGE,
        "bmi": BMI_PLAUSIBLE_RANGE,
    }.items():
        values = pd.to_numeric(out[col], errors="coerce")
        bad = values.notna() & ((values < low) | (values > high))
        report[col] = int(bad.sum())
        out[col] = values.mask(bad)
    return out, report


def _to_float(X):
    """Hàm chuyển đổi kiểu dữ liệu ở cấp độ module để hỗ trợ pickle/joblib."""
    return np.asarray(X, dtype=float)

def get_preprocessor() -> ColumnTransformer:
    """Bộ tiền xử lý chưa fit.

    - Biến số: điền median + cờ đánh dấu khuyết thiếu + chuẩn hóa.
    - Biến phân loại: one-hot, bỏ qua danh mục lạ ở dữ liệu mới.
    - Biến nhị phân: ép kiểu số thực, tránh lỗi khi dữ liệu mới từ form gửi dạng chuỗi.
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    binary = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("to_float", FunctionTransformer(_to_float, feature_names_out="one-to-one")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("bin", binary, BINARY_FEATURES),
        ]
    )
