"""Báo cáo chất lượng dữ liệu theo ngưỡng y khoa (glucose, BMI, nhóm thiếu BMI).

Chạy (ở thư mục gốc repo):
    python scripts/clinical_data_report.py

Ghi ra: results/clinical_data_report.md. Chỉ đọc dữ liệu, không đụng tới mô hình.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.preprocessing import (  # noqa: E402
    BMI_CATEGORY_ORDER,
    GLUCOSE_CATEGORY_ORDER,
    bmi_category,
    glucose_category,
    get_preprocessor,
    load_data,
    validate_clinical_ranges,
)

OUT_PATH = ROOT / "results" / "clinical_data_report.md"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%".replace(".", ",")


def table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> int:
    df = load_data()

    # 1) Làm sạch không được đổi dữ liệu huấn luyện gốc.
    cleaned, report = validate_clinical_ranges(df)
    unchanged = df[["avg_glucose_level", "bmi"]].equals(cleaned[["avg_glucose_level", "bmi"]])
    X = df.drop(columns=["stroke"])
    a = get_preprocessor().fit_transform(X)
    b = get_preprocessor().fit_transform(cleaned.drop(columns=["stroke"]))
    same_matrix = bool(np.allclose(a, b, equal_nan=True))

    # 2) Nhóm glucose.
    df["glucose_group"] = glucose_category(df["avg_glucose_level"]).values
    g = (
        df.groupby("glucose_group")
        .stroke.agg(["count", "sum", "mean"])
        .reindex(GLUCOSE_CATEGORY_ORDER)
        .dropna()
    )
    g_tbl = pd.DataFrame({
        "Nhóm glucose (mg/dL)": g.index,
        "Số ca": g["count"].astype(int).values,
        "Số ca stroke": g["sum"].astype(int).values,
        "Tỷ lệ stroke": [pct(v) for v in g["mean"]],
    })

    # 3) Nhóm BMI (có xét tuổi).
    df["bmi_group"] = bmi_category(df["bmi"], df["age"]).values
    b = (
        df.groupby("bmi_group")
        .stroke.agg(["count", "sum", "mean"])
        .reindex(BMI_CATEGORY_ORDER)
        .dropna()
    )
    b_tbl = pd.DataFrame({
        "Nhóm BMI (kg/m²)": b.index,
        "Số ca": b["count"].astype(int).values,
        "Số ca stroke": b["sum"].astype(int).values,
        "Tỷ lệ stroke": [pct(v) for v in b["mean"]],
    })

    # 4) Hồ sơ nhóm thiếu BMI.
    df["bmi_missing"] = df["bmi"].isna()
    df["age_group"] = pd.cut(df["age"], [0, 18, 40, 60, 80, 120])
    rows = []
    for col, label in [
        ("age_group", "Tuổi"), ("stroke", "Stroke"), ("hypertension", "Tăng huyết áp"),
        ("heart_disease", "Bệnh tim"), ("smoking_status", "Hút thuốc"),
    ]:
        t = df.groupby(col, observed=True).bmi_missing.agg(["sum", "count", "mean"])
        for k, r in t.iterrows():
            rows.append({
                "Đặc điểm": label, "Nhóm": str(k), "Số ca thiếu BMI": int(r["sum"]),
                "Tổng": int(r["count"]), "Tỷ lệ thiếu": pct(r["mean"]),
            })
    m_tbl = pd.DataFrame(rows)

    md = f"""# Báo cáo dữ liệu theo ngưỡng y khoa

Tạo bởi `scripts/clinical_data_report.py`. Ngưỡng định nghĩa trong `src/preprocessing.py`
(glucose: CDC/ADA; BMI: CDC). Các nhóm chỉ mang tính tham chiếu, không phải chẩn đoán:
dataset không ghi loại xét nghiệm glucose hay thời điểm lấy mẫu.

## Kiểm tra giá trị ngoài khoảng hợp lý
- Số giá trị bị đổi thành NaN: glucose = {report['avg_glucose_level']}, BMI = {report['bmi']}.
- Dữ liệu sau làm sạch giống dữ liệu gốc: {'có' if unchanged else 'KHÔNG'}.
- Ma trận sau tiền xử lý giống nhau: {'có' if same_matrix else 'KHÔNG'}.

## Tỷ lệ stroke theo nhóm glucose
{table(g_tbl)}

## Tỷ lệ stroke theo nhóm BMI (người lớn từ 20 tuổi dùng ngưỡng CDC)
{table(b_tbl)}

Nhóm "Dưới 20 tuổi" không xếp theo ngưỡng người lớn vì BMI trẻ em phải đọc theo percentile tuổi/giới.

## Hồ sơ nhóm thiếu BMI ({int(df['bmi_missing'].sum())} ca, {pct(df['bmi_missing'].mean())})
{table(m_tbl)}

Thiếu BMI không ngẫu nhiên: tập trung ở người lớn tuổi, người từng bị stroke và người có
bệnh nền. Dataset không ghi lý do thiếu; đây là mối liên quan, không phải nguyên nhân.
"""
    OUT_PATH.write_text(md, encoding="utf-8")
    print(f"Đã ghi {OUT_PATH}")
    print("Không đổi dữ liệu gốc:", unchanged, "| ma trận giống:", same_matrix, "| NaN mới:", report)
    return 0 if unchanged and same_matrix else 1


if __name__ == "__main__":
    raise SystemExit(main())
