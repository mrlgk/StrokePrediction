"""Ứng dụng minh họa dự đoán nguy cơ đột quỵ – người phụ trách: C.

Chạy:  streamlit run app.py

Khung này chạy được ngay cả khi chưa có mô hình (models/stroke_pipeline.pkl).
Khi B đã chọn xong mô hình cuối và C lưu pipeline, app tự dùng mô hình đó.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "models" / "stroke_pipeline.pkl"
THRESHOLD = 0.5  # B sẽ chọn ngưỡng ở tuần 3, cập nhật giá trị này sau

st.set_page_config(page_title="Stroke Prediction", page_icon="🩺")
st.title("Dự đoán nguy cơ đột quỵ")
st.warning(
    "Đây là mô hình minh họa cho đồ án học tập, **không phải công cụ chẩn đoán y tế**. "
    "Không dùng kết quả này để đưa ra quyết định về sức khỏe."
)


@st.cache_resource
def load_pipeline():
    """Tải pipeline (gồm cả tiền xử lý và mô hình). Trả về None nếu chưa có file."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()
if pipeline is None:
    st.info("Chưa có mô hình (models/stroke_pipeline.pkl). Đang chạy ở chế độ khung: chỉ hiển thị dữ liệu nhập.")

st.subheader("Thông tin bệnh nhân")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Giới tính", ["Male", "Female", "Other"])
    age = st.number_input("Tuổi", min_value=0.0, max_value=120.0, value=50.0, step=1.0)
    hypertension = st.selectbox("Tăng huyết áp", [0, 1], format_func=lambda v: "Có" if v else "Không")
    heart_disease = st.selectbox("Bệnh tim", [0, 1], format_func=lambda v: "Có" if v else "Không")
    ever_married = st.selectbox("Đã từng kết hôn", ["Yes", "No"], format_func=lambda v: "Có" if v == "Yes" else "Không")

with col2:
    work_type = st.selectbox("Loại công việc", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Khu vực sống", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Glucose trung bình (mg/dL)", min_value=0.0, max_value=400.0, value=100.0, step=1.0)
    bmi_unknown = st.checkbox("Không biết BMI")
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1, disabled=bmi_unknown)
    smoking_status = st.selectbox("Tình trạng hút thuốc", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# Tên cột và thứ tự phải khớp với X lúc huấn luyện (sau khi bỏ id và stroke)
input_df = pd.DataFrame([{
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": residence_type,
    "avg_glucose_level": avg_glucose_level,
    "bmi": np.nan if bmi_unknown else bmi,  # pipeline có imputer nên xử lý được giá trị thiếu
    "smoking_status": smoking_status,
}])

with st.expander("Dữ liệu đầu vào gửi vào mô hình"):
    st.dataframe(input_df)

if st.button("Dự đoán"):
    if pipeline is None:
        st.error("Chưa có mô hình để dự đoán.")
    else:
        proba = float(pipeline.predict_proba(input_df)[0, 1])
        st.metric("Xác suất mô hình ước tính", f"{proba:.1%}")
        if proba >= THRESHOLD:
            st.error("Kết quả: NGUY CƠ CAO theo mô hình (theo ngưỡng đã chọn)")
        else:
            st.success("Kết quả: nguy cơ thấp theo mô hình (theo ngưỡng đã chọn)")
        st.caption(
            "Lưu ý: mô hình được huấn luyện với SMOTE trên dữ liệu mất cân bằng nên xác suất "
            "thường cao hơn thực tế; hãy đọc như một điểm số tương đối, không phải xác suất y khoa."
        )