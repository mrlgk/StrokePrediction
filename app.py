"""
StrokeCare AI - Giao diện đánh giá nguy cơ đột quỵ
Chạy: streamlit run app.py

Giao diện theo phong cách phần mềm quản lý y tế:
- Nền xám xanh nhạt
- Card trắng/xám nhạt
- Hospital Blue làm màu nhấn
- Ba khung thông tin hiển thị đồng thời trên cùng một trang, không dùng tab
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = Path(__file__).resolve().parent / "models" / "stroke_pipeline.pkl"
THRESHOLD = 0.50

st.set_page_config(
    page_title="StrokeCare AI | Đánh giá nguy cơ đột quỵ",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MEDICAL / HOSPITAL UI
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;600;700&display=swap');

    :root {
        --hospital-blue: #2f6f95;
        --hospital-blue-dark: #245a7a;
        --hospital-cyan: #239bc0;
        --page-bg: #eef2f5;
        --card-bg: #ffffff;
        --field-bg: #e9eef2;
        --field-bg-hover: #dde4e9;
        --field-border: #c9d5dd;
        --field-text: #243e50;
        --section-bg: #e7f0f6;
        --border: #d5dfe6;
        --text: #263f50;
        --text-dark: #1f3545;
        --muted: #647986;
        --white: #ffffff;
    }

    html, body, [class*="css"] {
        font-family: 'Roboto', Arial, sans-serif !important;
    }

    .stApp {
        background: var(--page-bg) !important;
        color: var(--text) !important;
    }

    .block-container {
        max-width: 100% !important;
        padding: 0.45rem 1.25rem 1.6rem 1.25rem !important;
    }

    /* Remove Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { background: transparent !important; }

    /* ========================================================
       SIDEBAR
       ======================================================== */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #d7e0e6 !important;
    }

    section[data-testid="stSidebar"] > div {
        padding: 1rem 0.8rem 1rem 0.8rem !important;
    }

    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 5px 4px 14px 4px;
        margin-bottom: 12px;
        border-bottom: 1px solid #dfe6ea;
    }

    .sidebar-logo {
        width: 42px;
        height: 42px;
        border-radius: 4px;
        background: linear-gradient(145deg, #2f6f95, #238dad);
        color: white !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        box-shadow: 0 2px 5px rgba(30, 75, 100, .16);
    }

    .sidebar-brand-name {
        color: #17658f !important;
        font-size: 16px;
        font-weight: 700;
        line-height: 1.1;
    }

    .sidebar-brand-sub {
        color: #667c89 !important;
        font-size: 9px;
        margin-top: 3px;
    }

    .sidebar-heading {
        color: #17658f !important;
        font-size: 17px;
        font-weight: 700;
        margin: 13px 3px 13px 3px;
    }

    .sidebar-card {
        background: #f9fbfc;
        border: 1px solid #d9e2e8;
        border-radius: 4px;
        padding: 12px 13px;
        margin: 10px 3px;
        font-size: 12px;
        line-height: 1.55;
    }

    .sidebar-card-title {
        color: #205f82 !important;
        font-weight: 700;
        font-size: 13px;
        margin-bottom: 4px;
    }

    .sidebar-card-text {
        color: #526a79 !important;
    }

    .sidebar-card-meta {
        color: #6f8591 !important;
        font-size: 11px;
        margin-top: 5px;
    }

    /* ========================================================
       TOP BAR
       ======================================================== */
    .topbar {
        height: 48px;
        background: #ffffff;
        border-bottom: 1px solid #dce4e9;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 12px;
        margin: -0.55rem -1.25rem 12px -1.25rem;
    }

    .breadcrumb {
        color: #5e7380;
        font-size: 13px;
    }

    .breadcrumb .home {
        color: #1f83b0;
        font-weight: 700;
    }

    .breadcrumb .current {
        color: #527088;
        font-weight: 600;
    }

    .deploy {
        color: #75909f;
        font-size: 11px;
    }

    /* ========================================================
       PAGE TITLE
       ======================================================== */
    .page-title {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 9px 0 13px 0;
        padding-left: 2px;
    }

    .page-title-icon {
        width: 31px;
        height: 31px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 4px;
        background: #d8eaf2;
        border: 1px solid #c3dce8;
        color: #21769d;
        font-size: 17px;
        font-weight: 700;
    }

    .page-title-text {
        color: #174f70 !important;
        font-size: 21px;
        font-weight: 700;
        letter-spacing: .005em;
    }

    /* ========================================================
       DISCLAIMER
       ======================================================== */
    div[data-testid="stAlert"] {
        border-radius: 4px !important;
        border-width: 1px !important;
        font-size: 12px !important;
        line-height: 1.55 !important;
    }

    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] li {
        color: #31576c !important;
    }

    div[data-testid="stAlert"] strong {
        color: #15577b !important;
        font-weight: 700 !important;
    }

    /* ========================================================
       HOSPITAL SECTION CARD
       ======================================================== */
    .medical-card {
        margin: 12px 0 5px 0;
    }

    .medical-card-header {
        min-height: 42px;
        display: flex;
        align-items: center;
        gap: 9px;
        padding: 0 12px;
        background: #e5edf2;
        border: 1px solid #d1dce3;
        border-left: 4px solid #2f6f95;
        border-radius: 4px;
        color: #245f80;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0;
        box-sizing: border-box;
    }

    .medical-card-icon {
        width: 24px;
        height: 24px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: #d3e4ed;
        color: #2b789d;
        font-size: 14px;
        flex: 0 0 24px;
    }

    /* The real Streamlit widgets are rendered outside raw HTML blocks.
       Give the surrounding page area a subtle hospital-grey appearance
       without creating a second, empty header/body panel. */
    .section-space {
        height: 3px;
    }

    .profile-card {
        background: #ffffff;
        border: 1px solid #d9e2e8;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 10px 0;
    }

    .profile-title {
        color: #23739a;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .profile-description {
        color: #667d8a;
        font-size: 11px;
    }

    /* ========================================================
       STREAMLIT INPUTS
       ======================================================== */
    label[data-testid="stWidgetLabel"] p {
        color: #2d4b5e !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }

    div[data-testid="stNumberInput"],
    div[data-testid="stSelectbox"],
    div[data-testid="stTextInput"],
    div[data-testid="stTextArea"],
    div[data-testid="stMultiSelect"],
    div[data-testid="stDateInput"] {
        margin-bottom: 7px !important;
    }

    /* Khung xám dùng chung cho MỌI loại khung nhập liệu (số, chọn, chữ...),
       kể cả khi browser/Streamlit đang ở dark theme. Selector cố tình viết rộng
       (cả theo data-testid lẫn data-baseweb độc lập) để không bị "trắng lại"
       nếu cấu trúc DOM của một phiên bản Streamlit khác đi đôi chút. */
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] [role="combobox"],
    div[data-testid="stSelectbox"] button[role="combobox"],
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"],
    div[data-testid="stDateInput"] input,
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"],
    div[data-baseweb="input"] {
        min-height: 36px !important;
        background: var(--field-bg) !important;
        background-color: var(--field-bg) !important;
        border: 1px solid var(--field-border) !important;
        border-radius: 4px !important;
        box-shadow: none !important;
        color: var(--field-text) !important;
        -webkit-text-fill-color: var(--field-text) !important;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] input,
    div[data-testid="stSelectbox"] button[role="combobox"] *,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] * {
        background: transparent !important;
        color: var(--field-text) !important;
        -webkit-text-fill-color: var(--field-text) !important;
        fill: #496c80 !important;
        font-size: 12px !important;
    }

    div[data-baseweb="select"] svg {
        color: #52758a !important;
        fill: #52758a !important;
    }

    /* Ô đã bị vô hiệu hoá (ví dụ BMI khi tick "Không biết BMI") vẫn giữ nền xám,
       chỉ làm nhạt chữ đi một chút để phân biệt trực quan là không thể sửa. */
    div[data-testid="stNumberInput"] input:disabled,
    div[data-testid="stTextInput"] input:disabled,
    div[data-testid="stSelectbox"] button[role="combobox"]:disabled {
        background: var(--field-bg) !important;
        background-color: var(--field-bg) !important;
        color: #7c8f9b !important;
        -webkit-text-fill-color: #7c8f9b !important;
        opacity: 1 !important;
    }

    /* Dropdown menu khi mở */
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="menu"] {
        background: #ffffff !important;
        color: #263f50 !important;
    }

    [data-baseweb="popover"] [role="option"] {
        background: #ffffff !important;
        color: #263f50 !important;
        font-size: 12px !important;
    }

    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [aria-selected="true"] {
        background: #e7f1f6 !important;
        color: #1f6287 !important;
    }

    div[data-testid="stNumberInput"] input {
        min-height: 36px !important;
        background: var(--field-bg) !important;
        background-color: var(--field-bg) !important;
        color: var(--field-text) !important;
        border: 1px solid var(--field-border) !important;
        border-radius: 4px 0 0 4px !important;
        font-size: 12px !important;
        box-shadow: none !important;
    }

    /* Nút +/- dùng cùng tông xám với ô nhập, chỉ đậm hơn một chút khi hover
       để vẫn phân biệt được là nút bấm, không tạo cảm giác "hai màu xám khác nhau". */
    div[data-testid="stNumberInput"] button {
        background: var(--field-bg) !important;
        color: #3f5a6a !important;
        border-color: var(--field-border) !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background: var(--field-bg-hover) !important;
    }

    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stTextInput"] input:focus,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #73aec7 !important;
        box-shadow: 0 0 0 1px rgba(47,111,149,.10) !important;
    }

    /* Checkbox sits below BMI */
    div[data-testid="stCheckbox"] {
        margin-top: -1px !important;
        margin-bottom: 4px !important;
    }

    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] label p {
        color: #385569 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }

    div[data-testid="stCheckbox"] svg {
        color: #2f6f95 !important;
    }

    /* ========================================================
       EXPANDER / RAW DATA
       ======================================================== */
    div[data-testid="stExpander"] {
        background: #f5f7f9 !important;
        border: 1px solid #d8e1e7 !important;
        border-radius: 4px !important;
        margin-top: 12px !important;
    }

    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary p {
        color: #216a90 !important;
        font-size: 13px !important;
        font-weight: 700 !important;
    }

    /* ========================================================
       PREDICTION
       ======================================================== */
    .prediction-title {
        color: #174f70;
        font-size: 19px;
        font-weight: 700;
        margin: 20px 0 9px 0;
    }

    .result-card {
        background: #ffffff;
        border: 1px solid #d6e0e6;
        border-radius: 4px;
        padding: 17px 19px;
        margin-top: 10px;
    }

    .result-label {
        color: #687d89;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .04em;
    }

    .result-value {
        color: #17658f;
        font-size: 32px;
        font-weight: 700;
        margin-top: 2px;
    }

    .risk-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 3px;
        font-size: 11px;
        font-weight: 700;
    }

    .risk-low {
        background: #eaf7f2;
        border: 1px solid #c6e7da;
        color: #237a63;
    }

    .risk-medium {
        background: #fff7e3;
        border: 1px solid #eddba9;
        color: #956e20;
    }

    .risk-high {
        background: #fff0f0;
        border: 1px solid #efcccc;
        color: #aa4a4a;
    }

    .recommendation {
        background: #f3f8fb;
        border-left: 3px solid #2c9cc0;
        padding: 11px 13px;
        margin-top: 10px;
        color: #3c5868;
        font-size: 12px;
        line-height: 1.6;
    }

    .recommendation strong {
        color: #1e6286;
    }

    /* Button */
    .stButton > button {
        min-height: 39px;
        background: #2f6f95 !important;
        border: 1px solid #2f6f95 !important;
        border-radius: 4px !important;
        color: white !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    .stButton > button:hover {
        background: #245a7a !important;
        border-color: #245a7a !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #d7e0e6 !important;
        border-radius: 4px !important;
        padding: 9px 12px !important;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        color: #748792 !important;
        font-size: 10px !important;
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: #2f6f95 !important;
        font-size: 18px !important;
    }

    .footer {
        text-align: center;
        color: #718590;
        font-size: 10px;
        padding: 18px 0 3px 0;
    }

    /* Progress */
    div[data-testid="stProgressBar"] > div {
        background: #dce4e9 !important;
        border-radius: 2px !important;
    }

    div[data-testid="stProgressBar"] > div > div {
        background: #2f7fa5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">♡</div>
            <div>
                <div class="sidebar-brand-name">StrokeCare AI</div>
                <div class="sidebar-brand-sub">Clinical Risk Screening • Academic Demo</div>
            </div>
        </div>

        <div class="sidebar-heading">⚙️ Cài đặt đánh giá</div>

        <div class="sidebar-card">
            <div class="sidebar-card-title">📍 Thông tin dự án</div>
            <div class="sidebar-card-text">
                Ứng dụng học máy dự đoán nguy cơ đột quỵ từ dữ liệu lâm sàng.
            </div>
        </div>

        <div class="sidebar-card">
            <div class="sidebar-card-title">🔒 Quyền riêng tư</div>
            <div class="sidebar-card-text">
                Chỉ sử dụng dữ liệu được nhập trên giao diện cho phiên dự đoán hiện tại.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TOP BAR
# ============================================================
st.markdown(
    """
    <div class="topbar">
        <div class="breadcrumb">
            <span class="home">⌂ &nbsp;Trang chủ</span>
            &nbsp; › &nbsp;
            <span>Đánh giá nguy cơ</span>
            &nbsp; › &nbsp;
            <span class="current">Bệnh nhân</span>
        </div>
        <div class="deploy">👤 &nbsp;Admin</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE TITLE
# ============================================================
st.markdown(
    """
    <div class="page-title">
        <span class="page-title-icon">👤</span>
        <span class="page-title-text">Thông tin bệnh nhân</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "⚕️ **Lưu ý quan trọng:** Đây là mô hình minh họa cho đồ án học tập, "
    "**không phải công cụ chẩn đoán y tế**. Không sử dụng kết quả để tự chẩn đoán, "
    "điều trị hoặc đưa ra quyết định sức khỏe thay cho bác sĩ."
)


# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def load_pipeline():
    """Tải pipeline gồm tiền xử lý và mô hình."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()

if pipeline is None:
    st.warning(
        "Chưa tìm thấy mô hình. Ứng dụng vẫn cho phép nhập dữ liệu nhưng "
        "chưa thể thực hiện dự đoán."
    )


# ============================================================
# MAPPING DISPLAY -> ORIGINAL DATASET VALUES
# ============================================================
GENDER_MAP = {
    "Nam": "Male",
    "Nữ": "Female",
    "Khác": "Other",
}

WORK_MAP = {
    "Tư nhân": "Private",
    "Tự kinh doanh": "Self-employed",
    "Cơ quan nhà nước": "Govt_job",
    "Trẻ em": "children",
    "Chưa từng làm việc": "Never_worked",
}

RESIDENCE_MAP = {
    "Thành thị": "Urban",
    "Nông thôn": "Rural",
}

SMOKING_MAP = {
    "Chưa từng hút thuốc": "never smoked",
    "Đã từng hút thuốc": "formerly smoked",
    "Đang hút thuốc": "smokes",
    "Không rõ": "Unknown",
}


# ============================================================
# 1. VITALS & BIOCHEMISTRY
# ============================================================
st.markdown(
    """
    <div class="medical-card">
        <div class="medical-card-header">
            <span class="medical-card-icon">⚗</span>
            <span>Chỉ số sinh hiệu &amp; sinh hóa</span>
        </div>
""",
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2, gap="large")

with c1:
    avg_glucose_level = st.number_input(
        "Glucose trung bình (mg/dL)",
        min_value=0.0,
        max_value=400.0,
        value=100.0,
        step=1.0,
        help=(
            "Glucose máu lúc đói thường được xem là bình thường khi dưới "
            "100 mg/dL. Khoảng tham chiếu có thể thay đổi theo xét nghiệm "
            "và bối cảnh lâm sàng."
        ),
    )

with c2:
    bmi = st.number_input(
        "BMI (kg/m²)",
        min_value=10.0,
        max_value=100.0,
        value=25.0,
        step=0.1,
        help=(
            "BMI 18.5–24.9 thường được phân loại là khoảng bình thường "
            "ở người trưởng thành. BMI chỉ là một chỉ số tham khảo."
        ),
    )

    bmi_unknown = st.checkbox(
        "Không biết BMI",
        help="Nếu không có thông tin BMI, pipeline sẽ nhận giá trị thiếu (NaN) để xử lý.",
    )

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 2. MEDICAL HISTORY
# ============================================================
st.markdown(
    """
    <div class="medical-card">
        <div class="medical-card-header">
            <span class="medical-card-icon">♡</span>
            <span>Tiền sử bệnh lý</span>
        </div>
""",
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2, gap="large")

with c1:
    hypertension = st.selectbox(
        "Tăng huyết áp",
        [0, 1],
        format_func=lambda v: "Có" if v else "Không",
        help="Cho biết bệnh nhân có tiền sử/tình trạng tăng huyết áp theo dữ liệu đầu vào hay không.",
    )

with c2:
    heart_disease = st.selectbox(
        "Bệnh tim",
        [0, 1],
        format_func=lambda v: "Có" if v else "Không",
        help="Cho biết bệnh nhân có tiền sử bệnh tim theo dữ liệu đầu vào hay không.",
    )

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 3. DEMOGRAPHICS & LIFESTYLE
# ============================================================
st.markdown(
    """
    <div class="medical-card">
        <div class="medical-card-header">
            <span class="medical-card-icon">👤</span>
            <span>Nhân khẩu học &amp; lối sống</span>
        </div>
""",
    unsafe_allow_html=True,
)


# Nhân khẩu học gồm tuổi, giới tính, hôn nhân, công việc, nơi sống và hút thuốc.
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    age = st.number_input(
        "Tuổi",
        min_value=0.0,
        max_value=120.0,
        value=50.0,
        step=1.0,
        help="Tuổi của bệnh nhân, tính theo năm.",
    )

with c2:
    gender_display = st.selectbox(
        "Giới tính",
        list(GENDER_MAP.keys()),
        help="Giá trị hiển thị tiếng Việt sẽ được ánh xạ về nhãn gốc của dataset.",
        key="gender_display",
    )

with c3:
    ever_married = st.selectbox(
        "Đã từng kết hôn",
        ["Yes", "No"],
        format_func=lambda v: "Có" if v == "Yes" else "Không",
    )

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    work_type_display = st.selectbox(
        "Loại công việc",
        list(WORK_MAP.keys()),
        help="Danh mục công việc được ánh xạ lại về format gốc trước khi gửi vào pipeline.",
        key="work_type_display",
    )

with c2:
    residence_display = st.selectbox(
        "Khu vực sống",
        list(RESIDENCE_MAP.keys()),
        key="residence_display",
    )

with c3:
    smoking_status_display = st.selectbox(
        "Tình trạng hút thuốc",
        list(SMOKING_MAP.keys()),
        help="Tình trạng hút thuốc được ánh xạ về đúng nhãn gốc của dataset.",
        key="smoking_status_display",
    )

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# BUILD MODEL INPUT
# ============================================================
gender = GENDER_MAP[gender_display]
work_type = WORK_MAP[work_type_display]
residence_type = RESIDENCE_MAP[residence_display]
smoking_status = SMOKING_MAP[smoking_status_display]

input_df = pd.DataFrame(
    [
        {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence_type,
            "avg_glucose_level": avg_glucose_level,
            "bmi": np.nan if bmi_unknown else bmi,
            "smoking_status": smoking_status,
        }
    ]
)


# ============================================================
# RAW DATA
# ============================================================
with st.expander("▣  Xem dữ liệu gốc gửi vào mô hình"):
    st.caption(
        "Bảng dưới đây giữ nguyên tên cột và nhãn dữ liệu gốc để tương thích với pipeline."
    )
    st.dataframe(
        input_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PREDICTION
# ============================================================
st.markdown(
    '<div class="prediction-title"><span class="page-title-icon">●</span> Phân tích nguy cơ</div>',
    unsafe_allow_html=True,
)

button_col, spacer = st.columns([1, 2])

with button_col:
    predict_clicked = st.button(
        "🩺  Dự đoán nguy cơ",
        use_container_width=True,
    )


if predict_clicked:
    if pipeline is None:
        st.error(
            "Chưa có mô hình để dự đoán. Hãy đặt mô hình vào thư mục `models/`."
        )
    else:
        try:
            proba = float(pipeline.predict_proba(input_df)[0, 1])

            # 3 mức hiển thị:
            # thấp < 25%, trung bình 25%–50%, cao >= 50%.
            medium_boundary = THRESHOLD * 0.50

            if proba >= THRESHOLD:
                risk_level = "Cao"
                risk_class = "risk-high"
                icon = "🔴"
                recommendation = (
                    "<strong>Khuyến nghị tham khảo:</strong> Nên trao đổi với bác sĩ "
                    "để được đánh giá các yếu tố nguy cơ tim mạch/thần kinh và xem xét "
                    "theo dõi huyết áp, đường huyết cùng các yếu tố liên quan theo chỉ định."
                )
            elif proba >= medium_boundary:
                risk_level = "Trung bình"
                risk_class = "risk-medium"
                icon = "🟡"
                recommendation = (
                    "<strong>Khuyến nghị tham khảo:</strong> Nên chú ý kiểm soát các "
                    "yếu tố nguy cơ có thể thay đổi như huyết áp, đường huyết, cân nặng "
                    "và hút thuốc; có thể trao đổi với nhân viên y tế nếu có yếu tố bất thường."
                )
            else:
                risk_level = "Thấp"
                risk_class = "risk-low"
                icon = "🟢"
                recommendation = (
                    "<strong>Khuyến nghị tham khảo:</strong> Duy trì lối sống lành mạnh, "
                    "vận động phù hợp, dinh dưỡng cân bằng và kiểm tra sức khỏe định kỳ "
                    "theo nhu cầu cá nhân."
                )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Điểm nguy cơ do mô hình ước tính</div>
                    <div class="result-value">{proba:.1%}</div>
                    <span class="risk-badge {risk_class}">
                        {icon} &nbsp;Mức nguy cơ: {risk_level}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(
                min(max(proba, 0.0), 1.0),
                text=f"Mức điểm mô hình: {proba:.1%}",
            )

            r1, r2, r3 = st.columns(3, gap="medium")
            r1.metric("Nguy cơ thấp", f"< {medium_boundary:.0%}")
            r2.metric(
                "Nguy cơ trung bình",
                f"{medium_boundary:.0%} – {THRESHOLD:.0%}",
            )
            r3.metric("Nguy cơ cao", f"≥ {THRESHOLD:.0%}")

            st.markdown(
                f'<div class="recommendation">💡 {recommendation}</div>',
                unsafe_allow_html=True,
            )

            st.warning(
                "⚠️ **Diễn giải mô hình:** Pipeline của đồ án được huấn luyện với SMOTE "
                "trên dữ liệu mất cân bằng, vì vậy điểm `predict_proba` có thể cao hơn "
                "xác suất thực tế. Hãy xem đây là **điểm số tương đối của mô hình**, "
                "không phải xác suất y khoa đã được hiệu chuẩn."
            )

        except Exception as exc:
            st.error(
                "Không thể thực hiện dự đoán. Hãy kiểm tra pipeline và cấu trúc dữ liệu đầu vào."
            )
            with st.expander("Chi tiết kỹ thuật"):
                st.exception(exc)


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        StrokeCare AI • Academic Demonstration<br>
        Kết quả được tạo bởi mô hình học máy và chỉ nhằm mục đích tham khảo trong đồ án.
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "⚕️ **Disclaimer y tế:** Ứng dụng này không thay thế bác sĩ hoặc quy trình chẩn đoán "
    "y khoa. Không tự điều trị hoặc thay đổi thuốc dựa trên kết quả dự đoán. "
    "Nếu xuất hiện các triệu chứng nghi ngờ đột quỵ cấp tính, cần tìm kiếm hỗ trợ "
    "y tế khẩn cấp thay vì dựa vào ứng dụng."
)
