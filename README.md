# Dự đoán nguy cơ đột quỵ (Stroke Prediction)

Đồ án môn Học máy: xây dựng mô hình phân lớp dự đoán khả năng bị đột quỵ (**không bệnh / có bệnh**) từ dữ liệu y tế cơ bản, kèm ứng dụng minh họa bằng Streamlit.

> **Lưu ý:** đây là đồ án học tập. Mô hình **không phải công cụ chẩn đoán y tế** và không được dùng để đưa ra quyết định về sức khỏe.

## Bộ dữ liệu

- Nguồn: [Stroke Prediction Dataset (fedesoriano) – Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- 5.110 dòng, 12 cột. Biến mục tiêu `stroke` (0/1); khoảng 4,9% (249 ca) là `stroke = 1`, tức dữ liệu **mất cân bằng nghiêm trọng**.
- File nằm tại `data/healthcare-dataset-stroke-data.csv`.

| Cột | Ý nghĩa |
|---|---|
| `id` | Mã bệnh nhân (bị loại bỏ khi huấn luyện) |
| `gender`, `age` | Giới tính, tuổi |
| `hypertension`, `heart_disease` | Tăng huyết áp, bệnh tim (0/1) |
| `ever_married`, `work_type`, `Residence_type` | Hôn nhân, loại công việc, khu vực sống |
| `avg_glucose_level`, `bmi` | Glucose trung bình, chỉ số BMI (`bmi` có giá trị thiếu) |
| `smoking_status` | Tình trạng hút thuốc |
| `stroke` | **Biến mục tiêu** |

## Phương pháp

```
EDA → Tiền xử lý → SMOTE + Stratified CV → 6 mô hình → Tuning → Đánh giá → Lưu pipeline → Streamlit
```

- **Chia dữ liệu:** train/test có `stratify`, `random_state=42`. Tập test chỉ dùng một lần ở cuối.
- **Tiền xử lý:** điền `bmi` thiếu bằng median, chuẩn hóa biến số (`StandardScaler`), one-hot các biến phân loại. Tất cả nằm trong `Pipeline` để không rò rỉ dữ liệu.
- **Mất cân bằng lớp:** SMOTE đặt **bên trong** `imblearn.Pipeline`, chỉ áp dụng trên phần train của từng fold.
- **Mô hình:** Logistic Regression, Decision Tree, KNN, SVM, Random Forest, XGBoost.
- **Tuning:** Random Forest và XGBoost bằng `RandomizedSearchCV`.
- **Chỉ số đánh giá:** Recall, Precision, F1, ROC-AUC, PR-AUC; đồng thời so sánh Recall/Precision/F1 qua nhiều threshold bằng bảng và biểu đồ. Không chọn mô hình theo accuracy vì mô hình luôn đoán "không bệnh" đã đạt khoảng 95% accuracy.
- **Giải thích mô hình:** feature importance và permutation importance.

## Kết quả

Các số dưới đây là trung bình 5 fold trên training set, chạy lại bằng preprocessing hiện có. PR-AUC được ghi là `average_precision`.

| Mô hình | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.779 | 0.140 | 0.237 | 0.848 | 0.226 |
| Decision Tree | 0.221 | 0.143 | 0.173 | 0.576 | 0.070 |
| KNN | 0.412 | 0.127 | 0.194 | 0.684 | 0.100 |
| SVM | 0.483 | 0.122 | 0.194 | 0.787 | 0.147 |
| Random Forest | 0.121 | 0.202 | 0.150 | 0.803 | 0.144 |
| XGBoost | 0.110 | 0.187 | 0.138 | 0.801 | 0.151 |

Tuning trên training set đạt Average Precision 0.177 với Random Forest và 0.214 với XGBoost. Nhóm đã chọn Logistic Regression theo AP CV. Threshold 0.74 được chọn bằng F1 trên dự đoán OOF của train. Trên holdout một lần, Recall = 0.640, Precision = 0.219, F1 = 0.327, ROC-AUC = 0.845 và AP = 0.275. Pipeline cuối nằm tại `models/stroke_pipeline.pkl`; app đọc threshold từ `results/final_model_metadata.json`. File test cũ dùng preprocessing trước khi tích hợp bản sửa được lưu riêng tại `results/legacy_test_results_before_preprocessing_fix.csv` và không dùng trong kết luận.

## Cài đặt

Yêu cầu: **Python 3.11 trở lên**, Git.

```bash
git clone https://github.com/mrlgk/StrokePrediction.git
cd StrokePrediction
python -m venv .venv
```

Bật môi trường ảo:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Cài thư viện và kiểm tra:

```bash
pip install -r requirements.txt
python -c "import sklearn, imblearn, xgboost, streamlit; print('OK')"
```

Nếu dùng notebook, chạy một lần trên mỗi máy để tự xóa output khi commit:

```bash
nbstripout --install
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`. Ứng dụng cần file `models/stroke_pipeline.pkl` để dự đoán; khi chưa có file này, app chạy ở chế độ khung (chỉ hiển thị dữ liệu nhập).

## Cấu trúc thư mục

```
StrokePrediction/
├── data/                 Dữ liệu gốc (healthcare-dataset-stroke-data.csv)
├── notebooks/            Notebook EDA, huấn luyện mô hình, feature importance
├── src/
│   ├── preprocessing.py  load_data, split_data, get_preprocessor
│   ├── models.py         Danh sách 6 mô hình
│   ├── tuning.py         Tuning Random Forest, XGBoost
│   └── evaluation.py     Chỉ số, confusion matrix, ROC/PR và threshold curves
├── models/               Pipeline đã huấn luyện (stroke_pipeline.pkl)
├── results/              Bảng kết quả CV và biểu đồ
├── app.py                Ứng dụng Streamlit
├── requirements.txt
├── HUONG_DAN_NHOM.md     Hướng dẫn làm việc nhóm qua GitHub
└── README.md
```

## Thành viên và phân công

| Vai trò | Thành viên | Phụ trách |
|---|---|---|
| A – Dữ liệu | *(điền tên)* | EDA, trực quan hóa, tiền xử lý (`preprocessing.py`) |
| B – Mô hình | *(điền tên)* | Huấn luyện, cross-validation, tuning, đánh giá trên tập test |
| C – Ứng dụng & tích hợp | *(điền tên)* | `evaluation.py`, lưu pipeline, Streamlit, README, ráp báo cáo |

Quy trình làm việc chung (nhánh, Pull Request, quy tắc Git) xem trong [`HUONG_DAN_NHOM.md`](HUONG_DAN_NHOM.md).

## Hạn chế

- Dữ liệu chỉ có 249 ca dương tính và có giá trị thiếu ở `bmi`, nên kết quả có thể dao động giữa các lần chia dữ liệu.
- SMOTE làm xác suất dự đoán thường cao hơn thực tế; xác suất trong app nên đọc như điểm số tương đối, không phải xác suất y khoa.
- Chưa kiểm định trên dữ liệu bên ngoài.

## Ghi chú

Dữ liệu và mã nguồn chỉ dùng cho mục đích học tập trong khuôn khổ môn Học máy.
