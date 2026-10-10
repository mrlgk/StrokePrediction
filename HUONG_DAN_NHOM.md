# Hướng dẫn làm việc nhóm

Tài liệu này thống nhất cách chia sẻ mã nguồn của đồ án Stroke Prediction. Mục tiêu là giữ `main` chạy được và giúp mỗi thành viên làm việc trên nhánh riêng.

## Thành viên và phạm vi

| Vai trò | Nhánh | Phạm vi chính |
|---|---|---|
| A – Dữ liệu | `feat/eda-preprocessing` | `notebooks/01_EDA.ipynb`, `src/preprocessing.py` |
| B – Mô hình | `feat/modeling` | `notebooks/03_Modeling.ipynb`, `src/models.py`, `src/tuning.py`, kết quả CV và tuning |
| C – Ứng dụng | `feat/app` | `app.py`, `src/evaluation.py`, README, tích hợp pipeline |

Các thành viên thống nhất chữ ký hàm dùng chung trước khi thay đổi. Không đổi tên hoặc kiểu trả về của `load_data()`, `split_data()` và `get_preprocessor()` nếu chưa báo cho cả nhóm.

## Môi trường

Yêu cầu Python 3.11 trở lên. Cả nhóm nên dùng cùng một phiên bản Python và cài các thư viện đã ghim trong `requirements.txt`.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra import chính:

```powershell
python -c "import pandas, sklearn, imblearn, xgboost, streamlit; print('OK')"
```

## Quy trình Git

1. Cập nhật `main` trước khi bắt đầu một phần việc:

   ```powershell
   git switch main
   git pull origin main
   git switch -c feat/ten-cong-viec
   ```

2. Commit thay đổi nhỏ, tên commit nêu rõ nội dung.
3. Đẩy nhánh của mình và mở Pull Request vào `main`.
4. Một thành viên khác đọc diff, kiểm tra thay đổi và xác nhận trước khi merge.
5. Sau khi merge, cập nhật lại nhánh làm việc từ `main`.

Không push trực tiếp lên `main`. Không sửa cùng lúc một notebook với người khác vì notebook thường khó giải quyết xung đột.

## Quy ước modeling và đánh giá

- Chia train/test có `stratify` trước khi fit preprocessing hoặc SMOTE.
- Đặt SMOTE trong `imblearn.Pipeline` để chỉ resample phần train của từng fold.
- Dùng `StratifiedKFold` và chọn mô hình bằng các metric phù hợp với dữ liệu mất cân bằng; không dùng accuracy làm tiêu chí duy nhất.
- Chọn threshold từ dữ liệu train bằng dự đoán out-of-fold. Giữ nguyên threshold khi đánh giá holdout.
- Lưu các bảng kết quả vào `results/` và ghi rõ mô hình, metric cùng threshold.

## Pipeline và ứng dụng

App đọc pipeline cuối từ `models/stroke_pipeline.pkl`. Pipeline tạm được tạo bởi `scripts/make_temp_pipeline.py` lưu riêng thành `models/stroke_pipeline_temp.pkl`, chỉ phục vụ kiểm tra app và ca biên; file tạm không được commit.

Chạy ứng dụng từ thư mục gốc repo:

```powershell
streamlit run app.py
```

Nếu chưa có pipeline cuối, app vẫn hiển thị form nhưng không thể thực hiện dự đoán.

## Trước khi bàn giao

- Tạo clone mới, cài thư viện từ `requirements.txt`, chạy notebook và app.
- Xác nhận các CSV và biểu đồ trong `results/` khớp với notebook.
- Xóa output notebook không cần thiết trước khi commit; không commit `.venv/`, secrets hoặc pipeline tạm.
- Ghi hạn chế của mô hình: dữ liệu mất cân bằng, SMOTE có thể làm xác suất lệch và mô hình chưa được kiểm định ngoài.
