# Báo cáo thay đổi: xử lý glucose và BMI theo ngưỡng y khoa

Nhánh: `clinical-thresholds` (commit `de96429`). Ngày: 07/11/2026.

## 1. Mục tiêu

Đưa các ngưỡng y khoa của glucose và BMI vào phần xử lý dữ liệu để kiểm tra chất lượng dữ liệu và mô tả kết quả, mà **không làm đổi mô hình đã lưu** (`models/stroke_pipeline.pkl`, ngưỡng quyết định 0,74).

## 2. Ngưỡng y khoa sử dụng

Các ngưỡng chỉ mang tính tham chiếu, không phải chẩn đoán. Dataset không ghi loại xét nghiệm glucose hay thời điểm lấy mẫu.

**Glucose (mg/dL)**

| Mức | Ý nghĩa | Nguồn |
|---|---|---|
| Dưới 54 | Hạ đường huyết nặng | CDC |
| Dưới 70 | Hạ đường huyết | CDC |
| Dưới 100 (lúc đói) | Bình thường | ADA |
| 100–125 (lúc đói) | Tiền đái tháo đường | ADA |
| 126 trở lên (lúc đói) | Mức đái tháo đường | ADA |

**BMI (kg/m²)**

| Đối tượng | Phân loại | Nguồn |
|---|---|---|
| Từ 20 tuổi | Thiếu cân dưới 18,5; bình thường 18,5 đến dưới 25; thừa cân 25 đến dưới 30; béo phì từ 30 | CDC |
| Dưới 20 tuổi | Dùng percentile theo tuổi và giới, không áp ngưỡng người lớn | CDC |

Không có "BMI tối thiểu" hay "glucose tối thiểu" cố định theo y khoa. Các mốc thấp nhất có ý nghĩa là glucose dưới 70 và BMI dưới 18,5.

## 3. Thay đổi trong code

**`src/preprocessing.py`** (chỉ thêm mới, không đổi hàm hay tham số hiện có, đúng hợp đồng nhóm):

- Hằng số ngưỡng y khoa cho glucose và BMI, kèm ghi chú nguồn.
- `glucose_category()`: chia glucose thành 5 nhóm, giá trị thiếu ghi "Thiếu".
- `bmi_category()`: chia nhóm BMI theo ngưỡng người lớn, có xét tuổi. Người dưới 20 tuổi được ghi "cần percentile".
- `validate_clinical_ranges()`: đổi glucose và BMI "không thể xảy ra" thành NaN. Giới hạn là 20–800 cho glucose và 8–100 cho BMI. Đây là lựa chọn kỹ thuật của nhóm, không phải khuyến cáo y khoa. Hàm không sửa dữ liệu đầu vào.

**`scripts/clinical_data_report.py`** (mới): sinh `results/clinical_data_report.md` gồm kiểm tra khoảng hợp lý, tỷ lệ đột quỵ theo nhóm glucose và BMI, và hồ sơ nhóm thiếu BMI.

**`results/clinical_data_report.md`** (mới): kết quả chạy script trên dữ liệu gốc.

## 4. Kiểm tra

- Không có giá trị glucose hay BMI nào bị đổi thành NaN trên dữ liệu gốc.
- Dữ liệu sau làm sạch giống dữ liệu gốc, và ma trận đặc trưng sau tiền xử lý giống hệt nhau.
- Đã thử riêng các giá trị biên của hai hàm phân nhóm (54, 70, 100, 126; 18,5, 25, 30; người dưới 20 tuổi; giá trị thiếu) và hàm làm sạch (glucose 0 và 900, BMI 5 và 120).
- Chưa kiểm tra bằng cách nạp `stroke_pipeline.pkl` vì môi trường chạy không cài được `imbalanced-learn`. Tuy vậy pipeline không bị sửa nên hành vi của mô hình không thay đổi.

## 5. Phát hiện chính từ dữ liệu

- Glucose: nhóm từ 126 trở lên có tỷ lệ đột quỵ 10,2%, so với khoảng 3,6% ở các nhóm thấp hơn. Dữ liệu nhỏ nhất là 55,12 nên không có ca hạ đường huyết nặng; 754 ca dưới 70 có tỷ lệ đột quỵ 3,6%, không cao hơn.
- BMI: tỷ lệ đột quỵ không đơn điệu theo nhóm (thiếu cân 2,8%, bình thường 4,3%, thừa cân 5,8%, béo phì 5,3%). Nhóm thiếu cân chỉ có 1 ca đột quỵ trên 36 ca nên không đủ để kết luận.
- Thiếu BMI: 201 ca (3,9%), không ngẫu nhiên. Tỷ lệ thiếu là 7,2% ở nhóm 60–80 tuổi, 16,1% ở người từng bị đột quỵ, 9,4% ở người tăng huyết áp, 12,0% ở người có bệnh tim. Tỷ lệ đột quỵ ở nhóm thiếu BMI là 19,9%. Dataset không ghi lý do thiếu.
- Các số liệu trên là mối liên quan thống kê, không phải quan hệ nhân quả. Khi viết báo cáo nên nói "tình trạng thiếu BMI mang thông tin dự báo", không nói "BMI là yếu tố quan trọng thứ hai" (xem thêm `docs/chuong_trien_khai.md`).

## 6. Chưa thực hiện (cần huấn luyện lại mô hình)

- Điền BMI thiếu theo nhóm tuổi thay vì median chung 28,1. Hiện một bé 2 tuổi thiếu BMI vẫn được điền 28,1, điều này không hợp lý về lâm sàng.
- Đưa nhóm glucose và BMI thành đặc trưng cho mô hình.
- Hiển thị nhóm glucose và BMI cạnh ô nhập trong `app.py` (không đổi mô hình, chỉ sửa giao diện).

Ba việc đầu đổi mô hình nên phải huấn luyện lại, chọn lại ngưỡng quyết định và cập nhật lại số liệu trong báo cáo.

## 7. Nguồn

- [CDC – Low Blood Sugar (Hypoglycemia)](https://www.cdc.gov/diabetes/about/low-blood-sugar-hypoglycemia.html)
- [ADA – Diabetes Diagnosis & Tests](https://diabetes.org/about-diabetes/diagnosis)
- [CDC – Adult BMI Categories](https://www.cdc.gov/bmi/adult-calculator/bmi-categories.html)
- [CDC – Child and Teen BMI Calculator](https://www.cdc.gov/bmi/child-teen-calculator/index.html)
