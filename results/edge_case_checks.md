# Kiểm tra ca biên của pipeline cuối

Chạy sau khi lưu và tải lại `models/stroke_pipeline.pkl`. Mỗi ca được gửi dưới dạng một dòng có đủ mười cột gốc; điểm dưới đây là đầu ra của Logistic Regression cuối. Tất cả ca trả về giá trị hữu hạn trong [0, 1] và không phát sinh lỗi.

| Ca đầu vào | Điểm lớp stroke |
|---|---:|
| Giá trị mặc định của form | 22,8% |
| Tuổi 1, nhóm công việc trẻ em | 0,8% |
| Tuổi 100 | 96,7% |
| BMI thiếu (`NaN`) | 59,2% |
| BMI 10 | 20,3% |
| BMI 100 | 38,1% |
| Glucose 400 | 48,8% |
| Glucose 0 | 16,7% |
| Giới tính `Other` | 28,8% |
| Hồ sơ nhiều yếu tố nguy cơ được đặt cùng nhau | 96,8% |
| Hồ sơ tuổi trẻ, chỉ số thấp | 1,7% |

Pipeline tạo 22 đặc trưng sau preprocessing. Các điểm chỉ xác nhận luồng nhập và dự đoán xử lý được các ca đã nêu. Chúng không phải xác suất y khoa đã hiệu chuẩn và không được dùng cho quyết định sức khỏe.

Kết quả máy có độ chính xác đầy đủ hơn được lưu tại `results/edge_case_checks_final.csv`.
